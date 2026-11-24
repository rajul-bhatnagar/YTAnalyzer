"""Phase 3 stage definitions (ANALYZING): signal events -> analysis frames -> AI windows -> fused highlights.

Reuses the Phase 2 stages for audio extraction, activity signals and (optional) transcription. Each stage is
fingerprinted on the inputs that determine it, e.g. changing the Shorts length only re-runs 'highlights'; changing
the AI model re-runs 'ai_windows' (window by window, with a cache) but never frame extraction.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from . import audio_analysis, event_detection, frame_sampler, gameplay_analysis, highlight_scoring, longform
from . import activity_signals as sig
from . import ffmpeg_tools as fx
from .database import Database, to_json, utcnow
from .job_manager import StageContext, StageResult, StageSpec
from .manifest import manifest_entry
from .paths import StudioPaths

PHASE2_STAGES = ["extract_audio", "voiceover", "signals", "transcribe", "subtitles"]
ANALYSIS_STAGES = ["signal_events", "frames", "ai_windows", "highlights"]
TARGETS = {"analysis": PHASE2_STAGES + ANALYSIS_STAGES, "candidates": ["highlights"]}
TARGET_LABELS = {"analysis": "Analyse gameplay & find highlights", "candidates": "Regenerate highlight candidates"}
STAGE_LABELS = {
    "signal_events": "Measure audio/visual evidence (transients, scene changes, voice emphasis)",
    "frames": "Extract analysis frames",
    "ai_windows": "AI analysis, window by window",
    "highlights": "Fuse evidence, score and build Shorts candidates",
}
_fp = longform._fp
_read_json = longform._read_json
_write_json = longform._write_json


def speech_runs_from_mask(grid: np.ndarray, speech: np.ndarray | None, hop: float) -> list[tuple[float, float]]:
    if speech is None or len(speech) == 0:
        return []
    d = np.diff(np.concatenate([[0], speech.astype(np.int8), [0]]))
    return [(float(grid[a]), float(grid[b - 1]) + hop) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def words_video_time(files: longform.ProjectFiles) -> list[dict[str, Any]]:
    data = _read_json(files.words_json, {})
    return data.get("words", []) if isinstance(data, dict) else []


def transcript_segments_video_time(files: longform.ProjectFiles) -> list[dict[str, Any]]:
    """Segments with the user's edits and the voice offset applied (built from words.json)."""
    segs: dict[int, dict[str, Any]] = {}
    for w in words_video_time(files):
        s = segs.setdefault(w.get("segment", 0), {"start": w["start"], "end": w["end"], "text": ""})
        s["end"] = max(s["end"], w["end"])
        s["text"] += w["word"]
    return [{**s, "text": s["text"].strip()} for _, s in sorted(segs.items())]


def make_video_chunk(source: Path, out_dir: Path, start: float, end: float) -> Path:
    """Low-resolution proxy of one window for optional Gemini video analysis (never the full-quality source)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"chunk_{int(start * 1000):09d}.mp4"
    fx.ffmpeg(["-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(source), "-map", "0:v:0",
               "-map", "0:a:0?", "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
               "-c:a", "aac", "-b:a", "64k", "-ac", "1", str(out)], timeout=900)
    return out


def build_stages(db: Database, paths: StudioPaths, project: dict[str, Any], settings: dict[str, Any],
                 target: str) -> tuple[longform.LongformContext, list[StageSpec]]:
    names = TARGETS[target]
    p2 = [n for n in names if n in PHASE2_STAGES]
    lc, specs = longform.build_stages(db, paths, project, settings, "audio", stage_names=p2) if p2 else \
        (longform.LongformContext(db, paths, project, settings), [])
    f = lc.files
    a_cfg, h_cfg, ai_cfg = settings["analysis"], settings["highlights"], settings["ai"]
    src_id = f"{project['source_path']}|{project['source_size']}|{project['source_quick_fp']}"
    context = {"game": project["game_name"], "style": project["content_style"], "topic": project["topic_note"],
               "language": project["language"]}

    # --- signal_events ------------------------------------------------------------------------
    def se_inputs(ctx):
        game16 = lc.activity_audio()
        voice16, off = lc.voice_16k()
        return [_fp(f.signals), _fp(game16) if game16 else "nogame", _fp(voice16) if voice16 else "novoice",
                _fp(f.words_json)], {"cfg": {k: a_cfg[k] for k in ("scene_change_k", "scene_change_min",
                                                                  "transient_min_db", "voice_emphasis_min_db",
                                                                  "event_merge_gap_sec")}, "off": off}

    def se_exec(ctx: StageContext) -> StageResult:
        if not f.signals.exists():
            raise ValueError("Activity signals are missing. Run the analysis from the start.")
        data = sig.load_signals(f.signals)
        hop = float(data["meta"].get("hop_sec", 0.5))
        game16 = lc.activity_audio()
        voice16, off = lc.voice_16k()
        speech = speech_runs_from_mask(data["grid"], data["speech"], hop)
        ctx.progress(0.1, "Measuring game-audio transients and voice emphasis")
        audio = audio_analysis.analyse(game16, voice16, off, speech, lc.duration, a_cfg, ctx.cancel_check)
        scenes = frame_sampler.scene_changes(data["grid"], data["motion"], float(a_cfg["scene_change_k"]),
                                             float(a_cfg["scene_change_min"]))
        bursts = frame_sampler.motion_bursts(data["grid"], data["motion"], hop)
        events = event_detection.detect(audio, scenes, bursts, words_video_time(f), lc.duration,
                                        float(a_cfg["event_merge_gap_sec"]))
        _write_json(f.signal_events, {"audio": audio, "scenes": scenes, "motion_bursts": bursts, "events": events,
                                      "duration": lc.duration})
        warnings = list(audio["notes"])
        if not events:
            warnings.append("No measurable audio/visual events were found; AI analysis (if enabled) still runs.")
        return StageResult(manifest=[manifest_entry(f.signal_events, paths.root, "signal_events", {"ok": True})],
                           warnings=warnings, data={"events": len(events)})

    # --- frames -------------------------------------------------------------------------------
    frame_keys = ("window_sec", "frame_interval_sec", "frame_width", "max_frames_per_window", "extra_frames_per_event")

    def fr_inputs(ctx):
        return [src_id, _fp(f.signal_events)], {k: a_cfg[k] for k in frame_keys}

    def fr_exec(ctx: StageContext) -> StageResult:
        se = _read_json(f.signal_events)
        windows = gameplay_analysis.build_windows(lc.duration, float(a_cfg["window_sec"]))
        peaks = [e["peak"] for e in se["events"]]
        frames: list[dict[str, Any]] = []
        for n, w in enumerate(windows):
            ctx.progress(n / len(windows), f"Extracting frames for window {n + 1} of {len(windows)}")
            times = frame_sampler.choose_timestamps(
                w.start, w.end, float(a_cfg["frame_interval_sec"]), [p for p in peaks if w.start <= p < w.end],
                int(a_cfg["extra_frames_per_event"]), int(a_cfg["max_frames_per_window"]), lc.duration)
            frames += [fi.to_dict() for fi in frame_sampler.sample_window(
                lc.source, f.frames_dir, w.idx, times, int(a_cfg["frame_width"]), paths.to_stored, ctx.cancel_check)]
        keep = {paths.from_stored(fr["path"]).name for fr in frames}
        for old in f.frames_dir.glob("*.jpg"):  # frames from earlier settings
            if old.name not in keep:
                old.unlink(missing_ok=True)
        _write_json(f.frames_index, {"windows": [w.__dict__ for w in windows], "frames": frames})
        warnings = []
        if not frames:
            raise ValueError("No frames could be extracted from the source video.")
        bad = sum(1 for fr in frames if fr["issues"])
        if bad:
            warnings.append(f"{bad} of {len(frames)} frames are nearly black or blurry (loading screens, fades or "
                            "motion blur); they lower the technical-quality factor.")
        return StageResult(manifest=[manifest_entry(f.frames_index, paths.root, "frames", {"ok": True})],
                           warnings=warnings, data={"frames": len(frames), "windows": len(windows)})

    # --- ai_windows ---------------------------------------------------------------------------
    def ai_inputs(ctx):
        plan_desc = ctx.data.get("plan_descriptor", {"provider": "none"})
        return [_fp(f.frames_index), _fp(f.signal_events), _fp(f.words_json)], {
            "plan": plan_desc, "context": context, "prompt": gameplay_analysis.PROMPT_VERSION,
            "cfg": {k: ai_cfg[k] for k in ("max_frames_per_request", "temperature", "gemini_video_max_windows")}}

    def ai_exec(ctx: StageContext) -> StageResult:
        plan: gameplay_analysis.ProviderPlan = ctx.data["plan"]
        fi = _read_json(f.frames_index)
        se = _read_json(f.signal_events)
        segs = transcript_segments_video_time(f)
        windows = [gameplay_analysis.Window(**w) for w in fi["windows"]]
        evidence = []
        for w in windows:
            evidence.append(gameplay_analysis.WindowEvidence(
                w, [fr for fr in fi["frames"] if fr["window"] == w.idx],
                [e for e in se["events"] if w.start <= e["peak"] < w.end],
                [s for s in segs if s["end"] > w.start and s["start"] < w.end], context))
        video_windows: set[int] = set()
        if plan.current is not None and plan.current.mode == "video":
            ranked = sorted(windows, key=lambda w: -event_detection.window_activity(se["events"], w.start, w.end))
            video_windows = {w.idx for w in ranked[: int(ai_cfg["gemini_video_max_windows"])]}
        budget = gameplay_analysis.Budget(int(ai_cfg["max_requests_per_job"]), float(ai_cfg["min_request_interval_sec"])
                                          if plan.current and plan.current.provider.cloud else 0.0,
                                          ctx.data.get("sleep") or time.sleep)
        try:
            report = gameplay_analysis.analyse_windows(
                evidence, plan, f.windows_dir, lc.duration, paths.from_stored, ai_cfg, budget,
                lambda x, note: ctx.progress(0.02 + 0.96 * x, note), ctx.cancel_check,
                ctx.data.get("pause_check", lambda: False), video_windows,
                (lambda w: make_video_chunk(lc.source, f.video_chunks_dir, w.start, w.end)) if video_windows else None,
                ctx.data.get("sleep") or time.sleep)
        finally:
            for chunk in f.video_chunks_dir.glob("*.mp4") if f.video_chunks_dir.exists() else []:
                chunk.unlink(missing_ok=True)
        report["plan_notes"] = ctx.data.get("plan_notes", [])
        report["started_with"] = ctx.data.get("plan_descriptor")
        cleanup = []
        for slot in (plan.primary, plan.fallback):
            cleanup += getattr(slot.provider, "cleanup_warnings", []) if slot else []
        report["cleanup_warnings"] = cleanup
        _write_json(f.ai_report, report)
        statuses = [w["status"] for w in report["windows"]]
        warnings = [*report["plan_notes"], *report["plan_events"], *cleanup]
        if statuses.count("ok") == 0:
            warnings.append("No window was analysed by AI. Highlights are ranked from measured signals only and need "
                            "your review.")
        for w in report["windows"]:
            for issue in w["issues"]:
                warnings.append(f"Window {w['window'] + 1}: {issue}")
        if any(w["mode"] == "text" for w in report["windows"] if w["status"] == "ok"):
            warnings.append("The AI model used is text-only: it read the transcript and measurements but did not see "
                            "the video.")
        return StageResult(manifest=[manifest_entry(f.ai_report, paths.root, "ai_windows", {"ok": True})],
                           warnings=warnings[:40], data={"statuses": statuses, "effective": report["effective"]})

    # --- highlights ---------------------------------------------------------------------------
    def hl_inputs(ctx):
        return [_fp(f.ai_report), _fp(f.signal_events), _fp(f.frames_index), _fp(f.signals)], {
            "cfg": h_cfg, "style": project["content_style"], "count": ctx.data.get("candidate_count")}

    def hl_exec(ctx: StageContext) -> StageResult:
        for need in (f.ai_report, f.signal_events, f.frames_index, f.signals):
            if not need.exists():
                raise ValueError("Run the full analysis first; regenerating candidates needs its results.")
        report = _read_json(f.ai_report)
        se = _read_json(f.signal_events)
        fi = _read_json(f.frames_index)
        results = gameplay_analysis.load_window_results(f.windows_dir, report)
        fused = highlight_scoring.fuse(se["events"], results)
        _write_json(f.fused_events, {"events": fused})
        kept = [dict(r) for r in db.query(
            "SELECT * FROM highlights WHERE project_id = ? AND (status != 'suggested' OR user_start_sec IS NOT NULL "
            "OR user_notes != '')", (project["id"],))]
        blocked = [(r["user_start_sec"] if r["user_start_sec"] is not None else r["start_sec"],
                    r["user_end_sec"] if r["user_end_sec"] is not None else r["end_sec"])
                   for r in kept if r["status"] != "rejected"]
        cands = highlight_scoring.build_candidates(
            fused, lc.duration, project["content_style"], h_cfg, sig.load_signals(f.signals), fi["frames"],
            ctx.data.get("candidate_count"), blocked)
        errors = validate_candidates(cands, lc.duration, h_cfg)
        if errors:
            raise ValueError("Candidate validation failed: " + "; ".join(errors[:5]))
        analysis_id = uuid.uuid4().hex[:10]
        persist(db, project["id"], analysis_id, fused, cands, ctx.job_id)
        _write_json(f.highlights_json, {"analysis_id": analysis_id, "created_at": utcnow(), "candidates": cands,
                                        "kept_from_earlier": len(kept)})
        warnings = []
        if not cands:
            warnings.append("No highlight candidates were found. Try a different content style, lower thresholds, "
                            "or an AI provider that can see frames.")
        if all(c["evidence"]["ai"] is None for c in cands) and cands:
            warnings.append("All candidates come from measurements only (no AI description). Watch each one.")
        if kept:
            warnings.append(f"{len(kept)} highlight(s) you approved, rejected or edited were kept unchanged.")
        return StageResult(manifest=[manifest_entry(f.fused_events, paths.root, "highlights", {"ok": True}),
                                     manifest_entry(f.highlights_json, paths.root, "highlights", {"ok": True})],
                           warnings=warnings, data={"candidates": len(cands), "analysis_id": analysis_id})

    all_specs = {
        "signal_events": StageSpec("signal_events", 1, se_exec, se_inputs),
        "frames": StageSpec("frames", 1, fr_exec, fr_inputs),
        "ai_windows": StageSpec("ai_windows", 1, ai_exec, ai_inputs),
        "highlights": StageSpec("highlights", 1, hl_exec, hl_inputs),
    }
    return lc, specs + [all_specs[n] for n in names if n in all_specs]


def validate_candidates(cands: list[dict[str, Any]], duration: float, cfg: dict[str, Any]) -> list[str]:
    errors = []
    lo, hi = float(cfg["min_sec"]), float(cfg["max_sec"])
    for c in cands:
        s, e = c["start_sec"], c["end_sec"]
        if not (0 <= s < e <= duration + 0.01):
            errors.append(f"#{c['rank']}: times {s}-{e} outside 0-{duration:.2f}")
        if e - s > hi + 0.01 or (duration >= lo and e - s < lo - 0.01):
            errors.append(f"#{c['rank']}: length {e - s:.1f}s outside {lo:.0f}-{hi:.0f}s")
        if not (s <= c["event_sec"] <= e):
            errors.append(f"#{c['rank']}: event time outside the clip")
        if not 0 <= c["score"] <= 100:
            errors.append(f"#{c['rank']}: score {c['score']} out of range")
        ev = c["evidence"]
        if not (ev["visual"] or ev["audio"] or ev["transcript"] or ev["signals"]):
            errors.append(f"#{c['rank']}: no evidence")
    return errors


def persist(db: Database, project_id: str, analysis_id: str, fused: list[dict[str, Any]],
            cands: list[dict[str, Any]], job_id: str) -> None:
    """Replace derived events and untouched suggestions in one transaction; user-touched highlights stay."""
    now = utcnow()
    with db.transaction() as conn:
        conn.execute("DELETE FROM events WHERE project_id = ?", (project_id,))
        for e in fused:
            modal = sorted({m for s in e["signals"] for m in s["modalities"]} | ({"ai"} if e["ai"] else set()))
            conf = (e["ai"] or {}).get("confidence") or {"signals": "measured"}
            conn.execute(
                "INSERT INTO events(project_id, job_id, start_sec, end_sec, event_type, description, evidence, "
                "confidence, source_modality, uncertainty, created_at, analysis_id, event_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (project_id, job_id, e["start"], e["end"], e["category"], e["description"],
                 to_json({"ai": e["ai"], "signals": [s["key"] for s in e["signals"]]}), to_json(conf),
                 ",".join(modal), e["uncertainty"], now, analysis_id, e["key"]))
        conn.execute("DELETE FROM highlights WHERE project_id = ? AND status = 'suggested' AND user_start_sec IS NULL "
                     "AND user_end_sec IS NULL AND user_notes = ''", (project_id,))
        for c in cands:
            conn.execute(
                "INSERT INTO highlights(project_id, start_sec, end_sec, event_sec, score, score_breakdown, rationale, "
                "evidence, confidence, hook_caption, suggested_title, status, created_at, updated_at, analysis_id, "
                "rank, score_band, handles, event_keys, uncertainty, category) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'suggested', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (project_id, c["start_sec"], c["end_sec"], c["event_sec"], c["score"], to_json(c["score_breakdown"]),
                 c["reason"], to_json(c["evidence"]), c["confidence"], c["hook_caption"], c["suggested_title"], now, now,
                 analysis_id, c["rank"], c["score_band"], to_json(c["handles"]), to_json(c["event_keys"]),
                 c["uncertainty"], c["category"]))


def stage_label(name: str) -> str:
    return STAGE_LABELS.get(name) or longform.STAGE_LABELS.get(name, name)


def load_json(path: Path, default: Any = None) -> Any:
    return _read_json(path, default)

