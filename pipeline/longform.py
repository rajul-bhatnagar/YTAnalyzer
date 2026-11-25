"""Phase 2 stage definitions (AUDIO_PREP): audio extraction, voiceover, activity signals, transcription,
subtitles, mix, inactivity analysis, rough cut and timeline export.

Each stage is fingerprinted on exactly the inputs that determine its output, so e.g. changing the mix
re-runs only the mix (and the rough cut that uses it), never extraction or analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import activity_signals as sig
from . import audio_mix, audio_prep, silence_cut, timeline_builder, timeline_export, transcribe
from .database import Database, to_json, utcnow
from .hardware import load_report
from .job_manager import StageContext, StageResult, StageSpec
from .manifest import manifest_entry, quick_fingerprint
from .models import MediaInfo, TrackRole
from .paths import StudioPaths

TARGETS = {
    "audio": ["extract_audio", "voiceover", "signals", "mix"],
    "transcript": ["extract_audio", "voiceover", "transcribe", "subtitles"],
    "subtitles": ["extract_audio", "voiceover", "transcribe", "subtitles"],
    "cuts": ["extract_audio", "voiceover", "signals", "cut_analysis"],
    "render": ["extract_audio", "voiceover", "signals", "mix", "cut_analysis", "rough_cut", "timeline_export"],
}

TARGET_LABELS = {
    "audio": "Prepare audio & render mix",
    "transcript": "Transcribe & build subtitles",
    "subtitles": "Rebuild subtitles",
    "cuts": "Analyse inactive scenes",
    "render": "Render rough cut & timeline",
}

# The smallest operation that re-runs a given stage (used by Retry / Resume).
_STAGE_TARGET = {"extract_audio": "audio", "voiceover": "audio", "signals": "audio", "mix": "audio",
                 "transcribe": "transcript", "subtitles": "subtitles", "cut_analysis": "cuts",
                 "rough_cut": "render", "timeline_export": "render"}


def target_for_stage(stage_name: str) -> str | None:
    """Operation that re-runs ``stage_name``; None for stages that are not long-form (e.g. inspection)."""
    return _STAGE_TARGET.get(stage_name)


STAGE_LABELS = {
    "extract_audio": "Extract audio tracks (sync-aligned WAV)",
    "voiceover": "Prepare voiceover",
    "signals": "Activity signals (audio level, motion, speech)",
    "transcribe": "Transcribe speech (faster-whisper)",
    "subtitles": "Build transcript, subtitles and word timings",
    "mix": "Render audio mix",
    "cut_analysis": "Inactive-scene analysis (cut report)",
    "rough_cut": "Render long-form rough cut",
    "timeline_export": "Export editor timeline (EDL / FCP7 XML)",
}


@dataclass(frozen=True)
class ProjectFiles:
    """All Phase 2 file locations for one project."""

    work: Path
    ready: Path

    @classmethod
    def of(cls, paths: StudioPaths, folder: str) -> "ProjectFiles":
        return cls(paths.project_work_dir(folder), paths.project_ready_dir(folder))

    @property
    def tracks_dir(self) -> Path:
        return self.work / "audio" / "tracks"

    def track_wav(self, idx: int, role: str) -> Path:
        return self.tracks_dir / f"track{idx + 1}_{role}.wav"

    def track_wav16(self, idx: int, role: str) -> Path:
        return self.tracks_dir / f"track{idx + 1}_{role}_16k.wav"

    @property
    def tracks_json(self) -> Path:
        return self.tracks_dir / "tracks.json"

    @property
    def voiceover_dir(self) -> Path:
        return self.work / "audio" / "voiceover"

    @property
    def mixes_dir(self) -> Path:
        return self.work / "audio" / "mixes"

    def mix_wav(self, mode: str) -> Path:
        return self.mixes_dir / f"mix_{mode}.wav"

    def mix_report(self, mode: str) -> Path:
        return self.mixes_dir / f"mix_{mode}.json"

    @property
    def previews_dir(self) -> Path:
        return self.work / "audio" / "previews"

    @property
    def signals(self) -> Path:
        return self.work / "signals" / "signals.npz"

    @property
    def transcript_raw(self) -> Path:
        return self.work / "transcript" / "transcript_raw.json"

    @property
    def transcript_edits(self) -> Path:
        return self.work / "transcript" / "edits.json"

    @property
    def cut_overrides(self) -> Path:
        return self.work / "cuts" / "overrides.json"

    @property
    def cut_decisions(self) -> Path:
        return self.ready / "cut_decisions.json"

    @property
    def transcript_txt(self) -> Path:
        return self.ready / "transcript.txt"

    @property
    def subtitles_srt(self) -> Path:
        return self.ready / "subtitles.srt"

    @property
    def words_json(self) -> Path:
        return self.ready / "words.json"

    @property
    def rough_cut(self) -> Path:
        return self.ready / "long_rough_cut.mp4"

    @property
    def timeline_json(self) -> Path:
        return self.ready / "timeline.json"

    @property
    def edl(self) -> Path:
        return self.ready / "long_rough_cut.edl"

    @property
    def fcp7_xml(self) -> Path:
        return self.ready / "long_rough_cut_fcp7.xml"

    # --- Phase 3: analysis ---------------------------------------------------------------
    @property
    def analysis_dir(self) -> Path:
        return self.work / "analysis"

    @property
    def signal_events(self) -> Path:
        return self.analysis_dir / "signal_events.json"

    @property
    def frames_dir(self) -> Path:
        return self.analysis_dir / "frames"

    @property
    def frames_index(self) -> Path:
        return self.analysis_dir / "frames.json"

    @property
    def windows_dir(self) -> Path:
        return self.analysis_dir / "windows"

    @property
    def video_chunks_dir(self) -> Path:
        return self.analysis_dir / "video_chunks"

    @property
    def ai_report(self) -> Path:
        return self.analysis_dir / "ai_report.json"

    @property
    def hud_ocr(self) -> Path:
        return self.analysis_dir / "hud_ocr.json"

    @property
    def fused_events(self) -> Path:
        return self.analysis_dir / "fused_events.json"

    @property
    def highlights_json(self) -> Path:
        return self.ready / "highlights.json"

    @property
    def clip_previews_dir(self) -> Path:
        return self.analysis_dir / "clip_previews"

    # --- Phase 4: content ------------------------------------------------------------------
    @property
    def shorts_dir(self) -> Path:
        return self.ready / "shorts"

    def short_out(self, highlight_id: int) -> Path:
        return self.shorts_dir / f"short_h{highlight_id}.mp4"

    def short_work(self, highlight_id: int) -> Path:
        return self.work / "shorts" / f"h{highlight_id}"

    @property
    def thumbs_dir(self) -> Path:
        return self.ready / "thumbnail_frames"

    @property
    def cutouts_dir(self) -> Path:
        return self.thumbs_dir / "cutouts"

    @property
    def thumbs_json(self) -> Path:
        return self.work / "assets" / "thumbnails.json"

    @property
    def user_thumbnail_dir(self) -> Path:
        return self.ready / "thumbnail"

    @property
    def pack_draft(self) -> Path:
        return self.work / "assets" / "pack_draft.json"

    @property
    def pack_edits(self) -> Path:
        return self.work / "assets" / "pack_edits.json"

    @property
    def pack_json(self) -> Path:
        return self.ready / "youtube_pack.json"

    @property
    def pack_txt(self) -> Path:
        return self.ready / "youtube_pack.txt"


def _fp(path: Path) -> str:
    return f"{path.name}:{quick_fingerprint(path)}" if path.exists() else f"{path.name}:missing"


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


class LongformContext:
    """Per-run view of the project: tracks, settings and resolved audio sources."""

    def __init__(self, db: Database, paths: StudioPaths, project: dict[str, Any], settings: dict[str, Any]):
        self.db, self.paths, self.project, self.settings = db, paths, project, settings
        self.files = ProjectFiles.of(paths, project["folder_name"])
        self.source = paths.from_stored(project["source_path"])
        self.info = MediaInfo.model_validate(json.loads(project["source_metadata"]))
        self.tracks = [dict(r) for r in db.query(
            "SELECT * FROM audio_tracks WHERE project_id = ? ORDER BY audio_index", (project["id"],))]
        self.duration = self.info.duration_sec

    @property
    def video_shift(self) -> float:
        """See media_probe.video_seek_shift: add to every seek into the source video."""
        from .media_probe import video_seek_shift  # noqa: PLC0415
        return video_seek_shift(self.info)

    # --- tracks / sources -------------------------------------------------------------
    def role_tracks(self) -> dict[str, int]:
        return audio_prep.roles_by_index(self.tracks)

    def track_files(self) -> dict[str, Path]:
        return {role: self.files.track_wav(idx, role) for role, idx in self.role_tracks().items()}

    def voiceover_wav(self) -> Path | None:
        rel = self.settings["audio"].get("voiceover_file") or ""
        if not rel:
            return None
        p = self.paths.from_stored(rel)
        return p if p.exists() else None

    def voiceover_16k(self) -> Path | None:
        vo = self.voiceover_wav()
        return vo.with_name(vo.stem + "_16k.wav") if vo else None

    def mix_settings(self) -> audio_mix.MixSettings:
        return audio_mix.MixSettings.from_settings(self.settings["audio"])

    def mix_sources(self) -> audio_mix.MixSources:
        return audio_mix.resolve_sources(self.track_files(), self.voiceover_wav(), self.mix_settings().voice_source)

    def voice_16k(self) -> tuple[Path | None, float]:
        """16 kHz voice used for speech evidence, and its offset to video time: the voiceover when it is the
        active voice source, else the live mic track."""
        if self.mix_sources().voice_kind == "voiceover":
            return self.voiceover_16k(), self.mix_settings().voice_offset_sec
        roles = self.role_tracks()
        if TrackRole.MIC.value in roles:
            return self.files.track_wav16(roles[TrackRole.MIC.value], TrackRole.MIC.value), 0.0
        return None, 0.0

    def activity_audio(self) -> Path | None:
        """16 kHz track describing game activity: game track, else combined mix."""
        roles = self.role_tracks()
        for role in (TrackRole.GAME.value, TrackRole.COMBINED_MIX.value):
            if role in roles:
                return self.files.track_wav16(roles[role], role)
        return None

    def transcription_source(self) -> tuple[str, Path | None, float]:
        """(kind, 16 kHz wav, time offset to video) per transcription.source setting."""
        pref = self.settings["transcription"].get("source", "auto")
        roles = self.role_tracks()
        vo16 = self.voiceover_16k()
        ms = self.mix_settings()
        options: list[tuple[str, Path | None, float]] = []
        if pref in ("auto", "voiceover") and vo16 is not None:
            options.append(("voiceover", vo16, ms.voice_offset_sec))
        for role in ((TrackRole.MIC.value, TrackRole.COMBINED_MIX.value, TrackRole.GAME.value) if pref == "auto"
                     else (pref,)):
            if role in roles:
                options.append((role, self.files.track_wav16(roles[role], role), 0.0))
        if pref == "none" or not options:
            return "none", None, 0.0
        return options[0]


def build_stages(db: Database, paths: StudioPaths, project: dict[str, Any], settings: dict[str, Any],
                 target: str, stage_names: list[str] | None = None) -> tuple[LongformContext, list[StageSpec]]:
    """Stages for a long-form ``target``, or exactly ``stage_names`` (used by the analysis pipeline)."""
    lc = LongformContext(db, paths, project, settings)
    f = lc.files
    src_id = f"{project['source_path']}|{project['source_size']}|{project['source_quick_fp']}"
    a_cfg, s_cfg, w_cfg, r_cfg = settings["audio"], settings["silence"], settings["whisper"], settings["render"]

    # --- extract_audio ---------------------------------------------------------------------
    def ex_inputs(ctx):
        return [src_id], {"roles": lc.role_tracks(), "sr": a_cfg["final_sample_rate"], "asr": a_cfg["asr_sample_rate"]}

    def ex_exec(ctx: StageContext) -> StageResult:
        roles = lc.role_tracks()
        entries, warnings, info_out = [], [], {}
        for n, (role, idx) in enumerate(sorted(roles.items(), key=lambda kv: kv[1])):
            ctx.progress(0.05 + 0.6 * n / max(len(roles), 1), f"Extracting audio track {idx + 1} ({role})")
            e = audio_prep.extract_track(lc.source, lc.info, idx, f.track_wav(idx, role), f.track_wav16(idx, role),
                                         a_cfg["final_sample_rate"], a_cfg["asr_sample_rate"], ctx.cancel_check)
            stats = audio_prep.audio_stats(e.wav_48k)
            clip = audio_prep.clipping_report(stats)
            if clip:
                warnings.append(f"Track {idx + 1} ({role}): {clip}")
            info_out[role] = {"audio_index": idx, "wav": paths.to_stored(e.wav_48k),
                              "wav_16k": paths.to_stored(e.wav_16k), "offset_sec": e.offset_sec, "stats": stats}
            for p in (e.wav_48k, e.wav_16k):
                entries.append(manifest_entry(p, paths.root, "extract_audio", {"ok": True}, checksum=False))
            db.execute("UPDATE audio_tracks SET extraction_path = ? WHERE project_id = ? AND audio_index = ?",
                       (paths.to_stored(e.wav_48k), project["id"], idx))
        if not roles:
            warnings.append("No mapped audio tracks (silent recording or all tracks set to 'other').")
        _write_json(f.tracks_json, info_out)
        entries.append(manifest_entry(f.tracks_json, paths.root, "extract_audio", {"ok": True}))
        return StageResult(manifest=entries, warnings=warnings)

    def ex_validate(ctx, res):
        errs = []
        for role, idx in lc.role_tracks().items():
            d = audio_prep.probe_audio_file(f.track_wav(idx, role))["duration_sec"]
            if abs(d - lc.duration) > 0.1:
                errs.append(f"track {idx + 1} duration {d:.2f}s != video {lc.duration:.2f}s (sync)")
        return errs

    # --- voiceover -------------------------------------------------------------------------
    def vo_inputs(ctx):
        vo = lc.voiceover_wav()
        return [_fp(vo) if vo else "none"], {"asr": a_cfg["asr_sample_rate"]}

    def vo_exec(ctx):
        vo = lc.voiceover_wav()
        if vo is None:
            marker = f.voiceover_dir / "none.json"
            _write_json(marker, {"voiceover": None})
            return StageResult(manifest=[manifest_entry(marker, paths.root, "voiceover", {"ok": True})])
        ctx.progress(0.7, "Preparing voiceover")
        vo16 = audio_prep.make_asr_copy(vo, lc.voiceover_16k(), a_cfg["asr_sample_rate"], ctx.cancel_check)
        info = audio_prep.probe_audio_file(vo)
        warnings = []
        clip = audio_prep.clipping_report(audio_prep.audio_stats(vo))
        if clip:
            warnings.append(f"Voiceover: {clip}")
        if info["duration_sec"] > lc.duration + 1:
            warnings.append(f"Voiceover ({info['duration_sec']:.0f}s) is longer than the video "
                            f"({lc.duration:.0f}s); the excess will be cut off.")
        return StageResult(manifest=[manifest_entry(vo16, paths.root, "voiceover", {"ok": True}, checksum=False)],
                           warnings=warnings)

    # --- signals ---------------------------------------------------------------------------
    def sg_voice() -> tuple[Path | None, float]:
        return lc.voice_16k()

    def sg_inputs(ctx):
        act = lc.activity_audio()
        voice, off = sg_voice()
        return [src_id, _fp(act) if act else "noaudio", _fp(voice) if voice else "novoice"], {
            "fps": s_cfg["signal_fps"], "hop": s_cfg["signal_hop_sec"], "voice_offset": off}

    def sg_exec(ctx):
        hop, fps = float(s_cfg["signal_hop_sec"]), float(s_cfg["signal_fps"])
        grid = np.arange(0, lc.duration, hop, dtype=np.float32)
        act = lc.activity_audio()
        game_db = None
        if act and act.exists():
            ctx.progress(0.1, "Measuring game audio activity")
            t, r = sig.rms_timeline(act, hop, ctx.cancel_check)
            game_db = sig.resample_to_grid(t, r, grid, -120.0)
        ctx.progress(0.3, "Measuring visual motion")
        tm, mo = sig.motion_timeline(lc.source, fps, lc.duration,
                                     progress=lambda x: ctx.progress(0.3 + 0.5 * x, "Measuring visual motion"),
                                     cancel_check=ctx.cancel_check)
        motion = sig.resample_to_grid(tm, mo, grid, 0.0)
        voice, off = sg_voice()
        speech = None
        segs: list = []
        if voice and voice.exists():
            ctx.progress(0.85, "Detecting speech (VAD)")
            segs = [(s + off, e + off) for s, e in sig.speech_segments(voice, cancel_check=ctx.cancel_check)]
            speech = sig.speech_mask(segs, grid, hop)
        meta = {"hop_sec": hop, "fps": fps, "has_game_audio": game_db is not None, "has_speech_track": speech is not None,
                "speech_segments": len(segs), "speech_seconds": round(sum(e - s for s, e in segs), 1),
                "motion_mean": round(float(motion.mean()), 3) if len(motion) else None}
        sig.save_signals(f.signals, grid, game_db if game_db is not None else np.zeros(0, np.float32), motion,
                         speech, meta)
        return StageResult(manifest=[manifest_entry(f.signals, paths.root, "signals", {"ok": True})], data=meta)

    # --- transcribe ------------------------------------------------------------------------
    def tr_choice():
        hw = load_report(paths.hardware_report_file)
        free = max((g.memory_free_mb or 0 for g in hw.gpus), default=None) if hw and hw.gpus else None
        return transcribe.choose_model(w_cfg, bool(hw and hw.cuda_verified), free)

    def tr_inputs(ctx):
        kind, wav, _off = lc.transcription_source()
        ch = tr_choice()
        return [_fp(wav) if wav else "none"], {"kind": kind, "model": ch.effective, "device": ch.device,
                                               "language": project["language"], "vad": w_cfg["vad_filter"],
                                               "beam": w_cfg["beam_size"]}

    def tr_exec(ctx):
        kind, wav, _off = lc.transcription_source()
        if wav is None:
            _write_json(f.transcript_raw, {"source": "none", "segments": [], "skipped": "no voice track"})
            return StageResult(manifest=[manifest_entry(f.transcript_raw, paths.root, "transcribe", {"ok": True})],
                               warnings=["No speech track to transcribe (no mic, voiceover or combined track)."])
        ch = tr_choice()
        ai = settings.get("ai", {})
        if ch.device == "cuda" and ai.get("ollama_model"):  # 6 GB GPU: an idle Ollama model must not hold the VRAM
            from .ai_providers.ollama_provider import OllamaProvider  # noqa: PLC0415
            OllamaProvider(ai.get("ollama_base_url", ""), ai["ollama_model"]).unload()
        result = transcribe.transcribe_file(
            wav, ch, project["language"], bool(w_cfg["vad_filter"]), int(w_cfg["beam_size"]),
            bool(w_cfg.get("allow_download")), float(w_cfg["uncertain_word_probability"]),
            progress=lambda x, note: ctx.progress(0.05 + 0.9 * x, note), cancel_check=ctx.cancel_check)
        result["source"] = kind
        _write_json(f.transcript_raw, result)
        warnings = []
        if ch.warning:
            warnings.append(ch.warning)
        for fb in result["model"]["fallbacks"]:
            warnings.append(f"GPU transcription failed and fell back to CPU: {fb['error'][:150]}")
        n_unc = sum(1 for w in transcribe.words_flat(result) if w.get("uncertain"))
        if n_unc:
            warnings.append(f"{n_unc} low-confidence words flagged for review.")
        if project["language"] == "hinglish":
            warnings.append("Hinglish: Whisper has no dedicated Hinglish mode; check mixed-script words.")
        return StageResult(manifest=[manifest_entry(f.transcript_raw, paths.root, "transcribe", {"ok": True})],
                           warnings=warnings, data={"model": result["model"]})

    # --- subtitles -------------------------------------------------------------------------
    def sb_inputs(ctx):
        _kind, _wav, off = lc.transcription_source()
        return [_fp(f.transcript_raw), _fp(f.transcript_edits)], {"offset": off, "duration": lc.duration}

    def sb_exec(ctx):
        raw = _read_json(f.transcript_raw, {"segments": []})
        edits = _read_json(f.transcript_edits, {})
        _kind, _wav, off = lc.transcription_source()
        final = transcribe.shift(transcribe.apply_edits(raw, edits), off, lc.duration)
        cues = transcribe.caption_chunks(final)
        f.ready.mkdir(parents=True, exist_ok=True)
        f.transcript_txt.write_text(transcribe.to_txt(final), encoding="utf-8")
        f.subtitles_srt.write_text(transcribe.to_srt(cues), encoding="utf-8")
        _write_json(f.words_json, {"language": raw.get("language"), "model": raw.get("model"),
                                   "offset_applied": off, "words": transcribe.words_flat(final)})
        return StageResult(manifest=[manifest_entry(p, paths.root, "subtitles", {"ok": True})
                                     for p in (f.transcript_txt, f.subtitles_srt, f.words_json)],
                           data={"cues": len(cues)})

    def sb_validate(ctx, res):
        errs = []
        prev_end = -1.0
        for c in transcribe.caption_chunks(transcribe.shift(transcribe.apply_edits(
                _read_json(f.transcript_raw, {"segments": []}), _read_json(f.transcript_edits, {})),
                lc.transcription_source()[2], lc.duration)):
            if c["start"] < prev_end - 1e-6 or c["end"] > lc.duration + 0.01:
                errs.append(f"caption timing out of order/bounds at {c['start']:.2f}s")
                break
            prev_end = c["end"]
        return errs

    # --- mix -------------------------------------------------------------------------------
    def mx_inputs(ctx):
        ms = lc.mix_settings()
        src = lc.mix_sources()
        files = [p for p in (src.game, src.voice) if p]
        return [_fp(p) for p in files], {"mix": ms.model_dump(), "kinds": [src.game_kind, src.voice_kind],
                                         "duration": lc.duration}

    def mx_exec(ctx):
        ms, src = lc.mix_settings(), lc.mix_sources()
        out = f.mix_wav(ms.mode)
        rep = audio_mix.render_mix(ms, src, lc.duration, out,
                                   progress=lambda x: ctx.progress(0.1 + 0.85 * x, f"Rendering {ms.mode} mix"),
                                   cancel_check=ctx.cancel_check)
        errors, details = audio_mix.validate_mix(out, lc.duration)
        rep["validation"] = {"errors": errors, **details}
        rep["sources"] = {"game": src.game_kind, "voice": src.voice_kind}
        _write_json(f.mix_report(ms.mode), rep)
        return StageResult(manifest=[manifest_entry(out, paths.root, "mix", {"ok": not errors, "errors": errors},
                                                    checksum=False),
                                     manifest_entry(f.mix_report(ms.mode), paths.root, "mix", {"ok": True})],
                           data={"errors": errors, "mode": ms.mode})

    def mx_validate(ctx, res):
        return res.data.get("errors", [])

    # --- cut analysis ----------------------------------------------------------------------
    def ct_inputs(ctx):
        return [_fp(f.signals), _fp(f.cut_overrides)], {"silence": s_cfg, "duration": lc.duration}

    def ct_exec(ctx):
        data = sig.load_signals(f.signals)
        game = data["game_db"] if data["game_db"].size else None
        cands = silence_cut.find_candidates(data["grid"], game, data["motion"], data["speech"],
                                            float(data["meta"].get("hop_sec", s_cfg["signal_hop_sec"])),
                                            lc.duration, s_cfg)
        dec = silence_cut.build_decisions(lc.duration, cands, s_cfg["policy"], _read_json(f.cut_overrides, {}))
        dec["parameters"] = s_cfg
        dec["signals_meta"] = data["meta"]
        _write_json(f.cut_decisions, dec)
        warns = []
        if s_cfg["policy"] == "none" and cands:
            warns.append(f"{len(cands)} inactive stretch(es) suggested; nothing removed (policy 'none').")
        return StageResult(manifest=[manifest_entry(f.cut_decisions, paths.root, "cut_analysis", {"ok": True})],
                           warnings=warns, data={"removed_sec": dec["removed_sec"]})

    # --- rough cut -------------------------------------------------------------------------
    def rc_encoder():
        hw = load_report(paths.hardware_report_file)
        return timeline_builder.choose_encoder(r_cfg["encoder"], bool(hw and hw.nvenc_encode_ok))

    def rc_inputs(ctx):
        ms = lc.mix_settings()
        enc, _ = rc_encoder()
        return [src_id, _fp(f.mix_wav(ms.mode)), _fp(f.cut_decisions)], {
            "encoder": enc, "quality": r_cfg["quality_preset"], "abr": r_cfg["audio_bitrate_k"]}

    def rc_exec(ctx):
        ms = lc.mix_settings()
        dec = _read_json(f.cut_decisions)
        ranges = [tuple(r) for r in dec["kept_ranges"]]
        enc, reason = rc_encoder()
        rep = timeline_builder.render_rough_cut(
            lc.source, f.mix_wav(ms.mode), lc.info, ranges, f.rough_cut, enc, r_cfg["quality_preset"],
            int(r_cfg["audio_bitrate_k"]), progress=lambda x: ctx.progress(0.05 + 0.8 * x, "Rendering rough cut"),
            cancel_check=ctx.cancel_check)
        ctx.progress(0.9, "Validating rough cut")
        mix_rms = (_read_json(f.mix_report(ms.mode), {}).get("validation", {}).get("stats", {}) or {}).get("rms_db")
        errors, details = timeline_builder.validate_render(f.rough_cut, lc.info, rep["expected_duration"],
                                                           len(ranges), mix_rms)
        rep.update({"encoder_reason": reason, "validation_errors": errors, "mix_mode": ms.mode})
        timeline_builder.write_timeline_json(f.timeline_json, project["source_path"], lc.info, dec,
                                             paths.to_stored(f.mix_wav(ms.mode)), ms.model_dump(), lc.tracks)
        db.execute(
            "INSERT INTO rendered_outputs(project_id, output_type, path, settings, validation_status, "
            "validation_details, created_at) VALUES (?, 'long_rough_cut', ?, ?, ?, ?, ?)",
            (project["id"], paths.to_stored(f.rough_cut), to_json(rep), "passed" if not errors else "failed",
             to_json({"errors": errors, "probe": details["probe"]}), utcnow()))
        return StageResult(
            manifest=[manifest_entry(f.rough_cut, paths.root, "rough_cut", {"ok": not errors, "errors": errors},
                                     media={"duration": details["probe"]["duration_sec"], "encoder": rep["encoder"]},
                                     checksum=False),
                      manifest_entry(f.timeline_json, paths.root, "rough_cut", {"ok": True})],
            warnings=rep["notes"], data={"errors": errors, "report": rep})

    def rc_validate(ctx, res):
        return res.data.get("errors", [])

    # --- timeline export (optional) ----------------------------------------------------------
    def te_inputs(ctx):
        return [_fp(f.timeline_json)], {"edl": settings["timeline"]["export_edl"],
                                        "xml": settings["timeline"]["export_fcp7_xml"]}

    def te_exec(ctx):
        tl = _read_json(f.timeline_json)
        mix = paths.from_stored(tl["audio"]["mix"])
        fps = lc.info.fps or 30.0
        entries, warnings = [], ["Timeline files are structurally validated but NOT yet verified to import "
                                 "into DaVinci Resolve. The MP4 rough cut is the reliable output."]
        if lc.info.vfr_measured:
            warnings.append("Source has variable frame rate; editor timecodes are approximate.")
        if settings["timeline"]["export_edl"]:
            text = timeline_export.build_edl(project["name"], tl["clips"], fps, lc.source.name, mix.name,
                                             timeline_export.fcp_pathurl(lc.source), timeline_export.fcp_pathurl(mix))
            if len(timeline_export.parse_edl(text)) != 2 * len(tl["clips"]):
                raise ValueError("EDL self-check failed")
            f.edl.write_text(text, encoding="utf-8")
            entries.append(manifest_entry(f.edl, paths.root, "timeline_export", {"ok": True, "resolve_verified": False}))
        if settings["timeline"]["export_fcp7_xml"]:
            w, h = lc.info.display_resolution or (1920, 1080)
            text = timeline_export.build_fcp7_xml(project["name"], tl["clips"], fps, w, h, lc.source,
                                                  lc.duration, mix)
            errs = timeline_export.validate_fcp7_xml(text, len(tl["clips"]))
            if errs:
                raise ValueError("FCP7 XML self-check failed: " + "; ".join(errs))
            f.fcp7_xml.write_text(text, encoding="utf-8")
            entries.append(manifest_entry(f.fcp7_xml, paths.root, "timeline_export",
                                          {"ok": True, "resolve_verified": False}))
        return StageResult(manifest=entries, warnings=warnings)

    all_specs = {
        "extract_audio": StageSpec("extract_audio", 1, ex_exec, ex_inputs, ex_validate),
        "voiceover": StageSpec("voiceover", 1, vo_exec, vo_inputs),
        "signals": StageSpec("signals", 1, sg_exec, sg_inputs),
        "transcribe": StageSpec("transcribe", 1, tr_exec, tr_inputs, optional=True),
        "subtitles": StageSpec("subtitles", 1, sb_exec, sb_inputs, sb_validate, optional=True),
        "mix": StageSpec("mix", 1, mx_exec, mx_inputs, mx_validate),
        "cut_analysis": StageSpec("cut_analysis", 1, ct_exec, ct_inputs),
        "rough_cut": StageSpec("rough_cut", 1, rc_exec, rc_inputs, rc_validate),
        "timeline_export": StageSpec("timeline_export", 1, te_exec, te_inputs, optional=True),
    }
    names = stage_names if stage_names is not None else TARGETS.get(target)
    if not names:
        raise ValueError(f"Unknown target '{target}'")
    return lc, [all_specs[name] for name in names]
