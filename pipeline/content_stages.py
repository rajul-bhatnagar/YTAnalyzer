"""Phase 4 stages: one stage per Short (SHORTS_RENDER), thumbnails and the creator package (ASSET_PREP).

Each Short is its own stage fingerprinted on its clip times, layout, captions, hook text, audio mix and transcript,
so editing one Short re-renders only that Short. Title/description edits never re-render video.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from . import (activity_signals as sig, audio_mix, captions, longform, short_edit, short_factcheck, shorts_builder,
               style_profiles, thumbnail_prep, timeline_builder, transcribe)
from . import youtube_metadata as ym
from .ai_providers.base import ProviderError, ProviderUnavailable, QuotaExceeded
from .analysis_stages import transcript_segments_video_time, words_video_time
from . import ffmpeg_tools as fx
from .database import Database, from_json, to_json, utcnow
from .hardware import load_report
from .job_manager import StageContext, StageResult, StageSpec
from .manifest import manifest_entry, quick_fingerprint
from .paths import StudioPaths

PACK_PROMPT_VERSION = 1
TARGET_LABELS = {"voiceover": "Apply voiceover: re-mix, re-transcribe, re-render Shorts, update package",
                 "shorts": "Render Shorts", "assets": "Prepare thumbnails & creator package",
                 "thumbnails": "Prepare thumbnail frames & prompts", "pack": "Write the creator package"}
STAGE_LABELS = {"thumbnails": "Thumbnail frames & prompts", "creator_pack": "Creator package (titles, script, "
                "description, tags)"}
_fp, _read_json, _write_json = longform._fp, longform._read_json, longform._write_json


def stage_label(name: str) -> str | None:
    if name.startswith("short_h"):
        return f"Short for highlight #{name.removeprefix('short_h')}"
    return STAGE_LABELS.get(name)


def default_short_settings(settings: dict[str, Any], style: str = "Custom") -> shorts_builder.ShortSettings:
    """A new Short starts from the content-style profile (layout, dead-air handling) and the project's caption
    settings. For a split layout whose profile reads HUD/score text, the bottom panel starts as a full-width strip
    along the bottom of the frame (adjustable on the Shorts page)."""
    s = settings["shorts"]
    prof = style_profiles.resolve(style, settings)
    strip = prof.default_shorts_layout == "split_stack" and prof.hud_strip_bottom
    return shorts_builder.ShortSettings(
        layout=prof.default_shorts_layout, quality=s["quality"], encoder=settings["render"]["encoder"],
        speedup_factor=float(s.get("speedup_factor") or 2.0),
        **({"bottom_mode": "region", "region_x": 0.0, "region_y": 1.0, "region_w": 1.0, "region_h": 0.18}
           if strip else {}),
        captions=captions.CaptionStyle(enabled=bool(s["captions_enabled"]), position=s["caption_position"],
                                       size=int(s["caption_size"]), hook_mode=s["hook_mode"]))


def approved_highlights(db: Database, project_id: str) -> list[dict[str, Any]]:
    rows = db.query("SELECT * FROM highlights WHERE project_id = ? AND status = 'approved' ORDER BY rank, id",
                    (project_id,))
    out = []
    for r in rows:
        d = dict(r)
        for k in ("score_breakdown", "evidence", "handles", "event_keys"):
            d[k] = from_json(d[k])
        d["eff_start"] = d["user_start_sec"] if d["user_start_sec"] is not None else d["start_sec"]
        d["eff_end"] = d["user_end_sec"] if d["user_end_sec"] is not None else d["end_sec"]
        out.append(d)
    return out


def short_settings(db: Database, highlight_id: int, settings: dict[str, Any],
                   style: str = "Custom") -> shorts_builder.ShortSettings:
    row = db.query_one("SELECT settings FROM shorts WHERE highlight_id = ?", (highlight_id,))
    if row is None:
        return default_short_settings(settings, style)
    return shorts_builder.ShortSettings.model_validate(json.loads(row["settings"]))


def build_stages(db: Database, paths: StudioPaths, project: dict[str, Any], settings: dict[str, Any], target: str,
                 highlight_ids: list[int] | None = None) -> tuple[longform.LongformContext, list[StageSpec]]:
    lc = longform.LongformContext(db, paths, project, settings)
    f = lc.files
    src_id = f"{project['source_path']}|{project['source_size']}|{project['source_quick_fp']}"
    has_audio = bool(lc.role_tracks()) or lc.voiceover_wav() is not None
    specs: list[StageSpec] = []

    if target in ("shorts", "voiceover"):
        approved = approved_highlights(db, project["id"])
        if highlight_ids is not None:
            approved = [h for h in approved if h["id"] in set(highlight_ids)]
        if not approved:
            raise ValueError("Approve at least one highlight first (Analysis & highlights page).")
        p2 = ["extract_audio", "voiceover", "transcribe", "subtitles"] + (["mix"] if has_audio else [])
        _lc, specs = longform.build_stages(db, paths, project, settings, "audio", stage_names=p2)
        for h in approved:
            specs.append(_short_spec(db, paths, lc, project, settings, h, src_id, has_audio))
        if target == "voiceover":
            specs.append(_pack_spec(db, paths, lc, project, settings))
        return lc, specs
    if target in ("assets", "thumbnails"):
        specs.append(_thumb_spec(db, paths, lc, project, settings, src_id))
    if target in ("assets", "pack"):
        specs.append(_pack_spec(db, paths, lc, project, settings))
    if not specs:
        raise ValueError(f"Unknown target '{target}'")
    return lc, specs


# --- Shorts ---------------------------------------------------------------------------------------------------
def _short_spec(db, paths, lc, project, settings, h, src_id, has_audio) -> StageSpec:
    f = lc.files
    hid = h["id"]

    def encoder() -> tuple[str, str]:
        s = short_settings(db, hid, settings, project["content_style"])
        hw = load_report(paths.hardware_report_file)
        return timeline_builder.choose_encoder(s.encoder, bool(hw and hw.nvenc_encode_ok))

    def dead_mode(s: shorts_builder.ShortSettings) -> str:
        return s.dead_air or style_profiles.resolve(project["content_style"], settings).shorts_dead_air_mode

    def inputs(ctx):
        s = short_settings(db, hid, settings, project["content_style"])
        mix = f.mix_wav(lc.mix_settings().mode)
        voice = paths.from_stored(s.voice_file) if s.voice_file else None
        return [src_id, _fp(mix) if has_audio else "noaudio", _fp(f.words_json), _fp(f.signals),
                _fp(voice) if voice else "novoice"], {
            "times": [h["eff_start"], h["eff_end"], h["event_sec"]], "hook": h["hook_caption"] or "",
            "settings": s.model_dump(), "dead_air": dead_mode(s), "encoder": encoder()[0],
            "shift": lc.video_shift, "v": 2}

    def execute(ctx: StageContext) -> StageResult:
        s = short_settings(db, hid, settings, project["content_style"])
        start, end = float(h["eff_start"]), float(h["eff_end"])
        if not 0 <= start < end <= lc.duration + 0.01:
            raise ValueError(f"Clip {start:.2f}-{end:.2f}s is outside the recording.")
        warnings: list[str] = []
        # 1. Dead air: trim / speed up static lead-in and tail, from the measured activity (this Short only).
        mode, segs = dead_mode(s), None
        if mode in ("trim", "speedup") and f.signals.exists():
            segs, notes = short_edit.plan_segments(
                sig.load_signals(f.signals), start, end, float(h["event_sec"]), mode, s.speedup_factor,
                float(settings["silence"]["motion_threshold"]))
            warnings += notes
        segs = segs or [(0.0, round(end - start, 3), 1.0)]
        out_dur = short_edit.output_length(segs)
        # 2. Audio: the project mix, or this Short's own voiceover over the game audio.
        audio = f.mix_wav(lc.mix_settings().mode) if has_audio else None
        if audio is not None and not audio.exists():
            raise ValueError("The audio mix is missing; it should have been rendered by the 'mix' stage.")
        audio_start = None
        words = captions.clip_words(words_video_time(f), start, end)
        if s.voice_file:
            audio, words, vw = short_voice_audio(paths, settings, project["language"], s, audio, start, end,
                                                 f.short_work(hid))
            audio_start = 0.0
            warnings += vw
        # 3. Captions in OUTPUT time (moved by the edits).
        words = short_edit.remap_words(words, segs)
        hook = h["hook_caption"] or ""
        ass = None
        if s.captions.enabled or (hook and s.captions.hook_enabled):
            ass, info = captions.build_ass(words, out_dur, s.captions, hook)
            errs = captions.validate_ass_timing(ass, out_dur)
            if errs:
                raise ValueError("Caption timing invalid: " + "; ".join(errs[:3]))
            if s.captions.enabled and not words:
                warnings.append("No transcribed speech in this clip, so there are no word captions"
                                + (" (the hook caption is shown instead)." if info["hook_shown"] else "."))
        # 4. Fact-check the hook and title against the evidence inside this clip (warnings, never silent edits).
        ev = short_factcheck.gather_evidence(db, project["id"], words_video_time(f), f.hud_ocr, project["game_name"],
                                             start, end)
        warnings += short_factcheck.check_short(hook, h["suggested_title"] or "", ev)["messages"]
        enc, why = encoder()
        out = f.short_out(hid)
        backup = timeline_builder.backup_existing(out)
        size = lc.info.display_resolution
        if size is None:
            raise ValueError("The recording has no video stream.")
        rep = shorts_builder.render_short(lc.source, audio, start, end, s, size, lc.info.fps, enc, f.short_work(hid),
                                          ass, out, progress=lambda x: ctx.progress(0.05 + 0.8 * x,
                                                                                     f"Rendering Short #{hid}"),
                                          cancel_check=ctx.cancel_check, segments=segs, shift=lc.video_shift,
                                          audio_start=audio_start)
        ctx.progress(0.9, f"Validating Short #{hid}")
        errors, details = shorts_builder.validate_short(out, out_dur, lc.source, start, audio is not None,
                                                        f.short_work(hid), audio if audio_start is None else None)
        warnings += details["warnings"]
        rep.update({"encoder_reason": why, "backup": paths.to_stored(backup) if backup else None,
                    "highlight_id": hid, "start": start, "end": end, "dead_air": mode})
        db.execute("INSERT INTO rendered_outputs(project_id, output_type, path, highlight_id, settings, "
                   "validation_status, validation_details, created_at) VALUES (?, 'short', ?, ?, ?, ?, ?, ?)",
                   (project["id"], paths.to_stored(out), hid, to_json({**rep, "settings": s.model_dump()}),
                    "passed" if not errors else "failed", to_json({"errors": errors, "probe": details["probe"]}),
                    utcnow()))
        if audio is None:
            warnings.append("The recording has no audio track, so this Short is silent.")
        return StageResult(manifest=[manifest_entry(out, paths.root, f"short_h{hid}", {"ok": not errors,
                                                                                      "errors": errors},
                                                    media={"duration": details["probe"]["duration_sec"],
                                                           "width": 1080, "height": 1920}, checksum=False)],
                           warnings=warnings, data={"errors": errors})

    return StageSpec(f"short_h{hid}", 1, execute, inputs, lambda ctx, res: res.data.get("errors", []), optional=True)


def short_voice_audio(paths: StudioPaths, settings: dict[str, Any], language: str, s: shorts_builder.ShortSettings,
                      mix_audio: Path | None, start: float, end: float,
                      work: Path) -> tuple[Path, list[dict[str, Any]], list[str]]:
    """A Short with its own voiceover: (clip-length audio, caption words in clip time, warnings).
    The game audio of the clip (from the project mix) is ducked under the voice and the result is loudness-normalised
    (two-pass EBU R128). Caption words come from Whisper on the voiceover, cached per file."""
    voice = paths.from_stored(s.voice_file)
    if not voice.exists():
        raise ValueError("This Short's voiceover file is missing. Upload it again on the Shorts page.")
    clip = round(end - start, 3)
    work.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    game_seg = None
    if mix_audio is not None:
        game_seg = work / "game_segment.wav"
        fx.ffmpeg(["-y", "-ss", f"{start:.3f}", "-t", f"{clip:.3f}", "-i", str(mix_audio), "-c:a", "pcm_s16le",
                   "-ar", "48000", str(game_seg)], timeout=300)
    ms = audio_mix.MixSettings(mode="both_auto_duck" if game_seg else "voice_only", voice_source="voiceover",
                               game_volume_db=0.0, voice_gain_db=s.voice_gain_db, voice_offset_sec=s.voice_offset,
                               loudnorm_enabled=True, limiter_enabled=True)
    out = work / "short_audio.wav"
    audio_mix.render_mix(ms, audio_mix.MixSources(game_seg, "game" if game_seg else None, voice, "voiceover"),
                         clip, out)
    cache = voice.with_name(voice.stem + ".words.json")
    fp = quick_fingerprint(voice)
    data = _read_json(cache, {}) or {}
    if data.get("fp") != fp:
        try:
            wav16 = work / "voice_16k.wav"
            fx.ffmpeg(["-y", "-i", str(voice), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav16)], timeout=300)
            hw = load_report(paths.hardware_report_file)
            free = max((g.memory_free_mb or 0 for g in hw.gpus), default=None) if hw and hw.gpus else None
            w = settings["whisper"]
            choice = transcribe.choose_model(w, bool(hw and hw.cuda_verified), free)
            res = transcribe.transcribe_file(wav16, choice, language, bool(w["vad_filter"]), int(w["beam_size"]),
                                             bool(w.get("allow_download")), float(w["uncertain_word_probability"]))
            data = {"fp": fp, "words": [{"start": x["start"], "end": x["end"], "word": x["word"]}
                                        for x in transcribe.words_flat(res)]}
            _write_json(cache, data)
        except Exception as e:  # noqa: BLE001 - captions are optional; the Short still renders with sound
            warnings.append(f"Voiceover captions unavailable ({type(e).__name__}: {str(e)[:150]}).")
            data = {"words": []}
    words = [{**x, "start": round(x["start"] + s.voice_offset, 3), "end": round(x["end"] + s.voice_offset, 3)}
             for x in data.get("words", []) if x["start"] + s.voice_offset < clip]
    return out, words, warnings


# --- thumbnails -------------------------------------------------------------------------------------------------
def _highlights_for_assets(db: Database, project_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    hs = approved_highlights(db, project_id)
    if hs:
        return hs, []
    rows = db.query("SELECT * FROM highlights WHERE project_id = ? AND status = 'suggested' ORDER BY rank LIMIT 3",
                    (project_id,))
    out = []
    for r in rows:
        d = dict(r)
        d["evidence"] = from_json(d["evidence"])
        d["event_keys"] = from_json(d["event_keys"])
        d["eff_start"] = d["user_start_sec"] if d["user_start_sec"] is not None else d["start_sec"]
        d["eff_end"] = d["user_end_sec"] if d["user_end_sec"] is not None else d["end_sec"]
        out.append(d)
    return out, (["No highlight is approved yet: the top suggestions were used."] if out else [])


def _hl_signature(hs: list[dict[str, Any]]) -> str:
    data = [(h["id"], h["eff_start"], h["eff_end"], h["event_sec"], h["hook_caption"], h["suggested_title"],
             h["status"]) for h in hs]
    return hashlib.sha256(json.dumps(data, default=str).encode()).hexdigest()[:16]


def _thumb_spec(db, paths, lc, project, settings, src_id) -> StageSpec:
    f = lc.files

    def inputs(ctx):
        hs, _ = _highlights_for_assets(db, project["id"])
        return [src_id], {"hl": _hl_signature(hs), "count": settings["thumbnails"]["count"]}

    def execute(ctx: StageContext) -> StageResult:
        hs, warnings = _highlights_for_assets(db, project["id"])
        if not hs:
            raise ValueError("No highlights yet. Run the analysis first.")
        frames = thumbnail_prep.select_frames(lc.source, hs, lc.duration, f.thumbs_dir,
                                              int(settings["thumbnails"]["count"]), ctx.cancel_check)
        if not frames:
            raise ValueError("No usable thumbnail frame could be extracted.")
        prompts = {}
        for h in hs:
            best = next((fr for fr in frames if fr["highlight_id"] == h["id"]), None)
            prompts[str(h["id"])] = thumbnail_prep.build_prompts(project["game_name"], project["content_style"], h,
                                                                 best["t"] if best else None)
        data = {"frames": [{**fr, "file": paths.to_stored(fr["file"])} for fr in frames], "prompts": prompts,
                "highlights": [h["id"] for h in hs], "created_at": utcnow()}
        _write_json(f.thumbs_json, data)
        bad = [fr for fr in frames if fr["issues"]]
        if len(frames) < int(settings["thumbnails"]["count"]):
            warnings.append(f"Only {len(frames)} distinct, usable frames were found around the highlights.")
        if bad:
            warnings.append(f"{len(bad)} chosen frame(s) have issues ({', '.join(sorted({i for b in bad for i in b['issues']}))}).")
        if any(p["needs_user_words"] for p in prompts.values()):
            warnings.append("Some prompts contain [placeholders]: add a title or hook to the highlight first.")
        return StageResult(manifest=[manifest_entry(fr["file"], paths.root, "thumbnails", {"ok": True}, checksum=False)
                                     for fr in frames] + [manifest_entry(f.thumbs_json, paths.root, "thumbnails",
                                                                         {"ok": True})],
                           warnings=warnings, data={"frames": len(frames)})

    return StageSpec("thumbnails", 1, execute, inputs)


# --- creator package --------------------------------------------------------------------------------------------
def pack_context(db: Database, paths: StudioPaths, lc: longform.LongformContext, project: dict[str, Any],
                 settings: dict[str, Any]) -> dict[str, Any]:
    hs = approved_highlights(db, project["id"])
    chosen_keys = {k for h in hs for k in (h["event_keys"] or [])}
    events = [{"key": r["event_key"], "start": r["start_sec"], "end": r["end_sec"], "category": r["event_type"],
               "description": r["description"], "chosen": r["event_key"] in chosen_keys}
              for r in db.query("SELECT * FROM events WHERE project_id = ? ORDER BY start_sec", (project["id"],))]
    lang = settings["pack"]["language"] or project["language"]

    def what(h: dict[str, Any]) -> str:
        return (h["rationale"] or "").split(" Supported by measurements")[0].split(" (No AI description")[0]

    return {"game": project["game_name"], "style": project["content_style"], "topic": project["topic_note"],
            "language": lang, "duration": lc.duration, "events": events,
            "highlights": [{"id": h["id"], "start": h["eff_start"], "end": h["eff_end"], "event": h["event_sec"],
                            "title": h["suggested_title"] or "", "hook": h["hook_caption"] or "", "what": what(h),
                            "mood": h.get("outcome_mood") or ""}
                           for h in hs],
            "transcript": transcript_segments_video_time(lc.files),
            "ocr": [t for x in (_read_json(lc.files.hud_ocr, {}) or {}).get("frames", []) for t in x["texts"]]}


def video_time_mapper(cut_decisions: dict[str, Any] | None) -> Callable[[float], float | None]:
    """Recording time -> rough-cut time (None if that moment was cut out). Identity without cut decisions."""
    if not cut_decisions:
        return lambda t: t
    segs = [s for s in cut_decisions["segments"] if s["keep"]]

    def m(t: float) -> float | None:
        for s in segs:
            if s["source_start"] <= t < s["source_end"]:
                return round(s["output_start"] + (t - s["source_start"]), 2)
        return None
    return m


def _pack_spec(db, paths, lc, project, settings) -> StageSpec:
    f = lc.files

    def inputs(ctx):
        hs = approved_highlights(db, project["id"])
        return [_fp(f.fused_events), _fp(f.words_json), _fp(f.thumbs_json), _fp(f.cut_decisions)], {
            "hl": _hl_signature(hs), "plan": ctx.data.get("plan_descriptor", {"provider": "none"}),
            "lang": settings["pack"]["language"] or project["language"], "v": PACK_PROMPT_VERSION}

    def execute(ctx: StageContext) -> StageResult:
        pc = pack_context(db, paths, lc, project, settings)
        plan = ctx.data.get("plan")
        warnings: list[str] = list(ctx.data.get("plan_notes", []))
        if not pc["highlights"]:
            warnings.append("No highlight is approved yet; the package covers the detected events only.")
        draft, issues, source = None, [], {"provider": "template", "model": "", "mode": "template"}
        slots = [s for s in ((plan.primary, plan.fallback) if plan else ()) if s is not None]
        system, prompt = ym.build_prompt(pc)
        for slot in slots:
            try:
                raw = slot.provider.generate_json(system, prompt, ym.schema())
                try:
                    draft, issues = ym.validate_draft(raw, pc)
                except ym.InvalidPack as bad:
                    raw = slot.provider.generate_json(system, prompt + f"\n\nYour previous answer was invalid ({bad}). "
                                                      f"Return ONLY corrected JSON.\n{raw[:4000]}", ym.schema())
                    draft, issues = ym.validate_draft(raw, pc)
                    issues.insert(0, "The first answer was invalid; a repaired answer was used.")
                source = {"provider": slot.provider.name, "model": slot.provider.model, "mode": "text"}
                break
            except (QuotaExceeded, ProviderUnavailable, ProviderError, ym.InvalidPack) as e:
                warnings.append(f"{slot.provider.name} could not write the package ({str(e)[:200]}).")
                draft = None
        if draft is None:
            draft, issues = ym.template_draft(pc), []
            warnings.append("The package is a template with [placeholders] (no AI provider produced a valid draft). "
                            "Edit it on the Metadata page, or regenerate later.")
        thumbs = _read_json(f.thumbs_json, {})
        prompts = None
        if pc["highlights"] and thumbs.get("prompts"):
            prompts = thumbs["prompts"].get(str(pc["highlights"][0]["id"]))
        elif thumbs.get("prompts"):
            prompts = next(iter(thumbs["prompts"].values()))
        pack = ym.assemble(draft, pc, source, issues, prompts, video_time_mapper(_read_json(f.cut_decisions)))
        _write_json(f.pack_draft, pack)
        write_effective_pack(f)
        if pack["number_flags"]:
            warnings.append(f"{len(pack['number_flags'])} number(s) are not supported by the evidence; see "
                            "'Check before publishing'.")
        return StageResult(manifest=[manifest_entry(p, paths.root, "creator_pack", {"ok": True})
                                     for p in (f.pack_draft, f.pack_json, f.pack_txt)],
                           warnings=warnings, data={"source": source})

    return StageSpec("creator_pack", 1, execute, inputs)


def write_effective_pack(f: longform.ProjectFiles) -> dict[str, Any] | None:
    """Draft + the user's edits -> youtube_pack.json (canonical) and youtube_pack.txt. No AI, no video work."""
    draft = _read_json(f.pack_draft)
    if draft is None:
        return None
    pack = ym.apply_edits(draft, _read_json(f.pack_edits, {}))
    pack["written_at"] = utcnow()
    _write_json(f.pack_json, pack)
    f.pack_txt.write_text(ym.to_text(pack), encoding="utf-8")
    return pack
