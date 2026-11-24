"""Long video & audio studio: mix selection and previews, voiceover, transcript/captions, inactive-scene
review and the rough-cut render. All processing goes through StudioService; this module only renders UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from pipeline import audio_mix, audio_prep, ffmpeg_tools, longform, timeline_builder, transcribe
from pipeline.models import TRACK_ROLE_LABELS, TrackRole
from pipeline.project_service import StudioService
from ui.common import fmt_duration, fmt_size, get_service, job_status_panel, require_project

_MEDIA_ERRORS = (ValueError, ffmpeg_tools.FFmpegError, ffmpeg_tools.FFmpegNotFound)

TRANSCRIPTION_SOURCES = {
    "auto": "Auto (voiceover, else live mic, else combined mix, else game track)",
    "voiceover": "Post-recorded voiceover",
    "mic": "Live microphone track",
    "combined_mix": "Combined mix track",
    "game": "Game track (in-game commentary)",
    "none": "Don't transcribe",
}
CUT_POLICIES = {
    "none": "Keep everything: only suggest inactive stretches (default)",
    "conservative": "Conservative: remove long stretches that are static AND quiet",
}
ENCODERS = {"auto": "Auto (NVENC only if its real encode test passed)", "libx264": "libx264 (CPU, always available)",
            "h264_nvenc": "NVIDIA NVENC (used only if verified on this PC)"}


# --- helpers -----------------------------------------------------------------------------------
def _k(pid: str, name: str) -> str:
    """Widget keys are per project so switching projects never carries widget state across."""
    return f"as_{pid}_{name}"


def _busy(svc: StudioService, pid: str) -> bool:
    job = svc.latest_job(pid)
    return bool(job and svc.is_job_running(job["id"]))


def _start(svc: StudioService, pid: str, target: str) -> None:
    try:
        svc.start_longform(pid, target)
        st.session_state[_k(pid, "started")] = target
        st.rerun()
    except ValueError as e:
        st.error(str(e))


def _ts(sec: float | None) -> str:
    if sec is None:
        return "-"
    m, s = divmod(max(float(sec), 0.0), 60)
    return f"{int(m)}:{s:05.2f}"


def _download(label: str, path: Path, mime: str, key: str) -> None:
    if path.exists():
        st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime, key=key)


def _inline_limit_bytes(settings: dict[str, Any]) -> int:
    return int(settings["ui"]["inline_video_max_mb"]) * 1024 * 1024


def _status(project_id: str) -> None:
    job_status_panel(project_id, longform.STAGE_LABELS, "as")


# --- audio mix -----------------------------------------------------------------------------------
def _mix_tab(svc: StudioService, proj: dict[str, Any], lc: longform.LongformContext, busy: bool) -> None:
    pid = proj["id"]
    ms = lc.mix_settings()
    roles = lc.role_tracks()
    if roles:
        st.caption("Mapped tracks: " + " · ".join(
            f"{TRACK_ROLE_LABELS[TrackRole(r)]} → track {i + 1}" for r, i in sorted(roles.items(), key=lambda x: x[1])))
    else:
        st.warning("No audio tracks are mapped. Upload a voiceover (Voiceover tab) to give the video sound.")

    vs_keys = list(audio_mix.VOICE_SOURCES)
    voice_source = st.selectbox("Voice source", vs_keys, index=vs_keys.index(ms.voice_source)
                                if ms.voice_source in vs_keys else 0, format_func=audio_mix.VOICE_SOURCES.get,
                                key=_k(pid, "voice_source"))
    src = audio_mix.resolve_sources(lc.track_files(), lc.voiceover_wav(), voice_source)
    avail = audio_mix.available_modes(src)
    modes = list(audio_mix.MODES)
    mode = st.radio("Audio mode", modes, index=modes.index(ms.mode) if ms.mode in modes else 0,
                    format_func=lambda m: audio_mix.MODES[m] + ("" if avail[m] is None else "  · unavailable"),
                    key=_k(pid, "mode"))
    if avail[mode]:
        st.warning(avail[mode])
    if src.game_kind == "combined_mix":
        st.info("No separate game track is mapped, so the combined mix is used as the game audio. A single mixed "
                "track cannot be separated into game and voice.")
    st.caption(f"Game: {src.game_kind or 'none'} · Voice: {src.voice_kind or 'none'}")

    c1, c2 = st.columns(2)
    game_db = c1.slider("Game volume (dB)", -40.0, 12.0, float(ms.game_volume_db), 0.5, format="%.1f dB",
                        key=_k(pid, "game_db"), help="0 dB = unchanged. Default −6 dB keeps voice intelligible.")
    voice_db = c2.slider("Voice gain (dB)", -20.0, 20.0, float(ms.voice_gain_db), 0.5, format="%.1f dB",
                         key=_k(pid, "voice_db"))
    offset = st.number_input("Voiceover offset (seconds; positive = voice starts later)", -600.0, 600.0,
                             float(ms.voice_offset_sec), 0.1, format="%.2f", key=_k(pid, "offset"),
                             help="Applies to a post-recorded voiceover. Captions shift with it automatically.")

    with st.expander("Ducking (game + voice mode)"):
        st.caption("FFmpeg sidechain compression: the game is turned down while the voice is above the threshold.")
        d1, d2, d3 = st.columns(3)
        thr = d1.number_input("Threshold (0–1 linear)", 0.001, 1.0, float(ms.duck_threshold), 0.005, format="%.3f",
                              key=_k(pid, "thr"))
        ratio = d2.number_input("Ratio", 1.0, 20.0, float(ms.duck_ratio), 0.5, key=_k(pid, "ratio"))
        makeup = d3.number_input("Makeup gain", 1.0, 64.0, float(ms.duck_makeup), 0.1, key=_k(pid, "makeup"))
        d4, d5 = st.columns(2)
        attack = d4.number_input("Attack (ms)", 0.01, 2000.0, float(ms.duck_attack_ms), 5.0, key=_k(pid, "attack"))
        release = d5.number_input("Release (ms)", 0.01, 9000.0, float(ms.duck_release_ms), 50.0, key=_k(pid, "release"))

    with st.expander("Audio quality (all optional; nothing aggressive is on by default)"):
        q1, q2 = st.columns(2)
        hp_on = q1.checkbox("High-pass filter on voice", ms.highpass_enabled, key=_k(pid, "hp_on"))
        hp_hz = q2.number_input("High-pass cutoff (Hz)", 20, 400, int(ms.highpass_hz), 10, key=_k(pid, "hp_hz"))
        nr_on = q1.checkbox("Noise reduction on voice (FFmpeg afftdn)", ms.noise_reduction_enabled, key=_k(pid, "nr_on"))
        nr_db = q2.number_input("Noise reduction (dB)", 1.0, 60.0, float(ms.noise_reduction_db), 1.0, key=_k(pid, "nr_db"))
        ln_on = q1.checkbox("Loudness normalisation (two-pass EBU R128)", ms.loudnorm_enabled, key=_k(pid, "ln_on"))
        ln_i = q2.number_input("Target loudness (LUFS)", -30.0, -5.0, float(ms.loudnorm_target_lufs), 0.5,
                               key=_k(pid, "ln_i"))
        lim = q1.checkbox("Peak limiter (about −1 dBFS)", ms.limiter_enabled, key=_k(pid, "lim"))
        ln_tp = q2.number_input("True peak ceiling (dBTP)", -9.0, 0.0, float(ms.loudnorm_true_peak), 0.5,
                                key=_k(pid, "ln_tp"))
        fi = q1.number_input("Fade in (s)", 0.0, 30.0, float(ms.fade_in_sec), 0.5, key=_k(pid, "fi"))
        fo = q2.number_input("Fade out (s)", 0.0, 30.0, float(ms.fade_out_sec), 0.5, key=_k(pid, "fo"))

    values = {"audio.mode": mode, "audio.voice_source": voice_source, "audio.game_volume_db": float(game_db),
              "audio.voice_gain_db": float(voice_db), "audio.voice_offset_sec": float(offset),
              "audio.duck_threshold": float(thr), "audio.duck_ratio": float(ratio), "audio.duck_makeup": float(makeup),
              "audio.duck_attack_ms": float(attack), "audio.duck_release_ms": float(release),
              "audio.highpass_enabled": bool(hp_on), "audio.highpass_hz": int(hp_hz),
              "audio.noise_reduction_enabled": bool(nr_on), "audio.noise_reduction_db": float(nr_db),
              "audio.loudnorm_enabled": bool(ln_on), "audio.loudnorm_target_lufs": float(ln_i),
              "audio.loudnorm_true_peak": float(ln_tp), "audio.limiter_enabled": bool(lim),
              "audio.fade_in_sec": float(fi), "audio.fade_out_sec": float(fo)}
    try:
        candidate = audio_mix.MixSettings(**{k.split(".", 1)[1]: v for k, v in values.items()})
    except ValidationError as e:
        st.error(f"Invalid mix settings: {e.errors()[0]['msg']}")
        return
    dirty = candidate.model_dump() != ms.model_dump()
    if dirty:
        st.caption("⚠️ Unsaved changes. Previews below already use them.")

    b1, b2 = st.columns(2)
    if b1.button("Save mix settings", disabled=not dirty, key=_k(pid, "save_mix")):
        svc.save_project_settings(pid, values)
        st.rerun()
    if b2.button("Save & render mix", type="primary", disabled=busy or avail[mode] is not None,
                 key=_k(pid, "render_mix"),
                 help="Extracts tracks the first time. Later, only the mix is redone when only mix settings change."):
        svc.save_project_settings(pid, values)
        _start(svc, pid, "audio")

    st.subheader("Preview & compare")
    st.caption("Previews use the settings above, saved or not, without loudness normalisation, so absolute "
               "levels may differ slightly from the final mix. Your existing mix is not replaced by a preview.")
    extracted = bool(lc.track_files()) and all(p.exists() for p in lc.track_files().values())
    p1, p2 = st.columns(2)
    max_start = max(lc.duration - 1.0, 0.0)
    start = p1.number_input("Start at (seconds)", 0.0, max_start, min(30.0, max_start), 1.0, key=_k(pid, "pv_start"))
    secs = p2.number_input("Length (seconds)", 3, 120, int(lc.settings["audio"]["preview_seconds"]), 1,
                           key=_k(pid, "pv_len"))
    if not extracted:
        st.info("Run **Save & render mix** once to extract the audio tracks; previews need them.")
    if st.button("Render previews of all available modes", disabled=busy or not extracted, key=_k(pid, "pv_go")):
        with st.spinner("Rendering previews..."):
            for m in modes:
                st.session_state.pop(_k(pid, f"pv_{m}"), None)
                if audio_mix.available_modes(src)[m] is not None:
                    continue
                try:
                    path = svc.mix_preview(pid, candidate.model_copy(update={"mode": m}), float(start), float(secs))
                    st.session_state[_k(pid, f"pv_{m}")] = str(path)
                except _MEDIA_ERRORS as e:
                    st.error(f"{audio_mix.MODES[m]}: {e}")
    for m in modes:
        p = st.session_state.get(_k(pid, f"pv_{m}"))
        if p and Path(p).exists():
            st.markdown(f"**{audio_mix.MODES[m]}**")
            st.audio(p)

    st.subheader("Current full mix")
    out = lc.files.mix_wav(ms.mode)
    rep_path = lc.files.mix_report(ms.mode)
    if not out.exists() or not rep_path.exists():
        st.caption(f"No '{ms.mode}' mix rendered yet.")
        return
    rep = json.loads(rep_path.read_text("utf-8"))
    val = rep.get("validation") or {}
    stats = val.get("stats") or {}
    errs = val.get("errors") or []
    (st.error if errs else st.success)(
        f"'{ms.mode}' mix: {fmt_duration((val.get('probe') or {}).get('duration_sec'))} · "
        f"peak {stats.get('peak_db')} dBFS · RMS {stats.get('rms_db')} dBFS · "
        + ("validation passed" if not errs else "validation failed: " + "; ".join(errs)))
    loud = rep.get("loudness") or {}
    if loud.get("measured"):
        st.caption(f"Loudness before normalisation: {loud['measured'].get('input_i')} LUFS → target "
                   f"{loud.get('target_lufs')} LUFS" + (f" ({loud['skipped']})" if loud.get("skipped") else ""))
    st.caption(f"File: `{out}` ({fmt_size(out.stat().st_size)})")


# --- voiceover -----------------------------------------------------------------------------------
def _voiceover_tab(svc: StudioService, proj: dict[str, Any], lc: longform.LongformContext, busy: bool) -> None:
    pid = proj["id"]
    st.write("Add narration recorded after gameplay. Your original file is kept untouched; every upload or "
             "recording becomes a new version you can switch back to.")
    current = lc.voiceover_wav()
    ms = lc.mix_settings()
    if current:
        st.success(f"Active voiceover: **{current.name}** · offset {ms.voice_offset_sec:+.2f} s · "
                   f"gain {ms.voice_gain_db:+.1f} dB (change these on the Audio mix tab)")
        if current.stat().st_size <= _inline_limit_bytes(lc.settings):
            st.audio(str(current))
    else:
        st.info("No voiceover is active.")

    exts = sorted(e.lstrip(".") for e in audio_prep.VOICEOVER_EXTS)
    up = st.file_uploader("Upload a voiceover file", type=exts, key=_k(pid, "vo_upload"), disabled=busy)
    if up is not None and st.button("Use uploaded file as new voiceover", type="primary", disabled=busy,
                                    key=_k(pid, "vo_use_upload")):
        try:
            with st.spinner("Converting to 48 kHz WAV..."):
                svc.set_voiceover(pid, up, up.name)
            st.rerun()
        except _MEDIA_ERRORS as e:
            st.error(str(e))

    st.markdown("**Or record in the browser**")
    try:
        rec = st.audio_input("Record voiceover (browser microphone)", sample_rate=48000, key=_k(pid, "vo_rec"),
                             disabled=busy)
    except Exception as e:  # noqa: BLE001 - recording widget unsupported: the upload path still works
        rec = None
        st.info(f"Browser recording is unavailable here ({type(e).__name__}). Record in any app and upload the file.")
    if rec is not None and st.button("Use this recording as new voiceover", type="primary", disabled=busy,
                                     key=_k(pid, "vo_use_rec")):
        try:
            with st.spinner("Saving recording..."):
                svc.set_voiceover(pid, rec, "browser_recording.wav")
            st.rerun()
        except _MEDIA_ERRORS as e:
            st.error(str(e))

    versions = svc.voiceover_versions(pid)
    if versions:
        st.markdown("**Versions**")
        names = [p.name for p in versions]
        idx = names.index(current.name) if current and current.name in names else None
        choice = st.radio("Voiceover version", names, index=idx, key=_k(pid, "vo_version"), horizontal=True)
        c1, c2 = st.columns(2)
        same = current is not None and choice == current.name
        if c1.button("Use selected version", disabled=bool(busy or choice is None or same),
                     key=_k(pid, "vo_switch")):
            svc.use_voiceover_version(pid, versions[names.index(choice)])
            st.rerun()
        if c2.button("Stop using a voiceover", disabled=busy or current is None, key=_k(pid, "vo_clear"),
                     help="Files are kept; you can switch back to any version."):
            svc.clear_voiceover(pid)
            st.rerun()
    st.caption("After changing the voiceover, re-render the mix and re-transcribe. Only stages that depend on "
               "the voiceover run again; visual analysis and inspection are not repeated.")


# --- transcript ----------------------------------------------------------------------------------
def _transcript_tab(svc: StudioService, proj: dict[str, Any], lc: longform.LongformContext, busy: bool) -> None:
    pid = proj["id"]
    w = lc.settings["whisper"]
    src_keys = list(TRANSCRIPTION_SOURCES)
    cur_src = lc.settings["transcription"]["source"]
    t_src = st.selectbox("Transcribe from", src_keys, index=src_keys.index(cur_src) if cur_src in src_keys else 0,
                         format_func=TRANSCRIPTION_SOURCES.get, key=_k(pid, "t_src"))
    models = ["auto", *transcribe.MODEL_CHOICES]

    def model_label(m: str) -> str:
        if m == "auto":
            return "Auto (CPU default, or the largest that fits verified GPU memory)"
        info = transcribe.MODEL_INFO[m]
        return f"{m} · ~{info['download_mb']} MB · {info['note']}"

    c1, c2 = st.columns([3, 1])
    model = c1.selectbox("Whisper model", models, index=models.index(w["model"]) if w["model"] in models else 0,
                         format_func=model_label, key=_k(pid, "w_model"))
    devices = ["auto", "cpu", "cuda"]
    device = c2.selectbox("Device", devices, index=devices.index(w["device"]) if w["device"] in devices else 0,
                          key=_k(pid, "w_device"))
    status = svc.whisper_status(pid, {"model": model, "device": device})
    choice = status["choice"]
    st.caption(f"Effective: **{choice.effective}** on **{choice.device}** ({choice.compute_type}) · {choice.reason}")
    if choice.warning:
        st.warning(choice.warning)
    cached = status["cached"].get(choice.effective, False)
    allow = bool(w.get("allow_download"))
    if not cached:
        size = transcribe.MODEL_INFO.get(choice.effective, {}).get("download_mb", "?")
        st.warning(f"Model '{choice.effective}' is not downloaded yet (about {size} MB from Hugging Face).")
        allow = st.checkbox(f"Allow a one-time download of '{choice.effective}' (~{size} MB)", value=allow,
                            key=_k(pid, "w_allow"))
    lang = proj["language"]
    if lang == "hinglish":
        st.info("Hinglish: Whisper has no dedicated Hinglish mode. Mixed Hindi/English speech may come out in "
                "Devanagari, romanised, or a mix. Low-confidence words are flagged below for review.")

    values = {"transcription.source": t_src, "whisper.model": model, "whisper.device": device,
              "whisper.allow_download": bool(allow)}
    if st.button("Save & transcribe", type="primary", disabled=busy or (not cached and not allow),
                 key=_k(pid, "t_go")):
        svc.save_project_settings(pid, values)
        _start(svc, pid, "transcript")

    raw, edits = svc.transcript(pid)
    if not raw:
        st.caption("No transcript yet.")
        return
    if raw.get("skipped"):
        st.info(f"Transcription skipped: {raw['skipped']}.")
        return
    m = raw.get("model") or {}
    st.caption(f"Transcript from **{raw.get('source', '?')}** · model {m.get('effective')} on {m.get('device')}"
               f" · detected language {raw.get('language')} ({raw.get('language_probability')})"
               + (" · ⚠️ GPU failed, CPU fallback used" if m.get("fallbacks") else ""))
    words_meta = json.loads(lc.files.words_json.read_text("utf-8")) if lc.files.words_json.exists() else {}
    off = float(words_meta.get("offset_applied") or 0.0)
    if off:
        st.caption(f"Times below include the voiceover offset ({off:+.2f} s).")
    rows = [{"id": str(s["id"]), "start": _ts(s["start"] + off), "end": _ts(s["end"] + off),
             "text": edits.get(str(s["id"]), s["text"]),
             "check": ", ".join(x["word"].strip() for x in s.get("words", []) if x.get("uncertain")),
             "edited": str(s["id"]) in edits} for s in raw["segments"]]
    if not rows:
        st.info("No speech was found.")
        return
    edited = st.data_editor(
        pd.DataFrame(rows), key=_k(pid, "t_editor"), hide_index=True,
        disabled=["id", "start", "end", "check", "edited"],
        column_config={"id": None, "text": st.column_config.TextColumn("Text (editable)", width="large"),
                       "check": st.column_config.TextColumn("Low-confidence words"),
                       "edited": st.column_config.CheckboxColumn("Edited")})
    new_edits = {r["id"]: str(r["text"]) for r in edited.to_dict("records")}
    c1, c2 = st.columns(2)
    if c1.button("Save edits", key=_k(pid, "t_save")):
        svc.save_transcript_edits(pid, new_edits)
        st.rerun()
    if c2.button("Save edits & rebuild subtitles", type="primary", disabled=busy, key=_k(pid, "t_rebuild"),
                 help="Re-times captions from your text. Transcription itself is not repeated."):
        svc.save_transcript_edits(pid, new_edits)
        _start(svc, pid, "subtitles")
    d1, d2, d3 = st.columns(3)
    with d1:
        _download("transcript.txt", lc.files.transcript_txt, "text/plain", _k(pid, "dl_txt"))
    with d2:
        _download("subtitles.srt", lc.files.subtitles_srt, "application/x-subrip", _k(pid, "dl_srt"))
    with d3:
        _download("words.json", lc.files.words_json, "application/json", _k(pid, "dl_words"))


# --- cuts & rough cut ------------------------------------------------------------------------------
def _cuts_tab(svc: StudioService, proj: dict[str, Any], lc: longform.LongformContext, busy: bool) -> None:
    pid = proj["id"]
    s = lc.settings["silence"]
    st.info("Microphone silence alone **never** removes footage. A stretch is only a cut candidate when the "
            "picture is nearly static **and** game audio is quiet **and** nobody speaks; sudden game sounds "
            "inside it keep it. Quiet but important scenes can always be kept below.")
    pol_keys = list(CUT_POLICIES)
    policy = st.radio("Cut policy", pol_keys, index=pol_keys.index(s["policy"]) if s["policy"] in pol_keys else 0,
                      format_func=CUT_POLICIES.get, key=_k(pid, "policy"))
    with st.expander("Detection parameters"):
        c1, c2, c3 = st.columns(3)
        min_inactive = c1.number_input("Min inactive length (s)", 5.0, 600.0, float(s["min_inactive_sec"]), 1.0,
                                       key=_k(pid, "min_inactive"))
        pre = c2.number_input("Keep before (pre-roll, s)", 0.0, 30.0, float(s["preroll_sec"]), 0.5, key=_k(pid, "pre"))
        post = c3.number_input("Keep after (post-roll, s)", 0.0, 30.0, float(s["postroll_sec"]), 0.5, key=_k(pid, "post"))
        min_cut = c1.number_input("Min cut length (s)", 1.0, 600.0, float(s["min_cut_sec"]), 1.0, key=_k(pid, "min_cut"))
        motion = c2.number_input("Motion threshold (lower = stricter)", 0.1, 50.0, float(s["motion_threshold"]), 0.1,
                                 key=_k(pid, "motion"))
        quiet = c3.number_input("Quiet game audio below (dBFS)", -90.0, -10.0, float(s["game_audio_threshold_db"]), 1.0,
                                key=_k(pid, "quiet"))
        event = c1.number_input("Sudden-sound override (dB above local level)", 3.0, 40.0,
                                float(s["event_override_db"]), 1.0, key=_k(pid, "event"))
    values = {"silence.policy": policy, "silence.min_inactive_sec": float(min_inactive),
              "silence.preroll_sec": float(pre), "silence.postroll_sec": float(post),
              "silence.min_cut_sec": float(min_cut), "silence.motion_threshold": float(motion),
              "silence.game_audio_threshold_db": float(quiet), "silence.event_override_db": float(event)}
    if st.button("Save & analyse inactive scenes", type="primary", disabled=busy, key=_k(pid, "cuts_go")):
        svc.save_project_settings(pid, values)
        _start(svc, pid, "cuts")

    dec = svc.cut_decisions(pid)
    if dec:
        m1, m2, m3 = st.columns(3)
        m1.metric("Source", fmt_duration(dec["source_duration"]))
        m2.metric("Rough cut", fmt_duration(dec["output_duration"]))
        m3.metric("Removed", fmt_duration(dec["removed_sec"]) if dec["removed_sec"] >= 0.5 else "nothing")
        cands = [x for x in dec["segments"] if x["kind"] == "candidate"]
        if not cands:
            st.success("No inactive stretches found. Nothing will be removed.")
        else:
            st.markdown(f"**{len(cands)} inactive stretch(es)** (policy at analysis: {dec['policy']}). "
                        "Tick **Keep** to leave a stretch in the video.")
            df = pd.DataFrame([{"id": c["id"], "from": _ts(c["source_start"]), "to": _ts(c["source_end"]),
                                "length": fmt_duration(c["source_end"] - c["source_start"]), "keep": bool(c["keep"]),
                                "status": c["status"], "why": c["reason"]} for c in cands])
            edited = st.data_editor(df, key=_k(pid, "cut_editor"), hide_index=True,
                                    disabled=["id", "from", "to", "length", "status", "why"],
                                    column_config={"id": None, "keep": st.column_config.CheckboxColumn("Keep")})
            default_keep = dec["policy"] != "conservative"
            overrides = {r["id"]: bool(r["keep"]) for r in edited.to_dict("records") if bool(r["keep"]) != default_keep}
            if st.button("Save keep/cut choices & re-analyse", disabled=busy, key=_k(pid, "cut_save")):
                svc.save_cut_overrides(pid, overrides)
                svc.save_project_settings(pid, values)
                _start(svc, pid, "cuts")
        with st.expander("Full cut-decision list (source and output times)"):
            st.dataframe(pd.DataFrame([{"kind": x["kind"], "keep": x["keep"], "source": f"{_ts(x['source_start'])}–"
                                        f"{_ts(x['source_end'])}", "output": "removed" if x["output_start"] is None else
                                        f"{_ts(x['output_start'])}–{_ts(x['output_end'])}", "reason": x["reason"]}
                                       for x in dec["segments"]]), hide_index=True)
            _download("cut_decisions.json", lc.files.cut_decisions, "application/json", _k(pid, "dl_cuts"))
    else:
        st.caption("No cut analysis yet. Rendering runs it automatically with the policy above.")

    st.subheader("Rough cut render")
    r = lc.settings["render"]
    presets = list(timeline_builder.QUALITY_PRESETS)
    c1, c2 = st.columns(2)
    quality = c1.selectbox("Quality", presets, index=presets.index(r["quality_preset"]) if r["quality_preset"] in presets
                           else 0, format_func=lambda p: timeline_builder.QUALITY_PRESETS[p]["label"],
                           key=_k(pid, "quality"))
    enc_keys = list(ENCODERS)
    enc = c2.selectbox("Video encoder", enc_keys, index=enc_keys.index(r["encoder"]) if r["encoder"] in enc_keys else 0,
                       format_func=ENCODERS.get, key=_k(pid, "encoder"))
    hw = svc.hardware_report()
    eff_enc, why = timeline_builder.choose_encoder(enc, bool(hw and hw.nvenc_encode_ok))
    st.caption(f"Will encode with **{eff_enc}** ({why}). Source footage is never modified; a previous render is "
               "moved to `_previous/`, not overwritten.")
    t = lc.settings["timeline"]
    t1, t2 = st.columns(2)
    edl = t1.checkbox("Export CMX3600 EDL", bool(t["export_edl"]), key=_k(pid, "edl"))
    xml = t2.checkbox("Export FCP7 XML", bool(t["export_fcp7_xml"]), key=_k(pid, "xml"))
    st.caption("Timeline files pass structural self-checks but have **not yet been verified to import into "
               "DaVinci Resolve**. The MP4 rough cut is the reliable output.")
    if st.button("Save & render rough cut", type="primary", disabled=busy, key=_k(pid, "render_go")):
        svc.save_project_settings(pid, {**values, "render.quality_preset": quality, "render.encoder": enc,
                                        "timeline.export_edl": bool(edl), "timeline.export_fcp7_xml": bool(xml)})
        _start(svc, pid, "render")

    outs = [o for o in svc.rendered_outputs(pid) if o["output_type"] == "long_rough_cut"]
    if not outs:
        st.caption("No rough cut rendered yet.")
        return
    last = outs[0]
    errs = (last["validation_details"] or {}).get("errors") or []
    rc = lc.files.rough_cut
    set_ = last["settings"] or {}
    (st.success if last["validation_status"] == "passed" else st.error)(
        f"Rough cut {last['validation_status']} validation · encoder {set_.get('encoder')} · mix "
        f"{set_.get('mix_mode')} · {last['created_at']}" + ("" if not errs else " · " + "; ".join(errs)))
    if rc.exists():
        if rc.stat().st_size <= _inline_limit_bytes(lc.settings):
            st.video(str(rc))
        else:
            st.caption(f"{fmt_size(rc.stat().st_size)}: too large to preview inline. Open the folder to play it.")
    if st.button("Open output folder", key=_k(pid, "open_ready")):
        try:
            svc.open_in_file_manager(lc.files.ready)
        except (ValueError, RuntimeError) as e:
            st.error(str(e))
    d1, d2, d3 = st.columns(3)
    with d1:
        _download("timeline.edl", lc.files.edl, "text/plain", _k(pid, "dl_edl"))
    with d2:
        _download("timeline (FCP7 XML)", lc.files.fcp7_xml, "application/xml", _k(pid, "dl_xml"))
    with d3:
        _download("timeline.json", lc.files.timeline_json, "application/json", _k(pid, "dl_tl"))
    if len(outs) > 1:
        st.caption(f"{len(outs) - 1} earlier render(s) are kept in `{lc.files.ready / '_previous'}`.")


def render() -> None:
    svc = get_service()
    proj = require_project(svc)
    if proj is None:
        return
    st.title("Long video & audio studio")
    reason = svc.longform_ready(proj["id"])
    if reason:
        st.info(reason)
        return
    _status(proj["id"])
    lc = svc.longform_context(proj["id"])
    busy = _busy(svc, proj["id"])
    if busy:
        st.caption("A job is running; actions that change files are disabled until it finishes.")
    tabs = st.tabs(["🎚️ Audio mix", "🎙️ Voiceover", "📝 Transcript & captions", "✂️ Cuts & rough cut"])
    with tabs[0]:
        _mix_tab(svc, proj, lc, busy)
    with tabs[1]:
        _voiceover_tab(svc, proj, lc, busy)
    with tabs[2]:
        _transcript_tab(svc, proj, lc, busy)
    with tabs[3]:
        _cuts_tab(svc, proj, lc, busy)
