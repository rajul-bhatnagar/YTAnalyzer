"""Whole-video voiceover quick loop, shown on the Shorts and Metadata pages: upload or record a voiceover, set its
offset, then re-mix, re-transcribe, re-render the approved Shorts and update the package in one job. Import,
inspection, track mapping, analysis and highlights are not repeated."""

from __future__ import annotations

import streamlit as st

from pipeline import audio_prep, ffmpeg_tools
from pipeline.project_service import StudioService


def render(svc: StudioService, pid: str, key: str, busy: bool) -> None:
    with st.expander("🎙️ Voiceover for the whole video (quick loop)"):
        lc = svc.longform_context(pid)
        cur = lc.voiceover_wav()
        st.caption("Record or upload narration, then apply it: the game sound is ducked under your voice, loudness is "
                   "normalised, the voiceover is transcribed for captions, approved Shorts are re-rendered and the "
                   "package is updated. Analysis and highlights are kept as they are.")
        if cur:
            st.success(f"Active voiceover: {cur.name}")
        exts = sorted(e.lstrip(".") for e in audio_prep.VOICEOVER_EXTS)
        up = st.file_uploader("Upload a voiceover", type=exts, key=f"{key}_vo_up", disabled=busy)
        try:
            rec = st.audio_input("Or record it here", sample_rate=48000, key=f"{key}_vo_rec", disabled=busy)
        except Exception:  # noqa: BLE001 - recording unsupported: the upload still works
            rec = None
        src = up or rec
        if src is not None and st.button("Use this voiceover", key=f"{key}_vo_use", disabled=busy):
            try:
                with st.spinner("Converting to 48 kHz WAV..."):
                    svc.set_voiceover(pid, src, getattr(src, "name", "") or "recording.wav")
                st.rerun()
            except (ValueError, ffmpeg_tools.FFmpegError) as e:
                st.error(str(e))
        offset = st.number_input("Voiceover starts at (seconds into the recording)", 0.0, 36000.0,
                                 float(lc.mix_settings().voice_offset_sec), 0.5, key=f"{key}_vo_off")
        if st.button("Apply voiceover: re-mix, re-transcribe, re-render Shorts, update package", type="primary",
                     key=f"{key}_vo_apply", disabled=busy or cur is None):
            try:
                svc.apply_voiceover(pid, offset)
                st.rerun()
            except ValueError as e:
                st.error(str(e))
