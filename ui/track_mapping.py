"""Audio track mapping: preview each detected stream and confirm its role."""

from __future__ import annotations

import streamlit as st

from pipeline import ffmpeg_tools
from pipeline.media_probe import single_track_notice, validate_role_mapping
from pipeline.models import TRACK_ROLE_LABELS, TrackRole
from ui.common import get_service, require_project


def render() -> None:
    svc = get_service()
    proj = require_project(svc)
    if proj is None:
        return
    st.title("Audio track mapping")
    st.write("The app never assumes which OBS track is which. Listen to each stream and assign its role. "
             "Roles drive the game / voice / auto-duck mixes in the next phase.")
    if proj["status"] == "imported":
        st.info("Inspection has not finished yet. See **Processing & logs**.")
        return
    tracks = svc.audio_tracks(proj["id"])
    notice = single_track_notice(len(tracks))
    if notice:
        st.info(notice)
    if not tracks:
        if not proj["tracks_confirmed"] and st.button("Confirm: no audio tracks", type="primary"):
            svc.save_track_roles(proj["id"], {})
            st.rerun()
        if proj["tracks_confirmed"]:
            st.success("Confirmed: recording has no audio.")
        return

    if proj["tracks_confirmed"]:
        st.success("Track roles are confirmed. You can change them below at any time.")
    else:
        st.warning("Suggested roles are shown below and are **not confirmed**. Check them before continuing.")

    roles_all = list(TrackRole)
    chosen: dict[int, TrackRole] = {}
    for t in tracks:
        idx = t["audio_index"]
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 3, 2])
            vol = f"{t['mean_volume_db']:.1f} dB" if t["mean_volume_db"] is not None else "silent / unknown"
            c1.markdown(
                f"**Audio track {idx + 1}** (stream #{t['stream_index']})  \n"
                f"{t['codec'] or '?'} · {t['channels'] or '?'} ch ({t['channel_layout'] or '-'}) · "
                f"{t['sample_rate'] or '?'} Hz  \n"
                f"Title: {t['title'] or '-'} · Language: {t['language'] or '-'}  \n"
                f"Mean level (first 2 min): {vol}")
            current = TrackRole(t["user_role"] or t["detected_role"])
            chosen[idx] = c2.selectbox(
                "Role", roles_all, index=roles_all.index(current), key=f"role_{proj['id']}_{idx}",
                format_func=lambda r: TRACK_ROLE_LABELS[r])
            c2.caption(f"Suggested: {TRACK_ROLE_LABELS[TrackRole(t['detected_role'])]}")
            if c3.button("Preview 20 s", key=f"prev_{proj['id']}_{idx}"):
                try:
                    with st.spinner("Extracting preview..."):
                        st.session_state[f"preview_{proj['id']}_{idx}"] = str(svc.track_preview(proj["id"], idx))
                except (ffmpeg_tools.FFmpegError, ffmpeg_tools.FFmpegNotFound) as e:
                    c3.error(str(e))
            p = st.session_state.get(f"preview_{proj['id']}_{idx}")
            if p:
                st.audio(p)

    errors = validate_role_mapping(chosen, len(tracks))
    for e in errors:
        st.error(e)
    if st.button("Confirm track roles", type="primary", disabled=bool(errors)):
        try:
            svc.save_track_roles(proj["id"], chosen)
            st.success("Saved.")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
