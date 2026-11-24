"""Project overview, job/stage status, retry/pause/cancel, logs."""

from __future__ import annotations

import streamlit as st

from pipeline import analysis_stages, content_stages, longform
from pipeline.media_probe import single_track_notice
from pipeline.models import CONTENT_STYLES, LANGUAGES
from pipeline.project_service import stage_error_text
from ui.common import fmt_duration, fmt_size, get_service, operation_label, require_project

STAGE_LABELS = {
    "probe": "Inspect streams & validate source",
    "checksum": "Source checksum (read-only)",
    "track_levels": "Audio track loudness",
    **longform.STAGE_LABELS,
    **analysis_stages.STAGE_LABELS,
    **content_stages.STAGE_LABELS,
}
STATUS_ICON = {"COMPLETED": "✅", "RUNNING": "⏳", "FAILED": "❌", "CANCELLED": "⛔", "PENDING": "•", "SKIPPED": "↷"}


def _overview(svc, proj) -> None:
    info = svc.media_info(proj["id"])
    st.subheader("Source recording")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", fmt_duration(proj["duration_sec"]))
    c2.metric("Size", fmt_size(proj["source_size"]))
    if info and info.video:
        res = info.display_resolution
        c3.metric("Resolution", f"{res[0]}×{res[1]}" if res else "-")
        c4.metric("Frame rate", f"{info.fps:.3f}" if info.fps else "-")
    st.caption(f"Path: `{proj['source_path']}` (never modified)")
    if info:
        n = len(info.audio)
        st.write(f"**Video:** {info.video.codec if info.video else '-'} · {info.video.pix_fmt if info.video else ''}"
                 f" · **Audio tracks:** {n} · **VFR:** "
                 f"{'yes (measured)' if info.vfr_measured else 'no' if info.vfr_measured is False else 'unknown'}")
        notice = single_track_notice(n)
        if notice:
            st.info(notice)
        for w in info.warnings:
            st.warning(w)

    with st.expander("Edit project details"):
        game = st.text_input("Game name", value=proj["game_name"])
        style = st.selectbox("Content style", CONTENT_STYLES, index=CONTENT_STYLES.index(proj["content_style"]))
        lang = st.selectbox("Language", list(LANGUAGES), index=list(LANGUAGES).index(proj["language"]),
                            format_func=lambda k: LANGUAGES[k])
        note = st.text_input("Topic note", value=proj["topic_note"], max_chars=300)
        if st.button("Save details"):
            try:
                svc.update_project_details(proj["id"], game, style, lang, note)
                st.success("Saved.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))


@st.fragment(run_every=2)
def _job_panel(project_id: str) -> None:
    svc = get_service()
    job = svc.latest_job(project_id)
    st.subheader("Processing")
    if job is None:
        st.write("No processing job yet.")
        if st.button("Start inspection", type="primary"):
            svc.start_inspection(project_id)
            st.rerun()
        return
    running = svc.is_job_running(job["id"])
    state = job["state"]
    # When a background run finishes, refresh the whole page once so the overview shows new results.
    flag = f"was_running_{job['id']}"
    if st.session_state.get(flag) and not running:
        st.session_state[flag] = False
        st.rerun(scope="app")
    st.session_state[flag] = running
    st.write(f"**Job** `{job['id']}` · state **{state}**{' · running' if running else ''}")
    st.progress(float(job["progress"] or 0), text=job["progress_note"] or "")
    if job["error_summary"]:
        st.error(job["error_summary"])

    c1, c2, c3 = st.columns(3)
    if running:
        if c1.button("Pause after current stage"):
            svc.pause_job(job["id"])
        if c2.button("Cancel now"):
            svc.cancel_job(job["id"])
    else:
        op = svc.last_operation(project_id)
        op_label = operation_label(op).lower()
        label = {"FAILED": f"Retry: {op_label}", "PAUSED": f"Resume: {op_label}",
                 "CANCELLED": f"Start again: {op_label}"}.get(state, "Re-run inspection")
        if c1.button(label, type="primary" if state in ("FAILED", "PAUSED") else "secondary",
                     help="Completed stages with unchanged inputs are reused, not repeated."):
            try:
                if state in ("FAILED", "PAUSED", "CANCELLED"):
                    svc.retry_job(project_id)
                else:
                    svc.start_inspection(project_id)
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        if state not in ("CANCELLED", "DRAFT_UPLOADED") and c2.button("Cancel job"):
            svc.cancel_job(job["id"])
            st.rerun()

    for s in svc.job_stages(job["id"]):
        icon = STATUS_ICON.get(s["status"], "•")
        dur = f" · {s['duration_sec']:.1f}s" if s["duration_sec"] else ""
        opt = " (optional)" if s["optional"] else ""
        with st.expander(f"{icon} {STAGE_LABELS.get(s['stage_name']) or content_stages.stage_label(s['stage_name']) or s['stage_name']}{opt} · "
                         f"{s['status']}{dur} · attempts {s['attempt_count']}",
                         expanded=s["status"] == "FAILED"):
            err = stage_error_text(s)
            if err:
                st.error(err)
                rec = (s["error_details"] or {}).get("recovery")
                if rec:
                    st.caption(rec)
            for w in s["warnings"]:
                st.warning(w)
            if s["output_manifest"]:
                st.json(s["output_manifest"], expanded=False)


def render() -> None:
    svc = get_service()
    proj = require_project(svc)
    if proj is None:
        return
    st.title(proj["name"])
    st.caption(f"{proj['game_name']} · {proj['content_style']} · {LANGUAGES.get(proj['language'])}"
               + (f" · “{proj['topic_note']}”" if proj["topic_note"] else ""))
    _job_panel(proj["id"])
    _overview(svc, proj)
    if proj["status"] == "inspected":
        st.info("Next step: confirm audio track roles on the **Audio track mapping** page.")
    with st.expander("Application log (last 200 lines)"):
        st.code(svc.log_tail(200) or "(empty)", language="log")
