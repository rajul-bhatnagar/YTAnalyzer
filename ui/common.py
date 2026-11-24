"""Shared UI helpers: service singleton, project selection, formatting."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from pipeline import analysis_stages, content_stages, longform
from pipeline.hardware import MODE_BADGES
from pipeline.paths import StudioPaths
from pipeline.project_service import StudioService, stage_error_text

PROJECT_KEY = "selected_project_id"


@st.cache_resource(show_spinner="Starting studio services...")
def get_service() -> StudioService:
    # One service (and one background worker) per Streamlit server process.
    # STUDIO_ROOT lets automated UI tests use an isolated folder; normal use leaves it unset.
    override = os.environ.get("STUDIO_ROOT", "").strip()
    return StudioService(StudioPaths(Path(override)) if override else None)


def selected_project(svc: StudioService) -> dict[str, Any] | None:
    pid = st.session_state.get(PROJECT_KEY)
    return svc.get_project(pid) if pid else None


def select_project(pid: str | None) -> None:
    st.session_state[PROJECT_KEY] = pid


PAGES_KEY = "_studio_pages"


def register_pages(pages: dict[str, Any]) -> None:
    st.session_state[PAGES_KEY] = pages


def goto(url_path: str) -> None:
    page = st.session_state.get(PAGES_KEY, {}).get(url_path)
    if page is None:
        st.rerun()
    st.switch_page(page)


def require_project(svc: StudioService) -> dict[str, Any] | None:
    proj = selected_project(svc)
    if proj is None:
        st.info("Select a project in the sidebar, or create one on the **Import** page.")
    return proj


def fmt_duration(sec: float | None) -> str:
    if not sec:
        return "-"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_size(n: int | None) -> str:
    if not n:
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore[assignment]
    return "-"


def hardware_badge(svc: StudioService) -> None:
    report = svc.hardware_report()
    if report is None:
        st.badge("Hardware: not checked", color="gray")
        st.caption("Open Settings & Diagnostics to run the capability check.")
        return
    label, color = MODE_BADGES.get(report.mode, (report.mode, "gray"))
    st.badge(f"Hardware: {label}", color=color)  # type: ignore[arg-type]


def operation_label(op: str) -> str:
    if op == "inspection":
        return "Inspection"
    return (longform.TARGET_LABELS.get(op) or analysis_stages.TARGET_LABELS.get(op)
            or content_stages.TARGET_LABELS.get(op, op))


@st.fragment(run_every=2)
def job_status_panel(project_id: str, stage_labels: dict[str, str], key_prefix: str,
                     include_content: bool = False) -> None:
    """Live status of the project's latest job: progress, cancel, retry/resume, and failures/warnings of the
    stages named in ``stage_labels``. Refreshes the whole page once when a run finishes."""
    svc = get_service()
    disk = svc.disk_status()
    if disk["message"]:
        (st.error if disk["level"] == "block" else st.warning)(disk["message"] + " (Settings → Disk & temporary files)")
    job = svc.latest_job(project_id)
    if job is None:
        st.caption("No processing has run for this project yet.")
        return
    running = svc.is_job_running(job["id"])
    flag = f"{key_prefix}_{project_id}_was_running_{job['id']}"
    if st.session_state.get(flag) and not running:
        st.session_state[flag] = False
        st.rerun(scope="app")
    st.session_state[flag] = running
    op = svc.last_operation(project_id)
    with st.container(border=True):
        c1, c2 = st.columns([5, 2])
        c1.markdown(f"**{operation_label(op)}** · job state **{job['state']}**{' · running' if running else ''}")
        c1.progress(float(job["progress"] or 0), text=job["progress_note"] or "")
        if running:
            if c2.button("Cancel", key=f"{key_prefix}_{project_id}_cancel"):
                svc.cancel_job(job["id"])
            if c2.button("Pause", key=f"{key_prefix}_{project_id}_pause",
                         help="Stops at the next safe point; finished work is kept."):
                svc.pause_job(job["id"])
        elif job["state"] in ("FAILED", "PAUSED", "CANCELLED") and op != "inspection":
            if c2.button("Retry" if job["state"] == "FAILED" else "Resume", type="primary",
                         key=f"{key_prefix}_{project_id}_retry", help="Completed work with unchanged inputs is reused."):
                try:
                    svc.retry_job(project_id)
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        if job["error_summary"] and not running:
            st.error(job["error_summary"])
        for s in svc.job_stages(job["id"]):
            label = stage_labels.get(s["stage_name"]) or (content_stages.stage_label(s["stage_name"])
                                                          if include_content else None)
            if label is None:
                continue
            if s["status"] == "FAILED":
                st.error(f"{label}: {stage_error_text(s) or 'failed'}")
            for w in s["warnings"] or []:
                st.warning(f"{label}: {w}")
        st.caption("Full stage details and logs are on **Processing & logs**.")


def sidebar_quick_settings(svc: StudioService) -> None:
    """The most-used per-project choices (spec: sidebar). Full controls stay on each page; both edit the same
    settings. Takes effect on the next run of the affected step."""
    from pipeline import transcribe  # noqa: PLC0415
    from pipeline.analysis_service import PROVIDERS  # noqa: PLC0415
    from pipeline.audio_mix import MODES  # noqa: PLC0415
    from pipeline.models import CONTENT_STYLES, LANGUAGES  # noqa: PLC0415
    from pipeline.shorts_builder import LAYOUTS  # noqa: PLC0415

    proj = selected_project(svc)
    if proj is None:
        return
    s = svc.effective_settings(proj["id"])
    with st.expander("Quick settings"), st.form(f"quick_{proj['id']}"):
        game = st.text_input("Game", proj["game_name"])
        style = st.selectbox("Content style", CONTENT_STYLES, index=CONTENT_STYLES.index(proj["content_style"]))
        langs = list(LANGUAGES)
        lang = st.selectbox("Language", langs, index=langs.index(proj["language"]), format_func=LANGUAGES.get)
        note = st.text_input("Topic note", proj["topic_note"], max_chars=300)
        models = ["auto", *transcribe.MODEL_CHOICES]
        wm = st.selectbox("Whisper model", models, index=models.index(s["whisper"]["model"])
                          if s["whisper"]["model"] in models else 0)
        provs = list(PROVIDERS)
        prov = st.selectbox("AI provider", provs, index=provs.index(s["ai"]["provider"]) if s["ai"]["provider"] in provs
                            else 0, format_func=PROVIDERS.get, help="Gemini also needs consent on Analysis & highlights.")
        modes = list(MODES)
        mode = st.selectbox("Audio mode", modes, index=modes.index(s["audio"]["mode"]) if s["audio"]["mode"] in modes
                            else 0, format_func=MODES.get)
        gv = st.slider("Game volume (dB)", -40.0, 12.0, float(s["audio"]["game_volume_db"]), 0.5)
        silence = st.selectbox("Inactive-scene cutting", ["none", "conservative"],
                               index=["none", "conservative"].index(s["silence"]["policy"]))
        lays = list(LAYOUTS)
        lay = st.selectbox("Default Shorts layout", lays, index=lays.index(s["shorts"]["layout"])
                           if s["shorts"]["layout"] in lays else 0, format_func=LAYOUTS.get)
        cap = st.checkbox("Captions on new Shorts", bool(s["shorts"]["captions_enabled"]))
        if st.form_submit_button("Save"):
            try:
                svc.update_project_details(proj["id"], game, style, lang, note)
                svc.save_project_settings(proj["id"], {
                    "whisper.model": wm, "ai.provider": prov, "audio.mode": mode, "audio.game_volume_db": float(gv),
                    "silence.policy": silence, "shorts.layout": lay, "shorts.captions_enabled": bool(cap)})
                st.success("Saved. Applies to the next run of each step.")
            except ValueError as e:
                st.error(str(e))


def sidebar_project_picker(svc: StudioService) -> None:
    projects = svc.list_projects()
    if not projects:
        st.caption("No projects yet.")
        return
    ids = [p["id"] for p in projects]
    labels = {p["id"]: f"{p['name']}  ({p['game_name']})" for p in projects}
    current = st.session_state.get(PROJECT_KEY)
    index = ids.index(current) if current in ids else 0
    choice = st.selectbox("Project", ids, index=index, format_func=lambda i: labels[i])
    select_project(choice)
