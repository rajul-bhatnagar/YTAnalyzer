"""Dashboard: project history and system status."""

from __future__ import annotations

import streamlit as st

from pipeline import config
from ui.common import fmt_duration, fmt_size, get_service, goto, select_project

STATUS_LABELS = {
    "imported": "Imported, not inspected",
    "inspected": "Inspected, confirm audio tracks",
    "tracks_confirmed": "Tracks confirmed",
}


def render() -> None:
    svc = get_service()
    st.title("Dashboard")

    hw = svc.hardware_report()
    secrets = config.secret_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("Hardware mode", hw.mode if hw else "Not checked")
    c2.metric("Gemini key", "Configured" if secrets["GEMINI_API_KEY"] else "Not set")
    c3.metric("Projects", len(svc.list_projects()))
    if hw is None:
        st.warning("Hardware capabilities have not been checked on this PC yet. Open **Settings & diagnostics**.")
    if svc.recovered:
        st.warning(f"{svc.recovered} stage(s) were interrupted when the app last stopped. "
                   "Open **Processing & logs** for the affected project and press Retry.")

    st.subheader("Projects")
    projects = svc.list_projects()
    if not projects:
        st.info("No projects yet. Put recordings in `01_RAW_INPUT` (optionally inside a folder named after the "
                "game) or upload one on the **Import / New project** page.")
        return
    for p in projects:
        job = svc.latest_job(p["id"])
        with st.container(border=True):
            a, b, c, d = st.columns([4, 2, 2, 1])
            a.markdown(f"**{p['name']}**  \n{p['game_name']} · {p['content_style']} · {p['language']}")
            b.write(f"Duration: {fmt_duration(p['duration_sec'])}  \nSize: {fmt_size(p['source_size'])}")
            state = job["state"] if job else "no job"
            c.write(f"{STATUS_LABELS.get(p['status'], p['status'])}  \nJob: {state}")
            if d.button("Open", key=f"open_{p['id']}"):
                select_project(p["id"])
                goto("processing")
