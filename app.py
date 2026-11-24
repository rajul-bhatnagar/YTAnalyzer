"""YT Creator Studio: Streamlit entry point. Run via launch_studio.bat (or `streamlit run app.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from pipeline import __version__  # noqa: E402
from ui import (audio_studio, dashboard, final_review, highlight_review, metadata_review, processing_view,  # noqa: E402
                project_setup, settings_view, shorts_review, thumbnail_review, track_mapping)
from ui.common import (get_service, hardware_badge, register_pages, sidebar_project_picker,  # noqa: E402
                       sidebar_quick_settings)

st.set_page_config(page_title="YT Creator Studio", page_icon="🎮", layout="wide")

try:
    svc = get_service()
except Exception as e:  # e.g. database newer than code, unwritable folder
    st.error(f"The studio could not start: {e}")
    st.stop()

with st.sidebar:
    st.markdown(f"### 🎮 YT Creator Studio\n<small>v{__version__} · local-first</small>", unsafe_allow_html=True)
    hardware_badge(svc)
    sidebar_project_picker(svc)
    sidebar_quick_settings(svc)

pages = {
    "Studio": [
        st.Page(dashboard.render, title="Dashboard", icon="🏠", url_path="dashboard", default=True),
        st.Page(project_setup.render, title="Import / New project", icon="📥", url_path="import"),
    ],
    "Current project": [
        st.Page(processing_view.render, title="Processing & logs", icon="⚙️", url_path="processing"),
        st.Page(track_mapping.render, title="Audio track mapping", icon="🎚️", url_path="tracks"),
        st.Page(audio_studio.render, title="Long video & audio", icon="🎬", url_path="longform"),
        st.Page(highlight_review.render, title="Analysis & highlights", icon="🎯", url_path="highlights"),
        st.Page(shorts_review.render, title="Shorts", icon="📱", url_path="shorts"),
        st.Page(thumbnail_review.render, title="Thumbnails", icon="🖼️", url_path="thumbnails"),
        st.Page(metadata_review.render, title="Metadata & scripts", icon="📝", url_path="metadata"),
        st.Page(final_review.render, title="Final approval & upload", icon="✅", url_path="final"),
    ],
    "System": [
        st.Page(settings_view.render, title="Settings & diagnostics", icon="🛠️", url_path="settings"),
    ],
}
register_pages({p.url_path: p for group in pages.values() for p in group})
st.navigation(pages).run()
