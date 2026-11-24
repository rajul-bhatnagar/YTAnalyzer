"""Headless UI tests with Streamlit's AppTest: every page renders without exceptions on real data."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pipeline.job_manager import JobRunner
from pipeline.models import TrackRole
from pipeline.project_service import StudioService
from tests.conftest import REPO, requires_ffmpeg

pytestmark = requires_ffmpeg


@pytest.fixture
def ui_root(studio_paths, media, monkeypatch):
    """Studio root with one inspected 3-track project, used through STUDIO_ROOT."""
    monkeypatch.setenv("STUDIO_ROOT", str(studio_paths.root))
    from ui.common import get_service
    get_service.clear()  # the cached service from a previous test points at another temp root
    svc =StudioService(studio_paths, runner=JobRunner(), configure_logging=False)
    src = studio_paths.raw_input / "Test Game" / "rec ü.mp4"
    src.parent.mkdir(parents=True)
    shutil.copy(media["three"], src)
    pid = svc.create_project(src, "Test Game", "Custom", "en")
    svc.start_inspection(pid)
    assert svc.runner.wait_idle(120)
    svc.refresh_hardware()
    return studio_paths, pid


@pytest.mark.parametrize("page", ["ui.dashboard", "ui.project_setup", "ui.processing_view",
                                  "ui.track_mapping", "ui.settings_view", "ui.audio_studio",
                                  "ui.highlight_review", "ui.shorts_review", "ui.thumbnail_review",
                                  "ui.metadata_review", "ui.final_review"])
def test_page_renders(ui_root, page):
    _paths, pid = ui_root
    at = AppTest.from_string(
        f"import sys\nsys.path.insert(0, {str(REPO)!r})\n"
        "import streamlit as st\n"
        f"st.session_state.setdefault('selected_project_id', {pid!r})\n"
        f"import importlib\nimportlib.import_module({page!r}).render()\n",
        default_timeout=60,
    )
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, "page rendered no title"


def test_full_app_default_page(ui_root):
    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Dashboard" in t.value for t in at.title)


def test_track_mapping_confirm_via_ui(ui_root):
    paths, pid = ui_root
    script = (f"import sys\nsys.path.insert(0, {str(REPO)!r})\n"
              "import streamlit as st\n"
              f"st.session_state.setdefault('selected_project_id', {pid!r})\n"
              "from ui import track_mapping\ntrack_mapping.render()\n")
    at = AppTest.from_string(script, default_timeout=60)
    at.run()
    assert any("not confirmed" in w.value for w in at.warning)
    # Make two tracks 'mic' -> validation error and disabled confirm button.
    role_boxes = [s for s in at.selectbox if s.label == "Role"]
    assert len(role_boxes) == 3
    role_boxes[0].set_value(role_boxes[2].value).run()
    assert any("Only one track" in e.value for e in at.error)
    # Fix it and confirm.
    role_boxes = [s for s in at.selectbox if s.label == "Role"]
    role_boxes[0].set_value(TrackRole.COMBINED_MIX).run()
    confirm = [b for b in at.button if b.label == "Confirm track roles"][0]
    confirm.click().run()
    assert not at.exception
    svc = StudioService(paths, runner=JobRunner(), configure_logging=False)
    assert svc.get_project(pid)["tracks_confirmed"] == 1
