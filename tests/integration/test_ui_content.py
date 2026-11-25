"""Shorts, Thumbnails and Metadata pages driven headlessly with AppTest on a project with real outputs."""

from __future__ import annotations

import shutil

import pytest
from streamlit.testing.v1 import AppTest

from pipeline import transcribe
from pipeline.job_manager import JobRunner
from pipeline.models import TrackRole
from pipeline.project_service import StudioService
from tests.conftest import REPO, requires_ffmpeg
from ui.common import get_service  # only to clear the cached service between tests

pytestmark = [requires_ffmpeg,
              pytest.mark.skipif(not transcribe.is_model_cached("tiny"), reason="Whisper 'tiny' model not downloaded")]


@pytest.fixture(scope="module")
def ready(tmp_path_factory, media):
    from pipeline.paths import StudioPaths
    root = tmp_path_factory.mktemp("ui content") / "studio ü"
    paths = StudioPaths(root)
    paths.ensure()
    shutil.copy(REPO / "config" / "default_settings.json", paths.default_settings_file)
    svc = StudioService(paths, runner=JobRunner(), configure_logging=False)
    src = paths.raw_input / "Game X" / "rec.mp4"
    src.parent.mkdir(parents=True)
    shutil.copy(media["gameplay"], src)
    pid = svc.create_project(src, "Game X", "Custom", "en")
    svc.start_inspection(pid)
    assert svc.runner.wait_idle(120)
    svc.save_track_roles(pid, {0: TrackRole.COMBINED_MIX, 1: TrackRole.GAME, 2: TrackRole.MIC})
    svc.save_project_settings(pid, {"whisper.model": "tiny", "analysis.window_sec": 45, "analysis.window_overlap_sec": 0, "ai.provider": "none",
                                    "ai.ollama_base_url": "http://127.0.0.1:9", "shorts.quality": "fast"})
    svc.start_analysis(pid)
    assert svc.runner.wait_idle(600)
    hid = svc.highlights(pid)[0]["id"]
    svc.update_highlight(hid, status="approved", hook="What a shot", title="Big moment")
    for target in ("shorts", "assets"):
        svc.start_content(pid, target)
        assert svc.runner.wait_idle(900)
    assert svc.short_output(hid)["validation_status"] == "passed"
    return paths, pid, hid


@pytest.fixture
def env(ready, monkeypatch):
    paths, pid, hid = ready
    monkeypatch.setenv("STUDIO_ROOT", str(paths.root))
    get_service.clear()
    return paths, pid, hid


def _app(pid, module):
    return AppTest.from_string(
        f"import sys\nsys.path.insert(0, {str(REPO)!r})\n"
        "import streamlit as st\n"
        f"st.session_state.setdefault('selected_project_id', {pid!r})\n"
        f"from ui import {module}\n{module}.render()\n", default_timeout=180)


def _button(at, suffix):
    return next(b for b in at.button if b.key and b.key.endswith(suffix))


def test_shorts_page_preview_and_approve(env):
    paths, pid, hid = env
    at = _app(pid, "shorts_review")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Validation passed" in s.value for s in at.success)
    assert len(at.get("video")) == 1
    _button(at, f"pv_{hid}").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert len(at.get("image")) >= 1
    _button(at, f"approved_{hid}").click().run()
    assert not at.exception
    assert StudioService(paths, runner=JobRunner(), configure_logging=False).short_review(hid)["status"] == "approved"


def test_thumbnails_page(env):
    _paths, pid, _hid = env
    at = _app(pid, "thumbnail_review")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("image")) >= 1
    codes = [c.value for c in at.code]
    assert any("Game X" in c and "bottom-right" in c.lower() for c in codes)


def test_metadata_page_edit_title(env):
    paths, pid, _hid = env
    at = _app(pid, "metadata_review")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    t1 = next(t for t in at.text_input if t.key and t.key.endswith("_t0"))
    t1.set_value("My edited title").run()
    _button(at, "save_titles").click().run()
    assert not at.exception, [e.value for e in at.exception]
    pack = StudioService(paths, runner=JobRunner(), configure_logging=False).creator_pack(pid)
    assert pack["titles"][0] == "My edited title" and "titles" in pack["edited_fields"]
    at.run()
    assert any("edited by you" in s.value for s in at.subheader)
