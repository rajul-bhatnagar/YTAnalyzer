"""Analysis & highlights page driven headlessly with AppTest on a project analysed with signals only."""

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


def _svc(paths):
    return StudioService(paths, runner=JobRunner(), configure_logging=False)


@pytest.fixture
def analysed(studio_paths, media, monkeypatch):
    monkeypatch.setenv("STUDIO_ROOT", str(studio_paths.root))
    get_service.clear()
    svc = _svc(studio_paths)
    src = studio_paths.raw_input / "My Game" / "run ü.mp4"
    src.parent.mkdir(parents=True)
    shutil.copy(media["gameplay"], src)
    pid = svc.create_project(src, "My Game", "Custom", "en")
    svc.start_inspection(pid)
    assert svc.runner.wait_idle(120)
    svc.save_track_roles(pid, {0: TrackRole.COMBINED_MIX, 1: TrackRole.GAME, 2: TrackRole.MIC})
    # Point Ollama at a closed port so the page never depends on a local Ollama during tests.
    svc.save_project_settings(pid, {"whisper.model": "tiny", "analysis.window_sec": 45, "ai.provider": "none",
                                    "ai.ollama_base_url": "http://127.0.0.1:9"})
    svc.start_analysis(pid)
    assert svc.runner.wait_idle(600)
    assert svc.jobs.get_job(svc.latest_job(pid)["id"])["state"] == "HIGHLIGHT_REVIEW"
    return studio_paths, pid


def _app(pid):
    return AppTest.from_string(
        f"import sys\nsys.path.insert(0, {str(REPO)!r})\n"
        "import streamlit as st\n"
        f"st.session_state.setdefault('selected_project_id', {pid!r})\n"
        "from ui import highlight_review\nhighlight_review.render()\n",
        default_timeout=120)


def _button(at, suffix):
    return next(b for b in at.button if b.key and b.key.endswith(suffix))


def test_page_shows_candidates_with_evidence(analysed):
    _paths, pid = analysed
    at = _app(pid)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["⚙️ Setup & run", "🎯 Highlights", "📋 All events"]
    md = " ".join(m.value for m in at.markdown)
    assert any(s.value.startswith("Suggestions") for s in at.subheader)
    assert "AI actually used:** none (signals only)" in md
    assert not at.error, [e.value for e in at.error]
    # Ollama unreachable is reported, not a crash; cloud consent is off by default.
    assert any("not reachable" in w.value for w in at.warning)
    assert any("nothing from this project is sent" in i.value for i in at.info)
    assert any("free-tier content is used to improve" in w.value for w in at.warning)


def test_approve_and_consent_via_buttons(analysed):
    paths, pid = analysed
    svc = _svc(paths)
    top = svc.highlights(pid)[0]
    at = _app(pid)
    at.run()
    _button(at, f"Approve_{top['id']}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert svc.get_highlight(top["id"])["status"] == "approved"
    assert svc.db.query_one("SELECT decision FROM review_decisions WHERE asset_ref = ?",
                            (f"highlight:{top['id']}",))["decision"] == "approved"
    _button(at, "consent_frames").click().run()
    assert svc.cloud_consent(pid)["mode"] == "frames"
    _button(at, "consent_revoke").click().run()
    assert svc.cloud_consent(pid)["mode"] == ""


def test_preview_button_renders_video(analysed):
    _paths, pid = analysed
    at = _app(pid)
    at.run()
    svc = _svc(_paths)
    top = svc.highlights(pid)[0]
    _button(at, f"pbtn_{top['id']}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert len(at.get("video")) == 1
