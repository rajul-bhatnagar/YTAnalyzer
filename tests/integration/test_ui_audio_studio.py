"""Long video & audio studio page, driven headlessly with AppTest on a real processed project:
renders all tabs, previews mixes, saves transcript edits and cut overrides through the UI, and retries
a failed long-form job with the right operation."""

from __future__ import annotations

import json
import shutil
import time

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
def processed(studio_paths, media, monkeypatch):
    """A 60 s 3-track project with audio, transcript and cut analysis done (conservative policy)."""
    monkeypatch.setenv("STUDIO_ROOT", str(studio_paths.root))
    get_service.clear()
    svc = _svc(studio_paths)
    src = studio_paths.raw_input / "Any Game" / "session ü.mp4"
    src.parent.mkdir(parents=True)
    shutil.copy(media["longform"], src)
    pid = svc.create_project(src, "Any Game", "Custom", "en")
    svc.start_inspection(pid)
    assert svc.runner.wait_idle(120)
    svc.save_track_roles(pid, {0: TrackRole.COMBINED_MIX, 1: TrackRole.GAME, 2: TrackRole.MIC})
    svc.save_project_settings(pid, {"whisper.model": "tiny", "silence.policy": "conservative"})
    for target in ("audio", "transcript", "cuts"):
        svc.start_longform(pid, target)
        assert svc.runner.wait_idle(300)
        assert svc.jobs.get_job(svc.latest_job(pid)["id"])["state"] == "AUDIO_PREP"
    return studio_paths, pid


def _app(pid):
    return AppTest.from_string(
        f"import sys\nsys.path.insert(0, {str(REPO)!r})\n"
        "import streamlit as st\n"
        f"st.session_state.setdefault('selected_project_id', {pid!r})\n"
        "from ui import audio_studio\naudio_studio.render()\n",
        default_timeout=120)


def _button(at, key_suffix):
    return next(b for b in at.button if b.key and b.key.endswith(key_suffix))


def test_studio_renders_all_tabs_with_results(processed):
    _paths, pid = processed
    at = _app(pid)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["🎚️ Audio mix", "🎙️ Voiceover", "📝 Transcript & captions",
                                          "✂️ Cuts & rough cut"]
    # The default game_only mix exists and passed validation.
    assert any("validation passed" in s.value for s in at.success)
    # Exactly one inactive stretch (the static loading screen) is listed for review.
    assert any("1 inactive stretch" in m.value for m in at.markdown)
    # Without a voiceover, voice_only still works from the mic, and ducking is available (separate game track).
    assert not any("unavailable" in w.value for w in at.warning)


def _stage(svc, pid, name):
    return next((s for s in svc.job_stages(svc.latest_job(pid)["id"]) if s["stage_name"] == name), None)


def _wait_stage_rerun(svc, pid, name, attempts_before, timeout=120.0):
    """The page's service runs jobs on its own worker; wait on the shared database for the stage to re-run."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = _stage(svc, pid, name)
        if s and s["attempt_count"] > attempts_before and s["status"] in ("COMPLETED", "FAILED"):
            return s
        time.sleep(0.25)
    raise AssertionError(f"stage {name} did not re-run")


def test_mix_previews_render_for_all_available_modes(processed):
    _paths, pid = processed
    at = _app(pid)
    at.run()
    _button(at, "pv_go").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert len(at.get("audio")) == 3  # game_only, voice_only and both_auto_duck


def test_ui_save_buttons_round_trip_edits_and_overrides(processed):
    """AppTest cannot type into data-editor cells, so edits are persisted through the service; the test then
    proves the page's own Save buttons re-read and keep them instead of silently discarding them."""
    paths, pid = processed
    svc = _svc(paths)
    raw, _ = svc.transcript(pid)
    first = str(raw["segments"][0]["id"])
    svc.save_transcript_edits(pid, {first: "Corrected by the user"})
    at = _app(pid)
    at.run()
    _button(at, "t_save").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert svc.transcript(pid)[1] == {first: "Corrected by the user"}

    # Mark the suggested cut as "keep", re-analyse, then press the page's Save & re-analyse button.
    cand = next(s for s in svc.cut_decisions(pid)["segments"] if s["kind"] == "candidate")
    svc.save_cut_overrides(pid, {cand["id"]: True})
    svc.start_longform(pid, "cuts")
    assert svc.runner.wait_idle(120)
    assert svc.cut_decisions(pid)["removed_sec"] < 0.5
    before = _stage(svc, pid, "cut_analysis")["attempt_count"]
    at.run()
    _button(at, "cut_save").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert _wait_stage_rerun(svc, pid, "cut_analysis", before)["status"] == "COMPLETED"
    assert json.loads(svc.files(pid).cut_overrides.read_text("utf-8")) == {cand["id"]: True}
    assert svc.cut_decisions(pid)["removed_sec"] < 0.5


def test_retry_uses_last_long_form_operation(processed):
    paths, pid = processed
    svc = _svc(paths)
    assert svc.last_operation(pid) == "cuts"
    # Force the next mix to fail: voice_only with no voice source.
    svc.save_project_settings(pid, {"audio.mode": "voice_only", "audio.voice_source": "none"})
    svc.start_longform(pid, "audio")
    assert svc.runner.wait_idle(120)
    assert svc.jobs.get_job(svc.latest_job(pid)["id"])["state"] == "FAILED"
    assert svc.last_operation(pid) == "audio"
    # Fix the setting; Retry re-runs the audio operation (not inspection) and reuses extraction.
    svc.save_project_settings(pid, {"audio.mode": "game_only", "audio.voice_source": "auto"})
    job = svc.retry_job(pid)
    assert svc.runner.wait_idle(120)
    j = svc.jobs.get_job(job)
    assert j["state"] == "AUDIO_PREP", j["error_summary"]
    stages = {s["stage_name"]: s for s in svc.job_stages(job)}
    assert stages["mix"]["status"] == "COMPLETED"
    assert stages["probe"]["attempt_count"] == 1  # inspection not repeated
