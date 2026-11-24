"""Phase 3 end-to-end on a synthetic 90 s recording with one real event at 40 s (loud burst + hard cut + speech).

Gemini is always a fake client (no network, no quota). Ollama runs through httpx.MockTransport here; a separate
optional test talks to a real local Ollama if one is running.
"""

from __future__ import annotations

import json
import shutil

import pytest

from pipeline import transcribe
from pipeline.ai_providers.gemini_provider import GeminiProvider
from pipeline.job_manager import JobRunner
from pipeline.models import TrackRole
from pipeline.project_service import StudioService
from tests.conftest import requires_ffmpeg

pytestmark = [requires_ffmpeg,
              pytest.mark.skipif(not transcribe.is_model_cached("tiny"), reason="Whisper 'tiny' model not downloaded")]


def _stages(svc, job):
    return {s["stage_name"]: s for s in svc.job_stages(job)}


@pytest.fixture
def project(studio_paths, media):
    svc = StudioService(studio_paths, runner=JobRunner(), configure_logging=False)
    src = studio_paths.raw_input / "Some Game" / "match ü.mp4"
    src.parent.mkdir(parents=True)
    shutil.copy(media["gameplay"], src)
    pid = svc.create_project(src, "Some Game", "Sports Clutch / Match", "en")
    svc.start_inspection(pid)
    assert svc.runner.wait_idle(120)
    svc.save_track_roles(pid, {0: TrackRole.COMBINED_MIX, 1: TrackRole.GAME, 2: TrackRole.MIC})
    svc.save_project_settings(pid, {"whisper.model": "tiny", "analysis.window_sec": 45,
                                    "ai.min_request_interval_sec": 0.0})
    return svc, pid


def _run(svc, pid, target="analysis", count=None, timeout=600):
    job = svc.start_analysis(pid, target, count)
    assert svc.runner.wait_idle(timeout)
    return job, svc.jobs.get_job(job)


def test_signals_only_analysis_finds_the_real_event(project):
    svc, pid = project
    svc.save_project_settings(pid, {"ai.provider": "none"})
    job, j = _run(svc, pid)
    assert j["state"] == "HIGHLIGHT_REVIEW", (j["error_summary"], _stages(svc, job))
    st = _stages(svc, job)
    assert all(st[n]["status"] == "COMPLETED" for n in ("signal_events", "frames", "ai_windows", "highlights"))
    files = svc.files(pid)
    se = json.loads(files.signal_events.read_text("utf-8"))
    kinds = {k for e in se["events"] for k in e["kinds"]}
    assert {"game_transient", "scene_change"} <= kinds
    hl = svc.highlights(pid)
    assert 1 <= len(hl) <= 3
    top = hl[0]
    # The top candidate contains the real event, has evidence, a breakdown, and honest signals-only labelling.
    assert top["eff_start"] <= 40.0 <= top["eff_end"]
    assert 20 <= top["eff_end"] - top["eff_start"] <= 50 and 0 <= top["eff_start"] and top["eff_end"] <= 90.01
    assert top["evidence"]["audio"] and top["evidence"]["visual"] and top["evidence"]["ai"] is None
    assert top["score"] <= 70 and top["score_breakdown"]["factors"]["signal_agreement"]["used"]
    assert "measurements only" in top["rationale"]
    assert top["hook_caption"] == "" and top["suggested_title"] == ""  # never invented without AI
    # Transcript evidence: the exact recognised words near 41 s.
    assert any("shot" in q["quote"].lower() for q in top["evidence"]["transcript"])
    # Frames come from the original source, with real files on disk.
    fr = json.loads(files.frames_index.read_text("utf-8"))["frames"]
    assert fr and all(svc.frame_path(f["path"]) for f in fr)
    assert any(abs(f["t"] - 40.0) <= 1.5 and f["reason"] == "evidence" for f in fr)
    # Events are stored in the DB with modality and evidence.
    ev = svc.analysis_events(pid)
    assert ev and all(e["event_type"] for e in ev)


def test_regenerate_candidates_reuses_analysis_and_keeps_user_decisions(project):
    svc, pid = project
    svc.save_project_settings(pid, {"ai.provider": "none"})
    job, _ = _run(svc, pid)
    before = _stages(svc, job)
    top = svc.highlights(pid)[0]
    svc.update_highlight(top["id"], start=top["eff_start"] + 1, status="approved", notes="keep this")
    # Ask for more candidates: only 'highlights' re-runs.
    job2, j = _run(svc, pid, "candidates", count=5)
    assert j["state"] == "HIGHLIGHT_REVIEW", j["error_summary"]
    after = _stages(svc, job2)
    for n in ("signal_events", "frames", "ai_windows"):
        assert after[n]["attempt_count"] == before[n]["attempt_count"]
    assert after["highlights"]["attempt_count"] == before["highlights"]["attempt_count"] + 1
    hl = svc.highlights(pid)
    approved = [h for h in hl if h["status"] == "approved"]
    assert len(approved) == 1 and approved[0]["user_notes"] == "keep this"
    assert approved[0]["user_start_sec"] == pytest.approx(top["eff_start"] + 1)
    # New suggestions never duplicate the approved clip.
    for h in hl:
        if h["status"] == "suggested":
            assert h["eff_end"] <= approved[0]["eff_start"] or h["eff_start"] >= approved[0]["eff_end"]


def _live_ollama_model() -> str | None:
    import httpx
    try:
        tags = httpx.get("http://localhost:11434/api/tags", timeout=3).json().get("models", [])
    except Exception:  # noqa: BLE001
        return None
    names = [m["name"] for m in tags]
    return "qwen2.5:3b" if "qwen2.5:3b" in names else None


@pytest.mark.skipif(_live_ollama_model() is None, reason="local Ollama with qwen2.5:3b not running")
def test_live_local_ollama_text_only(project):
    """Real call to a locally installed text-only model. Checks integration and validation, not answer quality."""
    svc, pid = project
    svc.save_project_settings(pid, {"ai.provider": "ollama", "ai.ollama_model": "qwen2.5:3b",
                                    "analysis.window_sec": 90})
    job, j = _run(svc, pid, timeout=900)
    assert j["state"] == "HIGHLIGHT_REVIEW", (j["error_summary"], _stages(svc, job))
    rep = svc.ai_report(pid)
    assert len(rep["windows"]) == 1 and rep["windows"][0]["mode"] == "text"
    assert rep["windows"][0]["status"] in ("ok", "invalid", "failed")
    if rep["windows"][0]["status"] == "ok":
        res = json.loads((svc.files(pid).windows_dir / "w0000.json").read_text("utf-8"))["result"]
        for e in res["events"]:
            assert e["frame_times"] == [] and e["confidence"]["visual"] == "none"
    assert svc.highlights(pid)


def test_highlight_edit_validation_and_preview(project):
    svc, pid = project
    svc.save_project_settings(pid, {"ai.provider": "none"})
    _run(svc, pid)
    h = svc.highlights(pid)[0]
    with pytest.raises(ValueError):
        svc.update_highlight(h["id"], start=50, end=40)
    with pytest.raises(ValueError):
        svc.update_highlight(h["id"], end=200)
    res = svc.update_highlight(h["id"], start=h["event_sec"] + 1, end=h["event_sec"] + 5)
    assert any("key moment" in w for w in res["warnings"]) and any("Length" in w for w in res["warnings"])
    svc.reset_highlight_times(h["id"])
    prev = svc.highlight_preview(h["id"])
    from pipeline import media_probe
    info = media_probe.probe(prev)
    assert info.video and info.video.height == 480
    assert info.duration_sec == pytest.approx(min(h["end_sec"] + 5, 90) - max(h["start_sec"] - 5, 0), abs=0.5)
