"""Integration tests on real FFmpeg-generated media (CPU only, no network, no paid services)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pipeline import hardware, media_probe
from pipeline.job_manager import JobRunner
from pipeline.models import HardwareMode, JobState, TrackRole
from pipeline.project_service import DuplicateProject, StudioService
from pipeline.validators import validate_source_media
from tests.conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


@pytest.mark.parametrize("key,n_audio", [("one", 1), ("two", 2), ("three", 3), ("noaudio", 0)])
def test_probe_track_counts(media, key, n_audio):
    info = media_probe.probe(media[key])
    assert len(info.audio) == n_audio
    assert info.video and info.video.width == 640 and info.video.height == 360
    assert info.duration_sec == pytest.approx(6, abs=0.2)
    assert validate_source_media(info, media[key]) == []


def test_probe_three_track_titles_and_roles(media):
    info = media_probe.probe(media["three"])
    assert [a.title for a in info.audio] == ["Mix", "Game", "Mic"]
    assert media_probe.suggest_roles(info.audio) == [TrackRole.COMBINED_MIX, TrackRole.GAME, TrackRole.MIC]


def test_vfr_measured_from_timestamps(media):
    info = media_probe.probe(media["vfr"])
    is_vfr, details = media_probe.measure_vfr(media["vfr"])
    assert is_vfr, details
    assert media_probe.measure_vfr(media["one"])[0] is False
    assert info.video is not None


def test_invalid_sources_rejected(media):
    info = media_probe.probe(media["audioonly"])
    assert any("No video" in e for e in validate_source_media(info, media["audioonly"]))
    with pytest.raises(Exception):
        media_probe.probe(media["truncated"])


def test_track_preview_and_levels(media, tmp_path):
    out = media_probe.extract_track_preview(media["three"], 2, tmp_path / "p.wav", 1.0, 2)
    p = media_probe.probe(out)
    assert p.audio[0].sample_rate == 48000 and p.duration_sec == pytest.approx(2, abs=0.1)
    assert media_probe.track_mean_volume(media["three"], 1, 6) is not None


def test_hardware_detect_on_this_machine(tmp_path):
    r = hardware.detect(run_whisper_smoke=False)
    assert r.ffmpeg_ok
    assert r.onnx_smoke_ok, r.onnx_error
    if not r.nvidia_smi_found:
        assert r.mode == HardwareMode.CPU.value
        assert not r.nvenc_encode_ok


# --- end-to-end service flow ------------------------------------------------------------------

@pytest.fixture
def service(studio_paths):
    runner = JobRunner()
    svc = StudioService(studio_paths, runner=runner, configure_logging=False)
    yield svc


def _place(svc: StudioService, src: Path, game: str, name: str) -> Path:
    dest = svc.paths.raw_input / game / "session 1" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    return dest


def test_full_inspection_flow_with_resume_and_duplicates(service: StudioService, media):
    src = _place(service, media["three"], "Cricket 24 ü", "match 01 ü.mp4")
    files = service.scan_raw_input()
    assert files[0].game_guess == "Cricket 24 ü"

    pid = service.create_project(src, files[0].game_guess, "Sports Clutch / Match", "hinglish", "last over chase")
    proj = service.get_project(pid)
    assert proj["source_path"] == "01_RAW_INPUT/Cricket 24 ü/session 1/match 01 ü.mp4"
    assert service.game_suggestions() == ["Cricket 24 ü"]

    job_id = service.start_inspection(pid)
    assert service.runner.wait_idle(120)
    job = service.latest_job(pid)
    assert job["id"] == job_id and job["state"] == "INSPECTING" and job["progress"] == 1.0
    stages = {s["stage_name"]: s for s in service.job_stages(job_id)}
    assert {k: v["status"] for k, v in stages.items()} == {
        "probe": "COMPLETED", "checksum": "COMPLETED", "track_levels": "COMPLETED"}

    proj = service.get_project(pid)
    assert proj["status"] == "inspected" and len(proj["source_sha256"]) == 64
    tracks = service.audio_tracks(pid)
    assert [t["detected_role"] for t in tracks] == ["combined_mix", "game", "mic"]
    assert all(t["mean_volume_db"] is not None for t in tracks)

    manifest = json.loads((service.paths.project_work_dir(proj["folder_name"]) / "manifest.json").read_text("utf-8"))
    assert manifest["source"]["sha256"] == proj["source_sha256"]
    assert manifest["software"]["python"].startswith("3.12")
    assert set(manifest["stages"]) == {"probe", "checksum", "track_levels"}

    # Confirm roles (user override of the suggestion).
    with pytest.raises(ValueError):
        service.save_track_roles(pid, {0: TrackRole.MIC, 1: TrackRole.MIC, 2: TrackRole.GAME})
    service.save_track_roles(pid, {0: TrackRole.COMBINED_MIX, 1: TrackRole.MIC, 2: TrackRole.GAME})
    assert service.get_project(pid)["tracks_confirmed"] == 1

    # Re-run: all stages reused, user roles preserved.
    attempts_before = {s["stage_name"]: s["attempt_count"] for s in service.job_stages(job_id)}
    service.start_inspection(pid)
    assert service.runner.wait_idle(60)
    attempts_after = {s["stage_name"]: s["attempt_count"] for s in service.job_stages(job_id)}
    assert attempts_after == attempts_before
    assert [t["user_role"] for t in service.audio_tracks(pid)] == ["combined_mix", "mic", "game"]
    assert service.get_project(pid)["status"] == "tracks_confirmed"

    # Duplicate import detected even under another name.
    copy = _place(service, media["three"], "Other", "renamed.mp4")
    with pytest.raises(DuplicateProject) as exc:
        service.create_project(copy, "Other", "Custom", "auto")
    assert exc.value.project_id == pid

    preview = service.track_preview(pid, 1, start_sec=1.0)
    assert preview.exists() and preview.stat().st_size > 1000


def test_no_audio_project_inspects_cleanly(service: StudioService, media):
    src = _place(service, media["noaudio"], "Silent Game", "no audio.mp4")
    pid = service.create_project(src, "Silent Game", "Custom", "auto")
    service.start_inspection(pid)
    assert service.runner.wait_idle(60)
    job = service.latest_job(pid)
    assert job["state"] == "INSPECTING", job
    assert service.audio_tracks(pid) == []
    stages = {s["stage_name"]: s for s in service.job_stages(job["id"])}
    assert any("No audio" in w for w in stages["probe"]["warnings"])
    service.save_track_roles(pid, {})


def test_failed_source_marks_job_failed_then_retry_succeeds(service: StudioService, media):
    src = _place(service, media["one"], "G", "rec.mp4")
    pid = service.create_project(src, "G", "Custom", "en")
    moved = src.with_name("moved.mp4")
    src.rename(moved)
    service.start_inspection(pid)
    assert service.runner.wait_idle(60)
    job = service.latest_job(pid)
    assert job["state"] == "FAILED" and "not found" in job["error_summary"]
    moved.rename(src)
    service.start_inspection(pid)
    assert service.runner.wait_idle(60)
    job = service.latest_job(pid)
    assert job["state"] == "INSPECTING"
    probe = [s for s in service.job_stages(job["id"]) if s["stage_name"] == "probe"][0]
    assert probe["attempt_count"] == 2 and probe["status"] == "COMPLETED"


def test_cancel_and_pause(service: StudioService, media):
    src = _place(service, media["two"], "G", "two.mkv")
    pid = service.create_project(src, "G", "Custom", "auto")
    job_id = service.jobs.create_job(pid)
    service.cancel_job(job_id)
    assert service.jobs.get_job(job_id)["state"] == "CANCELLED"
    # A new job is created after cancellation.
    new_job = service.start_inspection(pid)
    assert new_job != job_id
    assert service.runner.wait_idle(60)
    # Pause request is honoured between stages on the next run.
    service.jobs.request_pause(new_job)
    service.runner.submit(new_job, lambda: service._run_inspection(pid, new_job))
    assert service.runner.wait_idle(60)
    assert service.jobs.get_job(new_job)["state"] == "PAUSED"
    service.start_inspection(pid)
    assert service.runner.wait_idle(60)
    assert service.jobs.get_job(new_job)["state"] == "INSPECTING"


def test_interrupted_stage_recovered_on_restart(studio_paths, media):
    svc = StudioService(studio_paths, runner=JobRunner(), configure_logging=False)
    src = _place(svc, media["one"], "G", "r.mp4")
    pid = svc.create_project(src, "G", "Custom", "auto")
    job = svc.jobs.create_job(pid)
    svc.jobs.advance_to(job, JobState.INSPECTING)
    svc.db.execute("INSERT INTO job_stages(job_id, stage_name, status) VALUES (?, 'probe', 'RUNNING')", (job,))
    svc2 = StudioService(studio_paths, runner=JobRunner(), configure_logging=False)
    assert svc2.recovered == 1
    svc2.start_inspection(pid)
    assert svc2.runner.wait_idle(60)
    assert svc2.latest_job(pid)["state"] == "INSPECTING"
