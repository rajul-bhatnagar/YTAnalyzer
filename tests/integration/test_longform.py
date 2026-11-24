"""Phase 2 end-to-end: extraction, mix modes, ducking, transcription (Whisper 'tiny'), edits, inactivity
analysis, rough cut render + validation, timeline export, voiceover with offset, and stage reuse."""

from __future__ import annotations

import json
import shutil

import pytest

from pipeline import audio_prep, media_probe, transcribe
from pipeline.job_manager import JobRunner
from pipeline.models import TrackRole
from pipeline.project_service import StudioService
from tests.conftest import requires_ffmpeg

pytestmark = [requires_ffmpeg,
              pytest.mark.skipif(not transcribe.is_model_cached("tiny"), reason="Whisper 'tiny' model not downloaded")]


def _attempts(svc, job_id):
    return {s["stage_name"]: s["attempt_count"] for s in svc.job_stages(job_id)}


def _status(svc, job_id):
    return {s["stage_name"]: s["status"] for s in svc.job_stages(job_id)}


@pytest.fixture
def project(studio_paths, media):
    svc = StudioService(studio_paths, runner=JobRunner(), configure_logging=False)
    src = studio_paths.raw_input / "Cricket 24" / "longform ü.mp4"
    src.parent.mkdir(parents=True)
    shutil.copy(media["longform"], src)
    pid = svc.create_project(src, "Cricket 24", "Sports Clutch / Match", "en")
    svc.start_inspection(pid)
    assert svc.runner.wait_idle(120)
    svc.save_track_roles(pid, {0: TrackRole.COMBINED_MIX, 1: TrackRole.GAME, 2: TrackRole.MIC})
    svc.save_project_settings(pid, {"whisper.model": "tiny", "render.quality_preset": "fast"})
    return svc, pid


def _run(svc, pid, target, timeout=300):
    job = svc.start_longform(pid, target)
    assert svc.runner.wait_idle(timeout)
    j = svc.jobs.get_job(job)
    assert j["state"] == "AUDIO_PREP", (j["state"], j["error_summary"], svc.job_stages(job))
    return job


def test_longform_pipeline_end_to_end(project, media):
    svc, pid = project
    files = svc.files(pid)

    # 1. Audio: extraction + signals + default game_only mix.
    job = _run(svc, pid, "audio")
    st = _status(svc, job)
    assert all(st[s] == "COMPLETED" for s in ("extract_audio", "voiceover", "signals", "mix")), st
    for role, idx in (("combined_mix", 0), ("game", 1), ("mic", 2)):
        info = audio_prep.probe_audio_file(files.track_wav(idx, role))
        assert info["duration_sec"] == pytest.approx(60, abs=0.05) and info["sample_rate"] == 48000
    assert audio_prep.probe_audio_file(files.mix_wav("game_only"))["duration_sec"] == pytest.approx(60, abs=0.05)
    sig_meta = json.loads(files.signals.with_suffix(".json").read_text())
    assert sig_meta["has_speech_track"] and sig_meta["speech_seconds"] > 3

    # 2. Change mix -> only the mix re-runs.
    before = _attempts(svc, job)
    svc.save_project_settings(pid, {"audio.mode": "both_auto_duck", "audio.game_volume_db": 0.0,
                                    "audio.loudnorm_enabled": False, "audio.limiter_enabled": False})
    _run(svc, pid, "audio")
    after = _attempts(svc, job)
    assert after["extract_audio"] == before["extract_audio"] and after["signals"] == before["signals"]
    assert after["mix"] == before["mix"] + 1
    assert audio_prep.probe_audio_file(files.mix_wav("both_auto_duck"))["duration_sec"] == pytest.approx(60, abs=0.05)
    # Ducking measurably lowers the game while the mic speaks (speech ~3-9 s). Compare the same mix with
    # ducking effectively off (ratio 1); the voice is turned right down so the game dominates both.
    base = svc.longform_context(pid).mix_settings().model_copy(update={"voice_gain_db": -20.0})
    ducked = svc.mix_preview(pid, base.model_copy(update={"duck_ratio": 6.0}), 4, 3)
    ducked_db = audio_prep.audio_stats(ducked)["rms_db"]
    plain = svc.mix_preview(pid, base.model_copy(update={"duck_ratio": 1.0}), 4, 3)
    plain_db = audio_prep.audio_stats(plain)["rms_db"]
    assert ducked_db is not None and plain_db is not None
    assert ducked_db < plain_db - 3, f"ducking did not lower game under speech: {ducked_db} vs {plain_db}"

    # 3. Transcript + subtitles.
    job = _run(svc, pid, "transcript")
    raw, _ = svc.transcript(pid)
    text = " ".join(s["text"] for s in raw["segments"]).lower()
    # Whisper may write the number as a word or digit; both are correct transcriptions.
    assert ("six" in text or " 6 " in f" {text} ") and "shot" in text
    assert raw["model"]["effective"] == "tiny" and raw["model"]["device"] == "cpu"
    words = json.loads(files.words_json.read_text("utf-8"))["words"]
    assert words and words[0]["start"] >= 2.5  # speech starts at 3 s in the recording
    assert files.subtitles_srt.read_text("utf-8").startswith("1\n")

    # 4. User correction -> subtitles rebuilt, transcription reused.
    t_attempts = _attempts(svc, job)["transcribe"]
    svc.save_transcript_edits(pid, {str(raw["segments"][0]["id"]): "WHAT A SHOT corrected text"})
    _run(svc, pid, "subtitles")
    assert _attempts(svc, job)["transcribe"] == t_attempts
    assert "WHAT A SHOT corrected text" in files.transcript_txt.read_text("utf-8")

    # 5. Cut analysis: only the static silent loading screen (15-45 s) is a candidate.
    svc.save_project_settings(pid, {"silence.policy": "conservative"})
    _run(svc, pid, "cuts")
    dec = svc.cut_decisions(pid)
    cands = [s for s in dec["segments"] if s["kind"] == "candidate"]
    assert len(cands) == 1
    assert 15 <= cands[0]["source_start"] <= 20 and 40 <= cands[0]["source_end"] <= 45
    assert 20 <= dec["removed_sec"] <= 30

    # 6. Render rough cut + timeline export.
    job = _run(svc, pid, "render", timeout=600)
    st = _status(svc, job)
    assert st["rough_cut"] == "COMPLETED" and st["timeline_export"] == "COMPLETED", st
    out = media_probe.probe(files.rough_cut)
    assert out.duration_sec == pytest.approx(dec["output_duration"], abs=0.3)
    assert out.video and (out.video.width, out.video.height) == (640, 360) and out.audio
    assert files.edl.exists() and files.fcp7_xml.exists() and files.timeline_json.exists()
    rows = svc.rendered_outputs(pid)
    assert rows[0]["output_type"] == "long_rough_cut" and rows[0]["validation_status"] == "passed"

    # 7. Override: keep the candidate -> new render, previous one backed up, no-cut path copies video.
    svc.save_cut_overrides(pid, {cands[0]["id"]: True})
    _run(svc, pid, "render", timeout=600)
    assert media_probe.probe(files.rough_cut).duration_sec == pytest.approx(60, abs=0.3)
    assert list((files.ready / "_previous").glob("long_rough_cut_*.mp4"))
    assert svc.rendered_outputs(pid)[0]["settings"]["encoder"] == "copy"


def test_voiceover_offset_shifts_captions_and_mixes(project, media, tmp_path):
    svc, pid = project
    _run(svc, pid, "audio")
    with open(media["speech"], "rb") as fh:
        wav = svc.set_voiceover(pid, fh, "my voiceover.wav")
    assert wav.exists() and svc.voiceover_versions(pid) == [wav]
    svc.save_project_settings(pid, {"audio.voice_source": "voiceover", "audio.voice_offset_sec": 20.0,
                                    "audio.mode": "both_auto_duck", "transcription.source": "voiceover"})
    job = _run(svc, pid, "transcript")
    words = json.loads(svc.files(pid).words_json.read_text("utf-8"))
    assert words["offset_applied"] == 20.0 and words["words"][0]["start"] >= 19.5
    before = _attempts(svc, job)
    # Changing only the offset must not re-run transcription, only subtitles.
    svc.save_project_settings(pid, {"audio.voice_offset_sec": 25.0})
    _run(svc, pid, "subtitles")
    after = _attempts(svc, job)
    assert after["transcribe"] == before["transcribe"] and after["subtitles"] == before["subtitles"] + 1
    assert json.loads(svc.files(pid).words_json.read_text("utf-8"))["words"][0]["start"] >= 24.5
    _run(svc, pid, "audio")
    rep = json.loads(svc.files(pid).mix_report("both_auto_duck").read_text())
    assert rep["sources"] == {"game": "game", "voice": "voiceover"}
    prev = svc.mix_preview(pid, svc.longform_context(pid).mix_settings(), 20, 5)
    assert audio_prep.probe_audio_file(prev)["duration_sec"] == pytest.approx(5, abs=0.1)
    # Preview sync: with offset 25 s the ~7 s voiceover plays at 25-32 s of video time.
    voice = svc.longform_context(pid).mix_settings().model_copy(update={"mode": "voice_only"})
    during = audio_prep.audio_stats(svc.mix_preview(pid, voice, 26, 3))["rms_db"]
    assert during is not None and during > -40, f"voiceover missing in preview at 26 s: {during}"
    before_vo = audio_prep.audio_stats(svc.mix_preview(pid, voice, 10, 3))["rms_db"]
    assert before_vo is None or before_vo < -60, f"voiceover audible before its offset: {before_vo}"


def test_unavailable_mix_mode_fails_clearly(studio_paths, media):
    svc = StudioService(studio_paths, runner=JobRunner(), configure_logging=False)
    src = studio_paths.raw_input / "G" / "one.mp4"
    src.parent.mkdir(parents=True)
    shutil.copy(media["one"], src)
    pid = svc.create_project(src, "G", "Custom", "auto")
    svc.start_inspection(pid)
    assert svc.runner.wait_idle(60)
    with pytest.raises(ValueError, match="Confirm the audio track roles"):
        svc.start_longform(pid, "audio")
    svc.save_track_roles(pid, {0: TrackRole.COMBINED_MIX})
    svc.save_project_settings(pid, {"audio.mode": "voice_only"})
    job = svc.start_longform(pid, "audio")
    assert svc.runner.wait_idle(120)
    j = svc.jobs.get_job(job)
    assert j["state"] == "FAILED" and "voice_only" in j["error_summary"]
