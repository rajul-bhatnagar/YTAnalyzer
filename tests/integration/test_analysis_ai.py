"""Phase 3 AI paths with fake providers: a fake google-genai client (never the network, never quota) and Ollama
through httpx.MockTransport. Proves validation, consent, quota fallback/pause, resume and repair behaviour."""

from __future__ import annotations

import json
import re
import shutil
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from pipeline import transcribe
from pipeline.ai_providers.gemini_provider import GeminiProvider
from pipeline.ai_providers.ollama_provider import OllamaProvider
from pipeline.job_manager import JobRunner
from pipeline.models import TrackRole
from pipeline.project_service import StudioService
from tests.conftest import requires_ffmpeg

pytestmark = [requires_ffmpeg,
              pytest.mark.skipif(not transcribe.is_model_cached("tiny"), reason="Whisper 'tiny' model not downloaded")]


def _event(start, end, **kw):
    e = {"start_sec": start, "end_sec": end, "category": "achievement", "description": "Big hit lands",
         "visual_observation": "", "frame_times": [], "audio_observation": "", "transcript_quote": "",
         "confidence": {"visual": "high", "audio": "medium", "transcript": "medium"}, "uncertainty": "",
         "factors": {"significance": 5, "stakes": 4, "surprise": 3, "skill": 4, "entertainment": 5,
                     "narrative_completeness": 4, "visual_clarity": 4, "standalone": 4},
         "hook_caption": "WHAT A SHOT", "suggested_title": "A huge six"}
    e.update(kw)
    return e


class FakeGenai:
    """Mimics the parts of google.genai.Client the provider uses. ``script`` yields one behaviour per call."""

    def __init__(self, script=None):
        self.calls = []
        self.script = list(script or [])
        self.models = SimpleNamespace(get=self._get, list=lambda: [], generate_content=self._generate)
        self.files = SimpleNamespace(upload=None, get=None, delete=None)

    def _get(self, model):
        return SimpleNamespace(name=f"models/{model}", supported_actions=["generateContent"], input_token_limit=1000)

    def enable_files(self):
        self.uploads, self.deleted, self.polls = [], [], 0

        def upload(file, config):
            self.uploads.append({"file": file, "mime": config.mime_type})
            return SimpleNamespace(name=f"files/f{len(self.uploads)}", state="PROCESSING", uri="https://x/f",
                                   mime_type=config.mime_type, error=None)

        def get(name):
            self.polls += 1
            return SimpleNamespace(name=name, state="ACTIVE", uri="https://x/f", mime_type="video/mp4", error=None)

        self.files = SimpleNamespace(upload=upload, get=get, delete=lambda name: self.deleted.append(name))

    def _generate(self, model, contents, config):
        if any(getattr(p, "file_data", None) for p in contents):
            prompt = [p.text for p in contents if getattr(p, "text", None)][-1]
            w0 = float(re.search(r"\(([\d.]+)-([\d.]+) s of the recording\)", prompt).group(1))
            self.calls.append({"video": True, "prompt": prompt})
            # Clip-relative times, as the video-mode prompt instructs.
            return SimpleNamespace(text=json.dumps({"summary": "", "events": [
                _event(40.0 - w0 - 0.5, 44.0 - w0, frame_times=[40.5 - w0], visual_observation="Bars appear")]}),
                candidates=[])
        texts = [p.text for p in contents if getattr(p, "text", None)]
        shown = [float(m) for t in texts for m in re.findall(r"\(([\d.]+) s\)", t)]
        prompt = texts[-1]
        if not shown:  # repair requests are text-only; the prompt still lists the frames that were shown
            m = re.search(r"Frames provided at \(s\): ([\d., ]+)", prompt)
            shown = [float(x) for x in m.group(1).split(",")] if m else []
        self.calls.append({"model": model, "shown": shown, "prompt": prompt, "images": sum(
            1 for p in contents if getattr(p, "inline_data", None))})
        action = self.script.pop(0) if self.script else "good"
        if action == "quota":
            raise genai_errors.ClientError(429, {"error": {"code": 429, "message": "Resource exhausted",
                                                           "status": "RESOURCE_EXHAUSTED",
                                                           "details": [{"retryDelay": "3600s"}]}})
        if action == "503":
            raise genai_errors.ServerError(503, {"error": {"code": 503, "message": "The model is overloaded.",
                                                           "status": "UNAVAILABLE"}})
        if action == "bad_json":
            return SimpleNamespace(text="{not json", candidates=[])
        w0 = float(re.search(r"\(([\d.]+)-([\d.]+) s of the recording\)", prompt).group(1))
        if w0 < 40 <= w0 + 45:
            near = min(shown, key=lambda t: abs(t - 40.5))
            events = [
                _event(39.5, 45.0, frame_times=[near, 12345.0], visual_observation="Colour bars replace the pitch",
                       transcript_quote="a massive 6"),
                _event(30.0, 33.0, transcript_quote="we won the world cup", frame_times=[],
                       confidence={"visual": "high", "audio": "low", "transcript": "high"}, category="Funny Moment"),
                _event(500.0, 510.0),  # outside the window
            ]
        else:
            events = []
        return SimpleNamespace(text=json.dumps({"summary": "test", "events": events, "uncertainty": ""}),
                               candidates=[])


def _ollama_transport(calls, caps=("completion",)):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.16"})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": list(caps)})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            calls.append(body)
            prompt = body["messages"][1]["content"]
            w0 = float(re.search(r"\(([\d.]+)-([\d.]+) s of the recording\)", prompt).group(1))
            events = [_event(40.0, 44.0, frame_times=[40.0], visual_observation="I see a bat",
                             transcript_quote="massive 6")] if w0 < 40 <= w0 + 45 else []
            return httpx.Response(200, json={"message": {"content": json.dumps({"summary": "", "events": events})},
                                             "done_reason": "stop"})
        return httpx.Response(404, json={"error": "not found"})
    return httpx.MockTransport(handler)


@pytest.fixture
def setup(studio_paths, media, monkeypatch):
    svc = StudioService(studio_paths, runner=JobRunner(), configure_logging=False)
    src = studio_paths.raw_input / "Cricket 24" / "final.mp4"
    src.parent.mkdir(parents=True)
    shutil.copy(media["gameplay"], src)
    pid = svc.create_project(src, "Cricket 24", "Sports Clutch / Match", "en")
    svc.start_inspection(pid)
    assert svc.runner.wait_idle(120)
    svc.save_track_roles(pid, {0: TrackRole.COMBINED_MIX, 1: TrackRole.GAME, 2: TrackRole.MIC})
    svc.save_project_settings(pid, {"whisper.model": "tiny", "analysis.window_sec": 45, "analysis.window_overlap_sec": 0,
                                    "ai.min_request_interval_sec": 0.0, "ai.provider": "gemini",
                                    "ai.gemini_visual_enabled": True, "ai.ollama_model": "qwen-test:1b"})
    fake = FakeGenai()
    ollama_calls: list = []
    monkeypatch.setattr(svc, "_gemini", lambda ai: GeminiProvider("test-key", ai["gemini_model"],
                                                                  client_factory=lambda k: fake, sleep=lambda s: None))
    monkeypatch.setattr(svc, "_ollama", lambda ai, model=None: OllamaProvider(
        ai["ollama_base_url"], ai["ollama_model"] if model is None else model, transport=_ollama_transport(ollama_calls)))
    return svc, pid, fake, ollama_calls


def _run(svc, pid, target="analysis"):
    job = svc.start_analysis(pid, target)
    assert svc.runner.wait_idle(600)
    return job, svc.jobs.get_job(job)


def test_no_consent_means_nothing_is_sent_to_gemini(setup):
    svc, pid, fake, ollama_calls = setup
    job, j = _run(svc, pid)
    assert j["state"] == "HIGHLIGHT_REVIEW", j["error_summary"]
    assert fake.calls == []
    rep = svc.ai_report(pid)
    assert any("not been approved" in n for n in rep["plan_notes"])
    # Fell back to local Ollama (text-only), which never saw frames.
    assert rep["effective"] == ["ollama:qwen-test:1b (text)"] and ollama_calls
    assert all("images" not in c["messages"][1] for c in ollama_calls)


def test_gemini_frames_are_validated_against_evidence(setup):
    svc, pid, fake, _ = setup
    svc.grant_cloud_consent(pid, "frames")
    job, j = _run(svc, pid)
    assert j["state"] == "HIGHLIGHT_REVIEW", j["error_summary"]
    assert len(fake.calls) == 2 and all(c["images"] > 0 for c in fake.calls)
    rep = svc.ai_report(pid)
    assert rep["effective"] == ["gemini:gemini-3.8-flash (images)"]
    w0 = json.loads((svc.files(pid).windows_dir / "w0000.json").read_text("utf-8"))
    evs = w0["result"]["events"]
    # Event outside the window dropped; unseen frame time removed; invented quote removed.
    assert len(evs) == 2
    good = evs[0]
    assert 12345.0 not in good["frame_times"] and len(good["frame_times"]) == 1
    assert "not shown" in good["uncertainty"] or "was not shown" in good["uncertainty"]
    assert good["transcript_quote"] == "a massive 6"
    fake_quote = evs[1]
    assert fake_quote["transcript_quote"] == "" and "not found in the transcript" in fake_quote["uncertainty"]
    assert fake_quote["category"] == "funny_moment" and fake_quote["confidence"]["visual"] == "low"
    assert any("outside the window" in i for i in w0["issues"])
    top = svc.highlights(pid)[0]
    assert top["eff_start"] <= 40 <= top["eff_end"]
    assert top["evidence"]["ai"]["provider"] == "gemini" and top["hook_caption"] == "WHAT A SHOT"
    assert top["confidence"] == "high" and top["score"] > 70
    assert top["evidence"]["visual"][0]["observation"] == "Colour bars replace the pitch"


def test_quota_falls_back_to_ollama(setup):
    svc, pid, fake, ollama_calls = setup
    svc.grant_cloud_consent(pid, "frames")
    fake.script = ["good", "quota"]
    job, j = _run(svc, pid)
    assert j["state"] == "HIGHLIGHT_REVIEW", j["error_summary"]
    rep = svc.ai_report(pid)
    assert [w["provider"] for w in rep["windows"]] == ["gemini", "ollama"]
    assert any("Switched from gemini" in e for e in rep["plan_events"])
    assert len(ollama_calls) == 1


def test_quota_pauses_then_resume_reuses_finished_windows(setup):
    svc, pid, fake, ollama_calls = setup
    svc.grant_cloud_consent(pid, "frames")
    svc.save_project_settings(pid, {"ai.on_quota": "pause"})
    fake.script = ["good", "quota"]
    job, j = _run(svc, pid)
    assert j["state"] == "PAUSED" and "quota" in j["progress_note"].lower()
    st = {s["stage_name"]: s for s in svc.job_stages(job)}
    assert st["ai_windows"]["status"] == "PENDING" and "highlights" not in st
    assert len(fake.calls) == 2 and not ollama_calls
    # Resume: window 1 comes from the cache; only window 2 is requested again.
    job2 = svc.retry_job(pid)
    assert svc.runner.wait_idle(600)
    assert svc.jobs.get_job(job2)["state"] == "HIGHLIGHT_REVIEW"
    assert len(fake.calls) == 3
    rep = svc.ai_report(pid)
    assert rep["windows"][0].get("cached") and rep["windows"][1]["status"] == "ok"


def test_invalid_json_is_repaired_once(setup):
    svc, pid, fake, _ = setup
    svc.grant_cloud_consent(pid, "frames")
    fake.script = ["bad_json", "good", "bad_json", "bad_json"]
    job, j = _run(svc, pid)
    assert j["state"] == "HIGHLIGHT_REVIEW", j["error_summary"]
    rep = svc.ai_report(pid)
    assert rep["windows"][0]["status"] == "ok" and "repaired" in rep["windows"][0]["issues"][0]
    assert rep["windows"][1]["status"] == "invalid"  # two bad answers -> signals only, flagged for review
    assert any("needs your review" in i for i in rep["windows"][1]["issues"])


def test_gemini_video_chunks_need_video_consent_and_are_deleted(setup):
    svc, pid, fake, _ = setup
    fake.enable_files()
    svc.save_project_settings(pid, {"ai.gemini_mode": "video", "ai.gemini_video_max_windows": 1})
    svc.grant_cloud_consent(pid, "frames")  # frames consent is not enough for video
    _run(svc, pid)
    assert fake.uploads == []
    svc.grant_cloud_consent(pid, "video")
    job, j = _run(svc, pid)
    assert j["state"] == "HIGHLIGHT_REVIEW", j["error_summary"]
    # Only the most active window (the one with the event) is uploaded, as a low-res proxy, then deleted.
    assert len(fake.uploads) == 1 and fake.deleted == ["files/f1"] and fake.polls >= 1
    assert fake.uploads[0]["mime"] == "video/mp4"
    assert not list(svc.files(pid).video_chunks_dir.glob("*.mp4"))  # local proxy removed too
    rep = svc.ai_report(pid)
    assert [w["mode"] for w in rep["windows"]] == ["video", "images"]
    ev = json.loads((svc.files(pid).windows_dir / "w0000.json").read_text("utf-8"))["result"]["events"][0]
    assert ev["start_sec"] == pytest.approx(39.5) and ev["frame_times"] == [pytest.approx(40.5)]


def test_text_only_ollama_cannot_claim_visuals(setup):
    svc, pid, _fake, ollama_calls = setup
    svc.save_project_settings(pid, {"ai.provider": "ollama"})
    job, j = _run(svc, pid)
    assert j["state"] == "HIGHLIGHT_REVIEW", j["error_summary"]
    ev = json.loads((svc.files(pid).windows_dir / "w0000.json").read_text("utf-8"))["result"]["events"][0]
    assert ev["frame_times"] == [] and ev["visual_observation"] == "" and ev["confidence"]["visual"] == "none"
    assert ev["factors"]["visual_clarity"] == 0 and "could not see" in ev["uncertainty"]
    warns = [w for s in svc.job_stages(job) if s["stage_name"] == "ai_windows" for w in s["warnings"]]
    assert any("text-only" in w for w in warns)


def test_failed_windows_are_retried_on_the_next_run(setup):
    """Gemini 503 on every request -> windows fail. Running the analysis again must really retry them (the stage
    must not be reused just because its inputs are unchanged), and finished windows must not be re-requested."""
    svc, pid, fake, _ = setup
    svc.grant_cloud_consent(pid, "frames")
    svc.save_project_settings(pid, {"ai.ollama_model": "",  # no fallback: the failures must surface
                                    "ai.retry_delays_sec": [0, 0, 0]})  # 4 attempts per window, no waiting in tests
    fake.script = ["503"] * 8  # 2 windows x 4 attempts
    job, j = _run(svc, pid)
    assert j["state"] == "HIGHLIGHT_REVIEW", j["error_summary"]
    assert [w["status"] for w in svc.ai_report(pid)["windows"]] == ["failed", "failed"]
    attempts = {s["stage_name"]: s["attempt_count"] for s in svc.job_stages(job)}
    calls = len(fake.calls)
    job2, j2 = _run(svc, pid)
    assert j2["state"] == "HIGHLIGHT_REVIEW", j2["error_summary"]
    assert len(fake.calls) == calls + 2  # both windows requested again, once each
    assert [w["status"] for w in svc.ai_report(pid)["windows"]] == ["ok", "ok"]
    st = {s["stage_name"]: s["attempt_count"] for s in svc.job_stages(job2)}
    assert st["ai_windows"] == attempts["ai_windows"] + 1 and st["frames"] == attempts["frames"]
    # A third run with everything finished reuses the stage and sends nothing.
    _run(svc, pid)
    assert len(fake.calls) == calls + 2


def test_vram_is_freed_before_local_vision_and_model_unloaded_after(setup, monkeypatch):
    """6 GB GPU: free this process's GPU memory before a local vision model runs; unload it afterwards."""
    svc, pid, _fake, _ = setup
    order: list[str] = []
    from pipeline import transcribe as tr
    monkeypatch.setattr(tr, "free_gpu_memory", lambda: order.append("free") or {})

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.16"})
        if path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "vision"]})
        if path == "/api/chat":
            order.append("chat")
            assert json.loads(request.content)["messages"][1].get("images")  # frames really sent to the vision model
            return httpx.Response(200, json={"message": {"content": json.dumps({"events": []})}, "done_reason": "stop"})
        if path == "/api/generate":
            assert json.loads(request.content)["keep_alive"] == 0
            order.append("unload")
            return httpx.Response(200, json={})
        return httpx.Response(404, json={})
    monkeypatch.setattr(svc, "_ollama", lambda ai, model=None: OllamaProvider(
        ai["ollama_base_url"], ai["ollama_model"] if model is None else model, transport=httpx.MockTransport(handler)))
    svc.save_project_settings(pid, {"ai.provider": "ollama", "ai.ollama_model": "llava-test:7b"})
    job, j = _run(svc, pid)
    assert j["state"] == "HIGHLIGHT_REVIEW", j["error_summary"]
    assert order[0] == "free" and order[-1] == "unload" and order.count("chat") >= 1
    assert svc.ai_report(pid)["effective"] == ["ollama:llava-test:7b (images)"]
