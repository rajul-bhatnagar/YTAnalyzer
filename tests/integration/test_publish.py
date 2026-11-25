"""Phase 5: final checklist, approval and the YouTube upload with a FAKE YouTube service (no network, no upload)."""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import pytest
from googleapiclient.errors import HttpError

from pipeline import transcribe
from pipeline.job_manager import JobRunner
from pipeline.models import TrackRole
from pipeline.project_service import StudioService
from tests.conftest import requires_ffmpeg

pytestmark = [requires_ffmpeg,
              pytest.mark.skipif(not transcribe.is_model_cached("tiny"), reason="Whisper 'tiny' model not downloaded")]


def _http_error(code, text):
    return HttpError(SimpleNamespace(status=code, reason=text), text.encode())


class FakeYouTube:
    def __init__(self, script):
        self.script = list(script)  # per next_chunk call: "ok", "progress", "500", "quota"
        self.bodies, self.thumbs = [], []

    def videos(self):
        return self

    def thumbnails(self):
        return SimpleNamespace(set=lambda videoId, media_body: SimpleNamespace(
            execute=lambda: self.thumbs.append(videoId)))

    def insert(self, part, body, media_body):
        assert part == "snippet,status" and media_body.resumable()
        self.bodies.append(body)
        return SimpleNamespace(next_chunk=self._next)

    def _next(self):
        a = self.script.pop(0)
        if a == "500":
            raise _http_error(500, "backend error")
        if a == "quota":
            raise _http_error(403, "quotaExceeded: The request cannot be completed because you have exceeded your quota")
        if a == "progress":
            return SimpleNamespace(progress=lambda: 0.5), None
        return None, {"id": "VID123", "status": {"privacyStatus": self.bodies[-1]["status"]["privacyStatus"]}}


@pytest.fixture
def ready(studio_paths, media):
    svc = StudioService(studio_paths, runner=JobRunner(), configure_logging=False)
    src = studio_paths.raw_input / "G" / "rec.mp4"
    src.parent.mkdir(parents=True)
    shutil.copy(media["gameplay"], src)
    pid = svc.create_project(src, "G", "Custom", "en")
    svc.start_inspection(pid)
    assert svc.runner.wait_idle(120)
    svc.save_track_roles(pid, {0: TrackRole.COMBINED_MIX, 1: TrackRole.GAME, 2: TrackRole.MIC})
    svc.save_project_settings(pid, {"whisper.model": "tiny", "analysis.window_sec": 45, "analysis.window_overlap_sec": 0, "ai.provider": "none",
                                    "shorts.quality": "fast"})
    svc.start_analysis(pid)
    assert svc.runner.wait_idle(600)
    hid = svc.highlights(pid)[0]["id"]
    svc.update_highlight(hid, status="approved", hook="What a shot", title="Big moment")
    for t in ("shorts", "pack"):
        svc.start_content(pid, t)
        assert svc.runner.wait_idle(900)
    return svc, pid, hid


def test_checklist_blocks_then_approves(ready):
    svc, pid, hid = ready
    chk = svc.final_check(pid)
    names = {i["name"]: i["status"] for i in chk["items"]}
    assert names["Creator package"] == "block"  # template description still has [placeholders]
    assert names["Something to publish"] == "block" and names["Shorts not reviewed"] == "warn"
    with pytest.raises(ValueError):
        svc.approve_final(pid, True)
    svc.save_pack_field(pid, "description", "A real description.")
    svc.set_short_status(hid, "approved")
    chk = svc.final_check(pid)
    assert not chk["blocking"], chk["items"]
    assert chk["needs_ack"]  # template uncertainty must be acknowledged
    with pytest.raises(ValueError, match="flagged"):
        svc.approve_final(pid, False)
    svc.approve_final(pid, True)
    assert svc.is_final_approved(pid)
    assert svc.jobs.get_job(svc.latest_job(pid)["id"])["state"] == "APPROVED"
    # Changing an approved file invalidates the approval.
    out = svc.short_output(hid)["file"]
    out.write_bytes(out.read_bytes() + b"x")
    assert not svc.is_final_approved(pid)


def test_upload_guards_success_retry_and_quota(ready):
    svc, pid, hid = ready
    svc.save_pack_field(pid, "description", "A real description.")
    svc.set_short_status(hid, "approved")
    svc.approve_final(pid, True)
    key = f"short_{hid}"
    args = dict(title="Big moment", description="d", tags=["g"])
    with pytest.raises(ValueError, match="made for kids"):
        svc.start_upload(pid, key, privacy="private", made_for_kids=None, confirmed=True, **args)
    with pytest.raises(ValueError, match="confirmation"):
        svc.start_upload(pid, key, privacy="private", made_for_kids=False, confirmed=False, **args)
    with pytest.raises(ValueError, match="visible to others"):
        svc.start_upload(pid, key, privacy="public", made_for_kids=False, confirmed=True, **args)
    with pytest.raises(ValueError, match="not part of the approved"):
        svc.start_upload(pid, "rough_cut", privacy="private", made_for_kids=False, confirmed=True, **args)

    fake = FakeYouTube(["progress", "500", "ok"])
    svc._youtube_factory = lambda: fake
    import pipeline.youtube_draft as yd
    orig = yd.upload
    yd.upload = lambda *a, **k: orig(*a, **{**k, "sleep": lambda s: None})
    try:
        svc.start_upload(pid, key, privacy="private", made_for_kids=False, confirmed=True, **args)
        assert svc.runner.wait_idle(120)
        up = svc.uploads(pid)[-1]
        assert up["ok"] and up["video_id"] == "VID123" and up["result_privacy"] == "private"
        assert up["url"] == "https://www.youtube.com/watch?v=VID123"
        body = fake.bodies[0]
        assert body["status"] == {"privacyStatus": "private", "selfDeclaredMadeForKids": False}
        assert body["snippet"]["title"] == "Big moment" and body["snippet"]["categoryId"] == "20"
        assert svc.jobs.get_job(svc.latest_job(pid)["id"])["state"] == "DRAFT_UPLOADED"

        svc._youtube_factory = lambda: FakeYouTube(["quota"])
        svc.start_upload(pid, key, privacy="private", made_for_kids=False, confirmed=True, **args)
        assert svc.runner.wait_idle(120)
        up = svc.uploads(pid)[-1]
        assert not up["ok"] and "quota" in up["error"].lower()
        assert svc.is_final_approved(pid)  # a failed upload never undoes the approval
    finally:
        yd.upload = orig
