"""Phase 4 end-to-end on the synthetic 90 s gameplay recording: Shorts (3 layouts, captions, selective re-render),
framing previews, Short approval, thumbnail frames/prompts, creator package (template and fake AI), edits."""

from __future__ import annotations

import io
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from pipeline import gameplay_analysis, media_probe, transcribe
from pipeline.job_manager import JobRunner
from pipeline.models import TrackRole
from pipeline.project_service import StudioService
from pipeline.shorts_builder import ShortSettings
from tests.conftest import requires_ffmpeg

pytestmark = [requires_ffmpeg,
              pytest.mark.skipif(not transcribe.is_model_cached("tiny"), reason="Whisper 'tiny' model not downloaded")]


def _stages(svc, job):
    return {s["stage_name"]: s for s in svc.job_stages(job)}


@pytest.fixture
def approved(studio_paths, media):
    svc = StudioService(studio_paths, runner=JobRunner(), configure_logging=False)
    src = studio_paths.raw_input / "Cricket 24" / "final over ü.mp4"
    src.parent.mkdir(parents=True)
    shutil.copy(media["gameplay"], src)
    pid = svc.create_project(src, "Cricket 24", "Sports Clutch / Match", "en", "Chasing 12 in the last over")
    svc.start_inspection(pid)
    assert svc.runner.wait_idle(120)
    svc.save_track_roles(pid, {0: TrackRole.COMBINED_MIX, 1: TrackRole.GAME, 2: TrackRole.MIC})
    svc.save_project_settings(pid, {"whisper.model": "tiny", "analysis.window_sec": 45, "ai.provider": "none",
                                    "shorts.quality": "fast"})
    svc.start_analysis(pid)
    assert svc.runner.wait_idle(600)
    top = svc.highlights(pid)[0]
    svc.update_highlight(top["id"], status="approved", hook="What a shot", title="Massive 6 in the final over")
    return svc, pid, top["id"]


def _run(svc, pid, target, **kw):
    job = svc.start_content(pid, target, **kw)
    assert svc.runner.wait_idle(900)
    return job, svc.jobs.get_job(job)


def test_shorts_render_rerender_only_changed_and_approve(approved):
    svc, pid, hid = approved
    h = svc.get_highlight(hid)
    dur = h["eff_end"] - h["eff_start"]
    with pytest.raises(ValueError, match="Render this Short"):
        svc.set_short_status(hid, "approved")
    job, j = _run(svc, pid, "shorts")
    st = _stages(svc, job)
    assert st[f"short_h{hid}"]["status"] == "COMPLETED", (j["error_summary"], st)
    out = svc.short_output(hid)
    assert out["validation_status"] == "passed" and out["file"].exists()
    info = media_probe.probe(out["file"])
    assert (info.video.width, info.video.height) == (1080, 1920) and info.duration_sec == pytest.approx(dur, abs=0.25)
    assert info.audio and out["settings"]["captions"]  # words exist (tiny transcription) -> captions burned
    assert (svc.files(pid).short_work(hid) / "captions.ass").read_text("utf-8").count("Dialogue:") > 3
    mix_attempts = st["mix"]["attempt_count"]

    # Change only the layout -> only this Short re-renders; the previous file is backed up.
    for layout, extra in (("center_crop", {"crop_x": 0.3}), ("split_stack", {"bottom_mode": "region"})):
        svc.save_short_settings(hid, ShortSettings(layout=layout, quality="fast", **extra))
        job2, j2 = _run(svc, pid, "shorts")
        st2 = _stages(svc, job2)
        assert st2[f"short_h{hid}"]["attempt_count"] >= 2 and st2["mix"]["attempt_count"] == mix_attempts
        o = svc.short_output(hid)
        assert o["validation_status"] == "passed" and o["settings"]["layout"] == layout, o["validation_details"]
    assert len(list((svc.files(pid).shorts_dir / "_previous").glob("short_h*.mp4"))) >= 2

    # A pack/title edit never re-renders video.
    before = _stages(svc, job2)[f"short_h{hid}"]["attempt_count"]
    _run(svc, pid, "pack")
    svc.save_pack_field(pid, "titles", ["New title"])
    job3, _ = _run(svc, pid, "shorts")
    assert _stages(svc, job3)[f"short_h{hid}"]["attempt_count"] == before

    svc.set_short_status(hid, "approved", "good")
    assert svc.short_review(hid) == {"status": "approved", "notes": "good"}


def test_preview_frame_and_hook_without_speech(approved):
    svc, pid, hid = approved
    s = ShortSettings(layout="center_crop")
    p = svc.short_preview_frame(hid, 2.0, s)
    with Image.open(p) as im:
        assert im.size == (1080, 1920)
    # Unapproved highlights cannot get Short settings.
    other = [h for h in svc.highlights(pid) if h["id"] != hid]
    if other:
        with pytest.raises(ValueError, match="Only approved"):
            svc.save_short_settings(other[0]["id"], s)


def test_thumbnails_and_prompts(approved):
    svc, pid, hid = approved
    job, j = _run(svc, pid, "thumbnails")
    assert j["error_summary"] is None, j["error_summary"]
    t = svc.thumbnails(pid)
    frames = t["frames"]
    assert 1 <= len(frames) <= 5 and all(fr["path"] and fr["path"].suffix == ".png" for fr in frames)
    assert all(fr["width"] == 640 for fr in frames)  # source is 640x360: never upscaled
    thumbs = [np.asarray(Image.open(fr["path"]).convert("L").resize((32, 18)), dtype=np.float32) for fr in frames]
    for i in range(len(thumbs)):
        for k in range(i + 1, len(thumbs)):
            assert np.mean(np.abs(thumbs[i] - thumbs[k])) >= 6.0  # no near-identical frames
    p = t["prompts"][str(hid)]
    assert "Cricket 24" in p["reference_edit"] and "bottom-right" in p["from_scratch"].lower()
    assert p["text"] == "WHAT A SHOT" and "Massive 6 in the final over" in p["from_scratch"]
    assert not p["needs_user_words"]


def test_template_pack_edits_survive_regeneration(approved):
    svc, pid, hid = approved
    job, j = _run(svc, pid, "assets")
    st = _stages(svc, job)
    assert st["thumbnails"]["status"] == "COMPLETED" and st["creator_pack"]["status"] == "COMPLETED", st
    pack = svc.creator_pack(pid)
    assert pack["source"]["provider"] == "template" and len(pack["titles"]) == 5
    assert pack["titles"][0] == "Massive 6 in the final over"  # the creator's own title comes first
    assert any("[" in t for t in pack["titles"])  # placeholders, not invented text
    assert pack["shorts"][0]["highlight_id"] == hid and pack["shorts"][0]["hook"] == "What a shot"
    assert pack["thumbnail_prompts"]["text"] == "WHAT A SHOT"
    files = svc.files(pid)
    assert json.loads(files.pack_json.read_text("utf-8"))["titles"] == pack["titles"]
    assert "VOICEOVER SCRIPT" in files.pack_txt.read_text("utf-8")
    # Edit, regenerate, edit kept; discard -> back to the draft.
    svc.save_pack_field(pid, "description", "My own description")
    with pytest.raises(ValueError):
        svc.save_pack_field(pid, "titles", ["x" * 101])
    svc.start_content(pid, "pack")
    assert svc.runner.wait_idle(300)
    assert svc.creator_pack(pid)["description"] == "My own description"
    assert svc.creator_pack(pid)["edited_fields"] == ["description"]
    svc.discard_pack_edits(pid)
    assert svc.creator_pack(pid)["description"] != "My own description"


class _StubProvider:
    name, model, cloud = "stub", "stub-1", False

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def generate_json(self, system, prompt, schema, images=None, video=None):
        self.prompts.append(prompt)
        return self.answers.pop(0)


def test_ai_pack_is_validated(approved, monkeypatch):
    svc, pid, hid = approved
    ev = svc.analysis_events(pid)
    key = next(e["event_key"] for e in ev if e["start_sec"] <= 40.5 <= e["end_sec"] + 1)
    start = next(e["start_sec"] for e in ev if e["event_key"] == key)
    answer = {"titles": ["Massive 6 to win", "We needed 12", "A 99-run miracle", "Massive 6 to win", "Final over"],
              "opening_hook": "Twelve needed, one over left.",
              "script": [{"time_sec": start - 10, "text": "Here comes the six", "event_key": key},
                         {"time_sec": 5, "text": "Intro", "event_key": ""},
                         {"time_sec": 50, "text": "Unknown thing", "event_key": "nope"}],
              "shorts": [{"highlight_id": hid, "title": "AI title", "hook": "AI hook"},
                         {"highlight_id": 99999, "title": "x", "hook": "y"}],
              "description": "A last-over chase.", "tags": ["#cricket", "Cricket 24", "cricket"],
              "pinned_comment": "", "style_rationale": "clutch", "uncertainty": []}
    stub = _StubProvider(["{broken", json.dumps(answer)])
    plan = gameplay_analysis.ProviderPlan(gameplay_analysis.ProviderSlot(stub, "text"))
    monkeypatch.setattr(svc, "build_provider_plan", lambda pid_, s: (plan, {"provider": "stub", "model": "stub-1",
                                                                           "mode": "text", "fallback": ""}, []))
    job, j = _run(svc, pid, "pack")
    pack = svc.creator_pack(pid)
    assert pack["source"]["provider"] == "stub" and "repaired" in pack["uncertainty"][0]
    assert "Cricket 24" in stub.prompts[0] and "Chasing 12" in stub.prompts[0] and "What a shot" in stub.prompts[0]
    assert pack["titles"][:2] == ["Massive 6 to win", "We needed 12"] and len(pack["titles"]) == 4 + 1
    assert any("99" in f for f in pack["number_flags"])  # invented number flagged
    assert not any("Title 2" in f for f in pack["number_flags"])  # 12 is in the evidence
    six = next(s for s in pack["script"] if s["text"] == "Here comes the six")
    assert six["time_sec"] == pytest.approx(start) and six["uncertain"]  # moved: was before the event
    assert next(s for s in pack["script"] if s["text"] == "Unknown thing")["uncertain"]
    assert [s["highlight_id"] for s in pack["shorts"]] == [hid]
    assert pack["shorts"][0]["title"] == "Massive 6 in the final over"  # the creator's words win
    assert pack["tags"] == ["cricket", "Cricket 24"]


def test_user_thumbnail_import(approved):
    svc, pid, _ = approved
    buf = io.BytesIO()
    Image.new("RGB", (1280, 720), (200, 30, 30)).save(buf, "PNG")
    buf.seek(0)
    res = svc.import_user_thumbnail(pid, buf, "mine.png")
    assert res["path"].exists() and res["preview"].exists() and res["warnings"] == []
    buf2 = io.BytesIO()
    Image.new("RGB", (800, 800)).save(buf2, "JPEG")
    buf2.seek(0)
    res2 = svc.import_user_thumbnail(pid, buf2, "square.jpg")
    assert any("16:9" in w for w in res2["warnings"]) and any("1280x720" in w for w in res2["warnings"])
    assert list((svc.files(pid).user_thumbnail_dir / "_previous").glob("thumbnail_user*"))
    with pytest.raises(ValueError):
        svc.import_user_thumbnail(pid, io.BytesIO(b"not an image"), "bad.png")
