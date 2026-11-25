"""Unit tests for the upgrade: style profiles, window overlap + gating, cross-window merge, Short fact-check,
dead-air planning, thumbnail words/composer, VRAM hook + Ollama unload, editor URIs, HUD OCR change events."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from pipeline import (event_detection, gameplay_analysis as ga, highlight_scoring as hs, hud_ocr, short_edit,
                      short_factcheck as fc, shorts_builder as sb, style_profiles, thumbnail_prep as tp,
                      timeline_export, transcribe)
from pipeline.ai_providers.ollama_provider import OllamaProvider
from pipeline.models import CONTENT_STYLES, AudioStreamInfo, MediaInfo, VideoStreamInfo


# --- style profiles ------------------------------------------------------------------------------------------------
def test_every_style_has_a_game_agnostic_profile():
    assert set(style_profiles.PROFILES) == set(CONTENT_STYLES)
    assert set(style_profiles.LAYOUT_KEYS) == set(sb.LAYOUTS)
    for st in ("Horror / Funny Reactions", "Story / Mod Experiment"):
        p = style_profiles.profile(st)
        assert p.shorts_dead_air_mode == "none" and p.hud_ocr_weight == 0.0
    for st in ("Sports Clutch / Match", "Shooter / High-Skill Plays"):
        assert style_profiles.profile(st).hud_ocr_weight > 0
    assert style_profiles.profile("Unknown style").style == "Custom"


def test_profile_overrides():
    st = "Sports Clutch / Match"
    empty = {"shorts": {"layout": "", "dead_air_mode": ""}, "analysis": {"hud_ocr_weight": None}}
    assert style_profiles.resolve(st, empty) == style_profiles.profile(st)
    own = {"shorts": {"layout": "blur_fit", "dead_air_mode": "none"}, "analysis": {"hud_ocr_weight": 2.0}}
    r = style_profiles.resolve(st, own)
    assert (r.default_shorts_layout, r.shorts_dead_air_mode, r.hud_ocr_weight) == ("blur_fit", "none", 1.0)
    bad = {"shorts": {"layout": "nope", "dead_air_mode": "fast"}, "analysis": {"hud_ocr_weight": 0}}
    r = style_profiles.resolve(st, bad)
    assert r.default_shorts_layout == "split_stack" and r.shorts_dead_air_mode == "trim" and r.hud_ocr_weight == 0.0


# --- windows: overlap, ranking, ranges ------------------------------------------------------------------------------
def test_overlapping_windows_cover_boundaries():
    w = ga.build_windows(600, 180, overlap_sec=30)
    assert [(x.start, x.end) for x in w] == [(0, 180), (150, 330), (300, 480), (450, 600)]
    # A moment straddling the old 180 s boundary sits whole inside the second window.
    assert any(x.start <= 175 and 185 <= x.end for x in w)
    assert [(x.start, x.end) for x in ga.build_windows(600, 180)] == [(0, 180), (180, 360), (360, 540), (540, 600)]


def test_rank_windows_top_n_pinned_and_done():
    w = ga.build_windows(1800, 180, overlap_sec=30)  # 12 windows
    activity = {x.idx: float(x.idx % 5) for x in w}
    top3 = ga.rank_windows(w, activity, 3)
    assert len(top3) == 3 and all(activity[i] == 4.0 or activity[i] == 3.0 for i in top3)
    pinned = ga.rank_windows(w, activity, 3, pinned=[(0, 10)], always={7})
    assert 0 in pinned and 7 in pinned and top3 <= pinned
    assert ga.rank_windows(w, activity, 0) == {x.idx for x in w}


def test_parse_ranges():
    assert ga.parse_ranges("12:30-15:00, 1:02:00-1:05:00") == [(750.0, 900.0), (3720.0, 3900.0)]
    assert ga.parse_ranges("") == []
    for bad in ("12:30", "15:00-12:30", "a-b"):
        with pytest.raises(ValueError):
            ga.parse_ranges(bad)


def test_skipped_windows_send_nothing(tmp_path):
    calls = []
    prov = SimpleNamespace(name="stub", model="m", cloud=False,
                           generate_json=lambda *a, **k: calls.append(1) or json.dumps({"events": []}))
    plan = ga.ProviderPlan(ga.ProviderSlot(prov, "text"))
    windows = ga.build_windows(400, 180)
    ev = [ga.WindowEvidence(w, [], [], [], {"game": "G", "style": "Custom", "topic": "", "language": "en"})
          for w in windows]
    rep = ga.analyse_windows(ev, plan, tmp_path, 400, Path, {"max_frames_per_request": 4}, ga.Budget(10, 0.0),
                             lambda *a: None, lambda: False, lambda: False, selected={1})
    assert [w["status"] for w in rep["windows"]] == ["skipped", "ok", "skipped"] and len(calls) == 1


def test_overlapping_windows_merge_one_moment():
    def res(cat, s, e):
        return {"provider": "p", "model": "m", "mode": "images", "result": {"events": [{
            "start_sec": s, "end_sec": e, "category": cat, "description": cat, "visual_observation": "",
            "frame_times": [], "audio_observation": "", "transcript_quote": "", "confidence": {}, "uncertainty": "",
            "factors": {}, "hook_caption": "", "suggested_title": ""}]}}
    fused = hs.fuse([], {0: res("clutch", 170, 182), 1: res("achievement", 171, 183)})
    assert len(fused) == 1


# --- per-Short fact-check -------------------------------------------------------------------------------------------
EV = {"window": "Super Over: India need 8 runs from 3 balls. Massive six.",
      "all": "Super Over: India need 8 runs from 3 balls. Massive six. Wicket falls at 12:40. 199 runs total",
      "hooks": ["8 OFF 3 BALLS", "WICKET!"]}


def test_hook_inside_clip_passes():
    r = fc.check_short("8 off 3 balls", "Super Over six", EV)
    assert r["ok"] and r["messages"] == []


def test_out_of_window_and_unsupported_claims_are_flagged():
    r = fc.check_short("Wicket drama", "99 runs record", EV)
    kinds = {(i["kind"], i["token"]) for i in r["issues"]["hook"] + r["issues"]["title"]}
    assert ("outside", "wicket") in kinds and ("unsupported_number", "99") in kinds
    assert any("not inside this clip" in m for m in r["messages"])
    assert r["suggestions"] == ["8 OFF 3 BALLS"]  # the in-window AI hook that passes the same check


def test_style_words_are_not_claims():
    assert fc.check_short("INSANE CLUTCH", "", EV)["ok"]


# --- dead air -------------------------------------------------------------------------------------------------------
def _sig(active_from: float, active_to: float, dur: float = 60.0):
    grid = np.arange(0, dur, 0.5, dtype=np.float32)
    motion = np.where((grid >= active_from) & (grid < active_to), 6.0, 0.2).astype(np.float32)
    return {"grid": grid, "motion": motion, "speech": None, "game_db": None, "meta": {"hop_sec": 0.5}}


def test_trim_lead_and_tail():
    segs, notes = short_edit.plan_segments(_sig(20, 38), 10.0, 50.0, 25.0, "trim")
    assert segs == [(9.0, 29.5, 1.0)] and len(notes) == 2
    assert short_edit.output_length(segs) == 20.5


def test_speedup_lead_and_guards():
    segs, _ = short_edit.plan_segments(_sig(20, 38), 10.0, 50.0, 25.0, "speedup", speed=2.5, min_total=10)
    assert segs[0] == (0.0, 9.0, 2.5) and segs[1][2] == 1.0
    whole = [(0.0, 40.0, 1.0)]
    assert short_edit.plan_segments(_sig(20, 38), 10.0, 50.0, 25.0, "none")[0] == whole
    assert short_edit.plan_segments(_sig(20, 38), 10.0, 50.0, 25.0, "trim", min_total=30)[0] == whole
    assert short_edit.plan_segments(_sig(0, 0), 10.0, 50.0, 25.0, "trim")[0] == whole  # nothing active


def test_caption_remap_and_atempo():
    segs = [(0.0, 9.0, 2.5), (9.0, 29.5, 1.0)]
    words = [{"start": 5.0, "end": 5.5, "word": "a"}, {"start": 12.0, "end": 12.4, "word": "b"},
             {"start": 35.0, "end": 35.5, "word": "trimmed"}]
    out = short_edit.remap_words(words, segs)
    assert [w["word"] for w in out] == ["a", "b"] and out[0]["start"] == 2.0 and out[1]["start"] == 6.6
    assert short_edit.atempo_chain(2.5) == "atempo=2.0,atempo=1.2500" and short_edit.atempo_chain(2.0) == "atempo=2.0000"


# --- thumbnails -----------------------------------------------------------------------------------------------------
def test_thumbnail_words_rules():
    assert tp.validate_words("super over win") == ("SUPER OVER WIN", None)
    assert tp.validate_words("one")[0] is None and tp.validate_words("a b c d")[0] is None
    assert tp.validate_words("win in")[0] is None and tp.validate_words("8 off")[0] is None
    assert tp.suggest_words("What a shot from") == "WHAT A SHOT" and tp.suggest_words("into the") == ""


def test_prompts_are_text_free_and_follow_mood():
    h = {"suggested_title": "Last-ball six", "hook_caption": "Win on the last ball", "outcome_mood": "clutch_tension"}
    p = tp.build_prompts("Any Game", "Custom", h, 12.0)
    for k in ("reference_edit", "from_scratch"):
        assert "Do not add any text" in p[k] and "darker vignette" in p[k] and "PC game" not in p[k]
    assert p["text"] == ""  # "WIN ON THE" minus trailing words leaves 1 word: no suggestion, never a fragment
    assert p["mood"] == "clutch_tension"


def test_compose_keeps_timestamp_corner(tmp_path):
    from PIL import Image
    src = tmp_path / "frame.png"
    Image.new("RGB", (1920, 1080), (20, 90, 40)).save(src)
    out = tp.compose_thumbnail(src, "big win today", "win_hype", tmp_path / "t.png")
    with Image.open(out) as im:
        assert im.size == (1280, 720)
        assert (255, 214, 0) in {im.getpixel((x, y)) for x in range(40, 400, 3) for y in range(40, 200, 3)}
        corner = {im.getpixel((x, y)) for x in range(1050, 1280, 10) for y in range(560, 720, 10)}
        assert corner == {(20, 90, 40)}
    badge = tp.safe_area_preview(out, tmp_path / "b.png", badge=True)
    with Image.open(badge) as im:
        assert im.getpixel((1200, 680)) != (20, 90, 40)
    with pytest.raises(ValueError):
        tp.compose_thumbnail(src, "win of", "", tmp_path / "x.png")


# --- VRAM -----------------------------------------------------------------------------------------------------------
def test_free_gpu_memory_is_safe_on_cpu():
    assert "gc_objects" in transcribe.free_gpu_memory()


def test_ollama_unload_sends_keep_alive_zero():
    seen = []

    def handler(r):
        seen.append((r.url.path, json.loads(r.content or b"{}")))
        return httpx.Response(200, json={})
    assert OllamaProvider("http://x", "llava:7b", transport=httpx.MockTransport(handler)).unload()
    assert seen == [("/api/generate", {"model": "llava:7b", "keep_alive": 0})]

    def down(r):
        raise httpx.ConnectError("down", request=r)
    assert OllamaProvider("http://x", "m", transport=httpx.MockTransport(down)).unload() is False


# --- VFR shift and editor URIs ----------------------------------------------------------------------------------------
def test_video_seek_shift():
    from pipeline.media_probe import video_seek_shift
    info = MediaInfo(path="x", duration_sec=10, video=VideoStreamInfo(stream_index=0, start_time=0.5),
                     audio=[AudioStreamInfo(stream_index=1, audio_index=0, start_time=0.0)])
    assert video_seek_shift(info) == 0.5
    info.video.start_time = 0.0
    assert video_seek_shift(info) == 0.0
    info.video.start_time = None
    assert video_seek_shift(info) == 0.0


def test_fcp7_and_edl_use_localhost_file_uris(tmp_path):
    src = tmp_path / "My Game ü" / "rec 1.mp4"
    src.parent.mkdir()
    src.write_bytes(b"x")
    url = timeline_export.fcp_pathurl(src)
    assert url.startswith("file://localhost/") and "%20" in url and "%C3%BC" in url and " " not in url
    assert url.endswith("/rec%201.mp4")
    clips = [{"source_in": 0, "source_out": 5, "record_in": 0, "record_out": 5}]
    xml = timeline_export.build_fcp7_xml("T", clips, 30, 1920, 1080, src, 10, src)
    assert f"<pathurl>{url}</pathurl>" in xml
    edl = timeline_export.build_edl("T", clips, 30, src.name, "mix.wav", url, url)
    assert f"* SOURCE FILE: {url}" in edl and len(timeline_export.parse_edl(edl)) == 2


# --- HUD OCR ----------------------------------------------------------------------------------------------------------
def _ocr(seq):
    return [{"t": float(i), "bands": [b], "texts": b} for i, b in enumerate(seq)]


def test_hud_changes_become_weighted_events():
    ocr = _ocr([["IND 0-0 0.1"], ["IND 0-0 0.1"], ["IND 6-0 0.2"], ["IND 6-0 0.2"], ["IND 12-0 0.3"]])
    ev = hud_ocr.change_events(ocr, 0.6)
    assert [a["peak"] for a in ev] == [2.0, 4.0] and all(a["strength"] == 0.6 for a in ev)
    assert hud_ocr.change_events(ocr, 0.0) == []
    clusters = event_detection.cluster(ev, [], 1.0, 10.0)
    assert clusters[0]["modalities"] == ["hud"] and "OCR can misread" in clusters[0]["uncertainty"]


def test_clock_bands_and_empty_ocr_are_ignored():
    clock = _ocr([[f"00:{s:02d}"] for s in range(10)])
    assert hud_ocr.change_events(clock, 1.0) == []
    assert hud_ocr.change_events(_ocr([[], [], []]), 1.0) == []


def test_ocr_reader_parses_engine_output_and_survives_failure(monkeypatch):
    monkeypatch.setattr(hud_ocr, "_engine_obj", lambda img: ([[[0, 0], "IND 8-0", 0.9], [[0, 0], " ", 0.1]], 0.01))
    assert hud_ocr.read_image(np.zeros((10, 10, 3), np.uint8)) == ["IND 8-0"]

    def boom(img):
        raise RuntimeError("model missing")
    monkeypatch.setattr(hud_ocr, "_engine_obj", boom)
    assert hud_ocr.read_image(np.zeros((10, 10, 3), np.uint8)) == []
