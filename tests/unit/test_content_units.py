"""Unit tests for Phase 4 logic: captions, layout maths, Shorts rendering edge cases, metadata validation, prompts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import captions as cp
from pipeline import content_stages, shorts_builder as sb, thumbnail_prep as tp
from pipeline import youtube_metadata as ym
from tests.conftest import requires_ffmpeg


# --- captions ---------------------------------------------------------------------------------------------------
def test_ass_color_is_bgr_and_validated():
    assert cp.ass_color("#FFD400") == "&H0000D4FF"
    assert cp.ass_color("#000000", 0x60) == "&H60000000"
    with pytest.raises(ValueError):
        cp.ass_color("yellow")


def test_ass_escape_blocks_override_tags_and_newlines():
    assert cp.ass_escape("a{\\b1}b\nc\\N") == "a(⧵b1)b c⧵N"


def test_clip_words_converts_and_clamps():
    words = [{"start": 9.0, "end": 10.5, "word": " hi"}, {"start": 12, "end": 13, "word": " there"},
             {"start": 30, "end": 31, "word": " out"}]
    out = cp.clip_words(words, 10.0, 20.0)
    assert out == [{"start": 0.0, "end": 0.5, "word": "hi"}, {"start": 2.0, "end": 3.0, "word": "there"}]


def test_chunking_and_word_highlight_timing():
    words = [{"start": i * 0.4, "end": i * 0.4 + 0.3, "word": w} for i, w in enumerate("one two three four five".split())]
    chunks = cp.chunk_words(words, 4, 2.5)
    assert [len(c) for c in chunks] == [4, 1]
    ass, info = cp.build_ass(words, 3.0, cp.CaptionStyle())
    dialogues = [ln for ln in ass.splitlines() if ln.startswith("Dialogue")]
    assert info["cues"] == 5 and len(dialogues) == 5
    assert "{\\c&H0000D4FF&}ONE{\\c&H00FFFFFF&} TWO THREE FOUR" in dialogues[0]
    assert cp.validate_ass_timing(ass, 3.0) == []
    assert cp.validate_ass_timing(ass, 1.0)  # cues beyond a 1 s clip are reported


@pytest.mark.parametrize("mode,has_words,shown", [("no_speech", False, True), ("no_speech", True, False),
                                                  ("always", True, True), ("never", False, False)])
def test_hook_modes(mode, has_words, shown):
    words = [{"start": 0.5, "end": 1.0, "word": "yes"}] if has_words else []
    style = cp.CaptionStyle(hook_mode=mode, hook_enabled=mode != "never")
    ass, info = cp.build_ass(words, 10.0, style, hook="Watch this {")
    assert info["hook_shown"] is shown
    if shown:
        assert "WATCH THIS (" in ass


# --- layout maths -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("w,h", [(1920, 1080), (1440, 1080), (2560, 1080), (1080, 1920), (3840, 2160), (640, 360)])
def test_crop_matches_aspect_inside_frame(w, h):
    for aspect in (1080 / 1920, 1080 / 960):
        for pos in (0.0, 0.5, 1.0):
            cw, ch, x, y = sb._crop_to_aspect(w, h, aspect, pos)
            assert 0 <= x and x + cw <= w and 0 <= y and y + ch <= h
            assert cw % 2 == 0 and ch % 2 == 0
            assert abs(cw / ch - aspect) < 0.01  # same shape as the target -> scaling never stretches


@pytest.mark.parametrize("layout", list(sb.LAYOUTS))
def test_video_graph_outputs_full_frame(layout):
    g = sb.video_graph(sb.ShortSettings(layout=layout, bottom_mode="region"), 1920, 1080, "captions.ass")
    assert g.endswith("ass=captions.ass,format=yuv420p[v]") and "[lay]setsar=1" in g
    if layout == "center_crop":
        assert "crop=606:1080:657:0,scale=1080:1920" in g


def test_output_fps_is_capped():
    assert sb.output_fps(144) == 60 and sb.output_fps(29.97) == 29.97 and sb.output_fps(None) == 30


@requires_ffmpeg
def test_silent_short_and_black_detection(tmp_path):
    from pipeline import ffmpeg_tools as fx
    src = tmp_path / "src ü.mp4"
    fx.ffmpeg(["-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=6", "-c:v",
               "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)])
    out = tmp_path / "out.mp4"
    sb.render_short(src, None, 1.0, 5.0, sb.ShortSettings(quality="fast"), (640, 360), 30.0, "libx264",
                    tmp_path / "w", None, out)
    errors, det = sb.validate_short(out, 4.0, src, 1.0, False, tmp_path / "w")
    assert errors == [] and not det["probe"]["audio"]
    black = tmp_path / "black.mp4"
    fx.ffmpeg(["-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:size=1080x1920:rate=30:duration=4",
               "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(black)])
    errors, _ = sb.validate_short(black, 4.0, src, 1.0, False, tmp_path / "w")
    assert any("black" in e for e in errors)


# --- metadata ---------------------------------------------------------------------------------------------------------
CTX = {"game": "Cricket 24", "style": "Sports Clutch / Match", "topic": "need twelve runs", "language": "en",
       "duration": 90.0, "transcript": [{"start": 41, "end": 44, "text": "What a shot, a massive six"}],
       "events": [{"key": "e1", "start": 40.0, "end": 41.0, "category": "clutch", "description": "Big hit",
                   "chosen": True}],
       "highlights": [{"id": 7, "start": 30, "end": 60, "event": 40, "title": "", "hook": "", "what": "Big hit"}]}


def test_number_guard_uses_evidence_and_number_words():
    allowed = ym.allowed_numbers(CTX)
    assert {"24", "6", "12"} <= allowed  # game name; "six" and "twelve" spoken/written as words
    assert ym.unsupported_numbers("A 99-run chase in 2026, 6 needed", allowed) == ["2026", "99"]


def _draft(**kw):
    d = {"titles": ["A", "a", "B"], "script": [{"time_sec": 30, "text": "early", "event_key": "e1"},
                                               {"time_sec": 200, "text": "late", "event_key": ""}],
         "shorts": [{"highlight_id": 7, "title": "t", "hook": "h"}, {"highlight_id": 8, "title": "x", "hook": "y"}],
         "tags": ["#one", "One", "two"]}
    d.update(kw)
    return json.dumps(d)


def test_validate_draft_fixes_and_reports():
    d, issues = ym.validate_draft(_draft(), CTX)
    assert d.titles == ["A", "B", "[Title idea 3]", "[Title idea 4]", "[Title idea 5]"]
    assert [(x.time_sec, x.uncertain) for x in d.script] == [(40.0, True)]
    assert [s.highlight_id for s in d.shorts] == [7] and d.tags == ["one", "two"]
    assert any("before it happened" in i for i in issues) and any("outside" in i for i in issues)
    with pytest.raises(ym.InvalidPack):
        ym.validate_draft("{nope", CTX)


def test_template_has_five_titles_and_placeholders():
    for hl in ([], [{"id": 1, "start": 0, "end": 30, "event": 5, "title": "Mine", "hook": "", "what": ""}]):
        d = ym.template_draft({**CTX, "highlights": hl})
        assert len(d.titles) == 5 and any("[" in t for t in d.titles)


def test_edits_override_and_text_export():
    d = ym.template_draft(CTX)
    pack = ym.assemble(d, CTX, {"provider": "template"}, [], None)
    eff = ym.apply_edits(pack, {"description": "mine", "bogus": 1})
    assert eff["description"] == "mine" and eff["edited_fields"] == ["description"] and "bogus" not in eff
    txt = ym.to_text(eff)
    assert "TITLE OPTIONS" in txt and "mine" in txt


def test_video_time_mapper():
    dec = {"segments": [{"keep": True, "source_start": 0, "source_end": 15, "output_start": 0},
                        {"keep": False, "source_start": 15, "source_end": 45, "output_start": None},
                        {"keep": True, "source_start": 45, "source_end": 60, "output_start": 15}]}
    m = content_stages.video_time_mapper(dec)
    assert m(10) == 10 and m(20) is None and m(50) == 20
    assert content_stages.video_time_mapper(None)(33) == 33


# --- thumbnails -----------------------------------------------------------------------------------------------------
def test_prompt_text_and_placeholders():
    h = {"suggested_title": "", "hook_caption": "what a shot, wow!", "rationale": "Measured: x."}
    p = tp.build_prompts("Game Y", "Custom", h, 40.0)
    assert p["text"] == "WHAT A SHOT" and "bottom-right" in p["reference_edit"] and "Game Y" in p["from_scratch"]
    p2 = tp.build_prompts("Game Y", "Custom", {"suggested_title": "", "hook_caption": "", "rationale": "Measured: x."},
                          None)
    assert p2["needs_user_words"] and "[describe the moment" in p2["from_scratch"] and "Do not add any text" in p2["from_scratch"]


def test_candidate_times_stay_in_clip_and_dedupe():
    h = {"id": 1, "eff_start": 30.0, "eff_end": 60.0, "event_sec": 40.0,
         "evidence": {"visual": [{"t": 40.1}, {"t": 80.0}]}}
    times = [t for t, _, _ in tp.candidate_times([h], 90.0)]
    assert all(29 <= t <= 61 for t in times) and 80.0 not in times
    assert all(b - a >= 0.3 for a, b in zip(times, times[1:]))


def test_user_thumbnail_validation(tmp_path: Path):
    from PIL import Image
    ok = tmp_path / "a.png"
    Image.new("RGB", (1920, 1080)).save(ok)
    assert tp.validate_user_thumbnail(ok)[:2] == ([], [])
    bad = tmp_path / "b.png"
    bad.write_bytes(b"xx")
    assert tp.validate_user_thumbnail(bad)[0]


@requires_ffmpeg
def test_silent_source_passes_with_warning_but_lost_audio_fails(tmp_path):
    from pipeline import ffmpeg_tools as fx
    src = tmp_path / "src.mp4"
    fx.ffmpeg(["-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=6", "-c:v",
               "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)])
    silent = tmp_path / "silent.wav"
    fx.ffmpeg(["-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "6", str(silent)])
    loud = tmp_path / "loud.wav"
    fx.ffmpeg(["-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=6:sample_rate=48000", str(loud)])
    out = tmp_path / "out.mp4"
    sb.render_short(src, silent, 1.0, 5.0, sb.ShortSettings(quality="fast"), (640, 360), 30.0, "libx264",
                    tmp_path / "w", None, out)
    errors, det = sb.validate_short(out, 4.0, src, 1.0, True, tmp_path / "w", silent)
    assert errors == [] and "no audible sound" in det["warnings"][0]
    errors, _ = sb.validate_short(out, 4.0, src, 1.0, True, tmp_path / "w", loud)
    assert errors and "although the audio mix has sound" in errors[0]
