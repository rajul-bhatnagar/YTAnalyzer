from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.media_probe import (analyse_frame_deltas, parse_ffprobe, single_track_notice, suggest_roles,
                                  validate_role_mapping)
from pipeline.models import AudioStreamInfo, TrackRole

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_parse_obs_three_track_mkv():
    info = parse_ffprobe(load("ffprobe_obs_3track.json"), "rec.mkv")
    assert info.duration_sec == pytest.approx(3605.48)
    assert info.video and info.video.width == 2560 and info.video.height == 1440
    assert info.video.avg_frame_rate == pytest.approx(60000 / 1001)
    assert [a.stream_index for a in info.audio] == [1, 2, 3]
    assert [a.audio_index for a in info.audio] == [0, 1, 2]
    assert info.audio[2].title == "Mic/Aux"
    assert info.audio[0].is_default
    assert not info.vfr_suspected


def test_parse_phone_rotation_cover_art_vfr_and_missing_duration():
    info = parse_ffprobe(load("ffprobe_rotated_vfr.json"), "clip.mp4")
    assert info.video.stream_index == 1  # stream 0 is attached cover art
    assert info.video.rotation == -90
    assert info.display_resolution == (1080, 1920)
    assert info.vfr_suspected
    assert info.duration_sec == pytest.approx(12.5)
    assert any("no duration" in w for w in info.warnings)
    assert any("start times differ" in w for w in info.warnings)


def test_generic_handler_names_ignored_and_mp4_name_tag_used():
    data = {"format": {"duration": "1"}, "streams": [
        {"index": 0, "codec_type": "audio", "tags": {"handler_name": "SoundHandler"}},
        {"index": 1, "codec_type": "audio", "tags": {"handler_name": "SoundHandler", "name": "Mic"}},
        {"index": 2, "codec_type": "audio", "tags": {"handler_name": "Desktop"}},
    ]}
    info = parse_ffprobe(data, "x.mp4")
    assert [a.title for a in info.audio] == [None, "Mic", "Desktop"]


def test_parse_no_streams():
    info = parse_ffprobe({"format": {"duration": "1.0"}, "streams": []}, "x")
    assert info.video is None and info.audio == []
    assert "No video stream found" in info.warnings


def _tracks(*titles):
    return [AudioStreamInfo(stream_index=i + 1, audio_index=i, title=t) for i, t in enumerate(titles)]


def test_suggest_roles_never_fixed_layout():
    assert suggest_roles([]) == []
    assert suggest_roles(_tracks(None)) == [TrackRole.COMBINED_MIX]
    assert suggest_roles(_tracks("Mic/Aux", "Desktop Audio")) == [TrackRole.MIC, TrackRole.GAME]
    assert suggest_roles(_tracks(None, None, None)) == [TrackRole.COMBINED_MIX, TrackRole.OTHER, TrackRole.OTHER]
    assert suggest_roles(_tracks("Track1", "Game", "Voice")) == [TrackRole.COMBINED_MIX, TrackRole.GAME, TrackRole.MIC]


def test_validate_role_mapping():
    ok = {0: TrackRole.COMBINED_MIX, 1: TrackRole.GAME, 2: TrackRole.MIC}
    assert validate_role_mapping(ok, 3) == []
    assert validate_role_mapping({0: TrackRole.MIC, 1: TrackRole.MIC}, 2)
    assert validate_role_mapping({0: TrackRole.GAME}, 2)  # missing track 1
    assert validate_role_mapping({0: TrackRole.MIC}, 1)
    assert validate_role_mapping({0: TrackRole.OTHER, 1: TrackRole.OTHER}, 2) == []
    assert validate_role_mapping({}, 0) == []


def test_single_track_notice():
    assert "cannot be separated" in single_track_notice(1)
    assert "no audio" in single_track_notice(0)
    assert single_track_notice(2) is None


def test_frame_delta_analysis():
    cfr = [i / 30 for i in range(300)]
    assert analyse_frame_deltas(cfr)[0] is False
    vfr = [i / 30 for i in range(30)] + [1 + i / 10 for i in range(30)]
    is_vfr, details = analyse_frame_deltas(vfr)
    assert is_vfr and details["fraction_regular"] < 0.98
    assert analyse_frame_deltas([0.0, 0.1])[0] is False


def test_decode_errors_ignore_null_muxer_timestamp_warnings():
    from pipeline.media_probe import decode_errors
    stderr = ("[null @ 0000022a80489a80] Application provided invalid, non monotonically increasing dts to muxer in "
              "stream 0: 29 >= 29\n[h264 @ 0000020] Invalid NAL unit size (1234 > 99).\n\n")
    assert decode_errors(stderr) == ["[h264 @ 0000020] Invalid NAL unit size (1234 > 99)."]
