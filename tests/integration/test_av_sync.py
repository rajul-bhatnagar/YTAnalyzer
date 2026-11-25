"""A/V sync when the video stream starts later than the audio (common in phone recordings).

The fixture has a white flash and a beep at the SAME real moment, with the video stream starting 0.5 s after the
file (and audio) start. Extracted audio is aligned to the video stream, so every video seek must add that 0.5 s
(media_probe.video_seek_shift); without it the flash and the beep come out 0.5 s apart."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import audio_prep, media_probe, shorts_builder, timeline_builder
from pipeline import ffmpeg_tools as fx
from tests.conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def offset_source(tmp_path_factory):
    d = tmp_path_factory.mktemp("avsync ü")
    video = d / "v.mkv"
    fx.ffmpeg(["-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=30:d=7",
               "-vf", "drawbox=enable='between(t,3,3.2)':color=white:t=fill", "-c:v", "libx264", "-preset",
               "ultrafast", "-pix_fmt", "yuv420p", str(video)])
    audio = d / "a.wav"
    fx.ffmpeg(["-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=1000:d=8:sample_rate=48000",
               "-af", "volume='if(between(t,3.5,3.7),0.8,0)':eval=frame", str(audio)])
    src = d / "phone style.mkv"  # flash at video time 3.0 = file time 3.5 = beep time
    fx.ffmpeg(["-y", "-v", "error", "-itsoffset", "0.5", "-i", str(video), "-i", str(audio), "-map", "0:v",
               "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(src)])
    info = media_probe.probe(src)
    if abs(media_probe.video_seek_shift(info) - 0.5) > 0.05:
        pytest.skip(f"container did not keep the 0.5 s video offset (shift {media_probe.video_seek_shift(info)})")
    wav = d / "track.wav"
    audio_prep.extract_track(src, info, 0, wav, None)
    return src, info, wav, d


def _flash_time(video) -> float:
    import subprocess
    proc = subprocess.run([str(fx.binaries().ffmpeg), "-nostdin", "-v", "error", "-i", str(video), "-vf",
                           "scale=32:18,format=gray", "-f", "rawvideo", "-"], capture_output=True, timeout=120,
                          check=True, stdin=subprocess.DEVNULL)
    raw = np.frombuffer(proc.stdout, np.uint8)
    frames = raw[: len(raw) // (32 * 18) * 32 * 18].reshape(-1, 32 * 18).mean(axis=1)
    fps = media_probe.probe(video).fps or 30.0
    return float(np.argmax(frames > 128)) / fps


def _beep_time(video, tmp) -> float:
    import soundfile as sf
    wav = tmp / "out_audio.wav"
    fx.ffmpeg(["-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "48000", str(wav)])
    data, sr = sf.read(str(wav))
    hop = sr // 100
    rms = np.sqrt(np.mean(data[: len(data) // hop * hop].reshape(-1, hop) ** 2, axis=1))
    return float(np.argmax(rms > 0.05)) / 100


def test_shift_detected(offset_source):
    _src, info, _wav, _d = offset_source
    assert media_probe.video_seek_shift(info) == pytest.approx(0.5, abs=0.05)


def test_short_keeps_flash_and_beep_together(offset_source):
    src, info, wav, d = offset_source
    out = d / "short.mp4"
    shorts_builder.render_short(src, wav, 2.0, 5.0, shorts_builder.ShortSettings(layout="center_crop", quality="fast"),
                                info.display_resolution, info.fps, "libx264", d / "w_short", None, out,
                                shift=media_probe.video_seek_shift(info))
    flash, beep = _flash_time(out), _beep_time(out, d)
    assert flash == pytest.approx(1.0, abs=0.1) and abs(flash - beep) < 0.1, (flash, beep)


def test_rough_cut_keeps_flash_and_beep_together(offset_source):
    src, info, wav, d = offset_source
    out = d / "rough.mp4"
    timeline_builder.render_rough_cut(src, wav, info, [(2.0, 5.0)], out, "libx264", "fast", 128)
    flash, beep = _flash_time(out), _beep_time(out, d)
    assert flash == pytest.approx(1.0, abs=0.1) and abs(flash - beep) < 0.1, (flash, beep)
