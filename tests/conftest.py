from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline import ffmpeg_tools as fx  # noqa: E402
from pipeline.paths import StudioPaths  # noqa: E402


def _ffmpeg_available() -> bool:
    try:
        fx.binaries()
        return True
    except fx.FFmpegNotFound:
        return False


HAS_FFMPEG = _ffmpeg_available()
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not installed")


@pytest.fixture
def studio_paths(tmp_path: Path) -> StudioPaths:
    """An isolated studio root (with spaces and Unicode) holding a copy of the real default settings."""
    root = tmp_path / "studio root ü"
    paths = StudioPaths(root)
    paths.ensure()
    shutil.copy(REPO / "config" / "default_settings.json", paths.default_settings_file)
    return paths


def _gen(out: Path, args: list[str]) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fx.ffmpeg(["-y", "-v", "error", *args, str(out)], timeout=120)
    return out


def make_tts(out: Path, text: str) -> Path:
    """Spoken English test audio from Windows' built-in speech synthesizer (offline, no copyrighted media)."""
    import subprocess
    script = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$s.SetOutputToWaveFile('{out}'); $s.Speak('{text}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], check=True,
                   capture_output=True, timeout=120)
    if not out.exists() or out.stat().st_size < 10000:
        pytest.skip("Windows speech synthesis unavailable")
    return out


@pytest.fixture(scope="session")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Synthetic fixture recordings generated with FFmpeg (no copyrighted media)."""
    if not HAS_FFMPEG:
        pytest.skip("FFmpeg not installed")
    d = tmp_path_factory.mktemp("media")
    v = ["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=6"]
    tone = lambda f: ["-f", "lavfi", "-i", f"sine=f={f}:d=6"]  # noqa: E731
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]
    out: dict[str, Path] = {}
    out["one"] = _gen(d / "one track.mp4", [*v, *tone(440), "-map", "0:v", "-map", "1:a", *enc])
    out["two"] = _gen(d / "two_track.mkv", [*v, *tone(220), *tone(880), "-map", "0:v", "-map", "1:a",
                                            "-map", "2:a", *enc])
    out["three"] = _gen(d / "three_track.mp4", [
        *v, *tone(330), *tone(220), *tone(880), "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:a",
        "-metadata:s:a:0", "title=Mix", "-metadata:s:a:1", "title=Game", "-metadata:s:a:2", "title=Mic", *enc])
    out["noaudio"] = _gen(d / "no_audio.mp4", [*v, "-an", "-c:v", "libx264", "-preset", "ultrafast"])
    out["vfr"] = _gen(d / "vfr.mkv", ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=4",
                                      "-vf", "setpts='if(lt(N,30),N/30,1+(N-30)/10)/TB'", "-fps_mode", "vfr",
                                      "-c:v", "libx264", "-preset", "ultrafast"])
    out["audioonly"] = _gen(d / "audio_only.mp4", [*tone(440), "-c:a", "aac"])
    speech = make_tts(d / "speech.wav", "What a shot! That is a massive six over long on. "
                                        "We need twelve runs from the last over.")
    out["speech"] = speech
    # 60 s "gameplay": moving 0-15 s, static loading screen 15-45 s, moving 45-60 s.
    # Tracks: combined (game+mic), game (tone, silent while loading), mic (speech at 3 s).
    out["longform"] = _gen(d / "longform game ü.mp4", [
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=15",
        "-f", "lavfi", "-i", "color=c=0x202040:size=640x360:rate=30:duration=30",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=15",
        "-f", "lavfi", "-i", "sine=f=300:d=60:sample_rate=48000",
        "-i", str(speech),
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v];"
        "[3:a]volume='if(between(t,15,45),0,0.5)':eval=frame,aformat=channel_layouts=stereo[g];"
        "[4:a]aresample=48000,adelay=3000:all=1,apad=whole_dur=60,aformat=channel_layouts=mono[m];"
        "[g]asplit=2[g1][g2];[m]asplit=2[m1][m2];[g1][m1]amix=inputs=2:normalize=0[c]",
        "-map", "[v]", "-map", "[c]", "-map", "[g2]", "-map", "[m2]",
        "-metadata:s:a:0", "title=Mix", "-metadata:s:a:1", "title=Game", "-metadata:s:a:2", "title=Mic",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", "60"])
    # 90 s "gameplay" with ONE real event at 40 s: a loud burst on the game track (40-41 s), a hard picture cut
    # at 40 s (test pattern -> colour bars) and the player's speech from 41 s. Another cut back at 55 s is silent.
    out["gameplay"] = _gen(d / "gameplay ü.mp4", [
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=40",
        "-f", "lavfi", "-i", "smptehdbars=size=640x360:rate=30:duration=15",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=35",
        "-f", "lavfi", "-i", "sine=f=220:d=90:sample_rate=48000",
        "-f", "lavfi", "-i", "anoisesrc=d=90:c=pink:r=48000:a=0.7",
        "-i", str(speech),
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v];"
        "[3:a]volume=0.05[bg];[4:a]volume='if(between(t,40,41),1,0)':eval=frame[burst];"
        "[bg][burst]amix=inputs=2:normalize=0,aformat=channel_layouts=stereo[g];"
        "[5:a]aresample=48000,adelay=41000:all=1,apad=whole_dur=90,aformat=channel_layouts=mono[m];"
        "[g]asplit=2[g1][g2];[m]asplit=2[m1][m2];[g1][m1]amix=inputs=2:normalize=0[c]",
        "-map", "[v]", "-map", "[c]", "-map", "[g2]", "-map", "[m2]",
        "-metadata:s:a:0", "title=Mix", "-metadata:s:a:1", "title=Game", "-metadata:s:a:2", "title=Mic",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", "90"])
    data = out["one"].read_bytes()
    trunc = d / "truncated.mp4"
    trunc.write_bytes(data[: len(data) // 2])
    out["truncated"] = trunc
    return out
