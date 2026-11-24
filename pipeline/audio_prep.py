"""Audio extraction (sync-preserving), voiceover import and audio measurements.

Every extracted WAV starts at video t=0 and lasts exactly the video duration, so all later
mixing and analysis can share one timeline. Stream start-time offsets are compensated explicitly.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import ffmpeg_tools as fx
from .models import MediaInfo, TrackRole

VOICEOVER_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}


@dataclass(frozen=True)
class ExtractedTrack:
    audio_index: int
    role: str
    wav_48k: Path
    wav_16k: Path | None
    offset_sec: float


def stream_offset(info: MediaInfo, audio_index: int) -> float:
    """Audio start relative to video start (seconds). Positive = audio starts later."""
    a = info.audio[audio_index]
    v0 = info.video.start_time if info.video and info.video.start_time is not None else 0.0
    a0 = a.start_time if a.start_time is not None else v0
    return round(a0 - v0, 6)


def _align_filters(offset: float, duration: float) -> list[str]:
    f: list[str] = []
    if offset > 0.0005:
        f.append(f"adelay=delays={int(round(offset * 1000))}:all=1")
    elif offset < -0.0005:
        f += [f"atrim=start={-offset:.6f}", "asetpts=PTS-STARTPTS"]
    f += [f"apad=whole_dur={duration:.6f}", f"atrim=end={duration:.6f}"]
    return f


def extract_track(source: Path, info: MediaInfo, audio_index: int, out_48k: Path, out_16k: Path | None,
                  sample_rate: int = 48000, asr_rate: int = 16000,
                  cancel_check: Callable[[], bool] | None = None) -> ExtractedTrack:
    """Decode one stream to PCM WAV (lossless from the decoded source) aligned to the video timeline."""
    offset = stream_offset(info, audio_index)
    chain = ",".join(_align_filters(offset, info.duration_sec))
    out_48k.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_48k.with_name(out_48k.stem + ".partial.wav")
    args = ["-y", "-i", str(source), "-map", f"0:a:{audio_index}", "-vn", "-sn", "-af", chain,
            "-ar", str(sample_rate), "-c:a", "pcm_s16le"]
    channels = info.audio[audio_index].channels or 2
    if channels > 2:
        args += ["-ac", "2"]  # downmix surround to stereo for a YouTube master
    args.append(str(tmp))
    fx.ffmpeg_progress(args, info.duration_sec, cancel_check=cancel_check)
    tmp.replace(out_48k)
    wav16 = None
    if out_16k is not None:
        wav16 = make_asr_copy(out_48k, out_16k, asr_rate, cancel_check)
    role = ""
    return ExtractedTrack(audio_index, role, out_48k, wav16, offset)


def make_asr_copy(src: Path, out: Path, rate: int = 16000, cancel_check: Callable[[], bool] | None = None) -> Path:
    """16 kHz mono copy for Whisper/VAD."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.wav")
    fx.ffmpeg(["-y", "-i", str(src), "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(tmp)],
              cancel_check=cancel_check)
    tmp.replace(out)
    return out


def import_voiceover(src: Path, dest_dir: Path, version: int, sample_rate: int = 48000) -> dict[str, Path]:
    """Keep the user's original file and create a 48 kHz WAV working copy. Never overwrites old versions."""
    if src.suffix.lower() not in VOICEOVER_EXTS:
        raise ValueError(f"Unsupported voiceover format '{src.suffix}'. Use WAV or MP3 (also M4A/FLAC/OGG).")
    info = probe_audio_file(src)
    if info["duration_sec"] <= 0.2:
        raise ValueError("The voiceover file is empty or too short.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    original = dest_dir / f"voiceover_v{version}_original{src.suffix.lower()}"
    if src.resolve() != original.resolve():
        shutil.copy2(src, original)
    wav = dest_dir / f"voiceover_v{version}.wav"
    tmp = wav.with_name(wav.stem + ".partial.wav")
    fx.ffmpeg(["-y", "-i", str(original), "-vn", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(tmp)])
    tmp.replace(wav)
    return {"original": original, "wav": wav}


def probe_audio_file(path: Path) -> dict[str, Any]:
    data = fx.ffprobe_json(path, ["-show_format", "-show_streams"])
    audio = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise ValueError(f"{Path(path).name} contains no audio stream.")
    dur = float((data.get("format") or {}).get("duration") or audio[0].get("duration") or 0)
    return {"duration_sec": dur, "channels": audio[0].get("channels"),
            "sample_rate": int(audio[0].get("sample_rate") or 0), "codec": audio[0].get("codec_name")}


_ASTATS_KEYS = {"Peak level dB": "peak_db", "RMS level dB": "rms_db", "Peak count": "peak_count",
                "Number of samples": "samples"}


def audio_stats(path: Path, start: float | None = None, duration: float | None = None) -> dict[str, float | None]:
    """Overall peak/RMS (dBFS) and how many samples sit at the peak (clipping indicator)."""
    args: list[str] = []
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    proc = fx.ffmpeg([*args, "-i", str(path), "-af",
                      "astats=measure_perchannel=none:measure_overall=Peak_level+RMS_level+Number_of_samples+Peak_count",
                      "-f", "null", "-"], timeout=None)
    out: dict[str, float | None] = {v: None for v in _ASTATS_KEYS.values()}
    for label, key in _ASTATS_KEYS.items():
        m = re.findall(rf"{re.escape(label)}:\s*(-?inf|-?[\d.]+)", proc.stderr)
        if m:
            val = m[-1]
            out[key] = float("-inf") if "inf" in val else float(val)
    return out


def clipping_report(stats: dict[str, float | None], ceiling_db: float = -0.1, min_count: int = 10) -> str | None:
    peak, count = stats.get("peak_db"), stats.get("peak_count") or 0
    if peak is not None and peak >= ceiling_db and count >= min_count:
        return f"Possible clipping: {int(count)} samples at {peak:.2f} dBFS."
    return None


def measure_loudness(path: Path, target_i: float = -14.0, tp: float = -1.5, lra: float = 11.0,
                     cancel_check: Callable[[], bool] | None = None) -> dict[str, float]:
    """First pass of two-pass EBU R128 loudness normalisation."""
    proc = fx.ffmpeg(["-i", str(path), "-af", f"loudnorm=I={target_i}:TP={tp}:LRA={lra}:print_format=json",
                      "-f", "null", "-"], cancel_check=cancel_check)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr, re.S)
    if not m:
        raise fx.FFmpegError("loudnorm did not report measurements")
    data = json.loads(m.group(0))
    return {k: float(v) for k, v in data.items() if k != "normalization_type"}


def roles_by_index(tracks: list[dict[str, Any]]) -> dict[str, int]:
    """Map confirmed role -> audio_index (unique roles only; 'other' ignored)."""
    out: dict[str, int] = {}
    for t in tracks:
        role = t.get("user_role") or t.get("detected_role")
        if role in (TrackRole.COMBINED_MIX.value, TrackRole.GAME.value, TrackRole.MIC.value):
            out[role] = t["audio_index"]
    return out
