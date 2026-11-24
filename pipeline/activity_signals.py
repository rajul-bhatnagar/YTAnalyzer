"""Timestamped activity signals shared by inactivity analysis (Phase 2) and highlight analysis (Phase 3):
audio loudness timeline, visual motion timeline, and speech activity (Silero VAD, processed in blocks)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import ffmpeg_tools as fx
from .ffmpeg_tools import Cancelled

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
FRAME_W, FRAME_H = 160, 90


def rms_timeline(wav: Path, hop_sec: float = 0.5,
                 cancel_check: Callable[[], bool] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-hop RMS in dBFS, streamed in blocks (constant memory for multi-hour files)."""
    import soundfile as sf  # noqa: PLC0415

    info = sf.info(str(wav))
    block = max(int(round(info.samplerate * hop_sec)), 1)
    vals: list[float] = []
    for i, chunk in enumerate(sf.blocks(str(wav), blocksize=block, dtype="float32", always_2d=True)):
        if cancel_check and i % 2000 == 0 and cancel_check():
            raise Cancelled("Cancelled during audio analysis")
        ms = float(np.mean(np.square(chunk))) if chunk.size else 0.0
        vals.append(10 * np.log10(ms) if ms > 1e-12 else -120.0)
    rms = np.asarray(vals, dtype=np.float32)
    times = np.arange(len(rms), dtype=np.float32) * hop_sec
    return times, rms


def motion_timeline(video: Path, fps: float = 2.0, duration: float | None = None,
                    progress: Callable[[float], None] | None = None,
                    cancel_check: Callable[[], bool] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean absolute luma difference between consecutive downscaled frames (0..255 scale)."""
    b = fx.binaries()
    cmd = [str(b.ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-i", str(video), "-map", "0:v:0",
           "-vf", f"fps={fps:g},scale={FRAME_W}:{FRAME_H}:flags=area,format=gray", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_CREATE_NO_WINDOW)
    frame_bytes = FRAME_W * FRAME_H
    diffs: list[float] = []
    prev: np.ndarray | None = None
    n = 0
    try:
        assert p.stdout is not None
        while True:
            buf = p.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            cur = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
            diffs.append(0.0 if prev is None else float(np.mean(np.abs(cur - prev))))
            prev = cur
            n += 1
            if n % 200 == 0:
                if cancel_check and cancel_check():
                    raise Cancelled("Cancelled during motion analysis")
                if progress and duration:
                    progress(min(n / fps / duration, 1.0))
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()
    if p.returncode not in (0, None) and not diffs:
        err = p.stderr.read().decode("utf-8", "replace") if p.stderr else ""
        raise fx.FFmpegError(f"Motion analysis failed: {err[-500:]}")
    if len(diffs) > 1:
        diffs[0] = diffs[1]
    motion = np.asarray(diffs, dtype=np.float32)
    return np.arange(len(motion), dtype=np.float32) / fps, motion


def speech_segments(wav_16k: Path, block_sec: float = 600.0,
                    cancel_check: Callable[[], bool] | None = None) -> list[tuple[float, float]]:
    """Silero VAD (bundled with faster-whisper), processed in blocks to bound memory."""
    import soundfile as sf  # noqa: PLC0415
    from faster_whisper.vad import VadOptions, get_speech_timestamps  # noqa: PLC0415

    info = sf.info(str(wav_16k))
    if info.samplerate != 16000:
        raise ValueError("VAD input must be 16 kHz")
    block = int(block_sec * 16000)
    out: list[tuple[float, float]] = []
    for i, chunk in enumerate(sf.blocks(str(wav_16k), blocksize=block, dtype="float32", always_2d=True)):
        if cancel_check and cancel_check():
            raise Cancelled("Cancelled during speech detection")
        mono = np.ascontiguousarray(chunk.mean(axis=1), dtype=np.float32)
        base = i * block_sec
        for ts in get_speech_timestamps(mono, VadOptions(min_silence_duration_ms=500)):
            out.append((base + ts["start"] / 16000, base + ts["end"] / 16000))
    merged: list[tuple[float, float]] = []
    for s, e in out:  # join segments split at block borders
        if merged and s - merged[-1][1] < 0.3:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def resample_to_grid(times: np.ndarray, values: np.ndarray, grid: np.ndarray, fill: float) -> np.ndarray:
    if len(values) == 0:
        return np.full(len(grid), fill, dtype=np.float32)
    idx = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(values) - 1)
    return values[idx].astype(np.float32)


def speech_mask(segments: list[tuple[float, float]], grid: np.ndarray, hop: float) -> np.ndarray:
    mask = np.zeros(len(grid), dtype=bool)
    for s, e in segments:
        mask[(grid + hop > s) & (grid < e)] = True
    return mask


def save_signals(path: Path, grid: np.ndarray, game_db: np.ndarray, motion: np.ndarray,
                 speech: np.ndarray | None, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, grid=grid, game_db=game_db, motion=motion,
                        speech=speech if speech is not None else np.zeros(0, dtype=bool))
    tmp.replace(path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_signals(path: Path) -> dict[str, Any]:
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    meta_file = path.with_suffix(".json")
    data["meta"] = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    if data["speech"].size == 0:
        data["speech"] = None
    return data
