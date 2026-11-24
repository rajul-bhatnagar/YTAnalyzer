"""Audio evidence for gameplay analysis: game transients/loudness jumps, emphatic voice delivery, clipping.

These are SIGNALS, not judgements: a loud moment can be an explosion, music or clipping. Downstream scoring never
treats an audio peak alone as proof of a highlight.
Input files are the 16 kHz mono copies made in Phase 2; everything is streamed in blocks (constant memory).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .ffmpeg_tools import Cancelled

HOP = 0.1  # seconds per analysis frame
_FLOOR_DB = -120.0


def frame_levels(wav: Path, hop: float = HOP, block_sec: float = 60.0,
                 cancel_check: Callable[[], bool] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-hop RMS dBFS, per-hop peak (abs, linear) and per-hop count of near-full-scale samples."""
    import soundfile as sf  # noqa: PLC0415

    info = sf.info(str(wav))
    n_hop = max(int(round(info.samplerate * hop)), 1)
    block = n_hop * max(int(block_sec / hop), 1)
    rms, peak, clip = [], [], []
    for i, chunk in enumerate(sf.blocks(str(wav), blocksize=block, dtype="float32", always_2d=True)):
        if cancel_check and cancel_check():
            raise Cancelled("Cancelled during audio feature extraction")
        mono = chunk.mean(axis=1)
        usable = len(mono) - len(mono) % n_hop
        if usable == 0:
            continue
        frames = mono[:usable].reshape(-1, n_hop)
        ms = np.mean(np.square(frames), axis=1)
        rms.append(np.where(ms > 1e-12, 10 * np.log10(np.maximum(ms, 1e-12)), _FLOOR_DB))
        a = np.abs(frames)
        peak.append(a.max(axis=1))
        clip.append((a >= 0.98).sum(axis=1))
    if not rms:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z
    return (np.concatenate(rms).astype(np.float32), np.concatenate(peak).astype(np.float32),
            np.concatenate(clip).astype(np.int32))


def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) == 0:
        return x
    win = max(win | 1, 3)  # odd
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    # Strided median over a sliding window; fine for hours of 0.1 s frames (36k frames per hour).
    view = np.lib.stride_tricks.sliding_window_view(xp, win)
    step = max(len(x) // 20000, 1)  # subsample very long inputs, then interpolate
    if step > 1:
        idx = np.arange(0, len(x), step)
        med = np.median(view[idx], axis=1)
        return np.interp(np.arange(len(x)), idx, med).astype(np.float32)
    return np.median(view, axis=1).astype(np.float32)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def game_audio_events(db: np.ndarray, hop: float, min_jump_db: float, floor_db: float = -50.0,
                      context_sec: float = 10.0) -> list[dict[str, Any]]:
    """Moments where game audio rises clearly above its local level (rolling median over ``context_sec``)."""
    if len(db) == 0:
        return []
    base = rolling_median(db, int(context_sec / hop))
    jump = db - base
    mask = (jump >= min_jump_db) & (db > floor_db)
    events = []
    for a, b in _runs(mask):
        # Allow tiny gaps inside one event.
        seg = jump[a:b]
        k = a + int(np.argmax(seg))
        # Rise time = from the run start to the first frame within 3 dB of the run's maximum. (Using the loudest
        # frame instead would call a flat, sudden 1 s burst "gradual" whenever its maximum lands late.)
        near_max = int(np.argmax(seg >= seg.max() - 3.0))
        fast = near_max * hop <= 0.3
        events.append({
            "kind": "game_transient" if fast else "loudness_rise",
            "start": round(a * hop, 2), "end": round(b * hop, 2), "peak": round(k * hop, 2),
            "strength_db": round(float(seg.max()), 1), "level_db": round(float(db[k]), 1),
            "note": ("Sudden loud game sound" if fast else "Game audio gets clearly louder")
                    + f" (+{float(seg.max()):.0f} dB above the local level). Could be action, music or an effect.",
        })
    return _merge_close(events, 1.0)


def _merge_close(events: list[dict[str, Any]], gap: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in sorted(events, key=lambda x: x["start"]):
        if out and e["start"] - out[-1]["end"] <= gap and e["kind"] == out[-1]["kind"]:
            last = out[-1]
            if e["strength_db"] > last["strength_db"]:
                last.update(peak=e["peak"], strength_db=e["strength_db"], level_db=e["level_db"], note=e["note"])
            last["end"] = max(last["end"], e["end"])
        else:
            out.append(dict(e))
    return out


def voice_emphasis_events(db: np.ndarray, hop: float, speech: list[tuple[float, float]], min_db: float,
                          min_segments: int = 3, min_speech_sec: float = 20.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Speech segments whose typical level is clearly above the speaker's level in the OTHER segments.

    Needs enough speech to know the speaker's usual level; with too little, nothing is reported (a single
    segment compared with itself would only show natural loudness variation, not emphasis)."""
    segs = []
    for s, e in speech:
        a, b = max(int(s / hop), 0), min(int(np.ceil(e / hop)), len(db))
        if b - a >= 3:
            segs.append((s, e, a, b))
    total = sum(e - s for s, e, _, _ in segs)
    meta: dict[str, Any] = {"speech_segments": len(segs), "speech_sec": round(total, 1)}
    if len(segs) < min_segments or total < min_speech_sec:
        meta["skipped"] = (f"Not enough speech to know the speaker's usual level (need {min_segments}+ segments and "
                           f"{min_speech_sec:.0f}+ s).")
        return [], meta
    levels = [float(np.median(db[a:b])) for _, _, a, b in segs]
    out = []
    for i, (s, e, a, b) in enumerate(segs):
        others = levels[:i] + levels[i + 1:]
        baseline = float(np.median(others))
        rise = levels[i] - baseline
        if rise >= min_db:
            k = a + int(np.argmax(db[a:b]))
            out.append({"kind": "voice_emphasis", "start": round(s, 2), "end": round(e, 2), "peak": round(k * hop, 2),
                        "strength_db": round(rise, 1),
                        "note": f"Voice clearly louder than usual (+{rise:.0f} dB). Emphasis is a signal of "
                                "excitement, not proof that the moment is good."})
    meta["baseline_db"] = round(float(np.median(levels)), 1)
    return out, meta


def clipping_ranges(clip_counts: np.ndarray, hop: float, min_samples: int = 3) -> list[dict[str, Any]]:
    out = []
    for a, b in _runs(clip_counts >= min_samples):
        out.append({"start": round(a * hop, 2), "end": round(b * hop, 2), "samples": int(clip_counts[a:b].sum())})
    return out


def analyse(game_wav: Path | None, voice_wav: Path | None, voice_offset: float, speech: list[tuple[float, float]],
            duration: float, cfg: dict[str, Any], cancel_check: Callable[[], bool] | None = None) -> dict[str, Any]:
    """All audio evidence for one recording, in video time (voice offset applied)."""
    out: dict[str, Any] = {"hop": HOP, "game_events": [], "voice_events": [], "clipping": [], "notes": []}
    if game_wav is not None and game_wav.exists():
        db, _peak, clip = frame_levels(game_wav, cancel_check=cancel_check)
        out["game_events"] = [e for e in game_audio_events(db, HOP, float(cfg["transient_min_db"]))
                              if e["start"] < duration]
        out["clipping"] = clipping_ranges(clip, HOP)
        out["game_level_median_db"] = round(float(np.median(db)), 1) if len(db) else None
    else:
        out["notes"].append("No game or combined audio track: no game-audio evidence.")
    if voice_wav is not None and voice_wav.exists():
        vdb, _vp, _vc = frame_levels(voice_wav, cancel_check=cancel_check)
        local_speech = [(s - voice_offset, e - voice_offset) for s, e in speech if e - voice_offset > 0]
        ev, meta = voice_emphasis_events(vdb, HOP, local_speech, float(cfg["voice_emphasis_min_db"]))
        for e in ev:
            for k in ("start", "end", "peak"):
                e[k] = round(e[k] + voice_offset, 2)
        out["voice_events"] = [e for e in ev if 0 <= e["start"] < duration]
        out["voice"] = meta
    else:
        out["notes"].append("No voice track: no commentary emphasis evidence.")
    return out
