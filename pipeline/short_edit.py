"""Per-Short micro-edits: trim or speed up static, quiet lead-in and tighten a static tail, from the measured
activity signals inside the clip. Only the Short's render changes; the source and long_rough_cut.mp4 never do.

A moment counts as ACTIVE when the picture moves, someone speaks, or game audio jumps above its local level.
Safety margins: at least ``keep_before`` seconds before the key moment, ``tail_keep`` after the last activity,
and the edited Short never drops below ``min_total`` seconds.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Segment = tuple[float, float, float]  # (start, end, speed) in clip time (0 = clip start)


def activity_mask(signals: dict[str, Any], motion_threshold: float, jump_db: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    grid, motion = signals["grid"], signals["motion"]
    active = motion >= motion_threshold
    speech = signals.get("speech")
    if speech is not None and len(speech) == len(grid):
        active = active | speech
    game = signals.get("game_db")
    if game is not None and len(game) == len(grid):
        k = max(int(10 / float(signals["meta"].get("hop_sec", 0.5))), 3)
        pad = k // 2
        view = np.lib.stride_tricks.sliding_window_view(np.pad(game, pad, mode="edge"), 2 * pad + 1)
        base = np.median(view, axis=1)[: len(game)]
        active = active | ((game - base) > jump_db)
    return grid, active


def plan_segments(signals: dict[str, Any], start: float, end: float, event_sec: float, mode: str,
                  speed: float = 2.0, motion_threshold: float = 1.5, keep_before: float = 2.0, lead_gap: float = 1.0,
                  tail_keep: float = 1.5, min_cut: float = 2.0, min_total: float = 15.0) -> tuple[list[Segment], list[str]]:
    """Segments for the Short (clip time) and notes describing what was changed."""
    dur = end - start
    whole: list[Segment] = [(0.0, round(dur, 3), 1.0)]
    if mode not in ("trim", "speedup"):
        return whole, []
    speed = min(max(float(speed), 2.0), 2.5)
    grid, active = activity_mask(signals, motion_threshold)
    hop = float(signals["meta"].get("hop_sec", 0.5))
    inside = (grid >= start) & (grid < end)
    times, act = grid[inside], active[inside]
    if len(times) < 4 or not act.any():
        return whole, ["No measurable activity inside the clip; left as recorded."]
    first, last = float(times[np.argmax(act)]), float(times[len(act) - 1 - np.argmax(act[::-1])]) + hop
    lead_end = min(first - lead_gap, event_sec - keep_before) - start
    tail_start = max(last + tail_keep, event_sec + tail_keep) - start
    segs: list[Segment] = []
    notes: list[str] = []
    a = 0.0
    if lead_end >= min_cut:
        if mode == "speedup":
            segs.append((0.0, round(lead_end, 3), speed))
            notes.append(f"Static lead-in {lead_end:.1f} s played at {speed:g}x.")
        else:
            notes.append(f"Static lead-in {lead_end:.1f} s trimmed.")
        a = lead_end
    b = dur
    if dur - tail_start >= min_cut:
        b = tail_start
        notes.append(f"Static tail {dur - tail_start:.1f} s trimmed (tighter ending).")
    segs.append((round(a, 3), round(b, 3), 1.0))
    if output_length(segs) < min(min_total, dur):
        return whole, [f"Not trimmed: the Short would be shorter than {min(min_total, dur):.0f} s."]
    return segs, notes


def output_length(segs: list[Segment]) -> float:
    return round(sum((e - s) / v for s, e, v in segs), 3)


def remap_time(t: float, segs: list[Segment]) -> float | None:
    """Clip time -> output time, or None if that moment was trimmed away."""
    out = 0.0
    for s, e, v in segs:
        if s <= t <= e:
            return round(out + (t - s) / v, 3)
        out += (e - s) / v
    return None


def remap_words(words: list[dict[str, Any]], segs: list[Segment]) -> list[dict[str, Any]]:
    """Caption words (clip time) moved to output time; words inside trimmed parts are dropped."""
    out = []
    for w in words:
        s, e = remap_time(w["start"], segs), remap_time(w["end"], segs)
        if s is not None and e is not None and e > s:
            out.append({**w, "start": s, "end": e})
    return out


def atempo_chain(speed: float) -> str:
    """FFmpeg atempo filters for a speed-up (each stage kept within 0.5-2.0, the range every FFmpeg build takes)."""
    parts, left = [], speed
    while left > 2.0 + 1e-9:
        parts.append("atempo=2.0")
        left /= 2.0
    if abs(left - 1.0) > 1e-6:
        parts.append(f"atempo={left:.4f}")
    return ",".join(parts) or "anull"
