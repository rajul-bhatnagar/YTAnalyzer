"""Conservative inactive-scene detection and a reviewable cut-decision report.

Rules (by design):
- Default policy is "none": nothing is removed; candidates are only suggested.
- A segment is a candidate only when the picture is nearly static AND game audio is quiet
  (and, if a voice track exists, nobody is speaking). Mic silence alone NEVER removes footage.
- Sudden game-audio events inside a quiet stretch override the cut (keep).
- Pre/post-roll is preserved around everything that is kept; very short cuts are discarded.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) == 0:
        return x
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(xp, 2 * pad + 1)
    return np.median(view, axis=1)[: len(x)]


def find_candidates(grid: np.ndarray, game_db: np.ndarray | None, motion: np.ndarray, speech: np.ndarray | None,
                    hop: float, duration: float, p: dict[str, Any]) -> list[dict[str, Any]]:
    n = len(grid)
    if n == 0:
        return []
    still = motion < float(p["motion_threshold"])
    if game_db is not None and len(game_db) == n:
        quiet = game_db < float(p["game_audio_threshold_db"])
        baseline = _rolling_median(game_db, max(int(20 / hop), 3))
        events = (game_db - baseline) > float(p["event_override_db"])
    else:
        quiet = np.ones(n, dtype=bool)  # no audio: decide on picture only
        events = np.zeros(n, dtype=bool)
    inactive = still & quiet & ~events
    if speech is not None and len(speech) == n:
        inactive &= ~speech
    cands: list[dict[str, Any]] = []
    pre, post = float(p["preroll_sec"]), float(p["postroll_sec"])
    for a, b in _runs(inactive):
        raw_s, raw_e = float(grid[a]), min(float(grid[b - 1]) + hop, duration)
        if raw_e - raw_s < float(p["min_inactive_sec"]):
            continue
        s = raw_s + (pre if raw_s > 0 else 0.0)
        e = raw_e - (post if raw_e < duration else 0.0)
        if e - s < float(p["min_cut_sec"]):
            continue
        seg = slice(a, b)
        cands.append({
            "source_start": round(s, 3), "source_end": round(e, 3),
            "inactive_start": round(raw_s, 3), "inactive_end": round(raw_e, 3),
            "mean_motion": round(float(np.mean(motion[seg])), 3),
            "mean_game_db": round(float(np.mean(game_db[seg])), 1) if game_db is not None and len(game_db) == n else None,
            "speech_present": bool(speech[seg].any()) if speech is not None and len(speech) == n else None,
        })
    return cands


def build_decisions(duration: float, candidates: list[dict[str, Any]], policy: str,
                    overrides: dict[str, bool] | None = None) -> dict[str, Any]:
    """Turn candidates into an ordered keep/cut segment list with source and output time ranges."""
    overrides = overrides or {}
    segments: list[dict[str, Any]] = []
    cursor = 0.0
    for i, c in enumerate(candidates):
        cid = f"c{i + 1}_{int(c['source_start'] * 1000)}"
        cut = policy == "conservative"
        if cid in overrides:
            cut = not overrides[cid]  # override value means "keep"
        reason_cut = (f"Static picture (motion {c['mean_motion']}) and quiet game audio "
                      f"({c['mean_game_db']} dBFS){', no speech' if c['speech_present'] is False else ''}")
        if c["source_start"] > cursor:
            segments.append({"id": f"k_{int(cursor * 1000)}", "kind": "content", "keep": True,
                             "source_start": round(cursor, 3), "source_end": c["source_start"],
                             "reason": "Active gameplay or within pre/post-roll"})
        segments.append({"id": cid, "kind": "candidate", "keep": not cut, "source_start": c["source_start"],
                         "source_end": c["source_end"], "reason": reason_cut,
                         "user_override": cid in overrides, "signals": c,
                         "status": "cut" if cut else ("kept by user" if cid in overrides else "suggested only")})
        cursor = c["source_end"]
    if cursor < duration:
        segments.append({"id": f"k_{int(cursor * 1000)}", "kind": "content", "keep": True,
                         "source_start": round(cursor, 3), "source_end": round(duration, 3),
                         "reason": "Active gameplay or within pre/post-roll"})
    out_t = 0.0
    for s in segments:
        if s["keep"]:
            length = s["source_end"] - s["source_start"]
            s["output_start"], s["output_end"] = round(out_t, 3), round(out_t + length, 3)
            out_t += length
        else:
            s["output_start"] = s["output_end"] = None
    kept = merge_ranges([(s["source_start"], s["source_end"]) for s in segments if s["keep"]])
    removed = duration - sum(e - s for s, e in kept)
    return {"policy": policy, "source_duration": round(duration, 3), "output_duration": round(out_t, 3),
            "removed_sec": round(removed, 3), "segments": segments, "kept_ranges": kept,
            "candidate_count": len(candidates)}


def merge_ranges(ranges: list[tuple[float, float]], gap: float = 0.001) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(ranges):
        if e <= s:
            continue
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return [(round(s, 3), round(e, 3)) for s, e in out]
