"""Event fusion, explainable scoring and Shorts-candidate construction.

Scores are RANKING AIDS, not objective truth:
- Factors the AI could not judge (rated 0) or that have no data are excluded, not counted as zero.
- Signals-only moments (no AI description) are capped at the "promising" band, because measurements say where
  something happens, not whether it is good.
- Scores are rounded to 5 points and shown with a band to avoid false precision.
"""

from __future__ import annotations

from typing import Any

import numpy as np

AI_FACTORS = ["significance", "stakes", "surprise", "skill", "entertainment", "narrative_completeness",
              "visual_clarity", "standalone"]
COMPUTED_FACTORS = ["signal_agreement", "audio_support", "commentary_support", "technical_quality"]
FACTOR_LABELS = {
    "significance": "Event significance", "stakes": "Stakes / consequence", "surprise": "Surprise / rarity",
    "skill": "Skill / difficulty", "entertainment": "Emotion / entertainment",
    "narrative_completeness": "Setup and payoff", "visual_clarity": "Visual clarity",
    "standalone": "Understandable without context", "signal_agreement": "Signals agree",
    "audio_support": "Game-audio support", "commentary_support": "Commentary support",
    "technical_quality": "Technical quality",
}

BASE_WEIGHTS = {"significance": 1.5, "stakes": 1.0, "surprise": 1.0, "skill": 1.0, "entertainment": 1.2,
                "narrative_completeness": 1.0, "visual_clarity": 0.8, "standalone": 0.8, "signal_agreement": 1.0,
                "audio_support": 0.6, "commentary_support": 0.6, "technical_quality": 0.5}

# The style changes emphasis and storytelling, not evidence standards.
STYLE_MULTIPLIERS: dict[str, dict[str, float]] = {
    "Challenge / Impossible Goal": {"stakes": 1.4, "narrative_completeness": 1.3, "significance": 1.2},
    "Sports Clutch / Match": {"stakes": 1.5, "significance": 1.3, "standalone": 1.2},
    "Open-World Chaos / Stunts": {"surprise": 1.4, "entertainment": 1.4, "visual_clarity": 1.2},
    "Horror / Funny Reactions": {"surprise": 1.5, "commentary_support": 1.6, "entertainment": 1.3},
    "Shooter / High-Skill Plays": {"skill": 1.6, "stakes": 1.2, "visual_clarity": 1.2},
    "Story / Mod Experiment": {"narrative_completeness": 1.5, "significance": 1.2, "standalone": 0.8},
    "Custom": {},
}

BANDS = [(75, "strong"), (55, "promising"), (35, "possible"), (0, "weak")]

# Fallback when the AI gave no outcome mood: from the event category; unknown stays "" (no guess).
CATEGORY_MOOD = {"achievement": "win_hype", "skill": "win_hype", "clutch": "clutch_tension",
                 "close_call": "clutch_tension", "failure": "fail_shock", "surprise": "fail_shock",
                 "funny": "funny_chaos"}


def outcome_mood(ev: dict[str, Any]) -> str:
    from .ai_providers.base import OUTCOME_MOODS  # noqa: PLC0415
    ai_mood = ((ev.get("ai") or {}).get("outcome_mood") or "").strip().lower()
    if ai_mood in OUTCOME_MOODS:
        return ai_mood
    return CATEGORY_MOOD.get(ev.get("category", ""), "")
SIGNALS_ONLY_CAP = 70


def weights_for(style: str) -> dict[str, float]:
    mult = STYLE_MULTIPLIERS.get(style, {})
    return {k: v * mult.get(k, 1.0) for k, v in BASE_WEIGHTS.items()}


def band(score: float) -> str:
    return next(name for lo, name in BANDS if score >= lo)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# --- fusion ---------------------------------------------------------------------------------------------
def fuse(signal_events: list[dict[str, Any]], window_results: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Link AI events to the measured signal events that support them; keep unexplained signals as their own
    'needs review' events. Each fused event keeps all evidence and where it came from."""
    ai_events = []
    for widx, res in sorted(window_results.items()):
        for i, e in enumerate(res["result"]["events"]):
            ai_events.append({**e, "_window": widx, "_idx": i, "_provider": res["provider"], "_model": res["model"],
                              "_mode": res["mode"]})
    # Merge near-duplicate AI events (adjacent windows can both describe a boundary moment).
    merged: list[dict[str, Any]] = []
    for e in sorted(ai_events, key=lambda x: x["start_sec"]):
        if merged:
            m = merged[-1]
            ov = _overlap(m["start_sec"], m["end_sec"], e["start_sec"], e["end_sec"])
            shorter = min(m["end_sec"] - m["start_sec"], e["end_sec"] - e["start_sec"]) or 1.0
            # Same moment described twice: same category, or by two overlapping windows (whatever they called it).
            if ov / shorter > 0.5 and (m["category"] == e["category"] or m["_window"] != e["_window"]):
                m["end_sec"] = max(m["end_sec"], e["end_sec"])
                m["frame_times"] = sorted(set(m["frame_times"]) | set(e["frame_times"]))
                continue
        merged.append(dict(e))
    used: set[str] = set()
    fused = []
    for e in merged:
        linked = [s for s in signal_events
                  if _overlap(e["start_sec"] - 2, e["end_sec"] + 2, s["start"], s["end"]) > 0]
        used |= {s["key"] for s in linked}
        peak = max(linked, key=lambda s: s["strength"])["peak"] if linked else (e["start_sec"] + e["end_sec"]) / 2
        peak = min(max(peak, e["start_sec"]), e["end_sec"])
        fused.append({
            "key": f"a{e['_window']}_{e['_idx']}_{int(round(e['start_sec'] * 1000))}",
            "start": round(e["start_sec"], 2), "end": round(e["end_sec"], 2), "peak": round(peak, 2),
            "category": e["category"], "description": e["description"], "source": "ai+signals" if linked else "ai",
            "ai": {k: e.get(k, "") for k in ("visual_observation", "frame_times", "audio_observation",
                                             "transcript_quote", "confidence", "uncertainty", "factors",
                                             "hook_caption", "suggested_title", "outcome_mood")}
                  | {"provider": e["_provider"], "model": e["_model"], "mode": e["_mode"]},
            "signals": linked,
            "uncertainty": " ".join(x for x in [e["uncertainty"], *[s["uncertainty"] for s in linked]] if x).strip(),
        })
    for s in signal_events:
        if s["key"] in used:
            continue
        fused.append({"key": s["key"], "start": s["start"], "end": s["end"], "peak": s["peak"], "category": "unclear",
                      "description": s["description"], "source": "signals", "ai": None, "signals": [s],
                      "uncertainty": (s["uncertainty"] + " Not described by AI: watch it and decide.").strip()})
    return sorted(fused, key=lambda x: x["start"])


# --- scoring ----------------------------------------------------------------------------------------------
def factor_scores(ev: dict[str, Any], has_game_audio: bool, has_voice: bool,
                  frames: list[dict[str, Any]]) -> dict[str, float | None]:
    f: dict[str, float | None] = {k: None for k in AI_FACTORS + COMPUTED_FACTORS}
    if ev["ai"]:
        for k in AI_FACTORS:
            v = int(ev["ai"]["factors"].get(k, 0) or 0)
            f[k] = v / 5.0 if v > 0 else None  # 0 = "cannot judge" -> excluded
    sig = ev["signals"]
    if sig:
        miss = 1.0
        for s in sig:
            miss *= 1.0 - s["strength"]
        f["signal_agreement"] = round(1.0 - miss, 3)
    else:
        f["signal_agreement"] = 0.0
    audio = [m for s in sig for m in s["members"] if m["kind"] in ("game_transient", "loudness_rise")]
    if has_game_audio:
        f["audio_support"] = round(max((min(m["strength_db"] / 24.0, 1.0) for m in audio), default=0.0), 3)
    voice = [m for s in sig for m in s["members"] if m["kind"] == "voice_emphasis"]
    quote = (ev["ai"] or {}).get("transcript_quote") or next((s["transcript_quote"] for s in sig if s["transcript_quote"]), "")
    if has_voice:
        f["commentary_support"] = round(max(max((min(m["strength_db"] / 12.0, 1.0) for m in voice), default=0.0),
                                            0.4 if quote else 0.0), 3)
    near = [fr for fr in frames if ev["start"] - 1 <= fr["t"] <= ev["end"] + 1]
    if near:
        bad = sum(1 for fr in near if fr["issues"])
        q = 1.0 - bad / len(near)
        if any(s.get("clipping") for s in sig):
            q -= 0.3
        f["technical_quality"] = round(max(q, 0.0), 3)
    return f


def score_event(factors: dict[str, float | None], style: str, signals_only: bool) -> tuple[int, dict[str, Any]]:
    w = weights_for(style)
    used = {k: v for k, v in factors.items() if v is not None}
    total_w = sum(w[k] for k in used) or 1.0
    raw = 100.0 * sum(w[k] * v for k, v in used.items()) / total_w
    capped = signals_only and raw > SIGNALS_ONLY_CAP
    score = int(5 * round(min(raw, SIGNALS_ONLY_CAP) / 5)) if signals_only else int(5 * round(raw / 5))
    breakdown = {k: {"label": FACTOR_LABELS[k], "value": None if v is None else round(v * 5, 1),
                     "weight": round(w[k], 2), "used": v is not None} for k, v in factors.items()}
    return score, {"factors": breakdown, "raw": round(raw, 1), "capped_signals_only": capped,
                   "excluded": sorted(k for k, v in factors.items() if v is None), "style": style}


def overall_confidence(ev: dict[str, Any]) -> str:
    sig_mods = {m for s in ev["signals"] for m in s["modalities"]}
    if ev["ai"]:
        conf = ev["ai"]["confidence"]
        levels = [conf.get("visual"), conf.get("audio"), conf.get("transcript")]
        if conf.get("visual") in ("high", "medium") and ev["ai"]["frame_times"] and len(sig_mods) >= 1:
            return "high" if len(sig_mods) >= 2 or "high" in levels else "medium"
        if any(x in ("high", "medium") for x in levels) or len(sig_mods) >= 2:
            return "medium"
        return "low"
    strength = max((s["strength"] for s in ev["signals"]), default=0.0)
    return "medium" if len(sig_mods) >= 2 and strength >= 0.5 else "low"


# --- candidates ----------------------------------------------------------------------------------------------
def _snap(t: float, grid: np.ndarray, motion: np.ndarray, radius: float, lo: float, hi: float) -> float:
    """Move a cut point to the calmest picture nearby (avoids starting/ending mid-action)."""
    if len(grid) == 0:
        return t
    sel = (grid >= max(t - radius, lo)) & (grid <= min(t + radius, hi))
    if not sel.any():
        return t
    idx = np.where(sel)[0]
    best = idx[int(np.argmin(motion[idx]))]
    return float(grid[best])


def _speech_runs(grid: np.ndarray, speech: np.ndarray | None, hop: float) -> list[tuple[float, float]]:
    if speech is None or len(speech) == 0:
        return []
    d = np.diff(np.concatenate([[0], speech.astype(np.int8), [0]]))
    return [(float(grid[a]), float(grid[b - 1]) + hop) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def build_clip(ev: dict[str, Any], duration: float, cfg: dict[str, Any], grid: np.ndarray, motion: np.ndarray,
               speech_runs: list[tuple[float, float]]) -> dict[str, Any]:
    lo_len, hi_len = float(cfg["min_sec"]), float(cfg["max_sec"])
    before, after = float(cfg["context_before_sec"]), float(cfg["aftermath_sec"])
    start, end = ev["start"] - before, ev["end"] + after
    notes = []
    if end - start > hi_len:  # long event: keep context before the peak, trim the tail
        start = max(ev["peak"] - before - 4, ev["start"] - before)
        end = start + hi_len
        notes.append(f"Event is longer than {hi_len:.0f} s; trimmed around its peak.")
    if end - start < lo_len:  # pad: 60% before, 40% after
        need = lo_len - (end - start)
        start -= need * 0.6
        end += need * 0.4
    start = _snap(start, grid, motion, 1.5, 0.0, duration)
    end = _snap(end, grid, motion, 1.5, 0.0, duration)
    for s0, s1 in speech_runs:  # never cut someone mid-sentence
        if s0 < start < s1:
            start = s0 if end - s0 <= hi_len else s1
            notes.append("Start moved to avoid cutting speech.")
        if s0 < end < s1:
            end = s1 if s1 - start <= hi_len else s0
            notes.append("End moved to avoid cutting speech.")
    if start < 0:
        end, start = min(end - start, duration), 0.0
    if end > duration:
        start, end = max(start - (end - duration), 0.0), duration
    if end - start > hi_len:
        end = start + hi_len
    if duration >= lo_len and end - start < lo_len:
        end = min(start + lo_len, duration)
        start = max(end - lo_len, 0.0)
    if end - start < lo_len:
        notes.append(f"Recording is shorter than the minimum clip length ({lo_len:.0f} s).")
    event_sec = min(max(ev["peak"], start), end)
    handle = float(cfg["handle_sec"])
    return {"start_sec": round(start, 2), "end_sec": round(end, 2), "event_sec": round(event_sec, 2),
            "handles": {"before": round(min(handle, start), 2), "after": round(min(handle, duration - end), 2)},
            "clip_notes": notes}


def build_candidates(fused: list[dict[str, Any]], duration: float, style: str, cfg: dict[str, Any],
                     signals: dict[str, Any], frames: list[dict[str, Any]], count: int | None = None,
                     blocked: list[tuple[float, float]] | None = None) -> list[dict[str, Any]]:
    """Score every fused event, build a clip around each, then pick the best non-duplicate clips."""
    grid, motion = signals["grid"], signals["motion"]
    hop = float(signals["meta"].get("hop_sec", 0.5))
    runs = _speech_runs(grid, signals.get("speech"), hop)
    has_game = bool(signals["meta"].get("has_game_audio"))
    has_voice = bool(signals["meta"].get("has_speech_track"))
    scored = []
    for ev in fused:
        f = factor_scores(ev, has_game, has_voice, frames)
        score, breakdown = score_event(f, style, ev["ai"] is None)
        clip = build_clip(ev, duration, cfg, grid, motion, runs)
        scored.append({"event": ev, "score": score, "breakdown": breakdown, "clip": clip,
                       "confidence": overall_confidence(ev)})
    scored.sort(key=lambda x: (-x["score"], -max((s["strength"] for s in x["event"]["signals"]), default=0)))
    want = int(count if count is not None else cfg["candidate_count"])
    chosen: list[dict[str, Any]] = []
    taken = list(blocked or [])
    for c in scored:
        if len(chosen) >= want:
            break
        s, e = c["clip"]["start_sec"], c["clip"]["end_sec"]
        dup = None
        for i, (t0, t1) in enumerate(taken):
            ov = _overlap(s, e, t0, t1)
            if ov > 0 and (not cfg.get("allow_overlap") or ov / min(e - s, t1 - t0) > 0.5):
                dup = i
                break
        if dup is not None:
            if dup >= len(blocked or []):  # same moment as an already chosen candidate: merge evidence
                chosen[dup - len(blocked or [])].setdefault("also_covers", []).append(c["event"]["key"])
            continue
        taken.append((s, e))
        chosen.append(c)
    return [_to_candidate(c, rank, frames) for rank, c in enumerate(chosen, 1)]


def _to_candidate(c: dict[str, Any], rank: int, frames: list[dict[str, Any]]) -> dict[str, Any]:
    ev, ai = c["event"], c["event"]["ai"]
    frame_by_t = {f["t"]: f for f in frames}
    if ai and ai["frame_times"]:
        vis = [{"t": t, "path": frame_by_t[t]["path"] if t in frame_by_t else None, "observation": ai["visual_observation"]}
               for t in ai["frame_times"]]
    else:
        near = sorted((f for f in frames if ev["start"] - 1 <= f["t"] <= ev["end"] + 1),
                      key=lambda f: abs(f["t"] - ev["peak"]))[:3]
        vis = [{"t": f["t"], "path": f["path"], "observation": ""} for f in sorted(near, key=lambda f: f["t"])]
    audio = [{"start": m["start"], "end": m["end"], "note": m["note"]}
             for s in ev["signals"] for m in s["members"] if m["kind"] in ("game_transient", "loudness_rise",
                                                                         "voice_emphasis")]
    if ai and ai["audio_observation"]:
        audio.insert(0, {"start": ev["start"], "end": ev["end"], "note": "AI: " + ai["audio_observation"]})
    quotes = []
    if ai and ai["transcript_quote"]:
        quotes.append({"start": ev["start"], "end": ev["end"], "quote": ai["transcript_quote"]})
    for s in ev["signals"]:
        if s["transcript_quote"] and all(q["quote"] != s["transcript_quote"] for q in quotes):
            quotes.append({"start": s["start"], "end": s["end"], "quote": s["transcript_quote"]})
    reason = ev["description"]
    if ai and ev["signals"]:
        reason += " Supported by measurements: " + "; ".join(s["description"].removeprefix("Measured: ").rstrip(".")
                                                            for s in ev["signals"]) + "."
    if not ai:
        reason += " (No AI description: ranked from measurements only.)"
    return {
        "rank": rank, **{k: c["clip"][k] for k in ("start_sec", "end_sec", "event_sec", "handles")},
        "category": ev["category"], "outcome_mood": outcome_mood(ev),
        "hook_caption": (ai or {}).get("hook_caption", ""),
        "suggested_title": (ai or {}).get("suggested_title", ""), "reason": reason,
        "score": c["score"], "score_band": band(c["score"]), "score_breakdown": c["breakdown"],
        "confidence": c["confidence"],
        "evidence": {"visual": vis, "audio": audio, "transcript": quotes,
                     "ai": None if not ai else {k: ai[k] for k in ("provider", "model", "mode", "confidence")},
                     "signals": [s["key"] for s in ev["signals"]]},
        "uncertainty": " ".join(x for x in [ev["uncertainty"], *c["clip"]["clip_notes"]] if x).strip(),
        "event_keys": [ev["key"], *c.get("also_covers", [])],
    }
