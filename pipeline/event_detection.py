"""Signal-level events: cluster atomic signals (game-audio jumps, emphatic voice, scene changes, motion bursts)
into timestamped candidate moments with evidence. Works with no AI at all (CPU-only path).

A signal event says WHERE something measurable happens and WHICH signals agree. It never claims WHAT happened in
the game; descriptions are factual statements about the measurements and exact transcript quotes.
"""

from __future__ import annotations

from typing import Any

MODALITY = {"game_transient": "audio", "loudness_rise": "audio", "voice_emphasis": "commentary",
            "scene_change": "visual", "motion_burst": "visual"}
KIND_LABEL = {"game_transient": "sudden loud game sound", "loudness_rise": "game audio gets louder",
              "voice_emphasis": "emphatic voice", "scene_change": "sudden picture change",
              "motion_burst": "high on-screen motion"}


def atom_strength(a: dict[str, Any]) -> float:
    k = a["kind"]
    if k == "game_transient":
        return min(a["strength_db"] / 24.0, 1.0)
    if k == "loudness_rise":
        return 0.8 * min(a["strength_db"] / 30.0, 1.0)
    if k == "voice_emphasis":
        return min(a["strength_db"] / 12.0, 1.0)
    if k == "scene_change":
        return 0.6 * min(a.get("strength", 1.0) * 0.5, 1.0)
    if k == "motion_burst":
        return 0.5 * min(a.get("strength", 1.0), 1.0)
    return 0.3


def transcript_excerpt(words: list[dict[str, Any]], start: float, end: float, max_words: int = 40) -> str:
    sel = [w for w in words if start <= w["start"] <= end]
    text = "".join(w["word"] for w in sel[:max_words]).strip()
    return text + (" …" if len(sel) > max_words else "")


def cluster(atoms: list[dict[str, Any]], words: list[dict[str, Any]], gap: float,
            duration: float) -> list[dict[str, Any]]:
    atoms = sorted((dict(a, s=atom_strength(a)) for a in atoms), key=lambda a: a["start"])
    groups: list[list[dict[str, Any]]] = []
    for a in atoms:
        if groups and a["start"] <= max(x["end"] for x in groups[-1]) + gap:
            groups[-1].append(a)
        else:
            groups.append([a])
    events = []
    for g in groups:
        start = max(min(a["start"] for a in g), 0.0)
        end = min(max(a["end"] for a in g), duration)
        best = max(g, key=lambda a: a["s"])
        per_kind: dict[str, float] = {}
        for a in g:
            per_kind[a["kind"]] = max(per_kind.get(a["kind"], 0.0), a["s"])
        modalities = sorted({MODALITY.get(k, "other") for k in per_kind})
        # Noisy-OR across kinds, then require agreement: one modality alone is capped.
        miss = 1.0
        for v in per_kind.values():
            miss *= 1.0 - v
        strength = 1.0 - miss
        uncertainty = []
        if len(modalities) == 1:
            strength *= 0.6
            if modalities == ["audio"]:
                uncertainty.append("Audio peak only: could be music, an effect or clipping, not necessarily a "
                                   "gameplay moment.")
            elif modalities == ["visual"]:
                uncertainty.append("Picture change only: could be a menu, cutscene or camera move.")
            else:
                uncertainty.append("Only the voice signals this; the game itself shows no measurable change.")
        quote = transcript_excerpt(words, start - 2.0, end + 3.0)
        parts = [f"{KIND_LABEL.get(k, k)}" + (f" (+{max(a['strength_db'] for a in g if a['kind'] == k):.0f} dB)"
                                              if any('strength_db' in a for a in g if a['kind'] == k) else "")
                 for k in sorted(per_kind, key=lambda k: -per_kind[k])]
        desc = "Measured: " + "; ".join(parts) + "."
        events.append({
            "key": f"s_{int(round(best['peak'] * 1000))}",
            "start": round(start, 2), "end": round(max(end, start + 0.5), 2), "peak": round(best["peak"], 2),
            "kinds": sorted(per_kind), "modalities": modalities, "strength": round(strength, 3),
            "description": desc, "transcript_quote": quote,
            "uncertainty": " ".join(uncertainty),
            "members": [{k: v for k, v in a.items() if k != "s"} for a in g],
        })
    return events


def detect(audio: dict[str, Any], scenes: list[dict[str, Any]], bursts: list[dict[str, Any]],
           words: list[dict[str, Any]], duration: float, gap: float) -> list[dict[str, Any]]:
    atoms = [*audio.get("game_events", []), *audio.get("voice_events", []), *scenes, *bursts]
    clipping = audio.get("clipping", [])
    events = cluster(atoms, words, gap, duration)
    for e in events:
        clips = [c for c in clipping if c["start"] < e["end"] and c["end"] > e["start"]]
        e["clipping"] = bool(clips)
        if clips:
            e["uncertainty"] = (e["uncertainty"] + " Possible audio clipping here.").strip()
    return events


def window_activity(events: list[dict[str, Any]], start: float, end: float) -> float:
    """How much measurable activity a window contains (used to pick windows for optional video upload)."""
    return round(sum(e["strength"] for e in events if start <= e["peak"] < end), 3)
