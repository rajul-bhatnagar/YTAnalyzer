"""Content-style profiles: per-style defaults for Shorts editing and HUD text reading. Game-agnostic: keyed only by
the content style the creator picks, never by game. Every value can be overridden per project in the UI; an empty
override ("" or null in settings) means "use the profile".

- shorts_dead_air_mode: what a Short does with static, quiet lead-in/tail around the action:
  "trim" (cut it), "speedup" (play it 2-2.5x), "none" (leave it; conservative for suspense and story).
- hud_ocr_weight: how much detected changes in on-screen text (score, timer, kill feed) count as a highlight
  signal; 0 disables HUD reading. Empty OCR never lowers a score.
- default_shorts_layout: the layout a new Short starts with.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

DEAD_AIR_MODES = {"trim": "Cut static lead-in / tail", "speedup": "Speed up static lead-in (2-2.5x)",
                  "none": "Leave as recorded"}
LAYOUT_KEYS = ("blur_fit", "center_crop", "split_stack")  # must match shorts_builder.LAYOUTS


@dataclass(frozen=True)
class StyleProfile:
    style: str
    shorts_dead_air_mode: str
    hud_ocr_weight: float
    default_shorts_layout: str
    hud_strip_bottom: bool  # split layout starts with a bottom HUD/score strip instead of the whole frame
    note: str


PROFILES: dict[str, StyleProfile] = {p.style: p for p in (
    StyleProfile("Sports Clutch / Match", "trim", 0.6, "split_stack", True,
                 "Score and clock changes matter; replays and walk-backs between plays are trimmed."),
    StyleProfile("Shooter / High-Skill Plays", "trim", 0.5, "center_crop", False,
                 "Kill feed and round timer changes matter; full-screen action reads best."),
    StyleProfile("Open-World Chaos / Stunts", "speedup", 0.2, "center_crop", False,
                 "Lead-up driving or travel is sped up rather than cut, so the setup is still visible."),
    StyleProfile("Challenge / Impossible Goal", "trim", 0.3, "blur_fit", False,
                 "Progress counters matter; the whole frame keeps them visible."),
    StyleProfile("Horror / Funny Reactions", "none", 0.0, "blur_fit", False,
                 "Quiet build-up IS the content: never cut or sped up."),
    StyleProfile("Story / Mod Experiment", "none", 0.0, "blur_fit", False,
                 "Pacing belongs to the story: never cut or sped up."),
    StyleProfile("Custom", "none", 0.0, "blur_fit", False, "Conservative defaults: nothing trimmed, no HUD reading."),
)}


def profile(style: str) -> StyleProfile:
    return PROFILES.get(style, PROFILES["Custom"])


def resolve(style: str, settings: dict[str, Any]) -> StyleProfile:
    """The style's profile with this project's overrides applied (empty override = keep the profile value)."""
    p = profile(style)
    sh, an = settings.get("shorts", {}), settings.get("analysis", {})
    dead = sh.get("dead_air_mode") or p.shorts_dead_air_mode
    layout = sh.get("layout") or p.default_shorts_layout
    w = an.get("hud_ocr_weight")
    return replace(p, shorts_dead_air_mode=dead if dead in DEAD_AIR_MODES else p.shorts_dead_air_mode,
                   default_shorts_layout=layout if layout in LAYOUT_KEYS else p.default_shorts_layout,
                   hud_ocr_weight=p.hud_ocr_weight if w is None or w == "" else max(0.0, min(float(w), 1.0)))
