"""Hybrid gameplay analysis, window by window.

For each analysis window (default 3 minutes) the evidence is: sampled frames, measured signal events and the
transcript. A provider (Gemini, Ollama vision, Ollama text-only) describes visible/audible moments as structured
JSON; every answer is validated against the evidence before it is used. Each window's result is cached under a
fingerprint, so an interrupted, cancelled or quota-paused run continues where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from . import frame_sampler
from .ai_providers.base import (AIProvider, ImageInput, ProviderError, ProviderUnavailable,
                                QuotaExceeded, VideoInput, WindowAnalysis, flat_schema)
from .ffmpeg_tools import Cancelled
from .job_manager import StagePaused

PROMPT_VERSION = 3

STYLE_EMPHASIS = {
    "Challenge / Impossible Goal": "progress toward the stated goal, near-misses, failures and the final outcome",
    "Sports Clutch / Match": "score, time/overs/period, match situation and outcome, but only when visible on screen",
    "Open-World Chaos / Stunts": "stunts, chaotic physics, crashes and unexpected interactions",
    "Horror / Funny Reactions": "suspense, jump scares, surprise and the player's reaction",
    "Shooter / High-Skill Plays": "skillful plays, eliminations, clutch situations and tactical outcomes",
    "Story / Mod Experiment": "story beats, discoveries and what the mod/experiment changes",
    "Custom": "whatever makes a moment meaningful for this recording",
}


@dataclass
class Window:
    idx: int
    start: float
    end: float


def build_windows(duration: float, window_sec: float, min_last_sec: float = 30.0) -> list[Window]:
    window_sec = max(float(window_sec), 30.0)
    bounds = []
    t = 0.0
    while t < duration - 1e-6:
        bounds.append([t, min(t + window_sec, duration)])
        t += window_sec
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < min_last_sec:
        # Two statements on purpose: in `bounds[-2][1] = bounds.pop()[1]` the pop runs first, so [-2] would then
        # point at the wrong window.
        tail = bounds.pop()
        bounds[-1][1] = tail[1]
    return [Window(i, round(s, 3), round(e, 3)) for i, (s, e) in enumerate(bounds)]


# --- evidence & prompts ---------------------------------------------------------------------------------
@dataclass
class WindowEvidence:
    window: Window
    frames: list[dict[str, Any]]
    signals: list[dict[str, Any]]
    transcript: list[dict[str, Any]]  # segments in video time: start, end, text
    context: dict[str, Any]

    def fingerprint(self) -> str:
        data = {"w": [self.window.start, self.window.end], "f": [(f["t"], f["path"]) for f in self.frames],
                "s": [(s["key"], s["strength"]) for s in self.signals],
                "t": [(round(x["start"], 2), x["text"]) for x in self.transcript], "c": self.context}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]

    def transcript_text(self) -> str:
        return " ".join(x["text"] for x in self.transcript)


def _system_prompt(mode: str, game: str) -> str:
    base = (f"You are a careful gameplay analyst helping a creator find highlight moments in a recording of the PC "
            f"game \"{game}\". Rules:\n"
            "- Describe only what the provided evidence shows. Never invent scores, player names, results, rules, "
            "dialogue or events.\n"
            "- If something is not visible or audible, say so in 'uncertainty' and use a lower confidence.\n"
            "- Rate each factor from 0 to 5; use 0 when you cannot judge it.\n"
            "- Return at most 8 events. Returning no events is correct when nothing notable happens.\n"
            "- Quote the transcript only with exact words from it.\n"
            "- hook_caption and suggested_title must not promise anything the evidence does not show.\n")
    if mode == "text":
        base += ("- You CANNOT see the video. Visual information is computed measurements only. Set confidence.visual "
                 "to 'none', factors.visual_clarity to 0, frame_times to [] and visual_observation to ''.\n"
                 "- All times are seconds from the start of the recording.\n")
    elif mode == "video":
        base += ("- You are given a short video clip. Report start_sec, end_sec and frame_times as seconds from the "
                 "START OF THE CLIP (the app converts them).\n")
    else:
        base += ("- Each image is labelled with the exact recording time of its frame(s). Cite those times in "
                 "frame_times; do not cite times you were not shown.\n"
                 "- All times are seconds from the start of the recording.\n")
    return base


def build_prompt(ev: WindowEvidence, mode: str, frame_times: list[float]) -> str:
    c, w = ev.context, ev.window
    lines = [f"Game: {c['game']}", f"Content style: {c['style']}. Pay attention to: "
             f"{STYLE_EMPHASIS.get(c['style'], STYLE_EMPHASIS['Custom'])}."]
    if c.get("topic"):
        lines.append(f"Creator's note about this recording: {c['topic']}")
    lines.append(f"Language of commentary: {c['language']}")
    lines.append(f"Analysis window: {frame_sampler.fmt_ts(w.start)} to {frame_sampler.fmt_ts(w.end)} "
                 f"({w.start:.1f}-{w.end:.1f} s of the recording).")
    if mode == "images" and frame_times:
        lines.append("Frames provided at (s): " + ", ".join(f"{t:.1f}" for t in frame_times))
    if mode == "video":
        lines.append(f"The clip covers {w.start:.1f}-{w.end:.1f} s of the recording; its time 0 = {w.start:.1f} s.")
    lines.append("\nMeasured signals (computed by software, not seen by a person):")
    if ev.signals:
        for s in ev.signals[:25]:
            lines.append(f"- {s['start']:.1f}-{s['end']:.1f} s: {s['description']}"
                         + (f" Uncertain: {s['uncertainty']}" if s["uncertainty"] else ""))
    else:
        lines.append("- none above the detection thresholds")
    lines.append("\nTranscript of the player's voice (speech recognition; may contain errors):")
    if ev.transcript:
        for x in ev.transcript[:80]:
            lines.append(f"[{x['start']:.1f}-{x['end']:.1f}] {x['text']}")
    else:
        lines.append("(no speech in this window or no voice track)")
    lines.append("\nTask: list the meaningful moments in this window with exact start/end times, category, what "
                 "happens, the supporting evidence, confidence per modality, uncertainty and factor ratings. Include "
                 "enough context to understand each moment.")
    return "\n".join(lines)


# --- validation ----------------------------------------------------------------------------------------------
class InvalidOutput(ValueError):
    pass


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def validate_result(raw: str, ev: WindowEvidence, mode: str, frame_times: list[float],
                    duration: float) -> tuple[WindowAnalysis, list[str]]:
    """Parse + check against the evidence. Fixes what can be fixed safely, drops what cannot, reports both."""
    try:
        wa = WindowAnalysis.model_validate_json(raw)
    except ValidationError as e:
        raise InvalidOutput(f"Output does not match the schema: {e.errors()[0]['loc']} {e.errors()[0]['msg']}") from e
    except ValueError as e:
        raise InvalidOutput(f"Output is not valid JSON: {e}") from e
    w = ev.window
    issues: list[str] = []
    kept = []
    transcript_norm = _norm(ev.transcript_text())
    for e in wa.events:
        if mode == "video":
            e.start_sec += w.start
            e.end_sec += w.start
            e.frame_times = [t + w.start for t in e.frame_times]
        if e.end_sec < e.start_sec:
            e.start_sec, e.end_sec = e.end_sec, e.start_sec
            issues.append(f"Swapped reversed times for '{e.description[:40]}'.")
        if e.end_sec < w.start - 5 or e.start_sec > w.end + 5:
            issues.append(f"Dropped an event at {e.start_sec:.1f}-{e.end_sec:.1f} s outside the window.")
            continue
        if e.end_sec - e.start_sec > (w.end - w.start) + 10:
            issues.append(f"Dropped an event spanning {e.end_sec - e.start_sec:.0f} s (longer than the window).")
            continue
        lo, hi = max(w.start - 5, 0.0), min(w.end + 5, duration)
        if e.start_sec < lo or e.end_sec > hi:
            e.start_sec, e.end_sec = max(e.start_sec, lo), min(e.end_sec, hi)
            issues.append("Clamped event times to the analysed range.")
        if e.end_sec - e.start_sec < 0.5:
            e.end_sec = min(e.start_sec + 0.5, duration)
        notes = []
        if mode == "text":
            if e.frame_times or e.confidence.visual != "none" or e.visual_observation:
                notes.append("Visual claims removed: the model could not see the video.")
            e.frame_times, e.visual_observation = [], ""
            e.confidence.visual = "none"
            e.factors.visual_clarity = 0
        elif mode == "images":
            snapped = []
            for t in e.frame_times:
                near = [f for f in frame_times if abs(f - t) <= 0.6]
                if near:
                    snapped.append(min(near, key=lambda f: abs(f - t)))
            if len(snapped) < len(e.frame_times):
                notes.append("The model cited frame times it was not shown; those references were removed.")
            e.frame_times = sorted(set(snapped))
            if not e.frame_times and e.confidence.visual in ("high", "medium"):
                e.confidence.visual = "low"
                notes.append("No supporting frame cited, so visual confidence was lowered.")
        if e.transcript_quote and _norm(e.transcript_quote) not in transcript_norm:
            notes.append(f"Quote not found in the transcript, removed: \"{e.transcript_quote[:60]}\".")
            e.transcript_quote = ""
            if e.confidence.transcript in ("high", "medium"):
                e.confidence.transcript = "low"
        # Known generic category, or the model's own short custom category (kept, as the spec allows).
        e.category = re.sub(r"[^a-z0-9_]+", "_", e.category.strip().lower()).strip("_")[:40] or "unclear"
        if notes:
            e.uncertainty = (e.uncertainty + " " + " ".join(notes)).strip()[:600]
        kept.append(e)
    wa.events = kept
    return wa, issues


# --- provider plan, budget and the window loop -----------------------------------------------------------
@dataclass
class ProviderSlot:
    provider: AIProvider
    mode: str  # "images" | "text" | "video"


@dataclass
class ProviderPlan:
    primary: ProviderSlot | None
    fallback: ProviderSlot | None = None
    on_quota: str = "fallback"  # "fallback" | "pause"
    current: ProviderSlot | None = None
    events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.current = self.primary

    def describe(self) -> str:
        if self.current is None:
            return "signals only (no AI provider)"
        return f"{self.current.provider.name}:{self.current.provider.model} ({self.current.mode})"

    def switch_to_fallback(self, reason: str) -> bool:
        if self.fallback is None or self.current is self.fallback:
            return False
        self.events.append(f"Switched from {self.describe()} to {self.fallback.provider.name}:"
                           f"{self.fallback.provider.model} because: {reason}")
        self.current = self.fallback
        return True

    def drop_to_signals(self, reason: str) -> None:
        self.events.append(f"Continuing with signals only because: {reason}")
        self.current = None


@dataclass
class Budget:
    max_requests: int
    min_interval: float
    sleep: Callable[[float], None] = time.sleep
    used: int = 0
    _last: float = 0.0

    def take(self, cancel_check: Callable[[], bool] | None) -> bool:
        if self.used >= self.max_requests:
            return False
        wait = self.min_interval - (time.monotonic() - self._last)
        while wait > 0:
            if cancel_check and cancel_check():
                raise Cancelled("Cancelled while pacing AI requests")
            step = min(wait, 1.0)
            self.sleep(step)
            wait -= step
        self._last = time.monotonic()
        self.used += 1
        return True


def _select_frames(frames: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    usable = [f for f in frames if f.get("near_duplicate_of") is None]
    ev = [f for f in usable if f["reason"] == "evidence"]
    grid = [f for f in usable if f["reason"] != "evidence"]
    chosen = ev[:limit]
    room = limit - len(chosen)
    if room > 0 and grid:
        step = max(len(grid) / room, 1.0)
        chosen += [grid[int(i)] for i in [j * step for j in range(room)] if int(i) < len(grid)]
    return sorted(chosen, key=lambda f: f["t"])


def _call(slot: ProviderSlot, ev: WindowEvidence, resolve: Callable[[str], Path], cfg: dict[str, Any],
          video_chunk: Callable[[Window], Path] | None) -> tuple[str, list[float], str]:
    """One provider request. Returns (raw text, frame times shown, prompt)."""
    mode = slot.mode
    shown: list[float] = []
    images: list[ImageInput] | None = None
    video: VideoInput | None = None
    if mode == "images":
        sel = _select_frames(ev.frames, int(cfg["max_frames_per_request"]))
        shown = [f["t"] for f in sel]
        if slot.provider.name == "ollama":
            images = []
            for i in range(0, len(sel), 12):
                group = sel[i:i + 12]
                jpeg = frame_sampler.contact_sheet([(f["t"], resolve(f["path"])) for f in group])
                images.append(ImageInput(jpeg, f"Contact sheet {i // 12 + 1}: frames at "
                                               + ", ".join(f"{f['t']:.1f}s" for f in group)))
        else:
            images = [ImageInput(resolve(f["path"]).read_bytes(),
                                 f"Frame at {frame_sampler.fmt_ts(f['t'])} ({f['t']:.1f} s)") for f in sel]
    elif mode == "video" and video_chunk is not None:
        video = VideoInput(video_chunk(ev.window), ev.window.start)
    prompt = build_prompt(ev, mode, shown)
    raw = slot.provider.generate_json(_system_prompt(mode, ev.context["game"]), prompt, flat_schema(WindowAnalysis),
                                      images=images, video=video)
    return raw, shown, prompt


def analyse_windows(evidence: list[WindowEvidence], plan: ProviderPlan, cache_dir: Path, duration: float,
                    resolve: Callable[[str], Path], cfg: dict[str, Any], budget: Budget,
                    progress: Callable[[float, str], None], cancel_check: Callable[[], bool],
                    pause_check: Callable[[], bool], video_windows: set[int] | None = None,
                    video_chunk: Callable[[Window], Path] | None = None,
                    sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Run every window through the current provider; returns the analysis report (results are in the cache)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"windows": [], "plan_events": plan.events, "requests": 0, "prompt_version": PROMPT_VERSION}
    for n, ev in enumerate(evidence):
        if cancel_check():
            raise Cancelled("Cancelled during AI analysis")
        if pause_check():
            raise StagePaused("Paused by you. Finished windows are kept; Resume continues from here.")
        progress(n / max(len(evidence), 1), f"Analysing window {n + 1} of {len(evidence)} with {plan.describe()}")
        frac = n / max(len(evidence), 1)
        entry = _analyse_one(ev, plan, cache_dir, duration, resolve, cfg, budget, cancel_check,
                             video_windows or set(), video_chunk, sleep, lambda msg, f=frac: progress(f, msg))
        report["windows"].append(entry)
    report["requests"] = budget.used
    report["effective"] = sorted({f"{w['provider']}:{w['model']} ({w['mode']})" for w in report["windows"]
                                  if w["status"] == "ok"})
    return report


def _analyse_one(ev: WindowEvidence, plan: ProviderPlan, cache_dir: Path, duration: float,
                 resolve: Callable[[str], Path], cfg: dict[str, Any], budget: Budget,
                 cancel_check: Callable[[], bool], video_windows: set[int],
                 video_chunk: Callable[[Window], Path] | None, sleep: Callable[[float], None],
                 notify: Callable[[str], None] | None = None) -> dict[str, Any]:
    w = ev.window
    path = cache_dir / f"w{w.idx:04d}.json"
    while True:
        slot = plan.current
        if slot is None:
            return {"window": w.idx, "start": w.start, "end": w.end, "status": "signals_only", "provider": "none",
                    "model": "", "mode": "signals", "issues": [], "events": 0}
        if slot.mode == "video" and w.idx not in video_windows:
            slot = ProviderSlot(slot.provider, "images")
        key = hashlib.sha256(json.dumps([ev.fingerprint(), slot.provider.name, slot.provider.model, slot.mode,
                                         PROMPT_VERSION, cfg.get("temperature")]).encode()).hexdigest()[:24]
        if path.exists():
            cached = json.loads(path.read_text("utf-8"))
            if cached.get("key") == key and cached.get("status") == "ok":
                return {k: v for k, v in cached.items() if k not in ("result", "prompt")} | {"cached": True}
        if not budget.take(cancel_check):
            plan.drop_to_signals(f"the request limit for one job ({budget.max_requests}) was reached")
            continue
        try:
            raw, shown, prompt = _with_retries(
                lambda: _call(slot, ev, resolve, cfg, video_chunk), sleep, cancel_check, retry_delays(slot, cfg),
                (lambda msg: notify(f"Window {w.idx + 1}: {msg}")) if notify else None)
            issues: list[str] = []
            try:
                wa, issues = validate_result(raw, ev, slot.mode, shown, duration)
            except InvalidOutput as first:
                if not budget.take(cancel_check):
                    raise
                fix = (f"Your previous answer was invalid: {first}. Return ONLY corrected JSON for the same schema. "
                       f"Previous answer:\n{raw[:4000]}")
                raw = slot.provider.generate_json(_system_prompt(slot.mode, ev.context["game"]),
                                                  prompt + "\n\n" + fix, flat_schema(WindowAnalysis))
                wa, issues = validate_result(raw, ev, slot.mode, shown, duration)
                issues.insert(0, "First answer was invalid; a repaired answer was used.")
        except QuotaExceeded as q:
            if q.retry_after_sec and q.retry_after_sec <= 60 and not getattr(q, "_retried", False):
                _sleep_cancellable(q.retry_after_sec, sleep, cancel_check)
                continue
            if plan.on_quota == "fallback" and plan.switch_to_fallback(str(q)):
                continue
            raise StagePaused(f"{q} Analysis paused at window {w.idx + 1}; finished windows are kept. Resume later "
                              "(free quotas usually reset daily) or switch to Ollama.") from q
        except ProviderUnavailable as u:
            if not plan.switch_to_fallback(str(u)):
                plan.drop_to_signals(str(u))
            continue
        except InvalidOutput as bad:
            entry = {"window": w.idx, "start": w.start, "end": w.end, "status": "invalid", "provider": slot.provider.name,
                     "model": slot.provider.model, "mode": slot.mode, "key": key, "events": 0,
                     "issues": [f"AI output unusable after one repair attempt ({bad}); this window uses signals only "
                                "and needs your review."]}
            path.write_text(json.dumps(entry, indent=1), encoding="utf-8")
            return entry
        except ProviderError as err:
            entry = {"window": w.idx, "start": w.start, "end": w.end, "status": "failed", "provider": slot.provider.name,
                     "model": slot.provider.model, "mode": slot.mode, "key": key, "events": 0,
                     "issues": [f"AI request failed ({err}); this window uses signals only."]}
            return entry  # not cached: a later run retries it
        entry = {"window": w.idx, "start": w.start, "end": w.end, "status": "ok", "provider": slot.provider.name,
                 "model": slot.provider.model, "mode": slot.mode, "key": key, "frames_shown": shown,
                 "events": len(wa.events), "issues": issues, "result": wa.model_dump(), "prompt": prompt}
        tmp = path.with_suffix(".partial.json")
        tmp.write_text(json.dumps(entry, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return {k: v for k, v in entry.items() if k not in ("result", "prompt")}


def _sleep_cancellable(sec: float, sleep: Callable[[float], None], cancel_check: Callable[[], bool]) -> None:
    left = sec
    while left > 0:
        if cancel_check():
            raise Cancelled("Cancelled while waiting for the AI rate limit")
        sleep(min(left, 1.0))
        left -= 1.0


LOCAL_RETRY_DELAYS = (5.0, 20.0)


def retry_delays(slot: ProviderSlot, cfg: dict[str, Any]) -> tuple[float, ...]:
    """Cloud overload (e.g. Gemini HTTP 503) usually clears within minutes, so cloud retries wait longer
    (configurable). A local timeout does not improve by waiting, so local retries stay short."""
    if slot.provider.cloud:
        return tuple(float(x) for x in cfg.get("retry_delays_sec") or (30, 60, 120))
    return LOCAL_RETRY_DELAYS


def _with_retries(fn: Callable[[], Any], sleep: Callable[[float], None], cancel_check: Callable[[], bool],
                  delays: tuple[float, ...] = LOCAL_RETRY_DELAYS,
                  notify: Callable[[str], None] | None = None) -> Any:
    for attempt in range(len(delays) + 1):
        try:
            return fn()
        except ProviderError as e:
            if not e.transient or isinstance(e, (QuotaExceeded, ProviderUnavailable)) or attempt == len(delays):
                raise
            if notify:
                notify(f"{str(e)[:120]} Retrying in {delays[attempt]:.0f} s (attempt {attempt + 2} of {len(delays) + 1}).")
            _sleep_cancellable(delays[attempt], sleep, cancel_check)
    raise AssertionError("unreachable")


def load_window_results(cache_dir: Path, report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Results of THIS run only: a cache file is used when its key matches the key the run's report recorded
    (so results from an earlier provider/model or evidence never leak into a new analysis)."""
    out = {}
    for entry in report.get("windows", []):
        if entry.get("status") != "ok":
            continue
        p = cache_dir / f"w{entry['window']:04d}.json"
        if p.exists():
            d = json.loads(p.read_text("utf-8"))
            if d.get("key") == entry.get("key") and d.get("status") == "ok":
                out[entry["window"]] = d
    return out
