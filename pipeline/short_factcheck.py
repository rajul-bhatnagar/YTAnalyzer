"""Fact-check a Short's hook caption and title against the evidence INSIDE its clip [start, end].

Evidence = transcript words, the AI's observations of events overlapping the clip, and HUD text read by OCR in the
clip, plus the game name. Two kinds of problem are flagged (never silently changed):
- "outside": a word or number that occurs in the recording's evidence but not inside this clip (e.g. a hook about a
  wicket that happened in another part of the match);
- "unsupported_number": a number found nowhere in the evidence.
Style words ("insane", "clutch") are not evidence claims and are not flagged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .database import Database, from_json
from .youtube_metadata import _NUMBER_WORDS

_STOP = {"this", "that", "with", "from", "what", "when", "they", "them", "have", "just", "into", "over", "your", "will",
         "then", "than", "were", "been", "their", "there", "here", "about", "after", "before", "only", "very", "most",
         "more", "some", "every", "watch", "look", "wait", "game", "play", "moment", "best", "ever", "time"}


def _norm(word: str) -> str:
    w = word.lower()
    return w[:-1] if len(w) > 4 and w.endswith("s") else w


def tokens(text: str) -> tuple[set[str], set[str]]:
    """(numbers, content words) of a text. Spelled-out numbers count as their digits."""
    nums = {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)?", text or "")}
    words = set()
    for w in re.findall(r"[A-Za-z][A-Za-z'’-]+", text or ""):
        lw = w.lower()
        if lw in _NUMBER_WORDS:
            nums.add(_NUMBER_WORDS[lw])
        elif len(lw) >= 4 and lw not in _STOP:
            words.add(_norm(lw))
    return nums, words


def check_text(text: str, window_text: str, all_text: str) -> list[dict[str, str]]:
    n_in, w_in = tokens(window_text)
    n_all, w_all = tokens(all_text)
    n, w = tokens(text)
    issues = []
    for x in sorted(n):
        if x not in n_in:
            kind = "outside" if x in n_all else "unsupported_number"
            issues.append({"kind": kind, "token": x})
    for x in sorted(w):
        if x not in w_in and x in w_all:
            issues.append({"kind": "outside", "token": x})
    return issues


def describe(field: str, issues: list[dict[str, str]]) -> list[str]:
    out = []
    outside = [i["token"] for i in issues if i["kind"] == "outside"]
    unsupported = [i["token"] for i in issues if i["kind"] == "unsupported_number"]
    if outside:
        out.append(f"{field} mentions {', '.join(outside)}: found elsewhere in the recording but not inside this clip.")
    if unsupported:
        out.append(f"{field} mentions {', '.join(unsupported)}: not found anywhere in the evidence.")
    return out


def gather_evidence(db: Database, project_id: str, words: list[dict[str, Any]], ocr_file: Path | None, game: str,
                    start: float, end: float) -> dict[str, Any]:
    """Evidence texts inside the clip and in the whole recording, plus in-clip AI hook suggestions."""
    win, allp, hooks = [game], [game], []
    for w in words:
        (win if start <= w["start"] <= end else allp).append(w["word"])
    for r in db.query("SELECT * FROM events WHERE project_id = ?", (project_id,)):
        ev = from_json(r["evidence"]) or {}
        ai = ev.get("ai") or {}
        parts = [r["description"], ai.get("visual_observation", ""), ai.get("transcript_quote", ""),
                 ai.get("audio_observation", "")]
        inside = r["start_sec"] < end and r["end_sec"] > start
        (win if inside else allp).extend(p for p in parts if p)
        if inside and ai.get("hook_caption"):
            hooks.append(ai["hook_caption"])
    if ocr_file is not None and ocr_file.exists():
        data = json.loads(ocr_file.read_text("utf-8"))
        for fr in data.get("frames", []):
            (win if start <= fr["t"] <= end else allp).extend(fr.get("texts", []))
    window_text = " ".join(win)
    return {"window": window_text, "all": window_text + " " + " ".join(allp), "hooks": hooks}


def check_short(hook: str, title: str, evidence: dict[str, Any]) -> dict[str, Any]:
    issues = {"hook": check_text(hook, evidence["window"], evidence["all"]) if hook else [],
              "title": check_text(title, evidence["window"], evidence["all"]) if title else []}
    messages = describe("Hook caption", issues["hook"]) + describe("Title", issues["title"])
    suggestions = [h for h in dict.fromkeys(evidence["hooks"])
                   if h.strip() and h.strip().lower() != hook.strip().lower()
                   and not check_text(h, evidence["window"], evidence["all"])]
    return {"ok": not messages, "messages": messages, "issues": issues, "suggestions": suggestions[:3]}
