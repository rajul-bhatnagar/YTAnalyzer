"""Creator package: titles, opening hook, timestamped voiceover script, Shorts titles/hooks, description, tags,
thumbnail prompts, pinned comment, rationale and uncertainty notes.

Facts come only from the evidence (approved highlights, detected events, transcript, the creator's own text).
An AI provider may write the wording; every draft is validated:
- numbers that do not appear in the evidence are flagged (scores, results, records are a common invention);
- script lines may not describe an event before it starts;
- lengths and counts are checked.
Without an AI provider a plain template with [placeholders] is produced instead of invented text.
The user's edits are stored per field and survive regeneration unless the user chooses to discard them.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .ai_providers.base import flat_schema
from .frame_sampler import fmt_ts

# YouTube limits as published by YouTube and known to the app; they can change, so the UI says "verify".
TITLE_MAX = 100
DESCRIPTION_MAX = 5000
TAGS_TOTAL_MAX = 500
EDITABLE_FIELDS = ["titles", "opening_hook", "script", "shorts", "description", "tags", "pinned_comment"]
LANG_NAMES = {"en": "English", "hi": "Hindi (Devanagari)", "hinglish": "Hinglish (natural Hindi-English mix in "
              "Latin script)", "auto": "the language of the transcript (English if unclear)"}


class ScriptLine(BaseModel):
    time_sec: float = Field(description="Recording time in seconds when this line should be spoken")
    text: str = Field(max_length=400)
    event_key: str = Field("", description="Key of the event this line describes, or '' for intro/outro")
    uncertain: bool = Field(False, description="True if this interpretation needs the creator's check")


class ShortText(BaseModel):
    highlight_id: int
    title: str = Field(max_length=100)
    hook: str = Field(max_length=80)


class PackDraft(BaseModel):
    titles: list[str] = Field(default_factory=list, max_length=8)
    opening_hook: str = Field("", max_length=700, description="About 20 seconds of spoken words (~50 words)")
    script: list[ScriptLine] = Field(default_factory=list, max_length=80)
    shorts: list[ShortText] = Field(default_factory=list, max_length=20)
    description: str = Field("", max_length=4500)
    tags: list[str] = Field(default_factory=list, max_length=15)
    pinned_comment: str = Field("", max_length=500)
    style_rationale: str = Field("", max_length=600)
    uncertainty: list[str] = Field(default_factory=list, max_length=20)


class InvalidPack(ValueError):
    pass


# --- evidence -----------------------------------------------------------------------------------------------
def evidence_text(ctx: dict[str, Any]) -> str:
    parts = [ctx["game"], ctx.get("topic", ""), *[x["text"] for x in ctx.get("transcript", [])], *ctx.get("ocr", [])]
    for e in ctx.get("events", []):
        parts.append(e["description"])
    for h in ctx.get("highlights", []):
        parts += [h.get("title", ""), h.get("hook", ""), h.get("what", "")]
    return " ".join(p for p in parts if p)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)?", text or "")}


_NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                 "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
                 "hundred": "100", "thousand": "1000"}


def allowed_numbers(ctx: dict[str, Any]) -> set[str]:
    text = evidence_text(ctx)
    allowed = _numbers(text)
    for word, digit in _NUMBER_WORDS.items():  # speech recognition may write "six" where a title says "6"
        if re.search(rf"\b{word}\b", text, re.I):
            allowed.add(digit)
    return allowed


def unsupported_numbers(text: str, allowed: set[str]) -> list[str]:
    return sorted(n for n in _numbers(text) if n not in allowed)


# --- prompt -----------------------------------------------------------------------------------------------------
def build_prompt(ctx: dict[str, Any]) -> tuple[str, str]:
    lang = LANG_NAMES.get(ctx["language"], "English")
    system = ("You write YouTube packaging for a gaming creator. Rules:\n"
              "- Use ONLY facts in the evidence below. Never invent player names, scores, results, records, rules, "
              "dialogue or events. If a fact is unknown, write a placeholder in square brackets, e.g. [final score].\n"
              "- No clickbait that promises something not shown in the recording.\n"
              "- Script lines describe events in recording order and never before they happen; set time_sec at or "
              "after the event's start and fill event_key with its key. Mark interpretations you are unsure about "
              "with uncertain=true.\n"
              f"- Write all viewer-facing text in {lang}.\n")
    L = [f"Game: {ctx['game']}", f"Content style: {ctx['style']}", f"Recording length: {fmt_ts(ctx['duration'])}"]
    if ctx.get("topic"):
        L.append(f"Creator's note: {ctx['topic']}")
    L.append("\nApproved highlights (the creator chose these):")
    for h in ctx["highlights"] or [{"id": None}]:
        if h.get("id") is None:
            L.append("(none approved yet)")
            break
        L.append(f"- id={h['id']} | {fmt_ts(h['start'])}-{fmt_ts(h['end'])} | key moment {fmt_ts(h['event'])} | "
                 f"creator title: {h.get('title') or '-'} | creator hook: {h.get('hook') or '-'} | what: {h['what']}")
    L.append("\nDetected events (key | time | category | description; 'Measured:' = software measurements only):")
    for e in ctx["events"][:60]:
        L.append(f"- {e['key']} | {fmt_ts(e['start'])}-{fmt_ts(e['end'])} | {e['category']} | {e['description']}")
    L.append("\nTranscript of the player's voice (speech recognition; may contain errors):")
    used = 0
    for x in ctx.get("transcript", []):
        line = f"[{fmt_ts(x['start'])}] {x['text']}"
        used += len(line)
        if used > 4000:
            L.append("[... transcript shortened ...]")
            break
        L.append(line)
    if not ctx.get("transcript"):
        L.append("(no speech)")
    L.append("\nWrite: 5 different titles (under 70 characters each), an opening hook of about 20 seconds spoken "
             "(~50 words), a timestamped play-by-play voiceover script following the events in order, a title and "
             "short hook for each approved highlight (by id), a YouTube description, 10 tags without '#', a "
             "suggested pinned comment, one or two sentences on why this fits the content style, and a list of "
             "anything uncertain the creator should check.")
    return system, "\n".join(L)


def schema() -> dict[str, Any]:
    return flat_schema(PackDraft)


# --- validation -------------------------------------------------------------------------------------------------
def validate_draft(raw: str, ctx: dict[str, Any]) -> tuple[PackDraft, list[str]]:
    try:
        d = PackDraft.model_validate_json(raw)
    except ValidationError as e:
        raise InvalidPack(f"{e.errors()[0]['loc']} {e.errors()[0]['msg']}") from e
    except ValueError as e:
        raise InvalidPack(f"not valid JSON: {e}") from e
    issues: list[str] = []
    titles, seen = [], set()
    for t in (x.strip() for x in d.titles):
        if t and t.lower() not in seen:
            seen.add(t.lower())
            titles.append(t[:TITLE_MAX])
    if len(titles) < 5:
        issues.append(f"Only {len(titles)} distinct titles were written; placeholders fill the rest.")
        titles += [f"[Title idea {i + 1}]" for i in range(len(titles), 5)]
    d.titles = titles[:5]
    events = {e["key"]: e for e in ctx["events"]}
    lines = []
    for ln in sorted(d.script, key=lambda x: x.time_sec):
        if not 0 <= ln.time_sec <= ctx["duration"]:
            issues.append(f"Dropped a script line at {ln.time_sec:.0f} s outside the recording.")
            continue
        ev = events.get(ln.event_key)
        if ln.event_key and ev is None:
            ln.event_key, ln.uncertain = "", True
            issues.append(f"Script line at {fmt_ts(ln.time_sec)} referenced an unknown event; marked for review.")
        elif ev is not None and ln.time_sec < ev["start"] - 1.0:
            ln.time_sec, ln.uncertain = ev["start"], True
            issues.append(f"Script line moved to {fmt_ts(ev['start'])}: it described an event before it happened.")
        lines.append(ln)
    d.script = lines
    approved = {h["id"] for h in ctx["highlights"]}
    d.shorts = [s for s in d.shorts if s.highlight_id in approved]
    tags = []
    for t in d.tags:
        t = t.strip().lstrip("#").strip()
        if t and t.lower() not in {x.lower() for x in tags}:
            tags.append(t[:60])
    while sum(len(t) for t in tags) + max(len(tags) - 1, 0) > TAGS_TOTAL_MAX:
        tags.pop()
    d.tags = tags[:10]
    if len(d.description) > DESCRIPTION_MAX:
        d.description = d.description[:DESCRIPTION_MAX]
    return d, issues


def number_flags(pack: dict[str, Any], ctx: dict[str, Any]) -> list[str]:
    allowed = allowed_numbers(ctx)
    fields = [(f"Title {i + 1}", t) for i, t in enumerate(pack["titles"])]
    fields += [("Opening hook", pack["opening_hook"]), ("Description", pack["description"]),
               ("Pinned comment", pack["pinned_comment"])]
    fields += [(f"Script at {fmt_ts(x['time_sec'])}", x["text"]) for x in pack["script"]]
    fields += [(f"Short #{s['highlight_id']} title", s["title"]) for s in pack["shorts"]]
    out = []
    for name, text in fields:
        bad = unsupported_numbers(text, allowed)
        if bad:
            out.append(f"{name} mentions {', '.join(bad)}, which does not appear in the evidence. Check it before "
                       "publishing.")
    return out


# --- template fallback ---------------------------------------------------------------------------------------------
def template_draft(ctx: dict[str, Any]) -> PackDraft:
    game, style = ctx["game"], ctx["style"]
    hs = ctx["highlights"]
    named = [h.get("title") for h in hs if h.get("title")]
    titles = list(dict.fromkeys(named))[:3]
    ideas = [f"{game}: [your headline about {ctx.get('topic') or 'this session'}]", f"[Best moment] in {game}",
             f"{game} – {style.split(' / ')[0]} [what happens]", f"[Question viewers want answered] | {game}",
             f"I tried [your challenge] in {game}"]
    titles += ideas[: 5 - len(titles)]  # always exactly 5 options
    script = [ScriptLine(time_sec=0.0, text=f"[Intro: what this {game} video is about]", event_key="")]
    for e in ctx["events"]:
        if e.get("chosen"):
            text = e["description"] if not e["description"].startswith("Measured:") else \
                f"[Describe what happens here. Measured: {e['description'].removeprefix('Measured: ')}]"
            script.append(ScriptLine(time_sec=e["start"], text=text, event_key=e["key"], uncertain=True))
    lines = [f"[One or two sentences about this {game} video.]", ""]
    if hs:
        lines.append("Highlights:")
        lines += [f"{fmt_ts(h['event'])} {h.get('title') or '[moment]'}" for h in hs]
    return PackDraft(
        titles=titles[:5], opening_hook=f"[About 20 seconds: set up the {game} session and promise only what "
                                        "really happens.]",
        script=script, shorts=[ShortText(highlight_id=h["id"], title=(h.get("title") or "[Short title]")[:100],
                                         hook=(h.get("hook") or "[hook]")[:80]) for h in hs],
        description="\n".join(lines),
        tags=[game, f"{game} gameplay", "PC gaming", "gameplay", style.split(" / ")[0], "gaming",
              f"{game} highlights", "no commentary" if not ctx.get("transcript") else "commentary", "Shorts",
              "let's play"][:10],
        pinned_comment="[Ask viewers a question about the video]",
        style_rationale=f"Template for the '{style}' style; replace the placeholders in square brackets.",
        uncertainty=["Written without AI: every [placeholder] needs your words."])


# --- assembling, edits, export ------------------------------------------------------------------------------------
def assemble(draft: PackDraft, ctx: dict[str, Any], source: dict[str, Any], issues: list[str],
             prompts: dict[str, Any] | None, video_time: Any = None) -> dict[str, Any]:
    pack = draft.model_dump()
    for ln in pack["script"]:
        ln["video_time_sec"] = video_time(ln["time_sec"]) if video_time else ln["time_sec"]
    by_id = {h["id"]: h for h in ctx["highlights"]}
    known = {s["highlight_id"] for s in pack["shorts"]}
    for h in ctx["highlights"]:  # every approved highlight gets an entry; the creator's own words win
        if h["id"] not in known:
            pack["shorts"].append({"highlight_id": h["id"], "title": h.get("title") or "[Short title]",
                                   "hook": h.get("hook") or "[hook]"})
    for s in pack["shorts"]:
        h = by_id[s["highlight_id"]]
        if h.get("title"):
            s["title"] = h["title"]
        if h.get("hook"):
            s["hook"] = h["hook"]
        s["start"], s["end"] = h["start"], h["end"]
        s["outcome_mood"] = h.get("mood", "")
    pack["thumbnail_prompts"] = prompts or {}
    pack["source"] = source
    pack["language"] = ctx["language"]
    pack["evidence_notes"] = [f"{len(ctx['highlights'])} approved highlight(s), {len(ctx['events'])} detected "
                              f"event(s), transcript {'available' if ctx.get('transcript') else 'not available'}."]
    pack["uncertainty"] = [*issues, *pack.get("uncertainty", [])]
    pack["number_flags"] = number_flags(pack, ctx)
    return pack


def apply_edits(pack: dict[str, Any], edits: dict[str, Any]) -> dict[str, Any]:
    out = dict(pack)
    for k, v in edits.items():
        if k in EDITABLE_FIELDS:
            out[k] = v
    out["edited_fields"] = sorted(k for k in edits if k in EDITABLE_FIELDS)
    return out


def to_text(pack: dict[str, Any]) -> str:
    L = ["YOUTUBE CREATOR PACKAGE", "=" * 24, ""]
    src = pack.get("source") or {}
    L.append(f"Written by: {src.get('provider', 'template')} {src.get('model', '')}".strip())
    L += ["", "TITLE OPTIONS", *[f"{i + 1}. {t}" for i, t in enumerate(pack["titles"])], "", "OPENING HOOK (~20 s)",
          pack["opening_hook"], "", "VOICEOVER SCRIPT (recording time | rough-cut time)"]
    for ln in pack["script"]:
        vt = ln.get("video_time_sec")
        vts = fmt_ts(vt) if vt is not None else "cut"
        L.append(f"[{fmt_ts(ln['time_sec'])} | {vts}] {ln['text']}{'  (check)' if ln.get('uncertain') else ''}")
    L += ["", "SHORTS"]
    for s in pack["shorts"]:
        L.append(f"- {fmt_ts(s.get('start', 0))}-{fmt_ts(s.get('end', 0))}: {s['title']} | hook: {s['hook']}")
    L += ["", "DESCRIPTION", pack["description"], "", "TAGS", ", ".join(pack["tags"]), ""]
    tp = pack.get("thumbnail_prompts") or {}
    if tp:
        L += ["THUMBNAIL PROMPT 1 (edit an uploaded screenshot)", tp.get("reference_edit", ""), "",
              "THUMBNAIL PROMPT 2 (from scratch)", tp.get("from_scratch", ""), ""]
    L += ["PINNED COMMENT", pack["pinned_comment"], "", "WHY THIS FITS THE STYLE", pack["style_rationale"], ""]
    checks = [*pack.get("number_flags", []), *pack.get("uncertainty", [])]
    if checks:
        L += ["CHECK BEFORE PUBLISHING", *[f"- {c}" for c in checks]]
    return "\n".join(L) + "\n"
