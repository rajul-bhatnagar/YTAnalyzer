"""Burned-in captions for vertical Shorts as ASS subtitles (rendered by FFmpeg's libass ``ass`` filter).

- Word-by-word: words are grouped into short phrases; while a word is spoken it is drawn in the emphasis colour.
- Hook caption: an optional short line near the top for the first seconds (used when there is no speech, or always).
- Positions keep a margin from the bottom/top because platform UI overlays cover parts of a Short. The exact overlay
  area is set by YouTube and changes over time, so the margins are configurable defaults, not a guarantee.
pycaps is not used (no package of that name exists on PyPI; see docs/PHASE0_ADR.md).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

POSITIONS = {"bottom": "Lower third (above the Shorts UI)", "middle": "Centre", "top": "Upper third"}


class CaptionStyle(BaseModel):
    enabled: bool = True
    font: str = "Arial"
    size: int = Field(72, ge=24, le=160)
    bold: bool = True
    text_color: str = "#FFFFFF"
    emphasis_color: str = "#FFD400"
    outline_color: str = "#000000"
    outline: float = Field(5.0, ge=0, le=20)
    background_box: bool = False
    position: str = "bottom"
    margin_bottom: int = Field(420, ge=0, le=1200)
    margin_side: int = Field(60, ge=0, le=400)
    max_words: int = Field(4, ge=1, le=10)
    max_chunk_sec: float = Field(2.5, ge=0.5, le=8)
    uppercase: bool = True
    hook_enabled: bool = True
    hook_mode: str = "no_speech"  # "no_speech" | "always" | "never"
    hook_sec: float = Field(3.0, ge=0.5, le=10)
    hook_size: int = Field(80, ge=24, le=160)


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """#RRGGBB -> &HAABBGGRR (ASS stores colours as alpha + BGR)."""
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", hex_rgb.strip())
    if not m:
        raise ValueError(f"Not a colour: {hex_rgb!r} (use #RRGGBB)")
    rgb = m.group(1)
    return f"&H{alpha:02X}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}".upper()


def ass_escape(text: str) -> str:
    """User/AI text must never inject ASS override tags or line breaks."""
    text = re.sub(r"[\r\n]+", " ", text)
    return text.replace("\\", "⧵").replace("{", "(").replace("}", ")")


def _ts(sec: float) -> str:
    sec = max(sec, 0.0)
    cs = int(round(sec * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def clip_words(words: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    """Words (video time) inside [start, end], converted to clip time and clamped."""
    out = []
    dur = end - start
    for w in words:
        if w["end"] <= start or w["start"] >= end:
            continue
        s, e = max(w["start"] - start, 0.0), min(w["end"] - start, dur)
        text = w["word"].strip()
        if text and e > s:
            out.append({"start": round(s, 3), "end": round(e, 3), "word": text})
    return out


def chunk_words(words: list[dict[str, Any]], max_words: int, max_sec: float, gap: float = 0.6) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    for w in words:
        cur = chunks[-1] if chunks else None
        if (cur is None or len(cur) >= max_words or w["end"] - cur[0]["start"] > max_sec
                or w["start"] - cur[-1]["end"] > gap or cur[-1]["word"].endswith((".", "!", "?"))):
            chunks.append([w])
        else:
            cur.append(w)
    return chunks


def build_ass(words: list[dict[str, Any]], duration: float, style: CaptionStyle, hook: str = "",
              width: int = 1080, height: int = 1920) -> tuple[str, dict[str, Any]]:
    """Return (ASS text, info). ``words`` are in clip time (see clip_words)."""
    align = {"bottom": 2, "middle": 5, "top": 8}.get(style.position, 2)
    margin_v = style.margin_bottom if style.position == "bottom" else (style.margin_bottom // 2 if
                                                                         style.position == "top" else 0)
    border = 3 if style.background_box else 1
    back = ass_color("#000000", 0x60)
    cap = (f"Style: Caption,{style.font},{style.size},{ass_color(style.text_color)},{ass_color(style.text_color)},"
           f"{ass_color(style.outline_color)},{back},{-1 if style.bold else 0},0,0,0,100,100,0,0,{border},"
           f"{style.outline:g},0,{align},{style.margin_side},{style.margin_side},{margin_v},1")
    hook_style = (f"Style: Hook,{style.font},{style.hook_size},{ass_color(style.text_color)},"
                  f"{ass_color(style.text_color)},{ass_color(style.outline_color)},{back},-1,0,0,0,100,100,0,0,"
                  f"{border},{style.outline:g},0,8,{style.margin_side},{style.margin_side},220,1")
    lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "WrapStyle: 0",
             "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
             "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
             "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
             "MarginL, MarginR, MarginV, Encoding", cap, hook_style, "", "[Events]",
             "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    events: list[tuple[float, float]] = []
    emph, norm = ass_color(style.emphasis_color), ass_color(style.text_color)
    n_cues = 0
    if style.enabled:
        for chunk in chunk_words(words, style.max_words, style.max_chunk_sec):
            texts = [ass_escape(w["word"].upper() if style.uppercase else w["word"]) for w in chunk]
            for i, w in enumerate(chunk):
                s = w["start"]
                e = chunk[i + 1]["start"] if i + 1 < len(chunk) else w["end"]
                e = min(max(e, s + 0.05), duration)
                if s >= duration:
                    continue
                body = " ".join(f"{{\\c{emph}&}}{t}{{\\c{norm}&}}" if j == i else t for j, t in enumerate(texts))
                lines.append(f"Dialogue: 0,{_ts(s)},{_ts(e)},Caption,,0,0,0,,{body}")
                events.append((s, e))
                n_cues += 1
    show_hook = bool(hook.strip()) and style.hook_enabled and (
        style.hook_mode == "always" or (style.hook_mode == "no_speech" and not words))
    if show_hook:
        h_end = min(style.hook_sec, duration)
        text = ass_escape(hook.strip().upper() if style.uppercase else hook.strip())
        lines.append(f"Dialogue: 1,{_ts(0)},{_ts(h_end)},Hook,,0,0,0,,{text}")
    info = {"cues": n_cues, "hook_shown": show_hook, "first": events[0][0] if events else None,
            "last_end": max((e for _, e in events), default=None)}
    return "\n".join(lines) + "\n", info


def validate_ass_timing(ass_text: str, duration: float) -> list[str]:
    """Every Dialogue must start before it ends and lie inside the clip."""
    errors = []
    for m in re.finditer(r"^Dialogue: \d+,(\d+):(\d\d):(\d\d)\.(\d\d),(\d+):(\d\d):(\d\d)\.(\d\d),", ass_text, re.M):
        g = [int(x) for x in m.groups()]
        s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 100
        e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 100
        if not (0 <= s < e <= duration + 0.011):
            errors.append(f"Caption {s:.2f}-{e:.2f}s is outside 0-{duration:.2f}s")
    return errors
