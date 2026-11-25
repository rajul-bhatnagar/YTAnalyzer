"""Thumbnail preparation: candidate frames, optional cut-outs, Gemini-web prompts and user thumbnail import.

The app never generates the thumbnail image itself and never calls a paid image API. It prepares clear frames and
two ready-to-copy prompts; the user creates the image in the free Gemini web app and may import the result.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import ffmpeg_tools as fx
from .ffmpeg_tools import Cancelled
from .frame_sampler import fmt_ts

MAX_W = 1920


def candidate_times(highlights: list[dict[str, Any]], duration: float) -> list[tuple[float, str, int | None]]:
    """(time, why, highlight id) around each highlight's key moment. Audio peaks alone are never used."""
    out: list[tuple[float, str, int | None]] = []
    for h in highlights:
        s, e, ev = h["eff_start"], h["eff_end"], h["event_sec"]
        pts = [(ev, "key moment"), (ev - 0.7, "just before the key moment"), (ev + 0.8, "just after the key moment"),
               ((s + e) / 2, "middle of the clip")]
        for v in (h.get("evidence") or {}).get("visual", []):
            pts.append((v["t"], "frame cited as evidence"))
        for t, why in pts:
            if 0 <= t < duration and s - 1 <= t <= e + 1:
                out.append((round(t, 2), why, h["id"]))
    seen: set[float] = set()
    uniq = []
    for t, why, hid in sorted(out):
        if all(abs(t - x) >= 0.3 for x in seen):
            seen.add(t)
            uniq.append((t, why, hid))
    return uniq


def extract_still(source: Path, t: float, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.png")
    try:
        fx.ffmpeg(["-y", "-ss", f"{t:.3f}", "-i", str(source), "-map", "0:v:0", "-frames:v", "1",
                   "-vf", f"scale='min({MAX_W},iw)':-2:flags=lanczos", str(tmp)], timeout=180)
    except fx.FFmpegError:
        tmp.unlink(missing_ok=True)
        return False
    if not tmp.exists() or tmp.stat().st_size < 500:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True


def frame_quality(path: Path) -> dict[str, Any]:
    """Sharpness (Laplacian variance), exposure, colourfulness (Hasler & Suesstrunk) and a small fingerprint."""
    import cv2  # noqa: PLC0415

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False}
    small = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    b, g, r = [c.astype(np.float32) for c in cv2.split(small)]
    rg, yb = np.abs(r - g), np.abs(0.5 * (r + g) - b)
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    issues = []
    if bright < 25:
        issues.append("too dark")
    if bright > 235:
        issues.append("washed out")
    if sharp < 30:
        issues.append("blurry")
    score = (0.45 * min(sharp / 600.0, 1.0) + 0.25 * min(colorful / 80.0, 1.0)
             + 0.30 * (1.0 - abs(bright - 128.0) / 128.0))
    thumb = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
    return {"ok": True, "sharpness": round(sharp, 1), "brightness": round(bright, 1),
            "colorfulness": round(colorful, 1), "score": round(score - 0.3 * len(issues), 3), "issues": issues,
            "_thumb": thumb, "width": int(img.shape[1]), "height": int(img.shape[0])}


def select_frames(source: Path, highlights: list[dict[str, Any]], duration: float, out_dir: Path, count: int = 5,
                  cancel_check: Callable[[], bool] | None = None) -> list[dict[str, Any]]:
    """Extract candidates, drop unusable and near-identical frames, keep the best ``count`` (spread over clips)."""
    cand_dir = out_dir / "_candidates"
    for old in cand_dir.glob("*.png") if cand_dir.exists() else []:
        old.unlink()
    scored = []
    for t, why, hid in candidate_times(highlights, duration):
        if cancel_check and cancel_check():
            raise Cancelled("Cancelled during thumbnail frame extraction")
        p = cand_dir / f"cand_{int(t * 1000):09d}.png"
        if not extract_still(source, t, p):
            continue
        q = frame_quality(p)
        if q.get("ok"):
            scored.append({"t": t, "why": why, "highlight_id": hid, "path": p, **q})
    scored.sort(key=lambda x: -x["score"])
    chosen: list[dict[str, Any]] = []
    for c in scored:
        if len(chosen) >= count:
            break
        if any(float(np.mean(np.abs(c["_thumb"] - k["_thumb"]))) < 6.0 for k in chosen):
            continue  # near-identical to a frame already chosen
        chosen.append(c)
    for old in out_dir.glob("thumb_*.png"):
        old.unlink()
    result = []
    for i, c in enumerate(sorted(chosen, key=lambda x: x["t"]), 1):
        dest = out_dir / f"thumb_{i:02d}_{int(c['t'] * 1000):09d}.png"
        shutil.copy2(c["path"], dest)
        result.append({k: v for k, v in c.items() if not k.startswith("_") and k != "path"} | {"file": dest})
    shutil.rmtree(cand_dir, ignore_errors=True)
    return result


# --- optional cut-outs -------------------------------------------------------------------------------------
def rembg_available() -> bool:
    try:
        import rembg  # noqa: F401, PLC0415
        return True
    except Exception:  # noqa: BLE001 - not installed or broken: the feature is simply unavailable
        return False


def make_cutout(src: Path, out: Path) -> Path:
    """Background removal with the optional rembg package (UNVERIFIED in this project: rembg is not installed on
    the dev PC). rembg may download its segmentation model on first use; the UI asks before running it."""
    from PIL import Image  # noqa: PLC0415
    from rembg import remove  # noqa: PLC0415

    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        res = remove(im)
    res.save(out, format="PNG")
    return out


# --- prompts ---------------------------------------------------------------------------------------------------
_TRAILING_BAD = {"OF", "TO", "IN", "ON", "AT", "FOR", "WITH", "BY", "FROM", "OFF", "INTO", "ONTO", "OVER", "UNDER",
                 "THE", "A", "AN", "AND", "OR", "BUT", "SO", "IS", "ARE", "WAS", "MY", "YOUR", "HIS", "HER", "THEIR"}

MOOD_LOOK = {
    "win_hype": "bright, energetic, triumphant colours and lighting",
    "clutch_tension": "dramatic, high-contrast lighting with a darker vignette that builds tension",
    "fail_shock": "a surprised, dramatic look with bold red accents",
    "funny_chaos": "playful, saturated colours with lively, chaotic energy",
    "": "clear, high-contrast colours",
}
MOOD_TEXT_COLOR = {"win_hype": (255, 214, 0), "clutch_tension": (255, 255, 255), "fail_shock": (255, 59, 48),
                   "funny_chaos": (0, 229, 255), "": (255, 255, 255)}
FONT_CANDIDATES = ("impact.ttf", "ariblk.ttf", "arialbd.ttf", "seguibl.ttf")


def validate_words(text: str) -> tuple[str | None, str | None]:
    """(clean UPPERCASE words, None) or (None, reason): 2-3 complete words, no trailing preposition/article."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’!?-]*", text or "")
    if not 2 <= len(words) <= 3:
        return None, "Use 2 or 3 words."
    up = [w.upper() for w in words]
    if up[-1].rstrip("!?") in _TRAILING_BAD:
        return None, f"Don't end on '{up[-1]}': the phrase reads unfinished."
    return " ".join(up), None


def suggest_words(hook: str) -> str:
    """2-3 uppercase words from the hook (trailing prepositions dropped), or '' when that leaves fewer than 2."""
    words = [w.upper() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’!?-]*", hook or "")][:3]
    while words and words[-1].rstrip("!?") in _TRAILING_BAD:
        words.pop()
    return " ".join(words) if len(words) >= 2 else ""


def thumbnail_words(hook: str, max_words: int = 3) -> str:  # kept for older callers
    return suggest_words(hook)


def moment_text(h: dict[str, Any]) -> tuple[str, bool]:
    """Best factual description of a highlight and whether it needs the user's words (placeholder)."""
    for key in ("suggested_title", "hook_caption"):
        if (h.get(key) or "").strip():
            return h[key].strip(), False
    reason = (h.get("rationale") or "").split(" Supported by measurements")[0].split(" (No AI description")[0]
    if reason and not reason.startswith("Measured:"):
        return reason.strip().rstrip("."), False
    return "[describe the moment in your own words]", True


def build_prompts(game: str, style: str, h: dict[str, Any], frame_time: float | None) -> dict[str, Any]:
    """Two TEXT-FREE prompts for the free Gemini web app; the look follows the clip's outcome mood. The title words
    are added afterwards by the local composer (compose_thumbnail), so the image model never misspells them."""
    moment, placeholder = moment_text(h)
    mood = h.get("outcome_mood") or ""
    look = MOOD_LOOK.get(mood, MOOD_LOOK[""])
    no_text = ("Do not add any text, letters or numbers to the image; the title words are added afterwards. "
               "Leave the upper-left area simple enough to hold 2-3 bold words.")
    common = ("Leave the bottom-right corner empty, because YouTube shows the video length there. "
              "Do not show anything that does not happen in the video: no extra characters, explosions, scores, "
              "trophies or logos that are not visible in the gameplay.")
    at = f" (screenshot at {fmt_ts(frame_time)})" if frame_time is not None else ""
    ref = (f"Turn the attached gameplay screenshot from the video game \"{game}\"{at} into a bold 16:9 YouTube "
           f"thumbnail (1280x720). The moment: {moment}. Keep the scene and the main subject recognisable and true to "
           f"the screenshot; crop or zoom to make the main action large and clear, and simplify the background. "
           f"Look: {look}. {no_text} {common}")
    scratch = (f"Create a 16:9 YouTube thumbnail (1280x720) for a \"{style}\" gaming video of the video game "
               f"\"{game}\". It shows this real moment from the video: {moment}. One clear focal subject, "
               f"uncluttered background, readable on a phone. Look: {look}. {no_text} {common}")
    return {"reference_edit": ref, "from_scratch": scratch, "text": suggest_words(h.get("hook_caption") or ""),
            "mood": mood, "needs_user_words": placeholder, "moment": moment}


def _bold_font(size: int) -> Any:
    import os  # noqa: PLC0415

    from PIL import ImageFont  # noqa: PLC0415

    fonts = Path(os.environ.get("WINDIR", r"C:\\Windows")) / "Fonts"
    for name in FONT_CANDIDATES:
        if (fonts / name).exists():
            return ImageFont.truetype(str(fonts / name), size)
    raise ValueError("No bold font found (looked for Impact, Arial Black, Arial Bold, Segoe UI Black).")


def compose_thumbnail(image: Path, words: str, mood: str, out: Path) -> Path:
    """1280x720 thumbnail: the image cropped to 16:9 (never stretched) with 2-3 bold uppercase words in the upper
    left, coloured by the outcome mood, with a thick outline. The bottom-right timestamp area is left untouched."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    clean, problem = validate_words(words)
    if clean is None:
        raise ValueError(problem)
    with Image.open(image) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w / h > 16 / 9:
            nw = int(h * 16 / 9)
            im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else:
            nh = int(w * 9 / 16)
            im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
        im = im.resize((1280, 720), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    parts = clean.split()
    lines = [clean]
    size = 150
    while size > 60:
        font = _bold_font(size)
        widest = max(d.textbbox((0, 0), ln, font=font)[2] for ln in lines)
        if widest <= 0.62 * 1280:
            break
        if len(lines) == 1 and len(parts) >= 2:
            cut = 2 if len(parts) == 3 else 1
            lines = [" ".join(parts[:cut]), " ".join(parts[cut:])]
            continue
        size -= 10
    font = _bold_font(size)
    y = 36
    for ln in lines:
        d.text((44, y), ln, font=font, fill=MOOD_TEXT_COLOR.get(mood, MOOD_TEXT_COLOR[""]),
               stroke_width=max(size // 12, 4), stroke_fill=(0, 0, 0))
        y += int(size * 1.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return out


# --- user thumbnail import ---------------------------------------------------------------------------------------
def validate_user_thumbnail(path: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    """(errors, warnings, info). Size/aspect guidance follows YouTube's published recommendations as known to the
    app (1280x720, 16:9, under 2 MB); YouTube may change them, so these are warnings except for unreadable files."""
    from PIL import Image  # noqa: PLC0415

    errors, warnings = [], []
    try:
        with Image.open(path) as im:
            w, h = im.size
            fmt = im.format
    except Exception as e:  # noqa: BLE001
        return [f"Not a readable image: {type(e).__name__}"], [], {}
    if fmt not in ("PNG", "JPEG"):
        warnings.append(f"Format {fmt}; PNG or JPG is the safest choice for YouTube.")
    if abs(w / h - 16 / 9) > 0.02:
        warnings.append(f"Aspect ratio {w}x{h} is not 16:9; YouTube will letterbox or crop it.")
    if w < 1280 or h < 720:
        warnings.append(f"{w}x{h} is below the recommended 1280x720.")
    size = path.stat().st_size
    if size > 2 * 1024 * 1024:
        warnings.append(f"File is {size / 1048576:.1f} MB; YouTube's limit is about 2 MB (verify in YouTube Studio).")
    return errors, warnings, {"width": w, "height": h, "format": fmt, "size": size}


def safe_area_preview(src: Path, out: Path, badge: bool = False) -> Path:
    """The thumbnail with the bottom-right timestamp area outlined, or (``badge``) a mock '12:34' length badge drawn
    where YouTube puts it. Approximate: YouTube does not publish the exact size."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im, "RGBA")
        bw, bh = int(w * 0.16), int(h * 0.12)
        box = (w - bw - int(w * 0.015), h - bh - int(h * 0.02), w - int(w * 0.015), h - int(h * 0.02))
        if badge:
            bx0, by0 = box[0] + bw // 3, box[1] + bh // 3
            d.rounded_rectangle((bx0, by0, box[2], box[3]), radius=max(h // 120, 4), fill=(0, 0, 0, 200))
            try:
                font = _bold_font(max(int((box[3] - by0) * 0.62), 10))
                d.text(((bx0 + box[2]) / 2, (by0 + box[3]) / 2), "12:34", font=font, fill=(255, 255, 255), anchor="mm")
            except ValueError:
                pass
        else:
            d.rectangle(box, fill=(255, 0, 0, 70), outline=(255, 0, 0, 255), width=max(w // 400, 2))
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out)
    return out
