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
def thumbnail_words(hook: str, max_words: int = 3) -> str:
    words = re.findall(r"[\w'’-]+", hook or "", flags=re.UNICODE)
    return " ".join(words[:max_words]).upper()


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
    moment, placeholder = moment_text(h)
    words = thumbnail_words(h.get("hook_caption") or "")
    text_rule = (f'Add the text "{words}" in large, bold, uppercase letters with a thick outline, placed in the top '
                 "area." if words else "Do not add any text (add 2-3 words of your own later if you want).")
    common = ("Leave the bottom-right corner empty, because YouTube shows the video length there. "
              "Do not show anything that does not happen in the video: no extra characters, explosions, scores, "
              "trophies or logos that are not visible in the gameplay.")
    at = f" (screenshot at {fmt_ts(frame_time)})" if frame_time is not None else ""
    ref = (f"Turn the attached gameplay screenshot from the PC game \"{game}\"{at} into a bold 16:9 YouTube "
           f"thumbnail (1280x720). The moment: {moment}. Keep the scene and the main subject recognisable and true to "
           f"the screenshot; crop or zoom to make the main action large and clear, raise contrast and saturation "
           f"slightly, and simplify the background. {text_rule} {common}")
    scratch = (f"Create a 16:9 YouTube thumbnail (1280x720) for a \"{style}\" gaming video of the PC game "
               f"\"{game}\". It shows this real moment from the video: {moment}. One clear focal subject, high "
               f"contrast, uncluttered background, readable on a phone. {text_rule} {common}")
    return {"reference_edit": ref, "from_scratch": scratch, "text": words, "needs_user_words": placeholder,
            "moment": moment}


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


def safe_area_preview(src: Path, out: Path) -> Path:
    """The thumbnail with the bottom-right timestamp area outlined (approximate; YouTube does not publish it)."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im, "RGBA")
        bw, bh = int(w * 0.16), int(h * 0.12)
        box = (w - bw - int(w * 0.015), h - bh - int(h * 0.02), w - int(w * 0.015), h - int(h * 0.02))
        d.rectangle(box, fill=(255, 0, 0, 70), outline=(255, 0, 0, 255), width=max(w // 400, 2))
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out)
    return out
