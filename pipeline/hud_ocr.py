"""Optional offline reading of on-screen HUD text (score, overs, timer, kill feed) from analysis frames.

Uses the optional ``rapidocr-onnxruntime`` package (CPU ONNX Runtime, already a core dependency). It is NOT installed
by default and this integration is UNVERIFIED against the real library on the dev PC: it is written to RapidOCR's
documented call, ``RapidOCR()(image) -> (result, elapse)`` with result items ``[box, text, score]``, and tested with a
stand-in engine. Absent, failing or empty OCR is a graceful no-op: it never lowers a score.

Uses: (1) a bonus activity signal when HUD text with digits changes (useful for muted recordings), weighted by the
content-style profile; (2) ground-truth on-screen text for the AI prompt and for fact-checks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .ffmpeg_tools import Cancelled

_engine_obj: Any = None
CLOCK_CHANGE_RATIO = 0.6  # a band whose digits change in more than 60% of frame pairs behaves like a clock: ignored


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401, PLC0415
        return True
    except Exception:  # noqa: BLE001 - not installed or broken
        return False


def _engine() -> Any:
    global _engine_obj
    if _engine_obj is None:
        from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415
        _engine_obj = RapidOCR()
    return _engine_obj


def read_image(img: np.ndarray) -> list[str]:
    """Text lines found in an image (BGR array). Any engine failure -> []."""
    try:
        out = _engine()(img)
    except Exception:  # noqa: BLE001 - OCR is a bonus, never a failure
        return []
    result = out[0] if isinstance(out, tuple) else out
    texts = []
    for item in result or []:
        try:
            text = str(item[1]).strip()
        except (IndexError, TypeError):
            continue
        if text:
            texts.append(text)
    return texts


def read_frames(frames: list[dict[str, Any]], bands: list[list[float]], resolve: Callable[[str], Path],
                cancel_check: Callable[[], bool] | None = None) -> list[dict[str, Any]]:
    """[{t, bands: [[lines of band 0], [lines of band 1], ...], texts: all lines}] for unique frame times."""
    import cv2  # noqa: PLC0415

    seen: set[float] = set()
    out = []
    for fr in sorted(frames, key=lambda x: x["t"]):
        if fr["t"] in seen:
            continue
        seen.add(fr["t"])
        if cancel_check and cancel_check():
            raise Cancelled("Cancelled during HUD text reading")
        img = cv2.imdecode(np.fromfile(str(resolve(fr["path"])), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        per_band = []
        for x, y, bw, bh in bands:
            crop = img[int(y * h):int(min(y + bh, 1.0) * h), int(x * w):int(min(x + bw, 1.0) * w)]
            per_band.append(read_image(crop) if crop.size else [])
        out.append({"t": fr["t"], "bands": per_band, "texts": [t for b in per_band for t in b]})
    return out


def _digits(lines: list[str]) -> frozenset[str]:
    return frozenset(tok for ln in lines for tok in re.findall(r"[\w./:-]*\d[\w./:-]*", ln.upper()))


def change_events(ocr: list[dict[str, Any]], weight: float) -> list[dict[str, Any]]:
    """Signal atoms where digit-bearing HUD text changes (a score, overs, round). Clock-like bands are ignored."""
    if weight <= 0 or len(ocr) < 2:
        return []
    n_bands = max((len(x["bands"]) for x in ocr), default=0)
    atoms = []
    for b in range(n_bands):
        seq = [(x["t"], _digits(x["bands"][b]) if b < len(x["bands"]) else frozenset()) for x in ocr]
        pairs = [(a, c) for a, c in zip(seq, seq[1:]) if a[1] and c[1]]
        changes = [(a, c) for a, c in pairs if a[1] != c[1]]
        if not changes or len(changes) > CLOCK_CHANGE_RATIO * len(pairs):
            continue
        for (t0, before), (t1, after) in changes:
            atoms.append({"kind": "hud_change", "start": round(t0, 2), "end": round(t1, 2), "peak": round(t1, 2),
                          "strength": round(min(max(weight, 0.0), 1.0), 3),
                          "note": f"On-screen text changed: {' '.join(sorted(before))[:40]} → "
                                  f"{' '.join(sorted(after))[:40]}"})
    return atoms


def lines_in(ocr: list[dict[str, Any]], start: float, end: float, limit: int = 30) -> list[dict[str, Any]]:
    """Distinct on-screen text lines inside [start, end) with the first time each was seen (for prompts)."""
    seen, out = set(), []
    for x in ocr:
        if start <= x["t"] < end:
            for t in x["texts"]:
                if t not in seen:
                    seen.add(t)
                    out.append({"t": x["t"], "text": t})
    return out[:limit]
