"""Analysis frames: choose timestamps (regular grid + extra frames around evidence), extract JPEGs from the
original source with exact seeking, measure sharpness/brightness, flag near-duplicates and build labelled
contact sheets. Analysis frames are evidence only; final renders always use the original source footage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import ffmpeg_tools as fx
from .ffmpeg_tools import Cancelled


def fmt_ts(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"


def scene_changes(grid: np.ndarray, motion: np.ndarray, k: float, min_level: float) -> list[dict[str, Any]]:
    """Hard cuts / sudden full-frame changes: motion spikes far above the typical level (median + k * MAD)."""
    if len(motion) < 3:
        return []
    med = float(np.median(motion))
    mad = float(np.median(np.abs(motion - med))) or 0.5
    thr = max(med + k * mad, min_level)
    out = []
    for i in range(len(motion)):
        v = motion[i]
        if v < thr:
            continue
        if (i > 0 and motion[i - 1] > v) or (i + 1 < len(motion) and motion[i + 1] > v):
            continue  # keep local maxima only
        out.append({"kind": "scene_change", "start": round(float(grid[i]), 2), "end": round(float(grid[i]) + 0.5, 2),
                    "peak": round(float(grid[i]), 2), "strength": round(float(v / thr), 2),
                    "note": f"Sudden full-picture change (motion {v:.1f}, typical {med:.1f})."})
    return out


def motion_bursts(grid: np.ndarray, motion: np.ndarray, hop: float, min_sec: float = 1.0) -> list[dict[str, Any]]:
    """Sustained high on-screen motion (top 10% of the recording), at least ``min_sec`` long."""
    if len(motion) < 10:
        return []
    thr = float(np.percentile(motion, 90))
    if thr <= float(np.median(motion)) * 1.2:
        return []  # motion is uniform; "bursts" would be noise
    mask = motion >= thr
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    out = []
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if (b - a) * hop < min_sec:
            continue
        k = a + int(np.argmax(motion[a:b]))
        out.append({"kind": "motion_burst", "start": round(float(grid[a]), 2), "end": round(float(grid[b - 1]) + hop, 2),
                    "peak": round(float(grid[k]), 2), "strength": round(float(motion[a:b].mean() / thr), 2),
                    "note": "Sustained high on-screen motion."})
    return out


def choose_timestamps(start: float, end: float, interval: float, priority: list[float], extra_per_event: int,
                      max_frames: int, duration: float) -> list[tuple[float, str]]:
    """(time, reason) pairs inside [start, end). Evidence frames are kept first; the regular grid fills the rest."""
    last = max(duration - 0.05, 0.0)
    pri: list[tuple[float, str]] = []
    offsets = [0.0, -1.0, 1.5, -2.5, 3.0][: max(1, extra_per_event + 1)]
    for p in sorted(priority):
        for o in offsets:
            t = min(max(p + o, start), min(end, last))
            if start <= t < end or (t == last and start <= t <= end):
                pri.append((round(t, 2), "evidence"))
    grid = [(round(t, 2), "grid") for t in np.arange(start + interval / 2, min(end, last + 1e-6), interval)]
    chosen: list[tuple[float, str]] = []

    def far(t: float) -> bool:
        return all(abs(t - c) >= 0.75 for c, _ in chosen)

    for t, r in pri:
        if len(chosen) >= max_frames:
            break
        if far(t):
            chosen.append((t, r))
    remaining = max_frames - len(chosen)
    if remaining > 0 and grid:
        step = max(len(grid) / remaining, 1.0)
        for i in np.arange(0, len(grid), step):
            t, r = grid[int(i)]
            if len(chosen) >= max_frames:
                break
            if far(t):
                chosen.append((t, r))
    if not chosen and start < duration:
        chosen.append((round(min((start + end) / 2, last), 2), "grid"))
    return sorted(chosen)


@dataclass
class FrameInfo:
    t: float
    path: str  # stored relative path
    reason: str
    window: int
    sharpness: float = 0.0
    brightness: float = 0.0
    near_duplicate_of: float | None = None
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_frame(source: Path, t: float, out: Path, width: int) -> bool:
    """Exact-seek single frame (-ss before -i seeks by timestamp, then decodes to the exact frame)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.jpg")
    try:
        fx.ffmpeg(["-y", "-ss", f"{t:.3f}", "-i", str(source), "-map", "0:v:0", "-frames:v", "1",
                   "-vf", f"scale={width}:-2:flags=bicubic", "-q:v", "3", str(tmp)], timeout=120)
    except fx.FFmpegError:
        tmp.unlink(missing_ok=True)
        return False
    if not tmp.exists() or tmp.stat().st_size < 200:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True


def measure(path: Path) -> tuple[float, float, np.ndarray]:
    """(sharpness = variance of Laplacian, mean brightness 0..255, 32x18 grey thumbnail for duplicate checks)."""
    import cv2  # noqa: PLC0415

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)  # unicode-safe path
    if img is None:
        return 0.0, 0.0, np.zeros((18, 32), np.float32)
    sharp = float(cv2.Laplacian(img, cv2.CV_64F).var())
    thumb = cv2.resize(img, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
    return sharp, float(img.mean()), thumb


def sample_window(source: Path, frames_dir: Path, window_idx: int, times: list[tuple[float, str]], width: int,
                  to_stored: Callable[[Path], str], cancel_check: Callable[[], bool] | None = None) -> list[FrameInfo]:
    out: list[FrameInfo] = []
    prev: tuple[float, np.ndarray] | None = None
    for t, reason in times:
        if cancel_check and cancel_check():
            raise Cancelled("Cancelled during frame extraction")
        p = frames_dir / f"w{window_idx:04d}_{int(round(t * 1000)):09d}.jpg"
        if not p.exists() and not extract_frame(source, t, p, width):
            continue
        sharp, bright, thumb = measure(p)
        fi = FrameInfo(t=t, path=to_stored(p), reason=reason, window=window_idx, sharpness=round(sharp, 1),
                       brightness=round(bright, 1))
        if bright < 12:
            fi.issues.append("nearly black")
        if sharp < 15:
            fi.issues.append("blurry or flat")
        if prev is not None and float(np.mean(np.abs(thumb - prev[1]))) < 2.0:
            fi.near_duplicate_of = prev[0]
        prev = (t, thumb)
        out.append(fi)
    return out


def contact_sheet(images: list[tuple[float, Path]], columns: int = 4, cell_w: int = 320) -> bytes:
    """Grid of frames, each labelled with its timestamp, encoded as JPEG (for vision models with image limits)."""
    import cv2  # noqa: PLC0415

    cells = []
    for t, p in images:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h = int(img.shape[0] * cell_w / img.shape[1])
        img = cv2.resize(img, (cell_w, h), interpolation=cv2.INTER_AREA)
        label = f"{fmt_ts(t)} ({t:.1f}s)"
        cv2.rectangle(img, (0, 0), (cell_w, 22), (0, 0, 0), -1)
        cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(img)
    if not cells:
        raise ValueError("No frames for a contact sheet")
    ch = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, ch - c.shape[0], 0, 0, cv2.BORDER_CONSTANT) for c in cells]
    while len(cells) % columns:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + columns]) for i in range(0, len(cells), columns)]
    ok, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Could not encode contact sheet")
    return buf.tobytes()
