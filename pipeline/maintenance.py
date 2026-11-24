"""Housekeeping: find and (after the user confirms) delete temporary files, and check free disk space.

Only regenerable intermediates are ever offered for deletion: previews, proxy chunks, per-Short work folders,
thumbnail candidates and orphaned ``*.partial.*`` files left by an interrupted run. Sources, extracted tracks,
mixes, analysis results, final outputs and manifests are never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import free_disk_gb, is_within

# Relative to a project's work folder (02_WORK/<project>/) or ready folder; all can be rebuilt by re-running.
_WORK_DIRS = {"audio/previews": "Audio mix previews", "analysis/clip_previews": "Highlight preview clips",
              "analysis/video_chunks": "Gemini video proxy chunks", "shorts": "Shorts work folders (previews, graphs)"}
_READY_DIRS = {"thumbnail_frames/_candidates": "Unused thumbnail candidates"}
HARD_MIN_FREE_GB = 2.0


@dataclass
class TempItem:
    path: Path
    kind: str
    size: int


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def find_temp(work_root: Path, ready_root: Path) -> list[TempItem]:
    items: list[TempItem] = []
    for root, dirs in ((work_root, _WORK_DIRS), (ready_root, _READY_DIRS)):
        if not root.exists():
            continue
        for project in (p for p in root.iterdir() if p.is_dir()):
            for rel, kind in dirs.items():
                d = project / rel
                if d.exists() and any(d.iterdir()):
                    items.append(TempItem(d, kind, _size(d)))
            for f in project.rglob("*.partial.*"):
                if f.is_file() and not any(is_within(f, i.path) for i in items):
                    items.append(TempItem(f, "Unfinished file from an interrupted run", f.stat().st_size))
    return items


def delete_temp(items: list[TempItem], work_root: Path, ready_root: Path) -> int:
    """Delete exactly the listed items (re-checked to be regenerable and inside the studio). Returns bytes freed."""
    import shutil  # noqa: PLC0415

    allowed = {i.path.resolve() for i in find_temp(work_root, ready_root)}
    freed = 0
    for i in items:
        p = i.path.resolve()
        if p not in allowed or not (is_within(p, work_root) or is_within(p, ready_root)):
            raise ValueError(f"Refusing to delete {i.path}: not a temporary file.")
        freed += _size(p)
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    return freed


def disk_status(root: Path, min_free_gb: float) -> dict[str, object]:
    free = free_disk_gb(root)
    level = "block" if free < HARD_MIN_FREE_GB else "warn" if free < min_free_gb else "ok"
    msg = {"block": f"Only {free:.1f} GB free on the studio drive. Free up space before processing "
                    f"(at least {HARD_MIN_FREE_GB:.0f} GB is required).",
           "warn": f"Low disk space: {free:.1f} GB free (your minimum is {min_free_gb:.0f} GB). Renders may fail.",
           "ok": ""}[level]
    return {"free_gb": round(free, 1), "level": level, "message": msg}
