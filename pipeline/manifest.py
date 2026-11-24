"""Checksums, input fingerprints and output manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .ffmpeg_tools import Cancelled

CHUNK = 4 * 1024 * 1024


def sha256_file(path: Path, cancel_check: Callable[[], bool] | None = None,
                progress: Callable[[float], None] | None = None) -> str:
    size = Path(path).stat().st_size
    h = hashlib.sha256()
    done = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            h.update(buf)
            done += len(buf)
            if cancel_check and cancel_check():
                raise Cancelled("Cancelled during checksum")
            if progress and size:
                progress(done / size)
    return h.hexdigest()


def quick_fingerprint(path: Path) -> str:
    """Fast duplicate key: size + SHA-256 of first, middle and last 4 MB."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(p, "rb") as f:
        for offset in sorted({0, max(size // 2 - CHUNK // 2, 0), max(size - CHUNK, 0)}):
            f.seek(offset)
            h.update(f.read(CHUNK))
    return h.hexdigest()


def fingerprint(stage: str, stage_version: int, inputs: list[str], settings: dict[str, Any]) -> str:
    """Deterministic fingerprint of everything that determines a stage's output."""
    payload = json.dumps(
        {"stage": stage, "v": stage_version, "inputs": sorted(inputs), "settings": settings},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_entry(path: Path, root: Path, stage: str, validation: dict[str, Any],
                   media: dict[str, Any] | None = None, checksum: bool = True) -> dict[str, Any]:
    p = Path(path)
    try:
        rel = p.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        rel = str(p.resolve())
    # Large intermediates (multi-GB WAVs) skip hashing; reuse checks rely on path + size + fingerprint.
    return {
        "path": rel,
        "size_bytes": p.stat().st_size,
        "sha256": sha256_file(p) if checksum else None,
        "media": media,
        "stage": stage,
        "validation": validation,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def verify_entry(entry: dict[str, Any], root: Path, full: bool = False) -> bool:
    """A manifest entry is still valid if the file exists with the same size (and hash when full=True)."""
    p = Path(entry["path"])
    p = p if p.is_absolute() else Path(root) / p
    if not p.exists() or p.stat().st_size != entry.get("size_bytes"):
        return False
    if entry.get("validation", {}).get("ok") is False:
        return False
    if full and entry.get("sha256"):
        return sha256_file(p) == entry["sha256"]
    return True


def write_manifest_file(file: Path, data: dict[str, Any]) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(file.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(file)
