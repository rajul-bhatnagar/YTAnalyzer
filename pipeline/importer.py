"""Recording discovery in 01_RAW_INPUT and safe saving of browser uploads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable

from .paths import StudioPaths, free_disk_gb, safe_filename

PARTIAL_SUFFIX = ".partial"


class ImportErrorUser(ValueError):
    """Import problem with a message suitable for the user."""


@dataclass(frozen=True)
class RawFile:
    path: Path
    rel: str
    size_bytes: int
    game_guess: str | None


def guess_game(raw_root: Path, file: Path) -> str | None:
    """Files inside a sub-folder of 01_RAW_INPUT take the first folder name as the game name."""
    try:
        parts = file.resolve().relative_to(raw_root.resolve()).parts
    except ValueError:
        return None
    return parts[0] if len(parts) > 1 else None


def scan_raw_input(paths: StudioPaths, extensions: list[str]) -> list[RawFile]:
    exts = {e.lower() for e in extensions}
    root = paths.raw_input
    if not root.exists():
        return []
    found: list[RawFile] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.name.startswith(".") or p.name.endswith(PARTIAL_SUFFIX):
            continue
        if p.suffix.lower() not in exts:
            continue
        found.append(RawFile(
            path=p,
            rel=p.relative_to(root).as_posix(),
            size_bytes=p.stat().st_size,
            game_guess=guess_game(root, p),
        ))
    found.sort(key=lambda r: r.rel.casefold())
    return found


def _unique_target(folder: Path, filename: str) -> Path:
    target = folder / filename
    stem, suffix = target.stem, target.suffix
    n = 1
    while target.exists() or target.with_name(target.name + PARTIAL_SUFFIX).exists():
        target = folder / f"{stem}_{n}{suffix}"
        n += 1
    return target


def save_upload(paths: StudioPaths, stream: BinaryIO, filename: str, game_name: str, size_bytes: int,
                extensions: list[str], max_upload_mb: int, min_free_gb: float,
                progress: Callable[[float], None] | None = None) -> Path:
    """Copy an uploaded file to 01_RAW_INPUT/<game>/ in chunks. Never overwrites an existing file."""
    name = safe_filename(Path(filename).name)
    if Path(name).suffix.lower() not in {e.lower() for e in extensions}:
        raise ImportErrorUser(f"Unsupported file type '{Path(name).suffix}'.")
    if size_bytes > max_upload_mb * 1024 * 1024:
        raise ImportErrorUser(
            f"File is {size_bytes / 1024**2:.0f} MB, above the {max_upload_mb} MB upload limit. "
            "Copy large recordings into the 01_RAW_INPUT folder instead.")
    folder = paths.raw_input / safe_filename(game_name.strip() or "Unsorted")
    folder.mkdir(parents=True, exist_ok=True)
    free_after = free_disk_gb(folder) - size_bytes / 1024**3
    if free_after < min_free_gb:
        raise ImportErrorUser(
            f"Not enough disk space: {free_after:.1f} GB would remain, minimum is {min_free_gb} GB.")
    target = _unique_target(folder, name)
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    written = 0
    try:
        with partial.open("xb") as out:
            while True:
                chunk = stream.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                if progress and size_bytes:
                    progress(min(written / size_bytes, 1.0))
        if size_bytes and written != size_bytes:
            raise ImportErrorUser(f"Upload incomplete: wrote {written} of {size_bytes} bytes.")
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return target
