"""Filesystem layout. The root is resolved from this file's location, never the CWD."""

from __future__ import annotations

import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class StudioPaths:
    """All well-known locations, derived from one root so tests can use a temp root."""

    root: Path

    @classmethod
    def default(cls) -> "StudioPaths":
        return cls(PROJECT_ROOT)

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def raw_input(self) -> Path:
        return self.root / "01_RAW_INPUT"

    @property
    def work(self) -> Path:
        return self.root / "02_WORK"

    @property
    def ready(self) -> Path:
        return self.root / "03_READY_FOR_REVIEW"

    @property
    def db_file(self) -> Path:
        return self.data_dir / "studio.db"

    @property
    def default_settings_file(self) -> Path:
        return self.config_dir / "default_settings.json"

    @property
    def games_history_file(self) -> Path:
        return self.config_dir / "games_history.json"

    @property
    def hardware_report_file(self) -> Path:
        return self.data_dir / "hardware_report.json"

    @property
    def env_file(self) -> Path:
        return self.root / ".env"

    def ensure(self) -> None:
        for d in (self.config_dir, self.data_dir, self.logs_dir, self.raw_input, self.work, self.ready):
            d.mkdir(parents=True, exist_ok=True)

    def project_work_dir(self, folder_name: str) -> Path:
        return self.work / folder_name

    def project_ready_dir(self, folder_name: str) -> Path:
        return self.ready / folder_name

    # --- portable path storage -------------------------------------------------
    def to_stored(self, path: Path) -> str:
        """Store paths relative to the root (POSIX separators) when inside it, else absolute."""
        p = Path(path).resolve()
        try:
            return p.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(p)

    def from_stored(self, stored: str) -> Path:
        p = Path(stored)
        return p if p.is_absolute() else (self.root / p)


_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def slugify(text: str, max_len: int = 40) -> str:
    """ASCII-safe folder name fragment. Non-Latin names fall back to 'project'."""
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_RE.sub("_", norm).strip("_").lower()
    return (slug[:max_len].rstrip("_")) or "project"


_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(name: str, max_len: int = 150) -> str:
    """Make a user-supplied name safe as a Windows file/folder name (keeps Unicode)."""
    cleaned = _INVALID_FILENAME.sub("_", name).strip().rstrip(". ")
    if not cleaned:
        cleaned = "unnamed"
    stem = cleaned.split(".")[0].upper()
    if stem in _RESERVED:
        cleaned = "_" + cleaned
    return cleaned[:max_len]


def is_within(child: Path, parent: Path) -> bool:
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def free_disk_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / (1024**3)
