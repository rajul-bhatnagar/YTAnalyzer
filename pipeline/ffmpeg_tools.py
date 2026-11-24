"""FFmpeg / FFprobe adapter. Argument lists only; never a shell string."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

# On Windows, prevent a console window from flashing for each child process.
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class FFmpegNotFound(RuntimeError):
    pass


class FFmpegError(RuntimeError):
    def __init__(self, message: str, returncode: int | None = None, stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class Cancelled(RuntimeError):
    """Raised when a running external process was stopped because cancellation was requested."""


@dataclass(frozen=True)
class FFmpegBinaries:
    ffmpeg: Path
    ffprobe: Path


def _registry_path_dirs() -> list[Path]:
    """PATH as stored in the registry (user + machine). Covers shells opened before FFmpeg was installed."""
    if os.name != "nt":
        return []
    import winreg  # noqa: PLC0415

    dirs: list[Path] = []
    for hive, sub in ((winreg.HKEY_CURRENT_USER, r"Environment"),
                      (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
        try:
            with winreg.OpenKey(hive, sub) as key:
                value, _ = winreg.QueryValueEx(key, "Path")
        except OSError:
            continue
        for part in str(value).split(";"):
            part = os.path.expandvars(part.strip().strip('"'))
            if part:
                dirs.append(Path(part))
    return dirs


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        # winget "command line alias" folder (not created on every system).
        dirs.append(Path(local) / "Microsoft" / "WinGet" / "Links")
    dirs.extend(_registry_path_dirs())
    return dirs


def locate_binaries() -> FFmpegBinaries:
    """Find ffmpeg/ffprobe: FFMPEG_BIN_DIR, then PATH, then winget aliases, then the registry PATH."""
    exe = ".exe" if os.name == "nt" else ""
    env_dir = os.environ.get("FFMPEG_BIN_DIR", "").strip()
    if env_dir:
        f, p = Path(env_dir) / f"ffmpeg{exe}", Path(env_dir) / f"ffprobe{exe}"
        if f.exists() and p.exists():
            return FFmpegBinaries(f, p)
    f_path, p_path = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if f_path and p_path:
        return FFmpegBinaries(Path(f_path), Path(p_path))
    for d in _candidate_dirs():
        f, p = d / f"ffmpeg{exe}", d / f"ffprobe{exe}"
        if f.exists() and p.exists():
            return FFmpegBinaries(f, p)
    raise FFmpegNotFound(
        "FFmpeg/FFprobe not found. Install FFmpeg (e.g. 'winget install Gyan.FFmpeg'), open a new terminal, "
        "or set FFMPEG_BIN_DIR in .env to the folder containing ffmpeg.exe and ffprobe.exe."
    )


_cached: FFmpegBinaries | None = None
_cache_lock = threading.Lock()


def binaries() -> FFmpegBinaries:
    global _cached
    with _cache_lock:
        if _cached is None:
            _cached = locate_binaries()
        return _cached


def reset_cache() -> None:
    global _cached
    with _cache_lock:
        _cached = None


def run(
    args: Sequence[str | os.PathLike],
    timeout: float | None = None,
    check: bool = True,
    cancel_check: Callable[[], bool] | None = None,
) -> subprocess.CompletedProcess:
    """Run a command (args[0] is an executable path). Optionally poll cancel_check every 0.5 s."""
    cmd = [str(a) for a in args]
    if cancel_check is None:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout, creationflags=_CREATE_NO_WINDOW,
            )
        except subprocess.TimeoutExpired as e:
            raise FFmpegError(f"Timed out after {timeout}s: {Path(cmd[0]).name}") from e
    else:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", creationflags=_CREATE_NO_WINDOW,
        )
        out_chunks: list[str] = []
        err_chunks: list[str] = []
        t_out = threading.Thread(target=lambda: out_chunks.append(p.stdout.read()), daemon=True)  # type: ignore[union-attr]
        t_err = threading.Thread(target=lambda: err_chunks.append(p.stderr.read()), daemon=True)  # type: ignore[union-attr]
        t_out.start()
        t_err.start()
        waited = 0.0
        while True:
            try:
                p.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                waited += 0.5
                if cancel_check():
                    p.kill()
                    p.wait()
                    raise Cancelled("Cancelled by user")
                if timeout is not None and waited >= timeout:
                    p.kill()
                    p.wait()
                    raise FFmpegError(f"Timed out after {timeout}s: {Path(cmd[0]).name}")
        t_out.join()
        t_err.join()
        proc = subprocess.CompletedProcess(cmd, p.returncode, "".join(out_chunks), "".join(err_chunks))
    if check and proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise FFmpegError(f"{Path(cmd[0]).name} failed (exit {proc.returncode}): {tail}", proc.returncode, proc.stderr)
    return proc


def ffprobe_json(path: Path, extra: Sequence[str] = (), timeout: float = 120) -> dict:
    b = binaries()
    args = [b.ffprobe, "-v", "error", "-print_format", "json", *extra, "-i", str(path)]
    proc = run(args, timeout=timeout)
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as e:
        raise FFmpegError(f"ffprobe returned invalid JSON for {Path(path).name}") from e


def ffmpeg(args: Sequence[str | os.PathLike], timeout: float | None = None,
           cancel_check: Callable[[], bool] | None = None) -> subprocess.CompletedProcess:
    b = binaries()
    return run([b.ffmpeg, "-hide_banner", "-nostdin", *args], timeout=timeout, cancel_check=cancel_check)


def ffmpeg_progress(args: Sequence[str | os.PathLike], total_sec: float,
                    progress: Callable[[float], None] | None = None,
                    cancel_check: Callable[[], bool] | None = None,
                    timeout: float | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run ffmpeg with `-progress pipe:1`, reporting 0..1 from out_time and honouring cancellation.
    ``cwd`` lets filters reference files by a bare relative name (e.g. ``ass=captions.ass``), which avoids
    escaping Windows drive-letter colons inside filter arguments."""
    b = binaries()
    cmd = [str(b.ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-progress", "pipe:1", *[str(a) for a in args]]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                         errors="replace", creationflags=_CREATE_NO_WINDOW, cwd=str(cwd) if cwd else None)
    err_chunks: list[str] = []
    t_err = threading.Thread(target=lambda: err_chunks.append(p.stderr.read()), daemon=True)  # type: ignore[union-attr]
    t_err.start()
    killed_reason: str | None = None
    started = time.monotonic()
    assert p.stdout is not None
    for line in p.stdout:
        if cancel_check and cancel_check():
            killed_reason = "cancel"
            p.kill()
            break
        if timeout is not None and time.monotonic() - started > timeout:
            killed_reason = "timeout"
            p.kill()
            break
        key, _, val = line.strip().partition("=")
        if key in ("out_time_us", "out_time_ms") and progress and total_sec > 0:
            try:
                progress(min(max(int(val) / 1_000_000 / total_sec, 0.0), 1.0))
            except ValueError:
                pass
    p.wait()
    t_err.join()
    stderr = "".join(err_chunks)
    if killed_reason == "cancel":
        raise Cancelled("Cancelled by user")
    if killed_reason == "timeout":
        raise FFmpegError(f"ffmpeg timed out after {timeout}s")
    if p.returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-15:])
        raise FFmpegError(f"ffmpeg failed (exit {p.returncode}): {tail}", p.returncode, stderr)
    return subprocess.CompletedProcess(cmd, p.returncode, "", stderr)


def version_line() -> str:
    b = binaries()
    out = run([b.ffmpeg, "-hide_banner", "-version"], timeout=30).stdout
    return out.splitlines()[0] if out else "unknown"


def list_filters() -> set[str]:
    b = binaries()
    out = run([b.ffmpeg, "-hide_banner", "-filters"], timeout=30).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        # Format: " TSC name  A->A  description"
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    return names


def list_encoders() -> set[str]:
    b = binaries()
    out = run([b.ffmpeg, "-hide_banner", "-encoders"], timeout=30).stdout
    names = set()
    started = False
    for line in out.splitlines():
        if line.strip().startswith("------"):
            started = True
            continue
        parts = line.split()
        if started and len(parts) >= 2:
            names.add(parts[1])
    return names


REQUIRED_FILTERS = (
    "sidechaincompress", "amix", "loudnorm", "silencedetect", "ass", "subtitles", "gblur",
    "scale", "crop", "overlay", "vstack", "highpass", "afftdn", "alimiter", "afade",
    "astats", "volumedetect", "asplit", "volume", "adelay", "aresample",
)
