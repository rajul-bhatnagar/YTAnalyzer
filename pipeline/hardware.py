"""Hardware capability detection. Every capability is proven by a real test, not inferred.

nvidia-smi success != working CUDA inference. NVENC listed by FFmpeg != NVENC usable.
"""

from __future__ import annotations

import json
import os
import site
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import ffmpeg_tools as fx
from .models import HardwareMode

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


@dataclass
class GpuInfo:
    name: str
    driver_version: str
    memory_total_mb: int | None
    memory_free_mb: int | None


@dataclass
class HardwareReport:
    created_at: float = field(default_factory=time.time)
    python_version: str = sys.version.split()[0]
    platform: str = sys.platform
    nvidia_smi_found: bool = False
    gpus: list[GpuInfo] = field(default_factory=list)
    ffmpeg_ok: bool = False
    ffmpeg_version: str | None = None
    nvenc_listed: bool = False
    nvenc_encode_ok: bool = False
    nvenc_error: str | None = None
    ct2_version: str | None = None
    ct2_cuda_devices: int = 0
    ct2_error: str | None = None
    onnx_providers: list[str] = field(default_factory=list)
    onnx_smoke_ok: bool = False
    onnx_error: str | None = None
    whisper_cuda_smoke: str = "not_run"  # not_run | passed | failed
    whisper_cuda_error: str | None = None
    nvidia_dll_dirs: list[str] = field(default_factory=list)
    mode: str = HardwareMode.CPU.value
    notes: list[str] = field(default_factory=list)

    @property
    def cuda_verified(self) -> bool:
        return self.whisper_cuda_smoke == "passed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HardwareReport":
        d = dict(d)
        d["gpus"] = [GpuInfo(**g) for g in d.get("gpus", [])]
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def _short(err: BaseException | str, limit: int = 400) -> str:
    s = str(err).strip().replace("\r", "")
    return s[-limit:]


def find_nvidia_smi() -> Path | None:
    """nvidia-smi on PATH, else where NVIDIA drivers install it on Windows (System32 for current drivers, the old
    NVSMI folder, or the driver store copy that DCH drivers keep)."""
    import shutil  # noqa: PLC0415

    found = shutil.which("nvidia-smi")
    if found:
        return Path(found)
    if os.name != "nt":
        return None
    sysroot = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    progfiles = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    for p in (sysroot / "System32" / "nvidia-smi.exe", progfiles / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"):
        if p.exists():
            return p
    store = sysroot / "System32" / "DriverStore" / "FileRepository"
    hits = sorted(store.glob("nv*.inf_amd64_*/nvidia-smi.exe"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if store.exists() else []
    return hits[0] if hits else None


_last_smi_problem: str | None = None


def query_nvidia_smi() -> list[GpuInfo] | None:
    """GPUs reported by nvidia-smi, or None. The reason for a None is kept in ``nvidia_smi_problem()``."""
    global _last_smi_problem
    exe = find_nvidia_smi()
    if exe is None:
        _last_smi_problem = ("nvidia-smi was not found (not on PATH, not in System32, NVSMI or the driver store): "
                             "the NVIDIA driver is probably not installed.")
        return None
    try:
        proc = subprocess.run(
            [str(exe), "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, creationflags=_CREATE_NO_WINDOW,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        _last_smi_problem = f"{exe} could not run: {_short(e, 200)}"
        return None
    if proc.returncode != 0:
        _last_smi_problem = f"{exe} failed (exit {proc.returncode}): {_short(proc.stdout + proc.stderr, 300)}"
        return None
    _last_smi_problem = None
    return parse_nvidia_smi(proc.stdout)


def nvidia_smi_problem() -> str | None:
    return _last_smi_problem


def windows_display_adapters() -> list[str]:
    """Display adapters as Windows sees them (name, status, driver version). Shows whether an NVIDIA GPU exists
    at all, even when its driver is missing or broken."""
    if os.name != "nt":
        return []
    cmd = ("Get-CimInstance Win32_VideoController | ForEach-Object { \"$($_.Name) | status $($_.Status) | "
           "driver $($_.DriverVersion) | error code $($_.ConfigManagerErrorCode)\" }")
    try:
        proc = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True,
                              text=True, timeout=30, creationflags=_CREATE_NO_WINDOW)
    except (subprocess.TimeoutExpired, OSError):
        return []
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def parse_nvidia_smi(text: str) -> list[GpuInfo]:
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue

        def num(x: str) -> int | None:
            try:
                return int(float(x))
            except ValueError:
                return None

        gpus.append(GpuInfo(parts[0], parts[1], num(parts[2]), num(parts[3])))
    return gpus


def nvenc_encode_test() -> tuple[bool, str | None]:
    """Encode one second of synthetic video with h264_nvenc to the null muxer."""
    try:
        fx.ffmpeg(["-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=1",
                   "-c:v", "h264_nvenc", "-f", "null", "-"], timeout=60)
        return True, None
    except (fx.FFmpegError, fx.FFmpegNotFound) as e:
        return False, _short(e)


def register_nvidia_dll_dirs() -> list[str]:
    """Make pip-installed NVIDIA runtime DLLs (nvidia-cublas-cu12, nvidia-cudnn-cu12) loadable on Windows.

    STATUS: unverified on hardware. Harmless when the packages are absent.
    """
    if os.name != "nt":
        return []
    found: list[str] = []
    roots = [Path(p) for p in site.getsitepackages()] + [Path(sys.prefix) / "Lib" / "site-packages"]
    for root in dict.fromkeys(roots):
        nv = root / "nvidia"
        if not nv.is_dir():
            continue
        for bin_dir in nv.glob("*/bin"):
            if bin_dir.is_dir() and str(bin_dir) not in found:
                try:
                    os.add_dll_directory(str(bin_dir))
                except (OSError, AttributeError):
                    continue
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                found.append(str(bin_dir))
    return found


def ct2_cuda_devices() -> tuple[str | None, int, str | None]:
    try:
        import ctranslate2  # noqa: PLC0415
        return ctranslate2.__version__, int(ctranslate2.get_cuda_device_count()), None
    except Exception as e:  # import or driver error
        return None, 0, _short(e)


def onnx_smoke() -> tuple[list[str], bool, str | None]:
    """Run the Silero VAD model bundled with faster-whisper through ONNX Runtime (no download)."""
    try:
        import numpy as np  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415
        from faster_whisper.vad import VadOptions, get_speech_timestamps  # noqa: PLC0415

        providers = list(ort.get_available_providers())
        get_speech_timestamps(np.zeros(16000, dtype=np.float32), VadOptions())
        return providers, True, None
    except Exception as e:
        return [], False, _short(e)


def whisper_cuda_smoke(model_name: str = "tiny") -> tuple[bool, str | None]:
    """Load a Whisper model on CUDA and transcribe 2 s of silence.

    Downloads the model from Hugging Face on first use, so only call after user consent.
    """
    try:
        import numpy as np  # noqa: PLC0415
        from faster_whisper import WhisperModel  # noqa: PLC0415

        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        segments, _info = model.transcribe(np.zeros(32000, dtype=np.float32), beam_size=1)
        list(segments)
        del model
        return True, None
    except Exception as e:
        return False, _short(e)


def decide_mode(r: HardwareReport) -> HardwareMode:
    if not r.nvidia_smi_found:
        return HardwareMode.CPU
    nvenc = r.nvenc_encode_ok
    cuda = r.cuda_verified
    if nvenc and cuda:
        return HardwareMode.NVIDIA_CUDA_AND_NVENC
    if nvenc:
        return HardwareMode.NVIDIA_NVENC_ONLY
    if cuda:
        return HardwareMode.NVIDIA_CUDA_ONLY
    return HardwareMode.GPU_DETECTED_BUT_UNAVAILABLE


def detect(run_whisper_smoke: bool = False, whisper_model: str = "tiny") -> HardwareReport:
    r = HardwareReport()
    gpus = query_nvidia_smi()
    r.nvidia_smi_found = gpus is not None and len(gpus) > 0
    r.gpus = gpus or []
    if not r.nvidia_smi_found:
        r.notes.append(nvidia_smi_problem() or "nvidia-smi reported no GPU.")
        adapters = windows_display_adapters()
        if adapters:
            r.notes.append("Display adapters seen by Windows: " + " ; ".join(adapters))
        if any("nvidia" in a.lower() for a in adapters):
            r.notes.append("Windows sees an NVIDIA adapter, but nvidia-smi does not work: install or repair the "
                           "NVIDIA driver (NVIDIA App or nvidia.com), restart, then run diagnostics again.")

    try:
        r.ffmpeg_version = fx.version_line()
        r.ffmpeg_ok = True
        r.nvenc_listed = "h264_nvenc" in fx.list_encoders()
    except (fx.FFmpegNotFound, fx.FFmpegError) as e:
        r.notes.append(f"FFmpeg unavailable: {_short(e, 200)}")

    if r.ffmpeg_ok and r.nvenc_listed and r.nvidia_smi_found:
        r.nvenc_encode_ok, r.nvenc_error = nvenc_encode_test()
    elif r.nvenc_listed and not r.nvidia_smi_found:
        r.nvenc_error = "Skipped: FFmpeg lists NVENC but no NVIDIA GPU was detected"

    if r.nvidia_smi_found:
        r.nvidia_dll_dirs = register_nvidia_dll_dirs()
    r.ct2_version, r.ct2_cuda_devices, r.ct2_error = ct2_cuda_devices()
    r.onnx_providers, r.onnx_smoke_ok, r.onnx_error = onnx_smoke()

    if run_whisper_smoke:
        if r.ct2_cuda_devices > 0:
            ok, err = whisper_cuda_smoke(whisper_model)
            r.whisper_cuda_smoke = "passed" if ok else "failed"
            r.whisper_cuda_error = err
            if not ok:
                r.notes.append("CUDA Whisper failed; transcription will use CPU. See whisper_cuda_error.")
        else:
            r.whisper_cuda_smoke = "failed"
            r.whisper_cuda_error = "CTranslate2 reports 0 CUDA devices"
    elif r.ct2_cuda_devices > 0:
        r.notes.append("CUDA device visible to CTranslate2 but not yet verified. Run "
                       "'scripts\\verify_installation.py --whisper-smoke' to enable CUDA transcription.")

    r.mode = decide_mode(r).value
    return r


def save_report(report: HardwareReport, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(".tmp")
    tmp.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(file)


def load_report(file: Path) -> HardwareReport | None:
    if not file.exists():
        return None
    try:
        return HardwareReport.from_dict(json.loads(file.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError, OSError):
        return None


MODE_BADGES = {
    HardwareMode.CPU.value: ("CPU", "gray"),
    HardwareMode.NVIDIA_NVENC_ONLY.value: ("NVIDIA NVENC", "green"),
    HardwareMode.NVIDIA_CUDA_AND_NVENC.value: ("NVIDIA CUDA + NVENC", "green"),
    HardwareMode.NVIDIA_CUDA_ONLY.value: ("Partial GPU (CUDA only)", "orange"),
    HardwareMode.GPU_DETECTED_BUT_UNAVAILABLE.value: ("GPU detected but unavailable", "red"),
}
