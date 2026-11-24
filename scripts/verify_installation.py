"""Verify the installation. Used by setup_windows.bat and launch_studio.bat.

Exit code 0 = all required checks passed; 1 = a required check failed.

  python scripts\\verify_installation.py            full check + hardware detection (saves data\\hardware_report.json)
  python scripts\\verify_installation.py --quick    fast pre-launch check (no hardware tests)
  python scripts\\verify_installation.py --whisper-smoke   also test CUDA Whisper (downloads 'tiny' model)
  python scripts\\verify_installation.py --check-nvidia    exit 0 if nvidia-smi finds a GPU, else 1
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED_PYTHON = (3, 12)
REQUIRED_IMPORTS = ["streamlit", "pydantic", "faster_whisper", "ctranslate2", "av", "onnxruntime", "numpy",
                    "PIL", "cv2", "soundfile", "google.genai", "httpx", "dotenv"]
OPTIONAL_IMPORTS = {"googleapiclient": "private YouTube upload", "google_auth_oauthlib": "private YouTube upload"}

results: list[tuple[str, bool, str, bool]] = []  # name, ok, detail, required


def check(name: str, ok: bool, detail: str = "", required: bool = True) -> bool:
    results.append((name, ok, detail, required))
    return ok


def nvidia_present() -> bool:
    try:
        p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return p.returncode == 0 and "GPU" in p.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--whisper-smoke", action="store_true")
    ap.add_argument("--check-nvidia", action="store_true")
    args = ap.parse_args()

    if args.check_nvidia:
        found = nvidia_present()
        print("NVIDIA GPU detected by nvidia-smi." if found else "No NVIDIA GPU detected by nvidia-smi.")
        return 0 if found else 1

    v = sys.version_info
    check("Python version", (v.major, v.minor) == REQUIRED_PYTHON,
          f"{v.major}.{v.minor}.{v.micro} (release is tested on {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} only)")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    check("Virtual environment active", in_venv, sys.prefix)

    missing = []
    for mod in REQUIRED_IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod} ({type(e).__name__}: {e})")
    check("Required Python packages", not missing, "all importable" if not missing else "; ".join(missing))
    for mod, feature in OPTIONAL_IMPORTS.items():
        try:
            importlib.import_module(mod)
            check(f"Optional: {mod}", True, feature, required=False)
        except Exception:  # noqa: BLE001
            check(f"Optional: {mod}", False, f"not installed; {feature} disabled", required=False)

    dists = {d.metadata["Name"].lower() for d in md.distributions() if d.metadata["Name"]}
    both_ort = "onnxruntime" in dists and "onnxruntime-gpu" in dists
    check("Single ONNX Runtime", not both_ort,
          "onnxruntime and onnxruntime-gpu are BOTH installed; recreate the venv" if both_ort else "ok")

    if missing:
        return report()

    from pipeline import ffmpeg_tools as fx  # noqa: PLC0415
    try:
        b = fx.binaries()
        check("FFmpeg / FFprobe", True, f"{b.ffmpeg} | {fx.version_line()}")
        filters = fx.list_filters()
        miss_f = [f for f in fx.REQUIRED_FILTERS if f not in filters]
        check("FFmpeg filters", not miss_f, "all present" if not miss_f else "missing: " + ", ".join(miss_f))
        check("FFmpeg libx264 encoder", "libx264" in fx.list_encoders(), "CPU H.264 encoder")
    except fx.FFmpegNotFound as e:
        check("FFmpeg / FFprobe", False, str(e))

    try:
        from pipeline.database import Database  # noqa: PLC0415
        from pipeline.paths import StudioPaths  # noqa: PLC0415
        paths = StudioPaths.default()
        paths.ensure()
        ver = Database(paths.db_file).migrate()
        check("Database", True, f"schema v{ver} at {paths.to_stored(paths.db_file)}")
        check(".env file", paths.env_file.exists(), "present" if paths.env_file.exists() else
              "missing (copy .env.example to .env)", required=False)
    except Exception as e:  # noqa: BLE001
        check("Database", False, str(e))

    if not args.quick:
        from pipeline import hardware  # noqa: PLC0415
        from pipeline.paths import StudioPaths  # noqa: PLC0415
        r = hardware.detect(run_whisper_smoke=args.whisper_smoke)
        hardware.save_report(r, StudioPaths.default().hardware_report_file)
        check("ONNX Runtime smoke test", r.onnx_smoke_ok, r.onnx_error or ", ".join(r.onnx_providers))
        gpu = "; ".join(f"{g.name}, driver {g.driver_version}, {g.memory_total_mb} MB" for g in r.gpus) or "none"
        check("NVIDIA GPU", True, gpu, required=False)
        check("NVENC encode test", r.nvenc_encode_ok, r.nvenc_error or "passed", required=False)
        check("CTranslate2 CUDA devices", r.ct2_cuda_devices > 0, str(r.ct2_cuda_devices) + (f" ({r.ct2_error})" if r.ct2_error else ""), required=False)
        check("CUDA Whisper test", r.whisper_cuda_smoke == "passed",
              r.whisper_cuda_smoke + (f": {r.whisper_cuda_error}" if r.whisper_cuda_error else ""), required=False)
        print(f"\nHardware mode: {r.mode}  (CPU fallback is always available)")
        for n in r.notes:
            print(f"  note: {n}")
    return report()


def report() -> int:
    print()
    width = max(len(r[0]) for r in results) + 2
    failed_required = False
    for name, ok, detail, required in results:
        tag = "OK  " if ok else ("FAIL" if required else "WARN")
        failed_required |= (not ok and required)
        print(f"[{tag}] {name.ljust(width)} {detail}")
    print("\nRESULT:", "all required checks passed" if not failed_required else "required checks FAILED")
    return 1 if failed_required else 0


if __name__ == "__main__":
    sys.exit(main())
