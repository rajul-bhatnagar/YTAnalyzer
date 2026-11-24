"""Write a shareable diagnostics report (no secrets) to data\\diagnostics_report.json and print it.

  python scripts\\diagnostics.py [--whisper-smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import config, hardware  # noqa: E402
from pipeline.database import Database  # noqa: E402
from pipeline.paths import StudioPaths, free_disk_gb  # noqa: E402
from pipeline.project_service import StudioService  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper-smoke", action="store_true",
                    help="Also test CUDA Whisper (downloads the 'tiny' model on first run)")
    args = ap.parse_args()
    paths = StudioPaths.default()
    svc = StudioService(paths, configure_logging=False)
    hw = svc.refresh_hardware(run_whisper_smoke=args.whisper_smoke)
    eff = svc.effective_settings()
    report = {
        "software": svc.software_versions(),
        "hardware": hw.to_dict(),
        "secrets_configured": config.secret_status(),  # booleans only
        "settings": {"ai": eff["ai"], "whisper": eff["whisper"], "import": eff["import"]},
        "database_schema": Database(paths.db_file).schema_version(),
        "projects": len(svc.list_projects()),
        "free_disk_gb": round(free_disk_gb(paths.root), 1),
        "studio_root_name": paths.root.name,  # folder name only, no personal path
    }
    out = paths.data_dir / "diagnostics_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    label = hardware.MODE_BADGES.get(hw.mode, (hw.mode,))[0]
    print(f"\nHardware mode: {label}\nReport written to {paths.to_stored(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
