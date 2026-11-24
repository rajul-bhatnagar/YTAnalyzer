"""Phase 6: temporary-file clean-up safety, disk-space checks, Settings clean-up flow and sidebar quick settings."""

from __future__ import annotations

import shutil

import pytest
from streamlit.testing.v1 import AppTest

from pipeline import maintenance
from pipeline.job_manager import JobRunner
from pipeline.project_service import StudioService
from tests.conftest import REPO, requires_ffmpeg
from ui.common import get_service  # only to clear the cached service between tests

pytestmark = requires_ffmpeg


@pytest.fixture
def setup(studio_paths, media, monkeypatch):
    monkeypatch.setenv("STUDIO_ROOT", str(studio_paths.root))
    get_service.clear()
    svc = StudioService(studio_paths, runner=JobRunner(), configure_logging=False)
    src = studio_paths.raw_input / "G" / "a.mp4"
    src.parent.mkdir(parents=True)
    shutil.copy(media["one"], src)
    pid = svc.create_project(src, "G", "Custom", "en")
    svc.start_inspection(pid)
    assert svc.runner.wait_idle(60)
    f = svc.files(pid)
    (f.previews_dir).mkdir(parents=True)
    (f.previews_dir / "preview_game_only_0s.wav").write_bytes(b"x" * 1000)
    (f.work / "audio" / "tracks").mkdir(parents=True)
    keep = f.work / "audio" / "tracks" / "track1_game.wav"
    keep.write_bytes(b"k" * 500)
    orphan = f.work / "audio" / "tracks" / "track2.partial.wav"
    orphan.write_bytes(b"p" * 200)
    return svc, pid, keep, orphan


def test_cleanup_only_touches_regenerable_files(setup):
    svc, pid, keep, orphan = setup
    items = svc.temp_files()
    kinds = {i.kind for i in items}
    assert "Audio mix previews" in kinds and "Unfinished file from an interrupted run" in kinds
    assert all(i.path != keep for i in items) and all("01_RAW_INPUT" not in str(i.path) for i in items)
    with pytest.raises(ValueError, match="not a temporary file"):
        svc.clean_temp([maintenance.TempItem(keep, "x", 1)])
    assert svc.clean_temp(items) == 1200
    assert keep.exists() and not orphan.exists() and svc.temp_files() == []
    assert svc.source_path(svc.get_project(pid)).exists()


def test_disk_levels(setup, monkeypatch):
    svc, pid, _keep, _orphan = setup
    monkeypatch.setattr(maintenance, "free_disk_gb", lambda p: 1.0)
    assert svc.disk_status()["level"] == "block"
    with pytest.raises(ValueError, match="GB free"):
        svc.start_longform(pid, "audio") if svc.longform_ready(pid) is None else svc.check_disk_for_job()
    monkeypatch.setattr(maintenance, "free_disk_gb", lambda p: 10.0)
    assert svc.disk_status()["level"] == "warn"  # default minimum is 20 GB
    monkeypatch.setattr(maintenance, "free_disk_gb", lambda p: 500.0)
    assert svc.disk_status() == {"free_gb": 500.0, "level": "ok", "message": ""}


def test_settings_cleanup_needs_confirmation(setup):
    svc, _pid, keep, orphan = setup
    at = AppTest.from_string(f"import sys\nsys.path.insert(0, {str(REPO)!r})\n"
                             "from ui import settings_view\nsettings_view.render()\n", default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    btn = next(b for b in at.button if b.label == "Delete temporary files")
    assert btn.disabled
    next(c for c in at.checkbox if c.label.startswith("Delete the selected")).check().run()
    next(b for b in at.button if b.label == "Delete temporary files").click().run()
    assert not at.exception and not orphan.exists() and keep.exists()


def test_sidebar_quick_settings_save(setup):
    svc, pid, _k, _o = setup
    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=120)
    at.session_state["selected_project_id"] = pid
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    side = at.sidebar
    next(t for t in side.text_input if t.label == "Game").set_value("Renamed Game")
    next(s for s in side.selectbox if s.label == "Default Shorts layout").set_value("center_crop")
    next(b for b in side.button if b.label == "Save").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert svc.get_project(pid)["game_name"] == "Renamed Game"
    assert svc.effective_settings(pid)["shorts"]["layout"] == "center_crop"
