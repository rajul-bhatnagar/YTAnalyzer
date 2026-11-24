"""Application service layer: the only API the Streamlit UI uses."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable

from . import (__version__, analysis_stages, maintenance, audio_mix, audio_prep, config, ffmpeg_tools, hardware, importer, inspection, longform,
               media_probe, transcribe)
from .analysis_service import AnalysisServiceMixin
from .content_service import CONTENT_TARGETS, ContentServiceMixin
from .publish_service import PublishServiceMixin
from .config import SettingsStore
from .database import Database, from_json, utcnow
from .ffmpeg_tools import Cancelled
from .job_manager import JobManager, JobRunner, StageContext, StageFailed
from .logging_setup import ctx_logger, setup_logging
from .manifest import quick_fingerprint, write_manifest_file
from .models import (CONTENT_STYLES, LANGUAGES, PROJECT_STATUS_IMPORTED, PROJECT_STATUS_INSPECTED,
                     PROJECT_STATUS_READY_AUDIO, JobState, MediaInfo, TrackRole)
from .paths import StudioPaths, is_within, slugify

_VOICEOVER_VERSION_RE = re.compile(r"voiceover_v(\d+)\.wav", re.IGNORECASE)

TRACKED_PACKAGES = ("streamlit", "pydantic", "faster-whisper", "ctranslate2", "onnxruntime", "av",
                    "numpy", "opencv-python-headless", "google-genai", "httpx")


class DuplicateProject(ValueError):
    def __init__(self, project_id: str, name: str):
        super().__init__(f"This recording is already imported as project '{name}'.")
        self.project_id = project_id
        self.name = name


class StudioService(AnalysisServiceMixin, ContentServiceMixin, PublishServiceMixin):
    def __init__(self, paths: StudioPaths | None = None, runner: JobRunner | None = None,
                 configure_logging: bool = True):
        self.paths = paths or StudioPaths.default()
        self.paths.ensure()
        config.load_env(self.paths)
        self.log_file = setup_logging(self.paths.logs_dir) if configure_logging else self.paths.logs_dir / "studio.log"
        self.db = Database(self.paths.db_file)
        self.db.migrate()
        self.settings = SettingsStore(self.db, self.paths)
        self.jobs = JobManager(self.db, self.paths)
        self.runner = runner or JobRunner()
        self.recovered = self.jobs.recover_interrupted()
        self.log = ctx_logger(__name__)
        if self.recovered:
            self.log.warning("marked %d interrupted stage(s) as failed (resumable)", self.recovered)

    # --- settings / environment -----------------------------------------------------------
    def effective_settings(self, project_id: str | None = None) -> dict[str, Any]:
        return self.settings.effective(project_id)

    def hardware_report(self) -> hardware.HardwareReport | None:
        return hardware.load_report(self.paths.hardware_report_file)

    def refresh_hardware(self, run_whisper_smoke: bool = False) -> hardware.HardwareReport:
        report = hardware.detect(run_whisper_smoke=run_whisper_smoke)
        report = hardware.carry_over_cuda_test(report, hardware.load_report(self.paths.hardware_report_file))
        hardware.save_report(report, self.paths.hardware_report_file)
        return report

    def software_versions(self) -> dict[str, Any]:
        versions: dict[str, Any] = {"app": __version__, "python": platform.python_version(),
                                    "os": platform.platform()}
        for pkg in TRACKED_PACKAGES:
            try:
                versions[pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                versions[pkg] = None
        try:
            versions["ffmpeg"] = ffmpeg_tools.version_line()
        except (ffmpeg_tools.FFmpegNotFound, ffmpeg_tools.FFmpegError):
            versions["ffmpeg"] = None
        return versions

    # --- discovery / import ---------------------------------------------------------------
    def scan_raw_input(self) -> list[importer.RawFile]:
        exts = self.effective_settings()["import"]["video_extensions"]
        return importer.scan_raw_input(self.paths, exts)

    def save_upload(self, stream: BinaryIO, filename: str, game_name: str, size_bytes: int,
                    progress: Callable[[float], None] | None = None) -> Path:
        cfg = self.effective_settings()["import"]
        return importer.save_upload(self.paths, stream, filename, game_name, size_bytes,
                                    cfg["video_extensions"], int(cfg["max_upload_mb"]),
                                    float(cfg["min_free_disk_gb"]), progress)

    def game_suggestions(self) -> list[str]:
        return config.game_suggestions(self.paths)

    def find_duplicate(self, source: Path) -> dict[str, Any] | None:
        size = source.stat().st_size
        qfp = quick_fingerprint(source)
        row = self.db.query_one(
            "SELECT id, name FROM projects WHERE source_size = ? AND source_quick_fp = ?", (size, qfp))
        return dict(row) if row else None

    def create_project(self, source: Path, game_name: str, content_style: str, language: str,
                       topic_note: str = "", name: str | None = None, allow_duplicate: bool = False) -> str:
        source = Path(source)
        game_name = game_name.strip()
        if not game_name:
            raise ValueError("Enter a game name (any game is allowed).")
        if not source.is_file():
            raise FileNotFoundError(f"Recording not found: {source}")
        if content_style not in CONTENT_STYLES:
            raise ValueError(f"Unknown content style '{content_style}'")
        if language not in LANGUAGES:
            raise ValueError(f"Unknown language '{language}'")
        exts = self.effective_settings()["import"]["video_extensions"]
        if source.suffix.lower() not in exts:
            raise ValueError(f"Unsupported file type '{source.suffix}'")
        size = source.stat().st_size
        qfp = quick_fingerprint(source)
        dup = self.db.query_one("SELECT id, name FROM projects WHERE source_size = ? AND source_quick_fp = ?",
                                (size, qfp))
        if dup and not allow_duplicate:
            raise DuplicateProject(dup["id"], dup["name"])

        project_id = uuid.uuid4().hex[:12]
        display = (name or f"{game_name} - {source.stem}").strip()
        folder = f"{slugify(game_name, 24)}_{datetime.now():%Y%m%d}_{project_id[:6]}"
        now = utcnow()
        if not is_within(source, self.paths.root):
            self.log.warning("source %s is outside the studio folder; the project will not be portable", source.name)
        self.db.execute(
            "INSERT INTO projects(id, name, folder_name, game_name, source_path, source_size, source_quick_fp, "
            "created_at, updated_at, content_style, language, topic_note, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (project_id, display, folder, game_name, self.paths.to_stored(source), size, qfp, now, now,
             content_style, language, topic_note.strip()[:300], PROJECT_STATUS_IMPORTED),
        )
        self.paths.project_work_dir(folder).mkdir(parents=True, exist_ok=True)
        config.record_game(self.paths, game_name)
        self.log.info("created project %s (%s)", project_id, display)
        return project_id

    def update_project_details(self, project_id: str, game_name: str, content_style: str, language: str,
                               topic_note: str) -> None:
        if content_style not in CONTENT_STYLES or language not in LANGUAGES or not game_name.strip():
            raise ValueError("Invalid project details")
        self.db.execute(
            "UPDATE projects SET game_name = ?, content_style = ?, language = ?, topic_note = ?, updated_at = ? "
            "WHERE id = ?", (game_name.strip(), content_style, language, topic_note.strip()[:300], utcnow(), project_id))
        config.record_game(self.paths, game_name)

    def list_projects(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.query("SELECT * FROM projects ORDER BY created_at DESC, rowid DESC")]

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM projects WHERE id = ?", (project_id,))
        return dict(row) if row else None

    def source_path(self, project: dict[str, Any]) -> Path:
        return self.paths.from_stored(project["source_path"])

    def media_info(self, project_id: str) -> MediaInfo | None:
        return inspection.load_media_info(self.db, project_id)

    # --- jobs -----------------------------------------------------------------------------
    def latest_job(self, project_id: str) -> dict[str, Any] | None:
        return self.jobs.latest_job(project_id)

    def job_stages(self, job_id: str) -> list[dict[str, Any]]:
        return self.jobs.stages(job_id)

    def is_job_running(self, job_id: str) -> bool:
        return self.runner.is_busy(job_id)

    def start_inspection(self, project_id: str) -> str:
        job = self.jobs.latest_job(project_id)
        if job is None or JobState(job["state"]) == JobState.CANCELLED:
            job_id = self.jobs.create_job(project_id)
        else:
            job_id = job["id"]
        if self.runner.is_busy(job_id):
            return job_id
        self.jobs.clear_requests(job_id)
        self.settings.set("jobs.last_operation", "inspection", scope=project_id)
        self.runner.submit(job_id, lambda: self._run_inspection(project_id, job_id))
        return job_id

    def last_operation(self, project_id: str) -> str:
        """'inspection' or a long-form target: what the project's job was last asked to do."""
        return str(self.effective_settings(project_id)["jobs"].get("last_operation") or "inspection")

    def retry_job(self, project_id: str) -> str:
        """Retry / resume whatever the job was last doing. Completed stages with unchanged inputs are reused."""
        op = self.last_operation(project_id)
        if op in longform.TARGETS and self.longform_ready(project_id) is None:
            return self.start_longform(project_id, op)
        if op in analysis_stages.TARGETS and self.analysis_ready(project_id) is None:
            return self.start_analysis(project_id, op)
        if op in CONTENT_TARGETS and self.longform_ready(project_id) is None:
            return self.start_content(project_id, op)
        return self.start_inspection(project_id)

    def cancel_job(self, job_id: str) -> None:
        self.jobs.request_cancel(job_id)
        if not self.runner.is_busy(job_id):
            job = self.jobs.get_job(job_id)
            if job and JobState(job["state"]) not in (JobState.CANCELLED, JobState.DRAFT_UPLOADED):
                self.jobs.transition(job_id, JobState.CANCELLED)

    def pause_job(self, job_id: str) -> None:
        """Pause takes effect at the next safe point (between stages)."""
        self.jobs.request_pause(job_id)

    def _run_inspection(self, project_id: str, job_id: str) -> None:
        log = ctx_logger(__name__, project_id, job_id)
        settings = self.effective_settings(project_id)
        stages = inspection.build_stages(self.db, settings)
        ctx = StageContext(
            job_id=job_id, project_id=project_id, paths=self.paths,
            cancel_check=lambda: self.jobs.is_cancel_requested(job_id),
            progress=lambda f, note: self.jobs.set_progress(job_id, f, note),
        )
        try:
            self.jobs.advance_to(job_id, JobState.INSPECTING)
            warnings: list[str] = []
            for spec in stages:
                if self.jobs.is_pause_requested(job_id):
                    self.jobs.transition(job_id, JobState.PAUSED)
                    log.info("paused before stage %s", spec.name)
                    return
                res = self.jobs.run_stage(spec, ctx)
                warnings.extend(res.warnings)
            with self.db.transaction() as conn:
                conn.execute("UPDATE projects SET status = CASE WHEN tracks_confirmed = 1 THEN ? ELSE ? END, "
                             "updated_at = ? WHERE id = ?",
                             (PROJECT_STATUS_READY_AUDIO, PROJECT_STATUS_INSPECTED, utcnow(), project_id))
            self.jobs.set_progress(job_id, 1.0, "Inspection complete. Confirm audio track roles.")
            self.write_project_manifest(project_id, job_id, settings)
        except Cancelled:
            self.jobs.transition(job_id, JobState.CANCELLED)
            self.jobs.set_progress(job_id, 0.0, "Cancelled")
        except StageFailed as e:
            self.jobs.transition(job_id, JobState.FAILED, error=str(e))
            self.jobs.set_progress(job_id, 0.0, "Failed. See stage details.")
        except Exception as e:  # noqa: BLE001 - unexpected; keep the job recoverable
            log.exception("inspection crashed")
            try:
                self.jobs.transition(job_id, JobState.FAILED, error=f"Unexpected error: {e}")
            except Exception:  # noqa: BLE001
                pass

    def write_project_manifest(self, project_id: str, job_id: str, settings: dict[str, Any]) -> Path:
        proj = self.get_project(project_id)
        assert proj is not None
        stages = {s["stage_name"]: {"status": s["status"], "fingerprint": s["input_fingerprint"],
                                    "files": s["output_manifest"], "warnings": s["warnings"]}
                  for s in self.jobs.stages(job_id)}
        hw = self.hardware_report()
        data = {
            "project_id": project_id,
            "job_id": job_id,
            "game_name": proj["game_name"],
            "source": {"path": proj["source_path"], "size_bytes": proj["source_size"],
                       "sha256": proj["source_sha256"], "quick_fingerprint": proj["source_quick_fp"]},
            "settings": settings,
            "software": self.software_versions(),
            "hardware_mode": hw.mode if hw else None,
            "stages": stages,
            "written_at": utcnow(),
        }
        out = self.paths.project_work_dir(proj["folder_name"]) / "manifest.json"
        write_manifest_file(out, data)
        return out

    # --- Phase 2: long-form media engine ------------------------------------------------------
    STAGE_DEPENDS = {"subtitles": "transcribe"}

    def files(self, project_id: str) -> longform.ProjectFiles:
        proj = self.get_project(project_id)
        if proj is None:
            raise ValueError("project not found")
        return longform.ProjectFiles.of(self.paths, proj["folder_name"])

    def longform_ready(self, project_id: str) -> str | None:
        """None if Phase 2 operations can run, else the reason they cannot."""
        proj = self.get_project(project_id)
        if proj is None:
            return "Project not found."
        if not proj["source_metadata"]:
            return "Inspection has not completed yet."
        if not proj["tracks_confirmed"]:
            return "Confirm the audio track roles first (Audio track mapping page)."
        return None

    def start_longform(self, project_id: str, target: str) -> str:
        reason = self.longform_ready(project_id)
        if reason:
            raise ValueError(reason)
        if target not in longform.TARGETS:
            raise ValueError(f"Unknown operation '{target}'")
        self.check_disk_for_job()
        job = self.jobs.latest_job(project_id)
        if job is None or JobState(job["state"]) == JobState.CANCELLED:
            job_id = self.jobs.create_job(project_id)
        else:
            job_id = job["id"]
        if self.runner.is_busy(job_id):
            raise ValueError("A job is already running for this project. Wait for it, or cancel it.")
        self.jobs.clear_requests(job_id)
        self.settings.set("jobs.last_operation", target, scope=project_id)
        self.runner.submit(job_id, lambda: self._run_longform(project_id, job_id, target))
        return job_id

    def _run_longform(self, project_id: str, job_id: str, target: str) -> None:
        log = ctx_logger(__name__, project_id, job_id)
        settings = self.effective_settings(project_id)
        ctx = StageContext(
            job_id=job_id, project_id=project_id, paths=self.paths,
            cancel_check=lambda: self.jobs.is_cancel_requested(job_id),
            progress=lambda f, note: self.jobs.set_progress(job_id, f, note),
        )
        try:
            proj = self.get_project(project_id)
            _lc, stages = longform.build_stages(self.db, self.paths, proj, settings, target)
            self.jobs.advance_to(job_id, JobState.AUDIO_PREP)
            failed_optional: set[str] = set()
            for spec in stages:
                if self.jobs.is_pause_requested(job_id):
                    self.jobs.transition(job_id, JobState.PAUSED)
                    return
                dep = self.STAGE_DEPENDS.get(spec.name)
                if dep in failed_optional:
                    failed_optional.add(spec.name)
                    log.warning("skipping %s because %s failed", spec.name, dep)
                    continue
                res = self.jobs.run_stage(spec, ctx)
                if res.failed_optional:
                    failed_optional.add(spec.name)
            note = {"audio": "Audio mix ready.", "transcript": "Transcript ready.", "subtitles": "Subtitles rebuilt.",
                    "cuts": "Cut analysis ready.", "render": "Rough cut rendered and validated."}[target]
            if failed_optional:
                note += f" Optional step(s) failed: {', '.join(sorted(failed_optional))}."
            self.jobs.set_progress(job_id, 1.0, note)
            self.write_project_manifest(project_id, job_id, settings)
        except Cancelled:
            self.jobs.transition(job_id, JobState.CANCELLED)
            self.jobs.set_progress(job_id, 0.0, "Cancelled")
        except StageFailed as e:
            self.jobs.transition(job_id, JobState.FAILED, error=str(e))
            self.jobs.set_progress(job_id, 0.0, "Failed. See stage details.")
        except Exception as e:  # noqa: BLE001
            log.exception("long-form job crashed")
            try:
                self.jobs.transition(job_id, JobState.FAILED, error=f"Unexpected error: {e}")
            except Exception:  # noqa: BLE001
                pass

    def longform_context(self, project_id: str) -> longform.LongformContext:
        proj = self.get_project(project_id)
        return longform.LongformContext(self.db, self.paths, proj, self.effective_settings(project_id))

    # mix settings -------------------------------------------------------------------------
    def save_project_settings(self, project_id: str, values: dict[str, Any]) -> None:
        self.settings.set_many(values, scope=project_id)

    def mix_preview(self, project_id: str, mix: audio_mix.MixSettings, start: float, seconds: float) -> Path:
        lc = self.longform_context(project_id)
        src = audio_mix.resolve_sources(lc.track_files(), lc.voiceover_wav(), mix.voice_source)
        missing = [p for p in (src.game, src.voice) if p and not p.exists()]
        if missing:
            raise ValueError("Audio tracks are not extracted yet. Run 'Prepare audio' first.")
        out = lc.files.previews_dir / f"preview_{mix.mode}_{int(start)}s.wav"
        return audio_mix.render_preview(mix, src, start, seconds, lc.duration, out)

    # voiceover ----------------------------------------------------------------------------
    def set_voiceover(self, project_id: str, stream: BinaryIO, filename: str) -> Path:
        files = self.files(project_id)
        files.voiceover_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(files.voiceover_dir.glob("voiceover_v*_original.*"))
        version = len(existing) + 1
        upload = files.voiceover_dir / f"upload_v{version}{Path(filename).suffix.lower()}"
        with upload.open("wb") as out:
            while chunk := stream.read(4 * 1024 * 1024):
                out.write(chunk)
        try:
            res = audio_prep.import_voiceover(upload, files.voiceover_dir, version)
        finally:
            upload.unlink(missing_ok=True)
        self.settings.set("audio.voiceover_file", self.paths.to_stored(res["wav"]), scope=project_id)
        self.log.info("voiceover v%d set for %s", version, project_id)
        return res["wav"]

    def clear_voiceover(self, project_id: str) -> None:
        """Stop using the voiceover. Files are kept so earlier versions can be restored."""
        self.settings.set("audio.voiceover_file", "", scope=project_id)

    def voiceover_versions(self, project_id: str) -> list[Path]:
        """Processed voiceover WAVs (``voiceover_v<N>.wav``) in version order. The untouched uploads
        (``_original``) and Whisper copies (``_16k``) are not selectable versions."""
        d = self.files(project_id).voiceover_dir
        if not d.exists():
            return []
        found = [(int(m.group(1)), p) for p in d.glob("voiceover_v*.wav")
                 if (m := _VOICEOVER_VERSION_RE.fullmatch(p.name))]
        return [p for _, p in sorted(found)]

    def use_voiceover_version(self, project_id: str, wav: Path) -> None:
        if not wav.exists():
            raise ValueError("Voiceover file not found")
        self.settings.set("audio.voiceover_file", self.paths.to_stored(wav), scope=project_id)

    # transcript ---------------------------------------------------------------------------
    def transcript(self, project_id: str) -> tuple[dict[str, Any] | None, dict[str, str]]:
        f = self.files(project_id)
        raw = longform._read_json(f.transcript_raw)
        return raw, longform._read_json(f.transcript_edits, {})

    def save_transcript_edits(self, project_id: str, edits: dict[str, str]) -> None:
        f = self.files(project_id)
        raw = longform._read_json(f.transcript_raw, {"segments": []})
        valid = {str(s["id"]): s["text"] for s in raw["segments"]}
        clean = {k: v for k, v in edits.items() if k in valid and v.strip() != valid[k].strip()}
        longform._write_json(f.transcript_edits, clean)

    def whisper_status(self, project_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Effective model/device for the saved whisper settings, or for ``overrides`` on top of them
        (so the UI can show the consequence of a selection before it is saved)."""
        settings = self.effective_settings(project_id)
        hw = self.hardware_report()
        free = max((g.memory_free_mb or 0 for g in hw.gpus), default=None) if hw and hw.gpus else None
        choice = transcribe.choose_model({**settings["whisper"], **(overrides or {})}, bool(hw and hw.cuda_verified),
                                         free)
        return {"choice": choice, "cached": {m: transcribe.is_model_cached(m) for m in transcribe.MODEL_CHOICES},
                "free_vram_mb": free}

    # cuts ---------------------------------------------------------------------------------
    def cut_decisions(self, project_id: str) -> dict[str, Any] | None:
        return longform._read_json(self.files(project_id).cut_decisions)

    def save_cut_overrides(self, project_id: str, overrides: dict[str, bool]) -> None:
        longform._write_json(self.files(project_id).cut_overrides, overrides)

    def rendered_outputs(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM rendered_outputs WHERE project_id = ? ORDER BY id DESC", (project_id,))
        return [{**dict(r), "settings": from_json(r["settings"]), "validation_details": from_json(r["validation_details"])}
                for r in rows]

    # --- audio track mapping --------------------------------------------------------------
    def audio_tracks(self, project_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.query(
            "SELECT * FROM audio_tracks WHERE project_id = ? ORDER BY audio_index", (project_id,))]

    def save_track_roles(self, project_id: str, roles: dict[int, TrackRole]) -> None:
        tracks = self.audio_tracks(project_id)
        errors = media_probe.validate_role_mapping(roles, len(tracks))
        if errors:
            raise ValueError(" ".join(errors))
        with self.db.transaction() as conn:
            for idx, role in roles.items():
                conn.execute("UPDATE audio_tracks SET user_role = ? WHERE project_id = ? AND audio_index = ?",
                             (role.value, project_id, idx))
            conn.execute("UPDATE projects SET tracks_confirmed = 1, status = ?, updated_at = ? WHERE id = ?",
                         (PROJECT_STATUS_READY_AUDIO, utcnow(), project_id))
        self.log.info("track roles confirmed for %s: %s", project_id, {k: v.value for k, v in roles.items()})

    def track_preview(self, project_id: str, audio_index: int, start_sec: float | None = None) -> Path:
        proj = self.get_project(project_id)
        if proj is None:
            raise ValueError("project not found")
        cfg = self.effective_settings(project_id)["probe"]
        seconds = float(cfg.get("track_preview_seconds", 20))
        dur = proj["duration_sec"] or 0
        start = start_sec if start_sec is not None else max(min(dur * 0.25, dur - seconds), 0)
        out = (self.paths.project_work_dir(proj["folder_name"]) / "previews" /
               f"track{audio_index + 1}_{int(start)}s.wav")
        if out.exists() and out.stat().st_size > 1000:
            return out
        return media_probe.extract_track_preview(self.source_path(proj), audio_index, out, start, seconds)

    def open_in_file_manager(self, path: Path) -> None:
        """Open a studio folder (or a file's folder) in Windows Explorer. Only paths inside the studio root."""
        target = path if path.is_dir() else path.parent
        if not target.exists() or not is_within(target, self.paths.root):
            raise ValueError("That folder does not exist inside the studio folder.")
        if not hasattr(os, "startfile"):
            raise RuntimeError(f"Opening folders is only supported on Windows. The folder is: {target}")
        os.startfile(str(target))  # noqa: S606 - fixed local folder, validated above

    # --- housekeeping ----------------------------------------------------------------------
    def disk_status(self) -> dict[str, Any]:
        return maintenance.disk_status(self.paths.root, float(self.effective_settings()["import"]["min_free_disk_gb"]))

    def check_disk_for_job(self) -> None:
        """Refuse to start heavy processing when the drive is nearly full (a partial render would fail anyway)."""
        st = self.disk_status()
        if st["level"] == "block":
            raise ValueError(st["message"])

    def temp_files(self) -> list[maintenance.TempItem]:
        return maintenance.find_temp(self.paths.work, self.paths.ready)

    def clean_temp(self, items: list[maintenance.TempItem]) -> int:
        """Delete the listed temporary items (the UI asks for confirmation first). Not while a job runs."""
        if self.runner.any_busy():
            raise ValueError("A job is running; its temporary files may be in use. Try again when it finishes.")
        freed = maintenance.delete_temp(items, self.paths.work, self.paths.ready)
        self.log.info("cleaned %d temporary item(s), %d bytes", len(items), freed)
        return freed

    # --- logs -----------------------------------------------------------------------------
    def log_tail(self, lines: int = 300) -> str:
        from .logging_setup import tail_log
        return tail_log(self.log_file, lines)


def stage_error_text(stage: dict[str, Any]) -> str | None:
    err = stage.get("error_details")
    if isinstance(err, str):
        err = from_json(err)
    if not err:
        return None
    return f"{err.get('type', 'Error')}: {err.get('message', '')}"


__all__ = ["StudioService", "DuplicateProject", "stage_error_text"]
