"""Job state machine, resumable stage runner and a single background worker.

Resume rule: a stage is reused only if a COMPLETED record (in any job of the same project) has the
same input fingerprint AND every file in its manifest still exists with the recorded size.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .database import Database, from_json, to_json, utcnow
from .ffmpeg_tools import Cancelled
from .logging_setup import ctx_logger, redact
from .manifest import fingerprint, verify_entry
from .models import WORKFLOW, JobState, StageStatus
from .paths import StudioPaths

TERMINAL = {JobState.CANCELLED}


class TransitionError(ValueError):
    pass


class StageFailed(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.message = message


class StagePaused(RuntimeError):
    """Raised by a stage that must stop safely and resume later (e.g. a cloud quota is exhausted). Work done so
    far must already be saved by the stage (per-chunk caches), so a later run continues instead of restarting."""


def can_transition(cur: JobState, new: JobState, resume_state: JobState | None = None) -> bool:
    if cur in TERMINAL or cur == new:
        return False
    if new == JobState.CANCELLED:
        return cur != JobState.DRAFT_UPLOADED
    if new in (JobState.PAUSED, JobState.FAILED):
        return cur in WORKFLOW and cur not in (JobState.APPROVED, JobState.DRAFT_UPLOADED)
    if cur == JobState.PAUSED:
        return resume_state is not None and new == resume_state
    if cur == JobState.FAILED:
        # Retry the failed state, or go back to rework an earlier state.
        return resume_state is not None and new in WORKFLOW and WORKFLOW.index(new) <= WORKFLOW.index(resume_state)
    if cur in WORKFLOW and new in WORKFLOW:
        ci, ni = WORKFLOW.index(cur), WORKFLOW.index(new)
        return ni == ci + 1 or ni < ci  # advance one step, or go back for rework
    return False


@dataclass
class StageContext:
    job_id: str
    project_id: str
    paths: StudioPaths
    cancel_check: Callable[[], bool]
    progress: Callable[[float, str], None]
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    manifest: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    reused: bool = False
    failed_optional: bool = False
    error: str | None = None


@dataclass
class StageSpec:
    name: str
    version: int
    execute: Callable[[StageContext], StageResult]
    inputs: Callable[[StageContext], tuple[list[str], dict[str, Any]]]
    validate: Callable[[StageContext, StageResult], list[str]] = lambda ctx, res: []
    optional: bool = False


class JobManager:
    def __init__(self, db: Database, paths: StudioPaths):
        self.db = db
        self.paths = paths
        self.log = ctx_logger(__name__)

    # --- jobs -----------------------------------------------------------------------------
    def create_job(self, project_id: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        now = utcnow()
        self.db.execute(
            "INSERT INTO jobs(id, project_id, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, project_id, JobState.CREATED.value, now, now),
        )
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return dict(row) if row else None

    def latest_job(self, project_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM jobs WHERE project_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1", (project_id,))
        return dict(row) if row else None

    def stages(self, job_id: str) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM job_stages WHERE job_id = ? ORDER BY id", (job_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["output_manifest"] = from_json(d["output_manifest"], [])
            d["warnings"] = from_json(d["warnings"], [])
            d["error_details"] = from_json(d["error_details"])
            out.append(d)
        return out

    def transition(self, job_id: str, new: JobState, error: str | None = None) -> None:
        with self.db.transaction() as conn:
            row = conn.execute("SELECT state, resume_state FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise TransitionError(f"Job {job_id} not found")
            cur = JobState(row["state"])
            resume = JobState(row["resume_state"]) if row["resume_state"] else None
            if not can_transition(cur, new, resume):
                raise TransitionError(f"Illegal transition {cur.value} -> {new.value}")
            new_resume = row["resume_state"]
            if new in (JobState.PAUSED, JobState.FAILED):
                new_resume = cur.value
            elif cur in (JobState.PAUSED, JobState.FAILED):
                new_resume = None
            conn.execute(
                "UPDATE jobs SET state = ?, resume_state = ?, updated_at = ?, error_summary = ?, "
                "pause_requested = CASE WHEN ? = 'PAUSED' THEN 0 ELSE pause_requested END WHERE id = ?",
                (new.value, new_resume, utcnow(), error if new == JobState.FAILED else None, new.value, job_id),
            )
        self.log.info("job %s: %s -> %s", job_id, cur.value, new.value)

    def advance_to(self, job_id: str, target: JobState) -> None:
        """Move forward one step at a time until target (no-op if already there)."""
        job = self.get_job(job_id)
        if job is None:
            raise TransitionError("job not found")
        cur = JobState(job["state"])
        if cur == target:
            return
        if cur in (JobState.PAUSED, JobState.FAILED):
            self.transition(job_id, JobState(job["resume_state"]))
            cur = JobState(job["resume_state"])
            if cur == target:
                return
        if cur not in WORKFLOW or target not in WORKFLOW:
            raise TransitionError(f"Cannot advance from {cur.value} to {target.value}")
        ci, ti = WORKFLOW.index(cur), WORKFLOW.index(target)
        if ti < ci:
            self.transition(job_id, target)
            return
        for st in WORKFLOW[ci + 1: ti + 1]:
            self.transition(job_id, st)

    def request_cancel(self, job_id: str) -> None:
        self.db.execute("UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?", (utcnow(), job_id))

    def request_pause(self, job_id: str) -> None:
        self.db.execute("UPDATE jobs SET pause_requested = 1, updated_at = ? WHERE id = ?", (utcnow(), job_id))

    def clear_requests(self, job_id: str) -> None:
        self.db.execute("UPDATE jobs SET cancel_requested = 0, pause_requested = 0 WHERE id = ?", (job_id,))

    def is_cancel_requested(self, job_id: str) -> bool:
        row = self.db.query_one("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,))
        return bool(row and row["cancel_requested"])

    def is_pause_requested(self, job_id: str) -> bool:
        row = self.db.query_one("SELECT pause_requested FROM jobs WHERE id = ?", (job_id,))
        return bool(row and row["pause_requested"])

    def set_progress(self, job_id: str, frac: float, note: str = "") -> None:
        self.db.execute("UPDATE jobs SET progress = ?, progress_note = ?, updated_at = ? WHERE id = ?",
                        (max(0.0, min(1.0, frac)), note[:300], utcnow(), job_id))

    def recover_interrupted(self) -> int:
        """At process start no worker is running, so any RUNNING stage was interrupted."""
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT id, job_id FROM job_stages WHERE status = ?", (StageStatus.RUNNING.value,)).fetchall()
            for r in rows:
                conn.execute(
                    "UPDATE job_stages SET status = ?, ended_at = ?, error_details = ? WHERE id = ?",
                    (StageStatus.FAILED.value, utcnow(),
                     to_json({"type": "Interrupted", "message": "The app stopped while this stage was running. Retry to resume."}),
                     r["id"]),
                )
                conn.execute("UPDATE jobs SET error_summary = ?, updated_at = ? WHERE id = ?",
                             ("Interrupted: the app stopped during processing. Use Retry to resume.", utcnow(), r["job_id"]))
        return len(rows)

    # --- stages ---------------------------------------------------------------------------
    def _cached_stage(self, project_id: str, stage: str, fp: str) -> dict[str, Any] | None:
        rows = self.db.query(
            "SELECT s.* FROM job_stages s JOIN jobs j ON j.id = s.job_id "
            "WHERE j.project_id = ? AND s.stage_name = ? AND s.status = ? AND s.input_fingerprint = ? "
            "ORDER BY s.ended_at DESC",
            (project_id, stage, StageStatus.COMPLETED.value, fp),
        )
        for r in rows:
            manifest = from_json(r["output_manifest"], [])
            if all(verify_entry(e, self.paths.root) for e in manifest):
                return dict(r)
        return None

    def run_stage(self, spec: StageSpec, ctx: StageContext, force: bool = False) -> StageResult:
        log = ctx_logger(__name__, ctx.project_id, ctx.job_id, spec.name)
        input_ids, settings = spec.inputs(ctx)
        fp = fingerprint(spec.name, spec.version, input_ids, settings)

        if not force:
            cached = self._cached_stage(ctx.project_id, spec.name, fp)
            if cached:
                res = StageResult(
                    manifest=from_json(cached["output_manifest"], []),
                    warnings=from_json(cached["warnings"], []),
                    reused=True,
                )
                self._upsert_stage(ctx.job_id, spec, StageStatus.COMPLETED, fp, res, started=cached["started_at"],
                                   ended=cached["ended_at"], duration=cached["duration_sec"], bump_attempt=False)
                log.info("reused completed stage (fingerprint match)")
                return res

        started = utcnow()
        t0 = time.monotonic()
        self._upsert_stage(ctx.job_id, spec, StageStatus.RUNNING, fp, StageResult(), started=started, bump_attempt=True)
        log.info("stage started")
        try:
            if ctx.cancel_check():
                raise Cancelled("Cancelled before start")
            res = spec.execute(ctx)
            errors = spec.validate(ctx, res)
            if errors:
                raise StageFailed(spec.name, "Output validation failed: " + "; ".join(errors))
        except Cancelled as e:
            self._upsert_stage(ctx.job_id, spec, StageStatus.CANCELLED, fp, StageResult(), started=started,
                               ended=utcnow(), duration=time.monotonic() - t0,
                               error={"type": "Cancelled", "message": str(e)}, bump_attempt=False)
            log.warning("stage cancelled")
            raise
        except StagePaused as e:
            self._upsert_stage(ctx.job_id, spec, StageStatus.PENDING, fp, StageResult(), started=started,
                               ended=utcnow(), duration=time.monotonic() - t0,
                               error={"type": "Paused", "message": redact(str(e))[:1000],
                                      "recovery": "Resume later; completed parts are reused."}, bump_attempt=False)
            log.warning("stage paused: %s", e)
            raise
        except Exception as e:  # noqa: BLE001 - recorded, then re-raised or downgraded
            details = {
                "type": type(e).__name__,
                "message": redact(str(e))[:2000],
                "traceback": redact("".join(traceback.format_exception(e))[-4000:]),
                "recovery": "Fix the cause shown above, then press Retry. Completed stages are reused.",
            }
            self._upsert_stage(ctx.job_id, spec, StageStatus.FAILED, fp, StageResult(), started=started,
                               ended=utcnow(), duration=time.monotonic() - t0, error=details, bump_attempt=False)
            log.error("stage failed: %s", details["message"])
            if spec.optional:
                return StageResult(failed_optional=True, error=details["message"],
                                   warnings=[f"Optional stage '{spec.name}' failed: {details['message']}"])
            raise StageFailed(spec.name, details["message"]) from e

        self._upsert_stage(ctx.job_id, spec, StageStatus.COMPLETED, fp, res, started=started, ended=utcnow(),
                           duration=time.monotonic() - t0, bump_attempt=False)
        log.info("stage completed in %.1fs", time.monotonic() - t0)
        return res

    def _upsert_stage(self, job_id: str, spec: StageSpec, status: StageStatus, fp: str, res: StageResult,
                      started: str | None = None, ended: str | None = None, duration: float | None = None,
                      error: dict | None = None, bump_attempt: bool = False) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO job_stages(job_id, stage_name, status, optional, started_at, ended_at, duration_sec, "
                "input_fingerprint, output_manifest, warnings, error_details, attempt_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(job_id, stage_name) DO UPDATE SET status = excluded.status, "
                "started_at = excluded.started_at, ended_at = excluded.ended_at, duration_sec = excluded.duration_sec, "
                "input_fingerprint = excluded.input_fingerprint, output_manifest = excluded.output_manifest, "
                "warnings = excluded.warnings, error_details = excluded.error_details, "
                "attempt_count = job_stages.attempt_count + ?",
                (job_id, spec.name, status.value, int(spec.optional), started, ended, duration, fp,
                 to_json(res.manifest), to_json(res.warnings), to_json(error), 1 if bump_attempt else 0,
                 1 if bump_attempt else 0),
            )


class JobRunner:
    """One background worker thread; jobs run sequentially so heavy media work never overlaps."""

    def __init__(self) -> None:
        self._q: queue.Queue[tuple[str, Callable[[], None]]] = queue.Queue()
        self._active: set[str] = set()
        self._queued: set[str] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="studio-job-runner", daemon=True)
        self._thread.start()
        self.log = ctx_logger(__name__)

    def submit(self, job_id: str, fn: Callable[[], None]) -> bool:
        with self._lock:
            if job_id in self._active or job_id in self._queued:
                return False
            self._queued.add(job_id)
        self._q.put((job_id, fn))
        return True

    def is_busy(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._active or job_id in self._queued

    def _loop(self) -> None:
        while True:
            job_id, fn = self._q.get()
            with self._lock:
                self._queued.discard(job_id)
                self._active.add(job_id)
            try:
                fn()
            except Exception:  # noqa: BLE001 - job functions record their own failures
                self.log.exception("unhandled error in job %s", job_id)
            finally:
                with self._lock:
                    self._active.discard(job_id)
                self._q.task_done()

    def any_busy(self) -> bool:
        with self._lock:
            return bool(self._active or self._queued)

    def wait_idle(self, timeout: float = 60) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if not self._active and not self._queued:
                    return True
            time.sleep(0.05)
        return False
