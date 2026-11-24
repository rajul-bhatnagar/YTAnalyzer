from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.database import Database, utcnow
from pipeline.ffmpeg_tools import Cancelled
from pipeline.job_manager import (JobManager, StageContext, StageFailed, StageResult, StageSpec, TransitionError,
                                  can_transition)
from pipeline.manifest import manifest_entry
from pipeline.models import JobState as S
from pipeline.models import StageStatus


@pytest.mark.parametrize("cur,new,resume,ok", [
    (S.CREATED, S.INSPECTING, None, True),
    (S.CREATED, S.ANALYZING, None, False),           # cannot skip steps
    (S.EDITING, S.HIGHLIGHT_REVIEW, None, True),     # rework allowed
    (S.FINAL_REVIEW, S.APPROVED, None, True),
    (S.EDITING, S.APPROVED, None, False),            # approval only from final review
    (S.ANALYZING, S.PAUSED, None, True),
    (S.PAUSED, S.ANALYZING, S.ANALYZING, True),
    (S.PAUSED, S.EDITING, S.ANALYZING, False),
    (S.FAILED, S.ANALYZING, S.ANALYZING, True),
    (S.FAILED, S.INSPECTING, S.ANALYZING, True),
    (S.FAILED, S.EDITING, S.ANALYZING, False),
    (S.CANCELLED, S.INSPECTING, None, False),        # terminal
    (S.INSPECTING, S.CANCELLED, None, True),
    (S.DRAFT_UPLOADED, S.CANCELLED, None, False),
    (S.APPROVED, S.PAUSED, None, False),
])
def test_transition_rules(cur, new, resume, ok):
    assert can_transition(cur, new, resume) is ok


@pytest.fixture
def jm(studio_paths) -> tuple[JobManager, str]:
    db = Database(studio_paths.db_file)
    db.migrate()
    db.execute(
        "INSERT INTO projects(id, name, folder_name, game_name, source_path, source_size, source_quick_fp, "
        "created_at, updated_at, content_style, language, status) VALUES ('p','n','f','g','s',1,'q',?,?,'Custom','auto','imported')",
        (utcnow(), utcnow()))
    return JobManager(db, studio_paths), "p"


def test_pause_resume_and_failure_roundtrip(jm):
    m, pid = jm
    j = m.create_job(pid)
    m.advance_to(j, S.ANALYZING)
    assert m.get_job(j)["state"] == "ANALYZING"
    m.transition(j, S.PAUSED)
    assert m.get_job(j)["resume_state"] == "ANALYZING"
    with pytest.raises(TransitionError):
        m.transition(j, S.EDITING)
    m.transition(j, S.ANALYZING)
    m.transition(j, S.FAILED, error="disk full")
    job = m.get_job(j)
    assert job["error_summary"] == "disk full" and job["resume_state"] == "ANALYZING"
    m.advance_to(j, S.ANALYZING)
    assert m.get_job(j)["state"] == "ANALYZING" and m.get_job(j)["resume_state"] is None


def _ctx(m: JobManager, job: str, pid: str, cancel=lambda: False) -> StageContext:
    return StageContext(job, pid, m.paths, cancel, lambda f, n: m.set_progress(job, f, n))


def _spec(m: JobManager, calls: list, setting: dict, out_name="out.txt", fail=False, optional=False) -> StageSpec:
    def execute(ctx: StageContext) -> StageResult:
        calls.append(1)
        if fail:
            raise OSError("simulated failure with api_key=SECRET123")
        f = ctx.paths.work / out_name
        f.write_text("data", encoding="utf-8")
        return StageResult(manifest=[manifest_entry(f, ctx.paths.root, "demo", {"ok": True})])
    return StageSpec("demo", 1, execute, lambda ctx: (["input-a"], setting), optional=optional)


def test_stage_reused_only_when_fingerprint_and_outputs_match(jm):
    m, pid = jm
    j = m.create_job(pid)
    calls: list = []
    r1 = m.run_stage(_spec(m, calls, {"x": 1}), _ctx(m, j, pid))
    assert not r1.reused and len(calls) == 1
    r2 = m.run_stage(_spec(m, calls, {"x": 1}), _ctx(m, j, pid))
    assert r2.reused and len(calls) == 1
    # A new job for the same project also reuses the completed work (resume after cancel).
    j2 = m.create_job(pid)
    assert m.run_stage(_spec(m, calls, {"x": 1}), _ctx(m, j2, pid)).reused
    # Settings change -> rerun.
    m.run_stage(_spec(m, calls, {"x": 2}), _ctx(m, j, pid))
    assert len(calls) == 2
    # Output deleted -> rerun even with same fingerprint.
    (m.paths.work / "out.txt").unlink()
    m.run_stage(_spec(m, calls, {"x": 2}), _ctx(m, j, pid))
    assert len(calls) == 3
    stage = m.stages(j)[0]
    assert stage["status"] == "COMPLETED" and stage["attempt_count"] == 3
    assert stage["output_manifest"][0]["path"] == "02_WORK/out.txt"


def test_failure_recorded_with_redacted_details(jm):
    m, pid = jm
    j = m.create_job(pid)
    with pytest.raises(StageFailed):
        m.run_stage(_spec(m, [], {}, fail=True), _ctx(m, j, pid))
    st = m.stages(j)[0]
    assert st["status"] == StageStatus.FAILED.value
    assert "SECRET123" not in st["error_details"]["message"]
    assert "SECRET123" not in st["error_details"]["traceback"]
    assert st["error_details"]["recovery"]


def test_optional_stage_failure_does_not_raise(jm):
    m, pid = jm
    j = m.create_job(pid)
    res = m.run_stage(_spec(m, [], {}, fail=True, optional=True), _ctx(m, j, pid))
    assert res.failed_optional and res.warnings


def test_cancel_marks_stage_cancelled(jm):
    m, pid = jm
    j = m.create_job(pid)
    with pytest.raises(Cancelled):
        m.run_stage(_spec(m, [], {}), _ctx(m, j, pid, cancel=lambda: True))
    assert m.stages(j)[0]["status"] == "CANCELLED"


def test_validation_failure_prevents_completion(jm):
    m, pid = jm
    j = m.create_job(pid)
    spec = _spec(m, [], {})
    spec.validate = lambda ctx, res: ["output too small"]
    with pytest.raises(StageFailed, match="output too small"):
        m.run_stage(spec, _ctx(m, j, pid))
    assert m.stages(j)[0]["status"] == "FAILED"


def test_recover_interrupted(jm):
    m, pid = jm
    j = m.create_job(pid)
    m.db.execute("INSERT INTO job_stages(job_id, stage_name, status) VALUES (?, 'probe', 'RUNNING')", (j,))
    assert m.recover_interrupted() == 1
    st = m.stages(j)[0]
    assert st["status"] == "FAILED" and st["error_details"]["type"] == "Interrupted"
    assert "Interrupted" in m.get_job(j)["error_summary"]
