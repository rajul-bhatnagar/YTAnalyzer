"""Application-service API for Phase 3 (gameplay analysis, AI providers, cloud consent, highlight review).
Mixed into StudioService; the Streamlit UI calls only these methods."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import analysis_stages, config, ffmpeg_tools, gameplay_analysis
from .ai_providers.gemini_provider import GeminiProvider
from .ai_providers.ollama_provider import OllamaProvider
from .database import from_json, utcnow
from .ffmpeg_tools import Cancelled
from .job_manager import StageContext, StageFailed, StagePaused
from .logging_setup import ctx_logger
from .models import JobState

if TYPE_CHECKING:  # pragma: no cover
    from .project_service import StudioService

PROVIDERS = {"gemini": "Gemini (cloud, free tier, opt-in)", "ollama": "Ollama (local)",
             "none": "No AI: measured signals only"}
CONSENT_MODES = {"frames": "Sampled frames + transcript + measurements",
                 "video": "Also low-resolution video chunks of the most active windows"}
HIGHLIGHT_STATUSES = ("suggested", "approved", "rejected")


class AnalysisServiceMixin:
    # --- providers ------------------------------------------------------------------------------------
    def _ollama(self: "StudioService", ai: dict[str, Any], model: str | None = None) -> OllamaProvider:
        return OllamaProvider(ai["ollama_base_url"], model if model is not None else ai["ollama_model"],
                              float(ai["request_timeout_sec"]), float(ai["temperature"]), int(ai["ollama_num_ctx"]))

    def _gemini(self: "StudioService", ai: dict[str, Any]) -> GeminiProvider:
        return GeminiProvider(config.get_secret("GEMINI_API_KEY"), ai["gemini_model"], float(ai["request_timeout_sec"]),
                              float(ai["temperature"]))

    def ollama_models(self: "StudioService", project_id: str | None = None) -> list[dict[str, Any]]:
        """Installed Ollama models (raises ProviderUnavailable if Ollama is not running)."""
        return self._ollama(self.effective_settings(project_id)["ai"], model="").list_models()

    def test_ai_provider(self: "StudioService", project_id: str | None, provider: str) -> dict[str, Any]:
        """Explicit connection/capability test. For Gemini this calls models.get only (no content generation)."""
        ai = self.effective_settings(project_id)["ai"]
        st = (self._gemini(ai) if provider == "gemini" else self._ollama(ai)).status()
        return {"ok": st.ok, "provider": st.provider, "model": st.model, "message": st.message,
                "images": st.capabilities.images, "video": st.capabilities.video, "notes": st.capabilities.notes}

    def cloud_consent(self: "StudioService", project_id: str) -> dict[str, str]:
        ai = self.effective_settings(project_id)["ai"]
        return {"mode": ai.get("cloud_consent") or "", "at": ai.get("cloud_consent_at") or ""}

    def grant_cloud_consent(self: "StudioService", project_id: str, mode: str) -> None:
        if mode not in CONSENT_MODES:
            raise ValueError("Unknown consent mode")
        self.settings.set_many({"ai.cloud_consent": mode, "ai.cloud_consent_at": utcnow()}, scope=project_id)
        self.log.info("cloud analysis consent granted for %s: %s", project_id, mode)

    def revoke_cloud_consent(self: "StudioService", project_id: str) -> None:
        self.settings.set_many({"ai.cloud_consent": "", "ai.cloud_consent_at": ""}, scope=project_id)
        self.log.info("cloud analysis consent revoked for %s", project_id)

    def cloud_disclosure(self: "StudioService", project_id: str) -> dict[str, Any]:
        """What cloud analysis would send for this project, shown before consent."""
        s = self.effective_settings(project_id)
        a, ai = s["analysis"], s["ai"]
        proj = self.get_project(project_id)
        dur = float(proj["duration_sec"] or 0)
        windows = gameplay_analysis.build_windows(dur, float(a["window_sec"])) if dur else []
        per_req = min(int(ai["max_frames_per_request"]), int(a["max_frames_per_window"]))
        return {"windows": len(windows), "frames_per_window_max": per_req, "frames_max": per_req * len(windows),
                "frame_width": int(a["frame_width"]), "video_windows_max": int(ai["gemini_video_max_windows"]),
                "video_chunk_sec": float(a["window_sec"]), "video_height": 360,
                "transcript": "the transcript text of each window (if a voice track exists)",
                "signals": "timestamps and sizes of measured audio/visual changes",
                "model": ai["gemini_model"], "max_requests": int(ai["max_requests_per_job"])}

    def build_provider_plan(self: "StudioService", project_id: str,
                            settings: dict[str, Any]) -> tuple[gameplay_analysis.ProviderPlan, dict[str, Any], list[str]]:
        """Decide which provider runs, with which input mode, and the fallback. Never uploads without consent."""
        ai = settings["ai"]
        notes: list[str] = []
        choice = ai["provider"]
        ollama_slot = None
        if choice in ("ollama", "gemini") and ai["ollama_model"]:
            op = self._ollama(ai)
            st = op.status()
            if st.ok:
                ollama_slot = gameplay_analysis.ProviderSlot(op, "images" if st.capabilities.images else "text")
            elif choice == "ollama":
                notes.append(f"Ollama unavailable: {st.message}")
        elif choice == "ollama":
            notes.append("No Ollama model selected (Settings → AI provider).")
        primary = None
        if choice == "gemini":
            consent = ai.get("cloud_consent") or ""
            if not consent:
                notes.append("Gemini was selected but cloud analysis has not been approved for this project, so "
                             "nothing was uploaded.")
            else:
                gp = self._gemini(ai)
                st = gp.status()
                if not st.ok:
                    notes.append(f"Gemini unavailable: {st.message}")
                else:
                    visual = bool(ai["gemini_visual_enabled"])
                    mode = ("video" if visual and consent == "video" and ai["gemini_mode"] == "video"
                            else "images" if visual else "text")
                    primary = gameplay_analysis.ProviderSlot(gp, mode)
        elif choice == "ollama":
            primary = ollama_slot
        if primary is None and ollama_slot is not None and choice == "gemini":
            primary = ollama_slot
            notes.append(f"Using local Ollama '{ollama_slot.provider.model}' instead.")
        fallback = ollama_slot if primary is not None and primary is not ollama_slot else None
        if primary is None and choice != "none":
            notes.append("No AI provider is available: highlights are ranked from measured signals only.")
        plan = gameplay_analysis.ProviderPlan(primary, fallback, ai["on_quota"])
        desc = {"provider": primary.provider.name if primary else "none",
                "model": primary.provider.model if primary else "", "mode": primary.mode if primary else "signals",
                "fallback": f"{fallback.provider.name}:{fallback.provider.model}:{fallback.mode}" if fallback else ""}
        return plan, desc, notes

    # --- jobs ---------------------------------------------------------------------------------------------
    def analysis_ready(self: "StudioService", project_id: str) -> str | None:
        return self.longform_ready(project_id)

    def start_analysis(self: "StudioService", project_id: str, target: str = "analysis",
                       candidate_count: int | None = None) -> str:
        reason = self.analysis_ready(project_id)
        if reason:
            raise ValueError(reason)
        if target not in analysis_stages.TARGETS:
            raise ValueError(f"Unknown operation '{target}'")
        if candidate_count is not None and not 1 <= int(candidate_count) <= 20:
            raise ValueError("Ask for between 1 and 20 candidates.")
        self.check_disk_for_job()
        job = self.jobs.latest_job(project_id)
        job_id = self.jobs.create_job(project_id) if job is None or JobState(job["state"]) == JobState.CANCELLED \
            else job["id"]
        if self.runner.is_busy(job_id):
            raise ValueError("A job is already running for this project. Wait for it, or cancel it.")
        self.jobs.clear_requests(job_id)
        self.settings.set("jobs.last_operation", target, scope=project_id)
        self.runner.submit(job_id, lambda: self._run_analysis(project_id, job_id, target, candidate_count))
        return job_id

    def _run_analysis(self: "StudioService", project_id: str, job_id: str, target: str,
                      candidate_count: int | None) -> None:
        log = ctx_logger(__name__, project_id, job_id)
        settings = self.effective_settings(project_id)
        data: dict[str, Any] = {"pause_check": lambda: self.jobs.is_pause_requested(job_id),
                                "candidate_count": candidate_count}
        ctx = StageContext(job_id=job_id, project_id=project_id, paths=self.paths,
                           cancel_check=lambda: self.jobs.is_cancel_requested(job_id),
                           progress=lambda f, note: self.jobs.set_progress(job_id, f, note), data=data)
        try:
            proj = self.get_project(project_id)
            _lc, stages = analysis_stages.build_stages(self.db, self.paths, proj, settings, target)
            if any(s.name == "ai_windows" for s in stages):
                plan, desc, notes = self.build_provider_plan(project_id, settings)
                data.update(plan=plan, plan_descriptor=desc, plan_notes=notes)
            self.jobs.advance_to(job_id, JobState.ANALYZING)
            failed_optional: set[str] = set()
            for spec in stages:
                if self.jobs.is_pause_requested(job_id):
                    self.jobs.transition(job_id, JobState.PAUSED)
                    self.jobs.set_progress(job_id, 0.0, "Paused. Resume continues from here.")
                    return
                dep = self.STAGE_DEPENDS.get(spec.name)
                if dep in failed_optional:
                    failed_optional.add(spec.name)
                    continue
                force = spec.name == "highlights" and target == "candidates"
                res = self.jobs.run_stage(spec, ctx, force=force)
                if res.failed_optional:
                    failed_optional.add(spec.name)
            self.jobs.advance_to(job_id, JobState.HIGHLIGHT_REVIEW)
            n = len(self.highlights(project_id, statuses=("suggested",)))
            note = f"{n} new highlight candidate(s) ready for review."
            if failed_optional:
                note += f" Optional step(s) failed: {', '.join(sorted(failed_optional))}."
            self.jobs.set_progress(job_id, 1.0, note)
            self.write_project_manifest(project_id, job_id, settings)
        except Cancelled:
            self.jobs.transition(job_id, JobState.CANCELLED)
            self.jobs.set_progress(job_id, 0.0, "Cancelled")
        except StagePaused as e:
            self.jobs.transition(job_id, JobState.PAUSED)
            self.jobs.set_progress(job_id, 0.0, str(e)[:290])
        except StageFailed as e:
            self.jobs.transition(job_id, JobState.FAILED, error=str(e))
            self.jobs.set_progress(job_id, 0.0, "Failed. See stage details.")
        except Exception as e:  # noqa: BLE001
            log.exception("analysis job crashed")
            try:
                self.jobs.transition(job_id, JobState.FAILED, error=f"Unexpected error: {e}")
            except Exception:  # noqa: BLE001
                pass

    # --- results ------------------------------------------------------------------------------------------
    def ai_report(self: "StudioService", project_id: str) -> dict[str, Any] | None:
        return analysis_stages.load_json(self.files(project_id).ai_report)

    def analysis_events(self: "StudioService", project_id: str) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM events WHERE project_id = ? ORDER BY start_sec", (project_id,))
        return [{**dict(r), "evidence": from_json(r["evidence"]), "confidence": from_json(r["confidence"])} for r in rows]

    def highlights(self: "StudioService", project_id: str,
                   statuses: tuple[str, ...] = HIGHLIGHT_STATUSES) -> list[dict[str, Any]]:
        marks = ",".join("?" * len(statuses))
        rows = self.db.query(f"SELECT * FROM highlights WHERE project_id = ? AND status IN ({marks}) "  # noqa: S608
                             "ORDER BY CASE status WHEN 'approved' THEN 0 WHEN 'suggested' THEN 1 ELSE 2 END, "
                             "rank, id", (project_id, *statuses))
        out = []
        for r in rows:
            d = dict(r)
            for k in ("score_breakdown", "evidence", "handles", "event_keys"):
                d[k] = from_json(d[k])
            d["eff_start"] = d["user_start_sec"] if d["user_start_sec"] is not None else d["start_sec"]
            d["eff_end"] = d["user_end_sec"] if d["user_end_sec"] is not None else d["end_sec"]
            out.append(d)
        return out

    def get_highlight(self: "StudioService", highlight_id: int) -> dict[str, Any]:
        row = self.db.query_one("SELECT project_id FROM highlights WHERE id = ?", (highlight_id,))
        if row is None:
            raise ValueError("Highlight not found")
        return next(h for h in self.highlights(row["project_id"]) if h["id"] == highlight_id)

    def update_highlight(self: "StudioService", highlight_id: int, *, start: float | None = None,
                         end: float | None = None, status: str | None = None, notes: str | None = None,
                         hook: str | None = None, title: str | None = None) -> dict[str, Any]:
        h = self.get_highlight(highlight_id)
        proj = self.get_project(h["project_id"])
        dur = float(proj["duration_sec"] or 0)
        cfg = self.effective_settings(h["project_id"])["highlights"]
        new_s = h["eff_start"] if start is None else round(float(start), 2)
        new_e = h["eff_end"] if end is None else round(float(end), 2)
        if not 0 <= new_s < new_e <= dur + 0.01:
            raise ValueError(f"Times must satisfy 0 ≤ start < end ≤ {dur:.2f} s.")
        if new_e - new_s > float(cfg["max_sec"]) * 3:
            raise ValueError(f"A highlight longer than {float(cfg['max_sec']) * 3:.0f} s is not a Short.")
        if status is not None and status not in HIGHLIGHT_STATUSES:
            raise ValueError("Unknown status")
        sets = {"updated_at": utcnow()}
        if start is not None or end is not None:
            sets |= {"user_start_sec": new_s, "user_end_sec": new_e}
        for col, val in (("status", status), ("user_notes", notes), ("hook_caption", hook), ("suggested_title", title)):
            if val is not None:
                sets[col] = val.strip() if isinstance(val, str) and col != "status" else val
        cols = ", ".join(f"{k} = ?" for k in sets)
        with self.db.transaction() as conn:
            conn.execute(f"UPDATE highlights SET {cols} WHERE id = ?", (*sets.values(), highlight_id))  # noqa: S608
            if status in ("approved", "rejected"):
                conn.execute("INSERT INTO review_decisions(project_id, asset_ref, decision, notes, created_at) "
                             "VALUES (?, ?, ?, ?, ?)", (h["project_id"], f"highlight:{highlight_id}", status,
                                                         notes or "", utcnow()))
        warnings = []
        if new_e - new_s < float(cfg["min_sec"]) or new_e - new_s > float(cfg["max_sec"]):
            warnings.append(f"Length {new_e - new_s:.1f} s is outside the recommended {cfg['min_sec']}–"
                            f"{cfg['max_sec']} s.")
        if not new_s <= h["event_sec"] <= new_e:
            warnings.append("The key moment is no longer inside the clip.")
        return {"highlight": self.get_highlight(highlight_id), "warnings": warnings}

    def reset_highlight_times(self: "StudioService", highlight_id: int) -> None:
        self.db.execute("UPDATE highlights SET user_start_sec = NULL, user_end_sec = NULL, updated_at = ? WHERE id = ?",
                        (utcnow(), highlight_id))

    def highlight_preview(self: "StudioService", highlight_id: int, with_handles: bool = True) -> Path:
        """Small 480p preview of a candidate cut from the original source (with handles for trimming)."""
        h = self.get_highlight(highlight_id)
        proj = self.get_project(h["project_id"])
        dur = float(proj["duration_sec"] or 0)
        pad = float(self.effective_settings(h["project_id"])["highlights"]["handle_sec"]) if with_handles else 0.0
        s, e = max(h["eff_start"] - pad, 0.0), min(h["eff_end"] + pad, dur)
        out = self.files(h["project_id"]).clip_previews_dir / f"h{highlight_id}_{int(s * 1000)}_{int(e * 1000)}.mp4"
        if out.exists() and out.stat().st_size > 1000:
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".partial.mp4")
        ffmpeg_tools.ffmpeg(["-y", "-ss", f"{s:.3f}", "-t", f"{e - s:.3f}", "-i", str(self.source_path(proj)),
                             "-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale=-2:480", "-c:v", "libx264",
                             "-preset", "veryfast", "-crf", "26", "-c:a", "aac", "-b:a", "128k",
                             "-movflags", "+faststart", str(tmp)], timeout=600)
        tmp.replace(out)
        return out

    def frame_path(self: "StudioService", stored: str | None) -> Path | None:
        if not stored:
            return None
        p = self.paths.from_stored(stored)
        return p if p.exists() else None

    def analysis_settings_changed(self: "StudioService", project_id: str, values: dict[str, Any]) -> None:
        """Save analysis/AI/highlight settings for a project (validated against known keys by the store)."""
        if any(k.startswith("ai.cloud_consent") for k in values):
            raise ValueError("Use grant_cloud_consent / revoke_cloud_consent for consent.")
        self.settings.set_many(values, scope=project_id)


def describe_window_statuses(report: dict[str, Any] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for w in (report or {}).get("windows", []):
        out[w["status"]] = out.get(w["status"], 0) + 1
    return out


__all__ = ["AnalysisServiceMixin", "PROVIDERS", "CONSENT_MODES", "HIGHLIGHT_STATUSES", "describe_window_statuses"]
