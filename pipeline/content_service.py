"""Application-service API for Phase 4 (Shorts, thumbnails, creator package). Mixed into StudioService."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from . import captions, content_stages, shorts_builder, thumbnail_prep
from . import youtube_metadata as ym
from .database import from_json, utcnow
from .ffmpeg_tools import Cancelled
from .job_manager import StageContext, StageFailed, StagePaused
from .logging_setup import ctx_logger
from .models import JobState

if TYPE_CHECKING:  # pragma: no cover
    from .project_service import StudioService

CONTENT_TARGETS = {"shorts": JobState.SHORTS_RENDER, "assets": JobState.ASSET_PREP,
                   "thumbnails": JobState.ASSET_PREP, "pack": JobState.ASSET_PREP}
SHORT_STATUSES = ("pending", "approved", "rejected")
_read_json, _write_json = content_stages._read_json, content_stages._write_json


class ContentServiceMixin:
    # --- jobs -------------------------------------------------------------------------------------------------
    def start_content(self: "StudioService", project_id: str, target: str, highlight_ids: list[int] | None = None,
                      force_ids: list[int] | None = None, discard_pack_edits: bool = False) -> str:
        reason = self.longform_ready(project_id)
        if reason:
            raise ValueError(reason)
        if target not in CONTENT_TARGETS:
            raise ValueError(f"Unknown operation '{target}'")
        self.check_disk_for_job()
        if target == "shorts" and not content_stages.approved_highlights(self.db, project_id):
            raise ValueError("Approve at least one highlight first (Analysis & highlights page).")
        job = self.jobs.latest_job(project_id)
        job_id = self.jobs.create_job(project_id) if job is None or JobState(job["state"]) == JobState.CANCELLED \
            else job["id"]
        if self.runner.is_busy(job_id):
            raise ValueError("A job is already running for this project. Wait for it, or cancel it.")
        if discard_pack_edits:
            self.files(project_id).pack_edits.unlink(missing_ok=True)
        self.jobs.clear_requests(job_id)
        self.settings.set("jobs.last_operation", target, scope=project_id)
        self.runner.submit(job_id, lambda: self._run_content(project_id, job_id, target, highlight_ids, force_ids))
        return job_id

    def _run_content(self: "StudioService", project_id: str, job_id: str, target: str,
                     highlight_ids: list[int] | None, force_ids: list[int] | None) -> None:
        log = ctx_logger(__name__, project_id, job_id)
        settings = self.effective_settings(project_id)
        data: dict[str, Any] = {}
        ctx = StageContext(job_id=job_id, project_id=project_id, paths=self.paths,
                           cancel_check=lambda: self.jobs.is_cancel_requested(job_id),
                           progress=lambda f, note: self.jobs.set_progress(job_id, f, note), data=data)
        try:
            proj = self.get_project(project_id)
            _lc, stages = content_stages.build_stages(self.db, self.paths, proj, settings, target, highlight_ids)
            if any(s.name == "creator_pack" for s in stages):
                plan, desc, notes = self.build_provider_plan(project_id, settings)
                data.update(plan=plan, plan_descriptor=desc, plan_notes=notes)
            self.jobs.advance_to(job_id, CONTENT_TARGETS[target])
            force = {f"short_h{i}" for i in (force_ids or [])}
            failed: list[str] = []
            for spec in stages:
                if self.jobs.is_pause_requested(job_id):
                    self.jobs.transition(job_id, JobState.PAUSED)
                    self.jobs.set_progress(job_id, 0.0, "Paused. Resume continues from here.")
                    return
                if spec.name == "subtitles" and "transcribe" in failed:
                    continue
                res = self.jobs.run_stage(spec, ctx, force=spec.name in force)
                if res.failed_optional:
                    failed.append(spec.name)
            shorts_failed = [n for n in failed if n.startswith("short_h")]
            note = {"shorts": "Shorts rendered and validated.", "assets": "Thumbnails and creator package ready.",
                    "thumbnails": "Thumbnail frames and prompts ready.", "pack": "Creator package ready."}[target]
            if shorts_failed:
                note = f"{len(shorts_failed)} Short(s) failed; the others are ready. See the Shorts page."
            elif failed:
                note += f" Optional step(s) failed: {', '.join(failed)}."
            self.jobs.set_progress(job_id, 1.0, note)
            self.write_project_manifest(project_id, job_id, settings)
        except Cancelled:
            self.jobs.transition(job_id, JobState.CANCELLED)
            self.jobs.set_progress(job_id, 0.0, "Cancelled")
        except StagePaused as e:
            self.jobs.transition(job_id, JobState.PAUSED)
            self.jobs.set_progress(job_id, 0.0, str(e)[:290])
        except (StageFailed, ValueError) as e:
            self.jobs.transition(job_id, JobState.FAILED, error=str(e))
            self.jobs.set_progress(job_id, 0.0, "Failed. See stage details.")
        except Exception as e:  # noqa: BLE001
            log.exception("content job crashed")
            try:
                self.jobs.transition(job_id, JobState.FAILED, error=f"Unexpected error: {e}")
            except Exception:  # noqa: BLE001
                pass

    # --- Shorts -------------------------------------------------------------------------------------------------
    def _approved(self: "StudioService", highlight_id: int) -> dict[str, Any]:
        h = self.get_highlight(highlight_id)
        if h["status"] != "approved":
            raise ValueError("Only approved highlights become Shorts. Approve it on the Analysis & highlights page.")
        return h

    def short_settings(self: "StudioService", highlight_id: int) -> shorts_builder.ShortSettings:
        h = self.get_highlight(highlight_id)
        return content_stages.short_settings(self.db, highlight_id, self.effective_settings(h["project_id"]))

    def save_short_settings(self: "StudioService", highlight_id: int, s: shorts_builder.ShortSettings) -> None:
        h = self._approved(highlight_id)
        if s.layout not in shorts_builder.LAYOUTS:
            raise ValueError("Unknown layout")
        if s.captions.position not in captions.POSITIONS:
            raise ValueError("Unknown caption position")
        captions.ass_color(s.captions.text_color)  # raises on bad colours
        captions.ass_color(s.captions.emphasis_color)
        captions.ass_color(s.captions.outline_color)
        with self.db.transaction() as conn:
            conn.execute("INSERT INTO shorts(highlight_id, project_id, settings, updated_at) VALUES (?, ?, ?, ?) "
                         "ON CONFLICT(highlight_id) DO UPDATE SET settings = excluded.settings, "
                         "updated_at = excluded.updated_at",
                         (highlight_id, h["project_id"], s.model_dump_json(), utcnow()))

    def short_preview_frame(self: "StudioService", highlight_id: int, at: float,
                            s: shorts_builder.ShortSettings | None = None) -> Path:
        """Framing preview (one 1080x1920 frame, with captions) before committing to a full render."""
        h = self._approved(highlight_id)
        lc = self.longform_context(h["project_id"])
        s = s or self.short_settings(highlight_id)
        words = captions.clip_words(content_stages.words_video_time(lc.files), h["eff_start"], h["eff_end"])
        ass = None
        if s.captions.enabled or (h["hook_caption"] and s.captions.hook_enabled):
            ass, _ = captions.build_ass(words, h["eff_end"] - h["eff_start"], s.captions, h["hook_caption"] or "")
        size = lc.info.display_resolution
        if size is None:
            raise ValueError("The recording has no video stream.")
        work = lc.files.short_work(highlight_id) / "preview"
        out = work / f"frame_{s.layout}_{int(at * 1000)}.png"
        return shorts_builder.preview_frame(lc.source, h["eff_start"], h["eff_end"], at, s, size, work, ass, out)

    def short_output(self: "StudioService", highlight_id: int) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM rendered_outputs WHERE output_type = 'short' AND highlight_id = ? "
                                "ORDER BY id DESC LIMIT 1", (highlight_id,))
        if row is None:
            return None
        d = dict(row)
        d["settings"], d["validation_details"] = from_json(d["settings"]), from_json(d["validation_details"])
        p = self.paths.from_stored(d["path"])
        d["file"] = p if p.exists() else None
        return d

    def short_review(self: "StudioService", highlight_id: int) -> dict[str, Any]:
        row = self.db.query_one("SELECT status, notes FROM shorts WHERE highlight_id = ?", (highlight_id,))
        return dict(row) if row else {"status": "pending", "notes": ""}

    def set_short_status(self: "StudioService", highlight_id: int, status: str, notes: str = "") -> None:
        if status not in SHORT_STATUSES:
            raise ValueError("Unknown status")
        h = self._approved(highlight_id)
        out = self.short_output(highlight_id)
        if status == "approved" and (out is None or out["validation_status"] != "passed" or out["file"] is None):
            raise ValueError("Render this Short and make sure it passes validation before approving it.")
        if self.db.query_one("SELECT 1 FROM shorts WHERE highlight_id = ?", (highlight_id,)) is None:
            self.save_short_settings(highlight_id, self.short_settings(highlight_id))
        with self.db.transaction() as conn:
            conn.execute("UPDATE shorts SET status = ?, notes = ?, updated_at = ? WHERE highlight_id = ?",
                         (status, notes.strip(), utcnow(), highlight_id))
            conn.execute("INSERT INTO review_decisions(project_id, asset_ref, decision, notes, created_at) "
                         "VALUES (?, ?, ?, ?, ?)", (h["project_id"], f"short:{highlight_id}", status, notes, utcnow()))

    # --- thumbnails -----------------------------------------------------------------------------------------------
    def thumbnails(self: "StudioService", project_id: str) -> dict[str, Any] | None:
        data = _read_json(self.files(project_id).thumbs_json)
        if not data:
            return None
        for fr in data["frames"]:
            p = self.paths.from_stored(fr["file"])
            fr["path"] = p if p.exists() else None
            cut = self.files(project_id).cutouts_dir / (Path(fr["file"]).stem + "_cutout.png")
            fr["cutout"] = cut if cut.exists() else None
        return data

    def cutouts_available(self: "StudioService") -> bool:
        return thumbnail_prep.rembg_available()

    def make_cutout(self: "StudioService", project_id: str, frame_file: Path) -> Path:
        if not thumbnail_prep.rembg_available():
            raise ValueError("Background cut-outs need the optional 'rembg' package, which is not installed.")
        if not frame_file.exists() or frame_file.parent != self.files(project_id).thumbs_dir:
            raise ValueError("Unknown thumbnail frame.")
        out = self.files(project_id).cutouts_dir / (frame_file.stem + "_cutout.png")
        return thumbnail_prep.make_cutout(frame_file, out)

    def import_user_thumbnail(self: "StudioService", project_id: str, stream: BinaryIO,
                              filename: str) -> dict[str, Any]:
        ext = Path(filename).suffix.lower()
        if ext not in (".png", ".jpg", ".jpeg"):
            raise ValueError("Use a PNG or JPG image.")
        d = self.files(project_id).user_thumbnail_dir
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"upload{ext}"
        with tmp.open("wb") as out:
            shutil.copyfileobj(stream, out, 4 * 1024 * 1024)
        errors, warnings, info = thumbnail_prep.validate_user_thumbnail(tmp)
        if errors:
            tmp.unlink(missing_ok=True)
            raise ValueError(" ".join(errors))
        for old in d.glob("thumbnail_user.*"):
            prev = d / "_previous"
            prev.mkdir(exist_ok=True)
            old.replace(prev / f"{old.stem}_{utcnow().replace(':', '')}{old.suffix}")
        final = d / f"thumbnail_user{'.jpg' if ext == '.jpeg' else ext}"
        tmp.replace(final)
        preview = thumbnail_prep.safe_area_preview(final, d / "_preview_safe_area.png")
        return {"path": final, "preview": preview, "warnings": warnings, "info": info}

    def user_thumbnail(self: "StudioService", project_id: str) -> dict[str, Any] | None:
        d = self.files(project_id).user_thumbnail_dir
        found = sorted(d.glob("thumbnail_user.*")) if d.exists() else []
        if not found:
            return None
        _e, warnings, info = thumbnail_prep.validate_user_thumbnail(found[0])
        prev = d / "_preview_safe_area.png"
        return {"path": found[0], "preview": prev if prev.exists() else None, "warnings": warnings, "info": info}

    # --- creator package ----------------------------------------------------------------------------------------
    def creator_pack(self: "StudioService", project_id: str) -> dict[str, Any] | None:
        f = self.files(project_id)
        draft = _read_json(f.pack_draft)
        return ym.apply_edits(draft, _read_json(f.pack_edits, {})) if draft else None

    def save_pack_field(self: "StudioService", project_id: str, field: str, value: Any) -> dict[str, Any]:
        """Save one edited field. Never triggers AI or video work."""
        if field not in ym.EDITABLE_FIELDS:
            raise ValueError(f"'{field}' cannot be edited")
        value = _clean_field(field, value)
        f = self.files(project_id)
        if _read_json(f.pack_draft) is None:
            raise ValueError("Generate the creator package first.")
        edits = _read_json(f.pack_edits, {})
        edits[field] = value
        _write_json(f.pack_edits, edits)
        return content_stages.write_effective_pack(f) or {}

    def discard_pack_edits(self: "StudioService", project_id: str, fields: list[str] | None = None) -> None:
        f = self.files(project_id)
        edits = _read_json(f.pack_edits, {})
        for k in list(edits) if fields is None else fields:
            edits.pop(k, None)
        _write_json(f.pack_edits, edits)
        content_stages.write_effective_pack(f)


def _clean_field(field: str, value: Any) -> Any:
    if field in ("opening_hook", "description", "pinned_comment"):
        text = str(value).strip()
        if field == "description" and len(text) > ym.DESCRIPTION_MAX:
            raise ValueError(f"The description is longer than {ym.DESCRIPTION_MAX} characters (YouTube's limit as "
                             "known to the app).")
        return text
    if field == "titles":
        titles = [str(t).strip() for t in value if str(t).strip()]
        if not titles:
            raise ValueError("Keep at least one title.")
        long = [t for t in titles if len(t) > ym.TITLE_MAX]
        if long:
            raise ValueError(f"Titles must be at most {ym.TITLE_MAX} characters.")
        return titles
    if field == "tags":
        tags = [str(t).strip().lstrip("#").strip() for t in value if str(t).strip()]
        if sum(len(t) for t in tags) + max(len(tags) - 1, 0) > ym.TAGS_TOTAL_MAX:
            raise ValueError(f"Tags are longer than about {ym.TAGS_TOTAL_MAX} characters in total.")
        return tags
    if field == "script":
        lines = []
        for ln in value:
            t = float(ln["time_sec"])
            if t < 0:
                raise ValueError("Script times cannot be negative.")
            lines.append({"time_sec": round(t, 2), "text": str(ln["text"]).strip(),
                          "event_key": str(ln.get("event_key") or ""), "uncertain": bool(ln.get("uncertain")),
                          "video_time_sec": ln.get("video_time_sec")})
        return sorted(lines, key=lambda x: x["time_sec"])
    if field == "shorts":
        return [{**s, "title": str(s["title"]).strip()[:100], "hook": str(s["hook"]).strip()[:80]} for s in value]
    raise ValueError(f"'{field}' cannot be edited")


__all__ = ["ContentServiceMixin", "CONTENT_TARGETS", "SHORT_STATUSES"]
