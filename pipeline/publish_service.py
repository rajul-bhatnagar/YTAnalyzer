"""Phase 5 service API: final checklist, approval (checksummed final manifest) and optional private YouTube upload.
Mixed into StudioService."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from . import config, youtube_draft
from .database import from_json, utcnow
from .manifest import manifest_entry, verify_entry
from .media_probe import probe
from .models import JobState

if TYPE_CHECKING:  # pragma: no cover
    from .project_service import StudioService

_PLACEHOLDER = re.compile(r"\[[^\]]+\]")


class PublishServiceMixin:
    _youtube_factory: Callable[[], Any] | None = None  # tests inject a fake YouTube service

    # --- final check ----------------------------------------------------------------------------------------------
    def _final_manifest_path(self: "StudioService", pid: str) -> Path:
        return self.files(pid).ready / "final_manifest.json"

    def _approved_shorts(self: "StudioService", pid: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.query("SELECT * FROM shorts WHERE project_id = ? ORDER BY highlight_id",
                                               (pid,))]

    def final_check(self: "StudioService", pid: str) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        uploadable: list[dict[str, Any]] = []

        def add(name: str, status: str, detail: str) -> None:
            items.append({"name": name, "status": status, "detail": detail})

        row = self.db.query_one("SELECT * FROM rendered_outputs WHERE project_id = ? AND output_type = "
                                "'long_rough_cut' ORDER BY id DESC LIMIT 1", (pid,))
        if row is None:
            add("Long-form rough cut", "warn", "Not rendered (optional).")
        else:
            p = self.paths.from_stored(row["path"])
            det = from_json(row["validation_details"]) or {}
            rec = (det.get("probe") or {}).get("duration_sec")
            if row["validation_status"] != "passed" or not p.exists():
                add("Long-form rough cut", "block", "Missing or failed validation. Render it again.")
            elif rec is not None and abs(probe(p).duration_sec - rec) > 0.5:
                add("Long-form rough cut", "block", "The file changed since it was validated. Render it again.")
            else:
                add("Long-form rough cut", "ok", p.name)
                uploadable.append({"key": "rough_cut", "label": f"Long video ({p.name})", "path": p})
        shorts = self._approved_shorts(pid)
        approved = [s for s in shorts if s["status"] == "approved"]
        for s in approved:
            out = self.short_output(s["highlight_id"])
            name = f"Short #{s['highlight_id']}"
            ok = out and out["validation_status"] == "passed" and out["file"]
            if ok:
                info = probe(out["file"])
                ok = info.video is not None and (info.video.width, info.video.height) == (1080, 1920)
            if ok:
                add(name, "ok", out["file"].name)
                uploadable.append({"key": f"short_{s['highlight_id']}", "label": f"{name} ({out['file'].name})",
                                   "path": out["file"]})
            else:
                add(name, "block", "Approved, but the file is missing, changed or failed validation. Re-render it.")
        # Rendered Shorts with no decision yet (a Short rendered with default settings has no 'shorts' row).
        decided = {s["highlight_id"] for s in shorts if s["status"] in ("approved", "rejected")}
        pending = [h for h in self.highlights(pid, statuses=("approved",))
                   if h["id"] not in decided and self.short_output(h["id"]) is not None]
        if pending:
            add("Shorts not reviewed", "warn", f"{len(pending)} Short(s) are neither approved nor rejected.")
        if not uploadable:
            add("Something to publish", "block", "Approve at least one Short or render the long-form rough cut.")
        pack = self.creator_pack(pid)
        ack: list[str] = []
        if not pack:
            add("Creator package", "block", "Not generated (Metadata & scripts page).")
        else:
            bad = [f for f, v in (("Title 1", pack["titles"][0] if pack["titles"] else ""),
                                  ("Description", pack["description"])) if not v.strip() or _PLACEHOLDER.search(v)]
            if bad:
                add("Creator package", "block", f"Replace the [placeholders] / empty text in: {', '.join(bad)}.")
            else:
                add("Creator package", "ok", f"Title: {pack['titles'][0]}")
            ack = [*pack.get("number_flags", []), *pack.get("uncertainty", [])]
        if self.user_thumbnail(pid):
            add("Thumbnail", "ok", "Your imported thumbnail.")
        else:
            add("Thumbnail", "warn", "No thumbnail imported; YouTube will pick a frame.")
        return {"items": items, "blocking": any(i["status"] == "block" for i in items), "needs_ack": ack,
                "uploadable": uploadable, "approved": self.is_final_approved(pid)}

    def approve_final(self: "StudioService", pid: str, acknowledged: bool) -> Path:
        chk = self.final_check(pid)
        if chk["blocking"]:
            raise ValueError("Fix the blocking items first.")
        if chk["needs_ack"] and not acknowledged:
            raise ValueError("Confirm that you checked the flagged items in the package.")
        pack = self.creator_pack(pid)
        entries = [{"key": u["key"], **manifest_entry(u["path"], self.paths.root, "final_approval", {"ok": True})}
                   for u in chk["uploadable"]]
        data = {"approved_at": utcnow(), "outputs": entries,
                "package": {k: pack[k] for k in ("titles", "description", "tags", "opening_hook")},
                "acknowledged": chk["needs_ack"]}
        path = self._final_manifest_path(pid)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.db.execute("INSERT INTO review_decisions(project_id, asset_ref, decision, notes, created_at) "
                        "VALUES (?, 'project:final', 'approved', '', ?)", (pid, utcnow()))
        job = self.jobs.latest_job(pid)
        if job is None or JobState(job["state"]) == JobState.CANCELLED:
            job = self.jobs.get_job(self.jobs.create_job(pid))
        if JobState(job["state"]) not in (JobState.APPROVED, JobState.DRAFT_UPLOADED):
            self.jobs.advance_to(job["id"], JobState.APPROVED)
        return path

    def is_final_approved(self: "StudioService", pid: str) -> bool:
        """Approved, and no approved file has changed since (sizes/checksums in the final manifest)."""
        p = self._final_manifest_path(pid)
        if not p.exists():
            return False
        data = json.loads(p.read_text("utf-8"))
        return bool(data["outputs"]) and all(verify_entry(e, self.paths.root) for e in data["outputs"])

    # --- YouTube ------------------------------------------------------------------------------------------------
    def _yt_token(self: "StudioService") -> Path:
        return self.paths.root / "data" / "youtube_token.json"

    def youtube_status(self: "StudioService") -> dict[str, Any]:
        secrets = config.get_secret("YOUTUBE_CLIENT_SECRETS_FILE")
        return {"client_configured": bool(secrets), "connected": self._yt_token().exists() or bool(self._youtube_factory)}

    def youtube_connect(self: "StudioService") -> None:
        secrets = config.get_secret("YOUTUBE_CLIENT_SECRETS_FILE")
        if not secrets:
            raise ValueError("Set YOUTUBE_CLIENT_SECRETS_FILE in .env to your OAuth client JSON file, then restart.")
        youtube_draft.connect(Path(secrets), self._yt_token())

    def youtube_disconnect(self: "StudioService") -> None:
        self._yt_token().unlink(missing_ok=True)

    def uploads(self: "StudioService", pid: str) -> list[dict[str, Any]]:
        p = self.files(pid).ready / "uploads.json"
        return json.loads(p.read_text("utf-8")) if p.exists() else []

    def start_upload(self: "StudioService", pid: str, key: str, title: str, description: str, tags: list[str],
                     privacy: str, made_for_kids: bool | None, confirmed: bool, confirm_public: bool = False) -> str:
        if not self.is_final_approved(pid):
            raise ValueError("Approve the final package first (or approve again: a file changed since approval).")
        if made_for_kids is None:
            raise ValueError("Answer whether the video is made for kids (YouTube requires this declaration).")
        if not confirmed:
            raise ValueError("Tick the confirmation to upload.")
        if privacy != "private" and not confirm_public:
            raise ValueError(f"'{privacy}' makes the video visible to others; confirm that explicitly.")
        manifest = json.loads(self._final_manifest_path(pid).read_text("utf-8"))
        entry = next((e for e in manifest["outputs"] if e["key"] == key), None)
        if entry is None:
            raise ValueError("That file is not part of the approved outputs.")
        body = youtube_draft.video_body(title, description, tags, privacy, made_for_kids)
        video = self.paths.root / entry["path"]
        thumb = self.user_thumbnail(pid)
        thumb_path = thumb["path"] if thumb and key == "rough_cut" else None
        job = self.jobs.latest_job(pid)
        job_id = job["id"] if job else self.jobs.create_job(pid)
        if self.runner.is_busy(job_id):
            raise ValueError("A job is already running for this project.")
        self.jobs.clear_requests(job_id)
        self.runner.submit(job_id, lambda: self._run_upload(pid, job_id, key, video, body, thumb_path))
        return job_id

    def _run_upload(self: "StudioService", pid: str, job_id: str, key: str, video: Path, body: dict[str, Any],
                    thumb: Path | None) -> None:
        rec: dict[str, Any] = {"key": key, "file": video.name, "title": body["snippet"]["title"],
                               "privacy": body["status"]["privacyStatus"], "started_at": utcnow(), "warnings": []}
        try:
            service = self._youtube_factory() if self._youtube_factory else youtube_draft.build_service(
                youtube_draft.load_credentials(self._yt_token()))
            self.jobs.set_progress(job_id, 0.0, f"Uploading {video.name} as {rec['privacy']}...")
            resp = youtube_draft.upload(service, video, body,
                                        progress=lambda f: self.jobs.set_progress(job_id, f * 0.95, "Uploading"),
                                        cancel_check=lambda: self.jobs.is_cancel_requested(job_id))
            rec.update(video_id=resp["id"], url=f"https://www.youtube.com/watch?v={resp['id']}",
                       result_privacy=(resp.get("status") or {}).get("privacyStatus", rec["privacy"]), ok=True)
            if thumb:
                w = youtube_draft.set_thumbnail(service, resp["id"], thumb)
                if w:
                    rec["warnings"].append(w)
            self.jobs.advance_to(job_id, JobState.DRAFT_UPLOADED)
            self.jobs.set_progress(job_id, 1.0, f"Uploaded as a {rec['result_privacy']} video: {rec['url']}")
        except Exception as e:  # noqa: BLE001 - reported to the user; the approved state is kept
            rec.update(ok=False, error=str(e)[:500])
            self.jobs.set_progress(job_id, 0.0, f"Upload failed: {str(e)[:250]}")
        rec["finished_at"] = utcnow()
        p = self.files(pid).ready / "uploads.json"
        hist = self.uploads(pid)
        hist.append(rec)
        p.write_text(json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
        self.db.execute("INSERT INTO review_decisions(project_id, asset_ref, decision, notes, created_at) "
                        "VALUES (?, ?, ?, ?, ?)", (pid, f"upload:{key}", "uploaded" if rec["ok"] else "upload_failed",
                                                   rec.get("url") or rec.get("error", ""), utcnow()))
