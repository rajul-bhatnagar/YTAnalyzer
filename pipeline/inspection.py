"""INSPECTING stages: probe + validation, source checksum, per-track loudness (optional)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import media_probe
from .database import Database, from_json, to_json, utcnow
from .ffmpeg_tools import Cancelled
from .job_manager import StageContext, StageResult, StageSpec
from .manifest import manifest_entry, sha256_file
from .models import MediaInfo
from .validators import validate_source_media


def _project(db: Database, project_id: str) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM projects WHERE id = ?", (project_id,))
    if row is None:
        raise RuntimeError(f"Project {project_id} not found")
    return dict(row)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def build_stages(db: Database, settings: dict[str, Any]) -> list[StageSpec]:
    probe_cfg = settings.get("probe", {})
    import_cfg = settings.get("import", {})

    def src(ctx: StageContext) -> tuple[dict[str, Any], Path, Path]:
        proj = _project(db, ctx.project_id)
        source = ctx.paths.from_stored(proj["source_path"])
        if not source.exists():
            raise FileNotFoundError(f"Source recording not found: {proj['source_path']}")
        work = ctx.paths.project_work_dir(proj["folder_name"]) / "inspection"
        return proj, source, work

    # --- probe --------------------------------------------------------------------------
    def probe_inputs(ctx: StageContext) -> tuple[list[str], dict[str, Any]]:
        proj = _project(db, ctx.project_id)
        return [f"{proj['source_path']}|{proj['source_size']}|{proj['source_quick_fp']}"], {
            "vfr_sample_seconds": probe_cfg.get("vfr_sample_seconds", 30)}

    def probe_exec(ctx: StageContext) -> StageResult:
        proj, source, work = src(ctx)
        ctx.progress(0.05, "Reading streams with FFprobe")
        info = media_probe.probe(source)
        errors = validate_source_media(info, source)
        if errors:
            raise ValueError("; ".join(errors))
        if ctx.cancel_check():
            raise Cancelled("Cancelled")
        ctx.progress(0.2, "Measuring frame timing (VFR check)")
        if info.video:
            vfr, details = media_probe.measure_vfr(source, float(probe_cfg.get("vfr_sample_seconds", 30)))
            info.vfr_measured = vfr
            info.vfr_details = details
            if vfr:
                info.warnings.append("Variable frame rate detected. Timestamps will be used, not frame counts.")
        roles = media_probe.suggest_roles(info.audio)
        out = work / "media_info.json"
        _write_json(out, info.model_dump())
        _store_probe(db, ctx, proj, info, roles)
        entry = manifest_entry(out, ctx.paths.root, "probe", {"ok": True})
        return StageResult(manifest=[entry], warnings=list(info.warnings), data={"media_info": info.model_dump()})

    def probe_validate(ctx: StageContext, res: StageResult) -> list[str]:
        info = res.data.get("media_info") or {}
        errs = []
        if not info.get("video"):
            errs.append("no video stream recorded")
        if (info.get("duration_sec") or 0) <= 0:
            errs.append("duration not recorded")
        return errs

    # --- checksum -----------------------------------------------------------------------
    def checksum_inputs(ctx: StageContext) -> tuple[list[str], dict[str, Any]]:
        proj = _project(db, ctx.project_id)
        return [f"{proj['source_path']}|{proj['source_size']}|{proj['source_quick_fp']}"], {
            "full_checksum": bool(import_cfg.get("full_checksum", True))}

    def checksum_exec(ctx: StageContext) -> StageResult:
        proj, source, work = src(ctx)
        warnings: list[str] = []
        if import_cfg.get("full_checksum", True):
            ctx.progress(0.3, "Computing SHA-256 of source (read-only)")
            digest = sha256_file(source, cancel_check=ctx.cancel_check,
                                 progress=lambda f: ctx.progress(0.3 + 0.5 * f, "Computing SHA-256 of source"))
        else:
            digest = None
            warnings.append("Full checksum disabled in settings; duplicate detection uses the quick fingerprint.")
        out = work / "source_checksum.json"
        _write_json(out, {"source": proj["source_path"], "size_bytes": proj["source_size"],
                          "quick_fingerprint": proj["source_quick_fp"], "sha256": digest})
        with db.transaction() as conn:
            conn.execute("UPDATE projects SET source_sha256 = ?, updated_at = ? WHERE id = ?",
                         (digest, utcnow(), ctx.project_id))
            conn.execute("UPDATE media_assets SET checksum = ? WHERE project_id = ? AND asset_type = 'source'",
                         (digest, ctx.project_id))
        return StageResult(manifest=[manifest_entry(out, ctx.paths.root, "checksum", {"ok": True})],
                           warnings=warnings)

    # --- track levels (optional) --------------------------------------------------------
    def levels_inputs(ctx: StageContext) -> tuple[list[str], dict[str, Any]]:
        proj = _project(db, ctx.project_id)
        return [f"{proj['source_path']}|{proj['source_quick_fp']}"], {"seconds": 120}

    def levels_exec(ctx: StageContext) -> StageResult:
        proj, source, work = src(ctx)
        tracks = db.query("SELECT audio_index FROM audio_tracks WHERE project_id = ? ORDER BY audio_index",
                          (ctx.project_id,))
        levels: dict[int, float | None] = {}
        for i, t in enumerate(tracks):
            ctx.progress(0.85 + 0.1 * i / max(len(tracks), 1), f"Measuring loudness of audio track {t['audio_index'] + 1}")
            levels[t["audio_index"]] = media_probe.track_mean_volume(source, t["audio_index"], 120, ctx.cancel_check)
        with db.transaction() as conn:
            for idx, lvl in levels.items():
                conn.execute("UPDATE audio_tracks SET mean_volume_db = ? WHERE project_id = ? AND audio_index = ?",
                             (lvl, ctx.project_id, idx))
        out = work / "track_levels.json"
        _write_json(out, {"window_sec": 120, "mean_volume_db": levels})
        warnings = [f"Audio track {i + 1} is silent in the first 2 minutes" for i, v in levels.items() if v is None or v < -60]
        return StageResult(manifest=[manifest_entry(out, ctx.paths.root, "track_levels", {"ok": True})],
                           warnings=warnings)

    return [
        StageSpec("probe", 1, probe_exec, probe_inputs, probe_validate),
        StageSpec("checksum", 1, checksum_exec, checksum_inputs),
        StageSpec("track_levels", 1, levels_exec, levels_inputs, optional=True),
    ]


def _store_probe(db: Database, ctx: StageContext, proj: dict[str, Any], info: MediaInfo, roles: list) -> None:
    v = info.video
    with db.transaction() as conn:
        conn.execute(
            "UPDATE projects SET duration_sec = ?, source_metadata = ?, updated_at = ? WHERE id = ?",
            (info.duration_sec, to_json(info.model_dump()), utcnow(), ctx.project_id),
        )
        conn.execute(
            "INSERT INTO media_assets(project_id, asset_type, path, duration_sec, codec, width, height, size_bytes, "
            "metadata, created_at) VALUES (?, 'source', ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(project_id, asset_type, path) DO UPDATE SET duration_sec = excluded.duration_sec, "
            "codec = excluded.codec, width = excluded.width, height = excluded.height, "
            "size_bytes = excluded.size_bytes, metadata = excluded.metadata",
            (ctx.project_id, proj["source_path"], info.duration_sec, v.codec if v else None,
             v.width if v else None, v.height if v else None, info.size_bytes,
             to_json({"fps": info.fps, "vfr": info.vfr_measured, "pix_fmt": v.pix_fmt if v else None}), utcnow()),
        )
        existing = {r["stream_index"]: r["user_role"] for r in conn.execute(
            "SELECT stream_index, user_role FROM audio_tracks WHERE project_id = ?", (ctx.project_id,)).fetchall()}
        conn.execute("DELETE FROM audio_tracks WHERE project_id = ?", (ctx.project_id,))
        for a, role in zip(info.audio, roles):
            conn.execute(
                "INSERT INTO audio_tracks(project_id, stream_index, audio_index, codec, channels, channel_layout, "
                "sample_rate, language, title, detected_role, user_role) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ctx.project_id, a.stream_index, a.audio_index, a.codec, a.channels, a.channel_layout,
                 a.sample_rate, a.language, a.title, role.value, existing.get(a.stream_index)),
            )
        if set(existing) != {a.stream_index for a in info.audio}:
            conn.execute("UPDATE projects SET tracks_confirmed = 0 WHERE id = ?", (ctx.project_id,))


def load_media_info(db: Database, project_id: str) -> MediaInfo | None:
    row = db.query_one("SELECT source_metadata FROM projects WHERE id = ?", (project_id,))
    data = from_json(row["source_metadata"]) if row else None
    return MediaInfo.model_validate(data) if data else None
