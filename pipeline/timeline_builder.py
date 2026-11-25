"""Long-form rough cut: renders kept source ranges with the chosen audio mix, then validates the result.

No cuts -> the video stream is copied bit-for-bit (no generation loss) and only audio is replaced.
With cuts -> one frame-accurate trim+concat encode from the ORIGINAL source (never a proxy).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import ffmpeg_tools as fx
from .audio_prep import audio_stats
from .media_probe import video_seek_shift, decode_check, probe
from .models import MediaInfo

COPYABLE_TO_MP4 = {"h264", "hevc", "av1", "mpeg4", "vp9"}

QUALITY_PRESETS = {
    "high": {"label": "High quality (slow)", "x264": ["-preset", "slow", "-crf", "18"],
             "nvenc": ["-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0"]},
    "balanced": {"label": "Balanced", "x264": ["-preset", "medium", "-crf", "20"],
                 "nvenc": ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "21", "-b:v", "0"]},
    "fast": {"label": "Fast draft", "x264": ["-preset", "veryfast", "-crf", "23"],
             "nvenc": ["-preset", "p3", "-tune", "hq", "-rc", "vbr", "-cq", "24", "-b:v", "0"]},
}


def choose_encoder(setting: str, nvenc_verified: bool) -> tuple[str, str]:
    """Return (encoder, reason). NVENC only after a real encode test passed on this PC."""
    if setting == "libx264":
        return "libx264", "selected in settings"
    if setting == "h264_nvenc":
        return ("h264_nvenc", "selected in settings (verified)") if nvenc_verified else \
               ("libx264", "NVENC selected but not verified on this PC; using libx264")
    return ("h264_nvenc", "auto: NVENC verified") if nvenc_verified else ("libx264", "auto: NVENC not verified")


def _pix_fmt(info: MediaInfo, encoder: str) -> str:
    src = info.video.pix_fmt if info.video else None
    if encoder == "libx264" and src in ("yuv420p10le",):
        return "yuv420p10le"  # keep 10-bit rather than truncating
    return "yuv420p"


def build_render_args(source: Path, mix_wav: Path, info: MediaInfo, ranges: list[tuple[float, float]],
                      out: Path, encoder: str, quality: str, audio_bitrate_k: int, graph_file: Path) -> list[str]:
    full = len(ranges) == 1 and ranges[0][0] <= 0.001 and abs(ranges[0][1] - info.duration_sec) <= 0.05
    v = info.video
    shift = video_seek_shift(info)  # mix audio starts at the first VIDEO frame; the file may start earlier
    a_args = ["-c:a", "aac", "-b:a", f"{audio_bitrate_k}k", "-ar", "48000", "-ac", "2"]
    if full and v and v.codec in COPYABLE_TO_MP4:
        return ["-y", "-i", str(source), *(["-itsoffset", f"{shift:.6f}"] if shift else []), "-i", str(mix_wav),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                *a_args, "-movflags", "+faststart", "-t", f"{info.duration_sec:.6f}", str(out)]
    parts = []
    n = len(ranges)
    parts.append(f"[0:v:0]split={n}" + "".join(f"[vs{i}]" for i in range(n)) if n > 1 else "[0:v:0]null[vs0]")
    parts.append(f"[1:a:0]asplit={n}" + "".join(f"[as{i}]" for i in range(n)) if n > 1 else "[1:a:0]anull[as0]")
    for i, (s, e) in enumerate(ranges):
        parts.append(f"[vs{i}]trim=start={s + shift:.6f}:end={e + shift:.6f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[as{i}]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{i}]")
    parts.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vout][aout]")
    graph_file.parent.mkdir(parents=True, exist_ok=True)
    graph_file.write_text(";\n".join(parts), encoding="utf-8")
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["high"])
    v_args = ["-c:v", encoder, *(preset["nvenc"] if encoder == "h264_nvenc" else preset["x264"]),
              "-pix_fmt", _pix_fmt(info, encoder)]
    fps_mode = ["-fps_mode", "vfr"] if (getattr(info, "vfr_measured", False)) else []
    return ["-y", "-i", str(source), "-i", str(mix_wav), "-/filter_complex", str(graph_file),
            "-map", "[vout]", "-map", "[aout]", *v_args, *fps_mode, *a_args, "-movflags", "+faststart", str(out)]


def backup_existing(out: Path) -> Path | None:
    """Never silently overwrite a previous render: move it to _previous/ with a timestamp."""
    if not out.exists():
        return None
    prev = out.parent / "_previous"
    prev.mkdir(parents=True, exist_ok=True)
    dest = prev / f"{out.stem}_{datetime.now():%Y%m%d_%H%M%S}{out.suffix}"
    shutil.move(str(out), dest)
    return dest


def render_rough_cut(source: Path, mix_wav: Path, info: MediaInfo, ranges: list[tuple[float, float]], out: Path,
                     encoder: str, quality: str, audio_bitrate_k: int = 320,
                     progress: Callable[[float], None] | None = None,
                     cancel_check: Callable[[], bool] | None = None) -> dict[str, Any]:
    if not ranges:
        raise ValueError("Nothing to render: every segment is cut.")
    total = sum(e - s for s, e in ranges)
    tmp = out.with_name(out.stem + ".partial" + out.suffix)
    graph = out.parent / f".{out.stem}.filtergraph.txt"
    used_encoder = encoder
    notes: list[str] = []
    args = build_render_args(source, mix_wav, info, ranges, tmp, encoder, quality, audio_bitrate_k, graph)
    stream_copy = "copy" in args
    try:
        fx.ffmpeg_progress(args, total, progress, cancel_check)
    except fx.FFmpegError as e:
        tmp.unlink(missing_ok=True)
        if encoder != "h264_nvenc" or stream_copy:
            raise
        notes.append(f"NVENC render failed ({str(e)[:200]}); retried with libx264.")
        used_encoder = "libx264"
        args = build_render_args(source, mix_wav, info, ranges, tmp, "libx264", quality, audio_bitrate_k, graph)
        fx.ffmpeg_progress(args, total, progress, cancel_check)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    backup = backup_existing(out)
    tmp.replace(out)
    graph.unlink(missing_ok=True)
    return {"encoder": "copy" if stream_copy else used_encoder, "quality": quality, "expected_duration": round(total, 3),
            "backup_of_previous": str(backup) if backup else None, "notes": notes}


def validate_render(out: Path, info: MediaInfo, expected_duration: float, segments: int,
                    mix_rms_db: float | None) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    r = probe(out)
    tol = max(0.3, 2.0 / (info.fps or 30) * segments)
    if abs(r.duration_sec - expected_duration) > tol:
        errors.append(f"Duration {r.duration_sec:.2f}s, expected {expected_duration:.2f}s (±{tol:.2f})")
    if not r.video:
        errors.append("Output has no video stream")
    elif info.display_resolution and r.display_resolution != info.display_resolution:
        errors.append(f"Resolution changed: {r.display_resolution} vs source {info.display_resolution}")
    if not r.audio:
        errors.append("Output has no audio stream")
    dec = decode_check(out, 3)
    if dec:
        errors.append("Start does not decode: " + "; ".join(dec[:2]))
    if r.duration_sec > 6:
        dec_end = decode_check(out, 3, start_sec=max(r.duration_sec - 4, 0))
        if dec_end:
            errors.append("End does not decode: " + "; ".join(dec_end[:2]))
    stats = audio_stats(out) if r.audio else {}
    if r.audio and mix_rms_db is not None and mix_rms_db > -70 and (stats.get("rms_db") or -120) < -70:
        errors.append("Output audio is silent although the mix is not")
    return errors, {"probe": r.model_dump(), "audio_stats": stats, "tolerance_sec": tol}


def write_timeline_json(path: Path, source_stored: str, info: MediaInfo, decisions: dict[str, Any],
                        mix_stored: str, mix_settings: dict[str, Any], tracks: list[dict[str, Any]]) -> None:
    rec = 0.0
    clips = []
    for s, e in decisions["kept_ranges"]:
        clips.append({"source_in": s, "source_out": e, "record_in": round(rec, 3), "record_out": round(rec + e - s, 3)})
        rec += e - s
    data = {"version": 1, "source": source_stored, "fps": info.fps, "vfr": info.vfr_measured,
            "resolution": info.display_resolution, "clips": clips, "audio": {"mix": mix_stored, "settings": mix_settings},
            "track_mapping": [{"audio_index": t["audio_index"], "role": t.get("user_role") or t["detected_role"]}
                              for t in tracks]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
