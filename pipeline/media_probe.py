"""Media inspection with FFprobe: streams, VFR detection, track-role suggestions, previews."""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any, Callable

from . import ffmpeg_tools as fx
from .models import AudioStreamInfo, MediaInfo, TrackRole, VideoStreamInfo


def _rate(value: str | None) -> float | None:
    if not value or value in ("0/0", "0"):
        return None
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            n, d = float(num), float(den)
            return None if d == 0 or n == 0 else n / d
        v = float(value)
        return v or None
    except ValueError:
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            r = _int(sd.get("rotation"))
            if r is not None:
                return r
    r = _int((stream.get("tags") or {}).get("rotate"))
    return r or 0


# Muxer-default handler names that carry no information about the track's content.
_GENERIC_HANDLERS = {"soundhandler", "videohandler", "core media audio", "core media video",
                     "iso media file produced by google inc.", "stereo", "mono"}


def _tag(stream: dict, *names: str) -> str | None:
    tags = {k.lower(): v for k, v in (stream.get("tags") or {}).items()}
    for n in names:
        v = tags.get(n.lower())
        if v in (None, "", "und"):
            continue
        if n.lower() == "handler_name" and str(v).strip().lower() in _GENERIC_HANDLERS:
            continue
        return str(v)
    return None


def parse_ffprobe(data: dict, path: str) -> MediaInfo:
    """Pure parser for `ffprobe -show_format -show_streams` JSON."""
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    warnings: list[str] = []

    video: VideoStreamInfo | None = None
    audio: list[AudioStreamInfo] = []
    subs = 0
    for s in streams:
        ctype = s.get("codec_type")
        disp = s.get("disposition") or {}
        if ctype == "video":
            if disp.get("attached_pic") == 1:
                continue  # cover art, not gameplay
            if video is None:
                video = VideoStreamInfo(
                    stream_index=int(s["index"]),
                    codec=s.get("codec_name"),
                    width=_int(s.get("width")),
                    height=_int(s.get("height")),
                    pix_fmt=s.get("pix_fmt"),
                    r_frame_rate=_rate(s.get("r_frame_rate")),
                    avg_frame_rate=_rate(s.get("avg_frame_rate")),
                    rotation=_rotation(s),
                    nb_frames=_int(s.get("nb_frames")),
                    start_time=_float(s.get("start_time")),
                )
            else:
                warnings.append(f"Additional video stream #{s.get('index')} ignored")
        elif ctype == "audio":
            audio.append(AudioStreamInfo(
                stream_index=int(s["index"]),
                audio_index=len(audio),
                codec=s.get("codec_name"),
                channels=_int(s.get("channels")),
                channel_layout=s.get("channel_layout"),
                sample_rate=_int(s.get("sample_rate")),
                language=_tag(s, "language"),
                title=_tag(s, "title", "name", "handler_name"),  # MP4 returns titles as "name"
                is_default=disp.get("default") == 1,
                start_time=_float(s.get("start_time")),
            ))
        elif ctype == "subtitle":
            subs += 1

    duration = _float(fmt.get("duration"))
    if duration is None:
        cands = [_float(s.get("duration")) for s in streams if _float(s.get("duration"))]
        duration = max(cands) if cands else 0.0
        warnings.append("Container has no duration; using longest stream duration")
    if video is None:
        warnings.append("No video stream found")
    if not audio:
        warnings.append("No audio streams found")

    vfr_suspected = False
    if video and video.r_frame_rate and video.avg_frame_rate:
        if abs(video.r_frame_rate - video.avg_frame_rate) / video.r_frame_rate > 0.01:
            vfr_suspected = True

    # Start-time offsets between streams matter for sync later.
    starts = [x for x in [video.start_time if video else None, *[a.start_time for a in audio]] if x is not None]
    if starts and (max(starts) - min(starts)) > 0.1:
        warnings.append(f"Stream start times differ by {max(starts) - min(starts):.3f}s; sync will be preserved by timestamp")

    return MediaInfo(
        path=path,
        format_name=fmt.get("format_name"),
        duration_sec=max(duration or 0.0, 0.0),
        size_bytes=_int(fmt.get("size")),
        bit_rate=_int(fmt.get("bit_rate")),
        video=video,
        audio=audio,
        subtitle_stream_count=subs,
        vfr_suspected=vfr_suspected,
        warnings=warnings,
    )


def probe(path: Path) -> MediaInfo:
    data = fx.ffprobe_json(path, ["-show_format", "-show_streams"])
    return parse_ffprobe(data, str(path))


def analyse_frame_deltas(pts: list[float], tolerance: float = 0.02) -> tuple[bool, dict[str, Any]]:
    """Decide VFR from presentation timestamps: CFR if >=98% of deltas are within tolerance of the median."""
    ts = sorted(t for t in pts if t is not None)
    deltas = [b - a for a, b in zip(ts, ts[1:]) if b - a > 0]
    if len(deltas) < 10:
        return False, {"samples": len(deltas), "note": "too few frames to measure"}
    med = statistics.median(deltas)
    within = sum(1 for d in deltas if abs(d - med) <= med * tolerance)
    ratio = within / len(deltas)
    details = {
        "samples": len(deltas),
        "median_delta": round(med, 6),
        "min_delta": round(min(deltas), 6),
        "max_delta": round(max(deltas), 6),
        "fraction_regular": round(ratio, 4),
    }
    return ratio < 0.98, details


def measure_vfr(path: Path, seconds: float = 30) -> tuple[bool, dict[str, Any]]:
    data = fx.ffprobe_json(path, [
        "-select_streams", "v:0", "-read_intervals", f"%+{seconds:g}",
        "-show_entries", "packet=pts_time",
    ])
    pts = [_float(p.get("pts_time")) for p in data.get("packets", [])]
    return analyse_frame_deltas([p for p in pts if p is not None])


_ROLE_HINTS: list[tuple[TrackRole, re.Pattern]] = [
    (TrackRole.MIC, re.compile(r"\b(mic|microphone|voice|commentary|narration|vo)\b", re.I)),
    (TrackRole.GAME, re.compile(r"\b(game|desktop|system|gameplay)\b", re.I)),
    (TrackRole.COMBINED_MIX, re.compile(r"\b(mix|master|combined|all)\b", re.I)),
]


def suggest_roles(tracks: list[AudioStreamInfo]) -> list[TrackRole]:
    """Suggestions only: the user must confirm. Never assumes a fixed OBS stream layout."""
    if not tracks:
        return []
    if len(tracks) == 1:
        return [TrackRole.COMBINED_MIX]
    roles: list[TrackRole] = []
    used: set[TrackRole] = set()
    for t in tracks:
        role = TrackRole.OTHER
        if t.title:
            for cand, pat in _ROLE_HINTS:
                if pat.search(t.title) and cand not in used:
                    role = cand
                    break
        roles.append(role)
        if role != TrackRole.OTHER:
            used.add(role)
    # Common OBS habit: track 1 is the full mix. Only a hint, shown as unconfirmed.
    if TrackRole.COMBINED_MIX not in used and roles[0] == TrackRole.OTHER:
        roles[0] = TrackRole.COMBINED_MIX
    return roles


def validate_role_mapping(roles: dict[int, TrackRole], track_count: int) -> list[str]:
    """Return human-readable errors (empty list = valid). Keys are audio_index values."""
    errors: list[str] = []
    if track_count == 0:
        return errors
    if set(roles) != set(range(track_count)):
        errors.append("Every audio track needs a role.")
    for unique in (TrackRole.COMBINED_MIX, TrackRole.GAME, TrackRole.MIC):
        n = sum(1 for r in roles.values() if r == unique)
        if n > 1:
            errors.append(f"Only one track can be '{unique.value}' ({n} selected).")
    if track_count == 1 and roles.get(0) == TrackRole.MIC:
        errors.append("A single-track recording cannot be treated as microphone-only unless you are sure it "
                      "contains no game audio. Choose 'combined_mix' or 'game'.")
    return errors


def single_track_notice(track_count: int) -> str | None:
    if track_count == 1:
        return ("This recording has a single audio track. Game audio and microphone (if any) are mixed together "
                "and cannot be separated by this app. Record separate OBS tracks, or add a voiceover later.")
    if track_count == 0:
        return "This recording has no audio. Analysis will rely on visual signals; you can add a voiceover later."
    return None


def track_mean_volume(path: Path, audio_index: int, seconds: float = 120,
                      cancel_check: Callable[[], bool] | None = None) -> float | None:
    """Mean volume (dB) of the first `seconds` of one audio stream, to help identify tracks."""
    proc = fx.ffmpeg(
        ["-t", f"{seconds:g}", "-i", str(path), "-map", f"0:a:{audio_index}", "-af", "volumedetect",
         "-vn", "-sn", "-f", "null", "-"],
        timeout=max(120, seconds * 2), cancel_check=cancel_check,
    )
    m = re.search(r"mean_volume:\s*(-?[\d.]+|-inf)\s*dB", proc.stderr)
    if not m or m.group(1) == "-inf":
        return None
    return float(m.group(1))


def extract_track_preview(path: Path, audio_index: int, out_file: Path, start_sec: float = 0.0,
                          seconds: float = 20) -> Path:
    """Short WAV clip of one stream for the track-mapping screen (source file untouched)."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_name(out_file.stem + ".partial" + out_file.suffix)
    fx.ffmpeg(
        ["-y", "-ss", f"{max(start_sec, 0):.3f}", "-t", f"{seconds:g}", "-i", str(path),
         "-map", f"0:a:{audio_index}", "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(tmp)],
        timeout=120,
    )
    if not tmp.exists() or tmp.stat().st_size < 1000:
        tmp.unlink(missing_ok=True)
        raise fx.FFmpegError("Preview extraction produced no audio")
    tmp.replace(out_file)
    return out_file


def decode_check(path: Path, seconds: float = 5, start_sec: float = 0.0) -> list[str]:
    """Decode a short span fully; return decoder error lines (empty = decodable)."""
    b = fx.binaries()
    proc = fx.run(
        [b.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{start_sec:.3f}", "-t", f"{seconds:g}",
         "-i", str(path), "-f", "null", "-"],
        timeout=300, check=False,
    )
    errs = decode_errors(proc.stderr)
    if proc.returncode != 0 and not errs:
        errs = [f"ffmpeg exited with {proc.returncode}"]
    return errs


# Warnings from the dummy output, not the decoder: phone/VFR recordings often have frames sharing a timestamp,
# which the null muxer reports ("non monotonically increasing dts"). They say nothing about decodability.
_HARMLESS = ("[null @", "non monotonically increasing dts", "non-monotonic dts")


def decode_errors(stderr: str) -> list[str]:
    return [ln for ln in stderr.splitlines() if ln.strip() and not any(h in ln.lower() for h in _HARMLESS)]
