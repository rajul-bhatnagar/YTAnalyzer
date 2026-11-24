"""Input/output validation helpers. A stage is complete only after its outputs pass these checks."""

from __future__ import annotations

from pathlib import Path

from .media_probe import decode_check
from .models import MediaInfo


class ValidationError(ValueError):
    pass


def clamp_range(start: float, end: float, duration: float, min_len: float = 0.0) -> tuple[float, float]:
    """Clamp [start, end] into [0, duration]; raise if nothing valid remains."""
    if duration <= 0:
        raise ValidationError("Media duration is unknown or zero")
    s = min(max(float(start), 0.0), duration)
    e = min(max(float(end), 0.0), duration)
    if e < s:
        s, e = e, s
    if e - s < min_len:
        raise ValidationError(f"Range {start:.2f}-{end:.2f}s is shorter than {min_len:.2f}s after clamping")
    return s, e


def validate_extension(path: Path, allowed: list[str]) -> None:
    if Path(path).suffix.lower() not in {a.lower() for a in allowed}:
        raise ValidationError(f"Unsupported file type '{Path(path).suffix}'. Allowed: {', '.join(allowed)}")


def validate_source_media(info: MediaInfo, path: Path, decode_seconds: float = 5) -> list[str]:
    """Hard errors that make a recording unusable as a source. Warnings stay in info.warnings."""
    errors: list[str] = []
    if info.video is None:
        errors.append("No video stream: this file cannot be used as a gameplay recording.")
    if info.duration_sec <= 0:
        errors.append("Duration is zero or unknown.")
    if info.video and (not info.video.width or not info.video.height):
        errors.append("Video resolution could not be read.")
    if not errors:
        dec = decode_check(path, seconds=min(decode_seconds, max(info.duration_sec, 0.1)))
        if dec:
            errors.append("Start of file failed to decode: " + "; ".join(dec[:3]))
        if info.duration_sec > 20:
            dec_end = decode_check(path, seconds=decode_seconds, start_sec=max(info.duration_sec - 10, 0))
            if dec_end:
                errors.append("End of file failed to decode (truncated recording?): " + "; ".join(dec_end[:3]))
    return errors
