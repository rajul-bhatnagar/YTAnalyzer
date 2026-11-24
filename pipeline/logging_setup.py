"""Rotating file logging with context fields and secret redaction."""

from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

SECRET_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "YOUTUBE_CLIENT_SECRET")

_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API key shape
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|authorization)"
               r"(\s*[=:]\s*|\"\s*:\s*\")(?:bearer\s+)?([^\s\"',&]+)"),
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._\-]+)"),
]

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | project=%(project)s job=%(job)s stage=%(stage)s | %(message)s"


def redact(text: str) -> str:
    for var in SECRET_ENV_VARS:
        val = os.environ.get(var)
        if val and len(val) >= 6:
            text = text.replace(val, "***REDACTED***")
    text = _PATTERNS[0].sub("***REDACTED***", text)
    text = _PATTERNS[1].sub(lambda m: f"{m.group(1)}{m.group(2)}***REDACTED***", text)
    text = _PATTERNS[2].sub(lambda m: f"{m.group(1)}***REDACTED***", text)
    return text


class ContextAndRedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for field in ("project", "job", "stage"):
            if not hasattr(record, field):
                setattr(record, field, "-")
        try:
            msg = record.getMessage()
        except Exception:  # malformed args: keep raw message
            msg = str(record.msg)
        record.msg = redact(msg)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


_configured: set[str] = set()


def setup_logging(logs_dir: Path, level: int = logging.INFO, filename: str = "studio.log") -> Path:
    """Idempotently attach a rotating file handler (5 MB x 5) and a console handler to the root logger."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = (logs_dir / filename).resolve()
    key = str(log_file)
    root = logging.getLogger()
    root.setLevel(level)
    if key in _configured:
        return log_file
    flt = ContextAndRedactFilter()
    fmt = logging.Formatter(LOG_FORMAT)
    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.addFilter(flt)
    root.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.addFilter(flt)
        root.addHandler(ch)
    _configured.add(key)
    return log_file


def ctx_logger(name: str, project: str | None = None, job: str | None = None, stage: str | None = None) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logging.getLogger(name), {"project": project or "-", "job": job or "-", "stage": stage or "-"})


def tail_log(log_file: Path, max_lines: int = 300) -> str:
    if not log_file.exists():
        return ""
    with log_file.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])
