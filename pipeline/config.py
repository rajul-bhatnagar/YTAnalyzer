"""Settings: defaults JSON < .env overrides < global DB settings < project DB settings.

Secrets live only in environment variables / .env and are never written to SQLite or logs.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .database import Database, from_json, to_json, utcnow
from .paths import StudioPaths

SECRET_KEYS = {"GEMINI_API_KEY", "YOUTUBE_CLIENT_SECRETS_FILE"}
_SECRET_WORDS = ("api_key", "apikey", "secret", "token", "password")

# Non-secret environment variables that override defaults (dotted settings key).
ENV_OVERRIDES = {
    "GEMINI_MODEL": "ai.gemini_model",
    "OLLAMA_BASE_URL": "ai.ollama_base_url",
    "OLLAMA_MODEL": "ai.ollama_model",
}

GLOBAL_SCOPE = "global"


def load_env(paths: StudioPaths) -> bool:
    """Load .env without overriding variables already set in the process environment."""
    if paths.env_file.exists():
        return load_dotenv(paths.env_file, override=False)
    return False


def get_secret(name: str) -> str | None:
    if name not in SECRET_KEYS:
        raise KeyError(f"{name} is not a recognised secret")
    val = os.environ.get(name, "").strip()
    return val or None


def secret_status() -> dict[str, bool]:
    """Which secrets are configured, without exposing values."""
    return {k: bool(os.environ.get(k, "").strip()) for k in sorted(SECRET_KEYS)}


def load_defaults(paths: StudioPaths) -> dict[str, Any]:
    with paths.default_settings_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def get_dotted(d: dict, key: str, default: Any = None) -> Any:
    cur: Any = d
    for p in key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _looks_secret(key: str) -> bool:
    k = key.lower()
    return any(w in k for w in _SECRET_WORDS)


class SettingsStore:
    def __init__(self, db: Database, paths: StudioPaths):
        self.db = db
        self.paths = paths

    def set(self, key: str, value: Any, scope: str = GLOBAL_SCOPE) -> None:
        if _looks_secret(key):
            raise ValueError(f"Refusing to store secret-like setting '{key}' in the database; use .env")
        defaults = load_defaults(self.paths)
        if get_dotted(defaults, key, _MISSING) is _MISSING:
            raise KeyError(f"Unknown setting '{key}'")
        self.db.execute(
            "INSERT INTO settings(key, scope, value, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key, scope) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, scope, to_json(value), utcnow()),
        )

    def set_many(self, values: dict[str, Any], scope: str = GLOBAL_SCOPE) -> None:
        for k, v in values.items():
            self.set(k, v, scope)

    def clear(self, key: str, scope: str = GLOBAL_SCOPE) -> None:
        self.db.execute("DELETE FROM settings WHERE key = ? AND scope = ?", (key, scope))

    def stored(self, scope: str = GLOBAL_SCOPE) -> dict[str, Any]:
        rows = self.db.query("SELECT key, value FROM settings WHERE scope = ?", (scope,))
        return {r["key"]: from_json(r["value"]) for r in rows}

    def effective(self, project_id: str | None = None) -> dict[str, Any]:
        merged = copy.deepcopy(load_defaults(self.paths))
        for env_name, key in ENV_OVERRIDES.items():
            val = os.environ.get(env_name, "").strip()
            if val:
                _set_dotted(merged, key, val)
        for k, v in self.stored(GLOBAL_SCOPE).items():
            _set_dotted(merged, k, v)
        if project_id:
            for k, v in self.stored(project_id).items():
                _set_dotted(merged, k, v)
        return merged


_MISSING = object()


# --- Game history (portable JSON, per machine, git-ignored) -----------------------------

def load_games_history(paths: StudioPaths) -> list[dict[str, Any]]:
    f = paths.games_history_file
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return [e for e in data if isinstance(e, dict) and isinstance(e.get("name"), str)]
    except (json.JSONDecodeError, OSError):
        return []


def record_game(paths: StudioPaths, name: str) -> None:
    name = name.strip()
    if not name:
        return
    history = load_games_history(paths)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for e in history:
        if e["name"].casefold() == name.casefold():
            e["count"] = int(e.get("count", 0)) + 1
            e["last_used"] = now
            break
    else:
        history.append({"name": name, "count": 1, "last_used": now})
    history.sort(key=lambda e: e.get("last_used", ""), reverse=True)
    paths.games_history_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.games_history_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(paths.games_history_file)


def game_suggestions(paths: StudioPaths) -> list[str]:
    return [e["name"] for e in load_games_history(paths)]
