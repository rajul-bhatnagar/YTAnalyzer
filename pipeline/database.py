"""SQLite persistence: versioned schema migrations, transactions, parameterized SQL only."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Each migration is an ordered list of single SQL statements, applied inside one transaction.
MIGRATIONS: list[tuple[int, list[str]]] = [
    (1, [
        """CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            folder_name TEXT NOT NULL UNIQUE,
            game_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_size INTEGER NOT NULL,
            source_quick_fp TEXT NOT NULL,
            source_sha256 TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            duration_sec REAL,
            source_metadata TEXT,
            content_style TEXT NOT NULL,
            language TEXT NOT NULL,
            topic_note TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            tracks_confirmed INTEGER NOT NULL DEFAULT 0
        )""",
        "CREATE INDEX idx_projects_quick_fp ON projects(source_quick_fp, source_size)",
        """CREATE TABLE jobs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            state TEXT NOT NULL,
            resume_state TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            progress REAL NOT NULL DEFAULT 0,
            progress_note TEXT NOT NULL DEFAULT '',
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            pause_requested INTEGER NOT NULL DEFAULT 0,
            error_summary TEXT
        )""",
        "CREATE INDEX idx_jobs_project ON jobs(project_id)",
        """CREATE TABLE job_stages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            stage_name TEXT NOT NULL,
            status TEXT NOT NULL,
            optional INTEGER NOT NULL DEFAULT 0,
            started_at TEXT,
            ended_at TEXT,
            duration_sec REAL,
            input_fingerprint TEXT,
            output_manifest TEXT,
            warnings TEXT,
            error_details TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            UNIQUE(job_id, stage_name)
        )""",
        "CREATE INDEX idx_stages_job ON job_stages(job_id)",
        """CREATE TABLE media_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            asset_type TEXT NOT NULL,
            path TEXT NOT NULL,
            duration_sec REAL,
            codec TEXT,
            width INTEGER,
            height INTEGER,
            sample_rate INTEGER,
            channels INTEGER,
            size_bytes INTEGER,
            checksum TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(project_id, asset_type, path)
        )""",
        "CREATE INDEX idx_assets_project ON media_assets(project_id)",
        """CREATE TABLE audio_tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            stream_index INTEGER NOT NULL,
            audio_index INTEGER NOT NULL,
            codec TEXT,
            channels INTEGER,
            channel_layout TEXT,
            sample_rate INTEGER,
            language TEXT,
            title TEXT,
            detected_role TEXT NOT NULL,
            user_role TEXT,
            extraction_path TEXT,
            mean_volume_db REAL,
            UNIQUE(project_id, stream_index)
        )""",
        "CREATE INDEX idx_tracks_project ON audio_tracks(project_id)",
        """CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            job_id TEXT,
            start_sec REAL NOT NULL,
            end_sec REAL NOT NULL,
            event_type TEXT NOT NULL,
            description TEXT NOT NULL,
            evidence TEXT,
            confidence TEXT,
            source_modality TEXT,
            uncertainty TEXT,
            created_at TEXT NOT NULL
        )""",
        "CREATE INDEX idx_events_project ON events(project_id)",
        """CREATE TABLE highlights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            start_sec REAL NOT NULL,
            end_sec REAL NOT NULL,
            event_sec REAL,
            user_start_sec REAL,
            user_end_sec REAL,
            score REAL,
            score_breakdown TEXT,
            rationale TEXT,
            evidence TEXT,
            confidence TEXT,
            hook_caption TEXT,
            suggested_title TEXT,
            status TEXT NOT NULL DEFAULT 'suggested',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        "CREATE INDEX idx_highlights_project ON highlights(project_id)",
        """CREATE TABLE rendered_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            output_type TEXT NOT NULL,
            path TEXT NOT NULL,
            highlight_id INTEGER,
            settings TEXT,
            validation_status TEXT NOT NULL,
            validation_details TEXT,
            created_at TEXT NOT NULL
        )""",
        "CREATE INDEX idx_outputs_project ON rendered_outputs(project_id)",
        """CREATE TABLE review_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            asset_ref TEXT NOT NULL,
            decision TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )""",
        "CREATE INDEX idx_reviews_project ON review_decisions(project_id)",
        """CREATE TABLE settings (
            key TEXT NOT NULL,
            scope TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (key, scope)
        )""",
    ]),
    # Phase 3: link events/highlights to the analysis run that produced them, keep ranking and user notes.
    (2, [
        "ALTER TABLE events ADD COLUMN analysis_id TEXT",
        "ALTER TABLE events ADD COLUMN event_key TEXT",
        "ALTER TABLE highlights ADD COLUMN analysis_id TEXT",
        "ALTER TABLE highlights ADD COLUMN rank INTEGER",
        "ALTER TABLE highlights ADD COLUMN score_band TEXT",
        "ALTER TABLE highlights ADD COLUMN handles TEXT",
        "ALTER TABLE highlights ADD COLUMN event_keys TEXT",
        "ALTER TABLE highlights ADD COLUMN uncertainty TEXT",
        "ALTER TABLE highlights ADD COLUMN category TEXT",
        "ALTER TABLE highlights ADD COLUMN user_notes TEXT NOT NULL DEFAULT ''",
        "CREATE INDEX idx_events_analysis ON events(project_id, analysis_id)",
        "CREATE INDEX idx_highlights_status ON highlights(project_id, status)",
    ]),
    # Phase 4: per-Short layout/caption settings and the review decision for each rendered Short.
    (3, [
        """CREATE TABLE shorts (
            highlight_id INTEGER PRIMARY KEY REFERENCES highlights(id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            settings TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            notes TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        )""",
        "CREATE INDEX idx_shorts_project ON shorts(project_id)",
    ]),
    # Upgrade: the clip's outcome mood drives the thumbnail look; empty for highlights made before it existed.
    (4, [
        "ALTER TABLE highlights ADD COLUMN outcome_mood TEXT NOT NULL DEFAULT ''",
    ]),
]

SCHEMA_VERSION = MIGRATIONS[-1][0]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, default=str)


def from_json(text: str | None, default: Any = None) -> Any:
    if text is None or text == "":
        return default
    return json.loads(text)


class Database:
    """Thin wrapper: one short-lived connection per unit of work (safe across threads, WAL mode)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._init_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE ... COMMIT, rolled back on any exception."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def migrate(self) -> int:
        with self._init_lock:
            with self.connect() as conn:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
                )
                row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
                current = row["v"] or 0
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema v{current} is newer than this app supports (v{SCHEMA_VERSION}). "
                    "Update the application code."
                )
            for version, statements in MIGRATIONS:
                if version <= current:
                    continue
                with self.transaction() as conn:
                    for stmt in statements:
                        conn.execute(stmt)
                    conn.execute("INSERT INTO schema_version(version, applied_at) VALUES (?, ?)", (version, utcnow()))
            return SCHEMA_VERSION

    def schema_version(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            return row["v"] or 0

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(sql, params).fetchone()

    def execute(self, sql: str, params: tuple | dict = ()) -> int:
        """Run one write statement in its own transaction; returns lastrowid."""
        with self.transaction() as conn:
            return conn.execute(sql, params).lastrowid
