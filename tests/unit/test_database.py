from __future__ import annotations

import sqlite3

import pytest

from pipeline.database import SCHEMA_VERSION, Database, utcnow

TABLES = {"projects", "jobs", "job_stages", "media_assets", "audio_tracks", "events", "highlights",
          "rendered_outputs", "review_decisions", "settings", "schema_version"}


def _project(db: Database, pid: str = "p1") -> None:
    db.execute(
        "INSERT INTO projects(id, name, folder_name, game_name, source_path, source_size, source_quick_fp, "
        "created_at, updated_at, content_style, language, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (pid, "n", f"f_{pid}", "g", "s.mp4", 1, "q", utcnow(), utcnow(), "Custom", "auto", "imported"))


def test_migrate_creates_schema_and_is_idempotent(tmp_path):
    db = Database(tmp_path / "s p ü" / "studio.db")
    assert db.migrate() == SCHEMA_VERSION
    assert db.migrate() == SCHEMA_VERSION
    names = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert TABLES <= names
    assert db.schema_version() == SCHEMA_VERSION
    assert db.query_one("PRAGMA journal_mode")[0] == "wal"


def test_newer_schema_is_refused(tmp_path):
    db = Database(tmp_path / "studio.db")
    db.migrate()
    db.execute("INSERT INTO schema_version(version, applied_at) VALUES (?, ?)", (SCHEMA_VERSION + 1, utcnow()))
    with pytest.raises(RuntimeError, match="newer"):
        db.migrate()


def test_transaction_rolls_back(tmp_path):
    db = Database(tmp_path / "studio.db")
    db.migrate()
    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute("INSERT INTO settings(key, scope, value, updated_at) VALUES ('a','global','1',?)", (utcnow(),))
            raise RuntimeError("boom")
    assert db.query("SELECT * FROM settings") == []


def test_foreign_keys_enforced_and_cascade(tmp_path):
    db = Database(tmp_path / "studio.db")
    db.migrate()
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO jobs(id, project_id, state, created_at, updated_at) VALUES ('j','missing','CREATED',?,?)",
                   (utcnow(), utcnow()))
    _project(db)
    db.execute("INSERT INTO jobs(id, project_id, state, created_at, updated_at) VALUES ('j','p1','CREATED',?,?)",
               (utcnow(), utcnow()))
    db.execute("DELETE FROM projects WHERE id = 'p1'")
    assert db.query("SELECT * FROM jobs") == []


def test_parameterized_values_are_not_executed(tmp_path):
    db = Database(tmp_path / "studio.db")
    db.migrate()
    evil = "x'); DROP TABLE projects; --"
    _project(db, pid=evil)
    assert db.query_one("SELECT id FROM projects")["id"] == evil
