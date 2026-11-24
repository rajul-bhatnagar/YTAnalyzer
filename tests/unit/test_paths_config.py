from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import config
from pipeline.config import SettingsStore
from pipeline.database import Database
from pipeline.paths import PROJECT_ROOT, StudioPaths, is_within, safe_filename, slugify


def test_project_root_resolved_from_source_not_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert (PROJECT_ROOT / "pipeline" / "paths.py").exists()
    assert StudioPaths.default().root == PROJECT_ROOT


@pytest.mark.parametrize("text,expected", [
    ("Grand Theft Auto V", "grand_theft_auto_v"),
    ("Pokémon Légendes", "pokemon_legendes"),
    ("क्रिकेट 24", "24"),
    ("円", "project"),
])
def test_slugify(text, expected):
    assert slugify(text) == expected


def test_safe_filename_windows_rules():
    assert safe_filename('a<b>:c"d/e\\f|g?h*') == "a_b__c_d_e_f_g_h_"
    assert safe_filename("CON.mp4") == "_CON.mp4"
    assert safe_filename("  trailing dot. ") == "trailing dot"
    assert safe_filename("क्रिकेट gameplay.mp4") == "क्रिकेट gameplay.mp4"


def test_stored_paths_portable(studio_paths: StudioPaths):
    f = studio_paths.raw_input / "My Game ü" / "rec 1.mp4"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    stored = studio_paths.to_stored(f)
    assert stored == "01_RAW_INPUT/My Game ü/rec 1.mp4"
    assert studio_paths.from_stored(stored).resolve() == f.resolve()
    outside = studio_paths.root.parent / "elsewhere.mp4"
    assert Path(studio_paths.to_stored(outside)).is_absolute()
    assert is_within(f, studio_paths.root) and not is_within(outside, studio_paths.root)


def test_settings_precedence_and_env(studio_paths: StudioPaths, monkeypatch):
    db = Database(studio_paths.db_file)
    db.migrate()
    store = SettingsStore(db, studio_paths)
    assert store.effective()["ai"]["gemini_model"] == "gemini-3.8-flash"
    monkeypatch.setenv("GEMINI_MODEL", "env-model")
    assert store.effective()["ai"]["gemini_model"] == "env-model"
    store.set("ai.gemini_model", "global-model")
    assert store.effective()["ai"]["gemini_model"] == "global-model"
    store.set("ai.gemini_model", "project-model", scope="p1")
    assert store.effective("p1")["ai"]["gemini_model"] == "project-model"
    assert store.effective("p2")["ai"]["gemini_model"] == "global-model"
    store.clear("ai.gemini_model")
    assert store.effective()["ai"]["gemini_model"] == "env-model"


def test_settings_reject_secrets_and_unknown_keys(studio_paths: StudioPaths):
    db = Database(studio_paths.db_file)
    db.migrate()
    store = SettingsStore(db, studio_paths)
    with pytest.raises(ValueError):
        store.set("ai.gemini_api_key", "AIzaSomething")
    with pytest.raises(KeyError):
        store.set("ai.not_a_setting", 1)
    rows = db.query("SELECT * FROM settings")
    assert rows == []


def test_secret_status_never_exposes_values(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AI" + "za" + "SyTESTVALUE" + "1234567890123456789012")  # fake, in pieces
    status = config.secret_status()
    assert status["GEMINI_API_KEY"] is True
    assert "AIza" not in json.dumps(status)
    with pytest.raises(KeyError):
        config.get_secret("PATH")


def test_games_history(studio_paths: StudioPaths):
    assert config.game_suggestions(studio_paths) == []
    config.record_game(studio_paths, "Cricket 24")
    config.record_game(studio_paths, "Phasmophobia")
    config.record_game(studio_paths, "cricket 24")
    hist = config.load_games_history(studio_paths)
    assert {e["name"]: e["count"] for e in hist} == {"Cricket 24": 2, "Phasmophobia": 1}
    assert config.game_suggestions(studio_paths)[0] == "Cricket 24"
    studio_paths.games_history_file.write_text("{not json", encoding="utf-8")
    assert config.load_games_history(studio_paths) == []
