"""Import a recording and create a project."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from pipeline.importer import ImportErrorUser
from pipeline.models import CONTENT_STYLES, LANGUAGES
from pipeline.project_service import DuplicateProject
from ui.common import fmt_size, get_service, goto, select_project

PENDING_KEY = "import_pending_source"


def _source_picker(svc) -> tuple[Path | None, str | None]:
    tab_folder, tab_upload = st.tabs(["From 01_RAW_INPUT folder", "Upload a file"])
    chosen: Path | None = None
    game_guess: str | None = None
    with tab_folder:
        files = svc.scan_raw_input()
        st.caption(f"Folder: `{svc.paths.raw_input}`. Subfolders are allowed; the first folder name is used "
                   "as the suggested game name.")
        if st.button("Rescan folder"):
            st.rerun()
        if not files:
            st.info("No supported recordings found in 01_RAW_INPUT.")
        else:
            labels = {f.rel: f"{f.rel}  ({fmt_size(f.size_bytes)})" for f in files}
            rel = st.selectbox("Recording", list(labels), format_func=lambda r: labels[r])
            pick = next(f for f in files if f.rel == rel)
            chosen, game_guess = pick.path, pick.game_guess
    with tab_upload:
        cfg = svc.effective_settings()["import"]
        st.caption(f"Browser uploads are held in memory; limit {cfg['max_upload_mb']} MB. "
                   "For long recordings, copy the file into 01_RAW_INPUT instead.")
        up_game = st.text_input("Game name for the upload folder", key="upload_game")
        up = st.file_uploader("Recording", type=[e.lstrip(".") for e in cfg["video_extensions"]])
        if up is not None and st.button("Save upload to 01_RAW_INPUT", disabled=not up_game.strip()):
            bar = st.progress(0.0, text="Saving upload")
            try:
                saved = svc.save_upload(up, up.name, up_game, up.size, progress=lambda f: bar.progress(f))
                st.session_state[PENDING_KEY] = (str(saved), up_game.strip())
                st.success(f"Saved to {svc.paths.to_stored(saved)}")
            except ImportErrorUser as e:
                st.error(str(e))
        pending = st.session_state.get(PENDING_KEY)
        if pending:
            chosen, game_guess = Path(pending[0]), pending[1]
            st.info(f"Using uploaded file: {Path(pending[0]).name}")
    return chosen, game_guess


def render() -> None:
    svc = get_service()
    st.title("Import / New project")
    source, guess = _source_picker(svc)
    if source is None:
        return

    st.subheader("Project details")
    history = svc.game_suggestions()
    col1, col2 = st.columns(2)
    with col1:
        quick = st.selectbox("Recent games", ["(type a name)"] + history,
                             help="Any game name is allowed; nothing is hardcoded.")
        default_game = guess or (quick if quick != "(type a name)" else "")
        game = st.text_input("Game name", value=default_game, key=f"game_{source}_{quick}")
        style = st.selectbox("Content style", CONTENT_STYLES, index=CONTENT_STYLES.index("Custom"))
    with col2:
        lang = st.selectbox("Language", list(LANGUAGES), format_func=lambda k: LANGUAGES[k])
        note = st.text_input("One-line challenge / topic note (optional)", max_chars=300)
        name = st.text_input("Project name (optional)", placeholder=f"{game or 'Game'} - {source.stem}")

    allow_dup = st.session_state.get("allow_duplicate_for") == str(source)
    if st.button("Create project and inspect", type="primary", disabled=not game.strip()):
        try:
            pid = svc.create_project(source, game, style, lang, note, name or None, allow_duplicate=allow_dup)
        except DuplicateProject as e:
            st.warning(f"{e} You can open the existing project, or tick the box below to import a second copy.")
            st.session_state["duplicate_of"] = (str(source), e.project_id)
        except (ValueError, FileNotFoundError) as e:
            st.error(str(e))
        else:
            st.session_state.pop(PENDING_KEY, None)
            st.session_state.pop("duplicate_of", None)
            select_project(pid)
            svc.start_inspection(pid)
            goto("processing")

    dup = st.session_state.get("duplicate_of")
    if dup and dup[0] == str(source):
        c1, c2 = st.columns(2)
        if c1.button("Open existing project"):
            select_project(dup[1])
            goto("processing")
        if c2.checkbox("Import anyway as a separate project", value=allow_dup):
            st.session_state["allow_duplicate_for"] = str(source)
        else:
            st.session_state.pop("allow_duplicate_for", None)
