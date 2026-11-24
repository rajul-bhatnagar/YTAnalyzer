"""Creator package review: titles, hook, timestamped voiceover script, Shorts titles, description, tags, pinned
comment. Every field is editable; edits are saved per field and survive regeneration unless you discard them."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from pipeline import longform
from pipeline import youtube_metadata as ym
from pipeline.frame_sampler import fmt_ts
from pipeline.models import LANGUAGES
from pipeline.project_service import StudioService
from ui.common import get_service, job_status_panel, require_project


def _k(pid: str, name: str) -> str:
    return f"md_{pid}_{name}"


def _busy(svc: StudioService, pid: str) -> bool:
    job = svc.latest_job(pid)
    return bool(job and svc.is_job_running(job["id"]))


def _save(svc: StudioService, pid: str, field: str, value: Any) -> None:
    try:
        svc.save_pack_field(pid, field, value)
        st.toast(f"Saved {field.replace('_', ' ')}.")
        st.rerun()
    except (ValueError, KeyError, TypeError) as e:
        st.error(str(e))


def _edited(pack: dict[str, Any], field: str) -> str:
    return " · ✏️ edited by you" if field in pack.get("edited_fields", []) else ""


def _generate(svc: StudioService, proj: dict[str, Any], busy: bool) -> None:
    pid = proj["id"]
    s = svc.effective_settings(pid)
    langs = ["", *LANGUAGES]
    cur = s["pack"]["language"]
    lang = st.selectbox("Language of the package", langs, index=langs.index(cur) if cur in langs else 0,
                        format_func=lambda k: f"Same as project ({LANGUAGES[proj['language']]})" if k == ""
                        else LANGUAGES[k], key=_k(pid, "lang"))
    if lang == "hinglish" or (lang == "" and proj["language"] == "hinglish"):
        st.caption("Hinglish output depends on the AI model; check the wording. The no-AI template is English only.")
    st.caption(f"Written by the AI provider set on Analysis & highlights ({s['ai']['provider']}), from your approved "
               "highlights, detected events and transcript only. Without a working AI provider you get a template "
               "with [placeholders].")
    c1, c2 = st.columns(2)
    if c1.button("Generate / update package", type="primary", disabled=busy, key=_k(pid, "gen"),
                 help="Your edited fields are kept."):
        svc.save_project_settings(pid, {"pack.language": lang})
        _start(svc, pid, discard=False)
    sure = c2.checkbox("Also discard my edits", key=_k(pid, "discard_ok"))
    if c2.button("Regenerate and discard edits", disabled=busy or not sure, key=_k(pid, "regen")):
        svc.save_project_settings(pid, {"pack.language": lang})
        _start(svc, pid, discard=True)


def _start(svc: StudioService, pid: str, discard: bool) -> None:
    try:
        svc.start_content(pid, "pack", discard_pack_edits=discard)
        st.rerun()
    except ValueError as e:
        st.error(str(e))


def _fields(svc: StudioService, pid: str, pack: dict[str, Any], busy: bool) -> None:
    src = pack.get("source") or {}
    st.caption(f"Written by **{src.get('provider', 'template')}** {src.get('model', '')} · language "
               f"{pack.get('language')} · {pack.get('written_at', '')}")
    checks = [*pack.get("number_flags", []), *pack.get("uncertainty", [])]
    if checks:
        with st.expander(f"⚠ Check before publishing ({len(checks)})", expanded=True):
            for c in checks:
                st.write(f"- {c}")

    st.subheader("Titles" + _edited(pack, "titles"))
    titles = [st.text_input(f"Title {i + 1}", t, max_chars=ym.TITLE_MAX, key=_k(pid, f"t{i}"))
              for i, t in enumerate((pack["titles"] + [""] * 5)[:5])]
    if st.button("Save titles", key=_k(pid, "save_titles"), disabled=busy):
        _save(svc, pid, "titles", titles)

    st.subheader("Opening hook (~20 s)" + _edited(pack, "opening_hook"))
    hook = st.text_area("Opening hook", pack["opening_hook"], height=100, key=_k(pid, "hook"),
                        label_visibility="collapsed")
    st.caption(f"{len(hook.split())} words (about 50 words ≈ 20 seconds spoken)")
    if st.button("Save hook", key=_k(pid, "save_hook"), disabled=busy):
        _save(svc, pid, "opening_hook", hook)

    st.subheader("Voiceover script" + _edited(pack, "script"))
    st.caption("Times are recording time; 'video' is the time in the rough cut (blank = cut out). Lines marked "
               "'check' need your confirmation.")
    df = pd.DataFrame([{"time_sec": float(x["time_sec"]), "at": fmt_ts(x["time_sec"]),
                        "video": fmt_ts(x["video_time_sec"]) if x.get("video_time_sec") is not None else "",
                        "text": x["text"], "check": bool(x.get("uncertain")), "event_key": x.get("event_key") or ""}
                       for x in pack["script"]])
    if df.empty:
        df = pd.DataFrame(columns=["time_sec", "at", "video", "text", "check", "event_key"])
    edited = st.data_editor(df, key=_k(pid, "script"), hide_index=True, num_rows="dynamic",
                            disabled=["at", "video", "event_key"],
                            column_config={"time_sec": st.column_config.NumberColumn("seconds", min_value=0.0,
                                                                                     step=0.5),
                                           "text": st.column_config.TextColumn("line", width="large"),
                                           "check": st.column_config.CheckboxColumn("check")})
    if st.button("Save script", key=_k(pid, "save_script"), disabled=busy):
        rows = [r for r in edited.to_dict("records") if str(r.get("text") or "").strip()]
        _save(svc, pid, "script", [{"time_sec": r["time_sec"], "text": r["text"], "uncertain": r["check"],
                                    "event_key": r.get("event_key") or ""} for r in rows])

    st.subheader("Shorts" + _edited(pack, "shorts"))
    sdf = pd.DataFrame([{"highlight_id": s["highlight_id"], "clip": f"{fmt_ts(s.get('start', 0))}–"
                         f"{fmt_ts(s.get('end', 0))}", "title": s["title"], "hook": s["hook"]} for s in pack["shorts"]])
    if not sdf.empty:
        sed = st.data_editor(sdf, key=_k(pid, "shorts"), hide_index=True, disabled=["highlight_id", "clip"])
        if st.button("Save Shorts titles", key=_k(pid, "save_shorts"), disabled=busy):
            by = {s["highlight_id"]: s for s in pack["shorts"]}
            _save(svc, pid, "shorts", [{**by[r["highlight_id"]], "title": r["title"], "hook": r["hook"]}
                                       for r in sed.to_dict("records")])
    else:
        st.caption("No approved highlights yet.")

    st.subheader("Description" + _edited(pack, "description"))
    desc = st.text_area("Description", pack["description"], height=220, key=_k(pid, "desc"),
                        label_visibility="collapsed")
    st.caption(f"{len(desc)} / {ym.DESCRIPTION_MAX} characters (YouTube's limit as known to the app; verify in Studio)")
    if st.button("Save description", key=_k(pid, "save_desc"), disabled=busy):
        _save(svc, pid, "description", desc)

    st.subheader("Tags" + _edited(pack, "tags"))
    tags_text = st.text_area("Tags (comma-separated, no #)", ", ".join(pack["tags"]), height=80, key=_k(pid, "tags"))
    tags = [t.strip() for t in tags_text.split(",") if t.strip()]
    total = sum(len(t) for t in tags) + max(len(tags) - 1, 0)
    st.caption(f"{len(tags)} tags · about {total} / {ym.TAGS_TOTAL_MAX} characters (approximate; verify in Studio)")
    if st.button("Save tags", key=_k(pid, "save_tags"), disabled=busy):
        _save(svc, pid, "tags", tags)

    st.subheader("Pinned comment (optional)" + _edited(pack, "pinned_comment"))
    pin = st.text_area("Pinned comment", pack["pinned_comment"], height=80, key=_k(pid, "pin"),
                       label_visibility="collapsed")
    if st.button("Save pinned comment", key=_k(pid, "save_pin"), disabled=busy):
        _save(svc, pid, "pinned_comment", pin)

    if pack.get("style_rationale"):
        st.caption(f"Why this fits the style: {pack['style_rationale']}")


def _copy(svc: StudioService, pid: str, pack: dict[str, Any]) -> None:
    st.subheader("Copy")
    f = svc.files(pid)
    txt = f.pack_txt.read_text("utf-8") if f.pack_txt.exists() else ym.to_text(pack)
    c1, c2 = st.columns(2)
    c1.download_button("Download youtube_pack.txt", txt, file_name="youtube_pack.txt", mime="text/plain",
                       key=_k(pid, "dl_txt"))
    if f.pack_json.exists():
        c2.download_button("Download youtube_pack.json", f.pack_json.read_bytes(), file_name="youtube_pack.json",
                           mime="application/json", key=_k(pid, "dl_json"))
    with st.expander("Individual fields (use the copy icon on each box)"):
        st.markdown("**Title 1**")
        st.code(pack["titles"][0] if pack["titles"] else "", language=None, wrap_lines=True)
        st.markdown("**Description**")
        st.code(pack["description"], language=None, wrap_lines=True)
        st.markdown("**Tags**")
        st.code(", ".join(pack["tags"]), language=None, wrap_lines=True)
    with st.expander("Complete package as text"):
        st.code(txt, language=None, wrap_lines=True)


def render() -> None:
    svc = get_service()
    proj = require_project(svc)
    if proj is None:
        return
    st.title("Metadata & scripts")
    reason = svc.longform_ready(proj["id"])
    if reason:
        st.info(reason)
        return
    pid = proj["id"]
    job_status_panel(pid, dict(longform.STAGE_LABELS), "md", include_content=True)
    busy = _busy(svc, pid)
    _generate(svc, proj, busy)
    pack = svc.creator_pack(pid)
    if not pack:
        st.info("No package yet. Approve highlights, then press **Generate / update package**.")
        return
    _fields(svc, pid, pack, busy)
    _copy(svc, pid, pack)
