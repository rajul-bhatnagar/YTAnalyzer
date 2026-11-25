"""Thumbnails: candidate frames, optional cut-outs, two Gemini-web prompts, and importing the finished thumbnail.
The app never generates images or calls a paid image API."""

from __future__ import annotations

import streamlit as st

from pipeline import longform, thumbnail_prep
from pipeline.ai_providers.base import OUTCOME_MOODS
from pipeline.frame_sampler import fmt_ts
from pipeline.project_service import StudioService
from ui.common import get_service, job_status_panel, require_project


def _k(pid: str, name: str) -> str:
    return f"th_{pid}_{name}"


def _busy(svc: StudioService, pid: str) -> bool:
    job = svc.latest_job(pid)
    return bool(job and svc.is_job_running(job["id"]))


def _frames(svc: StudioService, pid: str, data: dict, busy: bool) -> None:
    st.subheader("Candidate frames")
    st.caption("Chosen around your approved highlights' key moments (not from loudness alone), at full resolution, "
               "ranked by sharpness, exposure and colour; near-identical frames are dropped.")
    can_cut = svc.cutouts_available()
    if not can_cut:
        st.caption("Background cut-outs: the optional `rembg` package is not installed (not needed for thumbnails).")
    else:
        st.warning("Cut-outs use the optional rembg package. This integration is untested in this project, and rembg "
                   "may download its segmentation model on first use.")
        st.checkbox("I understand; allow cut-outs", key=_k(pid, "cut_ok"))
    cols = st.columns(3)
    for i, fr in enumerate(data["frames"]):
        with cols[i % 3]:
            if fr["path"]:
                st.image(str(fr["path"]), caption=f"{fmt_ts(fr['t'])} · {fr['why']}")
                note = f"{fr['width']}×{fr['height']} · sharpness {fr['sharpness']:.0f} · score {fr['score']:.2f}"
                if fr["issues"]:
                    note += " · ⚠ " + ", ".join(fr["issues"])
                st.caption(note)
                st.download_button("Download PNG", fr["path"].read_bytes(), file_name=fr["path"].name,
                                   mime="image/png", key=_k(pid, f"dl_{i}"))
                if fr["cutout"]:
                    st.image(str(fr["cutout"]), caption="Cut-out")
                elif can_cut and st.button("Make cut-out", key=_k(pid, f"cut_{i}"),
                                           disabled=busy or not st.session_state.get(_k(pid, "cut_ok"))):
                    try:
                        with st.spinner("Removing background..."):
                            svc.make_cutout(pid, fr["path"])
                        st.rerun()
                    except Exception as e:  # noqa: BLE001 - optional feature: report, never crash the page
                        st.error(f"Cut-out failed: {type(e).__name__}: {e}")


def _prompts(svc: StudioService, pid: str, data: dict) -> None:
    st.subheader("Prompts for the free Gemini web app")
    st.caption("Copy a prompt (copy icon on each box), open Gemini in your browser, attach a frame for prompt 1, and "
               "generate. The prompts describe only what happens in your video.")
    labels = {h["id"]: h["suggested_title"] or h["hook_caption"] or f"Highlight #{h['id']}"
              for h in svc.highlights(pid)}
    for hid, p in data["prompts"].items():
        with st.expander(labels.get(int(hid), f"Highlight #{hid}"), expanded=len(data["prompts"]) == 1):
            if p["needs_user_words"]:
                st.warning("This highlight has no title or hook yet, so the prompt contains a [placeholder]. Add a title "
                           "or hook on the Analysis & highlights page and prepare again.")
            st.markdown("**1 · Edit a screenshot** (attach one of the frames above)")
            st.code(p["reference_edit"], language=None, wrap_lines=True)
            st.markdown("**2 · From scratch (16:9)**")
            st.code(p["from_scratch"], language=None, wrap_lines=True)
            st.caption(f"Suggested text: {p['text'] or 'none'} (2–3 uppercase words at most).")


def _compose(svc: StudioService, pid: str, data: dict | None, busy: bool) -> None:
    st.subheader("Add title words (local, free)")
    st.caption("Puts 2-3 bold uppercase words on a frame or on the image you made in Gemini, coloured by the clip's "
               "outcome mood, keeping the bottom-right corner clear. Saved as your thumbnail; the previous one is kept.")
    options: dict[str, Path] = {}
    cur = svc.user_thumbnail(pid)
    if cur:
        options[f"Your imported image ({cur['path'].name})"] = cur["path"]
    for fr in (data or {}).get("frames", []):
        if fr["path"]:
            options[f"Frame at {fmt_ts(fr['t'])}"] = fr["path"]
    if not options:
        st.caption("Prepare frames or import an image first.")
        return
    first = next(iter((data or {}).get("prompts", {}).values()), {})
    src = st.selectbox("Image", list(options), key=_k(pid, "cmp_src"))
    c1, c2 = st.columns(2)
    words = c1.text_input("Words (2-3)", first.get("text", ""), key=_k(pid, "cmp_words"), placeholder="e.g. SUPER OVER WIN")
    moods = ["", *OUTCOME_MOODS]
    mood = c2.selectbox("Mood", moods, index=moods.index(first.get("mood", "")) if first.get("mood", "") in moods else 0,
                        format_func=lambda m: OUTCOME_MOODS.get(m, "Neutral"), key=_k(pid, "cmp_mood"))
    clean, problem = thumbnail_prep.validate_words(words)
    if words and problem:
        st.warning(problem)
    if st.button("Compose thumbnail", type="primary", key=_k(pid, "cmp_go"), disabled=busy or clean is None):
        try:
            res = svc.compose_thumbnail(pid, options[src], clean or "", mood)
            for w in res["warnings"]:
                st.warning(w)
            st.rerun()
        except ValueError as e:
            st.error(str(e))


def _import(svc: StudioService, pid: str, busy: bool) -> None:
    st.subheader("Your finished thumbnail")
    cur = svc.user_thumbnail(pid)
    up = st.file_uploader("Import the thumbnail you made (PNG or JPG)", type=["png", "jpg", "jpeg"],
                          key=_k(pid, "up"), disabled=busy)
    if up is not None and st.button("Use this thumbnail", type="primary", key=_k(pid, "use")):
        try:
            res = svc.import_user_thumbnail(pid, up, up.name)
            for w in res["warnings"]:
                st.warning(w)
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    badge = st.toggle("Preview with a YouTube video-length badge", True, key=_k(pid, "badge"))
    cur = svc.user_thumbnail(pid, badge=badge) if cur else None
    if cur:
        c1, c2 = st.columns(2)
        c1.image(str(cur["path"]), caption=f"{cur['info'].get('width')}×{cur['info'].get('height')}")
        if cur["preview"]:
            c2.image(str(cur["preview"]), caption="Mock '12:34' badge where YouTube shows the length (approximate)"
                     if badge else "Red box: approximate area covered by YouTube's video length badge")
        for w in cur["warnings"]:
            st.warning(w)


def render() -> None:
    svc = get_service()
    proj = require_project(svc)
    if proj is None:
        return
    st.title("Thumbnails")
    reason = svc.longform_ready(proj["id"])
    if reason:
        st.info(reason)
        return
    pid = proj["id"]
    job_status_panel(pid, dict(longform.STAGE_LABELS), "th", include_content=True)
    busy = _busy(svc, pid)
    if st.button("Prepare thumbnail frames & prompts", type="primary", disabled=busy, key=_k(pid, "go")):
        try:
            svc.start_content(pid, "thumbnails")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    data = svc.thumbnails(pid)
    if not data:
        st.info("Nothing prepared yet. Approve highlights first for the best frames, then press the button above.")
    else:
        _frames(svc, pid, data, busy)
        _prompts(svc, pid, data)
    _compose(svc, pid, data, busy)
    _import(svc, pid, busy)
