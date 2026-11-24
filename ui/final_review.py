"""Final approval checklist and the optional private YouTube upload."""

from __future__ import annotations

import streamlit as st

from pipeline import youtube_draft
from pipeline.project_service import StudioService
from ui.common import get_service, job_status_panel, require_project

ICON = {"ok": "✅", "warn": "⚠️", "block": "⛔"}


def _k(pid: str, name: str) -> str:
    return f"fr_{pid}_{name}"


def _busy(svc: StudioService, pid: str) -> bool:
    job = svc.latest_job(pid)
    return bool(job and svc.is_job_running(job["id"]))


def _checklist(svc: StudioService, pid: str, busy: bool) -> dict:
    chk = svc.final_check(pid)
    st.subheader("Checklist")
    for i in chk["items"]:
        st.write(f"{ICON[i['status']]} **{i['name']}**: {i['detail']}")
    ack = False
    if chk["needs_ack"]:
        with st.expander(f"Flagged in the package ({len(chk['needs_ack'])})", expanded=True):
            for n in chk["needs_ack"]:
                st.write(f"- {n}")
        ack = st.checkbox("I checked these items", key=_k(pid, "ack"))
    if chk["approved"]:
        st.success("Approved. The approved files have not changed since approval.")
    if st.button("Approve final package", type="primary", key=_k(pid, "approve"),
                 disabled=busy or chk["blocking"] or (bool(chk["needs_ack"]) and not ack)):
        try:
            path = svc.approve_final(pid, ack)
            st.success(f"Approved. Manifest: {path.name}")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    return chk


def _upload(svc: StudioService, pid: str, chk: dict, busy: bool) -> None:
    st.subheader("Optional: upload to YouTube")
    st.caption("Uploads use the official YouTube Data API with the 'youtube.upload' permission (upload videos to your "
               "channel). The result is a video on your channel with the privacy you choose (private by default), "
               "not a YouTube Studio draft. Nothing is published unless you choose public. Untested from the dev PC.")
    yt = svc.youtube_status()
    if not yt["client_configured"]:
        st.info("To enable uploads, create an OAuth client (Desktop app) in Google Cloud, save its JSON outside the "
                "repository and set YOUTUBE_CLIENT_SECRETS_FILE in .env. Otherwise, upload the files manually.")
        return
    c1, c2 = st.columns(2)
    if not yt["connected"]:
        if c1.button("Connect YouTube account", key=_k(pid, "connect")):
            try:
                with st.spinner("Waiting for you to sign in in the browser window..."):
                    svc.youtube_connect()
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not connect: {e}")
        return
    if c2.button("Disconnect YouTube account", key=_k(pid, "disconnect")):
        svc.youtube_disconnect()
        st.rerun()
    if not chk["approved"]:
        st.info("Approve the final package first.")
        return
    pack = svc.creator_pack(pid)
    opts = {u["key"]: u["label"] for u in chk["uploadable"]}
    key = st.selectbox("Video", list(opts), format_func=opts.get, key=_k(pid, "video"))
    title = st.selectbox("Title", pack["titles"], key=_k(pid, "title"))
    privacy = st.radio("Privacy", list(youtube_draft.PRIVACY), index=0, horizontal=True, key=_k(pid, "privacy"))
    kids = st.radio("Is this video made for kids?", ["Choose…", "No", "Yes"], horizontal=True, key=_k(pid, "kids"))
    confirm_public = True
    if privacy != "private":
        st.warning(f"'{privacy}' makes the video visible to others.")
        confirm_public = st.checkbox(f"Yes, upload as {privacy}", key=_k(pid, "pub"))
    confirmed = st.checkbox(f"Upload '{opts[key]}' to my channel as {privacy}", key=_k(pid, "confirm"))
    if st.button("Upload", type="primary", key=_k(pid, "upload"),
                 disabled=busy or kids == "Choose…" or not confirmed or not confirm_public):
        try:
            svc.start_upload(pid, key, title, pack["description"], pack["tags"], privacy,
                             {"No": False, "Yes": True}.get(kids), confirmed, confirm_public)
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    for u in reversed(svc.uploads(pid)):
        if u.get("ok"):
            st.success(f"{u['file']}: uploaded as {u.get('result_privacy')} · {u['url']}")
        else:
            st.error(f"{u['file']}: upload failed: {u.get('error')}")
        for w in u.get("warnings", []):
            st.warning(w)


def render() -> None:
    svc = get_service()
    proj = require_project(svc)
    if proj is None:
        return
    st.title("Final approval & upload")
    reason = svc.longform_ready(proj["id"])
    if reason:
        st.info(reason)
        return
    pid = proj["id"]
    job_status_panel(pid, {}, "fr")
    busy = _busy(svc, pid)
    chk = _checklist(svc, pid, busy)
    st.caption(f"Approved files are in `{svc.files(pid).ready}`; you can always upload them manually.")
    _upload(svc, pid, chk, busy)
