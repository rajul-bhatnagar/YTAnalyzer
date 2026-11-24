"""Gameplay analysis & highlight review: provider setup and cloud consent, running the hybrid analysis, and
reviewing every candidate with its evidence, score breakdown, confidence and uncertainty."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from pipeline import analysis_stages, config, ffmpeg_tools, longform
from pipeline.ai_providers.base import EVENT_CATEGORIES, ProviderError
from pipeline.analysis_service import CONSENT_MODES, PROVIDERS, describe_window_statuses
from pipeline.project_service import StudioService
from ui.common import fmt_duration, get_service, job_status_panel, require_project

_STAGE_LABELS = {**longform.STAGE_LABELS, **analysis_stages.STAGE_LABELS}
STATUS_ICON = {"approved": "✅", "suggested": "💡", "rejected": "🚫"}
CONF_COLOR = {"high": "green", "medium": "orange", "low": "red"}


def _k(pid: str, name: str) -> str:
    return f"hl_{pid}_{name}"


def _ts(sec: float | None) -> str:
    if sec is None:
        return "-"
    m, s = divmod(max(float(sec), 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


def _busy(svc: StudioService, pid: str) -> bool:
    job = svc.latest_job(pid)
    return bool(job and svc.is_job_running(job["id"]))


def _category(cat: str) -> str:
    return EVENT_CATEGORIES.get(cat, cat.replace("_", " ").capitalize() + " (custom)")


# --- setup & run ----------------------------------------------------------------------------------------
def _provider_setup(svc: StudioService, pid: str, ai: dict[str, Any], busy: bool) -> dict[str, Any]:
    keys = list(PROVIDERS)
    provider = st.radio("AI provider", keys, index=keys.index(ai["provider"]) if ai["provider"] in keys else 1,
                        format_func=PROVIDERS.get, key=_k(pid, "provider"), horizontal=True)
    values: dict[str, Any] = {"ai.provider": provider}

    with st.expander("Local Ollama", expanded=provider == "ollama"):
        url = st.text_input("Ollama address", ai["ollama_base_url"], key=_k(pid, "ollama_url"))
        values["ai.ollama_base_url"] = url.strip() or "http://localhost:11434"
        try:
            models = svc.ollama_models(pid)
        except ProviderError as e:
            models = []
            st.warning(str(e))
        names = [m["name"] for m in models]
        if names:
            labels = {m["name"]: f"{m['name']} · {m.get('parameter_size') or '?'} · "
                                 f"{'vision' if 'vision' in m['capabilities'] else 'text-only'}" for m in models}
            cur = ai["ollama_model"] if ai["ollama_model"] in names else None
            model = st.selectbox("Model (installed models only; the app never downloads models)", names,
                                 index=names.index(cur) if cur else None, format_func=labels.get,
                                 placeholder="Choose a model", key=_k(pid, "ollama_model"))
            values["ai.ollama_model"] = model or ""
            if model and "vision" not in next(m for m in models if m["name"] == model)["capabilities"]:
                st.info("This model is text-only: it reads the transcript and measurements but cannot see frames. "
                        "For local visual understanding, install a vision-capable model yourself in Ollama.")
        else:
            values["ai.ollama_model"] = ai["ollama_model"]
            st.caption(f"Selected model: `{ai['ollama_model'] or 'none'}`")
        if st.button("Test Ollama", key=_k(pid, "test_ollama"), disabled=busy):
            svc.analysis_settings_changed(pid, values)
            res = svc.test_ai_provider(pid, "ollama")
            (st.success if res["ok"] else st.error)(res["message"])

    with st.expander("Gemini (cloud, free tier)", expanded=provider == "gemini"):
        has_key = config.secret_status().get("GEMINI_API_KEY", False)
        st.write(f"API key: {'✅ set in `.env`' if has_key else '⚪ not set: add `GEMINI_API_KEY` to `.env` and restart'}")
        model = st.text_input("Gemini model", ai["gemini_model"], key=_k(pid, "gemini_model"),
                              help="Validated against your key's model list by the connection test.")
        visual = st.checkbox("Send visual evidence (frames or video chunks)", bool(ai["gemini_visual_enabled"]),
                             key=_k(pid, "gemini_visual"),
                             help="Off = Gemini only reads the transcript and measurements.")
        modes = ["frames", "video"]
        gmode = st.radio("Visual input", modes, index=modes.index(ai["gemini_mode"]) if ai["gemini_mode"] in modes
                         else 0, format_func={"frames": "Sampled frames (default)",
                                               "video": "Low-res video chunks of the most active windows"}.get,
                         key=_k(pid, "gemini_mode"), horizontal=True, disabled=not visual)
        values |= {"ai.gemini_model": model.strip(), "ai.gemini_visual_enabled": bool(visual),
                   "ai.gemini_mode": gmode}
        if st.button("Test Gemini connection", key=_k(pid, "test_gemini"), disabled=busy or not has_key,
                     help="Uses a model-information request, not a content-generation request."):
            svc.analysis_settings_changed(pid, values)
            res = svc.test_ai_provider(pid, "gemini")
            (st.success if res["ok"] else st.error)(res["message"])
        _consent(svc, pid)

    with st.expander("Limits and quota behaviour"):
        q = {"fallback": "Switch to local Ollama (if available), else pause", "pause": "Pause the job"}
        on_quota = st.radio("When the free Gemini quota or rate limit is reached", list(q),
                            index=list(q).index(ai["on_quota"]) if ai["on_quota"] in q else 0, format_func=q.get,
                            key=_k(pid, "on_quota"))
        c1, c2 = st.columns(2)
        max_req = c1.number_input("Max AI requests per job", 1, 2000, int(ai["max_requests_per_job"]),
                                  key=_k(pid, "max_req"))
        interval = c2.number_input("Seconds between cloud requests", 0.0, 120.0, float(ai["min_request_interval_sec"]),
                                   1.0, key=_k(pid, "interval"))
        values |= {"ai.on_quota": on_quota, "ai.max_requests_per_job": int(max_req),
                   "ai.min_request_interval_sec": float(interval)}
    return values


def _consent(svc: StudioService, pid: str) -> None:
    c = svc.cloud_consent(pid)
    d = svc.cloud_disclosure(pid)
    st.markdown("**Cloud analysis consent for this project**")
    if c["mode"]:
        st.success(f"Approved on {c['at']}: {CONSENT_MODES[c['mode']]}.")
    else:
        st.info("Not approved: nothing from this project is sent to Google.")
    st.markdown(
        f"If approved, the app sends, for each of about **{d['windows']}** analysis windows: up to "
        f"**{d['frames_per_window_max']}** frames ({d['frame_width']} px wide; about {d['frames_max']} in total), "
        f"{d['transcript']}, and {d['signals']}. With video approval it also uploads up to "
        f"**{d['video_windows_max']}** low-resolution ({d['video_height']}p) clips of about "
        f"{d['video_chunk_sec']:.0f} s from the most active windows; uploaded clips are deleted after use. "
        f"The full-quality recording is never uploaded. At most {d['max_requests']} requests per job go to "
        f"`{d['model']}`.")
    st.warning("Per Google's pricing page (checked 2026-11-24), Gemini API **free-tier content is used to improve "
               "Google's products**. Free-tier limits are set by Google, are not guaranteed and can change. The app "
               "never enables billing, but it cannot see whether your key's Google Cloud project has billing "
               "enabled: use a key from a project **without billing** for a hard ₹0 guarantee.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Approve frames + transcript", key=_k(pid, "consent_frames")):
        svc.grant_cloud_consent(pid, "frames")
        st.rerun()
    if c2.button("Approve frames + video chunks", key=_k(pid, "consent_video")):
        svc.grant_cloud_consent(pid, "video")
        st.rerun()
    if c3.button("Revoke", key=_k(pid, "consent_revoke"), disabled=not c["mode"]):
        svc.revoke_cloud_consent(pid)
        st.rerun()


def _setup_tab(svc: StudioService, proj: dict[str, Any], busy: bool) -> None:
    pid = proj["id"]
    s = svc.effective_settings(pid)
    a, h = s["analysis"], s["highlights"]
    st.caption(f"Game: **{proj['game_name']}** · style: **{proj['content_style']}** (change on Processing & logs). "
               "The style changes what is emphasised, never the evidence standard.")
    values = _provider_setup(svc, pid, s["ai"], busy)
    with st.expander("Analysis parameters"):
        c1, c2, c3 = st.columns(3)
        values |= {
            "analysis.window_sec": int(c1.number_input("Window length (s)", 30, 600, int(a["window_sec"]), 15,
                                                       key=_k(pid, "win"))),
            "analysis.frame_interval_sec": float(c2.number_input("Frame every (s)", 1.0, 60.0,
                                                                 float(a["frame_interval_sec"]), 1.0, key=_k(pid, "fi"))),
            "analysis.max_frames_per_window": int(c3.number_input("Max frames per window", 4, 120,
                                                                  int(a["max_frames_per_window"]), key=_k(pid, "mf"))),
            "analysis.transient_min_db": float(c1.number_input("Game-audio jump threshold (dB)", 3.0, 40.0,
                                                               float(a["transient_min_db"]), 1.0, key=_k(pid, "tr"))),
            "analysis.voice_emphasis_min_db": float(c2.number_input("Voice emphasis threshold (dB)", 2.0, 30.0,
                                                                    float(a["voice_emphasis_min_db"]), 1.0,
                                                                    key=_k(pid, "ve"))),
            "analysis.scene_change_k": float(c3.number_input("Scene-change sensitivity (lower = more)", 2.0, 20.0,
                                                             float(a["scene_change_k"]), 0.5, key=_k(pid, "sc"))),
        }
    with st.expander("Shorts candidates", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        count = int(c1.number_input("How many", 1, 20, int(h["candidate_count"]), key=_k(pid, "count")))
        lo = int(c2.number_input("Min length (s)", 5, 120, int(h["min_sec"]), key=_k(pid, "min")))
        hi = int(c3.number_input("Max length (s)", 10, 180, int(h["max_sec"]), key=_k(pid, "max")))
        before = float(c4.number_input("Context before (s)", 0.0, 60.0, float(h["context_before_sec"]), 1.0,
                                       key=_k(pid, "before")))
        values |= {"highlights.candidate_count": count, "highlights.min_sec": lo, "highlights.max_sec": hi,
                   "highlights.context_before_sec": before}
        if lo >= hi:
            st.error("Min length must be shorter than max length.")
    b1, b2, b3 = st.columns(3)
    if b1.button("Save settings", key=_k(pid, "save")):
        svc.analysis_settings_changed(pid, values)
        st.success("Saved.")
    if b2.button("Save & analyse", type="primary", disabled=busy or lo >= hi, key=_k(pid, "run"),
                 help="Reuses audio extraction, signals, frames and finished AI windows whenever their inputs are "
                      "unchanged."):
        svc.analysis_settings_changed(pid, values)
        _start(svc, pid, "analysis")
    if b3.button("Regenerate candidates only", disabled=busy or lo >= hi, key=_k(pid, "regen"),
                 help="Re-ranks and rebuilds clips from the existing analysis. Approved, rejected and edited "
                      "highlights are kept."):
        svc.analysis_settings_changed(pid, values)
        _start(svc, pid, "candidates", count)
    _report(svc, pid)


def _start(svc: StudioService, pid: str, target: str, count: int | None = None) -> None:
    try:
        svc.start_analysis(pid, target, count)
        st.rerun()
    except ValueError as e:
        st.error(str(e))


def _report(svc: StudioService, pid: str) -> None:
    rep = svc.ai_report(pid)
    if not rep:
        return
    st.subheader("Last analysis")
    counts = describe_window_statuses(rep)
    eff = ", ".join(rep.get("effective") or []) or "none (signals only)"
    st.markdown(f"**AI actually used:** {eff} · windows: " +
                " · ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f" · requests: {rep.get('requests', 0)}")
    for n in [*rep.get("plan_notes", []), *rep.get("plan_events", []), *rep.get("cleanup_warnings", [])]:
        st.info(n)


# --- highlight cards --------------------------------------------------------------------------------------
def _evidence(svc: StudioService, h: dict[str, Any]) -> None:
    ev = h["evidence"] or {}
    if ev.get("ai"):
        a = ev["ai"]
        conf = a.get("confidence") or {}
        st.caption(f"AI: {a['provider']} `{a['model']}` ({a['mode']}) · confidence: visual {conf.get('visual')}, "
                   f"audio {conf.get('audio')}, transcript {conf.get('transcript')}")
    else:
        st.caption("No AI description: this moment was found from measurements only.")
    vis = [v for v in ev.get("visual", []) if svc.frame_path(v.get("path"))]
    if vis:
        cols = st.columns(min(len(vis), 4))
        for i, v in enumerate(vis[:4]):
            cols[i].image(str(svc.frame_path(v["path"])), caption=f"{_ts(v['t'])}" +
                          (f": {v['observation']}" if v.get("observation") and i == 0 else ""))
    for a in ev.get("audio", []):
        st.write(f"🔊 {_ts(a['start'])}–{_ts(a['end'])}: {a['note']}")
    for q in ev.get("transcript", []):
        st.write(f"💬 {_ts(q['start'])}: “{q['quote']}”")


def _breakdown(h: dict[str, Any]) -> None:
    b = h["score_breakdown"] or {}
    # One type per column (Arrow): the rating is text so "not judged" can sit next to numbers.
    rows = [{"factor": v["label"], "rating (0-5)": f"{v['value']:.1f}" if v["used"] else "not judged",
             "weight": float(v["weight"])} for v in (b.get("factors") or {}).values()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    note = f"Weighted for style '{b.get('style')}'. Unjudged factors are excluded, not counted as zero."
    if b.get("capped_signals_only"):
        note += f" Raw {b.get('raw')} was capped at 70 because no AI described this moment."
    st.caption(note)


def _card(svc: StudioService, h: dict[str, Any], dur: float, busy: bool) -> None:
    hid = h["id"]
    pid = h["project_id"]
    head = (f"{STATUS_ICON.get(h['status'], '')} **#{h['rank'] or '-'} · {_category(h['category'] or 'unclear')}** · "
            f"{h['score_band']} ({h['score']:.0f}) · {_ts(h['eff_start'])}–{_ts(h['eff_end'])} "
            f"({h['eff_end'] - h['eff_start']:.0f} s)")
    with st.container(border=True):
        st.markdown(head)
        st.badge(f"confidence: {h['confidence']}", color=CONF_COLOR.get(h["confidence"], "gray"))  # type: ignore[arg-type]
        left, right = st.columns([3, 2])
        with left:
            pkey = _k(pid, f"prev_{hid}")
            if st.button("▶ Preview clip", key=_k(pid, f"pbtn_{hid}"), disabled=busy):
                try:
                    with st.spinner("Cutting a 480p preview from the original recording..."):
                        st.session_state[pkey] = str(svc.highlight_preview(hid))
                except (ffmpeg_tools.FFmpegError, ValueError) as e:
                    st.error(str(e))
            p = st.session_state.get(pkey)
            if p:
                handles = float(svc.effective_settings(pid)["highlights"]["handle_sec"])
                pre = min(handles, h["eff_start"])
                st.video(p, start_time=int(pre), end_time=int(pre + h["eff_end"] - h["eff_start"]) + 1)
                st.caption(f"Plays the clip; the file also holds up to {handles:.0f} s before/after for trimming.")
            st.write(h["rationale"])
            if h["uncertainty"]:
                st.warning(h["uncertainty"])
            with st.expander("Evidence"):
                _evidence(svc, h)
            with st.expander("Score breakdown"):
                _breakdown(h)
        with right:
            c1, c2 = st.columns(2)
            s = c1.number_input("Start (s)", 0.0, dur, float(h["eff_start"]), 0.5, key=_k(pid, f"s_{hid}"))
            e = c2.number_input("End (s)", 0.0, dur, float(h["eff_end"]), 0.5, key=_k(pid, f"e_{hid}"))
            st.caption(f"Key moment at {_ts(h['event_sec'])}. AI suggestion: {_ts(h['start_sec'])}–{_ts(h['end_sec'])}.")
            hook = st.text_input("Hook caption", h["hook_caption"] or "", max_chars=80, key=_k(pid, f"hook_{hid}"),
                                 placeholder="Write what actually happens")
            title = st.text_input("Suggested title", h["suggested_title"] or "", max_chars=100,
                                  key=_k(pid, f"title_{hid}"), placeholder="Your title")
            notes = st.text_input("Notes", h["user_notes"] or "", key=_k(pid, f"notes_{hid}"))
            b1, b2, b3 = st.columns(3)
            for col, label, status in ((b1, "Save", None), (b2, "Approve", "approved"), (b3, "Reject", "rejected")):
                if col.button(label, key=_k(pid, f"{label}_{hid}"), type="primary" if status == "approved" else
                              "secondary", disabled=busy):
                    try:
                        moved = abs(s - h["eff_start"]) > 1e-6 or abs(e - h["eff_end"]) > 1e-6
                        res = svc.update_highlight(hid, start=s if moved else None, end=e if moved else None,
                                                   status=status, notes=notes, hook=hook, title=title)
                        if moved:  # the old preview no longer matches the clip
                            st.session_state.pop(_k(pid, f"prev_{hid}"), None)
                        st.session_state[_k(pid, f"warn_{hid}")] = res["warnings"]
                        st.rerun()
                    except ValueError as ex:
                        st.error(str(ex))
            for w in st.session_state.get(_k(pid, f"warn_{hid}"), []):
                st.warning(w)
            if h["user_start_sec"] is not None and st.button("Reset to suggested times", key=_k(pid, f"reset_{hid}")):
                svc.reset_highlight_times(hid)
                st.session_state.pop(_k(pid, f"prev_{hid}"), None)
                st.rerun()
            if h["status"] != "suggested" and st.button("Back to undecided", key=_k(pid, f"undo_{hid}")):
                svc.update_highlight(hid, status="suggested")
                st.rerun()


def _highlights_tab(svc: StudioService, proj: dict[str, Any], busy: bool) -> None:
    pid = proj["id"]
    hl = svc.highlights(pid)
    if not hl:
        st.info("No highlights yet. Run the analysis on the Setup tab.")
        return
    st.caption("Scores are ranking aids from the evidence shown, not objective truth. Approve the clips worth making "
               "into Shorts; nothing is published from here.")
    dur = float(proj["duration_sec"] or 0)
    for group, title in (("approved", "Approved"), ("suggested", "Suggestions"), ("rejected", "Rejected")):
        items = [h for h in hl if h["status"] == group]
        if not items:
            continue
        if group == "rejected":
            with st.expander(f"{title} ({len(items)})"):
                for h in items:
                    _card(svc, h, dur, busy)
        else:
            st.subheader(f"{title} ({len(items)})")
            for h in items:
                _card(svc, h, dur, busy)


def _events_tab(svc: StudioService, pid: str) -> None:
    ev = svc.analysis_events(pid)
    if not ev:
        st.info("No events yet.")
        return
    rows = [{"from": _ts(e["start_sec"]), "to": _ts(e["end_sec"]), "category": _category(e["event_type"]),
             "evidence from": e["source_modality"], "description": e["description"],
             "uncertainty": e["uncertainty"] or ""} for e in ev]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Every detected moment, including ones not chosen as candidates. 'ai' means an AI model described it; "
               "audio/visual/commentary are measured signals.")


def render() -> None:
    svc = get_service()
    proj = require_project(svc)
    if proj is None:
        return
    st.title("Gameplay analysis & highlights")
    reason = svc.analysis_ready(proj["id"])
    if reason:
        st.info(reason)
        return
    job_status_panel(proj["id"], _STAGE_LABELS, "hl")
    busy = _busy(svc, proj["id"])
    st.caption(f"Recording length {fmt_duration(proj['duration_sec'])}.")
    tabs = st.tabs(["⚙️ Setup & run", "🎯 Highlights", "📋 All events"])
    with tabs[0]:
        _setup_tab(svc, proj, busy)
    with tabs[1]:
        _highlights_tab(svc, proj, busy)
    with tabs[2]:
        _events_tab(svc, proj["id"])
