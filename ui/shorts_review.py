"""Shorts review: per-Short layout/crop/trim/caption settings with framing previews, rendering only what changed,
validation results and approve/reject."""

from __future__ import annotations

from typing import Any

import streamlit as st

from pipeline import captions, content_stages, ffmpeg_tools, longform, timeline_builder
from pipeline.content_service import SHORT_STATUSES
from pipeline.project_service import StudioService
from pipeline.shorts_builder import LAYOUTS, ShortSettings
from ui.common import fmt_size, get_service, job_status_panel, require_project

STATUS_ICON = {"pending": "⏳", "approved": "✅", "rejected": "🚫"}
HOOK_MODES = {"no_speech": "Only when nobody speaks in the clip", "always": "Always, for the first seconds",
              "never": "Never"}


def _k(pid: str, name: str) -> str:
    return f"sh_{pid}_{name}"


def _ts(sec: float) -> str:
    m, s = divmod(max(float(sec), 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


def _busy(svc: StudioService, pid: str) -> bool:
    job = svc.latest_job(pid)
    return bool(job and svc.is_job_running(job["id"]))


def _settings_form(svc: StudioService, pid: str, h: dict[str, Any], s: ShortSettings) -> ShortSettings:
    hid = h["id"]
    keys = list(LAYOUTS)
    layout = st.radio("Layout", keys, index=keys.index(s.layout), format_func=LAYOUTS.get, key=_k(pid, f"lay_{hid}"))
    upd: dict[str, Any] = {"layout": layout}
    if layout == "blur_fit":
        upd["fit_y"] = st.slider("Frame position (top → bottom)", 0.0, 1.0, s.fit_y, 0.05, key=_k(pid, f"fy_{hid}"))
    if layout in ("center_crop", "split_stack"):
        upd["crop_x"] = st.slider("Crop position (left → right)" if layout == "center_crop" else
                                  "Top half: crop position (left → right)", 0.0, 1.0, s.crop_x, 0.05,
                                  key=_k(pid, f"cx_{hid}"))
    if layout == "split_stack":
        bm = st.radio("Bottom half", ["full", "region"], index=["full", "region"].index(s.bottom_mode),
                      format_func={"full": "Whole frame (all HUD visible)", "region": "A chosen region (e.g. a "
                                   "scoreboard)"}.get, key=_k(pid, f"bm_{hid}"), horizontal=True)
        upd["bottom_mode"] = bm
        if bm == "region":
            c1, c2, c3 = st.columns(3)
            upd["region_x"] = c1.slider("Region left → right", 0.0, 1.0, s.region_x, 0.05, key=_k(pid, f"rx_{hid}"))
            upd["region_y"] = c2.slider("Region top → bottom", 0.0, 1.0, s.region_y, 0.05, key=_k(pid, f"ry_{hid}"))
            upd["region_w"] = c3.slider("Region width", 0.1, 1.0, s.region_w, 0.05, key=_k(pid, f"rw_{hid}"))
    presets = list(timeline_builder.QUALITY_PRESETS)
    upd["quality"] = st.selectbox("Quality", presets, index=presets.index(s.quality) if s.quality in presets else 0,
                                  format_func=lambda p: timeline_builder.QUALITY_PRESETS[p]["label"],
                                  key=_k(pid, f"q_{hid}"))
    with st.expander("Captions"):
        c = s.captions
        en = st.checkbox("Word-by-word captions (needs a transcript)", c.enabled, key=_k(pid, f"ce_{hid}"))
        pos_keys = list(captions.POSITIONS)
        c1, c2 = st.columns(2)
        pos = c1.selectbox("Position", pos_keys, index=pos_keys.index(c.position), format_func=captions.POSITIONS.get,
                           key=_k(pid, f"cp_{hid}"))
        size = c2.slider("Size", 24, 160, c.size, 2, key=_k(pid, f"cs_{hid}"))
        c3, c4, c5 = st.columns(3)
        tc = c3.color_picker("Text", c.text_color, key=_k(pid, f"ct_{hid}"))
        ec = c4.color_picker("Spoken word", c.emphasis_color, key=_k(pid, f"cm_{hid}"))
        oc = c5.color_picker("Outline", c.outline_color, key=_k(pid, f"co_{hid}"))
        c6, c7 = st.columns(2)
        box = c6.checkbox("Dark box behind text", c.background_box, key=_k(pid, f"cb_{hid}"))
        upper = c7.checkbox("UPPERCASE", c.uppercase, key=_k(pid, f"cu_{hid}"))
        margin = st.slider("Distance from the bottom (px of 1920)", 0, 1200, c.margin_bottom, 20,
                           key=_k(pid, f"mb_{hid}"),
                           help="Shorts UI covers part of the bottom; the exact area is set by YouTube.")
        hm_keys = list(HOOK_MODES)
        hm = st.selectbox("Hook caption", hm_keys, index=hm_keys.index(c.hook_mode) if c.hook_mode in hm_keys else 0,
                          format_func=HOOK_MODES.get, key=_k(pid, f"hm_{hid}"))
        upd["captions"] = c.model_copy(update={"enabled": en, "position": pos, "size": size, "text_color": tc,
                                               "emphasis_color": ec, "outline_color": oc, "background_box": box,
                                               "uppercase": upper, "margin_bottom": margin, "hook_mode": hm,
                                               "hook_enabled": hm != "never"})
    return s.model_copy(update=upd)


def _card(svc: StudioService, pid: str, h: dict[str, Any], dur: float, busy: bool) -> None:
    hid = h["id"]
    review = svc.short_review(hid)
    out = svc.short_output(hid)
    title = h["suggested_title"] or h["hook_caption"] or f"Highlight #{hid}"
    with st.container(border=True):
        st.markdown(f"{STATUS_ICON[review['status']]} **{title}** · {_ts(h['eff_start'])}–{_ts(h['eff_end'])} "
                    f"({h['eff_end'] - h['eff_start']:.0f} s)")
        left, right = st.columns([3, 2])
        with left:
            c1, c2 = st.columns(2)
            start = c1.number_input("Start (s)", 0.0, dur, float(h["eff_start"]), 0.5, key=_k(pid, f"s_{hid}"))
            end = c2.number_input("End (s)", 0.0, dur, float(h["eff_end"]), 0.5, key=_k(pid, f"e_{hid}"))
            hook = st.text_input("Hook caption", h["hook_caption"] or "", max_chars=80, key=_k(pid, f"hook_{hid}"))
            new = _settings_form(svc, pid, h, svc.short_settings(hid))
            at = st.slider("Preview moment (s into the clip)", 0.0, max(float(h["eff_end"] - h["eff_start"]) - 0.1, 0.0),
                           min(2.0, float(h["eff_end"] - h["eff_start"]) / 2), 0.5, key=_k(pid, f"at_{hid}"))
            b1, b2, b3 = st.columns(3)
            if b1.button("Preview framing", key=_k(pid, f"pv_{hid}"), disabled=busy):
                try:
                    with st.spinner("Rendering one frame..."):
                        st.session_state[_k(pid, f"img_{hid}")] = str(svc.short_preview_frame(hid, at, new))
                except (ValueError, ffmpeg_tools.FFmpegError) as e:
                    st.error(str(e))
            changed_clip = abs(start - h["eff_start"]) > 1e-6 or abs(end - h["eff_end"]) > 1e-6 or \
                hook != (h["hook_caption"] or "")
            if b2.button("Save", key=_k(pid, f"save_{hid}"), disabled=busy):
                _save(svc, h, new, start, end, hook, changed_clip, render=False)
            if b3.button("Save & render", type="primary", key=_k(pid, f"render_{hid}"), disabled=busy,
                         help="Renders this Short only; unchanged Shorts are not re-rendered."):
                _save(svc, h, new, start, end, hook, changed_clip, render=True)
            img = st.session_state.get(_k(pid, f"img_{hid}"))
            if img:
                st.image(img, caption="Framing preview (1080×1920)", width=270)
        with right:
            if out is None:
                st.info("Not rendered yet.")
            else:
                ok = out["validation_status"] == "passed"
                errs = (out["validation_details"] or {}).get("errors") or []
                (st.success if ok else st.error)(f"Validation {out['validation_status']}" +
                                                 ("" if ok else ": " + "; ".join(errs)))
                if out["file"]:
                    st.video(str(out["file"]))
                    st.caption(f"{fmt_size(out['file'].stat().st_size)} · {out['settings'].get('layout')} · "
                               f"{out['settings'].get('encoder')} · {out['created_at']}")
            notes = st.text_input("Notes", review["notes"], key=_k(pid, f"notes_{hid}"))
            a1, a2, a3 = st.columns(3)
            for col, label, status in ((a1, "Approve", "approved"), (a2, "Reject", "rejected"),
                                       (a3, "Undecided", "pending")):
                if col.button(label, key=_k(pid, f"{status}_{hid}"), disabled=busy or review["status"] == status,
                              type="primary" if status == "approved" else "secondary"):
                    try:
                        svc.set_short_status(hid, status, notes)
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))


def _save(svc: StudioService, h: dict[str, Any], new: ShortSettings, start: float, end: float, hook: str,
          changed_clip: bool, render: bool) -> None:
    try:
        if changed_clip:
            res = svc.update_highlight(h["id"], start=start, end=end, hook=hook)
            for w in res["warnings"]:
                st.warning(w)
        svc.save_short_settings(h["id"], new)
        if render:
            svc.start_content(h["project_id"], "shorts", highlight_ids=[h["id"]])
        st.rerun()
    except ValueError as e:
        st.error(str(e))


def render() -> None:
    svc = get_service()
    proj = require_project(svc)
    if proj is None:
        return
    st.title("Shorts")
    reason = svc.longform_ready(proj["id"])
    if reason:
        st.info(reason)
        return
    pid = proj["id"]
    job_status_panel(pid, {**longform.STAGE_LABELS}, "sh", include_content=True)
    busy = _busy(svc, pid)
    hs = content_stages.approved_highlights(svc.db, pid)
    if not hs:
        st.info("No approved highlights yet. Approve candidates on **Analysis & highlights**; each approved highlight "
                "becomes a Short here.")
        return
    st.caption("Shorts are cut from the original recording with your selected audio mix, at 1080×1920. Nothing is "
               "stretched. Only Shorts whose settings, times or captions changed are re-rendered.")
    if st.button(f"Render all {len(hs)} approved Short(s)", type="primary", disabled=busy, key=_k(pid, "all")):
        try:
            svc.start_content(pid, "shorts")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    counts = {s: 0 for s in SHORT_STATUSES}
    for h in hs:
        counts[svc.short_review(h["id"])["status"]] += 1
    st.caption(f"Review: {counts['approved']} approved · {counts['pending']} undecided · {counts['rejected']} rejected")
    dur = float(proj["duration_sec"] or 0)
    for h in hs:
        _card(svc, pid, h, dur, busy)
