"""Shorts review: per-Short layout/crop/trim/caption settings with framing previews, rendering only what changed,
validation results and approve/reject."""

from __future__ import annotations

from typing import Any

import streamlit as st

from pipeline import captions, content_stages, ffmpeg_tools, longform, style_profiles, timeline_builder
from pipeline.content_service import SHORT_STATUSES
from pipeline.project_service import StudioService
from pipeline.shorts_builder import LAYOUTS, ShortSettings
from ui import voiceover_panel
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
        if bm == "full":
            st.warning("With one recording (no facecam), the bottom half repeats the whole frame. Better: pick a "
                       "region (e.g. a score strip, HUD or minimap), or use 'Full frame over blurred background'.")
        if bm == "region":
            c1, c2 = st.columns(2)
            upd["region_x"] = c1.slider("Region left → right", 0.0, 1.0, s.region_x, 0.05, key=_k(pid, f"rx_{hid}"))
            upd["region_y"] = c2.slider("Region top → bottom", 0.0, 1.0, s.region_y, 0.05, key=_k(pid, f"ry_{hid}"))
            upd["region_w"] = c1.slider("Region width", 0.1, 1.0, s.region_w, 0.05, key=_k(pid, f"rw_{hid}"))
            upd["region_h"] = c2.slider("Region height", 0.05, 1.0, float(s.region_h if s.region_h is not None
                                                                          else 0.25), 0.01, key=_k(pid, f"rh_{hid}"),
                                        help="Any shape: a thin full-width strip suits a score bar.")
            auto = st.checkbox("Size the bottom panel to the region automatically", s.split_ratio is None,
                               key=_k(pid, f"auto_{hid}"))
            upd["split_ratio"] = None if auto else st.slider(
                "Top panel share of the height", 0.3, 0.9, float(s.split_ratio or 0.5), 0.05, key=_k(pid, f"sr_{hid}"))
    presets = list(timeline_builder.QUALITY_PRESETS)
    upd["quality"] = st.selectbox("Quality", presets, index=presets.index(s.quality) if s.quality in presets else 0,
                                  format_func=lambda p: timeline_builder.QUALITY_PRESETS[p]["label"],
                                  key=_k(pid, f"q_{hid}"))
    prof = svc.style_profile(pid)
    dm_keys = ["", "trim", "speedup", "none"]
    dm_labels = {"": f"From style profile ({prof.shorts_dead_air_mode})", "trim": "Trim static lead-in and tail",
                 "speedup": "Speed up static lead-in, trim tail", "none": "Leave as recorded"}
    upd["dead_air"] = st.selectbox("Dead air", dm_keys, index=dm_keys.index(s.dead_air) if s.dead_air in dm_keys
                                   else 0, format_func=dm_labels.get, key=_k(pid, f"dead_{hid}"),
                                   help="Only this Short changes; the long video never does.")
    if (upd["dead_air"] or prof.shorts_dead_air_mode) == "speedup":
        upd["speedup_factor"] = st.slider("Speed-up", 2.0, 2.5, float(s.speedup_factor), 0.1, key=_k(pid, f"spd_{hid}"))
    if s.voice_file:
        v1, v2 = st.columns(2)
        upd["voice_offset"] = v1.number_input("Voiceover starts at (s into the Short)", 0.0, 600.0,
                                              float(s.voice_offset), 0.5, key=_k(pid, f"vof_{hid}"))
        upd["voice_gain_db"] = v2.slider("Voiceover level (dB)", -20.0, 20.0, float(s.voice_gain_db), 0.5,
                                         key=_k(pid, f"vg_{hid}"))
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
                        st.session_state[_k(pid, f"img_{hid}")] = str(svc.short_preview_frame(
                            hid, at, new, safe_zone=st.session_state.get(_k(pid, f"safe_{hid}"), True)))
                except (ValueError, ffmpeg_tools.FFmpegError) as e:
                    st.error(str(e))
            changed_clip = abs(start - h["eff_start"]) > 1e-6 or abs(end - h["eff_end"]) > 1e-6 or \
                hook != (h["hook_caption"] or "")
            if b2.button("Save", key=_k(pid, f"save_{hid}"), disabled=busy):
                _save(svc, h, new, start, end, hook, changed_clip, render=False)
            if b3.button("Save & render", type="primary", key=_k(pid, f"render_{hid}"), disabled=busy,
                         help="Renders this Short only; unchanged Shorts are not re-rendered."):
                _save(svc, h, new, start, end, hook, changed_clip, render=True)
            st.checkbox("Show the Shorts player's UI areas (bottom 22%, right 12%; approximate)", True,
                        key=_k(pid, f"safe_{hid}"))
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
            fc = svc.short_fact_check(hid)
            if fc["ok"]:
                st.caption("✅ Hook and title match the evidence inside this clip.")
            for m in fc["messages"]:
                st.warning(m)
            for n, sug in enumerate(fc["suggestions"] if not fc["ok"] else []):
                if st.button(f"Use hook: {sug}", key=_k(pid, f"sug_{hid}_{n}"), disabled=busy):
                    svc.update_highlight(hid, hook=sug)
                    st.rerun()
            with st.expander("Voiceover for this Short"):
                cur = new.voice_file
                if cur:
                    st.caption(f"Using: {cur.rsplit('/', 1)[-1]}")
                vf = st.file_uploader("Upload", type=["wav", "mp3", "m4a", "flac", "ogg", "opus", "aac"],
                                      key=_k(pid, f"svo_{hid}"), disabled=busy)
                if vf is not None and st.button("Use for this Short", key=_k(pid, f"svo_use_{hid}"), disabled=busy):
                    try:
                        svc.set_short_voiceover(hid, vf, vf.name)
                        st.rerun()
                    except (ValueError, ffmpeg_tools.FFmpegError) as e:
                        st.error(str(e))
                if cur and st.button("Remove (use the project mix)", key=_k(pid, f"svo_rm_{hid}"), disabled=busy):
                    svc.clear_short_voiceover(hid)
                    st.rerun()
                st.caption("Rendering mixes it over this clip's game sound (ducked), normalises loudness and "
                           "transcribes it for captions. Adjust its start and level below, then Save & render.")
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


def _style_panel(svc: StudioService, pid: str, busy: bool) -> None:
    proj = svc.get_project(pid)
    prof = svc.style_profile(pid)
    s = svc.effective_settings(pid)
    with st.expander(f"Style profile: {proj['content_style']}"):
        st.caption(prof.note + " These are defaults for NEW Shorts; each Short can still be changed below.")
        lay_keys = ["", *LAYOUTS]
        lay = st.selectbox("Default layout", lay_keys, index=lay_keys.index(s["shorts"]["layout"])
                           if s["shorts"]["layout"] in lay_keys else 0,
                           format_func=lambda k: f"From profile ({LAYOUTS[prof.default_shorts_layout]})" if k == ""
                           else LAYOUTS[k], key=_k(pid, "sp_lay"))
        dm_keys = ["", *style_profiles.DEAD_AIR_MODES]
        dm = st.selectbox("Default dead-air handling", dm_keys, index=dm_keys.index(s["shorts"]["dead_air_mode"])
                          if s["shorts"]["dead_air_mode"] in dm_keys else 0,
                          format_func=lambda k: f"From profile ({prof.shorts_dead_air_mode})" if k == ""
                          else style_profiles.DEAD_AIR_MODES[k], key=_k(pid, "sp_dm"))
        own = s["analysis"]["hud_ocr_weight"] is not None
        use_own = st.checkbox(f"Override HUD-text weight (profile: {prof.hud_ocr_weight:g})", own, key=_k(pid, "sp_w_on"))
        w = st.slider("HUD-text weight (0 = off)", 0.0, 1.0, float(prof.hud_ocr_weight), 0.1, key=_k(pid, "sp_w"),
                      disabled=not use_own)
        if st.button("Save style overrides", key=_k(pid, "sp_save"), disabled=busy):
            svc.save_project_settings(pid, {"shorts.layout": lay, "shorts.dead_air_mode": dm,
                                            "analysis.hud_ocr_weight": float(w) if use_own else None})
            st.rerun()


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
    _style_panel(svc, pid, busy)
    voiceover_panel.render(svc, pid, "sh_vo", busy)
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
