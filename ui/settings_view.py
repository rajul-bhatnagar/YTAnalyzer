"""Settings, hardware diagnostics, secrets status and logs."""

from __future__ import annotations

import streamlit as st

from pipeline import config
from pipeline.hardware import MODE_BADGES
from pipeline.paths import free_disk_gb
from ui.common import fmt_size, get_service


def _hardware(svc) -> None:
    st.subheader("Hardware capabilities")
    st.caption("Each capability is proven by a real test. A detected GPU is not used until its test passes.")
    report = svc.hardware_report()
    cuda_visible = bool(report and report.ct2_cuda_devices > 0)
    smoke = st.checkbox(
        "Also run the CUDA Whisper test (downloads the ~75 MB 'tiny' model from Hugging Face on first run)",
        value=False, disabled=not cuda_visible,
        help="Only available when CTranslate2 sees a CUDA device. Required before Whisper uses the GPU.")
    if st.button("Run capability check", type="primary"):
        with st.spinner("Testing FFmpeg, NVENC, CUDA and ONNX Runtime..."):
            report = svc.refresh_hardware(run_whisper_smoke=smoke)
        st.success("Check finished.")
    if report is None:
        st.info("Not checked yet.")
        return
    label, color = MODE_BADGES.get(report.mode, (report.mode, "gray"))
    st.badge(f"Mode: {label}", color=color)  # type: ignore[arg-type]
    rows = [
        ("NVIDIA GPU (nvidia-smi)", ", ".join(f"{g.name} ({g.memory_free_mb}/{g.memory_total_mb} MB free)"
                                             for g in report.gpus) or "not found"),
        ("FFmpeg", report.ffmpeg_version or "not found"),
        ("NVENC listed by FFmpeg", "yes" if report.nvenc_listed else "no"),
        ("NVENC real encode test", "passed" if report.nvenc_encode_ok else f"not usable ({report.nvenc_error or 'not run'})"),
        ("CTranslate2 CUDA devices", f"{report.ct2_cuda_devices} (ctranslate2 {report.ct2_version})"),
        ("CUDA Whisper test", f"{report.whisper_cuda_smoke}" + (f": {report.whisper_cuda_error}" if report.whisper_cuda_error else "")),
        ("ONNX Runtime", f"{'passed' if report.onnx_smoke_ok else 'failed'} · providers {', '.join(report.onnx_providers)}"),
        ("Python", report.python_version),
    ]
    st.table({"Check": [r[0] for r in rows], "Result": [r[1] for r in rows]})
    for n in report.notes:
        st.info(n)


def _settings(svc) -> None:
    st.subheader("Import & inspection settings")
    eff = svc.effective_settings()
    imp, probe = eff["import"], eff["probe"]
    with st.form("settings_form"):
        max_mb = st.number_input("Max browser upload (MB)", 100, 2048, int(imp["max_upload_mb"]), step=100,
                                 help="Hard ceiling 2048 MB (set in .streamlit/config.toml).")
        min_free = st.number_input("Minimum free disk space to keep (GB)", 1.0, 500.0, float(imp["min_free_disk_gb"]))
        full = st.checkbox("Compute full SHA-256 of each source (slower on huge files, stronger duplicate check)",
                           value=bool(imp["full_checksum"]))
        vfr_s = st.number_input("Seconds sampled for variable-frame-rate check", 5, 600, int(probe["vfr_sample_seconds"]))
        prev_s = st.number_input("Track preview length (s)", 5, 120, int(probe["track_preview_seconds"]))
        if st.form_submit_button("Save settings"):
            svc.settings.set_many({
                "import.max_upload_mb": int(max_mb), "import.min_free_disk_gb": float(min_free),
                "import.full_checksum": bool(full), "probe.vfr_sample_seconds": int(vfr_s),
                "probe.track_preview_seconds": int(prev_s),
            })
            st.success("Saved. Changed settings invalidate only the stages that depend on them.")
    st.caption(f"Free disk space at the studio folder: {free_disk_gb(svc.paths.root):.1f} GB")


def _housekeeping(svc) -> None:
    st.subheader("Disk & temporary files")
    d = svc.disk_status()
    if d["message"]:
        (st.error if d["level"] == "block" else st.warning)(d["message"])
    else:
        st.caption(f"{d['free_gb']} GB free.")
    items = svc.temp_files()
    if not items:
        st.caption("No temporary files to clean.")
        return
    labels = {str(i.path): f"{i.kind} · {fmt_size(i.size)} · {i.path.relative_to(svc.paths.root)}" for i in items}
    chosen = st.multiselect("Temporary files (all can be regenerated; sources and outputs are never listed)",
                            list(labels), default=list(labels), format_func=labels.get)
    total = sum(i.size for i in items if str(i.path) in chosen)
    sure = st.checkbox(f"Delete the selected items ({fmt_size(total)})")
    if st.button("Delete temporary files", disabled=not chosen or not sure):
        try:
            freed = svc.clean_temp([i for i in items if str(i.path) in chosen])
            st.success(f"Freed {fmt_size(freed)}.")
            st.rerun()
        except ValueError as e:
            st.error(str(e))


def _secrets() -> None:
    st.subheader("Credentials (.env)")
    st.caption("Values are never shown, logged or stored in the database. Edit `.env` in the studio folder "
               "and restart the app.")
    for k, ok in config.secret_status().items():
        st.write(f"{'✅' if ok else '⚪'} `{k}`: {'set' if ok else 'not set'}")


def render() -> None:
    svc = get_service()
    st.title("Settings & diagnostics")
    _hardware(svc)
    st.divider()
    _settings(svc)
    st.divider()
    _housekeeping(svc)
    st.divider()
    _secrets()
    st.divider()
    st.subheader("Software versions")
    st.json(svc.software_versions(), expanded=False)
    st.subheader("Log file")
    st.caption(f"`{svc.log_file}`")
    st.code(svc.log_tail(300) or "(empty)", language="log")
