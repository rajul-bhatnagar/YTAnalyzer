from __future__ import annotations

import io
import logging

import pytest

from pipeline import hardware
from pipeline.importer import ImportErrorUser, guess_game, save_upload, scan_raw_input
from pipeline.logging_setup import ContextAndRedactFilter, redact
from pipeline.manifest import fingerprint, manifest_entry, quick_fingerprint, sha256_file, verify_entry
from pipeline.models import HardwareMode
from pipeline.validators import ValidationError, clamp_range

EXTS = [".mp4", ".mkv"]


# --- logging redaction ---------------------------------------------------------------------
def test_redact_patterns(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "my-very-secret-value")
    fake_key = "AI" + "za" + "SyA1234567890" + "abcdefghijklmnopqrstuv"  # built in pieces: not a real key
    text = (f"key {fake_key} used; api_key=abc123 "
            "token: \"tok\" Authorization: Bearer eyJhbGciOi.payload; env my-very-secret-value")
    out = redact(text)
    for secret in ("AIzaSyA1234567890", "abc123", "eyJhbGciOi", "my-very-secret-value"):
        assert secret not in out


def test_log_filter_adds_context_and_redacts():
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "using %s", ("api_key=zzz999",), None)
    assert ContextAndRedactFilter().filter(rec)
    assert rec.project == "-" and "zzz999" not in rec.getMessage()


# --- manifest ------------------------------------------------------------------------------
def test_checksums_and_manifest(tmp_path):
    f = tmp_path / "a b ü.bin"
    f.write_bytes(b"0" * 10_000_000)
    q1 = quick_fingerprint(f)
    assert len(sha256_file(f)) == 64
    e = manifest_entry(f, tmp_path, "stage", {"ok": True})
    assert e["path"] == "a b ü.bin" and e["size_bytes"] == 10_000_000
    assert verify_entry(e, tmp_path, full=True)
    with f.open("r+b") as h:  # change a byte in the middle chunk
        h.seek(5_000_000)
        h.write(b"1")
    assert quick_fingerprint(f) != q1
    assert verify_entry(e, tmp_path) and not verify_entry(e, tmp_path, full=True)
    f.unlink()
    assert not verify_entry(e, tmp_path)
    assert not verify_entry({**e, "validation": {"ok": False}}, tmp_path)


def test_fingerprint_is_order_independent_and_settings_sensitive():
    a = fingerprint("s", 1, ["x", "y"], {"k": 1, "j": 2})
    assert a == fingerprint("s", 1, ["y", "x"], {"j": 2, "k": 1})
    assert a != fingerprint("s", 1, ["x", "y"], {"k": 2, "j": 2})
    assert a != fingerprint("s", 2, ["x", "y"], {"k": 1, "j": 2})


# --- validators ----------------------------------------------------------------------------
def test_clamp_range():
    assert clamp_range(-5, 20, 10) == (0.0, 10.0)
    assert clamp_range(8, 3, 10) == (3.0, 8.0)
    with pytest.raises(ValidationError):
        clamp_range(12, 15, 10, min_len=1)
    with pytest.raises(ValidationError):
        clamp_range(0, 1, 0)


# --- importer ------------------------------------------------------------------------------
def test_scan_nested_folders_and_game_guess(studio_paths):
    raw = studio_paths.raw_input
    (raw / "Cricket 24" / "day 1").mkdir(parents=True)
    (raw / "Cricket 24" / "day 1" / "match ü.mkv").write_bytes(b"x")
    (raw / "loose.mp4").write_bytes(b"xx")
    (raw / "notes.txt").write_text("no")
    (raw / "partial.mp4.partial").write_bytes(b"x")
    (raw / ".gitkeep").write_text("")
    found = scan_raw_input(studio_paths, EXTS)
    assert [f.rel for f in found] == ["Cricket 24/day 1/match ü.mkv", "loose.mp4"]
    assert found[0].game_guess == "Cricket 24" and found[1].game_guess is None
    assert guess_game(raw, studio_paths.root / "x.mp4") is None


def test_save_upload_never_overwrites(studio_paths):
    data = b"v" * 1024
    p1 = save_upload(studio_paths, io.BytesIO(data), "rec.mp4", "My Game", len(data), EXTS, 10, 0)
    p2 = save_upload(studio_paths, io.BytesIO(data), "rec.mp4", "My Game", len(data), EXTS, 10, 0)
    assert p1 != p2 and p1.read_bytes() == data and p2.name == "rec_1.mp4"
    assert p1.parent == studio_paths.raw_input / "My Game"


def test_save_upload_rejections_leave_no_partial(studio_paths):
    with pytest.raises(ImportErrorUser, match="Unsupported"):
        save_upload(studio_paths, io.BytesIO(b"x"), "a.exe", "G", 1, EXTS, 10, 0)
    with pytest.raises(ImportErrorUser, match="upload limit"):
        save_upload(studio_paths, io.BytesIO(b"x"), "a.mp4", "G", 11 * 1024 * 1024, EXTS, 10, 0)
    with pytest.raises(ImportErrorUser, match="disk space"):
        save_upload(studio_paths, io.BytesIO(b"x"), "a.mp4", "G", 1, EXTS, 10, 10**9)
    with pytest.raises(ImportErrorUser, match="incomplete"):
        save_upload(studio_paths, io.BytesIO(b"x"), "b.mp4", "G", 5, EXTS, 10, 0)
    assert not list(studio_paths.raw_input.rglob("*.partial"))


# --- hardware decisions ----------------------------------------------------------------------
def test_parse_nvidia_smi():
    gpus = hardware.parse_nvidia_smi("NVIDIA GeForce RTX 4050 Laptop GPU, 560.94, 6141, 5800\n")
    assert gpus[0].name.startswith("NVIDIA GeForce RTX 4050") and gpus[0].memory_total_mb == 6141


@pytest.mark.parametrize("smi,nvenc,cuda,mode", [
    (False, True, "passed", HardwareMode.CPU),
    (True, True, "passed", HardwareMode.NVIDIA_CUDA_AND_NVENC),
    (True, True, "not_run", HardwareMode.NVIDIA_NVENC_ONLY),   # CUDA device count alone is not proof
    (True, False, "passed", HardwareMode.NVIDIA_CUDA_ONLY),
    (True, False, "failed", HardwareMode.GPU_DETECTED_BUT_UNAVAILABLE),
])
def test_decide_mode(smi, nvenc, cuda, mode):
    r = hardware.HardwareReport(nvidia_smi_found=smi, nvenc_encode_ok=nvenc, whisper_cuda_smoke=cuda,
                                ct2_cuda_devices=1)
    assert hardware.decide_mode(r) == mode


def test_report_roundtrip(tmp_path):
    r = hardware.HardwareReport(gpus=[hardware.GpuInfo("g", "1", 6000, 5000)], notes=["n"])
    hardware.save_report(r, tmp_path / "hw.json")
    back = hardware.load_report(tmp_path / "hw.json")
    assert back and back.gpus[0].memory_total_mb == 6000 and back.notes == ["n"]
    (tmp_path / "bad.json").write_text("{")
    assert hardware.load_report(tmp_path / "bad.json") is None


def test_cuda_test_result_survives_a_check_without_smoke():
    g = hardware.GpuInfo(name="NVIDIA GeForce RTX 4050 Laptop GPU", driver_version="617.42", memory_total_mb=6141,
                         memory_free_mb=5900)
    prev = hardware.HardwareReport(nvidia_smi_found=True, gpus=[g], nvenc_encode_ok=True, ct2_version="4.9.1",
                                   ct2_cuda_devices=1, whisper_cuda_smoke="passed")
    new = hardware.HardwareReport(nvidia_smi_found=True, gpus=[g], nvenc_encode_ok=True, ct2_version="4.9.1",
                                  ct2_cuda_devices=1, notes=["CUDA device visible to CTranslate2 but not yet verified."])
    r = hardware.carry_over_cuda_test(new, prev)
    assert r.whisper_cuda_smoke == "passed" and r.mode == "NVIDIA_CUDA_AND_NVENC" and len(r.notes) == 1
    g2 = hardware.GpuInfo(name=g.name, driver_version="620.00", memory_total_mb=6141, memory_free_mb=5900)
    other = hardware.HardwareReport(nvidia_smi_found=True, gpus=[g2], ct2_version="4.9.1", ct2_cuda_devices=1)
    assert hardware.carry_over_cuda_test(other, prev).whisper_cuda_smoke == "not_run"  # new driver -> retest
    failed = hardware.HardwareReport(nvidia_smi_found=True, gpus=[g], ct2_version="4.9.1", ct2_cuda_devices=1,
                                     whisper_cuda_smoke="failed")
    assert hardware.carry_over_cuda_test(failed, prev).whisper_cuda_smoke == "failed"  # a real new result wins
