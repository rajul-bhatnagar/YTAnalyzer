"""Local transcription with faster-whisper: word timestamps, VRAM-aware model choice, CUDA->CPU fallback,
user corrections, and SRT/TXT/JSON exports.

Hinglish note: Whisper has no dedicated Hinglish mode. The Hinglish workflow uses automatic per-segment
language detection; output may mix Devanagari and Latin script. Low-confidence words are flagged for review.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .ffmpeg_tools import Cancelled

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3-turbo"]

# Rough planning figures, NOT measurements: download size and float16 GPU memory incl. overhead.
MODEL_INFO = {
    "tiny": {"download_mb": 75, "vram_mb": 600, "note": "fastest, lowest accuracy (testing)"},
    "base": {"download_mb": 145, "vram_mb": 800, "note": "fast, modest accuracy"},
    "small": {"download_mb": 485, "vram_mb": 1500, "note": "good CPU default"},
    "medium": {"download_mb": 1500, "vram_mb": 3500, "note": "better accuracy, slow on CPU"},
    "large-v3-turbo": {"download_mb": 1600, "vram_mb": 4000, "note": "best accuracy here; needs a GPU in practice"},
}

LANG_CODES = {"auto": None, "en": "en", "hi": "hi", "hinglish": None}

_CUDA_ERRORS = ("cuda", "cudnn", "cublas", "out of memory", "cudaerror", "libcudnn", "cuda driver")


@dataclass
class ModelChoice:
    requested: str
    effective: str
    device: str
    compute_type: str
    reason: str
    warning: str | None = None


def is_model_cached(name: str) -> bool:
    try:
        from faster_whisper.utils import download_model  # noqa: PLC0415
        download_model(name, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not available offline"
        return False


def vram_warning(model: str, free_mb: int | None) -> str | None:
    if free_mb is None:
        return None
    need = MODEL_INFO.get(model, {}).get("vram_mb")
    if need and need > free_mb - 500:
        return (f"'{model}' needs roughly {need} MB of GPU memory (rough estimate); {free_mb} MB is free. "
                "It may run out of memory; the app will then retry on CPU.")
    return None


def choose_model(whisper_cfg: dict[str, Any], cuda_verified: bool, free_vram_mb: int | None) -> ModelChoice:
    requested = whisper_cfg.get("model", "auto")
    device_pref = whisper_cfg.get("device", "auto")
    use_cuda = cuda_verified and device_pref in ("auto", "cuda")
    device = "cuda" if use_cuda else "cpu"
    compute = whisper_cfg.get("cuda_compute_type", "float16") if use_cuda else whisper_cfg.get("cpu_compute_type", "int8")
    if device_pref == "cuda" and not cuda_verified:
        reason = "CUDA requested but not verified on this PC; using CPU."
    else:
        reason = "CUDA verified" if use_cuda else "CPU"
    if requested == "auto":
        if use_cuda:
            effective = "small"
            for m in ("large-v3-turbo", "medium", "small"):
                if vram_warning(m, free_vram_mb) is None:
                    effective = m
                    break
            reason += f"; auto-selected '{effective}' for available GPU memory"
        else:
            effective = whisper_cfg.get("cpu_default_model", "small")
            reason += f"; auto-selected CPU default '{effective}'"
    else:
        effective = requested
    warning = vram_warning(effective, free_vram_mb) if use_cuda else None
    return ModelChoice(requested, effective, device, compute, reason, warning)


def free_gpu_memory() -> dict[str, Any]:
    """Release GPU memory held by this process before another GPU user (e.g. a local Ollama vision model) runs.
    faster-whisper models are already deleted after each transcription; this collects anything left and, if PyTorch
    happens to be installed, empties its CUDA cache. Safe to call on CPU-only machines."""
    done: dict[str, Any] = {"gc_objects": gc.collect()}
    try:
        import torch  # noqa: PLC0415 - optional, not a dependency of this app
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            done["torch_cache_cleared"] = True
    except Exception:  # noqa: BLE001 - no torch or no CUDA: nothing more to free
        pass
    return done


def _is_cuda_error(e: BaseException) -> bool:
    msg = f"{type(e).__name__} {e}".lower()
    return any(k in msg for k in _CUDA_ERRORS)


def transcribe_file(wav_16k: Path, choice: ModelChoice, language: str, vad_filter: bool = True, beam_size: int = 5,
                    allow_download: bool = False, uncertain_below: float = 0.5,
                    progress: Callable[[float, str], None] | None = None,
                    cancel_check: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Transcribe; on a CUDA/cuDNN/cuBLAS/OOM error, free the model and retry once on CPU."""
    attempts: list[dict[str, str]] = []
    plan = [(choice.device, choice.compute_type)]
    if choice.device == "cuda":
        plan.append(("cpu", "int8"))
    last_err: BaseException | None = None
    for device, compute in plan:
        try:
            result = _run(wav_16k, choice.effective, device, compute, language, vad_filter, beam_size,
                          allow_download, uncertain_below, progress, cancel_check)
            result["model"] = {"requested": choice.requested, "effective": choice.effective, "device": device,
                               "compute_type": compute, "reason": choice.reason, "fallbacks": attempts}
            return result
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            gc.collect()
            if device == "cuda" and _is_cuda_error(e):
                attempts.append({"device": device, "error": str(e)[:300]})
                if progress:
                    progress(0.0, "GPU transcription failed; retrying on CPU")
                continue
            raise
    raise RuntimeError(f"Transcription failed: {last_err}")


def load_wav_16k(path: Path):
    """Load our own 16 kHz mono PCM WAV as float32.

    We deliberately do NOT let faster-whisper decode files itself: faster-whisper 1.2.1 passes
    `metadata_errors=` to av.open(), which PyAV 19 removed (TypeError). Verified 2026-11-24.
    """
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if rate != 16000:
        raise ValueError(f"{Path(path).name} is {rate} Hz; transcription input must be 16 kHz")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


def _run(wav: Path, model_name: str, device: str, compute: str, language: str, vad_filter: bool, beam_size: int,
         allow_download: bool, uncertain_below: float, progress, cancel_check) -> dict[str, Any]:
    from faster_whisper import WhisperModel  # noqa: PLC0415

    if not allow_download and not is_model_cached(model_name):
        raise RuntimeError(
            f"Whisper model '{model_name}' is not downloaded yet (about {MODEL_INFO.get(model_name, {}).get('download_mb', '?')} MB). "
            "Allow the download on the Transcript page, or pick a model that is already available.")
    if device == "cuda":
        # pip-installed cuBLAS/cuDNN DLLs live under .venv/Lib/site-packages/nvidia/*/bin; this process must
        # register them first,
        # otherwise CTranslate2 fails with 'cublas64_12.dll is not found' (seen on the RTX laptop).
        from .hardware import register_nvidia_dll_dirs  # noqa: PLC0415
        register_nvidia_dll_dirs()
    model = WhisperModel(model_name, device=device, compute_type=compute, local_files_only=not allow_download)
    try:
        lang = LANG_CODES.get(language)
        kwargs: dict[str, Any] = dict(language=lang, word_timestamps=True, vad_filter=vad_filter, beam_size=beam_size,
                                      condition_on_previous_text=False)
        if language == "hinglish":
            kwargs["multilingual"] = True
        segments_iter, info = model.transcribe(load_wav_16k(wav), **kwargs)
        total = float(info.duration or 0) or 1.0
        segments: list[dict[str, Any]] = []
        for seg in segments_iter:
            if cancel_check and cancel_check():
                raise Cancelled("Cancelled during transcription")
            # faster-whisper may return NumPy scalars; convert to plain JSON-safe Python types.
            words = [{"start": round(float(w.start), 3), "end": round(float(w.end), 3), "word": str(w.word),
                      "probability": round(float(w.probability), 3),
                      "uncertain": bool(float(w.probability) < uncertain_below)}
                     for w in (seg.words or [])]
            segments.append({"id": len(segments), "start": round(float(seg.start), 3), "end": round(float(seg.end), 3),
                             "text": str(seg.text).strip(), "avg_logprob": round(float(seg.avg_logprob), 3),
                             "no_speech_prob": round(float(seg.no_speech_prob), 3), "words": words})
            if progress:
                progress(min(seg.end / total, 1.0), f"Transcribed {seg.end:.0f}s of {total:.0f}s")
        return {"language": str(info.language), "language_probability": round(float(info.language_probability), 3),
                "duration": float(info.duration), "segments": segments, "requested_language": language}
    finally:
        del model
        gc.collect()


# --- corrections and exports ----------------------------------------------------------------

def apply_edits(transcript: dict[str, Any], edits: dict[str, str]) -> dict[str, Any]:
    """Replace segment text with user corrections. Word timings for an edited segment are re-spread
    evenly across the new words inside the original segment bounds (approximate, marked as edited)."""
    out = json.loads(json.dumps(transcript))
    for seg in out["segments"]:
        new = edits.get(str(seg["id"]))
        if new is None or new.strip() == seg["text"].strip():
            continue
        tokens = new.split()
        seg["text"] = new.strip()
        seg["edited"] = True
        if not tokens:
            seg["words"] = []
            continue
        span = max(seg["end"] - seg["start"], 0.01)
        step = span / len(tokens)
        seg["words"] = [{"start": round(seg["start"] + i * step, 3), "end": round(seg["start"] + (i + 1) * step, 3),
                         "word": (" " if i else "") + t, "probability": 1.0, "uncertain": False, "edited": True}
                        for i, t in enumerate(tokens)]
    return out


def shift(transcript: dict[str, Any], offset: float, max_time: float | None = None) -> dict[str, Any]:
    """Move all timestamps (e.g. voiceover offset) and drop anything outside [0, max_time]."""
    out = json.loads(json.dumps(transcript))
    segs = []
    for seg in out["segments"]:
        s, e = seg["start"] + offset, seg["end"] + offset
        if e <= 0 or (max_time is not None and s >= max_time):
            continue
        seg["start"], seg["end"] = round(max(s, 0), 3), round(min(e, max_time) if max_time else e, 3)
        seg["words"] = [{**w, "start": round(max(w["start"] + offset, 0), 3), "end": round(w["end"] + offset, 3)}
                        for w in seg["words"] if w["end"] + offset > 0 and (max_time is None or w["start"] + offset < max_time)]
        segs.append(seg)
    out["segments"] = segs
    out["offset_applied"] = offset
    return out


def _ts(sec: float, sep: str = ",") -> str:
    ms = int(round(max(sec, 0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def caption_chunks(transcript: dict[str, Any], max_chars: int = 42, max_dur: float = 6.0) -> list[dict[str, Any]]:
    """Split segments into readable caption cues using word timings."""
    cues: list[dict[str, Any]] = []
    for seg in transcript["segments"]:
        words = seg.get("words") or []
        if not words:
            if seg["text"]:
                cues.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
            continue
        groups: list[list[dict]] = []
        cur: list[dict] = []
        for w in words:
            text = "".join(x["word"] for x in cur + [w]).strip()
            if cur and (len(text) > max_chars or w["end"] - cur[0]["start"] > max_dur):
                groups.append(cur)
                cur = []
            cur.append(w)
            cur_text = "".join(x["word"] for x in cur).strip()
            # Prefer breaking after sentence/clause punctuation once the cue is reasonably long.
            if cur_text[-1:] in ".!?,;:" and len(cur_text) >= max_chars * 0.5:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        # Merge a very short trailing cue (orphan word) into the previous one when it still fits.
        if len(groups) >= 2:
            last, prev = groups[-1], groups[-2]
            merged_len = len("".join(x["word"] for x in prev + last).strip())
            if (len(last) <= 2 or last[-1]["end"] - last[0]["start"] < 1.0) and merged_len <= max_chars + 16:
                groups[-2] = prev + last
                groups.pop()
        for g in groups:
            cues.append({"start": g[0]["start"], "end": g[-1]["end"], "text": "".join(x["word"] for x in g).strip()})
    for a, b in zip(cues, cues[1:]):  # no overlaps
        if a["end"] > b["start"]:
            a["end"] = b["start"]
    return [c for c in cues if c["end"] > c["start"] and c["text"]]


def to_srt(cues: list[dict[str, Any]]) -> str:
    return "\n".join(f"{i}\n{_ts(c['start'])} --> {_ts(c['end'])}\n{c['text']}\n" for i, c in enumerate(cues, 1))


def to_txt(transcript: dict[str, Any]) -> str:
    lines = [f"[{_ts(s['start'], '.')[:-4]}] {s['text']}" for s in transcript["segments"]]
    return "\n".join(lines) + ("\n" if lines else "")


def words_flat(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**w, "segment": s["id"]} for s in transcript["segments"] for w in s.get("words", [])]
