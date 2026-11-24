"""Three-way mix: game_only, voice_only, both_auto_duck (sidechain ducking), with an optional
gentle quality chain and two-pass EBU R128 loudness normalisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from . import ffmpeg_tools as fx
from .audio_prep import audio_stats, measure_loudness
from .models import TrackRole

MODES = {
    "game_only": "Game only (game track incl. any in-game commentary)",
    "voice_only": "Voice only (live mic or voiceover)",
    "both_auto_duck": "Game + voice, game ducked while you speak",
}
VOICE_SOURCES = {"auto": "Auto (voiceover if present, else live mic)", "mic": "Live microphone track",
                 "voiceover": "Post-recorded voiceover", "none": "No voice"}

_FMT = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"


class MixSettings(BaseModel):
    mode: str = "game_only"
    voice_source: str = "auto"
    game_volume_db: float = Field(-6.0, ge=-40, le=12)
    voice_gain_db: float = Field(0.0, ge=-20, le=20)
    voice_offset_sec: float = Field(0.0, ge=-600, le=600)
    duck_threshold: float = Field(0.03, gt=0, le=1)
    duck_ratio: float = Field(6.0, ge=1, le=20)
    duck_attack_ms: float = Field(20, ge=0.01, le=2000)
    duck_release_ms: float = Field(400, ge=0.01, le=9000)
    duck_makeup: float = Field(1.0, ge=1, le=64)
    highpass_enabled: bool = False
    highpass_hz: int = Field(80, ge=20, le=400)
    noise_reduction_enabled: bool = False
    noise_reduction_db: float = Field(12, ge=1, le=60)
    loudnorm_enabled: bool = True
    loudnorm_target_lufs: float = Field(-14.0, ge=-30, le=-5)
    loudnorm_true_peak: float = Field(-1.5, ge=-9, le=0)
    limiter_enabled: bool = True
    fade_in_sec: float = Field(0.0, ge=0, le=30)
    fade_out_sec: float = Field(0.0, ge=0, le=30)

    @classmethod
    def from_settings(cls, audio: dict[str, Any]) -> "MixSettings":
        return cls(**{k: v for k, v in audio.items() if k in cls.model_fields})


@dataclass(frozen=True)
class MixSources:
    game: Path | None          # game track, or the combined mix when no separate game track exists
    game_kind: str | None      # "game" | "combined_mix"
    voice: Path | None         # live mic track or voiceover WAV
    voice_kind: str | None     # "mic" | "voiceover"


def resolve_sources(tracks: dict[str, Path], voiceover: Path | None, voice_source: str) -> MixSources:
    """tracks: role -> extracted 48 kHz WAV."""
    if TrackRole.GAME.value in tracks:
        game, gkind = tracks[TrackRole.GAME.value], "game"
    elif TrackRole.COMBINED_MIX.value in tracks:
        game, gkind = tracks[TrackRole.COMBINED_MIX.value], "combined_mix"
    else:
        game, gkind = None, None
    mic = tracks.get(TrackRole.MIC.value)
    voice, vkind = None, None
    if voice_source == "voiceover" or (voice_source == "auto" and voiceover is not None):
        voice, vkind = (voiceover, "voiceover") if voiceover else (None, None)
    elif voice_source in ("mic", "auto") and mic is not None:
        voice, vkind = mic, "mic"
    return MixSources(game, gkind, voice, vkind)


def available_modes(src: MixSources) -> dict[str, str | None]:
    """mode -> None if available, else the reason it is not."""
    out: dict[str, str | None] = {}
    out["game_only"] = None if src.game else "No game or combined audio track is mapped."
    out["voice_only"] = None if src.voice else "No live mic track or voiceover is available."
    if not src.game or not src.voice:
        out["both_auto_duck"] = "Needs both a game/combined track and a voice source."
    elif src.game_kind == "combined_mix" and src.voice_kind == "mic":
        out["both_auto_duck"] = ("The combined mix already contains the live mic; mixing them again would double "
                                 "your voice. Map a separate game track, or use a voiceover.")
    else:
        out["both_auto_duck"] = None
    return out


def _voice_chain(s: MixSettings, duration: float) -> list[str]:
    # asetpts: inputs may be seeked (previews); always work in timestamps starting at 0.
    f = [_FMT, "asetpts=PTS-STARTPTS"]
    if s.voice_offset_sec > 0.0005:
        f.append(f"adelay=delays={int(round(s.voice_offset_sec * 1000))}:all=1")
    elif s.voice_offset_sec < -0.0005:
        f += [f"atrim=start={-s.voice_offset_sec:.6f}", "asetpts=PTS-STARTPTS"]
    if s.highpass_enabled:
        f.append(f"highpass=f={s.highpass_hz}")
    if s.noise_reduction_enabled:
        f.append(f"afftdn=nr={s.noise_reduction_db:g}")
    f += [f"apad=whole_dur={duration:.6f}", f"atrim=end={duration:.6f}"]
    return f


def build_filter_graph(s: MixSettings, src: MixSources, duration: float, apply_limiter: bool = True,
                       fades: bool = True) -> tuple[list[Path], str]:
    """Return (input files in order, filter_complex string ending in [out])."""
    reason = available_modes(src).get(s.mode, "Unknown mode")
    if reason:
        raise ValueError(f"Mix mode '{s.mode}' unavailable: {reason}")
    inputs: list[Path] = []
    parts: list[str] = []
    if s.mode in ("game_only", "both_auto_duck"):
        inputs.append(src.game)  # type: ignore[arg-type]
        parts.append(f"[0:a]{_FMT},asetpts=PTS-STARTPTS,volume={s.game_volume_db:g}dB,apad=whole_dur={duration:.6f},"
                     f"atrim=end={duration:.6f}[g]")
    if s.mode in ("voice_only", "both_auto_duck"):
        vi = len(inputs)
        inputs.append(src.voice)  # type: ignore[arg-type]
        parts.append(f"[{vi}:a]{','.join(_voice_chain(s, duration))}[vc]")
    if s.mode == "game_only":
        last = "g"
    elif s.mode == "voice_only":
        parts.append(f"[vc]volume={s.voice_gain_db:g}dB[vg]")
        last = "vg"
    else:
        # The ducking detector listens to the voice BEFORE the user's gain, so voice volume and
        # ducking sensitivity are independent controls.
        parts.append("[vc]asplit=2[vraw][sc]")
        parts.append(f"[vraw]volume={s.voice_gain_db:g}dB[v]")
        parts.append(f"[g][sc]sidechaincompress=threshold={s.duck_threshold:g}:ratio={s.duck_ratio:g}:"
                     f"attack={s.duck_attack_ms:g}:release={s.duck_release_ms:g}:makeup={s.duck_makeup:g}[gd]")
        parts.append("[gd][v]amix=inputs=2:duration=first:normalize=0[m]")
        last = "m"
    post: list[str] = []
    if fades and s.fade_in_sec > 0:
        post.append(f"afade=t=in:st=0:d={s.fade_in_sec:g}")
    if fades and s.fade_out_sec > 0:
        post.append(f"afade=t=out:st={max(duration - s.fade_out_sec, 0):.3f}:d={s.fade_out_sec:g}")
    if apply_limiter and s.limiter_enabled:
        post.append("alimiter=limit=0.891:level=0")  # about -1 dBFS sample peak, no auto-gain
    post.append("aresample=48000")
    # Exact length. Measured on FFmpeg 9.1: amix/sidechaincompress non-deterministically drop the final 1-2
    # audio frames at end of stream (0-181 ms over 12 identical runs). Every short render was bit-identical
    # to a full one over its whole length, so the loss is only at the tail (sync is unaffected), and padding
    # the tail with silence restores the exact video duration.
    post += [f"apad=whole_dur={duration:.6f}", f"atrim=end={duration:.6f}"]
    parts.append(f"[{last}]{','.join(post)}[out]")
    return inputs, ";".join(parts)


def _write_graph(graph: str, work_dir: Path, name: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    f = work_dir / f"{name}.filtergraph.txt"
    f.write_text(graph, encoding="utf-8")
    return f


def render_mix(s: MixSettings, src: MixSources, duration: float, out_wav: Path,
               progress: Callable[[float], None] | None = None,
               cancel_check: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Full-length mix. With loudnorm: pass 1 renders the mix, measures it, pass 2 normalises linearly."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    stage1 = out_wav.with_name(out_wav.stem + ".pre.partial.wav")
    use_ln = s.loudnorm_enabled
    inputs, graph = build_filter_graph(s, src, duration, apply_limiter=not use_ln)
    gfile = _write_graph(graph, out_wav.parent, out_wav.stem)
    args: list[str] = ["-y"]
    for p in inputs:
        args += ["-i", str(p)]
    args += ["-/filter_complex", str(gfile), "-map", "[out]", "-c:a", "pcm_s16le", "-ar", "48000",
             str(stage1 if use_ln else out_wav.with_name(out_wav.stem + ".partial.wav"))]
    split = 0.5 if use_ln else 1.0
    fx.ffmpeg_progress(args, duration, (lambda f: progress(f * split)) if progress else None, cancel_check)
    report: dict[str, Any] = {"mode": s.mode, "settings": s.model_dump(), "inputs": [str(p) for p in inputs],
                              "loudness": None}
    if use_ln:
        m = measure_loudness(stage1, s.loudnorm_target_lufs, s.loudnorm_true_peak, cancel_check=cancel_check)
        report["loudness"] = {"measured": m, "target_lufs": s.loudnorm_target_lufs, "true_peak": s.loudnorm_true_peak}
        if m["input_i"] == float("-inf") or m["input_i"] < -70:
            # Silent mix: normalising would only amplify noise.
            post = "anull"
            report["loudness"]["skipped"] = "mix is silent"
        else:
            post = (f"loudnorm=I={s.loudnorm_target_lufs}:TP={s.loudnorm_true_peak}:LRA=11:"
                    f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
                    f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
        if s.limiter_enabled:
            post += ",alimiter=limit=0.891:level=0"
        # loudnorm drops a few hundred ms at the end; restore the exact video duration.
        post += f",aresample=48000,apad=whole_dur={duration:.6f},atrim=end={duration:.6f}"
        final_tmp = out_wav.with_name(out_wav.stem + ".partial.wav")
        fx.ffmpeg_progress(["-y", "-i", str(stage1), "-af", post, "-c:a", "pcm_s16le", "-ar", "48000",
                            str(final_tmp)], duration,
                           (lambda f: progress(0.5 + 0.5 * f)) if progress else None, cancel_check)
        stage1.unlink(missing_ok=True)
    out_wav.with_name(out_wav.stem + ".partial.wav").replace(out_wav)
    return report


def render_preview(s: MixSettings, src: MixSources, start: float, seconds: float, duration: float,
                   out_wav: Path) -> Path:
    """Short excerpt with the same graph (loudness normalisation not applied to previews).

    The graph runs in excerpt-local time (0..seconds). The game track is aligned with the video, so it
    is seeked to ``start``. The voice is offset from the video by ``voice_offset_sec``, so it is
    seeked to ``start - offset`` when that is positive, otherwise delayed by ``offset - start``.
    Seeking past the end of a short voiceover is fine: the chain pads it with silence."""
    start = min(max(start, 0.0), max(duration - 0.5, 0.0))
    seconds = max(min(seconds, duration - start), 0.5)
    voice_pos = start - s.voice_offset_sec          # where the excerpt begins in voice-file time
    voice_seek = max(voice_pos, 0.0)
    # model_copy skips validation on purpose: the local offset may exceed the UI's +/-600 s range.
    local = s.model_copy(update={"voice_offset_sec": max(-voice_pos, 0.0)})
    inputs, graph = build_filter_graph(local, src, seconds, fades=False)
    args: list[str] = ["-y"]
    for p in inputs:
        seek = voice_seek if (src.voice is not None and p == src.voice) else start
        if seek > 0.0005:
            args += ["-ss", f"{seek:.3f}"]
        args += ["-i", str(p)]
    gfile = _write_graph(graph, out_wav.parent, out_wav.stem)
    tmp = out_wav.with_name(out_wav.stem + ".partial.wav")
    fx.ffmpeg([*args, "-/filter_complex", str(gfile), "-map", "[out]", "-c:a", "pcm_s16le", str(tmp)], timeout=300)
    tmp.replace(out_wav)
    return out_wav


def validate_mix(path: Path, expected_duration: float) -> tuple[list[str], dict[str, Any]]:
    from .audio_prep import probe_audio_file  # noqa: PLC0415

    errors: list[str] = []
    info = probe_audio_file(path)
    if abs(info["duration_sec"] - expected_duration) > 0.1:
        errors.append(f"Mix duration {info['duration_sec']:.3f}s differs from video {expected_duration:.3f}s")
    if info["sample_rate"] != 48000:
        errors.append(f"Mix sample rate is {info['sample_rate']} Hz, expected 48000")
    stats = audio_stats(path)
    if stats["peak_db"] is not None and stats["peak_db"] > 0.0:
        errors.append(f"Mix peaks above 0 dBFS ({stats['peak_db']:.2f})")
    return errors, {"probe": info, "stats": stats}
