"""Vertical Shorts (1080x1920) from the ORIGINAL recording and the project's audio mix, in three layouts.

Layouts never stretch the picture: every region is cropped to the exact target aspect ratio before scaling.
- blur_fit:    whole frame (HUD included) fitted to the width, over a blurred, darkened copy of itself.
- center_crop: a 9:16 crop; the horizontal position is adjustable (0 = left edge, 1 = right edge).
- split_stack: top half = 9:8 crop of the action; bottom half = the whole frame fitted (default) or a chosen
               9:8 region (e.g. a scoreboard or minimap), each 1080x960.
Captions are burned with FFmpeg's ``ass`` filter, referenced by a bare filename (FFmpeg runs in the caption folder),
which avoids escaping Windows drive-letter colons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from . import ffmpeg_tools as fx
from .audio_prep import audio_stats
from .captions import CaptionStyle
from .frame_sampler import measure
from .media_probe import decode_check, probe
from .short_edit import Segment, atempo_chain, output_length
from .timeline_builder import QUALITY_PRESETS

OUT_W, OUT_H = 1080, 1920
LAYOUTS = {"blur_fit": "Full frame over blurred background (keeps the whole HUD)",
           "center_crop": "Vertical crop (fills the screen; edges are cut off)",
           "split_stack": "Split: zoomed action on top, full frame (or a chosen region) below"}


class ShortSettings(BaseModel):
    layout: str = "blur_fit"
    crop_x: float = Field(0.5, ge=0, le=1, description="center_crop / split top: horizontal position")
    fit_y: float = Field(0.4, ge=0, le=1, description="blur_fit: vertical position of the frame (0 top, 1 bottom)")
    bottom_mode: str = "full"  # split_stack bottom: "full" | "region"
    region_x: float = Field(0.5, ge=0, le=1)
    region_y: float = Field(1.0, ge=0, le=1)
    region_w: float = Field(0.35, ge=0.1, le=1)
    region_h: float | None = Field(None, ge=0.05, le=1, description="split region height (any shape); None = legacy "
                                                                      "region shaped like the bottom panel")
    split_ratio: float | None = Field(None, ge=0.3, le=0.9, description="top panel share of the height; None = 0.5, or "
                                                                         "fitted to a region strip")
    dead_air: str = Field("", description="'trim' | 'speedup' | 'none'; '' = from the content-style profile")
    speedup_factor: float = Field(2.0, ge=2.0, le=2.5)
    voice_file: str = Field("", description="per-Short voiceover (stored path); '' = use the project audio mix")
    voice_offset: float = Field(0.0, ge=0, le=600, description="seconds into the Short where the voiceover starts")
    voice_gain_db: float = Field(0.0, ge=-20, le=20)
    quality: str = "high"
    encoder: str = "auto"
    audio_fade_sec: float = Field(0.08, ge=0, le=2)
    captions: CaptionStyle = Field(default_factory=CaptionStyle)


def _even(x: float) -> int:
    return max(int(x) // 2 * 2, 2)


def _crop_to_aspect(w: int, h: int, aspect: float, pos_x: float = 0.5, pos_y: float = 0.5) -> tuple[int, int, int, int]:
    """Largest crop of aspect (w/h) inside a w x h frame, positioned by pos_x/pos_y in 0..1. (cw, ch, x, y)."""
    if w / h >= aspect:
        ch, cw = _even(h), _even(h * aspect)
    else:
        cw, ch = _even(w), _even(w / aspect)
    cw, ch = min(cw, _even(w)), min(ch, _even(h))
    return cw, ch, int((w - cw) * pos_x), int((h - ch) * pos_y)


def _fit_over_blur(label_in: str, w: int, h: int, cell_w: int, cell_h: int, fit_y: float, out: str) -> str:
    """Fit the whole frame into a cell (no stretch) over a blurred cover-scaled copy of the same frame."""
    if w / h >= cell_w / cell_h:
        fw, fh = cell_w, _even(cell_w * h / w)
    else:
        fw, fh = _even(cell_h * w / h), cell_h
    sw, sh = max(cell_w // 4, 2), max(cell_h // 4, 2)  # blur at quarter size: same look, much faster
    return (f"[{label_in}]split=2[{out}_a][{out}_b];"
            f"[{out}_a]scale={sw}:{sh}:force_original_aspect_ratio=increase,crop={sw}:{sh},gblur=sigma=12,"
            f"scale={cell_w}:{cell_h},eq=brightness=-0.08[{out}_bg];"
            f"[{out}_b]scale={fw}:{fh}[{out}_fg];"
            f"[{out}_bg][{out}_fg]overlay=(W-w)/2:(H-h)*{fit_y:.3f}[{out}]")


def split_geometry(s: ShortSettings, src_w: int, src_h: int) -> dict[str, int]:
    """Panel heights and the bottom region for the split layout. A region of any shape (e.g. a thin score strip)
    gets a bottom panel sized to it, unless split_ratio fixes the panel heights."""
    g: dict[str, int] = {}
    free = s.bottom_mode == "region" and s.region_h is not None
    if free:
        rw, rh = min(_even(src_w * s.region_w), _even(src_w)), min(_even(src_h * s.region_h), _even(src_h))
        g.update(rw=rw, rh=rh, rx=int((src_w - rw) * s.region_x), ry=int((src_h - rh) * s.region_y))
    if s.split_ratio is not None:
        top_h = _even(OUT_H * s.split_ratio)
    elif free:
        top_h = OUT_H - min(max(_even(OUT_W * g["rh"] / g["rw"]), 160), 960)
    else:
        top_h = OUT_H // 2
    g.update(top_h=top_h, bot_h=OUT_H - top_h)
    return g


def video_graph(s: ShortSettings, src_w: int, src_h: int, captions_file: str | None, in_label: str = "0:v") -> str:
    """Filter graph from [in_label] to [v] (1080x1920, yuv420p)."""
    if s.layout == "blur_fit":
        body = _fit_over_blur(in_label, src_w, src_h, OUT_W, OUT_H, s.fit_y, "lay")
    elif s.layout == "center_crop":
        cw, ch, x, y = _crop_to_aspect(src_w, src_h, OUT_W / OUT_H, s.crop_x)
        body = f"[{in_label}]crop={cw}:{ch}:{x}:{y},scale={OUT_W}:{OUT_H}:flags=lanczos[lay]"
    elif s.layout == "split_stack":
        g = split_geometry(s, src_w, src_h)
        top_h, bot_h = g["top_h"], g["bot_h"]
        cw, ch, x, y = _crop_to_aspect(src_w, src_h, OUT_W / top_h, s.crop_x)
        top = f"[top_in]crop={cw}:{ch}:{x}:{y},scale={OUT_W}:{top_h}:flags=lanczos[top]"
        if "rw" in g:  # a free-shaped region, fitted (never stretched) over a blur of itself
            bottom = (f"[bot_in]crop={g['rw']}:{g['rh']}:{g['rx']}:{g['ry']}[reg];"
                      + _fit_over_blur("reg", g["rw"], g["rh"], OUT_W, bot_h, 0.5, "bot"))
        elif s.bottom_mode == "region":  # legacy: a region shaped like the bottom panel
            rw = _even(src_w * s.region_w)
            rh = _even(rw * bot_h / OUT_W)
            if rh > src_h:
                rh = _even(src_h)
                rw = _even(rh * OUT_W / bot_h)
            rx, ry = int((src_w - rw) * s.region_x), int((src_h - rh) * s.region_y)
            bottom = f"[bot_in]crop={rw}:{rh}:{rx}:{ry},scale={OUT_W}:{bot_h}:flags=lanczos[bot]"
        else:
            bottom = _fit_over_blur("bot_in", src_w, src_h, OUT_W, bot_h, 0.5, "bot")
        body = f"[{in_label}]split=2[top_in][bot_in];{top};{bottom};[top][bot]vstack=inputs=2[lay]"
    else:
        raise ValueError(f"Unknown layout '{s.layout}'")
    tail = f",ass={captions_file}" if captions_file else ""
    return f"{body};[lay]setsar=1{tail},format=yuv420p[v]"


def _video_args(encoder: str, quality: str) -> list[str]:
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["high"])
    return ["-c:v", encoder, *(preset["nvenc"] if encoder == "h264_nvenc" else preset["x264"])]


def output_fps(src_fps: float | None) -> float:
    fps = src_fps or 30.0
    return round(min(fps, 60.0), 3)


def render_short(source: Path, audio: Path | None, start: float, end: float, s: ShortSettings, src_size: tuple[int, int],
                 src_fps: float | None, encoder: str, work_dir: Path, captions_ass: str | None, out: Path,
                 progress: Callable[[float], None] | None = None,
                 cancel_check: Callable[[], bool] | None = None, segments: list[Segment] | None = None,
                 shift: float = 0.0, audio_start: float | None = None) -> dict[str, Any]:
    """Render one Short. ``captions_ass`` is the ASS text in OUTPUT time (written into work_dir) or None.
    ``segments`` (clip time, see short_edit) trim or speed up parts; ``shift`` = media_probe.video_seek_shift;
    ``audio_start`` = where the clip starts in ``audio`` (default ``start``; 0 for a per-Short audio file)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    clip = round(end - start, 3)
    segs = segments or [(0.0, clip, 1.0)]
    dur = output_length(segs)
    cap_name = None
    if captions_ass:
        cap_name = "captions.ass"
        (work_dir / cap_name).write_text(captions_ass, encoding="utf-8")
    a0 = start if audio_start is None else audio_start
    args = ["-y", "-ss", f"{start + shift:.3f}", "-t", f"{clip:.3f}", "-i", str(source)]
    if audio is not None:
        args += ["-ss", f"{a0:.3f}", "-t", f"{clip:.3f}", "-i", str(audio)]
    pre, v_in, a_in = [], "0:v", "1:a"
    if segs != [(0.0, clip, 1.0)]:  # cut the clip into segments, speed some up, join them again
        n = len(segs)
        pre.append(f"[0:v]split={n}" + "".join(f"[sv{k}]" for k in range(n)) if n > 1 else "[0:v]null[sv0]")
        if audio is not None:
            pre.append(f"[1:a]aresample=48000,asplit={n}" + "".join(f"[sa{k}]" for k in range(n)) if n > 1
                       else "[1:a]aresample=48000,anull[sa0]")
        for k, (a, b, v) in enumerate(segs):
            pre.append(f"[sv{k}]trim=start={a:.3f}:end={b:.3f},setpts=(PTS-STARTPTS)/{v:g}[cv{k}]")
            if audio is not None:
                pre.append(f"[sa{k}]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,{atempo_chain(v)}[ca{k}]")
        joined = "".join(f"[cv{k}][ca{k}]" if audio is not None else f"[cv{k}]" for k in range(n))
        pre.append(joined + f"concat=n={n}:v=1:a={1 if audio is not None else 0}[segv]" +
                   ("[sega]" if audio is not None else ""))
        v_in, a_in = "segv", "sega"
    graph = ";".join([*pre, video_graph(s, src_size[0], src_size[1], cap_name, in_label=v_in)])
    fade = min(s.audio_fade_sec, dur / 4)
    if audio is not None:
        a_chain = f"[{a_in}]aresample=48000,asetpts=PTS-STARTPTS"
        if fade > 0:
            a_chain += f",afade=t=in:d={fade:.3f},afade=t=out:st={max(dur - fade, 0):.3f}:d={fade:.3f}"
        graph += f";{a_chain},apad=whole_dur={dur:.3f},atrim=end={dur:.3f}[a]"
    (work_dir / "graph.txt").write_text(graph, encoding="utf-8")
    tmp = out.with_name(out.stem + ".partial.mp4")
    fps = output_fps(src_fps)
    args += ["-/filter_complex", str(work_dir / "graph.txt"), "-map", "[v]"]
    # Always stereo: FFmpeg 9.1's AAC encoder stalled (never finished) on MONO input at 192 kb/s in testing.
    args += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"] if audio is not None else ["-an"]
    args += [*_video_args(encoder, s.quality), "-r", f"{fps:g}", "-fps_mode", "cfr", "-t", f"{dur:.3f}",
             "-movflags", "+faststart", str(tmp)]
    # Generous limit (a 60 s Short normally takes minutes at most) so a stuck FFmpeg can never hang a job forever.
    fx.ffmpeg_progress(args, dur, progress, cancel_check, cwd=work_dir, timeout=max(600.0, dur * 40))
    tmp.replace(out)
    return {"duration": dur, "clip_duration": clip, "fps": fps, "encoder": encoder, "layout": s.layout,
            "captions": bool(cap_name), "segments": [list(x) for x in segs], "graph": graph}


def preview_frame(source: Path, start: float, end: float, at: float, s: ShortSettings, src_size: tuple[int, int],
                  work_dir: Path, captions_ass: str | None, out: Path, shift: float = 0.0) -> Path:
    """One framing preview at clip time ``at`` (with captions as they will appear at that moment; dead-air edits
    are not applied to previews)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cap_name = None
    if captions_ass:
        cap_name = "captions.ass"
        (work_dir / cap_name).write_text(captions_ass, encoding="utf-8")
    (work_dir / "graph.txt").write_text(video_graph(s, src_size[0], src_size[1], cap_name), encoding="utf-8")
    at = min(max(at, 0.0), max(end - start - 0.05, 0.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    # Output-side -ss keeps clip timestamps, so captions show exactly what is on screen at that moment.
    fx.ffmpeg_progress(["-y", "-ss", f"{start + shift:.3f}", "-t", f"{end - start:.3f}", "-i", str(source),
                        "-/filter_complex", str(work_dir / "graph.txt"), "-map", "[v]", "-ss", f"{at:.3f}",
                        "-frames:v", "1", str(out)], end - start, cwd=work_dir, timeout=300)
    return out


def safe_zone_overlay(png: Path, out: Path, bottom: float = 0.22, right: float = 0.12) -> Path:
    """The framing preview with the areas YouTube's Shorts player usually covers shaded: the bottom (title, channel,
    music) and the right edge (like, comment, share buttons). Approximate: YouTube sets and changes these areas."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    with Image.open(png) as im:
        im = im.convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle((0, int(h * (1 - bottom)), w, h), fill=(255, 40, 40, 90), outline=(255, 40, 40, 255), width=3)
        d.rectangle((int(w * (1 - right)), 0, w, int(h * (1 - bottom))), fill=(255, 40, 40, 90),
                    outline=(255, 40, 40, 255), width=3)
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out)
    return out


def _frame_brightness(video: Path, t: float, work: Path, name: str, vf: str | None = None) -> float | None:
    p = work / f"{name}.png"
    try:
        fx.ffmpeg(["-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", *(["-vf", vf] if vf else []),
                   str(p)], timeout=120)
    except fx.FFmpegError:
        return None
    if not p.exists():
        return None
    return measure(p)[1]


def validate_short(out: Path, expected_dur: float, source: Path, start: float, audio_expected: bool,
                   work_dir: Path, audio_src: Path | None = None) -> tuple[list[str], dict[str, Any]]:
    """(errors, details). details["warnings"] holds non-blocking findings (e.g. the recording itself is silent)."""
    errors: list[str] = []
    warnings: list[str] = []
    info = probe(out)
    if abs(info.duration_sec - expected_dur) > 0.25:
        errors.append(f"Duration {info.duration_sec:.2f}s, expected {expected_dur:.2f}s")
    if not info.video:
        errors.append("No video stream")
    elif (info.video.width, info.video.height) != (OUT_W, OUT_H):
        errors.append(f"Size {info.video.width}x{info.video.height}, expected {OUT_W}x{OUT_H}")
    stats: dict[str, Any] = {}
    if audio_expected:
        if not info.audio:
            errors.append("No audio stream")
        else:
            stats = audio_stats(out)
            if (stats.get("rms_db") or -120) < -70:
                # Silent output is an error only if the same part of the audio mix is NOT silent.
                src_rms = (audio_stats(audio_src, start, expected_dur).get("rms_db") or -120) if audio_src else None
                if src_rms is not None and src_rms < -70:
                    warnings.append("This part of the recording has no audible sound, so the Short is silent. "
                                    "Add a voiceover or music if you want sound.")
                else:
                    errors.append("Audio is silent although the audio mix has sound here (rendering problem)")
    dec = decode_check(out, 2)
    if dec:
        errors.append("Start does not decode: " + "; ".join(dec[:2]))
    if info.duration_sec > 4:
        dec_end = decode_check(out, 2, start_sec=max(info.duration_sec - 3, 0))
        if dec_end:
            errors.append("End does not decode: " + "; ".join(dec_end[:2]))
    # Blank/black output from a broken filter graph: compare with the source at the same moments.
    black = []
    for i, frac in enumerate((0.25, 0.5, 0.75)):
        t = expected_dur * frac
        o = _frame_brightness(out, t, work_dir, f"val_out_{i}")
        s = _frame_brightness(source, start + t, work_dir, f"val_src_{i}", "scale=320:-2")
        if o is not None and s is not None and o < 8 and s > 25:
            black.append(round(t, 1))
    if len(black) >= 2:
        errors.append(f"Output is black at {black} s although the source is not (filter error)")
    return errors, {"probe": info.model_dump(), "audio_stats": stats, "warnings": warnings}
