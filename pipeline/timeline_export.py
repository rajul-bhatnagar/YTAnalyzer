"""Editor interchange: CMX3600 EDL and Final Cut Pro 7 XML (xmeml v5).

STATUS: structurally validated by our own parser tests. Import into DaVinci Resolve has NOT been
verified yet (no Resolve installation available); the UI labels these files as unverified.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

RECORD_START_HOURS = 1


def timebase(fps: float) -> tuple[int, bool]:
    """(integer timebase, ntsc flag), e.g. 29.97 -> (30, True), 60 -> (60, False)."""
    rounded = int(round(fps))
    ntsc = abs(fps - rounded * 1000 / 1001) < 0.01 and abs(fps - rounded) > 0.01
    return max(rounded, 1), ntsc


def frames(sec: float, fps: float) -> int:
    return int(round(sec * fps))


def tc(frame_count: int, base: int) -> str:
    """Non-drop-frame timecode."""
    f = frame_count % base
    s_total = frame_count // base
    return f"{s_total // 3600:02d}:{(s_total // 60) % 60:02d}:{s_total % 60:02d}:{f:02d}"


def fcp_pathurl(path: Path) -> str:
    """Final Cut Pro 7 XML style file URL: file://localhost/ + the absolute path with forward slashes, percent-encoded
    (spaces, non-ASCII), drive colon kept: file://localhost/D:/My%20Videos/game%20%C3%BC.mp4."""
    from urllib.parse import quote  # noqa: PLC0415
    return "file://localhost/" + quote(Path(path).resolve().as_posix().lstrip("/"), safe="/:")


def build_edl(title: str, clips: list[dict[str, Any]], fps: float, video_name: str, audio_name: str,
              video_url: str | None = None, audio_url: str | None = None) -> str:
    base, ntsc = timebase(fps)
    lines = [f"TITLE: {title[:60]}", "FCM: NON-DROP FRAME", ""]
    rec0 = RECORD_START_HOURS * 3600 * base
    n = 1
    for c in clips:
        si, so = frames(c["source_in"], fps), frames(c["source_out"], fps)
        ri, ro = rec0 + frames(c["record_in"], fps), rec0 + frames(c["record_out"], fps)
        for track, name, url in (("V", video_name, video_url), ("A", audio_name, audio_url)):
            lines.append(f"{n:03d}  AX       {track:<5} C        {tc(si, base)} {tc(so, base)} {tc(ri, base)} {tc(ro, base)}")
            lines.append(f"* FROM CLIP NAME: {name}")
            if url:
                lines.append(f"* SOURCE FILE: {url}")
            lines.append("")
            n += 1
    if ntsc:
        lines.insert(2, "* NOTE: NTSC rate written as non-drop-frame timecode")
    return "\n".join(lines)


def parse_edl(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 8 and parts[0].isdigit():
            events.append({"num": int(parts[0]), "track": parts[2], "src_in": parts[4], "src_out": parts[5],
                           "rec_in": parts[6], "rec_out": parts[7]})
    return events


def tc_to_frames(code: str, base: int) -> int:
    h, m, s, f = (int(x) for x in code.split(":"))
    return ((h * 60 + m) * 60 + s) * base + f


def _rate(parent: ET.Element, base: int, ntsc: bool) -> None:
    r = ET.SubElement(parent, "rate")
    ET.SubElement(r, "timebase").text = str(base)
    ET.SubElement(r, "ntsc").text = "TRUE" if ntsc else "FALSE"


def build_fcp7_xml(title: str, clips: list[dict[str, Any]], fps: float, width: int, height: int,
                   source_path: Path, source_duration: float, mix_path: Path) -> str:
    base, ntsc = timebase(fps)
    total = frames(clips[-1]["record_out"], fps) if clips else 0
    root = ET.Element("xmeml", version="5")
    seq = ET.SubElement(root, "sequence", id="sequence-1")
    ET.SubElement(seq, "name").text = title
    ET.SubElement(seq, "duration").text = str(total)
    _rate(seq, base, ntsc)
    media = ET.SubElement(seq, "media")
    video = ET.SubElement(media, "video")
    fmt = ET.SubElement(ET.SubElement(video, "format"), "samplecharacteristics")
    _rate(fmt, base, ntsc)
    ET.SubElement(fmt, "width").text = str(width)
    ET.SubElement(fmt, "height").text = str(height)
    vtrack = ET.SubElement(video, "track")
    audio = ET.SubElement(media, "audio")
    atrack = ET.SubElement(audio, "track")
    src_frames = frames(source_duration, fps)
    for i, c in enumerate(clips, 1):
        si, so = frames(c["source_in"], fps), frames(c["source_out"], fps)
        ri, ro = frames(c["record_in"], fps), frames(c["record_out"], fps)
        for kind, track, path in (("video", vtrack, source_path), ("audio", atrack, mix_path)):
            item = ET.SubElement(track, "clipitem", id=f"{kind}-clip-{i}")
            ET.SubElement(item, "name").text = path.name
            ET.SubElement(item, "duration").text = str(src_frames)
            _rate(item, base, ntsc)
            ET.SubElement(item, "start").text = str(ri)
            ET.SubElement(item, "end").text = str(ro)
            ET.SubElement(item, "in").text = str(si)
            ET.SubElement(item, "out").text = str(so)
            file_id = f"file-{kind}"
            f = ET.SubElement(item, "file", id=file_id)
            if i == 1:  # full file definition once; later clips reference the id
                ET.SubElement(f, "name").text = path.name
                ET.SubElement(f, "pathurl").text = fcp_pathurl(path)
                _rate(f, base, ntsc)
                ET.SubElement(f, "duration").text = str(src_frames)
            if kind == "audio":
                st = ET.SubElement(item, "sourcetrack")
                ET.SubElement(st, "mediatype").text = "audio"
                ET.SubElement(st, "trackindex").text = "1"
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + ET.tostring(root, encoding="unicode")


def validate_fcp7_xml(text: str, expected_clips: int) -> list[str]:
    errors = []
    try:
        root = ET.fromstring(text.split("\n", 2)[2] if text.startswith("<?xml") else text)
    except ET.ParseError as e:
        return [f"XML does not parse: {e}"]
    vclips = root.findall("./sequence/media/video/track/clipitem")
    aclips = root.findall("./sequence/media/audio/track/clipitem")
    if len(vclips) != expected_clips or len(aclips) != expected_clips:
        errors.append(f"Expected {expected_clips} clips per track, found {len(vclips)} video / {len(aclips)} audio")
    for c in vclips + aclips:
        s, e = int(c.findtext("start")), int(c.findtext("end"))
        i, o = int(c.findtext("in")), int(c.findtext("out"))
        if e - s != o - i or e <= s:
            errors.append(f"Clip {c.get('id')} has inconsistent in/out")
    return errors
