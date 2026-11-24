"""Domain types shared by services and UI."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobState(str, Enum):
    CREATED = "CREATED"
    INSPECTING = "INSPECTING"
    AUDIO_PREP = "AUDIO_PREP"
    ANALYZING = "ANALYZING"
    HIGHLIGHT_REVIEW = "HIGHLIGHT_REVIEW"
    EDITING = "EDITING"
    SHORTS_RENDER = "SHORTS_RENDER"
    ASSET_PREP = "ASSET_PREP"
    FINAL_REVIEW = "FINAL_REVIEW"
    APPROVED = "APPROVED"
    DRAFT_UPLOADED = "DRAFT_UPLOADED"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


# Forward workflow order. PAUSED/CANCELLED/FAILED are side states.
WORKFLOW: list[JobState] = [
    JobState.CREATED, JobState.INSPECTING, JobState.AUDIO_PREP, JobState.ANALYZING,
    JobState.HIGHLIGHT_REVIEW, JobState.EDITING, JobState.SHORTS_RENDER, JobState.ASSET_PREP,
    JobState.FINAL_REVIEW, JobState.APPROVED, JobState.DRAFT_UPLOADED,
]


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"


class TrackRole(str, Enum):
    COMBINED_MIX = "combined_mix"
    GAME = "game"
    MIC = "mic"
    OTHER = "other"


TRACK_ROLE_LABELS = {
    TrackRole.COMBINED_MIX: "Combined reference mix",
    TrackRole.GAME: "Game audio / in-game commentary",
    TrackRole.MIC: "Live microphone",
    TrackRole.OTHER: "Other / unknown",
}


class HardwareMode(str, Enum):
    CPU = "CPU"
    NVIDIA_NVENC_ONLY = "NVIDIA_NVENC_ONLY"
    NVIDIA_CUDA_AND_NVENC = "NVIDIA_CUDA_AND_NVENC"
    NVIDIA_CUDA_ONLY = "NVIDIA_CUDA_ONLY"
    GPU_DETECTED_BUT_UNAVAILABLE = "GPU_DETECTED_BUT_UNAVAILABLE"


CONTENT_STYLES = [
    "Challenge / Impossible Goal",
    "Sports Clutch / Match",
    "Open-World Chaos / Stunts",
    "Horror / Funny Reactions",
    "Shooter / High-Skill Plays",
    "Story / Mod Experiment",
    "Custom",
]

LANGUAGES = {"auto": "Auto", "en": "English", "hi": "Hindi", "hinglish": "Hinglish (mixed Hindi/English)"}

PROJECT_STATUS_IMPORTED = "imported"
PROJECT_STATUS_INSPECTED = "inspected"
PROJECT_STATUS_READY_AUDIO = "tracks_confirmed"


# --- Media inspection results -------------------------------------------------------------

class VideoStreamInfo(BaseModel):
    stream_index: int
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    pix_fmt: str | None = None
    r_frame_rate: float | None = None
    avg_frame_rate: float | None = None
    rotation: int = 0
    nb_frames: int | None = None
    start_time: float | None = None


class AudioStreamInfo(BaseModel):
    stream_index: int
    audio_index: int  # position among audio streams (ffmpeg "a:N")
    codec: str | None = None
    channels: int | None = None
    channel_layout: str | None = None
    sample_rate: int | None = None
    language: str | None = None
    title: str | None = None
    is_default: bool = False
    start_time: float | None = None


class MediaInfo(BaseModel):
    path: str
    format_name: str | None = None
    duration_sec: float = Field(ge=0)
    size_bytes: int | None = None
    bit_rate: int | None = None
    video: VideoStreamInfo | None = None
    audio: list[AudioStreamInfo] = Field(default_factory=list)
    subtitle_stream_count: int = 0
    vfr_suspected: bool = False
    vfr_measured: bool | None = None
    vfr_details: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def fps(self) -> float | None:
        if not self.video:
            return None
        return self.video.avg_frame_rate or self.video.r_frame_rate

    @property
    def display_resolution(self) -> tuple[int, int] | None:
        if not self.video or not self.video.width or not self.video.height:
            return None
        w, h = self.video.width, self.video.height
        if abs(self.video.rotation) % 180 == 90:
            w, h = h, w
        return w, h
