"""Vendor-neutral AI provider contract: structured output schemas, errors and the provider protocol.

Rules every provider follows:
- Output is JSON constrained by a schema, then validated with Pydantic by the caller. Nothing is trusted blindly.
- Providers never download models, never enable billing and never call paid endpoints.
- Errors are mapped to the small set of exceptions below so the analysis layer can decide: retry, fall back to
  another provider, pause the job, or continue with computed signals only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low", "none"]

# Generic categories (no game list). Models may also return a custom short category; it is kept as-is.
EVENT_CATEGORIES = {
    "achievement": "Achievement / success",
    "failure": "Failure / unexpected outcome",
    "close_call": "Close call",
    "clutch": "Clutch / high stakes",
    "skill": "Skillful execution",
    "surprise": "Surprise / twist",
    "funny": "Funny / chaotic moment",
    "story": "Story / emotional beat",
    "discovery": "Exploration / discovery",
    "tutorial": "Tutorial-worthy moment",
    "reaction": "Strong reaction",
    "unclear": "Unclear / needs review",
}


# How a clip ENDS, which sets the thumbnail's look (never its facts).
OUTCOME_MOODS = {"win_hype": "Win / hype", "clutch_tension": "Clutch / tension", "fail_shock": "Fail / shock",
                 "funny_chaos": "Funny / chaos"}


# --- structured outputs -------------------------------------------------------------------------
class FactorRatings(BaseModel):
    """0 = absent / cannot judge, 5 = very strong. Ranking aids, not objective truth."""

    significance: int = Field(0, ge=0, le=5, description="How much this moment matters in the game")
    stakes: int = Field(0, ge=0, le=5, description="Visible consequence or pressure")
    surprise: int = Field(0, ge=0, le=5, description="Unexpected or rare")
    skill: int = Field(0, ge=0, le=5, description="Skill or difficulty shown")
    entertainment: int = Field(0, ge=0, le=5, description="Emotional or entertainment value")
    narrative_completeness: int = Field(0, ge=0, le=5, description="Has a clear setup and payoff in the window")
    visual_clarity: int = Field(0, ge=0, le=5, description="Easy to see what happens; 0 if you cannot see video")
    standalone: int = Field(0, ge=0, le=5, description="Understandable without earlier context")


class ModalityConfidence(BaseModel):
    visual: Confidence = "none"
    audio: Confidence = "none"
    transcript: Confidence = "none"


class AIEvent(BaseModel):
    start_sec: float = Field(description="Start, seconds from the start of the recording")
    end_sec: float = Field(description="End, seconds from the start of the recording")
    category: str = Field(description="One of: " + ", ".join(EVENT_CATEGORIES) + "; or a short custom category")
    description: str = Field(description="What visibly/audibly happens. Facts only, no guesses", max_length=400)
    visual_observation: str = Field("", description="What the frames show; empty if you cannot see frames",
                                    max_length=400)
    frame_times: list[float] = Field(default_factory=list,
                                     description="Timestamps of the provided frames that support this event")
    audio_observation: str = Field("", max_length=300)
    transcript_quote: str = Field("", description="Exact words from the transcript, if relevant", max_length=300)
    confidence: ModalityConfidence = Field(default_factory=ModalityConfidence)
    uncertainty: str = Field("", description="What you are unsure about", max_length=300)
    factors: FactorRatings = Field(default_factory=FactorRatings)
    hook_caption: str = Field("", description="Short on-screen hook, max 8 words, must match what is shown",
                              max_length=80)
    suggested_title: str = Field("", description="Short honest title; no claims beyond the evidence", max_length=100)
    outcome_mood: str = Field("", description="How the moment ends: win_hype, clutch_tension, fail_shock, funny_chaos; "
                                              "'' if unclear")


class WindowAnalysis(BaseModel):
    summary: str = Field("", description="One or two sentences on what happens in this window", max_length=600)
    events: list[AIEvent] = Field(default_factory=list, max_length=8)
    uncertainty: str = Field("", max_length=400)


def flat_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Self-contained JSON schema: $refs inlined, titles/defaults dropped. Both Gemini (response_json_schema) and
    Ollama (format) accept this subset; no null types are produced because no field is Optional."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return walk(copy.deepcopy(defs[name]))
            return {k: walk(v) for k, v in node.items() if k not in ("title", "default")}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


# --- errors ----------------------------------------------------------------------------------------
class ProviderError(RuntimeError):
    """Request failed. ``transient`` errors may be retried."""

    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


class ProviderUnavailable(ProviderError):
    """Not configured / not running / model missing / no consent. Not retried."""


class QuotaExceeded(ProviderError):
    """Free quota or rate limit reached (HTTP 429). Caller pauses or falls back; never upgrades to paid."""

    def __init__(self, message: str, retry_after_sec: float | None = None):
        super().__init__(message, transient=False)
        self.retry_after_sec = retry_after_sec


# --- provider protocol -------------------------------------------------------------------------------
@dataclass
class Capabilities:
    text: bool = True
    images: bool = False
    video: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class ImageInput:
    jpeg: bytes
    label: str  # e.g. "Frame at 12:03.5 (723.5 s)" or "Contact sheet: frames 1-12"


@dataclass
class VideoInput:
    path: Any  # pathlib.Path of a low-resolution proxy chunk (Gemini Files API only)
    clip_start_sec: float


@dataclass
class ProviderStatus:
    ok: bool
    provider: str
    model: str
    message: str
    capabilities: Capabilities = field(default_factory=Capabilities)
    details: dict[str, Any] = field(default_factory=dict)


class AIProvider(Protocol):
    name: str
    model: str
    cloud: bool

    def status(self) -> ProviderStatus: ...

    def generate_json(self, system: str, prompt: str, schema: dict[str, Any], images: list[ImageInput] | None = None,
                      video: VideoInput | None = None) -> str:
        """Return the model's raw JSON text. Raises ProviderError / ProviderUnavailable / QuotaExceeded."""
        ...
