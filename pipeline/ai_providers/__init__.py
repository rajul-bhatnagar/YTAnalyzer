"""AI provider adapters (Gemini via the official google-genai SDK, Ollama via its local HTTP API)."""

from .base import (EVENT_CATEGORIES, AIEvent, AIProvider, Capabilities, ImageInput, ProviderError, ProviderStatus,
                   ProviderUnavailable, QuotaExceeded, VideoInput, WindowAnalysis, flat_schema)

__all__ = ["EVENT_CATEGORIES", "AIEvent", "AIProvider", "Capabilities", "ImageInput", "ProviderError",
           "ProviderStatus", "ProviderUnavailable", "QuotaExceeded", "VideoInput", "WindowAnalysis", "flat_schema"]
