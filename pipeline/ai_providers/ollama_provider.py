"""Ollama adapter (local HTTP API, default http://localhost:11434).

Verified against Ollama 0.40.16 on the dev PC: /api/tags lists models with ``capabilities``; /api/show returns
``capabilities`` (``vision`` present only for image-capable models); /api/chat accepts a JSON schema in ``format``
and base64 images in ``messages[].images``; an unknown model returns HTTP 404.
This adapter never calls /api/pull: models are installed by the user, never downloaded by the app.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from .base import Capabilities, ImageInput, ProviderError, ProviderStatus, ProviderUnavailable, VideoInput


class OllamaProvider:
    name = "ollama"
    cloud = False

    def __init__(self, base_url: str, model: str, timeout_sec: float = 300, temperature: float = 0.2,
                 num_ctx: int = 8192, transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.model = model or ""
        self.timeout = float(timeout_sec)
        self.temperature = float(temperature)
        self.num_ctx = int(num_ctx)
        self._transport = transport  # tests inject httpx.MockTransport
        self._caps: Capabilities | None = None

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=timeout, transport=self._transport)

    # --- discovery ------------------------------------------------------------------------------
    def list_models(self) -> list[dict[str, Any]]:
        """Installed models with capabilities. Raises ProviderUnavailable when Ollama is not reachable."""
        try:
            with self._client(10) as c:
                r = c.get("/api/tags")
                r.raise_for_status()
                models = r.json().get("models", [])
        except (httpx.HTTPError, ValueError) as e:
            raise ProviderUnavailable(f"Ollama is not reachable at {self.base_url} ({type(e).__name__}). "
                                      "Start Ollama, or choose another provider.") from e
        return [{"name": m.get("name") or m.get("model"), "size": m.get("size"),
                 "capabilities": m.get("capabilities") or [],
                 "parameter_size": (m.get("details") or {}).get("parameter_size")} for m in models]

    def capabilities(self) -> Capabilities:
        if self._caps is not None:
            return self._caps
        try:
            with self._client(20) as c:
                r = c.post("/api/show", json={"model": self.model})
        except httpx.HTTPError as e:
            raise ProviderUnavailable(f"Ollama is not reachable at {self.base_url}.") from e
        if r.status_code == 404:
            raise ProviderUnavailable(f"Ollama model '{self.model}' is not installed. Install it yourself with "
                                      f"`ollama pull {self.model}` (the app never downloads models).")
        if r.status_code >= 400:
            raise ProviderError(f"Ollama /api/show failed: HTTP {r.status_code} {r.text[:200]}")
        caps = r.json().get("capabilities") or []
        vision = "vision" in caps
        notes = [] if vision else ["Text-only model: it cannot see frames. Visual evidence comes from computed "
                                   "signals (motion, scene changes, sharpness) and your review."]
        self._caps = Capabilities(text="completion" in caps or not caps, images=vision, video=False, notes=notes)
        return self._caps

    def status(self) -> ProviderStatus:
        if not self.model:
            return ProviderStatus(False, self.name, "", "No Ollama model selected.")
        try:
            version = ""
            with self._client(10) as c:
                r = c.get("/api/version")
                if r.status_code == 200:
                    version = r.json().get("version", "")
            caps = self.capabilities()
        except ProviderError as e:
            return ProviderStatus(False, self.name, self.model, str(e))
        kind = "vision (can see frames)" if caps.images else "text-only (cannot see frames)"
        return ProviderStatus(True, self.name, self.model, f"Ollama {version}: '{self.model}' is installed, {kind}.",
                              caps, {"version": version})

    # --- inference ------------------------------------------------------------------------------
    def generate_json(self, system: str, prompt: str, schema: dict[str, Any], images: list[ImageInput] | None = None,
                      video: VideoInput | None = None) -> str:
        if not self.model:
            raise ProviderUnavailable("No Ollama model selected.")
        if video is not None:
            raise ProviderUnavailable("Ollama cannot analyse video files; frames are used instead.")
        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            if not self.capabilities().images:
                raise ProviderUnavailable(f"Ollama model '{self.model}' cannot read images.")
            msg["content"] = prompt + "\n\nImages, in order: " + "; ".join(i.label for i in images)
            msg["images"] = [base64.b64encode(i.jpeg).decode("ascii") for i in images]
        body = {"model": self.model, "stream": False, "format": schema,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
                "messages": [{"role": "system", "content": system}, msg]}
        try:
            with self._client(self.timeout) as c:
                r = c.post("/api/chat", json=body)
        except httpx.TimeoutException as e:
            raise ProviderError(f"Ollama did not answer within {self.timeout:.0f} s. A smaller model or a shorter "
                                "analysis window may help.", transient=True) from e
        except httpx.HTTPError as e:
            raise ProviderUnavailable(f"Ollama is not reachable at {self.base_url}: {type(e).__name__}") from e
        if r.status_code == 404:
            raise ProviderUnavailable(f"Ollama model '{self.model}' is not installed.")
        if r.status_code >= 500:
            raise ProviderError(f"Ollama error HTTP {r.status_code}: {r.text[:300]}", transient=True)
        if r.status_code >= 400:
            raise ProviderError(f"Ollama rejected the request: HTTP {r.status_code} {r.text[:300]}")
        try:
            data = r.json()
        except ValueError as e:
            raise ProviderError("Ollama returned a non-JSON response.", transient=True) from e
        if data.get("done_reason") == "length":
            raise ProviderError("Ollama stopped at its output length limit; the answer is incomplete.")
        content = (data.get("message") or {}).get("content", "")
        if not content.strip():
            raise ProviderError("Ollama returned an empty answer.", transient=True)
        return content
