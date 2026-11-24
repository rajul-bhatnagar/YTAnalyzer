"""Gemini adapter using the official ``google-genai`` SDK (``from google import genai``), version 3.6.1.

Verified against the installed SDK (signatures introspected on the dev PC, 2026-11-24): ``genai.Client(api_key=)``,
``client.models.get/list/generate_content``, ``client.files.upload/get/delete``, ``types.Part.from_bytes/from_text/
from_uri``, ``types.GenerateContentConfig(response_mime_type, response_json_schema, ...)``, ``types.FileState``
and ``errors.ClientError/ServerError`` with ``.code``. No live call has been made from this PC (no API key here).

Cost and privacy rules:
- Only the configured model is called; nothing enables billing or a paid feature. The app cannot see whether the
  key's Google Cloud project has billing enabled, so the UI tells the user to use a key from a project WITHOUT
  billing if they want a hard guarantee of zero cost.
- This provider is only constructed after the user consented to cloud analysis for the project.
- Uploaded video chunks are deleted after use (best effort; otherwise Google expires them automatically).
"""

from __future__ import annotations

import mimetypes
import re
import time
from typing import Any, Callable

from .base import (Capabilities, ImageInput, ProviderError, ProviderStatus, ProviderUnavailable, QuotaExceeded,
                   VideoInput)


def _find_retry_delay(obj: Any) -> float | None:
    """Look for google.rpc.RetryInfo 'retryDelay' (e.g. '39s') anywhere in an error payload."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "retryDelay" and isinstance(v, str):
                m = re.match(r"([\d.]+)s", v)
                if m:
                    return float(m.group(1))
            found = _find_retry_delay(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_retry_delay(v)
            if found is not None:
                return found
    return None


class GeminiProvider:
    name = "gemini"
    cloud = True

    def __init__(self, api_key: str | None, model: str, timeout_sec: float = 300, temperature: float = 0.2,
                 client_factory: Callable[[str], Any] | None = None, sleep: Callable[[float], None] = time.sleep,
                 file_poll_timeout_sec: float = 600):
        self.model = (model or "").strip()
        self._api_key = api_key or ""
        self.timeout = float(timeout_sec)
        self.temperature = float(temperature)
        self._factory = client_factory
        self._client: Any = None
        self._sleep = sleep
        self.file_poll_timeout = file_poll_timeout_sec
        self.cleanup_warnings: list[str] = []

    def __repr__(self) -> str:  # never expose the key
        return f"GeminiProvider(model={self.model!r})"

    # --- client -----------------------------------------------------------------------------
    def _get_client(self) -> Any:
        if not self._api_key:
            raise ProviderUnavailable("GEMINI_API_KEY is not set in .env. Add your key and restart the app, "
                                      "or use Ollama / signals-only analysis.")
        if not self.model:
            raise ProviderUnavailable("No Gemini model is configured.")
        if self._client is None:
            if self._factory is not None:
                self._client = self._factory(self._api_key)
            else:
                from google import genai  # noqa: PLC0415
                from google.genai import types  # noqa: PLC0415
                self._client = genai.Client(api_key=self._api_key,
                                            http_options=types.HttpOptions(timeout=int(self.timeout * 1000)))
        return self._client

    def _map_error(self, e: Exception) -> ProviderError:
        try:
            from google.genai import errors  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            errors = None  # type: ignore[assignment]
        code = getattr(e, "code", None)
        msg = str(getattr(e, "message", "") or e)[:400]
        if errors is not None and isinstance(e, errors.APIError) or isinstance(code, int):
            if code == 429:
                return QuotaExceeded("Gemini free-tier quota or rate limit reached (HTTP 429). Limits are set by "
                                     "Google and can change.", _find_retry_delay(getattr(e, "details", None)))
            if code in (401, 403):
                return ProviderUnavailable(f"Gemini rejected the API key or permission (HTTP {code}): {msg}")
            if code == 404:
                return ProviderUnavailable(f"Gemini model '{self.model}' was not found (HTTP 404). It may be retired "
                                           "or not available to your key; pick another model in Settings.")
            if isinstance(code, int) and code >= 500:
                return ProviderError(f"Gemini server error (HTTP {code}): {msg}", transient=True)
            return ProviderError(f"Gemini request failed (HTTP {code}): {msg}")
        name = type(e).__name__.lower()
        if "timeout" in name or "connect" in name or "network" in name:
            return ProviderError(f"Network problem talking to Gemini: {type(e).__name__}", transient=True)
        return ProviderError(f"Gemini request failed: {type(e).__name__}: {msg}")

    # --- discovery --------------------------------------------------------------------------
    def status(self) -> ProviderStatus:
        """Validates the key and that the configured model supports generateContent. Uses models.get only
        (no content generation)."""
        try:
            client = self._get_client()
            m = client.models.get(model=self.model)
        except ProviderError as e:
            return ProviderStatus(False, self.name, self.model, str(e))
        except Exception as e:  # noqa: BLE001
            return ProviderStatus(False, self.name, self.model, str(self._map_error(e)))
        actions = list(getattr(m, "supported_actions", None) or [])
        if actions and "generateContent" not in actions:
            return ProviderStatus(False, self.name, self.model,
                                  f"Model '{self.model}' does not support generateContent.")
        caps = Capabilities(text=True, images=True, video=True,
                            notes=["Image and video input per Google's model page; video-chunk upload is tried at "
                                   "runtime and falls back to frames if it fails."])
        return ProviderStatus(True, self.name, self.model,
                              f"Gemini model '{getattr(m, 'name', self.model)}' is available to this key.", caps,
                              {"input_token_limit": getattr(m, "input_token_limit", None)})

    def list_models(self) -> list[str]:
        client = self._get_client()
        try:
            out = []
            for m in client.models.list():
                acts = list(getattr(m, "supported_actions", None) or [])
                if "generateContent" in acts:
                    out.append(str(m.name).removeprefix("models/"))
            return sorted(out)
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e

    # --- inference --------------------------------------------------------------------------
    def _upload_video(self, client: Any, video: VideoInput) -> Any:
        from google.genai import types  # noqa: PLC0415

        mime = mimetypes.guess_type(str(video.path))[0] or "video/mp4"
        f = client.files.upload(file=str(video.path), config=types.UploadFileConfig(mime_type=mime))
        deadline = time.monotonic() + self.file_poll_timeout
        state = str(getattr(f, "state", "") or "")
        while "PROCESSING" in state:
            if time.monotonic() > deadline:
                self._delete_file(client, f)
                raise ProviderError("Gemini did not finish processing the uploaded video in time.", transient=True)
            self._sleep(3)
            f = client.files.get(name=f.name)
            state = str(getattr(f, "state", "") or "")
        if "FAILED" in state:
            self._delete_file(client, f)
            raise ProviderError(f"Gemini could not process the uploaded video: {getattr(f, 'error', '')}")
        return f

    def _delete_file(self, client: Any, f: Any) -> None:
        try:
            client.files.delete(name=f.name)
        except Exception as e:  # noqa: BLE001 - reported, not fatal
            self.cleanup_warnings.append(f"Could not delete uploaded file {getattr(f, 'name', '?')} from Gemini "
                                         f"({type(e).__name__}); Google deletes uploaded files automatically later.")

    def generate_json(self, system: str, prompt: str, schema: dict[str, Any], images: list[ImageInput] | None = None,
                      video: VideoInput | None = None) -> str:
        from google.genai import types  # noqa: PLC0415

        client = self._get_client()
        uploaded = None
        try:
            parts: list[Any] = []
            if video is not None:
                uploaded = self._upload_video(client, video)
                parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type))
            for img in images or []:
                parts.append(types.Part.from_text(text=img.label))
                parts.append(types.Part.from_bytes(data=img.jpeg, mime_type="image/jpeg"))
            parts.append(types.Part.from_text(text=prompt))
            config = types.GenerateContentConfig(system_instruction=system, temperature=self.temperature,
                                                 response_mime_type="application/json",
                                                 response_json_schema=schema)
            resp = client.models.generate_content(model=self.model, contents=parts, config=config)
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e
        finally:
            if uploaded is not None:
                self._delete_file(client, uploaded)
        text = getattr(resp, "text", None) or ""
        if not text.strip():
            reason = ""
            cands = getattr(resp, "candidates", None) or []
            if cands:
                reason = str(getattr(cands[0], "finish_reason", "") or "")
            raise ProviderError(f"Gemini returned no text{f' (finish reason {reason})' if reason else ''}.")
        return text
