"""Optional YouTube upload with the official Google API client (google-api-python-client 2.202.0,
google-auth-oauthlib 1.5.0). Field and method names were checked against the YouTube Data API v3 discovery
document bundled with the client: videos.insert(part, body, media_body), status.privacyStatus
(public|unlisted|private), status.selfDeclaredMadeForKids, snippet.categoryId/tags, thumbnails.set(videoId).

Rules: private by default; public/unlisted only with explicit confirmation; nothing uploads without the user's
action. The result is a PRIVATE VIDEO on the channel, not a YouTube Studio "draft". No live upload has been tested
from the dev PC (no OAuth client configured there).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
PRIVACY = ("private", "unlisted", "public")
GAMING_CATEGORY_ID = "20"  # YouTube's "Gaming" category as known to the app; verify in YouTube Studio if it fails.
CHUNK = 8 * 1024 * 1024


class UploadError(RuntimeError):
    pass


class NeedsReconnect(UploadError):
    pass


def connect(secrets_file: Path, token_file: Path) -> None:
    """Interactive OAuth consent in the browser; stores the token (git-ignored data/ folder)."""
    from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: PLC0415

    if not secrets_file.exists():
        raise UploadError(f"OAuth client file not found: {secrets_file}. Set YOUTUBE_CLIENT_SECRETS_FILE in .env.")
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_file), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True, timeout_seconds=300)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(creds.to_json(), encoding="utf-8")


def load_credentials(token_file: Path) -> Any:
    from google.auth.exceptions import RefreshError  # noqa: PLC0415
    from google.auth.transport.requests import Request  # noqa: PLC0415
    from google.oauth2.credentials import Credentials  # noqa: PLC0415

    if not token_file.exists():
        raise NeedsReconnect("YouTube account not connected. Use 'Connect YouTube account' first.")
    creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as e:
                raise NeedsReconnect("The YouTube sign-in expired or was revoked. Connect again.") from e
            token_file.write_text(creds.to_json(), encoding="utf-8")
        else:
            raise NeedsReconnect("The YouTube sign-in is not valid. Connect again.")
    return creds


def build_service(creds: Any) -> Any:
    from googleapiclient.discovery import build  # noqa: PLC0415

    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def video_body(title: str, description: str, tags: list[str], privacy: str, made_for_kids: bool,
               category_id: str = GAMING_CATEGORY_ID) -> dict[str, Any]:
    if privacy not in PRIVACY:
        raise UploadError(f"Unknown privacy '{privacy}'")
    if not title.strip() or len(title) > 100:
        raise UploadError("The title must be 1-100 characters.")
    return {"snippet": {"title": title.strip(), "description": description, "tags": tags, "categoryId": category_id},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": bool(made_for_kids)}}


def _status(e: Exception) -> int | None:
    return getattr(e, "status_code", None) or getattr(getattr(e, "resp", None), "status", None)


def upload(service: Any, video: Path, body: dict[str, Any], progress: Callable[[float], None] | None = None,
           cancel_check: Callable[[], bool] | None = None, sleep: Callable[[float], None] = time.sleep,
           max_retries: int = 5) -> dict[str, Any]:
    """Resumable upload; retries transient errors (5xx, network) with backoff. Returns the API response."""
    from googleapiclient.errors import HttpError  # noqa: PLC0415
    from googleapiclient.http import MediaFileUpload  # noqa: PLC0415

    media = MediaFileUpload(str(video), mimetype="video/mp4", chunksize=CHUNK, resumable=True)
    request = service.videos().insert(part="snippet,status", body=body, media_body=media)
    response, retries = None, 0
    while response is None:
        if cancel_check and cancel_check():
            raise UploadError("Upload cancelled. Nothing was published; a partial upload is discarded by YouTube.")
        try:
            status, response = request.next_chunk()
            retries = 0
            if status is not None and progress:
                progress(float(status.progress()))
        except HttpError as e:
            code = _status(e)
            text = str(e)
            if code == 401:
                raise NeedsReconnect("YouTube rejected the sign-in (401). Connect again.") from e
            if code == 403 and "quota" in text.lower():
                raise UploadError("YouTube API quota exceeded for today. Try again after the quota resets.") from e
            if code is not None and code >= 500 and retries < max_retries:
                retries += 1
                sleep(min(2 ** retries, 60))
                continue
            raise UploadError(f"YouTube upload failed (HTTP {code}): {text[:300]}") from e
        except (ConnectionError, TimeoutError, OSError) as e:
            if retries >= max_retries:
                raise UploadError(f"Network problem during upload: {type(e).__name__}") from e
            retries += 1
            sleep(min(2 ** retries, 60))
    if not isinstance(response, dict) or not response.get("id"):
        raise UploadError(f"YouTube did not return a video id: {str(response)[:200]}")
    return response


def set_thumbnail(service: Any, video_id: str, image: Path) -> str | None:
    """Returns a warning text on failure (custom thumbnails may need a verified channel)."""
    from googleapiclient.http import MediaFileUpload  # noqa: PLC0415

    mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    try:
        service.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(image), mimetype=mime)).execute()
        return None
    except Exception as e:  # noqa: BLE001 - optional step
        return f"The video uploaded, but the custom thumbnail was not set ({type(e).__name__}: {str(e)[:150]})."
