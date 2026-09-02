"""Google Drive access: credentials, folder resolution, upload and download.

Two callers share this module, and they want different things from it:

* ``scripts/upload_to_drive.py`` pushes a dataset once, interactively, from a
  laptop that has a browser.
* :mod:`fa_ocr_lora.training.checkpoint_sync` pushes checkpoints continuously,
  unattended, from a rented GPU box that has neither a browser nor a display.

The second caller is why credentials are resolved three ways instead of one.
The interactive consent flow that serves the first (``run_local_server``) opens
``http://localhost:<port>`` and waits for a human to click through Google's
consent screen; on a headless A100 there is nothing to open it with and the call
blocks until the run is killed. So there are three routes, tried in this order:

1. **A service-account key** (``service_account=``). No browser, no expiry, no
   interaction - the right answer for a server. The catch is that a service
   account has no Drive of its own and no storage quota, so it can only write
   into a Shared Drive or into a folder that a real account has shared with the
   service account's e-mail address. Uploading to its "My Drive" fails with
   ``storageQuotaExceeded``, which :func:`_is_transient` deliberately does not
   retry.
2. **A cached OAuth token** (``token=``), refreshed silently when it has a
   refresh token. Authorise once on the laptop, copy ``drive_token.json`` up to
   the server, and this path needs no browser either.
3. **The interactive consent flow** (``client_secret=``), only when
   ``interactive=True``. Anything else would hang a server run.

Both credential files are secrets. Their locations come from the environment
(``.env``) rather than from source, and the token this module writes is chmodded
to ``0600``.

Scope note: this asks only for ``drive.file``, which grants access to files the
application itself created. Folders this module makes with
:meth:`DriveClient.ensure_folder` are therefore fully visible to it, but a
folder created by hand in the Drive UI is not - pass its id explicitly in that
case, since resolving it by name would create a second folder alongside it.
"""

from __future__ import annotations

import io
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Final

from .config import ensure_dotenv_loaded, env_path

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

ensure_dotenv_loaded()

#: Per-file access to what this application creates - the narrowest scope that
#: can still upload. See the module docstring for what it cannot see.
SCOPES: Final[list[str]] = ["https://www.googleapis.com/auth/drive.file"]

DEFAULT_CLIENT_SECRET: Final[Path] = env_path(
    "GOOGLE_DRIVE_CREDENTIALS", REPO_ROOT / "secrets" / "drive_client_secret.json"
)
DEFAULT_TOKEN: Final[Path] = env_path(
    "GOOGLE_DRIVE_TOKEN", REPO_ROOT / "secrets" / "drive_token.json"
)
DEFAULT_SERVICE_ACCOUNT: Final[Path] = env_path(
    "GOOGLE_DRIVE_SERVICE_ACCOUNT", REPO_ROOT / "secrets" / "drive_service_account.json"
)

FOLDER_MIME: Final[str] = "application/vnd.google-apps.folder"
#: Bytes above which a resumable upload beats a single-shot one.
RESUMABLE_THRESHOLD: Final[int] = 8 * 1024 * 1024
#: Chunk size for resumable transfers in both directions.
CHUNK_SIZE: Final[int] = 8 * 1024 * 1024
MAX_ATTEMPTS: Final[int] = 6
RETRY_STATUSES: Final[frozenset[int]] = frozenset({403, 408, 429, 500, 502, 503, 504})


class DriveError(RuntimeError):
    """A Drive operation failed for a reason the caller should read.

    Raised for missing credentials and unusable configuration - the cases where
    a message is more useful than a traceback. Transport failures keep their own
    exception types so that :meth:`DriveClient._execute` can classify them.
    """


def _import_google() -> dict[str, Any]:
    """Import the Google client libraries, with an actionable error if absent.

    Returns:
        The handful of names this module needs, keyed by name, so callers pick
        what they want instead of unpacking a positional tuple that grows.

    Raises:
        DriveError: If the libraries are not installed.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google.oauth2.service_account import (
            Credentials as ServiceAccountCredentials,
        )
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DriveError(
            "Missing Google client libraries. Install them with:\n"
            "  pip install google-api-python-client google-auth-oauthlib\n"
            "or:\n"
            "  pip install -e '.[drive]'"
        ) from exc
    return {
        "Request": Request,
        "Credentials": Credentials,
        "ServiceAccountCredentials": ServiceAccountCredentials,
        "InstalledAppFlow": InstalledAppFlow,
        "build": build,
        "HttpError": HttpError,
        "MediaFileUpload": MediaFileUpload,
        "MediaIoBaseDownload": MediaIoBaseDownload,
    }


def load_credentials(
    client_secret: Path | None = None,
    token: Path | None = None,
    *,
    service_account: Path | None = None,
    interactive: bool = True,
) -> Any:
    """Return Drive credentials, preferring whatever needs no human present.

    Args:
        client_secret: OAuth desktop-client JSON, used only by the consent flow.
            Defaults to :data:`DEFAULT_CLIENT_SECRET`.
        token: Cached OAuth token, read if present and rewritten when refreshed.
            Defaults to :data:`DEFAULT_TOKEN`.
        service_account: Service-account key JSON. Tried first when it exists,
            because it is the only route that needs no browser and never
            expires. Defaults to :data:`DEFAULT_SERVICE_ACCOUNT`.
        interactive: Whether opening a browser is acceptable. Pass ``False`` on
            a server: the consent flow would otherwise block forever on a
            machine with no display, which is worse than a clear failure.

    Returns:
        A credentials object accepted by ``googleapiclient.discovery.build``.

    Raises:
        DriveError: If no route can produce credentials, with the specific
            remedy for the situation in the message.
    """
    google = _import_google()
    client_secret = DEFAULT_CLIENT_SECRET if client_secret is None else client_secret
    token = DEFAULT_TOKEN if token is None else token
    if service_account is None:
        service_account = DEFAULT_SERVICE_ACCOUNT

    if service_account.exists():
        return google["ServiceAccountCredentials"].from_service_account_file(
            str(service_account), scopes=SCOPES
        )

    creds = None
    if token.exists():
        creds = google["Credentials"].from_authorized_user_file(str(token), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(google["Request"]())
    elif not interactive:
        # The distinction matters for the message: a server with no token at all
        # needs a file copied to it, whereas one whose token cannot refresh has
        # had access revoked and needs re-authorising at the source.
        detail = (
            f"the cached token at {token} cannot be refreshed"
            if creds
            else f"no service-account key at {service_account} and no token at {token}"
        )
        raise DriveError(
            f"Cannot authenticate to Drive without a browser: {detail}.\n"
            "On a headless machine, pick one:\n"
            "  * create a service-account key in Google Cloud Console, save it "
            f"to {service_account}, and share the destination Drive folder with "
            "the service account's e-mail address; or\n"
            "  * run the upload once on a machine with a browser and copy the "
            f"resulting {token.name} to {token}."
        )
    else:
        if not client_secret.exists():
            raise DriveError(
                f"OAuth client secret not found at {client_secret}.\n"
                "Create a 'Desktop app' OAuth client in Google Cloud Console "
                "(with the Drive API enabled), download its JSON, and save it "
                "there (or pass --credentials)."
            )
        flow = google["InstalledAppFlow"].from_client_secrets_file(
            str(client_secret), SCOPES
        )
        creds = flow.run_local_server(port=0)

    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text(creds.to_json(), encoding="utf-8")
    os.chmod(token, 0o600)
    return creds


def _is_transient(exc: Any) -> bool:
    """Return whether an ``HttpError`` is worth retrying.

    A 403 is ambiguous in the Drive API: it covers both quota throttling
    (retry) and hard failures such as a permission error or a full Drive (do
    not retry), so the reason string in the response body decides. A service
    account writing outside a Shared Drive lands here as
    ``storageQuotaExceeded``, which no amount of retrying will fix.
    """
    status = getattr(exc.resp, "status", None)
    if status == 403:
        body = getattr(exc, "content", b"") or b""
        if b"storageQuotaExceeded" in body:
            return False
        return b"ateLimitExceeded" in body or b"quotaExceeded" in body
    return status in RETRY_STATUSES


class DriveClient:
    """Thin, thread-safe-enough wrapper over the Drive v3 files API.

    Thread safety is by construction rather than by lock: the ``service``
    object is built per calling thread, because ``googleapiclient``'s underlying
    ``http`` object is not safe to share. That is what lets an upload pool - or
    the training uploader's background worker running alongside the main
    process - use one client.
    """

    def __init__(self, credentials: Any) -> None:
        """Store credentials and prepare per-thread service objects."""
        self._credentials = credentials
        self._local = threading.local()
        self._folder_cache: dict[tuple[str, str], str] = {}
        self._folder_lock = threading.Lock()
        google = _import_google()
        self._build = google["build"]
        self._http_error = google["HttpError"]
        self._media = google["MediaFileUpload"]
        self._download = google["MediaIoBaseDownload"]

    @property
    def service(self) -> Any:
        """Return a Drive service bound to the calling thread."""
        service = getattr(self._local, "service", None)
        if service is None:
            service = self._build(
                "drive", "v3", credentials=self._credentials, cache_discovery=False
            )
            self._local.service = service
        return service

    def _execute(self, request: Any) -> Any:
        """Execute a Drive request with exponential backoff on transient errors."""
        for attempt in range(MAX_ATTEMPTS):
            try:
                return request.execute()
            except self._http_error as exc:
                if attempt == MAX_ATTEMPTS - 1 or not _is_transient(exc):
                    raise
            except (TimeoutError, ConnectionError, OSError):
                if attempt == MAX_ATTEMPTS - 1:
                    raise
            time.sleep(min(2**attempt, 32) + random.random())
        raise RuntimeError("unreachable")

    def ensure_folder(self, relative_path: str, root_id: str) -> str:
        """Resolve (creating as needed) a ``a/b/c`` folder path under ``root_id``."""
        parent = root_id
        for part in (p for p in relative_path.split("/") if p):
            key = (parent, part)
            with self._folder_lock:
                cached = self._folder_cache.get(key)
            if cached:
                parent = cached
                continue
            parent = self._find_or_create_folder(part, parent)
            with self._folder_lock:
                self._folder_cache[key] = parent
        return parent

    def _find_or_create_folder(self, name: str, parent_id: str) -> str:
        """Return the id of the child folder ``name``, creating it if missing."""
        query = (
            f"name = '{name}' and mimeType = '{FOLDER_MIME}' "
            f"and '{parent_id}' in parents and trashed = false"
        )
        response = self._execute(
            self.service.files().list(
                q=query,
                spaces="drive",
                fields="files(id)",
                pageSize=1,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
        )
        files = response.get("files", [])
        if files:
            return files[0]["id"]
        created = self._execute(
            self.service.files().create(
                body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
                fields="id",
                supportsAllDrives=True,
            )
        )
        return created["id"]

    def find_folder(self, relative_path: str, root_id: str) -> str | None:
        """Resolve a folder path without creating anything.

        The read-only counterpart to :meth:`ensure_folder`, for asking whether a
        previous run left anything behind. Creating the folder in order to find
        out would answer the question but litter Drive with empty directories
        every time a run starts fresh.

        Returns:
            The folder id, or ``None`` if any segment of the path is missing.
        """
        parent = root_id
        for part in (p for p in relative_path.split("/") if p):
            response = self._execute(
                self.service.files().list(
                    q=(
                        f"name = '{part}' and mimeType = '{FOLDER_MIME}' "
                        f"and '{parent}' in parents and trashed = false"
                    ),
                    spaces="drive",
                    fields="files(id)",
                    pageSize=1,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
            )
            files = response.get("files", [])
            if not files:
                return None
            parent = files[0]["id"]
        return parent

    def list_folder(self, folder_id: str) -> dict[str, tuple[str, int]]:
        """Return ``{name: (file_id, size_bytes)}`` for one Drive folder."""
        found: dict[str, tuple[str, int]] = {}
        page_token = None
        while True:
            response = self._execute(
                self.service.files().list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    spaces="drive",
                    fields="nextPageToken, files(id, name, size)",
                    pageSize=1000,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
            )
            for item in response.get("files", []):
                found[item["name"]] = (item["id"], int(item.get("size") or 0))
            page_token = response.get("nextPageToken")
            if not page_token:
                return found

    def list_subfolders(self, folder_id: str) -> dict[str, str]:
        """Return ``{name: folder_id}`` for the child folders of one folder.

        Separate from :meth:`list_folder` because the checkpoint sync asks two
        different questions - which runs exist, and which files a run holds -
        and conflating them would make every caller filter by MIME type.
        """
        found: dict[str, str] = {}
        page_token = None
        while True:
            response = self._execute(
                self.service.files().list(
                    q=(
                        f"'{folder_id}' in parents and trashed = false "
                        f"and mimeType = '{FOLDER_MIME}'"
                    ),
                    spaces="drive",
                    fields="nextPageToken, files(id, name)",
                    pageSize=1000,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
            )
            for item in response.get("files", []):
                found[item["name"]] = item["id"]
            page_token = response.get("nextPageToken")
            if not page_token:
                return found

    def upload(
        self,
        local_path: Path,
        drive_name: str,
        folder_id: str,
        mime_type: str,
        existing_id: str | None,
    ) -> str:
        """Create or update one file in Drive and return its id."""
        size = local_path.stat().st_size
        media = self._media(
            str(local_path),
            mimetype=mime_type,
            resumable=size > RESUMABLE_THRESHOLD,
            chunksize=CHUNK_SIZE,
        )
        if existing_id:
            request = self.service.files().update(
                fileId=existing_id,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
        else:
            request = self.service.files().create(
                body={"name": drive_name, "parents": [folder_id]},
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
        return self._execute(request)["id"]

    def delete(self, file_id: str) -> None:
        """Permanently delete a Drive file or folder.

        Used only to enforce a retention limit on mirrored checkpoints. This
        bypasses the trash rather than moving the item there, because a trashed
        folder still counts against the account's quota - which defeats the
        point of pruning.
        """
        self._execute(self.service.files().delete(fileId=file_id, supportsAllDrives=True))

    def download(self, file_id: str, destination: Path) -> int:
        """Download one Drive file, writing it atomically.

        Written to a ``.part`` sibling and renamed on success, so an interrupted
        transfer cannot leave a half-written file that a later resume would
        mistake for a complete checkpoint shard.

        Args:
            file_id: Drive file id.
            destination: Local path to write. Parent directories are created.

        Returns:
            Bytes written.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        try:
            with partial.open("wb") as handle:
                downloader = self._download(handle, request, chunksize=CHUNK_SIZE)
                done = False
                while not done:
                    # The retry policy differs from _execute's on purpose: the
                    # downloader owns its own request loop, so what is retried
                    # here is one chunk rather than the whole transfer.
                    for attempt in range(MAX_ATTEMPTS):
                        try:
                            _, done = downloader.next_chunk()
                            break
                        except self._http_error as exc:
                            if attempt == MAX_ATTEMPTS - 1 or not _is_transient(exc):
                                raise
                        except (TimeoutError, ConnectionError, OSError):
                            if attempt == MAX_ATTEMPTS - 1:
                                raise
                        time.sleep(min(2**attempt, 32) + random.random())
            partial.replace(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return destination.stat().st_size

    def read_text(self, file_id: str) -> str:
        """Return the contents of a small Drive file as UTF-8 text.

        For the marker files the checkpoint sync writes, which are a few dozen
        bytes and not worth a temporary file.
        """
        buffer = io.BytesIO()
        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        downloader = self._download(buffer, request, chunksize=CHUNK_SIZE)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue().decode("utf-8", errors="replace")
