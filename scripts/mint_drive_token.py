#!/usr/bin/env python3
"""Mint the Google Drive refresh token on a machine that has a browser.

The training run itself is headless, so it can never complete the OAuth consent
dance: ``fa_ocr_lora.drive`` only opens a browser when ``interactive=True``, and
``drive.interactive_auth`` is ``false`` in every server config. What a server
needs is the cached token this script produces.

``scripts/upload_to_drive.py --dry-run`` looks like the tool for the job - its
docstring says so - but it returns at line 456, before ``load_credentials()`` is
ever reached, and the non-dry path first requires the local dataset to exist.
This script does only the auth.

Two outputs, both needed:

* ``secrets/drive_token.json`` - the refresh token, chmod 600, which is the path
  the training process reads by default.
* one base64 line - because a JSON body spans several lines and a ``.env`` file
  is line-oriented. Paste that line into marimo's Secrets panel and the
  notebook's secrets cell writes the file back on the server.

Usage::

    python scripts/mint_drive_token.py  # consent + print the .env line
    python scripts/mint_drive_token.py --force  # re-consent, backing up the token
    python scripts/mint_drive_token.py --verify  # also create the Drive root folder
    python scripts/mint_drive_token.py --headless  # report state, never open a browser
    python scripts/mint_drive_token.py --out FILE  # write the line to FILE, not stdout

Exit codes: 0 ok, 1 a ``DriveError``, 2 preflight failed.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fa_ocr_lora.drive import (  # noqa: E402
    DEFAULT_CLIENT_SECRET,
    DEFAULT_SERVICE_ACCOUNT,
    DEFAULT_TOKEN,
    DriveClient,
    DriveError,
    load_credentials,
)

#: Name of the ``.env`` entry the notebook decodes back into a file.
ENV_KEY: Final[str] = "DRIVE_TOKEN_B64"


def _existing(path: Path | None) -> bool:
    """Whether a configured credential path points at a real file.

    Args:
        path: A credential path, or ``None`` when the variable behind it was
            never set.

    Returns:
        ``True`` only for an existing regular file. ``is_file`` rather than
        ``exists`` because an empty ``GOOGLE_DRIVE_*`` variable yields ``.``,
        which exists and would otherwise be mistaken for a key.
    """
    return path is not None and path.is_file()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(
        description="Mint the Drive refresh token and emit it as one .env line."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="back up any cached token and consent again",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="create/resolve the Drive root folder to prove the scope works",
    )
    parser.add_argument(
        "--folder",
        default="fa-ocr-lora",
        help="root folder the run mirrors into (default: fa-ocr-lora)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="never open a browser; fail instead if consent is still required",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="write the .env line to FILE instead of only stdout",
    )
    return parser.parse_args(argv)


def _backup_token() -> Path:
    """Move the cached token aside so a fresh consent can replace it.

    Returns:
        Where the previous token was moved to.
    """
    backup = DEFAULT_TOKEN.with_suffix(".json.bak")
    backup.write_bytes(DEFAULT_TOKEN.read_bytes())
    DEFAULT_TOKEN.unlink()
    return backup


def preflight(args: argparse.Namespace) -> int | None:
    """Reject the two states that silently skip consent, and force a re-consent.

    Args:
        args: Parsed command-line options.

    Returns:
        An exit code to leave the process with, or ``None`` to carry on.
    """
    if _existing(DEFAULT_SERVICE_ACCOUNT):
        print(
            "error: a service-account key already exists at\n"
            f"  {DEFAULT_SERVICE_ACCOUNT}\n"
            "fa_ocr_lora.drive prefers it (drive.py:161), so OAuth consent would be\n"
            "skipped entirely and no token file would be written. Move it aside if\n"
            "you want the personal-My-Drive token; otherwise stop here - a service\n"
            "account needs no encoding, only a folder shared with its e-mail.",
            file=sys.stderr,
        )
        return 2
    if not _existing(DEFAULT_CLIENT_SECRET):
        print(
            "error: OAuth desktop-client JSON not found at\n"
            f"  {DEFAULT_CLIENT_SECRET}\n"
            "Create it once in Google Cloud Console:\n"
            "  1. APIs & Services -> Library -> enable 'Google Drive API'\n"
            "  2. APIs & Services -> OAuth consent screen -> External, then click\n"
            "     'Publish app' (see the note below before you consent)\n"
            "  3. APIs & Services -> Credentials -> Create credentials ->\n"
            "     OAuth client ID -> Application type 'Desktop app' -> download JSON\n"
            "Save that file at the path above, then re-run this script.",
            file=sys.stderr,
        )
        return 2

    print(
        "note: a consent screen left in 'Testing' status issues refresh tokens\n"
        "      that expire after 7 days, which would stop checkpoint mirroring\n"
        "      mid-run without a single error. Click 'Publish app' on the OAuth\n"
        "      consent screen BEFORE consenting below. An 'unverified app' warning\n"
        "      at consent is expected and harmless for a personal-use app.\n",
        flush=True,
    )
    if _existing(DEFAULT_TOKEN) and not args.headless:
        if args.force:
            print(f"backed up the previous token to {_backup_token()}")
        else:
            print(
                f"cached token found at {DEFAULT_TOKEN}; reusing it, so no browser\n"
                "will open. Pass --force to consent again.\n"
            )
    return None


def encode_token() -> tuple[str, dict[str, object]]:
    """Base64 the cached token into one line a ``.env`` file can hold.

    Returns:
        The ``KEY=value`` line, and the token's JSON payload - the latter used
        only to report non-secret metadata such as the granted scope.
    """
    if not _existing(DEFAULT_TOKEN):
        raise DriveError(
            f"no token at {DEFAULT_TOKEN}; the consent flow did not run. On a\n"
            "headless host run this on a machine with a browser first."
        )
    payload: dict[str, object] = json.loads(DEFAULT_TOKEN.read_text(encoding="utf-8"))
    if not payload.get("refresh_token"):
        raise DriveError(
            f"{DEFAULT_TOKEN} has no refresh_token, so it cannot outlive an\n"
            "expired access token. Re-run with --force."
        )
    encoded = base64.b64encode(DEFAULT_TOKEN.read_bytes()).decode("ascii")
    return f"{ENV_KEY}={encoded}", payload


def mint(args: argparse.Namespace) -> object:
    """Return Drive credentials, re-consenting when the cached token is dead.

    ``load_credentials`` refreshes an expired token before it ever considers the
    consent flow, and google.auth answers a refused refresh with a bare
    ``RefreshError`` - a deleted OAuth client, a revoked grant, a password
    change. Left alone that reaches the user as a traceback instead of the
    remedy, so this is where those are translated.

    Args:
        args: Parsed command-line options.

    Returns:
        A credentials object, accepted by ``DriveClient`` and by ``--verify``.

    Raises:
        DriveError: If no route can produce credentials, with the remedy in the
            message.
    """
    try:
        return load_credentials(
            DEFAULT_CLIENT_SECRET,
            DEFAULT_TOKEN,
            interactive=not args.headless,
        )
    except DriveError:
        raise
    except Exception as exc:
        detail = str(exc)
        if "deleted_client" in detail or "invalid_client" in detail:
            raise DriveError(
                "the OAuth client behind the cached token was deleted in Google\n"
                "Cloud Console, so neither that token nor a new consent can work.\n"
                "Create a fresh OAuth client ID (Desktop app), save its JSON over\n"
                f"{DEFAULT_CLIENT_SECRET}, then re-run with --force."
            ) from exc
        if args.headless:
            raise DriveError(
                f"the cached token cannot be refreshed: {detail}\n"
                "Re-run without --headless on a machine with a browser; add --force\n"
                "if the token itself is the problem."
            ) from exc
        print(f"cached token is unusable ({detail})")
        print(f"backed up the previous token to {_backup_token()}")
        return load_credentials(
            DEFAULT_CLIENT_SECRET,
            DEFAULT_TOKEN,
            interactive=True,
        )


def main(argv: list[str] | None = None) -> int:
    """Consent, encode, and print the credential line.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` on a ``DriveError``, ``2`` if preflight failed.
    """
    args = parse_args(argv)
    exit_code = preflight(args)
    if exit_code is not None:
        return exit_code

    try:
        credentials = mint(args)
        line, payload = encode_token()
        folder_id = ""
        if args.verify:
            folder_id = DriveClient(credentials).ensure_folder(args.folder, "root")
    except DriveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"token   : {DEFAULT_TOKEN}")
    print(f"client  : {str(payload.get('client_id', ''))[:16]}...")
    print(f"scopes  : {payload.get('scopes') or 'unknown'}")
    print("refresh : yes" if payload.get("refresh_token") else "refresh : MISSING")
    if folder_id:
        print(f"folder  : {args.folder} -> {folder_id}")

    if args.out is not None:
        args.out.write_text(line + "\n", encoding="utf-8")
        print(f"env line: written to {args.out}")
    else:
        print("\nPaste this into marimo's Secrets panel (it becomes one .env line):")
        print(line)
    print(
        "\nnext: Secrets panel (Ctrl/Cmd+J) -> add the line above -> run the\n"
        "notebook's secrets cell, which writes <repo>/secrets/drive_token.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
