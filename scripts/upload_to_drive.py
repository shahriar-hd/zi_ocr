"""Upload the Colab-relevant slice of the local dataset to Google Drive.

What gets uploaded (and nothing else):

* ``data/dataset/generate/image/*.png``      -> Drive ``datasets/generate/image/``
* ``data/dataset/generate/json/paddle_farmer.jsonl``
                                             -> Drive ``datasets/generate/json/``
* the plaintext records that were never rendered, bundled into one JSONL
                                             -> Drive ``datasets/text/json/``
* a combined manifest covering both datasets  -> Drive ``datasets/manifest.csv``

Markdown, PDF, annotations and ``paddle_ocr_vl.jsonl`` are deliberately skipped.

Setup (one time):

1.  ``pip install google-api-python-client google-auth-oauthlib``
    (or ``pip install -e '.[drive]'``)
2.  In Google Cloud Console: create/pick a project, enable the **Google Drive
    API**, then create an OAuth 2.0 Client ID of type **Desktop app** and
    download its JSON.
3.  Save it as ``secrets/drive_client_secret.json`` (or point
    ``GOOGLE_DRIVE_CREDENTIALS`` at it in ``.env``, or pass ``--credentials``).
    The first run opens a browser for consent and caches a refresh token in
    ``secrets/drive_token.json`` (or ``GOOGLE_DRIVE_TOKEN``).

Usage::

    python scripts/upload_to_drive.py --dry-run     # plan only, no network
    python scripts/upload_to_drive.py               # full upload
    python scripts/upload_to_drive.py --skip-images # metadata only

The upload is resumable: files already present in the destination folder with
the same name and byte size are skipped, so re-running after an interruption
only transfers what is missing.  Pass ``--force`` to re-upload everything.

Note on scope: the script asks only for ``drive.file`` (per-file access to what
it creates), so it cannot *see* a pre-existing ``datasets`` folder made by hand
in the Drive UI and would create a second one next to it.  Either let it own its
folder tree, or pass ``--drive-root-id <folder id>`` (the id is the last path
segment of the folder's URL in the browser) to upload into an existing folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# The Drive client itself lives in the package, not here, because the training
# loop uploads checkpoints through the same code (see
# fa_ocr_lora.training.checkpoint_sync). A second copy in this script would be
# the one that stops getting fixes.
from fa_ocr_lora.drive import (  # noqa: E402
    DEFAULT_CLIENT_SECRET,
    DEFAULT_TOKEN,
    DriveClient,
    DriveError,
    load_credentials,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Local inputs.
GENERATE_DIR = PROJECT_ROOT / "data" / "dataset" / "generate"
IMAGE_DIR = GENERATE_DIR / "image"
GENERATE_MANIFEST = GENERATE_DIR / "manifest.csv"
PADDLE_JSONL = GENERATE_DIR / "json" / "paddle_farmer.jsonl"
PLAINTEXT_DIR = PROJECT_ROOT / "data" / "text" / "plaintext"

# Locally staged artefacts that are built by this script before being uploaded.
STAGING_DIR = PROJECT_ROOT / "data" / "dataset" / "drive_staging"
PENDING_BUNDLE_NAME = "plaintext_pending.jsonl"
COMBINED_MANIFEST_NAME = "manifest.csv"

# Drive layout, relative to the root folder given by --drive-root.
DRIVE_IMAGE_DIR = "generate/image"
DRIVE_GENERATE_JSON_DIR = "generate/json"
DRIVE_TEXT_JSON_DIR = "text/json"


@dataclass(frozen=True)
class UploadItem:
    """One local file bound for one Drive folder."""

    local_path: Path
    drive_name: str
    drive_dir: str  # "" means the root folder itself
    mime_type: str


# --------------------------------------------------------------------------- #
# Local data collection
# --------------------------------------------------------------------------- #


def read_generate_manifest(path: Path) -> list[dict[str, str]]:
    """Return the rows of the synthetic-dataset manifest CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rendered_source_ids(rows: Iterable[dict[str, str]]) -> set[str]:
    """Return the ``plaintext_*`` ids that have already been rendered."""
    return {row["source_sample_id"] for row in rows if row.get("source_sample_id")}


def pending_plaintext_files(rendered: set[str]) -> list[Path]:
    """Return plaintext records that were never turned into page images."""
    if not PLAINTEXT_DIR.is_dir():
        return []
    return sorted(
        path
        for path in PLAINTEXT_DIR.glob("plaintext_*.json")
        if path.stem not in rendered
    )


def build_pending_bundle(paths: list[Path], out_path: Path) -> tuple[int, int]:
    """Concatenate pending plaintext records into one JSONL file.

    Returns:
        ``(written, skipped)`` record counts.  A record that fails to parse is
        logged and skipped rather than killing the whole pass.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in paths:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  ! skipping {path.name}: {exc}", file=sys.stderr)
                skipped += 1
                continue
            record.setdefault("sample_id", path.stem)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def _to_drive_relative(local_image_path: str) -> str:
    """Map a manifest image path onto its Drive-relative counterpart."""
    return f"{DRIVE_IMAGE_DIR}/{Path(local_image_path).name}"


def build_combined_manifest(
    rows: list[dict[str, str]],
    pending: list[Path],
    out_path: Path,
) -> int:
    """Write one manifest describing both datasets, with Drive-relative paths.

    Every path in the output is relative to the Drive root folder, so a Colab
    notebook can join it with its own Drive mount point without rewriting.

    Returns:
        Number of data rows written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "sample_id",
        "source_sample_id",
        "status",
        "page_count",
        "image_paths",
        "record_path",
        "seed",
        "page_size",
        "dpi",
    ]
    pending_record_path = f"{DRIVE_TEXT_JSON_DIR}/{PENDING_BUNDLE_NAME}"
    generate_record_path = f"{DRIVE_GENERATE_JSON_DIR}/{PADDLE_JSONL.name}"

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        count = 0
        for row in rows:
            images = [p for p in (row.get("image_paths") or "").split(";") if p]
            writer.writerow(
                {
                    "dataset": "generate",
                    "sample_id": row["sample_id"],
                    "source_sample_id": row.get("source_sample_id", ""),
                    "status": "rendered",
                    "page_count": row.get("page_count", len(images)),
                    "image_paths": ";".join(_to_drive_relative(p) for p in images),
                    "record_path": generate_record_path,
                    "seed": row.get("seed", ""),
                    "page_size": row.get("page_size", ""),
                    "dpi": row.get("dpi", ""),
                }
            )
            count += 1
        for path in pending:
            writer.writerow(
                {
                    "dataset": "text",
                    "sample_id": path.stem,
                    "source_sample_id": "",
                    "status": "pending",
                    "page_count": 0,
                    "image_paths": "",
                    "record_path": pending_record_path,
                    "seed": "",
                    "page_size": "",
                    "dpi": "",
                }
            )
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def human_bytes(size: float) -> str:
    """Format a byte count for logs."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def plan_uploads(
    *,
    skip_images: bool,
    image_limit: int | None,
    pending_bundle: Path,
    combined_manifest: Path,
) -> list[UploadItem]:
    """Build the ordered list of files to push, cheapest metadata first."""
    items: list[UploadItem] = [
        UploadItem(
            PADDLE_JSONL, PADDLE_JSONL.name, DRIVE_GENERATE_JSON_DIR, "application/json"
        ),
        UploadItem(
            pending_bundle, PENDING_BUNDLE_NAME, DRIVE_TEXT_JSON_DIR, "application/json"
        ),
        UploadItem(combined_manifest, COMBINED_MANIFEST_NAME, "", "text/csv"),
    ]
    if not skip_images:
        images = sorted(IMAGE_DIR.glob("*.png"))
        if image_limit is not None:
            images = images[:image_limit]
        items += [
            UploadItem(path, path.name, DRIVE_IMAGE_DIR, "image/png") for path in images
        ]
    return items


def _iter_results(
    futures: dict,
) -> Iterator[tuple[UploadItem, str | None, Exception | None]]:
    """Yield ``(item, status, error)`` as uploads finish."""
    for future in as_completed(futures):
        item = futures[future]
        try:
            yield item, future.result(), None
        except Exception as exc:  # noqa: BLE001 - reported per file, never fatal
            yield item, None, exc


def run_uploads(
    client: DriveClient,
    items: list[UploadItem],
    root_id: str,
    *,
    workers: int,
    force: bool,
) -> tuple[int, int, int]:
    """Upload every planned item, skipping ones already present unchanged.

    Returns:
        ``(uploaded, skipped, failed)`` file counts.
    """
    # Resolve every destination folder once, on the main thread.
    folder_ids = {
        drive_dir: client.ensure_folder(drive_dir, root_id)
        for drive_dir in sorted({item.drive_dir for item in items})
    }
    folder_ids[""] = root_id
    existing = {
        drive_dir: client.list_folder(folder_id)
        for drive_dir, folder_id in folder_ids.items()
    }
    for drive_dir, contents in sorted(existing.items()):
        print(f"  remote {drive_dir or '.'}/: {len(contents)} existing file(s)")

    todo: list[tuple[UploadItem, str | None]] = []
    skipped = 0
    for item in items:
        remote = existing[item.drive_dir].get(item.drive_name)
        if remote and not force and remote[1] == item.local_path.stat().st_size:
            skipped += 1
            continue
        todo.append((item, remote[0] if remote else None))

    total_bytes = sum(item.local_path.stat().st_size for item, _ in todo)
    print(
        f"  {len(todo)} file(s) to transfer ({human_bytes(total_bytes)}), "
        f"{skipped} already up to date"
    )
    if not todo:
        return 0, skipped, 0

    uploaded = failed = 0
    done_bytes = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                client.upload,
                item.local_path,
                item.drive_name,
                folder_ids[item.drive_dir],
                item.mime_type,
                existing_id,
            ): item
            for item, existing_id in todo
        }
        for item, _file_id, error in _iter_results(futures):
            if error is not None:
                failed += 1
                print(f"  ! failed {item.drive_name}: {error}", file=sys.stderr)
                continue
            uploaded += 1
            done_bytes += item.local_path.stat().st_size
            if uploaded % 25 == 0 or uploaded == len(todo):
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = done_bytes / elapsed
                print(
                    f"  {uploaded}/{len(todo)} uploaded "
                    f"({human_bytes(done_bytes)}, {human_bytes(rate)}/s)",
                    flush=True,
                )
    return uploaded, skipped, failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--drive-root",
        default="datasets",
        help="root folder path inside My Drive (default: datasets)",
    )
    parser.add_argument(
        "--drive-root-id",
        default=None,
        metavar="ID",
        help=(
            "upload into an existing Drive folder id instead of resolving "
            "--drive-root by name (use this to reuse a folder this script did "
            "not create; the OAuth scope only lets it see its own files)"
        ),
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=DEFAULT_CLIENT_SECRET,
        help=f"OAuth desktop-client JSON (default: {DEFAULT_CLIENT_SECRET})",
    )
    parser.add_argument(
        "--token",
        type=Path,
        default=DEFAULT_TOKEN,
        help=f"cached OAuth token file (default: {DEFAULT_TOKEN})",
    )
    parser.add_argument(
        "--workers", type=int, default=6, help="parallel uploads (default: 6)"
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="upload only the JSONL/manifest metadata (~8 GB of PNGs skipped)",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        metavar="N",
        help="upload only the first N page images (smoke test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-upload files even if an identically sized copy already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stage the bundle/manifest locally and print the plan, no network",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Stage the derived artefacts, then mirror everything into Drive."""
    args = parse_args(argv)

    for required in (GENERATE_MANIFEST, PADDLE_JSONL, IMAGE_DIR):
        if not required.exists():
            print(f"error: missing local input {required}", file=sys.stderr)
            return 1

    rows = read_generate_manifest(GENERATE_MANIFEST)
    rendered = rendered_source_ids(rows)
    pending = pending_plaintext_files(rendered)
    print(
        f"local: {len(rows)} rendered sample(s) from {len(rendered)} plaintext "
        f"record(s); {len(pending)} plaintext record(s) still unrendered"
    )

    pending_bundle = STAGING_DIR / PENDING_BUNDLE_NAME
    written, bad = build_pending_bundle(pending, pending_bundle)
    print(
        f"staged {pending_bundle.relative_to(PROJECT_ROOT)}: {written} record(s)"
        + (f", {bad} unreadable" if bad else "")
    )

    combined_manifest = STAGING_DIR / COMBINED_MANIFEST_NAME
    manifest_rows = build_combined_manifest(rows, pending, combined_manifest)
    print(
        f"staged {combined_manifest.relative_to(PROJECT_ROOT)}: "
        f"{manifest_rows} row(s)"
    )

    items = plan_uploads(
        skip_images=args.skip_images,
        image_limit=args.limit_images,
        pending_bundle=pending_bundle,
        combined_manifest=combined_manifest,
    )
    total = sum(item.local_path.stat().st_size for item in items)
    print(
        f"plan: {len(items)} file(s), {human_bytes(total)} -> "
        f"My Drive/{args.drive_root}/"
    )
    if args.dry_run:
        for item in items[:5]:
            dest = f"{args.drive_root}/{item.drive_dir}".rstrip("/")
            print(f"  {item.local_path.relative_to(PROJECT_ROOT)} -> {dest}/")
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more")
        return 0

    try:
        client = DriveClient(load_credentials(args.credentials, args.token))
    except DriveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    root_id = args.drive_root_id or client.ensure_folder(args.drive_root, "root")
    print(f"drive root folder id: {root_id}")

    uploaded, skipped, failed = run_uploads(
        client, items, root_id, workers=args.workers, force=args.force
    )
    print(f"done: {uploaded} uploaded, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
