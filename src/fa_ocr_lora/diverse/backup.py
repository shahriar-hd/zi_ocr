"""Mirror the expensive artifacts to Google Drive.

What gets mirrored, and why it is not everything
------------------------------------------------

The split is by *reproducibility cost*, not by size.

The LLM text cannot be reproduced. It cost API calls and days of wall clock,
and re-generating it would produce different prose even at the same seed
because the endpoint's models change under you. So ``lexicon.parquet``,
``plans.jsonl``, the per-sample texts and ``texts_clean.jsonl`` are mirrored
first and always.

The rendered images *can* be reproduced, exactly, from (text, seed, config) -
that is what the seeded RNG is for. Mirroring 50-200 GB of PNG from the molab
runtime would take longer than re-rendering it, so ``mirror.images`` defaults
to off. The packaged shards are mirrored because they are the deliverable and
because the Hub, while durable, is not a backup you control.

How it uploads
--------------

The text stage is tens of thousands of small JSON files, and Drive's
per-file overhead dominates at that count: uploading 100,000 files one by one
takes hours of round trips for a few hundred megabytes of data. They are
tarred in batches first. The shards are already large, so they go up as they
are.

Everything is idempotent. A remote file of the same name and size is skipped,
so a re-run after an interruption transfers only what is missing.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterator, Sequence
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .config import DiverseConfig, load_diverse_config

__all__ = ["BackupError", "backup", "main"]

LOGGER: Final[logging.Logger] = logging.getLogger("fa_diverse.backup")


#: MIME types by extension. Drive infers nothing useful from the bytes, and a
#: file stored as application/octet-stream cannot be previewed in the web UI.
_MIME: Final[dict[str, str]] = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".parquet": "application/vnd.apache.parquet",
    ".csv": "text/csv",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".md": "text/markdown",
    ".gz": "application/gzip",
    ".yaml": "text/yaml",
}


def _mime_for(name: str) -> str:
    """Return the MIME type for a filename.

    Args:
        name: The filename.

    Returns:
        A MIME type, defaulting to ``application/octet-stream``.
    """
    return _MIME.get(Path(name).suffix.lower(), "application/octet-stream")


class BackupError(RuntimeError):
    """Raised when the mirror cannot proceed."""


@dataclass(frozen=True, slots=True)
class _Item:
    """One thing to upload.

    Attributes:
        local: File on disk.
        remote_dir: Drive folder, relative to the run folder.
        name: Remote filename.
        temporary: Whether ``local`` is a tarball this module built and should
            delete afterwards.
    """

    local: Path
    remote_dir: str
    name: str
    temporary: bool = False


def _batched_tars(directory: Path, batch: int, label: str, work: Path) -> Iterator[_Item]:
    """Tar a tree of many small files into batches.

    Args:
        directory: Tree to pack.
        batch: Files per tarball.
        label: Prefix for the tarball names.
        work: Scratch directory for the tarballs.

    Yields:
        One item per tarball.
    """
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        return
    for index in range(0, len(files), batch):
        chunk = files[index : index + batch]
        target = work / f"{label}_{index // batch:05d}.tar.gz"
        with tarfile.open(target, "w:gz") as archive:
            for path in chunk:
                archive.add(path, arcname=str(path.relative_to(directory)))
        LOGGER.info(
            "  packed %s (%d file(s), %.1f MB)",
            target.name,
            len(chunk),
            target.stat().st_size / 1e6,
        )
        yield _Item(local=target, remote_dir=label, name=target.name, temporary=True)


def _plan(config: DiverseConfig, work: Path) -> list[_Item]:
    """Decide what to mirror, honouring ``drive.mirror``.

    Args:
        config: The active configuration.
        work: Scratch directory for tarballs.

    Returns:
        The items to upload, in priority order - the irreproducible text
        first, so an interrupted mirror has saved the expensive part.
    """
    mirror = config.drive.mirror
    paths = config.paths
    items: list[_Item] = []

    def add_file(path: Path, remote_dir: str) -> None:
        """Queue a single file if it exists."""
        if path.is_file():
            items.append(_Item(local=path, remote_dir=remote_dir, name=path.name))

    if mirror.get("lexicon"):
        add_file(paths.lexicon_parquet, "lexicon")
        add_file(paths.lexicon_report, "lexicon")
    if mirror.get("plans"):
        add_file(paths.plans_jsonl, "planning")
        add_file(paths.coverage_json, "planning")
    if mirror.get("texts_clean"):
        add_file(paths.texts_clean_jsonl, "textqc")
        add_file(paths.rejections_jsonl, "textqc")
    if mirror.get("texts"):
        add_file(paths.texts_jsonl, "generation")
        if paths.texts_dir.is_dir():
            items.extend(
                _batched_tars(paths.texts_dir, config.drive.tar_batch, "texts", work)
            )
    if mirror.get("reports") and paths.reports_dir.is_dir():
        for path in sorted(paths.reports_dir.rglob("*")):
            if path.is_file():
                items.append(
                    _Item(
                        local=path,
                        remote_dir=(
                            f"reports/{path.parent.name}"
                            if path.parent != paths.reports_dir
                            else "reports"
                        ),
                        name=path.name,
                    )
                )
    if mirror.get("shards"):
        root = config.packaging.output_dir
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root).parent.as_posix()
                items.append(
                    _Item(
                        local=path,
                        remote_dir=f"hf/{relative}" if relative != "." else "hf",
                        name=path.name,
                    )
                )
    if mirror.get("images") and paths.images_dir.is_dir():
        # Off by default: these are reproducible from (text, seed, config), and
        # at 100k samples this is 50-200 GB that would take longer to upload
        # than to re-render.
        items.extend(_batched_tars(paths.images_dir, 500, "images", work))
        if paths.annotations_dir.is_dir():
            items.extend(
                _batched_tars(
                    paths.annotations_dir, config.drive.tar_batch, "annotations", work
                )
            )
    return items


def backup(
    config: DiverseConfig, *, dry_run: bool = False, workers: int | None = None
) -> dict[str, Any]:
    """Mirror the configured artifacts to Drive.

    Args:
        config: The active configuration.
        dry_run: Plan and report without uploading.
        workers: Parallel uploads; defaults to ``drive.workers``.

    Returns:
        A report of what was uploaded and skipped.

    Raises:
        BackupError: If Drive is disabled, unreachable, or the folder cannot
            be resolved.
    """
    if not config.drive.enabled:
        raise BackupError("drive.enabled is false; nothing to do")

    with tempfile.TemporaryDirectory(prefix="fa_diverse_backup_") as scratch:
        work = Path(scratch)
        items = _plan(config, work)
        total = sum(item.local.stat().st_size for item in items)

        LOGGER.info("folder      %s/%s", config.drive.folder, config.run_name)
        LOGGER.info("items       %d", len(items))
        LOGGER.info("total       %.1f MB", total / 1e6)
        mirrored = [name for name, on in config.drive.mirror.items() if on]
        LOGGER.info("mirroring   %s", ", ".join(sorted(mirrored)))
        if not config.drive.mirror.get("images"):
            LOGGER.info(
                "            (images excluded: reproducible from text + seed + config)"
            )
        if dry_run:
            by_dir: dict[str, int] = {}
            for item in items:
                by_dir[item.remote_dir] = by_dir.get(item.remote_dir, 0) + 1
            for name, count in sorted(by_dir.items()):
                LOGGER.info("  %-24s %d file(s)", name, count)
            LOGGER.info("")
            LOGGER.info("Dry run: nothing uploaded.")
            return {"items": len(items), "bytes": total, "dry_run": True}

        if not items:
            return {"items": 0, "bytes": 0, "uploaded": 0, "skipped": 0}

        try:
            from ..drive import DriveClient, load_credentials
        except ImportError as error:  # pragma: no cover
            raise BackupError(f"cannot import the Drive client: {error}") from error

        try:
            # interactive=False: a molab runtime has no browser, so a run that
            # needed consent would hang rather than fail in its first second
            # with instructions.
            client = DriveClient(load_credentials(interactive=False))
        except Exception as error:  # noqa: BLE001 - surfaced with a fix hint
            raise BackupError(
                f"cannot authenticate to Drive ({error}). Mint a token with "
                "`python scripts/mint_drive_token.py` on a machine with a browser, "
                "or set GOOGLE_DRIVE_SERVICE_ACCOUNT."
            ) from error

        root_id = config.drive.folder_id or "root"
        try:
            run_folder = client.ensure_folder(
                f"{config.drive.folder}/{config.run_name}", root_id
            )
        except Exception as error:  # noqa: BLE001
            raise BackupError(f"cannot create the Drive folder: {error}") from error

        # Resolve each remote directory once, and list it once, so the skip
        # check costs one API call per directory rather than one per file.
        folders: dict[str, str] = {}
        existing: dict[str, dict[str, tuple[str, int]]] = {}
        for remote_dir in sorted({item.remote_dir for item in items}):
            folder_id = client.ensure_folder(remote_dir, run_folder)
            folders[remote_dir] = folder_id
            try:
                existing[remote_dir] = client.list_folder(folder_id)
            except Exception:  # noqa: BLE001 - an empty listing is safe
                existing[remote_dir] = {}

        worker_count = max(1, workers if workers is not None else config.drive.workers)
        uploaded = 0
        skipped = 0
        failures: list[str] = []
        started = time.monotonic()
        sent_bytes = 0

        def send(item: _Item) -> tuple[_Item, str]:
            """Upload one item unless an identical copy is already there."""
            size = item.local.stat().st_size
            present = existing.get(item.remote_dir, {}).get(item.name)
            if present is not None and present[1] == size:
                return item, "skipped"
            try:
                client.upload(
                    item.local,
                    item.name,
                    folders[item.remote_dir],
                    _mime_for(item.name),
                    present[0] if present is not None else None,
                )
                return item, "uploaded"
            except Exception as error:  # noqa: BLE001 - per-item failure
                return item, f"failed: {str(error)[:160]}"

        with futures.ThreadPoolExecutor(worker_count) as pool:
            for item, outcome in pool.map(send, items):
                if outcome == "uploaded":
                    uploaded += 1
                    sent_bytes += item.local.stat().st_size
                    elapsed = max(1e-6, time.monotonic() - started)
                    if uploaded % 20 == 0:
                        LOGGER.info(
                            "  %d/%d uploaded (%.1f MB/s)",
                            uploaded,
                            len(items),
                            sent_bytes / elapsed / 1e6,
                        )
                elif outcome == "skipped":
                    skipped += 1
                else:
                    failures.append(f"{item.remote_dir}/{item.name}: {outcome}")
                    LOGGER.error("  %s/%s %s", item.remote_dir, item.name, outcome)

    report = {
        "folder": f"{config.drive.folder}/{config.run_name}",
        "items": len(items),
        "uploaded": uploaded,
        "skipped": skipped,
        "failures": failures,
        "bytes": total,
    }
    LOGGER.info("")
    LOGGER.info("=" * 56)
    LOGGER.info("DRIVE BACKUP SUMMARY")
    LOGGER.info("=" * 56)
    LOGGER.info("uploaded  %d", uploaded)
    LOGGER.info("skipped   %d (already present at the same size)", skipped)
    LOGGER.info("failures  %d", len(failures))
    if failures:
        LOGGER.error("Re-run to retry the failures; uploads are idempotent.")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="path to diverse_config.yaml")
    parser.add_argument("--workers", type=int, default=None, help="parallel uploads")
    parser.add_argument("--dry-run", action="store_true", help="plan without uploading")
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="also mirror the rendered images, which are otherwise excluded as "
        "reproducible from text, seed and config",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    from dataclasses import replace

    config = load_diverse_config(args.config)
    if args.include_images:
        config = replace(
            config,
            drive=replace(config.drive, mirror={**config.drive.mirror, "images": True}),
        )
    try:
        report = backup(config, dry_run=args.dry_run, workers=args.workers)
    except BackupError as error:
        LOGGER.error("%s", error)
        return 1
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
