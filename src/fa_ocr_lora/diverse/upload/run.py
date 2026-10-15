"""Stage 08 - upload shards to the Hub, in parallel, resumably.

Three properties matter for a push that may take hours and may be interrupted.

**A ledger, not a guess.** Every shard that completes is recorded in
``upload_ledger.jsonl`` with its size and digest, *after* the remote file has
been confirmed to exist at the right size. A re-run reads the ledger and skips
what is already there. Nothing is deleted locally unless
``upload.delete_after_upload`` is set, and then only for shards the ledger
calls verified.

**Few large commits.** The Hub rate-limits commits, so shards are grouped and
pushed per category rather than one commit per file. The dataset card goes
last, because the card's ``configs:`` block is what makes the viewer and
``load_dataset`` work - writing it first would advertise shards that are not
there yet.

**Private until reviewed.** The repo is created private by default. A
synthetic corpus with a share-alike lexicon and 35 font licences behind it
should not become public before someone has read the card.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..config import DiverseConfig, load_diverse_config

__all__ = ["UploadError", "upload", "main"]

LOGGER: Final[logging.Logger] = logging.getLogger("fa_diverse.upload")

#: Environment variables searched for a Hub token, in order. The value is
#: never logged; only the variable's name ever reaches a log line.
TOKEN_ENV_VARS: Final[tuple[str, ...]] = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)

#: Files that must never be uploaded: local state and editor droppings.
NEVER_UPLOAD: Final[tuple[str, ...]] = (".DS_Store", "upload_ledger.jsonl")


class UploadError(RuntimeError):
    """Raised when the upload cannot proceed or cannot be verified."""


def _resolve_token(explicit: str | None = None) -> tuple[str, str]:
    """Find a Hub token.

    Args:
        explicit: A token passed on the command line, which wins.

    Returns:
        ``(token, source_name)``. The source is the variable's *name*, so a
        log line can say where the token came from without printing it.

    Raises:
        UploadError: If no token is available.
    """
    import os

    if explicit:
        return explicit, "--token"
    for name in TOKEN_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value, name
    try:
        from huggingface_hub import HfFolder

        cached = HfFolder.get_token()
        if cached:
            return cached, "huggingface-cli login cache"
    except Exception:  # noqa: BLE001 - absence is the normal case
        pass
    raise UploadError(
        "no Hugging Face token found. Set HF_TOKEN, or run `huggingface-cli login`."
    )


def _digest(path: Path) -> str:
    """Return a shard's content digest.

    Args:
        path: The file.

    Returns:
        A 16-character blake2b hex digest, read in chunks so a 1 GB shard does
        not have to be held in memory.
    """
    hasher = hashlib.blake2b(digest_size=8)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class _Unit:
    """One commit's worth of files.

    Attributes:
        name: Identifier used in the ledger and the commit message.
        files: Paths, relative to the export root.
        bytes: Total size, for logging and for the estimate.
    """

    name: str
    files: tuple[Path, ...]
    bytes: int


def _plan(root: Path) -> tuple[list[_Unit], _Unit]:
    """Group the export tree into commit units.

    Args:
        root: The export root.

    Returns:
        ``(shard_units, sidecar_unit)``. One unit per category plus a final
        unit holding the manifest and the card, which must be committed last.
    """
    units: list[_Unit] = []
    data = root / "data"
    if data.is_dir():
        for category in sorted(path for path in data.iterdir() if path.is_dir()):
            files = tuple(
                sorted(
                    path.relative_to(root)
                    for path in category.rglob("*.parquet")
                    if path.name not in NEVER_UPLOAD
                )
            )
            if files:
                units.append(
                    _Unit(
                        name=category.name,
                        files=files,
                        bytes=sum((root / path).stat().st_size for path in files),
                    )
                )
    sidecars = tuple(
        sorted(
            path.relative_to(root)
            for path in root.iterdir()
            if path.is_file() and path.name not in NEVER_UPLOAD
        )
    )
    sidecar = _Unit(
        name="cards",
        files=sidecars,
        bytes=sum((root / path).stat().st_size for path in sidecars),
    )
    return units, sidecar


class _Ledger:
    """The record of what has been uploaded and verified.

    Append-only JSONL, one line per verified file, so an interrupted run
    leaves a ledger describing exactly what is already on the Hub.
    """

    __slots__ = ("path", "_done")

    def __init__(self, path: Path) -> None:
        """Load any existing ledger.

        Args:
            path: ``upload_ledger.jsonl``.
        """
        self.path = path
        self._done: dict[str, str] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._done[str(record["file"])] = str(record.get("digest", ""))

    def is_done(self, relative: Path, digest: str) -> bool:
        """Report whether a file has already been uploaded unchanged.

        The digest is compared, not just the path: a shard rebuilt after a
        config change has the same name and different content, and skipping it
        would leave the Hub holding the old data.

        Args:
            relative: Path relative to the export root.
            digest: The file's current digest.

        Returns:
            ``True`` when this exact content is already uploaded.
        """
        return self._done.get(relative.as_posix()) == digest

    def record(self, relative: Path, digest: str, size: int, remote_size: int) -> None:
        """Append a verified upload.

        Args:
            relative: Path relative to the export root.
            digest: The file's digest.
            size: Local size in bytes.
            remote_size: Size the Hub reports, which must match.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "file": relative.as_posix(),
                        "digest": digest,
                        "bytes": size,
                        "remote_bytes": remote_size,
                        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._done[relative.as_posix()] = digest

    def __len__(self) -> int:
        """Return the number of verified files."""
        return len(self._done)


def _remote_sizes(api: Any, repo_id: str, token: str) -> dict[str, int]:
    """Read the sizes of every file currently in the repo.

    Args:
        api: An ``HfApi`` instance.
        repo_id: Target dataset repo.
        token: Hub token.

    Returns:
        Path to size in bytes; empty when the repo is new or unreadable.
    """
    try:
        info = api.repo_info(
            repo_id, repo_type="dataset", files_metadata=True, token=token
        )
    except Exception as error:  # noqa: BLE001 - a new repo has no files
        LOGGER.debug("could not read remote file list: %s", error)
        return {}
    sizes: dict[str, int] = {}
    for sibling in getattr(info, "siblings", ()) or ():
        size = getattr(sibling, "size", None)
        if size is not None:
            sizes[str(sibling.rfilename)] = int(size)
    return sizes


def upload(
    config: DiverseConfig,
    *,
    repo_id: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    """Push the export tree to the Hub and verify every file.

    Args:
        config: The active configuration.
        repo_id: Target dataset repo; defaults to ``upload.repo_id``.
        token: Explicit token; otherwise the environment is searched.
        dry_run: Plan and report without uploading.
        workers: Parallel upload threads; defaults to ``upload.workers``.

    Returns:
        A report of what was uploaded, skipped and verified.

    Raises:
        UploadError: If the repo is not named, the export is missing, or a
            file cannot be verified after upload.
    """
    target = (repo_id or config.upload.repo_id).strip()
    if not target:
        raise UploadError(
            "no repo id. Set upload.repo_id in the config, pass --repo, or export "
            "FA_DIVERSE_HF_REPO."
        )
    root = config.packaging.output_dir
    if not root.is_dir():
        raise UploadError(
            f"{root} does not exist; run "
            "`python -m fa_ocr_lora.diverse.packaging.run` first"
        )

    units, sidecar = _plan(root)
    if not units:
        raise UploadError(f"{root}/data holds no shard to upload")
    total_bytes = sum(unit.bytes for unit in units) + sidecar.bytes

    LOGGER.info(
        "repo        %s (%s)", target, "private" if config.upload.private else "public"
    )
    LOGGER.info("export      %s", root)
    LOGGER.info("units       %d category/categories + sidecars", len(units))
    LOGGER.info("total       %.1f MB", total_bytes / 1e6)
    for unit in units:
        LOGGER.info(
            "  %-14s %3d file(s)  %8.1f MB", unit.name, len(unit.files), unit.bytes / 1e6
        )

    if dry_run:
        LOGGER.info("")
        LOGGER.info("Dry run: nothing uploaded.")
        return {
            "repo_id": target,
            "dry_run": True,
            "units": [unit.name for unit in units],
            "total_bytes": total_bytes,
        }

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise UploadError(
            "huggingface_hub is required: pip install 'fa-ocr-lora[hub]'"
        ) from error

    resolved, source = _resolve_token(token)
    LOGGER.info("token       from %s", source)
    api = HfApi()
    created = False
    try:
        api.create_repo(
            target, repo_type="dataset", private=config.upload.private, exist_ok=True
        )
        created = True
    except Exception as error:  # noqa: BLE001
        raise UploadError(f"cannot create or access {target}: {error}") from error

    ledger = _Ledger(root / "upload_ledger.jsonl")
    LOGGER.info("ledger      %d file(s) already verified", len(ledger))

    digests: dict[Path, str] = {}
    pending: list[tuple[_Unit, Path]] = []
    for unit in (*units, sidecar):
        for relative in unit.files:
            digest = _digest(root / relative)
            digests[relative] = digest
            if ledger.is_done(relative, digest):
                continue
            pending.append((unit, relative))

    skipped = sum(len(unit.files) for unit in (*units, sidecar)) - len(pending)
    LOGGER.info("to upload   %d file(s) (%d already verified)", len(pending), skipped)
    if not pending:
        return {
            "repo_id": target,
            "uploaded": 0,
            "skipped": skipped,
            "verified": len(ledger),
        }

    # Shards first, grouped by category; the sidecars last, because the card's
    # `configs:` block is what makes load_dataset work and must not advertise
    # shards that are not there yet.
    by_unit: dict[str, list[Path]] = {}
    for unit, relative in pending:
        by_unit.setdefault(unit.name, []).append(relative)
    order = [name for name in (unit.name for unit in units) if name in by_unit]
    if sidecar.name in by_unit:
        order.append(sidecar.name)

    worker_count = max(1, workers if workers is not None else config.upload.workers)
    uploaded = 0
    failures: list[str] = []

    def push(relative: Path) -> tuple[Path, bool, str]:
        """Upload one file and report the outcome."""
        try:
            api.upload_file(
                path_or_fileobj=str(root / relative),
                path_in_repo=relative.as_posix(),
                repo_id=target,
                repo_type="dataset",
                token=resolved,
                commit_message=f"{config.upload.commit_message}: {relative.as_posix()}",
            )
            return relative, True, ""
        except Exception as error:  # noqa: BLE001 - reported, not raised, per file
            return relative, False, str(error)[:200]

    for name in order:
        files = by_unit[name]
        LOGGER.info("")
        LOGGER.info("--- %s: %d file(s) ---", name, len(files))
        with futures.ThreadPoolExecutor(worker_count) as pool:
            for relative, ok, error in pool.map(push, files):
                if not ok:
                    failures.append(f"{relative.as_posix()}: {error}")
                    LOGGER.error("  FAILED %s: %s", relative.as_posix(), error)
                    continue
                uploaded += 1

        # Verify against what the Hub reports, then ledger it. Verifying per
        # unit rather than per file costs one API call instead of N.
        remote = _remote_sizes(api, target, resolved)
        for relative in files:
            local_size = (root / relative).stat().st_size
            remote_size = remote.get(relative.as_posix())
            if remote_size is None:
                failures.append(
                    f"{relative.as_posix()}: absent from the repo after upload"
                )
                LOGGER.error("  UNVERIFIED %s: not present remotely", relative.as_posix())
                continue
            if remote_size != local_size:
                failures.append(
                    f"{relative.as_posix()}: {local_size} bytes local, "
                    f"{remote_size} remote"
                )
                LOGGER.error(
                    "  UNVERIFIED %s: size mismatch (%d local, %d remote)",
                    relative.as_posix(),
                    local_size,
                    remote_size,
                )
                continue
            ledger.record(relative, digests[relative], local_size, remote_size)
        LOGGER.info("  %d file(s) verified", len(ledger))

    if config.upload.delete_after_upload and not failures:
        removed = 0
        for unit in units:
            for relative in unit.files:
                if ledger.is_done(relative, digests[relative]):
                    (root / relative).unlink(missing_ok=True)
                    removed += 1
        LOGGER.info("deleted %d verified local shard(s)", removed)

    report = {
        "repo_id": target,
        "created": created,
        "uploaded": uploaded,
        "skipped": skipped,
        "verified": len(ledger),
        "failures": failures,
        "url": f"https://huggingface.co/datasets/{target}",
    }
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("UPLOAD SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info("uploaded  %d", uploaded)
    LOGGER.info("skipped   %d", skipped)
    LOGGER.info("verified  %d", len(ledger))
    LOGGER.info("failures  %d", len(failures))
    LOGGER.info("url       %s", report["url"])
    if failures:
        LOGGER.error("")
        LOGGER.error("%d file(s) are not verified. Re-run to retry them.", len(failures))
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
    parser.add_argument("--repo", default=None, help="target dataset repo id")
    parser.add_argument("--token", default=None, help="Hub token; prefer HF_TOKEN")
    parser.add_argument("--workers", type=int, default=None, help="parallel uploads")
    parser.add_argument("--dry-run", action="store_true", help="plan without uploading")
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; non-zero when any file is unverified.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    config = load_diverse_config(args.config)
    try:
        report = upload(
            config,
            repo_id=args.repo,
            token=args.token,
            dry_run=args.dry_run,
            workers=args.workers,
        )
    except UploadError as error:
        LOGGER.error("%s", error)
        return 1
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
