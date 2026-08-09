#!/usr/bin/env python3
"""Download the page-level subset of the Persian Pixel OCR dataset.

The `Omarrran/Persian_Pixel <https://huggingface.co/datasets/Omarrran/Persian_Pixel>`_
dataset publishes three render granularities (``sentence``, ``paragraph`` and
``page``) as Parquet shards. This project only needs the page-level renders,
i.e. the rows whose ``sample_type`` column equals ``page``. Those rows live
exclusively in the ``data/page-*.parquet`` shards (56 shards, 31,108 rows,
~12.3 GB), so this script downloads that shard subset and nothing else.

For every downloaded shard the script:

1. verifies row by row that ``sample_type == "page"`` (non-page rows are
   counted and logged, never silently dropped),
2. writes the embedded PNG bytes to ``images/``,
3. appends one record per sample to ``labels.csv``.

The upstream repository ships no CSV, so ``labels.csv`` is built locally from
the Parquet columns. It is the manifest that
``fa_ocr_lora.dataset.real.persian_pixel_filter`` consumes downstream.

Output layout under ``--output-dir`` (default ``data/persian_pixel``)::

    data/persian_pixel/
    |-- hf/
    |   |-- data/page-000NN.parquet     raw shards mirrored from the Hub
    |   `-- metadata/page_summary.json  upstream provenance metadata
    |-- images/image_XXXXXXX.png        extracted page renders
    |-- labels.csv                      image path + transcription + metadata
    |-- state.json                      resume checkpoint
    `-- download_report.json            summary of the last run

Runs are resumable: completed shards are recorded in ``state.json`` and skipped
on the next invocation, and ``labels.csv`` is rewound to its last shard
boundary so an interrupted shard never leaves half-written rows behind.

Examples:
    Smoke test on a single shard::

        python scripts/download_persian_pixel.py --max-shards 1 --limit 20

    Full download::

        python scripts/download_persian_pixel.py

    Full download, reclaiming shard disk space as it goes::

        python scripts/download_persian_pixel.py --delete-parquet-after-extract

Requires ``huggingface_hub`` and ``pyarrow``.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence, TextIO

REPO_ID: Final[str] = "Omarrran/Persian_Pixel"
REPO_TYPE: Final[str] = "dataset"
SAMPLE_TYPE: Final[str] = "page"
SHARD_PATTERN: Final[str] = "data/page-*.parquet"
SUMMARY_FILE: Final[str] = "metadata/page_summary.json"

#: Parquet columns pulled from each shard. ``image`` is a struct of
#: ``{bytes, path}``; the remaining columns are scalar metadata.
PARQUET_COLUMNS: Final[tuple[str, ...]] = (
    "image",
    "bytes",
    "path",
    "text",
    "sample_type",
    "source_run_id",
    "image_path",
    "width",
    "height",
    "text_chars",
    "line_count",
)

#: Column order of the generated ``labels.csv``.
CSV_FIELDNAMES: Final[tuple[str, ...]] = (
    "sample_id",
    "image_path",
    "text",
    "sample_type",
    "source_run_id",
    "source_image_path",
    "width",
    "height",
    "text_chars",
    "line_count",
    "shard",
)

#: Rows decoded from Parquet at a time. Page images average ~110 KB, so a
#: small batch keeps peak memory in the low tens of megabytes.
BATCH_SIZE: Final[int] = 32

DEFAULT_OUTPUT_DIR: Final[str] = "data/persian_pixel"
MIRROR_DIRNAME: Final[str] = "hf"
IMAGES_DIRNAME: Final[str] = "images"
LABELS_FILENAME: Final[str] = "labels.csv"
STATE_FILENAME: Final[str] = "state.json"
REPORT_FILENAME: Final[str] = "download_report.json"

LOGGER: Final[logging.Logger] = logging.getLogger("download_persian_pixel")


class DatasetDownloadError(RuntimeError):
    """Raised when the dataset cannot be downloaded or extracted."""


@dataclass(frozen=True)
class DownloadConfig:
    """Runtime configuration for one download run.

    Attributes:
        output_dir: Directory that receives the mirror, images and CSV.
        revision: Git revision (branch, tag or commit SHA) of the dataset repo.
        limit: Stop after this many extracted samples; ``None`` means no limit.
        max_shards: Process at most this many shards; ``None`` means all.
        extract_images: Whether to write the embedded PNG bytes to disk.
        delete_parquet_after_extract: Delete each shard once it is extracted.
        overwrite: Discard any existing state and start from scratch.
        dry_run: List the shards that would be downloaded, then exit.
    """

    output_dir: Path
    revision: str = "main"
    limit: int | None = None
    max_shards: int | None = None
    extract_images: bool = True
    delete_parquet_after_extract: bool = False
    overwrite: bool = False
    dry_run: bool = False

    @property
    def mirror_dir(self) -> Path:
        """Directory mirroring the Hub repository layout."""
        return self.output_dir / MIRROR_DIRNAME

    @property
    def images_dir(self) -> Path:
        """Directory holding the extracted page images."""
        return self.output_dir / IMAGES_DIRNAME

    @property
    def labels_path(self) -> Path:
        """Path of the generated labels CSV."""
        return self.output_dir / LABELS_FILENAME

    @property
    def state_path(self) -> Path:
        """Path of the resume checkpoint."""
        return self.output_dir / STATE_FILENAME

    @property
    def report_path(self) -> Path:
        """Path of the machine-readable run report."""
        return self.output_dir / REPORT_FILENAME


@dataclass
class RunState:
    """Resume checkpoint persisted between runs.

    Attributes:
        revision: Resolved commit SHA the checkpoint was built from.
        completed_shards: Repo-relative paths of fully extracted shards.
        rows_written: Number of data rows already present in ``labels.csv``.
        csv_bytes: Size of ``labels.csv`` at the last shard boundary.
        skipped_non_page: Rows skipped because ``sample_type != "page"``.
        skipped_no_image: Rows skipped because they carried no image bytes.
    """

    revision: str = ""
    completed_shards: list[str] = field(default_factory=list)
    rows_written: int = 0
    csv_bytes: int = 0
    skipped_non_page: int = 0
    skipped_no_image: int = 0

    @classmethod
    def load(cls, path: Path) -> RunState:
        """Read a checkpoint from disk, returning a blank state if absent.

        Args:
            path: Location of the JSON checkpoint.

        Returns:
            The stored state, or a fresh one if the file is missing or invalid.
        """
        if not path.is_file():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring unreadable state file %s: %s", path, exc)
            return cls()
        if not isinstance(payload, dict):
            LOGGER.warning("Ignoring malformed state file %s", path)
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in payload.items() if key in known})

    def save(self, path: Path) -> None:
        """Atomically write the checkpoint to disk.

        Args:
            path: Location of the JSON checkpoint.
        """
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)


@dataclass(frozen=True)
class ShardResult:
    """Per-shard extraction counters.

    Attributes:
        rows_written: Rows appended to ``labels.csv``.
        skipped_non_page: Rows whose ``sample_type`` was not ``page``.
        skipped_no_image: Rows carrying no usable image bytes.
        completed: Whether the whole shard was consumed (``False`` when the
            sample limit cut it short).
    """

    rows_written: int
    skipped_non_page: int
    skipped_no_image: int
    completed: bool


def configure_logging(verbose: bool) -> None:
    """Set up console logging.

    Args:
        verbose: Emit ``DEBUG`` records instead of ``INFO``.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _require_pyarrow() -> Any:
    """Import ``pyarrow.parquet``, or fail with an actionable message.

    Returns:
        The imported ``pyarrow.parquet`` module.

    Raises:
        DatasetDownloadError: If pyarrow is not installed.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DatasetDownloadError(
            "pyarrow is required to read the dataset shards. Install it with: "
            "conda install -c conda-forge pyarrow"
        ) from exc
    return pq


def _require_hub() -> Any:
    """Import ``huggingface_hub``, or fail with an actionable message.

    Returns:
        The imported ``huggingface_hub`` module.

    Raises:
        DatasetDownloadError: If huggingface_hub is not installed.
    """
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DatasetDownloadError(
            "huggingface_hub is required to reach the Hub. Install it with: "
            "pip install huggingface_hub"
        ) from exc
    return huggingface_hub


def format_size(num_bytes: int) -> str:
    """Render a byte count as a human-readable string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        The size formatted with a binary unit suffix, e.g. ``"1.5 GiB"``.
    """
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def basename_of(raw_path: str) -> str:
    """Return the trailing component of a POSIX- or Windows-style path.

    Args:
        raw_path: Path string coming from the dataset.

    Returns:
        The trailing path component, stripped of separators and whitespace.
    """
    if not raw_path:
        return ""
    normalised = raw_path.replace("\\", "/").rstrip("/")
    return normalised.rsplit("/", 1)[-1].strip()


def safe_image_name(raw_path: str, fallback: str) -> str:
    """Derive a safe local file name from an untrusted dataset path.

    Only the basename is kept, so a ``../`` prefix stored in the dataset cannot
    escape the images directory.

    Args:
        raw_path: Value of the row's ``image_path`` or embedded image path.
        fallback: Name to use when ``raw_path`` yields nothing usable.

    Returns:
        A bare file name with no directory component.
    """
    candidate = basename_of(raw_path)
    if not candidate or candidate in {".", ".."}:
        return fallback
    return candidate


def resolve_revision(revision: str) -> str:
    """Resolve a branch or tag to an immutable commit SHA.

    Pinning the SHA keeps a download reproducible even if the dataset is later
    updated upstream.

    Args:
        revision: Branch, tag or commit SHA to resolve.

    Returns:
        The resolved commit SHA, or ``revision`` itself if resolution fails.
    """
    hub = _require_hub()
    try:
        info = hub.HfApi().dataset_info(REPO_ID, revision=revision)
    except Exception as exc:  # noqa: BLE001 - the hub raises many transport errors
        LOGGER.warning("Could not resolve revision %r: %s", revision, exc)
        return revision
    return info.sha or revision


def list_page_shards(revision: str) -> list[tuple[str, int]]:
    """List the Parquet shards that hold the ``page`` samples.

    Args:
        revision: Dataset revision to inspect.

    Returns:
        ``(repo_path, size_in_bytes)`` pairs sorted by path. The size is ``0``
        when the Hub reports none.

    Raises:
        DatasetDownloadError: If the repository cannot be listed or holds no
            page shards.
    """
    hub = _require_hub()
    try:
        entries = list(
            hub.HfApi().list_repo_tree(
                REPO_ID,
                repo_type=REPO_TYPE,
                revision=revision,
                recursive=True,
            )
        )
    except Exception as exc:  # noqa: BLE001 - the hub raises many transport errors
        raise DatasetDownloadError(
            f"Could not list files of {REPO_ID}@{revision}: {exc}"
        ) from exc

    shards: list[tuple[str, int]] = []
    for entry in entries:
        path = getattr(entry, "path", "")
        if not fnmatch.fnmatch(path, SHARD_PATTERN):
            continue
        lfs = getattr(entry, "lfs", None)
        size = getattr(lfs, "size", None) or getattr(entry, "size", 0) or 0
        shards.append((path, int(size)))

    if not shards:
        raise DatasetDownloadError(
            f"No file matching {SHARD_PATTERN!r} found in {REPO_ID}@{revision}."
        )
    return sorted(shards)


def download_repo_file(repo_path: str, config: DownloadConfig, revision: str) -> Path:
    """Download one file from the dataset repo into the local mirror.

    Files already present are validated against their remote hash and reused,
    so calling this repeatedly is cheap.

    Args:
        repo_path: Repo-relative path, e.g. ``data/page-00000.parquet``.
        config: Active run configuration.
        revision: Dataset revision to download from.

    Returns:
        Path of the local file.

    Raises:
        DatasetDownloadError: If the download fails.
    """
    hub = _require_hub()
    try:
        local_path = hub.hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_path,
            repo_type=REPO_TYPE,
            revision=revision,
            local_dir=str(config.mirror_dir),
        )
    except Exception as exc:  # noqa: BLE001 - the hub raises many transport errors
        raise DatasetDownloadError(
            f"Failed to download {repo_path} from {REPO_ID}@{revision}: {exc}"
        ) from exc
    return Path(local_path)


def _image_payload(row: dict[str, Any]) -> tuple[bytes | None, str]:
    """Extract the encoded image payload from a decoded Parquet row.

    The dataset stores images in a struct column ``image`` of ``{bytes, path}``,
    while some exports keep those as two flat columns; both shapes are handled.

    Args:
        row: Row mapping produced by ``RecordBatch.to_pylist()``.

    Returns:
        The encoded image bytes (or ``None``) and the stored image path.
    """
    image = row.get("image")
    if isinstance(image, dict):
        return image.get("bytes"), str(image.get("path") or "")
    return row.get("bytes"), str(row.get("path") or "")


def extract_shard(
    parquet_path: Path,
    repo_path: str,
    writer: csv.DictWriter,
    config: DownloadConfig,
    seen_names: set[str],
    start_index: int,
    remaining: int | None,
) -> ShardResult:
    """Extract one Parquet shard into image files plus CSV rows.

    Args:
        parquet_path: Local path of the downloaded shard.
        repo_path: Repo-relative path of the shard, recorded in the CSV.
        writer: Open CSV writer positioned at the end of ``labels.csv``.
        config: Active run configuration.
        seen_names: Image file names already used; updated in place so that
            duplicate names across shards receive a unique suffix.
        start_index: Global index of the first row of this shard, used to build
            ``sample_id`` values.
        remaining: Samples still allowed by ``--limit``; ``None`` is unlimited.

    Returns:
        The per-shard counters.

    Raises:
        DatasetDownloadError: If the shard cannot be opened.
    """
    pq = _require_pyarrow()
    try:
        parquet_file = pq.ParquetFile(parquet_path)
    except Exception as exc:  # noqa: BLE001 - pyarrow raises OSError/ArrowInvalid
        raise DatasetDownloadError(
            f"Cannot read Parquet file {parquet_path}: {exc}"
        ) from exc

    available = set(parquet_file.schema_arrow.names)
    columns = [name for name in PARQUET_COLUMNS if name in available]

    rows_written = 0
    skipped_non_page = 0
    skipped_no_image = 0
    completed = True

    try:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=columns):
            for row in batch.to_pylist():
                if remaining is not None and rows_written >= remaining:
                    completed = False
                    break

                sample_type = str(row.get("sample_type") or SAMPLE_TYPE)
                if sample_type != SAMPLE_TYPE:
                    skipped_non_page += 1
                    LOGGER.debug(
                        "Skipping row with sample_type=%r in %s", sample_type, repo_path
                    )
                    continue

                image_bytes, embedded_path = _image_payload(row)
                source_image_path = str(row.get("image_path") or embedded_path)
                index = start_index + rows_written

                if config.extract_images and not image_bytes:
                    skipped_no_image += 1
                    LOGGER.debug("Skipping row %d of %s: no image bytes", index, repo_path)
                    continue

                name = safe_image_name(
                    source_image_path or embedded_path,
                    fallback=f"image_{index:07d}.png",
                )
                if name in seen_names:
                    stem, dot, suffix = name.partition(".")
                    name = f"{stem}_{index:07d}{dot}{suffix}"
                seen_names.add(name)

                if config.extract_images and image_bytes:
                    (config.images_dir / name).write_bytes(image_bytes)

                writer.writerow(
                    {
                        "sample_id": f"persian_pixel_page_{index:07d}",
                        "image_path": f"{IMAGES_DIRNAME}/{name}",
                        "text": row.get("text") or "",
                        "sample_type": sample_type,
                        "source_run_id": row.get("source_run_id") or "",
                        "source_image_path": source_image_path,
                        "width": row.get("width"),
                        "height": row.get("height"),
                        "text_chars": row.get("text_chars"),
                        "line_count": row.get("line_count"),
                        "shard": repo_path,
                    }
                )
                rows_written += 1
            if not completed:
                break
    finally:
        parquet_file.close()

    return ShardResult(rows_written, skipped_non_page, skipped_no_image, completed)


def _prepare_output_dirs(config: DownloadConfig) -> None:
    """Create the output directory tree.

    Args:
        config: Active run configuration.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.mirror_dir.mkdir(parents=True, exist_ok=True)
    if config.extract_images:
        config.images_dir.mkdir(parents=True, exist_ok=True)


def _load_state(config: DownloadConfig, revision: str) -> RunState:
    """Load and validate the resume checkpoint for this revision.

    Args:
        config: Active run configuration.
        revision: Resolved commit SHA of the current run.

    Returns:
        A checkpoint to resume from, or a blank one.

    Raises:
        DatasetDownloadError: If an existing checkpoint targets a different
            revision and ``--overwrite`` was not passed.
    """
    if config.overwrite:
        config.state_path.unlink(missing_ok=True)
        config.labels_path.unlink(missing_ok=True)
        return RunState(revision=revision)

    state = RunState.load(config.state_path)
    if state.revision and state.revision != revision:
        raise DatasetDownloadError(
            f"Existing download in {config.output_dir} targets revision "
            f"{state.revision}, but this run targets {revision}. Re-run with "
            "--overwrite, or pass --revision to match."
        )
    state.revision = revision
    return state


def _open_labels_csv(config: DownloadConfig, state: RunState) -> tuple[TextIO, csv.DictWriter]:
    """Open ``labels.csv`` for appending, rewound to the last checkpoint.

    Truncating to ``state.csv_bytes`` discards rows written by a shard that was
    interrupted mid-way, so the CSV always matches ``state.rows_written``.

    Args:
        config: Active run configuration.
        state: Checkpoint describing how much of the CSV is trustworthy.

    Returns:
        The open file handle and its CSV writer.
    """
    path = config.labels_path
    resume = path.is_file() and state.csv_bytes > 0 and state.rows_written > 0

    if resume:
        with path.open("r+b") as raw:
            raw.truncate(state.csv_bytes)
        handle = path.open("a", encoding="utf-8", newline="")
        return handle, csv.DictWriter(handle, fieldnames=list(CSV_FIELDNAMES))

    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDNAMES))
    writer.writeheader()
    handle.flush()
    state.rows_written = 0
    state.csv_bytes = path.stat().st_size
    return handle, writer


def _collect_seen_names(config: DownloadConfig, state: RunState) -> set[str]:
    """Rebuild the set of already-used image file names after a resume.

    Args:
        config: Active run configuration.
        state: Checkpoint for the current run.

    Returns:
        Image file names already referenced by ``labels.csv``.
    """
    if state.rows_written == 0 or not config.labels_path.is_file():
        return set()
    names: set[str] = set()
    with config.labels_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = basename_of(row.get("image_path") or "")
            if name:
                names.add(name)
    return names


def _write_report(
    config: DownloadConfig,
    state: RunState,
    revision: str,
    shards: Sequence[tuple[str, int]],
    truncated: bool,
) -> None:
    """Write a JSON summary of the run.

    Args:
        config: Active run configuration.
        state: Final checkpoint of the run.
        revision: Resolved commit SHA the data came from.
        shards: All page shards discovered on the Hub.
        truncated: Whether ``--limit`` or ``--max-shards`` cut the run short.
    """
    report = {
        "repo_id": REPO_ID,
        "revision": revision,
        "sample_type": SAMPLE_TYPE,
        "shards_total": len(shards),
        "shards_completed": len(state.completed_shards),
        "rows_written": state.rows_written,
        "skipped_non_page": state.skipped_non_page,
        "skipped_no_image": state.skipped_no_image,
        "images_extracted": config.extract_images,
        "truncated_run": truncated,
        "labels_csv": LABELS_FILENAME,
        "images_dir": IMAGES_DIRNAME,
    }
    config.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def download_persian_pixel_pages(config: DownloadConfig) -> RunState:
    """Download and extract every ``sample_type == "page"`` sample.

    Args:
        config: Active run configuration.

    Returns:
        The final checkpoint, holding the run counters.

    Raises:
        DatasetDownloadError: If the Hub cannot be reached or a shard cannot be
            read.
    """
    revision = resolve_revision(config.revision)
    LOGGER.info("Dataset %s pinned at revision %s", REPO_ID, revision)

    shards = list_page_shards(revision)
    total_bytes = sum(size for _, size in shards)
    LOGGER.info(
        "Found %d page shards (%s) matching %s",
        len(shards),
        format_size(total_bytes),
        SHARD_PATTERN,
    )

    if config.dry_run:
        for repo_path, size in shards:
            LOGGER.info("  %s (%s)", repo_path, format_size(size))
        LOGGER.info("Dry run: nothing downloaded.")
        return RunState(revision=revision)

    _prepare_output_dirs(config)
    state = _load_state(config, revision)

    try:
        summary_path = download_repo_file(SUMMARY_FILE, config, revision)
        LOGGER.info("Fetched upstream metadata %s", summary_path)
    except DatasetDownloadError as exc:
        LOGGER.warning("Optional metadata file unavailable: %s", exc)

    selected = shards if config.max_shards is None else shards[: config.max_shards]
    done = set(state.completed_shards)
    pending = [item for item in selected if item[0] not in done]
    if done:
        LOGGER.info("Resuming: %d of %d shard(s) already done", len(done), len(selected))

    handle, writer = _open_labels_csv(config, state)
    seen_names = _collect_seen_names(config, state)
    truncated = len(selected) < len(shards)

    try:
        for position, (repo_path, size) in enumerate(pending, start=1):
            remaining = None if config.limit is None else config.limit - state.rows_written
            if remaining is not None and remaining <= 0:
                truncated = True
                LOGGER.info("Sample limit of %d reached; stopping.", config.limit)
                break

            LOGGER.info(
                "[%d/%d] %s (%s)", position, len(pending), repo_path, format_size(size)
            )
            local_path = download_repo_file(repo_path, config, revision)
            result = extract_shard(
                parquet_path=local_path,
                repo_path=repo_path,
                writer=writer,
                config=config,
                seen_names=seen_names,
                start_index=state.rows_written,
                remaining=remaining,
            )

            handle.flush()
            state.rows_written += result.rows_written
            state.skipped_non_page += result.skipped_non_page
            state.skipped_no_image += result.skipped_no_image
            state.csv_bytes = config.labels_path.stat().st_size
            if result.completed:
                state.completed_shards.append(repo_path)
            else:
                truncated = True
            state.save(config.state_path)

            LOGGER.info(
                "    +%d rows (total %d)%s",
                result.rows_written,
                state.rows_written,
                "" if result.completed else " [stopped at sample limit]",
            )

            if config.delete_parquet_after_extract and result.completed:
                local_path.unlink(missing_ok=True)
                LOGGER.debug("Deleted shard %s", local_path)

            if not result.completed:
                break
    finally:
        handle.close()

    _write_report(config, state, revision, shards, truncated)
    return state


def parse_args(argv: Sequence[str] | None = None) -> DownloadConfig:
    """Parse command line arguments into a configuration object.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download the sample_type=page subset of the Omarrran/Persian_Pixel "
            "dataset and build a labels CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Destination directory for the mirror, images and labels CSV.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision (branch, tag or commit SHA) to download.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many extracted samples (useful for smoke tests).",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Process at most this many Parquet shards.",
    )
    parser.add_argument(
        "--no-images",
        dest="extract_images",
        action="store_false",
        help="Only build the CSV; do not write PNG files (saves ~3.5 GB).",
    )
    parser.add_argument(
        "--delete-parquet-after-extract",
        action="store_true",
        help="Delete each shard once extracted, to cap peak disk usage.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard any previous state and CSV in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the shards that would be downloaded, then exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    return DownloadConfig(
        output_dir=args.output_dir,
        revision=args.revision,
        limit=args.limit,
        max_shards=args.max_shards,
        extract_images=args.extract_images,
        delete_parquet_after_extract=args.delete_parquet_after_extract,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the downloader from the command line.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` on a handled failure, ``130`` on interrupt.
    """
    config = parse_args(argv)
    try:
        state = download_persian_pixel_pages(config)
    except DatasetDownloadError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; re-run the same command to resume.")
        return 130

    if not config.dry_run:
        LOGGER.info(
            "Done: %d page samples -> %s (skipped %d non-page, %d without image)",
            state.rows_written,
            config.labels_path,
            state.skipped_non_page,
            state.skipped_no_image,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
