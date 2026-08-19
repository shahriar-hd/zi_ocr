"""Pack a rendered corpus into one Parquet family per artifact, with a card.

The rendered tree is 20k loose images beside 20k annotation JSONs, 20k PDFs and
20k Markdown files. That is right for inspecting one sample and wrong for
publishing: a directory of that many small files is slow to copy, cannot be
streamed, and will not drive the Hub's dataset viewer. This module reads the
tree back and writes the transport form:

    <output_dir>/
    |-- README.md                # dataset card; the Hub reads its front matter
    |-- dataset_info.json        # per-family features, per-split counts, the plan
    |-- metadata.csv             # flat per-page index, for a reader without pyarrow
    |-- splits.csv               # sample_id, source_sample_id, split
    |-- attribution/
    |   `-- attribution.csv      # the logo corpus' per-file licences, republished
    `-- data/
        |-- images/       train-00000-of-00040.parquet, test-00000-of-00005.parquet
        |-- plaintext/    train-00000-of-00001.parquet, ...
        |-- markdown/     ...
        |-- pdf/          ...
        |-- annotations/  ...
        |-- metadata/     ...
        `-- .export_state.json   # resume state for every family x split

Six decisions are worth stating up front, because each is a fork that could
reasonably have gone the other way:

* **Every artifact is packed, not just the images.** One family per artifact
  kind, one subdirectory each - and a subdirectory under ``data/`` is what the
  Hub reads as a *config* name, so ``load_dataset(repo, "images")`` and
  ``load_dataset(repo, "pdf")`` both work against the same repository. A
  consumer who wants only pages downloads only pages.
* **The per-sample metadata is repeated on every row of every family.** It is
  about thirty small columns against a 300 KB image, so the duplication is
  free, and it means a consumer who loaded one family never has to join to
  filter by ``has_table`` or ``letter_genre``.
* **The split is a hash of the source letter id, not a shuffle.** Every page of
  one letter lands in one split, and adding plaintext later cannot re-deal the
  samples already exported. :mod:`.splits` carries the reasoning.
* **The image travels inside the row, not beside it.** The ``image`` column is
  ``struct<bytes, path>`` - HuggingFace's ``Image`` feature - so
  ``load_dataset`` hands back decoded pages with no second download and no
  tarball to unpack.
* **``-of-NNNNN`` is not knowable while streaming**, so shards are written under
  a staging name and renamed in one pass at close. A resumed export renames the
  shards a previous run already named, because the total changed.
* **Samples are read from disk once and written to every family.** The exporter
  never holds the corpus in memory: :func:`~.families.load_bundle` reads one
  sample's artifacts, the twelve open writers take what each needs, and it is
  dropped. Peak memory is the sum of the row-group buffers, not one shard.

``pyarrow`` is imported lazily and lives in the optional ``hf`` extra, so the
render pipeline still installs and runs on a machine that will never export.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterator, Mapping, Sequence

from ...config import HuggingFaceConfig, SynthConfig
from ..common.io_utils import ensure_dir, read_manifest
from ..common.schema import AnnotationError, DocumentAnnotation, SCHEMA_VERSION
from .attribution import (
    LOGO_CORPUS,
    NON_COMMERCIAL_NOTICE,
    SIGNATURE_CORPUS,
    SYNTHETIC_NOTICE,
    TEXT_MODEL_FALLBACK,
    LogoAttribution,
    write_logo_attribution,
)
from .errors import ExportError
from .families import (
    FAMILIES,
    HEADER_COLUMNS,
    FamilySpec,
    SampleBundle,
    family_specs,
    load_bundle,
    row_payload_bytes,
)
from .splits import SplitPlan

__all__ = [
    "ExportError",
    "ExportSummary",
    "ExportWriter",
    "METADATA_COLUMNS",
    "SPLITS_COLUMNS",
    "STATE_FILENAME",
    "ShardRecord",
    "ShardWriter",
    "SplitShards",
    "export_dataset",
    "iter_sample_ids",
    "iter_shard_paths",
    "load_annotation",
]

_LOGGER: Final[logging.Logger] = logging.getLogger("fa_ocr_lora.export.huggingface")

#: Resume state, inside ``data/`` beside the shard directories it describes.
#: Leading dot so a glob for ``*.parquet`` - or a Hub upload - passes over it.
STATE_FILENAME: Final[str] = ".export_state.json"

#: Bumped when the state file's shape changes incompatibly. A resume against an
#: older state is refused rather than misread: the previous layout wrote one
#: flat ``data/`` with a single split, and reusing its shard numbering would
#: interleave two different trees.
STATE_VERSION: Final[str] = "2"

#: Columns of ``metadata.csv``: one row per page, every value a string.
#:
#: ``file_name`` is first and named that by HuggingFace's ``imagefolder``
#: convention, but this file is *not* a working imagefolder manifest - the pages
#: live inside Parquet and the corpus is split. It is here as the one view of
#: the corpus a reader without ``pyarrow`` can open, and as something ``awk``
#: can count.
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "file_name",
    "sample_id",
    "source_sample_id",
    "split",
    "page_index",
    "page_count",
    "width",
    "height",
    "dpi",
    "page_size",
    "orientation",
    "texture",
    "seed",
    "page_element_count",
    "has_table",
    "has_chart",
    "has_signature",
    "has_stamp",
    "overflowed",
    "text",
)

#: Columns of ``splits.csv``. Deliberately three: the whole point is that
#: ``cut -d, -f2,3 splits.csv | sort -u | cut -d, -f1 | uniq -d`` prints nothing,
#: which is the group-disjointness claim checked without reading Parquet.
SPLITS_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "source_sample_id",
    "split",
)

#: Dataset-card template, rendered with jinja2 - already a hard dependency for
#: the letter templates, so this adds nothing to the install.
_CARD_TEMPLATE: Final[str] = "dataset_card.md.j2"


def _require_pyarrow() -> tuple[Any, Any]:
    """Import ``pyarrow`` on demand.

    Returns:
        The ``pyarrow`` and ``pyarrow.parquet`` modules.

    Raises:
        ExportError: If ``pyarrow`` is not installed, naming the extra that
            provides it.
    """
    try:
        import pyarrow
        import pyarrow.parquet
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ExportError(
            "the HuggingFace export needs pyarrow; install it with "
            "`pip install -e '.[hf]'`"
        ) from error
    return pyarrow, pyarrow.parquet


# --------------------------------------------------------------------------- #
# Reading the rendered tree back
# --------------------------------------------------------------------------- #


def iter_sample_ids(config: SynthConfig) -> list[str]:
    """List the samples available to export, in render order.

    The render manifest is preferred over globbing the annotations directory
    because it *is* the pipeline's own index: it records what completed, in the
    order it completed, and a sample that failed mid-render never reached it.
    Globbing is the fallback for a tree whose manifest was moved or was never
    written, and sorts by id - which for ``synth_%06d`` is render order anyway.

    Args:
        config: Loaded configuration, for ``paths.output_dir``.

    Returns:
        Render-stage sample ids. Empty when nothing has been rendered.
    """
    manifest = config.paths.render_manifest
    if manifest.is_file():
        rows = read_manifest(manifest)
        seen: dict[str, None] = {}
        for row in rows:
            sample_id = (row.get("sample_id") or "").strip()
            if sample_id:
                # A re-render appends a second row for the same id; the export
                # must not emit the sample twice.
                seen.setdefault(sample_id, None)
        if seen:
            return list(seen)
        _LOGGER.warning("%s has no sample_id values; falling back to a glob", manifest)
    annotations = config.paths.annotations_dir
    if not annotations.is_dir():
        return []
    return sorted(path.stem for path in annotations.glob("*.json"))


def load_annotation(config: SynthConfig, sample_id: str) -> DocumentAnnotation:
    """Read one sample's annotation JSON back into its dataclass.

    Args:
        config: Loaded configuration, for ``paths.annotations_dir``.
        sample_id: Render-stage id.

    Returns:
        The parsed ground truth.

    Raises:
        ExportError: If the file is missing, unreadable, or not a valid
            annotation. The id is in the message, because at 20k samples
            "which one" is the whole question.
    """
    path = config.paths.annotations_dir / f"{sample_id}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ExportError(f"{sample_id}: annotation not found at {path}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError(f"{sample_id}: annotation unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise ExportError(f"{sample_id}: annotation is not a JSON object")
    try:
        return DocumentAnnotation.from_dict(payload)
    except (AnnotationError, ValueError, TypeError) as error:
        raise ExportError(f"{sample_id}: annotation invalid: {error}") from error


# --------------------------------------------------------------------------- #
# What one export produced
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ShardRecord:
    """One written shard, as the resume state records it.

    Attributes:
        index: Zero-based shard number within its ``(family, split)`` cell; the
            final filename is derived from it and the cell's total, so a resumed
            run can rename what a previous run named.
        samples: Samples the shard holds.
        rows: Rows the shard holds - pages for the ``images`` family, samples
            for every other.
        payload_bytes: Uncompressed payload the shard accumulated.
    """

    index: int
    samples: int
    rows: int
    payload_bytes: int

    def as_dict(self) -> dict[str, int]:
        """Return the record as JSON-serialisable state."""
        return {
            "index": self.index,
            "samples": self.samples,
            "rows": self.rows,
            "payload_bytes": self.payload_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShardRecord":
        """Rebuild a record from the state file.

        Args:
            payload: One entry of a cell's ``shards`` list.

        Returns:
            The record.

        Raises:
            ExportError: If a field is missing or not an integer.
        """
        try:
            return cls(
                index=int(payload["index"]),
                samples=int(payload["samples"]),
                rows=int(payload["rows"]),
                payload_bytes=int(payload.get("payload_bytes", 0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ExportError(
                f"export state holds an unreadable shard record {payload!r}: {error}"
            ) from error


@dataclass(slots=True)
class SplitShards:
    """One cell of the family x split grid.

    Attributes:
        family: Family name.
        split: Split name.
        shards: One record per shard, in shard order, including shards a
            previous run wrote.
        names: Final filenames, in shard order, relative to the family's
            directory. Filled at close, once the total is known.
        stepped_down: Whether the byte ceiling forced this cell's batch size
            down to ``min_shard_samples``.
    """

    family: str
    split: str
    shards: list[ShardRecord] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    stepped_down: bool = False

    @property
    def shard_count(self) -> int:
        """How many shards the cell holds."""
        return len(self.shards)

    @property
    def samples(self) -> int:
        """Samples across the cell's shards."""
        return sum(shard.samples for shard in self.shards)

    @property
    def rows(self) -> int:
        """Rows across the cell's shards."""
        return sum(shard.rows for shard in self.shards)

    @property
    def payload_bytes(self) -> int:
        """Uncompressed payload across the cell's shards."""
        return sum(shard.payload_bytes for shard in self.shards)

    def paths(self) -> list[str]:
        """Return the cell's shards as tree-relative paths."""
        return [f"data/{self.family}/{name}" for name in self.names]

    def as_dict(self) -> dict[str, Any]:
        """Return the cell as JSON-serialisable state.

        Filenames are left out on purpose: they are a function of the index and
        the total, and storing them would give a resume two sources of truth to
        disagree about.
        """
        return {
            "shards": [shard.as_dict() for shard in self.shards],
            "stepped_down": self.stepped_down,
        }


@dataclass(slots=True)
class ExportSummary:
    """What one export produced, across every family and split.

    Attributes:
        output_dir: The export tree's root.
        grid: ``family -> split -> cell``, in registry and split order.
        assignments: ``sample_id -> split`` for the whole tree, previous runs
            included. Doubles as the resume skip-set.
        include_elements: Whether the ``images`` rows carry element lists.
        new_samples: Samples this run added.
        skipped_samples: Samples this run skipped as already exported.
        missing: ``family -> samples whose artifact was absent``, so a family
            that came up short is a number in the summary rather than a silence.
        text_models: Model ids credited with drafting the letter bodies, as
            observed in the source records.
        logos: What the republished logo licence manifest covered.
    """

    output_dir: Path
    grid: dict[str, dict[str, SplitShards]] = field(default_factory=dict)
    assignments: dict[str, str] = field(default_factory=dict)
    include_elements: bool = True
    new_samples: int = 0
    skipped_samples: int = 0
    missing: dict[str, int] = field(default_factory=dict)
    text_models: tuple[str, ...] = ()
    logos: LogoAttribution = field(default_factory=LogoAttribution)

    @property
    def families(self) -> tuple[str, ...]:
        """Families this export wrote, in registry order."""
        return tuple(self.grid)

    @property
    def splits(self) -> tuple[str, ...]:
        """Splits this export wrote, in plan order."""
        for cells in self.grid.values():
            return tuple(cells)
        return ()

    @property
    def samples(self) -> int:
        """Samples in the export tree, previous runs included."""
        return len(self.assignments)

    @property
    def shard_count(self) -> int:
        """Shards across every family and split."""
        return sum(
            cell.shard_count for cells in self.grid.values() for cell in cells.values()
        )

    @property
    def rows(self) -> int:
        """Rows across every family and split."""
        return sum(cell.rows for cells in self.grid.values() for cell in cells.values())

    @property
    def payload_bytes(self) -> int:
        """Uncompressed payload across every family and split."""
        return sum(
            cell.payload_bytes
            for cells in self.grid.values()
            for cell in cells.values()
        )

    @property
    def stepped_down(self) -> bool:
        """Whether any cell's batch size fell to ``min_shard_samples``."""
        return any(
            cell.stepped_down
            for cells in self.grid.values()
            for cell in cells.values()
        )

    def split_samples(self) -> dict[str, int]:
        """Return how many samples each split holds, from the assignments."""
        counts = {split: 0 for split in self.splits}
        for split in self.assignments.values():
            counts[split] = counts.get(split, 0) + 1
        return counts

    def cell(self, family: str, split: str) -> SplitShards:
        """Return one cell of the grid.

        Args:
            family: Family name.
            split: Split name.

        Returns:
            The cell, or an empty one when the pair produced nothing.
        """
        return self.grid.get(family, {}).get(
            split, SplitShards(family=family, split=split)
        )

    def shard_paths(self) -> list[str]:
        """Return every shard as a tree-relative path, in grid order."""
        return [
            path
            for cells in self.grid.values()
            for cell in cells.values()
            for path in cell.paths()
        ]


# --------------------------------------------------------------------------- #
# Writing one family's shards for one split
# --------------------------------------------------------------------------- #


class ShardWriter:
    """Streams one family's rows for one split into Parquet shards.

    Three limits close a shard, and which one fired matters.

    * Reaching ``shard_samples`` is the normal case - but only once the shard
      also holds ``min_shard_bytes`` of payload. Without that second condition
      the blob-free families would shard on the same sample count as the images
      and become forty 200 KB files.
    * Reaching ``max_shard_bytes`` means a full batch does not fit, which is the
      signal to step the batch size down to ``min_shard_samples`` for everything
      after - once, and logged.
    * :meth:`close` closes whatever is left.

    A sample's rows are always written together. If one sample alone exceeds the
    ceiling it still gets a shard to itself, with a warning: splitting it would
    break the batch unit the request is stated in.

    Nothing is renamed to its ``-of-NNNNN`` name until close, because the total
    is not known before then.
    """

    def __init__(
        self,
        settings: HuggingFaceConfig,
        spec: FamilySpec,
        split: str,
        *,
        include_elements: bool,
        restored: SplitShards | None = None,
    ) -> None:
        """Open a writer over one family's directory.

        Args:
            settings: The ``huggingface`` config section.
            spec: The family being written; supplies the schema and the
                row-group budget.
            split: Split this writer holds.
            include_elements: Whether the conditional columns are present.
            restored: The cell a previous run left behind, when resuming.

        Raises:
            ExportError: If ``pyarrow`` is not installed.
        """
        self._pyarrow, self._parquet = _require_pyarrow()
        self._settings = settings
        self._spec = spec
        self._split = split
        self._include_elements = include_elements
        self._schema = spec.schema(self._pyarrow, include_elements=include_elements)
        self._dir = settings.family_dir(spec.name)

        cell = restored or SplitShards(family=spec.name, split=split)
        self._shards: list[ShardRecord] = list(cell.shards)
        self._stepped_down = cell.stepped_down
        self._batch_size = (
            settings.min_shard_samples
            if self._stepped_down
            else settings.shard_samples
        )

        self._writer: Any = None
        self._shard_path: Path | None = None
        self._shard_samples = 0
        self._shard_rows = 0
        self._shard_bytes = 0
        self._buffer: list[dict[str, Any]] = []
        self._buffer_bytes = 0

    @property
    def label(self) -> str:
        """Identify the cell in a log line."""
        return f"{self._spec.name}/{self._split}"

    # -- shards ------------------------------------------------------------ #

    def _open_shard(self) -> None:
        """Start a new shard under its staging name."""
        index = len(self._shards)
        ensure_dir(self._dir)
        self._shard_path = self._dir / self._settings.staging_name(self._split, index)
        self._writer = self._parquet.ParquetWriter(
            self._shard_path, self._schema, compression=self._settings.compression
        )
        self._shard_samples = 0
        self._shard_rows = 0
        self._shard_bytes = 0

    def _flush_buffer(self) -> None:
        """Write the buffered rows as one row group."""
        if not self._buffer:
            return
        batch = self._pyarrow.RecordBatch.from_pylist(self._buffer, schema=self._schema)
        self._writer.write_batch(batch)
        self._buffer.clear()
        self._buffer_bytes = 0

    def _close_shard(self, *, on_ceiling: bool) -> None:
        """Finish the open shard and record it.

        Args:
            on_ceiling: Whether the byte ceiling closed it rather than the batch
                size. A ceiling close on a short batch is what triggers the
                step-down.
        """
        if self._writer is None:
            return
        self._flush_buffer()
        self._writer.close()
        self._writer = None
        self._shard_path = None
        self._shards.append(
            ShardRecord(
                index=len(self._shards),
                samples=self._shard_samples,
                rows=self._shard_rows,
                payload_bytes=self._shard_bytes,
            )
        )
        _LOGGER.info(
            "%s shard %d closed: %d sample(s), %d row(s), %.1f MiB payload%s",
            self.label,
            len(self._shards) - 1,
            self._shard_samples,
            self._shard_rows,
            self._shard_bytes / 1024**2,
            " (byte ceiling)" if on_ceiling else "",
        )
        if (
            on_ceiling
            and not self._stepped_down
            and self._shard_samples < self._batch_size
            and self._settings.min_shard_samples < self._batch_size
        ):
            _LOGGER.warning(
                "%s: a %d-sample shard does not fit under %.2f GiB (shard %d held "
                "%d); stepping the batch size down to %d for the rest of the export",
                self.label,
                self._batch_size,
                self._settings.max_shard_bytes / 1024**3,
                len(self._shards) - 1,
                self._shard_samples,
                self._settings.min_shard_samples,
            )
            self._batch_size = self._settings.min_shard_samples
            self._stepped_down = True

    def _find_shard_file(self, index: int) -> Path | None:
        """Locate shard ``index`` under either its staging or a final name.

        Args:
            index: Zero-based shard number.

        Returns:
            The file, or ``None`` when it is absent.
        """
        staging = self._dir / self._settings.staging_name(self._split, index)
        if staging.is_file():
            return staging
        for candidate in sorted(
            self._dir.glob(f"{self._split}-{index:05d}-of-*.parquet")
        ):
            return candidate
        return None

    def _rename_shards(self) -> list[str]:
        """Give every shard its ``-of-NNNNN`` name.

        Runs over all shards, not just this run's: a resumed export changes the
        total, so shards a previous run named ``-of-00003`` have to become
        ``-of-00005``.

        Returns:
            The final filenames, in shard order.

        Raises:
            ExportError: If a recorded shard's file cannot be found.
        """
        total = len(self._shards)
        names: list[str] = []
        for shard in self._shards:
            final = self._settings.shard_name(self._split, shard.index, total)
            names.append(final)
            target = self._dir / final
            if target.is_file():
                continue
            source = self._find_shard_file(shard.index)
            if source is None:
                raise ExportError(
                    f"{self.label}: shard {shard.index} is recorded in the export "
                    f"state but its file is missing from {self._dir}; delete "
                    f"{self._settings.data_dir / STATE_FILENAME} to restart the "
                    "export"
                )
            source.rename(target)
        return names

    # -- public API -------------------------------------------------------- #

    def add_sample(self, sample_id: str, rows: Sequence[dict[str, Any]]) -> None:
        """Append one sample's rows, opening or closing shards as needed.

        Args:
            sample_id: Render-stage id, for the log line only - the assignment
                is recorded by :class:`ExportWriter`.
            rows: The sample's rows for this family. An empty sequence is
                ignored, which is how a family whose artifact was missing skips
                the sample without skipping it everywhere.
        """
        if not rows:
            return
        payload = sum(row_payload_bytes(row) for row in rows)
        if self._writer is None:
            self._open_shard()
        elif self._shard_bytes + payload > self._settings.max_shard_bytes:
            self._close_shard(on_ceiling=True)
            self._open_shard()
        if payload > self._settings.max_shard_bytes:
            _LOGGER.warning(
                "%s: %s alone is %.2f GiB, over the %.2f GiB ceiling; it gets a "
                "shard of its own rather than being split across two",
                self.label,
                sample_id,
                payload / 1024**3,
                self._settings.max_shard_bytes / 1024**3,
            )
        self._buffer.extend(rows)
        self._buffer_bytes += payload
        self._shard_samples += 1
        self._shard_rows += len(rows)
        self._shard_bytes += payload
        if self._buffer_bytes >= self._spec.row_group_bytes:
            self._flush_buffer()
        if (
            self._shard_samples >= self._batch_size
            and self._shard_bytes >= self._settings.min_shard_bytes
        ):
            self._close_shard(on_ceiling=False)

    def close(self) -> SplitShards:
        """Close the open shard and rename every shard in the cell.

        Returns:
            The cell, with its final filenames.
        """
        self._close_shard(on_ceiling=False)
        names = self._rename_shards()
        return SplitShards(
            family=self._spec.name,
            split=self._split,
            shards=list(self._shards),
            names=names,
            stepped_down=self._stepped_down,
        )

    def abort(self) -> None:
        """Discard a shard abandoned mid-write.

        A Parquet file whose footer was never written is not readable, so on an
        error the staging file is removed rather than left to be renamed into
        the tree on the next resume.
        """
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        if self._shard_path is not None and self._shard_path.is_file():
            self._shard_path.unlink()
            _LOGGER.warning(
                "%s: export failed mid-shard; discarded %s",
                self.label,
                self._shard_path,
            )
        self._shard_path = None


# --------------------------------------------------------------------------- #
# Writing the whole grid
# --------------------------------------------------------------------------- #


class ExportWriter:
    """Owns one :class:`ShardWriter` per family and split, and routes samples.

    Twelve writers - six families, two splits - are open at once, which is why
    the row-group budget is per family rather than one global constant: the sum
    of the budgets is the process's peak buffer, roughly twice that at flush.

    The resume state lives in one file for the whole grid rather than one per
    cell, because the thing that must not drift is the *assignment*: a sample
    recorded as ``train`` has to be ``train`` in all six families, and a single
    ``sample_id -> split`` map is the only representation of that which cannot
    be half-updated.

    Use as a context manager, or call :meth:`close` yourself.
    """

    def __init__(
        self,
        settings: HuggingFaceConfig,
        plan: SplitPlan,
        *,
        families: Sequence[str] | None = None,
        include_elements: bool | None = None,
        resume: bool = False,
    ) -> None:
        """Open the grid over ``settings.data_dir``.

        Args:
            settings: The ``huggingface`` config section.
            plan: The split plan; its fingerprint is what a resume is checked
                against.
            families: Families to write. Defaults to ``settings.families``.
            include_elements: Override for ``settings.include_elements``.
            resume: Continue an existing export, keeping its shards and skipping
                the samples they already hold.

        Raises:
            ExportError: If ``pyarrow`` is missing, a family name is unknown, or
                a resumed export's state disagrees with this run.
        """
        self._settings = settings
        self._plan = plan
        self._include_elements = (
            settings.include_elements if include_elements is None else include_elements
        )
        self._specs = family_specs(
            settings.families if families is None else families
        )
        self._splits = plan.names()
        self._needs = frozenset(
            need for spec in self._specs for need in spec.needs
        )
        self._family_names = tuple(spec.name for spec in self._specs)
        self._state_path = ensure_dir(settings.data_dir) / STATE_FILENAME

        self._assigned: dict[str, str] = {}
        self._restored: dict[tuple[str, str], SplitShards] = {}
        self._models: set[str] = set()
        if resume:
            self._load_state()
        self._warn_stale_layout()

        self._writers: dict[tuple[str, str], ShardWriter] = {
            (spec.name, split): ShardWriter(
                settings,
                spec,
                split,
                include_elements=self._include_elements,
                restored=self._restored.get((spec.name, split)),
            )
            for spec in self._specs
            for split in self._splits
        }
        self._missing: dict[str, int] = {spec.name: 0 for spec in self._specs}
        self._new_samples = 0
        self._skipped = 0

    # -- state ------------------------------------------------------------- #

    def _load_state(self) -> None:
        """Restore the assignment map and every cell's shard numbering.

        A missing state file is not an error: it is what a first run looks like,
        and ``--resume`` on a fresh tree should start rather than complain.

        Raises:
            ExportError: If the state is unreadable, or was written by an export
                whose split plan, element flag or family set differs from this
                run's. Appending under a changed plan is exactly the leak the
                split exists to prevent, so it is refused rather than warned
                about.
        """
        if not self._state_path.is_file():
            _LOGGER.info(
                "no export state at %s; starting a new export", self._state_path
            )
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ExportError(
                f"{self._state_path}: export state unreadable: {error}; delete it "
                "to restart the export from shard 0"
            ) from error
        if not isinstance(payload, dict):
            raise ExportError(f"{self._state_path}: export state is not an object")

        version = str(payload.get("state_version", "1"))
        if version != STATE_VERSION:
            raise ExportError(
                f"{self._state_path}: export state is version {version}, this "
                f"build writes {STATE_VERSION}; the layout changed, so export "
                "into a fresh output_dir rather than resuming into this one"
            )

        recorded = payload.get("split_plan") or {}
        fingerprint = str(recorded.get("fingerprint", ""))
        if fingerprint != self._plan.fingerprint():
            raise ExportError(
                f"{self._state_path}: the existing export was split as "
                f"{_describe_stored_plan(recorded)} "
                f"(fingerprint {fingerprint[:12] or 'absent'}), this run asks for "
                f"{self._plan.describe()} (fingerprint "
                f"{self._plan.fingerprint()[:12]}). Resuming would deal the new "
                "samples by a different rule than the ones already written, "
                "which puts pages of one letter on both sides of the split. "
                "Restore the previous splits/group_by/split_salt, or export into "
                "a fresh output_dir."
            )

        if bool(payload.get("include_elements", True)) != self._include_elements:
            raise ExportError(
                f"{self._state_path}: the existing export was written with "
                f"include_elements={payload.get('include_elements')!r}, this run "
                f"asks for {self._include_elements!r}; the images schema differs "
                "between the two, so the shards would not concatenate. Match the "
                "flag or export into a fresh output_dir."
            )

        previous = tuple(str(name) for name in payload.get("families", ()))
        if set(previous) != set(self._family_names):
            raise ExportError(
                f"{self._state_path}: the existing export holds families "
                f"{list(previous)}, this run asks for {list(self._family_names)}. "
                "A resume only writes the samples not already recorded, so the "
                "added family would cover just the new ones. Export the full "
                "family set into a fresh output_dir instead."
            )

        for family, cells in (payload.get("shards") or {}).items():
            for split, cell in (cells or {}).items():
                self._restored[(str(family), str(split))] = SplitShards(
                    family=str(family),
                    split=str(split),
                    shards=[
                        ShardRecord.from_dict(item)
                        for item in (cell or {}).get("shards", ())
                    ],
                    stepped_down=bool((cell or {}).get("stepped_down", False)),
                )
        samples = payload.get("samples") or {}
        if not isinstance(samples, dict):
            raise ExportError(
                f"{self._state_path}: 'samples' must map sample_id to split"
            )
        self._assigned = {str(key): str(value) for key, value in samples.items()}
        self._models = {str(name) for name in payload.get("text_models", ()) if name}
        _LOGGER.info(
            "resuming export: %d sample(s) already written across %d cell(s)",
            len(self._assigned),
            len(self._restored),
        )

    def _write_state(self, grid: Mapping[str, Mapping[str, SplitShards]]) -> None:
        """Persist the assignment map and every cell's shard numbering.

        Args:
            grid: The closed cells, keyed family then split.
        """
        plan = self._plan.as_dict()
        payload = {
            "state_version": STATE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "include_elements": self._include_elements,
            "families": list(self._family_names),
            "split_plan": plan,
            "text_models": sorted(self._models),
            "shards": {
                family: {split: cell.as_dict() for split, cell in cells.items()}
                for family, cells in grid.items()
            },
            "samples": dict(sorted(self._assigned.items())),
        }
        self._state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _warn_stale_layout(self) -> None:
        """Warn about shards left flat in ``data/`` by the pre-family exporter.

        The card points the Hub at ``data/<family>/``, so stray files one level
        up are not loaded and not overwritten - they just sit there looking like
        part of the dataset. Saying so once is cheaper than a consumer finding
        them.
        """
        stale = sorted(self._settings.data_dir.glob("*.parquet"))
        if stale:
            _LOGGER.warning(
                "%d Parquet file(s) sit directly in %s, where an older export "
                "wrote them; this export writes data/<family>/ and will neither "
                "read nor replace them. Delete them once you have checked "
                "nothing depends on them: %s",
                len(stale),
                self._settings.data_dir,
                ", ".join(path.name for path in stale[:3])
                + (", ..." if len(stale) > 3 else ""),
            )

    # -- public API -------------------------------------------------------- #

    @property
    def already_exported(self) -> frozenset[str]:
        """Samples a previous run already wrote, when resuming."""
        return frozenset(self._assigned)

    @property
    def needs(self) -> frozenset[str]:
        """Which artifacts :func:`~.families.load_bundle` has to read."""
        return self._needs

    @property
    def include_elements(self) -> bool:
        """Whether the ``images`` rows carry element lists."""
        return self._include_elements

    def skip(self, sample_id: str) -> None:
        """Record that a sample was skipped as already exported.

        Args:
            sample_id: The sample not written.
        """
        self._skipped += 1
        _LOGGER.debug("%s already exported; skipping", sample_id)

    def add(self, bundle: SampleBundle) -> None:
        """Write one sample's rows into every family that has any.

        Args:
            bundle: The sample's artifacts, already routed to a split.

        Raises:
            ExportError: If the bundle's split is not part of the plan, which
                would mean the router and the writer disagree.
        """
        if bundle.split not in self._splits:
            raise ExportError(
                f"{bundle.sample_id}: assigned to split {bundle.split!r}, which "
                f"this export does not write ({list(self._splits)})"
            )
        for spec in self._specs:
            rows = spec.build_rows(bundle, include_elements=self._include_elements)
            self._writers[(spec.name, bundle.split)].add_sample(
                bundle.sample_id, rows
            )
        for family in bundle.missing:
            if family in self._missing:
                self._missing[family] += 1
        model = _text_model(bundle)
        if model:
            self._models.add(model)
        self._assigned[bundle.sample_id] = bundle.split
        self._new_samples += 1

    def close(self) -> ExportSummary:
        """Close every cell, rename its shards, and persist the state.

        Returns:
            What the export tree now holds, previous runs included.
        """
        grid: dict[str, dict[str, SplitShards]] = {}
        for spec in self._specs:
            cells: dict[str, SplitShards] = {}
            for split in self._splits:
                cell = self._writers[(spec.name, split)].close()
                cells[split] = cell
                if not cell.shards:
                    _LOGGER.warning(
                        "%s/%s holds no shards; the dataset card will omit that "
                        "split for this config, because a config whose data_files "
                        "glob matches nothing fails to load",
                        spec.name,
                        split,
                    )
            grid[spec.name] = cells
        self._write_state(grid)
        models = tuple(sorted(self._models)) or (TEXT_MODEL_FALLBACK,)
        return ExportSummary(
            output_dir=self._settings.output_dir,
            grid=grid,
            assignments=dict(self._assigned),
            include_elements=self._include_elements,
            new_samples=self._new_samples,
            skipped_samples=self._skipped,
            missing=dict(self._missing),
            text_models=models,
        )

    def abort(self) -> None:
        """Discard every partially written shard."""
        for writer in self._writers.values():
            writer.abort()

    def __enter__(self) -> "ExportWriter":
        """Return the writer, for ``with`` use."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Discard partial shards on an exception; otherwise leave the close to
        the caller, which needs the summary :meth:`close` returns.
        """
        if exc_type is not None:
            self.abort()


def _text_model(bundle: SampleBundle) -> str:
    """Read which model drafted this sample's letter body.

    Args:
        bundle: The sample's artifacts.

    Returns:
        The model id, or an empty string when the source record did not carry
        one - an older plaintext record, or a sample whose source was missing.
    """
    raw = bundle.header.get("generation") or ""
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("model") or "")


# --------------------------------------------------------------------------- #
# Sidecar files
# --------------------------------------------------------------------------- #


def _describe_stored_plan(recorded: Mapping[str, Any]) -> str:
    """Summarise a split plan read back from the state file.

    Used only inside an error message, so it must not raise: a state file
    corrupt enough that the plan will not parse is exactly when the mismatch
    message needs to still come out.

    Args:
        recorded: The state file's ``split_plan`` block.

    Returns:
        A one-line summary, or a raw repr when the block cannot be parsed.
    """
    try:
        return SplitPlan.from_dict(recorded).describe()
    except Exception:  # noqa: BLE001 - any parse failure falls back to the repr
        return repr(dict(recorded))


def _requested_fraction(settings: HuggingFaceConfig, split: str) -> float:
    """Return the fraction the config asked for, or ``0.0`` if it did not.

    A split can be present in the state file's assignments and absent from the
    current config only if the state was hand-edited past the fingerprint guard.
    Reporting it as ``0.0`` keeps ``dataset_info.json`` writable so the number
    is visible, rather than failing the export over a field nothing reads.

    Args:
        settings: The ``huggingface`` config section.
        split: Split name.

    Returns:
        The requested fraction.
    """
    try:
        return settings.fraction_of(split)
    except KeyError:
        _LOGGER.warning(
            "split %r is not in the configured plan; reporting fraction 0.0", split
        )
        return 0.0


def _metadata_rows(bundle: SampleBundle) -> list[dict[str, str]]:
    """Build one ``metadata.csv`` row per page.

    Args:
        bundle: The sample's artifacts.

    Returns:
        Rows keyed by :data:`METADATA_COLUMNS`, every value a string. Note that
        ``page_element_count`` counts the page's own elements, where the Parquet
        header's ``element_count`` counts the whole sample's - two different
        numbers, hence two different names.
    """
    annotation = bundle.annotation
    header = bundle.header
    page_count = len(annotation.pages)
    rows: list[dict[str, str]] = []
    for page in annotation.pages:
        rows.append(
            {
                "file_name": page.image_path,
                "sample_id": annotation.sample_id,
                "source_sample_id": annotation.source_sample_id,
                "split": bundle.split,
                "page_index": str(page.page_index),
                "page_count": str(page_count),
                "width": str(page.width),
                "height": str(page.height),
                "dpi": str(annotation.dpi),
                "page_size": annotation.page_size,
                "orientation": annotation.orientation,
                "texture": annotation.texture or "",
                "seed": str(annotation.seed),
                "page_element_count": str(len(page.elements)),
                "has_table": str(bool(header["has_table"])).lower(),
                "has_chart": str(bool(header["has_chart"])).lower(),
                "has_signature": str(bool(header["has_signature"])).lower(),
                "has_stamp": str(bool(header["has_stamp"])).lower(),
                "overflowed": str(bool(header["overflowed"])).lower(),
                "text": page.transcription(),
            }
        )
    return rows


def _splits_row(bundle: SampleBundle) -> dict[str, str]:
    """Build one ``splits.csv`` row.

    Args:
        bundle: The sample's artifacts.

    Returns:
        A mapping keyed by :data:`SPLITS_COLUMNS`.
    """
    return {
        "sample_id": bundle.sample_id,
        "source_sample_id": bundle.annotation.source_sample_id,
        "split": bundle.split,
    }


def _open_csv(
    path: Path, columns: Sequence[str], *, resume: bool
) -> tuple[Any, Any]:
    """Open a sidecar CSV for appending, writing the header once.

    Args:
        path: The CSV file.
        columns: Its field names.
        resume: Whether to append to an existing file rather than replace it.

    Returns:
        The open handle and a :class:`csv.DictWriter` over it. The caller closes
        the handle.
    """
    ensure_dir(path.parent)
    exists = resume and path.is_file() and path.stat().st_size > 0
    handle = path.open("a" if exists else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns))
    if not exists:
        writer.writeheader()
    return handle, writer


def _config_payloads(
    settings: HuggingFaceConfig, summary: ExportSummary
) -> list[dict[str, Any]]:
    """Describe every family as one HuggingFace config.

    Args:
        settings: The ``huggingface`` config section.
        summary: What the export produced.

    Returns:
        One entry per family, in registry order. ``splits`` lists only the
        splits that actually produced shards, because a config whose
        ``data_files`` glob matches nothing fails to load.
    """
    payloads: list[dict[str, Any]] = []
    for family in summary.families:
        spec = FAMILIES[family]
        splits: list[dict[str, Any]] = []
        for split in summary.splits:
            cell = summary.cell(family, split)
            if not cell.shards:
                continue
            splits.append(
                {
                    "name": split,
                    "path": f"data/{family}/{split}-*.parquet",
                    "num_examples": cell.rows,
                    "num_samples": cell.samples,
                    "num_bytes": cell.payload_bytes,
                    "num_shards": cell.shard_count,
                    "download_size": _shard_file_bytes(settings, cell),
                    "requested_fraction": _requested_fraction(settings, split),
                    "shards": cell.paths(),
                }
            )
        payloads.append(
            {
                "name": family,
                "granularity": spec.granularity,
                "summary": spec.summary,
                "row_group_bytes": spec.row_group_bytes,
                "columns": [
                    {"name": column.name, "kind": column.kind, "note": column.note}
                    for column in spec.payload_columns(
                        include_elements=summary.include_elements
                    )
                ],
                "features": spec.features(
                    include_elements=summary.include_elements
                ),
                "yaml_features": spec.yaml_features(
                    include_elements=summary.include_elements
                ),
                "splits": splits,
                "rows": sum(entry["num_examples"] for entry in splits),
                "payload_bytes": sum(entry["num_bytes"] for entry in splits),
                "download_size": sum(entry["download_size"] for entry in splits),
                "shard_count": sum(entry["num_shards"] for entry in splits),
                "missing_samples": summary.missing.get(family, 0),
            }
        )
    return payloads


def _shard_file_bytes(settings: HuggingFaceConfig, cell: SplitShards) -> int:
    """Return the on-disk size of one cell's Parquet files.

    This is the Hub's ``download_size``: what a consumer actually transfers,
    compressed, as opposed to the uncompressed ``num_bytes`` the exporter
    accumulated. It is measured rather than estimated, which is only possible
    because the sidecars are written after every shard has closed.

    Args:
        settings: The ``huggingface`` config section.
        cell: One family x split cell.

    Returns:
        Total bytes, counting a shard that has somehow gone missing as zero.
    """
    total = 0
    for relative in cell.paths():
        path = settings.output_dir / relative
        try:
            total += path.stat().st_size
        except OSError as error:
            # Not fatal: the size is descriptive metadata, and failing the whole
            # export at the last step over it would throw away every shard.
            _LOGGER.warning("cannot size shard %s: %s", relative, error)
    return total


def _dataset_info_yaml(configs: Sequence[Mapping[str, Any]]) -> str:
    """Render the card's ``dataset_info`` front-matter block.

    Serialised with ``yaml.safe_dump`` rather than written by the template: the
    block is six configs of roughly forty columns each, and a hand-indented
    version of that is one stray space away from front matter the Hub cannot
    parse - which would take the dataset viewer down with it.

    Only ``num_examples`` has to be exact. ``datasets`` compares it against what
    it actually reads and raises ``NonMatchingSplitsSizesError`` on a mismatch,
    while ``num_bytes`` and ``download_size`` are only ever displayed.

    Args:
        configs: The loadable entries of :func:`_config_payloads`.

    Returns:
        The YAML for the block's value, with no trailing newline, or ``""`` when
        there is nothing loadable to declare.

    Raises:
        ExportError: If PyYAML is unavailable.
    """
    if not configs:
        return ""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - PyYAML is a hard dependency
        raise ExportError("PyYAML is required to write the dataset card") from error
    entries = [
        {
            "config_name": config["name"],
            "features": config["yaml_features"],
            "splits": [
                {
                    "name": split["name"],
                    "num_bytes": split["num_bytes"],
                    "num_examples": split["num_examples"],
                }
                for split in config["splits"]
            ],
            "download_size": config["download_size"],
            "dataset_size": config["payload_bytes"],
        }
        for config in configs
    ]
    text = yaml.safe_dump(
        entries,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=1000,
    )
    return text.rstrip("\n")



def write_dataset_info(
    settings: HuggingFaceConfig,
    plan: SplitPlan,
    summary: ExportSummary,
) -> Path:
    """Write ``dataset_info.json``.

    A superset of what ``datasets`` puts in a ``DatasetInfo``: every config's
    features, per-split row counts, ``download_size`` and ``dataset_size`` in the
    library's own vocabulary, plus the things it has no field for - the split
    plan and its fingerprint, the shard list, the per-family skip counts and the
    provenance of the letter text, the logos and the signatures.

    The card's ``dataset_info`` front matter is the copy ``load_dataset`` reads;
    this file is the auditable one, and both are generated from the same
    :func:`_config_payloads` so they cannot disagree.

    No timestamp is recorded, deliberately: CLAUDE.md 8 requires every generated
    file to be reproducible from its config and seed, and a clock reading would
    make two exports of the same corpus differ.

    Args:
        settings: The ``huggingface`` config section.
        plan: The split plan the export used.
        summary: What the export produced.

    Returns:
        The path written.
    """
    configs = _config_payloads(settings, summary)
    split_plan = plan.as_dict()
    loadable = [entry for entry in configs if entry["splits"]]
    payload = {
        "dataset_name": settings.dataset_name,
        "description": (
            "Synthetic Persian (Farsi) administrative letters rendered as page "
            "images, PDFs, Markdown and structured annotations, built to "
            "fine-tune PaddleOCR-VL 1.6 with a LoRA adapter."
        ),
        "license": settings.license_id,
        "language": [settings.language],
        "schema_version": SCHEMA_VERSION,
        "include_elements": summary.include_elements,
        "compression": settings.compression,
        "default_config_name": next(
            (entry["name"] for entry in loadable if entry["name"] == "images"),
            loadable[0]["name"] if loadable else "",
        ),
        "split_plan": split_plan,
        "splits": {
            split: {
                "num_samples": count,
                "requested_fraction": _requested_fraction(settings, split),
                "actual_fraction": (
                    round(count / summary.samples, 6) if summary.samples else 0.0
                ),
            }
            for split, count in summary.split_samples().items()
        },
        "samples": summary.samples,
        "download_size": sum(entry["download_size"] for entry in configs),
        "dataset_size": sum(entry["payload_bytes"] for entry in configs),
        "configs": {
            entry["name"]: {
                key: entry[key]
                for key in (
                    "granularity",
                    "summary",
                    "features",
                    "splits",
                    "rows",
                    "payload_bytes",
                    "download_size",
                    "shard_count",
                    "missing_samples",
                )
            }
            for entry in configs
        },
        "provenance": {
            "letter_text_models": list(summary.text_models),
            "logos": summary.logos.as_dict(),
            "signatures": dict(SIGNATURE_CORPUS),
            "synthetic_notice": SYNTHETIC_NOTICE,
            "non_commercial_notice": NON_COMMERCIAL_NOTICE,
        },
    }
    path = settings.output_dir / "dataset_info.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _template_dir() -> Path:
    """Return the directory holding the dataset-card template."""
    return Path(__file__).with_name("templates")


def write_dataset_card(
    settings: HuggingFaceConfig,
    plan: SplitPlan,
    summary: ExportSummary,
) -> Path:
    """Render ``README.md``, the dataset card the Hub reads.

    The YAML front matter is the machine-readable half of the card, and three of
    its keys are what make ``load_dataset`` work without arguments:

    * ``configs`` maps each family to its shard globs, so
      ``load_dataset(repo, "images", split="train")`` resolves.
    * ``default_config_name`` is what a bare ``load_dataset(repo)`` falls back
      to. Without it, a multi-config repository raises and asks the caller to
      name a config.
    * ``dataset_info`` declares each config's features and per-split row counts,
      which is what fills in the row counts and download sizes on the dataset
      page and lets ``datasets`` describe the schema before downloading a shard.

    ``dataset_info`` is generated from the same :class:`~.families.Column`
    declarations as the Arrow schema and serialised by PyYAML, not hand-indented
    in the template - it has to agree with the Parquet exactly, since
    ``load_dataset`` casts every shard to the features declared here.

    Args:
        settings: The ``huggingface`` config section.
        plan: The split plan the export used.
        summary: What the export produced.

    Returns:
        The path written.

    Raises:
        ExportError: If jinja2 is unavailable or the template fails to render.
    """
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError as error:  # pragma: no cover - jinja2 is a hard dependency
        raise ExportError("jinja2 is required to write the dataset card") from error
    environment = Environment(
        loader=FileSystemLoader(str(_template_dir())),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    configs = _config_payloads(settings, summary)
    loadable = [entry for entry in configs if entry["splits"]]
    default_config = next(
        (entry["name"] for entry in loadable if entry["name"] == "images"),
        loadable[0]["name"] if loadable else "",
    )
    try:
        text = environment.get_template(_CARD_TEMPLATE).render(
            settings=settings,
            summary=summary,
            plan=plan,
            configs=configs,
            loadable=loadable,
            default_config=default_config,
            dataset_info_yaml=_dataset_info_yaml(loadable),
            header_columns=HEADER_COLUMNS,
            split_samples=summary.split_samples(),
            schema_version=SCHEMA_VERSION,
            size_category=_size_category(
                max((entry["rows"] for entry in configs), default=0)
            ),
            logos=summary.logos,
            logo_corpus=LOGO_CORPUS,
            signature_corpus=SIGNATURE_CORPUS,
            text_models=summary.text_models,
            synthetic_notice=SYNTHETIC_NOTICE,
            non_commercial_notice=NON_COMMERCIAL_NOTICE,
        )
    except OSError as error:
        raise ExportError(f"dataset card template unusable: {error}") from error
    path = settings.output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def _size_category(rows: int) -> str:
    """Return the Hub's ``size_categories`` bucket for a row count.

    Args:
        rows: Number of examples in the largest config.

    Returns:
        A bucket label such as ``10K<n<100K``.
    """
    buckets = (
        (1_000, "n<1K"),
        (10_000, "1K<n<10K"),
        (100_000, "10K<n<100K"),
        (1_000_000, "100K<n<1M"),
    )
    for ceiling, label in buckets:
        if rows < ceiling:
            return label
    return "1M<n<10M"


# --------------------------------------------------------------------------- #
# The export itself
# --------------------------------------------------------------------------- #


def export_dataset(
    config: SynthConfig,
    *,
    sample_ids: Sequence[str] | None = None,
    resume: bool = False,
    include_elements: bool | None = None,
) -> ExportSummary:
    """Pack a rendered corpus into the HuggingFace export tree.

    Reads each sample's artifacts from ``paths.output_dir`` and the source
    record from ``paths.plaintext_dir``, routes the sample to a split by a hash
    of its source id, streams it into every configured family, and writes the
    sidecars. Nothing under ``paths.output_dir`` is modified.

    Args:
        config: Loaded configuration.
        sample_ids: Samples to export, in order. Defaults to every sample the
            render manifest lists - which is what ``--hugging-face-only``
            wants; the pipeline passes just the slice it rendered.
        resume: Continue an existing export instead of starting at shard 0,
            skipping samples the state file already records.
        include_elements: Override for ``huggingface.include_elements``.

    Returns:
        What the export tree now holds.

    Raises:
        ExportError: If ``pyarrow`` is missing, there is nothing to export, a
            sample's annotation or page image cannot be read, or a resumed
            export's state disagrees with this run.
    """
    settings = config.huggingface
    wanted = list(sample_ids) if sample_ids is not None else iter_sample_ids(config)
    if not wanted:
        raise ExportError(
            f"nothing to export: no rendered samples under {config.paths.output_dir}"
        )
    plan = SplitPlan.from_settings(settings)
    ensure_dir(settings.output_dir)

    _LOGGER.info(
        "exporting %d sample(s) to %s: %s, %s, batch %d (min %.0f MiB, ceiling "
        "%.2f GiB), %s",
        len(wanted),
        settings.output_dir,
        ", ".join(settings.families),
        plan.describe(),
        settings.shard_samples,
        settings.min_shard_bytes / 1024**2,
        settings.max_shard_bytes / 1024**3,
        settings.compression,
    )

    handles: list[Any] = []
    try:
        with ExportWriter(
            settings, plan, include_elements=include_elements, resume=resume
        ) as writer:
            already = writer.already_exported
            metadata_writer = None
            splits_writer = None
            if settings.write_metadata_csv:
                handle, metadata_writer = _open_csv(
                    settings.output_dir / "metadata.csv",
                    METADATA_COLUMNS,
                    resume=bool(already),
                )
                handles.append(handle)
            if settings.write_splits_csv:
                handle, splits_writer = _open_csv(
                    settings.output_dir / "splits.csv",
                    SPLITS_COLUMNS,
                    resume=bool(already),
                )
                handles.append(handle)
            for sample_id in wanted:
                if sample_id in already:
                    writer.skip(sample_id)
                    continue
                annotation = load_annotation(config, sample_id)
                split = plan.assign(
                    sample_id=annotation.sample_id,
                    source_sample_id=annotation.source_sample_id,
                )
                bundle = load_bundle(
                    config, annotation, split=split, families=writer.needs
                )
                writer.add(bundle)
                if metadata_writer is not None:
                    metadata_writer.writerows(_metadata_rows(bundle))
                if splits_writer is not None:
                    splits_writer.writerow(_splits_row(bundle))
            summary = writer.close()
    finally:
        for handle in handles:
            handle.close()

    summary.logos = write_logo_attribution(config, settings)
    write_dataset_info(settings, plan, summary)
    write_dataset_card(settings, plan, summary)

    counts = summary.split_samples()
    _LOGGER.info(
        "export complete: %d shard(s) across %d famil(ies), %d row(s) from %d "
        "sample(s) (%s), %.2f GiB payload (%d new, %d skipped)",
        summary.shard_count,
        len(summary.families),
        summary.rows,
        summary.samples,
        ", ".join(f"{split} {count}" for split, count in counts.items()),
        summary.payload_bytes / 1024**3,
        summary.new_samples,
        summary.skipped_samples,
    )
    for family, missing in sorted(summary.missing.items()):
        if missing:
            _LOGGER.warning(
                "%s: %d sample(s) had no artifact to pack and were left out of "
                "that family only",
                family,
                missing,
            )
    return summary


def iter_shard_paths(
    settings: HuggingFaceConfig, *, family: str | None = None
) -> Iterator[Path]:
    """Yield the export's shard files.

    Args:
        settings: The ``huggingface`` config section.
        family: Restrict to one family. Defaults to every configured family, in
            registry order.

    Yields:
        Each ``.parquet`` file under ``data/<family>/``, sorted by name - which
        for ``split-NNNNN-of-NNNNN`` groups the splits and orders the shards
        within each.
    """
    families = (family,) if family is not None else settings.families
    for name in families:
        directory = settings.family_dir(name)
        if not directory.is_dir():
            continue
        yield from sorted(directory.glob("*.parquet"))
