"""Corpus selection, splitting, and the unified manifest.

This is the module that turns "two files on disk" into "one reproducible
training corpus". It does three jobs that all have to be deterministic, because
CLAUDE.md 8 requires every artifact to be reproducible from its config and seed:

**Selection.** Both corpora are larger than the run needs - 26,848 letter pages
and ~150k croppable regions against a 2,500 + 2,500 budget. Which records get
picked is decided by a hash of the record id, not by a shuffle, so the same
config picks the same 10,000 records on any machine and in any Python build.

**Splitting.** Splits are assigned per *group*, never per record. A letter's
pages and its own region crops share a ``sample_id``, and Persian Pixel rows cut
from one scan share a ``source_parquet``; letting either straddle the boundary
would leak the answer into validation and quietly inflate every metric. Group
scores are hashed too, so growing the corpus keeps the old holdout mostly intact
and a pilot metric stays comparable with a full-run one.

**Digit weighting.** Persian OCR fails hardest on numerals, and the fields
CLAUDE.md 5.2 calls highest-priority are numeric. Records in the top quartile of
digit density get a repeat weight, so the sampler sees them more often without
anyone hand-curating a digit subset.

The output is ``unified_manifest.parquet`` plus a ``split_summary.json`` that
records the mix, the skip counts by reason, and an explicit assertion that no
group appears in two splits.
"""

from __future__ import annotations

import hashlib
import heapq
import logging
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ..dataset.common.io_utils import ensure_dir, relative_to_repo, write_json
from .config import TrainingConfig
from .images import fit_pixel_budget, image_tokens, is_feedable
from .sources import (
    LETTER_CROP,
    LETTER_PAGE,
    PIXEL_PAGE,
    SOURCE_TAGS,
    SkipLog,
    SourceRecord,
    iter_letter_crops,
    iter_letter_pages,
    iter_pixel_rows,
)

__all__ = [
    "MANIFEST_COLUMNS",
    "SPLIT_NAMES",
    "ManifestSummary",
    "build_manifest",
    "digit_density",
    "load_manifest",
    "parse_crop_box",
    "stable_unit",
]

#: Columns of ``unified_manifest.parquet``, in written order.
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "record_id",
    "source",
    "group_id",
    "image_path",
    "crop_box",
    "region_type",
    "text",
    "src_width",
    "src_height",
    "fed_width",
    "fed_height",
    "image_tokens",
    "text_chars",
    "digit_density",
    "digit_quartile",
    "weight",
    "split",
    "seed",
)

#: Split labels, in reporting order.
SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "validation", "test")

#: Latin, Arabic-Indic and Extended Arabic-Indic (Persian) digits. Persian text
#: mixes all three - a date may be Persian, an invoice number Latin - so digit
#: metrics that only counted one would miss most of the errors that matter.
_DIGITS: Final[frozenset[str]] = frozenset(
    "0123456789"  # Latin
    "٠١٢٣٤٥٦٧٨٩"  # Arabic-Indic
    "۰۱۲۳۴۵۶۷۸۹"  # Persian
)

#: Full 64-bit space of the hash prefix used for stable scoring.
_HASH_SPACE: Final[float] = float(1 << 64)

#: Ceiling on the validation split as a share of the selected corpus. Guards the
#: ``--limit`` path, where the configured 800-record holdout would otherwise
#: consume a 200-record pilot corpus entirely.
_MAX_HOLDOUT_SHARE: Final[float] = 0.2

LOGGER = logging.getLogger(__name__)


def stable_unit(seed: int, purpose: str, key: str) -> float:
    """Map a key to a stable pseudo-random number in ``[0, 1)``.

    Uses BLAKE2b rather than :func:`hash`, which is salted per process and would
    make selection and splitting differ between runs of the same config. The
    ``purpose`` string keeps independent decisions independent: a record that
    scores low for selection must not therefore score low for splitting.

    Args:
        seed: Master seed from the configuration.
        purpose: What the score decides, e.g. ``"select"`` or ``"split"``.
        key: The record id or group id being scored.

    Returns:
        A deterministic value in ``[0, 1)``.
    """
    digest = hashlib.blake2b(f"{seed}:{purpose}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / _HASH_SPACE


def digit_density(text: str) -> float:
    """Fraction of a string's non-whitespace characters that are digits.

    Args:
        text: Target transcription.

    Returns:
        A value in ``[0, 1]``; ``0.0`` for text with no non-whitespace
        characters.
    """
    body = [char for char in text if not char.isspace()]
    if not body:
        return 0.0
    return sum(1 for char in body if char in _DIGITS) / len(body)


def _quartile_cut(values: Sequence[float]) -> float:
    """Third-quartile cut of a sample, computed without numpy.

    Args:
        values: Digit densities of the training records.

    Returns:
        The value at the 75th percentile, or ``inf`` for an empty sample so that
        nothing is oversampled.
    """
    if not values:
        return float("inf")
    ordered = sorted(values)
    index = int(0.75 * (len(ordered) - 1))
    return ordered[index]


def format_crop_box(box: tuple[int, int, int, int] | None) -> str:
    """Render a crop box for the manifest.

    A comma string rather than a list column: parquet can hold lists, but a flat
    string survives every tool that might open this file - pandas, DuckDB, a
    text editor - and is unambiguous when empty.

    Args:
        box: ``(left, top, right, bottom)`` or ``None``.

    Returns:
        ``"left,top,right,bottom"``, or ``""`` for no crop.
    """
    return "" if box is None else ",".join(str(value) for value in box)


def parse_crop_box(value: Any) -> tuple[int, int, int, int] | None:
    """Read a crop box back from a manifest cell.

    Args:
        value: Cell contents, normally a comma string or an empty string.

    Returns:
        The box, or ``None`` when the cell is empty or missing.

    Raises:
        ValueError: If the cell holds something that is neither empty nor four
            integers.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError(f"crop_box must hold four integers, got {value!r}")
    left, top, right, bottom = (int(part) for part in parts)
    return left, top, right, bottom


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    """What a manifest build produced, for the run log and for tests.

    Attributes:
        manifest_path: Where the parquet landed.
        summary_path: Where the JSON summary landed.
        rows: Total records written.
        per_split: Record count per split name.
        per_source: Record count per source tag.
        mix: Record count per ``(split, source)`` pair, as a nested mapping.
        skipped: Rejection count per reason.
        crop_allocation: Per region type, ``(taken, available)``. Records the
            effect of the type weights, so an over- or under-served field is
            visible without re-deriving it from the parquet.
        oversampled: How many training records carry a repeat weight above 1.
        digit_cut: Digit-density threshold that defined the top quartile.
        groups: Distinct group ids across all splits.
    """

    manifest_path: Path
    summary_path: Path
    rows: int
    per_split: Mapping[str, int]
    per_source: Mapping[str, int]
    mix: Mapping[str, Mapping[str, int]]
    skipped: Mapping[str, int]
    crop_allocation: Mapping[str, tuple[int, int]]
    oversampled: int
    digit_cut: float
    groups: int

    def to_dict(self) -> dict[str, Any]:
        """Render as JSON-serialisable data."""
        return {
            "manifest_path": relative_to_repo(self.manifest_path),
            "rows": self.rows,
            "groups": self.groups,
            "per_split": dict(self.per_split),
            "per_source": dict(self.per_source),
            "mix": {split: dict(counts) for split, counts in self.mix.items()},
            "skipped": dict(self.skipped),
            "skipped_total": sum(self.skipped.values()),
            "crop_allocation": {
                name: {"taken": taken, "available": available}
                for name, (taken, available) in self.crop_allocation.items()
            },
            "oversampled_train_records": self.oversampled,
            "digit_density_q3": self.digit_cut,
        }


@dataclass
class _SourcePlan:
    """One source's share of the run: its stream and how many records to keep."""

    tag: str
    limit: int
    stream: Iterable[SourceRecord]
    kept: list[SourceRecord] = field(default_factory=list)


def _allocate(
    available: Mapping[str, int], weights: Mapping[str, float], limit: int
) -> dict[str, int]:
    """Divide a record budget between region types by weight, capped by supply.

    Largest-share-first with redistribution: each round hands out the remaining
    budget in proportion to the weights of the types that still have surplus, so
    a type that runs out does not strand its share. The single-unit fallback
    handles the tail, where every proportional share rounds to zero.

    Args:
        available: How many records exist per type.
        weights: Relative weight per type. A weight of zero excludes the type.
        limit: Total records to allocate.

    Returns:
        Records to take per type, summing to ``min(limit, total available)``.
    """
    quota: dict[str, int] = dict.fromkeys(available, 0)
    active = {
        name
        for name, count in available.items()
        if count > 0 and weights.get(name, 0.0) > 0.0
    }
    remaining = min(limit, sum(available[name] for name in active))
    while remaining > 0 and active:
        total_weight = sum(weights[name] for name in active)
        granted = 0
        for name in sorted(active):
            share = int(remaining * weights[name] / total_weight)
            take = min(share, available[name] - quota[name])
            quota[name] += take
            granted += take
        if granted == 0:
            # Every proportional share rounded to zero: hand out one record at a
            # time, heaviest weight first, until the budget is spent.
            for name in sorted(active, key=lambda key: (-weights[key], key)):
                if remaining == 0:
                    break
                if quota[name] < available[name]:
                    quota[name] += 1
                    remaining -= 1
        else:
            remaining -= granted
        active = {name for name in active if quota[name] < available[name]}
    return quota


def _select_crops(
    stream: Iterable[SourceRecord], limit: int, config: TrainingConfig, skips: SkipLog
) -> tuple[list[SourceRecord], dict[str, tuple[int, int]]]:
    """Select region crops under a per-type quota rather than uniformly.

    Two passes over one materialised candidate set: the first buckets candidates
    per type into bounded heaps, the second allocates the budget across types and
    takes the best-scoring records of each. Each heap is capped at ``limit``,
    which bounds memory at ``types x limit`` records instead of the corpus's
    ~325,000 croppable regions.

    Args:
        stream: Candidate crops from :func:`~fa_ocr_lora.training.sources.
            iter_letter_crops`.
        limit: Total crops wanted.
        config: Loaded configuration, for the type weights and the seed.
        skips: Rejection log, updated in place.

    Returns:
        ``(records, allocation)`` - the selected crops ordered by type and then
        by selection score, and per type a ``(taken, available)`` pair. Crops left
        unselected are surplus, not rejects, so they are reported here rather than
        in the skip log, whose counts mean "unusable".
    """
    if limit <= 0:
        return [], {}
    seed = config.seed
    heaps: dict[str, list[tuple[float, str, SourceRecord]]] = {}
    seen: Counter[str] = Counter()
    for record in stream:
        if not is_feedable(record.width, record.height):
            skips.skip(
                "crops_bad_aspect",
                record.record_id,
                f"{record.width}x{record.height}",
            )
            continue
        region_type = record.region_type or "unlabelled"
        seen[region_type] += 1
        if config.data.crop_weight(region_type) <= 0.0:
            continue
        heap = heaps.setdefault(region_type, [])
        entry = (-stable_unit(seed, "select", record.record_id), record.record_id, record)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    available = {name: len(heap) for name, heap in heaps.items()}
    weights = {name: config.data.crop_weight(name) for name in available}
    quota = _allocate(available, weights, limit)
    if sum(quota.values()) < limit:
        skips.skip(
            "source_underfilled",
            LETTER_CROP,
            f"asked {limit}, corpus yielded {sum(quota.values())}",
        )

    selected: list[SourceRecord] = []
    for name in sorted(quota):
        take = quota[name]
        if take <= 0:
            continue
        ordered = sorted(heaps[name], reverse=True)
        selected.extend(record for _, _, record in ordered[:take])
    allocation = {
        name: (quota.get(name, 0), count) for name, count in sorted(seen.items())
    }
    return selected, allocation


def _select(
    stream: Iterable[SourceRecord], limit: int, seed: int, skips: SkipLog
) -> list[SourceRecord]:
    """Keep the ``limit`` lowest-scoring records of a stream.

    Scoring every candidate and keeping the lowest is a hash-stable substitute
    for a shuffle: the chosen subset depends only on ``(seed, record_id)``, so it
    survives a reordered corpus and a different Python build. Records whose
    geometry the vision tower cannot accept are dropped here, where the reason
    can still be counted.

    The kept set is held in a bounded heap rather than a sorted list of every
    candidate. That matters for crops: the letters corpus offers a few hundred
    thousand croppable regions against a limit of a few thousand, and holding
    them all would cost more memory than the training loop.

    Args:
        stream: Candidate records.
        limit: How many to keep; ``0`` keeps none and consumes nothing.
        seed: Master seed.
        skips: Rejection log, updated in place.

    Returns:
        The kept records, ordered by their selection score.
    """
    if limit <= 0:
        return []
    # Min-heap of (-score, id, record): the root is the worst kept candidate, so
    # one pop per overflow keeps exactly the `limit` lowest scores.
    heap: list[tuple[float, str, SourceRecord]] = []
    candidates = 0
    tag = ""
    for record in stream:
        tag = tag or record.source
        if not is_feedable(record.width, record.height):
            skips.skip(
                f"{record.source}_bad_aspect",
                record.record_id,
                f"{record.width}x{record.height}",
            )
            continue
        candidates += 1
        entry = (-stable_unit(seed, "select", record.record_id), record.record_id, record)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    if candidates < limit:
        skips.skip(
            "source_underfilled",
            tag or "unknown",
            f"asked {limit}, corpus yielded {candidates}",
        )
    return [record for _, _, record in sorted(heap, reverse=True)]


def _assign_splits(
    records: Sequence[SourceRecord],
    *,
    seed: int,
    validation_size: int,
    test_fraction: float,
) -> dict[str, str]:
    """Assign each group to a split, holding out whole groups at a time.

    Groups are ordered by a hash of their id, then filled into validation until
    the requested record count is reached, then into test. Because the ordering
    is a property of the group id alone, a larger corpus holds out a superset of
    the same groups instead of a fresh random draw.

    Args:
        records: Every selected record.
        seed: Master seed.
        validation_size: Records wanted in validation. The realised count is the
            first group boundary at or past this number, so it is approximate by
            design - a whole group is never split to hit an exact total.
        test_fraction: Fraction of the remaining records to hold out as test.

    Returns:
        Mapping from group id to split name.
    """
    sizes: dict[str, int] = {}
    for record in records:
        sizes[record.group_id] = sizes.get(record.group_id, 0) + 1
    ordered = sorted(sizes, key=lambda group: (stable_unit(seed, "split", group), group))

    assignment: dict[str, str] = {}
    filled = 0
    index = 0
    while index < len(ordered) and filled < validation_size:
        group = ordered[index]
        assignment[group] = "validation"
        filled += sizes[group]
        index += 1

    remaining = len(records) - filled
    test_target = int(round(test_fraction * remaining)) if test_fraction > 0 else 0
    filled_test = 0
    while index < len(ordered) and filled_test < test_target:
        group = ordered[index]
        assignment[group] = "test"
        filled_test += sizes[group]
        index += 1

    for group in ordered[index:]:
        assignment[group] = "train"
    return assignment


def _scaled_counts(config: TrainingConfig, limit: int | None) -> dict[str, int]:
    """Per-source record counts, optionally scaled down to a total.

    ``--limit`` exists so the whole pipeline can be exercised on 200 records in
    seconds. Scaling proportionally rather than truncating keeps the mix - and
    therefore the shape of what is being tested - the same as the real run.

    Args:
        config: Loaded configuration.
        limit: Total records wanted, or ``None`` for the configured counts.

    Returns:
        Mapping from source tag to record count.
    """
    counts = {
        LETTER_PAGE: config.data.letters_pages,
        LETTER_CROP: config.data.letters_crops,
        PIXEL_PAGE: config.data.pixel_pages,
    }
    total = sum(counts.values())
    if limit is None or limit >= total or total == 0:
        return counts
    ratio = limit / total
    scaled = {tag: int(count * ratio) for tag, count in counts.items()}
    for tag, count in counts.items():
        if count > 0 and scaled[tag] == 0:
            scaled[tag] = 1
    return scaled


def build_manifest(
    config: TrainingConfig, *, limit: int | None = None
) -> ManifestSummary:
    """Read both corpora, select, split, weight, and write the manifest.

    Args:
        config: Loaded configuration; supplies paths, counts and the seed.
        limit: Cap on total records, scaling the per-source mix proportionally.
            Used by ``--limit`` to smoke-test the pipeline.

    Returns:
        A summary of what was written.

    Raises:
        FileNotFoundError: If either corpus file is missing.
        ImportError: If pandas is unavailable, which means the parquet cannot be
            written.
        ValueError: If selection yields no usable records at all.
    """
    skips = SkipLog(config.paths.skipped_log_path)
    counts = _scaled_counts(config, limit)
    data = config.data
    seed = config.seed
    crop_allocation: dict[str, tuple[int, int]] = {}

    plans = [
        _SourcePlan(
            LETTER_PAGE,
            counts[LETTER_PAGE],
            iter_letter_pages(config.paths.letters_manifest, data, skips),
        ),
        _SourcePlan(
            LETTER_CROP,
            counts[LETTER_CROP],
            iter_letter_crops(config.paths.letters_manifest, data, skips),
        ),
        _SourcePlan(
            PIXEL_PAGE,
            counts[PIXEL_PAGE],
            iter_pixel_rows(config.paths.pixel_metadata, data, skips),
        ),
    ]
    for plan in plans:
        if plan.tag == LETTER_CROP:
            plan.kept, crop_allocation = _select_crops(
                plan.stream, plan.limit, config, skips
            )
        else:
            plan.kept = _select(plan.stream, plan.limit, seed, skips)

    selected = [record for plan in plans for record in plan.kept]
    if not selected:
        raise ValueError(
            "no usable records selected; check paths.letters_manifest and "
            f"paths.pixel_metadata, and see {config.paths.skipped_log_path}"
        )

    # The configured holdout assumes the configured corpus size. Under --limit,
    # or when a corpus yields less than asked, an 800-record holdout out of 100
    # records would swallow the training split entirely - so cap it at a share of
    # what was actually selected and say so.
    holdout = min(data.validation_size, max(1, int(_MAX_HOLDOUT_SHARE * len(selected))))
    if holdout != data.validation_size:
        LOGGER.warning(
            "validation_size %d exceeds %.0f%% of the %d selected records; "
            "holding out %d instead",
            data.validation_size,
            _MAX_HOLDOUT_SHARE * 100,
            len(selected),
            holdout,
        )

    assignment = _assign_splits(
        selected,
        seed=seed,
        validation_size=holdout,
        test_fraction=data.test_fraction,
    )

    train_densities = [
        digit_density(record.text)
        for record in selected
        if assignment[record.group_id] == "train"
    ]
    digit_cut = _quartile_cut(train_densities)

    rows = list(
        _rows(
            selected,
            assignment=assignment,
            config=config,
            digit_cut=digit_cut,
        )
    )
    summary = _write(
        rows,
        config=config,
        skips=skips,
        digit_cut=digit_cut,
        crop_allocation=crop_allocation,
    )
    skips.flush()
    return summary


def _rows(
    records: Sequence[SourceRecord],
    *,
    assignment: Mapping[str, str],
    config: TrainingConfig,
    digit_cut: float,
) -> Iterator[dict[str, Any]]:
    """Turn selected records into manifest rows.

    Args:
        records: Selected records.
        assignment: Group-to-split mapping.
        config: Loaded configuration.
        digit_cut: Third-quartile digit density of the training records.

    Yields:
        One row dict per record, keyed by :data:`MANIFEST_COLUMNS`.
    """
    image = config.image
    multiplier = config.data.digit_oversample_multiplier
    for index, record in enumerate(records):
        split = assignment[record.group_id]
        fed_w, fed_h = fit_pixel_budget(
            record.width,
            record.height,
            min_pixels=image.min_pixels,
            max_pixels=image.max_pixels,
        )
        density = digit_density(record.text)
        # Only training records are ever repeated. Oversampling validation would
        # change what the metric measures between two runs with different cuts.
        is_top = density >= digit_cut and split == "train"
        yield {
            "record_id": record.record_id,
            "source": record.source,
            "group_id": record.group_id,
            "image_path": relative_to_repo(record.image_path),
            "crop_box": format_crop_box(record.crop_box),
            "region_type": record.region_type,
            "text": unicodedata.normalize("NFC", record.text),
            "src_width": record.width,
            "src_height": record.height,
            "fed_width": fed_w,
            "fed_height": fed_h,
            "image_tokens": image_tokens(fed_w, fed_h),
            "text_chars": len(record.text),
            "digit_density": round(density, 6),
            "digit_quartile": 4 if density >= digit_cut else 1,
            "weight": multiplier if is_top else 1.0,
            "split": split,
            # Per-record seed, so a sample's own augmentation and ordering are
            # reproducible from the manifest alone - CLAUDE.md 8.
            "seed": config.seed + index,
        }


def _write(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: TrainingConfig,
    skips: SkipLog,
    digit_cut: float,
    crop_allocation: Mapping[str, tuple[int, int]],
) -> ManifestSummary:
    """Write the parquet and the JSON summary.

    Args:
        rows: Manifest rows.
        config: Loaded configuration.
        skips: Rejection log, read for its counts.
        digit_cut: Third-quartile digit density.
        crop_allocation: Per region type, ``(taken, available)``.

    Returns:
        The summary describing what was written.

    Raises:
        ImportError: If pandas is unavailable.
        AssertionError: If a group id ended up in more than one split, which
            would mean the holdout leaks.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ImportError(
            "pandas is required to write the manifest; install the 'local-train' extra"
        ) from exc

    frame = pd.DataFrame(list(rows), columns=list(MANIFEST_COLUMNS))
    manifest_path = config.paths.manifest_path
    ensure_dir(manifest_path.parent)
    frame.to_parquet(manifest_path, index=False)

    per_split = {name: int((frame["split"] == name).sum()) for name in SPLIT_NAMES}
    per_source = {tag: int((frame["source"] == tag).sum()) for tag in SOURCE_TAGS}
    mix = {
        name: {
            tag: int(((frame["split"] == name) & (frame["source"] == tag)).sum())
            for tag in SOURCE_TAGS
        }
        for name in SPLIT_NAMES
    }

    # The whole point of group-aware splitting; assert it rather than trust it.
    spans = frame.groupby("group_id")["split"].nunique()
    leaked = spans[spans > 1].index.tolist()
    if leaked:
        raise AssertionError(
            f"{len(leaked)} group(s) span more than one split, e.g. {leaked[:5]}"
        )

    summary = ManifestSummary(
        manifest_path=manifest_path,
        summary_path=config.paths.split_summary_path,
        rows=len(frame),
        per_split=per_split,
        per_source=per_source,
        mix=mix,
        skipped=skips.counts,
        crop_allocation=dict(crop_allocation),
        oversampled=int((frame["weight"] > 1.0).sum()),
        digit_cut=float(digit_cut) if digit_cut != float("inf") else -1.0,
        groups=int(frame["group_id"].nunique()),
    )
    write_json(summary.summary_path, summary.to_dict())
    return summary


def load_manifest(path: Path, split: str | None = None) -> Any:
    """Read a manifest back, optionally filtered to one split.

    Args:
        path: Path of ``unified_manifest.parquet``.
        split: Split name to keep, or ``None`` for every row.

    Returns:
        A pandas ``DataFrame``. Typed as ``Any`` so importing this module does
        not require pandas at type-check time.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ImportError: If pandas is unavailable.
        ValueError: If ``split`` is not a known split name.
    """
    if split is not None and split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {list(SPLIT_NAMES)}, got {split!r}")
    if not path.is_file():
        raise FileNotFoundError(
            f"manifest not found: {path}; run 'python -m "
            "fa_ocr_lora.training.prepare_manifest' first"
        )
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ImportError(
            "pandas is required to read the manifest; install the 'local-train' extra"
        ) from exc

    frame = pd.read_parquet(path)
    if split is not None:
        frame = frame[frame["split"] == split].reset_index(drop=True)
    return frame
