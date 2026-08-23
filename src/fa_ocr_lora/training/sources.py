"""Readers that turn the two local corpora into one record type.

The letters corpus and the Persian Pixel corpus disagree about almost
everything - one is JSONL with nested region lists, the other a CSV whose text
column contains embedded newlines - and neither is in the shape a trainer wants.
This module is the only place that knows either format. Everything downstream
sees :class:`SourceRecord`, so the manifest builder, the dataset and the encoder
never branch on which corpus a sample came from.

Three record streams come out of two files:

``admin_letter_page``
    One record per page of a synthetic letter: the full-page transcription in
    reading order, which is what inference-time input looks like.
``admin_letter_crop``
    One record per labelled region of the same pages. A region box is a quarter
    the cost of a page on the GPU and carries the administrative metadata
    fields CLAUDE.md 5.2 calls the highest-priority target, so these are the
    cheap half of the mix and the half that teaches digits.
``persian_pixel_page``
    One record per row of the real-scan corpus.

Rejections are never silent. Every record dropped for a missing image, an empty
target, or a sliver-sized box is counted by reason and written to
``skipped_records.log`` - CLAUDE.md 5.1 asks for exactly that, so a corpus that
quietly halves in size cannot go unnoticed.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .config import DataConfig

__all__ = [
    "LETTER_CROP",
    "LETTER_PAGE",
    "PIXEL_PAGE",
    "SOURCE_TAGS",
    "SkipLog",
    "SourceRecord",
    "iter_letter_crops",
    "iter_letter_pages",
    "iter_pixel_rows",
]

#: Source tag of a full letter page.
LETTER_PAGE: Final[str] = "admin_letter_page"

#: Source tag of a cropped region of a letter page.
LETTER_CROP: Final[str] = "admin_letter_crop"

#: Source tag of a Persian Pixel row.
PIXEL_PAGE: Final[str] = "persian_pixel_page"

#: Every tag the manifest may carry, in mix-reporting order.
SOURCE_TAGS: Final[tuple[str, ...]] = (LETTER_PAGE, LETTER_CROP, PIXEL_PAGE)


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One training pair, before any image is opened.

    Deliberately holds a path and a box rather than pixels: at 10k records the
    decoded images would be gigabytes, and the dataset opens them lazily one at
    a time. ``width``/``height`` describe the region that will actually be fed -
    the crop's size when ``crop_box`` is set, the page's otherwise - so the
    manifest can budget sequence length without decoding anything.

    Attributes:
        record_id: Stable identifier, unique across all sources.
        source: One of :data:`SOURCE_TAGS`.
        group_id: Split-assignment unit. Every record sharing one lands in the
            same split, which is what stops a letter's own pages and crops from
            appearing on both sides of the validation boundary.
        image_path: Absolute path of the page image on disk.
        crop_box: ``(left, top, right, bottom)`` to crop before feeding, or
            ``None`` to use the whole image.
        text: The target transcription.
        width: Width in pixels of the region fed to the model.
        height: Height in pixels of the region fed to the model.
        region_type: Label type for a crop, e.g. ``metadata_field``; empty for
            a page.
    """

    record_id: str
    source: str
    group_id: str
    image_path: Path
    crop_box: tuple[int, int, int, int] | None
    text: str
    width: int
    height: int
    region_type: str = ""

    @property
    def pixels(self) -> int:
        """Area of the fed region, for pixel-budget accounting."""
        return self.width * self.height


class SkipLog:
    """Counts and records every rejected input, by reason.

    Held open for the length of a manifest build and flushed at the end, so the
    log is one file per run rather than one append per rejection. The counter is
    what the run summary prints; the file is what a human reads to find out
    which twelve images were missing.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Prepare a log, optionally backed by a file.

        Args:
            path: Destination for the detail lines. ``None`` keeps counts only,
                which is what unit tests want.
        """
        self._path = path
        self._counts: Counter[str] = Counter()
        self._lines: list[str] = []

    def skip(self, reason: str, record_id: str, detail: str = "") -> None:
        """Record one rejection.

        Args:
            reason: Short, stable category, used as the counter key.
            record_id: Identifier of the rejected record.
            detail: Optional extra context for the log line.
        """
        self._counts[reason] += 1
        suffix = f" - {detail}" if detail else ""
        self._lines.append(f"{reason}\t{record_id}{suffix}")

    @property
    def counts(self) -> dict[str, int]:
        """Rejection count per reason, highest first."""
        return dict(self._counts.most_common())

    @property
    def total(self) -> int:
        """How many records were rejected in all."""
        return sum(self._counts.values())

    def flush(self) -> Path | None:
        """Write the accumulated detail lines to disk.

        Returns:
            The path written, or ``None`` when the log is memory-only or empty.
        """
        if self._path is None or not self._lines:
            return None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")
        return self._path


def _accept_text(text: Any, data: DataConfig) -> str | None:
    """Normalise a target and decide whether it is usable.

    Args:
        text: Raw value from the corpus.
        data: Filters from the configuration.

    Returns:
        The stripped text, or ``None`` when it is too short or too long.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if len(stripped) < data.min_text_chars or len(stripped) > data.max_text_chars:
        return None
    return stripped


def _clip_box(
    box: Sequence[float], width: int, height: int, padding: int
) -> tuple[int, int, int, int] | None:
    """Pad a region box and clip it to the page.

    Padding is what makes a crop legible: a box drawn tight around glyphs cuts
    ascenders and diacritics, and Persian carries a lot of both.

    Args:
        box: ``(left, top, right, bottom)`` as stored in the corpus.
        width: Page width in pixels.
        height: Page height in pixels.
        padding: Pixels added on every side before clipping.

    Returns:
        The clipped box, or ``None`` if it is degenerate or off-page.
    """
    if len(box) != 4:
        return None
    try:
        left, top, right, bottom = (int(round(float(value))) for value in box)
    except (TypeError, ValueError):
        return None
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _iter_jsonl(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Stream a JSONL file, yielding line numbers alongside records.

    Args:
        path: File to read.

    Yields:
        ``(line_number, record)`` pairs; malformed lines are skipped by the
        caller, which needs the line number to report them.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"letters manifest not found: {path}")
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                yield number, {}
                continue
            if isinstance(payload, Mapping):
                yield number, payload
            else:
                yield number, {}


def iter_letter_pages(
    manifest_path: Path, data: DataConfig, skips: SkipLog
) -> Iterator[SourceRecord]:
    """Yield one full-page record per line of the letters JSONL.

    Args:
        manifest_path: Path of ``paddle_ocr_vl.jsonl``.
        data: Filters from the configuration.
        skips: Rejection log, updated in place.

    Yields:
        Records tagged :data:`LETTER_PAGE`, in file order.

    Raises:
        FileNotFoundError: If the manifest is absent.
    """
    root = manifest_path.parent
    for number, payload in _iter_jsonl(manifest_path):
        if not payload:
            skips.skip("letters_malformed_line", f"line:{number}")
            continue
        sample_id = str(payload.get("sample_id", "")).strip()
        page_index = payload.get("page_index", 0)
        record_id = f"{sample_id}:p{page_index}" if sample_id else f"line:{number}"
        relative = str(payload.get("image", "")).strip()
        if not relative:
            skips.skip("letters_no_image_field", record_id)
            continue
        image_path = root / relative
        if not image_path.is_file():
            skips.skip("letters_image_missing", record_id, relative)
            continue
        text = _accept_text(payload.get("text"), data)
        if text is None:
            skips.skip("letters_text_rejected", record_id)
            continue
        try:
            width = int(payload["width"])
            height = int(payload["height"])
        except (KeyError, TypeError, ValueError):
            skips.skip("letters_bad_dimensions", record_id)
            continue
        yield SourceRecord(
            record_id=record_id,
            source=LETTER_PAGE,
            group_id=sample_id or record_id,
            image_path=image_path,
            crop_box=None,
            text=text,
            width=width,
            height=height,
        )


def iter_letter_crops(
    manifest_path: Path, data: DataConfig, skips: SkipLog
) -> Iterator[SourceRecord]:
    """Yield one record per labelled region of the letters JSONL.

    A region is croppable when it carries non-empty text, is not of an excluded
    type, and survives padding and the size floors. The non-empty-text rule
    alone removes every container and decoration box - ``header_bar``,
    ``stamp``, ``signature_image`` - because the corpus stores those with an
    empty string.

    Args:
        manifest_path: Path of ``paddle_ocr_vl.jsonl``.
        data: Crop geometry and text filters from the configuration.
        skips: Rejection log, updated in place.

    Yields:
        Records tagged :data:`LETTER_CROP`, page by page and then in reading
        order within a page.

    Raises:
        FileNotFoundError: If the manifest is absent.
    """
    root = manifest_path.parent
    excluded = frozenset(data.crop_exclude_types)
    for number, payload in _iter_jsonl(manifest_path):
        if not payload:
            skips.skip("crops_malformed_line", f"line:{number}")
            continue
        sample_id = str(payload.get("sample_id", "")).strip()
        page_index = payload.get("page_index", 0)
        page_id = f"{sample_id}:p{page_index}" if sample_id else f"line:{number}"
        regions = payload.get("label")
        if not isinstance(regions, Sequence) or isinstance(regions, str):
            skips.skip("crops_no_regions", page_id)
            continue
        relative = str(payload.get("image", "")).strip()
        image_path = root / relative if relative else None
        if image_path is None or not image_path.is_file():
            skips.skip("crops_image_missing", page_id, relative)
            continue
        try:
            page_width = int(payload["width"])
            page_height = int(payload["height"])
        except (KeyError, TypeError, ValueError):
            skips.skip("crops_bad_dimensions", page_id)
            continue

        for region in regions:
            if not isinstance(region, Mapping):
                continue
            order = region.get("reading_order", "?")
            record_id = f"{page_id}:r{order}"
            region_type = str(region.get("type", "")).strip()
            if region_type in excluded:
                skips.skip("crops_excluded_type", record_id, region_type)
                continue
            text = _accept_text(region.get("text"), data)
            if text is None:
                # The overwhelmingly common case: a container box with no text
                # of its own. Counted, but not worth a line each at this volume.
                skips.skip("crops_text_rejected", record_id, region_type)
                continue
            raw_box = region.get("bbox")
            if not isinstance(raw_box, Sequence) or isinstance(raw_box, str):
                skips.skip("crops_bad_bbox", record_id)
                continue
            box = _clip_box(raw_box, page_width, page_height, data.crop_padding_px)
            if box is None:
                skips.skip("crops_bad_bbox", record_id)
                continue
            left, top, right, bottom = box
            width, height = right - left, bottom - top
            if min(width, height) < data.crop_min_side_px:
                skips.skip("crops_too_thin", record_id, f"{width}x{height}")
                continue
            if width * height < data.crop_min_area_px:
                skips.skip("crops_too_small", record_id, f"{width}x{height}")
                continue
            yield SourceRecord(
                record_id=record_id,
                source=LETTER_CROP,
                # Same group as the page it came from: a letter is one unit, so
                # its crops can never end up validating a model trained on its
                # own pages.
                group_id=sample_id or page_id,
                image_path=image_path,
                crop_box=box,
                text=text,
                width=width,
                height=height,
                region_type=region_type,
            )


def _resolve_pixel_image(root: Path, recorded: str, filename: str) -> Path | None:
    """Find a Persian Pixel image on disk, tolerating a stale path column.

    In this checkout the CSV's ``image_path`` column holds HuggingFace-export
    names (``images/image_0005621.png``) while the files on disk are named after
    ``image_filename`` (``images/page-00010_row0_image.png``). Neither column is
    authoritative on its own, so both are tried: the recorded path first, so a
    corrected export keeps working, then the filename under ``images/`` and at
    the root.

    Args:
        root: Directory the CSV sits in.
        recorded: Value of the ``image_path`` column.
        filename: Value of the ``image_filename`` column.

    Returns:
        The first candidate that exists, or ``None`` if none do.
    """
    candidates = []
    if recorded:
        candidates.append(root / recorded)
    if filename:
        candidates.extend((root / "images" / filename, root / filename))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def iter_pixel_rows(
    metadata_path: Path, data: DataConfig, skips: SkipLog
) -> Iterator[SourceRecord]:
    """Yield one record per row of the Persian Pixel metadata CSV.

    The file needs a real CSV parser, not a line loop: it carries a UTF-8 BOM
    and its ``text`` column contains embedded newlines, so one logical row spans
    several physical lines.

    Args:
        metadata_path: Path of ``all_features_metadata.csv``. Never
            ``labels.csv``, which is header-only in this checkout.
        data: Filters from the configuration.
        skips: Rejection log, updated in place.

    Yields:
        Records tagged :data:`PIXEL_PAGE`, in file order.

    Raises:
        FileNotFoundError: If the metadata file is absent.
        ValueError: If the expected columns are not present.
    """
    if not metadata_path.is_file():
        raise FileNotFoundError(f"pixel metadata not found: {metadata_path}")
    root = metadata_path.parent
    # utf-8-sig strips the BOM, which would otherwise become part of the first
    # column's name and make every lookup of "image_filename" miss.
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_filename", "text", "source_parquet"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{metadata_path}: missing column(s) {sorted(missing)}; found "
                f"{list(reader.fieldnames or ())}"
            )
        for number, row in enumerate(reader, start=1):
            filename = (row.get("image_filename") or "").strip()
            record_id = filename or f"row:{number}"
            image_path = _resolve_pixel_image(
                root, (row.get("image_path") or "").strip(), filename
            )
            if image_path is None:
                skips.skip("pixel_image_missing", record_id, row.get("image_path") or "")
                continue
            text = _accept_text(row.get("text"), data)
            if text is None:
                skips.skip("pixel_text_rejected", record_id)
                continue
            try:
                width = int(row["width"])
                height = int(row["height"])
            except (KeyError, TypeError, ValueError):
                skips.skip("pixel_bad_dimensions", record_id)
                continue
            # Rows of one source parquet are strips of the same scanned page, so
            # they share a group and cannot straddle the split boundary.
            group = (row.get("source_parquet") or record_id).strip()
            yield SourceRecord(
                record_id=record_id,
                source=PIXEL_PAGE,
                group_id=group or record_id,
                image_path=image_path,
                crop_box=None,
                text=text,
                width=width,
                height=height,
            )
