"""Frozen view of one letter, assembled from a manifest row and its JSON.

The renderer should never reach into raw CSV strings or nested JSON dicts.
:class:`LetterSource` is the single place where the text stage's output shape
is understood: it resolves asset paths, parses the body, reconciles the table
that the drafting model duplicated across ``body`` and ``attachments``, and
records - rather than raises on - any asset that turned up missing, so one
broken logo cannot abort a 2000-sample run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from ...config import SynthConfig
from ..common.io_utils import read_json, relative_to_repo, resolve_repo_path
from .elements.charts import chart_series_block
from .markdown_blocks import Block, BlockKind, parse_body

__all__ = ["ContactInfo", "LetterSource", "load_letter_source", "iter_letter_sources"]

#: Fallback brand colour when a logo row carries no usable hex.
_DEFAULT_BRAND_HEX: Final[str] = "#1F3A5F"

_HEX_RE: Final[re.Pattern[str]] = re.compile(r"^#?(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

#: Leading enumeration on an attachment line (``۱.``, ``1.``, ``-``), stripped
#: so the composer can re-number the list in its own style.
_ATTACHMENT_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:[0-9]{1,2}|[۰-۹]{1,2}|الف|ب|ج|د|ه|و)\s*[.)\-–]\s*|^\s*[-•*]\s*"
)


def _normalise_hex(value: str) -> str:
    """Return a ``#rrggbb`` colour, falling back when the input is unusable.

    Args:
        value: Colour as stored in the logo corpus, with or without ``#``.

    Returns:
        A valid CSS hex colour.
    """
    candidate = (value or "").strip()
    if not _HEX_RE.match(candidate):
        return _DEFAULT_BRAND_HEX
    if not candidate.startswith("#"):
        candidate = f"#{candidate}"
    if len(candidate) == 4:
        candidate = "#" + "".join(char * 2 for char in candidate[1:])
    return candidate.upper()


@dataclass(frozen=True, slots=True)
class ContactInfo:
    """Footer contact details for the sending organisation."""

    address: str = ""
    phone: str = ""
    email: str = ""
    website: str = ""

    @property
    def fields(self) -> tuple[tuple[str, str], ...]:
        """Return ``(label, value)`` pairs for every populated field."""
        labels = (
            ("address", "نشانی", self.address),
            ("phone", "تلفن", self.phone),
            ("email", "رایانامه", self.email),
            ("website", "وبگاه", self.website),
        )
        return tuple((key, value) for key, _, value in labels if value.strip())

    @property
    def labelled_fields(self) -> tuple[tuple[str, str, str], ...]:
        """Return ``(key, Persian label, value)`` for every populated field."""
        labels = (
            ("address", "نشانی", self.address),
            ("phone", "تلفن", self.phone),
            ("email", "رایانامه", self.email),
            ("website", "وبگاه", self.website),
        )
        return tuple(item for item in labels if item[2].strip())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContactInfo":
        """Build contact details from the letter record's ``contact_info``."""
        return cls(
            address=str(payload.get("address", "") or "").strip(),
            phone=str(payload.get("phone", "") or "").strip(),
            email=str(payload.get("email", "") or "").strip(),
            website=str(payload.get("website", "") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class LetterSource:
    """Everything the renderer needs about one letter.

    Attributes:
        missing_assets: Human-readable notes about assets that could not be
            found. The sample still renders - without a logo, or without a
            signature - and the pipeline records the note in its manifest.
    """

    sample_id: str
    sample_index: int
    company_name: str
    company_category: str
    brand_hex: str
    logo_title: str
    destination_organization: str
    letter_number: str
    date: str
    national_id: str
    registration_number: str
    subject: str
    signatory_title: str
    chart_table: str
    contact: ContactInfo
    body_blocks: tuple[Block, ...] = ()
    attachment_lines: tuple[str, ...] = ()
    logo_path: Path | None = None
    signature_path: Path | None = None
    missing_assets: tuple[str, ...] = ()

    @property
    def has_signature(self) -> bool:
        """Return whether a signature image is available for this letter."""
        return self.signature_path is not None

    @property
    def has_logo(self) -> bool:
        """Return whether a logo SVG is available for this letter."""
        return self.logo_path is not None

    @property
    def body_tables(self) -> tuple[Block, ...]:
        """Return the body's table blocks, excluding any chart data series.

        On a chart-bearing letter the ``| برچسب | مقدار |`` table *is* the
        chart's data, and :mod:`.elements.charts` renders it as bars. Handing
        it back here as well would put the same numbers on the page twice, so
        it is filtered out for chart letters only.
        """
        tables = tuple(block for block in self.body_blocks if block.is_table)
        if not self.wants_chart:
            return tables
        consumed = chart_series_block(self.body_blocks)
        if consumed is None:
            return tables
        return tuple(block for block in tables if block is not consumed)

    @property
    def renderable_blocks(self) -> tuple[Block, ...]:
        """Return body blocks in reading order, minus any chart data series.

        This is what the page composer walks: the chart's own series table is
        replaced by the rendered chart at the same point in the flow, so it
        must not also appear as prose or as a grid.
        """
        if not self.wants_chart:
            return self.body_blocks
        consumed = chart_series_block(self.body_blocks)
        if consumed is None:
            return self.body_blocks
        return tuple(block for block in self.body_blocks if block is not consumed)

    @property
    def wants_chart(self) -> bool:
        """Return whether the text stage marked this letter as chart-bearing."""
        return self.chart_table == "chart"

    @property
    def wants_table(self) -> bool:
        """Return whether the text stage marked this letter as table-bearing."""
        return self.chart_table == "table"

    @property
    def letter_fields(self) -> dict[str, str]:
        """Return the header/metadata field values, for the annotation."""
        return {
            "company_name": self.company_name,
            "destination_organization": self.destination_organization,
            "subject": self.subject,
            "letter_number": self.letter_number,
            "date": self.date,
            "national_id": self.national_id,
            "registration_number": self.registration_number,
            "signatory_title": self.signatory_title,
        }


def _parse_attachments(raw: str, body_blocks: Sequence[Block]) -> tuple[str, ...]:
    """Split the attachments field into clean lines.

    The drafting model duplicated any Markdown table verbatim into both
    ``body`` and ``attachments``. Rendering both would put the same table on
    the page twice, so a table already present in the body is dropped here.

    Args:
        raw: ``letter.attachments`` as stored.
        body_blocks: Parsed body, used to detect the duplicate table.

    Returns:
        One entry per attachment, enumeration markers removed.
    """
    if not raw or not raw.strip():
        return ()
    body_tables = {block.rows for block in body_blocks if block.is_table}
    lines: list[str] = []
    for block in parse_body(raw):
        if block.is_table:
            if block.rows in body_tables:
                continue
            # A table unique to the attachments still reads as a list entry;
            # the element stage picks it up separately for rendering.
            continue
        text = _ATTACHMENT_MARKER_RE.sub("", block.text).strip()
        if text:
            lines.append(text)
    return tuple(lines)


def load_letter_source(
    row: Mapping[str, str],
    config: SynthConfig,
) -> LetterSource:
    """Build a :class:`LetterSource` from one manifest row.

    Args:
        row: A row of ``data/text/generator.csv``.
        config: Loaded synthetic-render configuration, for asset roots.

    Returns:
        The assembled source view.

    Raises:
        FileNotFoundError: If the row's plaintext JSON is missing - unlike an
            absent logo or signature, there is no letter to render without it.
    """
    sample_path = resolve_repo_path(row["sample_path"])
    if not sample_path.is_file():
        # Manifests written on another machine carry absolute paths that do
        # not exist here; fall back to the configured plaintext directory.
        sample_path = config.paths.plaintext_dir / f"{row['sample_id']}.json"
    if not sample_path.is_file():
        raise FileNotFoundError(
            f"{row['sample_id']}: plaintext record not found at {sample_path}"
        )

    record = read_json(sample_path)
    inputs: Mapping[str, Any] = record.get("inputs", {})
    letter: Mapping[str, Any] = record.get("letter", {})
    missing: list[str] = []

    logo_relative = (row.get("logo_path") or inputs.get("svg_path") or "").strip()
    logo_path: Path | None = None
    if logo_relative:
        candidate = config.paths.logos_dir / logo_relative
        if candidate.is_file():
            logo_path = candidate
        else:
            missing.append(f"logo:{logo_relative}")
    else:
        missing.append("logo:unset")

    signature_relative = (row.get("signature_path") or "").strip()
    signature_path: Path | None = None
    if signature_relative:
        candidate = resolve_repo_path(signature_relative)
        if candidate.is_file():
            signature_path = candidate
        else:
            missing.append(f"signature:{signature_relative}")

    body_blocks = parse_body(str(letter.get("body", "")))

    return LetterSource(
        sample_id=str(record.get("sample_id", row["sample_id"])),
        sample_index=int(record.get("sample_index", row.get("sample_index", 0))),
        company_name=str(
            letter.get("company_name", "") or inputs.get("company_name", "")
        ),
        company_category=str(letter.get("company_category", "") or ""),
        brand_hex=_normalise_hex(str(inputs.get("brand_hex", ""))),
        logo_title=str(row.get("logo_title", "") or inputs.get("company_name", "")),
        destination_organization=str(letter.get("destination_organization", "") or ""),
        letter_number=str(letter.get("letter_number", "") or ""),
        date=str(letter.get("date", "") or ""),
        national_id=str(letter.get("national_id", "") or ""),
        registration_number=str(letter.get("registration_number", "") or ""),
        subject=str(letter.get("subject", "") or ""),
        signatory_title=str(letter.get("signatory_title", "") or ""),
        chart_table=str(
            inputs.get("chart_table", row.get("chart_table", "none")) or "none"
        ),
        contact=ContactInfo.from_dict(letter.get("contact_info", {}) or {}),
        body_blocks=body_blocks,
        attachment_lines=_parse_attachments(
            str(letter.get("attachments", "")), body_blocks
        ),
        logo_path=logo_path,
        signature_path=signature_path,
        missing_assets=tuple(missing),
    )


def iter_letter_sources(
    rows: Sequence[Mapping[str, str]],
    config: SynthConfig,
) -> list[LetterSource]:
    """Load many sources, skipping rows whose record is missing.

    Args:
        rows: Manifest rows.
        config: Loaded synthetic-render configuration.

    Returns:
        One source per readable row, in input order.
    """
    sources: list[LetterSource] = []
    for row in rows:
        try:
            sources.append(load_letter_source(row, config))
        except (FileNotFoundError, ValueError, KeyError):
            continue
    return sources
