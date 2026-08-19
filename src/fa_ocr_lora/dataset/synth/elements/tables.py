"""Table elements: parse the body's own table, or synthesise a plausible one.

Two sources feed this module. Roughly 118 of the 2000 letter bodies contain a
genuine Markdown pipe table, and those are used verbatim - real data always
beats invented data as a training target. The rest of the letters flagged
``chart_table == "table"`` only *mention* a table in prose, so a deterministic
table is synthesised from the letter's subject and seed.

The rendered fragment carries ``data-ann-*`` attributes on the table and on
every cell, which is what gives the annotation a box per cell rather than one
box for the whole grid.
"""

from __future__ import annotations

import html
import random
from dataclasses import dataclass
from typing import Final, Sequence

from ....config import TableConfig, TableVocabulary
from ...common.io_utils import to_persian_digits
from ..markdown_blocks import Block

__all__ = [
    "TableSpec",
    "build_table",
    "synthesize_table",
    "table_to_html",
    "table_to_markdown",
]

#: Fallback for a caller that does not hand over the loaded config - the test
#: suite, mostly. Read off an instance rather than restated here, so a literal
#: cannot drift from the dataclass default.
_DEFAULT_TABLE: Final[TableConfig] = TableConfig()

#: Rials per unit of ``elements.table.amount_millions``. Intentionally not a
#: knob: it is fixed by the ``مبلغ (ریال)`` header wording that declares the
#: unit, so changing one without the other would print a wrong figure.
_AMOUNT_SCALE: Final[int] = 1_000_000

#: The header sets, the cell-value pools they draw from, the month names and the
#: caption wordings all live in ``configs/vocabulary_fa.yaml`` and arrive as
#: ``config.vocabulary.elements.table``. A header set is one full row rather than
#: a bag of columns, so a synthesised table reads as a coherent document; and the
#: header text is what *chooses* the value pool below, which is why retuning the
#: two together in one file matters.
_DEFAULT_VOCABULARY: Final[TableVocabulary] = TableVocabulary()


def _group_digits(value: int) -> str:
    """Format an integer with thousands separators in Persian digits."""
    return to_persian_digits(f"{value:,}")


def _synth_cell(
    header: str,
    row_index: int,
    rng: random.Random,
    config: TableConfig = _DEFAULT_TABLE,
    vocabulary: TableVocabulary = _DEFAULT_VOCABULARY,
) -> str:
    """Invent one cell value appropriate to its column header.

    Args:
        header: The column's header text.
        row_index: Zero-based row index, used for the ``ردیف`` column.
        rng: The sample's seeded RNG.
        config: The ``elements.table`` section, supplying each value range.
        vocabulary: The ``elements.table`` pools from ``vocabulary_fa.yaml``,
            supplying every non-numeric value.

    Returns:
        A Persian cell value.
    """
    if "ردیف" in header:
        return to_persian_digits(row_index + 1)
    if "مبلغ" in header or "هزینه" in header:
        return _group_digits(config.amount_millions.draw(rng) * _AMOUNT_SCALE)
    if "تعداد" in header:
        return to_persian_digits(config.count.draw(rng))
    if "درصد" in header:
        return f"{to_persian_digits(config.percent.draw(rng))}٪"
    if "زمان" in header or "تاریخ" in header:
        return (
            f"{to_persian_digits(config.day.draw(rng))} "
            f"{rng.choice(vocabulary.months)} "
            f"{to_persian_digits(config.year.draw(rng))}"
        )
    if "وضعیت" in header:
        return rng.choice(vocabulary.status_values)
    if "مدرک" in header or "نام" in header:
        return rng.choice(vocabulary.document_labels)
    if "مرحله" in header:
        return rng.choice(vocabulary.stage_labels)
    if "واحد" in header:
        return rng.choice(vocabulary.unit_labels)
    if "وظیفه" in header or "توضیح" in header:
        return rng.choice(vocabulary.detail_values)
    return rng.choice(vocabulary.service_labels)


@dataclass(frozen=True, slots=True)
class TableSpec:
    """A table ready to render, plus structured content for the annotation.

    Attributes:
        header: Column headers.
        rows: Body rows, each the same width as ``header``.
        caption: Optional caption rendered above the table.
        synthesized: ``True`` when the data was invented rather than taken
            from the letter body. Recorded in the annotation so a consumer can
            weight real and synthetic content differently.
    """

    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    caption: str = ""
    synthesized: bool = False

    def __post_init__(self) -> None:
        if not self.header:
            raise ValueError("table must have at least one column")
        for index, row in enumerate(self.rows):
            if len(row) != len(self.header):
                raise ValueError(
                    f"table row {index} has {len(row)} cells, "
                    f"expected {len(self.header)}"
                )

    @property
    def column_count(self) -> int:
        """Number of columns."""
        return len(self.header)

    @property
    def row_count(self) -> int:
        """Number of body rows, excluding the header."""
        return len(self.rows)

    @property
    def text(self) -> str:
        """Flat reading text of the whole table, for the transcription."""
        lines = [" | ".join(self.header)]
        lines.extend(" | ".join(row) for row in self.rows)
        if self.caption:
            lines.insert(0, self.caption)
        return "\n".join(lines)

    def as_content(self) -> dict[str, object]:
        """Return the annotation's structured payload for this table."""
        return {
            "caption": self.caption,
            "columns": list(self.header),
            "rows": [list(row) for row in self.rows],
            "row_count": self.row_count,
            "column_count": self.column_count,
            "synthesized": self.synthesized,
        }


def _from_block(block: Block) -> TableSpec:
    """Convert a parsed Markdown table block into a spec."""
    header, *rows = block.rows
    return TableSpec(header=header, rows=tuple(rows), synthesized=False)


def synthesize_table(
    subject: str,
    rng: random.Random,
    config: TableConfig = _DEFAULT_TABLE,
    vocabulary: TableVocabulary = _DEFAULT_VOCABULARY,
) -> TableSpec:
    """Invent a Persian table consistent with the letter's subject.

    Args:
        subject: The letter subject, used to bias the caption wording.
        rng: The sample's seeded RNG.
        config: The ``elements.table`` section, supplying the row count and
            every invented value's range.
        vocabulary: The ``elements.table`` pools from ``vocabulary_fa.yaml``:
            header sets, cell values, months and captions.

    Returns:
        A synthesised table. The column count is the width of the header row
        drawn from the vocabulary, 3-4 columns as shipped; the row count comes
        from ``config.rows``.
    """
    header = rng.choice(vocabulary.header_sets)
    row_count = config.rows.draw(rng)
    rows = tuple(
        tuple(_synth_cell(column, index, rng, config, vocabulary) for column in header)
        for index in range(row_count)
    )
    caption = rng.choice(vocabulary.captions)
    if "مبلغ" in " ".join(header) and "هزینه" in subject:
        caption = vocabulary.cost_caption
    return TableSpec(header=header, rows=rows, caption=caption, synthesized=True)


def build_table(
    body_tables: Sequence[Block],
    subject: str,
    rng: random.Random,
    config: TableConfig = _DEFAULT_TABLE,
    vocabulary: TableVocabulary = _DEFAULT_VOCABULARY,
) -> TableSpec:
    """Return the table to render, preferring real data over synthesis.

    Args:
        body_tables: Table blocks found in the letter body.
        subject: Letter subject, for caption wording.
        rng: The sample's seeded RNG.
        config: The ``elements.table`` section. Reaches only the synthesis
            path - a real table from the body is rendered verbatim.
        vocabulary: The ``elements.table`` pools, likewise synthesis-only.

    Returns:
        A spec drawn from the body when one exists, else a synthesised one.
    """
    usable = [block for block in body_tables if block.rows and len(block.rows) > 1]
    if usable:
        return _from_block(usable[0])
    return synthesize_table(subject, rng, config, vocabulary)


def table_to_html(spec: TableSpec, element_id: str) -> str:
    """Render a table spec to annotated HTML.

    Every cell carries its own ``data-ann-type``, so the extractor emits a
    bounding box per cell while knowing nothing about tables.

    Args:
        spec: The table to render.
        element_id: Stable id prefix, e.g. ``body.table0``.

    Returns:
        An HTML fragment.
    """
    safe_id = html.escape(element_id, quote=True)
    parts: list[str] = [
        f'<figure class="element element-table" data-ann-type="table" '
        f'data-ann-id="{safe_id}">'
    ]
    if spec.caption:
        parts.append(
            f'<figcaption class="table-caption">'
            f"{html.escape(spec.caption)}</figcaption>"
        )
    parts.append("<table><thead><tr>")
    for column_index, column in enumerate(spec.header):
        cell_id = html.escape(f"{element_id}.h{column_index}", quote=True)
        parts.append(
            f'<th data-ann-type="table_cell" data-ann-id="{cell_id}" '
            f'data-ann-row="0" data-ann-col="{column_index}">'
            f"{html.escape(column)}</th>"
        )
    parts.append("</tr></thead><tbody>")
    for row_index, row in enumerate(spec.rows):
        parts.append("<tr>")
        for column_index, cell in enumerate(row):
            cell_id = html.escape(
                f"{element_id}.r{row_index}c{column_index}", quote=True
            )
            parts.append(
                f'<td data-ann-type="table_cell" data-ann-id="{cell_id}" '
                f'data-ann-row="{row_index + 1}" data-ann-col="{column_index}">'
                f"{html.escape(cell)}</td>"
            )
        parts.append("</tr>")
    parts.append("</tbody></table></figure>")
    return "".join(parts)


def table_to_markdown(spec: TableSpec) -> str:
    """Render a table spec as Markdown, for the ``.md`` artifact.

    Args:
        spec: The table to render.

    Returns:
        A Markdown table, preceded by its caption when present.
    """
    lines: list[str] = []
    if spec.caption:
        lines.extend((f"**{spec.caption}**", ""))
    lines.append("| " + " | ".join(spec.header) + " |")
    lines.append("|" + "|".join(["---"] * spec.column_count) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in spec.rows)
    return "\n".join(lines)
