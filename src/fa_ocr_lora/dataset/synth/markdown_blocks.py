"""Block parser for the narrow Markdown dialect the letter bodies use.

The text stage produced bodies in a *known, small* subset of Markdown, so this
is a purpose-built parser rather than a Markdown library. Two properties a
general library would not give us:

* Every block comes back as a typed object the template can tag individually,
  which is what lets each paragraph, list item, and table cell own a bounding
  box in the annotation.
* Generator scaffolding that never belongs on a rendered letter - literal
  ``پاراگراف ۱`` markers, a ``موضوع:`` line duplicating the subject field, an
  unsubstituted ``signatory_title`` placeholder - is dropped here, so the
  rendered page and its ground truth agree.

Measured over the 1600-row corpus, the dialect is: ``**bold**`` spans, Persian
(``۱.``) and ASCII (``1.``) ordered lists, abjad lettering (``الف)``), ``-``
and ``•`` bullets, and GitHub-style pipe tables. There are no fenced code
blocks and no ``#`` headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Iterable, Iterator, Sequence

__all__ = [
    "BlockKind",
    "InlineSpan",
    "Block",
    "parse_body",
    "clean_body",
    "blocks_to_markdown",
]


class BlockKind(StrEnum):
    """The kinds of block a letter body can contain."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    ORDERED_ITEM = "ordered_item"
    BULLET_ITEM = "bullet_item"
    TABLE = "table"


#: Religious openers the text stage prefixed to bodies. They are stripped here
#: and re-added by the page composer as its own annotated element, so the
#: opener's placement and styling are the renderer's decision.
RELIGIOUS_OPENER_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:بسمه\s*الرحمن\s*الرحیم"
    r"|بسم\s*الله\s*الرحمن\s*الرحیم"
    r"|بسمه\s*تعالی"
    r"|به\s*نام\s*خدا(?:وند)?(?:\s*بخشنده\s*مهربان)?"
    r"|هوالعلیم"
    r"|هو\s*الله)\s*[.:،]?\s*",
    re.UNICODE,
)

#: Scaffolding the drafting model emitted as structure hints. These are not
#: letter content and would teach the OCR model to expect literal
#: "Paragraph 1" headings, so they never reach the page.
_SCAFFOLD_RE: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^\s*پاراگراف\s*[۰-۹0-9]*\s*[:.]?\s*$"),
    re.compile(r"^\s*\**\s*(?:بند|قسمت|بخش)\s*[۰-۹0-9]+\s*\**\s*[:.]?\s*$"),
    re.compile(r"^\s*signatory_title\s*$"),
    re.compile(r"^\s*\{\{?\s*\w+\s*\}?\}\s*$"),
    re.compile(r"^\s*به\s*[‌\s]*اسناد\s*$"),
)

#: A ``موضوع:`` line inside the body duplicates the ``subject`` field, which
#: the composer renders as its own element.
_SUBJECT_LINE_RE: Final[re.Pattern[str]] = re.compile(r"^\s*\**\s*موضوع\s*\**\s*[:：]")

#: Ordered markers: ASCII digits, Persian digits, or abjad lettering, closed by
#: ``.``, ``)``, ``-``, or an en dash.
_ORDERED_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<marker>(?:[0-9]{1,2}|[۰-۹]{1,2}|الف|ب|ج|د|ه|و|ز|ح|ط|ی)"
    r"\s*[.)\-–])"
    r"[ \t]+(?P<text>\S.*)$"
)

_BULLET_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-•*•])[ \t]+(?P<text>\S.*)$"
)

_BOLD_RE: Final[re.Pattern[str]] = re.compile(r"\*\*(?P<text>.+?)\*\*")

#: A table row is any line whose first non-space character is a pipe.
_TABLE_ROW_RE: Final[re.Pattern[str]] = re.compile(r"^\s*\|.*\|?\s*$")

#: The ``|---|---|`` alignment row that follows a table header.
_TABLE_SEPARATOR_RE: Final[re.Pattern[str]] = re.compile(r"^\s*\|[\s:|\-─–]+\|?\s*$")

#: Longest a line can be and still read as a section heading rather than prose.
_HEADING_MAX_CHARS: Final[int] = 70


@dataclass(frozen=True, slots=True)
class InlineSpan:
    """A run of text with uniform emphasis."""

    text: str
    bold: bool = False


@dataclass(frozen=True, slots=True)
class Block:
    """One structural block of a letter body.

    Attributes:
        kind: Which kind of block this is.
        text: Plain text with list markers and emphasis syntax removed. This
            is what lands in the annotation's ``text`` field.
        spans: The same text split into emphasis runs, for rendering.
        marker: The original list marker (``۱.``, ``-``, ``الف)``), empty for
            non-list blocks.
        indent: Nesting depth, zero for top level.
        rows: Table rows including the header, empty for non-table blocks.
    """

    kind: BlockKind
    text: str
    spans: tuple[InlineSpan, ...] = ()
    marker: str = ""
    indent: int = 0
    rows: tuple[tuple[str, ...], ...] = ()

    @property
    def is_list_item(self) -> bool:
        """Return whether this block renders as a list item."""
        return self.kind in (BlockKind.ORDERED_ITEM, BlockKind.BULLET_ITEM)

    @property
    def is_table(self) -> bool:
        """Return whether this block carries tabular data."""
        return self.kind is BlockKind.TABLE


def _split_spans(text: str) -> tuple[InlineSpan, ...]:
    """Split ``**bold**`` runs out of a line.

    Args:
        text: Raw line text, possibly containing emphasis markers.

    Returns:
        Consecutive spans covering the whole line.
    """
    spans: list[InlineSpan] = []
    cursor = 0
    for match in _BOLD_RE.finditer(text):
        if match.start() > cursor:
            spans.append(InlineSpan(text[cursor : match.start()], bold=False))
        spans.append(InlineSpan(match.group("text"), bold=True))
        cursor = match.end()
    if cursor < len(text):
        spans.append(InlineSpan(text[cursor:], bold=False))
    return tuple(span for span in spans if span.text)


def _strip_emphasis(text: str) -> str:
    """Remove emphasis syntax, leaving the plain reading text."""
    return _BOLD_RE.sub(lambda match: match.group("text"), text).replace("**", "")


def _is_fully_bold(text: str) -> bool:
    """Return whether a line is entirely wrapped in a single bold span."""
    stripped = text.strip()
    match = _BOLD_RE.fullmatch(stripped)
    return match is not None and "**" not in match.group("text")


def _parse_table_row(line: str) -> tuple[str, ...]:
    """Split one pipe-table line into trimmed cell values."""
    trimmed = line.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]
    return tuple(_strip_emphasis(cell).strip() for cell in trimmed.split("|"))


def clean_body(body: str) -> str:
    """Strip the religious opener and generator scaffolding from a raw body.

    Args:
        body: ``letter.body`` exactly as stored in the plaintext JSON.

    Returns:
        The body with the opener and scaffolding lines removed, and runs of
        blank lines collapsed.
    """
    without_opener = RELIGIOUS_OPENER_RE.sub("", body, count=1)
    kept: list[str] = []
    for line in without_opener.splitlines():
        if any(pattern.match(line) for pattern in _SCAFFOLD_RE):
            continue
        if _SUBJECT_LINE_RE.match(line):
            continue
        kept.append(line.rstrip())
    collapsed: list[str] = []
    for line in kept:
        if not line and collapsed and not collapsed[-1]:
            continue
        collapsed.append(line)
    return "\n".join(collapsed).strip("\n")


def _consume_table(lines: Sequence[str], start: int) -> tuple[Block | None, int]:
    """Read a pipe table beginning at ``start``.

    Args:
        lines: All body lines.
        start: Index of the candidate header row.

    Returns:
        The parsed block and the index just past the table, or ``(None,
        start)`` when the lines do not form a table.
    """
    if start + 1 >= len(lines) or not _TABLE_SEPARATOR_RE.match(lines[start + 1]):
        return None, start
    header = _parse_table_row(lines[start])
    rows: list[tuple[str, ...]] = [header]
    cursor = start + 2
    while cursor < len(lines) and _TABLE_ROW_RE.match(lines[cursor]):
        if _TABLE_SEPARATOR_RE.match(lines[cursor]):
            cursor += 1
            continue
        row = _parse_table_row(lines[cursor])
        # Pad or trim so every row matches the header width; ragged rows would
        # otherwise desynchronise the per-cell bounding boxes.
        if len(row) < len(header):
            row = row + ("",) * (len(header) - len(row))
        rows.append(row[: len(header)])
        cursor += 1
    if len(rows) < 2:
        return None, start
    text = " ".join(" ".join(cell for cell in row if cell) for row in rows).strip()
    return Block(kind=BlockKind.TABLE, text=text, rows=tuple(rows)), cursor


def parse_body(body: str) -> tuple[Block, ...]:
    """Parse a letter body into typed blocks.

    Args:
        body: Raw ``letter.body`` text; cleaned internally, so callers need
            not call :func:`clean_body` first.

    Returns:
        The body's blocks in reading order.
    """
    lines = clean_body(body).splitlines()
    blocks: list[Block] = []
    pending: list[str] = []

    def flush_paragraph() -> None:
        """Emit any buffered prose lines as one paragraph."""
        if not pending:
            return
        joined = " ".join(part.strip() for part in pending if part.strip())
        pending.clear()
        if not joined:
            return
        plain = _strip_emphasis(joined).strip()
        if not plain:
            return
        kind = (
            BlockKind.HEADING
            if _is_fully_bold(joined) and len(plain) <= _HEADING_MAX_CHARS
            else BlockKind.PARAGRAPH
        )
        blocks.append(Block(kind=kind, text=plain, spans=_split_spans(joined)))

    index = 0
    while index < len(lines):
        line = lines[index]

        if not line.strip():
            flush_paragraph()
            index += 1
            continue

        if _TABLE_ROW_RE.match(line):
            table, next_index = _consume_table(lines, index)
            if table is not None:
                flush_paragraph()
                blocks.append(table)
                index = next_index
                continue

        ordered = _ORDERED_RE.match(line)
        bullet = None if ordered else _BULLET_RE.match(line)
        match = ordered or bullet
        if match:
            flush_paragraph()
            raw_text = match.group("text").strip()
            plain = _strip_emphasis(raw_text).strip()
            indent = len(match.group("indent").expandtabs(4)) // 3
            kind = BlockKind.ORDERED_ITEM if ordered else BlockKind.BULLET_ITEM
            if _is_fully_bold(raw_text) and len(plain) <= _HEADING_MAX_CHARS:
                kind = BlockKind.HEADING
            blocks.append(
                Block(
                    kind=kind,
                    text=plain,
                    spans=_split_spans(raw_text),
                    marker=match.group("marker").strip(),
                    indent=indent,
                )
            )
            index += 1
            continue

        pending.append(line)
        index += 1

    flush_paragraph()
    return tuple(blocks)


def blocks_to_markdown(blocks: Iterable[Block]) -> str:
    """Render parsed blocks back to Markdown for the ``.md`` artifact.

    Args:
        blocks: Blocks as returned by :func:`parse_body`.

    Returns:
        A Markdown document mirroring what was rendered to the page.
    """
    lines: list[str] = []
    for block in blocks:
        if block.kind is BlockKind.TABLE:
            if not block.rows:
                continue
            header, *body_rows = block.rows
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            lines.extend("| " + " | ".join(row) + " |" for row in body_rows)
            lines.append("")
        elif block.kind is BlockKind.HEADING:
            prefix = f"{block.marker} " if block.marker else ""
            lines.append(f"### {prefix}{block.text}")
            lines.append("")
        elif block.is_list_item:
            pad = "  " * block.indent
            marker = block.marker or "-"
            lines.append(f"{pad}{marker} {block.text}")
        else:
            lines.append(block.text)
            lines.append("")
    return "\n".join(lines).strip() + "\n"
