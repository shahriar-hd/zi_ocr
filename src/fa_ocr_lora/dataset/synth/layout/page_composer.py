"""Assembly of one letter: specs in, a renderable HTML document and Markdown out.

This module owns *what* goes on the page and in what order. It draws every
random decision for a sample, calls the layout and element modules to build
their specs, turns those specs into HTML fragments, and hands the result to
:mod:`.render.browser`, which owns *how* the page rasterises. Nothing here
touches Playwright, so the whole composition step is testable in-process.

Three invariants hold the ground truth together:

* **Flow order is reading order.** The fragments are emitted into ``#flow`` in
  the order a person reads the letter, and :mod:`..annotation` only refines
  that within a line. A block moved here moves in the transcription too.
* **Ids are stable and structural.** ``meta.date``, ``body.p3``,
  ``body.table0.r2c1`` - an id says which region a box belongs to without the
  extractor knowing anything about letters.
* **Structured content travels beside the HTML, not inside it.** Each spec's
  ``as_content()`` payload is collected into :attr:`ComposedLetter.content_map`
  keyed by element id, so the annotation can attach a table's cell matrix or a
  chart's series to the right box without re-parsing the DOM.
"""

from __future__ import annotations

import html
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ....config import LetterVocabulary, PageSizeSpec, SynthConfig
from ...common.io_utils import sample_rng, to_persian_digits, weighted_choice
from ..elements.charts import ChartSpec, build_chart, chart_series_block
from ..elements.charts import chart_to_markdown, chart_to_svg
from ..elements.tables import TableSpec, build_table, table_to_html, table_to_markdown
from ..letter_source import LetterSource
from ..markdown_blocks import Block, BlockKind, InlineSpan
from ..metadata_block import (
    MetadataBlockSpec,
    build_metadata_block,
    metadata_to_html,
    metadata_to_markdown,
)
from ..render.fonts import font_face_css, load_font_library
from .header_footer import (
    FooterSpec,
    HeaderSpec,
    build_footer,
    build_header,
    footer_to_html,
    footer_to_markdown,
    header_to_html,
    header_to_markdown,
    opener_to_html,
)
from .signature import (
    SignatureSpec,
    build_signature,
    signature_block_to_html,
    signature_block_to_markdown,
)
from .stamp import StampSpec, build_stamp, city_from_address, stamp_to_html
from .texture import Texture, background_css, choose_texture
from .theme import BarStyle, Theme, build_theme

__all__ = ["PageGeometry", "ComposedLetter", "compose_letter"]

#: How the recipient, sender and subject lines are worded, and the attachment
#: list's heading, all live in ``configs/vocabulary_fa.yaml`` and arrive as
#: ``config.vocabulary.letter``. This module reads them off the config it is
#: handed, so there is no module-level pool here to fall out of step with the
#: file.


@dataclass(frozen=True, slots=True)
class PageGeometry:
    """The paper the template lays out against.

    Attributes:
        width_mm: Sheet width.
        height_mm: Sheet height.
        margin_mm: Printable margin on all four sides.
        name: Config name of the page size, for the annotation.
        orientation: ``portrait`` or ``landscape``.
    """

    width_mm: float
    height_mm: float
    margin_mm: float
    name: str
    orientation: str

    @classmethod
    def from_spec(cls, spec: PageSizeSpec, margin_mm: float) -> "PageGeometry":
        """Build geometry from a config page size.

        Args:
            spec: The chosen page size, already oriented.
            margin_mm: Margin drawn for this sample.

        Returns:
            The template's view of the paper.
        """
        return cls(
            width_mm=spec.width_mm,
            height_mm=spec.height_mm,
            margin_mm=margin_mm,
            name=spec.name,
            orientation=spec.orientation,
        )


@dataclass(frozen=True, slots=True)
class ComposedLetter:
    """One letter, composed but not yet rasterised.

    Attributes:
        sample_id: Render-stage id, e.g. ``synth_000000``.
        source: The plaintext record this was built from.
        seed: Seed every decision here was derived from.
        page_spec: The config page size drawn for this sample.
        geometry: The template's view of the paper.
        theme: Palette, typography and bar treatments.
        texture: Paper scan composited under the page by the scan stage,
            or ``None`` for paper with no grain.
        header: The letterhead spec, enabled or not.
        footer: The footer spec, enabled or not.
        metadata: The administrative metadata block.
        table: The rendered table, when the letter carries one.
        chart: The rendered chart, when the letter carries one.
        signature: The prepared signature, or ``None``.
        stamp: The prepared seal, or ``None``.
        context: Ready-to-render template context.
        markdown: The letter as Markdown, mirroring the page.
        content_map: Structured payloads keyed by element id.
    """

    sample_id: str
    source: LetterSource
    seed: int
    page_spec: PageSizeSpec
    geometry: PageGeometry
    theme: Theme
    texture: Texture | None
    header: HeaderSpec
    footer: FooterSpec
    metadata: MetadataBlockSpec
    table: TableSpec | None
    chart: ChartSpec | None
    signature: SignatureSpec | None
    stamp: StampSpec | None
    context: Mapping[str, Any]
    markdown: str
    content_map: Mapping[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def texture_name(self) -> str | None:
        """Return the background scan's filename, or ``None``."""
        return self.texture.name if self.texture is not None else None


def _spans_to_html(spans: Sequence[InlineSpan], fallback: str) -> str:
    """Render inline spans, keeping bold runs bold.

    Args:
        spans: Emphasis runs from the block parser.
        fallback: Plain text to use when the block carries no spans.

    Returns:
        Escaped HTML for the inline content.
    """
    if not spans:
        return html.escape(fallback)
    parts: list[str] = []
    for span in spans:
        escaped = html.escape(span.text)
        parts.append(f"<strong>{escaped}</strong>" if span.bold else escaped)
    return "".join(parts)


def _paragraph_html(block: Block, element_id: str) -> str:
    """Render one prose paragraph."""
    return (
        f'<p class="para" data-ann-type="paragraph" data-ann-id="{element_id}">'
        f"{_spans_to_html(block.spans, block.text)}</p>"
    )


def _heading_html(block: Block, element_id: str) -> str:
    """Render one section heading."""
    marker = f"{html.escape(block.marker)} " if block.marker else ""
    return (
        f'<div class="heading" data-ann-type="heading" data-ann-id="{element_id}">'
        f"{marker}{_spans_to_html(block.spans, block.text)}</div>"
    )


def _list_item_html(block: Block, element_id: str) -> str:
    """Render one list item, marker and text in separate boxes' parent.

    The marker stays inside the item's own box rather than getting one of its
    own: an OCR model transcribing ``۱. متن`` reads one line, and splitting the
    numeral off would ask it to emit two.
    """
    indent = min(block.indent, 2)
    classes = f"list-item indent-{indent}" if indent else "list-item"
    marker = block.marker or "•"
    return (
        f'<div class="{classes}" data-ann-type="list_item" '
        f'data-ann-id="{element_id}">'
        f'<span class="list-marker">{html.escape(marker)}</span>'
        f'<span class="list-text">{_spans_to_html(block.spans, block.text)}</span>'
        "</div>"
    )


def _party_html(text: str, kind: str, element_id: str) -> str:
    """Render a sender or recipient line.

    Args:
        text: The line as printed, prefix included.
        kind: ``sender`` or ``recipient``, which becomes the element type.
        element_id: Stable id.

    Returns:
        An HTML fragment, empty when there is no text.
    """
    if not text.strip():
        return ""
    return (
        f'<div class="letter-party" data-ann-type="{kind}" '
        f'data-ann-id="{element_id}">{html.escape(text)}</div>'
    )


def _subject_html(text: str) -> str:
    """Render the subject line, which the CSS sets in the display face."""
    if not text.strip():
        return ""
    return (
        '<div class="letter-subject" data-ann-type="subject" '
        f'data-ann-id="subject">{html.escape(text)}</div>'
    )


def _attachments_html(title: str, lines: Sequence[str]) -> str:
    """Render the attachment list as its own annotated region.

    Args:
        title: Persian heading for the list.
        lines: One entry per attachment, markers already stripped.

    Returns:
        An HTML fragment, empty when the letter has no attachments.
    """
    if not lines:
        return ""
    parts: list[str] = [
        '<div class="attachments-block" data-ann-type="attachments_block" '
        'data-ann-id="attachments">',
        f'<div class="attachments-title">{html.escape(title)}</div>',
    ]
    for index, line in enumerate(lines):
        marker = to_persian_digits(f"{index + 1}.")
        parts.append(
            f'<div class="list-item" data-ann-type="list_item" '
            f'data-ann-id="attachments.i{index}">'
            f'<span class="list-marker">{marker}</span>'
            f'<span class="list-text">{html.escape(line)}</span></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _element_anchor(blocks: Sequence[Block]) -> int:
    """Return where a synthesised table or chart is inserted into the body.

    A figure dropped before the opening paragraph reads as a report rather than
    a letter, so it goes after the first paragraph - or at the end when the
    body has none.

    Args:
        blocks: The body's renderable blocks.

    Returns:
        An insertion index into ``blocks``.
    """
    for index, block in enumerate(blocks):
        if block.kind is BlockKind.PARAGRAPH:
            return index + 1
    return len(blocks)


def _table_from_block(block: Block) -> TableSpec | None:
    """Wrap a body table block as a spec, for the second and later tables.

    Args:
        block: A table block from the parser.

    Returns:
        The spec, or ``None`` when the block has no body rows.
    """
    if len(block.rows) < 2:
        return None
    return TableSpec(
        header=block.rows[0],
        rows=tuple(block.rows[1:]),
        caption="",
        synthesized=False,
    )


@dataclass(slots=True)
class _BodyStream:
    """Accumulator for the body walk: fragments, Markdown, and content."""

    fragments: list[str] = field(default_factory=list)
    markdown: list[str] = field(default_factory=list)
    content: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, fragment: str, markdown: str) -> None:
        """Append one block's HTML and its Markdown mirror."""
        if fragment:
            self.fragments.append(fragment)
        if markdown:
            self.markdown.append(markdown)


def _compose_body(
    source: LetterSource,
    table: TableSpec | None,
    chart: ChartSpec | None,
    accent: str,
) -> _BodyStream:
    """Walk the body, interleaving the letter's one structured element.

    Args:
        source: The letter being composed.
        table: The table to render, or ``None``.
        chart: The chart to render, or ``None``.
        accent: Palette accent, used as the chart's mark colour.

    Returns:
        The accumulated fragments, Markdown and structured content.
    """
    blocks = source.renderable_blocks
    stream = _BodyStream()
    counters = {"paragraph": 0, "heading": 0, "item": 0, "table": 0}

    body_table_present = bool(source.body_tables)
    inserts: dict[int, str] = {}
    if chart is not None:
        consumed = chart_series_block(source.body_blocks)
        anchor = next(
            (
                index
                for index, block in enumerate(source.body_blocks)
                if block is consumed
            ),
            None,
        )
        if anchor is None:
            anchor = _element_anchor(blocks)
        inserts[min(anchor, len(blocks))] = "chart"
    if table is not None and not body_table_present:
        inserts.setdefault(_element_anchor(blocks), "table")

    pending_table = table

    def emit_insert(index: int) -> None:
        """Emit whichever element is anchored at ``index``."""
        nonlocal pending_table
        kind = inserts.pop(index, "")
        if kind == "chart" and chart is not None:
            element_id = "body.chart0"
            stream.add(chart_to_svg(chart, element_id, accent), chart_to_markdown(chart))
            stream.content[element_id] = chart.as_content()
        elif kind == "table" and pending_table is not None:
            element_id = f"body.table{counters['table']}"
            stream.add(
                table_to_html(pending_table, element_id),
                table_to_markdown(pending_table),
            )
            stream.content[element_id] = pending_table.as_content()
            counters["table"] += 1
            pending_table = None

    for index, block in enumerate(blocks):
        emit_insert(index)

        if block.is_table:
            spec = pending_table
            if spec is None:
                spec = _table_from_block(block)
            if spec is None:
                continue
            pending_table = None
            element_id = f"body.table{counters['table']}"
            stream.add(table_to_html(spec, element_id), table_to_markdown(spec))
            stream.content[element_id] = spec.as_content()
            counters["table"] += 1
            continue

        if block.kind is BlockKind.HEADING:
            element_id = f"body.h{counters['heading']}"
            counters["heading"] += 1
            marker = f"{block.marker} " if block.marker else ""
            stream.add(_heading_html(block, element_id), f"### {marker}{block.text}")
            continue

        if block.is_list_item:
            element_id = f"body.li{counters['item']}"
            counters["item"] += 1
            pad = "  " * min(block.indent, 2)
            marker = block.marker or "-"
            stream.add(
                _list_item_html(block, element_id),
                f"{pad}{marker} {block.text}",
            )
            continue

        element_id = f"body.p{counters['paragraph']}"
        counters["paragraph"] += 1
        stream.add(_paragraph_html(block, element_id), block.text)

    for index in sorted(inserts):
        emit_insert(index)
    return stream


def _pick_page(config: SynthConfig, rng: random.Random) -> tuple[PageSizeSpec, float]:
    """Draw this sample's paper.

    Args:
        config: Loaded configuration.
        rng: The sample's ``page`` RNG.

    Returns:
        ``(page size, margin in mm)``.
    """
    pool = tuple((spec, spec.weight) for spec in config.render.page_sizes)
    page = weighted_choice(pool, rng)
    low, high = config.render.margin_mm_range
    return page, round(rng.uniform(low, high), 1)


def _compose_frame(
    source: LetterSource,
    header: HeaderSpec,
    sender_line_probability: float,
    rng: random.Random,
    vocabulary: LetterVocabulary,
) -> tuple[str, str, str]:
    """Draw the recipient, sender and subject lines as printed.

    Args:
        source: The letter being composed.
        header: The letterhead, which may already name the sender.
        sender_line_probability: Chance of repeating the sender under a
            letterhead that already names it.
        rng: The sample's ``compose`` RNG.
        vocabulary: The ``letter`` pools from ``vocabulary_fa.yaml``. Each
            prefix carries its own trailing space, so it is concatenated
            rather than joined.

    Returns:
        ``(recipient line, sender line, subject line)``, any of which may be
        empty when the record has no value for it.
    """
    recipient = source.destination_organization.strip()
    if recipient:
        recipient = weighted_choice(vocabulary.recipient_prefixes, rng) + recipient

    sender_name = (source.company_name or source.logo_title).strip()
    # Repeating the sender under a letterhead that already names it is
    # something real letters do, but rarely; without a letterhead the page
    # would otherwise never say who wrote it.
    wants_sender = rng.random() < sender_line_probability or not header.enabled
    sender = ""
    if sender_name and wants_sender:
        sender = weighted_choice(vocabulary.sender_prefixes, rng) + sender_name

    subject = source.subject.strip()
    if subject:
        subject = weighted_choice(vocabulary.subject_prefixes, rng) + subject
    return recipient, sender, subject


def _compose_markdown(
    *,
    header: HeaderSpec,
    footer: FooterSpec,
    metadata: MetadataBlockSpec,
    recipient: str,
    sender: str,
    subject: str,
    body_markdown: Sequence[str],
    attachment_title: str,
    attachment_lines: Sequence[str],
    signatory_title: str,
    signature: SignatureSpec | None,
    has_stamp: bool,
) -> str:
    """Render the letter as Markdown, in the same order as the page.

    This is a *second view of the same content*, not a second generation pass:
    every section comes from the spec that produced the corresponding HTML, so
    the ``.md`` artifact and the render cannot disagree about what the letter
    says.

    Args:
        header: Letterhead spec.
        footer: Footer spec.
        metadata: Administrative metadata block.
        recipient: Recipient line as printed.
        sender: Sender line as printed, possibly empty.
        subject: Subject line as printed.
        body_markdown: Per-block Markdown from the body walk.
        attachment_title: Heading used for the attachment list.
        attachment_lines: Attachment entries.
        signatory_title: The printed post.
        signature: Prepared signature, or ``None``.
        has_stamp: Whether a seal was composited.

    Returns:
        A Markdown document ending in a single newline.
    """
    sections: list[str] = []
    head = header_to_markdown(header)
    if head:
        sections.append(head)
    meta = metadata_to_markdown(metadata)
    if meta:
        sections.append(meta)
    for line in (recipient, sender):
        if line:
            sections.append(f"**{line}**")
    if subject:
        sections.append(f"## {subject}")
    sections.extend(part for part in body_markdown if part)
    if attachment_lines:
        entries = "\n".join(
            f"{to_persian_digits(f'{index + 1}.')} {line}"
            for index, line in enumerate(attachment_lines)
        )
        sections.append(f"**{attachment_title}**\n{entries}")
    closing = signature_block_to_markdown(
        signatory_title, signature, has_stamp=has_stamp
    )
    if closing:
        sections.append(closing)
    tail = footer_to_markdown(footer)
    if tail:
        sections.append(tail)
    return "\n\n".join(sections).strip() + "\n"


def compose_letter(
    source: LetterSource,
    config: SynthConfig,
    *,
    sample_id: str,
    seed: int | None = None,
) -> ComposedLetter:
    """Compose one letter: every random decision, every fragment, the Markdown.

    All randomness is derived from ``(seed, purpose, source.sample_index)``, so
    the same row and seed always produce the same page - the reproducibility
    requirement in CLAUDE.md 8.

    Args:
        source: The letter, already loaded from its plaintext record.
        config: Loaded synthetic-render configuration.
        sample_id: Render-stage id to record in the annotation.
        seed: Master seed; defaults to the configured one.

    Returns:
        The composed letter, ready for :meth:`.render.browser.PageRenderer.render`.
    """
    master_seed = config.seed if seed is None else seed
    index = source.sample_index

    def rng_for(purpose: str) -> random.Random:
        """Return the sample's RNG for one purpose."""
        return sample_rng(master_seed, purpose, index)

    page_spec, margin_mm = _pick_page(config, rng_for("page"))
    geometry = PageGeometry.from_spec(page_spec, margin_mm)
    theme = build_theme(source.brand_hex, config.theme, rng_for("theme"))
    palette = theme.palette
    probabilities = config.probabilities
    layout = config.layout
    words = config.vocabulary

    texture = choose_texture(
        config.paths.texture_dir, probabilities.texture, rng_for("texture")
    )

    company_name = (source.company_name or source.logo_title).strip()
    header = build_header(
        company_name,
        source.company_category,
        source.logo_path,
        theme.header_bar,
        rng_for("header"),
        layout=layout,
        header_probability=probabilities.header,
        slogan_probability=probabilities.header_slogan,
        opener_probability=probabilities.religious_opener,
        category_probability=probabilities.header_category,
        vocabulary=words.header,
    )
    footer = build_footer(
        source.contact.labelled_fields,
        theme.footer_bar,
        rng_for("footer"),
        layout=layout,
        footer_probability=probabilities.footer,
        page_number_probability=probabilities.page_number,
        label_probability=probabilities.footer_labels,
        vocabulary=words.header,
    )
    metadata = build_metadata_block(
        source.letter_number,
        source.registration_number,
        source.date,
        source.national_id,
        bool(source.attachment_lines),
        rng_for("metadata"),
        logo_side=header.logo_side,
        layout=layout,
        same_side_probability=probabilities.metadata_same_side,
        boxed_probability=probabilities.metadata_boxed,
        label_probability=probabilities.metadata_labels,
        vocabulary=words.metadata_block,
    )

    table = (
        build_table(
            source.body_tables,
            source.subject,
            rng_for("table"),
            config.elements.table,
            words.elements.table,
        )
        if source.wants_table or source.body_tables
        else None
    )
    chart = (
        build_chart(
            source.body_blocks,
            rng_for("chart"),
            config.elements.chart,
            words.elements.chart,
        )
        if source.wants_chart
        else None
    )
    signature = build_signature(
        source.signature_path,
        rng_for("signature"),
        probability=probabilities.signature,
        config=config.signature,
    )
    stamp = build_stamp(
        source.logo_path,
        company_name,
        city_from_address(source.contact.address, words.stamp),
        config.theme.stamp_inks,
        rng_for("stamp"),
        probability=probabilities.stamp,
        config=config.stamp,
        vocabulary=words.stamp,
    )

    compose_rng = rng_for("compose")
    recipient, sender, subject = _compose_frame(
        source, header, probabilities.sender_line, compose_rng, words.letter
    )
    subject_underline = compose_rng.random() < probabilities.subject_underline
    signature_side = weighted_choice(layout.signature_sides, compose_rng)
    attachment_title = compose_rng.choice(words.letter.attachment_titles)
    stripe_mm = round(layout.stripe_mm.draw(compose_rng), 1)

    body = _compose_body(source, table, chart, palette.accent)
    stamp_html = stamp_to_html(stamp) if stamp is not None else ""
    closing_html = signature_block_to_html(
        source.signatory_title,
        signature,
        side=signature_side,
        stamp_html=stamp_html,
    )

    flow_blocks = tuple(
        fragment
        for fragment in (
            opener_to_html(header),
            metadata_to_html(metadata),
            _party_html(recipient, "recipient", "party.recipient"),
            _party_html(sender, "sender", "party.sender"),
            _subject_html(subject),
            *body.fragments,
            _attachments_html(attachment_title, source.attachment_lines),
            closing_html,
        )
        if fragment
    )

    content_map: dict[str, dict[str, Any]] = dict(body.content)
    content_map["header"] = header.as_content()
    content_map["footer"] = footer.as_content()
    content_map["meta"] = metadata.as_content()
    for item in metadata.fields:
        content_map[f"meta.{item.key}"] = item.as_content()
    if texture is not None:
        content_map["texture"] = texture.as_content()
    if signature is not None:
        content_map["signature.image"] = signature.as_content()
    if stamp is not None:
        content_map["stamp"] = stamp.as_content()

    library = load_font_library(config.paths.fonts_dir)
    font_css = font_face_css(
        (theme.typography.body_family, theme.typography.display_family), library
    )

    context: dict[str, Any] = {
        "sample_id": sample_id,
        "page": geometry,
        "theme": theme,
        "palette": palette,
        "font_css": font_css,
        "sheet_css": background_css(palette.surface_css),
        "side_stripe": theme.header_bar == BarStyle.SIDE_STRIPE,
        "stripe_mm": stripe_mm,
        "subject_underline": subject_underline,
        "header_html": header_to_html(header, bar_text=palette.bar_text),
        "footer_html": footer_to_html(footer),
        "flow_blocks": flow_blocks,
    }

    markdown = _compose_markdown(
        header=header,
        footer=footer,
        metadata=metadata,
        recipient=recipient,
        sender=sender,
        subject=subject,
        body_markdown=body.markdown,
        attachment_title=attachment_title,
        attachment_lines=source.attachment_lines,
        signatory_title=source.signatory_title,
        signature=signature,
        has_stamp=stamp is not None,
    )

    return ComposedLetter(
        sample_id=sample_id,
        source=source,
        seed=master_seed,
        page_spec=page_spec,
        geometry=geometry,
        theme=theme,
        texture=texture,
        header=header,
        footer=footer,
        metadata=metadata,
        table=table,
        chart=chart,
        signature=signature,
        stamp=stamp,
        context=context,
        markdown=markdown,
        content_map=content_map,
    )
