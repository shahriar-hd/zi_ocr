"""Chart elements, rendered as inline SVG we generate ourselves.

The letter corpus encodes charts as a bare Persian chart-type line - such as
``نمودار میله‌ای`` (bar) or ``نمودار خطی`` (line) - followed by a two-column
``| برچسب | مقدار |`` table. Of the 71 letters flagged ``chart_table ==
"chart"``, about 43 carry such a table and are charted from real numbers; the
rest only mention a chart in prose, so a series is synthesised from the seed.

Inline SVG is used rather than a raster chart library for three reasons:
Chromium shapes the Persian axis labels natively, the output stays sharp at
any DPI, and each label is a real DOM node, so it can carry its own bounding
box.
"""

from __future__ import annotations

import html
import math
import random
import re
from dataclasses import dataclass
from typing import Final, Sequence

from ....config import CHART_KINDS, ChartConfig, ChartVocabulary
from ...common.io_utils import to_persian_digits
from ..markdown_blocks import Block

__all__ = [
    "ChartKind",
    "ChartSpec",
    "build_chart",
    "chart_series_block",
    "synthesize_chart",
    "chart_to_svg",
    "chart_to_markdown",
]

#: Fallback for a caller that does not hand over the loaded config - the test
#: suite, mostly. Read off an instance rather than restated here, so a literal
#: cannot drift from the dataclass default.
_DEFAULT_CHART: Final[ChartConfig] = ChartConfig()

#: Persian chart-type labels as they appear in the letter bodies, mapped to
#: the kind we render.
_KIND_BY_LABEL: Final[dict[str, str]] = {
    "نمودار میله‌ای": "bar",
    "نمودار ستونی": "bar",
    "نمودار خطی": "line",
    "نمودار دایره‌ای": "pie",
    "نمودار کیکی": "pie",
}

#: Header cells that mark a two-column label/value table as chart data.
_LABEL_HEADERS: Final[frozenset[str]] = frozenset(
    {"برچسب", "عنوان", "شرح", "مورد"}
)
_VALUE_HEADERS: Final[frozenset[str]] = frozenset(
    {"مقدار", "عدد", "میزان", "درصد"}
)

#: Rows that summarise rather than contribute a data point. Charting a total
#: alongside its own components would dwarf every real bar.
_TOTAL_MARKERS: Final[tuple[str, ...]] = ("مجموع", "جمع کل", "جمع", "کل")

_DIGIT_MAP: Final[dict[int, int]] = {
    ord(persian): ord(ascii_digit)
    for persian, ascii_digit in zip("۰۱۲۳۴۵۶۷۸۹", "0123456789")
}

_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"-?\d+(?:\.\d+)?")

#: The synthesised series - each a coherent set of categories with the unit it is
#: measured in - and the per-kind captions live in ``configs/vocabulary_fa.yaml``
#: and arrive as ``config.vocabulary.elements.chart``. The unit's wording is what
#: picks the value range below, so the two belong in one file.
_DEFAULT_VOCABULARY: Final[ChartVocabulary] = ChartVocabulary()

#: Chart canvas geometry in CSS pixels. Chosen to sit comfortably inside the
#: narrowest supported page (A5) without horizontal overflow. Intentionally not
#: a knob: the pads are meaningful only against the width and height they sit
#: inside, and a wider canvas would overflow A5 silently rather than visibly.
_WIDTH: Final[int] = 460
_HEIGHT: Final[int] = 250
_PAD_TOP: Final[int] = 16
_PAD_BOTTOM: Final[int] = 46
_PAD_SIDE: Final[int] = 54


class ChartKind:
    """Supported chart kinds."""

    BAR: Final[str] = "bar"
    LINE: Final[str] = "line"
    PIE: Final[str] = "pie"
    #: Read from the config's vocabulary rather than written out again, so the
    #: pool the loader validates against and the pool the renderer can draw are
    #: the same tuple by construction.
    ALL: Final[tuple[str, ...]] = CHART_KINDS


def _to_number(text: str) -> float | None:
    """Parse a Persian or ASCII numeric cell into a float.

    Args:
        text: Cell text, possibly carrying separators, ``٪`` or a unit.

    Returns:
        The parsed value, or ``None`` when the cell holds no number.
    """
    normalised = text.translate(_DIGIT_MAP).replace(",", "").replace("٫", ".")
    match = _NUMBER_RE.search(normalised)
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _is_total_row(label: str) -> bool:
    """Return whether a row label marks a summary total."""
    return any(marker in label for marker in _TOTAL_MARKERS)


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """A chart ready to render, plus structured content for the annotation.

    Attributes:
        kind: One of :class:`ChartKind`'s values.
        labels: Category labels, in series order.
        values: Numeric values, parallel to ``labels``.
        unit: Axis/value unit as shown to the reader.
        caption: Caption rendered above the chart.
        synthesized: ``True`` when the series was invented rather than parsed
            from the letter body.
    """

    kind: str
    labels: tuple[str, ...]
    values: tuple[float, ...]
    unit: str = ""
    caption: str = ""
    synthesized: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ChartKind.ALL:
            raise ValueError(f"unknown chart kind {self.kind!r}")
        if not self.labels:
            raise ValueError("chart must have at least one data point")
        if len(self.labels) != len(self.values):
            raise ValueError(
                f"chart has {len(self.labels)} labels but {len(self.values)} values"
            )

    @property
    def text(self) -> str:
        """Flat reading text of the chart, for the transcription."""
        lines: list[str] = []
        if self.caption:
            lines.append(self.caption)
        lines.extend(
            f"{label}: {to_persian_digits(_format_value(value))}"
            for label, value in zip(self.labels, self.values)
        )
        return "\n".join(lines)

    def as_content(self) -> dict[str, object]:
        """Return the annotation's structured payload for this chart."""
        return {
            "kind": self.kind,
            "caption": self.caption,
            "unit": self.unit,
            "labels": list(self.labels),
            "values": list(self.values),
            "synthesized": self.synthesized,
        }


def _format_value(value: float) -> str:
    """Format a value with thousands separators, dropping a trailing ``.0``."""
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.1f}"


def _persian_value(value: float) -> str:
    """Format a value for display inside the SVG."""
    return to_persian_digits(_format_value(value))


def _detect_kind(
    blocks: Sequence[Block],
    rng: random.Random,
    config: ChartConfig = _DEFAULT_CHART,
) -> str:
    """Infer the chart kind from any Persian chart-type line in the body.

    Args:
        blocks: All parsed body blocks.
        rng: The sample's seeded RNG, used when no label is present.
        config: The ``elements.chart`` section, supplying the fallback pool.

    Returns:
        A chart kind.
    """
    for block in blocks:
        if block.is_table:
            continue
        stripped = block.text.strip()
        for label, kind in _KIND_BY_LABEL.items():
            if label in stripped:
                return kind
    return rng.choice(config.fallback_kinds)


def _series_from_block(
    block: Block,
) -> tuple[tuple[str, ...], tuple[float, ...]] | None:
    """Extract a label/value series from a two-column table block.

    Args:
        block: A parsed table block.

    Returns:
        Parallel label and value tuples, or ``None`` when the table is not a
        chart series.
    """
    if not block.is_table or len(block.rows) < 3:
        return None
    header = block.rows[0]
    if len(header) != 2:
        return None
    label_header, value_header = (cell.strip() for cell in header)
    header_matches = label_header in _LABEL_HEADERS or value_header in _VALUE_HEADERS
    if not header_matches:
        return None
    labels: list[str] = []
    values: list[float] = []
    for row in block.rows[1:]:
        label, raw_value = (cell.strip() for cell in row)
        if not label or _is_total_row(label):
            continue
        value = _to_number(raw_value)
        if value is None:
            continue
        labels.append(label)
        values.append(value)
    if len(labels) < 2:
        return None
    return tuple(labels), tuple(values)


def synthesize_chart(
    kind: str,
    rng: random.Random,
    config: ChartConfig = _DEFAULT_CHART,
    vocabulary: ChartVocabulary = _DEFAULT_VOCABULARY,
) -> ChartSpec:
    """Invent a plausible Persian data series.

    Args:
        kind: The chart kind to build.
        rng: The sample's seeded RNG.
        config: The ``elements.chart`` section, supplying the series length and
            the value range each unit draws from.
        vocabulary: The ``elements.chart`` pools from ``vocabulary_fa.yaml``:
            the series categories and the captions.

    Returns:
        A synthesised chart spec.
    """
    unit, pool = rng.choice(vocabulary.series_pools)
    # The vocabulary caps the length: a range wider than the shortest pool
    # simply plots the whole pool rather than repeating a category.
    count = min(len(pool), config.series_points.draw(rng))
    labels = tuple(pool[:count])
    if "درصد" in unit:
        values = tuple(float(config.percent_values.draw(rng)) for _ in labels)
    elif "تعداد" in unit:
        values = tuple(float(config.count_values.draw(rng)) for _ in labels)
    else:
        values = tuple(float(config.amount_values.draw(rng)) for _ in labels)
    return ChartSpec(
        kind=kind,
        labels=labels,
        values=values,
        unit=unit,
        caption=vocabulary.caption_for(kind),
        synthesized=True,
    )


def chart_series_block(body_blocks: Sequence[Block]) -> Block | None:
    """Return the body table a chart would consume as its data, if any.

    Every one of the 40 chart-bearing letters that ships real data stores it
    as a ``| برچسب | مقدار |`` table in the body. That table *is* the chart,
    so the page composer must not also render it as a standalone table -
    doing so would put the same numbers on the page twice, once as bars and
    once as a grid. Both callers ask this function, so they cannot disagree
    about which block was consumed.

    Args:
        body_blocks: All parsed body blocks.

    Returns:
        The block :func:`build_chart` would read its series from, or ``None``
        when the body carries no chartable series.
    """
    for block in body_blocks:
        if _series_from_block(block) is not None:
            return block
    return None


def build_chart(
    body_blocks: Sequence[Block],
    rng: random.Random,
    config: ChartConfig = _DEFAULT_CHART,
    vocabulary: ChartVocabulary = _DEFAULT_VOCABULARY,
) -> ChartSpec:
    """Return the chart to render, preferring the body's own series.

    Args:
        body_blocks: All parsed body blocks.
        rng: The sample's seeded RNG.
        config: The ``elements.chart`` section. The fallback pool is consulted
            whichever branch wins; the value ranges reach only synthesis.
        vocabulary: The ``elements.chart`` pools. The caption is read whichever
            branch wins; the series categories reach only synthesis.

    Returns:
        A spec parsed from a label/value table when one exists, else a
        synthesised series.
    """
    kind = _detect_kind(body_blocks, rng, config)
    block = chart_series_block(body_blocks)
    if block is not None:
        series = _series_from_block(block)
        if series is not None:
            labels, values = series
            unit = block.rows[0][1].strip() if len(block.rows[0]) == 2 else ""
            return ChartSpec(
                kind=kind,
                labels=labels,
                values=values,
                unit=unit if unit not in _VALUE_HEADERS else "",
                caption=vocabulary.caption_for(kind),
                synthesized=False,
            )
    return synthesize_chart(kind, rng, config, vocabulary)


def chart_to_markdown(spec: ChartSpec) -> str:
    """Render a chart as a Markdown label/value table.

    Markdown cannot carry a chart, so the ``.md`` artifact reproduces the
    underlying series - which is what a text-only consumer needs anyway.

    Args:
        spec: The chart to render.

    Returns:
        A Markdown fragment.
    """
    lines: list[str] = []
    if spec.caption:
        lines.extend((f"**{spec.caption}**", ""))
    value_header = spec.unit or "مقدار"
    lines.append(f"| برچسب | {value_header} |")
    lines.append("|---|---|")
    lines.extend(
        f"| {label} | {_persian_value(value)} |"
        for label, value in zip(spec.labels, spec.values)
    )
    return "\n".join(lines)


def _sanitize(value: str) -> str:
    """Escape a string for use inside an SVG attribute or text node."""
    return html.escape(value, quote=True)


def chart_to_svg(spec: ChartSpec, element_id: str, ink: str) -> str:
    """Render a chart spec as an annotated inline SVG fragment.

    Args:
        spec: The chart to render.
        element_id: Stable id prefix for the annotation.
        ink: CSS colour for the marks, usually the document's accent.

    Returns:
        An SVG fragment. The root and every text label carry
        ``data-ann-*`` attributes so the extractor can box them.
    """
    safe_id = html.escape(element_id, quote=True)
    parts: list[str] = [
        f'<figure class="element element-chart" data-ann-type="chart" '
        f'data-ann-id="{safe_id}">'
    ]
    if spec.caption:
        parts.append(
            f'<figcaption class="chart-caption">'
            f"{_sanitize(spec.caption)}</figcaption>"
        )
    if spec.kind == ChartKind.PIE:
        parts.append(_pie_svg(spec, safe_id, ink))
    elif spec.kind == ChartKind.LINE:
        parts.append(_line_svg(spec, safe_id, ink))
    else:
        parts.append(_bar_svg(spec, safe_id, ink))
    parts.append("</figure>")
    return "".join(parts)


def _label_element(anchor: str, text: str, x: float, y: float) -> str:
    """Emit an annotated SVG text label."""
    return (
        f'<text data-ann-type="paragraph" data-ann-id="{anchor}" '
        f'x="{x:.1f}" y="{y:.1f}" text-anchor="middle">{_sanitize(text)}</text>'
    )


def _bar_svg(spec: ChartSpec, element_id: str, ink: str) -> str:
    """Render a vertical bar chart."""
    count = len(spec.labels)
    maximum = max(spec.values) or 1.0
    plot_width = _WIDTH - 2 * _PAD_SIDE
    plot_height = _HEIGHT - _PAD_TOP - _PAD_BOTTOM
    slot = plot_width / count
    bar_width = min(42.0, slot * 0.55)
    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {_WIDTH} {_HEIGHT}" '
        f'role="img" aria-label="{_sanitize(spec.caption)}">'
    ]
    for index, (label, value) in enumerate(zip(spec.labels, spec.values)):
        bar_height = max(4.0, value / maximum * plot_height)
        x = _PAD_SIDE + slot * index + (slot - bar_width) / 2
        y = _PAD_TOP + plot_height - bar_height
        value_id = f"{element_id}.bar{index}"
        label_id = f"{element_id}.label{index}"
        parts.append(
            f'<rect data-ann-type="chart" data-ann-id="{value_id}" '
            f'x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{ink}" rx="2"/>'
        )
        parts.append(
            f'<text data-ann-type="chart" data-ann-id="{value_id}.value" '
            f'x="{x + bar_width / 2:.1f}" y="{y - 4:.1f}" '
            f'text-anchor="middle" font-size="11" fill="{ink}">'
            f"{_persian_value(value)}</text>"
        )
        parts.append(
            _label_element(
                label_id,
                label,
                x + bar_width / 2,
                _PAD_TOP + plot_height + 18,
            )
        )
    parts.append(
        f'<line x1="{_PAD_SIDE}" y1="{_PAD_TOP + plot_height:.1f}" '
        f'x2="{_WIDTH - _PAD_SIDE:.1f}" y2="{_PAD_TOP + plot_height:.1f}" '
        f'stroke="#333" stroke-width="1"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _line_svg(spec: ChartSpec, element_id: str, ink: str) -> str:
    """Render a line chart."""
    count = len(spec.labels)
    maximum = max(spec.values) or 1.0
    minimum = min(spec.values)
    span = (maximum - minimum) or 1.0
    plot_width = _WIDTH - 2 * _PAD_SIDE
    plot_height = _HEIGHT - _PAD_TOP - _PAD_BOTTOM
    points: list[tuple[float, float]] = []
    for index, value in enumerate(spec.values):
        x = _PAD_SIDE + plot_width * index / max(count - 1, 1)
        y = _PAD_TOP + plot_height - (value - minimum) / span * plot_height
        points.append((x, y))
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {_WIDTH} {_HEIGHT}" '
        f'role="img" aria-label="{_sanitize(spec.caption)}">',
        f'<polyline data-ann-type="chart" data-ann-id="{element_id}.line" '
        f'points="{path}" fill="none" stroke="{ink}" stroke-width="2.5" '
        'stroke-linejoin="round" stroke-linecap="round"/>',
    ]
    for index, ((x, y), label, value) in enumerate(
        zip(points, spec.labels, spec.values)
    ):
        value_id = f"{element_id}.point{index}"
        label_id = f"{element_id}.label{index}"
        parts.append(
            f'<circle data-ann-type="chart" data-ann-id="{value_id}" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{ink}"/>'
        )
        parts.append(
            f'<text data-ann-type="chart" data-ann-id="{value_id}.value" '
            f'x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
            f'font-size="11" fill="{ink}">{_persian_value(value)}</text>'
        )
        parts.append(_label_element(label_id, label, x, _HEIGHT - 18))
    parts.append("</svg>")
    return "".join(parts)


def _pie_svg(spec: ChartSpec, element_id: str, ink: str) -> str:
    """Render a pie chart with a Persian legend."""
    total = sum(spec.values) or 1.0
    center_x, center_y, radius = 150.0, 130.0, 100.0
    legend_x, legend_y, row_gap = 300.0, 44.0, 22.0
    parts: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {_WIDTH} {_HEIGHT}" '
        f'role="img" aria-label="{_sanitize(spec.caption)}">'
    ]
    start = -math.pi / 2
    for index, (label, value) in enumerate(zip(spec.labels, spec.values)):
        sweep = value / total * 2 * math.pi
        end = start + sweep
        x1 = center_x + radius * math.cos(start)
        y1 = center_y + radius * math.sin(start)
        x2 = center_x + radius * math.cos(end)
        y2 = center_y + radius * math.sin(end)
        large = 1 if sweep > math.pi else 0
        segment_id = f"{element_id}.slice{index}"
        parts.append(
            f'<path data-ann-type="chart" data-ann-id="{segment_id}" '
            f'd="M {center_x:.1f} {center_y:.1f} L {x1:.1f} {y1:.1f} '
            f'A {radius:.1f} {radius:.1f} 0 {large} 1 {x2:.1f} {y2:.1f} Z" '
            f'fill="{ink}" opacity="{0.45 + 0.45 * (index % 2)}"/>'
        )
        legend_id = f"{element_id}.legend{index}"
        parts.append(
            f'<circle data-ann-type="chart" data-ann-id="{legend_id}.swatch" '
            f'cx="{legend_x - 10:.1f}" cy="{legend_y:.1f}" r="5" '
            f'fill="{ink}" opacity="{0.45 + 0.45 * (index % 2)}"/>'
        )
        parts.append(
            _label_element(
                legend_id,
                f"{label} — {_persian_value(value)}",
                legend_x,
                legend_y,
            )
        )
        legend_y += row_gap
        start = end
    parts.append("</svg>")
    return "".join(parts)
