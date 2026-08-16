"""Unit tests for the embedded-element generators: tables and charts.

The element stage has two jobs and both are tested here: parse real Markdown
data when the drafting model happened to produce it, and invent coherent
Persian data when it did not. Every generated fragment must also carry the
``data-ann-*`` attributes the extractor depends on - a table that renders
beautifully but boxes nothing is useless as training data.
"""

from __future__ import annotations

import random
import re
import xml.etree.ElementTree as ElementTree

import pytest

from fa_ocr_lora.dataset.common.io_utils import sample_rng
from fa_ocr_lora.dataset.synth.elements.charts import (
    ChartKind,
    ChartSpec,
    build_chart,
    chart_series_block,
    chart_to_markdown,
    chart_to_svg,
    synthesize_chart,
)
from fa_ocr_lora.dataset.synth.elements.tables import (
    TableSpec,
    build_table,
    synthesize_table,
    table_to_html,
    table_to_markdown,
)
from fa_ocr_lora.dataset.synth.markdown_blocks import BlockKind, parse_body

#: A body carrying a real pipe table, of the shape the drafting model emits.
_TABLE_BODY = """
با سلام و احترام،

به پیوست جدول هزینه‌ها تقدیم می‌گردد.

| ردیف | شرح خدمات | مبلغ (ریال) |
| --- | --- | --- |
| ۱ | خدمات پشتیبانی | ۱۲۰٬۰۰۰٬۰۰۰ |
| ۲ | خدمات آموزش | ۸۵٬۰۰۰٬۰۰۰ |

خواهشمند است دستور فرمایید.
"""

#: A body carrying a chart's data series as a two-column table.
_CHART_BODY = """
با سلام،

روند فروش سه ماه گذشته به شرح زیر است.

| برچسب | مقدار |
| --- | --- |
| فروردین | ۱۲۰ |
| اردیبهشت | ۱۹۵ |
| خرداد | ۲۴۰ |
"""


def _rng(purpose: str = "table", index: int = 0) -> random.Random:
    """Return a seeded RNG, mirroring how the pipeline derives one."""
    return sample_rng(1337, purpose, index)


class TestTableParsing:
    """A pipe table present in the body must be used verbatim."""

    def test_body_table_is_parsed(self) -> None:
        blocks = parse_body(_TABLE_BODY)
        tables = [block for block in blocks if block.is_table]
        assert len(tables) == 1
        assert tables[0].kind is BlockKind.TABLE
        assert tables[0].rows[0] == ("ردیف", "شرح خدمات", "مبلغ (ریال)")
        assert len(tables[0].rows) == 3

    def test_build_table_prefers_real_data(self) -> None:
        tables = [block for block in parse_body(_TABLE_BODY) if block.is_table]
        spec = build_table(tables, "هزینه‌ها", _rng())
        assert spec.synthesized is False
        assert spec.header == ("ردیف", "شرح خدمات", "مبلغ (ریال)")
        assert spec.row_count == 2
        assert spec.rows[1] == ("۲", "خدمات آموزش", "۸۵٬۰۰۰٬۰۰۰")

    def test_build_table_synthesizes_without_data(self) -> None:
        spec = build_table((), "تمدید قرارداد پشتیبانی", _rng())
        assert spec.synthesized is True
        assert 3 <= spec.column_count <= 4
        assert 4 <= spec.row_count <= 8

    def test_rows_are_rectangular(self) -> None:
        spec = synthesize_table("گزارش عملکرد", _rng())
        assert all(len(row) == spec.column_count for row in spec.rows)

    def test_ragged_rows_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="cells"):
            TableSpec(header=("الف", "ب"), rows=(("۱",),))

    def test_empty_header_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one column"):
            TableSpec(header=(), rows=())


class TestTableDeterminism:
    """The same seed and subject must invent the same table."""

    def test_same_seed_same_table(self) -> None:
        first = synthesize_table("گزارش مالی", _rng())
        second = synthesize_table("گزارش مالی", _rng())
        assert first == second

    def test_different_index_differs(self) -> None:
        first = synthesize_table("گزارش مالی", _rng(index=0))
        second = synthesize_table("گزارش مالی", _rng(index=7))
        assert first != second


class TestTableHtml:
    """Every cell must be independently boxable."""

    def test_cells_carry_annotation_attributes(self) -> None:
        spec = TableSpec(
            header=("ردیف", "عنوان"),
            rows=(("۱", "الف"), ("۲", "ب")),
            caption="جدول ۱",
        )
        html = table_to_html(spec, "body.table0")
        assert 'data-ann-type="table"' in html
        assert 'data-ann-id="body.table0"' in html
        assert html.count('data-ann-type="table_cell"') == 6
        assert 'data-ann-id="body.table0.h0"' in html
        # Ids number body rows from zero; ``data-ann-row`` counts the header as
        # row 0, so the second body row is ``r1c1`` at row 2.
        assert 'data-ann-id="body.table0.r1c1"' in html
        assert 'data-ann-row="0"' in html
        assert 'data-ann-row="2"' in html
        assert 'data-ann-col="1"' in html

    def test_cell_ids_are_unique(self) -> None:
        spec = synthesize_table("گزارش", _rng())
        html = table_to_html(spec, "body.table0")
        ids = re.findall(r'data-ann-id="([^"]+)"', html)
        assert len(ids) == len(set(ids))

    def test_markdown_round_trips_the_grid(self) -> None:
        spec = TableSpec(header=("الف", "ب"), rows=(("۱", "۲"),))
        markdown = table_to_markdown(spec)
        reparsed = [block for block in parse_body(markdown) if block.is_table]
        assert reparsed[0].rows == (("الف", "ب"), ("۱", "۲"))

    def test_text_covers_every_cell(self) -> None:
        spec = TableSpec(header=("الف", "ب"), rows=(("۱", "۲"),))
        assert spec.text == "الف | ب\n۱ | ۲"


class TestChartParsing:
    """A chart's series comes from the body when the body has one."""

    def test_series_block_is_detected(self) -> None:
        blocks = parse_body(_CHART_BODY)
        consumed = chart_series_block(blocks)
        assert consumed is not None
        assert consumed.rows[0] == ("برچسب", "مقدار")

    def test_series_block_absent_without_data(self) -> None:
        assert chart_series_block(parse_body("سلام. متن ساده.")) is None

    def test_build_chart_uses_body_series(self) -> None:
        spec = build_chart(parse_body(_CHART_BODY), _rng("chart"))
        assert spec.synthesized is False
        assert spec.labels == ("فروردین", "اردیبهشت", "خرداد")
        assert spec.values == (120.0, 195.0, 240.0)

    def test_build_chart_synthesizes_without_data(self) -> None:
        spec = build_chart(parse_body("سلام. متن ساده."), _rng("chart"))
        assert spec.synthesized is True
        assert spec.kind in ChartKind.ALL
        assert len(spec.labels) == len(spec.values) >= 1

    def test_mismatched_series_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            ChartSpec(kind="bar", labels=("الف", "ب"), values=(1.0,))

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown chart kind"):
            ChartSpec(kind="scatter", labels=("الف",), values=(1.0,))


class TestChartDeterminism:
    """The same seed must invent the same series."""

    @pytest.mark.parametrize("kind", sorted(ChartKind.ALL))
    def test_same_seed_same_chart(self, kind: str) -> None:
        first = synthesize_chart(kind, _rng("chart"))
        second = synthesize_chart(kind, _rng("chart"))
        assert first == second


class TestChartSvg:
    """The SVG must be well-formed and fully annotated."""

    @pytest.mark.parametrize("kind", sorted(ChartKind.ALL))
    def test_svg_is_well_formed(self, kind: str) -> None:
        spec = synthesize_chart(kind, _rng("chart"))
        fragment = chart_to_svg(spec, "body.chart0", "#1F3A5F")
        # The fragment is a <figure> wrapping the <svg>; parse the whole thing
        # so a malformed attribute anywhere fails the test.
        ElementTree.fromstring(fragment)

    @pytest.mark.parametrize("kind", sorted(ChartKind.ALL))
    def test_ids_are_unique_and_prefixed(self, kind: str) -> None:
        spec = synthesize_chart(kind, _rng("chart"))
        fragment = chart_to_svg(spec, "body.chart0", "#1F3A5F")
        ids = re.findall(r'data-ann-id="([^"]+)"', fragment)
        assert ids, "chart produced no annotated nodes"
        assert len(ids) == len(set(ids))
        assert all(item.startswith("body.chart0") for item in ids)

    def test_figure_is_typed_as_chart(self) -> None:
        spec = synthesize_chart("bar", _rng("chart"))
        fragment = chart_to_svg(spec, "body.chart0", "#1F3A5F")
        assert 'data-ann-type="chart"' in fragment
        assert 'data-ann-id="body.chart0"' in fragment

    def test_every_label_appears(self) -> None:
        spec = ChartSpec(
            kind="bar",
            labels=("فروردین", "اردیبهشت"),
            values=(10.0, 20.0),
            unit="میلیون ریال",
            caption="نمودار فروش",
        )
        fragment = chart_to_svg(spec, "body.chart0", "#1F3A5F")
        for label in spec.labels:
            assert label in fragment
        assert spec.caption in fragment

    def test_markdown_carries_the_series(self) -> None:
        spec = ChartSpec(kind="line", labels=("الف",), values=(5.0,), unit="تن")
        markdown = chart_to_markdown(spec)
        assert "الف" in markdown
        assert "|" in markdown

    def test_content_payload_is_json_safe(self) -> None:
        import json

        spec = synthesize_chart("pie", _rng("chart"))
        payload = spec.as_content()
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
