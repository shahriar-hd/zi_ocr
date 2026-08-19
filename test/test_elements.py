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

from fa_ocr_lora.config import (
    CHART_KINDS,
    ChartConfig,
    ConfigError,
    IntRange,
    TableConfig,
)
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


class TestIntRangeDraw:
    """The count type must consume the generator exactly as the old code did.

    Every value in this module used to be an inline ``rng.randrange(a, b)``.
    ``IntRange`` exists so those could move into YAML without shifting the
    stream, which only holds if its draw is byte-for-byte the same call.
    """

    def test_draw_matches_randrange_on_the_same_stream(self) -> None:
        assert IntRange(4, 8).draw(_rng()) == _rng().randrange(4, 9)

    def test_a_whole_stream_of_draws_matches(self) -> None:
        # One shared generator, so a difference in bits consumed by any single
        # draw shows up as divergence in every later one.
        direct = _rng()
        through_range = _rng()
        bounds = IntRange(5, 949)
        assert [direct.randrange(5, 950) for _ in range(20)] == [
            bounds.draw(through_range) for _ in range(20)
        ]

    def test_both_ends_are_reachable(self) -> None:
        drawn = {IntRange(1, 2).draw(_rng(index=index)) for index in range(40)}
        assert drawn == {1, 2}

    def test_equal_ends_are_a_constant(self) -> None:
        assert IntRange(7, 7).draw(_rng()) == 7


class TestTableConfigReachesTheMark:
    """The ``elements.table`` section must decide the invented values."""

    def test_row_count_comes_from_the_config(self) -> None:
        spec = synthesize_table("گزارش", _rng(), TableConfig(rows=IntRange(2, 2)))
        assert spec.row_count == 2

    def test_amount_column_respects_its_range(self) -> None:
        config = TableConfig(rows=IntRange(6, 6), amount_millions=IntRange(3, 3))
        for _ in range(12):
            spec = synthesize_table("برآورد هزینه‌ها", _rng(), config)
            amounts = [
                cell
                for header, index in zip(spec.header, range(spec.column_count))
                if "مبلغ" in header or "هزینه" in header
                for cell in (row[index] for row in spec.rows)
            ]
            # 3 millions, grouped and in Persian digits.
            assert all(cell == "۳٬۰۰۰٬۰۰۰" for cell in amounts)

    def test_percent_and_count_columns_respect_their_ranges(self) -> None:
        config = TableConfig(
            rows=IntRange(8, 8), percent=IntRange(42, 42), count=IntRange(9, 9)
        )
        seen = {"percent": 0, "count": 0}
        for index in range(24):
            spec = synthesize_table("گزارش پیشرفت", _rng(index=index), config)
            for column, header in enumerate(spec.header):
                cells = [row[column] for row in spec.rows]
                if "درصد" in header:
                    seen["percent"] += 1
                    assert all(cell == "۴۲٪" for cell in cells)
                elif "تعداد" in header:
                    seen["count"] += 1
                    assert all(cell == "۹" for cell in cells)
        assert seen["percent"] and seen["count"], "neither column was ever drawn"

    def test_date_column_respects_its_ranges(self) -> None:
        config = TableConfig(rows=IntRange(4, 4), day=IntRange(5, 5), year=IntRange(
            1402, 1402
        ))
        matched = 0
        for index in range(24):
            spec = synthesize_table("زمان‌بندی اجرا", _rng(index=index), config)
            for column, header in enumerate(spec.header):
                if "زمان" not in header and "تاریخ" not in header:
                    continue
                matched += 1
                for row in spec.rows:
                    assert row[column].startswith("۵ ")
                    assert row[column].endswith(" ۱۴۰۲")
        assert matched, "no date column was ever drawn"

    def test_a_body_table_ignores_the_config(self) -> None:
        tables = [block for block in parse_body(_TABLE_BODY) if block.is_table]
        spec = build_table(tables, "هزینه‌ها", _rng(), TableConfig(rows=IntRange(1, 1)))
        assert spec.synthesized is False
        assert spec.row_count == 2


class TestChartConfigReachesTheMark:
    """The ``elements.chart`` section must decide the invented series."""

    def test_kinds_are_the_config_vocabulary(self) -> None:
        # Same tuple, not merely equal: the pool the loader validates against
        # and the pool the renderer can draw cannot drift apart.
        assert ChartKind.ALL is CHART_KINDS

    def test_fallback_pool_is_switchable(self) -> None:
        blocks = parse_body("سلام. متن ساده.")
        for kind in ("bar", "line", "pie"):
            config = ChartConfig(fallback_kinds=(kind,))
            spec = build_chart(blocks, _rng("chart"), config)
            assert spec.kind == kind

    def test_an_explicit_kind_in_the_body_beats_the_fallback(self) -> None:
        blocks = parse_body("با سلام.\n\nنمودار دایره‌ای زیر را ملاحظه فرمایید.")
        spec = build_chart(blocks, _rng("chart"), ChartConfig(fallback_kinds=("bar",)))
        assert spec.kind == "pie"

    def test_series_length_comes_from_the_config(self) -> None:
        spec = synthesize_chart(
            "bar", _rng("chart"), ChartConfig(series_points=IntRange(2, 2))
        )
        assert len(spec.labels) == 2

    def test_series_length_is_capped_by_the_vocabulary(self) -> None:
        # The shortest label pool has four entries, so a request for 40 points
        # plots the whole pool rather than repeating a category.
        for index in range(12):
            spec = synthesize_chart(
                "line",
                _rng("chart", index),
                ChartConfig(series_points=IntRange(40, 40)),
            )
            assert 4 <= len(spec.labels) <= 6
            assert len(set(spec.labels)) == len(spec.labels)

    def test_each_unit_draws_from_its_own_range(self) -> None:
        config = ChartConfig(
            percent_values=IntRange(11, 11),
            count_values=IntRange(22, 22),
            amount_values=IntRange(33, 33),
        )
        expected = {"درصد": 11.0, "تعداد": 22.0}
        seen: set[str] = set()
        for index in range(24):
            spec = synthesize_chart("bar", _rng("chart", index), config)
            marker = next(
                (key for key in expected if key in spec.unit), "amount"
            )
            seen.add(marker)
            assert set(spec.values) == {expected.get(marker, 33.0)}
        assert seen == {"درصد", "تعداد", "amount"}, f"units not covered: {seen}"

    def test_a_body_series_ignores_the_value_ranges(self) -> None:
        spec = build_chart(
            parse_body(_CHART_BODY),
            _rng("chart"),
            ChartConfig(count_values=IntRange(1, 1), amount_values=IntRange(1, 1)),
        )
        assert spec.synthesized is False
        assert spec.values == (120.0, 195.0, 240.0)


class TestElementConfigValidation:
    """A setting that would break a render must be rejected at load time."""

    def test_a_table_with_no_body_rows_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="rows.min must be at least 1"):
            TableConfig(rows=IntRange(0, 4))

    def test_an_impossible_day_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match=r"day must stay within \[1, 31\]"):
            TableConfig(day=IntRange(1, 32))

    def test_a_negative_table_value_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="count.min must not be negative"):
            TableConfig(count=IntRange(-1, 10))

    def test_a_descending_range_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="ascending"):
            IntRange(9, 4)

    def test_a_fractional_count_is_rejected_rather_than_truncated(self) -> None:
        with pytest.raises(ConfigError, match="expected a whole number"):
            IntRange.parse({"min": 4.5, "max": 8}, "elements.table.rows", IntRange(4, 8))

    def test_a_boolean_is_not_a_count(self) -> None:
        with pytest.raises(ConfigError, match="expected a whole number"):
            IntRange.parse(True, "elements.table.rows", IntRange(4, 8))

    def test_an_empty_fallback_pool_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="at least one kind"):
            ChartConfig(fallback_kinds=())

    def test_a_kind_the_renderer_cannot_draw_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="unknown kind"):
            ChartConfig(fallback_kinds=("bar", "scatter"))

    def test_a_single_point_series_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="series_points.min must be at least 2"):
            ChartConfig(series_points=IntRange(1, 6))

    def test_a_negative_chart_value_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="count_values.min must not be negative"):
            ChartConfig(count_values=IntRange(-5, 10))

    def test_every_shipped_fallback_kind_is_renderable(self) -> None:
        for kind in ChartConfig().fallback_kinds:
            spec = synthesize_chart(kind, _rng("chart"))
            ElementTree.fromstring(chart_to_svg(spec, "body.chart0", "#1F3A5F"))
