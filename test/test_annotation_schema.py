"""Unit tests for the shared annotation schema and the box extractor.

Two things are checked here, and they are the load-bearing ones for training:

* :mod:`fa_ocr_lora.dataset.common.schema` must round-trip losslessly and
  refuse anything a trainer could not consume - a box off the page, a repeated
  id, a gap in the page indices.
* :mod:`fa_ocr_lora.dataset.synth.annotation` must turn Chromium's flat list of
  CSS-pixel rects into nested, right-to-left-ordered device-pixel elements
  whose text is transcribed exactly once.

No browser is launched: the harvest is a plain list of dictionaries, so the
extractor is tested against boxes whose geometry is known by construction
rather than measured.
"""

from __future__ import annotations

from typing import Any

import pytest

from fa_ocr_lora.dataset.common.schema import (
    SCHEMA_VERSION,
    AnnotationError,
    BBox,
    DocumentAnnotation,
    Element,
    ElementType,
    Page,
)
from fa_ocr_lora.dataset.synth.annotation import build_pages
from fa_ocr_lora.dataset.synth.render.browser import RenderResult, RenderedSheet

_SHEET_WIDTH = 600
_SHEET_HEIGHT = 800


def _box(
    element_id: str,
    element_type: str,
    rect: tuple[float, float, float, float],
    *,
    text: str = "",
    sheet: int = 0,
    content: dict[str, Any] | None = None,
    **attrs: Any,
) -> dict[str, Any]:
    """Build one harvested box, in the shape ``paginate.js`` returns."""
    return {
        "id": element_id,
        "type": element_type,
        "rect": list(rect),
        "text": text,
        "sheet": sheet,
        "attrs": attrs,
        "content": content,
    }


def _result(
    boxes: list[dict[str, Any]],
    *,
    scale: float = 1.0,
    sheet_count: int = 1,
    overflowed: bool = False,
) -> RenderResult:
    """Wrap boxes in a :class:`RenderResult` with plausible blank sheets."""
    sheets = tuple(
        RenderedSheet(
            index=index,
            png=b"",
            width=int(round(_SHEET_WIDTH * scale)),
            height=int(round(_SHEET_HEIGHT * scale)),
        )
        for index in range(sheet_count)
    )
    return RenderResult(
        sheets=sheets,
        pdf=b"",
        boxes=tuple(boxes),
        scale=scale,
        overflowed=overflowed,
    )


class TestBBox:
    """Geometry primitives."""

    def test_degenerate_box_is_rejected(self) -> None:
        with pytest.raises(AnnotationError, match="positive extent"):
            BBox(10, 10, 10, 20)

    def test_negative_origin_is_rejected(self) -> None:
        with pytest.raises(AnnotationError, match="non-negative"):
            BBox(-1, 0, 10, 10)

    def test_from_iterable_rounds_to_pixels(self) -> None:
        assert BBox.from_iterable((1.4, 2.6, 10.7, 20.49)).as_list() == [1, 3, 11, 20]

    def test_from_iterable_needs_four_values(self) -> None:
        with pytest.raises(AnnotationError, match="exactly 4"):
            BBox.from_iterable((1, 2, 3))

    def test_clipping_pulls_a_box_back_onto_the_page(self) -> None:
        clipped = BBox(500, 700, 620, 830).clipped_to(_SHEET_WIDTH, _SHEET_HEIGHT)
        assert clipped.as_list() == [500, 700, 600, 800]
        assert clipped.within(_SHEET_WIDTH, _SHEET_HEIGHT)

    def test_center_y_is_the_midpoint(self) -> None:
        assert BBox(0, 100, 10, 140).center_y == 120.0


class TestPageValidation:
    """A page must refuse what a trainer cannot use."""

    def test_duplicate_ids_are_rejected(self) -> None:
        element = Element(id="a", type=ElementType.PARAGRAPH, bbox=BBox(0, 0, 10, 10))
        with pytest.raises(AnnotationError, match="duplicate element id"):
            Page(
                page_index=0,
                width=100,
                height=100,
                image_path="images/p01.png",
                elements=(element, element),
            )

    def test_escaping_bbox_is_rejected(self) -> None:
        element = Element(
            id="a",
            type=ElementType.PARAGRAPH,
            bbox=BBox(0, 0, 200, 10),
        )
        with pytest.raises(AnnotationError, match="escapes page"):
            Page(
                page_index=0,
                width=100,
                height=100,
                image_path="images/p01.png",
                elements=(element,),
            )

    def test_non_positive_size_is_rejected(self) -> None:
        with pytest.raises(AnnotationError, match="non-positive size"):
            Page(page_index=0, width=0, height=10, image_path="x.png")

    def test_transcription_follows_reading_order(self) -> None:
        page = Page(
            page_index=0,
            width=100,
            height=100,
            image_path="images/p01.png",
            elements=(
                Element(
                    id="second",
                    type=ElementType.PARAGRAPH,
                    bbox=BBox(0, 40, 90, 60),
                    text="دوم",
                    reading_order=1,
                ),
                Element(
                    id="first",
                    type=ElementType.SUBJECT,
                    bbox=BBox(0, 0, 90, 20),
                    text="اول",
                    reading_order=0,
                ),
            ),
        )
        assert page.transcription() == "اول\nدوم"

    def test_non_text_elements_are_not_transcribed(self) -> None:
        page = Page(
            page_index=0,
            width=100,
            height=100,
            image_path="images/p01.png",
            elements=(
                Element(
                    id="logo",
                    type=ElementType.LOGO,
                    bbox=BBox(0, 0, 20, 20),
                    text="ignored",
                ),
            ),
        )
        assert page.transcription() == ""


class TestDocumentAnnotation:
    """Document-level invariants and serialisation."""

    @staticmethod
    def _annotation(**overrides: Any) -> DocumentAnnotation:
        payload: dict[str, Any] = {
            "sample_id": "synth_000000",
            "source_sample_id": "plaintext_000000",
            "seed": 1337,
            "dpi": 300,
            "page_size": "A4",
            "orientation": "portrait",
            "pages": (
                Page(
                    page_index=0,
                    width=2480,
                    height=3508,
                    image_path="images/synth_000000_p01.png",
                    elements=(
                        Element(
                            id="subject",
                            type=ElementType.SUBJECT,
                            bbox=BBox(100, 200, 900, 260),
                            text="موضوع: آزمون",
                        ),
                    ),
                ),
            ),
            "letter_fields": {"subject": "آزمون"},
            "texture": "page_3.webp",
        }
        payload.update(overrides)
        return DocumentAnnotation(**payload)

    def test_pages_are_required(self) -> None:
        with pytest.raises(AnnotationError, match="no pages"):
            self._annotation(pages=())

    def test_page_indices_must_be_contiguous(self) -> None:
        second = Page(
            page_index=2,
            width=10,
            height=10,
            image_path="images/p03.png",
        )
        first = Page(
            page_index=0,
            width=10,
            height=10,
            image_path="images/p01.png",
        )
        with pytest.raises(AnnotationError, match="contiguous"):
            self._annotation(pages=(first, second))

    def test_transcription_is_auto_filled(self) -> None:
        annotation = self._annotation()
        assert annotation.full_transcription == "موضوع: آزمون"

    def test_round_trip_is_lossless(self) -> None:
        original = self._annotation()
        restored = DocumentAnnotation.from_dict(original.as_dict())
        assert restored == original
        assert restored.as_dict() == original.as_dict()

    def test_serialised_shape_matches_the_documented_schema(self) -> None:
        payload = self._annotation().as_dict()
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["texture"] == "page_3.webp"
        assert payload["pages"][0]["elements"][0]["bbox"] == [100, 200, 900, 260]

    def test_missing_key_is_reported(self) -> None:
        payload = self._annotation().as_dict()
        del payload["seed"]
        with pytest.raises(AnnotationError, match="missing required key"):
            DocumentAnnotation.from_dict(payload)

    def test_element_count_spans_pages(self) -> None:
        assert self._annotation().element_count == 1


class TestBoxExtraction:
    """Chromium's flat rect list becomes nested, ordered, scaled elements."""

    def test_css_pixels_are_scaled_to_device_pixels(self) -> None:
        result = _result(
            [_box("body.p0", "paragraph", (10, 20, 110, 50), text="متن")],
            scale=2.0,
        )
        page = build_pages(result, image_names=["images/p01.png"])[0]
        assert page.width == 1200 and page.height == 1600
        assert page.elements[0].bbox.as_list() == [20, 40, 220, 100]

    def test_boxes_are_clipped_to_the_page(self) -> None:
        result = _result(
            [_box("footer", "footer_bar", (0, 780, 640, 820))],
        )
        page = build_pages(result, image_names=["images/p01.png"])[0]
        assert page.elements[0].bbox.within(page.width, page.height)

    def test_zero_area_boxes_are_dropped(self) -> None:
        result = _result(
            [
                _box("rule", "header_bar", (0, 100, 600, 100.2)),
                _box("body.p0", "paragraph", (10, 200, 500, 240), text="متن"),
            ],
        )
        page = build_pages(result, image_names=["images/p01.png"])[0]
        assert [element.id for element in page.elements] == ["body.p0"]

    def test_unknown_types_are_skipped(self) -> None:
        result = _result(
            [
                _box("mystery", "not_a_real_type", (10, 10, 100, 40), text="x"),
                _box("body.p0", "paragraph", (10, 200, 500, 240), text="متن"),
            ],
        )
        page = build_pages(result, image_names=["images/p01.png"])[0]
        assert [element.id for element in page.elements] == ["body.p0"]

    def test_reading_order_is_right_to_left_within_a_line(self) -> None:
        result = _result(
            [
                _box("left", "paragraph", (40, 302, 200, 332), text="چپ"),
                _box("right", "paragraph", (400, 300, 560, 330), text="راست"),
            ],
        )
        page = build_pages(result, image_names=["images/p01.png"])[0]
        assert [element.id for element in page.elements] == ["right", "left"]

    def test_reading_order_is_top_to_bottom_across_lines(self) -> None:
        result = _result(
            [
                _box("lower", "paragraph", (400, 400, 560, 430), text="پایین"),
                _box("upper", "paragraph", (40, 100, 200, 130), text="بالا"),
            ],
        )
        page = build_pages(result, image_names=["images/p01.png"])[0]
        assert [element.id for element in page.elements] == ["upper", "lower"]

    def test_container_precedes_its_children(self) -> None:
        result = _result(
            [
                _box("meta.date", "metadata_field", (390, 150, 585, 180), text="تاریخ"),
                _box("meta", "metadata_block", (380, 100, 590, 190), text="شماره تاریخ"),
                _box("meta.number", "metadata_field", (390, 110, 585, 140), text="شماره"),
            ],
        )
        page = build_pages(result, image_names=["images/p01.png"])[0]
        assert [element.id for element in page.elements] == [
            "meta",
            "meta.number",
            "meta.date",
        ]

    def test_container_text_is_cleared_so_it_is_transcribed_once(self) -> None:
        result = _result(
            [
                _box("t", "table", (40, 100, 560, 220), text="سر ستون\nمقدار"),
                _box(
                    "t.h0",
                    "table_cell",
                    (300, 105, 555, 145),
                    text="سر ستون",
                    row="0",
                    col="0",
                ),
                _box("t.r0c0", "table_cell", (300, 150, 555, 190), text="مقدار"),
            ],
        )
        page = build_pages(result, image_names=["images/p01.png"])[0]
        table = next(element for element in page.elements if element.id == "t")
        assert table.text == ""
        assert page.transcription().count("سر ستون") == 1
        assert page.transcription().count("مقدار") == 1

    def test_leaf_text_survives(self) -> None:
        result = _result([_box("body.p0", "paragraph", (40, 100, 560, 140), text="متن")])
        page = build_pages(result, image_names=["images/p01.png"])[0]
        assert page.elements[0].text == "متن"

    def test_non_text_types_lose_their_text(self) -> None:
        result = _result([_box("header.logo", "logo", (480, 20, 560, 90), text="لوگو")])
        page = build_pages(result, image_names=["images/p01.png"])[0]
        assert page.elements[0].text == ""

    def test_dom_attributes_and_spec_payloads_merge(self) -> None:
        result = _result(
            [
                _box(
                    "t.r0c1",
                    "table_cell",
                    (300, 150, 555, 190),
                    text="۱۲",
                    content={"numeric": True},
                    row="1",
                    col="1",
                ),
            ],
        )
        page = build_pages(
            result,
            image_names=["images/p01.png"],
            content_map={"t.r0c1": {"source": "synthesized"}},
        )[0]
        assert page.elements[0].content == {
            "source": "synthesized",
            "numeric": True,
            "row": "1",
            "col": "1",
        }

    def test_duplicate_ids_keep_the_larger_box(self) -> None:
        result = _result(
            [
                _box("body.p0", "paragraph", (40, 100, 200, 140), text="کوچک"),
                _box("body.p0", "paragraph", (40, 100, 560, 200), text="بزرگ"),
            ],
        )
        page = build_pages(result, image_names=["images/p01.png"])[0]
        assert len(page.elements) == 1
        assert page.elements[0].bbox.as_list() == [40, 100, 560, 200]

    def test_boxes_are_grouped_by_sheet(self) -> None:
        result = _result(
            [
                _box("body.p0", "paragraph", (40, 100, 560, 140), text="یک", sheet=0),
                _box("body.p1", "paragraph", (40, 100, 560, 140), text="دو", sheet=1),
            ],
            sheet_count=2,
        )
        pages = build_pages(result, image_names=["images/p01.png", "images/p02.png"])
        assert [element.id for element in pages[0].elements] == ["body.p0"]
        assert [element.id for element in pages[1].elements] == ["body.p1"]
        assert pages[1].image_path == "images/p02.png"

    def test_image_name_count_must_match_sheet_count(self) -> None:
        result = _result([], sheet_count=2)
        with pytest.raises(AnnotationError, match="expected 2 image names"):
            build_pages(result, image_names=["images/p01.png"])

    def test_reading_order_is_dense_and_zero_based(self) -> None:
        result = _result(
            [
                _box("a", "paragraph", (400, 100, 560, 130), text="الف"),
                _box("b", "paragraph", (40, 100, 200, 130), text="ب"),
                _box("c", "paragraph", (40, 200, 560, 240), text="ج"),
            ],
        )
        page = build_pages(result, image_names=["images/p01.png"])[0]
        orders = [element.reading_order for element in page.elements]
        assert orders == list(range(len(orders)))
