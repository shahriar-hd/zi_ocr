"""Unit tests for the corpus readers.

These pin the parsing hazards that were found in the real files, not invented
ones. Two of them corrupt data silently rather than raising, which is why they get
tests of their own:

* **The Persian Pixel CSV carries a UTF-8 BOM and embedded newlines in ``text``.**
  A line-by-line read of that file produces truncated targets and phantom rows, and
  nothing about the result looks wrong - the manifest just quietly trains on
  fragments. The fixture below reproduces both.
* **Its ``image_path`` column is stale in this checkout**, holding HuggingFace
  export names while the files on disk are named after ``image_filename``. Trusting
  one column loses the entire corpus, so the fixture keeps them disagreeing.

Every rejection is asserted to be *counted*, not merely absent from the output.
CLAUDE.md 5.1 asks for dropped samples to be logged and counted, and for good
reason: a reader that skips silently is indistinguishable from a corpus that is
smaller than you think it is.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from fa_ocr_lora.training.config import DataConfig
from fa_ocr_lora.training.sources import (
    LETTER_CROP,
    LETTER_PAGE,
    PIXEL_PAGE,
    SkipLog,
    SourceRecord,
    iter_letter_crops,
    iter_letter_pages,
    iter_pixel_rows,
)

_PERSIAN = "متن نمونه برای آزمون اداری"
_PAGE_W, _PAGE_H = 1240, 1754


def _write_image(root, name, size):
    """Create a blank image so the readers' existence checks pass."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def _letters_fixture(tmp_path):
    """A letters JSONL with one usable page, one missing image, one blank target.

    Page dimensions are carried in the record, matching the real corpus: the reader
    must not have to open an image to budget sequence length.
    """
    root = tmp_path / "dataset"
    root.mkdir()
    _write_image(root, "images/letter_0.png", (_PAGE_W, _PAGE_H))
    _write_image(root, "images/letter_1.png", (_PAGE_W, _PAGE_H))
    records = [
        {
            "sample_id": "synth_000001",
            "page_index": 0,
            "image": "images/letter_0.png",
            "text": _PERSIAN,
            "width": _PAGE_W,
            "height": _PAGE_H,
            "label": [
                # Has text, is not an excluded type, survives the floors: the one
                # region here that should become a crop.
                {
                    "type": "metadata_field",
                    "text": "شماره ثبت: ۱۲۳۴۵",
                    "bbox": [900, 120, 1150, 190],
                    "reading_order": 1,
                },
                # Excluded by type. Also textless, but the type list fires first.
                {
                    "type": "header_bar",
                    "text": "",
                    "bbox": [0, 0, 1240, 100],
                    "reading_order": 0,
                },
                # A croppable type with no text of its own - this is what tests the
                # non-empty-text rule independently of the exclusion list.
                {
                    "type": "table_cell",
                    "text": "",
                    "bbox": [200, 800, 500, 860],
                    "reading_order": 4,
                },
                # Text and a legal type, but 26x28 px after padding = 728 px^2,
                # under the 900 px^2 floor. A sliver.
                {
                    "type": "page_number",
                    "text": "۲۳",
                    "bbox": [600, 1700, 610, 1712],
                    "reading_order": 9,
                },
            ],
        },
        {
            "sample_id": "synth_000002",
            "page_index": 0,
            "image": "images/does_not_exist.png",
            "text": _PERSIAN,
            "width": _PAGE_W,
            "height": _PAGE_H,
            "label": [],
        },
        {
            "sample_id": "synth_000003",
            "page_index": 0,
            "image": "images/letter_1.png",
            "text": "  ",
            "width": _PAGE_W,
            "height": _PAGE_H,
            "label": [],
        },
    ]
    manifest = root / "paddle_ocr_vl.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return manifest


def _pixel_fixture(tmp_path):
    """A pixel CSV reproducing both hazards of the real one.

    Written with ``utf-8-sig`` and a quoted multi-line field, and with
    ``image_path`` deliberately stale so only ``image_filename`` resolves - exactly
    the disagreement present in this checkout.
    """
    root = tmp_path / "persian_pixel"
    root.mkdir()
    _write_image(root, "images/page-00010_row0_image.png", (768, 276))
    _write_image(root, "images/page-00010_row1_image.png", (768, 276))
    path = root / "all_features_metadata.csv"
    path.write_text(
        "image_path,image_filename,text,width,height,sample_type,source_parquet\n"
        "images/image_0005621.png,page-00010_row0_image.png,"
        '"خط اول\nخط دوم\nخط سوم",768,276,page,shard_00.parquet\n'
        "images/image_0005622.png,page-00010_row1_image.png,"
        "یک خط ساده,768,276,page,shard_00.parquet\n"
        "images/image_0005623.png,page-99999_row0_image.png,"
        "متن گمشده,768,276,page,shard_01.parquet\n",
        encoding="utf-8-sig",
    )
    return path


def test_letter_pages_yield_one_record_per_usable_line(tmp_path):
    manifest = _letters_fixture(tmp_path)
    records = list(iter_letter_pages(manifest, DataConfig.parse({}), SkipLog()))
    assert [r.source for r in records] == [LETTER_PAGE]
    assert records[0].text == _PERSIAN
    assert records[0].crop_box is None
    assert (records[0].width, records[0].height) == (_PAGE_W, _PAGE_H)
    assert records[0].record_id == "synth_000001:p0"


def test_letter_pages_group_by_letter_not_by_page(tmp_path):
    """The split unit is the letter, so its pages cannot straddle the boundary."""
    manifest = _letters_fixture(tmp_path)
    records = list(iter_letter_pages(manifest, DataConfig.parse({}), SkipLog()))
    assert records[0].group_id == "synth_000001"


def test_letter_pages_count_every_rejection_rather_than_dropping_it(tmp_path):
    manifest = _letters_fixture(tmp_path)
    skips = SkipLog()
    list(iter_letter_pages(manifest, DataConfig.parse({}), skips))
    # One unresolvable image, one blank target - and each under its own reason, so
    # the run summary distinguishes a broken path from a broken label.
    assert skips.total == 2
    assert skips.counts == {"letters_image_missing": 1, "letters_text_rejected": 1}


def test_letter_pages_reject_a_record_with_no_dimensions(tmp_path):
    """Dimensions are load-bearing: without them the sequence budget is a guess."""
    root = tmp_path / "dataset"
    root.mkdir()
    _write_image(root, "images/letter_0.png", (_PAGE_W, _PAGE_H))
    manifest = root / "paddle_ocr_vl.jsonl"
    manifest.write_text(
        json.dumps(
            {"sample_id": "s1", "image": "images/letter_0.png", "text": _PERSIAN},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    skips = SkipLog()
    assert list(iter_letter_pages(manifest, DataConfig.parse({}), skips)) == []
    assert skips.counts == {"letters_bad_dimensions": 1}


def test_a_malformed_line_is_counted_and_does_not_stop_the_stream(tmp_path):
    manifest = _letters_fixture(tmp_path)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")
    skips = SkipLog()
    records = list(iter_letter_pages(manifest, DataConfig.parse({}), skips))
    assert len(records) == 1
    assert skips.counts["letters_malformed_line"] == 1


def test_letter_crops_take_only_regions_that_carry_text(tmp_path):
    manifest = _letters_fixture(tmp_path)
    skips = SkipLog()
    crops = list(iter_letter_crops(manifest, DataConfig.parse({}), skips))
    assert [c.source for c in crops] == [LETTER_CROP]
    assert crops[0].region_type == "metadata_field"
    assert crops[0].text == "شماره ثبت: ۱۲۳۴۵"
    assert crops[0].record_id == "synth_000001:p0:r1"
    # The textless table_cell is rejected on text, not on type: it proves the
    # non-empty-text rule stands on its own.
    assert skips.counts["crops_text_rejected"] >= 1


def test_excluded_region_types_never_become_crops(tmp_path):
    manifest = _letters_fixture(tmp_path)
    skips = SkipLog()
    crops = list(iter_letter_crops(manifest, DataConfig.parse({}), skips))
    assert all(c.region_type != "header_bar" for c in crops)
    assert skips.counts["crops_excluded_type"] == 1


def test_sliver_regions_are_rejected_by_the_area_floor(tmp_path):
    """26x28 px of padded box is 728 px^2, under the 900 px^2 default floor.

    The side floor does not bind here on purpose - if it did, this test would pass
    for the wrong reason and the area bound would go unexercised.
    """
    manifest = _letters_fixture(tmp_path)
    skips = SkipLog()
    crops = list(iter_letter_crops(manifest, DataConfig.parse({}), skips))
    assert all(c.region_type != "page_number" for c in crops)
    assert skips.counts["crops_too_small"] == 1
    assert "crops_too_thin" not in skips.counts


def test_letter_crops_pad_the_box_and_clip_it_to_the_page(tmp_path):
    manifest = _letters_fixture(tmp_path)
    data = DataConfig.parse({"crop_padding_px": 8})
    crop = next(iter(iter_letter_crops(manifest, data, SkipLog())))
    left, top, right, bottom = crop.crop_box
    assert (left, top, right, bottom) == (892, 112, 1158, 198)
    assert 0 <= left < right <= _PAGE_W
    assert 0 <= top < bottom <= _PAGE_H
    # The recorded size is the crop, not the page. A page-sized value here would
    # misbudget every crop in the manifest and defeat the point of cropping.
    assert (crop.width, crop.height) == (right - left, bottom - top)
    assert crop.pixels == 266 * 86


def test_padding_is_clipped_at_every_page_edge(tmp_path):
    """A box in a corner must not pad off-page.

    The metadata block sits in a top corner - CLAUDE.md 5.2's priority target - so
    it is the box most likely to run out of page, and an unclipped box raises inside
    ``Image.crop``. ``crop_padding_px`` is capped at 64 by the config, which no box
    in the shared fixture can overrun, so this builds its own page with one region
    hard against the top-right corner and one against the bottom-left. Between them
    all four clamps fire.
    """
    root = tmp_path / "dataset"
    root.mkdir()
    _write_image(root, "images/corners.png", (_PAGE_W, _PAGE_H))
    manifest = root / "paddle_ocr_vl.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "synth_000010",
                "page_index": 0,
                "image": "images/corners.png",
                "text": _PERSIAN,
                "width": _PAGE_W,
                "height": _PAGE_H,
                "label": [
                    {
                        "type": "metadata_field",
                        "text": "تاریخ: ۱۴۰۳/۰۵/۱۲",
                        "bbox": [1200, 10, 1235, 40],
                        "reading_order": 1,
                    },
                    {
                        "type": "page_number",
                        "text": "۲۳",
                        "bbox": [5, 1720, 60, 1748],
                        "reading_order": 9,
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    data = DataConfig.parse({"crop_padding_px": 64})
    crops = list(iter_letter_crops(manifest, data, SkipLog()))
    assert len(crops) == 2
    # Top-right: 10 - 64 clamps to 0, and 1235 + 64 clamps to the page width.
    assert crops[0].crop_box == (1136, 0, _PAGE_W, 104)
    # Bottom-left: 5 - 64 clamps to 0, and 1748 + 64 clamps to the page height.
    assert crops[1].crop_box == (0, 1656, 124, _PAGE_H)
    for crop in crops:
        left, top, right, bottom = crop.crop_box
        assert 0 <= left < right <= _PAGE_W
        assert 0 <= top < bottom <= _PAGE_H
        assert (crop.width, crop.height) == (right - left, bottom - top)


def test_letter_crops_share_the_page_group_id(tmp_path):
    """A crop and the page it came from must land in the same split.

    Otherwise the model is validated on a region it was trained on, and the
    reported CER is optimistic by an amount nobody can estimate afterwards.
    """
    manifest = _letters_fixture(tmp_path)
    data = DataConfig.parse({})
    page = next(iter(iter_letter_pages(manifest, data, SkipLog())))
    crop = next(iter(iter_letter_crops(manifest, data, SkipLog())))
    assert crop.group_id == page.group_id == "synth_000001"


def test_crop_weights_favour_the_priority_metadata_fields(tmp_path):
    """CLAUDE.md 5.2 makes the metadata block the first-class target.

    The corpus is unbalanced the other way - ~94k list_item regions against ~31k
    metadata_field - so the weight, not the natural frequency, has to decide.
    """
    data = DataConfig.parse({})
    assert data.crop_weight("metadata_field") > data.crop_weight("list_item")
    assert data.crop_weight("metadata_field") > data.crop_weight("header_slogan")
    # An unseen type still contributes rather than vanishing from a regenerated
    # corpus without anyone noticing.
    assert data.crop_weight("some_future_type") == data.crop_default_weight


def test_pixel_rows_survive_a_bom_and_embedded_newlines(tmp_path):
    """The load-bearing test for this module.

    A line-based read of the fixture returns five rows, three of them fragments of
    one target, and raises nothing at all.
    """
    path = _pixel_fixture(tmp_path)
    rows = list(iter_pixel_rows(path, DataConfig.parse({}), SkipLog()))
    assert [r.source for r in rows] == [PIXEL_PAGE, PIXEL_PAGE]
    assert rows[0].text.split("\n") == ["خط اول", "خط دوم", "خط سوم"]
    assert rows[1].text == "یک خط ساده"
    # If the BOM had been absorbed into the first column name, image_path would be
    # unreachable and the resolver would fall through on every row.
    assert rows[0].record_id == "page-00010_row0_image.png"


def test_pixel_rows_fall_back_to_image_filename_when_image_path_is_stale(tmp_path):
    """The whole corpus depends on this fallback in the current checkout."""
    path = _pixel_fixture(tmp_path)
    rows = list(iter_pixel_rows(path, DataConfig.parse({}), SkipLog()))
    assert rows[0].image_path.name == "page-00010_row0_image.png"
    assert rows[0].image_path.is_file()
    assert (rows[0].width, rows[0].height) == (768, 276)
    assert rows[0].crop_box is None


def test_pixel_rows_group_by_source_shard(tmp_path):
    """Strips cut from one scanned page share a parquet, so they share a split."""
    path = _pixel_fixture(tmp_path)
    rows = list(iter_pixel_rows(path, DataConfig.parse({}), SkipLog()))
    assert rows[0].group_id == rows[1].group_id == "shard_00.parquet"


def test_pixel_rows_count_the_unresolvable_image(tmp_path):
    path = _pixel_fixture(tmp_path)
    skips = SkipLog()
    list(iter_pixel_rows(path, DataConfig.parse({}), skips))
    assert skips.counts == {"pixel_image_missing": 1}


def test_pixel_reader_refuses_a_file_without_the_expected_columns(tmp_path):
    """``labels.csv`` sits beside the real file and is header-only.

    Reading it yields an empty manifest rather than an error unless the columns are
    checked, and an empty manifest looks like a filter that was merely too strict.
    """
    root = tmp_path / "persian_pixel"
    root.mkdir()
    bad = root / "labels.csv"
    bad.write_text("id,label\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        list(iter_pixel_rows(bad, DataConfig.parse({}), SkipLog()))


def test_missing_files_raise_rather_than_yielding_nothing(tmp_path):
    data = DataConfig.parse({})
    with pytest.raises(FileNotFoundError):
        list(iter_letter_pages(tmp_path / "nope.jsonl", data, SkipLog()))
    with pytest.raises(FileNotFoundError):
        list(iter_pixel_rows(tmp_path / "nope.csv", data, SkipLog()))


def test_text_length_bounds_are_enforced_on_both_corpora(tmp_path):
    manifest = _letters_fixture(tmp_path)
    pixel = _pixel_fixture(tmp_path)
    data = DataConfig.parse({"min_text_chars": 500, "max_text_chars": 6000})
    skips = SkipLog()
    assert list(iter_letter_pages(manifest, data, skips)) == []
    assert list(iter_pixel_rows(pixel, data, skips)) == []
    assert skips.counts["letters_text_rejected"] == 2
    assert skips.counts["pixel_text_rejected"] == 2


def test_source_record_reports_the_area_actually_fed(tmp_path):
    record = SourceRecord(
        record_id="r",
        source=LETTER_CROP,
        group_id="g",
        image_path=tmp_path / "x.png",
        crop_box=(0, 0, 100, 50),
        text="متن",
        width=100,
        height=50,
    )
    assert record.pixels == 5000
    assert record.region_type == ""


def test_skip_log_writes_one_line_per_rejection(tmp_path):
    log_path = tmp_path / "logs" / "skipped_records.log"
    skips = SkipLog(log_path)
    skips.skip("letters_image_missing", "synth_000002:p0", "images/does_not_exist.png")
    skips.skip("letters_text_rejected", "synth_000003:p0")
    assert skips.flush() == log_path
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert lines[0] == (
        "letters_image_missing\tsynth_000002:p0 - images/does_not_exist.png"
    )
    assert lines[1] == "letters_text_rejected\tsynth_000003:p0"


def test_skip_log_counts_are_ordered_by_frequency(tmp_path):
    """The summary print leads with the reason worth investigating first."""
    skips = SkipLog()
    skips.skip("rare", "a")
    for index in range(3):
        skips.skip("common", str(index))
    assert list(skips.counts) == ["common", "rare"]
    assert skips.total == 4


def test_skip_log_without_a_path_keeps_counts_only():
    skips = SkipLog()
    skips.skip("letters_image_missing", "x")
    assert skips.flush() is None
    assert skips.counts == {"letters_image_missing": 1}
