"""End-to-end tests for the synthetic sample pipeline.

The suite is split by cost:

* **Composition tests** run the whole chain up to the HTML string - source
  loading, theming, element synthesis, fragment assembly, Markdown - against
  real rows of ``data/text/generator.csv``. They need no browser and are the
  ones that catch a broken contract between two modules.
* **Render tests** are marked ``slow`` because they launch Chromium and write
  files. They are the only place the PNG/PDF/JSON artifacts are produced, and
  they write into ``tmp_path`` rather than the dataset directory so running
  them never pollutes ``data/synth/``.

Run the cheap set with ``pytest -m 'not slow'``.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping

import pytest

from fa_ocr_lora.config import (
    ConfigError,
    ImageFormatConfig,
    SynthConfig,
    load_config,
)
from fa_ocr_lora.dataset.common.io_utils import read_manifest
from fa_ocr_lora.dataset.common.schema import (
    BBox,
    DocumentAnnotation,
    Element,
    ElementType,
    Page,
)
from fa_ocr_lora.dataset.synth.layout.page_composer import compose_letter
from fa_ocr_lora.dataset.synth.letter_source import LetterSource, load_letter_source
from fa_ocr_lora.dataset.synth.pipeline import (
    MANIFEST_COLUMNS,
    _apply_overrides,
    _select_rows,
    artifact_paths,
    is_complete,
    paddle_farmer_records,
    paddle_records,
    parse_subdir_overrides,
    render_sample,
    sample_id_for,
)
from fa_ocr_lora.dataset.synth.render.browser import sheet_image_names
from fa_ocr_lora.dataset.synth.render.document import render_document

#: Rows chosen because each exercises a different branch of the composer.
_TABLE_ROW = "plaintext_000004"
_CHART_ROW = "plaintext_000005"
_UNSIGNED_ROW = "plaintext_000002"
_LONG_ROW = "plaintext_007666"


@pytest.fixture(scope="module")
def config() -> SynthConfig:
    """Return the shipped configuration, skipping the suite if it is absent."""
    try:
        return load_config()
    except (FileNotFoundError, OSError) as error:  # pragma: no cover - env issue
        pytest.skip(f"synth config unavailable: {error}")


@pytest.fixture(scope="module")
def rows(config: SynthConfig) -> dict[str, Mapping[str, str]]:
    """Return the text-stage manifest keyed by source sample id."""
    if not config.paths.manifest.is_file():
        pytest.skip(f"text manifest not generated yet: {config.paths.manifest}")
    return {row["sample_id"]: row for row in read_manifest(config.paths.manifest)}


def _source(
    rows: Mapping[str, Mapping[str, str]],
    config: SynthConfig,
    sample_id: str,
) -> LetterSource:
    """Load one letter, skipping the test when that row was never generated."""
    row = rows.get(sample_id)
    if row is None:
        pytest.skip(f"{sample_id} not present in the text manifest")
    return load_letter_source(row, config)


class TestSourceLoading:
    """The plaintext record and its assets must resolve as documented."""

    def test_table_row_is_marked_table_bearing(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _TABLE_ROW)
        assert source.wants_table is True
        assert source.wants_chart is False
        assert source.subject

    def test_chart_row_is_marked_chart_bearing(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _CHART_ROW)
        assert source.wants_chart is True

    def test_unsigned_row_has_no_signature(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _UNSIGNED_ROW)
        assert source.has_signature is False

    def test_missing_assets_are_reported_not_raised(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _UNSIGNED_ROW)
        assert isinstance(source.missing_assets, tuple)

    def test_letter_fields_are_complete(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        fields = _source(rows, config, _TABLE_ROW).letter_fields
        assert set(fields) >= {
            "company_name",
            "subject",
            "letter_number",
            "date",
            "registration_number",
        }


class TestComposition:
    """The composer's output must be renderable and fully annotated."""

    def test_context_carries_everything_the_template_needs(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _TABLE_ROW)
        composed = compose_letter(source, config, sample_id="synth_000004")
        assert set(composed.context) >= {
            "sample_id",
            "page",
            "theme",
            "palette",
            "font_css",
            "sheet_css",
            "header_html",
            "footer_html",
            "flow_blocks",
        }
        assert composed.context["flow_blocks"]

    def test_html_renders_and_is_rtl(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _TABLE_ROW)
        composed = compose_letter(source, config, sample_id="synth_000004")
        html = render_document(composed.context)
        assert '<html dir="rtl" lang="fa">' in html
        assert 'id="sheet-template"' in html
        assert 'id="flow"' in html
        # The stylesheet is inlined; a URL reference would never resolve in a
        # ``set_content`` page.
        assert "<style>" in html
        assert "<link" not in html

    def test_annotated_ids_are_unique_in_the_document(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _TABLE_ROW)
        composed = compose_letter(source, config, sample_id="synth_000004")
        html = render_document(composed.context)
        ids = re.findall(r'data-ann-id="([^"]+)"', html)
        assert len(ids) == len(set(ids))

    def test_every_annotated_type_is_known_to_the_schema(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _TABLE_ROW)
        composed = compose_letter(source, config, sample_id="synth_000004")
        html = render_document(composed.context)
        for raw in set(re.findall(r'data-ann-type="([^"]+)"', html)):
            ElementType(raw)

    def test_table_row_composes_a_table(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _TABLE_ROW)
        composed = compose_letter(source, config, sample_id="synth_000004")
        assert composed.table is not None
        assert any(key.startswith("body.table") for key in composed.content_map)

    def test_chart_row_composes_a_chart(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _CHART_ROW)
        composed = compose_letter(source, config, sample_id="synth_000005")
        assert composed.chart is not None
        html = render_document(composed.context)
        assert "<svg" in html

    def test_unsigned_row_has_no_signature_or_stamp_fragment(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _UNSIGNED_ROW)
        composed = compose_letter(source, config, sample_id="synth_000002")
        assert composed.signature is None
        html = render_document(composed.context)
        assert 'data-ann-type="signature_image"' not in html

    def test_markdown_mirrors_the_page(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _TABLE_ROW)
        composed = compose_letter(source, config, sample_id="synth_000004")
        assert composed.markdown.endswith("\n")
        assert source.subject.split()[0] in composed.markdown

    def test_composition_is_deterministic(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _TABLE_ROW)
        first = compose_letter(source, config, sample_id="synth_000004", seed=99)
        second = compose_letter(source, config, sample_id="synth_000004", seed=99)
        assert first.markdown == second.markdown
        assert render_document(first.context) == render_document(second.context)

    def test_a_different_seed_changes_the_page(
        self, rows: dict[str, Mapping[str, str]], config: SynthConfig
    ) -> None:
        source = _source(rows, config, _TABLE_ROW)
        first = compose_letter(source, config, sample_id="synth_000004", seed=1)
        second = compose_letter(source, config, sample_id="synth_000004", seed=2)
        assert render_document(first.context) != render_document(second.context)

    @pytest.mark.parametrize("page_name", ("A4", "A5", "A3", "US Letter"))
    def test_every_page_size_composes(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        page_name: str,
    ) -> None:
        specs = [
            spec
            for spec in config.render.page_sizes
            if spec.name.lower().startswith(page_name.lower())
        ]
        if not specs:
            pytest.skip(f"{page_name} is not in the configured page pool")
        pinned = replace(config, render=replace(config.render, page_sizes=(specs[0],)))
        source = _source(rows, config, _TABLE_ROW)
        composed = compose_letter(source, pinned, sample_id="synth_000004")
        assert composed.page_spec.name == specs[0].name
        html = render_document(composed.context)
        assert f"{specs[0].width_mm}mm" in html


class TestManifestPlumbing:
    """The runner's bookkeeping, tested without rendering anything."""

    def test_sample_ids_are_derived_from_the_row_position(self) -> None:
        assert sample_id_for(0) == "synth_000000"
        assert sample_id_for(42) == "synth_000042"

    def test_artifact_paths_live_under_the_output_dir(
        self, config: SynthConfig, tmp_path: Path
    ) -> None:
        scoped = replace(config, paths=replace(config.paths, output_dir=tmp_path))
        paths = artifact_paths(scoped, "synth_000000")
        assert paths["pdf"] == tmp_path / "pdf" / "synth_000000.pdf"
        assert paths["markdown"] == tmp_path / "markdown" / "synth_000000.md"
        assert paths["annotation"] == tmp_path / "annotations" / "synth_000000.json"

    def test_resume_keys_on_the_annotation(
        self, config: SynthConfig, tmp_path: Path
    ) -> None:
        scoped = replace(config, paths=replace(config.paths, output_dir=tmp_path))
        assert is_complete(scoped, "synth_000000") is False
        target = artifact_paths(scoped, "synth_000000")["annotation"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
        assert is_complete(scoped, "synth_000000") is True

    def test_slicing_preserves_manifest_positions(self) -> None:
        manifest = [{"sample_id": f"plaintext_{index:06d}"} for index in range(10)]
        selected = _select_rows(manifest, start=3, count=2)
        assert [index for index, _ in selected] == [3, 4]

    def test_only_ignores_start_and_count(self) -> None:
        manifest = [{"sample_id": f"plaintext_{index:06d}"} for index in range(10)]
        selected = _select_rows(
            manifest,
            start=5,
            count=1,
            only=("plaintext_000001", "plaintext_000009"),
        )
        assert [index for index, _ in selected] == [1, 9]

    def test_image_names_sort_into_page_order(self) -> None:
        names = sheet_image_names("synth_000000", 3)
        assert names == (
            "synth_000000_p01.png",
            "synth_000000_p02.png",
            "synth_000000_p03.png",
        )
        assert list(names) == sorted(names)


class TestImageFormatPlumbing:
    """The chosen format must reach the filenames, not just the encoder.

    A page written as JPEG under a ``.png`` name is worse than either: the
    annotation points at a file whose extension lies, and every consumer that
    dispatches on the suffix - including HuggingFace's ``imagefolder`` - guesses
    wrong. The suffix therefore comes from the same object as the encoding.
    """

    @pytest.mark.parametrize(
        ("spelling", "suffix"),
        [("png", ".png"), ("jpeg", ".jpg"), ("jpg", ".jpg"), ("webp", ".webp")],
    )
    def test_page_names_carry_the_configured_suffix(
        self, spelling: str, suffix: str
    ) -> None:
        spec = ImageFormatConfig(format=spelling)
        names = sheet_image_names("synth_000000", 2, spec.suffix)
        assert names == (
            f"synth_000000_p01{suffix}",
            f"synth_000000_p02{suffix}",
        )

    def test_recorded_image_path_follows_the_format(
        self, config: SynthConfig
    ) -> None:
        spec = ImageFormatConfig(format="jpg")
        name = sheet_image_names("synth_000000", 1, spec.suffix)[0]
        assert config.paths.image_ref(name) == "images/synth_000000_p01.jpg"

    def test_cli_overrides_the_format_and_quality(self, config: SynthConfig) -> None:
        assert config.render.image.format == "png"
        overridden = _apply_overrides(
            config,
            dpi=None,
            output_dir=None,
            image_format="jpg",
            image_quality=80,
        )
        assert overridden.render.image.format == "jpeg"
        assert overridden.render.image.quality == 80
        assert overridden.render.image.suffix == ".jpg"
        # Unrelated settings survive an image-only override.
        assert overridden.render.dpi == config.render.dpi
        assert overridden.paths.output_dir == config.paths.output_dir

    def test_a_quality_override_keeps_the_configured_format(
        self, config: SynthConfig
    ) -> None:
        overridden = _apply_overrides(
            config, dpi=None, output_dir=None, image_quality=60
        )
        assert overridden.render.image.format == config.render.image.format
        assert overridden.render.image.quality == 60

    def test_an_out_of_range_quality_is_rejected_on_the_cli_too(
        self, config: SynthConfig
    ) -> None:
        with pytest.raises(ConfigError):
            _apply_overrides(config, dpi=None, output_dir=None, image_quality=0)

    def test_no_overrides_returns_the_same_config(self, config: SynthConfig) -> None:
        assert _apply_overrides(config, dpi=None, output_dir=None) is config


class TestArtifactLayout:
    """Renaming the artifact directories, which a consumer's tree dictates."""

    def test_defaults_match_the_documented_tree(self, config: SynthConfig) -> None:
        paths = config.paths
        assert paths.images_dir == paths.output_dir / "images"
        assert paths.annotations_dir == paths.output_dir / "annotations"
        assert paths.json_dir == paths.output_dir
        assert paths.image_ref("synth_000000_p01.png") == (
            "images/synth_000000_p01.png"
        )

    def test_overrides_move_every_artifact_and_the_recorded_path(
        self, config: SynthConfig, tmp_path: Path
    ) -> None:
        paths = replace(config.paths, output_dir=tmp_path).with_subdirs(
            {"images": "image", "json": "json"}
        )
        assert paths.images_dir == tmp_path / "image"
        assert paths.json_dir == tmp_path / "json"
        assert paths.image_ref("synth_000000_p01.png") == (
            "image/synth_000000_p01.png"
        )
        assert artifact_paths(
            replace(config, paths=paths), "synth_000000"
        )["annotation"] == (tmp_path / "annotations" / "synth_000000.json")

    def test_an_empty_name_writes_into_the_output_dir(
        self, config: SynthConfig, tmp_path: Path
    ) -> None:
        paths = replace(config.paths, output_dir=tmp_path).with_subdirs(
            {"images": ""}
        )
        assert paths.images_dir == tmp_path
        assert paths.image_ref("p.png") == "p.png"

    def test_a_subdir_cannot_escape_the_output_dir(
        self, config: SynthConfig
    ) -> None:
        with pytest.raises(ConfigError):
            config.paths.with_subdirs({"images": "../elsewhere"})
        with pytest.raises(ConfigError):
            config.paths.with_subdirs({"images": "/tmp/elsewhere"})

    def test_unknown_artifact_kinds_are_rejected(self, config: SynthConfig) -> None:
        with pytest.raises(ConfigError):
            config.paths.with_subdirs({"pictures": "image"})

    def test_cli_pairs_parse_into_overrides(self) -> None:
        assert parse_subdir_overrides(["images=image", "json=json"]) == {
            "images": "image",
            "json": "json",
        }

    def test_malformed_cli_pairs_are_rejected(self) -> None:
        with pytest.raises(ConfigError):
            parse_subdir_overrides(["image"])
        with pytest.raises(ConfigError):
            parse_subdir_overrides(["pictures=image"])


class TestPaddleFarmerProjection:
    """The flat ``{image, text}`` training file, built without a browser."""

    @staticmethod
    def _annotation() -> DocumentAnnotation:
        """Return a two-page annotation with one text region per page."""
        pages = tuple(
            Page(
                page_index=index,
                width=1240,
                height=1754,
                image_path=f"images/synth_000000_p{index + 1:02d}.png",
                elements=(
                    Element(
                        id=f"body.p{index}",
                        type=ElementType.PARAGRAPH,
                        bbox=BBox(60, 200, 1180, 320),
                        text=f"بند شماره {index}",
                        reading_order=0,
                    ),
                ),
            )
            for index in range(2)
        )
        return DocumentAnnotation(
            sample_id="synth_000000",
            source_sample_id="plaintext_000000",
            seed=1337,
            dpi=300,
            page_size="A4",
            orientation="portrait",
            pages=pages,
        )

    def test_one_record_per_page_in_page_order(self) -> None:
        records = paddle_farmer_records(self._annotation())
        assert [record["page_index"] for record in records] == [0, 1]
        assert [record["image"] for record in records] == [
            "images/synth_000000_p01.png",
            "images/synth_000000_p02.png",
        ]

    def test_records_carry_the_page_text_and_no_geometry(self) -> None:
        records = paddle_farmer_records(self._annotation())
        assert records[0]["text"] == "بند شماره 0"
        assert set(records[0]) == {"image", "text", "sample_id", "page_index"}
        assert all("bbox" not in json.dumps(record) for record in records)

    def test_image_paths_stay_relative_to_the_dataset_root(self) -> None:
        for record in paddle_farmer_records(self._annotation()):
            assert not Path(record["image"]).is_absolute()
            assert record["image"].startswith("images/")


@pytest.mark.slow
class TestRenderedSample:
    """Full renders: Chromium runs, artifacts land in ``tmp_path``.

    Marked ``slow`` because each test launches a browser. These write nothing
    to ``data/synth/`` - the config's ``output_dir`` is redirected first - so a
    test run never mixes test pages into the dataset.
    """

    @pytest.fixture
    def renderer(self) -> Iterator[Any]:
        """Yield an open :class:`PageRenderer`, skipping if Chromium is absent."""
        from fa_ocr_lora.dataset.synth.render.browser import PageRenderer, RenderError

        try:
            instance = PageRenderer()
        except RenderError as error:
            pytest.skip(f"Chromium unavailable: {error}")
        try:
            yield instance
        finally:
            instance.close()

    @staticmethod
    def _scoped(config: SynthConfig, tmp_path: Path, dpi: int = 150) -> SynthConfig:
        """Return the config with output redirected and the DPI lowered."""
        return replace(
            config,
            paths=replace(config.paths, output_dir=tmp_path),
            render=replace(config.render, dpi=dpi),
        )

    def test_one_sample_produces_every_artifact(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _TABLE_ROW)
        artifacts = render_sample(
            source,
            scoped,
            renderer,
            sample_id="synth_000004",
        )
        assert artifacts.pdf_path.is_file()
        assert artifacts.markdown_path.is_file()
        assert artifacts.annotation_path.is_file()
        assert all(path.is_file() for path in artifacts.image_paths)
        assert len(artifacts.image_paths) == len(artifacts.annotation.pages)
        assert artifacts.pdf_path.read_bytes().startswith(b"%PDF")

    def test_annotation_json_validates_and_round_trips(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        from fa_ocr_lora.dataset.common.schema import DocumentAnnotation

        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _TABLE_ROW)
        artifacts = render_sample(source, scoped, renderer, sample_id="synth_000004")
        payload = json.loads(artifacts.annotation_path.read_text(encoding="utf-8"))
        restored = DocumentAnnotation.from_dict(payload)
        assert restored.as_dict() == artifacts.annotation.as_dict()

    def test_no_bbox_escapes_its_page(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _TABLE_ROW)
        artifacts = render_sample(source, scoped, renderer, sample_id="synth_000004")
        for page in artifacts.annotation.pages:
            for element in page.elements:
                assert element.bbox.within(page.width, page.height), element.id

    def test_image_pixels_match_the_annotated_page_size(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        pillow = pytest.importorskip("PIL.Image")
        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _TABLE_ROW)
        artifacts = render_sample(source, scoped, renderer, sample_id="synth_000004")
        for page, path in zip(artifacts.annotation.pages, artifacts.image_paths):
            with pillow.open(path) as image:
                assert image.size == (page.width, page.height)

    def test_metadata_fields_are_boxed_individually(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _TABLE_ROW)
        artifacts = render_sample(source, scoped, renderer, sample_id="synth_000004")
        first_page = artifacts.annotation.pages[0]
        fields = first_page.elements_of(ElementType.METADATA_FIELD)
        assert fields, "the metadata block is the highest-priority region"
        assert all(field.text for field in fields)

    def test_table_cells_are_boxed_inside_their_table(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _TABLE_ROW)
        artifacts = render_sample(source, scoped, renderer, sample_id="synth_000004")
        tables = [
            element
            for _, element in artifacts.annotation.iter_elements()
            if element.type is ElementType.TABLE
        ]
        cells = [
            element
            for _, element in artifacts.annotation.iter_elements()
            if element.type is ElementType.TABLE_CELL
        ]
        assert tables and cells
        table = tables[0]
        assert table.text == "", "a container must not duplicate its cells' text"
        # The container keeps the structure instead: the cell matrix lives here,
        # under the key ``TableSpec.as_content`` actually emits.
        assert table.content.get("columns")
        assert table.content.get("rows")
        assert table.content["row_count"] == len(table.content["rows"])
        # Every cell has to sit inside the table that owns it, or the nesting
        # taught the model a containment that is not on the page.
        box = table.bbox
        for cell in cells:
            assert cell.bbox.x0 >= box.x0 - 1 and cell.bbox.x1 <= box.x1 + 1
            assert cell.bbox.y0 >= box.y0 - 1 and cell.bbox.y1 <= box.y1 + 1

    def test_long_letter_paginates_with_per_page_boxes(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _LONG_ROW)
        artifacts = render_sample(source, scoped, renderer, sample_id="synth_007666")
        assert len(artifacts.annotation.pages) > 1
        for page in artifacts.annotation.pages:
            assert page.elements, f"page {page.page_index} has no boxes"

    def test_unsigned_letter_has_no_signature_box(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _UNSIGNED_ROW)
        artifacts = render_sample(source, scoped, renderer, sample_id="synth_000002")
        types = {
            element.type for _, element in artifacts.annotation.iter_elements()
        }
        assert ElementType.SIGNATURE_IMAGE not in types

    def test_transcription_carries_the_subject_once(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _TABLE_ROW)
        artifacts = render_sample(source, scoped, renderer, sample_id="synth_000004")
        transcription = artifacts.annotation.full_transcription
        assert transcription
        assert source.subject[:20] in transcription

    def test_paddle_records_mirror_the_pages(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _TABLE_ROW)
        artifacts = render_sample(source, scoped, renderer, sample_id="synth_000004")
        records = paddle_records(artifacts.annotation)
        assert len(records) == len(artifacts.annotation.pages)
        for record, page in zip(records, artifacts.annotation.pages):
            assert record["image"] == page.image_path
            assert record["width"] == page.width
            assert len(record["label"]) == len(page.elements)
            json.dumps(record, ensure_ascii=False)

    def test_manifest_row_has_every_column(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _TABLE_ROW)
        artifacts = render_sample(source, scoped, renderer, sample_id="synth_000004")
        row = artifacts.manifest_row()
        assert set(row) == set(MANIFEST_COLUMNS)
        assert row["seed"] and row["page_count"] == str(
            len(artifacts.annotation.pages)
        )

    def test_render_is_reproducible_from_the_seed(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _TABLE_ROW)
        first = render_sample(
            source,
            scoped,
            renderer,
            sample_id="synth_000004",
            seed=7,
            write=False,
        )
        second = render_sample(
            source,
            scoped,
            renderer,
            sample_id="synth_000004",
            seed=7,
            write=False,
        )
        assert first.annotation.as_dict() == second.annotation.as_dict()

    def test_in_memory_render_writes_nothing(
        self,
        rows: dict[str, Mapping[str, str]],
        config: SynthConfig,
        renderer: Any,
        tmp_path: Path,
    ) -> None:
        scoped = self._scoped(config, tmp_path)
        source = _source(rows, config, _TABLE_ROW)
        render_sample(
            source,
            scoped,
            renderer,
            sample_id="synth_000004",
            write=False,
        )
        assert not list(tmp_path.rglob("*.png"))
        assert not list(tmp_path.rglob("*.json"))
