"""Orchestration: one manifest row in, four artifacts out.

A sample is produced in five steps, and every one of them is a pure function
of the previous plus the sample's seed:

``row -> LetterSource -> ComposedLetter -> RenderResult -> DocumentAnnotation``

Two properties of that chain are worth stating, because the rest of the module
is arranged around them:

* **The seed is the sample.** Nothing consults the clock, the filesystem
  listing order, or a global RNG. Re-running with the same ``--seed`` and the
  same row rewrites byte-identical JSON, which is the reproducibility
  requirement in CLAUDE.md 8 and also what makes ``--resume`` safe.
* **The browser is expensive; everything else is not.** One
  :class:`~.render.browser.PageRenderer` is held open for the whole batch and
  handed each document in turn. A run that crashes on sample 900 therefore
  loses one sample, not a Chromium launch per sample.

Artifacts land under the configured ``output_dir``: one image per page in the
configured format, one PDF, one
Markdown reconstruction, one annotation JSON, and one manifest row carrying the
seed so a later split can be reproduced. Two training files are appended
alongside them - ``paddle_farmer.jsonl`` (image path plus the page's full text)
and ``paddle_ocr_vl.jsonl`` (the same plus per-region boxes) - both projections
of the annotation, which stays the source of truth. The projections themselves
live in :mod:`~fa_ocr_lora.dataset.common.projections`, since the HuggingFace
exporter needs them too; they are re-exported from here for callers that had
them before. See ``docs/dataset_schema.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Iterable, Mapping, Sequence

from ...config import (
    DEFAULT_CONFIG_PATH,
    HF_FAMILIES,
    HF_GROUP_KEYS,
    IMAGE_FORMAT_CHOICES,
    SUBDIR_FIELDS,
    ConfigError,
    ImageFormatConfig,
    SynthConfig,
    load_config,
    parse_split_spec,
)
from ..common.io_utils import (
    ensure_dir,
    read_manifest,
    relative_to_repo,
    resolve_repo_path,
    sample_rng,
    write_bytes,
    write_json,
    write_text,
)
from ..common.projections import paddle_farmer_records, paddle_records
from ..common.schema import AnnotationError, DocumentAnnotation
from ..export.huggingface import ExportError, export_dataset
from ..export.upload import SIDECAR_UNIT, UPLOAD_MODES, upload_export
from .annotation import build_annotation, remap_boxes
from .layout.page_composer import ComposedLetter, compose_letter
from .letter_source import LetterSource, load_letter_source
from .render.browser import (
    PageRenderer,
    RenderError,
    RenderResult,
    RenderTarget,
    sheet_image_names,
)
from .render.document import TemplateError, render_document
from .render.scan import (
    SheetGeometry,
    apply_scan,
    build_scan_profile,
    flatten_clean,
)

__all__ = [
    "MANIFEST_COLUMNS",
    "SampleArtifacts",
    "paddle_farmer_records",
    "paddle_records",
    "parse_subdir_overrides",
    "render_sample",
    "run_batch",
    "main",
]

_LOGGER: Final[logging.Logger] = logging.getLogger("fa_ocr_lora.synth.pipeline")

#: Columns of ``data/synth/manifest.csv``. ``seed`` is what makes a sample
#: reproducible; the artifact paths are repo-relative so the manifest survives
#: being copied to a training host.
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "source_sample_id",
    "seed",
    "page_size",
    "orientation",
    "dpi",
    "page_count",
    "element_count",
    "texture",
    "has_table",
    "has_chart",
    "has_signature",
    "has_stamp",
    "overflowed",
    "image_paths",
    "pdf_path",
    "markdown_path",
    "annotation_path",
    "missing_assets",
)

#: Render-stage id format, e.g. ``synth_000042``.
_SAMPLE_ID_FORMAT: Final[str] = "synth_{index:06d}"

#: PaddleOCR-VL training file, appended one line per page, regions included.
_PADDLE_JSONL: Final[str] = "paddle_ocr_vl.jsonl"

#: PaddleFarmer training file: image path plus the page's full text, no
#: geometry. Appended one line per page.
_FARMER_JSONL: Final[str] = "paddle_farmer.jsonl"


@dataclass(frozen=True, slots=True)
class SampleArtifacts:
    """Where one sample's files landed, plus the annotation itself.

    Attributes:
        sample_id: Render-stage id.
        composed: The composed letter, kept so a caller can inspect the
            decisions that produced the page without re-deriving them.
        annotation: The ground truth written to ``annotations/``.
        image_paths: One image per page, in order.
        pdf_path: The whole document as one PDF.
        markdown_path: The Markdown reconstruction.
        annotation_path: The annotation JSON.
    """

    sample_id: str
    composed: ComposedLetter
    annotation: DocumentAnnotation
    image_paths: tuple[Path, ...]
    pdf_path: Path
    markdown_path: Path
    annotation_path: Path

    def manifest_row(self) -> dict[str, str]:
        """Return this sample as one manifest CSV row.

        Returns:
            A mapping keyed by :data:`MANIFEST_COLUMNS`; every value is a
            string so :mod:`csv` writes it without further coercion.
        """
        composed = self.composed
        fields = self.annotation.letter_fields
        return {
            "sample_id": self.sample_id,
            "source_sample_id": self.annotation.source_sample_id,
            "seed": str(self.annotation.seed),
            "page_size": self.annotation.page_size,
            "orientation": self.annotation.orientation,
            "dpi": str(self.annotation.dpi),
            "page_count": str(len(self.annotation.pages)),
            "element_count": str(self.annotation.element_count),
            "texture": self.annotation.texture or "",
            "has_table": str(composed.table is not None).lower(),
            "has_chart": str(composed.chart is not None).lower(),
            "has_signature": str(composed.signature is not None).lower(),
            "has_stamp": str(composed.stamp is not None).lower(),
            "overflowed": str(bool(fields.get("overflowed"))).lower(),
            "image_paths": ";".join(
                relative_to_repo(path) for path in self.image_paths
            ),
            "pdf_path": relative_to_repo(self.pdf_path),
            "markdown_path": relative_to_repo(self.markdown_path),
            "annotation_path": relative_to_repo(self.annotation_path),
            "missing_assets": ";".join(composed.source.missing_assets),
        }


def sample_id_for(index: int) -> str:
    """Return the render-stage id for a manifest position.

    Args:
        index: Zero-based position in the text-stage manifest.

    Returns:
        The id, e.g. ``synth_000042``. Derived from the row's position rather
        than a counter, so re-rendering a subset keeps every id it had.
    """
    return _SAMPLE_ID_FORMAT.format(index=index)


def artifact_paths(config: SynthConfig, sample_id: str) -> dict[str, Path]:
    """Return the four output paths for one sample.

    Args:
        config: Loaded configuration, for ``output_dir``.
        sample_id: Render-stage id.

    Returns:
        Mapping with ``pdf``, ``markdown`` and ``annotation`` keys. Image
        paths depend on the page count and are resolved after rendering.
    """
    paths = config.paths
    return {
        "pdf": paths.pdf_dir / f"{sample_id}.pdf",
        "markdown": paths.markdown_dir / f"{sample_id}.md",
        "annotation": paths.annotations_dir / f"{sample_id}.json",
    }


def is_complete(config: SynthConfig, sample_id: str) -> bool:
    """Return whether a sample's non-image artifacts already exist.

    The annotation is written last, so its presence means the whole sample
    survived. That is what ``--resume`` tests, rather than counting page images -
    a run killed between the first and second page screenshot would otherwise
    look finished.

    Args:
        config: Loaded configuration.
        sample_id: Render-stage id.

    Returns:
        ``True`` when the sample can be skipped.
    """
    return artifact_paths(config, sample_id)["annotation"].is_file()


def render_sample(
    source: LetterSource,
    config: SynthConfig,
    renderer: PageRenderer,
    *,
    sample_id: str,
    seed: int | None = None,
    write: bool = True,
) -> SampleArtifacts:
    """Compose, rasterise and annotate one letter.

    Args:
        source: The letter, loaded from its plaintext record.
        config: Loaded configuration.
        renderer: An open Chromium wrapper, reused across the batch.
        sample_id: Render-stage id for this sample.
        seed: Master seed; defaults to the configured one.
        write: Whether to persist the artifacts. ``False`` runs the whole
            chain in memory, which is what the tests use to check a real
            render without depositing files in the dataset directory.

    Returns:
        The artifacts, with paths filled in whether or not they were written.

    Raises:
        TemplateError: If the letter templates fail to render.
        RenderError: If Chromium cannot lay the document out.
        AnnotationError: If the harvested boxes violate the shared schema.
    """
    composed = compose_letter(source, config, sample_id=sample_id, seed=seed)
    html = render_document(composed.context)
    target = RenderTarget.from_config(
        composed.page_spec,
        composed.geometry.margin_mm,
        config.render,
    )
    result = renderer.render(html, target)

    image_names = sheet_image_names(
        sample_id, result.page_count, config.render.image.suffix
    )
    # The annotation stores ``<images_subdir>/<name>``: a path relative to the
    # dataset root rather than a bare filename, so a training host can resolve
    # it from the manifest's directory alone.
    annotation = build_annotation(
        composed,
        result,
        image_names=tuple(
            config.paths.image_ref(name) for name in image_names
        ),
    )

    # Chromium's capture is RGBA over a transparent ground. Something has to put
    # paper behind it before it is an image, so this step is not optional even
    # for the samples that skip the degradation.
    pngs, geometries, scan_content = _scan_sheets(
        result,
        config,
        composed,
        index=source.sample_index,
    )
    annotation = remap_boxes(annotation, geometries, scan=scan_content)

    paths = artifact_paths(config, sample_id)
    image_paths = tuple(config.paths.images_dir / name for name in image_names)
    if write:
        for png, path in zip(pngs, image_paths):
            write_bytes(path, png)
        write_bytes(paths["pdf"], result.pdf)
        write_text(paths["markdown"], composed.markdown)
        write_json(paths["annotation"], annotation.as_dict())

    return SampleArtifacts(
        sample_id=sample_id,
        composed=composed,
        annotation=annotation,
        image_paths=image_paths,
        pdf_path=paths["pdf"],
        markdown_path=paths["markdown"],
        annotation_path=paths["annotation"],
    )


def _scan_sheets(
    result: RenderResult,
    config: SynthConfig,
    composed: ComposedLetter,
    *,
    index: int,
) -> tuple[tuple[bytes, ...], tuple[SheetGeometry | None, ...], dict[str, Any] | None]:
    """Put every rasterised sheet of one document through the scanner.

    The profile is drawn once per document - one letter is printed on one kind
    of paper and passes the platen once - while the noise RNG is per sheet, so
    two pages of the same letter do not carry identical grain.

    Args:
        result: What the renderer returned, for its sheets.
        config: Loaded configuration, for the ``scan`` section.
        composed: The composed letter, for its seed and its chosen paper.
        index: The sample's index in the text-stage manifest, which is what
            makes the draws reproducible.

    Returns:
        ``(pngs, geometries, provenance)``: the flattened sheets in page order,
        the geometric map each one went through (``None`` where the sheet was
        not moved), and the scan decisions to record in the annotation
        (``None`` when this sample stayed born-digital).
    """
    profile = build_scan_profile(
        config.scan,
        composed.texture,
        result.page_count,
        sample_rng(composed.seed, "scan", index),
    )
    if profile is None:
        # The clean branch. The capture is still transparent, so it is flattened
        # onto the theme's own paper tone rather than shipped with an alpha
        # channel a training loader would read as black.
        return (
            tuple(
                flatten_clean(
                    sheet.png,
                    composed.theme.palette.surface,
                    (sheet.width, sheet.height),
                    config.render.image,
                )
                for sheet in result.sheets
            ),
            (None,) * len(result.sheets),
            None,
        )

    pngs: list[bytes] = []
    geometries: list[SheetGeometry | None] = []
    for sheet in result.sheets:
        png, geometry = apply_scan(
            sheet.png,
            profile.paper,
            profile.sheets[sheet.index],
            sample_rng(composed.seed, f"scan-noise:{sheet.index}", index),
            (sheet.width, sheet.height),
            config.render.image,
        )
        pngs.append(png)
        geometries.append(None if geometry.is_identity else geometry)
    return tuple(pngs), tuple(geometries), profile.as_content()


def _append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Append records to a JSONL file, creating it if absent.

    Args:
        path: Destination file.
        records: Objects to serialise, one per line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _append_manifest_row(path: Path, row: Mapping[str, str]) -> None:
    """Append one row to the render manifest, writing the header once.

    Appending per sample rather than at the end means a run killed at sample
    900 still leaves 899 usable rows.

    Args:
        path: The manifest CSV.
        row: One row keyed by :data:`MANIFEST_COLUMNS`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in MANIFEST_COLUMNS})


def _select_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    start: int,
    count: int | None,
    only: Sequence[str] = (),
) -> list[tuple[int, Mapping[str, str]]]:
    """Return the ``(index, row)`` pairs a run should process.

    The index is the row's position in the whole manifest, not in the slice,
    because it is what :func:`sample_id_for` and the per-sample RNG are keyed
    on - slicing must not renumber anything.

    Args:
        rows: Every row of the text-stage manifest.
        start: First row to consider.
        count: How many rows to take, or ``None`` for all remaining.
        only: Source sample ids to restrict to, ignoring ``start``/``count``.

    Returns:
        The selected pairs, in manifest order.
    """
    indexed = list(enumerate(rows))
    if only:
        wanted = set(only)
        return [pair for pair in indexed if pair[1].get("sample_id") in wanted]
    sliced = indexed[max(0, start) :]
    if count is not None:
        sliced = sliced[: max(0, count)]
    return sliced


def run_batch(
    config: SynthConfig,
    *,
    start: int = 0,
    count: int | None = None,
    only: Sequence[str] = (),
    seed: int | None = None,
    resume: bool = False,
    headless: bool = True,
) -> list[SampleArtifacts]:
    """Render a slice of the text-stage manifest.

    One Chromium instance serves the whole slice. A sample that fails is
    logged and skipped: at 20k samples a single malformed record must not end
    the run, and the manifest records exactly which samples exist.

    Args:
        config: Loaded configuration.
        start: First manifest row to render.
        count: How many rows to render, or ``None`` for all remaining.
        only: Source sample ids to render instead of a contiguous slice.
        seed: Master seed; defaults to the configured one.
        resume: Skip samples whose annotation already exists.
        headless: Passed to :class:`~.render.browser.PageRenderer`.

    Returns:
        The artifacts of every sample that rendered successfully.

    Raises:
        FileNotFoundError: If the text-stage manifest is missing.
        RenderError: If Chromium cannot be launched at all.
    """
    rows = read_manifest(config.paths.manifest)
    selected = _select_rows(rows, start=start, count=count, only=only)
    if not selected:
        _LOGGER.warning("no manifest rows selected; nothing to render")
        return []

    for directory in (
        config.paths.images_dir,
        config.paths.pdf_dir,
        config.paths.markdown_dir,
        config.paths.annotations_dir,
        config.paths.json_dir,
    ):
        ensure_dir(directory)

    manifest_path = config.paths.render_manifest
    paddle_path = config.paths.json_dir / _PADDLE_JSONL
    farmer_path = config.paths.json_dir / _FARMER_JSONL
    produced: list[SampleArtifacts] = []

    with PageRenderer(
        headless=headless, launch_args=config.render.chromium_args
    ) as renderer:
        for index, row in selected:
            sample_id = sample_id_for(index)
            if resume and is_complete(config, sample_id):
                _LOGGER.debug("%s already rendered; skipping", sample_id)
                continue
            try:
                source = load_letter_source(row, config)
            except (FileNotFoundError, KeyError, ValueError) as error:
                _LOGGER.warning("%s: source unusable: %s", sample_id, error)
                continue
            try:
                artifacts = render_sample(
                    source,
                    config,
                    renderer,
                    sample_id=sample_id,
                    seed=seed,
                )
            except (TemplateError, RenderError, AnnotationError, OSError) as error:
                _LOGGER.warning("%s: render failed: %s", sample_id, error)
                continue

            _append_manifest_row(manifest_path, artifacts.manifest_row())
            _append_jsonl(paddle_path, paddle_records(artifacts.annotation))
            _append_jsonl(
                farmer_path, paddle_farmer_records(artifacts.annotation)
            )
            produced.append(artifacts)
            _LOGGER.info(
                "%s: %d page(s), %d element(s), %s",
                sample_id,
                len(artifacts.annotation.pages),
                artifacts.annotation.element_count,
                artifacts.annotation.page_size,
            )

    _LOGGER.info("rendered %d of %d selected samples", len(produced), len(selected))
    return produced


def _build_parser() -> argparse.ArgumentParser:
    """Return the batch runner's argument parser.

    Returns:
        A parser whose defaults all come from the YAML config, so the CLI
        overrides settings rather than defining them (CLAUDE.md 8).
    """
    parser = argparse.ArgumentParser(
        prog="fa-ocr-synth",
        description=(
            "Render Persian administrative letters from the text stage's "
            "manifest into images + PDF + Markdown + bbox annotations."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "path to the synth config YAML (default: $FA_OCR_CONFIG, else "
            f"{DEFAULT_CONFIG_PATH})"
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="first manifest row to render (default: 0)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="how many rows to render (default: all remaining)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=(),
        metavar="SAMPLE_ID",
        help="render just these source sample ids, ignoring --start/--count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="master seed override; the same seed reproduces the same pages",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="raster resolution override for the page images",
    )
    parser.add_argument(
        "--image-format",
        choices=IMAGE_FORMAT_CHOICES,
        default=None,
        help=(
            "page image format override; the page filenames and the "
            "annotation's image_path follow it (default: render.image.format)"
        ),
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=None,
        metavar="1-100",
        help="encoder quality for jpeg/webp; ignored by png",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="write artifacts here instead of the configured output_dir",
    )
    parser.add_argument(
        "--subdir",
        action="append",
        default=[],
        metavar="KIND=NAME",
        help=(
            "rename an artifact directory, e.g. --subdir images=image "
            "--subdir json=json; kinds: "
            + ", ".join(sorted(SUBDIR_FIELDS))
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip samples whose annotation JSON already exists; with a "
            "--hugging-face flag, also continue an existing export instead of "
            "restarting it at shard 0"
        ),
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="run Chromium with a window, for debugging layout (no PDF)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="log warnings only",
    )

    export = parser.add_argument_group(
        "HuggingFace export",
        (
            "Pack the rendered corpus into sharded Parquet with a dataset card. "
            "The images are embedded in the rows, so a shard is as large as its "
            "pages - which is why the byte ceiling, not the sample count, is "
            "usually what closes a shard."
        ),
    )
    export.add_argument(
        "--hugging-face",
        action="store_true",
        help="export after rendering, covering the samples this run produced",
    )
    export.add_argument(
        "--hugging-face-only",
        action="store_true",
        help=(
            "export an already-rendered output_dir and render nothing; this is "
            "the mode to use against a corpus you do not want to regenerate"
        ),
    )
    export.add_argument(
        "--hf-output-dir",
        default=None,
        help=(
            "write the export here instead of the configured "
            "huggingface.output_dir (default: $FA_OCR_HF_OUTPUT_DIR, else the "
            "YAML value)"
        ),
    )
    export.add_argument(
        "--hf-splits",
        default=None,
        metavar="train=0.9,test=0.1",
        help=(
            "split shares; must sum to 1.0. Every page of one letter lands in "
            "one split, so this divides letters rather than pages. Changing it "
            "re-deals the whole corpus, which is why the value belongs in the "
            "config once a dataset has been published"
        ),
    )
    export.add_argument(
        "--hf-group-by",
        choices=HF_GROUP_KEYS,
        default=None,
        help=(
            "unit a split is assigned over (default: "
            "huggingface.group_by). source_sample_id keeps every render of one "
            "plaintext file together; sample_id keys per render and leaks when "
            "a file was rendered twice"
        ),
    )
    export.add_argument(
        "--hf-families",
        default=None,
        metavar="LIST",
        help=(
            "comma-separated artifact families to pack, each becoming one "
            "loadable config: " + ", ".join(HF_FAMILIES)
        ),
    )
    export.add_argument(
        "--hf-shard-samples",
        type=int,
        default=None,
        metavar="N",
        help="nominal samples per shard (default: huggingface.shard_samples)",
    )
    export.add_argument(
        "--hf-max-shard-bytes",
        type=int,
        default=None,
        metavar="BYTES",
        help="hard ceiling on one shard's payload; a full batch that would "
        "breach it closes early and the batch size steps down",
    )

    hub = parser.add_argument_group(
        "HuggingFace Hub upload",
        (
            "Push a finished export to a dataset repository, one commit per "
            "family with the dataset card last - the card's configs block is "
            "what the Hub viewer acts on, so it should not arrive before the "
            "shards it describes. The write token is read from HF_TOKEN (or "
            "`huggingface-cli login`) and is never passed on the command line."
        ),
    )
    hub.add_argument(
        "--hf-upload",
        action="store_true",
        help="upload the export tree after writing it",
    )
    hub.add_argument(
        "--hf-upload-only",
        action="store_true",
        help=(
            "upload an export tree that already exists and render nothing; "
            "this is the mode to re-run after a failed or partial upload"
        ),
    )
    hub.add_argument(
        "--hf-repo",
        default=None,
        metavar="OWNER/NAME",
        help=(
            "dataset repository to push to (default: $FA_OCR_HF_REPO_ID, else "
            "huggingface.hub_repo_id). Created if it does not exist"
        ),
    )
    hub.add_argument(
        "--hf-upload-mode",
        choices=UPLOAD_MODES,
        default="folder",
        help=(
            "folder: one commit per family, best up to a few GB each. "
            "large: the Hub's chunked resumable uploader, which survives an "
            "interrupted multi-GB push at the cost of many small commits"
        ),
    )
    hub.add_argument(
        "--hf-upload-units",
        default=None,
        metavar="LIST",
        help=(
            "comma-separated subset to push instead of everything present: a "
            "family name, or '" + SIDECAR_UNIT + "' for the card, "
            "dataset_info.json, the CSV indexes and the logo licence manifest"
        ),
    )
    hub.add_argument(
        "--hf-private",
        action="store_true",
        help=(
            "make the repository private if this run creates it; ignored for "
            "one that already exists, which the Hub does not let an upload flip"
        ),
    )
    hub.add_argument(
        "--hf-revision",
        default=None,
        metavar="BRANCH",
        help="commit to this branch, creating it if needed (folder mode only)",
    )
    hub.add_argument(
        "--hf-dry-run",
        action="store_true",
        help=(
            "list what would be pushed and stop. The token is still resolved, "
            "because a missing token is worth finding before a 40 GB upload"
        ),
    )
    return parser


def parse_subdir_overrides(pairs: Sequence[str]) -> dict[str, str]:
    """Parse ``--subdir KIND=NAME`` arguments into an override mapping.

    Args:
        pairs: Raw ``KIND=NAME`` strings as argparse collected them.

    Returns:
        Artifact kind mapped to directory name; empty when nothing was passed.

    Raises:
        ConfigError: If an argument has no ``=`` or names an unknown kind.
    """
    overrides: dict[str, str] = {}
    for pair in pairs:
        kind, separator, name = pair.partition("=")
        if not separator:
            raise ConfigError(f"--subdir expects KIND=NAME, got {pair!r}")
        kind = kind.strip()
        if kind not in SUBDIR_FIELDS:
            raise ConfigError(
                f"--subdir: unknown artifact kind {kind!r}; "
                f"expected any of {sorted(SUBDIR_FIELDS)}"
            )
        overrides[kind] = name.strip()
    return overrides


def _apply_overrides(
    config: SynthConfig,
    *,
    dpi: int | None,
    output_dir: str | None,
    image_format: str | None = None,
    image_quality: int | None = None,
    subdirs: Mapping[str, str] = MappingProxyType({}),
) -> SynthConfig:
    """Return a copy of the config with the CLI's overrides applied.

    Args:
        config: The loaded configuration.
        dpi: Raster resolution override, or ``None``.
        output_dir: Output directory override, or ``None``.
        image_format: Page image format override, or ``None``. Accepts the same
            spellings as the YAML key, aliases included.
        image_quality: Encoder quality override, or ``None``.
        subdirs: Artifact directory renames, as parsed by
            :func:`parse_subdir_overrides`.

    Returns:
        The config to run with; the original when nothing was overridden.

    Raises:
        ConfigError: If ``subdirs`` names an unknown artifact kind, or an image
            override is out of range - the same validator the YAML goes
            through, so the CLI cannot smuggle a value the file could not hold.
    """
    if (
        dpi is None
        and output_dir is None
        and image_format is None
        and image_quality is None
        and not subdirs
    ):
        return config
    render = config.render
    paths = config.paths
    if dpi is not None:
        render = replace(render, dpi=dpi)
    if image_format is not None or image_quality is not None:
        image = ImageFormatConfig(
            format=image_format if image_format is not None else render.image.format,
            quality=(
                image_quality if image_quality is not None else render.image.quality
            ),
            optimize=render.image.optimize,
            lossless=render.image.lossless,
        )
        render = replace(render, image=image)
    if output_dir is not None:
        paths = replace(paths, output_dir=resolve_repo_path(output_dir))
    paths = paths.with_subdirs(subdirs)
    return replace(config, render=render, paths=paths)


def _apply_hf_overrides(
    config: SynthConfig,
    *,
    output_dir: str | None = None,
    splits: str | None = None,
    group_by: str | None = None,
    families: str | None = None,
    shard_samples: int | None = None,
    max_shard_bytes: int | None = None,
    hub_repo_id: str | None = None,
    hub_private: bool = False,
) -> SynthConfig:
    """Return a copy of the config with the export's CLI overrides applied.

    Kept separate from :func:`_apply_overrides` because it touches a different
    section and is a no-op for every run that does not export - folding eight
    more parameters into a function the render path always calls would make the
    common case harder to read for no gain.

    The overrides go through :func:`dataclasses.replace`, so
    :class:`~....config.HuggingFaceConfig` re-runs every validator the YAML
    passes through - fractions summing to 1.0, family names, the shard limits'
    ordering. A ``--hf-shard-samples`` below the configured
    ``min_shard_samples`` would otherwise invert the step-down; it is caught
    here instead, and the minimum follows the new batch size when the CLI lowers
    it past that point.

    Args:
        config: The configuration as loaded and env-overlaid.
        output_dir: Export directory override, or ``None``.
        splits: ``train=0.9,test=0.1`` share override, or ``None``.
        group_by: Split group key override, or ``None``.
        families: Comma-separated family list override, or ``None``.
        shard_samples: Nominal batch size override, or ``None``.
        max_shard_bytes: Byte ceiling override, or ``None``.
        hub_repo_id: Upload destination override, or ``None``.
        hub_private: Whether ``--hf-private`` was passed. A flag, so ``False``
            means "not requested" and never overrides a configured ``True``.

    Returns:
        The config to export with; the original when nothing was overridden.

    Raises:
        ConfigError: If an override is out of range.
    """
    changes: dict[str, Any] = {}
    if output_dir is not None:
        changes["output_dir"] = resolve_repo_path(output_dir)
    if splits is not None:
        changes["splits"] = parse_split_spec(splits, field="--hf-splits")
    if group_by is not None:
        changes["group_by"] = group_by
    if families is not None:
        changes["families"] = _parse_family_list(families)
    if shard_samples is not None:
        changes["shard_samples"] = shard_samples
        # The step-down has to stay a smaller batch than the nominal one, so a
        # CLI batch below the configured minimum drags the minimum down with it
        # rather than failing validation the user did not ask to trip.
        changes["min_shard_samples"] = min(
            config.huggingface.min_shard_samples, shard_samples
        )
    if max_shard_bytes is not None:
        changes["max_shard_bytes"] = max_shard_bytes
    if hub_repo_id is not None:
        changes["hub_repo_id"] = hub_repo_id
    if hub_private:
        changes["hub_private"] = True
    if not changes:
        return config
    return replace(config, huggingface=replace(config.huggingface, **changes))


def _parse_family_list(raw: str) -> tuple[str, ...]:
    """Parse ``--hf-families images,pdf`` into family names.

    Args:
        raw: The comma-separated argument.

    Returns:
        The names, unvalidated - :class:`~....config.HuggingFaceConfig` is what
        rejects an unknown one, so the CLI and the YAML fail identically.

    Raises:
        ConfigError: If the argument names nothing.
    """
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise ConfigError("--hf-families must name at least one family")
    return names


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``fa-ocr-synth``.

    Args:
        argv: Command line arguments; defaults to :data:`sys.argv`.

    Returns:
        ``0`` when the run produced something, ``1`` otherwise - so a shell
        script driving a sweep can tell an empty run from a working one. With
        ``--hugging-face-only`` "something" is a completed export, and with
        ``--hf-upload-only`` a completed upload, since nothing is rendered in
        either mode; otherwise it is at least one rendered sample, and a
        requested export or upload that then fails turns a ``0`` into a ``1``.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    wants_export = args.hugging_face or args.hugging_face_only
    wants_upload = args.hf_upload or args.hf_upload_only

    try:
        config = _apply_overrides(
            load_config(args.config),
            dpi=args.dpi,
            output_dir=args.output_dir,
            image_format=args.image_format,
            image_quality=args.image_quality,
            subdirs=parse_subdir_overrides(args.subdir),
        )
        config = _apply_hf_overrides(
            config,
            output_dir=args.hf_output_dir,
            splits=args.hf_splits,
            group_by=args.hf_group_by,
            families=args.hf_families,
            shard_samples=args.hf_shard_samples,
            max_shard_bytes=args.hf_max_shard_bytes,
            hub_repo_id=args.hf_repo,
            hub_private=args.hf_private,
        )
    except (ConfigError, OSError) as error:
        _LOGGER.error("configuration unusable: %s", error)
        return 1

    if args.hf_upload_only:
        # The export tree is the input, so nothing is rendered and nothing is
        # re-packed: this is the mode to re-run after a failed upload.
        return 0 if _upload(config, args) else 1

    if args.hugging_face_only:
        # Nothing is rendered, so the export covers whatever the manifest lists
        # rather than a slice of this run.
        try:
            export_dataset(config, resume=args.resume)
        except ExportError as error:
            _LOGGER.error("export failed: %s", error)
            return 1
        return 0 if not args.hf_upload else (0 if _upload(config, args) else 1)

    try:
        produced = run_batch(
            config,
            start=args.start,
            count=args.count,
            only=tuple(args.only),
            seed=args.seed,
            resume=args.resume,
            headless=not args.headed,
        )
    except (FileNotFoundError, RenderError) as error:
        _LOGGER.error("run aborted: %s", error)
        return 1

    if wants_export and produced:
        try:
            export_dataset(
                config,
                sample_ids=[artifacts.sample_id for artifacts in produced],
                resume=args.resume,
            )
        except ExportError as error:
            _LOGGER.error("rendering succeeded but the export failed: %s", error)
            return 1

    if wants_upload and produced:
        # Only after an export: uploading a tree this run did not refresh would
        # publish shards that do not match the corpus just rendered.
        if not wants_export:
            _LOGGER.warning(
                "--hf-upload without --hugging-face uploads the export tree as "
                "it already stands; the samples rendered by this run are not in "
                "it"
            )
        if not _upload(config, args):
            return 1

    return 0 if produced else 1


def _upload(config: SynthConfig, args: argparse.Namespace) -> bool:
    """Push the export tree to the Hub, reporting failure as ``False``.

    Split out because three call sites need it and each of them has a different
    idea of what to return.

    Args:
        config: The configuration, already carrying any ``--hf-repo`` override.
        args: Parsed arguments.

    Returns:
        Whether the upload (or the dry run) succeeded.
    """
    try:
        summary = upload_export(
            config,
            families=(
                None
                if args.hf_upload_units is None
                else _parse_family_list(args.hf_upload_units)
            ),
            mode=args.hf_upload_mode,
            revision=args.hf_revision,
            dry_run=args.hf_dry_run,
        )
    except (ConfigError, ExportError) as error:
        _LOGGER.error("upload failed: %s", error)
        return False
    if not summary.dry_run:
        _LOGGER.info(
            "pushed %s to %s", ", ".join(summary.uploaded), summary.plan.url
        )
    return True


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
