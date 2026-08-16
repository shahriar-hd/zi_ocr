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

Artifacts land under the configured ``output_dir``: per-page PNGs, one PDF, one
Markdown reconstruction, one annotation JSON, and one manifest row carrying the
seed so a later split can be reproduced. Two training files are appended
alongside them - ``paddle_farmer.jsonl`` (image path plus the page's full text)
and ``paddle_ocr_vl.jsonl`` (the same plus per-region boxes) - both projections
of the annotation, which stays the source of truth. See
``docs/dataset_schema.md``.
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

from ..common.config import (
    DEFAULT_CONFIG_PATH,
    SUBDIR_FIELDS,
    ConfigError,
    SynthConfig,
    load_synth_config,
)
from ..common.io_utils import (
    ensure_dir,
    read_manifest,
    relative_to_repo,
    resolve_repo_path,
    write_bytes,
    write_json,
    write_text,
)
from ..common.schema import AnnotationError, DocumentAnnotation
from .annotation import build_annotation
from .layout.page_composer import ComposedLetter, compose_letter
from .letter_source import LetterSource, load_letter_source
from .render.browser import (
    PageRenderer,
    RenderError,
    RenderTarget,
    sheet_image_names,
)
from .render.document import TemplateError, render_document

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
        image_paths: One PNG per page, in order.
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
    survived. That is what ``--resume`` tests, rather than counting PNGs -
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

    image_names = sheet_image_names(sample_id, result.page_count)
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

    paths = artifact_paths(config, sample_id)
    image_paths = tuple(config.paths.images_dir / name for name in image_names)
    if write:
        for sheet, path in zip(result.sheets, image_paths):
            write_bytes(path, sheet.png)
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


def paddle_farmer_records(annotation: DocumentAnnotation) -> list[dict[str, Any]]:
    """Return one image-and-text record per page, for PaddleFarmer ingestion.

    This is the flat form of the ground truth: the path of a rendered page and
    every string printed on it, in RTL reading order, with no geometry at all.
    It is what a full-page recognition fine-tune actually reads, and it stays
    valid regardless of how precise the bounding boxes happen to be.

    ``image`` is relative to the dataset root - the directory holding the page
    images and this file - so the manifest is portable between machines.

    Args:
        annotation: The ground truth for one rendered letter.

    Returns:
        One record per page, in page order.
    """
    return [
        {
            "image": page.image_path,
            "text": page.transcription(),
            "sample_id": annotation.sample_id,
            "page_index": page.page_index,
        }
        for page in annotation.pages
    ]


def paddle_records(annotation: DocumentAnnotation) -> list[dict[str, Any]]:
    """Return one PaddleOCR-VL training record per page.

    PaddleOCR-VL consumes a JSONL of ``{image, label}`` pairs where the label
    is a list of regions. This is a projection of
    :class:`~..common.schema.DocumentAnnotation` - deliberately lossy, keeping
    only what the trainer reads - so the annotation JSON stays the source of
    truth and this file can be regenerated from it at any time.

    See :func:`paddle_farmer_records` for the geometry-free variant.

    Args:
        annotation: The ground truth for one rendered letter.

    Returns:
        One record per page, in page order.
    """
    records: list[dict[str, Any]] = []
    for page in annotation.pages:
        regions = [
            {
                "bbox": element.bbox.as_list(),
                "type": str(element.type),
                "text": element.text,
                "reading_order": element.reading_order,
            }
            for element in page.elements
        ]
        records.append(
            {
                "image": page.image_path,
                "width": page.width,
                "height": page.height,
                "sample_id": annotation.sample_id,
                "page_index": page.page_index,
                "text": page.transcription(),
                "label": regions,
            }
        )
    return records


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

    with PageRenderer(headless=headless) as renderer:
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
            "manifest into PNG + PDF + Markdown + bbox annotations."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"path to the synth config YAML (default: {DEFAULT_CONFIG_PATH})",
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
        help="raster resolution override for the page PNGs",
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
        help="skip samples whose annotation JSON already exists",
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
    subdirs: Mapping[str, str] = MappingProxyType({}),
) -> SynthConfig:
    """Return a copy of the config with the CLI's overrides applied.

    Args:
        config: The loaded configuration.
        dpi: Raster resolution override, or ``None``.
        output_dir: Output directory override, or ``None``.
        subdirs: Artifact directory renames, as parsed by
            :func:`parse_subdir_overrides`.

    Returns:
        The config to run with; the original when nothing was overridden.

    Raises:
        ConfigError: If ``subdirs`` names an unknown artifact kind.
    """
    if dpi is None and output_dir is None and not subdirs:
        return config
    render = config.render
    paths = config.paths
    if dpi is not None:
        render = replace(render, dpi=dpi)
    if output_dir is not None:
        paths = replace(paths, output_dir=resolve_repo_path(output_dir))
    paths = paths.with_subdirs(subdirs)
    return replace(config, render=render, paths=paths)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``fa-ocr-synth``.

    Args:
        argv: Command line arguments; defaults to :data:`sys.argv`.

    Returns:
        ``0`` when at least one sample rendered, ``1`` otherwise - so a shell
        script driving a sweep can tell an empty run from a working one.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        config = _apply_overrides(
            load_synth_config(args.config),
            dpi=args.dpi,
            output_dir=args.output_dir,
            subdirs=parse_subdir_overrides(args.subdir),
        )
    except (ConfigError, OSError) as error:
        _LOGGER.error("configuration unusable: %s", error)
        return 1

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

    return 0 if produced else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
