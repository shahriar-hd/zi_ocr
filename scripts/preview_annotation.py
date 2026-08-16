"""Draw an annotation's bounding boxes onto its page images.

A schema-valid annotation can still be visually wrong: a box can be one line
too low, a container can swallow its neighbour, a cell can be measured before
the table finished laying out. None of that is detectable from the JSON. This
script is the check that catches it - it re-draws every box on the page it was
measured on, colour-coded by element type and captioned with the type and
reading order, and writes the result beside the original.

Usage::

    python scripts/preview_annotation.py synth_000004
    python scripts/preview_annotation.py --all --limit 20
    python scripts/preview_annotation.py synth_000004 --types table_cell logo

Output lands in ``data/synth/preview/`` by default, one PNG per annotated page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fa_ocr_lora.dataset.common.schema import (  # noqa: E402  (path set above)
    DocumentAnnotation,
    Element,
)

#: Outline colour per element type. Regions that must be pixel-accurate get
#: saturated colours; decoration gets muted ones, so a misaligned metadata
#: field is obvious at a glance while a header bar does not shout.
_COLOURS: Final[dict[str, tuple[int, int, int]]] = {
    "metadata_block": (255, 0, 0),
    "metadata_field": (255, 64, 0),
    "table": (0, 128, 255),
    "table_cell": (0, 200, 255),
    "chart": (170, 0, 255),
    "subject": (0, 190, 0),
    "heading": (0, 150, 60),
    "paragraph": (60, 60, 220),
    "list_item": (120, 90, 220),
    "recipient": (220, 140, 0),
    "sender": (200, 170, 0),
    "attachments_block": (150, 110, 0),
    "signature_block": (255, 0, 190),
    "signature_image": (255, 90, 200),
    "signatory_title": (200, 0, 150),
    "signatory_name": (230, 0, 170),
    "stamp": (140, 0, 90),
    "logo": (120, 120, 120),
    "header_bar": (170, 170, 170),
    "header_company_name": (90, 90, 90),
    "header_slogan": (140, 140, 140),
    "religious_opener": (0, 170, 150),
    "footer_bar": (170, 170, 170),
    "footer_field": (110, 110, 110),
    "page_number": (110, 110, 110),
}

_DEFAULT_COLOUR: Final[tuple[int, int, int]] = (0, 0, 0)

#: Caption height as a fraction of the page height, so an A5 preview is as
#: readable as an A3 one.
_CAPTION_SCALE: Final[float] = 0.011


def _colour_for(element: Element) -> tuple[int, int, int]:
    """Return the outline colour for one element."""
    return _COLOURS.get(str(element.type), _DEFAULT_COLOUR)


def _load_font(size: int) -> Any:
    """Return a truetype font at ``size``, falling back to Pillow's default.

    Args:
        size: Desired pixel size.

    Returns:
        A Pillow font object. Captions are ASCII (the type name and index), so
        the fallback bitmap font is legible even without a system face.
    """
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _draw_page(
    image_path: Path,
    elements: Sequence[Element],
    output_path: Path,
    *,
    caption: bool = True,
) -> Path:
    """Draw one page's boxes and save the overlay.

    Args:
        image_path: The rendered page PNG.
        elements: Elements measured on that page.
        output_path: Where to write the overlay.
        caption: Whether to label each box with its type and reading order.

    Returns:
        The written path.

    Raises:
        FileNotFoundError: If the page image is missing.
    """
    from PIL import Image, ImageDraw

    if not image_path.is_file():
        raise FileNotFoundError(f"page image not found: {image_path}")

    with Image.open(image_path) as source:
        canvas = source.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    font_size = max(11, int(canvas.height * _CAPTION_SCALE))
    font = _load_font(font_size)
    width = max(2, canvas.width // 900)

    for element in elements:
        colour = _colour_for(element)
        x0, y0, x1, y1 = element.bbox.as_list()
        draw.rectangle((x0, y0, x1, y1), outline=colour, width=width)
        if not caption:
            continue
        label = f"{element.reading_order}:{element.type}"
        # Captions sit above the box, or inside it when the box touches the
        # top edge, so a header label is never clipped off the page.
        text_y = y0 - font_size - 2
        if text_y < 0:
            text_y = y0 + 2
        draw.text((x0 + 2, text_y), label, fill=colour, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path


def preview(
    annotation_path: Path,
    *,
    dataset_root: Path,
    output_dir: Path,
    types: Sequence[str] = (),
    caption: bool = True,
) -> list[Path]:
    """Render every page of one annotation with its boxes drawn on.

    Args:
        annotation_path: The ``annotations/synth_XXXXXX.json`` file.
        dataset_root: Directory the annotation's ``image_path`` values are
            relative to, normally ``data/synth``.
        output_dir: Where the overlays are written.
        types: Restrict drawing to these element types; empty draws all.
        caption: Whether to label each box.

    Returns:
        One path per written overlay.

    Raises:
        FileNotFoundError: If the annotation or one of its images is missing.
    """
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation = DocumentAnnotation.from_dict(payload)
    wanted = {name.strip() for name in types if name.strip()}

    written: list[Path] = []
    for page in annotation.pages:
        elements = [
            element
            for element in page.elements
            if not wanted or str(element.type) in wanted
        ]
        image_path = dataset_root / page.image_path
        target = output_dir / f"{annotation.sample_id}_p{page.page_index + 1:02d}.png"
        written.append(
            _draw_page(image_path, elements, target, caption=caption)
        )
    return written


def _iter_annotations(directory: Path, limit: int | None) -> Iterable[Path]:
    """Yield annotation files in sample order, up to ``limit``."""
    paths = sorted(directory.glob("synth_*.json"))
    return paths if limit is None else paths[:limit]


def _display(path: Path) -> str:
    """Return ``path`` relative to the repo when possible, else absolute.

    A dataset root outside the checkout - a scratch directory, an external
    disk - is a legitimate way to run this, so the printed path must not
    depend on the target sitting under the repo.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _build_parser() -> argparse.ArgumentParser:
    """Return this script's argument parser."""
    parser = argparse.ArgumentParser(
        description="Draw annotation bounding boxes onto their page images.",
    )
    parser.add_argument(
        "sample_ids",
        nargs="*",
        help="render-stage ids, e.g. synth_000004 (default: use --all)",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/synth",
        help="dataset directory holding images/ and annotations/",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="where to write overlays (default: <dataset-root>/preview)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="preview every annotation found, subject to --limit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="with --all, stop after this many samples",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=(),
        metavar="TYPE",
        help="draw only these element types, e.g. metadata_field table_cell",
    )
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="draw outlines only, for a cleaner look on dense pages",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command line arguments; defaults to :data:`sys.argv`.

    Returns:
        ``0`` when at least one overlay was written, ``1`` otherwise.
    """
    args = _build_parser().parse_args(argv)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = REPO_ROOT / dataset_root
    annotations_dir = dataset_root / "annotations"
    output_dir = (
        Path(args.output_dir) if args.output_dir else dataset_root / "preview"
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    if args.all:
        targets = list(_iter_annotations(annotations_dir, args.limit))
    else:
        targets = [annotations_dir / f"{name}.json" for name in args.sample_ids]

    if not targets:
        print(
            "nothing to preview: pass sample ids or --all",
            file=sys.stderr,
        )
        return 1

    written = 0
    for annotation_path in targets:
        if not annotation_path.is_file():
            print(f"skipping {annotation_path.name}: not found", file=sys.stderr)
            continue
        try:
            paths = preview(
                annotation_path,
                dataset_root=dataset_root,
                output_dir=output_dir,
                types=tuple(args.types),
                caption=not args.no_caption,
            )
        except (FileNotFoundError, ValueError, OSError) as error:
            print(f"skipping {annotation_path.name}: {error}", file=sys.stderr)
            continue
        written += len(paths)
        for path in paths:
            print(_display(path))

    print(f"wrote {written} overlay(s) to {_display(output_dir)}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
