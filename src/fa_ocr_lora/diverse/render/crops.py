"""Derived crops: line, paragraph and word images cut from a rendered page.

A page teaches a model about layout; a line crop teaches it about letterforms.
Both are needed, and cutting crops from pages that already exist is far cheaper
than generating more pages - the LLM call, the font embedding and the browser
render are all already paid for.

Three rules keep the crops honest:

* **The crop's annotation is the parent's, translated.** Boxes are shifted by
  the crop origin, not re-measured, so a crop cannot disagree with the page it
  came from.
* **Crops inherit ``group_id``.** A page and every crop taken from it land in
  the same split. Without this the model would see a line in training and its
  own parent page in test, and the test score would be meaningless.
* **Padding is random but bounded.** A crop cut exactly on the box clips
  ascenders and descenders; one with huge padding is just the page again.
  Padding is drawn per crop as a fraction of line height, so it scales with
  the text rather than with the page.

Word crops are deliberately a small fraction. A 40x20 pixel image carries
little training signal per unit of compute, and a corpus dominated by them
would spend most of its bytes teaching the model almost nothing.
"""

from __future__ import annotations

import io
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .schema import Annotation, Block, BlockType, Line, Word, bounds, rectangle

__all__ = ["Crop", "derive_crops"]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: Smallest crop worth keeping, in pixels. Below this the image is noise:
#: a model cannot read an 8-pixel-tall line and training on it teaches
#: hallucination.
_MIN_SIDE: Final[int] = 16

#: Smallest line height, in pixels, for a line crop to be emitted at all.
_MIN_LINE_HEIGHT: Final[float] = 10.0


@dataclass(frozen=True, slots=True)
class Crop:
    """One derived image and its annotation.

    Attributes:
        sample_id: Identifier of the crop.
        category: ``line``, ``paragraph`` or ``word``.
        payload: Encoded image bytes.
        image_format: ``png`` or ``jpeg``.
        annotation: Geometry in the crop's own coordinate system.
        text_render: The crop's label.
        parent_id: The page this came from.
    """

    sample_id: str
    category: str
    payload: bytes
    image_format: str
    annotation: Annotation
    text_render: str
    parent_id: str


def _clamp_box(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Clamp a float box to integer pixel bounds inside the canvas.

    Args:
        x0: Left edge.
        y0: Top edge.
        x1: Right edge.
        y1: Bottom edge.
        width: Canvas width.
        height: Canvas height.

    Returns:
        The integer ``(x0, y0, x1, y1)``.
    """
    return (
        max(0, int(round(x0))),
        max(0, int(round(y0))),
        min(width, int(round(x1))),
        min(height, int(round(y1))),
    )


def _crop_annotation(
    parent: Annotation,
    lines: Sequence[Line],
    box: tuple[int, int, int, int],
    *,
    sample_id: str,
    category: str,
    block_type: BlockType,
) -> Annotation:
    """Build the annotation of a crop by translating the parent's boxes.

    Args:
        parent: The page's annotation.
        lines: The lines falling inside the crop.
        box: The crop rectangle in parent coordinates.
        sample_id: Identifier of the crop.
        category: Crop category.
        block_type: Type of the single block the crop contains.

    Returns:
        The crop's annotation, in crop-local coordinates.
    """
    left, top, right, bottom = box
    width, height = right - left, bottom - top

    def shift(polygon: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
        """Translate a polygon into crop-local coordinates and clamp it."""
        return tuple(
            (
                min(max(x - left, 0.0), float(width)),
                min(max(y - top, 0.0), float(height)),
            )
            for x, y in polygon
        )

    shifted = [
        Line(
            text=line.text,
            polygon=shift(line.polygon),
            reading_order=index,
            direction=line.direction,
            block_index=0,
            words=tuple(
                Word(text=word.text, polygon=shift(word.polygon)) for word in line.words
            ),
        )
        for index, line in enumerate(lines)
    ]
    return Annotation(
        sample_id=sample_id,
        parent_id=parent.sample_id,
        group_id=parent.group_id,
        category=category,
        width=width,
        height=height,
        lines=tuple(shifted),
        blocks=(
            Block(
                block_type=block_type,
                polygon=rectangle(0.0, 0.0, float(width), float(height)),
                reading_order=0,
                line_indices=tuple(range(len(shifted))),
            ),
        ),
    )


def derive_crops(
    page_payload: bytes,
    annotation: Annotation,
    rng: random.Random,
    *,
    want_line: bool,
    want_paragraph: bool,
    word_fraction: float,
    padding_min: float,
    padding_max: float,
    image_format: str = "png",
    jpeg_quality: int = 90,
    max_line_crops: int = 24,
) -> list[Crop]:
    """Cut line, paragraph and word crops from a rendered page.

    Args:
        page_payload: The page's encoded image.
        annotation: The page's annotation, already in final (post-degradation)
            coordinates.
        rng: The sample's seeded RNG.
        want_line: Whether to emit line crops.
        want_paragraph: Whether to emit paragraph (block) crops.
        word_fraction: Probability per word of emitting a word crop.
        padding_min: Minimum padding as a fraction of line height.
        padding_max: Maximum padding as a fraction of line height.
        image_format: Encoding for the crops.
        jpeg_quality: JPEG quality when encoding as JPEG.
        max_line_crops: Ceiling on line crops per page. A dense newspaper page
            has 200 lines; taking all of them would let one page dominate the
            ``line`` category and quietly narrow its font and layout variety.

    Returns:
        The crops, possibly empty.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - declared dependency
        LOGGER.error("Pillow is required to derive crops")
        return []

    from .degrade import encode_image

    crops: list[Crop] = []
    with Image.open(io.BytesIO(page_payload)) as opened:
        page = opened.convert("RGB")
    width, height = page.size

    def emit(
        box: tuple[int, int, int, int],
        lines: Sequence[Line],
        *,
        suffix: str,
        category: str,
        block_type: BlockType,
    ) -> None:
        """Cut one crop and append it, if it is large enough to be useful."""
        left, top, right, bottom = box
        if right - left < _MIN_SIDE or bottom - top < _MIN_SIDE:
            return
        text = "\n".join(line.text for line in lines).strip()
        if not text:
            return
        image = page.crop(box)
        sample_id = f"{annotation.sample_id}_{suffix}"
        try:
            crop_annotation = _crop_annotation(
                annotation,
                lines,
                box,
                sample_id=sample_id,
                category=category,
                block_type=block_type,
            )
        except Exception as error:  # noqa: BLE001 - a bad crop is skipped, not fatal
            LOGGER.debug("skipping crop %s: %s", sample_id, error)
            return
        crops.append(
            Crop(
                sample_id=sample_id,
                category=category,
                payload=encode_image(
                    image, image_format=image_format, quality=jpeg_quality
                ),
                image_format=image_format,
                annotation=crop_annotation,
                text_render=text,
                parent_id=annotation.sample_id,
            )
        )

    ordered = annotation.ordered_lines()

    if want_line and ordered:
        candidates = [line for line in ordered if line.height >= _MIN_LINE_HEIGHT]
        if len(candidates) > max_line_crops:
            candidates = rng.sample(candidates, max_line_crops)
        for line in candidates:
            x0, y0, x1, y1 = bounds(line.polygon)
            pad = line.height * rng.uniform(padding_min, padding_max)
            # Horizontal padding is deliberately larger: a line's box ends at
            # the last glyph's advance, and Persian finals often overhang it.
            emit(
                _clamp_box(
                    x0 - pad * 1.6, y0 - pad, x1 + pad * 1.6, y1 + pad, width, height
                ),
                [line],
                suffix=f"l{line.reading_order:03d}",
                category="line",
                block_type=BlockType.PARAGRAPH,
            )

    if want_paragraph:
        for block in annotation.blocks:
            members = [
                line for line in ordered if line.block_index == block.reading_order
            ]
            if len(members) < 2:
                continue  # a one-line block is a line crop, not a paragraph
            x0, y0, x1, y1 = bounds(block.polygon)
            reference = max(line.height for line in members)
            pad = reference * rng.uniform(padding_min, padding_max)
            emit(
                _clamp_box(x0 - pad, y0 - pad, x1 + pad, y1 + pad, width, height),
                members,
                suffix=f"p{block.reading_order:03d}",
                category="paragraph",
                block_type=block.block_type,
            )

    if word_fraction > 0.0:
        for line in ordered:
            for index, word in enumerate(line.words):
                if rng.random() >= word_fraction:
                    continue
                x0, y0, x1, y1 = bounds(word.polygon)
                pad = max(2.0, line.height * rng.uniform(padding_min, padding_max))
                single = Line(
                    text=word.text,
                    polygon=word.polygon,
                    reading_order=0,
                    direction=line.direction,
                    block_index=0,
                    words=(word,),
                )
                emit(
                    _clamp_box(x0 - pad, y0 - pad, x1 + pad, y1 + pad, width, height),
                    [single],
                    suffix=f"w{line.reading_order:03d}_{index:02d}",
                    category="word",
                    block_type=BlockType.OTHER,
                )

    return crops
