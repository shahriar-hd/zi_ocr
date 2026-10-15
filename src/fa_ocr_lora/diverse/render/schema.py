"""The annotation schema: lines, words and blocks with explicit reading order.

Separate from :mod:`fa_ocr_lora.dataset.common.schema`, which models an
administrative letter (``metadata_block``, ``signature_image``, ``stamp``) and
would have to be bent out of shape to describe a chat screenshot or a poetry
page. This schema describes what an OCR model is asked to produce: text lines
in reading order, optionally words, grouped into typed blocks.

Three invariants are enforced in ``__post_init__`` rather than trusted, because
each one, if violated, produces a sample that trains the model to be wrong:

* **Every box lies inside the canvas.** A line whose box extends past the edge
  means the text was clipped, so the label claims characters the image does not
  show.
* **Reading order is explicit and contiguous.** For right-to-left multi-column
  pages the order cannot be recovered from coordinates by any simple rule, so
  the renderer records it and the schema checks it is a permutation.
* **Polygons, not just rectangles.** After a perspective warp a line is no
  longer axis-aligned. Keeping four corners means the geometric augmentation
  can be applied to the annotation as faithfully as to the pixels.

The text of a sample is *derived* from its lines, never stored twice: see
:meth:`Annotation.text_from_lines`, which image QC compares against
``text_render`` to catch any drift between the label and the boxes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

__all__ = [
    "SCHEMA_VERSION",
    "Annotation",
    "AnnotationError",
    "Block",
    "BlockType",
    "Line",
    "Polygon",
    "Word",
]

#: Bumped when a field changes meaning. Recorded per sample so a mixed-version
#: corpus can be read.
SCHEMA_VERSION: Final[str] = "2.0"


class AnnotationError(ValueError):
    """Raised when an annotation is internally inconsistent."""


class BlockType(StrEnum):
    """What kind of region a block is.

    Deliberately coarse and layout-oriented rather than semantic. A model can
    learn "this is a heading" from geometry and weight; it cannot learn
    "this is the registration number" without domain supervision, and the
    previous dataset's fine-grained letter types do not generalise to a poster.
    """

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CAPTION = "caption"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    TABLE_HEADER = "table_header"
    FOOTER = "footer"
    HEADER = "header"
    PAGE_NUMBER = "page_number"
    CHAT_MESSAGE = "chat_message"
    CHAT_META = "chat_meta"
    BANNER_TEXT = "banner_text"
    VERSE = "verse"
    QUOTE = "quote"
    FOOTNOTE = "footnote"
    FIELD_LABEL = "field_label"
    FIELD_VALUE = "field_value"
    UI_STRING = "ui_string"
    LINK = "link"
    OTHER = "other"


#: A quadrilateral as four ``(x, y)`` corners in device pixels, clockwise from
#: top-left. A rectangle is the special case where corners share coordinates.
Polygon = tuple[tuple[float, float], ...]


def _validate_polygon(polygon: Polygon, context: str) -> None:
    """Check a polygon has four finite, non-negative corners.

    Args:
        polygon: The polygon.
        context: Identifier used in the error message.

    Raises:
        AnnotationError: If the shape is not a usable quadrilateral.
    """
    if len(polygon) != 4:
        raise AnnotationError(f"{context}: expected 4 corners, got {len(polygon)}")
    for index, corner in enumerate(polygon):
        if len(corner) != 2:
            raise AnnotationError(f"{context}: corner {index} is not an (x, y) pair")
        for value in corner:
            if not isinstance(value, (int, float)) or value != value:  # NaN check
                raise AnnotationError(f"{context}: corner {index} holds {value!r}")


def bounds(polygon: Polygon) -> tuple[float, float, float, float]:
    """Return the axis-aligned bounding box of a polygon.

    Args:
        polygon: The polygon.

    Returns:
        ``(x0, y0, x1, y1)``.
    """
    xs = [corner[0] for corner in polygon]
    ys = [corner[1] for corner in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def rectangle(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    """Build a polygon from an axis-aligned box.

    Args:
        x0: Left edge.
        y0: Top edge.
        x1: Right edge.
        y1: Bottom edge.

    Returns:
        The four corners, clockwise from top-left.
    """
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


@dataclass(frozen=True, slots=True)
class Word:
    """One word box.

    Attributes:
        text: The word exactly as drawn.
        polygon: Its four corners in device pixels.
    """

    text: str
    polygon: Polygon

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        return {"text": self.text, "polygon": [list(corner) for corner in self.polygon]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Word:
        """Rebuild from the JSON form.

        Args:
            payload: A mapping as produced by :meth:`as_dict`.

        Returns:
            The word.
        """
        return cls(
            text=str(payload["text"]),
            polygon=tuple(
                tuple(float(value) for value in corner) for corner in payload["polygon"]
            ),
        )


@dataclass(frozen=True, slots=True)
class Line:
    """One text line, which is the unit an OCR model predicts.

    Attributes:
        text: The line exactly as drawn.
        polygon: Its four corners in device pixels.
        reading_order: Zero-based position in the page's reading order.
        direction: ``rtl`` or ``ltr``. Recorded per line because a Persian page
            carrying a Latin product code has lines of both directions, and the
            direction is not recoverable from the text alone.
        block_index: Index of the owning block, or ``-1`` when unassigned.
        words: Optional word boxes.
    """

    text: str
    polygon: Polygon
    reading_order: int
    direction: str = "rtl"
    block_index: int = -1
    words: tuple[Word, ...] = ()

    def __post_init__(self) -> None:
        """Validate the polygon and the direction."""
        _validate_polygon(self.polygon, f"line {self.reading_order}")
        if self.direction not in {"rtl", "ltr"}:
            raise AnnotationError(
                f"line {self.reading_order}: direction must be rtl or ltr, "
                f"got {self.direction!r}"
            )

    @property
    def height(self) -> float:
        """Return the line's bounding-box height in pixels."""
        _, y0, _, y1 = bounds(self.polygon)
        return y1 - y0

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        payload: dict[str, Any] = {
            "text": self.text,
            "polygon": [list(corner) for corner in self.polygon],
            "reading_order": self.reading_order,
            "direction": self.direction,
            "block_index": self.block_index,
        }
        if self.words:
            payload["words"] = [word.as_dict() for word in self.words]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Line:
        """Rebuild from the JSON form.

        Args:
            payload: A mapping as produced by :meth:`as_dict`.

        Returns:
            The line.
        """
        return cls(
            text=str(payload["text"]),
            polygon=tuple(
                tuple(float(value) for value in corner) for corner in payload["polygon"]
            ),
            reading_order=int(payload["reading_order"]),
            direction=str(payload.get("direction", "rtl")),
            block_index=int(payload.get("block_index", -1)),
            words=tuple(Word.from_dict(item) for item in payload.get("words", ())),
        )


@dataclass(frozen=True, slots=True)
class Block:
    """A typed region grouping consecutive lines.

    Attributes:
        block_type: What kind of region this is.
        polygon: Its four corners in device pixels.
        reading_order: Zero-based position among blocks.
        line_indices: Indices into :attr:`Annotation.lines`.
    """

    block_type: BlockType
    polygon: Polygon
    reading_order: int
    line_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate the polygon."""
        _validate_polygon(self.polygon, f"block {self.reading_order}")

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        return {
            "type": str(self.block_type),
            "polygon": [list(corner) for corner in self.polygon],
            "reading_order": self.reading_order,
            "line_indices": list(self.line_indices),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Block:
        """Rebuild from the JSON form.

        Args:
            payload: A mapping as produced by :meth:`as_dict`.

        Returns:
            The block.
        """
        return cls(
            block_type=BlockType(payload["type"]),
            polygon=tuple(
                tuple(float(value) for value in corner) for corner in payload["polygon"]
            ),
            reading_order=int(payload["reading_order"]),
            line_indices=tuple(int(value) for value in payload.get("line_indices", ())),
        )


@dataclass(frozen=True, slots=True)
class Annotation:
    """Everything known about one rendered image.

    Attributes:
        sample_id: Identifier of this image.
        parent_id: Identifier of the page this was cropped from, or ``""`` for
            a full render. Links a crop to its source so group-aware splitting
            can keep them together.
        group_id: Splitting key. A page and all its crops share it, so no
            group straddles train and test.
        category: Visual category, which is also the Hub config name.
        width: Image width in pixels.
        height: Image height in pixels.
        lines: Text lines in reading order.
        blocks: Typed regions.
        schema_version: Format version.
    """

    sample_id: str
    parent_id: str
    group_id: str
    category: str
    width: int
    height: int
    lines: tuple[Line, ...] = ()
    blocks: tuple[Block, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the canvas, the boxes and the reading order."""
        if self.width <= 0 or self.height <= 0:
            raise AnnotationError(
                f"{self.sample_id}: canvas must be positive, "
                f"got {self.width}x{self.height}"
            )
        # A box outside the canvas means the text was clipped, so the label
        # describes characters the image does not show. One pixel of slack
        # absorbs Chromium's sub-pixel rounding.
        for line in self.lines:
            x0, y0, x1, y1 = bounds(line.polygon)
            if x0 < -1.0 or y0 < -1.0 or x1 > self.width + 1.0 or y1 > self.height + 1.0:
                raise AnnotationError(
                    f"{self.sample_id}: line {line.reading_order} box "
                    f"({x0:.1f},{y0:.1f})-({x1:.1f},{y1:.1f}) escapes the "
                    f"{self.width}x{self.height} canvas"
                )
        orders = sorted(line.reading_order for line in self.lines)
        if orders != list(range(len(self.lines))):
            raise AnnotationError(
                f"{self.sample_id}: line reading_order must be a contiguous "
                f"permutation of 0..{len(self.lines) - 1}, got {orders[:12]}"
            )

    @property
    def n_lines(self) -> int:
        """Return the number of lines."""
        return len(self.lines)

    @property
    def n_words(self) -> int:
        """Return the number of word boxes across all lines."""
        return sum(len(line.words) for line in self.lines)

    def ordered_lines(self) -> tuple[Line, ...]:
        """Return the lines sorted by reading order."""
        return tuple(sorted(self.lines, key=lambda line: line.reading_order))

    def text_from_lines(self) -> str:
        """Reconstruct the page text from the line boxes.

        Image QC compares this against ``text_render``. A mismatch means the
        harvest dropped, duplicated or reordered a line, which is the one bug
        class that silently poisons every label on the page.

        Returns:
            The lines joined by newlines, in reading order.
        """
        return "\n".join(line.text for line in self.ordered_lines())

    def min_line_height(self) -> float:
        """Return the shortest line's height, for the legibility check.

        Returns:
            The minimum height in pixels, or ``0.0`` when there are no lines.
        """
        return min((line.height for line in self.lines), default=0.0)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        return {
            "sample_id": self.sample_id,
            "parent_id": self.parent_id,
            "group_id": self.group_id,
            "category": self.category,
            "width": self.width,
            "height": self.height,
            "schema_version": self.schema_version,
            "lines": [line.as_dict() for line in self.ordered_lines()],
            "blocks": [block.as_dict() for block in self.blocks],
        }

    def to_json(self) -> str:
        """Return the compact JSON text written to disk and to Parquet."""
        return json.dumps(self.as_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Annotation:
        """Rebuild from the JSON form.

        Args:
            payload: A mapping as produced by :meth:`as_dict`.

        Returns:
            The annotation.
        """
        return cls(
            sample_id=str(payload["sample_id"]),
            parent_id=str(payload.get("parent_id", "")),
            group_id=str(payload["group_id"]),
            category=str(payload["category"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            lines=tuple(Line.from_dict(item) for item in payload.get("lines", ())),
            blocks=tuple(Block.from_dict(item) for item in payload.get("blocks", ())),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )

    def transformed(
        self,
        mapper: PolygonMapper,
        *,
        width: int,
        height: int,
        sample_id: str | None = None,
    ) -> Annotation:
        """Apply a geometric transform to every polygon.

        This is what keeps the ground truth attached to the pixels through
        rotation, perspective and rescaling: the same mapping that moves the
        ink moves the boxes. Lines whose transformed box leaves the canvas are
        dropped and the reading order is renumbered, because a partially
        visible line would otherwise claim text the image no longer shows.

        Args:
            mapper: Something that maps a point to its new position.
            width: New canvas width.
            height: New canvas height.
            sample_id: Override the sample id, for a derived crop.

        Returns:
            A new annotation with transformed geometry.
        """
        kept: list[Line] = []
        for line in self.ordered_lines():
            polygon = tuple(mapper.map_point(x, y) for x, y in line.polygon)
            x0, y0, x1, y1 = bounds(polygon)
            if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
                continue
            clipped = tuple(
                (min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height)))
                for x, y in polygon
            )
            words = tuple(
                Word(
                    text=word.text,
                    polygon=tuple(
                        (
                            min(max(px, 0.0), float(width)),
                            min(max(py, 0.0), float(height)),
                        )
                        for px, py in (
                            mapper.map_point(wx, wy) for wx, wy in word.polygon
                        )
                    ),
                )
                for word in line.words
            )
            kept.append(
                Line(
                    text=line.text,
                    polygon=clipped,
                    reading_order=len(kept),
                    direction=line.direction,
                    block_index=line.block_index,
                    words=words,
                )
            )

        surviving = {line.block_index for line in kept}
        blocks = tuple(
            Block(
                block_type=block.block_type,
                polygon=tuple(
                    (
                        min(max(x, 0.0), float(width)),
                        min(max(y, 0.0), float(height)),
                    )
                    for x, y in (mapper.map_point(bx, by) for bx, by in block.polygon)
                ),
                reading_order=index,
                line_indices=(),
            )
            for index, block in enumerate(
                block for block in self.blocks if block.reading_order in surviving
            )
        )
        return Annotation(
            sample_id=sample_id or self.sample_id,
            parent_id=self.parent_id,
            group_id=self.group_id,
            category=self.category,
            width=width,
            height=height,
            lines=tuple(kept),
            blocks=blocks,
        )


class PolygonMapper:
    """Protocol for a geometric transform applied to annotations.

    Implemented by the degradation stage's geometry so the pixels and the boxes
    move through exactly the same mapping. An identity implementation is
    provided for the clean-render path.
    """

    def map_point(self, x: float, y: float) -> tuple[float, float]:
        """Map one point to its transformed position.

        Args:
            x: Source x in pixels.
            y: Source y in pixels.

        Returns:
            The destination ``(x, y)``.
        """
        return x, y
