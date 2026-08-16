"""Shared annotation schema for real and synthetic Persian OCR samples.

Every dataset producer in this project - the Persian Pixel filter and each
synthetic generator - emits :class:`DocumentAnnotation`. Downstream training
code therefore never special-cases the data source or the element type.

Coordinates are always **device pixels** of the rendered page image, with the
origin at the top-left corner, matching the PNG that ships alongside the
annotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Iterable, Sequence

__all__ = [
    "SCHEMA_VERSION",
    "AnnotationError",
    "ElementType",
    "BBox",
    "Element",
    "Page",
    "DocumentAnnotation",
]

SCHEMA_VERSION: Final[str] = "1.0"


class AnnotationError(ValueError):
    """Raised when an annotation violates the shared schema."""


class ElementType(StrEnum):
    """Every annotatable region a page may contain.

    The values double as the ``data-ann-type`` attribute in the HTML
    template, so renaming one requires updating the template too.
    """

    # Header region.
    LOGO = "logo"
    HEADER_BAR = "header_bar"
    HEADER_COMPANY_NAME = "header_company_name"
    HEADER_SLOGAN = "header_slogan"
    RELIGIOUS_OPENER = "religious_opener"

    # Administrative metadata - highest priority per CLAUDE.md 5.2.
    METADATA_BLOCK = "metadata_block"
    METADATA_FIELD = "metadata_field"

    # Letter frame.
    SENDER = "sender"
    RECIPIENT = "recipient"
    SUBJECT = "subject"

    # Body content.
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    CHART = "chart"
    ATTACHMENTS_BLOCK = "attachments_block"

    # Closing region.
    SIGNATURE_BLOCK = "signature_block"
    SIGNATURE_IMAGE = "signature_image"
    SIGNATORY_NAME = "signatory_name"
    SIGNATORY_TITLE = "signatory_title"
    STAMP = "stamp"

    # Footer region.
    FOOTER_BAR = "footer_bar"
    FOOTER_FIELD = "footer_field"
    PAGE_NUMBER = "page_number"


#: Types whose text is machine-generated decoration rather than transcribable
#: content. They still get a bounding box, but are skipped when building the
#: full-page transcription so it reads as the letter, not as the layout.
NON_TEXT_TYPES: Final[frozenset[ElementType]] = frozenset(
    {
        ElementType.LOGO,
        ElementType.HEADER_BAR,
        ElementType.FOOTER_BAR,
        ElementType.SIGNATURE_IMAGE,
        ElementType.STAMP,
    }
)

#: Container types that wrap finer-grained children. Their text duplicates the
#: children's, so the transcription uses the children instead.
CONTAINER_TYPES: Final[frozenset[ElementType]] = frozenset(
    {
        ElementType.METADATA_BLOCK,
        ElementType.TABLE,
        ElementType.SIGNATURE_BLOCK,
    }
)


@dataclass(frozen=True, slots=True)
class BBox:
    """An axis-aligned bounding box in device pixels, ``[x0, y0, x1, y1]``."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise AnnotationError(
                f"bbox must have positive extent, got {self.as_list()}"
            )
        if self.x0 < 0 or self.y0 < 0:
            raise AnnotationError(f"bbox must be non-negative, got {self.as_list()}")

    @property
    def width(self) -> int:
        """Box width in pixels."""
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """Box height in pixels."""
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def center_y(self) -> float:
        """Vertical midpoint, used when grouping boxes into reading lines."""
        return (self.y0 + self.y1) / 2

    def as_list(self) -> list[int]:
        """Return the box as a ``[x0, y0, x1, y1]`` list for JSON output."""
        return [self.x0, self.y0, self.x1, self.y1]

    def within(self, width: int, height: int) -> bool:
        """Return whether the box lies fully inside a ``width`` x ``height`` page."""
        return self.x1 <= width and self.y1 <= height

    def clipped_to(self, width: int, height: int) -> "BBox":
        """Return a copy clamped to the given page bounds."""
        return BBox(
            x0=max(0, min(self.x0, width - 1)),
            y0=max(0, min(self.y0, height - 1)),
            x1=max(1, min(self.x1, width)),
            y1=max(1, min(self.y1, height)),
        )

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "BBox":
        """Build a box from any four-value iterable, rounding to whole pixels."""
        coords = [int(round(float(value))) for value in values]
        if len(coords) != 4:
            raise AnnotationError(f"bbox needs exactly 4 values, got {len(coords)}")
        return cls(*coords)


@dataclass(frozen=True, slots=True)
class Element:
    """One annotated region on a rendered page.

    Attributes:
        id: Stable, page-unique key such as ``meta.registration_number``, so
            consumers can address a region without re-deriving it from
            geometry.
        type: Which kind of region this is.
        bbox: Device-pixel bounds on the page image.
        text: The text the region renders, empty for non-text decoration.
        reading_order: Zero-based index in RTL reading order within the page.
        content: Element-specific structured payload - table cells, chart
            series, metadata label/value pairs. Empty for plain text runs.
    """

    id: str
    type: ElementType
    bbox: BBox
    text: str = ""
    reading_order: int = 0
    content: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise AnnotationError("element id must be non-empty")
        if self.reading_order < 0:
            raise AnnotationError(
                f"reading_order must be non-negative, got {self.reading_order}"
            )

    @property
    def is_transcribable(self) -> bool:
        """Return whether this element contributes to the page transcription."""
        return (
            self.type not in NON_TEXT_TYPES
            and self.type not in CONTAINER_TYPES
            and bool(self.text.strip())
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize to the JSON shape documented in ``docs/dataset_schema.md``."""
        payload: dict[str, Any] = {
            "id": self.id,
            "type": str(self.type),
            "bbox": self.bbox.as_list(),
            "text": self.text,
            "reading_order": self.reading_order,
        }
        if self.content:
            payload["content"] = self.content
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Element":
        """Rebuild an element from its serialized form."""
        try:
            element_type = ElementType(payload["type"])
        except (KeyError, ValueError) as exc:
            raise AnnotationError(f"unknown element type: {payload.get('type')}") from exc
        return cls(
            id=str(payload["id"]),
            type=element_type,
            bbox=BBox.from_iterable(payload["bbox"]),
            text=str(payload.get("text", "")),
            reading_order=int(payload.get("reading_order", 0)),
            content=dict(payload.get("content") or {}),
        )


@dataclass(frozen=True, slots=True)
class Page:
    """One rasterized page and every region annotated on it."""

    page_index: int
    width: int
    height: int
    image_path: str
    elements: tuple[Element, ...] = ()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise AnnotationError(
                f"page {self.page_index} has non-positive size "
                f"{self.width}x{self.height}"
            )
        seen: set[str] = set()
        for element in self.elements:
            if element.id in seen:
                raise AnnotationError(
                    f"duplicate element id {element.id!r} on page {self.page_index}"
                )
            seen.add(element.id)
            if not element.bbox.within(self.width, self.height):
                raise AnnotationError(
                    f"element {element.id!r} bbox {element.bbox.as_list()} "
                    f"escapes page {self.page_index} ({self.width}x{self.height})"
                )

    def transcription(self) -> str:
        """Return this page's text in reading order, one element per line."""
        ordered = sorted(
            (el for el in self.elements if el.is_transcribable),
            key=lambda el: el.reading_order,
        )
        return "\n".join(el.text.strip() for el in ordered)

    def elements_of(self, element_type: ElementType) -> tuple[Element, ...]:
        """Return every element of a given type, in reading order."""
        return tuple(
            sorted(
                (el for el in self.elements if el.type is element_type),
                key=lambda el: el.reading_order,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the page and its elements."""
        return {
            "page_index": self.page_index,
            "width": self.width,
            "height": self.height,
            "image_path": self.image_path,
            "elements": [element.as_dict() for element in self.elements],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Page":
        """Rebuild a page from its serialized form."""
        return cls(
            page_index=int(payload["page_index"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            image_path=str(payload["image_path"]),
            elements=tuple(
                Element.from_dict(item) for item in payload.get("elements", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class DocumentAnnotation:
    """Ground truth for one rendered document, real or synthetic.

    This is the single interchange format between dataset sources and
    training code (CLAUDE.md 5.3). ``prepare_manifest.py`` merges sources by
    reading this and nothing else.
    """

    sample_id: str
    source_sample_id: str
    seed: int
    dpi: int
    page_size: str
    orientation: str
    pages: tuple[Page, ...]
    letter_fields: dict[str, Any] = field(default_factory=dict)
    texture: str | None = None
    full_transcription: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.pages:
            raise AnnotationError(f"{self.sample_id}: annotation has no pages")
        for expected, page in enumerate(self.pages):
            if page.page_index != expected:
                raise AnnotationError(
                    f"{self.sample_id}: page indices must be contiguous from 0, "
                    f"found {page.page_index} at position {expected}"
                )
        if not self.full_transcription:
            object.__setattr__(self, "full_transcription", self.build_transcription())

    def build_transcription(self) -> str:
        """Join every page's transcription with a blank line between pages."""
        return "\n\n".join(page.transcription() for page in self.pages).strip()

    @property
    def element_count(self) -> int:
        """Total number of annotated regions across all pages."""
        return sum(len(page.elements) for page in self.pages)

    def iter_elements(self) -> Sequence[tuple[int, Element]]:
        """Return ``(page_index, element)`` pairs across the whole document."""
        return [
            (page.page_index, element)
            for page in self.pages
            for element in page.elements
        ]

    def as_dict(self) -> dict[str, Any]:
        """Serialize to the JSON shape written under ``annotations/``."""
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "source_sample_id": self.source_sample_id,
            "seed": self.seed,
            "dpi": self.dpi,
            "page_size": self.page_size,
            "orientation": self.orientation,
            "texture": self.texture,
            "letter_fields": self.letter_fields,
            "full_transcription": self.full_transcription,
            "pages": [page.as_dict() for page in self.pages],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentAnnotation":
        """Rebuild an annotation from its serialized form."""
        try:
            return cls(
                sample_id=str(payload["sample_id"]),
                source_sample_id=str(payload["source_sample_id"]),
                seed=int(payload["seed"]),
                dpi=int(payload["dpi"]),
                page_size=str(payload["page_size"]),
                orientation=str(payload["orientation"]),
                pages=tuple(Page.from_dict(item) for item in payload["pages"]),
                letter_fields=dict(payload.get("letter_fields") or {}),
                texture=payload.get("texture"),
                full_transcription=str(payload.get("full_transcription", "")),
                schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            )
        except KeyError as exc:
            raise AnnotationError(f"annotation missing required key: {exc}") from exc
