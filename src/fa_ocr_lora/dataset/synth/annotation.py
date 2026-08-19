"""Ground truth: Chromium's DOM rects turned into a :class:`DocumentAnnotation`.

The browser hands back one flat list of boxes in CSS pixels. Four things have
to happen before that list is training data:

1. **Scale and clip.** Boxes are measured in CSS pixels and the images are
   rasterised at ``dpi / 96``, so every coordinate is multiplied by the same
   factor the screenshot used. A box that then sticks a pixel past the paper
   edge - a border rounding outwards - is clipped, because a box outside the
   image is a box the model can never learn.

2. **Nest.** The flat list is turned into a containment forest: a table's cells
   sit inside the table, the metadata fields inside their block, the chart's
   bars inside the figure. This is what makes the reading order come out right
   and it is what identifies containers.

3. **Order, right to left.** Boxes are grouped into visual lines by their
   vertical centre, then read from the right edge leftwards - Persian reading
   order. Containers are emitted before their contents, so a table appears in
   the stream immediately before its own cells.

4. **De-duplicate the text.** A container's text is, by construction, its
   children's text concatenated: transcribing both would teach the model to
   emit every table twice. Only leaves keep their text; containers keep their
   structured content, which is where a table's cell matrix lives anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Final, Iterable, Mapping, Protocol, Sequence

from ...config import CSS_PPI
from ..common.schema import (
    AnnotationError,
    BBox,
    DocumentAnnotation,
    Element,
    ElementType,
    NON_TEXT_TYPES,
    Page,
)
from .layout.page_composer import ComposedLetter
from .render.browser import RenderResult

__all__ = [
    "BoxMap",
    "build_annotation",
    "build_pages",
    "reading_order",
    "remap_boxes",
]

#: Two boxes belong to the same visual line when their vertical centres sit
#: within this fraction of the shorter box's height. Persian body text at 12 pt
#: is ~40 device pixels tall at 300 DPI, so 0.6 tolerates the baseline shift
#: between a numeral and a descender without merging adjacent lines.
_LINE_TOLERANCE: Final[float] = 0.6

#: Absolute floor for the line tolerance, in device pixels, so a very short box
#: - a bare digit - is not given a near-zero band of its own.
_LINE_TOLERANCE_FLOOR: Final[float] = 6.0

#: A box counts as contained by another when at least this much of its area
#: falls inside. Not 1.0: an inline node can overhang its parent by a fraction
#: of a pixel through subpixel layout, and a table cell's border is shared with
#: the table's own.
_CONTAINMENT_RATIO: Final[float] = 0.86


@dataclass(slots=True)
class _Node:
    """One box plus its place in the containment forest."""

    element_id: str
    type: ElementType
    bbox: BBox
    text: str
    content: dict[str, Any]
    children: list["_Node"] = field(default_factory=list)

    @property
    def is_container(self) -> bool:
        """Return whether this node encloses other annotated nodes."""
        return bool(self.children)


def _overlap_ratio(inner: BBox, outer: BBox) -> float:
    """Return how much of ``inner``'s area falls inside ``outer``.

    Args:
        inner: The candidate child.
        outer: The candidate parent.

    Returns:
        A ratio in ``[0, 1]``; zero when the boxes are disjoint.
    """
    width = min(inner.x1, outer.x1) - max(inner.x0, outer.x0)
    height = min(inner.y1, outer.y1) - max(inner.y0, outer.y0)
    if width <= 0 or height <= 0:
        return 0.0
    return (width * height) / float(inner.area)


def _resolve_type(raw: str) -> ElementType | None:
    """Map a ``data-ann-type`` string onto the schema's enum.

    Args:
        raw: The attribute value as written in the template fragments.

    Returns:
        The element type, or ``None`` when the attribute names a type the
        schema does not know - which is a template bug, and is skipped rather
        than allowed to abort a 20k-sample run.
    """
    try:
        return ElementType(raw)
    except ValueError:
        return None


def _merged_content(
    box: Mapping[str, Any],
    element_id: str,
    content_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine the composer's payload with whatever the DOM carried.

    Three sources can describe one element: the composer's ``content_map``
    (the authoritative spec, e.g. a table's full cell matrix), the node's
    ``data-ann-content`` JSON, and its remaining ``data-ann-*`` attributes
    (a cell's row and column). They are merged in that order of increasing
    specificity, so a per-node attribute wins over a block-level payload.

    Args:
        box: One harvested box from ``paginate.js``.
        element_id: The box's ``data-ann-id``.
        content_map: :attr:`ComposedLetter.content_map`.

    Returns:
        A fresh dictionary, empty when nothing described this element.
    """
    content: dict[str, Any] = {}
    spec_payload = content_map.get(element_id)
    if spec_payload:
        content.update(spec_payload)
    dom_payload = box.get("content")
    if isinstance(dom_payload, Mapping):
        content.update(dom_payload)
    attrs = box.get("attrs")
    if isinstance(attrs, Mapping):
        for key, value in attrs.items():
            if key in {"id", "type", "content"}:
                continue
            content[key] = value
    return content


def _node_from_box(
    box: Mapping[str, Any],
    *,
    scale: float,
    width: int,
    height: int,
    content_map: Mapping[str, Mapping[str, Any]],
) -> _Node | None:
    """Turn one harvested CSS-pixel box into a device-pixel node.

    Args:
        box: One entry from :attr:`RenderResult.boxes`.
        scale: CSS-pixel to device-pixel factor, ``dpi / 96``.
        width: The sheet's raster width, for clipping.
        height: The sheet's raster height, for clipping.
        content_map: Structured payloads keyed by element id.

    Returns:
        The node, or ``None`` when the box names an unknown type, carries no
        id, or collapsed to nothing once rounded to whole pixels.
    """
    element_id = str(box.get("id") or "").strip()
    raw_type = str(box.get("type") or "").strip()
    if not element_id or not raw_type:
        return None
    element_type = _resolve_type(raw_type)
    if element_type is None:
        return None

    rect = box.get("rect")
    if not isinstance(rect, Sequence) or len(rect) != 4:
        return None
    try:
        scaled = BBox.from_iterable(float(value) * scale for value in rect)
    except (AnnotationError, TypeError, ValueError):
        # A sub-pixel sliver rounds to zero extent. It is not annotatable, but
        # it is also not a failure - a hairline rule legitimately measures
        # 0.4 device pixels tall.
        return None
    bbox = scaled.clipped_to(width, height)

    text = "" if element_type in NON_TEXT_TYPES else str(box.get("text") or "").strip()
    return _Node(
        element_id=element_id,
        type=element_type,
        bbox=bbox,
        text=text,
        content=_merged_content(box, element_id, content_map),
    )


def _deduplicate(nodes: Sequence[_Node]) -> list[_Node]:
    """Drop repeated ids, keeping the largest box for each.

    Ids are meant to be unique per page and the spec modules make them so, but
    a paginated block that Chromium split across sheets, or a template edit
    that reuses an id, must not reach :class:`Page` - which rejects duplicates
    outright and would fail the whole sample.

    Args:
        nodes: Nodes harvested from one sheet.

    Returns:
        The surviving nodes, in their original order.
    """
    best: dict[str, _Node] = {}
    order: list[str] = []
    for node in nodes:
        existing = best.get(node.element_id)
        if existing is None:
            best[node.element_id] = node
            order.append(node.element_id)
        elif node.bbox.area > existing.bbox.area:
            best[node.element_id] = node
    return [best[element_id] for element_id in order]


def _nest(nodes: Sequence[_Node]) -> list[_Node]:
    """Build the containment forest and return its roots.

    Boxes are sorted largest first so that when a node is placed, every
    possible parent has already been placed. The parent chosen is the smallest
    box that contains it, which is what makes a table cell land in its own
    table rather than in the page's outermost region.

    Args:
        nodes: Deduplicated nodes from one sheet.

    Returns:
        The top-level nodes; children hang off :attr:`_Node.children`.
    """
    ranked = sorted(nodes, key=lambda node: node.bbox.area, reverse=True)
    roots: list[_Node] = []
    placed: list[_Node] = []
    for node in ranked:
        parent: _Node | None = None
        for candidate in placed:
            if candidate.bbox.area <= node.bbox.area:
                continue
            if _overlap_ratio(node.bbox, candidate.bbox) < _CONTAINMENT_RATIO:
                continue
            if parent is None or candidate.bbox.area < parent.bbox.area:
                parent = candidate
        if parent is None:
            roots.append(node)
        else:
            parent.children.append(node)
        placed.append(node)
    return roots


def _group_lines(nodes: Sequence[_Node]) -> list[list[_Node]]:
    """Group sibling nodes into visual lines, top to bottom.

    Args:
        nodes: Siblings sharing one parent.

    Returns:
        One list per line, each line in arbitrary horizontal order.
    """
    lines: list[list[_Node]] = []
    for node in sorted(nodes, key=lambda item: (item.bbox.center_y, -item.bbox.x1)):
        if lines:
            current = lines[-1]
            reference = min(current, key=lambda item: item.bbox.height)
            tolerance = max(
                _LINE_TOLERANCE_FLOOR,
                _LINE_TOLERANCE * min(reference.bbox.height, node.bbox.height),
            )
            if abs(node.bbox.center_y - reference.bbox.center_y) <= tolerance:
                current.append(node)
                continue
        lines.append([node])
    return lines


def _walk(nodes: Sequence[_Node]) -> Iterable[_Node]:
    """Yield nodes in Persian reading order, containers before their contents.

    Args:
        nodes: Siblings sharing one parent.

    Yields:
        Each node, then recursively its children.
    """
    for line in _group_lines(nodes):
        for node in sorted(line, key=lambda item: (-item.bbox.x1, item.bbox.y0)):
            yield node
            if node.children:
                yield from _walk(node.children)


def reading_order(nodes: Sequence[_Node]) -> list[_Node]:
    """Return one sheet's nodes flattened into reading order.

    Exposed for the tests, which assert the right-to-left rule directly rather
    than inferring it from a rendered page.

    Args:
        nodes: Nodes harvested from one sheet, un-nested.

    Returns:
        The same nodes, nested and flattened in reading order.
    """
    return list(_walk(_nest(_deduplicate(nodes))))


def _to_elements(roots: Sequence[_Node]) -> tuple[Element, ...]:
    """Flatten a containment forest into ordered schema elements.

    The container-text rule is applied here: a node with children has its own
    ``text`` cleared, because that text is exactly its children's text joined
    and transcribing both would teach the model to emit every table, metadata
    block and attachment list twice. The structured ``content`` survives, which
    is where a table's cell matrix and a chart's series live anyway.

    Args:
        roots: Top-level nodes for one sheet.

    Returns:
        Elements with ``reading_order`` assigned from the walk.
    """
    elements: list[Element] = []
    for index, node in enumerate(_walk(roots)):
        elements.append(
            Element(
                id=node.element_id,
                type=node.type,
                bbox=node.bbox,
                text="" if node.is_container else node.text,
                reading_order=index,
                content=dict(node.content),
            )
        )
    return tuple(elements)


def build_pages(
    result: RenderResult,
    *,
    image_names: Sequence[str],
    content_map: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Page, ...]:
    """Turn one render's boxes into per-sheet annotated pages.

    Args:
        result: What :meth:`.render.browser.PageRenderer.render` returned.
        image_names: One image filename per sheet, as written to disk. Stored
            verbatim so the annotation stays valid after the files move, as
            long as they move together.
        content_map: Structured payloads keyed by element id, normally
            :attr:`.layout.page_composer.ComposedLetter.content_map`.

    Returns:
        One :class:`Page` per rasterised sheet, in document order.

    Raises:
        AnnotationError: If the number of image names does not match the
            number of sheets - a mismatch there would silently point half the
            annotations at the wrong picture.
    """
    if len(image_names) != result.page_count:
        raise AnnotationError(
            f"expected {result.page_count} image names, got {len(image_names)}"
        )

    payloads = content_map or {}
    grouped: dict[int, list[_Node]] = {index: [] for index in range(result.page_count)}
    for box in result.boxes:
        sheet_index = int(box.get("sheet", -1))
        sheet_nodes = grouped.get(sheet_index)
        if sheet_nodes is None:
            # A box measured against a sheet that was never rasterised - only
            # reachable if pagination and screenshotting disagreed.
            continue
        sheet = result.sheets[sheet_index]
        node = _node_from_box(
            box,
            scale=result.scale,
            width=sheet.width,
            height=sheet.height,
            content_map=payloads,
        )
        if node is not None:
            sheet_nodes.append(node)

    pages: list[Page] = []
    for index, sheet in enumerate(result.sheets):
        roots = _nest(_deduplicate(grouped[index]))
        pages.append(
            Page(
                page_index=index,
                width=sheet.width,
                height=sheet.height,
                image_path=image_names[index],
                elements=_to_elements(roots),
            )
        )
    return tuple(pages)


def build_annotation(
    composed: ComposedLetter,
    result: RenderResult,
    *,
    image_names: Sequence[str],
) -> DocumentAnnotation:
    """Assemble the ground truth for one rendered letter.

    Args:
        composed: The composed letter, for its seed, paper and content map.
        result: The Chromium pass that rasterised it.
        image_names: One image filename per page, in order.

    Returns:
        The complete annotation, with ``full_transcription`` built from the
        pages' leaf text in reading order.

    Raises:
        AnnotationError: If the boxes cannot be reconciled with the sheets, or
            an assembled page violates the shared schema.
    """
    pages = build_pages(
        result,
        image_names=image_names,
        content_map=composed.content_map,
    )
    letter_fields = dict(composed.source.letter_fields)
    letter_fields["overflowed"] = result.overflowed
    letter_fields["missing_assets"] = list(composed.source.missing_assets)
    return DocumentAnnotation(
        sample_id=composed.sample_id,
        source_sample_id=composed.source.sample_id,
        seed=composed.seed,
        dpi=int(round(result.scale * CSS_PPI)),
        page_size=composed.page_spec.name,
        orientation=composed.page_spec.orientation,
        pages=pages,
        letter_fields=letter_fields,
        texture=composed.texture_name,
    )


class BoxMap(Protocol):
    """Anything that can move a box from the rendered page to the scanned one.

    :class:`.render.scan.SheetGeometry` is the implementation. It is taken
    structurally rather than imported so this module stays free of the scan
    stage's numpy and Pillow dependencies - the ground truth has to be
    buildable on a machine that only reads annotations.
    """

    def map_bbox(self, bbox: Sequence[float]) -> tuple[int, int, int, int]:
        """Return the box's axis-aligned extent after the transform."""
        ...


def remap_boxes(
    annotation: DocumentAnnotation,
    geometries: Sequence[BoxMap | None],
    *,
    scan: Mapping[str, Any] | None = None,
) -> DocumentAnnotation:
    """Move every box through the geometry its own sheet was scanned with.

    The scan stage rotates, warps and bows each sheet *after* the boxes were
    measured, so an un-remapped annotation describes a page that no longer
    exists. Reading order is deliberately left as it was: it was computed on
    the un-rotated page, where the lines are exactly horizontal, and a
    sub-degree rotation can only make that grouping worse.

    A box that the transform pushes off the page collapses to zero extent and
    is dropped rather than clamped to a sliver, since a one-pixel box at the
    paper edge is not something worth teaching.

    Args:
        annotation: The ground truth as measured on the clean render.
        geometries: One map per page, in page order; ``None`` for a page that
            was not transformed. Shorter sequences leave the remaining pages
            untouched.
        scan: The scan provenance to record on the annotation, normally
            :meth:`.render.scan.ScanProfile.as_content`.

    Returns:
        A new annotation. The original is left alone, as are its pages when
        nothing about them changed.
    """
    pages: list[Page] = []
    for index, page in enumerate(annotation.pages):
        geometry = geometries[index] if index < len(geometries) else None
        if geometry is None:
            pages.append(page)
            continue
        elements: list[Element] = []
        for element in page.elements:
            x0, y0, x1, y1 = geometry.map_bbox(element.bbox.as_list())
            if x1 <= x0 or y1 <= y0:
                continue
            elements.append(replace(element, bbox=BBox(x0, y0, x1, y1)))
        pages.append(replace(page, elements=tuple(elements)))
    return replace(
        annotation,
        pages=tuple(pages),
        scan=dict(scan) if scan is not None else annotation.scan,
    )
