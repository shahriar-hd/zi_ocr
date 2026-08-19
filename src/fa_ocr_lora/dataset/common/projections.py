"""Trainer-facing projections of an annotation.

:class:`~fa_ocr_lora.dataset.common.schema.DocumentAnnotation` is the source of
truth, and it is shaped for *completeness*: every region, every box, every
structured payload. No trainer reads that shape. Each one wants a flat record
per page, and each wants a different subset - PaddleOCR-VL wants the boxes,
PaddleFarmer wants only the text.

The two functions here are those subsets. Both are deliberately lossy and both
are pure functions of the annotation, so the JSONL files they feed can be
deleted and regenerated at any time without re-rendering a page.

They live in ``common/`` rather than beside the pipeline that first wrote them
because two callers now need them: :mod:`~fa_ocr_lora.dataset.synth.pipeline`
appends them to its JSONL files, and the HuggingFace exporter folds them into
the ``annotations`` family. The pipeline imports the exporter, so the exporter
cannot import the pipeline - this module is the shared floor beneath both.
"""

from __future__ import annotations

from typing import Any

from .schema import DocumentAnnotation

__all__ = ["paddle_farmer_records", "paddle_records"]


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
    :class:`~.schema.DocumentAnnotation` - deliberately lossy, keeping only
    what the trainer reads - so the annotation JSON stays the source of truth
    and this file can be regenerated from it at any time.

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
