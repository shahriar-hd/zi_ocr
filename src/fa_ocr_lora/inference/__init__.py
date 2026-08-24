"""Read Persian documents with the trained LoRA adapter.

Phase 3 of CLAUDE.md: the thin layer that turns the adapter produced by
:mod:`fa_ocr_lora.training` into something usable on a document nobody has
transcribed. Two entry points, one library and one shell:

    >>> from fa_ocr_lora.inference import open_reader, read_document
    >>> reader = open_reader()
    >>> reading = read_document(reader, Path("scan.png"))
    >>> reading.text[:40]

    $ fa-ocr-lora predict --image scan.png --json

Everything here reuses the training package rather than re-implementing it - the
same prompt template, the same pixel budget, the same greedy decode, the same
sensitive-field rules. That is deliberate and load-bearing: a prompt rendered
twice drifts, and the drift shows up as a production character error rate worse
than the published one with nothing in the code to point at.
"""

from __future__ import annotations

from .loader import DocumentReader, load_reader, open_reader, resolve_adapter
from .predict import READING_SCHEMA_VERSION, PageReading, read_document

__all__ = [
    "READING_SCHEMA_VERSION",
    "DocumentReader",
    "PageReading",
    "load_reader",
    "open_reader",
    "read_document",
    "resolve_adapter",
]
