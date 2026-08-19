"""Exporters that repackage a rendered corpus for a downstream consumer.

The render pipeline writes what is convenient to *produce*: one image, one PDF,
one Markdown file and one annotation JSON per sample, plus two JSONL
projections. That tree is good for inspecting a sample and bad for shipping
20k of them - a directory of 40k loose files is slow to copy, slow to upload,
and cannot be streamed.

This package holds the other direction: readers that walk the rendered tree and
emit a form built for transport. Every exporter here reads the *written*
artifacts rather than in-memory pipeline state, so an export can be re-run,
resumed, or pointed at a corpus rendered weeks ago.

One exporter, in four layers:

* :mod:`~fa_ocr_lora.dataset.export.splits` - :class:`SplitPlan`, the
  group-aware train/test assignment. Hashes the *source letter id*, so every
  page of one letter lands in one split and adding plaintext later cannot
  re-deal the samples already exported.
* :mod:`~fa_ocr_lora.dataset.export.families` - :data:`FAMILIES`, one
  :class:`FamilySpec` per artifact kind. Each declares its columns once, and
  both the Arrow schema and the dataset card's documentation derive from that
  declaration.
* :mod:`~fa_ocr_lora.dataset.export.attribution` - the licensing facts the
  dataset card is obliged to state, and the republished logo licence manifest.
* :mod:`~fa_ocr_lora.dataset.export.huggingface` - the writers: sharded Parquet
  per family and split, the sidecar indexes, and the card the Hub reads.
* :mod:`~fa_ocr_lora.dataset.export.upload` - pushing that finished tree to a
  Hub dataset repository, one commit per family. A separate step on purpose: a
  40 GB upload should be retryable without re-reading a single page image.

The layering is strictly one-way - ``errors`` before ``splits``/``families``/
``attribution`` before ``huggingface`` before ``upload`` - and nothing here
imports the render pipeline, which imports this package.
"""

from __future__ import annotations

from .attribution import LogoAttribution, write_logo_attribution
from .errors import ExportError
from .families import FAMILIES, Column, FamilySpec, SampleBundle, load_bundle
from .huggingface import (
    ExportSummary,
    ExportWriter,
    ShardWriter,
    export_dataset,
    iter_sample_ids,
    iter_shard_paths,
)
from .splits import SplitPlan
from .upload import (
    SIDECAR_UNIT,
    UPLOAD_MODES,
    UploadPlan,
    UploadSummary,
    UploadUnit,
    plan_upload,
    resolve_token,
    token_source,
    upload_export,
)

__all__ = [
    "Column",
    "ExportError",
    "ExportSummary",
    "ExportWriter",
    "FAMILIES",
    "FamilySpec",
    "LogoAttribution",
    "SIDECAR_UNIT",
    "SampleBundle",
    "ShardWriter",
    "SplitPlan",
    "UPLOAD_MODES",
    "UploadPlan",
    "UploadSummary",
    "UploadUnit",
    "export_dataset",
    "iter_sample_ids",
    "iter_shard_paths",
    "load_bundle",
    "plan_upload",
    "resolve_token",
    "token_source",
    "upload_export",
    "write_logo_attribution",
]
