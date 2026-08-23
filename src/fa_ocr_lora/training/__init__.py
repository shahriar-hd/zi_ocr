"""Local LoRA training for PaddleOCR-VL on a single consumer GPU.

The package is arranged so each stage can be used on its own - the manifest is
readable without a GPU, the encoder is testable without a model, the metrics are
computable from two strings. Three command-line entry points tie them together, and
they are meant to be run in this order:

1. :mod:`~fa_ocr_lora.training.prepare_manifest` - merge the local corpora into one
   manifest with group-aware splits.
2. :mod:`~fa_ocr_lora.training.train_lora` - ``--smoke``, then ``--probe 30``, then
   the real run.
3. :mod:`~fa_ocr_lora.training.eval` - ``--no-adapter`` for the baseline, then again
   for the trained adapter.

Importing this package pulls in :mod:`torch` and :mod:`transformers` by way of the
model and data layers, so the light modules are re-exported here individually rather
than star-imported: reading a manifest should not cost a CUDA initialisation.
"""

from __future__ import annotations

from .config import (
    DataConfig,
    EvalConfig,
    ImageConfig,
    LoraConfig,
    ModelConfig,
    OptimConfig,
    PathsConfig,
    RuntimeConfig,
    TrainingConfig,
    load_training_config,
)
from .manifest import ManifestSummary, build_manifest, digit_density, load_manifest
from .metrics import EvalMetrics, MetricAccumulator
from .sources import SOURCE_TAGS, SourceRecord

__all__ = [
    "SOURCE_TAGS",
    "DataConfig",
    "EvalConfig",
    "EvalMetrics",
    "ImageConfig",
    "LoraConfig",
    "ManifestSummary",
    "MetricAccumulator",
    "ModelConfig",
    "OptimConfig",
    "PathsConfig",
    "RuntimeConfig",
    "SourceRecord",
    "TrainingConfig",
    "build_manifest",
    "digit_density",
    "load_manifest",
    "load_training_config",
]
