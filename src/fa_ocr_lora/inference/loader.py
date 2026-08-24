"""Load the base model, apply the trained adapter, and hold what reading needs.

Phase 3's job is narrow: put the artifact the training run produced behind an API
that reads an arbitrary document. Almost nothing here is new machinery, and that
is deliberate - every component is the one evaluation already used:

===========================  ===========================================
``load_eval_model``          base weights, quantisation, adapter, cache
``SampleEncoder``            the prompt, rendered from the model's own
                             chat template
``load_field_validation_config``  the sensitive-field rules
===========================  ===========================================

Reusing them is the whole correctness argument. An inference path that rendered
its own prompt, or picked its own resolution, or applied its own field patterns
would produce numbers that no longer relate to the character error rate the eval
report published - and the drift would be invisible, because both paths would
still return plausible Persian text.

The one thing this module adds is the base-model fallback CLAUDE.md 7 asks for:
:func:`resolve_adapter` decides whether an adapter is applied at all, so the same
CLI can read a document with the adapter and without it and the difference can be
seen rather than assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..dataset.common.io_utils import resolve_repo_path
from ..training.config import TrainingConfig, load_training_config
from ..training.encoding import SampleEncoder
from ..training.modeling import load_eval_model
from ..validation import FieldValidationConfig, load_field_validation_config

__all__ = [
    "DocumentReader",
    "load_reader",
    "open_reader",
    "resolve_adapter",
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DocumentReader:
    """A loaded model with everything one document needs to be read.

    Held together in one object because the four parts must agree and there is no
    way to check that they do after the fact: the encoder's prompt has to be the
    one this checkpoint's template renders, the pixel budget has to be the one the
    adapter was trained at, and the field rules have to be the ones the eval report
    quoted. Passing them around separately invites three of the four to be right.

    Attributes:
        model: The base model, adapter applied unless this is the baseline.
        processor: The base model's ``AutoProcessor``.
        encoder: Prompt and image encoder, shared with the training path.
        config: The loaded training configuration this was assembled from.
        rules: Sensitive-field rules, or ``None`` when the audit is off.
        dtype: Resolved compute dtype.
        quantized: Whether 4-bit quantisation is active.
        weight_mib: Measured device memory held by weights, in MiB.
        adapter_dir: Adapter that was applied, or ``None`` for the base model.
    """

    model: nn.Module
    processor: Any
    encoder: SampleEncoder
    config: TrainingConfig
    rules: FieldValidationConfig | None
    dtype: torch.dtype
    quantized: bool
    weight_mib: float
    adapter_dir: Path | None = None

    @property
    def device(self) -> torch.device:
        """Device the weights live on.

        Returns:
            The device of the first parameter, which is where generation inputs
            must be placed.
        """
        return next(self.model.parameters()).device

    def describe(self) -> str:
        """One-line summary for a startup banner.

        Returns:
            A human-readable description of what was loaded.
        """
        precision = "nf4" if self.quantized else str(self.dtype).replace("torch.", "")
        which = self.adapter_dir.name if self.adapter_dir else "base model only"
        audit = "on" if self.rules is not None else "off"
        return (
            f"{precision}, weights {self.weight_mib:.0f} MiB, adapter: {which}, "
            f"field audit: {audit}"
        )


def resolve_adapter(
    config: TrainingConfig,
    *,
    adapter_dir: Path | None = None,
    use_adapter: bool = True,
) -> Path | None:
    """Decide which adapter, if any, to apply.

    Args:
        config: Loaded training configuration, supplying the default location.
        adapter_dir: Explicit adapter directory, or ``None`` for the default.
        use_adapter: ``False`` reads with the untrained base model, which is the
            fallback CLAUDE.md 7 requires and the only way to see what the adapter
            is contributing on a given document.

    Returns:
        The adapter directory, or ``None`` for base-model-only reading.
    """
    if not use_adapter:
        return None
    if adapter_dir is not None:
        return resolve_repo_path(adapter_dir)
    return config.paths.adapter_dir


def load_reader(
    config: TrainingConfig, *, adapter_dir: Path | None = None
) -> DocumentReader:
    """Assemble a reader from an already-loaded configuration.

    Args:
        config: Loaded training configuration. Taken loaded rather than as a path
            so a caller that has applied command-line overrides gets those
            overrides honoured here, validated by the same loader that validated
            the YAML.
        adapter_dir: Adapter to apply, or ``None`` to read with the base model.

    Returns:
        The assembled reader.

    Raises:
        FileNotFoundError: If ``adapter_dir`` holds no adapter.
        ConfigError: If the sensitive-field rules cannot be loaded. Raised rather
            than warned past, for the reason evaluation raises it: an audit that
            silently stops checking a field reports no faults for it, which reads
            as a clean document.
    """
    loaded = load_eval_model(config, adapter_dir=adapter_dir)
    encoder = SampleEncoder(loaded.processor, model=config.model, image=config.image)
    rules: FieldValidationConfig | None = None
    if config.eval.field_audit:
        rules = load_field_validation_config(config.eval.field_config_path)
        LOGGER.info(
            "field audit on: %s (%d field(s))", rules.source_path, len(rules.specs)
        )
    return DocumentReader(
        model=loaded.model,
        processor=loaded.processor,
        encoder=encoder,
        config=config,
        rules=rules,
        dtype=loaded.dtype,
        quantized=loaded.quantized,
        weight_mib=loaded.weight_mib,
        adapter_dir=loaded.adapter_dir,
    )


def open_reader(
    config_path: Path | None = None,
    *,
    adapter_dir: Path | None = None,
    use_adapter: bool = True,
) -> DocumentReader:
    """Load a reader in one call, for library callers.

    Loading the base weights takes tens of seconds and holds ~1.2 GiB of VRAM, so
    build one reader and pass it to
    :func:`~fa_ocr_lora.inference.predict.read_document` for every page. Calling
    this per document would reload the checkpoint each time.

    Args:
        config_path: Training YAML; ``None`` uses ``configs/lora_config.yaml``.
        adapter_dir: Adapter to apply; ``None`` uses ``<output_dir>/adapter_only``.
        use_adapter: ``False`` reads with the untrained base model.

    Returns:
        The assembled reader.

    Raises:
        FileNotFoundError: If the configuration or the adapter is missing.
        ConfigError: If the configuration or the field rules are invalid.
    """
    config = load_training_config(config_path)
    return load_reader(
        config,
        adapter_dir=resolve_adapter(
            config, adapter_dir=adapter_dir, use_adapter=use_adapter
        ),
    )
