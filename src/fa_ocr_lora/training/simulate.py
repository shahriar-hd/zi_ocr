"""Project the cost of a training run before renting the machine to do it.

This module answers one question: *if this configuration were run on that GPU,
what would it consume and how long would it take?* It loads no model, allocates
no CUDA memory, and reads no image - it works from the manifest arithmetic the
rest of the package already performs, plus a parameter census of the base
checkpoint's safetensors header.

The distinction that makes the output worth reading is **calibration**. Three
kinds of number appear below and they are not equally trustworthy:

* **Measured** - parameter counts from the checkpoint header, image token counts
  from :func:`~fa_ocr_lora.training.images.fit_pixel_budget` over real corpus
  dimensions, and the wall-clock of the completed 4 GB run. These are facts.
* **Calibrated** - throughput, derived by back-solving the achieved FLOPS of that
  completed run and re-projecting it onto different hardware. A projection that
  cannot reproduce the one run we actually have is not worth believing, so
  :func:`calibrate` reports the residual and the report prints it.
* **Modelled** - VRAM and RAM, built component by component from tensor shapes.
  Reasoned, but never validated above batch 1 on this codebase.

Every figure the report prints is tagged with which of the three it is. A
simulation that hides that distinction is just a confident guess.

Run it with::

    fa-ocr-simulate --config configs/lora_config_a100.yaml --preset a100_40gb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..config.base import ConfigError
from ..dataset.common.io_utils import write_json
from .config import PIXELS_PER_TOKEN, TrainingConfig, load_training_config
from .images import fit_pixel_budget, image_tokens
from .manifest import _MAX_HOLDOUT_SHARE, _allocate
from .sources import (
    LETTER_CROP,
    LETTER_PAGE,
    PIXEL_PAGE,
    SkipLog,
    iter_letter_crops,
    iter_letter_pages,
    iter_pixel_rows,
)

__all__ = [
    "BASE_PARAMETERS",
    "Hardware",
    "HARDWARE_PRESETS",
    "SourceProfile",
    "CorpusProfile",
    "MemoryModel",
    "ThroughputModel",
    "Simulation",
    "calibrate",
    "profile_corpus",
    "simulate",
    "main",
]

LOGGER = logging.getLogger(__name__)

MIB: Final[float] = 1024.0**2
GIB: Final[float] = 1024.0**3

# --------------------------------------------------------------------------
# Facts about the base checkpoint.
#
# Read once from the safetensors header of revision 2b77538ef936 - the header is
# JSON at the top of the file, so shapes and dtypes are exact without loading a
# single weight. Recorded as constants because they describe a pinned revision:
# if the revision moves, these must be re-measured, which is why the pin exists.
# --------------------------------------------------------------------------

#: Total parameters in the base model, all bfloat16.
BASE_PARAMETERS: Final[int] = 958_588_736

#: Parameters in the frozen vision tower and projector (``visual`` + ``mlp_AR``).
VISION_PARAMETERS: Final[int] = 491_934_528

#: Bytes per parameter at bf16.
BF16_BYTES: Final[int] = 2

#: Decoder geometry, from ``config.json`` of the pinned revision.
DECODER_LAYERS: Final[int] = 18
DECODER_HIDDEN: Final[int] = 1024
DECODER_INTERMEDIATE: Final[int] = 3072
DECODER_HEADS: Final[int] = 16
DECODER_KV_HEADS: Final[int] = 2
DECODER_HEAD_DIM: Final[int] = 128
VOCAB_SIZE: Final[int] = 103_424

#: Vision tower geometry.
VISION_LAYERS: Final[int] = 27
VISION_HIDDEN: Final[int] = 1152
VISION_INTERMEDIATE: Final[int] = 4304
#: Attention heads in the tower. Only the *memory* model needs this: a score
#: matrix is ``[heads, patches, patches]``, so the head count multiplies it, while
#: the FLOP count of ``QK^T`` over all heads is head-count-independent.
VISION_HEADS: Final[int] = 16

#: Bytes an AdamW moment pair costs per trainable parameter, at fp32.
ADAM_STATE_BYTES: Final[int] = 8

#: CUDA context, cuBLAS workspaces and the allocator's own bookkeeping. Measured
#: as the gap between torch's reported peak and nvidia-smi on the 4 GB card.
CUDA_OVERHEAD_MIB: Final[float] = 800.0

#: The caching allocator does not return every freed block to a perfectly packed
#: heap. Applied to the sum of the dynamic components, not to the weights.
FRAGMENTATION: Final[float] = 1.10


@dataclass(frozen=True, slots=True)
class Hardware:
    """A machine the run might be placed on.

    Attributes:
        name: Human-readable label.
        vram_gib: Usable device memory. For a rented card this is what the
            provider exposes, which can be less than the silicon's nominal size.
        bf16_tflops: Dense bf16 matmul throughput, vendor peak. Never achieved;
            :class:`ThroughputModel` applies an efficiency factor to it.
        vcpus: Host cores available to the dataloader.
        ram_gib: Host memory.
        storage_gib: Persistent disk.
        pcie_gbps: Host-to-device bandwidth, for the input-pipeline check.
    """

    name: str
    vram_gib: float
    bf16_tflops: float
    vcpus: int
    ram_gib: float
    storage_gib: float
    pcie_gbps: float = 25.0


#: Machines this module knows how to project onto.
#:
#: The A100 entry describes the rented configuration this simulation was written
#: for: 40 GB of VRAM exposed, 32 vCPU, 64 GB RAM, 200 GB disk. Its 312 TFLOPS is
#: the vendor's dense bf16 tensor-core peak.
HARDWARE_PRESETS: Final[dict[str, Hardware]] = {
    "a100_40gb": Hardware(
        name="A100 40 GB (rented: 32 vCPU / 64 GB RAM / 200 GB disk)",
        vram_gib=40.0,
        bf16_tflops=312.0,
        vcpus=32,
        ram_gib=64.0,
        storage_gib=200.0,
    ),
    "rtx_3050ti": Hardware(
        name="RTX 3050 Ti Laptop 4 GB (the development card)",
        vram_gib=4.0,
        bf16_tflops=30.4,
        vcpus=8,
        ram_gib=14.0,
        storage_gib=21.0,
        pcie_gbps=8.0,
    ),
}


@dataclass(frozen=True, slots=True)
class SourceProfile:
    """Token and geometry statistics for one corpus source.

    Every field is an average over the records the manifest would actually draw,
    computed from real page dimensions and real text lengths - not a guess at a
    representative sample.

    Attributes:
        source: Manifest source tag.
        records: How many records the configuration draws from this source.
        image_tokens: Mean visual tokens per record, after the sequence budget.
        answer_tokens: Mean supervised tokens per record.
        seq_len: Mean total sequence length.
        seq_len_max: Longest sequence this source produces.
        patches: Mean pre-merge 14 px patches (``image_tokens * 4``).
        patches_sq: Mean of *squared* patch count. Vision attention is quadratic
            in patches, so its expected cost needs ``E[p^2]``, not ``E[p]^2`` -
            and for a bimodal source like the letter pages those differ enough to
            matter.
        patches_max: Largest patch count, which sets peak activation memory.
        bytes_on_disk: Mean size of the source image file.
    """

    source: str
    records: int
    image_tokens: float
    answer_tokens: float
    seq_len: float
    seq_len_max: int
    patches: float
    patches_sq: float
    patches_max: int
    bytes_on_disk: float = 0.0


@dataclass(frozen=True, slots=True)
class CorpusProfile:
    """The whole training mix, as the numbers a cost model needs.

    Attributes:
        sources: One profile per source, in manifest order.
        train_records: Records in the training split, after the holdout.
        validation_records: Records held out for evaluation.
        test_records: Records reserved for a final untouched score.
        oversample_factor: Ratio of samples seen per epoch to training records,
            once digit-dense records are repeated.
    """

    sources: tuple[SourceProfile, ...]
    train_records: int
    validation_records: int
    test_records: int
    oversample_factor: float

    @property
    def total_records(self) -> int:
        """Records drawn from all sources before splitting."""
        return sum(source.records for source in self.sources)

    @property
    def epoch_samples(self) -> int:
        """Samples the loop sees in one epoch, oversampling included."""
        return int(round(self.train_records * self.oversample_factor))

    def _weighted(self, attribute: str) -> float:
        """Mean of one attribute, weighted by record count."""
        total = self.total_records
        if total <= 0:
            return 0.0
        return sum(getattr(s, attribute) * s.records for s in self.sources) / total

    @property
    def mean_seq_len(self) -> float:
        """Mean sequence length across the mix."""
        return self._weighted("seq_len")

    @property
    def mean_patches(self) -> float:
        """Mean patch count across the mix."""
        return self._weighted("patches")

    @property
    def mean_patches_sq(self) -> float:
        """Mean squared patch count across the mix."""
        return self._weighted("patches_sq")

    @property
    def mean_answer_tokens(self) -> float:
        """Mean supervised tokens across the mix."""
        return self._weighted("answer_tokens")


# --------------------------------------------------------------------------
# FLOPs. Each function returns forward FLOPs for one sample; the backward
# multipliers are applied in `sample_flops`, because they differ per component.
# --------------------------------------------------------------------------

#: Parameters in the ``mlp_AR`` projector that maps merged visual features into
#: the decoder's width. From the same header census as :data:`BASE_PARAMETERS`.
PROJECTOR_PARAMETERS: Final[int] = 25_960_192

#: ``2 * (4*H^2 + 2*H*I) * layers`` for the vision tower: QKV+O projections and
#: the two MLP matmuls, per patch.
_VISION_LINEAR: Final[int] = (
    VISION_LAYERS * 2 * (4 * VISION_HIDDEN**2 + 2 * VISION_HIDDEN * VISION_INTERMEDIATE)
)

#: ``4 * H * layers``: QK^T and AV, both quadratic in patch count. The vision
#: tower is bidirectional, so unlike the decoder there is no causal halving -
#: which is why raising ``max_pixels`` costs more than its area ratio suggests.
_VISION_ATTENTION: Final[int] = VISION_LAYERS * 4 * VISION_HIDDEN

_Q_WIDTH: Final[int] = DECODER_HEADS * DECODER_HEAD_DIM
_KV_WIDTH: Final[int] = DECODER_KV_HEADS * DECODER_HEAD_DIM

#: Per-token decoder linear FLOPs: q/k/v/o plus gate/up/down, all layers. The GQA
#: k and v projections are 8x narrower than q, which this accounts for.
_DECODER_LINEAR: Final[int] = (
    DECODER_LAYERS
    * 2
    * (
        DECODER_HIDDEN * _Q_WIDTH
        + 2 * DECODER_HIDDEN * _KV_WIDTH
        + _Q_WIDTH * DECODER_HIDDEN
        + 3 * DECODER_HIDDEN * DECODER_INTERMEDIATE
    )
)

#: Per-token-squared decoder attention FLOPs, halved for causal masking.
_DECODER_ATTENTION: Final[int] = DECODER_LAYERS * 2 * _Q_WIDTH

#: Backward cost relative to forward for a frozen-weight layer. A trainable layer
#: needs both ``dX`` and ``dW`` (2x forward); here the base weights take no
#: gradient, so only ``dX`` propagates and the LoRA factors' own ``dW`` is over
#: matrices three orders of magnitude smaller. Hence ~1.0 rather than ~2.0.
BACKWARD_MULTIPLIER: Final[float] = 1.0


def vision_flops(patches: float, patches_sq: float) -> float:
    """Forward FLOPs of the frozen vision tower.

    Args:
        patches: Mean pre-merge patch count.
        patches_sq: Mean *squared* patch count, for the quadratic attention term.

    Returns:
        Forward FLOPs. There is no backward term: with every visual parameter
        frozen and ``pixel_values`` not requiring grad, the tower builds no
        autograd graph, so its cost is paid exactly once.
    """
    return _VISION_LINEAR * patches + _VISION_ATTENTION * patches_sq


def decoder_flops(seq_len: float, seq_len_sq: float) -> float:
    """Forward FLOPs of the language decoder.

    Args:
        seq_len: Mean sequence length.
        seq_len_sq: Mean squared sequence length.

    Returns:
        Forward FLOPs for one sample.
    """
    return _DECODER_LINEAR * seq_len + _DECODER_ATTENTION * seq_len_sq


def projector_flops(visual_tokens: float) -> float:
    """Forward FLOPs of the ``mlp_AR`` visual projector.

    Args:
        visual_tokens: Mean merged visual tokens per sample.

    Returns:
        Forward FLOPs for one sample.
    """
    return 2.0 * visual_tokens * PROJECTOR_PARAMETERS


def head_flops(answer_tokens: float) -> float:
    """Forward FLOPs of the sliced vocabulary projection.

    Only supervised positions are projected, which is the whole point of
    :class:`~fa_ocr_lora.training.modeling.SlicedLMHead`: projecting the full
    sequence would be ~7x this at page lengths.

    Args:
        answer_tokens: Mean supervised tokens per sample.

    Returns:
        Forward FLOPs for one sample.
    """
    return 2.0 * answer_tokens * DECODER_HIDDEN * VOCAB_SIZE


def sample_flops(
    profile: SourceProfile | CorpusProfile,
    *,
    gradient_checkpointing: bool,
    backward_multiplier: float = BACKWARD_MULTIPLIER,
) -> dict[str, float]:
    """FLOPs for one average sample, broken down by component.

    Args:
        profile: A single source or a whole mix; both expose the same means.
        gradient_checkpointing: Whether the decoder is recomputed during
            backward. Adds one extra decoder forward per sample when set.
        backward_multiplier: Backward cost relative to forward, see
            :data:`BACKWARD_MULTIPLIER`.

    Returns:
        Mapping of component name to FLOPs, plus a ``"total"`` key.
    """
    if isinstance(profile, CorpusProfile):
        patches = profile.mean_patches
        patches_sq = profile.mean_patches_sq
        seq = profile.mean_seq_len
        answers = profile.mean_answer_tokens
    else:
        patches = profile.patches
        patches_sq = profile.patches_sq
        seq = profile.seq_len
        answers = profile.answer_tokens
    visual_tokens = patches / 4.0

    # E[s^2] is not carried through a mix, so the square of the mean is corrected
    # upward by the measured patch-level dispersion. Decoder attention is the
    # smaller of the two decoder terms at these lengths, so the residual error
    # from that substitution stays well under a percent of the total.
    dispersion = patches_sq / (patches**2) if patches > 0 else 1.0
    seq_sq = (seq**2) * dispersion

    recompute = 1.0 if gradient_checkpointing else 0.0
    vision = vision_flops(patches, patches_sq)
    projector = projector_flops(visual_tokens)
    # The loss chunks are checkpointed unconditionally, so the head always pays
    # one recompute regardless of the decoder-wide setting.
    head = head_flops(answers) * (1.0 + backward_multiplier + 1.0)
    decoder = decoder_flops(seq, seq_sq) * (1.0 + backward_multiplier + recompute)
    return {
        "vision": vision,
        "projector": projector,
        "head": head,
        "decoder": decoder,
        "total": vision + projector + head + decoder,
    }


# --------------------------------------------------------------------------
# Memory.
# --------------------------------------------------------------------------

#: Whether the vision tower's attention materialises a ``[patches, patches]``
#: matrix in device memory.
#:
#: ``False`` assumes the packed/flash path: a batch's images are attended
#: block-diagonally per image, with the scores tiled in SRAM rather than written
#: to HBM. That is how this class of model is built for multi-image batching, and
#: it is what makes ``per_device_batch_size: 8`` at full page resolution possible
#: at all.
#:
#: Measured on this checkpoint rather than assumed, by
#: ``scripts/probe_vision_attention.py``: forwarding the same image at batches 1
#: through 4 (1,024 patches each) peaked at 245.4 / 704.9 / 1058.3 / 1410.2 MiB,
#: which is 352.4 / 352.8 / 352.6 MiB per image across the last three - constant to
#: within 0.1%, an exponent of 1.00. Dense attention would have doubled the
#: per-image cost from batch 2 to batch 4; it did not move.
#:
#: Batch 1 is off that line (245 MiB) because it pays one-off workspace and picks
#: different kernel tilings, which is why the probe fits its verdict between the two
#: largest batches rather than over all of them.
#:
#: It stays a named flag rather than becoming a bare ``False`` so the counterfactual
#: is still priceable: set it ``True`` to see what the fallback path would need. The
#: report prints both.
VISION_ATTENTION_MATERIALISED: Final[bool] = False


@dataclass(frozen=True, slots=True)
class MemoryModel:
    """Device memory, component by component, at the run's worst batch.

    Every field is MiB. The split matters more than the total: it says which knob
    to turn if the total is too large, and which knobs would not help.

    Attributes:
        weights: Frozen base parameters, resident for the whole run.
        adapter: LoRA factors, trainable.
        gradients: One gradient per trainable parameter.
        optimizer: AdamW's two fp32 moments.
        decoder_activations: Saved tensors of the language decoder, the term that
            scales with batch size and sequence length.
        vision_transient: Peak of the frozen tower's forward. Transient rather
            than resident, but it coincides with the decoder's peak on the same
            microbatch, so it is counted.
        loss_logits: The chunked vocabulary projection's fp32 logits.
        cuda_overhead: Context, workspaces and allocator bookkeeping.
        fragmentation: Allocator slack applied to the dynamic components.
    """

    weights: float
    adapter: float
    gradients: float
    optimizer: float
    decoder_activations: float
    vision_transient: float
    loss_logits: float
    cuda_overhead: float
    fragmentation: float

    @property
    def total(self) -> float:
        """Projected peak device memory in MiB."""
        return (
            self.weights
            + self.adapter
            + self.gradients
            + self.optimizer
            + self.decoder_activations
            + self.vision_transient
            + self.loss_logits
            + self.cuda_overhead
            + self.fragmentation
        )

    def breakdown(self) -> tuple[tuple[str, float], ...]:
        """Component names and sizes, largest first.

        Returns:
            Pairs of ``(label, mib)`` ordered by size.
        """
        items = (
            ("frozen base weights", self.weights),
            ("decoder activations", self.decoder_activations),
            ("vision tower (transient)", self.vision_transient),
            ("CUDA context + workspaces", self.cuda_overhead),
            ("sliced loss logits (fp32)", self.loss_logits),
            ("allocator fragmentation", self.fragmentation),
            ("AdamW state (fp32 m+v)", self.optimizer),
            ("LoRA gradients", self.gradients),
            ("LoRA parameters", self.adapter),
        )
        return tuple(sorted(items, key=lambda pair: pair[1], reverse=True))


def trainable_parameters(config: TrainingConfig) -> int:
    """Count LoRA parameters the configuration would create.

    Derived from the adapter shape and the pinned architecture rather than by
    building the model, and verified against the 9,388,032 parameters in the
    adapter this repository has already trained.

    Args:
        config: The training configuration.

    Returns:
        Number of trainable parameters.
    """
    rank = config.lora.r
    targets = set(config.lora.target_modules)
    shapes: dict[str, tuple[int, int]] = {
        "q_proj": (DECODER_HIDDEN, _Q_WIDTH),
        "k_proj": (DECODER_HIDDEN, _KV_WIDTH),
        "v_proj": (DECODER_HIDDEN, _KV_WIDTH),
        "o_proj": (_Q_WIDTH, DECODER_HIDDEN),
        "gate_proj": (DECODER_HIDDEN, DECODER_INTERMEDIATE),
        "up_proj": (DECODER_HIDDEN, DECODER_INTERMEDIATE),
        "down_proj": (DECODER_INTERMEDIATE, DECODER_HIDDEN),
    }
    per_layer = sum(
        rank * (fan_in + fan_out)
        for name, (fan_in, fan_out) in shapes.items()
        if name in targets
    )
    total = per_layer * DECODER_LAYERS
    if config.lora.adapt_embeddings:
        total += rank * (VOCAB_SIZE + DECODER_HIDDEN)
    if config.lora.adapt_lm_head:
        total += rank * (DECODER_HIDDEN + VOCAB_SIZE)
    return total


def weight_bytes(config: TrainingConfig) -> float:
    """Bytes the frozen base weights occupy under the configured precision.

    NF4 stores 4 bits per weight plus a block scale, and with
    ``bnb_4bit_use_double_quant`` the scales are themselves quantised - about 0.55
    bytes per parameter. ``lm_head`` is always excluded from quantisation by
    :func:`~fa_ocr_lora.training.modeling.quantization_config`, and the input
    embedding is not an ``nn.Linear`` at all, so both stay at the compute dtype
    whatever the setting.

    Args:
        config: The training configuration.

    Returns:
        Size in bytes.
    """
    unquantisable = 2 * VOCAB_SIZE * DECODER_HIDDEN  # embed_tokens + lm_head
    if config.model.quantization == "none":
        return BASE_PARAMETERS * BF16_BYTES
    quantised = BASE_PARAMETERS - unquantisable
    if not config.model.quantize_vision:
        quantised -= VISION_PARAMETERS
    full = BASE_PARAMETERS - quantised
    return quantised * 0.55 + full * BF16_BYTES


def memory_model(
    config: TrainingConfig,
    profile: CorpusProfile,
    *,
    worst_case: bool = True,
    materialise_vision_attention: bool = VISION_ATTENTION_MATERIALISED,
) -> MemoryModel:
    """Project peak device memory for one training step.

    The worst step is not the average one. The length-grouped sampler sorts each
    megabatch longest-first, so the first microbatch of every window holds that
    window's longest sequences - peak memory is reached predictably and often,
    not rarely. That is what ``worst_case`` models.

    Args:
        config: The training configuration.
        profile: The corpus mix.
        worst_case: Size activations from the longest sequence any source
            produces rather than from the mean.
        materialise_vision_attention: Whether to charge for a dense
            ``[patches, patches]`` attention matrix in the vision tower. See
            :data:`VISION_ATTENTION_MATERIALISED`.

    Returns:
        The projected breakdown.
    """
    batch = config.optim.per_device_batch_size
    trainable = trainable_parameters(config)

    if worst_case:
        seq = float(max((s.seq_len_max for s in profile.sources), default=0))
        patches = float(max((s.patches_max for s in profile.sources), default=0))
    else:
        seq = profile.mean_seq_len
        patches = profile.mean_patches

    # Decoder saved tensors. Per layer and token a transformer block keeps its
    # residual stream, the attention projections and both MLP branches alive for
    # backward; ~20*H covers them at this width. With checkpointing only the layer
    # boundaries survive, plus whichever layer is being recomputed.
    per_token_per_layer = 20 * DECODER_HIDDEN * BF16_BYTES
    if config.optim.gradient_checkpointing:
        activations = batch * seq * DECODER_HIDDEN * BF16_BYTES * DECODER_LAYERS
        activations += batch * seq * per_token_per_layer
    else:
        activations = batch * seq * per_token_per_layer * DECODER_LAYERS
        activations += batch * DECODER_HEADS * seq * DECODER_HEAD_DIM * BF16_BYTES * 2

    # The tower runs without an autograd graph, so only a few layers' tensors are
    # alive at once - but attention over 11,088 patches is still the largest
    # transient allocation in the run when a batch is full of pages.
    vision_transient = batch * patches * VISION_HIDDEN * BF16_BYTES * 4
    if materialise_vision_attention:
        # One [patches, patches] score matrix per head, per image in the batch.
        # At 11,088 patches that is 245 MiB per head before the head count is
        # applied, which is why this branch decides whether the run is possible.
        vision_transient += batch * VISION_HEADS * patches * patches * BF16_BYTES

    logits = config.image.loss_chunk_rows * VOCAB_SIZE * 4 * 2  # fp32, plus grad

    dynamic = activations + vision_transient + logits
    return MemoryModel(
        weights=weight_bytes(config) / MIB,
        adapter=trainable * BF16_BYTES / MIB,
        gradients=trainable * BF16_BYTES / MIB,
        optimizer=trainable * ADAM_STATE_BYTES / MIB,
        decoder_activations=activations / MIB,
        vision_transient=vision_transient / MIB,
        loss_logits=logits / MIB,
        cuda_overhead=CUDA_OVERHEAD_MIB,
        fragmentation=(dynamic / MIB) * (FRAGMENTATION - 1.0),
    )


# --------------------------------------------------------------------------
# Throughput.
# --------------------------------------------------------------------------

#: Fraction of the card's peak bf16 rate a run of this shape sustains.
#:
#: Not a guess from thin air: the 4 GB run this repository completed is
#: back-solved by :func:`calibrate` into an achieved-FLOPS figure, and these
#: brackets say how much better the A100 does at batch 8 with no NF4
#: dequantisation. They are brackets rather than a point estimate because that
#: transfer is the least certain step in the whole projection, and a single number
#: would hide it.
#:
#: The bounds come from what changes between the two runs: batch 1 leaves the
#: tensor cores idle between tiny GEMMs, while batch 8 amortises every launch; NF4
#: dequantises each weight tile on the fly, which bf16 does not. Both push the same
#: way, so the 4 GB figure is a floor, not a midpoint.
EFFICIENCY_SCENARIOS: Final[dict[str, float]] = {
    "pessimistic": 0.25,
    "expected": 0.35,
    "optimistic": 0.45,
}


@dataclass(frozen=True, slots=True)
class Calibration:
    """What a completed run says about achievable throughput.

    The point of this class is falsifiability. A cost model that has never been
    checked against a real run is a spreadsheet; one that reproduces a real run to
    within a stated residual is a projection. :attr:`vram_residual` is the check
    on the memory model, and :attr:`mfu` is the number the throughput model
    inherits.

    Attributes:
        run: Directory the measurements came from.
        hardware: The card that ran it.
        samples: Samples the reported wall-clock actually covers - every
            accumulation micro-batch of every epoch, not one epoch's worth.
        seconds: Wall-clock of the run.
        seconds_per_sample: As reported by the run.
        flops_per_sample: What this module's FLOPs model says that mix cost.
        achieved_tflops: Back-solved sustained rate.
        mfu: Achieved rate over the card's vendor peak.
        peak_vram_mib: Peak device memory the run actually reported.
        predicted_vram_mib: What :func:`memory_model` predicts for that run.
    """

    run: str
    hardware: Hardware
    samples: int
    seconds: float
    seconds_per_sample: float
    flops_per_sample: float
    achieved_tflops: float
    mfu: float
    peak_vram_mib: float
    predicted_vram_mib: float

    @property
    def vram_residual(self) -> float:
        """Predicted peak over measured peak. 1.0 would be exact."""
        if self.peak_vram_mib <= 0:
            return float("nan")
        return self.predicted_vram_mib / self.peak_vram_mib


@dataclass(frozen=True, slots=True)
class ThroughputModel:
    """Wall-clock projection for one hardware and efficiency assumption.

    Attributes:
        hardware: Target machine.
        flops_per_sample: Mean cost of a sample in the configured mix.
        mfu: Assumed fraction of peak sustained.
        label: Which efficiency scenario this is.
    """

    hardware: Hardware
    flops_per_sample: float
    mfu: float
    label: str

    @property
    def achieved_tflops(self) -> float:
        """Sustained rate this scenario assumes."""
        return self.hardware.bf16_tflops * self.mfu

    @property
    def seconds_per_sample(self) -> float:
        """Compute time for one average sample."""
        rate = self.achieved_tflops * 1e12
        if rate <= 0:
            return float("inf")
        return self.flops_per_sample / rate

    def hours(self, samples: int) -> float:
        """Hours to process a number of samples.

        Args:
            samples: Sample count, oversampling included.

        Returns:
            Wall-clock hours, compute-bound.
        """
        return self.seconds_per_sample * samples / 3600.0


def calibrate(
    run_dir: Path,
    *,
    hardware: Hardware,
    config: TrainingConfig,
    profile: CorpusProfile,
) -> Calibration:
    """Back-solve achieved throughput from a run that already happened.

    Args:
        run_dir: Directory holding ``training_summary.json``.
        hardware: The card that ran it.
        config: The configuration that run used, for its precision and batching.
        profile: The mix that run trained on, profiled under that configuration.

    Returns:
        The calibration.

    Raises:
        ConfigError: If the summary is missing or lacks the timing fields.
    """
    summary_path = run_dir / "training_summary.json"
    if not summary_path.is_file():
        raise ConfigError(
            f"no training_summary.json under {run_dir}; run a probe or a short "
            "training run first, or pass --no-calibrate to project from the "
            "efficiency brackets alone"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # pragma: no cover - I/O guard
        raise ConfigError(f"{summary_path}: unreadable ({exc})") from exc

    seconds = float(summary.get("seconds") or 0.0)
    # How many samples that wall-clock actually covers. `epoch_samples` is the
    # size of ONE pass while `seconds` is the whole run, so dividing one by the
    # other overstates the per-sample cost by the epoch count. The divisor the
    # trainer itself uses is `steps x batch_size` (loop.TrainingOutcome.samples),
    # and the sources below are ordered by how directly they recover it.
    #
    # `batch_size` is read from the summary only, never from `config`. The
    # configuration being simulated describes the TARGET machine; the run being
    # calibrated happened on a different one, under a different batch size. The
    # lora_1000 summary predates the `batch_size` field and was written by a run
    # at effective batch 8, so borrowing this A100 config's 16 halved the measured
    # per-sample cost and doubled the apparent MFU to an implausible 45%.
    steps = int(summary.get("steps") or 0)
    batch = int(summary.get("batch_size") or 0)
    reported = float(summary.get("seconds_per_sample") or 0.0)
    if steps > 0 and batch > 0:
        samples = steps * batch
    elif int(summary.get("samples_consumed") or 0) > 0:
        samples = int(summary["samples_consumed"])
    elif reported > 0:
        # Back-solved from the trainer's own rate, which it computed with the
        # right divisor even when it did not record what that divisor was.
        samples = int(round(seconds / reported))
    else:
        samples = int(summary.get("epoch_samples") or 0)
    if seconds <= 0 or samples <= 0:
        raise ConfigError(
            f"{summary_path}: needs positive 'seconds' and a sample count "
            "('steps' x 'batch_size', 'samples_consumed', 'seconds_per_sample', "
            "or 'epoch_samples') to calibrate against"
        )

    flops = sample_flops(
        profile, gradient_checkpointing=config.optim.gradient_checkpointing
    )["total"]
    achieved = flops * samples / seconds / 1e12
    predicted = memory_model(config, profile, worst_case=True).total
    return Calibration(
        run=str(run_dir),
        hardware=hardware,
        samples=samples,
        seconds=seconds,
        seconds_per_sample=seconds / samples,
        flops_per_sample=flops,
        achieved_tflops=achieved,
        mfu=achieved / hardware.bf16_tflops if hardware.bf16_tflops else 0.0,
        peak_vram_mib=float(summary.get("peak_vram_mib") or 0.0),
        predicted_vram_mib=predicted,
    )


# --------------------------------------------------------------------------
# Corpus profiling.
#
# The whole point of this section is that it does not guess. It walks the real
# corpora through the real source iterators, so every filter, exclusion and size
# floor the trainer applies is applied here too - a duplicated filter would drift
# from `sources.py` and quietly invalidate the projection.
#
# The one thing it will not do is load a tokenizer: the answer length is derived
# from character count via a measured ratio. That keeps the simulation runnable on
# a machine with no model cache, which is the situation you are in when deciding
# whether to rent the machine.
# --------------------------------------------------------------------------

#: Prompt tokens other than image placeholders, from the pinned revision's
#: ``chat_template.jinja``. Measured, not assumed:
#: :attr:`~fa_ocr_lora.training.encoding.SampleEncoder.prompt_overhead` computes
#: exactly this at runtime and the two agree.
PROMPT_OVERHEAD: Final[int] = 13

#: Characters per token, measured per source over the whole corpus with the real
#: tokenizer.
#:
#: Persian is tokenised inconsistently across these sources, which is why the
#: ratio is per source rather than global: the letters are template-generated with
#: many repeated administrative phrases that land on single tokens, while Persian
#: Pixel is natural scanned prose. :func:`~fa_ocr_lora.training.data.token_lengths`
#: hardcodes 2.2 for its sampler, which these numbers confirm is a fair global
#: compromise.
CHARS_PER_TOKEN: Final[dict[str, float]] = {
    "admin_letter_page": 1.976,
    "admin_letter_crop": 1.949,
    "persian_pixel_page": 2.218,
}

#: Images per source, for the storage model. Sampled rather than summed: a `du`
#: over 27,000 files is slower than the rest of this module put together.
_STAT_SAMPLE: Final[int] = 400


def _feed_geometry(
    width: int, height: int, answer_tokens: int, image: Any
) -> tuple[int, int]:
    """Visual tokens and total sequence length for one record, as fed.

    Replicates :meth:`~fa_ocr_lora.training.encoding.SampleEncoder._image_budget`
    exactly, because that is where the interesting behaviour lives: the image is
    given whatever the sequence budget has left *after* the answer, so a long
    transcription silently costs resolution. Any model of this run that applies
    ``max_pixels`` unconditionally overstates the cost of its longest samples and
    understates how often they are degraded.

    Args:
        width: Region width in pixels, before resizing.
        height: Region height in pixels.
        answer_tokens: Supervised token count, EOS included.
        image: The ``image`` section of the configuration.

    Returns:
        ``(image_tokens, sequence_length)``.
    """
    spare = image.max_total_seq_len - PROMPT_OVERHEAD - answer_tokens
    allowed = max(0, spare) * PIXELS_PER_TOKEN
    budget = max(image.min_pixels, min(image.max_pixels, allowed))
    fitted = image_tokens(
        *fit_pixel_budget(width, height, min_pixels=image.min_pixels, max_pixels=budget)
    )
    total = min(image.max_total_seq_len, PROMPT_OVERHEAD + fitted + answer_tokens)
    return fitted, total


class _Accumulator:
    """Running statistics for one source, in a single pass.

    Keeps the sums needed for both a mean and a mean-of-squares, because vision
    attention is quadratic in patch count and ``E[p^2]`` is not recoverable from
    ``E[p]`` on a bimodal corpus like these letter pages - 66% of them land at
    2,772 visual tokens and 16% at 1,364.
    """

    __slots__ = (
        "ans",
        "clamped",
        "img",
        "n",
        "pat",
        "pat_max",
        "pat_sq",
        "seq",
        "seq_max",
        "seq_sq",
    )

    def __init__(self) -> None:
        self.n = 0
        self.img = 0.0
        self.ans = 0.0
        self.seq = 0.0
        self.seq_sq = 0.0
        self.pat = 0.0
        self.pat_sq = 0.0
        self.seq_max = 0
        self.pat_max = 0
        self.clamped = 0

    def add(self, visual: int, answer: int, seq: int, *, clamped: bool) -> None:
        """Fold one record in."""
        patches = visual * 4
        self.n += 1
        self.img += visual
        self.ans += answer
        self.seq += seq
        self.seq_sq += float(seq) * seq
        self.pat += patches
        self.pat_sq += float(patches) * patches
        self.seq_max = max(self.seq_max, seq)
        self.pat_max = max(self.pat_max, patches)
        self.clamped += int(clamped)

    def finish(self, source: str, records: int, bytes_on_disk: float) -> SourceProfile:
        """Turn the sums into a profile.

        Args:
            source: Manifest source tag.
            records: How many records the configuration draws, which may be fewer
                than were scanned - the mix samples, so the scanned mean is an
                unbiased estimate of the drawn mean.
            bytes_on_disk: Mean image file size.

        Returns:
            The profile.
        """
        n = max(1, self.n)
        return SourceProfile(
            source=source,
            records=records,
            image_tokens=self.img / n,
            answer_tokens=self.ans / n,
            seq_len=self.seq / n,
            seq_len_max=self.seq_max,
            patches=self.pat / n,
            patches_sq=self.pat_sq / n,
            patches_max=self.pat_max,
            bytes_on_disk=bytes_on_disk,
        )


def _mean_file_size(paths: Sequence[Path]) -> float:
    """Mean size in bytes of a sample of files.

    Args:
        paths: Candidate paths; a bounded sample is stat'ed.

    Returns:
        Mean size, or 0.0 if nothing could be stat'ed.
    """
    if not paths:
        return 0.0
    step = max(1, len(paths) // _STAT_SAMPLE)
    sizes = []
    for path in paths[::step][:_STAT_SAMPLE]:
        try:
            sizes.append(path.stat().st_size)
        except OSError:
            continue
    return sum(sizes) / len(sizes) if sizes else 0.0


def profile_corpus(
    config: TrainingConfig, *, pixel_available: int | None = None
) -> CorpusProfile:
    """Walk both corpora and report what the configured mix would feed.

    Args:
        config: Loaded configuration.
        pixel_available: How many Persian Pixel page rows exist in full, when the
            local copy is a partial download. The complete ``sample_type = page``
            set is about 31,000 rows; a partial copy would otherwise cap the
            projection at whatever happens to be on this disk and silently
            understate the run.

    Returns:
        The profiled mix.

    Raises:
        ConfigError: If neither corpus can be read.
    """
    skips = SkipLog(None)
    data = config.data
    image = config.image

    pages = _Accumulator()
    page_paths: list[Path] = []
    per_type: dict[str, _Accumulator] = {}

    letters = config.paths.letters_manifest
    if letters.is_file():
        for record in iter_letter_pages(letters, data, skips):
            answer = int(len(record.text) / CHARS_PER_TOKEN[LETTER_PAGE]) + 1
            visual, seq = _feed_geometry(record.width, record.height, answer, image)
            full, _ = _feed_geometry(record.width, record.height, 0, image)
            pages.add(visual, answer, seq, clamped=visual < full)
            page_paths.append(record.image_path)
        for record in iter_letter_crops(letters, data, skips):
            answer = int(len(record.text) / CHARS_PER_TOKEN[LETTER_CROP]) + 1
            visual, seq = _feed_geometry(record.width, record.height, answer, image)
            full, _ = _feed_geometry(record.width, record.height, 0, image)
            region = record.region_type or "unlabelled"
            bucket = per_type.setdefault(region, _Accumulator())
            bucket.add(visual, answer, seq, clamped=visual < full)
    else:
        LOGGER.warning("letters manifest %s is missing; letters excluded", letters)

    pixels = _Accumulator()
    pixel_paths: list[Path] = []
    metadata = config.paths.pixel_metadata
    if metadata.is_file():
        for record in iter_pixel_rows(metadata, data, skips):
            answer = int(len(record.text) / CHARS_PER_TOKEN[PIXEL_PAGE]) + 1
            visual, seq = _feed_geometry(record.width, record.height, answer, image)
            full, _ = _feed_geometry(record.width, record.height, 0, image)
            pixels.add(visual, answer, seq, clamped=visual < full)
            pixel_paths.append(record.image_path)
    else:
        LOGGER.warning("pixel metadata %s is missing; Persian Pixel excluded", metadata)

    if pages.n == 0 and pixels.n == 0:
        raise ConfigError(
            f"no readable corpus: tried {letters} and {metadata}. Fix "
            "paths.letters_manifest / paths.pixel_metadata, or run the simulation "
            "on the machine that holds the data."
        )

    sources: list[SourceProfile] = []
    if pages.n:
        sources.append(
            pages.finish(
                LETTER_PAGE,
                min(data.letters_pages, pages.n),
                _mean_file_size(page_paths),
            )
        )
    if per_type:
        sources.append(_crop_profile(config, per_type, _mean_file_size(page_paths)))
    if pixels.n:
        supply = pixels.n if pixel_available is None else max(pixels.n, pixel_available)
        sources.append(
            pixels.finish(
                PIXEL_PAGE,
                min(data.pixel_pages, supply),
                _mean_file_size(pixel_paths),
            )
        )

    train, validation, test = _split_sizes(config, sum(s.records for s in sources))
    return CorpusProfile(
        sources=tuple(sources),
        train_records=train,
        validation_records=validation,
        test_records=test,
        oversample_factor=_oversample_factor(config),
    )


def _crop_profile(
    config: TrainingConfig,
    per_type: dict[str, _Accumulator],
    bytes_on_disk: float,
) -> SourceProfile:
    """Blend per-type crop statistics under the configured type quotas.

    Crops are not drawn uniformly: :func:`~fa_ocr_lora.training.manifest.
    _select_crops` allocates the budget across region types by weight, so the A100
    config's ``metadata_field: 8.0`` means those tiny high-value boxes are drawn
    far above their natural share. Averaging crops uniformly would profile a mix
    the trainer never sees, so the real allocator is replayed here.

    Args:
        config: Loaded configuration, for the crop weights.
        per_type: Accumulated statistics keyed by region type.
        bytes_on_disk: Mean page file size; crops are read from the page images.

    Returns:
        The quota-weighted crop profile.
    """
    available = {name: acc.n for name, acc in per_type.items()}
    weights = {name: config.data.crop_weight(name) for name in available}
    quota = _allocate(available, weights, config.data.letters_crops)
    drawn = sum(quota.values())

    blend = _Accumulator()
    blend.n = drawn
    for name, take in quota.items():
        if take <= 0:
            continue
        acc = per_type[name]
        share = take / max(1, acc.n)
        blend.img += acc.img * share
        blend.ans += acc.ans * share
        blend.seq += acc.seq * share
        blend.seq_sq += acc.seq_sq * share
        blend.pat += acc.pat * share
        blend.pat_sq += acc.pat_sq * share
        blend.clamped += int(acc.clamped * share)
        blend.seq_max = max(blend.seq_max, acc.seq_max)
        blend.pat_max = max(blend.pat_max, acc.pat_max)
    return blend.finish(LETTER_CROP, drawn, bytes_on_disk)


def _split_sizes(config: TrainingConfig, selected: int) -> tuple[int, int, int]:
    """Train/validation/test record counts for a selection size.

    Mirrors :func:`~fa_ocr_lora.training.manifest.build_manifest`, including its
    cap on the holdout: an 800-record validation set out of 200 selected records
    would swallow the training split, so the builder clamps it to a share.

    Args:
        config: Loaded configuration.
        selected: Records selected across all sources.

    Returns:
        ``(train, validation, test)``.
    """
    holdout = min(config.data.validation_size, max(1, int(_MAX_HOLDOUT_SHARE * selected)))
    test = int(selected * config.data.test_fraction)
    return max(0, selected - holdout - test), holdout, test


def _oversample_factor(config: TrainingConfig) -> float:
    """Samples seen per epoch, per training record.

    Digit-dense records are repeated so the model sees more of the fields whose
    errors matter most. The manifest cuts at the training split's upper quartile,
    so a quarter of records carry the multiplier.

    Args:
        config: Loaded configuration.

    Returns:
        A factor at or above 1.0.
    """
    return 1.0 + 0.25 * (config.data.digit_oversample_multiplier - 1.0)


# --------------------------------------------------------------------------
# Host: RAM, the input pipeline, and disk.
# --------------------------------------------------------------------------

#: Resident set of the training process before any batch exists: the Python
#: interpreter, torch's own libraries, the CUDA runtime and cuDNN/cuBLAS kernel
#: images. Dominated by the shared objects, so it barely moves with model size.
PARENT_RSS_MIB: Final[float] = 3200.0

#: Resident set of one dataloader worker. Forked after torch is imported, so much
#: of the parent's text pages are shared copy-on-write; what is genuinely private
#: is the processor, the Pillow decode buffers and the worker's slice of the
#: manifest.
WORKER_RSS_MIB: Final[float] = 900.0

#: Batches each worker prepares ahead of the consumer. Torch's default.
PREFETCH_FACTOR: Final[int] = 2

#: Bytes per patch in the collated ``pixel_values`` tensor: 3 channels of a
#: 14x14 patch at fp32, which is what the image processor emits before the model
#: casts it. This is the term people forget - it makes one full-resolution page
#: 26 MB of host memory on its way to the device, dwarfing the JPEG it came from.
PIXEL_BYTES_PER_PATCH: Final[int] = 3 * 14 * 14 * 4

#: Seconds of CPU time to turn one page image into ``pixel_values``: decode,
#: convert, resize, normalise, and reshape into patches. Scaled by patch count
#: from a page-sized baseline, since every stage is linear in pixels.
CPU_SECONDS_PER_KPATCH: Final[float] = 0.0072


@dataclass(frozen=True, slots=True)
class HostModel:
    """Host-side resources: memory, the input pipeline, and disk.

    Attributes:
        parent_mib: Resident set of the training process.
        workers_mib: Resident set of all dataloader workers together.
        inflight_mib: Collated batches in flight between workers and the device.
        pinned_mib: Pinned staging buffers for host-to-device copies.
        samples_per_second_supply: What the workers can produce.
        samples_per_second_demand: What the GPU will consume.
        bytes_per_second: Host-to-device traffic for ``pixel_values``.
        storage: Disk requirement by component, in GiB.
    """

    parent_mib: float
    workers_mib: float
    inflight_mib: float
    pinned_mib: float
    samples_per_second_supply: float
    samples_per_second_demand: float
    bytes_per_second: float
    storage: tuple[tuple[str, float], ...]

    @property
    def ram_mib(self) -> float:
        """Total host memory in MiB."""
        return self.parent_mib + self.workers_mib + self.inflight_mib + self.pinned_mib

    @property
    def storage_gib(self) -> float:
        """Total disk requirement in GiB."""
        return sum(size for _, size in self.storage)

    @property
    def input_bound(self) -> bool:
        """Whether the dataloader, not the GPU, would set the pace."""
        return self.samples_per_second_supply < self.samples_per_second_demand


def host_model(
    config: TrainingConfig,
    profile: CorpusProfile,
    hardware: Hardware,
    *,
    samples_per_second: float,
) -> HostModel:
    """Project host memory, input-pipeline throughput, and disk.

    Args:
        config: Loaded configuration.
        profile: The corpus mix.
        hardware: Target machine.
        samples_per_second: What the GPU is projected to consume, so supply can
            be compared against demand rather than reported in isolation.

    Returns:
        The projection.
    """
    workers = max(1, config.runtime.dataloader_workers)
    batch = config.optim.per_device_batch_size

    batch_bytes = batch * profile.mean_patches * PIXEL_BYTES_PER_PATCH
    inflight = batch_bytes * workers * PREFETCH_FACTOR
    # `loop.py` sets dataloader_pin_memory=True unconditionally, so the staging
    # buffers are always there - roughly one batch being copied and one queued.
    pinned = batch_bytes * 2

    # A worker cannot outrun the core it is on, so the supply is capped by cores
    # actually available - one reserved for the training process itself.
    effective = min(workers, max(1, hardware.vcpus - 1))
    per_sample_cpu = profile.mean_patches / 1000.0 * CPU_SECONDS_PER_KPATCH
    supply = effective / per_sample_cpu if per_sample_cpu > 0 else float("inf")

    adapter_mib = trainable_parameters(config) * BF16_BYTES / MIB
    checkpoints = max(1, config.runtime.save_total_limit)
    # Crops are windows onto page files that are already counted, so they add no
    # bytes - only the page-level sources contribute to the corpus footprint.
    images = sum(
        s.bytes_on_disk * s.records for s in profile.sources if "crop" not in s.source
    )
    storage = (
        ("base checkpoint (bf16 + HF cache)", BASE_PARAMETERS * BF16_BYTES * 2 / GIB),
        ("corpus images", images / GIB),
        ("manifest + parquet + logs", 0.5),
        (
            f"checkpoints ({checkpoints} x adapter, embeddings excluded)",
            checkpoints * adapter_mib * 1.2 / 1024.0,
        ),
        ("eval outputs and headroom", 2.0),
    )
    host_to_device = samples_per_second * profile.mean_patches * PIXEL_BYTES_PER_PATCH
    return HostModel(
        parent_mib=PARENT_RSS_MIB,
        workers_mib=workers * WORKER_RSS_MIB,
        inflight_mib=inflight / MIB,
        pinned_mib=pinned / MIB,
        samples_per_second_supply=supply,
        samples_per_second_demand=samples_per_second,
        bytes_per_second=host_to_device,
        storage=storage,
    )


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Simulation:
    """Everything the projection produced, ready to render or serialise.

    Attributes:
        config: The configuration simulated.
        hardware: The machine simulated onto.
        profile: The corpus mix, measured.
        flops: FLOPs per average sample by component.
        memory: Peak device memory, modelled at the worst batch.
        memory_typical: The same, at the mean batch, for the gap between them.
        memory_dense_vision: Peak if the vision tower materialised attention.
        throughput: One entry per efficiency bracket.
        host: Host memory, input pipeline and disk.
        calibration: The completed run this was anchored to, if any.
        steps_per_epoch: Optimizer steps per epoch.
        total_steps: Optimizer steps over the whole run.
    """

    config: TrainingConfig
    hardware: Hardware
    profile: CorpusProfile
    flops: dict[str, float]
    memory: MemoryModel
    memory_typical: MemoryModel
    memory_dense_vision: MemoryModel
    throughput: tuple[ThroughputModel, ...]
    host: HostModel
    calibration: Calibration | None
    steps_per_epoch: int
    total_steps: int

    @property
    def expected(self) -> ThroughputModel:
        """The middle efficiency bracket, which the summary quotes."""
        for model in self.throughput:
            if model.label == "expected":
                return model
        return self.throughput[len(self.throughput) // 2]

    @property
    def epoch_samples(self) -> int:
        """Samples seen per epoch."""
        return self.profile.epoch_samples

    @property
    def total_samples(self) -> int:
        """Samples seen over the whole run."""
        return int(round(self.epoch_samples * self.config.optim.num_epochs))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the projection.

        Returns:
            A JSON-safe mapping, so a run can be compared against its projection
            afterwards rather than the projection being forgotten.
        """
        return {
            "hardware": {
                "name": self.hardware.name,
                "vram_gib": self.hardware.vram_gib,
                "bf16_tflops": self.hardware.bf16_tflops,
                "vcpus": self.hardware.vcpus,
                "ram_gib": self.hardware.ram_gib,
                "storage_gib": self.hardware.storage_gib,
            },
            "corpus": {
                "sources": [
                    {
                        "source": s.source,
                        "records": s.records,
                        "image_tokens": round(s.image_tokens, 1),
                        "answer_tokens": round(s.answer_tokens, 1),
                        "seq_len": round(s.seq_len, 1),
                        "seq_len_max": s.seq_len_max,
                        "patches": round(s.patches, 1),
                        "patches_max": s.patches_max,
                        "tflops_per_sample": round(
                            sample_flops(
                                s,
                                gradient_checkpointing=(
                                    self.config.optim.gradient_checkpointing
                                ),
                            )["total"]
                            / 1e12,
                            2,
                        ),
                    }
                    for s in self.profile.sources
                ],
                "train_records": self.profile.train_records,
                "validation_records": self.profile.validation_records,
                "test_records": self.profile.test_records,
                "oversample_factor": round(self.profile.oversample_factor, 3),
                "epoch_samples": self.epoch_samples,
            },
            "flops_per_sample_tflops": {
                key: round(value / 1e12, 3) for key, value in self.flops.items()
            },
            "vram_mib": {
                "peak_worst_batch": round(self.memory.total, 1),
                "peak_mean_batch": round(self.memory_typical.total, 1),
                "peak_if_dense_vision_attention": round(
                    self.memory_dense_vision.total, 1
                ),
                "components": {
                    name: round(value, 1) for name, value in self.memory.breakdown()
                },
                "budget": self.hardware.vram_gib * 1024.0,
            },
            "ram_mib": {
                "total": round(self.host.ram_mib, 1),
                "parent": round(self.host.parent_mib, 1),
                "workers": round(self.host.workers_mib, 1),
                "inflight_batches": round(self.host.inflight_mib, 1),
                "pinned": round(self.host.pinned_mib, 1),
                "budget": self.hardware.ram_gib * 1024.0,
            },
            "storage_gib": {
                "total": round(self.host.storage_gib, 2),
                "components": {name: round(size, 2) for name, size in self.host.storage},
                "budget": self.hardware.storage_gib,
            },
            "input_pipeline": {
                "samples_per_second_supply": round(
                    self.host.samples_per_second_supply, 1
                ),
                "samples_per_second_demand": round(
                    self.host.samples_per_second_demand, 2
                ),
                "host_to_device_mib_per_second": round(
                    self.host.bytes_per_second / MIB, 1
                ),
                "input_bound": self.host.input_bound,
            },
            "schedule": {
                "steps_per_epoch": self.steps_per_epoch,
                "total_steps": self.total_steps,
                "epochs": self.config.optim.num_epochs,
                "hours": {
                    model.label: round(model.hours(self.total_samples), 2)
                    for model in self.throughput
                },
            },
            "calibration": (
                None
                if self.calibration is None
                else {
                    "run": self.calibration.run,
                    "hardware": self.calibration.hardware.name,
                    "measured_seconds_per_sample": self.calibration.seconds_per_sample,
                    "modelled_tflops_per_sample": round(
                        self.calibration.flops_per_sample / 1e12, 3
                    ),
                    "achieved_tflops": round(self.calibration.achieved_tflops, 2),
                    "mfu": round(self.calibration.mfu, 4),
                    "measured_peak_vram_mib": self.calibration.peak_vram_mib,
                    "predicted_peak_vram_mib": round(
                        self.calibration.predicted_vram_mib, 1
                    ),
                    # None rather than NaN when the run reported no peak: NaN is
                    # not valid JSON and would poison every consumer downstream.
                    "vram_residual": (
                        None
                        if self.calibration.peak_vram_mib <= 0
                        else round(self.calibration.vram_residual, 3)
                    ),
                }
            ),
            "findings": [
                {"severity": severity, "text": text} for severity, text in self.findings()
            ],
        }

    def findings(self) -> tuple[tuple[str, str], ...]:
        """Derive the conclusions a reader should act on.

        Computed from the projection rather than written by hand, so they cannot
        drift out of agreement with the numbers above them.

        Returns:
            ``(severity, text)`` pairs; severity is ``"blocker"``, ``"warning"``
            or ``"note"``.
        """
        out: list[tuple[str, str]] = []
        budget = self.hardware.vram_gib * 1024.0
        peak = self.memory.total

        if peak > budget:
            out.append(
                (
                    "blocker",
                    f"Projected peak {peak / 1024:.1f} GiB exceeds the "
                    f"{budget / 1024:.0f} GiB card. Halve per_device_batch_size to "
                    f"{max(1, self.config.optim.per_device_batch_size // 2)} and double "
                    "gradient_accumulation_steps to keep the effective batch.",
                )
            )
        elif peak > 0.85 * budget:
            out.append(
                (
                    "warning",
                    f"Peak {peak / 1024:.1f} GiB leaves only "
                    f"{(budget - peak) / 1024:.1f} GiB spare. That is thin for a first "
                    "run; enable gradient_checkpointing if step one OOMs.",
                )
            )
        else:
            out.append(
                (
                    "note",
                    f"Peak {peak / 1024:.1f} GiB of {budget / 1024:.0f} GiB leaves "
                    f"{(budget - peak) / 1024:.1f} GiB spare - room to raise "
                    "per_device_batch_size or lora.r if throughput matters more than "
                    "the margin.",
                )
            )

        dense = self.memory_dense_vision.total
        if dense > budget:
            out.append(
                (
                    "note",
                    "That peak rests on the vision tower attending block-diagonally "
                    "per image, which was the one unverified assumption here and is "
                    "now measured: scripts/probe_vision_attention.py forwards one "
                    "image at batches 1-4 and finds cost per image flat to within "
                    "0.1% (exponent 1.00) across the last three. Had attention been "
                    f"dense across the batch, peak would be {dense / 1024:.1f} GiB "
                    "and step one would OOM.",
                )
            )

        crop = next((s for s in self.profile.sources if s.source == LETTER_CROP), None)
        if crop is not None and crop.records:
            flops = sample_flops(
                crop,
                gradient_checkpointing=self.config.optim.gradient_checkpointing,
            )
            share = flops["vision"] / flops["total"]
            if share > 0.5:
                out.append(
                    (
                        "note",
                        f"{share:.0%} of each crop's cost is the vision tower running on "
                        f"an image upscaled to the min_pixels floor "
                        f"({self.config.image.min_pixels:,} px = "
                        f"{self.config.image.min_pixels // PIXELS_PER_TOKEN} tokens), "
                        f"though the mean crop carries only "
                        f"{crop.answer_tokens:.0f} target tokens. Crops are "
                        f"{crop.records / max(1, self.profile.total_records):.0%} of the "
                        "mix, so lowering min_pixels is the cheapest throughput lever "
                        "available - at the cost of legibility on small print.",
                    )
                )

        if self.host.input_bound:
            out.append(
                (
                    "blocker",
                    f"The dataloader supplies {self.host.samples_per_second_supply:.1f} "
                    f"samples/s but the GPU wants "
                    f"{self.host.samples_per_second_demand:.1f}. Raise "
                    f"dataloader_workers toward {self.hardware.vcpus - 1}; the "
                    "wall-clock below assumes compute-bound and would be optimistic.",
                )
            )
        else:
            headroom = self.host.samples_per_second_supply / max(
                0.01, self.host.samples_per_second_demand
            )
            out.append(
                (
                    "note",
                    f"Input pipeline supplies {headroom:.1f}x what the GPU consumes "
                    f"({self.host.samples_per_second_supply:.0f} vs "
                    f"{self.host.samples_per_second_demand:.1f} samples/s), so the run "
                    "is compute-bound and the projection below holds.",
                )
            )

        ram = self.host.ram_mib
        ram_budget = self.hardware.ram_gib * 1024.0
        if ram > 0.85 * ram_budget:
            out.append(
                (
                    "warning",
                    f"Host memory {ram / 1024:.1f} GiB against {ram_budget / 1024:.0f} "
                    "GiB. Lower dataloader_workers - each one costs about "
                    f"{WORKER_RSS_MIB / 1024:.1f} GiB resident plus its in-flight "
                    "batches.",
                )
            )

        disk = self.host.storage_gib
        if disk > self.hardware.storage_gib:
            out.append(
                (
                    "blocker",
                    f"Disk requirement {disk:.0f} GiB exceeds the "
                    f"{self.hardware.storage_gib:.0f} GiB volume.",
                )
            )
        else:
            out.append(
                (
                    "note",
                    f"Disk {disk:.0f} GiB of {self.hardware.storage_gib:.0f} GiB. Only "
                    "the letter images, the JSONL, the Persian Pixel images and the CSV "
                    "need uploading - the hf/, pdf/, markdown/ and annotations/ trees "
                    "are local scratch and the trainer never reads them.",
                )
            )

        if self.config.model.quantization == "none":
            out.append(
                (
                    "note",
                    f"quantization: none keeps the base at bf16 - "
                    f"{weight_bytes(self.config) / MIB:,.0f} MiB, "
                    f"{weight_bytes(self.config) / MIB / budget:.1%} of the card. There "
                    "is no higher-fidelity setting than this: NF4 exists to fit a small "
                    "card, and skipping it removes both the dequantisation cost and the "
                    "bitsandbytes dependency.",
                )
            )

        clamp_note = self._clamp_finding()
        if clamp_note is not None:
            out.append(clamp_note)
        return tuple(out)

    def _clamp_finding(self) -> tuple[str, str] | None:
        """Report how often the sequence budget costs image resolution."""
        page = next((s for s in self.profile.sources if s.source == LETTER_PAGE), None)
        if page is None:
            return None
        if page.seq_len_max >= self.config.image.max_total_seq_len:
            return (
                "note",
                f"The longest page fills max_total_seq_len exactly "
                f"({self.config.image.max_total_seq_len}), so the tail of the "
                "distribution is being clipped rather than fed. That is intended - "
                "raising the cap costs quadratic decoder attention on every sample "
                "to serve a fraction of a percent of them.",
            )
        return None


def simulate(
    config: TrainingConfig,
    hardware: Hardware,
    *,
    profile: CorpusProfile | None = None,
    calibration: Calibration | None = None,
    pixel_available: int | None = None,
) -> Simulation:
    """Project a full training run.

    Args:
        config: The configuration to simulate.
        hardware: The machine to simulate onto.
        profile: A pre-computed corpus profile, or ``None`` to measure one.
        calibration: A completed run to anchor throughput to, or ``None`` to use
            the efficiency brackets alone.
        pixel_available: Full size of the Persian Pixel page set, when the local
            copy is partial.

    Returns:
        The projection.
    """
    if profile is None:
        profile = profile_corpus(config, pixel_available=pixel_available)

    flops = sample_flops(
        profile, gradient_checkpointing=config.optim.gradient_checkpointing
    )
    memory = memory_model(config, profile, worst_case=True)
    typical = memory_model(config, profile, worst_case=False)
    dense = memory_model(
        config, profile, worst_case=True, materialise_vision_attention=True
    )

    scenarios = dict(EFFICIENCY_SCENARIOS)
    if calibration is not None:
        # The measured run is a floor, not a midpoint: batch 1 and NF4 both drag
        # it down and neither applies to the target. Keeping it as the pessimistic
        # bound means the range is anchored to something real at one end.
        #
        # When the measurement lands above a bracket bound, the whole bracket
        # lifts by the same offset rather than just the floor. Raising the floor
        # alone would print a "pessimistic" run faster than the "optimistic" one,
        # which is not a conservative estimate but an unreadable table - and the
        # bracket's width is the honest part of it, so the width is preserved.
        lift = calibration.mfu - scenarios["pessimistic"]
        if lift > 0:
            scenarios = {k: v + lift for k, v in scenarios.items()}
    throughput = tuple(
        ThroughputModel(
            hardware=hardware,
            flops_per_sample=flops["total"],
            mfu=mfu,
            label=label,
        )
        for label, mfu in scenarios.items()
    )

    samples_per_step = (
        config.optim.per_device_batch_size * config.optim.gradient_accumulation_steps
    )
    steps_per_epoch = max(1, profile.epoch_samples // max(1, samples_per_step))
    total_steps = int(round(steps_per_epoch * config.optim.num_epochs))

    expected = next(m for m in throughput if m.label == "expected")
    host = host_model(
        config,
        profile,
        hardware,
        samples_per_second=1.0 / max(1e-9, expected.seconds_per_sample),
    )
    return Simulation(
        config=config,
        hardware=hardware,
        profile=profile,
        flops=flops,
        memory=memory,
        memory_typical=typical,
        memory_dense_vision=dense,
        throughput=throughput,
        host=host,
        calibration=calibration,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
    )


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------

_WIDTH: Final[int] = 78
_SEVERITY: Final[dict[str, str]] = {
    "blocker": "BLOCKER",
    "warning": "WARNING",
    "note": "note   ",
}


def _rule(title: str = "") -> str:
    """A section rule, optionally titled."""
    if not title:
        return "-" * _WIDTH
    return f"-- {title} " + "-" * max(0, _WIDTH - len(title) - 4)


def _wrap(text: str, indent: int) -> list[str]:
    """Wrap a paragraph to the report width.

    Args:
        text: Paragraph to wrap.
        indent: Left margin of continuation lines.

    Returns:
        Wrapped lines, the first unindented.
    """
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > _WIDTH - indent and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    pad = " " * indent
    return [lines[0]] + [pad + line for line in lines[1:]] if lines else []


def render(simulation: Simulation) -> str:
    """Format a projection as a plain-text report.

    Args:
        simulation: The projection to render.

    Returns:
        The report.
    """
    sim = simulation
    config = sim.config
    out: list[str] = []
    add = out.append

    add("=" * _WIDTH)
    add("  Training-run projection - no model loaded, no CUDA allocated")
    add(f"  target: {sim.hardware.name}")
    add(f"  config: {config.source_path or '<defaults>'}")
    add("=" * _WIDTH)

    # ---- corpus -----------------------------------------------------------
    add("")
    add(_rule("corpus (measured: real dimensions, real text lengths)"))
    add(
        f"  {'source':<20} {'records':>9} {'img tok':>8} {'tgt tok':>8} "
        f"{'seq':>7} {'TFLOP':>7}"
    )
    for source in sim.profile.sources:
        flops = sample_flops(
            source, gradient_checkpointing=config.optim.gradient_checkpointing
        )
        add(
            f"  {source.source:<20} {source.records:>9,} "
            f"{source.image_tokens:>8.0f} {source.answer_tokens:>8.0f} "
            f"{source.seq_len:>7.0f} {flops['total'] / 1e12:>7.2f}"
        )
    add(
        f"  {'weighted mean':<20} {sim.profile.total_records:>9,} "
        f"{sim.profile.mean_patches / 4:>8.0f} "
        f"{sim.profile.mean_answer_tokens:>8.0f} "
        f"{sim.profile.mean_seq_len:>7.0f} {sim.flops['total'] / 1e12:>7.2f}"
    )
    add("")
    add(
        f"  split       train {sim.profile.train_records:,} / validation "
        f"{sim.profile.validation_records:,} / test {sim.profile.test_records:,}"
    )
    add(
        f"  per epoch   {sim.epoch_samples:,} samples "
        f"({sim.profile.oversample_factor:.2f}x for digit oversampling), "
        f"{sim.steps_per_epoch:,} steps"
    )
    add(
        f"  whole run   {config.optim.num_epochs:g} epochs = "
        f"{sim.total_samples:,} samples, {sim.total_steps:,} steps at "
        f"{config.optim.per_device_batch_size}x"
        f"{config.optim.gradient_accumulation_steps} effective batch "
        f"{config.optim.per_device_batch_size * config.optim.gradient_accumulation_steps}"
    )

    # ---- flops ------------------------------------------------------------
    add("")
    add(_rule("cost of one average sample (modelled from tensor shapes)"))
    total = sim.flops["total"]
    for name in ("vision", "projector", "decoder", "head"):
        value = sim.flops[name]
        bar = "#" * int(round(40 * value / total)) if total else ""
        add(f"  {name:<12} {value / 1e12:>7.2f} TFLOP  {value / total:>6.1%}  {bar}")
    add(f"  {'total':<12} {total / 1e12:>7.2f} TFLOP")
    add("")
    recompute_note = (
        " plus one recompute for gradient checkpointing."
        if config.optim.gradient_checkpointing
        else ", with no recompute since gradient_checkpointing is off."
    )
    for line in _wrap(
        "The vision tower is forward-only: every visual parameter is frozen and "
        "pixel_values takes no gradient, so it builds no autograd graph and is "
        "paid once. The decoder pays forward plus backward" + recompute_note,
        2,
    ):
        add(f"  {line}" if not line.startswith("  ") else line)

    # ---- vram -------------------------------------------------------------
    budget = sim.hardware.vram_gib * 1024.0
    add("")
    add(_rule("VRAM (modelled)"))
    for name, value in sim.memory.breakdown():
        if value < 1.0:
            continue
        bar = "#" * int(round(40 * value / max(1.0, sim.memory.total)))
        add(f"  {name:<28} {value:>9,.0f} MiB  {bar}")
    add(f"  {'':<28} {'-' * 13}")
    add(
        f"  {'peak, worst batch':<28} {sim.memory.total:>9,.0f} MiB   "
        f"{sim.memory.total / budget:>5.1%} of {budget / 1024:.0f} GiB"
    )
    add(
        f"  {'peak, mean batch':<28} {sim.memory_typical.total:>9,.0f} MiB   "
        f"{sim.memory_typical.total / budget:>5.1%}"
    )
    add(f"  {'headroom':<28} {budget - sim.memory.total:>9,.0f} MiB")
    add("")
    for line in _wrap(
        "The worst batch is not a rare event: DigitAwareLengthGroupedSampler sorts "
        f"each megabatch longest-first, so the first microbatch of every "
        f"{config.optim.per_device_batch_size * 32}-sample window holds that "
        "window's longest sequences. Peak is reached early and often, which is "
        "the useful property - an OOM shows up in the first minutes, not at 03:00.",
        2,
    ):
        add(f"  {line}" if not line.startswith("  ") else line)

    # ---- host -------------------------------------------------------------
    ram_budget = sim.hardware.ram_gib * 1024.0
    add("")
    add(_rule("host RAM and input pipeline (modelled)"))
    add(f"  {'training process':<28} {sim.host.parent_mib:>9,.0f} MiB")
    add(
        f"  {'dataloader workers':<28} {sim.host.workers_mib:>9,.0f} MiB   "
        f"{config.runtime.dataloader_workers} x {WORKER_RSS_MIB:,.0f} MiB"
    )
    add(f"  {'batches in flight':<28} {sim.host.inflight_mib:>9,.0f} MiB")
    add(f"  {'pinned staging buffers':<28} {sim.host.pinned_mib:>9,.0f} MiB")
    add(f"  {'':<28} {'-' * 13}")
    add(
        f"  {'total':<28} {sim.host.ram_mib:>9,.0f} MiB   "
        f"{sim.host.ram_mib / ram_budget:>5.1%} of {ram_budget / 1024:.0f} GiB"
    )
    add("")
    add(
        f"  supply {sim.host.samples_per_second_supply:>7.1f} samples/s"
        f"   demand {sim.host.samples_per_second_demand:>6.2f} samples/s"
        f"   {'INPUT-BOUND' if sim.host.input_bound else 'compute-bound'}"
    )
    pcie = sim.host.bytes_per_second / GIB
    add(
        f"  host->device {pcie:>6.2f} GiB/s of {sim.hardware.pcie_gbps:.0f} GB/s "
        f"({pcie * 1.074 / sim.hardware.pcie_gbps:.1%})"
    )

    # ---- storage ----------------------------------------------------------
    add("")
    add(_rule("storage (measured file sizes, projected counts)"))
    for name, size in sim.host.storage:
        add(f"  {name:<44} {size:>8.1f} GiB")
    add(f"  {'':<44} {'-' * 12}")
    add(
        f"  {'total':<44} {sim.host.storage_gib:>8.1f} GiB   "
        f"{sim.host.storage_gib / sim.hardware.storage_gib:.0%} of "
        f"{sim.hardware.storage_gib:.0f} GiB"
    )

    # ---- wall clock -------------------------------------------------------
    add("")
    add(_rule("wall clock (calibrated)"))
    add(
        f"  {'scenario':<14} {'MFU':>6} {'TFLOPS':>8} {'s/sample':>9} "
        f"{'s/step':>8} {'run':>9}"
    )
    samples_per_step = (
        config.optim.per_device_batch_size * config.optim.gradient_accumulation_steps
    )
    for model in sorted(sim.throughput, key=lambda m: m.mfu):
        hours = model.hours(sim.total_samples)
        add(
            f"  {model.label:<14} {model.mfu:>6.1%} "
            f"{model.achieved_tflops:>8.1f} {model.seconds_per_sample:>9.3f} "
            f"{model.seconds_per_sample * samples_per_step:>8.2f} "
            f"{hours:>7.1f} h"
        )
    add("")
    if sim.calibration is not None:
        cal = sim.calibration
        add(f"  anchored to {cal.run}")
        add(
            f"    measured  {cal.seconds_per_sample:.3f} s/sample on "
            f"{cal.hardware.name}"
        )
        add(
            f"    modelled  {cal.flops_per_sample / 1e12:.2f} TFLOP/sample "
            f"-> {cal.achieved_tflops:.1f} TFLOPS = {cal.mfu:.1%} MFU"
        )
        add(
            f"    VRAM      predicted {cal.predicted_vram_mib:,.0f} MiB vs "
            f"measured {cal.peak_vram_mib:,.0f} MiB "
            f"(residual {cal.vram_residual:.2f}x)"
        )
        add("")
        for line in _wrap(
            "That measured run used batch 1 and NF4, both of which the target run "
            "drops, so its efficiency is treated as the floor of the range rather "
            "than its centre.",
            4,
        ):
            add(f"    {line}" if not line.startswith("    ") else line)
    else:
        add("  not anchored to any completed run - brackets only")

    # ---- findings ---------------------------------------------------------
    add("")
    add(_rule("conclusions"))
    for severity, text in sim.findings():
        tag = _SEVERITY.get(severity, severity)
        body = _wrap(text, 11)
        add(f"  {tag}  {body[0]}")
        for line in body[1:]:
            add(f"  {line}")
        add("")
    add("=" * _WIDTH)
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="fa-ocr-simulate",
        description=(
            "Project VRAM, RAM, disk and wall-clock for a training run without "
            "running it."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="training config to simulate (default: the package default)",
    )
    parser.add_argument(
        "--preset",
        default="a100_40gb",
        choices=sorted(HARDWARE_PRESETS),
        help="hardware to project onto (default: a100_40gb)",
    )
    parser.add_argument(
        "--calibrate-from",
        type=Path,
        default=Path("data/training/lora_1000"),
        help=(
            "directory of a completed run holding training_summary.json, used to "
            "anchor throughput (default: data/training/lora_1000)"
        ),
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=Path("configs/lora_config.yaml"),
        help="config that the calibration run used (default: configs/lora_config.yaml)",
    )
    parser.add_argument(
        "--calibration-preset",
        default="rtx_3050ti",
        choices=sorted(HARDWARE_PRESETS),
        help="hardware the calibration run used (default: rtx_3050ti)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="skip calibration and project from the efficiency brackets alone",
    )
    parser.add_argument(
        "--pixel-available",
        type=int,
        default=None,
        help=(
            "full size of the Persian Pixel page set, when the local copy is a "
            "partial download (the complete sample_type=page set is ~31000)"
        ),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the projection as JSON to this path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the simulation from the command line.

    Args:
        argv: Argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` on success, ``2`` on a configuration or corpus error, and ``1`` if
        the projection found a blocker - so a script can gate on it.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    hardware = HARDWARE_PRESETS[args.preset]

    try:
        config = load_training_config(args.config)
        profile = profile_corpus(config, pixel_available=args.pixel_available)
    except (ConfigError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    calibration: Calibration | None = None
    if not args.no_calibrate:
        try:
            baseline = load_training_config(args.calibration_config)
            calibration = calibrate(
                args.calibrate_from,
                hardware=HARDWARE_PRESETS[args.calibration_preset],
                config=baseline,
                profile=profile_corpus(baseline),
            )
        except (ConfigError, FileNotFoundError, OSError) as exc:
            LOGGER.warning("not calibrating: %s", exc)

    simulation = simulate(config, hardware, profile=profile, calibration=calibration)
    print(render(simulation))
    if args.json is not None:
        write_json(args.json, simulation.to_dict())
        print(f"wrote {args.json}")
    return 1 if any(s == "blocker" for s, _ in simulation.findings()) else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
