"""Model loading, LoRA injection, and the loss head that makes 4 GB possible.

The base checkpoint is four parts, and only two of them are trained:

===================  ========  ==========================================
``visual``             466 M   NaViT-style tower, 27 layers - **frozen**
``mlp_AR``              26 M   vision-to-text projector - **frozen**
``model``            360.7 M   Ernie4.5 decoder, 18 layers, plus the
                               103,424 x 1024 input embedding - adapted
``lm_head``          105.9 M   103,424-wide output projection - adapted
===================  ========  ==========================================

Two details of that table drive most of this module.

**The vision tower shares projection names with the decoder.** Both use
``q_proj``, ``k_proj`` and ``v_proj``. PEFT matches ``target_modules`` by suffix,
so the seven names in the config would inject 27 layers' worth of adapters into
the tower we just froze - silently, since nothing checks afterwards. So targets
are resolved to **full module paths** here and screened by prefix, and
:func:`assert_frozen` verifies the result rather than trusting it.

**The output projection is the largest tensor in the run, not the largest
weight.** The upstream forward does ``logits = self.lm_head(hidden_states)`` over
every position and then ``logits.float()``: at 1,600 positions that is 331 MB in
bf16 and 662 MB upcast, plus a same-sized gradient - more than the quantised
weights of the whole model. Only the answer positions carry a label, so
:class:`SlicedLossModel` projects only those, in row-chunks, recomputing each
chunk in backward. That makes the cost of the loss independent of how long the
answer is, which is what lets a 1,525-token transcription train on this card at
all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import LoraConfig as LoraSection
from .config import ModelConfig, OptimConfig, TrainingConfig
from .encoding import IGNORE_INDEX

__all__ = [
    "DECODER_PREFIX",
    "FROZEN_PREFIXES",
    "EvalModel",
    "LoadedModel",
    "SlicedLossModel",
    "assert_frozen",
    "auto_model_class",
    "load_eval_model",
    "load_training_model",
    "lora_target_paths",
    "parameter_groups",
    "resolve_dtype",
    "resolve_optimizer",
    "unwrap_base",
]

LOGGER = logging.getLogger(__name__)

#: Module path prefix of the text decoder. Everything LoRA touches lives under
#: this, ``model.embed_tokens``, or ``lm_head``.
DECODER_PREFIX: Final[str] = "model.layers."

#: Submodules that must stay frozen. The tower is frozen because 466 M
#: parameters will not fit an optimizer state here, and the projector because
#: adapting it while the tower is frozen just relearns a fixed basis.
FROZEN_PREFIXES: Final[tuple[str, ...]] = ("visual", "mlp_AR")

#: Path of the input embedding, adapted when ``lora.adapt_embeddings`` is set.
EMBED_PATH: Final[str] = "model.embed_tokens"

#: Path of the output projection, adapted when ``lora.adapt_lm_head`` is set.
HEAD_PATH: Final[str] = "lm_head"


def resolve_dtype(model: ModelConfig) -> torch.dtype:
    """Pick the compute dtype for this GPU.

    ``bfloat16`` needs compute capability 8.0. Below that it is emulated and
    slower than ``float16``, so the probe picks by capability rather than
    defaulting to one everywhere. This card is 8.6, so ``auto`` gives bf16 -
    which matters beyond speed: bf16 has fp32's exponent range, so the loss
    cannot silently overflow to ``inf`` the way fp16 does without a grad scaler.

    Args:
        model: Model configuration section.

    Returns:
        The dtype to load weights and run compute in.
    """
    if model.dtype != "auto":
        return getattr(torch, model.dtype)
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def bnb_available() -> bool:
    """Whether bitsandbytes can be imported.

    Returns:
        ``True`` if the module is installed. Import is not attempted here -
        loading bitsandbytes initialises CUDA, which the caller may not want yet.
    """
    return find_spec("bitsandbytes") is not None


def resolve_optimizer(optim: OptimConfig) -> str:
    """Resolve the optimizer name Trainer should use.

    Args:
        optim: Optimizer configuration section.

    Returns:
        ``paged_adamw_8bit`` when bitsandbytes is installed, otherwise
        ``adamw_torch_fused``. Paged state is chosen not for its size - the
        adapter is only ~9 M parameters, so 8-bit state saves tens of MB - but
        because paging lets the optimizer spill to host memory instead of
        aborting the run when a long page transiently peaks.
    """
    if optim.optimizer != "auto":
        return optim.optimizer
    if bnb_available():
        return "paged_adamw_8bit"
    LOGGER.warning(
        "bitsandbytes is not installed; falling back to adamw_torch_fused "
        "(no paged optimizer state)"
    )
    return "adamw_torch_fused"


def quantization_config(model: ModelConfig, dtype: torch.dtype) -> Any | None:
    """Build the bitsandbytes configuration, or ``None`` for full precision.

    Only ``nn.Linear`` layers are quantised, so the 103,424 x 1024 input
    embedding stays in ``dtype`` regardless. ``lm_head`` is skipped explicitly:
    it is the layer the adapter steers most directly, and 4-bit rounding on the
    logits costs accuracy exactly where the metadata-block digits are decided.

    Args:
        model: Model configuration section.
        dtype: Compute dtype, used for dequantised matmuls.

    Returns:
        A ``BitsAndBytesConfig``, or ``None`` when quantisation is off or
        unavailable.
    """
    if model.quantization == "none":
        return None
    if not bnb_available():
        LOGGER.warning(
            "quantization=%s requested but bitsandbytes is not installed; "
            "loading in %s instead",
            model.quantization,
            dtype,
        )
        return None
    from transformers import BitsAndBytesConfig

    skip = [HEAD_PATH]
    if not model.quantize_vision:
        skip.extend(FROZEN_PREFIXES)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model.quantization,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip,
    )


def freeze_vision(model: nn.Module) -> tuple[int, int]:
    """Freeze the vision tower and the projector.

    Freezing has a second effect worth stating, because the pixel budget depends
    on it: with every visual parameter frozen and ``pixel_values`` not requiring
    grad, the tower's forward builds no autograd graph at all. Its activations
    are freed as it goes, so a 1,260-token page costs transient memory rather
    than resident memory, and the tower can stay in bf16 while only the decoder
    is a candidate for quantisation.

    Args:
        model: The loaded base model.

    Returns:
        ``(frozen_parameters, remaining_trainable_parameters)``.
    """
    frozen = 0
    for name, parameter in model.named_parameters():
        if name.startswith(FROZEN_PREFIXES):
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return frozen, trainable


def assert_frozen(model: nn.Module) -> None:
    """Verify that nothing trainable was injected into the frozen submodules.

    Called after LoRA injection, because that is when it can go wrong: the tower
    and the decoder share projection names, so one missed path prefix turns 466 M
    frozen parameters into 466 M adapted ones and the run OOMs with no
    indication why.

    Args:
        model: The model after adapters have been applied, wrapped or not.

    Raises:
        AssertionError: If any parameter under :data:`FROZEN_PREFIXES` requires
            grad.
    """
    leaked = [
        name
        for name, parameter in unwrap_base(model).named_parameters()
        if parameter.requires_grad and _strip_peft(name).startswith(FROZEN_PREFIXES)
    ]
    if leaked:
        raise AssertionError(
            f"{len(leaked)} parameter(s) in the frozen vision path require grad, "
            f"e.g. {leaked[:3]}; LoRA targets matched by suffix instead of path"
        )


def _strip_peft(name: str) -> str:
    """Remove the wrapper segments PEFT inserts into parameter paths.

    Args:
        name: A parameter path, possibly containing ``base_model``,
            ``model`` wrappers and ``base_layer`` segments.

    Returns:
        The path as it reads on the unwrapped model, so prefix checks against
        :data:`FROZEN_PREFIXES` stay meaningful after injection.
    """
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace(".base_layer", "")


def lora_target_paths(model: nn.Module, lora: LoraSection) -> list[str]:
    """Resolve configured module suffixes to full module paths.

    Full paths rather than suffixes, screened by :data:`DECODER_PREFIX`: the
    vision tower's attention uses the same ``q_proj``/``k_proj``/``v_proj``
    names, and PEFT's suffix matching cannot tell them apart.

    Args:
        model: The loaded base model.
        lora: LoRA configuration section.

    Returns:
        Module paths to adapt, decoder projections first, then the embedding and
        output head if enabled.

    Raises:
        ValueError: If no decoder module matched, which means the checkpoint's
            layout changed and the configured names are stale.
    """
    wanted = set(lora.target_modules)
    paths = [
        name
        for name, module in model.named_modules()
        if name.startswith(DECODER_PREFIX)
        and isinstance(module, nn.Linear)
        and name.rsplit(".", 1)[-1] in wanted
    ]
    if not paths:
        raise ValueError(
            f"none of the configured target_modules {sorted(wanted)} matched a "
            f"Linear under {DECODER_PREFIX!r}; the checkpoint layout has changed"
        )
    if lora.adapt_embeddings:
        paths.append(EMBED_PATH)
    if lora.adapt_lm_head:
        paths.append(HEAD_PATH)
    return paths


def _chunk_loss(
    head: nn.Module, hidden: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Summed cross-entropy over one chunk of supervised positions.

    Args:
        head: The output projection.
        hidden: Hidden states of the chunk, shape ``(rows, H)``.
        target: Target ids of the chunk, shape ``(rows,)``.

    Returns:
        A scalar: the sum, not the mean, so chunks can be added before one
        division by the true token count.
    """
    # float() matches the upstream loss exactly: at bf16's 8-bit mantissa a
    # 103,424-way log-softmax loses enough precision to change the gradient.
    return functional.cross_entropy(head(hidden).float(), target, reduction="sum")


class SlicedLossModel(nn.Module):
    """Computes the language-model loss over supervised positions only.

    Wraps the PEFT model rather than subclassing it, so the adapter's module
    tree - and therefore the key names ``save_pretrained`` writes - stays exactly
    what a plain PEFT loader expects at inference time.

    The upstream ``forward`` computes full-vocabulary logits unconditionally, so
    the head is swapped for a passthrough for the duration of the call and
    restored in a ``finally``. That is the whole trick: the base model returns
    hidden states in the ``logits`` field, cheaply, and the real projection then
    runs here over the ~255-600 positions that actually carry a label instead of
    all 1,528.

    Attributes:
        base: The wrapped PEFT model.
        chunk_rows: Supervised positions projected per chunk.
    """

    #: Trainer inspects this to decide whether to pass ``num_items_in_batch``
    #: into ``forward``. The loss here is already normalised per token, so the
    #: extra kwarg would double-normalise it.
    accepts_loss_kwargs: bool = False

    #: Trainer refuses to train a model that is quantized but carries no adapter,
    #: and it recognises an adapter in one of two ways: the model *is* a
    #: ``PeftModel``, or it declares this attribute. Wrapping rather than
    #: subclassing costs us the first, so we assert the second - which is simply
    #: true here, since ``build_model`` attaches the adapter before this wraps it.
    #: Without it the run dies at ``Trainer.__init__`` with "You cannot perform
    #: fine-tuning on purely quantized models". Blast radius is that one check;
    #: nothing in transformers, accelerate or peft reads the attribute elsewhere.
    _hf_peft_config_loaded: bool = True

    def __init__(self, base: nn.Module, *, chunk_rows: int) -> None:
        """Wrap a model.

        Args:
            base: The PEFT-wrapped base model.
            chunk_rows: Supervised positions per chunk; from
                ``image.loss_chunk_rows``.

        Raises:
            ValueError: If the output projection cannot be located, which would
                mean the slicing silently did nothing.
        """
        super().__init__()
        self.base = base
        self.chunk_rows = max(1, chunk_rows)
        self._owner, self._attr = _locate_head(base)
        self._head: nn.Module = getattr(self._owner, self._attr)

    @property
    def head(self) -> nn.Module:
        """The real output projection."""
        return self._head

    def __getattr__(self, name: str) -> Any:
        """Fall back to the wrapped model for anything this wrapper lacks.

        ``Trainer`` inspects a model with a long list of ``getattr`` probes -
        ``config``, ``is_loaded_in_4bit``, ``hf_device_map``,
        ``generation_config``, ``warnings_issued`` - and takes their absence as
        meaningful. Left opaque, the wrapper reads as an ordinary fp32 module, and
        the first consequence is Trainer trying to ``.to(device)`` a 4-bit model
        that is already placed. Delegating makes the wrapper transparent to those
        probes while ``forward`` stays overridden, which is the only behaviour it
        exists to change.

        Args:
            name: Attribute being looked up.

        Returns:
            The attribute from the wrapped model.

        Raises:
            AttributeError: If neither this module nor the wrapped one has it.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            # _modules directly, not self.base: during __init__ the attribute may
            # not exist yet, and going through self.base would recurse.
            base = self.__dict__.get("_modules", {}).get("base")
            if base is None:
                raise
            return getattr(base, name)

    def forward(
        self,
        *,
        labels: torch.Tensor | None = None,
        **inputs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the model and compute the sliced loss.

        Args:
            labels: Targets with prompt and padding at :data:`IGNORE_INDEX`,
                shape ``(B, L)``. When ``None``, no loss is computed and full
                logits are returned, which is what generation needs.
            **inputs: Remaining model inputs, forwarded unchanged.

        Returns:
            ``{"loss": ..., "logits": ...}`` in training, or the base model's own
            output when ``labels`` is ``None``.
        """
        if labels is None:
            return self.base(**inputs)

        identity = nn.Identity()
        setattr(self._owner, self._attr, identity)
        try:
            outputs = self.base(**inputs, use_cache=False)
        finally:
            # A restore in `finally` and not after the call: an OOM here is
            # recoverable at the step level, but leaving a passthrough installed
            # would make every later generation produce 1024-wide "logits".
            setattr(self._owner, self._attr, self._head)

        hidden = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        return {"loss": self._sliced_loss(hidden, labels), "logits": hidden}

    def _sliced_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over labelled positions, projected in chunks.

        Args:
            hidden: Decoder hidden states, shape ``(B, L, H)``.
            labels: Targets, shape ``(B, L)``.

        Returns:
            A scalar loss, mean over supervised tokens - the same normalisation
            the upstream loss uses, so numbers stay comparable to a full-logit
            run.
        """
        # Shift once, then gather: predicting token t+1 from position t means the
        # last position has no target and the first label is never predicted.
        shifted = hidden[:, :-1, :]
        targets = labels[:, 1:]
        supervised = targets != IGNORE_INDEX
        rows = int(supervised.sum())
        if rows == 0:
            # Can only happen if a collated batch is entirely padding. Return a
            # zero that is still attached to the graph, or the optimizer step
            # would fail on a leaf-less loss.
            return (hidden.float() * 0.0).sum()

        flat_hidden = shifted[supervised]
        flat_targets = targets[supervised]
        total = flat_hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, rows, self.chunk_rows):
            stop = start + self.chunk_rows
            # Checkpointed so the chunk's (rows x 103,424) float32 logits are
            # freed immediately and rebuilt during backward. Peak cost becomes
            # one chunk rather than the whole answer.
            total = total + checkpoint(
                _chunk_loss,
                self._head,
                flat_hidden[start:stop],
                flat_targets[start:stop],
                use_reentrant=False,
            )
        return total / rows

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate generation to the wrapped model.

        Returns:
            Whatever the base model's ``generate`` returns.
        """
        return self.base.generate(*args, **kwargs)

    def save_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate adapter saving to the wrapped PEFT model.

        Returns:
            Whatever the base model's ``save_pretrained`` returns.
        """
        return self.base.save_pretrained(*args, **kwargs)


def unwrap_base(model: nn.Module) -> nn.Module:
    """Return the PEFT model underneath any :class:`SlicedLossModel` wrapper.

    Every path-matching helper here reads parameter names against the module tree
    as PEFT lays it out. The wrapper inserts a ``base.`` segment ahead of all of
    them, and the failure that causes is silent in both directions: a prefix test
    for the frozen tower stops matching, so :func:`assert_frozen` passes on a
    model that leaks; and a test for the embedding and head stops matching, so
    :func:`parameter_groups` returns one group and the differential learning rate
    is quietly not applied. Unwrapping first removes the whole class of bug, which
    growing a list of prefixes to strip would not.

    Enumerating from the returned model loses nothing: the wrapper's only other
    submodule reference is the output projection, which already lives inside the
    PEFT tree, and ``named_parameters`` deduplicates by identity.

    Args:
        model: Either a wrapped model or an already-unwrapped one.

    Returns:
        The innermost non-wrapper module. Returned unchanged if it is not
        wrapped, so callers need not know which they hold.
    """
    while isinstance(model, SlicedLossModel):
        model = model.base
    return model


def _locate_head(model: nn.Module) -> tuple[nn.Module, str]:
    """Find the module that owns the output projection.

    PEFT inserts wrapper levels, so the owner is found by search rather than by a
    hardcoded attribute path.

    Args:
        model: The PEFT-wrapped model.

    Returns:
        ``(owner_module, attribute_name)`` such that
        ``getattr(owner, attr)`` is the output projection.

    Raises:
        ValueError: If no attribute named :data:`HEAD_PATH` is found.
    """
    for name, _module in model.named_modules():
        if name.rsplit(".", 1)[-1] == HEAD_PATH:
            parent_path = name.rsplit(".", 1)[0] if "." in name else ""
            owner = model.get_submodule(parent_path) if parent_path else model
            return owner, HEAD_PATH
    raise ValueError(f"no {HEAD_PATH!r} submodule found; cannot slice the loss")


def parameter_groups(model: nn.Module, optim: OptimConfig) -> list[dict[str, Any]]:
    """Split trainable parameters into differential learning-rate groups.

    The embedding and output head see every token of every sample, whereas an
    attention adapter in layer 9 sees a residual stream. At one shared rate the
    head moves far enough in the first few hundred steps to drift the base
    model's output distribution, which shows up as the loss falling quickly and
    then the character error rate getting *worse*. A lower rate on those two
    keeps the Persian glyph and digit coverage they are there to extend without
    destabilising the rest.

    Args:
        model: The model after LoRA injection, wrapped or not.
        optim: Optimizer configuration section.

    Returns:
        Parameter groups for the optimizer, each with its own learning rate.
        Always two groups on the default config; one only if the embedding and
        head were left unadapted.

    Raises:
        ValueError: If no parameter requires grad, which means injection failed.
    """
    fast: list[nn.Parameter] = []
    slow: list[nn.Parameter] = []
    for name, parameter in unwrap_base(model).named_parameters():
        if not parameter.requires_grad:
            continue
        stripped = _strip_peft(name)
        target = slow if stripped.startswith((EMBED_PATH, HEAD_PATH)) else fast
        target.append(parameter)
    if not fast and not slow:
        raise ValueError("no trainable parameters; LoRA injection produced nothing")
    groups: list[dict[str, Any]] = []
    if fast:
        groups.append(
            {
                "params": fast,
                "lr": optim.learning_rate,
                "weight_decay": optim.weight_decay,
                "name": "lora_projections",
            }
        )
    if slow:
        groups.append(
            {
                "params": slow,
                "lr": optim.embed_head_learning_rate,
                "weight_decay": optim.weight_decay,
                "name": "lora_embed_head",
            }
        )
    return groups


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Everything the training loop needs from the model layer.

    Attributes:
        model: The wrapped, adapter-injected model, ready for ``Trainer``.
        processor: The base model's ``AutoProcessor``.
        dtype: Resolved compute dtype.
        trainable: Number of parameters that require grad.
        total: Number of parameters in the model.
        quantized: Whether 4-bit quantisation is active.
        weight_mib: Measured device memory held by weights, in MiB.
    """

    model: SlicedLossModel
    processor: Any
    dtype: torch.dtype
    trainable: int
    total: int
    quantized: bool
    weight_mib: float

    def describe(self) -> str:
        """One-line summary for the startup banner.

        Returns:
            A human-readable description of what was loaded.
        """
        precision = "nf4" if self.quantized else str(self.dtype).replace("torch.", "")
        return (
            f"{precision}, weights {self.weight_mib:.0f} MiB, "
            f"trainable {self.trainable / 1e6:.2f}M / {self.total / 1e6:.0f}M "
            f"({100 * self.trainable / max(1, self.total):.2f}%)"
        )


def auto_model_class(model: ModelConfig) -> Any:
    """Pick the ``AutoModel*`` class that loads this profile's weights.

    The original checkpoint publishes only an ``AutoModelForCausalLM`` entry in
    its ``auto_map`` and needs the repository's own modeling file, so it loads as
    causal language modelling with ``trust_remote_code`` on. PaddleOCR-VL-1.6 is
    native to transformers 5 and its card loads it as image-text-to-text with no
    remote code at all. ``model.auto_class`` states which of the two a profile
    is, so a wrong choice fails at startup instead of silently loading a
    different class than the one the adapter will later be attached to.

    Args:
        model: The profile's ``model`` section.

    Returns:
        The transformers auto class to call ``from_pretrained`` on.
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    if model.auto_class == "image_text_to_text":
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def _load_base(config: TrainingConfig) -> tuple[Any, Any, torch.dtype, bool]:
    """Load the processor and the raw base model.

    Shared by the training and evaluation loaders on purpose. Dtype, quantisation
    mode, revision and attention implementation all have to match between the two,
    or a measured character error rate is not a measurement of the trained
    adapter: an adapter tuned against a 4-bit tower and scored against a bf16 one
    meets features it never saw. Two copies of this call would let that drift in
    silently.

    Args:
        config: Loaded training configuration.

    Returns:
        ``(base_model, processor, dtype, quantized)``.
    """
    from transformers import AutoProcessor

    dtype = resolve_dtype(config.model)
    quant = quantization_config(config.model, dtype)
    LOGGER.info(
        "loading %s (%s, %s)",
        config.model.model_id,
        str(dtype).replace("torch.", ""),
        config.model.quantization if quant is not None else "unquantized",
    )
    processor = AutoProcessor.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    base = auto_model_class(config.model).from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        dtype=dtype,
        attn_implementation=config.model.attn_implementation,
        quantization_config=quant,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    return base, processor, dtype, quant is not None


def load_training_model(config: TrainingConfig) -> LoadedModel:
    """Load the base model, freeze the tower, and inject LoRA adapters.

    Args:
        config: Loaded training configuration.

    Returns:
        The assembled model plus the facts the banner and the loop need.

    Raises:
        AssertionError: If adapters leaked into the frozen vision path.
        ValueError: If the configured LoRA targets match nothing.
    """
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    base, processor, dtype, quantized = _load_base(config)
    base.config.use_cache = False

    frozen, _ = freeze_vision(base)
    targets = lora_target_paths(base, config.lora)
    LOGGER.info(
        "frozen %.0fM visual parameters; adapting %d module(s)",
        frozen / 1e6,
        len(targets),
    )

    peft_model = get_peft_model(
        base,
        PeftLoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    assert_frozen(peft_model)

    if config.optim.gradient_checkpointing:
        _enable_checkpointing(peft_model)

    model = SlicedLossModel(peft_model, chunk_rows=config.image.loss_chunk_rows)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    weight_mib = (
        torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    )
    return LoadedModel(
        model=model,
        processor=processor,
        dtype=dtype,
        trainable=trainable,
        total=total,
        quantized=quantized,
        weight_mib=weight_mib,
    )


@dataclass(frozen=True, slots=True)
class EvalModel:
    """A model ready to generate, with or without the trained adapter.

    Attributes:
        model: The base model, adapter applied if one was given.
        processor: The base model's ``AutoProcessor``.
        dtype: Resolved compute dtype.
        quantized: Whether 4-bit quantisation is active.
        weight_mib: Measured device memory held by weights, in MiB.
        adapter_dir: Adapter that was applied, or ``None`` for the baseline.
    """

    model: nn.Module
    processor: Any
    dtype: torch.dtype
    quantized: bool
    weight_mib: float
    adapter_dir: Path | None = None

    def describe(self) -> str:
        """One-line summary for the report header.

        Returns:
            A human-readable description of what was loaded and scored.
        """
        precision = "nf4" if self.quantized else str(self.dtype).replace("torch.", "")
        which = self.adapter_dir.name if self.adapter_dir else "base model only"
        return f"{precision}, weights {self.weight_mib:.0f} MiB, adapter: {which}"


def load_eval_model(
    config: TrainingConfig, *, adapter_dir: Path | None = None
) -> EvalModel:
    """Load the base model for generation, optionally applying a trained adapter.

    Differs from :func:`load_training_model` in three ways, each of which matters
    for a number to be worth reporting:

    * ``use_cache`` is on and gradient checkpointing is off. The two are mutually
      exclusive upstream, and with checkpointing on transformers silently disables
      the cache - turning a 1,024-token greedy decode from seconds into minutes.
    * No :class:`SlicedLossModel`. The sliced loss exists to avoid materialising
      full logits during training; generation needs exactly those logits, one
      position at a time, which the head already gives cheaply.
    * Adapters load frozen. ``is_trainable=False`` also merges nothing, so the
      scored model is the same composition the training loop produced.

    Quantisation follows the config, which is what keeps a baseline comparable to
    a trained score - see :func:`_load_base`.

    Args:
        config: Loaded training configuration.
        adapter_dir: Directory holding ``adapter_model.safetensors`` and
            ``adapter_config.json``. ``None`` scores the untrained base model,
            which is the baseline the adapter has to beat.

    Returns:
        The loaded model and processor.

    Raises:
        FileNotFoundError: If ``adapter_dir`` is given but holds no adapter.
    """
    if adapter_dir is not None and not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"{adapter_dir} has no adapter_config.json; train first, or pass "
            f"--no-adapter to score the base model"
        )

    base, processor, dtype, quantized = _load_base(config)
    base.config.use_cache = True
    # Both, and the second is the one that counts: `generate` copies use_cache out
    # of the *generation* config into the forward kwargs, so a False there wins over
    # config.use_cache. This model ships generation_config.json with use_cache
    # false, and the cost of missing it is not subtle - measured on one validation
    # page, 0.795 s per token against a 1,273-token prompt, because every decode
    # step re-forwards the whole prompt and re-runs the vision tower. With the
    # cache on the same decode is ~40x faster, which is the difference between a
    # baseline that takes minutes and one that takes most of a day.
    if base.generation_config is not None:
        base.generation_config.use_cache = True
    model: nn.Module = base
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        LOGGER.info("applied adapter from %s", adapter_dir)
    model.eval()
    weight_mib = (
        torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    )
    return EvalModel(
        model=model,
        processor=processor,
        dtype=dtype,
        quantized=quantized,
        weight_mib=weight_mib,
        adapter_dir=adapter_dir,
    )


def _enable_checkpointing(model: nn.Module) -> None:
    """Turn on non-reentrant gradient checkpointing for the decoder.

    ``use_reentrant=False`` is required, not preferred. The reentrant
    implementation decides whether to build a backward graph from whether its
    *inputs* require grad, and with a frozen quantised base the decoder's inputs
    do not - the trainable parameters are inside the checkpointed block. It would
    therefore return no gradients at all, and the run would train to a loss that
    never moves.

    Checkpointing is applied to the decoder only. The vision tower is frozen and
    builds no graph, so checkpointing it would pay recomputation for activations
    that were never stored.

    Args:
        model: The PEFT-wrapped model.
    """
    decoder = getattr(model.get_base_model(), "model", None)
    target = decoder if hasattr(decoder, "gradient_checkpointing_enable") else model
    target.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if not getattr(target, "gradient_checkpointing", False):
        LOGGER.warning(
            "%s did not report gradient checkpointing as enabled; activation "
            "memory will be higher than the configured budget assumes",
            type(target).__name__,
        )
    else:
        LOGGER.info("gradient checkpointing on %s (non-reentrant)", type(target).__name__)
