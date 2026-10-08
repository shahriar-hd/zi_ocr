"""Typed loader for ``configs/lora_config.yaml``.

CLAUDE.md 8 forbids hardcoded constants, and a training run is where that rule
earns its keep: a learning rate or pixel budget buried in a module is a number
nobody can audit after the fact. Every knob the local LoRA run turns arrives
here as a frozen, validated dataclass, and loading is strict - an unknown key is
an error, so a typo fails at startup instead of after two hours of training.

The validation is deliberately narrow-minded about physically impossible
values. A pixel budget that is not a multiple of the processor's token area, a
LoRA alpha of zero, a validation holdout larger than the corpus: each of those
produces a confusing failure hours downstream if it is allowed through, so each
is rejected here with the dotted path of the offending key.

Machine-specific paths do not belong in the file. Those come from the
environment, and :func:`load_training_config` overlays them once the file has
been parsed. The full precedence chain is::

    CLI flag  >  environment  >  YAML  >  dataclass default
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Final

import yaml

from ..config.base import ConfigError
from ..dataset.common.io_utils import DEFAULT_SEED, resolve_repo_path

__all__ = [
    "AUTO_CLASS_CHOICES",
    "DEFAULT_TRAINING_CONFIG_PATH",
    "DTYPE_CHOICES",
    "ENV_PREFIX",
    "MERGE_SIZE",
    "OVERSIZE_POLICIES",
    "PATCH_SIZE",
    "PIXELS_PER_TOKEN",
    "QUANTIZATION_CHOICES",
    "DataConfig",
    "DriveConfig",
    "EvalConfig",
    "ImageConfig",
    "LoraConfig",
    "ModelConfig",
    "OptimConfig",
    "PathsConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "load_training_config",
]

#: Default location of the training configuration.
DEFAULT_TRAINING_CONFIG_PATH: Final[str] = "configs/lora_config.yaml"

#: Prefix of every environment override, e.g. ``FA_OCR_TRAIN_OUTPUT_DIR``.
ENV_PREFIX: Final[str] = "FA_OCR_TRAIN_"

#: Vision patch edge, in pixels, as fixed by the base model's preprocessor.
PATCH_SIZE: Final[int] = 14

#: How many patches per edge the projector merges into one image token.
MERGE_SIZE: Final[int] = 2

#: Pixels one image token covers: ``(14 * 2) ** 2``. Every pixel budget in this
#: config is validated against this number, because a budget that is not a
#: whole number of tokens is silently rounded down by the processor - the run
#: would then train at a resolution nobody chose.
PIXELS_PER_TOKEN: Final[int] = (PATCH_SIZE * MERGE_SIZE) ** 2

#: Which ``AutoModel*`` class the loader uses. ``causal_lm`` is the remote-code
#: path: the original checkpoint's ``auto_map`` publishes only an
#: ``AutoModelForCausalLM`` entry and needs the repository's own modeling file.
#: ``image_text_to_text`` is how the PaddleOCR-VL-1.6 card loads that release,
#: which transformers 5 carries natively - passing ``trust_remote_code`` there
#: would reach for remote code that is not what the card uses.
AUTO_CLASS_CHOICES: Final[frozenset[str]] = frozenset({"causal_lm", "image_text_to_text"})

#: Weight precisions the loader accepts. ``auto`` probes compute capability.
DTYPE_CHOICES: Final[frozenset[str]] = frozenset(
    {"auto", "bfloat16", "float16", "float32"}
)

#: Quantisation modes. ``nf4`` needs bitsandbytes; the model layer falls back
#: to ``none`` with a warning rather than dying if the import fails.
QUANTIZATION_CHOICES: Final[frozenset[str]] = frozenset({"none", "nf4"})

#: What to do with a record that will not fit the sequence budget.
OVERSIZE_POLICIES: Final[frozenset[str]] = frozenset(
    {"shrink_image", "truncate_text", "skip"}
)

#: Optimiser names accepted verbatim by :class:`transformers.TrainingArguments`,
#: plus ``auto`` which picks by whether bitsandbytes is importable.
_OPTIMIZERS: Final[frozenset[str]] = frozenset(
    {"auto", "paged_adamw_8bit", "adamw_8bit", "adamw_torch_fused", "adamw_torch"}
)

_SCHEDULERS: Final[frozenset[str]] = frozenset(
    {
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    }
)

_LORA_BIAS: Final[frozenset[str]] = frozenset({"none", "all", "lora_only"})


def _section(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Fetch one top-level section, defaulting to empty rather than failing.

    A missing section means "take the dataclass defaults", which is what makes
    a minimal config file legal. A section present but not a mapping is a
    mistake worth naming.

    Args:
        payload: The decoded YAML mapping.
        key: Section name.

    Returns:
        The section's mapping, or an empty one when absent.

    Raises:
        ConfigError: If the key is present but does not hold a mapping.
    """
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key}: expected a mapping, got {type(value).__name__}")
    return value


def _defaults(cls: type) -> dict[str, Any]:
    """Field defaults of a dataclass, read from its field metadata.

    Not ``getattr(cls, name)``: ``slots=True`` makes dataclasses rebuild the
    class with slot descriptors and drop the class-level defaults, so attribute
    access on the class raises. The field metadata survives that rebuild.

    Args:
        cls: A dataclass whose every field carries a default.

    Returns:
        Mapping of field name to default value.
    """
    return {f.name: f.default for f in fields(cls)}


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Sequence[str], context: str
) -> None:
    """Fail on any key the dataclass cannot consume.

    Strictness is the point: a silently ignored ``max_pixel`` would train the
    whole corpus at the default resolution and give no hint why.

    Args:
        payload: Mapping as read from YAML.
        allowed: Keys the section understands.
        context: Dotted path used in the error message.

    Raises:
        ConfigError: If the mapping carries keys outside ``allowed``.
    """
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise ConfigError(
            f"{context}: unknown key(s) {unknown}; expected one of {sorted(allowed)}"
        )


def _as_int(
    payload: Mapping[str, Any],
    key: str,
    default: int,
    context: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read an integer, enforcing optional inclusive bounds."""
    raw = payload.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"{context}.{key}: expected an integer, got {raw!r}")
    if isinstance(raw, float) and not raw.is_integer():
        raise ConfigError(f"{context}.{key}: expected a whole number, got {raw!r}")
    value = int(raw)
    if minimum is not None and value < minimum:
        raise ConfigError(f"{context}.{key}: must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{context}.{key}: must be <= {maximum}, got {value}")
    return value


def _as_float(
    payload: Mapping[str, Any],
    key: str,
    default: float,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Read a float, enforcing optional inclusive bounds."""
    raw = payload.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"{context}.{key}: expected a number, got {raw!r}")
    value = float(raw)
    if minimum is not None and value < minimum:
        raise ConfigError(f"{context}.{key}: must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{context}.{key}: must be <= {maximum}, got {value}")
    return value


def _as_bool(payload: Mapping[str, Any], key: str, default: bool, context: str) -> bool:
    """Read a strict boolean; ``"yes"`` and ``1`` are rejected as ambiguous."""
    raw = payload.get(key, default)
    if not isinstance(raw, bool):
        raise ConfigError(f"{context}.{key}: expected true or false, got {raw!r}")
    return raw


def _as_choice(
    payload: Mapping[str, Any],
    key: str,
    default: str,
    context: str,
    choices: frozenset[str],
) -> str:
    """Read a string constrained to a fixed set of spellings."""
    raw = payload.get(key, default)
    if not isinstance(raw, str):
        raise ConfigError(f"{context}.{key}: expected a string, got {raw!r}")
    value = raw.strip()
    if value not in choices:
        raise ConfigError(
            f"{context}.{key}: expected one of {sorted(choices)}, got {value!r}"
        )
    return value


def _as_str_tuple(
    payload: Mapping[str, Any], key: str, default: tuple[str, ...], context: str
) -> tuple[str, ...]:
    """Read a list of strings, rejecting a bare string as a likely mistake."""
    raw = payload.get(key, default)
    if isinstance(raw, str):
        raise ConfigError(
            f"{context}.{key}: expected a list of strings, got the string {raw!r}"
        )
    if not isinstance(raw, Sequence):
        raise ConfigError(f"{context}.{key}: expected a list, got {raw!r}")
    values = tuple(str(item).strip() for item in raw)
    if any(not item for item in values):
        raise ConfigError(f"{context}.{key}: contains an empty entry")
    return values


def _as_weight_pairs(
    payload: Mapping[str, Any],
    key: str,
    default: tuple[tuple[str, float], ...],
    context: str,
) -> tuple[tuple[str, float], ...]:
    """Read a name-to-weight mapping as an ordered, hashable tuple of pairs.

    A tuple of pairs rather than a dict because the config dataclasses are frozen
    and a dict field would make them unhashable and quietly mutable. Order is
    preserved so the allocation report reads in the order the file declares.
    """
    raw = payload.get(key, default)
    if isinstance(raw, Mapping):
        items: list[tuple[str, Any]] = list(raw.items())
    elif isinstance(raw, Sequence) and not isinstance(raw, str):
        items = []
        for entry in raw:
            is_pair = isinstance(entry, Sequence) and not isinstance(entry, str)
            if is_pair and len(entry) == 2:
                items.append((entry[0], entry[1]))
            else:
                raise ConfigError(
                    f"{context}.{key}: expected 'name: weight' pairs, got {entry!r}"
                )
    else:
        raise ConfigError(
            f"{context}.{key}: expected a mapping of type name to weight, got {raw!r}"
        )
    pairs: list[tuple[str, float]] = []
    for name, weight in items:
        label = str(name).strip()
        if not label:
            raise ConfigError(f"{context}.{key}: contains an empty type name")
        try:
            value = float(weight)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{context}.{key}.{label}: expected a number, got {weight!r}"
            ) from exc
        if value < 0.0:
            raise ConfigError(f"{context}.{key}.{label}: must not be negative")
        pairs.append((label, value))
    if pairs and not any(weight > 0.0 for _, weight in pairs):
        raise ConfigError(
            f"{context}.{key}: every weight is zero, so no crop could be selected"
        )
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Which base model to adapt, and in what precision.

    Attributes:
        model_id: HuggingFace repo of the base model. Defaults to
            PaddleOCR-VL-1.6, which is the supported base: it is what the
            molab profile trains and the only one verified on transformers 5.
        revision: Git revision to pin, so a Hub update cannot silently change
            the weights under a resumed run.
        auto_class: ``AutoModel*`` class that loads the weights; see
            :data:`AUTO_CLASS_CHOICES`. ``image_text_to_text`` by default,
            because that is how transformers 5 loads 1.6 natively. The older
            ``causal_lm`` path exists for the original release, whose
            ``auto_map`` publishes only an ``AutoModelForCausalLM`` entry, and
            the profiles that still want it now say so explicitly.
        trust_remote_code: Whether ``from_pretrained`` may execute the Hub
            repository's own modeling code. ``False`` by default: 1.6 is native
            to transformers, so reaching for remote code would load something
            other than what the model card describes. Only the original
            release needs it, and it is mutually exclusive with
            ``image_text_to_text`` - see :meth:`parse`.
        attn_implementation: Attention kernel passed to ``from_pretrained``.
        dtype: ``auto`` resolves by compute capability at load time.
        quantization: ``none`` or ``nf4``.
        quantize_vision: Whether ``nf4`` also covers the frozen vision tower.
        task_prompt: Instruction placed after the image in the user turn. Taken
            verbatim from the model card - the base model was trained with these
            exact strings, and a paraphrase would train the adapter on a prompt
            no inference caller will ever send.
    """

    model_id: str = "PaddlePaddle/PaddleOCR-VL-1.6"
    revision: str = "main"
    auto_class: str = "image_text_to_text"
    trust_remote_code: bool = False
    attn_implementation: str = "sdpa"
    dtype: str = "auto"
    quantization: str = "nf4"
    quantize_vision: bool = False
    task_prompt: str = "OCR:"

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> ModelConfig:
        """Build the section from YAML.

        Args:
            payload: The ``model`` mapping.

        Returns:
            The validated section.

        Raises:
            ConfigError: On an unknown key, an out-of-range value, or a
                combination of ``auto_class`` and ``trust_remote_code`` that
                names two different loading strategies at once.
        """
        context = "model"
        defaults = _defaults(cls)
        _reject_unknown(payload, defaults, context)
        model_id = str(payload.get("model_id", defaults["model_id"])).strip()
        if not model_id:
            raise ConfigError(f"{context}.model_id: must not be empty")
        auto_class = _as_choice(
            payload,
            "auto_class",
            defaults["auto_class"],
            context,
            AUTO_CLASS_CHOICES,
        )
        trust_remote_code = _as_bool(
            payload, "trust_remote_code", defaults["trust_remote_code"], context
        )
        # The two fields name one decision between them, and the contradictory
        # pairing is silent rather than loud: transformers would honour the
        # remote code and load a class the adapter was never shaped against, so
        # the weights load, training runs, and the result is quietly wrong.
        if auto_class == "image_text_to_text" and trust_remote_code:
            raise ConfigError(
                f"{context}: auto_class 'image_text_to_text' loads the model "
                "natively from transformers, so trust_remote_code must be false. "
                "PaddleOCR-VL-1.6 is native; only the original "
                "PaddlePaddle/PaddleOCR-VL release needs remote code, and that "
                "profile must also set auto_class: causal_lm."
            )
        return cls(
            model_id=model_id,
            revision=str(payload.get("revision", defaults["revision"])).strip() or "main",
            auto_class=auto_class,
            trust_remote_code=trust_remote_code,
            attn_implementation=str(
                payload.get("attn_implementation", defaults["attn_implementation"])
            ).strip(),
            dtype=_as_choice(payload, "dtype", defaults["dtype"], context, DTYPE_CHOICES),
            quantization=_as_choice(
                payload,
                "quantization",
                defaults["quantization"],
                context,
                QUANTIZATION_CHOICES,
            ),
            quantize_vision=_as_bool(payload, "quantize_vision", False, context),
            task_prompt=str(payload.get("task_prompt", defaults["task_prompt"])).strip()
            or defaults["task_prompt"],
        )


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Where the corpora live and where the run writes.

    Attributes:
        letters_manifest: JSONL of letter pages, one record per page.
        pixel_metadata: Persian Pixel metadata CSV. Never ``labels.csv``,
            which is header-only in this checkout.
        output_dir: Root for the manifest, checkpoints, adapter and metrics.
    """

    letters_manifest: Path = Path("data/dataset/paddle_ocr_vl.jsonl")
    pixel_metadata: Path = Path("data/persian_pixel/all_features_metadata.csv")
    output_dir: Path = Path("data/training/local_run")

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> PathsConfig:
        """Build the section from YAML, resolving against the repository root.

        Args:
            payload: The ``paths`` mapping.

        Returns:
            The validated section, with absolute paths.

        Raises:
            ConfigError: On an unknown or empty key.
        """
        context = "paths"
        defaults = _defaults(cls)
        _reject_unknown(payload, defaults, context)
        resolved: dict[str, Path] = {}
        for name, default in defaults.items():
            raw = payload.get(name, default)
            text = str(raw).strip()
            if not text:
                raise ConfigError(f"{context}.{name}: must not be empty")
            resolved[name] = resolve_repo_path(text)
        return cls(**resolved)

    @property
    def letters_root(self) -> Path:
        """Directory that ``image`` values in the letters JSONL are relative to."""
        return self.letters_manifest.parent

    @property
    def pixel_root(self) -> Path:
        """Directory that ``image_path`` values in the pixel CSV are relative to."""
        return self.pixel_metadata.parent

    @property
    def manifest_path(self) -> Path:
        """Where :mod:`~fa_ocr_lora.training.manifest` writes the merged corpus."""
        return self.output_dir / "unified_manifest.parquet"

    @property
    def split_summary_path(self) -> Path:
        """Where the split and mix counts are recorded for the run's README."""
        return self.output_dir / "split_summary.json"

    @property
    def skipped_log_path(self) -> Path:
        """Append-only log of every record the readers rejected, and why."""
        return self.output_dir / "skipped_records.log"

    @property
    def adapter_dir(self) -> Path:
        """Standalone LoRA adapter output."""
        return self.output_dir / "adapter_only"

    @property
    def merged_dir(self) -> Path:
        """Base-plus-adapter merged weights output."""
        return self.output_dir / "merged_model"


@dataclass(frozen=True, slots=True)
class DataConfig:
    """How many records to draw from each source, and how to shape them.

    Attributes:
        letters_pages: Full-page records drawn from the letters corpus.
        letters_crops: Region-crop records drawn from the same corpus.
        pixel_pages: Records drawn from the Persian Pixel corpus.
        validation_size: Records held out before any training.
        test_fraction: Share of the remainder kept as an untouched test split.
        crop_padding_px: Padding added around a region bbox before cropping.
        crop_min_side_px: Shortest crop edge worth keeping.
        crop_min_area_px: Smallest crop area worth keeping.
        crop_exclude_types: Region types never cropped.
        crop_type_weights: Relative share of the crop budget per region type.
            The letters corpus is wildly unbalanced - 94k ``list_item`` regions
            against 31k ``metadata_field`` - so a uniform draw would spend the
            crop budget on list items and boilerplate. Weights are relative, not
            fractions; they need not sum to anything.
        crop_default_weight: Weight for a region type absent from
            ``crop_type_weights``, so a regenerated corpus with a new type still
            contributes instead of vanishing silently.
        min_text_chars: Shortest usable target.
        max_text_chars: Longest target accepted.
        digit_oversample_multiplier: Repeat factor for the top digit-density
            quartile; ``1.0`` disables the reweighting.
    """

    letters_pages: int = 2500
    letters_crops: int = 2500
    pixel_pages: int = 5000
    validation_size: int = 800
    test_fraction: float = 0.0
    crop_padding_px: int = 8
    crop_min_side_px: int = 16
    crop_min_area_px: int = 900
    crop_exclude_types: tuple[str, ...] = (
        "header_bar",
        "footer_bar",
        "metadata_block",
        "attachments_block",
        "signature_block",
        "signature_image",
        "stamp",
    )
    crop_type_weights: tuple[tuple[str, float], ...] = (
        # The administrative metadata fields CLAUDE.md 5.2 names as the
        # highest-priority target: شماره ثبت / تاریخ / پیوست. Weighted far above
        # their natural share on purpose.
        ("metadata_field", 8.0),
        # Numeric and tabular content, where Persian OCR fails hardest.
        ("table_cell", 4.0),
        ("page_number", 2.0),
        ("footer_field", 2.0),
        # Ordinary prose, for coverage of general layout and long lines.
        ("paragraph", 3.0),
        ("list_item", 3.0),
        ("subject", 2.0),
        ("recipient", 2.0),
        ("sender", 1.5),
        ("signatory_title", 1.5),
        ("heading", 1.0),
        # Boilerplate repeated across thousands of letters: a handful teaches the
        # glyph shapes, a thousand teaches nothing new.
        ("header_company_name", 0.75),
        ("header_slogan", 0.5),
        ("religious_opener", 0.5),
        ("chart", 0.5),
    )
    crop_default_weight: float = 0.5
    min_text_chars: int = 2
    max_text_chars: int = 6000
    digit_oversample_multiplier: float = 2.0

    def __post_init__(self) -> None:
        """Reject a mix that leaves nothing to train on.

        Raises:
            ConfigError: If every source count is zero, or the holdout is large
                enough to consume the whole corpus.
        """
        if self.total_records == 0:
            raise ConfigError(
                "data: letters_pages, letters_crops and pixel_pages are all zero, "
                "so there is nothing to train on"
            )
        if self.validation_size >= self.total_records:
            raise ConfigError(
                f"data.validation_size: holdout of {self.validation_size} leaves no "
                f"training records out of {self.total_records}"
            )
        if self.max_text_chars <= self.min_text_chars:
            raise ConfigError(
                f"data: max_text_chars ({self.max_text_chars}) must exceed "
                f"min_text_chars ({self.min_text_chars})"
            )
        if self.crop_min_area_px < self.crop_min_side_px**2:
            raise ConfigError(
                f"data.crop_min_area_px: {self.crop_min_area_px} is below "
                f"crop_min_side_px squared ({self.crop_min_side_px ** 2}), so the "
                "area bound can never bind"
            )

    @property
    def total_records(self) -> int:
        """Records requested across every source."""
        return self.letters_pages + self.letters_crops + self.pixel_pages

    def crop_weight(self, region_type: str) -> float:
        """Relative crop-budget weight of a region type.

        Args:
            region_type: Type label as recorded in the corpus.

        Returns:
            The configured weight, or :attr:`crop_default_weight` for a type not
            named in the configuration.
        """
        for name, weight in self.crop_type_weights:
            if name == region_type:
                return weight
        return self.crop_default_weight

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> DataConfig:
        """Build the section from YAML.

        Args:
            payload: The ``data`` mapping.

        Returns:
            The validated section.

        Raises:
            ConfigError: On an unknown key or an inconsistent combination.
        """
        context = "data"
        defaults = _defaults(cls)
        _reject_unknown(payload, defaults, context)
        return cls(
            letters_pages=_as_int(
                payload, "letters_pages", defaults["letters_pages"], context, minimum=0
            ),
            letters_crops=_as_int(
                payload, "letters_crops", defaults["letters_crops"], context, minimum=0
            ),
            pixel_pages=_as_int(
                payload, "pixel_pages", defaults["pixel_pages"], context, minimum=0
            ),
            validation_size=_as_int(
                payload,
                "validation_size",
                defaults["validation_size"],
                context,
                minimum=1,
            ),
            test_fraction=_as_float(
                payload,
                "test_fraction",
                defaults["test_fraction"],
                context,
                minimum=0.0,
                maximum=0.5,
            ),
            crop_padding_px=_as_int(
                payload,
                "crop_padding_px",
                defaults["crop_padding_px"],
                context,
                minimum=0,
                maximum=64,
            ),
            crop_min_side_px=_as_int(
                payload,
                "crop_min_side_px",
                defaults["crop_min_side_px"],
                context,
                minimum=1,
            ),
            crop_min_area_px=_as_int(
                payload,
                "crop_min_area_px",
                defaults["crop_min_area_px"],
                context,
                minimum=1,
            ),
            crop_exclude_types=_as_str_tuple(
                payload, "crop_exclude_types", defaults["crop_exclude_types"], context
            ),
            crop_type_weights=_as_weight_pairs(
                payload, "crop_type_weights", defaults["crop_type_weights"], context
            ),
            crop_default_weight=_as_float(
                payload,
                "crop_default_weight",
                defaults["crop_default_weight"],
                context,
                minimum=0.0,
                maximum=100.0,
            ),
            min_text_chars=_as_int(
                payload, "min_text_chars", defaults["min_text_chars"], context, minimum=1
            ),
            max_text_chars=_as_int(
                payload, "max_text_chars", defaults["max_text_chars"], context, minimum=2
            ),
            digit_oversample_multiplier=_as_float(
                payload,
                "digit_oversample_multiplier",
                defaults["digit_oversample_multiplier"],
                context,
                minimum=1.0,
                maximum=10.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class ImageConfig:
    """Resolution and sequence budgets, the two knobs that decide VRAM.

    Attributes:
        max_pixels: Ceiling on pixels per image, a whole number of image tokens.
        min_pixels: Floor, matching the processor's own.
        autoprobe_vram: Whether to clamp ``max_pixels`` to measured free VRAM.
        reserved_vram_mib: Override for the projected peak VRAM requirement, in
            MiB. ``0`` uses the coefficients measured in
            :mod:`fa_ocr_lora.training.vram`, which describe this model on an
            RTX 3050 Ti; set a figure when running elsewhere.
        max_total_seq_len: Prompt plus answer, image placeholders included.
        loss_chunk_rows: Answer rows projected to vocabulary at once.
        on_oversize: Policy for a record that still does not fit.
    """

    max_pixels: int = 1_003_520
    min_pixels: int = 147_384
    autoprobe_vram: bool = True
    reserved_vram_mib: int = 0
    max_total_seq_len: int = 3072
    loss_chunk_rows: int = 128
    on_oversize: str = "shrink_image"

    def __post_init__(self) -> None:
        """Validate the pixel budget against the processor's own arithmetic.

        Raises:
            ConfigError: If the bounds are inverted, the budget is not a whole
                number of image tokens, or the image alone fills the sequence.
        """
        if self.max_pixels <= self.min_pixels:
            raise ConfigError(
                f"image: max_pixels ({self.max_pixels}) must exceed min_pixels "
                f"({self.min_pixels})"
            )
        if self.max_pixels % PIXELS_PER_TOKEN:
            raise ConfigError(
                f"image.max_pixels: {self.max_pixels} is not a multiple of "
                f"{PIXELS_PER_TOKEN} (one image token), so the processor would "
                f"round it down to "
                f"{self.max_pixels // PIXELS_PER_TOKEN * PIXELS_PER_TOKEN}"
            )
        if self.max_image_tokens >= self.max_total_seq_len:
            raise ConfigError(
                f"image: max_pixels is {self.max_image_tokens} image tokens, which "
                f"leaves no room for text inside max_total_seq_len "
                f"({self.max_total_seq_len})"
            )
        if 0 < self.reserved_vram_mib < 256:
            raise ConfigError(
                f"image.reserved_vram_mib: {self.reserved_vram_mib} MiB could not "
                f"hold the CUDA context alone, so it is far likelier to be a typo "
                f"than a budget; use 0 to reason from measured coefficients instead"
            )

    @property
    def max_image_tokens(self) -> int:
        """Image tokens the pixel ceiling corresponds to."""
        return self.max_pixels // PIXELS_PER_TOKEN

    @property
    def min_image_tokens(self) -> int:
        """Image tokens the pixel floor corresponds to."""
        return self.min_pixels // PIXELS_PER_TOKEN

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> ImageConfig:
        """Build the section from YAML.

        Args:
            payload: The ``image`` mapping.

        Returns:
            The validated section.

        Raises:
            ConfigError: On an unknown key or a budget that cannot hold text.
        """
        context = "image"
        defaults = _defaults(cls)
        _reject_unknown(payload, defaults, context)
        return cls(
            max_pixels=_as_int(
                payload,
                "max_pixels",
                defaults["max_pixels"],
                context,
                minimum=PIXELS_PER_TOKEN,
            ),
            min_pixels=_as_int(
                payload,
                "min_pixels",
                defaults["min_pixels"],
                context,
                minimum=PIXELS_PER_TOKEN,
            ),
            autoprobe_vram=_as_bool(payload, "autoprobe_vram", True, context),
            # 0 is the "use measured coefficients" sentinel; anything positive
            # is a real budget, and a figure under 256 MiB could not hold the
            # CUDA context alone, so it is far likelier to be a typo.
            reserved_vram_mib=_as_int(
                payload,
                "reserved_vram_mib",
                defaults["reserved_vram_mib"],
                context,
                minimum=0,
            ),
            max_total_seq_len=_as_int(
                payload,
                "max_total_seq_len",
                defaults["max_total_seq_len"],
                context,
                minimum=64,
            ),
            loss_chunk_rows=_as_int(
                payload,
                "loss_chunk_rows",
                defaults["loss_chunk_rows"],
                context,
                minimum=8,
                maximum=4096,
            ),
            on_oversize=_as_choice(
                payload,
                "on_oversize",
                defaults["on_oversize"],
                context,
                OVERSIZE_POLICIES,
            ),
        )


@dataclass(frozen=True, slots=True)
class LoraConfig:
    """Adapter shape and which modules it wraps.

    Attributes:
        r: LoRA rank.
        alpha: LoRA scaling numerator; the update is scaled by ``alpha / r``.
        dropout: Dropout on the adapter input.
        bias: Which biases to train, in peft's spelling.
        target_modules: Projection suffixes adapted inside the decoder.
        adapt_embeddings: Whether ``embed_tokens`` is adapted.
        adapt_lm_head: Whether ``lm_head`` is adapted.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    adapt_embeddings: bool = True
    adapt_lm_head: bool = True

    def __post_init__(self) -> None:
        """Reject an adapter with no target projections.

        Raises:
            ConfigError: If ``target_modules`` is empty.
        """
        if not self.target_modules:
            raise ConfigError("lora.target_modules: must name at least one projection")

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> LoraConfig:
        """Build the section from YAML.

        Args:
            payload: The ``lora`` mapping.

        Returns:
            The validated section.

        Raises:
            ConfigError: On an unknown key or an out-of-range value.
        """
        context = "lora"
        defaults = _defaults(cls)
        _reject_unknown(payload, defaults, context)
        return cls(
            r=_as_int(payload, "r", defaults["r"], context, minimum=1, maximum=512),
            alpha=_as_int(payload, "alpha", defaults["alpha"], context, minimum=1),
            dropout=_as_float(
                payload, "dropout", defaults["dropout"], context, minimum=0.0, maximum=0.9
            ),
            bias=_as_choice(payload, "bias", defaults["bias"], context, _LORA_BIAS),
            target_modules=_as_str_tuple(
                payload, "target_modules", defaults["target_modules"], context
            ),
            adapt_embeddings=_as_bool(payload, "adapt_embeddings", True, context),
            adapt_lm_head=_as_bool(payload, "adapt_lm_head", True, context),
        )


@dataclass(frozen=True, slots=True)
class OptimConfig:
    """Learning schedule and the batch shape it accumulates over.

    Attributes:
        learning_rate: LR for attention and MLP adapters.
        embed_head_learning_rate: LR for ``embed_tokens`` and ``lm_head``.
        weight_decay: AdamW decay.
        warmup_ratio: Share of total steps spent warming up.
        lr_scheduler_type: Decay shape after warmup.
        max_grad_norm: Gradient clipping threshold.
        num_epochs: Passes over the training split.
        per_device_batch_size: Samples per forward pass.
        gradient_accumulation_steps: Forward passes per optimiser step.
        optimizer: ``auto`` picks by bitsandbytes availability.
        gradient_checkpointing: Whether the decoder recomputes activations.
    """

    learning_rate: float = 2.0e-4
    embed_head_learning_rate: float = 7.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.08
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    num_epochs: float = 2.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    optimizer: str = "auto"
    gradient_checkpointing: bool = True

    @property
    def effective_batch_size(self) -> int:
        """Samples per optimiser step."""
        return self.per_device_batch_size * self.gradient_accumulation_steps

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> OptimConfig:
        """Build the section from YAML.

        Args:
            payload: The ``optim`` mapping.

        Returns:
            The validated section.

        Raises:
            ConfigError: On an unknown key or an out-of-range value.
        """
        context = "optim"
        defaults = _defaults(cls)
        _reject_unknown(payload, defaults, context)
        return cls(
            learning_rate=_as_float(
                payload,
                "learning_rate",
                defaults["learning_rate"],
                context,
                minimum=1e-8,
                maximum=1.0,
            ),
            embed_head_learning_rate=_as_float(
                payload,
                "embed_head_learning_rate",
                defaults["embed_head_learning_rate"],
                context,
                minimum=1e-8,
                maximum=1.0,
            ),
            weight_decay=_as_float(
                payload,
                "weight_decay",
                defaults["weight_decay"],
                context,
                minimum=0.0,
                maximum=1.0,
            ),
            warmup_ratio=_as_float(
                payload,
                "warmup_ratio",
                defaults["warmup_ratio"],
                context,
                minimum=0.0,
                maximum=0.5,
            ),
            lr_scheduler_type=_as_choice(
                payload,
                "lr_scheduler_type",
                defaults["lr_scheduler_type"],
                context,
                _SCHEDULERS,
            ),
            max_grad_norm=_as_float(
                payload, "max_grad_norm", defaults["max_grad_norm"], context, minimum=0.0
            ),
            num_epochs=_as_float(
                payload,
                "num_epochs",
                defaults["num_epochs"],
                context,
                minimum=0.01,
                maximum=100.0,
            ),
            per_device_batch_size=_as_int(
                payload,
                "per_device_batch_size",
                defaults["per_device_batch_size"],
                context,
                minimum=1,
                maximum=64,
            ),
            gradient_accumulation_steps=_as_int(
                payload,
                "gradient_accumulation_steps",
                defaults["gradient_accumulation_steps"],
                context,
                minimum=1,
                maximum=1024,
            ),
            optimizer=_as_choice(
                payload, "optimizer", defaults["optimizer"], context, _OPTIMIZERS
            ),
            gradient_checkpointing=_as_bool(
                payload, "gradient_checkpointing", True, context
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Loop mechanics: workers, logging cadence, checkpointing, shortcuts.

    Attributes:
        dataloader_workers: Worker processes decoding images.
        logging_steps: Optimiser steps between log lines.
        save_steps: Optimiser steps between checkpoints.
        save_total_limit: Checkpoints kept on disk.
        eval_steps: Optimiser steps between validation passes.
        resume_from_checkpoint: Whether to continue from the newest checkpoint.
        smoke_test: Restrict to a handful of samples and steps.
        smoke_samples: Samples used when ``smoke_test`` is set.
        smoke_steps: Optimiser steps taken when ``smoke_test`` is set.
        probe_steps: Steps to time and then stop; ``0`` trains normally.
    """

    dataloader_workers: int = 2
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    eval_steps: int = 200
    resume_from_checkpoint: bool = True
    smoke_test: bool = False
    smoke_samples: int = 16
    smoke_steps: int = 2
    probe_steps: int = 0

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> RuntimeConfig:
        """Build the section from YAML.

        Args:
            payload: The ``runtime`` mapping.

        Returns:
            The validated section.

        Raises:
            ConfigError: On an unknown key or an out-of-range value.
        """
        context = "runtime"
        defaults = _defaults(cls)
        _reject_unknown(payload, defaults, context)
        return cls(
            dataloader_workers=_as_int(
                payload,
                "dataloader_workers",
                defaults["dataloader_workers"],
                context,
                minimum=0,
                maximum=32,
            ),
            logging_steps=_as_int(
                payload, "logging_steps", defaults["logging_steps"], context, minimum=1
            ),
            save_steps=_as_int(
                payload, "save_steps", defaults["save_steps"], context, minimum=1
            ),
            save_total_limit=_as_int(
                payload,
                "save_total_limit",
                defaults["save_total_limit"],
                context,
                minimum=1,
            ),
            eval_steps=_as_int(
                payload, "eval_steps", defaults["eval_steps"], context, minimum=1
            ),
            resume_from_checkpoint=_as_bool(
                payload, "resume_from_checkpoint", True, context
            ),
            smoke_test=_as_bool(payload, "smoke_test", False, context),
            smoke_samples=_as_int(
                payload, "smoke_samples", defaults["smoke_samples"], context, minimum=2
            ),
            smoke_steps=_as_int(
                payload, "smoke_steps", defaults["smoke_steps"], context, minimum=1
            ),
            probe_steps=_as_int(
                payload, "probe_steps", defaults["probe_steps"], context, minimum=0
            ),
        )


@dataclass(frozen=True, slots=True)
class DriveConfig:
    """Mirroring checkpoints and the run log to Google Drive.

    Off by default, because a local run has nothing to gain from it. It exists
    for the rented-GPU case: an instance that is reclaimed takes its disk with
    it, so the only durable copy of a long run's checkpoints is a remote one.
    :mod:`fa_ocr_lora.training.checkpoint_sync` does the work, on a background
    thread, so enabling this does not slow a step down.

    The three credential paths are alternatives, tried in the order the module
    documents: service account, cached token, then the interactive flow. On a
    headless server only the first two can work, which is why
    ``interactive_auth`` defaults to ``False`` - a consent flow there would block
    on a browser that will never open, and a run that hangs at startup is worse
    than one that says why it cannot authenticate.

    Attributes:
        enabled: Whether to mirror at all.
        folder: Drive folder path holding every run, resolved by name.
        folder_id: An explicit Drive folder id, which wins over ``folder``.
            Required to write into a folder created outside this code, since the
            ``drive.file`` scope cannot see one by name and would silently make a
            second folder beside it.
        run_name: Remote subfolder for this run. Empty derives it from
            ``paths.output_dir``, so two runs writing to different output
            directories never mix their checkpoints.
        service_account_path: Service-account key JSON. ``null`` falls back to
            the module default, and then to the other two routes.
        token_path: Cached OAuth token JSON. ``null`` uses the module default.
        client_secret_path: OAuth client secret. ``null`` uses the default.
        interactive_auth: Permit opening a browser for consent.
        workers: Parallel file transfers within one checkpoint.
        upload_logs: Mirror the run log next to the checkpoints.
        upload_optimizer_state: Include ``optimizer.pt`` and friends. They are
            the bulk of a checkpoint; excluding them makes the mirror much
            faster but means a resumed run restarts its optimiser state.
        resume_from_drive: When resuming and no local checkpoint exists, pull the
            newest complete remote one.
        keep_remote: Remote checkpoints retained, oldest deleted first. ``0``
            keeps every one, which over a 17,000-step run at ``save_steps: 200``
            means 86 of them.
        drain_timeout_s: Seconds to wait for queued uploads once training ends.
    """

    enabled: bool = False
    folder: str = "fa-ocr-lora"
    folder_id: str = ""
    run_name: str = ""
    service_account_path: Path | None = None
    token_path: Path | None = None
    client_secret_path: Path | None = None
    interactive_auth: bool = False
    workers: int = 4
    upload_logs: bool = True
    upload_optimizer_state: bool = True
    resume_from_drive: bool = True
    keep_remote: int = 5
    drain_timeout_s: float = 900.0

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> DriveConfig:
        """Build the section from YAML.

        Args:
            payload: The ``drive`` mapping.

        Returns:
            The validated section.

        Raises:
            ConfigError: On an unknown key or an out-of-range value.
        """
        context = "drive"
        defaults = _defaults(cls)
        _reject_unknown(payload, defaults, context)

        def optional_path(key: str) -> Path | None:
            """Read a path that may be ``null``, meaning "use the default"."""
            raw = payload.get(key, defaults[key])
            if raw is None:
                return None
            text = str(raw).strip()
            return resolve_repo_path(text) if text else None

        folder = str(payload.get("folder", defaults["folder"])).strip()
        if not folder:
            raise ConfigError(f"{context}.folder: must not be empty")
        return cls(
            enabled=_as_bool(payload, "enabled", False, context),
            folder=folder,
            folder_id=str(payload.get("folder_id", defaults["folder_id"])).strip(),
            run_name=str(payload.get("run_name", defaults["run_name"])).strip(),
            service_account_path=optional_path("service_account_path"),
            token_path=optional_path("token_path"),
            client_secret_path=optional_path("client_secret_path"),
            interactive_auth=_as_bool(payload, "interactive_auth", False, context),
            workers=_as_int(
                payload, "workers", defaults["workers"], context, minimum=1, maximum=16
            ),
            upload_logs=_as_bool(payload, "upload_logs", True, context),
            upload_optimizer_state=_as_bool(
                payload, "upload_optimizer_state", True, context
            ),
            resume_from_drive=_as_bool(payload, "resume_from_drive", True, context),
            keep_remote=_as_int(
                payload, "keep_remote", defaults["keep_remote"], context, minimum=0
            ),
            drain_timeout_s=_as_float(
                payload,
                "drain_timeout_s",
                defaults["drain_timeout_s"],
                context,
                minimum=0.0,
            ),
        )

    def resolved_run_name(self, output_dir: Path) -> str:
        """Return the remote folder name for a run writing to ``output_dir``.

        Deriving it from the output directory rather than defaulting to a fixed
        string is what keeps a smoke test from overwriting an overnight run's
        checkpoints in Drive.
        """
        return self.run_name or output_dir.name


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Scoring settings for :mod:`~fa_ocr_lora.training.eval`.

    Attributes:
        max_samples: Validation records scored; ``None`` means all of them.
        max_new_tokens: Generation cap per record.
        num_beams: Beam count; ``1`` is greedy, which keeps CER seed-independent.
            Also the only setting under which per-field confidence is recorded, as
            it is the only one where the emitted token is the argmax - see
            :mod:`fa_ocr_lora.validation.recorder`.
        confusion_top_k: Substitution pairs printed in the confusion summary.
        field_audit: Whether to run the sensitive-field audit alongside CER/WER.
            On by default: the metadata block is CLAUDE.md 5.2's highest-priority
            field and aggregate error rate cannot see it. It runs on the decoded
            strings the metrics already need, so it costs no extra generation.
        field_config_path: Rules for that audit; ``None`` uses
            ``configs/field_validation.yaml`` or ``FA_OCR_FIELD_CONFIG``. Separate
            from this file because Phase 3 inference applies the same patterns with
            no training profile at all.
        confidence_floor: Weakest-token probability below which a predicted field
            is reported as uncertain.
        max_findings: Individual faults retained in the audit report. Those beyond
            it are counted, never silently dropped.
        batch_size: Prompts per ``generate`` call. ``1`` is the original
            one-record-at-a-time path; anything above it left-pads the prompts and
            decodes them together, which is what turns a 2% utilised 96 GB card
            into a busy one. A measured test pass went from 4.9 s/record at 1 to
            well under a second at 16-32. Raise it until VRAM or the input
            pipeline binds; the cost is bounded by ``batch_size`` KV caches of
            ``max_new_tokens``, so it scales with that product rather than with
            the split size.
        dataloader_workers: Worker processes encoding prompts ahead of the GPU.
            Distinct from ``runtime.dataloader_workers``, which sizes the training
            loader: evaluation runs alone, so it can have the whole host. ``0``
            encodes inline, which is the right setting only for a debugger.
    """

    max_samples: int | None = 300
    max_new_tokens: int = 1024
    num_beams: int = 1
    confusion_top_k: int = 20
    field_audit: bool = True
    field_config_path: str | None = None
    confidence_floor: float = 0.5
    max_findings: int = 200
    batch_size: int = 16
    dataloader_workers: int = 4

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> EvalConfig:
        """Build the section from YAML.

        Args:
            payload: The ``eval`` mapping.

        Returns:
            The validated section.

        Raises:
            ConfigError: On an unknown key or an out-of-range value.
        """
        context = "eval"
        defaults = _defaults(cls)
        _reject_unknown(payload, defaults, context)
        raw_max = payload.get("max_samples", defaults["max_samples"])
        max_samples = (
            None
            if raw_max is None
            else _as_int({"max_samples": raw_max}, "max_samples", 300, context, minimum=1)
        )
        raw_path = payload.get("field_config_path", defaults["field_config_path"])
        if raw_path is not None and not isinstance(raw_path, str):
            raise ConfigError(
                f"{context}.field_config_path: expected a path string or null, "
                f"got {raw_path!r}"
            )
        field_config_path = raw_path.strip() or None if raw_path else None
        return cls(
            max_samples=max_samples,
            max_new_tokens=_as_int(
                payload,
                "max_new_tokens",
                defaults["max_new_tokens"],
                context,
                minimum=8,
                maximum=8192,
            ),
            num_beams=_as_int(
                payload,
                "num_beams",
                defaults["num_beams"],
                context,
                minimum=1,
                maximum=16,
            ),
            confusion_top_k=_as_int(
                payload,
                "confusion_top_k",
                defaults["confusion_top_k"],
                context,
                minimum=1,
                maximum=200,
            ),
            field_audit=_as_bool(
                payload, "field_audit", defaults["field_audit"], context
            ),
            field_config_path=field_config_path,
            confidence_floor=_as_float(
                payload,
                "confidence_floor",
                defaults["confidence_floor"],
                context,
                minimum=0.0,
                maximum=1.0,
            ),
            max_findings=_as_int(
                payload,
                "max_findings",
                defaults["max_findings"],
                context,
                minimum=0,
                maximum=100_000,
            ),
            batch_size=_as_int(
                payload,
                "batch_size",
                defaults["batch_size"],
                context,
                minimum=1,
                maximum=512,
            ),
            dataloader_workers=_as_int(
                payload,
                "dataloader_workers",
                defaults["dataloader_workers"],
                context,
                minimum=0,
                maximum=64,
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """The whole training configuration, as one immutable value.

    Attributes:
        seed: Master seed; per-record seeds derive from it.
        model: Base model and precision.
        paths: Corpus and output locations.
        data: Source mix and record filters.
        image: Resolution and sequence budgets.
        lora: Adapter shape.
        optim: Learning schedule.
        runtime: Loop mechanics.
        drive: Google Drive checkpoint mirroring.
        eval: Scoring settings.
        source_path: File this was loaded from, for the run's README.
    """

    seed: int = DEFAULT_SEED
    model: ModelConfig = ModelConfig()
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    image: ImageConfig = ImageConfig()
    lora: LoraConfig = LoraConfig()
    optim: OptimConfig = OptimConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    drive: DriveConfig = DriveConfig()
    eval: EvalConfig = EvalConfig()
    source_path: Path | None = None

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, source_path: Path | None = None
    ) -> TrainingConfig:
        """Build the configuration from a decoded YAML mapping.

        Args:
            payload: Top-level mapping.
            source_path: File it came from, recorded for provenance.

        Returns:
            The validated configuration.

        Raises:
            ConfigError: If any section is unknown, malformed, or inconsistent.
        """
        known = {
            "seed",
            "model",
            "paths",
            "data",
            "image",
            "lora",
            "optim",
            "runtime",
            "drive",
            "eval",
        }
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ConfigError(
                f"unknown top-level section(s) {unknown}; expected one of "
                f"{sorted(known)}"
            )
        return cls(
            seed=_as_int(payload, "seed", DEFAULT_SEED, "config", minimum=0),
            model=ModelConfig.parse(_section(payload, "model")),
            paths=PathsConfig.parse(_section(payload, "paths")),
            data=DataConfig.parse(_section(payload, "data")),
            image=ImageConfig.parse(_section(payload, "image")),
            lora=LoraConfig.parse(_section(payload, "lora")),
            optim=OptimConfig.parse(_section(payload, "optim")),
            runtime=RuntimeConfig.parse(_section(payload, "runtime")),
            drive=DriveConfig.parse(_section(payload, "drive")),
            eval=EvalConfig.parse(_section(payload, "eval")),
            source_path=source_path,
        )

    def to_dict(self) -> dict[str, Any]:
        """Render the configuration as JSON-safe nested dicts.

        Used by the run's README and by the checkpoint metadata, so a finished
        run always carries the exact settings that produced it.

        Returns:
            Nested plain-Python mapping, with paths as strings.
        """

        def render(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return list(value)
            return value

        payload: dict[str, Any] = {"seed": self.seed}
        for name in (
            "model",
            "paths",
            "data",
            "image",
            "lora",
            "optim",
            "runtime",
            "drive",
            "eval",
        ):
            section = getattr(self, name)
            payload[name] = {
                f.name: render(getattr(section, f.name)) for f in fields(section)
            }
        payload["source_path"] = render(self.source_path)
        return payload


def _env_overrides(config: TrainingConfig) -> TrainingConfig:
    """Overlay ``FA_OCR_TRAIN_*`` variables onto a validated configuration.

    Applied after the YAML is parsed, so a bad value from the environment hits
    the same validators a bad value in the file would and the error names the
    same field. Only the settings that genuinely differ per machine are
    exposed - paths, the memory budget, the run-shortening switches, and where
    Drive mirroring writes.

    Args:
        config: Configuration as loaded from YAML.

    Returns:
        The configuration with overrides applied, or ``config`` if none are set.

    Raises:
        ConfigError: If a variable is present but unusable.
    """
    path_vars = {
        f"{ENV_PREFIX}LETTERS_MANIFEST": "letters_manifest",
        f"{ENV_PREFIX}PIXEL_METADATA": "pixel_metadata",
        f"{ENV_PREFIX}OUTPUT_DIR": "output_dir",
    }
    path_changes = {
        name: resolve_repo_path(os.environ[variable].strip())
        for variable, name in path_vars.items()
        if os.environ.get(variable, "").strip()
    }

    changes: dict[str, Any] = {}
    if path_changes:
        changes["paths"] = replace(config.paths, **path_changes)

    image_payload: dict[str, Any] = {}
    for variable, key in (
        (f"{ENV_PREFIX}MAX_PIXELS", "max_pixels"),
        (f"{ENV_PREFIX}RESERVED_VRAM_MIB", "reserved_vram_mib"),
    ):
        raw = os.environ.get(variable, "").strip()
        if not raw:
            continue
        try:
            image_payload[key] = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{variable}: expected an integer, got {raw!r}") from exc
    if image_payload:
        merged = {f.name: getattr(config.image, f.name) for f in fields(ImageConfig)}
        merged.update(image_payload)
        changes["image"] = ImageConfig.parse(merged)

    # Drive mirroring is per-machine by nature: the same config file is read on
    # the laptop, which has no reason to upload, and on the rented box, which
    # does. The credential *paths* are deliberately not here - those come from
    # GOOGLE_DRIVE_* (see fa_ocr_lora.drive), keeping secrets on one convention.
    drive_payload: dict[str, Any] = {}
    enabled = os.environ.get(f"{ENV_PREFIX}DRIVE_ENABLED", "").strip()
    if enabled:
        if enabled.lower() not in {"0", "1", "true", "false", "yes", "no"}:
            raise ConfigError(
                f"{ENV_PREFIX}DRIVE_ENABLED: expected a boolean, got {enabled!r}"
            )
        drive_payload["enabled"] = enabled.lower() in {"1", "true", "yes"}
    for variable, key in (
        (f"{ENV_PREFIX}DRIVE_FOLDER_ID", "folder_id"),
        (f"{ENV_PREFIX}DRIVE_RUN_NAME", "run_name"),
    ):
        raw = os.environ.get(variable, "").strip()
        if raw:
            drive_payload[key] = raw
    if drive_payload:
        merged_drive = {
            f.name: getattr(config.drive, f.name) for f in fields(DriveConfig)
        }
        merged_drive.update(drive_payload)
        changes["drive"] = DriveConfig.parse(merged_drive)

    quantization = os.environ.get(f"{ENV_PREFIX}QUANTIZATION", "").strip()
    if quantization:
        merged_model = {
            f.name: getattr(config.model, f.name) for f in fields(ModelConfig)
        }
        merged_model["quantization"] = quantization
        changes["model"] = ModelConfig.parse(merged_model)

    return replace(config, **changes) if changes else config


def load_training_config(
    path: str | Path | None = None, *, apply_env: bool = True
) -> TrainingConfig:
    """Load and validate the local training configuration.

    Precedence is ``CLI flag > environment > YAML > dataclass default``. This
    function owns the middle two: it reads the YAML, then overlays whatever the
    environment sets. A caller that has parsed CLI flags applies those last, by
    :func:`dataclasses.replace` on the returned value.

    Args:
        path: YAML file to read, absolute or relative to the repository root.
            Defaults to ``FA_OCR_TRAIN_CONFIG`` when set, and to
            :data:`DEFAULT_TRAINING_CONFIG_PATH` otherwise - so a CLI
            ``--config`` must pass ``None`` rather than its own default to let
            the environment through.
        apply_env: Whether to overlay environment overrides. ``False`` loads the
            file exactly as written, which is what a test comparing against the
            committed YAML wants.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is missing, empty, or fails validation.
    """
    if path is None:
        env_path = os.environ.get(f"{ENV_PREFIX}CONFIG", "").strip()
        path = env_path if (apply_env and env_path) else DEFAULT_TRAINING_CONFIG_PATH
    target = resolve_repo_path(path)
    if not target.is_file():
        raise ConfigError(f"training config not found: {target}")
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{target}: invalid YAML - {exc}") from exc
    if payload is None:
        raise ConfigError(f"{target}: file is empty")
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{target}: expected a YAML mapping at the top level")
    config = TrainingConfig.from_dict(payload, source_path=target)
    return _env_overrides(config) if apply_env else config
