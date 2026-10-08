"""Load PaddleOCR-VL-1.6 natively and attach the LoRA adapter.

Mirrors the probe notebook's verified path: ``AutoProcessor`` then
``AutoModelForImageTextToText`` (transformers >= 5, ``trust_remote_code``
off), dtype probed from compute capability, optional NF4 via
``BitsAndBytesConfig``, ``device_map={"": 0}`` on CUDA, adapter attached with
PEFT in inference mode, and ``use_cache`` forced on (the checkpoint ships it
off, which costs a ~40x decode slowdown per the training docs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import AppConfig

__all__ = ["LoadedApp", "load_app_model"]

LOGGER = logging.getLogger("fa_ocr_lora.app")


@dataclass(slots=True)
class LoadedApp:
    """A loaded base model + processor, held together for one CLI run.

    Attributes:
        model: Base model with the adapter applied (or the bare base model).
        processor: Matching ``AutoProcessor`` (chat template + image limits).
        config: Resolved runtime configuration.
        model_min_pixels: Processor floor; the OOM backoff never goes below it.
        model_max_pixels: Processor ceiling; the budget is clamped to it.
    """

    model: Any
    processor: Any
    config: AppConfig
    model_min_pixels: int
    model_max_pixels: int
    _device: Any = field(default=None, repr=False)

    @property
    def device(self) -> Any:
        """Device the model lives on (resolved lazily, cached)."""
        if self._device is None:
            import torch

            try:
                self._device = next(self.model.parameters()).device
            except StopIteration:
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        return self._device


def load_app_model(config: AppConfig) -> LoadedApp:
    """Load the base model and attach the LoRA adapter if given.

    Args:
        config: Resolved runtime configuration.

    Returns:
        The loaded model bundle.

    Raises:
        FileNotFoundError: If the adapter directory holds no adapter.
        RuntimeError: If transformers < 5 is installed.
    """
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if int(transformers.__version__.split(".")[0]) < 5:
        raise RuntimeError(
            f"transformers {transformers.__version__} is < 5; "
            "PaddleOCR-VL-1.6 needs transformers>=5.0.0"
        )
    if config.adapter_dir is not None and not (
        config.adapter_dir / "adapter_config.json"
    ).is_file():
        raise FileNotFoundError(
            f"{config.adapter_dir} has no adapter_config.json; "
            "pass --no-adapter to run the base model alone"
        )
    dtype = _resolve_dtype(config)
    quant = _resolve_quant(config, dtype)
    LOGGER.info("loading %s as %s", config.model_id, str(dtype).replace("torch.", ""))
    processor = AutoProcessor.from_pretrained(config.model_id, revision=config.revision)
    load_kwargs: dict[str, Any] = {
        "revision": config.revision,
        "dtype": dtype,
        "trust_remote_code": False,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = {"": 0}
    if quant is not None:
        load_kwargs["quantization_config"] = quant
    try:
        load_kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForImageTextToText.from_pretrained(
            config.model_id, **load_kwargs
        )
    except ValueError:
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForImageTextToText.from_pretrained(
            config.model_id, **load_kwargs
        )
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    if config.adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(config.adapter_dir), is_trainable=False
        )
        LOGGER.info("applied adapter from %s", config.adapter_dir)
    model.eval()
    image_processor = processor.image_processor
    model_min = int(getattr(image_processor, "min_pixels", config.min_pixels))
    fallback = max(config.max_pixels, config.min_pixels + 1)
    model_max = int(getattr(image_processor, "max_pixels", fallback))
    if torch.cuda.is_available():
        weight_mib = torch.cuda.memory_allocated() / 2**20
    else:
        weight_mib = 0.0
    LOGGER.info(
        "ready: %s, weights %.0f MiB, pixel budget %d-%d",
        "nf4" if quant is not None else str(dtype).replace("torch.", ""),
        weight_mib,
        model_min,
        model_max,
    )
    return LoadedApp(
        model=model, processor=processor, config=config,
        model_min_pixels=model_min, model_max_pixels=model_max,
    )


def _resolve_dtype(config: AppConfig) -> Any:
    """Probe the compute dtype: bf16 on capability >= 8, fp16 below, fp32 CPU."""
    import torch

    if config.dtype != "auto":
        return getattr(torch, config.dtype)
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def _resolve_quant(config: AppConfig, dtype: Any) -> Any | None:
    """Build the NF4 quantisation config, or ``None`` for full precision."""
    if config.quantization == "none":
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        LOGGER.warning("bitsandbytes not installed; loading in %s instead", dtype)
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["lm_head"],
    )
