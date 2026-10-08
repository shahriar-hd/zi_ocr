"""Local VLM application: browse images, run PaddleOCR-VL-1.6 + LoRA, custom prompt.

Thin interactive layer over the native 1.6 stack the probe notebook verified:
``AutoProcessor`` + ``AutoModelForImageTextToText`` (transformers >= 5, no
remote code), optional PEFT adapter from ``checkpoints/``, image budget passed
per-call through ``images_kwargs`` so global processor state is never mutated.

Why a new package instead of extending ``inference/predict.py``: that path is a
fixed transcription endpoint (``SampleEncoder`` + ``"OCR:"`` task prompt +
field audit), and its correctness argument is byte-identical prompts to
training. This app takes an arbitrary user prompt per run, so sharing that
encoder would couple two prompts that must stay free to diverge. The model
*loading* (dtype probe, NF4 quantisation, adapter attach, ``use_cache`` fix)
mirrors ``training.modeling``, which is the part that must not drift.

Memory model (RTX 3050 Ti laptop, 4 GB):

* NF4 base weights (~870 MiB with vision quantised, per lora_config.yaml).
* Sequential generate (batch 1): one KV cache, no batch VRAM multiplier.
* Parallelism lives on the CPU side: a ThreadPoolExecutor resizes/prefetches
  images while the GPU generates, so the GPU never waits on Pillow.
* Per-image OOM is caught, the pixel budget is halved, and the image is
  retried before giving up.
"""

from __future__ import annotations

from .batch import run_batch
from .browse import browse_images
from .config import AppConfig, AppResult
from .discovery import collect_images, default_checkpoint, discover_checkpoints
from .loader import LoadedApp, load_app_model
from .runner import run_prompt

__all__ = [
    "AppConfig",
    "AppResult",
    "LoadedApp",
    "browse_images",
    "collect_images",
    "default_checkpoint",
    "discover_checkpoints",
    "load_app_model",
    "run_batch",
    "run_prompt",
]
