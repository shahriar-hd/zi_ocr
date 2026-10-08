"""Resolved runtime configuration and per-image result for the local VLM app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "APP_SCHEMA_VERSION",
    "DEFAULT_PROMPT",
    "IMAGE_SUFFIXES",
    "OOM_RETRIES",
    "PREFETCH_DEPTH",
    "AppConfig",
    "AppResult",
]

#: Version of the JSON bundle the CLI emits.
APP_SCHEMA_VERSION: Final[str] = "1.0"

#: Image suffixes accepted by --image-dir / --browse.
IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
)

#: Default user prompt when none is given: plain transcription.
DEFAULT_PROMPT: Final[str] = "OCR:"

#: Images prefetched ahead of the GPU. Threads, not processes: Pillow decode
#: and resize release the GIL in C, and threads share the PIL image without
#: pickling it back.
PREFETCH_DEPTH: Final[int] = 2

#: OOM retries per image, each halving the pixel budget.
OOM_RETRIES: Final[int] = 2


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Resolved runtime configuration for one app invocation.

    Attributes:
        model_id: Base model repo, from the training YAML unless overridden.
        revision: Pinned Hub revision, from the same YAML unless overridden.
        adapter_dir: LoRA adapter directory, or ``None`` for base model only.
        prompt: User prompt text sent with every image.
        max_new_tokens: Generation cap per image.
        max_pixels: Starting image pixel budget; halved on OOM retries.
        min_pixels: Floor the budget never drops below.
        quantization: ``nf4`` or ``none``; overrides the YAML on low VRAM.
        dtype: ``auto`` probes compute capability (bf16 on an 8.6 card).
        device: ``cuda`` when available, else ``cpu``.
        workers: Prefetch threads decoding images ahead of the GPU.
    """

    model_id: str = "PaddlePaddle/PaddleOCR-VL-1.6"
    revision: str = "c5630abae1d940eafe0697512a0325494b02ab42"
    adapter_dir: Path | None = None
    prompt: str = DEFAULT_PROMPT
    max_new_tokens: int = 512
    max_pixels: int = 1003520
    min_pixels: int = 147384
    quantization: str = "nf4"
    dtype: str = "auto"
    device: str = "cuda"
    workers: int = 4


@dataclass(slots=True)
class AppResult:
    """One image's outcome: text on success, error classification on failure.

    Attributes:
        image: Source image path (absolute).
        text: Generated output, ``""`` when the image failed.
        seconds: Wall time for this image, including retries.
        pixels: Pixel budget the successful attempt used (0 when failed).
        error: ``None`` on success, else ``OOM`` / ``DecodeError`` / class name.
    """

    image: Path
    text: str = ""
    seconds: float = 0.0
    pixels: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable record for this image."""
        return {
            "image": str(self.image),
            "text": self.text,
            "seconds": round(self.seconds, 2),
            "pixels": self.pixels,
            "error": self.error,
        }
