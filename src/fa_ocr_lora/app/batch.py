"""Batch runner: prefetch on threads, generate serially, OOM backoff per image."""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import OOM_RETRIES, AppResult
from .loader import LoadedApp
from .runner import run_prompt

__all__ = ["prefetch_images", "run_batch"]


LOGGER = logging.getLogger("fa_ocr_lora.app")


def prefetch_images(images: list[Path], budget: int, workers: int) -> dict[Path, Any]:
    """Decode + resize every image on threads while the GPU is busy elsewhere.

    Args:
        images: Images to prefetch, in run order.
        budget: Starting pixel budget for this run.
        workers: Prefetch thread count.

    Returns:
        Mapping of image path to ready PIL image (or the exception raised).
    """
    from PIL import Image

    def _load_one(path: Path) -> tuple[Path, Any]:
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            scale = min(1.0, (budget / max(image.width * image.height, 1)) ** 0.5)
            if scale < 1.0:
                new_size = (
                    max(28, int(image.width * scale)),
                    max(28, int(image.height * scale)),
                )
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            return path, image
        except Exception as exc:  # noqa: BLE001 - recorded per image
            return path, exc

    ready: dict[Path, Any] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for path, result in pool.map(_load_one, images):
            ready[path] = result
    return ready


def run_batch(
    loader: LoadedApp, images: list[Path], *, show_progress: bool = True
) -> list[AppResult]:
    """Run every image through the model, sequentially on the GPU.

    Batch-1 generation is deliberate: a greedy decode at batch N holds N KV
    caches, and the 4 GB card this app targets has no headroom for that
    multiplier. Throughput comes from the prefetch pool keeping Pillow off
    the critical path, not from batching the GPU.

    Args:
        loader: Loaded model bundle.
        images: Image paths in run order.
        show_progress: Print a per-image progress line to stderr.

    Returns:
        One :class:`AppResult` per image, in run order.
    """
    import torch

    prompt = loader.config.prompt
    prefetched = prefetch_images(images, loader.config.max_pixels, loader.config.workers)
    results: list[AppResult] = []
    total = len(images)
    for index, path in enumerate(images, start=1):
        started = time.time()
        ready = prefetched.get(path)
        if isinstance(ready, Exception):
            results.append(AppResult(image=path, seconds=time.time() - started,
                                     error="DecodeError"))
            LOGGER.error("[%d/%d] %s: cannot decode (%s)", index, total, path.name, ready)
            continue
        budget = min(loader.config.max_pixels, loader.model_max_pixels)
        floor = max(loader.config.min_pixels, loader.model_min_pixels)
        text, error, used = "", None, budget
        for attempt in range(OOM_RETRIES + 1):
            try:
                if attempt > 0:
                    ready = _reload_at_budget(path, budget)
                text = run_prompt(loader, ready, prompt, budget)
                used, error = budget, None
                break
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                error = "OOM"
                halved = budget // 2
                if attempt < OOM_RETRIES and halved >= floor:
                    LOGGER.warning("[%d/%d] %s: OOM at %d px, retry at %d px",
                                   index, total, path.name, budget, halved)
                    budget = halved
                    continue
                LOGGER.error(
                    "[%d/%d] %s: out of memory, giving up", index, total, path.name
                )
                break
            except Exception as exc:  # noqa: BLE001 - one bad image ends no batch
                error = type(exc).__name__
                LOGGER.error("[%d/%d] %s: %s: %s", index, total, path.name, error, exc)
                break
        elapsed = time.time() - started
        results.append(AppResult(image=path, text=text, seconds=elapsed,
                                 pixels=used if error is None else 0, error=error))
        if show_progress:
            if error is None:
                status = f"{len(text)} chars, {elapsed:.1f}s"
            else:
                status = f"FAILED ({error})"
            print(f"[{index}/{total}] {path.name}: {status}", file=sys.stderr)
    return results


def _reload_at_budget(path: Path, budget: int) -> Any:
    """Re-decode one image sized for a retry budget after an OOM."""
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
    scale = (budget / max(image.width * image.height, 1)) ** 0.5
    if scale < 1.0:
        image = image.resize(
            (max(28, int(image.width * scale)), max(28, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image
