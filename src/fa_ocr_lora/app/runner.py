"""Run one user prompt over one image through the loaded 1.6 model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .loader import LoadedApp

__all__ = ["load_app_model", "run_prompt"]

LOGGER = logging.getLogger("fa_ocr_lora.app")

from .loader import load_app_model  # noqa: E402  (re-export for package root)


def run_prompt(
    loader: LoadedApp,
    image: Any,
    prompt: str,
    budget: int,
    *,
    min_budget: int | None = None,
) -> str:
    """Run one image + user prompt through the loaded model.

    Args:
        loader: Loaded model bundle.
        image: Ready PIL image (or a path, loaded on demand).
        prompt: User prompt text for the ``text`` turn after the image.
        budget: Pixel budget for this attempt, sent as ``longest_edge``.
        min_budget: Floor sent as ``shortest_edge``; defaults to the
            processor's own floor clamped below ``budget`` so the resize
            always has a valid ``[shortest, longest]`` window.

    Returns:
        The decoded generation, stripped.
    """
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            image = handle.convert("RGB")
    longest = max(int(budget), 28 * 28)
    floor = loader.model_min_pixels if min_budget is None else min_budget
    shortest = max(28 * 28, min(int(floor), longest))
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]
    inputs = loader.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": shortest, "longest_edge": longest}},
    )
    device = loader.device
    try:
        inputs = inputs.to(device)
    except AttributeError:
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]
    import torch  # noqa: PLC0415 - heavy dep, only needed for real inference

    with torch.inference_mode():
        generated = loader.model.generate(
            **inputs,
            max_new_tokens=loader.config.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    completion = generated[0][prompt_len:]
    return loader.processor.decode(completion, skip_special_tokens=True).strip()
