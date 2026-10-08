"""Resolve CLI flags + YAML + checkpoint discovery into one AppConfig."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..dataset.common.io_utils import resolve_repo_path
from ..training.config import load_training_config
from .config import DEFAULT_PROMPT, AppConfig
from .discovery import default_checkpoint

__all__ = ["build_app_config"]

LOGGER = logging.getLogger("fa_ocr_lora.app")


def build_app_config(args: argparse.Namespace) -> AppConfig:
    """Fold CLI flags, YAML defaults, and checkpoint discovery into one config.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The resolved application configuration.

    Raises:
        FileNotFoundError: If the named adapter holds no adapter.
        ValueError: If the prompt resolves empty.
    """
    model_id = args.model_id
    revision = args.revision
    if (model_id is None or revision is None) and args.config is not None:
        yaml_config = load_training_config(args.config)
        model_id = model_id or yaml_config.model.model_id
        revision = revision or yaml_config.model.revision
    model_id = model_id or "PaddlePaddle/PaddleOCR-VL-1.6"
    revision = revision or "c5630abae1d940eafe0697512a0325494b02ab42"
    if args.no_adapter:
        adapter: Path | None = None
    elif args.adapter_dir is not None:
        adapter = Path(args.adapter_dir).expanduser()
        if not adapter.is_absolute():
            repo_try = resolve_repo_path(adapter)
            if (repo_try / "adapter_config.json").is_file():
                adapter = repo_try
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"{adapter} has no adapter_config.json; "
                "pass --no-adapter to run the base model alone"
            )
        adapter = adapter.resolve()
    else:
        discovered = default_checkpoint(args.checkpoints)
        adapter = discovered.resolve() if discovered is not None else None
        if adapter is None:
            LOGGER.warning("no adapter under %s; base model alone", args.checkpoints)
    if args.no_quant:
        quantization = "none"
    elif args.low_vram:
        quantization = "nf4"
    elif args.config is not None:
        try:
            quantization = load_training_config(args.config).model.quantization
        except Exception:  # noqa: BLE001 - safe default wins
            quantization = "nf4"
    else:
        quantization = "nf4"
    return AppConfig(
        model_id=model_id, revision=revision, adapter_dir=adapter,
        prompt=_resolve_prompt(args), max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels, min_pixels=args.min_pixels,
        quantization=quantization, dtype=args.dtype, workers=args.workers,
    )


def _resolve_prompt(args: argparse.Namespace) -> str:
    """Resolve the user prompt from --prompt, --prompt-file, or the default."""
    if args.prompt_file is not None:
        path = Path(args.prompt_file).expanduser()
        if not path.is_absolute():
            repo_try = resolve_repo_path(path)
            if repo_try.is_file():
                path = repo_try
        if not path.is_file():
            raise FileNotFoundError(f"no such prompt file: {args.prompt_file}")
        text = path.read_text(encoding="utf-8").strip()
    elif args.prompt is not None:
        text = args.prompt.strip()
    else:
        text = DEFAULT_PROMPT
    if not text:
        raise ValueError("the prompt is empty; pass --prompt or --prompt-file")
    return text
