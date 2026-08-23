"""Train the Persian LoRA adapter for PaddleOCR-VL on one consumer GPU.

Reads the manifest built by :mod:`fa_ocr_lora.training.prepare_manifest`, loads the
base model with the frozen vision tower, and trains the adapter. Nothing here
touches the corpora directly; if the manifest is missing this exits with an error
rather than building one implicitly, because the mix is a decision worth making
explicitly.

Three modes, in the order they are worth running:

``--smoke``
    A couple of steps over a handful of records. Proves the whole path - manifest,
    encoder, model load, backward, optimiser - in about a minute, and saves
    nothing.

``--probe N``
    N real steps over the full split, reporting measured seconds per step, peak
    VRAM, and a projected hours-per-epoch. This number, not an estimate from token
    counts, is what should decide the epoch count. Also saves nothing.

``no flag``
    The configured run, with checkpoints every ``runtime.save_steps`` and
    automatic resume from the newest one.

Examples:
    Prove the wiring::

        python -m fa_ocr_lora.training.train_lora --smoke

    Measure throughput before committing a night to it::

        python -m fa_ocr_lora.training.train_lora --probe 30

    Train, overriding the schedule without editing the YAML::

        python -m fa_ocr_lora.training.train_lora --epochs 3 --grad-accum 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from ..config.base import ConfigError
from . import vram
from .config import TrainingConfig, load_training_config
from .loop import TrainingOutcome, run_training

__all__ = ["main", "parse_args"]

LOGGER = logging.getLogger("fa_ocr_lora.train_lora")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the Persian LoRA adapter for PaddleOCR-VL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training YAML; defaults to configs/lora_config.yaml.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run runtime.smoke_steps steps over runtime.smoke_samples records.",
    )
    mode.add_argument(
        "--probe",
        type=int,
        default=None,
        metavar="N",
        help="Run N steps over the full split, report throughput, save nothing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap training records, taken in manifest order.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Forward passes per optimiser step. Raise to smooth gradients "
        "without spending VRAM; per-device batch size stays 1 on 4 GB.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Image budget per sample. Lower it to trade resolution for VRAM.",
    )
    parser.add_argument(
        "--quantize",
        choices=("nf4", "none"),
        default=None,
        help="Base-model precision. 'none' needs a GPU with the desktop session "
        "off it; see the quantization comment in the YAML.",
    )
    parser.add_argument(
        "--no-quantize-vision",
        action="store_true",
        help="Keep the frozen vision tower in bf16. Better features, 640 MiB "
        "more VRAM; inference must then match.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any checkpoint in the output directory.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log every configuration value actually in effect.",
    )
    return parser.parse_args(argv)


def _apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Apply CLI flags on top of a loaded configuration.

    Rebuilt through ``from_dict`` rather than mutated, so an override is validated
    by the same rules a YAML value is: ``--max-pixels 5`` fails here exactly as it
    would in the file.

    Args:
        config: Configuration as loaded from YAML plus environment.
        args: Parsed CLI arguments.

    Returns:
        A new configuration with the overrides applied.

    Raises:
        ConfigError: If an override is out of range or inconsistent.
    """
    payload = config.to_dict()
    payload.pop("source_path", None)
    mapping = {
        ("paths", "output_dir"): args.output_dir,
        ("optim", "num_epochs"): args.epochs,
        ("optim", "learning_rate"): args.learning_rate,
        ("optim", "gradient_accumulation_steps"): args.grad_accum,
        ("image", "max_pixels"): args.max_pixels,
        ("model", "quantization"): args.quantize,
    }
    for (section, key), value in mapping.items():
        if value is not None:
            payload[section][key] = str(value) if isinstance(value, Path) else value
    if args.no_quantize_vision:
        payload["model"]["quantize_vision"] = False
    if args.no_resume:
        payload["runtime"]["resume_from_checkpoint"] = False
    if args.smoke:
        payload["runtime"]["smoke_test"] = True
    if args.probe is not None:
        payload["runtime"]["probe_steps"] = args.probe
    if args.seed is not None:
        payload["seed"] = args.seed
    return TrainingConfig.from_dict(payload, source_path=config.source_path)


def _report_oom(config: TrainingConfig, exc: BaseException) -> None:
    """Turn an out-of-memory failure into an actionable message.

    A CUDA OOM traceback names a tensor allocation, which says nothing about
    which knob to turn. The banner does, and it orders the options by how much
    each was measured to buy, so the first suggestion is the one worth trying.

    Args:
        config: Configuration the failed run used.
        exc: The exception, quoted for the allocator's own numbers.
    """
    LOGGER.error("out of memory: %s", exc)
    report = vram.probe_vram(config.model, config.image)
    LOGGER.error("%s", vram.banner(report, config.model))
    LOGGER.error("levers, largest measured saving first:")
    if config.model.quantization != "nf4":
        LOGGER.error("  --quantize nf4                 base weights 1730 -> 1510 MiB")
    if not config.model.quantize_vision:
        LOGGER.error("  drop --no-quantize-vision      base weights 1510 ->  870 MiB")
    LOGGER.error(
        "  --max-pixels %-17d activations, ~0.16 MiB per token given up",
        max(config.image.min_pixels, config.image.max_pixels // 2),
    )
    LOGGER.error("  close whatever else holds the GPU (see 'holding' above)")


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        ``0`` on success, ``1`` on a configuration, corpus or memory error.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    try:
        config = _apply_overrides(load_training_config(args.config), args)
    except (ConfigError, FileNotFoundError) as exc:
        LOGGER.error("configuration error: %s", exc)
        return 1

    if not config.paths.manifest_path.is_file():
        LOGGER.error("no manifest at %s", config.paths.manifest_path)
        LOGGER.error("build it first: python -m fa_ocr_lora.training.prepare_manifest")
        return 1

    smoke = config.runtime.smoke_test
    probe_steps = config.runtime.probe_steps
    if smoke:
        mode = "smoke"
    elif probe_steps:
        mode = f"probe {probe_steps} steps"
    else:
        mode = "train"
    LOGGER.info("config      %s", config.source_path)
    LOGGER.info("manifest    %s", config.paths.manifest_path)
    LOGGER.info("output      %s", config.paths.output_dir)
    LOGGER.info(
        "mode        %s, %d epoch(s), effective batch %d",
        mode,
        config.optim.num_epochs,
        config.optim.effective_batch_size,
    )
    if args.verbose:
        LOGGER.debug("full configuration: %s", config.to_dict())

    try:
        outcome: TrainingOutcome = run_training(
            config, smoke=smoke, probe_steps=probe_steps, limit=args.limit
        )
    except torch.OutOfMemoryError as exc:
        _report_oom(config, exc)
        return 1
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("training failed: %s", exc)
        return 1

    if probe_steps:
        LOGGER.info(
            "at %.2fs/sample, %d epoch(s) over %d samples is %.1f h",
            outcome.seconds_per_sample,
            int(config.optim.num_epochs) or 1,
            outcome.epoch_samples,
            outcome.hours_per_epoch * max(1.0, config.optim.num_epochs),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
