r"""Score a split with CER/WER, digit-isolated metrics, and character confusions.

Runs greedy generation over the validation (or test) split and reports what the
model actually reads. Two things about the design matter more than the metric
arithmetic, which lives in :mod:`fa_ocr_lora.training.metrics`:

**Run the baseline first.** ``--no-adapter`` scores the untrained base model
through the identical pipeline - same encoder, same resolution, same quantisation,
same prompt. A trained CER means nothing on its own; it means something against
that number. The pair is what tells you whether 5 hours of training bought
anything.

**Score records, not batches.** Generation is one record at a time. Batched greedy
decode would need left padding and a per-row length bookkeeping that is easy to get
subtly wrong, and the 4 GB budget cannot hold several 1,024-token KV caches anyway.
At roughly 3-10 s per page this makes a 300-record pass a coffee break, which is why
``eval.max_samples`` defaults well below the full split.

Examples:
    Baseline, before training::

        python -m fa_ocr_lora.training.eval --no-adapter --max-samples 50

    Score the trained adapter::

        python -m fa_ocr_lora.training.eval

    Score a specific checkpoint on the held-out test split::

        python -m fa_ocr_lora.training.eval \\
            --adapter-dir data/training/local_run/checkpoints/checkpoint-800 \\
            --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

from ..config.base import ConfigError
from ..dataset.common.io_utils import resolve_repo_path, write_json
from . import vram
from .config import TrainingConfig, load_training_config
from .encoding import IGNORE_INDEX, SampleEncoder, SampleTooLong
from .manifest import load_manifest, parse_crop_box
from .metrics import EvalMetrics, MetricAccumulator
from .modeling import EvalModel, load_eval_model

__all__ = ["main", "parse_args", "score_split"]

LOGGER = logging.getLogger("fa_ocr_lora.eval")

#: Predictions written alongside the metrics, so a bad CER can be read rather
#: than guessed at. One JSON object per line.
PREDICTIONS_NAME = "eval_predictions.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Score a split with CER/WER and digit-isolated metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training YAML; defaults to configs/lora_config.yaml.",
    )
    adapter = parser.add_mutually_exclusive_group()
    adapter.add_argument(
        "--no-adapter",
        action="store_true",
        help="Score the untrained base model, for a baseline to compare against.",
    )
    adapter.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Adapter to score; defaults to <output_dir>/adapter_only.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test", "train"),
        default="validation",
        help="Manifest split to score.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Records to score; overrides eval.max_samples. 0 means all of them.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Metrics JSON destination; defaults to <output_dir>/eval_<split>.json.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log every prediction as it is scored.",
    )
    return parser.parse_args(argv)


def _prompt_length(labels: torch.Tensor) -> int:
    """Number of leading positions that carry no supervision.

    The encoder masks exactly the prompt, so the first supervised position is the
    boundary. Deriving the generation prompt this way rather than re-rendering the
    template guarantees the model is asked at eval time precisely what it was
    trained on - one template, one image resolution, one code path.

    Args:
        labels: Label tensor from an encoded sample, shape ``(L,)``.

    Returns:
        The prompt length in tokens.

    Raises:
        ValueError: If the sample has no supervised position, which would mean an
            empty target reached the encoder.
    """
    supervised = (labels != IGNORE_INDEX).nonzero()
    if supervised.numel() == 0:
        raise ValueError("encoded sample has no supervised position")
    return int(supervised[0].item())


def score_split(
    config: TrainingConfig,
    loaded: EvalModel,
    *,
    split: str = "validation",
    max_samples: int | None = None,
    predictions_path: Path | None = None,
    verbose: bool = False,
) -> EvalMetrics:
    """Generate over a split and accumulate metrics.

    Failed records are counted and logged, never silently skipped, and they are not
    scored as empty predictions either - a file that will not open says nothing
    about how well the model reads, and folding it into the CER would understate
    the model in a way no later reader could untangle.

    Args:
        config: Loaded training configuration.
        loaded: Model and processor from
            :func:`~fa_ocr_lora.training.modeling.load_eval_model`.
        split: Manifest split to score.
        max_samples: Records to score; ``None`` uses ``eval.max_samples``, and a
            ``None`` there means the whole split.
        predictions_path: Where to write per-record predictions, or ``None`` to
            write none.
        verbose: Log every prediction as it is produced.

    Returns:
        The accumulated metrics.

    Raises:
        FileNotFoundError: If the manifest has not been built.
        ValueError: If the split is empty.
    """
    frame = load_manifest(config.paths.manifest_path)
    frame = frame[frame["split"] == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            f"split {split!r} of {config.paths.manifest_path} is empty; rebuild "
            "the manifest, or score a split the mix actually produced"
        )
    limit = config.eval.max_samples if max_samples is None else max_samples
    if limit:
        frame = frame.iloc[:limit]

    encoder = SampleEncoder(loaded.processor, model=config.model, image=config.image)
    tokenizer = loaded.processor.tokenizer
    device = next(loaded.model.parameters()).device
    accumulator = MetricAccumulator(confusion_top_k=config.eval.confusion_top_k)
    failures: dict[str, int] = {}
    handle = predictions_path.open("w", encoding="utf-8") if predictions_path else None
    started = time.perf_counter()

    try:
        for position in range(len(frame)):
            row = frame.iloc[position]
            try:
                sample = encoder.encode(
                    image_path=resolve_repo_path(row["image_path"]),
                    crop_box=parse_crop_box(row["crop_box"]),
                    text=row["text"],
                    record_id=row["record_id"],
                )
            except (SampleTooLong, OSError, ValueError) as exc:
                reason = type(exc).__name__
                failures[reason] = failures.get(reason, 0) + 1
                LOGGER.warning("skipping %s: %s", row["record_id"], exc)
                continue

            cut = _prompt_length(sample.labels)
            with torch.inference_mode():
                generated = loaded.model.generate(
                    input_ids=sample.input_ids[:cut].unsqueeze(0).to(device),
                    attention_mask=torch.ones((1, cut), dtype=torch.long, device=device),
                    pixel_values=sample.pixel_values.to(device, loaded.dtype),
                    image_grid_thw=sample.image_grid_thw.to(device),
                    max_new_tokens=config.eval.max_new_tokens,
                    num_beams=config.eval.num_beams,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prediction = tokenizer.decode(generated[0][cut:], skip_special_tokens=True)
            accumulator.update(row["text"], prediction)

            if verbose:
                LOGGER.debug("%s -> %r", row["record_id"], prediction[:120])
            if handle is not None:
                handle.write(
                    json.dumps(
                        {
                            "record_id": row["record_id"],
                            "source": row["source"],
                            "reference": row["text"],
                            "prediction": prediction,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if (position + 1) % 25 == 0:
                elapsed = time.perf_counter() - started
                LOGGER.info(
                    "  %d/%d scored (%.1fs/record)",
                    position + 1,
                    len(frame),
                    elapsed / (position + 1),
                )
    finally:
        if handle is not None:
            handle.close()

    if failures:
        total = sum(failures.values())
        detail = ", ".join(f"{name}={count}" for name, count in sorted(failures.items()))
        LOGGER.warning(
            "%d of %d record(s) could not be encoded and are excluded from the "
            "metrics: %s",
            total,
            len(frame),
            detail,
        )
    LOGGER.info(
        "scored %d record(s) in %.0fs",
        accumulator.result().samples,
        time.perf_counter() - started,
    )
    return accumulator.result()


def _resolve_adapter(config: TrainingConfig, args: argparse.Namespace) -> Path | None:
    """Decide which adapter, if any, to score.

    Args:
        config: Loaded training configuration.
        args: Parsed CLI arguments.

    Returns:
        The adapter directory, or ``None`` for the base-model baseline.
    """
    if args.no_adapter:
        return None
    if args.adapter_dir is not None:
        return resolve_repo_path(args.adapter_dir)
    return config.paths.adapter_dir


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        ``0`` on success, ``1`` on a configuration, manifest or memory error.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    try:
        config = load_training_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        LOGGER.error("configuration error: %s", exc)
        return 1
    if args.max_new_tokens is not None:
        payload = config.to_dict()
        payload.pop("source_path", None)
        payload["eval"]["max_new_tokens"] = args.max_new_tokens
        config = TrainingConfig.from_dict(payload, source_path=config.source_path)

    if not config.paths.manifest_path.is_file():
        LOGGER.error("no manifest at %s", config.paths.manifest_path)
        LOGGER.error("build it first: python -m fa_ocr_lora.training.prepare_manifest")
        return 1

    adapter_dir = _resolve_adapter(config, args)
    report = vram.probe_vram(config.model, config.image)
    LOGGER.info("%s", vram.banner(report, config.model))

    try:
        loaded = load_eval_model(config, adapter_dir=adapter_dir)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("%s", loaded.describe())

    destination = (
        resolve_repo_path(args.output)
        if args.output is not None
        else config.paths.output_dir / f"eval_{args.split}.json"
    )
    predictions_path = destination.parent / PREDICTIONS_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        metrics = score_split(
            config,
            loaded,
            split=args.split,
            max_samples=args.max_samples,
            predictions_path=predictions_path,
            verbose=args.verbose,
        )
    except torch.OutOfMemoryError as exc:
        LOGGER.error("out of memory during generation: %s", exc)
        LOGGER.error(
            "lower --max-new-tokens (currently %d) or image.max_pixels; a long "
            "greedy decode holds a KV cache proportional to both",
            config.eval.max_new_tokens,
        )
        return 1
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("evaluation failed: %s", exc)
        return 1

    LOGGER.info("%s", metrics.describe())
    write_json(
        destination,
        {
            "split": args.split,
            "adapter_dir": str(adapter_dir) if adapter_dir else None,
            "quantization": config.model.quantization,
            "quantize_vision": config.model.quantize_vision,
            "max_pixels": config.image.max_pixels,
            "max_new_tokens": config.eval.max_new_tokens,
            **metrics.to_dict(),
        },
    )
    LOGGER.info("metrics     %s", destination)
    LOGGER.info("predictions %s", predictions_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
