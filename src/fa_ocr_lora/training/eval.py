r"""Score a split with CER/WER, digit-isolated metrics, and character confusions.

Runs greedy generation over the validation (or test) split and reports what the
model actually reads. Two things about the design matter more than the metric
arithmetic, which lives in :mod:`fa_ocr_lora.training.metrics`:

**Run the baseline first.** ``--no-adapter`` scores the untrained base model
through the identical pipeline - same encoder, same resolution, same quantisation,
same prompt. A trained CER means nothing on its own; it means something against
that number. The pair is what tells you whether 5 hours of training bought
anything.

**Records are scored in batches.** :func:`score_split` keeps the one-record-at-a-time
loop, because it is the simplest thing that can be checked by eye and the 4 GB
development card cannot hold several 1,024-token KV caches. It is no longer the
default: ``eval.batch_size`` above 1 routes :func:`main` through
:func:`fa_ocr_lora.training.batch_eval.score_split_batched`, which left-pads the
prompts and decodes them together. On a 96 GB card that is the difference between
4.9 s per record and a fraction of a second, and the arithmetic is identical -
only the scheduling changed. ``--batch-size 1`` returns to this module's loop,
which is how a disagreement between the two gets bisected.

Examples:
    Baseline, before training::

        python -m fa_ocr_lora.training.eval --no-adapter --max-samples 50

    Score the trained adapter::

        python -m fa_ocr_lora.training.eval

    Score a specific checkpoint on the held-out test split, 32 rows at a time::

        python -m fa_ocr_lora.training.eval \\
            --adapter-dir data/training/local_run/checkpoints/checkpoint-800 \\
            --split test --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..config.base import ConfigError
from ..dataset.common.io_utils import resolve_repo_path, write_json
from ..validation import (
    FieldAudit,
    FieldAuditor,
    LazySpanConfidence,
    extract_fields,
    load_field_validation_config,
    summarize_fields,
)
from ..validation.recorder import processor_list, recorder_for
from . import vram
from .config import TrainingConfig, load_training_config
from .encoding import IGNORE_INDEX, SampleEncoder, SampleTooLong
from .manifest import load_manifest, parse_crop_box
from .metrics import EvalMetrics, MetricAccumulator
from .modeling import EvalModel, load_eval_model

__all__ = ["EvalOutcome", "main", "parse_args", "run_score_split", "score_split"]

LOGGER = logging.getLogger("fa_ocr_lora.eval")

#: Predictions written alongside the metrics, so a bad CER can be read rather
#: than guessed at. One JSON object per line.
PREDICTIONS_NAME = "eval_predictions.jsonl"


@dataclass(frozen=True, slots=True)
class EvalOutcome:
    """What one scoring pass produced.

    Two results rather than one because they answer different questions and must
    not be averaged into each other: CER and WER say how much of the page was
    read, the audit says whether the شماره ثبت and the تاریخ were read. A page is
    ~700 characters of prose against ~20 of metadata, so the first number cannot
    express the second.

    Attributes:
        metrics: Text, digit and confusion metrics over the whole transcription.
        audit: Sensitive-field audit, or ``None`` when ``eval.field_audit`` is off.
    """

    metrics: EvalMetrics
    audit: FieldAudit | None = None


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
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Prompts per generate() call; overrides eval.batch_size. 1 uses the "
            "unbatched loop in this module."
        ),
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=None,
        help=(
            "Worker processes encoding prompts ahead of the GPU; overrides "
            "eval.dataloader_workers. Ignored when --batch-size is 1."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Metrics JSON destination; defaults to <output_dir>/eval_<split>.json.",
    )
    parser.add_argument(
        "--field-config",
        type=Path,
        default=None,
        help=(
            "Sensitive-field rules; overrides eval.field_config_path. Defaults to "
            "configs/field_validation.yaml."
        ),
    )
    parser.add_argument(
        "--no-field-audit",
        action="store_true",
        help="Skip the sensitive-field audit and report CER/WER alone.",
    )
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=None,
        help=(
            "Weakest-token probability below which a predicted field is flagged as "
            "uncertain; overrides eval.confidence_floor."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log every prediction as it is scored, and every finding.",
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
) -> EvalOutcome:
    """Generate over a split, accumulating metrics and the sensitive-field audit.

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
        The metrics and, unless ``eval.field_audit`` is off, the field audit.

    Raises:
        FileNotFoundError: If the manifest has not been built.
        ValueError: If the split is empty.
        ConfigError: If ``eval.field_config_path`` names unloadable rules. Raised
            rather than warned past: an audit that silently stops checking a field
            reports a perfect score for it.
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
    auditor: FieldAuditor | None = None
    if config.eval.field_audit:
        rules = load_field_validation_config(config.eval.field_config_path)
        auditor = FieldAuditor(
            config=rules,
            confidence_floor=config.eval.confidence_floor,
            max_findings=config.eval.max_findings,
        )
        LOGGER.info(
            "field audit on: %s (%d field(s))", rules.source_path, len(rules.specs)
        )
    # Uncertainty is only recorded under greedy decoding, where the emitted token
    # is the argmax the recorder sees. Beam search would attribute a probability
    # to a token that was never produced, so the audit reports it as unavailable.
    recorder = recorder_for(config.eval.num_beams) if auditor else None
    processors = processor_list(recorder)
    if auditor and recorder is None:
        LOGGER.info(
            "field confidence off: num_beams=%d is not greedy", config.eval.num_beams
        )
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
            if recorder is not None:
                recorder.reset()
            extra: dict[str, torch.Tensor] = {}
            if sample.mm_token_type_ids is not None:
                # transformers 5's native 1.6 needs per-token modality ids for
                # M-RoPE and raises without them; the older checkpoint reads
                # position from the grid alone and never looks at this key.
                extra["mm_token_type_ids"] = (
                    sample.mm_token_type_ids[:cut].unsqueeze(0).to(device)
                )
            with torch.inference_mode():
                generated = loaded.model.generate(
                    input_ids=sample.input_ids[:cut].unsqueeze(0).to(device),
                    attention_mask=torch.ones((1, cut), dtype=torch.long, device=device),
                    pixel_values=sample.pixel_values.to(device, loaded.dtype),
                    image_grid_thw=sample.image_grid_thw.to(device),
                    **extra,
                    max_new_tokens=config.eval.max_new_tokens,
                    num_beams=config.eval.num_beams,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    logits_processor=processors,
                )
            new_tokens = generated[0][cut:]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
            accumulator.update(row["text"], prediction)

            audited: tuple[dict[str, object], ...] = ()
            findings: tuple[object, ...] = ()
            if auditor is not None:
                # The lookup decodes prefixes only if a field is actually found, so
                # an empty or garbage prediction costs nothing to audit.
                lookup = (
                    LazySpanConfidence(
                        tokenizer=tokenizer,
                        token_ids=new_tokens.tolist(),
                        steps=tuple(recorder.steps),
                    )
                    if recorder is not None
                    else None
                )
                findings = auditor.update(
                    row["record_id"], row["text"], prediction, confidence=lookup
                )
                audited = summarize_fields(extract_fields(prediction, auditor.config))
                if verbose and findings:
                    for finding in findings:
                        LOGGER.debug("  %s", finding.describe())

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
                            "fields": list(audited),
                            "findings": [item.to_dict() for item in findings],
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
    return EvalOutcome(
        metrics=accumulator.result(),
        audit=auditor.result() if auditor is not None else None,
    )


def run_score_split(
    config: TrainingConfig,
    loaded: EvalModel,
    *,
    split: str = "validation",
    max_samples: int | None = None,
    predictions_path: Path | None = None,
    verbose: bool = False,
) -> EvalOutcome:
    """Score a split, batching when ``eval.batch_size`` allows it.

    The single entry point every caller should use - :mod:`.eval`,
    :mod:`.compare_eval` and the notebooks all go through here, so the batched
    and unbatched paths can never diverge in which one a given config selects.

    Imported lazily because :mod:`.batch_eval` pulls in ``torch.utils.data``,
    which a caller that only wants :class:`EvalOutcome` has no reason to pay for.

    Args:
        config: Loaded training configuration.
        loaded: Model and processor from
            :func:`~fa_ocr_lora.training.modeling.load_eval_model`.
        split: Manifest split to score.
        max_samples: Records to score; ``None`` uses ``eval.max_samples``.
        predictions_path: Where to write per-record predictions, or ``None``.
        verbose: Log every prediction as it is produced.

    Returns:
        The metrics and, unless ``eval.field_audit`` is off, the field audit.
    """
    if config.eval.batch_size <= 1:
        LOGGER.info("unbatched scoring: eval.batch_size is 1")
        return score_split(
            config,
            loaded,
            split=split,
            max_samples=max_samples,
            predictions_path=predictions_path,
            verbose=verbose,
        )
    from .batch_eval import score_split_batched

    return score_split_batched(
        config,
        loaded,
        split=split,
        max_samples=max_samples,
        predictions_path=predictions_path,
        verbose=verbose,
    )


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
    overrides: dict[str, object] = {}
    if args.max_new_tokens is not None:
        overrides["max_new_tokens"] = args.max_new_tokens
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.eval_workers is not None:
        overrides["dataloader_workers"] = args.eval_workers
    if args.no_field_audit:
        overrides["field_audit"] = False
    if args.field_config is not None:
        overrides["field_config_path"] = str(args.field_config)
    if args.confidence_floor is not None:
        overrides["confidence_floor"] = args.confidence_floor
    if overrides:
        # Round-tripped through the loader rather than mutated in place, so a bad
        # flag is rejected by the same validation the YAML goes through.
        payload = config.to_dict()
        payload.pop("source_path", None)
        payload["eval"].update(overrides)
        try:
            config = TrainingConfig.from_dict(payload, source_path=config.source_path)
        except ConfigError as exc:
            LOGGER.error("invalid override: %s", exc)
            return 1

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
        outcome = run_score_split(
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
            "lower eval.batch_size (currently %d), --max-new-tokens (currently "
            "%d) or image.max_pixels; a batched greedy decode holds one KV cache "
            "per row, proportional to both",
            config.eval.batch_size,
            config.eval.max_new_tokens,
        )
        return 1
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("evaluation failed: %s", exc)
        return 1

    LOGGER.info("%s", outcome.metrics.describe())
    payload: dict[str, object] = {
        "split": args.split,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "quantization": config.model.quantization,
        "quantize_vision": config.model.quantize_vision,
        "max_pixels": config.image.max_pixels,
        "max_new_tokens": config.eval.max_new_tokens,
        **outcome.metrics.to_dict(),
    }
    if outcome.audit is not None:
        LOGGER.info("")
        LOGGER.info("%s", outcome.audit.describe())
        # Nested under one key rather than merged, so the field rates cannot be
        # mistaken for more CER-like numbers by a reader of the JSON.
        payload["field_audit"] = outcome.audit.to_dict()
    write_json(destination, payload)
    LOGGER.info("metrics     %s", destination)
    LOGGER.info("predictions %s", predictions_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
