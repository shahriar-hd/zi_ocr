r"""Score the base model and a LoRA adapter over the same split, per dataset.

One CER answers "how much of the corpus was read" and hides the question this
project actually asks: whether the adapter moved Persian Pixel as much as it
moved letters, or moved it at all. So two variants - the untrained base and the
trained adapter - are generated over *identical* records (same split, same
``eval.max_samples`` slice, same greedy decoding), each writes its own
predictions file, and both files are re-scored per dataset with the same
:class:`~fa_ocr_lora.training.metrics.MetricAccumulator` the headline numbers
are built from. Per-dataset figures and totals are then one measurement rather
than two that happen to share a name.

Usage::

    python -m fa_ocr_lora.training.compare_eval --split test
    python -m fa_ocr_lora.training.compare_eval --split validation --max-samples 50

The base scored here is the model the adapter was trained from
(``model.model_id`` at ``model.revision``), so the difference isolates the
adapter: anything better is what training bought, and anything worse is what it
cost. The report states the signed difference (adapter minus base, negative
meaning the adapter won) and the relative change, and the field audit of both
variants is written to ``compare_<split>.json`` beside the per-record
predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import torch

from ..config.base import ConfigError
from ..dataset.common.io_utils import resolve_repo_path, write_json
from . import vram
from .config import TrainingConfig, load_training_config
from .eval import EvalOutcome, run_score_split
from .metrics import EvalMetrics, MetricAccumulator
from .modeling import load_eval_model
from .sources import LETTER_CROP, LETTER_PAGE, PIXEL_PAGE

__all__ = [
    "DATASETS",
    "aggregate",
    "compare_datasets",
    "dataset_for",
    "main",
    "parse_args",
    "read_predictions",
    "render",
    "score_variant",
]

LOGGER = logging.getLogger("fa_ocr_lora.compare_eval")

#: Source tag to the dataset it was drawn from. A tag that is not listed here is
#: reported under its own name rather than dropped, so a new source cannot
#: vanish from the comparison silently.
DATASETS: dict[str, str] = {
    LETTER_PAGE: "letters",
    LETTER_CROP: "letters",
    PIXEL_PAGE: "persian_pixel",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Score base and adapter over one split and compare them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training YAML; defaults to configs/lora_config.yaml.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test", "train"),
        default="test",
        help="Manifest split to score.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Records per variant; defaults to eval.max_samples.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Adapter to score; defaults to <output_dir>/adapter_only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report JSON destination; defaults to <output_dir>/compare_<split>.json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Prompts per generate() call, applied to both variants; overrides "
            "eval.batch_size. 1 scores one record at a time."
        ),
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=None,
        help=(
            "Worker processes encoding prompts ahead of the GPU; overrides "
            "eval.dataloader_workers."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log every prediction as it is produced.",
    )
    return parser.parse_args(argv)


def dataset_for(tag: str) -> str:
    """Map a manifest source tag to the dataset it came from.

    Args:
        tag: ``source`` column of one manifest record.

    Returns:
        ``"letters"``, ``"persian_pixel"``, or ``tag`` itself when the source is
        not one this module knows about.
    """
    return DATASETS.get(tag, tag)


def read_predictions(path: Path) -> list[dict[str, object]]:
    """Read a predictions file written by :func:`score_split`.

    Args:
        path: JSONL file to read.

    Returns:
        One mapping per scored record, each carrying ``source``, ``reference``
        and ``prediction``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate(
    records: Iterable[dict[str, object]],
    key: Callable[[str], str],
    confusion_top_k: int,
) -> dict[str, EvalMetrics]:
    """Re-score predictions under a grouping key.

    One accumulator per group rather than an average of per-record rates: CER is
    a ratio of counts, so averaging ratios would let a fifty-character crop
    weigh as much as a seven-hundred-character page.

    Args:
        records: Prediction mappings as returned by :func:`read_predictions`.
        key: Grouping function applied to each record's ``source``.
        confusion_top_k: Confusion pairs reported for each group.

    Returns:
        Metrics per group, ordered by group name.
    """
    buckets: dict[str, MetricAccumulator] = {}
    for row in records:
        group = key(str(row["source"]))
        bucket = buckets.setdefault(
            group, MetricAccumulator(confusion_top_k=confusion_top_k)
        )
        bucket.update(str(row["reference"]), str(row["prediction"]))
    return {group: bucket.result() for group, bucket in sorted(buckets.items())}


def compare_datasets(
    base: dict[str, EvalMetrics], adapter: dict[str, EvalMetrics]
) -> list[dict[str, object]]:
    """Pair two per-dataset reports, dataset by dataset.

    Args:
        base: Metrics per dataset for the untrained model.
        adapter: The same for the adapter, scored over the same records.

    Returns:
        One mapping per dataset named in either report, ordered by name, holding
        each variant's sample count and rates plus ``delta_cer`` (adapter minus
        base, so negative means the adapter won) and ``cer_change``, the
        difference as a fraction of the base rate - ``None`` when the base
        scored zero or only one variant has the dataset.
    """
    rows: list[dict[str, object]] = []
    for name in sorted(set(base) | set(adapter)):
        before = base.get(name)
        after = adapter.get(name)
        delta = after.cer - before.cer if before is not None and after else None
        change = None
        if delta is not None and before is not None and before.cer:
            change = delta / before.cer
        rows.append(
            {
                "dataset": name,
                "samples": {
                    "base": before.samples if before is not None else 0,
                    "adapter": after.samples if after is not None else 0,
                },
                "base": before.to_dict() if before is not None else None,
                "adapter": after.to_dict() if after is not None else None,
                "delta_cer": round(delta, 5) if delta is not None else None,
                "cer_change": round(change, 5) if change is not None else None,
            }
        )
    return rows


def score_variant(
    config: TrainingConfig,
    *,
    adapter_dir: Path | None,
    split: str,
    max_samples: int | None,
    predictions_path: Path,
    verbose: bool = False,
) -> EvalOutcome:
    """Load one variant, score it, and release it before the next one loads.

    The base pass and the adapter pass run in sequence rather than side by side
    so peak memory stays that of one model plus one adapter - which is what lets
    this run on a card that is already holding 95 GB for training.

    Args:
        config: Loaded training configuration.
        adapter_dir: Adapter directory, or ``None`` for the untrained base.
        split: Manifest split to score.
        max_samples: Records to score; ``None`` uses ``eval.max_samples``.
        predictions_path: Where to write this variant's predictions.
        verbose: Log every prediction as it is produced.

    Returns:
        The metrics and, unless ``eval.field_audit`` is off, the field audit.

    Raises:
        FileNotFoundError: If the manifest or the adapter is missing.
        ValueError: If the split is empty.
    """
    loaded = load_eval_model(config, adapter_dir=adapter_dir)
    try:
        LOGGER.info("%s", loaded.describe())
        return run_score_split(
            config,
            loaded,
            split=split,
            max_samples=max_samples,
            predictions_path=predictions_path,
            verbose=verbose,
        )
    finally:
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def render(
    split: str,
    adapter_dir: Path,
    overall: dict[str, EvalMetrics],
    base_by_dataset: dict[str, EvalMetrics],
    adapter_by_dataset: dict[str, EvalMetrics],
) -> str:
    """Format the comparison as one aligned table.

    Args:
        split: Manifest split that was scored.
        adapter_dir: Adapter directory, printed so a quoted number always says
            which weights produced it.
        overall: Metrics per variant over the whole split.
        base_by_dataset: Per-dataset metrics for the untrained model.
        adapter_by_dataset: The same for the adapter.

    Returns:
        The report, ready to log.
    """
    base_all = overall["base"]
    adapter_all = overall["adapter"]
    lines = [
        f"== compare {split}",
        f"adapter    {adapter_dir}",
        f"records    {base_all.samples} base | {adapter_all.samples} adapter",
        "",
        f"{'dataset':<16} {'n':>4} | {'base CER':>8} {'WER':>7}"
        f" | {'adapter CER':>11} {'WER':>7} | {'delta CER':>16}",
    ]

    def _row(label: str, before: EvalMetrics, after: EvalMetrics) -> tuple[str, float]:
        delta = after.cer - before.cer
        change = f" ({delta / before.cer:+.0%})" if before.cer else ""
        text = (
            f"{label:<16} {max(before.samples, after.samples):>4} |"
            f" {before.cer:>8.4f} {before.wer:>7.4f}"
            f" | {after.cer:>11.4f} {after.wer:>7.4f} |"
            f" {delta:>+8.4f}{change}"
        )
        return text, delta

    verdict: list[tuple[str, float]] = []
    for name in sorted(set(base_by_dataset) | set(adapter_by_dataset)):
        before = base_by_dataset.get(name)
        after = adapter_by_dataset.get(name)
        if before is None or after is None:
            lines.append(f"{name:<16} {'-':>4} | only one variant scored this")
            continue
        text, delta = _row(name, before, after)
        lines.append(text)
        verdict.append((name, delta))

    text, delta = _row("ALL", base_all, adapter_all)
    lines.append(text)

    if base_all.digit_samples or adapter_all.digit_samples:
        lines.append(
            f"digit       base CER {base_all.digit_cer:.4f}"
            f" exact {base_all.digit_exact:.4f}"
            f" -> adapter CER {adapter_all.digit_cer:.4f}"
            f" exact {adapter_all.digit_exact:.4f}"
        )
        # Run accuracy is bounded where digit_cer is not, so when the model
        # invents numbers the two disagree and this is the line to believe.
        lines.append(
            f"digit runs  base {base_all.digit_runs_matched}/{base_all.digit_runs}"
            f" ({base_all.digit_run_accuracy:.4f})"
            f" -> adapter {adapter_all.digit_runs_matched}/{adapter_all.digit_runs}"
            f" ({adapter_all.digit_run_accuracy:.4f})"
        )

    # How much of each variant's error is orthographic convention rather than
    # misreading. A large base share with a small adapter share is the adapter
    # having learned the corpus's spelling of ی and ک - real, but not reading.
    lines.append(
        f"folded CER  base {base_all.cer_normalized:.4f}"
        f" ({base_all.convention_share:.0%} of its error was convention)"
        f" -> adapter {adapter_all.cer_normalized:.4f}"
        f" ({adapter_all.convention_share:.0%})"
    )

    if not verdict:
        lines.append("verdict     no dataset scored by both variants")
        return "\n".join(lines)
    better = [name for name, value in verdict if value < 0]
    worse = [name for name, value in verdict if value > 0]
    summary = f"verdict     adapter better on {len(better)} of {len(verdict)} datasets"
    if better:
        summary += f" ({', '.join(better)})"
    if worse:
        summary += f"; worse on {len(worse)} ({', '.join(worse)})"
    if delta:
        share = delta / base_all.cer if base_all.cer else 0.0
        summary += f"; overall {delta:+.4f} CER ({share:+.0%})"
    lines.append(summary)

    # The distinction that one CER cannot make. An adapter can cut the strict
    # rate hard by learning to emit ی where the base emitted ي, which is worth
    # having but is not improved reading; only the folded delta shows the latter.
    folded_delta = adapter_all.cer_normalized - base_all.cer_normalized
    if delta < 0 <= folded_delta:
        lines.append(
            f"            but folded CER {folded_delta:+.4f}: the gain is"
            " orthographic convention, not better reading"
        )
    elif folded_delta < 0:
        lines.append(
            f"            folded CER {folded_delta:+.4f} too, so reading"
            " genuinely improved"
        )
    return "\n".join(lines)



def main(argv: list[str] | None = None) -> int:
    """Run the comparison.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        ``0`` on success, ``1`` on a configuration, corpus or adapter error.
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
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.eval_workers is not None:
        overrides["dataloader_workers"] = args.eval_workers
    if overrides:
        # Round-tripped through the loader rather than mutated in place, so a bad
        # flag is rejected by the same validation the YAML goes through - and both
        # variants are scored under one identical config, which is the whole
        # premise of the comparison.
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

    adapter_dir = (
        resolve_repo_path(args.adapter_dir)
        if args.adapter_dir is not None
        else config.paths.adapter_dir
    )
    if not adapter_dir.is_dir():
        LOGGER.error("no adapter at %s", adapter_dir)
        LOGGER.error("run the train stage first, or pass --adapter-dir")
        return 1

    report = vram.probe_vram(config.model, config.image)
    LOGGER.info("%s", vram.banner(report, config.model))

    destination = (
        resolve_repo_path(args.output)
        if args.output is not None
        else config.paths.output_dir / f"compare_{args.split}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    variants: dict[str, Path | None] = {"base": None, "adapter": adapter_dir}
    outcomes: dict[str, EvalOutcome] = {}
    records: dict[str, list[dict[str, object]]] = {}
    for name, directory in variants.items():
        predictions = destination.parent / f"{destination.stem}_{name}_predictions.jsonl"
        LOGGER.info("-- variant: %s", name)
        try:
            outcomes[name] = score_variant(
                config,
                adapter_dir=directory,
                split=args.split,
                max_samples=args.max_samples,
                predictions_path=predictions,
                verbose=args.verbose,
            )
        except (ConfigError, FileNotFoundError, ValueError) as exc:
            LOGGER.error("%s variant failed: %s", name, exc)
            return 1
        records[name] = read_predictions(predictions)

    top_k = config.eval.confusion_top_k
    by_dataset = {
        name: aggregate(rows, dataset_for, top_k) for name, rows in records.items()
    }
    by_source = {
        name: aggregate(rows, lambda tag: tag, top_k) for name, rows in records.items()
    }
    overall = {name: outcome.metrics for name, outcome in outcomes.items()}
    LOGGER.info(
        "%s",
        render(
            args.split,
            adapter_dir,
            overall,
            by_dataset["base"],
            by_dataset["adapter"],
        ),
    )

    payload = {
        "split": args.split,
        "config": str(config.source_path),
        "adapter": str(adapter_dir),
        "records": {name: len(value) for name, value in records.items()},
        "overall": {name: metrics.to_dict() for name, metrics in overall.items()},
        "comparison": compare_datasets(by_dataset["base"], by_dataset["adapter"]),
        "by_dataset": {
            dataset: {
                name: variant[dataset].to_dict() if dataset in variant else None
                for name, variant in by_dataset.items()
            }
            for dataset in sorted({key for v in by_dataset.values() for key in v})
        },
        "by_source": {
            tag: {
                name: variant[tag].to_dict() if tag in variant else None
                for name, variant in by_source.items()
            }
            for tag in sorted({key for v in by_source.values() for key in v})
        },
        "field_audit": {
            name: outcome.audit.to_dict() if outcome.audit else None
            for name, outcome in outcomes.items()
        },
    }
    written = write_json(destination, payload)
    LOGGER.info("report      %s", written)
    LOGGER.info(
        "predictions %s_*_predictions.jsonl",
        destination.with_suffix(""),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
