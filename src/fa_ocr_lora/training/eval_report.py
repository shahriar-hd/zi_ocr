r"""Turn a comparison into a report bundle: CSV, JSON, Markdown tables and charts.

:mod:`fa_ocr_lora.training.compare_eval` writes one ``compare_<split>.json`` and two
predictions files. That is the right primary artefact - it is complete and it is
machine-readable - but it is not what a reader or a reviewer wants, and it is not
what survives a session teardown unless something copies it off the box. This
module is that something.

What it produces, all under one directory:

=================================  ===========================================
File                               Contents
=================================  ===========================================
``summary.json``                   Every headline number, one flat object
``overall.csv``                    Base vs adapter vs delta, one row per metric
``by_dataset.csv``                 The same split per dataset (letters, pixel)
``by_source.csv``                  And per manifest source tag
``fields.csv``                     Per-field recall/exact/digits, both variants
``confusions.csv``                 Top substitution pairs, both variants
``training_curve.csv``             Loss and eval-loss against step, if available
``report.md``                      The tables above, rendered for a human
``charts/*.png``                   The same comparisons, drawn
=================================  ===========================================

Charts are optional by construction: matplotlib is imported inside the function
that draws them, so a box without it still produces every table. That matters
because the tables are the deliverable and a missing font should not cost them.

The CSVs are deliberately long-format (``metric,variant,value``) rather than wide
where a reader might expect wide: a spreadsheet pivots long data trivially and
cannot unpivot wide data at all, and the number of datasets is not fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config.base import ConfigError
from ..dataset.common.io_utils import resolve_repo_path, write_json
from .config import load_training_config

__all__ = [
    "METRIC_LABELS",
    "build_bundle",
    "main",
    "parse_args",
    "write_charts",
]

LOGGER = logging.getLogger("fa_ocr_lora.eval_report")

#: Metrics promoted into the headline tables, in reading order, with the label
#: used in every rendering. Keys are :class:`~fa_ocr_lora.training.metrics.EvalMetrics`
#: field names. ``lower_is_better`` decides the arrow in the Markdown table and
#: the colour of the bar in the chart - error rates improve downwards, match
#: rates upwards, and a single table carries both.
METRIC_LABELS: tuple[tuple[str, str, bool], ...] = (
    ("cer", "CER", True),
    ("wer", "WER", True),
    ("cer_normalized", "CER (folded)", True),
    ("wer_normalized", "WER (folded)", True),
    ("digit_cer", "digit CER", True),
    ("digit_exact", "digit exact match", False),
    ("empty_predictions", "empty predictions", True),
)

#: Per-field rates lifted out of the audit, as ``(key, label)``. ``recall`` and
#: the two match rates are reference-based; ``malformed`` and ``low_confidence``
#: are not, and are kept because they are the two an inference deployment can
#: still compute.
FIELD_RATES: tuple[tuple[str, str], ...] = (
    ("reference_count", "references"),
    ("recall", "recall"),
    ("exact_rate", "exact"),
    ("digit_rate", "digits"),
    ("malformed_rate", "malformed"),
    ("low_confidence_rate", "low confidence"),
    ("mean_confidence", "mean confidence"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Render a comparison JSON as CSV, Markdown and charts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training YAML, used to locate the run directory.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which comparison to read: <output_dir>/compare_<split>.json.",
    )
    parser.add_argument(
        "--compare-json",
        type=Path,
        default=None,
        help="Comparison JSON to read; overrides --split resolution.",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional eval_<split>.json to fold in as a named variant, as "
            "name=path. Repeatable."
        ),
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=None,
        help=(
            "training_summary.json or trainer_state.json to read the loss curve "
            "from; defaults to both, in the run directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Bundle destination; defaults to <output_dir>/report_<split>.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Write tables only, skipping matplotlib entirely.",
    )
    return parser.parse_args(argv)


def _number(value: Any) -> float | None:
    """Coerce a JSON scalar to a float, or ``None`` when it is not one.

    Args:
        value: Whatever the JSON carried.

    Returns:
        The float, or ``None`` for ``None``, booleans and non-numeric values.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _delta(base: float | None, adapter: float | None) -> float | None:
    """Adapter minus base, so a negative delta is an improvement on error rates.

    Args:
        base: Base-model value.
        adapter: Adapter value.

    Returns:
        The difference, or ``None`` when either side is missing.
    """
    if base is None or adapter is None:
        return None
    return adapter - base


def _relative(base: float | None, adapter: float | None) -> float | None:
    """Fractional change against the base.

    Args:
        base: Base-model value.
        adapter: Adapter value.

    Returns:
        ``(adapter - base) / base``, or ``None`` when that is undefined.
    """
    if base is None or adapter is None or base == 0:
        return None
    return (adapter - base) / base


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> Path:
    """Write one CSV with a header.

    ``newline=""`` and ``utf-8-sig``: the first is what stops csv doubling line
    endings on Windows, the second is what makes Excel read Persian field values
    as UTF-8 instead of mojibake - these files are opened by people, not only by
    pandas.

    Args:
        path: Destination.
        header: Column names.
        rows: Row values.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _overall_rows(overall: Mapping[str, Any]) -> list[list[Any]]:
    """Build the base/adapter/delta table over the headline metrics.

    Args:
        overall: ``overall`` object of the comparison JSON.

    Returns:
        Rows of ``[metric, base, adapter, delta, relative]``.
    """
    base = overall.get("base") or {}
    adapter = overall.get("adapter") or {}
    rows: list[list[Any]] = []
    for key, label, _lower in METRIC_LABELS:
        before = _number(base.get(key))
        after = _number(adapter.get(key))
        rows.append(
            [label, before, after, _delta(before, after), _relative(before, after)]
        )
    for key, label in (("samples", "samples"), ("digit_samples", "digit records")):
        rows.append(
            [label, _number(base.get(key)), _number(adapter.get(key)), None, None]
        )
    return rows


def _group_rows(group: Mapping[str, Any]) -> list[list[Any]]:
    """Build a per-group table from ``by_dataset`` or ``by_source``.

    Args:
        group: Mapping of group name to ``{variant: metrics}``.

    Returns:
        Rows of ``[group, n, base CER, adapter CER, delta, base WER, adapter WER,
        delta, base digit CER, adapter digit CER, delta]``.
    """
    rows: list[list[Any]] = []
    for name in sorted(group):
        variants = group[name] or {}
        base = variants.get("base") or {}
        adapter = variants.get("adapter") or {}
        row: list[Any] = [name, _number(adapter.get("samples") or base.get("samples"))]
        for key in ("cer", "wer", "digit_cer", "digit_exact"):
            before = _number(base.get(key))
            after = _number(adapter.get(key))
            row.extend([before, after, _delta(before, after)])
        rows.append(row)
    return rows


def _field_rows(audit: Mapping[str, Any]) -> list[list[Any]]:
    """Build the per-field audit table for both variants.

    Args:
        audit: ``field_audit`` object of the comparison JSON.

    Returns:
        Rows of ``[field, rate, base, adapter, delta]``.
    """
    variants = {
        name: ((audit.get(name) or {}).get("fields") or {})
        for name in ("base", "adapter")
    }
    names = sorted({key for value in variants.values() for key in value})
    rows: list[list[Any]] = []
    for field in names:
        base = variants["base"].get(field) or {}
        adapter = variants["adapter"].get(field) or {}
        for key, label in FIELD_RATES:
            before = _number(base.get(key))
            after = _number(adapter.get(key))
            if before is None and after is None:
                continue
            rows.append([field, label, before, after, _delta(before, after)])
    return rows


def _confusion_rows(overall: Mapping[str, Any]) -> list[list[Any]]:
    """Build the confusion table for both variants.

    Args:
        overall: ``overall`` object of the comparison JSON.

    Returns:
        Rows of ``[variant, rank, expected, predicted, count]``.
    """
    rows: list[list[Any]] = []
    for variant in ("base", "adapter"):
        pairs = (overall.get(variant) or {}).get("confusions") or []
        for rank, entry in enumerate(pairs, start=1):
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue
            expected, predicted, count = entry[0], entry[1], entry[2]
            rows.append([variant, rank, expected, predicted, _number(count)])
    return rows


def _curve_rows(summary: Mapping[str, Any]) -> list[list[Any]]:
    """Extract a loss curve from a training summary or trainer state.

    Both shapes are accepted because both exist in a run directory and neither is
    guaranteed: the trainer writes ``trainer_state.json`` with a ``log_history``,
    and this project's loop writes ``training_summary.json``. A run that has
    neither simply gets no curve, which is why the caller treats this as optional.

    Args:
        summary: Parsed JSON of either file.

    Returns:
        Rows of ``[step, epoch, loss, eval_loss, learning_rate, vram_mib]``.
    """
    history = summary.get("log_history")
    if not isinstance(history, list):
        fallback = summary.get("history")
        history = fallback if isinstance(fallback, list) else []
    rows: list[list[Any]] = []
    for entry in history:
        if not isinstance(entry, Mapping):
            continue
        step = _number(entry.get("step"))
        if step is None:
            continue
        rows.append(
            [
                step,
                _number(entry.get("epoch")),
                _number(entry.get("loss")),
                _number(entry.get("eval_loss")),
                _number(entry.get("learning_rate")),
                _number(entry.get("vram_mib")),
            ]
        )
    rows.sort(key=lambda row: (row[0], row[3] is None))
    return rows


def _markdown(
    split: str,
    payload: Mapping[str, Any],
    overall_rows: Sequence[Sequence[Any]],
    dataset_rows: Sequence[Sequence[Any]],
    field_rows: Sequence[Sequence[Any]],
) -> str:
    """Render the bundle's tables as Markdown.

    Args:
        split: Split that was compared.
        payload: The whole comparison JSON.
        overall_rows: Output of :func:`_overall_rows`.
        dataset_rows: Output of :func:`_group_rows` over ``by_dataset``.
        field_rows: Output of :func:`_field_rows`.

    Returns:
        The document.
    """

    def cell(value: Any, places: int = 4) -> str:
        number = _number(value)
        if number is None:
            return "—" if value is None else str(value)
        # Counts are counts: "samples 5855.0000" reads as a rate that happens to
        # be large, which is exactly the confusion the audit exists to avoid.
        if places == 0 or (number == int(number) and abs(number) >= 1000):
            return f"{int(number):,}"
        return f"{number:.{places}f}"

    lines = [
        f"# Evaluation report — {split} split",
        "",
        f"- Adapter: `{payload.get('adapter', 'n/a')}`",
        f"- Config: `{payload.get('config', 'n/a')}`",
        f"- Records: {payload.get('records', {})}",
        "",
        "## Overall",
        "",
        "| Metric | Base | Base + LoRA | Δ | Relative |",
        "|---|---|---|---|---|",
    ]
    for row in overall_rows:
        relative = _number(row[4])
        lines.append(
            f"| {row[0]} | {cell(row[1])} | {cell(row[2])} | {cell(row[3])} | "
            f"{'—' if relative is None else f'{relative * 100:+.1f} %'} |"
        )
    lines += [
        "",
        "## By dataset",
        "",
        "| Dataset | n | Base CER | LoRA CER | Δ CER | Base WER | LoRA WER | Δ WER |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in dataset_rows:
        lines.append(
            f"| {row[0]} | {cell(row[1], 0)} | {cell(row[2])} | {cell(row[3])} | "
            f"{cell(row[4])} | {cell(row[5])} | {cell(row[6])} | {cell(row[7])} |"
        )
    if field_rows:
        lines += [
            "",
            "## Sensitive fields",
            "",
            "| Field | Rate | Base | Base + LoRA | Δ |",
            "|---|---|---|---|---|",
        ]
        for row in field_rows:
            lines.append(
                f"| {row[0]} | {row[1]} | {cell(row[2])} | {cell(row[3])} | "
                f"{cell(row[4])} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_charts(
    directory: Path,
    overall_rows: Sequence[Sequence[Any]],
    dataset_rows: Sequence[Sequence[Any]],
    field_rows: Sequence[Sequence[Any]],
    curve_rows: Sequence[Sequence[Any]],
) -> list[Path]:
    """Draw the comparison as PNGs.

    matplotlib is imported here rather than at module scope so that a host
    without it still gets every table - the charts are a convenience, the CSVs
    are the artefact.

    Args:
        directory: Destination directory, created if absent.
        overall_rows: Output of :func:`_overall_rows`.
        dataset_rows: Output of :func:`_group_rows` over ``by_dataset``.
        field_rows: Output of :func:`_field_rows`.
        curve_rows: Output of :func:`_curve_rows`.

    Returns:
        The files written, which is empty when matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is not installed; writing tables without charts")
        return []

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def grouped_bars(
        path: Path,
        labels: Sequence[str],
        base: Sequence[float],
        adapter: Sequence[float],
        title: str,
        ylabel: str,
    ) -> None:
        if not labels:
            return
        positions = range(len(labels))
        figure, axes = plt.subplots(figsize=(max(6.0, 1.3 * len(labels)), 4.2))
        offset = 0.2
        axes.bar(
            [p - offset for p in positions],
            base,
            width=0.38,
            label="base",
            color="#b0413e",
        )
        axes.bar(
            [p + offset for p in positions],
            adapter,
            width=0.38,
            label="base + LoRA",
            color="#2f6f4f",
        )
        axes.set_xticks(list(positions))
        axes.set_xticklabels(labels, rotation=20, ha="right")
        axes.set_ylabel(ylabel)
        axes.set_title(title)
        axes.legend()
        axes.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

    # Error rates and match rates on separate axes: a chart mixing a CER of 0.56
    # with a digit CER of 1.79 and a match rate of 0.0 makes the small bars
    # unreadable, and "lower is better" cannot be true of all of them at once.
    errors = [
        (row[0], _number(row[1]) or 0.0, _number(row[2]) or 0.0)
        for row in overall_rows
        if row[0] in {"CER", "WER", "CER (folded)", "WER (folded)", "digit CER"}
    ]
    grouped_bars(
        directory / "overall_error_rates.png",
        [name for name, _b, _a in errors],
        [b for _n, b, _a in errors],
        [a for _n, _b, a in errors],
        "Error rates — lower is better",
        "rate",
    )
    matches = [
        (row[0], _number(row[1]) or 0.0, _number(row[2]) or 0.0)
        for row in overall_rows
        if row[0] in {"digit exact match"}
    ]
    grouped_bars(
        directory / "overall_match_rates.png",
        [name for name, _b, _a in matches],
        [b for _n, b, _a in matches],
        [a for _n, _b, a in matches],
        "Match rates — higher is better",
        "rate",
    )
    grouped_bars(
        directory / "cer_by_dataset.png",
        [str(row[0]) for row in dataset_rows],
        [_number(row[2]) or 0.0 for row in dataset_rows],
        [_number(row[3]) or 0.0 for row in dataset_rows],
        "CER by dataset — lower is better",
        "CER",
    )
    recall = [row for row in field_rows if row[1] == "recall"]
    grouped_bars(
        directory / "field_recall.png",
        [str(row[0]) for row in recall],
        [_number(row[2]) or 0.0 for row in recall],
        [_number(row[3]) or 0.0 for row in recall],
        "Sensitive-field recall — higher is better",
        "recall",
    )
    exact = [row for row in field_rows if row[1] == "exact"]
    grouped_bars(
        directory / "field_exact.png",
        [str(row[0]) for row in exact],
        [_number(row[2]) or 0.0 for row in exact],
        [_number(row[3]) or 0.0 for row in exact],
        "Sensitive-field exact match — higher is better",
        "exact",
    )

    if curve_rows:
        steps = [row[0] for row in curve_rows if row[2] is not None]
        losses = [row[2] for row in curve_rows if row[2] is not None]
        eval_steps = [row[0] for row in curve_rows if row[3] is not None]
        eval_losses = [row[3] for row in curve_rows if row[3] is not None]
        if losses or eval_losses:
            figure, axes = plt.subplots(figsize=(7.5, 4.2))
            if losses:
                axes.plot(steps, losses, lw=0.8, alpha=0.6, label="train loss")
            if eval_losses:
                axes.plot(
                    eval_steps, eval_losses, lw=2.0, marker="o", ms=3, label="eval loss"
                )
            axes.set_xlabel("step")
            axes.set_ylabel("loss")
            axes.set_title("Training curve")
            axes.legend()
            axes.grid(alpha=0.3)
            figure.tight_layout()
            path = directory / "training_curve.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            written.append(path)
    return written


def build_bundle(
    compare_json: Path,
    destination: Path,
    *,
    split: str = "test",
    training_summary: Path | None = None,
    extra_evals: Mapping[str, Path] | None = None,
    charts: bool = True,
) -> dict[str, Any]:
    """Write the whole report bundle from a comparison JSON.

    Args:
        compare_json: ``compare_<split>.json`` produced by
            :mod:`fa_ocr_lora.training.compare_eval`.
        destination: Bundle directory, created if absent.
        split: Split label, used in titles and the summary.
        training_summary: Training summary or trainer state for the loss curve.
        extra_evals: Named ``eval_<split>.json`` files to record alongside, such
            as the single-variant validation passes.
        charts: Whether to draw PNGs.

    Returns:
        The summary object that was written, so a caller can print from it.

    Raises:
        FileNotFoundError: If the comparison JSON does not exist. Raised rather
            than warned past: an empty bundle looks like a finished one.
    """
    if not compare_json.is_file():
        raise FileNotFoundError(
            f"no comparison at {compare_json}; run the compare stage first "
            "(scripts/train_molab.sh compare --split test)"
        )
    payload = json.loads(compare_json.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)

    overall = payload.get("overall") or {}
    overall_rows = _overall_rows(overall)
    dataset_rows = _group_rows(payload.get("by_dataset") or {})
    source_rows = _group_rows(payload.get("by_source") or {})
    field_rows = _field_rows(payload.get("field_audit") or {})
    confusion_rows = _confusion_rows(overall)

    curve_rows: list[list[Any]] = []
    if training_summary is not None and training_summary.is_file():
        try:
            curve_rows = _curve_rows(
                json.loads(training_summary.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("could not read %s: %s", training_summary, exc)

    written: list[Path] = [
        _write_csv(
            destination / "overall.csv",
            ("metric", "base", "adapter", "delta", "relative"),
            overall_rows,
        ),
        _write_csv(
            destination / "by_dataset.csv",
            (
                "dataset",
                "n",
                "base_cer",
                "adapter_cer",
                "delta_cer",
                "base_wer",
                "adapter_wer",
                "delta_wer",
                "base_digit_cer",
                "adapter_digit_cer",
                "delta_digit_cer",
                "base_digit_exact",
                "adapter_digit_exact",
                "delta_digit_exact",
            ),
            dataset_rows,
        ),
        _write_csv(
            destination / "by_source.csv",
            (
                "source",
                "n",
                "base_cer",
                "adapter_cer",
                "delta_cer",
                "base_wer",
                "adapter_wer",
                "delta_wer",
                "base_digit_cer",
                "adapter_digit_cer",
                "delta_digit_cer",
                "base_digit_exact",
                "adapter_digit_exact",
                "delta_digit_exact",
            ),
            source_rows,
        ),
        _write_csv(
            destination / "fields.csv",
            ("field", "rate", "base", "adapter", "delta"),
            field_rows,
        ),
        _write_csv(
            destination / "confusions.csv",
            ("variant", "rank", "expected", "predicted", "count"),
            confusion_rows,
        ),
    ]
    if curve_rows:
        written.append(
            _write_csv(
                destination / "training_curve.csv",
                ("step", "epoch", "loss", "eval_loss", "learning_rate", "vram_mib"),
                curve_rows,
            )
        )

    base = overall.get("base") or {}
    adapter = overall.get("adapter") or {}
    summary: dict[str, Any] = {
        "split": split,
        "adapter": payload.get("adapter"),
        "config": payload.get("config"),
        "records": payload.get("records"),
        "headline": {
            label: {
                "base": _number(base.get(key)),
                "adapter": _number(adapter.get(key)),
                "delta": _delta(_number(base.get(key)), _number(adapter.get(key))),
                "relative": _relative(
                    _number(base.get(key)), _number(adapter.get(key))
                ),
                "lower_is_better": lower,
            }
            for key, label, lower in METRIC_LABELS
        },
        "comparison": payload.get("comparison"),
        "by_dataset": payload.get("by_dataset"),
        "field_audit_rates": {
            row[0]: {} for row in field_rows
        },
        "source_compare_json": str(compare_json),
    }
    for field, rate, before, after, delta in field_rows:
        summary["field_audit_rates"].setdefault(field, {})[rate] = {
            "base": before,
            "adapter": after,
            "delta": delta,
        }
    if extra_evals:
        folded: dict[str, Any] = {}
        for name, path in extra_evals.items():
            if not Path(path).is_file():
                LOGGER.warning("skipping %s: %s does not exist", name, path)
                continue
            folded[name] = json.loads(Path(path).read_text(encoding="utf-8"))
        if folded:
            summary["eval_runs"] = folded
    written.append(write_json(destination / "summary.json", summary))

    markdown = _markdown(split, payload, overall_rows, dataset_rows, field_rows)
    report_path = destination / "report.md"
    report_path.write_text(markdown, encoding="utf-8")
    written.append(report_path)

    if charts:
        written.extend(
            write_charts(
                destination / "charts",
                overall_rows,
                dataset_rows,
                field_rows,
                curve_rows,
            )
        )
    for path in written:
        LOGGER.info("wrote %s", path)
    summary["files"] = [str(path) for path in written]
    return summary


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        ``0`` on success, ``1`` on a configuration or missing-input error.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    try:
        config = load_training_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        LOGGER.error("configuration error: %s", exc)
        return 1

    run_dir = config.paths.output_dir
    compare_json = (
        resolve_repo_path(args.compare_json)
        if args.compare_json is not None
        else run_dir / f"compare_{args.split}.json"
    )
    destination = (
        resolve_repo_path(args.output_dir)
        if args.output_dir is not None
        else run_dir / f"report_{args.split}"
    )
    if args.training_summary is not None:
        summary_path: Path | None = resolve_repo_path(args.training_summary)
    else:
        candidates = [
            run_dir / "trainer_state.json",
            run_dir / "training_summary.json",
            run_dir / "logs" / "trainer_state.json",
        ]
        summary_path = next((path for path in candidates if path.is_file()), None)

    extra: dict[str, Path] = {}
    for entry in args.eval_json or ():
        text = str(entry)
        name, _, raw = text.partition("=")
        if not raw:
            name, raw = Path(text).stem, text
        extra[name] = resolve_repo_path(Path(raw))

    try:
        summary = build_bundle(
            compare_json,
            destination,
            split=args.split,
            training_summary=summary_path,
            extra_evals=extra,
            charts=not args.no_charts,
        )
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1

    headline = summary.get("headline", {})
    LOGGER.info("")
    LOGGER.info("%-20s %10s %10s %10s", "metric", "base", "adapter", "delta")
    for label, values in headline.items():
        base = values.get("base")
        adapter = values.get("adapter")
        delta = values.get("delta")
        LOGGER.info(
            "%-20s %10s %10s %10s",
            label,
            "—" if base is None else f"{base:.4f}",
            "—" if adapter is None else f"{adapter:.4f}",
            "—" if delta is None else f"{delta:+.4f}",
        )
    LOGGER.info("")
    LOGGER.info("bundle      %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
