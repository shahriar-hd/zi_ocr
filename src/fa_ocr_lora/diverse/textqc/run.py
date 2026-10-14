"""Stage 03 CLI - normalise, filter, dedupe, and report honestly.

Runs on the laptop. No API calls, so it is cheap to re-run whenever a threshold
moves - which is the point of keeping the response cache in stage 02.

Two outputs. ``texts_clean.jsonl`` carries the survivors with both text fields
filled: ``text_render`` is exactly what the renderer will draw and therefore
what the label says, and ``text_norm`` is the evaluation form.
``rejections.jsonl`` carries one row per dropped sample with its reason, which
is what makes the rejection report actionable.

The stage refuses to pass silently. Above
``text_qc.max_rejection_rate`` it exits non-zero with the reason breakdown,
because a 40% rejection rate means a prompt or a model is broken and the right
response is to fix it, not to generate 250k samples hoping enough survive.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

from ...dataset.common.io_utils import write_json
from ..config import CATEGORIES, DiverseConfig, load_diverse_config
from ..planning.sampler import Plan
from ..text_policy import (
    detect_digit_scripts,
    diacritic_ratio,
    digit_density,
    has_diacritics,
    has_zwnj,
    normalise_for_eval,
    persian_letter_ratio,
    prepare_render_text,
    word_count,
)
from .dedupe import MinHashIndex
from .filters import REJECTION_REASONS, run_filters, strip_artefacts

__all__ = ["clean_texts", "main"]

LOGGER: Final[logging.Logger] = logging.getLogger("fa_diverse.textqc")


def _iter_raw(path: Path) -> Iterator[dict[str, object]]:
    """Stream the stage 02 records.

    Args:
        path: ``texts.jsonl``.

    Yields:
        One record per line.

    Raises:
        FileNotFoundError: If stage 02 has not run.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `python -m fa_ocr_lora.diverse.generation.run` first"
        )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_record(
    raw: dict[str, object],
    plan: Plan,
    text_clean: str,
    config: DiverseConfig,
    metrics: dict[str, object],
) -> dict[str, object]:
    """Build one surviving record, with both text fields and all metadata.

    Args:
        raw: The stage 02 record.
        plan: The sample's plan.
        text_clean: The text after artefact stripping and filtering.
        config: The active configuration.
        metrics: Measured values from the filters.

    Returns:
        The record written to ``texts_clean.jsonl``.
    """
    # text_render is produced by the one policy, with the plan's own flags. It
    # is what the renderer draws, so nothing downstream may alter it.
    text_render = prepare_render_text(
        text_clean,
        digit_script=plan.digit_script,
        keep_diacritics=plan.keep_diacritics,
        keep_zwnj=config.text_policy.keep_zwnj and not plan.drop_zwnj,
        arabic_forms=plan.arabic_forms,
    )
    text_norm = normalise_for_eval(text_render)
    scripts = sorted(detect_digit_scripts(text_render))

    return {
        "sample_id": plan.plan_id.replace("plan_", "fa_"),
        "plan_id": plan.plan_id,
        "index": plan.index,
        "group_id": plan.plan_id,
        "text_render": text_render,
        "text_norm": text_norm,
        "n_chars": len(text_render),
        "n_words": word_count(text_render),
        "persian_ratio": round(persian_letter_ratio(text_render), 4),
        "digit_script": plan.digit_script,
        "digit_scripts_present": scripts,
        "digit_density": round(digit_density(text_render), 4),
        "has_zwnj": has_zwnj(text_render),
        "has_diacritics": has_diacritics(text_render),
        "diacritic_ratio": round(diacritic_ratio(text_render), 4),
        "arabic_forms": plan.arabic_forms,
        "category": plan.category,
        "genre": plan.genre,
        "register": plan.register,
        "persona": plan.persona,
        "length_target": plan.length_target,
        "output_form": plan.output_form,
        "features": plan.features,
        "lexicon_word_ids": plan.word_ids,
        "lexicon_words": plan.words,
        "domain_tags": sorted({domain for domain in plan.word_domains if domain}),
        "word_eras": plan.word_eras,
        "word_bands": plan.word_bands,
        "rare_word_count": sum(
            1 for band in plan.word_bands if band in {"rare", "very_rare"}
        ),
        "lexicon_hit_rate": metrics.get("lexicon_hit_rate", 0.0),
        "llm_model": raw.get("llm_model", ""),
        "profile": raw.get("profile", ""),
        "temperature": raw.get("temperature", 0.0),
        "prompt_hash": raw.get("prompt_hash", ""),
        "seed": config.seed,
    }


def clean_texts(config: DiverseConfig) -> dict[str, object]:
    """Run stage 03 end to end.

    Args:
        config: The active configuration.

    Returns:
        The report, also written beside the outputs.
    """
    settings = config.text_qc
    index = MinHashIndex(permutations=settings.minhash_permutations, seed=config.seed)
    # One index per genre as well as the global one, so the stricter
    # within-genre threshold can be applied where repetition concentrates.
    genre_indexes: dict[str, MinHashIndex] = {}

    accepted: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    seen_exact: dict[str, str] = {}
    total = 0

    for raw in _iter_raw(config.paths.texts_jsonl):
        total += 1
        plan = Plan(**raw["plan"])  # type: ignore[arg-type]
        text_clean = strip_artefacts(str(raw.get("text_raw", "")))

        reason, metrics = run_filters(text_clean, plan, settings)

        if reason is None:
            # Exact duplicate first: cheaper than MinHash and unambiguous.
            fingerprint = normalise_for_eval(text_clean)
            previous = seen_exact.get(fingerprint)
            if previous is not None:
                reason = f"duplicate: exact match with {previous}"
            else:
                seen_exact[fingerprint] = plan.plan_id

        if reason is None:
            signature = index.signature_of(text_clean, settings.shingle_size)
            genre_index = genre_indexes.setdefault(
                plan.genre,
                MinHashIndex(
                    permutations=settings.minhash_permutations, seed=config.seed
                ),
            )
            match = genre_index.query(signature, settings.dedupe_threshold_same_genre)
            if match is not None:
                reason = (
                    f"duplicate: {match[1]:.2f} similar to {match[0]} within genre "
                    f"{plan.genre!r} (limit {settings.dedupe_threshold_same_genre:.2f})"
                )
            else:
                match = index.query(signature, settings.dedupe_threshold)
                if match is not None:
                    reason = (
                        f"duplicate: {match[1]:.2f} similar to {match[0]} "
                        f"(limit {settings.dedupe_threshold:.2f})"
                    )
            if reason is None:
                index.add(plan.plan_id, signature)
                genre_index.add(plan.plan_id, signature)

        if reason is not None:
            reasons[reason.split(":", 1)[0]] += 1
            rejections.append(
                {
                    "plan_id": plan.plan_id,
                    "category": plan.category,
                    "genre": plan.genre,
                    "profile": raw.get("profile", ""),
                    "llm_model": raw.get("llm_model", ""),
                    "reason": reason,
                    "metrics": metrics,
                    "text_head": text_clean[:200],
                }
            )
            continue

        accepted.append(_build_record(raw, plan, text_clean, config, metrics))

    _write_jsonl(accepted, config.paths.texts_clean_jsonl)
    _write_jsonl(rejections, config.paths.rejections_jsonl)

    report = _report(config, total, accepted, rejections, reasons)
    write_json(config.paths.work_dir / config.paths.TEXTQC / "textqc_report.json", report)
    return report


def _write_jsonl(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write rows to JSONL atomically.

    Args:
        rows: The rows.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _report(
    config: DiverseConfig,
    total: int,
    accepted: Sequence[dict[str, object]],
    rejections: Sequence[dict[str, object]],
    reasons: Counter[str],
) -> dict[str, object]:
    """Build and print the stage 03 report.

    Args:
        config: The active configuration.
        total: Samples read.
        accepted: Surviving records.
        rejections: Dropped records.
        reasons: Rejection reason tallies.

    Returns:
        The JSON-serialisable report.
    """
    rate = len(rejections) / max(1, total)
    by_model: Counter[str] = Counter()
    for row in rejections:
        by_model[str(row.get("llm_model", "?"))] += 1
    model_totals: Counter[str] = Counter()
    for row in accepted:
        model_totals[str(row.get("llm_model", "?"))] += 1
    for row in rejections:
        model_totals[str(row.get("llm_model", "?"))] += 1

    report = {
        "run_name": config.run_name,
        "read": total,
        "accepted": len(accepted),
        "rejected": len(rejections),
        "rejection_rate": round(rate, 4),
        "max_rejection_rate": config.text_qc.max_rejection_rate,
        "by_reason": {name: reasons.get(name, 0) for name in REJECTION_REASONS},
        "rejection_rate_by_model": {
            model: round(by_model.get(model, 0) / count, 4)
            for model, count in model_totals.most_common()
        },
        "accepted_by_category": {
            name: sum(1 for row in accepted if row["category"] == name)
            for name in CATEGORIES
        },
        "accepted_by_profile": dict(
            Counter(str(row["profile"]) for row in accepted).most_common()
        ),
        "digit_script": dict(
            Counter(str(row["digit_script"]) for row in accepted).most_common()
        ),
        "with_zwnj": sum(1 for row in accepted if row["has_zwnj"]),
        "with_diacritics": sum(1 for row in accepted if row["has_diacritics"]),
        "arabic_forms": sum(1 for row in accepted if row["arabic_forms"]),
        "mean_words": round(
            sum(int(row["n_words"]) for row in accepted) / max(1, len(accepted)), 1
        ),
        "mean_lexicon_hit_rate": round(
            sum(float(row["lexicon_hit_rate"]) for row in accepted)
            / max(1, len(accepted)),
            3,
        ),
        "mean_rare_words": round(
            sum(int(row["rare_word_count"]) for row in accepted) / max(1, len(accepted)),
            2,
        ),
    }

    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("TEXT QC REPORT")
    LOGGER.info("=" * 70)
    LOGGER.info("read      %d", total)
    LOGGER.info("accepted  %d", len(accepted))
    LOGGER.info("rejected  %d  (%.1f%%)", len(rejections), rate * 100.0)
    LOGGER.info("")
    LOGGER.info("By reason:")
    for name in REJECTION_REASONS:
        count = reasons.get(name, 0)
        if count:
            LOGGER.info(
                "  %-16s %5d  (%.1f%%)", name, count, 100.0 * count / max(1, total)
            )
    LOGGER.info("")
    LOGGER.info("Rejection rate per model (the signal for dropping a profile):")
    for model, value in report["rejection_rate_by_model"].items():  # type: ignore[union-attr]
        LOGGER.info("  %-42s %.1f%%", model, value * 100.0)
    LOGGER.info("")
    LOGGER.info("Accepted by category:  %s", report["accepted_by_category"])
    LOGGER.info("Digit script:          %s", report["digit_script"])
    LOGGER.info(
        "ZWNJ %d | diacritics %d | Arabic forms %d",
        report["with_zwnj"],
        report["with_diacritics"],
        report["arabic_forms"],
    )
    LOGGER.info(
        "mean words %.1f | mean lexicon hit rate %.3f | mean rare words %.2f",
        report["mean_words"],
        report["mean_lexicon_hit_rate"],
        report["mean_rare_words"],
    )
    LOGGER.info("")
    LOGGER.info("Wrote %s", config.paths.texts_clean_jsonl)
    LOGGER.info("Wrote %s", config.paths.rejections_jsonl)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="path to diverse_config.yaml")
    parser.add_argument(
        "--allow-high-rejection",
        action="store_true",
        help="exit 0 even above text_qc.max_rejection_rate (for a deliberate experiment)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; non-zero when the rejection rate is too high to
        continue without a look at the reasons.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    config = load_diverse_config(args.config)
    report = clean_texts(config)

    rate = float(report["rejection_rate"])
    ceiling = float(report["max_rejection_rate"])
    if rate > ceiling and not args.allow_high_rejection:
        LOGGER.error("")
        LOGGER.error(
            "Rejection rate %.1f%% exceeds the %.0f%% ceiling. Stopping rather than "
            "continuing: at this rate a prompt or a model is broken, and scaling up "
            "would spend the budget on samples that will be discarded. Read "
            "%s, fix the cause, then re-run (stage 02 is cached, so a re-run is free).",
            rate * 100.0,
            ceiling * 100.0,
            config.paths.rejections_jsonl,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
