"""Stage 01 CLI - draw plans and report coverage.

Runs on the laptop, immediately after stage 00 and before any API spend.

The report it prints is the thing to read before launching generation. It
answers the two questions that decide whether the run is worth paying for:
*is the variety actually there* (axis spread, era and band spend), and *was any
of it silently impossible* (capacity clamping notes). A clamped era or a
saturated budget means the requested stratification did not happen, and it is
far cheaper to learn that here than from the finished corpus.

Usage::

    python -m fa_ocr_lora.diverse.planning.build
    python -m fa_ocr_lora.diverse.planning.build --total-samples 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import Final

from ...dataset.common.io_utils import write_json
from ..config import CATEGORIES, ERAS, DiverseConfig, load_diverse_config
from .sampler import PlanSampler, load_pool, write_plans

__all__ = ["build_plans", "main"]

LOGGER: Final[logging.Logger] = logging.getLogger("fa_diverse.plan")


def _axis_report(plans: Sequence[object]) -> dict[str, dict[str, int]]:
    """Tally every axis across the drawn plans.

    Args:
        plans: The drawn plans.

    Returns:
        Axis name to value counts, most common first.
    """
    tallies: dict[str, Counter[str]] = {
        "category": Counter(),
        "genre": Counter(),
        "register": Counter(),
        "persona": Counter(),
        "length_target": Counter(),
        "output_form": Counter(),
        "digit_script": Counter(),
        "profile": Counter(),
        "feature": Counter(),
    }
    for plan in plans:
        tallies["category"][plan.category] += 1  # type: ignore[attr-defined]
        tallies["genre"][plan.genre] += 1  # type: ignore[attr-defined]
        tallies["register"][plan.register] += 1  # type: ignore[attr-defined]
        tallies["persona"][plan.persona] += 1  # type: ignore[attr-defined]
        tallies["length_target"][plan.length_target] += 1  # type: ignore[attr-defined]
        tallies["output_form"][plan.output_form] += 1  # type: ignore[attr-defined]
        tallies["digit_script"][plan.digit_script] += 1  # type: ignore[attr-defined]
        tallies["profile"][plan.profile] += 1  # type: ignore[attr-defined]
        for feature in plan.features:  # type: ignore[attr-defined]
            tallies["feature"][feature] += 1
    return {name: dict(counter.most_common()) for name, counter in tallies.items()}


def _report(
    config: DiverseConfig,
    plans: Sequence[object],
    coverage: dict[str, object],
    axes: dict,
) -> None:
    """Print the stage 01 report.

    Args:
        config: The active configuration.
        plans: The drawn plans.
        coverage: Output of :meth:`PlanSampler.coverage`.
        axes: Output of :func:`_axis_report`.
    """
    total = len(plans)
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("PLAN SUMMARY  (%d sample(s))", total)
    LOGGER.info("=" * 70)

    LOGGER.info("")
    LOGGER.info("Category         target   actual")
    for name in CATEGORIES:
        want = config.plan.category_weights[name] * total
        got = axes["category"].get(name, 0)
        LOGGER.info("  %-13s %8.1f %8d", name, want, got)

    LOGGER.info("")
    LOGGER.info("Lexicon coverage")
    LOGGER.info("  pool                 %d", coverage["pool_size"])
    LOGGER.info(
        "  words used           %d  (%.1f%% of pool)",
        coverage["words_used"],
        100.0 * int(coverage["words_used"]) / max(1, int(coverage["pool_size"])),
    )
    LOGGER.info(
        "  met the use floor    %d  (floor=%d)",
        coverage["words_at_or_above_floor"],
        coverage["min_uses_per_word"],
    )
    LOGGER.info(
        "  hit the use ceiling  %d  (ceiling=%d)",
        coverage["words_at_ceiling"],
        coverage["max_uses_per_word"],
    )
    LOGGER.info("  total word slots     %d", coverage["total_word_slots"])

    LOGGER.info("")
    LOGGER.info("Era       requested  effective     spent")
    requested = coverage["era_weights_requested"]
    effective = coverage["era_weights_effective"]
    spend = coverage["spend_by_era"]
    slots = max(1, int(coverage["total_word_slots"]))
    for name in ERAS:
        LOGGER.info(
            "  %-10s %8.3f   %8.3f  %8.3f",
            name,
            requested.get(name, 0.0),  # type: ignore[union-attr]
            effective.get(name, 0.0),  # type: ignore[union-attr]
            spend.get(name, 0) / slots,  # type: ignore[union-attr]
        )

    LOGGER.info("")
    LOGGER.info("Band spend   %s", coverage["spend_by_band"])
    LOGGER.info("")
    LOGGER.info("Length       %s", axes["length_target"])
    LOGGER.info("Digit script %s", axes["digit_script"])
    LOGGER.info("Profile      %s", axes["profile"])
    LOGGER.info("Registers    %d distinct", len(axes["register"]))
    LOGGER.info("Personas     %d distinct", len(axes["persona"]))
    LOGGER.info("Genres       %d distinct", len(axes["genre"]))
    LOGGER.info("Features     %s", axes["feature"])

    notes = coverage["era_clamping_notes"]
    if notes:
        LOGGER.warning("")
        LOGGER.warning(
            "Capacity adjustments (the requested stratification was not fully met):"
        )
        for era, note in sorted(notes.items()):  # type: ignore[union-attr]
            LOGGER.warning("  %-11s %s", era, note)


def build_plans(config: DiverseConfig) -> dict[str, object]:
    """Run stage 01 end to end.

    Args:
        config: The active configuration.

    Returns:
        The coverage report, also written to
        :attr:`~fa_ocr_lora.diverse.config.PathsConfig.coverage_json`.
    """
    pool = load_pool(config.paths.lexicon_parquet)
    LOGGER.info("Loaded %d injectable word(s)", len(pool))

    sampler = PlanSampler(config, pool)
    plans = list(sampler.draw_all())
    written = write_plans(plans, config.paths.plans_jsonl)

    coverage = sampler.coverage(pool)
    axes = _axis_report(plans)
    report = {
        "run_name": config.run_name,
        "seed": config.seed,
        "total_samples": written,
        "coverage": coverage,
        "axes": axes,
    }
    write_json(config.paths.coverage_json, report)
    _report(config, plans, coverage, axes)

    LOGGER.info("")
    LOGGER.info("Wrote %s (%d plan(s))", config.paths.plans_jsonl, written)
    LOGGER.info("Wrote %s", config.paths.coverage_json)
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
        "--total-samples",
        type=int,
        default=None,
        help="override plan.total_samples (the pilot/scale knob)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    config = load_diverse_config(args.config)
    if args.total_samples is not None:
        config = replace(
            config, plan=replace(config.plan, total_samples=args.total_samples)
        )
    build_plans(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
