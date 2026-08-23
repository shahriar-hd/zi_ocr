r"""Build the unified training manifest from the two local corpora.

Reads the letters JSONL and the Persian Pixel CSV, selects the configured mix,
assigns group-aware splits, weights digit-dense records, and writes
``unified_manifest.parquet`` plus ``split_summary.json``. Run this once before
training; the trainer reads only the manifest.

Examples:
    Full configured run::

        python -m fa_ocr_lora.training.prepare_manifest

    Exercise the whole pipeline on 200 records, keeping the source mix::

        python -m fa_ocr_lora.training.prepare_manifest --limit 200

    Retarget the corpora without editing the YAML::

        python -m fa_ocr_lora.training.prepare_manifest \\
            --letters-manifest data/other/pages.jsonl --output-dir /tmp/run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..config.base import ConfigError
from .config import TrainingConfig, load_training_config
from .manifest import ManifestSummary, build_manifest
from .sources import SOURCE_TAGS

__all__ = ["main", "parse_args"]

LOGGER = logging.getLogger("fa_ocr_lora.prepare_manifest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build the unified Persian OCR training manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training YAML; defaults to configs/lora_config.yaml.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total records, scaling the source mix proportionally.",
    )
    parser.add_argument("--letters-manifest", type=Path, default=None)
    parser.add_argument("--pixel-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--letters-pages", type=int, default=None)
    parser.add_argument("--letters-crops", type=int, default=None)
    parser.add_argument("--pixel-pages", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log every configuration value actually in effect.",
    )
    return parser.parse_args(argv)


def _apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Apply CLI flags on top of a loaded configuration.

    Flags win over both the YAML and the environment, which is the precedence
    documented in the config header. Rebuilding through ``from_dict`` rather than
    mutating means an override is validated by exactly the same rules as a YAML
    value - a negative ``--letters-pages`` fails here too.

    Args:
        config: Configuration as loaded from YAML plus environment.
        args: Parsed CLI arguments.

    Returns:
        A new configuration with the overrides applied.

    Raises:
        ConfigError: If an override is out of range.
    """
    payload = config.to_dict()
    # to_dict() records provenance that from_dict() deliberately refuses to
    # accept as an input section; carry it across out-of-band.
    payload.pop("source_path", None)
    mapping = {
        ("paths", "letters_manifest"): args.letters_manifest,
        ("paths", "pixel_metadata"): args.pixel_metadata,
        ("paths", "output_dir"): args.output_dir,
        ("data", "letters_pages"): args.letters_pages,
        ("data", "letters_crops"): args.letters_crops,
        ("data", "pixel_pages"): args.pixel_pages,
        ("data", "validation_size"): args.validation_size,
    }
    for (section, key), value in mapping.items():
        if value is not None:
            payload[section][key] = str(value) if isinstance(value, Path) else value
    if args.seed is not None:
        payload["seed"] = args.seed
    return TrainingConfig.from_dict(payload, source_path=config.source_path)


def _report(summary: ManifestSummary) -> None:
    """Print the mix, the splits and the rejections.

    Args:
        summary: What the build produced.
    """
    LOGGER.info("wrote %s", summary.manifest_path)
    LOGGER.info("      %s", summary.summary_path)
    LOGGER.info("records %d across %d groups", summary.rows, summary.groups)
    for split, counts in summary.mix.items():
        total = summary.per_split[split]
        if not total:
            continue
        detail = "  ".join(
            f"{tag.replace('admin_letter_', 'letter_')}={counts[tag]}"
            for tag in SOURCE_TAGS
            if counts[tag]
        )
        LOGGER.info("  %-11s %5d   %s", split, total, detail)
    if summary.digit_cut >= 0:
        LOGGER.info(
            "digit oversampling: %d train records at density >= %.3f",
            summary.oversampled,
            summary.digit_cut,
        )
    if summary.crop_allocation:
        served = {
            name: pair for name, pair in summary.crop_allocation.items() if pair[0] > 0
        }
        LOGGER.info("crop budget by region type (taken / available):")
        for name, (taken, available) in sorted(
            served.items(), key=lambda item: -item[1][0]
        ):
            LOGGER.info("  %-24s %5d / %7d", name, taken, available)
        unserved = sorted(set(summary.crop_allocation) - set(served))
        if unserved:
            LOGGER.info("  no budget reached: %s", ", ".join(unserved))
    if summary.skipped:
        LOGGER.info("rejected %d record(s):", sum(summary.skipped.values()))
        for reason, count in summary.skipped.items():
            LOGGER.info("  %-28s %7d", reason, count)
    else:
        LOGGER.info("no records rejected")


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        ``0`` on success, ``1`` on a configuration or corpus error.
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

    LOGGER.info("config      %s", config.source_path)
    LOGGER.info("letters     %s", config.paths.letters_manifest)
    LOGGER.info("pixel       %s", config.paths.pixel_metadata)
    LOGGER.info("output      %s", config.paths.output_dir)
    LOGGER.info(
        "budget      %d px = %d image tokens",
        config.image.max_pixels,
        config.image.max_image_tokens,
    )
    if args.verbose:
        LOGGER.debug("full configuration: %s", config.to_dict())

    try:
        summary = build_manifest(config, limit=args.limit)
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        LOGGER.error("manifest build failed: %s", exc)
        return 1
    _report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
