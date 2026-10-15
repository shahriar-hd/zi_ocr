"""Stage 07 - splits, shards and the dataset card.

Two things here decide whether the corpus can be trusted to measure anything.

**Group-aware splitting.** A page and every crop taken from it share a
``group_id``, and so do all samples derived from one plan. The split is a hash
of that key, so a group never straddles train and test. Without it a model
would meet a line crop in training and its own parent page in test, and the
test score would be a memorisation score.

**The hard tests.** A random test split measures interpolation. These measure
the thing the previous dataset could not:

``test_unseen_fonts``
    Rendered in font families held out of training entirely.
``test_unseen_templates``
    Layouts held out of training entirely.
``test_unseen_generator``
    Text from an LLM profile held out of training entirely.

Membership was decided upstream - at layout time for fonts and templates, at
planning time for the generator - because by the time a sample reaches this
stage it is already rendered. Here they are only routed.

Output is sharded Parquet, one directory per category, which the Hub reads as
one loadable config each: ``load_dataset(repo, "chat")``. Images are embedded
in the Parquet rather than referenced, because a Hub dataset of loose files
hits the per-repo file ceiling long before it hits the byte ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ...dataset.common.io_utils import write_json
from ..config import CATEGORIES, DiverseConfig, load_diverse_config

__all__ = ["SPLITS", "assign_split", "package", "main"]

LOGGER: Final[logging.Logger] = logging.getLogger("fa_diverse.package")

#: Every split a sample can land in. The three ``test_unseen_*`` splits are
#: populated from upstream flags, not by sampling.
SPLITS: Final[tuple[str, ...]] = (
    "train",
    "validation",
    "test",
    "test_unseen_fonts",
    "test_unseen_templates",
    "test_unseen_generator",
)

#: Manifest columns. One row per sample, written once at the root so the whole
#: corpus can be filtered, counted and audited without opening a shard.
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "group_id",
    "parent_id",
    "category",
    "template_id",
    "split",
    "text_render",
    "text_norm",
    "n_chars",
    "n_words",
    "n_lines",
    "digit_script",
    "digit_density",
    "has_zwnj",
    "has_diacritics",
    "arabic_forms",
    "genre",
    "register",
    "persona",
    "length_target",
    "lexicon_word_ids",
    "domain_tags",
    "rare_word_count",
    "font_families",
    "font_licences",
    "dpi",
    "width",
    "height",
    "image_format",
    "clean",
    "dark_mode",
    "contrast",
    "llm_model",
    "profile",
    "temperature",
    "prompt_hash",
    "seed",
    "shard_file",
)


def assign_split(
    group_id: str,
    *,
    ratios: dict[str, float],
    salt: str,
    holdout_font: bool,
    holdout_template: bool,
    holdout_generator: bool,
) -> str:
    """Decide which split one sample's group belongs to.

    The three holdout flags win over the random draw, and in a fixed order, so
    a sample that is both unseen-font and unseen-template lands in exactly one
    split rather than being counted twice.

    Args:
        group_id: The splitting key, shared by a page and its crops.
        ratios: Train/validation/test shares.
        salt: Changes the deal without changing the ratios.
        holdout_font: Whether this sample used a held-out font family.
        holdout_template: Whether it used a held-out template.
        holdout_generator: Whether its text came from the held-out profile.

    Returns:
        A value from :data:`SPLITS`.
    """
    if holdout_font:
        return "test_unseen_fonts"
    if holdout_template:
        return "test_unseen_templates"
    if holdout_generator:
        return "test_unseen_generator"

    # blake2b rather than hash(): the built-in is salted per process, so the
    # split would change between runs over the same data.
    digest = hashlib.blake2b(f"{salt}:{group_id}".encode(), digest_size=8).digest()
    position = int.from_bytes(digest, "big") / float(1 << 64)
    cumulative = 0.0
    for name in ("train", "validation", "test"):
        cumulative += ratios.get(name, 0.0)
        if position < cumulative:
            return name
    return "test"


def _iter_accepted(path: Path) -> Iterator[dict[str, Any]]:
    """Stream the stage 06 output.

    Args:
        path: ``accepted.jsonl``.

    Yields:
        One record per line.

    Raises:
        FileNotFoundError: If stage 06 has not run.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `python -m fa_ocr_lora.diverse.qc.run` first"
        )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


@dataclass(slots=True)
class _ShardWriter:
    """Accumulates rows and flushes them to a Parquet shard.

    Attributes:
        directory: Where the shards go.
        split: Which split these rows belong to.
        shard_samples: Soft sample ceiling.
        max_bytes: Hard byte ceiling.
        compression: Parquet codec.
    """

    directory: Path
    split: str
    shard_samples: int
    max_bytes: int
    compression: str
    rows: list[dict[str, Any]] = None  # type: ignore[assignment]
    pending_bytes: int = 0
    index: int = 0
    written: list[tuple[str, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialise the mutable accumulators."""
        self.rows = []
        self.written = []

    def add(self, row: dict[str, Any], payload_bytes: int) -> None:
        """Buffer one row, flushing when a ceiling is reached.

        Args:
            row: The row.
            payload_bytes: Size of its image, for the byte ceiling.
        """
        self.rows.append(row)
        self.pending_bytes += payload_bytes
        if len(self.rows) >= self.shard_samples or self.pending_bytes >= self.max_bytes:
            self.flush()

    def flush(self) -> None:
        """Write the buffered rows as one shard."""
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"{self.split}-{self.index:05d}.parquet"
        table = pa.Table.from_pylist(self.rows)
        pq.write_table(table, self.directory / name, compression=self.compression)
        self.written.append((name, len(self.rows)))
        LOGGER.info(
            "    %s  %d sample(s), %.1f MB",
            name,
            len(self.rows),
            self.pending_bytes / 1e6,
        )
        self.rows = []
        self.pending_bytes = 0
        self.index += 1

    def finalise(self) -> list[tuple[str, int]]:
        """Flush and rename shards to the ``-of-NNNNN`` convention.

        The total is unknowable while streaming, so shards are written under a
        provisional name and renamed once the count is known. The Hub does not
        require the convention, but ``datasets`` users expect it and it makes a
        truncated upload obvious.

        Returns:
            ``(filename, sample_count)`` pairs, after renaming.
        """
        self.flush()
        total = len(self.written)
        renamed: list[tuple[str, int]] = []
        for position, (name, count) in enumerate(self.written):
            final = f"{self.split}-{position:05d}-of-{total:05d}.parquet"
            (self.directory / name).replace(self.directory / final)
            renamed.append((final, count))
        self.written = renamed
        return renamed


def _row_for(
    record: dict[str, Any],
    split: str,
    config: DiverseConfig,
    licences: dict[str, str],
) -> tuple[dict[str, Any], int]:
    """Build one Parquet row, with the image embedded.

    Args:
        record: An accepted stage 06 record.
        split: Its split.
        config: The active configuration.
        licences: Font family to licence, for the row's provenance.

    Returns:
        ``(row, payload_bytes)``.

    Raises:
        OSError: If the image cannot be read.
    """
    from ..text_policy import digit_density, normalise_for_eval, word_count

    image_file = config.paths.work_dir / str(record["image_path"])
    payload = image_file.read_bytes()
    text = str(record.get("text_render", ""))
    families = list(record.get("font_families", ()))
    aug = record.get("aug", {})

    row = {
        "sample_id": str(record["sample_id"]),
        "group_id": str(record["group_id"]),
        "parent_id": str(record.get("parent_id", "")),
        "category": str(record["category"]),
        "template_id": str(record.get("template_id", "")),
        "split": split,
        # The Hub renders a struct of {bytes, path} as an Image feature.
        "image": {"bytes": payload, "path": image_file.name},
        "text_render": text,
        "text_norm": normalise_for_eval(text),
        "n_chars": len(text),
        "n_words": word_count(text),
        "n_lines": len(record.get("lines", ())),
        "digit_script": str(record.get("digit_script", "")),
        "digit_density": round(digit_density(text), 4),
        "has_zwnj": bool(record.get("has_zwnj", "\u200c" in text)),
        "has_diacritics": bool(record.get("has_diacritics", False)),
        "arabic_forms": bool(record.get("arabic_forms", False)),
        "genre": str(record.get("genre", "")),
        "register": str(record.get("register", "")),
        "persona": str(record.get("persona", "")),
        "length_target": str(record.get("length_target", "")),
        "lexicon_word_ids": list(record.get("lexicon_word_ids", ())),
        "domain_tags": list(record.get("domain_tags", ())),
        "rare_word_count": int(record.get("rare_word_count", 0) or 0),
        "font_families": families,
        "font_licences": [licences.get(name, "") for name in families],
        "dpi": int(record.get("dpi", 0) or 0),
        "width": int(record["width"]),
        "height": int(record["height"]),
        "image_format": str(record.get("image_format", "png")),
        "clean": bool(aug.get("clean", False)),
        "dark_mode": bool(record.get("dark_mode", False)),
        "contrast": float(record.get("qc", {}).get("contrast", 0.0) or 0.0),
        "llm_model": str(record.get("llm_model", "")),
        "profile": str(record.get("profile", "")),
        "temperature": float(record.get("temperature", 0.0) or 0.0),
        "prompt_hash": str(record.get("prompt_hash", "")),
        "seed": int(record.get("seed", 0) or 0),
        # Annotations travel as JSON text: the geometry is nested and variable
        # length, and a reader who wants it can parse one column rather than
        # having every consumer pay for a deep Arrow schema.
        "annotation": json.dumps(
            {
                "lines": record.get("lines", []),
                "blocks": record.get("blocks", []),
            },
            ensure_ascii=False,
        ),
        "aug": json.dumps(aug, ensure_ascii=False),
    }
    return row, len(payload)


def _font_licences(config: DiverseConfig) -> dict[str, str]:
    """Read family-to-licence from the font catalogue.

    Args:
        config: The active configuration.

    Returns:
        Family name to licence identifier.
    """
    import csv

    out: dict[str, str] = {}
    if not config.paths.fonts_csv.is_file():
        return out
    with config.paths.fonts_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out[row["family"].strip()] = row.get("licence", "").strip()
    return out


def package(config: DiverseConfig) -> dict[str, Any]:
    """Run stage 07 end to end.

    Args:
        config: The active configuration.

    Returns:
        The report, also written to the export root.

    Raises:
        RuntimeError: If pyarrow is unavailable.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to write shards: pip install 'fa-ocr-lora[hf]'"
        ) from error

    settings = config.packaging
    root = settings.output_dir
    root.mkdir(parents=True, exist_ok=True)
    licences = _font_licences(config)
    holdout_profile = config.splits.holdout_generator

    # Pass one: decide every group's split. A page and its crops must agree,
    # so the decision is made per group and then applied, never per sample.
    group_split: dict[str, str] = {}
    group_flags: dict[str, dict[str, bool]] = defaultdict(
        lambda: {"font": False, "template": False, "generator": False}
    )
    records: list[dict[str, Any]] = []
    for record in _iter_accepted(config.paths.accepted_jsonl):
        records.append(record)
        group = str(record["group_id"])
        aug = record.get("aug", {})
        flags = group_flags[group]
        flags["font"] |= bool(aug.get("holdout_font"))
        flags["template"] |= bool(aug.get("holdout_template"))
        flags["generator"] |= bool(
            holdout_profile and str(record.get("profile", "")) == holdout_profile
        )

    for group, flags in group_flags.items():
        group_split[group] = assign_split(
            group,
            ratios=config.splits.ratios,
            salt=config.splits.split_salt,
            holdout_font=flags["font"],
            holdout_template=flags["template"],
            holdout_generator=flags["generator"],
        )

    # Cap each hard test. Crops amplify - one held-out page carries about
    # twenty-five of them - so an unseen-generator profile weighted at 12%
    # produced a 21% test split. Groups past the ceiling are *dropped*, not
    # moved to train: their font, template or generator is meant to be unseen,
    # so training on them would silently void the test they exist for.
    group_size: Counter[str] = Counter(str(record["group_id"]) for record in records)
    ceiling = config.splits.max_holdout_samples
    dropped_groups: set[str] = set()
    for name in ("test_unseen_fonts", "test_unseen_templates", "test_unseen_generator"):
        members = sorted(
            (group for group, split in group_split.items() if split == name),
            # Deterministic: the same groups are kept on every run.
            key=lambda group: hashlib.blake2b(
                f"{name}:{group}".encode(), digest_size=8
            ).hexdigest(),
        )
        running = 0
        for group in members:
            if running >= ceiling:
                dropped_groups.add(group)
                continue
            running += group_size[group]
        if dropped_groups:
            LOGGER.info(
                "  %s capped at %d sample(s); %d group(s) dropped",
                name,
                ceiling,
                sum(1 for group in members if group in dropped_groups),
            )

    # Pass two: write the shards, one writer per (category, split).
    writers: dict[tuple[str, str], _ShardWriter] = {}
    manifest: list[dict[str, Any]] = []
    by_split: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    failures = 0

    dropped_samples = 0
    for record in records:
        group = str(record["group_id"])
        if group in dropped_groups:
            dropped_samples += 1
            continue
        split = group_split[group]
        category = str(record["category"])
        try:
            row, payload_bytes = _row_for(record, split, config, licences)
        except OSError as error:
            failures += 1
            LOGGER.warning("skipping %s: %s", record.get("sample_id"), error)
            continue
        key = (category, split)
        writer = writers.get(key)
        if writer is None:
            writer = _ShardWriter(
                directory=root / "data" / category,
                split=split,
                shard_samples=settings.shard_samples,
                max_bytes=settings.max_shard_bytes,
                compression=settings.compression,
            )
            writers[key] = writer
            LOGGER.info("  %s / %s", category, split)
        writer.add(row, payload_bytes)
        by_split[split] += 1
        by_category[category] += 1
        manifest.append(
            {
                column: row.get(column)
                for column in MANIFEST_COLUMNS
                if column != "shard_file"
            }
            | {"shard_file": f"data/{category}/{split}"}
        )

    shards: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for (category, split), writer in sorted(writers.items()):
        named = writer.finalise()
        shards[category][split] = [name for name, _ in named]

    # The manifest ships as its own Parquet at the root, so the corpus can be
    # filtered and counted without opening an image shard.
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.Table.from_pylist(manifest),
        root / "manifest.parquet",
        compression=settings.compression,
    )

    report = {
        "run_name": config.run_name,
        "samples": len(manifest),
        "failures": failures,
        "dropped_over_holdout_cap": dropped_samples,
        "max_holdout_samples": ceiling,
        "by_split": dict(by_split.most_common()),
        "by_category": {name: by_category.get(name, 0) for name in CATEGORIES},
        "shards": {category: dict(splits) for category, splits in shards.items()},
        "split_ratios_requested": dict(config.splits.ratios),
        "holdout_generator": holdout_profile,
        "holdout_templates": config.splits.holdout_templates,
        "output_dir": str(root),
    }
    write_json(root / "packaging_report.json", report)

    LOGGER.info("")
    LOGGER.info("=" * 66)
    LOGGER.info("PACKAGING SUMMARY")
    LOGGER.info("=" * 66)
    LOGGER.info("samples   %d", len(manifest))
    LOGGER.info("failures  %d", failures)
    LOGGER.info(
        "dropped   %d (past the hard-test ceiling of %d)", dropped_samples, ceiling
    )
    LOGGER.info("")
    LOGGER.info("By split:")
    for name in SPLITS:
        count = by_split.get(name, 0)
        share = count / max(1, len(manifest))
        LOGGER.info("  %-24s %6d  (%.1f%%)", name, count, share * 100)
    LOGGER.info("")
    LOGGER.info("By category: %s", report["by_category"])
    LOGGER.info("")
    LOGGER.info("Wrote %s", root / "manifest.parquet")

    # Validation must contain every category. The previous corpus had a
    # validation split with no persian_pixel records at all, which made its
    # validation curve describe a different distribution from its test score.
    missing = [
        name
        for name in CATEGORIES
        if by_category.get(name, 0) > 0
        and not any(
            record["category"] == name and record["split"] == "validation"
            for record in manifest
        )
    ]
    if missing:
        LOGGER.warning("")
        LOGGER.warning(
            "Validation holds no sample from: %s. At this corpus size that is "
            "expected; at full scale it would mean the validation curve "
            "describes a different distribution from the test score.",
            missing,
        )
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
    package(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
