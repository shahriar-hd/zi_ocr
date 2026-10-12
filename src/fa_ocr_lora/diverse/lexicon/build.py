"""Stage 00 - harvest, clean, tag and band the Persian lexicon.

Runs on the laptop. Produces ``lexicon.parquet``, which stages 01 and 02
consume, plus ``lexicon_summary.json`` for the dataset card and the coverage
report.

The merge rule
--------------

Three sources disagree, and the order in which evidence is accepted decides
what the lexicon says. The rules, in priority order:

1. **Lilak membership proves the word is Persian.** Persian Wiktionary
   documents other languages in Persian, so a page categorised
   ``فعل‌های انگلیسی`` describes an English verb. Those are dropped - unless
   Lilak, a Persian spell-checker, also lists the word, in which case it is a
   genuine Persian word that happens to be a loan.
2. **Wiktionary wins on era and domain.** It is the only source that carries
   them; Lilak has no era metadata and the titles list has no metadata at all.
3. **An explicit label beats an inferred one.** The titles list tags everything
   ``technical`` by inference, so it never overwrites a Wiktionary ``archaic``
   or ``neologism``.
4. **Lilak wins on part of speech** where Wiktionary's categories said nothing,
   because its POS column is a curated field rather than a category-name guess.

Usage::

    python -m fa_ocr_lora.diverse.lexicon.build            # full build
    python -m fa_ocr_lora.diverse.lexicon.build --max-documents 50000   # quick
    python -m fa_ocr_lora.diverse.lexicon.build --force    # ignore the cache
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from ...dataset.common.io_utils import write_json
from ..config import BANDS, ERAS, DiverseConfig, load_diverse_config
from . import frequency as freq_module
from .parsers import (
    Headword,
    parse_lilak,
    parse_wikipedia_titles,
    parse_wiktionary,
)
from .sources import download, open_maybe_compressed, stream_bz2_text

__all__ = ["build_lexicon", "main"]

LOGGER: Final[logging.Logger] = logging.getLogger("fa_diverse.lexicon")

#: Columns of ``lexicon.parquet``. ``in_pool`` is the column stage 01 filters
#: on; everything else is provenance or stratification metadata.
LEXICON_COLUMNS: Final[tuple[str, ...]] = (
    "word_id",
    "word",
    "pos",
    "era",
    "domain",
    "freq_band",
    "frequency",
    "rank",
    "in_pool",
    "sources",
    "categories",
    "licence_note",
)


class LexiconBuildError(RuntimeError):
    """Raised when stage 00 cannot produce a usable lexicon."""


def _merge(entry: Headword, incoming: Headword) -> None:
    """Fold a new observation of a word into the existing record.

    Args:
        entry: The accumulated record, mutated in place.
        incoming: A fresh observation from another source.
    """
    for name in incoming.sources:
        entry.add_source(name)

    # Wiktionary is the only source of era and domain, so an explicit label
    # from it always wins over the titles list's blanket `technical` guess.
    explicit = incoming.era not in {"unlabelled", "technical", "foreign"}
    if explicit or entry.era in {"unlabelled", "foreign"}:
        if incoming.era != "foreign":
            entry.era = incoming.era
    if incoming.domain and not entry.domain:
        entry.domain = incoming.domain
    if incoming.pos != "other" and entry.pos == "other":
        entry.pos = incoming.pos
    if incoming.categories and not entry.categories:
        entry.categories = incoming.categories


def harvest(config: DiverseConfig, *, force: bool = False) -> dict[str, Headword]:
    """Fetch and parse every configured source into one merged mapping.

    Args:
        config: The active configuration.
        force: Re-download sources already in the cache.

    Returns:
        Normalised word to its merged record. Foreign-language Wiktionary
        entries that no other source confirms are already removed.

    Raises:
        LexiconBuildError: If a source kind has no parser, or if the merge
            produced too few words to be usable.
    """
    cache = config.paths.cache_dir
    minimum = config.lexicon.length.minimum
    maximum = config.lexicon.length.maximum
    merged: dict[str, Headword] = {}
    #: Words seen only as a foreign-language Wiktionary entry. Resolved after
    #: every source has been read, because Lilak may be parsed after it.
    foreign_only: set[str] = set()
    per_source: Counter[str] = Counter()

    for source in config.lexicon.sources:
        LOGGER.info("")
        LOGGER.info("--- %s (%s, %s) ---", source.name, source.kind, source.licence)

        if source.kind == "lilak_lexicon":
            path = download(source.url, cache, force=force)
            stream = parse_lilak(path, minimum=minimum, maximum=maximum)
        elif source.kind == "wiktionary_dump":
            # Materialised rather than streamed: the dump is only ~18 MB and
            # the parse is re-run whenever a tagging rule changes, so paying
            # the download once makes every later run offline.
            download(source.url, cache, force=force)
            chunks = stream_bz2_text(source.url, cache, use_cache=True)
            stream = parse_wiktionary(chunks, minimum=minimum, maximum=maximum)
        elif source.kind == "wikipedia_titles":
            path = download(source.url, cache, force=force)
            stream = parse_wikipedia_titles(
                open_maybe_compressed(path), minimum=minimum, maximum=maximum
            )
        else:
            raise LexiconBuildError(f"{source.name}: no parser for kind {source.kind!r}")

        for incoming in stream:
            per_source[source.name] += 1
            if incoming.era == "foreign":
                foreign_only.add(incoming.word)
            existing = merged.get(incoming.word)
            if existing is None:
                merged[incoming.word] = incoming
            else:
                _merge(existing, incoming)

    # A word is Persian if any source other than a foreign-tagged Wiktionary
    # page vouched for it.
    dropped = 0
    for word in sorted(foreign_only):
        entry = merged.get(word)
        if entry is None:
            continue
        vouched = [name for name in entry.sources if name != "fa_wiktionary"]
        if not vouched:
            del merged[word]
            dropped += 1
        elif entry.era == "foreign":
            entry.era = "unlabelled"

    LOGGER.info("")
    LOGGER.info("Per-source accepted rows: %s", dict(per_source))
    LOGGER.info(
        "Merged %d distinct word(s); dropped %d unconfirmed foreign entry/entries",
        len(merged),
        dropped,
    )
    if len(merged) < 1000:
        raise LexiconBuildError(
            f"only {len(merged)} word(s) survived the merge; expected tens of thousands. "
            "Check the source URLs and the length limits before continuing."
        )
    return merged


def band_and_rank(
    config: DiverseConfig, words: dict[str, Headword], *, force: bool = False
) -> None:
    """Fill ``frequency``, ``rank`` and ``freq_band`` on every record.

    Args:
        config: The active configuration.
        words: Merged records, mutated in place.
        force: Re-download the frequency corpus.
    """
    LOGGER.info("")
    LOGGER.info("--- frequency corpus ---")
    chunks = stream_bz2_text(
        config.lexicon.frequency_url, config.paths.cache_dir, use_cache=not force
    )
    table = freq_module.count_corpus(chunks, max_documents=config.lexicon.max_documents)
    bands = freq_module.assign_bands(
        sorted(words), table, config.lexicon.band_percentiles
    )
    for word, entry in words.items():
        entry.frequency = table.frequency_of(word)
        entry.rank = table.rank_of(word)
        entry.freq_band = bands[word]


def apply_pool_filter(
    config: DiverseConfig, words: dict[str, Headword]
) -> tuple[int, int]:
    """Decide which words may be injected into prompts.

    This is the stage that acts on the observation that an LLM writes
    high-frequency Persian unprompted. A word inside the exclusion rank stays
    in the parquet - its frequency is what proves it is common - but is never
    spent on a prompt slot.

    Args:
        config: The active configuration.
        words: Merged records with ranks already filled.

    Returns:
        ``(in_pool, excluded)`` counts.
    """
    limit = config.lexicon.exclude_top_frequency_rank
    in_pool = 0
    excluded = 0
    for entry in words.values():
        common = 0 < entry.rank <= limit
        if common:
            excluded += 1
        else:
            in_pool += 1
    LOGGER.info(
        "Pool filter: %d word(s) injectable, %d excluded as too common (rank <= %d)",
        in_pool,
        excluded,
        limit,
    )
    return in_pool, excluded


def _rows(config: DiverseConfig, words: dict[str, Headword]) -> list[dict[str, object]]:
    """Build the parquet rows.

    Args:
        config: The active configuration, for the per-source licence notes.
        words: Merged records, fully filled.

    Returns:
        One row per word, sorted by word so the file is byte-stable across
        runs with the same inputs.
    """
    notes = {source.name: source.licence_note for source in config.lexicon.sources}
    limit = config.lexicon.exclude_top_frequency_rank
    rows: list[dict[str, object]] = []
    for index, word in enumerate(sorted(words)):
        entry = words[word]
        rows.append(
            {
                "word_id": f"w{index:07d}",
                "word": entry.word,
                "pos": entry.pos,
                "era": entry.era,
                "domain": entry.domain,
                "freq_band": entry.freq_band,
                "frequency": int(entry.frequency),
                "rank": int(entry.rank),
                "in_pool": not (0 < entry.rank <= limit),
                "sources": ";".join(entry.sources),
                "categories": ";".join(entry.categories),
                "licence_note": " ".join(
                    notes.get(name, "") for name in entry.sources
                ).strip(),
            }
        )
    return rows


def _write_parquet(rows: Sequence[dict[str, object]], target: Path) -> None:
    """Write the lexicon to parquet.

    Args:
        rows: Rows from :func:`_rows`.
        target: Destination file.

    Raises:
        LexiconBuildError: If pyarrow is unavailable.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise LexiconBuildError(
            "pyarrow is required to write lexicon.parquet: pip install 'fa-ocr-lora[hf]'"
        ) from error

    schema = pa.schema(
        [
            ("word_id", pa.string()),
            ("word", pa.string()),
            ("pos", pa.string()),
            ("era", pa.string()),
            ("domain", pa.string()),
            ("freq_band", pa.string()),
            ("frequency", pa.int64()),
            ("rank", pa.int64()),
            ("in_pool", pa.bool_()),
            ("sources", pa.string()),
            ("categories", pa.string()),
            ("licence_note", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(list(rows), schema=schema)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target, compression="zstd")


def _summary(
    config: DiverseConfig, rows: Sequence[dict[str, object]]
) -> dict[str, object]:
    """Build the summary recorded to disk and printed.

    Args:
        config: The active configuration.
        rows: The parquet rows.

    Returns:
        A JSON-serialisable summary.
    """
    pool = [row for row in rows if row["in_pool"]]
    return {
        "run_name": config.run_name,
        "total_words": len(rows),
        "injectable_words": len(pool),
        "excluded_as_common": len(rows) - len(pool),
        "exclude_top_frequency_rank": config.lexicon.exclude_top_frequency_rank,
        "by_band": {
            name: sum(1 for row in rows if row["freq_band"] == name) for name in BANDS
        },
        "by_band_in_pool": {
            name: sum(1 for row in pool if row["freq_band"] == name) for name in BANDS
        },
        "by_era": {name: sum(1 for row in rows if row["era"] == name) for name in ERAS},
        "by_era_in_pool": {
            name: sum(1 for row in pool if row["era"] == name) for name in ERAS
        },
        "by_pos": dict(Counter(str(row["pos"]) for row in rows).most_common()),
        "by_domain": dict(
            Counter(str(row["domain"]) for row in rows if row["domain"]).most_common(40)
        ),
        "by_source": dict(
            Counter(
                name for row in rows for name in str(row["sources"]).split(";") if name
            ).most_common()
        ),
        "sources": [
            {
                "name": source.name,
                "kind": source.kind,
                "url": source.url,
                "licence": source.licence,
                "licence_note": source.licence_note,
                "attribution": source.attribution,
            }
            for source in config.lexicon.sources
        ],
        "frequency_corpus": {
            "url": config.lexicon.frequency_url,
            "licence": config.lexicon.frequency_licence,
            "max_documents": config.lexicon.max_documents,
        },
    }


def _report(summary: dict[str, object]) -> None:
    """Print the stage 00 report.

    Args:
        summary: Output of :func:`_summary`.
    """
    LOGGER.info("")
    LOGGER.info("=" * 68)
    LOGGER.info("LEXICON SUMMARY")
    LOGGER.info("=" * 68)
    LOGGER.info("Total words        %d", summary["total_words"])
    LOGGER.info(
        "Injectable         %d  (excluded as too common: %d, rank <= %d)",
        summary["injectable_words"],
        summary["excluded_as_common"],
        summary["exclude_top_frequency_rank"],
    )
    LOGGER.info("")
    LOGGER.info("  %-12s %10s %10s", "band", "total", "in pool")
    for name in BANDS:
        LOGGER.info(
            "  %-12s %10d %10d",
            name,
            summary["by_band"][name],  # type: ignore[index]
            summary["by_band_in_pool"][name],  # type: ignore[index]
        )
    LOGGER.info("")
    LOGGER.info("  %-12s %10s %10s", "era", "total", "in pool")
    for name in ERAS:
        LOGGER.info(
            "  %-12s %10d %10d",
            name,
            summary["by_era"][name],  # type: ignore[index]
            summary["by_era_in_pool"][name],  # type: ignore[index]
        )
    LOGGER.info("")
    LOGGER.info("Part of speech     %s", summary["by_pos"])
    LOGGER.info("Sources            %s", summary["by_source"])
    domains = summary["by_domain"]
    LOGGER.info("Domains (top 12)   %s", dict(list(domains.items())[:12]))  # type: ignore[union-attr]


def build_lexicon(config: DiverseConfig, *, force: bool = False) -> dict[str, object]:
    """Run stage 00 end to end.

    Args:
        config: The active configuration.
        force: Re-download every source, ignoring the cache.

    Returns:
        The summary, also written to
        :attr:`~fa_ocr_lora.diverse.config.PathsConfig.lexicon_report`.

    Raises:
        LexiconBuildError: If the lexicon would be unusably small, or the era
            stratification the plan stage needs cannot be satisfied.
    """
    words = harvest(config, force=force)
    band_and_rank(config, words, force=force)
    apply_pool_filter(config, words)

    rows = _rows(config, words)
    _write_parquet(rows, config.paths.lexicon_parquet)
    summary = _summary(config, rows)
    write_json(config.paths.lexicon_report, summary)
    _report(summary)

    LOGGER.info("")
    LOGGER.info("Wrote %s", config.paths.lexicon_parquet)
    LOGGER.info("Wrote %s", config.paths.lexicon_report)

    # The plan sampler draws an era then a word. An era with a non-zero weight
    # and no injectable words would make the sampler loop or silently
    # substitute, so the mismatch is reported here where it can be fixed.
    pool_by_era = summary["by_era_in_pool"]
    starved = [
        name
        for name, weight in config.plan.era_weights.items()
        if weight > 0 and pool_by_era.get(name, 0) < 50  # type: ignore[union-attr]
    ]
    if starved:
        LOGGER.warning("")
        LOGGER.warning(
            "Era(s) with a non-zero plan weight but fewer than 50 injectable words: %s. "
            "The sampler will fall back to the nearest populated era; either lower "
            "those weights in plan.era_weights or widen the tagging rules.",
            starved,
        )
    return summary


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
        "--max-documents",
        type=int,
        default=None,
        help="override lexicon.frequency.max_documents (useful for a quick dry run)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download every source, ignoring the cache",
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
    if args.max_documents is not None:
        from dataclasses import replace

        config = replace(
            config,
            lexicon=replace(config.lexicon, max_documents=args.max_documents),
        )
    try:
        build_lexicon(config, force=args.force)
    except LexiconBuildError as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
