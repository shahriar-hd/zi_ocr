"""Stage 04 CLI - choose a format, template, fonts, colours and degradation.

Runs anywhere; needs the font pool and the cleaned text, not the API keys. In
the split pipeline it runs on the molab runtime, immediately before rendering,
because the specs it writes embed the fonts and are therefore large.

Every decision is drawn from the sample's own seeded RNG, so a layout is
reproducible from (config, seed, sample) alone and two runs over the same text
produce the same corpus.

The one non-obvious choice here is **template holdout**. ``test_unseen_templates``
needs samples whose template never appeared in training, so the holdout cannot
be decided at split time - by then the sample is already rendered. It is
decided here: a fixed number of templates per category are marked held-out, by
hashing their id so the choice is stable, and any sample that drew one is
flagged. Stage 07 then routes flagged samples into the hard test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

from ...dataset.common.io_utils import sample_rng, weighted_choice, write_json
from ..config import CATEGORIES, DiverseConfig, load_diverse_config
from .fonts import FontPool, FontSelectionError, choose_font, load_font_pool
from .spec import RenderSpec, build_palette
from .templates import TEMPLATE_IDS, build_document

__all__ = ["build_layouts", "holdout_templates", "main"]

LOGGER: Final[logging.Logger] = logging.getLogger("fa_diverse.layout")

#: Style-class preferences per category. A poetry page should lean naskh and
#: nastaliq; a banner should lean display; a chat screenshot is always sans,
#: because that is what phones render. Weights are relative.
_STYLE_BY_CATEGORY: Final[dict[str, dict[str, float]]] = {
    "page": {
        "sans": 0.38,
        "naskh": 0.24,
        "serif": 0.28,
        "display": 0.08,
        "nastaliq": 0.02,
    },
    "paragraph": {
        "sans": 0.42,
        "naskh": 0.20,
        "serif": 0.26,
        "display": 0.10,
        "nastaliq": 0.02,
    },
    "line": {
        "sans": 0.40,
        "display": 0.32,
        "serif": 0.14,
        "naskh": 0.12,
        "nastaliq": 0.02,
    },
    "poetry": {
        "naskh": 0.34,
        "nastaliq": 0.26,
        "serif": 0.26,
        "display": 0.10,
        "sans": 0.04,
    },
    "web": {"sans": 0.78, "serif": 0.14, "display": 0.08},
    "chat": {"sans": 0.94, "display": 0.06},
    "banner": {"display": 0.68, "sans": 0.22, "naskh": 0.06, "serif": 0.04},
    "story": {
        "serif": 0.38,
        "naskh": 0.30,
        "sans": 0.22,
        "display": 0.08,
        "nastaliq": 0.02,
    },
    "table_form": {"sans": 0.72, "serif": 0.16, "naskh": 0.12},
    "mixed": {"sans": 0.82, "serif": 0.12, "display": 0.06},
}

#: Categories rendered in a dark theme some of the time. Only screens have
#: dark modes; paper does not.
_DARK_PROBABILITY: Final[dict[str, float]] = {"chat": 0.38, "web": 0.14, "line": 0.10}

#: Stage 03 fields the manifest needs. Copied verbatim into the spec rather
#: than re-derived, because stage 05 writes the manifest's source record and
#: has no other access to the text's provenance.
_META_KEYS: Final[tuple[str, ...]] = (
    "genre",
    "register",
    "persona",
    "length_target",
    "output_form",
    "features",
    "digit_script",
    "digit_density",
    "has_zwnj",
    "has_diacritics",
    "arabic_forms",
    "lexicon_word_ids",
    "lexicon_words",
    "domain_tags",
    "word_eras",
    "word_bands",
    "rare_word_count",
    "lexicon_hit_rate",
    "llm_model",
    "profile",
    "temperature",
    "prompt_hash",
    "persian_ratio",
    "index",
)

#: Share of samples rendered with the held-out font families. Small, because
#: these samples leave the train split entirely - they exist only so
#: ``test_unseen_fonts`` has something to measure. Too large and the corpus
#: loses usable training data to a test set nobody needs thousands of.
_HOLDOUT_FONT_FRACTION: Final[float] = 0.03


def holdout_templates(config: DiverseConfig) -> frozenset[str]:
    """Choose the templates reserved for ``test_unseen_templates``.

    Decided by hashing the template id rather than by shuffling, so the same
    templates are held out on every machine and in every run - a holdout that
    moved between runs would make the hard test incomparable with itself.

    The count is a total, not per category. One template per category is 10 of
    41, which routed 22.8% of the corpus into a hard test and took that much
    layout variety out of training; four templates is about 3%.

    Args:
        config: The active configuration.

    Returns:
        The reserved template ids.
    """
    wanted = config.splits.holdout_templates
    if wanted <= 0:
        return frozenset()

    # Every template except the last of any category: a category whose only
    # template were held out would leave train nothing to render.
    eligible: list[str] = []
    for ids in TEMPLATE_IDS.values():
        if len(ids) > 1:
            eligible.extend(ids)

    ranked = sorted(
        eligible,
        key=lambda name: hashlib.blake2b(
            f"{config.splits.split_salt}:{name}".encode(), digest_size=8
        ).hexdigest(),
    )
    owner = {name: category for category, ids in TEMPLATE_IDS.items() for name in ids}
    taken: Counter[str] = Counter()
    reserved: list[str] = []
    for name in ranked:
        if len(reserved) >= wanted:
            break
        category = owner[name]
        # Never take every template a category has, however the hashes fall.
        if taken[category] + 1 >= len(TEMPLATE_IDS[category]):
            continue
        reserved.append(name)
        taken[category] += 1
    return frozenset(reserved)


def _iter_clean(path: Path) -> Iterator[dict[str, Any]]:
    """Stream the stage 03 output.

    Args:
        path: ``texts_clean.jsonl``.

    Yields:
        One record per line.

    Raises:
        FileNotFoundError: If stage 03 has not run.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `python -m fa_ocr_lora.diverse.textqc.run` first"
        )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_spec(
    record: dict[str, Any],
    config: DiverseConfig,
    pool: FontPool,
    reserved: frozenset[str],
) -> RenderSpec:
    """Build the render spec for one cleaned text.

    Args:
        record: A stage 03 record.
        config: The active configuration.
        pool: The font pool.
        reserved: Template ids held out for the unseen-template test.

    Returns:
        The spec.

    Raises:
        FontSelectionError: If no font can draw this sample's text, in which
            case the caller should reject the sample rather than render it
            with a substituted face.
    """
    sample_id = str(record["sample_id"])
    category = str(record["category"])
    text = str(record["text_render"])
    index = int(record["index"])
    rng = sample_rng(config.seed, "layout", index)

    template_id = rng.choice(list(TEMPLATE_IDS[category]))
    is_holdout_template = template_id in reserved

    dpi = weighted_choice(tuple(config.render.dpi_choices.items()), rng)
    dark = rng.random() < _DARK_PROBABILITY.get(category, 0.0)
    palette = build_palette(
        rng,
        paper_tints=config.degradation.paper.tints,
        display=category == "banner",
        dark_mode=dark,
    )

    # The font must cover this sample's exact text. A sample whose text no
    # font can draw is rejected upstream of the browser, because Chromium
    # would substitute silently and the label would stop matching the pixels.
    #
    # A small fraction renders with the held-out families, which is what gives
    # test_unseen_fonts anything to measure: the split cannot invent samples
    # in a font nothing rendered. Those samples are flagged and stage 07
    # routes them out of train.
    style_weights = _STYLE_BY_CATEGORY.get(category)
    use_holdout_fonts = (
        bool(pool.holdout_families) and rng.random() < _HOLDOUT_FONT_FRACTION
    )
    body = choose_font(
        pool, text, rng, style_weights=style_weights, holdout_only=use_holdout_fonts
    )
    display = choose_font(
        pool,
        text,
        rng,
        style_weights={"display": 0.55, "serif": 0.2, "naskh": 0.15, "sans": 0.10},
        holdout_only=use_holdout_fonts,
        want_bold=True,
    )
    faces = [body] if body.font_id == display.font_id else [body, display]

    html, width, height = build_document(
        template_id,
        text,
        palette,
        font_css=pool.face_css(faces),
        family=body.family,
        display_family=display.family,
        rng=rng,
        digit_script=str(record.get("digit_script", "persian")),
    )

    clean = rng.random() < config.render.clean_fraction
    aug: dict[str, Any] = {
        "planned_clean": clean,
        "holdout_template": is_holdout_template,
        "holdout_font": use_holdout_fonts,
    }

    return RenderSpec(
        sample_id=sample_id,
        group_id=str(record.get("group_id", sample_id)),
        category=category,
        template_id=template_id,
        html=html,
        width=width,
        height=height,
        dpi=int(dpi),
        font_ids=tuple(face.font_id for face in faces),
        font_families=tuple(face.family for face in faces),
        palette=palette,
        clean=clean,
        dark_mode=dark,
        aug=aug,
        text_render=text,
        meta={key: record[key] for key in _META_KEYS if key in record},
        seed=config.seed,
    )


def build_layouts(config: DiverseConfig, *, limit: int | None = None) -> dict[str, Any]:
    """Run stage 04 end to end.

    Args:
        config: The active configuration.
        limit: Process at most this many records.

    Returns:
        The report, also written beside the output.
    """
    pool = load_font_pool(config.paths.fonts_csv, holdout=config.render.fonts.holdout)
    reserved = holdout_templates(config)
    LOGGER.info(
        "%d template(s) held out for test_unseen_templates: %s",
        len(reserved),
        ", ".join(sorted(reserved)) or "none",
    )

    target = config.paths.layouts_jsonl
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")

    by_category: Counter[str] = Counter()
    by_template: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_dpi: Counter[int] = Counter()
    rejected: list[dict[str, str]] = []
    written = 0
    clean_count = 0
    dark_count = 0

    with temporary.open("w", encoding="utf-8") as handle:
        for record in _iter_clean(config.paths.texts_clean_jsonl):
            if limit is not None and written >= limit:
                break
            try:
                spec = build_spec(record, config, pool, reserved)
            except FontSelectionError as error:
                rejected.append(
                    {"sample_id": str(record.get("sample_id", "?")), "reason": str(error)}
                )
                continue
            except KeyError as error:
                rejected.append(
                    {
                        "sample_id": str(record.get("sample_id", "?")),
                        "reason": f"unknown category or template: {error}",
                    }
                )
                continue
            handle.write(json.dumps(spec.as_dict(), ensure_ascii=False) + "\n")
            written += 1
            by_category[spec.category] += 1
            by_template[spec.template_id] += 1
            for family in spec.font_families:
                by_family[family] += 1
            by_dpi[spec.dpi] += 1
            clean_count += int(spec.clean)
            dark_count += int(spec.dark_mode)
    temporary.replace(target)

    report = {
        "run_name": config.run_name,
        "written": written,
        "rejected": len(rejected),
        "rejections": rejected[:50],
        "holdout_templates": sorted(reserved),
        "holdout_fonts": sorted(pool.holdout_families),
        "by_category": {name: by_category.get(name, 0) for name in CATEGORIES},
        "by_template": dict(by_template.most_common()),
        "by_font_family": dict(by_family.most_common()),
        "by_dpi": {str(key): value for key, value in sorted(by_dpi.items())},
        "clean_fraction": round(clean_count / max(1, written), 4),
        "dark_fraction": round(dark_count / max(1, written), 4),
    }
    write_json(
        config.paths.work_dir / config.paths.LAYOUTS / "layout_report.json", report
    )

    LOGGER.info("")
    LOGGER.info("=" * 68)
    LOGGER.info("LAYOUT SUMMARY")
    LOGGER.info("=" * 68)
    LOGGER.info("written    %d", written)
    LOGGER.info("rejected   %d (no font covered the text)", len(rejected))
    LOGGER.info(
        "clean      %.1f%% (target %.0f%%)",
        report["clean_fraction"] * 100,
        config.render.clean_fraction * 100,
    )
    LOGGER.info("dark theme %.1f%%", report["dark_fraction"] * 100)
    LOGGER.info("")
    LOGGER.info("By category:  %s", report["by_category"])
    LOGGER.info("Templates:    %d distinct", len(by_template))
    LOGGER.info("Font families: %d distinct", len(by_family))
    LOGGER.info("DPI:          %s", report["by_dpi"])
    LOGGER.info("")
    LOGGER.info("Wrote %s", target)
    if rejected:
        LOGGER.warning("%d sample(s) had no covering font; first few:", len(rejected))
        for item in rejected[:5]:
            LOGGER.warning("  %s: %s", item["sample_id"], item["reason"][:100])
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
        "--limit", type=int, default=None, help="process at most N records"
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
    build_layouts(config, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
