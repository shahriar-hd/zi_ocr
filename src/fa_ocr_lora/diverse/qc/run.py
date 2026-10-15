"""Stage 06 - image QC and the review contact sheet.

Every check here answers one question: *would this sample teach the model
something false?* A blurry page is fine - real scans are blurry. A page whose
label claims a line the pixels do not show is not, and no amount of volume
compensates for it.

The checks, and what each prevents:

``overflow``
    Content escaped the canvas, so the label describes clipped text.
``annotation_text``
    The lines, concatenated, disagree with ``text_render``. This catches a
    harvest that dropped, duplicated or reordered a line - the one bug class
    that silently corrupts every label on a page.
``legibility``
    The shortest line is below an x-height floor. Text a human cannot read is
    a label a human cannot verify.
``contrast``
    Michelson contrast collapsed under degradation.
``blank``
    The image is near-uniform: the render produced nothing.
``duplicate``
    A perceptual hash collides with an accepted sample. Two identical images
    with different labels is worse than one fewer sample.
``digit_script``
    The rendered text carries a digit script the manifest does not claim.

A contact sheet is written per category so the pilot can be reviewed by eye,
because no automatic check catches "this looks wrong in a way I did not think
to test for".
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

from ...dataset.common.io_utils import write_json
from ..config import CATEGORIES, DiverseConfig, load_diverse_config
from ..render.schema import Annotation
from ..text_policy import detect_digit_scripts, normalise_for_eval

__all__ = ["REJECTION_REASONS", "check_sample", "run_qc", "main"]

LOGGER: Final[logging.Logger] = logging.getLogger("fa_diverse.imageqc")

#: Every rejection reason, for the report's columns.
REJECTION_REASONS: Final[tuple[str, ...]] = (
    "unreadable",
    "no_lines",
    "overflow",
    "annotation_text",
    "legibility",
    "contrast",
    "blank",
    "duplicate",
    "digit_script",
)

#: Side length of the thumbnail used for perceptual hashing. 8x8 after a
#: downscale is the classic average hash; it is scale- and format-invariant,
#: which is what is wanted - two samples that differ only in JPEG quality are
#: duplicates.
_HASH_SIDE: Final[int] = 8


def _iter_annotations(directory: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Stream every annotation written by stage 05.

    Args:
        directory: The annotation tree.

    Yields:
        ``(path, record)`` pairs.
    """
    for path in sorted(directory.rglob("*.json")):
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            LOGGER.warning("unreadable annotation %s: %s", path.name, error)


#: Characters that are layout instructions rather than content: table pipes,
#: separator rules and list bullets. Templates turn them into borders and
#: markers, so they are removed from both sides of the label comparison.
_LAYOUT_MARKUP: Final[re.Pattern[str]] = re.compile(
    r"[|\u2022*]+|-{2,}|^[-*\u2022]\s+", re.MULTILINE
)


def _strip_layout_markup(text: str) -> str:
    """Normalise text and drop pure layout markup.

    Args:
        text: Either the label or the harvested page text.

    Returns:
        The normalised text with table pipes, separator rules and list bullets
        removed.
    """
    return normalise_for_eval(_LAYOUT_MARKUP.sub(" ", text))


def _average_hash(image: Any) -> int:
    """Return the 64-bit average hash of an image.

    Args:
        image: A Pillow image.

    Returns:
        The hash as an integer.
    """
    from PIL import Image

    small = image.convert("L").resize((_HASH_SIDE, _HASH_SIDE), Image.Resampling.BILINEAR)
    pixels = list(small.tobytes())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for index, value in enumerate(pixels):
        if value >= mean:
            bits |= 1 << index
    return bits


def _michelson_contrast(image: Any, annotation: Annotation | None = None) -> float:
    """Return the contrast between the text and its local background.

    Measured **inside the line boxes**, not over the whole image. Global
    percentiles were the first attempt and they measure ink *coverage* rather
    than legibility: on a page that is 98% paper the 2nd percentile is still
    paper, so a perfectly crisp sparse page reported near-zero contrast and 39
    sound samples were rejected. Cropping to the lines puts ink and its
    immediate background in the same sample, which is what a reader's eye
    actually compares.

    Args:
        image: A Pillow image.
        annotation: The sample's annotation. When ``None`` or empty, falls
            back to the whole image, which is correct for a blank-image check.

    Returns:
        A value in ``[0, 1]``.
    """
    from PIL import Image

    from ..render.schema import bounds

    pixels: list[int] = []
    if annotation is not None and annotation.lines:
        # A handful of lines is enough and keeps this cheap on a 200-line page.
        for line in annotation.ordered_lines()[:12]:
            x0, y0, x1, y1 = bounds(line.polygon)
            box = (
                max(0, int(x0)),
                max(0, int(y0)),
                min(image.width, int(x1) + 1),
                min(image.height, int(y1) + 1),
            )
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            patch = image.crop(box).convert("L")
            if patch.width * patch.height > 4096:
                patch = patch.resize((64, 64), Image.Resampling.BILINEAR)
            pixels.extend(patch.tobytes())

    if not pixels:
        small = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
        pixels = list(small.tobytes())
    if not pixels:
        return 0.0

    pixels.sort()
    # Stroke darkness against local background, not a symmetric percentile
    # pair. Inside a line box at 11px type the strokes are under 5% of the
    # pixels, so a p5/p95 span measures background against background and
    # reported 0.06 for perfectly crisp text. The darkest 2% are the ink; the
    # 70th percentile is the paper around it.
    low = pixels[int(len(pixels) * 0.02)]
    high = pixels[int(len(pixels) * 0.70)]
    if high + low == 0:
        return 0.0
    return (high - low) / (high + low)


def check_sample(
    record: dict[str, Any],
    config: DiverseConfig,
    *,
    seen_hashes: dict[str, dict[int, str]],
) -> tuple[str | None, dict[str, Any]]:
    """Run every check on one rendered sample.

    Args:
        record: An annotation record from stage 05.
        config: The active configuration.
        seen_hashes: Accepted image hashes, bucketed by normalised text and
            mutated with this sample's.

    Returns:
        ``(reason, metrics)``; ``reason`` is ``None`` when the sample passes.
    """
    settings = config.image_qc
    metrics: dict[str, Any] = {}

    image_path = config.paths.work_dir / str(record.get("image_path", ""))
    try:
        from PIL import Image

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            digest = _average_hash(image)
    except Exception as error:  # noqa: BLE001 - an unopenable image is a rejection
        return f"unreadable: {error}", metrics

    metrics["width"] = width
    metrics["height"] = height

    try:
        annotation = Annotation.from_dict(record)
    except Exception as error:  # noqa: BLE001
        return f"unreadable: annotation invalid ({error})", metrics

    contrast = _michelson_contrast(image, annotation)
    metrics["contrast"] = round(contrast, 4)

    metrics["n_lines"] = annotation.n_lines
    metrics["n_words"] = annotation.n_words
    metrics["min_line_height"] = round(annotation.min_line_height(), 2)

    if annotation.n_lines == 0:
        return "no_lines: the harvest found no text", metrics

    if record.get("aug", {}).get("overflowed"):
        return "overflow: content escaped the canvas", metrics

    # The label must equal what the boxes say. Compared on normalised,
    # whitespace-collapsed text, because the harvest reports a line per visual
    # line while text_render carries the author's paragraph breaks.
    # Layout markup is stripped from both sides before comparing. A pipe in
    # the text is a column boundary that a table template draws as a border,
    # and a leading bullet becomes a list marker - neither is content the
    # model should be asked to read, and neither survives into the pixels.
    expected = _strip_layout_markup(str(record.get("text_render", "")))
    harvested = _strip_layout_markup(annotation.text_from_lines())
    expected_tokens = expected.split()
    harvested_tokens = harvested.split()
    if expected_tokens and harvested_tokens:
        missing = set(expected_tokens) - set(harvested_tokens)
        coverage = 1.0 - len(missing) / len(set(expected_tokens))
        metrics["text_coverage"] = round(coverage, 4)
        # A template may legitimately add chrome (a timestamp, a page number),
        # so the harvest can be a superset. It must not be a subset: a missing
        # token means the label claims text no box covers.
        if coverage < 0.995:
            sample = ", ".join(sorted(missing)[:6])
            return (
                f"annotation_text: {len(missing)} label token(s) have no box "
                f"(coverage {coverage:.3f}): {sample}"
            ), metrics

    low_res = bool(record.get("aug", {}).get("rescale"))
    floor = settings.min_x_height_px_low_res if low_res else settings.min_x_height_px
    if annotation.min_line_height() < floor:
        return (
            f"legibility: shortest line is {annotation.min_line_height():.1f}px, "
            f"below {floor:.1f}"
        ), metrics

    if contrast < settings.min_contrast:
        return f"contrast: {contrast:.3f} below {settings.min_contrast:.2f}", metrics

    if contrast < 0.02:
        return "blank: image is near-uniform", metrics

    # Perceptual duplicate, but only among samples whose *text* is identical.
    #
    # An 8x8 average hash cannot tell two lines of Persian apart - at that
    # resolution every short line of dark-on-light text looks the same - so
    # comparing images alone flagged 515 sound samples, mostly sibling line
    # crops from one page with entirely different words. What "duplicate"
    # should mean here is the same content rendered twice, and the text is
    # already known, so it is the primary key and the image hash only
    # separates genuinely different renders of identical text.
    text_key = normalise_for_eval(str(record.get("text_render", "")))
    bucket = seen_hashes.setdefault(text_key, {})
    for other, other_id in bucket.items():
        if bin(digest ^ other).count("1") <= settings.phash_max_distance:
            return (
                f"duplicate: same text and perceptually identical to {other_id}"
            ), metrics

    claimed = str(record.get("digit_script", "")) or None
    if claimed and claimed != "mixed":
        present = detect_digit_scripts(str(record.get("text_render", "")))
        if present and claimed not in present:
            return (
                f"digit_script: manifest claims {claimed} but the text carries "
                f"{sorted(present)}"
            ), metrics

    bucket[digest] = str(record.get("sample_id", "?"))
    return None, metrics


def _contact_sheet(
    accepted: Sequence[dict[str, Any]], config: DiverseConfig
) -> list[Path]:
    """Write one contact sheet per category.

    Args:
        accepted: Accepted records.
        config: The active configuration.

    Returns:
        The sheet paths written.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        LOGGER.warning("Pillow unavailable; skipping contact sheets")
        return []

    import random

    # The sheet exists to be read by a human, and Pillow's default bitmap font
    # has no Persian glyphs - the snippets came out as rows of tofu boxes,
    # which defeats the point of a review sheet. Any face from the pool will
    # do; Pillow does not shape Persian correctly either, but unshaped Persian
    # is still recognisable where boxes are not.
    caption_font = None
    try:
        from ..layouts.fonts import load_font_pool

        pool = load_font_pool(config.paths.fonts_csv)
        preferred = [
            face
            for face in pool.faces
            if face.style_class == "sans" and face.weight in range(400, 600)
        ]
        face = (preferred or list(pool.faces))[0]
        caption_font = ImageFont.truetype(str(face.path), 13)
    except Exception as error:  # noqa: BLE001 - a plain sheet beats no sheet
        LOGGER.debug("contact sheet falling back to the default font: %s", error)

    thumb = config.image_qc.contact_sheet_thumbnail_px
    per_category = config.image_qc.contact_sheet_per_category
    directory = config.paths.reports_dir / "contact_sheets"
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    by_category: dict[str, list[dict[str, Any]]] = {}
    for record in accepted:
        by_category.setdefault(str(record["category"]), []).append(record)

    for category, records in sorted(by_category.items()):
        rng = random.Random(f"{config.seed}:sheet:{category}")
        chosen = rng.sample(records, min(per_category, len(records)))
        columns = min(4, len(chosen)) or 1
        rows = (len(chosen) + columns - 1) // columns
        caption = 34
        sheet = Image.new(
            "RGB",
            (columns * (thumb + 8) + 8, rows * (thumb + caption + 8) + 8),
            "#f0f0f0",
        )
        draw = ImageDraw.Draw(sheet)
        for index, record in enumerate(chosen):
            path = config.paths.work_dir / str(record["image_path"])
            try:
                with Image.open(path) as opened:
                    tile = opened.convert("RGB")
            except Exception:  # noqa: BLE001 - a missing tile is not fatal
                continue
            tile.thumbnail((thumb, thumb))
            column, row = index % columns, index // columns
            x = 8 + column * (thumb + 8) + (thumb - tile.width) // 2
            y = 8 + row * (thumb + caption + 8)
            sheet.paste(tile, (x, y))
            label = f"{record['sample_id']} {record['template_id']}"
            draw.text(
                (8 + column * (thumb + 8), y + thumb + 4),
                label[:48],
                fill="#202020",
                font=caption_font,
            )
            snippet = normalise_for_eval(str(record.get("text_render", "")))[:46]
            draw.text(
                (8 + column * (thumb + 8), y + thumb + 18),
                snippet,
                fill="#505050",
                font=caption_font,
            )
        target = directory / f"{category}.png"
        sheet.save(target)
        written.append(target)
    return written


def run_qc(config: DiverseConfig) -> dict[str, Any]:
    """Run stage 06 end to end.

    Args:
        config: The active configuration.

    Returns:
        The report, also written beside the outputs.
    """
    accepted: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    seen_hashes: dict[str, dict[int, str]] = {}
    total = 0

    for _, record in _iter_annotations(config.paths.annotations_dir):
        total += 1
        reason, metrics = check_sample(record, config, seen_hashes=seen_hashes)
        if reason is not None:
            reasons[reason.split(":", 1)[0]] += 1
            rejections.append(
                {
                    "sample_id": record.get("sample_id"),
                    "category": record.get("category"),
                    "template_id": record.get("template_id"),
                    "reason": reason,
                    "metrics": metrics,
                }
            )
            continue
        record["qc"] = metrics
        accepted.append(record)

    config.paths.accepted_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.paths.accepted_jsonl.open("w", encoding="utf-8") as handle:
        for record in accepted:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with config.paths.image_rejections_jsonl.open("w", encoding="utf-8") as handle:
        for record in rejections:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    sheets = _contact_sheet(accepted, config)
    rate = len(rejections) / max(1, total)
    report = {
        "run_name": config.run_name,
        "read": total,
        "accepted": len(accepted),
        "rejected": len(rejections),
        "rejection_rate": round(rate, 4),
        "max_rejection_rate": config.image_qc.max_rejection_rate,
        "by_reason": {name: reasons.get(name, 0) for name in REJECTION_REASONS},
        "accepted_by_category": {
            name: sum(1 for record in accepted if record["category"] == name)
            for name in CATEGORIES
        },
        "contact_sheets": [str(path) for path in sheets],
    }
    write_json(config.paths.work_dir / config.paths.QC / "image_qc_report.json", report)

    LOGGER.info("")
    LOGGER.info("=" * 66)
    LOGGER.info("IMAGE QC REPORT")
    LOGGER.info("=" * 66)
    LOGGER.info("read      %d", total)
    LOGGER.info("accepted  %d", len(accepted))
    LOGGER.info("rejected  %d  (%.1f%%)", len(rejections), rate * 100)
    LOGGER.info("")
    for name in REJECTION_REASONS:
        count = reasons.get(name, 0)
        if count:
            LOGGER.info("  %-16s %5d", name, count)
    LOGGER.info("")
    LOGGER.info("Accepted by category: %s", report["accepted_by_category"])
    LOGGER.info(
        "Contact sheets: %d written to %s",
        len(sheets),
        config.paths.reports_dir / "contact_sheets",
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
    parser.add_argument(
        "--allow-high-rejection",
        action="store_true",
        help="exit 0 even above image_qc.max_rejection_rate",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; non-zero when too much was rejected to continue
        without a look at the reasons.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    config = load_diverse_config(args.config)
    report = run_qc(config)
    rate = float(report["rejection_rate"])
    ceiling = float(report["max_rejection_rate"])
    if rate > ceiling and not args.allow_high_rejection:
        LOGGER.error("")
        LOGGER.error(
            "Rejection rate %.1f%% exceeds the %.0f%% ceiling. Read %s before scaling.",
            rate * 100,
            ceiling * 100,
            config.paths.image_rejections_jsonl,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
