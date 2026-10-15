"""Stage 05 CLI - render, degrade, crop and annotate.

Runs on the molab runtime, where Chromium and the bandwidth are. Needs only
``layouts.jsonl``: the specs embed their fonts, so this stage is independent of
the lexicon, the font directory and the API keys.

Parallelism is one browser per worker thread. Chromium is not thread-safe
across contexts in the sync Playwright API, so sharing one instance would
serialise the work or crash; a browser per worker costs about 80 MB and
scales linearly to the core count.

Resumability is by annotation file. The annotation is written *last*, after the
image and after the crops, so its presence means the sample is completely
done - a run killed between the image and the annotation redoes that sample
rather than leaving a page with no ground truth.

The verso pool
--------------

Show-through needs another sample's render. Rather than render each ghost
twice, workers deposit their finished pages into a small bounded pool and draw
a ghost from it. The pool is capped because holding every page in memory would
exhaust the box at scale, and it is seeded from samples the worker has already
produced, so an early sample simply gets no ghost - which is correct, since
show-through is probabilistic anyway.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import random
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

from ...dataset.common.io_utils import sample_rng, write_bytes, write_json
from ..config import CATEGORIES, DiverseConfig, load_diverse_config
from ..layouts.spec import RenderSpec
from .browser import PageRenderer, RenderError, verify_shaping
from .crops import derive_crops
from .degrade import apply_degradation
from .schema import Annotation

__all__ = ["render_batch", "main"]

LOGGER: Final[logging.Logger] = logging.getLogger("fa_diverse.render")

#: How many finished pages to keep for use as show-through ghosts. Small: the
#: ghost is unrecognisable anyway, so variety matters more than depth, and a
#: large pool would hold tens of megabytes of PNG per worker.
_VERSO_POOL: Final[int] = 12


def _iter_specs(path: Path) -> Iterator[RenderSpec]:
    """Stream the stage 04 output.

    Args:
        path: ``layouts.jsonl``.

    Yields:
        One spec per line.

    Raises:
        FileNotFoundError: If stage 04 has not run.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `python -m fa_ocr_lora.diverse.layouts.build` first"
        )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield RenderSpec.from_dict(json.loads(line))


def image_path(config: DiverseConfig, sample_id: str, suffix: str) -> Path:
    """Return the path of a rendered image.

    Sharded by the first four characters of the sample id, so no directory
    holds more than a few thousand files - which keeps ``ls`` usable and
    avoids the per-directory limits some filesystems and the Hub impose.

    Args:
        config: The active configuration.
        sample_id: Identifier of the image.
        suffix: File extension including the dot.

    Returns:
        The path.
    """
    shard = sample_id[3:7] if sample_id.startswith("fa_") else sample_id[:4]
    return config.paths.images_dir / shard / f"{sample_id}{suffix}"


def annotation_path(config: DiverseConfig, sample_id: str) -> Path:
    """Return the path of a sample's annotation.

    Args:
        config: The active configuration.
        sample_id: Identifier of the sample.

    Returns:
        The path.
    """
    shard = sample_id[3:7] if sample_id.startswith("fa_") else sample_id[:4]
    return config.paths.annotations_dir / shard / f"{sample_id}.json"


class _Counters:
    """Thread-safe tallies for one render run."""

    __slots__ = (
        "skipped",
        "rendered",
        "crops",
        "failed",
        "overflowed",
        "by_category",
        "by_format",
        "_lock",
    )

    def __init__(self, skipped: int = 0) -> None:
        """Start the tallies.

        Args:
            skipped: Samples already complete before the run began.
        """
        self.skipped = skipped
        self.rendered = 0
        self.crops = 0
        self.failed = 0
        self.overflowed = 0
        self.by_category: Counter[str] = Counter()
        self.by_format: Counter[str] = Counter()
        self._lock = threading.Lock()

    def record(
        self, category: str, image_format: str, crops: int, overflowed: bool
    ) -> int:
        """Record one finished sample.

        Args:
            category: Visual category.
            image_format: ``png`` or ``jpeg``.
            crops: Crops derived from it.
            overflowed: Whether content escaped the canvas.

        Returns:
            The new ``rendered`` total.
        """
        with self._lock:
            self.rendered += 1
            self.crops += crops
            self.by_category[category] += 1
            self.by_format[image_format] += 1
            if overflowed:
                self.overflowed += 1
            return self.rendered

    def record_failure(self) -> None:
        """Record a sample that could not be rendered."""
        with self._lock:
            self.failed += 1

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable summary."""
        return {
            "skipped": self.skipped,
            "rendered": self.rendered,
            "crops": self.crops,
            "failed": self.failed,
            "overflowed": self.overflowed,
            "by_category": {name: self.by_category.get(name, 0) for name in CATEGORIES},
            "by_image_format": dict(self.by_format),
        }


def _write_sample(
    config: DiverseConfig,
    spec: RenderSpec,
    payload: bytes,
    image_format: str,
    annotation: Annotation,
    aug: dict[str, Any],
) -> None:
    """Persist one image and its annotation.

    The annotation is written last and is what ``--resume`` tests, so a run
    interrupted mid-sample leaves no image without ground truth.

    Args:
        config: The active configuration.
        spec: The sample's spec.
        payload: Encoded image bytes.
        image_format: ``png`` or ``jpeg``.
        annotation: Final geometry.
        aug: What was applied, for the manifest.
    """
    suffix = ".jpg" if image_format == "jpeg" else ".png"
    target = image_path(config, spec.sample_id, suffix)
    write_bytes(target, payload)
    record = annotation.as_dict()
    record.update(
        {
            "image_path": str(target.relative_to(config.paths.work_dir)),
            "image_format": image_format,
            "template_id": spec.template_id,
            "font_ids": list(spec.font_ids),
            "font_families": list(spec.font_families),
            "dpi": spec.dpi,
            "palette": spec.palette.as_dict(),
            "dark_mode": spec.dark_mode,
            "text_render": spec.text_render,
            "aug": aug,
            "seed": spec.seed,
        }
    )
    write_json(annotation_path(config, spec.sample_id), record)


def render_batch(
    config: DiverseConfig,
    *,
    limit: int | None = None,
    overwrite: bool = False,
    workers: int | None = None,
    headless: bool = True,
) -> _Counters:
    """Render every spec not already complete.

    Args:
        config: The active configuration.
        limit: Process at most this many pending specs.
        overwrite: Re-render samples already on disk.
        workers: Parallel browsers; defaults to ``render.workers``.
        headless: Whether Chromium runs headless.

    Returns:
        The run tallies.
    """
    specs = list(_iter_specs(config.paths.layouts_jsonl))
    pending = [
        spec
        for spec in specs
        if overwrite or not annotation_path(config, spec.sample_id).is_file()
    ]
    counters = _Counters(skipped=len(specs) - len(pending))
    if limit is not None:
        pending = pending[:limit]

    LOGGER.info(
        "%d spec(s): %d already rendered, %d to do",
        len(specs),
        counters.skipped,
        len(pending),
    )
    if not pending:
        return counters

    worker_count = max(1, workers if workers is not None else config.render.workers)
    crop_config = config.render.derive_crops

    work: queue.Queue[RenderSpec] = queue.Queue()
    for spec in pending:
        work.put(spec)

    verso_pool: deque[tuple[str, bytes]] = deque(maxlen=_VERSO_POOL)
    verso_lock = threading.Lock()
    stop = threading.Event()
    log_every = max(1, len(pending) // 40)
    started = time.monotonic()

    def run_worker(index: int) -> None:
        """Drain the queue with one browser.

        Args:
            index: Worker number, used only in log messages.
        """
        try:
            with PageRenderer(headless=headless) as renderer:
                if index == 0:
                    verify_shaping(renderer)
                while not stop.is_set():
                    try:
                        spec = work.get_nowait()
                    except queue.Empty:
                        return
                    try:
                        _render_one(
                            config, spec, renderer, verso_pool, verso_lock, crop_config
                        )
                    except RenderError as error:
                        counters.record_failure()
                        LOGGER.warning("%s: %s", spec.sample_id, error)
                        continue
                    except Exception:  # noqa: BLE001 - a thread must not die silently
                        counters.record_failure()
                        LOGGER.exception("%s: unexpected render failure", spec.sample_id)
                        continue
        except RenderError as error:
            LOGGER.error("worker %d could not start: %s", index, error)
            stop.set()

    def _render_one(
        config: DiverseConfig,
        spec: RenderSpec,
        renderer: PageRenderer,
        pool: deque,
        lock: threading.Lock,
        crops_config: Any,
    ) -> None:
        """Render, degrade, crop and persist one sample."""
        result = renderer.render(
            spec.html,
            width=spec.width,
            height=spec.height,
            scale=spec.scale,
            sample_id=spec.sample_id,
            group_id=spec.group_id,
            category=spec.category,
            timeout_ms=config.render.max_page_time_ms,
            want_words=True,
        )

        with lock:
            verso = random.choice(list(pool)) if pool else None

        rng = sample_rng(config.seed, "degrade", hash(spec.sample_id) & 0xFFFFFF)
        degraded = apply_degradation(
            result.png,
            config.degradation,
            rng,
            clean=spec.clean,
            jpeg_quality=(
                config.render.jpeg_quality.minimum,
                config.render.jpeg_quality.maximum,
            ),
            verso_png=verso[1] if verso else None,
            verso_id=verso[0] if verso else "",
            # Measured from the clean render, so the degradation knows how much
            # ink this sample can afford to lose.
            text_height_px=result.annotation.min_line_height() or 32.0,
        )

        # The same mapping that moved the ink moves the boxes. Without this the
        # annotation would describe the pre-warp page.
        annotation = result.annotation
        if degraded.geometry is not None:
            annotation = annotation.transformed(
                degraded.geometry, width=degraded.width, height=degraded.height
            )

        aug = dict(degraded.aug)
        aug.update(
            {
                "overflowed": result.overflowed,
                "holdout_template": bool(spec.aug.get("holdout_template")),
                "holdout_font": bool(spec.aug.get("holdout_font")),
            }
        )
        _write_sample(
            config, spec, degraded.payload, degraded.image_format, annotation, aug
        )

        crop_count = 0
        if spec.category in {"page", "paragraph", "web", "story", "table_form"}:
            for crop in derive_crops(
                degraded.payload,
                annotation,
                rng,
                want_line=crops_config.line,
                want_paragraph=crops_config.paragraph,
                word_fraction=crops_config.word_fraction,
                padding_min=crops_config.padding.minimum,
                padding_max=crops_config.padding.maximum,
                image_format=degraded.image_format,
            ):
                suffix = ".jpg" if crop.image_format == "jpeg" else ".png"
                target = image_path(config, crop.sample_id, suffix)
                write_bytes(target, crop.payload)
                record = crop.annotation.as_dict()
                # `overflowed` deliberately does not propagate. A crop is cut
                # from pixels that are inside the canvas by construction, so a
                # line taken from the visible part of an overflowing page is a
                # perfectly good sample. Propagating the parent's flag rejected
                # 270 sound crops on the first QC run.
                crop_aug = {
                    key: value for key, value in aug.items() if key != "overflowed"
                }
                crop_aug["derived_from"] = spec.sample_id
                record.update(
                    {
                        "image_path": str(target.relative_to(config.paths.work_dir)),
                        "image_format": crop.image_format,
                        "template_id": spec.template_id,
                        "font_ids": list(spec.font_ids),
                        "font_families": list(spec.font_families),
                        "dpi": spec.dpi,
                        "palette": spec.palette.as_dict(),
                        "dark_mode": spec.dark_mode,
                        "text_render": crop.text_render,
                        "aug": crop_aug,
                        "seed": spec.seed,
                    }
                )
                write_json(annotation_path(config, crop.sample_id), record)
                crop_count += 1

        with lock:
            pool.append((spec.sample_id, result.png))

        done = counters.record(
            spec.category, degraded.image_format, crop_count, result.overflowed
        )
        if done % log_every == 0 or done == len(pending):
            elapsed = max(1e-6, time.monotonic() - started)
            LOGGER.info(
                "  %d/%d rendered (%.1f/min, %d crop(s), %d failed)",
                done,
                len(pending),
                done / elapsed * 60.0,
                counters.crops,
                counters.failed,
            )

    LOGGER.info("Starting %d render worker(s)", worker_count)
    threads = [
        threading.Thread(
            target=run_worker, args=(index,), name=f"render-{index}", daemon=True
        )
        for index in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; finishing in flight. Re-run to resume.")
        stop.set()
        for thread in threads:
            thread.join(timeout=60)
        raise
    return counters


def _report(config: DiverseConfig, counters: _Counters) -> None:
    """Print and persist the stage 05 summary.

    Args:
        config: The active configuration.
        counters: The run tallies.
    """
    summary = counters.as_dict()
    LOGGER.info("")
    LOGGER.info("=" * 64)
    LOGGER.info("RENDER SUMMARY")
    LOGGER.info("=" * 64)
    LOGGER.info("skipped     %d", counters.skipped)
    LOGGER.info("rendered    %d", counters.rendered)
    LOGGER.info("crops       %d", counters.crops)
    LOGGER.info("failed      %d", counters.failed)
    LOGGER.info("overflowed  %d (image QC will reject these)", counters.overflowed)
    LOGGER.info("by category %s", summary["by_category"])
    LOGGER.info("by format   %s", summary["by_image_format"])
    write_json(
        config.paths.work_dir / config.paths.RENDER / "render_summary.json", summary
    )


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
        "--limit", type=int, default=None, help="render at most N pending specs"
    )
    parser.add_argument("--workers", type=int, default=None, help="parallel browsers")
    parser.add_argument(
        "--overwrite", action="store_true", help="re-render existing samples"
    )
    parser.add_argument(
        "--headed", action="store_true", help="show the browser, for debugging"
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
    counters = render_batch(
        config,
        limit=args.limit,
        overwrite=args.overwrite,
        workers=args.workers,
        headless=not args.headed,
    )
    _report(config, counters)
    return 1 if counters.rendered == 0 and counters.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
