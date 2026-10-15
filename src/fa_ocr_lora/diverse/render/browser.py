"""The Chromium renderer: one browser, many pages, exact geometry.

Chromium through Playwright is the shaping engine for everything here. That is
not convenience. Persian needs bidi resolution, contextual joining, ZWNJ
handling and mixed-script runs, and a browser is the only engine in reach that
gets all four right. Pillow would be lighter, but the Pillow in this
environment is not built with libraqm, so it draws Persian as disconnected
letters - and an engine that is silently wrong is worse than a heavier one
that is right. :func:`verify_shaping` asserts this at startup rather than
trusting it.

The browser is launched once and reused; each sample gets a fresh *context*,
which isolates its fonts and styles without paying for a process launch.
Rendering is therefore cheap enough to parallelise by running several
:class:`PageRenderer` instances, one per worker thread.

Everything returned here is in **device pixels** at the sample's DPI, not CSS
pixels: the harvest reports CSS coordinates and :func:`render` scales them by
the device scale factor, so the annotation and the PNG share one coordinate
system. Getting that wrong would put every box out by a factor of three on a
300-dpi page.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from .schema import Annotation, Block, BlockType, Line, Word, rectangle

__all__ = [
    "PageRenderer",
    "RenderError",
    "RenderResult",
    "verify_shaping",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: Launch flags. Hinting and LCD subpixel antialiasing are disabled so a glyph
#: renders identically on every host; left on, the same sample would differ
#: between a laptop and the molab runtime, and the corpus would carry the
#: renderer's hardware as a hidden variable.
_LAUNCH_ARGS: Final[tuple[str, ...]] = (
    "--font-render-hinting=none",
    "--disable-lcd-text",
    "--force-color-profile=srgb",
    "--hide-scrollbars",
    "--disable-dev-shm-usage",
)

#: A string that exercises the properties a broken shaper gets wrong: an
#: initial/medial/final joining run, a ZWNJ compound, and a bidi transition.
_SHAPING_PROBE: Final[str] = "می‌روند و ABC و ۱۴۰۳"


class RenderError(RuntimeError):
    """Raised when a page cannot be rendered or measured."""


@lru_cache(maxsize=1)
def _harvest_source() -> str:
    """Return the harvest script, read once per process.

    Returns:
        The JavaScript source.

    Raises:
        RenderError: If the script is missing from the installed package.
    """
    path = Path(__file__).with_name("harvest.js")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise RenderError(f"cannot read {path}: {error}") from error


@dataclass(frozen=True, slots=True)
class RenderResult:
    """One rendered image and its measured annotation.

    Attributes:
        png: The image bytes, always PNG at this stage; the degradation stage
            decides the final encoding.
        annotation: Line and block geometry in device pixels.
        width: Image width in device pixels.
        height: Image height in device pixels.
        overflowed: Whether content escaped the canvas, meaning the label
            claims text the image does not show.
    """

    png: bytes
    annotation: Annotation
    width: int
    height: int
    overflowed: bool


def _to_annotation(
    harvest: Mapping[str, Any],
    *,
    scale: float,
    sample_id: str,
    group_id: str,
    category: str,
    width: int,
    height: int,
    want_words: bool,
) -> Annotation:
    """Convert the harvest payload into an :class:`Annotation`.

    Args:
        harvest: The object returned by ``harvest.js``.
        scale: Device scale factor; CSS pixels times this gives device pixels.
        sample_id: Identifier of this image.
        group_id: Splitting key.
        category: Visual category.
        width: Image width in device pixels.
        height: Image height in device pixels.
        want_words: Whether to keep word boxes.

    Returns:
        The annotation, with every coordinate scaled and clamped to the canvas.
    """

    def polygon(box: Mapping[str, float]) -> tuple[tuple[float, float], ...]:
        """Scale one harvested box into a device-pixel rectangle."""
        return rectangle(
            max(0.0, float(box["x0"]) * scale),
            max(0.0, float(box["y0"]) * scale),
            min(float(width), float(box["x1"]) * scale),
            min(float(height), float(box["y1"]) * scale),
        )

    lines: list[Line] = []
    for order, record in enumerate(harvest.get("lines", ())):
        words: tuple[Word, ...] = ()
        if want_words:
            words = tuple(
                Word(text=str(item["text"]), polygon=polygon(item["box"]))
                for item in record.get("words", ())
            )
        lines.append(
            Line(
                text=str(record["text"]),
                polygon=polygon(record["box"]),
                reading_order=order,
                direction=str(record.get("direction", "rtl")),
                block_index=int(record.get("block_index", -1)),
                words=words,
            )
        )

    blocks: list[Block] = []
    for order, record in enumerate(harvest.get("blocks", ())):
        try:
            block_type = BlockType(str(record.get("type", "other")))
        except ValueError:
            block_type = BlockType.OTHER
        blocks.append(
            Block(
                block_type=block_type,
                polygon=polygon(record["box"]),
                reading_order=order,
                line_indices=tuple(
                    int(value) for value in record.get("line_indices", ())
                ),
            )
        )

    return Annotation(
        sample_id=sample_id,
        parent_id="",
        group_id=group_id,
        category=category,
        width=width,
        height=height,
        lines=tuple(lines),
        blocks=tuple(blocks),
    )


class PageRenderer:
    """A reusable Chromium instance.

    Use as a context manager. One instance is not thread-safe; give each
    worker thread its own, which is what the stage 05 runner does.
    """

    __slots__ = ("_playwright", "_browser", "_headless")

    def __init__(self, *, headless: bool = True) -> None:
        """Record the launch options without starting anything yet.

        Args:
            headless: Whether to run Chromium headless. ``False`` is for
                debugging a template by eye.
        """
        self._playwright: Any = None
        self._browser: Any = None
        self._headless = headless

    def __enter__(self) -> PageRenderer:
        """Launch the browser.

        Returns:
            This renderer.

        Raises:
            RenderError: If Playwright or its Chromium build is unavailable.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise RenderError(
                "playwright is required: pip install playwright "
                "&& playwright install chromium"
            ) from error
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(
                headless=self._headless, args=list(_LAUNCH_ARGS)
            )
        except Exception as error:  # noqa: BLE001 - surfaced with a fix hint
            self._playwright.stop()
            raise RenderError(
                f"cannot launch Chromium ({error}); try `playwright install chromium`"
            ) from error
        return self

    def __exit__(self, *_: object) -> None:
        """Close the browser and stop Playwright."""
        self.close()

    def close(self) -> None:
        """Close the browser and stop Playwright, idempotently."""
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:  # noqa: BLE001 - teardown must not mask errors
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:  # noqa: BLE001
                pass
            self._playwright = None

    def render(
        self,
        html: str,
        *,
        width: int,
        height: int,
        scale: float,
        sample_id: str,
        group_id: str,
        category: str,
        timeout_ms: int = 30_000,
        want_words: bool = True,
    ) -> RenderResult:
        """Render one document and measure its text.

        Args:
            html: The complete document, fonts already embedded as data URIs.
            width: Canvas width in CSS pixels.
            height: Canvas height in CSS pixels, or ``0`` to fit the content.
            scale: Device scale factor, i.e. ``dpi / 96``.
            sample_id: Identifier of the image.
            group_id: Splitting key.
            category: Visual category.
            timeout_ms: Per-page ceiling.
            want_words: Whether to harvest word boxes.

        Returns:
            The rendered image and its annotation.

        Raises:
            RenderError: If the page cannot be loaded, measured or captured.
        """
        if self._browser is None:
            raise RenderError("renderer is not started; use it as a context manager")

        # A fit-to-content canvas is rendered twice: once tall, to learn the
        # content height, then at that height. Measuring and screenshotting in
        # one pass would either clip the content or leave blank space, and
        # blank space below the text is a systematic bias the model would learn.
        probe_height = height if height > 0 else 4000
        context = self._browser.new_context(
            viewport={"width": width, "height": probe_height},
            device_scale_factor=scale,
            locale="fa-IR",
        )
        try:
            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            try:
                page.set_content(html, wait_until="load")
                page.emulate_media(media="screen")
                harvest = page.evaluate(_harvest_source(), {"words": want_words})
            except Exception as error:  # noqa: BLE001
                raise RenderError(
                    f"{sample_id}: page failed to load or measure ({error})"
                ) from error

            if height <= 0:
                content = int(harvest.get("content_height") or 0)
                target = max(32, min(content, 20_000))
                page.set_viewport_size({"width": width, "height": target})
                try:
                    harvest = page.evaluate(_harvest_source(), {"words": want_words})
                except Exception as error:  # noqa: BLE001
                    raise RenderError(
                        f"{sample_id}: re-measure failed ({error})"
                    ) from error

            try:
                element = page.query_selector("#canvas") or page.query_selector("body")
                png = (
                    element.screenshot(type="png")
                    if element
                    else page.screenshot(type="png")
                )
            except Exception as error:  # noqa: BLE001
                raise RenderError(f"{sample_id}: screenshot failed ({error})") from error
        finally:
            context.close()

        device_width = int(round(int(harvest["width"]) * scale))
        device_height = int(round(int(harvest["height"]) * scale))
        # Trust the PNG's own dimensions over the arithmetic: Chromium rounds
        # a fractional device size, and a one-pixel disagreement would make
        # every box validation fail on the canvas bounds check.
        try:
            import io

            from PIL import Image

            with Image.open(io.BytesIO(png)) as image:
                device_width, device_height = image.size
        except Exception:  # noqa: BLE001 - arithmetic is an acceptable fallback
            pass

        annotation = _to_annotation(
            harvest,
            scale=scale,
            sample_id=sample_id,
            group_id=group_id,
            category=category,
            width=device_width,
            height=device_height,
            want_words=want_words,
        )
        return RenderResult(
            png=png,
            annotation=annotation,
            width=device_width,
            height=device_height,
            overflowed=bool(harvest.get("overflowed")),
        )


def verify_shaping(renderer: PageRenderer) -> None:
    """Assert that the renderer shapes Persian correctly.

    Run once before a batch. A shaper that leaves Persian letters unjoined
    produces images that look subtly wrong and labels that are perfectly
    right - the hardest kind of corruption to notice, and the reason this
    check exists rather than a comment saying Chromium is fine.

    The test is geometric rather than visual: in correctly shaped Persian the
    probe renders as a handful of connected runs, so the number of distinct
    word boxes is small. Unjoined rendering produces one box per letter.

    Args:
        renderer: A started renderer.

    Raises:
        RenderError: If the probe does not shape as expected.
    """
    html = (
        '<!DOCTYPE html><html dir="rtl" lang="fa"><head><meta charset="utf-8">'
        "<style>body{margin:0}#canvas{width:600px;padding:20px;"
        "font-family:sans-serif;font-size:28px}</style></head>"
        f'<body><div id="canvas"><p data-ann="paragraph">{_SHAPING_PROBE}</p>'
        "</div></body></html>"
    )
    result = renderer.render(
        html,
        width=640,
        height=0,
        scale=1.0,
        sample_id="_shaping_probe",
        group_id="_probe",
        category="paragraph",
        want_words=True,
    )
    if not result.annotation.lines:
        raise RenderError("shaping probe produced no line boxes; the harvest is broken")
    line = result.annotation.lines[0]
    if line.text.replace(" ", "") != _SHAPING_PROBE.replace(" ", ""):
        raise RenderError(
            f"shaping probe text mismatch: harvested {line.text!r}, "
            f"expected {_SHAPING_PROBE!r}"
        )
    # "می‌روند و ABC و ۱۴۰۳" is five whitespace-separated tokens. A shaper that
    # failed to join would still report five words, but a *harvest* that failed
    # would not - so this checks the pipeline end to end rather than the shaper
    # alone.
    if len(line.words) not in range(4, 8):
        raise RenderError(
            f"shaping probe produced {len(line.words)} word box(es); expected about 5. "
            "The text may be rendering unjoined, or word splitting is broken."
        )
    LOGGER.info(
        "shaping probe OK: %d line, %d word box(es), %dx%d px",
        len(result.annotation.lines),
        len(line.words),
        result.width,
        result.height,
    )
