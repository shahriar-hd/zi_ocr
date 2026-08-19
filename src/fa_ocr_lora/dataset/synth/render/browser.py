"""Chromium rendering: HTML in, per-sheet PNGs plus a PDF plus DOM rects out.

Chromium is used rather than a Python drawing library for one reason that
matters more than any other here: it is the only renderer that can be *asked*
where it put something. Persian shaping, bidi resolution and justification all
move glyphs in ways a compositor written in Pillow would have to reimplement,
and every such reimplementation is a chance for the bounding boxes to disagree
with the pixels. ``getBoundingClientRect()`` cannot disagree with them.

Two consequences shape this module:

* **One browser per run.** Launching Chromium costs ~300 ms, which at 20k
  samples is an hour and a half of pure startup. :class:`PageRenderer` holds
  the browser open and creates a fresh *context* per sample instead, which is
  cheap and still isolates the page's DPI and viewport.
* **One document, three artifacts.** The screenshots, the PDF and the rects are
  all read from the same laid-out DOM, in that order, without re-navigating.
  They therefore cannot drift apart - which is the whole reason the PDF is not
  generated separately by a PDF library.

DPI is expressed as ``device_scale_factor = dpi / 96``: the page is laid out in
CSS pixels at its true millimetre size and rasterised at a multiple of that, so
a 300 DPI render and a 96 DPI render share one layout and one set of boxes,
scaled.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Mapping, Sequence

from ....config import CSS_PPI, PageSizeSpec, RenderConfig

__all__ = [
    "RenderError",
    "RenderTarget",
    "RenderedSheet",
    "RenderResult",
    "PageRenderer",
    "sheet_image_names",
    "as_mapping_sequence",
]

#: Injected pagination script, kept beside this module and shipped as package
#: data (see ``pyproject.toml``).
_PAGINATE_JS: Final[Path] = Path(__file__).with_name("paginate.js")

#: Chromium flags used when a caller passes none. The shipped config carries
#: the same list as ``render.chromium_args`` and the pipeline always passes it;
#: this default exists so a direct ``PageRenderer()`` in a script or a test still
#: gets the deterministic text rendering the corpus depends on.
_DEFAULT_LAUNCH_ARGS: Final[tuple[str, ...]] = (
    "--font-render-hinting=none",
    "--disable-lcd-text",
    "--force-color-profile=srgb",
    "--hide-scrollbars",
)


class RenderError(RuntimeError):
    """Raised when a page cannot be laid out or rasterised."""


@lru_cache(maxsize=1)
def _paginate_source() -> str:
    """Return the pagination script, read once per process.

    Returns:
        The JavaScript source as a single function expression.

    Raises:
        RenderError: If the script is missing from the installed package.
    """
    try:
        return _PAGINATE_JS.read_text(encoding="utf-8")
    except OSError as error:
        raise RenderError(f"pagination script unavailable: {error}") from error


@dataclass(frozen=True, slots=True)
class RenderTarget:
    """The paper one sample is rendered onto.

    Attributes:
        page: Chosen page size, already oriented.
        dpi: Raster resolution for the PNGs.
        margin_mm: Printable margin, passed to the template.
        max_pages: Hard cap on sheets; a longer letter is reported as
            ``overflowed`` rather than silently growing.
        timeout_ms: Per-page budget for layout and evaluation.
    """

    page: PageSizeSpec
    dpi: int
    margin_mm: float
    max_pages: int = 6
    timeout_ms: int = 30_000

    @classmethod
    def from_config(
        cls,
        page: PageSizeSpec,
        margin_mm: float,
        render: RenderConfig,
    ) -> "RenderTarget":
        """Build a target from the loaded render config.

        Args:
            page: Page size drawn for this sample.
            margin_mm: Margin drawn for this sample.
            render: The config's ``render`` section.

        Returns:
            The assembled target.
        """
        return cls(
            page=page,
            dpi=render.dpi,
            margin_mm=margin_mm,
            max_pages=render.max_pages,
            timeout_ms=render.timeout_ms,
        )

    @property
    def scale(self) -> float:
        """Return the CSS-pixel to device-pixel ratio."""
        return self.dpi / CSS_PPI

    @property
    def viewport(self) -> dict[str, int]:
        """Return the CSS-pixel viewport matching one sheet.

        The viewport is one sheet tall even for a multi-sheet document: the
        sheets stack below the fold and are screenshotted individually, and a
        taller viewport would only change which of them Chromium happens to
        have composited.
        """
        return {
            "width": max(1, round(self.page.css_width)),
            "height": max(1, round(self.page.css_height)),
        }

    @property
    def device_size(self) -> tuple[int, int]:
        """Return one sheet's size in device pixels at this DPI."""
        return self.page.device_size(self.dpi)


@dataclass(frozen=True, slots=True)
class RenderedSheet:
    """One rasterised page.

    Attributes:
        index: Zero-based sheet number, matching the PDF's page order.
        png: PNG bytes at the target DPI.
        width: Raster width in device pixels.
        height: Raster height in device pixels.
    """

    index: int
    png: bytes
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Everything one Chromium pass produced.

    Attributes:
        sheets: Rasterised pages in order.
        pdf: The whole document as a single PDF, one page per sheet.
        boxes: Raw harvest from ``paginate.js``, in CSS pixels.
        scale: Factor to multiply CSS pixels by to reach device pixels.
        overflowed: Whether content had to be clipped to fit ``max_pages``.
    """

    sheets: tuple[RenderedSheet, ...]
    pdf: bytes
    boxes: tuple[Mapping[str, Any], ...]
    scale: float
    overflowed: bool

    @property
    def page_count(self) -> int:
        """Return how many sheets were produced."""
        return len(self.sheets)


class PageRenderer:
    """A reusable Chromium instance that renders letters.

    Use as a context manager so the browser is closed even when a sample
    raises::

        with PageRenderer() as renderer:
            result = renderer.render(html, target)
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        launch_args: Sequence[str] | None = None,
    ) -> None:
        """Start Playwright and launch Chromium.

        Args:
            headless: Whether to run without a visible window. PDF generation
                requires headless Chromium, so this is only ever turned off
                for interactive debugging of the layout.
            launch_args: Chromium flags, normally ``config.render.chromium_args``.
                Defaults to :data:`_DEFAULT_LAUNCH_ARGS`, which is what the
                shipped config carries.

        Raises:
            RenderError: If Playwright or its Chromium build is unavailable.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:  # pragma: no cover - environment issue
            raise RenderError(
                "playwright is required to render pages; "
                "install it and run 'python -m playwright install chromium'"
            ) from error

        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(
                headless=headless,
                args=list(
                    launch_args if launch_args is not None else _DEFAULT_LAUNCH_ARGS
                ),
            )
        except Exception as error:  # pragma: no cover - environment issue
            self._playwright.stop()
            raise RenderError(f"could not launch Chromium: {error}") from error

    def __enter__(self) -> "PageRenderer":
        """Return self, for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the browser."""
        self.close()

    def close(self) -> None:
        """Shut Chromium and Playwright down. Safe to call twice."""
        browser = getattr(self, "_browser", None)
        if browser is not None:
            browser.close()
            self._browser = None
        playwright = getattr(self, "_playwright", None)
        if playwright is not None:
            playwright.stop()
            self._playwright = None

    def render(self, html: str, target: RenderTarget) -> RenderResult:
        """Lay out one letter and return its rasters, PDF and boxes.

        Args:
            html: The complete document produced from ``letter.html.j2``.
            target: Paper size, DPI and page cap for this sample.

        Returns:
            The rendered artifacts and the raw box harvest.

        Raises:
            RenderError: If the browser is closed, the page fails to lay out,
                or pagination yielded no sheet at all.
        """
        if getattr(self, "_browser", None) is None:
            raise RenderError("renderer is closed")

        context = self._browser.new_context(
            viewport=target.viewport,
            device_scale_factor=target.scale,
        )
        try:
            page = context.new_page()
            page.set_default_timeout(target.timeout_ms)
            # Screen media, not print: the PDF must show the same tinted bands
            # and background scans the PNGs do, and Chromium's print stylesheet
            # would strip them.
            page.emulate_media(media="screen")
            page.set_content(html, wait_until="load")

            harvest = page.evaluate(
                _paginate_source(),
                {"maxPages": target.max_pages},
            )
            sheet_count = len(harvest.get("sheets", ()))
            if sheet_count == 0:
                raise RenderError("pagination produced no sheets")

            sheets = self._screenshot_sheets(page, sheet_count, target)
            pdf = page.pdf(
                width=f"{target.page.width_mm}mm",
                height=f"{target.page.height_mm}mm",
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                print_background=True,
                prefer_css_page_size=True,
            )
        except RenderError:
            raise
        except Exception as error:
            raise RenderError(f"render failed: {error}") from error
        finally:
            context.close()

        return RenderResult(
            sheets=sheets,
            pdf=pdf,
            boxes=tuple(harvest.get("boxes", ())),
            scale=target.scale,
            overflowed=bool(harvest.get("overflowed", False)),
        )

    @staticmethod
    def _screenshot_sheets(
        page: Any,
        count: int,
        target: RenderTarget,
    ) -> tuple[RenderedSheet, ...]:
        """Rasterise each ``.sheet`` element separately.

        Element screenshots are used rather than one full-page capture with
        clips because the element *is* the page: Chromium measures the crop
        from the same box the annotation was measured from, so a sheet's raster
        and its boxes share an origin by construction.

        The capture omits the browser's background, so the PNG is ``RGBA`` with
        a transparent ground: the page's light areas are an *absence of ink*
        rather than a coat of white. ``render/scan.py`` then composites paper
        underneath them. Flattening here instead would bury the paper.

        Args:
            page: The Playwright page.
            count: How many sheets pagination created.
            target: The render target, for the expected device size.

        Returns:
            One :class:`RenderedSheet` per sheet, in document order.
        """
        width, height = target.device_size
        locator = page.locator(".sheet")
        sheets: list[RenderedSheet] = []
        for index in range(count):
            png = locator.nth(index).screenshot(
                type="png", animations="disabled", omit_background=True
            )
            sheets.append(
                RenderedSheet(index=index, png=png, width=width, height=height)
            )
        return tuple(sheets)


def sheet_image_names(
    sample_id: str, count: int, suffix: str = ".png"
) -> tuple[str, ...]:
    """Return the conventional per-page image filenames for a sample.

    Args:
        sample_id: The render-stage sample id, e.g. ``synth_000000``.
        count: Number of pages.
        suffix: File extension, including the dot. Comes from
            :attr:`~fa_ocr_lora.config.ImageFormatConfig.suffix`, so the name
            and the encoding cannot disagree.

    Returns:
        One name per page, ``synth_000000_p01.png`` style, so a directory
        listing sorts into reading order.
    """
    return tuple(f"{sample_id}_p{index + 1:02d}{suffix}" for index in range(count))


def as_mapping_sequence(boxes: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Copy a raw box harvest into plain dictionaries.

    Playwright returns nested JSON already, but the objects come back as
    read-only views in some transports; the annotation stage mutates while it
    assigns reading order, so it works on copies.

    Args:
        boxes: Raw boxes from :attr:`RenderResult.boxes`.

    Returns:
        A tuple of mutable dictionaries.
    """
    return tuple(dict(box) for box in boxes)
