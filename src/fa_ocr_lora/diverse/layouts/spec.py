"""The render spec: what one sample should look like, decided before rendering.

Stage 04 produces one of these per sample and writes it to ``layouts.jsonl``;
stage 05 consumes it and produces pixels. Keeping the decision separate from
the drawing has two payoffs. A layout can be inspected, counted and
re-balanced without launching a browser - which is how the coverage report
knows the corpus has 40 templates and not 4 - and the molab render stage
becomes a pure function of a text file plus a spec, so it is reproducible and
trivially parallel.

The palette is the part worth reading. Colour is a strong confound in OCR: a
model trained only on black-on-white loses accuracy the moment contrast drops,
which is most real scans. So ink and paper are sampled per sample, but never
freely - every pair is checked against a WCAG-style contrast ratio, because a
sample whose text is genuinely illegible is not hard training data, it is a
wrong label. :func:`build_palette` darkens the ink until the pair clears the
floor rather than rejecting the draw, so the distribution stays wide.
"""

from __future__ import annotations

import colorsys
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

__all__ = [
    "Palette",
    "RenderSpec",
    "build_palette",
    "contrast_ratio",
    "hex_to_rgb",
    "rgb_to_hex",
]

#: Minimum contrast ratio between ink and paper. 4.5 is WCAG AA for body text;
#: below about 3 a human starts guessing, and a label a human cannot verify is
#: a label that cannot be trusted.
_CONTRAST_FLOOR: Final[float] = 4.5

#: Looser floor for display text - banners and posters are large, so lower
#: contrast is still legible and is what real signage looks like.
_CONTRAST_FLOOR_DISPLAY: Final[float] = 3.0

#: How far to darken the ink per attempt when the pair is too close.
_DARKEN_STEP: Final[float] = 0.12

#: Attempts before giving up and using near-black.
_MAX_DARKEN: Final[int] = 10


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Parse a ``#rrggbb`` colour.

    Args:
        value: Hex colour, with or without the leading ``#``.

    Returns:
        The ``(r, g, b)`` components, 0-255.

    Raises:
        ValueError: If the string is not six hex digits.
    """
    text = value.lstrip("#")
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    if len(text) != 6:
        raise ValueError(f"expected #rrggbb, got {value!r}")
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: Sequence[int]) -> str:
    """Format an ``(r, g, b)`` triple as ``#rrggbb``.

    Args:
        rgb: The components, 0-255.

    Returns:
        The hex string.
    """
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(value))) for value in rgb))


def _relative_luminance(rgb: Sequence[int]) -> float:
    """Return the WCAG relative luminance of a colour.

    Args:
        rgb: The components, 0-255.

    Returns:
        Luminance in ``[0, 1]``.
    """
    channels = []
    for value in rgb:
        scaled = value / 255.0
        channels.append(
            scaled / 12.92 if scaled <= 0.04045 else ((scaled + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(first: str, second: str) -> float:
    """Return the WCAG contrast ratio between two colours.

    Args:
        first: A ``#rrggbb`` colour.
        second: A ``#rrggbb`` colour.

    Returns:
        The ratio, from 1.0 (identical) to 21.0 (black on white).
    """
    light = _relative_luminance(hex_to_rgb(first))
    dark = _relative_luminance(hex_to_rgb(second))
    if light < dark:
        light, dark = dark, light
    return (light + 0.05) / (dark + 0.05)


@dataclass(frozen=True, slots=True)
class Palette:
    """The colours of one sample.

    Attributes:
        paper: Background colour.
        ink: Body text colour.
        accent: Headings, rules and links.
        muted: Secondary text - captions, timestamps, footers.
        panel: Table stripes and callout backgrounds.
        contrast: Achieved ink-on-paper contrast ratio, recorded so image QC
            can cross-check the rendered pixels against the intent.
    """

    paper: str
    ink: str
    accent: str
    muted: str
    panel: str
    contrast: float

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form."""
        return asdict(self)


def _shift(value: str, *, lightness: float = 0.0, saturation: float = 0.0) -> str:
    """Nudge a colour's lightness and saturation in HLS space.

    Args:
        value: A ``#rrggbb`` colour.
        lightness: Additive change, clamped to ``[0, 1]``.
        saturation: Additive change, clamped to ``[0, 1]``.

    Returns:
        The adjusted colour.
    """
    red, green, blue = (channel / 255.0 for channel in hex_to_rgb(value))
    hue, light, sat = colorsys.rgb_to_hls(red, green, blue)
    light = min(1.0, max(0.0, light + lightness))
    sat = min(1.0, max(0.0, sat + saturation))
    return rgb_to_hex(
        [round(channel * 255) for channel in colorsys.hls_to_rgb(hue, light, sat)]
    )


def build_palette(
    rng: random.Random,
    *,
    paper_tints: Mapping[str, float],
    display: bool = False,
    dark_mode: bool = False,
) -> Palette:
    """Sample a legible palette.

    The ink is drawn as a hue-and-lightness perturbation of near-black rather
    than from a fixed list, so the corpus contains the warm greys, blue-blacks
    and faded browns that real printing produces - and then it is darkened
    until the contrast floor is met, so widening the distribution never
    produces an unreadable page.

    Args:
        rng: The sample's seeded RNG.
        paper_tints: Relative weights over paper colours, from the config.
        display: Whether this is large display text, which permits lower
            contrast.
        dark_mode: Whether to invert for a dark-theme screenshot.

    Returns:
        The palette.
    """
    from ...dataset.common.io_utils import weighted_choice

    floor = _CONTRAST_FLOOR_DISPLAY if display else _CONTRAST_FLOOR

    if dark_mode:
        # Dark chat and dashboard themes. Near-black backgrounds with a slight
        # hue, because a real app's dark mode is rarely pure #000.
        base = rng.choice(["#0f1114", "#121417", "#16181d", "#1b1d22", "#101418"])
        paper = _shift(base, lightness=rng.uniform(-0.02, 0.03))
        ink = _shift("#e8eaed", lightness=rng.uniform(-0.08, 0.04))
        accent = rgb_to_hex(
            [
                round(channel * 255)
                for channel in colorsys.hls_to_rgb(
                    rng.random(), rng.uniform(0.55, 0.72), rng.uniform(0.45, 0.85)
                )
            ]
        )
        muted = _shift(ink, lightness=-rng.uniform(0.18, 0.30))
        panel = _shift(paper, lightness=rng.uniform(0.03, 0.08))
        return Palette(paper, ink, accent, muted, panel, contrast_ratio(ink, paper))

    paper = weighted_choice(tuple(paper_tints.items()), rng)
    # Ink: near-black with a hue cast and occasional faded density.
    hue = rng.random()
    lightness = rng.uniform(0.04, 0.30)
    saturation = rng.uniform(0.0, 0.35)
    ink = rgb_to_hex(
        [
            round(channel * 255)
            for channel in colorsys.hls_to_rgb(hue, lightness, saturation)
        ]
    )

    # Darken until legible. Widening the ink distribution is only safe if the
    # floor is enforced afterwards.
    for _ in range(_MAX_DARKEN):
        if contrast_ratio(ink, paper) >= floor:
            break
        ink = _shift(ink, lightness=-_DARKEN_STEP)
    else:
        ink = "#1a1a1a"

    accent_hue = (hue + rng.uniform(0.08, 0.92)) % 1.0
    accent = rgb_to_hex(
        [
            round(channel * 255)
            for channel in colorsys.hls_to_rgb(
                accent_hue, rng.uniform(0.22, 0.42), rng.uniform(0.35, 0.80)
            )
        ]
    )
    muted = _shift(ink, lightness=rng.uniform(0.18, 0.34), saturation=-0.05)
    # Captions, timestamps and footers use `muted`, so it needs a floor of its
    # own. Lightening the ink by up to a third breached legibility and image QC
    # rejected the captions outright.
    for _ in range(_MAX_DARKEN):
        if contrast_ratio(muted, paper) >= _CONTRAST_FLOOR_DISPLAY:
            break
        muted = _shift(muted, lightness=-_DARKEN_STEP)
    panel = _shift(paper, lightness=-rng.uniform(0.03, 0.07))
    return Palette(paper, ink, accent, muted, panel, contrast_ratio(ink, paper))


@dataclass(frozen=True, slots=True)
class RenderSpec:
    """Everything stage 05 needs to draw one sample.

    Attributes:
        sample_id: Identifier of the image to produce.
        group_id: Splitting key shared with the sample's crops.
        category: Visual category, also the Hub config name.
        template_id: Which template produced the HTML. Recorded because
            ``test_unseen_templates`` holds out whole templates, and the split
            cannot be built without knowing which one each sample used.
        html: The complete document, fonts already embedded.
        width: Viewport width in CSS pixels.
        height: Viewport height in CSS pixels, or ``0`` to fit the content.
        dpi: Target resolution; the device scale factor is ``dpi / 96``.
        font_ids: Faces used, for the manifest and the licence report.
        font_families: Their CSS family names.
        palette: The colours used.
        clean: Whether this sample skips degradation entirely.
        dark_mode: Whether the template rendered a dark theme.
        aug: The degradation plan, as a JSON-serialisable mapping. Opaque here;
            interpreted by :mod:`fa_ocr_lora.diverse.render.degrade`.
        text_render: The exact label, carried through so stage 05 can verify
            the harvest reproduces it.
        seed: Per-sample seed, so rendering is reproducible in isolation.
    """

    sample_id: str
    group_id: str
    category: str
    template_id: str
    html: str
    width: int
    height: int
    dpi: int
    font_ids: tuple[str, ...]
    font_families: tuple[str, ...]
    palette: Palette
    clean: bool
    dark_mode: bool
    aug: Mapping[str, Any]
    text_render: str
    seed: int

    @property
    def scale(self) -> float:
        """Return the device scale factor Chromium should use."""
        return self.dpi / 96.0

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form written to ``layouts.jsonl``.

        The HTML is included: it is large, but it makes the render stage a pure
        function of this one file, which is what lets the molab runtime render
        without the lexicon, the fonts or the config.
        """
        return {
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "category": self.category,
            "template_id": self.template_id,
            "html": self.html,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "font_ids": list(self.font_ids),
            "font_families": list(self.font_families),
            "palette": self.palette.as_dict(),
            "clean": self.clean,
            "dark_mode": self.dark_mode,
            "aug": dict(self.aug),
            "text_render": self.text_render,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RenderSpec:
        """Rebuild from the JSON form.

        Args:
            payload: A mapping as produced by :meth:`as_dict`.

        Returns:
            The spec.
        """
        colours = payload["palette"]
        return cls(
            sample_id=str(payload["sample_id"]),
            group_id=str(payload["group_id"]),
            category=str(payload["category"]),
            template_id=str(payload["template_id"]),
            html=str(payload["html"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            dpi=int(payload["dpi"]),
            font_ids=tuple(payload.get("font_ids", ())),
            font_families=tuple(payload.get("font_families", ())),
            palette=Palette(**colours),
            clean=bool(payload["clean"]),
            dark_mode=bool(payload.get("dark_mode", False)),
            aug=dict(payload.get("aug", {})),
            text_render=str(payload["text_render"]),
            seed=int(payload["seed"]),
        )
