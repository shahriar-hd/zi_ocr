"""Per-sample visual theme: palette, typography, and rule weights.

Every letter in the corpus should look like it came from a different
organisation, but never like a design accident. This module is the single
place that turns a row's ``brand_hex`` plus the sample's seeded RNG into a
coordinated set of colours and type choices, so the composer and the CSS
template never invent a colour of their own.

Two constraints shape the palette:

* **Light surfaces, dark ink.** The user asked for tinted backgrounds with
  dark text rather than the reverse, which is also what Persian
  administrative letterheads actually do.
* **A measured contrast floor.** Body ink against its surface is checked
  against the WCAG relative-luminance ratio from
  ``theme.min_contrast_ratio`` and darkened until it passes, so no seed can
  produce an unreadable - and therefore untrainable - page.

Fonts are sampled from the weighted pools in ``synth_config.yaml``. Each
family carries an optical-scale factor because the legacy B-series faces draw
noticeably smaller than IRANSansX at the same point size; without the
correction, font choice would silently change the body text's rendered size.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final, Sequence

from ....config import BAR_STYLES, FontSpec, ThemeConfig
from ...common.io_utils import weighted_choice

__all__ = [
    "BarStyle",
    "Palette",
    "Typography",
    "Theme",
    "build_theme",
    "composite",
    "contrast_ratio",
    "hex_to_rgb",
    "rgba",
]

#: How the header and footer bars are drawn. The composer maps these to CSS
#: classes; the annotation records the name so a consumer can tell a solid
#: block apart from a hairline rule. The canonical list lives in
#: :data:`fa_ocr_lora.config.BAR_STYLES` so ``theme.bar_weights`` can be
#: validated against it at load time; these attributes are the readable
#: spellings for use at a call site.
class BarStyle:
    """Header/footer bar treatments."""

    SOLID: Final[str] = "solid"
    TINTED_BAND: Final[str] = "tinted_band"
    RULE: Final[str] = "rule"
    DOUBLE_RULE: Final[str] = "double_rule"
    SIDE_STRIPE: Final[str] = "side_stripe"
    NONE: Final[str] = "none"
    ALL: Final[tuple[str, ...]] = BAR_STYLES


_SRGB_THRESHOLD: Final[float] = 0.04045
_SRGB_DIVISOR: Final[float] = 12.92
_SRGB_OFFSET: Final[float] = 0.055
_SRGB_SCALE: Final[float] = 1.055
_SRGB_EXPONENT: Final[float] = 2.4

#: Luminance coefficients from WCAG 2.1 / Rec. 709.
_LUMA_R: Final[float] = 0.2126
_LUMA_G: Final[float] = 0.7152
_LUMA_B: Final[float] = 0.0722

#: Last resort when even full darkening cannot clear the contrast floor.
#: Intentionally not a knob: it is the value that makes the floor a guarantee
#: rather than a preference, and a config file able to raise it could ship an
#: unreadable page.
_INK_FLOOR: Final[str] = "#111111"


def hex_to_rgb(hex_colour: str) -> tuple[int, int, int]:
    """Split a hex colour into 0-255 channels.

    Args:
        hex_colour: ``#rrggbb`` or the ``#rgb`` shorthand, ``#`` optional.

    Returns:
        The three channels.

    Raises:
        ValueError: If the string is not a 3- or 6-digit hex colour.
    """
    value = hex_colour.lstrip("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    if len(value) != 6:
        raise ValueError(f"not a hex colour: {hex_colour!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _to_hex(rgb: Sequence[float]) -> str:
    """Join channels back into an uppercase ``#rrggbb`` string."""
    clamped = (max(0, min(255, round(channel))) for channel in rgb)
    return "#" + "".join(f"{channel:02X}" for channel in clamped)


def _relative_luminance(hex_colour: str) -> float:
    """Return the WCAG relative luminance of a colour.

    Args:
        hex_colour: A ``#rrggbb`` colour.

    Returns:
        Luminance in ``[0, 1]``.
    """
    channels: list[float] = []
    for raw in hex_to_rgb(hex_colour):
        proportion = raw / 255.0
        if proportion <= _SRGB_THRESHOLD:
            channels.append(proportion / _SRGB_DIVISOR)
        else:
            channels.append(
                ((proportion + _SRGB_OFFSET) / _SRGB_SCALE) ** _SRGB_EXPONENT
            )
    return _LUMA_R * channels[0] + _LUMA_G * channels[1] + _LUMA_B * channels[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG contrast ratio between two colours.

    Args:
        foreground: Text colour as ``#rrggbb``.
        background: Surface colour as ``#rrggbb``.

    Returns:
        A ratio in ``[1, 21]``, where higher is more legible.
    """
    first = _relative_luminance(foreground)
    second = _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _mix(first: str, second: str, weight: float) -> str:
    """Blend two colours.

    Args:
        first: Colour at ``weight == 0``.
        second: Colour at ``weight == 1``.
        weight: Blend position in ``[0, 1]``.

    Returns:
        The blended ``#rrggbb`` colour.
    """
    start, end = hex_to_rgb(first), hex_to_rgb(second)
    return _to_hex(
        [
            start[index] + (end[index] - start[index]) * weight
            for index in range(3)
        ]
    )


def _darken(hex_colour: str, amount: float) -> str:
    """Move a colour toward black by ``amount`` in ``[0, 1]``."""
    return _mix(hex_colour, "#000000", amount)


def _lighten(hex_colour: str, amount: float) -> str:
    """Move a colour toward white by ``amount`` in ``[0, 1]``."""
    return _mix(hex_colour, "#FFFFFF", amount)


def _ensure_contrast(
    foreground: str,
    background: str,
    minimum: float,
    config: ThemeConfig,
) -> str:
    """Darken a colour until it clears a contrast floor against its surface.

    A light brand colour used directly as text can fall below the readable
    threshold. Darkening preserves the hue - and so the brand identity - while
    the ratio climbs, which recolouring to black would not.

    Args:
        foreground: Desired text colour.
        background: The surface behind it.
        minimum: Required contrast ratio.
        config: Theme constraints, for the darkening step and its attempt cap.

    Returns:
        The original colour when it already passes, else a darkened variant.
        Falls back to near-black if even full darkening cannot pass.
    """
    return _ensure_contrast_over(foreground, background, 1.0, minimum, config)


def _ensure_contrast_over(
    foreground: str,
    background: str,
    alpha: float,
    minimum: float,
    config: ThemeConfig,
) -> str:
    """Darken a colour until its *composite* clears a contrast floor.

    Ink is painted with an alpha (§ :attr:`Palette.ink_alpha`) so the paper
    grain beneath shows through it, which lightens what the eye - and the
    model - actually sees. The floor therefore has to be measured on the
    composite rather than on the nominal colour, or a translucent ink could
    pass the check and still render unreadably pale.

    Args:
        foreground: Desired text colour, before compositing.
        background: The surface behind it.
        alpha: Opacity the colour will be painted with, in ``[0, 1]``.
        minimum: Required contrast ratio.
        config: Theme constraints, for the darkening step and its attempt cap.

    Returns:
        The base colour to paint at ``alpha``. Falls back to near-black when
        even full darkening cannot pass.
    """
    candidate = foreground
    for _ in range(config.max_darken_steps):
        if contrast_ratio(composite(background, candidate, alpha), background) >= minimum:
            return candidate
        candidate = _darken(candidate, config.darken_step)
    return _INK_FLOOR


def composite(background: str, foreground: str, alpha: float) -> str:
    """Return the colour a translucent foreground actually shows as.

    Every ink on the page is painted with an alpha, so its nominal hex is not
    what the eye or the model receives. Contrast checks - here and in the tests -
    have to be made against this, the composited result.

    Args:
        background: Colour behind the paint, as ``#rrggbb``.
        foreground: Colour being painted, as ``#rrggbb``.
        alpha: Opacity it is painted with; values outside ``[0, 1]`` are clamped.

    Returns:
        The resulting ``#rrggbb`` colour.
    """
    return _mix(background, foreground, max(0.0, min(1.0, alpha)))


def rgba(hex_colour: str, alpha: float) -> str:
    """Return a CSS ``rgba()`` string for a hex colour and an opacity.

    Every light colour on the page is expressed this way rather than as an
    opaque hex, so "white" is an absence of ink - transparent - instead of a
    sprayed-on layer that would hide the paper texture composited underneath.

    Args:
        hex_colour: A ``#rrggbb`` colour.
        alpha: Opacity in ``[0, 1]``, rounded to three decimals.

    Returns:
        e.g. ``rgba(26, 26, 26, 0.94)``.
    """
    red, green, blue = hex_to_rgb(hex_colour)
    return f"rgba({red}, {green}, {blue}, {max(0.0, min(1.0, alpha)):.3f})"


@dataclass(frozen=True, slots=True)
class Palette:
    """The colours one document is allowed to use.

    Colours are stored as opaque hex - the contrast maths needs a definite
    value - and *painted* through the ``*_css`` properties, which attach the
    matching alpha. Nothing on the page is fully opaque white: light areas are
    transparent, so the paper tone and the scanned grain composited beneath the
    sheet remain visible through them.

    Attributes:
        brand: The row's own brand colour, unmodified.
        accent: Brand colour darkened enough to read as text and chart ink.
        ink: Body text colour.
        muted_ink: Secondary text - footer fields, page numbers, captions.
        surface: The page behind everything.
        tint: A pale brand wash for banded bars and table header rows.
        rule: Hairline colour for borders and separators.
        bar_text: Text colour that sits on a solid brand bar.
        surface_alpha: Opacity of the paper wash. Deliberately low: the sheet
            is a tint over the scanned paper, not a coat of paint over it.
        ink_alpha: Opacity of text. Toner is never fully opaque, and the
            residual transparency is what lets grain break up the glyph edges.
        tint_alpha: Opacity of pale washes - banded bars, zebra rows.
        panel_alpha: Opacity of saturated fills - solid bars, side stripes.
        rule_alpha: Opacity of hairlines.
    """

    brand: str
    accent: str
    ink: str
    muted_ink: str
    surface: str
    tint: str
    rule: str
    bar_text: str
    surface_alpha: float = 0.24
    ink_alpha: float = 0.94
    tint_alpha: float = 0.6
    panel_alpha: float = 0.95
    rule_alpha: float = 0.82

    @property
    def surface_css(self) -> str:
        """Paper wash for the sheet, translucent over the scanned paper."""
        return rgba(self.surface, self.surface_alpha)

    @property
    def ink_css(self) -> str:
        """Body text colour."""
        return rgba(self.ink, self.ink_alpha)

    @property
    def muted_ink_css(self) -> str:
        """Secondary text colour."""
        return rgba(self.muted_ink, self.ink_alpha)

    @property
    def accent_css(self) -> str:
        """Brand-coloured text and chart ink."""
        return rgba(self.accent, self.ink_alpha)

    @property
    def brand_css(self) -> str:
        """Saturated fill for solid bars and stripes."""
        return rgba(self.brand, self.panel_alpha)

    @property
    def accent_panel_css(self) -> str:
        """Saturated accent fill, for the full-height side stripe."""
        return rgba(self.accent, self.panel_alpha)

    @property
    def tint_css(self) -> str:
        """Pale brand wash for banded bars and table header rows."""
        return rgba(self.tint, self.tint_alpha)

    @property
    def tint_soft_css(self) -> str:
        """Half-strength wash, for zebra-striped table body rows."""
        return rgba(self.tint, self.tint_alpha * 0.5)

    @property
    def rule_css(self) -> str:
        """Hairline colour for borders and separators."""
        return rgba(self.rule, self.rule_alpha)

    @property
    def bar_text_css(self) -> str:
        """Text sitting on a solid brand bar."""
        return rgba(self.bar_text, self.ink_alpha)

    def as_content(self) -> dict[str, object]:
        """Return the palette for the annotation's provenance block."""
        return {
            "brand": self.brand,
            "accent": self.accent,
            "ink": self.ink,
            "surface": self.surface,
            "tint": self.tint,
            "surface_alpha": round(self.surface_alpha, 3),
            "ink_alpha": round(self.ink_alpha, 3),
        }


@dataclass(frozen=True, slots=True)
class Typography:
    """The type choices one document is allowed to use.

    Attributes:
        body_family: CSS family name for running text.
        display_family: CSS family name for the subject line and headings.
        body_pt: Body point size, already optical-scale corrected.
        line_height: Unitless CSS ``line-height`` for body text.
        heading_scale: Multiplier applied to ``body_pt`` for headings.
        subject_scale: Multiplier applied to ``body_pt`` for the subject line.
        justify: Whether body paragraphs are justified rather than ragged.
    """

    body_family: str
    display_family: str
    body_pt: float
    line_height: float
    heading_scale: float
    subject_scale: float
    justify: bool

    @property
    def body_css(self) -> str:
        """Return the body ``font-family`` stack, with an Arabic fallback."""
        return f'"{self.body_family}", "Noto Naskh Arabic", serif'

    @property
    def display_css(self) -> str:
        """Return the display ``font-family`` stack."""
        return f'"{self.display_family}", "Noto Naskh Arabic", serif'

    @property
    def heading_pt(self) -> float:
        """Point size for section headings."""
        return round(self.body_pt * self.heading_scale, 2)

    @property
    def subject_pt(self) -> float:
        """Point size for the subject line."""
        return round(self.body_pt * self.subject_scale, 2)

    def as_content(self) -> dict[str, object]:
        """Return the type choices for the annotation's provenance block."""
        return {
            "body_family": self.body_family,
            "display_family": self.display_family,
            "body_pt": self.body_pt,
            "line_height": self.line_height,
            "justify": self.justify,
        }


@dataclass(frozen=True, slots=True)
class Theme:
    """Everything visual that one document commits to up front.

    Attributes:
        palette: The document's colours.
        typography: The document's type choices.
        header_bar: Bar treatment above the letter.
        footer_bar: Bar treatment below the letter.
        rule_width_px: Hairline width for rules and table borders.
        corner_radius_px: Border radius on tables, bands, and stamps.
        table_zebra: Whether table body rows alternate shading.
    """

    palette: Palette
    typography: Typography
    header_bar: str
    footer_bar: str
    rule_width_px: float
    corner_radius_px: float
    table_zebra: bool

    def as_content(self) -> dict[str, object]:
        """Return the whole theme for the annotation's provenance block."""
        return {
            "palette": self.palette.as_content(),
            "typography": self.typography.as_content(),
            "header_bar": self.header_bar,
            "footer_bar": self.footer_bar,
        }


def _pick_font(pool: Sequence[FontSpec], rng: random.Random) -> FontSpec:
    """Pick one family from a weighted pool.

    Args:
        pool: Candidate families with their weights.
        rng: The sample's seeded RNG.

    Returns:
        The chosen spec.
    """
    total = sum(spec.weight for spec in pool)
    target = rng.random() * total
    cumulative = 0.0
    for spec in pool:
        cumulative += spec.weight
        if target < cumulative:
            return spec
    return pool[-1]


def _build_palette(
    brand_hex: str,
    config: ThemeConfig,
    rng: random.Random,
) -> Palette:
    """Derive a coordinated palette from one brand colour.

    Args:
        brand_hex: The row's brand colour, already normalised to ``#rrggbb``.
        config: Theme constraints, for the contrast floor.
        rng: The sample's seeded RNG.

    Returns:
        A palette whose body ink clears the configured contrast floor once
        composited at its own alpha.
    """
    surface = rng.choice(config.surface_candidates)
    # Alphas first, because the contrast floor is measured on the composite.
    surface_alpha = round(config.surface_alpha.draw(rng), 3)
    ink_alpha = round(config.ink_alpha.draw(rng), 3)
    ink = _ensure_contrast_over(
        rng.choice(config.ink_candidates),
        surface,
        ink_alpha,
        config.min_contrast_ratio,
        config,
    )
    accent = _ensure_contrast_over(
        brand_hex, surface, ink_alpha, config.min_contrast_ratio, config
    )
    muted_ink = _ensure_contrast_over(
        _lighten(ink, config.muted_ink_lighten.draw(rng)),
        surface,
        ink_alpha,
        config.muted_ink_contrast_ratio,
        config,
    )
    tint = _lighten(brand_hex, config.tint_lighten.draw(rng))
    rule = _lighten(accent, config.rule_lighten.draw(rng))
    # Text printed on a solid brand bar needs the bar as its background, not
    # the page, so it gets its own contrast decision.
    bar_text = (
        config.bar_text_light
        if contrast_ratio(config.bar_text_light, brand_hex) >= config.min_contrast_ratio
        else _ensure_contrast(
            config.bar_text_dark, brand_hex, config.min_contrast_ratio, config
        )
    )
    return Palette(
        brand=brand_hex,
        accent=accent,
        ink=ink,
        muted_ink=muted_ink,
        surface=surface,
        tint=tint,
        rule=rule,
        bar_text=bar_text,
        surface_alpha=surface_alpha,
        ink_alpha=ink_alpha,
        tint_alpha=round(config.tint_alpha.draw(rng), 3),
        panel_alpha=round(config.panel_alpha.draw(rng), 3),
        rule_alpha=round(config.rule_alpha.draw(rng), 3),
    )


def build_theme(
    brand_hex: str,
    config: ThemeConfig,
    rng: random.Random,
) -> Theme:
    """Derive the whole visual theme for one document.

    Args:
        brand_hex: The row's brand colour as ``#rrggbb``.
        config: Theme constraints from ``synth_config.yaml``.
        rng: The sample's seeded RNG, conventionally ``purpose="theme"``.

    Returns:
        A theme whose body ink is guaranteed to clear the contrast floor.
    """
    palette = _build_palette(brand_hex, config, rng)

    body_font = _pick_font(config.font_families, rng)
    display_font = _pick_font(config.display_font_families, rng)
    minimum, maximum = config.body_font_pt_range
    body_pt = round(rng.uniform(minimum, maximum) * body_font.scale, 2)
    line_minimum, line_maximum = config.line_height_range
    typography = Typography(
        body_family=body_font.name,
        # An empty display family means "reuse the body face", which is what a
        # plain administrative letter does.
        display_family=display_font.name or body_font.name,
        body_pt=body_pt,
        line_height=round(rng.uniform(line_minimum, line_maximum), 2),
        heading_scale=round(config.heading_scale.draw(rng), 3),
        subject_scale=round(config.subject_scale.draw(rng), 3),
        justify=rng.random() < config.justify_probability,
    )

    header_bar = weighted_choice(config.bar_weights, rng)
    # A side stripe runs the full page height, so it cannot also be a footer
    # treatment; pairing it with itself would double the stripe.
    footer_options = tuple(
        (style, weight)
        for style, weight in config.bar_weights
        if style != BarStyle.SIDE_STRIPE
    )
    footer_bar = weighted_choice(footer_options, rng)

    return Theme(
        palette=palette,
        typography=typography,
        header_bar=header_bar,
        footer_bar=footer_bar,
        rule_width_px=round(config.rule_width_px.draw(rng), 2),
        corner_radius_px=round(rng.choice(config.corner_radii_px), 1),
        table_zebra=rng.random() < config.table_zebra_probability,
    )
