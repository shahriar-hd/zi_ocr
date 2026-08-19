"""The rubber stamp (مهر) composited beside the signature.

An Iranian administrative letter is not official until it carries a stamp, so
roughly half the signed samples get one: the company's own logo inside a
geometric border, ringed with Persian text, printed in a single ink.

**The stamp is inline SVG, not a Pillow PNG.** The plan called for a PNG, but
the ring text settled it: Persian on a curved path needs per-glyph rotation,
and rotating glyphs individually in Pillow breaks the cursive joining that
makes Arabic script legible - ``شرکت`` would come apart into four unconnected
letters. Chromium shapes text along an SVG ``<textPath>`` correctly for free,
which is the same reason :mod:`..elements.charts` emits SVG rather than
matplotlib. Everything a raster stamp would have given is still here:

* **Uneven ink** comes from an SVG filter, not from pixel noise. A
  ``feTurbulence`` field displaces the whole mark by a pixel or two, so the
  border wobbles the way a hand-pressed rubber edge does, and a second, finer
  turbulence is thresholded into a hole mask that eats patches out of the ink.
* **Rotation and opacity** are a group ``transform`` and ``opacity``, applied
  per sample so no two stamps land at the same angle or the same weight.
* **Resolution is free.** The mark stays crisp at 300 or 600 DPI instead of
  being resampled from a fixed-size bitmap.

The logo goes in as a nested ``<image>`` with its own ``data:`` URI, the same
sealed-rendering-context argument as :mod:`.header_footer` - a logo carrying
its own ``<style>`` block cannot restyle the stamp around it.
"""

from __future__ import annotations

import html
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from ....config import STAMP_ROUND_SHAPES, StampConfig, StampVocabulary
from ...common.io_utils import weighted_choice
from .header_footer import logo_data_uri

__all__ = [
    "StampSpec",
    "build_stamp",
    "stamp_to_html",
    "city_from_address",
]

#: Fallback for a caller that does not hand over the loaded config - the test
#: suite, mostly. Read off an instance rather than restated here, so a literal
#: cannot drift from the dataclass default.
_DEFAULT_STAMP: Final[StampConfig] = StampConfig()

#: Border shapes, their weights, and every per-sample range the mark varies
#: over now live in the ``stamp`` config section. Shapes whose ring text
#: follows an arc come from :data:`~fa_ocr_lora.config.STAMP_ROUND_SHAPES`:
#: that is a fact about each shape, not a choice, since text curved around a
#: corner is not a thing real stamps do.

#: The cities the lower ring can name and the words printed beside them live in
#: ``configs/vocabulary_fa.yaml`` and arrive as ``config.vocabulary.stamp``. The
#: city list is read twice: once to sample a fallback, and once by
#: :func:`city_from_address` to recognise a city the footer already names.
_DEFAULT_VOCABULARY: Final[StampVocabulary] = StampVocabulary()

#: The stamp's own coordinate system. Every geometry constant below is in these
#: units, so the mark scales by changing only the CSS width. Intentionally not
#: a knob: these numbers are meaningful only relative to each other.
_BOX: Final[int] = 200
_CENTRE: Final[float] = 100.0

#: Outer border radius, and the radii the two text rings sit on.
_OUTER_R: Final[float] = 94.0
_INNER_R: Final[float] = 84.0
_TOP_TEXT_R: Final[float] = 76.0
_BOTTOM_TEXT_R: Final[float] = 74.0

#: Ring text size in viewBox units. The logo's share of the box is a knob and
#: lives in ``stamp.logo_fraction``; these two are part of the frame above.
_RING_FONT: Final[float] = 15.0
_LINE_FONT: Final[float] = 16.0

#: Alpha row of the logo density matrix: ``A - luminance(RGB)`` with the
#: Rec. 709 coefficients. Dark marks ink fully, white knockouts drop out, and
#: anything already transparent stays transparent - which is what turns an
#: arbitrary brand logo into a one-colour seal impression. Intentionally not a
#: knob: it is the definition of relative luminance, not a preference.
_DENSITY_ROW: Final[str] = "-0.2126 -0.7152 -0.0722 1 0"

#: Below this the ring text stops being legible at 300 DPI, so a name too long
#: to fit is truncated rather than shrunk further.
_MIN_RING_FONT: Final[float] = 8.5

#: Mean glyph advance as a fraction of the em, used to estimate a line's width
#: without measuring it. Persian runs narrower than this, but the logo titles
#: that feed the upper ring are often Latin (``Managed Service For Microsoft``)
#: and Latin capitals are wider, so the estimate is set for the wider case -
#: overestimating shrinks the text slightly, underestimating runs it into the
#: border.
_GLYPH_ADVANCE: Final[float] = 0.56

#: Arc length a ring line may occupy, in viewBox units. Past this the text
#: creeps down the sides of the seal instead of sitting along the top.
_RING_SPAN: Final[float] = 150.0

#: Longest ring line kept intact. The corpus has names up to 24 characters
#: (``SSIS Lift And Shift IR``), which at the minimum size still overruns the
#: border.
_MAX_RING_CHARS: Final[int] = 30


def _fit_font_size(text: str, base: float) -> float:
    """Shrink a ring line until it fits inside :data:`_RING_SPAN`.

    Args:
        text: The line to be drawn.
        base: Preferred font size in viewBox units.

    Returns:
        A size in ``[_MIN_RING_FONT, base]``.
    """
    if not text:
        return base
    estimated = len(text) * base * _GLYPH_ADVANCE
    if estimated <= _RING_SPAN:
        return base
    return round(max(_MIN_RING_FONT, base * _RING_SPAN / estimated), 2)


def _clip_ring(text: str) -> str:
    """Cut a ring line down to what :data:`_MAX_RING_CHARS` allows.

    Shrinking alone is not enough past a certain length: at
    :data:`_MIN_RING_FONT` the glyphs are already at the legibility floor for
    300 DPI, so a name longer than the ring can hold is cut instead. The cut
    prefers the last word boundary, because a seal reading ``شرکت فناوری`` is
    plausible where one reading ``شرکت فناور`` is not.

    Args:
        text: The line as designed.

    Returns:
        The line, at most :data:`_MAX_RING_CHARS` characters long.
    """
    stripped = text.strip()
    if len(stripped) <= _MAX_RING_CHARS:
        return stripped
    head = stripped[:_MAX_RING_CHARS]
    boundary = head.rfind(" ")
    # Only honour a boundary that leaves most of the line intact; a space near
    # the start would throw away more of the name than the overflow did.
    if boundary >= _MAX_RING_CHARS // 2:
        return head[:boundary].strip()
    return head.strip()


def city_from_address(
    address: str, vocabulary: StampVocabulary = _DEFAULT_VOCABULARY
) -> str:
    """Find the city named in a postal address, if it names one.

    The generated addresses mention a city only about half the time
    (``خیابان آزادی، پلاک ...، تهران، کدپستی ...`` versus the same line without
    it), so the caller needs to know whether to sample one instead of printing
    a stamp that contradicts the footer.

    Args:
        address: The letter's ``contact_info.address``.
        vocabulary: The ``stamp`` pools from ``vocabulary_fa.yaml``; only
            ``cities`` is read here.

    Returns:
        The matched city, or ``""`` when the address names none.
    """
    for city in vocabulary.cities:
        if city in address:
            return city
    return ""


@dataclass(frozen=True, slots=True)
class StampSpec:
    """One letter's seal, resolved to a self-contained SVG.

    Attributes:
        logo_uri: The company mark as a ``data:`` URI, recoloured to the stamp
            ink, or ``""`` when the row has no usable logo - the seal then
            prints as text and border alone, which real seals also do.
        logo_name: Source filename, for provenance.
        shape: Border treatment, one of
            :data:`~fa_ocr_lora.config.STAMP_SHAPES`.
        ink: The single colour everything is printed in.
        top_text: Upper ring, normally the organisation name.
        bottom_text: Lower ring, normally city plus a seal word.
        width_mm: Rendered width on the page; the mark is square.
        rotate_deg: Tilt of the whole mark.
        opacity: Overall ink weight.
        wobble: Displacement amplitude in viewBox units.
        erosion: Hole-mask intercept; more negative eats more ink.
        noise_seed: Seed handed to ``feTurbulence``, so a sample's grain is
            reproducible rather than up to the renderer.
        stroke_width: Border weight in viewBox units.
        logo_fraction: The logo's share of the box's width.
        ink_gain: Gain applied to the logo's density before it is flooded with
            ink; see :func:`_ink_filter_svg`.

    The last two are carried on the spec rather than read from the config where
    they are used, because :func:`stamp_to_html` and :func:`_ink_filter_svg`
    are handed only a spec. Both keep the shipped default so an existing
    construction stays valid.
    """

    logo_uri: str
    logo_name: str
    shape: str
    ink: str
    top_text: str
    bottom_text: str
    width_mm: float
    rotate_deg: float
    opacity: float
    wobble: float
    erosion: float
    noise_seed: int
    stroke_width: float
    logo_fraction: float = _DEFAULT_STAMP.logo_fraction
    ink_gain: float = _DEFAULT_STAMP.ink_gain

    @property
    def is_round(self) -> bool:
        """Whether the ring text follows an arc rather than a straight line."""
        return self.shape in STAMP_ROUND_SHAPES

    @property
    def text(self) -> str:
        """The seal's text in reading order, for the transcription."""
        return "\n".join(line for line in (self.top_text, self.bottom_text) if line)

    def as_content(self) -> dict[str, object]:
        """Return the seal's structured payload for the annotation."""
        return {
            "shape": self.shape,
            "ink": self.ink,
            "top_text": self.top_text,
            "bottom_text": self.bottom_text,
            "logo": self.logo_name,
            "width_mm": self.width_mm,
            "rotate_deg": self.rotate_deg,
        }


def build_stamp(
    logo_path: Path | None,
    company_name: str,
    city: str,
    ink_pool: Sequence[str],
    rng: random.Random,
    probability: float = 0.55,
    config: StampConfig = _DEFAULT_STAMP,
    vocabulary: StampVocabulary = _DEFAULT_VOCABULARY,
) -> StampSpec | None:
    """Design the seal for one letter, or none.

    Args:
        logo_path: The row's logo SVG, or ``None``.
        company_name: Organisation name for the upper ring.
        city: City for the lower ring; one is sampled when this is empty.
        ink_pool: Candidate stamp inks from ``theme.stamp_inks``.
        rng: The sample's seeded RNG, conventionally ``purpose="stamp"``.
        probability: Chance the letter is stamped at all.
        config: The ``stamp`` config section - shape pool, per-sample ranges,
            and the two values carried onto the spec for rendering.
        vocabulary: The ``stamp`` pools from ``vocabulary_fa.yaml``: the seal
            words and the fallback city list.

    Returns:
        The designed seal, or ``None`` when this letter carries no stamp.
    """
    # Drawn unconditionally, as in texture.py and signature.py, so retuning the
    # stamp rate does not shift every later choice in the sample.
    draw = rng.random()
    shape = weighted_choice(config.shapes, rng)
    ink = ink_pool[rng.randrange(len(ink_pool))] if ink_pool else "#1B3F94"
    seal_word = weighted_choice(vocabulary.seal_words, rng)
    cities = vocabulary.cities
    fallback_city = cities[rng.randrange(len(cities))]
    width_mm = round(config.width_mm.draw(rng), 1)
    rotate_deg = round(config.rotate_deg.draw(rng), 2)
    opacity = round(config.opacity.draw(rng), 3)
    wobble = round(config.wobble.draw(rng), 2)
    erosion = round(config.erosion.draw(rng), 2)
    noise_seed = rng.randrange(10_000)
    stroke_width = round(config.stroke_width.draw(rng), 2)
    if draw >= probability:
        return None

    name = company_name.strip()
    prefixed = name and rng.random() < config.name_prefix_probability
    top_text = f"مهر {name}" if prefixed else name
    bottom_parts = [city.strip() or fallback_city]
    if seal_word:
        bottom_parts.append(seal_word)

    return StampSpec(
        # Forced to flat black, not to the ink: rewriting the root fill only
        # reaches a logo that has no fills of its own, and 2 of every 5 in the
        # corpus colour their own paths. The ink is applied downstream by the
        # density filter in :func:`stamp_to_html`, which recolours every mark
        # the same way - a seal is pressed with one pad, so a brand-coloured
        # logo inside a monochrome ring would give the trick away.
        logo_uri=logo_data_uri(logo_path, "#000000") if logo_path is not None else "",
        logo_name=logo_path.name if logo_path is not None else "",
        shape=shape,
        ink=ink,
        top_text=_clip_ring(top_text),
        bottom_text=_clip_ring(" - ".join(bottom_parts)),
        width_mm=width_mm,
        rotate_deg=rotate_deg,
        opacity=opacity,
        wobble=wobble,
        erosion=erosion,
        noise_seed=noise_seed,
        stroke_width=stroke_width,
        logo_fraction=config.logo_fraction,
        ink_gain=config.ink_gain,
    )


def _ink_filter_svg(spec: StampSpec, uid: str) -> str:
    """Build the filter that reduces any logo to the stamp's single ink.

    ``logo_data_uri`` can only rewrite a root ``fill``, which does nothing for
    the 2-in-5 marks that colour their own paths - those rendered in full
    brand colour inside a monochrome seal. Recolouring has to happen after
    rasterisation, where the mark is just pixels:

    1. ``feColorMatrix`` replaces alpha with ``A - luminance``, so dark ink
       stays opaque, white knockouts and pale tints fall away, and the
       transparent margin is untouched.
    2. ``feComponentTransfer`` multiplies that density by ``spec.ink_gain``,
       because a mid-tone brand colour would otherwise print as a ghost.
    3. ``feFlood`` + ``feComposite operator="in"`` paint the surviving shape
       in the seal's ink.

    Args:
        spec: The designed seal.
        uid: Unique suffix for the filter id.

    Returns:
        A ``<filter>`` element.
    """
    return (
        f'<filter id="stamp-ink-{uid}" x="-5%" y="-5%" width="110%" height="110%"'
        ' color-interpolation-filters="sRGB">'
        '<feColorMatrix type="matrix" values="'
        "0 0 0 0 0 "
        "0 0 0 0 0 "
        "0 0 0 0 0 "
        f'{_DENSITY_ROW}" result="density"/>'
        '<feComponentTransfer in="density" result="inked">'
        f'<feFuncA type="linear" slope="{spec.ink_gain}" intercept="0"/>'
        "</feComponentTransfer>"
        f'<feFlood flood-color="{spec.ink}" result="pad"/>'
        '<feComposite in="pad" in2="inked" operator="in"/>'
        "</filter>"
    )


def _filter_svg(spec: StampSpec, uid: str) -> str:
    """Build the roughening filter that makes the mark read as pressed rubber.

    Two turbulence fields do separate jobs. The coarse one displaces the whole
    mark so the border wobbles; the fine one is thresholded into a mask and
    composited out, which punches the dropouts a dry pad leaves behind.

    Args:
        spec: The designed seal.
        uid: Unique suffix for the filter id.

    Returns:
        A ``<filter>`` element.
    """
    return (
        f'<filter id="stamp-rough-{uid}" x="-15%" y="-15%" width="130%" height="130%">'
        '<feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="3"'
        f' seed="{spec.noise_seed}" result="warp"/>'
        '<feDisplacementMap in="SourceGraphic" in2="warp"'
        f' scale="{spec.wobble}" xChannelSelector="R" yChannelSelector="G"'
        ' result="wobbled"/>'
        '<feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2"'
        f' seed="{spec.noise_seed + 1}" result="grain"/>'
        '<feColorMatrix in="grain" type="luminanceToAlpha" result="grainAlpha"/>'
        '<feComponentTransfer in="grainAlpha" result="holes">'
        f'<feFuncA type="linear" slope="4" intercept="{spec.erosion}"/>'
        "</feComponentTransfer>"
        '<feComposite in="wobbled" in2="holes" operator="out"/>'
        "</filter>"
    )


def _border_svg(spec: StampSpec) -> str:
    """Draw the seal's border for its chosen shape.

    Args:
        spec: The designed seal.

    Returns:
        One or more SVG shape elements, unfilled and stroked in the ink.
    """
    stroke = f'fill="none" stroke="{spec.ink}" stroke-width="{spec.stroke_width}"'
    thin = (
        f'fill="none" stroke="{spec.ink}"'
        f' stroke-width="{round(spec.stroke_width * 0.5, 2)}"'
    )
    if spec.shape == "circle":
        return f'<circle cx="{_CENTRE}" cy="{_CENTRE}" r="{_OUTER_R}" {stroke}/>'
    if spec.shape == "double_circle":
        return (
            f'<circle cx="{_CENTRE}" cy="{_CENTRE}" r="{_OUTER_R}" {stroke}/>'
            f'<circle cx="{_CENTRE}" cy="{_CENTRE}" r="{_INNER_R}" {thin}/>'
        )
    if spec.shape == "oval":
        return (
            f'<ellipse cx="{_CENTRE}" cy="{_CENTRE}"'
            f' rx="{_OUTER_R}" ry="{round(_OUTER_R * 0.72, 2)}" {stroke}/>'
        )
    if spec.shape in {"square", "rounded_square"}:
        inset = _CENTRE - _OUTER_R * 0.92
        side = (_CENTRE - inset) * 2
        radius = 12 if spec.shape == "rounded_square" else 0
        return (
            f'<rect x="{round(inset, 2)}" y="{round(inset, 2)}"'
            f' width="{round(side, 2)}" height="{round(side, 2)}"'
            f' rx="{radius}" {stroke}/>'
        )
    # Hexagon, point-up, inscribed in the outer radius.
    points = " ".join(
        f"{round(_CENTRE + _OUTER_R * math.sin(math.radians(angle)), 2)},"
        f"{round(_CENTRE - _OUTER_R * math.cos(math.radians(angle)), 2)}"
        for angle in range(0, 360, 60)
    )
    return f'<polygon points="{points}" {stroke}/>'


def _ring_text_svg(spec: StampSpec, uid: str) -> str:
    """Lay the seal's two text lines out for its shape.

    Round seals carry their text on arcs; angular ones print straight lines
    above and below the logo, because text bent around a corner is not
    something a real seal does.

    Args:
        spec: The designed seal.
        uid: Unique suffix for the path ids the text references.

    Returns:
        The ``<defs>`` paths (if any) and the ``<text>`` elements.
    """
    top = html.escape(spec.top_text)
    bottom = html.escape(spec.bottom_text)
    common = (
        f'fill="{spec.ink}" stroke="none" text-anchor="middle"'
        ' font-weight="bold" direction="rtl"'
    )

    if not spec.is_round:
        # A straight line inside a square has the full inner width to work
        # with, not an arc's usable span, so it fits a little more before it
        # has to shrink.
        return (
            f'<text x="{_CENTRE}" y="{_CENTRE - 52}"'
            f' font-size="{_fit_font_size(spec.top_text, _LINE_FONT)}"'
            f" {common}>{top}</text>"
            f'<text x="{_CENTRE}" y="{_CENTRE + 62}"'
            f' font-size="{_fit_font_size(spec.bottom_text, _LINE_FONT)}"'
            f" {common}>{bottom}</text>"
        )

    radius_y = _TOP_TEXT_R * (0.72 if spec.shape == "oval" else 1.0)
    bottom_y = _BOTTOM_TEXT_R * (0.72 if spec.shape == "oval" else 1.0)
    # The top arc sweeps clockwise from the left point, so its text rides over
    # the crown upright. The bottom arc sweeps *counter*-clockwise from the
    # left point: drawing it clockwise from the right point puts the glyph
    # baselines on the inside of the curve, which renders the lower ring
    # upside down and mirrored - the first version did exactly that.
    top_path = (
        f"M {_CENTRE - _TOP_TEXT_R} {_CENTRE}"
        f" A {_TOP_TEXT_R} {radius_y} 0 0 1 {_CENTRE + _TOP_TEXT_R} {_CENTRE}"
    )
    bottom_path = (
        f"M {_CENTRE - _BOTTOM_TEXT_R} {_CENTRE}"
        f" A {_BOTTOM_TEXT_R} {bottom_y} 0 0 0 {_CENTRE + _BOTTOM_TEXT_R} {_CENTRE}"
    )
    # Both paths run left point -> right point, which is what ``direction=rtl``
    # wants: the bidi pass hands the shaper glyphs in visual order, so laying
    # them along a rightward path prints the first Persian character at the
    # right-hand end, where a reader looks for it.
    return (
        f'<defs><path id="stamp-top-{uid}" d="{top_path}"/>'
        f'<path id="stamp-bottom-{uid}" d="{bottom_path}"/></defs>'
        f'<text font-size="{_fit_font_size(spec.top_text, _RING_FONT)}" {common}>'
        f'<textPath href="#stamp-top-{uid}" startOffset="50%">{top}</textPath></text>'
        f'<text font-size="{_fit_font_size(spec.bottom_text, _RING_FONT)}" {common}>'
        f'<textPath href="#stamp-bottom-{uid}" startOffset="50%">{bottom}</textPath>'
        "</text>"
    )


def stamp_to_html(spec: StampSpec, element_id: str = "stamp") -> str:
    """Render the seal as an annotated inline ``<svg>``.

    Args:
        spec: The designed seal.
        element_id: Annotation id for the stamp region.

    Returns:
        One ``<svg>`` element, self-contained and ready to drop beside the
        signature.
    """
    safe_id = html.escape(element_id, quote=True)
    uid = safe_id.replace(".", "-")
    logo = ""
    ink_filter = ""
    if spec.logo_uri:
        size = _BOX * spec.logo_fraction
        offset = _CENTRE - size / 2
        ink_filter = _ink_filter_svg(spec, uid)
        logo = (
            f'<image href="{spec.logo_uri}" x="{round(offset, 2)}"'
            f' y="{round(offset, 2)}" width="{round(size, 2)}"'
            f' height="{round(size, 2)}" preserveAspectRatio="xMidYMid meet"'
            f' filter="url(#stamp-ink-{uid})"/>'
        )
    return (
        '<svg class="stamp-mark" xmlns="http://www.w3.org/2000/svg"'
        f' viewBox="0 0 {_BOX} {_BOX}"'
        f' style="width:{spec.width_mm}mm;height:{spec.width_mm}mm"'
        ' data-ann-type="stamp"'
        f' data-ann-id="{safe_id}" role="img">'
        f"<defs>{_filter_svg(spec, uid)}{ink_filter}</defs>"
        f'<g filter="url(#stamp-rough-{uid})" opacity="{spec.opacity}"'
        f' transform="rotate({spec.rotate_deg} {_CENTRE} {_CENTRE})">'
        f"{_border_svg(spec)}{logo}{_ring_text_svg(spec, uid)}"
        "</g></svg>"
    )
