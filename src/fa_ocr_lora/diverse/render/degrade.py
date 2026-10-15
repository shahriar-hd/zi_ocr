"""Degradation: paper, show-through, geometry, light and noise.

A clean Chromium raster is a screenshot, and a model trained only on
screenshots fails on the first scanned page it meets. This module turns a
raster into something that plausibly came off a scanner, a photocopier or a
phone camera - and, critically, moves the annotation through the same
transform, so the ground truth stays attached to the ink.

Order matters, and it is the order light actually encounters the page:

    paper tint -> wash -> watermark -> verso show-through -> ink
                       -> geometry -> light -> noise -> encode

Everything before ``ink`` is composited **multiplicatively**. That is not a
shortcut, it is the physics: ink and paper absorb light, they do not add it. A
multiply also solves the problem that made the previous pipeline's
show-through useless - a faint layer blended *over* the page would sit on top
of the text, whereas a multiply darkens only where the layer has ink, leaving
the already-dark recto text visually untouched.

Verso show-through
------------------

Hold a printed sheet up to the light and the other side shows through:
mirrored, blurred, lower in contrast, and tinted by the paper rather than by
the ink. Every duplex scan carries it.

The letters pipeline approximated this by mirroring *the same page*, which is
wrong in a way that matters: a model can learn to read the ghost as a mirrored
copy of what it is already reading, so the effect teaches it nothing. Here the
ghost is **a different sample's rendered text**, supplied by the caller from
the corpus. The model has to learn that faint mirrored strokes are not
content - which is the actual skill. The ghost is never annotated; its source
is recorded in ``aug.show_through.source_id`` so a surprising image can be
explained.
"""

from __future__ import annotations

import colorsys
import io
import logging
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from ..config import DegradationConfig
from .schema import PolygonMapper

__all__ = [
    "DegradeResult",
    "Geometry",
    "apply_degradation",
    "encode_image",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: Resolution at which smooth fields (light, vignette, stains) are built before
#: being scaled up. Light falls off smoothly, so building it small and
#: upsampling is not an approximation worth apologising for - and it is two
#: orders of magnitude cheaper than per-pixel work on a 2500x3500 page.
_FIELD: Final[int] = 64

#: Persian words used as watermarks. Generic administrative vocabulary, so no
#: real organisation is implied.
_WATERMARK_WORDS: Final[tuple[str, ...]] = (
    "رونوشت",
    "محرمانه",
    "نمونه",
    "پیش‌نویس",
    "باطل شد",
    "کپی",
    "اصل",
    "تأییدشده",
)

#: Floor on the mean of the lighting field. Vignette, shadow and low light are
#: each defensible alone and compose into a uniformly grey page; this keeps the
#: local variation while keeping the paper light.
_LIGHT_MEAN_FLOOR: Final[float] = 0.82


def _require_pillow() -> tuple[Any, Any, Any, Any, Any]:
    """Import Pillow, with a clear message when it is missing.

    Returns:
        The modules this file needs.

    Raises:
        RuntimeError: If Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

        return Image, ImageChops, ImageDraw, ImageFilter, ImageOps
    except ImportError as error:  # pragma: no cover - declared dependency
        raise RuntimeError(
            "Pillow is required for degradation: pip install pillow"
        ) from error


class Geometry(PolygonMapper):
    """A rotation-plus-perspective warp, and the matching point mapping.

    Pillow's ``Image.transform`` wants the *inverse* homography (it asks, for
    each destination pixel, where in the source to sample), while annotations
    need the *forward* one. Keeping both in one object is what guarantees the
    boxes and the pixels cannot drift apart - the bug class this whole module
    is arranged to prevent.

    Attributes:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        forward: Row-major 3x3 source-to-destination homography.
    """

    __slots__ = ("width", "height", "forward", "_inverse")

    def __init__(self, width: int, height: int, forward: Sequence[float]) -> None:
        """Store the homography and precompute its inverse.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            forward: Nine coefficients, row-major.
        """
        self.width = width
        self.height = height
        self.forward = tuple(float(value) for value in forward)
        self._inverse = _invert_3x3(self.forward)

    @property
    def is_identity(self) -> bool:
        """Report whether this warp leaves the image unchanged."""
        identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        return all(abs(a - b) < 1e-9 for a, b in zip(self.forward, identity, strict=True))

    def map_point(self, x: float, y: float) -> tuple[float, float]:
        """Map a source point to its destination.

        Args:
            x: Source x in pixels.
            y: Source y in pixels.

        Returns:
            The destination ``(x, y)``.
        """
        a, b, c, d, e, f, g, h, i = self.forward
        denominator = g * x + h * y + i
        if abs(denominator) < 1e-12:
            return x, y
        return (a * x + b * y + c) / denominator, (d * x + e * y + f) / denominator

    def pillow_coefficients(self) -> tuple[float, ...]:
        """Return the eight coefficients Pillow's PERSPECTIVE transform wants.

        Returns:
            The inverse homography, normalised so its last element is one.
        """
        a, b, c, d, e, f, g, h, i = self._inverse
        if abs(i) < 1e-12:
            i = 1e-12
        return (a / i, b / i, c / i, d / i, e / i, f / i, g / i, h / i)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form recorded in ``aug``."""
        return {"forward": list(self.forward)}


def _invert_3x3(matrix: Sequence[float]) -> tuple[float, ...]:
    """Invert a row-major 3x3 matrix.

    Args:
        matrix: Nine coefficients.

    Returns:
        The inverse, or the identity when the matrix is singular - a singular
        warp is a bug, and returning the identity degrades the sample rather
        than crashing the run.
    """
    a, b, c, d, e, f, g, h, i = matrix
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) < 1e-12:
        LOGGER.warning("singular geometry matrix; falling back to identity")
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return (
        (e * i - f * h) / determinant,
        (c * h - b * i) / determinant,
        (b * f - c * e) / determinant,
        (f * g - d * i) / determinant,
        (a * i - c * g) / determinant,
        (c * d - a * f) / determinant,
        (d * h - e * g) / determinant,
        (b * g - a * h) / determinant,
        (a * e - b * d) / determinant,
    )


def _homography(
    source: Sequence[tuple[float, float]], destination: Sequence[tuple[float, float]]
) -> tuple[float, ...]:
    """Solve for the homography mapping four source corners to four targets.

    Args:
        source: Four ``(x, y)`` points.
        destination: Their four images.

    Returns:
        Nine row-major coefficients.
    """
    rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        values.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.append(v)

    # Gaussian elimination with partial pivoting on the 8x8 system.
    size = 8
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-12:
            return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        rows[column], rows[pivot] = rows[pivot], rows[column]
        values[column], values[pivot] = values[pivot], values[column]
        for row in range(size):
            if row == column:
                continue
            factor = rows[row][column] / rows[column][column]
            if factor == 0.0:
                continue
            for index in range(column, size):
                rows[row][index] -= factor * rows[column][index]
            values[row] -= factor * values[column]
    solution = [values[index] / rows[index][index] for index in range(size)]
    return tuple(solution) + (1.0,)


def _build_geometry(
    width: int,
    height: int,
    config: DegradationConfig,
    rng: random.Random,
    chosen: set[str],
) -> Geometry:
    """Compose the sample's rotation and perspective into one homography.

    Combining them into a single resample matters: every resample softens the
    ink, so applying rotation and perspective separately would blur the page
    twice for no visual gain.

    Args:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        config: Degradation settings.
        rng: The sample's seeded RNG.
        chosen: Effects selected for this sample; mutated with what was used.

    Returns:
        The composed geometry.
    """
    corners = [
        (0.0, 0.0),
        (float(width), 0.0),
        (float(width), float(height)),
        (0.0, float(height)),
    ]
    targets = [list(corner) for corner in corners]

    if "skew" in chosen:
        angle = math.radians(config.strength["skew_deg"].draw(rng))
        cos, sin = math.cos(angle), math.sin(angle)
        centre_x, centre_y = width / 2.0, height / 2.0
        for index, (x, y) in enumerate(corners):
            dx, dy = x - centre_x, y - centre_y
            targets[index] = [
                centre_x + dx * cos - dy * sin,
                centre_y + dx * sin + dy * cos,
            ]

    if "perspective" in chosen:
        amount = config.strength["perspective"].draw(rng)
        for index in range(4):
            targets[index][0] += rng.uniform(-amount, amount) * width
            targets[index][1] += rng.uniform(-amount, amount) * height

    forward = _homography(corners, [(point[0], point[1]) for point in targets])
    return Geometry(width, height, forward)


def _tint_rgb(value: str) -> tuple[int, int, int]:
    """Parse a ``#rrggbb`` colour into components.

    Args:
        value: Hex colour.

    Returns:
        The ``(r, g, b)`` components.
    """
    text = value.lstrip("#")
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _paper_layer(
    size: tuple[int, int], config: DegradationConfig, rng: random.Random
) -> tuple[Any, dict[str, Any]]:
    """Build the multiplicative paper layer: tint, yellowing, wash, watermark.

    Returned as an RGB image where white means "leave this pixel alone", so it
    can be multiplied into the page. Everything here sits *behind* the text.

    Args:
        size: Canvas size in pixels.
        config: Degradation settings.
        rng: The sample's seeded RNG.

    Returns:
        ``(layer, record)`` - the RGB multiplier and what was applied, for
        ``aug``.
    """
    Image, _, ImageDraw, ImageFilter, _ = _require_pillow()
    from ...dataset.common.io_utils import weighted_choice

    width, height = size
    record: dict[str, Any] = {}
    paper = config.paper

    tint = weighted_choice(tuple(paper.tints.items()), rng)
    layer = Image.new("RGB", size, _tint_rgb(tint))
    record["tint"] = tint

    # Age yellowing, strongest at the edges and corners, as paper oxidises
    # from the outside in.
    if rng.random() < paper.yellowing.probability:
        strength = paper.yellowing.strength.draw(rng)
        field = Image.new("L", (_FIELD, _FIELD), 255)
        draw = ImageDraw.Draw(field)
        for step in range(10):
            inset = step * (_FIELD // 22 + 1)
            level = int(255 - strength * 255 * (1.0 - step / 10.0))
            draw.rectangle(
                [inset, inset, _FIELD - 1 - inset, _FIELD - 1 - inset], outline=level
            )
        field = field.filter(ImageFilter.GaussianBlur(3)).resize(
            size, Image.Resampling.BICUBIC
        )
        warm = Image.new("RGB", size, (255, 236, 196))
        layer = Image.composite(layer, Image.blend(layer, warm, 0.85), field)
        record["yellowing"] = round(strength, 4)

    # A coloured wash or panel behind the text: a form's tinted field, a
    # certificate's border, a highlighted callout. This is the part that gives
    # the model low-contrast text on the same page as high-contrast text.
    if rng.random() < paper.wash.probability:
        alpha = paper.wash.alpha.draw(rng)
        hue = rng.choice(list(paper.wash.hues)) / 360.0
        red, green, blue = colorsys.hls_to_rgb(
            hue, rng.uniform(0.55, 0.78), rng.uniform(0.35, 0.9)
        )
        colour = (round(red * 255), round(green * 255), round(blue * 255))
        shape = weighted_choice(tuple(paper.wash.shapes.items()), rng)
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        level = int(alpha * 255)
        if shape == "flood":
            draw.rectangle([0, 0, width, height], fill=level)
        elif shape == "band":
            band = rng.randint(
                int(height * 0.08), max(int(height * 0.08) + 1, int(height * 0.28))
            )
            top = rng.randint(0, max(1, height - band))
            draw.rectangle([0, top, width, top + band], fill=level)
        elif shape == "panel":
            inset_x = rng.randint(
                int(width * 0.04), max(int(width * 0.04) + 1, int(width * 0.16))
            )
            inset_y = rng.randint(
                int(height * 0.04), max(int(height * 0.04) + 1, int(height * 0.16))
            )
            draw.rounded_rectangle(
                [inset_x, inset_y, width - inset_x, height - inset_y],
                radius=rng.choice([0, 8, 18, 32]),
                fill=level,
            )
        else:  # bloom
            field = Image.new("L", (_FIELD, _FIELD), 0)
            bloom = ImageDraw.Draw(field)
            cx, cy = rng.randint(8, _FIELD - 8), rng.randint(8, _FIELD - 8)
            radius = rng.randint(_FIELD // 4, _FIELD // 2)
            bloom.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius], fill=level
            )
            mask = field.filter(ImageFilter.GaussianBlur(_FIELD // 8)).resize(
                size, Image.Resampling.BICUBIC
            )
        if shape != "bloom":
            mask = mask.filter(
                ImageFilter.GaussianBlur(max(2, min(width, height) // 160))
            )
        layer = Image.composite(Image.new("RGB", size, colour), layer, mask)
        record["wash"] = {
            "shape": shape,
            "alpha": round(alpha, 4),
            "hue": round(hue * 360, 1),
        }

    # A faint repeated watermark. Not annotated: it is noise with provenance.
    if rng.random() < paper.watermark.probability:
        alpha = paper.watermark.alpha.draw(rng)
        word = rng.choice(_WATERMARK_WORDS)
        scale = paper.watermark.scale.draw(rng)
        angle = paper.watermark.rotate_deg.draw(rng)
        stamp = Image.new("L", size, 0)
        draw = ImageDraw.Draw(stamp)
        # Drawn with Pillow's built-in bitmap font: the glyphs are wrong for
        # Persian, which is exactly why this is unannotated decoration rather
        # than text. What matters is a plausible ink smudge behind the words.
        size_px = max(12, int(min(width, height) * scale * 0.25))
        try:
            draw.text(
                (width // 2, height // 2),
                word,
                fill=int(alpha * 255 * 2),
                anchor="mm",
                font_size=size_px,
            )
        except (TypeError, AttributeError):
            draw.text((width // 2, height // 2), word, fill=int(alpha * 255 * 2))
        stamp = stamp.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
        stamp = stamp.filter(ImageFilter.GaussianBlur(max(1, size_px // 24)))
        layer = Image.composite(Image.new("RGB", size, (90, 90, 110)), layer, stamp)
        record["watermark"] = {
            "word": word,
            "alpha": round(alpha, 4),
            "rotate_deg": round(angle, 2),
        }

    return layer, record


def _show_through_layer(
    size: tuple[int, int],
    verso_png: bytes,
    config: DegradationConfig,
    rng: random.Random,
) -> tuple[Any, dict[str, Any]]:
    """Build the multiplicative ghost of the sheet's other side.

    Args:
        size: Canvas size in pixels.
        verso_png: Another sample's rendered PNG - the text on the back.
        config: Degradation settings.
        rng: The sample's seeded RNG.

    Returns:
        ``(layer, record)``, where ``layer`` is an RGB multiplier that is white
        away from the ghost's ink.
    """
    Image, _, _, ImageFilter, ImageOps = _require_pillow()
    settings = config.show_through
    width, height = size

    with Image.open(io.BytesIO(verso_png)) as opened:
        verso = opened.convert("L")

    # Mirror: physically correct for show-through. The small unmirrored
    # fraction imitates the other real case - the previous page of a stack
    # seen through a thin sheet.
    mirrored = rng.random() < settings.mirror_probability
    if mirrored:
        verso = ImageOps.mirror(verso)

    scale = settings.scale.draw(rng)
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    verso = verso.resize(target, Image.Resampling.BILINEAR)

    angle = settings.rotate_deg.draw(rng)
    if abs(angle) > 0.01:
        verso = verso.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)

    # Paper scatters light, so the ghost is always softer than the recto.
    blur = settings.blur_px.draw(rng)
    verso = verso.filter(ImageFilter.GaussianBlur(blur))

    canvas = Image.new("L", size, 255)
    offset_x = int(settings.offset.draw(rng) * width) + (width - target[0]) // 2
    offset_y = int(settings.offset.draw(rng) * height) + (height - target[1]) // 2
    canvas.paste(verso, (offset_x, offset_y))

    # Flatten the ghost's contrast to `alpha`: 0 becomes (1 - alpha) * 255, so
    # the darkest ghost ink darkens the page by at most `alpha`.
    alpha = settings.alpha.draw(rng)
    floor = int(round((1.0 - alpha) * 255))
    canvas = canvas.point(lambda level: floor + (level * (255 - floor)) // 255)

    # Show-through picks up the paper's colour rather than the ink's, so the
    # ghost is tinted and desaturated relative to the recto.
    hue_shift = settings.hue_shift_deg.draw(rng)
    desaturate = settings.desaturate.draw(rng)
    hue = (
        ((hue_shift % 360) / 360.0)
        if hue_shift >= 0
        else ((360 + hue_shift % -360) % 360) / 360.0
    )
    red, green, blue = colorsys.hls_to_rgb(hue, 0.5, max(0.0, 1.0 - desaturate))
    strength = (red, green, blue)
    channels = []
    for component in strength:
        # A channel with more of the tint is darkened slightly less, which is
        # what produces the warm brown or cool grey cast of real show-through.
        lighten = int(round((1.0 - component) * (255 - floor) * 0.35))
        channels.append(canvas.point(lambda level, lift=lighten: min(255, level + lift)))
    layer = Image.merge("RGB", channels)

    return layer, {
        "alpha": round(alpha, 4),
        "blur_px": round(blur, 2),
        "hue_shift_deg": round(hue_shift, 1),
        "desaturate": round(desaturate, 3),
        "scale": round(scale, 4),
        "rotate_deg": round(angle, 2),
        "mirrored": mirrored,
    }


def _light_field(
    size: tuple[int, int], config: DegradationConfig, rng: random.Random, chosen: set[str]
) -> tuple[Any, dict[str, Any]]:
    """Build the multiplicative lighting field: vignette, shadow, glare.

    Args:
        size: Canvas size in pixels.
        config: Degradation settings.
        rng: The sample's seeded RNG.
        chosen: Effects selected for this sample.

    Returns:
        ``(field, record)`` - an ``L`` image where 255 means untouched - or
        ``(None, {})`` when the page was lit evenly.
    """
    Image, _, ImageDraw, ImageFilter, _ = _require_pillow()
    record: dict[str, Any] = {}
    active = {"vignette", "shadow", "glare", "low_light"} & chosen
    if not active:
        return None, record

    field = Image.new("L", (_FIELD, _FIELD), 255)
    draw = ImageDraw.Draw(field)

    if "vignette" in chosen:
        amount = config.strength["vignette"].draw(rng)
        for step in range(_FIELD // 2):
            level = int(255 * (1.0 - amount * (step / (_FIELD / 2.0)) ** 2))
            draw.rectangle(
                [step, step, _FIELD - 1 - step, _FIELD - 1 - step], outline=level
            )
        record["vignette"] = round(amount, 4)

    if "shadow" in chosen:
        # A soft band along one edge: a hand, a lid that did not close, the
        # page's own curl. Kept light (150+) because this field is multiplied
        # with the vignette and the low-light factor, and a dark band here
        # compounds into a uniformly muddy page.
        side = rng.choice(["top", "bottom", "left", "right"])
        depth = rng.randint(_FIELD // 8, _FIELD // 3)
        level = rng.randint(150, 215)
        box = {
            "top": [0, 0, _FIELD, depth],
            "bottom": [0, _FIELD - depth, _FIELD, _FIELD],
            "left": [0, 0, depth, _FIELD],
            "right": [_FIELD - depth, 0, _FIELD, _FIELD],
        }[side]
        draw.rectangle(box, fill=level)
        record["shadow"] = {"side": side, "level": level}

    if "glare" in chosen:
        cx, cy = rng.randint(0, _FIELD), rng.randint(0, _FIELD)
        radius = rng.randint(_FIELD // 6, _FIELD // 3)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
        record["glare"] = True

    if "low_light" in chosen:
        factor = rng.uniform(0.72, 0.92)
        field = field.point(lambda level: int(level * factor))
        record["low_light"] = round(factor, 3)

    field = field.filter(ImageFilter.GaussianBlur(_FIELD // 10)).resize(
        size, Image.Resampling.BICUBIC
    )

    # Bound the total darkening. Each effect above is defensible alone, but
    # vignette, shadow and low light multiply together, and three moderate
    # draws compose into a page that is uniformly grey - a systematic bias
    # rather than the local variation that was wanted. Lifting the field so
    # its mean stays above a floor preserves the spatial structure while
    # keeping the paper paper-coloured.
    statistics = list(field.resize((32, 32), Image.Resampling.BILINEAR).tobytes())
    mean = (sum(statistics) / len(statistics) / 255.0) if statistics else 1.0
    if mean < _LIGHT_MEAN_FLOOR:
        lift = _LIGHT_MEAN_FLOOR / max(mean, 1e-6)
        field = field.point(lambda level: min(255, int(level * lift)))
        record["light_lift"] = round(lift, 3)

    return field, record


def _add_noise(
    image: Any, config: DegradationConfig, rng: random.Random, chosen: set[str]
) -> tuple[Any, dict[str, Any]]:
    """Add sensor noise, specks and optional blur.

    Args:
        image: The RGB page.
        config: Degradation settings.
        rng: The sample's seeded RNG.
        chosen: Effects selected for this sample.

    Returns:
        ``(image, record)``.
    """
    Image, _, ImageDraw, ImageFilter, _ = _require_pillow()
    record: dict[str, Any] = {}

    if "blur" in chosen:
        radius = config.strength["blur_radius"].draw(rng)
        image = image.filter(ImageFilter.GaussianBlur(radius))
        record["blur_radius"] = round(radius, 3)

    if "scanner_noise" in chosen:
        sigma = config.strength["noise_sigma"].draw(rng)
        try:
            import numpy as np

            array = np.asarray(image, dtype=np.float32)
            noise = np.random.default_rng(rng.getrandbits(32)).normal(
                0.0, sigma, array.shape
            )
            array = np.clip(array + noise, 0, 255).astype("uint8")
            image = Image.fromarray(array, "RGB")
        except ImportError:
            pass
        record["noise_sigma"] = round(sigma, 3)

    if "stains" in chosen:
        width, height = image.size
        overlay = Image.new("L", image.size, 255)
        draw = ImageDraw.Draw(overlay)
        for _ in range(rng.randint(1, 5)):
            radius = rng.randint(
                max(6, min(width, height) // 60), max(12, min(width, height) // 12)
            )
            cx, cy = rng.randint(0, width), rng.randint(0, height)
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=rng.randint(200, 246),
            )
        overlay = overlay.filter(
            ImageFilter.GaussianBlur(max(3, min(width, height) // 120))
        )
        _, ImageChops, _, _, _ = _require_pillow()
        image = ImageChops.multiply(
            image, Image.merge("RGB", (overlay, overlay, overlay))
        )
        record["stains"] = True

    if "folds" in chosen:
        width, height = image.size
        overlay = Image.new("L", image.size, 255)
        draw = ImageDraw.Draw(overlay)
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.5:
                y = rng.randint(height // 6, height - height // 6)
                draw.line(
                    [0, y, width, y], fill=rng.randint(170, 228), width=rng.randint(2, 6)
                )
            else:
                x = rng.randint(width // 6, width - width // 6)
                draw.line(
                    [x, 0, x, height], fill=rng.randint(170, 228), width=rng.randint(2, 6)
                )
        overlay = overlay.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 5.0)))
        _, ImageChops, _, _, _ = _require_pillow()
        image = ImageChops.multiply(
            image, Image.merge("RGB", (overlay, overlay, overlay))
        )
        record["folds"] = True

    if "highlighter" in chosen:
        width, height = image.size
        overlay = Image.new("RGB", image.size, (255, 255, 255))
        draw = ImageDraw.Draw(overlay)
        hue = rng.choice([0.14, 0.18, 0.32, 0.52, 0.88])
        red, green, blue = colorsys.hls_to_rgb(hue, 0.72, 0.95)
        colour = (round(red * 255), round(green * 255), round(blue * 255))
        for _ in range(rng.randint(1, 3)):
            y = rng.randint(0, max(1, height - 24))
            thickness = rng.randint(max(10, height // 70), max(18, height // 34))
            x0 = rng.randint(0, width // 2)
            draw.rectangle(
                [x0, y, x0 + rng.randint(width // 6, width - x0), y + thickness],
                fill=colour,
            )
        overlay = overlay.filter(ImageFilter.GaussianBlur(2.5))
        _, ImageChops, _, _, _ = _require_pillow()
        image = ImageChops.multiply(image, overlay)
        record["highlighter"] = True

    return image, record


def _resample(
    image: Any, config: DegradationConfig, rng: random.Random, chosen: set[str]
) -> tuple[Any, dict[str, Any]]:
    """Downscale and re-upscale, imitating a low-resolution capture.

    Args:
        image: The RGB page.
        config: Degradation settings.
        rng: The sample's seeded RNG.
        chosen: Effects selected for this sample.

    Returns:
        ``(image, record)``. The canvas size is unchanged, so annotations stay
        valid - the loss is in detail, not in geometry.
    """
    Image, _, _, _, _ = _require_pillow()
    if "rescale" not in chosen:
        return image, {}
    factor = config.strength["rescale"].draw(rng)
    width, height = image.size
    small = (max(16, int(width * factor)), max(16, int(height * factor)))
    image = image.resize(small, Image.Resampling.BILINEAR).resize(
        (width, height), Image.Resampling.BICUBIC
    )
    return image, {"rescale": round(factor, 4)}


@dataclass(frozen=True, slots=True)
class DegradeResult:
    """A degraded image and the record of what was done to it.

    Attributes:
        payload: The encoded image bytes.
        image_format: ``png`` or ``jpeg``.
        width: Final width in pixels.
        height: Final height in pixels.
        geometry: The warp applied, so annotations can be mapped through it.
        aug: Everything applied, for the manifest.
    """

    payload: bytes
    image_format: str
    width: int
    height: int
    geometry: Geometry | None
    aug: Mapping[str, Any]


def encode_image(image: Any, *, image_format: str, quality: int = 90) -> bytes:
    """Encode a Pillow image to bytes.

    Args:
        image: The image.
        image_format: ``png`` or ``jpeg``.
        quality: JPEG quality; ignored for PNG.

    Returns:
        The encoded bytes.
    """
    buffer = io.BytesIO()
    if image_format == "jpeg":
        image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    else:
        image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def apply_degradation(
    png: bytes,
    config: DegradationConfig,
    rng: random.Random,
    *,
    clean: bool,
    jpeg_quality: tuple[int, int] = (55, 95),
    verso_png: bytes | None = None,
    verso_id: str = "",
    keep_background: bool = True,
) -> DegradeResult:
    """Degrade one rendered page.

    Args:
        png: The clean Chromium raster.
        config: Degradation settings.
        rng: The sample's seeded RNG.
        clean: When ``True``, skip every effect and re-encode as PNG. A quarter
            of the corpus takes this path so the model also sees pristine text.
        jpeg_quality: Quality range for the JPEG path.
        verso_png: Another sample's render, used as the show-through ghost.
        verso_id: That sample's id, recorded in ``aug`` so the pairing is
            auditable.
        keep_background: Whether the page already carries its own background.
            Templates paint their own, so the paper layer is multiplied in
            rather than composited behind.

    Returns:
        The degraded image and its record.
    """
    Image, ImageChops, _, _, _ = _require_pillow()

    with Image.open(io.BytesIO(png)) as opened:
        image = opened.convert("RGB")
    width, height = image.size

    if clean:
        return DegradeResult(
            payload=encode_image(image, image_format="png"),
            image_format="png",
            width=width,
            height=height,
            geometry=None,
            aug={"clean": True},
        )

    aug: dict[str, Any] = {"clean": False}
    chosen = {
        name
        for name, probability in config.probabilities.items()
        if rng.random() < probability
    }
    aug["effects"] = sorted(chosen)

    # --- behind the ink: paper, wash, watermark, show-through -------------
    if "paper_texture" in chosen:
        layer, record = _paper_layer((width, height), config, rng)
        image = ImageChops.multiply(image, layer)
        aug["paper"] = record

    if verso_png is not None and rng.random() < config.show_through.probability:
        layer, record = _show_through_layer((width, height), verso_png, config, rng)
        image = ImageChops.multiply(image, layer)
        record["source_id"] = verso_id
        aug["show_through"] = record

    # --- the ink itself ---------------------------------------------------
    if "ink_bleed" in chosen:
        _, _, _, ImageFilter, _ = _require_pillow()
        bleed = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.8)))
        image = Image.blend(image, bleed, rng.uniform(0.15, 0.45))
        aug["ink_bleed"] = True

    if "uneven_toner" in chosen:
        _, _, ImageDraw, ImageFilter, _ = _require_pillow()
        field = Image.new("L", (_FIELD, _FIELD), 255)
        draw = ImageDraw.Draw(field)
        for _ in range(rng.randint(2, 6)):
            cx, cy = rng.randint(0, _FIELD), rng.randint(0, _FIELD)
            radius = rng.randint(_FIELD // 8, _FIELD // 3)
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=rng.randint(205, 248),
            )
        field = field.filter(ImageFilter.GaussianBlur(_FIELD // 8)).resize(
            (width, height), Image.Resampling.BICUBIC
        )
        image = ImageChops.multiply(image, Image.merge("RGB", (field, field, field)))
        aug["uneven_toner"] = True

    # --- geometry: pixels and boxes move through one matrix ---------------
    geometry: Geometry | None = None
    if {"skew", "perspective"} & chosen:
        geometry = _build_geometry(width, height, config, rng, chosen)
        if not geometry.is_identity:
            background = image.getpixel((2, 2))
            image = image.transform(
                (width, height),
                Image.Transform.PERSPECTIVE,
                geometry.pillow_coefficients(),
                resample=Image.Resampling.BICUBIC,
                fillcolor=background,
            )
            aug["geometry"] = geometry.as_dict()
        else:
            geometry = None

    # --- light ------------------------------------------------------------
    field, record = _light_field((width, height), config, rng, chosen)
    if field is not None:
        image = ImageChops.multiply(image, Image.merge("RGB", (field, field, field)))
        aug["light"] = record

    # --- noise and resolution --------------------------------------------
    image, record = _add_noise(image, config, rng, chosen)
    aug.update(record)
    image, record = _resample(image, config, rng, chosen)
    aug.update(record)

    # --- encode -----------------------------------------------------------
    if "jpeg" in chosen:
        quality = rng.randint(jpeg_quality[0], jpeg_quality[1])
        payload = encode_image(image, image_format="jpeg", quality=quality)
        aug["jpeg_quality"] = quality
        image_format = "jpeg"
    else:
        payload = encode_image(image, image_format="png")
        image_format = "png"

    return DegradeResult(
        payload=payload,
        image_format=image_format,
        width=width,
        height=height,
        geometry=geometry,
        aug=aug,
    )
