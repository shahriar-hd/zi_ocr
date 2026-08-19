"""Scanner simulation: turn a clean render into something that looks scanned.

Chromium hands back a mathematically perfect page - exact geometry, even
illumination, hard glyph edges, pure tones. No document an OCR model meets in
production looks like that. Training only on clean renders teaches the adapter
that ink is sharp, paper is flat and text is black, and the first real scan then
falls outside everything it has seen.

This module reintroduces the physics, in the order a real page acquires it:

1. **Paper.** The render is captured with a transparent ground (see
   ``browser.PageRenderer``), so the light areas of the letter carry no ink at
   all. A paper base is built - a tone from ``scan.paper.tints`` plus, half the
   time, one of the prepared scans in ``data/texture`` put through the passes a
   scanner's own software applies - and the page is composited onto it. Nothing
   is ever pure white.
2. **Ink.** Gaussian blur softens the glyph edges, a lightening pass lifts the
   darkest ink off pure black, and an optional bleed-through ghost and
   morphological thickening imitate toner spread and a second printed side.
3. **Geometry.** One resample applies sheet rotation, camera perspective, page
   bow and lens barrel distortion together. Doing it in a single pass matters:
   each resample softens the image, and the softening is not part of the effect.
4. **Light.** An uneven illumination field, a vignette, an optional local
   shadow, then gamma, brightness and contrast - the sensor's response.
5. **Noise.** Gaussian sensor noise, a handful of soft dust specks, and finally
   the output pass the scanner's software performs: greyscale or 1-bit
   thresholding, scanning below nominal DPI, JPEG compression.

Every decision comes from a seeded RNG, so a sample reproduces byte for byte
(CLAUDE.md 8). Paper is drawn once per document because a letter is printed on
one sheet of paper; geometry, light and noise are drawn per sheet because each
page crosses the platen separately.

Ground truth has to follow the pixels. :class:`SheetGeometry` exposes the same
mapping the resample used, so ``pipeline.py`` can push each bounding box through
it and the annotation keeps describing where the text actually is.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from random import Random
from typing import Any, Final, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps, ImageStat

from ....config import ImageFormatConfig, ScanConfig
from ..layout.texture import Texture, load_texture
from .encode import encode_image

__all__ = [
    "PaperProfile",
    "SheetProfile",
    "ScanProfile",
    "SheetGeometry",
    "build_scan_profile",
    "build_paper",
    "apply_scan",
    "flatten_clean",
]

#: Encoding used when a caller passes no format. PNG rather than the config's
#: default so a test or a one-off call gets lossless bytes unless it asks
#: otherwise; the pipeline always passes ``config.render.image`` explicitly.
_DEFAULT_IMAGE_FORMAT: Final[ImageFormatConfig] = ImageFormatConfig(format="png")

#: Rows processed at a time when noise is added. A full A3 page at 300 DPI is
#: ~50 MPx; widening an int16 copy of all of it costs several hundred MB, and
#: nothing about the operation needs the whole array resident.
_NOISE_BAND_ROWS: Final[int] = 512

#: How far a speck's Gaussian profile is stamped, in multiples of its radius.
#: Past 2.5 sigma the profile is under 5% of its peak and rounds away.
_SPECK_EXTENT: Final[float] = 2.5

#: Share of specks that are dark. Dust on the platen, loose toner and paper
#: fibre are dark; a bright dropout needs a dead sensor cell, which is rarer.
_SPECK_DARK_SHARE: Final[float] = 0.68

#: Per-speck multiplier on the sheet's peak strength, so one sheet's specks vary
#: in depth instead of being a set of identical blots.
_SPECK_PEAK_JITTER: Final[tuple[float, float]] = (0.45, 1.0)

#: Ceiling on speck density, in specks per megapixel. Not a tuning knob - it is
#: there so a mistyped config (0.5 written as 500) cannot spend minutes stamping
#: a page into mud. The default range is an order of magnitude below it.
_SPECK_MAX_PER_MPX: Final[float] = 60.0

#: Resolution of the illumination field before it is scaled to the page. Light
#: falls off smoothly, so a field this small upscales without a visible seam and
#: costs nothing to build.
_LIGHT_FIELD: Final[int] = 64

#: Darkest the light field may pull the page. A scanner's worst corner is dim,
#: not black, and an unbounded falloff would swallow the text sitting in it.
_LIGHT_FLOOR: Final[float] = 0.62

#: Mesh density for the geometric resample. Rotation is exactly reproduced by
#: any grid; the bow and the barrel term are curved, and this many quads keeps
#: the piecewise error well inside a pixel at 300 DPI.
_MESH_COLUMNS: Final[int] = 16
_MESH_ROWS: Final[int] = 22

#: Iterations used to invert the geometric map for a single point. The map is a
#: near-identity, so the fixed point converges in a handful of steps.
_INVERSE_STEPS: Final[int] = 12


@dataclass(frozen=True, slots=True)
class PaperProfile:
    """The paper one document is printed on. Drawn once per document.

    Attributes:
        tint: Base paper tone as ``#rrggbb``. Never pure white.
        texture: The prepared scan multiplied onto the tone, or ``None``.
        strength: Weight of the scan against the flat tone, in ``[0, 1]``.
        grayscale: Whether the scan is desaturated first, as a scanner's
            greyscale mode would.
        autocontrast: Whether the scan's own histogram is stretched, which is
            what makes a flat photograph of paper read as grain.
        brightness: Multiplier applied to the scan before compositing.
    """

    tint: str
    texture: Texture | None
    strength: float
    grayscale: bool
    autocontrast: bool
    brightness: float

    def as_content(self) -> dict[str, object]:
        """Return the paper decisions for the annotation's provenance block."""
        return {
            "tint": self.tint,
            "texture": None if self.texture is None else self.texture.name,
            "strength": round(self.strength, 3),
            "grayscale": self.grayscale,
            "autocontrast": self.autocontrast,
            "brightness": round(self.brightness, 3),
        }


@dataclass(frozen=True, slots=True)
class SheetProfile:
    """One sheet's own trip through the scanner. Drawn per page.

    Attributes:
        blur: Gaussian blur radius in pixels, softening the glyph edges.
        lighten: How far the ink is lifted toward the paper, in ``[0, 1]``.
        bleed: Opacity of the mirrored ghost printed from the sheet's other
            side. Zero disables it.
        dilate: Whether toner spread is imitated by thickening dark pixels.
        rotation_deg: Sheet rotation on the platen, in degrees.
        perspective: Corner displacement as a fraction of the page, one value
            per corner, clockwise from top-left.
        warp: Page bow as a fraction of the page height.
        barrel: Radial lens distortion; negative barrels, positive pincushions.
        illumination: Strength of the uneven light gradient, in ``[0, 1]``.
        light_angle: Direction the light falls from, in radians.
        vignette: Strength of the corner falloff, in ``[0, 1]``.
        shadow_edge: Which edge a local shadow falls along, or ``""``.
        shadow_strength: Depth of that shadow, in ``[0, 1]``.
        gamma: Sensor gamma. Below 1 lightens, above 1 darkens.
        brightness: Post-gamma brightness multiplier.
        contrast: Post-gamma contrast multiplier.
        sigma: Standard deviation of the Gaussian sensor noise, in levels.
        speck_density: Dust specks per megapixel. Per megapixel rather than as
            a pixel fraction so the same setting means the same visible
            dustiness at 150 and at 400 DPI.
        speck_radius: Gaussian radius of one speck, in pixels. A speck is a
            soft blot with a smooth falloff, never a single flipped pixel -
            nothing in a scanner produces a lone black pixel surrounded by
            clean paper, and a model that learns to ignore that pattern has
            learned nothing transferable.
        speck_strength: How far the deepest speck on this sheet pulls its
            pixels toward black or white, in ``[0, 1]``. Below 1, so a speck
            tints the paper rather than punching a hole in it.
        jpeg_quality: Quality of a JPEG round trip, or ``None`` to skip it.
        resample_scale: Fraction of nominal resolution the page is scanned at
            before being scaled back up. ``1.0`` skips the round trip.
        mode: Output mode - ``color``, ``gray`` or ``bilevel``.
    """

    blur: float
    lighten: float
    bleed: float
    dilate: bool
    rotation_deg: float
    perspective: tuple[float, float, float, float]
    warp: float
    barrel: float
    illumination: float
    light_angle: float
    vignette: float
    shadow_edge: str
    shadow_strength: float
    gamma: float
    brightness: float
    contrast: float
    sigma: float
    speck_density: float
    speck_radius: float
    speck_strength: float
    jpeg_quality: int | None
    resample_scale: float
    mode: str

    def as_content(self) -> dict[str, object]:
        """Return the sheet's decisions for the annotation's provenance block."""
        return {
            "blur": round(self.blur, 3),
            "lighten": round(self.lighten, 3),
            "bleed": round(self.bleed, 3),
            "rotation_deg": round(self.rotation_deg, 3),
            "warp": round(self.warp, 4),
            "barrel": round(self.barrel, 4),
            "illumination": round(self.illumination, 3),
            "vignette": round(self.vignette, 3),
            "gamma": round(self.gamma, 3),
            "sigma": round(self.sigma, 2),
            "speck_density": round(self.speck_density, 2),
            "speck_radius": round(self.speck_radius, 2),
            "speck_strength": round(self.speck_strength, 3),
            "jpeg_quality": self.jpeg_quality,
            "resample_scale": round(self.resample_scale, 3),
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class ScanProfile:
    """A whole document's scanner treatment.

    Attributes:
        paper: The document's paper, shared by every sheet.
        sheets: One profile per page, in page order.
    """

    paper: PaperProfile
    sheets: tuple[SheetProfile, ...]

    def as_content(self) -> dict[str, object]:
        """Return the profile for the annotation's provenance block."""
        return {
            "paper": self.paper.as_content(),
            "sheets": [sheet.as_content() for sheet in self.sheets],
        }


class SheetGeometry:
    """The geometric map one sheet's resample applied, in both directions.

    Pillow's ``Image.transform`` is defined *backwards*: for each destination
    pixel it asks which source pixel to sample. That is the natural direction
    for resampling and the wrong one for ground truth, which needs to know where
    a source box ended up. Both directions live here so the two can never
    disagree - the mesh handed to Pillow and the box mapping handed to the
    annotation are generated from the same parameters.

    The inverse direction has no closed form once the bow and the barrel term
    are involved, so it is solved by fixed-point iteration. The map is a
    near-identity by construction (fractions of a degree, fractions of a
    percent), which makes the iteration converge in a few steps and stay well
    inside a pixel.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        rotation_deg: float,
        perspective: Sequence[float],
        warp: float,
        barrel: float,
    ) -> None:
        """Build the map for one sheet.

        Args:
            width: Page width in pixels.
            height: Page height in pixels.
            rotation_deg: Sheet rotation on the platen.
            perspective: Four corner displacements as fractions of the page,
                clockwise from top-left.
            warp: Page bow as a fraction of the page height.
            barrel: Radial distortion coefficient.
        """
        self.width = width
        self.height = height
        self.warp = warp
        self.barrel = barrel
        self._half_w = width / 2.0
        self._half_h = height / 2.0
        self._matrix = _corner_matrix(width, height, rotation_deg, perspective)

    @property
    def is_identity(self) -> bool:
        """Whether the map leaves every pixel where it is."""
        return (
            abs(self.warp) < 1e-9
            and abs(self.barrel) < 1e-9
            and np.allclose(self._matrix, np.eye(3), atol=1e-9)
        )

    def source_of(self, x: float, y: float) -> tuple[float, float]:
        """Return the source pixel that lands at destination ``(x, y)``.

        Args:
            x: Destination column.
            y: Destination row.

        Returns:
            The source coordinate to sample, which may fall outside the page.
        """
        u = (x - self._half_w) / self._half_w
        v = (y - self._half_h) / self._half_h
        if self.barrel:
            radius_sq = u * u + v * v
            factor = 1.0 + self.barrel * radius_sq
            u *= factor
            v *= factor
        if self.warp:
            # A single hump across the page: the middle of every line bows away
            # from the platen, which is what a bound or curled sheet does.
            v += self.warp * math.sin(math.pi * (u + 1.0) / 2.0)
        px = u * self._half_w + self._half_w
        py = v * self._half_h + self._half_h
        return _apply_matrix(self._matrix, px, py)

    def destination_of(self, x: float, y: float) -> tuple[float, float]:
        """Return where source pixel ``(x, y)`` ended up.

        Args:
            x: Source column.
            y: Source row.

        Returns:
            The destination coordinate, found by inverting :meth:`source_of`.
        """
        guess_x, guess_y = x, y
        for _ in range(_INVERSE_STEPS):
            got_x, got_y = self.source_of(guess_x, guess_y)
            error_x, error_y = x - got_x, y - got_y
            if abs(error_x) < 1e-3 and abs(error_y) < 1e-3:
                break
            guess_x += error_x
            guess_y += error_y
        return guess_x, guess_y

    def map_bbox(self, bbox: Sequence[float]) -> tuple[int, int, int, int]:
        """Map one axis-aligned box through the transform.

        A rotated or warped rectangle is no longer a rectangle, and the schema
        stores axis-aligned boxes, so the result is the bounding box of the four
        mapped corners - the tightest axis-aligned box that still contains all
        the ink it contained before.

        Args:
            bbox: ``[x0, y0, x1, y1]`` in source pixels.

        Returns:
            The mapped box, rounded to integers and clipped to the page.
        """
        x0, y0, x1, y1 = (float(value) for value in bbox)
        corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        mapped = [self.destination_of(cx, cy) for cx, cy in corners]
        xs = [point[0] for point in mapped]
        ys = [point[1] for point in mapped]
        left = max(0, min(self.width, int(math.floor(min(xs)))))
        top = max(0, min(self.height, int(math.floor(min(ys)))))
        right = max(0, min(self.width, int(math.ceil(max(xs)))))
        bottom = max(0, min(self.height, int(math.ceil(max(ys)))))
        return left, top, max(left, right), max(top, bottom)

    def mesh(self) -> list[tuple[tuple[int, int, int, int], tuple[int, ...]]]:
        """Build the quad mesh Pillow needs to perform the resample.

        Returns:
            One ``(destination_box, source_quad)`` pair per grid cell, which is
            exactly the argument :meth:`PIL.Image.transform` expects for
            ``Image.MESH``.
        """
        mesh: list[tuple[tuple[int, int, int, int], tuple[int, ...]]] = []
        edges_x = [
            round(self.width * index / _MESH_COLUMNS)
            for index in range(_MESH_COLUMNS + 1)
        ]
        edges_y = [
            round(self.height * index / _MESH_ROWS) for index in range(_MESH_ROWS + 1)
        ]
        for row in range(_MESH_ROWS):
            for column in range(_MESH_COLUMNS):
                left, right = edges_x[column], edges_x[column + 1]
                top, bottom = edges_y[row], edges_y[row + 1]
                if right <= left or bottom <= top:
                    continue
                quad: list[int] = []
                for cx, cy in ((left, top), (left, bottom), (right, bottom), (right, top)):
                    source_x, source_y = self.source_of(cx, cy)
                    quad.extend((int(round(source_x)), int(round(source_y))))
                mesh.append(((left, top, right, bottom), tuple(quad)))
        return mesh


def _apply_matrix(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Apply a 3x3 homography to one point."""
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    if abs(denominator) < 1e-12:
        return x, y
    mapped_x = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator
    mapped_y = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator
    return mapped_x, mapped_y


def _homography(source: Sequence[tuple[float, float]],
                destination: Sequence[tuple[float, float]]) -> np.ndarray:
    """Solve the 3x3 homography that carries four points onto four others.

    Args:
        source: Four ``(x, y)`` points.
        destination: Where each of those points must land.

    Returns:
        The homography, normalised so its bottom-right entry is 1. Falls back to
        the identity when the system is degenerate, so a pathological corner
        draw degrades to "no transform" instead of raising.
    """
    rows: list[list[float]] = []
    targets: list[float] = []
    for (sx, sy), (dx, dy) in zip(source, destination):
        rows.append([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx])
        targets.append(dx)
        rows.append([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy])
        targets.append(dy)
    try:
        solved = np.linalg.solve(np.array(rows, dtype=np.float64),
                                np.array(targets, dtype=np.float64))
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)
    return np.append(solved, 1.0).reshape(3, 3)


def _corner_matrix(
    width: int,
    height: int,
    rotation_deg: float,
    perspective: Sequence[float],
) -> np.ndarray:
    """Build the destination-to-source homography for a sheet.

    Rotation and perspective are expressed the same way - as displacements of
    the four page corners - and solved together, so the page is resampled once
    rather than twice.

    Args:
        width: Page width in pixels.
        height: Page height in pixels.
        rotation_deg: Rotation about the page centre.
        perspective: Four corner displacements as fractions of the page,
            clockwise from top-left.

    Returns:
        The homography mapping a destination pixel to its source pixel.
    """
    corners = ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)),
               (0.0, float(height)))
    angle = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    centre_x, centre_y = width / 2.0, height / 2.0
    moved: list[tuple[float, float]] = []
    for (x, y), shift in zip(corners, tuple(perspective) + (0.0, 0.0, 0.0, 0.0)):
        offset_x, offset_y = x - centre_x, y - centre_y
        rotated_x = centre_x + offset_x * cos_a - offset_y * sin_a
        rotated_y = centre_y + offset_x * sin_a + offset_y * cos_a
        # A corner pulled toward the page centre reads as that corner being
        # further from the lens, which is all camera perspective is here.
        moved.append(
            (
                rotated_x + (centre_x - rotated_x) * shift,
                rotated_y + (centre_y - rotated_y) * shift,
            )
        )
    # Solved destination -> source, which is the direction Pillow samples in.
    return _homography(moved, corners)


def build_scan_profile(
    config: ScanConfig,
    texture: Texture | None,
    page_count: int,
    rng: Random,
) -> ScanProfile | None:
    """Draw a whole document's scanner treatment.

    Args:
        config: The ``scan`` section of the synth config.
        texture: The document's paper scan, or ``None``.
        page_count: How many sheets the document rendered to.
        rng: The sample's seeded RNG, conventionally ``purpose="scan"``.

    Returns:
        The profile, or ``None`` when the stage is disabled or this sample drew
        the clean branch. The draw happens even when the stage is off, so
        toggling it never shifts a sample's other decisions.
    """
    draw = rng.random()
    if not config.enabled or draw >= config.probability or page_count <= 0:
        return None

    paper = PaperProfile(
        tint=rng.choice(config.paper_tints),
        texture=texture,
        strength=config.texture_strength.draw(rng),
        grayscale=rng.random() < config.texture_grayscale,
        autocontrast=rng.random() < config.texture_autocontrast,
        brightness=config.texture_brightness.draw(rng),
    )
    sheets = tuple(_draw_sheet(config, rng) for _ in range(page_count))
    return ScanProfile(paper=paper, sheets=sheets)


def _draw_sheet(config: ScanConfig, rng: Random) -> SheetProfile:
    """Draw one sheet's trip through the scanner.

    Args:
        config: The ``scan`` section.
        rng: The sample's seeded RNG.

    Returns:
        The sheet's profile. Every range is drawn unconditionally, including
        those a later gate discards, so the stream stays stable as the
        probabilities are retuned.
    """
    perspective_limit = config.perspective.draw(rng)
    # One corner is pushed hardest and the others follow at random depths, which
    # is what holding a camera slightly off-axis does to a page.
    corners = tuple(rng.uniform(0.0, perspective_limit) for _ in range(4))
    shadow_edge = ("top", "bottom", "right", "left")[rng.randrange(4)]
    mode = "color"
    if rng.random() < config.bilevel:
        mode = "bilevel"
    elif rng.random() < config.grayscale:
        mode = "gray"
    return SheetProfile(
        blur=config.ink_blur.draw(rng),
        lighten=config.ink_lighten.draw(rng),
        bleed=(
            config.ink_bleed_alpha.draw(rng)
            if rng.random() < config.ink_bleed
            else 0.0
        ),
        dilate=rng.random() < config.ink_dilate,
        rotation_deg=config.rotation_deg.draw(rng),
        perspective=corners,
        warp=config.warp_amplitude.draw(rng),
        barrel=config.barrel.draw(rng),
        illumination=config.illumination.draw(rng),
        light_angle=rng.uniform(0.0, 2.0 * math.pi),
        vignette=config.vignette.draw(rng),
        shadow_edge=shadow_edge if rng.random() < config.shadow else "",
        shadow_strength=config.shadow_strength.draw(rng),
        gamma=config.gamma.draw(rng),
        brightness=config.brightness.draw(rng),
        contrast=config.contrast.draw(rng),
        sigma=config.gaussian_sigma.draw(rng),
        speck_density=config.speck_density.draw(rng),
        speck_radius=config.speck_radius.draw(rng),
        speck_strength=config.speck_strength.draw(rng),
        jpeg_quality=(
            int(round(config.jpeg_quality.draw(rng)))
            if rng.random() < config.jpeg
            else None
        ),
        resample_scale=(
            config.resample_scale.draw(rng) if rng.random() < config.resample else 1.0
        ),
        mode=mode,
    )


def _hex_to_rgb(hex_colour: str) -> tuple[int, int, int]:
    """Split a ``#rrggbb`` string into channels, tolerating a missing hash."""
    value = hex_colour.lstrip("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    if len(value) != 6:
        return 250, 248, 242
    try:
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    except ValueError:
        return 250, 248, 242


def build_paper(paper: PaperProfile, size: tuple[int, int]) -> Image.Image:
    """Build the paper one sheet is printed on.

    The prepared scans are raw photographs with arbitrary exposure: several are
    mid-grey overall, and multiplying one onto a paper tone directly turns the
    page to slate. What the page wants from them is their *variation*, not their
    exposure, so each channel is flat-fielded - rescaled so its own mean lands
    just below white - and only the departure from that mean survives, scaled by
    ``strength``. The tone therefore keeps its brightness and gains the grain.

    Rescaling is done with a lookup table rather than array arithmetic, which
    keeps the whole operation at 8 bits and O(1) in extra memory; a float field
    the size of an A3 page at 300 DPI would be several hundred megabytes.

    Args:
        paper: The document's paper decisions.
        size: Page size in pixels.

    Returns:
        An opaque ``RGB`` image of the paper at ``size``.
    """
    base = Image.new("RGB", size, _hex_to_rgb(paper.tint))
    if paper.texture is None or paper.strength <= 0.0:
        return base
    try:
        scan = load_texture(paper.texture.path)
    except OSError:
        return base

    grain = scan
    if paper.texture.rotate_180:
        grain = grain.transpose(Image.Transpose.ROTATE_180)
    if paper.grayscale:
        # A scanner's greyscale mode, applied to the paper rather than the page:
        # the grain survives, the photograph's colour cast does not.
        grain = ImageOps.grayscale(grain).convert("RGB")
    if paper.autocontrast:
        # A flat photograph of paper is nearly featureless. Stretching its own
        # histogram is what turns it back into visible grain.
        grain = ImageOps.autocontrast(grain, cutoff=1)
    grain = _cover(grain, size)
    if abs(paper.brightness - 1.0) > 1e-3:
        grain = ImageEnhance.Brightness(grain).enhance(paper.brightness)
    return ImageChops.multiply(base, _flat_field(grain, paper.strength))


#: Level the flat-fielded paper's own mean is mapped to. Just short of white, so
#: the brighter half of the grain has somewhere to go before ``multiply`` clips.
_PAPER_MEAN_LEVEL: Final[float] = 247.0

#: Darkest the paper's grain may pull the tone. Real blank paper varies by a few
#: percent; a scan whose histogram was stretched can otherwise swing far more.
_PAPER_FLOOR: Final[int] = 186


def _flat_field(grain: Image.Image, strength: float) -> Image.Image:
    """Rescale each channel so its mean is near-white and its variation scales.

    Args:
        grain: The resized paper scan in ``RGB``.
        strength: How much of the scan's departure from its own mean survives.

    Returns:
        An ``RGB`` image suitable for multiplying onto a paper tone: mean close
        to white, with the scan's texture as a small modulation around it.
    """
    table: list[int] = []
    for channel in grain.split():
        mean = ImageStat.Stat(channel).mean[0] or 1.0
        table.extend(
            max(
                _PAPER_FLOOR,
                min(
                    255,
                    int(
                        round(
                            _PAPER_MEAN_LEVEL
                            * (1.0 + strength * (level / mean - 1.0))
                        )
                    ),
                ),
            )
            for level in range(256)
        )
    return grain.point(table)


def _cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale an image to cover a target size, cropping the overhang.

    Args:
        image: Source image.
        size: Target size in pixels.

    Returns:
        An image of exactly ``size``, centred on the source. Aspect ratio is
        preserved, so a 3:4 scan on a 1:1.41 sheet is cropped rather than
        stretched into a different grain.
    """
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    scaled = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    if scale > 1.6:
        # Several of the prepared scans are small and JPEG-compressed. Upscaled
        # 6x, their 8x8 blocks become inch-wide rectangles - a regular grid is
        # exactly the artefact an OCR model would happily learn - and a blur
        # proportional to the upscale removes them while keeping the tone
        # variation the texture is there for.
        scaled = scaled.filter(ImageFilter.GaussianBlur(min(8.0, scale * 0.9)))
    left = max(0, (scaled.width - target_w) // 2)
    top = max(0, (scaled.height - target_h) // 2)
    return scaled.crop((left, top, left + target_w, top + target_h))


def _soften_ink(page: Image.Image, sheet: SheetProfile) -> Image.Image:
    """Soften and lighten the printed ink.

    Three effects, all applied to the transparent page rather than to a
    composite, so each one changes the ink and leaves the paper alone:

    * **Thickening.** Dilating the alpha channel spreads toner the way an
      absorbent stock does, before any blur rounds the result off.
    * **Blur.** No printer lays down a hard edge and no lens resolves one.
    * **Lightening.** Scaling the alpha down lifts the darkest ink off pure
      black by letting the paper through it - which is what a scan of printed
      text actually is, and why its histogram never touches zero.

    Args:
        page: The rendered page in ``RGBA``, light areas transparent.
        sheet: The sheet's profile.

    Returns:
        A new ``RGBA`` image.
    """
    softened = page
    if sheet.dilate:
        red, green, blue, alpha = softened.split()
        softened = Image.merge(
            "RGBA", (red, green, blue, alpha.filter(ImageFilter.MaxFilter(3)))
        )
    if sheet.blur > 0.01:
        softened = softened.filter(ImageFilter.GaussianBlur(sheet.blur))
    if sheet.lighten > 0.001:
        red, green, blue, alpha = softened.split()
        faded = alpha.point(
            lambda level: int(round(level * (1.0 - sheet.lighten)))
        )
        softened = Image.merge("RGBA", (red, green, blue, faded))
    return softened


def _bleed_through(page: Image.Image, sheet: SheetProfile) -> Image.Image | None:
    """Build the ghost of the sheet's other printed side.

    Args:
        page: The rendered page in ``RGBA``.
        sheet: The sheet's profile.

    Returns:
        A faint, mirrored, heavily blurred copy to composite *under* the page,
        or ``None`` when this sheet prints one-sided.
    """
    if sheet.bleed <= 0.0:
        return None
    ghost = page.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    ghost = ghost.filter(ImageFilter.BoxBlur(max(1.5, sheet.blur * 3.0 + 1.5)))
    red, green, blue, alpha = ghost.split()
    faded = alpha.point(lambda level: int(round(level * sheet.bleed)))
    return Image.merge("RGBA", (red, green, blue, faded))


def _light_field(size: tuple[int, int], sheet: SheetProfile) -> Image.Image | None:
    """Build the multiplicative light field for one sheet.

    Three contributions are summed at low resolution and scaled up: a linear
    gradient (the lamp is never centred), a radial falloff (the vignette every
    lens has), and one optional soft band along an edge (a hand, a lid that did
    not close, the shadow of the sheet's own curl).

    Building it at :data:`_LIGHT_FIELD` resolution is not an approximation worth
    apologising for - light falls off smoothly, and the bicubic upscale is
    smoother than any field a scanner produces.

    Args:
        size: Page size in pixels.
        sheet: The sheet's profile.

    Returns:
        An ``L`` image the size of the page where 255 means untouched, or
        ``None`` when this sheet was lit evenly.
    """
    if (
        sheet.illumination < 0.005
        and sheet.vignette < 0.005
        and not sheet.shadow_edge
    ):
        return None
    axis_x = np.linspace(-1.0, 1.0, _LIGHT_FIELD, dtype=np.float32)
    axis_y = np.linspace(-1.0, 1.0, _LIGHT_FIELD, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)

    field = np.ones((_LIGHT_FIELD, _LIGHT_FIELD), dtype=np.float32)
    if sheet.illumination > 0.0:
        direction = (
            math.cos(sheet.light_angle) * grid_x + math.sin(sheet.light_angle) * grid_y
        )
        field -= sheet.illumination * (direction + 1.0) / 2.0
    if sheet.vignette > 0.0:
        radius = np.sqrt(grid_x**2 + grid_y**2) / math.sqrt(2.0)
        field -= sheet.vignette * radius**2
    if sheet.shadow_edge:
        distance = {
            "top": (grid_y + 1.0) / 2.0,
            "bottom": (1.0 - grid_y) / 2.0,
            "right": (1.0 - grid_x) / 2.0,
            "left": (grid_x + 1.0) / 2.0,
        }[sheet.shadow_edge]
        field -= sheet.shadow_strength * np.clip(1.0 - distance * 4.0, 0.0, 1.0)

    field = np.clip(field, 0.25, 1.0)
    # The three terms only ever subtract, so their sum would dim the whole page
    # as well as make it uneven - and a scan of white paper is not a grey page.
    # Dividing by the field's own mean re-centres it on 1.0; the half that lands
    # above 1.0 clips back, since a multiply cannot brighten, but the page keeps
    # its paper tone and the falloff keeps its shape.
    field /= max(1e-6, float(field.mean()))
    field = np.clip(field, _LIGHT_FLOOR, 1.0)
    small = Image.fromarray((field * 255.0).astype(np.uint8), mode="L")
    return small.resize(size, Image.Resampling.BICUBIC)


def _apply_tone(image: Image.Image, sheet: SheetProfile) -> Image.Image:
    """Apply the sensor's gamma, brightness and contrast response.

    Args:
        image: The composited page in ``RGB``.
        sheet: The sheet's profile.

    Returns:
        A new ``RGB`` image. Gamma goes through a 256-entry lookup table, which
        is exact for 8-bit data and far cheaper than a per-pixel power.
    """
    result = image
    if abs(sheet.gamma - 1.0) > 0.005:
        table = [
            min(255, max(0, int(round(255.0 * (level / 255.0) ** sheet.gamma))))
            for level in range(256)
        ]
        result = result.point(table * len(result.getbands()))
    if abs(sheet.brightness - 1.0) > 0.005:
        result = ImageEnhance.Brightness(result).enhance(sheet.brightness)
    if abs(sheet.contrast - 1.0) > 0.005:
        result = ImageEnhance.Contrast(result).enhance(sheet.contrast)
    return result


def _speck_count(width: int, height: int, sheet: SheetProfile) -> int:
    """How many dust specks this sheet's density asks for on a page this size.

    Args:
        width: Page width in pixels.
        height: Page height in pixels.
        sheet: The sheet's profile.

    Returns:
        The speck count, zero when the sheet drew no specks.
    """
    density = min(max(sheet.speck_density, 0.0), _SPECK_MAX_PER_MPX)
    if density <= 0.0 or sheet.speck_strength <= 0.0 or sheet.speck_radius <= 0.0:
        return 0
    return int(round(density * (width * height) / 1_000_000.0))


def _speck_kernel(radius: float) -> tuple[np.ndarray, int]:
    """Build one speck's Gaussian profile.

    Args:
        radius: Gaussian sigma in pixels.

    Returns:
        A square ``float32`` profile peaking at ``1.0`` in the centre and
        falling smoothly to near zero at its edge, and its half-width. Stamping
        this profile *is* a blurred point - the falloff is the blur - so no
        separate blur pass is needed and no speck ever has a hard edge.
    """
    extent = max(1, int(math.ceil(_SPECK_EXTENT * radius)))
    offsets = np.arange(-extent, extent + 1, dtype=np.float32)
    squared = offsets**2
    profile = np.exp(
        -(squared[:, None] + squared[None, :]) / (2.0 * radius * radius)
    ).astype(np.float32)
    return profile, extent


def _add_specks(array: np.ndarray, sheet: SheetProfile, generator: Any) -> None:
    """Stamp soft dust specks into a page, in place.

    The specks a flatbed scanner adds are dust, loose toner and paper fibre:
    a few dozen on a whole page, each a soft blot two or three pixels across
    that darkens the paper without replacing it. This deliberately does *not*
    implement salt-and-pepper noise - flipping isolated pixels to pure black
    and white at a percent-of-all-pixels rate produces a texture no scanner
    makes, and a model trained on it learns to discount a pattern it will never
    meet again.

    Args:
        array: The page as ``uint8`` ``RGB``, modified in place.
        sheet: The sheet's profile.
        generator: The seeded array generator, so the specks reproduce with the
            sample.
    """
    height, width = array.shape[:2]
    count = _speck_count(width, height, sheet)
    if count == 0:
        return
    profile, extent = _speck_kernel(sheet.speck_radius)
    centres_x = generator.integers(0, width, size=count)
    centres_y = generator.integers(0, height, size=count)
    darks = generator.random(count) < _SPECK_DARK_SHARE
    low, high = _SPECK_PEAK_JITTER
    peaks = sheet.speck_strength * generator.uniform(low, high, size=count)
    for index in range(count):
        centre_x = int(centres_x[index])
        centre_y = int(centres_y[index])
        x0 = max(0, centre_x - extent)
        y0 = max(0, centre_y - extent)
        x1 = min(width, centre_x + extent + 1)
        y1 = min(height, centre_y + extent + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        # Crop the profile the same way the region was cropped, so a speck at
        # the page edge is the same blot with part of it off the sheet.
        window = profile[
            y0 - (centre_y - extent) : y1 - (centre_y - extent),
            x0 - (centre_x - extent) : x1 - (centre_x - extent),
        ]
        alpha = (window * float(peaks[index]))[:, :, None]
        target = 0.0 if darks[index] else 255.0
        region = array[y0:y1, x0:x1].astype(np.float32)
        array[y0:y1, x0:x1] = np.clip(
            region * (1.0 - alpha) + target * alpha, 0.0, 255.0
        ).astype(np.uint8)


def _add_noise(image: Image.Image, sheet: SheetProfile, rng: Random) -> Image.Image:
    """Add sensor noise to a page, band by band.

    Gaussian noise is drawn once per band as a single channel and added to all
    three, because a scanner's dominant noise is luminance noise; independent
    per-channel noise reads as colour speckle no CCD produces at these levels.

    The banding is a memory decision, not a visual one: a widened copy of a
    whole A3 page at 300 DPI runs to hundreds of megabytes, and 20k samples of
    that in a batch loop is how a generator gets itself killed. Specks are
    stamped afterwards, over the whole page rather than per band, because a
    blot straddling a band boundary must not be cut in half.

    Args:
        image: The page in ``RGB``.
        sheet: The sheet's profile.
        rng: The sheet's seeded RNG, used to seed the array generator so the
            noise reproduces with the sample.

    Returns:
        A new ``RGB`` image, or the input when this sheet drew no noise.
    """
    specks = _speck_count(image.width, image.height, sheet)
    if sheet.sigma < 0.05 and specks == 0:
        return image
    generator = np.random.default_rng(rng.getrandbits(64))
    array = np.asarray(image, dtype=np.uint8)
    output = np.empty_like(array)
    height = array.shape[0]
    for start in range(0, height, _NOISE_BAND_ROWS):
        stop = min(height, start + _NOISE_BAND_ROWS)
        band = array[start:stop].astype(np.int16)
        if sheet.sigma >= 0.05:
            grain = generator.normal(
                0.0, sheet.sigma, size=band.shape[:2]
            ).astype(np.int16)
            band += grain[:, :, None]
        output[start:stop] = np.clip(band, 0, 255).astype(np.uint8)
    _add_specks(output, sheet, generator)
    return Image.fromarray(output, mode="RGB")


def _finish(image: Image.Image, sheet: SheetProfile) -> Image.Image:
    """Apply the output pass a scanner's own software performs.

    Args:
        image: The noisy page in ``RGB``.
        sheet: The sheet's profile.

    Returns:
        A new ``RGB`` image. Greyscale and 1-bit results are converted back to
        three channels so every artifact in the corpus has the same shape - the
        loss of colour is in the pixels, not in the file format.
    """
    result = image
    if sheet.mode == "gray":
        result = ImageOps.grayscale(result).convert("RGB")
    elif sheet.mode == "bilevel":
        # A fax-grade "B&W" pass. Error diffusion, not a hard threshold: a hard
        # threshold at these gammas eats the lightest strokes outright.
        result = ImageOps.grayscale(result).convert(
            "1", dither=Image.Dither.FLOYDSTEINBERG
        ).convert("RGB")
    if sheet.resample_scale < 0.995:
        # Scanned below nominal resolution and scaled back up, as a cheap
        # scanner's "draft" mode does.
        small = (
            max(1, round(result.width * sheet.resample_scale)),
            max(1, round(result.height * sheet.resample_scale)),
        )
        result = result.resize(small, Image.Resampling.BILINEAR).resize(
            (result.width, result.height), Image.Resampling.BICUBIC
        )
    if sheet.jpeg_quality is not None:
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=sheet.jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            result = decoded.convert("RGB")
    return result


def _fit(page: Image.Image, size: tuple[int, int] | None) -> Image.Image:
    """Trim or pad a capture to the size the annotation was measured against.

    Chromium rounds an element screenshot outwards, so a sheet whose device
    width works out to 1240.16 px is captured 1241 px wide while every box was
    scaled against 1240. One pixel of paper edge is nothing to look at, but an
    image whose dimensions disagree with its own ground truth is a defect, so
    the capture is squared up here - at the far edge, where the coordinate
    origin is untouched.

    Args:
        page: The decoded capture in ``RGBA``.
        size: The size the renderer promised, or ``None`` to accept the capture.

    Returns:
        The capture at exactly ``size``. Padding is transparent, so a short
        capture gains paper rather than a black band.
    """
    if size is None or page.size == size:
        return page
    fitted = Image.new("RGBA", size, (0, 0, 0, 0))
    fitted.paste(page.crop((0, 0, min(page.width, size[0]), min(page.height, size[1]))))
    return fitted


def apply_scan(
    png: bytes,
    paper: PaperProfile,
    sheet: SheetProfile,
    rng: Random,
    size: tuple[int, int] | None = None,
    image_format: ImageFormatConfig | None = None,
) -> tuple[bytes, SheetGeometry]:
    """Put one rendered sheet through the scanner.

    Args:
        png: The sheet as captured by Chromium: ``RGBA`` with a transparent
            ground, so light areas of the letter carry no ink.
        paper: The document's paper.
        sheet: This sheet's profile.
        rng: The sheet's seeded RNG, for the noise arrays.
        size: Device size the boxes were measured against. Passing it makes the
            output match the annotation exactly; ``None`` keeps the capture's
            own size.
        image_format: How to encode the result. Defaults to PNG. Note this is
            the *corpus* format and is unrelated to ``sheet.jpeg_quality``,
            which is a simulated scanner artifact applied inside the pipeline
            whatever the output format.

    Returns:
        ``(image_bytes, geometry)`` - the degraded sheet re-encoded in the
        configured format, and the geometric map it went through so the caller
        can move the bounding boxes with it.

    Raises:
        OSError: If the input bytes are not a decodable image.
    """
    with Image.open(io.BytesIO(png)) as opened:
        page = _fit(opened.convert("RGBA"), size)
    size = page.size

    ink = _soften_ink(page, sheet)
    ghost = _bleed_through(ink, sheet)

    geometry = SheetGeometry(
        size[0],
        size[1],
        rotation_deg=sheet.rotation_deg,
        perspective=sheet.perspective,
        warp=sheet.warp,
        barrel=sheet.barrel,
    )
    if not geometry.is_identity:
        # The ink is warped and the paper is not. A sheet physically carries its
        # grain with it, but grain is stochastic - nobody can see that it stayed
        # put - and warping only the ink layer keeps the exposed corners filled
        # with paper instead of with holes.
        mesh = geometry.mesh()
        ink = ink.transform(
            size, Image.Transform.MESH, mesh, resample=Image.Resampling.BICUBIC
        )
        if ghost is not None:
            ghost = ghost.transform(
                size, Image.Transform.MESH, mesh, resample=Image.Resampling.BICUBIC
            )

    composite = build_paper(paper, size).convert("RGBA")
    if ghost is not None:
        composite = Image.alpha_composite(composite, ghost)
    flattened = Image.alpha_composite(composite, ink).convert("RGB")

    field = _light_field(size, sheet)
    if field is not None:
        flattened = ImageChops.multiply(flattened, field.convert("RGB"))
    flattened = _apply_tone(flattened, sheet)
    flattened = _add_noise(flattened, sheet, rng)
    flattened = _finish(flattened, sheet)

    return encode_image(flattened, image_format or _DEFAULT_IMAGE_FORMAT), geometry


def flatten_clean(
    png: bytes,
    surface: str = "#FDFBF6",
    size: tuple[int, int] | None = None,
    image_format: ImageFormatConfig | None = None,
) -> bytes:
    """Composite an unscanned sheet onto plain paper.

    The clean branch of the corpus still cannot ship the raw capture: it is
    ``RGBA`` with a transparent ground, and a consumer that drops the alpha
    channel would be handed black paper. This gives those samples the same
    opaque ``RGB`` shape as the scanned ones.

    Args:
        png: The captured sheet in ``RGBA``.
        surface: Paper tone to composite onto, as ``#rrggbb``.
        size: Device size the boxes were measured against, as in
            :func:`apply_scan`.
        image_format: How to encode the result. Defaults to PNG.

    Returns:
        The flattened sheet, encoded in the configured format.

    Raises:
        OSError: If the input bytes are not a decodable image.
    """
    with Image.open(io.BytesIO(png)) as opened:
        page = _fit(opened.convert("RGBA"), size)
    base = Image.new("RGBA", page.size, _hex_to_rgb(surface) + (255,))
    flattened = Image.alpha_composite(base, page).convert("RGB")
    return encode_image(flattened, image_format or _DEFAULT_IMAGE_FORMAT)


