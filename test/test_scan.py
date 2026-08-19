"""Unit tests for the scanner simulation: config ranges, geometry, pixels.

No browser is needed here. The scan stage takes an RGBA capture and returns
degraded bytes, so a hand-drawn page stands in for Chromium's output and every
property that matters is checkable in-process:

* the ranges parse, and reject a malformed config rather than silently clamping
  it,
* the geometry maps both ways consistently, which is what keeps a bounding box
  on top of the ink it describes,
* the paper stays *paper* - a scan of a printed page is off-white with dark grey
  ink, never a grey page with black ink,
* and the same seed produces the same bytes, per CLAUDE.md 8.
"""

from __future__ import annotations

import io
import math
from dataclasses import replace as dataclass_replace
from pathlib import Path
from random import Random

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageStat

from fa_ocr_lora.config import (
    ConfigError,
    ImageFormatConfig,
    Range,
    ScanConfig,
    load_config,
)
from fa_ocr_lora.dataset.common.schema import (
    BBox,
    DocumentAnnotation,
    Element,
    ElementType,
    Page,
)
from fa_ocr_lora.dataset.synth.annotation import remap_boxes
from fa_ocr_lora.dataset.synth.layout.texture import Texture, available_textures
from fa_ocr_lora.dataset.synth.render.encode import decode_image, encode_image
from fa_ocr_lora.dataset.synth.render.scan import (
    PaperProfile,
    SheetGeometry,
    SheetProfile,
    apply_scan,
    build_paper,
    build_scan_profile,
    flatten_clean,
)

#: A page at 150 DPI: big enough that the banded noise path runs more than one
#: band, small enough that a full sweep stays fast.
PAGE_SIZE = (1240, 1754)

#: A solid slab of ink, and a region safely inside it. The margin matters: the
#: geometry moves the ink by up to ~15 px at this size, so a measurement window
#: hugging the edge would sample paper and read as "the ink got lighter".
INK_SLAB = (150, 180, 1090, 340)
INK = (350, 240, 890, 290)

#: A region of :func:`_capture` no ink ever reaches, i.e. bare paper.
BLANK = (250, 1300, 990, 1650)


def _capture() -> bytes:
    """Draw a stand-in for Chromium's capture: dark ink on a transparent ground.

    Returns:
        PNG bytes in ``RGBA``. Ink occupies the top half only, so :data:`BLANK`
        is guaranteed bare however the geometry moves the page.
    """
    page = Image.new("RGBA", PAGE_SIZE, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(page)
    drawing.rectangle(INK_SLAB, fill=(24, 24, 30, 242))
    for row in range(16):
        top = 400 + row * 44
        drawing.rectangle(
            (150, top, 1090 - (row % 5) * 90, top + 15), fill=(28, 28, 34, 240)
        )
    buffer = io.BytesIO()
    page.save(buffer, format="PNG")
    return buffer.getvalue()


def _mean(png: bytes, box: tuple[int, int, int, int]) -> float:
    """Return the mean grey level inside one region of an encoded page."""
    with Image.open(io.BytesIO(png)) as image:
        return ImageStat.Stat(image.convert("L").crop(box)).mean[0]


def _paper(**overrides: object) -> PaperProfile:
    """Build a paper profile, defaulting to no scan behind the tone."""
    base = {
        "tint": "#FBF8F1",
        "texture": None,
        "strength": 0.4,
        "grayscale": False,
        "autocontrast": False,
        "brightness": 1.0,
    }
    base.update(overrides)
    return PaperProfile(**base)  # type: ignore[arg-type]


def _sheet(**overrides: object) -> SheetProfile:
    """Build a sheet profile with every effect off, then apply ``overrides``.

    A neutral baseline is what makes a per-effect assertion possible: with the
    lighting, noise and output passes at identity, a measurement isolates the one
    stage under test instead of the sum of eighteen random draws.
    """
    base = {
        "blur": 0.0,
        "lighten": 0.0,
        "bleed": 0.0,
        "dilate": False,
        "rotation_deg": 0.0,
        "perspective": (0.0, 0.0, 0.0, 0.0),
        "warp": 0.0,
        "barrel": 0.0,
        "illumination": 0.0,
        "light_angle": 0.0,
        "vignette": 0.0,
        "shadow_edge": "",
        "shadow_strength": 0.0,
        "gamma": 1.0,
        "brightness": 1.0,
        "contrast": 1.0,
        "sigma": 0.0,
        "speck_density": 0.0,
        "speck_radius": 1.0,
        "speck_strength": 0.0,
        "jpeg_quality": None,
        "resample_scale": 1.0,
        "mode": "color",
    }
    base.update(overrides)
    return SheetProfile(**base)  # type: ignore[arg-type]


def _geometry(**overrides: object) -> SheetGeometry:
    """Build a sheet geometry over :data:`PAGE_SIZE` with identity defaults."""
    base = {
        "rotation_deg": 0.0,
        "perspective": (0.0, 0.0, 0.0, 0.0),
        "warp": 0.0,
        "barrel": 0.0,
    }
    base.update(overrides)
    return SheetGeometry(PAGE_SIZE[0], PAGE_SIZE[1], **base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# config.py: Range and ScanConfig
# --------------------------------------------------------------------------


def test_range_parses_scalars_mappings_and_none() -> None:
    default = Range(0.0, 1.0)
    assert Range.parse(None, "scan.x", default) == default
    # A bare number pins the range, which is how a config turns one knob off.
    assert Range.parse(0.5, "scan.x", default) == Range(0.5, 0.5)
    assert Range.parse({"min": 1, "max": 2}, "scan.x", default) == Range(1.0, 2.0)
    # A half-given mapping keeps the other end of the default.
    assert Range.parse({"max": 3}, "scan.x", default) == Range(0.0, 3.0)


def test_range_draws_inside_its_own_bounds() -> None:
    span = Range(-0.9, 0.9)
    rng = Random("range")
    assert all(-0.9 <= span.draw(rng) <= 0.9 for _ in range(500))
    # A pinned range is not random, but it still consumes a draw - that is what
    # keeps a retuned knob from shifting every later decision in the sample.
    before = rng.getstate()
    assert Range(2.0, 2.0).draw(rng) == 2.0
    assert rng.getstate() != before


@pytest.mark.parametrize(
    "payload",
    [
        {"min": 2, "max": 1},
        {"min": 0, "max": 1, "mid": 0.5},
        {"minimum": 0},
        {"min": "wide"},
        "0.5",
        [0.0, 1.0],
        True,
    ],
)
def test_range_rejects_malformed_payloads(payload: object) -> None:
    with pytest.raises(ConfigError):
        Range.parse(payload, "scan.x", Range(0.0, 1.0))


def test_scan_config_rejects_unknown_keys_and_sections() -> None:
    with pytest.raises(ConfigError):
        ScanConfig.from_dict({"paper": {"nonsense": 1}})
    with pytest.raises(ConfigError):
        ScanConfig.from_dict({"nonsense": {}})
    with pytest.raises(ConfigError):
        ScanConfig.from_dict({"ink": "heavy"})
    with pytest.raises(ConfigError):
        ScanConfig.from_dict({"probability": 1.4})
    with pytest.raises(ConfigError):
        # An empty tint list would leave the paper with no colour to draw.
        ScanConfig.from_dict({"paper": {"tints": []}})


def test_scan_config_maps_yaml_sections_onto_flat_fields() -> None:
    config = ScanConfig.from_dict(
        {
            "probability": 0.5,
            "paper": {"tints": ["#F0EEE8"], "strength": {"min": 0.2, "max": 0.3}},
            "geometry": {"rotation_deg": 0.0},
            "noise": {"bilevel": 0.0},
        }
    )
    assert config.probability == 0.5
    assert config.paper_tints == ("#F0EEE8",)
    assert config.texture_strength == Range(0.2, 0.3)
    assert config.rotation_deg == Range(0.0, 0.0)
    assert config.bilevel == 0.0
    # Unmentioned knobs keep their defaults rather than being zeroed.
    assert config.ink_blur == ScanConfig().ink_blur


def test_scan_config_reads_the_shipped_yaml() -> None:
    config = load_config().scan
    assert config.enabled
    assert 0.0 < config.probability <= 1.0
    assert config.paper_tints
    # Pure white is not a paper tone, and the whole RGBA composite depends on
    # that: a white sheet has nothing for the paper beneath it to show through.
    assert all(tint.upper() != "#FFFFFF" for tint in config.paper_tints)
    # Rotation must straddle zero, or every page in the corpus tilts one way.
    assert config.rotation_deg.minimum < 0.0 < config.rotation_deg.maximum


# --------------------------------------------------------------------------
# scan.py: profile drawing
# --------------------------------------------------------------------------


def test_profile_is_none_when_disabled_but_still_consumes_a_draw() -> None:
    assert build_scan_profile(ScanConfig(enabled=False), None, 1, Random("s")) is None
    assert build_scan_profile(ScanConfig(probability=0.0), None, 1, Random("s")) is None
    # A page count of zero has no sheet to degrade.
    assert build_scan_profile(ScanConfig(probability=1.0), None, 0, Random("s")) is None

    # The gate draw happens either way, so toggling the stage never shifts a
    # sample's other decisions.
    off, on = Random("s"), Random("s")
    build_scan_profile(ScanConfig(enabled=False), None, 1, off)
    build_scan_profile(ScanConfig(probability=0.0), None, 1, on)
    assert off.random() == on.random()


def test_profile_draws_one_paper_and_one_sheet_per_page() -> None:
    profile = build_scan_profile(ScanConfig(probability=1.0), None, 3, Random("p"))
    assert profile is not None
    assert len(profile.sheets) == 3
    # One letter, one sheet of paper: paper is a document-level decision, so it
    # is drawn once no matter how many pages the letter runs to.
    content = profile.as_content()
    assert content["paper"]["tint"] in ScanConfig().paper_tints
    assert len(content["sheets"]) == 3
    # Light and geometry are per-sheet, because each page crosses the platen
    # separately.
    assert profile.sheets[0] != profile.sheets[1]


def test_profile_stays_inside_the_configured_ranges() -> None:
    config = ScanConfig(probability=1.0)
    profile = build_scan_profile(config, None, 40, Random("bounds"))
    assert profile is not None
    for sheet in profile.sheets:
        assert config.ink_blur.minimum <= sheet.blur <= config.ink_blur.maximum
        assert (
            config.rotation_deg.minimum
            <= sheet.rotation_deg
            <= config.rotation_deg.maximum
        )
        assert config.barrel.minimum <= sheet.barrel <= config.barrel.maximum
        assert 0.0 <= min(sheet.perspective) <= max(sheet.perspective)
        assert max(sheet.perspective) <= config.perspective.maximum
        assert sheet.mode in {"color", "gray", "bilevel"}
        assert sheet.shadow_edge in {"", "top", "bottom", "left", "right"}
        assert sheet.jpeg_quality is None or 1 <= sheet.jpeg_quality <= 100


def test_profile_is_deterministic() -> None:
    config = ScanConfig(probability=1.0)
    first = build_scan_profile(config, None, 2, Random("seed:scan:4"))
    second = build_scan_profile(config, None, 2, Random("seed:scan:4"))
    assert first == second


# --------------------------------------------------------------------------
# scan.py: geometry
# --------------------------------------------------------------------------


def test_geometry_round_trips_between_both_directions() -> None:
    """The inverse map is what moves the boxes, so it has to close on itself."""
    geometry = _geometry(
        rotation_deg=0.8, perspective=(0.004, 0.001, 0.003, 0.0), warp=0.008, barrel=0.012
    )
    assert not geometry.is_identity
    for x, y in ((10.0, 20.0), (620.0, 877.0), (1230.0, 1740.0), (1000.0, 300.0)):
        mapped = geometry.destination_of(x, y)
        back_x, back_y = geometry.source_of(*mapped)
        assert abs(back_x - x) < 0.05
        assert abs(back_y - y) < 0.05


def test_identity_geometry_leaves_boxes_untouched() -> None:
    geometry = _geometry()
    assert geometry.is_identity
    assert geometry.map_bbox((10, 20, 30, 40)) == (10, 20, 30, 40)
    # Each knob on its own is enough to make the map non-trivial.
    assert not _geometry(rotation_deg=0.3).is_identity
    assert not _geometry(warp=0.005).is_identity
    assert not _geometry(barrel=0.01).is_identity
    assert not _geometry(perspective=(0.003, 0.0, 0.0, 0.0)).is_identity


def test_rotation_turns_the_page_about_its_centre() -> None:
    """Rotation must be a rotation: fixed centre, arc length, mirrored sign.

    A sign error in a transform is invisible in a thumbnail and fatal in ground
    truth, so the direction is pinned and not just the magnitude. The page is in
    screen coordinates with y downwards, so a positive angle turns it clockwise:
    a point above the centre swings to the right.
    """
    centre = (PAGE_SIZE[0] / 2.0, PAGE_SIZE[1] / 2.0)
    above = (centre[0], 200.0)

    clockwise = _geometry(rotation_deg=1.2).destination_of(*above)
    assert clockwise[0] > above[0]
    counter = _geometry(rotation_deg=-1.2).destination_of(*above)
    assert counter[0] < above[0]
    # Equal and opposite angles must displace by the same amount.
    assert abs((clockwise[0] - above[0]) + (counter[0] - above[0])) < 0.05

    # The centre of rotation does not move.
    fixed = _geometry(rotation_deg=1.2).destination_of(*centre)
    assert math.hypot(fixed[0] - centre[0], fixed[1] - centre[1]) < 0.05

    # And every other point travels its own radius times the angle.
    radius = math.hypot(above[0] - centre[0], above[1] - centre[1])
    travelled = math.hypot(clockwise[0] - above[0], clockwise[1] - above[1])
    assert abs(travelled - radius * math.radians(1.2)) < 0.5


def test_mapped_boxes_move_and_stay_on_the_page() -> None:
    geometry = _geometry(
        rotation_deg=-0.7,
        perspective=(0.005, 0.002, 0.004, 0.001),
        warp=0.009,
        barrel=-0.01,
    )
    box = (300, 1200, 900, 1260)
    x0, y0, x1, y1 = geometry.map_bbox(box)
    assert (x0, y0, x1, y1) != box
    assert 0 <= x0 < x1 <= PAGE_SIZE[0]
    assert 0 <= y0 < y1 <= PAGE_SIZE[1]
    # A rotated rectangle is not a rectangle, so the axis-aligned hull of its
    # four mapped corners can only grow. It must not grow far, though - a box
    # that doubled would stop describing the text inside it.
    assert (box[2] - box[0]) * 0.9 <= (x1 - x0) <= (box[2] - box[0]) * 1.15
    assert (y1 - y0) >= (box[3] - box[1])


def test_mapped_boxes_are_clipped_to_the_page() -> None:
    """Ground truth may not point off the raster, however hard the map pushes."""
    geometry = _geometry(rotation_deg=0.9, perspective=(0.006, 0.006, 0.006, 0.006))
    x0, y0, x1, y1 = geometry.map_bbox((0, 0, PAGE_SIZE[0], PAGE_SIZE[1]))
    assert (x0, y0) >= (0, 0)
    assert x1 <= PAGE_SIZE[0]
    assert y1 <= PAGE_SIZE[1]


def test_mesh_tiles_the_destination_exactly() -> None:
    geometry = SheetGeometry(
        800,
        1000,
        rotation_deg=0.5,
        perspective=(0.002, 0.0, 0.001, 0.0),
        warp=0.004,
        barrel=0.0,
    )
    mesh = geometry.mesh()
    assert mesh
    covered = sum(
        (right - left) * (bottom - top) for (left, top, right, bottom), _ in mesh
    )
    # A gap between quads would resample to a hole in the page.
    assert covered == 800 * 1000
    assert all(len(source) == 8 for _, source in mesh)


# --------------------------------------------------------------------------
# scan.py: pixels
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def textures() -> tuple[Texture, ...]:
    """Every prepared scan, or skip when the corpus is absent."""
    candidates = available_textures(load_config().paths.texture_dir)
    if not candidates:
        pytest.skip("texture corpus not present")
    probed: list[Texture] = []
    for path in candidates:
        with Image.open(path) as image:
            size = image.size
            image_format = (image.format or "").upper()
        probed.append(
            Texture(
                path=path,
                width=size[0],
                height=size[1],
                image_format=image_format,
                rotate_180=False,
            )
        )
    return tuple(probed)


def test_paper_keeps_its_tone_whatever_the_scan_exposure(
    textures: tuple[Texture, ...],
) -> None:
    """Flat-fielding is what stops a dark photograph greying out the page.

    The prepared scans are raw photographs with arbitrary exposure - several are
    mid-grey overall. Multiplied onto the tone directly they turned the paper to
    slate, so each channel is rescaled to its own mean and only the variation
    survives. Every asset is checked, because the failure was asset-dependent:
    it only showed on the dark ones.
    """
    for texture in textures:
        for strength in (0.18, 0.55):
            paper = build_paper(
                _paper(texture=texture, strength=strength, autocontrast=True),
                (620, 877),
            )
            level = ImageStat.Stat(paper.convert("L")).mean[0]
            assert 200.0 <= level <= 250.0, f"{texture.name} at {strength}: {level}"


def test_paper_grain_is_visible_and_scales_with_strength(
    textures: tuple[Texture, ...],
) -> None:
    """Strength has to buy grain depth - otherwise the knob does nothing."""
    profile = _paper(texture=textures[0], autocontrast=True)
    faint = ImageStat.Stat(
        build_paper(dataclass_replace(profile, strength=0.1), (620, 877)).convert("L")
    ).stddev[0]
    strong = ImageStat.Stat(
        build_paper(dataclass_replace(profile, strength=0.6), (620, 877)).convert("L")
    ).stddev[0]
    assert strong > faint > 0.0


def test_paper_without_a_texture_is_the_flat_tint() -> None:
    paper = build_paper(_paper(tint="#FBF8F1", strength=0.5), (64, 64))
    assert paper.mode == "RGB"
    assert paper.getextrema() == ((251, 251), (248, 248), (241, 241))
    # A zero strength is the same as no texture at all.
    assert build_paper(_paper(strength=0.0), (8, 8)).getextrema() == (
        (251, 251),
        (248, 248),
        (241, 241),
    )


def test_ink_is_soft_dark_grey_on_light_paper(textures: tuple[Texture, ...]) -> None:
    """The two properties this stage exists for, measured directly.

    With the lighting and noise passes at identity, what is left is the paper
    build and the ink softening - so this pins the request that font colour is
    not pure black and that the page it sits on is not white.
    """
    png, _ = apply_scan(
        _capture(),
        _paper(texture=textures[0], autocontrast=True),
        _sheet(blur=0.9, lighten=0.12),
        Random("ink"),
    )
    paper_level = _mean(png, BLANK)
    ink_level = _mean(png, INK)
    # The exact paper band is pinned per asset by the build_paper test above;
    # here it only has to be paper rather than page.
    assert paper_level > 195.0, f"paper is not off-white: {paper_level}"
    # Neither pure black nor washed out: a real scan's histogram has ink well
    # clear of zero and still far from the paper.
    assert 30.0 < ink_level < 110.0, f"ink is not soft dark grey: {ink_level}"
    assert paper_level - ink_level > 120.0

    # The blur has to soften the *edge*, which is the whole point of the pass.
    with Image.open(io.BytesIO(png)) as image:
        edge = image.convert("L").crop((600, INK_SLAB[3] - 4, 700, INK_SLAB[3] + 4))
    assert 60.0 < ImageStat.Stat(edge).mean[0] < 220.0


def test_full_random_profiles_keep_the_page_readable(
    textures: tuple[Texture, ...],
) -> None:
    """Every knob live at once must still leave a legible page.

    This is the regression guard for the two darkening bugs: raw texture
    exposure multiplied onto the tone, and a light field whose three terms only
    ever subtracted. Both showed up as a page that measured grey end to end.

    The thresholds come from a 120-profile sweep across the whole texture corpus,
    which floored at paper 178, ink 82 and a separation of 111. The twelve seeds
    below are fixed, so this test is a bound, not a sample.
    """
    config = ScanConfig(probability=1.0)
    capture = _capture()
    for index in range(12):
        texture = textures[index % len(textures)]
        profile = build_scan_profile(config, texture, 1, Random(f"px:{index}"))
        assert profile is not None
        sheet = profile.sheets[0]
        png, _ = apply_scan(capture, profile.paper, sheet, Random(f"noise:{index}"))
        paper_level = _mean(png, BLANK)
        ink_level = _mean(png, INK)
        assert paper_level > 170.0, f"page {index} turned grey: {paper_level}"
        assert ink_level < 150.0, f"page {index} lost its ink: {ink_level}"
        # Contrast is what an OCR model actually needs to survive.
        assert paper_level - ink_level > 70.0, f"page {index} is a smudge"


def test_scan_output_is_opaque_rgb_at_the_requested_size(
    textures: tuple[Texture, ...],
) -> None:
    png, _ = apply_scan(
        _capture(),
        _paper(texture=textures[0]),
        _sheet(rotation_deg=0.6, mode="gray"),
        Random("rgb"),
        PAGE_SIZE,
    )
    with Image.open(io.BytesIO(png)) as image:
        assert image.mode == "RGB"
        assert image.size == PAGE_SIZE
        # Warping the ink alone is what keeps the corners paper rather than
        # holes: a rotated page must not open a transparent - or black - wedge.
        corners = [
            image.convert("L").getpixel(point)
            for point in ((2, 2), (PAGE_SIZE[0] - 3, 2), (2, PAGE_SIZE[1] - 3))
        ]
    assert all(level > 150 for level in corners), corners


def test_scan_squares_up_a_capture_chromium_rounded_outwards() -> None:
    """Chromium rounds an element screenshot up; the annotation must still fit."""
    oversized = Image.new("RGBA", (PAGE_SIZE[0] + 1, PAGE_SIZE[1] + 1), (0, 0, 0, 40))
    buffer = io.BytesIO()
    oversized.save(buffer, format="PNG")
    flattened = flatten_clean(buffer.getvalue(), "#FBF8F1", PAGE_SIZE)
    with Image.open(io.BytesIO(flattened)) as image:
        assert image.size == PAGE_SIZE
        assert image.mode == "RGB"

    png, _ = apply_scan(
        buffer.getvalue(), _paper(), _sheet(), Random("fit"), PAGE_SIZE
    )
    with Image.open(io.BytesIO(png)) as image:
        assert image.size == PAGE_SIZE


def test_scan_is_deterministic(textures: tuple[Texture, ...]) -> None:
    profile = build_scan_profile(
        ScanConfig(probability=1.0), textures[0], 1, Random("det")
    )
    assert profile is not None
    capture = _capture()
    first, geometry = apply_scan(
        capture, profile.paper, profile.sheets[0], Random("det:0")
    )
    second, again = apply_scan(
        capture, profile.paper, profile.sheets[0], Random("det:0")
    )
    assert first == second
    assert geometry.map_bbox((10, 20, 300, 60)) == again.map_bbox((10, 20, 300, 60))


def test_noise_changes_the_pixels_without_moving_the_page() -> None:
    clean, _ = apply_scan(_capture(), _paper(), _sheet(), Random("q"))
    noisy, _ = apply_scan(_capture(), _paper(), _sheet(sigma=3.0), Random("q"))
    assert clean != noisy
    # Noise is zero-mean, so it must not shift the paper's level.
    assert abs(_mean(clean, BLANK) - _mean(noisy, BLANK)) < 3.0


def _levels(png: bytes) -> "np.ndarray":
    """Return an encoded page as a 2-D array of grey levels."""
    with Image.open(io.BytesIO(png)) as image:
        return np.asarray(image.convert("L"), dtype=np.int16)


def _changed(clean: bytes, dirty: bytes, *, threshold: int = 6) -> "np.ndarray":
    """Return the mask of pixels one effect moved by more than ``threshold``."""
    return np.abs(_levels(dirty) - _levels(clean)) > threshold


def test_specks_are_soft_blots_not_single_flipped_pixels() -> None:
    """The point of the speck pass, asserted directly.

    Salt-and-pepper noise fails both halves of this: it flips lone pixels, and
    it flips them to pure black or pure white. Neither happens on a real platen,
    so both are checked rather than trusted.
    """
    sheet = _sheet(speck_density=9.0, speck_radius=1.4, speck_strength=0.6)
    clean, _ = apply_scan(_capture(), _paper(), _sheet(), Random("dust"))
    specked, _ = apply_scan(_capture(), _paper(), sheet, Random("dust"))
    changed = _changed(clean, specked)
    assert changed.any(), "the sheet asked for specks and got none"

    # Every changed pixel touches another changed pixel: a speck is a blot with
    # a falloff, so it cannot exist as an isolated pixel.
    neighbours = np.zeros_like(changed)
    neighbours[1:, :] |= changed[:-1, :]
    neighbours[:-1, :] |= changed[1:, :]
    neighbours[:, 1:] |= changed[:, :-1]
    neighbours[:, :-1] |= changed[:, 1:]
    isolated = int(np.count_nonzero(changed & ~neighbours))
    assert isolated == 0, f"{isolated} speck pixel(s) stand alone"

    # And no speck punches through to an extreme: it tints the paper.
    levels = _levels(specked)
    assert int(levels.min()) > 0
    assert int(levels.max()) < 255


def test_speck_density_is_low_and_scales_with_the_setting() -> None:
    clean, _ = apply_scan(_capture(), _paper(), _sheet(), Random("dust"))
    light, _ = apply_scan(
        _capture(),
        _paper(),
        _sheet(speck_density=4.0, speck_radius=1.2, speck_strength=0.5),
        Random("dust"),
    )
    heavy, _ = apply_scan(
        _capture(),
        _paper(),
        _sheet(speck_density=40.0, speck_radius=1.2, speck_strength=0.5),
        Random("dust"),
    )
    few = int(np.count_nonzero(_changed(clean, light)))
    many = int(np.count_nonzero(_changed(clean, heavy)))
    assert 0 < few < many
    # At the top of the shipped range the dust covers a fraction of a percent of
    # the page. This is the "very low" part of the requirement, pinned.
    assert few / _levels(clean).size < 0.005


def test_zero_speck_density_leaves_the_page_untouched() -> None:
    clean, _ = apply_scan(_capture(), _paper(), _sheet(), Random("dust"))
    off, _ = apply_scan(
        _capture(),
        _paper(),
        _sheet(speck_density=0.0, speck_strength=0.6),
        Random("dust"),
    )
    assert clean == off


def test_speck_settings_reach_the_provenance_block() -> None:
    content = _sheet(
        speck_density=3.0, speck_radius=1.5, speck_strength=0.4
    ).as_content()
    assert content["speck_density"] == 3.0
    assert content["speck_radius"] == 1.5
    assert content["speck_strength"] == 0.4


def test_retired_salt_pepper_key_names_its_replacement() -> None:
    with pytest.raises(ConfigError) as error:
        ScanConfig.from_dict({"noise": {"salt_pepper": {"min": 0.0, "max": 0.01}}})
    message = str(error.value)
    assert "salt_pepper" in message
    assert "speck_density" in message


def test_flatten_clean_gives_the_unscanned_branch_opaque_paper() -> None:
    flattened = flatten_clean(_capture(), "#FAF9F5")
    with Image.open(io.BytesIO(flattened)) as image:
        assert image.mode == "RGB"
        assert image.size == PAGE_SIZE
    # The transparent ground became paper, not black.
    assert _mean(flattened, BLANK) > 240.0
    # And the ink came through as ink: a clean render is still dark on light.
    assert _mean(flattened, INK) < 60.0


def test_undecodable_bytes_are_reported_not_silently_skipped() -> None:
    with pytest.raises(OSError):
        flatten_clean(b"not an image at all")
    with pytest.raises(OSError):
        apply_scan(b"not an image at all", _paper(), _sheet(), Random("bad"))


# --------------------------------------------------------------------------
# encode.py: the format the corpus is written in
# --------------------------------------------------------------------------


#: Every shipped format, with the header Pillow reads back out of the bytes.
_FORMATS = [
    ("png", "PNG"),
    ("jpeg", "JPEG"),
    ("webp", "WEBP"),
]


@pytest.mark.parametrize(("spelling", "header"), _FORMATS)
def test_scan_branch_writes_the_requested_format(
    spelling: str, header: str, textures: tuple[Texture, ...]
) -> None:
    """The scanned exit encodes as configured, at the size it was asked for."""
    spec = ImageFormatConfig(format=spelling, quality=90)
    data, _ = apply_scan(
        _capture(),
        _paper(texture=textures[0]),
        _sheet(rotation_deg=0.4),
        Random(f"fmt:{spelling}"),
        PAGE_SIZE,
        spec,
    )
    with decode_image(data) as image:
        assert image.format == header
        assert image.mode == "RGB"
        assert image.size == PAGE_SIZE


@pytest.mark.parametrize(("spelling", "header"), _FORMATS)
def test_clean_branch_writes_the_requested_format(
    spelling: str, header: str
) -> None:
    """And so does the unscanned exit, which shares no code with it but encode."""
    spec = ImageFormatConfig(format=spelling, quality=90)
    data = flatten_clean(_capture(), "#FAF9F5", PAGE_SIZE, spec)
    with decode_image(data) as image:
        assert image.format == header
        assert image.mode == "RGB"
        assert image.size == PAGE_SIZE


@pytest.mark.parametrize(("spelling", "_header"), _FORMATS)
def test_every_format_keeps_the_page_readable(spelling: str, _header: str) -> None:
    """Compression must not cost the contrast the OCR model reads.

    This is the reason quality is a setting rather than left at Pillow's default
    75: at the shipped 92 a page survives, and this test is what would notice if
    a lower default were slipped in.
    """
    spec = ImageFormatConfig(format=spelling, quality=92)
    data = flatten_clean(_capture(), "#FAF9F5", PAGE_SIZE, spec)
    assert _mean(data, BLANK) > 240.0
    assert _mean(data, INK) < 70.0
    assert _mean(data, BLANK) - _mean(data, INK) > 170.0


def test_png_encoding_is_byte_identical_to_the_old_default() -> None:
    """The default format is still PNG, so an existing corpus does not move."""
    spec = ImageFormatConfig()
    assert spec.format == "png"
    assert spec.suffix == ".png"
    assert spec.is_lossless
    explicit = flatten_clean(_capture(), "#FAF9F5", PAGE_SIZE, spec)
    implicit = flatten_clean(_capture(), "#FAF9F5", PAGE_SIZE)
    assert explicit == implicit


def test_encoding_refuses_to_discard_a_transparent_ground() -> None:
    """A silent flatten onto white would bury the paper the pipeline composites.

    ``encode_image`` is handed the finished page, so an RGBA image reaching a
    format that cannot hold alpha means a caller skipped the compositing step.
    Erroring says so; flattening would ship a page with no texture and nobody
    would find out until the corpus was trained on.
    """
    page = Image.new("RGBA", (8, 8), (0, 0, 0, 40))
    with pytest.raises(ValueError, match="composite"):
        encode_image(page, ImageFormatConfig(format="jpeg"))
    # PNG and WebP hold alpha, so the same image is fine there.
    assert encode_image(page, ImageFormatConfig(format="png"))
    assert encode_image(page, ImageFormatConfig(format="webp"))


def test_jpeg_is_written_without_chroma_subsampling() -> None:
    """4:4:4, because 4:2:0 averages colour over the width of a diacritic."""
    assert ImageFormatConfig(format="jpeg").save_kwargs()["subsampling"] == 0


@pytest.mark.parametrize("spelling", ["gif", "tiff", "", "pngg"])
def test_an_unknown_format_is_rejected_at_load(spelling: str) -> None:
    with pytest.raises(ConfigError, match="render.image.format"):
        ImageFormatConfig(format=spelling)


@pytest.mark.parametrize("quality", [0, -1, 101])
def test_an_out_of_range_quality_is_rejected_at_load(quality: int) -> None:
    with pytest.raises(ConfigError, match="render.image.quality"):
        ImageFormatConfig(quality=quality)


def test_undecodable_bytes_are_reported_as_one_exception_type() -> None:
    """``decode_image`` normalises Pillow's own error, which callers cannot see."""
    with pytest.raises(ValueError, match="not a readable image"):
        decode_image(b"not an image at all")


# --------------------------------------------------------------------------
# annotation.py: remapping the ground truth
# --------------------------------------------------------------------------


def _annotation() -> DocumentAnnotation:
    """Build a one-page annotation with two boxes, for the remap tests."""
    return DocumentAnnotation(
        sample_id="synth_000000",
        source_sample_id="plaintext_000000",
        seed=1337,
        dpi=150,
        page_size="A4",
        orientation="portrait",
        pages=(
            Page(
                page_index=0,
                width=PAGE_SIZE[0],
                height=PAGE_SIZE[1],
                image_path="image/synth_000000_p01.png",
                elements=(
                    Element(
                        id="body.para0",
                        type=ElementType.PARAGRAPH,
                        bbox=BBox(150, 400, 1090, 460),
                        text="متن نخست",
                        reading_order=0,
                    ),
                    Element(
                        id="body.para1",
                        type=ElementType.PARAGRAPH,
                        bbox=BBox(150, 500, 1090, 560),
                        text="متن دوم",
                        reading_order=1,
                    ),
                ),
            ),
        ),
    )


def test_remap_moves_every_box_and_records_the_provenance() -> None:
    annotation = _annotation()
    remapped = remap_boxes(
        annotation,
        (
            _geometry(
                rotation_deg=0.9,
                perspective=(0.005, 0.001, 0.003, 0.0),
                warp=0.009,
                barrel=0.01,
            ),
        ),
        scan={"paper": {"tint": "#F9F5EC"}},
    )
    assert remapped is not annotation
    assert remapped.scan == {"paper": {"tint": "#F9F5EC"}}
    before = annotation.pages[0].elements
    after = remapped.pages[0].elements
    assert len(after) == len(before)
    for original, moved in zip(before, after, strict=True):
        assert moved.bbox.as_list() != original.bbox.as_list()
        assert moved.id == original.id
        assert moved.text == original.text
        # Reading order is computed on the exactly-horizontal clean page and
        # deliberately not recomputed: a sub-degree tilt cannot reorder a letter,
        # and re-deriving it from tilted boxes could.
        assert moved.reading_order == original.reading_order
        assert moved.bbox.within(PAGE_SIZE[0], PAGE_SIZE[1])
    assert remapped.full_transcription == annotation.full_transcription
    # The original is untouched, so a caller can keep the clean ground truth.
    assert annotation.pages[0].elements[0].bbox.as_list() == [150, 400, 1090, 460]


def test_remap_leaves_untransformed_pages_alone() -> None:
    annotation = _annotation()
    assert remap_boxes(annotation, (None,)).pages == annotation.pages
    # A short sequence is not an error: it means the later sheets did not move.
    assert remap_boxes(annotation, ()).pages == annotation.pages
    # And a page that did not move keeps its scan provenance from the document.
    assert remap_boxes(annotation, (None,), scan=None).scan is None


def test_remap_round_trips_through_the_schema() -> None:
    remapped = remap_boxes(_annotation(), (None,), scan={"paper": {"tint": "#FBF8F1"}})
    restored = DocumentAnnotation.from_dict(remapped.as_dict())
    assert restored == remapped
    assert restored.scan == {"paper": {"tint": "#FBF8F1"}}
    # A born-digital page carries no scan block at all.
    assert DocumentAnnotation.from_dict(_annotation().as_dict()).scan is None


def test_texture_corpus_is_readable() -> None:
    """A guard on the asset the whole paper stage depends on."""
    directory = load_config().paths.texture_dir
    assert isinstance(directory, Path)
    if not directory.is_dir():
        pytest.skip("texture corpus not present")
    assert available_textures(directory), "no decodable scan in data/texture"
