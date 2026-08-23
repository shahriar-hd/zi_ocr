"""Unit tests for the image geometry and VRAM budgeting behind the 4 GB run.

Both modules exist to make one claim true, and that claim is what is tested here:
**a size the geometry returns is a size the processor accepts unchanged.** If the
processor re-rounds what we hand it, the predicted token count is wrong, the
sequence budget in :mod:`fa_ocr_lora.training.encoding` is wrong with it, and the
first symptom is an OOM 900 steps into a run rather than an error at startup.

No model and no GPU are involved: the processor's resize rule is arithmetic, and
:func:`~fa_ocr_lora.training.images.fit_pixel_budget` reimplements it so it can run
at manifest time. The tests assert the fixed-point property directly rather than
comparing against a loaded processor, so they run on a machine with no CUDA device.
"""

from __future__ import annotations

import pytest

from fa_ocr_lora.training.config import ImageConfig, ModelConfig
from fa_ocr_lora.training.images import (
    MAX_ASPECT_RATIO,
    PIXELS_PER_TOKEN,
    SIZE_FACTOR,
    aspect_ratio,
    fit_pixel_budget,
    image_tokens,
    is_feedable,
)
from fa_ocr_lora.training.vram import (
    ACTIVATION_PER_TOKEN_MIB,
    SAFETY_MIB,
    VramReport,
    banner,
    clamp_max_pixels,
    projected_peak_mib,
)

_MIN_PIXELS = 147_384
_MAX_PIXELS = 1_003_520

# Sizes actually present in the two corpora, plus the degenerate shapes that broke
# earlier versions of the rounding: a 1-px-tall sliver, an exact multiple, and a
# size one pixel under one.
_SIZES = [
    (1240, 1754),  # letters, A4 at 150 DPI
    (874, 1240),  # letters, the smaller page size
    (768, 276),  # persian_pixel strip
    (145, 40),  # a metadata_field crop
    (2480, 3508),  # A4 at 300 DPI
    (28, 28),
    (27, 27),
    (29, 29),
    (4000, 30),
    (30, 4000),
]


def test_pixels_per_token_is_the_processor_arithmetic():
    assert SIZE_FACTOR == 28
    assert PIXELS_PER_TOKEN == SIZE_FACTOR * SIZE_FACTOR == 784


@pytest.mark.parametrize(("width", "height"), _SIZES)
def test_fit_pixel_budget_returns_multiples_of_the_size_factor(width, height):
    fitted_w, fitted_h = fit_pixel_budget(
        width, height, min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS
    )
    assert fitted_w % SIZE_FACTOR == 0
    assert fitted_h % SIZE_FACTOR == 0
    assert fitted_w > 0 and fitted_h > 0


@pytest.mark.parametrize(("width", "height"), _SIZES)
def test_fit_pixel_budget_is_a_fixed_point(width, height):
    """Re-applying the rule must change nothing.

    This is the property the whole sequence budget rests on: the processor runs its
    own resize over whatever we give it, so our output has to be something its rule
    maps to itself.
    """
    once = fit_pixel_budget(width, height, min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS)
    twice = fit_pixel_budget(*once, min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS)
    assert twice == once


@pytest.mark.parametrize(("width", "height"), _SIZES)
def test_fit_pixel_budget_respects_the_lower_bound(width, height):
    fitted_w, fitted_h = fit_pixel_budget(
        width, height, min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS
    )
    assert fitted_w * fitted_h >= _MIN_PIXELS


@pytest.mark.parametrize(("width", "height"), _SIZES)
def test_fit_pixel_budget_respects_the_upper_bound(width, height):
    fitted_w, fitted_h = fit_pixel_budget(
        width, height, min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS
    )
    assert fitted_w * fitted_h <= _MAX_PIXELS


def test_narrow_bounds_favour_the_lower_one():
    """When the bounds cannot both hold, ``min_pixels`` wins.

    It is a property of the vision tower; ``max_pixels`` is only our own budget.
    An image under the tower's floor is not something the model can consume, so the
    documented resolution is to exceed the ceiling and let the caller check.
    """
    fitted_w, fitted_h = fit_pixel_budget(
        1240, 1754, min_pixels=_MIN_PIXELS, max_pixels=_MIN_PIXELS + 1
    )
    assert fitted_w * fitted_h >= _MIN_PIXELS


def test_fit_pixel_budget_rejects_degenerate_input():
    with pytest.raises(ValueError, match="must be positive"):
        fit_pixel_budget(0, 100, min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS)
    with pytest.raises(ValueError, match="exceeds max_pixels"):
        fit_pixel_budget(100, 100, min_pixels=_MAX_PIXELS, max_pixels=_MIN_PIXELS)


def test_image_tokens_counts_merged_patch_blocks():
    assert image_tokens(28, 28) == 1
    assert image_tokens(SIZE_FACTOR * 40, SIZE_FACTOR * 32) == 40 * 32
    # A budget of 1,003,520 px is exactly 1,280 tokens, which is why that number
    # was chosen: no remainder to reason about.
    assert _MAX_PIXELS // PIXELS_PER_TOKEN == 1280


def test_aspect_ratio_and_feedability():
    assert aspect_ratio(100, 50) == pytest.approx(2.0)
    assert aspect_ratio(50, 100) == pytest.approx(2.0)
    assert aspect_ratio(0, 100) == float("inf")
    assert is_feedable(768, 276)
    assert not is_feedable(0, 100)
    assert not is_feedable(MAX_ASPECT_RATIO * 10 + 10, 10)


def test_clamp_max_pixels_is_inert_without_a_shortfall():
    assert (
        clamp_max_pixels(_MAX_PIXELS, shortfall_mib=0.0, min_pixels=_MIN_PIXELS)
        == _MAX_PIXELS
    )
    assert (
        clamp_max_pixels(_MAX_PIXELS, shortfall_mib=-50.0, min_pixels=_MIN_PIXELS)
        == _MAX_PIXELS
    )


def test_clamp_max_pixels_gives_up_roughly_the_shortfall():
    shortfall = 100.0
    clamped = clamp_max_pixels(
        _MAX_PIXELS, shortfall_mib=shortfall, min_pixels=_MIN_PIXELS
    )
    assert clamped < _MAX_PIXELS
    assert clamped % PIXELS_PER_TOKEN == 0
    given_up = (_MAX_PIXELS - clamped) // PIXELS_PER_TOKEN
    # One token of slack: the implementation rounds up so the clamp always covers
    # the shortfall rather than landing a token short of it.
    expected = int(shortfall / ACTIVATION_PER_TOKEN_MIB) + 1
    assert given_up == expected


def test_clamp_max_pixels_never_goes_below_the_tower_floor():
    """A clamp that undercuts ``min_pixels`` would spend what it set out to save.

    The processor upscales anything below its floor, so the tokens given up here
    would come straight back - with a resample on top.
    """
    clamped = clamp_max_pixels(
        _MAX_PIXELS, shortfall_mib=100_000.0, min_pixels=_MIN_PIXELS
    )
    assert clamped >= _MIN_PIXELS


def test_projected_peak_grows_with_the_sequence_ceiling():
    model = ModelConfig.parse({})
    short = ImageConfig.parse({"max_total_seq_len": 1600})
    long = ImageConfig.parse({"max_total_seq_len": 3072})
    peak_short = projected_peak_mib(model, short)
    peak_long = projected_peak_mib(model, long)
    assert peak_long > peak_short
    # The slope is the measured one, not an invented margin.
    assert peak_long - peak_short == pytest.approx(
        ACTIVATION_PER_TOKEN_MIB * (3072 - 1600), abs=1.0
    )


def test_reserved_vram_overrides_the_whole_projection():
    """The escape hatch replaces the estimate rather than adding to it.

    Taking it as one term among several would make its meaning ambiguous on
    hardware the measured coefficients do not describe - which is the only reason
    to set it.
    """
    model = ModelConfig.parse({})
    image = ImageConfig.parse({"reserved_vram_mib": 1234})
    assert projected_peak_mib(model, image) == 1234.0


def test_report_fits_accounts_for_the_safety_margin():
    report = VramReport(
        available=True,
        total_mib=3746,
        free_mib=2000,
        projected_mib=1850.0,
        budget_mib=2000 - SAFETY_MIB,
        max_pixels=_MAX_PIXELS,
        clamped=False,
    )
    assert not report.fits
    assert report.image_tokens == 1280


def test_banner_names_the_largest_unpulled_lever():
    """An OOM warning has to say which knob to turn, in the order that pays.

    Quantising the frozen tower is worth 640 MiB - more than every other option
    combined - so it has to be the suggestion when it is still available.
    """
    model = ModelConfig.parse({"quantization": "nf4", "quantize_vision": False})
    report = VramReport(
        available=True,
        total_mib=3746,
        free_mib=1800,
        projected_mib=2400.0,
        budget_mib=1600,
        max_pixels=_MAX_PIXELS,
        clamped=False,
        holders=(("gnome-shell", 128),),
    )
    text = banner(report, model)
    assert "WARNING" in text
    assert "quantize_vision" in text
    assert "gnome-shell" in text


def test_banner_without_a_gpu_says_so_rather_than_printing_zeros():
    model = ModelConfig.parse({})
    report = VramReport(
        available=False,
        total_mib=0,
        free_mib=0,
        projected_mib=0.0,
        budget_mib=0,
        max_pixels=_MAX_PIXELS,
        clamped=False,
    )
    assert "no CUDA device" in banner(report, model)
    assert report.fits  # a CPU machine must not be reported as over budget
