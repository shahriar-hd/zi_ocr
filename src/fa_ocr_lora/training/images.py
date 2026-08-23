"""Image geometry: cropping, resizing, and the token arithmetic behind both.

PaddleOCR-VL has no fixed input resolution. It patches an image at 14 px, merges
2x2 patches into one visual token, and feeds however many tokens that produces -
so an image's cost is its *area*, and the only way to control VRAM is to control
area. Every number in this module exists to make that cost predictable before an
image is decoded.

The one property everything here protects: a size this module returns is a size
the processor will accept unchanged. The processor runs its own resize, rounding
each side to a multiple of ``patch_size * merge_size`` (28) and clamping the area
into ``[min_pixels, max_pixels]``. If we hand it sides that are already multiples
of 28 within those bounds, its rounding is the identity - so the token count we
predicted is the token count the model gets, and the sequence-length budget in
:mod:`fa_ocr_lora.training.encoding` is exact rather than hopeful.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

from PIL import Image

from .config import MERGE_SIZE, PATCH_SIZE, PIXELS_PER_TOKEN

__all__ = [
    "MAX_ASPECT_RATIO",
    "PIXELS_PER_TOKEN",
    "SIZE_FACTOR",
    "aspect_ratio",
    "fit_image",
    "fit_pixel_budget",
    "image_tokens",
    "is_feedable",
    "load_region",
    "prepare_image",
]

#: Both sides must be a multiple of this: 14 px patches merged 2x2.
SIZE_FACTOR: Final[int] = PATCH_SIZE * MERGE_SIZE

#: Longest:shortest side ratio the vision tower tolerates. A strip beyond this
#: cannot be resized into the pixel budget without collapsing one side below a
#: single patch row, so it is rejected rather than fed as mush.
MAX_ASPECT_RATIO: Final[int] = 200


def aspect_ratio(width: int, height: int) -> float:
    """Longest side divided by shortest side.

    Args:
        width: Width in pixels.
        height: Height in pixels.

    Returns:
        The ratio, always ``>= 1.0``; ``inf`` if either side is zero.
    """
    short, long = min(width, height), max(width, height)
    if short <= 0:
        return math.inf
    return long / short


def image_tokens(width: int, height: int) -> int:
    """Visual tokens an image of this size becomes.

    Assumes the size has already been through :func:`fit_pixel_budget`, which is
    what makes the floor division exact rather than approximate.

    Args:
        width: Width in pixels, ideally a multiple of :data:`SIZE_FACTOR`.
        height: Height in pixels, ideally a multiple of :data:`SIZE_FACTOR`.

    Returns:
        Token count, one per merged 2x2 block of 14 px patches.
    """
    return (width // SIZE_FACTOR) * (height // SIZE_FACTOR)


def is_feedable(width: int, height: int) -> bool:
    """Whether a region can be fed to the vision tower at all.

    Args:
        width: Width in pixels.
        height: Height in pixels.

    Returns:
        ``True`` when both sides are positive and the aspect ratio is within
        :data:`MAX_ASPECT_RATIO`.
    """
    if width <= 0 or height <= 0:
        return False
    return aspect_ratio(width, height) <= MAX_ASPECT_RATIO


def fit_pixel_budget(
    width: int, height: int, *, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Snap a size to multiples of 28 and into the pixel budget.

    This reimplements the processor's own ``smart_resize`` rather than calling
    it, because it must run at manifest time - before any model is loaded - to
    budget sequence length for 10,000 records. Reimplementing means the two must
    agree, which is why the rounding modes below are not interchangeable: the
    downscale branch floors (never exceed ``max_pixels``, the VRAM ceiling) and
    the upscale branch ceils (never fall under ``min_pixels``, the tower's
    floor).

    The processor applies its rule again to whatever we hand it, using *its*
    configured bounds, so a size is only safe if it is a fixed point of that
    rule: sides already multiples of 28, area already inside the bounds. The
    ``elif`` structure alone does not guarantee that. Flooring the downscale
    loses up to one 28-step per side, which is negligible when the bounds are
    far apart but can drop the area below ``min_pixels`` when a caller narrows
    ``max_pixels`` down towards it - and then the processor upscales, and the
    token count we predicted is wrong. Hence the final floor pass.

    When the two bounds genuinely conflict, ``min_pixels`` wins: it is a
    property of the vision tower, whereas ``max_pixels`` is our own budget, and
    an image below the tower's floor is not something the model can consume. A
    caller that must respect a hard ceiling has to check the returned area.

    Args:
        width: Source width in pixels.
        height: Source height in pixels.
        min_pixels: Lower area bound, the processor's own ``min_pixels``.
        max_pixels: Upper area bound, the run's VRAM-derived ceiling.

    Returns:
        ``(width, height)`` with both sides positive multiples of
        :data:`SIZE_FACTOR` and area at least ``min_pixels``, and at most
        ``max_pixels`` unless the bounds conflict.

    Raises:
        ValueError: If a side is not positive or the bounds are inverted.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    if min_pixels > max_pixels:
        raise ValueError(f"min_pixels {min_pixels} exceeds max_pixels {max_pixels}")

    scaled_w = max(SIZE_FACTOR, round(width / SIZE_FACTOR) * SIZE_FACTOR)
    scaled_h = max(SIZE_FACTOR, round(height / SIZE_FACTOR) * SIZE_FACTOR)

    if scaled_w * scaled_h > max_pixels:
        beta = math.sqrt((width * height) / max_pixels)
        scaled_w = max(SIZE_FACTOR, math.floor(width / beta / SIZE_FACTOR) * SIZE_FACTOR)
        scaled_h = max(SIZE_FACTOR, math.floor(height / beta / SIZE_FACTOR) * SIZE_FACTOR)
    elif scaled_w * scaled_h < min_pixels:
        beta = math.sqrt(min_pixels / (width * height))
        scaled_w = math.ceil(width * beta / SIZE_FACTOR) * SIZE_FACTOR
        scaled_h = math.ceil(height * beta / SIZE_FACTOR) * SIZE_FACTOR

    if scaled_w * scaled_h < min_pixels:
        # Recomputed from the floored size, not the source: it is the floored
        # size the processor would see, so this is the correction it would make.
        beta = math.sqrt(min_pixels / (scaled_w * scaled_h))
        scaled_w = math.ceil(scaled_w * beta / SIZE_FACTOR) * SIZE_FACTOR
        scaled_h = math.ceil(scaled_h * beta / SIZE_FACTOR) * SIZE_FACTOR

    return scaled_w, scaled_h


def load_region(
    image_path: Path, crop_box: tuple[int, int, int, int] | None
) -> Image.Image:
    """Open an image as RGB and crop it to a region.

    RGB conversion is unconditional: the corpora mix RGB pages, greyscale scans
    and palette PNGs, and the processor's normalisation expects three channels.

    Args:
        image_path: File to open.
        crop_box: ``(left, top, right, bottom)``, or ``None`` for the whole
            image.

    Returns:
        The cropped RGB image.

    Raises:
        OSError: If the file is missing or not a decodable image.
    """
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    if crop_box is not None:
        image = image.crop(crop_box)
    return image


def fit_image(
    image: Image.Image, *, min_pixels: int, max_pixels: int
) -> tuple[Image.Image, int]:
    """Resize an already-loaded image into a pixel budget.

    Args:
        image: RGB image to fit.
        min_pixels: Lower area bound.
        max_pixels: Upper area bound.

    Returns:
        ``(image, token_count)``. The input is returned unchanged when it already
        fits, so a Persian Pixel strip - already well under any budget we set -
        costs no resample at all.

    Raises:
        ValueError: If the image has a zero-length side.
    """
    target_w, target_h = fit_pixel_budget(
        image.width, image.height, min_pixels=min_pixels, max_pixels=max_pixels
    )
    if (target_w, target_h) != (image.width, image.height):
        # LANCZOS on the downscale path: these are text images, and a box or
        # bilinear filter smears thin Persian strokes badly enough to change what
        # the target transcription should have been.
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return image, image_tokens(target_w, target_h)


def prepare_image(
    image_path: Path,
    crop_box: tuple[int, int, int, int] | None,
    *,
    min_pixels: int,
    max_pixels: int,
) -> tuple[Image.Image, int]:
    """Load, crop, and resize one region to its final fed size.

    Resizing here rather than inside the processor is what keeps peak host memory
    flat: the processor would otherwise build a float32 patch tensor from the
    full-resolution page, which for a 2.17 MP A4 scan is ~26 MB per worker per
    sample before any downscale happens.

    Args:
        image_path: Page image on disk.
        crop_box: Region to crop before resizing, or ``None``.
        min_pixels: Lower area bound.
        max_pixels: Upper area bound.

    Returns:
        ``(image, token_count)`` - the resized RGB image and the number of
        visual tokens it will occupy in the sequence.

    Raises:
        OSError: If the image cannot be read.
        ValueError: If the region is degenerate.
    """
    image = load_region(image_path, crop_box)
    return fit_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
