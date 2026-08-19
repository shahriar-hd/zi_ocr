"""Paper selection: which prepared scan one document is printed on.

Half the corpus renders on untextured paper and half on one of the scans in
``data/texture``, so the adapter sees both clean digital pages and pages that
carry the grain, warmth and vignetting of real paper. The choice is one per
document, not one per sheet: a letter is printed on a single kind of paper.

The scan is **not** a CSS background. It is composited underneath the rendered
page by ``render/scan.py`` after rasterisation, for two reasons:

* The page itself is captured with a transparent ground, so light areas of the
  letter are an absence of ink rather than a coat of white. Only a real
  composite can let the paper show through them.
* The physical effects the scan stage applies - printer-style greyscale and
  autocontrast passes, uneven light, sensor noise - are pixel operations. CSS
  can lay an image behind text; it cannot expose that image to the rest of the
  degradation chain.

Two properties of the prepared assets shape this module:

* **Extensions lie.** ``page_4.jpg`` is really a WebP. The format is taken from
  the decoded header rather than the suffix, so a file is judged by its bytes.
* **Sizes span 190x265 to 3000x4000.** Every scan is resized to cover the
  sheet. Tiling the small ones was tried first, on the theory that a 190px scan
  upscaled 13x would smear into a gradient; the tiles instead laid a periodic
  rectangular seam across the page, which is exactly the kind of spurious
  regularity an OCR model will learn. The upscaled scan is soft but seamless,
  and soft grain is what a low-resolution scan of blank paper looks like anyway.

The texture is background only. It gets no bounding box, but its filename is
recorded in the annotation so a sample's appearance stays traceable to the
asset that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from random import Random
from typing import Final

from PIL import Image, UnidentifiedImageError

__all__ = [
    "Texture",
    "available_textures",
    "choose_texture",
    "load_texture",
    "background_css",
]

#: Formats Pillow can decode here. Keyed by what the header actually reports,
#: not by the file's extension. AVIF needs Pillow 11.3+ or ``pillow-avif``;
#: a build without it simply reports the file as unreadable and skips it.
_READABLE_FORMATS: Final[frozenset[str]] = frozenset(
    {"JPEG", "PNG", "WEBP", "AVIF", "GIF", "TIFF", "BMP"}
)


def _probe(path: Path) -> tuple[int, int, str] | None:
    """Read one texture's dimensions and true format from its header.

    Args:
        path: Image file to inspect.

    Returns:
        ``(width, height, format)``, or ``None`` when the file is unreadable,
        is not an image, or is in a format this build cannot decode.
    """
    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
    except (OSError, UnidentifiedImageError, ValueError):
        return None
    if image_format not in _READABLE_FORMATS or width <= 0 or height <= 0:
        return None
    return width, height, image_format


@lru_cache(maxsize=8)
def available_textures(directory: Path | str | None) -> tuple[Path, ...]:
    """List the usable texture files in one directory, in sorted order.

    Sorting matters: the pick is made by index from a seeded RNG, so the
    directory listing has to be stable for a sample to reproduce.

    Args:
        directory: Where the prepared scans live, or ``None`` to disable
            textures entirely.

    Returns:
        Every file whose header identifies it as a decodable image, sorted by
        name. Empty when the directory is absent.
    """
    if directory is None:
        return ()
    root = Path(directory).expanduser()
    if not root.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(root.iterdir())
        if path.is_file() and _probe(path) is not None
    )


@dataclass(frozen=True, slots=True)
class Texture:
    """One document's chosen paper.

    Attributes:
        path: The source scan on disk.
        width: Source width in pixels.
        height: Source height in pixels.
        image_format: True format, read from the header rather than the suffix.
        rotate_180: Whether the scan is flipped, so one asset does not always
            put its darkest corner in the same place.
    """

    path: Path
    width: int
    height: int
    image_format: str
    rotate_180: bool

    @property
    def name(self) -> str:
        """Filename recorded in the annotation for provenance."""
        return self.path.name

    def as_content(self) -> dict[str, object]:
        """Return the texture's provenance block for the annotation."""
        return {
            "name": self.name,
            "source_size": [self.width, self.height],
            "format": self.image_format,
            "rotate_180": self.rotate_180,
        }


@lru_cache(maxsize=6)
def load_texture(path: Path) -> Image.Image:
    """Decode one texture to RGB, cached because a batch reuses the same few.

    Args:
        path: Texture file to read.

    Returns:
        The decoded image in ``RGB`` mode. The returned object is shared with
        every other caller, so treat it as read-only and copy before mutating.

    Raises:
        OSError: If the file cannot be read or decoded.
    """
    with Image.open(path) as image:
        return image.convert("RGB")


def choose_texture(
    directory: Path | str | None,
    probability: float,
    rng: Random,
) -> Texture | None:
    """Pick the paper for one document, or none.

    Args:
        directory: Where the prepared scans live.
        probability: Chance of using a texture at all, in ``[0, 1]``.
        rng: The sample's seeded RNG, conventionally ``purpose="texture"``.

    Returns:
        The chosen texture, or ``None`` for a page with no paper scan behind
        it - which still carries the theme's paper tone, just no grain.
    """
    # Drawn unconditionally so the RNG advances identically either way, and a
    # sample's later choices do not shift when the texture rate is retuned.
    draw = rng.random()
    candidates = available_textures(directory)
    if not candidates or draw >= probability:
        return None

    path = candidates[rng.randrange(len(candidates))]
    probed = _probe(path)
    if probed is None:
        return None
    width, height, image_format = probed

    return Texture(
        path=path,
        width=width,
        height=height,
        image_format=image_format,
        rotate_180=rng.random() < 0.5,
    )


def background_css(surface_css: str) -> str:
    """Build the sheet's own ``background`` declarations.

    Only the translucent paper tone lands here. Painting it with an alpha is
    what makes the composite possible: the sheet tints the paper beneath it
    instead of hiding it, and the lightest areas of the page - what a
    born-digital document would call white - carry no ink at all.

    Args:
        surface_css: The theme's paper wash, already an ``rgba()`` string.

    Returns:
        CSS declarations for the ``.sheet`` rule, without the braces.
    """
    return f"background-color:{surface_css};"
