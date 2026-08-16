"""Paper texture behind the sheet, drawn from prepared blank-page scans.

Half the corpus renders on plain white and half on one of the scans in
``data/texture``, so the adapter sees both clean digital pages and pages that
carry the grain, warmth and vignetting of a real scanner. The choice is one
per document, not one per sheet: a letter is printed on a single kind of
paper.

Three properties of the prepared assets shape this module:

* **Extensions lie.** ``page_4.jpg`` is really a WebP. The MIME type is taken
  from the decoded header rather than the suffix, so the ``data:`` URI never
  claims a format the bytes are not.
* **Sizes span 190x265 to 3000x4000.** Every scan is stretched to ``cover``
  regardless. Tiling the small ones was tried first, on the theory that a
  190px scan upscaled 13x would smear into a gradient; rendered side by side,
  the tiles instead laid a periodic rectangular seam across the page, which is
  exactly the kind of spurious regularity an OCR model will learn. The
  upscaled scan is soft but seamless, and soft grain is what a low-resolution
  scan of blank paper looks like anyway.
* **Pixels never pass through Pillow.** Only the header is read, for
  dimensions and true format; the file's own bytes go to Chromium untouched.
  That keeps AVIF and WebP working - Chromium decodes both natively - and
  avoids a re-encode that would soften the grain the texture exists to
  provide.

The texture is background only. It gets no bounding box, but its filename is
recorded in the annotation so a sample's appearance stays traceable to the
asset that produced it.
"""

from __future__ import annotations

import base64
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
    "background_css",
    "backdrop_css",
]

#: Pillow format name mapped to the MIME type Chromium should be told about.
#: Keyed by what the header actually says, not by the file's extension.
_MIME_BY_FORMAT: Final[dict[str, str]] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "AVIF": "image/avif",
    "GIF": "image/gif",
}

#: Translucent white laid over the texture so body ink keeps its contrast
#: against the paper grain. Zero would print the scan at full strength.
_WASH_RANGE: Final[tuple[float, float]] = (0.10, 0.34)


def _probe(path: Path) -> tuple[int, int, str] | None:
    """Read one texture's dimensions and true MIME type from its header.

    Args:
        path: Image file to inspect.

    Returns:
        ``(width, height, mime)``, or ``None`` when the file is unreadable,
        is not an image, or is in a format Chromium cannot display.
    """
    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or ""
    except (OSError, UnidentifiedImageError, ValueError):
        return None
    mime = _MIME_BY_FORMAT.get(image_format.upper())
    if mime is None or width <= 0 or height <= 0:
        return None
    return width, height, mime


@lru_cache(maxsize=8)
def available_textures(directory: Path | str | None) -> tuple[Path, ...]:
    """List the usable texture files in one directory, in sorted order.

    Sorting matters: the pick is made by index from a seeded RNG, so the
    directory listing has to be stable for a sample to reproduce.

    Args:
        directory: Where the prepared scans live, or ``None`` to disable
            textures entirely.

    Returns:
        Every file whose header identifies it as a Chromium-displayable
        image, sorted by name. Empty when the directory is absent.
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
    """One document's chosen paper, resolved to a CSS background layer.

    Attributes:
        path: The source scan on disk.
        width: Source width in pixels.
        height: Source height in pixels.
        mime: True MIME type, read from the header rather than the suffix.
        wash: Opacity of the white veil laid over the scan, in ``[0, 1]``.
        rotate_180: Whether the scan is flipped, so one asset does not always
            put its darkest corner in the same place.
    """

    path: Path
    width: int
    height: int
    mime: str
    wash: float
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
            "mime": self.mime,
            "wash": self.wash,
            "rotate_180": self.rotate_180,
        }


@lru_cache(maxsize=16)
def _data_uri(path: Path, mime: str) -> str:
    """Base64 one texture file, cached because a batch reuses the same few.

    The bytes go to Chromium exactly as they sit on disk - no Pillow re-encode
    - so AVIF and WebP keep working and the grain is not softened.

    Args:
        path: Texture file to inline.
        mime: MIME type to declare.

    Returns:
        A ``data:`` URI, or an empty string when the file cannot be read.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def choose_texture(
    directory: Path | str | None,
    probability: float,
    rng: Random,
) -> Texture | None:
    """Pick the paper for one document, or plain white.

    Args:
        directory: Where the prepared scans live.
        probability: Chance of using a texture at all, in ``[0, 1]``.
        rng: The sample's seeded RNG, conventionally ``purpose="texture"``.

    Returns:
        The chosen texture, or ``None`` for a plain-white page.
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
    width, height, mime = probed

    return Texture(
        path=path,
        width=width,
        height=height,
        mime=mime,
        wash=round(rng.uniform(*_WASH_RANGE), 3),
        rotate_180=rng.random() < 0.5,
    )


def background_css(texture: Texture | None, surface: str) -> str:
    """Build the sheet's own ``background`` declarations.

    Only the flat surface colour lands here. The scan itself goes on the
    sheet's backdrop layer via :func:`backdrop_css`, because a background
    image cannot be flipped in place - CSS transforms rotate the element and
    everything in it, which would stand the letter on its head.

    Args:
        texture: The chosen paper, or ``None`` for plain white.
        surface: The theme's surface colour, painted under everything.

    Returns:
        CSS declarations for the ``.sheet`` rule, without the braces.
    """
    del texture  # The sheet paints the same base colour either way.
    return f"background-color:{surface};"


def backdrop_css(texture: Texture | None) -> str:
    """Build the declarations for the sheet's backdrop layer.

    The template owns the layer's geometry - an absolutely positioned
    ``.sheet::before`` pinned to all four insets and stacked below the
    content - and this function supplies only what varies per sample. The
    white veil is a gradient stacked above the scan rather than an ``opacity``
    on the sheet, which would fade the text along with the paper.

    Args:
        texture: The chosen paper, or ``None`` for plain white.

    Returns:
        CSS declarations for the ``.sheet::before`` rule, or an empty string
        when the page renders on plain white.
    """
    if texture is None:
        return ""
    uri = _data_uri(texture.path, texture.mime)
    if not uri:
        return ""

    veil = f"rgba(255,255,255,{texture.wash})"
    declarations = [
        f"background-image:linear-gradient({veil},{veil}),url({uri});",
        "background-size:auto,cover;",
        "background-repeat:repeat,no-repeat;",
        "background-position:center,center;",
    ]
    if texture.rotate_180:
        # Safe here and not on .sheet: this layer holds no text, so flipping
        # it only moves the scan's vignette to the opposite corner.
        declarations.append("transform:rotate(180deg);")
    return "".join(declarations)

