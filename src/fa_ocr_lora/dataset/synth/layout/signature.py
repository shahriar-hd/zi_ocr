"""Handwritten signature, lifted off its scan and inked onto the letter.

The UTSig corpus stores one signature per TIFF as an 8-bit grayscale scan of
white paper: ~4-8% of pixels are ink, the rest is near-white. Pasting that
image onto the page would print a white rectangle over the letterhead's tint
and over the paper texture, and the seam would be an obvious, learnable
artifact. So the paper is thrown away and only the ink is kept, as the alpha
channel of a transparent PNG.

Three decisions follow from what the scans actually look like:

* **Alpha comes from ink darkness, not a hard threshold.** A binary mask would
  render the strokes with aliased, rubber-stamp edges. Inverting the gray
  channel keeps each stroke's own antialiasing, so the pen reads as pen.
* **A paper floor is still needed.** Scanner haze puts most "white" pixels at
  240-254 rather than 255; carried through as faint alpha it prints a visible
  grey box. Everything at or below ``signature.paper_floor`` is cut to fully
  transparent and the remainder is rescaled, which removes the box without
  eating stroke ends.
* **The ink is recoloured.** A scan's ink is near-black; real signatures are
  blue or blue-black about as often. The extracted alpha is filled with one
  sampled pen colour, which also stops the signature from matching the body
  text exactly.

Geometry is jittered per sample - width, rotation, and baseline offset - because
a corpus where every signature sits at the same angle and size teaches the
model the placement rather than the shape.
"""

from __future__ import annotations

import base64
import html
import io
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from PIL import Image, ImageOps, UnidentifiedImageError

from ....config import SignatureConfig
from ...common.io_utils import weighted_choice
from .theme import hex_to_rgb

__all__ = [
    "SignatureSpec",
    "build_signature",
    "signature_to_html",
    "signature_block_to_html",
    "signature_block_to_markdown",
]

#: Fallback for a caller that does not hand over the loaded config - the test
#: suite, mostly. Read off an instance so a literal here cannot drift from the
#: dataclass default.
_DEFAULT_SIGNATURE: Final[SignatureConfig] = SignatureConfig()


@dataclass(frozen=True, slots=True)
class SignatureSpec:
    """One letter's signature, resolved to an inline image.

    Attributes:
        path: Source TIFF in the UTSig corpus.
        data_uri: Transparent PNG as a ``data:`` URI.
        width_mm: Rendered width on the page.
        height_mm: Rendered height, preserving the ink's aspect ratio.
        ink: Pen colour the strokes were filled with.
        rotate_deg: Tilt already baked into the PNG, recorded for provenance.
        source_size: The scan's original ``(width, height)`` in pixels.
        overlap_mm: How far the signature rides up over the signatory title,
            as real signatures do rather than sitting neatly below it.
    """

    path: Path
    data_uri: str
    width_mm: float
    height_mm: float
    ink: str
    rotate_deg: float
    source_size: tuple[int, int]
    overlap_mm: float = 0.0

    @property
    def name(self) -> str:
        """Filename recorded in the annotation for provenance."""
        return self.path.name

    def as_content(self) -> dict[str, object]:
        """Return the signature's provenance block for the annotation."""
        return {
            "name": self.name,
            "source_size": list(self.source_size),
            "size_mm": [self.width_mm, self.height_mm],
            "ink": self.ink,
            "rotate_deg": self.rotate_deg,
        }


@lru_cache(maxsize=64)
def _extract_alpha(
    path: Path, paper_floor: int, crop_pad: float, max_edge_px: int
) -> tuple[Image.Image, tuple[int, int]] | None:
    """Lift the ink off one scan as a standalone alpha mask.

    Cached because a 20k-sample run reuses each of the 5175 scans several
    times, and the decode plus level correction is the expensive half. The
    three extraction parameters are arguments rather than config reads so they
    are part of the cache key: two configs cutting at different floors in one
    process must not share a mask.

    Args:
        path: A UTSig ``.tif``.
        paper_floor: ``signature.paper_floor`` - alpha at or below which a
            pixel is paper.
        crop_pad: ``signature.crop_pad`` - padding kept around the ink.
        max_edge_px: ``signature.max_edge_px`` - cap on the longest edge.

    Returns:
        ``(alpha, source_size)`` where ``alpha`` is an ``L``-mode image
        cropped to the ink, or ``None`` when the file is unreadable or holds
        no ink at all.
    """
    try:
        with Image.open(path) as handle:
            source_size = handle.size
            gray = handle.convert("L")
    except (OSError, UnidentifiedImageError, ValueError):
        return None

    # Dark ink becomes opaque, white paper becomes transparent. Inverting the
    # channel rather than thresholding keeps each stroke's antialiased edge.
    alpha = ImageOps.invert(gray)

    # Scanner haze leaves "white" at 240-254, i.e. alpha 1-15. Left in, it
    # prints as a faint grey rectangle over the letterhead. Cutting at the
    # floor and rescaling the remainder removes the box while keeping faint
    # stroke ends at their relative weight.
    span = 255 - paper_floor
    alpha = alpha.point(
        [
            0
            if level <= paper_floor
            else min(255, round((level - paper_floor) * 255 / span))
            for level in range(256)
        ]
    )

    box = alpha.getbbox()
    if box is None:
        return None
    pad = round(max(box[2] - box[0], box[3] - box[1]) * crop_pad)
    alpha = alpha.crop(
        (
            max(0, box[0] - pad),
            max(0, box[1] - pad),
            min(alpha.width, box[2] + pad),
            min(alpha.height, box[3] + pad),
        )
    )
    if alpha.width < 2 or alpha.height < 2:
        return None
    if max(alpha.size) > max_edge_px:
        alpha.thumbnail((max_edge_px, max_edge_px), Image.LANCZOS)
    return alpha, source_size


def _ink_png(alpha: Image.Image, ink: str, rotate_deg: float) -> tuple[str, int, int]:
    """Fill an alpha mask with pen colour and encode it as a PNG data URI.

    Args:
        alpha: The extracted ink mask.
        ink: Pen colour as ``#rrggbb``.
        rotate_deg: Counter-clockwise tilt in degrees.

    Returns:
        ``(data_uri, width_px, height_px)`` for the rotated result.
    """
    layer = Image.new("RGBA", alpha.size, hex_to_rgb(ink) + (0,))
    layer.putalpha(alpha)
    if abs(rotate_deg) > 0.05:
        # expand=True grows the canvas to fit the corners, so no stroke is
        # clipped and the caller can keep using the image's own aspect ratio.
        layer = layer.rotate(rotate_deg, resample=Image.BICUBIC, expand=True)
    buffer = io.BytesIO()
    layer.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", layer.width, layer.height


def build_signature(
    path: Path | None,
    rng: random.Random,
    probability: float = 1.0,
    config: SignatureConfig = _DEFAULT_SIGNATURE,
) -> SignatureSpec | None:
    """Prepare the signature for one letter, or none.

    Args:
        path: The row's signature scan, or ``None`` when the manifest left
            ``signature_path`` empty - about 30% of rows, which render
            unsigned and must not claim a signature bounding box.
        rng: The sample's seeded RNG, conventionally ``purpose="signature"``.
        probability: Chance of drawing a signature when the row has one.
        config: The ``signature`` config section: pen colours, size and tilt
            ranges, and the ink-extraction thresholds.

    Returns:
        The prepared signature, or ``None`` when the letter is unsigned or the
        scan turned out to be unreadable.
    """
    # Drawn unconditionally, as in texture.py, so an unsigned row advances the
    # RNG exactly as a signed one does and the sample's later choices do not
    # shift with the signature rate.
    draw = rng.random()
    ink = weighted_choice(config.pen_inks, rng)
    width_mm = round(config.width_mm.draw(rng), 1)
    rotate_deg = round(config.rotate_deg.draw(rng), 2)
    overlap_mm = round(config.overlap_mm.draw(rng), 1)
    if path is None or draw >= probability:
        return None

    extracted = _extract_alpha(
        path, config.paper_floor, config.crop_pad, config.max_edge_px
    )
    if extracted is None:
        return None
    alpha, source_size = extracted
    data_uri, width_px, height_px = _ink_png(alpha, ink, rotate_deg)
    if not data_uri or width_px <= 0:
        return None

    return SignatureSpec(
        path=path,
        data_uri=data_uri,
        width_mm=width_mm,
        height_mm=round(width_mm * height_px / width_px, 2),
        ink=ink,
        rotate_deg=rotate_deg,
        source_size=source_size,
        overlap_mm=overlap_mm,
    )


def signature_to_html(spec: SignatureSpec, element_id: str = "signature") -> str:
    """Render the signature image as an annotated ``<img>``.

    Args:
        spec: The prepared signature.
        element_id: Id prefix for the enclosing signature block.

    Returns:
        One ``<img>`` element carrying its own annotation attributes.
    """
    return (
        '<img class="signature-ink"'
        ' data-ann-type="signature_image"'
        f' data-ann-id="{html.escape(element_id, quote=True)}.image"'
        f' src="{spec.data_uri}" alt=""'
        f' style="width:{spec.width_mm}mm;height:{spec.height_mm}mm">'
    )


def signature_block_to_html(
    signatory_title: str,
    spec: SignatureSpec | None,
    *,
    side: str = "left",
    stamp_html: str = "",
    element_id: str = "signature",
) -> str:
    """Render the closing block: signatory title, signature, and stamp.

    The title comes first in the DOM and the ink is pulled up over it by
    ``spec.overlap_mm``, which is how a signed letter actually looks - the pen
    crosses the printed line rather than sitting politely beneath it. Reading
    order stays title-then-ink, which is what the transcription wants, since
    the ink carries no text.

    Args:
        signatory_title: The printed post, e.g. ``مدیرعامل``.
        spec: The prepared signature, or ``None`` for an unsigned letter.
        side: ``right``, ``center`` or ``left`` - which margin the block hugs.
        stamp_html: Pre-rendered stamp fragment from :mod:`.stamp`, or ``""``.
        element_id: Id prefix for every node in the block.

    Returns:
        The block's HTML, or an empty string when there is nothing to show.
    """
    title = signatory_title.strip()
    if not title and spec is None and not stamp_html:
        return ""

    safe_id = html.escape(element_id, quote=True)
    parts: list[str] = []
    if title:
        parts.append(
            '<div class="signatory-title"'
            ' data-ann-type="signatory_title"'
            f' data-ann-id="{safe_id}.title">{html.escape(title)}</div>'
        )
    if spec is not None or stamp_html:
        offset = f"margin-top:-{spec.overlap_mm}mm;" if spec is not None and title else ""
        parts.append(
            f'<div class="signature-marks" style="{offset}">'
            f"{signature_to_html(spec, element_id) if spec is not None else ''}"
            f"{stamp_html}"
            "</div>"
        )
    return (
        f'<div class="signature-block signature-{html.escape(side, quote=True)}"'
        ' data-ann-type="signature_block"'
        f' data-ann-id="{safe_id}">{"".join(parts)}</div>'
    )


def signature_block_to_markdown(
    signatory_title: str,
    spec: SignatureSpec | None,
    *,
    has_stamp: bool = False,
) -> str:
    """Render the closing block as Markdown.

    Args:
        signatory_title: The printed post.
        spec: The prepared signature, or ``None``.
        has_stamp: Whether a stamp was composited beside the signature.

    Returns:
        The Markdown fragment, or an empty string when the block is absent.
    """
    lines: list[str] = []
    if signatory_title.strip():
        lines.append(f"**{signatory_title.strip()}**")
    marks = [
        name
        for name, present in (("امضا", spec is not None), ("مهر", has_stamp))
        if present
    ]
    if marks:
        lines.append("_(" + " و ".join(marks) + ")_")
    return "\n\n".join(lines)
