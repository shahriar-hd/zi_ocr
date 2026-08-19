"""One place where a page raster becomes bytes.

The render stage has three exits that write an image - the scanned branch, the
clean branch, and the HuggingFace export re-encoding what it reads - and before
this module each of them called ``Image.save(buffer, format="PNG")`` on its own.
That is three places to change when the corpus format changes, and three chances
for one of them to disagree with the filename the annotation recorded.

Everything about the encoding now comes from
:class:`~fa_ocr_lora.config.ImageFormatConfig`, whose ``suffix`` is the same
value that names the file. The format and the extension therefore cannot drift.
"""

from __future__ import annotations

import io

from PIL import Image

from ....config import ImageFormatConfig

__all__ = ["encode_image", "decode_image"]


def encode_image(image: Image.Image, spec: ImageFormatConfig) -> bytes:
    """Encode a page raster according to ``spec``.

    Args:
        image: The finished page. Must already be flattened to a mode the
            format can store - see below.
        spec: Format, quality and encoder options.

    Returns:
        The encoded file contents.

    Raises:
        ValueError: If ``image`` carries an alpha channel that the format cannot
            store. This is deliberately an error rather than a silent flatten
            onto white: a page capture is RGBA precisely because a light area is
            an *absence* of ink, so the caller has to say what goes behind it.
            Flattening here would quietly discard whatever paper was meant to
            show through.
    """
    if not spec.supports_alpha and image.mode in {"RGBA", "LA", "PA", "P"}:
        raise ValueError(
            f"{spec.format} cannot store mode {image.mode!r}; composite the page "
            "onto its paper first"
        )
    buffer = io.BytesIO()
    image.save(buffer, format=spec.pillow_format, **spec.save_kwargs())
    return buffer.getvalue()


def decode_image(data: bytes) -> Image.Image:
    """Decode encoded image bytes, whatever format they are in.

    Pillow reads the format from the header, so this works on a corpus that
    mixes formats - which one written before a format change and re-exported
    afterwards will.

    Args:
        data: Encoded image file contents.

    Returns:
        The decoded image.

    Raises:
        ValueError: If the bytes are not a readable image, so a caller sees one
            exception type rather than Pillow's ``UnidentifiedImageError``.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except OSError as error:
        raise ValueError(f"not a readable image: {error}") from error
    return image
