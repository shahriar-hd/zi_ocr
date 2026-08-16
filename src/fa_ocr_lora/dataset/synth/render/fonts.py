"""Font discovery and ``@font-face`` embedding for the rendered page.

Chromium is asked to draw text in families like ``B Nazanin`` and
``IRANSansX``. It cannot find them by name here: this environment exports
``XDG_DATA_HOME`` into a VS Code snap sandbox, so fontconfig expands its
``<dir prefix="xdg">fonts</dir>`` rule to a path that does not exist and never
indexes the real ``~/.local/share/fonts``. Every Persian family then resolves
to the same fallback face - measured, all 13 rendered to an identical width,
byte-for-byte the width of a deliberately nonexistent family.

Rather than repair the host's font state, the faces are read off disk and
inlined into each document as base64 ``@font-face`` rules. That also buys the
property a dataset pipeline needs: a sample renders identically on a machine
with none of these fonts installed, so the corpus is reproducible from
(config, seed) alone rather than from (config, seed, whatever is in the
system font cache).

Only the families a document actually uses are embedded - two or three faces,
about 200 KB of base64 - not the whole directory.
"""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Iterable

__all__ = [
    "FontFace",
    "FontLibrary",
    "load_font_library",
    "font_face_css",
]

#: Font file suffixes worth opening, mapped to their CSS ``format()`` token.
_FORMATS: Final[dict[str, str]] = {
    ".ttf": "truetype",
    ".otf": "opentype",
    ".woff2": "woff2",
    ".woff": "woff",
}

#: OpenType weight class at or above which a face counts as bold. The
#: installed faces cluster at 80 (regular) and 200 (bold) on fontconfig's
#: scale; in ``OS/2.usWeightClass`` terms that is 400 and 700.
_BOLD_WEIGHT_FLOOR: Final[int] = 600

#: ``name`` table identifiers.
_NAME_FAMILY: Final[int] = 1
_NAME_SUBFAMILY: Final[int] = 2
_NAME_TYPOGRAPHIC_FAMILY: Final[int] = 16

#: Encodings seen in the ``name`` table, by (platform_id, encoding_id).
_UTF16_PLATFORMS: Final[frozenset[tuple[int, int]]] = frozenset(
    {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (3, 1), (3, 10)}
)

#: Subfamily names that mark a face as italic or oblique.
_ITALIC_MARKERS: Final[tuple[str, ...]] = ("italic", "oblique")


@dataclass(frozen=True, slots=True)
class FontFace:
    """One font file, with the metadata needed to write its CSS rule.

    Attributes:
        family: Family name as CSS should refer to it, e.g. ``B Nazanin``.
        path: Absolute path of the font file.
        weight: CSS weight, 400 for regular and 700 for bold.
        weight_class: Raw ``OS/2.usWeightClass``, kept because binning to
            400/700 collapses Light, Regular and Medium onto one key and the
            library needs the original value to break that tie.
        italic: Whether the face is italic.
        format_hint: CSS ``format()`` token for the file type.
    """

    family: str
    path: Path
    weight: int
    weight_class: int
    italic: bool
    format_hint: str

    @property
    def style(self) -> str:
        """CSS ``font-style`` value for this face."""
        return "italic" if self.italic else "normal"


def _read_u16(payload: bytes, offset: int) -> int:
    """Read a big-endian ``uint16``."""
    return struct.unpack_from(">H", payload, offset)[0]


def _decode_name(raw: bytes, platform_id: int, encoding_id: int) -> str:
    """Decode one ``name`` table record.

    Args:
        raw: The record's bytes.
        platform_id: ``name`` record platform id.
        encoding_id: ``name`` record encoding id.

    Returns:
        The decoded string, empty when it cannot be decoded.
    """
    encoding = (
        "utf-16-be" if (platform_id, encoding_id) in _UTF16_PLATFORMS else "latin-1"
    )
    try:
        return raw.decode(encoding).strip("\x00").strip()
    except (UnicodeDecodeError, LookupError):
        return ""


def _read_tables(payload: bytes) -> dict[bytes, tuple[int, int]]:
    """Return the font's table directory as ``tag -> (offset, length)``.

    Args:
        payload: The whole font file.

    Returns:
        One entry per table.

    Raises:
        ValueError: If the file is too short or is not an sfnt font.
    """
    if len(payload) < 12:
        raise ValueError("file is too short to be a font")
    table_count = _read_u16(payload, 4)
    tables: dict[bytes, tuple[int, int]] = {}
    for index in range(table_count):
        record = 12 + index * 16
        if record + 16 > len(payload):
            break
        tag = payload[record : record + 4]
        offset, length = struct.unpack_from(">II", payload, record + 8)
        tables[tag] = (offset, length)
    if not tables:
        raise ValueError("font has no table directory")
    return tables


def _parse_names(payload: bytes, offset: int) -> dict[int, str]:
    """Extract the ``name`` table entries this module cares about.

    Args:
        payload: The whole font file.
        offset: Offset of the ``name`` table.

    Returns:
        ``name_id -> value`` for the family and subfamily records.
    """
    count = _read_u16(payload, offset + 2)
    strings_offset = offset + _read_u16(payload, offset + 4)
    wanted = (_NAME_FAMILY, _NAME_SUBFAMILY, _NAME_TYPOGRAPHIC_FAMILY)
    names: dict[int, str] = {}
    for index in range(count):
        record = offset + 6 + index * 12
        if record + 12 > len(payload):
            break
        platform_id = _read_u16(payload, record)
        encoding_id = _read_u16(payload, record + 2)
        name_id = _read_u16(payload, record + 6)
        if name_id not in wanted:
            continue
        length = _read_u16(payload, record + 8)
        value_offset = strings_offset + _read_u16(payload, record + 10)
        raw = payload[value_offset : value_offset + length]
        decoded = _decode_name(raw, platform_id, encoding_id)
        # A UTF-16 record supersedes a legacy Mac one for the same id.
        if decoded and (
            name_id not in names or (platform_id, encoding_id) in _UTF16_PLATFORMS
        ):
            names[name_id] = decoded
    return names


def _read_face(path: Path) -> FontFace | None:
    """Read one font file's family, weight, and slant.

    The ``name`` and ``OS/2`` tables are parsed directly rather than through
    fontconfig, because fontconfig is exactly the layer that is misconfigured
    here, and through Pillow, which does not expose ``usWeightClass``.

    Args:
        path: Font file to inspect.

    Returns:
        The parsed face, or ``None`` when the file is unreadable or not an
        sfnt font.
    """
    format_hint = _FORMATS.get(path.suffix.lower())
    if format_hint is None:
        return None
    try:
        payload = path.read_bytes()
        tables = _read_tables(payload)
        name_offset = tables.get(b"name")
        if name_offset is None:
            return None
        names = _parse_names(payload, name_offset[0])
    except (OSError, ValueError, struct.error):
        return None

    family = names.get(_NAME_TYPOGRAPHIC_FAMILY) or names.get(_NAME_FAMILY)
    if not family:
        return None
    subfamily = names.get(_NAME_SUBFAMILY, "").lower()

    weight_class = 400
    os2 = tables.get(b"OS/2")
    if os2 is not None:
        try:
            weight_class = _read_u16(payload, os2[0] + 4)
        except struct.error:
            weight_class = 400
    is_bold = weight_class >= _BOLD_WEIGHT_FLOOR or "bold" in subfamily

    return FontFace(
        family=family,
        path=path,
        weight=700 if is_bold else 400,
        weight_class=weight_class,
        italic=any(marker in subfamily for marker in _ITALIC_MARKERS),
        format_hint=format_hint,
    )


@dataclass(frozen=True, slots=True)
class FontLibrary:
    """Font files by family, deduplicated by (family, weight, italic).

    Attributes:
        families: All distinct family names in the library.
        faces: ``(family, weight, italic) -> face``.
    """

    families: tuple[str, ...]
    faces: dict[tuple[str, int, bool], FontFace]

    @classmethod
    def from_directory(cls, directory: Path) -> "FontLibrary":
        """Build a library from every font file in one directory.

        Args:
            directory: Where font files live; non-font files are ignored.

        Returns:
            The parsed library, with one face per (family, weight, italic)
            key. Several files can land on one key - Vazir ships Light (300),
            Regular (400) and Medium (500), which all bin to CSS 400 - so the
            winner is the file whose ``usWeightClass`` sits closest to the
            key's canonical weight. Picking by filename order instead would
            make ``Vazir`` render as ``Vazir-Light``.
        """
        faces: dict[tuple[str, int, bool], FontFace] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            face = _read_face(path)
            if face is None:
                continue
            key = (face.family, face.weight, face.italic)
            incumbent = faces.get(key)
            if incumbent is not None and abs(
                incumbent.weight_class - face.weight
            ) <= abs(face.weight_class - face.weight):
                continue
            faces[key] = face
        families = tuple(sorted({family for family, _, _ in faces}))
        return cls(families=families, faces=faces)

    def face_for(
        self,
        family: str,
        weight: int = 400,
        italic: bool = False,
    ) -> FontFace | None:
        """Return the best face for one family, falling back across weights.

        Neither direction of fallback is hypothetical: the legacy B-series
        ships regular-only, while B Titr ships bold-only, so a family can be
        present yet missing the exact weight asked for.

        Args:
            family: The CSS family name.
            weight: Requested weight, 400 or 700.
            italic: Whether an italic face is wanted.

        Returns:
            The exact face, then the same weight upright, then the family's
            other weight, or ``None`` if the family is not in the library.
        """
        other = 400 if weight == 700 else 700
        for key in (
            (family, weight, italic),
            (family, weight, False),
            (family, other, italic),
            (family, other, False),
        ):
            face = self.faces.get(key)
            if face is not None:
                return face
        return None


@lru_cache(maxsize=8)
def load_font_library(directory: Path | str | None) -> FontLibrary:
    """Load (and cache) the font library for one directory.

    Args:
        directory: Font directory, or ``None`` for an empty library - the
            caller then relies on system faces, which works on hosts whose
            fontconfig is configured correctly.

    Returns:
        The parsed library.

    Raises:
        NotADirectoryError: If the path exists and is not a directory.
    """
    if directory is None:
        return FontLibrary(families=(), faces={})
    path = Path(directory).expanduser()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"fonts_dir is not a directory: {path}")
    if not path.exists():
        return FontLibrary(families=(), faces={})
    return FontLibrary.from_directory(path)


def _inline_font(face: FontFace) -> str:
    """Read one face and encode it as a ``data:`` URI."""
    try:
        raw = face.path.read_bytes()
    except OSError:
        return ""
    encoded = base64.b64encode(raw).decode("ascii")
    return (
        "data:font/" + ("ttf" if face.format_hint == "truetype" else face.format_hint)
        + f";charset=utf-8;base64,{encoded}"
    )


def font_face_css(required: Iterable[str], library: FontLibrary) -> str:
    """Build the ``@font-face`` rules for the families a document needs.

    ``font-weight`` is normalised to 400 and 700: the legacy B-series faces
    have no bold variant (B Titr ships only bold, and Chrome applies its own
    fake bold to the rest), while IRANSansX, Sahel and Vazir carry real bold
    faces.

    Args:
        required: Family names used on the page, in no particular order.
        library: The loaded font library.

    Returns:
        A ``<style>`` block with one rule per (family, weight, italic) face,
        or an empty string when no face is available.
    """
    blocks: list[str] = []
    for family in dict.fromkeys(required):
        if family not in library.families:
            continue
        for weight in (400, 700):
            face = library.face_for(family, weight)
            if face is None:
                continue
            if (face.family, face.weight, face.italic) != (family, weight, False):
                continue
            source = _inline_font(face)
            if not source:
                continue
            blocks.append(
                "@font-face{"
                f'font-family:"{family}";font-style:normal;font-weight:{weight};'
                f"src:url({source}) format('{face.format_hint}');"
                "}"
            )
    return "\n".join(blocks)
