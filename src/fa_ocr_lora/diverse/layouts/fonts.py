"""The font pool: coverage checking, holdout, and CSS embedding.

This module exists to prevent one specific, expensive failure. Chromium, asked
to draw a character the chosen font lacks, does not complain - it silently
substitutes another installed face, or draws a tofu box. Either way the pixels
stop matching ``text_render``, and the sample becomes a training example that
teaches the model the wrong letterform. At 100k samples nobody will notice by
eye.

So coverage is checked per sample, against the actual text, before rendering:
:func:`choose_font` reads the candidate's character map and rejects it if a
single required codepoint is missing. A sample whose text no font in the pool
can draw is rejected outright rather than rendered with a fallback.

Faces are embedded in the page as base64 ``data:`` URIs rather than referenced
by family name. That makes a render reproducible on a host with none of these
fonts installed - the molab runtime, for instance - and removes any dependence
on fontconfig, which is unreliable inside sandboxes.

**Holdout.** A few families are reserved for the ``test_unseen_fonts`` split
and never appear in train. The choice is deterministic (a hash of the family
name, not a shuffle) so it is stable across runs and across machines, and it
excludes the handful of families marked ``holdout_eligible=false`` in
``fonts.csv`` - holding out the most typical family in the pool would measure
font *frequency* rather than font *novelty*.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from ..config import STYLE_CLASSES

__all__ = [
    "FontFace",
    "FontPool",
    "FontSelectionError",
    "choose_font",
    "load_font_pool",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: Weights over style classes, used when a category expresses no preference.
#: Sans dominates real Persian screen and office text; nastaliq is rare and
#: slow to shape, and illegible below a certain size, so it is deliberately
#: thin here and boosted only by the categories that suit it (poetry).
_DEFAULT_STYLE_WEIGHTS: Final[Mapping[str, float]] = {
    "sans": 0.46,
    "naskh": 0.18,
    "serif": 0.16,
    "display": 0.14,
    "nastaliq": 0.05,
    "mono": 0.01,
}

#: Characters whose absence never disqualifies a font. Whitespace carries no
#: glyph, and ZWNJ is a formatting control that fonts are not required to map:
#: Chromium honours it through the shaper regardless of the cmap.
_IGNORABLE: Final[frozenset[str]] = frozenset(
    {" ", "\n", "\r", "\t", "\u200c", "\u200b", "\u00a0", "\ufeff"}
)


class FontSelectionError(RuntimeError):
    """Raised when no font in the pool can draw a sample's text."""


@dataclass(frozen=True, slots=True)
class FontFace:
    """One font file and everything needed to use and credit it.

    Attributes:
        font_id: Stable identifier, ``family-slug/filename``.
        family: CSS family name.
        path: Absolute path to the font file.
        style_class: One of :data:`fa_ocr_lora.diverse.config.STYLE_CLASSES`.
        weight: CSS weight, from the OS/2 table.
        italic: Whether the face is italic.
        licence: SPDX-ish identifier, recorded in the manifest and the card.
        attribution: Human-readable credit.
        holdout_eligible: Whether this family may be reserved for the
            unseen-font test split.
    """

    font_id: str
    family: str
    path: Path
    style_class: str
    weight: int
    italic: bool
    licence: str
    attribution: str
    holdout_eligible: bool

    @property
    def format_hint(self) -> str:
        """Return the CSS ``format()`` hint for this file's extension."""
        return {
            ".ttf": "truetype",
            ".otf": "opentype",
            ".woff": "woff",
            ".woff2": "woff2",
        }.get(self.path.suffix.lower(), "truetype")


@lru_cache(maxsize=256)
def _coverage(path: str) -> frozenset[int]:
    """Return the set of codepoints a font file can draw.

    Cached because a 100k-sample run checks the same few dozen files
    repeatedly, and parsing a font's tables is far from free.

    Args:
        path: Absolute path to the font file, as a string so the result is
            hashable for :func:`functools.lru_cache`.

    Returns:
        Every codepoint present in any of the font's character maps. Empty
        when the file cannot be parsed, which makes an unreadable font fail
        coverage rather than pass it silently.
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        LOGGER.error("fontTools is required for glyph coverage: pip install fonttools")
        return frozenset()
    try:
        font = TTFont(path, fontNumber=0, lazy=True)
        covered: set[int] = set()
        for table in font["cmap"].tables:
            covered.update(table.cmap.keys())
        font.close()
        return frozenset(covered)
    except Exception as error:  # noqa: BLE001 - any parse failure is zero coverage
        LOGGER.warning("cannot read coverage of %s: %s", path, error)
        return frozenset()


def missing_glyphs(text: str, face: FontFace) -> str:
    """Return the characters of ``text`` that ``face`` cannot draw.

    Args:
        text: The exact string that will be rendered, i.e. ``text_render``.
        face: Candidate font face.

    Returns:
        The distinct missing characters, in first-appearance order; empty when
        coverage is complete.
    """
    covered = _coverage(str(face.path))
    if not covered:
        return "?"
    missing = {
        character
        for character in text
        if character not in _IGNORABLE and ord(character) not in covered
    }
    if not missing:
        return ""
    return "".join(dict.fromkeys(character for character in text if character in missing))


@lru_cache(maxsize=64)
def _embed(path: str, format_hint: str) -> str:
    """Return a font file as a base64 ``data:`` URI.

    Cached: at 100k samples the same file is embedded tens of thousands of
    times, and base64 of a 300 KB font is not cheap.

    Args:
        path: Absolute path to the font file.
        format_hint: CSS ``format()`` hint.

    Returns:
        The ``url(data:...) format(...)`` source string.
    """
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    mime = {
        "truetype": "font/ttf",
        "opentype": "font/otf",
        "woff": "font/woff",
        "woff2": "font/woff2",
    }.get(format_hint, "font/ttf")
    return f"url(data:{mime};base64,{payload}) format('{format_hint}')"


class FontPool:
    """Every usable face, indexed by style class and family.

    Attributes:
        faces: All faces in the pool.
        holdout_families: Families reserved for the unseen-font test split.
    """

    __slots__ = ("faces", "holdout_families", "_by_class", "_by_family")

    def __init__(self, faces: Sequence[FontFace], holdout: int = 0) -> None:
        """Index the pool and choose the holdout families.

        Args:
            faces: Accepted faces, typically from ``fonts.csv``.
            holdout: Number of families to reserve for ``test_unseen_fonts``.
        """
        self.faces = tuple(faces)
        self._by_family: dict[str, list[FontFace]] = {}
        for face in self.faces:
            self._by_family.setdefault(face.family, []).append(face)

        self.holdout_families = self._pick_holdout(holdout)

        self._by_class: dict[str, list[FontFace]] = {}
        for face in self.faces:
            self._by_class.setdefault(face.style_class, []).append(face)

    def _pick_holdout(self, count: int) -> frozenset[str]:
        """Choose the families reserved for the unseen-font test.

        Chosen by hashing the family name rather than by shuffling, so the
        same families are held out on every machine and in every run - a
        holdout that moves between runs makes the test incomparable.

        Args:
            count: How many families to reserve.

        Returns:
            The reserved family names.
        """
        if count <= 0:
            return frozenset()
        eligible = sorted(
            family
            for family, faces in self._by_family.items()
            if any(face.holdout_eligible for face in faces)
        )
        if count >= len(eligible):
            raise FontSelectionError(
                f"render.fonts.holdout is {count} but only "
                f"{len(eligible)} family/families "
                "are eligible; holding out all of them would leave train with no fonts"
            )
        ranked = sorted(
            eligible,
            key=lambda family: hashlib.blake2b(
                family.encode("utf-8"), digest_size=8
            ).hexdigest(),
        )
        return frozenset(ranked[:count])

    def families(self, *, include_holdout: bool = True) -> tuple[str, ...]:
        """Return the pool's family names.

        Args:
            include_holdout: Whether to include the reserved families.

        Returns:
            Sorted family names.
        """
        names = sorted(self._by_family)
        if include_holdout:
            return tuple(names)
        return tuple(name for name in names if name not in self.holdout_families)

    def candidates(
        self,
        *,
        style_class: str | None = None,
        holdout_only: bool = False,
        exclude_holdout: bool = True,
    ) -> list[FontFace]:
        """Return the faces matching a selection.

        Args:
            style_class: Restrict to this style class, or ``None`` for any.
            holdout_only: Return *only* the reserved families, which is what
                the unseen-font test split renders with.
            exclude_holdout: Drop the reserved families. Ignored when
                ``holdout_only`` is set.

        Returns:
            The matching faces.
        """
        pool = self._by_class.get(style_class, []) if style_class else list(self.faces)
        if holdout_only:
            return [face for face in pool if face.family in self.holdout_families]
        if exclude_holdout:
            return [face for face in pool if face.family not in self.holdout_families]
        return list(pool)

    def face_css(self, faces: Iterable[FontFace]) -> str:
        """Build ``@font-face`` rules embedding each face as a data URI.

        Args:
            faces: Faces used by the page.

        Returns:
            The CSS text.
        """
        rules: list[str] = []
        for face in faces:
            rules.append(
                "@font-face{"
                f"font-family:'{face.family}';"
                f"font-style:{'italic' if face.italic else 'normal'};"
                f"font-weight:{face.weight};"
                f"src:{_embed(str(face.path), face.format_hint)};"
                "font-display:block}"
            )
        return "".join(rules)

    def __len__(self) -> int:
        """Return the number of faces in the pool."""
        return len(self.faces)


def load_font_pool(fonts_csv: Path, *, holdout: int = 0) -> FontPool:
    """Load the pool from the catalogue written by the download script.

    Args:
        fonts_csv: Path to ``fonts.csv``.
        holdout: Families to reserve for ``test_unseen_fonts``.

    Returns:
        The pool.

    Raises:
        FontSelectionError: If the catalogue is missing, empty, or holds no
            face that passed the download script's coverage gate.
    """
    if not fonts_csv.is_file():
        raise FontSelectionError(
            f"{fonts_csv} not found; run `python scripts/download_diverse_fonts.py` first"
        )
    faces: list[FontFace] = []
    root = fonts_csv.parent
    with fonts_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("covers_persian", "").strip().lower() != "true":
                continue
            style = row.get("style_class", "sans").strip()
            if style not in STYLE_CLASSES:
                LOGGER.warning(
                    "skipping %s: unknown style class %r", row.get("font_id"), style
                )
                continue
            path = root / row["file"]
            if not path.is_file():
                LOGGER.warning("skipping %s: file missing", row.get("font_id"))
                continue
            faces.append(
                FontFace(
                    font_id=row.get("font_id", path.name),
                    family=row["family"].strip(),
                    path=path,
                    style_class=style,
                    weight=int(row.get("weight") or 400),
                    italic=row.get("italic", "false").strip().lower() == "true",
                    licence=row.get("licence", "").strip(),
                    attribution=row.get("attribution", "").strip(),
                    holdout_eligible=row.get("holdout_eligible", "true").strip().lower()
                    == "true",
                )
            )
    if not faces:
        raise FontSelectionError(f"{fonts_csv} holds no usable face")
    pool = FontPool(faces, holdout=holdout)
    LOGGER.info(
        "font pool: %d face(s), %d family/families, %d held out "
        "for test_unseen_fonts (%s)",
        len(pool),
        len(pool.families()),
        len(pool.holdout_families),
        ", ".join(sorted(pool.holdout_families)) or "none",
    )
    return pool


def choose_font(
    pool: FontPool,
    text: str,
    rng: random.Random,
    *,
    style_weights: Mapping[str, float] | None = None,
    holdout_only: bool = False,
    want_bold: bool = False,
) -> FontFace:
    """Pick a face that can draw every character of ``text``.

    Candidates are tried in a seeded random order and the first with complete
    coverage wins. This is the check that keeps labels honest: without it a
    display font missing ``ژ`` would render a tofu box where the label claims a
    letter, and the sample would teach the model that mistake.

    Args:
        pool: The font pool.
        text: Exactly the string that will be drawn.
        rng: The sample's seeded RNG.
        style_weights: Relative weights over style classes; defaults to
            :data:`_DEFAULT_STYLE_WEIGHTS`.
        holdout_only: Draw from the reserved families, for the unseen-font
            test split.
        want_bold: Prefer a face at weight 600 or above, for headings and
            banners.

    Returns:
        A face with complete coverage of ``text``.

    Raises:
        FontSelectionError: If no face in the pool covers the text. The caller
            should reject the sample; rendering it with a fallback would
            produce a mislabelled image.
    """
    from ...dataset.common.io_utils import weighted_choice

    weights = dict(style_weights or _DEFAULT_STYLE_WEIGHTS)
    classes = [
        (name, weight)
        for name, weight in weights.items()
        if weight > 0 and pool.candidates(style_class=name, holdout_only=holdout_only)
    ]
    if not classes:
        raise FontSelectionError(
            "no style class has any candidate face"
            + (" among the holdout families" if holdout_only else "")
        )

    tried: set[str] = set()
    worst = ""
    # Try the preferred class first, then every class, so a sample whose text
    # one class cannot draw still renders rather than being dropped.
    order = [weighted_choice(classes, rng)] + [name for name, _ in classes]
    for style in order:
        candidates = pool.candidates(style_class=style, holdout_only=holdout_only)
        if not candidates:
            continue
        if want_bold:
            bold = [face for face in candidates if face.weight >= 600]
            candidates = bold or candidates
        rng.shuffle(candidates)
        for face in candidates:
            if face.font_id in tried:
                continue
            tried.add(face.font_id)
            missing = missing_glyphs(text, face)
            if not missing:
                return face
            if not worst:
                worst = missing

    raise FontSelectionError(
        f"no font covers this sample; first missing run was {worst[:24]!r} "
        f"({len(tried)} face(s) tried)"
    )
