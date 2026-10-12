"""The single Persian text policy: normalisation, digit scripts, ZWNJ, diacritics.

Every stage that touches a string routes through this module. That is not
tidiness, it is the only defence against the failure mode that makes an OCR
dataset worthless: **the label not matching the pixels**.

Two fields, never one
--------------------

Each sample carries two strings and they mean different things:

``text_render``
    Exactly the character sequence handed to the renderer. It is what the
    pixels show. Nothing downstream may "clean" it, because the moment it is
    changed the label stops describing the image.

``text_norm``
    The evaluation form. Arabic letter variants folded to Persian, digits
    folded to one script, diacritics stripped, whitespace collapsed. This is
    what CER is computed against, so a model that writes ``ي`` where the page
    shows ``ي`` is not punished for a distinction no reader makes.

The previous administrative-letter run had no such separation and paid for it
twice: digit script drifted inside a single page (a Persian ``۱۴۰۳`` beside a
Latin ``1403``, neither labelled), and the Arabic/Persian ``ye`` landed in
labels inconsistently depending on which model wrote the text. Both are now
*controlled variables*: :func:`apply_digit_script` sets the digit script per
sample from the plan, and :func:`fold_arabic_letters` is applied to
``text_render`` by default so Arabic forms appear only when a plan asked for
them and said so in metadata.

Why fold in ``text_render`` at all
----------------------------------

It would be defensible to render whatever the model wrote. It is not
defensible to do so *silently*: an LLM emits ``ي`` unpredictably, so without
folding, roughly a tenth of the corpus would train the model that Persian text
contains Arabic yeh, with no field recording which tenth. Folding by default
and opting a labelled minority back in (``arabic_forms`` in the layout plan)
keeps the variation while keeping it countable.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Final

__all__ = [
    "ARABIC_INDIC_DIGITS",
    "DIACRITICS",
    "DIGIT_SCRIPTS",
    "LATIN_DIGITS",
    "PERSIAN_DIGITS",
    "TATWEEL",
    "ZWNJ",
    "collapse_whitespace",
    "count_digits",
    "detect_digit_scripts",
    "diacritic_ratio",
    "digit_density",
    "fold_arabic_letters",
    "has_diacritics",
    "has_zwnj",
    "letter_counts",
    "normalise_for_eval",
    "persian_letter_ratio",
    "prepare_render_text",
    "strip_diacritics",
    "strip_tatweel",
    "to_digit_script",
    "word_count",
]

#: Zero-width non-joiner. Persian writes compounds such as ``می‌رود`` and
#: plurals such as ``کتاب‌ها`` with it. It is invisible, it is one codepoint
#: wide in the label, and it changes the rendered glyph shapes - so it is
#: preserved in ``text_render`` and recorded in ``has_zwnj``, never guessed at.
ZWNJ: Final[str] = "\u200c"

#: Zero-width joiner, Arabic letter mark, and the bidi control characters.
#: These are invisible, they do not change the visible glyph sequence in a
#: ``dir="rtl"`` block, and a model cannot possibly predict them. Any that an
#: LLM emits are removed from ``text_render`` before rendering, because a label
#: containing an unpredictable invisible character is an unlearnable label.
_INVISIBLE_DROP: Final[frozenset[str]] = frozenset(
    {
        "\u200b",  # zero-width space
        "\u200d",  # zero-width joiner
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u061c",  # Arabic letter mark
        "\u202a",  # LRE
        "\u202b",  # RLE
        "\u202c",  # PDF
        "\u202d",  # LRO
        "\u202e",  # RLO
        "\u2066",  # LRI
        "\u2067",  # RLI
        "\u2068",  # FSI
        "\u2069",  # PDI
        "\ufeff",  # BOM / zero-width no-break space
    }
)

#: Arabic tatweel (kashida). A decorative elongation with no phonetic value.
#: Renderers stretch it by an amount that depends on justification, so a label
#: that contains it describes a width the model cannot infer. Stripped.
TATWEEL: Final[str] = "\u0640"

#: Persian (extended Arabic-Indic) digits, ``۰``-``۹``.
PERSIAN_DIGITS: Final[str] = (
    "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
)

#: Arabic-Indic digits, ``٠``-``٩``. Visually distinct from the Persian set for
#: four, five and six, which is exactly why they are a separate labelled class
#: rather than folded together.
ARABIC_INDIC_DIGITS: Final[str] = (
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
)

#: Latin (ASCII) digits, ``0``-``9``.
LATIN_DIGITS: Final[str] = "0123456789"

#: The digit scripts a sample may be rendered in. ``mixed`` means the sample
#: deliberately carries more than one and is labelled as such; it is never the
#: result of an accident.
DIGIT_SCRIPTS: Final[tuple[str, ...]] = ("persian", "latin", "arabic_indic", "mixed")

#: Arabic harakat and related marks (tashkeel). Kept for a labelled subset of
#: the corpus (``has_diacritics``) and stripped for the rest, because a model
#: trained only on undiacriticised text fails on Qur'anic and pedagogical
#: material, while one trained only on diacriticised text hallucinates marks.
DIACRITICS: Final[frozenset[str]] = frozenset(
    {
        "\u064b",  # fathatan
        "\u064c",  # dammatan
        "\u064d",  # kasratan
        "\u064e",  # fatha
        "\u064f",  # damma
        "\u0650",  # kasra
        "\u0651",  # shadda
        "\u0652",  # sukun
        "\u0653",  # maddah above
        "\u0654",  # hamza above
        "\u0655",  # hamza below
        "\u0656",  # subscript alef
        "\u0657",  # inverted damma
        "\u0658",  # mark noon ghunna
        "\u0659",  # zwarakay
        "\u065a",  # vowel sign small v above
        "\u065b",  # vowel sign inverted small v above
        "\u065c",  # vowel sign dot below
        "\u065d",  # reversed damma
        "\u065e",  # fatha with two dots
        "\u065f",  # wavy hamza below
        "\u0670",  # superscript alef
        "\u06d6",  # small high ligature sad
        "\u06d7",
        "\u06d8",
        "\u06d9",
        "\u06da",
        "\u06db",
        "\u06dc",
        "\u06df",
        "\u06e0",
        "\u06e1",
        "\u06e2",
        "\u06e3",
        "\u06e4",
        "\u06e7",
        "\u06e8",
        "\u06ea",
        "\u06eb",
        "\u06ec",
        "\u06ed",
    }
)

#: Arabic letter and punctuation forms folded to their Persian equivalents.
#:
#: The pairs are not arbitrary; each one is a case where two codepoints render
#: as the same glyph in a Persian font, so leaving both in circulation teaches
#: the model a distinction the pixels do not carry:
#:
#: * ``ي``/``ى``/``ئ``-less forms -> ``ی``: Arabic yeh and alef maksura both
#:   draw as Persian yeh. ``ئ`` keeps its hamza and is *not* folded.
#: * ``ك`` -> ``ک``: Arabic kaf draws with the Persian kaf's shape in every
#:   Persian font in the pool.
#: * ``ة`` -> ``ه``: Persian orthography does not use teh marbuta.
#: * ``ۀ`` -> ``ه``: heh with yeh above is a rendering of the ezafe, inconsistent
#:   across sources.
#: * Presentation-form hamza and alef variants collapse to their base letters.
#: * Arabic comma/semicolon/question mark are *kept* (``،`` ``؛`` ``؟``) - those
#:   are correct Persian punctuation with distinct glyphs.
_ARABIC_TO_PERSIAN: Final[Mapping[str, str]] = {
    "\u064a": "\u06cc",  # ARABIC YEH -> FARSI YEH
    "\u0649": "\u06cc",  # ALEF MAKSURA -> FARSI YEH
    "\u06d2": "\u06cc",  # YEH BARREE -> FARSI YEH
    "\u06d3": "\u06cc",  # YEH BARREE WITH HAMZA -> FARSI YEH
    "\u0643": "\u06a9",  # ARABIC KAF -> KEHEH
    "\u06aa": "\u06a9",  # SWASH KAF -> KEHEH
    "\u0629": "\u0647",  # TEH MARBUTA -> HEH
    "\u06c0": "\u0647",  # HEH WITH YEH ABOVE -> HEH
    "\u06c1": "\u0647",  # HEH GOAL -> HEH
    "\u06c2": "\u0647",  # HEH GOAL WITH HAMZA -> HEH
    "\u0675": "\u0627",  # HIGH HAMZA ALEF -> ALEF
    "\u0676": "\u0648",  # HIGH HAMZA WAW -> WAW
    "\u0677": "\u0648",  # U WITH HAMZA ABOVE -> WAW
    "\u0678": "\u06cc",  # HIGH HAMZA YEH -> FARSI YEH
    "\u06a4": "\u0641",  # VEH -> FEH (Persian writes ف)
    "\u06a5": "\u0641",
    "\u0691": "\u0631",  # RREH -> REH
    "\u06ba": "\u0646",  # NOON GHUNNA -> NOON
    "\u06bb": "\u0646",
    "\u06bc": "\u0646",
    "\u06be": "\u0647",
    "\u0624": "\u0648\u0654",  # WAW WITH HAMZA stays visible as waw + hamza
}

#: Characters whose Unicode name starts with one of these is Persian/Arabic
#: script for the purposes of :func:`persian_letter_ratio`.
_PERSIAN_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x0620, 0x064A),  # Arabic letters
    (0x0671, 0x06D3),  # Arabic supplement and Persian letters
    (0x06FA, 0x06FF),
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)

#: Runs of horizontal whitespace, for :func:`collapse_whitespace`. ZWNJ is
#: deliberately absent: it is content, not spacing.
_HORIZONTAL_WS: Final[re.Pattern[str]] = re.compile(
    r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+"
)

#: Three or more newlines collapse to two. Two is a paragraph break the layout
#: modules act on; five is the model padding its answer.
_BLANK_RUN: Final[re.Pattern[str]] = re.compile(r"\n{3,}")

#: Translation tables built once. ``str.translate`` on a prepared table is an
#: order of magnitude faster than a comprehension, and this runs over every
#: character of a 150k-sample corpus several times.
_FOLD_TABLE: Final[dict[int, str]] = {
    ord(source): target for source, target in _ARABIC_TO_PERSIAN.items()
}
_DROP_INVISIBLE: Final[dict[int, None]] = {ord(ch): None for ch in _INVISIBLE_DROP}
_DROP_DIACRITICS: Final[dict[int, None]] = {ord(ch): None for ch in DIACRITICS}
_DROP_TATWEEL: Final[dict[int, None]] = {ord(TATWEEL): None}

_DIGIT_TABLES: Final[Mapping[str, dict[int, str]]] = {
    "persian": {
        **{ord(d): PERSIAN_DIGITS[i] for i, d in enumerate(LATIN_DIGITS)},
        **{ord(d): PERSIAN_DIGITS[i] for i, d in enumerate(ARABIC_INDIC_DIGITS)},
    },
    "latin": {
        **{ord(d): LATIN_DIGITS[i] for i, d in enumerate(PERSIAN_DIGITS)},
        **{ord(d): LATIN_DIGITS[i] for i, d in enumerate(ARABIC_INDIC_DIGITS)},
    },
    "arabic_indic": {
        **{ord(d): ARABIC_INDIC_DIGITS[i] for i, d in enumerate(LATIN_DIGITS)},
        **{ord(d): ARABIC_INDIC_DIGITS[i] for i, d in enumerate(PERSIAN_DIGITS)},
    },
}

#: Every digit codepoint in any of the three scripts, for density counting.
_ALL_DIGITS: Final[frozenset[str]] = frozenset(
    PERSIAN_DIGITS + LATIN_DIGITS + ARABIC_INDIC_DIGITS
)


def _is_persian_letter(character: str) -> bool:
    """Report whether a character belongs to the Arabic/Persian script block.

    Args:
        character: A single character.

    Returns:
        ``True`` for a letter in one of :data:`_PERSIAN_RANGES`, excluding
        digits and diacritics, which are counted separately.
    """
    if character in _ALL_DIGITS or character in DIACRITICS:
        return False
    code = ord(character)
    return any(low <= code <= high for low, high in _PERSIAN_RANGES)


def fold_arabic_letters(text: str) -> str:
    """Fold Arabic letter variants to their Persian equivalents.

    Args:
        text: Any string.

    Returns:
        The string with the pairs in :data:`_ARABIC_TO_PERSIAN` substituted.
        Punctuation that is genuinely Persian (``،``, ``؛``, ``؟``) is left
        alone, as are letters that carry a visible hamza (``ئ``, ``أ``, ``إ``,
        ``آ``) - those draw differently and so are real label content.
    """
    return text.translate(_FOLD_TABLE)


def strip_diacritics(text: str) -> str:
    """Remove tashkeel and related marks.

    Args:
        text: Any string.

    Returns:
        The string without any character in :data:`DIACRITICS`.
    """
    return text.translate(_DROP_DIACRITICS)


def strip_tatweel(text: str) -> str:
    """Remove the Arabic tatweel (kashida).

    Args:
        text: Any string.

    Returns:
        The string without :data:`TATWEEL`.
    """
    return text.translate(_DROP_TATWEEL)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines without touching ZWNJ.

    Args:
        text: Any string.

    Returns:
        The string with horizontal whitespace runs reduced to one space, runs
        of three or more newlines reduced to two, trailing spaces removed from
        each line, and the whole trimmed.
    """
    flattened = _HORIZONTAL_WS.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [line.strip(" ") for line in flattened.split("\n")]
    return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()


def to_digit_script(text: str, script: str) -> str:
    """Rewrite every digit in one script.

    Args:
        text: Any string.
        script: One of :data:`DIGIT_SCRIPTS`. ``mixed`` is a no-op, because a
            mixed sample is one whose digits were chosen per number upstream;
            folding them here would undo the very variation that was asked for.

    Returns:
        The string with digits in the requested script.

    Raises:
        ValueError: If ``script`` is not a known digit script. Defaulting a
            typo to Persian would produce a corpus whose ``digit_script``
            column lies about its own contents.
    """
    if script not in DIGIT_SCRIPTS:
        raise ValueError(
            f"unknown digit script {script!r}; expected one of {DIGIT_SCRIPTS}"
        )
    table = _DIGIT_TABLES.get(script)
    return text if table is None else text.translate(table)


def detect_digit_scripts(text: str) -> frozenset[str]:
    """Report which digit scripts actually occur in a string.

    Used by image QC as a cross-check: if a sample is labelled ``persian`` but
    this returns ``{"persian", "latin"}``, something between planning and
    rendering leaked a digit and the sample is rejected rather than shipped
    with a wrong label.

    Args:
        text: Any string.

    Returns:
        The subset of ``{"persian", "latin", "arabic_indic"}`` present; empty
        when the string holds no digits.
    """
    found: set[str] = set()
    for character in text:
        if character in PERSIAN_DIGITS:
            found.add("persian")
        elif character in LATIN_DIGITS:
            found.add("latin")
        elif character in ARABIC_INDIC_DIGITS:
            found.add("arabic_indic")
    return frozenset(found)


def prepare_render_text(
    text: str,
    *,
    digit_script: str = "persian",
    keep_diacritics: bool = False,
    keep_zwnj: bool = True,
    arabic_forms: bool = False,
) -> str:
    """Produce ``text_render``: exactly what the renderer will draw.

    The order of operations matters and is fixed here so no caller can get it
    wrong. Invisible controls go first (they would otherwise survive every
    later step), then letter folding, then diacritics, then digits, then
    whitespace - digits after folding because the fold table does not touch
    digits, and whitespace last because stripping marks can leave double
    spaces.

    Args:
        text: The model's text, already freed of preambles and fences by
            :mod:`fa_ocr_lora.diverse.textqc.filters`.
        digit_script: One of :data:`DIGIT_SCRIPTS`; from the sample's plan.
        keep_diacritics: Whether to preserve tashkeel. ``True`` only for the
            labelled diacritics subset.
        keep_zwnj: Whether to preserve ZWNJ. ``False`` removes it, which is a
            legitimate (and labelled) variation - plenty of real Persian text
            omits it - but must never happen by accident.
        arabic_forms: Whether to skip the Arabic-to-Persian letter fold, for
            the small labelled fraction that trains robustness to Arabic
            orthography.

    Returns:
        The exact string to render, which becomes the sample's label.
    """
    out = text.translate(_DROP_INVISIBLE) if keep_zwnj else text
    if not keep_zwnj:
        out = out.translate({**_DROP_INVISIBLE, ord(ZWNJ): None})
    out = strip_tatweel(out)
    if not arabic_forms:
        out = fold_arabic_letters(out)
    if not keep_diacritics:
        out = strip_diacritics(out)
    out = to_digit_script(out, digit_script)
    return collapse_whitespace(out)


def normalise_for_eval(text: str) -> str:
    """Produce ``text_norm``: the form CER and dedupe are computed on.

    Unconditionally folds everything that is a rendering distinction rather
    than a content distinction, so two samples whose pixels differ only in
    Arabic-versus-Persian yeh or Latin-versus-Persian digits compare equal.

    Applies NFC first, because an LLM occasionally emits a decomposed
    ``آ`` (alef + maddah) that is one glyph on the page but two codepoints in
    the string - an invisible off-by-one in every character-level metric.

    Args:
        text: Either ``text_render`` or a model prediction.

    Returns:
        The normalised comparison form: NFC, no invisibles except ZWNJ, no
        tatweel, Persian letter forms, no diacritics, Persian digits,
        collapsed whitespace.
    """
    out = unicodedata.normalize("NFC", text)
    out = out.translate(_DROP_INVISIBLE)
    out = strip_tatweel(out)
    out = fold_arabic_letters(out)
    out = strip_diacritics(out)
    out = to_digit_script(out, "persian")
    return collapse_whitespace(out)


def letter_counts(text: str) -> tuple[int, int]:
    """Count Persian-script letters and all letters.

    Args:
        text: Any string.

    Returns:
        ``(persian_letters, total_letters)``. Digits, punctuation and
        whitespace are excluded from both, so a line of ``۱۴۰۳/۰۵/۱۲`` does
        not read as 0% Persian.
    """
    persian = 0
    total = 0
    for character in text:
        if character in _ALL_DIGITS or character in DIACRITICS:
            continue
        if not character.isalpha():
            continue
        total += 1
        if _is_persian_letter(character):
            persian += 1
    return persian, total


def persian_letter_ratio(text: str) -> float:
    """Return the fraction of letters that are Persian script.

    Args:
        text: Any string.

    Returns:
        A value in ``[0, 1]``; ``1.0`` for a string with no letters at all,
        because a page of pure digits is not evidence of a wrong language and
        rejecting it would silently delete the numeric-heavy formats (receipts,
        timetables, invoices) that the corpus specifically wants.
    """
    persian, total = letter_counts(text)
    return 1.0 if total == 0 else persian / total


def count_digits(text: str) -> int:
    """Count digit characters in any of the three scripts.

    Args:
        text: Any string.

    Returns:
        The number of digits.
    """
    return sum(1 for character in text if character in _ALL_DIGITS)


def digit_density(text: str) -> float:
    """Return digits as a fraction of non-whitespace characters.

    Recorded per sample because digit recognition was the weakest axis of the
    previous model and the training sampler oversamples the top quartile.

    Args:
        text: Any string.

    Returns:
        A value in ``[0, 1]``; ``0.0`` for an empty string.
    """
    body = [character for character in text if not character.isspace()]
    if not body:
        return 0.0
    return sum(1 for character in body if character in _ALL_DIGITS) / len(body)


def has_zwnj(text: str) -> bool:
    """Report whether the string contains a zero-width non-joiner.

    Args:
        text: Any string.

    Returns:
        ``True`` if :data:`ZWNJ` occurs.
    """
    return ZWNJ in text


def has_diacritics(text: str) -> bool:
    """Report whether the string carries any tashkeel.

    Args:
        text: Any string.

    Returns:
        ``True`` if any character is in :data:`DIACRITICS`.
    """
    return any(character in DIACRITICS for character in text)


def diacritic_ratio(text: str) -> float:
    """Return diacritics as a fraction of Persian letters.

    Args:
        text: Any string.

    Returns:
        A value in ``[0, inf)``; ``0.0`` when there are no Persian letters.
        Fully vocalised Persian runs around 0.5 to 0.9, so a value above about
        0.2 marks a sample as belonging to the diacriticised subset.
    """
    persian, _ = letter_counts(text)
    if persian == 0:
        return 0.0
    return sum(1 for character in text if character in DIACRITICS) / persian


def word_count(text: str) -> int:
    """Count whitespace-separated tokens.

    ZWNJ does *not* split a word: ``می‌رود`` is one word, which is what a
    length target in the plan means and what a Persian reader would say.

    Args:
        text: Any string.

    Returns:
        The number of tokens.
    """
    return len(text.split())


def stems(words: Iterable[str]) -> frozenset[str]:
    """Reduce words to crude stems for the lexicon-usage check.

    This is deliberately not a real stemmer. The question stage 03 asks is
    only "did the model use the word I gave it, in some inflected form", and a
    light suffix strip answers that without a dependency. A real stemmer would
    also conflate unrelated words and inflate the hit rate.

    Args:
        words: Words to reduce, in any letter form.

    Returns:
        The normalised stems, ZWNJ removed and common Persian suffixes
        stripped.
    """
    out: set[str] = set()
    for word in words:
        base = normalise_for_eval(word).replace(ZWNJ, "")
        if not base:
            continue
        out.add(base)
        for suffix in _SUFFIXES:
            if len(base) > len(suffix) + 2 and base.endswith(suffix):
                out.add(base[: -len(suffix)])
                break
    return frozenset(out)


#: Persian suffixes stripped by :func:`stems`, longest first so ``ها`` does not
#: pre-empt ``هایی``. Verb endings are included because a lexicon verb is given
#: as an infinitive and used conjugated.
_SUFFIXES: Final[tuple[str, ...]] = (
    "هایی",
    "هایم",
    "هایت",
    "هایش",
    "ترین",
    "های",
    "ها",
    "ان",
    "یم",
    "ید",
    "ند",
    "ام",
    "ات",
    "اش",
    "تر",
    "ی",
    "م",
    "ت",
    "ش",
)
