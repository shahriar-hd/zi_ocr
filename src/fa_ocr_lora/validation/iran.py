"""Digit handling and Jalali calendar arithmetic for Persian documents.

Two jobs, both prerequisites for auditing a sensitive field.

**Numeral normalisation.** Persian text mixes three digit families - Persian
۰-۹, Arabic-Indic ٠-٩ and Latin 0-9 - and the base model confuses them with each
other freely. Comparing a predicted ``شماره ثبت`` against a reference has to
decide whether ``۱۲۳۴۵`` and ``12345`` are the same number. They are, as a
number; they are not, as a transcription. So both comparisons are made and
reported separately, and :func:`normalize_digits` is what makes the first one
possible. It preserves string length, which matters because the confidence layer
maps character offsets back onto generated tokens.

**Jalali validity.** A date is the one sensitive field whose correctness is
partly decidable with no reference at all: ``۱۴۰۲/۱۲/۳۰`` is not a day that
existed, because Esfand has 29 days in a common year. ``۱۴۰۳/۱۲/۳۰`` is fine -
1403 is a leap year. Anything the model reads that fails this is provably a
misread, which is the kind of finding a production system can act on.

The same check also catches a real bug in this repository's own synthetic
generator, which draws the day uniformly from 1-30 for every month: a small
fraction of training targets carry an Esfand 30 in a common year. See
``docs/decisions.md``.

Deliberately *not* here: the modulo-11 checksums for کد ملی and شناسه ملی. They
work, and they detect ~99% of single-digit misreads, but the corpus generates
national ids as uniform random digits, so a checksum would flag ~91% of the
ground truth as invalid and tell you nothing about the model. What matters for
that field is whether the digits the model read are the digits that were there -
see :mod:`fa_ocr_lora.validation.fields`.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "DIGIT_CHARACTERS",
    "JALALI_LEAP_RESIDUES",
    "JALALI_MONTH_NAMES",
    "all_digits_identical",
    "digits_of",
    "jalali_date_is_valid",
    "jalali_is_leap",
    "jalali_month_length",
    "normalize_digits",
    "parse_jalali_date",
]

#: Every digit character the corpora use, across all three numeral families.
#: Duplicated from :data:`fa_ocr_lora.training.metrics.DIGITS` on purpose: this
#: package must be importable by the inference layer without dragging in the
#: training package and its torch dependency.
DIGIT_CHARACTERS: Final[str] = "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩0123456789"

_DIGIT_SET: Final[frozenset[str]] = frozenset(DIGIT_CHARACTERS)

#: Years whose remainder modulo 33 marks a Jalali leap year. The 33-year-cycle
#: rule, which is exact across the whole range administrative correspondence can
#: carry and wrong only for years far outside it. Verified against the known leap
#: years 1387, 1391, 1395, 1399 and 1403.
JALALI_LEAP_RESIDUES: Final[frozenset[int]] = frozenset({1, 5, 9, 13, 17, 22, 26, 30})

#: Widest year range a date field may claim. Wide enough for any real document,
#: narrow enough to reject the two failure modes that actually occur: a Gregorian
#: year read straight off the page (``2024``), and a truncated one (``14``).
_YEAR_RANGE: Final[tuple[int, int]] = (1200, 1499)

#: Persian and Arabic-Indic digits mapped onto Latin, plus the Arabic decimal and
#: thousands separators. Every entry is one character wide in both directions, so
#: translation preserves offsets.
_DIGIT_TRANSLATION: Final[dict[int, str]] = {
    **{ord("۰") + offset: str(offset) for offset in range(10)},
    **{ord("٠") + offset: str(offset) for offset in range(10)},
    ord("٫"): ".",
    ord("٬"): ",",
}


def normalize_digits(text: str) -> str:
    """Rewrite Persian and Arabic-Indic digits as Latin ones.

    Nothing else about the string is touched - letters, separators and whitespace
    pass through unchanged, and the result has the same length as the input, so
    a character offset into one indexes the same position in the other.

    Args:
        text: Any string, in any mixture of the three numeral families.

    Returns:
        The string with digits in Latin, and the Arabic decimal and thousands
        separators as ``.`` and ``,``.
    """
    return text.translate(_DIGIT_TRANSLATION)


def digits_of(text: str) -> str:
    """Extract the digits of a string as Latin digits, in order.

    The numeral-blind view of a value. ``۱۲۳۴۵`` and ``12345`` both reduce to
    ``"12345"``, which is what makes "did the model read the right number?"
    answerable separately from "did it read it in the right script?".

    Args:
        text: Any string.

    Returns:
        The digits, concatenated and normalised to Latin.
    """
    return normalize_digits("".join(char for char in text if char in _DIGIT_SET))


def all_digits_identical(text: str) -> bool:
    """Report whether a value's digits are all the same digit.

    The signature of a decoder stuck in a repetition loop, and the cheapest
    structural fault to detect: no identifier in real correspondence is
    ``۱۱۱۱۱۱۱۱۱۱``, so a value like it is a misread regardless of what the
    reference says. Two or more digits are required, since a single-digit value
    is trivially uniform and says nothing.

    Args:
        text: The field value, in any numeral family.

    Returns:
        ``True`` if the value has at least two digits and they are all equal.
    """
    digits = digits_of(text)
    return len(digits) >= 2 and len(set(digits)) == 1


def jalali_is_leap(year: int) -> bool:
    """Report whether a Jalali year has 366 days.

    Args:
        year: Jalali year, e.g. ``1403``.

    Returns:
        ``True`` for a leap year under the 33-year-cycle rule.
    """
    return year % 33 in JALALI_LEAP_RESIDUES


def jalali_month_length(year: int, month: int) -> int:
    """Days in one Jalali month.

    Args:
        year: Jalali year.
        month: Month number, 1-12.

    Returns:
        31 for Farvardin-Shahrivar, 30 for Mehr-Bahman, and 29 or 30 for Esfand
        by whether the year is leap. ``0`` for a month outside 1-12, so a caller
        can treat an out-of-range month as a zero-day month without a special
        case.
    """
    if not 1 <= month <= 12:
        return 0
    if month <= 6:
        return 31
    if month <= 11:
        return 30
    return 30 if jalali_is_leap(year) else 29


#: The twelve Jalali months in order, so index + 1 is the month number. Needed
#: because a تاریخ is not always written numerically: 11.7% of the dated labels in
#: this corpus read ``۲۲ مهر ۱۴۰۳``. A date pattern that accepts only digits
#: scores every one of those malformed, on reference text, where the model cannot
#: possibly be at fault.
JALALI_MONTH_NAMES: Final[tuple[str, ...]] = (
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
)

#: Month spellings that are not the canonical name, mapped to their month number.
#: ``امرداد`` is the archaic form of Mordad. The rest are the same names with the
#: madda dropped from the initial alef - not a spelling anyone chooses, but what
#: an OCR model produces when it misses the diacritic, and this map is what keeps
#: that one misread character from also being reported as a malformed field.
_MONTH_ALIASES: Final[dict[str, int]] = {"امرداد": 5, "ابان": 8, "اذر": 9}

#: Characters dropped before a month name is looked up: the zero-width non-joiner
#: and the two Arabic presentation forms of alef, which Persian text uses
#: interchangeably with the bare letter.
_MONTH_STRIP: Final[dict[int, str | None]] = {
    ord("\u200c"): None,
    ord("\u200f"): None,
    ord("\u064a"): "\u06cc",
    ord("\u0643"): "\u06a9",
}

#: Suffix a Persian writer may append to a month name - ``مهرماه`` for ``مهر`` -
#: joined or spaced. Stripped before lookup rather than listed as fifteen more
#: aliases.
_MONTH_SUFFIX: Final[str] = "ماه"

#: Trimmed from both ends of a written month before lookup. The numeric
#: separators are here because ``۲۷/مهر/۱۴۰۳`` occurs in this corpus: a date
#: punctuated like the numeric form but with the month spelled out.
_MONTH_TRIM: Final[str] = " /-.\u200c"

#: Splits a numeric date on any separator the shipped pattern allows.
_DATE_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"[/\-.]")

#: Matches ``DD <month name> YYYY``, the spelled-out form.
_SPELLED_DATE: Final[re.Pattern[str]] = re.compile(r"^(\d{1,2})\s*([^\d\s]+)\s*(\d{4})$")

#: Below this, a two-digit leading number is a day rather than a year. No Jalali
#: month number exceeds 12, so a value above it can only be a day - which is what
#: disambiguates ``۲۷/۰۹/۱۴۰۳`` (day first) from ``۱۴۰۳/۰۷/۲۲`` (year first)
#: without guessing.
_MAX_MONTH: Final[int] = 12


def _month_number(name: str) -> int:
    """Resolve a written month name to its number.

    Args:
        name: The month as written, in any of the spellings the corpus and the
            model produce.

    Returns:
        The month number 1-12, or ``0`` when the text names no month.
    """
    cleaned = name.translate(_MONTH_STRIP).strip(_MONTH_TRIM)
    if cleaned.endswith(_MONTH_SUFFIX) and cleaned != _MONTH_SUFFIX:
        cleaned = cleaned[: -len(_MONTH_SUFFIX)].strip(_MONTH_TRIM)
    if cleaned in JALALI_MONTH_NAMES:
        return JALALI_MONTH_NAMES.index(cleaned) + 1
    if cleaned in _MONTH_ALIASES:
        return _MONTH_ALIASES[cleaned]
    bare = cleaned.replace("\u0622", "\u0627")
    for index, month in enumerate(JALALI_MONTH_NAMES):
        if bare == month.replace("\u0622", "\u0627"):
            return index + 1
    return 0


def parse_jalali_date(value: str) -> tuple[int, int, int] | None:
    """Read a date value written in any of the three forms this corpus uses.

    Which form a letter uses is not the model's choice and not a correctness
    signal, so all three resolve to the same triple:

    * ``۱۴۰۳/۰۷/۲۲`` - year first, the form ``synthesize_jalali_date`` emits;
    * ``۲۷/۰۹/۱۴۰۳`` - day first, which appears in LLM-drafted letter bodies;
    * ``۲۲ مهر ۱۴۰۳`` - spelled-out month, 11.7% of the dated labels here,
      also written ``۲۷/مهر/۱۴۰۳`` and ``۲۲ مهرماه ۱۴۰۳``.

    Order is resolved from the values rather than assumed: a leading group above
    12 cannot be a month, and a four-digit group can only be the year. An
    ambiguous ``۰۳/۰۷/۱۴`` is read year-last, matching the day-first majority of
    the ambiguous cases in this corpus.

    Args:
        value: The matched field value, in any numeral family.

    Returns:
        ``(year, month, day)``, or ``None`` when the text is not a date in any
        recognised form. ``None`` means "undecidable", not "invalid" - the
        validity question is :func:`jalali_date_is_valid`, and the two are kept
        apart so an unparseable value is not reported as an impossible day.
    """
    text = normalize_digits(value).strip()
    spelled = _SPELLED_DATE.match(text)
    if spelled:
        month = _month_number(spelled.group(2))
        if not month:
            return None
        return int(spelled.group(3)), month, int(spelled.group(1))

    parts = _DATE_SEPARATORS.split(text)
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    first, middle, last = (int(part) for part in parts)
    if len(parts[0]) == 4:
        return first, middle, last
    if len(parts[2]) == 4:
        return last, middle, first
    if first > _MAX_MONTH:
        return last, middle, first
    return first, middle, last


def jalali_date_is_valid(year: int, month: int, day: int) -> bool:
    """Check that a Jalali year/month/day names a day that existed.

    Args:
        year: Jalali year.
        month: Month number.
        day: Day of month.

    Returns:
        ``True`` if the date is real.
    """
    low, high = _YEAR_RANGE
    if not low <= year <= high:
        return False
    return 1 <= day <= jalali_month_length(year, month)
