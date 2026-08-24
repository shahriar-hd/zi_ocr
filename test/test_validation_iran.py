"""Unit tests for Jalali calendar arithmetic and numeral handling.

Two properties are load-bearing for the field audit and are defended here.

**Offset preservation.** :func:`normalize_digits` must not change the length of a
string. The confidence layer maps a field's character offsets onto the tokens that
generated it, so a normalisation that shifted positions by even one character would
attribute a digit's uncertainty to the wrong token - silently, and only on the
fields the audit cares about most.

**Jalali month lengths.** ``۱۴۰۲/۱۲/۳۰`` is a provable misread and ``۱۴۰۳/۱۲/۳۰``
is a real date, and the only thing separating them is whether 1403 is a leap year.
Getting the leap rule wrong in either direction breaks the one check in this whole
layer that works with no ground truth at all: too strict and it condemns valid
dates, too lax and it stops catching anything.

**Month names in order.** ``۲۲ مهر ۱۴۰۳`` is 11.7% of the dated labels in this
corpus, so resolving a written month to its number is on the path of the date
check. An off-by-one in the name table would turn Mehr into Shahrivar and hand the
calendar check a month with 31 days instead of 30 - wrong in the one direction
that stops it catching anything.

The leap years asserted below (1387, 1391, 1395, 1399, 1403) and the month names
are published values, not outputs of the implementation being tested.
"""

from __future__ import annotations

import pytest

from fa_ocr_lora.validation.iran import (
    DIGIT_CHARACTERS,
    JALALI_MONTH_NAMES,
    all_digits_identical,
    digits_of,
    jalali_date_is_valid,
    jalali_is_leap,
    jalali_month_length,
    normalize_digits,
    parse_jalali_date,
)


def test_normalize_digits_preserves_length():
    """Offsets into the original must index the same position in the result.

    The confidence layer depends on this: it resolves a field's ``start``/``end``
    against the decoded string, so any normalisation that changed the length would
    map a digit onto a neighbouring token's certainty.
    """
    for text in ["۱۲۳۴۵", "شماره ثبت: ۱۲۳۴۵", "٠١٢٣٤٥٦٧٨٩", "۱۲٬۰۰۰٫۵", "no digits"]:
        assert len(normalize_digits(text)) == len(text), text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("۱۲۳۴۵", "12345"),
        ("٠١٢٣٤٥٦٧٨٩", "0123456789"),
        ("0123456789", "0123456789"),
        ("۱۲٬۰۰۰", "12,000"),
        ("۱۲٫۵", "12.5"),
    ],
)
def test_normalize_digits_maps_all_three_numeral_families(text, expected):
    assert normalize_digits(text) == expected


def test_normalize_digits_leaves_persian_letters_alone():
    """Only digits and the two Arabic separators are touched.

    Folding ي to ی or ك to ک here would hide exactly the substitutions this
    adapter exists to remove.
    """
    assert normalize_digits("نامه اداری") == "نامه اداری"
    assert normalize_digits("مبلغ ۱۲,۰۰۰ ریال") == "مبلغ 12,000 ریال"


def test_digits_of_extracts_across_families_in_order():
    assert digits_of("شماره ۱۲۳ / کد ٤٥٦ / no 789") == "123456789"
    assert digits_of("no digits here") == ""


def test_digit_characters_covers_every_family():
    """The character set must be complete, since it drives every digit comparison."""
    assert len(DIGIT_CHARACTERS) == 30
    assert len(set(DIGIT_CHARACTERS)) == 30
    for family_start in ("۰", "٠", "0"):
        for offset in range(10):
            assert chr(ord(family_start) + offset) in DIGIT_CHARACTERS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("۱۱۱۱۱۱۱۱۱۱", True),
        ("0000", True),
        ("۱1١", True),
        ("۱۲۳۴۵", False),
        ("۷", False),
        ("", False),
        ("ابجد", False),
    ],
)
def test_all_digits_identical(value, expected):
    """A repdigit run is the signature of a decoder stuck in a loop.

    ``۱1١`` is True on purpose: the same digit in three scripts is still one
    repeated digit, and the numeral family is reported by a different check.
    A single digit is False because it is trivially uniform and says nothing.
    """
    assert all_digits_identical(value) is expected


@pytest.mark.parametrize("year", [1387, 1391, 1395, 1399, 1403])
def test_known_leap_years(year):
    assert jalali_is_leap(year) is True


@pytest.mark.parametrize("year", [1400, 1401, 1402, 1404, 1405])
def test_known_common_years(year):
    assert jalali_is_leap(year) is False


@pytest.mark.parametrize(
    ("month", "length"),
    [(1, 31), (6, 31), (7, 30), (11, 30)],
)
def test_month_lengths_are_leap_independent_outside_esfand(month, length):
    assert jalali_month_length(1402, month) == length
    assert jalali_month_length(1403, month) == length


def test_esfand_is_the_only_month_whose_length_moves():
    assert jalali_month_length(1402, 12) == 29
    assert jalali_month_length(1403, 12) == 30


def test_an_out_of_range_month_has_zero_days():
    """So a caller can treat a bad month as a zero-day month with no special case."""
    assert jalali_month_length(1403, 0) == 0
    assert jalali_month_length(1403, 13) == 0


def test_esfand_thirty_is_rejected_in_a_common_year_and_accepted_in_a_leap_one():
    """The one field-level check that needs no reference at all.

    This is also the case the repository's own generator gets wrong: it draws the
    day uniformly from 1-30 for every month, so a small fraction of training
    targets carry a date that never existed. See docs/decisions.md.
    """
    assert jalali_date_is_valid(1402, 12, 30) is False
    assert jalali_date_is_valid(1403, 12, 30) is True
    assert jalali_date_is_valid(1402, 12, 29) is True


@pytest.mark.parametrize(
    ("year", "month", "day"),
    [
        (1403, 1, 0),
        (1403, 1, 32),
        (1403, 7, 31),
        (1403, 0, 1),
        (1403, 13, 1),
        (2024, 5, 12),
        (14, 5, 12),
    ],
)
def test_invalid_dates_are_rejected(year, month, day):
    """Including the two failure modes that actually occur.

    ``2024`` is a Gregorian year read straight off the page and ``14`` a truncated
    one; both are outside the accepted range rather than merely unusual.
    """
    assert jalali_date_is_valid(year, month, day) is False


def test_valid_dates_are_accepted():
    for year, month, day in [(1403, 5, 12), (1400, 1, 1), (1399, 12, 30), (1402, 7, 30)]:
        assert jalali_date_is_valid(year, month, day) is True, (year, month, day)


@pytest.mark.parametrize(
    ("number", "name"),
    [
        (1, "فروردین"),
        (2, "اردیبهشت"),
        (3, "خرداد"),
        (4, "تیر"),
        (5, "مرداد"),
        (6, "شهریور"),
        (7, "مهر"),
        (8, "آبان"),
        (9, "آذر"),
        (10, "دی"),
        (11, "بهمن"),
        (12, "اسفند"),
    ],
)
def test_month_names_are_in_calendar_order(number, name):
    """Index + 1 must be the month number, for all twelve.

    Pinned name by name rather than by length, because an off-by-one here is
    invisible: it produces a valid month number for a valid month name, and only
    shows up as a date check that quietly stops rejecting Esfand 30.
    """
    assert JALALI_MONTH_NAMES[number - 1] == name
    assert parse_jalali_date(f"۱۵ {name} ۱۴۰۳") == (1403, number, 15)


def test_the_month_table_is_exactly_twelve_long():
    """A thirteenth entry or a missing one would silently shift every later month."""
    assert len(JALALI_MONTH_NAMES) == 12
    assert len(set(JALALI_MONTH_NAMES)) == 12


def test_a_month_name_with_the_madda_dropped_still_resolves():
    """``ابان`` for ``آبان`` is a one-character misread, not a broken field.

    The model drops the diacritic. Letting that also fail the field's format
    check would report one wrong character as a structurally malformed date.
    """
    assert parse_jalali_date("۲۲ ابان ۱۴۰۳") == (1403, 8, 22)
    assert parse_jalali_date("۲۲ اذر ۱۴۰۳") == (1403, 9, 22)


def test_day_first_and_year_first_are_told_apart_by_value_not_by_assumption():
    """Both orders occur in this corpus, so the order is read off the digits.

    A four-digit group is the year wherever it sits; a leading group above 12
    cannot be a month. Guessing a fixed order instead would silently transpose
    day and month on half the dates and hand the calendar check nonsense.
    """
    assert parse_jalali_date("۱۴۰۳/۰۷/۲۲") == (1403, 7, 22)
    assert parse_jalali_date("۲۷/۰۹/۱۴۰۳") == (1403, 9, 27)
