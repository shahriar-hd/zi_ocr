"""Unit tests for field extraction, comparison and the audit report.

Extraction has two hazards, and both would corrupt a metric silently rather than
raise, so both get direct tests.

**Prefix-colliding labels.** ``شماره`` (letter_number) is a prefix of
``شماره ثبت`` (registration_number). Python's alternation returns the first branch
that matches at a position, so unless every label of every field is sorted by
descending length - globally, not per field - every registration number in the
corpus is recorded as a letter number, and the highest-priority field shows a
perfect score it never earned.

**The whitespace-only separator.** The corpus draws it 20% of the time. If the
extractor cannot cross it, a fifth of all fields read as "label present, no
readable value".

**Labels that are also ordinary words.** ``تاریخ`` and ``شماره`` occur in running
prose - "در تاریخ ۱۳۸۹ مبنی بر ..." is a sentence, not a date field. Reporting
those as malformed fields accounted for 67% of all malformed findings measured on
this corpus's *reference* text, where the model cannot be at fault, so the
suppression rule and its safety property both get direct tests.

Beyond those, the tests pin the distinctions the report is built on: a malformed
field is not an absent one, a script error is not a misread digit, a hallucinated
field is not a wrong one, and the reference-free checks stay reference-free.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from fa_ocr_lora.validation import (
    FieldAuditor,
    FieldConfidence,
    FieldKind,
    FindingKind,
    LazySpanConfidence,
    TokenConfidence,
    compare_fields,
    extract_fields,
    load_field_validation_config,
    parse_jalali_date,
    summarize_fields,
)


@pytest.fixture(scope="module")
def rules():
    """The committed field-validation rules.

    Returns:
        The loaded configuration.
    """
    return load_field_validation_config()


def _kinds(text, rules):
    """Extract and reduce to ``(kind, value)`` pairs.

    Args:
        text: Transcription to extract from.
        rules: Loaded configuration.

    Returns:
        One tuple per extracted field, in reading order.
    """
    return [(item.kind, item.value) for item in extract_fields(text, rules)]


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def test_registration_number_wins_over_the_bare_letter_number_label(rules):
    """The prefix collision. Without longest-label-first this returns letter_number.

    ``شماره ثبت`` and ``شماره`` are both real labels in the corpus, and the second
    is a prefix of the first, so matching shortest-first would silently reclassify
    every registration number on every page.
    """
    assert _kinds("شماره ثبت: ۱۲۳۴۵", rules) == [(FieldKind.REGISTRATION_NUMBER, "۱۲۳۴۵")]


def test_the_bare_label_still_finds_a_letter_number(rules):
    """Longest-first must not make ``شماره`` unreachable, only lower-priority."""
    assert _kinds("شماره: ۴۵۶/ب/۰۴", rules) == [(FieldKind.LETTER_NUMBER, "۴۵۶/ب/۰۴")]


def test_a_whitespace_only_separator_is_crossed(rules):
    """20% of the corpus separates a label from its value with whitespace alone."""
    assert _kinds("شمارهٔ ثبت ۹۸۷۶۵", rules) == [(FieldKind.REGISTRATION_NUMBER, "۹۸۷۶۵")]


@pytest.mark.parametrize("separator", [":", "ـــــ", ".....", " "])
def test_every_corpus_separator_is_crossed(rules, separator):
    text = f"تاریخ {separator} ۱۴۰۳/۰۵/۱۲"
    assert _kinds(text, rules) == [(FieldKind.DATE, "۱۴۰۳/۰۵/۱۲")]


@pytest.mark.parametrize(
    "value", ["۱۴۰۳/۰۵/۱۲", "1403/05/12", "١٤٠٣/٠٥/١٢", "۱۴۰۳-۰۵-۱۲", "۱۴۰۳.۰۵.۱۲"]
)
def test_all_three_numeral_families_and_all_separators_extract(rules, value):
    """A Persian ۵ read as a Latin 5 is a transcription error, not a malformed field.

    Rejecting it during extraction would fold two different failures into one
    number; the audit reports script-exact and digits-only agreement separately
    instead.
    """
    assert _kinds(f"تاریخ: {value}", rules) == [(FieldKind.DATE, value)]


def test_a_label_with_no_readable_value_is_malformed_not_absent(rules):
    """The reference-free fault the audit exists to surface.

    Dropping the field would score it as absent, which reads as "the page did not
    have one" rather than "the model could not read the one it had".
    """
    found = extract_fields("شماره ثبت: ابجد", rules)
    assert len(found) == 1
    assert found[0].kind is FieldKind.REGISTRATION_NUMBER
    assert found[0].format_ok is False
    assert found[0].value == ""
    assert found[0].excerpt == ": ابجد"


# --------------------------------------------------------------------------
# Labels used as ordinary words
# --------------------------------------------------------------------------


#: Lines lifted from the corpus measurement, each carrying a label that is not
#: introducing a field. Regression cases, not invented input.
PROSE_LINES = [
    "در تاریخ ۱۳۸۹ مبنی بر این موضوع مکاتبه شد",
    "شماره تماس: ۰۲۱-۷۷۷۷۷۷۷۷",
    "تاریخ پایان پروژه",
    "شماره ثبت فعلی شرکت ۱۰۲۳۴۵ می‌باشد و ادامه دارد",
    "شناسه ملی مربوط به مدیر عامل اعلام گردید",
]


@pytest.mark.parametrize("line", PROSE_LINES)
def test_a_label_used_as_a_word_is_not_a_field(rules, line):
    """A label in prose must not be reported as a malformed field.

    These five shapes were 67% of every malformed finding the audit produced on
    reference text. A finding on the ground truth cannot be a model error, so
    each one was pure noise in the metric the user asked for.
    """
    assert extract_fields(line, rules) == ()


@pytest.mark.parametrize("line", PROSE_LINES)
def test_the_literal_reading_still_reports_them(rules, line):
    """The suppression must be the flag's doing, not an accident elsewhere.

    With ``require_field_context`` off these same lines produce findings, which
    is what makes the flag a usable measuring instrument on a new corpus: turn it
    off and the difference is exactly what the rule is suppressing.
    """
    literal = dataclasses.replace(rules, require_field_context=False)
    found = extract_fields(line, literal)
    assert found
    assert all(item.format_ok is False for item in found)


def test_an_explicit_separator_makes_it_a_field_whatever_follows(rules):
    """Someone who wrote ``شماره ثبت:`` was introducing a field.

    So junk after the separator stays a finding - this is the case the audit is
    for, and suppressing it would be the expensive kind of false negative.
    """
    found = extract_fields("شماره ثبت: ابجد", rules)
    assert len(found) == 1
    assert found[0].format_ok is False


def test_a_column_aligned_label_opening_a_line_is_a_field(rules):
    """The layout the empty separator exists for.

    A letterhead that aligns values in a column writes ``شناسه ملی۲۴۶۷۵۸۱۳۲``
    with no separator at all. The value here is nine digits where the rules want
    ten or eleven, so it must be reported - it is a real corpus defect.
    """
    found = extract_fields("شناسه ملی۲۴۶۷۵۸۱۳۲", rules)
    assert len(found) == 1
    assert found[0].kind is FieldKind.NATIONAL_ID
    assert found[0].format_ok is False


def test_a_readable_field_survives_regardless_of_context(rules):
    """The safety property: the rule is consulted only when no value matched.

    Two fields on one line with no separators and neither at the line start -
    the exact shape the suppression rule would otherwise reject - both extract,
    because a label followed by a well-formed value is a field by demonstration.
    """
    text = "شماره ثبت ۴۵۶۷۸    تاریخ ۱۴۰۳/۰۷/۲۲"
    assert _kinds(text, rules) == [
        (FieldKind.REGISTRATION_NUMBER, "۴۵۶۷۸"),
        (FieldKind.DATE, "۱۴۰۳/۰۷/۲۲"),
    ]
    literal = dataclasses.replace(rules, require_field_context=False)
    assert _kinds(text, literal) == _kinds(text, rules)


# --------------------------------------------------------------------------
# The written forms of a date
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("۱۴۰۳/۰۷/۲۲", (1403, 7, 22)),
        ("۱۴۰۳-۰۷-۲۲", (1403, 7, 22)),
        ("۲۷/۰۹/۱۴۰۳", (1403, 9, 27)),
        ("۲۲ مهر ۱۴۰۳", (1403, 7, 22)),
        ("۱۵ اردیبهشت ۱۴۰۴", (1404, 2, 15)),
        ("۲۲ مهرماه ۱۴۰۳", (1403, 7, 22)),
        ("۲۷/مهر/۱۴۰۳", (1403, 7, 27)),
        ("۲۲ ابان ۱۴۰۳", (1403, 8, 22)),
        ("22 خرداد 1403", (1403, 3, 22)),
    ],
)
def test_every_written_date_form_resolves_to_the_same_triple(written, expected):
    """Which form a letter uses is not the model's choice and not an error.

    A pattern accepting only the numeric year-first form scored 97.2% of every
    تاریخ in this corpus's reference text as malformed. The spelled-out month
    alone is 11.7% of dated labels; year-first numeric is 2.8%.
    """
    assert parse_jalali_date(written) == expected


@pytest.mark.parametrize("written", ["ابجد", "", "۱۳۸۹", "۲۲ ماه ۱۴۰۳", "۱/۲"])
def test_a_non_date_is_undecidable_not_invalid(written):
    """``None`` means "not a date", which is not the same as "an impossible day".

    Folding the two together would report an unparseable value as a calendar
    fault, and a calendar fault is a claim that the digits were misread.
    """
    assert parse_jalali_date(written) is None


@pytest.mark.parametrize(
    "written", ["۲۲ مهر ۱۴۰۳", "۲۷/مهر/۱۴۰۳", "۲۷/۰۹/۱۴۰۳", "۲۲ مهرماه ۱۴۰۳"]
)
def test_the_committed_pattern_accepts_every_form_it_documents(rules, written):
    """The YAML pattern and ``parse_jalali_date`` must agree on what a date is.

    They are edited separately - one is configuration, the other is code - so a
    form the pattern matches but the parser cannot read would surface as a
    calendar fault on a perfectly good date.
    """
    found = extract_fields(f"تاریخ: {written}", rules)
    assert len(found) == 1
    assert found[0].value == written
    assert found[0].check_ok is True


def test_a_spelled_out_date_still_fails_the_calendar_check_when_impossible(rules):
    """Widening the pattern must not weaken the check behind it.

    Esfand has 29 days in a common year whether the month is spelled or numbered.
    """
    found = extract_fields("تاریخ: ۳۰ اسفند ۱۴۰۲", rules)
    assert len(found) == 1
    assert found[0].format_ok is True
    assert found[0].check_ok is False


def test_an_amount_is_found_with_no_label_at_all(rules):
    """A table cell has lost its header by the time a flat transcription arrives.

    The thousands separator is what makes the label unnecessary: no other field in
    the corpus is digit-grouped, so this cannot collide with a registration number
    or a national id.
    """
    assert _kinds("جمع کل ۱۲,۰۰۰,۰۰۰ ریال", rules) == [(FieldKind.AMOUNT, "۱۲,۰۰۰,۰۰۰")]


def test_two_amounts_on_one_line_are_both_found(rules):
    found = _kinds("۱,۵۰۰,۰۰۰ و ۲۵,۰۰۰", rules)
    assert found == [(FieldKind.AMOUNT, "۱,۵۰۰,۰۰۰"), (FieldKind.AMOUNT, "۲۵,۰۰۰")]


def test_an_ungrouped_number_is_not_an_amount(rules):
    """Otherwise every bare five-digit number on the page becomes a currency value."""
    assert _kinds("موجودی ۱۲۰۰۰۰۰ ریال", rules) == []


def test_an_unlabelled_pattern_cannot_re_report_an_already_matched_value(tmp_path):
    """A span consumed by a labelled field is off limits to the pattern-only pass.

    Without the guard, a labelled value that happens to satisfy an unlabelled
    field's pattern is counted twice - once as its own field and once as an amount
    - which inflates the amount denominator with values that are not amounts. The
    committed rules cannot express that collision (no labelled pattern admits a
    comma), so it is built here rather than left untested.
    """
    path = tmp_path / "overlap.yaml"
    path.write_text(
        "fields:\n"
        "  registration_number:\n"
        '    labels: ["شماره ثبت"]\n'
        "    pattern: '[۰-۹0-9,]{4,12}'\n"
        "  amount:\n"
        "    labels: []\n"
        "    pattern: '[۰-۹0-9]{1,3}(?:,[۰-۹0-9]{3})+'\n",
        encoding="utf-8",
    )
    overlapping = load_field_validation_config(path)
    found = extract_fields("شماره ثبت: ۱۲,۳۴۵", overlapping)
    assert [item.kind for item in found] == [FieldKind.REGISTRATION_NUMBER]
    assert found[0].value == "۱۲,۳۴۵"


def test_the_value_is_anchored_at_the_label_not_searched_for(rules):
    """A five-digit pattern must not reach into a number later in the paragraph.

    Anchoring is what keeps a malformed field malformed instead of borrowing a
    plausible value from elsewhere and scoring a field the model got wrong.
    """
    found = extract_fields("شماره ثبت: ندارد و تلفن ۸۸۷۷۶۶۵۵ است", rules)
    registration = [f for f in found if f.kind is FieldKind.REGISTRATION_NUMBER]
    assert len(registration) == 1
    assert registration[0].value == ""


def test_extraction_is_ordered_by_position(rules):
    text = "تاریخ: ۱۴۰۳/۰۵/۱۲\nشماره ثبت: ۱۲۳۴۵\nکد ملی ۰۴۹۹۳۷۰۸۹۹"
    found = extract_fields(text, rules)
    assert [item.start for item in found] == sorted(item.start for item in found)
    assert [item.kind for item in found] == [
        FieldKind.DATE,
        FieldKind.REGISTRATION_NUMBER,
        FieldKind.NATIONAL_ID,
    ]


def test_offsets_point_at_the_value_in_the_source_text(rules):
    """The confidence layer indexes the decoded string with these offsets."""
    text = "شماره ثبت: ۱۲۳۴۵ و تاریخ: ۱۴۰۳/۰۵/۱۲"
    for item in extract_fields(text, rules):
        assert text[item.start : item.end] == item.value


# --------------------------------------------------------------------------
# Reference-free checks
# --------------------------------------------------------------------------


def test_esfand_thirty_fails_the_calendar_check_in_a_common_year(rules):
    """Provably a misread, with no ground truth needed.

    Also the fault that catches this repository's own generator, which draws the
    day uniformly from 1-30 for every month.
    """
    (found,) = extract_fields("تاریخ: ۱۴۰۲/۱۲/۳۰", rules)
    assert found.check_ok is False
    (valid,) = extract_fields("تاریخ: ۱۴۰۳/۱۲/۳۰", rules)
    assert valid.check_ok is True


def test_a_field_with_no_semantic_check_reports_none(rules):
    """``None`` and ``False`` are different verdicts and must not collapse.

    ``None`` means nothing further is decidable from the value; ``False`` means the
    value is provably wrong.
    """
    (found,) = extract_fields("شماره ثبت: ۱۲۳۴۵", rules)
    assert found.check_ok is None


def test_a_repdigit_national_id_is_flagged_without_a_checksum(rules):
    """No checksum is applied - the corpus generates these digits at random.

    A modulo-11 check would condemn ~91% of the ground truth and say nothing about
    the model. What is checkable with no reference is that the value is not one
    digit repeated, which is the signature of a decoder stuck in a loop.
    """
    (found,) = extract_fields("کد ملی ۱۱۱۱۱۱۱۱۱۱", rules)
    assert found.repeated_digits is True
    assert found.check_ok is None
    (ordinary,) = extract_fields("کد ملی ۰۴۹۹۳۷۰۸۹۹", rules)
    assert ordinary.repeated_digits is False


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _compare(reference, hypothesis, rules):
    """Extract both sides and compare.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model output.
        rules: Loaded configuration.

    Returns:
        The comparisons that have a field on at least one side.
    """
    return [
        item
        for item in compare_fields(
            extract_fields(reference, rules), extract_fields(hypothesis, rules)
        )
        if item.reference is not None or item.hypothesis is not None
    ]


def test_an_identical_reading_is_exact_and_digit_equal(rules):
    (comparison,) = _compare("شماره ثبت: ۱۲۳۴۵", "شماره ثبت: ۱۲۳۴۵", rules)
    assert comparison.exact is True
    assert comparison.digits_equal is True


def test_one_wrong_digit_fails_both_comparisons(rules):
    (comparison,) = _compare("شماره ثبت: ۱۲۳۴۵", "شماره ثبت: ۱۲۳۴۶", rules)
    assert comparison.exact is False
    assert comparison.digits_equal is False


def test_a_numeral_family_error_fails_exact_but_passes_digits(rules):
    """The distinction that separates a normalisation bug from a misread digit.

    Collapsing the two into one number would leave nobody able to tell which of
    the two problems they have.
    """
    (comparison,) = _compare("تاریخ: ۱۴۰۳/۰۵/۱۲", "تاریخ: 1403/05/12", rules)
    assert comparison.exact is False
    assert comparison.digits_equal is True


def test_whitespace_around_a_value_is_not_a_reading_error(rules):
    """Layout is not transcription: extra spacing must not cost an exact match."""
    (comparison,) = _compare("شماره ثبت: ۱۲۳۴۵", "شماره ثبت:  ۱۲۳۴۵ ", rules)
    assert comparison.exact is True


def test_a_dropped_field_is_missing_not_wrong(rules):
    (comparison,) = _compare("کد ملی ۰۴۹۹۳۷۰۸۹۹", "متن بدون شناسه", rules)
    assert comparison.is_missing is True
    assert comparison.is_spurious is False


def test_an_invented_field_is_spurious(rules):
    """On a sensitive field this is the most damaging error: it looks like data."""
    (comparison,) = _compare("متن بدون شناسه", "کد ملی ۰۴۹۹۳۷۰۸۹۹", rules)
    assert comparison.is_spurious is True
    assert comparison.is_missing is False


def test_pairing_is_positional_within_a_kind(rules):
    """So a transposition is visible rather than silently matched up by value."""
    reference = "تاریخ: ۱۴۰۳/۰۵/۱۲ و تاریخ: ۱۴۰۳/۰۶/۰۱"
    hypothesis = "تاریخ: ۱۴۰۳/۰۶/۰۱ و تاریخ: ۱۴۰۳/۰۵/۱۲"
    comparisons = _compare(reference, hypothesis, rules)
    assert len(comparisons) == 2
    assert all(item.exact is False for item in comparisons)


def test_every_field_on_both_sides_is_accounted_for_exactly_once(rules):
    reference = "شماره ثبت: ۱۲۳۴۵\nتاریخ: ۱۴۰۳/۰۵/۱۲"
    hypothesis = "شماره ثبت: ۱۲۳۴۵\nکد ملی ۰۴۹۹۳۷۰۸۹۹"
    comparisons = _compare(reference, hypothesis, rules)
    assert len(comparisons) == 3
    assert sum(item.is_missing for item in comparisons) == 1
    assert sum(item.is_spurious for item in comparisons) == 1


# --------------------------------------------------------------------------
# The audit report
# --------------------------------------------------------------------------


def _confidence(probability):
    """Build a stand-in confidence at a fixed probability.

    Args:
        probability: Per-token probability to report.

    Returns:
        A resolver suitable for :meth:`FieldAuditor.update`.
    """
    value = FieldConfidence(
        tokens=3,
        mean_logprob=math.log(probability),
        min_logprob=math.log(probability),
        max_entropy=0.5,
        min_margin=1.0,
    )
    return lambda start, end: value


def test_rates_are_taken_over_reference_fields(rules):
    """Dropping a field must not be able to improve the accuracy score.

    With ``found`` as the denominator, a model that emits only the fields it is
    sure of would score higher than one that attempts them all.
    """
    auditor = FieldAuditor(config=rules)
    auditor.update("a", "شماره ثبت: ۱۲۳۴۵", "شماره ثبت: ۱۲۳۴۵")
    auditor.update("b", "شماره ثبت: ۵۴۳۲۱", "بدون هیچ فیلدی")
    stats = auditor.result().stats_for(FieldKind.REGISTRATION_NUMBER)
    assert stats.reference_count == 2
    assert stats.missing == 1
    assert stats.exact == 1
    assert stats.exact_rate == pytest.approx(0.5)
    assert stats.recall == pytest.approx(0.5)


def test_every_configured_field_gets_a_row_even_when_unseen(rules):
    """A zero row and a missing row read very differently.

    The first says the split contained no such field; the second looks like the
    audit forgot to check.
    """
    auditor = FieldAuditor(config=rules)
    auditor.update("a", "شماره ثبت: ۱۲۳۴۵", "شماره ثبت: ۱۲۳۴۵")
    audit = auditor.result()
    assert {entry.kind for entry in audit.stats} == set(FieldKind)
    amount = audit.stats_for(FieldKind.AMOUNT)
    assert amount.reference_count == 0
    assert amount.recall == 0.0


def test_a_script_error_gets_its_own_fault_name(rules):
    """Fixed by numeral normalisation, not by the vision tower - so counted apart."""
    auditor = FieldAuditor(config=rules)
    findings = auditor.update("a", "تاریخ: ۱۴۰۳/۰۵/۱۲", "تاریخ: 1403/05/12")
    assert [item.fault for item in findings] == [FindingKind.SCRIPT_ONLY]
    stats = auditor.result().stats_for(FieldKind.DATE)
    assert stats.exact == 0
    assert stats.digits_equal == 1


def test_a_malformed_field_is_not_also_reported_as_a_mismatch(rules):
    """It cannot equal the reference, so two findings would double the count.

    The malformed finding carries both sides, so nothing is lost by suppressing
    the redundant one.
    """
    auditor = FieldAuditor(config=rules)
    findings = auditor.update("a", "شماره ثبت: ۱۲۳۴۵", "شماره ثبت: ابجد")
    assert [item.fault for item in findings] == [FindingKind.MALFORMED]
    assert findings[0].reference == "۱۲۳۴۵"
    assert "ابجد" in findings[0].detail
    stats = auditor.result().stats_for(FieldKind.REGISTRATION_NUMBER)
    assert stats.malformed == 1
    assert stats.malformed_rate == pytest.approx(1.0)
    assert stats.exact == 0


def test_a_calendar_fault_and_a_mismatch_are_both_reported(rules):
    """Unlike the malformed case, these carry different information.

    One is decidable with no reference and survives into production; the other
    needs the reference. Merging them would lose the first.
    """
    auditor = FieldAuditor(config=rules)
    findings = auditor.update("a", "تاریخ: ۱۴۰۲/۱۲/۲۹", "تاریخ: ۱۴۰۲/۱۲/۳۰")
    assert {item.fault for item in findings} == {
        FindingKind.CHECK_FAILED,
        FindingKind.MISMATCH,
    }
    assert auditor.result().stats_for(FieldKind.DATE).check_failed == 1


def test_a_low_confidence_field_is_flagged_even_when_it_is_correct(rules):
    """The whole point of the uncertainty column.

    A correct reading the model was unsure about is exactly what a human should
    re-check, and no reference-based metric can surface it.
    """
    auditor = FieldAuditor(config=rules, confidence_floor=0.5)
    findings = auditor.update(
        "a", "شماره ثبت: ۱۲۳۴۵", "شماره ثبت: ۱۲۳۴۵", confidence=_confidence(0.2)
    )
    assert [item.fault for item in findings] == [FindingKind.LOW_CONFIDENCE]
    stats = auditor.result().stats_for(FieldKind.REGISTRATION_NUMBER)
    assert stats.exact == 1
    assert stats.low_confidence == 1
    assert stats.mean_confidence == pytest.approx(0.2)


def test_a_confident_field_is_not_flagged(rules):
    auditor = FieldAuditor(config=rules, confidence_floor=0.5)
    findings = auditor.update(
        "a", "شماره ثبت: ۱۲۳۴۵", "شماره ثبت: ۱۲۳۴۵", confidence=_confidence(0.95)
    )
    assert findings == ()
    audit = auditor.result()
    assert audit.confidence_available is True
    assert audit.stats_for(FieldKind.REGISTRATION_NUMBER).low_confidence == 0


def test_confidence_is_reported_unavailable_when_none_was_recorded(rules):
    """Under beam search the numbers would describe tokens never emitted.

    Reporting them as absent is the only honest option; a plausible-looking number
    would be worse than none.
    """
    auditor = FieldAuditor(config=rules)
    auditor.update("a", "شماره ثبت: ۱۲۳۴۵", "شماره ثبت: ۱۲۳۴۵")
    audit = auditor.result()
    assert audit.confidence_available is False
    assert "not recorded" in audit.describe()
    assert "confidence" not in audit.to_dict()["fields"]["registration_number"]


def test_findings_beyond_the_cap_are_counted_not_dropped(rules):
    """A truncated list must never read as a clean run."""
    auditor = FieldAuditor(config=rules, max_findings=2)
    for index in range(5):
        auditor.update(f"r{index}", "شماره ثبت: ۱۲۳۴۵", "شماره ثبت: ۵۴۳۲۱")
    audit = auditor.result()
    assert len(audit.findings) == 2
    assert audit.findings_dropped == 3
    assert dict(audit.fault_counts)[FindingKind.MISMATCH] == 5


def test_the_report_embeds_the_rules_that_produced_it(rules):
    """A field-validity rate is uninterpretable without its pattern."""
    auditor = FieldAuditor(config=rules)
    auditor.update("a", "شماره ثبت: ۱۲۳۴۵", "شماره ثبت: ۱۲۳۴۵")
    payload = auditor.result().to_dict()
    assert payload["rules"]["fields"]["registration_number"]["pattern"]
    assert payload["records"] == 1


def test_the_report_is_json_serialisable(rules):
    import json

    auditor = FieldAuditor(config=rules, confidence_floor=0.5)
    auditor.update(
        "a", "شماره ثبت: ۱۲۳۴۵", "شماره ثبت: ۹۹۹۹۹", confidence=_confidence(0.3)
    )
    text = json.dumps(auditor.result().to_dict(), ensure_ascii=False)
    assert "registration_number" in text


def test_summarize_fields_renders_a_per_record_row(rules):
    rows = summarize_fields(extract_fields("تاریخ: ۱۴۰۲/۱۲/۳۰ و شماره ثبت: ابجد", rules))
    by_field = {row["field"]: row for row in rows}
    assert by_field["registration_number"]["format_ok"] is False
    assert by_field["registration_number"]["excerpt"] == ": ابجد"
    assert by_field["date"]["check_ok"] is False
    assert by_field["date"]["value"] == "۱۴۰۲/۱۲/۳۰"


# --------------------------------------------------------------------------
# Character-span to token mapping
# --------------------------------------------------------------------------


class _FakeTokenizer:
    """Decodes one character per token, so spans are predictable."""

    def __init__(self, text):
        """Store the string this tokenizer pretends to have produced.

        Args:
            text: The full decoded output.
        """
        self.text = text

    def decode(self, token_ids, **kwargs):
        """Return the prefix of the text as long as the id list.

        Args:
            token_ids: Ids to decode; only the count matters.
            **kwargs: Ignored.

        Returns:
            The corresponding prefix.
        """
        return self.text[: len(list(token_ids))]


def _lookup(text):
    """Build a lazy resolver over a one-character-per-token decoding of ``text``.

    Args:
        text: The string the fake tokenizer pretends to have generated.

    Returns:
        A resolver whose every token carries probability 0.9.
    """
    steps = [
        TokenConfidence(token_id=index, logprob=math.log(0.9), entropy=0.1, margin=1.0)
        for index in range(len(text))
    ]
    return LazySpanConfidence(
        tokenizer=_FakeTokenizer(text), token_ids=list(range(len(text))), steps=steps
    )


def test_lazy_span_confidence_does_not_decode_until_asked():
    """An empty or garbage prediction must cost nothing to audit.

    ``token_spans`` is one detokenisation per generated token, so building the map
    for a prediction with no fields in it would be pure waste.
    """
    text = "شماره ثبت: ۱۲۳۴۵"
    lookup = _lookup(text)
    assert lookup._spans is None
    resolved = lookup(text.index("۱"), len(text))
    assert lookup._spans is not None
    assert resolved is not None
    assert resolved.tokens == 5
    assert resolved.confidence == pytest.approx(0.9)
    assert resolved.weakest_probability == pytest.approx(0.9)


def test_an_empty_span_has_no_confidence():
    """A malformed field has a zero-width value; there is no reading to score."""
    assert _lookup("abc")(2, 2) is None


def test_no_recorded_steps_yields_no_confidence():
    lookup = LazySpanConfidence(
        tokenizer=_FakeTokenizer("abc"), token_ids=[0, 1, 2], steps=[]
    )
    assert lookup(0, 3) is None
