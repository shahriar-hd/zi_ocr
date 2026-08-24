"""Locate sensitive fields in a transcription and judge what was read.

This is the layer that turns a wall of transcribed text into a per-field verdict.
It exists because CER cannot express the thing that matters most about an
administrative letter: a page is ~700 characters of prose and ~20 characters of
شماره ثبت and تاریخ, so a model that reads every word of the body and mangles
every digit of the metadata block scores a CER around 3% and is useless for the
field CLAUDE.md 5.2 calls the highest priority.

Three questions are asked of every field, and they are kept apart on purpose
because they answer to different audiences.

**Is it well formed?** Does a value matching the field's pattern actually follow
the label. Answerable with no ground truth, so it is the check that survives into
production - a ``شماره ثبت:`` followed by nothing numeric is a fault the moment it
is read, on a document nobody has ever transcribed.

**Is it internally consistent?** Only ``date`` has such a check here:
``۱۴۰۲/۱۲/۳۰`` is provably a misread because Esfand has 29 days in a common year.
Also reference-free. The national-id checksums that would fit this slot are
deliberately absent - the corpus generates those digits at random, so a checksum
would condemn ~91% of the ground truth and say nothing about the model.

**Does it agree with the reference?** The real correctness signal during
evaluation, and the only one that is unavailable in production. Reported twice,
because two different errors hide inside one number: ``exact`` compares the value
as written, ``digits`` compares it numeral-blind. A ``۵`` read as a Latin ``5``
fails the first and passes the second, and that distinction is the difference
between a script-normalisation bug and a misread digit.

Extraction order is the one subtle thing here. Every label of every field is
sorted by descending length before matching, globally rather than per field,
because ``شماره`` (letter_number) is a prefix of ``شماره ثبت``
(registration_number). Matched shortest-first, every registration number in the
corpus would be recorded as a letter number and the priority field would show a
perfect score it never earned.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from .config import FieldCheck, FieldKind, FieldSpec, FieldValidationConfig
from .iran import (
    DIGIT_CHARACTERS,
    all_digits_identical,
    digits_of,
    jalali_date_is_valid,
    parse_jalali_date,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ExtractedField",
    "FieldComparison",
    "compare_fields",
    "extract_fields",
]

#: Every digit character, as a set, for the "is this a value or a word?" test.
_DIGITS: Final[frozenset[str]] = frozenset(DIGIT_CHARACTERS)

#: Collapses whitespace before an exact comparison, so a value broken across a
#: line in the reference still equals the same value on one line in the
#: prediction. Line breaks are a layout artefact, not a reading error.
_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ExtractedField:
    """One sensitive field found in a transcription.

    Attributes:
        kind: Which field this is.
        label: The label wording that introduced it, or ``""`` for a field found
            by pattern alone.
        value: The value as it appeared, or ``""`` when the label was found but
            no well-formed value followed it. The empty case is retained rather
            than dropped: a label with no readable value is the malformation the
            audit is for, and dropping it would score the field as absent.
        start: Character offset of the value in the source text. Equal to
            :attr:`end` when no value was matched, in which case both point just
            past the label.
        end: Character offset one past the value.
        check_ok: Verdict of the field's semantic check, or ``None`` when the
            field has no such check or has no value to check.
        repeated_digits: Whether the value's digits are all the same digit - the
            signature of a decoder stuck in a repetition loop.
        excerpt: What stood where the value should have been, when none matched.
            Empty for a well-formed field. Carried so a malformed field is
            actionable in the report - "the model wrote ``ابجد`` after
            ``شماره ثبت``" is a finding, "field 3 was malformed" is not.
    """

    kind: FieldKind
    label: str
    value: str
    start: int
    end: int
    check_ok: bool | None = None
    repeated_digits: bool = False
    excerpt: str = ""

    @property
    def format_ok(self) -> bool:
        """Whether a value matching the field's pattern was found.

        Returns:
            ``False`` when a label was present with no well-formed value after
            it, which is the reference-free fault this audit exists to surface.
        """
        return bool(self.value)

    @property
    def digits(self) -> str:
        """The value's digits, normalised to Latin.

        Returns:
            The digit characters in order, for a numeral-blind comparison.
        """
        return digits_of(self.value)

    def describe(self) -> str:
        """One-line rendering for a log or a report.

        Returns:
            The field, its value and any fault, e.g.
            ``date '۱۴۰۲/۱۲/۳۰' [check failed]``.
        """
        faults = []
        if not self.format_ok:
            faults.append(f"malformed, found {self.excerpt!r}")
        if self.check_ok is False:
            faults.append("check failed")
        if self.repeated_digits:
            faults.append("repeated digits")
        suffix = f" [{'; '.join(faults)}]" if faults else ""
        return f"{self.kind.value} {self.value!r}{suffix}"


@dataclass(frozen=True, slots=True)
class FieldComparison:
    """One reference field set beside what the model read for it.

    Pairing is positional within a field kind: the first ``date`` in the
    reference is compared against the first ``date`` in the prediction. Reading
    order is preserved by both the corpus and the model, so position is a sound
    pairing key - and the alternative, matching by value similarity, would hide
    the case where the model transposed two dates.

    Attributes:
        kind: Which field this is.
        reference: The reference field, or ``None`` when the model produced a
            field the reference does not have.
        hypothesis: The predicted field, or ``None`` when the model missed a
            field the reference has.
    """

    kind: FieldKind
    reference: ExtractedField | None
    hypothesis: ExtractedField | None

    @property
    def is_missing(self) -> bool:
        """Whether the model failed to produce this reference field at all.

        Returns:
            ``True`` when the reference has the field and the prediction does
            not. Counted separately from a wrong value, because a silently
            dropped registration number and a misread one need different fixes.
        """
        return self.reference is not None and self.hypothesis is None

    @property
    def is_spurious(self) -> bool:
        """Whether the model invented a field the reference does not have.

        Returns:
            ``True`` when the prediction has the field and the reference does
            not - a hallucinated value, which on a sensitive field is the most
            damaging error of all because it looks like data.
        """
        return self.reference is None and self.hypothesis is not None

    @property
    def exact(self) -> bool:
        """Whether the predicted value equals the reference character for character.

        Returns:
            ``True`` only when both sides have a value and the two agree after
            whitespace collapse. Numeral family is significant here.
        """
        if self.reference is None or self.hypothesis is None:
            return False
        return _collapse(self.reference.value) == _collapse(self.hypothesis.value)

    @property
    def digits_equal(self) -> bool:
        """Whether the predicted value carries the same digits as the reference.

        The numeral-blind comparison: ``۱۲۳۴۵`` and ``12345`` agree. Passing this
        while failing :attr:`exact` isolates a script error from a misread digit,
        which are different bugs with different fixes.

        Returns:
            ``True`` when both sides have at least one digit and the digit
            sequences match.
        """
        if self.reference is None or self.hypothesis is None:
            return False
        reference_digits = self.reference.digits
        return bool(reference_digits) and reference_digits == self.hypothesis.digits


def _collapse(text: str) -> str:
    """Collapse whitespace runs to a single space and strip the ends.

    Args:
        text: Any string.

    Returns:
        The normalised string.
    """
    return _WHITESPACE.sub(" ", text).strip()


def _check_value(spec: FieldSpec, value: str) -> bool | None:
    """Apply a field's semantic check to a matched value.

    Args:
        spec: The field's spec, naming which check applies.
        value: The matched value.

    Returns:
        The verdict, or ``None`` when the field has no semantic check.
    """
    if spec.check is FieldCheck.NONE:
        return None
    if spec.check is FieldCheck.JALALI_DATE:
        parsed = parse_jalali_date(value)
        if parsed is None:
            return False
        return jalali_date_is_valid(*parsed)
    return None


def _build_field(
    spec: FieldSpec, label: str, value: str, start: int, excerpt: str = ""
) -> ExtractedField:
    """Assemble one extracted field and run its checks.

    Args:
        spec: The field's spec.
        label: The matched label, or ``""``.
        value: The matched value, or ``""`` for a label with no readable value.
        start: Character offset of the value.
        excerpt: What stood where the value should have been, when none matched.

    Returns:
        The field with its verdicts filled in.
    """
    return ExtractedField(
        kind=spec.kind,
        label=label,
        value=value,
        start=start,
        end=start + len(value),
        check_ok=_check_value(spec, value) if value else None,
        repeated_digits=all_digits_identical(value),
        excerpt="" if value else _collapse(excerpt),
    )


def _label_alternation(
    specs: Sequence[FieldSpec],
) -> tuple[re.Pattern[str], dict[str, FieldSpec]]:
    """Compile one regex matching any label of any labelled field.

    Labels are sorted by descending length across all fields, not within each
    one. Python's alternation returns the first branch that matches at a
    position, so longest-first is what makes ``شماره ثبت`` win over ``شماره`` -
    and without it every registration number in the corpus would be recorded as
    a letter number.

    Args:
        specs: Every field spec, labelled or not.

    Returns:
        The compiled alternation and a lookup from label text to its spec. Both
        are empty when no field carries a label.
    """
    owners: dict[str, FieldSpec] = {}
    for spec in specs:
        for label in spec.labels:
            owners.setdefault(label, spec)
    if not owners:
        return re.compile(r"(?!)"), owners
    ordered = sorted(owners, key=len, reverse=True)
    return re.compile("|".join(re.escape(label) for label in ordered)), owners


def _value_after_label(
    text: str, spec: FieldSpec, cursor: int, config: FieldValidationConfig
) -> tuple[str, int]:
    """Find the value that follows a label.

    Tries each configured separator, longest first, then skips whitespace and
    anchors the field's pattern at that point. Anchoring rather than searching is
    what keeps a 5-digit registration-number pattern from picking a fragment out
    of a phone number later in the paragraph.

    Args:
        text: The full transcription.
        spec: The field whose value is wanted.
        cursor: Offset just past the matched label.
        config: The loaded rules, supplying separators and the gap bound.

    Returns:
        ``(value, start)``, with ``value`` empty when nothing well formed follows
        the label. ``start`` then points at the first non-gap character, so the
        report can quote what was there instead.
    """
    window = text[cursor : cursor + config.max_gap_chars + 1]
    best_start = cursor + (len(window) - len(window.lstrip()))
    for separator in sorted(config.separators, key=len, reverse=True):
        offset = 0
        if separator:
            stripped = window.lstrip()
            lead = len(window) - len(stripped)
            if not stripped.startswith(separator):
                continue
            offset = lead + len(separator)
        remainder = window[offset:]
        start = cursor + offset + (len(remainder) - len(remainder.lstrip()))
        match = spec.pattern.match(text, start)
        if match and match.end() > match.start():
            return match.group(0), start
    return "", best_start


def _introduces_a_field(
    text: str, start: int, end: int, config: FieldValidationConfig
) -> bool:
    """Judge whether a label with no readable value was used as a field at all.

    Only consulted when the value pattern found nothing, which is the property
    that makes this safe: a label followed by a well-formed value is a field by
    demonstration and never reaches here, so no readable field can be suppressed.

    The question is real because every label in this file is also an ordinary
    Persian word. ``در تاریخ ۱۳۸۹ مبنی بر`` is a sentence, not a date field, and
    ``شماره تماس: ۰۲۱-۷۷۷۷۷۷۷۷`` is a phone number that begins with the
    letter-number label. Counted over the reference text of 10,000 manifest rows,
    those two shapes were 67% of every malformed finding the audit produced - on
    ground truth, where the model cannot be at fault.

    Two signals decide it, and both exploit the fact that all five audited fields
    have numeric values:

    * an explicit separator directly after the label - somebody wrote
      ``تاریخ:`` - which is a field introduction whatever follows it, so junk
      after it stays a finding;
    * failing that, the label opening its line *and* a digit next. This is the
      column-aligned layout the empty separator exists for (``تاریخ۲۲ مهر
      ۱۴۰۳``). Both halves are needed: prose runs mid-line, and a label followed
      by another word (``تاریخ پایان پروژه``, ``شماره ثبت فعلی شرکت``) is the
      label used as a word.

    Args:
        text: The full transcription.
        start: Offset of the matched label.
        end: Offset one past the matched label.
        config: The loaded rules, supplying the separator pool.

    Returns:
        ``True`` when the occurrence should be reported as a malformed field.
    """
    tail = text[end : end + config.max_gap_chars + 1]
    stripped = tail.lstrip()
    if any(sep and stripped.startswith(sep) for sep in config.separators):
        return True
    line_start = text.rfind("\n", 0, start)
    if text[line_start + 1 : start].strip():
        return False
    return bool(stripped) and stripped[0] in _DIGITS


def extract_fields(
    text: str, config: FieldValidationConfig
) -> tuple[ExtractedField, ...]:
    """Find every sensitive field in a transcription.

    Labelled fields are located by their label and the value that follows it;
    unlabelled ones - currently only ``amount`` - by their value pattern alone,
    which is sound because that pattern requires a thousands separator no other
    field uses.

    A label with no well-formed value after it is reported with an empty value
    rather than omitted, so a malformed field stays distinguishable from an
    absent one - unless ``require_field_context`` is set and the occurrence was
    the label used as an ordinary Persian word, for which see
    :func:`_introduces_a_field`.

    Args:
        text: A reference transcription or a model prediction.
        config: The loaded validation rules.

    Returns:
        The fields, ordered by where their value appears in the text.
    """
    alternation, owners = _label_alternation(config.specs)
    found: list[ExtractedField] = []
    consumed: list[tuple[int, int]] = []

    for match in alternation.finditer(text):
        spec = owners[match.group(0)]
        value, start = _value_after_label(text, spec, match.end(), config)
        if value:
            found.append(_build_field(spec, match.group(0), value, start))
            consumed.append((start, start + len(value)))
            continue
        if config.require_field_context and not _introduces_a_field(
            text, match.start(), match.end(), config
        ):
            LOGGER.debug(
                "not a %s field: %r used as a word in %r",
                spec.kind.value,
                match.group(0),
                _collapse(text[match.start() : match.end() + 24]),
            )
            continue
        excerpt = text[start : start + config.max_gap_chars]
        found.append(_build_field(spec, match.group(0), "", start, excerpt))

    for spec in config.specs:
        if spec.is_labelled:
            continue
        for match in spec.pattern.finditer(text):
            if any(
                match.start() < end and start < match.end() for start, end in consumed
            ):
                continue
            found.append(_build_field(spec, "", match.group(0), match.start()))

    found.sort(key=lambda field: (field.start, field.kind.value))
    return tuple(found)


def compare_fields(
    reference: Iterable[ExtractedField], hypothesis: Iterable[ExtractedField]
) -> tuple[FieldComparison, ...]:
    """Pair a reference's fields against a prediction's, per kind and in order.

    Args:
        reference: Fields extracted from the ground-truth transcription.
        hypothesis: Fields extracted from the model's output.

    Returns:
        One comparison per pair, plus one per unpaired field on either side, so
        every field on both sides is accounted for exactly once.
    """
    reference_by_kind: dict[FieldKind, list[ExtractedField]] = {}
    hypothesis_by_kind: dict[FieldKind, list[ExtractedField]] = {}
    for field in reference:
        reference_by_kind.setdefault(field.kind, []).append(field)
    for field in hypothesis:
        hypothesis_by_kind.setdefault(field.kind, []).append(field)

    comparisons: list[FieldComparison] = []
    for kind in FieldKind:
        left = reference_by_kind.get(kind, [])
        right = hypothesis_by_kind.get(kind, [])
        for index in range(max(len(left), len(right))):
            comparisons.append(
                FieldComparison(
                    kind=kind,
                    reference=left[index] if index < len(left) else None,
                    hypothesis=right[index] if index < len(right) else None,
                )
            )
    return tuple(comparisons)
