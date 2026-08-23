"""Evaluation metrics for Persian OCR.

Character error rate on its own is a poor guide for this adapter. A page of an
administrative letter is ~700 characters of prose and perhaps 20 characters of
registration number and date - so a model that reads every word of the body and
mangles every digit of the شماره ثبت scores a CER around 3% and is useless for
the field CLAUDE.md 5.2 calls the highest priority. Aggregate error rate has no
way to say that.

So this module reports three things:

*Text accuracy*: CER and WER over the whole transcription, for comparability with
published numbers and with the base model.

*Digit accuracy*: the same two rates computed over digits alone, plus exact-match
on the digit sequence. Persian text mixes three numeral systems - Persian ۰-۹,
Arabic-Indic ٠-٩ and Latin 0-9 - and the base model confuses them with each
other as readily as it confuses ۵ with ۶. Digit-only CER isolates that.

*Confusions*: the most frequent character substitutions, as ranked pairs. This is
the diagnostic that tells you what to fix; a CER of 8% does not, but "۴ read as
۳ in 210 of 300 samples" does.

Normalisation is deliberately minimal - Unicode NFC and whitespace collapse, and
nothing else. Folding Arabic ي to Persian ی or ك to ک would hide exactly the
errors this adapter is meant to remove.

Two different alignments are used on purpose. Error *counts* come from
:mod:`rapidfuzz`, which computes a true minimal edit distance; the *confusion
pairs* come from :class:`difflib.SequenceMatcher`, whose opcodes name which
character replaced which. They are not interchangeable: difflib maximises the
longest matching block rather than minimising the edit script, so a single
substitution at the head of a long run comes back as an insert plus a delete -
100 identical characters with the first one wrong scored 2 errors instead of 1.
That overstates CER by half on exactly the kind of near-miss this adapter is meant
to produce, so the counting job went to the library that gets it right.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Final

from rapidfuzz.distance import Levenshtein

__all__ = [
    "DIGITS",
    "EvalMetrics",
    "MetricAccumulator",
    "character_error_rate",
    "digits_only",
    "normalize_text",
    "word_error_rate",
]

#: Every digit character the corpora use: Persian, Arabic-Indic and Latin. Kept
#: as one string rather than three ranges so ``digits_only`` is a single pass.
DIGITS: Final[str] = "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩0123456789"

_DIGIT_SET: Final[frozenset[str]] = frozenset(DIGITS)


def normalize_text(text: str) -> str:
    """Normalise a transcription for comparison.

    NFC composition and whitespace collapse only. Persian text has several
    codepoints that render identically - the zero-width non-joiner, Arabic versus
    Persian yeh and kaf - and folding those would make the metric blind to a
    class of error the adapter is specifically meant to fix.

    Args:
        text: Raw reference or hypothesis.

    Returns:
        The normalised string, with runs of whitespace collapsed to one space and
        the ends stripped.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def digits_only(text: str) -> str:
    """Extract the digit characters of a string, in order.

    Args:
        text: Any string.

    Returns:
        The digits, concatenated, preserving numeral system and order.
    """
    return "".join(character for character in text if character in _DIGIT_SET)


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Minimal edit distance between two token sequences.

    Delegates to :mod:`rapidfuzz`, which is jiwer's own engine: a C++ Levenshtein
    over the token lists. The pure-Python alternative is a DP table, and at ~3,000
    characters per page over a 300-record split that is ~2.7 billion cell updates -
    slower than the generation being scored.

    Args:
        reference: Ground-truth tokens.
        hypothesis: Predicted tokens.

    Returns:
        The minimal number of substitutions, deletions and insertions.
    """
    return int(Levenshtein.distance(reference, hypothesis))


def character_error_rate(reference: str, hypothesis: str) -> tuple[int, int]:
    """Character-level errors and reference length.

    Returns the pair rather than the ratio so a corpus rate can be computed as
    total errors over total length. Averaging per-sample rates instead would let
    a three-character crop weigh as much as a 700-character page.

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted text.

    Returns:
        ``(errors, reference_length)``.
    """
    ref, hyp = list(reference), list(hypothesis)
    return _edit_distance(ref, hyp), len(ref)


def word_error_rate(reference: str, hypothesis: str) -> tuple[int, int]:
    """Word-level errors and reference length.

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted text.

    Returns:
        ``(errors, reference_word_count)``.
    """
    ref, hyp = reference.split(), hypothesis.split()
    return _edit_distance(ref, hyp), len(ref)


def _confusions(reference: str, hypothesis: str) -> Counter[tuple[str, str]]:
    """Count character substitutions between two strings.

    Only aligned one-for-one replacements are counted. A ``replace`` opcode
    spanning unequal lengths is a reflow rather than a substitution, and reading a
    pair out of it would invent confusions that did not happen.

    difflib rather than rapidfuzz here because the question is different: the
    counts need a minimal edit script, this needs to know *which* character stood
    where, and difflib's opcodes carry that alignment directly. Since only
    equal-length spans are read, the non-minimality that disqualified it for
    counting cannot add a pair - it can only decline to report one.

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted text.

    Returns:
        A counter over ``(expected, predicted)`` character pairs.
    """
    counts: Counter[tuple[str, str]] = Counter()
    matcher = SequenceMatcher(a=list(reference), b=list(hypothesis), autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            continue
        for offset in range(i2 - i1):
            counts[(reference[i1 + offset], hypothesis[j1 + offset])] += 1
    return counts


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """Scored results over one evaluation pass.

    Attributes:
        samples: Records scored.
        cer: Character error rate over all text.
        wer: Word error rate over all text.
        digit_cer: Character error rate over digits alone.
        digit_exact: Fraction of records whose digit sequence is exactly right,
            counted only over records whose reference contains a digit.
        digit_samples: Records whose reference contains at least one digit.
        empty_predictions: Records the model answered with nothing. Tracked
            separately because they flatter CER when the reference is short and
            would otherwise hide a model that has learned to stay silent.
        confusions: Most frequent ``(expected, predicted)`` character pairs.
    """

    samples: int
    cer: float
    wer: float
    digit_cer: float
    digit_exact: float
    digit_samples: int
    empty_predictions: int
    confusions: tuple[tuple[tuple[str, str], int], ...] = ()

    def to_dict(self) -> dict[str, float | int | list[list[object]]]:
        """Flatten to a JSON-serialisable mapping.

        Returns:
            A mapping suitable for ``json.dump`` and for Trainer's metric log.
        """
        return {
            "samples": self.samples,
            "cer": round(self.cer, 5),
            "wer": round(self.wer, 5),
            "digit_cer": round(self.digit_cer, 5),
            "digit_exact": round(self.digit_exact, 5),
            "digit_samples": self.digit_samples,
            "empty_predictions": self.empty_predictions,
            "confusions": [[list(pair), count] for pair, count in self.confusions],
        }

    def describe(self) -> str:
        """Multi-line human-readable report.

        Returns:
            The metrics formatted for a terminal, confusions included.
        """
        lines = [
            f"samples            {self.samples}",
            f"CER                {self.cer:.4f}",
            f"WER                {self.wer:.4f}",
            f"digit CER          {self.digit_cer:.4f}  ({self.digit_samples} records)",
            f"digit exact match  {self.digit_exact:.4f}",
        ]
        if self.empty_predictions:
            lines.append(f"empty predictions  {self.empty_predictions}")
        if self.confusions:
            lines.append("top confusions (expected -> predicted):")
            lines.extend(
                f"  {expected!r} -> {predicted!r}   {count}"
                for (expected, predicted), count in self.confusions
            )
        return "\n".join(lines)


@dataclass
class MetricAccumulator:
    """Accumulates metric counts one record at a time.

    Streaming rather than batch, because the alternative is holding 300 pairs of
    full-page transcriptions in memory during evaluation on a machine that is
    already short of it.

    Attributes:
        confusion_top_k: How many confusion pairs :meth:`result` reports.
    """

    confusion_top_k: int = 20
    _samples: int = 0
    _char_errors: int = 0
    _char_total: int = 0
    _word_errors: int = 0
    _word_total: int = 0
    _digit_errors: int = 0
    _digit_total: int = 0
    _digit_samples: int = 0
    _digit_exact: int = 0
    _empty: int = 0
    _confusions: Counter[tuple[str, str]] = field(default_factory=Counter)

    def update(self, reference: str, hypothesis: str) -> None:
        """Score one prediction.

        Args:
            reference: Ground-truth transcription.
            hypothesis: Model output.
        """
        ref = normalize_text(reference)
        hyp = normalize_text(hypothesis)
        self._samples += 1
        if not hyp:
            self._empty += 1

        errors, total = character_error_rate(ref, hyp)
        self._char_errors += errors
        self._char_total += total
        errors, total = word_error_rate(ref, hyp)
        self._word_errors += errors
        self._word_total += total

        ref_digits, hyp_digits = digits_only(ref), digits_only(hyp)
        if ref_digits:
            self._digit_samples += 1
            self._digit_exact += int(ref_digits == hyp_digits)
            errors, total = character_error_rate(ref_digits, hyp_digits)
            self._digit_errors += errors
            self._digit_total += total

        self._confusions.update(_confusions(ref, hyp))

    def result(self) -> EvalMetrics:
        """Finalise the accumulated counts.

        Returns:
            The metrics. Rates over an empty reference are reported as ``0.0``
            rather than ``nan``, so an eval on a split with no digits still
            produces a loggable number.
        """
        return EvalMetrics(
            samples=self._samples,
            cer=self._char_errors / self._char_total if self._char_total else 0.0,
            wer=self._word_errors / self._word_total if self._word_total else 0.0,
            digit_cer=(
                self._digit_errors / self._digit_total if self._digit_total else 0.0
            ),
            digit_exact=(
                self._digit_exact / self._digit_samples if self._digit_samples else 0.0
            ),
            digit_samples=self._digit_samples,
            empty_predictions=self._empty,
            confusions=tuple(self._confusions.most_common(self.confusion_top_k)),
        )
