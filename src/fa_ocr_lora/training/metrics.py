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

*Two CERs, not one.* The headline ``cer``/``wer`` normalise as little as
possible - Unicode NFC and whitespace collapse, nothing else - because folding
Arabic ي to Persian ی would hide an error class this adapter is meant to remove.
But a strict rate alone turned out to be unreadable: on the first 1.6 baseline,
207 of the top 20 confusions were ي/ی, ك/ک and ZWNJ-versus-space, so a CER of
0.4179 was mostly reporting which Unicode convention the corpus was typed in
rather than whether the model could read the page.

So :func:`fold_persian` defines the orthographic equivalences a *renderer* cannot
distinguish - the glyphs are identical on the page, and no amount of looking at
the image tells the model which codepoint the typist used - and ``cer_normalized``
/ ``wer_normalized`` report the rate with those folded away. Read together they
decompose the error: ``cer`` is what the adapter must fix, ``cer_normalized`` is
what the model genuinely misread, and the gap between them is convention. Digit
systems are deliberately *not* folded - ۵ against 5 is a real error here, and
§"Digit accuracy" above exists precisely to count it.

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
    "PERSIAN_FOLD",
    "EvalMetrics",
    "MetricAccumulator",
    "character_error_rate",
    "digit_runs",
    "digits_only",
    "fold_persian",
    "normalize_text",
    "word_error_rate",
]

#: Every digit character the corpora use: Persian, Arabic-Indic and Latin. Kept
#: as one string rather than three ranges so ``digits_only`` is a single pass.
DIGITS: Final[str] = "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩0123456789"

_DIGIT_SET: Final[frozenset[str]] = frozenset(DIGITS)

#: Orthographic equivalences that are invisible in a rendered page, mapped to the
#: Persian-convention codepoint. Every entry here is a pair of codepoints whose
#: glyphs are identical in print, so a model looking at the image has no evidence
#: to choose between them and a strict CER is scoring the typist, not the OCR.
#:
#: What is deliberately absent matters as much as what is present:
#:
#: * **Digits** (۵ / ٥ / 5) render differently and are a stated target - see the
#:   module docstring - so they are never folded.
#: * **Hamza carriers** (أ / ا, ئ / ی) are visibly different marks; dropping the
#:   hamza is a real misread.
#: * **Punctuation** (؛ / :، ، / .) is visibly different too. The 1.6 baseline's
#:   ``'؛' -> ':'`` is a genuine error and stays counted.
PERSIAN_FOLD: Final[dict[str, str]] = {
    "\u064a": "\u06cc",  # ARABIC YEH -> FARSI YEH
    "\u0649": "\u06cc",  # ARABIC ALEF MAKSURA -> FARSI YEH
    "\u06d2": "\u06cc",  # ARABIC YEH BARREE -> FARSI YEH
    "\u0643": "\u06a9",  # ARABIC KAF -> ARABIC KEHEH
    "\u06aa": "\u06a9",  # ARABIC SWASH KAF -> ARABIC KEHEH
    "\u0647\u200d": "\u0647",  # HEH + ZWJ, a rendering-only ligature form
    "\u200c": " ",  # ZERO WIDTH NON-JOINER: a space the renderer does not draw
    "\u200b": " ",  # ZERO WIDTH SPACE
    "\u200d": "",  # ZERO WIDTH JOINER
    "\ufeff": "",  # BYTE ORDER MARK, when a corpus file leaks one into the text
    "\u0640": "",  # ARABIC TATWEEL: pure typographic stretch, carries no content
}

#: Built once; ``str.translate`` wants ordinals, and the multi-character key in
#: :data:`PERSIAN_FOLD` cannot go through a translation table at all.
_FOLD_TABLE: Final[dict[int, str]] = {
    ord(source): target
    for source, target in PERSIAN_FOLD.items()
    if len(source) == 1
}
_FOLD_SEQUENCES: Final[tuple[tuple[str, str], ...]] = tuple(
    (source, target) for source, target in PERSIAN_FOLD.items() if len(source) > 1
)


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


def fold_persian(text: str) -> str:
    """Collapse orthographic variants that render identically.

    Applies :data:`PERSIAN_FOLD`, then re-collapses whitespace, because folding a
    zero-width non-joiner to a space can produce a double space where the
    reference had one. The result is only ever compared against another folded
    string, so the mapping needs to be consistent rather than canonical.

    This is what separates ``cer`` from ``cer_normalized``: the strict rate is
    computed on the unfolded text and counts a ي typed where the corpus uses ی,
    the normalised rate is computed on the output of this function and does not.

    Args:
        text: Text already through :func:`normalize_text`.

    Returns:
        The text with Persian/Arabic orthographic variants unified.
    """
    folded = text.translate(_FOLD_TABLE)
    for source, target in _FOLD_SEQUENCES:
        folded = folded.replace(source, target)
    return " ".join(folded.split())


def digits_only(text: str) -> str:
    """Extract the digit characters of a string, in order.

    Args:
        text: Any string.

    Returns:
        The digits, concatenated, preserving numeral system and order.
    """
    return "".join(character for character in text if character in _DIGIT_SET)


def digit_runs(text: str) -> list[str]:
    """Split out the maximal runs of digits, in order.

    The companion to :func:`digits_only`, and the reason both exist: concatenating
    every digit on a page into one string makes ``digit_cer`` unbounded, because
    the rate divides by the *reference* length and a model that hallucinates
    numbers can emit three times as many digits as the page holds. The 1.6
    baseline's ``digit CER 1.9678`` is that, not a 197% character error rate.

    Runs keep the structure the concatenation throws away: a page has twelve
    numbers on it, and the useful question - the one CLAUDE.md §5.2 asks about
    the شماره ثبت - is how many of those twelve came back exactly right.

    Args:
        text: Any string.

    Returns:
        Each maximal run of digit characters, in reading order.
    """
    runs: list[str] = []
    current: list[str] = []
    for character in text:
        if character in _DIGIT_SET:
            current.append(character)
            continue
        if current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    return runs


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


def _aligned_matches(reference: list[str], hypothesis: list[str]) -> int:
    """Count reference entries the hypothesis reproduces exactly, in order.

    Used for numeric runs. The alignment matters: a page whose first number the
    model missed still read the other eleven, and zipping the two lists
    position-by-position would score all twelve wrong. ``SequenceMatcher``'s
    matching blocks give the in-order common subsequence instead, so a dropped or
    an inserted number costs exactly itself.

    Non-minimality is harmless here for the same reason it is in
    :func:`_confusions`: only ``equal`` spans are counted, so an imperfect
    alignment can undercount a match but can never invent one.

    Args:
        reference: Ground-truth entries.
        hypothesis: Predicted entries.

    Returns:
        How many reference entries are matched, at most ``len(reference)``.
    """
    if not reference or not hypothesis:
        return 0
    matcher = SequenceMatcher(a=reference, b=hypothesis, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks())


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """Scored results over one evaluation pass.

    Attributes:
        samples: Records scored.
        cer: Character error rate over all text, strictly normalised.
        wer: Word error rate over all text, strictly normalised.
        digit_cer: Character error rate over the page's digits concatenated.
            May exceed 1.0 - the denominator is the reference's digit count, so
            a model that emits more digits than the page holds scores above
            100%. ``digit_ref_chars`` and ``digit_hyp_chars`` say whether that
            is what happened.
        digit_exact: Fraction of records whose digit sequence is exactly right,
            counted only over records whose reference contains a digit.
        digit_samples: Records whose reference contains at least one digit.
        empty_predictions: Records the model answered with nothing. Tracked
            separately because they flatter CER when the reference is short and
            would otherwise hide a model that has learned to stay silent.
        confusions: Most frequent ``(expected, predicted)`` character pairs.
        cer_normalized: Character error rate after :func:`fold_persian`. The gap
            to ``cer`` is the share of the error that is orthographic convention
            rather than misreading.
        wer_normalized: Word error rate after :func:`fold_persian`.
        digit_runs: Numeric runs across all references - "how many numbers were
            on these pages".
        digit_runs_matched: Those runs an aligned hypothesis run reproduced
            exactly.
        digit_ref_chars: Digit characters in the references.
        digit_hyp_chars: Digit characters in the hypotheses. Larger than
            ``digit_ref_chars`` means the model is inventing numbers, which is
            what pushes ``digit_cer`` over 1.0.
    """

    samples: int
    cer: float
    wer: float
    digit_cer: float
    digit_exact: float
    digit_samples: int
    empty_predictions: int
    confusions: tuple[tuple[tuple[str, str], int], ...] = ()
    cer_normalized: float = 0.0
    wer_normalized: float = 0.0
    digit_runs: int = 0
    digit_runs_matched: int = 0
    digit_ref_chars: int = 0
    digit_hyp_chars: int = 0

    @property
    def digit_run_accuracy(self) -> float:
        """Fraction of reference numeric runs reproduced exactly.

        Bounded in ``[0, 1]`` by construction, which ``digit_cer`` is not, so
        this is the digit number to read when the two disagree.
        """
        return self.digit_runs_matched / self.digit_runs if self.digit_runs else 0.0

    @property
    def convention_share(self) -> float:
        """Share of the strict character error that folding away variants removes.

        ``0.0`` when the two rates agree, approaching ``1.0`` when essentially
        all of the strict error was ي-versus-ی and friends. Undefined with no
        error at all, reported as ``0.0``.
        """
        return (self.cer - self.cer_normalized) / self.cer if self.cer else 0.0

    def to_dict(self) -> dict[str, float | int | list[list[object]]]:
        """Flatten to a JSON-serialisable mapping.

        Returns:
            A mapping suitable for ``json.dump`` and for Trainer's metric log.
        """
        return {
            "samples": self.samples,
            "cer": round(self.cer, 5),
            "wer": round(self.wer, 5),
            "cer_normalized": round(self.cer_normalized, 5),
            "wer_normalized": round(self.wer_normalized, 5),
            "convention_share": round(self.convention_share, 5),
            "digit_cer": round(self.digit_cer, 5),
            "digit_exact": round(self.digit_exact, 5),
            "digit_samples": self.digit_samples,
            "digit_runs": self.digit_runs,
            "digit_runs_matched": self.digit_runs_matched,
            "digit_run_accuracy": round(self.digit_run_accuracy, 5),
            "digit_ref_chars": self.digit_ref_chars,
            "digit_hyp_chars": self.digit_hyp_chars,
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
            f"CER                {self.cer:.4f}  (folded {self.cer_normalized:.4f},"
            f" {self.convention_share:.0%} convention)",
            f"WER                {self.wer:.4f}  (folded {self.wer_normalized:.4f})",
            f"digit CER          {self.digit_cer:.4f}  ({self.digit_samples} records)",
            f"digit exact match  {self.digit_exact:.4f}",
            f"digit runs         {self.digit_runs_matched}/{self.digit_runs} exact"
            f"  ({self.digit_run_accuracy:.4f})",
        ]
        if self.digit_hyp_chars > self.digit_ref_chars:
            # The explanation for a digit CER above 1.0, printed only when that
            # is actually what is happening rather than on every run.
            lines.append(
                f"digit chars        {self.digit_ref_chars} expected,"
                f" {self.digit_hyp_chars} produced"
                f" ({self.digit_hyp_chars / max(self.digit_ref_chars, 1):.1f}x)"
            )
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
    _folded_char_errors: int = 0
    _folded_char_total: int = 0
    _folded_word_errors: int = 0
    _folded_word_total: int = 0
    _digit_errors: int = 0
    _digit_total: int = 0
    _digit_samples: int = 0
    _digit_exact: int = 0
    _digit_runs: int = 0
    _digit_runs_matched: int = 0
    _digit_hyp_chars: int = 0
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

        # Scored over the same record, not a second pass over the split, so the
        # strict and folded rates always describe an identical set of samples.
        folded_ref, folded_hyp = fold_persian(ref), fold_persian(hyp)
        errors, total = character_error_rate(folded_ref, folded_hyp)
        self._folded_char_errors += errors
        self._folded_char_total += total
        errors, total = word_error_rate(folded_ref, folded_hyp)
        self._folded_word_errors += errors
        self._folded_word_total += total

        ref_digits, hyp_digits = digits_only(ref), digits_only(hyp)
        self._digit_hyp_chars += len(hyp_digits)
        if ref_digits:
            self._digit_samples += 1
            self._digit_exact += int(ref_digits == hyp_digits)
            errors, total = character_error_rate(ref_digits, hyp_digits)
            self._digit_errors += errors
            self._digit_total += total

        # Run-level agreement, aligned rather than zipped: a page that drops its
        # first number must not count every later one as wrong, which is exactly
        # what comparing position-by-position would do.
        reference_runs, hypothesis_runs = digit_runs(ref), digit_runs(hyp)
        self._digit_runs += len(reference_runs)
        self._digit_runs_matched += _aligned_matches(reference_runs, hypothesis_runs)

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
            cer_normalized=(
                self._folded_char_errors / self._folded_char_total
                if self._folded_char_total
                else 0.0
            ),
            wer_normalized=(
                self._folded_word_errors / self._folded_word_total
                if self._folded_word_total
                else 0.0
            ),
            digit_cer=(
                self._digit_errors / self._digit_total if self._digit_total else 0.0
            ),
            digit_exact=(
                self._digit_exact / self._digit_samples if self._digit_samples else 0.0
            ),
            digit_samples=self._digit_samples,
            digit_runs=self._digit_runs,
            digit_runs_matched=self._digit_runs_matched,
            digit_ref_chars=self._digit_total,
            digit_hyp_chars=self._digit_hyp_chars,
            empty_predictions=self._empty,
            confusions=tuple(self._confusions.most_common(self.confusion_top_k)),
        )

