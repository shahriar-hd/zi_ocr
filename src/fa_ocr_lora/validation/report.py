"""Turn per-field verdicts into numbers, and faults into traceable findings.

This is the reporting half of the answer to "don't settle for overall CER/WER".
:mod:`fa_ocr_lora.validation.fields` decides what happened to one field on one
page; this module accumulates those decisions across an evaluation split and emits
two things that CER cannot produce.

**Per-field rates.** One row per sensitive field, so ``registration_number`` has
its own recall, its own exact-match and its own malformation rate rather than
being averaged into 700 characters of body prose. The rates are deliberately not
collapsed into a single score: recall (did the field survive at all), exact match
(character for character), digit match (the same number in a different numeral
family) and malformation (a label with nothing readable after it) are four
distinct failures with four distinct fixes, and a weighted average of them would
be a number nobody could act on.

**Findings.** Every fault, named, with the reference value, what the model wrote,
and how certain it was. This is the traceability the request asks for: a rate tells
you the model is 4% wrong on registration numbers, a finding tells you *which*
letter, which value, and whether the model itself was unsure - which is what turns
an aggregate into something reviewable by hand.

Findings are capped. An evaluation over 800 records could otherwise produce
thousands of them and bury the report; the cap keeps the JSON readable, and the
count of what it dropped is reported so the cap can never be mistaken for a clean
run.

One asymmetry is intentional. Extraction rates are computed against the reference,
because the reference defines what the page actually contained - so a field the
model invented is a *spurious* finding rather than a denominator. Malformation and
low confidence are computed over predicted fields, because those are the faults
that exist in production, where there is no reference at all.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from .confidence import FieldConfidence
from .config import FieldKind, FieldValidationConfig
from .fields import ExtractedField, FieldComparison, compare_fields, extract_fields

__all__ = [
    "DEFAULT_CONFIDENCE_FLOOR",
    "DEFAULT_MAX_FINDINGS",
    "FieldAudit",
    "FieldAuditor",
    "FieldFinding",
    "FieldStats",
    "FindingKind",
    "summarize_fields",
]

#: Below this probability a predicted field is reported as uncertain. Chosen as a
#: reviewable rate rather than a calibrated threshold: on a well-fitted decoder
#: most field tokens sit above 0.9, so 0.5 on the *weakest* token flags the cases
#: where the model was genuinely undecided without flooding the report. It is a
#: constructor argument precisely because the right value depends on the run.
DEFAULT_CONFIDENCE_FLOOR: Final[float] = 0.5

#: How many findings the report keeps. Enough to review by hand; the rest are
#: counted, never silently discarded.
DEFAULT_MAX_FINDINGS: Final[int] = 200


class FindingKind:
    """The named faults a finding can report.

    A plain namespace of string constants rather than an enum: these are written
    straight into JSON and read back by eye, and a ``StrEnum`` would add nothing
    but a conversion at every boundary.
    """

    MISSING: Final[str] = "missing"
    SPURIOUS: Final[str] = "spurious"
    MALFORMED: Final[str] = "malformed"
    CHECK_FAILED: Final[str] = "check_failed"
    REPEATED_DIGITS: Final[str] = "repeated_digits"
    MISMATCH: Final[str] = "mismatch"
    SCRIPT_ONLY: Final[str] = "script_only"
    LOW_CONFIDENCE: Final[str] = "low_confidence"


@dataclass(frozen=True, slots=True)
class FieldFinding:
    """One field-level fault, attributable to a record.

    Attributes:
        record_id: Which evaluated record this came from.
        kind: Which sensitive field.
        fault: One of the :class:`FindingKind` constants.
        reference: The reference value, or ``""`` for a spurious field.
        prediction: What the model wrote, or ``""`` for a missing field.
        detail: Free-text context - for a malformed field, the text that stood
            where the value should have been.
        confidence: The model's confidence in the predicted value, or ``None``
            when uncertainty was not recorded or there was no value to score.
    """

    record_id: str
    kind: FieldKind
    fault: str
    reference: str = ""
    prediction: str = ""
    detail: str = ""
    confidence: FieldConfidence | None = None

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a JSON-serialisable mapping.

        Returns:
            A mapping suitable for ``json.dump``; the confidence block is omitted
            when none was recorded.
        """
        payload: dict[str, Any] = {
            "record_id": self.record_id,
            "field": self.kind.value,
            "fault": self.fault,
            "reference": self.reference,
            "prediction": self.prediction,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.confidence is not None:
            payload["confidence"] = self.confidence.to_dict()
        return payload

    def describe(self) -> str:
        """One-line rendering for a terminal.

        Returns:
            The record, field, fault and whichever values exist, e.g.
            ``letter_0042  registration_number  mismatch  '۱۲۳۴۵' -> '۱۲۳۴۶'``. A
            one-sided fault prints one value rather than an arrow from nothing.
        """
        if self.reference and self.prediction:
            values = f"{self.reference!r} -> {self.prediction!r}"
        else:
            values = repr(self.reference or self.prediction)
        suffix = f"  ({self.detail})" if self.detail else ""
        certainty = f"  {self.confidence.describe()}" if self.confidence else ""
        return (
            f"{self.record_id}  {self.kind.value}  {self.fault}  {values}"
            f"{suffix}{certainty}"
        )


@dataclass(frozen=True, slots=True)
class FieldStats:
    """Accumulated rates for one sensitive field.

    Attributes:
        kind: Which field these describe.
        reference_count: Times the field appeared in a reference. The denominator
            for :attr:`recall`, :attr:`exact_rate` and :attr:`digit_rate`.
        found: Times the model produced a field paired against a reference one,
            whether or not its value was readable - see :attr:`recall`.
        missing: Reference fields the model did not produce.
        spurious: Predicted fields with no reference counterpart.
        exact: Paired fields whose values agree character for character.
        digits_equal: Paired fields whose digits agree, ignoring numeral family.
            Always at least :attr:`exact`; the gap between them is the number of
            values read correctly in the wrong script.
        predicted_count: Predicted fields, whether paired or not. The denominator
            for the reference-free rates below, which are the ones that still mean
            something in production.
        malformed: Predicted fields whose label had no well-formed value after it.
        check_failed: Predicted values that failed the field's semantic check -
            for ``date``, a day that never existed.
        repeated_digits: Predicted values whose digits are all the same digit.
        low_confidence: Predicted values whose weakest token fell below the floor.
        scored_confidence: Predicted values that had a confidence at all, the
            denominator for :attr:`mean_confidence`.
        confidence_total: Sum of per-field confidences, for the mean.
        weakest_total: Sum of per-field weakest-token probabilities.
    """

    kind: FieldKind
    reference_count: int = 0
    found: int = 0
    missing: int = 0
    spurious: int = 0
    exact: int = 0
    digits_equal: int = 0
    predicted_count: int = 0
    malformed: int = 0
    check_failed: int = 0
    repeated_digits: int = 0
    low_confidence: int = 0
    scored_confidence: int = 0
    confidence_total: float = 0.0
    weakest_total: float = 0.0

    @property
    def recall(self) -> float:
        """Fraction of reference fields the model located.

        Returns:
            ``found / reference_count``, or ``0.0`` when the field never appeared.

        Note:
            This measures *location*, not readability. A field whose label the
            model emitted with nothing legible after it counts as found, because
            recall exists to separate a dropped field from a fumbled one - a model
            that omits the registration number entirely would otherwise be
            indistinguishable from one that misreads it. Whether the located value
            was readable is :attr:`malformed_rate`, and the two are meant to be
            read together: recall 1.0 with a malformed rate of 0.2 means the model
            found every field and could not read a fifth of them.
        """
        return self.found / self.reference_count if self.reference_count else 0.0

    @property
    def exact_rate(self) -> float:
        """Fraction of reference fields read character for character.

        Returns:
            ``exact / reference_count``, or ``0.0`` when the field never appeared.
            Over reference fields rather than found ones, so dropping a field
            cannot improve the score.
        """
        return self.exact / self.reference_count if self.reference_count else 0.0

    @property
    def digit_rate(self) -> float:
        """Fraction of reference fields whose digits were read correctly.

        Returns:
            ``digits_equal / reference_count``, or ``0.0``. The numeral-blind
            score; compare against :attr:`exact_rate` to size the script problem.
        """
        return self.digits_equal / self.reference_count if self.reference_count else 0.0

    @property
    def malformed_rate(self) -> float:
        """Fraction of predicted fields with no readable value.

        Returns:
            ``malformed / predicted_count``, or ``0.0``. Reference-free, so this
            is the rate an inference deployment can compute on its own documents.
        """
        return self.malformed / self.predicted_count if self.predicted_count else 0.0

    @property
    def mean_confidence(self) -> float:
        """Average confidence over predicted values that had one.

        Returns:
            The mean, or ``0.0`` when no confidence was recorded.
        """
        return (
            self.confidence_total / self.scored_confidence
            if self.scored_confidence
            else 0.0
        )

    @property
    def mean_weakest(self) -> float:
        """Average weakest-token probability over predicted values.

        Returns:
            The mean, or ``0.0`` when no confidence was recorded. Lower than
            :attr:`mean_confidence` by construction, and the more honest of the
            two for a field that is only correct if every digit is.
        """
        return (
            self.weakest_total / self.scored_confidence if self.scored_confidence else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a JSON-serialisable mapping.

        Returns:
            Counts and rates for this field; the confidence block is omitted when
            uncertainty was not recorded.
        """
        payload: dict[str, Any] = {
            "reference_count": self.reference_count,
            "predicted_count": self.predicted_count,
            "recall": round(self.recall, 5),
            "exact_rate": round(self.exact_rate, 5),
            "digit_rate": round(self.digit_rate, 5),
            "missing": self.missing,
            "spurious": self.spurious,
            "malformed": self.malformed,
            "malformed_rate": round(self.malformed_rate, 5),
            "check_failed": self.check_failed,
            "repeated_digits": self.repeated_digits,
        }
        if self.scored_confidence:
            payload["confidence"] = {
                "scored": self.scored_confidence,
                "mean": round(self.mean_confidence, 5),
                "mean_weakest_token": round(self.mean_weakest, 5),
                "below_floor": self.low_confidence,
            }
        return payload


@dataclass(frozen=True, slots=True)
class FieldAudit:
    """The finished audit over one evaluation split.

    Attributes:
        records: Records audited.
        stats: One entry per audited field, in configuration order.
        findings: Individual faults, capped.
        findings_dropped: Faults beyond the cap. Reported so the findings list
            cannot be mistaken for exhaustive.
        fault_counts: Every fault kind and how often it occurred, uncapped - the
            totals stay complete even when the examples do not.
        confidence_available: Whether uncertainty was recorded. ``False`` under
            beam search or sampling, where the numbers would be meaningless.
        confidence_floor: The probability below which a value was flagged.
        rules: The loaded configuration, so a saved report carries the patterns
            that produced it. A field-validity rate is uninterpretable without
            the pattern it was measured against.
    """

    records: int
    stats: tuple[FieldStats, ...]
    findings: tuple[FieldFinding, ...] = ()
    findings_dropped: int = 0
    fault_counts: tuple[tuple[str, int], ...] = ()
    confidence_available: bool = False
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR
    rules: Mapping[str, Any] | None = None

    def stats_for(self, kind: FieldKind) -> FieldStats | None:
        """Look up one field's statistics.

        Args:
            kind: The field wanted.

        Returns:
            Its stats, or ``None`` if the field was not audited.
        """
        for entry in self.stats:
            if entry.kind is kind:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a JSON-serialisable mapping.

        Returns:
            A mapping suitable for ``json.dump``, nested under one key per field.
        """
        return {
            "records": self.records,
            "confidence_available": self.confidence_available,
            "confidence_floor": self.confidence_floor,
            "fields": {entry.kind.value: entry.to_dict() for entry in self.stats},
            "fault_counts": dict(self.fault_counts),
            "findings": [finding.to_dict() for finding in self.findings],
            "findings_dropped": self.findings_dropped,
            "rules": dict(self.rules) if self.rules is not None else None,
        }

    def describe(self, max_findings: int = 15) -> str:
        """Multi-line human-readable report.

        Args:
            max_findings: How many example findings to print.

        Returns:
            A per-field table followed by example faults, formatted for a
            terminal.
        """
        lines = [f"field audit over {self.records} records"]
        header = (
            f"  {'field':<21}{'ref':>5}{'recall':>8}{'exact':>8}{'digits':>8}"
            f"{'malf':>6}{'miss':>6}{'spur':>6}"
        )
        lines.append(header)
        for entry in self.stats:
            lines.append(
                f"  {entry.kind.value:<21}{entry.reference_count:>5}"
                f"{entry.recall:>8.4f}{entry.exact_rate:>8.4f}{entry.digit_rate:>8.4f}"
                f"{entry.malformed:>6}{entry.missing:>6}{entry.spurious:>6}"
            )
        if self.confidence_available:
            lines.append(f"  confidence (floor {self.confidence_floor:.2f}):")
            for entry in self.stats:
                if not entry.scored_confidence:
                    continue
                lines.append(
                    f"    {entry.kind.value:<21}mean {entry.mean_confidence:.3f}  "
                    f"weakest {entry.mean_weakest:.3f}  "
                    f"below floor {entry.low_confidence}"
                )
        else:
            lines.append("  confidence: not recorded (decoding is not greedy)")
        if self.fault_counts:
            summary = ", ".join(f"{name} {count}" for name, count in self.fault_counts)
            lines.append(f"  faults: {summary}")
        if self.findings:
            lines.append("  examples:")
            lines.extend(
                f"    {finding.describe()}" for finding in self.findings[:max_findings]
            )
            hidden = len(self.findings) - max_findings
            if hidden > 0 or self.findings_dropped:
                lines.append(
                    f"    ... {max(hidden, 0) + self.findings_dropped} more not shown"
                )
        return "\n".join(lines)


#: Resolves a value's character span to its confidence. Supplied per record by the
#: caller, which is what keeps this module free of torch: the audit never sees a
#: tokenizer or a score vector, only a function it can ask about a span.
ConfidenceLookup = Callable[[int, int], FieldConfidence | None]


@dataclass
class FieldAuditor:
    """Accumulates field-level verdicts one record at a time.

    Streaming, for the same reason
    :class:`~fa_ocr_lora.training.metrics.MetricAccumulator` is: evaluation runs on
    a machine with 4 GB of VRAM and little spare RAM, and holding several hundred
    pairs of full-page transcriptions to score them in a batch is memory this run
    does not have.

    Attributes:
        config: The loaded field-validation rules.
        confidence_floor: Weakest-token probability below which a predicted value
            is flagged as uncertain.
        max_findings: How many individual faults to retain.
    """

    config: FieldValidationConfig
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR
    max_findings: int = DEFAULT_MAX_FINDINGS
    _records: int = 0
    _counts: dict[FieldKind, dict[str, float]] = field(default_factory=dict)
    _findings: list[FieldFinding] = field(default_factory=list)
    _dropped: int = 0
    _faults: Counter[str] = field(default_factory=Counter)
    _confidence_seen: bool = False

    def update(
        self,
        record_id: str,
        reference: str,
        hypothesis: str,
        confidence: ConfidenceLookup | None = None,
    ) -> tuple[FieldFinding, ...]:
        """Audit one prediction against its reference.

        Args:
            record_id: Identifier of the evaluated record, used to attribute
                findings.
            reference: Ground-truth transcription.
            hypothesis: Model output.
            confidence: Optional resolver from a predicted value's character span
                to its recorded confidence.

        Returns:
            The findings for this record, so a caller can write them alongside the
            prediction. The same findings are accumulated internally, subject to
            the cap.
        """
        self._records += 1
        comparisons = compare_fields(
            extract_fields(reference, self.config),
            extract_fields(hypothesis, self.config),
        )
        found: list[FieldFinding] = []
        for comparison in comparisons:
            found.extend(self._score(record_id, comparison, confidence))
        return self._retain(found)

    def inspect(
        self,
        record_id: str,
        prediction: str,
        confidence: ConfidenceLookup | None = None,
    ) -> tuple[FieldFinding, ...]:
        """Audit a prediction with no reference to compare it against.

        The production path. Three of the faults :meth:`update` reports need a
        reference - missing, spurious, mismatch - and three do not: a label with
        nothing readable after it, a Jalali date that never existed, and a value
        whose weakest token the model itself gave a low probability. Those three
        are exactly what makes a reading on an untranscribed document reviewable,
        so they are reported here through the same code path and against the same
        threshold as during evaluation. A fault flagged in production therefore
        means what the eval report said it means.

        Counters that require a reference stay untouched, so the recall and
        exact-match rows of a :meth:`result` built from ``inspect`` calls alone are
        zero rather than misleading - there was nothing to be right about.

        Args:
            record_id: Identifier of the document, used to attribute findings.
            prediction: Model output for one document.
            confidence: Optional resolver from a predicted value's character span
                to its recorded confidence.

        Returns:
            The findings for this document, subject to the retention cap.
        """
        self._records += 1
        found: list[FieldFinding] = []
        for predicted in extract_fields(prediction, self.config):
            bag = self._bucket(predicted.kind)
            bag["predicted_count"] = bag.get("predicted_count", 0) + 1
            certainty = self._resolve(bag, predicted, confidence)
            found.extend(self._structural_faults(record_id, predicted, certainty))
        return self._retain(found)

    def _retain(self, found: list[FieldFinding]) -> tuple[FieldFinding, ...]:
        """Count findings and keep as many as the cap allows.

        Args:
            found: The findings one record produced.

        Returns:
            The same findings, so a caller can write them beside the prediction
            whether or not the cap retained them internally.
        """
        for finding in found:
            self._faults[finding.fault] += 1
            if len(self._findings) < self.max_findings:
                self._findings.append(finding)
            else:
                self._dropped += 1
        return tuple(found)

    def result(self) -> FieldAudit:
        """Finalise the accumulated counts.

        Returns:
            The audit. Every configured field gets a row even if it never
            appeared, because a zero row and a missing row read very differently:
            the first says the split contained no such field, the second looks
            like the audit forgot to check.
        """
        stats = tuple(self._stats_for(spec.kind) for spec in self.config.specs)
        return FieldAudit(
            records=self._records,
            stats=stats,
            findings=tuple(self._findings),
            findings_dropped=self._dropped,
            fault_counts=tuple(self._faults.most_common()),
            confidence_available=self._confidence_seen,
            confidence_floor=self.confidence_floor,
            rules=self.config.to_dict(),
        )

    def _bucket(self, kind: FieldKind) -> dict[str, float]:
        """Fetch or create the counter bag for one field.

        Args:
            kind: The field wanted.

        Returns:
            Its mutable counters.
        """
        return self._counts.setdefault(kind, {})

    def _stats_for(self, kind: FieldKind) -> FieldStats:
        """Freeze one field's counters into a :class:`FieldStats`.

        Args:
            kind: The field wanted.

        Returns:
            The stats, all-zero when the field never appeared.
        """
        bag = self._counts.get(kind, {})
        return FieldStats(
            kind=kind,
            reference_count=int(bag.get("reference_count", 0)),
            found=int(bag.get("found", 0)),
            missing=int(bag.get("missing", 0)),
            spurious=int(bag.get("spurious", 0)),
            exact=int(bag.get("exact", 0)),
            digits_equal=int(bag.get("digits_equal", 0)),
            predicted_count=int(bag.get("predicted_count", 0)),
            malformed=int(bag.get("malformed", 0)),
            check_failed=int(bag.get("check_failed", 0)),
            repeated_digits=int(bag.get("repeated_digits", 0)),
            low_confidence=int(bag.get("low_confidence", 0)),
            scored_confidence=int(bag.get("scored_confidence", 0)),
            confidence_total=bag.get("confidence_total", 0.0),
            weakest_total=bag.get("weakest_total", 0.0),
        )

    def _score(
        self,
        record_id: str,
        comparison: FieldComparison,
        confidence: ConfidenceLookup | None,
    ) -> list[FieldFinding]:
        """Fold one comparison into the counters and emit its faults.

        Args:
            record_id: Identifier of the evaluated record.
            comparison: One reference field beside what the model read for it.
            confidence: Optional span-to-confidence resolver.

        Returns:
            The findings this comparison produced, possibly none.
        """
        bag = self._bucket(comparison.kind)
        reference, predicted = comparison.reference, comparison.hypothesis
        findings: list[FieldFinding] = []

        if reference is not None:
            bag["reference_count"] = bag.get("reference_count", 0) + 1
        if predicted is None:
            bag["missing"] = bag.get("missing", 0) + 1
            findings.append(
                FieldFinding(
                    record_id=record_id,
                    kind=comparison.kind,
                    fault=FindingKind.MISSING,
                    reference=reference.value if reference else "",
                )
            )
            return findings

        bag["predicted_count"] = bag.get("predicted_count", 0) + 1
        certainty = self._resolve(bag, predicted, confidence)
        findings.extend(
            self._structural_faults(
                record_id, predicted, certainty, reference.value if reference else ""
            )
        )

        if reference is None:
            bag["spurious"] = bag.get("spurious", 0) + 1
            findings.append(
                FieldFinding(
                    record_id=record_id,
                    kind=comparison.kind,
                    fault=FindingKind.SPURIOUS,
                    prediction=predicted.value,
                    confidence=certainty,
                )
            )
            return findings

        bag["found"] = bag.get("found", 0) + 1
        if comparison.exact:
            bag["exact"] = bag.get("exact", 0) + 1
        if comparison.digits_equal:
            bag["digits_equal"] = bag.get("digits_equal", 0) + 1
        if not comparison.exact and predicted.format_ok:
            # A malformed prediction is excluded here, not overlooked: it cannot
            # equal the reference, and the malformed finding above already carries
            # both sides. Two findings for one fault would double the count.
            #
            # A value whose digits are right but whose script is wrong is reported
            # under its own fault, not as a mismatch. Both are errors, but one is
            # fixed by numeral normalisation and the other by the vision tower,
            # and merging them into one count hides which of the two you have.
            fault = (
                FindingKind.SCRIPT_ONLY
                if comparison.digits_equal
                else FindingKind.MISMATCH
            )
            findings.append(
                FieldFinding(
                    record_id=record_id,
                    kind=comparison.kind,
                    fault=fault,
                    reference=reference.value,
                    prediction=predicted.value,
                    confidence=certainty,
                )
            )
        return findings

    def _resolve(
        self,
        bag: dict[str, float],
        predicted: ExtractedField,
        confidence: ConfidenceLookup | None,
    ) -> FieldConfidence | None:
        """Look up a predicted value's confidence and fold it into the counters.

        Args:
            bag: The field's mutable counters.
            predicted: The predicted field.
            confidence: Optional span-to-confidence resolver.

        Returns:
            The confidence, or ``None`` when none was recorded or the value was
            empty.
        """
        if confidence is None or not predicted.value:
            return None
        certainty = confidence(predicted.start, predicted.end)
        if certainty is None:
            return None
        self._confidence_seen = True
        bag["scored_confidence"] = bag.get("scored_confidence", 0) + 1
        bag["confidence_total"] = bag.get("confidence_total", 0.0) + certainty.confidence
        bag["weakest_total"] = (
            bag.get("weakest_total", 0.0) + certainty.weakest_probability
        )
        return certainty

    def _structural_faults(
        self,
        record_id: str,
        predicted: ExtractedField,
        certainty: FieldConfidence | None,
        reference_value: str = "",
    ) -> list[FieldFinding]:
        """Emit the reference-free faults of one predicted field.

        These are the checks that still work at inference time, so they are decided
        without consulting the reference. It is only *reported* here when there is
        one, so a finding a reviewer reads is not missing half the story.

        Args:
            record_id: Identifier of the evaluated record.
            predicted: The predicted field.
            certainty: Its recorded confidence, if any.
            reference_value: The reference value when evaluating, ``""`` in
                production. Never used to decide a fault, only to annotate it.

        Returns:
            One finding per fault found, possibly none.
        """
        bag = self._bucket(predicted.kind)
        findings: list[FieldFinding] = []
        if not predicted.format_ok:
            bag["malformed"] = bag.get("malformed", 0) + 1
            findings.append(
                FieldFinding(
                    record_id=record_id,
                    kind=predicted.kind,
                    fault=FindingKind.MALFORMED,
                    reference=reference_value,
                    detail=f"label {predicted.label!r} followed by {predicted.excerpt!r}",
                )
            )
        if predicted.check_ok is False:
            bag["check_failed"] = bag.get("check_failed", 0) + 1
            findings.append(
                FieldFinding(
                    record_id=record_id,
                    kind=predicted.kind,
                    fault=FindingKind.CHECK_FAILED,
                    prediction=predicted.value,
                    detail="not a date that existed",
                    confidence=certainty,
                )
            )
        if predicted.repeated_digits:
            bag["repeated_digits"] = bag.get("repeated_digits", 0) + 1
            findings.append(
                FieldFinding(
                    record_id=record_id,
                    kind=predicted.kind,
                    fault=FindingKind.REPEATED_DIGITS,
                    prediction=predicted.value,
                    confidence=certainty,
                )
            )
        floor = self.confidence_floor
        if certainty is not None and certainty.weakest_probability < floor:
            bag["low_confidence"] = bag.get("low_confidence", 0) + 1
            findings.append(
                FieldFinding(
                    record_id=record_id,
                    kind=predicted.kind,
                    fault=FindingKind.LOW_CONFIDENCE,
                    prediction=predicted.value,
                    confidence=certainty,
                )
            )
        return findings


def summarize_fields(fields: Iterable[ExtractedField]) -> tuple[dict[str, Any], ...]:
    """Render extracted fields for a per-record JSONL row.

    Written beside each prediction so a reviewer can see what the audit found on
    that page without re-running the extractor.

    Args:
        fields: The fields extracted from one transcription.

    Returns:
        One mapping per field, in reading order.
    """
    return tuple(
        {
            "field": item.kind.value,
            "label": item.label,
            "value": item.value,
            "start": item.start,
            "end": item.end,
            "format_ok": item.format_ok,
            **({"check_ok": item.check_ok} if item.check_ok is not None else {}),
            **({"repeated_digits": True} if item.repeated_digits else {}),
            **({"excerpt": item.excerpt} if item.excerpt else {}),
        }
        for item in fields
    )
