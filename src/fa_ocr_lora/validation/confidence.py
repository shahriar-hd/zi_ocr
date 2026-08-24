"""Per-field uncertainty: the data model and the character-to-token mapping.

A field-level verdict answers "is this value well formed, and does it agree with
the reference?". In production there is no reference, and a well-formed value can
still be wrong - a ``۵`` read as ``۶`` is a perfectly legal registration number.
What is left to report is how sure the model was, and that is recoverable exactly:
greedy decoding emits the argmax of a distribution, and the probability it
assigned to that argmax is the model's own statement of confidence.

This module holds the parts of that with no framework dependency - the recorded
statistics, their aggregation over a field, and the mapping from a character span
onto the tokens that produced it. The generation hook that fills them in lives in
:mod:`fa_ocr_lora.validation.recorder`, which needs torch; keeping the two apart
is what lets the report and its tests run without it.

**Mapping fields onto tokens.** A field is a character span in the decoded string;
confidence lives on tokens. :func:`token_spans` recovers the boundary by decoding
each prefix of the token sequence and recording how long the result was - the
standard incremental-detokenisation trick, exact for this tokeniser provided the
caller compares against the same ``decode(..., skip_special_tokens=True)`` string
the extractor ran on. Tokens that render to nothing, such as the EOS, get an empty
span rather than being dropped, so the result stays index-aligned with the tokens.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

__all__ = [
    "FieldConfidence",
    "LazySpanConfidence",
    "TokenConfidence",
    "span_confidence",
    "token_spans",
]

#: Entropy is recorded in nats and reported in bits, so a reader can compare it
#: against the ln/log2 of the vocabulary size (103,424 -> 16.7 bits at maximum
#: uncertainty) without knowing which base was used.
_LN_TWO: Final[float] = math.log(2.0)


class SupportsDecode(Protocol):
    """The one tokenizer method :func:`token_spans` needs."""

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        """Render token ids as text.

        Args:
            token_ids: The ids to decode.
            **kwargs: Passed through to the implementation.

        Returns:
            The decoded string.
        """
        ...


@dataclass(frozen=True, slots=True)
class TokenConfidence:
    """What the model believed at one generation step.

    Attributes:
        token_id: The emitted token, i.e. the argmax of the final scores.
        logprob: Natural log of the probability assigned to that token. ``0.0``
            is complete certainty; ``-2.303`` is a probability of 0.1.
        entropy: Shannon entropy of the whole distribution, in nats. Kept
            alongside the top-1 probability because the two describe different
            situations: a confident token drawn from an otherwise flat
            distribution is not the same as one drawn from a sharp one.
        margin: Log-probability gap to the runner-up token. A margin near zero is
            a coin flip between two readings - which for a digit is exactly the
            ``۵``/``۶`` confusion this adapter exists to remove.
    """

    token_id: int
    logprob: float
    entropy: float
    margin: float

    @property
    def probability(self) -> float:
        """The emitted token's probability.

        Returns:
            ``exp(logprob)``, in ``[0, 1]``.
        """
        return math.exp(self.logprob)


@dataclass(frozen=True, slots=True)
class FieldConfidence:
    """Confidence aggregated over the tokens of one field.

    Attributes:
        tokens: How many generated tokens the field's characters span.
        mean_logprob: Mean per-token log-probability over the span.
        min_logprob: The weakest token's log-probability. This is the number to
            threshold a sensitive field on: a registration number is wrong if any
            one of its digits is wrong, so the value is only as trustworthy as its
            least certain token - whereas a mean over five confident digits and
            one guess reads as confident.
        max_entropy: The most spread-out distribution in the span, in nats.
        min_margin: The smallest gap to a runner-up in the span.
    """

    tokens: int
    mean_logprob: float
    min_logprob: float
    max_entropy: float
    min_margin: float

    @property
    def confidence(self) -> float:
        """Geometric-mean token probability over the field.

        Returns:
            ``exp(mean_logprob)`` in ``[0, 1]``. Geometric rather than arithmetic
            because it is the probability the model assigned to the value as a
            sequence, normalised for its length.
        """
        return math.exp(self.mean_logprob)

    @property
    def weakest_probability(self) -> float:
        """Probability of the least certain token in the field.

        Returns:
            ``exp(min_logprob)`` in ``[0, 1]``.
        """
        return math.exp(self.min_logprob)

    def to_dict(self) -> dict[str, float | int]:
        """Flatten to a JSON-serialisable mapping.

        Returns:
            The five recorded statistics plus the two derived probabilities,
            rounded, with entropy converted to bits.
        """
        return {
            "tokens": self.tokens,
            "confidence": round(self.confidence, 5),
            "weakest_probability": round(self.weakest_probability, 5),
            "mean_logprob": round(self.mean_logprob, 5),
            "min_logprob": round(self.min_logprob, 5),
            "max_entropy_bits": round(self.max_entropy / _LN_TWO, 5),
            "min_margin": round(self.min_margin, 5),
        }

    def describe(self) -> str:
        """One-line rendering for a log.

        Returns:
            The confidence and its weakest token, e.g.
            ``conf 0.62 (weakest 0.31 over 6 tokens)``.
        """
        return (
            f"conf {self.confidence:.2f} "
            f"(weakest {self.weakest_probability:.2f} over {self.tokens} tokens)"
        )


def token_spans(
    tokenizer: SupportsDecode, token_ids: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Map each generated token to the character span it contributed.

    Args:
        tokenizer: The model's tokenizer; only ``decode`` is used.
        token_ids: The generated tokens, excluding the prompt.

    Returns:
        One ``(start, end)`` pair per token, index-aligned with ``token_ids``. A
        token that renders to nothing gets an empty span at the current offset.
    """
    spans: list[tuple[int, int]] = []
    previous = 0
    ids = list(token_ids)
    for index in range(1, len(ids) + 1):
        length = len(tokenizer.decode(ids[:index], skip_special_tokens=True))
        # Detokenisation is not guaranteed monotone where a trailing space is
        # absorbed into the next piece, so clamp instead of emitting a negative
        # span - one misaligned character is preferable to an unusable index.
        spans.append((previous, max(length, previous)))
        previous = max(length, previous)
    return tuple(spans)


def span_confidence(
    steps: Sequence[TokenConfidence],
    spans: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> FieldConfidence | None:
    """Aggregate confidence over the tokens covering a character span.

    Args:
        steps: Recorded steps, index-aligned with ``spans``.
        spans: Character span per token, from :func:`token_spans`.
        start: First character of the field's value.
        end: One past the last character.

    Returns:
        The aggregate, or ``None`` when the span is empty or no token overlaps it.
        A malformed field has a zero-width value and so has no confidence, which
        is correct: there is no reading to be uncertain about.
    """
    if end <= start:
        return None
    overlapping = [
        step
        for step, (token_start, token_end) in zip(steps, spans, strict=False)
        if token_start < end and start < token_end
    ]
    if not overlapping:
        return None
    logprobs = [step.logprob for step in overlapping]
    return FieldConfidence(
        tokens=len(overlapping),
        mean_logprob=sum(logprobs) / len(logprobs),
        min_logprob=min(logprobs),
        max_entropy=max(step.entropy for step in overlapping),
        min_margin=min(step.margin for step in overlapping),
    )


@dataclass
class LazySpanConfidence:
    """Resolves character spans to confidence, decoding only when first asked.

    :func:`token_spans` costs one detokenisation per generated token, so building
    the map for every record would spend a measurable fraction of an evaluation on
    predictions that turn out to contain no sensitive field at all - a garbage or
    empty output being exactly the case where it is wasted. Deferring the work to
    the first query makes the cost proportional to the fields actually found.

    Attributes:
        tokenizer: The model's tokenizer.
        token_ids: The generated tokens, excluding the prompt.
        steps: Recorded per-step statistics, in the same order.
    """

    tokenizer: SupportsDecode
    token_ids: Sequence[int]
    steps: Sequence[TokenConfidence]
    _spans: tuple[tuple[int, int], ...] | None = field(default=None, repr=False)

    def __call__(self, start: int, end: int) -> FieldConfidence | None:
        """Resolve one character span.

        Args:
            start: First character of the value.
            end: One past the last character.

        Returns:
            The aggregated confidence, or ``None`` when nothing was recorded or no
            token overlaps the span.
        """
        if not self.steps:
            return None
        if self._spans is None:
            self._spans = token_spans(self.tokenizer, self.token_ids)
        return span_confidence(self.steps, self._spans, start, end)
