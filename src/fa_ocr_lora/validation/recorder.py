"""The generation hook that records what the model believed, step by step.

Separated from :mod:`fa_ocr_lora.validation.confidence` because this is the only
part of the uncertainty layer that needs torch and transformers. Everything that
consumes the recorded numbers - aggregation, the report, the tests - imports the
other module and stays framework-free.

**Why a logits processor.** The alternatives both cost more than the numbers are
worth. ``generate(output_scores=True)`` retains every score vector: 103,424 floats
by up to 1,024 steps is ~424 MiB of fp32, on a card with 4,096 MiB total, competing
with the KV cache for the same headroom. A second forward pass over the finished
sequence to read log-probabilities back costs a full vocabulary projection over
~1,600 positions - the exact tensor
:class:`fa_ocr_lora.training.modeling.SlicedLMHead` exists to avoid materialising.
A processor sees each distribution once, keeps four scalars from it and lets it go:
~32 bytes per step.

**Why greedy only.** The recorder identifies the emitted token as the argmax of
the scores it is shown, which is precisely what greedy decoding does, so the
identification is exact. Under sampling or beam search the emitted token is not
the argmax, and the recorded probability would belong to a token that was never
produced - worse than no number, because it looks like one. So
:func:`recorder_for` returns ``None`` for any other strategy and the audit reports
uncertainty as unavailable rather than guessing. ``eval.num_beams`` defaults to 1
for an unrelated reason - greedy keeps CER reproducible - so the usual path does
get confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from .confidence import TokenConfidence

__all__ = [
    "BatchConfidenceRecorder",
    "ConfidenceRecorder",
    "batch_recorder_for",
    "processor_list",
    "recorder_for",
]


@dataclass
class ConfidenceRecorder(LogitsProcessor):
    """Records the model's certainty at every greedy generation step.

    ``generate`` merges a caller-supplied processor onto the end of its own list,
    so the scores this sees are the ones the token is actually chosen from - after
    any repetition penalty or length constraint the generation config imposes. The
    scores are returned untouched: this observes generation, it does not steer it.

    Batch size must be 1. Evaluation generates one record at a time, and reading a
    batch would need the per-row output lengths that only ``generate`` knows.

    Attributes:
        steps: One entry per generation step, in order.
    """

    steps: list[TokenConfidence] = field(default_factory=list)

    def reset(self) -> None:
        """Discard the recorded steps, so one instance can serve many records."""
        self.steps.clear()

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Record one step and pass the scores through unmodified.

        Args:
            input_ids: The sequence so far. Unused; present for the interface.
            scores: Final score vector for this step, shape ``(1, vocab)``.

        Returns:
            ``scores``, unmodified.

        Raises:
            ValueError: If the batch is not size 1, since the recorded numbers
                would silently describe whichever row came first.
        """
        if scores.shape[0] != 1:
            raise ValueError(
                f"ConfidenceRecorder needs batch size 1, got {scores.shape[0]}; "
                "evaluation generates one record at a time"
            )
        # float32 regardless of the model dtype. A bf16 softmax over 103k logits
        # has ~3 decimal digits of mantissa, enough to reorder two near-tied
        # candidates - and the gap between those two is what `margin` reports.
        logprobs = torch.log_softmax(scores[0].float(), dim=-1)
        top = torch.topk(logprobs, k=2)
        # nansum, not sum: a masked token has probability 0 and log-probability
        # -inf, whose product is NaN. Its true contribution to the entropy is 0.
        entropy = float(-(logprobs.exp() * logprobs).nansum())
        self.steps.append(
            TokenConfidence(
                token_id=int(top.indices[0]),
                logprob=float(top.values[0]),
                entropy=entropy,
                margin=float(top.values[0] - top.values[1]),
            )
        )
        return scores


def recorder_for(num_beams: int, do_sample: bool = False) -> ConfidenceRecorder | None:
    """Build a recorder, but only when the decoding strategy makes it sound.

    Args:
        num_beams: Beam count from the eval configuration.
        do_sample: Whether generation samples rather than taking the argmax.

    Returns:
        A recorder for greedy decoding, or ``None`` for beam search or sampling,
        where the emitted token is not the argmax so a recorded probability would
        describe a token that was never produced.
    """
    if num_beams != 1 or do_sample:
        return None
    return ConfidenceRecorder()


@dataclass
class BatchConfidenceRecorder(LogitsProcessor):
    """Records per-row certainty at every greedy step of a batched decode.

    The batch-size-1 restriction on :class:`ConfidenceRecorder` exists because a
    single ``steps`` list cannot say which row a number came from. This class
    keeps one list per row instead, so batched evaluation gets the same
    weakest-token confidence the field audit needs - which is the whole reason
    the audit can name *which page to re-check*.

    Rows finish at different steps. ``generate`` keeps feeding finished rows (it
    emits padding past their EOS), so the tail of a short row's list describes
    tokens that were never part of its answer. That is harmless and deliberate:
    :func:`~fa_ocr_lora.validation.confidence.span_confidence` is indexed by the
    *trimmed* token ids, so the surplus entries are never reached. Trimming here
    instead would need the per-row output lengths that only the caller knows.

    Attributes:
        rows: One list of per-step statistics per batch row, in row order.
    """

    rows: list[list[TokenConfidence]] = field(default_factory=list)

    def reset(self, batch_size: int) -> None:
        """Start recording a new batch.

        Args:
            batch_size: Rows the next ``generate`` call will carry.
        """
        self.rows = [[] for _ in range(batch_size)]

    def steps_for(self, row: int) -> tuple[TokenConfidence, ...]:
        """Recorded steps for one row of the current batch.

        Args:
            row: Row index within the batch.

        Returns:
            The per-step statistics, in generation order.
        """
        return tuple(self.rows[row])

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Record one step for every row and pass the scores through unmodified.

        Args:
            input_ids: The sequences so far. Unused; present for the interface.
            scores: Final score vectors for this step, shape ``(B, vocab)``.

        Returns:
            ``scores``, unmodified.

        Raises:
            ValueError: If the batch width does not match :meth:`reset`, which
                would silently misattribute every number to the wrong record.
        """
        width = int(scores.shape[0])
        if width != len(self.rows):
            raise ValueError(
                f"BatchConfidenceRecorder was reset for {len(self.rows)} row(s) "
                f"but generation produced {width}; call reset(batch_size) before "
                "each generate()"
            )
        # One fused pass over the whole batch rather than a Python loop over
        # rows: the log_softmax is the expensive part and it vectorises, which
        # is the point of batching in the first place. float32 for the same
        # reason as the single-row path - bf16 cannot resolve `margin`.
        logprobs = torch.log_softmax(scores.float(), dim=-1)
        top = torch.topk(logprobs, k=2, dim=-1)
        entropy = -(logprobs.exp() * logprobs).nansum(dim=-1)
        # One host transfer for the batch, not four per row.
        best_ids = top.indices[:, 0].tolist()
        best = top.values[:, 0].tolist()
        second = top.values[:, 1].tolist()
        entropies = entropy.tolist()
        for row in range(width):
            self.rows[row].append(
                TokenConfidence(
                    token_id=int(best_ids[row]),
                    logprob=float(best[row]),
                    entropy=float(entropies[row]),
                    margin=float(best[row] - second[row]),
                )
            )
        return scores


def batch_recorder_for(
    num_beams: int, do_sample: bool = False
) -> BatchConfidenceRecorder | None:
    """Build a batched recorder, under the same soundness rule as the single one.

    Args:
        num_beams: Beam count from the eval configuration.
        do_sample: Whether generation samples rather than taking the argmax.

    Returns:
        A recorder for greedy decoding, or ``None`` for any other strategy.
    """
    if num_beams != 1 or do_sample:
        return None
    return BatchConfidenceRecorder()


def processor_list(
    recorder: ConfidenceRecorder | BatchConfidenceRecorder | None,
) -> LogitsProcessorList | None:
    """Wrap a recorder for ``generate(logits_processor=...)``.

    Args:
        recorder: The recorder, or ``None``.

    Returns:
        A single-element processor list, or ``None`` so the argument can be passed
        straight through without the caller branching.
    """
    return None if recorder is None else LogitsProcessorList([recorder])
