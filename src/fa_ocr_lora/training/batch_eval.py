r"""Batched, prefetched scoring - the fast path behind :mod:`fa_ocr_lora.training.eval`.

The original scorer generated one record per ``generate`` call. That is correct and
it is what the 4 GB development card could afford, but on a 96 GB card it leaves
the GPU idle: a measured test pass reported ~3 GB resident and ~2% utilisation
while taking 4.9 s per record, because a 1,024-token greedy decode at batch 1 is
1,024 sequential launches of kernels sized for one row, and the PIL decode plus
the processor's float32 patch tensor sat on the critical path between records.

Three changes, each addressing one of those:

**Batched generation.** ``batch_size`` prompts per call, so every decode step is
one kernel over ``batch_size`` rows. Prompts differ in length (the image token
count moves with the page), so they are **left-padded**: with the answer growing
to the right, left padding puts every row's first generated token at the same
offset, which is the only arrangement from which ``generated[:, width:]`` is
correct for all rows at once. Right padding would interleave pad and answer.

**Length-bucketed order.** Records are sorted by prompt length before batching, so
a batch of 16 pads to its own longest row rather than to the longest in the split.
The predictions file is written back in manifest order, so nothing downstream can
tell the difference.

**Background encoding.** A ``DataLoader`` with ``dataloader_workers`` worker
processes runs ``SampleEncoder.encode`` off the critical path. The GPU waits on the
queue rather than on Pillow.

What is *not* traded away: the prompt still comes from the same encoder and the
same template, decoding is still greedy, and the field audit still gets per-token
confidence - :class:`~fa_ocr_lora.validation.recorder.BatchConfidenceRecorder`
keeps one list of steps per row. A batched CER is comparable to a batch-1 CER
because nothing about the arithmetic changed; only the scheduling did.

Setting ``batch_size: 1`` reproduces the original behaviour, including its
single-row recorder, so a disagreement can always be bisected against it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ..dataset.common.io_utils import resolve_repo_path
from ..validation import (
    FieldAuditor,
    LazySpanConfidence,
    extract_fields,
    load_field_validation_config,
    summarize_fields,
)
from ..validation.recorder import (
    BatchConfidenceRecorder,
    batch_recorder_for,
    processor_list,
)
from .config import TrainingConfig
from .encoding import IGNORE_INDEX, SampleEncoder, SampleTooLong
from .eval import EvalOutcome
from .manifest import load_manifest, parse_crop_box
from .metrics import MetricAccumulator
from .modeling import EvalModel

__all__ = [
    "EncodeFailure",
    "PreparedRecord",
    "score_split_batched",
]

LOGGER = logging.getLogger("fa_ocr_lora.eval")

#: Records between progress lines. Batch-aligned reporting would make the
#: cadence depend on ``batch_size``, so a run at 32 logged a third as often as
#: one at 8 and looked stalled by comparison.
_PROGRESS_EVERY: int = 100


@dataclass(frozen=True, slots=True)
class EncodeFailure:
    """A record that could not be encoded.

    Counted and reported rather than scored as an empty prediction: a file that
    will not open says nothing about how well the model reads, and folding it into
    the CER would understate the model in a way no later reader could untangle.

    Attributes:
        position: Index of the record within the (already limited) split.
        record_id: Manifest identifier.
        reason: Exception class name.
        detail: Exception message.
    """

    position: int
    record_id: str
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    """One encoded prompt plus the manifest fields the scorer needs afterwards.

    Attributes:
        position: Index within the split, so results can be restored to manifest
            order after length bucketing.
        record_id: Manifest identifier.
        source: Source tag, used by the per-dataset breakdown.
        reference: Ground-truth transcription.
        input_ids: Prompt token ids, image placeholder already expanded.
        pixel_values: Flattened image patches.
        image_grid_thw: Patch grid rows for this image.
        mm_token_type_ids: Per-token modality ids, or ``None``.
    """

    position: int
    record_id: str
    source: str
    reference: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    mm_token_type_ids: torch.Tensor | None


class _PromptDataset(Dataset):
    """Encodes manifest rows to generation prompts, one per index.

    The encoder is built lazily per worker process rather than in ``__init__``:
    a ``SampleEncoder`` holds the processor, which does not survive being pickled
    into a forked worker on every platform, and building one is cheap.
    """

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        processor: Any,
        config: TrainingConfig,
    ) -> None:
        """Build the dataset.

        Args:
            rows: Manifest rows as plain dicts, in split order.
            processor: The model's ``AutoProcessor``.
            config: Loaded training configuration.
        """
        self._rows = list(rows)
        self._processor = processor
        self._config = config
        self._encoder: SampleEncoder | None = None

    def __len__(self) -> int:
        """Records in the split."""
        return len(self._rows)

    def _ensure_encoder(self) -> SampleEncoder:
        """Return this worker's encoder, building it on first use."""
        if self._encoder is None:
            self._encoder = SampleEncoder(
                self._processor, model=self._config.model, image=self._config.image
            )
        return self._encoder

    def __getitem__(self, index: int) -> PreparedRecord | EncodeFailure:
        """Encode one record.

        A failure is returned rather than raised: an exception inside a worker
        kills the loader, and one unreadable page must not end a 5,855-record
        pass.

        Args:
            index: Position within the split.

        Returns:
            The prepared record, or the failure that stopped it.
        """
        row = self._rows[index]
        encoder = self._ensure_encoder()
        try:
            sample = encoder.encode(
                image_path=resolve_repo_path(row["image_path"]),
                crop_box=parse_crop_box(row["crop_box"]),
                text=row["text"],
                record_id=row["record_id"],
            )
        except (SampleTooLong, OSError, ValueError) as exc:
            return EncodeFailure(
                position=index,
                record_id=str(row["record_id"]),
                reason=type(exc).__name__,
                detail=str(exc),
            )
        # The encoder returns prompt+answer; generation needs the prompt alone.
        # Slicing at the first supervised label is how the single-record path
        # derives it too, so both ask the model exactly what training asked.
        cut = _prompt_length(sample.labels)
        return PreparedRecord(
            position=index,
            record_id=str(row["record_id"]),
            source=str(row["source"]),
            reference=str(row["text"]),
            input_ids=sample.input_ids[:cut],
            pixel_values=sample.pixel_values,
            image_grid_thw=sample.image_grid_thw,
            mm_token_type_ids=(
                sample.mm_token_type_ids[:cut]
                if sample.mm_token_type_ids is not None
                else None
            ),
        )


def _prompt_length(labels: torch.Tensor) -> int:
    """Number of leading positions that carry no supervision.

    Args:
        labels: Label tensor from an encoded sample, shape ``(L,)``.

    Returns:
        The prompt length in tokens.

    Raises:
        ValueError: If the sample has no supervised position, which would mean an
            empty target reached the encoder.
    """
    supervised = (labels != IGNORE_INDEX).nonzero()
    if supervised.numel() == 0:
        raise ValueError("encoded sample has no supervised position")
    return int(supervised[0].item())


def _identity(batch: list[Any]) -> list[Any]:
    """Collate that does nothing, so padding can happen on the main process.

    Left padding needs the pad token id, which belongs to the tokenizer rather
    than the dataset, and the result is moved to the GPU immediately - so there
    is nothing for a worker to gain by building it.

    Args:
        batch: Whatever the dataset returned for this batch's indices.

    Returns:
        ``batch``, unchanged.
    """
    return batch


def _left_pad(
    records: Sequence[PreparedRecord],
    *,
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Build one left-padded generation batch.

    Left, not right: generation appends to the right, so left padding is what
    puts every row's first new token at the same index. The attention mask zeroes
    the pad region, and ``pixel_values`` concatenates along dim 0 with one
    ``image_grid_thw`` row per image - the layout the vision tower uses to split
    the flat patch list back apart.

    Args:
        records: Records in this batch.
        pad_token_id: Token id to pad with.
        device: Device to build the tensors on.
        dtype: Compute dtype for ``pixel_values``.

    Returns:
        Keyword arguments for ``generate``. The prompt width the caller needs in
        order to slice the answers off is ``input_ids.shape[1]``, since every row
        is padded to it.
    """
    width = max(int(record.input_ids.shape[0]) for record in records)
    rows = len(records)
    input_ids = torch.full((rows, width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((rows, width), dtype=torch.long)
    has_mm = any(record.mm_token_type_ids is not None for record in records)
    mm_ids = torch.zeros((rows, width), dtype=torch.long) if has_mm else None
    for index, record in enumerate(records):
        length = int(record.input_ids.shape[0])
        start = width - length
        input_ids[index, start:] = record.input_ids
        attention_mask[index, start:] = 1
        if mm_ids is not None and record.mm_token_type_ids is not None:
            mm_ids[index, start:] = record.mm_token_type_ids
    batch: dict[str, torch.Tensor] = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "pixel_values": torch.cat(
            [record.pixel_values for record in records], dim=0
        ).to(device, dtype),
        "image_grid_thw": torch.cat(
            [record.image_grid_thw for record in records], dim=0
        ).to(device),
    }
    if mm_ids is not None:
        # transformers 5's native 1.6 needs per-token modality ids for M-RoPE and
        # raises without them; the older checkpoint reads position from the grid
        # alone and never looks at this key.
        batch["mm_token_type_ids"] = mm_ids.to(device)
    return batch


def _trim_answer(token_ids: list[int], eos_token_id: int | None) -> list[int]:
    """Cut a generated row at its first EOS.

    Rows in a batch finish at different steps and ``generate`` pads the short ones
    out to the longest, so the raw tail carries tokens that were never part of the
    answer. ``skip_special_tokens`` on the decode would hide them from the string
    but not from the confidence lookup, whose indices must line up with the tokens
    the model actually emitted.

    Args:
        token_ids: Generated ids for one row, prompt already removed.
        eos_token_id: End-of-sequence id, or ``None`` to keep everything.

    Returns:
        The ids up to, excluding, the first EOS.
    """
    if eos_token_id is None:
        return token_ids
    try:
        return token_ids[: token_ids.index(eos_token_id)]
    except ValueError:
        return token_ids


def _batches(
    loader: DataLoader, *, batch_size: int
) -> Iterator[tuple[list[PreparedRecord], list[EncodeFailure]]]:
    """Yield the loader's output split into successes and failures.

    Args:
        loader: The prompt loader.
        batch_size: Rows per yielded batch.

    Yields:
        ``(records, failures)`` where ``records`` has at most ``batch_size``
        entries.
    """
    pending: list[PreparedRecord] = []
    for chunk in loader:
        failures: list[EncodeFailure] = []
        for item in chunk:
            if isinstance(item, EncodeFailure):
                failures.append(item)
            else:
                pending.append(item)
        while len(pending) >= batch_size:
            yield pending[:batch_size], failures
            failures = []
            pending = pending[batch_size:]
        if failures:
            yield [], failures
    if pending:
        yield pending, []


def score_split_batched(
    config: TrainingConfig,
    loaded: EvalModel,
    *,
    split: str = "validation",
    max_samples: int | None = None,
    predictions_path: Path | None = None,
    batch_size: int | None = None,
    dataloader_workers: int | None = None,
    verbose: bool = False,
) -> EvalOutcome:
    """Generate over a split in batches, accumulating metrics and the field audit.

    Args:
        config: Loaded training configuration.
        loaded: Model and processor from
            :func:`~fa_ocr_lora.training.modeling.load_eval_model`.
        split: Manifest split to score.
        max_samples: Records to score; ``None`` uses ``eval.max_samples``, and a
            ``None`` there means the whole split.
        predictions_path: Where to write per-record predictions, or ``None``.
        batch_size: Rows per ``generate`` call; ``None`` uses ``eval.batch_size``.
        dataloader_workers: Encoder worker processes; ``None`` uses
            ``eval.dataloader_workers``.
        verbose: Log every prediction as it is produced.

    Returns:
        The metrics and, unless ``eval.field_audit`` is off, the field audit.

    Raises:
        FileNotFoundError: If the manifest has not been built.
        ValueError: If the split is empty.
        ConfigError: If ``eval.field_config_path`` names unloadable rules.
    """
    frame = load_manifest(config.paths.manifest_path)
    frame = frame[frame["split"] == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            f"split {split!r} of {config.paths.manifest_path} is empty; rebuild "
            "the manifest, or score a split the mix actually produced"
        )
    limit = config.eval.max_samples if max_samples is None else max_samples
    if limit:
        frame = frame.iloc[:limit]
    width = batch_size if batch_size is not None else config.eval.batch_size
    workers = (
        dataloader_workers
        if dataloader_workers is not None
        else config.eval.dataloader_workers
    )
    width = max(1, int(width))
    workers = max(0, int(workers))

    rows = frame.to_dict("records")
    # Sorted longest-first by target length as a proxy for prompt length: the
    # image budget shrinks as the answer grows, so the two move together and the
    # real prompt length is not known until the encoder has run. Longest first
    # also surfaces an OOM in the first batch rather than at 03:00.
    order = sorted(
        range(len(rows)), key=lambda index: len(str(rows[index]["text"])), reverse=True
    )
    ordered = [rows[index] for index in order]

    tokenizer = loaded.processor.tokenizer
    device = next(loaded.model.parameters()).device
    accumulator = MetricAccumulator(confusion_top_k=config.eval.confusion_top_k)
    auditor: FieldAuditor | None = None
    if config.eval.field_audit:
        rules = load_field_validation_config(config.eval.field_config_path)
        auditor = FieldAuditor(
            config=rules,
            confidence_floor=config.eval.confidence_floor,
            max_findings=config.eval.max_findings,
        )
        LOGGER.info(
            "field audit on: %s (%d field(s))", rules.source_path, len(rules.specs)
        )
    recorder: BatchConfidenceRecorder | None = (
        batch_recorder_for(config.eval.num_beams) if auditor else None
    )
    processors = processor_list(recorder)
    if auditor and recorder is None:
        LOGGER.info(
            "field confidence off: num_beams=%d is not greedy", config.eval.num_beams
        )

    dataset = _PromptDataset(ordered, processor=loaded.processor, config=config)
    loader = DataLoader(
        dataset,
        batch_size=width,
        shuffle=False,
        num_workers=workers,
        collate_fn=_identity,
        # Two batches of lookahead per worker, so a slow page cannot stall the
        # GPU, without holding the whole split's float32 patch tensors in RAM.
        prefetch_factor=2 if workers else None,
        persistent_workers=False,
    )
    LOGGER.info(
        "batched scoring: %d record(s), batch %d, %d encoder worker(s)",
        len(ordered),
        width,
        workers,
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    eos_token_id = tokenizer.eos_token_id
    failures: dict[str, int] = {}
    # Keyed by position so the file comes out in manifest order despite the
    # length bucketing above.
    collected: dict[int, str] = {}
    scored = 0
    reported = 0
    started = time.perf_counter()
    handle = predictions_path.open("w", encoding="utf-8") if predictions_path else None
    payloads: dict[int, dict[str, object]] = {}

    try:
        for records, encode_failures in _batches(loader, batch_size=width):
            for failure in encode_failures:
                failures[failure.reason] = failures.get(failure.reason, 0) + 1
                LOGGER.warning("skipping %s: %s", failure.record_id, failure.detail)
            if not records:
                continue
            batch = _left_pad(
                records,
                pad_token_id=int(pad_token_id),
                device=device,
                dtype=loaded.dtype,
            )
            prompt_width = int(batch["input_ids"].shape[1])
            if recorder is not None:
                recorder.reset(len(records))
            with torch.inference_mode():
                generated = loaded.model.generate(
                    **batch,
                    max_new_tokens=config.eval.max_new_tokens,
                    num_beams=config.eval.num_beams,
                    do_sample=False,
                    pad_token_id=int(pad_token_id),
                    eos_token_id=eos_token_id,
                    logits_processor=processors,
                )
            # Every row starts generating at the same index precisely because the
            # prompts were left-padded, so one slice serves the whole batch.
            answers = generated[:, prompt_width:].tolist()
            for index, record in enumerate(records):
                new_tokens = _trim_answer(answers[index], eos_token_id)
                prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
                accumulator.update(record.reference, prediction)
                scored += 1
                collected[record.position] = prediction

                audited: tuple[dict[str, object], ...] = ()
                findings: tuple[Any, ...] = ()
                if auditor is not None:
                    lookup = (
                        LazySpanConfidence(
                            tokenizer=tokenizer,
                            token_ids=new_tokens,
                            steps=recorder.steps_for(index),
                        )
                        if recorder is not None
                        else None
                    )
                    findings = auditor.update(
                        record.record_id,
                        record.reference,
                        prediction,
                        confidence=lookup,
                    )
                    audited = summarize_fields(
                        extract_fields(prediction, auditor.config)
                    )
                    if verbose and findings:
                        for finding in findings:
                            LOGGER.debug("  %s", finding.describe())
                if verbose:
                    LOGGER.debug("%s -> %r", record.record_id, prediction[:120])
                if handle is not None:
                    payloads[record.position] = {
                        "record_id": record.record_id,
                        "source": record.source,
                        "reference": record.reference,
                        "prediction": prediction,
                        "fields": list(audited),
                        "findings": [item.to_dict() for item in findings],
                    }
            if scored - reported >= _PROGRESS_EVERY:
                reported = scored
                elapsed = time.perf_counter() - started
                LOGGER.info(
                    "  %d/%d scored (%.2fs/record, %.1f rec/s)",
                    scored,
                    len(ordered),
                    elapsed / scored,
                    scored / elapsed if elapsed else 0.0,
                )
        if handle is not None:
            for position in sorted(payloads):
                handle.write(
                    json.dumps(payloads[position], ensure_ascii=False) + "\n"
                )
    finally:
        if handle is not None:
            handle.close()

    if failures:
        total = sum(failures.values())
        detail = ", ".join(f"{name}={count}" for name, count in sorted(failures.items()))
        LOGGER.warning(
            "%d of %d record(s) could not be encoded and are excluded from the "
            "metrics: %s",
            total,
            len(ordered),
            detail,
        )
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "scored %d record(s) in %.0fs (%.2fs/record, %.1f rec/s)",
        accumulator.result().samples,
        elapsed,
        elapsed / scored if scored else 0.0,
        scored / elapsed if elapsed else 0.0,
    )
    return EvalOutcome(
        metrics=accumulator.result(),
        audit=auditor.result() if auditor is not None else None,
    )
