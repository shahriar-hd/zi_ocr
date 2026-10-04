"""Dataset, sampler and collator over the unified manifest.

Three things here are shaped by the 4 GB budget rather than by convention:

*Nothing is cached.* A page decodes to ~6 MB and the processor's patch tensor is
another ~26 MB in float32. Two workers holding a handful of those is already a
gigabyte of the 16 GB the machine has, so images are opened, resized and dropped
inside :meth:`PersianOcrDataset.__getitem__` and never retained. The manifest -
18 short columns for 10,000 rows - is the only thing held in memory.

*A bad record must not end the run.* An unreadable PNG four hours into an
overnight run should cost one sample, not the run, so ``__getitem__`` advances to
the next record a bounded number of times and counts what it skipped. Silence
would be worse than the crash, so every skip is logged with its reason and the
totals are reported at the end (CLAUDE.md 5.1).

*Digit oversampling is an index expansion, not a sampler weight.* The manifest
gives digit-dense records ``weight: 2.0``; a ``WeightedRandomSampler`` would
implement that by sampling with replacement, which makes an "epoch" a random
draw and the loss curve incomparable between runs. Repeating the index instead
keeps the epoch an exact pass over a known multiset, so step counts, the LR
schedule and resume all stay deterministic.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import torch
from torch.utils.data import Dataset, Sampler

from ..dataset.common.io_utils import resolve_repo_path
from .config import TrainingConfig
from .encoding import IGNORE_INDEX, EncodedSample, SampleEncoder, SampleTooLong
from .manifest import load_manifest, parse_crop_box

__all__ = [
    "CollatedBatch",
    "DatasetStats",
    "DigitAwareLengthGroupedSampler",
    "PersianOcrDataset",
    "collate_samples",
]

LOGGER = logging.getLogger(__name__)

#: How many following records ``__getitem__`` may try before giving up. Small on
#: purpose: if a dozen consecutive records fail, the corpus is broken and the run
#: should say so rather than quietly training on a shifted index.
MAX_ADVANCE: Final[int] = 12


@dataclass
class DatasetStats:
    """Running tally of what the dataset did to the records it served.

    Not a debugging aid: it is what the end-of-run report uses to say how much
    supervision was actually delivered, so a run that silently truncated a third
    of its targets is visible rather than merely slow to converge.

    Attributes:
        served: Records returned successfully.
        shrunk: Records whose image budget was reduced to fit the window.
        truncated: Records whose target was cut to fit the window.
        failed: Reason-to-count map of records that could not be served.
    """

    served: int = 0
    shrunk: int = 0
    truncated: int = 0
    failed: dict[str, int] = field(default_factory=dict)

    def note(self, sample: EncodedSample) -> None:
        """Record one successfully served sample.

        Args:
            sample: The encoded sample about to be returned.
        """
        self.served += 1
        self.shrunk += int(sample.shrunk)
        self.truncated += int(sample.truncated)

    def fail(self, reason: str) -> None:
        """Record one record that could not be served.

        Args:
            reason: Short, groupable reason tag.
        """
        self.failed[reason] = self.failed.get(reason, 0) + 1

    def summary(self) -> str:
        """One-line human-readable summary.

        Returns:
            A string naming served, degraded and failed counts.
        """
        parts = [f"served {self.served}"]
        if self.shrunk:
            parts.append(f"shrunk {self.shrunk}")
        if self.truncated:
            parts.append(f"truncated {self.truncated}")
        if self.failed:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(self.failed.items()))
            parts.append(f"failed {sum(self.failed.values())} ({detail})")
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class CollatedBatch:
    """A collated batch, in the keyword form the model's ``forward`` expects.

    Attributes:
        input_ids: Token ids, shape ``(B, L)``, right-padded.
        attention_mask: 1 on real tokens, 0 on padding, shape ``(B, L)``.
        labels: Targets with padding and prompt masked to
            :data:`~fa_ocr_lora.training.encoding.IGNORE_INDEX`, shape ``(B, L)``.
        pixel_values: Patches of every image in the batch, concatenated along
            dim 0, shape ``(sum(P_i), 3, 14, 14)``.
        image_grid_thw: One row per image, shape ``(B, 3)``.
        mm_token_type_ids: Modality ids per token, shape ``(B, L)``, or ``None``
            where no sample carried them.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    mm_token_type_ids: torch.Tensor | None = None

    def to_kwargs(self) -> dict[str, torch.Tensor]:
        """Return the batch as model ``forward`` keyword arguments.

        The modality ids ride along only when a sample carried them: omitting
        the key keeps the older checkpoint's ``forward`` untouched, while
        transformers 5's native 1.6 raises if it receives the grid without them.

        Returns:
            A mapping ready to splat into the model call.
        """
        kwargs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
        }
        if self.mm_token_type_ids is not None:
            kwargs["mm_token_type_ids"] = self.mm_token_type_ids
        return kwargs


class PersianOcrDataset(Dataset[dict[str, torch.Tensor]]):
    """Serves encoded samples for one split of the unified manifest.

    Returns plain dictionaries rather than :class:`EncodedSample` because
    ``Trainer``'s dataloader plumbing and the default worker collation both
    expect mappings of tensors.

    Attributes:
        stats: Running tally, meaningful only in the main process (worker
            processes accumulate their own copies, which is why the totals are
            re-derived from the log rather than summed from here when
            ``dataloader_workers > 0``).
    """

    def __init__(
        self,
        config: TrainingConfig,
        processor: Any,
        *,
        split: str = "train",
        limit: int | None = None,
        oversample: bool = True,
    ) -> None:
        """Build a dataset over one split.

        Args:
            config: Loaded training configuration.
            processor: The model's ``AutoProcessor``.
            split: Manifest split to serve: ``train``, ``validation`` or
                ``test``.
            limit: Keep at most this many records, taken in manifest order.
                Used by ``--smoke`` and ``--limit``.
            oversample: Expand digit-dense records by their manifest weight.
                Always ``False`` for evaluation, where every record must be
                scored exactly once.

        Raises:
            FileNotFoundError: If the manifest has not been built.
            ValueError: If the split is empty, which almost always means the
                manifest was built with a different mix than the trainer expects.
        """
        frame = load_manifest(config.paths.manifest_path)
        frame = frame[frame["split"] == split].reset_index(drop=True)
        if limit is not None:
            frame = frame.iloc[:limit].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"split {split!r} of {config.paths.manifest_path} is empty; "
                "rebuild the manifest with fa_ocr_lora.training.prepare_manifest"
            )
        self._frame = frame
        self._encoder = SampleEncoder(processor, model=config.model, image=config.image)
        self._index = self._expand(frame, oversample=oversample)
        self.stats = DatasetStats()
        LOGGER.info(
            "%s split: %d records -> %d samples per epoch%s",
            split,
            len(frame),
            len(self._index),
            " (digit-dense records repeated)" if oversample else "",
        )

    @staticmethod
    def _expand(frame: Any, *, oversample: bool) -> tuple[int, ...]:
        """Build the sample index, repeating records by their weight.

        A weight of 2.5 means "twice always, a third time for half the records";
        the fractional part is resolved by the record's own row position rather
        than by an RNG, so the epoch is identical on every run and every worker
        without threading a seed through.

        Args:
            frame: The split's manifest rows.
            oversample: Whether to honour the weight column at all.

        Returns:
            Row positions to serve, in manifest order, with repeats adjacent.
        """
        if not oversample:
            return tuple(range(len(frame)))
        index: list[int] = []
        for position, weight in enumerate(frame["weight"].tolist()):
            repeats = int(weight)
            remainder = float(weight) - repeats
            # Deterministic tie-break on row position: no RNG, no per-worker
            # divergence, and the same epoch on a rerun.
            if remainder > 0 and (position % 100) < round(remainder * 100):
                repeats += 1
            index.extend([position] * max(1, repeats))
        return tuple(index)

    def __len__(self) -> int:
        """Samples per epoch, after weight expansion."""
        return len(self._index)

    def token_lengths(self) -> tuple[int, ...]:
        """Approximate sequence length of every sample, for length grouping.

        Uses the manifest's precomputed ``image_tokens`` plus a character-based
        estimate of the answer, which is enough to order samples by cost without
        tokenising 10,000 targets up front.

        Returns:
            One length per sample in this epoch, aligned with :meth:`__len__`.
        """
        tokens = self._frame["image_tokens"].tolist()
        chars = self._frame["text_chars"].tolist()
        # ~2.2 characters per token on Persian text in this tokenizer; only the
        # relative order matters here, so a coarse ratio is sufficient.
        return tuple(int(tokens[row] + chars[row] / 2.2) for row in self._index)

    def weights(self) -> tuple[float, ...]:
        """Manifest weight of every sample in this epoch.

        Returns:
            One weight per sample, aligned with :meth:`__len__`.
        """
        column = self._frame["weight"].tolist()
        return tuple(float(column[row]) for row in self._index)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        """Encode one sample, advancing past records that cannot be encoded.

        The manifest stores ``image_path`` relative to the repository root, so
        that one manifest is valid on any checkout. Resolving it here rather than
        treating it as relative to the process working directory is what lets the
        trainer be launched from anywhere.

        Args:
            position: Index into this epoch's sample list.

        Returns:
            The encoded sample as a mapping of tensors.

        Raises:
            RuntimeError: If :data:`MAX_ADVANCE` consecutive records fail, which
                means the corpus or the manifest is broken rather than one file
                being bad.
        """
        for offset in range(MAX_ADVANCE):
            row = self._frame.iloc[self._index[(position + offset) % len(self._index)]]
            try:
                sample = self._encoder.encode(
                    image_path=resolve_repo_path(row["image_path"]),
                    crop_box=parse_crop_box(row["crop_box"]),
                    text=row["text"],
                    record_id=row["record_id"],
                )
            except SampleTooLong:
                self.stats.fail("too_long")
            except (OSError, ValueError) as exc:
                # OSError: unreadable or truncated image. ValueError: a degenerate
                # region, or the processor disagreeing with our token prediction -
                # both are per-record faults, not run-ending ones.
                self.stats.fail(type(exc).__name__)
                LOGGER.warning("skipping %s: %s", row["record_id"], exc)
            else:
                self.stats.note(sample)
                record: dict[str, torch.Tensor] = {
                    "input_ids": sample.input_ids,
                    "labels": sample.labels,
                    "pixel_values": sample.pixel_values,
                    "image_grid_thw": sample.image_grid_thw,
                }
                # Present exactly when the encoder's processor returns it - absent
                # for older checkpoints, whose records stay byte-identical.
                if sample.mm_token_type_ids is not None:
                    record["mm_token_type_ids"] = sample.mm_token_type_ids
                return record
        raise RuntimeError(
            f"{MAX_ADVANCE} consecutive records failed to encode from position "
            f"{position}; the corpus or the manifest is unusable "
            f"({self.stats.summary()})"
        )


def collate_samples(
    features: Sequence[dict[str, torch.Tensor]], *, pad_token_id: int, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Pad a list of encoded samples into one batch.

    Padded positions get :data:`~fa_ocr_lora.training.encoding.IGNORE_INDEX` in
    the labels *and* a zero attention mask. Either alone is a silent correctness
    bug: an unmasked pad is attended to, and an unignored pad is trained on.

    ``pixel_values`` are concatenated rather than stacked because images in one
    batch have different patch counts; the vision tower splits them back apart
    using ``image_grid_thw``, which is why that tensor has one row per image.

    Args:
        features: Samples as returned by :class:`PersianOcrDataset`.
        pad_token_id: Token id used for padding.
        dtype: Compute dtype of the model, applied to ``pixel_values``. The
            processor emits float32; feeding that into a bf16 tower costs a
            silent upcast of the largest tensor in the batch.

    Returns:
        The batch as model ``forward`` keyword arguments.

    Raises:
        ValueError: If ``features`` is empty.
    """
    if not features:
        raise ValueError("cannot collate an empty batch")
    lengths = [int(item["input_ids"].shape[0]) for item in features]
    width = max(lengths)
    batch = len(features)
    reference = features[0]["input_ids"]

    input_ids = torch.full((batch, width), pad_token_id, dtype=reference.dtype)
    labels = torch.full((batch, width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((batch, width), dtype=torch.long)
    for row, item in enumerate(features):
        length = lengths[row]
        input_ids[row, :length] = item["input_ids"]
        labels[row, :length] = item["labels"]
        attention_mask[row, :length] = 1

    mm_token_type_ids = None
    if "mm_token_type_ids" in features[0]:
        # Padding with 0 (text): the model filters masked rows before the RoPE
        # pass, so what sits under the mask carries no position either way.
        first = features[0]["mm_token_type_ids"]
        mm_token_type_ids = torch.zeros((batch, width), dtype=first.dtype)
        for row, item in enumerate(features):
            mm_token_type_ids[row, : lengths[row]] = item["mm_token_type_ids"]

    collated = CollatedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pixel_values=torch.cat([item["pixel_values"] for item in features], dim=0).to(
            dtype
        ),
        image_grid_thw=torch.cat([item["image_grid_thw"] for item in features], dim=0),
        mm_token_type_ids=mm_token_type_ids,
    )
    return collated.to_kwargs()


class DigitAwareLengthGroupedSampler(Sampler[int]):
    """Orders an epoch so that similarly sized samples land in the same step.

    With ``per_device_batch_size: 1`` this changes no padding - there is nothing
    to pad against - but it still matters, for two reasons specific to this run:

    * Gradient accumulation is 16, so one optimizer step spans 16 samples. A step
      that mixes a 220-token crop with a 1,600-token page has a peak set by the
      page and an average nowhere near it; grouping makes peak VRAM per step
      predictable, which is what lets ``--probe`` produce a number worth trusting.
    * Digit-dense records are the ones the adapter exists to fix, so they are
      spread across the epoch rather than left adjacent to their own repeats,
      where a single step would see the same page twice.

    Grouping is applied inside shuffled megabatches rather than globally, because
    a globally sorted epoch would feed all crops first and all pages last, and
    the LR schedule would then correlate with sample difficulty.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        weights: Sequence[float],
        *,
        batch_size: int,
        megabatch_multiplier: int = 32,
        seed: int = 0,
    ) -> None:
        """Build the sampler.

        Args:
            lengths: Approximate token length of each sample.
            weights: Manifest weight of each sample; anything above 1 is treated
                as digit-dense and spread out.
            batch_size: Samples per accumulation slice, normally
                ``per_device_batch_size * gradient_accumulation_steps``.
            megabatch_multiplier: How many batches are sorted together. Larger
                groups pack better and shuffle less.
            seed: Base seed; the epoch number is mixed in by
                :meth:`set_epoch`.

        Raises:
            ValueError: If the two sequences disagree in length or the batch size
                is not positive.
        """
        if len(lengths) != len(weights):
            raise ValueError(
                f"lengths ({len(lengths)}) and weights ({len(weights)}) differ"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._lengths = tuple(lengths)
        self._weights = tuple(weights)
        self._megabatch = max(1, batch_size * megabatch_multiplier)
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Reseed for a new epoch.

        Args:
            epoch: Zero-based epoch number.
        """
        self._epoch = epoch

    def __len__(self) -> int:
        """Number of samples in one epoch."""
        return len(self._lengths)

    def __iter__(self) -> Iterator[int]:
        """Yield sample indices for one epoch.

        Returns:
            An iterator over every index exactly once.
        """
        rng = random.Random(f"{self._seed}:sampler:{self._epoch}")
        order = list(range(len(self._lengths)))
        rng.shuffle(order)

        # Interleave the repeated (digit-dense) samples through the epoch so a
        # single accumulation window never sees one record twice.
        dense = [i for i in order if self._weights[i] > 1.0]
        plain = [i for i in order if self._weights[i] <= 1.0]
        order = _interleave(plain, dense)

        for start in range(0, len(order), self._megabatch):
            window = order[start : start + self._megabatch]
            window.sort(key=lambda index: self._lengths[index], reverse=True)
            yield from window


def _interleave(major: list[int], minor: list[int]) -> list[int]:
    """Spread one list evenly through another.

    Args:
        major: The longer list, kept in its given order.
        minor: Items to distribute through ``major`` at even spacing.

    Returns:
        A single list containing every element of both.
    """
    if not minor:
        return list(major)
    if not major:
        return list(minor)
    stride = len(major) / len(minor)
    merged: list[int] = []
    taken = 0
    for position, item in enumerate(major):
        while taken < len(minor) and taken * stride <= position:
            merged.append(minor[taken])
            taken += 1
        merged.append(item)
    merged.extend(minor[taken:])
    return merged
