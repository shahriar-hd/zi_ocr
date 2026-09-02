"""The training loop: ``Trainer`` wiring, checkpointing, and the throughput probe.

Everything here exists because the stock :class:`~transformers.Trainer` makes four
assumptions this run breaks:

*The model is a* ``PreTrainedModel`` *or a* ``PeftModel``. Ours is a plain
:class:`~fa_ocr_lora.training.modeling.SlicedLossModel` wrapping the PEFT model, so
Trainer's PEFT detection misses and its default ``_save`` would write a 1.7 GB
checkpoint of frozen quantised base weights - three times over, given
``save_total_limit: 3``. :meth:`LoraTrainer._save` writes the 36 MB adapter instead,
and :meth:`LoraTrainer._load_from_checkpoint` reads it back.

*One learning rate fits every parameter.* It does not here: the embedding and output
head need a lower rate than the attention adapters (see
:func:`~fa_ocr_lora.training.modeling.parameter_groups`), and Trainer's
``create_optimizer`` would collapse both groups into one.

*Sampling order does not matter.* With ``gradient_accumulation_steps: 16`` a step
spans 16 samples, and mixing 220-token crops with 1,600-token pages makes peak VRAM
per step unpredictable - which would make ``--probe``'s measurement useless.

*Evaluation can keep every model output.* ``forward`` returns hidden states under the
key ``logits`` (the head is swapped out during the sliced loss), so Trainer would
accumulate 803 x 6 MB of them on the GPU during a validation pass. Hence
``prediction_loss_only=True``, which is not an optimisation but a requirement.

The other job of this module is to answer "how long will this take, and will it fit"
*before* a night is committed to it: :func:`run_training` with ``probe_steps`` runs a
handful of real steps and reports measured seconds per step, peak VRAM and a
projected epoch time.

On a rented GPU there is a third job: surviving the instance. :func:`run_log_file`
tees every log line to ``<output_dir>/logs/train.log`` so a killed terminal does not
take the run's history with it, and :class:`DriveSyncCallback` mirrors each checkpoint
to Google Drive in the background (see
:mod:`~fa_ocr_lora.training.checkpoint_sync`). Both are inert on a laptop, where
``drive.enabled`` is false and the disk outlives the process.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import Sampler
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

from ..dataset.common.io_utils import ensure_dir, write_json
from . import vram
from .checkpoint_sync import CheckpointSync
from .config import DriveConfig, TrainingConfig
from .data import (
    DigitAwareLengthGroupedSampler,
    PersianOcrDataset,
    collate_samples,
)
from .modeling import (
    LoadedModel,
    load_training_model,
    parameter_groups,
    resolve_optimizer,
    unwrap_base,
)

__all__ = [
    "DriveSyncCallback",
    "EpochSeedCallback",
    "LoraTrainer",
    "TrainingOutcome",
    "build_arguments",
    "build_trainer",
    "find_last_checkpoint",
    "run_log_file",
    "run_training",
]

LOGGER = logging.getLogger(__name__)

#: Adapter weights inside a checkpoint directory, as PEFT names them.
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"

#: Where :func:`run_training` records what it measured.
SUMMARY_NAME = "training_summary.json"


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """What a run actually did, in measured numbers rather than configured ones.

    Attributes:
        steps: Optimiser steps completed.
        epoch_samples: Samples in one full pass of the training split, after
            digit oversampling. The projection below is scaled from this, so a
            ``--smoke`` run reports a projection for its 16 samples and not for
            the corpus.
        seconds: Wall-clock spent inside ``train()``.
        train_loss: Mean training loss over the run.
        peak_mib: Highest allocated VRAM seen at any logging boundary.
        batch_size: Samples per optimiser step that produced these numbers.
            Recorded rather than re-read from the config so a summary file stays
            interpretable after the config is retuned.
        adapter_dir: Where the adapter was written, or ``None`` for a probe or
            smoke run, which deliberately saves nothing.
        resumed_from: Checkpoint the run continued from, if any.
    """

    steps: int
    epoch_samples: int
    seconds: float
    train_loss: float
    peak_mib: float
    batch_size: int = 1
    adapter_dir: Path | None = None
    resumed_from: Path | None = None

    @property
    def samples(self) -> int:
        """Samples consumed, counting each accumulation micro-batch."""
        return self.steps * self.batch_size

    @property
    def seconds_per_step(self) -> float:
        """Wall-clock per optimiser step."""
        return self.seconds / self.steps if self.steps else 0.0

    @property
    def seconds_per_sample(self) -> float:
        """Wall-clock per training sample."""
        return self.seconds / self.samples if self.samples else 0.0

    @property
    def hours_per_epoch(self) -> float:
        """Projected wall-clock for one pass over ``epoch_samples``."""
        return self.epoch_samples * self.seconds_per_sample / 3600.0

    def describe(self) -> str:
        """Format the outcome for the console.

        Returns:
            A multi-line summary, no trailing newline.
        """
        lines = [
            f"steps      {self.steps} in {self.seconds:.0f}s "
            f"({self.seconds_per_step:.2f}s/step, "
            f"{self.seconds_per_sample:.2f}s/sample)",
            f"loss       {self.train_loss:.4f}",
            f"peak VRAM  {self.peak_mib:.0f} MiB",
            f"projected  {self.hours_per_epoch:.2f} h per epoch "
            f"over {self.epoch_samples} samples",
        ]
        if self.resumed_from is not None:
            lines.append(f"resumed    {self.resumed_from}")
        if self.adapter_dir is not None:
            lines.append(f"adapter    {self.adapter_dir}")
        return "\n".join(lines)


@contextmanager
def run_log_file(output_dir: Path, *, level: int = logging.INFO) -> Iterator[Path]:
    """Tee the root logger to a file for the duration of a run.

    A rented instance is reached over SSH, and a dropped connection takes the
    console scrollback with it - including the throughput numbers and the VRAM
    peak that the run existed to measure. The file handler is attached to the
    *root* logger rather than this package's, so transformers' own warnings (the
    ones that explain a stalled dataloader or a skipped checkpoint) land in the
    same place.

    Opened in append mode: a resumed run continues its predecessor's log instead
    of truncating the history of the steps it is about to skip over.

    Args:
        output_dir: The run's output directory; the log goes in ``logs/`` under it.
        level: Minimum level to record.

    Yields:
        The path being written, so the caller can hand it to the Drive mirror.
    """
    path = ensure_dir(output_dir / "logs") / "train.log"
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(level)
    root = logging.getLogger()
    # A CLI that never called basicConfig leaves the root logger at WARNING, which
    # would drop every info line on the floor before the handler ever sees it.
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)
    root.addHandler(handler)
    LOGGER.info("run log -> %s", path)
    try:
        yield path
    finally:
        root.removeHandler(handler)
        handler.close()


class DriveSyncCallback(TrainerCallback):
    """Hands each freshly written checkpoint to the background Drive uploader.

    ``on_save`` is the one hook that covers both triggers: ``Trainer`` calls it
    after ``_save_checkpoint`` whether the save came from ``save_strategy:
    steps`` or ``save_strategy: epoch``, so there is no second code path for
    epoch boundaries.

    The callback does no I/O of its own - :meth:`CheckpointSync.enqueue` hard-links
    the directory and returns - so the cost to the training loop is a few
    microseconds per save rather than the minutes an upload takes.
    """

    def __init__(self, sync: CheckpointSync, checkpoint_root: Path) -> None:
        """Bind the callback to an already-started mirror.

        Args:
            sync: The running uploader.
            checkpoint_root: Trainer output directory holding ``checkpoint-*``.
        """
        self._sync = sync
        self._root = checkpoint_root

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Queue the checkpoint just written for this step.

        Args:
            args: Training arguments, unused.
            state: Trainer state, read for the global step.
            control: Flow control object, unused.
            **kwargs: Remaining Trainer callback arguments, unused.
        """
        directory = self._root / f"{PREFIX_CHECKPOINT_DIR}-{int(state.global_step)}"
        if not directory.is_dir():
            # Reached when Trainer renames or relocates a checkpoint; a missing
            # directory is worth a line in the log but is not worth failing over.
            LOGGER.warning("drive sync: no checkpoint at %s to upload", directory)
            return
        self._sync.enqueue(directory)


def build_checkpoint_sync(
    drive: DriveConfig,
    *,
    output_dir: Path,
    log_path: Path | None,
) -> CheckpointSync:
    """Construct (but do not start) the Drive mirror for a run.

    Args:
        drive: The run's ``drive`` section.
        output_dir: The run's output directory, which names the remote folder and
            holds the local staging area.
        log_path: Run log to mirror, or ``None`` to mirror checkpoints only.

    Returns:
        A configured, not-yet-authenticated mirror.
    """
    return CheckpointSync(
        run_name=drive.resolved_run_name(output_dir),
        output_dir=output_dir,
        root_folder=drive.folder,
        root_folder_id=drive.folder_id or None,
        service_account=drive.service_account_path,
        token=drive.token_path,
        client_secret=drive.client_secret_path,
        interactive=drive.interactive_auth,
        workers=drive.workers,
        upload_optimizer_state=drive.upload_optimizer_state,
        keep_remote=drive.keep_remote,
        log_path=log_path if drive.upload_logs else None,
    )


class EpochSeedCallback(TrainerCallback):
    """Reseeds the length-grouped sampler at the start of every epoch.

    Accelerate's dataloader wrapper usually forwards ``set_epoch`` to the sampler,
    but only when the batch sampler nests exactly the way it expects. If that
    forwarding silently stops, every epoch replays the identical order - which
    looks like nothing at all in the logs and quietly halves the value of the
    second epoch. Calling it here as well is idempotent and removes the
    dependency on that internal.
    """

    def __init__(self, sampler: DigitAwareLengthGroupedSampler) -> None:
        """Bind the callback to a sampler.

        Args:
            sampler: The sampler whose epoch should be advanced.
        """
        self._sampler = sampler

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Advance the sampler's epoch.

        Args:
            args: Training arguments, unused.
            state: Trainer state, read for the epoch number.
            control: Flow control object, unused.
            **kwargs: Remaining Trainer callback arguments, unused.
        """
        self._sampler.set_epoch(int(state.epoch or 0))


class LoraTrainer(Trainer):
    """``Trainer`` specialised for a LoRA adapter behind a loss wrapper.

    Attributes:
        peak_mib: Highest allocated VRAM observed at a logging boundary, in MiB.
    """

    def __init__(self, *, optim_config: Any, **kwargs: Any) -> None:
        """Build the trainer.

        Args:
            optim_config: The run's ``optim`` section, needed for the
                differential learning rates that :meth:`create_optimizer` applies.
            **kwargs: Forwarded to :class:`~transformers.Trainer`.
        """
        self._optim = optim_config
        self.peak_mib = 0.0
        super().__init__(**kwargs)
        # forward() takes **inputs, so Trainer infers that it accepts loss kwargs
        # and switches to the sum-of-tokens normalisation, passing
        # num_items_in_batch straight through to the base decoder. Our loss is
        # already a mean over supervised tokens, so the classic path - divide by
        # gradient_accumulation_steps - is the correct one, and the inference is
        # simply wrong about this model.
        self.model_accepts_loss_kwargs = False

    def _get_train_sampler(self, *args: Any, **kwargs: Any) -> Sampler[int]:
        """Return the digit-aware length-grouped sampler.

        Args:
            *args: Accepted and ignored; the argument list of this hook changed
                across transformers 4.x and none of its values are needed here.
            **kwargs: Likewise ignored.

        Returns:
            The sampler over the training split.
        """
        dataset = self.train_dataset
        assert isinstance(dataset, PersianOcrDataset)  # noqa: S101 - internal wiring
        return DigitAwareLengthGroupedSampler(
            dataset.token_lengths(),
            dataset.weights(),
            batch_size=self._optim.effective_batch_size,
            seed=self.args.seed,
        )

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimiser over differential learning-rate groups.

        Returns:
            The optimiser, also stored on ``self.optimizer``.
        """
        if self.optimizer is None:
            groups = parameter_groups(self.model, self._optim)
            # Signature is (args, model) on transformers 4.57, the version this
            # config pins; it resolves optim strings including the bitsandbytes
            # paged variants, which is why it is reused rather than reimplemented.
            cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
            kwargs.pop("params", None)
            self.optimizer = cls(groups, **kwargs)
            LOGGER.info(
                "optimizer %s over %d group(s): %s",
                cls.__name__,
                len(groups),
                ", ".join(f"{group['name']}={group['lr']:.1e}" for group in groups),
            )
        return self.optimizer

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        """Attach peak VRAM to every logged step.

        Injected here rather than from a callback so the figure reaches
        ``state.log_history`` and the printed line alike: callbacks run after the
        history entry has been copied, so a callback mutating ``logs`` would
        change neither.

        The peak is reset after each read, making the reported number the peak
        *within* the logging window instead of a run-long high-water mark that
        stops being informative after step 10.

        Args:
            logs: Metrics for this step, extended in place.
            *args: Forwarded to the base implementation.
            **kwargs: Forwarded to the base implementation.
        """
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 2**20
            logs["vram_mib"] = round(peak, 1)
            self.peak_mib = max(self.peak_mib, peak)
            torch.cuda.reset_peak_memory_stats()
        super().log(logs, *args, **kwargs)

    def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:
        """Write the LoRA adapter, not the base model.

        ``state_dict`` is ignored on purpose: it is the wrapper's full state,
        frozen quantised base weights included, and PEFT already knows how to
        extract exactly the trainable tensors.

        No ``training_args.bin`` is written. Resume does not read it, and the
        run's hyperparameters live in the version-controlled YAML this trainer
        was built from, which is a better record than a pickled dataclass.

        ``save_embedding_layers=False`` is not an optimisation. PEFT's default
        ``"auto"`` sees ``embed_tokens`` and ``lm_head`` among the LoRA targets,
        assumes the vocabulary was resized, and writes both full base matrices -
        404 MiB of frozen weights bit-identical to the base checkpoint, turning a
        36 MiB adapter into a 461 MiB one. This run never resizes the vocabulary,
        so those tensors are recoverable from the base model by definition; the
        trained ``lora_embedding_A``/``lora_embedding_B`` factors are saved either
        way.

        Args:
            output_dir: Destination; defaults to the arguments' output directory.
            state_dict: Ignored, see above.
        """
        target = ensure_dir(Path(output_dir or self.args.output_dir))
        unwrap_base(self.model).save_pretrained(str(target), save_embedding_layers=False)
        LOGGER.debug("wrote adapter to %s", target)

    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: Any = None
    ) -> None:
        """Load adapter weights from a checkpoint directory.

        Args:
            resume_from_checkpoint: Directory written by :meth:`_save`.
            model: Model to load into; defaults to the trainer's own.

        Raises:
            FileNotFoundError: If the directory holds no adapter weights.
            RuntimeError: If the checkpoint contains adapter tensors this model
                has no home for, which means it was trained with a different LoRA
                shape and silently loading the subset that does match would
                produce an adapter that is part-trained and part-random.
        """
        weights = Path(resume_from_checkpoint) / ADAPTER_WEIGHTS_NAME
        if not weights.is_file():
            raise FileNotFoundError(
                f"{weights} not found; checkpoint {resume_from_checkpoint} was "
                f"not written by this trainer"
            )
        target = unwrap_base(model if model is not None else self.model)
        result = set_peft_model_state_dict(target, load_file(str(weights)))
        unexpected = tuple(getattr(result, "unexpected_keys", ()) or ())
        if unexpected:
            raise RuntimeError(
                f"{weights} has {len(unexpected)} adapter tensor(s) not present "
                f"in this model, first {unexpected[0]!r}; the checkpoint's LoRA "
                f"config differs from configs/lora_config.yaml"
            )
        LOGGER.info("resumed adapter weights from %s", resume_from_checkpoint)


def find_last_checkpoint(output_dir: Path) -> Path | None:
    """Locate the most recent checkpoint under a run directory.

    Auto-resume is not a convenience here. An epoch over 9,197 samples takes
    hours on this hardware, so a closed lid or an OOM caused by a browser tab
    must cost at most ``save_steps`` worth of work rather than the whole run.

    Args:
        output_dir: The trainer's checkpoint root.

    Returns:
        The checkpoint directory, or ``None`` when there is nothing to resume.
    """
    if not output_dir.is_dir():
        return None
    found = get_last_checkpoint(str(output_dir))
    return Path(found) if found else None


def build_arguments(
    config: TrainingConfig,
    *,
    dtype: torch.dtype,
    max_steps: int = -1,
    evaluate: bool = True,
    save: bool = True,
) -> TrainingArguments:
    """Translate the config into ``TrainingArguments``.

    Args:
        config: Loaded training configuration.
        dtype: Compute dtype chosen by
            :func:`~fa_ocr_lora.training.modeling.resolve_dtype`, which decides
            which mixed-precision flag is set.
        max_steps: Hard step cap; ``-1`` runs the configured epochs. Set by
            ``--smoke`` and ``--probe``.
        evaluate: Whether to run validation passes during training.
        save: Whether to write checkpoints. Off for probe and smoke runs, which
            should not leave a half-trained adapter behind.

    Returns:
        The arguments object.
    """
    optim, runtime = config.optim, config.runtime
    return TrainingArguments(
        output_dir=str(config.paths.output_dir / "checkpoints"),
        per_device_train_batch_size=optim.per_device_batch_size,
        per_device_eval_batch_size=optim.per_device_batch_size,
        gradient_accumulation_steps=optim.gradient_accumulation_steps,
        learning_rate=optim.learning_rate,
        weight_decay=optim.weight_decay,
        max_grad_norm=optim.max_grad_norm,
        warmup_ratio=optim.warmup_ratio,
        lr_scheduler_type=optim.lr_scheduler_type,
        num_train_epochs=optim.num_epochs,
        max_steps=max_steps,
        optim=resolve_optimizer(optim),
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        # False even though optim.gradient_checkpointing is true: checkpointing is
        # already enabled inside load_training_model, on the decoder only and with
        # use_reentrant=False. Letting Trainer enable it again would apply it to
        # the frozen vision tower as well, where it costs recompute and saves
        # nothing, and would do so through the reentrant implementation, which
        # returns no gradients when its inputs do not require grad - exactly this
        # model's situation.
        gradient_checkpointing=False,
        dataloader_num_workers=runtime.dataloader_workers,
        dataloader_pin_memory=True,
        # Not persistent: 16 GB of system RAM is the tighter constraint than the
        # once-per-epoch respawn cost of two workers.
        dataloader_persistent_workers=False,
        logging_steps=runtime.logging_steps,
        logging_first_step=True,
        # NaN and inf are the signal that a step diverged. Filtering them out of
        # the log leaves a run that looks healthy while learning nothing.
        logging_nan_inf_filter=False,
        save_strategy="steps" if save else "no",
        save_steps=runtime.save_steps,
        save_total_limit=runtime.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps" if evaluate else "no",
        eval_steps=runtime.eval_steps,
        # Mandatory, not tuning: forward() returns decoder hidden states under the
        # key "logits" because the output head is swapped out while the sliced
        # loss runs. Without this, Trainer accumulates one (L, 1024) tensor per
        # validation record on the GPU and the first eval pass OOMs.
        prediction_loss_only=True,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        # Our own sampler does the grouping, over precomputed manifest lengths
        # rather than by tokenising every target up front.
        group_by_length=False,
    )


def build_trainer(
    config: TrainingConfig,
    loaded: LoadedModel,
    *,
    smoke: bool = False,
    probe_steps: int = 0,
    limit: int | None = None,
) -> LoraTrainer:
    """Assemble datasets, arguments and trainer.

    Args:
        config: Loaded training configuration.
        loaded: Model and processor from
            :func:`~fa_ocr_lora.training.modeling.load_training_model`.
        smoke: Restrict to ``runtime.smoke_samples`` records and
            ``runtime.smoke_steps`` steps, to prove the wiring end to end.
        probe_steps: Stop after this many steps over the full dataset, to measure
            throughput. Ignored when zero.
        limit: Cap on training records, independent of ``smoke``.

    Returns:
        The configured trainer.
    """
    short_run = smoke or probe_steps > 0
    train_limit = config.runtime.smoke_samples if smoke else limit
    max_steps = config.runtime.smoke_steps if smoke else (probe_steps or -1)

    train_dataset = PersianOcrDataset(
        config, loaded.processor, split="train", limit=train_limit
    )
    eval_dataset = None
    if not short_run:
        # Loss only, on a subset. Character error rate needs generation, which is
        # ~30x the cost of a forward pass and belongs in eval.py; what the loop
        # needs from validation is a divergence signal, and loss gives that.
        eval_dataset = PersianOcrDataset(
            config,
            loaded.processor,
            split="validation",
            limit=config.eval.max_samples,
            oversample=False,
        )

    pad_id = loaded.processor.tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError(
            "the processor's tokenizer has no pad_token_id; collation cannot "
            "know which positions to mask"
        )

    arguments = build_arguments(
        config,
        dtype=loaded.dtype,
        max_steps=max_steps,
        evaluate=eval_dataset is not None,
        save=not short_run,
    )
    trainer = LoraTrainer(
        optim_config=config.optim,
        model=loaded.model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=loaded.processor,
        data_collator=lambda features: collate_samples(
            features, pad_token_id=int(pad_id), dtype=loaded.dtype
        ),
    )
    sampler = trainer._get_train_sampler()  # noqa: SLF001 - to bind the callback
    if isinstance(sampler, DigitAwareLengthGroupedSampler):
        trainer.add_callback(EpochSeedCallback(sampler))
    return trainer


def run_training(
    config: TrainingConfig,
    *,
    smoke: bool = False,
    probe_steps: int = 0,
    limit: int | None = None,
) -> TrainingOutcome:
    """Probe VRAM, load the model, train, and save the adapter.

    The VRAM probe runs before the weights are loaded, so a configuration that
    cannot fit says so in one second rather than after a 1.8 GB download and a
    two-minute load. When ``image.autoprobe_vram`` is on and the projection
    exceeds what is free, the image budget is clamped for this run only - the
    config file is not rewritten.

    Args:
        config: Loaded training configuration.
        smoke: Run ``runtime.smoke_steps`` steps over ``runtime.smoke_samples``
            records and save nothing.
        probe_steps: Run this many steps over the full split, report measured
            throughput, and save nothing. Ignored when zero.
        limit: Cap on training records.

    Returns:
        What the run measured.
    """
    with run_log_file(config.paths.output_dir) as log_path:
        return _run_training(
            config,
            log_path=log_path,
            smoke=smoke,
            probe_steps=probe_steps,
            limit=limit,
        )


def _run_training(
    config: TrainingConfig,
    *,
    log_path: Path,
    smoke: bool,
    probe_steps: int,
    limit: int | None,
) -> TrainingOutcome:
    """Body of :func:`run_training`, run with the file log already attached.

    Split out only so the ``with`` block does not reindent the whole function;
    every argument means what it does there, plus:

    Args:
        config: Loaded training configuration.
        log_path: The run log, mirrored to Drive alongside each checkpoint.
        smoke: See :func:`run_training`.
        probe_steps: See :func:`run_training`.
        limit: See :func:`run_training`.

    Returns:
        What the run measured.
    """
    report = vram.probe_vram(config.model, config.image)
    LOGGER.info("%s", vram.banner(report, config.model))
    if report.clamped:
        config = replace(
            config, image=replace(config.image, max_pixels=report.max_pixels)
        )

    short_run = smoke or probe_steps > 0

    # Authenticated here, before the 1.8 GB of weights load, because a bad
    # credential must end the run in its first second. Deferring it would turn a
    # typo in a service-account path into six hours of training whose only
    # surviving copy dies with the instance - discovered as an empty Drive folder.
    # Short runs skip it: nothing they produce is worth keeping.
    sync: CheckpointSync | None = None
    if config.drive.enabled and not short_run:
        sync = build_checkpoint_sync(
            config.drive, output_dir=config.paths.output_dir, log_path=log_path
        )
        sync.start()
    try:
        return _train(
            config,
            sync=sync,
            short_run=short_run,
            smoke=smoke,
            probe_steps=probe_steps,
            limit=limit,
        )
    finally:
        # Reached on the happy path too, where it is a no-op: _train drains the
        # queue itself so the summary can report what landed. This is here for
        # the paths that do not get that far - an OOM, a Ctrl-C - where the
        # checkpoints already staged are exactly the ones worth waiting for.
        if sync is not None:
            sync.close(config.drive.drain_timeout_s)


def _train(
    config: TrainingConfig,
    *,
    sync: CheckpointSync | None,
    short_run: bool,
    smoke: bool,
    probe_steps: int,
    limit: int | None,
) -> TrainingOutcome:
    """Load the model, run the loop, save the adapter, write the summary.

    Args:
        config: Loaded training configuration, already VRAM-clamped.
        sync: A started Drive mirror, or ``None`` when mirroring is off.
        short_run: Whether this is a smoke or probe run, which saves nothing.
        smoke: See :func:`run_training`.
        probe_steps: See :func:`run_training`.
        limit: See :func:`run_training`.

    Returns:
        What the run measured.
    """
    loaded = load_training_model(config)
    LOGGER.info("%s", loaded.describe())

    trainer = build_trainer(
        config, loaded, smoke=smoke, probe_steps=probe_steps, limit=limit
    )
    checkpoint_root = Path(trainer.args.output_dir)

    resume: Path | None = None
    if config.runtime.resume_from_checkpoint and not short_run:
        # Local first: a checkpoint on this disk is at least as new as the remote
        # copy and costs nothing to read. Drive is the fallback for the case the
        # mirror exists for - a fresh instance whose disk starts empty.
        resume = find_last_checkpoint(checkpoint_root)
        if resume is None and sync is not None and config.drive.resume_from_drive:
            resume = sync.fetch_resume(ensure_dir(checkpoint_root))
            if resume is not None:
                LOGGER.info("resuming from Drive checkpoint %s", resume.name)

    if sync is not None:
        trainer.add_callback(DriveSyncCallback(sync, checkpoint_root))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    elapsed = time.perf_counter() - started

    adapter_dir: Path | None = None
    if not short_run:
        adapter_dir = ensure_dir(config.paths.adapter_dir)
        # See PeftTrainer._save for why the embedding layers are excluded.
        unwrap_base(trainer.model).save_pretrained(
            str(adapter_dir), save_embedding_layers=False
        )
        loaded.processor.save_pretrained(str(adapter_dir))
        LOGGER.info("adapter written to %s", adapter_dir)
        if sync is not None:
            # publish=False: this is the deliverable, not a resume point. Naming
            # it in the LATEST marker would send the next resume at a directory
            # with no trainer state and break a restart that would have worked.
            sync.enqueue(adapter_dir, publish=False)

    outcome = TrainingOutcome(
        steps=int(trainer.state.global_step),
        epoch_samples=len(trainer.train_dataset),
        seconds=elapsed,
        train_loss=float(result.metrics.get("train_loss", float("nan"))),
        # max() because a run shorter than one logging window never reaches log()
        # and would otherwise report a peak of zero.
        peak_mib=max(
            trainer.peak_mib,
            (
                torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available()
                else 0.0
            ),
        ),
        adapter_dir=adapter_dir,
        resumed_from=resume,
        batch_size=config.optim.effective_batch_size,
    )
    LOGGER.info("%s", outcome.describe())

    if sync is not None:
        # Drained before the summary so its counters are final rather than a
        # snapshot of a still-running upload. The caller's finally block calls
        # this again, where it returns immediately.
        sync.close(config.drive.drain_timeout_s)

    if not short_run:
        write_json(
            config.paths.output_dir / SUMMARY_NAME,
            {
                "steps": outcome.steps,
                # Written so a consumer can recover the sample count the timings
                # are per: `steps * batch_size`, not `epoch_samples`, which is one
                # pass and not the whole run. training/simulate.py calibrates
                # against these three and would be off by the epoch count without
                # this field.
                "batch_size": outcome.batch_size,
                "epoch_samples": outcome.epoch_samples,
                "samples_consumed": outcome.samples,
                "seconds": round(outcome.seconds, 1),
                "seconds_per_step": round(outcome.seconds_per_step, 3),
                "seconds_per_sample": round(outcome.seconds_per_sample, 3),
                "hours_per_epoch": round(outcome.hours_per_epoch, 3),
                "train_loss": outcome.train_loss,
                "peak_vram_mib": round(outcome.peak_mib, 1),
                "resumed_from": str(resume) if resume else None,
                "adapter_dir": str(adapter_dir) if adapter_dir else None,
                "drive": sync.stats.as_dict() if sync is not None else None,
            },
        )
    return outcome
