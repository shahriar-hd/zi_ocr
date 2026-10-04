r"""Read a document with the trained adapter, from Python or the command line.

What this returns, and what it does not
--------------------------------------

The adapter is trained on transcription: a page image in, the page's text in
reading order out. So a reading is text plus what can be *derived* from that text -
the sensitive fields and their verdicts - and nothing else. It carries no element
bounding boxes, because the model was never asked to predict any, and emitting a
``DocumentAnnotation`` from :mod:`fa_ocr_lora.dataset.common.schema` would mean
inventing the ``bbox``, ``seed``, ``dpi`` and ``page_size`` that schema requires. A
fabricated box is worse than an absent one: a consumer cannot tell it apart from a
measured one. Hence :data:`READING_SCHEMA_VERSION`, a small schema of its own.

Layout *is* reachable, from the other half of the training mix: the adapter also
saw region crops, so ``--region`` reads one box the caller already knows. That is
transcription of a known region, not detection of an unknown one.

Why the field audit is on by default
------------------------------------

On an untranscribed document there is no CER to compute and no reference to compare
against, so the only signal about correctness is the one
:mod:`fa_ocr_lora.validation` produces without a reference: a ``شماره ثبت`` with
nothing numeric after it, a تاریخ that never existed, digits that repeat the way a
stuck decoder repeats them, and the model's own certainty on each field. Those
verdicts travel in the JSON and are logged as warnings, so a wrong field is
traceable rather than merely wrong.

Examples:
    One page, transcription on stdout::

        fa-ocr-lora predict --image scan.png

    Full JSON, with the field audit::

        fa-ocr-lora predict --image scan.png --json

    Read one known region - the metadata block, say - instead of the page::

        fa-ocr-lora predict --image scan.png --region 820 90 1180 210

    Compare against the untrained base model::

        fa-ocr-lora predict --image scan.png --no-adapter
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from PIL import Image

from ..config.base import ConfigError
from ..dataset.common.io_utils import relative_to_repo, resolve_repo_path, write_json
from ..training.config import TrainingConfig, load_training_config
from ..training.encoding import SampleTooLong
from ..validation import (
    FieldAuditor,
    FieldFinding,
    LazySpanConfidence,
    extract_fields,
    summarize_fields,
)
from ..validation.recorder import processor_list, recorder_for
from .loader import DocumentReader, load_reader, resolve_adapter

__all__ = [
    "READING_SCHEMA_VERSION",
    "PageReading",
    "main",
    "parse_args",
    "read_document",
]

LOGGER = logging.getLogger("fa_ocr_lora.predict")

#: Version of the reading record this module emits. Distinct from
#: ``schema.SCHEMA_VERSION``, which versions the generation-time annotation with
#: its element boxes - see the module docstring for why the two cannot be one.
READING_SCHEMA_VERSION: Final[str] = "1.0"


@dataclass(frozen=True, slots=True)
class PageReading:
    """What the model read from one document, and how sure it was.

    Attributes:
        image_path: The document that was read.
        region: The region read, or ``None`` for the whole page.
        source_size: ``(width, height)`` of the page before resizing, in the pixel
            units of the file, so a caller's own coordinates still apply.
        fed_size: ``(width, height)`` as the model saw it, after the pixel budget.
        image_tokens: Sequence positions the image occupied.
        prompt_tokens: Total prompt length, image included.
        text: The transcription, in reading order.
        fields: One row per sensitive field found, each with its verdicts and,
            under greedy decoding, the model's confidence in the value.
        findings: Reference-free faults, the traceability signal for a document
            nobody has transcribed.
        generated_tokens: How many tokens the model produced.
        truncated: Whether generation stopped at ``max_new_tokens`` rather than at
            an end-of-sequence token, which means the page is cut short. Surfaced
            because a silently truncated transcription looks like a complete one.
        seconds: Wall-clock time for the encode and the decode.
    """

    image_path: Path
    region: tuple[int, int, int, int] | None
    source_size: tuple[int, int]
    fed_size: tuple[int, int]
    image_tokens: int
    prompt_tokens: int
    text: str
    fields: tuple[dict[str, Any], ...] = ()
    findings: tuple[FieldFinding, ...] = ()
    generated_tokens: int = 0
    truncated: bool = False
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a JSON-serialisable mapping.

        Returns:
            The reading record. The path is repo-relative when the document lies
            inside the repository, so a record generated here is portable.
        """
        return {
            "image_path": relative_to_repo(self.image_path),
            "region": list(self.region) if self.region is not None else None,
            "source_size": list(self.source_size),
            "fed_size": list(self.fed_size),
            "image_tokens": self.image_tokens,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "truncated": self.truncated,
            "seconds": round(self.seconds, 2),
            "text": self.text,
            "fields": list(self.fields),
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def describe(self) -> str:
        """One-line summary for a log.

        Returns:
            Size, token counts, timing, and how many fields and faults were found.
        """
        faults = f", {len(self.findings)} fault(s)" if self.findings else ""
        cut = ", TRUNCATED" if self.truncated else ""
        return (
            f"{self.image_path.name}: {len(self.text)} chars, "
            f"{self.generated_tokens} tokens, {self.image_tokens} image tokens, "
            f"{len(self.fields)} field(s){faults}{cut}, {self.seconds:.1f}s"
        )


def _clip_region(
    image_path: Path, region: tuple[int, int, int, int] | None
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int]]:
    """Clip a requested region to the image, and report the image's size.

    ``Image.crop`` pads out-of-bounds requests with black rather than failing, so
    an off-by-a-thousand region would be read as a black rectangle and transcribed
    as nothing at all - a wrong answer that looks like a confident one. Clipping
    first turns that into either a smaller region or a named error.

    Args:
        image_path: Document to measure. Only the header is read; the pixels are
            not decoded here.
        region: ``(left, top, right, bottom)``, or ``None`` for the whole page.

    Returns:
        ``(clipped_region, (width, height))``, with the region ``None`` when it
        covers the whole page after clipping.

    Raises:
        OSError: If the image cannot be opened.
        ValueError: If the region is inverted, or lies entirely off the page.
    """
    with Image.open(image_path) as handle:
        width, height = handle.size
    if region is None:
        return None, (width, height)
    left, top, right, bottom = region
    if right <= left or bottom <= top:
        raise ValueError(
            f"region {list(region)} is empty; expected left < right and top < bottom"
        )
    clipped = (
        max(0, min(left, width)),
        max(0, min(top, height)),
        max(0, min(right, width)),
        max(0, min(bottom, height)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"region {list(region)} lies outside the {width}x{height} page")
    if clipped != region:
        LOGGER.warning(
            "region %s clipped to %s by the %dx%d page",
            list(region),
            list(clipped),
            width,
            height,
        )
    if clipped == (0, 0, width, height):
        return None, (width, height)
    return clipped, (width, height)


def _field_rows(
    text: str,
    reader: DocumentReader,
    lookup: LazySpanConfidence | None,
) -> tuple[dict[str, Any], ...]:
    """Render the sensitive fields of one reading, with confidence attached.

    The confidence block is what makes a field reviewable in production: a
    ``شماره ثبت`` whose weakest token the model gave a 0.2 probability is worth a
    human's attention even though nothing about it is provably wrong.

    Args:
        text: The transcription.
        reader: The loaded reader, supplying the field rules.
        lookup: Span-to-confidence resolver, or ``None`` when uncertainty was not
            recorded.

    Returns:
        One mapping per field, in reading order. Empty when the audit is off.
    """
    if reader.rules is None:
        return ()
    fields = extract_fields(text, reader.rules)
    rows: list[dict[str, Any]] = []
    for row, item in zip(summarize_fields(fields), fields, strict=True):
        certainty = lookup(item.start, item.end) if lookup and item.value else None
        rows.append({**row, **({"confidence": certainty.to_dict()} if certainty else {})})
    return tuple(rows)


def read_document(
    reader: DocumentReader,
    image_path: Path,
    *,
    region: tuple[int, int, int, int] | None = None,
    record_id: str = "",
) -> PageReading:
    """Read one document.

    Generation mirrors :func:`fa_ocr_lora.training.eval.score_split` exactly - same
    encoder, same prompt, same resolution, same greedy decode - so a reading
    produced here is the same operation the published character error rate
    measured. One document at a time, for the reason evaluation scores one at a
    time: a 1,024-token KV cache is most of what a 4 GB card has spare.

    Args:
        reader: A loaded reader from
            :func:`~fa_ocr_lora.inference.loader.load_reader`.
        image_path: Document image on disk.
        region: ``(left, top, right, bottom)`` to read, or ``None`` for the page.
        record_id: Identifier used to attribute findings; defaults to the filename.

    Returns:
        The reading.

    Raises:
        SampleTooLong: If the page cannot fit the sequence budget alongside the
            requested ``max_new_tokens``.
        OSError: If the image cannot be read.
        ValueError: If the region is empty or off the page, or if the processor's
            image expansion disagrees with the predicted token count.
        torch.OutOfMemoryError: If generation exhausts device memory.
    """
    config = reader.config
    clipped, source_size = _clip_region(image_path, region)
    started = time.perf_counter()
    prompt = reader.encoder.encode_prompt(
        image_path=image_path,
        crop_box=clipped,
        reserve_tokens=config.eval.max_new_tokens,
    )
    tokenizer = reader.processor.tokenizer
    device = reader.device
    cut = len(prompt)
    # Uncertainty is only sound under greedy decoding, where the emitted token is
    # the argmax the recorder sees - the same condition evaluation applies.
    recorder = recorder_for(config.eval.num_beams) if reader.rules else None
    extra: dict[str, torch.Tensor] = {}
    if prompt.mm_token_type_ids is not None:
        # Same contract as scoring: required by the native 1.6 model, ignored
        # by the older checkpoint.
        extra["mm_token_type_ids"] = prompt.mm_token_type_ids.unsqueeze(0).to(device)
    with torch.inference_mode():
        generated = reader.model.generate(
            input_ids=prompt.input_ids.unsqueeze(0).to(device),
            attention_mask=torch.ones((1, cut), dtype=torch.long, device=device),
            pixel_values=prompt.pixel_values.to(device, reader.dtype),
            image_grid_thw=prompt.image_grid_thw.to(device),
            **extra,
            max_new_tokens=config.eval.max_new_tokens,
            num_beams=config.eval.num_beams,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=processor_list(recorder),
        )
    new_tokens = generated[0][cut:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    elapsed = time.perf_counter() - started

    lookup = (
        LazySpanConfidence(
            tokenizer=tokenizer,
            token_ids=new_tokens.tolist(),
            steps=tuple(recorder.steps),
        )
        if recorder is not None
        else None
    )
    findings: tuple[FieldFinding, ...] = ()
    if reader.rules is not None:
        auditor = FieldAuditor(
            config=reader.rules,
            confidence_floor=config.eval.confidence_floor,
            max_findings=config.eval.max_findings,
        )
        findings = auditor.inspect(record_id or image_path.name, text, confidence=lookup)
    return PageReading(
        image_path=image_path,
        region=clipped,
        source_size=source_size,
        fed_size=(prompt.width, prompt.height),
        image_tokens=prompt.image_tokens,
        prompt_tokens=cut,
        text=text,
        fields=_field_rows(text, reader, lookup),
        findings=findings,
        generated_tokens=int(new_tokens.numel()),
        truncated=bool(new_tokens.numel())
        and int(new_tokens[-1]) != tokenizer.eos_token_id,
        seconds=elapsed,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        The parsed namespace, always carrying ``command == "predict"``.
    """
    parser = argparse.ArgumentParser(
        prog="fa-ocr-lora",
        description="Read Persian documents with the PaddleOCR-VL LoRA adapter.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser(
        "predict",
        help="Transcribe one or more document images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    predict.add_argument(
        "--image",
        type=Path,
        nargs="+",
        required=True,
        help="Document image(s) to read.",
    )
    predict.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training YAML; defaults to configs/lora_config.yaml.",
    )
    adapter = predict.add_mutually_exclusive_group()
    adapter.add_argument(
        "--no-adapter",
        action="store_true",
        help="Read with the untrained base model, to see what the adapter adds.",
    )
    adapter.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Adapter to apply; defaults to <output_dir>/adapter_only.",
    )
    predict.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=None,
        help="Read one known region instead of the whole page. One image only.",
    )
    predict.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Generation cap; overrides eval.max_new_tokens.",
    )
    predict.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the full JSON here, in addition to whatever goes to stdout.",
    )
    predict.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON to stdout instead of the transcription alone.",
    )
    predict.add_argument(
        "--field-config",
        type=Path,
        default=None,
        help="Sensitive-field rules; overrides eval.field_config_path.",
    )
    predict.add_argument(
        "--no-field-audit",
        action="store_true",
        help="Return the transcription with no sensitive-field verdicts.",
    )
    predict.add_argument(
        "--confidence-floor",
        type=float,
        default=None,
        help=(
            "Weakest-token probability below which a field is flagged as "
            "uncertain; overrides eval.confidence_floor."
        ),
    )
    predict.add_argument(
        "--verbose",
        action="store_true",
        help="Log the transcription and every field as it is read.",
    )
    return parser.parse_args(argv)


def _apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Fold command-line overrides into the loaded configuration.

    Round-tripped through the loader rather than mutated in place, so a bad flag is
    rejected by the same validation the YAML goes through.

    Args:
        config: Configuration as loaded from YAML.
        args: Parsed CLI arguments.

    Returns:
        The configuration with overrides applied, or the input unchanged when
        there were none.

    Raises:
        ConfigError: If an override produces an invalid configuration.
    """
    overrides: dict[str, object] = {}
    if args.max_new_tokens is not None:
        overrides["max_new_tokens"] = args.max_new_tokens
    if args.no_field_audit:
        overrides["field_audit"] = False
    if args.field_config is not None:
        overrides["field_config_path"] = str(args.field_config)
    if args.confidence_floor is not None:
        overrides["confidence_floor"] = args.confidence_floor
    if not overrides:
        return config
    payload = config.to_dict()
    payload.pop("source_path", None)
    payload["eval"].update(overrides)
    return TrainingConfig.from_dict(payload, source_path=config.source_path)


def _resolve_image(path: Path) -> Path:
    """Find a document image, looking in the working directory first.

    ``--image`` comes from a shell, so a relative path means relative to where the
    user is standing. The repository root is tried second, because the corpora this
    was trained on are addressed that way throughout the repo and
    ``--image data/dataset/...`` should keep working from anywhere.

    Args:
        path: Path as given on the command line.

    Returns:
        An existing absolute path.

    Raises:
        FileNotFoundError: If neither location holds the file. Both are named, so
            the message says where it looked.
    """
    candidate = path.expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        fallback = resolve_repo_path(candidate)
        if fallback.is_file():
            return fallback
        raise FileNotFoundError(f"no such image: {candidate} (also tried {fallback})")
    raise FileNotFoundError(f"no such image: {candidate}")


def _log_reading(reading: PageReading, *, verbose: bool) -> None:
    """Report one reading to the log stream.

    Faults are logged at warning level whether or not ``--verbose`` is set: the
    point of the audit is that a questionable field is noticed without anyone
    having asked for detail.

    Args:
        reading: The reading to report.
        verbose: Also log the transcription and every field found.
    """
    LOGGER.info("%s", reading.describe())
    if reading.truncated:
        LOGGER.warning(
            "%s: generation stopped at the token cap, so the transcription is "
            "incomplete; raise --max-new-tokens or read the page in regions",
            reading.image_path.name,
        )
    for finding in reading.findings:
        LOGGER.warning("  %s", finding.describe())
    if verbose:
        for row in reading.fields:
            LOGGER.debug("  field %s", json.dumps(row, ensure_ascii=False))
        LOGGER.debug("  text %r", reading.text)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``fa-ocr-lora``.

    Logs go to stderr and the result to stdout, so
    ``fa-ocr-lora predict --image scan.png > scan.txt`` yields the transcription
    and nothing else.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        ``0`` when every document was read, ``1`` on a configuration, model or
        image failure - including a partial failure, so a batch that lost a page
        cannot exit as though it had not.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    if args.region is not None and len(args.image) != 1:
        LOGGER.error(
            "--region reads one box out of one page, but %d images were given",
            len(args.image),
        )
        return 1

    try:
        config = _apply_overrides(load_training_config(args.config), args)
    except (ConfigError, FileNotFoundError) as exc:
        LOGGER.error("configuration error: %s", exc)
        return 1

    try:
        images = [_resolve_image(path) for path in args.image]
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1

    adapter_dir = resolve_adapter(
        config, adapter_dir=args.adapter_dir, use_adapter=not args.no_adapter
    )
    try:
        reader = load_reader(config, adapter_dir=adapter_dir)
    except (ConfigError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        LOGGER.error(
            "train an adapter first, or pass --no-adapter to read with the base "
            "model alone"
        )
        return 1
    LOGGER.info("%s", reader.describe())

    region = tuple(args.region) if args.region is not None else None
    readings: list[PageReading] = []
    failures: dict[str, int] = {}
    for image_path in images:
        try:
            reading = read_document(reader, image_path, region=region)
        except torch.OutOfMemoryError as exc:
            LOGGER.error("%s: out of memory: %s", image_path.name, exc)
            LOGGER.error(
                "lower --max-new-tokens (currently %d) or image.max_pixels",
                config.eval.max_new_tokens,
            )
            failures["OutOfMemoryError"] = failures.get("OutOfMemoryError", 0) + 1
            continue
        except (SampleTooLong, OSError, ValueError) as exc:
            reason = type(exc).__name__
            failures[reason] = failures.get(reason, 0) + 1
            LOGGER.error("%s: %s", image_path.name, exc)
            continue
        readings.append(reading)
        _log_reading(reading, verbose=args.verbose)

    payload: dict[str, Any] = {
        "schema_version": READING_SCHEMA_VERSION,
        "model_id": config.model.model_id,
        "revision": config.model.revision,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "quantization": config.model.quantization,
        "quantize_vision": config.model.quantize_vision,
        "max_pixels": config.image.max_pixels,
        "max_new_tokens": config.eval.max_new_tokens,
        "num_beams": config.eval.num_beams,
        "field_audit": reader.rules is not None,
        "documents": [reading.to_dict() for reading in readings],
    }
    if failures:
        # Counted in the payload as well as the log: a JSON consumer reading a
        # short document list has no other way to tell a partial batch from a
        # small one.
        payload["failed"] = failures

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for reading in readings:
            if len(readings) > 1:
                print(f"--- {reading.image_path} ---")
            print(reading.text)
    if args.output is not None:
        LOGGER.info("json %s", write_json(args.output, payload))

    if failures:
        detail = ", ".join(f"{name}={count}" for name, count in sorted(failures.items()))
        LOGGER.error(
            "%d of %d document(s) could not be read: %s",
            sum(failures.values()),
            len(images),
            detail,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
