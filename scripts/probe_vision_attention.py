"""Measure how a training batch's activation memory scales with batch size.

The A100 projection in :mod:`fa_ocr_lora.training.simulate` assumes the NaViT-style
vision tower attends *block-diagonally*: a batch of images arrives as one packed
patch sequence, but each image's patches attend only within that image. That
assumption is load-bearing. If the tower instead builds one dense
``packed x packed`` score matrix per batch, the projected peak rises from 29 GiB to
62 GiB and the first step of an A100 run out-of-memories.

This settles it by measurement. With ``N`` images of ``P`` patches each, the vision
attention term is:

    block-diagonal    N x heads x P^2      -> linear in N
    dense             heads x (N.P)^2      -> quadratic in N

Everything else in a forward pass is linear in ``N``, so fitting the exponent of
peak-allocated against batch size separates the two: a value near 1 means packed, a
value near 2 means dense. The term is quadratic in ``P`` as well, so a large
``--patches`` makes the two hypotheses diverge faster; the default is sized to fit a
4 GB card.

The batch is built through this project's own encoder and collate function rather
than from hand-shaped tensors, so what is measured is the code path training
actually runs - and the probe needs no knowledge of the vendor's internals, which is
the point: it characterises the packing rather than reading it.

    python scripts/probe_vision_attention.py --patches 1024 --batches 1 2 4

Forward-only under ``no_grad`` and with no optimiser. That is the right scope: the
vision term is forward-only during training too, because the tower is frozen and
``pixel_values`` carries no gradient.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # the runtime import waits for the sys.path bootstrap below
    from fa_ocr_lora.training.config import TrainingConfig

LOGGER = logging.getLogger("probe_vision_attention")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

#: Patch side in pixels, from the processor's own ``patch_size``.
PATCH_PX = 14


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Probe how batch activation memory scales with batch size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="Training YAML.")
    parser.add_argument(
        "--patches",
        type=int,
        default=1024,
        help="Patches per image. Larger separates the hypotheses faster.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Batch sizes to measure. Ascending; an OOM stops the sweep.",
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=64,
        help="Target length per sample, kept short so the decoder stays quiet.",
    )
    return parser.parse_args(argv)


def _square_image(patches: int, directory: Path) -> tuple[Path, int]:
    """Write a square PNG whose patch grid is even in both dimensions.

    Args:
        patches: Requested patches per image.
        directory: Where to write it.

    Returns:
        The path, and the patch count actually achieved.
    """
    from PIL import Image

    side = int(math.isqrt(patches))
    side -= side % 2  # merge_size 2 needs an even grid
    pixels = side * PATCH_PX
    path = directory / f"probe_{side}x{side}.png"
    # Mid-grey rather than noise: content does not affect allocation, and a flat
    # fill keeps the PNG small.
    Image.new("RGB", (pixels, pixels), (128, 128, 128)).save(path)
    return path, side * side


def _forward_peak(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> float:
    """Return peak MiB allocated by one forward pass, resident weights excluded."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with torch.no_grad():
        model(**batch)
    peak = torch.cuda.max_memory_allocated() - baseline
    gc.collect()
    torch.cuda.empty_cache()
    return peak / 1024**2


def _build_batch(
    encoder: Any,
    *,
    image_path: Path,
    text: str,
    batch: int,
    pad_token_id: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Encode one image ``batch`` times and collate it, exactly as training does.

    The same image repeated is the right stimulus here: equal patch counts per
    image make the packed length exactly ``batch x P``, so the exponent fitted
    downstream is not confounded by a varying image size.
    """
    from fa_ocr_lora.training.data import collate_samples

    sample = encoder.encode(image_path=image_path, crop_box=None, text=text)
    feature = {
        "input_ids": sample.input_ids,
        "labels": sample.labels,
        "pixel_values": sample.pixel_values,
        "image_grid_thw": sample.image_grid_thw,
    }
    collated = collate_samples([feature] * batch, pad_token_id=pad_token_id, dtype=dtype)
    return {k: v.to("cuda") for k, v in collated.items()}


def _load(config: TrainingConfig) -> tuple[torch.nn.Module, Any, torch.dtype]:
    """Load the base model and processor the way the trainer does."""
    from fa_ocr_lora.training.encoding import SampleEncoder

    # The private loader, deliberately: this probe wants the base model with no
    # adapter and no sliced-loss wrapper, so that the peak it reports is the
    # vision and decoder activation cost alone.
    from fa_ocr_lora.training.modeling import _load_base

    LOGGER.info("loading %s (%s)", config.model.model_id, config.model.revision)
    model, processor, dtype, _ = _load_base(config)
    model.eval()
    encoder = SampleEncoder(processor, model=config.model, image=config.image)
    return model, encoder, dtype


def _exponent(measurements: list[tuple[int, float]]) -> float:
    """Least-squares slope of log(peak) against log(batch).

    Args:
        measurements: ``(batch, peak_mib)`` pairs, at least two, with at least
            two distinct batch sizes.

    Returns:
        The exponent ``k`` in ``peak ~ batch**k``.
    """
    xs = [math.log(b) for b, _ in measurements]
    ys = [math.log(p) for _, p in measurements]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    pairs = zip(xs, ys, strict=True)
    return sum((x - mx) * (y - my) for x, y in pairs) / denom


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if not torch.cuda.is_available():
        LOGGER.error("no CUDA device; this probe measures allocator peaks")
        return 1

    from fa_ocr_lora.training.config import load_training_config

    config = load_training_config(args.config)
    model, encoder, dtype = _load(config)
    pad = encoder.processor.tokenizer.pad_token_id or 0
    text = "تست " * max(1, args.target_chars // 5)

    measurements: list[tuple[int, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        image_path, per_image = _square_image(args.patches, Path(tmp))
        LOGGER.info("image %s, %d patches per image", image_path.name, per_image)
        LOGGER.info("")
        LOGGER.info("  batch   packed patches   peak MiB   MiB/image")
        for batch in sorted(args.batches):
            try:
                inputs = _build_batch(
                    encoder,
                    image_path=image_path,
                    text=text,
                    batch=batch,
                    pad_token_id=pad,
                    dtype=dtype,
                )
                peak = _forward_peak(model, inputs)
                del inputs
            except torch.OutOfMemoryError:
                LOGGER.info("  %5d   OOM - stopping the sweep here", batch)
                gc.collect()
                torch.cuda.empty_cache()
                break
            measurements.append((batch, peak))
            LOGGER.info(
                "  %5d   %14d   %8.1f   %9.1f",
                batch,
                per_image * batch,
                peak,
                peak / batch,
            )

    if len(measurements) < 2:
        LOGGER.error("need at least two batch sizes to fit an exponent")
        return 1

    LOGGER.info("")
    LOGGER.info("  peak ~ batch^%.2f  over all points", _exponent(measurements))
    # Batch 1 is not on the asymptote: it pays one-off workspace and picks
    # different kernel tilings, which inflates its share and so drags a fit over
    # all points upward. The exponent between the two largest batches is the
    # statistic that actually discriminates, because both points carry that same
    # fixed cost and it cancels in the ratio.
    asymptotic = _exponent(measurements[-2:])
    LOGGER.info("  peak ~ batch^%.2f  between the two largest batches", asymptotic)

    LOGGER.info("")
    if asymptotic < 1.35:
        LOGGER.info("  PACKED - attention is per-image; the 29 GiB projection holds")
    elif asymptotic > 1.65:
        LOGGER.info("  DENSE - attention spans the batch; the projection is wrong")
    else:
        LOGGER.info("  INCONCLUSIVE - raise --patches or widen --batches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
