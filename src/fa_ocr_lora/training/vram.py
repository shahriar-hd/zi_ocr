"""VRAM budgeting: measure what is free, project what the run needs, say so.

On a 4 GB laptop GPU the run does not get 4 GB. A GNOME session with a file
manager, a document viewer and a terminal open holds between 1.5 and 1.8 GB of
it, and that figure moves by ~200 MiB as windows open and close. Measured on the
development machine over several minutes: free VRAM ranged 1984-2182 MiB of 3746
MiB visible. A configuration that fits at one moment therefore OOMs at another,
and the failure lands mid-epoch rather than at startup.

So this module does three things before any weights are loaded:

*Projects the requirement* from measured coefficients (see :data:`WEIGHTS_MIB`
and :data:`ACTIVATION_PER_TOKEN_MIB`) rather than from arithmetic on parameter
counts. Predicting activation cost from first principles was tried and was wrong
by more than the margin being defended: quantised weights carry allocator
overhead, and gradient checkpointing's real cost depends on where the recompute
boundaries fall. The numbers here come from a sweep on this model.

*Names what holds the card.* "Not enough VRAM" is not actionable; "nautilus is
holding 128 MiB" is. The banner lists the processes so closing one is an obvious
option.

*Clamps the image budget* as a last resort. This is deliberately the weakest of
the three levers and the banner says so: at these settings weights are ~1.5 GB of
a ~1.9 GB peak, so halving image resolution buys far less than quantising the
frozen vision tower does. Clamping keeps a run alive on a busy desktop; it does
not rescue a configuration that never fit.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Final

import torch

from .config import ImageConfig, ModelConfig
from .images import SIZE_FACTOR

__all__ = [
    "ACTIVATION_BASE_MIB",
    "ACTIVATION_PER_TOKEN_MIB",
    "BASELINE_SEQ",
    "SAFETY_MIB",
    "VramReport",
    "WEIGHTS_MIB",
    "banner",
    "clamp_max_pixels",
    "gpu_holders",
    "probe_vram",
    "projected_peak_mib",
]

LOGGER = logging.getLogger(__name__)

#: Resident weight cost in MiB, keyed by ``(quantization, quantize_vision)``.
#:
#: Measured with ``torch.cuda.memory_allocated()`` immediately after
#: ``load_training_model`` on PaddleOCR-VL at revision 2b77538ef936, not derived
#: from parameter counts - NF4 stores per-block absmax and a double-quantised
#: scale on top of the 4-bit payload, and the difference from the naive estimate
#: was ~400 MiB. Used only for the pre-load warning; once the model is loaded the
#: exact figure is read from the allocator and reported instead.
WEIGHTS_MIB: Final[dict[tuple[str, bool], int]] = {
    ("nf4", True): 870,
    ("nf4", False): 1510,
    # bf16 throughout. Not measured on this machine - it does not fit here - so
    # this is the parameter-count figure and is flagged as an estimate:
    # 958,588,736 parameters x 2 bytes = 1,917,177,472 B = 1,828 MiB. An earlier
    # 1,730 sat here, from a census that summed the four blocks to 868M and so
    # missed ~90M parameters; it understated the resident cost by ~98 MiB.
    ("none", False): 1828,
    ("none", True): 1828,
}

#: Sequence length the activation measurements are anchored at.
BASELINE_SEQ: Final[int] = 1600

#: Peak-over-weights cost at :data:`BASELINE_SEQ`, in MiB. Includes the vision
#: tower's transient forward, the checkpointed decoder, and the sliced loss.
ACTIVATION_BASE_MIB: Final[int] = 275

#: Marginal peak cost per additional sequence token, in MiB. From a sweep at
#: seq 1600/1900/2200/2500 measuring 275/333/378/423 MiB over weights: a slope of
#: 148 MiB per 900 tokens. Linear across that range to within a few MiB.
ACTIVATION_PER_TOKEN_MIB: Final[float] = 148.0 / 900.0

#: Margin held back from free VRAM. Covers the ~200 MiB the desktop session was
#: measured to swing by, so a browser tab opening mid-epoch does not end the run.
SAFETY_MIB: Final[int] = 200


def projected_peak_mib(
    model: ModelConfig, image: ImageConfig, *, weights_mib: int | None = None
) -> float:
    """Project peak allocated VRAM for the longest permitted sequence.

    ``image.reserved_vram_mib``, when non-zero, replaces the whole projection: it
    is the escape hatch for hardware the measured coefficients do not describe,
    and taking it as the answer rather than as one term keeps its meaning
    unambiguous.

    Args:
        model: Model configuration, for the quantization mode.
        image: Image configuration, for the sequence ceiling and the override.
        weights_mib: Measured weight cost. When ``None`` the estimate from
            :data:`WEIGHTS_MIB` is used, which is what the pre-load banner has.

    Returns:
        Projected peak in MiB at ``image.max_total_seq_len``.
    """
    if image.reserved_vram_mib > 0:
        return float(image.reserved_vram_mib)
    resident = (
        weights_mib
        if weights_mib is not None
        else WEIGHTS_MIB.get((model.quantization, model.quantize_vision), 1828)
    )
    over = ACTIVATION_BASE_MIB + ACTIVATION_PER_TOKEN_MIB * max(
        0, image.max_total_seq_len - BASELINE_SEQ
    )
    return resident + over


def gpu_holders() -> tuple[tuple[str, int], ...]:
    """List the processes holding GPU memory, largest first.

    Uses ``nvidia-smi`` because the CUDA API reports only a total. Note that it
    lists *compute* contexts, so the display server's own allocation shows up in
    the total but not here - which is why the banner reports both.

    Returns:
        ``(process_name, mib)`` pairs. Empty when ``nvidia-smi`` is missing or
        fails, since a banner is not worth failing a run over.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return ()
    try:
        completed = subprocess.run(  # noqa: S603 - fixed binary, no shell
            [
                binary,
                "--query-compute-apps=used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        LOGGER.debug("could not list GPU processes: %s", exc)
        return ()

    holders: list[tuple[str, int]] = []
    for line in completed.stdout.splitlines():
        parts = [field.strip() for field in line.split(",", 1)]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        holders.append((parts[1].rsplit("/", 1)[-1], int(parts[0])))
    return tuple(sorted(holders, key=lambda item: -item[1]))


@dataclass(frozen=True, slots=True)
class VramReport:
    """What the GPU has, what the run wants, and the gap between them.

    Attributes:
        available: Whether a CUDA device is present at all.
        total_mib: Memory the driver reports on the device.
        free_mib: Memory free at probe time.
        projected_mib: Projected peak for the configured sequence ceiling.
        budget_mib: Free memory minus :data:`SAFETY_MIB`, the figure the run may
            actually spend.
        max_pixels: Image budget after clamping.
        clamped: Whether ``max_pixels`` was reduced from the configured value.
        holders: Processes holding GPU memory.
    """

    available: bool
    total_mib: int
    free_mib: int
    projected_mib: float
    budget_mib: int
    max_pixels: int
    clamped: bool
    holders: tuple[tuple[str, int], ...] = ()

    @property
    def fits(self) -> bool:
        """Whether the projected peak is inside the spendable budget."""
        return not self.available or self.projected_mib <= self.budget_mib

    @property
    def image_tokens(self) -> int:
        """Visual tokens the clamped budget allows for one image."""
        return self.max_pixels // (SIZE_FACTOR * SIZE_FACTOR)


def clamp_max_pixels(configured: int, *, shortfall_mib: float, min_pixels: int) -> int:
    """Reduce the image budget to recover a projected shortfall.

    Each visual token covers ``SIZE_FACTOR**2`` pixels and costs
    :data:`ACTIVATION_PER_TOKEN_MIB` of peak, so the shortfall converts directly
    into a token count to give up. The result is snapped down to a whole number of
    tokens and floored at ``min_pixels``: below the processor's own floor the
    image would be upscaled again, which would spend the memory this clamp is
    trying to save.

    Args:
        configured: ``image.max_pixels`` as written in the config.
        shortfall_mib: Projected peak minus spendable budget. Zero or negative
            leaves the budget untouched.
        min_pixels: Lower bound, the processor's ``min_pixels``.

    Returns:
        The clamped pixel budget, never below ``min_pixels``.
    """
    if shortfall_mib <= 0:
        return configured
    # Work in tokens rather than pixels: a token is the indivisible unit of both
    # cost and area, so the result is exact instead of rounded twice.
    per_token_px = SIZE_FACTOR * SIZE_FACTOR
    give_up = int(shortfall_mib / ACTIVATION_PER_TOKEN_MIB) + 1
    tokens = max(1, configured // per_token_px - give_up)
    return max(min_pixels, tokens * per_token_px)


def probe_vram(model: ModelConfig, image: ImageConfig) -> VramReport:
    """Measure free VRAM and clamp the image budget to what fits.

    Args:
        model: Model configuration section.
        image: Image configuration section.

    Returns:
        The report. On a machine with no CUDA device every figure is zero and
        ``available`` is ``False``, so a CPU-only import still works.
    """
    if not torch.cuda.is_available():
        return VramReport(
            available=False,
            total_mib=0,
            free_mib=0,
            projected_mib=0.0,
            budget_mib=0,
            max_pixels=image.max_pixels,
            clamped=False,
        )

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_mib = int(free_bytes // 2**20)
    budget_mib = max(0, free_mib - SAFETY_MIB)
    projected = projected_peak_mib(model, image)

    max_pixels = image.max_pixels
    if image.autoprobe_vram:
        max_pixels = clamp_max_pixels(
            image.max_pixels,
            shortfall_mib=projected - budget_mib,
            min_pixels=image.min_pixels,
        )
    return VramReport(
        available=True,
        total_mib=int(total_bytes // 2**20),
        free_mib=free_mib,
        projected_mib=projected,
        budget_mib=budget_mib,
        max_pixels=max_pixels,
        clamped=max_pixels != image.max_pixels,
        holders=gpu_holders(),
    )


def banner(report: VramReport, model: ModelConfig) -> str:
    """Format the startup banner.

    Says what is free, what the run projects, and - when those two disagree -
    which lever to pull, ordered by how much each is measured to buy. Quantising
    the frozen tower is first because it is worth ~640 MiB, more than every other
    option combined.

    Args:
        report: A probe result.
        model: Model configuration, to know which levers are still unpulled.

    Returns:
        The banner, newline-separated and without a trailing newline.
    """
    if not report.available:
        return "no CUDA device: training would run on CPU and is not supported"

    lines = [
        f"GPU        {report.free_mib} MiB free of {report.total_mib} MiB",
        f"projected  {report.projected_mib:.0f} MiB peak "
        f"({model.quantization}"
        f"{', vision quantized' if model.quantize_vision else ', vision bf16'}), "
        f"spendable {report.budget_mib} MiB after a {SAFETY_MIB} MiB margin",
    ]
    if report.clamped:
        lines.append(
            f"clamped    image budget to {report.max_pixels} px "
            f"({report.image_tokens} tokens) to fit"
        )
    if report.holders:
        held = ", ".join(f"{name} {mib} MiB" for name, mib in report.holders[:6])
        lines.append(f"holding    {held}")
    if not report.fits:
        lines.append(
            f"WARNING    projected peak exceeds the budget by "
            f"{report.projected_mib - report.budget_mib:.0f} MiB; this run may OOM"
        )
        if not model.quantize_vision and model.quantization == "nf4":
            lines.append(
                "           set model.quantize_vision: true - measured to save "
                "640 MiB, the largest single saving available"
            )
        elif model.quantization == "none":
            lines.append(
                "           set model.quantization: nf4 - measured to save "
                "~220 MiB on the decoder alone"
            )
        if report.holders:
            lines.append(
                f"           or close {report.holders[0][0]} to recover "
                f"{report.holders[0][1]} MiB"
            )
    return "\n".join(lines)
