"""Command-line arguments for the local VLM app (``fa-ocr-app``)."""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["parse_args"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="fa-ocr-app",
        description=(
            "Run PaddleOCR-VL-1.6 + your LoRA adapter on document images "
            "with a custom prompt. Tuned for a 4 GB laptop GPU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, nargs="+", default=None,
                        help="Image file(s) to run.")
    parser.add_argument("--image-dir", type=Path, default=None,
                        help="Directory of images to run (sorted by name).")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into --image-dir subdirectories.")
    parser.add_argument("--browse", type=Path, default=None, metavar="DIR",
                        help="List images in DIR and pick interactively.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="User prompt sent with every image.")
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help="File holding the prompt (for long prompts).")
    parser.add_argument("--adapter-dir", type=Path, default=None,
                        help="Adapter to apply; defaults to checkpoints/adapter_only.")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Run the base model without any adapter.")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"),
                        help="Checkpoints root for discovery.")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List usable adapters and exit.")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override the base model repo.")
    parser.add_argument("--revision", type=str, default=None,
                        help="Override the pinned Hub revision.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Training YAML for model/revision/quant defaults.")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Generation cap per image.")
    parser.add_argument("--max-pixels", type=int, default=1003520,
                        help="Starting image pixel budget (halved on OOM).")
    parser.add_argument("--min-pixels", type=int, default=147384,
                        help="Floor the OOM backoff never drops below.")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="Compute dtype; auto picks bf16 on a 3050 Ti.")
    parser.add_argument("--low-vram", action="store_true",
                        help="Force NF4 even if the YAML says none.")
    parser.add_argument("--no-quant", action="store_true",
                        help="Disable NF4 (for cards with headroom).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Prefetch threads decoding ahead of the GPU.")
    parser.add_argument("--output", type=Path, default=None,
                        help=".json writes the full bundle; any other suffix writes "
                        "plain text plus a .json sidecar.")
    parser.add_argument("--json", action="store_true",
                        help="Print the full JSON bundle to stdout.")
    parser.add_argument("--no-progress", action="store_true",
                        help="Suppress per-image progress lines on stderr.")
    parser.add_argument("--verbose", action="store_true", help="Debug logging.")
    return parser.parse_args(argv)
