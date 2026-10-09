"""Entry point for ``fa-ocr-app``: wire flags, model, batch, and output."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from .args import parse_args
from .batch import run_batch
from .browse import browse_images
from .config import APP_SCHEMA_VERSION
from .discovery import collect_images, default_checkpoint, discover_checkpoints
from .loader import load_app_model
from .resolve import build_app_config

__all__ = ["main"]

LOGGER = logging.getLogger("fa_ocr_lora.app")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``fa-ocr-app``.

    Logs go to stderr and results to stdout, so
    ``fa-ocr-app --image scan.png > out.txt`` yields text and nothing else.

    Args:
        argv: Argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        ``0`` when every image succeeded, ``1`` otherwise.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s", stream=sys.stderr,
    )
    if args.list_checkpoints:
        found = discover_checkpoints(args.checkpoints)
        if not found:
            print(f"no adapters under {args.checkpoints}")
            return 1
        preferred = default_checkpoint(args.checkpoints)
        for path in found:
            marker = "  (default)" if preferred is not None and path == preferred else ""
            print(f"{path}{marker}")
        return 0
    try:
        prompt_images: list[Path] = []
        if args.browse is not None:
            prompt_images.extend(browse_images(args.browse))
        prompt_images.extend(collect_images(args.image, args.image_dir,
                                            recursive=args.recursive))
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    if not prompt_images:
        LOGGER.error("no images: pass --image, --image-dir, or --browse")
        return 1
    try:
        app_config = build_app_config(args)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info(
        "model %s @ %s | adapter %s | %d image(s)",
        app_config.model_id, app_config.revision[:12],
        app_config.adapter_dir if app_config.adapter_dir else "none (base)",
        len(prompt_images),
    )
    try:
        loader = load_app_model(app_config)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        LOGGER.error("model load failed: %s", exc)
        return 1
    results = run_batch(loader, prompt_images, show_progress=not args.no_progress)
    failures: dict[str, int] = {}
    for result in results:
        if result.error:
            failures[result.error] = failures.get(result.error, 0) + 1
    payload: dict[str, Any] = {
        "schema_version": APP_SCHEMA_VERSION,
        "model_id": app_config.model_id,
        "revision": app_config.revision,
        "adapter_dir": str(app_config.adapter_dir) if app_config.adapter_dir else None,
        "prompt": app_config.prompt,
        "quantization": app_config.quantization,
        "max_pixels": app_config.max_pixels,
        "max_new_tokens": app_config.max_new_tokens,
        "documents": [r.to_dict() for r in results],
    }
    if failures:
        payload["failed"] = failures
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for result in results:
            if len(results) > 1:
                print(f"--- {result.image.name} ---")
            print(result.text if not result.error else f"[FAILED: {result.error}]")
    if args.output is not None:
        out = args.output.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".json":
            out.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            LOGGER.info("json -> %s", args.output)
        else:
            # A text suffix (.md/.txt/...) gets the human-readable
            # transcription, not the JSON bundle: Markdown renderers choke
            # on a JSON blob, and the bundle is still one --json away.
            # The sidecar keeps machine-readable provenance beside it.
            parts: list[str] = []
            for result in results:
                if len(results) > 1:
                    parts.append(f"# {result.image.name}\n")
                if result.error:
                    parts.append(f"[FAILED: {result.error}]")
                else:
                    parts.append(result.text)
            out.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")
            sidecar = out.with_suffix(out.suffix + ".json")
            sidecar.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            LOGGER.info("text -> %s (+ %s)", args.output, sidecar.name)
    if failures:
        detail = ", ".join(f"{n}={c}" for n, c in sorted(failures.items()))
        LOGGER.error("%d of %d image(s) failed: %s", sum(failures.values()),
                     len(results), detail)
        return 1
    total_chars = sum(len(r.text) for r in results)
    total_time = sum(r.seconds for r in results)
    LOGGER.info("done: %d image(s), %d chars in %.1fs",
                len(results), total_chars, total_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
