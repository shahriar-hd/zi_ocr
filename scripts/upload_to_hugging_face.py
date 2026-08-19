"""Publish a finished HuggingFace export to the Hub as a dataset page.

This is the same code path as ``fa-ocr-synth --hf-upload``, exposed as a script
so publishing does not require a render flag and can be re-run on its own after
an interruption. It reads a directory the export stage already wrote; it never
renders, re-packs, or modifies anything locally.

Setup (one time):

1.  ``pip install 'huggingface_hub>=0.26'`` (or ``pip install -e '.[hub]'``).
2.  Create a **write** token at https://huggingface.co/settings/tokens and put
    it in ``.env`` as ``HF_TOKEN=hf_...``. Alternatively run
    ``huggingface-cli login`` once on this machine - the token cache is honoured
    too. The token is never read from the command line, never logged, and never
    written to a config file.
3.  Decide the destination and put it in ``.env`` as
    ``FA_OCR_HF_REPO_ID=<your-username>/persian-official-letters-synth``, or pass
    ``--repo`` each time. The repository is created on first upload, so there is
    no need to make it in the web UI first - though you may, at
    https://huggingface.co/new-dataset.

Usage::

    # what would be pushed, no network traffic
    python scripts/upload_to_hugging_face.py --dry-run

    # create the repo (private) and push everything
    python scripts/upload_to_hugging_face.py --private

    # re-push one family after re-exporting it
    python scripts/upload_to_hugging_face.py --units images

    # fix a typo in the card only
    python scripts/upload_to_hugging_face.py --units cards

    # a multi-GB corpus: chunked, resumable, many small commits
    python scripts/upload_to_hugging_face.py --mode large

Upload order is families first, dataset card last: the card's ``configs:`` block
is what the Hub's dataset viewer acts on, so publishing it before the shards it
describes leaves the page reporting a broken dataset for the length of the
upload.

Re-running is safe. The Hub deduplicates by content hash, so a file that is
already there is not re-sent, and ``--units`` lets an interrupted run resume at
the family it stopped on rather than at the beginning.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fa_ocr_lora.config import HF_FAMILIES, ConfigError, load_config  # noqa: E402
from fa_ocr_lora.dataset.common.io_utils import resolve_repo_path  # noqa: E402
from fa_ocr_lora.dataset.export.errors import ExportError  # noqa: E402
from fa_ocr_lora.dataset.export.upload import (  # noqa: E402
    SIDECAR_UNIT,
    UPLOAD_MODES,
    upload_export,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the script's arguments.

    Args:
        argv: Argument list; defaults to :data:`sys.argv`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The token comes from HF_TOKEN in .env (or `huggingface-cli "
            "login`); it is never accepted as an argument."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config YAML to read (default: $FA_OCR_CONFIG, else the committed one)",
    )
    parser.add_argument(
        "--repo",
        default=None,
        metavar="OWNER/NAME",
        help=(
            "dataset repository to push to (default: $FA_OCR_HF_REPO_ID, else "
            "huggingface.hub_repo_id in the config)"
        ),
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help=(
            "export tree to publish (default: $FA_OCR_HF_OUTPUT_DIR, else "
            "huggingface.output_dir)"
        ),
    )
    parser.add_argument(
        "--units",
        default=None,
        metavar="LIST",
        help=(
            "comma-separated subset to push instead of everything present: "
            + ", ".join(HF_FAMILIES)
            + f", or {SIDECAR_UNIT} for the dataset card, dataset_info.json, "
            "the CSV indexes and the logo licence manifest"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=UPLOAD_MODES,
        default="folder",
        help=(
            "folder (default): one commit per family. large: the Hub's "
            "chunked resumable uploader, for a corpus of many GB"
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        metavar="BRANCH",
        help="commit to this branch, creating it if needed (folder mode only)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help=(
            "make the repository private if this run creates it; ignored for "
            "one that already exists"
        ),
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="fail instead of creating a repository that does not exist yet",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be pushed and stop, without any network traffic",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="log warnings only",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the config, then push the export tree.

    Args:
        argv: Argument list; defaults to :data:`sys.argv`.

    Returns:
        ``0`` on success, ``1`` on any configuration or upload failure.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        config = load_config(args.config)
    except ConfigError as error:
        print(f"error: configuration unusable: {error}", file=sys.stderr)
        return 1

    if args.export_dir is not None:
        config = replace(
            config,
            huggingface=replace(
                config.huggingface,
                output_dir=resolve_repo_path(args.export_dir),
            ),
        )

    units = (
        None
        if args.units is None
        else tuple(part.strip() for part in args.units.split(",") if part.strip())
    )
    try:
        summary = upload_export(
            config,
            repo_id=args.repo,
            families=units,
            mode=args.mode,
            private=args.private or None,
            revision=args.revision,
            create=not args.no_create,
            dry_run=args.dry_run,
        )
    except (ConfigError, ExportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if summary.dry_run:
        print(
            f"dry run: {summary.plan.files} file(s), "
            f"{summary.plan.bytes / 1024**3:.2f} GiB -> {summary.plan.url}"
        )
        print(f"token would come from: {summary.token_source}")
        return 0
    print(
        f"pushed {len(summary.uploaded)} unit(s) "
        f"({', '.join(summary.uploaded)}) to {summary.plan.url}"
    )
    if summary.created:
        print(
            "repository created"
            + (" (private)" if summary.plan.private else " (public)")
        )
    for url in summary.commits:
        print(f"  {url}")
    print(f"dataset page: {summary.plan.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
