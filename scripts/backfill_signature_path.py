#!/usr/bin/env python3
"""Backfill the ``signature_path`` column into an existing generator manifest.

``generator.csv`` files written before the signature column existed have no
record of which UTSig image belongs to each letter. This script adds it in
place, so a manifest produced by an earlier run stops being the odd one out.

The value is not drawn at random per run: it comes from
``test_text_generator.assign_signature``, which derives the choice from
``(seed, sample_index)``. Backfilling therefore produces exactly the path a
fresh rebuild would, and re-running this script is a no-op. Rows that already
carry a non-empty ``signature_path`` are left untouched.

Examples:
    Backfill the default manifest::

        python scripts/backfill_signature_path.py

    Inspect the change without writing it::

        python scripts/backfill_signature_path.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "test"))

import test_text_generator as helper  # noqa: E402

DEFAULT_MANIFEST: Final[Path] = REPO_ROOT / "data" / "text" / "generator.csv"
COLUMN: Final[str] = "signature_path"


def backfill(manifest_path: Path, signatures_dir: Path, dry_run: bool) -> int:
    """Add or complete the signature column in one manifest.

    Args:
        manifest_path: The ``generator.csv`` to rewrite.
        signatures_dir: Root of the UTSig corpus.
        dry_run: Report the change without writing the file.

    Returns:
        The number of rows given a signature path.

    Raises:
        FileNotFoundError: If the manifest or the signature corpus is missing.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest {manifest_path} not found; generate letters first.")

    signatures = helper.load_signatures(signatures_dir)
    if not signatures:
        raise FileNotFoundError(
            f"No .tif signatures under {signatures_dir}. "
            "Expected the UTSig corpus with Genuine/ and Forgery/ subfolders."
        )

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise FileNotFoundError(f"Manifest {manifest_path} has no header row.")
    if COLUMN not in fieldnames:
        fieldnames.append(COLUMN)

    filled = 0
    for row in rows:
        if row.get(COLUMN):
            continue
        try:
            index = int(row["sample_index"])
        except (KeyError, ValueError):
            print(f"  skipping row with unusable sample_index: {row.get('sample_id', '?')}")
            row[COLUMN] = ""
            continue
        row[COLUMN] = helper.assign_signature(index, signatures)
        if row[COLUMN]:
            filled += 1

    signed = sum(1 for row in rows if row.get(COLUMN))
    print(f"{len(rows)} row(s) in {manifest_path}")
    print(f"{len(signatures)} signature(s) available under {signatures_dir}")
    print(f"{filled} row(s) newly assigned; {signed}/{len(rows)} now signed")

    if dry_run:
        print("dry run: manifest left unchanged")
        return filled

    # Write beside the target, then swap, so an interrupted run cannot leave a
    # half-written manifest where the complete one used to be.
    temp_path = manifest_path.with_suffix(".csv.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(manifest_path)
    print(f"wrote {manifest_path}")
    return filled


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the backfill."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--signatures-dir", type=Path, default=helper.DEFAULT_SIGNATURES_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        backfill(args.manifest, args.signatures_dir, args.dry_run)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
