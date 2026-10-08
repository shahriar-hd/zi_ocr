"""Interactive image picker for the local VLM app (replaced by Flet later)."""

from __future__ import annotations

import sys
from pathlib import Path

from ..dataset.common.io_utils import resolve_repo_path
from .config import IMAGE_SUFFIXES

__all__ = ["browse_images"]


def browse_images(image_dir: str | Path) -> list[Path]:
    """Interactively pick images from a directory listing.

    Prints a numbered list and reads a selection like ``1 3 5-8`` or ``all``.
    Kept dependency-free (plain ``input()``) so the CLI works on any box the
    model itself runs on - the Flet GUI later replaces exactly this function.

    Args:
        image_dir: Directory to list images from (non-recursive).

    Returns:
        The selected absolute image paths, in listing order.

    Raises:
        FileNotFoundError: If the directory holds no images.
    """
    root = Path(image_dir).expanduser()
    if not root.is_absolute():
        repo_try = resolve_repo_path(root)
        if repo_try.is_dir():
            root = repo_try
    images = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"no images under {root}")
    print(f"{len(images)} image(s) under {root}:")
    for index, path in enumerate(images, start=1):
        try:
            size_kb = path.stat().st_size // 1024
            print(f"  [{index:>3}] {path.name}  ({size_kb} KiB)")
        except OSError:
            print(f"  [{index:>3}] {path.name}")
    raw = input("Select [e.g. '1 3 5-8', 'all', Enter=all]: ").strip().lower()
    if raw in ("", "all", "*"):
        return [p.resolve() for p in images]
    selected: list[int] = []
    for token in raw.replace(",", " ").split():
        if "-" in token:
            start_s, _, end_s = token.partition("-")
            try:
                start, end = int(start_s), int(end_s)
            except ValueError:
                print(f"ignoring {token!r}", file=sys.stderr)
                continue
            lo, hi = min(start, end), max(start, end)
            selected.extend(range(lo, hi + 1))
        else:
            try:
                selected.append(int(token))
            except ValueError:
                print(f"ignoring {token!r}", file=sys.stderr)
    valid = sorted(set(selected))
    picked = [images[i - 1].resolve() for i in valid if 1 <= i <= len(images)]
    if not picked:
        print("nothing selected; using all images.", file=sys.stderr)
        return [p.resolve() for p in images]
    return picked
