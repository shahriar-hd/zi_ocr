"""Checkpoint discovery and image collection for the local VLM app."""

from __future__ import annotations

from pathlib import Path

from ..dataset.common.io_utils import resolve_repo_path
from .config import IMAGE_SUFFIXES

__all__ = [
    "collect_images",
    "default_checkpoint",
    "discover_checkpoints",
]


def discover_checkpoints(root: str | Path = "checkpoints") -> list[Path]:
    """List checkpoint dirs holding an adapter, newest first.

    A directory counts when it holds ``adapter_model.safetensors`` (or the
    ``.bin`` legacy name) beside an ``adapter_config.json`` - exactly what the
    runner can attach. ``checkpoints/logs`` is skipped implicitly.

    Args:
        root: Checkpoints directory, repo-relative or absolute.

    Returns:
        Adapter directories sorted by modification time, newest first.
    """
    base = resolve_repo_path(root)
    if not base.is_dir():
        return []
    found: list[Path] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        has_config = (child / "adapter_config.json").is_file()
        has_weights = (child / "adapter_model.safetensors").is_file() or (
            child / "adapter_model.bin"
        ).is_file()
        if has_config and has_weights:
            found.append(child)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def default_checkpoint(root: str | Path = "checkpoints") -> Path | None:
    """Pick the adapter to use when the user names none.

    Prefers ``adapter_only`` (the exported artifact) when present, otherwise
    the newest ``checkpoint-N``. Returns ``None`` when no adapter exists, in
    which case the caller runs the base model alone.

    Args:
        root: Checkpoints directory, repo-relative or absolute.

    Returns:
        The preferred adapter directory, or ``None``.
    """
    base = resolve_repo_path(root)
    exported = base / "adapter_only"
    if (exported / "adapter_config.json").is_file():
        return exported
    found = discover_checkpoints(root)
    return found[0] if found else None


def collect_images(
    explicit: list[Path] | None = None,
    image_dir: Path | None = None,
    recursive: bool = False,
) -> list[Path]:
    """Collect existing image files from explicit paths and/or a directory.

    Args:
        explicit: ``--image`` paths; each must be an existing file.
        image_dir: ``--image-dir`` directory to scan for images.
        recursive: Scan subdirectories too.

    Returns:
        Deduplicated absolute paths, directory scan in sorted order.

    Raises:
        FileNotFoundError: If an explicit path is missing or the directory
            holds no images.
    """
    collected: list[Path] = []
    seen: set[Path] = set()
    if explicit:
        for raw in explicit:
            candidate = raw.expanduser()
            resolved = candidate
            found = candidate.is_file()
            if not found and not candidate.is_absolute():
                fallback = resolve_repo_path(candidate)
                if fallback.is_file():
                    resolved = fallback
                    found = True
            if not found:
                raise FileNotFoundError(f"no such image: {raw}")
            _remember(collected, seen, resolved)
    if image_dir is not None:
        root = Path(image_dir).expanduser()
        if not root.is_absolute():
            repo_try = resolve_repo_path(root)
            if repo_try.is_dir():
                root = repo_try
        if not root.is_dir():
            raise FileNotFoundError(f"no such directory: {image_dir}")
        pattern = "**/*" if recursive else "*"
        scanned = sorted(
            p
            for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not scanned:
            raise FileNotFoundError(f"no images under {root}")
        for found_image in scanned:
            _remember(collected, seen, found_image)
    return collected


def _remember(collected: list[Path], seen: set[Path], candidate: Path) -> None:
    """Append one deduplicated absolute path to the collection."""
    resolved = candidate.expanduser()
    if not resolved.is_absolute():
        fallback = resolve_repo_path(resolved)
        if fallback.is_file():
            resolved = fallback
    resolved = resolved.resolve() if resolved.exists() else resolved
    if resolved not in seen:
        seen.add(resolved)
        collected.append(resolved)
