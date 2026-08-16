"""Filesystem, path, and manifest helpers shared across dataset stages.

Everything here is stage-agnostic: the text generator, the layout renderer,
the Persian Pixel filter, and the manifest-preparation step all reach for the
same repo-root resolution, atomic writes, and signature assignment. Keeping a
single copy is what stops the text stage and the render stage from disagreeing
about where ``data/`` lives or which signature belongs to a sample.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Final, Iterator, Sequence, TypeVar

#: Element type of a weighted pool, so the picked value keeps its own type.
_T = TypeVar("_T")

__all__ = [
    "DEFAULT_SEED",
    "SIGNATURE_PROBABILITY",
    "MANIFEST_COLUMNS",
    "TO_PERSIAN_DIGITS",
    "repo_root",
    "resolve_repo_path",
    "relative_to_repo",
    "ensure_dir",
    "read_json",
    "write_json",
    "write_bytes",
    "write_text",
    "to_persian_digits",
    "sample_rng",
    "weighted_choice",
    "load_signatures",
    "assign_signature",
    "read_manifest",
    "iter_manifest",
]

#: Master seed shared by every stage, so a sample index maps to the same
#: choices in the text stage and the render stage.
DEFAULT_SEED: Final[int] = 1337

#: Chance a sample carries a signature at all; ~30% of rows render unsigned.
SIGNATURE_PROBABILITY: Final[float] = 0.7

#: Digit translation table for Persian numerals.
TO_PERSIAN_DIGITS: Final[dict[int, str]] = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

#: Marker files that identify the repository root, in priority order.
_ROOT_MARKERS: Final[tuple[str, ...]] = ("pyproject.toml", "CLAUDE.md", ".git")


def repo_root() -> Path:
    """Locate the repository root by walking up from this module.

    Anchoring on the module rather than :func:`Path.cwd` means a path resolves
    identically whether the caller runs from the repo root, from ``src/``, or
    from a pytest working directory. The two existing copies of
    :func:`assign_signature` disagreed on exactly this point.

    Returns:
        Absolute path of the repository root.

    Raises:
        RuntimeError: If no marker file is found in any ancestor directory.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    raise RuntimeError(f"cannot locate repository root above {here}")


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repository root.

    Absolute paths pass through untouched, so callers can accept either form
    from a config file without branching.

    Args:
        path: Absolute path, or one relative to the repository root.

    Returns:
        An absolute path.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def relative_to_repo(path: str | Path) -> str:
    """Express a path relative to the repository root, for manifests and JSON.

    Args:
        path: Any filesystem path.

    Returns:
        The repo-relative POSIX string, or the unchanged path when it lies
        outside the repository.
    """
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return candidate.as_posix()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if absent and return it.

    Args:
        path: Directory to create.

    Returns:
        The resolved directory path.
    """
    directory = resolve_repo_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object from disk.

    Args:
        path: File to read.

    Returns:
        The decoded object.

    Raises:
        ValueError: If the file does not hold a JSON object.
    """
    target = resolve_repo_path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"{target}: expected a JSON object, got {type(payload).__name__}"
        )
    return payload


def write_json(path: str | Path, payload: dict[str, Any], *, indent: int = 2) -> Path:
    """Write a JSON object atomically.

    The file lands on a sibling temporary path and is then renamed, so a run
    killed mid-write leaves no half-written JSON for a resume scan to trip
    over.

    Args:
        path: Destination file.
        payload: Object to serialise.
        indent: Indentation passed to :func:`json.dumps`.

    Returns:
        The destination path.
    """
    target = resolve_repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    temp_path.replace(target)
    return target


def write_bytes(path: str | Path, payload: bytes) -> Path:
    """Write binary data atomically, for rendered PNG and PDF artifacts.

    Args:
        path: Destination file.
        payload: Bytes to write.

    Returns:
        The destination path.
    """
    target = resolve_repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".tmp")
    temp_path.write_bytes(payload)
    temp_path.replace(target)
    return target


def write_text(path: str | Path, payload: str) -> Path:
    """Write UTF-8 text atomically, for the Markdown reconstruction.

    Args:
        path: Destination file.
        payload: Text to write.

    Returns:
        The destination path.
    """
    target = resolve_repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".tmp")
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(target)
    return target


def to_persian_digits(value: Any) -> str:
    """Convert ASCII digits in a value to Persian numerals.

    Args:
        value: Any value; stringified before translation.

    Returns:
        The string with ``0-9`` replaced by ``۰-۹``.
    """
    return str(value).translate(TO_PERSIAN_DIGITS)


def sample_rng(seed: int, purpose: str, index: int) -> random.Random:
    """Derive the deterministic RNG for one sample and one purpose.

    Every consumer derives its randomness from this single idiom -
    ``f"{seed}:{purpose}:{index}"`` - so a sample renders identically no
    matter which stage or module draws first, and the corpus is reproducible
    from (config, seed) alone (CLAUDE.md 8).

    Args:
        seed: Master seed from the configuration.
        purpose: Stable name of the drawing consumer, e.g. ``"theme"`` or
            ``"stamp"``.
        index: Sample index.

    Returns:
        A new, seeded :class:`random.Random` instance.
    """
    return random.Random(f"{seed}:{purpose}:{index}")


def weighted_choice(
    options: Sequence[tuple[_T, float]],
    rng: random.Random,
) -> _T:
    """Pick one option by relative weight.

    Weights are relative, not percentages, so a pool can be retuned by editing
    one entry without renormalising the rest. Every visual module draws its
    variants through this one function, which is what keeps a single
    ``rng.random()`` call per decision - the property that lets a pool be
    reordered without shifting every later draw in the sample.

    Args:
        options: ``(value, weight)`` pairs; weights need not sum to one.
        rng: The sample's seeded RNG.

    Returns:
        The chosen value.

    Raises:
        ValueError: If the pool is empty or its weights sum to zero or less.
    """
    if not options:
        raise ValueError("weighted_choice needs at least one option")
    total = sum(weight for _, weight in options)
    if total <= 0:
        raise ValueError(f"weighted_choice needs positive weights, got {total}")
    target = rng.random() * total
    cumulative = 0.0
    for value, weight in options:
        cumulative += weight
        if target < cumulative:
            return value
    return options[-1][0]


def load_signatures(signatures_dir: str | Path) -> list[Path]:
    """Collect every signature image under the UTSig corpus.

    Both ``Genuine`` and ``Forgery`` are used: the page composer only needs a
    plausible ink shape, and a forged signature looks no less like handwriting
    than a genuine one, so the whole corpus widens the visual variety.

    Args:
        signatures_dir: Root of the UTSig tree.

    Returns:
        Sorted paths of the signature images; empty if the corpus is absent.
    """
    root = resolve_repo_path(signatures_dir)
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.tif"))


def assign_signature(
    sample_index: int,
    signatures: Sequence[Path],
    seed: int = DEFAULT_SEED,
    probability: float = SIGNATURE_PROBABILITY,
) -> str:
    """Pick the signature image for one sample, or none.

    Derived from ``(seed, sample_index)`` rather than drawn from a shared RNG,
    so a sample keeps the same signature no matter when it is generated, in
    which order, or how many times the manifest is rebuilt. That is what lets
    the backfill script complete old rows without contradicting a later run.

    Args:
        sample_index: Zero-based sample index.
        signatures: Pool returned by :func:`load_signatures`.
        seed: Master seed, matching the run's own.
        probability: Chance this sample carries a signature at all.

    Returns:
        A path relative to the repository root, or ``""`` when unsigned.
    """
    if not signatures:
        return ""
    rng = random.Random(f"{seed}:signature:{sample_index}")
    if rng.random() >= probability:
        return ""
    return relative_to_repo(rng.choice(signatures))


#: Manifest columns of the text stage. Long free text (``body``,
#: ``attachments``) is deliberately left in the JSON: at 20k samples it would
#: add ~35 MB of duplicated corpus to the CSV, and multi-line Persian prose
#: makes the file fragile to read back. ``sample_path`` points at the letter
#: for anything not summarised here.
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "sample_index",
    "sample_path",
    "logo_path",
    "logo_title",
    "logo_categories",
    "subject",
    "destination_organization",
    "letter_number",
    "date",
    "national_id",
    "registration_number",
    "signatory_title",
    "chart_table",
    "letter_genre",
    "letter_tone",
    "body_chars",
    "has_attachments",
    "filled_fields",
    "signature_path",
)


def iter_manifest(path: str | Path) -> Iterator[dict[str, str]]:
    """Stream a manifest CSV row by row.

    Args:
        path: Manifest CSV, e.g. ``data/text/generator.csv``.

    Yields:
        One dict per row, keyed by the CSV header.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    target = resolve_repo_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"manifest not found: {target}")
    with target.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a whole manifest CSV into memory.

    Args:
        path: Manifest CSV, e.g. ``data/text/generator.csv``.

    Returns:
        Every row, in file order.
    """
    return list(iter_manifest(path))
