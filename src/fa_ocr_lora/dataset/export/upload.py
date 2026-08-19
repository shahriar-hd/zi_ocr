"""Pushing a finished export to the HuggingFace Hub.

The export stage writes a directory; this module is what turns that directory
into a dataset page. It is deliberately a separate step from
:mod:`.huggingface`: a 40 GB upload should be retryable without re-reading a
single page image, and a corpus should be inspectable locally before it is
published.

**One commit per family, sidecars last.** Each family is pushed as its own
commit, so a repository's history reads ``images``, ``plaintext``, ``pdf`` …
rather than six identical lines, and an interrupted run resumes at family
granularity. ``README.md`` and the other sidecars go up *after* the shards,
because the card's ``configs:`` block is what the Hub's dataset viewer acts on:
publish it first and the viewer spends the whole upload reporting a broken
dataset.

**The token is read, never written and never printed.** It comes from the
environment (:data:`~fa_ocr_lora.config.HF_TOKEN_ENV_VARS`) or from the machine's
existing ``huggingface-cli login``, and nothing here logs it, stores it, or puts
it in a config file. What gets logged is *which* source supplied it.

**Nothing is uploaded that the export did not mean to publish.**
``.export_state.json`` is resume bookkeeping, not data, and is excluded
explicitly rather than by hoping it does not match a pattern.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Sequence

from ...config import HF_TOKEN_ENV_VARS, HuggingFaceConfig, SynthConfig
from .errors import ExportError
from .families import FAMILIES

__all__ = [
    "HUB_URL",
    "SIDECAR_UNIT",
    "UPLOAD_MODES",
    "UploadPlan",
    "UploadSummary",
    "UploadUnit",
    "plan_upload",
    "resolve_token",
    "token_source",
    "upload_export",
]

logger = logging.getLogger(__name__)

#: Where a dataset repository lives, for the log line and the summary.
HUB_URL: Final[str] = "https://huggingface.co/datasets"

#: Name of the pseudo-family covering the card and the other root-level files.
#: Not a real family - it has no Parquet - but it is one upload unit, and giving
#: it a name means the summary and the ``--hf-upload-only`` filter treat it like
#: any other.
SIDECAR_UNIT: Final[str] = "cards"

#: Upload strategies. ``folder`` makes one commit per unit and is right up to a
#: few GB per family. ``large`` uses the Hub's chunked, resumable uploader, which
#: makes many small commits instead and survives an interrupted 40 GB push; it
#: is the documented choice above roughly 20 GB or a few thousand files.
UPLOAD_MODES: Final[tuple[str, ...]] = ("folder", "large")

#: Files that must exist before an upload is allowed to start. Their absence
#: means the export did not finish, and a repository whose shards arrive without
#: a card is one the Hub viewer cannot read.
_REQUIRED_SIDECARS: Final[tuple[str, ...]] = ("README.md", "dataset_info.json")

#: Resume bookkeeping, never published.
_NEVER_UPLOAD: Final[tuple[str, ...]] = (
    "**/.export_state.json",
    ".export_state.json",
    "**/.gitattributes.local",
    "**/.DS_Store",
)

#: Payload above which ``folder`` mode is worth a warning pointing at ``large``.
_LARGE_HINT_BYTES: Final[int] = 20 * 1024**3


def token_source() -> str | None:
    """Return the name of the environment variable holding a Hub token.

    Returns:
        The first variable in :data:`~fa_ocr_lora.config.HF_TOKEN_ENV_VARS` that
        is set and non-empty, or ``None``. The *name*, never the value, so a
        caller can say where the token came from without printing it.
    """
    for name in HF_TOKEN_ENV_VARS:
        if os.environ.get(name, "").strip():
            return name
    return None


def resolve_token(explicit: str | None = None) -> tuple[str, str]:
    """Find a Hub write token.

    Precedence is the same as everywhere else in the pipeline: an explicit
    argument beats the environment, and the environment beats what the machine
    happens to have cached.

    Args:
        explicit: A token passed in directly. Accepted for completeness but not
            wired to a CLI flag on purpose - a token on a command line ends up
            in the shell history and in ``ps`` output.

    Returns:
        ``(token, source)``, where ``source`` is a short label such as
        ``HF_TOKEN`` or ``huggingface-cli login`` for logging.

    Raises:
        ExportError: If no token can be found, with the two ways to provide one.
    """
    if explicit and explicit.strip():
        return explicit.strip(), "argument"
    name = token_source()
    if name is not None:
        return os.environ[name].strip(), name
    cached = _cached_token()
    if cached:
        return cached, "huggingface-cli login"
    raise ExportError(
        "no HuggingFace token found. Either add HF_TOKEN=hf_... to .env "
        "(a write token from https://huggingface.co/settings/tokens) or run "
        "`huggingface-cli login` on this machine. The token is read from the "
        "environment and is never written to a config file."
    )


def _cached_token() -> str | None:
    """Return the token ``huggingface-cli login`` stored, if any."""
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    try:
        return get_token()
    except OSError:  # pragma: no cover - unreadable token cache
        return None


def _require_hub() -> Any:
    """Import ``huggingface_hub``, or explain how to install it.

    Returns:
        The imported module.

    Raises:
        ExportError: If it is not installed.
    """
    try:
        import huggingface_hub
    except ImportError as error:
        raise ExportError(
            "huggingface_hub is required to upload an export; install it with "
            "`pip install 'huggingface_hub>=0.26'` (or `pip install -e '.[hub]'`)"
        ) from error
    return huggingface_hub


@dataclass(frozen=True, slots=True)
class UploadUnit:
    """One commit's worth of files.

    Attributes:
        name: Family name, or :data:`SIDECAR_UNIT` for the root-level files.
        patterns: ``allow_patterns`` selecting this unit's files, relative to the
            export root.
        files: The local files the patterns matched, for the dry run and the
            byte count. Resolved once at planning time so a ``--dry-run`` and
            the real upload describe the same set.
        bytes: Their total size on disk.
    """

    name: str
    patterns: tuple[str, ...]
    files: tuple[Path, ...] = ()
    bytes: int = 0

    @property
    def file_count(self) -> int:
        """How many files this unit carries."""
        return len(self.files)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly view used by the summary."""
        return {
            "name": self.name,
            "patterns": list(self.patterns),
            "files": self.file_count,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class UploadPlan:
    """What an upload will push, and where.

    Attributes:
        repo_id: ``owner/name`` on the Hub.
        root: Local export tree.
        units: Upload units in commit order - families first, sidecars last.
        mode: One of :data:`UPLOAD_MODES`.
        private: Whether a repository *created* by this run is private.
        revision: Branch to commit to; ``None`` means the repository default.
        commit_message: First line of each commit; the unit name is appended.
    """

    repo_id: str
    root: Path
    units: tuple[UploadUnit, ...]
    mode: str = "folder"
    private: bool = False
    revision: str | None = None
    commit_message: str = "Upload synthetic Persian letter corpus"

    @property
    def files(self) -> int:
        """Total files across every unit."""
        return sum(unit.file_count for unit in self.units)

    @property
    def bytes(self) -> int:
        """Total bytes across every unit."""
        return sum(unit.bytes for unit in self.units)

    @property
    def url(self) -> str:
        """The dataset page this plan targets."""
        return f"{HUB_URL}/{self.repo_id}"

    def unit(self, name: str) -> UploadUnit:
        """Return one unit by name.

        Raises:
            KeyError: If the plan has no such unit.
        """
        for entry in self.units:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def describe(self) -> str:
        """Return a one-line summary for a log or a dry run."""
        parts = ", ".join(
            f"{unit.name}={unit.file_count}f/{unit.bytes / 1024**2:.1f}MiB"
            for unit in self.units
        )
        return f"{self.repo_id} [{self.mode}] {parts}"

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly view used by the summary."""
        return {
            "repo_id": self.repo_id,
            "url": self.url,
            "root": str(self.root),
            "mode": self.mode,
            "private": self.private,
            "revision": self.revision,
            "units": [unit.as_dict() for unit in self.units],
            "files": self.files,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class UploadSummary:
    """What an upload did.

    Attributes:
        plan: The plan that was executed.
        uploaded: Unit names that were pushed.
        commits: Commit URLs the Hub returned, where it returned one.
        token_source: Where the token came from - a variable name or
            ``huggingface-cli login``, never the token.
        created: Whether this run created the repository.
        dry_run: Whether anything was actually sent.
    """

    plan: UploadPlan
    uploaded: tuple[str, ...] = ()
    commits: tuple[str, ...] = ()
    token_source: str = ""
    created: bool = False
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly view, safe to log."""
        return {
            "plan": self.plan.as_dict(),
            "uploaded": list(self.uploaded),
            "commits": list(self.commits),
            "token_source": self.token_source,
            "created": self.created,
            "dry_run": self.dry_run,
        }


def _iter_files(root: Path, patterns: Sequence[str]) -> Iterator[Path]:
    """Yield the files under ``root`` matching any of ``patterns``.

    Args:
        root: Export tree.
        patterns: ``glob``-style patterns relative to ``root``.

    Yields:
        Existing files, de-duplicated, in sorted order.
    """
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def _unit(root: Path, name: str, patterns: Sequence[str]) -> UploadUnit:
    """Build one :class:`UploadUnit`, resolving its files now."""
    files = tuple(_iter_files(root, patterns))
    return UploadUnit(
        name=name,
        patterns=tuple(patterns),
        files=files,
        bytes=sum(path.stat().st_size for path in files),
    )


def plan_upload(
    settings: HuggingFaceConfig,
    *,
    repo_id: str | None = None,
    families: Sequence[str] | None = None,
    mode: str = "folder",
    private: bool | None = None,
    revision: str | None = None,
    commit_message: str | None = None,
) -> UploadPlan:
    """Work out what would be pushed, without touching the network.

    Every check that can be made locally is made here, so a dry run catches a
    misconfiguration that would otherwise surface after the first commit.

    Args:
        settings: The ``huggingface`` config section.
        repo_id: Override for ``settings.hub_repo_id``.
        families: Subset of families to push; ``None`` means every family the
            export tree actually holds. :data:`SIDECAR_UNIT` is accepted here to
            push only the card and the other root-level files - which is what a
            fixed typo in the README needs.
        mode: One of :data:`UPLOAD_MODES`.
        private: Override for ``settings.hub_private``.
        revision: Branch to commit to.
        commit_message: Override for ``settings.hub_commit_message``.

    Returns:
        The plan, with every unit's files already resolved.

    Raises:
        ExportError: If the target, the mode, the family names or the export
            tree are unusable.
    """
    target = (repo_id or settings.hub_repo_id).strip()
    if not target:
        raise ExportError(
            "no Hub repository configured. Set FA_OCR_HF_REPO_ID=owner/name in "
            ".env, or pass --hf-repo owner/name. Create the dataset first at "
            "https://huggingface.co/new-dataset, or let --hf-upload create it."
        )
    if target.count("/") != 1 or not all(part.strip() for part in target.split("/")):
        raise ExportError(
            f"Hub repository must be 'owner/name', got {target!r}"
        )
    if mode not in UPLOAD_MODES:
        raise ExportError(
            f"upload mode must be one of {list(UPLOAD_MODES)}, got {mode!r}"
        )
    root = settings.output_dir
    if not root.is_dir():
        raise ExportError(
            f"export tree not found: {root}. Run the export first "
            "(--hugging-face, or --hugging-face-only over an existing render "
            "tree)."
        )
    missing = [name for name in _REQUIRED_SIDECARS if not (root / name).is_file()]
    if missing:
        raise ExportError(
            f"{root} is missing {missing}, so the export did not finish. "
            "Re-run the export before uploading: a repository whose shards "
            "arrive without a dataset card has no working viewer."
        )

    requested = _requested_units(settings, root, families)
    units: list[UploadUnit] = []
    for name in requested:
        if name == SIDECAR_UNIT:
            continue
        unit = _unit(root, name, (f"data/{name}/*.parquet",))
        if not unit.files:
            logger.warning(
                "family %s has no shards under %s; skipping it",
                name,
                root / "data" / name,
            )
            continue
        units.append(unit)
    if SIDECAR_UNIT in requested:
        units.append(_unit(root, SIDECAR_UNIT, _sidecar_patterns()))
    if not units:
        raise ExportError(
            f"nothing to upload from {root}: no shards matched and no sidecars "
            "were selected"
        )

    plan = UploadPlan(
        repo_id=target,
        root=root,
        units=tuple(units),
        mode=mode,
        private=settings.hub_private if private is None else bool(private),
        revision=revision,
        commit_message=(commit_message or settings.hub_commit_message).strip()
        or "Upload dataset",
    )
    if plan.mode == "folder" and plan.bytes > _LARGE_HINT_BYTES:
        logger.warning(
            "%.1f GiB in one commit per family; --hf-upload-mode large is "
            "resumable and is the documented choice at this size",
            plan.bytes / 1024**3,
        )
    return plan


def _requested_units(
    settings: HuggingFaceConfig,
    root: Path,
    families: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve which units to push, in commit order.

    Args:
        settings: The ``huggingface`` config section.
        root: Export tree.
        families: Requested names, or ``None`` for "everything present".

    Returns:
        Family names in registry order, with :data:`SIDECAR_UNIT` last.

    Raises:
        ExportError: If a requested name is neither a family nor the sidecar
            unit.
    """
    if families is None:
        present = [
            name
            for name in settings.families
            if (root / "data" / name).is_dir()
        ]
        # The tree, not the config, decides: an export that ran with a narrower
        # `families` list should still upload cleanly under the committed YAML.
        extra = [
            name
            for name in FAMILIES
            if name not in present and (root / "data" / name).is_dir()
        ]
        return (*present, *extra, SIDECAR_UNIT)
    names: list[str] = []
    for entry in families:
        name = str(entry).strip()
        if name != SIDECAR_UNIT and name not in FAMILIES:
            raise ExportError(
                f"unknown upload unit {name!r}; expected one of "
                f"{[*FAMILIES, SIDECAR_UNIT]}"
            )
        if name not in names:
            names.append(name)
    if not names:
        raise ExportError("no upload units requested")
    ordered = [name for name in FAMILIES if name in names]
    if SIDECAR_UNIT in names:
        ordered.append(SIDECAR_UNIT)
    return tuple(ordered)


def _sidecar_patterns() -> tuple[str, ...]:
    """Return the patterns covering everything that is not a Parquet shard."""
    return (
        "README.md",
        "dataset_info.json",
        "metadata.csv",
        "splits.csv",
        "attribution/*",
    )


def upload_export(
    config: SynthConfig,
    *,
    repo_id: str | None = None,
    families: Sequence[str] | None = None,
    mode: str = "folder",
    private: bool | None = None,
    revision: str | None = None,
    commit_message: str | None = None,
    create: bool = True,
    dry_run: bool = False,
    token: str | None = None,
) -> UploadSummary:
    """Push an export tree to a Hub dataset repository.

    Args:
        config: The whole configuration; only ``config.huggingface`` is read.
        repo_id: Override for ``huggingface.hub_repo_id``.
        families: Which units to push; ``None`` means every family present plus
            the sidecars.
        mode: ``folder`` for one commit per family, ``large`` for the chunked
            resumable uploader.
        private: Whether a repository created by this call is private.
        revision: Branch to commit to; ``None`` uses the repository default.
        commit_message: Override for ``huggingface.hub_commit_message``.
        create: Whether to create the repository when it does not exist.
        dry_run: List what would be pushed and return without a network call.
            The token is still resolved, because "no token" is the failure worth
            finding before a 40 GB upload rather than after it.
        token: An explicit token; normally left ``None`` so the environment is
            used.

    Returns:
        What was pushed.

    Raises:
        ExportError: If the plan is unusable, ``huggingface_hub`` is missing, no
            token can be found, or the Hub rejects the upload.
    """
    settings = config.huggingface
    plan = plan_upload(
        settings,
        repo_id=repo_id,
        families=families,
        mode=mode,
        private=private,
        revision=revision,
        commit_message=commit_message,
    )
    source = ""
    if dry_run:
        # Resolved but unused: a dry run that "passes" without a token would be
        # telling the caller the least useful half of the truth.
        _, source = resolve_token(token)
        logger.info("dry run: would upload %s", plan.describe())
        for unit in plan.units:
            logger.info(
                "  %-11s %4d file(s) %8.1f MiB  %s",
                unit.name,
                unit.file_count,
                unit.bytes / 1024**2,
                ", ".join(unit.patterns),
            )
        return UploadSummary(plan=plan, token_source=source, dry_run=True)

    hub = _require_hub()
    secret, source = resolve_token(token)
    logger.info("uploading to %s (token from %s)", plan.url, source)
    api = hub.HfApi(token=secret)
    created = False
    if create:
        created = _ensure_repo(hub, api, plan)

    uploaded: list[str] = []
    commits: list[str] = []
    for unit in plan.units:
        logger.info(
            "pushing %s: %d file(s), %.1f MiB",
            unit.name,
            unit.file_count,
            unit.bytes / 1024**2,
        )
        result = _push(hub, api, plan, unit)
        uploaded.append(unit.name)
        if result:
            commits.append(result)
    logger.info("upload complete: %s", plan.url)
    return UploadSummary(
        plan=plan,
        uploaded=tuple(uploaded),
        commits=tuple(commits),
        token_source=source,
        created=created,
    )


def _ensure_repo(hub: Any, api: Any, plan: UploadPlan) -> bool:
    """Create the dataset repository if it is not there already.

    Args:
        hub: The ``huggingface_hub`` module.
        api: An ``HfApi`` instance carrying the token.
        plan: The upload plan.

    Returns:
        Whether this call created the repository.

    Raises:
        ExportError: If the Hub refuses - most often a token without write
            access, or a name already taken by another owner.
    """
    existed = True
    try:
        existed = api.repo_exists(repo_id=plan.repo_id, repo_type="dataset")
    except Exception as error:  # noqa: BLE001 - hub errors are a family
        logger.debug("repo_exists check failed, assuming it must be created: %s", error)
        existed = False
    try:
        api.create_repo(
            repo_id=plan.repo_id,
            repo_type="dataset",
            private=plan.private,
            exist_ok=True,
        )
    except Exception as error:  # noqa: BLE001 - hub errors are a family
        raise ExportError(
            f"cannot create or reach dataset {plan.repo_id}: {error}. Check "
            "that the token has write access to that namespace, and that the "
            "owner is you or an organisation you belong to."
        ) from error
    if plan.revision:
        try:
            api.create_branch(
                repo_id=plan.repo_id,
                repo_type="dataset",
                branch=plan.revision,
                exist_ok=True,
            )
        except Exception as error:  # noqa: BLE001 - hub errors are a family
            raise ExportError(
                f"cannot create branch {plan.revision!r} on {plan.repo_id}: "
                f"{error}"
            ) from error
    if not existed:
        logger.info(
            "created dataset %s (%s)",
            plan.repo_id,
            "private" if plan.private else "public",
        )
    return not existed


def _push(hub: Any, api: Any, plan: UploadPlan, unit: UploadUnit) -> str:
    """Upload one unit.

    Args:
        hub: The ``huggingface_hub`` module.
        api: An ``HfApi`` instance carrying the token.
        plan: The upload plan.
        unit: The unit to push.

    Returns:
        The commit URL, or the empty string in ``large`` mode - which commits
        many times and returns nothing to point at.

    Raises:
        ExportError: If the Hub rejects the upload.
    """
    shared: dict[str, Any] = {
        "repo_id": plan.repo_id,
        "repo_type": "dataset",
        "folder_path": str(plan.root),
        "allow_patterns": list(unit.patterns),
        "ignore_patterns": list(_NEVER_UPLOAD),
    }
    try:
        if plan.mode == "large":
            # No commit message and no revision: the chunked uploader owns both,
            # which is the price of it being resumable.
            api.upload_large_folder(**shared, print_report=False)
            return ""
        info = api.upload_folder(
            **shared,
            revision=plan.revision,
            commit_message=f"{plan.commit_message} ({unit.name})",
        )
    except Exception as error:  # noqa: BLE001 - hub errors are a family
        raise ExportError(
            f"upload of {unit.name} to {plan.repo_id} failed: {error}"
        ) from error
    return str(getattr(info, "commit_url", "") or "")
