"""``.env`` loading and typed environment accessors.

Two kinds of setting have no business living in a YAML file that is committed
to a git repository: **secrets** (API keys, OAuth tokens) and **machine
specific paths** (where a particular checkout keeps its ``data/`` tree, where
a training host mounts its output volume). Both belong in the environment, and
in development the environment is most conveniently a ``.env`` file.

The parser here is deliberately dependency-free. ``python-dotenv`` is a fine
library, but the text stage already treats it as optional, and a pipeline that
silently ignores ``.env`` on a host where the package happens to be missing is
a reproducibility bug waiting to happen. Forty lines of parser removes the
branch entirely; :func:`load_dotenv` is what ``python-dotenv`` would have done.

Two rules the parser does not bend:

* **A real environment variable always wins.** ``.env`` is a default for a
  developer's shell, not an override of the deployment's. Passing
  ``override=True`` is possible but nothing in the pipeline does it.
* **A malformed line is an error, not a skip.** A line that does not parse is
  a line the author expected to take effect, and silently dropping it is how a
  20k-sample run ends up pointed at the wrong output directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Mapping

__all__ = [
    "DOTENV_FILENAME",
    "EnvError",
    "parse_dotenv",
    "load_dotenv",
    "ensure_dotenv_loaded",
    "env_str",
    "env_int",
    "env_float",
    "env_bool",
    "env_path",
    "env_list",
]

#: Conventional filename, looked up at the repository root.
DOTENV_FILENAME: Final[str] = ".env"

#: Accepted spellings of a boolean, lowercased.
_TRUE: Final[frozenset[str]] = frozenset({"1", "true", "yes", "y", "on"})
_FALSE: Final[frozenset[str]] = frozenset({"0", "false", "no", "n", "off"})

#: Characters that separate the entries of a list-valued variable. Commas are
#: what the existing ``NVIDIA_API_KEY`` uses; whitespace is accepted too so a
#: long list can be wrapped in a shell export.
_LIST_SEPARATORS: Final[str] = ",\t\n "

#: Escapes honoured inside a double-quoted value, as in a POSIX shell.
_ESCAPES: Final[Mapping[str, str]] = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "\\": "\\",
    '"': '"',
    "'": "'",
}

#: Files already applied by :func:`ensure_dotenv_loaded`, so importing the
#: config from several modules re-reads nothing.
_LOADED: set[Path] = set()


class EnvError(ValueError):
    """Raised when an environment value is missing or cannot be parsed."""


def _unquote(raw: str, *, source: str, line_number: int) -> str:
    """Strip quotes and inline comments from one ``.env`` value.

    Args:
        raw: The text to the right of the first ``=``, already stripped.
        source: File the line came from, for error messages.
        line_number: One-based line number, for error messages.

    Returns:
        The value with surrounding quotes removed, escapes expanded inside
        double quotes, and any unquoted trailing ``# comment`` dropped.

    Raises:
        EnvError: If a quote is opened and never closed.
    """
    if not raw:
        return ""
    quote = raw[0]
    if quote not in "\"'":
        # Unquoted: a ``#`` preceded by whitespace starts a comment. Requiring
        # the whitespace keeps a value that legitimately contains ``#`` - a
        # colour, a URL fragment - intact.
        head, separator, _ = raw.partition(" #")
        return (head if separator else raw).strip()
    closing = raw.find(quote, 1)
    if closing == -1:
        raise EnvError(f"{source}:{line_number}: unterminated {quote} quote")
    body = raw[1:closing]
    if quote == "'":
        # Single quotes are literal, as in a POSIX shell.
        return body
    out: list[str] = []
    index = 0
    while index < len(body):
        character = body[index]
        if character == "\\" and index + 1 < len(body):
            out.append(_ESCAPES.get(body[index + 1], "\\" + body[index + 1]))
            index += 2
            continue
        out.append(character)
        index += 1
    return "".join(out)


def parse_dotenv(text: str, *, source: str = "<string>") -> dict[str, str]:
    """Parse ``.env`` text into a mapping.

    Args:
        text: The whole file.
        source: Name used in error messages.

    Returns:
        Variable name mapped to value, in file order; a repeated name keeps
        its last assignment, matching shell semantics.

    Raises:
        EnvError: On a line that is neither blank, a comment, nor a
            ``KEY=VALUE`` assignment, or on an unterminated quote.
    """
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        name, separator, raw = stripped.partition("=")
        if not separator:
            raise EnvError(
                f"{source}:{line_number}: expected KEY=VALUE, got {line.strip()!r}"
            )
        name = name.strip()
        if not name:
            raise EnvError(f"{source}:{line_number}: empty variable name")
        values[name] = _unquote(raw.strip(), source=source, line_number=line_number)
    return values


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Read a ``.env`` file into :data:`os.environ`.

    Args:
        path: File to read; defaults to :data:`DOTENV_FILENAME` at the
            repository root. A missing file is not an error - most hosts do not
            have one - but a malformed one is.
        override: Whether a value in the file replaces one already exported in
            the process environment. Left ``False`` everywhere in the pipeline,
            so a deployment's real environment always wins.

    Returns:
        The parsed mapping, whether or not each entry was applied. Empty when
        the file does not exist.

    Raises:
        EnvError: If the file exists but does not parse.
    """
    target = Path(path) if path is not None else _default_dotenv_path()
    if not target.is_file():
        return {}
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as error:
        raise EnvError(f"cannot read {target}: {error}") from error
    values = parse_dotenv(text, source=str(target))
    for name, value in values.items():
        if override or name not in os.environ:
            os.environ[name] = value
    return values


def ensure_dotenv_loaded(path: str | Path | None = None) -> None:
    """Apply a ``.env`` file once per process.

    :func:`load_config` calls this before it reads any environment override,
    so every entry point - the CLI, a test, a notebook importing the package -
    sees the same variables without each having to remember to load them.

    Args:
        path: File to read; defaults to the repository root's ``.env``.
    """
    target = Path(path) if path is not None else _default_dotenv_path()
    resolved = target.expanduser()
    if resolved in _LOADED:
        return
    _LOADED.add(resolved)
    load_dotenv(resolved)


def _default_dotenv_path() -> Path:
    """Return the repository root's ``.env``, wherever the caller ran from."""
    from ..dataset.common.io_utils import repo_root

    return repo_root() / DOTENV_FILENAME


def env_str(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Return an environment variable as a string.

    Args:
        name: Variable name.
        default: Value to use when the variable is unset or empty.
        required: Whether an unset variable is an error rather than a default.

    Returns:
        The value, stripped of surrounding whitespace, or ``default``.

    Raises:
        EnvError: If ``required`` and the variable is unset or empty.
    """
    raw = os.environ.get(name, "").strip()
    if raw:
        return raw
    if required:
        raise EnvError(f"required environment variable {name} is not set")
    return default


def env_int(name: str, default: int | None = None) -> int | None:
    """Return an environment variable as an integer.

    Args:
        name: Variable name.
        default: Value to use when the variable is unset.

    Returns:
        The parsed integer, or ``default``.

    Raises:
        EnvError: If the variable is set but is not an integer.
    """
    raw = env_str(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise EnvError(f"{name}: expected an integer, got {raw!r}") from error


def env_float(name: str, default: float | None = None) -> float | None:
    """Return an environment variable as a float.

    Args:
        name: Variable name.
        default: Value to use when the variable is unset.

    Returns:
        The parsed float, or ``default``.

    Raises:
        EnvError: If the variable is set but is not a number.
    """
    raw = env_str(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise EnvError(f"{name}: expected a number, got {raw!r}") from error


def env_bool(name: str, default: bool | None = None) -> bool | None:
    """Return an environment variable as a boolean.

    Args:
        name: Variable name.
        default: Value to use when the variable is unset.

    Returns:
        The parsed boolean, or ``default``.

    Raises:
        EnvError: If the variable is set to something that is neither a
            recognised truth value nor a recognised false value. ``FOO=maybe``
            is a typo, and defaulting it either way hides the typo.
    """
    raw = env_str(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise EnvError(
        f"{name}: expected a boolean "
        f"({'/'.join(sorted(_TRUE))} or {'/'.join(sorted(_FALSE))}), got {raw!r}"
    )


def env_path(name: str, default: Path | None = None) -> Path | None:
    """Return an environment variable as a filesystem path.

    ``~`` is expanded; the path is *not* resolved against the repository root,
    because the caller knows whether that is the right anchor - the config's
    ``paths`` section runs its own values through
    :func:`~fa_ocr_lora.dataset.common.io_utils.resolve_repo_path`, while
    ``fonts_dir`` is deliberately a host path.

    Args:
        name: Variable name.
        default: Value to use when the variable is unset.

    Returns:
        The expanded path, or ``default``.
    """
    raw = env_str(name)
    if raw is None:
        return default
    return Path(raw).expanduser()


def env_list(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Return an environment variable as a tuple of entries.

    Entries are separated by commas or whitespace, so one variable can hold
    several values - which is how ``NVIDIA_API_KEY`` already carries a pool of
    keys - without needing numbered variants.

    Args:
        name: Variable name.
        default: Value to use when the variable is unset or holds no entries.

    Returns:
        The entries in order, with empties dropped.
    """
    raw = env_str(name)
    if raw is None:
        return default
    entries = tuple(
        part for part in (item.strip() for item in _split(raw)) if part
    )
    return entries or default


def _split(raw: str) -> list[str]:
    """Split a list-valued variable on any of :data:`_LIST_SEPARATORS`."""
    out: list[str] = []
    current: list[str] = []
    for character in raw:
        if character in _LIST_SEPARATORS:
            out.append("".join(current))
            current = []
            continue
        current.append(character)
    out.append("".join(current))
    return out
