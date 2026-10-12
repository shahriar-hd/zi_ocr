"""Fetching and caching the lexicon source dumps.

Downloads here are large (tens to hundreds of megabytes) and the network is
the least reliable part of the pipeline, so three properties matter more than
speed:

* **Cached by URL.** A file already in ``cache_dir`` is not re-fetched. Stage
  00 is re-run every time the band cut-offs or the exclusion rank change, and
  re-downloading 400 MB to recompute a percentile is absurd.
* **Resumable.** A partial file is continued with a ``Range`` request rather
  than restarted. A dropped connection at 90% costs seconds.
* **Streamable with an early stop.** The frequency corpus is a 370 MB bz2 of
  Wikipedia articles, but the frequency table converges long before the end of
  it. :func:`stream_bz2_text` decompresses incrementally as bytes arrive and
  lets the caller stop, which turns a 370 MB download into whatever prefix the
  caller actually needed.

Nothing here parses content; see :mod:`fa_ocr_lora.diverse.lexicon.parsers`.
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import logging
import shutil
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Final

__all__ = [
    "SourceFetchError",
    "cache_path_for",
    "download",
    "open_maybe_compressed",
    "stream_bz2_text",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: Read size for streaming. Large enough that the bz2 decompressor gets real
#: work per call, small enough that an early stop wastes little bandwidth.
_CHUNK: Final[int] = 1 << 20

#: Sent on every request. Wikimedia asks for a descriptive agent and throttles
#: anonymous stdlib defaults.
_USER_AGENT: Final[str] = (
    "fa-diverse-ocr/1.0 (Persian OCR dataset build; +https://github.com/anomalyco/opencode)"
)


class SourceFetchError(RuntimeError):
    """Raised when a source cannot be downloaded or decompressed."""


def cache_path_for(url: str, cache_dir: Path) -> Path:
    """Return the deterministic cache location for a URL.

    The basename is kept for readability and a short URL hash is appended, so
    two dumps that happen to share a filename cannot collide.

    Args:
        url: Source URL.
        cache_dir: Directory holding downloaded dumps.

    Returns:
        The path the download will occupy.
    """
    digest = hashlib.blake2b(url.encode("utf-8"), digest_size=4).hexdigest()
    name = url.rstrip("/").rsplit("/", 1)[-1] or "download"
    return cache_dir / f"{digest}-{name}"


def _request(url: str, *, offset: int = 0) -> urllib.request.Request:
    """Build a GET request, optionally resuming from a byte offset.

    Args:
        url: Source URL.
        offset: Byte to resume from; ``0`` requests the whole file.

    Returns:
        The prepared request.
    """
    headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"}
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(url, headers=headers)


def download(url: str, cache_dir: Path, *, force: bool = False) -> Path:
    """Fetch a URL into the cache, resuming a partial download.

    Args:
        url: Source URL.
        cache_dir: Directory holding downloaded dumps.
        force: Re-download even when a complete file is present.

    Returns:
        Path of the cached file.

    Raises:
        SourceFetchError: On any network or filesystem failure.
    """
    target = cache_path_for(url, cache_dir)
    partial = target.with_suffix(target.suffix + ".part")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if target.is_file() and not force:
        LOGGER.info("cached  %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
        return target
    if force and partial.exists():
        partial.unlink()

    offset = partial.stat().st_size if partial.is_file() else 0
    if offset:
        LOGGER.info("resume  %s from %.1f MB", target.name, offset / 1e6)
    else:
        LOGGER.info("fetch   %s", url)

    try:
        with urllib.request.urlopen(
            _request(url, offset=offset), timeout=120
        ) as response:
            # A server that ignores Range replies 200 with the whole body; in
            # that case the partial file must be discarded or the result is a
            # file with a duplicated prefix - which decompresses to garbage
            # only noticed much later.
            if offset and response.status != 206:
                LOGGER.warning("server ignored Range; restarting %s", target.name)
                offset = 0
            mode = "ab" if offset else "wb"
            with partial.open(mode) as handle:
                while True:
                    block = response.read(_CHUNK)
                    if not block:
                        break
                    handle.write(block)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        OSError,
    ) as error:
        raise SourceFetchError(f"{url}: {error}") from error

    partial.replace(target)
    LOGGER.info("done    %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
    return target


def open_maybe_compressed(path: Path) -> Iterator[str]:
    """Yield decoded lines from a plain, gzip or bzip2 file.

    Args:
        path: File to read.

    Yields:
        One decoded line at a time, newline stripped.

    Raises:
        SourceFetchError: If the file cannot be opened or decompressed.
    """
    try:
        if path.suffix == ".gz":
            handle = gzip.open(path, "rt", encoding="utf-8", errors="replace")
        elif path.suffix == ".bz2":
            handle = bz2.open(path, "rt", encoding="utf-8", errors="replace")
        else:
            handle = path.open("rt", encoding="utf-8", errors="replace")
        with handle:
            for line in handle:
                yield line.rstrip("\n")
    except (OSError, EOFError) as error:
        raise SourceFetchError(f"{path}: {error}") from error


def stream_bz2_text(
    url: str, cache_dir: Path, *, use_cache: bool = True
) -> Iterator[str]:
    """Stream a bz2 file as decoded text, decompressing as bytes arrive.

    This is what makes the frequency stage cheap. The caller consumes articles
    and stops when its counts have converged; the generator closing tears down
    the HTTP connection, so the remaining bytes are never transferred. A fully
    cached file is read from disk instead, which is both faster and kinder to
    Wikimedia on a re-run.

    Args:
        url: Source URL of a ``.bz2`` file.
        cache_dir: Directory holding downloaded dumps.
        use_cache: Whether to prefer an already-complete cached file.

    Yields:
        Decoded text chunks, not aligned to line or element boundaries.

    Raises:
        SourceFetchError: On any network or decompression failure.
    """
    target = cache_path_for(url, cache_dir)
    if use_cache and target.is_file():
        LOGGER.info("stream  %s (cached)", target.name)
        decompressor = bz2.BZ2Decompressor()
        try:
            with target.open("rb") as handle:
                while True:
                    block = handle.read(_CHUNK)
                    if not block:
                        break
                    chunk = decompressor.decompress(block)
                    if chunk:
                        yield chunk.decode("utf-8", errors="replace")
        except (OSError, EOFError) as error:
            raise SourceFetchError(f"{target}: {error}") from error
        return

    LOGGER.info("stream  %s", url)
    decompressor = bz2.BZ2Decompressor()
    try:
        with urllib.request.urlopen(_request(url), timeout=120) as response:
            while True:
                block = response.read(_CHUNK)
                if not block:
                    break
                try:
                    chunk = decompressor.decompress(block)
                except (OSError, EOFError, ValueError) as error:
                    raise SourceFetchError(
                        f"{url}: decompression failed ({error})"
                    ) from error
                if chunk:
                    yield chunk.decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        raise SourceFetchError(f"{url}: {error}") from error


def clear_cache(cache_dir: Path) -> int:
    """Delete every cached dump.

    Args:
        cache_dir: Directory holding downloaded dumps.

    Returns:
        Bytes reclaimed.
    """
    if not cache_dir.is_dir():
        return 0
    freed = sum(path.stat().st_size for path in cache_dir.iterdir() if path.is_file())
    shutil.rmtree(cache_dir)
    return freed
