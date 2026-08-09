#!/usr/bin/env python3
"""Download a unique subset of brand logos from the theSVG icon repository.

`glincker/thesvg <https://github.com/glincker/thesvg>`_ (MIT) ships ~6.5k brands
as SVG under ``public/icons/<slug>/<variant>.svg``, catalogued by
``src/data/icons.json``. The synthetic-letter pipeline needs a pool of
transparent logo marks to drop into generated headers
(``dataset/synth/layout/header_footer.py``), so this script materialises a
deterministic, deduplicated slice of that repository locally.

The whole repository is fetched **once** as a ~11 MB tarball pinned to a commit
SHA, rather than issuing one HTTP request per logo. Both representations are
kept for every accepted logo:

* ``svg/<slug>.svg`` — the original vector, a few KB each, re-renderable at any
  resolution or tint later on;
* ``png/<slug>.png`` — an RGBA raster (default 120x120, fully transparent
  background) that the page composer can paste directly.

Uniqueness is enforced at three levels, so no brand and no artwork repeats:

1. one variant per brand slug (``dark``/``light``/``wordmark`` duplicates of the
   same brand are never both taken);
2. SHA-256 of the SVG source, catching brands that share byte-identical art;
3. SHA-256 of the rendered raster, catching art that differs in source but is
   visually identical once rasterised.

Because the pages are composited on white paper, logos whose opaque pixels are
essentially white are rejected as invisible (see ``--max-luminance``), as are
degenerate (blank) and extremely elongated renders (see ``--max-aspect``).

The catalogue mixes open licences with trademark-only and no-derivatives
artwork. ``--permissive`` (or an explicit ``--license``) restricts the run to
redistributable brands, and every emitted file is credited in
``attribution.csv`` with a commit-pinned permalink back to its upstream source.

Selection is a seeded shuffle of the catalogue, not the alphabetical head, so a
1,000-logo sample is spread across the whole brand list; the same ``--seed``
plus the same commit SHA always yields the same set.

Output layout under ``--output-dir`` (default ``data/thesvg``)::

    data/thesvg/
    |-- svg/<slug>.svg           original vector artwork
    |-- png/<slug>.png           RGBA raster, transparent background
    |-- logos.csv                manifest consumed by the synth pipeline
    |-- attribution.csv          per-file credit: source, licence, author
    `-- download_report.json     summary of the last run

Examples:
    Smoke test on a handful of logos::

        python scripts/download_thesvg_logo.py --limit 20

    Default run (1,000 logos at 120x120)::

        python scripts/download_thesvg_logo.py

    Every openly-licensed brand (CC0-1.0, MIT, Apache-2.0)::

        python scripts/download_thesvg_logo.py --permissive --limit 0

    Larger, higher-resolution pool restricted to two categories::

        python scripts/download_thesvg_logo.py --limit 3000 --size 256 \\
            --category AI --category Software

Requires ``cairosvg`` and ``Pillow``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import random
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Sequence

REPO_ID: Final[str] = "glincker/thesvg"
CATALOG_PATH: Final[str] = "src/data/icons.json"
ICON_ROOT: Final[str] = "public"

#: Commit metadata endpoint, used to pin a branch to an immutable SHA.
COMMIT_API_URL: Final[str] = "https://api.github.com/repos/{repo}/commits/{ref}"
TARBALL_URL: Final[str] = "https://codeload.github.com/{repo}/tar.gz/{ref}"

USER_AGENT: Final[str] = "fa-ocr-lora-logo-downloader/1.0"
NETWORK_TIMEOUT: Final[int] = 120

#: Variant names tried in order. Icon marks beat wordmarks (a wordmark is a
#: block of Latin text, which would pollute Persian page transcriptions), and
#: colour beats monochrome because ``currentColor`` art rasterises to flat
#: black. ``dark``/``light`` are deliberately absent: they are per-theme
#: duplicates of the same mark.
VARIANT_PREFERENCE: Final[tuple[str, ...]] = ("default", "color", "mono", "line", "icon")

#: Column order of the generated ``logos.csv``.
CSV_FIELDNAMES: Final[tuple[str, ...]] = (
    "logo_id",
    "slug",
    "title",
    "variant",
    "svg_path",
    "png_path",
    "categories",
    "license",
    "brand_url",
    "brand_hex",
    "content_width",
    "content_height",
    "aspect_ratio",
    "mean_luminance",
    "svg_sha256",
    "png_sha256",
    "source_path",
)

#: Column order of ``attribution.csv``, the credit file that travels with the
#: artwork. Deliberately human-readable and independent of the machine-facing
#: manifest, so it can be shipped verbatim alongside a published dataset.
ATTRIBUTION_FIELDNAMES: Final[tuple[str, ...]] = (
    "File_Name",
    "Source",
    "License",
    "Author",
    "Original_File",
)

#: Licences considered redistributable for a public dataset. Passed to
#: ``--license`` to exclude trademark-only and no-derivatives artwork.
PERMISSIVE_LICENSES: Final[tuple[str, ...]] = ("CC0-1.0", "MIT", "Apache-2.0")

#: Permalink to a file, pinned to the commit the artwork was taken from.
SOURCE_URL: Final[str] = "https://github.com/{repo}/blob/{commit}/{path}"

DEFAULT_OUTPUT_DIR: Final[str] = "data/thesvg"
DEFAULT_LIMIT: Final[int] = 1000
DEFAULT_SIZE: Final[int] = 120
DEFAULT_SEED: Final[int] = 1337

SVG_DIRNAME: Final[str] = "svg"
PNG_DIRNAME: Final[str] = "png"
CSV_FILENAME: Final[str] = "logos.csv"
ATTRIBUTION_FILENAME: Final[str] = "attribution.csv"
REPORT_FILENAME: Final[str] = "download_report.json"
TARBALL_FILENAME: Final[str] = "thesvg-{ref}.tar.gz"

#: The SVG is rasterised at ``size * SUPERSAMPLE`` and downscaled with LANCZOS,
#: which keeps thin strokes legible at 120 px.
SUPERSAMPLE: Final[int] = 4

#: Opaque-pixel threshold for the alpha channel when measuring luminance.
ALPHA_THRESHOLD: Final[int] = 32

#: Rec. 709 luma coefficients, used to reject near-white (invisible) marks.
LUMA_WEIGHTS: Final[tuple[float, float, float]] = (0.2126, 0.7152, 0.0722)

#: Trimmed artwork thinner than this (in supersampled pixels) is degenerate.
MIN_CONTENT_PX: Final[int] = 8

LOGGER: Final[logging.Logger] = logging.getLogger("download_thesvg_logo")


class LogoDownloadError(RuntimeError):
    """Raised when the logo set cannot be downloaded or rendered."""


@dataclass(frozen=True)
class DownloadConfig:
    """Runtime configuration for one download run.

    Attributes:
        output_dir: Directory that receives the SVGs, PNGs and CSV.
        ref: Git ref (branch, tag or commit SHA) of the theSVG repository.
        limit: Number of unique logos to keep; ``None`` means every candidate.
        size: Edge length in pixels of the square PNG canvas.
        margin: Transparent padding in pixels kept on each side of the canvas.
        seed: RNG seed for the deterministic catalogue shuffle.
        categories: Keep only brands tagged with one of these categories;
            empty means no category filter.
        collections: Keep only brands from these catalogue collections; empty
            means no collection filter.
        licenses: Keep only brands whose catalogue licence matches one of
            these (compared case-insensitively); empty means no licence
            filter. Use this to exclude trademark-restricted artwork before
            redistributing a dataset.
        max_aspect: Reject artwork wider or taller than this ratio.
        max_luminance: Reject artwork whose mean opaque luminance exceeds this
            (0-1) threshold, i.e. marks invisible on white paper.
        cache_dir: Directory holding the downloaded tarball; defaults to
            ``<output_dir>/.cache``.
        overwrite: Re-render logos even when their output files already exist.
        dry_run: Report what would be produced, then exit.
    """

    output_dir: Path
    ref: str = "main"
    limit: int | None = DEFAULT_LIMIT
    size: int = DEFAULT_SIZE
    margin: int = 4
    seed: int = DEFAULT_SEED
    categories: tuple[str, ...] = ()
    collections: tuple[str, ...] = ()
    licenses: tuple[str, ...] = ()
    max_aspect: float = 6.0
    max_luminance: float = 0.97
    cache_dir: Path | None = None
    overwrite: bool = False
    dry_run: bool = False

    @property
    def svg_dir(self) -> Path:
        """Directory holding the original vector artwork."""
        return self.output_dir / SVG_DIRNAME

    @property
    def png_dir(self) -> Path:
        """Directory holding the rasterised logos."""
        return self.output_dir / PNG_DIRNAME

    @property
    def csv_path(self) -> Path:
        """Path of the generated logo manifest."""
        return self.output_dir / CSV_FILENAME

    @property
    def attribution_path(self) -> Path:
        """Path of the generated attribution/credit file."""
        return self.output_dir / ATTRIBUTION_FILENAME

    @property
    def report_path(self) -> Path:
        """Path of the machine-readable run report."""
        return self.output_dir / REPORT_FILENAME

    @property
    def tarball_dir(self) -> Path:
        """Directory the repository tarball is cached in."""
        return self.cache_dir or (self.output_dir / ".cache")


@dataclass(frozen=True)
class CatalogEntry:
    """One brand selected from ``icons.json``.

    Attributes:
        slug: Catalogue identifier, also the output file stem.
        title: Human-readable brand name.
        variant: Name of the chosen variant, e.g. ``"default"``.
        source_path: Repo-relative path of the SVG, without the tarball prefix.
        categories: Catalogue categories the brand is tagged with.
        license: SPDX-ish licence string reported by the catalogue.
        brand_url: Official brand URL, for attribution.
        brand_hex: Primary brand colour as an ``RRGGBB`` string.
    """

    slug: str
    title: str
    variant: str
    source_path: str
    categories: tuple[str, ...]
    license: str
    brand_url: str
    brand_hex: str


@dataclass(frozen=True)
class RenderedLogo:
    """A rasterised logo plus the metrics used to accept or reject it.

    Attributes:
        png_bytes: Encoded RGBA PNG of the square canvas.
        raster_sha256: Digest of the raw pixel buffer, used for dedup.
        content_width: Width of the trimmed artwork on the final canvas.
        content_height: Height of the trimmed artwork on the final canvas.
        aspect_ratio: Long edge divided by short edge of the trimmed artwork.
        mean_luminance: Mean Rec. 709 luma of the opaque pixels, in ``[0, 1]``.
    """

    png_bytes: bytes
    raster_sha256: str
    content_width: int
    content_height: int
    aspect_ratio: float
    mean_luminance: float


@dataclass(frozen=True)
class AcceptedLogo:
    """A logo that passed every filter and was written to disk.

    Attributes:
        entry: Catalogue entry the logo came from.
        render: Metrics of the accepted render.
        svg_sha256: SHA-256 of the SVG source.
    """

    entry: CatalogEntry
    render: RenderedLogo
    svg_sha256: str


@dataclass
class RunCounters:
    """Tallies describing why candidates were kept or dropped.

    Attributes:
        written: Logos written to disk.
        reused: Logos already present on disk and left untouched.
        skipped_duplicate_svg: Dropped because the SVG source repeated.
        skipped_duplicate_render: Dropped because the raster repeated.
        skipped_blank: Dropped because the render had no opaque pixels.
        skipped_aspect: Dropped for exceeding ``--max-aspect``.
        skipped_luminance: Dropped for exceeding ``--max-luminance``.
        skipped_render_error: Dropped because rasterisation raised.
        skipped_missing: Dropped because the catalogue path was not in the tar.
    """

    written: int = 0
    reused: int = 0
    skipped_duplicate_svg: int = 0
    skipped_duplicate_render: int = 0
    skipped_blank: int = 0
    skipped_aspect: int = 0
    skipped_luminance: int = 0
    skipped_render_error: int = 0
    skipped_missing: int = 0

    @property
    def accepted(self) -> int:
        """Number of logos present in the manifest after this run."""
        return self.written + self.reused


def configure_logging(verbose: bool) -> None:
    """Set up console logging.

    Args:
        verbose: Emit ``DEBUG`` records instead of ``INFO``.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _require_cairosvg() -> Any:
    """Import ``cairosvg``, or fail with an actionable message.

    Returns:
        The imported ``cairosvg`` module.

    Raises:
        LogoDownloadError: If cairosvg is not installed.
    """
    try:
        import cairosvg
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LogoDownloadError(
            "cairosvg is required to rasterise the logos. Install it with: "
            "pip install cairosvg  (it needs the system libcairo2 package)"
        ) from exc
    return cairosvg


def _require_pillow() -> Any:
    """Import ``PIL.Image``, or fail with an actionable message.

    Returns:
        The imported ``PIL.Image`` module.

    Raises:
        LogoDownloadError: If Pillow is not installed.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LogoDownloadError(
            "Pillow is required to post-process the logos. Install it with: "
            "pip install Pillow"
        ) from exc
    return Image


def format_size(num_bytes: int) -> str:
    """Render a byte count as a human-readable string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        The size formatted with a binary unit suffix, e.g. ``"1.5 GiB"``.
    """
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _http_get(url: str) -> bytes:
    """Fetch a URL and return its body.

    Args:
        url: Absolute HTTP(S) URL.

    Returns:
        The response body.

    Raises:
        LogoDownloadError: If the request fails.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT) as response:
            return response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise LogoDownloadError(f"Request to {url} failed: {exc}") from exc


def resolve_commit(ref: str) -> str:
    """Resolve a branch or tag to an immutable commit SHA.

    Pinning the SHA keeps a download reproducible even if theSVG later adds or
    removes brands.

    Args:
        ref: Branch, tag or commit SHA to resolve.

    Returns:
        The resolved commit SHA, or ``ref`` itself if resolution fails.
    """
    url = COMMIT_API_URL.format(repo=REPO_ID, ref=ref)
    try:
        payload = json.loads(_http_get(url))
    except (LogoDownloadError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not resolve ref %r: %s", ref, exc)
        return ref
    sha = payload.get("sha") if isinstance(payload, dict) else None
    return str(sha) if sha else ref


def download_tarball(ref: str, cache_dir: Path) -> Path:
    """Download the repository tarball, reusing a cached copy when present.

    Args:
        ref: Commit SHA (or branch) to download.
        cache_dir: Directory the tarball is stored in.

    Returns:
        Local path of the tarball.

    Raises:
        LogoDownloadError: If the download fails or yields an unreadable file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / TARBALL_FILENAME.format(ref=ref)
    if target.is_file() and target.stat().st_size > 0:
        LOGGER.info("Reusing cached tarball %s (%s)", target, format_size(target.stat().st_size))
        return target

    url = TARBALL_URL.format(repo=REPO_ID, ref=ref)
    LOGGER.info("Downloading %s", url)
    payload = _http_get(url)
    if not payload:
        raise LogoDownloadError(f"Tarball at {url} was empty.")

    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(target)
    LOGGER.info("Fetched tarball %s (%s)", target, format_size(len(payload)))
    return target


def _strip_prefix(member_name: str) -> str:
    """Drop the ``thesvg-<ref>/`` prefix GitHub adds to tarball members.

    Args:
        member_name: Member name as stored in the archive.

    Returns:
        The repo-relative path.
    """
    _, separator, remainder = member_name.partition("/")
    return remainder if separator else member_name


def index_archive(tarball: Path) -> dict[str, tarfile.TarInfo]:
    """Build a repo-relative index of the regular files in the tarball.

    Members are indexed rather than extracted, so nothing outside the output
    directory is ever written and no path-traversal entry can take effect.

    Args:
        tarball: Local path of the repository tarball.

    Returns:
        Mapping of repo-relative path to its archive member.

    Raises:
        LogoDownloadError: If the archive cannot be read.
    """
    try:
        with tarfile.open(tarball, mode="r:gz") as archive:
            return {
                _strip_prefix(member.name): member
                for member in archive.getmembers()
                if member.isfile()
            }
    except (tarfile.TarError, OSError) as exc:
        raise LogoDownloadError(f"Cannot read tarball {tarball}: {exc}") from exc


def read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    """Read one member out of an open archive.

    Args:
        archive: Open tar archive.
        member: Member to read.

    Returns:
        The member's bytes.

    Raises:
        LogoDownloadError: If the member cannot be read.
    """
    handle = archive.extractfile(member)
    if handle is None:
        raise LogoDownloadError(f"Archive member {member.name} is not a regular file.")
    with handle:
        return handle.read()


def load_catalog(archive: tarfile.TarFile, index: dict[str, tarfile.TarInfo]) -> list[dict[str, Any]]:
    """Read and parse ``src/data/icons.json`` from the archive.

    Args:
        archive: Open tar archive.
        index: Repo-relative member index.

    Returns:
        The catalogue entries as raw dictionaries.

    Raises:
        LogoDownloadError: If the catalogue is missing or malformed.
    """
    member = index.get(CATALOG_PATH)
    if member is None:
        raise LogoDownloadError(f"{CATALOG_PATH} is missing from the repository tarball.")
    try:
        payload = json.loads(read_member(archive, member))
    except json.JSONDecodeError as exc:
        raise LogoDownloadError(f"{CATALOG_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise LogoDownloadError(f"{CATALOG_PATH} should hold a list, got {type(payload).__name__}.")
    return [entry for entry in payload if isinstance(entry, dict)]


def select_variant(variants: dict[str, Any]) -> tuple[str, str] | None:
    """Pick the single best variant for a brand.

    Args:
        variants: The catalogue's ``variants`` mapping, name to repo-absolute
            path such as ``"/icons/astro/default.svg"``.

    Returns:
        The chosen ``(variant_name, repo_relative_path)``, or ``None`` when the
        brand exposes no usable variant.
    """
    for name in VARIANT_PREFERENCE:
        raw = variants.get(name)
        if isinstance(raw, str) and raw.strip():
            return name, f"{ICON_ROOT}/{raw.lstrip('/')}"
    return None


def build_candidates(catalog: Sequence[dict[str, Any]], config: DownloadConfig) -> list[CatalogEntry]:
    """Turn raw catalogue rows into a deterministic, shuffled candidate list.

    One entry is produced per brand, so a brand can never appear twice in the
    output regardless of how many variants it ships.

    Args:
        catalog: Parsed ``icons.json`` rows.
        config: Active run configuration.

    Returns:
        Candidates in the order they should be attempted.
    """
    wanted_categories = {item.casefold() for item in config.categories}
    wanted_collections = {item.casefold() for item in config.collections}
    wanted_licenses = {item.casefold() for item in config.licenses}

    candidates: list[CatalogEntry] = []
    seen_slugs: set[str] = set()

    for row in catalog:
        slug = str(row.get("slug") or "").strip()
        if not slug or slug in seen_slugs:
            continue

        license_name = str(row.get("license") or "").strip()
        if wanted_licenses and license_name.casefold() not in wanted_licenses:
            continue

        collection = str(row.get("collection") or "")
        if wanted_collections and collection.casefold() not in wanted_collections:
            continue

        categories = tuple(str(item) for item in row.get("categories") or ())
        if wanted_categories and not {item.casefold() for item in categories} & wanted_categories:
            continue

        variants = row.get("variants")
        if not isinstance(variants, dict):
            continue
        chosen = select_variant(variants)
        if chosen is None:
            continue

        seen_slugs.add(slug)
        variant_name, source_path = chosen
        candidates.append(
            CatalogEntry(
                slug=slug,
                title=str(row.get("title") or slug),
                variant=variant_name,
                source_path=source_path,
                categories=categories,
                license=license_name,
                brand_url=str(row.get("url") or ""),
                brand_hex=str(row.get("hex") or ""),
            )
        )

    candidates.sort(key=lambda entry: entry.slug)
    random.Random(config.seed).shuffle(candidates)
    return candidates


def _measure_luminance(image: Any) -> float:
    """Compute the mean Rec. 709 luma of an image's opaque pixels.

    Args:
        image: Trimmed ``RGBA`` image.

    Returns:
        Mean luminance in ``[0, 1]``; ``0.0`` when nothing is opaque enough.
    """
    red, green, blue, alpha = image.split()
    opaque = sum(alpha.histogram()[ALPHA_THRESHOLD:])
    if opaque == 0:
        return 0.0

    mask = alpha.point(lambda value: 255 if value >= ALPHA_THRESHOLD else 0)
    weighted = 0.0
    for channel, weight in zip((red, green, blue), LUMA_WEIGHTS):
        histogram = channel.histogram(mask)
        weighted += weight * sum(level * hits for level, hits in enumerate(histogram))
    return weighted / (opaque * 255.0)


def render_logo(svg_bytes: bytes, config: DownloadConfig) -> RenderedLogo | None:
    """Rasterise an SVG onto a square, fully transparent canvas.

    The artwork is rendered supersampled, trimmed to its alpha bounding box so
    that every logo fills the canvas consistently regardless of the padding
    baked into its ``viewBox``, then scaled to fit and centred.

    Args:
        svg_bytes: Raw SVG source.
        config: Active run configuration.

    Returns:
        The rendered logo, or ``None`` when the render is blank.

    Raises:
        LogoDownloadError: If cairosvg or Pillow is unavailable.
    """
    cairosvg = _require_cairosvg()
    image_module = _require_pillow()

    edge = config.size * SUPERSAMPLE
    raw = cairosvg.svg2png(bytestring=svg_bytes, output_width=edge, output_height=edge)
    rendered = image_module.open(io.BytesIO(raw)).convert("RGBA")

    bbox = rendered.getbbox()
    if bbox is None:
        return None
    trimmed = rendered.crop(bbox)
    if min(trimmed.size) < MIN_CONTENT_PX:
        return None

    inner = max(1, config.size - 2 * config.margin)
    scale = min(inner / trimmed.width, inner / trimmed.height)
    target = (max(1, round(trimmed.width * scale)), max(1, round(trimmed.height * scale)))
    resized = trimmed.resize(target, image_module.LANCZOS)

    canvas = image_module.new("RGBA", (config.size, config.size), (0, 0, 0, 0))
    canvas.paste(resized, ((config.size - target[0]) // 2, (config.size - target[1]) // 2))

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)

    long_edge, short_edge = max(trimmed.size), min(trimmed.size)
    return RenderedLogo(
        png_bytes=buffer.getvalue(),
        raster_sha256=hashlib.sha256(canvas.tobytes()).hexdigest(),
        content_width=target[0],
        content_height=target[1],
        aspect_ratio=long_edge / short_edge,
        mean_luminance=_measure_luminance(trimmed),
    )


def _prepare_output_dirs(config: DownloadConfig) -> None:
    """Create the output directory tree.

    Args:
        config: Active run configuration.
    """
    config.svg_dir.mkdir(parents=True, exist_ok=True)
    config.png_dir.mkdir(parents=True, exist_ok=True)


def _csv_row(entry: CatalogEntry, render: RenderedLogo, svg_digest: str) -> dict[str, Any]:
    """Build one manifest row.

    Args:
        entry: Catalogue entry the logo came from.
        render: Metrics of the accepted render.
        svg_digest: SHA-256 of the SVG source.

    Returns:
        A mapping keyed by :data:`CSV_FIELDNAMES`.
    """
    return {
        "logo_id": f"thesvg_{entry.slug}",
        "slug": entry.slug,
        "title": entry.title,
        "variant": entry.variant,
        "svg_path": f"{SVG_DIRNAME}/{entry.slug}.svg",
        "png_path": f"{PNG_DIRNAME}/{entry.slug}.png",
        "categories": "|".join(entry.categories),
        "license": entry.license,
        "brand_url": entry.brand_url,
        "brand_hex": entry.brand_hex,
        "content_width": render.content_width,
        "content_height": render.content_height,
        "aspect_ratio": round(render.aspect_ratio, 4),
        "mean_luminance": round(render.mean_luminance, 4),
        "svg_sha256": svg_digest,
        "png_sha256": render.raster_sha256,
        "source_path": entry.source_path,
    }


def _attribution_rows(entry: CatalogEntry, commit: str) -> list[dict[str, Any]]:
    """Build the credit rows for one logo, one per emitted file.

    Both the SVG and its rendered PNG are derivatives of the same upstream
    file, so each is credited separately and points back at the same pinned
    permalink.

    Args:
        entry: Catalogue entry the logo came from.
        commit: Commit SHA the artwork was taken from.

    Returns:
        Mappings keyed by :data:`ATTRIBUTION_FIELDNAMES`.
    """
    source = SOURCE_URL.format(repo=REPO_ID, commit=commit, path=entry.source_path)
    #: theSVG records no per-icon author; for brand marks the rights holder is
    #: the brand itself, so the brand name is the correct credit.
    author = entry.title
    return [
        {
            "File_Name": f"{SVG_DIRNAME}/{entry.slug}.svg",
            "Source": source,
            "License": entry.license,
            "Author": author,
            "Original_File": entry.source_path,
        },
        {
            "File_Name": f"{PNG_DIRNAME}/{entry.slug}.png",
            "Source": source,
            "License": entry.license,
            "Author": author,
            "Original_File": entry.source_path,
        },
    ]


def _iter_accepted(
    archive: tarfile.TarFile,
    index: dict[str, tarfile.TarInfo],
    candidates: Sequence[CatalogEntry],
    config: DownloadConfig,
    counters: RunCounters,
) -> Iterator[AcceptedLogo]:
    """Render candidates in order, yielding every logo that passes the filters.

    Args:
        archive: Open tar archive holding the SVG sources.
        index: Repo-relative member index.
        candidates: Shuffled candidate list.
        config: Active run configuration.
        counters: Tallies updated in place as candidates are judged.

    Yields:
        One record per accepted logo.
    """
    seen_svg: set[str] = set()
    seen_raster: set[str] = set()
    log_every = max(1, (config.limit or len(candidates)) // 10)

    for entry in candidates:
        if config.limit is not None and counters.accepted >= config.limit:
            break

        member = index.get(entry.source_path)
        if member is None:
            counters.skipped_missing += 1
            LOGGER.debug("%s: %s not in tarball", entry.slug, entry.source_path)
            continue

        svg_bytes = read_member(archive, member)
        svg_digest = hashlib.sha256(svg_bytes).hexdigest()
        if svg_digest in seen_svg:
            counters.skipped_duplicate_svg += 1
            LOGGER.debug("%s: duplicate SVG source", entry.slug)
            continue

        try:
            render = render_logo(svg_bytes, config)
        except LogoDownloadError:
            raise
        except Exception as exc:  # noqa: BLE001 - cairosvg raises many parse errors
            counters.skipped_render_error += 1
            LOGGER.debug("%s: render failed (%s: %s)", entry.slug, type(exc).__name__, exc)
            continue

        if render is None:
            counters.skipped_blank += 1
            LOGGER.debug("%s: render is blank", entry.slug)
            continue
        if render.aspect_ratio > config.max_aspect:
            counters.skipped_aspect += 1
            LOGGER.debug("%s: aspect ratio %.2f too extreme", entry.slug, render.aspect_ratio)
            continue
        if render.mean_luminance > config.max_luminance:
            counters.skipped_luminance += 1
            LOGGER.debug("%s: too light (luma %.3f) for white paper", entry.slug, render.mean_luminance)
            continue
        if render.raster_sha256 in seen_raster:
            counters.skipped_duplicate_render += 1
            LOGGER.debug("%s: duplicate render", entry.slug)
            continue

        seen_svg.add(svg_digest)
        seen_raster.add(render.raster_sha256)

        svg_path = config.svg_dir / f"{entry.slug}.svg"
        png_path = config.png_dir / f"{entry.slug}.png"
        if not config.overwrite and svg_path.is_file() and png_path.is_file():
            counters.reused += 1
        else:
            svg_path.write_bytes(svg_bytes)
            png_path.write_bytes(render.png_bytes)
            counters.written += 1

        if counters.accepted % log_every == 0:
            LOGGER.info("  %d logo(s) accepted (latest: %s)", counters.accepted, entry.slug)

        yield AcceptedLogo(entry=entry, render=render, svg_sha256=svg_digest)


def _write_report(
    config: DownloadConfig,
    commit: str,
    counters: RunCounters,
    candidates_total: int,
) -> None:
    """Write a JSON summary of the run.

    Args:
        config: Active run configuration.
        commit: Resolved commit SHA the artwork came from.
        counters: Final tallies of the run.
        candidates_total: Number of brands considered before rendering.
    """
    report = {
        "repo_id": REPO_ID,
        "commit": commit,
        "seed": config.seed,
        "png_size": config.size,
        "png_margin": config.margin,
        "variant_preference": list(VARIANT_PREFERENCE),
        "categories_filter": list(config.categories),
        "collections_filter": list(config.collections),
        "licenses_filter": list(config.licenses),
        "max_aspect": config.max_aspect,
        "max_luminance": config.max_luminance,
        "candidates_total": candidates_total,
        "requested_limit": config.limit,
        "counters": asdict(counters),
        "logos_accepted": counters.accepted,
        "manifest": CSV_FILENAME,
        "attribution": ATTRIBUTION_FILENAME,
        "svg_dir": SVG_DIRNAME,
        "png_dir": PNG_DIRNAME,
    }
    config.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def download_thesvg_logos(config: DownloadConfig) -> RunCounters:
    """Download, deduplicate and rasterise a slice of the theSVG catalogue.

    Args:
        config: Active run configuration.

    Returns:
        The final tallies of the run.

    Raises:
        LogoDownloadError: If the repository cannot be fetched or read.
    """
    commit = resolve_commit(config.ref)
    LOGGER.info("Repository %s pinned at %s", REPO_ID, commit)

    tarball = download_tarball(commit, config.tarball_dir)
    index = index_archive(tarball)
    LOGGER.info("Indexed %d file(s) from the tarball", len(index))

    counters = RunCounters()
    with tarfile.open(tarball, mode="r:gz") as archive:
        catalog = load_catalog(archive, index)
        candidates = build_candidates(catalog, config)
        LOGGER.info(
            "Catalogue holds %d brand(s); %d candidate(s) after filtering, seed=%d",
            len(catalog),
            len(candidates),
            config.seed,
        )

        if config.dry_run:
            for entry in candidates[: config.limit or len(candidates)][:20]:
                LOGGER.info("  %s (%s) -> %s", entry.slug, entry.variant, entry.source_path)
            LOGGER.info("Dry run: nothing downloaded.")
            return counters

        _prepare_output_dirs(config)
        with (
            config.csv_path.open("w", encoding="utf-8", newline="") as manifest_handle,
            config.attribution_path.open("w", encoding="utf-8", newline="") as credit_handle,
        ):
            manifest = csv.DictWriter(manifest_handle, fieldnames=list(CSV_FIELDNAMES))
            manifest.writeheader()
            credits = csv.DictWriter(credit_handle, fieldnames=list(ATTRIBUTION_FIELDNAMES))
            credits.writeheader()

            for accepted in _iter_accepted(archive, index, candidates, config, counters):
                manifest.writerow(
                    _csv_row(accepted.entry, accepted.render, accepted.svg_sha256)
                )
                credits.writerows(_attribution_rows(accepted.entry, commit))

    _write_report(config, commit, counters, len(candidates))
    return counters


def parse_args(argv: Sequence[str] | None = None) -> DownloadConfig:
    """Parse command line arguments into a configuration object.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download a unique, deduplicated set of brand logos from the "
            "glincker/thesvg repository as SVG plus transparent PNG."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Destination directory for the SVGs, PNGs and manifest.",
    )
    parser.add_argument(
        "--ref",
        default="main",
        help="Repository ref (branch, tag or commit SHA) to download.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of unique logos to keep. Use 0 for every candidate.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="Edge length in pixels of the square PNG canvas.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=4,
        help="Transparent padding in pixels kept on each side of the canvas.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for the deterministic catalogue shuffle.",
    )
    parser.add_argument(
        "--category",
        dest="categories",
        action="append",
        default=[],
        metavar="NAME",
        help="Keep only brands in this category; repeatable.",
    )
    parser.add_argument(
        "--collection",
        dest="collections",
        action="append",
        default=[],
        metavar="NAME",
        help="Keep only brands in this catalogue collection; repeatable.",
    )
    parser.add_argument(
        "--license",
        dest="licenses",
        action="append",
        default=[],
        metavar="SPDX",
        help=(
            "Keep only brands under this licence; repeatable. "
            f"--permissive is shorthand for {' '.join(PERMISSIVE_LICENSES)}."
        ),
    )
    parser.add_argument(
        "--permissive",
        action="store_true",
        help=(
            "Keep only openly-licensed brands, i.e. "
            f"--license {' --license '.join(PERMISSIVE_LICENSES)}."
        ),
    )
    parser.add_argument(
        "--max-aspect",
        type=float,
        default=6.0,
        help="Reject artwork whose long/short edge ratio exceeds this value.",
    )
    parser.add_argument(
        "--max-luminance",
        type=float,
        default=0.97,
        help="Reject artwork this bright (0-1); near-white marks vanish on paper.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for the repository tarball (default: <output-dir>/.cache).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render logos even when their output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the candidate set, then exit without writing files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging, including a reason for every rejected logo.",
    )

    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.size <= 0:
        parser.error("--size must be positive.")
    if args.margin < 0 or 2 * args.margin >= args.size:
        parser.error("--margin must be non-negative and smaller than half of --size.")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be zero or positive.")

    licenses = tuple(args.licenses)
    if args.permissive:
        known = {item.casefold() for item in licenses}
        licenses += tuple(
            item for item in PERMISSIVE_LICENSES if item.casefold() not in known
        )

    return DownloadConfig(
        output_dir=args.output_dir,
        ref=args.ref,
        limit=args.limit or None,
        size=args.size,
        margin=args.margin,
        seed=args.seed,
        categories=tuple(args.categories),
        collections=tuple(args.collections),
        licenses=licenses,
        max_aspect=args.max_aspect,
        max_luminance=args.max_luminance,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the downloader from the command line.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` on a handled failure, ``130`` on interrupt.
    """
    config = parse_args(argv)
    try:
        counters = download_thesvg_logos(config)
    except LogoDownloadError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; re-run the same command to resume.")
        return 130

    if config.dry_run:
        return 0

    LOGGER.info(
        "Done: %d unique logo(s) -> %s (%d written, %d reused)",
        counters.accepted,
        config.output_dir,
        counters.written,
        counters.reused,
    )
    LOGGER.info(
        "Skipped: %d duplicate svg, %d duplicate render, %d blank, %d wide, "
        "%d too light, %d render error, %d missing",
        counters.skipped_duplicate_svg,
        counters.skipped_duplicate_render,
        counters.skipped_blank,
        counters.skipped_aspect,
        counters.skipped_luminance,
        counters.skipped_render_error,
        counters.skipped_missing,
    )
    if config.limit is not None and counters.accepted < config.limit:
        LOGGER.warning(
            "Only %d of the requested %d logo(s) passed the filters.",
            counters.accepted,
            config.limit,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
