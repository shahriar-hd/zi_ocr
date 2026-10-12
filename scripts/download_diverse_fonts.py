"""Download the open-licence Persian font pool and record every licence.

Why this script exists
----------------------

The administrative-letter dataset rendered with whatever was in
``~/.local/share/fonts``, which on this machine is mostly the legacy *B-series*
(``BNazanin``, ``BMitra``, ``BTitr`` ...) and ``IRANSansX``. Those are
**proprietary**. They were fine for a private experiment and they are not fine
for a dataset published on the Hub: nobody can legally redistribute the images,
and nobody downstream can reproduce them. ``.claude/dataset.md`` is explicit -
*do not use proprietary fonts* - so the diverse dataset renders from a pool
built entirely from SIL Open Font License 1.1 families, fetched here.

The pool is also much wider than the old one on purpose. Font identity is one
of the strongest confounds in OCR: a model trained on four sans families learns
those four families' letterforms, not Persian. The pool below spans six style
classes, and three families are held out of training entirely so
``test_unseen_fonts`` can measure exactly that failure.

What it guarantees
------------------

1. **Licence provenance.** Every family carries its licence text to disk and a
   row in ``fonts.csv`` naming the licence, the source URL and the attribution.
   A family whose licence file cannot be fetched is *rejected*, not assumed.
2. **Glyph coverage.** Every file is opened with ``fontTools`` and checked
   against the full Persian alphabet, Persian digits and Persian punctuation.
   A font missing even ``گ`` is rejected, because Chromium would silently
   substitute a fallback face and the label would stop describing the pixels.
   This is the single check that stops the most expensive class of silent
   corruption in the whole pipeline.
3. **Idempotence.** A family already on disk and already passing is skipped, so
   re-running costs one HTTP HEAD per family rather than the whole download.

Usage::

    python scripts/download_diverse_fonts.py                 # full pool
    python scripts/download_diverse_fonts.py --only amiri vazirmatn
    python scripts/download_diverse_fonts.py --verify-only   # re-check on disk
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LOGGER: Final[logging.Logger] = logging.getLogger("download_diverse_fonts")

#: Where the pool lands. Kept inside the dataset work tree rather than in
#: ``~/.local/share/fonts`` so the proprietary host fonts and the
#: redistributable pool can never be confused for one another.
DEFAULT_TARGET: Final[str] = "data/diverse/fonts"

#: Columns of ``fonts.csv``. ``style_class`` and ``holdout_eligible`` are what
#: the layout stage samples on; ``licence`` and ``attribution`` are what the
#: dataset card is built from.
FONTS_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "font_id",
    "family",
    "file",
    "style_class",
    "weight",
    "italic",
    "licence",
    "licence_file",
    "source",
    "source_url",
    "attribution",
    "glyphs",
    "covers_persian",
    "holdout_eligible",
)

#: GitHub raw host for the Google Fonts repository. Every directory under
#: ``ofl/`` is, by that repository's own contract, SIL OFL 1.1 and ships an
#: ``OFL.txt`` - which is why this is the preferred source: the licence comes
#: from the same commit as the binary.
_GF_API: Final[str] = "https://api.github.com/repos/google/fonts/contents/ofl"
_GF_RAW: Final[str] = "https://raw.githubusercontent.com/google/fonts/main/ofl"

#: Characters a font must draw for Persian text to be renderable at all.
#:
#: Split by purpose so a rejection message says *what* is missing. The four
#: Persian-only letters (``پ چ ژ گ``) are listed first because they are the
#: usual casualty: an Arabic-only font covers everything else and silently
#: falls back on exactly these, producing a page that looks almost right.
_REQUIRED_PERSIAN_LETTERS: Final[str] = (
    "\u067e\u0686\u0698\u06af"  # pe che zhe gaf - Persian-only
    "\u0627\u0628\u062a\u062b\u062c\u062d\u062e\u062f\u0630\u0631"
    "\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063a\u0641"
    "\u0642\u06a9\u0644\u0645\u0646\u0648\u0647\u06cc"
    "\u0622\u0623\u0624\u0626\u0621\u0629"  # alef-maddah, hamza carriers
)

#: Persian digits. A family that lacks these cannot render a date, an invoice
#: or a page number, which rules out most of the ``table_form`` category.
_REQUIRED_PERSIAN_DIGITS: Final[str] = (
    "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
)

#: Persian punctuation and the Latin digits, which appear in the ``latin`` and
#: ``mixed`` digit-script samples.
_REQUIRED_PUNCTUATION: Final[str] = "\u060c\u061b\u061f\u00ab\u00bb\u002e\u003a0123456789"

#: Style classes the layout stage samples across. A category draws a class
#: first and a family second, so adding ten more sans families does not make
#: every poetry page sans.
STYLE_CLASSES: Final[tuple[str, ...]] = (
    "sans",
    "naskh",
    "serif",
    "nastaliq",
    "display",
    "mono",
)


@dataclass(frozen=True, slots=True)
class FontSource:
    """One font family and where to get it from.

    Attributes:
        family: Human family name as it will appear in CSS and in ``fonts.csv``.
        kind: ``google_fonts`` (directory under ``ofl/``), ``github_release``
            (a zip asset from the latest release) or ``github_zip`` (a zip at a
            fixed URL).
        locator: Directory name, ``owner/repo``, or full URL, per ``kind``.
        style_class: One of :data:`STYLE_CLASSES`.
        licence: SPDX-ish identifier recorded in ``fonts.csv``.
        attribution: Human-readable credit for the dataset card.
        holdout_eligible: Whether this family may be chosen for the
            ``test_unseen_fonts`` split. The widest, most typical families are
            marked ``False``: holding out the one font that half the corpus
            would otherwise use makes the hard test measure font *frequency*
            rather than font *novelty*.
        globs: Filename patterns to keep from an archive. Static instances are
            preferred over variable fonts where both ship, because Chromium
            renders a named static weight more predictably than a ``wght``
            axis default.
    """

    family: str
    kind: str
    locator: str
    style_class: str
    licence: str
    attribution: str
    holdout_eligible: bool = True
    globs: tuple[str, ...] = ("*.ttf", "*.otf")


#: The pool.
#:
#: Thirty-odd families, every one SIL OFL 1.1. The mix is deliberate:
#:
#: * **sans** dominates real Persian screen and office text, so it has the most
#:   families - but not the most *weight* in the layout sampler.
#: * **naskh** and **serif** cover books, religious and scholarly printing.
#: * **nastaliq** is the hard case: Gulzar and Noto Nastaliq Urdu produce deeply
#:   overlapping, steeply sloped lines that no sans-trained model reads. Two
#:   families only, because they are also the slowest to shape and the easiest
#:   to render illegibly small.
#: * **display** covers banners, posters and shop signs, where Persian type is
#:   at its most extreme.
#:
#: ``rastikerdar``'s families are the de-facto standard for Persian web text and
#: are not in Google Fonts, so they come from their own releases.
FONT_SOURCES: Final[tuple[FontSource, ...]] = (
    # ---- sans: Persian web and UI standards -------------------------------
    FontSource(
        "Vazirmatn",
        "google_fonts",
        "vazirmatn",
        "sans",
        "OFL-1.1",
        "Vazirmatn by Saber Rastikerdar",
        holdout_eligible=False,
    ),
    FontSource(
        "Sahel",
        "github_release",
        "rastikerdar/sahel-font",
        "sans",
        "OFL-1.1",
        "Sahel by Saber Rastikerdar",
    ),
    FontSource(
        "Shabnam",
        "github_release",
        "rastikerdar/shabnam-font",
        "sans",
        "OFL-1.1",
        "Shabnam by Saber Rastikerdar",
    ),
    FontSource(
        "Samim",
        "github_release",
        "rastikerdar/samim-font",
        "sans",
        "OFL-1.1",
        "Samim by Saber Rastikerdar",
    ),
    FontSource(
        "Gandom",
        "github_release",
        "rastikerdar/gandom-font",
        "sans",
        "OFL-1.1",
        "Gandom by Saber Rastikerdar",
    ),
    FontSource(
        "Tanha",
        "github_release",
        "rastikerdar/tanha-font",
        "sans",
        "OFL-1.1",
        "Tanha by Saber Rastikerdar",
    ),
    FontSource(
        "Estedad",
        "github_release",
        "aminabedi68/Estedad",
        "sans",
        "OFL-1.1",
        "Estedad by Amin Abedi",
    ),
    FontSource(
        "Noto Sans Arabic",
        "google_fonts",
        "notosansarabic",
        "sans",
        "OFL-1.1",
        "Noto Sans Arabic by Google",
        holdout_eligible=False,
    ),
    FontSource(
        "IBM Plex Sans Arabic",
        "google_fonts",
        "ibmplexsansarabic",
        "sans",
        "OFL-1.1",
        "IBM Plex Sans Arabic by IBM",
    ),
    FontSource(
        "Cairo", "google_fonts", "cairo", "sans", "OFL-1.1", "Cairo by Mohamed Gaber"
    ),
    FontSource(
        "Almarai", "google_fonts", "almarai", "sans", "OFL-1.1", "Almarai by Boutros"
    ),
    FontSource("Mada", "google_fonts", "mada", "sans", "OFL-1.1", "Mada by Khaled Hosny"),
    FontSource(
        "Baloo Bhaijaan 2",
        "google_fonts",
        "baloobhaijaan2",
        "sans",
        "OFL-1.1",
        "Baloo Bhaijaan 2 by Ek Type",
    ),
    # ---- naskh: books, scholarly and religious printing -------------------
    FontSource(
        "Noto Naskh Arabic",
        "google_fonts",
        "notonaskharabic",
        "naskh",
        "OFL-1.1",
        "Noto Naskh Arabic by Google",
        holdout_eligible=False,
    ),
    FontSource(
        "Scheherazade New",
        "google_fonts",
        "scheherazadenew",
        "naskh",
        "OFL-1.1",
        "Scheherazade New by SIL International",
    ),
    FontSource(
        "Lateef",
        "google_fonts",
        "lateef",
        "naskh",
        "OFL-1.1",
        "Lateef by SIL International",
    ),
    FontSource(
        "Harmattan",
        "google_fonts",
        "harmattan",
        "naskh",
        "OFL-1.1",
        "Harmattan by SIL International",
    ),
    # ---- serif / literary -------------------------------------------------
    FontSource(
        "Amiri", "google_fonts", "amiri", "serif", "OFL-1.1", "Amiri by Khaled Hosny"
    ),
    FontSource(
        "Markazi Text",
        "google_fonts",
        "markazitext",
        "serif",
        "OFL-1.1",
        "Markazi Text by Borna Izadpanah",
    ),
    FontSource(
        "El Messiri",
        "google_fonts",
        "elmessiri",
        "serif",
        "OFL-1.1",
        "El Messiri by Mohamed Gaber",
    ),
    FontSource(
        "Parastoo",
        "github_release",
        "rastikerdar/parastoo-font",
        "serif",
        "OFL-1.1",
        "Parastoo by Saber Rastikerdar",
    ),
    # ---- nastaliq: the hard case ------------------------------------------
    FontSource(
        "Gulzar",
        "google_fonts",
        "gulzar",
        "nastaliq",
        "OFL-1.1",
        "Gulzar by Borna Izadpanah, Fiona Ross and Alice Savoie",
    ),
    FontSource(
        "Noto Nastaliq Urdu",
        "google_fonts",
        "notonastaliqurdu",
        "nastaliq",
        "OFL-1.1",
        "Noto Nastaliq Urdu by Google",
    ),
    # ---- display: banners, posters, signage -------------------------------
    FontSource(
        "Noto Kufi Arabic",
        "google_fonts",
        "notokufiarabic",
        "display",
        "OFL-1.1",
        "Noto Kufi Arabic by Google",
    ),
    FontSource(
        "Reem Kufi",
        "google_fonts",
        "reemkufi",
        "display",
        "OFL-1.1",
        "Reem Kufi by Khaled Hosny",
    ),
    FontSource(
        "Lalezar",
        "google_fonts",
        "lalezar",
        "display",
        "OFL-1.1",
        "Lalezar by Borna Izadpanah",
    ),
    FontSource(
        "Rakkas", "google_fonts", "rakkas", "display", "OFL-1.1", "Rakkas by Zeynep Akay"
    ),
    FontSource(
        "Jomhuria",
        "google_fonts",
        "jomhuria",
        "display",
        "OFL-1.1",
        "Jomhuria by Kourosh Beigpour",
    ),
    FontSource(
        "Katibeh",
        "google_fonts",
        "katibeh",
        "display",
        "OFL-1.1",
        "Katibeh by Kourosh Beigpour",
    ),
    FontSource(
        "Mirza", "google_fonts", "mirza", "display", "OFL-1.1", "Mirza by Tipogral"
    ),
    FontSource(
        "Aref Ruqaa",
        "google_fonts",
        "arefruqaa",
        "display",
        "OFL-1.1",
        "Aref Ruqaa by Abdullah Aref",
    ),
    FontSource(
        "Changa",
        "google_fonts",
        "changa",
        "display",
        "OFL-1.1",
        "Changa by Eduardo Tunni",
    ),
    FontSource(
        "Vibes",
        "google_fonts",
        "vibes",
        "display",
        "OFL-1.1",
        "Vibes by Kimberly Geswein",
    ),
    FontSource(
        "Kufam", "google_fonts", "kufam", "display", "OFL-1.1", "Kufam by Eduardo Tunni"
    ),
    FontSource(
        "Qahiri", "google_fonts", "qahiri", "display", "OFL-1.1", "Qahiri by Khaled Hosny"
    ),
)


class FontDownloadError(RuntimeError):
    """Raised when a family cannot be fetched or fails its licence check."""


def _http_get(url: str, *, timeout: float = 120.0) -> bytes:
    """Fetch a URL, following redirects, with a non-default user agent.

    GitHub's raw host rejects the stdlib default user agent on some paths, and
    a 403 that looks like a missing file is a confusing way to learn that.

    Args:
        url: Absolute URL.
        timeout: Socket timeout in seconds.

    Returns:
        The response body.

    Raises:
        FontDownloadError: On any HTTP or network failure.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": "fa-diverse-ocr/1.0 (+dataset font pool)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        raise FontDownloadError(f"{url}: {error}") from error


def _github_listing(url: str) -> list[dict[str, object]]:
    """List a GitHub contents API directory.

    Args:
        url: A ``/contents/`` API URL.

    Returns:
        The decoded JSON array.

    Raises:
        FontDownloadError: If the response is not a JSON array.
    """
    payload = json.loads(_http_get(url).decode("utf-8"))
    if not isinstance(payload, list):
        raise FontDownloadError(f"{url}: expected a directory listing")
    return payload


def _fetch_google_fonts(source: FontSource, destination: Path) -> list[Path]:
    """Download one family from the Google Fonts ``ofl`` tree.

    Static instances are preferred over the variable font when both are
    present: a static ``Amiri-Bold.ttf`` renders at a known weight, whereas a
    variable font's default instance depends on Chromium's axis handling and
    has produced surprise weight shifts between Chromium versions.

    Args:
        source: The family to fetch.
        destination: Directory to write into; created if absent.

    Returns:
        The font files written.

    Raises:
        FontDownloadError: If the directory holds no font or no ``OFL.txt``.
    """
    entries = _github_listing(f"{_GF_API}/{source.locator}")
    names = [str(entry["name"]) for entry in entries if entry.get("type") == "file"]

    licence_names = [name for name in names if name.upper().startswith("OFL")]
    if not licence_names:
        raise FontDownloadError(
            f"{source.family}: no OFL.txt in google/fonts/ofl/{source.locator}; "
            "refusing to ship a font whose licence text is not alongside it"
        )

    statics = [
        name for name in names if name.endswith((".ttf", ".otf")) and "[" not in name
    ]
    variables = [
        name for name in names if name.endswith((".ttf", ".otf")) and "[" in name
    ]
    wanted = statics or variables
    if not wanted:
        raise FontDownloadError(f"{source.family}: no .ttf/.otf in ofl/{source.locator}")

    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in sorted(wanted):
        quoted = urllib.parse.quote(name)
        target = destination / name
        target.write_bytes(_http_get(f"{_GF_RAW}/{source.locator}/{quoted}"))
        written.append(target)
    licence_name = licence_names[0]
    (destination / licence_name).write_bytes(
        _http_get(f"{_GF_RAW}/{source.locator}/{urllib.parse.quote(licence_name)}")
    )
    return written


def _fetch_github_release(source: FontSource, destination: Path) -> list[Path]:
    """Download one family from the latest GitHub release zip.

    Args:
        source: The family to fetch; ``locator`` is ``owner/repo``.
        destination: Directory to write into; created if absent.

    Returns:
        The font files written.

    Raises:
        FontDownloadError: If the release carries no zip asset, or the zip
            holds no font file.
    """
    meta = json.loads(
        _http_get(
            f"https://api.github.com/repos/{source.locator}/releases/latest"
        ).decode("utf-8")
    )
    assets = [
        str(asset["browser_download_url"])
        for asset in meta.get("assets", [])
        if str(asset.get("name", "")).endswith(".zip")
    ]
    if not assets:
        raise FontDownloadError(
            f"{source.family}: latest release of {source.locator} has no zip"
        )
    archive = zipfile.ZipFile(io.BytesIO(_http_get(assets[0])))
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    licence_written = False
    for member in archive.namelist():
        name = Path(member).name
        if not name or member.endswith("/"):
            continue
        lowered = name.lower()
        # Prefer the ``ttf`` subtree when a release ships both ttf and woff2;
        # Chromium reads both, but a ttf is what fontTools can audit.
        if lowered.endswith((".ttf", ".otf")) and "webfonts" not in member.lower():
            target = destination / name
            target.write_bytes(archive.read(member))
            written.append(target)
        elif not licence_written and (
            "ofl" in lowered or "license" in lowered or "licence" in lowered
        ):
            (destination / "OFL.txt").write_bytes(archive.read(member))
            licence_written = True

    if not written:
        raise FontDownloadError(f"{source.family}: zip held no .ttf/.otf")
    if not licence_written:
        # The rastikerdar releases are OFL but some zips omit the text. Fetch
        # it from the repository root rather than asserting the licence.
        for candidate in ("LICENSE", "OFL.txt", "LICENSE.txt", "OFL-FAQ.txt"):
            try:
                body = _http_get(
                    f"https://raw.githubusercontent.com/{source.locator}/master/{candidate}"
                )
            except FontDownloadError:
                continue
            (destination / "OFL.txt").write_bytes(body)
            licence_written = True
            break
    if not licence_written:
        raise FontDownloadError(
            f"{source.family}: could not obtain licence text from {source.locator}; "
            "refusing to add a font whose licence cannot be shipped"
        )
    return written


def _missing_glyphs(path: Path) -> str:
    """Return the required characters a font file cannot draw.

    Opens the font's character map directly. This is the check that prevents
    the dataset's worst silent failure: Chromium substituting a fallback face
    for a handful of characters, so the image shows one letterform while the
    label claims another.

    Args:
        path: A ``.ttf`` or ``.otf`` file.

    Returns:
        The missing characters, in required order; empty when coverage is
        complete.

    Raises:
        FontDownloadError: If the file cannot be parsed as a font.
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise FontDownloadError("fontTools is required: pip install fonttools") from error

    try:
        font = TTFont(str(path), fontNumber=0, lazy=True)
        covered: set[int] = set()
        for table in font["cmap"].tables:
            covered.update(table.cmap.keys())
        font.close()
    except Exception as error:  # noqa: BLE001 - any parse failure is a rejection
        raise FontDownloadError(f"{path.name}: cannot read cmap ({error})") from error

    required = (
        _REQUIRED_PERSIAN_LETTERS + _REQUIRED_PERSIAN_DIGITS + _REQUIRED_PUNCTUATION
    )
    return "".join(character for character in required if ord(character) not in covered)


def _face_metadata(path: Path) -> tuple[str, int, bool, int]:
    """Read family name, weight class, italic flag and glyph count.

    Args:
        path: A font file.

    Returns:
        ``(family, weight, italic, glyph_count)``. ``family`` falls back to the
        filename stem when the name table is unreadable.

    Raises:
        FontDownloadError: If the file cannot be opened.
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError as error:  # pragma: no cover
        raise FontDownloadError("fontTools is required: pip install fonttools") from error
    try:
        font = TTFont(str(path), fontNumber=0, lazy=True)
        family = path.stem
        for record in font["name"].names:
            if record.nameID in (16, 1):
                try:
                    family = str(record.toUnicode()).strip() or family
                except UnicodeDecodeError:
                    continue
                if record.nameID == 16:
                    break
        os2 = font.get("OS/2")
        weight = int(getattr(os2, "usWeightClass", 400) or 400)
        italic = bool(int(getattr(os2, "fsSelection", 0) or 0) & 0x01)
        glyphs = int(font["maxp"].numGlyphs)
        font.close()
    except FontDownloadError:
        raise
    except Exception as error:  # noqa: BLE001
        raise FontDownloadError(f"{path.name}: cannot read metadata ({error})") from error
    return family, weight, italic, glyphs


def _licence_file(directory: Path) -> str:
    """Return the name of the licence file in a family directory.

    Args:
        directory: A family directory.

    Returns:
        The filename, or ``""`` when none is present.
    """
    for candidate in sorted(directory.iterdir()):
        if candidate.is_file() and candidate.name.upper().startswith(("OFL", "LICEN")):
            return candidate.name
    return ""


def fetch_family(source: FontSource, root: Path, *, force: bool = False) -> list[Path]:
    """Download one family unless it is already present.

    Args:
        source: The family to fetch.
        root: Pool root; the family lands in ``root/<slug>``.
        force: Re-download even when the directory already holds fonts.

    Returns:
        Font files for this family, whether freshly downloaded or already
        present.

    Raises:
        FontDownloadError: On a fetch failure or an unknown ``kind``.
    """
    slug = source.family.lower().replace(" ", "-")
    destination = root / slug
    existing = sorted(
        path for path in destination.glob("*") if path.suffix.lower() in {".ttf", ".otf"}
    )
    if existing and not force:
        LOGGER.info("%-22s already present (%d file(s))", source.family, len(existing))
        return existing

    if destination.exists() and force:
        shutil.rmtree(destination)

    LOGGER.info("%-22s fetching from %s", source.family, source.kind)
    if source.kind == "google_fonts":
        return _fetch_google_fonts(source, destination)
    if source.kind == "github_release":
        return _fetch_github_release(source, destination)
    raise FontDownloadError(f"{source.family}: unknown source kind {source.kind!r}")


def build_catalogue(
    sources: Sequence[FontSource],
    root: Path,
    *,
    force: bool = False,
    verify_only: bool = False,
) -> tuple[list[dict[str, object]], list[str]]:
    """Fetch every family and audit the result.

    Args:
        sources: Families to process.
        root: Pool root.
        force: Re-download families already on disk.
        verify_only: Skip downloading; audit what is already there.

    Returns:
        ``(rows, rejections)`` where ``rows`` are ``fonts.csv`` records for
        every file that passed and ``rejections`` are human-readable reasons.
    """
    rows: list[dict[str, object]] = []
    rejections: list[str] = []

    for source in sources:
        slug = source.family.lower().replace(" ", "-")
        destination = root / slug
        try:
            if verify_only:
                files = sorted(
                    path
                    for path in destination.glob("*")
                    if path.suffix.lower() in {".ttf", ".otf"}
                )
                if not files:
                    rejections.append(f"{source.family}: nothing on disk")
                    continue
            else:
                files = fetch_family(source, root, force=force)
        except FontDownloadError as error:
            rejections.append(f"{source.family}: {error}")
            continue

        licence_name = _licence_file(destination)
        if not licence_name:
            rejections.append(f"{source.family}: no licence file on disk")
            continue

        kept = 0
        for path in files:
            try:
                missing = _missing_glyphs(path)
                family, weight, italic, glyphs = _face_metadata(path)
            except FontDownloadError as error:
                rejections.append(str(error))
                continue
            if missing:
                rejections.append(
                    f"{path.name}: missing {len(missing)} required glyph(s): {missing!r}"
                )
                continue
            rows.append(
                {
                    "font_id": f"{slug}/{path.name}",
                    "family": source.family,
                    "file": path.relative_to(root).as_posix(),
                    "style_class": source.style_class,
                    "weight": weight,
                    "italic": "true" if italic else "false",
                    "licence": source.licence,
                    "licence_file": f"{slug}/{licence_name}",
                    "source": source.kind,
                    "source_url": source.locator,
                    "attribution": source.attribution,
                    "glyphs": glyphs,
                    "covers_persian": "true",
                    "holdout_eligible": "true" if source.holdout_eligible else "false",
                }
            )
            kept += 1
        if kept == 0:
            rejections.append(f"{source.family}: every face failed glyph coverage")
        else:
            LOGGER.info(
                "%-22s %d face(s) accepted (%s)", source.family, kept, source.style_class
            )

    return rows, rejections


def write_catalogue(rows: Sequence[dict[str, object]], root: Path) -> Path:
    """Write ``fonts.csv``.

    Args:
        rows: Accepted records from :func:`build_catalogue`.
        root: Pool root.

    Returns:
        The path written.
    """
    target = root / "fonts.csv"
    root.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FONTS_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["font_id"])))
    return target


def _summarise(rows: Sequence[dict[str, object]]) -> str:
    """Build the per-style-class tally printed at the end of a run.

    Args:
        rows: Accepted records.

    Returns:
        A one-line summary.
    """
    per_class: dict[str, set[str]] = {}
    for row in rows:
        per_class.setdefault(str(row["style_class"]), set()).add(str(row["family"]))
    parts = [
        f"{name} {len(per_class.get(name, ()))}"
        for name in STYLE_CLASSES
        if per_class.get(name)
    ]
    return ", ".join(parts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET, help="pool root (default: %(default)s)"
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="fetch only these families (case-insensitive)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download families already on disk"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="audit what is on disk, download nothing",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def _select(only: Iterable[str]) -> tuple[FontSource, ...]:
    """Filter :data:`FONT_SOURCES` by family name or locator.

    Args:
        only: Names to keep; empty keeps everything.

    Returns:
        The selected sources.

    Raises:
        SystemExit: If a requested name matches nothing, because silently
            fetching the whole pool when the user asked for one family wastes
            a few hundred megabytes.
    """
    wanted = {name.lower() for name in only}
    if not wanted:
        return FONT_SOURCES
    selected = tuple(
        source
        for source in FONT_SOURCES
        if source.family.lower() in wanted or source.locator.lower() in wanted
    )
    matched = {source.family.lower() for source in selected} | {
        source.locator.lower() for source in selected
    }
    unknown = sorted(wanted - matched)
    if unknown:
        raise SystemExit(f"unknown family name(s): {unknown}")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    """Download the pool, audit it, and write ``fonts.csv``.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; ``1`` when the pool is too small or too narrow to
        render the corpus.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    from fa_ocr_lora.dataset.common.io_utils import resolve_repo_path

    root = resolve_repo_path(args.target)
    sources = _select(args.only)
    LOGGER.info("Pool root: %s", root)
    LOGGER.info("%d family/families selected", len(sources))

    rows, rejections = build_catalogue(
        sources, root, force=args.force, verify_only=args.verify_only
    )
    target = write_catalogue(rows, root)

    families = {str(row["family"]) for row in rows}
    LOGGER.info("")
    LOGGER.info("Accepted %d face(s) across %d family/families", len(rows), len(families))
    LOGGER.info("By style class: %s", _summarise(rows))
    LOGGER.info("Catalogue: %s", target)

    if rejections:
        LOGGER.warning("")
        LOGGER.warning("%d rejection(s):", len(rejections))
        for reason in rejections:
            LOGGER.warning("  - %s", reason)

    # A pool with one nastaliq family or three families total cannot support the
    # holdout splits, and discovering that after rendering 10k pages is worse
    # than failing here.
    per_class = {
        name: {str(row["family"]) for row in rows if row["style_class"] == name}
        for name in STYLE_CLASSES
    }
    if len(families) < 12:
        LOGGER.error("Pool too small: %d families, need at least 12", len(families))
        return 1
    thin = [name for name in ("sans", "naskh", "display") if len(per_class[name]) < 2]
    if thin:
        LOGGER.error("Style class(es) with fewer than 2 families: %s", thin)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
