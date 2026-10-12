"""Parsers for each lexicon source, and the era/domain tagging rules.

Three sources, each contributing something the others cannot:

``lilak``
    A Persian spell-checking dictionary (Apache-2.0) with ~94k headwords and
    part-of-speech tags. It is the only source that is *confirmed Persian* -
    it exists to spell-check Persian prose - so membership in Lilak is used as
    a positive signal that protects a Wiktionary word from the foreign-language
    filter below.

``wiktionary``
    The only source that **labels era and domain**. The category statistics of
    the live dump drive every rule in this module; nothing here was guessed.
    Three findings shaped the design:

    * ``واژه‌های مصوب فرهنگستان زبان و ادب فارسی`` has ~17,800 members. These
      are the Academy of Persian Language and Literature's approved coinages -
      the ``neologism`` class, and by far the largest single labelled group.
      An LLM writing unprompted Persian almost never reaches for them.
    * ``کهن واژه``, ``زبان معیار باستان``, ``اسم‌های پهلوی`` and
      ``اصطلاحات پیشا هند و ایرانی`` give the ``archaic`` class, and
      ``اصطلاحات شاهنامه`` / ``فرهنگ شمس`` the ``literary`` one.
    * Persian Wiktionary documents *other languages* in Persian. ``اسم‌های
      انگلیسی`` alone has ~2,400 members and ``فعل‌های انگلیسی`` ~2,200. Left
      in, roughly ten thousand English, French and Turkish words would enter a
      Persian lexicon and be injected into Persian prompts. :data:`_FOREIGN`
      removes them.

``wikipedia_titles``
    Single-token article titles, which supply scientific and technical
    terminology that neither of the other two carries.

Arabic-derived vocabulary is deliberately **not** filtered. Something like a
third of formal Persian is Arabic in origin and those words are ordinary
Persian; excluding them would strip the bureaucratic and legal registers the
corpus needs.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..text_policy import ZWNJ, normalise_for_eval, persian_letter_ratio

__all__ = [
    "Headword",
    "parse_lilak",
    "parse_wikipedia_titles",
    "parse_wiktionary",
    "tag_categories",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: Part-of-speech labels this pipeline records. Deliberately coarse: the plan
#: stage only needs to know whether a word can head a noun phrase or must be
#: conjugated, and a finer scheme would be unreliable across three sources
#: that disagree about everything else.
POS_LABELS: Final[tuple[str, ...]] = (
    "noun",
    "verb",
    "adjective",
    "adverb",
    "proper_noun",
    "other",
)

#: Lilak's own POS column mapped onto :data:`POS_LABELS`.
_LILAK_POS: Final[Mapping[str, str]] = {
    "noun_singular": "noun",
    "noun_plural": "noun",
    "noun": "noun",
    "verb": "verb",
    "adjective": "adjective",
    "adverb": "adverb",
    "pronoun": "other",
    "preposition": "other",
    "conjunction": "other",
    "interjection": "other",
    "number": "other",
    "abbreviation": "other",
    "prefix": "other",
    "suffix": "other",
}

#: Wiktionary category fragments that identify a part of speech. Matched as
#: substrings, longest first, because the live categories are inflected
#: (``اسم‌های فارسی``, ``اسم‌های مرکب فارسی``, ``اسم مصدر``).
_POS_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("اسم خاص", "proper_noun"),
    ("اسم مصدر", "noun"),
    ("اسم‌های مصدر", "noun"),
    ("مصدر", "verb"),
    ("فعل", "verb"),
    ("صفت", "adjective"),
    ("قید", "adverb"),
    ("اسم", "noun"),
    ("حرف", "other"),
    ("ضمیر", "other"),
)

#: Category fragments that identify an era/register class. Order matters:
#: a word in both ``واژه‌های مصوب فرهنگستان`` and some ``اصطلاحات`` category is a
#: neologism first, because that is the rarer and more informative label.
_ERA_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    # --- neologism: the Academy's approved modern coinages -----------------
    ("مصوب فرهنگستان", "neologism"),
    ("فرهنگستان", "neologism"),
    ("واژه‌های نو", "neologism"),
    # --- archaic: classical, obsolete and pre-modern strata ----------------
    ("کهن واژه", "archaic"),
    ("کهن", "archaic"),
    ("باستان", "archaic"),
    ("پهلوی", "archaic"),
    ("اوستایی", "archaic"),
    ("پیشا هند و ایرانی", "archaic"),
    ("هند و ایرانی", "archaic"),
    ("فارسی میانه", "archaic"),
    ("منسوخ", "archaic"),
    ("متروک", "archaic"),
    # --- literary: poetic and classical-text register ----------------------
    ("شاهنامه", "literary"),
    ("فرهنگ شمس", "literary"),
    ("ادبی", "literary"),
    ("استعاره", "literary"),
    ("شعر", "literary"),
    ("حافظ", "literary"),
    ("مولوی", "literary"),
    ("سعدی", "literary"),
    ("نظامی گنجوی", "literary"),
    ("عرفان", "literary"),
    # --- dialectal: regional forms -----------------------------------------
    ("گویش", "dialectal"),
    ("لهجه", "dialectal"),
    ("آذرگشسپ", "dialectal"),
    ("محلی", "dialectal"),
)

#: Domain labels, keyed by the category fragment that implies them. These feed
#: the ``domain`` column and the plan sampler's genre matching, so a medical
#: term can be steered into a medical leaflet - and, 15% of the time,
#: deliberately into a sports column instead.
_DOMAIN_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("پزشکی", "medicine"),
    ("داروی", "medicine"),
    ("کالبدشناسی", "medicine"),
    ("گیاه‌شناسی", "botany"),
    ("گیاه", "botany"),
    ("جانورشناسی", "zoology"),
    ("جانور", "zoology"),
    ("پرنده", "zoology"),
    ("زیست‌شناسی", "biology"),
    ("ریاضیات", "mathematics"),
    ("ریاضی", "mathematics"),
    ("آمار", "mathematics"),
    ("فیزیک", "physics"),
    ("شیمی", "chemistry"),
    ("حقوق", "law"),
    ("فقه", "religion"),
    ("دین", "religion"),
    ("اسلام", "religion"),
    ("قرآن", "religion"),
    ("نظامی", "military"),
    ("ارتش", "military"),
    ("موسیقی", "music"),
    ("معماری", "architecture"),
    ("رایانه", "computing"),
    ("رایانگ", "computing"),
    ("مخابرات", "computing"),
    ("ورزش", "sport"),
    ("فوتبال", "sport"),
    ("اقتصاد", "economics"),
    ("بازرگانی", "trade"),
    ("تجارت", "trade"),
    ("زمین‌شناسی", "geology"),
    ("جغرافیا", "geography"),
    ("کشاورزی", "agriculture"),
    ("دامپرور", "agriculture"),
    ("نجوم", "astronomy"),
    ("فلسفه", "philosophy"),
    ("منطق", "philosophy"),
    ("زبان‌شناسی", "linguistics"),
    ("دستور", "linguistics"),
    ("هنر", "art"),
    ("خوراک", "food"),
    ("آشپزی", "food"),
    ("پوشاک", "textile"),
    ("نساجی", "textile"),
    ("حمل و نقل", "transport"),
    ("ترابری", "transport"),
    ("هواشناسی", "meteorology"),
    ("روان‌شناسی", "psychology"),
    ("آموزش", "education"),
    ("سیاست", "politics"),
    ("مدیریت", "management"),
    ("مهندسی", "engineering"),
    ("برق", "engineering"),
    ("خودرو", "engineering"),
    ("هوافضا", "aerospace"),
)

#: Language names whose presence in a category marks the page as documenting a
#: *foreign* word rather than a Persian one. Arabic is absent on purpose - see
#: the module docstring.
_FOREIGN: Final[frozenset[str]] = frozenset(
    {
        "انگلیسی",
        "فرانسوی",
        "فرانسه",
        "آلمانی",
        "ایتالیایی",
        "اسپانیایی",
        "پرتغالی",
        "روسی",
        "اوکراینی",
        "لهستانی",
        "چکی",
        "مجاری",
        "رومانیایی",
        "یونانی",
        "لاتین",
        "عبری",
        "ژاپنی",
        "چینی",
        "کره‌ای",
        "هندی",
        "اردو",
        "سانسکریت",
        "بنگالی",
        "ارمنی",
        "گرجی",
        "سوئدی",
        "نروژی",
        "دانمارکی",
        "فنلاندی",
        "هلندی",
        "ترکی استانبولی",
        "اسپرانتو",
        "ولزی",
        "ایرلندی",
        "باسکی",
        "کاتالان",
        "اندونزیایی",
        "مالایی",
        "تایلندی",
        "ویتنامی",
        "سواحیلی",
        "آفریکانس",
        "ایسلندی",
        "آلبانیایی",
        "صربی",
        "کرواتی",
        "بلغاری",
        "اسلوونی",
        "استونیایی",
        "لتونی",
        "لیتوانیایی",
    }
)

#: Wiktionary namespaces other than 0 are templates, help pages, discussions.
_NS0: Final[re.Pattern[str]] = re.compile(r"<ns>0</ns>")

#: One page element. Non-greedy so a malformed dump cannot swallow the file.
_PAGE: Final[re.Pattern[str]] = re.compile(r"<page>(.*?)</page>", re.DOTALL)

#: Title and category extraction.
_TITLE: Final[re.Pattern[str]] = re.compile(r"<title>(.*?)</title>", re.DOTALL)
_CATEGORY: Final[re.Pattern[str]] = re.compile(r"\[\[\s*رده\s*:\s*([^\]\|]{1,80})")

#: Section headers, used as a POS fallback when categories say nothing.
_HEADER: Final[re.Pattern[str]] = re.compile(
    r"^={2,4}\s*([^=\n]{1,40}?)\s*={2,4}\s*$", re.MULTILINE
)

#: A headword must be a single token of Persian letters, optionally joined by
#: ZWNJ. Spaces, digits and Latin letters disqualify it: multi-word phrases and
#: transliterations are a different kind of list and would skew the prompts.
_HEADWORD: Final[re.Pattern[str]] = re.compile(
    rf"^[\u0621-\u063a\u0641-\u064a\u0670-\u06d3\u06f0-\u06f9{ZWNJ}]+$"
)


@dataclass(slots=True)
class Headword:
    """One harvested word and everything known about it.

    Mutable because the merge step accumulates evidence from several sources
    into one record: Lilak supplies the POS, Wiktionary the era and domain, and
    the frequency pass the band. A frozen type would mean rebuilding the record
    three times per word across ~150k words.

    Attributes:
        word: The headword, normalised by the project's single policy.
        pos: One of :data:`POS_LABELS`.
        era: One of :data:`fa_ocr_lora.diverse.config.ERAS`.
        domain: Domain label, or ``""`` when unknown.
        sources: Source names that contributed, in discovery order.
        categories: Raw Wiktionary categories, kept for auditing a surprising
            tag without re-parsing the dump.
        frequency: Occurrences in the frequency corpus; ``-1`` until stage 00's
            frequency pass fills it.
        rank: Frequency rank, ``1`` being the most common; ``-1`` until filled.
        freq_band: One of :data:`fa_ocr_lora.diverse.config.BANDS`.
    """

    word: str
    pos: str = "other"
    era: str = "unlabelled"
    domain: str = ""
    sources: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    frequency: int = -1
    rank: int = -1
    freq_band: str = "very_rare"

    def add_source(self, name: str) -> None:
        """Record that a source contributed this word.

        Args:
            name: Source name.
        """
        if name not in self.sources:
            self.sources.append(name)


def is_persian_headword(word: str, *, minimum: int, maximum: int) -> bool:
    """Report whether a string is usable as a Persian headword.

    Args:
        word: Candidate string, already normalised.
        minimum: Shortest acceptable length in characters.
        maximum: Longest acceptable length in characters.

    Returns:
        ``True`` when the string is a single Persian-script token of acceptable
        length. Length is measured without ZWNJ, so ``می‌رود`` counts as five
        letters rather than six - otherwise the limits would quietly favour
        words that happen not to be compounds.
    """
    if not word or not _HEADWORD.match(word):
        return False
    bare = word.replace(ZWNJ, "")
    if not minimum <= len(bare) <= maximum:
        return False
    # A token of only digits is a number, not a word.
    return persian_letter_ratio(bare) >= 0.999 and any(
        character.isalpha() for character in bare
    )


def tag_categories(categories: Sequence[str]) -> tuple[str, str, str, bool]:
    """Derive POS, era, domain and a foreign-word flag from raw categories.

    Args:
        categories: Raw Wiktionary category names for one page.

    Returns:
        ``(pos, era, domain, foreign)``. ``pos`` and ``domain`` are ``"other"``
        and ``""`` when nothing matched; ``era`` is ``"unlabelled"``.
        ``foreign`` is ``True`` when a category names a non-Persian language,
        in which case the caller should drop the word unless another source
        confirms it is Persian.
    """
    joined = " | ".join(categories)
    foreign = any(language in joined for language in _FOREIGN)

    pos = "other"
    for fragment, label in _POS_PATTERNS:
        if fragment in joined:
            pos = label
            break

    era = "unlabelled"
    for fragment, label in _ERA_PATTERNS:
        if fragment in joined:
            era = label
            break

    domain = ""
    for fragment, label in _DOMAIN_PATTERNS:
        if fragment in joined:
            domain = label
            break

    # `اصطلاحات X` means "terminology of X". If a domain was matched the word is
    # technical by construction; if the category says `اصطلاحات` but names no
    # domain we know, it is still technical vocabulary.
    if era == "unlabelled" and (domain or "اصطلاحات" in joined):
        era = "technical"

    return pos, era, domain, foreign


def parse_lilak(path: Path, *, minimum: int, maximum: int) -> Iterator[Headword]:
    """Parse the Lilak lexicon CSV.

    The file is ``word,pos,...`` with ``#``-prefixed licence header lines.

    Args:
        path: The downloaded ``lexicon`` file.
        minimum: Shortest acceptable headword length.
        maximum: Longest acceptable headword length.

    Yields:
        One :class:`Headword` per accepted row, with ``pos`` filled and ``era``
        left ``unlabelled`` - Lilak carries no era metadata, and inventing one
        would be worse than recording its absence.
    """
    kept = 0
    seen = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            seen += 1
            word = normalise_for_eval(row[0].strip())
            if not is_persian_headword(word, minimum=minimum, maximum=maximum):
                continue
            raw_pos = row[1].strip().lower() if len(row) > 1 else ""
            entry = Headword(word=word, pos=_LILAK_POS.get(raw_pos, "other"))
            entry.add_source("lilak")
            kept += 1
            yield entry
    LOGGER.info("lilak: %d/%d row(s) accepted", kept, seen)


def parse_wiktionary(
    chunks: Iterable[str], *, minimum: int, maximum: int
) -> Iterator[Headword]:
    """Parse a Persian Wiktionary XML dump stream.

    Reads the stream incrementally and emits a word as soon as its ``</page>``
    arrives, so the 18 MB dump never needs to be held in memory as one string.

    Args:
        chunks: Decoded text chunks, as produced by
            :func:`fa_ocr_lora.diverse.lexicon.sources.stream_bz2_text`. Chunks
            need not align to element boundaries.
        minimum: Shortest acceptable headword length.
        maximum: Longest acceptable headword length.

    Yields:
        One :class:`Headword` per accepted page, with ``era``, ``domain`` and
        ``categories`` filled. Foreign-language entries are emitted with
        ``era`` set to ``"foreign"`` so the merge step can drop them unless
        Lilak confirms the word; that decision does not belong here.
    """
    buffer = ""
    pages = 0
    kept = 0
    foreign = 0
    for chunk in chunks:
        buffer += chunk
        while True:
            match = _PAGE.search(buffer)
            if match is None:
                break
            body = match.group(1)
            buffer = buffer[match.end() :]
            pages += 1

            if not _NS0.search(body):
                continue
            title_match = _TITLE.search(body)
            if title_match is None:
                continue
            word = normalise_for_eval(title_match.group(1).strip())
            if not is_persian_headword(word, minimum=minimum, maximum=maximum):
                continue

            categories = [name.strip() for name in _CATEGORY.findall(body)]
            pos, era, domain, is_foreign = tag_categories(categories)
            if pos == "other":
                # Fall back to the section headers, which name the POS even on
                # pages nobody has categorised.
                headers = " | ".join(_HEADER.findall(body))
                for fragment, label in _POS_PATTERNS:
                    if fragment in headers:
                        pos = label
                        break
            if is_foreign:
                foreign += 1
                era = "foreign"

            entry = Headword(
                word=word, pos=pos, era=era, domain=domain, categories=categories[:12]
            )
            entry.add_source("fa_wiktionary")
            kept += 1
            yield entry

    # Keep only the tail that might still hold a partial page.
    LOGGER.info(
        "fa_wiktionary: %d page(s) read, %d accepted, %d flagged foreign",
        pages,
        kept,
        foreign,
    )


def parse_wikipedia_titles(
    lines: Iterable[str], *, minimum: int, maximum: int
) -> Iterator[Headword]:
    """Parse the Persian Wikipedia ``all-titles-in-ns0`` list.

    Only single-token titles are kept. A multi-word title is an article name
    (``جنگ جهانی دوم``), not a headword, and injecting one into a prompt asks
    the model to work a phrase in rather than a word.

    Args:
        lines: Lines of the decompressed titles file; underscores separate
            words, and the first line is a header.
        minimum: Shortest acceptable headword length.
        maximum: Longest acceptable headword length.

    Yields:
        One :class:`Headword` per accepted title, tagged ``technical`` - an
        ordinary Persian word rarely needs its own encyclopedia article, and
        the top-rank frequency exclusion removes the ones that do.
    """
    kept = 0
    seen = 0
    for line in lines:
        seen += 1
        raw = line.strip()
        if not raw or raw == "page_title" or "_" in raw:
            continue
        word = normalise_for_eval(raw)
        if not is_persian_headword(word, minimum=minimum, maximum=maximum):
            continue
        entry = Headword(word=word, era="technical")
        entry.add_source("fa_wikipedia_titles")
        kept += 1
        yield entry
    LOGGER.info("fa_wikipedia_titles: %d/%d title(s) accepted", kept, seen)
