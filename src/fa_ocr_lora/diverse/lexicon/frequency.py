"""Word frequency over a Persian Wikipedia sample, and the band assignment.

Frequencies are computed here rather than downloaded from a published list for
one reason: **the bands must match this pipeline's tokenisation**. A published
frequency list tokenises its own way, treats ZWNJ its own way, and normalises
Arabic yeh its own way. Any of those disagreements shifts a word across a band
boundary, and the band is what decides whether the word is injected into
prompts at all.

The pass serves two purposes, and the second is the one that matters most
after the user's observation that an LLM writes high-frequency words
unprompted:

1. **Band assignment.** Percentile cut-offs over the rank distribution name
   each word ``very_common`` through ``very_rare``.
2. **Exclusion.** Words inside ``exclude_top_frequency_rank`` are removed from
   the *sampling pool* entirely. They stay in ``lexicon.parquet`` - the
   frequency table is what proves they are common - but they are never spent
   on a prompt slot, because the model supplies them free of charge.

Cost control: the article dump is streamed and decompressed incrementally and
the reader stops as soon as ``max_documents`` articles have been tokenised, so
the download is only as large as the sample needs.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Final

from ..text_policy import ZWNJ, normalise_for_eval

__all__ = [
    "FrequencyTable",
    "assign_bands",
    "count_corpus",
    "tokenise",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: The article body inside a MediaWiki XML dump.
_TEXT: Final[re.Pattern[str]] = re.compile(r"<text[^>]*>(.*?)</text>", re.DOTALL)

#: Wikitext constructs stripped before tokenising. Templates, references, HTML
#: comments, tables and file links are markup, not prose, and counting their
#: contents inflates exactly the technical vocabulary the bands must rank
#: honestly.
_MARKUP: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<ref[^>]*/\s*>", re.IGNORECASE),
    re.compile(r"\{\{[^{}]*\}\}"),
    re.compile(r"\{\|.*?\|\}", re.DOTALL),
    re.compile(
        r"\[\[(?:پرونده|تصویر|رده|File|Image|Category)\s*:[^\]]*\]\]", re.IGNORECASE
    ),
    re.compile(r"<[^>]{1,200}>"),
    re.compile(r"https?://\S+"),
)

#: Link syntax, reduced to its display text rather than removed: ``[[سامانه]]``
#: is a real occurrence of a real word.
_LINK: Final[re.Pattern[str]] = re.compile(r"\[\[(?:[^\]\|]*\|)?([^\]\|]*)\]\]")

#: A Persian word: letters and ZWNJ only. Digits and Latin are excluded because
#: the lexicon holds neither.
_TOKEN: Final[re.Pattern[str]] = re.compile(
    rf"[\u0621-\u063a\u0641-\u064a\u0670-\u06d3{ZWNJ}]{{2,}}"
)


def _strip_markup(text: str) -> str:
    """Reduce wikitext to approximate prose.

    Args:
        text: Raw wikitext.

    Returns:
        The text with templates, references, tables, file links and HTML
        removed, and links reduced to their display text.
    """
    out = text
    for pattern in _MARKUP:
        out = pattern.sub(" ", out)
    out = _LINK.sub(r"\1", out)
    return out.replace("'''", " ").replace("''", " ")


def tokenise(text: str) -> list[str]:
    """Split Persian prose into normalised word tokens.

    Normalisation runs through the project's single policy, so a word counted
    here is the same string the lexicon stage stores and the plan stage
    injects. ZWNJ is preserved: ``می‌رود`` and ``میرود`` are genuinely
    different spellings and conflating them would misreport both.

    Args:
        text: Prose, markup already stripped.

    Returns:
        The tokens, in order.
    """
    return _TOKEN.findall(normalise_for_eval(text))


def _iter_articles(chunks: Iterable[str], limit: int) -> Iterator[str]:
    """Yield article bodies from a streamed MediaWiki dump.

    Args:
        chunks: Decoded text chunks, not aligned to element boundaries.
        limit: Stop after this many articles.

    Yields:
        One article's wikitext at a time.
    """
    buffer = ""
    seen = 0
    for chunk in chunks:
        buffer += chunk
        while True:
            match = _TEXT.search(buffer)
            if match is None:
                break
            body = match.group(1)
            buffer = buffer[match.end() :]
            seen += 1
            yield body
            if seen >= limit:
                return
        # A buffer with no complete <text> element can still grow without
        # bound on a malformed dump; cap it at a few megabytes, which is far
        # larger than any single article.
        if len(buffer) > 8_000_000:
            buffer = buffer[-1_000_000:]


class FrequencyTable:
    """Token counts over the sampled corpus, with ranks.

    Attributes:
        counts: Token to occurrence count.
        documents: Articles tokenised.
        tokens: Total tokens counted.
    """

    __slots__ = ("counts", "documents", "tokens", "_ranks")

    def __init__(self, counts: Counter[str], documents: int, tokens: int) -> None:
        """Store the counts and build the rank index.

        Args:
            counts: Token to occurrence count.
            documents: Articles tokenised.
            tokens: Total tokens counted.
        """
        self.counts = counts
        self.documents = documents
        self.tokens = tokens
        #: Rank 1 is the most frequent. Ties are broken by the token itself so
        #: the ranking is deterministic across runs - otherwise the exclusion
        #: boundary would wobble and two runs would inject different words.
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        self._ranks: dict[str, int] = {
            token: index for index, (token, _) in enumerate(ordered, start=1)
        }

    def __len__(self) -> int:
        """Return the number of distinct tokens."""
        return len(self.counts)

    def frequency_of(self, word: str) -> int:
        """Return a word's occurrence count.

        Args:
            word: Normalised token.

        Returns:
            The count, or ``0`` when the word never occurred.
        """
        return self.counts.get(word, 0)

    def rank_of(self, word: str) -> int:
        """Return a word's frequency rank.

        Args:
            word: Normalised token.

        Returns:
            The rank, ``1`` being most frequent, or ``0`` when the word never
            occurred. Zero is used rather than infinity so the value fits an
            integer column, and callers test ``rank > 0`` before comparing.
        """
        return self._ranks.get(word, 0)

    def summary(self) -> dict[str, int]:
        """Return a loggable summary.

        Returns:
            Document, token and vocabulary counts.
        """
        return {
            "documents": self.documents,
            "tokens": self.tokens,
            "vocabulary": len(self.counts),
        }


def count_corpus(chunks: Iterable[str], *, max_documents: int) -> FrequencyTable:
    """Tokenise a streamed dump and count words.

    Args:
        chunks: Decoded text chunks from the article dump.
        max_documents: Articles to tokenise before stopping. The generator is
            abandoned at that point, which closes the HTTP connection and ends
            the download.

    Returns:
        The frequency table.
    """
    counts: Counter[str] = Counter()
    documents = 0
    tokens = 0
    for body in _iter_articles(chunks, max_documents):
        documents += 1
        words = tokenise(_strip_markup(body))
        counts.update(words)
        tokens += len(words)
        if documents % 50_000 == 0:
            LOGGER.info(
                "  %d article(s), %d token(s), %d distinct",
                documents,
                tokens,
                len(counts),
            )
    LOGGER.info(
        "frequency: %d article(s), %d token(s), %d distinct word(s)",
        documents,
        tokens,
        len(counts),
    )
    return FrequencyTable(counts, documents, tokens)


def assign_bands(
    words: Sequence[str],
    table: FrequencyTable,
    percentiles: Mapping[str, float],
) -> dict[str, str]:
    """Assign a frequency band to every word.

    Bands are percentiles **of the lexicon's own words**, not of the corpus
    vocabulary. That distinction matters: the corpus vocabulary is dominated by
    inflected forms and proper nouns that never enter the lexicon, so using it
    would push almost every dictionary headword into ``very_rare`` and make the
    bands useless for stratification.

    A word absent from the corpus is ``very_rare`` by definition - it is a
    dictionary word that 400k Wikipedia articles never used, which is exactly
    the kind this corpus wants.

    Args:
        words: Lexicon headwords.
        table: Frequency table from :func:`count_corpus`.
        percentiles: Descending cut-offs keyed by band name, as validated by
            :class:`~fa_ocr_lora.diverse.config.LexiconConfig`.

    Returns:
        Word to band name.
    """
    present = sorted(
        (
            (word, table.frequency_of(word))
            for word in words
            if table.frequency_of(word) > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )
    total = len(present)
    bands: dict[str, str] = {}

    if total == 0:
        LOGGER.warning(
            "no lexicon word occurred in the frequency corpus; "
            "every word banded very_rare"
        )
        return {word: "very_rare" for word in words}

    # Walk the lexicon's own words from most to least frequent and cut at the
    # configured percentiles.
    cuts = [
        ("very_common", 1.0 - percentiles["very_common"]),
        ("common", 1.0 - percentiles["common"]),
        ("uncommon", 1.0 - percentiles["uncommon"]),
        ("rare", 1.0 - percentiles["rare"]),
    ]
    for index, (word, _) in enumerate(present):
        position = index / total
        label = "very_rare"
        for name, cut in cuts:
            if position <= cut:
                label = name
                break
        bands[word] = label

    for word in words:
        bands.setdefault(word, "very_rare")
    return bands
