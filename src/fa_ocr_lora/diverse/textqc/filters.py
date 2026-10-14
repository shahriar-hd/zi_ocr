"""Stage 03 filters: every reason a generated text is rejected.

Each filter returns a reason string or ``None``. The reason is logged per
sample, so the rejection report names *why* the corpus shrank rather than just
by how much - which is the difference between tuning a threshold and guessing
at one.

The hard-won filter here is :func:`check_foreign_script`. The first live
generation run produced Cyrillic and CJK inside Persian prose in 4 of 12
samples - ``наблюده‌ها``, ``phénomènes``, ``θερمولویجیکی`` - *even with* an
explicit Persian-only instruction in the prompt. A ratio threshold does not
catch it: one such sample measured 80.6% Persian letters and would have passed
a 0.80 floor while being unusable. So script contamination is checked as a
*kind*, not a quantity: a single Cyrillic letter in a Persian sample is a
defect, because no Persian text contains one and the model would be taught to
emit it.

Latin is different, and is treated differently. Persian technical and
commercial writing really does contain Latin terms, so Latin is permitted when
the plan asked for it (``latin_terms``, or the ``mixed`` category) and capped
otherwise.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping
from typing import Final

from ..config import TextQcConfig
from ..planning.sampler import Plan
from ..text_policy import (
    normalise_for_eval,
    persian_letter_ratio,
    stems,
    word_count,
)

__all__ = [
    "REJECTION_REASONS",
    "check_foreign_script",
    "check_latin",
    "check_length",
    "check_lexicon_usage",
    "check_persian_ratio",
    "check_pii",
    "check_preamble",
    "check_refusal",
    "check_repetition",
    "run_filters",
    "strip_artefacts",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: Every rejection reason this module can produce, for the report's columns.
REJECTION_REASONS: Final[tuple[str, ...]] = (
    "empty",
    "foreign_script",
    "latin_unwanted",
    "persian_ratio",
    "preamble",
    "refusal",
    "repetition",
    "length",
    "lexicon_usage",
    "pii",
    "duplicate",
)

#: Scripts that are never legitimate in a Persian sample. Any occurrence is a
#: rejection regardless of ratio: the model leaked another language's alphabet,
#: and a label containing it teaches exactly that leak.
#:
#: Latin is deliberately absent - see :func:`check_latin`.
_FOREIGN_SCRIPTS: Final[Mapping[str, re.Pattern[str]]] = {
    "cyrillic": re.compile(r"[\u0400-\u04ff\u0500-\u052f]"),
    "greek": re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]"),
    "han": re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"),
    "kana": re.compile(r"[\u3040-\u30ff]"),
    "hangul": re.compile(r"[\uac00-\ud7af\u1100-\u11ff]"),
    "hebrew": re.compile(r"[\u0590-\u05ff]"),
    "devanagari": re.compile(r"[\u0900-\u097f]"),
    "armenian": re.compile(r"[\u0530-\u058f]"),
    "georgian": re.compile(r"[\u10a0-\u10ff]"),
    "thai": re.compile(r"[\u0e00-\u0e7f]"),
    "ethiopic": re.compile(r"[\u1200-\u137f]"),
}

#: Latin letters, counted rather than forbidden.
_LATIN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]")

#: Markdown and fencing artefacts. The model was told not to emit these; when
#: it does they are stripped, because the alternative - rejecting the sample -
#: throws away otherwise good text over a pair of asterisks.
_ARTEFACTS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"^```[a-zA-Z]*\s*$", re.MULTILINE), ""),
    (re.compile(r"```"), ""),
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1"),
    (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL), r"\1"),
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    (re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE), ""),
    (re.compile(r"<\s*/?\s*(?:br|p|div|span|b|i|strong|em)\s*/?\s*>", re.IGNORECASE), ""),
    # A thinking block that the endpoint failed to suppress.
    (re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE), ""),
    (re.compile(r"<\|.*?\|>", re.DOTALL), ""),
)

#: Preamble openers. A model that says "here is the text you asked for" has put
#: English or Persian scaffolding into what will become the label.
_PREAMBLES: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"^\s*(?:here\s+is|here's|sure[,!]|certainly[,!]|of\s+course)", re.IGNORECASE
    ),
    re.compile(r"^\s*(?:در\s+اینجا|در\s+زیر|بفرمایید|حتماً|البته)\b"),
    re.compile(r"^\s*(?:متن\s+(?:زیر|درخواستی|مورد\s+نظر)|این\s+متن)\s*[:،]"),
    re.compile(r"^\s*(?:پاسخ|خروجی|نتیجه)\s*[:：]"),
    re.compile(r"^\s*\**\s*(?:title|عنوان)\s*[:：]", re.IGNORECASE),
)

#: Trailing notes, which are the same problem at the other end.
_TRAILERS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\n\s*(?:note|توضیح|یادداشت|نکته)\s*[:：].*$", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"\n\s*(?:امیدوارم|اگر\s+نیاز|در\s+صورت\s+نیاز|بگویید\s+تا).*$", re.DOTALL
    ),
    re.compile(r"\n\s*(?:let me know|hope this|feel free).*$", re.IGNORECASE | re.DOTALL),
)

#: Refusals. Short and unmistakable; a refusal is not text to render.
_REFUSALS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"\b(?:i\s+can(?:'|no)t|i'm\s+unable|as\s+an\s+ai|i\s+apologi[sz]e)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:نمی‌توانم|نمیتوانم|قادر\s+نیستم|متأسفانه\s+نمی)"),
    re.compile(r"\b(?:sorry,?\s+(?:but|i))\b", re.IGNORECASE),
)

#: Patterns that look like real Iranian personal data. The prompt asks for
#: invented values, and invented values are what these catch too - which is
#: fine, because the point is not to detect real people but to keep anything
#: shaped like a national ID or a mobile number out of a published corpus.
_PII: Final[Mapping[str, re.Pattern[str]]] = {
    "national_id": re.compile(r"(?<!\d)\d{10}(?!\d)"),
    "mobile": re.compile(r"(?<!\d)(?:\+?98|0)9\d{9}(?!\d)"),
    "iban": re.compile(r"\bIR\d{24}\b", re.IGNORECASE),
    "card": re.compile(r"(?<!\d)\d{16}(?!\d)"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
}

#: Converted to Latin digits before the PII check, so a Persian-digit national
#: ID is caught too.
_DIGIT_FOLD: Final[dict[int, str]] = {
    **{0x06F0 + i: str(i) for i in range(10)},
    **{0x0660 + i: str(i) for i in range(10)},
}


def strip_artefacts(text: str) -> str:
    """Remove markdown, HTML and thinking-block artefacts.

    Stripped rather than rejected: the model was told not to emit these, but a
    pair of stray asterisks is not a reason to discard a page of otherwise
    good Persian. A *preamble* is, because it changes what the text says.

    Args:
        text: The raw model reply.

    Returns:
        The text with artefacts removed and surrounding quotes unwrapped.
    """
    out = text
    for pattern, replacement in _ARTEFACTS:
        out = pattern.sub(replacement, out)
    for pattern in _TRAILERS:
        out = pattern.sub("", out)
    out = out.strip()
    # A whole reply wrapped in quotes is the model quoting itself; the quotes
    # are not part of the text that goes on the page.
    for opener, closer in (('"', '"'), ("«", "»"), ("'", "'"), ("“", "”")):
        if len(out) > 2 and out.startswith(opener) and out.endswith(closer):
            inner = out[1:-1]
            if opener not in inner and closer not in inner:
                out = inner.strip()
    return out


def check_foreign_script(text: str) -> str | None:
    """Reject any sample containing a script Persian never uses.

    Args:
        text: The candidate text.

    Returns:
        A reason naming the script and a sample of the offending characters,
        or ``None`` when clean.
    """
    found: list[str] = []
    for name, pattern in _FOREIGN_SCRIPTS.items():
        matches = pattern.findall(text)
        if matches:
            sample = "".join(dict.fromkeys(matches))[:8]
            found.append(f"{name}({len(matches)}):{sample}")
    if found:
        return "foreign_script: " + ", ".join(found)
    return None


def check_latin(text: str, plan: Plan, *, allowance: float = 0.06) -> str | None:
    """Reject unrequested Latin script beyond a small allowance.

    A little Latin is normal in Persian writing - a unit, a brand, an acronym -
    so a hard ban would reject good text. A lot of Latin in a sample that did
    not ask for it means the model switched languages.

    Args:
        text: The candidate text.
        plan: The sample's plan, which says whether Latin was wanted.
        allowance: Permitted Latin letters as a fraction of all letters when
            Latin was not requested.

    Returns:
        A reason, or ``None``.
    """
    if "latin_terms" in plan.features or plan.category == "mixed":
        return None
    latin = len(_LATIN.findall(text))
    if latin == 0:
        return None
    letters = sum(1 for character in text if character.isalpha())
    if letters == 0:
        return None
    share = latin / letters
    if share > allowance:
        return f"latin_unwanted: {share:.2f} of letters are Latin ({latin} characters)"
    return None


def check_persian_ratio(text: str, plan: Plan, settings: TextQcConfig) -> str | None:
    """Reject text that is not mostly Persian.

    Args:
        text: The candidate text.
        plan: The sample's plan.
        settings: Stage 03 thresholds.

    Returns:
        A reason, or ``None``.
    """
    mixed = "latin_terms" in plan.features or plan.category == "mixed"
    floor = settings.persian_ratio_min_mixed if mixed else settings.persian_ratio_min
    ratio = persian_letter_ratio(text)
    if ratio < floor:
        return f"persian_ratio: {ratio:.3f} below {floor:.2f}"
    return None


def check_preamble(text: str) -> str | None:
    """Reject text that opens with scaffolding rather than content.

    Args:
        text: The candidate text, already passed through
            :func:`strip_artefacts`.

    Returns:
        A reason, or ``None``.
    """
    head = text[:160]
    for pattern in _PREAMBLES:
        match = pattern.search(head)
        if match:
            return f"preamble: {match.group(0).strip()!r}"
    return None


def check_refusal(text: str) -> str | None:
    """Reject a refusal.

    Args:
        text: The candidate text.

    Returns:
        A reason, or ``None``.
    """
    for pattern in _REFUSALS:
        match = pattern.search(text)
        if match:
            return f"refusal: {match.group(0).strip()!r}"
    return None


def check_repetition(text: str, settings: TextQcConfig) -> str | None:
    """Reject degenerate repetition.

    Two tests, because they catch different failures: an identical-line counter
    finds a model stuck in a loop, and an n-gram ratio finds the subtler case
    where the text paraphrases itself indefinitely.

    Args:
        text: The candidate text.
        settings: Stage 03 thresholds.

    Returns:
        A reason, or ``None``.
    """
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 12]
    if lines:
        line_counts = Counter(lines)
        worst_line, worst_count = line_counts.most_common(1)[0]
        if worst_count > settings.max_line_repeat:
            return (
                f"repetition: a line repeats {worst_count} times "
                f"(limit {settings.max_line_repeat}): {worst_line[:48]!r}"
            )

    tokens = normalise_for_eval(text).split()
    size = settings.ngram_size
    if len(tokens) >= size * 3:
        grams = [tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]
        counts = Counter(grams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        ratio = repeated / len(grams)
        if ratio > settings.max_ngram_ratio:
            return (
                f"repetition: {ratio:.3f} of {size}-grams repeat "
                f"(limit {settings.max_ngram_ratio:.2f})"
            )
    return None


def check_length(text: str, plan: Plan, settings: TextQcConfig) -> str | None:
    """Reject text far outside its length target.

    The tolerance is generous and asymmetric in effect: "about 40 to 80 words"
    is a soft request, and a model that writes 95 has not failed. One that
    writes 900 has ignored the instruction, and the layout planned for 80 words
    cannot hold it.

    Args:
        text: The candidate text.
        plan: The sample's plan.
        settings: Stage 03 thresholds.

    Returns:
        A reason, or ``None``.
    """
    words = word_count(text)
    if words == 0:
        return "length: no words"
    tolerance = settings.length_tolerance
    floor = max(1, int(plan.length_min * (1.0 - tolerance)))
    ceiling = int(plan.length_max * (1.0 + tolerance)) + 2
    if words < floor:
        return (
            f"length: {words} word(s), below {floor} "
            f"(target {plan.length_min}-{plan.length_max})"
        )
    if words > ceiling:
        return (
            f"length: {words} word(s), above {ceiling} "
            f"(target {plan.length_min}-{plan.length_max})"
        )
    return None


def check_lexicon_usage(
    text: str, plan: Plan, settings: TextQcConfig
) -> tuple[str | None, float]:
    """Check that the requested lexicon words actually appear.

    This is the filter that decides whether stage 00 and 01 did any good. A
    sample that ignored its rare words is an ordinary LLM paragraph, and the
    whole corpus would drift back toward the model's own vocabulary.

    Matching is done on light stems, because the words are given as headwords
    and used inflected: ``سگالش`` may legitimately appear as ``سگالشی``.

    Args:
        text: The candidate text.
        plan: The sample's plan.
        settings: Stage 03 thresholds.

    Returns:
        ``(reason, hit_rate)``. ``reason`` is ``None`` when the hit rate meets
        the floor. ``hit_rate`` is always reported, because the distribution is
        more informative than the pass/fail.
    """
    if not plan.words:
        return None, 1.0
    haystack = normalise_for_eval(text)
    haystack_stems = stems(haystack.split())
    hits = 0
    for word in plan.words:
        needle = normalise_for_eval(word)
        if needle and needle in haystack:
            hits += 1
            continue
        # Fall back to stem overlap for an inflected form.
        if stems([word]) & haystack_stems:
            hits += 1
    rate = hits / len(plan.words)
    if rate < settings.lexicon_hit_rate_min:
        return (
            f"lexicon_usage: {hits}/{len(plan.words)} requested word(s) used "
            f"({rate:.2f} below {settings.lexicon_hit_rate_min:.2f})"
        ), rate
    return None, rate


def check_pii(text: str) -> str | None:
    """Reject anything shaped like real personal data.

    Args:
        text: The candidate text.

    Returns:
        A reason naming the pattern, or ``None``.
    """
    folded = text.translate(_DIGIT_FOLD)
    for name, pattern in _PII.items():
        if pattern.search(folded):
            return f"pii: matched {name}"
    return None


def run_filters(
    text: str, plan: Plan, settings: TextQcConfig
) -> tuple[str | None, dict[str, object]]:
    """Apply every single-sample filter in order.

    Order is cheapest-and-most-decisive first, so a Cyrillic-contaminated
    sample costs one regex rather than an n-gram pass.

    Args:
        text: The candidate text, already passed through
            :func:`strip_artefacts`.
        plan: The sample's plan.
        settings: Stage 03 thresholds.

    Returns:
        ``(reason, metrics)``. ``reason`` is ``None`` when the sample passes.
        ``metrics`` always carries the measured values, so the report can show
        distributions for samples that passed as well as those that did not.
    """
    hit_reason, hit_rate = check_lexicon_usage(text, plan, settings)
    metrics: dict[str, object] = {
        "n_words": word_count(text),
        "n_chars": len(text),
        "persian_ratio": round(persian_letter_ratio(text), 4),
        "lexicon_hit_rate": round(hit_rate, 4),
    }

    if not text.strip():
        return "empty", metrics
    for reason in (
        check_foreign_script(text),
        check_refusal(text),
        check_preamble(text),
        check_latin(text, plan),
        check_persian_ratio(text, plan, settings),
        check_pii(text),
        check_length(text, plan, settings),
        check_repetition(text, settings),
        hit_reason,
    ):
        if reason is not None:
            return reason, metrics
    return None, metrics
