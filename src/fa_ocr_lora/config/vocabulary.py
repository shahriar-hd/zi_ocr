"""Persian content pools, loaded from ``configs/vocabulary_fa.yaml``.

This module holds the *words* a generated letter is built from - organisational
prefixes, religious openers, footer labels, seal wordings, table column headers
- as distinct from ``config.py``, which holds the *numbers* and probabilities
that decide how they are arranged. They are separate files because they are
edited by different people for different reasons: retuning a logo size is a
layout decision, while rewording ``شماره اندیکاتور`` is a question about how
Iranian administrative correspondence is actually written.

Two properties of every pool here matter more than they look:

*Order is load-bearing.* A weighted pool is drawn with exactly one
``rng.random()`` and an unweighted one with ``rng.choice``, so entries may be
reworded or reweighted freely, but *reordering* a pool changes which phrase a
given seed picks. Python preserves YAML mapping order, so the file's order is
the pool's order.

*Whitespace is content.* A prefix like ``"موضوع: "`` is concatenated straight
onto the subject that follows it, so its trailing space is part of the value.
This is why the pools here are parsed by :func:`_parse_phrase_pool` rather than
by ``config.py``'s ``_parse_weight_pool``, which strips names and rejects the
empty one: an identifier pool wants that normalisation and a phrase pool is
broken by it. The empty phrase is a real, common choice - it is how a letter
prints a bare brand name with no ``شرکت``, or a value with no separator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import yaml

from ..dataset.common.io_utils import resolve_repo_path
from .base import CHART_KINDS, ConfigError

__all__ = [
    "DEFAULT_VOCABULARY_PATH",
    "VOCABULARY_PATH_ENV_VAR",
    "ChartVocabulary",
    "ElementsVocabulary",
    "HeaderVocabulary",
    "LetterVocabulary",
    "MetadataVocabulary",
    "StampVocabulary",
    "TableVocabulary",
    "Vocabulary",
    "load_vocabulary",
]

#: Repository-relative default, used when neither a path nor the environment
#: variable names one.
DEFAULT_VOCABULARY_PATH: Final[str] = "configs/vocabulary_fa.yaml"

#: Variable naming the vocabulary file, honoured when no path is passed.
VOCABULARY_PATH_ENV_VAR: Final[str] = "FA_OCR_VOCABULARY"

# --------------------------------------------------------------------------- #
# Shipped pools
#
# These are the dataclass defaults, so every generator works with no file on
# disk - which is what lets the test suite call a builder directly. The
# committed YAML restates them, and ``test_vocabulary.py`` asserts the two
# agree, so the duplication cannot drift unnoticed.
# --------------------------------------------------------------------------- #

_NAME_PREFIXES: Final[tuple[tuple[str, float], ...]] = (
    ("", 34.0),
    ("شرکت", 30.0),
    ("گروه", 10.0),
    ("شرکت فناوری", 8.0),
    ("مؤسسه", 6.0),
    ("شرکت مهندسی", 6.0),
    ("هلدینگ", 6.0),
)

_RELIGIOUS_OPENERS: Final[tuple[tuple[str, float], ...]] = (
    ("بسمه تعالی", 46.0),
    ("به نام خدا", 24.0),
    ("بنام خداوند بخشنده مهربان", 12.0),
    ("هو الحق", 10.0),
    ("به نام خداوند جان و خرد", 8.0),
)

_SLOGANS: Final[tuple[str, ...]] = (
    "کیفیت، تعهد، نوآوری",
    "همراه همیشگی شما",
    "پیشرو در ارائه خدمات تخصصی",
    "با تکیه بر دانش و تجربه",
    "اعتماد شما، سرمایه ما",
    "نوآوری در خدمت مشتری",
    "راهکارهای هوشمند کسب‌وکار",
    "تعهد به تعالی سازمانی",
)

_CONTACT_LABELS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("address", ("نشانی", "آدرس")),
    ("phone", ("تلفن", "تلفن تماس", "شماره تماس")),
    ("email", ("رایانامه", "ایمیل", "پست الکترونیک")),
    ("website", ("وبگاه", "وب‌سایت", "نشانی اینترنتی")),
)

_PAGE_TEMPLATES: Final[tuple[str, ...]] = (
    "صفحه {page} از {total}",
    "{page} از {total}",
    "- {page} -",
    "صفحه {page}",
)

_RECIPIENT_PREFIXES: Final[tuple[tuple[str, float], ...]] = (
    ("", 30.0),
    ("به: ", 22.0),
    ("جناب آقای/سرکار خانم ", 8.0),
    ("ریاست محترم ", 18.0),
    ("مدیریت محترم ", 14.0),
)

_SENDER_PREFIXES: Final[tuple[tuple[str, float], ...]] = (
    ("از: ", 46.0),
    ("فرستنده: ", 30.0),
    ("", 24.0),
)

_SUBJECT_PREFIXES: Final[tuple[tuple[str, float], ...]] = (
    ("موضوع: ", 62.0),
    ("موضوع : ", 12.0),
    ("", 26.0),
)

_ATTACHMENT_TITLES: Final[tuple[str, ...]] = (
    "پیوست‌ها:",
    "فهرست پیوست‌ها:",
    "ضمائم:",
    "مدارک پیوست:",
)

_METADATA_LABELS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "letter_number",
        ("شماره", "شماره نامه", "شمارهٔ نامه", "شماره اندیکاتور"),
    ),
    ("registration_number", ("شماره ثبت", "شمارهٔ ثبت", "شماره ثبت نامه")),
    ("date", ("تاریخ", "تاریخ نامه", "مورخ")),
    ("attachment", ("پیوست", "ضمائم", "پیوست‌ها")),
    ("national_id", ("شناسه ملی", "شناسهٔ ملی", "شناسه ملی شرکت")),
)

_METADATA_LABEL_WEIGHTS: Final[tuple[float, ...]] = (60.0, 18.0, 14.0, 8.0)

_SEPARATORS: Final[tuple[tuple[str, float], ...]] = (
    (":", 62.0),
    ("", 20.0),
    ("ـــــ", 10.0),
    (".....", 8.0),
)

_ATTACHMENT_PRESENT: Final[tuple[str, ...]] = ("دارد", "دارد (۱ برگ)", "به پیوست")

_ATTACHMENT_ABSENT: Final[tuple[str, ...]] = ("ندارد", "-", "فاقد پیوست")

_CITIES: Final[tuple[str, ...]] = (
    "تهران",
    "اصفهان",
    "مشهد",
    "شیراز",
    "تبریز",
    "کرج",
    "قم",
    "اهواز",
    "رشت",
    "کرمان",
    "یزد",
    "ارومیه",
)

_SEAL_WORDS: Final[tuple[tuple[str, float], ...]] = (
    ("مهر رسمی", 26.0),
    ("", 24.0),
    ("ثبت شده", 18.0),
    ("مهر شرکت", 16.0),
    ("دبیرخانه", 16.0),
)

_TABLE_HEADER_SETS: Final[tuple[tuple[str, ...], ...]] = (
    ("ردیف", "شرح خدمات", "مبلغ (ریال)", "توضیحات"),
    ("ردیف", "عنوان", "تعداد", "مبلغ (ریال)"),
    ("ردیف", "مرحله اجرا", "زمان‌بندی", "درصد پیشرفت"),
    ("ردیف", "شرح هزینه", "مبلغ (ریال)"),
    ("ردیف", "نام مدرک", "تاریخ صدور", "وضعیت"),
    ("ردیف", "واحد سازمانی", "تعداد نیرو", "شرح وظیفه"),
)

_SERVICE_LABELS: Final[tuple[str, ...]] = (
    "طراحی و پیاده‌سازی سامانه",
    "پشتیبانی فنی و نگهداری",
    "به‌روزرسانی زیرساخت ابری",
    "آموزش کارکنان",
    "تأمین تجهیزات جانبی",
    "مشاوره فنی و امنیتی",
    "میزبانی و پهنای باند",
    "ممیزی امنیت اطلاعات",
    "تهیه مستندات فنی",
    "استقرار و راه‌اندازی",
)

_STAGE_LABELS: Final[tuple[str, ...]] = (
    "مطالعات امکان‌سنجی",
    "طراحی تفصیلی",
    "اجرای فاز نخست",
    "اجرای فاز دوم",
    "آزمون و تحویل موقت",
    "بهره‌برداری آزمایشی",
    "تحویل قطعی",
)

_DOCUMENT_LABELS: Final[tuple[str, ...]] = (
    "گواهی ثبت شرکت",
    "اساسنامه",
    "گواهی‌نامه ایزو ۹۰۰۱",
    "صورت‌های مالی حسابرسی‌شده",
    "گواهی حسن انجام کار",
    "بیمه‌نامه مسئولیت",
    "پروانه بهره‌برداری",
)

_UNIT_LABELS: Final[tuple[str, ...]] = (
    "اداره فناوری اطلاعات",
    "واحد پشتیبانی فنی",
    "دفتر برنامه‌ریزی",
    "امور مالی",
    "واحد حراست",
    "دبیرخانه مرکزی",
    "واحد آموزش",
)

_STATUS_VALUES: Final[tuple[str, ...]] = ("تأیید شده", "در دست بررسی", "تکمیل شده")

_DETAIL_VALUES: Final[tuple[str, ...]] = (
    "مطابق شرح خدمات پیوست",
    "بر اساس قرارداد جاری",
    "نیازمند تأیید کارفرما",
    "طبق برنامه زمان‌بندی",
    "-",
)

_MONTHS: Final[tuple[str, ...]] = (
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
)

_TABLE_CAPTIONS: Final[tuple[str, ...]] = (
    "جدول ۱: ریز اقلام و مبالغ",
    "جدول ۱: جزئیات اجرایی",
    "جدول ۱: خلاصه وضعیت",
    "جدول ۱: فهرست مدارک پیوست",
)

#: Caption used when the header row names amounts and the subject names costs.
_TABLE_COST_CAPTION: Final[str] = "جدول ۱: برآورد هزینه‌ها"

_SERIES_POOLS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "هزینه (میلیون ریال)",
        ("حقوق و دستمزد", "زیرساخت ابری", "تجهیزات", "آموزش", "پشتیبانی"),
    ),
    (
        "تعداد درخواست",
        ("فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور"),
    ),
    (
        "درصد پیشرفت",
        ("فاز نخست", "فاز دوم", "فاز سوم", "تحویل نهایی"),
    ),
    (
        "تعداد سامانه",
        ("مدیریت محتوا", "تحلیل داده", "میزبانی", "امنیت", "پشتیبان‌گیری"),
    ),
)

_CHART_CAPTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("bar", "نمودار ۱: مقایسه مقادیر"),
    ("line", "نمودار ۱: روند تغییرات"),
    ("pie", "نمودار ۱: سهم هر بخش"),
)


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #


def _parse_phrase_pool(
    payload: Any,
    context: str,
    default: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    """Parse a weighted pool of *phrases* from YAML.

    Deliberately not ``config.py``'s ``_parse_weight_pool``: that one strips
    each name and rejects the empty one, which is right for an identifier
    (``circle: 30``) and wrong for a phrase. ``"موضوع: "`` needs its trailing
    space because the subject is concatenated onto it, and ``""`` is a real
    entry meaning "print no prefix at all".

    Args:
        payload: A ``{phrase: weight}`` mapping, a sequence of
            ``{phrase, weight}`` mappings, or ``None`` for the default.
        context: Dotted config path, for error messages.
        default: Pool to use when the key is absent.

    Returns:
        The pool as ordered ``(phrase, weight)`` pairs, in file order.

    Raises:
        ConfigError: If the payload is neither shape, a phrase is not a string,
            or a weight is not a number.
    """
    if payload is None:
        return default
    if isinstance(payload, Mapping):
        items: list[tuple[Any, Any]] = list(payload.items())
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        items = []
        for index, entry in enumerate(payload):
            if not isinstance(entry, Mapping):
                raise ConfigError(
                    f"{context}[{index}]: expected a mapping with phrase and "
                    f"weight, got {entry!r}"
                )
            items.append((entry.get("phrase", ""), entry.get("weight", 1.0)))
    else:
        raise ConfigError(
            f"{context}: expected a {{phrase: weight}} mapping or a list of "
            f"them, got {payload!r}"
        )
    entries: list[tuple[str, float]] = []
    for phrase, weight in items:
        if phrase is None:
            raise ConfigError(
                f"{context}: an empty phrase must be written as \"\", because a "
                "bare key parses as null and would print as the word None"
            )
        if not isinstance(phrase, str):
            raise ConfigError(f"{context}: expected text, got {phrase!r}")
        try:
            entries.append((phrase, float(weight)))
        except (TypeError, ValueError) as error:
            raise ConfigError(
                f"{context}[{phrase!r}]: weight must be a number, got {weight!r}"
            ) from error
    if not entries:
        raise ConfigError(f"{context}: must list at least one entry")
    if sum(weight for _, weight in entries) <= 0:
        raise ConfigError(f"{context}: weights must add up to more than zero")
    return tuple(entries)


def _parse_phrases(
    payload: Any,
    context: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    """Parse an unweighted pool of phrases from YAML.

    Args:
        payload: A list of strings, or ``None`` for the default.
        context: Dotted config path, for error messages.
        default: Pool to use when the key is absent.

    Returns:
        The phrases in file order, which is the order ``rng.choice`` indexes.

    Raises:
        ConfigError: If the payload is not a list, is empty, or holds a
            non-string. A bare string is rejected rather than accepted as a
            one-entry pool, because ``rng.choice`` over it would pick single
            characters.
    """
    if payload is None:
        return default
    if isinstance(payload, str) or not isinstance(payload, Sequence):
        raise ConfigError(f"{context}: expected a list of phrases, got {payload!r}")
    phrases: list[str] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, str):
            raise ConfigError(f"{context}[{index}]: expected text, got {entry!r}")
        phrases.append(entry)
    if not phrases:
        raise ConfigError(f"{context}: must list at least one phrase")
    return tuple(phrases)


def _parse_variants(
    payload: Any,
    context: str,
    default: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Parse a ``{key: [wording, ...]}`` mapping of label variants.

    Stored as ordered pairs rather than a dict so the surrounding dataclass
    stays hashable like every other config section.

    Args:
        payload: A mapping of machine key to a list of Persian wordings, or
            ``None`` for the default.
        context: Dotted config path, for error messages.
        default: Mapping to use when the key is absent.

    Returns:
        Ordered ``(key, wordings)`` pairs.

    Raises:
        ConfigError: If the payload is not a mapping, or any entry's value is
            not a non-empty list of strings.
    """
    if payload is None:
        return default
    if not isinstance(payload, Mapping):
        raise ConfigError(
            f"{context}: expected a mapping of key to wordings, got {payload!r}"
        )
    entries: list[tuple[str, tuple[str, ...]]] = []
    for key, wordings in payload.items():
        entries.append((str(key), _parse_phrases(wordings, f"{context}.{key}", ())))
    if not entries:
        raise ConfigError(f"{context}: must list at least one key")
    return tuple(entries)


def _lookup(
    entries: tuple[tuple[str, tuple[str, ...]], ...],
    key: str,
) -> tuple[str, ...] | None:
    """Return the wordings registered for ``key``, or ``None``."""
    for name, wordings in entries:
        if name == key:
            return wordings
    return None


def _reject_unknown(payload: Mapping[str, Any], known: set[str], context: str) -> None:
    """Raise when ``payload`` carries a key outside ``known``."""
    unknown = set(payload) - known
    if unknown:
        raise ConfigError(
            f"{context}: unknown key(s) {sorted(unknown)}; "
            f"expected any of {sorted(known)}"
        )


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HeaderVocabulary:
    """Wording for the letterhead and the footer.

    Attributes:
        name_prefixes: Organisational prefixes for the brand name, weighted.
            The empty entry is the bare brand name.
        religious_openers: Openers printed above the body, weighted.
        slogans: Slogan line under the company name.
        contact_labels: Persian wordings per contact field key, in the order
            ``rng.choice`` indexes them. The first is the most common.
        page_templates: Page-number wordings; ``{page}`` and ``{total}`` are
            filled by ``paginate.js`` once the sheet count is known.
    """

    name_prefixes: tuple[tuple[str, float], ...] = _NAME_PREFIXES
    religious_openers: tuple[tuple[str, float], ...] = _RELIGIOUS_OPENERS
    slogans: tuple[str, ...] = _SLOGANS
    contact_labels: tuple[tuple[str, tuple[str, ...]], ...] = _CONTACT_LABELS
    page_templates: tuple[str, ...] = _PAGE_TEMPLATES

    def __post_init__(self) -> None:
        for template in self.page_templates:
            if "{page}" not in template:
                raise ConfigError(
                    "vocabulary.header.page_templates: every wording must carry "
                    f"a {{page}} placeholder, got {template!r}"
                )

    def labels_for(self, key: str, fallback: str) -> tuple[str, ...]:
        """Return the wordings for a contact field, or just ``fallback``.

        Args:
            key: Contact field key, e.g. ``phone``.
            fallback: Label to use when the field has no wordings configured.

        Returns:
            A non-empty tuple, so the caller's ``rng.choice`` always has a pool.
        """
        return _lookup(self.contact_labels, key) or (fallback,)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HeaderVocabulary":
        """Build the header wording from its YAML mapping."""
        known = set(cls.__dataclass_fields__)
        _reject_unknown(payload, known, "vocabulary.header")
        defaults = cls()
        return cls(
            name_prefixes=_parse_phrase_pool(
                payload.get("name_prefixes"),
                "vocabulary.header.name_prefixes",
                defaults.name_prefixes,
            ),
            religious_openers=_parse_phrase_pool(
                payload.get("religious_openers"),
                "vocabulary.header.religious_openers",
                defaults.religious_openers,
            ),
            slogans=_parse_phrases(
                payload.get("slogans"),
                "vocabulary.header.slogans",
                defaults.slogans,
            ),
            contact_labels=_parse_variants(
                payload.get("contact_labels"),
                "vocabulary.header.contact_labels",
                defaults.contact_labels,
            ),
            page_templates=_parse_phrases(
                payload.get("page_templates"),
                "vocabulary.header.page_templates",
                defaults.page_templates,
            ),
        )


@dataclass(frozen=True, slots=True)
class LetterVocabulary:
    """Wording for the letter's own lines.

    Every prefix is concatenated directly onto the name or subject that follows
    it, so a trailing space in the pool is deliberate and is preserved.

    Attributes:
        recipient_prefixes: How the recipient line is introduced, weighted.
        sender_prefixes: How the sender line is introduced, weighted.
        subject_prefixes: How the subject line is labelled, weighted.
        attachment_titles: Headings for the attachment list.
    """

    recipient_prefixes: tuple[tuple[str, float], ...] = _RECIPIENT_PREFIXES
    sender_prefixes: tuple[tuple[str, float], ...] = _SENDER_PREFIXES
    subject_prefixes: tuple[tuple[str, float], ...] = _SUBJECT_PREFIXES
    attachment_titles: tuple[str, ...] = _ATTACHMENT_TITLES

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LetterVocabulary":
        """Build the letter wording from its YAML mapping."""
        known = set(cls.__dataclass_fields__)
        _reject_unknown(payload, known, "vocabulary.letter")
        defaults = cls()
        return cls(
            **{
                name: _parse_phrase_pool(
                    payload.get(name),
                    f"vocabulary.letter.{name}",
                    getattr(defaults, name),
                )
                for name in ("recipient_prefixes", "sender_prefixes",
                             "subject_prefixes")
            },
            attachment_titles=_parse_phrases(
                payload.get("attachment_titles"),
                "vocabulary.letter.attachment_titles",
                defaults.attachment_titles,
            ),
        )


@dataclass(frozen=True, slots=True)
class MetadataVocabulary:
    """Wording for the administrative metadata block.

    Attributes:
        labels: Wordings per field key, most common first.
        label_weights: Weights applied to ``labels`` positionally and then
            truncated to the number of wordings a field actually has, so one
            list serves fields with three and four variants alike.
        separators: What is drawn between label and value, weighted. The empty
            entry is a real treatment - many letterheads align values in a
            column and use whitespace alone.
        attachment_present: Values for a letter that has attachments.
        attachment_absent: Values for one that has none. Real letters print
            this rather than omitting the row.
    """

    labels: tuple[tuple[str, tuple[str, ...]], ...] = _METADATA_LABELS
    label_weights: tuple[float, ...] = _METADATA_LABEL_WEIGHTS
    separators: tuple[tuple[str, float], ...] = _SEPARATORS
    attachment_present: tuple[str, ...] = _ATTACHMENT_PRESENT
    attachment_absent: tuple[str, ...] = _ATTACHMENT_ABSENT

    def __post_init__(self) -> None:
        if not self.label_weights:
            raise ConfigError("vocabulary.metadata_block.label_weights is empty")
        if any(weight < 0 for weight in self.label_weights):
            raise ConfigError(
                "vocabulary.metadata_block.label_weights must not be negative, "
                f"got {self.label_weights}"
            )
        widest = max((len(wordings) for _, wordings in self.labels), default=0)
        if widest > len(self.label_weights):
            raise ConfigError(
                "vocabulary.metadata_block.label_weights needs one weight per "
                f"wording of the widest field, which has {widest}; got "
                f"{len(self.label_weights)}"
            )

    def weighted_labels(self, key: str) -> tuple[tuple[str, float], ...]:
        """Return a field's wordings zipped with their weights.

        Args:
            key: Field key, e.g. ``letter_number``.

        Returns:
            ``(wording, weight)`` pairs ready for ``weighted_choice``, or an
            empty tuple when the key has no wordings - which the caller reads
            as "render the key itself".
        """
        wordings = _lookup(self.labels, key)
        if not wordings:
            return ()
        return tuple(zip(wordings, self.label_weights[: len(wordings)]))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetadataVocabulary":
        """Build the metadata wording from its YAML mapping."""
        known = set(cls.__dataclass_fields__)
        _reject_unknown(payload, known, "vocabulary.metadata_block")
        defaults = cls()
        weights_payload = payload.get("label_weights")
        if weights_payload is None:
            weights = defaults.label_weights
        elif isinstance(weights_payload, str) or not isinstance(
            weights_payload, Sequence
        ):
            raise ConfigError(
                "vocabulary.metadata_block.label_weights: expected a list of "
                f"numbers, got {weights_payload!r}"
            )
        else:
            try:
                weights = tuple(float(value) for value in weights_payload)
            except (TypeError, ValueError) as error:
                raise ConfigError(
                    "vocabulary.metadata_block.label_weights: every entry must "
                    f"be a number, got {weights_payload!r}"
                ) from error
        return cls(
            labels=_parse_variants(
                payload.get("labels"),
                "vocabulary.metadata_block.labels",
                defaults.labels,
            ),
            label_weights=weights,
            separators=_parse_phrase_pool(
                payload.get("separators"),
                "vocabulary.metadata_block.separators",
                defaults.separators,
            ),
            attachment_present=_parse_phrases(
                payload.get("attachment_present"),
                "vocabulary.metadata_block.attachment_present",
                defaults.attachment_present,
            ),
            attachment_absent=_parse_phrases(
                payload.get("attachment_absent"),
                "vocabulary.metadata_block.attachment_absent",
                defaults.attachment_absent,
            ),
        )


@dataclass(frozen=True, slots=True)
class StampVocabulary:
    """Wording printed around the rubber stamp.

    Attributes:
        cities: Cities for the lower ring, sampled when the letter's address
            does not name one.
        seal_words: Words printed beside the city, weighted. The empty entry
            prints the city alone.
    """

    cities: tuple[str, ...] = _CITIES
    seal_words: tuple[tuple[str, float], ...] = _SEAL_WORDS

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StampVocabulary":
        """Build the stamp wording from its YAML mapping."""
        known = set(cls.__dataclass_fields__)
        _reject_unknown(payload, known, "vocabulary.stamp")
        defaults = cls()
        return cls(
            cities=_parse_phrases(
                payload.get("cities"), "vocabulary.stamp.cities", defaults.cities
            ),
            seal_words=_parse_phrase_pool(
                payload.get("seal_words"),
                "vocabulary.stamp.seal_words",
                defaults.seal_words,
            ),
        )


@dataclass(frozen=True, slots=True)
class TableVocabulary:
    """Column headers and cell wordings for a synthesised table.

    Attributes:
        header_sets: Whole header rows, so a synthesised table reads as one
            coherent document rather than a mix of unrelated columns. The row's
            width is the table's column count.
        service_labels: Values for a شرح خدمات column, and the fallback for a
            column no other pool claims.
        stage_labels: Values for a مرحله column.
        document_labels: Values for a مدرک / نام column.
        unit_labels: Values for a واحد column.
        status_values: Values for a وضعیت column.
        detail_values: Values for a توضیحات / شرح وظیفه column.
        months: Jalali month names for a date column.
        captions: Captions drawn above the table.
        cost_caption: Caption preferred when the header row names amounts and
            the letter's subject names costs.
    """

    header_sets: tuple[tuple[str, ...], ...] = _TABLE_HEADER_SETS
    service_labels: tuple[str, ...] = _SERVICE_LABELS
    stage_labels: tuple[str, ...] = _STAGE_LABELS
    document_labels: tuple[str, ...] = _DOCUMENT_LABELS
    unit_labels: tuple[str, ...] = _UNIT_LABELS
    status_values: tuple[str, ...] = _STATUS_VALUES
    detail_values: tuple[str, ...] = _DETAIL_VALUES
    months: tuple[str, ...] = _MONTHS
    captions: tuple[str, ...] = _TABLE_CAPTIONS
    cost_caption: str = _TABLE_COST_CAPTION

    def __post_init__(self) -> None:
        if not self.header_sets:
            raise ConfigError("vocabulary.elements.table.header_sets is empty")
        for row in self.header_sets:
            if not row:
                raise ConfigError(
                    "vocabulary.elements.table.header_sets: a header row must "
                    "name at least one column"
                )
        if len(self.months) != 12:
            raise ConfigError(
                "vocabulary.elements.table.months must name all twelve Jalali "
                f"months, got {len(self.months)}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TableVocabulary":
        """Build the table wording from its YAML mapping."""
        known = set(cls.__dataclass_fields__)
        _reject_unknown(payload, known, "vocabulary.elements.table")
        defaults = cls()
        header_payload = payload.get("header_sets")
        if header_payload is None:
            header_sets = defaults.header_sets
        elif isinstance(header_payload, str) or not isinstance(
            header_payload, Sequence
        ):
            raise ConfigError(
                "vocabulary.elements.table.header_sets: expected a list of "
                f"header rows, got {header_payload!r}"
            )
        else:
            header_sets = tuple(
                _parse_phrases(
                    row, f"vocabulary.elements.table.header_sets[{index}]", ()
                )
                for index, row in enumerate(header_payload)
            )
        cost = payload.get("cost_caption", defaults.cost_caption)
        if not isinstance(cost, str):
            raise ConfigError(
                f"vocabulary.elements.table.cost_caption: expected text, got {cost!r}"
            )
        plain = (
            "service_labels",
            "stage_labels",
            "document_labels",
            "unit_labels",
            "status_values",
            "detail_values",
            "months",
            "captions",
        )
        return cls(
            header_sets=header_sets,
            cost_caption=cost,
            **{
                name: _parse_phrases(
                    payload.get(name),
                    f"vocabulary.elements.table.{name}",
                    getattr(defaults, name),
                )
                for name in plain
            },
        )


@dataclass(frozen=True, slots=True)
class ChartVocabulary:
    """Series categories and captions for a synthesised chart.

    Attributes:
        series_pools: ``(unit, labels)`` pairs. The unit's wording decides
            which value range the series draws from - ``درصد`` reads as
            percentages, ``تعداد`` as counts, anything else as amounts - so
            renaming a unit changes the numbers plotted under it.
        captions: Caption per chart kind. Every renderable kind needs an entry,
            because the renderer indexes this by kind; an intentionally
            captionless kind is written as an empty string.
    """

    series_pools: tuple[tuple[str, tuple[str, ...]], ...] = _SERIES_POOLS
    captions: tuple[tuple[str, str], ...] = _CHART_CAPTIONS

    def __post_init__(self) -> None:
        if not self.series_pools:
            raise ConfigError("vocabulary.elements.chart.series_pools is empty")
        for unit, labels in self.series_pools:
            if len(labels) < 2:
                raise ConfigError(
                    "vocabulary.elements.chart.series_pools: pool "
                    f"{unit!r} needs at least two categories to plot"
                )
        named = {kind for kind, _ in self.captions}
        missing = set(CHART_KINDS) - named
        if missing:
            raise ConfigError(
                "vocabulary.elements.chart.captions must name every renderable "
                f"kind; missing {sorted(missing)}. Write \"\" for a kind that "
                "should render without a caption."
            )
        unknown = named - set(CHART_KINDS)
        if unknown:
            raise ConfigError(
                "vocabulary.elements.chart.captions: unknown kind(s) "
                f"{sorted(unknown)}; expected any of {sorted(CHART_KINDS)}"
            )

    def caption_for(self, kind: str) -> str:
        """Return the caption for a chart kind.

        Args:
            kind: One of the renderable chart kinds.

        Returns:
            The configured caption. Validation guarantees every renderable kind
            has one, so the empty fallback is unreachable in practice.
        """
        for name, caption in self.captions:
            if name == kind:
                return caption
        return ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChartVocabulary":
        """Build the chart wording from its YAML mapping."""
        known = set(cls.__dataclass_fields__)
        _reject_unknown(payload, known, "vocabulary.elements.chart")
        defaults = cls()
        pools_payload = payload.get("series_pools")
        if pools_payload is None:
            pools = defaults.series_pools
        elif isinstance(pools_payload, str) or not isinstance(pools_payload, Sequence):
            raise ConfigError(
                "vocabulary.elements.chart.series_pools: expected a list of "
                f"{{unit, labels}} mappings, got {pools_payload!r}"
            )
        else:
            entries: list[tuple[str, tuple[str, ...]]] = []
            for index, entry in enumerate(pools_payload):
                context = f"vocabulary.elements.chart.series_pools[{index}]"
                if not isinstance(entry, Mapping):
                    raise ConfigError(
                        f"{context}: expected a mapping with unit and labels, "
                        f"got {entry!r}"
                    )
                _reject_unknown(entry, {"unit", "labels"}, context)
                unit = entry.get("unit")
                if not isinstance(unit, str) or not unit.strip():
                    raise ConfigError(f"{context}.unit: expected text, got {unit!r}")
                entries.append(
                    (unit, _parse_phrases(entry.get("labels"), f"{context}.labels", ()))
                )
            pools = tuple(entries)
        captions_payload = payload.get("captions")
        if captions_payload is None:
            captions = defaults.captions
        elif not isinstance(captions_payload, Mapping):
            raise ConfigError(
                "vocabulary.elements.chart.captions: expected a mapping of kind "
                f"to caption, got {captions_payload!r}"
            )
        else:
            captions = tuple(
                (str(kind), str(caption)) for kind, caption in captions_payload.items()
            )
        return cls(series_pools=pools, captions=captions)


@dataclass(frozen=True, slots=True)
class ElementsVocabulary:
    """Wording for the optional structured elements.

    Attributes:
        table: Table headers and cell wordings.
        chart: Chart series categories and captions.
    """

    table: TableVocabulary = field(default_factory=TableVocabulary)
    chart: ChartVocabulary = field(default_factory=ChartVocabulary)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ElementsVocabulary":
        """Build the element wording from its YAML mapping."""
        _reject_unknown(payload, {"table", "chart"}, "vocabulary.elements")
        return cls(
            table=TableVocabulary.from_dict(payload.get("table", {}) or {}),
            chart=ChartVocabulary.from_dict(payload.get("chart", {}) or {}),
        )


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Every Persian content pool the renderer draws from.

    Attributes:
        header: Letterhead and footer wording.
        letter: The letter's own recipient, sender and subject lines.
        metadata_block: The administrative metadata block.
        stamp: Wording around the rubber stamp.
        elements: Tables and charts.
        source_path: File the pools came from, or ``None`` for the shipped
            defaults. Recorded so a manifest can name what a run was rendered
            from.
    """

    header: HeaderVocabulary = field(default_factory=HeaderVocabulary)
    letter: LetterVocabulary = field(default_factory=LetterVocabulary)
    metadata_block: MetadataVocabulary = field(default_factory=MetadataVocabulary)
    stamp: StampVocabulary = field(default_factory=StampVocabulary)
    elements: ElementsVocabulary = field(default_factory=ElementsVocabulary)
    source_path: Path | None = None

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, source_path: Path | None = None
    ) -> "Vocabulary":
        """Build the whole vocabulary from an already-parsed YAML mapping.

        Args:
            payload: Parsed YAML document.
            source_path: File the mapping came from, kept for provenance.

        Returns:
            The validated vocabulary.

        Raises:
            ConfigError: If a section or key is unknown, or a pool is malformed.
        """
        known = {"header", "letter", "metadata_block", "stamp", "elements"}
        _reject_unknown(payload, known, "vocabulary")
        return cls(
            header=HeaderVocabulary.from_dict(payload.get("header", {}) or {}),
            letter=LetterVocabulary.from_dict(payload.get("letter", {}) or {}),
            metadata_block=MetadataVocabulary.from_dict(
                payload.get("metadata_block", {}) or {}
            ),
            stamp=StampVocabulary.from_dict(payload.get("stamp", {}) or {}),
            elements=ElementsVocabulary.from_dict(payload.get("elements", {}) or {}),
            source_path=source_path,
        )


def load_vocabulary(path: str | Path | None = None) -> Vocabulary:
    """Load and validate the Persian content pools.

    Args:
        path: YAML file to read, absolute or relative to the repository root.
            ``None`` falls back to :data:`DEFAULT_VOCABULARY_PATH`.

    Returns:
        The validated vocabulary.

    Raises:
        ConfigError: If the file is missing, is not a YAML mapping, or fails
            validation. A missing file is an error rather than a silent fall
            back to the shipped defaults: a run that quietly ignored a
            misspelled path would produce a corpus nobody could explain.
    """
    target = resolve_repo_path(path if path is not None else DEFAULT_VOCABULARY_PATH)
    if not target.is_file():
        raise ConfigError(f"vocabulary file not found: {target}")
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{target}: invalid YAML - {exc}") from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{target}: expected a YAML mapping at the top level")
    return Vocabulary.from_dict(payload, source_path=target)
