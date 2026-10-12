"""The plan axes: every independent dimension of variation.

This module is the dataset's variety, written down. ``.claude/dataset.md`` is
blunt about why it exists: the previous corpus was *one* generator writing
*one* genre into *one* template, so the model learned that generator rather
than Persian. The fix is a plan drawn from independent axes whose cross product
is far larger than the corpus, so no two samples share a configuration by
accident.

Each axis is a weighted pool. Weights are relative, not percentages, so one
entry can be retuned without renormalising the rest, and they are drawn through
:func:`~fa_ocr_lora.dataset.common.io_utils.weighted_choice` - one
``rng.random()`` per decision - which is what lets a pool be reordered without
shifting every later draw in the sample.

Genres are bound to categories rather than drawn freely. A legal clause does
not belong in a chat screenshot and a messenger exchange does not belong on an
A4 certificate; pairing them at random would produce text whose register fights
its layout, and a model learns nothing useful from an incoherent sample.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

__all__ = [
    "GENRES_BY_CATEGORY",
    "LENGTH_BY_CATEGORY",
    "LENGTH_TARGETS",
    "OUTPUT_FORMS",
    "PERSONAS",
    "REGISTERS",
    "genres_for",
    "length_words",
    "lengths_for",
]

#: Writing registers. The pool is weighted toward formal and journalistic
#: Persian because that is what most printed material is, but colloquial and
#: slangy carry real weight because chat and social copy are a tenth of the
#: corpus and their orthography differs sharply - verb contractions, dropped
#: ZWNJ, spoken forms such as ``می‌خوام`` for ``می‌خواهم``.
REGISTERS: Final[tuple[tuple[str, float], ...]] = (
    ("رسمی", 0.20),  # formal
    ("ادبی", 0.11),  # literary
    ("خبری", 0.14),  # journalistic
    ("محاوره‌ای تهرانی", 0.13),  # colloquial spoken Tehrani
    ("عامیانه و خودمانی", 0.07),  # slangy, intimate
    ("کهن و باستانی", 0.05),  # archaic
    ("اداری و بروکراتیک", 0.12),  # bureaucratic
    ("کودکانه", 0.05),  # childlike
    ("دانشگاهی", 0.09),  # academic
    ("تبلیغاتی", 0.04),  # promotional
)

#: Invented speakers. A persona changes word choice, sentence length and the
#: amount of hedging far more cheaply than any prompt instruction, and it costs
#: nothing but a line in the prompt. None names a real person.
PERSONAS: Final[tuple[tuple[str, float], ...]] = (
    ("یک آموزگار بازنشسته", 1.0),
    ("یک پرستار بخش اورژانس", 1.0),
    ("یک مغازه‌دار بازار", 1.0),
    ("یک نوجوان گیمر", 1.0),
    ("یک ریش‌سفید روستا", 0.9),
    ("یک وکیل دادگستری", 1.0),
    ("یک مهندس عمران", 1.0),
    ("یک شاعر معاصر", 0.9),
    ("یک روزنامه‌نگار اقتصادی", 1.0),
    ("یک دانشجوی کارشناسی ارشد", 1.0),
    ("یک راننده تاکسی اینترنتی", 0.9),
    ("یک پژوهشگر تاریخ", 0.9),
    ("یک کارمند بانک", 1.0),
    ("یک کشاورز", 0.8),
    ("یک مادر دو فرزند", 1.0),
    ("یک برنامه‌نویس", 1.0),
    ("یک داروساز", 0.8),
    ("یک مربی ورزش", 0.8),
    ("یک کتاب‌دار", 0.7),
    ("یک خبرنگار ورزشی", 0.8),
)

#: Length targets as ``(name, min_words, max_words, weight)``.
#:
#: The weights here are the *corpus-wide* shape, used only for reporting. The
#: actual draw goes through :data:`LENGTH_BY_CATEGORY`, because length and
#: category are not independent - see the comment there.
#:
#: The distribution is deliberately broad and deliberately heavy at the short
#: end. The previous corpus was all full pages, so the model never saw a label,
#: a sign or a single UI string - and those are a large share of real Persian
#: OCR work.
LENGTH_TARGETS: Final[tuple[tuple[str, int, int, float], ...]] = (
    ("very_short", 1, 6, 0.14),  # a label, a sign, a button
    ("single_line", 6, 18, 0.18),  # a title, a headline, a caption
    ("short_paragraph", 40, 80, 0.21),
    ("long_paragraph", 120, 200, 0.19),
    ("full_page", 300, 600, 0.19),
    ("multi_page", 700, 1200, 0.09),
)

#: Length targets permitted per category, with weights.
#:
#: Drawing length independently of category was a real bug, caught in the first
#: live generation run: a ``line`` sample asked for 300-600 words and a ``page``
#: sample asked for 6-18. Both are incoherent, and the model obeys the word
#: count rather than the category, so the resulting text cannot be laid out as
#: the category it was planned for - a 238-word "single line", or a blank A4
#: page holding six words.
#:
#: A category's length range is part of what the category *means*, so the two
#: axes are correlated on purpose. Within the permitted set the draw is still
#: free, which is where the variety lives.
LENGTH_BY_CATEGORY: Final[Mapping[str, tuple[tuple[str, float], ...]]] = {
    "page": (
        ("short_paragraph", 0.08),
        ("long_paragraph", 0.24),
        ("full_page", 0.50),
        ("multi_page", 0.18),
    ),
    "paragraph": (
        ("single_line", 0.08),
        ("short_paragraph", 0.47),
        ("long_paragraph", 0.45),
    ),
    "line": (
        ("very_short", 0.45),
        ("single_line", 0.55),
    ),
    "poetry": (
        ("single_line", 0.08),
        ("short_paragraph", 0.42),
        ("long_paragraph", 0.35),
        ("full_page", 0.15),
    ),
    "web": (
        ("short_paragraph", 0.20),
        ("long_paragraph", 0.36),
        ("full_page", 0.38),
        ("multi_page", 0.06),
    ),
    "chat": (
        ("short_paragraph", 0.38),
        ("long_paragraph", 0.42),
        ("full_page", 0.20),
    ),
    "banner": (
        ("very_short", 0.58),
        ("single_line", 0.42),
    ),
    "story": (
        ("long_paragraph", 0.20),
        ("full_page", 0.45),
        ("multi_page", 0.35),
    ),
    "table_form": (
        ("short_paragraph", 0.26),
        ("long_paragraph", 0.40),
        ("full_page", 0.34),
    ),
    "mixed": (
        ("single_line", 0.12),
        ("short_paragraph", 0.36),
        ("long_paragraph", 0.34),
        ("full_page", 0.18),
    ),
}

#: How the text is laid out as text, before any visual layout is chosen.
OUTPUT_FORMS: Final[tuple[tuple[str, float], ...]] = (
    ("پاراگراف پیوسته", 0.40),  # continuous prose
    ("سطرهای شکسته شعر", 0.09),  # line-broken verse
    ("فهرست شماره‌دار", 0.11),  # numbered list
    ("فهرست نقطه‌ای", 0.09),  # bulleted list
    ("گفت‌وگوی نوبتی", 0.11),  # dialogue turns
    ("سطرهای جدول", 0.08),  # table rows
    ("جفت‌های کلید و مقدار", 0.08),  # key-value fields
    ("تک‌سطری", 0.04),  # a single line
)

#: Genres, bound to the visual category that can plausibly carry them.
#:
#: Every entry is ``(persian_genre_name, weight)``. The Persian name goes
#: straight into the prompt, because a prompt written in Persian produces
#: markedly better Persian than one written in English asking for Persian.
GENRES_BY_CATEGORY: Final[Mapping[str, tuple[tuple[str, float], ...]]] = {
    "page": (
        ("گزارش خبری", 1.0),
        ("یادداشت تحلیلی روزنامه", 0.9),
        ("چکیده مقاله علمی", 0.9),
        ("راهنمای فنی دستگاه", 0.9),
        ("ماده قراردادی حقوقی", 0.9),
        ("بروشور دارویی", 0.8),
        ("بخشی از کتاب درسی", 1.0),
        ("مدخل دانشنامه", 0.9),
        ("نامه رسمی اداری", 0.8),
        ("مقاله فلسفی", 0.7),
        ("گزارش سالانه شرکت", 0.8),
        ("دستورالعمل ایمنی", 0.8),
        ("متن سخنرانی", 0.7),
        ("پرونده پژوهشی تاریخی", 0.7),
        ("آیین‌نامه و شرح وظایف", 0.8),
    ),
    "paragraph": (
        ("پاراگراف توصیفی", 1.0),
        ("بند استدلالی", 0.9),
        ("یادداشت روزنوشت", 0.9),
        ("سفرنامه کوتاه", 0.9),
        ("نقد کتاب", 0.8),
        ("توضیح علمی عامه‌فهم", 1.0),
        ("خاطره شخصی", 0.9),
        ("شرح یک اصطلاح تخصصی", 0.9),
        ("مقدمه کتاب", 0.7),
        ("زیرنویس تصویر", 0.6),
    ),
    "line": (
        ("تیتر خبری", 1.0),
        ("عنوان کتاب", 0.9),
        ("برچسب و راهنمای کوتاه", 1.0),
        ("نام بخش و سرفصل", 0.9),
        ("رشته متنی رابط کاربری", 1.0),
        ("شعار کوتاه", 0.8),
        ("ضرب‌المثل", 0.9),
        ("زیرنویس فیلم", 0.9),
        ("عنوان نمودار", 0.7),
        ("نام محصول و مدل", 0.8),
    ),
    "poetry": (
        ("غزل به سبک کلاسیک", 1.0),
        ("رباعی", 0.9),
        ("مثنوی", 0.9),
        ("قصیده", 0.6),
        ("دوبیتی", 0.8),
        ("شعر سپید", 1.0),
        ("شعر نیمایی", 0.9),
        ("ترانه اصیل", 0.7),
        ("شعر کودکانه", 0.8),
    ),
    "web": (
        ("مقاله وب‌سایت", 1.0),
        ("پست وبلاگ", 1.0),
        ("رشته انجمن گفتگو", 0.9),
        ("صفحه معرفی محصول", 1.0),
        ("خبر سایت خبری", 1.0),
        ("پرسش و پاسخ متداول", 0.9),
        ("بخش نظرات کاربران", 0.9),
        ("توضیحات داشبورد", 0.7),
        ("نتایج جست‌وجو", 0.6),
        ("صفحه درباره ما", 0.7),
    ),
    "chat": (
        ("گفتگوی دونفره پیام‌رسان", 1.0),
        ("گفتگوی گروهی", 0.9),
        ("پیامک کوتاه", 0.9),
        ("پیام پشتیبانی مشتری", 0.9),
        ("هماهنگی کاری در گروه", 1.0),
        ("گفتگوی خانوادگی", 1.0),
        ("پیام خرید و فروش", 0.8),
    ),
    "banner": (
        ("شعار تبلیغاتی", 1.0),
        ("پوستر رویداد", 1.0),
        ("تابلو مغازه", 0.9),
        ("کارت شبکه اجتماعی", 0.9),
        ("تراکت تبلیغاتی", 0.9),
        ("دعوت‌نامه مراسم", 0.8),
        ("بیلبورد شهری", 0.8),
        ("اعلامیه اطلاع‌رسانی", 0.9),
    ),
    "story": (
        ("داستان کوتاه", 1.0),
        ("داستانک", 0.9),
        ("افسانه عامیانه", 0.9),
        ("قصه کودکانه", 1.0),
        ("صحنه گفت‌وگوی داستانی", 0.9),
        ("فصلی از رمان", 0.8),
        ("حکایت اخلاقی", 0.8),
    ),
    "table_form": (
        ("جدول آماری", 1.0),
        ("فرم درخواست", 1.0),
        ("صورت‌حساب و فاکتور", 1.0),
        ("رسید فروش", 0.9),
        ("جدول زمانی و برنامه", 0.9),
        ("فهرست قیمت", 0.9),
        ("دفتر ثبت و لجر", 0.7),
        ("منوی رستوران", 0.9),
        ("کارنامه و ریز نمرات", 0.8),
    ),
    "mixed": (
        ("متن فنی با اصطلاحات انگلیسی", 1.0),
        ("رابط کاربری دوزبانه", 0.9),
        ("برچسب محصول وارداتی", 0.9),
        ("مقاله فناوری با نام‌های لاتین", 1.0),
        ("دستور نصب نرم‌افزار", 0.9),
        ("رزومه دوزبانه", 0.8),
    ),
}


def genres_for(category: str) -> tuple[tuple[str, float], ...]:
    """Return the genre pool bound to one visual category.

    Args:
        category: A value from :data:`fa_ocr_lora.diverse.config.CATEGORIES`.

    Returns:
        The ``(genre, weight)`` pool.

    Raises:
        KeyError: If the category has no genre pool. Raised rather than
            defaulted, because a category with no genres would silently plan
            samples with an empty prompt.
    """
    try:
        return GENRES_BY_CATEGORY[category]
    except KeyError as error:
        raise KeyError(
            f"no genre pool for category {category!r}; "
            f"known categories are {sorted(GENRES_BY_CATEGORY)}"
        ) from error


def length_words(name: str) -> tuple[int, int]:
    """Return the word-count bounds of a named length target.

    Args:
        name: A length target name from :data:`LENGTH_TARGETS`.

    Returns:
        ``(minimum, maximum)`` words.

    Raises:
        KeyError: If the name is unknown.
    """
    for candidate, low, high, _ in LENGTH_TARGETS:
        if candidate == name:
            return low, high
    raise KeyError(f"unknown length target {name!r}")


def lengths_for(category: str) -> tuple[tuple[str, float], ...]:
    """Return the length-target pool permitted for one visual category.

    Args:
        category: A value from :data:`fa_ocr_lora.diverse.config.CATEGORIES`.

    Returns:
        The ``(length_target, weight)`` pool.

    Raises:
        KeyError: If the category has no length pool, which would otherwise
            mean planning a sample whose length contradicts its layout.
    """
    try:
        return LENGTH_BY_CATEGORY[category]
    except KeyError as error:
        raise KeyError(
            f"no length pool for category {category!r}; "
            f"known categories are {sorted(LENGTH_BY_CATEGORY)}"
        ) from error


def pool_names(pool: Sequence[tuple[str, float]]) -> tuple[str, ...]:
    """Return just the values of a weighted pool, for reporting.

    Args:
        pool: A ``(value, weight)`` pool.

    Returns:
        The values in order.
    """
    return tuple(value for value, _ in pool)
