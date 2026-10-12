"""Prompt construction for stage 02.

The prompt is written **in Persian**. That is a measured choice, not a
stylistic one: an instruction-tuned model asked in English to write Persian
produces text with English discourse structure - translated-sounding
connectives, English paragraph rhythm, and a tendency to gloss unusual words in
parentheses. Asked in Persian it writes Persian. The genre, register and
persona names in :mod:`fa_ocr_lora.diverse.planning.axes` are stored in Persian
for the same reason: they go into the prompt verbatim.

Four instruction groups matter more than the rest, each fixing a failure the
previous dataset actually produced:

* **Bare output.** The model must emit only the text. A ``در اینجا متن
  درخواستی شما است:`` preamble, a Markdown fence or a trailing note is not just
  untidy - it becomes part of ``text_render`` and therefore part of the label,
  teaching the model to hallucinate English preambles onto Persian pages.
* **Words used, not explained.** Given a rare word, a helpful model defines it.
  A definition is a different genre from the one requested, and it collapses
  the register variety the plan worked to create.
* **Invented specifics only.** Real phone numbers, national IDs and addresses
  are privacy hazards that no amount of downstream filtering reliably removes.
  Inventing them is also better data, because a plausible fake number has the
  same digit statistics as a real one.
* **Original verse.** For poetry the model must compose, never reproduce. A
  reproduced Hafez ghazal is a copyright problem and a duplicate-detection
  problem at once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Final

from ...dataset.common.io_utils import to_persian_digits
from ..planning.sampler import Plan

__all__ = ["SYSTEM_PROMPT", "build_prompt", "prompt_hash"]

#: The system prompt. Short on purpose: long system prompts get compressed into
#: a persona the model then applies uniformly, which is precisely the
#: homogenisation this dataset exists to avoid. The variety belongs in the user
#: prompt, which differs per sample.
SYSTEM_PROMPT: Final[str] = (
    "تو یک نویسنده حرفه‌ای فارسی هستی که در سبک‌ها و قالب‌های گوناگون می‌نویسد. "
    "فقط متن نهایی را می‌نویسی: بدون مقدمه، بدون توضیح، بدون علامت‌گذاری مارک‌داون، "
    "بدون ترجمه و بدون هیچ یادداشتی. پاسخ تو همان متنی است که روی کاغذ چاپ خواهد شد."
)

#: Persian descriptions of each content feature, keyed by the feature name used
#: in the plan. Phrased as requests rather than schemas, because a model asked
#: for "a bulleted list" writes one and a model handed a JSON schema writes
#: JSON.
_FEATURE_TEXT: Final[Mapping[str, str]] = {
    "numbers": (
        "چند عدد واقعی‌نما در متن بیاور: تاریخ، مبلغ، درصد، اندازه یا شماره. "
        "همه اعداد باید ساختگی باشند"
    ),
    "latin_terms": (
        "چند اصطلاح انگلیسی یا لاتین را در جای طبیعی خود در متن فارسی به کار ببر"
    ),
    "quotations": "یک نقل قول کوتاه از یک گوینده ساختگی را داخل گیومه «» بیاور",
    "parentheticals": "یک یا دو توضیح کوتاه را داخل پرانتز در متن جای بده",
    "bullet_list": "بخشی از متن را به صورت فهرست نقطه‌ای بنویس",
    "table": (
        "یک جدول ساده با سرستون‌های فارسی در متن بگنجان و سطرهای آن را با "
        "علامت | از هم جدا کن"
    ),
    "headings": "متن را با یک یا چند تیتر کوتاه بخش‌بندی کن",
    "footnotes": "یک پانویس کوتاه در پایان متن بیاور و در متن به آن اشاره کن",
    "diacritics": (
        "روی واژه‌های دشوار و کم‌کاربرد اعراب‌گذاری (فتحه، کسره، ضمه، تشدید) انجام بده"
    ),
    "zwnj_heavy": (
        "در واژه‌های مرکب و پیشوند و پسوندها از نیم‌فاصله درست استفاده کن "
        "(مانند می‌رود، کتاب‌ها، بی‌نهایت)"
    ),
    "emoji": "چند ایموجی در جاهای طبیعی گفتگو به کار ببر",
}

#: Persian names of the digit scripts, so the instruction names what the writer
#: should type rather than a Unicode block.
_DIGIT_TEXT: Final[Mapping[str, str]] = {
    "persian": "اعداد را با رقم‌های فارسی بنویس (۰۱۲۳۴۵۶۷۸۹)",
    "latin": "اعداد را با رقم‌های لاتین بنویس (0123456789)",
    "arabic_indic": "اعداد را با رقم‌های عربی بنویس (٠١٢٣٤٥٦٧٨٩)",
    "mixed": "در نوشتن اعداد هم از رقم فارسی و هم از رقم لاتین استفاده کن",
}

#: Extra instruction per visual category, describing the artefact the text will
#: become. A model told the text is going on a shop sign writes a shop sign;
#: told nothing, it writes a paragraph and the banner layout truncates it.
_CATEGORY_TEXT: Final[Mapping[str, str]] = {
    "page": "این متن روی یک صفحه کامل کاغذ چاپ می‌شود",
    "paragraph": "این متن یک بند مستقل است که در یک کادر چاپ می‌شود",
    "line": "این متن فقط یک سطر کوتاه است؛ بیش از یک سطر ننویس",
    "poetry": (
        "این متن شعر است و در قالب مصرع‌های جداگانه چاپ می‌شود. "
        "هر مصرع را در یک سطر بنویس. شعر باید کاملاً سروده خودت باشد"
    ),
    "web": "این متن در یک صفحه وب نمایش داده می‌شود",
    "chat": (
        "این متن یک گفتگوی پیام‌رسان است. هر پیام را در یک سطر و با قالب "
        "«نام: متن پیام» بنویس"
    ),
    "banner": (
        "این متن روی یک بنر یا پوستر چاپ می‌شود، پس باید بسیار کوتاه و کوبنده باشد"
    ),
    "story": "این متن بخشی از یک کتاب داستان است",
    "table_form": (
        "این متن یک جدول یا فرم است. سطرها را با علامت | ستون‌بندی کن و "
        "سرستون‌ها را در سطر اول بنویس"
    ),
    "mixed": "این متن فارسی است اما اصطلاحات انگلیسی و لاتین در آن به کار می‌رود",
}

#: The standing rules, appended to every prompt. Numbered because a numbered
#: list is followed more reliably than a paragraph of prose.
#:
#: Rule 9 is the expensive lesson of this stage's model evaluation. Asked for
#: Persian containing rare words, every tested model leaked other scripts into
#: the output - ``систеمتان sourced knowledge`` from one, ``θερمولویجیکی`` and
#: ``circuit`` from another. Adding the rule took ``nemotron-3-super`` from
#: 85.4% Persian letters to 98.6%. It is suppressed for the ``latin_terms``
#: and ``mixed`` samples, which *want* Latin script and say so in metadata.
_RULES: Final[tuple[str, ...]] = (
    "فقط و فقط متن نهایی را بنویس. هیچ مقدمه، عنوان توضیحی، توضیح پایانی یا "
    "یادداشتی ننویس.",
    "از علامت‌گذاری مارک‌داون مانند ** یا ## یا ``` استفاده نکن.",
    "کل متن را داخل گیومه یا نقل قول قرار نده.",
    "واژه‌های خواسته‌شده را در جمله و در جای دستوری درست به کار ببر؛ آن‌ها را "
    "معنی نکن، تعریف نکن و در پرانتز توضیح نده. املای آن‌ها را عیناً و بدون "
    "هیچ تغییری بنویس.",
    "متن باید کاملاً نوشته خودت باشد و هیچ بخشی از آن از شعر، ترانه یا نوشته "
    "موجود کسی دیگر نقل نشود.",
    "از نام، شماره تلفن، کد ملی، نشانی یا مشخصات هیچ شخص واقعی استفاده نکن؛ "
    "همه این‌ها را ساختگی و باورپذیر بساز.",
    "محتوای متن باید سالم باشد: بدون محتوای جنسی، خشونت تصویری، توهین، "
    "نفرت‌پراکنی و بدون بدگویی از اشخاص واقعی.",
    "اگر متن به آیه، حدیث یا متن مقدس نیاز داشت، آن را عیناً نقل نکن؛ "
    "به صورت نقل به مضمون بنویس.",
)

#: Appended as the final rule unless the plan asked for Latin script. Kept
#: separate from :data:`_RULES` because it is the one rule that is sometimes
#: wrong to apply.
_PERSIAN_ONLY_RULE: Final[str] = (
    "هیچ واژه انگلیسی، لاتین، روسی، یونانی یا غیرفارسی در متن ننویس. "
    "تمام متن باید با الفبای فارسی نوشته شود."
)

#: Features whose presence means Latin script is wanted on purpose.
_ALLOWS_LATIN: Final[frozenset[str]] = frozenset({"latin_terms"})


def _length_sentence(plan: Plan) -> str:
    """Build the length instruction.

    Numbers are written in Persian digits. The instruction is Persian prose, so
    Latin digits inside it are a register mismatch the model notices and
    sometimes imitates - emitting ``40`` in a sample whose ``digit_script`` is
    ``persian`` and whose label therefore claims ``۴۰``.

    Args:
        plan: The sample's plan.

    Returns:
        A Persian sentence naming the target length. Very short targets get an
        explicit ceiling because a word count alone is routinely overshot by an
        order of magnitude on one-line requests.
    """
    low = to_persian_digits(plan.length_min)
    high = to_persian_digits(plan.length_max)
    if plan.length_max <= 6:
        return f"طول متن: حداکثر {high} واژه. بسیار کوتاه بنویس."
    if plan.length_max <= 18:
        return f"طول متن: بین {low} تا {high} واژه، در یک سطر."
    return f"طول متن: حدود {low} تا {high} واژه. از این محدوده بیرون نرو."


def build_prompt(plan: Plan) -> str:
    """Build the user prompt for one plan.

    Args:
        plan: The sample's plan.

    Returns:
        The complete Persian prompt.
    """
    lines: list[str] = []
    lines.append(f"یک متن فارسی در قالب «{plan.genre}» بنویس.")
    lines.append("")
    lines.append(f"لحن و سطح زبان: {plan.register}.")
    lines.append(f"نویسنده متن: {plan.persona}. از زبان او بنویس.")
    lines.append(f"ساختار متن: {plan.output_form}.")
    lines.append(_length_sentence(plan))

    category_note = _CATEGORY_TEXT.get(plan.category)
    if category_note:
        lines.append(f"{category_note}.")

    digit_note = _DIGIT_TEXT.get(plan.digit_script)
    if digit_note:
        lines.append(f"{digit_note}.")

    if plan.features:
        lines.append("")
        lines.append("ویژگی‌های محتوایی که باید در متن باشد:")
        for feature in plan.features:
            text = _FEATURE_TEXT.get(feature)
            if text:
                lines.append(f"- {text}.")

    if plan.words:
        lines.append("")
        lines.append(
            "این واژه‌ها را باید در متن به کار ببری. هر کدام را در جمله‌ای "
            "طبیعی و در معنای درست خودش بیاور:"
        )
        lines.append("، ".join(plan.words))
        if plan.off_domain:
            lines.append(
                "بعضی از این واژه‌ها به موضوع متن نزدیک نیستند؛ با خلاقیت "
                "جایی طبیعی برای آن‌ها در متن پیدا کن."
            )

    lines.append("")
    lines.append("قواعدی که باید رعایت کنی:")
    rules = list(_RULES)
    # The Persian-only rule is skipped exactly where Latin script is the point.
    if not (_ALLOWS_LATIN & set(plan.features)) and plan.category != "mixed":
        rules.append(_PERSIAN_ONLY_RULE)
    for index, rule in enumerate(rules, start=1):
        lines.append(f"{index}. {rule}")

    lines.append("")
    lines.append("حالا فقط متن را بنویس:")
    return "\n".join(lines)


def prompt_hash(prompt: str, profile_name: str, seed: int) -> str:
    """Return the cache key for one request.

    Keyed on the prompt, the profile and the seed together, so changing a
    sampling temperature or the run seed produces a cache miss rather than
    silently reusing text generated under different settings.

    Args:
        prompt: The user prompt.
        profile_name: Generation profile name.
        seed: The run seed.

    Returns:
        A 16-character hex digest.
    """
    digest = hashlib.blake2b(
        f"{profile_name}\x00{seed}\x00{prompt}".encode(), digest_size=8
    )
    return digest.hexdigest()


def describe_axes(plans: Sequence[Plan]) -> dict[str, int]:
    """Count distinct values per axis, for the pre-flight report.

    Args:
        plans: Plans about to be generated.

    Returns:
        Axis name to distinct-value count.
    """
    return {
        "category": len({plan.category for plan in plans}),
        "genre": len({plan.genre for plan in plans}),
        "register": len({plan.register for plan in plans}),
        "persona": len({plan.persona for plan in plans}),
        "length_target": len({plan.length_target for plan in plans}),
        "output_form": len({plan.output_form for plan in plans}),
        "profile": len({plan.profile for plan in plans}),
    }
