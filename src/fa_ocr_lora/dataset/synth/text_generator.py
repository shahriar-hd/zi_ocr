"""Generate Persian official-letter text with a hosted OpenAI-compatible API.

The synthetic page pipeline needs a large pool of *textual* letter content
before anything is rendered. This module produces exactly that: one JSON file
per letter under ``data/text/plaintext/plaintext_000000.json``, so a 20,000
sample run can be interrupted -- by a rate limit, a network drop, or Ctrl-C --
and resumed by re-running the same command. A sample whose file already exists
is skipped, so resuming never re-pays for work already on disk.

Python supplies only the sender identity, taken from the logo manifest so the
letter can later be composited with the matching artwork, plus a genre and tone
nudge. The model invents everything else: the recipient organisation, the
subject, the legal body, and every administrative identifier. That is what
keeps 20,000 samples from reading like one letter with the numbers swapped.

Roughly one letter in ten also carries a Markdown table or chart data inside
its ``attachments`` field, for the element generators under ``elements/`` to
render later.

The module is organised in layers, each independently testable: company loading
(:func:`load_companies`), prompt building (:func:`build_prompt`), transport
(:class:`KeyWorker`), validation (:func:`validate_letter`), storage
(:func:`write_record`) and the concurrent runner (:func:`generate_letters`).

Several API keys can be used at once: one worker thread is started per key, so
N keys generate N letters concurrently and a key that hits its rate limit waits
on its own without stalling the others. Supply them in the environment or a
``.env`` file as a comma-separated ``NVIDIA_API_KEY``, as numbered
``NVIDIA_API_KEY_1`` .. ``NVIDIA_API_KEY_32`` variables, or via
``--api-keys-file``.

Examples:
    Smoke test with four letters::

        python -m fa_ocr_lora.dataset.synth.text_generator --count 4

    Full run, resuming whatever is already on disk::

        python -m fa_ocr_lora.dataset.synth.text_generator --count 20000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

from openai import APIError, OpenAI

LOGGER: Final[logging.Logger] = logging.getLogger("text_generator")

DEFAULT_BASE_URL: Final[str] = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL: Final[str] = "nvidia/nemotron-3-nano-30b-a3b"
# DEFAULT_MODEL: Final[str] = "google/gemma-4-31b-it"
# DEFAULT_MODEL: Final[str] = "meta/llama-3.3-70b-instruct"
# DEFAULT_MODEL: Final[str] = "z-ai/glm-5.2"
DEFAULT_API_KEY_ENV: Final[str] = "NVIDIA_API_KEY"

DEFAULT_LOGOS_CSV: Final[str] = "data/thesvg/logos.csv"
DEFAULT_OUTPUT_DIR: Final[str] = "data/text/plaintext"
DEFAULT_COUNT: Final[int] = 20_000

#: Root of the UTSig signature corpus, scanned recursively for ``.tif`` files.
DEFAULT_SIGNATURES_DIR: Final[str] = "data/UTSig"
#: Share of letters that carry a signature. Not every administrative letter is
#: signed in practice (CLAUDE.md 5.2 calls for random placement), so the
#: remainder deliberately gets an empty ``signature_path``.
SIGNATURE_PROBABILITY: Final[float] = 0.7

#: Request budget per minute, applied *per key*: one worker thread is started
#: per key, so a pool of N keys sustains N times this rate.
DEFAULT_RPM: Final[int] = 40
#: Highest ``<VAR>_<N>`` suffix probed when collecting a key pool.
MAX_NUMBERED_KEYS: Final[int] = 32

#: Sampling temperature. Deliberately high: the run needs 20k *distinct*
#: letters, and a low temperature collapses the model onto a few openings.
DEFAULT_TEMPERATURE: Final[float] = 0.4
DEFAULT_TOP_P: Final[float] = 0.7
#: Token ceiling per letter. A long letter is a better letter, so this is set
#: well above the ~1000 completion tokens a typical draft uses: hitting the
#: ceiling truncates the JSON mid-string and forces a full retry, which costs
#: far more than the unused headroom does.
DEFAULT_MAX_TOKENS: Final[int] = 16384
DEFAULT_SEED: Final[int] = 1337

REQUEST_TIMEOUT: Final[float] = 180.0
#: Attempts per sample, covering both rejected drafts and transport errors.
MAX_ATTEMPTS: Final[int] = 5
#: Backoff schedule for retryable failures, in seconds, before jitter.
BACKOFF_BASE: Final[float] = 4.0
BACKOFF_CAP: Final[float] = 120.0
#: Status codes worth retrying: rate limits plus transient server faults.
RETRY_STATUS: Final[frozenset[int]] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: One embedded element on average every ``1 / ELEMENT_PROBABILITY`` letters.
ELEMENT_PROBABILITY: Final[float] = 0.1
#: Element kinds, tables weighted ahead of charts (CLAUDE.md 5.2).
ELEMENT_KINDS: Final[tuple[str, ...]] = ("table", "chart")
ELEMENT_WEIGHTS: Final[tuple[float, ...]] = (0.65, 0.35)

#: Minimum body length in characters. A low floor on purpose: a longer letter
#: is always better, so this only rejects a stunted or broken draft and never
#: an expansive one. There is deliberately no upper bound.
MIN_BODY_CHARS: Final[int] = 250

#: Ceremonial openers a model tends to prepend to a Persian letter even when
#: told not to. They are stripped from the body: the page composer owns the
#: letterhead, so an opener baked into the text would be duplicated on the
#: rendered page and would drift into the OCR ground truth.
#:
#: Only the opening *word* is matched, then the rest of that line is consumed
#: (``[^\n]*``). Matching the full phrase instead invites a partial strip: a
#: pattern for optional «الرحمن الرحیم» removes «بسم الله» from
#: "بسم الله الرحمن الرحیم اداری" and leaves the mangled "الرحمن الرحیم اداری".
#: An opener always occupies its own line or leads the first sentence, so
#: consuming the line is both simpler and safer.
#:
#: Spaces are matched loosely because Persian text mixes ordinary spaces with
#: the zero-width non-joiner (U+200C).
RELIGIOUS_OPENER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[\s‌]*(?:"
    r"بسم[\s‌]*الله"
    r"|بسم[\s‌]*رب"
    r"|با?سمه[\s‌]*تعالی"
    r"|به?[\s‌]*نام[\s‌]*خدا"
    r"|بنام[\s‌]*خدا"
    r"|هو[\s‌]*ال"
    r"|به[\s‌]*یاری[\s‌]*خدا"
    r"|یا[\s‌]*(?:الله|علی|حق)"
    r")[^\n]*\n?",
    re.UNICODE,
)

#: Administrative header lines that belong in their own JSON fields. A model
#: that repeats them inside the body would have the page composer render them
#: twice, so a leading run of them is dropped.
HEADER_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[\s‌]*(?:شماره|تاریخ|پیوست|موضوع|به|از|گیرنده|فرستنده)"
    r"[\s‌]*:[^\n]*\n?",
    re.UNICODE,
)

#: Keys every accepted letter must carry.
REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "company_name",
    "company_category",
    "destination_organization",
    "letter_number",
    "date",
    "national_id",
    "registration_number",
    "subject",
    "body",
    "attachments",
    "signatory_title",
)
#: Keys required inside ``contact_info``.
CONTACT_KEYS: Final[tuple[str, ...]] = ("address", "phone", "email", "website")

#: Fields the model alone can supply -- the creative core of the letter. A
#: draft missing any of these is rejected and retried.
MODEL_ONLY_KEYS: Final[frozenset[str]] = frozenset(
    {"destination_organization", "subject", "body"}
)

#: Signatory titles used to repair an empty ``signatory_title``.
SIGNATORY_TITLES: Final[tuple[str, ...]] = (
    "مدیرعامل و رئیس هیئت‌مدیره",
    "مدیرعامل",
    "رئیس هیئت‌مدیره",
    "نایب‌رئیس هیئت‌مدیره",
    "معاون اجرایی",
    "معاون اداری و مالی",
    "معاون فنی و مهندسی",
    "معاون توسعه و برنامه‌ریزی",
    "مدیر امور اداری",
    "مدیر امور مالی",
    "مدیر روابط عمومی",
    "مدیر دفتر مرکزی",
    "سرپرست امور قراردادها",
    "رئیس اداره پشتیبانی",
    "دبیر هیئت‌مدیره",
)

#: Attachment phrases used to repair an empty ``attachments``.
FALLBACK_ATTACHMENTS: Final[tuple[str, ...]] = (
    "تصویر آگهی تأسیس و آخرین تغییرات شرکت",
    "تصویر گواهی‌نامه صلاحیت فنی",
    "کپی برابر اصل اساسنامه شرکت",
    "فاکتور رسمی و پیش‌فاکتور اقلام",
    "تصویر شناسه ملی و کارت اقتصادی",
    "رونوشت قرارداد منعقدشده",
    "صورت‌جلسه مجمع عمومی عادی",
    "گزارش کارشناسی و مستندات فنی",
    "تصویر مدارک شناسایی امضاکنندگان مجاز",
    "فهرست ریز اقلام و زمان‌بندی اجرا",
)

#: Provinces used to build a plausible address and landline dialling code.
IRAN_CITIES: Final[tuple[tuple[str, str], ...]] = (
    ("تهران", "۰۲۱"),
    ("مشهد", "۰۵۱"),
    ("اصفهان", "۰۳۱"),
    ("شیراز", "۰۷۱"),
    ("تبریز", "۰۴۱"),
    ("کرج", "۰۲۶"),
    ("اهواز", "۰۶۱"),
    ("قم", "۰۲۵"),
    ("کرمانشاه", "۰۸۳"),
    ("رشت", "۰۱۳"),
    ("یزد", "۰۳۵"),
    ("کرمان", "۰۳۴"),
)

#: Street-name components used to build a plausible administrative address.
STREET_NAMES: Final[tuple[str, ...]] = (
    "خیابان ولیعصر",
    "خیابان شریعتی",
    "بلوار کشاورز",
    "خیابان آزادی",
    "بلوار فردوسی",
    "خیابان امام خمینی",
    "خیابان سعدی",
    "بلوار دانشگاه",
    "خیابان طالقانی",
    "بلوار معلم",
)

#: E-mail mailbox names used when the model omits a contact address.
EMAIL_MAILBOXES: Final[tuple[str, ...]] = ("info", "office", "secretariat", "admin", "contact")

#: Latin digits to their Persian-Indic counterparts, for generated numerals.
TO_PERSIAN_DIGITS: Final[dict[int, str]] = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

#: Strips a reasoning preamble emitted by thinking-capable models.
THINK_BLOCK_RE: Final[re.Pattern[str]] = re.compile(r"<think>.*?</think>", re.DOTALL)


class TextGenerationError(RuntimeError):
    """Raised when a usable letter cannot be produced for a sample."""


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompanyProfile:
    """Sender identity taken from one row of the logo manifest.

    Attributes:
        logo_id: Manifest identifier, links the letter back to its artwork.
        slug: Latin slug, reusable for e-mail and website domains.
        title: Brand name, used verbatim as the sender company name.
        categories: Business-activity tags that steer the letter topic.
        svg_path: Manifest-relative path of the vector logo.
        png_path: Manifest-relative path of the raster logo.
        brand_hex: Brand colour, without a leading ``#``.
    """

    logo_id: str
    slug: str
    title: str
    categories: tuple[str, ...]
    svg_path: str = ""
    png_path: str = ""
    brand_hex: str = ""

    @property
    def category_text(self) -> str:
        """Return the activity tags as one Persian-punctuated string."""
        return "، ".join(self.categories) if self.categories else "خدمات عمومی"


def load_companies(logos_csv: Path) -> list[CompanyProfile]:
    """Read the sender-company pool out of the logo manifest.

    Args:
        logos_csv: Path of ``logos.csv``.

    Returns:
        One profile per manifest row that carries a title.

    Raises:
        TextGenerationError: If the manifest is missing or holds no usable row.
    """
    if not logos_csv.is_file():
        raise TextGenerationError(
            f"Logo manifest {logos_csv} not found. Run scripts/download_thesvg_logo.py first."
        )

    companies: list[CompanyProfile] = []
    with logos_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            title = (row.get("title") or "").strip()
            if not title:
                continue
            companies.append(
                CompanyProfile(
                    logo_id=(row.get("logo_id") or "").strip(),
                    slug=(row.get("slug") or "").strip(),
                    title=title,
                    categories=tuple(
                        item.strip()
                        for item in (row.get("categories") or "").split("|")
                        if item.strip()
                    ),
                    svg_path=(row.get("svg_path") or "").strip(),
                    png_path=(row.get("png_path") or "").strip(),
                    brand_hex=(row.get("brand_hex") or "").strip(),
                )
            )

    if not companies:
        raise TextGenerationError(f"Logo manifest {logos_csv} holds no usable row.")
    return companies


@dataclass(frozen=True)
class GeneratorConfig:
    """Runtime configuration for one generation run.

    Attributes:
        output_dir: Directory the ``plaintext_*.json`` files are written to.
        logos_csv: Manifest supplying the sender-company pool.
        count: Total number of letters the output directory should hold.
        model: Model identifier passed to the API.
        base_url: OpenAI-compatible API endpoint.
        api_key_env: Environment variable holding the API key, or several keys
            separated by commas or whitespace.
        api_keys_file: Optional file holding one API key per line.
        signatures_dir: Root of the UTSig corpus supplying ``signature_path``.
        manifest_path: CSV file appended per written letter; ``None`` disables
            the manifest.
        rpm: Per-key request budget per minute.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling cutoff.
        max_tokens: Upper bound on generated tokens per letter.
        seed: Master seed; sample ``i`` derives its own RNG from it.
        element_probability: Chance a letter carries a table or chart.
        overwrite: Regenerate samples whose file already exists.
    """

    output_dir: Path
    logos_csv: Path
    count: int = DEFAULT_COUNT
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = DEFAULT_API_KEY_ENV
    api_keys_file: Path | None = None
    signatures_dir: Path = Path(DEFAULT_SIGNATURES_DIR)
    manifest_path: Path | None = None
    rpm: int = DEFAULT_RPM
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int = DEFAULT_SEED
    element_probability: float = ELEMENT_PROBABILITY
    overwrite: bool = False


@dataclass
class SampleInputs:
    """Everything Python decided for one sample before the model was called.

    These values are persisted alongside the letter so a downstream consumer
    can reproduce the request, and so the manifest builder can join a letter
    back to its logo without re-deriving anything.

    Attributes:
        sample_index: Zero-based index, also the resume key and file number.
        sample_id: Stem of the output file, e.g. ``plaintext_000042``.
        company: Sender profile sampled from the manifest.
        genre: Letter genre drawn from :data:`LETTER_GENRES`.
        tone: Register drawn from :data:`LETTER_TONES`.
        chart_table: ``"table"``, ``"chart"`` or ``"none"``.
    """

    sample_index: int
    sample_id: str
    company: CompanyProfile
    genre: str
    tone: str
    chart_table: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the inputs."""
        return {
            "logo_id": self.company.logo_id,
            "logo_slug": self.company.slug,
            "company_name": self.company.title,
            "company_category": self.company.category_text,
            "company_categories": list(self.company.categories),
            "svg_path": self.company.svg_path,
            "png_path": self.company.png_path,
            "brand_hex": self.company.brand_hex,
            "letter_genre": self.genre,
            "letter_tone": self.tone,
            "chart_table": self.chart_table,
        }


def build_sample_inputs(index: int, config: GeneratorConfig, companies: Sequence[CompanyProfile]) -> SampleInputs:
    """Decide the inputs for one sample.

    The RNG is derived from ``(config.seed, index)``, so sample ``index`` draws
    the same company, genre, tone and element regardless of how the work is
    scheduled across workers or how many times the run is resumed.

    Args:
        index: Zero-based sample index.
        config: Active run configuration.
        companies: Sender-company pool.

    Returns:
        The sample's inputs.
    """
    rng = random.Random(f"{config.seed}:{index}")
    chart_table = "none"
    if rng.random() < config.element_probability:
        chart_table = rng.choices(ELEMENT_KINDS, weights=ELEMENT_WEIGHTS, k=1)[0]
    return SampleInputs(
        sample_index=index,
        sample_id=f"plaintext_{index:06d}",
        company=rng.choice(companies),
        genre=rng.choice(LETTER_GENRES),
        tone=rng.choice(LETTER_TONES),
        chart_table=chart_table,
    )


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

#: Letter genres, sampled per letter. Each entry is only a nudge -- the model
#: still invents the recipient, the subject and the concrete request. The list
#: exists so the corpus spreads across letter types instead of converging on
#: the single "administrative request" the model reaches for unprompted.
LETTER_GENRES: Final[tuple[str, ...]] = (
    "درخواست رسمی برای صدور، تمدید یا اصلاح مجوز و پروانه فعالیت",
    "مکاتبه با اداره ثبت شرکت‌ها درباره ثبت شرکت، تغییر نام، افزایش سرمایه یا تغییرات هیئت‌مدیره",
    "بخشنامه یا نامه داخلی خطاب به واحدهای سازمان درباره رویه‌ها و دستورالعمل‌های اجرایی",
    "مکاتبه خارجی با یک سازمان یا شرکت دیگر درباره همکاری، تفاهم‌نامه یا تبادل اطلاعات",
    "درخواست خرید تجهیزات، ملزومات اداری یا قطعات فنی همراه با شرح اقلام و شرایط تحویل",
    "دعوت به مناقصه، اعلام برنده مناقصه یا شرح بندهای قراردادی و ضمانت‌نامه‌ها",
    "مکاتبه مالی درباره تسویه صورت‌حساب، مالیات بر ارزش افزوده یا اسناد پرداخت",
    "نامه کارگزینی و منابع انسانی درباره استخدام، انتقال، مرخصی، ارتقا یا احکام حقوقی کارکنان",
    "ارائه گزارش دوره‌ای عملکرد فنی، اجرایی یا مالی به مقام بالادست همراه با تحلیل ارقام",
    "دعوت‌نامه رسمی برای جلسه، مجمع عمومی، همایش یا بازدید فنی با ذکر دستور جلسه",
    "اطلاعیه رسمی درباره تغییر ساعات کاری، توقف موقت سرویس یا راه‌اندازی سامانه جدید",
    "مکاتبه با پیمانکار درباره پیشرفت پروژه، صورت‌وضعیت، تحویل موقت یا تحویل قطعی",
    "طرح شکایت اداری یا پیگیری عدم اجرای تعهدات با استناد به بندهای قرارداد",
    "صدور گواهی اشتغال، تأیید سابقه همکاری یا تأییدیه فنی برای ارائه به مرجع ثالث",
    "درخواست استرداد وجه، رفع توقیف حساب یا اصلاح اشتباه در اسناد رسمی",
    "صورت‌جلسه مذاکرات و توافقات حاصل‌شده در نشست مشترک دو سازمان",
    "اعلام فسخ، تعلیق یا تمدید قرارداد با ذکر مستندات و مهلت‌های قانونی",
    "درخواست تخصیص اعتبار، وام یا تسهیلات بانکی همراه با توجیه فنی و اقتصادی",
)

#: Formality registers, sampled per letter so two letters sharing a genre still
#: differ in sentence rhythm and stance.
LETTER_TONES: Final[tuple[str, ...]] = (
    "لحن بسیار رسمی و حقوقی با ارجاع صریح به مواد قانونی و مصوبات",
    "لحن رسمی اداری و محافظه‌کارانه با تأکید بر رویه‌های جاری دبیرخانه",
    "لحن رسمی و قاطع همراه با تعیین ضرب‌الاجل و هشدار نسبت به پیامدهای تأخیر",
    "لحن رسمی و توضیحی با شرح فنی مفصل و مستندسازی گام‌به‌گام",
    "لحن رسمی و تعاملی با تأکید بر همکاری متقابل و حسن نیت طرفین",
)

SYSTEM_PROMPT: Final[str] = (
    "تو یک مشاور ارشد حقوقی و مدیر دفتر اداری مسلط به نگارش اسناد رسمی، صورت‌جلسه‌ها و "
    "نامه‌های پیچیده اداری در ایران هستی. وظیفه تو نگارش نامه‌ها و درخواست‌های اداری "
    "کاملاً جامع، مفصل، رسمی و با جزئیات کامل حقوقی است.\n"
    "خروجی تو باید صرفاً و فقط یک JSON معتبر باشد؛ هیچ متن، سلام، مقدمه، توضیح یا "
    "علامتی مانند ```json قبل یا بعد از آن قرار نده."
)

#: Attachment rule when a table was drawn for this sample.
TABLE_RULE: Final[str] = (
    "در کلید «attachments» علاوه بر ذکر نام پیوست‌ها، یک **جدول کامل به فرمت Markdown** "
    "تولید کن (شامل لیست اقلام، مبالغ، ریز هزینه‌ها، مشخصات فنی یا جدول زمان‌بندی مرتبط "
    "با موضوع) که دست‌کم ۳ ستون و ۴ ردیف داشته باشد، و در متن body نیز به این جدول ارجاع بده."
)
#: Attachment rule when a chart was drawn for this sample.
CHART_RULE: Final[str] = (
    "در کلید «attachments» علاوه بر ذکر نام پیوست‌ها، داده‌های یک نمودار را به فرمت "
    "**جدول Markdown دو ستونی** (ستون اول برچسب و ستون دوم مقدار عددی، دست‌کم ۴ ردیف) "
    "بنویس و پیش از جدول نوع نمودار را با یکی از عبارت‌های «نمودار میله‌ای»، «نمودار خطی» "
    "یا «نمودار دایره‌ای» مشخص کن، و در متن body به این نمودار ارجاع بده."
)
#: Attachment rule for a plain letter.
PLAIN_RULE: Final[str] = "در کلید «attachments» صرفاً لیست متنی پیوست‌های فرضی را بنویس."

#: The JSON contract, shown to the model verbatim.
OUTPUT_SCHEMA_BLOCK: Final[str] = """{
  "company_name": "نام شخص فرستنده بهمراه سمت و عنوان سازمان (دقیقاً همان مقداری که در بالا داده شد)",
  "company_category": "حوزه فعالیت شرکت فرستنده (دقیقاً همان مقداری که در بالا داده شد)",
  "destination_organization": "نام بخش مقصد مانند مدیریت بخشی یا معاونت یا فرد و کارمند خاصی که خودت انتخاب می‌کنی",
  "letter_number": "شماره نامه رندوم رسمی (مثلا: ۱۲۳/الف/۰۳)",
  "date": "تاریخ رندوم شمسی",
  "national_id": "شناسه ملی ۱۱ رقمی رندوم",
  "registration_number": "شماره ثبت رندوم ۵ رقمی",
  "subject": "موضوع دقیق نامه که خودت تعیین می‌کنی",
  "body": "متن نامه که مستقیماً با «احتراماً» یا «به استناد» آغاز می‌شود: کامل، مفصل و حقوقی، شامل تمام پاراگراف‌ها، جزئیات، مبالغ، شروط و بندهای اداری...",
  "attachments": "لیست پیوست‌ها طبق قانون ۴",
  "signatory_title": "سمت امضاکننده که خودت انتخاب می‌کنی (مثلا: مدیرعامل، معاون اجرایی، رئیس هیئت‌مدیره)",
  "contact_info": {
    "address": "آدرس کامل اداری رندوم با کدپستی ۱۰ رقمی",
    "phone": "شماره تلفن ثابت رندوم همراه با کد استان",
    "email": "ایمیل رسمی رندوم با دامنه شرکت",
    "website": "آدرس وب‌سایت رندوم"
  }
}"""


def build_prompt(inputs: SampleInputs) -> str:
    """Compose the user prompt for one letter.

    The prompt fixes only the sender company, a genre nudge and a tone; the
    recipient, subject, request and every identifier are left to the model,
    which is what produces genuinely varied letters across a large run.

    Args:
        inputs: The sample's inputs.

    Returns:
        The prompt text.
    """
    attachment_rule = {"table": TABLE_RULE, "chart": CHART_RULE}.get(inputs.chart_table, PLAIN_RULE)

    return f"""یک نامه/درخواست اداری کاملاً جامع، مفصل و رسمی بنویس.

مشخصات پایه:
- نام شرکت/سازمان مبدأ: {inputs.company.title}
- حوزه فعالیت شرکت/سازمان مبدأ: {inputs.company.category_text}
- نوع و زمینه نامه: {inputs.genre}
- لحن مورد انتظار: {inputs.tone}

قوانین و الزامات مهم نگارش:
۱. بخش مقصد، موضوع نامه و شرح درخواست را خودت با خلاقیت و متناسب با حوزه فعالیت \
شرکت مبدأ انتخاب کن. تمام داده‌های عددی و شناختی (از جمله: شماره ثبت، شناسه ملی، \
شماره نامه، تاریخ شمسی، کد پستی، آدرس، تلفن، ایمیل، وب‌سایت، مبالغ، نام اعضا یا \
قطعات) را به‌صورت رندوم اما کاملاً دقیق، واقعی‌نما و متناسب با عرف اداری ایران تولید کن.
۲. بخش body باید حداقل ۶ الی ۸ پاراگراف مفصل و کامل باشد، متناسب با حوزه فعالیت شرکت \
تنظیم شود و هر چه طولانی‌تر، دقیق‌تر و پرجزئیات‌تر باشد بهتر است (هیچ محدودیتی برای \
طول متن وجود ندارد). body باید شامل موارد زیر گردد:
   - مقدمه رسمی و اشاره به قوانین، بخشنامه‌ها یا مصوبات مربوطه.
   - شرح کامل جزئیات، مشخصات فنی/حقوقی، دلایل و مستندات درخواست.
   - جداسازی شروط، تعهدات یا اقلام در قالب بندها یا شماره‌گذاری‌های شفاف داخل متن.
   - جمع‌بندی رسمی، قدردانی و تعیین مهلت یا درخواست اقدام عاجل.
۳. اعداد داخل متن با ارقام فارسی نوشته شوند، پاراگراف‌ها با \\n از هم جدا شوند و هیچ \
کاراکتر کنترلی دیگری در متن نیاید. نام و نام خانوادگی امضاکننده را بنویس؛ و سمت او \
در signatory_title بیاید.
۴. متن body را مستقیماً با نخستین جملهٔ محتوایی نامه آغاز کن؛ برای مثال با \
«احتراماً، ...» یا «به استناد ...» یا «پیرو مکاتبه شماره ...». پیش از آن هیچ سربرگ، \
عبارت افتتاحیه، تشریفات آغازین یا سطر جداگانه‌ای ننویس. سطرهای «شماره:»، «تاریخ:»، \
«پیوست:»، «به:»، «از:» و «موضوع:» را هم داخل body تکرار نکن، چون این مقادیر جداگانه \
در کلیدهای مخصوص خودشان قرار می‌گیرند.
۵. {attachment_rule}
۶. این نامه باید کاملاً منحصربه‌فرد باشد: از جملات کلیشه‌ای و ساختار تکراری پرهیز کن و \
سناریوی خاص و باورپذیر خودت را بساز.
7. نامه ای که تولید میکنی باید بصورت داخلی باشد یعنی نامه درون سازمانی باشد و به هیچ عنوان نامه ای که برای عموم مردم یا مشتریان باشد تولید نکن. شخص یا بخش مبدا و مقصد رو به درستی انتخاب کن.

فرمت دقیق خروجی JSON:
{OUTPUT_SCHEMA_BLOCK}"""


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


def _retry_after_seconds(exc: APIError) -> float | None:
    """Read a ``Retry-After`` hint off a failed response, if the server sent one.

    Args:
        exc: The API error raised by the client.

    Returns:
        The requested delay in seconds, or ``None`` when absent or unparsable.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def load_api_keys(env_var: str, keys_file: Path | None = None) -> list[str]:
    """Collect every API key available for this run.

    Keys are gathered from three places so a pool can be supplied whichever way
    is convenient: the base variable (which may itself hold a comma- or
    whitespace-separated list), numbered variants ``<VAR>_1`` .. ``<VAR>_32``,
    and an optional file with one key per line.

    Args:
        env_var: Base environment variable name, e.g. ``NVIDIA_API_KEY``.
        keys_file: Optional file holding one key per line; ``#`` starts a
            comment.

    Returns:
        The keys, de-duplicated, in discovery order.

    Raises:
        TextGenerationError: If no key was found anywhere.
    """
    found: list[str] = []

    def absorb(raw: str | None) -> None:
        """Split one raw value on commas and whitespace into individual keys."""
        for piece in re.split(r"[,\s]+", (raw or "").strip()):
            if piece:
                found.append(piece)

    absorb(os.getenv(env_var))
    for index in range(1, MAX_NUMBERED_KEYS + 1):
        absorb(os.getenv(f"{env_var}_{index}"))

    if keys_file is not None:
        if not keys_file.is_file():
            raise TextGenerationError(f"API key file {keys_file} not found.")
        for line in keys_file.read_text(encoding="utf-8").splitlines():
            absorb(line.split("#", 1)[0])

    keys = list(dict.fromkeys(found))
    if not keys:
        raise TextGenerationError(
            f"No API key found. Set {env_var} (one key, or several separated by commas), "
            f"or {env_var}_1 / {env_var}_2 ..., or pass --api-keys-file."
        )
    return keys


def mask_key(key: str) -> str:
    """Render an API key safely for logs.

    Args:
        key: The full key.

    Returns:
        A short masked form such as ``...a1b2``.
    """
    return f"...{key[-4:]}" if len(key) > 4 else "..."


class KeyWorker:
    """One API key, its own client, and the requests it issues sequentially.

    Each key gets exactly one worker, and a worker holds one request in flight
    at a time. That makes rate limiting trivial: a throttled key simply sleeps
    in place, and because workers share no scheduling state, the other keys
    keep generating letters throughout. No cursor, cooldown table or scheduler
    lock is needed.
    """

    def __init__(self, key: str, config: GeneratorConfig) -> None:
        """Bind a client to one API key.

        Args:
            key: The API key this worker owns.
            config: Active run configuration.
        """
        self.label = mask_key(key)
        self.throttled = 0
        self._config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=key,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,  # retries and backoff are handled in complete()
        )
        #: Minimum spacing between this key's own requests, for --rpm.
        self._min_interval = 60.0 / config.rpm if config.rpm > 0 else 0.0
        self._next_slot = 0.0

    def _pace(self) -> None:
        """Sleep until this key is allowed to issue its next request."""
        if self._min_interval <= 0.0:
            return
        wait = self._next_slot - time.monotonic()
        if wait > 0.0:
            time.sleep(wait)
        self._next_slot = time.monotonic() + self._min_interval

    def complete(self, prompt: str, rng: random.Random) -> str:
        """Request one completion, waiting out this key's own rate limits.

        Args:
            prompt: User prompt built by :func:`build_prompt`.
            rng: Per-sample RNG, used to jitter the backoff.

        Returns:
            The model's raw reply text.

        Raises:
            TextGenerationError: If every attempt failed.
        """
        last_error = "no attempt was made"
        for attempt in range(MAX_ATTEMPTS):
            self._pace()
            try:
                completion = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    max_tokens=self._config.max_tokens,
                    #: Deliberately no response_format={"type": "json_object"}.
                    #: This endpoint's guided-JSON decoder truncates every
                    #: string at 512 characters and returns nested objects as
                    #: {}, which costs the letter body and all of contact_info.
                    #: parse_model_json() tolerates fences and prose instead.
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except APIError as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and status not in RETRY_STATUS:
                    raise TextGenerationError(f"API rejected the request ({status}): {exc}") from exc
                delay = _retry_after_seconds(exc) or min(BACKOFF_CAP, BACKOFF_BASE * 2**attempt)
                delay += rng.uniform(0.0, 1.5)
                self.throttled += 1
                last_error = f"transport error ({status or 'connection'}): {exc}"
                #: Only this key waits; every other key keeps working.
                LOGGER.warning("key %s: %s -- waiting %.1fs", self.label, last_error, delay)
                time.sleep(delay)
                continue

            if not completion.choices:
                last_error = "reply carried no choices"
                LOGGER.debug("key %s: %s (attempt %d)", self.label, last_error, attempt + 1)
                continue

            choice = completion.choices[0]
            text = choice.message.content
            #: Distinguish a real token-budget truncation from a short draft, so
            #: a run that needs a larger --max-tokens says so plainly instead of
            #: reporting a missing trailing field like signatory_title.
            if choice.finish_reason == "length":
                last_error = (
                    f"reply hit the {self._config.max_tokens}-token ceiling; raise --max-tokens"
                )
                LOGGER.warning("key %s: %s (attempt %d)", self.label, last_error, attempt + 1)
                continue
            if text and text.strip():
                return text
            last_error = "model returned an empty reply"
            LOGGER.debug("key %s: %s (attempt %d)", self.label, last_error, attempt + 1)

        raise TextGenerationError(f"after {MAX_ATTEMPTS} attempts: {last_error}")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _extract_json_span(text: str) -> tuple[str, bool]:
    """Isolate the outermost JSON object in a reply.

    Scans for balanced braces while ignoring any that sit inside a JSON string,
    which a greedy ``{.*}`` regex cannot do: it over-captures when the reply
    ends with prose containing a brace, and captures nothing at all when the
    reply was cut off before its closing brace.

    Args:
        text: Cleaned reply text.

    Returns:
        The object's source span, and whether it was left unterminated.

    Raises:
        TextGenerationError: If the text holds no opening brace.
    """
    start = text.find("{")
    if start < 0:
        raise TextGenerationError("reply contains no JSON object")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1], False

    return text[start:], True


def _repair_truncated_json(span: str) -> str:
    """Close a JSON object that the model stopped emitting mid-way.

    A reply cut off inside the trailing fields still carries a complete body
    and subject, so salvaging it saves a whole API call;
    :func:`fill_missing_fields` supplies whatever the lost tail contained.

    Args:
        span: Unterminated JSON source.

    Returns:
        A source string with strings and containers closed off.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in span:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and stack:
            stack.pop()

    repaired = span[:-1] if escaped else span
    if in_string:
        repaired += '"'
    #: Drop a dangling comma or a key whose value never arrived, so the
    #: reconstructed object closes as valid JSON.
    repaired = re.sub(r",\s*$", "", repaired.rstrip())
    repaired = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", repaired)
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def parse_model_json(raw: str) -> dict[str, Any]:
    """Extract the JSON object from a model reply.

    Tolerates a ``<think>`` preamble, ```` ```json ```` fences, surrounding
    prose and a truncated tail, since not every model honours a bare "JSON
    only" instruction.

    Args:
        raw: Raw reply text.

    Returns:
        The decoded object.

    Raises:
        TextGenerationError: If no JSON object can be decoded.
    """
    cleaned = THINK_BLOCK_RE.sub("", raw).replace("```json", "").replace("```", "").strip()
    span, truncated = _extract_json_span(cleaned)

    candidates = [span]
    if truncated:
        LOGGER.debug("reply was truncated mid-object; attempting repair")
        candidates.append(_repair_truncated_json(span))

    error = "no candidate parsed"
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            error = str(exc)
            continue
        if not isinstance(payload, dict):
            raise TextGenerationError(f"reply is a {type(payload).__name__}, expected an object")
        return payload

    raise TextGenerationError(f"reply is not valid JSON: {error}")


def clean_body(body: str) -> str:
    """Strip ceremonial openers and duplicated header lines from a letter body.

    The prompt already forbids both, but a model asked for a formal Persian
    letter reaches for «بسم الله» by reflex, so the guarantee is enforced here
    rather than merely requested. Rendering owns the letterhead; leaving an
    opener in the text would duplicate it on the page and teach the OCR model
    to expect it.

    Args:
        body: Raw body text from the model.

    Returns:
        The body starting at its first line of real content.
    """
    cleaned = body.strip()
    #: Loop: a reply may stack an opener and header lines in either order.
    while True:
        stripped = RELIGIOUS_OPENER_RE.sub("", cleaned, count=1)
        stripped = HEADER_LINE_RE.sub("", stripped, count=1)
        stripped = stripped.lstrip()
        if stripped == cleaned:
            return cleaned
        cleaned = stripped


def validate_letter(payload: dict[str, Any], inputs: SampleInputs) -> dict[str, Any]:
    """Check a draft against the output contract and normalise it.

    Only the fields in :data:`MODEL_ONLY_KEYS` are enforced here. Everything
    else is repaired by :func:`fill_missing_fields`, so a well-written letter
    is never discarded over an omitted phone number.

    Args:
        payload: Object decoded from the model reply.
        inputs: The sample's inputs, used to repair the echoed sender fields.

    Returns:
        The letter, with every contract key present and stripped.

    Raises:
        TextGenerationError: If a model-only field is missing or the body is
            shorter than :data:`MIN_BODY_CHARS`.
    """
    missing = [key for key in MODEL_ONLY_KEYS if not str(payload.get(key, "")).strip()]
    if missing:
        raise TextGenerationError(f"missing field(s): {', '.join(sorted(missing))}")

    #: Cleaned before the length check, so a body padded out with an opener and
    #: repeated header lines is not credited for them.
    body = clean_body(str(payload["body"]))
    if len(body) < MIN_BODY_CHARS:
        raise TextGenerationError(f"body is {len(body)} chars, need {MIN_BODY_CHARS}")

    letter = {key: str(payload.get(key, "")).strip() for key in REQUIRED_KEYS}
    letter["body"] = body
    #: The sender fields were dictated, not invented -- trust the manifest over
    #: whatever the model echoed back, so the join key stays exact.
    letter["company_name"] = inputs.company.title
    letter["company_category"] = inputs.company.category_text

    contact = payload.get("contact_info")
    contact = contact if isinstance(contact, dict) else {}
    letter["contact_info"] = {key: str(contact.get(key, "")).strip() for key in CONTACT_KEYS}
    return letter


# --------------------------------------------------------------------------- #
# Local fallbacks
# --------------------------------------------------------------------------- #


def _persian(number: int) -> str:
    """Render an integer with Persian-Indic digits.

    Args:
        number: Value to render.

    Returns:
        The number as a Persian-digit string.
    """
    return str(number).translate(TO_PERSIAN_DIGITS)


def synthesize_contact_info(inputs: SampleInputs, rng: random.Random) -> dict[str, str]:
    """Build a contact block consistent with the sender company.

    The e-mail and website are derived from the company's own slug, so a
    repaired contact block still agrees with the letterhead the page will be
    composited with, rather than naming an unrelated domain.

    Args:
        inputs: The sample's inputs.
        rng: Per-sample RNG, keeping the result reproducible from the seed.

    Returns:
        A full contact block.
    """
    domain = (inputs.company.slug or "company").lower() or "company"
    city, dial_code = rng.choice(IRAN_CITIES)
    postal_code = _persian(rng.randrange(10**9, 10**10))
    return {
        "address": (
            f"{city}، {rng.choice(STREET_NAMES)}، "
            f"پلاک {_persian(rng.randrange(1, 400))}، "
            f"طبقه {_persian(rng.randrange(1, 12))}، کدپستی {postal_code}"
        ),
        "phone": f"{dial_code}-{_persian(rng.randrange(10**7, 10**8))}",
        "email": f"{rng.choice(EMAIL_MAILBOXES)}@{domain}.ir",
        "website": f"https://www.{domain}.ir",
    }


def synthesize_letter_number(rng: random.Random) -> str:
    """Build a plausible Iranian administrative letter number.

    Args:
        rng: Per-sample RNG.

    Returns:
        A letter number such as ``۱۲۳/الف/۰۳``.
    """
    series = rng.choice(("الف", "ب", "ج", "د", "ص", "ط", "ک", "م"))
    #: Zero-pad before transliterating: padding the Persian string would splice
    #: an ASCII "0" into the numeral and corrupt the OCR ground truth.
    return f"{_persian(rng.randrange(10, 999))}/{series}/{_persian(f'{rng.randrange(1, 13):02d}')}"


def synthesize_jalali_date(rng: random.Random) -> str:
    """Build a plausible Jalali date string.

    Day is capped at 30 so the date stays valid in every Jalali month without
    pulling in a calendar dependency.

    Args:
        rng: Per-sample RNG.

    Returns:
        A date such as ``۱۴۰۳/۰۷/۲۲``.
    """
    year = rng.randrange(1400, 1405)
    month = rng.randrange(1, 13)
    day = rng.randrange(1, 31)
    #: Zero-pad before transliterating, otherwise an ASCII "0" ends up inside
    #: an otherwise Persian numeral.
    return f"{_persian(year)}/{_persian(f'{month:02d}')}/{_persian(f'{day:02d}')}"


def fill_missing_fields(
    letter: dict[str, Any],
    inputs: SampleInputs,
    rng: random.Random,
) -> list[str]:
    """Repair the fields a letter can survive without the model supplying.

    Identifiers, the signatory title and the contact block are pure boilerplate:
    regenerating a whole letter because one of them came back empty wastes an
    API call for no gain in data quality. They are filled deterministically
    from the sample RNG, so a repaired letter is still reproducible from its
    seed, and every repair is recorded for provenance.

    Args:
        letter: Validated letter, modified in place.
        inputs: The sample's inputs.
        rng: Per-sample RNG.

    Returns:
        The names of the fields that were filled, for the record's audit trail.
    """
    filled: list[str] = []

    generators = {
        "letter_number": lambda: synthesize_letter_number(rng),
        "date": lambda: synthesize_jalali_date(rng),
        "national_id": lambda: _persian(rng.randrange(10**10, 10**11)),
        "registration_number": lambda: _persian(rng.randrange(10**4, 10**5)),
        "signatory_title": lambda: rng.choice(SIGNATORY_TITLES),
        "attachments": lambda: "، ".join(
            rng.sample(FALLBACK_ATTACHMENTS, k=rng.randrange(1, 4))
        ),
    }
    for key, generate in generators.items():
        if not str(letter.get(key, "")).strip():
            letter[key] = generate()
            filled.append(key)

    fallback_contact = synthesize_contact_info(inputs, rng)
    contact = letter.setdefault("contact_info", {})
    for key in CONTACT_KEYS:
        if not str(contact.get(key, "")).strip():
            contact[key] = fallback_contact[key]
            filled.append(f"contact_info.{key}")

    return filled


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def output_path_for(config: GeneratorConfig, inputs: SampleInputs) -> Path:
    """Return the JSON path for one sample.

    Args:
        config: Active run configuration.
        inputs: The sample's inputs.

    Returns:
        ``<output_dir>/plaintext_000042.json``.
    """
    return config.output_dir / f"{inputs.sample_id}.json"


def build_record(
    letter: dict[str, Any],
    inputs: SampleInputs,
    config: GeneratorConfig,
    attempts: int,
    filled_fields: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble the record persisted for one sample.

    Args:
        letter: Validated letter fields.
        inputs: The sample's inputs.
        config: Active run configuration.
        attempts: How many attempts the sample took.
        filled_fields: Fields repaired locally rather than written by the model.

    Returns:
        A JSON-serialisable record holding the letter, its inputs and the
        generation settings that produced it.
    """
    return {
        "sample_id": inputs.sample_id,
        "sample_index": inputs.sample_index,
        "inputs": inputs.as_dict(),
        "letter": letter,
        "generation": {
            "model": config.model,
            "base_url": config.base_url,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
            "attempts": attempts,
            "filled_fields": list(filled_fields),
        },
    }


def write_record(path: Path, record: dict[str, Any]) -> None:
    """Write one record to disk atomically.

    The file is written to a sibling temporary path and then renamed, so a run
    killed mid-write leaves no half-written JSON for the resume scan to trip
    over.

    Args:
        path: Destination path.
        record: Record to serialise.
    """
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def load_signatures(signatures_dir: Path) -> list[Path]:
    """Collect every signature image under the UTSig corpus.

    Both ``Genuine`` and ``Forgery`` are used: the page composer only needs a
    plausible ink shape, and a forged signature looks no less like handwriting
    than a genuine one, so the whole corpus widens the visual variety.

    Args:
        signatures_dir: Root of the UTSig tree.

    Returns:
        Sorted paths of the signature images; empty if the corpus is absent.
    """
    if not signatures_dir.is_dir():
        return []
    return sorted(signatures_dir.rglob("*.tif"))


def assign_signature(
    sample_index: int,
    signatures: Sequence[Path],
    seed: int = DEFAULT_SEED,
    probability: float = SIGNATURE_PROBABILITY,
) -> str:
    """Pick the signature image for one sample, or none.

    Derived from ``(seed, sample_index)`` rather than drawn from a shared RNG,
    so a sample keeps the same signature no matter when it is generated, in
    which order, or how many times the manifest is rebuilt. That is what lets
    the backfill script complete old rows without contradicting a later run.

    Args:
        sample_index: Zero-based sample index.
        signatures: Pool returned by :func:`load_signatures`.
        seed: Master seed, matching the run's own.
        probability: Chance this sample carries a signature at all.

    Returns:
        A path relative to the current directory, or ``""`` when unsigned.
    """
    if not signatures:
        return ""
    rng = random.Random(f"{seed}:signature:{sample_index}")
    if rng.random() >= probability:
        return ""
    chosen = rng.choice(signatures)
    try:
        return str(chosen.relative_to(Path.cwd()))
    except ValueError:
        return str(chosen)


#: Manifest columns. Long free text (``body``, ``attachments``) is deliberately
#: left in the JSON: at 20k samples it would add ~35 MB of duplicated corpus to
#: the CSV, and multi-line Persian prose makes the file fragile to read back.
#: ``sample_path`` points at the letter for anything not summarised here.
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "sample_index",
    "sample_path",
    "logo_path",
    "logo_title",
    "logo_categories",
    "subject",
    "destination_organization",
    "letter_number",
    "date",
    "national_id",
    "registration_number",
    "signatory_title",
    "chart_table",
    "letter_genre",
    "letter_tone",
    "body_chars",
    "has_attachments",
    "filled_fields",
    "signature_path",
)


def manifest_row(
    record: dict[str, Any],
    sample_path: Path,
    signature_path: str,
) -> dict[str, str]:
    """Flatten one letter record into a manifest row.

    Args:
        record: The record as built by :func:`build_record`.
        sample_path: Path of the letter's JSON file.
        signature_path: Value for the ``signature_path`` column.

    Returns:
        One row keyed by :data:`MANIFEST_COLUMNS`.
    """
    inputs = record.get("inputs", {})
    letter = record.get("letter", {})
    filled = record.get("generation", {}).get("filled_fields") or []
    return {
        "sample_id": record.get("sample_id", ""),
        "sample_index": str(record.get("sample_index", "")),
        "sample_path": str(sample_path),
        "logo_path": inputs.get("svg_path", ""),
        "logo_title": inputs.get("company_name", ""),
        "logo_categories": inputs.get("company_category", ""),
        "subject": letter.get("subject", ""),
        "destination_organization": letter.get("destination_organization", ""),
        "letter_number": letter.get("letter_number", ""),
        "date": letter.get("date", ""),
        "national_id": letter.get("national_id", ""),
        "registration_number": letter.get("registration_number", ""),
        "signatory_title": letter.get("signatory_title", ""),
        "chart_table": inputs.get("chart_table", "none"),
        "letter_genre": inputs.get("letter_genre", ""),
        "letter_tone": inputs.get("letter_tone", ""),
        "body_chars": str(len(letter.get("body", ""))),
        "has_attachments": "yes" if letter.get("attachments", "").strip() else "no",
        "filled_fields": "|".join(filled),
        "signature_path": signature_path,
    }


class ManifestWriter:
    """Append-as-you-go CSV manifest, one row per letter written.

    The manifest used to be rebuilt from disk after a run finished, which meant
    a 20k run interrupted at sample 19,000 left no manifest at all. Appending
    each row as its JSON lands makes the CSV survive a Ctrl-C or a crash, and
    keeps it usable while a long run is still going.

    Rows are flushed immediately and guarded by a lock, since several key
    worker threads write concurrently. Indices already present in the file are
    remembered so a resumed run does not duplicate them.
    """

    def __init__(self, path: Path) -> None:
        """Open the manifest for appending, writing a header if it is new.

        Args:
            path: CSV file to append to; parent directories are created.
        """
        self._path = path
        self._lock = threading.Lock()
        self._seen: set[str] = set()

        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.is_file() or path.stat().st_size == 0
        if not is_new:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    index = (row.get("sample_index") or "").strip()
                    if index:
                        self._seen.add(index)

        self._handle = path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=MANIFEST_COLUMNS)
        if is_new:
            self._writer.writeheader()
            self._handle.flush()

    @property
    def existing_indices(self) -> frozenset[str]:
        """Return sample indices already recorded when the file was opened."""
        return frozenset(self._seen)

    def append(
        self,
        record: dict[str, Any],
        sample_path: Path,
        signature_path: str,
    ) -> bool:
        """Append one letter's row, skipping an index already written.

        Args:
            record: The record just written to disk.
            sample_path: Path of the letter's JSON file.
            signature_path: Value for the ``signature_path`` column.

        Returns:
            ``True`` if a row was written, ``False`` if it was a duplicate.
        """
        row = manifest_row(record, sample_path, signature_path)
        with self._lock:
            if row["sample_index"] in self._seen:
                return False
            self._seen.add(row["sample_index"])
            self._writer.writerow(row)
            self._handle.flush()
        return True

    def close(self) -> None:
        """Close the underlying file handle."""
        with self._lock:
            self._handle.close()

    def __enter__(self) -> ManifestWriter:
        """Return self so the writer can be used as a context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the manifest, including when the run raises."""
        self.close()


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


@dataclass
class RunCounters:
    """Tallies describing how a run progressed.

    Attributes:
        skipped: Samples already on disk when the run started.
        written: Letters written by this run.
        failed: Samples abandoned after :data:`MAX_ATTEMPTS` attempts.
        rejected: Individual drafts rejected by validation and retried.
        filled: Letters that needed at least one field repaired locally.
        by_element: Written letters per ``chart_table`` value.
    """

    skipped: int = 0
    written: int = 0
    failed: int = 0
    rejected: int = 0
    filled: int = 0
    by_element: dict[str, int] = field(default_factory=dict)


def generate_one(
    inputs: SampleInputs,
    client: KeyWorker,
    config: GeneratorConfig,
    counters: RunCounters,
) -> dict[str, Any]:
    """Generate, validate and return the record for one sample.

    Args:
        inputs: The sample's inputs.
        client: The calling thread's key worker.
        config: Active run configuration.
        counters: Tallies updated in place.

    Returns:
        The record to persist.

    Raises:
        TextGenerationError: If every attempt was rejected.
    """
    rng = random.Random(f"{config.seed}:draft:{inputs.sample_index}")
    prompt = build_prompt(inputs)

    last_error = "no attempt was made"
    for attempt in range(MAX_ATTEMPTS):
        raw = client.complete(prompt, rng)
        try:
            letter = validate_letter(parse_model_json(raw), inputs)
        except TextGenerationError as exc:
            last_error = str(exc)
            counters.rejected += 1
            LOGGER.debug("%s attempt %d rejected: %s", inputs.sample_id, attempt + 1, last_error)
            continue

        filled = fill_missing_fields(letter, inputs, rng)
        if filled:
            counters.filled += 1
            LOGGER.debug("%s filled locally: %s", inputs.sample_id, ", ".join(filled))
        return build_record(letter, inputs, config, attempt + 1, filled)

    raise TextGenerationError(f"{inputs.sample_id} failed after {MAX_ATTEMPTS} attempts: {last_error}")


def generate_letters(config: GeneratorConfig) -> RunCounters:
    """Generate letters until the output directory holds ``config.count`` files.

    Samples whose file already exists are skipped unless ``config.overwrite``
    is set, which is what makes an interrupted run resumable.

    Args:
        config: Active run configuration.

    Returns:
        The final tallies of the run.

    Raises:
        TextGenerationError: If the company pool or the API keys are unavailable.
    """
    companies = load_companies(config.logos_csv)
    api_keys = load_api_keys(config.api_key_env, config.api_keys_file)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    all_inputs = [build_sample_inputs(index, config, companies) for index in range(config.count)]
    pending = [
        item
        for item in all_inputs
        if config.overwrite or not output_path_for(config, item).is_file()
    ]
    counters = RunCounters(skipped=config.count - len(pending))

    LOGGER.info(
        "%d company profile(s); %d/%d sample(s) already on disk, %d to generate "
        "(model=%s, keys=%d, rpm=%d/key, temperature=%.2f)",
        len(companies),
        counters.skipped,
        config.count,
        len(pending),
        config.model,
        len(api_keys),
        config.rpm,
        config.temperature,
    )
    if not pending:
        return counters

    #: One queue feeds every key, so a fast key simply takes more samples than
    #: a throttled one -- no work needs to be partitioned up front.
    work: queue.Queue[SampleInputs] = queue.Queue()
    for item in pending:
        work.put(item)

    workers = [KeyWorker(key, config) for key in api_keys]
    write_lock = threading.Lock()
    stop = threading.Event()
    log_every = max(1, len(pending) // 50)

    signatures = load_signatures(config.signatures_dir)
    if config.manifest_path is not None and not signatures:
        LOGGER.warning(
            "No .tif signatures under %s; signature_path will be empty",
            config.signatures_dir,
        )
    #: Opened for the whole run and appended per letter, so an interrupted run
    #: still leaves a manifest describing everything already on disk.
    manifest = ManifestWriter(config.manifest_path) if config.manifest_path else None

    def run_key(client: KeyWorker) -> None:
        """Pull samples off the queue until it drains, using one key."""
        while not stop.is_set():
            try:
                item = work.get_nowait()
            except queue.Empty:
                return
            try:
                record = generate_one(item, client, config, counters)
            except TextGenerationError as exc:
                with write_lock:
                    counters.failed += 1
                LOGGER.warning("%s", exc)
                continue
            except Exception:  # noqa: BLE001 - logged with context, thread must not die silently
                with write_lock:
                    counters.failed += 1
                LOGGER.exception("key %s: unexpected failure on %s", client.label, item.sample_id)
                continue

            sample_path = output_path_for(config, item)
            with write_lock:
                write_record(sample_path, record)
                counters.written += 1
                counters.by_element[item.chart_table] = (
                    counters.by_element.get(item.chart_table, 0) + 1
                )
                written = counters.written

            #: Appended right after the JSON lands, not at the end of the run.
            if manifest is not None:
                manifest.append(
                    record,
                    sample_path,
                    assign_signature(item.sample_index, signatures, config.seed),
                )

            if written % log_every == 0:
                LOGGER.info(
                    "  %d/%d written (latest %s, %d rejected, %d failed)",
                    written,
                    len(pending),
                    item.sample_id,
                    counters.rejected,
                    counters.failed,
                )

    LOGGER.info("Starting %d worker thread(s), one per API key", len(workers))
    threads = [
        threading.Thread(target=run_key, args=(client,), name=f"key-{client.label}", daemon=True)
        for client in workers
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; finishing in-flight requests. Re-run to resume.")
        stop.set()
        for thread in threads:
            thread.join(timeout=REQUEST_TIMEOUT)
        raise
    finally:
        if manifest is not None:
            manifest.close()

    throttled = ", ".join(f"{client.label}: {client.throttled}" for client in workers)
    LOGGER.info("Per-key throttle waits: %s", throttled)
    return counters


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


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
    logging.getLogger("httpx").setLevel(logging.WARNING)


def load_dotenv_if_available() -> None:
    """Load a local ``.env`` file when ``python-dotenv`` is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def parse_args(argv: Sequence[str] | None = None) -> GeneratorConfig:
    """Parse command line arguments into a configuration object.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate Persian official letters through an OpenAI-compatible API "
            "and write one resumable JSON file per letter."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR),
                        help="Directory the plaintext_*.json files are written to.")
    parser.add_argument("--logos-csv", type=Path, default=Path(DEFAULT_LOGOS_CSV),
                        help="Logo manifest supplying the sender-company pool.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="Total number of letters the output directory should hold.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API endpoint.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV,
                        help="Environment variable holding the API key(s). May hold several "
                             "keys separated by commas or whitespace; numbered variants "
                             "(<VAR>_1, <VAR>_2, ...) are also collected.")
    parser.add_argument("--api-keys-file", type=Path, default=None,
                        help="Optional file holding one API key per line.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="CSV manifest appended as each letter is written; "
                             "omit to skip the manifest.")
    parser.add_argument("--signatures-dir", type=Path, default=Path(DEFAULT_SIGNATURES_DIR),
                        help="Root of the UTSig corpus supplying signature_path.")
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM,
                        help="Request budget per minute per key; 0 disables pacing.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="Sampling temperature; higher means more varied letters.")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P,
                        help="Nucleus-sampling cutoff.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Token ceiling per letter.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Master seed; sample i derives its own RNG from it.")
    parser.add_argument("--element-probability", type=float, default=ELEMENT_PROBABILITY,
                        help="Chance a letter carries a table or chart.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate samples whose file already exists.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging, including every rejected draft.")

    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.count <= 0:
        parser.error("--count must be positive.")
    if not 0.0 <= args.element_probability <= 1.0:
        parser.error("--element-probability must be between 0 and 1.")

    return GeneratorConfig(
        output_dir=args.output_dir,
        logos_csv=args.logos_csv,
        count=args.count,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        api_keys_file=args.api_keys_file,
        signatures_dir=args.signatures_dir,
        manifest_path=args.manifest,
        rpm=args.rpm,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        element_probability=args.element_probability,
        overwrite=args.overwrite,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the letter generator from the command line.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` on a handled failure, ``130`` on interrupt.
    """
    load_dotenv_if_available()
    config = parse_args(argv)
    try:
        counters = generate_letters(config)
    except TextGenerationError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; re-run the same command to resume.")
        return 130

    total = counters.skipped + counters.written
    LOGGER.info(
        "Done: %d/%d letter(s) in %s (%d written this run, %d already present)",
        total,
        config.count,
        config.output_dir,
        counters.written,
        counters.skipped,
    )
    LOGGER.info(
        "Elements: %d table(s), %d chart(s), %d plain. Rejected drafts: %d. "
        "Locally filled: %d. Failed samples: %d",
        counters.by_element.get("table", 0),
        counters.by_element.get("chart", 0),
        counters.by_element.get("none", 0),
        counters.rejected,
        counters.filled,
        counters.failed,
    )
    if total < config.count:
        LOGGER.warning(
            "%d sample(s) short of the target; re-run the same command to fill the gaps.",
            config.count - total,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
