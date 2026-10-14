"""Score the endpoint's models on Persian generation, and rank them.

Why this is a script and not a one-off
--------------------------------------

The model choice in ``configs/diverse_config.yaml`` is the single biggest lever
on corpus quality, and it cannot be made from a catalogue listing. Measuring it
the first time produced two findings that no amount of reading model cards
would have:

* Of 63 chat models the endpoint lists, only 13 were callable on this account.
  The rest return ``404 Not found for account``.
* ``nemotron-3-super-120b-a12b`` is fast and fluent and **leaks other scripts
  into Persian prose**. Over 150 real samples it produced Han, Cyrillic, Kana,
  Devanagari, Hangul, Hebrew and Thai characters, and 73.6% of its output was
  rejected by stage 03. ``kimi-k3`` leaked none.

That will change as NVIDIA adds and removes models, so the measurement needs
to be repeatable. Run this before every scale-up.

What it measures
----------------

The metrics are the stage 03 rejection reasons, applied to a fixed prompt set
so models are compared on identical work:

``foreign``
    Fraction of samples containing a script Persian never uses. This is the
    decisive metric and it is **not** a ratio - one Cyrillic letter is a
    defect, because a label containing it teaches the model to emit it.
``persian``
    Mean Persian-letter ratio.
``words``
    Mean hit rate on the requested rare words, after light stemming. A model
    that quietly drops the vocabulary it was given defeats stages 00 and 01.
``length``
    Fraction within the requested word range. Catches models that answer a
    60-word request with 900 words.
``clean``
    Fraction that would survive stage 03 entirely. This is the number to rank
    on, because it is the fraction of spend that becomes data.

Usage::

    python scripts/evaluate_diverse_models.py                  # callable models
    python scripts/evaluate_diverse_models.py --models a b c   # just these
    python scripts/evaluate_diverse_models.py --repeats 3      # tighter bars
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import logging
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LOGGER: Final[logging.Logger] = logging.getLogger("evaluate_diverse_models")

#: Attempts per request before a model is scored as having failed. Three is
#: enough to ride out a throttle without turning the run into an hour of
#: backoff when a model is genuinely unavailable.
_MAX_ATTEMPTS: Final[int] = 3

#: Model name fragments that mark a model as not a Persian text generator:
#: embeddings, rerankers, safety classifiers, OCR and vision-only models.
_SKIP: Final[tuple[str, ...]] = (
    "embed",
    "reward",
    "safety",
    "guard",
    "parse",
    "rerank",
    "ocr",
    "-vl",
    "vila",
    "clip",
    "retriev",
    "translate",
    "diffusion",
)

#: A fixed prompt set, one per visual category family, so every model is scored
#: on the same spread of registers and lengths. Each entry is
#: ``(genre, register, length_low, length_high, words)``.
#:
#: The words are real lexicon entries drawn from the live build: two Academy
#: neologisms, a literary term, a dialect form and a technical compound. They
#: are deliberately hard - the whole point of stage 00 - because a model that
#: handles common words and breaks on these is a model that will break on the
#: corpus.
_TASKS: Final[tuple[tuple[str, str, int, int, tuple[str, ...]], ...]] = (
    ("گزارش خبری", "خبری", 60, 90, ("خاموشگر", "بالگرد")),
    ("مدخل دانشنامه", "دانشگاهی", 60, 90, ("گرمامرگی", "عصبشناسی")),
    ("غزل به سبک کلاسیک", "ادبی", 40, 70, ("آوردگه", "رادمردی")),
    ("گفتگوی دونفره پیام‌رسان", "محاوره‌ای تهرانی", 40, 70, ("یخ‌قهوه",)),
    ("شعار تبلیغاتی", "تبلیغاتی", 1, 6, ("نوآورانه",)),
    ("ماده قراردادی حقوقی", "اداری و بروکراتیک", 60, 90, ("تسری", "دادخواه")),
)


@dataclass
class ModelScore:
    """Accumulated metrics for one model.

    Attributes:
        model: Model id.
        calls: Attempts made.
        failures: Calls that raised or returned nothing.
        latencies: Per-call wall time in seconds.
        persian: Per-call Persian-letter ratio.
        word_rates: Per-call rare-word hit rate.
        foreign: Per-call flag, whether a forbidden script appeared.
        in_length: Per-call flag, whether the length target was met.
        clean: Per-call flag, whether the sample would pass stage 03.
        scripts: Tally of which forbidden scripts appeared.
        samples: A couple of excerpts, for eyeballing.
        last_error: Why the most recent call failed. Present so a zero score
            reads as "404 for this account" or "429 throttled" rather than as
            a verdict on the model's Persian.
    """

    model: str
    calls: int = 0
    failures: int = 0
    latencies: list[float] = field(default_factory=list)
    persian: list[float] = field(default_factory=list)
    word_rates: list[float] = field(default_factory=list)
    foreign: list[bool] = field(default_factory=list)
    in_length: list[bool] = field(default_factory=list)
    clean: list[bool] = field(default_factory=list)
    scripts: dict[str, int] = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)
    last_error: str = ""

    @property
    def ok(self) -> int:
        """Return the number of calls that produced text."""
        return self.calls - self.failures

    def mean(self, values: Sequence[float | bool]) -> float:
        """Return a mean, or zero for an empty series.

        Args:
            values: The series.

        Returns:
            The mean.
        """
        return statistics.mean(float(value) for value in values) if values else 0.0

    def as_row(self) -> dict[str, object]:
        """Return the JSON-serialisable summary row."""
        return {
            "model": self.model,
            "calls": self.calls,
            "ok": self.ok,
            "median_latency_s": (
                round(statistics.median(self.latencies), 1) if self.latencies else None
            ),
            "foreign_rate": round(self.mean(self.foreign), 3),
            "persian_ratio": round(self.mean(self.persian), 3),
            "word_hit_rate": round(self.mean(self.word_rates), 3),
            "in_length_rate": round(self.mean(self.in_length), 3),
            "clean_rate": round(self.mean(self.clean), 3),
            "scripts_leaked": dict(sorted(self.scripts.items(), key=lambda kv: -kv[1])),
            "last_error": self.last_error,
        }


def _candidates(client: object) -> list[str]:
    """List the endpoint's plausible Persian text generators.

    Args:
        client: An OpenAI-compatible client.

    Returns:
        Model ids with the non-generative families filtered out.
    """
    ids = sorted(model.id for model in client.models.list())  # type: ignore[attr-defined]
    return [name for name in ids if not any(skip in name.lower() for skip in _SKIP)]


def _prompt(genre: str, register: str, low: int, high: int, words: Sequence[str]) -> str:
    """Build the evaluation prompt.

    Mirrors :func:`fa_ocr_lora.diverse.generation.prompts.build_prompt` closely
    enough to be predictive, while staying fixed so scores are comparable
    across runs of this script.

    Args:
        genre: Persian genre name.
        register: Persian register name.
        low: Word-count floor.
        high: Word-count ceiling.
        words: Rare words the model must use.

    Returns:
        The prompt.
    """
    from fa_ocr_lora.dataset.common.io_utils import to_persian_digits

    return "\n".join(
        [
            f"یک متن فارسی در قالب «{genre}» بنویس.",
            "",
            f"لحن و سطح زبان: {register}.",
            f"طول متن: حدود {to_persian_digits(low)} تا {to_persian_digits(high)} واژه.",
            "",
            "این واژه‌ها را باید در متن به کار ببری، در جمله و در جای دستوری درست، "
            "و معنی‌شان را توضیح نده:",
            "، ".join(words),
            "املای این واژه‌ها را عیناً بنویس و حتی یک حرف آن را تغییر نده.",
            "",
            "قواعد:",
            "۱. فقط متن نهایی را بنویس؛ بدون مقدمه و بدون توضیح.",
            "۲. از مارک‌داون استفاده نکن و کل متن را در گیومه نگذار.",
            "۳. هیچ واژه انگلیسی، لاتین، روسی، چینی یا غیرفارسی ننویس. "
            "تمام متن باید با الفبای فارسی باشد.",
            "",
            "حالا فقط متن را بنویس:",
        ]
    )


def _score_one(
    text: str, low: int, high: int, words: Sequence[str], score: ModelScore
) -> None:
    """Measure one reply and fold it into a model's score.

    Args:
        text: The model's reply.
        low: Word-count floor.
        high: Word-count ceiling.
        words: Rare words that were requested.
        score: The accumulator, mutated in place.
    """
    from fa_ocr_lora.diverse.text_policy import (
        normalise_for_eval,
        persian_letter_ratio,
        stems,
        word_count,
    )
    from fa_ocr_lora.diverse.textqc.filters import check_foreign_script, strip_artefacts

    clean = strip_artefacts(text)
    reason = check_foreign_script(clean)
    leaked = reason is not None
    if reason:
        for part in reason.split(":", 1)[1].split(","):
            name = part.strip().split("(")[0]
            score.scripts[name] = score.scripts.get(name, 0) + 1

    ratio = persian_letter_ratio(clean)
    haystack = normalise_for_eval(clean)
    haystack_stems = stems(haystack.split())
    hits = sum(
        1
        for word in words
        if normalise_for_eval(word) in haystack or (stems([word]) & haystack_stems)
    )
    rate = hits / max(1, len(words))

    count = word_count(clean)
    within = max(1, int(low * 0.45)) <= count <= int(high * 1.55) + 2

    score.persian.append(ratio)
    score.word_rates.append(rate)
    score.foreign.append(leaked)
    score.in_length.append(within)
    score.clean.append(not leaked and ratio >= 0.80 and rate >= 0.5 and within)
    if len(score.samples) < 2:
        score.samples.append(clean[:260])


def evaluate(
    models: Sequence[str],
    keys: Sequence[str],
    *,
    base_url: str,
    temperature: float,
    repeats: int,
    max_tokens: int,
    workers: int,
) -> list[ModelScore]:
    """Run the prompt set against every model.

    Args:
        models: Model ids to score.
        keys: API keys, used round-robin so the load spreads.
        base_url: Endpoint.
        temperature: Sampling temperature, matched to the production config.
        repeats: Passes over the prompt set per model.
        max_tokens: Output ceiling.
        workers: Concurrent requests.

    Returns:
        One score per model, ranked by clean rate descending.
    """
    from openai import OpenAI

    scores = {model: ModelScore(model) for model in models}
    jobs: list[tuple[int, str, tuple[str, str, int, int, tuple[str, ...]]]] = [
        (index, model, task)
        for index, (model, task) in enumerate(
            (model, task) for model in models for _ in range(repeats) for task in _TASKS
        )
    ]
    system = (
        "تو یک نویسنده حرفه‌ای فارسی هستی. فقط متن نهایی را می‌نویسی: "
        "بدون مقدمه، بدون توضیح، بدون مارک‌داون."
    )

    def run(
        job: tuple[int, str, tuple[str, str, int, int, tuple[str, ...]]],
    ) -> tuple[str, str | None, float, int, int, tuple[str, ...], str | None]:
        """Issue one request, retrying through throttles, and return the outcome.

        Retries matter here more than they look. Without them the first models
        in the job list consume the account's rate-limit budget and every later
        model scores zero - which reads as "this model cannot write Persian"
        when it means "this script asked too fast". That misreading nearly
        disqualified three usable models on the first run.
        """
        index, model, (genre, register, low, high, words) = job
        client = OpenAI(
            base_url=base_url,
            api_key=keys[index % len(keys)],
            timeout=240,
            max_retries=0,
        )
        prompt = _prompt(genre, register, low, high, words)
        started = time.monotonic()
        last = "no attempt"
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    return model, text, time.monotonic() - started, low, high, words, None
                last = "empty reply"
            except Exception as error:  # noqa: BLE001 - a failure is a score
                status = getattr(error, "status_code", None)
                last = f"{status or 'connection'}: {str(error)[:100]}"
                if status is not None and status not in {408, 429, 500, 502, 503, 504}:
                    # A 400 or 404 will not improve with waiting.
                    break
            time.sleep(min(30.0, 3.0 * 2**attempt) + index % 3)
        return model, None, time.monotonic() - started, low, high, words, last

    with futures.ThreadPoolExecutor(workers) as pool:
        for model, text, elapsed, low, high, words, error in pool.map(run, jobs):
            score = scores[model]
            score.calls += 1
            if not text:
                score.failures += 1
                if error:
                    score.last_error = error
                    LOGGER.debug("%s failed: %s", model, error)
                continue
            score.latencies.append(elapsed)
            _score_one(text, low, high, words, score)

    return sorted(
        scores.values(), key=lambda item: (item.mean(item.clean), item.ok), reverse=True
    )


def _print_table(scores: Sequence[ModelScore]) -> None:
    """Print the ranking table.

    Args:
        scores: Scores, already ranked.
    """
    LOGGER.info("")
    LOGGER.info(
        "%-44s %4s %6s %7s %7s %6s %7s",
        "model",
        "ok",
        "med_s",
        "clean",
        "foreign",
        "fa",
        "words",
    )
    LOGGER.info("-" * 88)
    for score in scores:
        row = score.as_row()
        if not score.ok:
            LOGGER.info(
                "%-44s %4d   -- no usable reply: %s",
                score.model,
                score.ok,
                score.last_error or "unknown",
            )
            continue
        LOGGER.info(
            "%-44s %4d %6.1f %6.0f%% %6.0f%% %5.0f%% %6.0f%%",
            score.model,
            score.ok,
            row["median_latency_s"] or 0.0,
            float(row["clean_rate"]) * 100.0,
            float(row["foreign_rate"]) * 100.0,
            float(row["persian_ratio"]) * 100.0,
            float(row["word_hit_rate"]) * 100.0,
        )
    LOGGER.info("")
    LOGGER.info("clean   = would survive stage 03 entirely. Rank on this.")
    LOGGER.info("foreign = leaked a script Persian never uses. Disqualifying.")
    leaking = [s for s in scores if s.scripts]
    if leaking:
        LOGGER.info("")
        LOGGER.info("Scripts leaked:")
        for score in leaking:
            LOGGER.info("  %-42s %s", score.model, score.as_row()["scripts_leaked"])


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
        "--models", nargs="*", default=[], help="score only these model ids"
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="passes over the prompt set"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.65, help="sampling temperature"
    )
    parser.add_argument("--max-tokens", type=int, default=1200, help="output ceiling")
    parser.add_argument("--workers", type=int, default=8, help="concurrent requests")
    parser.add_argument(
        "--json", default=None, help="also write the ranking to this file"
    )
    parser.add_argument(
        "--show-samples", action="store_true", help="print two excerpts per model"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    from openai import OpenAI

    from fa_ocr_lora.config.env import ensure_dotenv_loaded
    from fa_ocr_lora.diverse.config import load_diverse_config
    from fa_ocr_lora.diverse.generation.client import load_api_keys, mask_key

    ensure_dotenv_loaded()
    config = load_diverse_config()
    keys = load_api_keys(config.generation.api_key_env)
    LOGGER.info("%d key(s): %s", len(keys), ", ".join(mask_key(key) for key in keys))

    probe = OpenAI(base_url=config.generation.base_url, api_key=keys[0])
    models = list(args.models) or _candidates(probe)
    LOGGER.info(
        "Scoring %d model(s) on %d prompt(s) x %d repeat(s), temperature %.2f",
        len(models),
        len(_TASKS),
        args.repeats,
        args.temperature,
    )

    scores = evaluate(
        models,
        keys,
        base_url=config.generation.base_url,
        temperature=args.temperature,
        repeats=args.repeats,
        max_tokens=args.max_tokens,
        workers=args.workers,
    )
    _print_table(scores)

    if args.show_samples:
        for score in scores:
            if not score.samples:
                continue
            LOGGER.info("")
            LOGGER.info("=" * 80)
            LOGGER.info("%s", score.model)
            for sample in score.samples:
                LOGGER.info("  %s", sample.replace("\n", " / ")[:240])

    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                [score.as_row() for score in scores], ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        LOGGER.info("")
        LOGGER.info("Wrote %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
