"""Unit tests for the evaluation metrics.

Every expected value here is computed by hand in the test, not captured from a run
of the code under test. That is the whole point: a metric that agrees with itself
tells you nothing, and the numbers these produce are what will decide whether the
adapter improved anything.

Two behaviours are load-bearing beyond plain CER:

* **Corpus rates are totals over totals, never means of per-record rates.** A
  three-character metadata crop must not weigh as much as a 700-character page.
* **Digits are scored separately.** The metadata block (شماره ثبت / تاریخ /
  پیوست) is CLAUDE.md 5.2's highest-priority field and is almost all digits, and a
  page-wide CER of 0.03 can hide every registration number being wrong.
"""

from __future__ import annotations

from fa_ocr_lora.training.metrics import (
    DIGITS,
    MetricAccumulator,
    character_error_rate,
    digits_only,
    normalize_text,
    word_error_rate,
)


def test_normalize_collapses_whitespace_only():
    assert normalize_text("  دو   کلمه \n سه ") == "دو کلمه سه"


def test_normalize_preserves_characters_that_render_alike():
    """Arabic and Persian yeh must stay distinct.

    Folding them would make the metric blind to a substitution the adapter is
    specifically meant to fix, and the resulting CER would improve without the
    model improving.
    """
    arabic_yeh, persian_yeh = "ي", "ی"
    assert normalize_text(arabic_yeh) != normalize_text(persian_yeh)
    zwnj = "‌"
    assert zwnj in normalize_text(f"می{zwnj}رود")


def test_digits_only_spans_all_three_numeral_systems():
    assert digits_only("شماره ۱۲۳ / ٤٥ / 67 پیوست") == "۱۲۳٤٥67"
    assert digits_only("بدون رقم") == ""
    assert len(DIGITS) == 30  # Persian + Arabic-Indic + Latin, ten each


def test_character_error_rate_counts_the_three_edit_kinds():
    # substitution
    assert character_error_rate("abc", "abd") == (1, 3)
    # deletion
    assert character_error_rate("abc", "ac") == (1, 3)
    # insertion
    assert character_error_rate("abc", "abxc") == (1, 3)
    # nothing right at all: three substitutions over a length of three
    assert character_error_rate("abc", "xyz") == (3, 3)
    # an empty hypothesis costs exactly the reference length
    assert character_error_rate("abcd", "") == (4, 4)


def test_word_error_rate_operates_on_whitespace_tokens():
    assert word_error_rate("یک دو سه", "یک دو سه") == (0, 3)
    assert word_error_rate("یک دو سه", "یک دو") == (1, 3)
    assert word_error_rate("یک دو سه", "یک ده سه") == (1, 3)


def test_accumulator_pools_errors_rather_than_averaging_rates():
    """A short record must not outweigh a long one.

    Averaging per-record rates would give these two samples (1/1 and 1/100) a CER
    of 0.505; pooling gives 2/101. The second is the number that describes the
    corpus.
    """
    accumulator = MetricAccumulator()
    accumulator.update("a", "b")
    accumulator.update("x" * 100, "y" + "x" * 99)
    result = accumulator.result()
    assert result.samples == 2
    assert result.cer == (1 + 1) / (1 + 100)


def test_accumulator_scores_digits_in_isolation():
    accumulator = MetricAccumulator()
    # Every letter right, the registration number wrong in one digit.
    accumulator.update("شماره ۱۲۳۴", "شماره ۱۲۳۵")
    # No digits at all: must not enter the digit denominators.
    accumulator.update("بدون رقم", "بدون رقم")
    result = accumulator.result()
    assert result.digit_samples == 1
    assert result.digit_cer == 1 / 4
    assert result.digit_exact == 0.0
    # The page-wide rate is far kinder than the digit rate, which is the reason
    # both are reported.
    assert result.cer < result.digit_cer


def test_accumulator_reports_digit_exact_match_over_digit_records_only():
    accumulator = MetricAccumulator()
    accumulator.update("شماره ۱۲۳", "شماره ۱۲۳")  # digits exactly right
    accumulator.update("تاریخ ۱۴۰۳", "تاریخ ۱۴۰۴")  # digits wrong
    accumulator.update("بدون رقم", "چیز دیگر")  # no digits: excluded
    result = accumulator.result()
    assert result.digit_samples == 2
    assert result.digit_exact == 0.5


def test_accumulator_counts_empty_predictions_separately():
    """Silence flatters CER and must be visible on its own.

    A model that has learned to answer nothing scores well on short references, so
    the count is reported rather than left to be inferred from the rate.
    """
    accumulator = MetricAccumulator()
    accumulator.update("متن", "")
    accumulator.update("متن", "متن")
    result = accumulator.result()
    assert result.empty_predictions == 1
    assert result.samples == 2


def test_confusions_report_aligned_substitutions_most_frequent_first():
    accumulator = MetricAccumulator(confusion_top_k=2)
    for _ in range(3):
        accumulator.update("۱۲۳", "۱۲۴")  # ۳ -> ۴, three times
    accumulator.update("۵۶", "۵۷")  # ۶ -> ۷, once
    result = accumulator.result()
    assert result.confusions[0] == (("۳", "۴"), 3)
    assert len(result.confusions) <= 2


def test_confusions_ignore_unequal_length_replacements():
    """A reflow is not a substitution.

    Reading character pairs out of a ``replace`` opcode that spans different
    lengths on each side would invent confusions that never happened, and those are
    exactly the rows a reader would act on.
    """
    accumulator = MetricAccumulator()
    accumulator.update("ab", "xyz")
    assert accumulator.result().confusions == ()


def test_empty_accumulator_produces_zeros_not_a_division_error():
    result = MetricAccumulator().result()
    assert result.samples == 0
    assert result.cer == 0.0
    assert result.digit_cer == 0.0
    assert result.digit_exact == 0.0


def test_metrics_serialise_to_json_safe_values():
    accumulator = MetricAccumulator()
    accumulator.update("۱۲۳", "۱۲۴")
    payload = accumulator.result().to_dict()
    assert set(payload) >= {"samples", "cer", "wer", "digit_cer", "confusions"}
    # Confusion keys are tuples, which json cannot encode; they must arrive as
    # lists or the whole eval report fails to write at the very end of a run.
    assert isinstance(payload["confusions"], list)
    assert all(isinstance(pair, list) for pair, _ in payload["confusions"])
