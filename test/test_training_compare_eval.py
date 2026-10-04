"""Unit tests for the base-versus-adapter comparison.

The comparison exists to answer one question - did the adapter help, and on
which dataset - so these tests pin the three ways that answer can be wrong
rather than merely unhelpful:

* **Grouping.** Letters and Persian Pixel must never be averaged into one
  number, and a source this module has never seen has to be reported under its
  own name instead of vanishing from the table.
* **The sign of the difference.** ``adapter - base`` means a negative delta is
  an improvement; a report that flipped it would read as a regression on a run
  that had gone well.
* **The wiring.** :func:`main` must score both variants over the same records
  and say which adapter produced the numbers, so the end-to-end test fakes the
  scorer instead of a model and asserts on the report it writes.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from fa_ocr_lora.training import compare_eval
from fa_ocr_lora.training.config import load_training_config
from fa_ocr_lora.training.eval import EvalOutcome
from fa_ocr_lora.training.metrics import EvalMetrics, MetricAccumulator
from fa_ocr_lora.training.sources import LETTER_CROP, LETTER_PAGE, PIXEL_PAGE

#: What the fake scorer writes: two letters rows and two Persian Pixel rows, so
#: a report that averaged the datasets together would still have numbers in it.
ROWS: list[dict[str, str]] = [
    {
        "record_id": "a",
        "source": LETTER_PAGE,
        "reference": "شماره ثبت ۱۲۳۴",
        "prediction": "",
    },
    {
        "record_id": "b",
        "source": LETTER_CROP,
        "reference": "سازمان توسعه",
        "prediction": "",
    },
    {
        "record_id": "c",
        "source": PIXEL_PAGE,
        "reference": "بسمه تعالی",
        "prediction": "",
    },
    {"record_id": "d", "source": PIXEL_PAGE, "reference": "امضا", "prediction": ""},
]


def _metrics(cer: float, *, samples: int = 10, wer: float = 0.0) -> EvalMetrics:
    """Build the metrics shape these tests compare.

    Args:
        cer: Character error rate to report.
        samples: Records the rates were computed over.
        wer: Word error rate to report.

    Returns:
        An :class:`EvalMetrics` with the digit fields empty.
    """
    return EvalMetrics(
        samples=samples,
        cer=cer,
        wer=wer,
        digit_cer=0.0,
        digit_exact=0.0,
        digit_samples=0,
        empty_predictions=0,
    )


def test_dataset_for_maps_the_three_sources() -> None:
    """The two letter streams roll up together, Persian Pixel stands alone."""
    assert compare_eval.dataset_for(LETTER_PAGE) == "letters"
    assert compare_eval.dataset_for(LETTER_CROP) == "letters"
    assert compare_eval.dataset_for(PIXEL_PAGE) == "persian_pixel"


def test_dataset_for_keeps_a_source_it_does_not_know() -> None:
    """A new source tag must still be reported rather than dropped."""
    assert compare_eval.dataset_for("handwritten_note") == "handwritten_note"


def test_read_predictions_round_trips(tmp_path: Path) -> None:
    """The predictions file is the input to every per-dataset number."""
    path = tmp_path / "report_base_predictions.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ROWS) + "\n",
        encoding="utf-8",
    )
    assert compare_eval.read_predictions(path) == ROWS


def test_read_predictions_missing_file_names_the_path(tmp_path: Path) -> None:
    """A missing scorer output must say so rather than score nothing."""
    with pytest.raises(FileNotFoundError):
        compare_eval.read_predictions(tmp_path / "absent.jsonl")


def test_aggregate_groups_by_dataset() -> None:
    """Per-dataset rates come from one accumulator per dataset."""
    grouped = compare_eval.aggregate(ROWS, compare_eval.dataset_for, 5)
    assert set(grouped) == {"letters", "persian_pixel"}
    assert grouped["letters"].samples == 2
    assert grouped["persian_pixel"].samples == 2


def test_aggregate_scores_empty_predictions_as_total_error() -> None:
    """Silence against a non-empty reference is a CER of 1.0, not 0.0."""
    grouped = compare_eval.aggregate(ROWS, compare_eval.dataset_for, 5)
    assert grouped["letters"].cer == pytest.approx(1.0)
    assert grouped["persian_pixel"].cer == pytest.approx(1.0)
    assert grouped["letters"].empty_predictions == 2


def test_aggregate_keeps_an_unknown_source_as_its_own_group() -> None:
    """Grouping by the raw tag leaves every tag visible."""
    grouped = compare_eval.aggregate(ROWS, lambda tag: tag, 5)
    assert set(grouped) == {LETTER_PAGE, LETTER_CROP, PIXEL_PAGE}
    assert grouped[LETTER_PAGE].samples == 1
    assert grouped[PIXEL_PAGE].samples == 2


def test_aggregate_is_micro_averaged_not_a_mean_of_rates() -> None:
    """A long record must outweigh a short one, as CER counts characters."""
    records = [
        {"source": LETTER_PAGE, "reference": "abcdef", "prediction": "abcdeg"},
        {"source": LETTER_CROP, "reference": "a", "prediction": "z"},
    ]
    metrics = compare_eval.aggregate(records, compare_eval.dataset_for, 5)["letters"]
    # 1 error in 6 characters, then 1 in 1: 2 errors over 7 characters.
    assert metrics.cer == pytest.approx(2 / 7)
    assert metrics.samples == 2


def test_compare_datasets_reports_a_negative_delta_as_a_win() -> None:
    """Negative means the adapter improved; positive means it regressed."""
    base = {"letters": _metrics(1.0), "persian_pixel": _metrics(0.5)}
    adapter = {"letters": _metrics(0.4), "persian_pixel": _metrics(0.6)}
    rows = {row["dataset"]: row for row in compare_eval.compare_datasets(base, adapter)}

    assert rows["letters"]["delta_cer"] == pytest.approx(-0.6)
    assert rows["letters"]["cer_change"] == pytest.approx(-0.6)
    assert rows["persian_pixel"]["delta_cer"] == pytest.approx(0.1)
    assert rows["persian_pixel"]["cer_change"] == pytest.approx(0.2)


def test_compare_datasets_handles_a_dataset_only_one_variant_saw() -> None:
    """A missing side reports no delta rather than a invented zero."""
    rows = compare_eval.compare_datasets({"letters": _metrics(1.0)}, {})
    assert len(rows) == 1
    assert rows[0]["base"] is not None
    assert rows[0]["adapter"] is None
    assert rows[0]["delta_cer"] is None
    assert rows[0]["cer_change"] is None


def test_render_names_the_datasets_and_states_the_verdict() -> None:
    """The table has to name datasets and say who won, not just print rates."""
    overall = {"base": _metrics(0.9, samples=4), "adapter": _metrics(0.5, samples=4)}
    text = compare_eval.render(
        "test",
        Path("/runs/adapter_only"),
        overall,
        {"letters": _metrics(1.0), "persian_pixel": _metrics(0.5)},
        {"letters": _metrics(0.4), "persian_pixel": _metrics(0.6)},
    )
    assert "== compare test" in text
    assert "/runs/adapter_only" in text
    assert "letters" in text
    assert "persian_pixel" in text
    assert "adapter better on 1 of 2 datasets (letters)" in text
    assert "worse on 1 (persian_pixel)" in text


def test_render_reports_nothing_when_no_dataset_is_shared() -> None:
    """An empty comparison says so instead of printing an empty verdict."""
    overall = {"base": _metrics(0.0), "adapter": _metrics(0.0)}
    text = compare_eval.render("test", Path("/adapter"), overall, {}, {})
    assert "no dataset scored by both variants" in text


def test_main_writes_a_report_for_both_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: two scoring passes, one report naming the adapter."""
    config = load_training_config("configs/lora_config.yaml")
    config = dataclasses.replace(
        config, paths=dataclasses.replace(config.paths, output_dir=tmp_path / "out")
    )
    config.paths.output_dir.mkdir(parents=True)
    config.paths.manifest_path.write_bytes(b"")  # never read: the scorer is faked
    config.paths.adapter_dir.mkdir(parents=True)
    monkeypatch.setattr(
        compare_eval, "load_training_config", lambda *_args, **_kwargs: config
    )

    def fake_score(
        config: object,
        *,
        adapter_dir: Path | None,
        split: str,
        max_samples: int | None,
        predictions_path: Path,
        verbose: bool = False,
    ) -> EvalOutcome:
        """Stand in for generation: the adapter fixes letters, not Persian Pixel."""
        accumulator = MetricAccumulator(confusion_top_k=2)
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in ROWS:
                score = dict(row)
                fixed = row["source"] in (LETTER_PAGE, LETTER_CROP)
                if adapter_dir is not None and fixed:
                    score["prediction"] = row["reference"]
                accumulator.update(score["reference"], score["prediction"])
                handle.write(json.dumps(score, ensure_ascii=False) + "\n")
        return EvalOutcome(metrics=accumulator.result(), audit=None)

    monkeypatch.setattr(compare_eval, "score_variant", fake_score)

    assert compare_eval.main(["--split", "test"]) == 0

    report = json.loads(
        (config.paths.output_dir / "compare_test.json").read_text(encoding="utf-8")
    )
    assert report["split"] == "test"
    assert report["adapter"] == str(config.paths.adapter_dir)
    assert report["records"] == {"base": 4, "adapter": 4}
    assert set(report["by_dataset"]) == {"letters", "persian_pixel"}

    letters = next(row for row in report["comparison"] if row["dataset"] == "letters")
    pixel = next(row for row in report["comparison"] if row["dataset"] == "persian_pixel")
    assert report["by_dataset"]["letters"]["adapter"]["cer"] == 0.0
    assert letters["delta_cer"] < 0
    assert pixel["delta_cer"] == 0.0
    assert report["field_audit"] == {"base": None, "adapter": None}

    for variant in ("base", "adapter"):
        assert (
            config.paths.output_dir / f"compare_test_{variant}_predictions.jsonl"
        ).is_file()


def test_main_refuses_without_an_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report cannot be produced from weights that do not exist yet."""
    config = load_training_config("configs/lora_config.yaml")
    config = dataclasses.replace(
        config, paths=dataclasses.replace(config.paths, output_dir=tmp_path / "out")
    )
    config.paths.output_dir.mkdir(parents=True)
    config.paths.manifest_path.write_bytes(b"")
    monkeypatch.setattr(
        compare_eval, "load_training_config", lambda *_args, **_kwargs: config
    )
    assert compare_eval.main(["--split", "test"]) == 1
