"""Unit tests for the evaluation report bundle.

The bundle is what a reader and a reviewer actually see, so these tests pin the
ways it can be wrong while still looking finished:

* **The sign and the direction.** ``adapter - base`` means a negative delta is an
  improvement on an error rate and a regression on a match rate, and
  :data:`~fa_ocr_lora.training.eval_report.METRIC_LABELS` is what records which is
  which. A bundle that lost that distinction would render a 15x digit-match
  improvement as if it were a 15x worsening.
* **Missing inputs.** A comparison with no field audit, no Persian Pixel rows or
  no training curve is normal - a validation-only run has the first, a tiny run
  has the second, a fresh session has the third - and must produce a complete
  bundle rather than a stack trace or a silently truncated one.
* **The absent comparison.** A missing ``compare_<split>.json`` has to raise:
  an empty bundle is indistinguishable from a finished one once it is sitting in
  Drive.
* **Charts are optional.** ``write_charts`` must degrade to "no PNGs" rather than
  taking the CSVs down with it when matplotlib is not installed, because the
  tables are the deliverable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fa_ocr_lora.training import eval_report

#: A comparison shaped exactly like the one `compare_eval.main` writes, carrying
#: the measured full-run numbers so a reader of a failure sees realistic values.
PAYLOAD: dict[str, object] = {
    "split": "test",
    "config": "configs/lora_config_rtx6000pro_16.yaml",
    "adapter": "data/training/rtx6000pro16_run/adapter_only",
    "records": {"base": 5855, "adapter": 5855},
    "overall": {
        "base": {
            "samples": 5855,
            "cer": 0.5636,
            "wer": 0.7326,
            "cer_normalized": 0.5532,
            "wer_normalized": 0.7000,
            "digit_cer": 1.5358,
            "digit_exact": 0.0404,
            "digit_samples": 3900,
            "empty_predictions": 0,
            "confusions": [["ی", "ي", 273], ["\u200c", " ", 145]],
        },
        "adapter": {
            "samples": 5855,
            "cer": 0.2154,
            "wer": 0.2701,
            "cer_normalized": 0.2144,
            "wer_normalized": 0.2600,
            "digit_cer": 0.2563,
            "digit_exact": 0.6031,
            "digit_samples": 3900,
            "empty_predictions": 0,
            "confusions": [["0", "۰", 41]],
        },
    },
    "comparison": {"letters": -0.5459, "persian_pixel": -0.0490},
    "by_dataset": {
        "letters": {
            "base": {"samples": 4813, "cer": 0.6366, "wer": 0.7926},
            "adapter": {"samples": 4813, "cer": 0.0907, "wer": 0.1133},
        },
        "persian_pixel": {
            "base": {"samples": 1042, "cer": 0.4531, "wer": 0.6497},
            "adapter": {"samples": 1042, "cer": 0.4041, "wer": 0.4865},
        },
    },
    "by_source": {
        "admin_letter_page": {
            "base": {"samples": 1466, "cer": 0.61, "wer": 0.78},
            "adapter": {"samples": 1466, "cer": 0.08, "wer": 0.10},
        }
    },
    "field_audit": {
        "base": {
            "fields": {
                "date": {
                    "reference_count": 90,
                    "recall": 0.1444,
                    "exact_rate": 0.0,
                    "digit_rate": 0.0,
                    "mean_confidence": 0.718,
                }
            }
        },
        "adapter": {
            "fields": {
                "date": {
                    "reference_count": 234,
                    "recall": 0.9573,
                    "exact_rate": 0.9231,
                    "digit_rate": 0.9231,
                    "mean_confidence": 0.992,
                }
            }
        },
    },
}


def _write_compare(directory: Path, payload: object = None) -> Path:
    """Write a comparison JSON and return its path."""
    path = directory / "compare_test.json"
    path.write_text(json.dumps(payload if payload is not None else PAYLOAD), "utf-8")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a bundle CSV back, honouring the BOM the writer emits for Excel."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_bundle_writes_every_table(tmp_path: Path) -> None:
    """The documented files all appear, and the CSVs parse."""
    bundle = tmp_path / "report_test"
    eval_report.build_bundle(
        _write_compare(tmp_path), bundle, split="test", charts=False
    )
    for name in (
        "overall.csv",
        "by_dataset.csv",
        "by_source.csv",
        "fields.csv",
        "confusions.csv",
        "summary.json",
        "report.md",
    ):
        assert (bundle / name).is_file(), f"{name} is missing from the bundle"
    assert _read_csv(bundle / "overall.csv")


def test_error_rate_delta_is_negative_when_the_adapter_wins(tmp_path: Path) -> None:
    """CER fell, so the delta must be negative - the sign is the whole report."""
    bundle = tmp_path / "report_test"
    summary = eval_report.build_bundle(
        _write_compare(tmp_path), bundle, split="test", charts=False
    )
    cer = summary["headline"]["CER"]
    assert cer["base"] == pytest.approx(0.5636)
    assert cer["adapter"] == pytest.approx(0.2154)
    assert cer["delta"] == pytest.approx(-0.3482)
    assert cer["relative"] == pytest.approx(-0.6178, abs=1e-4)
    assert cer["lower_is_better"] is True


def test_match_rate_is_flagged_as_higher_is_better(tmp_path: Path) -> None:
    """Digit exact match rose; a positive delta there is the good direction."""
    bundle = tmp_path / "report_test"
    summary = eval_report.build_bundle(
        _write_compare(tmp_path), bundle, split="test", charts=False
    )
    digits = summary["headline"]["digit exact match"]
    assert digits["delta"] == pytest.approx(0.5627)
    assert digits["lower_is_better"] is False


def test_datasets_are_never_averaged_together(tmp_path: Path) -> None:
    """Letters and Persian Pixel keep their own rows and their own deltas."""
    bundle = tmp_path / "report_test"
    eval_report.build_bundle(
        _write_compare(tmp_path), bundle, split="test", charts=False
    )
    rows = {row["dataset"]: row for row in _read_csv(bundle / "by_dataset.csv")}
    assert set(rows) == {"letters", "persian_pixel"}
    assert float(rows["letters"]["delta_cer"]) == pytest.approx(-0.5459)
    assert float(rows["persian_pixel"]["delta_cer"]) == pytest.approx(-0.0490)
    # The two must not have been collapsed into one averaged figure.
    assert rows["letters"]["adapter_cer"] != rows["persian_pixel"]["adapter_cer"]


def test_field_rates_carry_both_variants(tmp_path: Path) -> None:
    """Per-field recall is reported for base and adapter, with the difference."""
    bundle = tmp_path / "report_test"
    eval_report.build_bundle(
        _write_compare(tmp_path), bundle, split="test", charts=False
    )
    recall = [
        row
        for row in _read_csv(bundle / "fields.csv")
        if row["field"] == "date" and row["rate"] == "recall"
    ]
    assert len(recall) == 1
    assert float(recall[0]["base"]) == pytest.approx(0.1444)
    assert float(recall[0]["adapter"]) == pytest.approx(0.9573)
    assert float(recall[0]["delta"]) == pytest.approx(0.8129)


def test_counts_render_as_integers(tmp_path: Path) -> None:
    """A sample count must not read as a rate that happens to be large."""
    bundle = tmp_path / "report_test"
    eval_report.build_bundle(
        _write_compare(tmp_path), bundle, split="test", charts=False
    )
    markdown = (bundle / "report.md").read_text(encoding="utf-8")
    assert "5,855" in markdown
    assert "5855.0000" not in markdown


def test_training_curve_is_optional_and_read_when_present(tmp_path: Path) -> None:
    """A trainer state yields a curve; its absence costs nothing else."""
    bundle = tmp_path / "with_curve"
    state = tmp_path / "trainer_state.json"
    state.write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 20, "epoch": 0.02, "loss": 1.56},
                    {"step": 17374, "epoch": 2.0, "eval_loss": 0.1695},
                    {"step": 17374, "epoch": 2.0, "loss": 0.0279},
                    {"not_a_step": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    eval_report.build_bundle(
        _write_compare(tmp_path),
        bundle,
        split="test",
        training_summary=state,
        charts=False,
    )
    rows = _read_csv(bundle / "training_curve.csv")
    assert [row["step"] for row in rows] == ["20.0", "17374.0", "17374.0"]
    assert rows[0]["loss"] == "1.56"

    without = tmp_path / "no_curve"
    eval_report.build_bundle(
        _write_compare(tmp_path), without, split="test", charts=False
    )
    assert not (without / "training_curve.csv").exists()
    assert (without / "overall.csv").is_file()


def test_missing_field_audit_still_produces_a_bundle(tmp_path: Path) -> None:
    """A CER-only comparison is valid input, not an error."""
    payload = json.loads(json.dumps(PAYLOAD))
    payload["field_audit"] = {"base": None, "adapter": None}
    bundle = tmp_path / "report_test"
    eval_report.build_bundle(
        _write_compare(tmp_path, payload), bundle, split="test", charts=False
    )
    assert _read_csv(bundle / "fields.csv") == []
    assert "## Sensitive fields" not in (bundle / "report.md").read_text("utf-8")
    assert _read_csv(bundle / "overall.csv")


def test_absent_comparison_raises(tmp_path: Path) -> None:
    """An empty bundle must not be mistaken for a finished one."""
    with pytest.raises(FileNotFoundError, match="no comparison at"):
        eval_report.build_bundle(
            tmp_path / "compare_test.json", tmp_path / "out", charts=False
        )


def test_charts_degrade_without_matplotlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No matplotlib means no PNGs and no exception - the tables still matter."""
    import builtins

    real_import = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", refuse)
    written = eval_report.write_charts(tmp_path / "charts", [], [], [], [])
    assert written == []


def test_extra_evals_are_folded_into_the_summary(tmp_path: Path) -> None:
    """Named eval JSONs ride along, so one summary covers every pass."""
    extra = tmp_path / "eval_validation.json"
    extra.write_text(json.dumps({"split": "validation", "cer": 0.0502}), "utf-8")
    bundle = tmp_path / "report_test"
    summary = eval_report.build_bundle(
        _write_compare(tmp_path),
        bundle,
        split="test",
        extra_evals={"validation_adapter": extra, "absent": tmp_path / "nope.json"},
        charts=False,
    )
    assert summary["eval_runs"]["validation_adapter"]["cer"] == pytest.approx(0.0502)
    assert "absent" not in summary["eval_runs"]
