"""Unit tests for the field-validation config loader.

The audit is only as trustworthy as its rules, and its failure mode is silent:
a pattern that stops matching does not raise, it reports the field as absent, and
a reader of the JSON sees a clean row where the model was never checked. So the
loader has to reject at load time what would otherwise become a false clean bill
of health - unknown keys, uncompilable patterns, a pattern that matches the empty
string, an empty ``fields`` block.

The committed ``configs/field_validation.yaml`` is exercised directly, because
nothing else in the suite reaches it: without a test here, a typo in it would sit
undetected until an evaluation quietly stopped auditing a field.

The empty separator gets its own test. It is drawn 20% of the time by the corpus
generator - a letterhead that aligns values in a column separates them with
whitespace alone - so dropping it from the list would report a fifth of every
field in the corpus as malformed, for reasons with nothing to do with the model.
"""

from __future__ import annotations

import pytest

from fa_ocr_lora.config.base import ConfigError
from fa_ocr_lora.validation.config import (
    DEFAULT_FIELD_CONFIG_PATH,
    FIELD_CONFIG_PATH_ENV_VAR,
    FieldCheck,
    FieldKind,
    FieldValidationConfig,
    load_field_validation_config,
)

_MINIMAL = """
fields:
  date:
    labels: ["تاریخ"]
    pattern: '[0-9]{4}/[0-9]{2}/[0-9]{2}'
    check: jalali_date
"""


def _write(tmp_path, text):
    """Write a config file and return its path.

    Args:
        tmp_path: Pytest temporary directory.
        text: YAML body.

    Returns:
        Path to the written file.
    """
    path = tmp_path / "rules.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_committed_config_loads():
    """The shipped rules must parse, and must audit all five sensitive fields."""
    config = load_field_validation_config()
    assert config.source_path is not None
    assert config.source_path.name == "field_validation.yaml"
    assert {spec.kind for spec in config.specs} == set(FieldKind)


def test_the_committed_config_keeps_the_empty_separator():
    """20% of the corpus separates a label from its value with whitespace alone.

    Removing ``""`` from the pool does not make the audit stricter, it makes it
    wrong: every such field would read as "label present, no readable value" and
    the malformation rate would jump by twenty points.
    """
    config = load_field_validation_config()
    assert "" in config.separators, "the whitespace-only separator must stay"
    assert ":" in config.separators


def test_the_committed_config_matches_the_generators_separator_pool():
    """These are exactly the four wordings configs/vocabulary_fa.yaml draws from.

    Anything extra is an invention that loosens the audit; anything missing turns
    a well-formed field into a malformed one.
    """
    config = load_field_validation_config()
    assert set(config.separators) == {":", "ـــــ", ".....", ""}


def test_only_amount_is_found_without_a_label():
    """Every field but ``amount`` lives in running text and needs its label.

    ``amount`` is a table cell whose structure is gone by the time a flat
    transcription reaches the audit, so it is matched by its thousands separator
    instead - which no other field in the corpus has.
    """
    config = load_field_validation_config()
    unlabelled = {spec.kind for spec in config.specs if not spec.is_labelled}
    assert unlabelled == {FieldKind.AMOUNT}


def test_letter_number_labels_include_the_bare_word():
    """``شماره`` alone is a real label and a prefix of ``شماره ثبت``.

    Its presence is what makes the extractor's longest-label-first ordering
    load-bearing rather than decorative, so it is asserted here to keep the two
    facts tied together.
    """
    config = load_field_validation_config()
    spec = config.spec_for(FieldKind.LETTER_NUMBER)
    assert spec is not None
    assert "شماره" in spec.labels
    registration = config.spec_for(FieldKind.REGISTRATION_NUMBER)
    assert registration is not None
    assert any(label.startswith("شماره") for label in registration.labels)


def test_a_minimal_config_uses_the_documented_defaults(tmp_path):
    config = load_field_validation_config(_write(tmp_path, _MINIMAL))
    assert len(config.specs) == 1
    spec = config.specs[0]
    assert spec.kind is FieldKind.DATE
    assert spec.check is FieldCheck.JALALI_DATE
    assert config.separators == (":", "")
    assert config.max_gap_chars == 12


def test_defaults_survive_slots_being_enabled(tmp_path):
    """``slots=True`` replaces class-level defaults with descriptors.

    Reading ``FieldValidationConfig.separators`` off the class therefore yields a
    slot descriptor rather than a tuple, so the loader must fall back to module
    constants. This asserts the fallback is a real value.
    """
    config = load_field_validation_config(_write(tmp_path, _MINIMAL))
    assert isinstance(config.separators, tuple)
    assert all(isinstance(item, str) for item in config.separators)
    assert isinstance(config.max_gap_chars, int)


def test_the_environment_variable_is_honoured(tmp_path, monkeypatch):
    path = _write(tmp_path, _MINIMAL)
    monkeypatch.setenv(FIELD_CONFIG_PATH_ENV_VAR, str(path))
    config = load_field_validation_config()
    assert config.source_path == path


def test_an_explicit_path_beats_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(FIELD_CONFIG_PATH_ENV_VAR, "configs/does_not_exist.yaml")
    path = _write(tmp_path, _MINIMAL)
    assert load_field_validation_config(path).source_path == path


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_field_validation_config(tmp_path / "absent.yaml")


def test_an_empty_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="empty"):
        load_field_validation_config(_write(tmp_path, "\n"))


def test_an_unknown_top_level_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_field_validation_config(_write(tmp_path, _MINIMAL + "\nnope: 1\n"))


def test_an_unknown_field_key_is_rejected(tmp_path):
    body = _MINIMAL.replace(
        "    check: jalali_date", "    check: jalali_date\n    oops: 1"
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_field_validation_config(_write(tmp_path, body))


def test_an_unauditable_field_name_is_rejected(tmp_path):
    body = "fields:\n  postcode:\n    pattern: '[0-9]{5}'\n"
    with pytest.raises(ConfigError, match="not an auditable field"):
        load_field_validation_config(_write(tmp_path, body))


def test_an_unknown_check_is_rejected(tmp_path):
    body = _MINIMAL.replace("check: jalali_date", "check: luhn")
    with pytest.raises(ConfigError, match="unknown check"):
        load_field_validation_config(_write(tmp_path, body))


def test_an_empty_fields_block_is_rejected(tmp_path):
    """A config that audits nothing would report a clean run."""
    with pytest.raises(ConfigError, match="non-empty mapping"):
        load_field_validation_config(_write(tmp_path, "fields: {}\n"))


def test_an_uncompilable_pattern_is_rejected(tmp_path):
    body = "fields:\n  date:\n    pattern: '[0-9'\n"
    with pytest.raises(ConfigError, match="invalid regex"):
        load_field_validation_config(_write(tmp_path, body))


def test_a_pattern_matching_the_empty_string_is_rejected(tmp_path):
    """Otherwise every value, including no value at all, reports as valid."""
    body = "fields:\n  date:\n    pattern: '[0-9]*'\n"
    with pytest.raises(ConfigError, match="matches the empty string"):
        load_field_validation_config(_write(tmp_path, body))


@pytest.mark.parametrize("gap", [-1, 65, "twelve", True, 1.5])
def test_an_out_of_range_gap_is_rejected(tmp_path, gap):
    body = f"{_MINIMAL}\nmax_gap_chars: {gap!r}\n"
    with pytest.raises(ConfigError, match="max_gap_chars"):
        load_field_validation_config(_write(tmp_path, body))


def test_separators_must_be_strings(tmp_path):
    body = f"{_MINIMAL}\nseparators: [1, 2]\n"
    with pytest.raises(ConfigError, match="expected strings"):
        load_field_validation_config(_write(tmp_path, body))


def test_to_dict_carries_the_patterns_that_produced_a_result():
    """A field-validity rate is uninterpretable without its pattern.

    So the saved report has to embed the rules, not just reference them by path -
    the file on disk may have changed by the time anyone reads the JSON.
    """
    payload = load_field_validation_config().to_dict()
    assert set(payload) == {
        "source_path",
        "separators",
        "max_gap_chars",
        "require_field_context",
        "fields",
    }
    for kind in FieldKind:
        entry = payload["fields"][kind.value]
        assert entry["pattern"]
        assert set(entry) == {"labels", "pattern", "check", "description"}


def test_spec_for_returns_none_for_an_unaudited_field(tmp_path):
    config = load_field_validation_config(_write(tmp_path, _MINIMAL))
    assert config.spec_for(FieldKind.DATE) is not None
    assert config.spec_for(FieldKind.AMOUNT) is None


def test_the_default_path_constant_points_at_the_committed_file():
    assert DEFAULT_FIELD_CONFIG_PATH == "configs/field_validation.yaml"
    assert isinstance(load_field_validation_config(), FieldValidationConfig)
