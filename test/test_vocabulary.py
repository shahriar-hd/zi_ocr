"""Unit tests for the Persian content pools: drift, whitespace, strictness.

Two things are being defended here, and they fail in different ways.

* **Drift.** Every pool has a compiled-in default *and* an entry in
  ``configs/vocabulary_fa.yaml``. If the two disagree, a developer editing the
  file gets one corpus and a test using the dataclass default gets another, and
  nothing complains. :func:`test_shipped_file_matches_dataclass_defaults` is the
  one test that makes that impossible to miss.
* **Whitespace and emptiness.** ``"موضوع: "`` needs its trailing space because
  the subject is concatenated onto it, and ``""`` is a real entry meaning
  "print no prefix". A parser that strips or drops either produces Persian that
  reads subtly wrong rather than output that crashes, so the parsers are pinned
  directly.

Nothing here writes to ``data/``: the shipped file is only read.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fa_ocr_lora.config import (
    CHART_KINDS,
    VOCABULARY_PATH_ENV_VAR,
    ChartVocabulary,
    ConfigError,
    HeaderVocabulary,
    LetterVocabulary,
    MetadataVocabulary,
    StampVocabulary,
    TableVocabulary,
    Vocabulary,
    load_config,
    load_vocabulary,
)
from fa_ocr_lora.dataset.common.io_utils import repo_root

#: Section name to dataclass, for the drift test and the strictness sweep.
SECTIONS = {
    "header": HeaderVocabulary,
    "letter": LetterVocabulary,
    "metadata_block": MetadataVocabulary,
    "stamp": StampVocabulary,
}


def write_vocabulary(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Write a vocabulary file to ``tmp_path`` and return its path."""
    target = tmp_path / "vocabulary.yaml"
    target.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return target


# --------------------------------------------------------------------------- #
# Drift between the shipped file and the compiled-in defaults
# --------------------------------------------------------------------------- #


def test_shipped_file_matches_dataclass_defaults() -> None:
    """``configs/vocabulary_fa.yaml`` must equal the compiled-in defaults.

    Field by field rather than by comparing the two objects, so a failure names
    the pool that drifted instead of dumping both vocabularies.
    """
    loaded = load_vocabulary()
    default = Vocabulary()
    for section in ("header", "letter", "metadata_block", "stamp"):
        shipped = getattr(loaded, section)
        baseline = getattr(default, section)
        for name in type(baseline).__dataclass_fields__:
            assert getattr(shipped, name) == getattr(baseline, name), (
                f"vocabulary.{section}.{name} in the YAML file differs from the "
                "dataclass default"
            )
    for element in ("table", "chart"):
        shipped = getattr(loaded.elements, element)
        baseline = getattr(default.elements, element)
        for name in type(baseline).__dataclass_fields__:
            assert getattr(shipped, name) == getattr(baseline, name), (
                f"vocabulary.elements.{element}.{name} in the YAML file differs "
                "from the dataclass default"
            )


def test_shipped_file_records_its_own_path() -> None:
    """The loaded vocabulary knows which file it came from."""
    loaded = load_vocabulary()
    assert loaded.source_path is not None
    assert loaded.source_path == repo_root() / "configs" / "vocabulary_fa.yaml"


def test_dataclass_defaults_record_no_path() -> None:
    """The compiled-in defaults are not attributed to a file."""
    assert Vocabulary().source_path is None


def test_vocabulary_is_hashable() -> None:
    """Every config section is hashable; the vocabulary is no exception.

    The pools are stored as tuples of pairs rather than dicts precisely so this
    holds - a ``dict`` field would make the frozen dataclass unhashable and
    break any caller that keys a cache on the config.
    """
    assert hash(load_vocabulary()) == hash(load_vocabulary())


def test_config_carries_the_vocabulary() -> None:
    """``load_config()`` resolves the ``vocabulary:`` pointer for the pipeline."""
    config = load_config()
    assert config.vocabulary.source_path is not None
    assert config.vocabulary.header.slogans


# --------------------------------------------------------------------------- #
# Whitespace and the empty phrase
# --------------------------------------------------------------------------- #


def test_trailing_space_in_a_prefix_survives() -> None:
    """``"موضوع: "`` keeps its trailing space, because the subject follows it."""
    prefixes = dict(load_vocabulary().letter.subject_prefixes)
    assert any(phrase.endswith(" ") for phrase in prefixes), (
        "every subject prefix lost its trailing space, so subjects will render "
        "as موضوع:متن"
    )


def test_empty_phrase_is_a_real_entry(tmp_path: Path) -> None:
    """``""`` means "print no prefix", so the parser must keep it."""
    path = write_vocabulary(
        tmp_path,
        {"header": {"name_prefixes": {"": 40.0, "شرکت ": 60.0}}},
    )
    prefixes = load_vocabulary(path).header.name_prefixes
    assert prefixes == (("", 40.0), ("شرکت ", 60.0))


def test_null_key_is_rejected_with_advice(tmp_path: Path) -> None:
    """``null:`` is the wrong way to write the empty phrase, and is caught.

    An author reaching for "no prefix" may write ``null`` or ``~`` rather than
    ``""``. YAML hands both to us as ``None``, and ``str(None)`` is the literal
    word ``None`` - which would be printed onto Persian letterhead. The error
    says what to write instead.
    """
    path = tmp_path / "vocabulary.yaml"
    path.write_text(
        "header:\n  name_prefixes:\n    null: 40.0\n    'شرکت ': 60.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="must be written"):
        load_vocabulary(path)


def test_pool_order_is_preserved(tmp_path: Path) -> None:
    """File order is load order, because it is what the RNG indexes.

    ``rng.choice`` picks by position, so reordering a pool re-points every seed
    at a different phrase. The parser must not sort.
    """
    path = write_vocabulary(
        tmp_path, {"header": {"slogans": ["سه", "یک", "دو"]}}
    )
    assert load_vocabulary(path).header.slogans == ("سه", "یک", "دو")


def test_bare_string_is_not_a_one_entry_pool(tmp_path: Path) -> None:
    """A string where a list belongs is an error, not a pool of characters."""
    path = write_vocabulary(tmp_path, {"header": {"slogans": "یک شعار"}})
    with pytest.raises(ConfigError, match="expected a list of phrases"):
        load_vocabulary(path)


# --------------------------------------------------------------------------- #
# Strictness
# --------------------------------------------------------------------------- #


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    """A missing vocabulary fails loudly rather than falling back silently."""
    with pytest.raises(ConfigError, match="not found"):
        load_vocabulary(tmp_path / "absent.yaml")


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    """A mistyped section name fails at startup."""
    path = write_vocabulary(tmp_path, {"headers": {"slogans": ["یک"]}})
    with pytest.raises(ConfigError, match="unknown key"):
        load_vocabulary(path)


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_unknown_key_in_each_section_is_rejected(
    tmp_path: Path, section: str
) -> None:
    """Every section rejects a key it does not define."""
    path = write_vocabulary(tmp_path, {section: {"not_a_pool": ["یک"]}})
    with pytest.raises(ConfigError, match="unknown key"):
        load_vocabulary(path)


def test_empty_pool_is_rejected(tmp_path: Path) -> None:
    """A truncated file fails at startup, not at sample 400."""
    path = write_vocabulary(tmp_path, {"header": {"slogans": []}})
    with pytest.raises(ConfigError, match="at least one phrase"):
        load_vocabulary(path)


def test_zero_total_weight_is_rejected(tmp_path: Path) -> None:
    """A pool whose weights sum to zero cannot be sampled from."""
    path = write_vocabulary(
        tmp_path, {"header": {"name_prefixes": {"شرکت ": 0.0, "مؤسسه ": 0.0}}}
    )
    with pytest.raises(ConfigError, match="more than zero"):
        load_vocabulary(path)


def test_non_numeric_weight_is_rejected(tmp_path: Path) -> None:
    """A weight that is not a number is reported, not coerced."""
    path = write_vocabulary(
        tmp_path, {"header": {"name_prefixes": {"شرکت ": "بسیار"}}}
    )
    with pytest.raises(ConfigError, match="must be a number"):
        load_vocabulary(path)


def test_page_template_without_placeholder_is_rejected(tmp_path: Path) -> None:
    """A page-number wording that names no ``{page}`` would print a literal."""
    path = write_vocabulary(tmp_path, {"header": {"page_templates": ["صفحه"]}})
    with pytest.raises(ConfigError, match=r"\{page\}"):
        load_vocabulary(path)


def test_label_weights_shorter_than_the_widest_field_is_rejected(
    tmp_path: Path,
) -> None:
    """Weights are applied positionally, so a short list would drop a wording."""
    path = write_vocabulary(
        tmp_path, {"metadata_block": {"label_weights": [60.0, 18.0]}}
    )
    with pytest.raises(ConfigError, match="one weight per wording"):
        load_vocabulary(path)


def test_negative_label_weight_is_rejected(tmp_path: Path) -> None:
    """A negative weight is a typo, not a way to suppress a wording."""
    path = write_vocabulary(
        tmp_path,
        {"metadata_block": {"label_weights": [60.0, -18.0, 14.0, 8.0]}},
    )
    with pytest.raises(ConfigError, match="must not be negative"):
        load_vocabulary(path)


def test_months_must_name_twelve(tmp_path: Path) -> None:
    """A short month list would silently never print the last months."""
    path = write_vocabulary(
        tmp_path, {"elements": {"table": {"months": ["فروردین", "اردیبهشت"]}}}
    )
    with pytest.raises(ConfigError, match="twelve"):
        load_vocabulary(path)


def test_empty_header_row_is_rejected(tmp_path: Path) -> None:
    """A header row's width is the table's column count, so it cannot be zero.

    From YAML the phrase parser catches it first; the dataclass guard is checked
    separately below, since it is what protects a caller constructing the
    section in Python.
    """
    path = write_vocabulary(
        tmp_path, {"elements": {"table": {"header_sets": [[]]}}}
    )
    with pytest.raises(ConfigError, match="at least one phrase"):
        load_vocabulary(path)
    with pytest.raises(ConfigError, match="at least one column"):
        TableVocabulary(header_sets=((),))


def test_single_category_series_pool_is_rejected(tmp_path: Path) -> None:
    """A one-category chart is not a chart."""
    path = write_vocabulary(
        tmp_path,
        {
            "elements": {
                "chart": {
                    "series_pools": [{"unit": "تعداد", "labels": ["فروردین"]}]
                }
            }
        },
    )
    with pytest.raises(ConfigError, match="at least two categories"):
        load_vocabulary(path)


def test_missing_chart_caption_is_rejected(tmp_path: Path) -> None:
    """The renderer indexes captions by kind, so every kind needs an entry."""
    path = write_vocabulary(
        tmp_path,
        {"elements": {"chart": {"captions": {"bar": "نمودار", "line": "روند"}}}},
    )
    with pytest.raises(ConfigError, match="missing"):
        load_vocabulary(path)


def test_unknown_chart_kind_in_captions_is_rejected(tmp_path: Path) -> None:
    """A caption for a kind the renderer cannot draw is a typo."""
    captions = {kind: f"نمودار {kind}" for kind in CHART_KINDS}
    captions["donut"] = "دونات"
    path = write_vocabulary(
        tmp_path, {"elements": {"chart": {"captions": captions}}}
    )
    with pytest.raises(ConfigError, match="unknown kind"):
        load_vocabulary(path)


def test_captionless_kind_may_be_written_empty(tmp_path: Path) -> None:
    """``""`` is how a kind opts out of a caption, and must be accepted."""
    captions = {kind: "" for kind in CHART_KINDS}
    path = write_vocabulary(
        tmp_path, {"elements": {"chart": {"captions": captions}}}
    )
    chart = load_vocabulary(path).elements.chart
    assert all(chart.caption_for(kind) == "" for kind in CHART_KINDS)


# --------------------------------------------------------------------------- #
# Lookups the generators depend on
# --------------------------------------------------------------------------- #


def test_weighted_labels_truncates_to_the_field_width() -> None:
    """A three-wording field takes the first three weights, not all four."""
    words = load_vocabulary().metadata_block
    date = words.weighted_labels("date")
    assert len(date) == 3
    assert [weight for _, weight in date] == list(words.label_weights[:3])


def test_weighted_labels_of_an_unknown_key_is_empty() -> None:
    """An unknown key returns nothing, which the caller reads as "use the key"."""
    assert load_vocabulary().metadata_block.weighted_labels("not_a_field") == ()


def test_labels_for_falls_back_to_the_given_label() -> None:
    """A contact field with no wordings renders the record's own label."""
    header = load_vocabulary().header
    assert header.labels_for("not_a_field", "تلفن") == ("تلفن",)


def test_caption_for_covers_every_renderable_kind() -> None:
    """The shipped file captions every kind the chart renderer can draw."""
    chart = load_vocabulary().elements.chart
    assert all(chart.caption_for(kind) for kind in CHART_KINDS)


def test_series_pools_carry_their_unit() -> None:
    """The unit's wording picks the value range, so it travels with the pool."""
    pools = load_vocabulary().elements.chart.series_pools
    assert all(unit and labels for unit, labels in pools)


# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #


def test_env_var_beats_the_yaml_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``FA_OCR_VOCABULARY`` overrides the ``vocabulary:`` key in the config."""
    override = write_vocabulary(tmp_path, {"header": {"slogans": ["شعار آزمایشی"]}})
    monkeypatch.setenv(VOCABULARY_PATH_ENV_VAR, str(override))
    assert load_config().vocabulary.header.slogans == ("شعار آزمایشی",)


def test_inline_mapping_is_accepted(tmp_path: Path) -> None:
    """A ``vocabulary:`` block may hold the pools directly, for tests.

    Loading from a mapping rather than a path means a test needs no second file
    on disk to pin one pool.
    """
    config_path = tmp_path / "synth.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "manifest": "data/text/generator.csv",
                    "plaintext_dir": "data/text/plaintext",
                    "logos_dir": "data/thesvg",
                    "signatures_dir": "data/UTSig",
                    "texture_dir": "data/texture",
                    "output_dir": "data/out",
                },
                "render": {
                    "dpi": 300,
                    "page_sizes": [
                        {
                            "name": "A4",
                            "width_mm": 210.0,
                            "height_mm": 297.0,
                            "weight": 1.0,
                        }
                    ],
                },
                "vocabulary": {"header": {"slogans": ["شعار درون‌خطی"]}},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.vocabulary.header.slogans == ("شعار درون‌خطی",)
    assert config.vocabulary.source_path is None


def test_partial_file_keeps_every_other_default(tmp_path: Path) -> None:
    """A file naming one pool inherits the rest, so an override stays small."""
    path = write_vocabulary(tmp_path, {"header": {"slogans": ["تنها شعار"]}})
    words = load_vocabulary(path)
    assert words.header.slogans == ("تنها شعار",)
    assert words.header.name_prefixes == HeaderVocabulary().name_prefixes
    assert words.letter.subject_prefixes == LetterVocabulary().subject_prefixes
    assert words.elements.table.months == TableVocabulary().months
    assert words.elements.chart.captions == ChartVocabulary().captions
    assert words.stamp.cities == StampVocabulary().cities
