"""Unit tests for the configuration layer: YAML strictness, ``.env``, precedence.

The pipeline's defence against a mistyped setting is that loading is strict:
an unknown key fails at startup rather than rendering 20k samples with a value
nobody chose. These tests pin that behaviour, plus the precedence chain
``CLI flag > environment > YAML > dataclass default`` that :func:`load_config`
owns the middle two links of.

Nothing here writes to ``data/``: every temporary file goes to ``tmp_path``,
and the tests that read the shipped config only read it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from fa_ocr_lora.config import (
    CONTACT_FIELDS,
    FOOTER_LAYOUT_STYLES,
    HF_FAMILIES,
    PAGE_SIDES,
    ConfigError,
    ElementsConfig,
    EnvError,
    HuggingFaceConfig,
    IntRange,
    LayoutConfig,
    PathsConfig,
    Range,
    SignatureConfig,
    StampConfig,
    SynthConfig,
    ThemeConfig,
    load_config,
    load_dotenv,
    parse_split_spec,
)
from fa_ocr_lora.config.env import parse_dotenv
from fa_ocr_lora.dataset.common.io_utils import repo_root

#: Sections a config file must carry for the loader to accept it at all.
MINIMAL_PAYLOAD = {
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
            {"name": "A4", "width_mm": 210.0, "height_mm": 297.0, "weight": 1.0}
        ],
    },
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every override this suite touches, so tests do not leak into each other.

    A developer's real ``.env`` is loaded by the pipeline, so without this a
    machine that sets ``FA_OCR_OUTPUT_DIR`` would fail the precedence tests for
    reasons that have nothing to do with the code.
    """
    for name in list(os.environ):
        if name.startswith("FA_OCR_"):
            monkeypatch.delenv(name, raising=False)


def write_config(directory: Path, payload: dict[str, object]) -> Path:
    """Write ``payload`` as a config file and return its path.

    Dumped with ``sort_keys=False`` so the file mirrors the payload's order.
    Weight pools are parsed in file order - that is what makes a seeded
    ``weighted_choice`` reproducible - so a helper that sorted keys would test a
    different pool than the one the caller wrote.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "synth_config.yaml"
    target.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return target


# --------------------------------------------------------------------------- #
# The shipped config
# --------------------------------------------------------------------------- #


def test_shipped_config_loads() -> None:
    """The committed YAML parses, which is what every other suite assumes."""
    config = load_config(apply_env=False)
    assert config.render.dpi > 0
    assert config.paths.output_dir.is_absolute()
    assert config.source_path is not None
    assert config.source_path.name == "synth_config.yaml"


def test_page_size_weights_are_positive() -> None:
    """Every page size can actually be drawn.

    A zero-weight entry is not an error the loader can see, but it is always a
    mistake: it means a page size was configured and will never be produced.
    """
    config = load_config(apply_env=False)
    assert config.render.page_sizes
    assert all(size.weight > 0 for size in config.render.page_sizes)


# --------------------------------------------------------------------------- #
# Strictness
# --------------------------------------------------------------------------- #


def test_unknown_top_level_section_is_rejected(tmp_path: Path) -> None:
    """A misspelled section name fails loudly rather than being ignored."""
    payload = dict(MINIMAL_PAYLOAD, probabilites={"header": 0.5})
    with pytest.raises(ConfigError, match="unknown top-level section"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_unknown_subdir_key_is_rejected(tmp_path: Path) -> None:
    """``paths.subdirs`` names artifact kinds, and only the known ones."""
    paths = dict(MINIMAL_PAYLOAD["paths"], subdirs={"imgs": "image"})
    payload = dict(MINIMAL_PAYLOAD, paths=paths)
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_absolute_subdir_is_rejected(tmp_path: Path) -> None:
    """An artifact tree must stay inside ``output_dir`` to be copyable in one move."""
    paths = dict(MINIMAL_PAYLOAD["paths"], subdirs={"images": "/var/tmp/images"})
    payload = dict(MINIMAL_PAYLOAD, paths=paths)
    with pytest.raises(ConfigError, match="relative name"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_missing_required_path_names_the_key(tmp_path: Path) -> None:
    """The error says which key is missing, not just that something is."""
    paths = {k: v for k, v in MINIMAL_PAYLOAD["paths"].items() if k != "texture_dir"}
    payload = dict(MINIMAL_PAYLOAD, paths=paths)
    with pytest.raises(ConfigError, match="texture_dir"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    """A wrong ``--config`` path is a config error, not a traceback."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml", apply_env=False)


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    """Malformed YAML is wrapped, so the CLI can report it like any other."""
    target = tmp_path / "broken.yaml"
    target.write_text("paths: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(target, apply_env=False)


def test_range_rejects_inverted_bounds() -> None:
    """``min`` above ``max`` would make ``draw`` silently return nonsense."""
    with pytest.raises(ConfigError, match="ascending"):
        Range.parse({"min": 2.0, "max": 1.0}, "test.range", Range(0.0, 1.0))


def test_range_rejects_an_unknown_key() -> None:
    """``{min, mx}`` is a typo that would otherwise keep the default silently."""
    with pytest.raises(ConfigError, match="unknown key"):
        Range.parse({"min": 1.0, "mx": 2.0}, "test.range", Range(0.0, 1.0))


# --------------------------------------------------------------------------- #
# The layout section
# --------------------------------------------------------------------------- #


def test_layout_defaults_survive_an_absent_section(tmp_path: Path) -> None:
    """A file with no ``layout`` still renders, on the dataclass defaults."""
    config = load_config(write_config(tmp_path, MINIMAL_PAYLOAD), apply_env=False)
    assert config.layout == LayoutConfig()
    assert config.layout.logo_sides[0] == ("right", 52.0)


def test_layout_rejects_an_unknown_key(tmp_path: Path) -> None:
    payload = {**MINIMAL_PAYLOAD, "layout": {"logo_mm_range": {"min": 1, "max": 2}}}
    with pytest.raises(ConfigError, match="layout: unknown key"):
        load_config(write_config(tmp_path, payload), apply_env=False)


@pytest.mark.parametrize("field_name", ["logo_sides", "signature_sides"])
def test_layout_rejects_a_side_that_is_not_a_margin(field_name: str) -> None:
    """A typo here would reach the CSS as ``logo-rihgt`` and place nothing."""
    with pytest.raises(ConfigError, match="unknown value"):
        LayoutConfig(**{field_name: (("rihgt", 1.0),)})  # type: ignore[arg-type]


def test_layout_rejects_an_unknown_footer_arrangement() -> None:
    with pytest.raises(ConfigError, match="footer_layouts"):
        LayoutConfig(footer_layouts=(("centred", 1.0),))


def test_layout_rejects_an_unknown_contact_field() -> None:
    with pytest.raises(ConfigError, match="contact_orders"):
        LayoutConfig(contact_orders=(("address", "fax"),))


def test_layout_rejects_a_repeated_contact_field() -> None:
    """The footer would print the field twice rather than fail visibly."""
    with pytest.raises(ConfigError, match="repeats a field"):
        LayoutConfig(contact_orders=(("phone", "address", "phone"),))


@pytest.mark.parametrize(
    "overrides",
    [
        {"logo_sides": ()},
        {"signature_sides": ()},
        {"footer_layouts": ()},
        {"contact_orders": ()},
        {"contact_orders": ((),)},
        {"logo_sides": (("right", -1.0),)},
        {"footer_layouts": (("inline", 0.0),)},
    ],
)
def test_layout_rejects_a_pool_nothing_can_be_drawn_from(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ConfigError):
        LayoutConfig(**overrides)  # type: ignore[arg-type]


def test_layout_accepts_both_weight_spellings(tmp_path: Path) -> None:
    """A mapping and a list of ``{name, weight}`` must load the same pool."""
    as_mapping = {**MINIMAL_PAYLOAD, "layout": {"logo_sides": {"left": 3, "right": 1}}}
    as_list = {
        **MINIMAL_PAYLOAD,
        "layout": {
            "logo_sides": [
                {"name": "left", "weight": 3},
                {"name": "right", "weight": 1},
            ]
        },
    }
    first = load_config(write_config(tmp_path / "a", as_mapping), apply_env=False)
    second = load_config(write_config(tmp_path / "b", as_list), apply_env=False)
    assert first.layout.logo_sides == second.layout.logo_sides == (
        ("left", 3.0),
        ("right", 1.0),
    )


def test_shipped_layout_covers_every_margin_and_arrangement() -> None:
    """The corpus should span the placements, not sit on one of them."""
    layout = load_config().layout
    assert {side for side, _ in layout.logo_sides} == set(PAGE_SIDES)
    assert {side for side, _ in layout.signature_sides} == set(PAGE_SIDES)
    assert {name for name, _ in layout.footer_layouts} == set(FOOTER_LAYOUT_STYLES)
    assert set().union(*layout.contact_orders) == set(CONTACT_FIELDS)


# --------------------------------------------------------------------------- #
# The signature section
# --------------------------------------------------------------------------- #


def test_signature_defaults_survive_an_absent_section(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, MINIMAL_PAYLOAD), apply_env=False)
    assert config.signature == SignatureConfig()


def test_signature_rejects_an_unknown_key(tmp_path: Path) -> None:
    payload = {**MINIMAL_PAYLOAD, "signature": {"paper_floor_px": 26}}
    with pytest.raises(ConfigError, match="signature: unknown key"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_signature_pen_inks_load_from_a_mapping(tmp_path: Path) -> None:
    """A hex key needs quoting in YAML, so this is the shape worth pinning."""
    payload = {**MINIMAL_PAYLOAD, "signature": {"pen_inks": {"#123456": 2}}}
    config = load_config(write_config(tmp_path, payload), apply_env=False)
    assert config.signature.pen_inks == (("#123456", 2.0),)


# --------------------------------------------------------------------------- #
# The stamp section
# --------------------------------------------------------------------------- #


def test_stamp_defaults_survive_an_absent_section(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, MINIMAL_PAYLOAD), apply_env=False)
    assert config.stamp == StampConfig()


def test_stamp_rejects_an_unknown_key(tmp_path: Path) -> None:
    payload = {**MINIMAL_PAYLOAD, "stamp": {"logo_scale": 0.42}}
    with pytest.raises(ConfigError, match="stamp: unknown key"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_stamp_shapes_load_from_a_mapping(tmp_path: Path) -> None:
    payload = {**MINIMAL_PAYLOAD, "stamp": {"shapes": {"oval": 3, "hexagon": 1}}}
    config = load_config(write_config(tmp_path, payload), apply_env=False)
    assert config.stamp.shapes == (("oval", 3.0), ("hexagon", 1.0))


def test_stamp_rejects_a_shape_the_renderer_cannot_draw(tmp_path: Path) -> None:
    """Caught at load, because the seal would otherwise print without a border."""
    payload = {**MINIMAL_PAYLOAD, "stamp": {"shapes": {"octagon": 1}}}
    with pytest.raises(ConfigError, match="unknown shape"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_stamp_ranges_accept_a_scalar_as_a_constant(tmp_path: Path) -> None:
    """``Range.parse`` still draws, so a pinned value cannot shift the stream."""
    payload = {**MINIMAL_PAYLOAD, "stamp": {"rotate_deg": 0, "width_mm": 36.0}}
    config = load_config(write_config(tmp_path, payload), apply_env=False)
    assert config.stamp.rotate_deg == Range(0.0, 0.0)
    assert config.stamp.width_mm == Range(36.0, 36.0)


# --------------------------------------------------------------------------- #
# The elements section
# --------------------------------------------------------------------------- #


def test_elements_defaults_survive_an_absent_section(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, MINIMAL_PAYLOAD), apply_env=False)
    assert config.elements == ElementsConfig()


def test_elements_rejects_an_unknown_subsection(tmp_path: Path) -> None:
    payload = {**MINIMAL_PAYLOAD, "elements": {"flowchart": {}}}
    with pytest.raises(ConfigError, match="elements: unknown key"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_elements_rejects_an_unknown_table_key(tmp_path: Path) -> None:
    payload = {**MINIMAL_PAYLOAD, "elements": {"table": {"columns": {"min": 3}}}}
    with pytest.raises(ConfigError, match="elements.table: unknown key"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_elements_rejects_an_unknown_chart_key(tmp_path: Path) -> None:
    payload = {**MINIMAL_PAYLOAD, "elements": {"chart": {"kinds": ["bar"]}}}
    with pytest.raises(ConfigError, match="elements.chart: unknown key"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_elements_counts_load_from_a_mapping(tmp_path: Path) -> None:
    payload = {
        **MINIMAL_PAYLOAD,
        "elements": {
            "table": {"rows": {"min": 2, "max": 3}},
            "chart": {"series_points": {"min": 3, "max": 5}},
        },
    }
    config = load_config(write_config(tmp_path, payload), apply_env=False)
    assert config.elements.table.rows == IntRange(2, 3)
    assert config.elements.chart.series_points == IntRange(3, 5)


def test_elements_counts_accept_a_scalar_as_a_constant(tmp_path: Path) -> None:
    """``IntRange.draw`` still draws, so a pinned count cannot shift the stream."""
    payload = {**MINIMAL_PAYLOAD, "elements": {"table": {"rows": 6}}}
    config = load_config(write_config(tmp_path, payload), apply_env=False)
    assert config.elements.table.rows == IntRange(6, 6)


def test_elements_rejects_a_fractional_count(tmp_path: Path) -> None:
    """A count is drawn with ``randrange``, so 4.5 is a mistake, not a request."""
    payload = {**MINIMAL_PAYLOAD, "elements": {"table": {"rows": {"min": 4.5}}}}
    with pytest.raises(ConfigError, match="expected a whole number"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_elements_rejects_a_chart_kind_the_renderer_cannot_draw(
    tmp_path: Path,
) -> None:
    """Caught at load, because the fallback would otherwise raise mid-render."""
    payload = {**MINIMAL_PAYLOAD, "elements": {"chart": {"fallback_kinds": ["donut"]}}}
    with pytest.raises(ConfigError, match="unknown kind"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_elements_rejects_a_string_as_the_kind_pool(tmp_path: Path) -> None:
    """A bare ``bar`` would otherwise iterate into the characters b, a, r."""
    payload = {**MINIMAL_PAYLOAD, "elements": {"chart": {"fallback_kinds": "bar"}}}
    with pytest.raises(ConfigError, match="expected a list of kinds"):
        load_config(write_config(tmp_path, payload), apply_env=False)


def test_elements_preserves_the_fallback_pool_order(tmp_path: Path) -> None:
    """The pool is drawn with ``rng.choice``, which indexes by position."""
    payload = {**MINIMAL_PAYLOAD, "elements": {"chart": {"fallback_kinds": [
        "pie", "bar"
    ]}}}
    config = load_config(write_config(tmp_path, payload), apply_env=False)
    assert config.elements.chart.fallback_kinds == ("pie", "bar")


# --------------------------------------------------------------------------- #
# Colour spelling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides",
    [
        {"ink_candidates": ("1A1A1A",)},
        {"surface_candidates": ("#FCFAF",)},
        {"stamp_inks": ("#GGGGGG",)},
        {"bar_text_light": "white"},
        {"bar_text_dark": "#1A1A1A1A"},
    ],
)
def test_theme_rejects_a_colour_that_is_not_hex(overrides: dict[str, object]) -> None:
    """CSS ignores a malformed colour and renders its own default instead.

    That failure is invisible in review - the page still looks like a page -
    so the spelling is checked at load rather than at use.
    """
    with pytest.raises(ConfigError, match="#rrggbb"):
        ThemeConfig(**overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The .env parser
# --------------------------------------------------------------------------- #


def test_parse_dotenv_handles_the_usual_shapes() -> None:
    """Comments, blank lines, ``export``, and both quote styles all parse."""
    values = parse_dotenv(
        "\n".join(
            [
                "# a comment",
                "",
                "PLAIN=value",
                "export EXPORTED=other",
                'QUOTED="has spaces"',
                "LITERAL='no $expansion'",
                "TRAILING=value  # inline comment",
                "HASH=#ff0000",
                "EMPTY=",
                "SPACED = padded ",
            ]
        )
    )
    assert values == {
        "PLAIN": "value",
        "EXPORTED": "other",
        "QUOTED": "has spaces",
        "LITERAL": "no $expansion",
        "TRAILING": "value",
        "HASH": "#ff0000",
        "EMPTY": "",
        "SPACED": "padded",
    }


def test_parse_dotenv_expands_escapes_in_double_quotes() -> None:
    """A double-quoted value behaves as in a shell; a single-quoted one is literal."""
    values = parse_dotenv('A="line\\nbreak"\nB=\'line\\nbreak\'\n')
    assert values["A"] == "line\nbreak"
    assert values["B"] == "line\\nbreak"


def test_malformed_dotenv_line_is_reported_not_skipped() -> None:
    """A line the author expected to take effect must not vanish silently."""
    with pytest.raises(EnvError, match="expected KEY=VALUE"):
        parse_dotenv("GOOD=1\nthis is not an assignment\n", source=".env")


def test_unterminated_quote_is_reported() -> None:
    """An unclosed quote names the line, so the file is easy to fix."""
    with pytest.raises(EnvError, match="unterminated"):
        parse_dotenv('KEY="unclosed\n', source=".env")


def test_load_dotenv_does_not_override_the_real_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``.env`` is a default for a shell, never an override of a deployment's."""
    monkeypatch.setenv("FA_OCR_SEED", "exported")
    target = tmp_path / ".env"
    target.write_text("FA_OCR_SEED=from_file\nFA_OCR_OUTPUT_DIR=/tmp/x\n", "utf-8")
    load_dotenv(target)
    assert os.environ["FA_OCR_SEED"] == "exported"
    assert os.environ["FA_OCR_OUTPUT_DIR"] == "/tmp/x"


def test_load_dotenv_tolerates_a_missing_file(tmp_path: Path) -> None:
    """Most hosts have no ``.env``; that is not an error."""
    assert load_dotenv(tmp_path / "absent") == {}


# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #


def test_env_beats_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An environment override replaces the value the file configured."""
    monkeypatch.setenv("FA_OCR_OUTPUT_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("FA_OCR_SEED", "4242")
    config = load_config(write_config(tmp_path, MINIMAL_PAYLOAD))
    assert config.paths.output_dir == tmp_path / "elsewhere"
    assert config.seed == 4242


def test_data_dir_reanchors_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``FA_OCR_DATA_DIR`` moves every path under the repo's ``data/`` tree."""
    monkeypatch.setenv("FA_OCR_DATA_DIR", str(tmp_path / "corpus"))
    config = load_config(write_config(tmp_path, MINIMAL_PAYLOAD))
    assert config.paths.manifest == tmp_path / "corpus" / "text" / "generator.csv"
    assert config.paths.texture_dir == tmp_path / "corpus" / "texture"


def test_data_dir_leaves_paths_outside_the_data_tree_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path the YAML points elsewhere is configuration, not corpus layout."""
    outside = tmp_path / "somewhere" / "logos"
    paths = dict(MINIMAL_PAYLOAD["paths"], logos_dir=str(outside))
    monkeypatch.setenv("FA_OCR_DATA_DIR", str(tmp_path / "corpus"))
    config = load_config(write_config(tmp_path, dict(MINIMAL_PAYLOAD, paths=paths)))
    assert config.paths.logos_dir == outside
    assert config.paths.texture_dir == tmp_path / "corpus" / "texture"


def test_specific_path_var_beats_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The narrower variable wins, so one directory can be moved on its own."""
    monkeypatch.setenv("FA_OCR_DATA_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("FA_OCR_TEXTURE_DIR", str(tmp_path / "paper"))
    config = load_config(write_config(tmp_path, MINIMAL_PAYLOAD))
    assert config.paths.texture_dir == tmp_path / "paper"
    assert config.paths.manifest == tmp_path / "corpus" / "text" / "generator.csv"


def test_config_path_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``FA_OCR_CONFIG`` is honoured when no path is passed, and only then."""
    payload = dict(MINIMAL_PAYLOAD, seed=777)
    monkeypatch.setenv("FA_OCR_CONFIG", str(write_config(tmp_path, payload)))
    assert load_config().seed == 777
    # An explicit path is the CLI's flag, which outranks the environment.
    other = write_config(tmp_path / "other", dict(MINIMAL_PAYLOAD, seed=888))
    assert load_config(other).seed == 888


def test_apply_env_false_loads_the_file_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The equivalence check needs a load that the environment cannot touch."""
    monkeypatch.setenv("FA_OCR_SEED", "4242")
    config = load_config(write_config(tmp_path, MINIMAL_PAYLOAD), apply_env=False)
    assert config.seed != 4242


def test_bad_env_value_fails_the_same_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overrides are applied after parsing, so they meet the same checks."""
    monkeypatch.setenv("FA_OCR_SEED", "not-a-number")
    with pytest.raises(EnvError, match="expected an integer"):
        load_config(write_config(tmp_path, MINIMAL_PAYLOAD))


# --------------------------------------------------------------------------- #
# .env.example stays in step with the code
# --------------------------------------------------------------------------- #


def test_env_example_documents_every_override() -> None:
    """Every variable the loader reads is listed in the committed example.

    A variable that exists in code but not in ``.env.example`` is a setting
    nobody will discover; this test is why the two cannot drift.
    """
    example = repo_root() / ".env.example"
    documented = set(parse_dotenv(example.read_text(encoding="utf-8")))
    from fa_ocr_lora.config.config import (
        CONFIG_PATH_ENV_VAR,
        DATA_DIR_ENV_VAR,
        HF_OUTPUT_DIR_ENV_VAR,
        HF_SPLITS_ENV_VAR,
        PATH_ENV_VARS,
        SEED_ENV_VAR,
    )

    expected = set(PATH_ENV_VARS) | {
        CONFIG_PATH_ENV_VAR,
        DATA_DIR_ENV_VAR,
        HF_OUTPUT_DIR_ENV_VAR,
        HF_SPLITS_ENV_VAR,
        SEED_ENV_VAR,
    }
    assert expected <= documented


def test_env_example_holds_no_values() -> None:
    """The example is committed, so it must never carry a real secret."""
    example = repo_root() / ".env.example"
    values = parse_dotenv(example.read_text(encoding="utf-8"))
    assert all(value == "" for value in values.values()), values


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #


def test_subdirs_place_artifacts_under_output_dir(tmp_path: Path) -> None:
    """``image/`` beside ``json/`` is what a PaddleFarmer corpus expects."""
    paths = PathsConfig(
        manifest=tmp_path / "m.csv",
        plaintext_dir=tmp_path / "plain",
        logos_dir=tmp_path / "logos",
        signatures_dir=tmp_path / "sig",
        texture_dir=tmp_path / "tex",
        output_dir=tmp_path / "out",
    ).with_subdirs({"images": "image", "json": "json"})
    assert paths.images_dir == tmp_path / "out" / "image"
    assert paths.json_dir == tmp_path / "out" / "json"
    assert paths.image_ref("a_p01.png") == "image/a_p01.png"


def test_empty_subdir_writes_into_output_dir(tmp_path: Path) -> None:
    """An empty name means "no subdirectory", which is the JSONL default."""
    paths = PathsConfig(
        manifest=tmp_path / "m.csv",
        plaintext_dir=tmp_path / "plain",
        logos_dir=tmp_path / "logos",
        signatures_dir=tmp_path / "sig",
        texture_dir=tmp_path / "tex",
        output_dir=tmp_path / "out",
        json_subdir="",
    )
    assert paths.json_dir == tmp_path / "out"


def test_with_subdirs_rejects_an_unknown_kind(tmp_path: Path) -> None:
    """A CLI ``--subdir thumbs=x`` is a typo, not a new artifact kind."""
    config: SynthConfig = load_config(apply_env=False)
    with pytest.raises(ConfigError, match="unknown artifact directory"):
        config.paths.with_subdirs({"thumbs": "thumb"})


# --------------------------------------------------------------------------- #
# HuggingFace export settings
# --------------------------------------------------------------------------- #


def test_shipped_config_exports_every_family_and_both_splits() -> None:
    """The committed YAML packs all six families and splits 0.9 / 0.1."""
    settings = load_config(apply_env=False).huggingface
    assert settings.families == HF_FAMILIES
    assert settings.splits == (("train", 0.9), ("test", 0.1))
    assert settings.group_by == "source_sample_id"


def test_splits_are_ordered_canonically_not_as_written() -> None:
    """Order comes from :data:`HF_SPLITS`, so the plan fingerprint is stable.

    Two files that request the same shares in a different order must produce
    the same deal, or a reordered YAML would silently re-split a published
    corpus.
    """
    written = HuggingFaceConfig.from_dict({"splits": {"test": 0.1, "train": 0.9}})
    assert written.splits == (("train", 0.9), ("test", 0.1))
    assert written.split_names() == ("train", "test")


def test_families_are_ordered_canonically_not_as_written() -> None:
    """``images`` leads whatever the file says, because the card follows it."""
    settings = HuggingFaceConfig.from_dict({"families": ["metadata", "images"]})
    assert settings.families == ("images", "metadata")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"splits": {"train": 0.8, "test": 0.1}}, "sum to 1.0"),
        ({"splits": {"train": 0.5, "holdout": 0.5}}, "unknown split"),
        ({"splits": {"train": 1.0, "test": 0.0}}, "must be > 0"),
        ({"splits": {}}, "at least one split"),
        ({"group_by": "page"}, "group_by must be one of"),
        ({"families": ["images", "thumbnails"]}, "unknown family"),
        ({"families": []}, "at least one family"),
        ({"families": "images"}, "list of family names"),
        ({"min_shard_bytes": -1}, "must not be negative"),
        ({"min_shard_bytes": 4 * 1024**3}, "must not exceed max_shard_bytes"),
    ],
)
def test_huggingface_rejects_a_bad_value(
    overrides: dict[str, object], message: str
) -> None:
    """Every export knob that can be wrong fails at load, not mid-export."""
    with pytest.raises(ConfigError, match=message):
        HuggingFaceConfig.from_dict(overrides)


def test_retired_split_key_names_its_replacement() -> None:
    """A config written for the single-split exporter says what to change.

    ``split: train`` is not a typo, it is a file that was correct one version
    ago, so "unknown key" would be an unhelpful thing to tell its author.
    """
    with pytest.raises(ConfigError, match="splits:"):
        HuggingFaceConfig.from_dict({"split": "train"})


def test_family_dir_is_the_loadable_config_name(tmp_path: Path) -> None:
    """A directory under ``data/`` is what the Hub reads as a config name."""
    settings = HuggingFaceConfig.from_dict({"output_dir": str(tmp_path / "hf")})
    assert settings.family_dir("images") == tmp_path / "hf" / "data" / "images"
    assert settings.shard_name("test", 2, 7) == "test-00002-of-00007.parquet"
    assert settings.staging_name("test", 2) == "test-00002.parquet"


def test_split_spec_parses_either_order() -> None:
    """``--hf-splits`` and ``FA_OCR_HF_SPLITS`` share one parser."""
    assert parse_split_spec("test=0.25,train=0.75") == (
        ("train", 0.75),
        ("test", 0.25),
    )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("train", "name=fraction"),
        ("train=most", "not a number"),
        ("train=0.5,test=0.4", "sum to 1.0"),
        (" , ", "no splits given"),
    ],
)
def test_split_spec_rejects_bad_syntax(spec: str, message: str) -> None:
    """A malformed override fails at startup with the offending text quoted."""
    with pytest.raises(ConfigError, match=message):
        parse_split_spec(spec)


def test_hf_splits_env_beats_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment overlays the file, and hits the same validators."""
    payload = dict(
        MINIMAL_PAYLOAD, huggingface={"splits": {"train": 0.9, "test": 0.1}}
    )
    path = write_config(tmp_path, payload)
    monkeypatch.setenv("FA_OCR_HF_SPLITS", "train=0.8,validation=0.1,test=0.1")
    config = load_config(path)
    assert config.huggingface.split_names() == ("train", "validation", "test")

    monkeypatch.setenv("FA_OCR_HF_SPLITS", "train=0.8,test=0.1")
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(path)


def test_hf_output_dir_and_splits_overlay_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting one export variable must not discard the other's override."""
    path = write_config(tmp_path, dict(MINIMAL_PAYLOAD))
    monkeypatch.setenv("FA_OCR_HF_OUTPUT_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("FA_OCR_HF_SPLITS", "train=1.0")
    settings = load_config(path).huggingface
    assert settings.output_dir == tmp_path / "elsewhere"
    assert settings.split_names() == ("train",)
