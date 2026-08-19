"""Unit tests for the layout stage: theme, letterhead, metadata, paper, marks.

These tests never launch a browser. Everything under ``layout/`` is a pure
function of (letter data, config, seeded RNG), so the properties that matter -
determinism, contrast, RTL placement, and the "no bbox for absent content"
rule - are all checkable in-process.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace as dataclass_replace
from pathlib import Path

import pytest

from fa_ocr_lora.config import (
    CONTACT_FIELDS,
    METADATA_FIELDS,
    STAMP_SHAPES,
    ConfigError,
    FontSpec,
    LayoutConfig,
    Range,
    SignatureConfig,
    StampConfig,
    ThemeConfig,
    load_config,
)
from fa_ocr_lora.dataset.common.io_utils import sample_rng
from fa_ocr_lora.dataset.synth.layout.header_footer import (
    build_footer,
    build_header,
    footer_to_html,
    footer_to_markdown,
    header_to_html,
    header_to_markdown,
    logo_data_uri,
    opener_to_html,
)
from fa_ocr_lora.dataset.synth.layout.signature import build_signature
from fa_ocr_lora.dataset.synth.layout.stamp import (
    build_stamp,
    city_from_address,
    stamp_to_html,
)
from fa_ocr_lora.dataset.synth.layout.texture import (
    background_css,
    choose_texture,
    load_texture,
)
from fa_ocr_lora.dataset.synth.layout.theme import (
    BarStyle,
    build_theme,
    composite,
    contrast_ratio,
    hex_to_rgb,
    rgba,
)
from fa_ocr_lora.dataset.synth.metadata_block import (
    build_metadata_block,
    metadata_to_html,
    metadata_to_markdown,
)

#: A spread of brand colours: near-black, mid, and a pale yellow that only
#: passes the contrast floor after the theme darkens it.
BRAND_COLOURS = ("#1F3A5F", "#C7A008", "#0A0A0A", "#7ED957", "#A81E22")

#: Sample indices used wherever a test sweeps "many samples".
SWEEP = range(64)


@pytest.fixture(scope="module")
def theme_config() -> ThemeConfig:
    """Return the real theme section, so tests track the shipped config."""
    return load_config().theme


def _rng(purpose: str, index: int = 0) -> random.Random:
    """Return a seeded RNG matching the pipeline's own derivation."""
    return sample_rng(1337, purpose, index)


# --------------------------------------------------------------------------
# theme.py
# --------------------------------------------------------------------------


def test_hex_to_rgb_accepts_both_lengths() -> None:
    assert hex_to_rgb("#FFFFFF") == (255, 255, 255)
    assert hex_to_rgb("abc") == (170, 187, 204)
    with pytest.raises(ValueError):
        hex_to_rgb("#12345")


def test_contrast_ratio_bounds() -> None:
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0, abs=1e-6)
    assert contrast_ratio("#123456", "#123456") == pytest.approx(1.0, abs=1e-6)


def test_body_ink_clears_the_configured_contrast_floor(
    theme_config: ThemeConfig,
) -> None:
    """No seed may produce an unreadable - and so untrainable - page.

    The floor is measured on the *composited* ink, not its nominal hex: every
    colour is painted with an alpha so the paper grain shows through it, and a
    translucent ink is lighter than the colour it was mixed from.
    """
    for brand in BRAND_COLOURS:
        for index in SWEEP:
            theme = build_theme(brand, theme_config, _rng("theme", index))
            palette = theme.palette
            floor = theme_config.min_contrast_ratio
            surface = palette.surface
            alpha = palette.ink_alpha
            assert (
                contrast_ratio(composite(surface, palette.ink, alpha), surface) >= floor
            )
            assert (
                contrast_ratio(composite(surface, palette.accent, alpha), surface)
                >= floor
            )
            assert contrast_ratio(palette.bar_text, palette.brand) >= floor


def test_palette_paints_no_opaque_white(theme_config: ThemeConfig) -> None:
    """Light areas are an absence of ink, so the paper beneath stays visible.

    Two things have to hold for the scan stage to have anything to composite
    under: no surface may be pure white, and no fill may be fully opaque.
    """
    for brand in BRAND_COLOURS:
        for index in SWEEP:
            palette = build_theme(brand, theme_config, _rng("theme", index)).palette
            assert palette.surface.upper() != "#FFFFFF"
            for alpha in (
                palette.surface_alpha,
                palette.ink_alpha,
                palette.tint_alpha,
                palette.panel_alpha,
                palette.rule_alpha,
            ):
                assert 0.0 < alpha < 1.0
            # The sheet is a wash, not a coat: most of what a reader sees there
            # is the paper composited underneath it.
            assert palette.surface_alpha <= 0.4
            assert palette.surface_css.startswith("rgba(")
            assert palette.ink_css.startswith("rgba(")


def test_rgba_clamps_and_formats() -> None:
    assert rgba("#FFFFFF", 0.5) == "rgba(255, 255, 255, 0.500)"
    assert rgba("#000000", 2.0) == "rgba(0, 0, 0, 1.000)"
    assert rgba("#000000", -1.0) == "rgba(0, 0, 0, 0.000)"


def test_composite_bounds() -> None:
    assert composite("#FFFFFF", "#000000", 0.0) == "#FFFFFF"
    assert composite("#FFFFFF", "#000000", 1.0) == "#000000"
    # Half-strength black on white is mid grey, which is exactly why a
    # translucent ink has to be contrast-checked after compositing.
    assert composite("#FFFFFF", "#000000", 0.5) == "#808080"


def test_theme_is_deterministic(theme_config: ThemeConfig) -> None:
    first = build_theme("#1F3A5F", theme_config, _rng("theme", 7))
    second = build_theme("#1F3A5F", theme_config, _rng("theme", 7))
    assert first == second


def test_theme_draws_in_the_order_the_migration_inherited(
    theme_config: ThemeConfig,
) -> None:
    """Moving a constant into YAML must not shift the RNG stream.

    Every value the theme draws used to be a literal inside ``theme.py``. If the
    migration to ``synth_config.yaml`` changed the *order* of the draws - or
    added or skipped one - every seed would render a different page and the
    corpus would silently stop being reproducible from (config, seed).

    This replays the sequence by hand against the same ``Random``, with the
    bounds the module used before the move, and asserts the built theme agrees.
    It is the only test that would catch a reordering, since every other property
    here holds equally well for a different-but-valid stream.
    """
    theme = build_theme("#1F3A5F", theme_config, _rng("theme", 3))
    rng = _rng("theme", 3)

    # _build_palette, in order.
    surface = rng.choice(theme_config.surface_candidates)
    surface_alpha = round(rng.uniform(0.12, 0.34), 3)
    ink_alpha = round(rng.uniform(0.88, 0.97), 3)
    rng.choice(theme_config.ink_candidates)
    rng.uniform(0.28, 0.42)  # muted ink lightening
    rng.uniform(0.86, 0.94)  # brand -> band tint
    rng.uniform(0.35, 0.6)  # accent -> rule
    tint_alpha = round(rng.uniform(0.42, 0.78), 3)
    panel_alpha = round(rng.uniform(0.9, 0.99), 3)
    rule_alpha = round(rng.uniform(0.7, 0.94), 3)

    assert theme.palette.surface == surface
    assert theme.palette.surface_alpha == surface_alpha
    assert theme.palette.ink_alpha == ink_alpha
    assert theme.palette.tint_alpha == tint_alpha
    assert theme.palette.panel_alpha == panel_alpha
    assert theme.palette.rule_alpha == rule_alpha

    # build_theme, in order: two font draws, size, leading, then the scales.
    rng.random()  # body family
    rng.random()  # display family
    rng.uniform(*theme_config.body_font_pt_range)
    line_height = round(rng.uniform(*theme_config.line_height_range), 2)
    heading_scale = round(rng.uniform(1.04, 1.16), 3)
    subject_scale = round(rng.uniform(1.08, 1.24), 3)
    justify = rng.random() < 0.65

    assert theme.typography.line_height == line_height
    assert theme.typography.heading_scale == heading_scale
    assert theme.typography.subject_scale == subject_scale
    assert theme.typography.justify is justify

    rng.random()  # header bar
    rng.random()  # footer bar
    assert theme.rule_width_px == round(rng.uniform(0.8, 2.4), 2)
    assert theme.corner_radius_px == round(rng.choice((0.0, 0.0, 2.0, 4.0, 6.0)), 1)
    assert theme.table_zebra is (rng.random() < 0.55)


def test_theme_reads_its_pools_from_the_config(theme_config: ThemeConfig) -> None:
    """A retuned pool must reach the page, not sit in the file unread."""
    narrowed = dataclass_replace(
        theme_config,
        surface_candidates=("#F0F0F0",),
        ink_candidates=("#020202",),
        corner_radii_px=(9.0,),
        justify_probability=0.0,
        table_zebra_probability=1.0,
    )
    for index in SWEEP:
        theme = build_theme("#1F3A5F", narrowed, _rng("theme", index))
        assert theme.palette.surface == "#F0F0F0"
        assert theme.corner_radius_px == 9.0
        assert theme.typography.justify is False
        assert theme.table_zebra is True


def test_bar_weights_decide_the_treatment(theme_config: ThemeConfig) -> None:
    pinned = dataclass_replace(theme_config, bar_weights=(("solid", 1.0),))
    for index in range(24):
        theme = build_theme("#1F3A5F", pinned, _rng("theme", index))
        assert theme.header_bar == BarStyle.SOLID
        assert theme.footer_bar == BarStyle.SOLID


def test_an_unknown_bar_style_is_rejected() -> None:
    with pytest.raises(ConfigError, match="bar_weights"):
        ThemeConfig(bar_weights=(("hairline", 1.0),))


def test_side_stripe_cannot_be_the_only_bar_style() -> None:
    """The footer draws from the pool without it, so it would have nothing."""
    with pytest.raises(ConfigError, match="side_stripe"):
        ThemeConfig(bar_weights=(("side_stripe", 1.0),))


@pytest.mark.parametrize(
    "overrides",
    [
        {"ink_candidates": ()},
        {"surface_candidates": ()},
        {"corner_radii_px": ()},
        {"bar_weights": ()},
        {"justify_probability": 1.4},
        {"table_zebra_probability": -0.1},
        {"darken_step": 0.0},
        {"max_darken_steps": 0},
        {"muted_ink_contrast_ratio": 0.5},
    ],
)
def test_theme_config_rejects_unusable_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        ThemeConfig(**overrides)  # type: ignore[arg-type]


def test_theme_font_choices_stay_inside_the_configured_pools(
    theme_config: ThemeConfig,
) -> None:
    body_pool = {spec.name for spec in theme_config.font_families}
    display_pool = {spec.name for spec in theme_config.display_font_families}
    for index in SWEEP:
        theme = build_theme("#1F3A5F", theme_config, _rng("theme", index))
        assert theme.typography.body_family in body_pool
        # An empty display entry means "reuse the body face", so the resolved
        # family is legitimately either pool's member.
        assert theme.typography.display_family in display_pool | body_pool


def test_theme_font_size_is_optically_scaled(theme_config: ThemeConfig) -> None:
    """A B-series page must not render smaller than an IRANSansX one."""
    scales = {spec.name: spec.scale for spec in theme_config.font_families}
    low, high = theme_config.body_font_pt_range
    for index in SWEEP:
        theme = build_theme("#1F3A5F", theme_config, _rng("theme", index))
        scale = scales[theme.typography.body_family]
        assert low * scale - 0.01 <= theme.typography.body_pt <= high * scale + 0.01


def test_display_family_falls_back_to_the_body_face() -> None:
    config = ThemeConfig(
        font_families=(FontSpec("B Nazanin"),),
        display_font_families=(FontSpec(""),),
    )
    theme = build_theme("#1F3A5F", config, _rng("theme"))
    assert theme.typography.display_family == "B Nazanin"


def test_side_stripe_is_never_also_the_footer_treatment(
    theme_config: ThemeConfig,
) -> None:
    """A full-height stripe is one element; pairing it with itself doubles it."""
    for index in range(400):
        theme = build_theme("#1F3A5F", theme_config, _rng("theme", index))
        assert theme.footer_bar != BarStyle.SIDE_STRIPE
        assert theme.header_bar in BarStyle.ALL


def test_typography_css_stacks_carry_an_arabic_fallback(
    theme_config: ThemeConfig,
) -> None:
    theme = build_theme("#1F3A5F", theme_config, _rng("theme"))
    assert "Noto Naskh Arabic" in theme.typography.body_css
    assert "Noto Naskh Arabic" in theme.typography.display_css
    assert theme.typography.subject_pt > theme.typography.body_pt


# --------------------------------------------------------------------------
# header_footer.py
# --------------------------------------------------------------------------


def test_header_reads_its_logo_placement_from_the_config() -> None:
    """A retuned pool must reach the page, not sit in the YAML unread."""
    pinned = dataclass_replace(
        LayoutConfig(), logo_sides=(("center", 1.0),), logo_mm=Range(14.0, 14.0)
    )
    for index in SWEEP:
        spec = build_header(
            "زیلبر", "", None, BarStyle.RULE, _rng("header", index),
            layout=pinned, header_probability=1.0,
        )
        assert spec.logo_side == "center"
        assert spec.logo_height_mm == 14.0
        assert "logo-center" in header_to_html(spec)


def test_header_fields_do_not_depend_on_whether_a_category_is_present() -> None:
    """The category draw happens whether or not the row has a category.

    A draw made only in one branch splits the RNG stream in two: two rows with
    the same seed would then disagree on everything decided after it. Nothing is
    decided after it today, so this is a forward guarantee - it pins the property
    before the next draw is appended to this function.
    """
    for index in SWEEP:
        with_category = build_header(
            "زیلبر", "فناوری", None, BarStyle.RULE, _rng("header", index)
        )
        without = build_header("زیلبر", "", None, BarStyle.RULE, _rng("header", index))
        bare = {"show_category": False, "category": ""}
        assert dataclass_replace(with_category, **bare) == dataclass_replace(
            without, **bare
        )


def test_footer_reads_its_order_and_arrangement_from_the_config() -> None:
    fields = tuple((key, key, "x") for key in CONTACT_FIELDS)
    pinned = dataclass_replace(
        LayoutConfig(),
        contact_orders=(("website", "address"),),
        footer_layouts=(("stacked", 1.0),),
    )
    for index in SWEEP:
        spec = build_footer(
            fields, BarStyle.RULE, _rng("footer", index),
            layout=pinned, footer_probability=1.0,
        )
        assert [key for key, _, _ in spec.fields] == ["website", "address"]
        assert spec.layout == "stacked"


def test_footer_labels_can_be_switched_off() -> None:
    """``probabilities.footer_labels`` decides labelled vs bare values."""
    fields = (("phone", "تلفن", "021-12345678"),)
    for index in SWEEP:
        bare = build_footer(
            fields, BarStyle.RULE, _rng("footer", index),
            footer_probability=1.0, label_probability=0.0,
        )
        labelled = build_footer(
            fields, BarStyle.RULE, _rng("footer", index),
            footer_probability=1.0, label_probability=1.0,
        )
        assert bare.fields[0][1] == ""
        assert labelled.fields[0][1] != ""


def test_header_html_tags_every_region_it_draws() -> None:
    spec = build_header(
        "زیلبر", "فناوری اطلاعات", None, BarStyle.RULE, _rng("header", 3),
        header_probability=1.0, slogan_probability=1.0, opener_probability=1.0,
    )
    markup = header_to_html(spec)
    assert 'data-ann-type="header_bar"' in markup
    assert 'data-ann-type="header_company_name"' in markup
    assert 'data-ann-type="header_slogan"' in markup
    assert "زیلبر" in spec.company_name
    # The opener belongs above the body, not inside the letterhead.
    assert 'data-ann-type="religious_opener"' not in markup
    assert 'data-ann-type="religious_opener"' in opener_to_html(spec)


def test_disabled_header_renders_nothing() -> None:
    spec = build_header(
        "زیلبر", "", None, BarStyle.RULE, _rng("header", 1), header_probability=0.0
    )
    assert not spec.enabled
    assert header_to_html(spec) == ""
    assert spec.logo_path is None


def test_header_logo_side_and_bar_style_reach_the_css_classes() -> None:
    for index in SWEEP:
        spec = build_header(
            "زیلبر", "", None, BarStyle.SOLID, _rng("header", index),
            header_probability=1.0,
        )
        markup = header_to_html(spec)
        assert f"logo-{spec.logo_side}" in markup
        assert "header-solid" in markup
        assert spec.logo_side in ("right", "center", "left")


def test_logo_data_uri_recolours_only_the_root_fill(tmp_path: Path) -> None:
    svg = tmp_path / "mark.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" fill="#123456">'
        '<path fill="#ABCDEF" d="M0 0h10v10H0z"/></svg>',
        encoding="utf-8",
    )
    import base64

    recoloured = logo_data_uri(svg, "#FF0000")
    payload = base64.b64decode(recoloured.split(",", 1)[1]).decode("utf-8")
    assert 'fill="#FF0000"' in payload
    # The path keeps its own colour: flattening every fill would turn a
    # multi-colour mark into a blob.
    assert 'fill="#ABCDEF"' in payload


def test_logo_data_uri_survives_an_unreadable_file(tmp_path: Path) -> None:
    assert logo_data_uri(tmp_path / "missing.svg") == ""


def test_footer_prints_only_populated_contact_fields() -> None:
    fields = (("phone", "تلفن", "021-12345678"), ("website", "وبگاه", "example.ir"))
    for index in SWEEP:
        spec = build_footer(
            fields, BarStyle.RULE, _rng("footer", index),
            footer_probability=1.0, page_number_probability=1.0,
        )
        assert {key for key, _, _ in spec.fields} <= {"phone", "website"}
        markup = footer_to_html(spec)
        for key, _, _ in spec.fields:
            assert f'data-ann-key="{key}"' in markup
        assert 'data-ann-type="page_number"' in markup
        assert "{page}" in spec.page_number_template or "{page}" not in markup


def test_footer_digits_are_persian() -> None:
    spec = build_footer(
        (("phone", "تلفن", "021-12345678"),), BarStyle.RULE, _rng("footer", 2),
        footer_probability=1.0, page_number_probability=0.0,
    )
    assert not re.search(r"[0-9]", spec.fields[0][2])
    assert not spec.has_page_number
    assert 'data-ann-type="page_number"' not in footer_to_html(spec)


def test_page_number_ships_a_template_for_paginate_js() -> None:
    spec = build_footer(
        (), BarStyle.NONE, _rng("footer", 9),
        footer_probability=1.0, page_number_probability=1.0,
    )
    markup = footer_to_html(spec)
    assert "data-page-template=" in markup
    # The fallback text must already read as a letter, so a screenshot taken
    # before pagination runs is not blank.
    assert re.search(r"[۰-۹]", markup)


def test_footer_with_no_content_at_all_renders_nothing() -> None:
    spec = build_footer(
        (), BarStyle.RULE, _rng("footer", 4),
        footer_probability=1.0, page_number_probability=0.0,
    )
    assert footer_to_html(spec) == ""
    assert footer_to_markdown(spec) == ""


def test_header_and_footer_markdown_mirror_the_page() -> None:
    header = build_header(
        "زیلبر", "", None, BarStyle.RULE, _rng("header", 5),
        header_probability=1.0, slogan_probability=1.0, opener_probability=1.0,
    )
    markdown = header_to_markdown(header)
    assert header.opener in markdown
    assert f"# {header.company_name}" in markdown
    footer = build_footer(
        (("phone", "تلفن", "021-1"),), BarStyle.RULE, _rng("footer", 5),
        footer_probability=1.0, page_number_probability=0.0,
    )
    assert footer_to_markdown(footer).startswith("---")


# --------------------------------------------------------------------------
# metadata_block.py - highest-priority region per CLAUDE.md 5.2
# --------------------------------------------------------------------------


def test_every_metadata_field_gets_its_own_annotated_node() -> None:
    spec = build_metadata_block(
        "۱۰۲/۴۵", "۷۷۸", "۱۴۰۳/۰۵/۱۲", "۱۰۳۲۰۰۰۰۰۰", True, _rng("metadata", 1)
    )
    markup = metadata_to_html(spec)
    assert markup.count('data-ann-type="metadata_field"') == len(spec.fields)
    assert 'data-ann-type="metadata_block"' in markup
    for item in spec.fields:
        assert f'data-ann-id="meta.{item.key}"' in markup
        assert item.value in markup


def test_metadata_never_claims_a_box_for_an_absent_value() -> None:
    spec = build_metadata_block("", "", "۱۴۰۳/۰۵/۱۲", "", False, _rng("metadata", 2))
    keys = {item.key for item in spec.fields}
    assert "letter_number" not in keys
    assert "registration_number" not in keys
    assert "national_id" not in keys
    assert "date" in keys


def test_metadata_block_is_never_empty() -> None:
    """A record with no metadata at all still yields a renderable block.

    The attachment row always carries a value - ``ندارد`` when there is
    nothing attached - so in practice it is what keeps the block populated;
    the date fallback covers a field order that omits it.
    """
    for index in SWEEP:
        spec = build_metadata_block("", "", "", "", False, _rng("metadata", index))
        assert spec.fields
        assert all(item.value for item in spec.fields)


def test_metadata_values_are_persian_digits() -> None:
    spec = build_metadata_block(
        "102/45", "778", "1403/05/12", "10320000", True, _rng("metadata", 4)
    )
    for item in spec.fields:
        assert not re.search(r"[0-9]", item.value)


def test_metadata_takes_a_margin_and_a_centred_logo_keeps_the_right() -> None:
    for index in SWEEP:
        spec = build_metadata_block(
            "۱", "۲", "۱۴۰۳/۰۱/۰۱", "", False, _rng("metadata", index),
            logo_side="center",
        )
        assert spec.side == "right"
        assert f"metadata-{spec.side}" in metadata_to_html(spec)
    sides = {
        build_metadata_block(
            "۱", "۲", "۱۴۰۳/۰۱/۰۱", "", False, _rng("metadata", index),
            logo_side="right",
        ).side
        for index in range(200)
    }
    # Both margins must occur, or half the corpus's layout variety is lost.
    assert sides == {"right", "left"}


def test_metadata_markdown_lists_every_field() -> None:
    spec = build_metadata_block(
        "۱۰۲", "۷۷۸", "۱۴۰۳/۰۵/۱۲", "", True, _rng("metadata", 6)
    )
    markdown = metadata_to_markdown(spec)
    for item in spec.fields:
        assert item.value in markdown


def test_metadata_reads_its_field_order_from_the_config() -> None:
    """A pinned order is the order rendered, top to bottom.

    The block is CLAUDE.md 5.2's highest-priority region, so this is the test
    that a corpus built for one organisation's letterhead can actually be
    pinned to that organisation's field order.
    """
    layout = dataclass_replace(
        LayoutConfig(), metadata_orders=(("date", "letter_number"),)
    )
    for index in SWEEP:
        spec = build_metadata_block(
            "۱۰۲", "۷۷۸", "۱۴۰۳/۰۵/۱۲", "۱۰۳۲", True, _rng("metadata", index),
            layout=layout,
        )
        assert [item.key for item in spec.fields] == ["date", "letter_number"]


def test_metadata_placement_probabilities_are_switchable() -> None:
    """The border and the labels are knobs, and 0/1 must mean never/always."""
    for index in SWEEP:
        bare = build_metadata_block(
            "۱۰۲", "۷۷۸", "۱۴۰۳/۰۵/۱۲", "", True, _rng("metadata", index),
            boxed_probability=0.0,
            label_probability=0.0,
        )
        assert not bare.boxed
        assert not bare.show_labels
        assert "metadata-boxed" not in metadata_to_html(bare)
        assert all(item.label for item in bare.fields)  # labelled in the JSON

        dressed = build_metadata_block(
            "۱۰۲", "۷۷۸", "۱۴۰۳/۰۵/۱۲", "", True, _rng("metadata", index),
            boxed_probability=1.0,
            label_probability=1.0,
        )
        assert dressed.boxed
        assert dressed.show_labels


def test_metadata_can_be_pinned_to_the_logo_side() -> None:
    """``same_side_probability`` at both extremes decides the margin outright."""
    for index in SWEEP:
        always = build_metadata_block(
            "۱", "۲", "۱۴۰۳/۰۱/۰۱", "", False, _rng("metadata", index),
            logo_side="right",
            same_side_probability=1.0,
        )
        never = build_metadata_block(
            "۱", "۲", "۱۴۰۳/۰۱/۰۱", "", False, _rng("metadata", index),
            logo_side="right",
            same_side_probability=0.0,
        )
        assert always.side == "right"
        assert never.side == "left"


def test_metadata_rejects_an_unknown_field_and_a_repeated_one() -> None:
    """A typo must fail at load: the block would silently lose a field.

    ``registeration_number`` renders nothing at all rather than raising, so
    without this the corpus would ship 20k letters missing the registration
    number - exactly what the field-level eval metric exists to measure.
    """
    with pytest.raises(ConfigError, match="metadata_orders"):
        LayoutConfig(metadata_orders=(("date", "registeration_number"),))
    with pytest.raises(ConfigError, match="repeats a field"):
        LayoutConfig(metadata_orders=(("date", "letter_number", "date"),))


def test_shipped_metadata_orders_cover_every_field() -> None:
    """Every field the block can render is reachable from the shipped config."""
    orders = load_config().layout.metadata_orders
    assert set().union(*orders) == set(METADATA_FIELDS)


# --------------------------------------------------------------------------
# texture.py
# --------------------------------------------------------------------------


def test_texture_is_disabled_by_a_zero_probability() -> None:
    config = load_config()
    chosen = choose_texture(config.paths.texture_dir, 0.0, _rng("texture", 1))
    assert chosen is None
    # The sheet still paints its own translucent paper tone; the scan is what is
    # absent, and it never was a CSS layer.
    assert background_css("rgba(252, 250, 244, 0.24)") == (
        "background-color:rgba(252, 250, 244, 0.24);"
    )


def test_texture_choice_is_deterministic_and_rate_faithful() -> None:
    config = load_config()
    directory = config.paths.texture_dir
    if not directory.is_dir():
        pytest.skip("texture corpus not present")
    first = choose_texture(directory, 0.5, _rng("texture", 11))
    second = choose_texture(directory, 0.5, _rng("texture", 11))
    assert first == second
    hits = sum(
        choose_texture(directory, 0.5, _rng("texture", index)) is not None
        for index in range(400)
    )
    assert 0.4 <= hits / 400 <= 0.6


def test_texture_reports_its_true_format_and_loads_as_rgb() -> None:
    config = load_config()
    directory = config.paths.texture_dir
    if not directory.is_dir():
        pytest.skip("texture corpus not present")
    chosen = next(
        (
            texture
            for index in range(200)
            if (texture := choose_texture(directory, 1.0, _rng("texture", index)))
        ),
        None,
    )
    assert chosen is not None
    # The extensions lie - page_4.jpg is a WebP - so the format has to come from
    # the decoded header, and the provenance block has to carry that reading.
    content = chosen.as_content()
    assert content["name"] == chosen.name
    assert content["format"] == chosen.image_format
    assert content["source_size"] == [chosen.width, chosen.height]
    image = load_texture(chosen.path)
    assert image.mode == "RGB"
    assert image.size == (chosen.width, chosen.height)
    # Cached, so a batch that reuses one scan decodes it once.
    assert load_texture(chosen.path) is image


# --------------------------------------------------------------------------
# signature.py
# --------------------------------------------------------------------------


def test_unsigned_row_yields_no_signature() -> None:
    assert build_signature(None, _rng("signature", 1)) is None


def test_zero_probability_suppresses_a_present_signature() -> None:
    config = load_config()
    scans = sorted(config.paths.signatures_dir.rglob("*.tif"))[:1]
    if not scans:
        pytest.skip("UTSig corpus not present")
    assert build_signature(scans[0], _rng("signature", 2), probability=0.0) is None


def test_signature_is_deterministic_and_transparent() -> None:
    config = load_config()
    scans = sorted(config.paths.signatures_dir.rglob("*.tif"))[:1]
    if not scans:
        pytest.skip("UTSig corpus not present")
    first = build_signature(scans[0], _rng("signature", 3))
    second = build_signature(scans[0], _rng("signature", 3))
    assert first is not None and first == second
    assert first.data_uri.startswith("data:image/png;base64,")
    assert first.height_mm > 0
    assert -20.0 <= first.rotate_deg <= 20.0


def test_signature_reads_its_pen_and_geometry_from_the_config() -> None:
    """A pinned pen and pinned ranges leave the sample nothing to vary."""
    config = load_config()
    scans = sorted(config.paths.signatures_dir.rglob("*.tif"))[:1]
    if not scans:
        pytest.skip("UTSig corpus not present")
    pinned = SignatureConfig(
        pen_inks=(("#A81E22", 1.0),),
        width_mm=Range(33.0, 33.0),
        rotate_deg=Range(0.0, 0.0),
        overlap_mm=Range(2.0, 2.0),
    )
    for index in SWEEP:
        spec = build_signature(scans[0], _rng("signature", index), config=pinned)
        assert spec is not None
        assert spec.ink == "#A81E22"
        assert spec.width_mm == 33.0
        assert spec.rotate_deg == 0.0
        assert spec.overlap_mm == 2.0


def test_signature_extraction_thresholds_are_part_of_the_cache_key() -> None:
    """Two floors in one process must not share a mask.

    ``_extract_alpha`` is ``lru_cache``d, so if the thresholds were read from
    config inside it rather than passed in, the second config would silently
    get the first one's mask. A high floor cuts more of the scan away, so the
    two must differ in size or in the ink they keep.
    """
    config = load_config()
    scans = sorted(config.paths.signatures_dir.rglob("*.tif"))[:1]
    if not scans:
        pytest.skip("UTSig corpus not present")
    low = build_signature(
        scans[0], _rng("signature", 8), config=SignatureConfig(paper_floor=1)
    )
    high = build_signature(
        scans[0], _rng("signature", 8), config=SignatureConfig(paper_floor=200)
    )
    assert low is not None and high is not None
    assert low.data_uri != high.data_uri


def test_signature_config_rejects_values_that_would_erase_the_ink() -> None:
    """Each bound is the point past which the block stops being a signature."""
    with pytest.raises(ConfigError, match="paper_floor"):
        SignatureConfig(paper_floor=255)
    with pytest.raises(ConfigError, match="crop_pad"):
        SignatureConfig(crop_pad=0.9)
    with pytest.raises(ConfigError, match="max_edge_px"):
        SignatureConfig(max_edge_px=1)
    with pytest.raises(ConfigError, match="width_mm"):
        SignatureConfig(width_mm=Range(0.0, 46.0))


def test_signature_config_rejects_a_malformed_pen_colour() -> None:
    """``hex_to_rgb`` would raise mid-render on sample 400 instead."""
    with pytest.raises(ConfigError, match="#rrggbb"):
        SignatureConfig(pen_inks=(("12295C", 1.0),))
    with pytest.raises(ConfigError, match="#rrggbb"):
        SignatureConfig(pen_inks=(("#12295", 1.0),))
    with pytest.raises(ConfigError, match="pen_inks"):
        SignatureConfig(pen_inks=())


# --------------------------------------------------------------------------
# stamp.py
# --------------------------------------------------------------------------


def test_city_from_address() -> None:
    assert city_from_address("تهران، خیابان ولیعصر، پلاک ۱۲") == "تهران"
    assert city_from_address("خیابان بی‌نام") == ""


def test_stamp_rate_follows_its_probability() -> None:
    hits = sum(
        build_stamp(None, "زیلبر", "تهران", ("#1B3F94",), _rng("stamp", index))
        is not None
        for index in range(2000)
    )
    assert 0.52 <= hits / 2000 <= 0.58


def test_stamp_is_deterministic_and_self_contained() -> None:
    first = build_stamp(
        None, "زیلبر", "تهران", ("#1B3F94",), _rng("stamp", 4), probability=1.0
    )
    second = build_stamp(
        None, "زیلبر", "تهران", ("#1B3F94",), _rng("stamp", 4), probability=1.0
    )
    assert first is not None and first == second
    markup = stamp_to_html(first)
    assert stamp_to_html(second) == markup
    assert 'data-ann-type="stamp"' in markup
    # No logo means no ink filter: an empty filter reference would drop the
    # whole seal in Chromium.
    assert first.logo_uri == ""
    assert "stamp-ink-" not in markup
    assert markup.count("<svg") == 1


def test_stamp_recolours_a_brand_logo_to_a_single_ink(tmp_path: Path) -> None:
    svg = tmp_path / "brand.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<path fill="#00A3E0" d="M0 0h10v10H0z"/></svg>',
        encoding="utf-8",
    )
    spec = build_stamp(
        svg, "زیلبر", "تهران", ("#A81E22",), _rng("stamp", 5), probability=1.0
    )
    assert spec is not None
    markup = stamp_to_html(spec)
    assert "stamp-ink-" in markup
    assert 'flood-color="#A81E22"' in markup
    # A pad transfers one weight of ink or none, so the mark's own colour must
    # not reach the page.
    assert "#00A3E0" not in markup


def test_long_company_names_stay_inside_the_border() -> None:
    """Past the ring's capacity the text is cut, not overflowed."""
    name = "Managed Service For Microsoft Azure Cloud Platform"
    spec = build_stamp(
        None, name, "تهران", ("#1B3F94",), _rng("stamp", 6), probability=1.0
    )
    assert spec is not None
    assert len(spec.top_text) <= 30
    assert spec.top_text.strip() == spec.top_text


def test_stamp_shapes_cover_the_configured_pool() -> None:
    shapes = {
        spec.shape
        for index in range(600)
        if (
            spec := build_stamp(
                None, "زیلبر", "تهران", ("#1B3F94",), _rng("stamp", index),
                probability=1.0,
            )
        )
    }
    assert {"circle", "double_circle", "square"} <= shapes


def test_stamp_reads_its_shape_and_geometry_from_the_config() -> None:
    """A pinned shape and pinned ranges leave the sample nothing to vary."""
    pinned = StampConfig(
        shapes=(("hexagon", 1.0),),
        width_mm=Range(36.0, 36.0),
        rotate_deg=Range(0.0, 0.0),
        opacity=Range(0.8, 0.8),
        wobble=Range(2.0, 2.0),
        erosion=Range(-2.5, -2.5),
        stroke_width=Range(3.0, 3.0),
    )
    for index in SWEEP:
        spec = build_stamp(
            None, "زیلبر", "تهران", ("#1B3F94",), _rng("stamp", index),
            probability=1.0, config=pinned,
        )
        assert spec is not None
        assert spec.shape == "hexagon"
        # A hexagon is not in STAMP_ROUND_SHAPES, so its ring text runs straight.
        assert not spec.is_round
        assert spec.width_mm == 36.0
        assert spec.rotate_deg == 0.0
        assert spec.opacity == 0.8
        assert spec.wobble == 2.0
        assert spec.erosion == -2.5
        assert spec.stroke_width == 3.0


def test_stamp_carries_the_render_only_values_onto_the_spec(tmp_path: Path) -> None:
    """``stamp_to_html`` is handed only a spec, so the two must ride along.

    ``logo_fraction`` sizes the nested ``<image>`` and ``ink_gain`` is the
    filter's alpha slope; both are read off the spec at render time rather than
    from a config the renderer never sees.
    """
    svg = tmp_path / "brand.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<rect width="10" height="10" fill="#00A3E0"/></svg>',
        encoding="utf-8",
    )
    spec = build_stamp(
        svg, "زیلبر", "تهران", ("#1B3F94",), _rng("stamp", 5), probability=1.0,
        config=StampConfig(logo_fraction=0.5, ink_gain=2.5),
    )
    assert spec is not None
    assert spec.logo_fraction == 0.5
    assert spec.ink_gain == 2.5
    markup = stamp_to_html(spec)
    # 0.5 of the 200-unit box is a 100-unit image inset 50 units on each side.
    assert 'width="100.0"' in markup and 'x="50.0"' in markup
    assert 'slope="2.5"' in markup


def test_stamp_name_prefix_probability_is_switchable() -> None:
    """At 0 the ring is the bare name; at 1 every seal is prefixed."""
    bare = {
        spec.top_text
        for index in SWEEP
        if (
            spec := build_stamp(
                None, "زیلبر", "تهران", ("#1B3F94",), _rng("stamp", index),
                probability=1.0,
                config=StampConfig(name_prefix_probability=0.0),
            )
        )
    }
    prefixed = {
        spec.top_text
        for index in SWEEP
        if (
            spec := build_stamp(
                None, "زیلبر", "تهران", ("#1B3F94",), _rng("stamp", index),
                probability=1.0,
                config=StampConfig(name_prefix_probability=1.0),
            )
        )
    }
    assert bare == {"زیلبر"}
    assert prefixed == {"مهر زیلبر"}


def test_stamp_config_rejects_a_shape_the_renderer_cannot_draw() -> None:
    """An unknown name would draw no border and print the seal as bare text."""
    with pytest.raises(ConfigError, match="unknown shape"):
        StampConfig(shapes=(("triangle", 1.0),))
    with pytest.raises(ConfigError, match="at least one shape"):
        StampConfig(shapes=())
    with pytest.raises(ConfigError, match="weights must not be negative"):
        StampConfig(shapes=(("circle", -1.0),))


def test_stamp_config_rejects_values_that_would_break_the_mark() -> None:
    """Each bound is the point past which the seal stops reading as pressed."""
    with pytest.raises(ConfigError, match="logo_fraction"):
        StampConfig(logo_fraction=1.4)
    with pytest.raises(ConfigError, match="logo_fraction"):
        StampConfig(logo_fraction=0.0)
    with pytest.raises(ConfigError, match="ink_gain"):
        StampConfig(ink_gain=0.0)
    with pytest.raises(ConfigError, match="opacity"):
        StampConfig(opacity=Range(0.5, 1.4))
    with pytest.raises(ConfigError, match="width_mm"):
        StampConfig(width_mm=Range(0.0, 42.0))
    with pytest.raises(ConfigError, match="name_prefix_probability"):
        StampConfig(name_prefix_probability=1.5)


def test_shipped_stamp_shapes_are_all_drawable() -> None:
    """The config's pool and the renderer's vocabulary must not drift apart."""
    assert {shape for shape, _ in StampConfig().shapes} <= set(STAMP_SHAPES)
    assert {shape for shape, _ in load_config().stamp.shapes} <= set(STAMP_SHAPES)

