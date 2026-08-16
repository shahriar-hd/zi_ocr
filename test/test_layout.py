"""Unit tests for the layout stage: theme, letterhead, metadata, paper, marks.

These tests never launch a browser. Everything under ``layout/`` is a pure
function of (letter data, config, seeded RNG), so the properties that matter -
determinism, contrast, RTL placement, and the "no bbox for absent content"
rule - are all checkable in-process.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from fa_ocr_lora.dataset.common.config import FontSpec, ThemeConfig, load_synth_config
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
    backdrop_css,
    background_css,
    choose_texture,
)
from fa_ocr_lora.dataset.synth.layout.theme import (
    BarStyle,
    build_theme,
    contrast_ratio,
    hex_to_rgb,
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
    return load_synth_config().theme


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
    """No seed may produce an unreadable - and so untrainable - page."""
    for brand in BRAND_COLOURS:
        for index in SWEEP:
            theme = build_theme(brand, theme_config, _rng("theme", index))
            palette = theme.palette
            floor = theme_config.min_contrast_ratio
            assert contrast_ratio(palette.ink, palette.surface) >= floor
            assert contrast_ratio(palette.accent, palette.surface) >= floor
            assert contrast_ratio(palette.bar_text, palette.brand) >= floor


def test_theme_is_deterministic(theme_config: ThemeConfig) -> None:
    first = build_theme("#1F3A5F", theme_config, _rng("theme", 7))
    second = build_theme("#1F3A5F", theme_config, _rng("theme", 7))
    assert first == second


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


# --------------------------------------------------------------------------
# texture.py
# --------------------------------------------------------------------------


def test_texture_is_disabled_by_a_zero_probability() -> None:
    config = load_synth_config()
    chosen = choose_texture(config.paths.texture_dir, 0.0, _rng("texture", 1))
    assert chosen is None
    assert backdrop_css(None) == ""
    assert "background-color:#FFFFFF" in background_css(None, "#FFFFFF")


def test_texture_choice_is_deterministic_and_rate_faithful() -> None:
    config = load_synth_config()
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


def test_texture_backdrop_inlines_the_scan_with_its_own_mime() -> None:
    config = load_synth_config()
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
    css = backdrop_css(chosen)
    assert f"data:{chosen.mime};base64," in css
    # The veil keeps ink legible against the grain; without it the scan prints
    # at full strength.
    assert f"rgba(255,255,255,{chosen.wash})" in css
    assert chosen.name in chosen.as_content()["name"]


# --------------------------------------------------------------------------
# signature.py
# --------------------------------------------------------------------------


def test_unsigned_row_yields_no_signature() -> None:
    assert build_signature(None, _rng("signature", 1)) is None


def test_zero_probability_suppresses_a_present_signature() -> None:
    config = load_synth_config()
    scans = sorted(config.paths.signatures_dir.rglob("*.tif"))[:1]
    if not scans:
        pytest.skip("UTSig corpus not present")
    assert build_signature(scans[0], _rng("signature", 2), probability=0.0) is None


def test_signature_is_deterministic_and_transparent() -> None:
    config = load_synth_config()
    scans = sorted(config.paths.signatures_dir.rglob("*.tif"))[:1]
    if not scans:
        pytest.skip("UTSig corpus not present")
    first = build_signature(scans[0], _rng("signature", 3))
    second = build_signature(scans[0], _rng("signature", 3))
    assert first is not None and first == second
    assert first.data_uri.startswith("data:image/png;base64,")
    assert first.height_mm > 0
    assert -20.0 <= first.rotate_deg <= 20.0


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
