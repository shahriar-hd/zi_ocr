"""Tests for the parts of the diverse pipeline that corrupt data silently.

Not a broad test suite. Each test here guards an invariant whose violation
produces a *plausible-looking* corpus with wrong labels - the failure mode that
costs a training run rather than raising an exception. Everything that fails
loudly is left to the stages' own error paths.
"""

from __future__ import annotations

import json
import random

import pytest

from fa_ocr_lora.diverse.config import BANDS, CATEGORIES, ERAS, load_diverse_config
from fa_ocr_lora.diverse.layouts.templates import TEMPLATE_IDS, build_document
from fa_ocr_lora.diverse.packaging.run import assign_split
from fa_ocr_lora.diverse.planning.sampler import clamp_era_weights
from fa_ocr_lora.diverse.render.schema import (
    Annotation,
    AnnotationError,
    Block,
    BlockType,
    Line,
    Word,
    rectangle,
)
from fa_ocr_lora.diverse.text_policy import (
    ZWNJ,
    detect_digit_scripts,
    digit_density,
    normalise_for_eval,
    persian_letter_ratio,
    prepare_render_text,
    word_count,
)
from fa_ocr_lora.diverse.textqc.dedupe import MinHashIndex
from fa_ocr_lora.diverse.textqc.filters import check_foreign_script, strip_artefacts

# --------------------------------------------------------------------------- #
# Text policy: the label is exactly what will be drawn
# --------------------------------------------------------------------------- #


def test_arabic_letters_fold_to_persian():
    """Arabic yeh and kaf render as Persian glyphs, so they must not survive."""
    assert normalise_for_eval("يك") == "یک"
    assert prepare_render_text("مكتوب شد") == "مکتوب شد"


def test_hamza_carriers_are_not_folded():
    """A visible hamza is real label content and must be preserved."""
    for word in ("مؤسسه", "تأیید", "مسئول", "آینده"):
        assert word in prepare_render_text(word), word


def test_zwnj_is_preserved_by_default_and_removable_on_request():
    """ZWNJ changes the drawn glyphs, so dropping it must be deliberate."""
    assert ZWNJ in prepare_render_text("می‌رود")
    assert ZWNJ not in prepare_render_text("می‌رود", keep_zwnj=False)


def test_zwnj_does_not_split_a_word():
    """A compound is one word; counting it as two would misreport length."""
    assert word_count("می‌رود و کتاب‌ها") == 3


def test_digit_scripts_are_exact_and_detectable():
    """A sample's digit script is a labelled variable, not an accident."""
    assert prepare_render_text("1403", digit_script="persian") == "۱۴۰۳"
    assert prepare_render_text("۱۴۰۳", digit_script="latin") == "1403"
    assert prepare_render_text("۱۴۰۳", digit_script="arabic_indic") == "١٤٠٣"
    assert detect_digit_scripts("۱۴۰۳ و 2024") == frozenset({"persian", "latin"})


def test_unknown_digit_script_raises():
    """Defaulting a typo would make the manifest lie about its own contents."""
    with pytest.raises(ValueError, match="unknown digit script"):
        prepare_render_text("۱۴", digit_script="farsi")


def test_diacritics_are_stripped_unless_requested():
    """The diacriticised subset is labelled; the rest must be clean."""
    vocalised = "کِتابٌ"
    assert prepare_render_text(vocalised) == "کتاب"
    assert prepare_render_text(vocalised, keep_diacritics=True) == vocalised


def test_invisible_controls_never_reach_the_label():
    """A model cannot predict an invisible character, so it cannot be a label."""
    for control in ("\u200e", "\u200f", "\u202b", "\ufeff", "\u200d"):
        assert control not in prepare_render_text(f"سلام{control}دنیا")


def test_persian_ratio_ignores_digits():
    """A date line is not evidence of the wrong language."""
    assert persian_letter_ratio("۱۴۰۳/۰۵/۱۲") == 1.0
    assert persian_letter_ratio("سلام") == 1.0
    assert persian_letter_ratio("hello") == 0.0


def test_digit_density_is_a_fraction_of_non_whitespace():
    """Used to oversample numeric text, so it must not count spaces."""
    assert digit_density("۱۲۳۴") == 1.0
    assert digit_density("") == 0.0


def test_normalisation_is_idempotent():
    """text_norm is compared against itself across runs; it must be stable."""
    for sample in ("مكتوب 1403 يك", "می‌رود\n\n\nکتاب", "  فاصله   زیاد  "):
        once = normalise_for_eval(sample)
        assert normalise_for_eval(once) == once


# --------------------------------------------------------------------------- #
# Foreign-script filter: the defect a ratio threshold cannot catch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "contaminated",
    [
        "متن فارسی با наблюде روسی",  # Cyrillic
        "واژه 城堡 چینی",  # Han
        "متن با 언어 کره‌ای",  # Hangul
        "واژه र्फ دیوناگری",  # Devanagari
        "متن θερμο یونانی",  # Greek
    ],
)
def test_foreign_script_is_rejected_as_a_kind_not_a_quantity(contaminated):
    """One foreign letter is a defect however high the Persian ratio is."""
    assert persian_letter_ratio(contaminated) > 0.5
    assert check_foreign_script(contaminated) is not None


def test_clean_persian_passes_the_script_filter():
    """The filter must not reject legitimate Persian with digits and quotes."""
    assert check_foreign_script("متن «سالم» با عدد ۱۴۰۳ و نیم‌فاصله می‌رود") is None


def test_markdown_artefacts_are_stripped_not_rejected():
    """Stray asterisks are not a reason to discard a page of good Persian."""
    assert strip_artefacts("**تیتر**") == "تیتر"
    assert strip_artefacts("```\nمتن\n```") == "متن"
    assert strip_artefacts("## عنوان") == "عنوان"


def test_self_quoting_is_unwrapped():
    """A whole reply in quotes is the model quoting itself, not content."""
    assert strip_artefacts("«متن کامل»") == "متن کامل"


# --------------------------------------------------------------------------- #
# Annotation schema: a box that lies about the pixels
# --------------------------------------------------------------------------- #


def _annotation(**overrides):
    """Build a minimal valid annotation, with fields overridable."""
    defaults = {
        "sample_id": "fa_0000001",
        "parent_id": "",
        "group_id": "g1",
        "category": "page",
        "width": 100,
        "height": 50,
        "lines": (
            Line(text="سلام", polygon=rectangle(5, 5, 60, 25), reading_order=0),
            Line(text="دنیا", polygon=rectangle(5, 28, 60, 45), reading_order=1),
        ),
    }
    defaults.update(overrides)
    return Annotation(**defaults)


def test_annotation_rejects_a_box_outside_the_canvas():
    """A box past the edge means the label claims clipped text."""
    with pytest.raises(AnnotationError, match="escapes"):
        _annotation(
            lines=(Line(text="x", polygon=rectangle(5, 5, 500, 25), reading_order=0),)
        )


def test_annotation_rejects_a_gap_in_reading_order():
    """Reading order must be a permutation or the page cannot be serialised."""
    with pytest.raises(AnnotationError, match="contiguous"):
        _annotation(
            lines=(
                Line(text="a", polygon=rectangle(0, 0, 10, 10), reading_order=0),
                Line(text="b", polygon=rectangle(0, 20, 10, 30), reading_order=5),
            )
        )


def test_annotation_rejects_a_non_positive_canvas():
    """A zero-size canvas means the render produced nothing."""
    with pytest.raises(AnnotationError, match="positive"):
        _annotation(width=0)


def test_text_from_lines_follows_reading_order():
    """This is what image QC compares against the label."""
    annotation = _annotation(
        lines=(
            Line(text="دوم", polygon=rectangle(5, 28, 60, 45), reading_order=1),
            Line(text="اول", polygon=rectangle(5, 5, 60, 25), reading_order=0),
        )
    )
    assert annotation.text_from_lines() == "اول\nدوم"


def test_annotation_round_trips_through_json():
    """Shards store annotations as JSON; a lossy round trip loses ground truth."""
    original = _annotation(
        lines=(
            Line(
                text="سلام",
                polygon=rectangle(5, 5, 60, 25),
                reading_order=0,
                words=(Word(text="سلام", polygon=rectangle(5, 5, 30, 25)),),
            ),
        ),
        blocks=(
            Block(
                block_type=BlockType.PARAGRAPH,
                polygon=rectangle(0, 0, 100, 50),
                reading_order=0,
                line_indices=(0,),
            ),
        ),
    )
    rebuilt = Annotation.from_dict(json.loads(original.to_json()))
    assert rebuilt.text_from_lines() == original.text_from_lines()
    assert rebuilt.lines[0].words[0].text == "سلام"
    assert rebuilt.blocks[0].block_type is BlockType.PARAGRAPH


def test_transform_drops_lines_pushed_off_the_canvas():
    """A line warped out of frame must not keep claiming text."""

    class _Shift:
        """Translate everything far to the right."""

        def map_point(self, x, y):
            """Move a point past the canvas edge."""
            return x + 1000, y

    moved = _annotation().transformed(_Shift(), width=100, height=50)
    assert moved.n_lines == 0


# --------------------------------------------------------------------------- #
# Group-aware splitting
# --------------------------------------------------------------------------- #


def test_a_group_always_lands_in_one_split():
    """A page and its crops share a group, so they must share a split."""
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    for index in range(200):
        group = f"g{index}"
        splits = {
            assign_split(
                group,
                ratios=ratios,
                salt="",
                holdout_font=False,
                holdout_template=False,
                holdout_generator=False,
            )
            for _ in range(5)
        }
        assert len(splits) == 1


def test_split_is_stable_across_processes():
    """blake2b, not hash(): the built-in is salted and would re-deal the corpus."""
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    assert assign_split(
        "group-42",
        ratios=ratios,
        salt="s",
        holdout_font=False,
        holdout_template=False,
        holdout_generator=False,
    ) == assign_split(
        "group-42",
        ratios=ratios,
        salt="s",
        holdout_font=False,
        holdout_template=False,
        holdout_generator=False,
    )


def test_holdouts_take_precedence_in_a_fixed_order():
    """A sample flagged twice must land in exactly one split, predictably."""
    ratios = {"train": 1.0, "validation": 0.0, "test": 0.0}
    assert (
        assign_split(
            "g",
            ratios=ratios,
            salt="",
            holdout_font=True,
            holdout_template=True,
            holdout_generator=True,
        )
        == "test_unseen_fonts"
    )


def test_split_ratios_are_approximately_honoured():
    """A hash-based split still has to produce the requested proportions."""
    ratios = {"train": 0.9, "validation": 0.05, "test": 0.05}
    counts = {"train": 0, "validation": 0, "test": 0}
    total = 4000
    for index in range(total):
        counts[
            assign_split(
                f"group-{index}",
                ratios=ratios,
                salt="",
                holdout_font=False,
                holdout_template=False,
                holdout_generator=False,
            )
        ] += 1
    assert abs(counts["train"] / total - 0.9) < 0.02


# --------------------------------------------------------------------------- #
# Era capacity clamping
# --------------------------------------------------------------------------- #


def test_era_weights_are_clamped_to_pool_capacity():
    """An era cannot supply more word-slots than it has words times the cap."""
    pools = {"literary": 100, "unlabelled": 100_000}
    weights, notes = clamp_era_weights(
        {"literary": 0.5, "unlabelled": 0.5},
        pools,
        total_slots=100_000,
        max_uses_per_word=30,
    )
    # 100 words x 30 uses = 3,000 of 100,000 slots = at most 3%.
    assert weights["literary"] <= 0.031
    assert "literary" in notes
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_tiny_eras_are_zeroed_rather_than_reused():
    """A handful of words cannot carry an era; reusing them is not variety."""
    weights, notes = clamp_era_weights(
        {"dialectal": 0.5, "unlabelled": 0.5},
        {"dialectal": 3, "unlabelled": 50_000},
        total_slots=10_000,
        max_uses_per_word=30,
    )
    assert weights["dialectal"] == 0.0
    assert "zeroed" in notes["dialectal"]


def test_clamping_is_a_no_op_when_there_is_headroom():
    """Small runs must not be perturbed by a mechanism meant for large ones."""
    requested = {"archaic": 0.3, "unlabelled": 0.7}
    weights, notes = clamp_era_weights(
        requested,
        {"archaic": 2000, "unlabelled": 100_000},
        total_slots=1_000,
        max_uses_per_word=30,
    )
    assert weights == pytest.approx(requested)
    assert not notes


# --------------------------------------------------------------------------- #
# Near-duplicate detection
# --------------------------------------------------------------------------- #


def test_identical_text_is_detected_as_a_duplicate():
    """Exact repeats must not reach the corpus twice."""
    index = MinHashIndex(permutations=128, seed=7)
    text = "این یک متن آزمایشی فارسی است که برای سنجش تکرار به کار می‌رود."
    index.add("a", index.signature_of(text, 5))
    assert index.query(index.signature_of(text, 5), 0.8) is not None


def test_unrelated_text_is_not_a_duplicate():
    """Two different paragraphs must survive, or the corpus collapses."""
    index = MinHashIndex(permutations=128, seed=7)
    index.add(
        "a",
        index.signature_of("گزارشی درباره وضعیت آب و هوای شهر در زمستان امسال.", 5),
    )
    other = index.signature_of("دستور پخت خورشت قیمه با لپه و گوشت گوسفندی.", 5)
    assert index.query(other, 0.8) is None


def test_dedupe_is_reproducible_across_instances():
    """A randomly seeded hash family would make the decision differ per run."""
    text = "متنی برای سنجش پایداری امضای مین‌هش در اجراهای گوناگون."
    first = MinHashIndex(permutations=64, seed=11).signature_of(text, 5)
    second = MinHashIndex(permutations=64, seed=11).signature_of(text, 5)
    assert first == second


def test_bands_must_divide_permutations():
    """A ragged band silently changes the threshold."""
    with pytest.raises(ValueError, match="must divide"):
        MinHashIndex(permutations=100, seed=1, bands=7)


# --------------------------------------------------------------------------- #
# Templates and config
# --------------------------------------------------------------------------- #


def test_every_declared_template_builds_a_document():
    """A planned template with no builder fails only at render time."""
    from fa_ocr_lora.diverse.layouts.spec import build_palette

    rng = random.Random(3)
    palette = build_palette(rng, paper_tints={"#ffffff": 1.0})
    text = "عنوان\n\nبند نخست با واژه خاموشگر.\n\nستون | مقدار\nردیف | ۱۴۰۳"
    for category, ids in TEMPLATE_IDS.items():
        assert category in CATEGORIES, category
        for template_id in ids:
            html, width, height = build_document(
                template_id,
                text,
                palette,
                font_css="",
                family="Vazirmatn",
                display_family="Lalezar",
                rng=rng,
                digit_script="persian",
            )
            assert html.startswith("<!DOCTYPE html>"), template_id
            assert 'dir="rtl"' in html, template_id
            assert "data-ann" in html, template_id
            assert width > 0, template_id
            assert height >= 0, template_id


def test_every_category_has_templates_and_weights():
    """A weighted category with no template would plan unrenderable samples."""
    config = load_diverse_config(apply_env=False)
    for category in CATEGORIES:
        assert TEMPLATE_IDS.get(category), category
        assert config.plan.category_weights[category] >= 0.0


def test_shipped_config_is_valid_and_internally_consistent():
    """The committed config must load, since every stage starts by reading it."""
    config = load_diverse_config(apply_env=False)
    assert abs(sum(config.plan.category_weights.values()) - 1.0) < 1e-6
    assert abs(sum(config.plan.band_weights.values()) - 1.0) < 1e-6
    assert abs(sum(config.plan.era_weights.values()) - 1.0) < 1e-6
    assert abs(sum(config.splits.ratios.values()) - 1.0) < 1e-6
    assert set(config.plan.band_weights) == set(BANDS)
    assert set(config.plan.era_weights) == set(ERAS)
    # At least two generation profiles, or the corpus learns one generator.
    assert len(config.generation.profiles) >= 2
    # The glyph gate must be absolute.
    assert config.render.fonts.coverage_min == 1.0
    # The holdout generator must name a real profile.
    if config.splits.holdout_generator:
        names = {profile.name for profile in config.generation.profiles}
        assert config.splits.holdout_generator in names
