"""Unit tests for the cost simulator.

The simulator's whole claim is that it is *anchored* - that its arithmetic
reproduces things that are independently known. So these tests do not check that
the model produces plausible-looking numbers; they check it against facts
established elsewhere:

* the parameter census of the pinned checkpoint, which the adapter this repository
  already trained agrees with to the parameter;
* the sequence arithmetic in ``encoding.SampleEncoder``, which the simulator
  duplicates and would silently drift from;
* the ordering properties that make the projection meaningful at all - that a
  frozen tower has no backward term, that the worst batch exceeds the mean one,
  that a bigger card is not projected to be slower.

A cost model whose errors are invisible is worse than no cost model, because
someone rents a machine on the strength of it.
"""

from __future__ import annotations

import json

import pytest

from fa_ocr_lora.config.base import ConfigError
from fa_ocr_lora.training.config import load_training_config
from fa_ocr_lora.training.simulate import (
    BASE_PARAMETERS,
    CHARS_PER_TOKEN,
    EFFICIENCY_SCENARIOS,
    HARDWARE_PRESETS,
    PROMPT_OVERHEAD,
    CorpusProfile,
    SourceProfile,
    ThroughputModel,
    _feed_geometry,
    _oversample_factor,
    _split_sizes,
    calibrate,
    host_model,
    memory_model,
    render,
    sample_flops,
    simulate,
    trainable_parameters,
    weight_bytes,
)

_A100 = "configs/lora_config_a100.yaml"
_LOCAL = "configs/lora_config.yaml"

#: The adapter this repository trained, from its own ``adapter_config.json``
#: shape. If the simulator's parameter arithmetic cannot reproduce this, none of
#: its memory numbers mean anything.
_KNOWN_TRAINABLE = 9_388_032


@pytest.fixture(scope="module")
def a100_config():
    """The A100 profile, loaded once."""
    return load_training_config(_A100)


@pytest.fixture(scope="module")
def local_config():
    """The 4 GB profile, loaded once."""
    return load_training_config(_LOCAL)


@pytest.fixture
def profile() -> CorpusProfile:
    """A synthetic mix that needs no corpus on disk.

    Deliberately hand-built rather than measured: these tests must run in CI
    where ``data/`` does not exist, and the properties under test are structural.
    """
    return CorpusProfile(
        sources=(
            SourceProfile(
                source="admin_letter_page",
                records=26_848,
                image_tokens=2534.6,
                answer_tokens=406.1,
                seq_len=2953.6,
                seq_len_max=4096,
                patches=10_138.3,
                patches_sq=107_014_827.0,
                patches_max=11_088,
                bytes_on_disk=1_400_000.0,
            ),
            SourceProfile(
                source="admin_letter_crop",
                records=60_000,
                image_tokens=223.3,
                answer_tokens=33.2,
                seq_len=269.5,
                seq_len_max=4089,
                patches=893.3,
                patches_sq=950_205.5,
                patches_max=7992,
            ),
            SourceProfile(
                source="persian_pixel_page",
                records=31_000,
                image_tokens=586.1,
                answer_tokens=370.1,
                seq_len=969.2,
                seq_len_max=1880,
                patches=2344.3,
                patches_sq=7_261_062.5,
                patches_max=4212,
                bytes_on_disk=120_000.0,
            ),
        ),
        train_records=113_006,
        validation_records=800,
        test_records=5892,
        oversample_factor=1.25,
    )


# ---------------------------------------------------------------- parameters


def test_trainable_parameter_count_reproduces_the_trained_adapter(a100_config):
    """The derived count must match the adapter already on disk exactly.

    Both configs use the same adapter shape, so both must land on the same
    number - and that number is not a target the arithmetic was fitted to, it is
    what PEFT actually created.
    """
    assert trainable_parameters(a100_config) == _KNOWN_TRAINABLE


def test_trainable_count_is_a_fraction_of_a_percent_of_the_base(a100_config):
    """The adapter is under 1% of the model, which is the point of LoRA."""
    ratio = trainable_parameters(a100_config) / BASE_PARAMETERS
    assert 0.009 < ratio < 0.011


def test_lora_rank_scales_the_parameter_count_linearly(a100_config):
    """Doubling the rank doubles the adapter, since every factor is rank-sized."""
    from dataclasses import replace

    doubled = replace(a100_config, lora=replace(a100_config.lora, r=32))
    assert trainable_parameters(doubled) == 2 * trainable_parameters(a100_config)


def test_bf16_weights_are_twice_the_parameter_count(a100_config):
    """``quantization: none`` means two bytes per parameter and nothing else."""
    assert a100_config.model.quantization == "none"
    assert weight_bytes(a100_config) == pytest.approx(BASE_PARAMETERS * 2)


def test_nf4_is_smaller_than_bf16_but_not_four_times_smaller(local_config):
    """NF4 saves real memory, but the head and embedding stay full precision.

    A model that assumed a flat 4x saving would under-project the 4 GB run by
    hundreds of MiB - which is exactly the error that makes a run OOM at step
    one after a projection said it would fit.
    """
    assert local_config.model.quantization == "nf4"
    nf4 = weight_bytes(local_config)
    bf16 = BASE_PARAMETERS * 2
    assert nf4 < bf16
    assert nf4 > bf16 / 4


# ---------------------------------------------------------------- geometry


def test_feed_geometry_matches_the_encoders_budget_rule(a100_config):
    """A short answer leaves the image its full configured budget."""
    image = a100_config.image
    visual, total = _feed_geometry(1240, 1754, 10, image)
    assert visual == 2772  # 44 x 63 patches of 28 px
    assert total == PROMPT_OVERHEAD + visual + 10


def test_a_long_answer_costs_the_image_resolution(a100_config):
    """This is the behaviour the whole projection turns on.

    ``_image_budget`` hands the image whatever the answer leaves, so a long
    transcription silently degrades its own page - before ``on_oversize`` is ever
    consulted. A simulator that applied ``max_pixels`` unconditionally would miss
    it entirely.
    """
    image = a100_config.image
    short, _ = _feed_geometry(1240, 1754, 10, image)
    long, total = _feed_geometry(1240, 1754, 3500, image)
    assert long < short
    assert total <= image.max_total_seq_len


def test_sequence_length_never_exceeds_the_configured_cap(a100_config):
    """Whatever the answer length, the reported sequence fits the budget."""
    image = a100_config.image
    for answer in (0, 500, 2000, 4000, 9000):
        _, total = _feed_geometry(1240, 1754, answer, image)
        assert total <= image.max_total_seq_len


def test_the_min_pixels_floor_applies_to_tiny_crops(a100_config):
    """A 145x40 metadata box is upscaled to the floor, not fed at its own size.

    This is why crops cost far more than their text suggests, and the finding
    the report raises about ``min_pixels`` depends on it being modelled.
    """
    image = a100_config.image
    visual, _ = _feed_geometry(145, 40, 12, image)
    assert visual * 784 >= image.min_pixels * 0.9


# ---------------------------------------------------------------- flops


def test_the_frozen_vision_tower_has_no_backward_term(profile):
    """Gradient checkpointing must not change the tower's cost.

    Frozen parameters plus non-grad ``pixel_values`` means no autograd graph, so
    there is nothing to recompute. If this ever changes, the wall-clock
    projection is wrong by roughly a third.
    """
    on = sample_flops(profile, gradient_checkpointing=True)
    off = sample_flops(profile, gradient_checkpointing=False)
    assert on["vision"] == off["vision"]
    assert on["decoder"] > off["decoder"]


def test_checkpointing_costs_one_extra_decoder_forward(profile):
    """Recompute adds exactly one forward, not a doubling."""
    on = sample_flops(profile, gradient_checkpointing=True)
    off = sample_flops(profile, gradient_checkpointing=False)
    assert on["decoder"] / off["decoder"] == pytest.approx(1.5, rel=1e-6)


def test_pages_cost_far_more_than_crops(profile):
    """The mix's cost is dominated by pages, which is why the ratio matters."""
    page, crop, _ = profile.sources
    page_cost = sample_flops(page, gradient_checkpointing=False)["total"]
    crop_cost = sample_flops(crop, gradient_checkpointing=False)["total"]
    assert page_cost > 10 * crop_cost


def test_squared_patch_mean_exceeds_the_square_of_the_mean(profile):
    """Jensen's inequality, and the reason ``patches_sq`` is carried separately.

    The letter pages are bimodal - most at 2,772 visual tokens, a sixth at
    1,364 - so using ``E[p]^2`` for a quadratic term understates the attention
    cost of the expensive half.
    """
    for source in profile.sources:
        assert source.patches_sq >= source.patches**2


# ---------------------------------------------------------------- memory


def test_the_worst_batch_needs_more_than_the_mean_batch(a100_config, profile):
    """Peak is set by the longest sequences, and the sampler serves them first."""
    worst = memory_model(a100_config, profile, worst_case=True)
    mean = memory_model(a100_config, profile, worst_case=False)
    assert worst.total > mean.total


def test_dense_vision_attention_would_not_fit_the_card(a100_config, profile):
    """The unverified assumption must be visibly decisive, not quietly absent.

    Whether the tower materialises a ``[heads, patches, patches]`` score matrix is
    the difference between a run that fits and a run that does not: at 11,088
    patches and 16 heads it is ~30 GiB on its own, on top of a projection that
    already sits near 30 GiB. So the report has to be able to say which side of
    the card each assumption lands on - a vaguer "dense is bigger" check would
    pass even if the term were off by the head count, which it was.
    """
    budget = HARDWARE_PRESETS["a100_40gb"].vram_gib * 1024
    packed = memory_model(a100_config, profile, materialise_vision_attention=False)
    dense = memory_model(a100_config, profile, materialise_vision_attention=True)
    assert packed.total < budget
    assert dense.total > budget


def test_the_a100_projection_fits_the_card(a100_config, profile):
    """The committed A100 config must be projected to fit 40 GiB."""
    peak = memory_model(a100_config, profile, worst_case=True).total
    assert peak < HARDWARE_PRESETS["a100_40gb"].vram_gib * 1024


def test_doubling_the_batch_roughly_doubles_the_dynamic_memory(a100_config, profile):
    """Activations scale with batch; weights and optimiser state do not."""
    from dataclasses import replace

    doubled = replace(
        a100_config,
        optim=replace(a100_config.optim, per_device_batch_size=16),
    )
    one = memory_model(a100_config, profile)
    two = memory_model(doubled, profile)
    assert two.decoder_activations == pytest.approx(2 * one.decoder_activations)
    assert two.weights == one.weights
    assert two.optimizer == one.optimizer


def test_the_breakdown_is_ordered_and_sums_to_the_total(a100_config, profile):
    """A breakdown that does not add up is a report that cannot be checked."""
    model = memory_model(a100_config, profile)
    items = model.breakdown()
    assert [size for _, size in items] == sorted(
        (size for _, size in items), reverse=True
    )
    assert sum(size for _, size in items) == pytest.approx(model.total)


# ---------------------------------------------------------------- throughput


def test_a_faster_card_is_projected_to_be_faster(profile):
    """Ordering sanity: the same work on 312 TFLOPS beats 30.4 TFLOPS."""
    flops = sample_flops(profile, gradient_checkpointing=False)["total"]
    fast = ThroughputModel(HARDWARE_PRESETS["a100_40gb"], flops, 0.35, "expected")
    slow = ThroughputModel(HARDWARE_PRESETS["rtx_3050ti"], flops, 0.35, "expected")
    assert fast.seconds_per_sample < slow.seconds_per_sample
    assert fast.hours(1000) < slow.hours(1000)


def test_efficiency_brackets_are_ordered():
    """Pessimistic must actually be pessimistic."""
    assert (
        EFFICIENCY_SCENARIOS["pessimistic"]
        < EFFICIENCY_SCENARIOS["expected"]
        < EFFICIENCY_SCENARIOS["optimistic"]
    )


def test_calibration_reads_a_completed_run(tmp_path, local_config, profile):
    """Back-solving must recover the achieved rate from wall-clock and FLOPs.

    The sample count must be the one the wall-clock covers. ``epoch_samples`` is
    one pass; this run reports 1,001 of them against 2,397.5 s spanning three
    epochs, so dividing by it would overstate the per-sample cost threefold and
    understate the achieved rate by the same factor - which then silently drags
    the pessimistic bracket of every projection anchored to it.
    """
    (tmp_path / "training_summary.json").write_text(
        json.dumps(
            {
                "epoch_samples": 1001,
                "seconds": 2397.5,
                "seconds_per_sample": 0.793,
                "peak_vram_mib": 1253.4,
            }
        ),
        encoding="utf-8",
    )
    result = calibrate(
        tmp_path,
        hardware=HARDWARE_PRESETS["rtx_3050ti"],
        config=local_config,
        profile=profile,
    )
    assert result.seconds_per_sample == pytest.approx(0.793, rel=1e-3)
    assert result.samples > 1001
    assert 0.0 < result.mfu < 1.0
    assert result.achieved_tflops < HARDWARE_PRESETS["rtx_3050ti"].bf16_tflops


def test_calibration_prefers_steps_times_batch_over_the_epoch_size(
    tmp_path, local_config, profile
):
    """``steps x batch_size`` is what the trainer itself divides by."""
    (tmp_path / "training_summary.json").write_text(
        json.dumps(
            {
                "steps": 378,
                "batch_size": 8,
                "epoch_samples": 1001,
                "seconds": 2397.5,
                "peak_vram_mib": 1253.4,
            }
        ),
        encoding="utf-8",
    )
    result = calibrate(
        tmp_path,
        hardware=HARDWARE_PRESETS["rtx_3050ti"],
        config=local_config,
        profile=profile,
    )
    assert result.samples == 378 * 8
    assert result.seconds_per_sample == pytest.approx(2397.5 / (378 * 8))


def test_calibration_never_borrows_the_target_batch_size(tmp_path, a100_config, profile):
    """A summary with no ``batch_size`` must not adopt the target machine's.

    The configuration being simulated describes the machine the run is projected
    ONTO; the calibration run happened somewhere else, at some other batch size.
    Substituting one for the other scales the measured rate by the ratio between
    them - here a100 batch 16 against the run's real 8, which would halve the
    per-sample cost and double the apparent MFU.
    """
    (tmp_path / "training_summary.json").write_text(
        json.dumps(
            {
                "steps": 378,
                "epoch_samples": 1001,
                "seconds": 2397.5,
                "seconds_per_sample": 0.793,
                "peak_vram_mib": 1253.4,
            }
        ),
        encoding="utf-8",
    )
    result = calibrate(
        tmp_path,
        hardware=HARDWARE_PRESETS["rtx_3050ti"],
        config=a100_config,
        profile=profile,
    )
    # Back-solved from the trainer's own rate, not from 378 x 16.
    assert result.samples == pytest.approx(2397.5 / 0.793, abs=1)
    assert result.seconds_per_sample == pytest.approx(0.793, rel=1e-3)


def test_calibration_refuses_a_directory_with_no_summary(tmp_path, local_config, profile):
    """Missing evidence must be an error, not a silently uncalibrated guess."""
    with pytest.raises(ConfigError, match="training_summary"):
        calibrate(
            tmp_path,
            hardware=HARDWARE_PRESETS["rtx_3050ti"],
            config=local_config,
            profile=profile,
        )


def test_calibration_refuses_a_summary_with_no_timing(tmp_path, local_config, profile):
    """A summary without seconds cannot calibrate anything."""
    (tmp_path / "training_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError, match="positive"):
        calibrate(
            tmp_path,
            hardware=HARDWARE_PRESETS["rtx_3050ti"],
            config=local_config,
            profile=profile,
        )


# ---------------------------------------------------------------- host


def test_workers_cannot_outrun_the_cores_they_run_on(a100_config, profile):
    """Supply is capped by vCPUs, not by the configured worker count.

    Otherwise a config asking for 64 workers on an 8-core box would project an
    input pipeline that cannot exist.
    """
    from dataclasses import replace

    greedy = replace(
        a100_config,
        runtime=replace(a100_config.runtime, dataloader_workers=64),
    )
    small = HARDWARE_PRESETS["rtx_3050ti"]
    host = host_model(greedy, profile, small, samples_per_second=1.0)
    capped = replace(
        greedy,
        runtime=replace(greedy.runtime, dataloader_workers=small.vcpus - 1),
    )
    per_core = host_model(capped, profile, small, samples_per_second=1.0)
    assert host.samples_per_second_supply == pytest.approx(
        per_core.samples_per_second_supply
    )


def test_the_a100_input_pipeline_keeps_up(a100_config, profile):
    """32 vCPUs must out-supply an A100 on this mix, or the projection is moot."""
    host = host_model(
        a100_config, profile, HARDWARE_PRESETS["a100_40gb"], samples_per_second=15.0
    )
    assert not host.input_bound


def test_storage_components_sum_to_the_total(a100_config, profile):
    """Same check as the memory breakdown, for the same reason."""
    host = host_model(
        a100_config, profile, HARDWARE_PRESETS["a100_40gb"], samples_per_second=15.0
    )
    assert sum(size for _, size in host.storage) == pytest.approx(host.storage_gib)


# ---------------------------------------------------------------- assembly


def test_split_sizes_cap_the_holdout_on_a_small_selection(a100_config):
    """An 800-record holdout out of 100 would swallow the training split."""
    train, validation, _ = _split_sizes(a100_config, 100)
    assert validation < a100_config.data.validation_size
    assert train > 0


def test_split_sizes_use_the_full_holdout_on_a_large_selection(a100_config):
    """At corpus scale the configured holdout applies unchanged."""
    train, validation, test = _split_sizes(a100_config, 118_000)
    assert validation == a100_config.data.validation_size
    assert train + validation + test == pytest.approx(118_000, rel=0.01)


def test_oversampling_repeats_the_top_quartile(a100_config):
    """A 2.0 multiplier over a quarter of records means 1.25x samples."""
    assert _oversample_factor(a100_config) == pytest.approx(1.25)


def test_simulate_assembles_a_consistent_projection(a100_config, profile):
    """End to end on a synthetic profile: the pieces must agree with each other."""
    sim = simulate(a100_config, HARDWARE_PRESETS["a100_40gb"], profile=profile)
    assert sim.total_samples > sim.epoch_samples  # 2 epochs
    assert sim.total_steps > sim.steps_per_epoch
    assert sim.memory.total > sim.memory_typical.total
    assert sim.expected.label == "expected"
    payload = sim.to_dict()
    assert payload["vram_mib"]["peak_worst_batch"] > 0
    assert payload["schedule"]["total_steps"] == sim.total_steps
    assert payload["findings"]


def test_calibration_never_lowers_a_bracket_or_disorders_them(
    a100_config, profile, tmp_path, local_config
):
    """A measured run anchors the low end; it must never lower the brackets.

    And when the measurement lands above a bound the whole bracket lifts, so
    "pessimistic" can never print a faster run than "optimistic".
    """
    (tmp_path / "training_summary.json").write_text(
        json.dumps({"epoch_samples": 1001, "seconds": 2397.5, "peak_vram_mib": 1253.4}),
        encoding="utf-8",
    )
    cal = calibrate(
        tmp_path,
        hardware=HARDWARE_PRESETS["rtx_3050ti"],
        config=local_config,
        profile=profile,
    )
    sim = simulate(
        a100_config, HARDWARE_PRESETS["a100_40gb"], profile=profile, calibration=cal
    )
    by_label = {m.label: m.mfu for m in sim.throughput}
    assert min(by_label.values()) >= EFFICIENCY_SCENARIOS["pessimistic"]
    assert by_label["pessimistic"] <= by_label["expected"] <= by_label["optimistic"]


def test_findings_flag_an_oversized_configuration(profile, a100_config):
    """A batch that cannot fit must produce a blocker, not a cheerful total."""
    from dataclasses import replace

    huge = replace(
        a100_config,
        optim=replace(a100_config.optim, per_device_batch_size=256),
    )
    sim = simulate(huge, HARDWARE_PRESETS["a100_40gb"], profile=profile)
    severities = {severity for severity, _ in sim.findings()}
    assert "blocker" in severities


def test_render_produces_a_report_without_raising(a100_config, profile):
    """The renderer touches every field; a formatting slip must fail here."""
    sim = simulate(a100_config, HARDWARE_PRESETS["a100_40gb"], profile=profile)
    text = render(sim)
    assert "Training-run projection" in text
    assert "VRAM" in text
    assert "conclusions" in text
    assert len(text.splitlines()) > 40


def test_chars_per_token_covers_every_source(profile):
    """A source with no ratio would be profiled with a KeyError at corpus time."""
    for source in profile.sources:
        assert source.source in CHARS_PER_TOKEN
