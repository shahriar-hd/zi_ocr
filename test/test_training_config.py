"""Unit tests for the training configuration loader.

Two things are being defended here.

The first is **strictness**. A training config is read once, at startup, and then
governs several hours of GPU time; a key that is silently ignored is worse than one
that is rejected, because the run completes and quietly used a default nobody chose.
So unknown keys, inverted bounds and physically impossible values must all raise,
and the error must name the dotted path.

The second is that **the committed YAML files actually load**. Both profiles -
the 4 GB local one and the A100 one - are exercised below. Neither is reachable from
any other test, so without this a typo in either sits undetected until someone
starts a run.
"""

from __future__ import annotations

import pytest

from fa_ocr_lora.config.base import ConfigError
from fa_ocr_lora.training.config import (
    DEFAULT_TRAINING_CONFIG_PATH,
    ENV_PREFIX,
    PIXELS_PER_TOKEN,
    DataConfig,
    EvalConfig,
    ImageConfig,
    LoraConfig,
    ModelConfig,
    OptimConfig,
    RuntimeConfig,
    TrainingConfig,
    load_training_config,
)

_A100_CONFIG_PATH = "configs/lora_config_a100.yaml"


def test_pixels_per_token_matches_the_processors_own_arithmetic():
    """784 px is patch_size 14 squared times merge_size 2 squared.

    Every pixel budget in the config is validated against this number, so if the
    constant drifts from the preprocessor the validation starts approving budgets
    the processor will round down.
    """
    assert PIXELS_PER_TOKEN == 784


def test_an_empty_payload_yields_the_documented_defaults():
    config = TrainingConfig.from_dict({})
    assert config.data.total_records == 10_000
    assert config.image.max_pixels == 1_003_520
    assert config.image.max_image_tokens == 1280
    assert config.optim.effective_batch_size == 16
    assert config.lora.r == 16


@pytest.mark.parametrize(
    "section",
    ["model", "paths", "data", "image", "lora", "optim", "runtime", "eval"],
)
def test_every_section_rejects_an_unknown_key(section):
    """Strictness has to hold in all eight sections, not just the ones with tests.

    Parametrised rather than spot-checked because each section has its own ``parse``
    and a new one could easily be written without the ``_reject_unknown`` call.
    """
    with pytest.raises(ConfigError, match="unknown key"):
        TrainingConfig.from_dict({section: {"not_a_real_key": 1}})


def test_an_unknown_top_level_section_is_rejected():
    with pytest.raises(ConfigError, match="unknown top-level section"):
        TrainingConfig.from_dict({"optimiser": {}})


def test_a_near_miss_key_is_rejected_and_the_error_names_the_alternatives():
    """``max_pixel`` for ``max_pixels`` is the typo this strictness exists for.

    Ignored, it would train the whole corpus at the default resolution with no hint
    why the run was slower or the CER worse than expected.
    """
    with pytest.raises(ConfigError) as excinfo:
        TrainingConfig.from_dict({"image": {"max_pixel": 500_000}})
    message = str(excinfo.value)
    assert "image:" in message
    assert "max_pixel" in message
    assert "max_pixels" in message


def test_a_section_that_is_not_a_mapping_is_named():
    with pytest.raises(ConfigError, match="image: expected a mapping"):
        TrainingConfig.from_dict({"image": [1, 2, 3]})


def test_a_missing_section_takes_defaults_rather_than_failing():
    """A minimal config file has to be legal, or nobody writes one."""
    config = TrainingConfig.from_dict({"seed": 7})
    assert config.seed == 7
    assert config.image == ImageConfig()
    assert config.runtime == RuntimeConfig()


def test_a_pixel_budget_that_is_not_a_whole_number_of_tokens_is_rejected():
    """Reject a pixel budget that is not a whole number of image tokens.

    The processor rounds such a budget down, so the run trains at a resolution
    nobody chose - a silent failure worth a loud error.
    """
    with pytest.raises(ConfigError, match="not a multiple of 784"):
        ImageConfig.parse({"max_pixels": 1_003_521})


def test_a_pixel_budget_that_leaves_no_room_for_text_is_rejected():
    """An image filling the sequence would mask the entire answer.

    Loss would then be computed over zero supervised positions and the run would
    train on nothing while reporting a plausible-looking zero loss.
    """
    with pytest.raises(ConfigError, match="leaves no room for text"):
        ImageConfig.parse({"max_pixels": 1_003_520, "max_total_seq_len": 1280})


def test_inverted_pixel_bounds_are_rejected():
    with pytest.raises(ConfigError, match="must exceed min_pixels"):
        ImageConfig.parse({"max_pixels": 147_384, "min_pixels": 1_003_520})


@pytest.mark.parametrize("value", [1, 128, 255])
def test_a_vram_reservation_too_small_for_the_cuda_context_is_rejected(value):
    """0 means "reason from measured coefficients"; 1-255 MiB is a typo.

    The CUDA context alone does not fit in 255 MiB, so such a value cannot be a
    real budget - and taken at face value it would clamp the image budget to the
    floor and make the run inexplicably bad rather than failing.
    """
    with pytest.raises(ConfigError, match="reserved_vram_mib"):
        ImageConfig.parse({"reserved_vram_mib": value})


@pytest.mark.parametrize("value", [0, 256, 4096])
def test_a_usable_vram_reservation_is_accepted(value):
    assert ImageConfig.parse({"reserved_vram_mib": value}).reserved_vram_mib == value


def test_a_mix_with_nothing_to_train_on_is_rejected():
    with pytest.raises(ConfigError, match="nothing to train on"):
        DataConfig.parse({"letters_pages": 0, "letters_crops": 0, "pixel_pages": 0})


def test_a_holdout_larger_than_the_corpus_is_rejected():
    with pytest.raises(ConfigError, match="leaves no training records"):
        DataConfig.parse(
            {
                "letters_pages": 10,
                "letters_crops": 0,
                "pixel_pages": 0,
                "validation_size": 10,
            }
        )


def test_an_area_floor_below_the_side_floor_squared_is_rejected():
    """Reject an area floor the side floor already subsumes.

    Such an area bound can never bind, so it reads as protection that is not there.
    """
    with pytest.raises(ConfigError, match="can never bind"):
        DataConfig.parse({"crop_min_side_px": 32, "crop_min_area_px": 900})


def test_crop_weights_are_relative_and_have_a_default_for_unseen_types():
    data = DataConfig.parse({"crop_type_weights": {"metadata_field": 9.0}})
    assert data.crop_weight("metadata_field") == 9.0
    # A type absent from the mapping still contributes, so a regenerated corpus
    # with a new region type does not silently vanish from the mix.
    assert data.crop_weight("brand_new_type") == data.crop_default_weight


def test_an_out_of_range_value_names_its_dotted_path():
    with pytest.raises(ConfigError, match="runtime.dataloader_workers"):
        RuntimeConfig.parse({"dataloader_workers": 99})


@pytest.mark.parametrize(
    ("section", "payload", "fragment"),
    [
        (ModelConfig, {"dtype": "float8"}, "dtype"),
        (ModelConfig, {"quantization": "int4"}, "quantization"),
        (ImageConfig, {"on_oversize": "explode"}, "on_oversize"),
        (OptimConfig, {"optimizer": "sgd"}, "optimizer"),
        (OptimConfig, {"lr_scheduler_type": "triangular"}, "lr_scheduler_type"),
        (LoraConfig, {"bias": "some"}, "bias"),
    ],
)
def test_a_value_outside_the_allowed_choices_is_rejected(section, payload, fragment):
    with pytest.raises(ConfigError, match=fragment):
        section.parse(payload)


def test_an_empty_target_module_list_is_rejected():
    """An adapter with no targets trains nothing while reporting success."""
    with pytest.raises(ConfigError, match="at least one projection"):
        LoraConfig.parse({"target_modules": []})


def test_eval_max_samples_accepts_null_for_the_whole_split():
    assert EvalConfig.parse({"max_samples": None}).max_samples is None
    assert EvalConfig.parse({"max_samples": 50}).max_samples == 50


def test_effective_batch_size_is_the_product_of_the_two_knobs():
    """The learning rate is only transferable between machines if this matches.

    Both committed profiles are set to 16 - 1x16 locally, 8x2 on the A100 - so
    2.0e-4 means the same thing in each.
    """
    optim = OptimConfig.parse(
        {"per_device_batch_size": 8, "gradient_accumulation_steps": 2}
    )
    assert optim.effective_batch_size == 16


def test_to_dict_is_json_safe_and_round_trips():
    """Render the config to JSON-safe dicts and read it back unchanged.

    The run's summary and checkpoint metadata are written from this, so a finished
    run always carries the settings that produced it.
    """
    original = TrainingConfig.from_dict({"seed": 99, "image": {"loss_chunk_rows": 256}})
    payload = original.to_dict()
    assert payload["seed"] == 99
    assert payload["image"]["loss_chunk_rows"] == 256
    # Paths render as strings and tuples as lists, or json.dump would fail.
    assert isinstance(payload["paths"]["output_dir"], str)
    assert isinstance(payload["lora"]["target_modules"], list)
    payload.pop("source_path")
    assert TrainingConfig.from_dict(payload) == original


def test_environment_overrides_win_over_the_file(tmp_path, monkeypatch):
    """Precedence is CLI > env > YAML > default; this is the middle boundary.

    Only the settings that genuinely differ per machine are exposed, which is why
    this is paths and the memory budget rather than every key.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "image:\n  max_pixels: 1003520\n  reserved_vram_mib: 0\n", encoding="utf-8"
    )
    monkeypatch.setenv(f"{ENV_PREFIX}MAX_PIXELS", "784000")
    monkeypatch.setenv(f"{ENV_PREFIX}RESERVED_VRAM_MIB", "2048")
    config = load_training_config(config_path)
    assert config.image.max_pixels == 784_000
    assert config.image.reserved_vram_mib == 2048


def test_apply_env_false_reads_the_file_exactly_as_written(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("image:\n  max_pixels: 1003520\n", encoding="utf-8")
    monkeypatch.setenv(f"{ENV_PREFIX}MAX_PIXELS", "784000")
    config = load_training_config(config_path, apply_env=False)
    assert config.image.max_pixels == 1_003_520


def test_an_override_from_the_environment_hits_the_same_validators(monkeypatch, tmp_path):
    """A bad value from the environment must fail like a bad value in the file.

    Otherwise the env path is a hole in the validation, and the error would surface
    hours later as a confusing runtime failure.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 1\n", encoding="utf-8")
    monkeypatch.setenv(f"{ENV_PREFIX}MAX_PIXELS", "1003521")
    with pytest.raises(ConfigError, match="not a multiple of 784"):
        load_training_config(config_path)


def test_a_non_numeric_environment_override_is_rejected(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 1\n", encoding="utf-8")
    monkeypatch.setenv(f"{ENV_PREFIX}MAX_PIXELS", "lots")
    with pytest.raises(ConfigError, match="expected an integer"):
        load_training_config(config_path)


def test_an_empty_or_missing_file_is_reported_clearly(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="file is empty"):
        load_training_config(empty)
    with pytest.raises(ConfigError, match="training config not found"):
        load_training_config(tmp_path / "absent.yaml")


def test_invalid_yaml_is_reported_as_such(tmp_path):
    broken = tmp_path / "broken.yaml"
    broken.write_text("image:\n  max_pixels: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_training_config(broken)


@pytest.mark.parametrize("path", [DEFAULT_TRAINING_CONFIG_PATH, _A100_CONFIG_PATH])
def test_the_committed_profiles_load_and_are_internally_consistent(path):
    """Load each committed profile and restate the invariants the loader enforces.

    The only test that reads the real files, so it is the only thing standing between
    a typo in either and a failed run start. ``apply_env=False`` throughout: the point
    is what the file says, and a developer with ``FA_OCR_TRAIN_MAX_PIXELS`` exported
    should not change the verdict.
    """
    config = load_training_config(path, apply_env=False)
    assert config.source_path is not None
    assert config.source_path.name == path.rsplit("/", 1)[-1]
    # The invariants the loader enforces, restated as assertions so a future edit
    # that trips one fails here rather than at model-load time.
    assert config.image.max_pixels % PIXELS_PER_TOKEN == 0
    assert config.image.max_image_tokens < config.image.max_total_seq_len
    assert config.data.validation_size < config.data.total_records
    assert config.optim.effective_batch_size >= 1
    assert config.lora.target_modules


def test_both_profiles_agree_on_what_the_learning_rate_means():
    """Check the two profiles share the settings a learning rate is relative to.

    The A100 profile reuses the local profile's learning rates, which is only sound
    because the effective batch size and the holdout are identical.
    """
    local = load_training_config(DEFAULT_TRAINING_CONFIG_PATH, apply_env=False)
    a100 = load_training_config(_A100_CONFIG_PATH, apply_env=False)
    assert a100.optim.effective_batch_size == local.optim.effective_batch_size
    assert a100.optim.learning_rate == local.optim.learning_rate
    assert a100.optim.embed_head_learning_rate == local.optim.embed_head_learning_rate
    # Same seed and holdout size means the same records are held out on both
    # machines, so a pilot CER is directly comparable to a full-run CER.
    assert a100.seed == local.seed
    assert a100.data.validation_size == local.data.validation_size
    # And they must not write into the same directory, or the full run overwrites
    # the pilot's manifest and metrics.
    assert a100.paths.output_dir != local.paths.output_dir


def test_the_a100_profile_actually_removes_the_4gb_concessions():
    """Guards the intent of the file, not just its syntax.

    Each of these is a concession the local config makes to fit 4 GB, and a
    well-meaning edit that "unified" the two configs would silently reimpose it on
    a 40 GB card.
    """
    a100 = load_training_config(_A100_CONFIG_PATH, apply_env=False)
    assert a100.model.quantization == "none"
    assert a100.model.quantize_vision is False
    assert a100.optim.gradient_checkpointing is False
    assert a100.optim.per_device_batch_size > 1
    assert a100.image.autoprobe_vram is False
    assert a100.eval.max_samples is None
    # Full page resolution: the corpus's largest page is 1240x1754 = 2,174,960 px,
    # so nothing is downscaled.
    assert a100.image.max_pixels >= 1240 * 1754
