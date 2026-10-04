"""Tests for how a profile reaches its base weights.

``model.auto_class`` is the only thing standing between loading the original,
remote-code checkpoint and loading PaddleOCR-VL-1.6 through the class its card
uses. Getting it backwards does not raise: it loads a class the LoRA adapter
would then be attached to for the wrong architecture. These pin the mapping for
both profiles on disk plus the default, so a new profile cannot silently pick
the wrong one.
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from fa_ocr_lora.training.config import ModelConfig, load_training_config
from fa_ocr_lora.training.modeling import auto_model_class

_ORIGINAL = "configs/lora_config_rtx6000pro.yaml"
_NATIVE = "configs/lora_config_rtx6000pro_16.yaml"


def test_the_original_profile_loads_as_causal_language_modelling():
    """Its auto_map publishes AutoModelForCausalLM and nothing else."""
    config = load_training_config(_ORIGINAL, apply_env=False)
    assert auto_model_class(config.model) is AutoModelForCausalLM


def test_the_1_6_profile_loads_as_image_text_to_text():
    """The 1.6 card's own class - native to transformers 5, no remote code."""
    config = load_training_config(_NATIVE, apply_env=False)
    assert auto_model_class(config.model) is AutoModelForImageTextToText


def test_the_default_is_the_causal_path():
    """A profile that says nothing keeps loading the way it always has."""
    assert auto_model_class(ModelConfig.parse({})) is AutoModelForCausalLM
