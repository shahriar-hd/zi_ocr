"""Tests for how a profile reaches its base weights.

``model.auto_class`` is the only thing standing between loading the original,
remote-code checkpoint and loading PaddleOCR-VL-1.6 through the class its card
uses. Getting it backwards does not raise: it loads a class the LoRA adapter
would then be attached to for the wrong architecture. These pin the mapping for
both profiles on disk plus the default, so a new profile cannot silently pick
the wrong one.
"""

from __future__ import annotations

import pytest
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from fa_ocr_lora.training.config import (
    LoraConfig,
    ModelConfig,
    OptimConfig,
    load_training_config,
)
from fa_ocr_lora.training.modeling import (
    ModelLayout,
    assert_frozen,
    auto_model_class,
    freeze_vision,
    lora_target_paths,
    model_layout,
    on_text_path,
    parameter_groups,
)

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


def test_the_default_is_the_native_path():
    """Omission means 1.6, which is the base this project actually trains.

    The defaults used to describe the original release, so a config that said
    nothing got remote code and the causal class. 1.6 is the supported base and
    loads natively, so that is what omission resolves to now; the profiles that
    still want the old path state it, and the parser rejects the incoherent mix.
    """
    assert auto_model_class(ModelConfig.parse({})) is AutoModelForImageTextToText


class _Projections(nn.Module):
    """A group of Linear projections named the way the config names them."""

    def __init__(self, names: tuple[str, ...]) -> None:
        super().__init__()
        for name in names:
            setattr(self, name, nn.Linear(4, 4))


class _Block(nn.Module):
    """One decoder block: attention and MLP projections under their usual names."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Projections(("q_proj", "k_proj", "v_proj", "o_proj"))
        self.mlp = _Projections(("gate_proj", "up_proj", "down_proj"))


class _Tower(nn.Module):
    """A vision tower whose attention uses the *same* projection names."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(_Block() for _ in range(depth))


def _original_release() -> nn.Module:
    """PaddlePaddle/PaddleOCR-VL: decoder at `model.layers`, tower at the top.

    The layout the hard-coded constants describe: `visual` and `mlp_AR` are
    top-level, `embed_tokens` and `layers` live under one `model` attribute.
    """
    outer = nn.Module()
    decoder = nn.Module()
    decoder.embed_tokens = nn.Embedding(8, 4)
    decoder.layers = nn.ModuleList(_Block() for _ in range(2))
    decoder.norm = nn.LayerNorm(4)  # final norm: text, but no config names it
    outer.model = decoder
    outer.visual = _Tower(2)
    outer.mlp_AR = nn.Linear(4, 4)
    outer.lm_head = nn.Linear(4, 4)
    return outer


def _native_16() -> nn.Module:
    """PaddlePaddle/PaddleOCR-VL-1.6: everything nested under `model`.

    The paths the real checkpoint produced through
    `AutoModelForImageTextToText.from_config`: `model.language_model.layers`,
    `model.language_model.embed_tokens`, `model.visual`, `model.projector`.
    """
    outer = nn.Module()
    language = nn.Module()
    language.embed_tokens = nn.Embedding(8, 4)
    language.layers = nn.ModuleList(_Block() for _ in range(2))
    language.norm = nn.LayerNorm(4)  # final norm: text, but no config names it
    outer.model = nn.Module()
    outer.model.language_model = language
    outer.model.visual = _Tower(2)
    outer.model.projector = nn.Linear(4, 4)
    outer.lm_head = nn.Linear(4, 4)
    return outer


_BOTH = pytest.mark.parametrize(
    "factory", [_original_release, _native_16], ids=["original", "1.6"]
)


def test_model_layout_reads_the_original_release():
    layout = model_layout(_original_release())
    assert layout == ModelLayout(
        "model", "model.layers.", "model.embed_tokens", "lm_head"
    )


def test_model_layout_reads_the_1_6_release():
    """The nesting is the whole difference between the two checkpoints."""
    layout = model_layout(_native_16())
    assert layout == ModelLayout(
        "model.language_model",
        "model.language_model.layers.",
        "model.language_model.embed_tokens",
        "lm_head",
    )


@_BOTH
def test_lora_targets_never_reach_the_vision_tower(factory):
    """The tower's attention shares the projection names, so screening is by path.

    Suffix matching would adapt 466 M frozen parameters and the run would OOM;
    this is the invariant that makes a bare list of module suffixes safe.
    """
    model = factory()
    layout = model_layout(model)
    paths = lora_target_paths(model, LoraConfig())
    assert paths
    assert all(".visual." not in path for path in paths), paths
    projections = [p for p in paths if p not in (layout.embed_path, layout.head_path)]
    assert projections and all(p.startswith(layout.decoder_prefix) for p in projections)
    assert layout.embed_path in paths and layout.head_path in paths


@_BOTH
def test_freezing_keeps_everything_off_the_text_path_frozen(factory):
    """Vision *and* projector frozen, decoder trainable, on either layout.

    Asserted as a predicate rather than a list of names, because 1.6 renamed
    both (`model.visual`, `model.projector` instead of `visual`, `mlp_AR`) and a
    prefix that matched nothing would freeze nothing while still passing.
    """
    model = factory()
    frozen, trainable = freeze_vision(model)
    assert frozen > 0
    assert trainable > 0
    layout = model_layout(model)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is on_text_path(name, layout), name
    # The decoder's final norm is text: no config names it, but reading it as
    # "outside the text path" would freeze it (and make assert_frozen flag it).
    norm = f"{layout.decoder_owner}.norm.weight"
    assert dict(model.named_parameters())[norm].requires_grad


def test_assert_frozen_catches_a_tower_parameter_that_leaked():
    """The check has to fire on the 1.6 layout, where the old prefixes match nothing."""
    model = _native_16()
    freeze_vision(model)
    tower = next(n for n, _ in model.named_parameters() if n.startswith("model.visual"))
    dict(model.named_parameters())[tower].requires_grad_(True)
    with pytest.raises(AssertionError, match="frozen vision path"):
        assert_frozen(model)


def test_parameter_groups_rate_the_1_6_head_and_embedding_slowly():
    """A prefix that matched nothing put them in the fast group - silently."""
    model = _native_16()
    freeze_vision(model)
    optim = OptimConfig()
    groups = parameter_groups(model, optim)
    rates = {group["lr"] for group in groups}
    assert rates == {optim.learning_rate, optim.embed_head_learning_rate}
    slow = next(g for g in groups if g["lr"] == optim.embed_head_learning_rate)
    # embed_tokens.weight, lm_head.weight, lm_head.bias
    assert len(slow["params"]) == 3
