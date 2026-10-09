"""Tests for the local VLM app's offline paths (no model weights needed)."""

import sys
from contextlib import nullcontext as _NullContext
from pathlib import Path

sys.path.insert(0, "src")

from fa_ocr_lora.app import (  # noqa: E402
    AppConfig,
    AppResult,
    collect_images,
    default_checkpoint,
    discover_checkpoints,
)
from fa_ocr_lora.app.__main__ import build_app_config  # noqa: E402
from fa_ocr_lora.app.args import parse_args  # noqa: E402


def test_discover_checkpoints_finds_both_adapters():
    found = {p.name for p in discover_checkpoints()}
    assert {"adapter_only", "checkpoint-17374"} <= found


def test_default_checkpoint_prefers_adapter_only():
    assert default_checkpoint().name == "adapter_only"


def test_collect_images_scans_directory():
    images = collect_images(image_dir=Path("data/dataset/images"))
    assert len(images) > 1000
    assert all(p.suffix.lower() == ".png" for p in images[:10])


def test_collect_images_rejects_missing():
    try:
        collect_images(explicit=[Path("does-not-exist.png")])
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_app_result_to_dict_json_roundtrip():
    import json

    record = AppResult(image=Path("/tmp/scan.png"), text="سلام", seconds=1.234,
                       pixels=100, error=None).to_dict()
    assert json.loads(json.dumps(record, ensure_ascii=False))["text"] == "سلام"


def test_build_app_config_defaults_to_16_and_adapter_only():
    args = parse_args(["--image", "x.png"])
    config: AppConfig = build_app_config(args)
    assert config.model_id == "PaddlePaddle/PaddleOCR-VL-1.6"
    assert config.adapter_dir is not None and config.adapter_dir.name == "adapter_only"
    assert config.prompt == "OCR:"
    assert config.quantization == "nf4"


def test_build_app_config_no_adapter():
    args = parse_args(["--image", "x.png", "--no-adapter", "--prompt", "Hi"])
    config = build_app_config(args)
    assert config.adapter_dir is None
    assert config.prompt == "Hi"


def test_run_prompt_sends_both_pixel_edges():
    """Regression: 1.6 requires shortest_edge AND longest_edge (ValueError)."""
    import sys
    import types as _types

    from fa_ocr_lora.app.runner import run_prompt

    seen: dict = {}
    fake_torch = _types.ModuleType("torch")
    fake_torch.inference_mode = lambda: _NullContext()  # type: ignore[attr-defined]
    sys.modules.setdefault("torch", fake_torch)

    class _FakeInputs(dict):
        def to(self, device):
            return self

    class _FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            seen.update(kwargs.get("images_kwargs", {}).get("size", {}))
            assert messages[0]["content"][1] == {"type": "text", "text": "OCR:"}
            return _FakeInputs({"input_ids": _FakeIds()})

        def decode(self, completion, skip_special_tokens=True):
            assert skip_special_tokens
            return "  متن  "

    class _FakeIds:
        shape = (1, 10)

    class _FakeModel:
        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 8
            assert kwargs["do_sample"] is False

            class _Out:
                def __getitem__(self, index):
                    assert index == 0
                    return list(range(14))

            return _Out()

    class _FakeLoader:
        import types as _t

        processor = _FakeProcessor()
        model = _FakeModel()
        config = _t.SimpleNamespace(max_new_tokens=8, prompt="OCR:")
        model_min_pixels = 147384
        device = "cpu"

    from PIL import Image

    image = Image.new("RGB", (64, 64))
    assert run_prompt(_FakeLoader(), image, "OCR:", 1003520) == "متن"
    assert seen == {"shortest_edge": 147384, "longest_edge": 1003520}


def test_run_prompt_clamps_floor_below_retry_budget():
    """An OOM-halved budget below the floor must still form a valid window."""
    import sys
    import types

    from fa_ocr_lora.app.runner import run_prompt

    seen: dict = {}
    fake_torch = types.ModuleType("torch")
    fake_torch.inference_mode = lambda: _NullContext()  # type: ignore[attr-defined]
    sys.modules.setdefault("torch", fake_torch)

    class _FakeInputs(dict):
        def to(self, device):
            return self

    class _FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            seen.update(kwargs.get("images_kwargs", {}).get("size", {}))
            return _FakeInputs({"input_ids": types.SimpleNamespace(shape=(1, 4))})

        def decode(self, completion, skip_special_tokens=True):
            return "x"

    class _FakeModel:
        def generate(self, **kwargs):
            class _Out:
                def __getitem__(self, index):
                    return [0] * 6

            return _Out()

    loader = types.SimpleNamespace(
        processor=_FakeProcessor(), model=_FakeModel(),
        config=types.SimpleNamespace(max_new_tokens=8),
        model_min_pixels=147384, device="cpu",
    )
    from PIL import Image

    run_prompt(loader, Image.new("RGB", (32, 32)), "p", 100, min_budget=147384)
    # budget=100 is below one 28x28 token, so longest clamps to 784 (28*28)
    # and shortest clamps under it: the window stays valid, never inverted.
    assert seen["shortest_edge"] <= seen["longest_edge"] == 784
