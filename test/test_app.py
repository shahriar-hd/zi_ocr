"""Tests for the local VLM app's offline paths (no model weights needed)."""

import sys
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
