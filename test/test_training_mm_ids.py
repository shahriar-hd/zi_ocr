"""Unit tests for the per-token modality ids (multimodal RoPE).

transformers 5's native PaddleOCR-VL-1.6 computes multimodal RoPE from one
modality id per token - text 0, image 1, video 2 - and raises when the image
grid arrives without them. The fixture processor below produces those ids
itself (text 0, the expanded image placeholder 1), which is the same contract
the real processor implements, so the tests pin what *our* code does with them
rather than what transformers does:

* **Alignment.** The ids have to be exactly as long as ``input_ids``: the
  processor's own flags for the prompt, zeros for the answer the encoder
  appends (model output is text, like any prompt text), and zeros again where
  the collator pads (the model masks that region before the RoPE pass, so the
  value there cannot matter).
* **The absent case.** A processor that returns none - every checkpoint older
  than 1.6 - must produce records byte-identical to before the ids existed: no
  field, no batch key, no behaviour change. That case is what keeps the old
  model working through the same code path.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from fa_ocr_lora.training.config import ImageConfig, ModelConfig
from fa_ocr_lora.training.data import DatasetStats, PersianOcrDataset, collate_samples
from fa_ocr_lora.training.encoding import IGNORE_INDEX, SampleEncoder
from fa_ocr_lora.training.images import fit_image

IMAGE_ID = 9011


class _Tokenizer:
    """One id per character; the whole placeholder is one id."""

    IMAGE_ID = IMAGE_ID
    eos_token = "</s>"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        ids: list[int] = []
        cursor = 0
        while True:
            hit = text.find("<image>", cursor)
            chunk = text[cursor:] if hit < 0 else text[cursor:hit]
            ids.extend(100 + (ord(character) % 800) for character in chunk)
            if hit < 0:
                break
            ids.append(self.IMAGE_ID)
            cursor = hit + len("<image>")
        return {"input_ids": ids}


class _Processor:
    """Expands the placeholder with the project's own budget, like its real one.

    The expansion size comes from :func:`fit_image`, the same function the
    encoder will check against - a processor that honours the budget - and the
    modality flags follow the tested-by-transformers convention: 0 for text,
    1 for the image placeholder.
    """

    image_token = "<image>"

    def __init__(self, min_pixels: int, max_pixels: int, with_mm_ids: bool) -> None:
        self.tokenizer = _Tokenizer()
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels
        self._with_mm_ids = with_mm_ids

    def apply_chat_template(
        self, messages: list, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        return f"User: <image> {messages[0]['content'][1]['text']}\n"

    def __call__(self, text: list[str], images: list, return_tensors: str = "pt") -> dict:
        (prompt,) = text
        (image,) = images
        _, tokens = fit_image(
            image, min_pixels=self._min_pixels, max_pixels=self._max_pixels
        )
        raw = self.tokenizer(prompt)["input_ids"]
        item_ids: list[int] = []
        kinds: list[int] = []
        for token in raw:
            if token == _Tokenizer.IMAGE_ID:
                item_ids.extend([IMAGE_ID] * tokens)
                kinds.extend([1] * tokens)
            else:
                item_ids.append(token)
                kinds.append(0)
        patches = tokens * 4  # MERGE_SIZE 2: four 14 px patches per token
        out: dict = {
            "input_ids": torch.tensor([item_ids], dtype=torch.long),
            "pixel_values": torch.zeros(patches, 3, 14, 14),
            "image_grid_thw": torch.tensor([[1, 8, 8]], dtype=torch.long),
        }
        if self._with_mm_ids:
            out["mm_token_type_ids"] = torch.tensor([kinds], dtype=torch.long)
        return out


def _encoder(with_mm_ids: bool) -> SampleEncoder:
    """An encoder against the fake processor, at the default pixel budget."""
    image = ImageConfig()
    processor = _Processor(image.min_pixels, image.max_pixels, with_mm_ids)
    return SampleEncoder(processor, model=ModelConfig(), image=image)


def _image_file(tmp_path: Path, width: int, height: int) -> Path:
    path = tmp_path / f"page_{width}x{height}.png"
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(path)
    return path


def test_encode_attaches_mm_ids_aligned_with_input_ids(tmp_path: Path) -> None:
    """Prompt flags come from the processor; answer and padding read as text."""
    encoder = _encoder(with_mm_ids=True)
    sample = encoder.encode(
        image_path=_image_file(tmp_path, 560, 560),
        crop_box=None,
        text="شماره ثبت",
        record_id="r1",
    )
    assert sample.mm_token_type_ids is not None
    assert tuple(sample.mm_token_type_ids.shape) == tuple(sample.input_ids.shape)

    flags = sample.mm_token_type_ids
    answer = (sample.labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    assert len(answer) > 0
    assert bool((flags[answer] == 0).all())
    # 560x560 snaps to itself at 28 px a side: 20x20 patches = 400 image tokens.
    assert int((flags == 1).sum()) == 400
    assert set(flags.tolist()) <= {0, 1}


def test_encode_prompt_carries_mm_ids_for_generation(tmp_path: Path) -> None:
    """The inference path needs them for the same reason training does."""
    encoder = _encoder(with_mm_ids=True)
    prompt = encoder.encode_prompt(
        image_path=_image_file(tmp_path, 560, 560), crop_box=None, reserve_tokens=64
    )
    assert prompt.mm_token_type_ids is not None
    assert tuple(prompt.mm_token_type_ids.shape) == tuple(prompt.input_ids.shape)
    assert int((prompt.mm_token_type_ids == 1).sum()) == 400


def test_encode_omits_mm_ids_when_the_processor_has_none(tmp_path: Path) -> None:
    """The older checkpoint's records must come out byte-identical to before."""
    encoder = _encoder(with_mm_ids=False)
    sample = encoder.encode(
        image_path=_image_file(tmp_path, 560, 560),
        crop_box=None,
        text="شماره ثبت",
        record_id="r1",
    )
    assert sample.mm_token_type_ids is None
    prompt = encoder.encode_prompt(
        image_path=_image_file(tmp_path, 560, 560), crop_box=None, reserve_tokens=64
    )
    assert prompt.mm_token_type_ids is None


def _dataset(encoder: SampleEncoder, paths: list[Path]) -> PersianOcrDataset:
    dataset = PersianOcrDataset.__new__(PersianOcrDataset)
    import pandas as pd

    dataset._frame = pd.DataFrame(
        [
            {
                "image_path": str(path),
                "crop_box": None,
                "text": "شماره",
                "record_id": f"r{i}",
            }
            for i, path in enumerate(paths)
        ]
    )
    dataset._index = tuple(range(len(paths)))
    dataset._encoder = encoder
    dataset.stats = DatasetStats()
    return dataset


def test_dataset_carries_mm_ids_into_the_record_dict(tmp_path: Path) -> None:
    """The collator can only forward what ``__getitem__`` handed it."""
    paths = [_image_file(tmp_path, 560, 560)]
    with_mm = _dataset(_encoder(with_mm_ids=True), paths)[0]
    without_mm = _dataset(_encoder(with_mm_ids=False), paths)[0]
    assert set(with_mm) == {
        "input_ids",
        "labels",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    }
    assert set(without_mm) == {
        "input_ids",
        "labels",
        "pixel_values",
        "image_grid_thw",
    }


def test_collate_pads_mm_ids_with_zero_like_the_mask(tmp_path: Path) -> None:
    """Padding reads as text, and the key reaches the model kwargs."""
    encoder = _encoder(with_mm_ids=True)
    paths = [_image_file(tmp_path, 560, 560), _image_file(tmp_path, 840, 420)]
    features = [_dataset(encoder, [path])[0] for path in paths]
    batch = collate_samples(features, pad_token_id=0, dtype=torch.float32)
    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    }
    mm = batch["mm_token_type_ids"]
    assert tuple(mm.shape) == tuple(batch["input_ids"].shape)
    padded = batch["attention_mask"] == 0
    assert int(padded.sum()) > 0  # the two images tokenise to different lengths
    assert bool((mm[padded] == 0).all())


def test_collate_omits_mm_ids_when_samples_lack_them(tmp_path: Path) -> None:
    """The older checkpoint's forward must never see this key."""
    encoder = _encoder(with_mm_ids=False)
    features = [_dataset(encoder, [_image_file(tmp_path, 560, 560)])[0]]
    batch = collate_samples(features, pad_token_id=0, dtype=torch.float32)
    assert "mm_token_type_ids" not in batch
    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values",
        "image_grid_thw",
    }


def test_image_sizes_stay_inside_the_budget() -> None:
    """The two test images sit inside the default pixel bounds, unscaled."""
    image = ImageConfig()
    for width, height in ((560, 560), (840, 420)):
        assert image.min_pixels <= width * height <= image.max_pixels
        assert width % 28 == 0 and height % 28 == 0
