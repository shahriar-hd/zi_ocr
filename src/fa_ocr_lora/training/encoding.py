"""Turn one manifest record into the tensors the model is trained on.

The encoding has to satisfy three things at once, and the awkward parts of this
module all come from the third:

1. The prompt must be byte-identical to what an inference caller will send, or
   the adapter learns a format nobody uses. So the prompt comes from the
   model's own ``chat_template.jinja`` via ``apply_chat_template``, never from a
   hand-written f-string.
2. Only the answer may be supervised. Prompt positions - including all 1,260
   image placeholders of an A4 page - are masked to ``-100``, so the loss never
   rewards the model for predicting its own input back.
3. The whole thing must fit a fixed sequence budget on a 4 GB card. The answer
   length is known before the image is resized, so the image is given whatever
   budget the answer leaves rather than a fixed one. That ordering is the only
   reason a 3,266-character transcription and a 1,260-token page can share a
   3,072-token window.

The prompt is built once at construction. The processor is asked to expand the
image placeholder, which is what keeps
:func:`~fa_ocr_lora.training.images.image_tokens` and the real sequence length in
agreement - verified equal on all three source types.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from .config import ImageConfig, ModelConfig
from .images import PIXELS_PER_TOKEN, fit_image, load_region

__all__ = [
    "IGNORE_INDEX",
    "EncodedPrompt",
    "EncodedSample",
    "SampleEncoder",
    "SampleTooLong",
]

#: Label value that cross-entropy ignores. Every prompt position carries it.
IGNORE_INDEX: Final[int] = -100


class SampleTooLong(ValueError):
    """A record cannot be made to fit the sequence budget.

    Raised only under ``on_oversize: skip``; the other policies shrink the image
    or truncate the target instead. The dataset catches it, counts it, and moves
    on, so one pathological record cannot end a run.
    """


@dataclass(frozen=True, slots=True)
class EncodedSample:
    """One encoded training pair, ready for the collator.

    Attributes:
        input_ids: Token ids, prompt followed by answer, shape ``(L,)``.
        labels: ``input_ids`` with every prompt position set to
            :data:`IGNORE_INDEX`, shape ``(L,)``.
        pixel_values: Flattened image patches, shape ``(P, 3, 14, 14)``.
        image_grid_thw: Patch grid as ``(1, 3)``, which the vision tower needs to
            rebuild the 2D layout from the flat patch list.
        supervised_tokens: How many positions carry a real label. The chunked
            loss head sizes its work from this, and the run log reports it.
        image_tokens: How many positions the image occupies.
        mm_token_type_ids: Per-token modality ids (text 0, image 1, video 2)
            for models that compute multimodal RoPE from them, aligned with
            ``input_ids``. ``None`` on models whose processor does not return
            them, which read position from the grid alone.
        shrunk: Whether the image budget was reduced to make the record fit.
        truncated: Whether the target was cut to make the record fit.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    supervised_tokens: int
    image_tokens: int
    mm_token_type_ids: torch.Tensor | None = None
    shrunk: bool = False
    truncated: bool = False

    def __len__(self) -> int:
        """Sequence length in tokens."""
        return int(self.input_ids.shape[0])


@dataclass(frozen=True, slots=True)
class EncodedPrompt:
    """One image and its prompt, encoded for generation.

    The inference-time counterpart of :class:`EncodedSample`, and deliberately not
    the same type: there is no answer, so there are no labels, and a caller that
    accidentally fed one of these to the loss would get a silent no-op rather than
    a type error.

    Attributes:
        input_ids: Prompt token ids with the image placeholder expanded, shape
            ``(L,)``. This is the whole generation prompt - unlike the training
            path, nothing needs slicing off it.
        pixel_values: Flattened image patches, shape ``(P, 3, 14, 14)``.
        image_grid_thw: Patch grid as ``(1, 3)``.
        image_tokens: How many positions the image occupies.
        mm_token_type_ids: The processor's per-token modality ids, aligned with
            ``input_ids``, or ``None`` where the processor returns none.
        width: Width in pixels of the image as fed, after resizing.
        height: Height in pixels of the image as fed, after resizing.
        source_width: Width of the region before resizing, so a caller can report
            coordinates in the units of the file it was handed.
        source_height: Height of the region before resizing.
    """

    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    image_tokens: int
    width: int
    height: int
    source_width: int
    source_height: int
    mm_token_type_ids: torch.Tensor | None = None

    def __len__(self) -> int:
        """Prompt length in tokens."""
        return int(self.input_ids.shape[0])


class SampleEncoder:
    """Encodes manifest records against a loaded processor.

    Construction is cheap but not free - it tokenises the prompt template to
    measure its fixed overhead - so build one per worker and reuse it.

    Attributes:
        processor: The model's ``AutoProcessor``.
        prompt: The rendered user turn, with the image placeholder unexpanded.
        prompt_overhead: Prompt tokens other than image placeholders.
    """

    def __init__(
        self,
        processor: Any,
        *,
        model: ModelConfig,
        image: ImageConfig,
    ) -> None:
        """Build an encoder.

        Args:
            processor: Loaded ``AutoProcessor`` for the base model.
            model: Model section, for the task prompt.
            image: Image section, for the pixel and sequence budgets.

        Raises:
            ValueError: If the rendered prompt does not contain exactly one image
                placeholder, which would mean the chat template changed and the
                sequence arithmetic below no longer holds.
        """
        self.processor = processor
        self._tokenizer = processor.tokenizer
        self._image = image
        self._eos = self._tokenizer.eos_token or "</s>"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": model.task_prompt},
                ],
            }
        ]
        self.prompt: str = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        placeholder = processor.image_token
        if self.prompt.count(placeholder) != 1:
            raise ValueError(
                f"expected exactly one {placeholder} in the rendered prompt, got "
                f"{self.prompt.count(placeholder)}; the chat template has changed"
            )
        # The placeholder is one token before the processor expands it, so the
        # fixed overhead is the raw prompt length minus that single token. This
        # is what lets the image budget be derived from the answer length without
        # tokenising the prompt again per sample.
        raw = self._tokenizer(self.prompt, add_special_tokens=False)["input_ids"]
        self.prompt_overhead: int = len(raw) - 1

    def encode_target(self, text: str) -> list[int]:
        """Tokenise a target transcription, terminated by the EOS token.

        ``add_special_tokens=False`` because the chat template already supplied
        the sentence-start token; letting the tokeniser add another would train
        the model to emit one mid-sequence.

        Args:
            text: Target transcription.

        Returns:
            Token ids of the answer, ending in the EOS id.
        """
        return list(
            self._tokenizer(text + self._eos, add_special_tokens=False)["input_ids"]
        )

    def _image_budget(self, answer_length: int) -> int:
        """Pixels the image may occupy, given how long the answer is.

        Args:
            answer_length: Answer length in tokens.

        Returns:
            A pixel ceiling, at least ``min_pixels``, never above the configured
            ``max_pixels``.
        """
        spare = self._image.max_total_seq_len - self.prompt_overhead - answer_length
        allowed = max(0, spare) * PIXELS_PER_TOKEN
        return max(self._image.min_pixels, min(self._image.max_pixels, allowed))

    def encode(
        self,
        *,
        image_path: Path,
        crop_box: tuple[int, int, int, int] | None,
        text: str,
        record_id: str = "",
    ) -> EncodedSample:
        """Encode one record.

        Args:
            image_path: Page image on disk.
            crop_box: Region to crop, or ``None`` for the whole page.
            text: Target transcription.
            record_id: Identifier, used only in error messages.

        Returns:
            The encoded sample.

        Raises:
            SampleTooLong: If the record exceeds the budget and the configured
                policy is ``skip``.
            OSError: If the image cannot be read.
            ValueError: If the processor's expansion disagrees with the predicted
                token count, which would mean the sequence budget is unsound.
        """
        answer_ids = self.encode_target(text)
        budget = self._image_budget(len(answer_ids))
        shrunk = budget < self._image.max_pixels
        truncated = False

        image = load_region(image_path, crop_box)
        image, predicted = fit_image(
            image, min_pixels=self._image.min_pixels, max_pixels=budget
        )

        total = self.prompt_overhead + predicted + len(answer_ids)
        if total > self._image.max_total_seq_len:
            # The image is already at min_pixels, so only the target can give.
            policy = self._image.on_oversize
            if policy == "skip":
                raise SampleTooLong(
                    f"{record_id or image_path.name}: {total} tokens exceeds "
                    f"max_total_seq_len {self._image.max_total_seq_len}"
                )
            keep = self._image.max_total_seq_len - self.prompt_overhead - predicted
            if keep < 1:
                raise SampleTooLong(
                    f"{record_id or image_path.name}: the image alone needs "
                    f"{predicted + self.prompt_overhead} tokens, leaving no room "
                    f"for a target within {self._image.max_total_seq_len}"
                )
            # Keep the EOS token: an answer that never terminates teaches the
            # model to run on past the end of the page.
            answer_ids = answer_ids[: keep - 1] + answer_ids[-1:]
            truncated = True

        batch = self.processor(text=[self.prompt], images=[image], return_tensors="pt")
        prompt_ids = batch["input_ids"][0]
        actual = int(prompt_ids.shape[0]) - self.prompt_overhead
        if actual != predicted:
            raise ValueError(
                f"{record_id or image_path.name}: predicted {predicted} image "
                f"tokens but the processor produced {actual}; the pixel budget "
                "and the processor's resize rule have diverged"
            )

        answer = torch.tensor(answer_ids, dtype=prompt_ids.dtype)
        input_ids = torch.cat([prompt_ids, answer])
        labels = torch.cat([torch.full_like(prompt_ids, IGNORE_INDEX), answer])
        processor_mm = batch.get("mm_token_type_ids")
        if processor_mm is not None:
            # Same layout as the prompt, which is exactly what the ids align
            # to; the answer positions are produced text, and the padded region
            # the collator adds keeps the same value because the model masks it
            # before the RoPE pass.
            mm_token_type_ids = torch.cat(
                [processor_mm[0], torch.zeros(len(answer), dtype=processor_mm.dtype)]
            )
        else:
            mm_token_type_ids = None
        return EncodedSample(
            input_ids=input_ids,
            labels=labels,
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            supervised_tokens=len(answer_ids),
            image_tokens=predicted,
            mm_token_type_ids=mm_token_type_ids,
            shrunk=shrunk,
            truncated=truncated,
        )

    def encode_prompt(
        self,
        *,
        image_path: Path,
        crop_box: tuple[int, int, int, int] | None = None,
        reserve_tokens: int = 0,
    ) -> EncodedPrompt:
        """Encode an image and the prompt alone, for generation.

        :meth:`encode` sizes the image from the tokenised answer, which is the one
        thing inference cannot know: the answer is what is being asked for. So the
        caller states how many tokens it intends to generate and the image gets
        whatever the sequence budget has left over - which at the default
        ``max_total_seq_len`` and ``eval.max_new_tokens`` is the full
        ``max_pixels``, so a document is read at the resolution the adapter was
        trained on.

        The prompt itself comes from the same cached template string, so the model
        is asked at inference exactly what it was asked during training. That is
        the reason this lives here beside :meth:`encode` rather than in the
        inference package: two renderings of one template drift, and the drift is
        invisible until the character error rate is inexplicably worse in
        production than it was on the validation split.

        Args:
            image_path: Document image on disk.
            crop_box: Region to read, or ``None`` for the whole page.
            reserve_tokens: Sequence budget to keep free for the generated answer.

        Returns:
            The encoded prompt.

        Raises:
            SampleTooLong: If the image at its smallest, plus the prompt, plus the
                reserved answer, cannot fit ``max_total_seq_len``. Unlike training
                there is no target to truncate, so the only remedies are the
                caller's - a smaller ``max_new_tokens`` or a cropped region - and
                they are named in the message.
            OSError: If the image cannot be read.
            ValueError: If the processor's expansion disagrees with the predicted
                token count, which would mean the sequence budget is unsound.
        """
        budget = self._image_budget(reserve_tokens)
        image = load_region(image_path, crop_box)
        source_width, source_height = image.width, image.height
        image, predicted = fit_image(
            image, min_pixels=self._image.min_pixels, max_pixels=budget
        )

        total = self.prompt_overhead + predicted + reserve_tokens
        if total > self._image.max_total_seq_len:
            raise SampleTooLong(
                f"{image_path.name}: the prompt and image need "
                f"{self.prompt_overhead + predicted} tokens and the answer is "
                f"allowed {reserve_tokens}, which exceeds max_total_seq_len "
                f"{self._image.max_total_seq_len}; lower max_new_tokens or read a "
                "region instead of the whole page"
            )

        batch = self.processor(text=[self.prompt], images=[image], return_tensors="pt")
        prompt_ids = batch["input_ids"][0]
        actual = int(prompt_ids.shape[0]) - self.prompt_overhead
        if actual != predicted:
            raise ValueError(
                f"{image_path.name}: predicted {predicted} image tokens but the "
                f"processor produced {actual}; the pixel budget and the "
                "processor's resize rule have diverged"
            )
        prompt_mm = batch.get("mm_token_type_ids")
        return EncodedPrompt(
            input_ids=prompt_ids,
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            image_tokens=predicted,
            mm_token_type_ids=prompt_mm[0] if prompt_mm is not None else None,
            width=image.width,
            height=image.height,
            source_width=source_width,
            source_height=source_height,
        )
