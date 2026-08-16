# Task: Build a Google Colab notebook for LoRA fine-tuning PaddleOCR-VL on Persian administrative letters

## Objective
Create a complete, well-commented, cell-by-cell Google Colab notebook (Python) that LoRA fine-tunes
`PaddlePaddle/PaddleOCR-VL` (a 0.9B VLM: NaViT-style dynamic-resolution vision encoder + 2-layer MLP
projector + ERNIE-4.5-0.3B decoder, loaded via `transformers.AutoModelForCausalLM` /
`transformers.AutoProcessor` with `trust_remote_code=True`) to improve Persian OCR word/character
accuracy, especially for numeric strings and administrative-letter vocabulary. Training runs on a
single Colab GPU (T4/A100, ~16GB+ VRAM) using bf16 (no 4-bit quantization needed).

## Environment setup (first cells)
- Mount Google Drive (`google.colab.drive`).
- `pip install -U transformers accelerate peft bitsandbytes trl pillow jiwer` (bitsandbytes only as an
  optional fallback path, not required by default since we have enough VRAM for bf16 LoRA).
- Print GPU info (`nvidia-smi`) and confirm bf16 support.

## Input data (all paths configurable via variables at the top of the notebook)

### Source A — Rendered administrative letters (page-level, richly annotated), ~1000 images
JSON records (one JSON object per page, `.jsonl` or list-of-dicts — detect and handle both) shaped like:
```json
{
  "image": "image/synth_000000_p01.png",
  "width": 2480, "height": 3508,
  "sample_id": "synth_000000", "page_index": 0,
  "text": "<full linearized page text with \\n between fields>",
  "label": [
    {"bbox": [x1,y1,x2,y2], "type": "header_company_name", "text": "...", "reading_order": 1},
    {"bbox": [x1,y1,x2,y2], "type": "paragraph", "text": "...", "reading_order": 10},
    ...
  ]
}
```
Build TWO kinds of training pairs from this source:
1. **Page-level pairs**: `{"image": <full page path>, "prompt": "OCR:", "target": record["text"]}`.
2. **Element-level pairs**: for each entry in `label` with non-empty `"text"`, crop the page image to
   `bbox` (with ~5-10px padding, clamped to image bounds) and emit
   `{"image": <cropped image, saved to a cache dir>, "prompt": "OCR:", "target": entry["text"]}`.
   Skip entries with empty text (e.g. pure container boxes like "header_bar", "metadata_block").
   Sort/keep `reading_order` available in case a paragraph-grouping strategy is added later.

### Source B — Simple flat page-level format (already converted for the same 1000 images, or usable
directly for any future batch), shaped like:
```json
{"image": "image/synth_000000_p01.png", "text": "<full page text>", "sample_id": "synth_000000", "page_index": 0}
```
Load directly as page-level pairs (same schema as 1a above). If both Source A and Source B exist for
the same image, prefer Source A's `text` field (it's the same content) and don't duplicate.

### Source C — Plaintext-only letter records (~9000, not yet rendered to images), shaped like:
```json
{
  "sample_id": "plaintext_009961",
  "inputs": {"company_name": "...", "letter_genre": "...", "letter_tone": "...", ...},
  "letter": {
    "company_name": "...", "destination_organization": "...", "letter_number": "۱۲۳/الف/۱۴۰۲",
    "date": "۲۲ خرداد ۱۴۰۲", "national_id": "...", "registration_number": "...",
    "subject": "...", "body": "...", "attachments": "...", "signatory_title": "...",
    "contact_info": {"address": "...", "phone": "...", "email": "...", "website": "..."}
  }
}
```
These are NOT yet images. The notebook must call an existing user-provided rendering function
(assume a Python module `letter_renderer.py` will be uploaded/imported — write the notebook to import
it with a placeholder `# TODO: import your existing page-rendering function here` and a clearly marked
integration point) that takes one `letter` dict and produces a full administrative-letter page image
(with paper texture background and randomized font per sample) plus, ideally, per-field bounding boxes
(same `label` schema as Source A) if the renderer can provide them cheaply — if not available, fall
back to emitting only the flat page-level pair (Source B schema). Batch-render all ~9000 remaining
plaintext records into `.png` files under a Drive output folder plus a matching `.jsonl` manifest,
reusing the same schema as Source A/B so they merge seamlessly into the same training pipeline.
Make the renderer call fail-safe (skip and log a sample if rendering throws).

### Source D — persian_pixel dataset (character/word/line-level crops of varied sizes)
Path and exact annotation format are not fully known in advance — write a flexible loader that:
- Looks for common companion-label patterns in the given directory: a single `labels.json`/`labels.jsonl`
  mapping image filename → text, OR per-image sidecar `.txt` / `.gt.txt` files with the same basename,
  OR a `.csv`/`.tsv` with `filename,text` columns.
- Auto-detects which pattern is present and logs how many (image, text) pairs were successfully paired,
  and how many images had no matching label (log and skip those, don't crash).
- Emits the same `{"image", "prompt": "OCR:", "target": text}` pair schema.

## Data pipeline requirements
- Unify all four sources into one list of `{"image_path", "target_text"}` records, tagging each with its
  `source` (A_page, A_crop, B, C_page, C_crop, D_pixel) for later stratified sampling/reporting.
- Resize every image with a pixel-budget-aware function BEFORE feeding to the processor:
  `min_pixels=147384`, `max_pixels=2822400` (read these from
  `AutoProcessor.from_pretrained(...).image_processor` at runtime rather than hardcoding, in case the
  values differ by checkpoint version). Use a safety margin (~92% of max_pixels) when downscaling.
  Preserve aspect ratio always (never force-square, never pad-distort) — this matches the NaViT native
  dynamic-resolution design.
- Convert all images to RGB.
- Train/validation split (~95/5), stratified by `source` so persian_pixel and admin-letter crops/pages
  are all represented in validation.
- Sampling ratio control: allow a configurable target mix (e.g. ~30% persian_pixel-family, ~70%
  admin-letter-family across page+crop) via a weighted sampler, since raw counts may not match the
  desired mix.
- Digit-density oversampling: compute a simple digit-density score per sample (fraction of characters
  that are Persian/Arabic-indic or Latin digits), and provide an optional weighted oversampling knob to
  upweight high-digit-density samples during training (default multiplier configurable, e.g. 2x for the
  top quartile by digit density) — this directly targets the observed failure mode (numeric string
  corruption).

## Model & LoRA setup
```python
model_id = "PaddlePaddle/PaddleOCR-VL"
# bf16 full precision load (no 4-bit quantization — Colab GPU has enough VRAM for this 0.9B model)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
                                              torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model.gradient_checkpointing_enable()

from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
                     "embed_tokens","lm_head"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```
- Freeze the vision encoder and projector explicitly (only the ERNIE-4.5-0.3B decoder LoRA adapters —
  including embed_tokens/lm_head — should be trainable); print trainable-vs-total parameter counts to
  confirm.
- Use two optimizer param groups with different learning rates: a lower LR for the `embed_tokens`/
  `lm_head` LoRA params, a higher LR for attention/MLP LoRA params (make both configurable constants
  near the top of the notebook).
- Prompt format must match the model's OCR instruction chat template exactly (use
  `processor.apply_chat_template` with role="user", content=[{"type":"image"}, {"type":"text","text":"OCR:"}]
  then append the assistant target text) — inspect the processor's chat template at runtime to confirm
  exact structure before hardcoding it.
- Implement loss masking so the cross-entropy loss is computed ONLY over the target/completion tokens,
  never over the prompt or image tokens (standard `-100` label masking).

## Dataset/Dataloader
- Custom `torch.utils.data.Dataset` that lazily loads + resizes images and tokenizes prompt+target via
  the processor, returning `input_ids`, `attention_mask`, `pixel_values` (and any image-grid metadata
  the processor requires), and `labels` (masked).
- Custom collate function that right-pads variable-length sequences and handles the variable-resolution
  image tensors the processor returns (do NOT assume a fixed image tensor shape across the batch).
- `batch_size=1` or `2` (configurable) with `gradient_accumulation_steps` (configurable, default 8-16)
  to reach an effective batch size, since image token counts vary a lot per sample.

## Training loop
- Use either `transformers.Trainer` or `trl.SFTTrainer`, whichever integrates more cleanly with a
  custom multimodal collator — pick one and justify the choice in a markdown cell.
- Cosine LR schedule with ~5-10% warmup steps.
- Small weight decay (~0.01) on LoRA params to reduce overfitting risk given the moderate dataset size.
- Cap training at 2-3 epochs by default (configurable), with checkpoint saved to Google Drive after
  every N steps (configurable) so Colab disconnects don't lose progress.
- Log training loss and periodically (every N steps) run a small qualitative eval: decode a handful of
  validation samples and print prediction vs. ground truth side by side.

## Evaluation
- Implement CER/WER computation (via `jiwer`) over the full validation set at the end of training and
  at each checkpoint.
- Implement a SEPARATE digit-only metric: extract digit substrings (Persian-indic `۰-۹`, Arabic-indic
  `٠-٩`, and Latin `0-9`) from both prediction and ground truth via regex, and report exact-match rate
  and CER restricted to those substrings — this isolates the numeric-copying failure mode described
  above from general text accuracy.
- Print a small confusion summary of the most common character-level substitution errors (e.g. غ↔ع,
  آ↔ا) to track whether specific orthographic issues are improving across checkpoints.

## Output
- At the end, merge the LoRA adapter into the base model (`model.merge_and_unload()`) and save BOTH
  the merged full model and the standalone adapter weights to Google Drive, in clearly separate
  folders, with a short README cell documenting the exact base model id/revision, LoRA config, and
  training data mix used (for reproducibility).

## General code-quality requirements
- Every cell should have a short markdown explanation above it.
- All paths, hyperparameters (r, alpha, learning rates, epochs, batch size, gradient accumulation,
  digit-oversampling multiplier, target source mix ratios) must be defined as named constants in a
  single configuration cell near the top — nothing hardcoded deep in the pipeline.
- Wrap all file I/O and rendering calls in try/except with logging, so a handful of malformed records
  don't crash the whole data-prep pass — log skipped records with reasons to a `skipped.log` file.
- Include a small "smoke test" cell that runs the full pipeline (data loading → one forward pass →
  one backward pass) on a tiny subset (e.g. 8 samples) before launching full training, to catch shape/
  format errors early without burning a full Colab session.
