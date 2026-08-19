# Task: Build a single Google Colab notebook for LoRA fine-tuning PaddleOCR-VL on Persian
administrative letters (T4 GPU)

This is a revision of an earlier spec — incorporates T4-specific precision requirements, explicit
memory-budget knobs, and the user's exact Google Drive folder layout. Build ONE self-contained
`.ipynb` notebook (not a package) — every cell should be copy-paste runnable top to bottom in a fresh
Colab session, and every tunable value must live in a single configuration cell near the top.

## Hardware assumption: Colab free/Pro tier T4 GPU (Turing architecture)
**Critical**: T4 does NOT have hardware acceleration for bfloat16 (that requires Ampere+, e.g. A100).
Use `torch.float16` everywhere (model dtype, `TrainingArguments(fp16=True)`), never bf16. If the
notebook detects a non-T4 GPU (e.g. A100) at runtime via `torch.cuda.get_device_name()`, it may switch
to bf16 automatically — implement this as a runtime check, not a hardcoded assumption.

## Cell 0 — Configuration (single source of truth, all other cells read from this)
```python
# --- paths ---
DRIVE_ROOT = "/content/drive/MyDrive/zilver_ocr"       # contains data/ and manifest.csv
DATA_SUBDIR = "data"                                    # DRIVE_ROOT/data/... matches manifest's
                                                         # relative paths like "data/dataset_x/img.png"
MANIFEST_CSV = f"{DRIVE_ROOT}/manifest.csv"             # existing manifest, relative paths as above
LOCAL_DATA_CACHE = "/content/local_data"                # local copy for fast I/O (see Cell 2)
OUTPUT_DIR = f"{DRIVE_ROOT}/training_output"            # checkpoints saved here (on Drive, survives
                                                         # session disconnects)
PERSIAN_PIXEL_DIR = f"{DRIVE_ROOT}/{DATA_SUBDIR}/persian_pixel"  # adjust to actual folder name

# --- model / precision ---
MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
TORCH_DTYPE = "float16"   # auto-upgraded to bfloat16 at runtime if GPU is Ampere+ (see Cell 1)

# --- LoRA ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
                        "embed_tokens","lm_head"]
LR_ATTN_MLP = 2e-4
LR_EMBED_HEAD = 7e-5      # lower LR for embedding/output-head LoRA params (more sensitive)

# --- memory budget (T4-specific — deliberately below the model's true max_pixels) ---
TRAIN_MAX_PIXELS = 1_500_000   # read true min/max from the processor at runtime (Cell 1) and clip to
                                # this training-time ceiling regardless of what the processor allows
TRAIN_SAFETY_MARGIN = 0.92
MAX_TOTAL_SEQ_LEN = 5500        # hard cap (image tokens + text tokens); longer samples are either
                                 # cropped further or skipped with a logged reason, never silently
                                 # truncated mid-training in a way that could cut off target text

# --- batching ---
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 20
GROUP_BY_LENGTH = True

# --- data mix ---
TARGET_MIX = {"admin_letter_page": 0.35, "admin_letter_crop": 0.35, "persian_pixel": 0.30}
DIGIT_OVERSAMPLE_MULTIPLIER = 2.0   # applied to samples in the top quartile of digit density

# --- training schedule ---
NUM_EPOCHS = 3
WARMUP_RATIO = 0.08
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
OPTIM = "paged_adamw_8bit"
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
RESUME_FROM_CHECKPOINT = True   # auto-detect latest checkpoint under OUTPUT_DIR and resume if present,
                                 # since free-tier Colab sessions can disconnect before training finishes
```

## Cell 1 — Environment setup
- `pip install -U transformers accelerate peft bitsandbytes trl pillow jiwer`
- Mount Drive (`google.colab.drive.mount`).
- Detect GPU name/compute capability; if Ampere+ (compute capability >= 8.0), set `TORCH_DTYPE =
  "bfloat16"` and print a note; otherwise keep `float16`.
- Load `AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)` and read the REAL
  `min_pixels`/`max_pixels` from `processor.image_processor` — print them, and assert
  `TRAIN_MAX_PIXELS <= max_pixels` from Cell 0 (warn if the config value is larger than what the model
  actually supports, and clip automatically).

## Cell 2 — Local disk caching of the dataset (performance-critical)
- Copy `f"{DRIVE_ROOT}/{DATA_SUBDIR}"` to `LOCAL_DATA_CACHE` using `shutil.copytree` (skip if already
  present, with an option to force re-copy). Show a progress indicator (`tqdm` over file count) since
  this can take a while for the first run in a session.
- Also copy `MANIFEST_CSV` locally.
- All downstream path resolution must accept a `data_root` argument and default to
  `LOCAL_DATA_CACHE`, with a fallback to reading directly from Drive if a specific file is missing
  locally (log a warning when this fallback triggers, since it indicates the local cache is
  incomplete/stale).

## Cell 3 — Path resolution helper
```python
import os

def resolve_path(relative_path: str, data_root: str = LOCAL_DATA_CACHE) -> str:
    """
    relative_path examples seen in manifest.csv: 'data/dataset_x/img_001.png'
    data_root is the LOCAL_DATA_CACHE dir, which mirrors DRIVE_ROOT/data (NOT DRIVE_ROOT/data/data).
    Strip a leading 'data/' from relative_path since data_root already points at the data/ level.
    """
    rel = relative_path[len(DATA_SUBDIR)+1:] if relative_path.startswith(DATA_SUBDIR + "/") else relative_path
    full = os.path.join(data_root, rel)
    if not os.path.exists(full):
        drive_fallback = os.path.join(DRIVE_ROOT, DATA_SUBDIR, rel)
        if os.path.exists(drive_fallback):
            print(f"[warn] local cache miss, falling back to Drive for: {rel}")
            return drive_fallback
        raise FileNotFoundError(f"Could not resolve {relative_path} under either {data_root} or Drive")
    return full
```

## Cell 4 — Load and unify all data sources
Reuse the same four-source logic as the earlier spec (rich bbox JSON pages → page-level + element-crop
pairs; flat page-level JSON; existing `manifest.csv` — parse its actual columns, whatever they are,
auto-detecting an image-path column and a text column; persian_pixel directory — auto-detect label
format: sidecar `.txt`/`.gt.txt` files, a `labels.json`/`labels.jsonl`, or a csv/tsv). Tag every record
with a `source_tag` matching the keys in `TARGET_MIX`. Compute a `digit_density` score per record
(fraction of Persian-indic/Arabic-indic/Latin digit characters in the target text). Write the unified
result to a single `unified_manifest.parquet` (or `.jsonl`) under `OUTPUT_DIR` so re-running the
notebook doesn't require re-parsing raw sources every time (check for this cache file first, rebuild
only if missing or if a `FORCE_REBUILD_MANIFEST` flag is set).

## Cell 5 — Preprocessing / resize function
Same pixel-budget-aware resize as before, but parameterized by `TRAIN_MAX_PIXELS` /
`TRAIN_SAFETY_MARGIN` from Cell 0 (not the processor's true max — deliberately more conservative for
training memory safety). Never upscale/downscale to a forced square; always preserve aspect ratio.

## Cell 6 — Dataset / collate / weighted sampler
- `torch.utils.data.Dataset` reading from the unified manifest, resizing images per Cell 5, building
  the prompt via `processor.apply_chat_template` (inspect the template at runtime and print an example
  before training starts, don't assume its exact structure), tokenizing target text, and masking prompt
  tokens with `-100` in `labels`.
- Weighted sampler combining `TARGET_MIX` (source-based reweighting) and `DIGIT_OVERSAMPLE_MULTIPLIER`
  (upweighting high digit-density samples), computed once and passed to the DataLoader.
- Collate function handling variable-length sequences (padding) and the processor's variable-resolution
  image tensors — do not assume a fixed image tensor shape.
- Drop or log-and-skip any sample whose total token length exceeds `MAX_TOTAL_SEQ_LEN`.

## Cell 7 — Model + LoRA setup
- Load base model with `TORCH_DTYPE` (from Cell 1's runtime-resolved value), `attn_implementation="sdpa"`,
  `device_map="auto"`.
- `model.gradient_checkpointing_enable()`.
- Apply `LoraConfig` using `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_TARGET_MODULES` from Cell 0.
- Explicitly freeze/verify the vision encoder and projector are NOT in the trainable parameter set;
  print trainable vs. total parameter counts as a sanity check.
- Build an optimizer with two parameter groups (`LR_ATTN_MLP` vs `LR_EMBED_HEAD`), using `OPTIM` from
  Cell 0 (`paged_adamw_8bit`).

## Cell 8 — Training loop
- `TrainingArguments` (or `SFTTrainer` config) using: `fp16=True` (or `bf16=True` if Cell 1 detected
  Ampere+), `per_device_train_batch_size=PER_DEVICE_BATCH_SIZE`,
  `gradient_accumulation_steps=GRAD_ACCUM_STEPS`, `group_by_length=GROUP_BY_LENGTH`,
  `num_train_epochs=NUM_EPOCHS`, `warmup_ratio=WARMUP_RATIO`, `weight_decay=WEIGHT_DECAY`,
  `lr_scheduler_type=LR_SCHEDULER`, `save_steps=SAVE_STEPS`, `save_total_limit=SAVE_TOTAL_LIMIT`,
  `output_dir=OUTPUT_DIR`.
- Before calling `.train()`, check `OUTPUT_DIR` for an existing checkpoint and pass
  `resume_from_checkpoint=<path>` if `RESUME_FROM_CHECKPOINT` is True and one is found — print clearly
  whether training is starting fresh or resuming, and from which step.
- Log loss every N steps; every `SAVE_STEPS` also run a small qualitative check: decode 3-5 validation
  samples and print prediction vs. ground truth.

## Cell 9 — Evaluation
Full validation-set CER/WER via `jiwer`, plus the digit-only-substring metric (regex-extract
Persian-indic/Arabic-indic/Latin digit runs from prediction and ground truth, report exact-match rate
and CER restricted to those substrings), plus a character-substitution confusion summary (top N most
common single-character substitution pairs, e.g. غ↔ع, آ↔ا) — print this after training and optionally
after each checkpoint if time permits.

## Cell 10 — Save outputs
- `model.merge_and_unload()` → save merged model to `f"{OUTPUT_DIR}/merged_model"`.
- Also save the standalone adapter (pre-merge) to `f"{OUTPUT_DIR}/adapter_only"`.
- Write a short `README.md` into `OUTPUT_DIR` documenting: exact `MODEL_ID` + revision/commit if
  available, the full Cell-0 config dict (dump as JSON), dataset mix actually used (counts per
  `source_tag`), and final validation metrics — for reproducibility.

## General requirements
- Every configurable value lives in Cell 0 only — no magic numbers scattered in later cells.
- Wrap all file I/O, rendering-integration calls, and per-record parsing in try/except with logging to
  a `skipped_records.log` under `OUTPUT_DIR` — a handful of malformed records must never crash the run.
- Include a final "smoke test" toggle (`SMOKE_TEST = True/False` in Cell 0) that, when True, restricts
  the unified manifest to ~16 samples and runs one training step end-to-end — use this to validate the
  whole notebook in under a minute before committing to a full run.
