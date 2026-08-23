# Task: Build a Universal Jupyter/Colab Notebook for LoRA Fine-Tuning PaddleOCR-VL on Persian Documents

## Objective
Create a complete, well-commented, cell-by-cell Jupyter Notebook (Python `.ipynb`) to LoRA fine-tune `PaddlePaddle/PaddleOCR-VL` (a 0.9B VLM: NaViT-style dynamic-resolution vision encoder + 2-layer MLP projector + ERNIE-4.5-0.3B decoder, loaded via `transformers.AutoModelForCausalLM` / `transformers.AutoProcessor` with `trust_remote_code=True`).

The core target is to dramatically improve Persian OCR accuracy, focusing on digits (Persian/Arabic/Latin), numeric strings, and administrative letter vocabulary.

The notebook must be **dual-compatible**: fully functional in both **Google Colab (T4 GPU / 16GB VRAM)** for pilot testing and **Ferdowsi Cloud / Simorgh HPC (Nvidia A100 / 40GB VRAM / Linux JupyterLab)** for full-scale production training.

---

## 1. Unified Configuration (Cell 0: Single Source of Truth)
All tunable paths, environment flags, and hyperparameters must live in a single configuration cell at the top. No hardcoded variables in later cells.

```python
# --- Execution Mode & Environment Auto-Detection ---
# RUN_MODE: "PILOT_COLAB" (T4 optimized, pixel budget capped) or "FULL_FERDOWSI" (A100 full capacity)
RUN_MODE = "PILOT_COLAB" 
SMOKE_TEST = False  # If True, restricts dataset to 16 samples and runs 1 step

# --- Paths & Hugging Face Dataset IDs ---
HF_LETTER_DATASET = "shahriarhd/persian-official-letters-synth"  # ~40k admin letter samples
HF_PIXEL_DATASET = "Omarrran/Persian_Pixel"                     # ~31k pages / 110k paragraphs

DRIVE_ROOT = "/content/drive/MyDrive/zilver_ocr"                # Google Drive fallback path
LOCAL_DATA_CACHE = "/content/local_data"                        # Fast NVMe/RAM local storage
OUTPUT_DIR = f"{DRIVE_ROOT}/training_output"                    # Checkpoint & output folder

# --- Model & Hardware Dtype ---
MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
# Auto-selected at runtime: float16 for T4 (Turing), bfloat16 for A100/RTX 3090+ (Ampere+)

# --- LoRA Hyperparameters ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj", 
    "embed_tokens", "lm_head"
]
LR_ATTN_MLP = 2e-4
LR_EMBED_HEAD = 7e-5  # Lower LR for sensitive embedding and output heads

# --- Dynamic Image Resolution & Memory Limits ---
# Colab T4 caps max_pixels to 1.5M to prevent OOM; A100 uses native 2.82M max_pixels
TRAIN_MAX_PIXELS_PILOT = 1_500_000
TRAIN_MAX_PIXELS_FULL = 2_822_400
MAX_TOTAL_SEQ_LEN = 5500

# --- Training Schedule & Sampling Control ---
NUM_EPOCHS = 2
WARMUP_RATIO = 0.08
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
OPTIM = "paged_adamw_8bit"
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
RESUME_FROM_CHECKPOINT = True

# Target sampling ratios across unified dataset
TARGET_MIX = {
    "admin_letter_page": 0.35, 
    "admin_letter_crop": 0.35, 
    "persian_pixel_para": 0.20, 
    "persian_pixel_page": 0.10
}
DIGIT_OVERSAMPLE_MULTIPLIER = 2.0  # Upweight top quartile digit-density samples

```

---

## 2. Environment Setup & Hardware Check

* Install dependencies: `pip install -U transformers accelerate peft bitsandbytes trl pillow jiwer datasets huggingface_hub`
* Detect environment (Google Colab vs. Linux JupyterLab VM). Mount Google Drive if running in Colab.
* **Strict Hardware Rule:** Print GPU name and compute capability. **Do not use Google Colab TPU v5e-1** (custom PyTorch/CUDA kernels in `trust_remote_code=True` are incompatible with Torch-XLA).
* **Automatic Precision Selection:**
* If GPU Architecture >= Ampere (e.g. A100, RTX 3090/4090): Set `TORCH_DTYPE = torch.bfloat16`.
* If GPU Architecture < Ampere (e.g. Colab T4): Set `TORCH_DTYPE = torch.float16`.



---

## 3. Data Pipeline & Hugging Face Hub Integration

### Data Sources

1. **Source A & B — `shahriarhd/persian-official-letters-synth` (~40,000 samples):**
* Load directly via `datasets` or local cache.
* Emit **Page-level pairs**: `{"image": <path/PIL>, "prompt": "OCR:", "target": full_text}`.
* Emit **Element-level pairs (Crops)**: Crop non-empty bounding box entries (padding 5–10px) to emit `{"image": <cropped_PIL>, "prompt": "OCR:", "target": bbox_text}`.


2. **Source C — `Omarrran/Persian_Pixel` (~31k pages / 110k paragraphs):**
* Flexible Loader: Auto-detect companion label formats (`labels.json`, `.gt.txt`, or `.csv`).
* **Domain Protection Sampling:** Sample ~18,000–22,000 paragraphs and ~4,000–6,000 pages (~25k total) to prevent diluting the administrative letter domain signal.



### Unified Pipeline & Stratification

* **Fixed Validation Holdout:** Immediately isolate a fixed holdout validation set (~1,500–2,000 samples) *before* any training. This set must remain identical across Colab pilot tests and A100 full runs.
* **Digit Density Score:** Compute `digit_density` (fraction of Persian/Arabic/Latin digits per text target). Bin samples into density quartiles for weighted oversampling (`DIGIT_OVERSAMPLE_MULTIPLIER`).
* **Source Tagging & Manifest Caching:** Merge sources into `unified_manifest.parquet` with source tags (`admin_letter_page`, `admin_letter_crop`, `persian_pixel_para`, `persian_pixel_page`).

---

## 4. Resolution & Preprocessing (NaViT Dynamic Budgeting)

* Read `min_pixels` and `max_pixels` dynamically from `processor.image_processor`.
* Set effective `max_pixels` limit based on `RUN_MODE`:
* `PILOT_COLAB`: Clip to `TRAIN_MAX_PIXELS_PILOT` (1.5M pixels) for T4 16GB VRAM safety.
* `FULL_FERDOWSI`: Use native `TRAIN_MAX_PIXELS_FULL` (2.82M pixels) for full quality on A100.


* Always preserve native image aspect ratios (never pad-distort or force-square).
* Ensure all images are converted to RGB.

---

## 5. Model & LoRA Architecture Setup

```python
model_id = MODEL_ID

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True,
    torch_dtype=TORCH_DTYPE, 
    attn_implementation="sdpa",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model.gradient_checkpointing_enable()

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=LORA_TARGET_MODULES,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

```

* **Freeze Encoder & Projector:** Explicitly verify that Vision Encoder and Projector are frozen. Only ERNIE-4.5-0.3B decoder LoRA parameters (including `embed_tokens` and `lm_head`) must be trainable. Print trainable vs. total parameters.
* **Differential Learning Rate Groups:** Set `LR_ATTN_MLP` for attention/MLP layers and `LR_EMBED_HEAD` for `embed_tokens`/`lm_head`.
* **Prompt Format & Loss Masking:** Use `processor.apply_chat_template` with `user` role (`[{"type": "image"}, {"type": "text", "text": "OCR:"}]`) + assistant target text. Apply `-100` label masking so cross-entropy loss is computed **only** on completion tokens.

---

## 6. Training & Execution Settings

### Batch Size & Accumulation Presets

* **Colab T4 Mode:** `per_device_train_batch_size = 1`, `gradient_accumulation_steps = 16 to 20`, `dataloader_num_workers = 2`.
* **Ferdowsi A100 Mode:** `per_device_train_batch_size = 2 or 4`, `gradient_accumulation_steps = 4 to 8`, `dataloader_num_workers = 8`.

### Execution Guardrails

* Checkpoint auto-resume: Auto-detect latest checkpoint in `OUTPUT_DIR` and resume if interrupted.
* Wrap file I/O in `try/except` and log skipped records to `skipped_records.log`.
* **Throughput Calibration (Step Check):** Run a 100-step smoke test before full training to measure actual `seconds/step` and verify memory usage.

---

## 7. Evaluation & Metrics

1. **General Text Accuracy:** Compute full validation set CER (Character Error Rate) and WER (Word Error Rate) via `jiwer`.
2. **Digit-Isolated Metrics:** Regex-extract numeric substrings (Persian `۰-۹`, Arabic `٠-٩`, Latin `0-9`) from predictions and ground truth. Report exact-match rate and digit CER.
3. **Confusion Summary:** Print top character substitution errors (e.g., `غ` ↔ `ع`, `آ` ↔ `ا`).

---

## 8. Saving & Export Outputs

1. Save standalone LoRA adapters to `{OUTPUT_DIR}/adapter_only`.
2. Merge adapter into base model (`model.merge_and_unload()`) and save to `{OUTPUT_DIR}/merged_model`.
3. Output a `README.md` into `OUTPUT_DIR` containing Cell 0 config JSON, dataset split counts, and evaluation metrics.
