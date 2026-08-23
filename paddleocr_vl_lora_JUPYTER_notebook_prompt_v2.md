# Task: Build ONE portable JupyterLab notebook for LoRA fine-tuning PaddleOCR-VL on Persian
administrative letters — runs unmodified on BOTH Google Colab (T4, 16GB) and Ferdowsi Cloud
(A100, 40GB)

This supersedes an earlier Colab-only spec. The key change: both environments run plain JupyterLab
(Ferdowsi Cloud is a rented VM with a standard Jupyter stack, not a Colab-specific runtime), so the
notebook must NOT hardcode any `google.colab`-only APIs (like `google.colab.drive.mount` or
`google.colab.userdata`) as hard dependencies — it must detect the environment and branch accordingly,
so the SAME notebook file is used for both the cheap pilot phase (Colab T4) and the final production
run (Ferdowsi A100), with only Cell 0's config differing.

## Two-phase workflow this notebook must support
1. **Pilot phase (Colab T4, 16GB VRAM)**: run on small, stratified subsets (e.g. 3k / 8k / 15k / 25k
   samples) to produce an empirical loss/CER-vs-dataset-size learning curve BEFORE committing to a
   large paid run. Cheap, disposable, fast iteration.
2. **Final phase (Ferdowsi Cloud A100, 40GB VRAM)**: run on the full selected training set (tens of
   thousands of samples) once the pilot phase has picked a data size / hyperparameter combination that
   actually shows a meaningful marginal return.

A single `PHASE = "pilot"` / `"final"` flag in Cell 0 must switch: dataset subset size, batch size,
gradient accumulation, pixel budget, and dtype — nothing else in the notebook should need editing.

## Environment detection (Cell 1, runs before anything else)
```python
import subprocess, torch

def detect_environment():
    try:
        import google.colab  # noqa
        env = "colab"
    except ImportError:
        env = "generic_jupyter"  # covers Ferdowsi Cloud and any other plain JupyterLab host

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    cc_major, cc_minor = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    supports_bf16 = cc_major >= 8  # Ampere+ (A100, RTX 3090/4090, etc.) vs Turing T4 (cc 7.5)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

    print(f"environment={env}  gpu={gpu_name}  vram={vram_gb:.1f}GB  bf16_capable={supports_bf16}")
    return {"env": env, "gpu_name": gpu_name, "supports_bf16": supports_bf16, "vram_gb": vram_gb}

RUNTIME = detect_environment()
```
All later cells read `RUNTIME["env"]`, `RUNTIME["supports_bf16"]`, `RUNTIME["vram_gb"]` to branch
behavior — never assume `google.colab` is importable, always guard with try/except as above.

## Cell 0 — Configuration (single source of truth)
```python
# --- phase switch: the ONLY thing that should change between pilot and final runs ---
PHASE = "pilot"   # "pilot" or "final"

PHASE_CONFIG = {
    "pilot": {
        "target_samples": 15_000,           # override per pilot run: 3000 / 8000 / 15000 / 25000
        "per_device_batch_size": 1,
        "grad_accum_steps": 20,
        "train_max_pixels": 1_500_000,      # conservative, T4-safe ceiling (below the model's true max)
        "num_epochs": 2.5,
    },
    "final": {
        "target_samples": 65_000,           # set from pilot-phase learning-curve conclusions
        "per_device_batch_size": 3,         # A100 40GB can hold more; tune empirically (see Cell 2 note)
        "grad_accum_steps": 6,
        "train_max_pixels": 2_822_400,      # full model ceiling — A100 has headroom, no need to cap
        "num_epochs": 1.75,
    },
}
CFG = PHASE_CONFIG[PHASE]

# --- dataset sources (Hugging Face Hub — see Cell 3 for loading) ---
HF_ADMIN_LETTERS_REPO = "shahriarhd/persian-official-letters-synth"
HF_PERSIAN_PIXEL_REPO = "Omarrran/Persian_Pixel"

# --- stratified target mix (fractions of target_samples, must sum to 1.0) ---
SOURCE_MIX = {
    "admin_letter": 1.00,          # use ALL available admin-letter samples up to target ceiling —
                                    # this is the primary domain signal, never subsample it away
    "persian_pixel_paragraph": None,  # computed to fill remaining budget, see Cell 4 — target ~18-20%
                                       # of total when combined with persian_pixel_page
    "persian_pixel_page": None,       # target ~15-20% of total, computed in Cell 4
}
# NOTE: since admin_letter gets priority at 100% of what's available (up to ~40k), the two
# persian_pixel components split the REMAINING room up to target_samples — implement this as
# "fill remaining budget" logic in Cell 4, not fixed absolute fractions of target_samples.

DIGIT_DENSITY_BINS = 5              # for stratified sampling within each source (see Cell 4)

# --- model / precision (auto-resolved from RUNTIME, but overridable) ---
MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
TORCH_DTYPE = "bfloat16" if RUNTIME["supports_bf16"] else "float16"  # T4 -> fp16, A100/3090/4090 -> bf16

# --- LoRA ---
LORA_R = 16          # bump to 24-32 for the "final" phase ONLY if the pilot learning curve shows
                      # capacity (not data volume) is the bottleneck — document this decision in a
                      # markdown cell if changed
LORA_ALPHA = LORA_R * 2
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
                        "embed_tokens","lm_head"]
LR_ATTN_MLP = 2e-4
LR_EMBED_HEAD = 7e-5

# --- fixed held-out validation set (see Cell 4) ---
VALIDATION_SIZE = 1_800
VALIDATION_SEED = 42     # MUST stay identical across pilot AND final runs so results are comparable —
                          # never change this seed once the first pilot run has used it

# --- paths (environment-aware, resolved in Cell 2) ---
WORK_DIR = "/content/work" if RUNTIME["env"] == "colab" else "/home/jupyter/work"
OUTPUT_DIR = f"{WORK_DIR}/training_output_{PHASE}"
HF_DATA_CACHE = f"{WORK_DIR}/hf_datasets_cache"

# --- other training knobs ---
WARMUP_RATIO = 0.08
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
OPTIM = "paged_adamw_8bit"
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
RESUME_FROM_CHECKPOINT = True
GROUP_BY_LENGTH = True
MAX_TOTAL_SEQ_LEN = 5500
```

## Cell 2 — Environment setup and dependency install
- `pip install -U transformers accelerate peft bitsandbytes trl datasets huggingface_hub pillow jiwer`
- **Do not** call `google.colab.drive.mount` unconditionally — wrap in
  `if RUNTIME["env"] == "colab": from google.colab import drive; drive.mount('/content/drive')`.
- Token handling must also branch:
  ```python
  def get_secret(name: str) -> str:
      if RUNTIME["env"] == "colab":
          from google.colab import userdata
          return userdata.get(name)
      else:
          import os
          val = os.environ.get(name)
          if not val:
              raise RuntimeError(f"Set the {name} environment variable before running on "
                                  f"generic JupyterLab (e.g. Ferdowsi Cloud): export {name}=...")
          return val

  HF_TOKEN = get_secret("HF_TOKEN")
  ```
- Print `nvidia-smi` output and `RUNTIME` dict as a sanity check before proceeding.
- Include a short markdown note: on the "final" phase (A100), after the first real training step,
  read `torch.cuda.max_memory_allocated()` and print it — if it's well under 40GB, this is the cue to
  raise `per_device_batch_size` in Cell 0 for the next run (empirical batch-size tuning, not guessing).

## Cell 3 — Load both datasets from Hugging Face Hub
Both datasets are hosted on the Hub, not on Drive — load them the same way regardless of environment:
```python
from datasets import load_dataset

admin_letters_ds = load_dataset(HF_ADMIN_LETTERS_REPO, split="train", cache_dir=HF_DATA_CACHE,
                                 token=HF_TOKEN)
persian_pixel_ds = load_dataset(HF_PERSIAN_PIXEL_REPO, split="train", cache_dir=HF_DATA_CACHE,
                                 token=HF_TOKEN)
```
- Inspect and print the actual column names/schema of both datasets before assuming structure — do
  not hardcode column names blindly; the notebook should print `admin_letters_ds.features` and
  `persian_pixel_ds.features` and adapt field-name references from that (e.g. the image column might
  be `image`, `png`, or something else; the text column might be `text` or `label`).
- `Omarrran/Persian_Pixel` is expected to contain BOTH page-level and paragraph-level items —
  auto-detect which granularity each row belongs to (e.g. via an existing metadata column if present,
  otherwise via a heuristic on image aspect ratio / pixel count / target-text length — document
  whichever heuristic is used clearly in a markdown cell) and tag each row with
  `source_tag ∈ {"persian_pixel_page", "persian_pixel_paragraph"}`.
- Tag every `admin_letters_ds` row with `source_tag = "admin_letter"`.
- Compute `digit_density` (fraction of Persian-indic/Arabic-indic/Latin digit characters in the target
  text) for every row in both datasets, for use in Cell 4's stratified sampling.

## Cell 4 — Stratified sampling and fixed validation split
```python
import random

def stratified_sample(rows, target_n, n_bins=DIGIT_DENSITY_BINS, seed=VALIDATION_SEED):
    rng = random.Random(seed)
    rows_sorted = sorted(rows, key=lambda r: r["digit_density"])
    bins = [rows_sorted[i::n_bins] for i in range(n_bins)]
    per_bin = target_n // n_bins
    selected = []
    for b in bins:
        selected += rng.sample(b, min(per_bin, len(b)))
    return selected
```
Selection logic per Cell 0's `SOURCE_MIX` intent:
1. First, carve out the FIXED validation set (`VALIDATION_SIZE`, seed=`VALIDATION_SEED`), stratified
   across all three `source_tag` values proportionally to their availability — do this ONCE, and if a
   cached validation-set manifest already exists under `OUTPUT_DIR`'s parent, reuse it rather than
   resampling, so the validation set is byte-identical across pilot and final runs.
2. From the remaining pool (all data minus validation), take ALL available `admin_letter` rows up to
   `CFG["target_samples"]` (this source is never subsampled below what's needed/available since it's
   the primary domain signal).
3. Fill the rest of `CFG["target_samples"]` from `persian_pixel_paragraph` and `persian_pixel_page`
   using `stratified_sample`, roughly in a 55:45 ratio between paragraph:page (adjust in Cell 0 if the
   qualitative check below suggests otherwise).
4. Before finalizing, print a markdown-rendered sample of 5 `persian_pixel` rows (image thumbnail +
   text) so the user can visually confirm whether this dataset is administrative-register text or
   general-purpose text — note in a markdown cell that the SOURCE_MIX ratio between admin letters and
   persian_pixel should be revisited if persian_pixel turns out to be closer in register to the admin
   letters than assumed (this changes the sampling weights that would be sensible).
5. Log final counts per `source_tag` and print the achieved digit-density distribution vs. the full
   pool's distribution, as a sanity check that stratification worked.

## Cell 5 — Preprocessing (pixel-budget-aware resize)
Same as before but reads `CFG["train_max_pixels"]` (which differs by phase) rather than a single fixed
constant, and reads true `min_pixels`/`max_pixels` from the processor at runtime to warn if
`CFG["train_max_pixels"]` exceeds what the model actually supports.

## Cell 6 — Dataset / collate / weighted sampler
Same structure as the earlier spec (variable-resolution image handling, `processor.apply_chat_template`
inspected at runtime, `-100` label masking on prompt tokens, weighted sampler folding in digit-density
oversampling) — parameterize batch/accumulation from `CFG`, not hardcoded.

## Cell 7 — Model + LoRA setup
Same as before, but `torch_dtype` comes from `TORCH_DTYPE` (auto-resolved per GPU in Cell 1), and
`attn_implementation="sdpa"`. Print trainable-vs-total parameter counts. Verify vision
encoder/projector params have `requires_grad=False`.

## Cell 8 — Training loop
`TrainingArguments` with `fp16=(TORCH_DTYPE=="float16")`, `bf16=(TORCH_DTYPE=="bfloat16")`,
`per_device_train_batch_size=CFG["per_device_batch_size"]`,
`gradient_accumulation_steps=CFG["grad_accum_steps"]`, `num_train_epochs=CFG["num_epochs"]`,
`output_dir=OUTPUT_DIR`, plus the other knobs from Cell 0. Resume-from-checkpoint logic identical to
the earlier spec (check `OUTPUT_DIR` for existing checkpoints, auto-resume if `RESUME_FROM_CHECKPOINT`).

## Cell 9 — Evaluation
Same CER/WER + digit-only-substring metric + character-substitution confusion summary as before, run
against the FIXED validation set (same one across pilot/final). Additionally: when `PHASE == "pilot"`,
append this run's `(target_samples, final_val_CER, digit_CER)` to a small CSV/JSON log file in a
location that persists across pilot runs (e.g. `f"{WORK_DIR}/pilot_learning_curve.json"`), so successive
pilot runs at different sizes accumulate into one comparable learning-curve table/plot rather than each
run only printing its own isolated number. Plot this curve (matplotlib) as the final cell output when
in pilot phase.

## Cell 10 — Save outputs
Same merge/save/README logic as before, using `OUTPUT_DIR` (which already differs by `PHASE` per
Cell 0). For the "final" phase, additionally push the merged model (or just the adapter, whichever the
user prefers) to a Hugging Face model repo via `huggingface_hub`, mirroring the pattern used for the
datasets — include this as an optional, clearly-labeled cell that can be skipped.

## General requirements (unchanged from before, restated for completeness)
- No hardcoded magic numbers outside Cell 0 / `PHASE_CONFIG`.
- Wrap all data loading and per-record processing in try/except with logging to
  `f"{OUTPUT_DIR}/skipped_records.log"`.
- Include a `SMOKE_TEST` toggle in Cell 0 that restricts to ~16 samples and runs one training step,
  usable identically on both Colab and Ferdowsi before committing to a real run.
- Every cell has a short markdown explanation above it.
- Nothing in cells 3 onward should contain an environment-specific branch — only Cells 1-2 (detection,
  setup, secrets) and the `WORK_DIR` resolution in Cell 0 are environment-aware; everything else must
  be written to work identically once `RUNTIME` and `CFG` are resolved.
