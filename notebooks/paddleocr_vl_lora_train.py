# %% [markdown]
# # LoRA fine-tuning PaddleOCR-VL 1.6 for Persian administrative letters
#
# One notebook, two machines. It runs unmodified on **Google Colab (T4, 16 GB, fp16)** for the
# cheap pilot phase and on **Ferdowsi Cloud / any plain JupyterLab VM (A100, 40 GB, bf16)** for the
# final production run. The only thing you edit between runs is `PHASE` in **Cell 0**.
#
# | Phase | Machine | What it is for |
# | --- | --- | --- |
# | `"pilot"` | Colab T4 | Small stratified subsets (3k / 8k / 15k / 25k) to build an empirical CER-vs-data-size learning curve *before* paying for a long run. |
# | `"final"` | Ferdowsi A100 | The full selected training set, once the pilot curve shows where the marginal return dies. |
#
# ### Execution order
# Cells are numbered as in the specification, but **Cell 1 (environment detection) runs before
# Cell 0 (configuration)** because the configuration reads `RUNTIME` to pick dtype, paths and
# dataloader workers. Run the cells top to bottom as they appear in this notebook.
#
# ### What you must provide
# * A GPU runtime. **Not a TPU** — the model is loaded with `trust_remote_code=True` and its custom
#   CUDA/SDPA paths are incompatible with Torch-XLA.
# * `HF_TOKEN`. On Colab: *Secrets* panel (key icon) → add `HF_TOKEN`. On plain JupyterLab:
#   `export HF_TOKEN=hf_...` **before** launching Jupyter.
#
# ### Model being adapted
# `PaddlePaddle/PaddleOCR-VL` — 0.9B parameters: a NaViT-style dynamic-resolution SigLIP vision
# encoder, a 2-layer MLP projector (`mlp_AR`), and an ERNIE-4.5-0.3B decoder. Only the decoder is
# adapted; the vision tower and projector stay frozen.

# %% [markdown]
# ## Cell 1 — Environment detection (runs first)
#
# Decides *where* we are and *what precision the GPU can do*, and nothing else. `google.colab` is
# probed inside a `try/except` so the notebook never hard-depends on it — Ferdowsi Cloud is a plain
# Jupyter stack where that import simply fails.
#
# `supports_bf16` is derived from compute capability: Ampere and newer (cc ≥ 8.0 — A100, RTX
# 3090/4090) get bf16; Turing (cc 7.5 — T4) falls back to fp16.
#
# If `torch` is not importable yet, this cell will say so and you should run **Cell 2** first, then
# re-run **Cell 1** and **Cell 0**.

# %%
"""Cell 1 - environment and hardware detection."""

import platform
import subprocess
import sys
from typing import Any


def detect_environment() -> dict[str, Any]:
    """Detect the host environment and the GPU's precision capability.

    Returns:
        Mapping with keys ``env`` (``"colab"`` or ``"generic_jupyter"``), ``gpu_name``,
        ``supports_bf16``, ``vram_gb``, ``compute_capability`` and ``torch_available``.
    """
    try:
        import google.colab  # noqa: F401

        env = "colab"
    except ImportError:
        env = "generic_jupyter"  # Ferdowsi Cloud and any other plain JupyterLab host

    try:
        import torch
    except ImportError:
        print(
            "torch is not importable yet. Run Cell 2 (dependency install), then re-run "
            "Cell 1 and Cell 0."
        )
        return {
            "env": env,
            "gpu_name": "none",
            "supports_bf16": False,
            "vram_gb": 0.0,
            "compute_capability": (0, 0),
            "torch_available": False,
        }

    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "none"
    cc_major, cc_minor = torch.cuda.get_device_capability(0) if has_cuda else (0, 0)
    supports_bf16 = cc_major >= 8  # Ampere+ (A100, RTX 3090/4090) vs Turing T4 (cc 7.5)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if has_cuda else 0.0

    print(
        f"environment={env}  gpu={gpu_name}  cc={cc_major}.{cc_minor}  "
        f"vram={vram_gb:.1f}GB  bf16_capable={supports_bf16}"
    )
    if not has_cuda:
        print(
            "WARNING: no CUDA device visible. On Colab pick Runtime > Change runtime type > "
            "T4 GPU. Do NOT pick a TPU: the trust_remote_code model does not run under Torch-XLA."
        )
    if "TPU" in gpu_name.upper():
        raise RuntimeError(
            "A TPU runtime was detected. PaddleOCR-VL ships custom PyTorch modules via "
            "trust_remote_code and does not run under Torch-XLA. Switch to a GPU runtime."
        )
    return {
        "env": env,
        "gpu_name": gpu_name,
        "supports_bf16": supports_bf16,
        "vram_gb": vram_gb,
        "compute_capability": (cc_major, cc_minor),
        "torch_available": True,
    }


RUNTIME = detect_environment()
print(f"python={sys.version.split()[0]}  platform={platform.platform()}")

# %% [markdown]
# ## Cell 0 — Configuration (single source of truth)
#
# Every tunable in the notebook lives here. Later cells read `CFG`, `RUNTIME` and the module-level
# constants; they never hardcode a number of their own.
#
# **The phase switch is the only line you normally edit.** It controls sample budget, batch size,
# gradient accumulation, the pixel budget and epoch count — and, through `RUNTIME`, the dtype.
#
# A few knobs are additions beyond the specification, each marked `# [addition]` with the reason:
# they exist because the specification's prose implies a decision that needs a name (for example
# what to do with a sample whose image + text exceeds `MAX_TOTAL_SEQ_LEN`).

# %%
"""Cell 0 - configuration. No other cell may introduce a magic number."""

# --- phase switch: the ONLY thing that should change between pilot and final runs ---
PHASE = "pilot"  # "pilot" or "final"

PHASE_CONFIG: dict[str, dict[str, Any]] = {
    "pilot": {
        "target_samples": 15_000,  # override per pilot run: 3000 / 8000 / 15000 / 25000
        "per_device_batch_size": 1,
        "grad_accum_steps": 20,
        "train_max_pixels": 1_500_000,  # conservative, T4-safe ceiling (below the model's true max)
        "num_epochs": 2.5,
        # [addition] generation-based evaluation is ~1-3 s/sample; capping it keeps the pilot loop
        # short. The validation SET is always the same 1800 rows; this only caps how many of them
        # are decoded. Keep it identical across pilot runs or the learning curve is not comparable.
        "eval_max_samples": 300,
    },
    "final": {
        "target_samples": 65_000,  # set from pilot-phase learning-curve conclusions
        "per_device_batch_size": 3,  # A100 40GB can hold more; tune empirically (see Cell 2 note)
        "grad_accum_steps": 6,
        "train_max_pixels": 2_822_400,  # full model ceiling - A100 has headroom, no need to cap
        "num_epochs": 1.75,
        "eval_max_samples": None,  # None = the whole fixed validation set
    },
}
CFG = PHASE_CONFIG[PHASE]

SMOKE_TEST = False  # True -> ~16 samples, one optimizer step. Same behaviour on both machines.
SMOKE_TEST_SAMPLES = 16
SMOKE_TEST_STEPS = 1
SMOKE_TEST_EVAL_SAMPLES = 4

# --- dataset sources (Hugging Face Hub - see Cell 3 for loading) ---
HF_ADMIN_LETTERS_REPO = "shahriarhd/persian-official-letters-synth"
HF_PERSIAN_PIXEL_REPO = "Omarrran/Persian_Pixel"

# [addition] Both repos are multi-config. The admin-letters exporter writes one config per artifact
# ("images", "plaintext", "markdown", "pdf", "annotations", "metadata"); only "images" carries
# image+text pairs. Persian_Pixel exposes one config per granularity, which is what lets Cell 3
# tag page vs paragraph exactly instead of guessing from aspect ratio.
ADMIN_LETTERS_CONFIGS = ["images", None]  # tried in order; None = repo default config
ADMIN_LETTERS_SPLITS = ["train"]  # the exporter's own group-aware "test" split is left untouched
PERSIAN_PIXEL_CONFIGS = {"persian_pixel_paragraph": "paragraph", "persian_pixel_page": "page"}
PERSIAN_PIXEL_SAMPLE_TYPE_COLUMN = "sample_type"  # fallback tag source if the configs disappear

# [addition] escape hatch: if the Hub repo is empty or unreachable, point this at a local export
# directory (a folder of parquet files, or a saved dataset) and Cell 3 will load that instead.
ADMIN_LETTERS_LOCAL_DIR: str | None = None

# [addition] column auto-detection candidates, in preference order. Cell 3 prints the real
# ``.features`` of each dataset and resolves against these lists rather than assuming a name.
IMAGE_COLUMN_CANDIDATES = ["image", "png", "img", "picture", "image_bytes"]
TEXT_COLUMN_CANDIDATES = ["text", "label", "ground_truth", "transcription", "target", "caption"]
WIDTH_COLUMN_CANDIDATES = ["width", "image_width"]
HEIGHT_COLUMN_CANDIDATES = ["height", "image_height"]

# --- stratified target mix (fractions of target_samples, must sum to 1.0) ---
SOURCE_MIX = {
    "admin_letter": 1.00,  # use ALL available admin-letter samples up to target ceiling -
    # this is the primary domain signal, never subsample it away
    "persian_pixel_paragraph": None,  # computed to fill remaining budget, see Cell 4 - target
    # ~18-20% of total when combined with persian_pixel_page
    "persian_pixel_page": None,  # target ~15-20% of total, computed in Cell 4
}
# NOTE: since admin_letter gets priority at 100% of what's available (up to ~40k), the two
# persian_pixel components split the REMAINING room up to target_samples - implemented as
# "fill remaining budget" logic in Cell 4, not fixed absolute fractions of target_samples.
PIXEL_PARAGRAPH_PAGE_RATIO = (0.55, 0.45)  # paragraph : page split of the remaining budget

DIGIT_DENSITY_BINS = 5  # for stratified sampling within each source (see Cell 4)

# [addition] digit character classes, used by both the density score (Cell 3) and the digit-only
# metrics (Cell 9), so the two can never disagree about what a digit is.
DIGIT_CHAR_RANGES = [
    ("0", "9"),  # Latin
    ("۰", "۹"),  # Persian (extended Arabic-Indic)
    ("٠", "٩"),  # Arabic-Indic
]
DIGIT_OVERSAMPLE_MULTIPLIER = 2.0  # weight applied to the top digit-density quantile in Cell 6
DIGIT_OVERSAMPLE_QUANTILE = 0.75  # "top quartile"

# --- model / precision (auto-resolved from RUNTIME, but overridable) ---
MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
TORCH_DTYPE = "bfloat16" if RUNTIME["supports_bf16"] else "float16"  # T4 -> fp16, A100 -> bf16
ATTN_IMPLEMENTATION = "sdpa"

# [addition] task prompts, verbatim from the model card. Training uses OCR only; the others are
# here so the adapter's prompt format matches the base model's at inference time.
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
TRAIN_TASK = "ocr"

# --- LoRA ---
LORA_R = 16  # bump to 24-32 for the "final" phase ONLY if the pilot learning curve shows
# capacity (not data volume) is the bottleneck - document that decision if changed
LORA_ALPHA = LORA_R * 2
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "embed_tokens",
    "lm_head",
]
# [addition] the vision tower uses the SAME leaf names (q_proj/k_proj/v_proj) as the decoder, so a
# bare suffix list would inject adapters into the encoder we are supposed to be freezing. Cell 7
# resolves the suffixes above to full module paths and drops anything under these prefixes.
LORA_EXCLUDE_MODULE_PREFIXES = ["visual", "mlp_AR"]
LR_ATTN_MLP = 2e-4
LR_EMBED_HEAD = 7e-5

# --- fixed held-out validation set (see Cell 4) ---
VALIDATION_SIZE = 1_800
VALIDATION_SEED = 42  # MUST stay identical across pilot AND final runs so results are comparable -
# never change this seed once the first pilot run has used it

# --- paths (environment-aware, resolved in Cell 2) ---
WORK_DIR = "/content/work" if RUNTIME["env"] == "colab" else "/home/jupyter/work"
OUTPUT_DIR = f"{WORK_DIR}/training_output_{PHASE}"
HF_DATA_CACHE = f"{WORK_DIR}/hf_datasets_cache"

# [addition] artefacts that must survive across phases live in WORK_DIR, not OUTPUT_DIR (which is
# per-phase): the validation manifest and the accumulating pilot learning curve.
VALIDATION_MANIFEST_PATH = f"{WORK_DIR}/validation_manifest.json"
LEARNING_CURVE_PATH = f"{WORK_DIR}/pilot_learning_curve.json"
SKIPPED_RECORDS_LOG = f"{OUTPUT_DIR}/skipped_records.log"
TRAIN_MANIFEST_PATH = f"{OUTPUT_DIR}/unified_manifest.jsonl"

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

# [addition] knobs the specification implies but does not name
LOGGING_STEPS = 10
GRADIENT_CHECKPOINTING = True  # mandatory on the T4; kept on for the A100 to buy batch size
GRADIENT_CHECKPOINTING_REENTRANT = False  # non-reentrant plays correctly with frozen submodules
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
MAX_GRAD_NORM = 1.0
DATALOADER_WORKERS = 2 if RUNTIME["vram_gb"] < 24 else 8  # T4 preset vs A100 preset
LOSS_EVAL_ENABLED = True  # cheap teacher-forced loss on a validation subset during training
LOSS_EVAL_SAMPLES = 200  # kept small: full-set loss eval every SAVE_STEPS would dominate runtime
SEED = 3407  # global training seed (data order, dropout); distinct from VALIDATION_SEED
# What to do when image tokens + text tokens exceed MAX_TOTAL_SEQ_LEN:
#   "shrink_image" - lower the pixel budget for that one sample so the full target still fits.
#                    Preferred: truncating an OCR target teaches the model to stop early.
#   "skip"         - drop the sample and log it.
#   "truncate"     - keep the image, cut the target (not recommended, listed for completeness).
ON_OVERSIZE = "shrink_image"
MAX_GETITEM_RETRIES = 8  # bounded fallback walk when a record fails to decode
# [addition] The vision tower concatenates every image in a forward pass into ONE sequence and only
# honours per-image boundaries (`cu_seqlens`) on the flash-attention path. Under `sdpa` there is no
# block mask, so with per_device_batch_size > 1 the patches of different pages attend to each other -
# silently, and differently from single-image inference. Left False, Cell 8 folds any per-device
# batch > 1 into gradient accumulation so the effective batch is unchanged and every forward pass
# sees exactly one page. Set True only with flash_attention_2 installed (Ampere+ only).
ALLOW_MULTI_IMAGE_BATCH = False
CHARS_PER_TOKEN_ESTIMATE = 2.5  # only used if the tokenizer is not a fast tokenizer

# --- evaluation ---
EVAL_MAX_NEW_TOKENS = 1024
EVAL_BATCH_SIZE = 1  # variable-resolution images: one at a time is the safe default
CONFUSION_TOP_N = 25
# Persian/Arabic look-alikes. Cell 9 reports CER both raw and after this normalisation, so you can
# see how much of the error is real versus orthographic variance.
PERSIAN_NORMALIZATION_MAP = {
    "ي": "ی",  # Arabic yeh -> Persian yeh
    "ى": "ی",  # alef maksura -> Persian yeh
    "ك": "ک",  # Arabic kaf -> Persian keheh
    "٠": "0",
    "١": "1",
    "٢": "2",
    "٣": "3",
    "٤": "4",
    "٥": "5",
    "٦": "6",
    "٧": "7",
    "٨": "8",
    "٩": "9",
    "۰": "0",
    "۱": "1",
    "۲": "2",
    "۳": "3",
    "۴": "4",
    "۵": "5",
    "۶": "6",
    "۷": "7",
    "۸": "8",
    "۹": "9",
    "‌": " ",  # ZWNJ -> space
    "ـ": "",  # tatweel
}

# --- secrets / publishing ---
REQUIRE_HF_TOKEN = True
SAVE_ADAPTER = True  # the artefact that matters: a few dozen MB of LoRA weights
SAVE_MERGED_MODEL = True  # standalone ~1.8GB checkpoint; set False if Drive space is tight
PUSH_TO_HUB = False  # opt-in, final phase only - see the last cell
HF_MODEL_REPO = "shahriarhd/paddleocr-vl-persian-lora"
PUSH_WHAT = "adapter"  # "adapter" or "merged"

# --- dependency pins (see Cell 2 for why these are pinned, not floating) ---
PIP_REQUIREMENTS = [
    "transformers==4.57.6",
    "peft>=0.17,<0.19",
    "accelerate>=1.6,<2",
    "datasets>=3.6,<6",
    "bitsandbytes>=0.45,<1",
    "einops>=0.8",
    "jiwer>=3.0",
    "pillow>=10",
    "matplotlib>=3.7",
]

assert PHASE in PHASE_CONFIG, f"PHASE must be one of {list(PHASE_CONFIG)}"
assert ON_OVERSIZE in {"shrink_image", "skip", "truncate"}
assert abs(sum(PIXEL_PARAGRAPH_PAGE_RATIO) - 1.0) < 1e-9
assert PUSH_WHAT in {"adapter", "merged"}
assert not PUSH_TO_HUB or (SAVE_ADAPTER if PUSH_WHAT == "adapter" else SAVE_MERGED_MODEL), (
    f"PUSH_WHAT={PUSH_WHAT!r} needs the matching SAVE_* flag enabled"
)

print(f"PHASE={PHASE}  dtype={TORCH_DTYPE}  workers={DATALOADER_WORKERS}  smoke_test={SMOKE_TEST}")
for key, value in CFG.items():
    print(f"  CFG.{key} = {value}")
print(f"WORK_DIR={WORK_DIR}")
print(f"OUTPUT_DIR={OUTPUT_DIR}")

# %% [markdown]
# ## Cell 2 — Dependencies, secrets, directories
#
# ### Why the versions are pinned
# PaddleOCR-VL is loaded with `trust_remote_code=True`, so the model code comes from the Hub repo,
# not from `transformers`. That code imports `transformers` internals directly —
# `masking_utils.create_causal_mask`, `modeling_layers.GradientCheckpointingLayer`,
# `integrations.use_kernel_forward_from_hub`, `utils.generic.check_model_inputs` — and its config
# declares `transformers_version: 4.55.0`. PyPI currently serves `transformers` 5.x, where those
# import paths have moved. A bare `pip install -U transformers` therefore breaks the model load
# with an `ImportError` that looks like a model bug. The pin `transformers==4.57.6` is the newest
# 4.x that still resolves every one of those imports.
#
# `trl` is **not** installed: this notebook trains with `transformers.Trainer` directly, and current
# `trl` releases pull a `transformers` 5.x resolution that would silently undo the pin.
# `einops` **is** installed although the specification's list omits it — the vision embedding module
# imports it at forward time, so without it the first training step raises `ModuleNotFoundError`.
#
# ### Empirical batch-size tuning (final phase)
# After the first real optimizer step, Cell 8 prints `torch.cuda.max_memory_allocated()`. On the
# A100 that number is the tuning signal: if peak allocation sits well under 40 GB, raise
# `per_device_batch_size` in `PHASE_CONFIG["final"]` for the next run and lower `grad_accum_steps`
# by the same factor to keep the effective batch constant. Do not guess the batch size upward
# without reading that number — the sequence length here is dominated by image tokens (up to 3600
# per page) and varies per sample, so peak memory is not a simple multiple of the batch size.

# %%
"""Cell 2a - dependency install. Restart the kernel and re-run from Cell 1 if pip says to."""

# fmt: off
_pip_args = " ".join(f'"{req}"' for req in PIP_REQUIREMENTS)
get_ipython().run_line_magic("pip", f"install -q -U {_pip_args}")  # noqa: F821
# fmt: on

# %% [markdown]
# ### Cell 2b — Versions, secrets, directories, logging
#
# Separate from the install so a kernel restart lands on a clean boundary: if the cell above upgraded
# `transformers` on Colab, pip will tell you to restart. Do it, then re-run **Cell 1 → Cell 0 → this
# cell** and skip the install. Everything below re-derives itself from `RUNTIME` and Cell 0, so a
# restart costs nothing but the download you already have cached.
#
# `get_secret()` reads Colab's `userdata` when it exists and falls back to the process environment,
# which is what makes the same cell work on Ferdowsi Cloud (`export HF_TOKEN=...` before `jupyter
# lab`). The token is never printed — only whether it was found and how long it is.

# %%
"""Cell 2b - version verification, secrets, mounts, directories, logging."""

import importlib
import json
import logging
import os
import subprocess

_EXPECTED_TRANSFORMERS_MAJOR = 4


def report_versions() -> dict[str, str]:
    """Import every runtime dependency and print its resolved version.

    Returns:
        Mapping of package name to version string (``"MISSING"`` when the import failed).

    Raises:
        RuntimeError: If the installed ``transformers`` major version cannot load the model's
            remote code.
    """
    packages = [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "bitsandbytes",
        "einops",
        "jiwer",
        "PIL",
        "matplotlib",
    ]
    versions: dict[str, str] = {}
    for name in packages:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError as exc:
            versions[name] = "MISSING"
            print(f"  {name}: MISSING ({exc})")
    for name, version in versions.items():
        if version != "MISSING":
            print(f"  {name}: {version}")

    transformers_version = versions.get("transformers", "MISSING")
    if transformers_version == "MISSING":
        raise RuntimeError("transformers failed to import; re-run Cell 2a.")
    major = int(transformers_version.split(".")[0])
    if major != _EXPECTED_TRANSFORMERS_MAJOR:
        raise RuntimeError(
            f"transformers {transformers_version} is installed, but PaddleOCR-VL's remote code "
            f"needs the {_EXPECTED_TRANSFORMERS_MAJOR}.x import layout. Re-run Cell 2a, then "
            "restart the kernel and run again from Cell 1."
        )
    return versions


VERSIONS = report_versions()


def get_secret(name: str, required: bool = True) -> str | None:
    """Read a secret from the Colab secrets store or the process environment.

    Args:
        name: Secret name, e.g. ``"HF_TOKEN"``.
        required: Raise instead of returning ``None`` when the secret is absent.

    Returns:
        The secret value, or ``None`` when it is absent and ``required`` is ``False``.

    Raises:
        RuntimeError: If the secret is absent and ``required`` is ``True``.
    """
    value: str | None = None
    if RUNTIME["env"] == "colab":
        try:
            from google.colab import userdata

            value = userdata.get(name)
        except ImportError:
            value = None
        except Exception as exc:  # google.colab raises SecretNotFoundError / NotebookAccessError
            print(f"Colab secret {name!r} unavailable ({type(exc).__name__}); trying os.environ.")
            value = None
    if not value:
        value = os.environ.get(name)
    if not value:
        if required:
            raise RuntimeError(
                f"Set the {name} environment variable before running on generic JupyterLab "
                f"(e.g. Ferdowsi Cloud): export {name}=...   "
                "On Colab, add it in the Secrets panel (key icon) and enable notebook access."
            )
        print(f"WARNING: {name} is not set; continuing without it (public repos only).")
        return None
    return value


HF_TOKEN = get_secret("HF_TOKEN", required=REQUIRE_HF_TOKEN)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# Google Drive is a Colab-only convenience: it keeps checkpoints alive across runtime resets.
# Never mount unconditionally - the import does not exist on a plain Jupyter host.
if RUNTIME["env"] == "colab":
    from google.colab import drive

    drive.mount("/content/drive")

for directory in (WORK_DIR, OUTPUT_DIR, HF_DATA_CACHE):
    os.makedirs(directory, exist_ok=True)

os.environ.setdefault("HF_DATASETS_CACHE", HF_DATA_CACHE)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # silences the fork warning per worker

logger = logging.getLogger("fa_ocr_lora")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
logger.addHandler(_stream_handler)
_file_handler = logging.FileHandler(SKIPPED_RECORDS_LOG, mode="a", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_file_handler)
logger.propagate = False
logger.info("run start: phase=%s dtype=%s env=%s", PHASE, TORCH_DTYPE, RUNTIME["env"])

try:
    print(
        subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False, timeout=60
        ).stdout
        or "nvidia-smi produced no output"
    )
except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
    print(f"nvidia-smi unavailable: {exc}")

print("RUNTIME =", json.dumps({k: str(v) for k, v in RUNTIME.items()}, indent=2))
print(f"skipped-record log: {SKIPPED_RECORDS_LOG}")

# %% [markdown]
# ## Cell 3 — Load both datasets from the Hub and index them
#
# Nothing about the column layout is assumed. Each dataset's real `.features` is printed, then the
# image and text columns are resolved against the candidate lists in Cell 0 — the image column is
# picked by *type* (`datasets.Image`) first and by name second, so a repo that calls it `png` works
# without an edit.
#
# ### How page vs paragraph is decided
# The specification allows a heuristic on aspect ratio or text length. It is not needed:
# `Omarrran/Persian_Pixel` publishes **one config per granularity** (`sentence`, `paragraph`,
# `page`, `full`) *and* carries an explicit `sample_type` column. So the notebook loads the
# `paragraph` and `page` configs directly and the tag is ground truth, not inference. Two documented
# fallbacks exist, in order:
#
# 1. configs missing → load the default config and tag from the `sample_type` column;
# 2. that column missing too → heuristic: a row is a *page* if `height / width > 1.15` **and** the
#    text is longer than the median, otherwise a *paragraph*. Only this last path is a guess, and it
#    prints a loud warning when it is used.
#
# ### `digit_density`
# Fraction of non-whitespace characters in the target text that are Latin, Persian or Arabic-Indic
# digits. It drives both the stratification in Cell 4 and the oversampling weights in Cell 6, which
# is the whole point of the exercise: administrative letters live and die on registration numbers
# and dates.
#
# ### If the admin-letters repo is empty
# `shahriarhd/persian-official-letters-synth` currently contains only `.gitattributes` — the data
# files never landed. The loader below reports that as a **blocker** rather than crashing, and
# continues with Persian_Pixel alone so the rest of the notebook stays runnable end to end. Re-upload
# the export (or set `ADMIN_LETTERS_LOCAL_DIR` in Cell 0) before treating any result as meaningful:
# without it there is no administrative-letter signal in the training set at all.

# %%
"""Cell 3 - dataset loading, source tagging, digit-density indexing."""

import re
import zlib
from collections.abc import Iterable
from statistics import median

import datasets as hf_datasets
from datasets import Dataset, load_dataset
from datasets import Image as HFImage

DIGIT_PATTERN = re.compile(
    "[" + "".join(f"{low}-{high}" for low, high in DIGIT_CHAR_RANGES) + "]"
)


def digit_density(text: str) -> float:
    """Fraction of non-whitespace characters that are digits in any of the three scripts.

    Args:
        text: Target transcription.

    Returns:
        Value in ``[0, 1]``; ``0.0`` for empty or whitespace-only text.
    """
    if not text:
        return 0.0
    non_space = [char for char in text if not char.isspace()]
    if not non_space:
        return 0.0
    return sum(1 for char in non_space if DIGIT_PATTERN.match(char)) / len(non_space)


def list_configs(repo: str) -> list[str]:
    """List a repo's dataset configs, tolerating both token-aware and token-free signatures."""
    from datasets import get_dataset_config_names

    for kwargs in ({"token": HF_TOKEN}, {}):
        try:
            return list(get_dataset_config_names(repo, **kwargs))
        except TypeError:
            continue
        except Exception as exc:  # empty repo, network error, gated repo
            logger.warning("could not list configs for %s: %s: %s", repo, type(exc).__name__, exc)
            return []
    return []


def load_split(repo: str, config: str | None, split: str) -> Dataset | None:
    """Load one split of one config, returning ``None`` (and logging) on any failure."""
    try:
        return load_dataset(
            repo, name=config, split=split, cache_dir=HF_DATA_CACHE, token=HF_TOKEN
        )
    except Exception as exc:
        logger.warning(
            "load failed repo=%s config=%s split=%s: %s: %s",
            repo,
            config,
            split,
            type(exc).__name__,
            exc,
        )
        return None


def load_local_dataset(path: str) -> Dataset | None:
    """Load an admin-letters export from a local directory (saved dataset or parquet folder)."""
    try:
        if os.path.exists(os.path.join(path, "dataset_info.json")):
            return hf_datasets.load_from_disk(path)
        return load_dataset("parquet", data_dir=path, split="train", cache_dir=HF_DATA_CACHE)
    except Exception as exc:
        logger.warning("local admin-letters load failed at %s: %s: %s", path, type(exc).__name__, exc)
        return None


def describe_features(name: str, dataset: Dataset) -> None:
    """Print a dataset's real schema and row count before any column is referenced."""
    print(f"\n--- {name}: {len(dataset):,} rows ---")
    for column, feature in dataset.features.items():
        print(f"    {column}: {feature}")


def detect_image_column(dataset: Dataset) -> str:
    """Resolve the image column by feature type first, then by candidate name.

    Raises:
        KeyError: If no column looks like an image.
    """
    for column, feature in dataset.features.items():
        if isinstance(feature, HFImage):
            return column
    for candidate in IMAGE_COLUMN_CANDIDATES:
        if candidate in dataset.column_names:
            return candidate
    raise KeyError(f"no image column found among {dataset.column_names}")


def detect_text_column(dataset: Dataset) -> str:
    """Resolve the target-text column against the Cell 0 candidate list.

    Raises:
        KeyError: If no candidate column exists.
    """
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in dataset.column_names:
            feature = dataset.features[candidate]
            if getattr(feature, "dtype", None) == "string":
                return candidate
    raise KeyError(f"no text column found among {dataset.column_names}")


def detect_optional_column(dataset: Dataset, candidates: Iterable[str]) -> str | None:
    """Return the first candidate column present in the dataset, or ``None``."""
    for candidate in candidates:
        if candidate in dataset.column_names:
            return candidate
    return None


SOURCES: dict[str, dict[str, Any]] = {}


def register_source(tag: str, dataset: Dataset) -> None:
    """Resolve a dataset's columns and register it under ``tag`` for later indexing."""
    SOURCES[tag] = {
        "dataset": dataset,
        "image_column": detect_image_column(dataset),
        "text_column": detect_text_column(dataset),
        "width_column": detect_optional_column(dataset, WIDTH_COLUMN_CANDIDATES),
        "height_column": detect_optional_column(dataset, HEIGHT_COLUMN_CANDIDATES),
    }
    resolved = {k: v for k, v in SOURCES[tag].items() if k != "dataset"}
    print(f"    resolved columns for {tag}: {resolved}")


# ---------------------------------------------------------------- admin letters
print(f"configs in {HF_ADMIN_LETTERS_REPO}: {list_configs(HF_ADMIN_LETTERS_REPO) or 'NONE'}")
admin_letters_ds: Dataset | None = None
if ADMIN_LETTERS_LOCAL_DIR:
    admin_letters_ds = load_local_dataset(ADMIN_LETTERS_LOCAL_DIR)
if admin_letters_ds is None:
    available_configs = list_configs(HF_ADMIN_LETTERS_REPO)
    for config in ADMIN_LETTERS_CONFIGS:
        if config is not None and available_configs and config not in available_configs:
            continue
        parts = [
            part
            for part in (load_split(HF_ADMIN_LETTERS_REPO, config, s) for s in ADMIN_LETTERS_SPLITS)
            if part is not None
        ]
        if parts:
            admin_letters_ds = parts[0] if len(parts) == 1 else hf_datasets.concatenate_datasets(parts)
            print(f"loaded admin letters from config={config!r}")
            break

if admin_letters_ds is None:
    print(
        "\n"
        + "!" * 78
        + f"\n!! BLOCKER: {HF_ADMIN_LETTERS_REPO} returned no data.\n"
        "!! The repo currently holds only .gitattributes - the parquet export never landed.\n"
        "!! Continuing with Persian_Pixel only so the pipeline stays runnable, but the run has\n"
        "!! ZERO administrative-letter signal and its metrics mean nothing for the real task.\n"
        "!! Fix: re-run the exporter/upload, or set ADMIN_LETTERS_LOCAL_DIR in Cell 0.\n" + "!" * 78
    )
    logger.error("admin-letters source unavailable: %s", HF_ADMIN_LETTERS_REPO)
else:
    describe_features("admin_letter", admin_letters_ds)
    register_source("admin_letter", admin_letters_ds)

# ---------------------------------------------------------------- Persian_Pixel
pixel_configs = list_configs(HF_PERSIAN_PIXEL_REPO)
print(f"\nconfigs in {HF_PERSIAN_PIXEL_REPO}: {pixel_configs or 'NONE'}")

granularity_from_configs = all(
    config in pixel_configs for config in PERSIAN_PIXEL_CONFIGS.values()
)
if granularity_from_configs:
    print("granularity source: per-config split (exact, no heuristic)")
    for tag, config in PERSIAN_PIXEL_CONFIGS.items():
        dataset = load_split(HF_PERSIAN_PIXEL_REPO, config, "train")
        if dataset is None:
            continue
        describe_features(tag, dataset)
        register_source(tag, dataset)
else:
    fallback_ds = load_split(HF_PERSIAN_PIXEL_REPO, None, "train")
    if fallback_ds is None:
        raise RuntimeError(
            f"{HF_PERSIAN_PIXEL_REPO} could not be loaded from any config; cannot continue."
        )
    describe_features("persian_pixel (single config)", fallback_ds)
    if PERSIAN_PIXEL_SAMPLE_TYPE_COLUMN in fallback_ds.column_names:
        print(f"granularity source: {PERSIAN_PIXEL_SAMPLE_TYPE_COLUMN!r} column (exact)")
        for tag, config in PERSIAN_PIXEL_CONFIGS.items():
            subset = fallback_ds.filter(
                lambda row, want=config: row[PERSIAN_PIXEL_SAMPLE_TYPE_COLUMN] == want
            )
            if len(subset):
                register_source(tag, subset)
    else:
        print(
            "WARNING: neither per-granularity configs nor a sample_type column exist. Falling back "
            "to the aspect-ratio + text-length heuristic; verify the thumbnails in Cell 4."
        )
        text_column = detect_text_column(fallback_ds)
        width_column = detect_optional_column(fallback_ds, WIDTH_COLUMN_CANDIDATES)
        height_column = detect_optional_column(fallback_ds, HEIGHT_COLUMN_CANDIDATES)
        median_chars = median(len(t or "") for t in fallback_ds.select_columns([text_column])[text_column])

        def _is_page(row: dict[str, Any]) -> bool:
            tall = True
            if width_column and height_column and row[width_column]:
                tall = row[height_column] / max(row[width_column], 1) > 1.15
            return tall and len(row[text_column] or "") > median_chars

        for tag, want_page in (("persian_pixel_page", True), ("persian_pixel_paragraph", False)):
            subset = fallback_ds.filter(lambda row, w=want_page: _is_page(row) == w)
            if len(subset):
                register_source(tag, subset)

if not SOURCES:
    raise RuntimeError("no usable data source was loaded; fix the blockers above before continuing.")

# ------------------------------------------------------- lightweight row index
INDEX: list[dict[str, Any]] = []


def build_index() -> None:
    """Build the in-memory row index used for sampling.

    Reads only the text and geometry columns (never the image column), so no image is decoded here.
    Appends one dict per row to the module-level ``INDEX``.
    """
    for tag, source in SOURCES.items():
        dataset: Dataset = source["dataset"]
        columns = [source["text_column"]]
        for key in ("width_column", "height_column"):
            if source[key]:
                columns.append(source[key])
        table = dataset.select_columns(columns).with_format(None)
        texts = table[source["text_column"]]
        widths = table[source["width_column"]] if source["width_column"] else [None] * len(texts)
        heights = table[source["height_column"]] if source["height_column"] else [None] * len(texts)

        known_widths = [w for w in widths if w]
        known_heights = [h for h in heights if h]
        fill_width = int(median(known_widths)) if known_widths else None
        fill_height = int(median(known_heights)) if known_heights else None
        n_filled = 0

        for row_index, text in enumerate(texts):
            text = text or ""
            if not text.strip():
                logger.warning("empty target text, skipped: %s:%d", tag, row_index)
                continue
            width, height = widths[row_index], heights[row_index]
            if not width or not height:
                width, height, n_filled = fill_width, fill_height, n_filled + 1
            uid = f"{tag}:{row_index}"
            INDEX.append(
                {
                    "uid": uid,
                    "source_tag": tag,
                    "row_index": row_index,
                    "text": text,
                    "n_chars": len(text),
                    "digit_density": digit_density(text),
                    "width": width,
                    "height": height,
                    # Per-sample seed recorded for reproducibility bookkeeping (CLAUDE.md s8):
                    # deterministic in uid, so a rerun reproduces it without storing RNG state.
                    "sample_seed": zlib.crc32(uid.encode("utf-8")) ^ VALIDATION_SEED,
                }
            )
        if n_filled:
            print(f"    {tag}: filled missing geometry on {n_filled:,} rows with the column median")


build_index()

print("\n--- index summary ---")
for tag in sorted(SOURCES):
    rows = [row for row in INDEX if row["source_tag"] == tag]
    if not rows:
        continue
    densities = sorted(row["digit_density"] for row in rows)
    print(
        f"  {tag:26s} n={len(rows):7,d}  "
        f"digit_density mean={sum(densities) / len(densities):.4f} "
        f"median={densities[len(densities) // 2]:.4f} max={densities[-1]:.4f}  "
        f"chars median={median(row['n_chars'] for row in rows):.0f}"
    )
print(f"  {'TOTAL':26s} n={len(INDEX):7,d}")

# %% [markdown]
# ## Cell 4 — Fixed validation carve-out, then stratified training selection
#
# Order matters. The validation set is carved out **first**, from the full index, and cached to
# `WORK_DIR/validation_manifest.json`. Every later run — a different pilot size, or the final A100
# run — reloads that file instead of resampling, so the 1800 validation rows are byte-identical
# across the whole campaign and the learning curve compares like with like. `VALIDATION_SEED = 42`
# must never change once the first pilot has used it.
#
# Then, from what is left:
#
# 1. **all** `admin_letter` rows, up to `CFG["target_samples"]` — the primary domain signal, never
#    subsampled away;
# 2. the remaining budget filled from Persian_Pixel, `paragraph : page ≈ 55 : 45`, via
#    `stratified_sample`. If one of the two pools runs short, the deficit spills into the other
#    rather than leaving the budget unfilled.
#
# ### About `stratified_sample`
# The specification's implementation deals the density-sorted rows into `n_bins` interleaved shards
# (`rows_sorted[i::n_bins]`) and samples evenly from each. That is systematic sampling: each shard is
# a miniature of the full density distribution, so the selection *preserves* the pool's digit-density
# profile — which is exactly what step 5's sanity check measures. It is kept as written, with one fix:
# floor division and short bins would otherwise return fewer rows than asked for, so a deterministic
# top-up pass fills the remainder from the unselected rows.
#
# Balancing digit-heavy samples *upward* is a separate mechanism — the weighted sampler in Cell 6.

# %%
"""Cell 4a - the fixed validation split and the stratified training selection."""

import hashlib
import random

BY_UID: dict[str, dict[str, Any]] = {row["uid"]: row for row in INDEX}


def stratified_sample(
    rows: list[dict[str, Any]],
    target_n: int,
    n_bins: int = DIGIT_DENSITY_BINS,
    seed: int = VALIDATION_SEED,
) -> list[dict[str, Any]]:
    """Sample ``target_n`` rows while preserving the pool's digit-density distribution.

    Args:
        rows: Candidate rows, each carrying a ``digit_density`` key.
        target_n: Number of rows wanted. Clamped to ``len(rows)``.
        n_bins: Number of density shards to deal the sorted rows into.
        seed: RNG seed; identical inputs and seed always yield the identical selection.

    Returns:
        The selected rows (order is deterministic but arbitrary).
    """
    if target_n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    rows_sorted = sorted(rows, key=lambda r: (r["digit_density"], r["uid"]))
    bins = [rows_sorted[i::n_bins] for i in range(n_bins)]
    per_bin = target_n // n_bins
    selected: list[dict[str, Any]] = []
    for shard in bins:
        selected += rng.sample(shard, min(per_bin, len(shard)))
    if len(selected) < target_n:  # top-up: floor division and short shards leave a remainder
        chosen = {row["uid"] for row in selected}
        remainder = [row for row in rows_sorted if row["uid"] not in chosen]
        selected += rng.sample(remainder, min(target_n - len(selected), len(remainder)))
    return selected


def text_fingerprint(text: str) -> str:
    """Short content hash used to detect that a cached uid no longer points at the same row."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def build_validation_split(size: int, seed: int) -> list[dict[str, Any]]:
    """Carve a validation set stratified across source tags proportionally to their availability."""
    per_tag: dict[str, list[dict[str, Any]]] = {}
    for row in INDEX:
        per_tag.setdefault(row["source_tag"], []).append(row)
    total = len(INDEX)
    size = min(size, total)
    quotas = {tag: int(size * len(rows) / total) for tag, rows in per_tag.items()}
    # Hand out rounding leftovers to the largest tags first, so the total lands exactly on `size`.
    for tag in sorted(per_tag, key=lambda t: -len(per_tag[t])):
        if sum(quotas.values()) >= size:
            break
        quotas[tag] += 1
    selected: list[dict[str, Any]] = []
    for tag, rows in sorted(per_tag.items()):
        selected += stratified_sample(rows, quotas[tag], seed=seed)
    return selected


def load_or_create_validation_manifest() -> list[dict[str, Any]]:
    """Load the cached validation manifest, or create and persist it on first use.

    Returns:
        The validation rows, resolved against the current index.
    """
    if os.path.exists(VALIDATION_MANIFEST_PATH):
        with open(VALIDATION_MANIFEST_PATH, encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("seed") != VALIDATION_SEED or cached.get("validation_size") != VALIDATION_SIZE:
            raise RuntimeError(
                f"{VALIDATION_MANIFEST_PATH} was built with seed="
                f"{cached.get('seed')} size={cached.get('validation_size')}, but Cell 0 now says "
                f"seed={VALIDATION_SEED} size={VALIDATION_SIZE}. Comparing runs across a changed "
                "validation set is meaningless - restore the old values, or delete the manifest and "
                "accept that earlier learning-curve points are no longer comparable."
            )
        rows, missing, drifted = [], [], []
        for uid in cached["uids"]:
            row = BY_UID.get(uid)
            if row is None:
                missing.append(uid)
                continue
            if cached["fingerprints"].get(uid) != text_fingerprint(row["text"]):
                drifted.append(uid)
            rows.append(row)
        if missing or drifted:
            print(
                f"WARNING: cached validation manifest is no longer exact - {len(missing)} uids are "
                f"absent from the current index and {len(drifted)} have different text. The "
                "validation set is NOT byte-identical to earlier runs; treat cross-run comparisons "
                "with care."
            )
            logger.warning("validation manifest drift: missing=%d drifted=%d", len(missing), len(drifted))
        print(f"reused cached validation manifest: {len(rows):,} rows from {VALIDATION_MANIFEST_PATH}")
        return rows

    rows = build_validation_split(VALIDATION_SIZE, VALIDATION_SEED)
    payload = {
        "validation_size": VALIDATION_SIZE,
        "seed": VALIDATION_SEED,
        "created_by_phase": PHASE,
        "uids": [row["uid"] for row in rows],
        "fingerprints": {row["uid"]: text_fingerprint(row["text"]) for row in rows},
    }
    with open(VALIDATION_MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    print(f"created validation manifest: {len(rows):,} rows -> {VALIDATION_MANIFEST_PATH}")
    return rows


VALIDATION_RECORDS = load_or_create_validation_manifest()
VALIDATION_UIDS = {row["uid"] for row in VALIDATION_RECORDS}


def select_training_records() -> list[dict[str, Any]]:
    """Apply Cell 0's SOURCE_MIX intent to the pool left after the validation carve-out."""
    pool = [row for row in INDEX if row["uid"] not in VALIDATION_UIDS]
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for row in pool:
        by_tag.setdefault(row["source_tag"], []).append(row)

    budget = CFG["target_samples"]
    admin_pool = by_tag.get("admin_letter", [])
    admin_selected = (
        list(admin_pool)
        if len(admin_pool) <= budget
        else stratified_sample(admin_pool, budget, seed=SEED)
    )
    remaining = max(budget - len(admin_selected), 0)

    paragraph_pool = by_tag.get("persian_pixel_paragraph", [])
    page_pool = by_tag.get("persian_pixel_page", [])
    paragraph_target = int(round(remaining * PIXEL_PARAGRAPH_PAGE_RATIO[0]))
    page_target = remaining - paragraph_target
    # Spill an unfillable quota into the other granularity instead of under-filling the budget.
    if paragraph_target > len(paragraph_pool):
        page_target += paragraph_target - len(paragraph_pool)
        paragraph_target = len(paragraph_pool)
    if page_target > len(page_pool):
        paragraph_target = min(paragraph_target + page_target - len(page_pool), len(paragraph_pool))
        page_target = len(page_pool)

    selected = (
        admin_selected
        + stratified_sample(paragraph_pool, paragraph_target, seed=SEED)
        + stratified_sample(page_pool, page_target, seed=SEED)
    )
    random.Random(SEED).shuffle(selected)
    if len(selected) < budget:
        print(
            f"NOTE: only {len(selected):,} of the requested {budget:,} samples exist across all "
            "sources; the run continues with everything available."
        )
    return selected


TRAIN_RECORDS = select_training_records()

if SMOKE_TEST:
    # Keep every tag represented so the smoke test exercises every code path.
    smoke: list[dict[str, Any]] = []
    per_tag_quota = max(SMOKE_TEST_SAMPLES // max(len(SOURCES), 1), 1)
    for tag in sorted(SOURCES):
        smoke += [row for row in TRAIN_RECORDS if row["source_tag"] == tag][:per_tag_quota]
    TRAIN_RECORDS = smoke[:SMOKE_TEST_SAMPLES]
    VALIDATION_RECORDS = VALIDATION_RECORDS[:SMOKE_TEST_EVAL_SAMPLES]
    print(f"SMOKE_TEST: train={len(TRAIN_RECORDS)} val={len(VALIDATION_RECORDS)}")


def write_train_manifest(records: list[dict[str, Any]], path: str) -> None:
    """Persist the exact training selection (metadata only, no image bytes or text bodies)."""
    with open(path, "w", encoding="utf-8") as handle:
        header = {
            "_manifest": "training-selection",
            "phase": PHASE,
            "target_samples": CFG["target_samples"],
            "selection_seed": SEED,
            "validation_seed": VALIDATION_SEED,
            "n_records": len(records),
        }
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for row in records:
            handle.write(
                json.dumps(
                    {
                        "uid": row["uid"],
                        "source_tag": row["source_tag"],
                        "row_index": row["row_index"],
                        "n_chars": row["n_chars"],
                        "digit_density": round(row["digit_density"], 6),
                        "width": row["width"],
                        "height": row["height"],
                        "sample_seed": row["sample_seed"],
                        "text_sha1": text_fingerprint(row["text"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"training manifest written: {path}")


write_train_manifest(TRAIN_RECORDS, TRAIN_MANIFEST_PATH)

# %% [markdown]
# ### Cell 4b — Qualitative register check
#
# Five Persian_Pixel rows, image plus target text. Look at them and answer one question: **is this
# administrative-register text, or general-purpose Persian?**
#
# It changes the sampling weights. If Persian_Pixel turns out to read like official correspondence
# (formal register, registration numbers, dates, institutional phrasing) then it is a much closer
# domain match than assumed, and the paragraph/page share of the budget should go **up** — the
# 55 : 45 split and the "fill the remainder" policy in Cell 0 exist to be adjusted, not obeyed.
# If it reads like Wikipedia prose or web text, keep it capped: its job is only to teach glyph and
# digit shapes, not the domain, and a larger share would dilute the administrative signal that the
# adapter exists to learn.

# %%
"""Cell 4b - print a handful of Persian_Pixel samples so the register can be judged by eye."""

from IPython.display import display

PREVIEW_COUNT = 5
PREVIEW_THUMBNAIL_WIDTH = 520


def preview_rows(rows: list[dict[str, Any]], count: int = PREVIEW_COUNT) -> None:
    """Display up to ``count`` rows as thumbnail plus target text."""
    for row in rows[:count]:
        source = SOURCES[row["source_tag"]]
        try:
            record = source["dataset"][row["row_index"]]
            image = record[source["image_column"]]
            if not hasattr(image, "size"):  # a bytes/dict column rather than a decoded PIL image
                from io import BytesIO

                from PIL import Image as PILImage

                payload = image["bytes"] if isinstance(image, dict) else image
                image = PILImage.open(BytesIO(payload))
            scale = PREVIEW_THUMBNAIL_WIDTH / image.width
            print(
                f"\n[{row['source_tag']}] uid={row['uid']}  {image.width}x{image.height}px  "
                f"chars={row['n_chars']}  digit_density={row['digit_density']:.3f}"
            )
            display(image.convert("RGB").resize((PREVIEW_THUMBNAIL_WIDTH, int(image.height * scale))))
            text = row["text"]
            print(text[:400] + (" [...]" if len(text) > 400 else ""))
        except Exception as exc:
            logger.warning("preview failed for %s: %s: %s", row["uid"], type(exc).__name__, exc)


pixel_preview = [row for row in TRAIN_RECORDS if row["source_tag"].startswith("persian_pixel")]
random.Random(VALIDATION_SEED).shuffle(pixel_preview)
preview_rows(pixel_preview)

# %% [markdown]
# ### Cell 4c — Did the stratification hold?
#
# The table below bins digit density into `DIGIT_DENSITY_BINS` quantile buckets of the **full pool**
# and reports the share of each bucket in the pool, in the training selection and in the validation
# set. If stratification worked, the three share columns track each other closely; a selection that
# has drifted toward the low-digit buckets is a selection that will not teach the model numbers.

# %%
"""Cell 4c - per-tag counts and the digit-density distribution check."""


def quantile_edges(values: list[float], n_bins: int) -> list[float]:
    """Inclusive upper edges of ``n_bins`` equal-count buckets over ``values``."""
    ordered = sorted(values)
    return [ordered[min(int(len(ordered) * (i + 1) / n_bins), len(ordered) - 1)] for i in range(n_bins)]


def bucket_shares(rows: list[dict[str, Any]], edges: list[float]) -> list[float]:
    """Fraction of ``rows`` falling into each bucket defined by ``edges``."""
    counts = [0] * len(edges)
    for row in rows:
        for index, edge in enumerate(edges):
            if row["digit_density"] <= edge:
                counts[index] += 1
                break
        else:
            counts[-1] += 1
    total = max(len(rows), 1)
    return [count / total for count in counts]


print("--- final counts per source_tag ---")
for tag in sorted(SOURCES):
    n_train = sum(1 for row in TRAIN_RECORDS if row["source_tag"] == tag)
    n_val = sum(1 for row in VALIDATION_RECORDS if row["source_tag"] == tag)
    share = n_train / max(len(TRAIN_RECORDS), 1)
    print(f"  {tag:26s} train={n_train:7,d} ({share:5.1%})   validation={n_val:5,d}")
print(f"  {'TOTAL':26s} train={len(TRAIN_RECORDS):7,d}          validation={len(VALIDATION_RECORDS):5,d}")

EDGES = quantile_edges([row["digit_density"] for row in INDEX], DIGIT_DENSITY_BINS)
pool_shares = bucket_shares(INDEX, EDGES)
train_shares = bucket_shares(TRAIN_RECORDS, EDGES)
val_shares = bucket_shares(VALIDATION_RECORDS, EDGES)

print(f"\n--- digit-density distribution ({DIGIT_DENSITY_BINS} quantile buckets of the full pool) ---")
print(f"  {'bucket (<= density)':22s} {'pool':>8s} {'train':>8s} {'val':>8s}")
for index, edge in enumerate(EDGES):
    print(
        f"  <= {edge:<19.4f} {pool_shares[index]:7.1%} {train_shares[index]:7.1%} "
        f"{val_shares[index]:7.1%}"
    )
max_drift = max(abs(p - t) for p, t in zip(pool_shares, train_shares, strict=True))
print(f"  largest pool-vs-train bucket drift: {max_drift:.1%}")

# %% [markdown]
# ## Cell 5 — Preprocessing: the pixel budget
#
# PaddleOCR-VL's image processor is NaViT-style: no fixed square input. `smart_resize` snaps the
# image to a multiple of `patch_size × merge_size = 28` on both axes while keeping the aspect ratio,
# subject to `min_pixels ≤ h·w ≤ max_pixels`. The number of image tokens the decoder then sees is
#
# ```
# image_tokens = (h/14) · (w/14) / merge_size²      # = h·w / 784
# ```
#
# so the budget is a *sequence-length* knob, not just a memory knob: 1.5 M px ≈ 1913 tokens,
# 2.82 M px ≈ 3600 tokens, against `MAX_TOTAL_SEQ_LEN = 5500`.
#
# Three things this cell does that the specification implies but does not spell out:
#
# * **Reads the real limits from the processor** and warns if `CFG["train_max_pixels"]` is outside
#   them, then clamps rather than silently exceeding the model's own ceiling.
# * **Does the resize itself.** The processor's `__call__` ignores per-call `images_kwargs` — the
#   pixel budget can only be changed by mutating `image_processor.min_pixels/max_pixels`, which is
#   process-global state and unusable for the per-sample budget that `ON_OVERSIZE="shrink_image"`
#   needs. Resizing here with the model's own `smart_resize` maths makes the processor's later
#   resize a no-op, and the budget becomes a per-sample argument.
# * **Encodes one sample at a time.** The image processor itself loops over a list correctly, but the
#   pixel budget is per sample under `ON_OVERSIZE="shrink_image"`, so Cell 6 encodes each record on
#   its own and the collator batches the results afterwards.

# %%
"""Cell 5 - processor, pixel-budget resolution, aspect-preserving resize."""

import math

from PIL import Image as PILImage
from PIL import ImageOps
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
tokenizer = processor.tokenizer
image_processor = processor.image_processor

PATCH_SIZE = int(image_processor.patch_size)
MERGE_SIZE = int(image_processor.merge_size)
RESIZE_FACTOR = PATCH_SIZE * MERGE_SIZE
MODEL_MIN_PIXELS = int(image_processor.min_pixels)
MODEL_MAX_PIXELS = int(image_processor.max_pixels)
PIXELS_PER_IMAGE_TOKEN = PATCH_SIZE * PATCH_SIZE * MERGE_SIZE * MERGE_SIZE

print(
    f"processor limits: min_pixels={MODEL_MIN_PIXELS:,} max_pixels={MODEL_MAX_PIXELS:,} "
    f"patch_size={PATCH_SIZE} merge_size={MERGE_SIZE} "
    f"({PIXELS_PER_IMAGE_TOKEN} px per image token)"
)

requested_max_pixels = int(CFG["train_max_pixels"])
if requested_max_pixels > MODEL_MAX_PIXELS:
    print(
        f"WARNING: CFG['train_max_pixels']={requested_max_pixels:,} exceeds the model's own "
        f"max_pixels={MODEL_MAX_PIXELS:,}; clamping. Raising it further would only cost memory - "
        "the processor would clamp it anyway."
    )
if requested_max_pixels < MODEL_MIN_PIXELS:
    raise ValueError(
        f"CFG['train_max_pixels']={requested_max_pixels:,} is below the model's "
        f"min_pixels={MODEL_MIN_PIXELS:,}; smart_resize would upscale every image straight back "
        "past the budget. Raise the budget in Cell 0."
    )
EFFECTIVE_MAX_PIXELS = min(requested_max_pixels, MODEL_MAX_PIXELS)
MAX_IMAGE_TOKENS = EFFECTIVE_MAX_PIXELS // PIXELS_PER_IMAGE_TOKEN
print(
    f"effective pixel budget: {EFFECTIVE_MAX_PIXELS:,} px -> at most {MAX_IMAGE_TOKENS:,} image "
    f"tokens per sample, leaving {MAX_TOTAL_SEQ_LEN - MAX_IMAGE_TOKENS:,} tokens for text "
    f"(MAX_TOTAL_SEQ_LEN={MAX_TOTAL_SEQ_LEN:,})"
)

# Belt and braces: the processor would clamp anything we missed. This is global, per-process state.
image_processor.max_pixels = EFFECTIVE_MAX_PIXELS
image_processor.size = {"min_pixels": MODEL_MIN_PIXELS, "max_pixels": EFFECTIVE_MAX_PIXELS}


def smart_resize(
    height: int,
    width: int,
    factor: int = RESIZE_FACTOR,
    min_pixels: int = MODEL_MIN_PIXELS,
    max_pixels: int = EFFECTIVE_MAX_PIXELS,
) -> tuple[int, int]:
    """Model-identical target geometry: axes divisible by ``factor``, pixels inside the budget.

    A local copy of the function inside the model's remote ``image_processing_paddleocr_vl`` module,
    so that resizing here and resizing there agree exactly and the processor's own resize is a no-op.

    Args:
        height: Source height in pixels.
        width: Source width in pixels.
        factor: Both output axes are multiples of this (``patch_size * merge_size``).
        min_pixels: Lower bound on the output pixel count; smaller images are upscaled.
        max_pixels: Upper bound on the output pixel count.

    Returns:
        ``(height, width)`` of the resized image.

    Raises:
        ValueError: If the aspect ratio exceeds 200:1, which the vision encoder rejects.
    """
    if height < factor:
        width = round((width * factor) / height)
        height = factor
    if width < factor:
        height = round((height * factor) / width)
        width = factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def estimate_image_tokens(width: int, height: int, max_pixels: int = EFFECTIVE_MAX_PIXELS) -> int:
    """Number of decoder tokens an image of this geometry will occupy under a pixel budget."""
    h_bar, w_bar = smart_resize(int(height), int(width), max_pixels=max_pixels)
    return (h_bar // PATCH_SIZE) * (w_bar // PATCH_SIZE) // (MERGE_SIZE * MERGE_SIZE)


def prepare_image(image: PILImage.Image, max_pixels: int = EFFECTIVE_MAX_PIXELS) -> PILImage.Image:
    """Normalise an image for the model: EXIF-uprighted, RGB, resized to the budget.

    Aspect ratio is always preserved - never padded to square, never distorted.

    Args:
        image: Source image.
        max_pixels: Pixel budget for this one sample.

    Returns:
        An RGB image whose dimensions are multiples of ``RESIZE_FACTOR`` and whose pixel count is
        inside ``[MODEL_MIN_PIXELS, max_pixels]``.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    target_height, target_width = smart_resize(image.height, image.width, max_pixels=max_pixels)
    if (image.height, image.width) != (target_height, target_width):
        image = image.resize((target_width, target_height), PILImage.Resampling.BICUBIC)
    return image


_sample_geometry = [
    (row["width"], row["height"])
    for row in TRAIN_RECORDS[:2000]
    if row["width"] and row["height"]
]
if _sample_geometry:
    _token_counts = sorted(estimate_image_tokens(w, h) for w, h in _sample_geometry)
    print(
        f"image tokens over {len(_token_counts):,} sampled training rows: "
        f"min={_token_counts[0]:,} median={_token_counts[len(_token_counts) // 2]:,} "
        f"max={_token_counts[-1]:,}"
    )

# %% [markdown]
# ## Cell 6 — Prompt format, label masking, dataset, collator, sampler
#
# ### Prompt and label masking
# The chat template is inspected at runtime rather than reproduced by hand: the prompt is
# `apply_chat_template([user], add_generation_prompt=True)` and the answer suffix is recovered by
# templating a sentinel target and splitting on it. So if the model card's template changes, this
# cell follows it instead of drifting from it.
#
# Prompt and answer are tokenized **separately** and concatenated. That is deliberate: at inference
# the prompt is tokenized alone and generation continues from there, so building training inputs the
# same way makes the training boundary and the inference boundary the same token sequence. Labels are
# `-100` over the prompt (image tokens included) and real ids over the answer, so cross-entropy sees
# completion tokens only.
#
# The single image placeholder in the prompt is expanded to exactly
# `grid_t·grid_h·grid_w / merge_size²` copies — the model's `masked_scatter` requires that count to
# equal the number of projected patch embeddings, so the cell asserts it per sample.
#
# ### Sequence budget
# Exact token lengths are computed up front for every selected record (fast tokenizer, batched).
# A record whose image + text exceeds `MAX_TOTAL_SEQ_LEN` is handled by `ON_OVERSIZE`; the default
# `"shrink_image"` lowers *that sample's* pixel budget so the full target survives, because a
# truncated OCR target teaches the model to stop mid-page. Records that cannot fit even at
# `min_pixels` are dropped and counted.
#
# ### Sampler
# `Trainer`'s own `group_by_length` would materialise `input_ids` for the entire dataset to measure
# lengths — decoding every image once per epoch just to sort. Instead `GROUP_BY_LENGTH` is
# implemented here on the precomputed lengths, inside a sampler that also applies the digit-density
# oversampling weights. `TrainingArguments.group_by_length` is therefore left `False`; the behaviour
# is not lost, only moved.

# %%
"""Cell 6 - encoding, dataset, collator and the digit-aware length-grouped sampler."""

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

IMAGE_TOKEN = processor.image_token
IMAGE_TOKEN_ID = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
PAD_TOKEN_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
LABEL_IGNORE_INDEX = -100
_SENTINEL = "TARGET"

_user_message = {
    "role": "user",
    "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[TRAIN_TASK]}],
}
PROMPT_TEXT: str = processor.apply_chat_template(
    [_user_message], tokenize=False, add_generation_prompt=True
)
_templated = processor.apply_chat_template(
    [_user_message, {"role": "assistant", "content": [{"type": "text", "text": _SENTINEL}]}],
    tokenize=False,
    add_generation_prompt=False,
)
_prefix, ANSWER_SUFFIX = _templated.split(_SENTINEL)
if _prefix != PROMPT_TEXT:
    raise RuntimeError(
        "the chat template's assistant prefix does not match its generation prompt; the "
        "prompt/answer boundary cannot be derived safely.\n"
        f"generation prompt: {PROMPT_TEXT!r}\nassistant prefix : {_prefix!r}"
    )

PROMPT_TEMPLATE_IDS: list[int] = tokenizer(PROMPT_TEXT, add_special_tokens=False).input_ids
if PROMPT_TEMPLATE_IDS.count(IMAGE_TOKEN_ID) != 1:
    raise RuntimeError(
        f"expected exactly one image placeholder in the prompt, found "
        f"{PROMPT_TEMPLATE_IDS.count(IMAGE_TOKEN_ID)}"
    )
IMAGE_TOKEN_POSITION = PROMPT_TEMPLATE_IDS.index(IMAGE_TOKEN_ID)
PROMPT_TEXT_TOKENS = len(PROMPT_TEMPLATE_IDS) - 1  # prompt tokens excluding the placeholder itself
MIN_IMAGE_TOKENS = MODEL_MIN_PIXELS // PIXELS_PER_IMAGE_TOKEN

print(f"prompt template   : {PROMPT_TEXT!r}")
print(f"answer suffix     : {ANSWER_SUFFIX!r}")
print(f"prompt token ids  : {PROMPT_TEMPLATE_IDS}")
print(
    f"image token id={IMAGE_TOKEN_ID} at position {IMAGE_TOKEN_POSITION}; "
    f"fixed text overhead={PROMPT_TEXT_TOKENS} tokens; pad id={PAD_TOKEN_ID}; "
    f"fast tokenizer={tokenizer.is_fast}"
)


class OversizeSample(RuntimeError):
    """Raised when a record cannot fit inside ``MAX_TOTAL_SEQ_LEN`` under the current policy."""


def sample_pixel_budget(n_answer_tokens: int) -> int:
    """Pixel budget for one sample given how many tokens its target needs.

    Args:
        n_answer_tokens: Token count of the target text plus the template's answer suffix.

    Returns:
        The pixel budget in pixels.

    Raises:
        OversizeSample: If no budget at or above the model's ``min_pixels`` leaves room for the
            target, and the policy is not ``"truncate"``.
    """
    room = MAX_TOTAL_SEQ_LEN - PROMPT_TEXT_TOKENS - n_answer_tokens
    if room >= MAX_IMAGE_TOKENS:
        return EFFECTIVE_MAX_PIXELS
    if ON_OVERSIZE == "skip":
        raise OversizeSample(
            f"answer needs {n_answer_tokens} tokens; only {room} left for {MAX_IMAGE_TOKENS} "
            "image tokens"
        )
    if ON_OVERSIZE == "truncate":
        return EFFECTIVE_MAX_PIXELS  # the caller cuts the answer instead
    if room < MIN_IMAGE_TOKENS:
        raise OversizeSample(
            f"answer needs {n_answer_tokens} tokens, leaving room for {room} image tokens, below "
            f"the model's floor of {MIN_IMAGE_TOKENS}"
        )
    return room * PIXELS_PER_IMAGE_TOKEN


def encode_example(image: PILImage.Image, target_text: str) -> dict[str, torch.Tensor]:
    """Turn one image and its transcription into a training example.

    Args:
        image: The page or crop, undecoded resizing/normalisation applied here.
        target_text: Ground-truth transcription in reading order.

    Returns:
        Mapping with ``input_ids``, ``labels``, ``attention_mask``, ``pixel_values`` and
        ``image_grid_thw``.

    Raises:
        OversizeSample: If the record cannot fit under ``MAX_TOTAL_SEQ_LEN``.
    """
    answer_ids: list[int] = tokenizer(
        target_text + ANSWER_SUFFIX, add_special_tokens=False
    ).input_ids
    budget = sample_pixel_budget(len(answer_ids))

    prepared = prepare_image(image, budget)
    image_inputs = image_processor(images=prepared, return_tensors="pt")
    grid = image_inputs["image_grid_thw"][0]
    n_image_tokens = int(grid.prod().item()) // (MERGE_SIZE * MERGE_SIZE)

    if ON_OVERSIZE == "truncate":
        room = MAX_TOTAL_SEQ_LEN - PROMPT_TEXT_TOKENS - n_image_tokens
        if room < 1:
            raise OversizeSample("no room left for any target token")
        if len(answer_ids) > room:
            answer_ids = answer_ids[: room - 1] + [tokenizer.eos_token_id]

    prompt_ids = (
        PROMPT_TEMPLATE_IDS[:IMAGE_TOKEN_POSITION]
        + [IMAGE_TOKEN_ID] * n_image_tokens
        + PROMPT_TEMPLATE_IDS[IMAGE_TOKEN_POSITION + 1 :]
    )
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > MAX_TOTAL_SEQ_LEN:
        raise OversizeSample(f"{len(input_ids)} tokens exceeds MAX_TOTAL_SEQ_LEN={MAX_TOTAL_SEQ_LEN}")

    patches_per_token = MERGE_SIZE * MERGE_SIZE
    if image_inputs["pixel_values"].shape[0] != n_image_tokens * patches_per_token:
        raise OversizeSample(
            "patch count does not match the expanded placeholder count; the masked_scatter in the "
            "model would fail"
        )

    labels = [LABEL_IGNORE_INDEX] * len(prompt_ids) + answer_ids
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "pixel_values": image_inputs["pixel_values"],
        "image_grid_thw": image_inputs["image_grid_thw"],
    }


def load_pil_image(record: dict[str, Any]) -> PILImage.Image:
    """Fetch and decode the image for one index record."""
    source = SOURCES[record["source_tag"]]
    row = source["dataset"][record["row_index"]]
    image = row[source["image_column"]]
    if hasattr(image, "size") and hasattr(image, "convert"):
        return image
    from io import BytesIO

    payload = image["bytes"] if isinstance(image, dict) else image
    return PILImage.open(BytesIO(payload))


class PersianOcrDataset(TorchDataset):
    """Index records on top of the loaded Hugging Face datasets, encoded on demand.

    A record that fails to decode or does not fit is logged to ``SKIPPED_RECORDS_LOG`` and replaced
    by the next usable record, so a bad row never kills a run mid-epoch.
    """

    def __init__(self, records: list[dict[str, Any]], name: str) -> None:
        self.records = records
        self.name = name

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        for attempt in range(MAX_GETITEM_RETRIES):
            record = self.records[(index + attempt) % len(self.records)]
            try:
                return encode_example(load_pil_image(record), record["text"])
            except (OversizeSample, OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "skipped %s uid=%s (%s: %s)", self.name, record["uid"], type(exc).__name__, exc
                )
        raise RuntimeError(
            f"{MAX_GETITEM_RETRIES} consecutive records failed to encode starting at index {index}; "
            f"see {SKIPPED_RECORDS_LOG}"
        )


def collate_fn(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pad a list of encoded examples into one batch.

    Padding uses the tokenizer pad id for ``input_ids``, ``-100`` for ``labels`` and ``0`` for
    ``attention_mask``; variable-resolution patch tensors are concatenated along their first axis,
    which is what the model's flat ``cu_seqlens`` vision path expects.
    """
    longest = max(feature["input_ids"].shape[0] for feature in features)
    input_ids, labels, attention_mask = [], [], []
    for feature in features:
        pad = longest - feature["input_ids"].shape[0]
        input_ids.append(torch.nn.functional.pad(feature["input_ids"], (0, pad), value=PAD_TOKEN_ID))
        labels.append(
            torch.nn.functional.pad(feature["labels"], (0, pad), value=LABEL_IGNORE_INDEX)
        )
        attention_mask.append(torch.nn.functional.pad(feature["attention_mask"], (0, pad), value=0))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
        "pixel_values": torch.cat([feature["pixel_values"] for feature in features], dim=0),
        "image_grid_thw": torch.cat([feature["image_grid_thw"] for feature in features], dim=0),
    }


def annotate_lengths(records: list[dict[str, Any]], batch_size: int = 512) -> None:
    """Attach exact (or estimated) token lengths to every record, in place.

    Adds ``n_answer_tokens``, ``n_image_tokens`` and ``n_total_tokens``. Uses the fast tokenizer in
    batches when available and falls back to ``CHARS_PER_TOKEN_ESTIMATE`` otherwise.
    """
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        if tokenizer.is_fast:
            encoded = tokenizer(
                [record["text"] + ANSWER_SUFFIX for record in chunk], add_special_tokens=False
            ).input_ids
            answer_lengths = [len(ids) for ids in encoded]
        else:
            answer_lengths = [
                int(len(record["text"]) / CHARS_PER_TOKEN_ESTIMATE) + 2 for record in chunk
            ]
        for record, n_answer in zip(chunk, answer_lengths, strict=True):
            record["n_answer_tokens"] = n_answer
            try:
                budget = sample_pixel_budget(n_answer)
                n_image = estimate_image_tokens(record["width"], record["height"], budget)
            except (OversizeSample, ValueError):
                n_image = -1
            record["n_image_tokens"] = n_image
            record["n_total_tokens"] = (
                -1 if n_image < 0 else PROMPT_TEXT_TOKENS + n_image + n_answer
            )


def drop_unfittable(records: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Remove records that cannot fit the sequence budget, logging and counting them."""
    kept = [r for r in records if 0 < r["n_total_tokens"] <= MAX_TOTAL_SEQ_LEN]
    dropped = len(records) - len(kept)
    if dropped:
        for record in records:
            if record not in kept:
                logger.warning(
                    "dropped %s uid=%s: needs %d tokens (answer=%d image=%d)",
                    name,
                    record["uid"],
                    record["n_total_tokens"],
                    record["n_answer_tokens"],
                    record["n_image_tokens"],
                )
        print(
            f"  {name}: dropped {dropped:,} of {len(records):,} records that cannot fit "
            f"MAX_TOTAL_SEQ_LEN={MAX_TOTAL_SEQ_LEN:,} under ON_OVERSIZE={ON_OVERSIZE!r}"
        )
    return kept


print("\ncomputing exact token lengths ...")
annotate_lengths(TRAIN_RECORDS)
annotate_lengths(VALIDATION_RECORDS)
TRAIN_RECORDS = drop_unfittable(TRAIN_RECORDS, "train")
VALIDATION_RECORDS = drop_unfittable(VALIDATION_RECORDS, "validation")
_lengths_sorted = sorted(record["n_total_tokens"] for record in TRAIN_RECORDS)
print(
    f"  train sequence length: min={_lengths_sorted[0]:,} "
    f"median={_lengths_sorted[len(_lengths_sorted) // 2]:,} "
    f"p95={_lengths_sorted[int(len(_lengths_sorted) * 0.95)]:,} max={_lengths_sorted[-1]:,} tokens"
)

TRAIN_LENGTHS = [record["n_total_tokens"] for record in TRAIN_RECORDS]
_densities = sorted(record["digit_density"] for record in TRAIN_RECORDS)
DIGIT_WEIGHT_THRESHOLD = _densities[min(int(len(_densities) * DIGIT_OVERSAMPLE_QUANTILE), len(_densities) - 1)]
TRAIN_WEIGHTS = [
    DIGIT_OVERSAMPLE_MULTIPLIER if record["digit_density"] >= DIGIT_WEIGHT_THRESHOLD else 1.0
    for record in TRAIN_RECORDS
]
print(
    f"  digit oversampling: {sum(1 for w in TRAIN_WEIGHTS if w > 1):,} records at "
    f"digit_density >= {DIGIT_WEIGHT_THRESHOLD:.4f} get weight {DIGIT_OVERSAMPLE_MULTIPLIER}"
)


class DigitAwareLengthGroupedSampler(Sampler[int]):
    """Weighted sampling by digit density, then length-grouped megabatches.

    Draws ``len(dataset)`` indices per epoch with replacement using the digit-density weights, then
    (when ``group_by_length``) sorts each megabatch by sequence length so that batches contain
    similarly sized pages and padding waste stays low. Megabatch order is shuffled.

    Args:
        lengths: Precomputed sequence length per record.
        weights: Sampling weight per record.
        batch_size: Per-device batch size.
        group_by_length: Whether to apply length grouping inside megabatches.
        seed: Base RNG seed; the epoch number is added to it so epochs differ.
        mega_batch_mult: Megabatch size as a multiple of ``batch_size``.
    """

    def __init__(
        self,
        lengths: list[int],
        weights: list[float],
        batch_size: int,
        group_by_length: bool = True,
        seed: int = SEED,
        mega_batch_mult: int = 50,
    ) -> None:
        self.lengths = lengths
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.batch_size = batch_size
        self.group_by_length = group_by_length
        self.seed = seed
        self.mega_batch_size = max(batch_size * mega_batch_mult, batch_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Called by the training loop so successive epochs draw different samples."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        drawn = torch.multinomial(
            self.weights, num_samples=len(self.lengths), replacement=True, generator=generator
        ).tolist()
        if not self.group_by_length:
            return iter(drawn)
        megabatches = [
            drawn[i : i + self.mega_batch_size] for i in range(0, len(drawn), self.mega_batch_size)
        ]
        megabatches = [
            sorted(batch, key=lambda index: self.lengths[index], reverse=True)
            for batch in megabatches
        ]
        order = torch.randperm(len(megabatches), generator=generator).tolist()
        return iter([index for position in order for index in megabatches[position]])


train_dataset = PersianOcrDataset(TRAIN_RECORDS, "train")
eval_dataset = PersianOcrDataset(
    VALIDATION_RECORDS[: (SMOKE_TEST_EVAL_SAMPLES if SMOKE_TEST else LOSS_EVAL_SAMPLES)],
    "eval_loss",
)

# ---------------------------------------------------- verify one encoded sample end to end
_example = train_dataset[0]
_n_masked = int((_example["labels"] == LABEL_IGNORE_INDEX).sum())
print("\n--- encoded sample check ---")
print(
    f"  input_ids={tuple(_example['input_ids'].shape)}  "
    f"pixel_values={tuple(_example['pixel_values'].shape)}  "
    f"grid={_example['image_grid_thw'].tolist()}"
)
print(f"  masked prompt tokens={_n_masked}  supervised answer tokens={len(_example['labels']) - _n_masked}")
print(
    "  prompt tail (decoded): "
    f"{tokenizer.decode(_example['input_ids'][_n_masked - 6 : _n_masked])!r}"
)
print(
    "  first supervised tokens: "
    f"{tokenizer.decode(_example['labels'][_n_masked : _n_masked + 24])!r}"
)
print(
    "  last supervised tokens : "
    f"{tokenizer.decode(_example['labels'][-8:])!r}"
)
_batch = collate_fn([train_dataset[0], train_dataset[1]]) if len(train_dataset) > 1 else None
if _batch is not None:
    print(
        f"  collated batch: input_ids={tuple(_batch['input_ids'].shape)} "
        f"pixel_values={tuple(_batch['pixel_values'].shape)} "
        f"image_grid_thw={tuple(_batch['image_grid_thw'].shape)}"
    )

# %% [markdown]
# ## Cell 7 — Base model, LoRA injection, what stays frozen
#
# `PaddleOCRVLForConditionalGeneration` is three stacks in one checkpoint:
#
# | Submodule | Params | Role | Adapted? |
# | --- | --- | --- | --- |
# | `visual` | SigLIP-style NaViT encoder, 27 layers, hidden 1152 | reads pixels | **frozen** |
# | `mlp_AR` | `Projector`: LayerNorm + 2 linears | 2×2 patch-merge into decoder space | **frozen** |
# | `model` + `lm_head` | ERNIE-4.5 decoder, 18 layers, hidden 1024, vocab 103 424 | writes text | **LoRA** |
#
# ### Why the target list is resolved to full paths
# The vision encoder's attention uses the *same leaf names* as the decoder — `q_proj`, `k_proj`,
# `v_proj` (`PaddleOCRAttention`). PEFT matches `target_modules` by suffix, so the specification's
# bare list would have injected adapters into all 27 encoder layers as well, quietly training the
# vision tower we are meant to be freezing and tripling the adapter size. This cell therefore walks
# `named_modules()`, keeps only paths whose leaf name is in `LORA_TARGET_MODULES` **and** whose path
# contains none of `LORA_EXCLUDE_MODULE_PREFIXES`, and passes those exact paths to `LoraConfig`.
# Both counts are printed so the filter is visible, not assumed.
#
# ### `embed_tokens` and `lm_head`
# Persian is under-represented in ERNIE-4.5's tokenizer, so adapting the embedding and the output
# head is what lets the adapter move Persian glyph and digit probabilities at all. `tie_word_embeddings`
# is `false` in this checkpoint, so the two are genuinely separate matrices and get separate adapters.
# They also carry a **lower learning rate** (`LR_EMBED_HEAD` = 7e-5 vs `LR_ATTN_MLP` = 2e-4): a
# vocabulary-sized matrix reached by every token is far easier to destabilise than a per-layer
# projection.
#
# ### fp16 detail
# On the T4 the weights load as fp16 and the trainable LoRA parameters are upcast back to fp32 — Adam
# moments in fp16 underflow at these learning rates. This is what `prepare_model_for_kbit_training`
# does for QLoRA, done explicitly here because we are not quantising.
#
# ### Two upstream quirks this cell works around
# * **Loss scaling under gradient accumulation.** `forward` ends in `**kwargs`, which is how
#   `Trainer` decides a model consumes `num_items_in_batch` and normalises the loss itself. This one
#   does not — it uses a mean-reduced `CrossEntropyLoss` — so `Trainer` would skip its own
#   `loss / grad_accum_steps` division and the gradients would come out `grad_accum_steps`× too
#   large (×20 in the pilot phase). Setting `accepts_loss_kwargs = False` on the model is the fix
#   `Trainer.compute_loss`'s own docstring prescribes; Cell 8 asserts it took effect.
# * **Checkpointing the frozen tower.** `gradient_checkpointing_enable()` sets the flag on *every*
#   module that owns one, the frozen vision encoder included, where it recomputes activations that no
#   backward pass will ever ask for. It is switched back off there.

# %%
"""Cell 7 - load PaddleOCR-VL, inject LoRA into the decoder only, freeze everything else."""

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=getattr(torch, TORCH_DTYPE),
    attn_implementation=ATTN_IMPLEMENTATION,
    low_cpu_mem_usage=True,
    token=HF_TOKEN,
)
base_model.config.use_cache = False  # incompatible with gradient checkpointing
# See the markdown above: without this the Trainer skips its loss/grad_accum normalisation.
base_model.accepts_loss_kwargs = False

print(
    f"loaded {type(base_model).__name__}: "
    f"{sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M params, "
    f"dtype={next(base_model.parameters()).dtype}, "
    f"attn={base_model.config._attn_implementation}"
)


def resolve_lora_targets(model: torch.nn.Module) -> tuple[list[str], list[str]]:
    """Expand the leaf-name target list into explicit module paths, minus the frozen stacks.

    Args:
        model: The unwrapped base model.

    Returns:
        ``(kept, excluded)`` module paths. ``kept`` goes to ``LoraConfig.target_modules``;
        ``excluded`` is printed so the filtering is auditable.
    """
    kept: list[str] = []
    excluded: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            continue
        if name.rsplit(".", 1)[-1] not in LORA_TARGET_MODULES:
            continue
        parts = name.split(".")
        if any(part in LORA_EXCLUDE_MODULE_PREFIXES for part in parts):
            excluded.append(name)
        else:
            kept.append(name)
    return kept, excluded


LORA_TARGET_PATHS, LORA_EXCLUDED_PATHS = resolve_lora_targets(base_model)
if not LORA_TARGET_PATHS:
    raise RuntimeError(
        f"none of {LORA_TARGET_MODULES} matched a Linear/Embedding module; the model layout has "
        "changed and Cell 0's target list needs updating."
    )
print(
    f"\nLoRA targets: {len(LORA_TARGET_PATHS)} modules kept, {len(LORA_EXCLUDED_PATHS)} excluded "
    f"by prefix {LORA_EXCLUDE_MODULE_PREFIXES}"
)
for leaf in LORA_TARGET_MODULES:
    n_kept = sum(1 for path in LORA_TARGET_PATHS if path.endswith(leaf))
    n_dropped = sum(1 for path in LORA_EXCLUDED_PATHS if path.endswith(leaf))
    print(f"  {leaf:14s} kept={n_kept:3d}  excluded={n_dropped:3d}")
if LORA_EXCLUDED_PATHS:
    print(f"  example exclusion: {LORA_EXCLUDED_PATHS[0]}")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_PATHS,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

if GRADIENT_CHECKPOINTING:
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": GRADIENT_CHECKPOINTING_REENTRANT}
    )
    base_model.enable_input_require_grads()
    # The vision stack is frozen: checkpointing it recomputes activations nothing will differentiate.
    for module in base_model.visual.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False

model = get_peft_model(base_model, lora_config)

# ------------------------------------------------------- freeze verification
FROZEN_PREFIXES = tuple(LORA_EXCLUDE_MODULE_PREFIXES)
leaked = [
    name
    for name, param in model.named_parameters()
    if param.requires_grad
    and any(part in FROZEN_PREFIXES for part in name.split("."))
]
if leaked:
    raise RuntimeError(
        f"{len(leaked)} parameters inside the frozen vision stack still require grad, e.g. "
        f"{leaked[:3]}. The exclusion filter did not work."
    )
for name, param in model.named_parameters():
    if any(part in FROZEN_PREFIXES for part in name.split(".")):
        param.requires_grad = False  # explicit, not merely implied by PEFT's defaults

if TORCH_DTYPE == "float16":
    n_upcast = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            n_upcast += 1
    print(f"\nfp16 run: upcast {n_upcast} trainable tensors to fp32 for optimizer stability")

# ------------------------------------------------------- parameter accounting
trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
n_trainable = sum(p.numel() for _, p in trainable_params)
n_total = sum(p.numel() for p in model.parameters())
print("\n--- parameter accounting ---")
print(f"  trainable : {n_trainable:12,d}  ({n_trainable / n_total:.4%} of total)")
print(f"  total     : {n_total:12,d}")
print(f"  adapter size at fp16 on disk: ~{n_trainable * 2 / 1e6:.1f} MB")

_frozen_report = {
    "visual (vision encoder)": base_model.visual,
    "mlp_AR (projector)": base_model.mlp_AR,
}
for label, submodule in _frozen_report.items():
    n_params = sum(p.numel() for p in submodule.parameters())
    n_grad = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
    print(f"  {label:24s} {n_params:12,d} params, requires_grad={n_grad:,d} (must be 0)")
    if n_grad:
        raise RuntimeError(f"{label} is not frozen")

_gc_states = {
    "decoder layers": base_model.model.layers[0].gradient_checkpointing,
    "vision encoder": base_model.visual.vision_model.encoder.gradient_checkpointing,
}
print(f"  gradient checkpointing: {_gc_states}")


def build_optimizer_param_groups() -> list[dict[str, Any]]:
    """Split the trainable LoRA parameters into the two differential-LR groups.

    Embedding and output-head adapters are reached by every token and destabilise easily, so they
    train at ``LR_EMBED_HEAD``; attention and MLP adapters train at ``LR_ATTN_MLP``. Bias and
    one-dimensional tensors are exempted from weight decay as usual.

    Returns:
        Parameter groups ready for the optimizer, each tagged with a ``name`` for logging.
    """
    embed_head_keys = ("embed_tokens", "lm_head")
    groups: dict[str, dict[str, Any]] = {
        "attn_mlp_decay": {"params": [], "lr": LR_ATTN_MLP, "weight_decay": WEIGHT_DECAY},
        "attn_mlp_nodecay": {"params": [], "lr": LR_ATTN_MLP, "weight_decay": 0.0},
        "embed_head_decay": {"params": [], "lr": LR_EMBED_HEAD, "weight_decay": WEIGHT_DECAY},
        "embed_head_nodecay": {"params": [], "lr": LR_EMBED_HEAD, "weight_decay": 0.0},
    }
    for name, param in trainable_params:
        family = "embed_head" if any(key in name for key in embed_head_keys) else "attn_mlp"
        decay = "nodecay" if param.ndim <= 1 else "decay"
        groups[f"{family}_{decay}"]["params"].append(param)
    return [{"name": name, **group} for name, group in groups.items() if group["params"]]


OPTIMIZER_PARAM_GROUPS = build_optimizer_param_groups()
print("\n--- learning-rate groups ---")
for group in OPTIMIZER_PARAM_GROUPS:
    n_group = sum(p.numel() for p in group["params"])
    print(
        f"  {group['name']:20s} lr={group['lr']:.1e} weight_decay={group['weight_decay']:.3f} "
        f"tensors={len(group['params']):3d} params={n_group:,d}"
    )

# %% [markdown]
# ## Cell 8 — Training arguments, optimizer, Trainer, run
#
# Everything numeric here comes from Cell 0 or `CFG`. What the cell adds is the wiring the
# specification leaves implicit:
#
# * **Differential learning rates** need an explicitly constructed optimizer, so `Trainer` gets
#   `optimizers=(optimizer, None)` — it still builds the cosine schedule with `WARMUP_RATIO`, and a
#   schedule scales all groups by the same factor, so the 2e-4 : 7e-5 ratio survives warmup and decay.
# * **`group_by_length` is handled by our sampler, not by `TrainingArguments`.** `Trainer`'s
#   implementation asks the dataset for `input_ids` lengths, which for an image dataset means decoding
#   and encoding every page before the first step. Cell 6 already knows the exact lengths, so the
#   argument stays `False` and `_get_train_sampler` returns `DigitAwareLengthGroupedSampler`.
# * **Gradient checkpointing is already on** (Cell 7, non-reentrant, vision tower excluded), so the
#   `TrainingArguments` flag stays `False` — setting it would re-enable checkpointing on the frozen
#   encoder with reentrant semantics.
# * **The batch/accumulation guard.** With `ALLOW_MULTI_IMAGE_BATCH = False`, a per-device batch above
#   1 is folded into gradient accumulation: the effective batch is unchanged, but each forward pass
#   carries exactly one page, so the vision encoder cannot attend across images (see Cell 0).
#
# ### Peak memory is the batch-size signal
# `PeakMemoryCallback` prints `torch.cuda.max_memory_allocated()` after the first optimizer step and
# at the end of training. That number, not intuition, is what to raise `per_device_batch_size` (and
# `ALLOW_MULTI_IMAGE_BATCH`, with flash-attention installed) against on the A100. Sequence length
# here is dominated by image tokens and varies per sample, so peak memory is not linear in batch size.
#
# ### Resuming
# `RESUME_FROM_CHECKPOINT = True` makes the cell look for `checkpoint-*` under `OUTPUT_DIR` and hand
# the newest one to `train()`; optimizer, scheduler and step counter come back with it. On Colab this
# is the difference between a disconnect costing minutes and costing the whole run — keep `OUTPUT_DIR`
# on mounted Drive there.

# %%
"""Cell 8 - training arguments, optimizer construction, Trainer subclass, training run."""

import inspect
import time

from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

# ------------------------------------------------- batch / accumulation resolution
per_device_batch_size = int(CFG["per_device_batch_size"])
grad_accum_steps = int(CFG["grad_accum_steps"])
if per_device_batch_size > 1 and not ALLOW_MULTI_IMAGE_BATCH:
    print(
        f"NOTE: folding per_device_batch_size={per_device_batch_size} into gradient accumulation.\n"
        f"      The vision encoder does not mask across images under "
        f"attn_implementation={ATTN_IMPLEMENTATION!r}, so batching pages together would let their\n"
        "      patches attend to each other. Effective batch is unchanged: "
        f"{per_device_batch_size} x {grad_accum_steps} -> 1 x {per_device_batch_size * grad_accum_steps}.\n"
        "      Set ALLOW_MULTI_IMAGE_BATCH = True in Cell 0 only with flash_attention_2 installed."
    )
    grad_accum_steps *= per_device_batch_size
    per_device_batch_size = 1
EFFECTIVE_BATCH_SIZE = per_device_batch_size * grad_accum_steps
STEPS_PER_EPOCH = max(len(train_dataset) // EFFECTIVE_BATCH_SIZE, 1)
print(
    f"batch: {per_device_batch_size} x {grad_accum_steps} accum = {EFFECTIVE_BATCH_SIZE} effective; "
    f"~{STEPS_PER_EPOCH:,} optimizer steps/epoch over {len(train_dataset):,} samples; "
    f"{CFG['num_epochs']} epochs -> ~{int(STEPS_PER_EPOCH * CFG['num_epochs']):,} steps"
)


def build_optimizer(param_groups: list[dict[str, Any]]) -> torch.optim.Optimizer:
    """Instantiate the optimizer named by ``OPTIM`` over pre-built parameter groups.

    ``paged_adamw_8bit`` keeps optimizer state in 8-bit and pages it to host memory under pressure,
    which is what makes the embedding and head adapters affordable on a 16 GB card.

    Args:
        param_groups: Groups from :func:`build_optimizer_param_groups`, each with its own ``lr``.

    Returns:
        The optimizer. Falls back to ``torch.optim.AdamW`` with a printed warning when the requested
        8-bit optimizer is unavailable (no bitsandbytes CUDA build).
    """
    groups = [{k: v for k, v in group.items() if k != "name"} for group in param_groups]
    shared = {"betas": ADAM_BETAS, "eps": ADAM_EPS}
    if OPTIM.endswith("8bit"):
        try:
            import bitsandbytes as bnb

            cls = bnb.optim.PagedAdamW8bit if OPTIM.startswith("paged") else bnb.optim.AdamW8bit
            optimizer = cls(groups, **shared)
            print(f"optimizer: {cls.__name__} (from OPTIM={OPTIM!r})")
            return optimizer
        except (ImportError, RuntimeError, AttributeError) as exc:
            print(
                f"WARNING: {OPTIM} unavailable ({type(exc).__name__}: {exc}); falling back to "
                "torch.optim.AdamW. Expect noticeably higher optimizer-state memory."
            )
    optimizer = torch.optim.AdamW(groups, **shared)
    print(f"optimizer: torch.optim.AdamW (requested OPTIM={OPTIM!r})")
    return optimizer


optimizer = build_optimizer(OPTIMIZER_PARAM_GROUPS)

# ------------------------------------------------- training arguments
training_kwargs: dict[str, Any] = {
    "output_dir": OUTPUT_DIR,
    "per_device_train_batch_size": per_device_batch_size,
    "per_device_eval_batch_size": EVAL_BATCH_SIZE,
    "gradient_accumulation_steps": grad_accum_steps,
    "num_train_epochs": float(CFG["num_epochs"]),
    "learning_rate": LR_ATTN_MLP,  # recorded for the logs; the real LRs live in the param groups
    "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO,
    "lr_scheduler_type": LR_SCHEDULER,
    "max_grad_norm": MAX_GRAD_NORM,
    "logging_steps": LOGGING_STEPS,
    "save_steps": SAVE_STEPS,
    "save_total_limit": SAVE_TOTAL_LIMIT,
    "save_safetensors": True,
    "fp16": TORCH_DTYPE == "float16",
    "bf16": TORCH_DTYPE == "bfloat16",
    "gradient_checkpointing": False,  # already enabled in Cell 7, with the vision tower excluded
    "group_by_length": False,  # implemented in DigitAwareLengthGroupedSampler instead
    "dataloader_num_workers": DATALOADER_WORKERS,
    "dataloader_pin_memory": True,
    "remove_unused_columns": False,
    "label_names": ["labels"],
    "report_to": [],
    "seed": SEED,
    "data_seed": SEED,
    "disable_tqdm": False,
    "eval_strategy": "steps" if LOSS_EVAL_ENABLED else "no",
    "eval_steps": SAVE_STEPS,
}
if SMOKE_TEST:
    training_kwargs.update(
        {
            "max_steps": SMOKE_TEST_STEPS,
            "num_train_epochs": 1.0,
            "save_steps": max(SMOKE_TEST_STEPS, 1),
            "eval_strategy": "no",
            "logging_steps": 1,
        }
    )

# `eval_strategy` was `evaluation_strategy` before transformers 4.46; filter to what this build takes.
_accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
if "eval_strategy" not in _accepted and "evaluation_strategy" in _accepted:
    training_kwargs["evaluation_strategy"] = training_kwargs.pop("eval_strategy")
_dropped = sorted(set(training_kwargs) - _accepted)
for key in _dropped:
    training_kwargs.pop(key)
if _dropped:
    print(f"NOTE: TrainingArguments in transformers {VERSIONS['transformers']} ignores {_dropped}")

training_args = TrainingArguments(**training_kwargs)


class PeakMemoryCallback(TrainerCallback):
    """Print peak CUDA allocation after the first optimizer step and at the end of training.

    This is the empirical signal for tuning ``per_device_batch_size``; see the markdown above.
    """

    def __init__(self) -> None:
        self.reported_first = False
        self.started_at = 0.0

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        self.started_at = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs) -> None:
        if self.reported_first or not torch.cuda.is_available():
            return
        self.reported_first = True
        print(
            f"\n[memory] after step 1: peak allocated="
            f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, reserved="
            f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB of "
            f"{RUNTIME['vram_gb']:.1f} GB. Raise per_device_batch_size only against this number.\n"
        )

    def on_train_end(self, args, state, control, **kwargs) -> None:
        elapsed = time.time() - self.started_at
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print(
            f"\n[memory] training done in {elapsed / 60:.1f} min; peak allocated={peak:.2f} GiB; "
            f"{state.global_step} steps"
        )


class PersianOcrTrainer(Trainer):
    """`Trainer` with the digit-aware, length-grouped sampler substituted in."""

    def _get_train_sampler(self, train_dataset=None) -> Sampler:  # noqa: ANN001
        return DigitAwareLengthGroupedSampler(
            lengths=TRAIN_LENGTHS,
            weights=TRAIN_WEIGHTS,
            batch_size=self.args.per_device_train_batch_size,
            group_by_length=GROUP_BY_LENGTH,
            seed=SEED,
        )


trainer = PersianOcrTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if LOSS_EVAL_ENABLED and not SMOKE_TEST else None,
    processing_class=processor,
    optimizers=(optimizer, None),
    callbacks=[PeakMemoryCallback()],
)

# Cell 7 set `accepts_loss_kwargs = False` on the model so that Trainer keeps normalising the loss by
# the accumulation count. Verify it survived the PEFT wrapping rather than trusting it.
if trainer.model_accepts_loss_kwargs:
    trainer.model_accepts_loss_kwargs = False
    print(
        "NOTE: forced trainer.model_accepts_loss_kwargs = False - the model does not consume "
        "num_items_in_batch, so without this the gradients would be scaled by "
        f"{grad_accum_steps}x (see Cell 7)."
    )

# ------------------------------------------------- resume detection
resume_target: str | bool | None = None
if RESUME_FROM_CHECKPOINT and os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint:
        resume_target = last_checkpoint
        print(f"resuming from {last_checkpoint}")
    else:
        print(f"no checkpoint found under {OUTPUT_DIR}; starting from scratch")

print(
    f"\nstarting training: phase={PHASE} dtype={TORCH_DTYPE} samples={len(train_dataset):,} "
    f"smoke_test={SMOKE_TEST}"
)
train_result = trainer.train(resume_from_checkpoint=resume_target)

trainer.save_state()
TRAIN_METRICS = dict(train_result.metrics)
TRAIN_METRICS["effective_batch_size"] = EFFECTIVE_BATCH_SIZE
TRAIN_METRICS["peak_gib"] = (
    round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
)
print("\n--- training metrics ---")
for key, value in TRAIN_METRICS.items():
    print(f"  {key}: {value}")

# %% [markdown]
# ## Cell 9 — Evaluation: CER, WER, digit fidelity, substitution confusion
#
# Four numbers, because one is not enough for this task:
#
# 1. **CER / WER, raw** — the headline text metric.
# 2. **CER after Persian normalisation** — `PERSIAN_NORMALIZATION_MAP` folds the Arabic/Persian
#    look-alikes (`ي`→`ی`, `ك`→`ک`), ZWNJ and all three digit scripts. The gap between raw and
#    normalised CER is the share of "error" that is orthographic variance rather than misreading.
# 3. **Digit CER + digit-sequence exact match** — every digit in the reference, in order, as its own
#    string. A registration number is wrong if one glyph is wrong, and a 2 % overall CER can hide a
#    30 % digit error rate because digits are a few percent of the characters on a page.
# 4. **Substitution confusion table** — which reference character became which predicted character,
#    ranked. This is the diagnostic that tells you *what* to add to the synthetic generator: a
#    ۴↔4 or ی↔ي block at the top means the font/normalisation mix is wrong, not the model.
#
# Metrics are also broken out per `source_tag`, since admin letters and Persian_Pixel pages are
# different difficulty regimes and a single average hides which one improved.
#
# Generation uses the **same prompt token sequence as training** (Cell 6's `PROMPT_TEMPLATE_IDS` with
# the placeholder expanded), so the boundary the model was trained on is the boundary it is asked at.
#
# ### Pilot learning curve
# In `PHASE == "pilot"` the cell appends `(target_samples, val_CER, digit_CER)` to
# `WORK_DIR/pilot_learning_curve.json` and re-plots the accumulated curve. Run the pilot at 3k → 8k →
# 15k → 25k and read where the curve flattens: that is the point past which more data buys little, and
# it is what `PHASE_CONFIG["final"]["target_samples"]` should be set from. Entries are keyed by
# `target_samples`, so re-running a size overwrites its point instead of duplicating it.

# %%
"""Cell 9 - generation-based evaluation, digit metrics, confusion table, learning curve."""

import difflib
import math
import time
from collections import Counter

import numpy as np

try:
    import jiwer

    _HAS_JIWER = True
except ImportError:  # pragma: no cover - jiwer is in PIP_REQUIREMENTS
    _HAS_JIWER = False
    print("WARNING: jiwer not importable; using the built-in edit-distance fallback.")


def edit_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    """Levenshtein distance between two sequences, computed row-wise with numpy."""
    if len(reference) == 0:
        return len(hypothesis)
    previous = np.arange(len(reference) + 1, dtype=np.int32)
    for index, symbol in enumerate(hypothesis, start=1):
        current = np.empty_like(previous)
        current[0] = index
        matches = np.fromiter((symbol != char for char in reference), dtype=bool, count=len(reference))
        for position in range(1, len(reference) + 1):
            current[position] = min(
                previous[position] + 1,
                current[position - 1] + 1,
                previous[position - 1] + int(matches[position - 1]),
            )
        previous = current
    return int(previous[-1])


def corpus_cer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level character error rate (total edits / total reference characters)."""
    if _HAS_JIWER:
        return float(jiwer.cer(references, hypotheses))
    edits = sum(edit_distance(ref, hyp) for ref, hyp in zip(references, hypotheses, strict=True))
    return edits / max(sum(len(ref) for ref in references), 1)


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level word error rate on whitespace-split tokens."""
    if _HAS_JIWER:
        return float(jiwer.wer(references, hypotheses))
    edits = sum(edit_distance(ref.split(), hyp.split()) for ref, hyp in zip(references, hypotheses, strict=True))
    return edits / max(sum(len(ref.split()) for ref in references), 1)


def normalize_persian(text: str) -> str:
    """Fold Persian/Arabic look-alikes, ZWNJ and all digit scripts onto one canonical form."""
    for source, target in PERSIAN_NORMALIZATION_MAP.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def digits_only(text: str) -> str:
    """Every digit in the text, in order, normalised to Latin digits."""
    normalized = text
    for source, target in PERSIAN_NORMALIZATION_MAP.items():
        if target.isdigit():
            normalized = normalized.replace(source, target)
    return "".join(char for char in normalized if char.isdigit() and char.isascii())


@torch.inference_mode()
def generate_transcription(record: dict[str, Any], max_new_tokens: int = EVAL_MAX_NEW_TOKENS) -> str:
    """Run the adapted model on one record and return its predicted transcription.

    Args:
        record: An index record from Cell 3/4.
        max_new_tokens: Generation cap. Targets longer than this are truncated, which shows up as
            deletion errors; the share of affected records is reported below.

    Returns:
        The decoded prediction with the template's answer suffix stripped.
    """
    device = next(model.parameters()).device
    prepared = prepare_image(load_pil_image(record), EFFECTIVE_MAX_PIXELS)
    image_inputs = image_processor(images=prepared, return_tensors="pt")
    grid = image_inputs["image_grid_thw"]
    n_image_tokens = int(grid[0].prod().item()) // (MERGE_SIZE * MERGE_SIZE)
    prompt_ids = (
        PROMPT_TEMPLATE_IDS[:IMAGE_TOKEN_POSITION]
        + [IMAGE_TOKEN_ID] * n_image_tokens
        + PROMPT_TEMPLATE_IDS[IMAGE_TOKEN_POSITION + 1 :]
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=image_inputs["pixel_values"].to(device),
        image_grid_thw=grid.to(device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,  # the shipped generation_config disables it, which is ~10x slower
        pad_token_id=PAD_TOKEN_ID,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion = generated[0, input_ids.shape[1] :]
    text = tokenizer.decode(completion, skip_special_tokens=True)
    suffix = ANSWER_SUFFIX.strip()
    if suffix and text.endswith(suffix):
        text = text[: -len(suffix)]
    return text.strip()


def evaluate(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Generate predictions for ``records`` and compute the full metric set.

    Args:
        records: Validation records to evaluate.
        label: Name used in progress output.

    Returns:
        Metrics mapping, plus the raw ``pairs`` so the confusion table can be built from them.
    """
    model.eval()
    references, hypotheses, tags, skipped = [], [], [], 0
    started = time.time()
    for position, record in enumerate(records, start=1):
        try:
            hypotheses.append(generate_transcription(record))
            references.append(record["text"])
            tags.append(record["source_tag"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("eval OOM on uid=%s; skipped", record["uid"])
            skipped += 1
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            logger.warning("eval failed uid=%s (%s: %s)", record["uid"], type(exc).__name__, exc)
            skipped += 1
        if position % 25 == 0 or position == len(records):
            rate = (time.time() - started) / position
            print(
                f"  [{label}] {position}/{len(records)} "
                f"({rate:.2f}s/sample, ~{rate * (len(records) - position) / 60:.1f} min left)"
            )
    if not references:
        raise RuntimeError("every evaluation sample failed; see the skipped-records log")

    normalized_refs = [normalize_persian(text) for text in references]
    normalized_hyps = [normalize_persian(text) for text in hypotheses]
    digit_refs = [digits_only(text) for text in references]
    digit_hyps = [digits_only(text) for text in hypotheses]
    with_digits = [i for i, ref in enumerate(digit_refs) if ref]

    metrics: dict[str, Any] = {
        "n_evaluated": len(references),
        "n_skipped": skipped,
        "cer_raw": corpus_cer(references, hypotheses),
        "wer_raw": corpus_wer(references, hypotheses),
        "cer_normalized": corpus_cer(normalized_refs, normalized_hyps),
        "wer_normalized": corpus_wer(normalized_refs, normalized_hyps),
        "digit_cer": (
            corpus_cer([digit_refs[i] for i in with_digits], [digit_hyps[i] for i in with_digits])
            if with_digits
            else float("nan")
        ),
        "digit_exact_match": (
            sum(1 for i in with_digits if digit_refs[i] == digit_hyps[i]) / len(with_digits)
            if with_digits
            else float("nan")
        ),
        "n_with_digits": len(with_digits),
        "empty_predictions": sum(1 for hyp in hypotheses if not hyp.strip()),
        "seconds_per_sample": (time.time() - started) / len(records),
    }
    for tag in sorted(set(tags)):
        indices = [i for i, t in enumerate(tags) if t == tag]
        metrics[f"cer_{tag}"] = corpus_cer(
            [references[i] for i in indices], [hypotheses[i] for i in indices]
        )
    metrics["pairs"] = list(zip(references, hypotheses, tags, strict=True))
    return metrics


def substitution_confusion(
    pairs: list[tuple[str, str, str]], top_n: int = CONFUSION_TOP_N
) -> list[tuple[str, str, int]]:
    """Rank character substitutions by frequency.

    Uses ``difflib`` opcodes on the raw (un-normalised) strings, so orthographic look-alikes show up
    explicitly instead of being folded away.

    Args:
        pairs: ``(reference, hypothesis, source_tag)`` triples.
        top_n: How many substitutions to return.

    Returns:
        ``(reference_char, predicted_char, count)`` triples, most frequent first.
    """
    counter: Counter[tuple[str, str]] = Counter()
    for reference, hypothesis, _ in pairs:
        matcher = difflib.SequenceMatcher(None, reference, hypothesis, autojunk=False)
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "replace" and (i2 - i1) == (j2 - j1):
                for offset in range(i2 - i1):
                    counter[(reference[i1 + offset], hypothesis[j1 + offset])] += 1
            elif op == "delete":
                for offset in range(i2 - i1):
                    counter[(reference[i1 + offset], "∅")] += 1
            elif op == "insert":
                for offset in range(j2 - j1):
                    counter[("∅", hypothesis[j1 + offset])] += 1
    return [(ref, hyp, count) for (ref, hyp), count in counter.most_common(top_n)]


EVAL_RECORDS = VALIDATION_RECORDS
_eval_cap = SMOKE_TEST_EVAL_SAMPLES if SMOKE_TEST else CFG["eval_max_samples"]
if _eval_cap is not None:
    EVAL_RECORDS = EVAL_RECORDS[: int(_eval_cap)]
_over_cap = sum(1 for r in EVAL_RECORDS if r["n_answer_tokens"] > EVAL_MAX_NEW_TOKENS)
print(
    f"evaluating {len(EVAL_RECORDS):,} of {len(VALIDATION_RECORDS):,} validation records "
    f"(cap={_eval_cap}); {_over_cap} of them have targets longer than "
    f"EVAL_MAX_NEW_TOKENS={EVAL_MAX_NEW_TOKENS} and will be truncated"
)

EVAL_METRICS = evaluate(EVAL_RECORDS, label=PHASE)
EVAL_PAIRS = EVAL_METRICS.pop("pairs")

print("\n--- validation metrics ---")
for key, value in EVAL_METRICS.items():
    print(f"  {key:24s} {value:.4f}" if isinstance(value, float) else f"  {key:24s} {value}")

print(f"\n--- top {CONFUSION_TOP_N} character substitutions (raw text) ---")
print(f"  {'reference':>12s} -> {'predicted':<12s} count")
for reference_char, predicted_char, count in substitution_confusion(EVAL_PAIRS):
    print(f"  {reference_char!r:>12s} -> {predicted_char!r:<12s} {count:,d}")

print("\n--- first two predictions (reference | prediction, truncated) ---")
for reference, hypothesis, tag in EVAL_PAIRS[:2]:
    print(f"\n  [{tag}] REF: {reference[:220]!r}")
    print(f"  [{tag}] HYP: {hypothesis[:220]!r}")

# %% [markdown]
# ### Cell 9b — Accumulated pilot learning curve
#
# Only meaningful in `PHASE == "pilot"`: each pilot run contributes one point. Two or more points draw
# the curve; the flattening region is where extra data stops paying for itself, and that is the number
# to promote into `PHASE_CONFIG["final"]["target_samples"]`.

# %%
"""Cell 9b - append this run's point to the learning curve and plot the accumulation."""

import datetime


def json_safe(value: float) -> float | None:
    """Round a metric for JSON, mapping NaN/inf to ``None`` so the file stays strict JSON."""
    return None if not math.isfinite(value) else round(float(value), 6)


if PHASE == "pilot" and not SMOKE_TEST:
    curve: list[dict[str, Any]] = []
    if os.path.exists(LEARNING_CURVE_PATH):
        with open(LEARNING_CURVE_PATH, encoding="utf-8") as handle:
            curve = json.load(handle)
    point = {
        "target_samples": CFG["target_samples"],
        "n_train_actual": len(train_dataset),
        "final_val_cer": json_safe(EVAL_METRICS["cer_raw"]),
        "final_val_cer_normalized": json_safe(EVAL_METRICS["cer_normalized"]),
        "digit_cer": json_safe(EVAL_METRICS["digit_cer"]),
        "digit_exact_match": json_safe(EVAL_METRICS["digit_exact_match"]),
        "n_evaluated": EVAL_METRICS["n_evaluated"],
        "train_loss": json_safe(float(TRAIN_METRICS.get("train_loss", float("nan")))),
        "gpu": RUNTIME["gpu_name"],
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    curve = [entry for entry in curve if entry["target_samples"] != point["target_samples"]]
    curve.append(point)
    curve.sort(key=lambda entry: entry["target_samples"])
    with open(LEARNING_CURVE_PATH, "w", encoding="utf-8") as handle:
        json.dump(curve, handle, ensure_ascii=False, indent=1)
    print(f"learning curve now has {len(curve)} point(s) -> {LEARNING_CURVE_PATH}")

    def series(key: str) -> list[float]:
        """Pull one metric across the curve, mapping missing/None entries to NaN for plotting."""
        return [float("nan") if entry.get(key) is None else float(entry[key]) for entry in curve]

    for entry, cer, digit in zip(
        curve, series("final_val_cer"), series("digit_cer"), strict=True
    ):
        print(f"  {entry['target_samples']:>7,d} samples: CER={cer:.4f} digit_CER={digit:.4f}")

    if len(curve) >= 2:
        import matplotlib.pyplot as plt

        sizes = [entry["target_samples"] for entry in curve]
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        axis.plot(sizes, series("final_val_cer"), "o-", label="CER (raw)")
        axis.plot(sizes, series("final_val_cer_normalized"), "s--", label="CER (normalized)")
        axis.plot(sizes, series("digit_cer"), "^-", label="digit CER")
        axis.set_xlabel("training samples")
        axis.set_ylabel("error rate")
        axis.set_title(f"Pilot learning curve - fixed {VALIDATION_SIZE:,}-row validation set")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        plt.show()
    else:
        print("only one point so far - run the other pilot sizes to see the curve.")
else:
    print(f"PHASE={PHASE!r} (smoke_test={SMOKE_TEST}): learning curve not updated.")

# %% [markdown]
# ## Cell 10 — Save the adapter, the merged model, and the README
#
# Three artefacts, in order of usefulness:
#
# 1. **`adapter_only/`** — the LoRA weights plus `adapter_config.json`, the processor, and a
#    `training_config.json` that records every Cell 0 value this run actually used. A few dozen MB,
#    and the thing to keep. Loading it needs the base model to be downloaded again at inference time.
# 2. **`merged_model/`** — LoRA folded into the base weights (`merge_and_unload`), a standalone
#    checkpoint of ~1.8 GB at fp16/bf16. Convenient for deployment, wasteful for storage. The custom
#    `trust_remote_code` modules are copied in alongside it, so the folder loads without reaching for
#    the original repo.
# 3. **`README.md`** — a model card carrying the config, the split counts, the metrics from Cell 9,
#    and the exact prompt format. Written into both folders.
#
# **Merging destroys the training model in memory** (`merge_and_unload` unwraps the PEFT layers), so
# this cell runs *after* evaluation, and re-running Cell 9 afterwards requires re-running Cell 7-8.
#
# ### Hub push
# `PUSH_TO_HUB` is `False` in Cell 0 and this cell does nothing outward-facing until you set it. When
# enabled it uploads `PUSH_WHAT` (`"adapter"` or `"merged"`) to `HF_MODEL_REPO` and prints exactly
# what is going where first. A push publishes under your token — a public repo is world-readable and
# gets indexed, so make the repo private first if the letters are not yours to release.

# %%
"""Cell 10 - persist the adapter, the merged model, the model card, and the run summary."""

import shutil
from collections import Counter

ADAPTER_DIR = f"{OUTPUT_DIR}/adapter_only"
MERGED_DIR = f"{OUTPUT_DIR}/merged_model"

TRAIN_TAG_COUNTS = Counter(record["source_tag"] for record in TRAIN_RECORDS)
VAL_TAG_COUNTS = Counter(record["source_tag"] for record in VALIDATION_RECORDS)

RUN_SUMMARY: dict[str, Any] = {
    "phase": PHASE,
    "smoke_test": SMOKE_TEST,
    "model_id": MODEL_ID,
    "runtime": {**RUNTIME, "compute_capability": list(RUNTIME["compute_capability"])},
    "versions": VERSIONS,
    "phase_config": dict(CFG),
    "dtype": TORCH_DTYPE,
    "attn_implementation": ATTN_IMPLEMENTATION,
    "lora": {
        "r": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
        "target_leaf_names": LORA_TARGET_MODULES,
        "excluded_prefixes": LORA_EXCLUDE_MODULE_PREFIXES,
        "n_target_modules": len(LORA_TARGET_PATHS),
        "n_excluded_modules": len(LORA_EXCLUDED_PATHS),
        "trainable_params": n_trainable,
        "total_params": n_total,
    },
    "optimization": {
        "lr_attn_mlp": LR_ATTN_MLP,
        "lr_embed_head": LR_EMBED_HEAD,
        "optim": OPTIM,
        "lr_scheduler": LR_SCHEDULER,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "seed": SEED,
    },
    "data": {
        "admin_letters_repo": HF_ADMIN_LETTERS_REPO,
        "persian_pixel_repo": HF_PERSIAN_PIXEL_REPO,
        "train_counts": dict(TRAIN_TAG_COUNTS),
        "validation_counts": dict(VAL_TAG_COUNTS),
        "n_train": len(TRAIN_RECORDS),
        "n_validation": len(VALIDATION_RECORDS),
        "validation_seed": VALIDATION_SEED,
        "train_max_pixels": CFG["train_max_pixels"],
        "effective_max_pixels": EFFECTIVE_MAX_PIXELS,
        "max_total_seq_len": MAX_TOTAL_SEQ_LEN,
        "digit_oversample_multiplier": DIGIT_OVERSAMPLE_MULTIPLIER,
        "digit_weight_threshold": round(DIGIT_WEIGHT_THRESHOLD, 6),
        "prompt": PROMPTS[TRAIN_TASK],
    },
    "train_metrics": {k: v for k, v in TRAIN_METRICS.items()},
    "eval_metrics": {
        key: (json_safe(value) if isinstance(value, float) else value)
        for key, value in EVAL_METRICS.items()
    },
}

# ------------------------------------------------------------------ adapter
if SAVE_ADAPTER:
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)
    with open(f"{ADAPTER_DIR}/training_config.json", "w", encoding="utf-8") as handle:
        json.dump(RUN_SUMMARY, handle, ensure_ascii=False, indent=2)
    adapter_bytes = sum(
        os.path.getsize(os.path.join(ADAPTER_DIR, name)) for name in os.listdir(ADAPTER_DIR)
    )
    print(f"adapter saved -> {ADAPTER_DIR} ({adapter_bytes / 2**20:.1f} MiB)")
    archive = shutil.make_archive(f"{OUTPUT_DIR}/adapter_{PHASE}", "zip", ADAPTER_DIR)
    print(f"  downloadable archive: {archive}")
else:
    print("SAVE_ADAPTER is False; adapter not written.")


def copy_remote_code(target_dir: str) -> list[str]:
    """Copy the model's ``trust_remote_code`` modules next to a saved checkpoint.

    The config's ``auto_map`` points at module names (for example
    ``modeling_paddleocr_vl.PaddleOCRVLForConditionalGeneration``). Without those ``.py`` files in the
    folder, loading the merged checkpoint from disk fails even with ``trust_remote_code=True``.

    Args:
        target_dir: Directory holding the saved checkpoint.

    Returns:
        The filenames copied in.
    """
    from huggingface_hub import hf_hub_download

    auto_map = getattr(base_model.config, "auto_map", None) or {}
    modules = {str(value).split(".")[0] for value in auto_map.values() if "." in str(value)}
    for sub_config in ("vision_config", "text_config"):
        nested = getattr(base_model.config, sub_config, None)
        nested_map = getattr(nested, "auto_map", None) or {}
        modules.update(str(v).split(".")[0] for v in nested_map.values() if "." in str(v))
    copied: list[str] = []
    for module in sorted(modules):
        filename = f"{module}.py"
        try:
            source = hf_hub_download(MODEL_ID, filename, token=HF_TOKEN)
        except Exception as exc:  # missing file, offline, gated repo
            logger.warning("could not fetch %s from %s: %s", filename, MODEL_ID, exc)
            continue
        shutil.copyfile(source, os.path.join(target_dir, filename))
        copied.append(filename)
    return copied


# ------------------------------------------------------------------ merged model
if SAVE_MERGED_MODEL:
    print("\nmerging LoRA into the base weights (the training model is consumed by this) ...")
    merged_model = model.merge_and_unload()
    merged_model.config.use_cache = True  # the exported checkpoint is for inference
    if hasattr(merged_model, "gradient_checkpointing_disable"):
        merged_model.gradient_checkpointing_disable()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
    processor.save_pretrained(MERGED_DIR)
    copied_modules = copy_remote_code(MERGED_DIR)
    merged_bytes = sum(
        os.path.getsize(os.path.join(MERGED_DIR, name)) for name in os.listdir(MERGED_DIR)
    )
    print(f"merged model saved -> {MERGED_DIR} ({merged_bytes / 2**30:.2f} GiB)")
    print(f"  remote-code modules copied in: {copied_modules or 'none found'}")
else:
    print("\nSAVE_MERGED_MODEL is False; only the adapter was written.")

# ------------------------------------------------------------------ model card
_metric_rows = [
    ("CER (raw)", EVAL_METRICS["cer_raw"]),
    ("CER (Persian-normalised)", EVAL_METRICS["cer_normalized"]),
    ("WER (raw)", EVAL_METRICS["wer_raw"]),
    ("WER (Persian-normalised)", EVAL_METRICS["wer_normalized"]),
    ("Digit CER", EVAL_METRICS["digit_cer"]),
    ("Digit-sequence exact match", EVAL_METRICS["digit_exact_match"]),
]

_usage_snippet = '''```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

base = AutoModelForCausalLM.from_pretrained(
    "PaddlePaddle/PaddleOCR-VL", trust_remote_code=True, torch_dtype="auto", device_map="cuda"
)
model = PeftModel.from_pretrained(base, "<path-or-repo>/adapter_only").eval()
processor = AutoProcessor.from_pretrained("PaddlePaddle/PaddleOCR-VL", trust_remote_code=True)

messages = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "OCR:"}]}
]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[page_image], return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)
print(processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```'''

readme_parts: list[str] = [
    "---",
    "library_name: peft",
    f"base_model: {MODEL_ID}",
    "language:\n- fa",
    "tags:\n- ocr\n- persian\n- farsi\n- lora\n- document-understanding",
    "pipeline_tag: image-to-text",
    "---",
    "",
    "# PaddleOCR-VL LoRA adapter for Persian administrative letters",
    "",
    f"A LoRA adapter for [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) specialised on Persian "
    "(Farsi) official/administrative letters (نامه‌های اداری). The vision encoder and the patch "
    "projector are frozen; only the ERNIE-4.5 decoder, its input embedding and its output head are "
    "adapted.",
    "",
    "## Training data",
    "",
    "| Source | Train rows | Validation rows |",
    "| --- | --- | --- |",
]
for tag in sorted(set(TRAIN_TAG_COUNTS) | set(VAL_TAG_COUNTS)):
    readme_parts.append(f"| `{tag}` | {TRAIN_TAG_COUNTS.get(tag, 0):,} | {VAL_TAG_COUNTS.get(tag, 0):,} |")
readme_parts += [
    f"| **total** | **{len(TRAIN_RECORDS):,}** | **{len(VALIDATION_RECORDS):,}** |",
    "",
    f"The validation split is a fixed carve-out (seed `{VALIDATION_SEED}`, size `{VALIDATION_SIZE:,}`), "
    "stratified by digit density and identical across every run, so numbers from different runs are "
    "directly comparable. Samples in the top digit-density quartile are oversampled during training "
    f"by {DIGIT_OVERSAMPLE_MULTIPLIER}x, because registration numbers and dates are the fields that "
    "matter most on this document type.",
    "",
    "## Evaluation",
    "",
    f"Greedy decoding, `max_new_tokens={EVAL_MAX_NEW_TOKENS}`, on "
    f"{EVAL_METRICS['n_evaluated']:,} held-out pages.",
    "",
    "| Metric | Value |",
    "| --- | --- |",
]
for label, value in _metric_rows:
    readme_parts.append(f"| {label} | {value:.4f} |" if math.isfinite(value) else f"| {label} | n/a |")
readme_parts += [
    "",
    "Per-source CER: "
    + ", ".join(
        f"`{key[4:]}` {value:.4f}"
        for key, value in EVAL_METRICS.items()
        if key.startswith("cer_") and key not in {"cer_raw", "cer_normalized"}
    ),
    "",
    "## Configuration",
    "",
    "| Setting | Value |",
    "| --- | --- |",
    f"| phase | `{PHASE}` |",
    f"| rank / alpha / dropout | {LORA_R} / {LORA_ALPHA} / {LORA_DROPOUT} |",
    f"| target modules | {', '.join(f'`{m}`' for m in LORA_TARGET_MODULES)} |",
    f"| adapted modules | {len(LORA_TARGET_PATHS)} (vision stack excluded: {len(LORA_EXCLUDED_PATHS)}) |",
    f"| trainable params | {n_trainable:,} ({n_trainable / n_total:.3%} of {n_total:,}) |",
    f"| learning rate | {LR_ATTN_MLP:.0e} attention/MLP, {LR_EMBED_HEAD:.0e} embedding/head |",
    f"| schedule | {LR_SCHEDULER}, warmup {WARMUP_RATIO}, weight decay {WEIGHT_DECAY} |",
    f"| optimizer | `{OPTIM}` |",
    f"| epochs | {CFG['num_epochs']} |",
    f"| effective batch size | {EFFECTIVE_BATCH_SIZE} |",
    f"| precision | {TORCH_DTYPE} on {RUNTIME['gpu_name']} |",
    f"| max image pixels | {EFFECTIVE_MAX_PIXELS:,} (<= {MAX_IMAGE_TOKENS:,} image tokens) |",
    f"| max sequence length | {MAX_TOTAL_SEQ_LEN:,} tokens |",
    "",
    "Full machine-readable configuration: `training_config.json`.",
    "",
    "## Usage",
    "",
    f"The prompt must match training exactly - the task string is `{PROMPTS[TRAIN_TASK]}` and the "
    "chat template supplies the image markers:",
    "",
    _usage_snippet,
    "",
    "## Limitations",
    "",
    "* Persian only. The adapter shifts decoder probabilities towards Persian script and digits and "
    "will degrade the base model's other languages.",
    f"* Trained at up to {EFFECTIVE_MAX_PIXELS:,} pixels per page; much larger scans should be "
    "downscaled to that budget rather than fed at native resolution.",
    "* Targets longer than the generation cap are truncated rather than continued.",
    "",
]
README_TEXT = "\n".join(readme_parts)

for directory in [d for d, enabled in ((ADAPTER_DIR, SAVE_ADAPTER), (MERGED_DIR, SAVE_MERGED_MODEL)) if enabled]:
    with open(f"{directory}/README.md", "w", encoding="utf-8") as handle:
        handle.write(README_TEXT)
    print(f"model card written -> {directory}/README.md")

with open(f"{OUTPUT_DIR}/run_summary.json", "w", encoding="utf-8") as handle:
    json.dump(RUN_SUMMARY, handle, ensure_ascii=False, indent=2)
print(f"run summary written -> {OUTPUT_DIR}/run_summary.json")

# %% [markdown]
# ### Cell 10b — Optional: publish to the Hugging Face Hub
#
# Off by default. This is the only cell that sends anything off the machine: it uploads a folder to
# `HF_MODEL_REPO` under your `HF_TOKEN`. A public repo is world-readable and indexed, and the model
# card above quotes dataset counts and metrics — check that both are yours to publish before enabling
# `PUSH_TO_HUB` in Cell 0.

# %%
"""Cell 10b - optional Hub upload, gated on PUSH_TO_HUB."""

if not PUSH_TO_HUB:
    print(
        "PUSH_TO_HUB is False - nothing was uploaded.\n"
        f"  To publish later: set PUSH_TO_HUB = True and PUSH_WHAT ('adapter' or 'merged') in Cell 0,\n"
        f"  then re-run this cell. Target repo would be: {HF_MODEL_REPO}"
    )
else:
    from huggingface_hub import HfApi

    push_dir = ADAPTER_DIR if PUSH_WHAT == "adapter" else MERGED_DIR
    if not os.path.isdir(push_dir):
        raise FileNotFoundError(
            f"PUSH_WHAT={PUSH_WHAT!r} but {push_dir} does not exist; enable the matching "
            "SAVE_ADAPTER / SAVE_MERGED_MODEL flag and re-run the previous cell."
        )
    payload = sorted(os.listdir(push_dir))
    total_bytes = sum(os.path.getsize(os.path.join(push_dir, name)) for name in payload)
    print("=" * 78)
    print("PUBLISHING TO THE HUGGING FACE HUB")
    print(f"  from : {push_dir}")
    print(f"  to   : https://huggingface.co/{HF_MODEL_REPO}")
    print(f"  size : {total_bytes / 2**20:.1f} MiB across {len(payload)} files")
    print(f"  files: {', '.join(payload)}")
    print("=" * 78)

    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_MODEL_REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=HF_MODEL_REPO,
        folder_path=push_dir,
        repo_type="model",
        commit_message=f"LoRA adapter for Persian letters ({PHASE} phase, CER={EVAL_METRICS['cer_raw']:.4f})",
    )
    print(
        f"\nuploaded to https://huggingface.co/{HF_MODEL_REPO} (created PRIVATE; flip visibility in "
        "the repo settings when you are ready)"
    )

# %% [markdown]
# ## Done — what to do next
#
# * **Pilot phase:** change `PHASE_CONFIG["pilot"]["target_samples"]` to the next size (3000 → 8000 →
#   15000 → 25000) and re-run the whole notebook. Cell 9b accumulates the points into
#   `pilot_learning_curve.json` and re-plots the curve after the second run.
# * **Then the final run:** set `PHASE = "final"`, copy the flattening point into
#   `PHASE_CONFIG["final"]["target_samples"]`, and run on the A100. `VALIDATION_SEED` must not change,
#   or the final numbers stop being comparable to the pilot curve.
# * **Check `skipped_records.log`** in `OUTPUT_DIR` before trusting any number: it lists every record
#   that failed to decode, was dropped for exceeding `MAX_TOTAL_SEQ_LEN`, or errored during
#   evaluation.
# * **Read the confusion table from Cell 9.** Top entries dominated by digit or yeh/kaf substitutions
#   point at the data mix (fonts, normalisation) rather than at model capacity — fix that before
#   raising `LORA_R`.
