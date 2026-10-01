# %% [markdown]
# # PaddleOCR-VL **1.6** Persian probe — tokenizer fertility, decoder capability, OCR error report
#
# A **read-only diagnostic** that answers one question before any LoRA run: *does the frozen
# `PaddlePaddle/PaddleOCR-VL-1.6` stack actually model Persian, and if not, is the tokenizer, the
# language decoder, or the vision encoder the bottleneck?*
#
# It produces three artefacts:
#
# | Cell | Question | Output |
# | --- | --- | --- |
# | 5 | How well does the ERNIE tokenizer digest Persian vs English? | fertility, whole-word rate, alphabet coverage, round-trip fidelity |
# | 6 | Does the **language decoder** assign plausible probability to Persian? | teacher-forced loss / bits-per-char on the same parallel EN+FA text |
# | 8-9 | What is the *end-to-end* error on real pages? | per-sample + corpus CER, WER, line-structure error, digit accuracy |
#
# ### API note — 1.6 is not the 0.9B remote-code model
# PaddleOCR-VL-1.6 ships as a **native transformers model**. The card's own example uses
# `AutoModelForImageTextToText` + `AutoProcessor` with **no `trust_remote_code`**, requires
# **`transformers>=5.0.0`**, and passes the image through
# `apply_chat_template(..., images_kwargs={"size": {...}})`. That is what this notebook does. (The
# older `PaddlePaddle/PaddleOCR-VL` 0.9B checkpoint is the one that needed remote code and the
# handwritten image-token expansion; none of that applies here.)
#
# ### Why this matters for LoRA
# A LoRA adapter perturbs the decoder's attention/MLP (and optionally embeddings). It cannot teach
# the tokenizer new units: if Persian words shatter into 4-6 byte-level pieces each, the effective
# context shrinks and the adapter has to relearn spelling before it can learn layout. If the decoder's
# teacher-forced loss on Persian is already near its English loss, the capacity is there and the LoRA
# is a fair bet.
#
# ### How to run on Colab
# Runtime → Change runtime type → **CPU** works, but generation on CPU is slow; the defaults
# auto-shrink the pixel budget and token cap on CPU. A T4 is far faster for Cells 8-9. Run the cells
# top to bottom. You need a Hugging Face token in Colab *Secrets* under the key `HF_TOKEN` (the
# download is public but rate-limited without one).

# %% [markdown]
# ## Cell 0 — Configuration
#
# Everything you might want to change lives here. `N_SAMPLES=10` is the sample count you asked for.

# %%
"""Cell 0 - user configuration."""

from pathlib import Path

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"

# Prediction target: read real pages, or evaluate the adapter once you have one.
ADAPTER_PATH: str | None = None  # e.g. "shahriarhd/paddleocr-vl-persian-lora/adapter_only"

# --- labelled data ---------------------------------------------------------
# "auto" -> use the local manifest if this checkout has it, otherwise the HF Hub mirror.
DATA_SOURCE = "auto"
LOCAL_JSONL = Path("data/dataset/paddle_ocr_vl.jsonl")
LOCAL_IMAGE_ROOT = Path("data/dataset")
HF_DATA_REPO = "shahriarhd/persian-official-letters-synth"
HF_DATA_CONFIG = "images"
HF_DATA_SPLIT = "train"

N_SAMPLES = 10
SAMPLE_STRIDE = 997  # prime stride over the manifest, spread across letters
SAMPLE_SEED = 42

# --- what to run -----------------------------------------------------------
RUN_TOKENIZER_REPORT = True
RUN_DECODER_PROBE = True
RUN_OCR = True

# Generation budget. Reduced automatically on CPU in Cell 1.
EVAL_MAX_NEW_TOKENS = 1024

# --- output ----------------------------------------------------------------
REPORT_PATH = Path("/content/paddleocr_vl_persian_probe.json")
PRINT_PREVIEW_CHARS = 400

# %% [markdown]
# ## Cell 1 — Environment and precision
#
# Detect Colab vs a plain host, whether a GPU is present, and pick the pixel/generation budget.
# PaddleOCR-VL-1.6's default OCR budget is `1280 * 28 * 28 = 1_003_520` px; its "spotting" mode uses
# `2048 * 28 * 28`. On CPU we keep the 1M default and cap generation, because attention over image
# tokens dominates runtime.

# %%
"""Cell 1 - environment and precision detection."""

import platform
import sys


def detect_environment() -> dict:
    """Detect host, device and precision capability."""
    try:
        import google.colab  # noqa: F401

        env = "colab"
    except ImportError:
        env = "generic_jupyter"

    try:
        import torch

        torch_available = True
        cuda = torch.cuda.is_available()
        if cuda:
            name = torch.cuda.get_device_name(0)
            capability = tuple(torch.cuda.get_device_capability(0))
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            name, capability, vram_gb = "cpu", (0, 0), 0.0
    except ImportError:
        torch_available, cuda = False, False
        name, capability, vram_gb = "none", (0, 0), 0.0

    return {
        "env": env,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_available": torch_available,
        "cuda": cuda,
        "gpu_name": name,
        "compute_capability": capability,
        "vram_gb": vram_gb,
        "supports_bf16": capability[0] >= 8,
        "device": "cuda" if cuda else "cpu",
    }


RUNTIME = detect_environment()

# PaddleOCR-VL-1.6 OCR default: 1280*28*28. GPU/spotting can afford 2048*28*28.
MAX_PIXELS = 2048 * 28 * 28 if RUNTIME["cuda"] else 1280 * 28 * 28
if not RUNTIME["cuda"]:
    EVAL_MAX_NEW_TOKENS = min(EVAL_MAX_NEW_TOKENS, 512)

for key, value in RUNTIME.items():
    print(f"{key:20} = {value}")
print(f"{'MAX_PIXELS':20} = {MAX_PIXELS:,}")
print(f"{'EVAL_MAX_NEW_TOKENS':20} = {EVAL_MAX_NEW_TOKENS}")

# %% [markdown]
# ## Cell 2 — Dependencies
#
# PaddleOCR-VL-1.6 requires **`transformers>=5.0.0`**; the card's `AutoModelForImageTextToText` path is
# native and does not use remote code. `peft` is only installed when you point `ADAPTER_PATH` at a
# LoRA adapter. `jiwer` is optional (it speeds up nothing here, but its `cer`/`wer` are the reference
# implementation if you want to cross-check).

# %%
"""Cell 2 - dependency install."""

PIP_REQUIREMENTS = [
    "transformers>=5.0.0,<6",
    "accelerate>=1.6,<2",
    "datasets>=3.6,<6",
    "einops>=0.8",
    "jiwer>=3.0",
    "pillow>=10",
]
if ADAPTER_PATH:
    PIP_REQUIREMENTS.append("peft>=0.17,<0.19")

if RUNTIME["env"] == "colab":
    # fmt: off
    _pip_args = " ".join(f'"{req}"' for req in PIP_REQUIREMENTS)
    get_ipython().run_line_magic("pip", f"install -q -U {_pip_args}")  # noqa: F821
    # fmt: on
    print("installed:", ", ".join(PIP_REQUIREMENTS))
else:
    print("not Colab; skipping install. Required:", ", ".join(PIP_REQUIREMENTS))

# %% [markdown]
# ## Cell 3 — Versions, token, imports
#
# The token is read from Colab's Secrets panel (`HF_TOKEN`) and falls back to `os.environ`. It is
# never printed — only its presence and length. The transformers major version is asserted because
# 1.6 will not load its processor on the 4.x line.

# %%
"""Cell 3 - version report, secrets, shared imports."""

import json
import logging
import os
from io import BytesIO

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("persian_probe")


def get_secret(name: str) -> str | None:
    """Read a Colab secret, falling back to the process environment."""
    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        if value:
            return value
    except Exception:  # noqa: BLE001 - SecretNotFoundError / NotebookAccessError / no colab
        pass
    return os.environ.get(name)


HF_TOKEN = get_secret("HF_TOKEN")

for package in ("torch", "transformers", "datasets", "einops", "jiwer", "PIL"):
    try:
        module = __import__(package)
        print(f"{package:14} {getattr(module, '__version__', 'unknown')}")
    except ImportError:
        print(f"{package:14} MISSING")

if not HF_TOKEN:
    logger.warning("HF_TOKEN not found; the Hub download may be rate-limited or fail")
else:
    print(f"HF_TOKEN       found ({len(HF_TOKEN)} chars)")

import transformers

if int(transformers.__version__.split(".")[0]) < 5:
    raise RuntimeError(
        f"transformers {transformers.__version__} is < 5; PaddleOCR-VL-1.6 needs >= 5.0.0. "
        "Re-run Cell 2 and restart the kernel."
    )

# %% [markdown]
# ## Cell 4 — Processor and tokenizer
#
# `AutoProcessor` (native for 1.6) gives the chat template, the image processor and the ERNIE
# tokenizer. We read the image processor's real limits rather than hard-coding them, so the budgets
# in Cell 1 can be validated against the checkpoint.

# %%
"""Cell 4 - processor and tokenizer."""

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer = getattr(processor, "tokenizer", None)
if tokenizer is None:
    raise RuntimeError(f"{type(processor).__name__} exposes no .tokenizer; cannot run the report")

image_processor = processor.image_processor
PATCH_SIZE = int(image_processor.patch_size)
MERGE_SIZE = int(image_processor.merge_size)
MODEL_MIN_PIXELS = int(image_processor.min_pixels)
MODEL_MAX_PIXELS = int(image_processor.max_pixels)
PIXELS_PER_IMAGE_TOKEN = PATCH_SIZE * PATCH_SIZE * MERGE_SIZE * MERGE_SIZE
EFFECTIVE_MAX_PIXELS = max(MODEL_MIN_PIXELS, min(MAX_PIXELS, MODEL_MAX_PIXELS))
MAX_IMAGE_TOKENS = EFFECTIVE_MAX_PIXELS // PIXELS_PER_IMAGE_TOKEN

print(f"processor      : {type(processor).__name__}")
print(f"tokenizer      : {type(tokenizer).__name__}  fast={tokenizer.is_fast}  vocab={tokenizer.vocab_size:,}")
print(
    f"image limits   : min={MODEL_MIN_PIXELS:,} max={MODEL_MAX_PIXELS:,} "
    f"patch={PATCH_SIZE} merge={MERGE_SIZE}"
)
print(f"effective budget: {EFFECTIVE_MAX_PIXELS:,} px -> <= {MAX_IMAGE_TOKENS:,} image tokens")

# %% [markdown]
# ## Cell 5 — Tokenizer fertility, English vs Persian
#
# The same six sentences in both languages (plus mixed-script ones). For each language we report:
#
# * **fertility** — tokens per whitespace word. English BPE lands near 1.2-1.4; a tokenizer that
#   never saw Persian lands at 3-6 because it falls back to byte/character pieces.
# * **compression** — characters per token (the reciprocal view).
# * **whole-word rate** — share of words encoded as a single token. This is the number the decoder
#   cares about: sub-word fragments must be reassembled by attention before any semantics.
# * **round-trip fidelity** — does `decode(encode(x)) == x` after Persian normalisation? Any loss is
#   information the model can never emit.
# * **alphabet coverage** — are the 32 Persian letters, Persian digits and ZWNJ present as *single*
#   tokens? Missing letters guarantee spelling errors.
#
# Mixed-script sentences (Persian text around Latin product names like `Kubernetes`) are included
# because that is what your administrative corpus actually contains.

# %%
"""Cell 5 - tokenizer fertility and coverage report."""

PERSIAN_NORMALIZATION_MAP = {
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
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
    "‌": " ",
    "ـ": "",
}


def normalize_persian(text: str) -> str:
    """Fold Persian/Arabic look-alikes and digit scripts, collapsing all whitespace.

    Used for CER/WER normalisation, where newlines carry no meaning.
    """
    for source, target in PERSIAN_NORMALIZATION_MAP.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def normalize_preserving_lines(text: str) -> str:
    """The same folding, but every newline is kept so line structure survives."""
    for source, target in PERSIAN_NORMALIZATION_MAP.items():
        text = text.replace(source, target)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(lines).strip()


# Parallel corpus: index i is the same meaning in both languages.
PARALLEL_EN = [
    "The meeting was postponed until next Tuesday because of the heavy rain.",
    "Please send the signed contract by email before the end of the month.",
    "The company registered its new subsidiary in Tehran in the year 1402.",
    "All employees must complete the training course before the deadline.",
    "Invoices must include the national ID and the registration number.",
    "The database was optimized using Kubernetes and PostgreSQL version 15.",
]
PARALLEL_FA = [
    "جلسه به دلیل بارش شدید باران تا سه‌شنبه آینده به تعویق افتاد.",
    "لطفاً قرارداد امضاشده را تا پایان ماه از طریق رایانامه ارسال کنید.",
    "شرکت زیرمجموعه جدید خود را در سال ۱۴۰۲ در تهران ثبت کرد.",
    "همه کارکنان باید دوره آموزشی را پیش از مهلت مقرر به پایان برسانند.",
    "صورت‌حساب‌ها باید شامل شناسه ملی و شماره ثبت باشند.",
    "پایگاه داده با استفاده از Kubernetes و PostgreSQL نسخه ۱۵ بهینه‌سازی شد.",
]

PERSIAN_LETTERS = list("آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءأإئؤة")
PERSIAN_DIGITS = list("۰۱۲۳۴۵۶۷۸۹")
EXTRA_SPECIALS = ["‌", "،", "؛", "؟", "»", "«"]  # ZWNJ, comma, semicolon, question, guillemets


def tokenize_words(text: str) -> list[int]:
    """Token count of a text with no special tokens added."""
    return tokenizer(text, add_special_tokens=False).input_ids


def fertility_report(lines: list[str], label: str) -> dict:
    """Aggregate tokenizer statistics over a list of texts."""
    total_tokens = total_words = total_chars = 0
    single_words = 0
    exact_roundtrips = 0
    for line in lines:
        ids = tokenize_words(line)
        total_tokens += len(ids)
        words = line.split()
        total_words += len(words)
        total_chars += len(line)
        single_words += sum(1 for word in words if len(tokenize_words(word)) == 1)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        if normalize_persian(decoded) == normalize_persian(line):
            exact_roundtrips += 1
    return {
        "label": label,
        "sentences": len(lines),
        "tokens": total_tokens,
        "words": total_words,
        "chars": total_chars,
        "fertility_tokens_per_word": total_tokens / max(total_words, 1),
        "compression_chars_per_token": total_chars / max(total_tokens, 1),
        "whole_word_rate": single_words / max(total_words, 1),
        "roundtrip_exact_rate": exact_roundtrips / max(len(lines), 1),
    }


def alphabet_coverage() -> dict:
    """How many Persian letters / digits / punctuation marks are single tokens."""

    def covered(chars: list[str]) -> tuple[int, list[str]]:
        missing = [c for c in chars if len(tokenize_words(c)) != 1]
        return len(chars) - len(missing), missing

    letters_ok, letters_missing = covered(PERSIAN_LETTERS)
    digits_ok, digits_missing = covered(PERSIAN_DIGITS)
    specials_ok, specials_missing = covered(EXTRA_SPECIALS)
    return {
        "letters_total": len(PERSIAN_LETTERS),
        "letters_single_token": letters_ok,
        "letters_missing": letters_missing,
        "persian_digits_total": len(PERSIAN_DIGITS),
        "persian_digits_single_token": digits_ok,
        "digits_missing": digits_missing,
        "punctuation_total": len(EXTRA_SPECIALS),
        "punctuation_single_token": specials_ok,
        "punctuation_missing": specials_missing,
    }


EN_REPORT = fertility_report(PARALLEL_EN, "english") if RUN_TOKENIZER_REPORT else {}
FA_REPORT = fertility_report(PARALLEL_FA, "persian") if RUN_TOKENIZER_REPORT else {}
COVERAGE = alphabet_coverage() if RUN_TOKENIZER_REPORT else {}

if RUN_TOKENIZER_REPORT:
    print("=" * 78)
    print("TOKENIZER FERTILITY (parallel EN/FA corpus, same meaning)")
    print("=" * 78)
    print(f"{'metric':32} {'english':>18} {'persian':>18}")
    for key in (
        "fertility_tokens_per_word",
        "compression_chars_per_token",
        "whole_word_rate",
        "roundtrip_exact_rate",
    ):
        print(f"{key:32} {EN_REPORT[key]:>18.4f} {FA_REPORT[key]:>18.4f}")
    print(f"{'tokens':32} {EN_REPORT['tokens']:>18} {FA_REPORT['tokens']:>18}")
    print(f"{'words':32} {EN_REPORT['words']:>18} {FA_REPORT['words']:>18}")
    print()
    print("PERSIAN ALPHABET COVERAGE")
    print(f"  letters as single token   : {COVERAGE['letters_single_token']}/{COVERAGE['letters_total']}")
    if COVERAGE["letters_missing"]:
        print(f"  fragmented letters        : {COVERAGE['letters_missing']}")
    print(
        f"  Persian digits as 1 token : "
        f"{COVERAGE['persian_digits_single_token']}/{COVERAGE['persian_digits_total']}"
    )
    if COVERAGE["digits_missing"]:
        print(f"  fragmented digits         : {COVERAGE['digits_missing']}")
    print(
        f"  punctuation as 1 token    : "
        f"{COVERAGE['punctuation_single_token']}/{COVERAGE['punctuation_total']}"
    )
    if COVERAGE["punctuation_missing"]:
        print(f"  fragmented punctuation    : {COVERAGE['punctuation_missing']}")

    print("\nWORD-LEVEL FRAGMENTATION (first three Persian sentences)")
    for sentence in PARALLEL_FA[:3]:
        for word in sentence.split()[:6]:
            pieces = tokenizer.convert_ids_to_tokens(tokenize_words(word))
            print(f"  {word:>14}  {len(pieces)}x  {pieces}")

# %% [markdown]
# ## Cell 6 — Load the model and probe the language decoder
#
# 1.6 is loaded as a native `AutoModelForImageTextToText`. The vision encoder is a NaViT-style tower;
# its tokens are *patches*, not characters. The Persian capability that matters for a LoRA therefore
# lives in the tokenizer (Cell 5) and the ERNIE decoder probed here.
#
# The probe is **teacher-forced loss**: how surprised the frozen decoder is by each gold token of a
# text. We compute it for the same parallel sentences in English and Persian and report loss per token
# and **bits per character** (comparable across tokenizations in a way perplexity is not). If Persian
# bits/char is close to English, the decoder has seen enough Persian to fine-tune; a large gap means
# the adapter starts from a model that barely models the language.
#
# The forward call is wrapped because a conditional-generation model *may* require image inputs; if so
# the probe reports `nan` rather than killing the notebook.

# %%
"""Cell 6 - model load and decoder teacher-forced loss probe."""

from transformers import AutoModelForImageTextToText

MODEL_DTYPE = (
    torch.bfloat16 if RUNTIME["supports_bf16"] else (torch.float16 if RUNTIME["cuda"] else torch.float32)
)
logger.info("loading %s on %s as %s", MODEL_ID, RUNTIME["device"], MODEL_DTYPE)

model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=MODEL_DTYPE, token=HF_TOKEN)
model = model.to(RUNTIME["device"]).eval()

if ADAPTER_PATH:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    logger.info("loaded LoRA adapter from %s", ADAPTER_PATH)

device = next(model.parameters()).device
print(f"model loaded on {device}; parameters={sum(p.numel() for p in model.parameters()):,}")


@torch.inference_mode()
def teacher_forced_loss(text: str) -> tuple[float, int]:
    """Mean per-token cross-entropy and token count for one text.

    Returns:
        ``(loss_per_token, n_tokens)``; loss is ``nan`` when the model refuses a text-only forward.
    """
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("nan"), int(input_ids.shape[1])
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=encoded["attention_mask"].to(device),
            labels=input_ids,
        )
        return float(outputs.loss.item()), int(input_ids.shape[1])
    except (TypeError, ValueError, RuntimeError) as exc:
        logger.warning("text-only forward failed (%s); decoder probe unavailable", type(exc).__name__)
        return float("nan"), int(input_ids.shape[1])


def bits_per_character(lines: list[str]) -> dict:
    """Corpus bits-per-character and mean loss per token for a list of texts."""
    total_nats = 0.0
    total_tokens = 0
    total_chars = 0
    for line in lines:
        loss, tokens = teacher_forced_loss(line)
        if tokens > 0 and np.isfinite(loss):
            total_nats += loss * tokens
            total_tokens += tokens
        total_chars += len(line)
    if total_tokens == 0:
        return {"bits_per_char": float("nan"), "loss_per_token": float("nan"), "tokens": 0}
    return {
        "bits_per_char": (total_nats / math.log(2)) / max(total_chars, 1),
        "loss_per_token": total_nats / total_tokens,
        "tokens": total_tokens,
    }


DECODER_PROBE = {"english": None, "persian": None}
if RUN_DECODER_PROBE:
    DECODER_PROBE["english"] = bits_per_character(PARALLEL_EN)
    DECODER_PROBE["persian"] = bits_per_character(PARALLEL_FA)
    print("=" * 78)
    print("LANGUAGE DECODER PROBE (teacher-forced, same parallel text)")
    print("=" * 78)
    print(f"{'metric':22} {'english':>16} {'persian':>16}")
    print(
        f"{'loss per token':22} {DECODER_PROBE['english']['loss_per_token']:>16.4f} "
        f"{DECODER_PROBE['persian']['loss_per_token']:>16.4f}"
    )
    print(
        f"{'bits per character':22} {DECODER_PROBE['english']['bits_per_char']:>16.4f} "
        f"{DECODER_PROBE['persian']['bits_per_char']:>16.4f}"
    )
    _en_bpc = DECODER_PROBE["english"]["bits_per_char"]
    _fa_bpc = DECODER_PROBE["persian"]["bits_per_char"]
    if np.isfinite(_en_bpc) and np.isfinite(_fa_bpc) and _en_bpc > 0:
        print(f"persian/english bits-per-char ratio: {_fa_bpc / _en_bpc:.3f}  (1.0 = equally modelled)")

# %% [markdown]
# ## Cell 7 — Load labelled samples
#
# Ten pages, spread with a prime stride so they come from different letters rather than ten pages of
# one. `auto` prefers the local `data/dataset/paddle_ocr_vl.jsonl` when this checkout has it, and
# otherwise streams from the HF Hub mirror of the same exporter. The local JSONL is scanned in two
# passes (count, then pick) so the stride covers the whole manifest without loading it all.

# %%
"""Cell 7 - load labelled image/text samples."""

from datasets import load_dataset


def _records_from_local(count: int, seed: int, stride: int) -> list[dict]:
    """Two-pass sample of image/text pairs from the local JSONL manifest."""
    if not LOCAL_JSONL.is_file():
        return []
    with LOCAL_JSONL.open(encoding="utf-8") as handle:
        total = sum(1 for _ in handle)
    if total == 0:
        return []
    wanted = {((offset * stride) + seed) % total for offset in range(count)}
    records: list[dict] = []
    with LOCAL_JSONL.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index not in wanted:
                continue
            row = json.loads(line)
            path = LOCAL_IMAGE_ROOT / row["image"]
            if not path.is_file():
                continue
            with PILImage.open(path) as image:
                records.append(
                    {"uid": row.get("sample_id", f"row{index}"), "text": row["text"], "image": image.convert("RGB")}
                )
    return records


def _records_from_hub(count: int, seed: int, stride: int) -> list[dict]:
    """Strided sample of image/text pairs from the HF Hub mirror."""
    dataset = load_dataset(HF_DATA_REPO, HF_DATA_CONFIG, split=HF_DATA_SPLIT, token=HF_TOKEN)
    indices = sorted({((offset * stride) + seed) % len(dataset) for offset in range(count)})
    records: list[dict] = []
    for index in indices:
        row = dataset[index]
        image = row["image"]
        if not hasattr(image, "convert"):
            payload = image["bytes"] if isinstance(image, dict) else image
            image = PILImage.open(BytesIO(payload))
        records.append(
            {"uid": str(row.get("sample_id", index)), "text": row["text"], "image": image.convert("RGB")}
        )
    return records


def load_samples(count: int, seed: int, stride: int) -> tuple[list[dict], str]:
    """Load ``count`` image/text records, preferring the local manifest over the Hub."""
    if DATA_SOURCE in {"auto", "local"} and LOCAL_JSONL.is_file():
        return _records_from_local(count, seed, stride), f"local:{LOCAL_JSONL}"
    if DATA_SOURCE == "local":
        raise FileNotFoundError(f"DATA_SOURCE='local' but {LOCAL_JSONL} does not exist")
    return _records_from_hub(count, seed, stride), f"hub:{HF_DATA_REPO}/{HF_DATA_CONFIG}"


SAMPLES, SAMPLES_SOURCE = load_samples(N_SAMPLES, SAMPLE_SEED, SAMPLE_STRIDE)
logger.info("loaded %d samples from %s", len(SAMPLES), SAMPLES_SOURCE)
print(f"loaded {len(SAMPLES)} samples from {SAMPLES_SOURCE}")
for record in SAMPLES[:2]:
    print("-" * 78)
    print(f"uid={record['uid']}  image={record['image'].size}  chars={len(record['text'])}")
    print(f"reference: {record['text'][:PRINT_PREVIEW_CHARS]!r}")

# %% [markdown]
# ## Cell 8 — Run OCR on the samples
#
# Exactly the model card's inference path: build a chat message with the PIL image, let the processor
# tokenize and expand the image tokens, then greedily decode. The image pixel budget goes through
# `images_kwargs` so it stays per-call and does not mutate global processor state.

# %%
"""Cell 8 - greedy OCR transcription of the samples."""

import time


@torch.inference_mode()
def generate_transcription(record: dict, max_new_tokens: int = EVAL_MAX_NEW_TOKENS) -> str:
    """Run the (optionally adapted) model on one record and return its transcription."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": record["image"]},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {"shortest_edge": MODEL_MIN_PIXELS, "longest_edge": EFFECTIVE_MAX_PIXELS}
        },
    ).to(device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    completion = generated[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(completion, skip_special_tokens=True).strip()


PREDICTIONS: list[dict] = []
if RUN_OCR:
    started = time.time()
    for position, record in enumerate(SAMPLES, start=1):
        begin = time.time()
        try:
            hypothesis = generate_transcription(record)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if RUNTIME["cuda"]:
                torch.cuda.empty_cache()
            logger.warning("generation failed uid=%s (%s)", record["uid"], type(exc).__name__)
            hypothesis = ""
        elapsed = time.time() - begin
        PREDICTIONS.append({"uid": record["uid"], "reference": record["text"], "hypothesis": hypothesis})
        print(f"[{position:>2}/{len(SAMPLES)}] {record['uid']}  {elapsed:6.1f}s  out_chars={len(hypothesis)}")
    print(f"total generation time: {time.time() - started:.1f}s")

# %% [markdown]
# ## Cell 9 — Error report: characters, words, structure
#
# Three families of error, all on the same pairs:
#
# * **Character error rate (CER)** — Levenshtein at character level. Reported raw *and* after Persian
#   normalisation (Arabic Yeh/Kaf and Persian digit folding), so you can separate real recognition
#   errors from orthographic variance.
# * **Word error rate (WER)** — at whitespace-token level, raw and normalised. The normalised WER is
#   the honest "how many words are wrong" number.
# * **Structure error** — the *layout* dimension a plain CER misses:
#   - `line_error` — Levenshtein over newline-separated lines. Captures reading order and where the
#     model chose to insert/merge line breaks.
#   - `line_count_error` — `|ref lines - hyp lines| / ref lines`, an over/under-segmentation measure.
#   - `structure_exact` — share of pages whose normalised line sequence matches exactly.
#   - `digit_accuracy` — how well the digits (dates, letter numbers, national IDs) survive, since
#     those are the highest-value tokens in an administrative letter.
#
# A model that reads glyphs well but scrambles reading order will show low CER and high `line_error`;
# a model that hallucinates fluent text will show high WER with plausible-looking output. That
# separation is the point of the report.

# %%
"""Cell 9 - CER / WER / structure metrics."""


def edit_distance(reference, hypothesis) -> int:
    """Levenshtein distance between two sequences of characters or words."""
    if len(reference) == 0:
        return len(hypothesis)
    previous = np.arange(len(reference) + 1, dtype=np.int32)
    for index, symbol in enumerate(hypothesis, start=1):
        current = np.empty_like(previous)
        current[0] = index
        matches = np.fromiter((symbol != item for item in reference), dtype=bool, count=len(reference))
        for position in range(1, len(reference) + 1):
            current[position] = min(
                previous[position] + 1,
                current[position - 1] + 1,
                previous[position - 1] + int(matches[position - 1]),
            )
        previous = current
    return int(previous[-1])


def corpus_cer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus character error rate (total edits / total reference characters)."""
    edits = sum(edit_distance(ref, hyp) for ref, hyp in zip(references, hypotheses))
    return edits / max(sum(len(ref) for ref in references), 1)


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus word error rate on whitespace-split tokens."""
    edits = sum(edit_distance(ref.split(), hyp.split()) for ref, hyp in zip(references, hypotheses))
    return edits / max(sum(len(ref.split()) for ref in references), 1)


def digits_only(text: str) -> str:
    """Every digit in the text, in order, normalised to Latin digits."""
    normalized = text
    for source, target in PERSIAN_NORMALIZATION_MAP.items():
        if target.isdigit():
            normalized = normalized.replace(source, target)
    return "".join(char for char in normalized if char.isdigit() and char.isascii())


def lines_of(text: str) -> list[str]:
    """Non-empty, stripped lines of the newline-preserving normalisation."""
    return [line for line in normalize_preserving_lines(text).split("\n") if line]


def line_error(reference: str, hypothesis: str) -> float:
    """Line-level edit distance over the reference line count."""
    ref_units = lines_of(reference)
    hyp_units = lines_of(hypothesis)
    if not ref_units:
        return 0.0 if not hyp_units else 1.0
    return edit_distance(ref_units, hyp_units) / len(ref_units)


def line_count_error(reference: str, hypothesis: str) -> float:
    """Absolute segmentation difference: |ref lines - hyp lines| / ref lines."""
    ref_units = lines_of(reference)
    hyp_units = lines_of(hypothesis)
    if not ref_units:
        return float("nan")
    return abs(len(ref_units) - len(hyp_units)) / len(ref_units)


def digit_accuracy(reference: str, hypothesis: str) -> float:
    """1 - CER over the digit-only projection; ``nan`` when the reference has no digits."""
    ref_digits, hyp_digits = digits_only(reference), digits_only(hypothesis)
    if not ref_digits:
        return float("nan")
    return 1.0 - edit_distance(ref_digits, hyp_digits) / len(ref_digits)


def evaluate_predictions(predictions: list[dict]) -> dict:
    """Compute the full metric set and a per-sample table."""
    raw_refs = [p["reference"] for p in predictions]
    raw_hyps = [p["hypothesis"] for p in predictions]
    norm_refs = [normalize_persian(p["reference"]) for p in predictions]
    norm_hyps = [normalize_persian(p["hypothesis"]) for p in predictions]

    per_sample = []
    for prediction, norm_ref, norm_hyp in zip(predictions, norm_refs, norm_hyps):
        per_sample.append(
            {
                "uid": prediction["uid"],
                "cer_raw": edit_distance(prediction["reference"], prediction["hypothesis"])
                / max(len(prediction["reference"]), 1),
                "cer_norm": edit_distance(norm_ref, norm_hyp) / max(len(norm_ref), 1),
                "wer_raw": edit_distance(prediction["reference"].split(), prediction["hypothesis"].split())
                / max(len(prediction["reference"].split()), 1),
                "wer_norm": edit_distance(norm_ref.split(), norm_hyp.split()) / max(len(norm_ref.split()), 1),
                "line_error": line_error(prediction["reference"], prediction["hypothesis"]),
                "line_count_error": line_count_error(prediction["reference"], prediction["hypothesis"]),
                "structure_exact": float(lines_of(prediction["reference"]) == lines_of(prediction["hypothesis"])),
                "digit_accuracy": digit_accuracy(prediction["reference"], prediction["hypothesis"]),
            }
        )

    def mean(key: str) -> float:
        values = [row[key] for row in per_sample if np.isfinite(row[key])]
        return float(np.mean(values)) if values else float("nan")

    return {
        "n_samples": len(predictions),
        "corpus_cer_raw": corpus_cer(raw_refs, raw_hyps),
        "corpus_cer_norm": corpus_cer(norm_refs, norm_hyps),
        "corpus_wer_raw": corpus_wer(raw_refs, raw_hyps),
        "corpus_wer_norm": corpus_wer(norm_refs, norm_hyps),
        "macro_line_error": mean("line_error"),
        "macro_line_count_error": mean("line_count_error"),
        "structure_exact_rate": mean("structure_exact"),
        "macro_digit_accuracy": mean("digit_accuracy"),
        "per_sample": per_sample,
    }


METRICS = evaluate_predictions(PREDICTIONS) if PREDICTIONS else {}
if METRICS:
    print("=" * 78)
    print("OCR ERROR REPORT (corpus level, n=%d)" % METRICS["n_samples"])
    print("=" * 78)
    for key in (
        "corpus_cer_raw",
        "corpus_cer_norm",
        "corpus_wer_raw",
        "corpus_wer_norm",
        "macro_line_error",
        "macro_line_count_error",
        "structure_exact_rate",
        "macro_digit_accuracy",
    ):
        print(f"  {key:24} {METRICS[key]:.4f}")
    print()
    header = f"{'uid':>20} {'CERraw':>8} {'CERnorm':>8} {'WERnorm':>8} {'line':>7} {'lcount':>7} {'exact':>6} {'digits':>7}"
    print(header)
    print("-" * len(header))
    for row in METRICS["per_sample"]:
        print(
            f"{row['uid']:>20} {row['cer_raw']:>8.3f} {row['cer_norm']:>8.3f} {row['wer_norm']:>8.3f} "
            f"{row['line_error']:>7.3f} {row['line_count_error']:>7.3f} {row['structure_exact']:>6.1f} "
            f"{row['digit_accuracy']:>7.3f}"
        )

# %% [markdown]
# ## Cell 10 — Save the report and read it
#
# The JSON on disk is the citable artefact. The interpretation block at the end is generated from the
# numbers, not written by hand, so it cannot drift from the data. **This is the paragraph to paste
# into the LoRA proposal.**

# %%
"""Cell 10 - persist the report and print the interpretation."""

report = {
    "runtime": RUNTIME,
    "model_id": MODEL_ID,
    "adapter_path": ADAPTER_PATH,
    "samples_source": SAMPLES_SOURCE,
    "image_budget_pixels": EFFECTIVE_MAX_PIXELS,
    "max_new_tokens": EVAL_MAX_NEW_TOKENS,
    "tokenizer": {"english": EN_REPORT, "persian": FA_REPORT, "coverage": COVERAGE},
    "decoder_probe": DECODER_PROBE,
    "ocr_metrics": METRICS,
    "predictions": [
        {"uid": p["uid"], "reference": p["reference"], "hypothesis": p["hypothesis"]} for p in PREDICTIONS
    ],
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {REPORT_PATH} ({REPORT_PATH.stat().st_size:,} bytes)")

print()
print("=" * 78)
print("INTERPRETATION")
print("=" * 78)
if RUN_TOKENIZER_REPORT:
    print(
        f"* Tokenizer fertility: English {EN_REPORT['fertility_tokens_per_word']:.2f} vs "
        f"Persian {FA_REPORT['fertility_tokens_per_word']:.2f} tokens/word "
        f"(ratio {FA_REPORT['fertility_tokens_per_word'] / max(EN_REPORT['fertility_tokens_per_word'], 1e-9):.2f}x)."
    )
    print(
        f"* Whole-word rate: English {EN_REPORT['whole_word_rate']:.1%} vs "
        f"Persian {FA_REPORT['whole_word_rate']:.1%}; round-trip fidelity Persian "
        f"{FA_REPORT['roundtrip_exact_rate']:.1%}."
    )
    print(
        f"* Persian alphabet present as single tokens: "
        f"{COVERAGE['letters_single_token']}/{COVERAGE['letters_total']}."
    )
if RUN_DECODER_PROBE and np.isfinite(DECODER_PROBE["persian"]["bits_per_char"]):
    print(
        f"* Decoder bits/char: English {DECODER_PROBE['english']['bits_per_char']:.3f} vs "
        f"Persian {DECODER_PROBE['persian']['bits_per_char']:.3f}."
    )
if METRICS:
    print(
        f"* End-to-end: normalized CER {METRICS['corpus_cer_norm']:.1%}, normalized WER "
        f"{METRICS['corpus_wer_norm']:.1%}, line-structure error {METRICS['macro_line_error']:.1%}, "
        f"digit accuracy {METRICS['macro_digit_accuracy']:.1%}."
    )
print(
    "\nRule of thumb: Persian fertility under ~1.8x English and bits/char within ~15% of English "
    "means the decoder has enough Persian capacity for LoRA to concentrate on layout and glyph "
    "errors. A fertility above ~2.5x or a bits/char gap above ~40% means the adapter must first "
    "relearn Persian spelling; budget more data and consider training embed_tokens/lm_head."
)
