# CLAUDE.md — Persian PaddleOCR-VL LoRA Adapter Project

This file gives Claude Code (or any coding agent working in this repository)
persistent context about the project: what it is, how it is organized, and
the standards every contribution must follow. Read this file in full before
writing or modifying any code.

## 1. Project Summary

We are building a lightweight **LoRA adapter** for **PaddleOCR-VL 1.6** that
specializes the base vision-language OCR model on **Persian (Farsi)
documents**, without retraining the full model. The adapter must stay small
and cheap to run while measurably improving recognition quality on Persian
administrative/official letters (نامه‌های اداری), scanned or typed.

The project has three deliverable phases:

1. **Dataset construction** — a mix of a filtered public dataset and a large
   synthetic dataset generated programmatically.
2. **LoRA training** — fine-tune PaddleOCR-VL 1.6 using **ERNIEKit** (or an
   equivalent PEFT-compatible library) on the constructed dataset.
3. **Packaging** — ship a small Python library + CLI + docs so the adapter
   can be loaded on top of the base model and used for inference.

## 2. Non-Goals

- We are **not** training PaddleOCR-VL from scratch.
- We are **not** targeting full multilingual coverage — Persian is the sole
  focus.
- We are **not** building a full document-management product; the output is
  a model artifact + a thin inference library.

## 3. Tech Stack

- **Base model**: PaddleOCR-VL 1.6 (vision-language OCR model).
- **Fine-tuning**: LoRA via **ERNIEKit** (PaddlePaddle's PEFT toolkit). If
  ERNIEKit proves unsuitable for PaddleOCR-VL's architecture, fall back to
  PaddleX/PaddleNLP PEFT utilities, and document the reason for the switch
  in `docs/decisions.md`.
- **Synthetic text generation**: a small local LLM (e.g. Qwen2.5-1.5B/1B
  instruct-class) used only to draft unique Persian letter bodies — never to
  generate labels/annotations, which must be produced deterministically by
  our own code.
- **Layout rendering**: Pillow / ReportLab / HTML+CSS→image (e.g.
  Playwright or wkhtmltoimage) for page composition; SVG for logos, header
  ornaments, and footer elements.
- **Tables/charts/formulas/flowcharts**: matplotlib or Pillow-drawn tables
  for tables/charts, `matplotlib`/`sympy`-rendered LaTeX for formulas,
  graphviz or a custom SVG flow-chart generator for flowcharts.
- **Signatures**: Kaggle "handwritten-signatures" dataset, composited onto
  the generated page.
- **Real dataset**: "Persian Pixel" dataset, filtered to the ~30k
  page-level (`label = page`) samples with usable ground truth.
- **Language/style**: All code, comments, docstrings, commit messages, and
  config keys are in **English only**, regardless of the fact that the
  domain data is Persian. Only literal Persian *content* (letter text,
  labels shown inside generated images) is in Persian.
- **Style**: strict **PEP 8**. Use `ruff` or `flake8` + `black` in CI to
  enforce it. Type hints are required on all public functions.

## 4. Repository Structure

```
persian-paddleocr-lora/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── lora_config.yaml            # LoRA rank, alpha, target modules, etc.
│   └── synth_config.yaml           # synthetic dataset generation knobs
├── src/
│   └── fa_ocr_lora/
│       ├── __init__.py
│       ├── dataset/
│       │   ├── real/
│       │   │   └── persian_pixel_filter.py   # filters the 392k → ~30k page set
│       │   ├── synth/
│       │   │   ├── text_generator.py         # Qwen-based letter body generator
│       │   │   ├── elements/
│       │   │   │   ├── tables.py
│       │   │   │   ├── charts.py
│       │   │   │   ├── formulas.py
│       │   │   │   └── flowcharts.py
│       │   │   ├── layout/
│       │   │   │   ├── page_composer.py      # assembles full A4 page
│       │   │   │   ├── header_footer.py      # random header/footer/logo via SVG
│       │   │   │   └── signature.py          # composites Kaggle signatures
│       │   │   ├── metadata_block.py         # شماره ثبت / تاریخ / پیوست generator
│       │   │   ├── annotation.py             # builds JSON + bbox ground truth
│       │   │   └── pipeline.py               # orchestrates one synthetic sample
│       │   └── common/
│       │       ├── schema.py                 # shared JSON annotation schema
│       │       └── io_utils.py
│       ├── training/
│       │   ├── prepare_manifest.py           # merges real + synthetic into train/val/test
│       │   ├── train_lora.py                 # ERNIEKit LoRA training entrypoint
│       │   └── eval.py                       # CER/WER + structure-accuracy eval
│       └── inference/
│           ├── loader.py                     # loads base model + LoRA adapter
│           └── predict.py                    # CLI/lib entrypoint for OCR inference
├── scripts/
│   ├── download_persian_pixel.sh
│   ├── download_kaggle_signatures.sh
│   └── run_full_pipeline.sh
├── tests/
│   ├── test_dataset_synth.py
│   ├── test_annotation_schema.py
│   └── test_inference.py
└── docs/
    ├── dataset_schema.md
    ├── decisions.md
    └── usage.md
```

## 5. Phase 1 — Dataset Construction

### 5.1 Real data: Persian Pixel

- Download/parse the Persian Pixel dataset.
- Filter to only samples where `label == "page"` (full-page ground truth),
  expected to be roughly 30k of the ~392k total images.
- Normalize its ground-truth format into our shared `schema.py` JSON format
  (see §5.3) so real and synthetic data are interchangeable during
  training.
- Discard/quarantine (do not silently drop — log and count) any sample that
  fails schema validation.

### 5.2 Synthetic data pipeline

One synthetic sample is produced by `dataset/synth/pipeline.py` running
these steps in order:

1. **Letter body generation** (`text_generator.py`)
   Use a small local instruct LLM (Qwen-1B class) to generate a unique
   Persian official-letter body (نامه اداری). Prompt it for topical
   variety (requests, notices, invitations, internal memos, contracts,
   etc.) and enforce uniqueness via a hash/near-duplicate check against
   previously generated bodies.

2. **Embedded elements** (`elements/`)
   With probability ~1/5, attach one extra structured element to the page:
   a table, a chart, a math formula, or a flow-chart — tables and charts
   should be weighted higher than formulas/flowcharts. Each element
   generator returns both a rendered image/fragment **and** its bounding
   box + a machine-readable description (e.g. table cell contents) for the
   annotation step.

3. **Page composition** (`layout/page_composer.py`)
   Lay the letter text and the optional element out on a single A4 canvas
   so it reads as one coherent, complete letter (not just floating blocks).

4. **Header/footer/logo** (`layout/header_footer.py`)
   Randomly decide whether to add a header and/or footer. If added, choose
   a random placement (top-right / top-center / top-left for the header
   logo; bottom-right / bottom-center / bottom-left for the footer), a
   random visual style for the bar itself (solid block, colored background,
   or a plain rule line), and insert a logo built/rendered as SVG.

5. **Signature** (`layout/signature.py`)
   Randomly decide whether to add a signature near the end of the letter,
   sampled from the downloaded Kaggle `handwritten-signatures` dataset, and
   composited at a plausible scale/position.

6. **Administrative metadata block** (`metadata_block.py`)
   Randomly placed top-right or top-left: registration number
   (شماره ثبت), date (تاریخ), attachment (پیوست), and similar fields.
   This block is the **highest-priority field for model accuracy** — treat
   its bounding box and text fidelity as a first-class citizen in
   annotation and in the eval metrics.

7. **Rasterization + annotation** (`annotation.py`)
   Flatten everything into one final page image, and emit a JSON record
   containing:
   - full-page transcription (all text on the page, in reading order)
   - a list of elements, each with `type`, `bbox` (`[x0, y0, x1, y1]` in
     pixel coordinates), and element-specific structured content (e.g.
     table → rows/cols/cells; formula → LaTeX source; metadata block →
     field name/value pairs)

   The exact schema is defined once in `dataset/common/schema.py` and
   documented in `docs/dataset_schema.md`; every generator must conform to
   it so downstream training code never special-cases element types.

8. **Persist** the image + JSON pair under a deterministic, config-driven
   output directory, with a manifest CSV/JSONL listing all generated
   samples for later train/val/test splitting.

### 5.3 Shared annotation schema

Define one Pydantic (or dataclass) model in `schema.py` used by *both* the
real-data adapter and every synthetic generator, so `prepare_manifest.py`
can merge them without per-source glue code.

## 6. Phase 2 — LoRA Training

- `configs/lora_config.yaml` holds rank, alpha, dropout, and the target
  module names inside PaddleOCR-VL 1.6 to adapt.
- `training/train_lora.py` wraps ERNIEKit's LoRA fine-tuning API,
  consuming the merged manifest from `prepare_manifest.py`.
- Keep the adapter small: prefer targeting attention projection layers
  only unless evaluation shows a clear need to widen the target set.
- `training/eval.py` reports CER/WER on plain text, plus a
  structure-aware metric (e.g. bbox-IoU + field-level accuracy) specific
  to the metadata block and table/chart elements, since raw CER alone
  under-values structural correctness.

## 7. Phase 3 — Packaging & Inference

- `inference/loader.py` loads the base PaddleOCR-VL 1.6 weights and applies
  the trained LoRA adapter (with a flag to fall back to base-model-only
  behavior).
- `inference/predict.py` exposes both a Python API and a CLI
  (`fa-ocr-lora predict --image path.png`) that returns the same JSON
  schema used for training annotations, so consumers get structured output
  (full text + element boxes), not just plain text.
- `README.md` documents installation, weight download, and a minimal usage
  example.

## 8. Coding Standards

- Python 3.10+, PEP 8, enforced via `ruff`/`flake8` + `black` in CI.
- Type hints required on all public functions/classes.
- Google- or NumPy-style docstrings on every public module, class, and
  function — in English.
- No bare `except:`; use specific exceptions and log context.
- Config via YAML + a typed config-loader, not hardcoded constants.
- Unit tests for: schema validation, each synthetic element generator, the
  Persian Pixel filter, and the inference loader (mocked weights OK).
- Every generated file must be reproducible from its config + seed
  (store the RNG seed per sample in the manifest).

## 9. Working Agreement for the Coding Agent

- Work incrementally: dataset pipeline first (Phase 1), then training
  (Phase 2), then packaging (Phase 3). Do not skip ahead without a working,
  tested previous phase.
- Before implementing a step that depends on an external asset (Persian
  Pixel download, Kaggle signatures, ERNIEKit installation), write a small
  script under `scripts/` and clearly state any manual step the user must
  perform (e.g. Kaggle API credentials).
- Prefer small, reviewable commits/PRs per module over one giant dump.
- When a design decision is ambiguous (e.g. exact LoRA target modules,
  image resolution), make a reasonable default, document it in
  `docs/decisions.md`, and flag it for the user to confirm rather than
  blocking.
- Keep GPU memory constraints in mind: development target is a single
  consumer GPU (assume ≤ 8GB VRAM unless told otherwise) — favor
  gradient checkpointing / small batch sizes / mixed precision.
