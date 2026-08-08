You are working inside a repository that will contain a Persian-specialized
LoRA adapter for **PaddleOCR-VL 1.6**. A `CLAUDE.md` file already exists (or
should be created if missing) at the repository root — read it fully before
doing anything else; it is the source of truth for architecture, coding
standards, and phase ordering. If `CLAUDE.md` is not yet present in this
repo, create it first using the specification below, then proceed.

## Your mission

Build, in order, the three phases described in `CLAUDE.md`:

1. **Dataset construction**
   - A loader/filter for the "Persian Pixel" dataset that keeps only the
     ~30k `label == "page"` full-page samples out of the ~392k total, and
     converts them into our shared JSON annotation schema.
   - A synthetic data generation pipeline that, for each sample:
     1. Generates a unique Persian official-letter (نامه اداری) body using
        a small local instruct LLM (Qwen-1B class).
     2. With ~1/5 probability, adds one structured element (table, chart,
        formula, or flowchart — weight tables/charts higher), each
        producing both a rendered fragment and machine-readable content.
     3. Composes the letter text and optional element onto a single A4
        page as a coherent, complete letter.
     4. Randomly adds a header and/or footer with an SVG-rendered logo at
        a randomly chosen corner/center position, and a randomly styled
        bar (solid block / colored background / plain rule line).
     5. Randomly composites a handwritten signature from the Kaggle
        `handwritten-signatures` dataset near the end of the letter.
     6. Adds an administrative metadata block (شماره ثبت، تاریخ، پیوست,
        etc.) at a random top-left/top-right position — this block's
        accuracy is the highest priority in the whole pipeline, so give it
        first-class treatment in both rendering and annotation.
     7. Rasterizes the full page to a single image and writes a JSON
        record with the full-page transcription plus a list of elements,
        each with its `bbox` and structured content, per the schema in
        `CLAUDE.md` §5.3.
     8. Persists image + JSON + manifest entry to a config-driven output
        directory.

2. **LoRA training**
   - Merge the real + synthetic manifests into train/val/test splits.
   - Fine-tune PaddleOCR-VL 1.6 with LoRA using ERNIEKit (or a documented
     PEFT fallback if ERNIEKit doesn't fit the model architecture),
     configured via `configs/lora_config.yaml`.
   - Implement an evaluation script reporting CER/WER plus a
     structure-aware metric (bbox-IoU + field-level accuracy), with
     special attention to the metadata block and table/chart fidelity.

3. **Packaging**
   - A small inference library that loads the base model + LoRA adapter
     and exposes a Python API and CLI returning the same structured JSON
     schema used in training.
   - A README with install instructions, weight-download steps, and a
     minimal usage example.

## How to work

- Follow the repository structure, coding standards (PEP 8, English-only
  code/comments/docstrings, type hints, no bare excepts), and phase
  ordering exactly as specified in `CLAUDE.md`.
- Do not proceed to Phase 2 until Phase 1's dataset pipeline is
  implemented, unit-tested, and can generate a small sample batch
  end-to-end. Do not proceed to Phase 3 until Phase 2 trains and evaluates
  on at least a small run.
- Before touching anything that needs an external asset (Persian Pixel
  download, Kaggle signatures download, ERNIEKit install), write the
  download/setup script under `scripts/` and clearly tell me any manual
  step I need to do myself (API keys, license acceptance, etc.) — don't
  silently assume credentials exist.
- When something in the spec is ambiguous or you must pick a default
  (LoRA target modules, canvas resolution, font choices for Persian
  rendering, etc.), pick a sensible default, write the reasoning to
  `docs/decisions.md`, and flag it to me in your summary instead of
  blocking on it.
- Keep GPU memory constraints in mind — assume a single consumer GPU with
  limited VRAM unless I tell you otherwise; prefer mixed precision,
  gradient checkpointing, and modest batch sizes.
- Work in small, reviewable increments (one module or one clear unit of
  functionality at a time) rather than one giant dump, and run/describe
  the tests you add for each piece.
- Ask me directly if you hit a genuine blocker (e.g. a dataset link is
  dead, ERNIEKit is incompatible with PaddleOCR-VL's LoRA targets) rather
  than guessing silently on something load-bearing.

Start by confirming (or creating) `CLAUDE.md`, then begin with the Persian
Pixel filter and the synthetic pipeline's text-generation module, since
everything else in Phase 1 depends on having a working annotation schema
and a first working generator to test it against.
