# Dataset schema and outputs

Every dataset producer in this repository — the Persian Pixel filter and the
synthetic renderer — emits the same annotation type, `DocumentAnnotation`,
defined once in [schema.py](../src/fa_ocr_lora/dataset/common/schema.py).
Training code reads that and nothing else, so it never special-cases the data
source or the element type (CLAUDE.md §5.3).

## Running a render

The package is not installed in this environment, so run it from the repo root
with `src/` on the path:

```bash
PYTHONPATH=src python -m fa_ocr_lora.dataset.synth.pipeline --count 10
```

After `pip install -e .` the same command is available as `fa-ocr-synth`.

The flags that matter: `--count N` / `--start N` take a slice of the text-stage
manifest, `--only plaintext_000004 …` renders named rows instead, `--resume`
skips samples whose annotation already exists, `--dpi` overrides the raster
resolution, `--seed` overrides the master seed, and `--output-dir` sends the
artifacts somewhere other than `data/synth`. Exit status is `0` when at least
one sample rendered.

### Choosing the directory names

`paths.subdirs` in [synth_config.yaml](../configs/synth_config.yaml) names each
artifact directory inside `output_dir`, and `--subdir KIND=NAME` overrides one
for a single run. A consumer that expects `image/` beside `json/` — the usual
PaddleFarmer tree — is a rename, not a post-run move:

```bash
PYTHONPATH=src python -m fa_ocr_lora.dataset.synth.pipeline \
    --count 10 \
    --output-dir data/dataset/generate \
    --subdir images=image --subdir json=json
```

```
data/dataset/generate/
├── image/synth_000000_p01.png
├── json/paddle_farmer.jsonl
├── json/paddle_ocr_vl.jsonl
├── pdf/, markdown/, annotations/    # written too; ignore if unused
└── manifest.csv
```

An empty name (the default for `json`) writes that artifact into `output_dir`
itself. The `image` field inside the JSONL follows the rename automatically,
because it is built from the configured name and stays relative to the dataset
root — the directory that holds the image directory and the JSONL.

## What a render run writes

```
data/synth/
├── images/synth_000000_p01.png     # one PNG per page, at the configured DPI
├── pdf/synth_000000.pdf            # whole document; pages match the PNGs 1:1
├── markdown/synth_000000.md        # the letter reconstructed as Markdown
├── annotations/synth_000000.json   # DocumentAnnotation (full ground truth)
├── paddle_farmer.jsonl             # image path + page text, no geometry
├── paddle_ocr_vl.jsonl             # image path + per-region text and bboxes
└── manifest.csv                    # one row per sample, with its seed
```

The two JSONL files are **projections** of the annotations, appended one line
per page as each sample completes. The annotation JSON is the source of truth;
either JSONL can be regenerated from it at any time.

## `paddle_farmer.jsonl` — the flat training file

One record per rendered page: where the image is, and every string printed on
it in RTL reading order. No coordinates.

```json
{
  "image": "images/synth_000000_p01.png",
  "text": "بسمه تعالی\nشرکت زیلبر\n…",
  "sample_id": "synth_000000",
  "page_index": 0
}
```

`image` is relative to the dataset root — the directory that holds `images/`
and this file — so the corpus can be moved between machines without rewriting
paths.

This is the file to point a full-page recognition fine-tune at. It stays valid
independently of how precise the bounding boxes are, which is why it is kept
separate from the region-level file below.

## `paddle_ocr_vl.jsonl` — the region-level file

Same one-record-per-page shape, plus geometry:

```json
{
  "image": "images/synth_000000_p01.png",
  "width": 2480,
  "height": 3508,
  "sample_id": "synth_000000",
  "page_index": 0,
  "text": "…the same page transcription…",
  "label": [
    {
      "bbox": [1893, 210, 2288, 254],
      "type": "metadata_field",
      "text": "شماره ثبت: ۱۲۳۴۵",
      "reading_order": 3
    }
  ]
}
```

## `annotations/synth_XXXXXX.json` — the full annotation

```jsonc
{
  "schema_version": "1.0",
  "sample_id": "synth_000000",
  "source_sample_id": "plaintext_000000",
  "seed": 1337,
  "dpi": 300,
  "page_size": "A4",
  "orientation": "portrait",
  "texture": "page_3.webp",        // null on a plain-white sample
  "letter_fields": {                // subject, date, numbers, plus
    "subject": "…",                 // "overflowed" and "missing_assets"
    "date": "۱۴۰۳/۰۵/۱۲"
  },
  "full_transcription": "…every page's text, joined by a blank line…",
  "pages": [{
    "page_index": 0,
    "width": 2480,
    "height": 3508,
    "image_path": "images/synth_000000_p01.png",
    "elements": [{
      "id": "meta.registration_number",
      "type": "metadata_field",
      "bbox": [1893, 210, 2288, 254],
      "text": "شماره ثبت: ۱۲۳۴۵",
      "reading_order": 3,
      "content": {"label": "شماره ثبت", "value": "۱۲۳۴۵"}
    }]
  }]
}
```

### Coordinates

`bbox` is `[x0, y0, x1, y1]` in **device pixels** of the page PNG, origin
top-left. Chromium measures in CSS pixels; `annotation.py` multiplies by
`dpi / 96` and clips to the page, so a box always indexes into its image
directly.

### Element types

`ElementType` is a `StrEnum` whose values double as the `data-ann-type`
attribute in the HTML template — renaming one means editing both.

| Region | Types |
| --- | --- |
| Header | `logo`, `header_bar`, `header_company_name`, `header_slogan`, `religious_opener` |
| Metadata | `metadata_block`, `metadata_field` |
| Frame | `sender`, `recipient`, `subject` |
| Body | `paragraph`, `heading`, `list_item`, `table`, `table_cell`, `chart`, `attachments_block` |
| Closing | `signature_block`, `signature_image`, `signatory_name`, `signatory_title`, `stamp` |
| Footer | `footer_bar`, `footer_field`, `page_number` |

Two rules keep the transcription readable:

- **`NON_TEXT_TYPES`** (`logo`, `header_bar`, `footer_bar`,
  `signature_image`, `stamp`) get a box but no text — their pixels carry no
  transcribable string.
- **Containers carry no text of their own.** Any element that has children in
  the containment forest has its `text` cleared, so a table's text is its
  cells and not the cells repeated. Every `content` payload survives.

### Reading order

`reading_order` is dense and zero-based per page. It is built by grouping
sibling boxes into lines by vertical centre, sorting each line right-to-left,
and walking the forest depth-first so a container always precedes its
children.

## Inspecting an annotation

```bash
python scripts/preview_annotation.py synth_000004
python scripts/preview_annotation.py --all --limit 20
python scripts/preview_annotation.py synth_000004 --types metadata_field table_cell
```

Overlays land in `data/synth/preview/`, one PNG per page, each box outlined in
its type's colour and captioned `reading_order:type`. A schema-valid
annotation can still be visually wrong, and this is the check that catches it.
