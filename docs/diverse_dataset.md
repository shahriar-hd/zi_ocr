# Persian Diverse OCR dataset

A second synthetic corpus for the PaddleOCR-VL Persian adapter, built because
the first one was too narrow.

The administrative-letter dataset taught the model to read letters well (CER
about 0.09) and barely moved a different distribution (persian_pixel stayed
near 0.40). That is the whole motivation here: **a model trained on one
generator learns that generator, not Persian.** So in this corpus *diversity is
the product, not volume* — a 60k-sample set with real variation in vocabulary,
genre, layout, font, degradation and generator beats a 300k-sample set that is
one template with different words.

---

## Contents

- [How an image is made](#how-an-image-is-made) ← the method, end to end
- [The ten visual categories](#the-ten-visual-categories)
- [Fonts](#fonts)
- [Degradation](#degradation)
- [Where the labels come from](#where-the-labels-come-from)
- [Running it](#running-it)
- [Splits and the hard tests](#splits-and-the-hard-tests)
- [Licensing](#licensing)
- [What this corpus cannot tell you](#what-this-corpus-cannot-tell-you)

---

## How an image is made

Nine stages. Stages 00–03 run on the laptop because they hold the API keys;
04–08 run on a marimo/molab runtime because they need Chromium and bandwidth.
The handoff is the text, mirrored to Google Drive.

```
00 lexicon    harvest Persian headwords, tag era/domain, band by frequency
01 plan       draw category × genre × register × persona × length × words
02 generate   several NVIDIA-hosted LLMs write the Persian text
03 text QC    normalise, filter, dedupe                → text_render + text_norm
--------------------------------------------- Drive handoff ------------------
04 layout     pick format, template, fonts, palette, degradation plan
05 render     Chromium draws it; geometry is measured back from the browser
06 image QC   reject anything whose label does not describe its pixels
07 package    group-aware splits, Parquet shards per category
08 upload     parallel, verified, resumable push to the Hub
```

### 00 — Lexicon: the words the model has never seen

The point of this stage is the **long tail**. Left alone, an instruction-tuned
model writes the high-frequency core of Persian — a few thousand words that
cover most of its output. Spending a prompt slot on those buys no vocabulary
the model has not already met. What it never reaches for unprompted is archaic
and classical words, dialect forms, domain terminology and the Academy's modern
coinages.

Two sources, both redistributable:

| source | licence | contributes |
|---|---|---|
| [Lilak](https://github.com/b00f/lilak) Persian spell-check dictionary | Apache-2.0 | ~94k headwords with part of speech |
| [Persian Wiktionary](https://fa.wiktionary.org) dump | CC BY-SA 4.0 | era and domain labels — the only source that has them |

Merged: **128,826 words**, of which 124,227 are injectable. Frequency bands
come from a streamed Persian Wikipedia chunk that stops as soon as the counts
converge.

The lever is `exclude_top_frequency_rank`: words inside rank 5,000 stay in the
parquet (their frequency is what proves they are common) but are **never
injected**. Result — 79% of injected word-slots land in `rare` or `very_rare`.

Era labels, from Wiktionary's own categories:

| era | injectable | examples |
|---|---|---|
| `neologism` | 9,226 | خاموشگر, گرمامرگی, عصب‌شناسی, یخ‌قهوه — Academy coinages |
| `technical` | 3,139 | domain terminology across medicine, botany, law, physics |
| `archaic` | 2,216 | شگن, آژن, اخص |
| `dialectal` | 1,236 | کرنو, کلکشک, دلنگوستی |
| `literary` | 916 | آوردگه, رادمردی |
| `unlabelled` | 107,494 | Lilak's bulk: real words, no era metadata |

> Persian Wikipedia **article titles** were tried and dropped. Of 324,015
> single-token titles, ~205,000 were words no other source knew — and they
> turned out to be transliterated place and person names (Winterthur,
> Hohenwart, Sergius), not vocabulary. They flooded the `technical` era with
> 68% of the lexicon.

> Dehkhoda and Moein are **deliberately absent**: both are copyrighted, their
> online terms forbid bulk scraping, and their definitions are not
> redistributable.

### 01 — Plan: designing the variation

One prompt template produces one kind of text. Instead, each sample is drawn
from independent axes whose cross product is far larger than the corpus:

- **category** (10) — decides the eventual visual format
- **genre** (~100, bound to category) — a legal clause does not belong in a chat screenshot
- **register** (10) — formal, literary, journalistic, colloquial Tehrani, slangy, archaic, bureaucratic, childlike, academic, promotional
- **persona** (20) — a retired teacher, a nurse, a teenage gamer, a village elder…
- **length target** (6) — constrained by category
- **output form** (8) — prose, verse lines, numbered list, dialogue turns, table rows, key-value
- **content features** (11) — numbers, Latin terms, quotations, parentheses, bullets, tables, headings, footnotes, diacritics, ZWNJ-heavy compounds, emoji
- **digit script** (4) — Persian, Latin, Arabic-Indic, mixed
- **generator profile** (5)
- **lexicon words** — 2–6, stratified by era and band

Two mechanisms keep it honest:

**Capacity clamping.** An era weight is a request; its pool size is a limit.
`literary` has 916 injectable words, so with a 30-use ceiling it can supply at
most 27,480 word-slots however much weight the config asks for. Weights are
clamped to capacity and the surplus redistributed, with every adjustment
logged — so raising `total_samples` from 150 to 150,000 cannot silently break
the stratification.

**Length is drawn per category.** Drawing it independently produced a 300–600
word "single line" and a six-word A4 page. A category's length range is part of
what the category means.

### 02 — Generation: which models, and why

Models were **measured, not chosen from the catalogue**. Of 63 chat models the
endpoint lists, only 13 were callable on this account. Scoring Persian-letter
ratio, rare-word fidelity and latency on a fixed prompt set
(`scripts/evaluate_diverse_models.py`), then validating against a real
150-sample run scored by stage 03:

| profile | model | temp | stage-03 rejection |
|---|---|---|---|
| `gemma_warm` / `gemma_cool` | google/gemma-4-31b-it | 0.75 / 0.45 | **0.0%** (36 samples) |
| `nemotron_ultra_warm` | nvidia/nemotron-3-ultra-550b-a55b | 0.50 | 16% at 0.40, 31% at 0.65 |
| `nemotron_ultra_cool` | nvidia/nemotron-3-ultra-550b-a55b | 0.35 | — |
| `kimi_warm` | moonshotai/kimi-k3 | 0.70 | **0.0%** (11 samples) |

`nemotron-3-super-120b` was in the first draft and is **removed**: at a 73.6%
rejection rate it burned most of its API budget on samples that were discarded,
leaking Han, Cyrillic, Kana, Devanagari, Hangul, Hebrew and Thai into Persian
prose. `nemotron-3.5-lightning` writes 76.6% Persian and overshoots an 80-word
target tenfold.

Transport: one worker thread per API key (`NVIDIA_API_KEY` accepts a
comma-separated list), a lock-guarded per-key pacer, exponential backoff, long
cooldown on quota exhaustion, and an on-disk response cache keyed by prompt
hash. Measured at ~22 samples/min on three keys.

### 03 — Text QC: two fields, never one

Each sample carries **two** strings:

- **`text_render`** — exactly the characters handed to the renderer. It is what
  the pixels show. Nothing downstream may "clean" it.
- **`text_norm`** — the evaluation form: Arabic letter variants folded to
  Persian, digits folded to one script, diacritics stripped, whitespace
  collapsed. CER is computed against this, so a model is not punished for a
  distinction no reader makes.

The decisive filter rejects **foreign script as a kind, not a quantity**. Handed
rare Persian words at high temperature, models fall back on tokens from other
alphabets — 城堡, 人間, 언어, र्फ and Cyrillic наблюї all appeared inside Persian
prose. A ratio threshold does not catch this: one such sample measured 80.6%
Persian letters and would have passed a 0.80 floor while being unusable.

Other filters: refusals, preambles, unrequested Latin, Persian ratio, PII-shaped
digit strings, length deviation, repetition (identical lines and n-gram ratio),
and MinHash near-duplicate detection — stricter within a genre than across the
corpus, because that is where a model repeats itself.

The stage **exits non-zero above a 20% rejection rate** rather than continuing:
at that rate a prompt or a model is broken, and scaling up only buys samples
that will be discarded.

### 04 — Layout: choosing the look

Per sample, from the seeded RNG: a template, a DPI, a palette, fonts, and a
degradation plan. Written to `layouts.jsonl` *with the fonts embedded*, which
makes stage 05 a pure function of one JSONL line — no font directory, no
lexicon, no config needed on the render host.

**Palettes are sampled, not fixed.** Ink is a hue-and-lightness perturbation of
near-black, so the corpus contains the warm greys, blue-blacks and faded browns
real printing produces — then darkened until it clears a WCAG contrast floor, so
widening the distribution never yields a page a human cannot read. `muted`
(captions, timestamps, footers) clears a floor of its own.

**Font size is solved, not guessed.** `_fit_font_px` finds the largest body size
whose text fits the canvas, and lets the canvas grow when even the minimum will
not. A tall page is honest; a clipped label is not.

### 05 — Render: Chromium, and why

**Chromium through Playwright does all the shaping.** Persian needs bidi
resolution, contextual joining, ZWNJ handling and mixed-script runs, and a
browser is the only engine in reach that gets all four right. Pillow here is not
built with libraqm, so it draws Persian as *disconnected letters* — an engine
that is silently wrong is worse than a heavier one that is right.
`verify_shaping()` asserts this at startup rather than trusting it.

Launch flags disable hinting and LCD subpixel antialiasing so a glyph renders
identically on a laptop and on a molab runtime; otherwise the renderer's
hardware becomes a hidden variable in the corpus.

One browser per worker thread, a fresh context per sample. Measured at ~146
samples/min on three workers.

---

## The ten visual categories

41 templates, each a *function* of (text, palette, fonts, RNG) rather than a
string template, so it can make structural decisions — two columns or one, zebra
striping, a sidebar, a drop cap.

| category | share | templates |
|---|---|---|
| `page` | 25% | prose, two-column, headed article, textbook (sidebar + footnote), report, newspaper (masthead, 2–3 columns) |
| `paragraph` | 15% | plain, boxed, caption under a figure, pull quote |
| `line` | 15% | title, label, UI string/button, subtitle on a dark band |
| `poetry` | 8% | two-hemistich classical, centred masnavi, free verse, ornamental double frame |
| `web` | 10% | article, blog, forum thread, product page, FAQ |
| `chat` | 8% | light messenger, dark messenger, SMS, group chat |
| `banner` | 7% | poster, square social card, wide shop sign, flyer |
| `story` | 4% | chapter opening, drop cap, plain with page number |
| `table_form` | 5% | data grid, invoice, form fields, timetable |
| `mixed` | 3% | technical prose with Latin terms, code-switched settings panel, product label |

Pages are also cut into **line and paragraph crops** (and a 2% fraction of word
crops), which inherit the parent's `group_id` so a crop can never land in test
while its own page is in train.

No real brand, app or logo appears anywhere. All messenger, news-site and shop
styling is invented.

---

## Fonts

**124 faces across 35 families, every one SIL OFL 1.1**, fetched by
`scripts/download_diverse_fonts.py` with each family's licence text saved
alongside it.

| class | families |
|---|---|
| sans (13) | Vazirmatn, Sahel, Shabnam, Samim, Gandom, Tanha, Estedad, Noto Sans Arabic, IBM Plex Sans Arabic, Cairo, Almarai, Mada, Baloo Bhaijaan 2 |
| naskh (4) | Noto Naskh Arabic, Scheherazade New, Lateef, Harmattan |
| serif (4) | Amiri, Markazi Text, El Messiri, Parastoo |
| nastaliq (2) | Gulzar, Noto Nastaliq Urdu |
| display (12) | Noto Kufi Arabic, Reem Kufi, Lalezar, Rakkas, Jomhuria, Katibeh, Mirza, Aref Ruqaa, Changa, Vibes, Kufam, Qahiri |

This replaces the proprietary B-series and IRANSansX the letters pipeline
rendered with, which could not be redistributed.

**Every face is gated on glyph coverage, per sample, against the actual text.**
Chromium substitutes a missing glyph silently, so a display face lacking `گ`
would draw a tofu box where the label claims a letter. A sample no font can
cover is rejected rather than rendered. The gate rejected the "without-Latin"
font variants and two Arabic-only families outright.

Faces are embedded in each page as base64 `data:` URIs, so a render is
reproducible on a host with none of them installed and does not depend on
fontconfig.

---

## Degradation

A clean raster is a screenshot, and a model trained only on screenshots fails on
the first scanned page it meets. Everything below the ink is composited
**multiplicatively**, in the order light meets the page — that is the physics
(ink and paper absorb light, they do not add it):

```
paper tint → yellowing → wash → watermark → verso show-through → INK
           → geometry → light → noise → resample → encode
```

### Verso show-through

Hold a printed sheet to the light and the other side shows through: mirrored,
blurred, lower in contrast, tinted by the paper rather than the ink.

The letters pipeline mirrored *the same page*, which teaches nothing — a model
can learn to read the ghost as a reflection of what it is already reading. Here
the ghost is **a different sample's rendered text**, so the model must learn
that faint mirrored strokes are not content.

Per sample: opacity 5–18%, Gaussian blur 1.2–4.5 px, hue rotation ±28° toward
the paper colour, desaturation 0.35–0.85, scale jitter 0.97–1.03,
misregistration ±2%, rotation ±1.5°, mirrored 85% of the time. The ghost is
never annotated; its source id is recorded in `aug.show_through.source_id`.

### Paper behind the text

- **Tint** — 11 stocks, weighted: bright white, warm white, cream, aged cream, cool grey, faint blue, faint pink, ledger green, recycled buff…
- **Yellowing** — edge-weighted, as paper oxidises from the outside in
- **Wash** — a coloured panel behind the text: full flood, a band, a rounded panel, or a soft radial bloom, 4–20% opacity. This is what puts low-contrast and high-contrast text *on the same page*.
- **Watermark** — a faint rotated word (رونوشت, محرمانه, پیش‌نویس…) at 3–12%

### Everything else

| group | effects |
|---|---|
| ink | bleed, uneven toner |
| geometry | rotation/skew ±3.5°, perspective warp — **one combined resample**, because each resample softens the ink |
| light | vignette, edge shadow, glare, low light — **bounded**: the field is lifted so its mean stays above a floor, because three moderate draws composed into a uniformly grey page |
| noise | Gaussian sensor noise, specks, stains, folds, highlighter marks |
| capture | JPEG at 55–95, downscale-and-upscale, DPI from 96 to 400 |

**25% of samples are rendered clean**, so the model also sees pristine text.

**Small type is protected.** Blur, ink bleed and downscaling are scaled by an
"ink budget" derived from the measured line height: a 1 px stroke at 11 px type
does not survive what a 3 px stroke shrugs off. Without this, 64 samples came
back from QC as illegible pale text — which is a wrong label, not hard training
data.

---

## Where the labels come from

**Annotations are measured, never inferred.** `harvest.js` runs in the page
after `document.fonts.ready`, walks each annotated subtree, and reads back a
rectangle *per character*, bucketing them into the lines the browser actually
chose. Guessing boxes from font metrics — the usual shortcut — goes wrong on
exactly the cases this corpus is full of: justified Persian, ZWNJ-joined
compounds, bidi runs, and nastaliq where one word's ink overlaps the next.

**Reading order comes from the DOM walk, not from coordinates.** For a
right-to-left multi-column page no coordinate rule recovers the order correctly.
The templates emit content in reading order by construction and CSS columns flow
in DOM order, so the DOM walk *is* the reading order — and it is stable, which a
geometric sort is not.

**Geometry moves the boxes and the pixels through the same matrix.** The warp is
one homography; `Geometry.map_point` applies it forward to the polygons while
Pillow applies its inverse to the image. They cannot drift apart.

Each sample carries `lines` (text, polygon, reading order, direction, optional
word boxes) and `blocks` (20 layout types: heading, paragraph, caption,
table_cell, chat_message, verse, banner_text, field_label…).

### Image QC

Every check answers one question: *would this sample teach the model something
false?* A blurry page is fine. A page whose label claims a line the pixels do
not show is not.

| check | catches |
|---|---|
| `overflow` | content escaped the canvas, so the label describes clipped text |
| `annotation_text` | the lines, concatenated, disagree with `text_render` — a harvest that dropped, duplicated or reordered a line |
| `legibility` | shortest line below an x-height floor |
| `contrast` | stroke-vs-background inside the line boxes, below a floor |
| `blank` | near-uniform image |
| `duplicate` | identical text *and* a colliding perceptual hash |
| `digit_script` | the text carries a script the manifest does not claim |

Contact sheets are written per category for review by eye, because no automatic
check catches "this is wrong in a way nobody thought to test for".

> QC drove rejection from **65% to 4.2%** on the pilot — and every fix was to
> the cause, not the threshold. It found: boxes clamped on one edge only,
> `overflowed` wrongly propagating to crops, perceptual dedup that could not
> tell two Persian lines apart, chat dropping the speaker's colon, tables whose
> pipes never reached the pixels, pages that guessed a font size, a contrast
> metric that measured ink *coverage* rather than legibility, and degradation
> that destroyed small type.

---

## Running it

### Laptop — text (stages 00–03)

```bash
conda activate zilber_rag

# 00  harvest the lexicon (downloads ~400 MB, cached)
python -m fa_ocr_lora.diverse.lexicon.build

# 01  draw the plans
./scripts/run_diverse_text.sh plan 100000

# 02  generate, detached and restart-safe (days)
./scripts/run_diverse_text.sh generate --detach
./scripts/run_diverse_text.sh status        # progress, throughput, gateway blocks

# 03  filter and dedupe
./scripts/run_diverse_text.sh qc

#     mirror the irreproducible part to Drive
python -m fa_ocr_lora.diverse.backup
```

Everything is resumable by default — there is no `--continue` flag because
continuing is the only behaviour. Each sample is written to its own file the
moment it arrives, and a restart skips what is on disk.

### molab — images (stages 04–08)

Open `notebooks/molab_diverse_render.py` in marimo, set the panel secrets
(`HF_TOKEN`, `DRIVE_TOKEN_B64`, `GH_TOKEN`), and run the cells. Or on any
machine with Chromium:

```bash
python scripts/download_diverse_fonts.py
python -m fa_ocr_lora.diverse.layouts.build
python -m fa_ocr_lora.diverse.render.run --workers 8
python -m fa_ocr_lora.diverse.qc.run
python -m fa_ocr_lora.diverse.packaging.run
python -m fa_ocr_lora.diverse.upload.run --dry-run --repo <user>/<dataset>
python -m fa_ocr_lora.diverse.upload.run --repo <user>/<dataset>
```

Every knob lives in `configs/diverse_config.yaml`. Loading is strict: an unknown
key is a startup error, not a warning.

---

## Splits and the hard tests

A `group_id` ties a page to every crop cut from it, and the split is a hash of
that key, so a group never straddles train and test. Without it a model would
meet a line in training and its own parent page in test.

| split | what it measures |
|---|---|
| `train` / `validation` / `test` | the usual, interpolation |
| `test_unseen_fonts` | font families held out of training entirely |
| `test_unseen_templates` | layouts held out of training entirely |
| `test_unseen_generator` | text from an LLM profile held out of training entirely |

Each hard test is **capped**, and groups past the ceiling are *dropped* rather
than moved to train — their font, template or generator is meant to be unseen,
so training on them would silently void the test they exist for.

Output is sharded Parquet, one directory per category, each a loadable Hub
config:

```python
from datasets import load_dataset
chat = load_dataset("<user>/<dataset>", "chat")
```

---

## Licensing

The dataset licence is bounded by its weakest-licensed ingredient.

| ingredient | licence | implication |
|---|---|---|
| Persian Wiktionary (headwords, era/domain labels) | CC BY-SA 4.0 | **share-alike** |
| Lilak dictionary | Apache-2.0 | permissive |
| Persian Wikipedia (frequency corpus) | CC BY-SA 4.0 | **share-alike** |
| 35 font families | SIL OFL 1.1 | rendered output is unencumbered |
| LLM-generated text | — | model terms apply to use, not redistribution |

Because Wiktionary and Wikipedia are share-alike, **the corpus cannot be
licensed more permissively than CC BY-SA 4.0**. That is what
`upload.license` is set to. Using Lilak alone would permit Apache-2.0, at the
cost of the era and domain labels — which are most of what makes stage 00 worth
running.

---

## What this corpus cannot tell you

Stated plainly, because a dataset card that only lists strengths is not useful.

- **LLM text carries its generators' style.** Five profiles across three model
  families reduce this; they do not remove it.
- **Rendering diversity is not real-world diversity.** Real scans, camera shake,
  paper quirks and printer defects are approximated, not reproduced.
- **Rare words appear because they were requested.** Their *meaning and context*
  are only as good as the model's knowledge. That is fine for an OCR label and
  not fine for teaching lexical semantics.
- **There is no handwriting.** Convincing synthetic Persian handwriting needs a
  different method; a separate track with licensed handwriting fonts or real
  data would be honest, and would still look synthetic.
- **No synthetic corpus certifies real-world performance.** Evaluate on the
  random test, the unseen-font test and the unseen-template test — and then on a
  small **real** benchmark, even 100–200 hand-labelled scanned or photographed
  Persian items. Nothing here substitutes for that.
