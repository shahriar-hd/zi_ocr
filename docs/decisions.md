# Decisions

Ambiguous choices made while building the pipeline, with the reasoning, so a
later change is a decision rather than a discovery. CLAUDE.md §9 asks for this
file; each entry is something a reviewer might otherwise have to reverse
engineer.

## Rendering

**Chromium via Playwright, not Pillow or ReportLab.** Persian needs correct
shaping, bidi and justification; a browser does all three for free. More
importantly it is the only option that reports the exact geometry of the
elements it drew, via `getBoundingClientRect()` on nodes tagged with
`data-ann-type`. One HTML document produces the PNGs (`screenshot`), the PDF
(`page.pdf`) and the boxes, so the three artifacts cannot drift apart.

**Pagination is ~60 lines of injected JS, not a library.** `paginate.js`
appends blocks to a fixed-size `.sheet` until it overflows, then starts a new
one. That yields per-page PNGs, a PDF whose pages match them 1:1, and boxes
measured against the sheet each element actually landed on.

**Charts are inline SVG we generate, not matplotlib PNGs.** Inline SVG text is
shaped by Chromium like any other text, so Persian axis labels come out
correct with no `arabic_reshaper` plumbing, stay crisp at any DPI, and expose
their own bounding boxes.

**Fonts are embedded as `@font-face` data URIs rather than resolved by family
name.** This shell runs with `XDG_DATA_HOME` pointing into a VS Code snap
sandbox, so fontconfig expands its `<dir prefix="xdg">fonts</dir>` rule to a
directory that does not exist and never indexes the real one — Chromium then
silently falls back for every Persian family. Embedding the bytes sidesteps
host font state, which a reproducible corpus needs anyway: the same seed must
render the same page on a training box that has none of these faces installed.

## Page geometry

**300 DPI default.** It is the scan resolution PaddleOCR-VL is typically fed,
and it keeps A4 at 2480×3508 — large enough that small metadata text survives
downsampling. Overridable with `--dpi`.

**Page-size weights** (relative, not percentages), from
[synth_config.yaml](../configs/synth_config.yaml):

| Size | Orientation | Weight |
| --- | --- | --- |
| A4 | portrait | 60 |
| US Letter | portrait | 15 |
| A5 | portrait | 10 |
| A4 | landscape | 8 |
| A3 | portrait | 7 |

A4 portrait dominates because it is what Persian administrative
correspondence actually uses; the rest exist so the adapter does not overfit
one aspect ratio.

**`max_pages: 6`.** A safety valve, not a layout rule: a body long enough to
exceed six pages is truncated rather than rendering an unbounded PDF. Bodies
in the current corpus top out near 5.6k characters, which fits comfortably.

## Scanner simulation

The renderer produces a mathematically perfect page: exact geometry, even
illumination, hard glyph edges, pure tones. No document an OCR model meets in
production looks like that, so [`render/scan.py`](../src/fa_ocr_lora/dataset/synth/render/scan.py)
reintroduces the physics. 85% of samples are scanned; the rest ship clean, so
the corpus still contains born-digital pages.

**Nothing on the page is pure white, and the page is captured on a transparent
ground.** `_screenshot_sheets` passes `omit_background=True`, so the capture is
RGBA where a light area of the letter is an *absence of ink* rather than a coat
of white paint. The theme paints its surfaces as `rgba()` with a low alpha for
the same reason. That is what lets paper be composited *underneath* the page
afterwards and show through the light areas; a sheet painted `#FFFFFF` would
hide whatever was behind it, and no amount of post-processing recovers the
grain. The consequence for contrast is that a WCAG check must run on the
*composite* (`theme.composite(surface, ink, alpha)`), never on the nominal hex,
which is what `_ensure_contrast_over` does.

**Texture is a post-render composite, not a CSS background.** It began as
`background-image` on `.sheet`, which works but locks the paper out of every
later stage: CSS can lay an image behind text, but it cannot expose that image
to the uneven light, the sensor noise, or the greyscale and autocontrast passes
a scanner's own software applies. Moving it into Pillow costs one thing — the
**PDF keeps the clean vector page** and never sees the grain, because
`page.pdf()` renders from the DOM. That is the right trade: the PNG is what
trains the model, and a vector PDF is more useful as a reference artifact than
a photographed one.

**Paper is flat-fielded before it is multiplied on.** The prepared scans are
raw photographs with arbitrary exposure — several are mid-grey overall — and
multiplying one onto a paper tone directly turned the page to slate (page mean
~165). What the page wants from them is their *variation*, not their exposure,
so each channel is rescaled by a 256-entry LUT so its own mean lands at level
247, floored at 186. `strength` therefore scales grain **depth**, not page
brightness. Doing it with a LUT rather than array arithmetic keeps the operation
at 8 bits: a float field the size of an A3 page at 300 DPI is several hundred
megabytes, and 20k of those in a batch loop is how a generator gets itself
killed.

**Scans upscaled more than 1.6× are blurred.** Several assets are small and
JPEG-compressed; at 6× their 8×8 blocks become inch-wide rectangles, which
autocontrast then amplifies into a visible grid — again, exactly the regularity
an OCR model would learn. A blur proportional to the upscale removes the blocks
and keeps the tone variation.

**The light field is mean-normalised, not just amplitude-limited.** Uneven
illumination, vignette and the local shadow are all *subtractive*, so their sum
dimmed the whole page as well as making it uneven (page mean 148 even after the
paper fix). Every lighting knob was doubling as a brightness knob. The field is
now divided by its own mean and re-clipped to `[0.62, 1.0]`; the half that
lands above 1.0 clips back, since a multiply cannot brighten, but the page keeps
its paper tone and the falloff keeps its shape. The `scan.lighting` ranges are
tuned as *depth*, not dimming — note that a `contrast` below 1.0 pulls a light
page toward 128 and so darkens paper, which is why that range sits at
`0.92–1.05` rather than reaching lower.

**All geometry is one resample.** Rotation, perspective, page bow and barrel
distortion are composed into a single mesh transform. Each resample softens the
image, and that softening is not part of the effect — applying them separately
blurred the page two or three times over. Rotation and perspective are both
expressed as displacements of the four page corners and solved together as one
homography; the bow and barrel terms are radial and are folded into the same
per-vertex sampling.

**Only the ink layer is warped; the paper is not.** A sheet physically carries
its grain with it, but grain is stochastic — nobody can see that it stayed put —
and warping the composite would rotate the paper's own edges into the frame,
leaving transparent wedges at the corners. Warping the ink and compositing it
over full-bleed paper means the corners are always paper.

**The bounding boxes go through the same map the pixels did.**
`SheetGeometry` exposes both directions: `source_of` (what Pillow's `MESH`
transform needs, which is destination→source) and `destination_of`, its inverse,
which is what ground truth needs. The inverse has no closed form once bow and
barrel are involved, so it is solved by fixed-point iteration — the map is a
near-identity by construction, so 12 steps converge to ~0.001 px.
`annotation.remap_boxes` takes the geometry through a structural `BoxMap`
Protocol, so `annotation.py` stays free of numpy and Pillow.

**A mapped box is the hull of its four mapped corners, and boxes pushed off the
page are dropped, not clamped.** A rotated rectangle is not a rectangle and the
schema stores axis-aligned boxes, so the hull is the tightest box that still
contains all the ink it contained before. **Reading order is deliberately not
recomputed** — it was derived on the exactly-horizontal clean page, a sub-degree
tilt cannot reorder a letter, and re-deriving it from tilted boxes could.

**The capture is squared up to the size the boxes were measured against.**
Chromium rounds an element screenshot outwards, so a sheet whose device width
works out to 1240.16 px is captured 1241 px wide while every box was scaled
against 1240. `_fit()` trims or transparently pads at the far edge, where the
coordinate origin is untouched, so the raster and its ground truth share
dimensions by construction rather than by a tolerance in a test.

**Bilevel output uses Floyd–Steinberg, not a hard threshold.** A fax-grade
"B&W" pass is in the corpus because cheap scanners produce it, but a hard
threshold at these gammas eats the lightest strokes outright. Greyscale and
1-bit results are converted back to three channels, so every artifact in the
corpus has the same shape — the loss is in the pixels, not the file format.

**Paper is drawn per document, everything else per sheet.** A letter is printed
on one sheet of paper; each page crosses the platen separately and gets its own
geometry, light and noise.

**Salt-and-pepper noise was removed and replaced by soft dust specks.** The
first version of the noise pass flipped a fraction of all pixels to pure black
or pure white (`scan.noise.salt_pepper`, up to 0.06% — about 5 000 dots on an A4
page at 300 DPI). Two things are wrong with that, and the user flagged both. It
is far too much: a scanned page carries dust you can count, not thousands of
dots. And a lone pixel at level 0 surrounded by clean paper is not a defect any
optical path can produce — every real speck is a blot with a falloff, because
dust has area and the lens has a point-spread function. A model trained on
isolated extremes learns to discount a texture it will never meet again.

The replacement (`_add_specks`) stamps a Gaussian profile per speck, so the blur
*is* the speck rather than a pass applied over one: `speck_density` counts
specks **per megapixel** — up to ~35 on an A4 page at 300 DPI, so the setting
means the same dustiness at 150 and at 400 DPI — `speck_radius` is that
profile's sigma in pixels, and `speck_strength` caps how far the deepest speck
blends toward black or white, well below 1 so a speck tints the paper instead of
punching through it. 68% are dark: platen grit, loose toner and paper fibre are
dark, while a bright dropout needs a dead sensor cell. Specks are stamped over
the whole page after the banded Gaussian pass, not inside it, so a blot
straddling a band boundary is not cut in half. `_SPECK_MAX_PER_MPX` is a
mistyped-config guard, not a tuning knob.

`scan.noise.salt_pepper` is listed in `ScanConfig._REMOVED`, so an older config
file gets an error naming the three replacements rather than "unknown key".

## Content

**Table and chart data is synthesized, not parsed.** `text_generator.py` asked
the model to embed Markdown tables, and the model largely ignored it —
`plaintext_000004` (`chart_table == "table"`) only *mentions* a table in prose.
So the element generators parse a real pipe table when one happens to be
present and otherwise synthesize a deterministic Persian table or series from
the subject plus the sample's seed. Confirmed with the user before building.

**Element probability follows the corpus, not CLAUDE.md.** §5.2 specifies
~1/5; `text_generator.py` generated at `0.1`, and the 1600 rows already on
disk carry 108 tables and 58 charts. The renderer uses `0.1` so it matches the
data it is given rather than starving on rows that were never flagged.

**Texture applies to 50% of samples.** Twenty-two prepared blank-page scans
live in `data/texture/` (19 JPEG, 2 WebP, 1 AVIF); a sample gets one or none,
shared across all its sheets (real letters are printed on one kind of paper).
Half the corpus stays untextured so the model sees both clean and
scanned-looking pages. The texture is background only — it gets no bounding
box, but its filename is recorded in the annotation for provenance.

**A texture's format is read from its header, not its suffix.** `page_4.jpg` is
really a WebP. `_probe()` opens the file and believes the decoder, so a
mislabelled asset is used rather than skipped — and an asset in a format this
Pillow build cannot decode is skipped rather than crashing a 20k run.

**Small scans are upscaled, not tiled.** The corpus spans 190×265 to
3000×4000. Tiling the small ones was tried first, on the theory that a 190 px
scan upscaled 13× would smear into a gradient; the tiles instead laid a
periodic rectangular seam across the page, which is exactly the spurious
regularity an OCR model will learn. The upscaled scan is soft but seamless, and
soft grain is what a low-resolution scan of blank paper looks like anyway.

**`formulas.py` and `flowcharts.py` are left empty.** CLAUDE.md §5.2 lists
them, but no upstream stage produces formula or flowchart content, and the
user scoped this build to tables and charts.

## Ground truth

**Two training projections, not one.** `paddle_farmer.jsonl` carries
`{image, text}` per page with no geometry; `paddle_ocr_vl.jsonl` carries
per-region boxes. Splitting them means a recognition fine-tune is unaffected
by bounding-box precision, and the annotation JSON remains the single source
either can be rebuilt from.

**Containment uses an 86% area threshold, not strict inclusion.** Subpixel
layout and shared table-cell borders make a child overhang its parent by a
fraction of a pixel; requiring 1.0 would orphan real children.

**The extractor drops bad boxes instead of failing the sample.** An unknown
`data-ann-type`, a rect that rounds to zero extent, a box attributed to a
sheet that was never rasterised, or a duplicate id (largest box wins) all skip
quietly. At 20k samples, aborting a run over one hairline rule is the wrong
trade.

## Reproducibility

**Per-sample RNG derived as `{seed}:{purpose}:{index}`**, so a sample renders
identically regardless of run order or batch boundaries, and each visual
concern (theme, texture, stamp…) draws from its own stream — adding a new
random choice does not shift every later one.

**Ids come from the manifest position**, not a counter, so re-rendering a
subset with `--only` keeps every `sample_id` stable.

**The annotation JSON is written last and is the resume key**, so a run killed
between page screenshots is retried rather than mistaken for complete. The
manifest CSV row and both JSONL files are appended per sample, so a crash at
sample 900 leaves 899 usable rows.

## Colab handoff (Google Drive)

**`scripts/upload_to_drive.py` mirrors only what training reads.** Page PNGs
and `paddle_farmer.jsonl` go up; markdown, PDF, per-page annotations and
`paddle_ocr_vl.jsonl` stay local. The box-level projection is only needed once
element-level crops enter the mix, and it would double the metadata transfer
for a notebook that starts page-level.

**The 8k unrendered plaintext records ship as one `plaintext_pending.jsonl`,
not 8k files.** Drive charges a round trip per file; one 39 MB bundle uploads
in seconds where 8k creates take tens of minutes and burn API quota. Colab
reads it with one `open()`.

**Every path in the uploaded `manifest.csv` is Drive-root-relative**
(`generate/image/synth_000000_p01.png`), so the notebook joins it with its own
mount point instead of rewriting absolute local paths. The combined manifest
carries both datasets, tagged `dataset=generate|text` and
`status=rendered|pending`, so Source A/B and Source C in the notebook prompt
select by column rather than by directory listing.

**OAuth scope is `drive.file`, not full `drive`.** The script only ever touches
files it created, which keeps consent narrow and avoids Google's restricted
scope review — at the cost of not being able to see a `datasets/` folder made
by hand in the web UI. `--drive-root-id` is the escape hatch for that case.

**Resume is by name plus byte size**, not a local state file: the destination
folder listing is the state. An interrupted 8 GB run resumes by re-listing and
skipping matches, and a re-rendered page with a different size re-uploads on
its own.

## HuggingFace export

**One Parquet family per artifact, not one dataset of images.** The render tree
holds five artifacts per sample; an export that packed only the page images threw
four of them away, so nobody downstream could rebuild the PDF, read the source
letter, or recover a box that `include_elements: false` dropped. Each family is
a subdirectory under `data/`, which is exactly what the Hub reads as a *config*
name, so `load_dataset(repo, "images")` and `load_dataset(repo, "pdf")` address
the same repository and a consumer who wants only pages downloads only pages.

**The per-sample header is repeated on every row of every family.** About thirty
small columns — genre, tone, the metadata-block fields, `has_table` /
`has_chart` / `has_signature` / `has_stamp`, `overflowed`, `missing_assets` —
against a 300 KB page image, so the duplication is free, and it means slicing an
error analysis by `letter_genre` never needs a join. The `has_*` flags are
derived from the annotation's own element types rather than read back from the
render manifest: the annotation is what shipped, so a flag derived from it cannot
disagree with the boxes in the same row.

**The split is a hash of the source letter id, not a shuffle.**
`blake2b(f"{salt}:{key}")` mapped into `[0, 1)` and walked against the cumulative
fractions. Four properties follow, and each is a failure mode avoided:

- Every page of one letter lands in one split, because the key is the letter.
  A naive per-page shuffle would put the same letterhead, signature and
  synthesised table on both sides of the eval.
- The key is `source_sample_id`, not `sample_id`, so a plaintext file rendered
  twice — a second pass, an augmentation sweep — cannot straddle the split.
  `group_by: sample_id` exists for the case where per-render assignment is
  genuinely wanted.
- Adding plaintext later never moves an existing sample, because assignment reads
  only the id. Shuffle-and-slice would re-deal the whole corpus on every export,
  and more plaintext is still arriving.
- `blake2b`, never `hash()`: Python salts `str.__hash__` per process, so `hash()`
  would assign a different split on every run.

**A changed split plan is refused on resume.** `SplitPlan.fingerprint()` is a
`sha256` over the canonicalised plan and lives in the state file; appending a
second batch under different fractions is precisely the leak the hash exists to
prevent, so it is an `ExportError` rather than a warning. A *fresh* export into
an empty directory cannot be caught this way and will happily produce a different
deal — hence the warning in `docs/publishing_to_huggingface.md` that the values
must not move once a dataset is published.

**`min_shard_bytes` (64 MiB) as well as a sample count.** A shard closes when it
has both `shard_samples` samples *and* `min_shard_bytes` of payload, or when the
next sample would breach `max_shard_bytes`. Without the byte floor, `metadata` at
20k samples would be forty files of 200 KB; with it, the blob-free families
collapse to one shard per split while `images` still closes on the sample count,
being long past 64 MiB by then.

**Row-group budgets are per family and are code, not config**: 32 MiB for
`images` and `pdf`, 8 MiB for the text families, 2 MiB for `metadata`. Twelve
writers are open at once (six families × two splits), so one shared 128 MiB
budget would have meant ~1.5 GiB of buffers.

**`paddle_records` and `paddle_farmer_records` moved to
`dataset/common/projections.py`.** `pipeline.py` imports them back and keeps them
in `__all__`, so no caller changed. The reason is layering: `pipeline.py` imports
`..export.huggingface`, so `export` reading the projections out of `pipeline`
would have been a cycle.

**`export/errors.py` holds `ExportError` alone**, so `splits.py`, `families.py`,
`attribution.py`, `huggingface.py` and `upload.py` can all raise it without any
two of them importing each other. The whole package is one-way:
`errors` → `splits`/`families`/`attribution` → `huggingface` → `upload`.

**One `Column` declaration generates the Arrow schema, the `datasets` feature
descriptor, the card's front matter and the card's prose tables.** A new column
cannot ship with the wrong type in one of the four, and cannot ship
undocumented, because the card's note is a field on the declaration.

**A missing artifact is not fatal to the whole export.** A missing *page image*
is still an `ExportError` — a row with no image is not a training example — but a
missing PDF, Markdown or plaintext file skips that one family's row, logs it, and
is counted in the summary and in `dataset_info.json`. A tree rendered before the
text stage existed still exports five families.

**`-of-NNNNN` is not knowable while streaming**, so shards are written under a
staging name and renamed in one pass at close; a resumed export renames what the
previous run named, because the total changed.

## Publishing to the Hub

**`dataset_info` *is* written into the card's front matter.** The first version
deliberately omitted it — the Parquet files already carry the authoritative
`b"huggingface"` descriptor in their Arrow metadata, and a hand-indented second
copy of six configs × ~38 columns is a copy that can be wrong. It is in now
because it is what makes the dataset load *professionally*: `load_dataset` knows
the features and row counts before downloading a shard, and the dataset page
shows them. The hazard is answered by generation rather than by omission — the
block comes from the same `Column` list as the Arrow schema, in the same order,
and is serialised by `yaml.safe_dump` instead of by the template, so indentation
cannot drift. Note that the front matter needs a *different spelling* of the same
features (`dtype: image`, `sequence: int32`, `list:`) than the Arrow metadata
(`{"_type": "Image"}`), because `datasets` reads YAML with
`Features._from_yaml_list`; both tables are keyed by the same column kinds and an
import-time check refuses a kind declared in only one of them.

**`default_config_name` is set**, so `load_dataset("owner/name")` with no config
argument returns the pages. A multi-config repository without it raises and asks
the caller to name a config.

**Only `num_examples` is taken from what was actually written.** `datasets`
compares it against what it reads and raises `NonMatchingSplitsSizesError` on a
mismatch, so it comes from the shard writers' own counts; `num_bytes` and
`download_size` are display-only, the first being the uncompressed payload the
exporter accumulated and the second measured by stat-ing the finished shards.

**Export and upload are separate commands.** A failed upload never costs the
export, and a card typo is a 20 KB re-push rather than a re-pack of 40 GB.
`--hf-upload-only` and `scripts/upload_to_hugging_face.py` both read a finished
tree and recompute nothing.

**One commit per family, and the card goes last.** The card's `configs:` block is
what the Hub's dataset viewer acts on, so a card that arrives before the shards
it describes makes the page report a broken dataset for the whole upload window.
Per-family commits also make an interrupted push resumable at family granularity
(`--units images`) instead of from the beginning.

**`folder` mode by default, `large` on request.** `upload_folder` gives one clean
commit per unit and is right up to a few GB; `upload_large_folder` is chunked and
resumable and survives a dropped connection, at the cost of many small commits
and no commit message or branch. The uploader warns and points at `large` past
20 GiB rather than choosing for you.

**The token is read from the environment only** — `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, then the
`huggingface-cli login` cache. It is never a CLI argument (which would put it in
shell history and in `ps`), never written to a config file, and never logged; the
uploader logs the *name* of the variable it came from. `--hf-dry-run` still
resolves it, because a missing token is worth discovering before a 40 GB push
rather than after.

**`.export_state.json` is excluded twice over** — it matches no unit pattern and
is also in `_NEVER_UPLOAD` — rather than relying on it happening not to match.

**The declared licence is `cc-by-nc-4.0`, not `cc-by-4.0`.** The pages composite
third-party brand marks and UTSig signatures, whose terms are research-only, so a
commercial-use grant is not ours to give. The card states explicitly that the
declared licence covers the pipeline's own output and cannot relicense the
composited material, and reproduces the logo corpus' per-file licences as
`attribution/attribution.csv`.

## LoRA training notebook — build and layout

**The notebook's source of truth is a jupytext percent-format `.py`, not the
`.ipynb`.** `notebooks/paddleocr_vl_lora_train.py` is one linear script whose
`# %%` / `# %% [markdown]` blocks are cells;
`scripts/build_training_notebook.py` converts it to
`notebooks/paddleocr_vl_lora_train.ipynb`. The reason is CLAUDE.md §8: `ruff`
and `black` cannot see inside a hand-edited notebook, and a notebook diff is
unreviewable. The builder has no third-party dependency (an `.ipynb` is JSON),
derives cell ids from the cell index so rebuilds are byte-identical, and offers
`--check` so CI can fail on a stale artefact. It also audits its own output for
the spec's "every cell has a markdown explanation above it" rule — that audit is
what caught Cell 2b sharing Cell 2a's header.

**`notebooks/*.py` gets a per-file-ignores block in `pyproject.toml`.** The
exempted rules are notebook idiom, not slack: `E501` because cells are written
to Jupyter's editor width (~110 columns, and the markdown table rows in the cell
headers cannot be wrapped at all); `E402`/`F811` because every cell re-imports
what it uses so it can be re-run alone; `D100` because the leading markdown cell
does the module docstring's job; `D102`/`D105`/`D107` and
`ANN001`/`ANN003`/`ANN204` because `Trainer`/`Callback`/`Sampler` overrides have
signatures fixed by upstream. Everything else — including `F821`, which proves
every cross-cell symbol is actually defined — stays on.

**`black` skips the notebook source and `ruff` skips the generated `.ipynb`.**
These are two different reasons. black at 90 columns rewrites
`SMOKE_TEST = False  # comment` into a three-line parenthesised block; in a
config cell that a human edits at a glance, that is worse than the long line, so
`force-exclude` (not `exclude`, so the skip holds when pre-commit passes the path
explicitly) leaves `notebooks/*.py` alone. ruff, conversely, lints `.ipynb`
natively — which on the *generated* notebook means 309 duplicate findings, since
its cells are byte-for-byte copies of source that was already linted, and the
`notebooks/*.py` ignores cannot apply to a `.ipynb` path. So the artefact is in
ruff's `extend-exclude`. Net effect: each line of notebook code is linted exactly
once, in the file where it can actually be fixed.

**One notebook, two machines, one switch.** `PHASE = "pilot" | "final"` in Cell 0
selects a `PHASE_CONFIG` entry (`target_samples`, `per_device_batch_size`,
`grad_accum_steps`, `train_max_pixels`, `num_epochs`) and nothing else. Only
Cell 1 (`detect_environment()`), Cell 2 (installs, `drive.mount`, `get_secret`)
and `WORK_DIR` resolution are environment-aware; from Cell 3 onward `RUNTIME` is
only ever *read* (to print the VRAM ceiling, to stamp the run summary), never
branched on. That is the property that makes the file portable, so it is worth
re-checking after any edit.

## LoRA training notebook — dependencies and data

**`transformers` is pinned to `4.57.6`, `trl` is dropped, `einops` is added.**
The pin is not caution: the remote code shipped with PaddleOCR-VL calls
transformers internals whose signatures move between minor releases, and Colab's
preinstalled version drifts weekly. `trl`'s `SFTTrainer` was removed because the
prompt/label masking here is bespoke (image-placeholder expansion has to happen
*after* the pixel budget is known), so `trl` added a dependency and an
abstraction without doing any of the work. `einops` is imported by the model's
own remote modelling file and is not a transformers dependency, so it has to be
installed explicitly or Cell 7 fails on import.

**The notebook owns the resize; the processor is not asked to do it.** Two
upstream facts force this: per-call `images_kwargs` are ignored by this
processor, and `min_pixels`/`max_pixels` are process-global attributes rather
than per-call arguments. So `prepare_image()` resizes to a multiple of
`patch_size × merge_size = 28` under an explicit budget before the processor sees
the image, and Cell 5 prints the processor's *real* `min_pixels`/`max_pixels` and
warns if `train_max_pixels` sits outside them.

**`ON_OVERSIZE = "shrink_image"`, not `"truncate"`.** When image tokens plus
target tokens exceed `MAX_TOTAL_SEQ_LEN`, the sample's *pixel* budget is lowered
so the whole transcription survives. Truncating the target would teach the model
to stop mid-page — the one failure mode that is expensive to detect and fatal in
an OCR model. Records that cannot fit even at the model's `min_pixels` are
dropped, logged and counted rather than silently mangled.

**Persian_Pixel granularity comes from the dataset's own `sample_type` /
config name, not from a heuristic.** An earlier version guessed page-vs-paragraph
from image aspect ratio and text length. Guessing here would corrupt the
55:45 paragraph:page mix that the spec asks for, and the metadata is right there.

**Digit-density oversampling is applied in a sampler, not by
`TrainingArguments`.** `Trainer`'s `group_by_length` materialises `input_ids` for
the whole dataset to measure lengths — decoding every image once per epoch just
to sort them. `DigitAwareLengthGroupedSampler` does both jobs on the precomputed
lengths: multinomial draw by digit-density weight, then length-sorted
megabatches to cut padding waste. `TrainingArguments.group_by_length` stays
`False`; the behaviour moved, it was not lost.

## LoRA training notebook — model, LoRA and optimizer

**LoRA targets are matched by full module path, with `visual` and `mlp_AR`
excluded by prefix.** PaddleOCR-VL's vision tower reuses the names
`q_proj`/`k_proj`/`v_proj`, so the usual bare-name `target_modules` list would
silently adapt the frozen SigLIP encoder too — quietly changing what the run is.
Cell 7 asserts the vision encoder and projector have no trainable parameters
after wrapping, and prints trainable-vs-total.

**`accepts_loss_kwargs = False` is set on the model.** When `Trainer` detects
`**loss_kwargs` in `forward`, it hands the model `num_items_in_batch` and skips
its own `loss / grad_accum` division, expecting the model to normalise. This
model does not. Left alone, gradients would be 20× too large in the pilot
config — a bug that looks like a bad learning rate, not like a bug.

**`ALLOW_MULTI_IMAGE_BATCH = False`, with a batch-fold guard in Cell 8.** Under
`sdpa` the flat `cu_seqlens` vision path lets attention leak across images
packed into one batch. Rather than accept a subtle correctness bug for
throughput, per-device batch size above 1 is folded into gradient accumulation
(same effective batch, same math) and the fold is printed.

**Gradient checkpointing is enabled non-reentrant on the language model only.**
The vision tower is frozen, so checkpointing it buys no memory that matters and
costs a second forward pass over the most expensive part of the graph.

**`paged_adamw_8bit` with differential learning-rate param groups.** The 8-bit
paged optimizer is what keeps the final config inside a T4's 16 GB; the separate
group for the LoRA `B` matrices follows the usual PEFT practice of letting the
zero-initialised side move faster.

**dtype follows compute capability: fp16 below 8.0, bf16 on Ampere and up.**
`TrainingArguments` derives `fp16=(TORCH_DTYPE == "float16")` and
`bf16=(TORCH_DTYPE == "bfloat16")` from that single value, so T4 and A100 need no
edit. No autocast wrapper is needed: peft's `lora.Linear`/`lora.Embedding` cast
their output back to the base layer's dtype, which is why peft is pinned too.

## LoRA training notebook — evaluation and packaging

**The eval prompt is built by splicing image tokens into `PROMPT_TEMPLATE_IDS`,
not by re-templating.** `prepare_inputs_for_generation` accepts `pixel_values`
and `image_grid_thw`, so generation can start from exactly the token sequence
training saw. Any other construction risks a one-token difference between train
and eval, which shows up as an inexplicable CER floor.

**`use_cache=True` is passed explicitly to `generate()`.** The shipped
`generation_config.json` sets `use_cache: false`; honouring it makes evaluation
roughly ten times slower for no benefit.

**Learning-curve JSON is NaN-free by construction.** A validation slice with no
digits yields a NaN digit-CER, and `json.dump` would happily write a bare `NaN`
that is not valid JSON — poisoning the accumulated
`pilot_learning_curve.json` for every later append. `json_safe()` maps
non-finite values to `null` on the way out and `series()` maps them back to NaN
for printing and plotting.

**The merged checkpoint gets its remote code copied in.** A `trust_remote_code`
model saved locally can fail to reload, because `config.auto_map` (and the
nested `vision_config`/`text_config` maps) point at `.py` modules that are not in
the folder. `copy_remote_code()` resolves those entries and downloads each
module next to the weights, so the merged directory is genuinely standalone.

**Hub push is opt-in, private by default, and prints what it is about to
upload.** Cell 10b is gated on `PUSH_TO_HUB`, creates the repo with
`private=True`, and prints the source directory, target URL, byte size and file
list before uploading. Cell 0 asserts that `PUSH_WHAT` has its matching `SAVE_*`
flag enabled, so a push cannot silently upload nothing.

**Open blocker: `shahriarhd/persian-official-letters-synth` is empty on the
Hub.** Cell 3 cannot load the admin-letter half of the mix. Rather than fail
with a stack trace, the notebook prints a labelled BLOCKER banner explaining what
is missing, and offers `ADMIN_LETTERS_LOCAL_DIR` as an escape hatch for a local
copy of the export. This needs the dataset pushed before either phase can run.

## Local training on a 4 GB GPU

The modules under [src/fa_ocr_lora/training/](../src/fa_ocr_lora/training/) are a
second training path, independent of the Colab/A100 notebook: same model, same
adapter shape, but sized for one RTX 3050 Ti Laptop (4096 MiB) with 16 GB of
system RAM, reading the corpora straight off local disk. The notebook is
untouched. Everything below is a decision that differs from it, or from the plan
that preceded these modules.

### What was measured, and on what

Free VRAM is not total VRAM. On a GNOME desktop with a terminal, a file manager
and a viewer open, `torch.cuda.mem_get_info()` reported **1984–2182 MiB free of
3746 MiB visible** across several minutes — the figure moves by ~200 MiB as
windows open and close. Every budget below is against ~1.98 GiB, not 4 GB, and
[vram.py](../src/fa_ocr_lora/training/vram.py) holds a 200 MiB margin back for
exactly that swing.

Resident weights, read from `torch.cuda.memory_allocated()` straight after the
load:

| Precision | Weights |
| --- | --- |
| `nf4` + `quantize_vision: true` | **870 MiB** |
| `nf4`, vision in bf16 | 1510 MiB |
| bf16 throughout | ~1730 MiB (estimate — it does not fit here, so it was never measured) |

Peak allocated against sequence length, at `nf4` + vision in bf16:

| Sequence | Peak | Over weights |
| --- | --- | --- |
| 1600 | 1794 MiB | 275 MiB |
| 1900 | 1852 MiB | 333 MiB |
| 2200 | 1898 MiB | 378 MiB |
| 2500 | 1943 MiB | 423 MiB |
| 2800 | OOM | — |

A slope of **148 MiB per 900 tokens**, linear to within a few MiB across that
range; those two coefficients are what `vram.projected_peak_mib` extrapolates
from. Note where the wall is: at 2500 tokens the run is 40 MiB from the edge with
the vision tower unquantised, which is why the default below is what it is.

With the default configuration — `nf4` *and* `quantize_vision: true` — the run was
measured end to end rather than extrapolated. Over 30 steps on the real 9,197-record
train split, at effective batch 16 and with sequences drawn from the corpus's own
length distribution, the allocator peaked at **1287 MiB** against a 2640 MiB
spendable budget and a 1387 MiB projection; the 2-step smoke run peaked at 1276 MiB.
The projection is therefore conservative by ~100 MiB, and roughly half the budget
goes unused — which is what makes `max_total_seq_len: 3072` a real ceiling rather
than an aspiration.

Throughput, from that same probe: **13.8 s/step, 0.86 s/sample**, so **2.8 h/epoch**
over the 11,497 samples one epoch expands to, and ~5.5 h for the two epochs the
config asks for. The plan estimated 0.86 s/sample from token counts alone, which
turned out to be right. The step time is not flat, though: length-grouped batching
orders each megabatch longest-first, and the probe's 30 steps ran from 25 s down to
5 s. Thirty steps taken from the front of a megabatch is a *pessimistic* sample of
the epoch average, not an optimistic one.

### NF4 by default, not bf16

The plan had bf16 as the default and NF4 as opt-in, on the argument that
bitsandbytes wheels against torch cu130 on Python 3.13 were unverified. They
verified fine (bitsandbytes 0.50.1), and the measurements above show bf16 has no
room: ~1730 MiB of weights plus a 275 MiB floor of activations is already 2005
MiB against a ~1780 MiB spendable budget. So `quantization: nf4` is the default,
`none` is the escape hatch for a machine with the display on another GPU, and the
`quant` extra in [pyproject.toml](../pyproject.toml) is separate precisely so a
`none` run needs none of it.

### `quantize_vision: true` by default, and what it costs elsewhere

Quantising the *frozen* tower saves **640 MiB** — more than every other lever
combined, and the difference between 2500-token records fitting and not. It is
still a real trade: the tower's features change slightly, so an adapter trained
against a 4-bit tower must be *served* against a 4-bit tower. That coupling is
the reason
[modeling.py](../src/fa_ocr_lora/training/modeling.py)'s `_load_base` is shared by
`load_training_model` and `load_eval_model` rather than duplicated: a CER measured
at a different quantisation than training is not a measurement of the adapter at
all, and two copies of that call would let the drift in silently. Phase 3's
inference loader inherits the same obligation.

### Chunked cross-entropy, not a sliced head module

Both the plan and the notebook wrap the LM head in a `SlicedLMHead` module. What
is here instead is `SlicedLossModel`, which during its own forward swaps the head
for `nn.Identity` (restored in a `finally`), then projects **only labelled
positions** in row-chunks under `torch.utils.checkpoint`. Same purpose — the
vocab is 103,424 and HF upcasts logits to fp32, so a 1600-token batch would want
662 MiB for logits and 1.3 GB with the gradient — but two consequences follow that
the module form does not have:

**Non-reentrant checkpointing is mandatory, not a preference.** The reentrant
implementation decides whether to build a backward graph by asking whether its
*inputs* require grad. With a frozen quantised base the decoder's inputs do not,
so gradients would come back `None` and the loss would never move —
silently. `use_reentrant=False` is load-bearing.

**`forward` returns hidden states under the key `"logits"`.** With the head as
Identity, that tensor is `(B, L, 1024)`, not vocab-width. Trainer's evaluation
loop would accumulate one per record on the GPU, so
`prediction_loss_only=True` in
[loop.py](../src/fa_ocr_lora/training/loop.py) is a **requirement**, not an
optimisation.

`loss_chunk_rows` was swept at 128 / 64 / 32 and gave peaks of 1794 / 1807 / 1807
MiB — identical within noise. **Lowering it cannot rescue an OOM**; it is exposed
only because the null result is worth being able to reproduce.

### Trainer assumptions this run breaks

Four, each with its counter in [loop.py](../src/fa_ocr_lora/training/loop.py):

- **`model_accepts_loss_kwargs` is forced to `False`.** `forward(self, *,
  labels=None, **inputs)` has a `VAR_KEYWORD` parameter, from which Trainer infers
  that the model accepts loss kwargs — and then switches to sum-of-tokens
  normalisation *and* passes `num_items_in_batch` through to the base decoder. Our
  loss is already a mean over supervised tokens, so the classic
  divide-by-accumulation-steps path is the correct one. Left alone, gradients come
  out inflated by the accumulation factor.
- **`_save` writes the adapter and nothing else.** The wrapper is neither a
  `PreTrainedModel` nor a `PeftModel`, so the default `_save` would write the full
  1.7 GB base three times over. It also deliberately does not write
  `training_args.bin`: resume never reads it, and the version-controlled YAML is a
  better hyperparameter record than a pickled dataclass.
- **VRAM is injected by overriding `log`, not from a callback.** Callbacks run
  *after* the `log_history` entry is copied, so a callback mutating `logs` would
  change neither the printed line nor the recorded history.
- **The sampler's epoch is advanced by an explicit callback.** Accelerate usually
  forwards `set_epoch`, but only when the batch sampler nests the way it expects.
  If that forwarding stops, every epoch replays the identical order — which looks
  like nothing at all in the logs and quietly halves the value of epoch two.

### The generation config, not the model config, decides the KV cache

`load_eval_model` sets `use_cache` in two places and the second one is what
actually does the work. `generate` reads `use_cache` off the model's **generation**
config and passes it into the forward kwargs, where it overrides whatever
`model.config.use_cache` says — and this model ships `generation_config.json` with
`use_cache: false`. Setting only `config.use_cache = True`, which is the obvious line
to write, changes nothing at all.

What that costs is worth stating, because nothing about it looks like a failure: the
eval completes, every record is scored, and the CER is correct. Measured on one
validation page with a 1,273-token prompt:

| `generation_config.use_cache` | Marginal cost per generated token |
| --- | --- |
| `false` (as the model ships) | **795 ms** |
| `true` | **14 ms** |

A factor of 57, because without a cache every decode step re-forwards the entire
prompt *and* re-runs the vision tower over the page. In wall-clock terms, the
`eval.max_samples: 300` default is ~75 minutes with the cache and most of a day
without it. The first baseline run made here predated the fix and took 622 s to
score four records; that figure is recorded only as a warning.

### LoRA targets are full module paths

The vision tower and the decoder both have `q_proj` / `k_proj` / `v_proj`. Passing
those suffixes to PEFT adapts **both**, which contradicts freezing the tower and
roughly doubles the adapter. Targets are therefore resolved as full paths under
`model.layers.` — 126 decoder projections, plus the embedding and head for 128
adapted modules — and `assert_frozen` fails the run if anything under `visual.` or
`mlp_AR.` comes back trainable.

### Digit oversampling by index expansion

The metadata block (شماره ثبت / تاریخ / پیوست) is CLAUDE.md §5.2's
highest-priority field and is almost entirely digits, so digit-dense records are
oversampled. This is done by **repeating indices** in the sampler's index list,
not with `WeightedRandomSampler`: repetition composes with length-grouped
batching, is exactly reproducible from the seed, and makes one epoch a definite
number of steps. Sampling with replacement gives up all three.

### Manifest details worth knowing

- **`image_path` in `all_features_metadata.csv` is stale** for part of the corpus
  and is not trusted; paths are rebuilt from the row's own identifiers. Records
  that still do not resolve are counted and logged.
- **The pixel CSV must be parsed as CSV.** It is UTF-8 **with BOM**
  (`encoding="utf-8-sig"`) and its `text` column contains embedded newlines, so
  any line-by-line read silently corrupts it. `labels.csv` in the same directory
  is header-only and must not be used.
- **`crop_type_weights` is a quota, not a probability.** Region types are given
  shares of the crop budget, unmet shares are redistributed to types that still
  have candidates, and both the taken and available counts are printed — so
  "`metadata_field` wanted 900 and the corpus only had 640" is visible rather
  than inferred from a short manifest.
- **`_MAX_HOLDOUT_SHARE` caps the validation split** relative to the corpus, so a
  `--limit 200` smoke build cannot hold out 800 of its 200 records.
- Splits are grouped by `group_id` — a letter's pages and crops never straddle
  splits, and neither do rows sharing a source page — because a page seen in
  training and scored in validation reports a number about memorisation.

### Revision pin

`revision: 2b77538ef936`. The current Hub head (commit `a1e4dc4c`, PR #91)
requires transformers 5.x; the environment has 4.57.6, which is also the pin the
notebook uses. Unpinning means the remote code can change under a resumed run —
between two halves of the same training job.

### The 10,000-record mix, as built

9,197 train / 803 validation across 3,899 groups: letter pages 2,336, letter
crops 2,337, pixel 4,524. Sequence lengths came out p50 911, p95 1847, p99 2130,
p100 2758 — every record under the 3,072-token ceiling, so `on_oversize` never
degrades a sample on this corpus. Both `max_total_seq_len` and the mix are YAML
knobs; the numbers here are what the defaults produce.

### Two alignments in the metrics, on purpose

Error counts come from `rapidfuzz` (a true minimal edit distance, and jiwer's own
engine); confusion pairs come from `difflib.SequenceMatcher`, whose opcodes say
*which* character stood where. The first implementation used difflib for both, and
it overstated CER: difflib maximises the longest matching block instead of
minimising the edit script, so 100 identical characters with the first one wrong
came back as an insert plus a delete — 2 errors where the answer is 1. On
near-misses, which are exactly what a trained adapter produces, that inflates the
rate by half. Since the confusion counter reads only equal-length replace spans,
difflib's non-minimality there cannot invent a pair; it can only decline to report
one.

### Known rough edge

`skipped_records.log` records 88,808 `crops_excluded_type` lines — by-design
rejections from the crop quota — among which 274 genuine faults are easy to miss.
Aggregating the by-design reasons into a count rather than a line each is
deferred, not forgotten.

## Tooling

**PEP 8 is enforced by `ruff` and `black`** at `line-length = 90`, declared in
`pyproject.toml` and installed in the dev environment (`pip install -e
'.[dev]'`). Where they are unavailable, `compileall` plus a column sweep is the
fallback — but note that `awk 'length($0) > 90'` counts *bytes* and will
over-report on Persian source; measure with Python's `len()` on decoded text.

### The base model's Persian baseline: CER 1.22

Scored with `eval.py --no-adapter --max-samples 50` over the pinned validation
holdout, 584 s at 11.7 s/record, greedy, `max_new_tokens: 1024`:

| Metric | Base model |
| --- | --- |
| CER | **1.2220** |
| WER | **1.6410** |
| digit CER (47 of 50 records) | **4.2063** |
| digit exact match | **0.0000** |

A CER above 1.0 is not a reporting bug and is worth explaining, because it is the
most informative number here. CER is edit distance divided by *reference* length, so
it exceeds 1.0 when the prediction is longer than the target — which is exactly what
the model does on Persian: it enters repetition loops and emits far more text than the
page contains. The digit CER of 4.2 says the same thing more sharply. Latin content on
the same pages is transcribed correctly and verbatim ("Better Stack", "Hedera Wood &
Furniture"), so the prompt splice and the image pipeline are right; what fails is
Persian specifically.

The confusion table names three distinct failure modes rather than one:

| Expected → predicted | Count | What it is |
| --- | --- | --- |
| `' '` → `'‌'`, `'‌'` → `' '` | 74, 57 | Space vs. zero-width non-joiner |
| `'ی'` → `'ي'`, `'ک'` → `'ك'` | 33 | Persian letters emitted in their Arabic forms |
| `'ا'` → `'ल'`, `'ی'` → `'ं'`, `'گ'` → `'क'` | 18, 12, 10 | Devanagari substituted for Persian |

The first is orthographic and should fall to a small amount of training. The second is
the model reaching for the wrong language's codepoints for the same glyph shapes. The
third is a tokenizer-level confusion between two unrelated scripts and is the clearest
evidence that Persian is genuinely under-served in the base model.

**1.2220 is the number the adapter has to beat**, and the earlier 4-record figure
(CER 0.9059) should be ignored: it predates the KV-cache fix above and 4 records is
not a baseline.
