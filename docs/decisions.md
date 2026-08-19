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

## Tooling

**PEP 8 is enforced by `ruff` and `black`** at `line-length = 90`, declared in
`pyproject.toml` and installed in the dev environment (`pip install -e
'.[dev]'`). Where they are unavailable, `compileall` plus a column sweep is the
fallback — but note that `awk 'length($0) > 90'` counts *bytes* and will
over-report on Persian source; measure with Python's `len()` on decoded text.
