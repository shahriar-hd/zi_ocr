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

**Texture applies to 50% of samples.** Five prepared blank-page scans live in
`data/texture/`; a sample gets one or none, shared across all its sheets (real
letters are printed on one kind of paper). Half the corpus stays plain white
so the model sees both clean and scanned-looking pages. The texture is
background only — it gets no bounding box, but its filename is recorded in the
annotation for provenance.

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

## Tooling

**PEP 8 is enforced by `compileall` plus a column sweep**, because `ruff` and
`black` are not installed in this environment. `pyproject.toml` declares both
at `line-length = 90` for CI. Note that `awk 'length($0) > 90'` counts *bytes*
and will over-report on Persian source; measure with Python's `len()` on
decoded text.
