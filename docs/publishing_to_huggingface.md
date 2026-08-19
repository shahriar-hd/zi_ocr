# Publishing the dataset to HuggingFace

End to end: from a rendered corpus on disk to a dataset page anyone can load
with `load_dataset("<you>/persian-official-letters-synth")`.

There are two halves, and they are deliberately separate commands:

1. **Export** — read the render tree, pack it into Parquet, write the card.
   Local, no network, no token.
2. **Upload** — push that finished tree to a Hub dataset repository. Network,
   token, nothing to recompute.

Splitting them means a failed upload never costs you the export, and a card
typo is a 20 KB re-push rather than a re-pack of 40 GB.

---

## 0. What you need once

### Install the two optional extras

```bash
pip install -e '.[hf,hub]'
# or, without installing the package:
pip install 'pyarrow>=15' 'huggingface_hub>=0.26'
```

`hf` (pyarrow) is the export. `hub` (huggingface_hub) is the upload. Both are
imported lazily, so a machine that only renders pages needs neither.

### Create a HuggingFace account and a write token

1. Sign up (or log in) at <https://huggingface.co>.
2. Go to <https://huggingface.co/settings/tokens>.
3. **New token** → give it a name (e.g. `fa-ocr-lora-upload`) → role **Write**.
   A read token cannot create a repository or push a file.
   - If you use the newer fine-grained tokens, tick *Write access to contents
     of all repos you can access* (or scope it to this one repository).
4. Copy the `hf_...` value. The Hub shows it once.

### Put the token in the environment

Add it to `.env` in the repository root (which is gitignored):

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Anything below also works, in this order of preference:
`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, then the
cache that `huggingface-cli login` writes to `~/.cache/huggingface/token`.

The token is read from the environment only. It is never accepted as a
command-line argument, never written into a config file, and never logged — the
uploader logs the *name* of the variable it came from, not the value.

### Name the destination

```dotenv
FA_OCR_HF_REPO_ID=<your-username>/persian-official-letters-synth
```

`owner/name`: the owner is your username or an organisation you can write to.
This lives in `.env` rather than `configs/synth_config.yaml` because it is
per-person; the config's `huggingface.hub_repo_id` is an alternative if you
prefer to commit it. `--hf-repo` / `--repo` overrides both.

### (Optional) create the repository in the web UI

You can skip this — the first upload creates it. If you would rather see it
first: <https://huggingface.co/new-dataset> → Owner + Dataset name → **Dataset**
(not Model) → public or private → *Create dataset*. Use exactly the same
`owner/name` as `FA_OCR_HF_REPO_ID`.

---

## 1. Export the corpus locally

Over a render tree that already exists:

```bash
fa-ocr-synth --hugging-face-only
```

Or render and export in one run:

```bash
fa-ocr-synth --count 20000 --hugging-face
```

What lands in `data/hf/` (or `huggingface.output_dir`):

```
data/hf/
├── README.md              # the dataset card; its front matter is what the Hub reads
├── dataset_info.json      # features, per-split counts, sizes, the split plan, provenance
├── metadata.csv           # flat per-page index, for a reader without pyarrow
├── splits.csv             # sample_id, source_sample_id, split — the assignment, auditable
├── attribution/
│   └── attribution.csv    # the logo corpus' per-file licences, republished
└── data/
    ├── images/       train-00000-of-00040.parquet, test-00000-of-00005.parquet
    ├── plaintext/    …
    ├── markdown/     …
    ├── pdf/          …
    ├── annotations/  …
    └── metadata/     …
```

Each subdirectory under `data/` is a HuggingFace **config**, which is what makes
`load_dataset(repo, "pdf")` work against the same repository as
`load_dataset(repo, "images")`.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--hf-families images,annotations` | export a subset of the six families |
| `--hf-splits train=0.9,test=0.1` | override the split shares (see the warning below) |
| `--hf-group-by source_sample_id` | what the split hashes; leave it alone unless you know why |
| `--hf-output-dir DIR` | write the tree somewhere else |
| `--hf-shard-samples N`, `--hf-max-shard-bytes B` | how a shard closes |

> **Do not change `splits`, `split_salt` or `group_by` after you have
> published.** Assignment is a hash of the letter id, so different shares re-deal
> every sample — test pages become train pages. A *resume* under a changed plan
> is refused outright (the plan's fingerprint is in the state file), but a fresh
> export into an empty directory will happily produce a different deal.

Check what you got before pushing:

```bash
python -c "import json; d=json.load(open('data/hf/dataset_info.json')); \
print(d['samples'], 'letters'); print({k: v['rows'] for k, v in d['configs'].items()}); \
print(d['splits'])"
head -40 data/hf/README.md
```

---

## 2. Dry-run the upload

```bash
python scripts/upload_to_hugging_face.py --dry-run
```

This makes no network request. It resolves the token (so a missing one fails
here, not 40 GB in), lists every unit it would push with file counts and sizes,
and reports which environment variable the token came from:

```
INFO dry run: would upload you/persian-official-letters-synth [folder] images=45f/38911.2MiB, …
INFO   images       45 file(s)  38911.2 MiB  data/images/*.parquet
…
INFO   cards         5 file(s)      1.2 MiB  README.md, dataset_info.json, metadata.csv, splits.csv, attribution/*
dry run: 63 file(s), 38.42 GiB -> https://huggingface.co/datasets/you/persian-official-letters-synth
token would come from: HF_TOKEN
```

Equivalently, through the pipeline CLI: `fa-ocr-synth --hf-upload-only --hf-dry-run`.

---

## 3. Upload

```bash
python scripts/upload_to_hugging_face.py            # public
python scripts/upload_to_hugging_face.py --private  # private, if this run creates it
```

or in one command with the export:

```bash
fa-ocr-synth --hugging-face-only --hf-upload
```

What happens, in order:

1. The repository is created if needed (`repo_type="dataset"`, `exist_ok=True`).
2. One commit per family: `images`, then `plaintext`, `markdown`, `pdf`,
   `annotations`, `metadata`.
3. **The card and the sidecars go last.** The card's `configs:` block is what the
   dataset viewer acts on, so publishing it before the shards it points at makes
   the page report a broken dataset for the whole upload window.

For a corpus of many gigabytes:

```bash
python scripts/upload_to_hugging_face.py --mode large
```

`--mode large` uses the Hub's `upload_large_folder`: chunked, resumable, many
small commits, and it recovers from a dropped connection by itself. It cannot
take a commit message or a branch. The default `folder` mode is one clean commit
per family and is the better choice up to a few GB; the uploader warns and points
at `large` past 20 GiB.

Other flags:

| Flag | Effect |
| --- | --- |
| `--units images` | push one family — after re-exporting just that one |
| `--units cards` | push only README.md / dataset_info.json / the CSVs / attribution |
| `--revision my-branch` | commit to a branch, creating it if needed (`folder` mode) |
| `--no-create` | fail rather than create a repository that does not exist |
| `--export-dir DIR` | publish a tree from somewhere other than the configured output |
| `--repo owner/name` | override `FA_OCR_HF_REPO_ID` |

Re-running is safe: the Hub deduplicates by content hash, so files already there
are not re-sent. An interrupted upload resumes with `--units <family it stopped on>`.

---

## 4. Verify the page

Open `https://huggingface.co/datasets/<owner>/<name>` and check:

- The **Dataset Viewer** shows rows with rendered page thumbnails. If it says
  *configuration error*, the `configs:` block and the shards disagree — usually a
  family whose Parquet files did not finish uploading. Re-push that family.
- The config dropdown lists all six families, and each has `train` and `test`.
- **Data Studio** / the file browser shows the row counts from the card's
  `dataset_info` front matter.
- The card body renders: the licence sections, the attribution tables, the split
  explanation.

Then load it the way a consumer will:

```python
from datasets import load_dataset

# the default config is `images`, so no config name is needed
dataset = load_dataset("<owner>/persian-official-letters-synth")
train = dataset["train"]
print(train)                       # features + num_rows, from the card's front matter
page = train[0]
page["image"].save("page.png")     # decoded PIL image, not a byte struct
print(page["subject"], page["registration_number"], page["split"])

# any other artifact by config name
pdfs = load_dataset("<owner>/persian-official-letters-synth", "pdf", split="train")
open("letter.pdf", "wb").write(pdfs[0]["pdf"])

# stream instead of downloading 40 GB
stream = load_dataset("<owner>/persian-official-letters-synth", streaming=True, split="train")
print(next(iter(stream))["text"][:200])
```

A private repository needs `load_dataset(..., token=True)` (or
`huggingface-cli login`) on the reader's side.

### Why the one-argument form works

Three keys in the card's YAML front matter, all generated by the exporter:

| Key | What it does |
| --- | --- |
| `configs` | maps each config name to its shard globs, so `load_dataset(repo, "pdf")` resolves |
| `default_config_name` | what a bare `load_dataset(repo)` uses — without it, a multi-config repository raises and asks you to name one |
| `dataset_info` | each config's `features` and per-split `num_examples` / `num_bytes` / `download_size`, so the schema and row counts are known before a shard is downloaded, and the page shows them |

`dataset_info.features` is generated from the same column declarations as the
Arrow schema, so the two cannot disagree — which matters, because `load_dataset`
casts every shard to the features declared there. `num_examples` is verified by
`datasets` on load: a wrong count raises `NonMatchingSplitsSizesError`, so it is
taken from what the writers actually wrote rather than estimated.

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `no HuggingFace token found` | `HF_TOKEN` is unset or `.env` was not picked up. Check with `python -c "import os; print(bool(os.getenv('HF_TOKEN')))"`. |
| `403 Forbidden` on push | the token is read-only, or the owner is an organisation you cannot write to. Make a **write** token. |
| `no Hub repository configured` | set `FA_OCR_HF_REPO_ID=owner/name` in `.env`, or pass `--repo`. |
| `export tree not found` | run the export first (`--hugging-face-only`). |
| `export tree at … has no README.md` | `… is missing ['README.md'], so the export did not finish` — re-run the export. |
| `huggingface_hub is required` | `pip install -e '.[hub]'`. |
| Viewer: *configuration error* | shards for a config are missing or still uploading. Re-push that family, then `--units cards`. |
| `NonMatchingSplitsSizesError` on load | the card and the shards are from different exports. Re-export, then push both. |
| Upload keeps dropping | `--mode large`. |

## What the licences oblige you to keep

The card is generated with the licensing sections already filled in — the UTSig
signature dataset and its paper, the logo corpus' per-file licences republished
as `attribution/attribution.csv`, the model credited with the letter text, and
the statement that every letter is synthetic and every logo and stamp is present
for research and testing only. `docs/decisions.md` records why the declared
licence is `cc-by-nc-4.0`. If you edit the card on the Hub, keep those sections;
they are the terms under which the source material may be redistributed.
