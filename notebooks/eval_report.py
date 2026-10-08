# %% [markdown]
# # PaddleOCR-VL Persian LoRA — evaluation and report bundle
#
# A session that does **only** evaluation, for the case the training notebook
# cannot cover: the run is finished, its adapter is in Drive, and the molab
# session that trained it is gone.
#
# 1. **Secrets & clone** — decode the Drive token, clone `zilber_ocr`.
# 2. **Environment** — torch from the cu128 index (Blackwell is `sm_120`), then
#    the project's extras plus `matplotlib` for the charts.
# 3. **Corpora & manifest** — fetch the datasets and rebuild
#    `unified_manifest.parquet`. The split is a hash of the group id, so the
#    same seed rebuilds the *same* test records the training run held out.
# 4. **Adapter** — pull `adapter_only` back out of Drive.
# 5. **Compare** — score base and base+LoRA over identical test records.
# 6. **Report** — CSV, JSON, Markdown and charts, then push the bundle to Drive.
#
# **Why this is not the old evaluation.** The previous pass took ~4.9 s/record
# and left the card at ~3 GB and ~2 % utilisation, because generation ran one
# record at a time. `eval.batch_size` now left-pads prompts and decodes them
# together, and `eval.dataloader_workers` encodes ahead of the GPU. Same
# arithmetic, same greedy decoding, same prompts - only the scheduling changed.
#
# **Before you start**
#
# - Attach the NVIDIA RTX PRO 6000 in the specs button.
# - Put `GH_TOKEN`, `HF_TOKEN` and `DRIVE_TOKEN_B64` in the Secrets panel
#   (Ctrl/Cmd+J).
# - A full-corpus test pass needs the images on disk: ~50 GB. Set
#   `DATA_MODE = "tiny"` to rehearse the whole notebook in minutes first.

# %% [markdown]
# ## 1 · Secrets and clone
#
# Everything the run needs that is not in git. The PAT goes into
# `~/.git-credentials` rather than onto `argv`, where `ps` and any failure
# message would show it.

# %%
import base64
import os
import subprocess
import sys
from pathlib import Path

REPO = "zilber_ocr"
REPO_URL = "https://github.com/shahriar-hd/zilber_ocr.git"
CONFIG = "configs/lora_config_rtx6000pro_16.yaml"

# Which split to score. "test" is the untouched holdout and the number worth
# quoting; "validation" is what the training loop already watched.
SPLIT = "test"

# Records per variant. None scores the whole split (5,855 on the full corpus);
# a number caps it, which is how a 10-minute sanity pass is done.
MAX_SAMPLES = None

# Prompts per generate() call. This is the setting that turns a 2%-utilised card
# into a busy one. 32 fits comfortably at 1,280 image tokens on 96 GB; drop it
# to 8 or 16 if the GPU is shared, raise it if nvidia-smi still shows headroom.
BATCH_SIZE = 32

# Worker processes decoding and resizing images ahead of the GPU. molab gives
# 4 vCPU by default; more workers than cores only queue behind each other.
EVAL_WORKERS = 4

# "tiny" is ~1% of each corpus and exists to rehearse this notebook end to end.
# "full" is what a quotable number needs.
DATA_MODE = "full"

# Letters image shards to fetch. test-00002 is the smallest file in the dataset.
LETTERS_SHARDS_BY_MODE = {
    "tiny": ("test-00002-of-00003",),
    "subset": ("test-00002-of-00003", "train-00000-of-00015", "train-00001-of-00015"),
    "full": None,  # every shard
}
PIXEL_SHARDS_BY_MODE = {"tiny": 1, "subset": 4, "full": 0}

REPO_DIR = Path.cwd() / REPO
LETTERS_SHARDS = LETTERS_SHARDS_BY_MODE[DATA_MODE]
PIXEL_SHARDS = PIXEL_SHARDS_BY_MODE[DATA_MODE]


def sh(*args: str, cwd=None, env=None, check=True, capture=False) -> subprocess.CompletedProcess[str]:
    """Run a command, streaming or capturing its output."""
    return subprocess.run(
        list(args),
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
        capture_output=capture,
    )


print(f"split {SPLIT} | batch {BATCH_SIZE} | workers {EVAL_WORKERS} | data {DATA_MODE}")

# %% [markdown]
# The three secrets, asserted before anything slow runs.
#
# **Every value is stripped.** A secret pasted into the Secrets panel very often
# arrives with a trailing newline or space, and a token with a newline in it is
# not a token: it breaks the credential line in half and GitHub answers
# "Invalid username or token", which reads like an expired token and is not one.
# The shape report below is what tells the two apart without printing a secret.

# %%
_WANTED = ("GH_TOKEN", "HF_TOKEN", "DRIVE_TOKEN_B64")
missing = [k for k in _WANTED if not os.environ.get(k)]
assert not missing, f"add these in the Secrets panel (Ctrl/Cmd+J): {missing}"
secrets = {k: os.environ[k].strip() for k in _WANTED}
empty = [k for k, v in secrets.items() if not v]
assert not empty, f"these secrets are whitespace only: {empty}"
for _key, _value in secrets.items():
    _raw = os.environ[_key]
    _note = " (stripped whitespace)" if _raw != _value else ""
    # Length and prefix only. Enough to recognise a truncated paste or the wrong
    # kind of token, never enough to use.
    print(f"{_key:18} {len(_value):>5} chars, starts {_value[:4]!r}{_note}")
    if any(c.isspace() for c in _value):
        print(f"  WARNING: {_key} contains whitespace *inside* it - the paste is mangled")

# %% [markdown]
# ### Is the GitHub token actually usable?
#
# `zilber_ocr` is a **private** repository, so the clone needs a working token
# and a failure there is the first thing that can end this notebook. Asking the
# API first turns four different problems into four different sentences:
#
# | Result | Meaning |
# | --- | --- |
# | `401` | the token is invalid, expired or revoked — make a new one |
# | `200` on `/user`, `404` on the repo | the token works but cannot see this repo (a fine-grained PAT with the repo not selected, or no `repo` scope) |
# | `200` on both | the token is fine; any clone failure after this is plumbing, not auth |
#
# A classic PAT needs the **`repo`** scope. A fine-grained PAT needs
# **Contents: Read** and `shahriar-hd/zilber_ocr` in its selected repositories.

# %%
import json as _json_probe
import urllib.error
import urllib.request

OWNER_REPO = "shahriar-hd/zilber_ocr"


def _github(path: str, token: str) -> tuple[int, dict]:
    """GET one GitHub API path with a bearer token, returning status and body."""
    request = urllib.request.Request(
        f"https://api.github.com/{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "zilber-ocr-eval-notebook",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, _json_probe.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            body = _json_probe.loads(exc.read() or b"{}")
        except ValueError:
            body = {}
        return exc.code, body


_status, _body = _github("user", secrets["GH_TOKEN"])
if _status == 401:
    raise SystemExit(
        "GH_TOKEN is rejected by GitHub (401 "
        f"{_body.get('message', '')!r}).\n"
        "The token is invalid, expired or revoked - this is NOT a code problem, "
        "and re-pasting the same value will not fix it.\n"
        "  1. https://github.com/settings/tokens -> generate a new token\n"
        "     classic: tick the 'repo' scope\n"
        "     fine-grained: Contents: Read, and add shahriar-hd/zilber_ocr to "
        "'Only select repositories'\n"
        "  2. Replace GH_TOKEN in the molab Secrets panel (Ctrl/Cmd+J)\n"
        "  3. Re-run this cell"
    )
assert _status == 200, f"unexpected GitHub response {_status}: {_body}"
print(f"token owner  {_body.get('login')} (/user -> 200)")

_repo_status, _repo_body = _github(f"repos/{OWNER_REPO}", secrets["GH_TOKEN"])
if _repo_status == 404:
    raise SystemExit(
        f"The token authenticates as {_body.get('login')} but cannot see "
        f"{OWNER_REPO} (404).\n"
        "The token is valid; it just lacks access to this private repository.\n"
        "  classic PAT      -> tick the 'repo' scope\n"
        "  fine-grained PAT -> Contents: Read, and add the repo under "
        "'Only select repositories'\n"
        "  organisation repo-> an admin may need to approve the token"
    )
assert _repo_status == 200, f"unexpected repo response {_repo_status}: {_repo_body}"
print(f"repo access  {_repo_body.get('full_name')} "
      f"(private={_repo_body.get('private')}, /repos -> 200)")
print("GH_TOKEN is usable - a clone failure after this is plumbing, not auth")

# %% [markdown]
# Clone the repository, or fast-forward the clone that is already here.
#
# **The token goes through the environment, not a file and not `argv`.** A
# `credential.helper` that is a shell snippet reading `$GH_TOKEN` keeps the
# secret out of `ps` (which shows the snippet, not the value) and out of
# `~/.git-credentials` — which matters here because molab's `$HOME` is not
# reliably writable, and a helper that cannot write its store fails in a way
# that looks exactly like a bad token.

# %%
# Prints the two lines git's credential protocol wants, reading the secret from
# the environment of the git process itself.
GIT_CREDENTIAL_HELPER = (
    '!f() { printf "username=x-access-token\\npassword=%s\\n" "$GH_TOKEN"; }; f'
)
git_env = {**os.environ, "GH_TOKEN": secrets["GH_TOKEN"], "GIT_TERMINAL_PROMPT": "0"}

if (REPO_DIR / ".git").is_dir():
    print("already cloned - fetching")
    sh(
        "git", "-c", f"credential.helper={GIT_CREDENTIAL_HELPER}",
        "-C", REPO, "fetch", "--depth", "1", "origin", "main",
        env=git_env, check=False,
    )
    sh("git", "-C", REPO, "reset", "--hard", "FETCH_HEAD", env=git_env, check=False)
else:
    sh(
        "git", "-c", f"credential.helper={GIT_CREDENTIAL_HELPER}",
        "clone", "--depth", "1", REPO_URL, REPO,
        env=git_env,
    )
git_head = sh("git", "-C", REPO, "rev-parse", "--short", "HEAD", capture=True).stdout.strip()
print("commit:", git_head)

# %% [markdown]
# Decode the Drive token. Both the adapter fetch and the report upload need it.

# %%
secrets_dir = REPO_DIR / "secrets"
secrets_dir.mkdir(parents=True, exist_ok=True)
token_path = secrets_dir / "drive_token.json"
token_path.write_bytes(base64.b64decode(secrets["DRIVE_TOKEN_B64"]))
token_path.chmod(0o600)
print("wrote", token_path, f"({token_path.stat().st_size} bytes)")

# %% [markdown]
# ## 2 · Environment
#
# `sm_120` has no kernels in the cu124/cu126 builds, so torch comes from the
# **cu128** index before anything else is installed.

# %%
import importlib.metadata as md
import importlib.util

if importlib.util.find_spec("torch") is None or "cu128" not in md.version("torch"):
    sh(
        sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    )
torch_version = sh(
    sys.executable, "-c", "import torch; print(torch.__version__, '| cuda', torch.version.cuda)",
    capture=True,
).stdout.strip()
print("torch", torch_version)
assert "cu128" in torch_version.split("|")[0], f"Blackwell needs the cu128 build, got {torch_version}"

# %% [markdown]
# The project itself, plus `report` for matplotlib. `train_molab.sh tf`
# reconciles TensorFlow/Keras so `from transformers import Trainer` imports.

# %%
sh(
    sys.executable, "-m", "pip", "install", "-e",
    ".[local-train,drive,hf,hub,report]", "datasets", "sentencepiece",
    cwd=REPO_DIR,
)
sh("bash", "scripts/train_molab.sh", "tf", cwd=REPO_DIR, check=False)
print("project editable in", REPO_DIR)

# %% [markdown]
# Prove the GPU is attached and supports bf16 before anything loads weights.

# %%
import torch

assert torch.cuda.is_available(), "no CUDA device - attach the RTX PRO 6000 in the specs button"
capability = torch.cuda.get_device_capability(0)
free, total = torch.cuda.mem_get_info()
print(f"gpu    {torch.cuda.get_device_name(0)} sm_{capability[0]}{capability[1]}")
print(f"vram   {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free")
assert torch.cuda.is_bf16_supported(), "this profile requires bf16"
print("bf16   yes")

# %% [markdown]
# ## 3 · Corpora
#
# The manifest describes records by path, so the images have to be on disk
# before it is rebuilt. Metadata first - it is small and lands immediately.

# %%
import csv

from huggingface_hub import hf_hub_download

sh("df", "-h", str(REPO_DIR.parent), check=False)
LETTERS_REPO = "shahriarhd/persian-official-letters-synth"
letters_root = REPO_DIR / "data" / "dataset"
letters_root.mkdir(parents=True, exist_ok=True)
letters_manifest = letters_root / "paddle_ocr_vl.jsonl"
for _name in ("manifest.csv", "paddle_farmer.jsonl", "paddle_ocr_vl.jsonl"):
    _local = hf_hub_download(LETTERS_REPO, _name, repo_type="dataset", local_dir=letters_root)
    print(f"{_name:24} {Path(_local).stat().st_size / 10**6:.1f} MB")

# %% [markdown]
# The page images, which live inside Parquet shards and are written out as PNGs
# in the layout `training/sources.py` reads.

# %%
import pyarrow.parquet as pq
from huggingface_hub import list_repo_files

letters_images = letters_root / "images"
letters_images.mkdir(parents=True, exist_ok=True)
if LETTERS_SHARDS is None:
    shards = tuple(
        Path(name).stem
        for name in list_repo_files(LETTERS_REPO, repo_type="dataset")
        if name.startswith("data/images/") and name.endswith(".parquet")
    )
else:
    shards = LETTERS_SHARDS
have = {p.name for p in letters_images.glob("*.png") if p.stat().st_size > 0}
print(f"{len(have)} images already local; {len(shards)} shard(s) to visit")
added = 0
for shard in shards:
    _local = hf_hub_download(LETTERS_REPO, f"data/images/{shard}.parquet", repo_type="dataset")
    _parquet = pq.ParquetFile(_local)
    for _batch in _parquet.iter_batches(batch_size=8, columns=["image"]):
        for _cell in _batch.column("image"):
            _entry = _cell.as_py()
            _name = Path(_entry["path"]).name
            if _name in have:
                continue
            (letters_images / _name).write_bytes(_entry["bytes"])
            have.add(_name)
            added += 1
print(f"letters images: {len(have)} on disk (+{added})")

# %% [markdown]
# Persian Pixel, reshaped into the CSV plus `images/` layout the loader expects.
# The CSV is UTF-8 **with BOM** and its `text` column contains newlines, so it is
# written and read as CSV rather than line by line.

# %%
PIXEL_REPO = "Omarrran/Persian_Pixel"
pixel_root = REPO_DIR / "data" / "persian_pixel"
pixel_images = pixel_root / "images"
pixel_images.mkdir(parents=True, exist_ok=True)
pixel_csv = pixel_root / "all_features_metadata.csv"
FIELDS = ["image_filename", "text", "source_parquet", "image_path", "width", "height"]
pixel_rows: dict[str, dict] = {}
if pixel_csv.is_file():
    with pixel_csv.open(encoding="utf-8-sig", newline="") as handle:
        for _row in csv.DictReader(handle):
            pixel_rows[_row["image_filename"]] = _row
pixel_files = sorted(
    name
    for name in list_repo_files(PIXEL_REPO, repo_type="dataset")
    if name.endswith(".parquet") and "page" in name
)
if PIXEL_SHARDS:
    pixel_files = pixel_files[:PIXEL_SHARDS]
print(f"{len(pixel_rows)} pixel rows already local; {len(pixel_files)} shard(s) to visit")
for _file in pixel_files:
    _local = hf_hub_download(PIXEL_REPO, _file, repo_type="dataset")
    _parquet = pq.ParquetFile(_local)
    for _batch in _parquet.iter_batches(batch_size=16):
        _table = _batch.to_pylist()
        for _record in _table:
            _image = _record.get("image") or {}
            _name = Path(_image.get("path", "")).name
            if not _name or _name in pixel_rows:
                continue
            (pixel_images / _name).write_bytes(_image["bytes"])
            pixel_rows[_name] = {
                "image_filename": _name,
                "text": _record.get("text", ""),
                "source_parquet": Path(_file).name,
                "image_path": f"images/{_name}",
                "width": _record.get("width", ""),
                "height": _record.get("height", ""),
            }
with pixel_csv.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    for _row in pixel_rows.values():
        writer.writerow({key: _row.get(key, "") for key in FIELDS})
print(f"pixel rows: {len(pixel_rows)} in {pixel_csv.name}")

# %% [markdown]
# ### Audit
#
# What is actually on disk, against the ceilings in the config. A subset run is
# expected to fall short; the manifest draws from what exists.

# %%
import json as _json

import yaml

config_path = REPO_DIR / CONFIG
assert config_path.is_file(), f"{config_path} is missing from the clone"
profile = yaml.safe_load(config_path.read_text(encoding="utf-8"))
letters_pages = 0
for _line in letters_manifest.open(encoding="utf-8"):
    if not _line.strip():
        continue
    image = _json.loads(_line).get("image", "")
    if image and (letters_manifest.parent / image).is_file():
        letters_pages += 1
with pixel_csv.open(encoding="utf-8-sig", newline="") as _handle:
    pixel_pages = sum(1 for _ in csv.DictReader(_handle))
run_dir = REPO_DIR / profile["paths"]["output_dir"]
print(f"config        {config_path.name} @ {git_head}")
print(f"letters pages {letters_pages:>6} of {profile['data']['letters_pages']} ({DATA_MODE})")
print(f"pixel  pages  {pixel_pages:>6} of {profile['data']['pixel_pages']} ({DATA_MODE})")
print(f"run dir       {run_dir}")
sh("du", "-sh", str(letters_images), str(pixel_root), check=False)

# %% [markdown]
# ## 4 · Manifest and adapter
#
# Rebuild the manifest. The splits are a hash of the group id and the seed has
# not moved, so this reproduces the **same** test records the training run held
# out - which is the only reason a fresh session's number is comparable.

# %%
stage_env = {**os.environ, "CONFIG": CONFIG}
sh("bash", "scripts/train_molab.sh", "check", cwd=REPO_DIR, env=stage_env, check=False)
sh(
    "bash", "scripts/train_molab.sh", "manifest",
    "--letters-pages", str(profile["data"]["letters_pages"]),
    "--letters-crops", str(profile["data"]["letters_crops"]),
    "--pixel-pages", str(profile["data"]["pixel_pages"]),
    cwd=REPO_DIR, env=stage_env, check=False,
)

# %% [markdown]
# Pull `adapter_only` back out of Drive. `fetch_folder` is the general case of
# the resume path: no marker names the adapter and it is not a checkpoint, so
# the step-number logic does not apply to it.
#
# If the adapter is already local - a session that just trained - this is a
# no-op and the files are reused by size match.

# %%
import sys as _sys

if str(REPO_DIR / "src") not in _sys.path:
    _sys.path.insert(0, str(REPO_DIR / "src"))

from fa_ocr_lora.training.checkpoint_sync import CheckpointSync
from fa_ocr_lora.training.config import load_training_config

config = load_training_config(config_path)
adapter_dir = config.paths.adapter_dir
sync = CheckpointSync(
    run_name=config.drive.resolve_run_name(config.paths.output_dir),
    output_dir=config.paths.output_dir,
    root_folder=config.drive.folder,
    root_folder_id=config.drive.folder_id or None,
    token=token_path,
    interactive=False,
    workers=config.drive.workers,
)
sync.start()
if adapter_dir.is_dir() and any(adapter_dir.iterdir()):
    print("adapter already local:", adapter_dir)
else:
    fetched = sync.fetch_folder("adapter_only", adapter_dir)
    assert fetched is not None, (
        "no adapter_only in Drive for this run - check drive.folder_id and "
        "that the training run finished its final sync"
    )
print("adapter:", adapter_dir)
for _file in sorted(adapter_dir.iterdir()):
    print(f"  {_file.name:32} {_file.stat().st_size / 10**6:>8.1f} MB")

# %% [markdown]
# ## 5 · Compare base against base + LoRA
#
# Both variants are scored over the **same** records, sequentially so peak
# memory stays that of one model. This is the long cell; at `BATCH_SIZE` 32 a
# full 5,855-record test split is a fraction of the ~9.5 h the unbatched pass
# took.
#
# The run is detached so a dropped connection does not take it with it, and the
# loop below keeps the kernel busy so marimo's 90-minute idle timer never fires.

# %%
compare_cmd = [
    "bash", "scripts/train_molab.sh", "compare",
    "--split", SPLIT,
    "--batch-size", str(BATCH_SIZE),
    "--eval-workers", str(EVAL_WORKERS),
]
if MAX_SAMPLES:
    compare_cmd += ["--max-samples", str(MAX_SAMPLES)]
compare_log = run_dir / f"compare_{SPLIT}.log"
run_dir.mkdir(parents=True, exist_ok=True)
print("running:", " ".join(compare_cmd))
with compare_log.open("w", encoding="utf-8") as _handle:
    compare_proc = subprocess.Popen(
        compare_cmd, cwd=str(REPO_DIR), env=stage_env,
        stdout=_handle, stderr=subprocess.STDOUT, text=True,
    )
print("pid", compare_proc.pid, "->", compare_log)

# %% [markdown]
# Watch it. Prints only what is new since the last poll, and stops as soon as
# the process exits - which also releases the cell.

# %%
import time

_seen = 0
while True:
    if compare_log.exists():
        _lines = compare_log.read_text(errors="replace").splitlines()
        _new = _lines[_seen:]
        if len(_new) > 25:
            print(f"... ({len(_new) - 25} earlier lines skipped)")
            _new = _new[-25:]
        for _line in _new:
            print(_line)
        _seen = len(_lines)
    _gpu = sh(
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader", check=False, capture=True,
    )
    print(time.strftime("%H:%M:%S"), "gpu", (_gpu.stdout or "").strip())
    if compare_proc.poll() is not None:
        break
    time.sleep(60)
print("compare exited", compare_proc.returncode)
assert compare_proc.returncode == 0, "compare failed - read the log tail above"

# %% [markdown]
# ## 6 · Report bundle
#
# Reads the comparison the previous cell wrote and renders it: `overall.csv`,
# `by_dataset.csv`, `by_source.csv`, `fields.csv`, `confusions.csv`,
# `training_curve.csv`, `summary.json`, `report.md` and `charts/*.png`.
#
# Loads no model, so it is cheap to re-run after editing a label.

# %%
report_dir = run_dir / f"report_{SPLIT}"
report_cmd = [
    "bash", "scripts/train_molab.sh", "report",
    "--split", SPLIT,
    "--output-dir", str(report_dir),
]
for _name in (f"eval_{SPLIT}.json", "eval_validation.json", "eval_validation_base.json"):
    _candidate = run_dir / _name
    if _candidate.is_file():
        report_cmd += ["--eval-json", f"{_candidate.stem}={_candidate}"]
sh(*report_cmd, cwd=REPO_DIR, env=stage_env, check=False)

# %% [markdown]
# The headline table, read back from the bundle so what is printed is exactly
# what was written.

# %%
import pandas as pd

summary = _json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
print(f"adapter {summary.get('adapter')}")
print(f"records {summary.get('records')}")
overall = pd.read_csv(report_dir / "overall.csv")
by_dataset = pd.read_csv(report_dir / "by_dataset.csv")
print("\n== overall ==")
print(overall.to_string(index=False))
print("\n== by dataset ==")
print(
    by_dataset[
        ["dataset", "n", "base_cer", "adapter_cer", "delta_cer", "base_wer", "adapter_wer"]
    ].to_string(index=False)
)

# %% [markdown]
# The per-field audit - the metric CER cannot see. A page is ~700 characters of
# prose against ~20 of شماره ثبت and تاریخ, so the block needs its own table.

# %%
fields = pd.read_csv(report_dir / "fields.csv")
if len(fields):
    pivot = fields.pivot_table(
        index="field", columns="rate", values=["base", "adapter"], aggfunc="first"
    )
    print(pivot.to_string())
else:
    print("no field audit in this comparison (eval.field_audit was off)")

# %% [markdown]
# The charts, inline.

# %%
import marimo as mo

chart_dir = report_dir / "charts"
charts = sorted(chart_dir.glob("*.png")) if chart_dir.is_dir() else []
print(f"{len(charts)} chart(s) in {chart_dir}")
mo.vstack([mo.image(str(path), caption=path.stem) for path in charts]) if charts else None

# %% [markdown]
# ## 7 · Push the bundle to Drive
#
# molab clears everything over 1 GB and all of `data/` when the session ends, so
# this is the step that makes the report outlive the box. The bundle lands under
# the run's own Drive folder, beside the checkpoints it describes.

# %%
uploaded = sync.push_directory(report_dir, remote_name=f"report_{SPLIT}")
# The predictions files are the evidence behind every number, so they go too -
# one JSONL per variant, uncapped, with per-record fields and findings.
predictions = sorted(run_dir.glob(f"compare_{SPLIT}_*_predictions.jsonl"))
if predictions:
    staging = run_dir / f"predictions_{SPLIT}"
    staging.mkdir(parents=True, exist_ok=True)
    for _path in predictions:
        (staging / _path.name).write_bytes(_path.read_bytes())
    uploaded += sync.push_directory(staging, remote_name=f"predictions_{SPLIT}")
sync.close(timeout=config.drive.drain_timeout_s)
print(f"uploaded {uploaded} file(s) to Drive under {sync.run_name}/")
print(f"  report_{SPLIT}/        tables, summary.json, report.md, charts/")
if predictions:
    print(f"  predictions_{SPLIT}/   per-record JSONL for both variants")

# %% [markdown]
# ### Restarting in a fresh session
#
# Everything above is idempotent. A new session re-downloads the corpora,
# rebuilds the identical manifest from the seed, re-fetches the adapter and
# re-scores. Only the corpus download is slow; the compare is minutes at
# `BATCH_SIZE` 32 rather than the hours it used to be.
#
# To score the validation holdout instead, set `SPLIT = "validation"` in cell 1
# and re-run from section 4.
