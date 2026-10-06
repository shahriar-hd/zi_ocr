# %% [markdown]
# # PaddleOCR-VL-1.6 Persian LoRA — molab run
#
# Takes a bare molab session to a finished LoRA adapter for
# `PaddlePaddle/PaddleOCR-VL-1.6`, with every checkpoint mirrored to Google
# Drive, and ends on the only question that matters: **did the adapter beat the
# base model, and on which dataset.**
#
# This notebook runs **top to bottom**. That is the one thing it does
# differently from `molab_train.ipynb`, which is a marimo app: marimo executes a
# cell only when something downstream consumes its value, so a setup cell whose
# name nobody reads is skipped in silence, and the published run of it spent
# eleven minutes polling a job that had already finished. Here, cell *n* runs
# after cell *n-1* and nothing is conditional on a dependency graph.
#
# ## Before you start
#
# 1. **Attach the GPU** (notebook specs button). Nothing here runs on CPU — the
#    environment cell asserts CUDA and bf16. Built for the RTX PRO 6000
#    Blackwell (`sm_120`, 96 GB), which is why torch comes from the cu128 index.
# 2. **Add three secrets** (Secrets panel, `Ctrl`/`Cmd`+`J`). They are read from
#    the environment, so molab's panel, a `.env` file at the repo root and a
#    plain `export` all work:
#
#    | Secret | What it is |
#    | --- | --- |
#    | `GH_TOKEN` | GitHub PAT with read access to the repo |
#    | `HF_TOKEN` | Hugging Face token — `Persian_Pixel` is gated |
#    | `DRIVE_TOKEN_B64` | `scripts/mint_drive_token.py` output, base64 |
#
# 3. **Leave `DATA_MODE = "tiny"` for the first pass.** ~0.6 GB, and the whole
#    pipeline finishes in minutes. Only move to `"full"` once a tiny run has
#    produced a comparison table you believe.
#
# Sessions die after 12 h and an *idle* kernel after 90 min. The monitor cell
# prints once a minute precisely so the kernel is never idle while the detached
# trainer works.

# %%
# ---------------------------------------------------------------------------
# Single source of truth. Nothing below this cell hardcodes a path or a knob.
# ---------------------------------------------------------------------------
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "zilber_ocr"
REPO_URL = "https://github.com/shahriar-hd/zilber_ocr.git"
REPO_BRANCH = "main"

# PaddleOCR-VL-1.6. The sibling profile (lora_config_rtx6000pro.yaml) trains the
# original 0.9B release through remote code on transformers 4; this one trains
# 1.6, which transformers 5 carries natively as AutoModelForImageTextToText.
# They are different base models — never score one's adapter against the other.
CONFIG = "configs/lora_config_rtx6000pro_16.yaml"

# "tiny"   ~1% of each corpus. check -> ... -> compare in minutes. Start here.
# "subset" ~20%, a wider dry run, ~15 GB.
# "full"   what a real adapter needs.
DATA_MODE = "tiny"

# Drive folder id to mirror checkpoints into. Leave "" to let the run resolve
# `drive.folder` ("fa-ocr-lora") by name under My Drive. REQUIRED if you
# authenticate with a service account, or if the folder was made by hand in the
# Drive UI — the drive.file scope cannot see a folder this code did not create.
DRIVE_FOLDER_ID = ""

# Which letters shards to pull. test-00002 is the smallest file in the dataset,
# 164 pages in 226 MB against 1,364 pages in 2 GB for a train shard, so a tiny
# run costs a tenth of the download for everything it actually uses.
LETTERS_SHARDS = {
    "tiny": ["test-00002-of-00003"],
    "subset": [f"train-{i:05d}-of-00018" for i in range(3)] + ["test-00000-of-00003"],
    "full": (
        [f"train-{i:05d}-of-00018" for i in range(18)]
        + [f"test-{i:05d}-of-00003" for i in range(3)]
    ),
}[DATA_MODE]

# Persian Pixel shards; each holds ~554 page-level rows.
PIXEL_SHARDS = {"tiny": 1, "subset": 2, "full": 56}[DATA_MODE]

# Ceilings handed to `manifest`. The builder takes min(quota, available), so
# these bind before the files on disk do — which is what keeps a tiny run tiny
# when an earlier subset run left its images behind. Empty tuple = the config's
# own ceilings. The holdout has to shrink with them or there is nothing to
# train on.
MANIFEST_ARGS: tuple[str, ...] = ()
if DATA_MODE == "tiny":
    MANIFEST_ARGS = (
        "--letters-pages", "268",
        "--letters-crops", "600",
        "--pixel-pages", "310",
        "--validation-size", "100",
    )

# Which split the final base-vs-adapter comparison scores. `test` is held out
# from both training and the in-loop validation, so it is the honest one.
COMPARE_SPLIT = "test"

REPO_DIR = Path(REPO).resolve()
os.environ.setdefault("HF_HOME", str(REPO_DIR / ".cache" / "huggingface"))


def sh(
    *args: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command, returning the completed process rather than discarding it."""
    return subprocess.run(
        args, cwd=cwd, env=env, check=check, capture_output=capture, text=True
    )


print(f"repo        {REPO_URL}@{REPO_BRANCH}")
print(f"config      {CONFIG}")
print(f"data mode   {DATA_MODE}")
print(f"letters     {len(LETTERS_SHARDS)} shard(s)")
print(f"pixel       {PIXEL_SHARDS} shard(s)")

# %% [markdown]
# ## 1 · Secrets and clone
#
# Secrets are read from the environment. molab's Secrets panel exports them
# directly; if you keep a `.env` at the repo root instead, the next cell parses
# it with the project's own loader so both routes behave identically.

# %%
SECRET_NAMES = ("GH_TOKEN", "HF_TOKEN", "DRIVE_TOKEN_B64")

# A .env beside the notebook, if there is one, without clobbering a real export:
# a value already in the environment always wins, same rule as config/env.py.
_dotenv = Path(".env")
if _dotenv.is_file():
    for _line in _dotenv.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _value = _line.partition("=")
        os.environ.setdefault(_key.strip(), _value.strip().strip("\"'"))
    print(f"read {_dotenv.resolve()}")

missing = [name for name in SECRET_NAMES if not os.environ.get(name)]
assert not missing, f"add these in the Secrets panel (Ctrl/Cmd+J) or .env: {missing}"
secrets = {name: os.environ[name] for name in SECRET_NAMES}

# Lengths only. Printing a token into notebook output persists it in the .ipynb.
for _name, _value in secrets.items():
    print(f"{_name:<16} present ({len(_value)} chars)")

# %%
if (REPO_DIR / ".git").is_dir():
    sh("git", "-C", REPO, "fetch", "--depth", "1", "origin", REPO_BRANCH, check=False)
    sh("git", "-C", REPO, "reset", "--hard", f"origin/{REPO_BRANCH}", check=False)
    print("already cloned - fast-forwarded to origin/" + REPO_BRANCH)
else:
    # The PAT goes into ~/.git-credentials, never onto argv, where `ps` and every
    # subprocess error message would carry it.
    _cred = Path.home() / ".git-credentials"
    _cred.write_text(
        f"https://x-access-token:{secrets['GH_TOKEN']}@github.com\n", encoding="utf-8"
    )
    _cred.chmod(0o600)
    sh("git", "config", "--global", "credential.helper", "store")
    sh("git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, REPO)

git_head = sh(
    "git", "-C", REPO, "rev-parse", "--short", "HEAD", capture=True
).stdout.strip()
git_subject = sh(
    "git", "-C", REPO, "log", "-1", "--pretty=%s", capture=True
).stdout.strip()
print(f"commit      {git_head}  {git_subject}")

# %% [markdown]
# ### Drive credentials
#
# `DRIVE_TOKEN_B64` is the base64 of an OAuth token minted once on a machine
# with a browser (`scripts/mint_drive_token.py`). It is decoded to
# `secrets/drive_token.json`, which is where `fa_ocr_lora.drive` looks by
# default — no browser is ever opened here, and `drive.interactive_auth: false`
# in the profile makes a missing token fail in the first second with
# instructions rather than hang forever on a consent page nobody can answer.

# %%
secrets_dir = REPO_DIR / "secrets"
secrets_dir.mkdir(parents=True, exist_ok=True)
token_path = secrets_dir / "drive_token.json"

try:
    _decoded = base64.b64decode(secrets["DRIVE_TOKEN_B64"], validate=True)
except (ValueError, TypeError) as error:
    raise SystemExit(
        f"DRIVE_TOKEN_B64 is not valid base64 ({error}). Re-mint it with "
        "`python scripts/mint_drive_token.py` and copy the whole line it prints."
    ) from error

# Fail here, on a readable message, rather than three cells later inside the
# Drive client on a KeyError for 'refresh_token'.
try:
    _payload = json.loads(_decoded)
except json.JSONDecodeError as error:
    raise SystemExit(
        f"DRIVE_TOKEN_B64 decodes to something that is not JSON ({error})."
    ) from error
_required = {"client_id", "client_secret", "refresh_token"}
_absent = _required - set(_payload)
assert not _absent, f"drive token is missing {sorted(_absent)} - re-mint it"

token_path.write_bytes(_decoded)
token_path.chmod(0o600)
print(f"wrote {token_path} ({token_path.stat().st_size} bytes)")
print(f"token account: {_payload.get('account') or 'unnamed'}")

if DRIVE_FOLDER_ID:
    # Read by training/config.py as an override of drive.folder_id, so the
    # profile stays machine-independent.
    os.environ["FA_OCR_TRAIN_DRIVE_FOLDER_ID"] = DRIVE_FOLDER_ID
    print(f"drive folder id override: {DRIVE_FOLDER_ID}")

# %% [markdown]
# ## 2 · Environment
#
# `sm_120` has no kernels in the cu124/cu126 wheels, so torch comes from the
# **cu128** index before anything else is installed. PaddleOCR-VL-1.6 needs
# **transformers 5** — it has no 4.x path at all.
#
# The molab image also ships TensorFlow with Keras 3, which transformers refuses
# to import `Trainer` past. `scripts/train_molab.sh tf` reconciles that by
# installing the matching `tf-keras`, or failing that by setting `USE_TF=0`, and
# only returns once `from transformers import Trainer` actually works.

# %%
import importlib.metadata as md  # noqa: E402
import importlib.util  # noqa: E402

if importlib.util.find_spec("torch") is None or "cu128" not in md.version("torch"):
    sh(
        sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    )

# Re-read in a fresh interpreter: this one still holds pre-install metadata.
torch_version = sh(
    sys.executable,
    "-c",
    "import torch; print(torch.__version__, '|', torch.version.cuda)",
    capture=True,
).stdout.strip()
print("torch", torch_version)
assert "cu128" in torch_version, f"Blackwell needs the cu128 build, got {torch_version}"

# %%
sh(
    sys.executable, "-m", "pip", "install", "-e", ".[local-train,drive,hf,hub]",
    "datasets", "sentencepiece",
    cwd=REPO_DIR,
)

versions = sh(
    sys.executable,
    "-c",
    "import importlib.metadata as m;"
    "print(' '.join(f'{p}={m.version(p)}' for p in"
    " ('transformers','peft','accelerate','datasets')))",
    capture=True,
).stdout.strip()
print("installed  ", versions)

# 1.6 loads natively through AutoModelForImageTextToText, which only exists on
# transformers 5. On 4.x the profile's trust_remote_code: false would send the
# loader looking for a class that is not there.
_transformers = next(p for p in versions.split() if p.startswith("transformers="))
assert int(_transformers.split("=")[1].split(".")[0]) >= 5, (
    f"PaddleOCR-VL-1.6 requires transformers >= 5, got {_transformers}"
)

compat = sh("bash", "scripts/train_molab.sh", "tf", cwd=REPO_DIR, capture=True, check=False)
print(compat.stdout, end="")
if compat.returncode:
    print(compat.stderr, end="")
    raise SystemExit("tensorflow/keras could not be reconciled - see above")

# %%
import torch  # noqa: E402

assert torch.cuda.is_available(), "no CUDA device - attach the GPU in the specs button"
capability = torch.cuda.get_device_capability(0)
free, total = torch.cuda.mem_get_info()
print(f"gpu         {torch.cuda.get_device_name(0)} sm_{capability[0]}{capability[1]}")
print(f"memory      {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free")
assert torch.cuda.is_bf16_supported(), "this profile trains in bf16"
print("bf16        yes")

# %% [markdown]
# ## 3 · Corpora
#
# Two datasets, both reshaped into the layout `training/sources.py` reads:
#
# | What | HF repo | Local target |
# | --- | --- | --- |
# | letters text + regions | `shahriarhd/persian-official-letters-synth` | `data/dataset/*.jsonl` |
# | letters pixels | same repo, `data/images/*.parquet` | `data/dataset/images/*.png` |
# | Persian Pixel | `Omarrran/Persian_Pixel` (`page`) | `data/persian_pixel/` |
#
# The metadata is small and lands immediately. The **pixels exist only inside
# the parquet shards** — ~2 GB each, fetched one at a time and deleted as soon
# as their PNGs are extracted — so that is the expensive half. A row whose image
# is missing is skipped by the manifest rather than failing the run, so a
# partial download still trains.

# %%
from huggingface_hub import hf_hub_download  # noqa: E402

sh("df", "-h", str(REPO_DIR.parent), check=False)

LETTERS_REPO = "shahriarhd/persian-official-letters-synth"
letters_root = REPO_DIR / "data" / "dataset"
letters_root.mkdir(parents=True, exist_ok=True)
letters_manifest = letters_root / "paddle_ocr_vl.jsonl"

for _name in ("manifest.csv", "paddle_farmer.jsonl", "paddle_ocr_vl.jsonl"):
    _local = hf_hub_download(
        LETTERS_REPO, _name, repo_type="dataset", local_dir=letters_root
    )
    print(f"{_name:24} {Path(_local).stat().st_size / 1e6:7.1f} MB")

print(
    f"{sum(1 for line in letters_manifest.open(encoding='utf-8') if line.strip())}"
    " pages of text and labels ready"
)

# %%
import pyarrow.parquet as pq  # noqa: E402

letters_images = letters_root / "images"
letters_images.mkdir(parents=True, exist_ok=True)

# Size > 0: a shard that died mid-write must be fetched again.
have = {f"images/{p.name}" for p in letters_images.glob("*.png") if p.stat().st_size}
print(f"{len(have)} images already local; {len(LETTERS_SHARDS)} shard(s) to visit")

for _shard in LETTERS_SHARDS:
    _local = hf_hub_download(
        LETTERS_REPO, f"data/images/{_shard}.parquet", repo_type="dataset"
    )
    for _batch in pq.ParquetFile(_local).iter_batches(batch_size=8, columns=["image"]):
        for _row in _batch.to_pylist():
            _relative = _row["image"]["path"]
            if _relative in have:
                continue
            _destination = letters_root / _relative
            _destination.parent.mkdir(parents=True, exist_ok=True)
            _destination.write_bytes(_row["image"]["bytes"])
            have.add(_relative)
    os.unlink(_local)  # ~2 GB a shard; the session disk cannot hold them all
    print(f"{_shard}: {len(have)} images total")

print(f"letters images: {len(have)} in {letters_images}")

# %%
import csv  # noqa: E402

PIXEL_REPO = "Omarrran/Persian_Pixel"
pixel_root = REPO_DIR / "data" / "persian_pixel"
pixel_root.mkdir(parents=True, exist_ok=True)
pixel_csv = pixel_root / "all_features_metadata.csv"
PIXEL_FIELDS = [
    "image_filename", "text", "source_parquet", "image_path", "width", "height",
]

pixel_rows: dict[str, dict[str, object]] = {}
if pixel_csv.is_file():
    with pixel_csv.open(encoding="utf-8-sig", newline="") as handle:
        for _row in csv.DictReader(handle):
            pixel_rows[_row["image_filename"]] = _row
print(f"{len(pixel_rows)} pixel rows already local; {PIXEL_SHARDS} shard(s) to visit")

for _index in range(PIXEL_SHARDS):
    _name = f"page-{_index:05d}.parquet"
    _local = hf_hub_download(
        PIXEL_REPO, f"data/{_name}", repo_type="dataset", token=secrets["HF_TOKEN"]
    )
    _columns = ["image", "image_path", "text", "width", "height"]
    for _batch in pq.ParquetFile(_local).iter_batches(batch_size=8, columns=_columns):
        for _row in _batch.to_pylist():
            _relative = _row["image_path"]
            _filename = _relative.rsplit("/", 1)[-1]
            if _filename in pixel_rows:
                continue
            _destination = pixel_root / _relative
            _destination.parent.mkdir(parents=True, exist_ok=True)
            _destination.write_bytes(_row["image"]["bytes"])
            pixel_rows[_filename] = {
                "image_filename": _filename,
                "text": _row["text"],
                "source_parquet": _name,
                "image_path": _relative,
                "width": int(_row["width"]),
                "height": int(_row["height"]),
            }
    os.unlink(_local)
    # Rewritten every shard, so an interrupted session keeps what it finished.
    with pixel_csv.open("w", encoding="utf-8", newline="") as handle:
        _writer = csv.DictWriter(handle, fieldnames=PIXEL_FIELDS)
        _writer.writeheader()
        for _key in sorted(pixel_rows):
            _writer.writerow(pixel_rows[_key])
    print(f"{_name}: {len(pixel_rows)} rows total")

print(f"pixel: {len(pixel_rows)} rows, {pixel_csv}")

# %% [markdown]
# ### Audit
#
# What is actually on disk, against the ceilings in the profile. A `tiny` or
# `subset` run is *expected* to fall short — the manifest draws from what
# exists.

# %%
import yaml  # noqa: E402

config_path = REPO_DIR / CONFIG
assert config_path.is_file(), f"{config_path} is missing from the clone"
profile = yaml.safe_load(config_path.read_text(encoding="utf-8"))
run_dir = REPO_DIR / profile["paths"]["output_dir"]

# Count what can be trained on, not what is described. The JSONL lists all
# 26,848 pages whether or not the PNG is here, and a CSV row's text can contain
# newlines - counting physical lines once reported 3,138 pixel pages for 531.
letters_pages = 0
for _line in letters_manifest.open(encoding="utf-8"):
    if not _line.strip():
        continue
    _image = json.loads(_line).get("image", "")
    if _image and (letters_manifest.parent / _image).is_file():
        letters_pages += 1
with pixel_csv.open(encoding="utf-8-sig", newline="") as _handle:
    pixel_pages = sum(1 for _ in csv.DictReader(_handle))

_optim = profile["optim"]
print(f"device        sm_{capability[0]}{capability[1]}")
print(f"config        {config_path.name} @ {git_head}")
print(f"base model    {profile['model']['model_id']} @ {profile['model']['revision'][:12]}")
print(f"loader        {profile['model']['auto_class']}"
      f" (trust_remote_code={profile['model']['trust_remote_code']})")
print(f"letters pages {letters_pages:>6} of {profile['data']['letters_pages']} ({DATA_MODE})")
print(f"pixel pages   {pixel_pages:>6} of {profile['data']['pixel_pages']} ({DATA_MODE})")
print(f"output_dir    {profile['paths']['output_dir']}")
print(f"batch         {_optim['per_device_batch_size']}"
      f" x {_optim['gradient_accumulation_steps']}"
      f" = {_optim['per_device_batch_size'] * _optim['gradient_accumulation_steps']}")
sh("du", "-sh", str(letters_images), str(pixel_root), check=False)

# %% [markdown]
# ## 4 · Checkpoints to Google Drive
#
# molab wipes everything over 1 GB and all of `data/` when a session ends, so
# the Drive mirror is not a backup — it is the only reason a run survives a
# session change. The profile already has it on:
#
# ```yaml
# drive:
#   enabled: true
#   folder: fa-ocr-lora
#   resume_from_drive: true    # empty local disk -> fetch the newest remote ckpt
#   keep_remote: 5             # retention, so Drive does not fill up
#   upload_optimizer_state: true
# ```
#
# The uploader runs on background threads during training, so a checkpoint is in
# Drive before the next one is written. This cell proves the credentials work
# *now*, rather than letting `train` discover it an hour in.

# %%
stage_env = {**os.environ, "CONFIG": CONFIG}

_drive = profile["drive"]
print(f"enabled           {_drive['enabled']}")
print(f"folder            {_drive['folder']}")
print(f"folder_id         {os.environ.get('FA_OCR_TRAIN_DRIVE_FOLDER_ID') or _drive['folder_id'] or '(resolve by name)'}")
print(f"run_name          {_drive['run_name'] or Path(profile['paths']['output_dir']).name}")
print(f"resume_from_drive {_drive['resume_from_drive']}")
print(f"keep_remote       {_drive['keep_remote']}")
print()

if not _drive["enabled"]:
    print("WARNING: drive.enabled is false - checkpoints die with the session")

# `check` authenticates, resolves the remote folder and reports the newest
# remote checkpoint. It is the cheapest possible proof the token works.
_probe = sh("bash", "scripts/train_molab.sh", "check", cwd=REPO_DIR, env=stage_env, check=False)
if _probe.returncode:
    raise SystemExit(
        "`check` failed - fix this before training. A Drive error here is almost "
        "always an expired DRIVE_TOKEN_B64 (re-mint it) or a folder_id the "
        "drive.file scope cannot see (set DRIVE_FOLDER_ID in the config cell)."
    )

# %% [markdown]
# ## 5 · Stages
#
# `scripts/train_molab.sh` is the stage driver — it exports `PYTHONPATH`,
# `HF_HOME` and `PYTORCH_CUDA_ALLOC_CONF`, honours `CONFIG`, and reconciles
# TensorFlow before any stage runs.
#
# | Stage | What it settles |
# | --- | --- |
# | `manifest` | merge both corpora, pin the holdout, oversample digit-dense records |
# | `smoke` | 2 steps — does a batch survive the forward and backward pass |
# | `probe` | 30 steps — measured s/step and peak VRAM, not projected |
# | `baseline` | **base 1.6, no adapter**: the number training has to beat |
#
# `baseline` is the half of the comparison that people skip and then cannot
# interpret their result. Run it.

# %%
for _stage in ("manifest", "smoke", "probe", "baseline"):
    _extra = MANIFEST_ARGS if _stage == "manifest" else ()
    _result = sh(
        "bash", "scripts/train_molab.sh", _stage, *_extra,
        cwd=REPO_DIR, env=stage_env, check=False,
    )
    if _result.returncode:
        raise SystemExit(
            f"stage '{_stage}' exited {_result.returncode} - fix it before continuing"
        )
print("\nall preflight stages passed")

# %% [markdown]
# ### Train
#
# `train` is `nohup`'d by the driver: this cell returns in about a second while
# the run continues in the background, surviving a dropped browser tab. It is
# resumable — `resume_from_checkpoint` picks up a local checkpoint, and failing
# that `resume_from_drive` fetches the newest complete remote one.

# %%
sh("bash", "scripts/train_molab.sh", "train", cwd=REPO_DIR, env=stage_env, check=False)

# %% [markdown]
# ### Monitor
#
# Prints the console tail and GPU state once a minute, which is also what keeps
# the kernel from being reaped as idle. It stops as soon as the run is finished
# rather than looping out its full budget.
#
# Deciding "finished" takes three signals, because none of them is sufficient
# alone: the pid can outlive the run as a zombie in a container that does not
# reap (which is what made an earlier version of this notebook poll a completed
# job for eleven minutes), `training_summary.json` is written only after the
# Drive drain completes, and a run that finished before this cell was pressed
# leaves neither a live pid nor a fresh summary.

# %%
sys.path.insert(0, str(REPO_DIR / "src"))
from fa_ocr_lora.utils.process import is_running  # noqa: E402

console = run_dir / "console.log"
pid_file = run_dir / "train.pid"
adapter_dir = run_dir / "adapter_only"
summary = run_dir / "training_summary.json"

POLL_SECONDS = 60
MAX_POLLS = 12 * 60  # a full 12 h session; the loop exits on completion anyway

print("pid:    ", pid_file.read_text().strip() if pid_file.exists() else "not started")
print("console:", console)
print()

_seen = 0
_summary_start = summary.stat().st_mtime_ns if summary.exists() else None
finished_cleanly = False

for _poll in range(MAX_POLLS):
    if console.exists():
        _lines = console.read_text(errors="replace").splitlines()
        _new = _lines[_seen:]
        if len(_new) > 30:
            print(f"... ({len(_new) - 30} earlier lines skipped)")
            _new = _new[-30:]
        for _line in _new:
            print(_line)
        _seen = len(_lines)

    _gpu = sh(
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader",
        check=False, capture=True,
    )
    print(time.strftime("%H:%M:%S"), "gpu", (_gpu.stdout or "").strip())

    if pid_file.exists():
        try:
            # is_running, not `kill -0`: a zombie answers kill -0 long after it
            # stopped, which is the bug this replaces.
            _alive = is_running(int(pid_file.read_text().strip()))
        except (ValueError, PermissionError):
            _alive = True  # unreadable pid file: assume the run is going
    else:
        _alive = False

    _fresh_summary = summary.exists() and (
        _summary_start is None or summary.stat().st_mtime_ns != _summary_start
    )
    if not _alive or _fresh_summary:
        break
    time.sleep(POLL_SECONDS)

_console_text = console.read_text(errors="replace") if console.exists() else ""
_crashed = "Traceback" in _console_text or "training failed" in _console_text
_has_adapter = adapter_dir.is_dir() and any(adapter_dir.iterdir())
finished_cleanly = _has_adapter and not _crashed

print()
if not _has_adapter and not console.exists():
    print("train was never seen running - run the train cell above first")
elif _crashed:
    print("train hit a traceback - read the tail above, fix, re-run the train cell")
elif not _has_adapter:
    print("train stopped without writing an adapter - re-run the train cell to resume")
else:
    print(f"train finished cleanly - adapter at {adapter_dir}")

# %% [markdown]
# ## 6 · Results — base 1.6 against base 1.6 + LoRA
#
# Two stages, answering two different questions:
#
# * **`eval`** scores the trained adapter on the validation holdout, with the
#   sensitive-field audit (شماره ثبت / تاریخ / شناسه ملی / مبلغ).
# * **`compare`** scores the **base and the adapter over identical records**,
#   same split, same greedy decoding, then breaks the result down *per dataset*
#   — so a gain on administrative letters cannot hide a regression on Persian
#   Pixel behind one pooled CER.
#
# ### Reading the numbers
#
# Three columns matter and they are not interchangeable:
#
# | Metric | What it means |
# | --- | --- |
# | `CER` | strict. Counts ي-for-ی, ك-for-ک and ZWNJ-for-space as errors. |
# | `folded CER` | the same records after `fold_persian` unifies glyphs that are **identical on the page**. |
# | `convention` | the share of the strict error that was only orthographic convention. |
#
# On the first 1.6 baseline, 207 of the top 20 confusions were exactly those
# variants, so a strict CER of 0.4179 was largely reporting which Unicode
# convention the corpus was typed in. An adapter that improves strict CER while
# leaving folded CER flat has learned the corpus's *spelling*, which is worth
# having but is not better reading — the report says so explicitly.
#
# `digit CER` can exceed 1.0 and that is not a bug: the denominator is the
# reference's digit count, so a model inventing numbers scores above 100%.
# `digit runs` is the bounded companion — of the numbers actually on the page,
# how many came back exactly right.

# %%
# `eval` overwrites eval_validation.json, and the file sitting there right now
# is the baseline. Preserve it before it is lost.
_eval_json = run_dir / "eval_validation.json"
_base_backup = run_dir / "eval_validation_base.json"
if _eval_json.is_file() and not _base_backup.is_file():
    _base_backup.write_bytes(_eval_json.read_bytes())
    print(f"baseline metrics preserved at {_base_backup.name}\n")

if not finished_cleanly:
    print("no adapter to score - finish a training run first")
else:
    sh("bash", "scripts/train_molab.sh", "eval", cwd=REPO_DIR, env=stage_env, check=False)
    sh(
        "bash", "scripts/train_molab.sh", "compare", "--split", COMPARE_SPLIT,
        cwd=REPO_DIR, env=stage_env, check=False,
    )

# %%
# The headline table, read back out of the JSON the two stages wrote so it can
# be re-printed without re-running a GPU job.


def _fmt(value: object, places: int = 4) -> str:
    """Render a metric, or a dash when the stage did not produce it."""
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:.{places}f}" if places else f"{int(value):d}"


def _delta(before: object, after: object, places: int = 4) -> str:
    """Signed change, a dash unless both sides are numbers."""
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return f"{after - before:+.{places}f}" if places else f"{int(after - before):+d}"
    return "-"


compare_path = run_dir / f"compare_{COMPARE_SPLIT}.json"
if not compare_path.is_file():
    print(f"{compare_path.name} not written yet - run the cell above")
else:
    report = json.loads(compare_path.read_text())
    overall = report.get("overall", {})
    base, adapter = overall.get("base", {}), overall.get("adapter", {})

    print(f"base model   {profile['model']['model_id']}")
    print(f"adapter      {report.get('adapter', adapter_dir)}")
    print(f"split        {COMPARE_SPLIT}  ({base.get('samples', '?')} records)")
    print()
    print(f"{'metric':<20} {'base 1.6':>10} {'+ LoRA':>10} {'delta':>10}")
    print("-" * 52)
    # places=0 marks a count rather than a rate; printing 5 empty predictions as
    # 5.0000 reads like a rate and invites comparing it to a CER.
    for _label, _key, _places in (
        ("CER (strict)", "cer", 4),
        ("CER (folded)", "cer_normalized", 4),
        ("WER (strict)", "wer", 4),
        ("WER (folded)", "wer_normalized", 4),
        ("digit CER", "digit_cer", 4),
        ("digit exact", "digit_exact", 4),
        ("digit run acc", "digit_run_accuracy", 4),
        ("convention share", "convention_share", 4),
        ("digit runs", "digit_runs", 0),
        ("empty predictions", "empty_predictions", 0),
    ):
        _b, _a = base.get(_key), adapter.get(_key)
        print(
            f"{_label:<20} {_fmt(_b, _places):>10} {_fmt(_a, _places):>10}"
            f" {_delta(_b, _a, _places):>10}"
        )

    print()
    print(f"{'dataset':<18} {'n':>5} {'base CER':>10} {'+LoRA CER':>10} {'delta':>10}")
    print("-" * 56)
    better = worse = unchanged = 0
    for _row in report.get("comparison", []):
        _b, _a = _row.get("base"), _row.get("adapter")
        _n = (_row.get("samples") or {}).get("adapter", "?")
        if not _b or not _a:
            print(f"{_row.get('dataset', '?'):<18} {_n!s:>5} only one variant scored")
            continue
        _change = _a["cer"] - _b["cer"]
        better += _change < 0
        worse += _change > 0
        unchanged += _change == 0
        print(
            f"{_row.get('dataset', '?'):<18} {_n!s:>5}"
            f" {_b['cer']:>10.4f} {_a['cer']:>10.4f} {_change:>+10.4f}"
        )

    print()
    _scored = better + worse + unchanged
    if not _scored:
        print("verdict: no dataset was scored by both variants")
    else:
        _verdict = f"verdict: adapter better on {better} of {_scored} dataset(s)"
        if unchanged:
            _verdict += f", unchanged on {unchanged}"
        print(_verdict)

    # The distinction a single CER cannot make.
    _strict = _delta(base.get("cer"), adapter.get("cer"))
    _folded = _delta(base.get("cer_normalized"), adapter.get("cer_normalized"))
    print(f"         strict {_strict}, folded {_folded}")
    if isinstance(base.get("cer_normalized"), float) and isinstance(
        adapter.get("cer_normalized"), float
    ):
        if adapter["cer_normalized"] < base["cer_normalized"]:
            print("         folded CER improved too - reading genuinely got better")
        elif adapter["cer"] < base["cer"]:
            print("         folded CER did not improve - the gain is orthographic")


# %% [markdown]
# ### Checkpoints in Drive
#
# What actually survives the session. `adapter_only` is the artefact to keep —
# a few MB of LoRA weights that load on top of stock `PaddleOCR-VL-1.6`.

# %%
sh("bash", "scripts/train_molab.sh", "status", cwd=REPO_DIR, env=stage_env, check=False)

_summary_path = run_dir / "training_summary.json"
if _summary_path.is_file():
    _summary = json.loads(_summary_path.read_text())
    _drive_stats = _summary.get("drive") or {}
    if _drive_stats:
        print("\ndrive mirror")
        for _key in (
            "checkpoints_uploaded", "files_uploaded", "bytes_uploaded",
            "files_skipped", "failures",
        ):
            if _key in _drive_stats:
                print(f"  {_key:<22} {_drive_stats[_key]}")
        if _drive_stats.get("failures"):
            print("  WARNING: some uploads failed - checkpoints may be incomplete")

if adapter_dir.is_dir():
    print(f"\nlocal adapter {adapter_dir}")
    for _file in sorted(adapter_dir.iterdir()):
        print(f"  {_file.name:<34} {_file.stat().st_size / 1e6:8.2f} MB")

# %% [markdown]
# ## Restarting in a fresh session
#
# molab clears everything over 1 GB and all of `data/` on teardown, so a new
# session starts at cell 1. Re-run top to bottom; the corpora re-download and
# `train` picks the run back up:
#
# * `resume_from_checkpoint: true` continues from a local checkpoint if one
#   survived, otherwise
# * `resume_from_drive: true` fetches the newest complete remote checkpoint.
#
# Nothing else changes between a `tiny` pass and a `full` run — the figures in
# `data:` are ceilings, so a partial corpus is a valid one.
#
# ### Stopping early
#
# ```python
# sh("bash", "scripts/train_molab.sh", "stop", cwd=REPO_DIR, env=stage_env, check=False)
# ```
#
# Checkpoints already mirrored to Drive stay valid and the next `train` resumes
# from the newest complete one.
