import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PaddleOCR-VL Persian LoRA — full run on molab

    This notebook takes a bare molab session to a training run with checkpoints
    mirrored to Google Drive:

    1. **Secrets & clone** — decode the Drive token, clone `zilber_ocr`.
    2. **Environment** — torch from the cu128 index (Blackwell is `sm_120`), then
       the project's own extras.
    3. **Corpora** — fetch the two HF datasets and reshape them into the local
       layout `training/sources.py` reads.
    4. **Stages** — `check → manifest → smoke → probe → baseline`, then `train`
       (nohup'd, resumable) and `eval`.

    **Before you start**

    - Attach the NVIDIA RTX PRO 6000 Blackwell in the notebook specs button.
      Nothing here runs on CPU: the environment cells and `check` assert CUDA and
      bf16.
    - Open the Secrets panel (`Ctrl`/`Cmd` + `J`) and add three variables:
      `GH_TOKEN` (repo read), `HF_TOKEN` (Persian_Pixel is a gated dataset) and
      `DRIVE_TOKEN_B64` (from `scripts/mint_drive_token.py`).
    - `DATA_MODE = "subset"` in the constants cell pulls ~15 GB so the whole
      pipeline can be proven in minutes. Set it to `"full"` before a real `train`
      stage.
    - Sessions die after 12 h, and an **idle** kernel after 90 min — keep the
      monitor cell running while the detached `train` process works.

    **How to run it**

    marimo only executes a cell when something you *are* running depends on it: a
    cell whose value nothing consumes is silently skipped. Every setup cell below
    therefore defines a name a later cell reads — `secrets`, `git_head`,
    `torch_version`, `letters_manifest`, `letters_images`, `pixel_csv`,
    `token_path` — so **press the audit cell once and the whole setup runs, in
    order**. The stage cells are actions, not setup, so you press those yourself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Secrets and clone
    """)
    return


@app.cell
def _():
    import base64
    import os
    import subprocess
    import sys
    from pathlib import Path

    REPO = "zilber_ocr"
    REPO_URL = "https://github.com/shahriar-hd/zilber_ocr.git"
    CONFIG = "configs/lora_config_rtx6000pro.yaml"

    # "subset" proves the pipeline cheaply; "full" is what a real run needs.
    DATA_MODE = "subset"
    LETTERS_TRAIN_SHARDS = 3 if DATA_MODE == "subset" else 18
    LETTERS_TEST_SHARDS = 1 if DATA_MODE == "subset" else 3
    PIXEL_SHARDS = 2 if DATA_MODE == "subset" else 56

    REPO_DIR = Path(REPO).resolve()
    os.environ.setdefault("HF_HOME", str(REPO_DIR / ".cache" / "huggingface"))


    def sh(*args: str, cwd=None, env=None, check=True, capture=False) -> subprocess.CompletedProcess[str]:
        """Run a command, returning it rather than discarding it when asked."""
        return subprocess.run(
            args,
            cwd=cwd,
            env=env,
            check=check,
            capture_output=capture,
            text=True,
        )

    return (
        CONFIG,
        DATA_MODE,
        LETTERS_TEST_SHARDS,
        LETTERS_TRAIN_SHARDS,
        PIXEL_SHARDS,
        Path,
        REPO,
        REPO_DIR,
        REPO_URL,
        base64,
        os,
        sh,
        sys,
    )


@app.cell
def _(os):
    missing = [k for k in ("GH_TOKEN", "HF_TOKEN", "DRIVE_TOKEN_B64") if not os.environ.get(k)]
    assert not missing, f"add these in the Secrets panel (Ctrl/Cmd+J): {missing}"
    secrets = {k: os.environ[k] for k in ("GH_TOKEN", "HF_TOKEN", "DRIVE_TOKEN_B64")}
    print("secrets present:", ", ".join(secrets))
    return (secrets,)


@app.cell
def _(Path, REPO, REPO_DIR, REPO_URL, secrets, sh):
    if (REPO_DIR / ".git").is_dir():
        print("already cloned")
    else:
        # The PAT goes into ~/.git-credentials, never onto argv where `ps` and any
        # failure message would show it.
        cred = Path.home() / ".git-credentials"
        cred.write_text(f"https://x-access-token:{secrets['GH_TOKEN']}@github.com\n", encoding="utf-8")
        cred.chmod(0o600)
        sh("git", "config", "--global", "credential.helper", "store")
        sh("git", "clone", "--depth", "1", REPO_URL, REPO)
    git_head = sh("git", "-C", REPO, "rev-parse", "--short", "HEAD", capture=True).stdout.strip()
    print("commit:", git_head)
    return (git_head,)


@app.cell
def _(REPO_DIR, base64, secrets):
    secrets_dir = REPO_DIR / "secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    token_path = secrets_dir / "drive_token.json"
    token_path.write_bytes(base64.b64decode(secrets["DRIVE_TOKEN_B64"]))
    token_path.chmod(0o600)
    print("wrote", token_path, f"({token_path.stat().st_size} bytes)")
    return (token_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Environment

    `sm_120` has no kernels in the cu124/cu126 builds, so torch comes from the
    **cu128** index before anything else is installed. Skip this cell only if
    `torch.__version__` already ends in `+cu128`.

    The image also ships TensorFlow with Keras 3, which transformers 4.57 refuses
    ("please install tf-keras") and cannot import `Trainer` past. The project-extras
    cell below therefore ends with `scripts/train_molab.sh tf`, which gives
    transformers a matching `tf-keras` - or, if that cannot be installed, sets
    `USE_TF=0` - then proves `from transformers import Trainer` imports. TensorFlow
    is never uninstalled, and `FA_OCR_TF=torch` skips the install altogether.
    """)
    return


@app.cell
def _(sh, sys):
    import importlib.metadata as md
    import importlib.util

    if importlib.util.find_spec("torch") is None or "cu128" not in md.version("torch"):
        sh(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        )
    # Re-read in a fresh interpreter: the running one still holds pre-install metadata.
    torch_version = sh(
        sys.executable,
        "-c",
        "import torch; print(torch.__version__, '| cuda', torch.version.cuda)",
        capture=True,
    ).stdout.strip()
    print("torch", torch_version)
    assert "cu128" in torch_version.split("|")[0], f"Blackwell needs the cu128 build, got {torch_version}"

    return (torch_version,)


@app.cell
def _(REPO_DIR, sh, sys, torch_version):
    sh(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        ".[local-train,drive,hf,hub]",
        "datasets",
        cwd=REPO_DIR,
    )
    datasets_version = sh(
        sys.executable, "-c", "import importlib.metadata as m; print(m.version('datasets'))", capture=True
    ).stdout.strip()
    print("project editable in", REPO_DIR)
    print("python", sys.version.split()[0], "| torch", torch_version.split("|")[0].strip(), "| datasets", datasets_version)

    # TensorFlow + Keras 3, reconciled by the driver rather than here: `tf` installs
    # the tf-keras matching the installed TensorFlow, or falls back to USE_TF=0, and
    # only returns once `from transformers import Trainer` actually imports. It needs
    # no GPU, so this cell stays runnable on a bare session.
    compat = sh("bash", "scripts/train_molab.sh", "tf", cwd=REPO_DIR, capture=True, check=False)
    print(compat.stdout, end="")
    if compat.returncode:
        print(compat.stderr, end="")
        raise SystemExit("tensorflow/keras could not be reconciled - see the output above")
    return (datasets_version,)


@app.cell
def _(datasets_version, torch_version):
    import torch

    assert torch.cuda.is_available(), "no CUDA device - attach the RTX PRO 6000 in the specs button"
    capability = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info()
    print(f"torch       {torch_version.split('|')[0].strip()}")
    print(f"datasets    {datasets_version}")
    print(f"gpu         {torch.cuda.get_device_name(0)} sm_{capability[0]}{capability[1]}, {total / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free")
    assert torch.cuda.is_bf16_supported(), "this profile requires bf16"
    print("bf16        yes")
    return capability, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Corpora

    Two datasets, both written into the layout `training/sources.py` expects:

    | What | HF repo | Local target |
    | --- | --- | --- |
    | letters text + regions | `shahriarhd/persian-official-letters-synth` root | `data/dataset/{manifest.csv,paddle_farmer.jsonl,paddle_ocr_vl.jsonl}` |
    | letters pixels | same repo, `data/images/*.parquet` | `data/dataset/images/*.png` |
    | Persian Pixel | `Omarrran/Persian_Pixel` (`page`) | `data/persian_pixel/all_features_metadata.csv` + `images/` |

    The letters metadata files are small (~160 MB) and land almost immediately.
    The **pixels exist only inside `data/images/*.parquet`** — ~2 GB a shard, one
    shard at a time, each deleted as soon as its PNGs are extracted — so that cell
    is the expensive half and the only one the JSONL alone cannot replace.

    Rows whose image is missing are skipped by the manifest rather than failing,
    so a partial download still runs.
    """)
    return


@app.cell
def _(Path, REPO_DIR, git_head, sh):
    from huggingface_hub import hf_hub_download
    # A subset pass needs ~15 GB: torch + the base model + these images.
    sh('df', '-sh', str(REPO_DIR.parent), check=False)
    LETTERS_REPO = 'shahriarhd/persian-official-letters-synth'
    letters_root = REPO_DIR / 'data' / 'dataset'
    letters_root.mkdir(parents=True, exist_ok=True)
    letters_manifest = letters_root / 'paddle_ocr_vl.jsonl'
    for _name in ('manifest.csv', 'paddle_farmer.jsonl', 'paddle_ocr_vl.jsonl'):
        _local = hf_hub_download(LETTERS_REPO, _name, repo_type='dataset', local_dir=letters_root)
        print(f'{_name:24} {Path(_local).stat().st_size / 1000000.0:7.1f} MB')
    pages = sum(1 for line in letters_manifest.open(encoding='utf-8') if line.strip())
    print(f'{pages} pages of text and labels ready (commit {git_head})')
    return LETTERS_REPO, hf_hub_download, letters_manifest, letters_root


@app.cell
def _(
    LETTERS_REPO,
    LETTERS_TEST_SHARDS,
    LETTERS_TRAIN_SHARDS,
    hf_hub_download,
    letters_root,
    os,
):
    import pyarrow.parquet as pq
    letters_images = letters_root / 'images'
    letters_images.mkdir(parents=True, exist_ok=True)
    shards = [f'train-{i:05d}-of-00018' for i in range(LETTERS_TRAIN_SHARDS)]
    shards += [f'test-{i:05d}-of-00003' for i in range(LETTERS_TEST_SHARDS)]
    # Size > 0: a shard that died mid-write must be fetched again.
    have = {f'images/{p.name}' for p in letters_images.glob('*.png') if p.stat().st_size > 0}
    print(f'{len(have)} images already local; {len(shards)} shard(s) to visit')
    added = 0
    for shard in shards:
        _local = hf_hub_download(LETTERS_REPO, f'data/images/{shard}.parquet', repo_type='dataset')
        _parquet = pq.ParquetFile(_local)
        for _batch in _parquet.iter_batches(batch_size=8, columns=['image']):
            for _row in _batch.to_pylist():
                _relative = _row['image']['path']
                if _relative in have:
                    continue
                _destination = letters_root / _relative
                _destination.parent.mkdir(parents=True, exist_ok=True)
                _destination.write_bytes(_row['image']['bytes'])
                have.add(_relative)
                added += 1
        os.unlink(_local)
        print(f'{shard}: +{added} images so far ({len(have)} total)')
    print(f'letters images: {len(have)} in {letters_images}')
    return letters_images, pq


@app.cell
def _(PIXEL_SHARDS, REPO_DIR, hf_hub_download, os, pq, secrets):
    PIXEL_REPO = 'Omarrran/Persian_Pixel'
    pixel_root = REPO_DIR / 'data' / 'persian_pixel'
    pixel_root.mkdir(parents=True, exist_ok=True)
    pixel_csv = pixel_root / 'all_features_metadata.csv'
    FIELDS = ['image_filename', 'text', 'source_parquet', 'image_path', 'width', 'height']
    import csv
    pixel_rows = {}
    if pixel_csv.is_file():
        with pixel_csv.open(encoding='utf-8-sig', newline='') as handle:
            for _row in csv.DictReader(handle):
                pixel_rows[_row['image_filename']] = _row
    print(f'{len(pixel_rows)} pixel rows already local; {PIXEL_SHARDS} shard(s) to visit')
    for index in range(PIXEL_SHARDS):
        _name = f'page-{index:05d}.parquet'
        _local = hf_hub_download(PIXEL_REPO, f'data/{_name}', repo_type='dataset', token=secrets['HF_TOKEN'])
        _parquet = pq.ParquetFile(_local)
        columns = ['image', 'image_path', 'text', 'width', 'height']
        for _batch in _parquet.iter_batches(batch_size=8, columns=columns):
            for _row in _batch.to_pylist():
                _relative = _row['image_path']
                filename = _relative.rsplit('/', 1)[-1]
                if filename in pixel_rows:
                    continue
                _destination = pixel_root / _relative
                _destination.parent.mkdir(parents=True, exist_ok=True)
                _destination.write_bytes(_row['image']['bytes'])
                pixel_rows[filename] = {'image_filename': filename, 'text': _row['text'], 'source_parquet': _name, 'image_path': _relative, 'width': int(_row['width']), 'height': int(_row['height'])}
        os.unlink(_local)
        # Rewritten every shard, so an interrupted session keeps what it finished.
        with pixel_csv.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for key in sorted(pixel_rows):
                writer.writerow(pixel_rows[key])
        print(f'{_name}: {len(pixel_rows)} rows total')
    print(f'pixel: {len(pixel_rows)} rows, {pixel_csv}')
    return pixel_csv, pixel_root


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Audit

    This is the cell to press in a fresh session: it pulls in every setup cell
    above it, then compares what is on disk against the ceilings in the config. A
    subset run is expected to fall short of `26848` / `31000`; the manifest simply
    draws from what exists.
    """)
    return


@app.cell
def _(
    CONFIG,
    DATA_MODE,
    REPO_DIR,
    capability,
    git_head,
    letters_images,
    letters_manifest,
    pixel_csv,
    pixel_root,
    sh,
    torch,
):
    import yaml

    config_path = REPO_DIR / CONFIG
    assert config_path.is_file(), (
        f"{config_path} is missing from the clone - commit and push it, "
        "or upload it through the file browser"
    )
    profile = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    letters_pages = sum(1 for line in letters_manifest.open(encoding="utf-8") if line.strip())
    pixel_pages = sum(1 for _ in pixel_csv.open(encoding="utf-8-sig")) - 1

    print(f"device       sm_{capability[0]}{capability[1]} ({torch.cuda.get_device_name(0)})")
    print(f"config       {config_path.name} @ {git_head}")
    print(f"letters pages {letters_pages:>6} of {profile['data']['letters_pages']} ({DATA_MODE})")
    print(f"pixel  pages {pixel_pages:>6} of {profile['data']['pixel_pages']} ({DATA_MODE})")
    print(f"output_dir    {profile['paths']['output_dir']}")
    print(f"batch         {profile['optim']['per_device_batch_size']} x {profile['optim']['gradient_accumulation_steps']} = "
          f"{profile['optim']['per_device_batch_size'] * profile['optim']['gradient_accumulation_steps']}")
    sh("du", "-sh", str(letters_images), str(pixel_root), check=False)
    return (profile,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Stages

    `scripts/train_molab.sh` is the stage driver — it exports `PYTHONPATH`,
    `HF_HOME` and `PYTORCH_CUDA_ALLOC_CONF` itself, honours `CONFIG`, and reconciles
    TensorFlow/Keras before any stage runs.
    Run them in order the first time; `train` alone is all a restart needs.

    Two cells below are for the run that is already going: `status` prints the
    console tail, GPU utilisation and free disk, and `stop` ends the detached
    process (checkpoints already in Drive stay valid).
    """)
    return


@app.cell
def _(CONFIG, REPO_DIR, os, sh, token_path):
    # `check` authenticates against Drive, so the token from section 1 must exist
    # before the first stage runs. Naming it in the signature is what makes marimo
    # execute that cell first - nothing in the body would otherwise pull it in.
    assert token_path.is_file(), (
        "Drive token missing - run the cell in section 1 that writes "
        "secrets/drive_token.json"
    )
    stage_env = {**os.environ, "CONFIG": CONFIG}
    for stage in ("check", "manifest", "smoke", "probe", "baseline"):
        result = sh("bash", "scripts/train_molab.sh", stage, cwd=REPO_DIR, env=stage_env, check=False)
        if result.returncode:
            raise SystemExit(f"stage '{stage}' exited {result.returncode} - fix it before continuing")
    return (stage_env,)


@app.cell
def _(REPO_DIR, sh, stage_env, token_path):
    assert token_path.is_file(), "Drive token missing - run the cell that writes secrets/drive_token.json"
    # `train` is nohup'd by the script: this cell returns in a second while the run
    # keeps going in the background, resumable after a crash or a session change.
    sh("bash", "scripts/train_molab.sh", "train", cwd=REPO_DIR, env=stage_env, check=False)
    return


@app.cell
def _(REPO_DIR, profile, sh):
    import time

    run_dir = REPO_DIR / profile["paths"]["output_dir"]
    console = run_dir / "console.log"
    pid_file = run_dir / "train.pid"
    print("pid:", pid_file.read_text().strip() if pid_file.exists() else "not started")
    print("console:", console)

    # One iteration a minute: the loop is also what keeps the kernel busy, which is
    # how the 90-minute idle shutdown is avoided while the detached run trains.
    for _ in range(60):
        if console.exists():
            for line in console.read_text(errors="replace").splitlines()[-12:]:
                print(line)
        gpu = sh(
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
            check=False,
            capture=True,
        )
        print(time.strftime("%H:%M:%S"), "gpu", (gpu.stdout or "").strip())
        time.sleep(60)
    return


@app.cell
def _(REPO_DIR, sh, stage_env):
    # One-shot version of the loop above: pid, console tail, GPU utilisation, free
    # disk. Cheap enough to press whenever you wonder if the run is still going.
    sh("bash", "scripts/train_molab.sh", "status", cwd=REPO_DIR, env=stage_env, check=False)
    return


@app.cell
def _(REPO_DIR, sh, stage_env):
    # Ends the detached run. Checkpoints already mirrored to Drive stay valid, and
    # the next `train` resumes from the newest complete one.
    sh("bash", "scripts/train_molab.sh", "stop", cwd=REPO_DIR, env=stage_env, check=False)
    return


@app.cell
def _(REPO_DIR, sh, stage_env):
    # Run this once checkpoints exist (every `save_steps: 500`). It reports trained
    # CER/WER plus the sensitive-field audit; run the `baseline` stage first to have
    # an untrained number to compare against.
    sh("bash", "scripts/train_molab.sh", "eval", cwd=REPO_DIR, env=stage_env, check=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Restarting in a fresh session

    molab clears everything over 1 GB and all of `data/` when a session ends, so a
    new session starts at cell 1. Set `DATA_MODE = "full"` to download the whole
    corpus, then press the audit cell again. `train` picks the run back up:

    - `resume_from_checkpoint: true` continues from a local checkpoint if one
      survived, otherwise
    - `resume_from_drive: true` fetches the newest complete remote checkpoint.

    Nothing else in the config changes between the subset pass and the full run —
    the ceilings in `data:` are ceilings, so a partial corpus is a valid one.
    """)
    return


if __name__ == "__main__":
    app.run()
