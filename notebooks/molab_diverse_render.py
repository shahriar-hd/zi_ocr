import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        r"""
        # Persian diverse OCR — render on molab

        The pipeline is split across two machines, and the split is not arbitrary.

        | stage | where | why |
        |---|---|---|
        | 00 lexicon, 01 plan, 02 generate, 03 text QC | **laptop** | holds the API keys, and generation is network-bound not compute-bound |
        | 04 layout, 05 render, 06 image QC, 07 package, 08 upload | **here** | needs Chromium, cores and upload bandwidth |

        The handoff is the text artifacts, mirrored to Google Drive. Nothing here
        needs the API keys, the lexicon or the font directory: a render spec embeds
        its own fonts, so this side is a pure function of `texts_clean.jsonl`.

        Everything is resumable. Each stage skips work already on disk, so a runtime
        that dies at hour three costs the in-flight samples and nothing else.
        """
    )
    return (mo,)


@app.cell
def _():
    # Secrets come from the molab panel, never from the notebook.
    #   HF_TOKEN       - to push the dataset
    #   DRIVE_TOKEN_B64- base64 of secrets/drive_token.json, to pull the text
    #   GH_TOKEN       - to clone, if the repo is private
    import os
    import subprocess
    import sys
    from pathlib import Path

    REPO = Path("/content/zilber_ocr") if Path("/content").exists() else Path.cwd()
    RUN_NAME = "fa_diverse_v1"

    print("python  ", sys.version.split()[0])
    print("repo    ", REPO)
    for name in ("HF_TOKEN", "DRIVE_TOKEN_B64", "GH_TOKEN"):
        print(f"{name:16s} {'set' if os.environ.get(name) else 'MISSING'}")
    return REPO, RUN_NAME, Path, os, subprocess, sys


@app.cell
def _(REPO, Path, os, subprocess):
    # Clone or update. A shallow clone is enough: this side needs the code, not
    # the history.
    def run(command, **kwargs):
        """Run a command, echoing it, and raise on failure."""
        print("$", " ".join(str(part) for part in command))
        result = subprocess.run(command, text=True, **kwargs)
        if result.returncode:
            raise SystemExit(f"command failed with {result.returncode}")
        return result

    if not (REPO / ".git").exists():
        token = os.environ.get("GH_TOKEN", "")
        auth = f"{token}@" if token else ""
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                f"https://{auth}github.com/anomalyco/zilber_ocr.git",
                str(REPO),
            ]
        )
    else:
        run(["git", "-C", str(REPO), "pull", "--ff-only"])
    return (run,)


@app.cell
def _(REPO, run, sys):
    # Chromium is the shaping engine, so `playwright install` is not optional.
    # The deps here are only what stages 04-08 touch; nothing CUDA.
    run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[hf,hub,drive]"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "playwright",
            "pillow",
            "numpy",
            "jinja2",
            "pyyaml",
            "fonttools",
        ]
    )
    run([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"])
    return


@app.cell
def _(REPO, Path, os, run, sys):
    # Restore the Drive token from the panel secret, so the pull below is
    # non-interactive. A molab runtime has no browser to answer a consent page.
    import base64

    token_blob = os.environ.get("DRIVE_TOKEN_B64", "")
    if token_blob:
        secrets = REPO / "secrets"
        secrets.mkdir(parents=True, exist_ok=True)
        (secrets / "drive_token.json").write_bytes(base64.b64decode(token_blob))
        print("wrote secrets/drive_token.json")
    else:
        print("DRIVE_TOKEN_B64 not set; the pull cell will fail")
    return


@app.cell
def _(REPO, RUN_NAME, Path, sys):
    # Pull the text stage from Drive. This is the handoff: everything the render
    # side needs, and nothing it does not.
    sys.path.insert(0, str(REPO / "src"))

    def pull_text_stage(repo: Path, run_name: str) -> int:
        """Download the laptop's text artifacts into the local work tree.

        Returns:
            The number of files written.
        """
        from fa_ocr_lora.drive import DriveClient, load_credentials

        client = DriveClient(load_credentials(interactive=False))
        root = client.find_folder(f"fa-diverse-ocr/{run_name}", "root")
        if root is None:
            raise SystemExit(
                f"no Drive folder fa-diverse-ocr/{run_name}; "
                "run `python -m fa_ocr_lora.diverse.backup` on the laptop first"
            )

        work = repo / "data" / "diverse"
        written = 0
        for remote, local in (
            ("lexicon", work / "lexicon"),
            ("planning", work / "planning"),
            ("textqc", work / "textqc"),
        ):
            folder = client.find_folder(remote, root)
            if folder is None:
                continue
            local.mkdir(parents=True, exist_ok=True)
            for name, (file_id, _size) in client.list_folder(folder).items():
                target = local / name
                if target.exists():
                    continue
                client.download(file_id, target)
                written += 1
                print(f"  {remote}/{name}")

        # The per-sample texts arrive as tarballs; stage 04 reads the rolled-up
        # texts_clean.jsonl, so these are only needed for auditing.
        folder = client.find_folder("generation", root)
        if folder is not None:
            local = work / "generation"
            local.mkdir(parents=True, exist_ok=True)
            for name, (file_id, _size) in client.list_folder(folder).items():
                if not name.endswith(".jsonl"):
                    continue
                target = local / name
                if target.exists():
                    continue
                client.download(file_id, target)
                written += 1
                print(f"  generation/{name}")
        return written

    print(f"pulled {pull_text_stage(REPO, RUN_NAME)} file(s)")
    return


@app.cell
def _(REPO, Path):
    # The font pool is rebuilt here rather than pulled: it is 26 MB of OFL fonts
    # that the download script fetches reproducibly, and shipping binaries
    # through Drive to save one HTTP fetch is not a trade worth making.
    import subprocess as _sp
    import sys as _sys

    fonts = REPO / "data" / "diverse" / "fonts" / "fonts.csv"
    if fonts.exists():
        print("font pool already present")
    else:
        _sp.run(
            [_sys.executable, str(REPO / "scripts" / "download_diverse_fonts.py")],
            cwd=str(REPO),
            check=True,
        )
    return


@app.cell
def _(REPO, Path):
    # How much text is waiting.
    clean = REPO / "data" / "diverse" / "textqc" / "texts_clean.jsonl"
    if not clean.exists():
        raise SystemExit("texts_clean.jsonl is missing; the Drive pull did not work")
    count = sum(1 for _ in clean.open(encoding="utf-8"))
    print(f"{count:,} cleaned text(s) ready to render")
    return


@app.cell
def _(REPO, run, sys):
    # Stage 04: choose format, template, fonts, colours. Fast and CPU-only.
    run(
        [sys.executable, "-u", "-m", "fa_ocr_lora.diverse.layouts.build"],
        cwd=str(REPO),
    )
    return


@app.cell
def _(REPO, run, sys):
    # Stage 05: render. One browser per worker; set workers to the core count.
    import os as _os

    workers = min(8, (_os.cpu_count() or 4))
    print(f"rendering with {workers} browser(s)")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "fa_ocr_lora.diverse.render.run",
            "--workers",
            str(workers),
        ],
        cwd=str(REPO),
    )
    return


@app.cell
def _(REPO, run, sys):
    # Stage 06: image QC, and the contact sheets to review by eye.
    run([sys.executable, "-u", "-m", "fa_ocr_lora.diverse.qc.run"], cwd=str(REPO))
    return


@app.cell
def _(REPO, mo, Path):
    # Look at the output before packaging it. No automatic check catches
    # "this is wrong in a way nobody thought to test for".
    sheets = sorted((REPO / "reports" / "diverse" / "contact_sheets").glob("*.png"))
    mo.vstack(
        [mo.md(f"### {path.stem}"), mo.image(str(path))] if False else mo.image(str(path))
        for path in sheets
    ) if sheets else mo.md("no contact sheets found")
    return


@app.cell
def _(REPO, run, sys):
    # Stage 07: group-aware splits and Parquet shards.
    run([sys.executable, "-u", "-m", "fa_ocr_lora.diverse.packaging.run"], cwd=str(REPO))
    return


@app.cell
def _(REPO, run, sys):
    # Stage 08: dry run first. Read the plan before spending the bandwidth.
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "fa_ocr_lora.diverse.upload.run",
            "--dry-run",
            "--repo",
            "shahriarhd/persian-diverse-ocr",
        ],
        cwd=str(REPO),
    )
    return


@app.cell
def _(REPO, run, sys):
    # The real push. Resumable: re-running skips whatever the ledger has
    # already verified, so an interrupted upload costs only the in-flight file.
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "fa_ocr_lora.diverse.upload.run",
            "--repo",
            "shahriarhd/persian-diverse-ocr",
        ],
        cwd=str(REPO),
    )
    return


@app.cell
def _(REPO, run, sys):
    # Mirror the shards back to Drive. The Hub is durable but it is not a
    # backup you control.
    run([sys.executable, "-u", "-m", "fa_ocr_lora.diverse.backup"], cwd=str(REPO))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## If something goes wrong

        **The runtime died mid-render.** Re-run the render cell. The annotation is
        written last, after the image and the crops, so its presence means a sample
        is complete; anything half-done is redone.

        **Image QC rejects more than 20%.** It exits non-zero on purpose. Read
        `data/diverse/qc/rejections.jsonl` — the reason strings name the cause, and
        every one of them has been a real bug at least once.

        **The upload stalled.** Re-run it. `upload_ledger.jsonl` records each file
        only after the Hub confirms its size, so the retry transfers only what is
        genuinely missing.

        **Chromium will not launch.** `playwright install --with-deps chromium`
        needs the system libraries; on a bare runtime run it before anything else.
        """
    )
    return


if __name__ == "__main__":
    app.run()
