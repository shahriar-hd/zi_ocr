#!/usr/bin/env bash
# Drive the local LoRA training path end to end inside the conda environment.
#
# Every stage is a separate invocation on purpose: the point of this file is that
# `probe` tells you what `train` will cost before you start it, and `baseline`
# gives the trained CER something to be compared against. Running them as one
# opaque job would throw both away.
#
# Usage:
#   scripts/train_local.sh                 # manifest, smoke, probe - the safe prefix
#   scripts/train_local.sh baseline        # CER of the untrained base model
#   scripts/train_local.sh train           # the real run, resumable
#   scripts/train_local.sh score           # CER of the trained adapter
#   scripts/train_local.sh all             # everything, in order, ~6 h
#
# Extra arguments are forwarded to the stage's CLI:
#   scripts/train_local.sh train --epochs 3 --grad-accum 8
#   scripts/train_local.sh probe --max-pixels 501760
#
# Environment:
#   CONDA_ENV   conda environment to run in            (default: zilber_rag)
#   CONFIG      training YAML                          (default: the CLI's own)
#   PROBE_STEPS steps the probe stage measures          (default: 30)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-zilber_rag}"
PROBE_STEPS="${PROBE_STEPS:-30}"
STAGE="${1:-prepare}"
shift || true

# The package is not pip-installed in the environment, so point Python at src/
# rather than requiring an editable install that would mutate the environment.
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Fragmentation, not exhaustion, is what usually ends a run on 4 GB: the peak is
# ~1.9 GiB against ~2.0 GiB free, so a few hundred MiB stranded between blocks is
# the difference between finishing and an OOM at step 900.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# model.revision is pinned to a commit hash, so there is nothing for the Hub to
# revalidate once the weights are cached - and with DNS down, huggingface_hub
# still retries a HEAD request five times per file before falling back to the
# cache, which turns a 40-second model load into minutes of stalling. Skip the
# network when the host does not resolve. OFFLINE=1 forces the cache path,
# OFFLINE=0 forces the network one (needed only for the very first download).
case "${OFFLINE:-auto}" in
    1) export HF_HUB_OFFLINE=1 ;;
    0) ;;
    *) getent hosts huggingface.co >/dev/null 2>&1 || export HF_HUB_OFFLINE=1 ;;
esac

CONFIG_ARGS=()
if [[ -n "${CONFIG:-}" ]]; then
    CONFIG_ARGS=(--config "${CONFIG}")
fi

run() {
    local module="$1"
    shift
    echo
    echo "=== ${module} ${*} ==="
    # --no-capture-output so the progress bar and the per-step loss stream live;
    # conda run buffers both to the end of the process without it, which for a
    # five-hour job means no output at all until it finishes.
    conda run -n "${CONDA_ENV}" --no-capture-output \
        python -m "fa_ocr_lora.training.${module}" "${CONFIG_ARGS[@]}" "$@"
}

stage_manifest() { run prepare_manifest "$@"; }
stage_smoke()    { run train_lora --smoke "$@"; }
stage_probe()    { run train_lora --probe "${PROBE_STEPS}" "$@"; }
stage_train()    { run train_lora "$@"; }
stage_baseline() { run eval --no-adapter "$@"; }
stage_score()    { run eval "$@"; }

case "${STAGE}" in
    manifest)  stage_manifest "$@" ;;
    smoke)     stage_smoke "$@" ;;
    probe)     stage_probe "$@" ;;
    train)     stage_train "$@" ;;
    baseline)  stage_baseline "$@" ;;
    score)     stage_score "$@" ;;
    prepare)
        # The safe prefix: nothing here writes an adapter or takes more than a
        # few minutes, and it fails on every mistake a five-hour run would.
        stage_manifest
        stage_smoke
        stage_probe
        echo
        echo "Throughput measured above. Pick the epoch count, then:"
        echo "  scripts/train_local.sh baseline   # before training, for comparison"
        echo "  scripts/train_local.sh train --epochs N"
        ;;
    all)
        stage_manifest
        stage_smoke
        stage_probe
        stage_baseline
        stage_train "$@"
        stage_score
        ;;
    *)
        echo "unknown stage: ${STAGE}" >&2
        echo "stages: manifest smoke probe baseline train score prepare all" >&2
        exit 2
        ;;
esac
