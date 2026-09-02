#!/usr/bin/env bash
# Drive one full LoRA run on a rented A100 (40 GB VRAM / 32 vCPU / 64 GB RAM),
# honouring every setting in configs/lora_config_a100.yaml.
#
# The stages are separate on purpose: each one is cheap and answers a question
# before the expensive stage that depends on it. Run them in order the first time,
# then use `train` alone for restarts.
#
#   ./scripts/train_a100.sh check      # env, GPU, corpora, Drive credentials
#   ./scripts/train_a100.sh manifest   # build unified_manifest.parquet (~10 min)
#   ./scripts/train_a100.sh smoke      # 2 steps, proves the loop runs
#   ./scripts/train_a100.sh probe      # 30 real steps -> measured s/step + peak VRAM
#   ./scripts/train_a100.sh baseline   # base-model CER/WER, so training has a target
#   ./scripts/train_a100.sh train      # the run itself (nohup'd, resumable)
#   ./scripts/train_a100.sh eval       # trained CER/WER + sensitive-field audit
#   ./scripts/train_a100.sh all        # check -> manifest -> smoke -> probe -> train
#
# Every stage reads the same config, so there is no second place to keep in sync.
# Overrides go through the environment rather than being edited in here:
#
#   FA_OCR_TRAIN_DRIVE_FOLDER_ID   the Drive folder to mirror into (required for a
#                                  service account; see docs/decisions.md)
#   FA_OCR_TRAIN_DRIVE_RUN_NAME    remote subfolder (default: output dir's name)
#   CONFIG                         a different profile than the A100 one
#   PY                             python interpreter (default: python)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Exported, not just set: the python snippets below read it from the environment.
export CONFIG="${CONFIG:-configs/lora_config_a100.yaml}"
PY="${PY:-python}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
# Trims allocator fragmentation at batch 8 of variable-resolution pages, where
# block sizes differ every step. Costs nothing when it is not needed.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# HF_HOME on the instance's big disk, not the 20 GB root some providers give: the
# base weights are 1.8 GB and the processor's remote code caches beside them.
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"

banner() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# Asked for lazily rather than at script startup: a broken config should surface
# through `check`'s own diagnostics, not as a traceback from this one line.
output_dir() {
  $PY - <<'EOF'
import os
import sys

sys.path.insert(0, "src")
from fa_ocr_lora.training.config import load_training_config

print(load_training_config(os.environ["CONFIG"]).paths.output_dir)
EOF
}

stage_check() {
  banner "environment"
  $PY -c "
import torch, transformers, peft
print(f'torch        {torch.__version__}')
print(f'transformers {transformers.__version__}')
print(f'peft         {peft.__version__}')
assert torch.cuda.is_available(), 'no CUDA device visible'
free, total = torch.cuda.mem_get_info()
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f'gpu          {name} sm_{cap[0]}{cap[1]}, {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free')
assert torch.cuda.is_bf16_supported(), 'bf16 unsupported; the A100 profile needs it'
print('bf16         yes')
"
  banner "config + VRAM projection"
  # Prints the same banner run_training does, and clamps nothing - a dry read.
  $PY -c "
import sys; sys.path.insert(0, 'src')
from fa_ocr_lora.training.config import load_training_config
from fa_ocr_lora.training import vram
c = load_training_config('$CONFIG')
print(vram.banner(vram.probe_vram(c.model, c.image), c.model))
print()
print('manifest  ', c.paths.manifest_path, '(exists:', c.paths.manifest_path.is_file(), ')')
print('output    ', c.paths.output_dir)
print('drive     ', 'on' if c.drive.enabled else 'off',
      '->', c.drive.folder_id or c.drive.folder, '/',
      c.drive.resolved_run_name(c.paths.output_dir))
"
  banner "drive credentials"
  # Authenticates and creates the remote run folder, so a bad key fails here in a
  # second rather than after the weights load.
  $PY -c "
import sys; sys.path.insert(0, 'src')
from fa_ocr_lora.training.config import load_training_config
from fa_ocr_lora.training.loop import build_checkpoint_sync
from fa_ocr_lora.drive import DriveError
c = load_training_config('$CONFIG')
if not c.drive.enabled:
    print('drive mirroring is off in the config; nothing to check')
    raise SystemExit(0)
sync = build_checkpoint_sync(c.drive, output_dir=c.paths.output_dir, log_path=None)
try:
    sync.start()
except DriveError as exc:
    print(f'FAILED: {exc}')
    raise SystemExit(1)
print('authenticated; remote folder ready')
print('newest remote checkpoint:', sync._latest_remote() or 'none')
sync.close(timeout=30)
"
}

stage_manifest() {
  banner "manifest"
  $PY -m fa_ocr_lora.training.prepare_manifest --config "$CONFIG" "$@"
}

stage_smoke() {
  banner "smoke (2 steps)"
  # --no-drive: a 2-step run produces nothing worth uploading, and this keeps the
  # remote folder free of throwaway checkpoints.
  $PY -m fa_ocr_lora.training.train_lora --config "$CONFIG" --smoke --no-drive "$@"
}

stage_probe() {
  banner "probe (30 steps, measured)"
  $PY -m fa_ocr_lora.training.train_lora --config "$CONFIG" --probe 30 --no-drive "$@"
}

stage_baseline() {
  banner "baseline eval (no adapter)"
  $PY -m fa_ocr_lora.training.eval --config "$CONFIG" --no-adapter --max-samples 100 "$@"
}

stage_train() {
  banner "train"
  local log_dir
  log_dir="$(output_dir)"
  mkdir -p "$log_dir"
  local console="${log_dir}/console.log"
  # nohup + setsid so a dropped SSH connection does not take the run with it.
  # The run also writes its own $LOG_DIR/logs/train.log, which is what gets
  # mirrored to Drive; this file additionally captures anything that bypasses
  # logging (tracebacks, tqdm, CUDA's own stderr).
  setsid nohup $PY -m fa_ocr_lora.training.train_lora \
    --config "$CONFIG" "$@" >>"$console" 2>&1 &
  local pid=$!
  echo "$pid" >"${log_dir}/train.pid"
  echo "started pid $pid"
  echo "  console : tail -f $console"
  echo "  run log : tail -f ${log_dir}/logs/train.log"
  echo "  stop    : kill $pid   (checkpoints already in Drive stay valid)"
  echo "  resume  : ./scripts/train_a100.sh train"
}

stage_eval() {
  banner "eval (trained adapter)"
  $PY -m fa_ocr_lora.training.eval --config "$CONFIG" "$@"
}

case "${1:-}" in
  check)    shift; stage_check ;;
  manifest) shift; stage_manifest "$@" ;;
  smoke)    shift; stage_smoke "$@" ;;
  probe)    shift; stage_probe "$@" ;;
  baseline) shift; stage_baseline "$@" ;;
  train)    shift; stage_train "$@" ;;
  eval)     shift; stage_eval "$@" ;;
  all)
    shift
    stage_check
    stage_manifest
    stage_smoke
    stage_probe
    stage_train "$@"
    ;;
  *)
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
