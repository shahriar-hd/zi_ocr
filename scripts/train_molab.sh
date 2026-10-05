#!/usr/bin/env bash
# Drive one LoRA run inside a molab (marimo) session on the RTX PRO 6000 Blackwell.
#
# Same stage order and semantics as scripts/train_a100.sh, retargeted at a notebook
# runtime: the interpreter is whatever marimo installed, the config is the Blackwell
# profile, `train` is detached so the cell that started it returns immediately, and
# the TensorFlow/Keras mismatch in the image is dealt with before anything imports
# transformers.
#
# Stages (run `env` first, then the rest in order):
#   env        interpreter, GPU, disk, TensorFlow/Keras report - no config read
#   tf         TensorFlow/Keras reconciliation only (fast, no GPU, no config)
#   check      environment + disk + config/VRAM projection + Drive credentials
#   manifest   build unified_manifest.parquet (~10 min)
#   smoke      2 steps, proves the loop runs
#   probe      30 real steps -> measured s/step + peak VRAM
#   baseline   base-model CER/WER, so training has a target
#   train      the run itself (nohup'd, resumable, mirrored to Drive)
#   eval       trained CER/WER + sensitive-field audit
#   compare    base vs adapter on one split, broken down per dataset
#   status     pid, last lines of the console, GPU, free disk
#   stop       kill the run - checkpoints already in Drive stay valid
#   all        check -> manifest -> smoke -> probe -> train
#
# Extra arguments are forwarded to the stage's CLI:
#   PROBE_STEPS=50 scripts/train_molab.sh probe
#   scripts/train_molab.sh train --epochs 2
#
# Environment:
#   CONFIG      training YAML   (default: configs/lora_config_rtx6000pro_16.yaml)
#   PY          interpreter      (default: first of python/python3/uv venv that has torch)
#   FA_OCR_TF   auto|keras|torch (default: auto - see below)
#
# TensorFlow and Keras:
#   The image ships TensorFlow with Keras 3. transformers 4.57 imports its TensorFlow
#   stack as soon as TensorFlow is importable and then refuses Keras 3 ("please
#   install tf-keras"), so `from transformers import Trainer` raises and every stage
#   dies at the import. Nothing here uses TensorFlow and nothing is uninstalled:
#     auto   give transformers tf-keras pinned to the installed TensorFlow minor, and
#            only if that cannot be installed fall back to USE_TF=0
#     keras  require tf-keras, fail instead of falling back
#     torch  USE_TF=0 from the start (fastest; TensorFlow stays installed)
#   Every path ends by importing `transformers.Trainer`, so the driver never proceeds
#   on a guess.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Exported, not just set: the python snippets below read these from the environment.
export CONFIG="${CONFIG:-configs/lora_config_rtx6000pro_16.yaml}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
# Trims allocator fragmentation: at batch 16 with variable-resolution pages the
# block sizes differ every step, and stranding memory is what ends a long run.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# HF_HOME on the session disk rather than the image's root: the base weights and
# their processor cache live here, and a session wipe clears it either way.
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"

banner() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# marimo runs inside a uv-managed venv that is not always first on PATH, and a
# bare `python` that lacks torch would fail with an import error that reads like a
# broken install. Pick an interpreter that has torch before anything else runs.
pick_python() {
  if [ -n "${PY:-}" ]; then
    return 0
  fi
  local cand
  for cand in python python3 /tmp/uv-venv/bin/python; do
    if command -v "$cand" >/dev/null 2>&1 &&
      "$cand" -c 'import importlib.util as u, sys; sys.exit(0 if u.find_spec("torch") else 1)' 2>/dev/null; then
      PY="$cand"
      return 0
    fi
  done
  PY=python
}
pick_python
export PY

# State of the TensorFlow/Keras pair, as the interpreter sees it:
#   0 = no TensorFlow installed, so transformers will not look for tf_keras
#   1 = TensorFlow and tf_keras both import - what transformers 4.57 expects
#   2 = TensorFlow without a usable tf_keras: the Keras 3 failure
# TensorFlow is really imported, not just found, because a package that exists but
# cannot load breaks the stage just the same.
tf_state() {
  $PY -c '
import importlib.util as u
import sys

if u.find_spec("tensorflow") is None:
    sys.exit(0)
try:
    import tf_keras  # Keras 2: the API transformers wants
except Exception:
    sys.exit(2)
sys.exit(1)
' >/dev/null 2>&1
}

# tf-keras is released in lockstep with TensorFlow, so the minor has to match: a bare
# `pip install tf-keras` can pull a wheel that does not line up with the TF under it.
install_tf_keras() {
  $PY - <<'EOF' || return 1
import importlib.metadata as md
import subprocess
import sys

for pkg in ("tensorflow", "tensorflow-cpu", "tf-nightly", "intel-tensorflow"):
    try:
        version = md.version(pkg)
        break
    except md.PackageNotFoundError:
        continue
else:
    raise SystemExit(3)  # no TensorFlow: there is nothing to make compatible

major, minor = version.split(".")[:2]
spec = f"tf-keras~={major}.{minor}.0"
print(f"  tensorflow {version} -> pip install {spec}", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", spec])
EOF
}

# Keep TensorFlow installed, stop transformers from looking at it. FORCE_TF_AVAILABLE
# overrides USE_TF, so it has to go as well.
pin_torch() {
  export USE_TORCH=1
  export USE_TF=0
  unset FORCE_TF_AVAILABLE
}

ensure_tf() {
  local mode="${FA_OCR_TF:-auto}" state
  if [ "$mode" = "torch" ]; then
    pin_torch
    echo "tf: transformers pinned to torch (FA_OCR_TF=torch), TensorFlow left installed"
    return 0
  fi
  state=0
  tf_state || state=$?
  case "$state" in
    0)
      echo "tf: no TensorFlow in this interpreter, nothing to reconcile"
      return 0
      ;;
    1)
      echo "tf: TensorFlow and tf_keras import cleanly, transformers may use TensorFlow"
      return 0
      ;;
  esac
  echo "tf: TensorFlow is installed without a usable tf_keras (Keras 3)"
  if install_tf_keras; then
    state=0
    tf_state || state=$?
    if [ "$state" = 1 ]; then
      echo "tf: tf-keras installed for the TensorFlow above, nothing was uninstalled"
      return 0
    fi
  fi
  if [ "$mode" = "keras" ]; then
    echo "tf: FA_OCR_TF=keras, but tf-keras could not be made to import" >&2
    return 1
  fi
  pin_torch
  # The one thing that has to work is the import the stages are about to do.
  if $PY -c "from transformers import Trainer" >/dev/null 2>&1; then
    echo "tf: falling back to USE_TF=0 - transformers stays on torch, TensorFlow stays installed"
    return 0
  fi
  echo "tf: cannot import transformers.Trainer" >&2
  return 1
}

stage_tf() {
  # Just the reconciliation: what the notebook's environment cell calls, so a
  # session without a GPU attached can still fix the transformers import.
  ensure_tf
}

stage_env() {
  banner "interpreter"
  echo "python : $PY ($($PY -V 2>&1))"
  echo "config : $CONFIG"
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
assert torch.cuda.is_bf16_supported(), 'bf16 unsupported; the rtx6000pro profile needs it'
print('bf16         yes')
from transformers import Trainer  # noqa: F401 - the import every stage depends on
print('transformers Trainer imports cleanly')
try:
    import sentencepiece
except ImportError:
    raise SystemExit('sentencepiece is missing - run: pip install sentencepiece')
print('sentencepiece', sentencepiece.__version__)
"
  banner "disk"
  # A subset pass needs ~15 GB, the full corpus ~55 GB, and a session's disk is
  # fixed: this is the one number worth checking before the downloads start.
  df -h "$ROOT"
}

stage_check() {
  stage_env
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
  banner "probe (measured)"
  $PY -m fa_ocr_lora.training.train_lora --config "$CONFIG" --probe "${PROBE_STEPS:-30}" --no-drive "$@"
}

stage_eval() {
  banner "eval (trained adapter)"
  $PY -m fa_ocr_lora.training.eval --config "$CONFIG" "$@"
}

stage_compare() {
  banner "compare (base vs adapter)"
  # Both variants are scored over the same records, so the delta isolates the
  # adapter rather than a difference in which rows were sampled.
  $PY -m fa_ocr_lora.training.compare_eval --config "$CONFIG" "$@"
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
  # nohup + setsid so the cell that started this returns immediately and neither a
  # dropped connection nor marimo re-running the cell takes the run with it. The run
  # also writes its own $LOG_DIR/logs/train.log, which is what gets mirrored to Drive;
  # this file additionally captures anything that bypasses logging (tracebacks,
  # tqdm, CUDA's own stderr).
  setsid nohup $PY -m fa_ocr_lora.training.train_lora \
    --config "$CONFIG" "$@" >>"$console" 2>&1 &
  local pid=$!
  echo "$pid" >"${log_dir}/train.pid"
  echo "started pid $pid"
  echo "  console : tail -f $console"
  echo "  run log : tail -f ${log_dir}/logs/train.log"
  echo "  status  : $0 status"
  echo "  stop    : $0 stop   (checkpoints already in Drive stay valid)"
  echo "  resume  : $0 train"
  echo "  note    : keep a cell printing every minute or marimo kills the idle kernel"
}

# Is the detached run still a process? `kill -0` is not enough: the run is
# orphaned when this script exits, and a container that does not reap leaves the
# finished run in the table as a zombie - the pid answers `kill -0` forever, so
# `status` called a run that had ended "alive" and the monitor cell kept polling
# it instead of evaluating the adapter. The state field of /proc settles it, read
# through the same helper the notebook imports, so the two never disagree.
pid_state() {
  $PY - "$1" <<'EOF'
import sys

from fa_ocr_lora.utils.process import pid_state

try:
    pid = int(sys.argv[1])
except ValueError:
    print("gone")
else:
    print(pid_state(pid))
EOF
}

stage_status() {
  local log_dir console pidfile pid
  log_dir="$(output_dir)"
  console="${log_dir}/console.log"
  pidfile="${log_dir}/train.pid"
  banner "run"
  if [ -f "$pidfile" ]; then
    pid="$(cat "$pidfile")"
    case "$(pid_state "$pid" || true)" in
      alive) echo "pid      : $pid (alive)" ;;
      zombie) echo "pid      : $pid (exited, never reaped - the run is over)" ;;
      *) echo "pid      : $pid (not running)" ;;
    esac
  else
    echo "pid      : none recorded - the train stage has not been started"
  fi
  echo "console  : $console"
  if [ -f "$console" ]; then
    echo
    tail -n 40 "$console"
  fi
  banner "gpu"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  banner "disk"
  df -h "$ROOT"
}

stage_stop() {
  local pidfile pid
  pidfile="$(output_dir)/train.pid"
  if [ ! -f "$pidfile" ]; then
    echo "no train.pid under $(output_dir) - nothing to stop"
    return 0
  fi
  pid="$(cat "$pidfile")"
  case "$(pid_state "$pid" || true)" in
    alive)
      kill "$pid"
      echo "sent TERM to $pid (checkpoints already in Drive stay valid)"
      ;;
    zombie)
      echo "pid $pid had already exited - nothing to stop"
      ;;
    *)
      echo "pid $pid was not running"
      ;;
  esac
  rm -f "$pidfile"
}

output_dir() {
  $PY - <<'EOF'
import sys

sys.path.insert(0, "src")
from fa_ocr_lora.training.config import load_training_config

import os

print(load_training_config(os.environ["CONFIG"]).paths.output_dir)
EOF
}

usage() { sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# status/stop do not import transformers, so they stay usable when the interpreter
# is broken - which is exactly when you want them.
case "${1:-}" in
  "") usage; exit 1 ;;
  -h | --help) usage; exit 0 ;;
  tf | status | stop) ;;
  *) ensure_tf ;;
esac

case "${1:-}" in
  env)      shift; stage_env "$@" ;;
  tf)       shift; stage_tf "$@" ;;
  check)    shift; stage_check "$@" ;;
  manifest) shift; stage_manifest "$@" ;;
  smoke)    shift; stage_smoke "$@" ;;
  probe)    shift; stage_probe "$@" ;;
  baseline) shift; stage_baseline "$@" ;;
  train)    shift; stage_train "$@" ;;
  eval)     shift; stage_eval "$@" ;;
  compare)  shift; stage_compare "$@" ;;
  status)   shift; stage_status "$@" ;;
  stop)     shift; stage_stop "$@" ;;
  all)
    shift
    stage_check
    stage_manifest
    stage_smoke
    stage_probe
    stage_train "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
