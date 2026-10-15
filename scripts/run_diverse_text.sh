#!/usr/bin/env bash
#
# Drive the Persian diverse-OCR text stages through a long, interruptible run.
#
# Generating 100k samples takes days, which makes the failure modes different
# from a ten-minute script. This wrapper handles the three that actually bite:
#
#   1. The terminal closes, or SSH drops. `setsid` detaches the run from this
#      shell's process group, so it survives. An earlier background attempt
#      used plain `nohup` and was killed with its parent shell.
#   2. The process dies mid-run - a transient network partition, an OOM, a
#      gateway ban that outlasts the retry budget. The loop restarts it. This
#      is safe because stage 02 is idempotent: a sample whose JSON is already
#      on disk is skipped, so a restart costs only the in-flight requests.
#   3. A genuine, permanent failure - no API key, a malformed config. Looping
#      on that forever would hide the error, so the loop gives up after
#      MAX_RESTARTS consecutive failures that produced no new samples.
#
# Usage:
#   scripts/run_diverse_text.sh plan 100000     # draw 100k plans (minutes)
#   scripts/run_diverse_text.sh generate        # generate text (days)
#   scripts/run_diverse_text.sh qc              # filter and dedupe (minutes)
#   scripts/run_diverse_text.sh status          # how far along is it
#
#   scripts/run_diverse_text.sh generate --detach   # detach and return
#
# Everything is resumable. Re-running `generate` picks up exactly where it
# stopped; there is no --continue flag because continuing is the only
# behaviour. Pass --overwrite to the module itself to force regeneration.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${FA_DIVERSE_PYTHON:-/home/shahriar/miniconda3/envs/zilber_rag/bin/python}"
LOG_DIR="${FA_DIVERSE_LOG_DIR:-$REPO/logs/diverse}"
TEXTS_DIR="${FA_DIVERSE_TEXTS_DIR:-$REPO/data/diverse/generation/texts}"

# Consecutive restarts that produced no new samples before giving up.
MAX_RESTARTS="${MAX_RESTARTS:-8}"
# Seconds to wait after a crash. Long enough for a gateway ban to clear.
RESTART_DELAY="${RESTART_DELAY:-120}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: python not found at $PYTHON" >&2
  echo "       set FA_DIVERSE_PYTHON to your interpreter" >&2
  exit 1
fi

count_texts() {
  find "$TEXTS_DIR" -name '*.json' 2>/dev/null | wc -l | tr -d ' '
}

count_plans() {
  local f="$REPO/data/diverse/planning/plans.jsonl"
  [[ -f "$f" ]] && wc -l < "$f" | tr -d ' ' || echo 0
}

cmd_plan() {
  local total="${1:-100000}"
  echo "Drawing $total plan(s)..."
  "$PYTHON" -u -m fa_ocr_lora.diverse.planning.build --total-samples "$total" \
    2>&1 | tee "$LOG_DIR/plan.log"
}

cmd_generate() {
  local log="$LOG_DIR/generate.log"
  local restarts=0
  local planned
  planned="$(count_plans)"

  if [[ "$planned" == "0" ]]; then
    echo "error: no plans. Run: scripts/run_diverse_text.sh plan 100000" >&2
    exit 1
  fi

  echo "Generating text for $planned plan(s); $(count_texts) already on disk."
  echo "Log: $log"
  echo

  while true; do
    local before after
    before="$(count_texts)"

    # Unbuffered, so `tail -f` on the log shows progress live rather than in
    # 8 KB bursts.
    "$PYTHON" -u -m fa_ocr_lora.diverse.generation.run "$@" 2>&1 | tee -a "$log"
    local status=${PIPESTATUS[0]}

    after="$(count_texts)"
    echo "[driver] exit=$status  samples: $before -> $after" | tee -a "$log"

    if [[ "$after" -ge "$planned" ]]; then
      echo "[driver] all $planned sample(s) present; done." | tee -a "$log"
      return 0
    fi
    if [[ "$status" -eq 0 ]]; then
      # A clean exit short of the target means every remaining plan failed
      # permanently (all keys out of quota, say). Stop rather than spin.
      echo "[driver] clean exit with $((planned - after)) sample(s) short." | tee -a "$log"
      echo "[driver] check the log for quota or fatal-request errors." | tee -a "$log"
      return 0
    fi

    if [[ "$after" -gt "$before" ]]; then
      restarts=0  # progress was made; the failure was transient
    else
      restarts=$((restarts + 1))
      if [[ "$restarts" -ge "$MAX_RESTARTS" ]]; then
        echo "[driver] $restarts restart(s) with no progress; giving up." | tee -a "$log"
        echo "[driver] this is a real error, not a blip. Read $log." | tee -a "$log"
        return 1
      fi
    fi
    echo "[driver] restarting in ${RESTART_DELAY}s (attempt $restarts/$MAX_RESTARTS)" \
      | tee -a "$log"
    sleep "$RESTART_DELAY"
  done
}

cmd_qc() {
  echo "Running text QC..."
  "$PYTHON" -u -m fa_ocr_lora.diverse.textqc.run 2>&1 | tee "$LOG_DIR/textqc.log"
}

cmd_status() {
  local planned texts clean
  planned="$(count_plans)"
  texts="$(count_texts)"
  echo "plans drawn       $planned"
  echo "texts generated   $texts"
  if [[ "$planned" -gt 0 ]]; then
    echo "progress          $(awk -v a="$texts" -v b="$planned" 'BEGIN{printf "%.1f%%", 100*a/b}')"
  fi
  local cleanf="$REPO/data/diverse/textqc/texts_clean.jsonl"
  if [[ -f "$cleanf" ]]; then
    clean="$(wc -l < "$cleanf" | tr -d ' ')"
    echo "texts accepted    $clean"
  fi
  if pgrep -f "fa_ocr_lora.diverse.generation.run" >/dev/null; then
    echo "generation        RUNNING"
  else
    echo "generation        stopped"
  fi
  local log="$LOG_DIR/generate.log"
  if [[ -f "$log" ]]; then
    echo
    echo "recent throughput:"
    grep -oE '[0-9]+/[0-9]+ written \([0-9.]+/min' "$log" | tail -3 | sed 's/^/  /'
    local blocks
    blocks="$(grep -c 'gateway block' "$log" 2>/dev/null || echo 0)"
    echo "  gateway blocks so far: $blocks (lower generation.workers_per_key if this grows)"
  fi
}

action="${1:-}"
[[ $# -gt 0 ]] && shift

# --detach re-invokes this script, fully detached from the current shell.
detach=0
args=()
for arg in "$@"; do
  if [[ "$arg" == "--detach" ]]; then detach=1; else args+=("$arg"); fi
done

if [[ "$detach" -eq 1 ]]; then
  mkdir -p "$LOG_DIR"
  setsid "$BASH_SOURCE" "$action" "${args[@]+"${args[@]}"}" \
    >>"$LOG_DIR/driver.log" 2>&1 &
  echo "detached: pid $!"
  echo "follow with:  tail -f $LOG_DIR/generate.log"
  echo "status with:  scripts/run_diverse_text.sh status"
  exit 0
fi

case "$action" in
  plan)     cmd_plan "${args[@]+"${args[@]}"}" ;;
  generate) cmd_generate "${args[@]+"${args[@]}"}" ;;
  qc)       cmd_qc ;;
  status)   cmd_status ;;
  *)
    sed -n '3,30p' "$BASH_SOURCE" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
