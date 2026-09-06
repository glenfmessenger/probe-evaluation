#!/usr/bin/env bash
# Phase 1 leg runner (VM). One model leg at a time: run -> check against expectations -> commit -> push.
# Stops at the first failed check so the next leg never starts on top of an unverified/unpushed one.
#
#   tools/vm_run_legs.sh R1 google/gemma-2-2b-it google/gemma-3-4b-it ...   # AAG probe training
#   tools/vm_run_legs.sh R2 <models...>                                     # AAG eval (injecagent)
#   tools/vm_run_legs.sh R3 <models...>                                     # AF eval (harmbench)
#   tools/vm_run_legs.sh R4 <models...>                                     # AILuminate (+ benign-encoded control)
#   tools/vm_run_legs.sh R6 <models...>                                     # latency
# Writes logs/<RUN>_<kind>_<model_safe>.log and a status line to logs/<RUN>_status.txt.
set -uo pipefail
RUN="$1"; shift
cd "$(dirname "$0")/.."
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 TOKENIZERS_PARALLELISM=false
STATUS="logs/${RUN}_status.txt"
case "$RUN" in
  R1) KIND=train-aag; CMD='python aase_vllm/scripts/train_probes_vllm.py --probe aag --model' ;;
  R2) KIND=injecagent; CMD='python aase_vllm/scripts/benchmark_vllm.py --benchmark injecagent --model' ;;
  R3) KIND=harmbench;  CMD='python aase_vllm/scripts/benchmark_vllm.py --benchmark harmbench --model' ;;
  R4) KIND=ailuminate; CMD='python aase_vllm/scripts/benchmark_vllm.py --benchmark ailuminate --model' ;;
  R6) KIND=latency;    CMD='python aase_vllm/scripts/benchmark_vllm.py --benchmark latency --model' ;;
  *) echo "unknown run $RUN"; exit 2 ;;
esac
for MODEL in "$@"; do
  SAFE=$(echo "$MODEL" | tr '/-' '__')
  LOG="logs/${RUN}_${KIND}_${SAFE}.log"
  echo "=== $(date -u +%FT%TZ) $RUN $KIND $MODEL START" | tee -a "$STATUS"
  df -h / | tail -1 | tee -a "$STATUS"
  $CMD "$MODEL" > "$LOG" 2>&1
  RC=$?
  echo "$(date -u +%FT%TZ) $RUN $MODEL exit=$RC" | tee -a "$STATUS"
  if [ $RC -ne 0 ]; then echo "RUN FAILED (see $LOG)" | tee -a "$STATUS"; tail -30 "$LOG"; exit 1; fi
  python tools/check_leg.py "$KIND" "$MODEL" | tee -a "$STATUS"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "CHECK FAILED — hard stop" | tee -a "$STATUS"; exit 1; fi
  git add -A aase_vllm/pretrained results/phase1 logs
  git commit -q -m "$RUN: $KIND $MODEL" && git push -q origin phase1-results
  if [ $? -ne 0 ]; then echo "PUSH FAILED — hard stop" | tee -a "$STATUS"; exit 1; fi
  echo "$(date -u +%FT%TZ) $RUN $MODEL pushed $(git rev-parse --short HEAD)" | tee -a "$STATUS"
done
echo "=== $(date -u +%FT%TZ) $RUN ALL LEGS DONE" | tee -a "$STATUS"
