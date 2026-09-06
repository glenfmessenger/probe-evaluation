#!/usr/bin/env bash
# Stage B VM runner: one arm at a time -> provenance check on the first output -> degenerate-score check -> commit -> push.
#   gates/run_gates.sh b1 b2 b2g b3        (from the repo root, inside the venv, under tmux session `gates`)
# Stops on any failure. The sign-off pins are verified inside gates.runb.provenance, so an arm cannot run on data the
# author did not review.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
STATUS=logs/gates_status.txt; mkdir -p logs results/gates

check_first_output() {   # $1 = results dir
  python - "$1" <<'EOF'
import glob, json, sys
sys.path.insert(0, ".")
from aase_eval.provenance import validate_provenance
files = sorted(glob.glob(sys.argv[1] + "/*.json"))
if not files:
    print("  NO OUTPUT FILES"); sys.exit(1)
d = json.load(open(files[0])); p = d["provenance"]
validate_provenance(p)
so = p.get("signoff", {})
assert so.get("reviewed_by") == "Glen Messenger", "sign-off record missing from provenance"
print(f"  provenance ok ({files[0]}): {p['git']['commit'][:12]} eval_config {p['eval_config_sha256'][:12]} "
      f"gates_config {p['gates_config_sha256'][:12]} signoff {so['reviewed_by']} {so['review_date']}")
print(f"  outputs: {len(files)} file(s)")
EOF
}

for ARM in "$@"; do
  case "$ARM" in
    b1) CMD="python -m gates.b1_paired_af";    DIR=results/gates/b1 ;;
    b2) CMD="python -m gates.b2_aag_designs";  DIR=results/gates/b2 ;;
    b3) CMD="python -m gates.b3_apc_external"; DIR=results/gates/b3 ;;
    b2g) CMD="python -m gates.b2_aag_designs --group-split"; DIR=results/gates/b2_grouped ;;   # leakage control
    b2t) CMD="python -m gates.b2_aag_designs --mode templated"; DIR=results/gates/b2_templated ;;  # mode observation
    *) echo "unknown arm $ARM"; exit 2 ;;
  esac
  echo "=== $(date -u +%FT%TZ) $ARM START" | tee -a "$STATUS"; df -h / | tail -1 | tee -a "$STATUS"
  $CMD > "logs/gates_${ARM}.log" 2>&1; RC=$?
  echo "$(date -u +%FT%TZ) $ARM exit=$RC" | tee -a "$STATUS"
  if [ $RC -ne 0 ]; then
    echo "ARM FAILED (see logs/gates_${ARM}.log)" | tee -a "$STATUS"; tail -30 "logs/gates_${ARM}.log"; exit 1
  fi
  check_first_output "$DIR" | tee -a "$STATUS"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "PROVENANCE CHECK FAILED — hard stop" | tee -a "$STATUS"; exit 1; fi
  git add -A results/gates logs && git commit -q -m "Stage B $ARM: $(ls -1 $DIR | tr '\n' ' ')" && git push -q origin gates
  if [ $? -ne 0 ]; then echo "PUSH FAILED — hard stop" | tee -a "$STATUS"; exit 1; fi
  echo "$(date -u +%FT%TZ) $ARM pushed $(git rev-parse --short HEAD)" | tee -a "$STATUS"
done
echo "=== $(date -u +%FT%TZ) ALL ARMS DONE" | tee -a "$STATUS"
