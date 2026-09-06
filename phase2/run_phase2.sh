#!/usr/bin/env bash
# Phase 2 VM runner: one part at a time -> provenance check on the first output -> commit -> push. Stops on any failure.
#   phase2/run_phase2.sh p1 p2 p3 p4 p2b p4b p2c   (run from the repo root inside the venv, under tmux session `phase2`)
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
STATUS=logs/phase2_status.txt; mkdir -p logs results/phase2
check_prov() {  # $1 = json file
  python - "$1" <<'EOF'
import json, sys
sys.path.insert(0, ".")
from aase_eval.provenance import validate_provenance
d = json.load(open(sys.argv[1])); validate_provenance(d["provenance"])
print("  provenance ok:", d["provenance"]["git"]["commit"][:12], "config", d["provenance"]["eval_config_sha256"][:12], "phase2_config", d["provenance"]["phase2_config_sha256"][:12])
EOF
}
for PART in "$@"; do
  case "$PART" in
    p1) CMD="python -m phase2.p1_decode";      OUT=results/phase2/p1_decode.json ;;
    p2) CMD="python -m phase2.p2_layer_sweep"; OUT=results/phase2/p2_summary.json ;;
    p3) CMD="python -m phase2.p3_circularity"; OUT=results/phase2/p3_circularity.json ;;
    p4) CMD="python -m phase2.p4_refusal";     OUT=results/phase2/p4_refusal.json ;;
    p2b) CMD="python -m phase2.p2_layer_sweep --feature mlp_out"; OUT=results/phase2/p2_summary_mlp.json ;;   # MLP-branch feature = Phase 1/January probe space
    p4b) CMD="python -m phase2.p4_refusal --feature mlp_out";     OUT=results/phase2/p4_refusal_mlp.json ;;
    p2c) CMD="python -m phase2.p2_confounds";  OUT=results/phase2/p2_confounds.json ;;                          # CPU: length control + encoded-vs-plain score coupling
    *) echo "unknown part $PART"; exit 2 ;;
  esac
  echo "=== $(date -u +%FT%TZ) $PART START" | tee -a "$STATUS"; df -h / | tail -1 | tee -a "$STATUS"
  $CMD > "logs/phase2_${PART}.log" 2>&1; RC=$?
  echo "$(date -u +%FT%TZ) $PART exit=$RC" | tee -a "$STATUS"
  if [ $RC -ne 0 ]; then echo "PART FAILED (see logs/phase2_${PART}.log)" | tee -a "$STATUS"; tail -25 "logs/phase2_${PART}.log"; exit 1; fi
  check_prov "$OUT" | tee -a "$STATUS" || { echo "PROVENANCE CHECK FAILED" | tee -a "$STATUS"; exit 1; }
  python - "$PART" <<'EOF' | tee -a "$STATUS"
import json, sys, numpy as np
part = sys.argv[1]
# degenerate-score stop rule on the part's per-case scores
if part in ("p2", "p2b"):
    z = np.load("results/phase2/p2_scores.npz" if part == "p2" else "results/phase2/p2_scores_mlp.npz"); bad = []
    for k in z.files:
        if k.startswith("ids|"): continue
        s = z[k].astype(float)
        if np.isnan(s).any() or np.allclose(s, s[0]) or len(np.unique(np.round(s, 6))) < 0.5 * len(s): bad.append(k)
    print(f"  p2 score cells: {len([k for k in z.files if not k.startswith('ids|')])}, degenerate: {len(bad)}")
    if bad: print("  DEGENERATE:", bad[:10]); sys.exit(1)
if part == "p3":
    d = json.load(open("results/phase2/p3_circularity.json")); bad = []
    for r, modes in d["recipes"].items():
        for m, blocks in modes.items():
            for b, cell in blocks.items():
                for c in ("none", "base64", "rot13", "leetspeak"):
                    if cell[c]["harmful"]["distinct"] < 0.5 * cell[c]["harmful"]["n"]: bad.append((r, m, b, c))
    print(f"  p3 cells checked, degenerate: {len(bad)}")
    if bad: print("  DEGENERATE:", bad[:10]); sys.exit(1)
print("  stop-rule check ok")
EOF
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "STOP RULE — hard stop" | tee -a "$STATUS"; exit 1; fi
  git add -A results/phase2 logs && git commit -q -m "Phase 2 $PART: $(basename "$OUT")" && git push -q origin phase2-obfuscation
  if [ $? -ne 0 ]; then echo "PUSH FAILED — hard stop" | tee -a "$STATUS"; exit 1; fi
  echo "$(date -u +%FT%TZ) $PART pushed $(git rev-parse --short HEAD)" | tee -a "$STATUS"
done
echo "=== $(date -u +%FT%TZ) ALL PARTS DONE" | tee -a "$STATUS"
