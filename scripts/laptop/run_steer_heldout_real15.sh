#!/usr/bin/env bash
# Steering with the setting picked on held-out sentences (review follow-up; step 3 of run_review_checks.sh). Log: results/steer_heldout.log
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
F='warn\|Loading weights\|it/s\]\|s/it\]'
{
for r in e2b-v2fix-real15; do   # restart 31 Oct: cot-80k finished before the 06:38 Windows Update reboot
  echo "=== steering, held-out selection: $r $(date +%H:%M)"
  ../.venv-score/Scripts/python.exe -u analysis/inside_claim4_steer2.py $r --heldout 2>&1 | grep --line-buffered -v -i "$F" | grep --line-buffered -i "picked\|stage1 L\|stage2\|stage3\|Error\|Trace"
done
echo "=== end $(date +%H:%M)"; echo DONE
} 2>&1 | tee -a results/steer_heldout.log
