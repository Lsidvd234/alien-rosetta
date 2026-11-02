#!/usr/bin/env bash
# Binding R4 test (amended rule in results/findings_2026-10-24.md): v2 passive0 model, then the untrained base model.
#     bash D:/Downloads/fllm-project/new-language/run_r4_v2.sh        Log: results/r4_v2.log
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
F='warn\|Loading weights\|it/s\]\|s/it\]'
{
echo "=== R4 v2 start $(date +%H:%M)"
../.venv-score/Scripts/python.exe -u analysis/inside_passive.py --v2 e2b-v2fix-passive0 base 2>&1 | grep --line-buffered -v -i "$F"
echo "=== end $(date +%H:%M)"
echo DONE
} 2>&1 | tee -a results/r4_v2.log
