#!/usr/bin/env bash
# Test A (claim 5): base model, passive0, baseline answer "who did it?" in plain English. Log: results/passive_qa_v2.log
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
F='warn\|Loading weights\|it/s\]\|s/it\]'
{ echo "=== start $(date +%H:%M)"; ../.venv-score/Scripts/python.exe -u analysis/passive_qa.py --v2 2>&1 | grep --line-buffered -v -i "$F"; echo "=== end $(date +%H:%M)"; echo DONE; } 2>&1 | tee -a results/passive_qa_v2.log
