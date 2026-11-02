#!/usr/bin/env bash
# 01 Nov (user: "do the hidden state check"): analysis/hidden_word_assembly.py on CPU (the GPU runs the scoring queue).
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
{ echo "=== start $(date +%H:%M)"
  ../.venv-score/Scripts/python.exe -u analysis/hidden_word_assembly.py 20 2>&1 | grep --line-buffered -v -i "warn\|Loading weights"
  echo "=== end $(date +%H:%M)"; echo DONE; } 2>&1 | tee results/hidden_check.log
