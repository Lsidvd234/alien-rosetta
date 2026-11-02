#!/usr/bin/env bash
# 01 Nov (user: give the writer the gold meanings and see where exactly the problem is): analysis/bottleneck_items.py
# on the laptop GPU after loan12 (results/loan12_words.log ends with DONE): one GPU job at a time.
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
{ echo "=== waiting for loan12 ($(date +%H:%M))"
  until [ -f results/loan12_words.log ] && [ "$(tail -1 results/loan12_words.log)" = "DONE" ]; do sleep 120; done
  echo "=== bottleneck_items $(date +%H:%M)"
  ../.venv-score/Scripts/python.exe -u analysis/bottleneck_items.py 2>&1 | grep --line-buffered -v -i "warn\|Loading weights\|it/s\]" | tail -60
  echo "=== end $(date +%H:%M)"; echo DONE; } 2>&1 | tee results/bottleneck_items.log
