#!/usr/bin/env bash
# 01 Nov (user: judge all of yesterday's runs): loan12 L2-L4 need analysis/loan_words.py (GPU). Waits for the
# scoring queue (results/score_1031.log ends with DONE) so only one GPU job runs at a time.
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
{ echo "=== waiting for the scoring queue ($(date +%H:%M))"
  until [ "$(tail -1 results/score_1031.log)" = "DONE" ]; do sleep 120; done
  echo "=== loan_words loan12 $(date +%H:%M)"
  ../.venv-score/Scripts/python.exe -u analysis/loan_words.py e2b-v2fix-loan12 --n 300 2>&1 | grep --line-buffered -v -i "warn\|Loading weights\|it/s\]" | tail -40
  echo "=== end $(date +%H:%M)"; echo DONE; } 2>&1 | tee results/loan12_words.log
