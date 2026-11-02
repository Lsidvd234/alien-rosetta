#!/usr/bin/env bash
# Decoding tricks (greedy / beam 4 / vote 5) on Tatoeba, laptop GPU. Run from Git Bash:
#     bash D:/Downloads/fllm-project/new-language/run_decode_tricks.sh
# Resumes from results/decode_tricks_<run>.partial.json if it was stopped. Log: results/decode_tricks.log.
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
echo "=== start $(date +%H:%M)" | tee -a results/decode_tricks.log
../.venv-score/Scripts/python.exe analysis/decode_tricks.py 2>&1 | grep -v -i 'warn\|Loading weights\|it/s\]\|s/it\]' | tee -a results/decode_tricks.log
echo "=== end $(date +%H:%M)" | tee -a results/decode_tricks.log
echo DONE
