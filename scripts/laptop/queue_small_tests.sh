#!/usr/bin/env bash
# Small tests of 27 Oct (laptop GPU, one at a time). Run from Git Bash:
#     bash D:/Downloads/fllm-project/new-language/queue_small_tests.sh
# Waits for the loanword scoring to finish first. Log: results/queue_small_tests.log. Keep the window open until DONE.

cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
P=../.venv-score/Scripts/python.exe
F='warn\|Loading weights\|it/s\]\|s/it\]\| [0-9]*/[0-9]*$'
st() { echo "=== $1 $(date +%H:%M)"; }

{
st "waiting for the loanword scoring (results/loanwords_e2b-v2fix-loan-r2.md)"
until [ -f results/loanwords_e2b-v2fix-loan-r2.md ]; do sleep 60; done

st "1 claim 5 in v2: epoch checkpoints of e2b-v2-cot-80k (e2a-v2-80k is already scored)"
$P analysis/epoch_curve.py e2b-v2-cot-80k 2>&1 | grep -v -i "$F"
$P analysis/epoch_curve.py --table 2>&1 | grep "e2a-v2-80k\|e2b-v2-cot-80k |"

st "2 decoding tricks on Tatoeba (greedy / beam 4 / vote 5)"
$P analysis/decode_tricks.py 2>&1 | grep -v -i "$F"

st end
echo DONE
} 2>&1 | tee results/queue_small_tests.log
