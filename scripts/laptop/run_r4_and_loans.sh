#!/usr/bin/env bash
# Laptop GPU, one job at a time. Run from Git Bash after (or while) run_decode_tricks.sh runs; it waits for it:
#     bash D:/Downloads/fllm-project/new-language/run_r4_and_loans.sh
# 1. R4 pilot: is a misread passive's doer still in the model? (v1 E1b 5 ep, then the untrained base model) ~20-30 min
# 2. loanwords again with every prediction saved + the recognition table (resumes if stopped)               ~1.5 h
# Log: results/r4_and_loans.log. Keep the window open and the laptop awake until DONE.
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
P=../.venv-score/Scripts/python.exe
F='warn\|Loading weights\|it/s\]\|s/it\]'
st() { echo "=== $1 $(date +%H:%M)"; }

{
st "waiting for the decoding test (results/decode_tricks_e2b-v2fix-cot-80k-noheld.md)"
until [ -f results/decode_tricks_e2b-v2fix-cot-80k-noheld.md ]; do sleep 60; done

st "1 R4 pilot: passive doer inside the model"
$P -u analysis/inside_passive.py e1b-hard-rep-cot-5ep base 2>&1 | grep --line-buffered -v -i "$F"

st "2 loanwords with saved predictions + recognition"
$P -u analysis/loan_words.py e2b-v2fix-loan-r2 --n 300 2>&1 | grep --line-buffered -v -i "$F"

st end
echo DONE
} 2>&1 | tee -a results/r4_and_loans.log
