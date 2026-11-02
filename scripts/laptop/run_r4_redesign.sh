#!/usr/bin/env bash
# After run_r4_and_loans.sh: the redesigned R4 probe (all four doer x patient number cells + baselines; amendment in
# results/findings_2026-10-24.md, 28 Oct), then the loanword tables re-scored with the fixed rule/list grouping (no model).
#     bash D:/Downloads/fllm-project/new-language/run_r4_redesign.sh        Log: results/r4_redesign.log
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
P=../.venv-score/Scripts/python.exe
F='warn\|Loading weights\|it/s\]\|s/it\]'
st() { echo "=== $1 $(date +%H:%M)"; }
{
st "waiting for run_r4_and_loans.sh to print DONE"
until grep -q '^DONE' results/r4_and_loans.log 2>/dev/null; do sleep 60; done
st "1 R4 redesign: e1b-hard-rep-cot-5ep, then base"
$P -u analysis/inside_passive.py e1b-hard-rep-cot-5ep base 2>&1 | grep --line-buffered -v -i "$F"
st "2 loanwords re-scored (fixed grouping, no model)"
CUDA_VISIBLE_DEVICES=-1 $P -u analysis/loan_words.py e2b-v2fix-loan-r2 --rescore 2>&1 | grep --line-buffered -v -i "$F"
st end
echo DONE
} 2>&1 | tee -a results/r4_redesign.log
