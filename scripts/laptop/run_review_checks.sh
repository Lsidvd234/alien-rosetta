#!/usr/bin/env bash
# Review follow-ups (30 Oct), laptop GPU, one job at a time. Log: results/review_checks.log
#   1 R4 positive control: the same probe on models whose output reads passives (passive300, baseline)
#   2 sealed error categories (labels only, separate files) -> writing judged on the model's own meaning
#   3 steering with the setting picked on held-out sentences (baseline, then real text seed 0)
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
P=../.venv-score/Scripts/python.exe
F='warn\|Loading weights\|it/s\]\|s/it\]'
st() { echo "=== $1 $(date +%H:%M)"; }
{
st "1 R4 positive control"
$P -u analysis/inside_passive.py --v2 e2b-v2fix-passive300 e2b-v2fix-cot-80k-noheld 2>&1 | grep --line-buffered -v -i "$F" | grep -v '^ \|^{\|^}'
st "2 sealed error categories"
for r in e2b-v2fix-cot-80k-noheld e2b-v2fix-ts1 e2b-v2fix-real15 e2b-v2fix-real15-ts1 e2b-v2fix-floor25 e2b-v2fix-focal-r2 e2b-v2fix-para e2b-v2fix-6ep e2b-v2fix-everyday e2b-v2fix-passive0 e2b-v2fix-passive50 e2b-v2fix-passive300; do
  [ -f results/sealed/$r.categories.json ] || $P -u analysis/score_sealed.py $r --categories-only 2>&1 | grep --line-buffered -v -i "$F"
done
$P -u analysis/error_split.py > /dev/null 2>&1 && echo "error split rewritten"
st "3 steering, held-out selection"
for r in e2b-v2fix-cot-80k-noheld e2b-v2fix-real15; do
  $P -u analysis/inside_claim4_steer2.py $r --heldout 2>&1 | grep --line-buffered -v -i "$F" | grep -i "picked\|stage2\|stage3\|Error"
done
st end
echo DONE
} 2>&1 | tee -a results/review_checks.log
