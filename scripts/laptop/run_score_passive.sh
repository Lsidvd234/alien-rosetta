#!/usr/bin/env bash
# Laptop scoring of the passive dose runs (user, 29 Oct). Log: results/score_passive.log. Resumable by re-running.
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
P=../.venv-score/Scripts/python.exe
F='warn\|Loading weights\|it/s\]\|s/it\]'
R="e2b-v2fix-passive0 e2b-v2fix-passive50 e2b-v2fix-passive300"
st() { echo "=== $1 $(date +%H:%M)"; }
{
st start
for r in $R; do
  [ -f results/tatoeba/$r.json ] || { st "tatoeba $r"; $P -u analysis/score_tatoeba.py $r 2>&1 | grep --line-buffered -v -i "$F"; }
  [ -f results/sealed/$r.json ] || { st "sealed $r"; $P -u analysis/score_sealed.py $r 2>&1 | grep --line-buffered -v -i "$F"; }
done
st "tables"
$P -u analysis/score_tatoeba.py --table 2>&1 | grep -i "passive\|noheld \|run"
$P -u analysis/score_sealed.py --table 2>&1 | grep -i "passive\|noheld \|run"
for r in $R; do
  [ -f results/freq_weighted/$r.json ] || { st "freq_weighted $r"; $P -u analysis/freq_weighted_eval.py $r 2>&1 | grep --line-buffered -v -i "$F" | tail -3; }
  [ -f results/subgraph/$r.json ] || { st "subgraph $r"; $P -u analysis/subgraph_check.py $r 2>&1 | grep --line-buffered -v -i "$F" | tail -3; }
done
st "error split + significance"
$P -u analysis/error_split.py > /dev/null 2>&1; $P -u analysis/significance.py > /dev/null 2>&1
st end
echo DONE
} 2>&1 | tee -a results/score_passive.log
