#!/usr/bin/env bash
# Laptop scoring of e2b-v2fix-everyday: Tatoeba, sealed (totals), then the tables and error split. Log: results/score_everyday.log
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
P=../.venv-score/Scripts/python.exe
F='warn\|Loading weights\|it/s\]\|s/it\]'
{
echo "=== start $(date +%H:%M)"
$P -u analysis/score_tatoeba.py e2b-v2fix-everyday 2>&1 | grep --line-buffered -v -i "$F"
$P -u analysis/score_sealed.py e2b-v2fix-everyday 2>&1 | grep --line-buffered -v -i "$F"
$P -u analysis/score_tatoeba.py --table 2>&1 | grep --line-buffered -v -i "$F"
$P -u analysis/score_sealed.py --table 2>&1 | grep --line-buffered -v -i "$F" | tail -25
echo "=== end $(date +%H:%M)"; echo DONE
} 2>&1 | tee -a results/score_everyday.log
