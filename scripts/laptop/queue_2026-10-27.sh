#!/usr/bin/env bash
# Jobs 3-4 of the 27 Oct laptop queue (jobs 1-2 are done). Run from Git Bash:
#     bash D:/Downloads/fllm-project/new-language/queue_2026-10-27.sh
# Everything is also written to results/queue_2026-10-27.log. Keep this window open until it prints DONE.

cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
P=../.venv-score/Scripts/python.exe
F='warn\|Loading weights\|it/s\]\|s/it\]'
st() { echo "=== $1 $(date +%H:%M)"; }

{
st "3 scratchpad maths"
mkdir -p results/kaggle/maths-meaning-scratch-longer/results
$P train/maths.py --init meaning --eval-every 500 --scratch --add-max 4 --mul-max 3x2 \
    --out results/kaggle/maths-meaning-scratch-longer/results/maths-meaning-scratch-longer.json 2>&1 | grep -v -i "$F"

st "4 epoch checkpoints"
for e in 1 2; do
    $P analysis/score_tatoeba.py e2b-v2fix-cot-80k-noheld@epoch$e 2>&1 | grep -v -i "$F" | tail -3
    $P analysis/score_sealed.py e2b-v2fix-cot-80k-noheld@epoch$e 2>&1 | grep -v -i "$F" | tail -3
done
$P analysis/score_tatoeba.py --table 2>&1 | tail -3
$P analysis/score_sealed.py --table 2>&1 | tail -3

st end
echo DONE
} 2>&1 | tee results/queue_2026-10-27.log
