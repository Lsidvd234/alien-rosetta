#!/usr/bin/env bash
# Laptop scoring of the 31 Oct Kaggle runs (user, 31 Oct 17:30: "yes start scoring"). Log: results/score_1031.log.
# Resumable: every step is skipped when its output already exists. GPU steps run one at a time.
# Not here: q17 on the sealed set (1.7B on the 4 GB GPU: separate decision); the writer run is scored only through
# bottleneck.py (its input is the meaning line, not English).
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 HF_HOME=D:/Downloads/wm-feasibility/hf_cache HF_HUB_OFFLINE=1
P=../.venv-score/Scripts/python.exe
F='warn\|Loading weights\|it/s\]\|s/it\]'
TR="e2b-v2fix-n98k e2b-v2fix-soundrule e2b-v2fix-loan12 e2b-v2fix-passive50-ts1 e2b-v2fix-passive0-meaning"
GLYPH="e2b-v2fix-n98k e2b-v2fix-soundrule e2b-v2fix-loan12 e2b-v2fix-passive50-ts1"
st() { echo "=== $1 $(date +%H:%M)"; }
real() {   # tatoeba + sealed for one run
  [ -f results/tatoeba/$1.json ] || { st "tatoeba $1"; $P -u analysis/score_tatoeba.py $1 2>&1 | grep --line-buffered -v -i "$F" | tail -4; }
  [ -f results/sealed/$1.json ] || { st "sealed $1"; $P -u analysis/score_sealed.py $1 2>&1 | grep --line-buffered -v -i "$F" | tail -2; }
}
gen() {    # generated-test analyses for one glyph run
  [ -f results/freq_weighted/$1.json ] || { st "freq_weighted $1"; $P -u analysis/freq_weighted_eval.py $1 2>&1 | grep --line-buffered -v -i "$F" | tail -3; }
  [ -f results/subgraph/$1.json ] || { st "subgraph $1"; $P -u analysis/subgraph_check.py $1 2>&1 | grep --line-buffered -v -i "$F" | tail -3; }
}
{
st start
# 1. maths (CPU, quick): forced routes / removed rows, and the role-dose "By order" lines
st "buildmix_blocks (one table, all mixed-rule runs)"   # writes one shared file: pass every run in one call
$P -u analysis/buildmix_blocks.py maths-meaning-buildmix-drop56 maths-meaning-buildmix-drop56-s1 maths-meaning-buildmix-drop34 \
  maths-meaning-buildmix-drop66 maths-meaning-buildmix-drop6row 2>&1 | grep -v -i "$F" | tail -12
for r in maths-meaning-build-drop56-role5 maths-meaning-build-drop56-role25 maths-meaning-build-drop56-role100 maths-meaning-build-drop56-role400; do
  [ -f results/scratch_missing_fact_$r.md ] || { st "scratch_missing_fact $r"; $P -u analysis/scratch_missing_fact.py $r 2>&1 | grep -v -i "$F" | tail -4; }
done
# 2. q17: Tatoeba from the predictions made on Kaggle (CPU)
[ -f results/tatoeba/e2b-v2fix-q17.json ] || { st "tatoeba q17 (from run)"; $P -u analysis/score_tatoeba.py --from-run e2b-v2fix-q17 2>&1 | grep -v -i "$F" | tail -4; }
# 3. real English (GPU)
for r in $TR; do real $r; done
# 4. sound-rule words (CPU)
[ -f results/sound_rule_words_e2b-v2fix-soundrule.md ] || { st "sound_rule_words"; $P -u analysis/sound_rule_words.py e2b-v2fix-soundrule 2>&1 | grep -v -i "$F" | tail -12; }
# 5. reader vs writer (GPU)
[ -f results/bottleneck.md ] || { st "bottleneck"; $P -u analysis/bottleneck.py 2>&1 | grep --line-buffered -v -i "$F" | tail -25; }
# 6. generated-test analyses (GPU/CPU)
for r in $GLYPH; do gen $r; done
# 7. passive0-ts1 is still on Kaggle (main account): wait for its result file, up to 3 h
R=e2b-v2fix-passive0-ts1
for i in $(seq 1 36); do [ -f results/kaggle/$R/results/$R.json ] && break; sleep 300; done
if [ -f results/kaggle/$R/results/$R.json ]; then real $R; gen $R; else st "$R not fetched yet, skipped"; fi
# 8. tables and cross-run analyses
st "tables"
$P -u analysis/score_tatoeba.py --table > /dev/null 2>&1; $P -u analysis/score_sealed.py --table > /dev/null 2>&1
$P -u analysis/error_split.py > /dev/null 2>&1; $P -u analysis/significance.py > /dev/null 2>&1
$P -u analysis/all_runs.py > /dev/null 2>&1
st end
echo DONE
} 2>&1 | tee -a results/score_1031.log
