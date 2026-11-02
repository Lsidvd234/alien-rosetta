#!/usr/bin/env bash
# 31 Oct evening (user: "push the needed 6 runs"): the wrong-sound draw runs on one account.
#   bash launch_1031.sh ACCOUNT CONFIG_DIR       e.g. main C:/Users/Kavin/.kaggle
# Waits until 05:40 (weekly quota reset ~05:30 IST), uploads the code, then runs the queue for
# kaggle_tools/jobs_ACCOUNT_1031.json (2 at a time, fetches each result). If quota or slots are not free yet the
# queue retries every minute; if the queue process dies (network), it is restarted (resume is safe: no duplicates).
ACC=$1; CFG=$2
cd D:/Downloads/fllm-project/new-language || exit 1
export PYTHONUTF8=1 KAGGLE_CONFIG_DIR="$CFG"
P=../.venv-vocab/Scripts/python.exe
LOG=results/queue_draws_1031.log
{
echo "=== $ACC: waiting for 05:40 ($(date +%H:%M))"
until [ "$(date +%H%M)" -ge 0540 ] && [ "$(date +%d)" = "31" ]; do sleep 60; done
for i in 1 2 3 4 5 6; do
  echo "=== $ACC: uploading code, try $i ($(date +%H:%M))"
  OUT=$($P kaggle_tools/kaggle_jobs.py code 2>&1 | grep -v -i "outdated\|403"); echo "$OUT"
  echo "$OUT" | grep -q "status: ready" && break
  sleep 120
done
for i in $(seq 1 30); do
  echo "=== $ACC: queue start $i ($(date +%H:%M))"
  $P -u kaggle_tools/kaggle_jobs.py queue kaggle_tools/jobs_draws_1031.json 2>&1 | grep --line-buffered -v -i outdated
  tail -40 "$LOG" | grep -q "QUEUE FINISHED" && break
  sleep 300
done
echo "=== $ACC: end ($(date +%H:%M))"; echo DONE
} 2>&1 | tee -a "$LOG"
