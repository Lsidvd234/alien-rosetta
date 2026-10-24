#!/bin/bash
# Wait for the RunPod stage runs to finish (driver.log says ALL DONE), then copy every run folder into
# results/kaggle/<run>/ (same layout as the Kaggle runs: results/<run>.json, model/, model_epoch1..4/, log.txt).
#   bash kaggle_tools/fetch_runpod.sh <host> <port>
H=$1; P=$2
SSH="ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -p $P root@$H"
DEST=D:/Downloads/fllm-project/new-language/results/kaggle
until $SSH "grep -q 'ALL DONE' /workspace/driver.log"; do
  $SSH "tail -n 1 /workspace/driver.log; for f in /workspace/runs/*/log.txt; do grep -E 'step|epoch' \$f | tail -n 1; done" 2>/dev/null | tail -n 4
  sleep 300
done
$SSH "cat /workspace/driver.log | grep -E '==='"
for run in $($SSH "ls /workspace/runs"); do
  $SSH "cd /workspace/runs && tar cf - $run" | tar xf - -C "$DEST" && echo "fetched $run: $(du -sh $DEST/$run | cut -f1)"
done
echo FETCH DONE
