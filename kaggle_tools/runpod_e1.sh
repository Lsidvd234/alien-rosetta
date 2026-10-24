#!/bin/bash
# Stage-completion runs on a RunPod pod (code unpacked in /workspace/code). Runs one job at a time; a job whose
# results JSON already exists is skipped, so the script can be restarted after an interruption.
#   nohup bash /workspace/code/kaggle_tools/runpod_e1.sh > /workspace/driver.log 2>&1 &
set -u
cd /workspace/code
export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
# Packages set up once by hand on the V100 image (torch 2.4): torch==2.6.0 (cu124 index), transformers==5.16.1
# (as on Kaggle), peft==0.21.2, accelerate; torchvision/torchaudio uninstalled (built for the old torch).
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv

COMMON="--preset tamil_like --script abugida_pua --add-tokens --model Qwen/Qwen3-0.6B-Base --epochs 5 --batch 16
 --grad-accum 2 --grad-ckpt off --precision fp16 --eval-batch 16 --n-train 50000 --new-tok-init latin --rep --rep-n 30000
 --rep-refuse 1500 --rep-replay 4000 --rep-eval-per 100 --rep-strict --max-batch-tokens 1600 --save-every-epoch"

run() {   # name, extra args
  local name=$1; shift
  local d=/workspace/runs/$name
  if [ -f "$d/results/$name.json" ]; then echo "skip $name (done)"; return; fi
  mkdir -p "$d/results"
  echo "=== $name start $(date -u +%H:%M) UTC"
  python -u -m train.run $COMMON "$@" --save-adapter "$d/model" --out "$d/results/$name.json" > "$d/log.txt" 2>&1
  echo "=== $name exit $? $(date -u +%H:%M) UTC"
}

run e1b-hard-rep-cot-5ep --cot --seed 0
run e1a-hard-rep-5ep --seed 0
run e1b-hard-rep-cot-5ep-seed1 --cot --seed 1
echo "ALL DONE $(date -u +%H:%M) UTC"
