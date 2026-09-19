#!/bin/bash
#SBATCH --job-name=cverify-train
#SBATCH --output=logs/cverify-train-%j.out
#SBATCH --error=logs/cverify-train-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="${CVERIFY_PYTHON:-/home/msai/siddhart022/LLM_Haloscope/.venv/bin/python}"

cd "$PROJECT"

"$PYTHON" -m cverify.cli merge \
  --runs \
    runs/truthfulqa/shard_0 \
    runs/truthfulqa/shard_1 \
    runs/truthfulqa/shard_2 \
    runs/truthfulqa/shard_3 \
    runs/truthfulqa/shard_4 \
    runs/truthfulqa/shard_5 \
    runs/truthfulqa/shard_6 \
    runs/truthfulqa/shard_7 \
    runs/truthfulqa/shard_8 \
  --output runs/truthfulqa/full

"$PYTHON" -m cverify.cli train \
  --run runs/truthfulqa/full \
  --output runs/truthfulqa/full/results \
  --bootstrap 1000

