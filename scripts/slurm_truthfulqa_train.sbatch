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
RUN_NAME="${CVERIFY_RUN_NAME:-truthfulqa}"

cd "$PROJECT"

"$PYTHON" -m cverify.cli merge \
  --runs \
    "runs/$RUN_NAME/shard_0" \
    "runs/$RUN_NAME/shard_1" \
    "runs/$RUN_NAME/shard_2" \
    "runs/$RUN_NAME/shard_3" \
    "runs/$RUN_NAME/shard_4" \
    "runs/$RUN_NAME/shard_5" \
    "runs/$RUN_NAME/shard_6" \
    "runs/$RUN_NAME/shard_7" \
    "runs/$RUN_NAME/shard_8" \
  --output "runs/$RUN_NAME/full"

"$PYTHON" -m cverify.cli train \
  --run "runs/$RUN_NAME/full" \
  --output "runs/$RUN_NAME/full/results_seed_42_val100_test25" \
  --test-size 0.25 \
  --dev-size 0.1633986928 \
  --bootstrap 1000 \
  --seed 42
