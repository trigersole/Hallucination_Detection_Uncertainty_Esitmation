#!/bin/bash
#SBATCH --job-name=cverify-tqa-v2
#SBATCH --output=logs/cverify-tqa-v2-%A_%a.out
#SBATCH --error=logs/cverify-tqa-v2-%A_%a.err
#SBATCH --array=0
#SBATCH --time=05:50:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

set -euo pipefail

export CVERIFY_RUN_NAME="${CVERIFY_RUN_NAME:-truthfulqa_sentence_yesno_l15}"
export CVERIFY_GENERATION_PROMPT=complete-sentence
export CVERIFY_EVALUATION_PROFILE=yes-no-v2
export CVERIFY_LAYERS="-1 -4 -8 15"

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
# Prefer the same shard to preserve the original per-shard generation seeds.
export CVERIFY_REFERENCE_RUN="${CVERIFY_REFERENCE_RUN:-$PROJECT/runs/truthfulqa_sentence/shard_${SLURM_ARRAY_TASK_ID}}"
bash "$PROJECT/scripts/slurm_truthfulqa_array.sbatch"
