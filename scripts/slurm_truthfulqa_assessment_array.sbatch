#!/bin/bash
#SBATCH --job-name=cverify-tqa-v3
#SBATCH --output=logs/cverify-tqa-v3-%A_%a.out
#SBATCH --error=logs/cverify-tqa-v3-%A_%a.err
#SBATCH --array=0
#SBATCH --time=05:50:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

set -euo pipefail

export CVERIFY_RUN_NAME="${CVERIFY_RUN_NAME:-truthfulqa_concise_assessment_l15}"
export CVERIFY_GENERATION_PROMPT=concise
export CVERIFY_EVALUATION_PROFILE=assessment-v3
export CVERIFY_LAYERS="-1 -4 -8 15"
# Fresh generation experiment: do not inherit an old-run answer-matching check.
unset CVERIFY_REFERENCE_RUN

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
bash "$PROJECT/scripts/slurm_truthfulqa_array.sbatch"
