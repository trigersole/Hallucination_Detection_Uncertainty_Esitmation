#!/bin/bash
#SBATCH --job-name=cverify-v3-train
#SBATCH --output=logs/cverify-v3-train-%j.out
#SBATCH --error=logs/cverify-v3-train-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

export CVERIFY_RUN_NAME="${CVERIFY_RUN_NAME:-truthfulqa_concise_assessment_l15}"
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
# Shared merger/trainer detects the representation-only feature schema.
bash "$PROJECT/scripts/slurm_truthfulqa_corrected_train.sbatch"
