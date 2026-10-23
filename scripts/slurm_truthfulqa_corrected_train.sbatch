#!/bin/bash
#SBATCH --job-name=cverify-v2-train
#SBATCH --output=logs/cverify-v2-train-%j.out
#SBATCH --error=logs/cverify-v2-train-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="${CVERIFY_PYTHON:-/home/msai/siddhart022/LLM_Haloscope/.venv/bin/python}"
RUN_NAME="${CVERIFY_RUN_NAME:-truthfulqa_sentence_yesno_l15}"
RUN="runs/$RUN_NAME/full"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

SHARDS=()
for INDEX in {0..8}; do
    SHARDS+=("runs/$RUN_NAME/shard_$INDEX")
done

if [[ -e "$RUN" ]]; then
    echo "ERROR: Refusing to overwrite $RUN; choose a fresh run or train from existing features with the CLI" >&2
    exit 2
fi

"$PYTHON" -m cverify.cli merge --runs "${SHARDS[@]}" --output "$RUN"

# Same split for all three layer conditions; do not choose a winner on test data.
for CONDITION in original_layers layer15 all_layers; do
    case "$CONDITION" in
        original_layers) LAYERS=(-1 -4 -8) ;;
        layer15) LAYERS=(15) ;;
        all_layers) LAYERS=(-1 -4 -8 15) ;;
    esac
    "$PYTHON" -m cverify.cli train \
        --run "$RUN" \
        --output "$RUN/results_seed_42_val100_test25_$CONDITION" \
        --layers "${LAYERS[@]}" \
        --test-size 0.25 \
        --dev-size 0.1633986928 \
        --bootstrap 1000 \
        --seed 42
done
