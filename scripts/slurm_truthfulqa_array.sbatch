#!/bin/bash
#SBATCH --job-name=cverify-tqa
#SBATCH --output=logs/cverify-tqa-%A_%a.out
#SBATCH --error=logs/cverify-tqa-%A_%a.err
# The cluster permits only two submitted jobs at once. This default runs the
# first wave; override it with --array=2-3, 4-5, 6-7, and 8 for later waves.
#SBATCH --array=0-1%2
#SBATCH --time=05:50:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="${CVERIFY_PYTHON:-/home/msai/siddhart022/LLM_Haloscope/.venv/bin/python}"
HALOSCOPE_OFFICIAL="${HALOSCOPE_OFFICIAL:-/home/msai/siddhart022/haloscope-official}"
MODEL="${CVERIFY_MODEL:-meta-llama/Llama-2-7b-chat-hf}"
RUN_NAME="${CVERIFY_RUN_NAME:-truthfulqa}"
GENERATION_PROMPT="${CVERIFY_GENERATION_PROMPT:-brief}"
MAX_NEW_TOKENS="${CVERIFY_MAX_NEW_TOKENS:-32}"
SHARD_SIZE="${CVERIFY_SHARD_SIZE:-100}"
OFFSET=$((SLURM_ARRAY_TASK_ID * SHARD_SIZE))
OUTPUT="$PROJECT/runs/$RUN_NAME/shard_${SLURM_ARRAY_TASK_ID}"

if [[ "$RUN_NAME" == "truthfulqa" && "$GENERATION_PROMPT" != "brief" ]]; then
    echo "ERROR: Refusing to put a non-brief prompt in the original truthfulqa run" >&2
    exit 2
fi
if [[ "$RUN_NAME" == "truthfulqa_sentence" && "$GENERATION_PROMPT" != "complete-sentence" ]]; then
    echo "ERROR: truthfulqa_sentence requires CVERIFY_GENERATION_PROMPT=complete-sentence" >&2
    exit 2
fi

export HF_HOME="${HF_HOME:-/home/msai/siddhart022/LLM_Haloscope/.cache/huggingface}"
export PYTHONPATH="$HALOSCOPE_OFFICIAL/.deps${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT"
mkdir -p logs "runs/$RUN_NAME"

echo "Job: ${SLURM_JOB_ID:-unknown}; task: $SLURM_ARRAY_TASK_ID; offset: $OFFSET; run: $RUN_NAME; prompt: $GENERATION_PROMPT"
nvidia-smi

srun "$PYTHON" -m cverify.cli extract \
  --data data/truthfulqa.jsonl \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --generation-prompt "$GENERATION_PROMPT" \
  --offset "$OFFSET" \
  --limit "$SHARD_SIZE" \
  --samples 5 \
  --temperature 0.7 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --layers -1 -4 -8 \
  --dtype float16 \
  --device auto \
  --labeler bleurt \
  --bleurt-model "$HALOSCOPE_OFFICIAL/models/BLEURT-20" \
  --label-threshold 0.5 \
  --labeler-device cuda
