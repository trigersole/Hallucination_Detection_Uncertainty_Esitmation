#!/bin/bash
#SBATCH --job-name=cverify-followup
#SBATCH --output=logs/followup-%j.out
#SBATCH --error=logs/followup-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=05:50:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/msai/siddhart022/LLM_Haloscope/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="${CVERIFY_PYTHON:-/home/msai/siddhart022/LLM_Haloscope/.venv/bin/python}"
case "${1:-}" in
    question|pairs-extract)
        if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES:-}" == "NoDevFiles" ]]; then
            echo "This stage needs a GPU. Submit with: sbatch --gres=gpu:1 scripts/run_followup.sbatch $*" >&2
            exit 1
        fi
        ;;
esac
exec srun --ntasks=1 --cpus-per-task=1 "$PYTHON" -u -m cverify.followup "$@"
