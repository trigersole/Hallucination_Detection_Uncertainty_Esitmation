#!/bin/bash
#SBATCH --job-name=cverify-resume
#SBATCH --output=logs/cverify-resume-%j.out
#SBATCH --error=logs/cverify-resume-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=05:50:00

set -euo pipefail
if [[ "${1:-}" == "--worker" ]]; then
    # Slurm executes a spool copy of this file; BASH_SOURCE is not the repository.
    PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
else
    PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
fi
cd "$PROJECT"
test -d "$PROJECT/src/cverify" || { echo "Not the project checkout: $PROJECT" >&2; exit 1; }
export CVERIFY_PYTHON="${CVERIFY_PYTHON:-/home/msai/siddhart022/LLM_Haloscope/.venv/bin/python}"
export HF_HOME="${HF_HOME:-/home/msai/siddhart022/LLM_Haloscope/.cache/huggingface}"
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
test -x "$CVERIFY_PYTHON" || { echo "Missing Python: $CVERIFY_PYTHON" >&2; exit 1; }

if [[ "${1:-}" == "--background" ]]; then
    shift
    nohup bash "$PROJECT/scripts/run_assessment.sh" "$@" >> "$PROJECT/logs/assessment-controller.log" 2>&1 < /dev/null &
    echo "Controller PID: $!; log: $PROJECT/logs/assessment-controller.log"
elif [[ "${1:-}" == "--worker" ]]; then
    shift
    exec srun --ntasks=1 --cpus-per-task=1 "$CVERIFY_PYTHON" -m cverify.workflow worker "$@"
else
    exec "$CVERIFY_PYTHON" -u -m cverify.workflow run --script "$PROJECT/scripts/run_assessment.sh" "$@"
fi
