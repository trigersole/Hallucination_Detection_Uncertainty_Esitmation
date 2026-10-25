#!/bin/bash
#SBATCH --job-name=cverify-transfer
#SBATCH --output=logs/transfer-%j.out
#SBATCH --error=logs/transfer-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON="${CVERIFY_PYTHON:-/home/msai/siddhart022/LLM_Haloscope/.venv/bin/python}"
RUN="${1:-runs/truthfulqa_concise_chat_audit_l15}"
AUDITED="$("$PYTHON" -c 'import json,sys; from pathlib import Path; s=json.loads((Path(sys.argv[1])/"finalize_status.json").read_text()); assert s["phase"] == "complete", s; print(s["results"])' "$RUN")"
exec srun --ntasks=1 --cpus-per-task=1 "$PYTHON" -u -m cverify.transfer --run "$AUDITED"
