#!/bin/bash
# Independent dataset replication; the existing controller submits all jobs.
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export CVERIFY_TRIVIAQA_ANALYSIS=within_dataset
export CVERIFY_TRIVIAQA_LABELS=reviewed
export CVERIFY_TRIVIAQA_RUN="${CVERIFY_TRIVIAQA_RUN:-triviaqa_validation500_replication_reviewed}"
exec bash "$PROJECT/scripts/run_triviaqa.sh" "$@"
