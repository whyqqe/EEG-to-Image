#!/usr/bin/env bash
# Submit remaining paper experiments.
#
# QOS note: peilab preempt often allows only 1 active/pending job per user.
#   - If W15 continue (527796) is running, this queues ONE follow-up job.
#   - Use SERIAL=1 for all-in-one (finalize + 10-subject train/eval).
#   - Use SERIAL=0 to try array jobs (may hit QOSMaxSubmitJobPerUserLimit).
#
# Usage:
#   bash scripts/submit_paper_experiments.sh [W15_JOBID]
#   SERIAL=1 bash scripts/submit_paper_experiments.sh 527796

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
cd "${ROOT}"

W15_JOB="${1:-}"
SERIAL="${SERIAL:-1}"

if [[ -z "${W15_JOB}" ]]; then
  W15_JOB=$(squeue -u "$USER" -h -o "%i %j" | awk '$2 ~ /erdc-w15c|erdc-w15$/ {print $1; exit}')
fi

DEP=()
if [[ -n "${W15_JOB}" ]]; then
  DEP=(--dependency=afterok:"${W15_JOB}")
  echo "[INFO] dependency afterok:${W15_JOB}"
fi

if [[ "${SERIAL}" == "1" ]]; then
  JOB=$(sbatch --parsable "${DEP[@]}" slurm/erdc_w16_paper_serial.sbatch)
  echo "[OK] serial paper pipeline: ${JOB}"
  cat <<EOF

Queued: ${JOB} (erdc-w16-all, up to 72h)
  → sub-08 finalize + 10-subject train/eval + tables

Monitor: tail -f outputs/slurm/erdc-w16-all-${JOB}.out

EOF
  exit 0
fi

LOSO_TRAIN=$(sbatch --parsable "${DEP[@]}" slurm/erdc_w16_loso_train.sbatch)
LOSO_EVAL=$(sbatch --parsable --dependency=afterok:"${LOSO_TRAIN}" slurm/erdc_w16_loso_eval.sbatch)
LOSO_SUM=$(sbatch --parsable --dependency=afterok:"${LOSO_EVAL}" slurm/erdc_w16_loso_summary.sbatch)
PAPER_FIN=$(sbatch --parsable --dependency=afterok:"${LOSO_SUM}" slurm/erdc_w16_paper_finalize.sbatch)
echo "train=${LOSO_TRAIN} eval=${LOSO_EVAL} sum=${LOSO_SUM} fin=${PAPER_FIN}"
