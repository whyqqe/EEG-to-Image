#!/usr/bin/env bash
#SBATCH --job-name=samga-loso
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-8
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/samga-loso-%A_%a.log
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/samga-loso-%A_%a.err
#
# =============================================================================
# SAMGA inter-subject encoders for the nine LOSO folds we do not already have.
#
# WHY AN ARRAY RATHER THAN NINE JOBS
# ----------------------------------
# The account's QoS allows only 10 submitted jobs and 8 running jobs at a time, so the
# obvious "one training job per fold, then one reconstruction job per fold" layout (19
# jobs) can never be fully submitted, let alone fully in flight. An array is one job
# against the submit limit and one element against the concurrency limit, which is what
# lets the whole LOSO protocol be queued in a single submission.
#
# WHY IT SKIPS FOLDS THAT ALREADY HAVE A CHECKPOINT
# -------------------------------------------------
# Two trainings for the same fold write into the same `seed<SEED>/*sub-XX` directory, and
# the reconstruction stage picks the newest match. A duplicate would therefore be able to
# hand the pipeline a half-trained encoder while every stage still reported success. The
# guard makes resubmission a no-op instead.
# =============================================================================
set -euo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"
mkdir -p "${ROOT}/outputs/slurm" "${ROOT}/outputs/samga_official"

# The held-out subjects, in array order. Fold 8 is absent: its encoder is already trained.
FOLDS=(1 2 3 4 5 6 7 9 10)
SEED="${SEED:-2025}"
CKPT_TAG="${CKPT_TAG:-checkpoint_last.pth}"

IDX="${SLURM_ARRAY_TASK_ID:?this script is meant to run as a job array}"
[[ "${IDX}" -lt "${#FOLDS[@]}" ]] || { echo "[FATAL] array index ${IDX} out of range"; exit 1; }
TARGET="${FOLDS[${IDX}]}"
TGT="sub-$(printf '%02d' "${TARGET}")"

echo "########## fold ${TGT} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds)"
nvidia-smi -L 2>/dev/null | head -1 || true

EXISTING=$(ls -1dt "${ROOT}/outputs/samga_official/inter/seed${SEED}"/*"${TGT}"/"${CKPT_TAG}" 2>/dev/null | head -1 || true)
if [[ -n "${EXISTING}" ]]; then
  echo "[SKIP] ${TGT} already has ${CKPT_TAG}: ${EXISTING}"
  exit 0
fi

export TARGET_SUBJECT="${TARGET}"
export SEED
bash "${ROOT}/scripts/run_samga_official_baseline.sh"
RC=$?
echo "########## fold ${TGT} rc=${RC} @ $(date -Iseconds)"
exit "${RC}"
