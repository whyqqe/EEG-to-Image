#!/usr/bin/env bash
#SBATCH --job-name=nv-a2-ceiling
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.err
#
# A2/A3 -- noise ceiling and effective dimensionality.  CPU only.
#
# These two numbers bound EVERYTHING downstream and no better model can lift
# them (docs Prop. 1 and Prop. 5):
#   A2 ceiling  = split-half RDM reliability (Spearman-Brown) -> the best RSA any
#                 image model can achieve; every RSA is reported as a fraction.
#                 Also inverted for the neural SNR N/S = (T/2)(1-rho)/rho.
#   A3 eff.dim  = participation ratio of the 63-ch EEG field -> caps the shared
#                 rank in Prop. 1, because volume conduction makes the scalp field
#                 a smooth low-rank mixture.
#
# This job decides the PAPER BRANCH (docs E5): low ceiling -> the result becomes
# an upper bound on neural visibility; high ceiling -> report the NV profile.
#
# Axis note: windows come from lib_windows, which encodes the VERIFIED axis
# (sample 0 == stimulus onset).  info.json's `times` field does not describe the
# stored arrays.
set -euo pipefail

ROOT=/project/peilab/why/eeg-retrieval/alignment
PY=${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "${ROOT}/outputs/a2" "${ROOT}/outputs/slurm"

echo "[job] A2/A3  id=${SLURM_JOB_ID:-local} host=$(hostname)"

"${PY}" "${ROOT}/scripts/a2_ceiling_effdim.py" \
  --out "${ROOT}/outputs/a2" \
  --subs ${SUBS:-sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10} \
  --clips ${CLIPS:-RN50 ViT-H-14} \
  --n-perm "${NPERM:-500}" \
  ${QUICK:+--quick}
RC=$?

BRANCH=$("${PY}" -c "
import json;print(json.load(open('${ROOT}/outputs/a2/a2_results.json'))['summary']['branch'])
" 2>/dev/null || echo UNKNOWN)

echo "{\"stage\":\"A2A3\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"branch\":\"${BRANCH}\",
 \"rc\":${RC},\"finished\":\"$(date -Iseconds)\"}" > "${ROOT}/outputs/a2/gate.json"
echo "[job] A2/A3 done rc=${RC} branch=${BRANCH}"
