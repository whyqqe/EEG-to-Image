#!/usr/bin/env bash
#SBATCH --job-name=nv-a0-synth
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.err
#
# GATE 1 -- synthetic validation of the estimator.  CPU only (numpy/SVD),
# no GPU requested, so it cannot idle a GPU node.
#
# This is a go/no-go gate: if the estimator cannot recover a KNOWN rank on
# synthetic data, no EEG result is interpretable and the project stops.
# The non-zero exit code on FAIL is what lets the pipeline halt downstream jobs
# via --dependency=afterok.
set -euo pipefail

ROOT=/project/peilab/why/eeg-retrieval/alignment
PY=${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${ROOT}/outputs/a0" "${ROOT}/outputs/slurm"

echo "[job] A0 gate  id=${SLURM_JOB_ID:-local} host=$(hostname) OMP=${OMP_NUM_THREADS}"

"${PY}" "${ROOT}/scripts/selfcheck_lib.py"

"${PY}" "${ROOT}/scripts/explore_spurious_floor.py" \
  > "${ROOT}/outputs/a0/spurious_floor.txt" 2>&1
echo "[job] spurious floor characterisation -> outputs/a0/spurious_floor.txt"

set +e
"${PY}" "${ROOT}/scripts/a0_synthetic.py" \
  --out "${ROOT}/outputs/a0" \
  --seeds "${SEEDS:-5}" \
  --n-list ${N_LIST:-200 500 1654} \
  --k-true-list ${K_LIST:-0 1 3 8} \
  --snr-list ${SNR_LIST:-0.01 0.05 0.20 0.60} \
  --p 15750 --q 1024 --D "${D:-400}" --K "${K:-32}" \
  --n-perm "${NPERM:-1000}" \
  ${QUICK:+--quick}
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
  echo "[job] *** GATE 1 FAILED (rc=$RC) -- downstream jobs must not run ***"
  echo "{\"gate\":\"A0\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"verdict\":\"FAIL\",\"rc\":$RC}" \
    > "${ROOT}/outputs/a0/gate.json"
  exit $RC
fi

echo "{\"gate\":\"A0\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"verdict\":\"PASS\",
       \"finished\":\"$(date -Iseconds)\"}" > "${ROOT}/outputs/a0/gate.json"
echo "[job] GATE 1 PASSED"
