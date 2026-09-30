#!/usr/bin/env bash
#SBATCH --job-name=nv-a1-controls
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.err
#
# GATE 2 -- control battery on real EEG.  CPU only.
#
# THE POINT: a pipeline that always returns k*=0 would trivially pass any lone
# negative control, so this runs a PAIRED battery.  Only
#     (all controls k*=0) AND (early evoked windows k*>0)
# counts as evidence that the pipeline discriminates rather than mutters zero.
#
# THERE IS NO PRE-STIMULUS CONTROL IN THIS DATASET.  The stored arrays start at
# stimulus onset; info.json's `times` field does not describe them.  Using the
# declared axis makes samples 0-50 look pre-stimulus when they are really the
# 0-200 ms early evoked response.  An earlier version of this gate made that
# mistake and declared a working pipeline contaminated.  See lib_windows.py.
#
# Controls used: C1 shuffled EEG (primary), C2 shuffled CLIP, C3 per-concept
# circular time shift.  C1 is stronger than a pre-stimulus window anyway: it
# preserves every nuisance structure and breaks only the concept<->brain link.
#
# Exit codes (propagated via --dependency=afterok):
#   0  PASS / INCONCLUSIVE_NO_SIGNAL
#   4  FAIL_CONTAMINATED -- hard stop
set -euo pipefail

ROOT=/project/peilab/why/eeg-retrieval/alignment
PY=${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "${ROOT}/outputs/a1" "${ROOT}/outputs/slurm"

echo "[job] A1 controls  id=${SLURM_JOB_ID:-local} host=$(hostname)"

# fail fast if the axis assumption has drifted
"${PY}" "${ROOT}/scripts/lib_windows.py" | tee "${ROOT}/outputs/a1/axis_verification.txt"

"${PY}" "${ROOT}/scripts/a1_controls.py" \
  --out "${ROOT}/outputs/a1" \
  --subs ${SUBS:-sub-01 sub-02 sub-03 sub-04 sub-05} \
  --clip "${CLIP:-ViT-H-14}" \
  --D "${D:-256}" --K "${K:-24}" \
  --n-perm "${NPERM:-500}" \
  --seeds "${SEEDS:-2}" \
  ${QUICK:+--quick}
RC=$?

VERDICT=$("${PY}" -c "
import json;print(json.load(open('${ROOT}/outputs/a1/a1_results.json'))['gate']['verdict'])
" 2>/dev/null || echo UNKNOWN)

echo "{\"gate\":\"A1\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"verdict\":\"${VERDICT}\",
 \"rc\":${RC},\"finished\":\"$(date -Iseconds)\"}" > "${ROOT}/outputs/a1/gate.json"

if [[ "$VERDICT" == "FAIL_CONTAMINATED" ]]; then
  echo "[job] *** GATE 2 FAILED: pipeline manufactures signal -- ALL RESULTS VOID ***"
  exit 4
fi
echo "[job] GATE 2 verdict=${VERDICT}"
