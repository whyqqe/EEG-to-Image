#!/usr/bin/env bash
#SBATCH --job-name=nv-axis-check
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.err
#
# Stage 0 sanity check: re-derive the stimulus onset from the data itself.
#
# This exists because the pipeline was very nearly invalidated by trusting a
# metadata field.  info.json advertises times = -0.2 .. +1.0 s, but the stored
# arrays hold 250 samples and their sample 0 IS stimulus onset; the -0.2 s
# baseline was already removed.  Following the metadata shifts every window by
# 200 ms, which turns the early evoked response into a fake "pre-stimulus"
# interval and makes a working pipeline look contaminated.
#
# Any job in this pipeline depends on the axis being right, so it is verified
# from the data (across-concept F-ratio, no CLIP, no whitening, no labels) rather
# than assumed.  Fails loudly if the axis is not what lib_windows declares.
set -euo pipefail

ROOT=/project/peilab/why/eeg-retrieval/alignment
PY=${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "${ROOT}/outputs/axis" "${ROOT}/outputs/slurm"

echo "[job] axis check  id=${SLURM_JOB_ID:-local} host=$(hostname)"

"${PY}" "${ROOT}/scripts/lib_windows.py" \
  | tee "${ROOT}/outputs/axis/lib_windows.txt"

"${PY}" "${ROOT}/scripts/time_resolved_reliability.py" \
  --out "${ROOT}/outputs/axis" \
  --subs ${SUBS:-sub-01 sub-02 sub-03} \
  --win 10 --step 10 \
  | tee "${ROOT}/outputs/axis/reliability.txt"

"${PY}" "${ROOT}/scripts/detect_onset.py" \
  --out "${ROOT}/outputs/axis/onset" \
  --subs ${SUBS:-sub-01 sub-02 sub-03} \
  | tee "${ROOT}/outputs/axis/onset.txt"

# assert the axis is what everything else assumes
"${PY}" - <<'EOF'
import json, sys
d = json.load(open("/project/peilab/why/eeg-retrieval/alignment/outputs/axis/onset/onset.json"))
v = d["verdict"]
print("shape-fit onsets:", v["shape_fit_onsets"])
if not v["arrays_start_at_onset"]:
    print("*** AXIS MISMATCH: arrays do NOT start at onset.")
    print("*** lib_windows.py declares onset at sample 0 -- every window is wrong.")
    sys.exit(5)
print("AXIS CONFIRMED: sample 0 == stimulus onset, as lib_windows declares.")
EOF

echo "{\"stage\":\"axis_check\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"verdict\":\"PASS\",
 \"finished\":\"$(date -Iseconds)\"}" > "${ROOT}/outputs/axis/gate.json"
echo "[job] axis check PASSED"
