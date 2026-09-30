#!/usr/bin/env bash
# Submit the leak-free (P1/P2-fixed) replication + inter-subject geometry calibration.
#
# What this fixes vs the existing pipeline:
#   P1  5 selection sites now choose checkpoints on held-in TRAIN concepts
#       (scripts/nda/leakfree.py) instead of on the 200 TEST concepts.
#       Measured bias at the root encoder: 73.0% (selected) vs 68.5% (final epoch).
#   P2  deployable prompt protocols (free / neutral); the GT-concept prompts are
#       kept only as flagged reference rows so the leakage is measurable.
#   RAG alpha in {0, 0.25, 0.5} all generated (shipped alpha=0.5 costs 13.5pp Top-1).
#   P1' inter-subject geometry calibration (linear whitening / unpaired flow /
#       source-PC removal / CSLS) as the measured test of the hubness hypothesis.
#
# P3 (the cross-subject encoder was trained with sub-08 included) is NOT fixed by
# this job; no inter-subject number here may be claimed as zero-shot.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
SBATCH_FILE="${NB_ROOT}/slurm/clean_p0p1_sub08.sbatch"
mkdir -p "${NB_ROOT}/outputs/slurm"

if [[ ! -f "${SBATCH_FILE}" ]]; then
  echo "[FATAL] missing ${SBATCH_FILE}" >&2
  exit 1
fi

JOB_ID="$(sbatch --parsable "${SBATCH_FILE}")"
echo "[OK] submitted clean_p0p1_sub08 -> job ${JOB_ID}"
echo "     log: ${NB_ROOT}/outputs/slurm/clean-p0p1-sub08-${JOB_ID}.out"
echo "     out: ${NB_ROOT}/outputs/clean_p0p1/sub-08"
echo
echo "  watch: squeue -j ${JOB_ID}"
echo "  tail : tail -f ${NB_ROOT}/outputs/slurm/clean-p0p1-sub08-${JOB_ID}.out"
