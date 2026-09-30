#!/usr/bin/env bash
#SBATCH --job-name=epd-sem-sum
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-sem-sum-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-sem-sum-%j.err
#
# Aggregates the ladder into `outputs/sub08/epd_sem_ladder.md`.
#
# Submit it chained to the array so nobody has to remember to run it:
#   sbatch --dependency=afterany:<array-job-id> slurm/epd_sem_summary.sbatch
#
# `afterany` rather than `afterok` on purpose. A summary over 13 of 15 (arm, seed)
# runs is still the right artifact -- it prints `n/3` seeds per arm and greys out the
# arms it cannot score -- whereas `afterok` would skip the summary entirely and leave
# the run with nothing to read. The script never invents a number for a missing arm.
#
# The job is CPU-only: it reads JSON that the array already wrote. It needs no GPU,
# which is why it goes to `cpu` and can start while the GPUs are still busy.

set -uo pipefail
cd /project/peilab/why/eeg-retrieval

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
# `PY` as well as `PYTHON`: the sibling runners export `PY`, and this script used
# `"${PY}"` while exporting only `PYTHON`. Under `set -u` that is an instant
# "PY: unbound variable" at exit 1, which is how the first submission of this job
# failed in 0 seconds AFTER the array it summarises had already finished. Both names
# are exported so the same convention works either way.
export PY="${PY:-${PYTHON}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

mkdir -p outputs/slurm outputs/sub08
OUT_FILE=outputs/sub08/epd_sem_ladder.md

echo "########## job ${SLURM_JOB_ID:-local} @ $(date -Iseconds) ##########"
"${PY}" -u scripts/epd/sem_ladder.py --out-file "${OUT_FILE}"
RC=$?
echo "########## rc=${RC} @ $(date -Iseconds) ##########"
[[ -s "${OUT_FILE}" ]] || { echo "[FATAL] ${OUT_FILE} was not written"; exit 2; }
echo "########## ladder written to ${OUT_FILE} ##########"
exit "${RC}"
