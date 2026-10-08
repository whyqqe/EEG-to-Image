#!/bin/bash
#SBATCH --job-name=v7-reps-cache
#SBATCH --partition=cpu
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v7-reps-cache-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v7-reps-cache-%j.err

# Pre-build the TEN train-repetition caches, in parallel, BEFORE the v7 sweep.
#
# WHY A SEPARATE JOB. `load_train_reps` is ~2 min per subject (read a 4.2 GB raw block, apply
# MVNN + the z-score, write a 2.1 GB float16 cache). Serially inside the sweep that is ~18
# min of dead time before the first training step, on a GPU allocation that is idle while it
# runs. The ten subjects are independent, so twelve CPUs build them in about the time one
# takes, and the GPU job then finds every cache already present.
#
# This is CPU-only work: it is file I/O and numpy, with no GPU in the loop, so taking a GPU
# node for it would only block someone else's training.
#
# Idempotent: `load_train_reps` returns the existing cache when it is present, so re-running
# this job after a partial failure costs only the missing subjects.

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

echo "[reps] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"

"${PY}" - <<'PY'
import time
from concurrent.futures import ProcessPoolExecutor

from samclip.data import things_eeg

MVNN = "train"          # the role every SOURCE subject uses in `load_loso`


def build(s: int):
    t0 = time.time()
    r = things_eeg.load_train_reps(s, None, mvnn=MVNN)
    return s, tuple(r.shape), str(r.dtype), time.time() - t0


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=10) as ex:
        for s, shape, dtype, dt in ex.map(build, range(1, 11)):
            print(f"[reps] sub-{s:02d} {shape} {dtype} {dt:.1f}s", flush=True)
    print("[reps] all caches ready", flush=True)
PY

echo "[reps] finished rc=$? at $(date -Is)"
