#!/usr/bin/env bash
# Submit GCC P0+P1 (sub-08): protocol/alpha ablation + cross-subject geometry fix.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
JID=$(sbatch --parsable "${NB_ROOT}/slurm/gcc_p0p1_sub08.sbatch")
echo "[SUBMITTED] gcc-p0p1-sub08 job ${JID}"
echo "  log : ${NB_ROOT}/outputs/slurm/gcc-p0p1-sub08-${JID}.out"
echo "  out : ${NB_ROOT}/outputs/gcc_p0p1/sub-08"
