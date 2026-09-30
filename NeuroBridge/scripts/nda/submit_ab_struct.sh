#!/usr/bin/env bash
# Submit the AB-STRUCT sub-08 overnight pipeline.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
SUB="${SUB:-sub-08}"
cd "${NB_ROOT}"

SB="${NB_ROOT}/slurm/ab_struct_sub08.sbatch"
[[ -f "${SB}" ]] || { echo "[FATAL] missing ${SB}" >&2; exit 1; }

# The old clean_p0p1 job (565872) was cancelled: its diagnostic assets were already
# on disk and its remaining generation grid targeted the Pareto-bounded structure
# path that this experiment replaces.
JID=$(sbatch --export=ALL,SUB="${SUB}" "${SB}" | awk '{print $NF}')
echo "[SUBMIT] ab-struct-sub08 job=${JID} sub=${SUB}"
echo "[SUBMIT] log: ${NB_ROOT}/outputs/slurm/ab-struct-sub08-${JID}.out"
echo "[INFO] first GPU stage is the depth cache for Part 2; Part 1 anchors are CPU-only"
echo "[INFO] watch: grep -E 'ANCHOR|GEN |verdict' ${NB_ROOT}/outputs/slurm/ab-struct-sub08-${JID}.out"
