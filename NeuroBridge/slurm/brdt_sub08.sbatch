#!/usr/bin/env bash
#SBATCH --job-name=brdt-sub08
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/brdt-sub08-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/brdt-sub08-%j.err

# BRDT: band-routed dual tower. Gate -> encode -> train -> generate -> evaluate.
#
# HCMA's main architecture (cross-subject backbone, EEG encoder slot, structure +
# semantic dual tower, multi-condition injection decoder) is preserved. The two
# changes are (1) which frequency band each tower reads and (2) a rank-r bottleneck
# between the towers whose capacity is measured, not chosen.
#
# The probe stage can veto the routing claim (Gate 1). In that case the pipeline runs
# the merged-band fallback and says so; a failed gate is never silently dropped.
set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export SUB="${SUB:-sub-08}"
bash "${NB_ROOT}/scripts/nda/run_brdt_sub08.sh"
