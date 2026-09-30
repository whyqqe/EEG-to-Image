#!/usr/bin/env bash
#SBATCH --job-name=lg-gate-sub08
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/lg-gate-sub08-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/lg-gate-sub08-%j.err

set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
# CN policy (per-sample router-gated CN scale within [0, 0.40])
export LG_CN_MIN=0.0
export LG_CN_MAX=0.40
export LG_STRENGTHS="0.82 0.86"
export LG_RUN_ORACLE=1
bash "${NB_ROOT}/scripts/nda/run_lg_gate_sub08.sh"
