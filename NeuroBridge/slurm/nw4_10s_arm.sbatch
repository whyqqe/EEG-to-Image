#!/bin/bash
#SBATCH --job-name=nw4_10s_arm
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=08:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_10s_arm_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_10s_arm_%j.err
#
# Stage 2: one arm, all 10 subjects.  Four of these run in parallel (one per arm), each
# with its own GPU, so the whole grid finishes in a couple of hours instead of overnight.
#
#   ARM=a_spa20_cal      spa20 head, calibrated          <- best candidate
#   ARM=a_spa20_raw      spa20 head, NOT calibrated      <- isolates the calibration gain
#   ARM=a_spa0_cal       spa0 head,  calibrated          <- isolates the SPA term
#   ARM=a_spa20_ret1cal  spa20 head, snap-to-real + cal  <- tests on-manifold placement
#
#   sbatch --export=ALL,ARM=a_spa20_cal slurm/nw4_10s_arm.sbatch
set -uo pipefail
cd /project/peilab/why/NeuroBridge
export STAGE=arm
export ARM="${ARM:?set ARM=<a_spa0_cal|a_spa0_ret1cal|a_spa20_cal|a_spa20_ret1>}"
export SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9,10}"
echo "[arm job] ARM=${ARM} SLURM_JOB_ID=${SLURM_JOB_ID} node=$(hostname)"
exec bash scripts/nda/run_nw4_10subj.sh
