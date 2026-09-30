#!/bin/bash
#SBATCH --job-name=nw4_10s_train
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_10s_train_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_10s_train_%j.err
#
# Stage 1 of the overnight 10-subject run: the spa0 head for sub-01..10.  Only one head
# is trained because SPA was measured not to convert into image quality (test RSA
# 0.181 -> 0.315 across w_spa 0 -> 20 buys -0.009 inception), so a second head would
# cost ~35 min of GPU per subject for nothing.
# Kept as its own job so the four arm jobs can start only once every head exists
# (submitted with --dependency=afterany, since a single missing subject should not
# block the other nine).
set -uo pipefail
cd /project/peilab/why/NeuroBridge
export STAGE=train
export SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9,10}"
export HEADS="${HEADS:-spa0:0.0}"
exec bash scripts/nda/run_nw4_10subj.sh
