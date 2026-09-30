#!/bin/bash
#SBATCH --job-name=cfmsf_sel
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/cfmsf_sel_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/cfmsf_sel_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export SEL_OUT=/project/peilab/why/NeuroBridge/outputs/cfmsf_sel/sub-08
# the frozen arm from job 581704 -- the SAME encoder behind the 0.3815 / 0.5445 numbers,
# so any recovered headroom is directly additive to the reported result
export Z_ROOT=/project/peilab/why/NeuroBridge/outputs/cfmsf_all/sub-08/frozen/enc
export EPOCHS="${EPOCHS:-80}"
export SWA_K="${SWA_K:-5,10,20}"
export FIXED_EPOCHS="${FIXED_EPOCHS:-5,9,15,30}"

echo "[slurm] CF-MSF selection-rule probe ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[slurm] Z_ROOT=${Z_ROOT}"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv || true

set +e
bash scripts/nda/run_cfmsf_sel_sub08.sh
rc=$?
set -e
echo "[slurm] selection probe rc=${rc} @ $(date -Iseconds)"
exit "${rc}"
