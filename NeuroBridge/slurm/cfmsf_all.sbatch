#!/bin/bash
#SBATCH --job-name=cfmsf_all
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/cfmsf_all_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/cfmsf_all_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export ALL_OUT=/project/peilab/why/NeuroBridge/outputs/cfmsf_all
export TARGET="${TARGET:-levels_mean}"
export ARMS="${ARMS:-joint,frozen}"
export EPOCHS="${EPOCHS:-40}"
export PROBE_EPOCHS="${PROBE_EPOCHS:-80}"
export SUBJECTS="${SUBJECTS:-8,1,2,3,4,5,6,7,9,10}"

echo "[slurm] CF-MSF all-subject pipeline ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[slurm] SUBJECTS=${SUBJECTS}"
echo "[slurm] TARGET=${TARGET} ARMS=${ARMS} EPOCHS=${EPOCHS} PROBE_EPOCHS=${PROBE_EPOCHS}"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv || true

# `run_cfmsf_all.sh` deliberately does not use `set -e` so one bad subject cannot
# discard the other nine; it exits non-zero at the end if anything failed.  Report
# that status rather than silently swallowing it.
set +e
bash scripts/nda/run_cfmsf_all.sh
rc=$?
set -e
echo "[slurm] pipeline rc=${rc} @ $(date -Iseconds)"
if (( rc != 0 )); then
  echo "[slurm] NOTE: at least one subject failed; see the [FAILURES] block above."
  echo "[slurm] Re-submitting the same job resumes: completed stages are skipped."
fi
exit "${rc}"
