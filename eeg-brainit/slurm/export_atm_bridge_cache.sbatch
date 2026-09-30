#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-atmexp
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Safe export job: reads ATM checkpoints, writes only outputs/atm_bridge/

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
python scripts/export_atm_bridge_cache.py \
  --atm-dir /project/peilab/why/eeg-to-image/checkpoints/atm \
  --output-dir outputs/atm_bridge
echo "[OK] ATM bridge cache exported"
