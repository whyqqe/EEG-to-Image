#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-dbg
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
python -c "import torch; assert torch.cuda.is_available(), 'no cuda'; print(torch.cuda.get_device_name(0))"
python scripts/train.py --config configs/base.yaml --override configs/debug_stage1.yaml
echo "[OK] debug stage1 finished"
