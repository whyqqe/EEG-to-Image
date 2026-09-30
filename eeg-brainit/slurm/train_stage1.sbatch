#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-s1
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
python scripts/train.py --config configs/base.yaml --override configs/stage1_warmup.yaml
echo "[OK] stage1 finished"
