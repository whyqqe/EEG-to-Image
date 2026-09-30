#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-clipteach
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
python -c "import open_clip; print('open_clip', open_clip.__version__)"

SMOKE_FLAG="${SMOKE:-0}"
if [[ "${SMOKE_FLAG}" == "1" ]]; then
  python scripts/precompute_clip_teachers.py --smoke --batch-size 32
else
  python scripts/precompute_clip_teachers.py --batch-size 64
fi
echo "[OK] CLIP teacher cache ready"
