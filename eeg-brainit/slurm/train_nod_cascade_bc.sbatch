#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-cascade-bc
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Resume cascade from Stage B (Stage A decoder_best.pt already done).

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export XFORMERS_DISABLED=1
export HOME=/home/sbaiae
export XDG_CACHE_HOME=/project/peilab/why/cache/eeg-brainit/xdg
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

test -f outputs/nod_cascade/neurobolt_bit_v1/decoder_best.pt

python scripts/train_nod_cascade.py \
  --pairs-root data/nod/processed/classmean \
  --clip-dir data/nod/processed/clip_vit_h14_all \
  --fallback-clip-dir data/nod/processed/clip_vit_h14 \
  --phase1-ckpt outputs/nod_eeg2fmri/neurobolt_classmean_v3/checkpoints/best.pt \
  --brainit-ckpt checkpoints/brain_it/decoder_clipg_state_dict.pt \
  --output-dir outputs/nod_cascade/neurobolt_bit_v1 \
  --stage-b-epochs 35 \
  --max-images 8 \
  --stage b

python scripts/train_nod_cascade.py \
  --pairs-root data/nod/processed/classmean \
  --clip-dir data/nod/processed/clip_vit_h14_all \
  --phase1-ckpt outputs/nod_eeg2fmri/neurobolt_classmean_v3/checkpoints/best.pt \
  --output-dir outputs/nod_cascade/neurobolt_bit_v1 \
  --max-images 8 \
  --stage c

echo "[OK] Stage B+C finished"
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/nod_cascade/neurobolt_bit_v1/cascade_metrics.json')
print(p.read_text() if p.is_file() else 'missing metrics')
PY
