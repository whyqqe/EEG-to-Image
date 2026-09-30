#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-dual-teacher
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Dual-Teacher CLIP Bridge on NOD sub-01:
#   dual / eeg_only / fake_teacher  (+ optional gen grids)

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

mkdir -p outputs/slurm outputs/nod_dual_teacher

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
python scripts/ensure_dual_teacher_assets.py

COMMON=(
  --pairs-root data/nod/processed/classmean
  --clip-dir data/nod/processed/clip_vit_h14_all
  --subject sub-01
  --fmri-teacher-ckpt outputs/eval/nod_phase2_gt_fmri2image/sub-01_classmean/fmri2clip.pt
  --epochs 40
  --batch-size 128
  --lr 3e-4
  --val-frac 0.1
  --seed 42
  --max-images 8
)

python scripts/train_nod_dual_teacher.py \
  "${COMMON[@]}" \
  --mode dual \
  --lambda-mse-fmri 0.25 \
  --output-dir outputs/nod_dual_teacher/sub-01_dual

python scripts/train_nod_dual_teacher.py \
  "${COMMON[@]}" \
  --mode eeg_only \
  --lambda-mse-fmri 0.0 \
  --output-dir outputs/nod_dual_teacher/sub-01_eeg_only

python scripts/train_nod_dual_teacher.py \
  "${COMMON[@]}" \
  --mode fake_teacher \
  --lambda-mse-fmri 0.25 \
  --output-dir outputs/nod_dual_teacher/sub-01_fake_teacher

echo "[OK] Dual-Teacher trio finished"
python - <<'PY'
import json
from pathlib import Path
root = Path('outputs/nod_dual_teacher')
for p in sorted(root.glob('sub-01_*/metrics.json')):
    m = json.loads(p.read_text())
    print(p.parent.name, 'best_top1=', round(m['best_val_top1']*100, 2), '%', 'chance=', round(m['chance_top1']*100, 3), '%')
PY
