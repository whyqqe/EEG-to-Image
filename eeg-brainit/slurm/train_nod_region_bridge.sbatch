#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-region-bridge
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Region-Bridge on NOD sub-01:
#   1) build EVC/Ventral ROI masks
#   2) build region pairs
#   3) train region / eeg_only / global / fake

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
CACHE=/project/peilab/why/cache/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh

# /home is full — keep writable caches under /project
export HOME="${CACHE}/xdg-home"
export XDG_CACHE_HOME="${CACHE}/xdg"
export XDG_CONFIG_HOME="${CACHE}/xdg-config"
export TMPDIR="${CACHE}/tmp"
export NEUROMAPS_DATA="${CACHE}/neuromaps-data"
export HF_HOME="${CACHE}/hf"
export HF_HUB_CACHE="${CACHE}/hf/hub"
export OPENCLIP_CACHE_DIR="${CACHE}/open_clip"
export XFORMERS_DISABLED=1
export PYTHONUNBUFFERED=1
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${TMPDIR}" \
  "${NEUROMAPS_DATA}" outputs/slurm outputs/nod_region_bridge

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

echo "[STEP] prepare visual ROIs (stimvar partition)"
python scripts/prepare_nod_visual_rois.py \
  --subject sub-01 \
  --method stimvar_partition \
  --stimvar-top-frac 0.12 \
  --output-dir data/nod/processed/visual_rois \
  --neuromaps-data "${NEUROMAPS_DATA}"

echo "[STEP] prepare region pairs (classmean x64)"
python scripts/prepare_nod_region_pairs.py \
  --subjects sub-01 \
  --roi-dir data/nod/processed/visual_rois \
  --output-dir data/nod/processed/region_pairs \
  --num-rois-per-region 64 \
  --use-classmean

COMMON=(
  --pairs-root data/nod/processed/region_pairs
  --subject sub-01
  --global-teacher-ckpt outputs/eval/nod_phase2_gt_fmri2image/sub-01_classmean/fmri2clip.pt
  --epochs 40
  --teacher-epochs 25
  --batch-size 128
  --lr 3e-4
  --val-frac 0.1
  --seed 42
  --lambda-nce 1.0
  --lambda-mse-img 1.0
  --lambda-evc 0.25
  --lambda-vent 0.35
  --lambda-global 0.25
  --early-win 0.06,0.15
  --late-win 0.15,0.35
  --max-images 8
)

echo "[STEP] train region"
python scripts/train_nod_region_bridge.py \
  "${COMMON[@]}" \
  --mode region \
  --output-dir outputs/nod_region_bridge/sub-01_region

echo "[STEP] train eeg_only"
python scripts/train_nod_region_bridge.py \
  "${COMMON[@]}" \
  --mode eeg_only \
  --output-dir outputs/nod_region_bridge/sub-01_eeg_only \
  --skip-gen

echo "[STEP] train global"
python scripts/train_nod_region_bridge.py \
  "${COMMON[@]}" \
  --mode global \
  --output-dir outputs/nod_region_bridge/sub-01_global \
  --skip-gen

echo "[STEP] train fake"
python scripts/train_nod_region_bridge.py \
  "${COMMON[@]}" \
  --mode fake \
  --output-dir outputs/nod_region_bridge/sub-01_fake \
  --skip-gen

echo "[OK] Region-Bridge quartet finished"
python - <<'PY'
import json, datetime
from pathlib import Path
root = Path('outputs/nod_region_bridge')
summary = {'finished_at': datetime.datetime.now().isoformat(), 'modes': {}}
for p in sorted(root.glob('sub-01_*/metrics.json')):
    m = json.loads(p.read_text())
    ceil = m.get('teacher_ceilings') or {}
    row = {
        'best_top1': m['best_val_top1'],
        'chance_top1': m['chance_top1'],
        'teacher_ceilings': ceil,
    }
    summary['modes'][p.parent.name] = row
    print(
        p.parent.name,
        'best_top1=', round(m['best_val_top1'] * 100, 2), '%',
        'chance=', round(m['chance_top1'] * 100, 3), '%',
        'ceil=', {k: round(v * 100, 2) for k, v in ceil.items()},
    )
Path('outputs/nod_region_bridge/JOB_COMPLETE.json').write_text(json.dumps(summary, indent=2))
print('[OK] wrote outputs/nod_region_bridge/JOB_COMPLETE.json')
PY
