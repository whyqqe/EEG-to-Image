#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-nod-p2
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Phase-2 smoke: NOD GT fMRI ROI → CLIP MLP → SDXL (+ Phase-1 cascade preview)

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

# Prefer classmean (Phase-1 matched). Also run stimvar if pairs exist.
python scripts/eval_nod_phase2_gt_fmri2image.py \
  --subject sub-01 \
  --target-mode classmean \
  --phase1-ckpt outputs/nod_eeg2fmri/neurobolt_classmean_v3/checkpoints/best.pt \
  --output-dir outputs/eval/nod_phase2_gt_fmri2image \
  --epochs 40 \
  --max-images 8 \
  --gen-steps 20 \
  --gen-size 512

if [[ -d data/nod/processed/stimvar/sub-01 ]]; then
  python scripts/eval_nod_phase2_gt_fmri2image.py \
    --subject sub-01 \
    --target-mode stimvar \
    --phase1-ckpt outputs/nod_eeg2fmri/neurobolt_stimvar_v3/checkpoints/best.pt \
    --output-dir outputs/eval/nod_phase2_gt_fmri2image \
    --epochs 40 \
    --max-images 8 \
    --gen-steps 20 \
    --gen-size 512 \
    --skip-cascade
fi

echo "[OK] Phase-2 smoke done"
ls -la outputs/eval/nod_phase2_gt_fmri2image/*/
python - <<'PY'
import json
from pathlib import Path
for p in sorted(Path('outputs/eval/nod_phase2_gt_fmri2image').glob('*/metrics.json')):
    m=json.loads(p.read_text())
    print(p.parent.name, 'retrieval=', {k:{kk:vv for kk,vv in v.items() if kk in ('top1','top5','n')} for k,v in m['retrieval'].items()})
PY
