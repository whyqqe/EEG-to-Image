#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-nb-ft
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Phase-1 v3: NeuroBOLT FT on NOD
#  1) classmean stimvar ROIs (category-level visual fMRI)
#  2) image-level stimvar ROIs (harder baseline)

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export HOME=/home/sbaiae
export XDG_CACHE_HOME=/project/peilab/why/cache/eeg-brainit/xdg
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0), torch.__version__)"
python -c "import linear_attention_transformer; print('linear_attention_transformer ok')"
python scripts/train_eeg2fmri_nod.py --config configs/nod_eeg2fmri.yaml --smoke-only

N_SUB=2
if [[ -d data/nod/raw/nod_fmri/derivatives/ciftify/sub-03 ]]; then N_SUB=3; fi
if [[ -d data/nod/raw/nod_fmri/derivatives/ciftify/sub-04 ]]; then N_SUB=4; fi
if [[ -d data/nod/raw/nod_fmri/derivatives/ciftify/sub-05 ]]; then N_SUB=5; fi
if [[ -d data/nod/raw/nod_fmri/derivatives/ciftify/sub-06 ]]; then N_SUB=6; fi
echo "[INFO] using max_subjects=${N_SUB}"

echo "[INFO] building classmean pairs (posterior EEG, stimvar mask)..."
python scripts/prepare_nod_pairs.py --build-pairs --max-subjects "${N_SUB}" \
  --target-mode classmean --num-rois 64 --stimvar-top-frac 0.12 --clip-dir ""

python scripts/train_eeg2fmri_nod.py --config configs/nod_eeg2fmri.yaml

echo "[INFO] building image-level stimvar pairs..."
python scripts/prepare_nod_pairs.py --build-pairs --max-subjects "${N_SUB}" \
  --target-mode stimvar --num-rois 64 --stimvar-top-frac 0.12 --clip-dir ""

python scripts/train_eeg2fmri_nod.py --config configs/nod_eeg2fmri_stimvar.yaml

echo "[OK] Phase-1 v3 finished"
ls -la outputs/nod_eeg2fmri/*/checkpoints/best.pt 2>/dev/null | tail -20
python - <<'PY'
import json
from pathlib import Path
for d in sorted(Path('outputs/nod_eeg2fmri').glob('*/history.json')):
    h=json.loads(d.read_text())
    if not h: continue
    best=max(h, key=lambda r: r.get('ema_val_corr', -1e9))
    print(f"{d.parent.name}: best_ema_val_corr={best['ema_val_corr']:.4f} val_corr={best['val_corr']:.4f} ep={best['epoch']} t5={best['fmri_top5']:.4f}")
PY
