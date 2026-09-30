#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-popformer
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# POP-Former + CBOA LOSO (large EEG Transformer; independent of subject_align)

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
CACHE=/project/peilab/why/cache/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh

export HOME="${CACHE}/xdg-home"
export XDG_CACHE_HOME="${CACHE}/xdg"
export TMPDIR="${CACHE}/tmp"
export HF_HOME="${CACHE}/hf"
export HF_HUB_CACHE="${CACHE}/hf/hub"
export XFORMERS_DISABLED=1
export PYTHONUNBUFFERED=1
mkdir -p outputs/slurm outputs/pop_former

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
# CLIP assets already under outputs/atm_bridge/; skip heavy ensure

OUT=outputs/pop_former/loso_v1
python scripts/train_pop_former_loso.py \
  --output-dir "${OUT}" \
  --d-model 512 \
  --n-layers 8 \
  --n-heads 8 \
  --mem-tokens 64 \
  --id-rank 8 \
  --epochs 50 \
  --warmup-epochs 3 \
  --max-per-sub 0 \
  --batch-size 64 \
  --lr 2e-4 \
  --weight-decay 0.05 \
  --lambda-nce 0.5 \
  --lambda-atm 0.2 \
  --lambda-id-ap 0.2 \
  --lambda-id-per 0.1 \
  --grl-max 1.0 \
  --mem-dropout 0.15 \
  --mem-topk 8 \
  --amp \
  --seed 42

echo "[OK] POP-Former / CBOA finished"
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path("outputs/pop_former/loso_v1/JOB_COMPLETE.json").read_text())
print("\n== POP-Former ==")
for k,st in m["results"].items():
    print(f"{k:16s} top1={st['top1_mean']*100:.2f}±{st.get('top1_std',0)*100:.2f}% top5={st['top5_mean']*100:.2f}%")
PY
