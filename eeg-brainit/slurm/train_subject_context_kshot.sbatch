#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-sc-kshot
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# LOSO: Subject-Context EEG (SC-EEG) + K-shot personalization

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
mkdir -p outputs/slurm outputs/subject_context_kshot

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
python scripts/ensure_physics_prior_assets.py

OUT=outputs/subject_context_kshot/loso_v1
python scripts/train_subject_context_loso.py \
  --output-dir "${OUT}" \
  --encoders atm_style \
  --epochs 8 \
  --max-per-sub 2500 \
  --batch-size 128 \
  --lr 3e-4 \
  --lambda-atm 0.15 \
  --ctx-m 32 \
  --null-profile-p 0.25 \
  --m-list 0,50,100 \
  --k-list 0,1,5,10,20 \
  --adapt-lr 1e-3 \
  --adapt-steps 50 \
  --seed 42

echo "[OK] SC-EEG + K-shot LOSO finished"
python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("outputs/subject_context_kshot/loso_v1/JOB_COMPLETE.json").read_text())
for enc, methods in m["results"].items():
    print("\n==", enc, "==")
    for name, st in sorted(methods.items()):
        print(
            f"{name:16s} top1={st['top1_mean']*100:.2f}±{st.get('top1_std',0)*100:.2f}% "
            f"top5={st['top5_mean']*100:.2f}%"
        )
PY
