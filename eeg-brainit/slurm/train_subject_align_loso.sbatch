#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-align
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# ENIGMA-style shared backbone + subject W_s; LOSO + calibration curve (only adapt W_new)

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
mkdir -p outputs/slurm outputs/subject_align_loso

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
python scripts/ensure_physics_prior_assets.py

OUT=outputs/subject_align_loso/loso_v1
python scripts/train_subject_align_loso.py \
  --output-dir "${OUT}" \
  --epochs 12 \
  --max-per-sub 4000 \
  --batch-size 256 \
  --lr 3e-4 \
  --lambda-nce 0.5 \
  --calib-list 0,100,500,2000,5000,all \
  --init-mode identity \
  --adapt-lr 1e-3 \
  --adapt-epochs 8 \
  --adapt-min-steps 50 \
  --adapt-max-steps 2000 \
  --adapt-batch-size 64 \
  --seed 42

echo "[OK] subject-align LOSO finished"
python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("outputs/subject_align_loso/loso_v1/JOB_COMPLETE.json").read_text())
print("\n== calibration curve ==")
for name, st in sorted(m["results"].items(), key=lambda kv: (len(kv[0]), kv[0])):
    print(f"{name:10s} top1={st['top1_mean']*100:.2f}±{st.get('top1_std',0)*100:.2f}% top5={st['top5_mean']*100:.2f}%")
PY
