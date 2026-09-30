#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-align3
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Subject-align LOSO v3: mixup + subject-adversarial + SATTC on strong encoder

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

OUT=outputs/subject_align_loso/loso_v3
python scripts/train_subject_align_loso_v3.py \
  --output-dir "${OUT}" \
  --backbone atm_style \
  --nz 256 \
  --epochs 25 \
  --max-per-sub 0 \
  --batch-size 128 \
  --lr 3e-4 \
  --lambda-nce 0.5 \
  --lambda-atm 0.2 \
  --lambda-adv 1.0 \
  --lambda-adv-ce 0.1 \
  --mixup-alpha 0.2 \
  --mixup-p 0.5 \
  --calib-list 0,100,500,2000,5000,all \
  --init-mode mean \
  --adapt-lr 1e-3 \
  --adapt-epochs 12 \
  --adapt-min-steps 100 \
  --adapt-max-steps-small 1500 \
  --adapt-max-steps-large 12000 \
  --adapt-batch-size 64 \
  --unfreeze-proj-n 2000 \
  --csls-k 10 \
  --poe-beta 0.5 \
  --seed 42

echo "[OK] subject-align LOSO v3 finished"
python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("outputs/subject_align_loso/loso_v3/JOB_COMPLETE.json").read_text())
print("\n== v3 calibration curve ==")
for name, st in sorted(m["results"].items(), key=lambda kv: (len(kv[0]), kv[0])):
    print(f"{name:16s} top1={st['top1_mean']*100:.2f}±{st.get('top1_std',0)*100:.2f}% top5={st['top5_mean']*100:.2f}%")
PY
