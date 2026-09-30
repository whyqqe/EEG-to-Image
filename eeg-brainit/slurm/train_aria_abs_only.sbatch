#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-aria-abs
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Path A Stage-1: ATM-style Abs-only qualification (MOST PROMISING)
# - backbone=atm (63x250, from-scratch shared encoder — matches Align recipe)
# - NO LaBraM / RSA / relative / GRL
# - Align-style MSE + InfoNCE + ATM distill
# Gate: LOSO absolute Top-1 mean >= 8% (target ~ Align n0 10.6%)

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
mkdir -p outputs/slurm outputs/aria

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

OUT=outputs/aria/abs_only_v2
python scripts/train_aria_loso.py \
  --mode abs_only \
  --backbone atm \
  --output-dir "${OUT}" \
  --nz 256 \
  --anchor-k 512 \
  --epochs 40 \
  --warmup-epochs 2 \
  --max-per-sub 0 \
  --batch-size 128 \
  --num-workers 4 \
  --log-every 200 \
  --lr 2e-4 \
  --weight-decay 0.05 \
  --lambda-abs 1.0 \
  --lambda-nce 0.5 \
  --lambda-atm 0.3 \
  --lambda-rsa 0.0 \
  --lambda-rel 0.0 \
  --lambda-id 0.0 \
  --grl-max 0.0 \
  --amp \
  --seed 42

echo "[OK] ATM Abs-only v2 finished"
python - <<'PY'
import json
from pathlib import Path
p=Path("outputs/aria/abs_only_v2/JOB_COMPLETE.json")
m=json.loads(p.read_text())
print("gate_pass=", m.get("abs_only_gate_pass"), "(need abs top1 >= 8%)")
for k,st in m["results"].items():
    t1=st.get("top1_mean")
    if t1 is None: continue
    print(f"{k:22s} top1={t1*100:.2f}%")
PY
