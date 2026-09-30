#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-aria-labram
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Fast LaBraM abs-only qualification:
#   n_times=200 (not 3000), max_per_sub=4000, bs=64, unfreeze last 2 blocks

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

OUT=outputs/aria/labram_abs_fast_v1
python scripts/train_aria_loso.py \
  --mode abs_only \
  --backbone labram \
  --output-dir "${OUT}" \
  --unfreeze-last-n 2 \
  --epochs 20 \
  --warmup-epochs 1 \
  --max-per-sub 4000 \
  --batch-size 64 \
  --num-workers 4 \
  --log-every 100 \
  --lr 2e-4 \
  --lr-backbone 1e-5 \
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

echo "[OK] LaBraM fast abs-only finished"
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path("outputs/aria/labram_abs_fast_v1/JOB_COMPLETE.json").read_text())
print("gate_pass=", m.get("abs_only_gate_pass"))
for k,st in m["results"].items():
    t1=st.get("top1_mean")
    if t1 is None: continue
    print(f"{k:22s} top1={t1*100:.2f}%")
PY
