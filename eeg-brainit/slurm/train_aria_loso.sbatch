#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-aria
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

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

echo "[1/2] P0 geometry diagnostic"
python scripts/diagnose_aria_geometry.py \
  --output-dir outputs/aria/diagnostics \
  --anchor-k 512

echo "[2/2] ARIA LOSO full 10-fold"
OUT=outputs/aria/loso_v1
python scripts/train_aria_loso.py \
  --output-dir "${OUT}" \
  --nz 256 \
  --anchor-k 512 \
  --epochs 30 \
  --warmup-epochs 2 \
  --max-per-sub 0 \
  --batch-size 64 \
  --lr 2e-4 \
  --lambda-rsa 1.0 \
  --lambda-rel 0.5 \
  --lambda-abs 0.1 \
  --lambda-id 0.1 \
  --lambda-atm 0.2 \
  --amp \
  --seed 42

echo "[OK] ARIA finished"
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path("outputs/aria/loso_v1/JOB_COMPLETE.json").read_text())
print("\n== ARIA ==")
for k,st in m["results"].items():
    t1=st.get("top1_mean")
    if t1 is None: continue
    print(f"{k:22s} top1={t1*100:.2f}%")
PY
