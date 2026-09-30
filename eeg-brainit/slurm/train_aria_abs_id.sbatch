#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-aria-id
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Stage-2: Abs + delayed identity GRL (prove innovation vs abs_only_v2)
# Same ATM backbone / Align-style abs loss; add λ_id + GRL after warmup.
# Reports: absolute Top-1 + identity probe acc (should drop toward chance).

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

OUT=outputs/aria/abs_id_v1
python scripts/train_aria_loso.py \
  --mode abs_id \
  --backbone atm \
  --output-dir "${OUT}" \
  --baseline-abs-dir outputs/aria/abs_only_v2 \
  --nz 256 \
  --anchor-k 512 \
  --epochs 40 \
  --warmup-epochs 2 \
  --id-warmup-epochs 8 \
  --max-per-sub 0 \
  --batch-size 128 \
  --num-workers 4 \
  --log-every 200 \
  --lr 2e-4 \
  --weight-decay 0.05 \
  --lambda-abs 1.0 \
  --lambda-nce 0.5 \
  --lambda-atm 0.3 \
  --lambda-id 0.15 \
  --grl-max 1.0 \
  --lambda-rsa 0.0 \
  --lambda-rel 0.0 \
  --amp \
  --seed 42

echo "[OK] abs_id_v1 finished"
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path("outputs/aria/abs_id_v1/JOB_COMPLETE.json").read_text())
print("compare", m.get("compare_to_abs_only"))
print("id_probe", m["results"].get("identity_probe_acc"))
print("absolute", m["results"].get("absolute"))
PY
