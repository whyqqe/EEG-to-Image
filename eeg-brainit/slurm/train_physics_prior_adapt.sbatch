#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-physics-prior
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Physics-Prior Hierarchical LoRA + gated MAML on THINGS-EEG2
# Smoke → full leave-one-out protocol for sub-08.

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
CACHE=/project/peilab/why/cache/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh

export HOME="${CACHE}/xdg-home"
export XDG_CACHE_HOME="${CACHE}/xdg"
export XDG_CONFIG_HOME="${CACHE}/xdg-config"
export TMPDIR="${CACHE}/tmp"
export HF_HOME="${CACHE}/hf"
export HF_HUB_CACHE="${CACHE}/hf/hub"
export OPENCLIP_CACHE_DIR="${CACHE}/open_clip"
export XFORMERS_DISABLED=1
export PYTHONUNBUFFERED=1
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}" outputs/slurm outputs/physics_prior_adapt

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

echo "[STEP] ensure assets (ATM bridge + test CLIP-H/14)"
python scripts/ensure_physics_prior_assets.py

MODE="${1:-smoke}"
if [[ "${MODE}" == "full" ]]; then
  OUT=outputs/physics_prior_adapt/sub-08_full
  EXTRA=(
    --meta-train-subjects sub-01,sub-02,sub-03,sub-04,sub-05,sub-06,sub-07,sub-09
    --pretrain-epochs 15
    --pretrain-max-per-sub 8000
    --meta-epochs 40
    --meta-max-per-sub 8000
    --tasks-per-epoch 16
    --k-shot 20
  )
else
  OUT=outputs/physics_prior_adapt/sub-08_smoke
  EXTRA=(
    --meta-train-subjects sub-01,sub-02,sub-03,sub-04
    --pretrain-epochs 3
    --pretrain-max-per-sub 1000
    --meta-epochs 8
    --meta-max-per-sub 2000
    --tasks-per-epoch 4
    --k-shot 20
  )
fi

echo "[STEP] train mode=${MODE} -> ${OUT}"
python scripts/train_physics_prior_adapt.py \
  --eeg-root data/processed/things-eeg2 \
  --atm-bridge-dir outputs/atm_bridge \
  --output-dir "${OUT}" \
  --meta-test-subject sub-08 \
  --clip-dim 1024 \
  --rank 8 \
  --q-size 64 \
  --inner-steps 5 \
  --inner-lr 1e-2 \
  --meta-lr 3e-4 \
  --backbone-lr 3e-4 \
  --batch-size 64 \
  --lambda-mid 1.0 \
  --lambda-low 0.2 \
  --lambda-high 0.1 \
  --lambda-atm 0.25 \
  --seed 42 \
  "${EXTRA[@]}"

echo "[OK] finished"
python - <<PY
import json
from pathlib import Path
p=Path('${OUT}/metrics.json')
m=json.loads(p.read_text())
print('physics_gated_maml', m['physics_gated_maml'])
for k,v in m['baselines'].items():
    print(k, v)
PY
