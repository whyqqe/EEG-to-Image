#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-sci-pipe
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# SCI overnight pipeline (canonical roadmap):
#   Phase-0 freeze main table from existing JOB_COMPLETE
#   Phase-1 ST-GATE LOSO (unlabeled profile → low-rank transport)
# Does not overwrite subject_align / aria experiment dirs.

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
# avoid Triton/nvc compile failure on login-module toolchains
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export TORCHINDUCTOR_DISABLE=1
mkdir -p outputs/slurm outputs/st_gate

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_compiled", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA unavailable — exclude/requeue"
print("GPU", torch.cuda.get_device_name(0))
PY

echo "========== PHASE 0: freeze SCI main table =========="
python scripts/freeze_sci_main_table.py

echo "========== PHASE 1: ST-GATE LOSO =========="
# bridge assets already present; skip ensure_physics_prior (avoids open_clip redownload)
OUT=outputs/st_gate/loso_v1
python scripts/train_st_gate_loso.py \
  --output-dir "${OUT}" \
  --backbone atm_style \
  --nz 256 \
  --profile-dim 128 \
  --rank 16 \
  --epochs 25 \
  --max-per-sub 0 \
  --batch-size 128 \
  --lr 3e-4 \
  --lambda-nce 0.5 \
  --lambda-atm 0.2 \
  --lambda-adv 0.1 \
  --lambda-geo 0.25 \
  --grl-max 1.0 \
  --ctx-m 32 \
  --null-profile-p 0.25 \
  --m-list 0,50,100 \
  --k-list 0,1,5,10,20 \
  --adapt-lr 5e-4 \
  --adapt-steps 100 \
  --eval-geo \
  --seed 42

echo "========== FINAL =========="
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path("outputs/st_gate/loso_v1/JOB_COMPLETE.json").read_text())
print("ST-GATE keys", sorted(m["results"].keys())[:8], "...")
for name in ["m0_k0", "m50_k0", "m100_k0", "m100_k20", "atm_ceiling"]:
    st=m["results"].get(name)
    if not st: continue
    print(f"  {name:12s} top1={st['top1_mean']*100:.2f}%")
print("[OK] SCI pipeline finished")
PY
