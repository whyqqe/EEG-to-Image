#!/bin/bash
# GPU validation of the Stage-2 fixes: does the frozen trunk survive, and does the
# support encoder stay input-sensitive?  Short episode count on purpose -- the point
# is the invariant, not the score.
#SBATCH --job-name=samclip-validate
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/validate-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/validate-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export TMPDIR=/tmp
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"

echo "[validate] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# 0. the invariant suite must pass on the GPU node too (same torch, same device)
"${PY}" scripts/smoke_test.py

# 1. Stage 2 against the REAL trained Stage-1 checkpoint. The frozen-drift assertion
#    inside train_stage2 will raise if the trunk's BatchNorm buffers move again.
"${PY}" scripts/run_stage2_meta.py \
  --config configs/meta_s2.yaml \
  --stage1-dir outputs/stage1/sub-08 \
  --target-subject 8 \
  --episodes 300 \
  --out-dir outputs/validate/s2

echo "[validate] stage2 rc=$? at $(date -Is)"

# 2. compare the trunk's BatchNorm buffers before/after, independently of the run
"${PY}" - <<'PY'
import glob
import torch

s1 = torch.load("/project/peilab/why/CLIP/outputs/stage1/sub-08/last.pt",
                map_location="cpu", weights_only=False)["model"]
v2 = torch.load("/project/peilab/why/CLIP/outputs/validate/s2/last.pt",
                map_location="cpu", weights_only=False)["model"]
bad = []
for k, a in s1.items():
    if "trunk" in k and "running_" in k:
        b = v2[k].float()
        r = float(b.abs().max() / a.float().abs().max().clamp_min(1e-12))
        if r > 1.001 or r < 0.999:
            bad.append((k, float(a.float().abs().max()), float(b.abs().max()), r))
print(f"[validate] trunk BatchNorm buffers compared: {len(bad)} drifted")
for k, a, b, r in bad:
    print(f"  {k}: {a:.4g} -> {b:.4g} (ratio {r:.3g})")
n1 = {k: float(v) for k, v in s1.items() if k.endswith("num_batches_tracked")}
n2 = {k: float(v) for k, v in v2.items() if k.endswith("num_batches_tracked")}
moved = [k for k in n1 if n2.get(k) != n1[k]]
print(f"[validate] num_batches_tracked changed on {len(moved)} buffers "
      f"(a frozen BatchNorm must not advance)")
for k in moved:
    print(f"  {k}: {n1[k]:.0f} -> {n2[k]:.0f}")
PY

echo "[validate] done at $(date -Is)"
