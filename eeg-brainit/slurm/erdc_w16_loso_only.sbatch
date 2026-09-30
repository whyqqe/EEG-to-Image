#!/usr/bin/env bash
#SBATCH --job-name=erdc-w16-10
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# W16 — ten-subject per-subject train + eval only

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XFORMERS_DISABLED=1
mkdir -p "$TMPDIR" outputs/erdc/w16_loso_metrics outputs/slurm
pip install -q torch-fidelity 2>/dev/null || true

# Ensure sub-01 partial runs can export bit_clip from last.ckpt
python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
from erdc_loso_paths import ensure_stage_best

root = Path(".")
for i in range(1, 11):
    sub = f"sub-{i:02d}"
    ensure_stage_best(root, sub, 1)
    ensure_stage_best(root, sub, 3)
print("[OK] ensured best ckpts from last where needed")
PY

bash scripts/erdc_loso_serial.sh
date
