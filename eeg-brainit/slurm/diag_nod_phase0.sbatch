#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-phase0
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Phase-0: information diagnosis (go/no-go for cascade Stage B).
# Primary: sub-01. Also pooled all subjects for comparison.

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export XFORMERS_DISABLED=1
export HOME=/home/sbaiae
export XDG_CACHE_HOME=/project/peilab/why/cache/eeg-brainit/xdg
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp

mkdir -p outputs/slurm outputs/nod_cascade/phase0_diag

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

test -f outputs/nod_eeg2fmri/neurobolt_classmean_v3/checkpoints/best.pt
test -f outputs/nod_cascade/neurobolt_bit_v1/decoder_best.pt
test -d data/nod/processed/classmean/sub-01
test -f data/nod/processed/clip_vit_h14_all/index.json

python scripts/diag_nod_phase0.py \
  --pairs-root data/nod/processed/classmean \
  --clip-dir data/nod/processed/clip_vit_h14_all \
  --fallback-clip-dir data/nod/processed/clip_vit_h14 \
  --phase1-ckpt outputs/nod_eeg2fmri/neurobolt_classmean_v3/checkpoints/best.pt \
  --decoder-ckpt outputs/nod_cascade/neurobolt_bit_v1/decoder_best.pt \
  --output-dir outputs/nod_cascade/phase0_diag \
  --subjects sub-01 \
  --also-all-subjects \
  --val-frac 0.1 \
  --seed 42 \
  --ridge-train-cap 4000

echo "[OK] Phase-0 finished"
python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/nod_cascade/phase0_diag/phase0_summary.json')
print(p.read_text() if p.is_file() else 'missing summary')
PY
