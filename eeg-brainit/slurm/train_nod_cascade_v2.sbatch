#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-cascade-v2
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Cascade v2: B0 (decoder adapt on pred+GT) -> B (moment-match + weak CLIP) -> C
# Starts from Stage-A decoder (GT-trained), Phase-1 NeuroBOLT classmean_v3.

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

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

OUT=outputs/nod_cascade/neurobolt_bit_v2
mkdir -p "${OUT}/generated" outputs/slurm
# Seed with Stage-A GT decoder (do NOT reuse unhealthy Stage-B NeuroBOLT).
test -f outputs/nod_cascade/neurobolt_bit_v1/decoder_best.pt
cp -n outputs/nod_cascade/neurobolt_bit_v1/decoder_best.pt "${OUT}/decoder_best.pt"
cp -n outputs/nod_cascade/neurobolt_bit_v1/decoder_best.pt "${OUT}/decoder_stage_a.pt"

python scripts/train_nod_cascade.py \
  --pairs-root data/nod/processed/classmean \
  --clip-dir data/nod/processed/clip_vit_h14_all \
  --fallback-clip-dir data/nod/processed/clip_vit_h14 \
  --phase1-ckpt outputs/nod_eeg2fmri/neurobolt_classmean_v3/checkpoints/best.pt \
  --brainit-ckpt checkpoints/brain_it/decoder_clipg_state_dict.pt \
  --output-dir "${OUT}" \
  --stage-b0-epochs 15 \
  --stage-b-epochs 40 \
  --max-images 8 \
  --stage b

python scripts/train_nod_cascade.py \
  --pairs-root data/nod/processed/classmean \
  --clip-dir data/nod/processed/clip_vit_h14_all \
  --phase1-ckpt outputs/nod_eeg2fmri/neurobolt_classmean_v3/checkpoints/best.pt \
  --output-dir "${OUT}" \
  --max-images 8 \
  --stage c

echo "[OK] Cascade v2 B0+B+C finished"
python - <<'PY'
import json
from pathlib import Path
root = Path('outputs/nod_cascade/neurobolt_bit_v2')
for name in ['stage_b0.json', 'stage_b.json', 'cascade_metrics.json']:
    p = root / name
    print('====', name, '====')
    if not p.is_file():
        print('missing')
        continue
    if name.endswith('metrics.json'):
        print(p.read_text())
    else:
        hist = json.loads(p.read_text())
        print('epochs', len(hist))
        print('last', hist[-1] if hist else None)
        if name == 'stage_b0.json' and hist:
            best = max(hist, key=lambda r: r.get('pred_top1', 0))
            print('best_pred', best)
        if name == 'stage_b.json' and hist:
            best = max(hist, key=lambda r: r.get('score', -1e9))
            print('best_score', best)
PY
