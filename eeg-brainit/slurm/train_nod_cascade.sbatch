#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-cascade
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Most-likely path to original vision:
#   NeuroBOLT → classmean visual fMRI → BiT-style decoder → CLIP/SDXL
#   with Phase-1b CLIP-aligned continuation.

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

N_SUB=2
for s in 03 04 05 06; do
  if [[ -d data/nod/raw/nod_fmri/derivatives/ciftify/sub-${s} ]]; then N_SUB=$((10#${s})); fi
done
# only count existing complete subjects up to 06
N_SUB=0
for s in 01 02 03 04 05 06; do
  if [[ -d data/nod/raw/nod_fmri/derivatives/ciftify/sub-${s} ]]; then N_SUB=$((N_SUB+1)); fi
done
echo "[INFO] subjects available=${N_SUB}"

# 1) CLIP for all paired images (sub-01..N)
CLIP_ALL=data/nod/processed/clip_vit_h14_all
if [[ ! -f ${CLIP_ALL}/index.json ]]; then
  echo "[INFO] precomputing CLIP for all NOD subjects with fMRI..."
  # Restrict events glob to subjects that have fMRI
  EVENTS=()
  for ((i=1; i<=N_SUB; i++)); do
    s=$(printf '%02d' "$i")
    f=data/nod/raw/ds005811/derivatives/detailed_events/sub-${s}_events.csv
    [[ -f "$f" ]] && EVENTS+=("$f")
  done
  if [[ ${#EVENTS[@]} -eq 0 ]]; then
    echo "[ERR] no event files found for N_SUB=${N_SUB}"
    exit 1
  fi
  TMP_EV=outputs/tmp/nod_events_subset
  rm -rf "${TMP_EV}"
  mkdir -p "${TMP_EV}"
  for f in "${EVENTS[@]}"; do ln -sfn "$(realpath "$f")" "${TMP_EV}/$(basename "$f")"; done
  ls -la "${TMP_EV}" | head
  python scripts/precompute_nod_clip.py \
    --events-glob "${TMP_EV}/sub-*_events.csv" \
    --output-dir "${CLIP_ALL}" \
    --batch-size 64
else
  echo "[INFO] CLIP all exists: ${CLIP_ALL}"
fi

# 2) Rebuild classmean pairs with CLIP attached
echo "[INFO] rebuilding classmean pairs with CLIP..."
python scripts/prepare_nod_pairs.py --build-pairs --max-subjects "${N_SUB}" \
  --target-mode classmean --num-rois 64 --stimvar-top-frac 0.12 \
  --clip-dir "${CLIP_ALL}"

# 3) Cascade train + eval + gen
python scripts/train_nod_cascade.py \
  --pairs-root data/nod/processed/classmean \
  --clip-dir "${CLIP_ALL}" \
  --fallback-clip-dir data/nod/processed/clip_vit_h14 \
  --phase1-ckpt outputs/nod_eeg2fmri/neurobolt_classmean_v3/checkpoints/best.pt \
  --brainit-ckpt checkpoints/brain_it/decoder_clipg_state_dict.pt \
  --output-dir outputs/nod_cascade/neurobolt_bit_v1 \
  --stage-a-epochs 25 \
  --stage-b-epochs 35 \
  --max-images 8 \
  --stage all

echo "[OK] cascade finished"
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/nod_cascade/neurobolt_bit_v1/cascade_metrics.json')
if p.is_file():
  print(json.dumps(json.loads(p.read_text()), indent=2)[:2000])
else:
  print('missing cascade_metrics.json')
PY
ls -la outputs/nod_cascade/neurobolt_bit_v1/ | head
