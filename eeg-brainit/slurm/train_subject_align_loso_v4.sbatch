#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-align4
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# v4: freeze projector + hybrid ridge/SGD adapt + ablations (both,none,mixup,adv)
# both also runs seed 42 and 43

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
mkdir -p outputs/slurm outputs/subject_align_loso

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
python scripts/ensure_physics_prior_assets.py

OUT=outputs/subject_align_loso/loso_v4

# Main: both @ seeds 42,43
python scripts/train_subject_align_loso_v4.py \
  --output-dir "${OUT}" \
  --ablations both \
  --seeds 42,43 \
  --adapt-mode hybrid \
  --epochs 25 \
  --max-per-sub 0 \
  --batch-size 128 \
  --calib-list 0,100,500,2000,5000,all \
  --seed 42

# Ablations @ seed 42 only
python scripts/train_subject_align_loso_v4.py \
  --output-dir "${OUT}" \
  --ablations none,mixup,adv \
  --seeds 42 \
  --adapt-mode hybrid \
  --epochs 25 \
  --max-per-sub 0 \
  --batch-size 128 \
  --calib-list 0,100,500,2000,5000,all \
  --seed 42

# Merge cards if both wrote JOB_COMPLETE (second overwrites — rebuild)
python - <<'PY'
import json
from pathlib import Path
out = Path("outputs/subject_align_loso/loso_v4")
cards = {}
for p in sorted(out.glob("metrics_*.json")):
    blob = json.loads(p.read_text())
    key = f"{blob['ablation']}_s{blob['seed']}"
    cards[key] = {
        k: {
            "top1_mean": v["top1"]["mean"],
            "top1_std": v["top1"]["std"],
            "top5_mean": v["top5"]["mean"],
        }
        for k, v in blob["summary"].items()
    }
(out / "JOB_COMPLETE.json").write_text(
    json.dumps({"status": "ok", "protocol": "subject-align LOSO v4", "results": cards}, indent=2),
    encoding="utf-8",
)
print("\n== v4 merged ==")
for name, st in sorted(cards.items()):
    n0 = st.get("n0", {})
    nall = st.get("n_all", {})
    print(
        f"{name:16s} n0={n0.get('top1_mean',0)*100:.2f}% "
        f"n500={st.get('n500',{}).get('top1_mean',0)*100:.2f}% "
        f"n_all={nall.get('top1_mean',0)*100:.2f}%"
    )
PY

echo "[OK] subject-align LOSO v4 finished"
