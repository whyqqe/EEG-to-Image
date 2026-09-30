#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-aria-night
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

# Overnight Path-A pipeline:
#   Phase-1 fair identity probe audit (abs_only vs abs_id) — analysis, no retrain
#   Phase-2 A2-1 abs_center LOSO train (subject-centering; no GRL/RSA/rel)
# Gate for Phase-2 success vs abs_only:
#   abs Top-1 >= abs_only (or not much worse) AND fresh id_probe ↓

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

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_compiled", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda_available", ok)
if not ok:
    raise SystemExit("CUDA unavailable on this node — try exclude/requeue")
print("GPU", torch.cuda.get_device_name(0))
PY

echo "========== PHASE 1: fair identity audit =========="
python scripts/audit_identity_probe_fair.py \
  --output-dir outputs/aria/identity_audit_v1 \
  --runs \
    abs_only:outputs/aria/abs_only_v2/checkpoints \
    abs_id:outputs/aria/abs_id_v1/checkpoints \
  --max-per-sub 400 \
  --batch-size 256 \
  --seed 42

echo "========== PHASE 2: A2-1 abs_center LOSO =========="
OUT=outputs/aria/abs_center_v1
python scripts/train_aria_loso.py \
  --mode abs_center \
  --backbone atm \
  --output-dir "${OUT}" \
  --baseline-abs-dir outputs/aria/abs_only_v2 \
  --nz 256 \
  --anchor-k 512 \
  --epochs 40 \
  --warmup-epochs 2 \
  --max-per-sub 0 \
  --batch-size 128 \
  --num-workers 4 \
  --log-every 200 \
  --lr 2e-4 \
  --weight-decay 0.05 \
  --lambda-abs 1.0 \
  --lambda-nce 0.5 \
  --lambda-atm 0.3 \
  --lambda-rsa 0.0 \
  --lambda-rel 0.0 \
  --lambda-id 0.0 \
  --grl-max 0.0 \
  --amp \
  --seed 42

echo "========== FINAL SUMMARY =========="
python - <<'PY'
import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text())

print("--- identity_audit_v1 ---")
a = load("outputs/aria/identity_audit_v1/JOB_COMPLETE.json")
for r in a["runs"]:
    s = r["summary"]
    print(f"  {r['name']}: probe={s['identity_probe_fresh_mean']*100:.1f}% abs={s['absolute_top1_mean']*100:.1f}%")
for c in a.get("compare") or []:
    print(f"  delta {c['this']}: probe={c['delta_probe']*100:+.1f}pp abs={c['delta_abs']*100:+.1f}pp")

print("--- abs_center_v1 ---")
m = load("outputs/aria/abs_center_v1/JOB_COMPLETE.json")
print("compare", m.get("compare_to_abs_only"))
print("absolute", m["results"].get("absolute"))
print("absolute_mu", m["results"].get("absolute_mu"))
print("id_probe", m["results"].get("identity_probe_acc"))
print("[OK] overnight pipeline finished")
PY
