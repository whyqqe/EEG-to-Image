#!/usr/bin/env bash
#SBATCH --job-name=erdc-w15-loso
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=1-10
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

# W15 LOSO — per-subject S3 train + Turbo bit fuse + metrics + FID

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
export XFORMERS_DISABLED=1
mkdir -p "$TMPDIR" outputs/erdc/w15_loso_metrics outputs/eval/atm_pipeline_loso outputs/slurm

IDX=${SLURM_ARRAY_TASK_ID:-1}
SUB=$(printf "sub-%02d" "${IDX}")
echo "[INFO] LOSO subject=${SUB}"

# Per-subject override yaml
python - <<PY
import yaml
from pathlib import Path
sub = "${SUB}"
base = yaml.safe_load(Path("configs/atm_distill_s3_sub08.yaml").read_text(encoding="utf-8"))
base["output_dir"] = f"outputs/atm_distill_s3_{sub}"
base["atm"]["subject"] = sub
Path(f"configs/_loso_s3_{sub}.yaml").write_text(
    yaml.dump(base, allow_unicode=True, default_flow_style=False), encoding="utf-8"
)
print(f"[OK] wrote configs/_loso_s3_{sub}.yaml")
PY

CKPT="outputs/atm_distill_s3_${SUB}/checkpoints/atm_stage3_best.pt"
if [[ ! -f "${CKPT}" ]]; then
  echo "----- Train S3 ${SUB} -----"
  python scripts/train_atm_bridge.py \
    --config configs/base.yaml \
    --override configs/_loso_s3_${SUB}.yaml
fi

echo "----- Export bit_clip embed ${SUB} -----"
python scripts/eval_atm_pipeline.py \
  --subject "${SUB}" \
  --prior-ckpt "checkpoints/atm_diffusion_prior/${SUB}/diffusion_prior.pt" \
  --s3-ckpt "${CKPT}" \
  --output-dir outputs/eval/atm_pipeline_loso \
  --skip-generate \
  --gen-sources bit_clip

BIT_NPY="outputs/eval/atm_pipeline_loso/${SUB}_bit_clip_1024.npy"
TURBO_OUT="outputs/erdc/w15_loso_turbo_bit_${SUB}"

if [[ ! -f "${TURBO_OUT}/metrics.json" ]]; then
  echo "----- Turbo bit ${SUB} -----"
  python scripts/erdc_official_atm_pipeline.py \
    --subject "${SUB}" \
    --embed-source bit_clip \
    --bit-npy "${BIT_NPY}" \
    --low-level-mode neighbor_image \
    --top-m 2 \
    --strengths 0.35,0.50,0.65,0.85 \
    --use-turbo --gen-steps 4 --gen-guidance 0.0 \
    --output-dir "${TURBO_OUT}"
fi

FUSE_OUT="outputs/erdc/w15_loso_fuse_${SUB}_l0p15"
if [[ ! -f "${FUSE_OUT}/metrics.json" ]]; then
  echo "----- Fuse ${SUB} -----"
  python scripts/erdc_fuse_reselect.py \
    --cand-dir "${TURBO_OUT}/candidates" \
    --eeg-npy "${BIT_NPY}" \
    --lambda-struct 0.15 \
    --output-dir "${FUSE_OUT}"
fi

TAG="w15_loso_fuse_${SUB}"
MET="outputs/erdc/w15_loso_metrics/${TAG}.json"
if [[ ! -f "${MET}" ]]; then
  python scripts/erdc_full_metrics.py \
    --gen-dir "${FUSE_OUT}/selected_fused" \
    --output-json "${MET}" \
    --tag "${TAG}"
fi

FID_MET="outputs/erdc/w15_loso_metrics/${TAG}_fid.json"
if [[ ! -f "${FID_MET}" ]]; then
  python scripts/erdc_fid_metrics.py \
    --gen-dir "${FUSE_OUT}/selected_fused" \
    --output-json "${FID_MET}" \
    --tag "${TAG}"
fi

echo "[OK] LOSO ${SUB} done"
date
