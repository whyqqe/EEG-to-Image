#!/usr/bin/env bash
#SBATCH --job-name=nb-nmb-sota08
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

# NeuroMem-Bridge SOTA pipeline + Fusion Prior THINGS finetune for sub-08.

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
CACHE=/project/peilab/why/cache/eeg-brainit

module load slurm nvhpc-hpcx-cuda12/23.11
cd "${NB_ROOT}"
# shellcheck disable=SC1091
source /project/peilab/why/eeg-brainit/scripts/activate.sh

export BRAIN_HIVE=/project/peilab/why/Brain-HIVE
# Pin deps compatible with eeg-brainit (avoid breaking shared venv on NFS)
pip install -q 'diffusers>=0.27,<0.32' 'accelerate>=0.27,<1.2' 'huggingface-hub>=0.25,<1.0' datasets einops 2>/dev/null || true
export PYTHONPATH="${BRAIN_HIVE}:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
export HOME="${CACHE}/xdg-home"
export HF_HOME="${CACHE}/hf"
export HF_HUB_CACHE="${CACHE}/hf/hub"
export HUGGINGFACE_HUB_CACHE="${CACHE}/hf/hub"
export TRANSFORMERS_CACHE="${CACHE}/hf/hub"
export DIFFUSERS_CACHE="${CACHE}/hf/hub"
export OPENCLIP_CACHE_DIR="${CACHE}/open_clip"
export TORCH_HOME="${CACHE}/torch"
export TMPDIR="${SLURM_TMPDIR:-${CACHE}/tmp}"
mkdir -p "${TMPDIR}"
export HF_DATASETS_CACHE="${TMPDIR}/hf_datasets"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=8
export DEVICE=cuda:0
export FUSION_PRIOR_BASE=/project/peilab/why/cache/fusion_prior/H14_B32_VAE
export MAX_IMAGES="${MAX_IMAGES:-0}"
export NB_ROOT
export OUT="${NB_ROOT}/outputs/nb_nmb_sota/sub-08"

# SOTA-tuned defaults
export CFT_EPOCHS=120
export CFT_PATIENCE=18
export ENSEMBLE_ALPHA=0.45
export IMG2IMG_STRENGTH=0.5
export RERANK_SAMPLES=8

# Fusion Prior finetune (THINGS)
export FT_EPOCHS=3
export FT_MAX_STEPS=2500
export FT_BATCH=8
export FT_LR=5e-5

mkdir -p outputs/slurm "${OUT}"
echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"started_at\":\"$(date -Iseconds)\",\"config\":\"sota+finetune\"}" > "${OUT}/job_running.json"

if [[ ! -f "${FUSION_PRIOR_BASE}/fusion_encoder/config.json" ]]; then
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('fakekungfu/Fusion-Prior-H14_B32_VAE', local_dir='${FUSION_PRIOR_BASE}')
"
fi

# SDXL-base for finetune (cache under HF_HOME, not only HF_HUB_CACHE/hub)
python - <<'PY'
import os
from pathlib import Path
hf_home = Path(os.environ["HF_HOME"])
candidates = [
    hf_home / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots",
    Path(os.environ["HF_HUB_CACHE"]) / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots",
]
ok = any(p.is_dir() and any((s / "model_index.json").is_file() for s in p.iterdir() if s.is_dir()) for p in candidates if p.is_dir())
if not ok:
    print("[INFO] downloading stable-diffusion-xl-base-1.0 ...")
    from huggingface_hub import snapshot_download
    snapshot_download("stabilityai/stable-diffusion-xl-base-1.0", cache_dir=str(hf_home))
else:
    print("[INFO] SDXL-base already cached")
PY

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
bash "${NB_ROOT}/scripts/nmb/run_nmb_sota_sub08.sh"

echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"finished_at\":\"$(date -Iseconds)\",\"config\":\"sota+finetune\"}" > "${OUT}/job_done.json"
