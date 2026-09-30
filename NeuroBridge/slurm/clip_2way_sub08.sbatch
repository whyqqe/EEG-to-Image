#!/usr/bin/env bash
#SBATCH --job-name=clip2way
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/clip_2way/sub-08"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
mkdir -p "${OUT}" outputs/slurm

T1="${NB_ROOT}/outputs/top1_structure/sub-08"
AB="${NB_ROOT}/outputs/overnight_ablation/sub-08"
COCA="${NB_ROOT}/outputs/coca_depth/sub-08"

# Build gen-dirs list for folders that still have PNGs
GEN_ARGS=""
add() { local tag="$1" path="$2"; [[ -f "${path}/000.png" ]] && GEN_ARGS="${GEN_ARGS:+$GEN_ARGS,}${tag}=${path}"; }
add ref_neighbor_depth "${T1}/generation/ref_neighbor_depth_cn0.5_ip1.0_cpa/generated"
add pred_cn0.5_cpa "${T1}/generation/pred_cn0.5_ip1.0_cpa/generated"
add pred_coca_cpa "${T1}/generation/pred_coca_ip1.0_cpa/generated"
add pred_coca_clean "${T1}/generation/pred_coca_ip1.0_clean/generated"
add pred_coca_notext "${T1}/generation/pred_coca_ip1.0/generated"
add oracle_gt_cpa "${T1}/generation/oracle_gt_cn0.5_ip1.0_cpa/generated"
add fuse_pred_cpa "${AB}/generation/fuse_pred_cn0.5_cpa/generated"
add vith_depth_cpa "${AB}/generation/vith_depth_coca_cpa/generated"
add coca_depth_champ "${COCA}/generation/depth_cn0.5_ip1.0_cpa/generated"
add canny_champ "${COCA}/generation/ref_canny_cn0.5_ip1.0_cpa/generated"

echo "[INFO] gen_args=${GEN_ARGS}"

# Prepare retrieval gallery in same 512-d space as dual-finetune embeds
python - <<'PY'
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
NB=Path("/project/peilab/why/NeuroBridge")
sys.path.insert(0,str(NB))
from module.projector import ProjectorLinear
out=Path("/project/peilab/why/NeuroBridge/outputs/clip_2way/sub-08")
out.mkdir(parents=True, exist_ok=True)
ckpt=torch.load(NB/"outputs/top1_structure/sub-08/track_t/checkpoint_clean_dual_best.pth", map_location="cpu", weights_only=False)
img=np.load(NB/"data/things_eeg/image_feature/RN50/image_test.npy").astype(np.float32)
if img.ndim==3: img=img.mean(1)
proj=ProjectorLinear(img.shape[-1], 512)
proj.load_state_dict(ckpt["img_projector_state_dict"]); proj.eval()
with torch.no_grad():
    gal=F.normalize(proj(torch.from_numpy(img)), dim=-1).cpu().numpy().astype(np.float32)
np.save(out/"gallery_clean_proj512.npy", gal)
eeg=np.load(NB/"outputs/top1_structure/sub-08/track_t/embeds/z_eeg_proj_test.npy").astype(np.float32)
np.save(out/"eeg_dual_test.npy", eeg)
# also baseline: encode with original NB ckpt quickly via existing chase embeds if any — else skip
print("gallery", gal.shape, "eeg", eeg.shape)
PY

python "${NB_ROOT}/scripts/nda/eval_clip_2way.py" \
  --gen-dirs "${GEN_ARGS}" \
  --output-json "${OUT}/clip_2way_report.json" \
  --eeg-embed-npy "${OUT}/eeg_dual_test.npy" \
  --img-gallery-npy "${OUT}/gallery_clean_proj512.npy" \
  --device cuda:0 \
  --batch-size 16

echo "{\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
