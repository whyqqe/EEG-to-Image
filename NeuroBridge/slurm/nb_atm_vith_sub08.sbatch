#!/usr/bin/env bash
#SBATCH --job-name=nb-atm-vith
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

# NB-ATM hybrid: ViT-H 1024 encoder + ATM prior + DecodeAligner + generation.

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/nb_atm_vith/sub-08"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
export NB_ROOT OUT DEVICE=cuda:0 RETRAIN="${RETRAIN:-0}"

mkdir -p "${OUT}" outputs/slurm
echo "{\"pipeline\":\"nb_atm_vith\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"started\":\"$(date -Iseconds)\"}" > "${OUT}/job_running.json"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -d /project/peilab/why/data/images_set/test_images
test -f /project/peilab/why/eeg-brainit/checkpoints/atm_diffusion_prior/sub-08/diffusion_prior.pt

chmod +x "${NB_ROOT}/scripts/nb_atm/run_nb_atm_pipeline_sub08.sh"
bash "${NB_ROOT}/scripts/nb_atm/run_nb_atm_pipeline_sub08.sh"

echo "{\"pipeline\":\"nb_atm_vith\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
