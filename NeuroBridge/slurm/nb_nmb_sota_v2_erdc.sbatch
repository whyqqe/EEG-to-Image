#!/usr/bin/env bash
#SBATCH --job-name=nb-v2-erdc
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/nb_nmb_sota_v2/sub-08"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
export NB_ROOT OUT

mkdir -p "${OUT}" outputs/slurm
echo "{\"job\":\"erdc\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"started\":\"$(date -Iseconds)\"}" > "${OUT}/job_erdc_running.json"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
bash "${NB_ROOT}/scripts/nmb/run_nmb_sota_v2_erdc_sub08.sh"

echo "{\"job\":\"erdc\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_erdc_done.json"
