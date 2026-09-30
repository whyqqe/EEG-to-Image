#!/usr/bin/env bash
#SBATCH --job-name=nb-scc
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/nb_scc/sub-08"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
export NB_ROOT OUT DEVICE=cuda:0 PROBE_SAMPLES=2000

mkdir -p "${OUT}" outputs/slurm
echo "{\"job\":\"scc\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"started\":\"$(date -Iseconds)\"}" > "${OUT}/job_running.json"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
bash "${NB_ROOT}/scripts/nmb/run_scc_sub08.sh"

echo "{\"job\":\"scc\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
