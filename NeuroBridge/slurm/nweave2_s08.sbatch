#!/bin/bash
#SBATCH --job-name=nweave2_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nweave2_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nweave2_s08_%j.err
#
# TIME BUDGET (round 1 measured 7 arms x 40 ep in ~3 min, so this is generous)
#   * 7 arms x 40 epochs                 ~7 min   (anytime_consist does 2 forwards)
#   * anytime eval, 7 arms x 4 windows   ~4 min
#   * 13-route probe x 7 arms, 40 ep    ~15 min   (round 1: 3 routes = 26 s)
#   * cycle, 3 spaces x 3 sets x 200 img ~6 min   (DINOv2 ViT-L + RN50 + CLIP)
#   => ~35 min.  4 h wall leaves room for a slow node and for resume.

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export NWEAVE2_OUT=/project/peilab/why/NeuroBridge/outputs/neuroweave2/sub-08
export ARMS="${ARMS:-frozen,lora,anytime_train,anytime_soft,anytime_curric,anytime_consist,hier_aux}"
export EPOCHS="${EPOCHS:-40}"
export PROBE_EPOCHS="${PROBE_EPOCHS:-40}"
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
mkdir -p outputs/slurm "${NWEAVE2_OUT}/logs"

echo "[slurm] NeuroWeave round 2 ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[slurm] ARMS=${ARMS} EPOCHS=${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

set +e
bash scripts/nda/run_neuroweave2_s08.sh
rc=$?
set -e
echo "[slurm] nweave2 rc=${rc} @ $(date -Iseconds)"
echo "[slurm] Re-submit resumes: finished arms/probes/cycle are skipped."
exit "${rc}"
