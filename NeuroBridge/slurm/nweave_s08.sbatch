#!/bin/bash
#SBATCH --job-name=nweave_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nweave_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nweave_s08_%j.err
#
# TIME BUDGET (measured analogues, not guesses)
#   * 6 arms x 40 epochs          ~ 6 * 12 min  = ~72 min  (cfmsf_fix-like)
#   * anytime eval (6 x 4 wins)   ~ 15 min
#   * quick probe 3 routes x 40ep ~ 6 * 8 min   = ~48 min
#   * cycle score (CLIP encode)   ~ 10-20 min
#   => ~2.5 h typical; 8 h wall for slow nodes / resume.

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export NWEAVE_OUT=/project/peilab/why/NeuroBridge/outputs/neuroweave/sub-08
export ARMS="${ARMS:-frozen,lora,direct,multi_head,anytime_train,causal_stage}"
export EPOCHS="${EPOCHS:-40}"
export PROBE_EPOCHS="${PROBE_EPOCHS:-40}"
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
mkdir -p outputs/slurm "${NWEAVE_OUT}/logs"

echo "[slurm] NeuroWeave Stage-1 ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[slurm] ARMS=${ARMS} EPOCHS=${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

set +e
bash scripts/nda/run_neuroweave_s08.sh
rc=$?
set -e
echo "[slurm] nweave rc=${rc} @ $(date -Iseconds)"
echo "[slurm] Re-submit resumes: finished arms/probes/cycle are skipped."
exit "${rc}"
