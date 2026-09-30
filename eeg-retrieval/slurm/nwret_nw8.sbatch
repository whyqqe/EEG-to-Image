#!/usr/bin/env bash
#SBATCH --job-name=nwret-nw8
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-nw8-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-nw8-%j.err
#
# ONE configuration for sub-08. See scripts/run_nw8.sh for what is fixed and why.
#
# Why one run and not the arm sweep
# ---------------------------------
# The sweep was ranking hyperparameters on top of a broken input interface: the
# EEG tokenizer learned a randomly-initialised projection on per-cell waveforms
# and never called the pretrained patch_embed at all. Every arm sat in a 16-21.5
# band (33.5-38.5 after the alignment layer was corrected) while a closed-form
# linear ridge reached 25.0 on the same features and split -- i.e. some arms were
# below a linear model. Tuning above a broken interface buys noise, so this job
# fixes the interface and spends the single shot there.
#
# Cost (measured, not estimated)
# ------------------------------
# 100 epochs x 29 steps/epoch = 2900 steps. Benchmarked on an H800: 443 ms/step at
# batch 512 (23.7 GB), against 1240 ms/step for the old 196-token grid path -- the
# predicted 2.8x saving from 70 tokens instead of 196, empirically confirmed.
# So ~21 min of GPU for the whole schedule, plus validation, the smoke gate, the
# unit tests and the fit diagnostic. The 2h wall clock is ~3x margin; a 4h request
# would only hold the node longer.
#
# Note batch 512 rather than EEGiT's 1024: batch 1024 measured 857 ms/step, i.e.
# the SAME total time for half the optimisation steps (1500 vs 2900). More steps at
# the same cost is strictly better here, since B1 is that the model was optimised
# for only ~800 steps.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUBJ="${SUBJ:-8}"

# Model weights live on /project, NOT on /home: /home is at 100% capacity and a
# download attempt there fails with ENOSPC mid-run. HF_HUB_OFFLINE is left unset
# so a genuinely missing artifact surfaces as a clear download error rather than a
# confusing cache miss, but the ViT backbone is already cached here.
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"

mkdir -p outputs/slurm outputs/logs

echo "########## job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

bash scripts/run_nw8.sh
RC=$?

echo "########## job ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
