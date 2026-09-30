#!/usr/bin/env bash
#SBATCH --job-name=epd-sem-final
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-sem-final-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-sem-final-%j.err
#
# =============================================================================
# ONE complete run of the optimal semantic configuration, end to end.
# =============================================================================
# `ARM=A0` is the whole of `arm_flags` being empty, and that is not a default left
# unexamined -- it is the configuration the evidence points at:
#
#   * A0 is the best semantic arm measured, at 50.83 test Top-1 over three seeds
#     (52.5/51.0/49.0), against 40.00 for the arm that was labelled "EEGiT exactly"
#     and 27.83 for `--fixed-temp --softplus`. Both of those PIN the temperature;
#     A0 LEARNS it. Pinning it hurts at either end of the range (2.727 or 457), so
#     the binding term is the pinning, not the value, and A0 is what remains.
#   * It is semantic-only by construction: no `--struct-*` flag appears anywhere in
#     `run_epd_sem.sh`, which is the architecture claim -- the depth-ControlNet
#     branch is out of the generation path.
#   * Generation is txt2img with the ControlNet at scale 0, so the ONLY input that
#     varies is whose EEG built the IP-Adapter condition.
#
# What this run adds over the existing A0 artefacts
# ------------------------------------------------
# The tag is different on purpose, because there are two things the recorded A0 could
# not say and this run can:
#
#   1. The temperature the selected checkpoint was actually trained under. `InfoNCE`
#      is a separate module, so `logit_scale` was never in `model.state_dict()`;
#      recovering A0's learned scale required inferring it from its LayerNorm gammas.
#      `train.py` now writes it to the checkpoint and the result json, and step [6]
#      prints it. This is the number that decides whether "the same loss" is true.
#   2. A self-contained artefact set. Every path here derives from the one tag, so the
#      checkpoint, export, 200 generations and seven metrics cannot be assembled from
#      different runs by mistake.
#
# Same `--seed 2025` and same `--split-seed 2025` as every scored run, so the
# per-concept vectors pair positionally against A0 and against the EEGiT arms.
#
# Submit:  sbatch slurm/epd_sem_final.sbatch
# Reproduce by hand on any node:
#   ARM=A0 SEED=2025 TAG=epd_sem_final_s2025 bash scripts/run_epd_sem.sh

set -uo pipefail
cd /project/peilab/why/eeg-retrieval

# One shared cache root, set BEFORE anything imports torch or timm: a download that
# starts mid-run stalls on a filesystem lock and looks like a hang in the model, and
# `/home` is out of space.
export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export TOKENIZERS_PARALLELISM=false
export SLURM_CPU_BIND=none

export ARM="A0"
export SEED="2025"
export TAG="epd_sem_final_s2025"

mkdir -p outputs/slurm outputs/logs

echo "########## epd-sem-final | job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# The wiring suite, first: this run depends on a change to `train.py` (the criterion
# block), and 171 checks in ~35s is cheaper than discovering a broken result json
# after 35 minutes of GPU.
"${PYTHON}" -u scripts/test_epd_arch.py || {
  echo "[FATAL] wiring tests failed; not starting training"; exit 2; }

# Then the flag list, with `--validate-only` (no dataset, no GPU).
ARM="${ARM}" SEED="${SEED}" TAG="${TAG}" bash scripts/run_epd_sem.sh --dry-run || {
  echo "[FATAL] dry run failed"; exit 2; }

LOG="outputs/logs/run_epd_sem_final_s2025_${SLURM_JOB_ID:-local}.log"
echo "########## training + export + generation + metrics -> ${LOG} @ $(date -Iseconds) ##########"
# `set -o pipefail` is scoped to the subshell so `tee` cannot mask the exit code.
if ( set -o pipefail; ARM="${ARM}" SEED="${SEED}" TAG="${TAG}" \
       bash scripts/run_epd_sem.sh 2>&1 | tee "${LOG}" ); then
  echo "########## complete @ $(date -Iseconds) ##########"
  exit 0
fi
rc=$?
echo "########## FAILED rc=${rc} @ $(date -Iseconds) ##########"
exit "${rc}"
