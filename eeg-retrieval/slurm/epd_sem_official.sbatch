#!/usr/bin/env bash
#SBATCH --job-name=epd-sem-official
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-sem-official-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-sem-official-%j.err
#
# One task, one arm: EEGiT's four-item design, trained on sub-08, then scored on the
# seven standard metrics through a pure IP-Adapter generator (no ControlNet, no depth
# map). See `scripts/run_epd_sem_official.sh` for what the four items are and for the
# one documented deviation (the target space).
#
# The gate before the GPU work is `test_eegit_official_interface.py`, which is the
# piece of this project that can actually fail: it asserts our EEG patch tensor and
# our antialias-resampled `pos_embed` match the official ones, rather than merely
# having the same shape. Everything downstream assumes that test passes.
#
# Submit:  sbatch slurm/epd_sem_official.sbatch

set -uo pipefail
cd /project/peilab/why/eeg-retrieval

# Set BEFORE anything imports torch or timm: a download that starts mid-run stalls on
# a filesystem lock and presents as a hang in the model, and `/home` has no space.
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

mkdir -p outputs/slurm outputs/logs

echo "########## epd-sem-official | job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# [gate 1] the official-interface equivalence test. Cheap, CPU, and the only thing
# standing between "we call it EEGiT" and "it is EEGiT".
echo "########## gate: official EEGiT interface ##########"
"${PYTHON}" -u scripts/test_eegit_official_interface.py || {
  echo "[FATAL] test_eegit_official_interface.py failed; the config is not EEGiT's"; exit 2; }

# [gate 2] the architecture wiring suite, which now also covers the criterion block.
echo "########## gate: epd wiring tests ##########"
"${PYTHON}" -u scripts/test_epd_arch.py || {
  echo "[FATAL] wiring tests failed"; exit 2; }

# [gate 3] the flag list itself, with `--validate-only` (no dataset, no GPU).
SUBJ=8 SEED=2025 TAG=epd_sem_official_s2025 bash scripts/run_epd_sem_official.sh --dry-run || {
  echo "[FATAL] dry run failed"; exit 2; }

LOG="outputs/logs/run_epd_sem_official_s2025_${SLURM_JOB_ID:-local}.log"
echo "########## train + export + generate + metrics -> ${LOG} @ $(date -Iseconds) ##########"
# `set -o pipefail` scoped to the subshell so `tee` cannot mask the runner's exit code.
if ( set -o pipefail; SUBJ=8 SEED=2025 TAG=epd_sem_official_s2025 \
       bash scripts/run_epd_sem_official.sh 2>&1 | tee "${LOG}" ); then
  echo "########## complete @ $(date -Iseconds) ##########"
  exit 0
fi
rc=$?
echo "########## FAILED rc=${rc} @ $(date -Iseconds) ##########"
exit "${rc}"
