#!/usr/bin/env bash
#SBATCH --job-name=epd-da2-depth8
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-da2-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-da2-%j.err
#
# EEG Patch, the Depth Anything V2 structural arm, on sub-08.
#
# One training arm and five generation arms. The training arm is single because
# the design was chosen by measurement rather than by search -- the coarse
# reachability ladder and its centred re-run in `probe_targets.py` -- and the
# generation arms are cheap (200 images x 28 steps each) while the training is
# not, so the sweep belongs where the cost is near zero.
#
# The five generation arms and the two differences that carry the conclusion:
#
#   sem_only     txt2img CN 0.00 IP(deploy)   semantic tower alone
#   depth_cn035  txt2img CN 0.35 IP(deploy)   + the depth condition
#   depth_cn070  txt2img CN 0.70 IP(deploy)   + the depth condition
#   null_txt2img txt2img CN 0.00 IP(noise)    no EEG in the semantic path
#   null_cn070   txt2img CN 0.70 IP(noise)    CN on, map from zeroed EEG
#
#   depth_cn - sem_only          prices the structural branch
#   null_cn  - null_txt2img      prices the structural CONDITION
#
# The second pair is what makes the first attributable. The fit-set mean field is
# a plausible scene prior -- "far at the top, near at the bottom" -- and it is the
# same map for every sample, so a depth conditioning image built mostly from it
# would improve every arm including the one driven by zeroed EEG. Without the
# null pair a positive first difference cannot be assigned to the EEG.
#
# Why 8x8 and why centred: see the header of `scripts/run_epd_da2.sh`.
#
# Time: 100 epochs at 70 + 70 tokens with a 25M structural trunk and a 92M
# semantic trunk, batch 128 fp32; then export (~10 min), 5 generation arms
# (~50 min) and 5 seven-metric evaluations (~30 min). Budgeted at 6 h, asking 8.
#
# `--time 08:00:00` covers the generation and metric stages, which cannot be
# resumed from a checkpoint but ARE resumable at arm granularity: both loops skip
# any arm whose 199.png or <arm>_seven.json is already on disk, so a resubmit
# continues rather than restarting.

set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
cd "${ROOT}"

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

# A single shared cache root, so nothing reaches the network mid-run. A download
# that starts during training would stall the loop on a filesystem lock and look,
# in the log, like a hang in the model.
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

echo "########## job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# The dry run first. It is flag-level only -- no dataset, no GPU, no cache reads --
# so it fails in seconds on the node rather than after the run has already spent
# its allocation discovering a typo. It also re-checks that every target cache the
# config names exists, which a login-node check cannot do for the LUSTRE paths this
# job actually reads.
bash scripts/run_epd_da2.sh --dry-run || { echo "[FATAL] dry run failed"; exit 2; }

LOG="outputs/logs/run_epd_da2_${SLURM_JOB_ID:-local}.log"
echo "########## full pipeline -> ${LOG} ##########"

set -o pipefail
bash scripts/run_epd_da2.sh 2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}

echo "########## job ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
