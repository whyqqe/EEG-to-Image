#!/usr/bin/env bash
#SBATCH --job-name=epd-sem
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-sem-%A_%a.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-sem-%A_%a.err
#SBATCH --array=0-4
#
# One array task per ARM; the three seeds run sequentially inside the task.
#
# Why not one task per (arm, seed): the account's QOS allows MaxSubmitJobs=10, and
# SLURM counts each array element against that, so the natural 15-element array is
# rejected outright ("QOSMaxSubmitJobPerUserLimit"). Five tasks stay under both the
# submit cap and the 8-job running cap, use five GPUs concurrently, and cost about
# 3 x ~45 min = 2.5 h of wall clock. A single-seed rerun is still reproducible: it is
# `SEED=<s> ARM=<a> scripts/run_epd_sem.sh` on any node.
#
# A failing seed does NOT abort the arm. The three seeds are independent runs of the
# same configuration, so one node hiccup should cost one seed, not the other two. The
# task still exits non-zero if any seed failed, so a partially-failed arm cannot be
# mistaken for a complete one in `squeue`.
#
# Resource sizing: the semantic trunk is 92M parameters with no structural trunk (the
# recorded DA2 run spent 43 min on 100 epochs WITH one). Export ~2 min, 200 generations
# ~6.5 min, seven metrics ~1 min. Three seeds = ~2.5 h; 5 h covers a slow node.

set -uo pipefail
cd /project/peilab/why/eeg-retrieval

# One shared cache root, so nothing reaches the network mid-run. A download that
# starts during training stalls the loop on a filesystem lock and looks, in the log,
# like a hang in the model.
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

# All arms write their metrics into ONE directory so the ladder can be summarised
# without globbing per-task directories. The tags are distinct, so nothing collides.
export MET="${MET:-/project/peilab/why/eeg-retrieval/outputs/sub08/epd_sem_metrics}"

# (arm, seed) is (index, element of SEEDS). `--split-seed` stays 2025 for every run --
# see the runner's header: the arms must be scored on identical val/test concepts or
# the per-concept vectors cannot be paired.
ARMS=(A0 A1 A2 A3 A4)
SEEDS=(2025 2026 2027)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
ARM="${ARMS[$IDX]:?array index $IDX is outside the arm table}"

mkdir -p outputs/slurm outputs/logs "${MET}"

echo "########## task ${IDX} = ARM=${ARM} | job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Fail in seconds rather than after the GPU allocation. The dry run validates the flag
# list with `--validate-only` (no dataset, no GPU) and checks that every script the
# pipeline calls exists.
ARM="${ARM}" SEED="${SEEDS[0]}" bash scripts/run_epd_sem.sh --dry-run || {
  echo "[FATAL] dry run failed for ARM=${ARM}"; exit 2; }

failed=()
for SEED in "${SEEDS[@]}"; do
  LOG="outputs/logs/run_epd_sem_${ARM}_s${SEED}_${SLURM_JOB_ID:-local}.log"
  echo "########## ${ARM} seed ${SEED} -> ${LOG} @ $(date -Iseconds) ##########"
  # `set -o pipefail` is scoped to the subshell so `tee` cannot mask the runner's
  # exit code, and the `if` keeps one bad seed from ending the arm. `rc` is captured
  # on the first line of the branch: `$?` would be stale by the time the `echo` runs.
  rc=0
  if ( set -o pipefail; ARM="${ARM}" SEED="${SEED}" bash scripts/run_epd_sem.sh 2>&1 \
         | tee "${LOG}" ); then
    echo "########## ${ARM} seed ${SEED} OK @ $(date -Iseconds) ##########"
  else
    rc=$?
    echo "########## ${ARM} seed ${SEED} FAILED rc=${rc} @ $(date -Iseconds) ##########"
    failed+=("${SEED}(rc=${rc})")
  fi
done

if (( ${#failed[@]} )); then
  echo "########## task ${IDX} ARM=${ARM} finished with FAILED seeds: ${failed[*]} ##########"
  exit 1
fi
echo "########## task ${IDX} ARM=${ARM} complete, all seeds OK @ $(date -Iseconds) ##########"
exit 0
