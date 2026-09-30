#!/usr/bin/env bash
#SBATCH --job-name=epd-sem-eegit
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-sem-eegit-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-sem-eegit-%j.err
#
# ONE task, ONE arm: EEGiT's objective, trained and scored end to end.
#
# Why this arm
# ------------
# `TAU_L2` is `--fixed-temp --no-eeg-l2norm`, i.e. BOTH elements of the released
# objective: tau pinned at 0.07 (scale 14.29, the `exp` path the codebase already
# uses) and only the image side L2-normalised, so the EEG embedding's norm stays free
# as a learned per-sample logit scale. `run_epd_sem.sh`'s A3 was previously read as
# "EEGiT's loss temperature", but it is `--fixed-temp --softplus`, and `--softplus`
# reproduces SAMGA's temperature (`softplus(log(1/0.07))` = 2.727, tau 0.367) which
# `losses.py` says in as many words. EEGiT's paper fixes tau at 0.07 "as in CLIP".
# So the combination in this job has never been run here.
#
# Why seed 2025
# -------------
# The recorded `A0` control at seed 2025 already has its checkpoint, export, 200
# generations and seven metrics, scored under the same Pearson protocol from the same
# `--split-seed 2025` split. Running this arm at the SAME seed makes the comparison a
# same-seed paired A/B: identical concepts, identical generator settings, identical
# generator seed 42, differing only in the two loss flags. A fresh seed would add
# seed noise to a comparison that can otherwise be exact.
#
# Why one job and not an array
# -----------------------------
# One arm, one seed, one task. The aggregation is CPU-only (JSON plus two .npy reads
# and a 10000-resample bootstrap over 200-length vectors), so it runs inline at the end
# of this same job rather than as a second submission -- there is nothing to wait for
# and no reason to spend a queue slot on it.
#
# Resource sizing: the semantic trunk is 92M parameters with no structural branch.
# Train ~30 min (1807 s recorded), export ~2 min, 200 generations ~7 min, seven metrics
# ~1 min, aggregation ~10 s. 4 h covers a slow node with room to spare.

set -uo pipefail
cd /project/peilab/why/eeg-retrieval

# One shared cache root, so nothing reaches the network mid-run. A download that starts
# during training stalls the loop on a filesystem lock and reads, in the log, as a hang
# in the model.
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

export MET="${MET:-/project/peilab/why/eeg-retrieval/outputs/sub08/epd_sem_eegit_metrics}"
ARM=TAU_L2
SEED=2025

mkdir -p outputs/slurm outputs/logs "${MET}"

echo "########## job ${SLURM_JOB_ID:-local} ARM=${ARM} SEED=${SEED} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Fail in seconds rather than after the GPU allocation. The dry run validates the flag
# list with `--validate-only` (no dataset, no GPU) and checks that every script the
# pipeline calls exists -- including the two that live in the NeuroBridge checkout.
if ! ARM="${ARM}" SEED="${SEED}" bash scripts/run_epd_sem_eegit.sh --dry-run; then
  echo "[FATAL] dry run failed for ARM=${ARM}"; exit 2
fi

LOG="outputs/logs/run_epd_sem_eegit_${ARM}_s${SEED}_${SLURM_JOB_ID:-local}.log"
echo "########## train/export/generate/score -> ${LOG} @ $(date -Iseconds) ##########"
# `set -o pipefail` is scoped to the subshell so `tee` cannot mask the runner's exit
# code. `rc` is captured on the first line of the branch: `$?` would be stale by the
# time the `echo` runs.
rc=0
if ( set -o pipefail; ARM="${ARM}" SEED="${SEED}" bash scripts/run_epd_sem_eegit.sh 2>&1 \
       | tee "${LOG}" ); then
  echo "########## ${ARM} seed ${SEED} pipeline OK @ $(date -Iseconds) ##########"
else
  rc=$?
  echo "########## ${ARM} seed ${SEED} pipeline FAILED rc=${rc} @ $(date -Iseconds) ##########"
fi

# ---------------------------------------------------------------------------
# aggregation -- inline, same job
# ---------------------------------------------------------------------------
# Runs even when the pipeline failed, because a partial run still has a checkpoint
# and a seven-metric file worth reporting, and `sem_eegit_analyze.py` reports an arm it
# cannot find as absent rather than imputing it. The exit code stays the pipeline's:
# the summary is not allowed to mask a failed run.
set +e
"${PYTHON}" -u scripts/epd/sem_eegit_analyze.py \
  --root /project/peilab/why/eeg-retrieval \
  --out-dir /project/peilab/why/eeg-retrieval/outputs/sub08 \
  --out-file /project/peilab/why/eeg-retrieval/outputs/sub08/epd_sem_eegit.md \
  --resamples 10000
sum_rc=$?
echo "########## aggregation rc=${sum_rc} @ $(date -Iseconds) ##########"
[[ ${sum_rc} -ne 0 ]] && echo "[warn] the report could not be written; see the trace above"

exit "${rc}"
