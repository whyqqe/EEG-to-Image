#!/usr/bin/env bash
#SBATCH --job-name=nwret-diag
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-diag-%A_%a.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-diag-%A_%a.err
#
# Stage A of the overfitting diagnosis: six arms, one variable each, on sub-08.
# See scripts/run_diag_sweep.sh for what each arm is and why it is in the list.
#
# Why six in parallel rather than a sequence
# ------------------------------------------
# The arms are not a pipeline -- no arm consumes another's output, and they share
# only read-only inputs (the EEG cache and the frozen CLIP feature bank). Running
# them as one array means they finish together, which is what makes the comparison
# legitimate: same node type, same code revision, same day. A sequence would let the
# code drift between arms, which is precisely how the `prev` run's export came to
# look for a checkpoint the training had named differently.
#
# Cost
# ----
# Per arm: 60 epochs at batch 128 over 15040 samples, semantic tower only. The
# structure trunk is disabled (`--struct-backbone ""`), which removes the 304M-param
# decoder from the step -- the diagnosed run took 57 s/epoch with it. Plus a 200-way
# test pass every 5 epochs (12 extra forward passes over 200 samples, negligible
# next to an epoch). Expect well under an hour per arm against a 3 h wall clock; the
# margin is for the first arm's weight downloads and for the possibility that a
# frozen-backbone arm (`frz8`) behaves differently from expectation.
#
# Memory
# ------
# 96G, down from the dual-tower job's 128G, because the structure tower's activations
# were the reason for the larger figure. The EEG cache is memmapped and the CLIP
# feature bank is read once.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUBJ="${SUBJ:-8}"

# array index -> arm
ARMS=(ctl60 lr1e4 soft frz8 noaug b512)
i="${SLURM_ARRAY_TASK_ID:-0}"
export ARM="${ARMS[$i]}"

mkdir -p outputs/slurm outputs/logs

echo "########## job ${SLURM_JOB_ID:-local}.${i} arm=${ARM} sub-${SUBJ} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"

bash scripts/run_diag_sweep.sh
RC=$?

echo "########## job ${SLURM_JOB_ID:-local}.${i} arm=${ARM} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
