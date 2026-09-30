#!/usr/bin/env bash
#SBATCH --job-name=epd-probe-inter
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-probe-inter-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-probe-inter-%j.err
#
# =============================================================================
# epd-probe-inter -- re-measure the target layer UNDER THE INTER PROTOCOL
# =============================================================================
# This is PROTOCOL_INTER.md section 8's open question, and it is a prerequisite for
# freezing the architecture rather than an optional extra.
#
# The contradiction it exists to resolve: Shallow Alignment reports that for ViT-H-14
# the best inter-subject alignment layer is L11 (32.3% relative depth, inter Top-1
# 19.0 / Top-5 44.2), against 7.8 / 23.5 for the final output -- a bigger swing than
# any other design choice in that table. But this project's own probe peaks at
# block24/26 (val 17.07 / 17.33, test 28.50 / 27.50) and finds block11 at test 16.50.
# The two disagree about which end of the depth curve to use, and we are currently
# training against block26 on the strength of the probe that disagrees with the paper.
#
# The disagreement is explainable, and the explanations are testable:
#   * Shallow Alignment probed "approximately ten evenly spaced intermediate layers",
#     so L11 may be the best AMONG PROBED rather than the true peak -- which a denser
#     sweep distinguishes, and a curve is what makes that visible.
#   * ridge and a trained encoder can prefer different depths, so a probe cannot
#     settle it for a trained model. But it CAN settle which layer carries more
#     linearly-decodable inter-subject signal, which is the question that decides
#     where the curve's mass is.
#   * and the most likely explanation of all: our probe is INTRA-subject. The claim is
#     an INTER-subject claim. Section 8 says so outright -- "it is an inter-subject
#     question ... it must be re-measured with 9 training subjects". That is this job.
#
# Cost: two eigen-decompositions of a 15750 x 15750 Gram matrix (one on all nine
# sources' rows, one on the 90% used to select lambda) plus one matmul per layer. The
# Gram is accumulated in chunks so the 9 GiB fit matrix never lands on the GPU whole.
# Minutes, not hours -- which is why it runs alongside the training job rather than
# gating it.
#
# The sweep covers both ends of the depth curve (9..31), the two layers our probe and
# the paper each nominate, and four fusion groups. Fusion is included because it is
# the one place the two sources AGREE -- Shallow Alignment: L28 82.6 -> {L24,L28} 87.8
# -> adaptive 91.3; SAMGA's ablation: fixed 87.8 / uniform 89.1 / learned 91.3. So
# "which single layer" may be the wrong question, and this measures that too.
set -euo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export PYTHONUNBUFFERED=1
export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"

mkdir -p outputs/slurm outputs/probe

echo "########## epd-probe-inter ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -g | head -2

# The MVNN whiteners must already exist: this probe reads the same per-role
# standardised arrays the training run does, so the two see the same signal.
for s in 1 2 3 4 5 6 7 9 10; do
  [[ -f "outputs/cache/mvnn_W_sub$(printf '%02d' "$s")_all63_train_lw.npy" ]] \
    || { echo "missing MVNN whitener for sub-$s; run scripts/epd/build_mvnn_cache.py"; exit 1; }
done
[[ -f "outputs/cache/mvnn_W_sub08_all63_test_lw.npy" ]] \
  || { echo "missing MVNN whitener for held-out sub-08"; exit 1; }

"${PYTHON}" -u scripts/epd/probe_inter.py \
  --target-subject 8 \
  --channels all \
  --mvnn train \
  --layers 9 11 13 15 17 19 21 24 26 28 31 \
  --fusion "17+24" "24+26" "17+24+26" "11+17+24+26" \
  --lams 1e1 1e2 1e3 1e4 1e5 1e6 1e7 \
  --val-frac 0.1 \
  --out outputs/probe/probe_inter_tgt08.json
RC=$?

echo "########## epd-probe-inter rc=${RC} @ $(date -Iseconds)"
exit "${RC}"
