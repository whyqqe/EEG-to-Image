#!/usr/bin/env bash
#SBATCH --job-name=nv-a3b
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.err
#
# =============================================================================
# A3b / A6 -- reproducible dimensionality and capacity-corrected coverage
# =============================================================================
# WHAT THIS JOB IS FOR
# --------------------
# The first version of the dimensionality estimator was wrong in a way that only
# running it could reveal:
#
#   r = PR(psd(S_obs - S_noise/T))  tracked the analysis dimension budget instead
#   of saturating (23% of D at D=100 ... 19% at D=1500), and a held-out screening
#   count behaved the same way (30 at D=150 -> 130 at D=900).  The circular-shift
#   control then reported MORE "reproducible" dimensions than the real data
#   (locked = -20, -50, -70), which proves the counted directions are not
#   stimulus-locked.  So the job does not report a single number; it reports the
#   full D-sweep plus the stimulus-locked contrast, and lets the slope decide.
#
#   (2) CAPACITY-CORRECTED COVERAGE (Claim 2, Prop. 5a) IS the robust half: with
#       n=1654 and q=1024 a RANDOM model already covers ~62% of the reproducible
#       subspace, so only the EXCESS over a shuffled-feature control counts.  The
#       analytic null r*m/n matches the shuffled control to three decimals, and
#       the excess is ~1.7 dims out of ~50 (p=0.0000).
#
# WHY CPU, NOT GPU
# ----------------
# This is thousands of SMALL eigen/SVD solves (d <= 1400) inside long Python
# loops, plus memory-bandwidth-bound chunked products.  LAPACK eigh(1400) is
# 0.29 s single-threaded; a GPU would lose the per-call launch overhead and the
# batched-eigh path is not faster at this size.  The real lever is that the 10
# subjects are independent -- hence --jobs.
#
# RUNTIME / RESOURCES
# -------------------
# Peak RSS ~9 GB per worker: the 4 GB float32 source array, the window copy, and
# a CHUNKED float64 residual product.  peak RSS is 9 GB only because
# prep_projection was made memory-lean (the obvious version promoted the whole
# 8.3 GB float64 array and was also 2x slower).  --jobs is therefore bounded by
# memory: 8 workers x 9 GB.  BLAS is pinned to one thread per worker because the
# dominant operation is single-threaded LAPACK anyway, and letting 8 processes
# each spawn 8 threads would only thrash.
# =============================================================================
set -uo pipefail

ROOT=/project/peilab/why/eeg-retrieval/alignment
cd "$ROOT"
mkdir -p outputs/slurm outputs/a3b

# one BLAS thread per worker: see the note above
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=============================================================="
echo "nv-a3b  job=${SLURM_JOB_ID:-local}  node=$(hostname)  $(date -Is)"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}MB  threads/worker=1"
echo "=============================================================="

# NOTE: --jobs 8 over 10 subjects means two waves, so wall clock ~= 2x the
# slowest subject.  Raising --jobs needs memory: 9 GB per worker.
time python -u scripts/a3b_reproducible_dim.py \
  --out outputs/a3b \
  --subs sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10 \
  --which train \
  --windows w0_000_100ms w1_100_200ms w2_200_300ms w3_300_500ms \
            w4_500_800ms w5_800_1000ms full_000_800ms full_000_1000ms \
  --clips ViT-H-14 RN50 \
  --d-grid 400 150 300 900 1400 \
  --shrink 0.05 \
  --n-rep 8 --n-sim-null 200 --n-boot 200 --n-perm 200 --n-shift 2 \
  --jobs 8
rc=$?

echo "=============================================================="
echo "nv-a3b finished rc=$rc  $(date -Is)"
echo "--- verdict lines ---"
grep -E "VERDICT_LINE|CLAIM1_|CLAIM2_|slope_" outputs/slurm/nv-a3b-${SLURM_JOB_ID}.out 2>/dev/null | tail -10
echo "=============================================================="
exit $rc
