#!/usr/bin/env bash
# =============================================================================
# Submit the COMPLETE LOSO reconstruction experiment: all 10 folds.
#
# WHAT "COMPLETE LOSO" REQUIRES, AND WHY IT IS MORE THAN ONE JOB
# --------------------------------------------------------------
# A LOSO fold is not just "hold out subject f". Each fold needs its OWN encoder, because
# SAMGA's inter-subject protocol trains one model per held-out subject. So fold f needs,
# in order:
#
#   1. a SAMGA encoder trained to hold out f       (~50 min GPU, early-stopped ~epoch 31)
#   2. that encoder applied to the nine sources' train splits and to f's test split
#   3. a generation head trained on those sources only
#   4. conditioning vectors for f, then SDXL-Turbo generation, then the metric suite
#
# Ten folds therefore means ten encoder trainings plus ten reconstruction pipelines. Fold
# 8's encoder already exists (job 609116), so nine trainings remain.
#
# WHY THIS IS THREE JOBS AND NOT NINETEEN
# ---------------------------------------
# The account's QoS allows only 10 submitted and 8 running jobs at a time. One job per
# fold (9 trainings + 10 reconstructions = 19) can therefore never be fully submitted, and
# a drip-fed version would need someone to babysit the queue. Arrays count once against
# the submit limit, so the whole protocol fits in:
#
#   [clip] -> [samga-loso array, folds 1-7,9,10] -> [samgar-recon array, folds 1-10]
#
# The reconstruction array depends on the encoder array with `afterany`, not `afterok`.
# With `afterok`, a single failed fold would cancel ALL ten reconstructions; with
# `afterany` each fold proceeds and guards itself, so one bad encoder costs one fold
# instead of the experiment. The encoder array still gates correctly: `afterany` waits for
# the whole array to terminate before the reconstructions start.
#
# THE RACE THAT WOULD SILENTLY CORRUPT EVERYTHING
# ----------------------------------------------
# The CLIP conditioning targets are fold-independent but stored at shared filenames, and
# every fold needs the complete array (all 1654 training concepts for head supervision,
# all 16540 images as the generation neighbour gallery). If each fold extracted it, ten
# concurrent writers would race on two paths, and a loser reads a partial array without
# any error -- just wrong numbers. So exactly one dedicated job produces it, and the fold
# jobs depend on it. The producer itself skips if the arrays are already complete.
#
# Usage:
#   bash scripts/recon/submit_loso.sh              # everything
#   DRY_RUN=1 bash scripts/recon/submit_loso.sh    # print sbatch invocations only
# =============================================================================
set -euo pipefail

RECON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${RECON_ROOT}"

SEED="${SEED:-2025}"
DRY_RUN="${DRY_RUN:-0}"
CKPT_TAG="${CKPT_TAG:-checkpoint_last.pth}"
CLIP_DIR="${RECON_ROOT}/data/image_feature/clip_h14_ip_adapter"

mkdir -p outputs/recon outputs/slurm

sb() {                       # sb(...) -> job id on stdout
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] sbatch %s\n' "$*" >&2
    echo ""
  else
    sbatch --parsable "$@"
  fi
}

has_ckpt() {                 # has_ckpt <fold> -> 0 if that fold's encoder exists
  local p
  p=$(ls -1dt "${RECON_ROOT}/outputs/samga_official/inter/seed${SEED}"/*"sub-$(printf '%02d' "$1")"/"${CKPT_TAG}" \
      2>/dev/null | head -1 || true)
  [[ -n "${p}" ]]
}

# ------------------------------------------------------------------ 1. CLIP targets
echo "=== [1/3] shared CLIP conditioning targets ==="
CLIP_JOB=""
if [[ -f "${CLIP_DIR}/clip_h14_train.npy" && -f "${CLIP_DIR}/clip_h14_test.npy" ]]; then
  echo "[SKIP] already extracted at ${CLIP_DIR}"
else
  CLIP_JOB="$(sb --job-name=samgar-clip slurm/samgar_clip.sbatch)"
  echo "[OK] clip job = ${CLIP_JOB:-<dry-run>}"
fi

# ------------------------------------------------------------------ 2. encoders
echo
echo "=== [2/3] SAMGA LOSO encoders (array: folds 1-7,9,10) ==="
MISSING=""
for f in 1 2 3 4 5 6 7 9 10; do has_ckpt "${f}" || MISSING="${MISSING} ${f}"; done
MISSING="${MISSING# }"

SAMGA_JOB=""
if [[ -z "${MISSING}" ]]; then
  echo "[SKIP] all nine encoders present (fold 8 came from 609116)"
else
  echo "[INFO] folds needing training: ${MISSING}"
  SAMGA_JOB="$(sb --job-name=samga-loso \
    --export=ALL,SEED="${SEED}" \
    slurm/samga_loso_array.sbatch)"
  echo "[OK] SAMGA array job = ${SAMGA_JOB:-<dry-run>}"
fi

# ------------------------------------------------------------------ 3. reconstructions
echo
echo "=== [3/3] SAMGA-R reconstruction (array: folds 1-10) ==="
DEPS=""
for j in "${CLIP_JOB}" "${SAMGA_JOB}"; do
  [[ -n "${j}" ]] && DEPS="${DEPS:+${DEPS}:}${j}"
done

RECON_JOB="$(sb --job-name=samgar-recon \
  --array=0-9 \
  ${DEPS:+--dependency=afterany:${DEPS}} \
  --time=08:00:00 \
  --export=ALL,STAGE=all,SEED="${SEED}" \
  slurm/samga_recon.sbatch)"
echo "[OK] recon array job = ${RECON_JOB:-<dry-run>}  (deps: afterany:${DEPS:-none})"

echo
echo "=== queue ==="
squeue -u "${USER}" -o "%.10i %.14j %.10T %.8M %.22R %.38E" || true
echo
cat <<'EOF'
Track progress:
  sacct -j <JOBID> --format=JobID,JobName%16,State,Elapsed,ExitCode
  tail -f outputs/recon/recon-sub<FF>.log        # per-fold pipeline log
  tail -f outputs/slurm/samga-loso-<JOBID>_<idx>.log
Results land in outputs/recon/sub-<FF>/metrics/.
EOF
