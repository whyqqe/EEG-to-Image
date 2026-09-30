#!/usr/bin/env bash
#SBATCH --job-name=g2-eval
#SBATCH --account=peilab
# Partition note: this job runs on `preempt`, not `normal`. The array submission
# already consumed the normal-QOS per-user submit limit
# (QOSMaxSubmitJobPerUserLimit), and the array is the part that must get resources.
# Evaluation is short relative to training and its GT feature cache makes it
# restartable, so preemption costs little here.
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/g2-eval-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/g2-eval-%j.err

# Single evaluation pass over every fold.
#
# Why this is one job and not part of the array: eval_standard7.py writes a shared GT
# feature cache and one results.json. Ten concurrent folds would race on both, and the
# GT features (the expensive part) would be recomputed ten times. One pass shares the
# cache and removes the race.
set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export OUT="${OUT:-${NB_ROOT}/outputs/g2}"
export IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"

# Eval output dir is ISOLATED from outputs/standard7_protocol on purpose. The first
# G2 job (566625) still runs its own stage 5 against that shared path, and the two
# jobs' timelines overlap around dawn; eval_standard7.py rewrites results.json
# wholesale, so sharing the path would silently drop whichever job wrote first.
# The SOTA table is read (never written) for the comparison report instead.
STD7="${OUT}/standard7_eval"
SOTA="${NB_ROOT}/outputs/standard7_protocol"
MAN="${OUT}/manifest_g2_all.json"

source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" 2>/dev/null || true
cd "${NB_ROOT}"
mkdir -p "${OUT}/logs" "${STD7}"

echo "===== build manifest @ $(date -Iseconds) ====="
python scripts/nda/g2_build_eval_manifest.py \
  --out "${OUT}" --manifest "${MAN}" 2>&1 | tee "${OUT}/logs/manifest.log"

echo "===== standard-7 @ $(date -Iseconds) ====="
python scripts/nda/eval_standard7.py \
  --manifest "${MAN}" --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16 \
  2>&1 | tee "${OUT}/logs/eval_standard7_all.log"
cp -f "${STD7}/results.json" "${OUT}/results_g2_all.json"

# ---- pooled FID: the standard distribution-level number ----
# fake = all folds' generations concatenated, real = the 200 unique test images.
#
# eval_pooled_fid.py raises FileNotFoundError as soon as any sub-* directory lacks
# the requested tag, so a single failed fold would destroy the pooled number for
# every variant. Instead of touching that shared script (other pipelines depend on
# it), build a symlink farm containing only the COMPLETE folds and run it there. The
# metric implementation stays exactly the one already validated; only the fold set is
# filtered, and the surviving fold count is printed so a partial result is never
# mistaken for a full sweep.
echo "===== pooled FID @ $(date -Iseconds) ====="
for v in direct cfm0; do
  PROOT="${OUT}/pooled_root_${v}"
  rm -rf "${PROOT}"
  mkdir -p "${PROOT}"
  NFOLD=0
  for sub in "${OUT}"/multi/sub-*; do
    [[ -d "${sub}" ]] || continue
    g="${sub}/generation/${v}/generated"
    if [[ -f "${g}/199.png" ]]; then
      ln -sfn "$(cd "${sub}" && pwd)" "${PROOT}/$(basename "${sub}")"
      NFOLD=$((NFOLD + 1))
    else
      echo "[pooled] ${v}: fold $(basename "${sub}") incomplete, excluded"
    fi
  done
  echo "[pooled] ${v}: ${NFOLD} complete folds"
  if (( NFOLD >= 2 )); then
    python scripts/nda/eval_pooled_fid.py \
      --root "${PROOT}" --tag "${v}" --images-root "${IMAGES_ROOT}" \
      --output-json "${OUT}/fid_pooled_${v}.json" --device "${DEVICE}" \
      2>&1 | tee -a "${OUT}/logs/eval_pooled_fid.log" \
      || echo "[WARN] pooled FID failed for variant ${v}"
  else
    echo "[SKIP] pooled FID ${v}: only ${NFOLD} complete fold(s)"
  fi
done

# ---- comparison table: G2 rows alongside the existing SOTA rows ----
echo "===== comparison table @ $(date -Iseconds) ====="
python scripts/nda/g2_compare_sota.py \
  --g2 "${STD7}/results.json" \
  --sota "${SOTA}/results.json" \
  --fid-glob "${OUT}/fid_pooled_*.json" \
  --out "${OUT}/results_g2_vs_sota.json" \
  2>&1 | tee "${OUT}/logs/comparison.log" || echo "[WARN] comparison table failed"

echo "{\"pipeline\":\"g2_eval\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/eval_done.json"
echo "===== G2 evaluation complete @ $(date -Iseconds) ====="
