#!/usr/bin/env bash
#SBATCH --job-name=cogcap-loso
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/cogcap-loso-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/cogcap-loso-%j.err
#
# One LOSO fold of POLARIS on CogCapPro. See docs/DESIGN_POLARIS_inter_reconstruction.md.
#
# `--mem=128G` is not generous, it is the requirement: the nine source subjects' averaged
# training EEG is 9 x 16540 x 63 x 250 float32 = 9.4 GB, three condition sets add 5.5 GB,
# and the held-out subject's unlabelled train split adds 1 GB. Under 64G the job dies in
# `LOSOData.__init__` after the whole cache read.
#
# The recovery stage runs in the same job on purpose. It needs the frozen encoder, it is
# closed-form plus a short Cayley optimisation, and splitting it out would mean persisting
# 16540 x 1024 activations per modality for no benefit.
set -euo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
PY="/project/peilab/why/eeg-brainit/.venv/bin/python"

cd "${ROOT}"
mkdir -p outputs/slurm outputs/cogcap

# The `cogcap` package lives under `scripts/`; cwd here is the repository root.
export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

TARGET="${TARGET:-8}"
SOURCES="${SOURCES:-1 2 3 4 5 6 7 9 10}"
MODALITIES="${MODALITIES:-image,depth,edge}"
OUT="${OUT:-outputs/cogcap/polaris_sub$(printf '%02d' "${TARGET}")}"

echo "=== host $(hostname) date $(date -Is) ==="
echo "target=${TARGET} sources=${SOURCES} modalities=${MODALITIES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Fail early and loudly rather than after the 42 GB cache read.
for m in ${MODALITIES//,/ }; do
  if [[ "${m}" != "image" ]]; then
    f="data/image_feature/cogcap_${m}/${m}_train.npy"
    [[ -f "${f}" ]] || { echo "[FATAL] missing ${f}; run slurm/cogcap_prep.sbatch first" >&2; exit 1; }
  fi
done

EXTRA=()
[[ "${NO_SEA:-0}" == "1" ]]      && EXTRA+=(--no-sea)
[[ "${NO_REPAIR:-0}" == "1" ]]   && EXTRA+=(--no-repair-topk)
[[ "${NO_RECOVERY:-0}" == "1" ]] && EXTRA+=(--no-recovery)
[[ "${NO_SAW:-0}" == "1" ]]      && EXTRA+=(--no-saw)
[[ "${NO_H1:-0}" == "1" ]]       && EXTRA+=(--no-h1)
[[ -n "${EPOCHS:-}" ]]           && EXTRA+=(--epochs "${EPOCHS}")
[[ -n "${PAIR_MODE:-}" ]]        && EXTRA+=(--pair-mode "${PAIR_MODE}")
[[ -n "${EXTRA_ARGS:-}" ]]       && EXTRA+=(${EXTRA_ARGS})

echo "=== launching ==="
"${PY}" -m cogcap.train \
    --sources ${SOURCES} \
    --target "${TARGET}" \
    --modalities "${MODALITIES}" \
    --out-dir "${OUT}" \
    --batch-size "${BATCH_SIZE:-1024}" \
    --lr "${LR:-1e-4}" \
    --weight-decay "${WD:-0.01}" \
    "${EXTRA[@]}"

echo "=== done $(date -Is) ==="
