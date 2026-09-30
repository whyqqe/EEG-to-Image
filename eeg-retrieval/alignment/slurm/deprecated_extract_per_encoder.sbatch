#!/usr/bin/env bash
#SBATCH --job-name=nv-extract
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.err
#
# GPU stage: layer-wise frozen feature extraction.
# One job per encoder (TAG), see scripts/submit_pipeline.sh.
#
# NOTE on requeue + idempotency: the extractor skips existing .npy files, so a
# requeued job resumes rather than restarting.  This is why --requeue is safe.
set -euo pipefail

ROOT=/project/peilab/why/eeg-retrieval/alignment
PY=${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}
OUT="${ROOT}/outputs/extract"
mkdir -p "${OUT}" "${ROOT}/outputs/slurm" "${ROOT}/data/features"

: "${TAG:?TAG must be set (e.g. dinov2_l)}"
: "${KIND:?KIND must be set (open_clip|timm)}"
: "${MODEL:?MODEL must be set}"

echo "[job] id=${SLURM_JOB_ID:-local} host=$(hostname) tag=${TAG} model=${MODEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -c "import torch;assert torch.cuda.is_available(),'no CUDA';print('torch',torch.__version__,'gpu',torch.cuda.get_device_name(0))"

python_args=(
  "${ROOT}/scripts/extract_features.py"
  --out "${ROOT}/data/features"
  --tag "${TAG}"
  --kind "${KIND}"
  --model "${MODEL}"
  --split "${SPLIT:-train}"
  --batch "${BATCH:-32}"
)
[[ -n "${PRETRAINED:-}" ]] && python_args+=( --pretrained "${PRETRAINED}" )
[[ "${VERIFY:-0}" == "1" ]] && python_args+=( --verify )
[[ "${OVERWRITE:-0}" == "1" ]] && python_args+=( --overwrite )

echo "[cmd] ${PY} ${python_args[*]}"
"${PY}" "${python_args[@]}"

cat > "${OUT}/${TAG}_${SPLIT:-train}.done.json" <<EOF
{"tag":"${TAG}","model":"${MODEL}","split":"${SPLIT:-train}",
 "job_id":"${SLURM_JOB_ID:-local}","finished":"$(date -Iseconds)"}
EOF
echo "[job] done tag=${TAG}"
