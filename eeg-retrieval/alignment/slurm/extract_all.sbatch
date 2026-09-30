#!/usr/bin/env bash
#SBATCH --job-name=nv-extract-all
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
# =============================================================================
# ONE GPU JOB that extracts all frozen encoders, sequentially.
# =============================================================================
# Encoders are FROZEN instruments: never trained, never see EEG.  This is what
# makes the NV interpretation valid -- candidate axes must come from a source
# statistically independent of the EEG, or the analysis is circular.
#
# Every transformer block is extracted, not only the final projection: a CLIP
# image tower's final projection has already been contrastively compressed onto
# the text-aligned manifold, and the layer profile must stay free to show which
# depth is most neurally visible.
#
# WHY ONE JOB RATHER THAN FIVE
#   The five encoders share a single GPU and a single model cache, and each takes
#   ~5-20 min.  Sequential extraction in one job avoids five queue entries and
#   five separate GPU allocations for what is one logical step.
#
# IDEMPOTENCY / REQUEUE SAFETY
#   extract_features.py skips any layer_*.npy that already exists, so a requeued
#   job resumes rather than restarting.  That is what makes --requeue safe here.
#
# ORDERING
#   Submit with --dependency=afterok:<gate job>.  If the gates fail, the features
#   would be validating a pipeline that is already known to be void, so they must
#   not be produced.  This is enforced by the dependency rather than by hand.
#
# Usage:
#   sbatch --dependency=afterok:<GATE_JOB_ID> slurm/extract_all.sbatch
# =============================================================================
set -uo pipefail

ROOT=/project/peilab/why/eeg-retrieval/alignment
PY=${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}
mkdir -p "${ROOT}/data/features" "${ROOT}/outputs/slurm/extract"

echo "started : $(date -Iseconds)"
echo "job_id  : ${SLURM_JOB_ID:-local}"
echo "host    : $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"${PY}" -c "import torch; assert torch.cuda.is_available(), 'no CUDA';
print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '| gpu', torch.cuda.get_device_name(0))"

# TAG|KIND|MODEL|PRETRAINED|VERIFY
#  verify=1 only for ViT-H-14: it checks our concept ordering and image-averaging
#  convention against the shipped features.  If that check fails, every EEG
#  alignment downstream is silently noise, so it is the one encoder that must be
#  verified.  Tagged encoders are chosen to span the design space:
#    clip_h14   CLIP, contrastively trained        -> the standard baseline
#    dinov2_l   self-supervised, no text           -> best on fMRI, WORST on EEG
#                                                      retrieval; the NV framework
#                                                      must explain this tension
#    siglip_so400m  sigmoid-contrastive, 384px     -> different objective/scale
#    eva02_l    reported best on EEG (BCI lit)     -> the competitor claim
#    rn50       shallow conv, cheap                -> low-capacity control
ENCODERS=(
  "clip_h14|open_clip|ViT-H-14|laion2B-s32B-b79K|1"
  "dinov2_l|timm|vit_large_patch14_dinov2.lvd142m||0"
  "siglip_so400m|timm|vit_so400m_patch14_siglip_384||0"
  "eva02_l|open_clip|EVA02-L-14|merged2b_s4b_b131k||0"
  "rn50|open_clip|RN50|openai||0"
)

FAILED=()
OK=()
t_all=$SECONDS

for spec in "${ENCODERS[@]}"; do
  IFS='|' read -r tag kind model pre verify <<< "$spec"
  echo ""
  echo "=============================================================="
  echo "[encoder] ${tag}  (${kind} ${model})   elapsed ${((SECONDS - t_all)/60)}m"
  echo "=============================================================="
  args=(
    "${ROOT}/scripts/extract_features.py"
    --out "${ROOT}/data/features" --tag "${tag}" --kind "${kind}"
    --model "${model}" --split "${SPLIT:-train}" --batch "${BATCH:-32}"
  )
  [[ -n "$pre" ]] && args+=( --pretrained "$pre" )
  [[ "$verify" == "1" ]] && args+=( --verify )

  t0=$SECONDS
  if "${PY}" "${args[@]}" 2>&1 | tee "${ROOT}/outputs/slurm/extract/${tag}.log"; then
    echo "[encoder] ${tag} OK ($((SECONDS - t0))s)"
    OK+=("$tag")
  else
    echo "[encoder] ${tag} FAILED rc=${PIPESTATUS[0]}"
    FAILED+=("$tag")
  fi
done

echo ""
echo "=============================================================="
echo "EXTRACTION SUMMARY   total $(( (SECONDS - t_all)/60 ))m"
echo "  ok     : ${OK[*]:-none}"
echo "  failed : ${FAILED[*]:-none}"
echo "=============================================================="

# write a machine-readable summary; list what each tag produced
"${PY}" - <<PYEOF
import json, os, datetime
root = "${ROOT}"
summ = {"job_id": "${SLURM_JOB_ID:-local}",
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "ok": "${OK[*]:-}".split(), "failed": "${FAILED[*]:-}".split(),
        "tags": {}}
fdir = os.path.join(root, "data", "features")
for tag in summ["ok"] + summ["failed"]:
    d = os.path.join(fdir, tag)
    man = os.path.join(d, "manifest.json")
    if os.path.exists(man):
        m = json.load(open(man))
        summ["tags"][tag] = {"layers": sorted(m.get("layers", {}).keys()),
                             "n_layers": len(m.get("layers", {})),
                             "verify": m.get("verify_vs_shipped"),
                             "elapsed_s": m.get("elapsed_s")}
out = os.path.join(root, "outputs", "pipeline", "extract_status.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(summ, open(out, "w"), indent=2)
print(json.dumps(summ, indent=2))
PYEOF

echo "finished: $(date -Iseconds)"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "SOME ENCODERS FAILED: ${FAILED[*]}"
  exit 1
fi
echo "ALL ENCODERS EXTRACTED."
