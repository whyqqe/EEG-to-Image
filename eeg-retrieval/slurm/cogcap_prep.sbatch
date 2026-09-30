#!/usr/bin/env bash
#SBATCH --job-name=cogcap-prep
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/cogcap-prep-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/cogcap-prep-%j.err
#
# STAGE 0 -- the two inputs the LOSO job cannot start without.
#
# 1. depth/edge conditions. CogCapPro aligns four branches (image/text/depth/edge); this
#    project has the RGB set and the `image` conditions but no depth or edge renders, so
#    they are generated here (Depth-Anything-V2-Small + Canny) and encoded through the same
#    frozen CLIP ViT-H-14 that produced `clip_h14_ip_adapter`. Same encoder, same
#    preprocessing and the same listing gate, because the branches share a `logit_scale`
#    and are averaged into one loss -- a differently-preprocessed target would show up as
#    one branch being easier to fit, not as an error.
#
#    The `text` branch is deliberately not generated. CogCapPro's text target is a
#    per-image BLIP2 caption read from `weights/texts/eeg/texts_BLIP2_{mode}.npy`; this
#    project's caption files are SDXL prompts keyed by concept, which is a different
#    object. Substituting it would look like a faithful reproduction while changing what
#    the branch is aligned to.
#
# 2. the averaged EEG cache. The raw arrays are 4.2 GB per subject (16540 x 4 x 63 x 250)
#    and training must average the 4 repetitions anyway, so that is done once into
#    outputs/cogcap/cache/. Without this each LOSO run re-reads ~42 GB.
#
# Cost: Depth-Anything-V2-Small over 16740 images on one GPU is ~10-20 min; the CLIP pass
# adds a few minutes; the EEG cache is ~42 GB of read plus 10 GB of write. Both are
# idempotent and skip when their outputs already exist, so re-submitting is free.
set -euo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
PY="/project/peilab/why/eeg-brainit/.venv/bin/python"

cd "${ROOT}"
mkdir -p outputs/slurm outputs/cogcap

# `cogcap` is a package under `scripts/`, so the module root has to be on the path: the
# job's cwd is the repository root, and `python -m cogcap.prep_features` would otherwise
# die with ModuleNotFoundError before touching a single image.
export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== host $(hostname) date $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== [1/2] depth + edge conditions ==="
if [[ -f data/image_feature/cogcap_depth/depth_train.npy && \
      -f data/image_feature/cogcap_edge/edge_train.npy && \
      "${FORCE:-0}" != "1" ]]; then
  echo "already present; skipping (FORCE=1 to rebuild)"
else
  "${PY}" -m cogcap.prep_features \
      --modalities "${MODALITIES:-depth,edge}" \
      --splits train,test \
      --batch "${BATCH:-64}" \
      ${LIMIT:+--limit "${LIMIT}"}
fi

echo "=== [2/2] averaged + standardised EEG cache ==="
"${PY}" - <<'PY'
import sys, time
sys.path.insert(0, "/project/peilab/why/eeg-retrieval/scripts")
from cogcap import config
from cogcap.data import ensure_eeg_cache

t0 = time.time()
for split in ("train", "test"):
    for s in config.SUBJECTS:
        p = ensure_eeg_cache(s, split)
        print(f"  {split} sub-{s:02d} -> {p.name}", flush=True)
print(f"cache ready in {time.time() - t0:.0f}s")
PY

echo "=== done $(date -Is) ==="
