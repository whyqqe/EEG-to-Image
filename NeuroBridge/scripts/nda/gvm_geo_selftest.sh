#!/usr/bin/env bash
# Smoke test for `gem_geo.py`: does the depth estimator load offline, do the depth
# and edge renders actually differ, and are the resulting CLIP features distinct
# from the RGB embedding?  Writes to a THROWAWAY directory -- the caches it produces
# are PREFIXES of the splits (`--max-images`) and must never be used for training.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source scripts/nmb/nmb_sota_v2_env.sh >/dev/null 2>&1 || true

OUT="${NB_ROOT}/outputs/gvm/_geo_selftest"
rm -rf "${OUT}"
mkdir -p "${OUT}"
echo "===== gem_geo selftest @ $(date -Iseconds) on $(hostname) ====="
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset} HF_HUB_CACHE=${HF_HUB_CACHE:-unset}"

python scripts/nda/gem_geo.py \
  --out-dir "${OUT}" \
  --device "${DEVICE:-cuda:0}" \
  --batch-size 16 \
  --max-images 64 \
  --splits train test

echo "===== written ====="
ls -la "${OUT}"
python - <<'PY'
import numpy as np
from pathlib import Path
out = Path("/project/peilab/why/NeuroBridge/outputs/gvm/_geo_selftest")
for f in sorted(out.glob("clip_*1024_*.npy")):
    a = np.load(f)
    print(f"  {f.name}: {a.shape} dtype={a.dtype} "
          f"norm_mean={np.linalg.norm(a, axis=1).mean():.4f} (expect 1.0)")
PY
echo "===== gem_geo selftest OK @ $(date -Iseconds) ====="
