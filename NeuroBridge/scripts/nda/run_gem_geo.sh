#!/usr/bin/env bash
# FULL depth/edge feature extraction (16540 train + 200 test).  Runs once and the
# caches are reused by every subsequent training run, exactly like
# `gem_clip_img.py`'s.  Idempotent: a split whose two caches already exist is left
# alone, so a resubmission after a partial failure does not redo finished work.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source scripts/nmb/nmb_sota_v2_env.sh >/dev/null 2>&1 || true

OUT="${NB_ROOT}/outputs/gem/cond_cache"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/gvm/sub-08/logs"

echo "===== gem_geo FULL @ $(date -Iseconds) on $(hostname) ====="
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset} HF_HUB_CACHE=${HF_HUB_CACHE:-unset}"

# split-by-split so a failure on one does not discard the other, and so the exit
# status names which split failed
for S in train test; do
  need=0
  for M in depth edge; do
    for D in 1024 1280; do
      [[ -f "${OUT}/clip_${M}${D}_${S}.npy" ]] || need=1
    done
  done
  if (( need == 0 )); then
    echo "[skip] ${S}: all four caches present"
    continue
  fi
  echo "----- ${S} -----"
  python scripts/nda/gem_geo.py \
    --out-dir "${OUT}" \
    --device "${DEVICE:-cuda:0}" \
    --batch-size 32 \
    --splits "${S}"
done

echo "===== written ====="
ls -la "${OUT}"/clip_depth* "${OUT}"/clip_edge*
python - <<'PY'
import json
import numpy as np
from pathlib import Path
out = Path("/project/peilab/why/NeuroBridge/outputs/gem/cond_cache")
rep = json.loads((out / "gem_geo_report.json").read_text(encoding="utf-8"))
for s, r in rep["splits"].items():
    print(f"  {s}: n={r['n']} cos(depth,img)={r.get('cos_depth_vs_image_1024'):+.4f} "
          f"cos(edge,img)={r.get('cos_edge_vs_image_1024'):+.4f} "
          f"cos(depth,edge)={r.get('cos_depth_vs_edge_1024'):+.4f}")
    for m in ("depth", "edge"):
        for d in (1024, 1280):
            f = out / f"clip_{m}{d}_{s}.npy"
            a = np.load(f, mmap_mode="r")
            print(f"    {f.name}: {a.shape} {a.dtype}")
PY
echo "===== gem_geo FULL OK @ $(date -Iseconds) ====="
