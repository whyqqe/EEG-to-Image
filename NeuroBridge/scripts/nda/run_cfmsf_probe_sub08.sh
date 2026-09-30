#!/usr/bin/env bash
# CF-MSF route-quality probe, sub-08.
#
# Holds the DECISION side fixed (leak-free split, same head, same gallery-NCE, same
# fusion code as run_cfmsf_sub08.sh) and varies what each route ALIGNS TO, to find
# out which target actually carries concept identity for this subject.  Job 581546
# established the baseline this has to beat: 4 routes at 15-31% single-route and
# 40.0% (CSLS) / 50.0% (+Sinkhorn) fused on the official 200-way test.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"

PYTHON=/project/peilab/why/eeg-brainit/.venv/bin/python
[[ -x "${PYTHON}" ]] || { echo "[FATAL] missing ${PYTHON}"; exit 1; }
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch

OUT="${CFMSF_PROBE_OUT:-${NB_ROOT}/outputs/cfmsf_probe/sub-08}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-80}"
mkdir -p "${OUT}"/{heads,logs}

log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; exit 1; }; }

# ---- inputs: EEG feature space, concept labels, every target bank -------------
require outputs/ocf/intra_z/sub-08/shared_r_train.npy
require outputs/ocf/intra_z/sub-08/shared_r_test.npy
require outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
require outputs/nda_ss/sub-08/clip_text/train/text_concept_clip.npy
require outputs/g2/captions/captions_train.jsonl
require outputs/leakfree/split.json
require outputs/gem/cond_cache/clip_img1024_train.npy
require outputs/gem/cond_cache/clip_depth1024_train.npy
require outputs/gem/cond_cache/clip_edge1024_train.npy
for lv in image GaussianBlur LowResolution Mosaic GaussianNoise; do
  if [[ "${lv}" == "image" ]]; then
    require data/things_eeg/image_feature/ViT-H-14/image_train.npy
    require data/things_eeg/image_feature/ViT-H-14/image_test.npy
    require data/things_eeg/image_feature/RN50/image_train.npy
    require data/things_eeg/image_feature/RN50/image_test.npy
  else
    require "data/things_eeg/image_feature/ViT-H-14/${lv}/train.npy"
    require "data/things_eeg/image_feature/ViT-H-14/${lv}/test.npy"
  fi
done
require "data/things_eeg/image_feature/ViT-H-14/GaussianBlur-GaussianNoise-LowResolution-Mosaic/train.npy"

log "===== CF-MSF route-quality probe (sub-08, epochs=${EPOCHS}, device=${DEVICE}) ====="
"${PYTHON}" scripts/nda/cfmsf_route_probe.py \
    --out "${OUT}" --test-subject 8 --epochs "${EPOCHS}" --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/probe.log"

log "===== summary ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
OUT = Path(os.environ["CFMSF_PROBE_OUT"])
r = json.loads((OUT / "route_probe.json").read_text())
print(f"{'arm':<24}{'dim':>5}{'mlp val':>10}{'mlp t1':>9}{'mlp t5':>9}"
      f"{'mlp csls':>10}{'rdg val':>9}{'rdg t1':>9}{'rdg csls':>10}")
for k, v in sorted(r["targets"].items(), key=lambda kv: -kv[1]["mlp"]["top1"]):
    m, d = v["mlp"], v["ridge"]
    print(f"{k:<24}{v['dim']:>5}{m['val_top1']:>10.4f}{m['top1']:>9.4f}{m['top5']:>9.4f}"
          f"{m['top1_csls']:>10.4f}{d['val_top1']:>9.4f}{d['top1']:>9.4f}{d['top1_csls']:>10.4f}")
print()
for est, f in r.get("fusion", {}).items():
    print(f"[fuse {est:<6}] routes={f['routes']}")
    for tag in ("raw", "csls"):
        m = f[tag]
        print(f"    {tag:<5} top1={m['top1']:.4f} top5={m['top5']:.4f} "
              f"rank={m['mean_rank']:.2f}  +sinkhorn={m['sinkhorn_top1']:.4f}")
print(f"\n[reference] job {r['reference_cfmsf_job']['job']}: "
      f"{r['reference_cfmsf_job']['primary']} top1={r['reference_cfmsf_job']['top1']:.2f} "
      f"+sinkhorn={r['reference_cfmsf_job']['top1_plus_sinkhorn']:.2f}")
PY
log "===== done ====="
du -sh "${OUT}" | sed 's/^/[disk] /'
