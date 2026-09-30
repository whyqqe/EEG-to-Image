#!/usr/bin/env bash
# Preflight + CPU smoke + submit for `cfmsf_probe_s08` (route-quality probe).
#
# The smoke runs 3 arms for 2 epochs on CPU.  That is not a performance check --
# it is a WIRING check: it proves every target bank loads, the head accepts a
# non-1024 out_dim (the concat arms), the ridge path selects an alpha, and the
# fusion block runs, all before a GPU is reserved.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
OUT="${CFMSF_PROBE_OUT:-${NB_ROOT}/outputs/cfmsf_probe/sub-08}"
mkdir -p "${OUT}/logs" outputs/slurm

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nda/cfmsf_route_probe.py
chk scripts/nda/cfmsf_train.py
chk scripts/nda/run_cfmsf_probe_sub08.sh
chk slurm/cfmsf_probe_s08.sbatch
chk outputs/leakfree/split.json
chk outputs/ocf/intra_z/sub-08/shared_r_train.npy
chk outputs/ocf/intra_z/sub-08/shared_r_test.npy
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
chk outputs/g2/captions/captions_train.jsonl
chk outputs/gem/cond_cache/clip_img1024_train.npy
chk outputs/gem/cond_cache/clip_depth1024_train.npy
chk outputs/gem/cond_cache/clip_edge1024_train.npy
chk data/things_eeg/image_feature/ViT-H-14/image_train.npy
chk data/things_eeg/image_feature/ViT-H-14/GaussianBlur/train.npy
chk data/things_eeg/image_feature/ViT-H-14/LowResolution/train.npy
chk data/things_eeg/image_feature/ViT-H-14/Mosaic/train.npy
chk data/things_eeg/image_feature/ViT-H-14/GaussianNoise/train.npy
chk data/things_eeg/image_feature/ViT-H-14/GaussianBlur-GaussianNoise-LowResolution-Mosaic/train.npy
chk data/things_eeg/image_feature/RN50/image_train.npy
bash -n scripts/nda/run_cfmsf_probe_sub08.sh || fail=1
bash -n slurm/cfmsf_probe_s08.sbatch || fail=1
python3 -m py_compile scripts/nda/cfmsf_route_probe.py || fail=1
python3 -m py_compile scripts/nda/cfmsf_train.py || fail=1
if (( fail )); then echo "[FATAL] preflight failed"; exit 1; fi

echo "===== CPU smoke: 3 arms x 2 epochs (wiring only) ====="
PY=/project/peilab/why/eeg-brainit/.venv/bin/python
[[ -x "${PY}" ]] || { echo "[FATAL] venv python missing: ${PY}"; exit 1; }
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
rm -rf /tmp/cfmsf_probe_smoke
"${PY}" scripts/nda/cfmsf_route_probe.py \
  --out /tmp/cfmsf_probe_smoke --test-subject 8 --device cpu --epochs 2 \
  --only img_clip,vith_cat3,vith_lowresolution 2>&1 | tail -n 14
"${PY}" - <<'PY'
import json
r = json.load(open("/tmp/cfmsf_probe_smoke/route_probe.json"))
t = r["targets"]
assert set(t) == {"img_clip", "vith_cat3", "vith_lowresolution"}, sorted(t)
assert t["vith_cat3"]["dim"] == 3072, t["vith_cat3"]["dim"]      # concat out_dim plumbing
assert "fusion" in r and len(r["fusion"]) >= 1, "fusion block did not run"
for k, v in t.items():
    for est in ("mlp", "ridge"):
        assert 0.0 <= v[est]["top1"] <= 1.0, (k, est)
print("[smoke] all target banks load; concat out_dim=3072 OK; ridge+fusion OK")
PY

echo "===== submit ====="
JOB=$(sbatch --parsable slurm/cfmsf_probe_s08.sbatch)
echo "submitted JOB=${JOB}"
echo "${JOB}" > "${OUT}/job_id.txt"
squeue -j "${JOB}" || true
