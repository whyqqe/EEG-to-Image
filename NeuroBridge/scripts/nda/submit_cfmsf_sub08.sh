#!/usr/bin/env bash
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
OUT="${CFMSF_OUT:-${NB_ROOT}/outputs/cfmsf_s08}"
mkdir -p "${OUT}/logs" outputs/slurm

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nda/cfmsf_train.py
chk scripts/nda/cfmsf_fuse_eval.py
chk scripts/nda/run_cfmsf_sub08.sh
chk slurm/cfmsf_s08.sbatch
chk scripts/nda/ocf_train.py
chk scripts/nda/leakfree.py
chk outputs/ocf/intra_z/sub-08/shared_r_train.npy
chk outputs/ocf/intra_z/sub-08/shared_r_test.npy
chk outputs/uck/shared/g_img_concept.npy
chk outputs/gem/cond_cache/clip_img1024_train.npy
chk outputs/gem/cond_cache/clip_depth1024_train.npy
chk outputs/gem/cond_cache/clip_edge1024_train.npy
chk outputs/gem/cond_cache/clip_img1024_test.npy
chk outputs/gem/cond_cache/clip_depth1024_test.npy
chk outputs/gem/cond_cache/clip_edge1024_test.npy
chk outputs/nda_ss/sub-08/clip_text/train/text_concept_clip.npy
chk outputs/nda_ss/sub-08/clip_text/train/text_flat_clip.npy
chk outputs/nda_ss/sub-08/clip_text/test/text_concept_clip.npy
chk outputs/leakfree/split.json
chk outputs/g2/captions/captions_train.jsonl
bash -n scripts/nda/run_cfmsf_sub08.sh || fail=1
bash -n slurm/cfmsf_s08.sbatch || fail=1
python3 -m py_compile scripts/nda/cfmsf_train.py || fail=1
python3 -m py_compile scripts/nda/cfmsf_fuse_eval.py || fail=1
if (( fail )); then echo "[FATAL] preflight failed"; exit 1; fi

echo "===== CPU smoke: 2-epoch single-route train ====="
PY=/project/peilab/why/eeg-brainit/.venv/bin/python
[[ -x "${PY}" ]] || { echo "[FATAL] venv python missing: ${PY}"; exit 1; }
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
mkdir -p "${OUT}/logs"
rm -rf /tmp/cfmsf_smoke
"${PY}" scripts/nda/cfmsf_train.py \
  --out /tmp/cfmsf_smoke --test-subject 8 --routes img \
  --epochs 2 --device cpu --inst-weight 0.2 2>&1 | tail -n 12
"${PY}" - <<'PY'
import json, sys
r = json.load(open("/tmp/cfmsf_smoke/train_report.json"))["routes"]["img"]
na = r["neural_address_val"]
print(f"[smoke] img val_top1={na['paired_top1']:.4f} shuf={na['shuffled_top1']:.4f} "
      f"gain={na['gain']:+.4f} chance={na['chance']:.4f}")
print(f"[smoke] test200_raw top1={r['test200_raw']['top1']:.4f}")
print("[smoke] train pipeline OK")
PY

echo "===== fuse smoke ====="
rm -rf /tmp/cfmsf_fuse_smoke
"${PY}" scripts/nda/cfmsf_fuse_eval.py \
  --train-out /tmp/cfmsf_smoke --out /tmp/cfmsf_fuse_smoke --routes img 2>&1 | tail -n 16
echo "[smoke] fuse pipeline OK"

echo "===== submit ====="
JOB=$(sbatch --parsable slurm/cfmsf_s08.sbatch)
echo "submitted JOB=${JOB}"
mkdir -p "${OUT}"
echo "${JOB}" > "${OUT}/job_id.txt"
squeue -j "${JOB}" || true
