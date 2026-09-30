#!/usr/bin/env bash
# Submit hybrid sub-08 feasibility job. Preflight + CPU export smoke, then sbatch.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
mkdir -p outputs/slurm outputs/hybrid_s08/logs

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nmb/nmb_sota_v2_env.sh
chk scripts/nda/hybrid_export.py
chk scripts/nda/run_hybrid_sub08.sh
chk scripts/nda/gem_calib.py
chk scripts/nda/generate_hcma_s_decode.py
chk scripts/nda/eval_official_seven_dir.py
chk slurm/hybrid_sub08.sbatch
chk outputs/uck/sub-08/full/conds/ip_mem_test.npy
chk outputs/uck/sub-08/full/conds/ip_q_test.npy
chk outputs/uck/sub-08/full/spatial/pred_depth_rgb_512/199.png
chk outputs/nat/sub-08/full/conds/ip_nat_test.npy
chk outputs/nat/sub-08/full/spatial/pred_depth_rgb_512/199.png
chk outputs/nat/sub-08/full/proto/mu_all.npy
chk outputs/uck/shared/g_img_concept.npy
chk outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512/199.png
chk outputs/g2f/prompts/prompts_deploy.json
chk outputs/gem/cond_cache/clip_img1024_train.npy
chk outputs/gem/cond_cache/clip_img1024_test.npy
bash -n scripts/nda/run_hybrid_sub08.sh || { echo "  [MISS] bash syntax"; fail=1; }
bash -n scripts/nda/submit_hybrid_sub08.sh || { echo "  [MISS] bash syntax submit"; fail=1; }
python3 -m py_compile scripts/nda/hybrid_export.py || { echo "  [MISS] py_compile"; fail=1; }
if (( fail )); then echo "[FATAL] preflight failed"; exit 1; fi
echo "  [ok]   syntax"

echo "===== CPU export smoke ====="
python3 scripts/nda/hybrid_export.py \
  --out outputs/hybrid_s08/full --test-subject 8 \
  --gallery-cache outputs/uck/shared \
  --uck-conds outputs/uck/sub-08/full/conds \
  --nat-proto outputs/nat/sub-08/full/proto \
  --device cpu \
  > outputs/hybrid_s08/logs/export_smoke.txt 2>&1
tail -n 25 outputs/hybrid_s08/logs/export_smoke.txt
[[ -f outputs/hybrid_s08/full/conds/ip_snap_test.npy ]] || { echo "[FATAL] export smoke produced no IP"; exit 1; }
echo "  [ok]   export smoke"

echo "===== submit ====="
EXCL="${HYB_EXCLUDE:-dgx-09}"
JID="$(sbatch --parsable --exclude="${EXCL}" slurm/hybrid_sub08.sbatch)"
echo "JOBID=${JID}  (excluded: ${EXCL})"
echo "  log: outputs/slurm/hyb_s08_${JID}.out"
echo "  err: outputs/slurm/hyb_s08_${JID}.err"
echo "  out: outputs/hybrid_s08"
echo "  rows: uck/nat/snap/short/gate/blend/hard +uckF, snap+natF"
