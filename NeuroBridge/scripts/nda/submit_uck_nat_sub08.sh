#!/usr/bin/env bash
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
mkdir -p outputs/slurm outputs/uck_nat_s08/logs

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nda/uck_nat_khyp_export.py
chk scripts/nda/uck_nat_select.py
chk scripts/nda/run_uck_nat_sub08.sh
chk slurm/uck_nat_sub08.sbatch
chk scripts/nda/generate_hcma_s_decode.py
chk scripts/nda/eval_official_seven_dir.py
chk scripts/nda/gem_calib.py
chk outputs/uck/sub-08/full/conds/ip_mem_test.npy
chk outputs/uck/sub-08/full/spatial/pred_depth_rgb_512/199.png
chk outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512/199.png
chk outputs/nat/sub-08/full/proto/mu_all.npy
chk outputs/uck/shared/g_img_concept.npy
chk outputs/gem/cond_cache/clip_img1024_train.npy
chk outputs/gem/cond_cache/clip_img1024_test.npy
chk outputs/ocf/intra_z/sub-08/shared_r_train.npy
chk outputs/ocf/intra_z/sub-08/shared_r_test.npy
chk outputs/leakfree/split.json
chk outputs/g2f/prompts/prompts_deploy.json
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
bash -n scripts/nda/run_uck_nat_sub08.sh || fail=1
bash -n scripts/nda/submit_uck_nat_sub08.sh || fail=1
python3 -m py_compile scripts/nda/uck_nat_khyp_export.py || fail=1
python3 -m py_compile scripts/nda/uck_nat_select.py || fail=1
if (( fail )); then echo "[FATAL] preflight failed"; exit 1; fi

echo "===== CPU export smoke (no GPU, seconds) ====="
python3 scripts/nda/uck_nat_khyp_export.py \
  --out outputs/uck_nat_s08 \
  --test-subject 8 \
  --K 8 \
  --lambdas 0,0.3,0.5,1.0 \
  --uck-ip outputs/uck/sub-08/full/conds/ip_mem_test.npy \
  --query-npy outputs/uck/sub-08/full/conds/ip_q_test.npy \
  --query-name uck_q \
  --alt-query-npy outputs/ack_s08/heads/conds/ip_q_test.npy \
  --exclude-self 1 \
  2>&1 | tee outputs/uck_nat_s08/logs/export_smoke.txt | tail -n 45

python3 - <<'PY'
import json,sys
r=json.load(open("outputs/uck_nat_s08/export_report.json"))
print("[smoke] fuse_ok:", r.get("fuse_lambda0_equals_uck"), r.get("fuse_lambda0_max_abs"))
print("[smoke] diversity:", r.get("diversity"))
print("[smoke] posthoc retrieval:", r.get("diagnostics_POSTHOC"))
if not r.get("fuse_lambda0_equals_uck"):
    print("[FATAL] lambda=0 is not UCK-identical"); sys.exit(2)
if min(r.get("diversity",{"x":1}).values()) > 0.999:
    print("[FATAL] conditions are not diverse"); sys.exit(2)
print("[smoke] OK to submit")
PY

echo "===== submit ====="
JOB=$(sbatch --parsable slurm/uck_nat_sub08.sbatch)
echo "submitted JOB=${JOB}"
echo "${JOB}" > outputs/uck_nat_s08/job_id.txt
squeue -j "${JOB}" || true
