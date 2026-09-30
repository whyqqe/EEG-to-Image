#!/usr/bin/env bash
# Submit the NAT overnight job. Preflight only -- no GPU smoke loop.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
mkdir -p outputs/slurm outputs/nat/logs

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nmb/nmb_sota_v2_env.sh
chk scripts/nda/nat_train.py
chk scripts/nda/nat_measure.py
chk scripts/nda/uck_train.py
chk scripts/nda/gem_calib.py
chk scripts/nda/generate_hcma_s_decode.py
chk scripts/nda/eval_official_seven_dir.py
chk scripts/nda/run_nat_overnight.sh
chk slurm/nat_overnight.sbatch
chk outputs/g2/captions/captions_train.jsonl
chk outputs/g2/captions/captions_test.jsonl
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
chk outputs/nda_ss/sub-08/clip_text/train/text_concept_clip.npy
chk outputs/gem/cond_cache/clip_img1024_train.npy
chk outputs/gem/cond_cache/clip_img1024_test.npy
chk outputs/uck/shared/g_img_concept.npy
chk outputs/uck/shared/gt_depth/train_depth_64.npy
chk outputs/hcma_s_full10/shared/gt_depth/test_depth_64.npy
chk outputs/leakfree/split.json
chk outputs/g2f/prompts/prompts_deploy.json
for s in 01 02 03 04 05 06 07 08 09 10; do
  chk "outputs/ocf/intra_z/sub-${s}/shared_r_train.npy"
  chk "outputs/ocf/intra_z/sub-${s}/shared_r_test.npy"
  chk "outputs/hcma_s_full10/sub-${s}/depth/pred_depth_rgb_512/199.png"
  chk "outputs/sdedit_ll_full10/sub-${s}/vae_head/pred_lowlevel_rgb_512/199.png"
done
chk outputs/uck/sub-08/full/conds/ip_mem_test.npy
for f in scripts/nda/run_nat_overnight.sh scripts/nda/submit_nat_overnight.sh; do
  bash -n "$f" || { echo "  [MISS] bash syntax $f"; fail=1; }
done
if (( fail )); then echo "[FATAL] missing assets; not submitting"; exit 1; fi

echo "===== python syntax ====="
python3 -m py_compile scripts/nda/nat_train.py scripts/nda/nat_measure.py scripts/nda/uck_train.py \
  || { echo "[FATAL] py_compile failed"; exit 1; }
echo "  [ok]   py_compile"

echo "===== submit ====="
EXCL="${NAT_EXCLUDE:-dgx-09}"
JID="$(sbatch --parsable --exclude="${EXCL}" slurm/nat_overnight.sbatch)"
echo "JOBID=${JID}  (excluded: ${EXCL})"
echo "  log: outputs/slurm/nat_night_${JID}.out"
echo "  err: outputs/slurm/nat_night_${JID}.err"
echo "  out: outputs/nat"
echo "  uck remasure 573149 left running (official UCK baseline; different out dir)"
