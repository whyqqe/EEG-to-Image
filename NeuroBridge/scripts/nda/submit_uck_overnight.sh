#!/usr/bin/env bash
# Submit the UCK overnight job. Preflight only -- no GPU smoke loop.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
mkdir -p outputs/slurm outputs/uck/logs

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nmb/nmb_sota_v2_env.sh
chk scripts/nda/uck_train.py
chk scripts/nda/uck_build_depth.py
chk scripts/nda/uck_measure.py
chk scripts/nda/gem_calib.py
chk scripts/nda/generate_hcma_s_decode.py
chk scripts/nda/generate_atm_aligned_decode.py
chk scripts/nda/eval_official_seven_dir.py
chk scripts/nda/run_uck_overnight.sh
chk slurm/uck_overnight.sbatch
chk outputs/g2/captions/captions_train.jsonl
chk outputs/g2/captions/captions_test.jsonl
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
chk outputs/nda_ss/sub-08/clip_text/train/text_concept_clip.npy
chk outputs/gem/cond_cache/clip_img1024_train.npy
chk outputs/gem/cond_cache/clip_img1024_test.npy
chk outputs/sdedit_ll_full10/shared/vae_cache/train_vae_latents_f16.npy
chk outputs/sdedit_ll_full10/shared/vae_cache/test_vae_latents_f16.npy
chk outputs/leakfree/split.json
chk outputs/g2f/prompts/prompts_deploy.json
for s in 01 02 03 04 05 06 07 08 09 10; do
  chk "outputs/ocf/intra_z/sub-${s}/shared_r_train.npy"
  chk "outputs/ocf/intra_z/sub-${s}/shared_r_test.npy"
  chk "outputs/hcma_s_full10/sub-${s}/depth/pred_depth_rgb_512/199.png"
  chk "outputs/sdedit_ll_full10/sub-${s}/vae_head/pred_lowlevel_rgb_512/199.png"
done
for f in scripts/nda/run_uck_overnight.sh scripts/nda/submit_uck_overnight.sh; do
  bash -n "$f" || { echo "  [MISS] bash syntax $f"; fail=1; }
done
if (( fail )); then echo "[FATAL] missing assets; not submitting"; exit 1; fi

echo "===== python syntax ====="
python3 -m py_compile scripts/nda/uck_train.py scripts/nda/uck_build_depth.py scripts/nda/uck_measure.py \
  || { echo "[FATAL] py_compile failed"; exit 1; }
echo "  [ok]   py_compile"

echo "===== submit ====="
EXCL="${UCK_EXCLUDE:-dgx-09}"
JID="$(sbatch --parsable --exclude="${EXCL}" slurm/uck_overnight.sbatch)"
echo "JOBID=${JID}  (excluded: ${EXCL})"
echo "  log: outputs/slurm/uck_night_${JID}.out"
echo "  err: outputs/slurm/uck_night_${JID}.err"
echo "  out: outputs/uck"
