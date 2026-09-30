#!/usr/bin/env bash
# Submit the single G3F job. One submission only: an array plus a dependent job
# has previously tripped QOSMaxSubmitJobPerUserLimit.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
mkdir -p outputs/slurm outputs/g3f/logs

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }

chk scripts/nda/g3f_train.py
chk scripts/nda/run_g3f.sh
chk scripts/nda/g3f_summary.py
chk scripts/nda/eval_standard7.py
chk scripts/nda/eval_pooled_fid.py
chk scripts/nda/generate_atm_aligned_decode.py
chk scripts/nda/generate_hcma_s_decode.py
chk scripts/nmb/nmb_sota_v2_env.sh
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
chk outputs/nda_ss/sub-08/clip_text/train/text_concept_clip.npy
chk outputs/g2/captions/captions_train.jsonl
chk outputs/g2/targets/sem_image_train.npy
chk outputs/g2/targets/perc_struct_train.npy
for s in 01 02 03 04 05 06 07 08 09 10; do
  chk "outputs/hcma_10subj/sub-${s}/zret/z_eeg_proj_test.npy"
  chk "outputs/sdedit_ll_full10/sub-${s}/vae_head/pred_lowlevel_rgb_512/199.png"
done
chk outputs/intra_hcma_s/sub-08/depth/pred_depth_rgb_512/199.png

echo "===== syntax/consistency ====="
python3 -c "import ast,sys
for f in ('scripts/nda/g3f_train.py','scripts/nda/g3f_summary.py'): ast.parse(open(f).read()); print('  [ok] syntax', f)"
bash -n scripts/nda/run_g3f.sh && echo "  [ok] bash -n run_g3f.sh"

# the interpreter check that keeps costing whole nights: the training script must
# import under the project venv, not the system python.
source scripts/nmb/nmb_sota_v2_env.sh >/dev/null 2>&1
python - <<'PY'
import importlib
for m in ("torch", "diffusers", "transformers", "open_clip", "skimage"):
    try:
        mod = importlib.import_module(m)
        print(f"  [ok] {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        raise SystemExit(f"  [FATAL] {m} import failed under the project venv: {e}")
import diffusers
print(f"  [info] diffusers {diffusers.__version__} (needs >=0.31 for the SDXL pipelines used here)")
PY

if (( fail )); then echo "[FATAL] missing assets; not submitting"; exit 1; fi

echo "===== submit ====="
JID="$(sbatch --parsable slurm/g3f.sbatch)"
echo "JOBID=${JID}"
echo "${JID}" > outputs/g3f_sub08.jobid
squeue -j "${JID}" -o "%.12i %.24j %.10T %.10M %.20R" || true
