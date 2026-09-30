#!/usr/bin/env bash
# Submit the single GEM sub-08 intra job.  One submission only.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
mkdir -p outputs/slurm outputs/gem/sub-08/logs

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nmb/nmb_sota_v2_env.sh
chk scripts/nda/gem_clip_img.py
chk scripts/nda/gem_front.py
chk scripts/nda/gem_train.py
chk scripts/nda/gem_calib.py
chk scripts/nda/gem_ground.py
chk scripts/nda/run_gem_intra.sh
chk slurm/gem_intra_s08.sbatch
chk outputs/g2/captions/captions_train.jsonl
chk outputs/g2/captions/captions_test.jsonl
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
chk outputs/nda_ss/sub-08/clip_text/train/text_concept_clip.npy
chk outputs/sdedit_ll_full10/shared/vae_cache/train_vae_latents_f16.npy
chk outputs/sdedit_ll_full10/shared/vae_cache/test_vae_latents_f16.npy
chk outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth
chk outputs/ocf/intra_z/sub-08/shared_r_train.npy
chk outputs/ocf/intra_z/sub-08/shared_r_test.npy
chk outputs/sdedit_ll_full10/sub-08/generation/sdedit_ll/generated/199.png
chk outputs/g3f/gen/sub-08/g3f_ll_selfgate/generated/199.png
for f in scripts/nda/run_gem_intra.sh; do bash -n "$f" || { echo "  [MISS] bash syntax $f"; fail=1; }; done
if (( fail )); then echo "[FATAL] missing assets; not submitting"; exit 1; fi

# `t5-base` must resolve OFFLINE: the export of the offline flags is what makes a
# missing snapshot fail here (in 20 s) instead of inside the 20-hour job after the
# CLIP extraction has already run.
echo "===== offline model check ====="
source scripts/nmb/nmb_sota_v2_env.sh >/dev/null 2>&1 || true
echo "  HF_HUB_CACHE=${HF_HUB_CACHE:-unset}  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset}"
python - <<'PY' || { echo "[FATAL] offline T5 unavailable"; exit 1; }
import os, sys
from transformers import AutoTokenizer, T5ForConditionalGeneration
try:
    tok = AutoTokenizer.from_pretrained("google-t5/t5-base")
    m = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
except Exception as e:                                     # noqa: BLE001
    sys.exit(f"  [MISS] {type(e).__name__}: {e}")
print(f"  [ok]   offline t5-base d_model={m.config.d_model} "
      f"params={sum(p.numel() for p in m.parameters())/1e6:.1f}M")
if m.config.d_model != 768:
    sys.exit("  [MISS] d_model is not 768; the towers are dimensioned for 768")
PY

echo "===== submit ====="
# `--exclude` rather than `-w` pinning: the pinned submission waited hours behind
# other users, and an unpinned one landed on dgx-09 whose driver (12.8) is older
# than this venv's torch (cu130).  That is not a tuning preference: the run must
# hard-fail rather than fall back to CPU, because a CPU fallback would cost ~7
# hours for the CLIP cache stage alone.
EXCL="${GEM_EXCLUDE:-dgx-09}"
JID="$(sbatch --parsable --exclude="${EXCL}" slurm/gem_intra_s08.sbatch)"
echo "JOBID=${JID}  (excluded: ${EXCL})"
echo "  log: outputs/slurm/gem_s08_${JID}.out"
echo "  err: outputs/slurm/gem_s08_${JID}.err"
