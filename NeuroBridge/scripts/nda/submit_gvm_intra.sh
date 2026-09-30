#!/usr/bin/env bash
# Submit the single GVM sub-08 intra job.  One submission only.
#
# The preflight is deliberately heavier than a file-existence check, because the
# GVM additions fail in ways that a missing-file check cannot see:
#   * M3 needs BOTH cached CLIP layers.  With only the 1024-d feature the visibility
#     A/B has one point and the whole mechanism is untestable, so this aborts rather
#     than running a "comparison" against nothing.
#   * M2 needs a description corpus rich enough to build a vocabulary with a
#     frequency floor.  If `--anchor-min-count` cannot be met the training script
#     already refuses, and this reports the count up front so that is not discovered
#     twenty minutes in.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
mkdir -p outputs/slurm outputs/gvm/sub-08/logs

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nmb/nmb_sota_v2_env.sh
chk scripts/nda/gem_clip_img.py
chk scripts/nda/gem_front.py
chk scripts/nda/gem_train.py
chk scripts/nda/gem_calib.py
chk scripts/nda/gem_ground.py
chk scripts/nda/gvm_baseline_check.py
chk scripts/nda/generate_atm_aligned_decode.py
chk scripts/nda/run_gem_intra.sh
chk slurm/gvm_intra_s08.sbatch
chk outputs/g2/captions/captions_train.jsonl
chk outputs/g2/captions/captions_test.jsonl
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
chk outputs/nda_ss/sub-08/clip_text/train/text_concept_clip.npy
chk outputs/sdedit_ll_full10/shared/vae_cache/train_vae_latents_f16.npy
chk outputs/sdedit_ll_full10/shared/vae_cache/test_vae_latents_f16.npy
chk outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth
chk outputs/ocf/intra_z/sub-08/shared_r_train.npy
chk outputs/ocf/intra_z/sub-08/shared_r_test.npy
chk outputs/gem/cond_cache/clip_img1024_train.npy
chk outputs/gem/cond_cache/clip_img1024_test.npy
# M3 is a comparison between TWO layers; asserting both is the point.
chk outputs/gem/cond_cache/clip_img1280_train.npy
chk outputs/gem/cond_cache/clip_img1280_test.npy
chk outputs/sdedit_ll_full10/sub-08/generation/sdedit_ll/generated/199.png
chk outputs/g3f/gen/sub-08/g3f_ll_selfgate/generated/199.png
for f in scripts/nda/run_gem_intra.sh; do bash -n "$f" || { echo "  [MISS] bash syntax $f"; fail=1; }; done
if (( fail )); then echo "[FATAL] missing assets; not submitting"; exit 1; fi

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

echo "===== M2 vocabulary feasibility (this is what the anchor head predicts) ====="
python - <<'PY' || { echo "[FATAL] anchor vocabulary would be too small"; exit 1; }
import json, re, sys
from collections import Counter
from pathlib import Path
GRANS = ("overall", "subject", "background", "detail")
caps = [json.loads(l) for l in
        Path("outputs/g2/captions/captions_train.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
STOP = {"the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "with", "is",
        "are", "as", "by", "for", "from", "that", "this", "it", "its", "his", "her",
        "their", "there", "some", "very", "while", "into", "over", "near", "next",
        "which", "has", "have", "been", "being", "was", "were", "be", "he", "she",
        "they", "you", "we", "i", "not", "but", "also", "can", "one", "two"}
c = Counter()
for row in caps:
    for g in GRANS:
        for w in set(w for w in re.findall(r"[a-z]+", str(row.get(g, "")).lower())
                     if w not in STOP and len(w) > 2):
            c[w] += 1
for thr in (12, 20, 50):
    n = sum(1 for v in c.values() if v >= thr)
    print(f"  [vocab] >= {thr:>3} TRAIN descriptions: {n:>5} words "
          f"(cap 384 -> {min(n, 384)} anchors)")
if sum(1 for v in c.values() if v >= 12) < 32:
    sys.exit("  [MISS] fewer than 32 words at the default floor; the anchor head "
             "would have almost nothing to predict")
# coverage: do real rows actually carry in-vocabulary words?
keep = {w for w, v in c.items() if v >= 12}
cov = []
for row in caps[:2000]:
    own = set(w for g in GRANS
              for w in re.findall(r"[a-z]+", str(row.get(g, "")).lower())
              if w not in STOP and len(w) > 2)
    cov.append(len(own & keep) / max(len(own), 1))
import statistics
print(f"  [vocab] mean in-vocabulary fraction of a row's content words "
      f"{statistics.mean(cov):.4f} (n=2000)")
PY

echo "===== submit ====="
EXCL="${GVM_EXCLUDE:-dgx-09}"
JID="$(sbatch --parsable --exclude="${EXCL}" slurm/gvm_intra_s08.sbatch)"
echo "JOBID=${JID}  (excluded: ${EXCL})"
echo "  log: outputs/slurm/gvm_s08_${JID}.out"
echo "  err: outputs/slurm/gvm_s08_${JID}.err"
