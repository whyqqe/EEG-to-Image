#!/usr/bin/env bash
# ============================================================================
# TDM-DT / sub-08 -- one leak-free overnight chain: measure, train, ablate,
#                    generate, evaluate.
#
# THE FRAMEWORK (the user's, made explicit)
#   cross-subject module  = `shared_r`, the shared branch of the intra encoder,
#                           trained on sub-08 alone in stage [1]
#   dual-tower encoding   = SEMANTIC  (image CLIP + overall/subject/background/
#                           detail descriptions)  and  STRUCTURAL (image CLIP
#                           spatial patch tokens + SDXL VAE low-frequency latent)
#   fused condition       = `h_fuse` consumes the semantic code, all four
#                           granularities, the concept, the image vector AND the
#                           structural spatial map, then `spherical()` places the
#                           result at a solved angle theta
#
# WHAT IS ACTUALLY NEW HERE (all on the encoder/alignment/control side)
#   [A] DLA  differentiable latency alignment -- per-band, per-channel group delay
#            as a Fourier phase ramp; `tau = max_lag * tanh(raw)`, raw = 0 at init.
#            The reconstruction literature uses ONE fixed time window for every
#            trial, which is the construct most exposed to trial-to-trial jitter.
#   [B] RSD  retinotopic spatial demixing -- per-band CxC linear unmixing,
#            identity at init.
#   [C] DNG  alpha divisive normalisation -- a spatially global gain that DIVIDES
#            the evidence (concat/routing cannot express division).
#   [D] GRANULARITY x TIME gates -- each granularity head attends over the 25
#            (band, time-patch) tokens with its own softmax gate, uniform at init,
#            and consumes it.  The gate is IN the gradient path on purpose: a gate
#            that no loss reads could never be learned, so the central claim is
#            only testable this way.
#            CENTRAL FALSIFIABLE CLAIM: mean gate time ordered
#                 overall < background < subject < detail
#            i.e. the fine description reads LATER than the coarse one.
#   [E] iREPA -- the 64 spatial tokens that decode the (4,64,64) latent are
#            aligned to the REAL image's 8x8 CLIP ViT-H-14 patch tokens, so the
#            spatial map is supervised in the same space AND layout as the target.
#            Nothing in this project had ever aligned EEG to a spatial feature map.
#   [F] HUB-AWARE InfoNCE -- local scaling (CSLS) against the hubness that a
#            1654-concept bank with 10 near-duplicate trials per image creates.
#
# THE CONTROL IS THE POINT
#   Stage [5] trains `--ablation all` from the SAME code on the SAME rows: DLA,
#   RSD, DNG, the time gates, the hub-aware loss and iREPA are all switched off,
#   which leaves exactly the historical `shared_r -> MLP` encoder with the OCF
#   read-out.  `tdm_abl_self` is generated and evaluated by the SAME generation
#   and metric code.  A mechanism that does not beat this row is reported as not
#   beating it.
#
# LEAK-FREE BY CONSTRUCTION
#   * encoder, TDM heads and VAE head all fit sub-08 train rows only;
#   * the prompt gallery is the 1654 TRAIN concepts, asserted DISJOINT from the
#     200 test concepts in stage [0] and hard-failed if not;
#   * the CLIP patch tokens are extracted for TRAIN images ONLY -- iREPA is a
#     training objective and the test images are never encoded;
#   * theta is solved on held-out TRAIN rows against a TRAIN statistic;
#   * the VAE head selects its checkpoint on a held-in TRAIN split and the run
#     hard-fails if it ever regresses to test-set selection;
#   * every generation hyper-parameter is FIXED A PRIORI below.
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
STAG="sub-08"
# NEW OUTPUT ROOT for the fixed run.  A fresh directory is not cosmetic: the
# stages below all `[SKIP]` when their outputs exist, and `tdm_train.py` resumes
# from `last.pth` by default, so pointing this at the old `outputs/tdm_intra`
# would have resumed the frozen model at epoch 26 (i.e. trained for zero steps)
# and regenerated images from the untrained conditions.  `v2` cannot collide.
OUT="${OUT:-${NB_ROOT}/outputs/tdm_intra_v2}"
ENC="${NB_ROOT}/outputs/ocf/intra_enc"
ZC="${NB_ROOT}/outputs/ocf/intra_z"
RAW="${RAW:-${NB_ROOT}/outputs/tdm/cache}"
PATCH="${PATCH:-${NB_ROOT}/outputs/tdm/clip_patch/train_patch_f16.npy}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-26}"

# ---- A PRIORI FIXED.  Never selected on test metrics.
SD_STRENGTH="0.82"        # the established sdedit_ll / g3f operating point
FRLA_STRENGTH="0.95"      # deliberately HIGHER: the FRLA claim is that one scalar
                          # strength cannot serve a 9.9x reliability range
FRLA_ETA="0.85"
GEN_STEPS="28"
GEN_GUIDANCE="5.0"
IP_SCALE="1.0"
W_IP="0.2"                # weak plain-cosine pull: its minimiser is the
                          # conditional mean, and that collapse was measured.
W_IREPA="0.5"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${OUT}/logs" "${OUT}/eval" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"

source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true
PYTHON="$(command -v python)"
echo "[env] PYTHON=${PYTHON}"
"${PYTHON}" - <<'PY'
import sys
try:
    import torch, diffusers
except Exception as e:                                   # noqa: BLE001
    sys.exit(f"[FATAL] torch/diffusers import failed: {e}")
if not torch.cuda.is_available():
    sys.exit("[FATAL] CUDA unavailable. A CPU fallback would silently train on a "
             "different device at a different speed.")
print(f"[env] torch {torch.__version__} diffusers {diffusers.__version__} "
      f"cuda ok: {torch.cuda.get_device_name(0)}")
PY

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }

CONCEPT_GALLERY="${NB_ROOT}/outputs/g2/targets/sem_concept_tmpl_train.npy"
CONCEPT_TEST="${NB_ROOT}/outputs/g2/targets/sem_concept_tmpl_test.npy"
VC="${NB_ROOT}/outputs/sdedit_ll_full10/shared/vae_cache"
CLIP_TEXT="${NB_ROOT}/outputs/nda_ss/sub-08/clip_text"
CAPTIONS="${NB_ROOT}/outputs/g2/captions/captions_train.jsonl"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
SPLIT="${SPLIT:-${NB_ROOT}/outputs/leakfree/split.json}"
require "${VC}/train_vae_latents_f16.npy"
require "${VC}/test_vae_latents_f16.npy"
require "${SPLIT}"
require "${CONCEPT_GALLERY}"

# ---------------------------------------------------------------- [0] leak audit
echo "===== [0] concept-disjointness audit @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import numpy as np, sys
tr = np.load("${CONCEPT_GALLERY}", allow_pickle=True)
te = np.load("${CONCEPT_TEST}", allow_pickle=True)
strs = lambda a: {str(x).strip().lower() for x in a}
a, b = strs(tr), strs(te)
inter = a & b
print(f"[audit] train gallery {len(a)}  test concepts {len(b)}  intersection {len(inter)}")
if inter:
    sys.exit(f"[FATAL] the prompt gallery shares {len(inter)} strings with the test "
             f"concepts, e.g. {sorted(inter)[:5]}. Refusing to generate.")
print("[audit] ok: no emitted prompt can name a test class")
PY

# ------------------------------------------- [1] pure intra encoder (sub-08 only)
ENC_CK="${ENC}/checkpoint_ss_calib_best.pth"
echo "===== [1] pure intra EEG encoder (sub-08 only) @ $(date -Iseconds) ====="
if [[ ! -f "${ENC_CK}" ]]; then
  "${PYTHON}" scripts/nda/nda_ss_pretrain.py \
    --train-subjects 8 --calib-subject 8 \
    --output-dir "${ENC}" --pretrain-epochs 30 --calib-epochs 15 \
    --batch-size 512 --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/enc_intra.log"
else
  echo "[SKIP] encoder exists: ${ENC_CK}"
fi
require "${ENC_CK}"

# --------------------------------------------------- [2] export shared_r (CSM z)
echo "===== [2] export intra shared_r (the cross-subject module) @ $(date -Iseconds) ====="
if [[ ! -f "${ZC}/sub-08/shared_r_test.npy" ]]; then
  "${PYTHON}" scripts/nda/ocf_export_intra_z.py \
    --subject 8 --checkpoint "${ENC_CK}" --out "${ZC}" --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/export_intra_z.log"
else
  echo "[SKIP] z export exists"
fi
require "${ZC}/sub-08/shared_r_test.npy"

# ------------------------------------- [2b] raw EEG cache for the TDM front-end
# THIS IS THE HARD REQUIREMENT: `tdm_train.py` hard-fails without it.  The
# probes in part 2 are MEASUREMENTS and are therefore run with `|| true` -- a
# probe that returns a negative result (or crashes) must not cost the night's
# training run, because the training is the deliverable and the probe is
# evidence about it.
echo "===== [2b] raw EEG cache + timing probes @ $(date -Iseconds) ====="
if [[ ! -f "${RAW}/sub08_train_eeg.npy" ]]; then
  "${PYTHON}" scripts/nda/tdm_gate0.py --subject 8 --device "${DEVICE}" \
    --targets-dir "${NB_ROOT}/outputs/g2/targets" --cache-dir "${RAW}" \
    --out-json "${NB_ROOT}/outputs/tdm/gate0_sub08.json" --cache-only \
    2>&1 | tee "${OUT}/logs/raw_cache.log"
else
  echo "[SKIP] raw EEG cache exists"
fi
require "${RAW}/sub08_train_eeg.npy"

echo "===== [2c] timing probes (JITTER / GAMMA-LINEARITY / GRANULARITY x TIME) ====="
G0="${NB_ROOT}/outputs/tdm/gate0_sub08.json"
if [[ -f "${G0}" ]]; then
  echo "[SKIP] probes already measured: ${G0}"
else
  "${PYTHON}" scripts/nda/tdm_gate0.py --subject 8 --device "${DEVICE}" \
    --targets-dir "${NB_ROOT}/outputs/g2/targets" --cache-dir "${RAW}" \
    --out-json "${G0}" \
    2>&1 | tee "${OUT}/logs/gate0.log" || echo "[WARN] probe failed; see gate0.log"
fi

# ------------------------------- [3] CLIP ViT-H-14 patch tokens (TRAIN images only)
echo "===== [3] iREPA targets: CLIP patch tokens (train only) @ $(date -Iseconds) ====="
if [[ ! -f "${PATCH}" ]]; then
  "${PYTHON}" scripts/nda/tdm_clip_patch.py \
    --captions-jsonl "${CAPTIONS}" --out "${PATCH}" --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/clip_patch.log"
else
  echo "[SKIP] patch tokens exist"
fi
# NOT `require`: iREPA reports itself as skipped rather than fake-zeroing, so a
# missing cache degrades the design instead of killing the run.
[[ -f "${PATCH}" ]] && echo "[ok] iREPA targets present" || echo "[WARN] no iREPA targets"

# -------------------------------------------------- [4] TDM-DT (full arm), intra
TD="${OUT}/tdm"
echo "===== [4] TDM-DT full arm (intra) @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/tdm_train.py \
  --train-subjects 8 --test-subject 8 \
  --z-root "${ZC}" --z-source shared_r --raw-cache "${RAW}" \
  --targets-dir "${NB_ROOT}/outputs/g2/targets" \
  --clip-text-dir "${CLIP_TEXT}" --captions-jsonl "${CAPTIONS}" \
  --clip-patch-npy "${PATCH}" \
  --out "${TD}" --epochs "${EPOCHS}" --device "${DEVICE}" \
  --w-ip "${W_IP}" --w-irepa "${W_IREPA}" --resume 0 \
  2>&1 | tee "${OUT}/logs/tdm_train_full.log"
require "${TD}/conds/ip_fused_test.npy"

# ------------------------------------------------------- [5] ablation arm, intra
AB="${OUT}/tdm_abl"
echo "===== [5] ablation arm (DLA/RSD/DNG/gates/hub/iREPA off) @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/tdm_train.py \
  --train-subjects 8 --test-subject 8 \
  --z-root "${ZC}" --z-source shared_r --raw-cache "${RAW}" \
  --targets-dir "${NB_ROOT}/outputs/g2/targets" \
  --clip-text-dir "${CLIP_TEXT}" --captions-jsonl "${CAPTIONS}" \
  --out "${AB}" --epochs "${EPOCHS}" --device "${DEVICE}" \
  --w-ip "${W_IP}" --ablation all --resume 0 \
  2>&1 | tee "${OUT}/logs/tdm_train_abl.log"
require "${AB}/conds/ip_fused_test.npy"

# ------------------------------------------------- [6] VAE low-level head, intra
VH="${OUT}/vae_head"
echo "===== [6] VAE low-level head (intra) @ $(date -Iseconds) ====="
if [[ ! -f "${VH}/pred_lowlevel_rgb_512/199.png" ]]; then
  mkdir -p "${VH}"
  # `--val-split-json` IS NOT OPTIONAL: without it this script selects the
  # checkpoint on TEST-set MAE and labels the run "test(contaminated)", which is
  # the selection bias the leak-free audit removed.
  "${PYTHON}" scripts/nda/train_eeg_vae_head.py \
    --eeg-train-npy "${ZC}/sub-08/shared_r_train.npy" \
    --eeg-test-npy  "${ZC}/sub-08/shared_r_test.npy" \
    --vae-train-npy "${VC}/train_vae_latents_f16.npy" \
    --vae-test-npy  "${VC}/test_vae_latents_f16.npy" \
    --val-split-json "${SPLIT}" \
    --output-dir "${VH}" --decode-rgb --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/vae_head_intra.log"
else
  echo "[SKIP] vae head exists"
fi
# runs OUTSIDE the branch on purpose: a head left over from an earlier run would
# otherwise be skipped straight past
"${PYTHON}" - <<PYCHK
import json, sys
d = json.load(open("${VH}/vae_head_report.json"))
sel = d["final"]["selected_on"]
print(f"[check] vae head selected_on = {sel} (best_epoch {d['best_epoch']})")
if sel != "val":
    sys.exit(f"[FATAL] VAE head selected on '{sel}'. That is test-set selection "
             f"bias; the run is contaminated and must not be reported.")
PYCHK
require "${VH}/pred_lowlevel_rgb_512/199.png"
require "${VH}/pred_vae_test.npy"

# ------------------------------------------------------------------ [7] generate
gen_ll() {                       # tag cond prompt_file [strength]
  local tag="$1" cond="$2" pf="$3" strength="${4:-${SD_STRENGTH}}"
  local gdir="${OUT}/gen/${STAG}/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] gen ${tag}"; return 0; fi
  [[ -f "${cond}" ]] || { echo "[WARN] ${tag}: missing cond ${cond}"; return 1; }
  [[ -f "${pf}" ]] || { echo "[WARN] ${tag}: missing prompts ${pf}"; return 1; }
  echo "===== gen ${tag} (strength ${strength}) @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
    --mode sdedit --embed-npy "${cond}" --prompts-json "${pf}" \
    --lowlevel-rgb-dir "${VH}/pred_lowlevel_rgb_512" \
    --output-dir "${gdir}" --tag "${tag}" \
    --strength "${strength}" --ip-scale "${IP_SCALE}" \
    --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
    --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/gen_${tag}.log"
}

gen_frla() {                     # tag arm cond prompt_file
  local tag="$1" arm="$2" cond="$3" pf="$4"
  local gdir="${OUT}/gen/${STAG}/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] gen ${tag}"; return 0; fi
  [[ -f "${cond}" ]] || { echo "[WARN] ${tag}: missing cond"; return 1; }
  echo "===== gen ${tag} (arm=${arm}) @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/idg_frla_decode.py \
    --embed-npy "${cond}" \
    --anchor-latent-npy "${VH}/pred_vae_test.npy" \
    --lowlevel-rgb-dir "${VH}/pred_lowlevel_rgb_512" \
    --prompts-json "${pf}" \
    --output-dir "${gdir}" --tag "${tag}" --arm "${arm}" --eta "${FRLA_ETA}" \
    --strength "${FRLA_STRENGTH}" --gen-steps "${GEN_STEPS}" \
    --gen-guidance "${GEN_GUIDANCE}" --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/gen_${tag}.log"
}

echo "===== [7] generation @ $(date -Iseconds) ====="
# headline: fused condition (both towers) + self prompt from the TRAIN gallery
gen_ll tdm_ll_self "${TD}/conds/ip_fused_test.npy"     "${TD}/prompts/prompts_self.json" || true
# the prompt ablation: identical condition, no concept text
gen_ll tdm_ll_gen  "${TD}/conds/ip_fused_test.npy"     "${TD}/prompts/prompts_generic.json" || true
# theta solved on held-in train rows to match the reference manifold
gen_ll tdm_th_star "${TD}/conds/ip_fused_star_test.npy" "${TD}/prompts/prompts_self.json" || true
# the matched control: same pipeline, every new mechanism off
gen_ll tdm_abl_self "${AB}/conds/ip_fused_test.npy"    "${AB}/prompts/prompts_self.json" || true
# FRLA: the anchoring law is the ONLY difference between these three, and
# `sdedit_ll_095` is their plain-SDEdit control at their own strength.
gen_ll sdedit_ll_095 "${TD}/conds/ip_fused_test.npy"   "${TD}/prompts/prompts_self.json" "${FRLA_STRENGTH}" || true
gen_frla frla_off     off     "${TD}/conds/ip_fused_test.npy" "${TD}/prompts/prompts_self.json" || true
gen_frla frla_uniform uniform "${TD}/conds/ip_fused_test.npy" "${TD}/prompts/prompts_self.json" || true
gen_frla frla_frla    frla    "${TD}/conds/ip_fused_test.npy" "${TD}/prompts/prompts_self.json" || true

# -------------------------------------------------------------------- [8] evaluate
echo "===== [8] evaluation @ $(date -Iseconds) ====="
eval_row() {                     # tag gen_dir
  local tag="$1" gdir="$2"
  [[ -f "${gdir}/199.png" ]] || { echo "[WARN] eval ${tag}: no images"; return 0; }
  echo "--- eval ${tag}"
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${gdir}" --output-json "${OUT}/eval/${tag}.json" --tag "${tag}" \
    --images-root "${IMAGES_ROOT}" --device "${DEVICE}" --skip-if-exists \
    2>&1 | tee -a "${OUT}/logs/eval.log" || echo "[WARN] eval failed: ${tag}"
}
for t in tdm_ll_self tdm_ll_gen tdm_th_star tdm_abl_self sdedit_ll_095 \
         frla_off frla_uniform frla_frla; do
  eval_row "${t}" "${OUT}/gen/${STAG}/${t}/generated"
done
# reference rows, re-scored by the SAME code and the SAME GT cache
eval_row sdedit_ll "${NB_ROOT}/outputs/sdedit_ll_full10/sub-08/generation/sdedit_ll/generated"
eval_row g3f_ll_selfgate "${NB_ROOT}/outputs/g3f/gen/sub-08/g3f_ll_selfgate/generated"

# -------------------------------------------------------------------- [9] summary
echo "===== [9] summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json, glob, os
rows = []
for p in sorted(glob.glob("${OUT}/eval/*.json")):
    d = json.load(open(p)); m = d.get("metrics", d)
    rows.append((os.path.basename(p)[:-5],
                 m.get("pixcorr"), m.get("ssim"), m.get("alexnet2"),
                 m.get("alexnet5"), m.get("inception"), m.get("clip"), m.get("swav")))
hdr = ("row", "PixCorr", "SSIM", "Alex2", "Alex5", "Inc", "CLIP", "SwAV")
print("".join(f"{h:>18}" for h in hdr))
for r in rows:
    print("".join(f"{v:>18.4f}" if isinstance(v, float) else f"{str(v):>18}" for v in r))
# mechanism verdicts, printed together so a claim and its control are adjacent
for tag, p in (("FULL", "${TD}/tdm_report.json"), ("ABLATION", "${AB}/tdm_report.json")):
    if not os.path.isfile(p):
        continue
    d = json.load(open(p)); mech = d.get("mechanisms", {})
    print(f"\n--- {tag}: best {d['best'].get('score'):.4f} @ ep{d['best'].get('epoch')}, "
          f"irepa {d.get('irepa')}")
    g = mech.get("granularity_time_claim", {})
    if g:
        print(f"    granularity mean gate time (ms): " +
              ", ".join(f"{k}={v:.0f}" for k, v in g.get("mean_time_ms", {}).items()))
        print(f"    claim overall<background<subject<detail: {g.get('pass')} | "
              f"detail>overall: {g.get('pass_detail_after_overall')}")
    if "dla" in mech:
        print(f"    DLA band mean |tau| ms: " +
              ", ".join(f"{v:.2f}" for v in mech['dla']['per_band_mean_abs_ms']) +
              f" | non-degenerate {mech['dla']['pass_non_degenerate']}"
              f" | LF>HF {mech['dla']['pass_lf_gt_hf']}")
    if "rsd" in mech:
        print(f"    RSD band off-diagonal: " +
              ", ".join(f"{v:.3f}" for v in mech['rsd']['per_band_offdiag_norm']) +
              f" | HF>LF {mech['rsd']['pass_hf_gt_lf']}")
    print(f"    ip_fused 2-way {d['ip_fused_disc']['twoway']:.4f} top1 "
          f"{d['ip_fused_disc']['top1']:.4f} | hub_skew "
          f"{d['hubness']['hub_skew']:.3f} | never-retrieved "
          f"{d['hubness']['n_never_retrieved']}")
PY
echo "===== done @ $(date -Iseconds) ====="
