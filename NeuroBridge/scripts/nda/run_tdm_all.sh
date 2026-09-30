#!/usr/bin/env bash
# ============================================================================
# TDM-DT / ALL 10 SUBJECTS -- overnight chain, leak-free, disk-frugal.
#
# PROTOCOL: intra-subject, for every subject in 1..10.  Each subject gets its own
# EEG encoder (`shared_r` of a shared/specific model trained on that subject
# alone), its own TDM-DT read-out, its own VAE low-level head, and its own
# generation rows.  This is the per-subject protocol every SOTA number on
# THINGS-EEG2 is reported under, and it is the direct extension of the sub-08 run.
#
# THE FRAMEWORK (unchanged, and now the reason this replicates per subject)
#   cross-subject module  = `shared_r`, the shared branch of the intra encoder
#   dual-tower encoding   = SEMANTIC (image CLIP + overall/subject/background/
#                           detail descriptions) and STRUCTURAL (image CLIP spatial
#                           patch tokens + SDXL VAE low-frequency latent)
#   fused condition       = `h_fuse` reads the semantic code, all four
#                           granularities, the concept, the image vector AND the
#                           structural spatial map, then `spherical()` places the
#                           result at an angle theta solved on held-in train rows
#
# WHAT IS NEW (encoder / alignment / control -- the three things asked for)
#   [A] DLA  per-band, per-channel group delay as a Fourier phase ramp
#   [B] RSD  per-band linear spatial unmixing, identity at init
#   [C] DNG  alpha-derived spatially global gain that DIVIDES the evidence
#   [D] GRANULARITY x TIME gates -- each granularity head attends over the 25
#            (band, time-patch) tokens with its own softmax gate and CONSUMES it,
#            so the gate is in the gradient path and the central claim
#                 mean gate time: overall < background < subject < detail
#            is measurable.  A gate no loss reads could never be learned, which is
#            why it was rewired into `h_<granularity>` rather than left as a
#            reporting side-branch.
#   [E] iREPA -- the same 64 spatial tokens that decode the (4,64,64) latent are
#            aligned to the real image's 8x8 CLIP ViT-H-14 patch tokens
#   [F] HUB-AWARE InfoNCE -- local scaling against the hubness created by 1654
#            concepts with 10 near-duplicate trials each
#
# THE CONTROL, PER SUBJECT
#   `tdm_abl_self` is generated and scored by the SAME generation and metric code
#   from an `--ablation all` arm trained by the same job on the same rows, with
#   DLA/RSD/DNG/time-gates/hub-loss/iREPA off.  That pair is the headline claim
#   and it exists for all 10 subjects, so the comparison is paired across subjects
#   rather than asserted from one.
#
# GRADIENT-FREEZE GUARD (the defect that produced the previous unusable run)
#   `l2t` used to floor the norm at 1e-8, i.e. a 1e8 Jacobian at the zero vector,
#   which is exactly what every zero-initialised read-out produced at step 0.  The
#   fp32 total gradient norm overflowed to `inf`, so `clip_grad_norm_` set its
#   coefficient to 0 and multiplied EVERY gradient by zero: 754 steps, no
#   parameter movement, an untrained random projection exported as the condition.
#   Three things now prevent it: the norm floor is 1e-3 (bounded Jacobian), the
#   read-out heads start from small random weights instead of exactly zero, and
#   `tdm_train.py` runs a PREFLIGHT forward+backward that ABORTS if the gradient
#   total is non-finite or no parameter has a nonzero gradient.  Every epoch also
#   prints a GRAD-SKIPS counter and the report carries `trained`.
#
# DISK DISCIPLINE -- measured budget, because /project is at 100% of 50T
#   kept per subject (~0.2 GB): conds (~12 MB), reports, eval JSONs, best.pth (2 x
#      77 MB)
#   DELETED as soon as it is no longer needed:
#     * `last.pth`            -- after the export, it is only a resume artifact
#     * raw EEG cache (281 MB) -- after that subject's training (regenerable with
#                                `tdm_gate0.py --cache-only`, ~90 s)
#     * generated PNGs (133 MB PER ROW) -- immediately after their metrics are
#                                written; the metrics are what the run produces
#     * VAE low-level RGB init (77 MB) and `pred_vae_test.npy` -- after that
#                                subject's LAST generation row
#     * z exports other than `shared_r_*` (135 MB of the 203 MB) -- after training
#     * `checkpoint_ss_pretrain_best.pth` (62 MB) -- after the shared_r export
#   shared ONCE for all subjects: the CLIP patch-token cache (2.7 GB) -- the
#   targets are image-level, so it is not per-subject.
#   Typical residency is therefore ~1 subject of scratch (~0.5 GB) instead of
#   ~11 GB of generated images per subject.
#
# LEAK-FREE
#   * encoder, TDM heads and VAE head fit that subject's train rows only;
#   * the prompt gallery is the 1654 TRAIN concepts, asserted DISJOINT from the
#     200 test concepts in stage [0];
#   * CLIP patch tokens are extracted for TRAIN images only;
#   * theta is solved on held-out TRAIN rows against a TRAIN statistic;
#   * the VAE head selects its checkpoint on a held-in TRAIN split and the run
#     hard-fails if it regresses to test-set selection;
#   * all generation hyper-parameters are fixed a priori below.
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/tdm_all}"
ENC_BASE="${NB_ROOT}/outputs/ocf/intra_enc"
ZC_BASE="${NB_ROOT}/outputs/ocf/intra_z"
RAW="${NB_ROOT}/outputs/tdm/cache"
PATCH="${NB_ROOT}/outputs/tdm/clip_patch/train_patch_f16.npy"
G0="${NB_ROOT}/outputs/tdm/gate0_sub08.json"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-26}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
# the one subject that carries the full ablation grid (FRLA arms + prompt arm);
# the other nine carry the three rows the paired claims need, which is the
# difference between a ~7 h and a ~14 h night
ABL_SUBJECT="${ABL_SUBJECT:-8}"
# set to 1 to keep the generated PNGs for inspection (costs ~11 GB)
KEEP_IMAGES="${KEEP_IMAGES:-0}"

# ---- A PRIORI FIXED.  Never selected on test metrics.
SD_STRENGTH="0.82"        # the established sdedit_ll / g3f operating point
FRLA_STRENGTH="0.95"      # deliberately HIGHER: the FRLA claim is that ONE scalar
                          # strength cannot serve a 9.9x reliability range
FRLA_ETA="0.85"
GEN_STEPS="28"
GEN_GUIDANCE="5.0"
IP_SCALE="1.0"
W_IP="0.2"                # weak plain-cosine pull: its minimiser is the
                          # conditional mean, and that collapse was measured
W_IREPA="0.5"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${OUT}/logs" "${OUT}/eval" "${OUT}/done" "${NB_ROOT}/outputs/slurm"
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

# ---------------------- [1] shared, image-level: CLIP patch tokens (train only)
echo "===== [1] iREPA targets: CLIP ViT-H-14 patch tokens (train images only) ====="
if [[ ! -f "${PATCH}" ]]; then
  "${PYTHON}" scripts/nda/tdm_clip_patch.py \
    --captions-jsonl "${CAPTIONS}" --out "${PATCH}" --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/clip_patch.log"
else
  echo "[SKIP] patch tokens exist ($(du -h "${PATCH}" | cut -f1))"
fi
# NOT `require`: iREPA reports itself as skipped rather than fake-zeroing
[[ -f "${PATCH}" ]] && echo "[ok] iREPA targets present" || echo "[WARN] no iREPA targets"

# ============================================================================
# per-subject body
# ============================================================================
gen_ll() {                       # subject tag cond prompt_file [strength]
  local sid="$1" tag="$2" cond="$3" pf="$4" strength="${5:-${SD_STRENGTH}}"
  local gdir="${OUT}/gen/sub-$(printf '%02d' "${sid}")/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] gen ${tag}"; return 0; fi
  if [[ -f "${OUT}/eval/s$(printf '%02d' "${sid}")_${tag}.json" ]]; then
    echo "[SKIP] gen ${tag} (already scored and cleaned up)"; return 0
  fi
  [[ -f "${cond}" ]] || { echo "[WARN] ${tag}: missing cond ${cond}"; return 1; }
  [[ -f "${pf}" ]] || { echo "[WARN] ${tag}: missing prompts ${pf}"; return 1; }
  echo "===== gen s${sid} ${tag} (strength ${strength}) @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
    --mode sdedit --embed-npy "${cond}" --prompts-json "${pf}" \
    --lowlevel-rgb-dir "${OUT}/vae_head/sub-$(printf '%02d' "${sid}")/pred_lowlevel_rgb_512" \
    --output-dir "${gdir}" --tag "${tag}" \
    --strength "${strength}" --ip-scale "${IP_SCALE}" \
    --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
    --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/gen_s${sid}_${tag}.log"
}

gen_frla() {                     # subject tag arm cond prompt_file
  local sid="$1" tag="$2" arm="$3" cond="$4" pf="$5"
  local gdir="${OUT}/gen/sub-$(printf '%02d' "${sid}")/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] gen ${tag}"; return 0; fi
  if [[ -f "${OUT}/eval/s$(printf '%02d' "${sid}")_${tag}.json" ]]; then
    echo "[SKIP] gen ${tag} (already scored and cleaned up)"; return 0
  fi
  [[ -f "${cond}" ]] || { echo "[WARN] ${tag}: missing cond"; return 1; }
  echo "===== gen s${sid} ${tag} (arm=${arm}) @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/idg_frla_decode.py \
    --embed-npy "${cond}" \
    --anchor-latent-npy "${OUT}/vae_head/sub-$(printf '%02d' "${sid}")/pred_vae_test.npy" \
    --lowlevel-rgb-dir "${OUT}/vae_head/sub-$(printf '%02d' "${sid}")/pred_lowlevel_rgb_512" \
    --prompts-json "${pf}" \
    --output-dir "${gdir}" --tag "${tag}" --arm "${arm}" --eta "${FRLA_ETA}" \
    --strength "${FRLA_STRENGTH}" --gen-steps "${GEN_STEPS}" \
    --gen-guidance "${GEN_GUIDANCE}" --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/gen_s${sid}_${tag}.log"
}

eval_row() {                     # subject tag gen_dir
  local sid="$1" tag="$2" gdir="$3"
  local out="${OUT}/eval/s$(printf '%02d' "${sid}")_${tag}.json"
  [[ -f "${out}" ]] && { echo "[SKIP] eval ${tag}"; return 0; }
  [[ -f "${gdir}/199.png" ]] || { echo "[WARN] eval ${tag}: no images"; return 0; }
  echo "--- eval ${tag}"
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${gdir}" --output-json "${out}" --tag "${tag}" \
    --images-root "${IMAGES_ROOT}" --device "${DEVICE}" --skip-if-exists \
    2>&1 | tee -a "${OUT}/logs/eval_s${sid}.log" || { echo "[WARN] eval failed: ${tag}"; return 0; }
  # delete the images ONLY after the metrics exist, and only after the file has
  # actually been written -- a failed evaluation must not cost the night's images
  if [[ -s "${out}" && "${KEEP_IMAGES}" != "1" ]]; then
    rm -rf "${gdir%/generated}"
  fi
}

cleanup_subject() {              # subject
  local sid="$1" sd
  sd="$(printf '%02d' "${sid}")"
  # raw EEG cache: regenerable in ~90 s, 281 MB per subject
  rm -f "${RAW}/sub${sd}_train_eeg.npy" "${RAW}/sub${sd}_test_eeg.npy" \
        "${RAW}/sub${sd}_train_row.npy" "${RAW}/sub${sd}_test_row.npy"
  # resume artifacts (the checkpoints are what matter)
  rm -f "${OUT}/tdm/sub-${sd}/last.pth" "${OUT}/tdm_abl/sub-${sd}/last.pth"
  # z exports the pipeline does not use: only `shared_r` feeds the CSM
  rm -f "${ZC_BASE}/sub-${sd}/fused_"*.npy "${ZC_BASE}/sub-${sd}/specific_s_"*.npy \
        "${ZC_BASE}/sub-${sd}/z_eeg_proj_"*.npy
  # encoder pretrain checkpoint, superseded by the calibration checkpoint
  rm -f "${ENC_BASE}/sub-${sd}/checkpoint_ss_pretrain_best.pth"
  # VAE head scratch: the RGB init and the test latent are only needed to generate
  rm -rf "${OUT}/vae_head/sub-${sd}/pred_lowlevel_rgb_512"
  rm -f "${OUT}/vae_head/sub-${sd}/pred_vae_train.npy"
  echo "[cleanup] sub-${sd}: scratch removed; kept reports, conds, eval JSONs, best.pth"
  du -sh "${OUT}" 2>/dev/null | sed 's/^/[cleanup] OUT now /'
}

run_subject() {
  local SID="$1" sd TD AB VH ENC_CK ZC
  sd="$(printf '%02d' "${SID}")"
  TD="${OUT}/tdm/sub-${sd}"
  AB="${OUT}/tdm_abl/sub-${sd}"
  VH="${OUT}/vae_head/sub-${sd}"
  ENC_CK="${ENC_BASE}/sub-${sd}/checkpoint_ss_calib_best.pth"
  ZC="${ZC_BASE}/sub-${sd}"
  mkdir -p "${TD}/conds" "${TD}/prompts" "${AB}/conds" "${AB}/prompts" \
           "${VH}" "${OUT}/gen/sub-${sd}"

  echo; echo "######################### SUB-${sd} @ $(date -Iseconds) #########################"

  # ---- [2] pure intra encoder (this subject only; 62 MB checkpoint)
  if [[ ! -f "${ENC_CK}" ]]; then
    echo "===== [2] intra EEG encoder (sub-${sd} alone) ====="
    "${PYTHON}" scripts/nda/nda_ss_pretrain.py \
      --train-subjects "${SID}" --calib-subject "${SID}" \
      --output-dir "${ENC_BASE}/sub-${sd}" \
      --pretrain-epochs 30 --calib-epochs 15 \
      --batch-size 512 --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/enc_s${sd}.log"
  else
    echo "[SKIP] encoder exists"
  fi
  require "${ENC_CK}"

  # ---- [3] export shared_r (the cross-subject module input)
  if [[ ! -f "${ZC}/shared_r_test.npy" ]]; then
    echo "===== [3] export shared_r ====="
    "${PYTHON}" scripts/nda/ocf_export_intra_z.py \
      --subject "${SID}" --checkpoint "${ENC_CK}" --out "${ZC_BASE}" \
      --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/export_z_s${sd}.log"
  else
    echo "[SKIP] z export exists"
  fi
  require "${ZC}/shared_r_test.npy"

  # ---- [4] raw EEG cache (hard requirement) + cheap per-subject timing probes
  echo "===== [4] raw EEG cache ====="
  if [[ ! -f "${RAW}/sub${sd}_train_eeg.npy" ]]; then
    "${PYTHON}" scripts/nda/tdm_gate0.py --subject "${SID}" --device "${DEVICE}" \
      --targets-dir "${NB_ROOT}/outputs/g2/targets" --cache-dir "${RAW}" \
      --out-json "${NB_ROOT}/outputs/tdm/gate0_sub${sd}.json" --cache-only \
      2>&1 | tee "${OUT}/logs/raw_cache_s${sd}.log"
  else
    echo "[SKIP] raw EEG cache exists"
  fi
  require "${RAW}/sub${sd}_train_eeg.npy"

  # G1 (latency jitter) and G2 (gamma linearity) are what decide whether DLA and
  # RSD have a premise at all.  They are cheap (~1 min) and are measured for EVERY
  # subject, because "the premise holds on sub-08" is exactly the kind of
  # single-subject result this project has been burned by.  G3 (the granularity x
  # time sweep) is the expensive one and runs for the reference subject only -- but
  # it DOES run there, because that is the subject every ablation row is generated
  # on, so its gate has to be measured under the same gate0 code as the others.
  echo "===== [4b] timing probes (G1 jitter / G2 linearity) ====="
  G3FLAG="--skip-g3"
  [[ "${SID}" == "${ABL_SUBJECT}" ]] && G3FLAG=""
  "${PYTHON}" scripts/nda/tdm_gate0.py --subject "${SID}" --device "${DEVICE}" \
    --targets-dir "${NB_ROOT}/outputs/g2/targets" --cache-dir "${RAW}" \
    --out-json "${NB_ROOT}/outputs/tdm/gate0_sub${sd}.json" ${G3FLAG} \
    2>&1 | tee "${OUT}/logs/gate0_s${sd}.log" \
    || echo "[WARN] probe failed for sub-${sd}; see gate0_s${sd}.log"

  # what did the premise measurement actually authorise?  Printed per subject so a
  # gate that flips between subjects is visible in the log, not just in JSON.
  "${PYTHON}" - <<PYGATE
import json
d = json.load(open("${NB_ROOT}/outputs/tdm/gate0_sub${sd}.json"))
v = d.get("verdict", {})
print("  [gate0] sub-${sd} verdict: " + "  ".join(f"{k}={val}" for k, val in v.items()))
PYGATE

  # ---- [5] TDM-DT full arm.  `--resume 1` (the default) so a wall-time requeue
  # continues instead of retraining; the PREFLIGHT inside aborts if the gradient
  # is non-finite, which is the failure that made the previous run unusable.
  echo "===== [5] TDM-DT full arm (sub-${sd}) ====="
  "${PYTHON}" scripts/nda/tdm_train.py \
    --train-subjects "${SID}" --test-subject "${SID}" \
    --z-root "${ZC_BASE}" --z-source shared_r --raw-cache "${RAW}" \
    --targets-dir "${NB_ROOT}/outputs/g2/targets" \
    --clip-text-dir "${CLIP_TEXT}" --captions-jsonl "${CAPTIONS}" \
    --clip-patch-npy "${PATCH}" \
    --out "${TD}" --epochs "${EPOCHS}" --device "${DEVICE}" \
    --w-ip "${W_IP}" --w-irepa "${W_IREPA}" \
    2>&1 | tee "${OUT}/logs/tdm_train_full_s${sd}.log"
  require "${TD}/conds/ip_fused_test.npy"

  # ---- [6] ablation arm: same code, same rows, every new mechanism off
  echo "===== [6] ablation arm (sub-${sd}) ====="
  "${PYTHON}" scripts/nda/tdm_train.py \
    --train-subjects "${SID}" --test-subject "${SID}" \
    --z-root "${ZC_BASE}" --z-source shared_r --raw-cache "${RAW}" \
    --targets-dir "${NB_ROOT}/outputs/g2/targets" \
    --clip-text-dir "${CLIP_TEXT}" --captions-jsonl "${CAPTIONS}" \
    --out "${AB}" --epochs "${EPOCHS}" --device "${DEVICE}" \
    --w-ip "${W_IP}" --ablation all \
    2>&1 | tee "${OUT}/logs/tdm_train_abl_s${sd}.log"
  require "${AB}/conds/ip_fused_test.npy"

  # ---- [6b] did either arm actually train?  This is not a formality: the whole
  # previous run looked normal in its logs while frozen.  A subject whose
  # `trained` flag is false is reported and its rows are still generated (so the
  # failure is visible in the metrics rather than missing), but the flag is in the
  # report and in the aggregate.
  "${PYTHON}" - <<PYCHK
import json, sys
for arm, p in (("full", "${TD}/tdm_report.json"), ("ablation", "${AB}/tdm_report.json")):
    d = json.load(open(p))
    ok = d.get("trained") and d.get("grad_skips", 0) == 0
    print(f"  [check] sub-${sd} {arm:<9} trained={d.get('trained')} "
          f"grad_skips={d.get('grad_skips')} best_score={d['best'].get('score'):.4f} "
          f"image={d['best'].get('va_image', float('nan')):.4f} "
          f"fused_2way={d['ip_fused_disc']['twoway']:.4f}")
    if not ok:
        print(f"  [WARN] sub-${sd} {arm}: the model did NOT train cleanly. The row "
              f"will still be generated but must be reported as such.")
PYCHK

  # ---- [7] VAE low-level head (this subject only)
  echo "===== [7] VAE low-level head (sub-${sd}) ====="
  if [[ ! -f "${VH}/pred_lowlevel_rgb_512/199.png" ]]; then
    # `--val-split-json` IS NOT OPTIONAL: without it this script selects the
    # checkpoint on TEST-set MAE and labels the run "test(contaminated)"
    "${PYTHON}" scripts/nda/train_eeg_vae_head.py \
      --eeg-train-npy "${ZC}/shared_r_train.npy" \
      --eeg-test-npy  "${ZC}/shared_r_test.npy" \
      --vae-train-npy "${VC}/train_vae_latents_f16.npy" \
      --vae-test-npy  "${VC}/test_vae_latents_f16.npy" \
      --val-split-json "${SPLIT}" \
      --output-dir "${VH}" --decode-rgb --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/vae_head_s${sd}.log"
  else
    echo "[SKIP] vae head exists"
  fi
  # runs OUTSIDE the branch on purpose: a head left over from an earlier run would
  # otherwise be skipped straight past
  "${PYTHON}" - <<PYCHK
import json, sys
d = json.load(open("${VH}/vae_head_report.json"))
sel = d["final"]["selected_on"]
print(f"  [check] sub-${sd} vae head selected_on = {sel} (best_epoch {d['best_epoch']})")
if sel != "val":
    sys.exit(f"[FATAL] VAE head selected on '{sel}'. That is test-set selection "
             f"bias; the run is contaminated and must not be reported.")
PYCHK
  require "${VH}/pred_lowlevel_rgb_512/199.png"
  require "${VH}/pred_vae_test.npy"

  # ---- [8] generation
  echo "===== [8] generation (sub-${sd}) @ $(date -Iseconds) ====="
  # the three rows every subject carries: the headline pair (full vs ablation,
  # which is the paired claim) and the solved-angle row
  gen_ll   "${SID}" tdm_ll_self   "${TD}/conds/ip_fused_test.npy"      "${TD}/prompts/prompts_self.json" || true
  gen_ll   "${SID}" tdm_abl_self  "${AB}/conds/ip_fused_test.npy"      "${AB}/prompts/prompts_self.json" || true
  gen_ll   "${SID}" tdm_th_star   "${TD}/conds/ip_fused_star_test.npy" "${TD}/prompts/prompts_self.json" || true
  if [[ "${SID}" == "${ABL_SUBJECT}" ]]; then
    # prompt ablation: identical condition, no concept text
    gen_ll   "${SID}" tdm_ll_gen    "${TD}/conds/ip_fused_test.npy"      "${TD}/prompts/prompts_generic.json" || true
    # FRLA: `sdedit_ll_095` is the plain-SDEdit control at the FRLA arms' own
    # strength, so the anchoring law is the only difference between these four
    gen_ll   "${SID}" sdedit_ll_095 "${TD}/conds/ip_fused_test.npy"      "${TD}/prompts/prompts_self.json" "${FRLA_STRENGTH}" || true
    gen_frla "${SID}" frla_off      off     "${TD}/conds/ip_fused_test.npy" "${TD}/prompts/prompts_self.json" || true
    gen_frla "${SID}" frla_uniform  uniform "${TD}/conds/ip_fused_test.npy" "${TD}/prompts/prompts_self.json" || true
    gen_frla "${SID}" frla_frla     frla    "${TD}/conds/ip_fused_test.npy" "${TD}/prompts/prompts_self.json" || true
  fi

  # ---- [9] evaluate (deletes each row's images as soon as its metrics land)
  echo "===== [9] evaluation (sub-${sd}) @ $(date -Iseconds) ====="
  local gb="${OUT}/gen/sub-${sd}"
  eval_row "${SID}" tdm_ll_self  "${gb}/tdm_ll_self/generated"
  eval_row "${SID}" tdm_abl_self "${gb}/tdm_abl_self/generated"
  eval_row "${SID}" tdm_th_star  "${gb}/tdm_th_star/generated"
  if [[ "${SID}" == "${ABL_SUBJECT}" ]]; then
    eval_row "${SID}" tdm_ll_gen    "${gb}/tdm_ll_gen/generated"
    eval_row "${SID}" sdedit_ll_095 "${gb}/sdedit_ll_095/generated"
    eval_row "${SID}" frla_off      "${gb}/frla_off/generated"
    eval_row "${SID}" frla_uniform  "${gb}/frla_uniform/generated"
    eval_row "${SID}" frla_frla     "${gb}/frla_frla/generated"
  fi
  # reference rows, re-scored once by the SAME code and the SAME GT cache.
  # `eval_row` is given the row's own tag and writes `s00_<tag>.json`, so the
  # reference rows are NOT deleted (they live outside OUT and are shared).
  if [[ ! -f "${OUT}/eval/s00_sdedit_ll.json" ]]; then
    "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
      --gen-dir "${NB_ROOT}/outputs/sdedit_ll_full10/sub-08/generation/sdedit_ll/generated" \
      --output-json "${OUT}/eval/s00_sdedit_ll.json" --tag "sdedit_ll" \
      --images-root "${IMAGES_ROOT}" --device "${DEVICE}" --skip-if-exists \
      2>&1 | tee -a "${OUT}/logs/eval_refs.log" || echo "[WARN] ref eval failed"
  fi
  if [[ ! -f "${OUT}/eval/s00_g3f_ll_selfgate.json" ]]; then
    "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
      --gen-dir "${NB_ROOT}/outputs/g3f/gen/sub-08/g3f_ll_selfgate/generated" \
      --output-json "${OUT}/eval/s00_g3f_ll_selfgate.json" --tag "g3f_ll_selfgate" \
      --images-root "${IMAGES_ROOT}" --device "${DEVICE}" --skip-if-exists \
      2>&1 | tee -a "${OUT}/logs/eval_refs.log" || echo "[WARN] ref eval failed"
  fi

  # ---- [10] per-subject cleanup (only AFTER evaluation has consumed the scratch)
  cleanup_subject "${SID}"
  touch "${OUT}/done/sub-${sd}.done"
}

# ============================================================================
for SID in ${SUBJECTS}; do
  sd="$(printf '%02d' "${SID}")"
  if [[ -f "${OUT}/done/sub-${sd}.done" ]]; then
    echo "[SKIP] sub-${sd} already complete"
    continue
  fi
  # one subject failing must not cost the rest of the night: the failure is
  # recorded, the loop continues, and the aggregate reports the subject as missing
  if ! run_subject "${SID}"; then
    echo "[ERROR] sub-${sd} failed; recorded and continuing to the next subject"
    mkdir -p "${OUT}/failed"; date -Iseconds > "${OUT}/failed/sub-${sd}.txt"
  fi
done

# -------------------------------------------------------------------- [11] summary
echo; echo "===== [11] aggregate @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json, glob, os, statistics as st

METS = ("pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav")

def load(tag):
    out = {}
    for p in glob.glob("${OUT}/eval/s??_%s.json" % tag):
        s = int(os.path.basename(p)[2:4])
        d = json.load(open(p))
        m = d.get("metrics", d)
        out[s] = {k: m.get(k) for k in METS}
    return out

rows = ["tdm_ll_self", "tdm_abl_self", "tdm_th_star", "tdm_ll_gen",
        "sdedit_ll_095", "frla_off", "frla_uniform", "frla_frla"]
data = {t: load(t) for t in rows}
subs = sorted({s for t in rows for s in data[t] if s != 0})

print(f"{'row':<16}" + "".join(f"{k:>10}" for k in METS) + f"{'n_subj':>8}")
for t in rows:
    v = data[t]
    cells = []
    for k in METS:
        xs = [v[s][k] for s in v if v[s].get(k) is not None]
        cells.append(f"{st.mean(xs):>10.4f}" if xs else f"{'-':>10}")
    print(f"{t:<16}" + "".join(cells) + f"{len(v):>8}")

# reference rows: they exist for sub-08 ONLY (that is where those image sets live),
# so they are printed with n=1 and must not be averaged against the 10-subject rows
for tag, path in (("REF sdedit_ll", "${OUT}/eval/s00_sdedit_ll.json"),
                  ("REF g3f_selfgate", "${OUT}/eval/s00_g3f_ll_selfgate.json")):
    if os.path.isfile(path):
        m = json.load(open(path))
        m = m.get("metrics", m)
        print(f"{tag:<16}" + "".join(
            f"{m[k]:>10.4f}" if m.get(k) is not None else f"{'-':>10}" for k in METS)
            + f"{1:>8}")

# THE PAIRED CLAIM: full vs ablation, same subjects, same generation code
print()
print("PAIRED full - ablation (positive = the TDM mechanisms help):")
print(f"  {'metric':<10}{'mean delta':>12}{'sem':>10}{'subjects_won':>14}{'n':>5}")
full, abl = data["tdm_ll_self"], data["tdm_abl_self"]
paired = sorted(set(full) & set(abl))
for k in METS:
    d = [full[s][k] - abl[s][k] for s in paired
         if full[s][k] is not None and abl[s][k] is not None]
    if not d:
        continue
    sem = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
    won = sum(1 for x in d if x > 0)
    print(f"  {k:<10}{st.mean(d):>+12.4f}{sem:>10.4f}{won:>8}/{len(d):<5}{len(d):>5}")

# mechanism verdicts, side by side so a claim and its control are adjacent
print()
print("MECHANISM READ-OUT (per subject):")
for sd in sorted({int(os.path.basename(p)[4:6]) for p in
                  glob.glob("${OUT}/tdm/sub-??/tdm_report.json")}):
    p = "${OUT}/tdm/sub-%02d/tdm_report.json" % sd
    if not os.path.isfile(p):
        continue
    d = json.load(open(p)); mech = d.get("mechanisms", {})
    g = mech.get("granularity_time_claim", {})
    mt = g.get("mean_time_ms", {})
    print(f"  sub-{sd:02d} trained={d.get('trained')} grad_skips={d.get('grad_skips')} "
          f"2way={d['ip_fused_disc']['twoway']:.4f} hub_skew={d['hubness']['hub_skew']:.2f} "
          f"| gate order={'<'.join(g.get('mean_time_order', []))} "
          f"pass={g.get('pass')} detail>overall={g.get('pass_detail_after_overall')}")
    if "dla" in mech:
        dl = mech["dla"]
        print(f"          DLA mean|tau| ms=" +
              ",".join(f"{v:.2f}" for v in dl["per_band_mean_abs_ms"]) +
              f" nondeg={dl['pass_non_degenerate']} LF>HF={dl['pass_lf_gt_hf']} | "
              f"RSD offdiag=" + ",".join(f"{v:.3f}" for v in mech["rsd"]["per_band_offdiag_norm"]))

# Gate 0 across subjects: does the DLA premise hold beyond sub-08?
print()
print("GATE 0 (per subject): does each mechanism still have a premise?")
for p in sorted(glob.glob("${NB_ROOT}/outputs/tdm/gate0_sub??.json")):
    d = json.load(open(p))
    j = d.get("G1_jitter", {}); l = d.get("G2_linearity", {})
    if "sd_ms" in j:
        print(f"  sub-{d['subject']:02d} jitter sd={j['sd_ms']:.2f}ms pass={j['pass']} | "
              f"RSD lin/mlp={l.get('linear_over_mlp', float('nan')):.3f} pass={l.get('pass_rsd')} | "
              f"DNG alpha_global={l.get('alpha_more_global')} pass={l.get('pass_dng')}")
PY
echo "===== done @ $(date -Iseconds) ====="
