#!/usr/bin/env bash
# ============================================================================
# OCF-INTRA / sub-08 -- pure intra-subject chain, one job, leak-free.
#
# WHAT "PURE INTRA" MEANS HERE, PRECISELY
#   Every project-specific weight is trained on sub-08 alone:
#     * the EEG encoder (SharedSpecificEncoder) -- mode A below trains it with
#       `--train-subjects 8 --calib-subject 8`, and ocf_export_intra_z.py
#       HARD-FAILS if the checkpoint lists any other subject, so an accidental
#       LOSO checkpoint cannot be mislabelled as intra;
#     * the OCF read-out heads;
#     * the VAE low-level head.
#   Only frozen public image models are shared (SDXL, IP-Adapter, CLIP, RN50),
#   which carry no THINGS-EEG subject information.
#
# THE THREE CHANGES THIS RUN TESTS (each one is a fix to a MEASURED defect)
# -----------------------------------------------------------------------
# 1. SPHERICAL mean/residual read-out with a SOLVED angle  (ocf_train.py)
#    defect: `l2(mu + res)` with an unbounded residual is degenerate under
#    scale-invariant losses; measured res_norm 8.11-9.68, i.e. the train mean
#    contributed ~1.2% of the condition, which is why the "centred residual"
#    design in g3f produced a Delta of exactly -0.009.
#    fix:    `l2(cos(theta) * l2(mu) + sin(theta) * l2(res))`, theta solved by
#    bisection on TRAIN rows so the achieved self-concentration matches the
#    reference TRAIN IP bank's own. The closed form arccos(c_self_ref) is only a
#    first guess because the measurement shows the residual is NOT orthogonal to
#    the mean (at theta = pi/2 the concentration is already 0.53, not 0).
#    This replaces the post-hoc `g3f_calibrate_cond.py` quantile hack with a
#    geometric knob inside the model.
#
# 2. THETA ABLATION FROM ONE CHECKPOINT  (ocf_train.py --export-angles)
#    theta interpolates towards a FIXED buffer (the train mean), so no retraining
#    is needed to move along the arc; three angles are exported and three rows
#    generated. c_self_ratio and top1/2-way are both reported, because "closer to
#    the manifold" and "more discriminative" can disagree and the run should show
#    which one happens rather than assume.
#
# 3. FRLA -- frequency-resolved latent anchoring  (idg_frla_decode.py)
#    defect: SDEdit carries ONE scalar `strength` over a spectrum whose measured
#    EEG reliability spans 9.9x (band_probe_ceiling.json: 0.1332 at r<0.0625 down
#    to 0.0134 at 0.5<r<2.0). One scalar cannot serve both ends, which is exactly
#    the Pareto collision seen in the historical grid, and 0.86/0.88 were never
#    even distinct under diffusers' integer truncation.
#    fix:    decompose the guidance score by band, w_b proportional to r_b^2, and
#    re-impose the anchor at EVERY active step (SDEdit anchors once and then lets
#    it decay). The claim is falsifiable and is tested here: FRLA at a HIGH
#    strength (0.95, i.e. more denoising -> better texture/FID) should beat plain
#    SDEdit at 0.95 on PixCorr/SSIM. The `uniform` arm applies the same TOTAL
#    force with flat weights, so if it matches `frla` the per-band resolution is
#    decoration and the effect is just extra anchoring.
#
# LEAK-FREE BY CONSTRUCTION
#   * the prompt gallery is TRAIN concepts only and DISJOINT from the 200 test
#     concepts (asserted below and hard-failed, same check as the g3f run);
#   * theta is solved on TRAIN rows against a TRAIN target statistic;
#   * band weights come from the train-side probe in band_probe_ceiling.json;
#   * strength/arm/eta are FIXED A PRIORI on this command line, never selected on
#     test metrics (that is the bias that was removed from the project);
#   * `sem_concept_tmpl_test.npy` is never read for training or selection.
#
# ROWS (all sub-08, n=200)
#   ref rows, re-evaluated here by the SAME code so the comparison is identical:
#     sdedit_ll      existing strong low-level row       (PixCorr ~0.165)
#     g3f_ll_selfgate  existing g3f row                  (angular-share contrast)
#   ours:
#     ocf_ll_gen     ip_fused + GENERIC prompt + SDEdit 0.82
#     ocf_ll_self    ip_fused + SELF prompt    + SDEdit 0.82   (headline)
#     ocf_theta_t90 / t70 / t52   same, ip_fused exported at three angles
#     frla_off / frla_uniform / frla_frla   strength 0.95 arms
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
STAG="sub-08"
OUT="${OUT:-${NB_ROOT}/outputs/ocf_intra}"
ENC="${NB_ROOT}/outputs/ocf/intra_enc"
ZC="${NB_ROOT}/outputs/ocf/intra_z"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-26}"
# A PRIORI FIXED -- never selected on test
SD_STRENGTH="0.82"          # the established sdedit_ll / g3f setting
FRLA_STRENGTH="0.95"        # deliberately HIGHER: the FRLA claim needs the
                            # strength where plain SDEdit loses the layout
FRLA_ETA="0.85"
GEN_STEPS="28"
GEN_GUIDANCE="5.0"
IP_SCALE="1.0"
# `w_ip` weights a PLAIN COSINE REGRESSION of the generated condition onto the IP
# target.  Its minimiser is the CONDITIONAL MEAN direction -- the same defect
# recorded for g3f's `h_ip`, whose cos-to-target (0.613-0.634) merely matched a
# constant (0.6147).  Leaving it at 1.0 makes it compete with InfoNCE for the
# same head and win as training proceeds: measured on sub-08 the self-
# concentration of the fused condition went 0.744 -> 1.000 between 2 and 6
# epochs, i.e. the condition collapsed towards a constant as it fit better.
# 0.2 keeps a weak fidelity pull while letting InfoNCE + class supervision +
# var_band own the head's dispersion; theta* then places the result back on the
# reference manifold.  Both this and theta are reported, so neither is assumed.
W_IP="0.2"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"

# the project venv -- the same one every successful run in this project used.
# A system-python fallback is a silent way to get a different torch build, so the
# interpreter is taken from the env script and then verified.
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true
PYTHON="$(command -v python)"
if [[ -x "${NB_ROOT}/.venv_ocf/bin/python" && "${USE_OCF_VENV:-0}" == "1" ]]; then
  PYTHON="${NB_ROOT}/.venv_ocf/bin/python"
fi
echo "[env] PYTHON=${PYTHON}"
"${PYTHON}" - <<'PY'
import sys
try:
    import torch, diffusers
except Exception as e:                                   # noqa: BLE001
    sys.exit(f"[FATAL] torch/diffusers import failed: {e}")
if not torch.cuda.is_available():
    sys.exit("[FATAL] CUDA unavailable. A CPU fallback would train on the wrong "
             "device at a different speed and silently produce a mixed-device run.")
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
require "${VC}/test_vae_latents_f16.npy"

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
    sys.exit(f"[FATAL] prompt gallery shares {len(inter)} strings with the test "
             f"concepts, e.g. {sorted(inter)[:5]}. That is the oracle leak this "
             f"run exists to avoid; refusing to generate.")
print("[audit] ok: no emitted prompt can name a test class")
PY

# -------------------------------------------- [1] pure intra encoder (sub-08 only)
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

# ------------------------------------------------- [2] export intra shared_r (z)
echo "===== [2] export intra shared_r @ $(date -Iseconds) ====="
if [[ ! -f "${ZC}/sub-08/shared_r_test.npy" ]]; then
  "${PYTHON}" scripts/nda/ocf_export_intra_z.py \
    --subject 8 --checkpoint "${ENC_CK}" --out "${ZC}" --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/export_intra_z.log"
else
  echo "[SKIP] z export exists"
fi
require "${ZC}/sub-08/shared_r_test.npy"

# --------------------------------------------------------- [3] OCF heads, intra
echo "===== [3] OCF heads (intra) @ $(date -Iseconds) ====="
tr=(); for s in 8; do tr+=(--train-subjects "${s}"); done
"${PYTHON}" scripts/nda/ocf_train.py \
  "${tr[@]}" --test-subject 8 \
  --z-source shared_r --z-cache-root "${ZC}" \
  --targets-dir "${NB_ROOT}/outputs/g2/targets" \
  --clip-text-dir "${CLIP_TEXT}" --captions-jsonl "${CAPTIONS}" \
  --out "${OUT}/ocf" --epochs "${EPOCHS}" --device "${DEVICE}" \
  --w-ip "${W_IP}" \
  2>&1 | tee "${OUT}/logs/ocf_train_intra.log"
require "${OUT}/ocf/conds/ip_fused_test.npy"

# ------------------------------------------------- [4] VAE low-level head, intra
echo "===== [4] VAE low-level head (intra) @ $(date -Iseconds) ====="
VH="${OUT}/ocf/vae_head"
if [[ ! -f "${VH}/pred_lowlevel_rgb_512/199.png" ]]; then
  mkdir -p "${VH}"
  # `--val-split-json` IS NOT OPTIONAL.  Without it this script falls back to
  # picking the checkpoint with the best TEST-set MAE and labels the run
  # "test(contaminated)" -- the exact selection bias the leakfree audit removed
  # elsewhere.  With it, selection happens on 820 held-in TRAIN-concept rows and
  # the test latents are exported but never used to choose anything.
  "${PYTHON}" scripts/nda/train_eeg_vae_head.py \
    --eeg-train-npy "${ZC}/sub-08/shared_r_train.npy" \
    --eeg-test-npy  "${ZC}/sub-08/shared_r_test.npy" \
    --vae-train-npy "${VC}/train_vae_latents_f16.npy" \
    --vae-test-npy  "${VC}/test_vae_latents_f16.npy" \
    --val-split-json "${SPLIT}" \
    --output-dir "${VH}" --decode-rgb --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/vae_head_intra.log"
  # hard-fail if the run ever regresses to test-based selection
  # unquoted heredoc on purpose: ${VH} must expand here
  "${PYTHON}" - <<PYCHK
import json, sys
p = "${VH}/vae_head_report.json"
d = json.load(open(p))
sel = d["final"]["selected_on"]
print(f"[check] vae head selected_on = {sel}")
if sel != "val":
    sys.exit(f"[FATAL] VAE head selected on '{sel}'. That is test-set selection "
             f"bias; the run is contaminated and must not be reported.")
PYCHK
else
  echo "[SKIP] vae head exists"
fi
require "${VH}/pred_lowlevel_rgb_512/199.png"
require "${SPLIT}"
require "${VH}/pred_vae_test.npy"

# ------------------------------------------------------------------ [5] generate
gen_ll() {                       # tag cond prompt_file
  local tag="$1" cond="$2" pf="$3"
  local gdir="${OUT}/gen/${STAG}/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] gen ${tag}"; return 0; fi
  [[ -f "${cond}" ]] || { echo "[WARN] ${tag}: missing cond ${cond}"; return 1; }
  [[ -f "${pf}" ]] || { echo "[WARN] ${tag}: missing prompts ${pf}"; return 1; }
  echo "===== gen ${tag} @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
    --mode sdedit --embed-npy "${cond}" --prompts-json "${pf}" \
    --lowlevel-rgb-dir "${VH}/pred_lowlevel_rgb_512" \
    --output-dir "${gdir}" --tag "${tag}" \
    --strength "${SD_STRENGTH}" --ip-scale "${IP_SCALE}" \
    --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
    --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/gen_${tag}.log"
}

gen_frla() {                     # tag arm
  local tag="$1" arm="$2"
  local gdir="${OUT}/gen/${STAG}/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] gen ${tag}"; return 0; fi
  echo "===== gen ${tag} (arm=${arm}) @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/idg_frla_decode.py \
    --embed-npy "${OUT}/ocf/conds/ip_fused_test.npy" \
    --anchor-latent-npy "${VH}/pred_vae_test.npy" \
    --lowlevel-rgb-dir "${VH}/pred_lowlevel_rgb_512" \
    --prompts-json "${OUT}/ocf/prompts/prompts_self.json" \
    --output-dir "${gdir}" --tag "${tag}" --arm "${arm}" --eta "${FRLA_ETA}" \
    --strength "${FRLA_STRENGTH}" --gen-steps "${GEN_STEPS}" \
    --gen-guidance "${GEN_GUIDANCE}" --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/gen_${tag}.log"
}

echo "===== [5] generation @ $(date -Iseconds) ====="
gen_ll ocf_ll_gen   "${OUT}/ocf/conds/ip_fused_test.npy"       "${OUT}/ocf/prompts/prompts_generic.json" || true
gen_ll ocf_ll_self  "${OUT}/ocf/conds/ip_fused_test.npy"       "${OUT}/ocf/prompts/prompts_self.json" || true
# theta rows: `train` is the operating point the weights were optimised at, `star`
# is the angle solved on held-out train rows to match the reference manifold, and
# `far` brackets the optimum so the rows sample the arc instead of one point
gen_ll ocf_th_train "${OUT}/ocf/conds/ip_fused_train_test.npy" "${OUT}/ocf/prompts/prompts_self.json" || true
gen_ll ocf_th_star  "${OUT}/ocf/conds/ip_fused_star_test.npy"  "${OUT}/ocf/prompts/prompts_self.json" || true
gen_ll ocf_th_far   "${OUT}/ocf/conds/ip_fused_far_test.npy"   "${OUT}/ocf/prompts/prompts_self.json" || true
# `frla_off` is the arm FRLA has to beat: plain SDEdit at the SAME strength 0.95,
# so the only difference between frla_off / uniform / frla is the anchoring law
gen_frla frla_off     off      || true
gen_frla frla_uniform uniform  || true
gen_frla frla_frla    frla     || true

# -------------------------------------------------------------------- [6] evaluate
echo "===== [6] evaluation @ $(date -Iseconds) ====="
eval_row() {                     # tag gen_dir
  local tag="$1" gdir="$2"
  [[ -f "${gdir}/199.png" ]] || { echo "[WARN] eval ${tag}: no images"; return 0; }
  echo "--- eval ${tag}"
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${gdir}" --output-json "${OUT}/eval/${tag}.json" --tag "${tag}" \
    --images-root "${IMAGES_ROOT}" --device "${DEVICE}" --skip-if-exists \
    2>&1 | tee -a "${OUT}/logs/eval.log" || echo "[WARN] eval failed: ${tag}"
}
mkdir -p "${OUT}/eval"
for t in ocf_ll_gen ocf_ll_self ocf_th_train ocf_th_star ocf_th_far frla_off frla_uniform frla_frla; do
  eval_row "${t}" "${OUT}/gen/${STAG}/${t}/generated"
done
# reference rows, evaluated by the SAME code and the SAME GT cache
eval_row sdedit_ll "${NB_ROOT}/outputs/sdedit_ll_full10/sub-08/generation/sdedit_ll/generated"
eval_row g3f_ll_selfgate "${NB_ROOT}/outputs/g3f/gen/sub-08/g3f_ll_selfgate/generated"

echo "===== [7] summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json, glob, os
rows = []
for p in sorted(glob.glob("${OUT}/eval/*.json")):
    d = json.load(open(p))
    m = d.get("metrics", d)
    rows.append((os.path.basename(p)[:-5],
                 m.get("pixcorr"), m.get("ssim"), m.get("alexnet2"),
                 m.get("alexnet5"), m.get("inception"), m.get("clip"), m.get("swav")))
hdr = ("row", "PixCorr", "SSIM", "Alex2", "Alex5", "Inc", "CLIP", "SwAV")
print("".join(f"{h:>16}" for h in hdr))
for r in rows:
    print("".join(f"{v:>16.4f}" if isinstance(v, float) else f"{str(v):>16}" for v in r))
PY
echo "===== done @ $(date -Iseconds) ====="
