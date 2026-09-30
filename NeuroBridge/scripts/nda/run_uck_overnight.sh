#!/usr/bin/env bash
# ============================================================================
# UCK overnight -- unified concept kernel, a few mechanisms, tight disk.
#
# One identity (intra shared_r).  One query against G_text and G_img.
# IP is train-gallery retrieval, never encode_image() regression.
# One shared 64x64 field for depth + VAE.  One ControlNet-Img2Img decode.
#
# Mechanisms (not extra towers):
#   full     dual-view q + shared F
#   text     q trained on G_text only (image view off)
#   mem      concept-gallery retrieval   vs  image-bank retrieval  vs  raw q
#   decode   HCMA-S (cn=0.40, s=0.82)    vs  ATM SDEdit
#   prompt   deploy generic / five train-gallery words / empty
#   noise    same weights, z silenced
#
# Disk: PNG generations are deleted as soon as the official-seven JSON lands.
# No last.pth, no train-side predictions, no raw EEG cache.  Peak is one
# 200-image folder (~130MB) plus the shared train-depth npy (~270MB).
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
# activate.sh exports ROOT=eeg-brainit. Source first, then pin our output
# under a name that script cannot overwrite.
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true
UCK_OUT="${UCK_OUT:-${NB_ROOT}/outputs/uck}"
Z_ROOT="${Z_ROOT:-${NB_ROOT}/outputs/ocf/intra_z}"
COND="${NB_ROOT}/outputs/gem/cond_cache"
SHARED="${UCK_OUT}/shared"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="$(command -v python)"
DEVICE="${DEVICE:-cuda:0}"
SUBJECTS="${SUBJECTS:-8,1,2,3,4,5,6,7,9,10}"
EPOCHS="${EPOCHS:-30}"
KEEP_IMAGES="${KEEP_IMAGES:-0}"

HS_CN="${HS_CN:-0.40}"
HS_STRENGTH="${HS_STRENGTH:-0.82}"
SD_STRENGTH="${SD_STRENGTH:-0.82}"
GEN_STEPS="${GEN_STEPS:-28}"
GEN_GUIDANCE="${GEN_GUIDANCE:-5.0}"
IP_SCALE="${IP_SCALE:-1.0}"

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${UCK_OUT}/logs" "${SHARED}/gt_depth" "${NB_ROOT}/outputs/slurm"
export UCK_OUT
echo "[env] PYTHON=${PYTHON} out=${UCK_OUT} subjects=${SUBJECTS}"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
warn() { echo "[WARN] $*" >&2; }

require scripts/nda/uck_train.py
require scripts/nda/uck_measure.py
require scripts/nda/gem_calib.py
require scripts/nda/generate_hcma_s_decode.py
require scripts/nda/generate_atm_aligned_decode.py
require scripts/nda/eval_official_seven_dir.py
require "${COND}/clip_img1024_train.npy"
require "${COND}/clip_img1024_test.npy"
require "${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
require "${NB_ROOT}/outputs/leakfree/split.json"

PDEP="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
DEPTH_TEST="${NB_ROOT}/outputs/hcma_s_full10/shared/gt_depth/test_depth_64.npy"
[[ -f "${DEPTH_TEST}" ]] || DEPTH_TEST="${NB_ROOT}/outputs/hcma_s/sub-08/gt_depth/test_depth_64.npy"
DEPTH_TRAIN="${SHARED}/gt_depth/train_depth_64.npy"

# ------------------------------------------------------------------ [0] shared train depth (no RGB)
echo "===== [0] shared train depth (npy only) @ $(date -Iseconds) ====="
if [[ ! -f "${DEPTH_TRAIN}" ]]; then
  if ! "${PYTHON}" scripts/nda/uck_build_depth.py \
      --images-root "${IMAGES_ROOT}" \
      --output-dir "${SHARED}/gt_depth" \
      --device "${DEVICE}" >> "${UCK_OUT}/logs/depth_train.log" 2>&1; then
    warn "train depth build failed; spatial heads will fall back to existing maps"
  fi
fi
SPATIAL=1
if [[ ! -f "${DEPTH_TRAIN}" || ! -f "${DEPTH_TEST}" ]]; then
  SPATIAL=0
  warn "no GT depth pair; --spatial 0 and existing HS/LL maps will be used"
fi
echo "[disk] after depth: $(du -sh "${UCK_OUT}" | cut -f1)"

# ------------------------------------------------------------------ helpers
calib() {
  local src="$1" dst="$2" tag="$3"
  if [[ -f "${dst}" ]]; then
    printf '%s\n' "${dst}"
    return 0
  fi
  if ! "${PYTHON}" scripts/nda/gem_calib.py \
      --in "${src}" --out "${dst}" \
      --ref "${COND}/clip_img1024_train.npy" --tag "${tag}" \
      >> "${UCK_OUT}/logs/calib.log" 2>&1; then
    warn "calib ${tag} failed; using raw"
    cp -f "${src}" "${dst}"
  fi
  printf '%s\n' "${dst}"
}

eval_row() {
  local tag="$1" gdir="$2" ev="$3"
  [[ -f "${ev}" ]] && { echo "[SKIP] eval ${tag}"; return 0; }
  [[ -f "${gdir}/199.png" ]] || { warn "eval ${tag}: no images"; return 0; }
  echo "--- eval ${tag}"
  if ! "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
      --gen-dir "${gdir}" --output-json "${ev}" --tag "${tag}" \
      --images-root "${IMAGES_ROOT}" --device "${DEVICE}" --skip-if-exists \
      >> "${UCK_OUT}/logs/eval.log" 2>&1; then
    warn "eval ${tag} failed"
    return 0
  fi
  if [[ -s "${ev}" && "${KEEP_IMAGES}" != "1" ]]; then
    rm -rf "$(dirname "${gdir}")"
  fi
}

pick_depth() {
  local sub="$1" arm="$2"
  local own="${UCK_OUT}/${sub}/${arm}/spatial/pred_depth_rgb_512/199.png"
  if [[ -f "${own}" ]]; then
    printf '%s\n' "${UCK_OUT}/${sub}/${arm}/spatial/pred_depth_rgb_512"
    return 0
  fi
  printf '%s\n' "${NB_ROOT}/outputs/hcma_s_full10/${sub}/depth/pred_depth_rgb_512"
}

pick_ll() {
  local sub="$1" arm="$2"
  local own="${UCK_OUT}/${sub}/${arm}/spatial/pred_lowlevel_rgb_512/199.png"
  if [[ -f "${own}" ]]; then
    printf '%s\n' "${UCK_OUT}/${sub}/${arm}/spatial/pred_lowlevel_rgb_512"
    return 0
  fi
  printf '%s\n' "${NB_ROOT}/outputs/sdedit_ll_full10/${sub}/vae_head/pred_lowlevel_rgb_512"
}

pick_vae() {
  local sub="$1" arm="$2"
  local own="${UCK_OUT}/${sub}/${arm}/spatial/pred_vae_test_scaled.npy"
  if [[ -f "${own}" ]]; then
    printf '%s\n' "${own}"
    return 0
  fi
  if [[ "${sub}" == "sub-08" && -f "${NB_ROOT}/outputs/uge/sub-08/full/pred_vae_test_scaled.npy" ]]; then
    printf '%s\n' "${NB_ROOT}/outputs/uge/sub-08/full/pred_vae_test_scaled.npy"
    return 0
  fi
  printf '%s\n' ""
}

gen_hs() {
  local sub="$1" tag="$2" cond="$3" pf="$4" arm="$5"
  local sd="${sub#sub-}"
  local ev="${UCK_OUT}/${sub}/eval/s${sd}_${tag}.json"
  local gdir="${UCK_OUT}/${sub}/gen/${tag}"
  [[ -f "${ev}" ]] && { echo "[SKIP] ${tag}"; return 0; }
  require "${cond}"; require "${pf}"
  local depth ll
  depth="$(pick_depth "${sub}" "${arm}")"
  ll="$(pick_ll "${sub}" "${arm}")"
  [[ -f "${depth}/199.png" ]] || { warn "no depth for ${tag}"; return 0; }
  [[ -f "${ll}/199.png" ]] || { warn "no ll for ${tag}"; return 0; }
  echo "===== HS ${sub} ${tag} @ $(date -Iseconds) ====="
  if ! "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
      --embed-npy "${cond}" --prompts-json "${pf}" \
      --depth-rgb-dir "${depth}" --lowlevel-rgb-dir "${ll}" \
      --output-dir "${gdir}" --tag "${tag}" \
      --cn-scale "${HS_CN}" --strength "${HS_STRENGTH}" --ip-scale "${IP_SCALE}" \
      --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
      >> "${UCK_OUT}/logs/gen_${sub}_${tag}.log" 2>&1; then
    warn "gen ${tag} failed"
    rm -rf "${gdir}"
    return 0
  fi
  eval_row "${tag}" "${gdir}/generated" "${ev}"
}

gen_atm() {
  local sub="$1" tag="$2" cond="$3" pf="$4" arm="$5"
  local sd="${sub#sub-}"
  local ev="${UCK_OUT}/${sub}/eval/s${sd}_${tag}.json"
  local gdir="${UCK_OUT}/${sub}/gen/${tag}"
  [[ -f "${ev}" ]] && { echo "[SKIP] ${tag}"; return 0; }
  require "${cond}"
  local vae pflag
  vae="$(pick_vae "${sub}" "${arm}")"
  pflag=(--prompts-json "")
  if [[ "${pf}" != "none" && -n "${pf}" ]]; then
    require "${pf}"
    pflag=(--prompts-json "${pf}")
  fi
  echo "===== ATM ${sub} ${tag} @ $(date -Iseconds) ====="
  local extra=()
  if [[ -n "${vae}" ]]; then
    extra+=(--vae-latent-npy "${vae}")
  else
    extra+=(--lowlevel-rgb-dir "$(pick_ll "${sub}" "${arm}")")
  fi
  if ! "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
      --mode sdedit --embed-npy "${cond}" "${pflag[@]}" \
      "${extra[@]}" \
      --output-dir "${gdir}" --tag "${tag}" \
      --strength "${SD_STRENGTH}" --ip-scale "${IP_SCALE}" \
      --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
      --device "${DEVICE}" >> "${UCK_OUT}/logs/gen_${sub}_${tag}.log" 2>&1; then
    warn "gen ${tag} failed"
    rm -rf "${gdir}"
    return 0
  fi
  eval_row "${tag}" "${gdir}/generated" "${ev}"
}

train_arm() {
  local sid="$1" arm="$2" views="$3" spatial="$4" mem="$5"
  local sub
  sub="$(printf 'sub-%02d' "${sid}")"
  local o="${UCK_OUT}/${sub}/${arm}"
  mkdir -p "${o}" "${UCK_OUT}/${sub}/eval" "${UCK_OUT}/${sub}/logs"
  if [[ -f "${o}/conds/ip_mem_test.npy" && -f "${o}/report.json" ]]; then
    echo "[SKIP] train ${sub}/${arm}"
    return 0
  fi
  echo "===== train ${sub}/${arm} views=${views} spatial=${spatial} @ $(date -Iseconds) ====="
  local extra=()
  if [[ "${spatial}" == "1" ]]; then
    extra+=(--depth-train "${DEPTH_TRAIN}" --depth-test "${DEPTH_TEST}")
  fi
  local tlog="${UCK_OUT}/${sub}/logs/train_${arm}.log"
  if ! "${PYTHON}" scripts/nda/uck_train.py \
      --out "${o}" --test-subject "${sid}" \
      --z-root "${Z_ROOT}" --gallery-cache "${SHARED}" \
      --views "${views}" --mem-bank "${mem}" --spatial "${spatial}" \
      --epochs "${EPOCHS}" --device "${DEVICE}" \
      "${extra[@]}" >> "${tlog}" 2>&1; then
    echo "[FATAL] train ${sub}/${arm} failed; last 40 lines:" >&2
    tail -n 40 "${tlog}" >&2 || true
    exit 1
  fi
  rm -f "${o}/last.pth"
}

sweep_subject_disk() {
  local sub="$1"
  rm -rf "${UCK_OUT}/${sub}/gen"
  find "${UCK_OUT}/${sub}" -name 'last.pth' -delete || true
  find "${UCK_OUT}/${sub}" -type d -name '_twoway_cache' -print0 2>/dev/null \
    | xargs -0 -r rm -rf || true
  echo "[disk] ${sub}: $(du -sh "${UCK_OUT}/${sub}" | cut -f1)   total $(du -sh "${UCK_OUT}" | cut -f1)"
}

# ------------------------------------------------------------------ per subject
IFS=',' read -ra SUBJ_ARR <<< "${SUBJECTS}"
for SID in "${SUBJ_ARR[@]}"; do
  SID="$(echo "${SID}" | tr -d ' ')"
  SUB="$(printf 'sub-%02d' "${SID}")"
  echo "########## ${SUB} @ $(date -Iseconds) ##########"
  require "${Z_ROOT}/${SUB}/shared_r_train.npy"
  require "${Z_ROOT}/${SUB}/shared_r_test.npy"
  mkdir -p "${UCK_OUT}/${SUB}/eval" "${UCK_OUT}/${SUB}/conds"

  train_arm "${SID}" full both "${SPATIAL}" both
  if [[ "${SID}" == "8" ]]; then
    train_arm "${SID}" text text 0 concept
  fi

  FULL="${UCK_OUT}/${SUB}/full"
  if [[ ! -f "${FULL}/conds/ip_mem_test.npy" ]]; then
    echo "[FATAL] ${SUB} has no full IP after train" >&2
    exit 1
  fi

  MEM="$(calib "${FULL}/conds/ip_mem_test.npy" "${UCK_OUT}/${SUB}/conds/mem.npy" "${SUB}_mem")"
  Q="$(calib "${FULL}/conds/ip_q_test.npy" "${UCK_OUT}/${SUB}/conds/q.npy" "${SUB}_q")"
  FIVE="${FULL}/prompts/prompts_five.json"
  PEMPTY="${FULL}/prompts/prompts_empty.json"
  [[ -f "${PEMPTY}" ]] || PEMPTY="${UCK_OUT}/${SUB}/prompts_empty.json"
  if [[ ! -f "${PEMPTY}" ]]; then
    mkdir -p "$(dirname "${PEMPTY}")"
    "${PYTHON}" -c "import json; from pathlib import Path; Path('${PEMPTY}').write_text(json.dumps(['']*200, indent=1))"
  fi

  # headline + decoder / prompt / mechanism ablations
  gen_hs "${SUB}" hs_mem_deploy "${MEM}" "${PDEP}" full
  gen_atm "${SUB}" atm_mem_deploy "${MEM}" "${PDEP}" full

  if [[ "${SID}" == "8" ]]; then
    gen_hs "${SUB}" hs_mem_five "${MEM}" "${FIVE}" full
    gen_hs "${SUB}" hs_mem_none "${MEM}" "${PEMPTY}" full
    gen_hs "${SUB}" hs_q_deploy "${Q}" "${PDEP}" full
    if [[ -f "${FULL}/conds/ip_mem_image_test.npy" ]]; then
      IMGB="$(calib "${FULL}/conds/ip_mem_image_test.npy" "${UCK_OUT}/${SUB}/conds/imgbank.npy" "${SUB}_imgbank")"
      gen_hs "${SUB}" hs_imgbank_deploy "${IMGB}" "${PDEP}" full
    fi
    if [[ -f "${FULL}/conds/ip_mem_noise_test.npy" ]]; then
      NZ="$(calib "${FULL}/conds/ip_mem_noise_test.npy" "${UCK_OUT}/${SUB}/conds/noise.npy" "${SUB}_noise")"
      gen_hs "${SUB}" hs_noise_deploy "${NZ}" "${PDEP}" full
    fi
    if [[ -f "${UCK_OUT}/${SUB}/text/conds/ip_mem_test.npy" ]]; then
      TXT="$(calib "${UCK_OUT}/${SUB}/text/conds/ip_mem_test.npy" "${UCK_OUT}/${SUB}/conds/text.npy" "${SUB}_text")"
      gen_hs "${SUB}" hs_text_deploy "${TXT}" "${PDEP}" full
    fi
  fi

  if ! "${PYTHON}" scripts/nda/uck_measure.py \
      --out-dir "${UCK_OUT}/${SUB}" \
      --bank "${COND}/clip_img1024_test.npy" \
      --report "${UCK_OUT}/${SUB}/measure.json" \
      >> "${UCK_OUT}/logs/measure.log" 2>&1; then
    warn "measure ${SUB} failed"
  fi
  sweep_subject_disk "${SUB}"
done

echo "===== summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
ROOT = Path(os.environ.get("UCK_OUT", "outputs/uck"))
bars = [
    ("CogCap sub-08", 0.175, 0.744),
    ("ATM sub-08",    0.160, 0.786),
    ("CogCap 10subj", 0.150, 0.715),
    ("g3f_ll_selfgate", 0.183, 0.779),
]
print(f"{'row':<28} {'Pix':>7} {'CLIP':>6}")
print("-" * 44)
for n, pix, cl in bars:
    print(f"{n:<28} {pix:7.3f} {cl:6.3f}")
print("-- this run --")
for p in sorted(ROOT.glob("sub-*/eval/s*.json")):
    d = json.loads(p.read_text())
    print(f"{p.parent.parent.name+'/'+p.stem:<28} "
          f"{d.get('pixcorr', float('nan')):7.3f} {d.get('clip', float('nan')):6.3f}")
meas = list(ROOT.glob("sub-*/measure.json"))
for m in meas:
    d = json.loads(m.read_text())
    print(f"{m.parent.name} measure: verdict={d.get('arm_agreement_verdict')} "
          f"full_vs_noise={d.get('full_vs_noise_rowcos')}")
PY

echo "===== done @ $(date -Iseconds) ====="
du -sh "${UCK_OUT}" | sed 's/^/[disk] /'
du -sh "${UCK_OUT}"/sub-* 2>/dev/null | sed 's/^/[disk] /' || true
