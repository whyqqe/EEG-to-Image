#!/usr/bin/env bash
# ============================================================================
# NAT overnight -- Neural Address then Translate, same z / two views.
#
#   z → neural book μ → translate G_img = IP
#   z → F = depth
#   residual and noise are diagnostics, not extra towers.
#
# Disk: PNG generations deleted as soon as the official-seven JSON lands.
# Writes to NAT_OUT (default outputs/nat). Never touches outputs/uck.
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true
NAT_OUT="${NAT_OUT:-${NB_ROOT}/outputs/nat}"
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
GEN_STEPS="${GEN_STEPS:-28}"
GEN_GUIDANCE="${GEN_GUIDANCE:-5.0}"
IP_SCALE="${IP_SCALE:-1.0}"

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${NAT_OUT}/logs" "${NB_ROOT}/outputs/slurm"
export NAT_OUT
echo "[env] PYTHON=${PYTHON} out=${NAT_OUT} subjects=${SUBJECTS}"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
warn() { echo "[WARN] $*" >&2; }

require scripts/nda/nat_train.py
require scripts/nda/nat_measure.py
require scripts/nda/gem_calib.py
require scripts/nda/generate_hcma_s_decode.py
require scripts/nda/eval_official_seven_dir.py
require "${COND}/clip_img1024_train.npy"
require "${COND}/clip_img1024_test.npy"
require "${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
require "${NB_ROOT}/outputs/leakfree/split.json"
require "${SHARED}/g_img_concept.npy"
require "${SHARED}/gt_depth/train_depth_64.npy"

PDEP="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
DEPTH_TEST="${NB_ROOT}/outputs/hcma_s_full10/shared/gt_depth/test_depth_64.npy"
[[ -f "${DEPTH_TEST}" ]] || DEPTH_TEST="${NB_ROOT}/outputs/hcma_s/sub-08/gt_depth/test_depth_64.npy"
DEPTH_TRAIN="${SHARED}/gt_depth/train_depth_64.npy"
require "${DEPTH_TEST}"

SPATIAL=1
echo "[disk] start: $(du -sh "${NAT_OUT}" 2>/dev/null | cut -f1 || echo 0)"

calib() {
  local src="$1" dst="$2" tag="$3"
  if [[ -f "${dst}" ]]; then
    printf '%s\n' "${dst}"
    return 0
  fi
  if ! "${PYTHON}" scripts/nda/gem_calib.py \
      --in "${src}" --out "${dst}" \
      --ref "${COND}/clip_img1024_train.npy" --tag "${tag}" \
      >> "${NAT_OUT}/logs/calib.log" 2>&1; then
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
      >> "${NAT_OUT}/logs/eval.log" 2>&1; then
    warn "eval ${tag} failed"
    return 0
  fi
  if [[ -s "${ev}" && "${KEEP_IMAGES}" != "1" ]]; then
    rm -rf "$(dirname "${gdir}")"
  fi
}

pick_depth() {
  local sub="$1" kind="$2"
  local base="${NAT_OUT}/${sub}/full/spatial"
  case "${kind}" in
    res)   if [[ -f "${base}/pred_depth_res_rgb_512/199.png" ]]; then
             printf '%s\n' "${base}/pred_depth_res_rgb_512"; return 0; fi ;;
    noise) if [[ -f "${base}/pred_depth_noise_rgb_512/199.png" ]]; then
             printf '%s\n' "${base}/pred_depth_noise_rgb_512"; return 0; fi ;;
    *)     if [[ -f "${base}/pred_depth_rgb_512/199.png" ]]; then
             printf '%s\n' "${base}/pred_depth_rgb_512"; return 0; fi ;;
  esac
  printf '%s\n' "${NB_ROOT}/outputs/hcma_s_full10/${sub}/depth/pred_depth_rgb_512"
}

pick_ll() {
  local sub="$1"
  printf '%s\n' "${NB_ROOT}/outputs/sdedit_ll_full10/${sub}/vae_head/pred_lowlevel_rgb_512"
}

gen_hs() {
  local sub="$1" tag="$2" cond="$3" pf="$4" dkind="$5"
  local sd="${sub#sub-}"
  local ev="${NAT_OUT}/${sub}/eval/s${sd}_${tag}.json"
  local gdir="${NAT_OUT}/${sub}/gen/${tag}"
  [[ -f "${ev}" ]] && { echo "[SKIP] ${tag}"; return 0; }
  require "${cond}"; require "${pf}"
  local depth ll
  depth="$(pick_depth "${sub}" "${dkind}")"
  ll="$(pick_ll "${sub}")"
  [[ -f "${depth}/199.png" ]] || { warn "no depth for ${tag}"; return 0; }
  [[ -f "${ll}/199.png" ]] || { warn "no ll for ${tag}"; return 0; }
  echo "===== HS ${sub} ${tag} depth=${dkind} @ $(date -Iseconds) ====="
  if ! "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
      --embed-npy "${cond}" --prompts-json "${pf}" \
      --depth-rgb-dir "${depth}" --lowlevel-rgb-dir "${ll}" \
      --output-dir "${gdir}" --tag "${tag}" \
      --cn-scale "${HS_CN}" --strength "${HS_STRENGTH}" --ip-scale "${IP_SCALE}" \
      --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
      >> "${NAT_OUT}/logs/gen_${sub}_${tag}.log" 2>&1; then
    warn "gen ${tag} failed"
    rm -rf "${gdir}"
    return 0
  fi
  eval_row "${tag}" "${gdir}/generated" "${ev}"
}

train_arm() {
  local sid="$1"
  local sub
  sub="$(printf 'sub-%02d' "${sid}")"
  local o="${NAT_OUT}/${sub}/full"
  mkdir -p "${o}" "${NAT_OUT}/${sub}/eval" "${NAT_OUT}/${sub}/logs"
  if [[ -f "${o}/conds/ip_nat_test.npy" && -f "${o}/report.json" ]]; then
    echo "[SKIP] train ${sub}/full"
    return 0
  fi
  echo "===== train ${sub}/full @ $(date -Iseconds) ====="
  local tlog="${NAT_OUT}/${sub}/logs/train_full.log"
  if ! "${PYTHON}" scripts/nda/nat_train.py \
      --out "${o}" --test-subject "${sid}" \
      --z-root "${Z_ROOT}" --gallery-cache "${SHARED}" \
      --depth-train "${DEPTH_TRAIN}" --depth-test "${DEPTH_TEST}" \
      --spatial "${SPATIAL}" --epochs "${EPOCHS}" --device "${DEVICE}" \
      >> "${tlog}" 2>&1; then
    echo "[FATAL] train ${sub}/full failed; last 40 lines:" >&2
    tail -n 40 "${tlog}" >&2 || true
    exit 1
  fi
  rm -f "${o}/last.pth"
}

sweep_subject_disk() {
  local sub="$1"
  rm -rf "${NAT_OUT}/${sub}/gen"
  find "${NAT_OUT}/${sub}" -name 'last.pth' -delete || true
  find "${NAT_OUT}/${sub}" -type d -name '_twoway_cache' -print0 2>/dev/null \
    | xargs -0 -r rm -rf || true
  echo "[disk] ${sub}: $(du -sh "${NAT_OUT}/${sub}" | cut -f1)   total $(du -sh "${NAT_OUT}" | cut -f1)"
}

IFS=',' read -ra SUBJ_ARR <<< "${SUBJECTS}"
for SID in "${SUBJ_ARR[@]}"; do
  SID="$(echo "${SID}" | tr -d ' ')"
  SUB="$(printf 'sub-%02d' "${SID}")"
  echo "########## ${SUB} @ $(date -Iseconds) ##########"
  require "${Z_ROOT}/${SUB}/shared_r_train.npy"
  require "${Z_ROOT}/${SUB}/shared_r_test.npy"
  mkdir -p "${NAT_OUT}/${SUB}/eval" "${NAT_OUT}/${SUB}/conds"

  train_arm "${SID}"

  FULL="${NAT_OUT}/${SUB}/full"
  if [[ ! -f "${FULL}/conds/ip_nat_test.npy" ]]; then
    echo "[FATAL] ${SUB} has no NAT IP after train" >&2
    exit 1
  fi

  NATIP="$(calib "${FULL}/conds/ip_nat_test.npy" "${NAT_OUT}/${SUB}/conds/nat.npy" "${SUB}_nat")"
  PEMPTY="${FULL}/prompts/prompts_empty.json"
  [[ -f "${PEMPTY}" ]] || PEMPTY="${NAT_OUT}/${SUB}/prompts_empty.json"
  if [[ ! -f "${PEMPTY}" ]]; then
    mkdir -p "$(dirname "${PEMPTY}")"
    "${PYTHON}" -c "import json; from pathlib import Path; Path('${PEMPTY}').write_text(json.dumps(['']*200, indent=1))"
  fi

  # headline: neural address + structure from the same z
  gen_hs "${SUB}" hs_nat_deploy "${NATIP}" "${PDEP}" full

  if [[ "${SID}" == "8" ]]; then
    gen_hs "${SUB}" hs_nat_none "${NATIP}" "${PEMPTY}" full
    if [[ -f "${FULL}/conds/ip_res_test.npy" ]]; then
      RESIP="$(calib "${FULL}/conds/ip_res_test.npy" "${NAT_OUT}/${SUB}/conds/res.npy" "${SUB}_res")"
      gen_hs "${SUB}" hs_resip_deploy "${RESIP}" "${PDEP}" full
    fi
    gen_hs "${SUB}" hs_resF_deploy "${NATIP}" "${PDEP}" res
    if [[ -f "${FULL}/conds/ip_noise_test.npy" ]]; then
      NZ="$(calib "${FULL}/conds/ip_noise_test.npy" "${NAT_OUT}/${SUB}/conds/noise.npy" "${SUB}_noise")"
      gen_hs "${SUB}" hs_noise_clean "${NZ}" "${PDEP}" noise
    fi
    # CLIP-space address control: UCK mem IP, NAT structure (same F)
    if [[ -f "${UCK_OUT}/${SUB}/full/conds/ip_mem_test.npy" ]]; then
      CLIPIP="$(calib "${UCK_OUT}/${SUB}/full/conds/ip_mem_test.npy" "${NAT_OUT}/${SUB}/conds/uckmem.npy" "${SUB}_uckmem")"
      gen_hs "${SUB}" hs_clipmem_deploy "${CLIPIP}" "${PDEP}" full
    else
      warn "no UCK mem IP for CLIP-space control"
    fi
  fi

  if ! "${PYTHON}" scripts/nda/nat_measure.py \
      --out-dir "${NAT_OUT}/${SUB}" \
      --bank "${COND}/clip_img1024_test.npy" \
      --report "${NAT_OUT}/${SUB}/measure.json" \
      >> "${NAT_OUT}/logs/measure.log" 2>&1; then
    warn "measure ${SUB} failed"
  fi
  sweep_subject_disk "${SUB}"
done

echo "===== summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
ROOT = Path(os.environ.get("NAT_OUT", "outputs/nat"))
bars = [
    ("CogCap sub-08", 0.175, 0.366, 0.744),
    ("ATM sub-08",    0.160, 0.345, 0.786),
    ("CogCap 10subj", 0.150, 0.347, 0.715),
]
print(f"{'row':<32} {'Pix':>7} {'SSIM':>6} {'CLIP':>6}")
print("-" * 56)
for n, pix, ss, cl in bars:
    print(f"{n:<32} {pix:7.3f} {ss:6.3f} {cl:6.3f}")
print("-- this run --")
for p in sorted(ROOT.glob("sub-*/eval/s*.json")):
    d = json.loads(p.read_text())
    print(f"{p.parent.parent.name+'/'+p.stem:<32} "
          f"{d.get('pixcorr', float('nan')):7.3f} "
          f"{d.get('ssim', float('nan')):6.3f} "
          f"{d.get('clip', float('nan')):6.3f}")
print("-- neural book --")
for p in sorted(ROOT.glob("sub-*/full/report.json")):
    d = json.loads(p.read_text())
    loo = d.get("loo_train_1654") or {}
    print(f"{p.parent.parent.name:<8} LOO1654={loo.get('top1', float('nan')):.4f} "
          f"val1654={d.get('val_b_neural_1654_top1', float('nan')):.4f} "
          f"val82={d.get('val_b_neural_82_top1', float('nan')):.4f} "
          f"clip1654={d.get('val_b_clipimg_1654_top1', float('nan')):.4f} "
          f"ip_vs_true={d.get('ip_nat_vs_true', float('nan')):.3f} "
          f"res_vs_true={d.get('ip_res_vs_true', float('nan')):.3f}")
for m in sorted(ROOT.glob("sub-*/measure.json")):
    d = json.loads(m.read_text())
    print(f"{m.parent.name} measure: verdict={d.get('arm_agreement_verdict')} "
          f"full_vs_noise={d.get('full_vs_noise_rowcos')} "
          f"residual={d.get('residual_ip_verdict')}")
PY

echo "===== done @ $(date -Iseconds) ====="
du -sh "${NAT_OUT}" | sed 's/^/[disk] /'
du -sh "${NAT_OUT}"/sub-* 2>/dev/null | sed 's/^/[disk] /' || true
