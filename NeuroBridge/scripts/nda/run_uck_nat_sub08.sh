#!/usr/bin/env bash
# ============================================================================
# UCK-NAT K-hypothesis full pipeline on sub-08.
# Stages: diag+train → K-candidate gen → render-verify select → official eval.
# Fuse: λ=0 must bit-wise match UCK IP; innovation is additive only.
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true

export UCK_NAT_OUT="${UCK_NAT_OUT:-${NB_ROOT}/outputs/uck_nat_s08}"
export UCK_OUT="${UCK_OUT:-${NB_ROOT}/outputs/uck}"
OUT="${UCK_NAT_OUT}"
COND="${NB_ROOT}/outputs/gem/cond_cache"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="$(command -v python)"
DEVICE="${DEVICE:-cuda:0}"
KEEP_IMAGES="${KEEP_IMAGES:-1}"   # need gens for selection; wipe at end

K="${K:-8}"
LAM="${LAM:-0.5}"
LAM_SWEEP="${LAM_SWEEP:-0,0.3,0.5,1.0}"
EPOCHS="${EPOCHS:-40}"
TRAIN_ADDR="${TRAIN_ADDR:-1}"
DO_RANDOM="${DO_RANDOM:-1}"

HS_CN="${HS_CN:-0.40}"
HS_STRENGTH="${HS_STRENGTH:-0.82}"
GEN_STEPS="${GEN_STEPS:-28}"
GEN_GUIDANCE="${GEN_GUIDANCE:-5.0}"
IP_SCALE="${IP_SCALE:-1.0}"

SUB="sub-08"
SD="08"
PF_DEPLOY="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
DEPTH="${UCK_OUT}/${SUB}/full/spatial/pred_depth_rgb_512"
LL="${NB_ROOT}/outputs/sdedit_ll_full10/${SUB}/vae_head/pred_lowlevel_rgb_512"
[[ -f "${LL}/199.png" ]] || LL="${UCK_OUT}/${SUB}/full/spatial/pred_lowlevel_rgb_512"

mkdir -p "${OUT}/logs" "${OUT}/eval" "${OUT}/gen" "${OUT}/select" "${OUT}/conds" \
         "${NB_ROOT}/outputs/slurm"

log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; exit 1; }; }

require "${PF_DEPLOY}"
require "${DEPTH}/199.png"
require "${LL}/199.png"
require "${UCK_OUT}/${SUB}/full/conds/ip_mem_test.npy"
require scripts/nda/uck_nat_khyp_export.py
require scripts/nda/uck_nat_select.py
require scripts/nda/generate_hcma_s_decode.py
require scripts/nda/eval_official_seven_dir.py
require scripts/nda/gem_calib.py

# ---------- 1. export conditions (CPU-only, no training) ----------
if [[ -f "${OUT}/export_report.json" && -f "${OUT}/conds/ip_lam${LAM}_K${K}_test.npy" ]]; then
  log "[SKIP] export"
else
  log "===== export K-hyp conditions ====="
  "${PYTHON}" scripts/nda/uck_nat_khyp_export.py \
      --out "${OUT}" \
      --test-subject 8 \
      --K "${K}" \
      --lambdas "${LAM_SWEEP}" \
      --uck-ip "${UCK_OUT}/${SUB}/full/conds/ip_mem_test.npy" \
      --query-npy "${UCK_OUT}/${SUB}/full/conds/ip_q_test.npy" \
      --query-name "uck_q" \
      --alt-query-npy "${NB_ROOT}/outputs/ack_s08/heads/conds/ip_q_test.npy" \
      --exclude-self 1 \
      2>&1 | tee "${OUT}/logs/export.log"
fi
require "${OUT}/export_report.json"

"${PYTHON}" - <<PY
import json,sys
r=json.load(open("${OUT}/export_report.json"))
print("[fuse] lambda0_equals_uck =", r.get("fuse_lambda0_equals_uck"),
      "max_abs =", r.get("fuse_lambda0_max_abs"))
print("[diversity]", r.get("diversity",{}))
print("[posthoc]", r.get("diagnostics_POSTHOC",{}))
print("[anchors]", {k:v for k,v in r.get("anchors",{}).items() if k!="topK_idx_head"})
if not r.get("fuse_lambda0_equals_uck"):
    sys.exit(2)
PY

calib() {
  local src="$1" dst="$2" tag="$3"
  if [[ -f "${dst}" ]]; then echo "[SKIP] calib ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/gem_calib.py \
      --in "${src}" --out "${dst}" \
      --ref "${COND}/clip_img1024_train.npy" \
      --tag "${tag}" --report "${OUT}/logs/calib_${tag}.json" \
      >> "${OUT}/logs/calib.log" 2>&1
}

eval_row() {
  local tag="$1" gdir="$2"
  local ev="${OUT}/eval/s${SD}_${tag}.json"
  if [[ -f "${ev}" ]]; then echo "[SKIP] eval ${tag}"; return 0; fi
  require "${gdir}"
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
      --gen-dir "${gdir}" --output-json "${ev}" --tag "${tag}" \
      --images-root "${IMAGES_ROOT}" --device "${DEVICE}" \
      >> "${OUT}/logs/eval.log" 2>&1
}

gen_hs() {
  local tag="$1" cond="$2"
  local gdir="${OUT}/gen/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then
    echo "[SKIP] gen ${tag}"; return 0
  fi
  require "${cond}"
  log "===== HS ${tag} ====="
  if ! "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
      --embed-npy "${cond}" --prompts-json "${PF_DEPLOY}" \
      --depth-rgb-dir "${DEPTH}" --lowlevel-rgb-dir "${LL}" \
      --output-dir "${gdir}" --tag "${tag}" \
      --cn-scale "${HS_CN}" --strength "${HS_STRENGTH}" --ip-scale "${IP_SCALE}" \
      --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
      >> "${OUT}/logs/gen_${tag}.log" 2>&1; then
    echo "[WARN] gen ${tag} failed"; rm -rf "${gdir}"; return 1
  fi
}

# ---------- 2. λ=0 fuse (must ≈ UCK) ----------
require "${OUT}/conds/ip_lam0_k0_test.npy"
calib "${OUT}/conds/ip_lam0_k0_test.npy" "${OUT}/conds/ip_lam0_k0_cal.npy" "lam0"
gen_hs "hs_lam0_uck_fuse" "${OUT}/conds/ip_lam0_k0_cal.npy"
eval_row "hs_lam0_uck_fuse" "${OUT}/gen/hs_lam0_uck_fuse/generated"

# ---------- 3. multi-hyp @ primary λ ----------
STACK="${OUT}/conds/ip_lam${LAM}_K${K}_test.npy"
require "${STACK}"

GEN_DIRS=()
for k in $(seq 0 $((K-1))); do
  src="${OUT}/conds/ip_lam${LAM}_k${k}_test.npy"
  dst="${OUT}/conds/ip_lam${LAM}_k${k}_cal.npy"
  calib "${src}" "${dst}" "lam${LAM}_k${k}"
  gen_hs "hs_lam${LAM}_k${k}" "${dst}"
  # per-k eval (optional but useful)
  eval_row "hs_lam${LAM}_k${k}" "${OUT}/gen/hs_lam${LAM}_k${k}/generated"
  GEN_DIRS+=("${OUT}/gen/hs_lam${LAM}_k${k}")
done

# ---------- 4. render-verify selection ----------
select_and_eval() {
  local maxk="$1"
  local tag="hs_lam${LAM}_selected_k${maxk}"
  local sdir="${OUT}/select/${tag}"
  local ev="${OUT}/eval/s${SD}_${tag}.json"
  if [[ -f "${ev}" ]]; then echo "[SKIP] select ${tag}"; return 0; fi
  log "===== select K=${maxk} ====="
  local dirs=()
  local i=0
  for g in "${GEN_DIRS[@]}"; do
    dirs+=("${g}")
    i=$((i+1))
    [[ ${i} -ge ${maxk} ]] && break
  done
  "${PYTHON}" scripts/nda/uck_nat_select.py \
      --gen-dirs "${dirs[@]}" \
      --cond-npy "${UCK_OUT}/${SUB}/full/conds/ip_mem_test.npy" \
      --out-dir "${sdir}" \
      --max-k "${maxk}" \
      --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/select_k${maxk}.log"
  eval_row "${tag}" "${sdir}/selected"
}

select_and_eval 4
if (( K >= 8 )); then
  select_and_eval 8
fi

# ---------- 5. random-K control (proves gain is from anchors) ----------
if [[ "${DO_RANDOM}" == "1" ]]; then
  RDIRS=()
  for k in 0 1 2 3; do
    src="${OUT}/conds/ip_rand_lam${LAM}_k${k}_test.npy"
    [[ -f "${src}" ]] || continue
    dst="${OUT}/conds/ip_rand_lam${LAM}_k${k}_cal.npy"
    calib "${src}" "${dst}" "rand_lam${LAM}_k${k}"
    gen_hs "hs_rand_lam${LAM}_k${k}" "${dst}"
    eval_row "hs_rand_lam${LAM}_k${k}" "${OUT}/gen/hs_rand_lam${LAM}_k${k}/generated"
    RDIRS+=("${OUT}/gen/hs_rand_lam${LAM}_k${k}")
  done
  if (( ${#RDIRS[@]} >= 4 )); then
    tag="hs_rand_lam${LAM}_selected_k4"
    sdir="${OUT}/select/${tag}"
    if [[ ! -f "${OUT}/eval/s${SD}_${tag}.json" ]]; then
      log "===== select RANDOM K=4 ====="
      "${PYTHON}" scripts/nda/uck_nat_select.py \
          --gen-dirs "${RDIRS[@]}" \
          --cond-npy "${UCK_OUT}/${SUB}/full/conds/ip_mem_test.npy" \
          --out-dir "${sdir}" --max-k 4 --device "${DEVICE}" \
          2>&1 | tee "${OUT}/logs/select_rand_k4.log"
      eval_row "${tag}" "${sdir}/selected"
    fi
  fi
fi

# ---------- 6. cleanup bulky gens (keep selected + fuse) ----------
if [[ "${KEEP_IMAGES}" != "1" ]]; then
  for d in "${OUT}/gen"/hs_lam${LAM}_k* "${OUT}/gen"/hs_rand_*; do
    [[ -d "${d}" ]] && rm -rf "${d}"
  done
fi

# ---------- 7. summary ----------
log "===== summary ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
OUT = Path(os.environ["UCK_NAT_OUT"])
UCK = Path(os.environ.get("UCK_OUT", "outputs/uck"))
print(f"{'row':<40} {'Pix':>7} {'SSIM':>6} {'CLIP':>6}")
print("-" * 64)
uref = UCK / "sub-08/eval/s08_hs_mem_deploy.json"
if uref.is_file():
    d = json.loads(uref.read_text())
    print(f"{'UCK hs_mem_deploy':<40} {d.get('pixcorr', float('nan')):7.3f} "
          f"{d.get('ssim', float('nan')):6.3f} {d.get('clip', float('nan')):6.3f}")
print("CogCap sub-08                             0.175      —  0.744")
print("ATM sub-08                                0.160      —  0.786")
print("-- this run --")
rows = {}
for p in sorted((OUT / "eval").glob("s08_*.json")):
    d = json.loads(p.read_text())
    rows[p.stem] = {k: d.get(k) for k in
                    ("pixcorr","ssim","clip","alex2","alex5","inception","swav","fid")}
    print(f"{p.stem:<40} {d.get('pixcorr', float('nan')):7.3f} "
          f"{d.get('ssim', float('nan')):6.3f} {d.get('clip', float('nan')):6.3f}")
summary = {
    "pipeline": "uck_nat_s08",
    "uck_ref": json.loads(uref.read_text()) if uref.is_file() else None,
    "export": json.loads((OUT/"export_report.json").read_text()) if (OUT/"export_report.json").is_file() else None,
    "rows": rows,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"[summary] -> {OUT/'summary.json'}")
PY

log "===== done ====="
du -sh "${OUT}" | sed 's/^/[disk] /'
