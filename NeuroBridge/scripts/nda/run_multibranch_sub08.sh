#!/usr/bin/env bash
# ============================================================================
# Multi-branch IP-Adapter (CogCapPro-style) + our K-candidate / render-verify
# selection, on sub-08.
#
# Goal: close the gap to CogCapPro (SSIM 0.398 / Inception 0.779 / CLIP 0.830)
# while PRESERVING the core innovations:
#   (a) neural-address retrieval anchors -> K generation hypotheses
#   (b) render-verify best-of-N selection over the rendered candidates
#   (c) score-level fusion of a semantic route and a STRUCTURE route
#
# Arms
#   mb_p1_i1       turbo, image branch only            (turbo baseline control)
#   mb_p2_i3       turbo, image+depth+edge branches    (CogCapPro-style)
#   mb_p3_i3_cn    base  , 3 branches + depth CN + LL init (our structure + multi-branch)
#   mb_p2_i3_k*    turbo, 3 branches, K=8 hypotheses   (innovation)
#   mb_p2_i3_sel8  render-verify selection (fused)     (innovation)
#   mb_p2_i3_sel8_sem  selection, semantic route only  (fusion ablation)
#   mb_p2_i3_rand_*    anchor-shuffled control
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true

export MB_OUT="${MB_OUT:-${NB_ROOT}/outputs/mb_s08}"
OUT="${MB_OUT}"
UCK_OUT="${UCK_OUT:-${NB_ROOT}/outputs/uck}"
ANCH="${NB_ROOT}/outputs/uck_nat_s08/conds"     # neural-address retrieval anchors (K hypotheses)
COND="${NB_ROOT}/outputs/gem/cond_cache"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="$(command -v python)"
DEVICE="${DEVICE:-cuda:0}"
KEEP_IMAGES="${KEEP_IMAGES:-1}"

K="${K:-8}"
LAM="${LAM:-0.5}"
STRUCT_W="${STRUCT_W:-0.5}"
TURBO_STEPS="${TURBO_STEPS:-5}"
BASE_STEPS="${BASE_STEPS:-28}"
BRANCH_SCALES="${BRANCH_SCALES:-1.0,1.0,1.0}"
DO_SANITY="${DO_SANITY:-1}"
SCOPE="${SCOPE:-all}"     # all | turbo | stage0

SUB="sub-08"; SD="08"
PF="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
DEPTH_RGB="${UCK_OUT}/${SUB}/full/spatial/pred_depth_rgb_512"
LL_RGB="${NB_ROOT}/outputs/sdedit_ll_full10/${SUB}/vae_head/pred_lowlevel_rgb_512"
[[ -f "${LL_RGB}/199.png" ]] || LL_RGB="${UCK_OUT}/${SUB}/full/spatial/pred_lowlevel_rgb_512"

mkdir -p "${OUT}"/{logs,conds,eval,gen,select,heads} "${NB_ROOT}/outputs/slurm"

log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; exit 1; }; }

# ---------------- 0. structure heads (EEG -> depth/edge CLIP) ----------------
if [[ ! -f "${OUT}/heads/report.json" ]]; then
  log "===== train depth/edge CLIP heads ====="
  "${PYTHON}" scripts/nda/multibranch_train_heads.py \
      --out "${OUT}/heads" --test-subject 8 --modalities depth,edge \
      --epochs "${HEAD_EPOCHS:-60}" --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/train_heads.log"
fi
require "${OUT}/heads/conds/depth_pred_test_cal.npy"
require "${OUT}/heads/conds/edge_pred_test_cal.npy"
[[ "${SCOPE}" == "stage0" ]] && { log "SCOPE=stage0 done"; exit 0; }

D="${OUT}/heads/conds/depth_pred_test_cal.npy"
E="${OUT}/heads/conds/edge_pred_test_cal.npy"
IP="${UCK_OUT}/${SUB}/full/conds/ip_mem_test.npy"

# ---------------- helpers ----------------
calib() { # src dst tag ref
  local src="$1" dst="$2" tag="$3" ref="$4"
  [[ -f "${dst}" ]] && { echo "[SKIP] calib ${tag}"; return 0; }
  "${PYTHON}" scripts/nda/gem_calib.py --in "${src}" --out "${dst}" --ref "${ref}" \
      --tag "${tag}" --report "${OUT}/logs/calib_${tag}.json" >> "${OUT}/logs/calib.log" 2>&1
}

gen() { # tag pipeline branches|csv scales steps guidance [use_cn]
  local tag="$1" pipeline="$2" conds="$3" scales="$4" steps="$5" guidance="$6"
  local gdir="${OUT}/gen/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] gen ${tag}"; return 0; fi
  local cn=()
  if [[ "${7:-0}" == "1" ]]; then
    cn=(--depth-rgb-dir "${DEPTH_RGB}" --lowlevel-rgb-dir "${LL_RGB}" \
        --cn-scale "${CN_SCALE:-0.40}" --strength "${STRENGTH:-0.82}")
  fi
  log "===== gen ${tag} (${pipeline}, ${steps} steps) ====="
  "${PYTHON}" scripts/nda/generate_multibranch_decode.py \
      --cond-npys "${conds}" --branch-scales "${scales}" \
      --prompts-json "${PF}" --output-dir "${gdir}" --tag "${tag}" \
      --pipeline "${pipeline}" "${cn[@]}" \
      --gen-steps "${steps}" --gen-guidance "${guidance}" \
      --seed 42 \
      >> "${OUT}/logs/gen_${tag}.log" 2>&1 || { echo "[WARN] gen ${tag} failed"; return 1; }
  [[ -f "${gdir}/generated/199.png" ]]
}

eval_row() { # tag gendir
  local tag="$1" gdir="$2"
  local ev="${OUT}/eval/s${SD}_${tag}.json"
  [[ -f "${ev}" ]] && { echo "[SKIP] eval ${tag}"; return 0; }
  require "${gdir}"
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
      --gen-dir "${gdir}" --output-json "${ev}" --tag "${tag}" \
      --images-root "${IMAGES_ROOT}" --device "${DEVICE}" \
      >> "${OUT}/logs/eval.log" 2>&1
}

select_row() { # tag maxk struct_weight(0=semantic only) [rand]
  local tag="$1" maxk="$2" sw="$3" prefix="${4:-k}"
  local sdir="${OUT}/select/${tag}"
  local ev="${OUT}/eval/s${SD}_${tag}.json"
  [[ -f "${ev}" ]] && { echo "[SKIP] select ${tag}"; return 0; }
  local dirs=()
  if [[ "${prefix}" == "k" ]]; then
    for i in $(seq 0 $((maxk-1))); do dirs+=("${OUT}/gen/mb_p2_i3_k${i}"); done
  else
    for i in $(seq 0 $((maxk-1))); do dirs+=("${OUT}/gen/mb_p2_i3_rand_k${i}"); done
  fi
  local extra=()
  if [[ "${sw}" != "0" ]]; then extra=(--struct-npy "${D}" --struct-weight "${sw}"); fi
  log "===== select ${tag} (K=${maxk}, struct_w=${sw}) ====="
  "${PYTHON}" scripts/nda/uck_nat_select.py \
      --gen-dirs "${dirs[@]}" --cond-npy "${IP}" "${extra[@]}" \
      --out-dir "${sdir}" --max-k "${maxk}" --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/select_${tag}.log"
  eval_row "${tag}" "${sdir}/selected"
}

# ---------------- 1. branch effectiveness sanity ----------------
if [[ "${DO_SANITY}" == "1" && ! -f "${OUT}/sanity.json" ]]; then
  log "===== branch sanity (1 vs 3 branches) ====="
  "${PYTHON}" scripts/nda/generate_multibranch_decode.py \
      --cond-npys "${IP},${D},${E}" --branch-scales "${BRANCH_SCALES}" \
      --prompts-json "${PF}" --output-dir "${OUT}/sanity_gen" --tag sanity \
      --pipeline turbo --gen-steps "${TURBO_STEPS}" --gen-guidance 0.0 \
      --max-images 3 --sanity-only --branch-sanity 3 \
      --sanity-out "${OUT}/sanity.json" 2>&1 | tail -40
  "${PYTHON}" - <<PY
import json,sys
d=json.load(open("${OUT}/sanity.json"))
print("[sanity]", d["verdict"], "pixdiff=", d["pixel_diff_1branch_vs_Nbranch_mean"])
if not d["verdict"].startswith("BRANCHES_ACTIVE"):
    sys.exit(2)
PY
  rm -rf "${OUT}/sanity_gen"
fi

# ---------------- 2. single-hypothesis controls ----------------
gen "mb_p1_i1" turbo "${IP}" "1.0" "${TURBO_STEPS}" 0.0 0
eval_row "mb_p1_i1" "${OUT}/gen/mb_p1_i1/generated"

gen "mb_p2_i3" turbo "${IP},${D},${E}" "${BRANCH_SCALES}" "${TURBO_STEPS}" 0.0 0
eval_row "mb_p2_i3" "${OUT}/gen/mb_p2_i3/generated"

if [[ "${SCOPE}" == "all" ]]; then
  gen "mb_p3_i3_cn" base "${IP},${D},${E}" "${BRANCH_SCALES}" "${BASE_STEPS}" 5.0 1
  eval_row "mb_p3_i3_cn" "${OUT}/gen/mb_p3_i3_cn/generated"
fi

[[ "${SCOPE}" == "turbo" ]] && { log "SCOPE=turbo done"; exit 0; }

# ---------------- 3. K neural-address hypotheses ----------------
require "${ANCH}/anchor_idx_selfex.npy"
for k in $(seq 0 $((K-1))); do
  ASRC="${ANCH}/ip_lam${LAM}_k${k}_test.npy"
  [[ -f "${ASRC}" ]] || ASRC="${ANCH}/ip_lam${LAM}_k${k}_cal.npy"
  require "${ASRC}"
  ADST="${OUT}/conds/anch_k${k}_cal.npy"
  calib "${ASRC}" "${ADST}" "anch_k${k}" "${COND}/clip_img1024_train.npy"
  gen "mb_p2_i3_k${k}" turbo "${ADST},${D},${E}" "${BRANCH_SCALES}" "${TURBO_STEPS}" 0.0 0
  eval_row "mb_p2_i3_k${k}" "${OUT}/gen/mb_p2_i3_k${k}/generated"
done

# ---------------- 4. render-verify selection ----------------
select_row "mb_p2_i3_sel4"     4 "${STRUCT_W}"
select_row "mb_p2_i3_sel8"     "${K}" "${STRUCT_W}"
select_row "mb_p2_i3_sel8_sem" "${K}" 0          # fusion ablation (semantic only)

# ---------------- 5. random-anchor control ----------------
for k in 0 1 2 3; do
  RS="${ANCH}/ip_rand_lam${LAM}_k${k}_test.npy"
  [[ -f "${RS}" ]] || continue
  RD="${OUT}/conds/rand_k${k}_cal.npy"
  calib "${RS}" "${RD}" "rand_k${k}" "${COND}/clip_img1024_train.npy"
  gen "mb_p2_i3_rand_k${k}" turbo "${RD},${D},${E}" "${BRANCH_SCALES}" "${TURBO_STEPS}" 0.0 0
  eval_row "mb_p2_i3_rand_k${k}" "${OUT}/gen/mb_p2_i3_rand_k${k}/generated"
done
select_row "mb_p2_i3_rand_sel4" 4 "${STRUCT_W}" rand

# ---------------- 6. cleanup ----------------
if [[ "${KEEP_IMAGES}" != "1" ]]; then
  for d in "${OUT}"/gen/mb_p2_i3_k[0-9] "${OUT}"/gen/mb_p2_i3_rand_k[0-9]; do
    [[ -d "${d}" ]] && rm -rf "${d}"
  done
fi

# ---------------- 7. summary ----------------
log "===== summary ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
OUT = Path(os.environ["MB_OUT"]); UCK = Path(os.environ.get("UCK_OUT","outputs/uck"))
print(f"{'row':<28}{'Pix':>8}{'SSIM':>8}{'Alex5':>8}{'Incep':>8}{'CLIP':>8}{'SwAV':>8}{'FID':>8}")
print("-"*84)
def line(name, d):
    f=lambda k: d.get(k)
    print(f"{name:<28}"+''.join(
        (f"{f(k):8.3f}" if isinstance(f(k),(int,float)) else f"{'—':>8}")
        for k in ("pixcorr","ssim","alex5","inception","clip","swav","fid")))
u = UCK/"sub-08/eval/s08_hs_mem_deploy.json"
if u.is_file(): line("UCK (28st, CN+init)", json.loads(u.read_text()))
print(f"{'CogCapPro (paper)':<28}{0.163:8.3f}{0.398:8.3f}{'—':>8}{0.779:8.3f}{0.830:8.3f}{0.553:8.3f}{'—':>8}")
print("-"*84)
rows={}
for p in sorted((OUT/"eval").glob("s08_*.json")):
    d=json.loads(p.read_text()); rows[p.stem]=d
    line(p.stem, d)
summ={"pipeline":"multibranch_s08","heads":None,"rows":rows}
h=OUT/"heads/report.json"
if h.is_file(): summ["heads"]=json.loads(h.read_text()).get("modalities")
s=OUT/"sanity.json"
if s.is_file(): summ["branch_sanity"]=json.loads(s.read_text())
(OUT/"summary.json").write_text(json.dumps(summ, indent=2))
print(f"[summary] -> {OUT/'summary.json'}")
PY
log "===== done ====="
du -sh "${OUT}" | sed 's/^/[disk] /'
