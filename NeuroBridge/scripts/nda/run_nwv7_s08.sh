#!/usr/bin/env bash
# =============================================================================
# NW-v7 sub-08: break the IMAGE/SEMANTIC bottleneck.
#
# Locked from NW-v6 (job 589824, cancelled mid-M6 after the 2x2 completed):
#   * generator is NOT the problem -- W4 (GT image + our structure) = incep 0.9557
#   * structure is nearly saturated when image is GT (recovers 82% of A9-A5)
#   * encoder-side stack (STH/SCM/fusion/uncertainty) did not move deployable numbers
#   * self-conditioning via CLIP(pass-1) is CLOSED: pass-1 CLIP is farther from GT
#     than SEM already is (rowcos 0.40 vs SEM 0.57)
#
# This job only changes the IMAGE condition bank, holding the verified operator
# (cc3 + band anchor + Turbo+CFG0 + empty prompt) and the verified structure banks
# (G3 head, JOINT depth/edge) fixed.
#
# Arms
#   V1  ahi_varest + G3 structure          control: same SEM content, varest form
#   V2  cfmsf_varest + G3                  identity transplant (top1 61x -> manifold)
#   V3  uge_varest + G3
#   V4  blend_c3 + G3                      cfmsf*0.3 + a_hi, then varest
#   V5  blend_c5 + G3
#   V6  dedicated + G3                     image-only head, cat features
#   V7  best-of-{V2..V6 by bank score} + JOINT structure   crosses structure source
#   V8  cfmsf_varest + JOINT structure, sigma 4.0
#   then M6 over the new pool + prior winners
# =============================================================================

set -uo pipefail
NB_ROOT="/project/peilab/why/NeuroBridge"
cd "${NB_ROOT}"

PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
STAG="${STAG:-sub-08}"
SUBJ="${SUBJ:-8}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-${NB_ROOT}/outputs/nw7_s08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
CC="${NB_ROOT}/outputs/gem/cond_cache"
SPECS="${NB_ROOT}/outputs/nw5_s08/specs"
CONDS="${OUT_ROOT}/conds"
NW5_CONDS="${NB_ROOT}/outputs/nw5_s08/conds"
NW6_CONDS="${NB_ROOT}/outputs/nw6_s08/conds"
INIT_RGB="${NB_ROOT}/outputs/sdedit_ll_full10/${STAG}/vae_head/pred_lowlevel_rgb_512"
EMPTY_PROMPTS="${NB_ROOT}/outputs/nw5_s08/conds/prompts_empty.json"
SEM_REP="${CONDS}/sem_${STAG}_report.json"

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export HOME="${XDG_CACHE_HOME}"

LOG="${OUT_ROOT}/logs"
mkdir -p "${LOG}" "${CONDS}" "${OUT_ROOT}/arms"
log() { echo "[$(date +%H:%M:%S)] $*"; }
hr()  { echo "------------------------------------------------------------------------"; }

for f in "${EMPTY_PROMPTS}" "${SPECS}/cc3.json" "${CC}/clip_img1024_train.npy" \
         "${NW5_CONDS}/head_depth1024_${STAG}_test.npy" \
         "${NW5_CONDS}/head_edge1024_${STAG}_test.npy" \
         "${NW6_CONDS}/joint_depth1024_${STAG}_test.npy" \
         "${NW6_CONDS}/joint_edge1024_${STAG}_test.npy" \
         "${NB_ROOT}/outputs/cfmsf_s08/train/conds/q_img_test.npy" \
         "${NB_ROOT}/outputs/nw4_10s/arms/a_hi/conds/${STAG}/cal_test.npy"; do
  [[ -f "${f}" ]] || { echo "[FATAL] missing ${f}" >&2; exit 1; }
done
[[ -d "${INIT_RGB}" ]] || { echo "[FATAL] missing ${INIT_RGB}" >&2; exit 1; }

log "===== [0] CUDA ====="
"${PYTHON}" - <<'PY' || exit 1
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print(f"  {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
PY

log "===== [1] rebuild IMAGE banks (transplant + blend + dedicated head) ====="
"${PYTHON}" "${NB_ROOT}/scripts/nda/nwv7_sem_banks.py" \
  --subject "${SUBJ}" --stag "${STAG}" --out-dir "${CONDS}" --out "${SEM_REP}" \
  --device cuda:0 --epochs 120 --patience 20 --batch 384 \
  > "${LOG}/sem_banks.log" 2>&1
RC=$?
grep -E "^\[sem\]" "${LOG}/sem_banks.log" | tail -40 || true
[[ "${RC}" -eq 0 ]] || { log "FATAL sem banks failed"; tail -n 30 "${LOG}/sem_banks.log"; exit 1; }

# pick the bank with the highest proxy score for the "best + JOINT structure" arm
BEST_SEM="$("${PYTHON}" - "${SEM_REP}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ranked = d.get("ranked") or [{"name": k} for k in d["banks"]]
# skip ahi_varest as "best" -- it is the control
for r in ranked:
    if r["name"] != "sem_ahi_varest":
        print(r["name"]); break
else:
    print(ranked[0]["name"])
PY
)"
log "  BEST_SEM (proxy) = ${BEST_SEM}"

cond_path() {
  case "$1" in
    AHIV)  echo "${CONDS}/sem_ahi_varest_${STAG}_test.npy" ;;
    CFMV)  echo "${CONDS}/sem_cfmsf_varest_${STAG}_test.npy" ;;
    UGEV)  echo "${CONDS}/sem_uge_varest_${STAG}_test.npy" ;;
    BLC3)  echo "${CONDS}/sem_blend_c3_${STAG}_test.npy" ;;
    BLC5)  echo "${CONDS}/sem_blend_c5_${STAG}_test.npy" ;;
    BLC7)  echo "${CONDS}/sem_blend_c7_${STAG}_test.npy" ;;
    DED)   echo "${CONDS}/sem_dedicated_${STAG}_test.npy" ;;
    BEST)  echo "${CONDS}/${BEST_SEM}_${STAG}_test.npy" ;;
    DEPH)  echo "${NW5_CONDS}/head_depth1024_${STAG}_test.npy" ;;
    EDGEH) echo "${NW5_CONDS}/head_edge1024_${STAG}_test.npy" ;;
    JDEP)  echo "${NW6_CONDS}/joint_depth1024_${STAG}_test.npy" ;;
    JEDGE) echo "${NW6_CONDS}/joint_edge1024_${STAG}_test.npy" ;;
    *)     echo "" ;;
  esac
}

gen_one() {
  local ARM="$1" CKEYS="$2" BLUR="$3" SEEDV="$4" GEN="$5"
  local CLIST="" p
  for k in ${CKEYS//,/ }; do
    p="$(cond_path "${k}")"
    if [[ -z "${p}" || ! -f "${p}" ]]; then log "WARN ${ARM}: missing ${k} -> ${p}"; return 1; fi
    CLIST="${CLIST:+${CLIST},}${p}"
  done
  if [[ -f "${GEN}/generated/199.png" ]]; then
    log "--- ${ARM}: present - skip"; return 0
  fi
  if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/generate_layered_decode.py" \
      --cond-npys "${CLIST}" --ip-scale-json "${SPECS}/cc3.json" \
      --prompts-json "${EMPTY_PROMPTS}" --output-dir "${GEN}" --tag "${ARM}_${STAG}" \
      --pipeline turbo --use-cn 0 --use-init 1 --lowlevel-rgb-dir "${INIT_RGB}" \
      --cn-scale 0.28 --strength 0.92 --init-blur-sigma "${BLUR}" \
      --gen-steps 15 --gen-guidance 0.0 --gen-size 512 --seed "${SEEDV}" --device cuda:0 \
      --layer-report "${OUT_ROOT}/arms/${ARM}/layer_report.json" \
      > "${LOG}/${ARM}_gen.log" 2>&1; then
    log "WARN ${ARM}: generation FAILED"; tail -n 12 "${LOG}/${ARM}_gen.log"; return 1
  fi
  return 0
}

eval_one() {
  local ARM="$1" GEN="$2"
  local EV="${OUT_ROOT}/arms/${ARM}/eval/${STAG}.json"
  mkdir -p "${OUT_ROOT}/arms/${ARM}/eval"
  if [[ ! -f "${EV}" ]]; then
    if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
        --gen-dir "${GEN}/generated" --output-json "${EV}" --tag "${ARM}_${STAG}" \
        --images-root "${IMAGES_ROOT}" --device cuda:0 \
        > "${LOG}/${ARM}_eval.log" 2>&1; then
      log "WARN ${ARM}: eval FAILED"; tail -n 12 "${LOG}/${ARM}_eval.log"; return 1
    fi
  fi
  "${PYTHON}" - "${EV}" "${ARM}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  [{sys.argv[2]}] pixcorr {d['pixcorr']:.4f}  ssim {d['ssim']:.4f}  "
      f"incep {d['inception']:.4f}  clip {d['clip']:.4f}  alex2 {d['alex2']:.4f}  "
      f"alex5 {d['alex5']:.4f}  swav {d['swav']:.4f}  fid {d['fid']:.2f}")
PY
}

run_arm() {
  local ARM="$1" GEN="${OUT_ROOT}/arms/$1/gen/${STAG}"
  mkdir -p "${GEN}"
  # skip if already fully generated (zero-padded 199)
  gen_one "$1" "$2" "$3" "$4" "${GEN}" || return 1
  eval_one "${ARM}" "${GEN}"
}

DONE=()
hr; log "image-bank x structure arms"
while IFS='|' read -r A K B; do
  [[ -z "${A}" ]] && continue
  # skip arms whose image bank file is missing (e.g. dedicated failed)
  ok=1
  for k in ${K%%,*}; do
    p="$(cond_path "${k}")"
    if [[ -z "${p}" || ! -f "${p}" ]]; then ok=0; log "SKIP ${A}: no ${k}"; fi
    break
  done
  [[ "${ok}" -eq 1 ]] || continue
  hr; log "arm ${A}: keys=${K} sigma=${B}"
  run_arm "${A}" "${K}" "${B}" "${SEED}" && DONE+=("${A}")
done <<'SPEC'
V1_ahi_g3_s30|AHIV,DEPH,EDGEH|3.0
V2_cfmsf_g3_s30|CFMV,DEPH,EDGEH|3.0
V3_uge_g3_s30|UGEV,DEPH,EDGEH|3.0
V4_blendc3_g3_s30|BLC3,DEPH,EDGEH|3.0
V5_blendc5_g3_s30|BLC5,DEPH,EDGEH|3.0
V6_ded_g3_s30|DED,DEPH,EDGEH|3.0
V7_best_joint_s30|BEST,JDEP,JEDGE|3.0
V8_cfmsf_joint_s40|CFMV,JDEP,JEDGE|4.0
SPEC

log "===== [3] M6 over new semantic arms + prior winners ====="
M6_POOL=""
for A in V1_ahi_g3_s30 V2_cfmsf_g3_s30 V3_uge_g3_s30 V4_blendc3_g3_s30 \
         V5_blendc5_g3_s30 V6_ded_g3_s30 V7_best_joint_s30 V8_cfmsf_joint_s40 \
         G3_head_s30 E2_varest_s30 F_s40 M6v5_a10b10 M6v6_a10b10; do
  if [[ "${A}" == G3_* || "${A}" == E2_* || "${A}" == F_* || "${A}" == M6v5_* ]]; then
    GD="${NB_ROOT}/outputs/nw5_s08/arms/${A}/gen/${STAG}"
  elif [[ "${A}" == M6v6_* ]]; then
    GD="${NB_ROOT}/outputs/nw6_s08/arms/${A}/gen/${STAG}"
  else
    GD="${OUT_ROOT}/arms/${A}/gen/${STAG}"
  fi
  if [[ -f "${GD}/generated/199.png" ]]; then
    M6_POOL="${M6_POOL:+${M6_POOL},}${GD}"
  fi
done
log "  pool size: $(echo "${M6_POOL}" | tr ',' '\n' | grep -c . || true)"
# M6 semantic score needs a SEM reference; use the BEST new bank (not a_hi) so selection
# agrees with the identity we are testing
SEM_FOR_M6="$(cond_path BEST)"
[[ -f "${SEM_FOR_M6}" ]] || SEM_FOR_M6="$(cond_path AHIV)"
M6SC="${OUT_ROOT}/m6_scores_nwv7.npz"
for ab in 1.0:0.0 1.0:0.5 1.0:1.0 0.5:1.0 0.0:1.0; do
  A="${ab%%:*}"; B="${ab##*:}"; NAME="M6v7_a${A/./}b${B/./}"
  ODIR="${OUT_ROOT}/arms/${NAME}/gen/${STAG}"; mkdir -p "${ODIR}"
  hr; log "M6 ${NAME}: alpha=${A} beta=${B}"
  if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/nw6_m6_select.py" \
      --cand-dirs "${M6_POOL}" --sem-cond "${SEM_FOR_M6}" --anchor-dir "${INIT_RGB}" \
      --scores-npz "${M6SC}" --standardize --alpha "${A}" --beta "${B}" \
      --out-dir "${ODIR}" --report "${OUT_ROOT}/arms/${NAME}/m6_report.json" \
      --device cuda:0 > "${LOG}/${NAME}_sel.log" 2>&1; then
    log "WARN ${NAME}: selection FAILED"; tail -n 10 "${LOG}/${NAME}_sel.log"; continue
  fi
  eval_one "${NAME}" "${ODIR}" && DONE+=("${NAME}")
done

log "===== [4] table + verdict ====="
"${PYTHON}" - "${OUT_ROOT}" "${NB_ROOT}" "${STAG}" "${SEM_REP}" "${BEST_SEM}" <<'PY'
import json, sys
from pathlib import Path
root, nb, stag, sem_rep, best_sem = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), sys.argv[5]
KEYS = ["pixcorr","ssim","inception","clip","alex2","alex5","swav","fid"]
LBL = {
    "E1_sem_s30": "E1   SEM only (floor)",
    "E2_varest_s30": "E2   + linear structure",
    "G3_head_s30": "G3   a_hi SEM + head structure     [prev best structure]",
    "M6v5_a10b10": "M6v5 selection (prev deployable)",
    "M6v6_a10b10": "M6v6 selection (nw6 pool)",
    "W2_joint_struct_s30": "W2   SEM/JOINT + JOINT structure",
    "W4_gtimg_jstruct_s30": "W4   GT image + JOINT structure   [oracle image]",
    "A5_band_cc3": "A5   SEM + GT structure            [oracle structure]",
    "V1_ahi_g3_s30": "V1   ahi_varest + G3               [control]",
    "V2_cfmsf_g3_s30": "V2   cfmsf_varest + G3             [identity transplant]",
    "V3_uge_g3_s30": "V3   uge_varest + G3",
    "V4_blendc3_g3_s30": "V4   blend cfmsf0.3+ahi + G3",
    "V5_blendc5_g3_s30": "V5   blend cfmsf0.5+ahi + G3",
    "V6_ded_g3_s30": "V6   dedicated image head + G3",
    "V7_best_joint_s30": f"V7   {best_sem} + JOINT structure",
    "V8_cfmsf_joint_s40": "V8   cfmsf_varest + JOINT, sigma 4",
    "M6v7_a10b00": "M6v7 a=1 b=0", "M6v7_a10b05": "M6v7 a=1 b=0.5",
    "M6v7_a10b10": "M6v7 a=1 b=1", "M6v7_a05b10": "M6v7 a=0.5 b=1",
    "M6v7_a00b10": "M6v7 a=0 b=1",
}
rows = {}
for f in (root/"arms").glob(f"*/eval/{stag}.json"):
    rows[f.parent.parent.name] = json.load(open(f))
for src in (nb/"outputs/nw5_s08", nb/"outputs/nw6_s08"):
    for f in (src/"arms").glob(f"*/eval/{stag}.json"):
        rows.setdefault(f.parent.parent.name, json.load(open(f)))

print()
print("="*128)
print(f"NW-v7 {stag}: image-bank breakthrough (structure + generator held fixed)")
print("="*128)
hdr = f"{'arm':<52}" + "".join(f"{k:>9}" for k in KEYS)
print(hdr); print("-"*len(hdr))
for a in [k for k in LBL if k in rows] + [k for k in sorted(rows) if k not in LBL]:
    r = rows[a]
    print(f"{LBL.get(a,a[:50]):<52}" + "".join(
        f"{r[k]:>9.4f}" if k!="fid" else f"{r[k]:>9.2f}" for k in KEYS))

def g(a,k):
    return rows[a][k] if a in rows and k in rows[a] else None

print()
print("="*128)
print("Did the new IMAGE bank buy generation?  (crossed vs G3_head_s30 = a_hi + G3)")
print("="*128)
base = "G3_head_s30"
for a in ("V1_ahi_g3_s30","V2_cfmsf_g3_s30","V3_uge_g3_s30","V4_blendc3_g3_s30",
          "V5_blendc5_g3_s30","V6_ded_g3_s30","V7_best_joint_s30","V8_cfmsf_joint_s40"):
    if a not in rows or base not in rows: continue
    d = {k: rows[a][k]-rows[base][k] for k in ("pixcorr","ssim","inception","clip","alex2","alex5","swav","fid")}
    # for swav/fid lower is better so flip sign in the summary arrow
    print(f"  {a} - G3:  pix {d['pixcorr']:+.4f}  ssim {d['ssim']:+.4f}  "
          f"incep {d['inception']:+.4f}  clip {d['clip']:+.4f}  "
          f"alex2 {d['alex2']:+.4f}  alex5 {d['alex5']:+.4f}  "
          f"swav {d['swav']:+.4f}  fid {d['fid']:+.2f}")

print()
print("distance to oracle-image (W4) and oracle-structure (A5):")
w4, a5, e1 = g("W4_gtimg_jstruct_s30","inception"), g("A5_band_cc3","inception"), g("E1_sem_s30","inception")
for a in ("G3_head_s30","V2_cfmsf_g3_s30","V6_ded_g3_s30","V7_best_joint_s30",
          "M6v5_a10b10","M6v6_a10b10","M6v7_a10b10","M6v7_a10b00"):
    v = g(a,"inception")
    if v is None or w4 is None or e1 is None: continue
    print(f"  {a:<22} incep {v:.4f}   of floor->W4 recovers {(v-e1)/(w4-e1):.1%}   "
          f"of floor->A5 recovers {(v-e1)/(a5-e1):.1%}" if a5 else f"  {a}: {v:.4f}")

print()
BAR = {"pixcorr":0.166,"ssim":0.409,"inception":0.831,"clip":0.903,"alex2":0.818,"alex5":0.913}
print("vs CogCapPro sub-08 (6 axes):")
for a in ("G3_head_s30","V2_cfmsf_g3_s30","V6_ded_g3_s30","M6v7_a10b10","M6v7_a10b00",
          "W4_gtimg_jstruct_s30","A5_band_cc3"):
    r = rows.get(a)
    if not r: continue
    wins = [(k, r[k] > b) for k,b in BAR.items()]
    print(f"  {a:<22} {sum(w for _,w in wins)}/6  " +
          " ".join(f"{k[:4]}={'W' if w else 'l'}" for k,w in wins))

if sem_rep.is_file():
    d = json.load(open(sem_rep))
    print()
    print("image-bank quality (pre-generation):")
    for r in d.get("ranked", []):
        print(f"  {r['name']:<22} score {r['score']:.3f}  top1 {r['top1']*200:>5.1f}x  "
              f"rowcos {r['rowcos']:.4f}  dev_corr {r['dev_corr']:.4f}")
PY

log "===== done ====="
