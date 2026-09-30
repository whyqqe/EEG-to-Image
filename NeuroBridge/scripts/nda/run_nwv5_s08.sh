#!/usr/bin/env bash
# =============================================================================
# NW-v5 sub-08: the optimal architecture, assembled from everything measured.
#
# THE ARCHITECTURE
# ----------------
#   0. injection layout    cc3 (CogCapPro's asymmetric dict), verbatim
#   1. pixel constraint    NONE.  No ControlNet.  Only the L3 band anchor:
#                          our predicted low-level image, Gaussian-blurred, as img2img init
#   2. semantic branch     a_hi semantic bank (our best; correctly scaled: %E(dev) 0.624
#                          against its target's 0.606)
#   3. structure branch    variance-restored structure, OR the trained S1 head if it
#                          clears its gate against linear
#   4. pipeline            SDXL-Turbo, 15 steps, CFG 0.0, empty prompt
#   5. selection           M6: multi-candidate render-verify over a heterogeneous pool
#
# WHY EACH PIECE IS THERE (one crossed variable each, all sub-08, these scorers)
# -----------------------------------------------------------------------------
#   layout is cc3, not uniform:
#       A3b  mass-matched uniform   incep 0.7280
#       A2   cc3                    incep 0.8394      the asymmetry is mechanical
#
#   no ControlNet, but a low-frequency anchor IS kept:
#       A2   cc3, no pixel path     incep 0.8394  ssim 0.2810
#       A5   cc3 + band anchor s=3  incep 0.8400  ssim 0.3756
#       A6   cc3, CN + init         incep 0.8034  ssim 0.3541
#       -> +0.095 SSIM from the anchor at ZERO semantic cost, and keeping ControlNet
#          instead costs -0.037 incep AND -0.022 ssim.
#
#   the anchor sigma sits at or above 3.0:
#       s=3.0  ssim 0.3706   s=2.0 0.3651   s=1.5 0.3600   s=0.5 0.2998
#       -> SSIM worsens monotonically as the anchor is sharpened.  Our anchor is a
#          PREDICTION, so its high frequencies are wrong and blurring them protects
#          SSIM.  The untested direction is UP, which is what this job sweeps.
#
#   structure is variance-restored, and the UCK path is rejected:
#       E1  SEM only                incep 0.7302
#       E2  + varest ridge          incep 0.7363  clip 0.8215  alex2 0.8063
#       E3  + varest UCK            incep 0.7190  <- BELOW the no-structure floor
#       -> the shipped UCK->CLIP path has cos(m) 0.664/0.180 against the GT means and
#          dev_corr 0.032/0.000, i.e. neither the right prior nor any trial information.
#          It is dead.  Do not patch it again.
#
#   M6 selection:
#       E2                          pixcorr 0.1940  clip 0.8215  alex2 0.8063
#       M6 (4 seeds, semantic only) pixcorr 0.1999  clip 0.8510  alex2 0.8191
#       -> +0.0295 CLIP for -0.0056 incep.  And beta > 0 has never been run, which is
#          the one selection criterion aimed at SSIM.
#
# THE THREE PARTS
# ---------------
#  [2]  S1 structural head, gated: a nonlinear head trained on the DEVIATION with an
#       InfoNCE objective, against a linear ridge fitted on the same rows with the same
#       validation.  This is the only remaining path to A5 (incep 0.8400), because the
#       linear map is provably exhausted: re-targeting it at the deviation leaves
#       dev_corr at 0.3599/0.3485 to four decimals.  The head only proceeds to
#       generation if it beats the linear baseline's validation dev_top1 by >= 10%;
#       otherwise the structure branch is reported as a dead end rather than tuned.
#  [3]  sigma sweep UPWARD (4.0 / 5.0 / 6.0 / 8.0) with the best available structure.
#       The only lever left on PixCorr/SSIM that needs no new mechanism.
#  [4]  M6 over a HETEROGENEOUS pool -- every arm generated in this job plus the E2 seed
#       variants -- with a real (alpha, beta) sweep.  Because the score matrices are
#       cached, each selection costs one evaluation and no generation, and beta > 0 is
#       the criterion that pushes SSIM directly.
#
# REPORTED AGAINST (10-subject means): CogCapPro pixcorr 0.163 ssim 0.398 incep 0.779
# clip 0.830;  D2-FOSA 0.193/0.350;  MB2C 0.188/0.333;  ENIGMA 0.167/0.426 incep 0.765
# clip 0.803;  ATM 0.160/0.345 incep 0.734 clip 0.786.  The only strictly comparable
# single-subject bar is CogCapPro sub-08: 0.166/0.409/0.831/0.903.
# =============================================================================

set -uo pipefail

NB_ROOT="/project/peilab/why/NeuroBridge"
cd "${NB_ROOT}"

PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
STAG="${STAG:-sub-08}"
SUBJ="${SUBJ:-8}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-${NB_ROOT}/outputs/nw5_s08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
CC="${NB_ROOT}/outputs/gem/cond_cache"
SPECS="${OUT_ROOT}/specs"
CONDS="${OUT_ROOT}/conds"
INIT_RGB="${NB_ROOT}/outputs/sdedit_ll_full10/${STAG}/vae_head/pred_lowlevel_rgb_512"

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export HOME="${XDG_CACHE_HOME}"

LOG="${OUT_ROOT}/logs"
EMPTY_PROMPTS="${CONDS}/prompts_empty.json"
SEM_COND="${NB_ROOT}/outputs/nw4_10s/arms/a_hi/conds/${STAG}/cal_test.npy"
mkdir -p "${LOG}" "${CONDS}" "${OUT_ROOT}/arms"

log() { echo "[$(date +%H:%M:%S)] $*"; }
hr()  { echo "------------------------------------------------------------------------"; }

for f in "${SEM_COND}" "${EMPTY_PROMPTS}" "${SPECS}/cc1.json" "${SPECS}/cc3.json" \
         "${CONDS}/ridge_depth1024_${STAG}_test.npy" \
         "${CONDS}/eeg_depth1024_${STAG}_test.npy"; do
  [[ -f "${f}" ]] || { echo "[FATAL] missing ${f}" >&2; exit 1; }
done
[[ -d "${INIT_RGB}" ]] || { echo "[FATAL] missing anchor dir ${INIT_RGB}" >&2; exit 1; }

# ---------------------------------------------------------------------------
log "===== [0] fail fast on CUDA ====="
"${PYTHON}" - <<'PY' || exit 1
import torch
if not torch.cuda.is_available():
    raise SystemExit("[FATAL] CUDA unavailable - refusing to silently fall back to CPU")
print(f"  torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
PY

# ---------------------------------------------------------------------------
log "===== [1] variance-restore the linear structure banks (the E2 recipe) ====="
log "  scale fixed from the GT TRAIN bank's variability only: no test rows, no labels"
"${PYTHON}" - "${CC}" "${CONDS}" "${STAG}" <<'PY' || exit 1
import sys, json
from pathlib import Path
import numpy as np
cc, conds, stag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

def l2n(x): return (x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)).astype(np.float32)
def esplit(x):
    x=l2n(x); m=x.mean(0,keepdims=True); d=x-m
    return float((m**2).sum()), float((d**2).sum(1).mean())
def solve_share(m, dev, tgt):
    lo,hi=0.0,1e7
    for _ in range(80):
        a=0.5*(lo+hi); c=l2n(m+dev*a); d=c-c.mean(0,keepdims=True)
        if float((d**2).sum(1).mean())<tgt: lo=a
        else: hi=a
    return 0.5*(lo+hi)

rep={}
tgt_share={lab: esplit(np.load(cc/f"clip_{lab}1024_train.npy").astype(np.float32))[1]
           for lab in ("depth","edge")}
for lab in ("depth","edge"):
    for tag, stem in (("varestridge", f"ridge_{lab}1024_{stag}_test.npy"),
                      ("varest_uck", f"eeg_{lab}1024_{stag}_test.npy")):
        p = conds/stem
        if not p.is_file(): print(f"  MISSING {p}"); continue
        P=np.load(p).astype(np.float32); m=P.mean(0,keepdims=True)
        a=solve_share(m, P-m, tgt_share[lab])
        arr=l2n(m+(P-m)*a); o=conds/f"{tag}_{lab}1024_{stag}_test.npy"
        np.save(o, arr)
        em,ed=esplit(P); em2,ed2=esplit(arr)
        rep[o.name]={"scale":float(a),"dev_energy_in":ed,"dev_energy_out":ed2}
        print(f"  {o.name:<46} dev energy {ed:.3f} -> {ed2:.3f} (target {tgt_share[lab]:.3f})")
(conds/"varest_report.json").write_text(json.dumps(rep,indent=2),encoding="utf-8")
PY

# ---------------------------------------------------------------------------
log "===== [2] S1 structural head, trained on the DEVIATION, gated against linear ====="
HEAD_REP="${CONDS}/head_${STAG}_report.json"
"${PYTHON}" "${NB_ROOT}/scripts/nda/nwv5_struct_head.py" \
  --subject "${SUBJ}" --stag "${STAG}" --out "${HEAD_REP}" --device cuda:0 \
  --epochs 140 --batch 384 --out-dir "${CONDS}" \
  > "${LOG}/struct_head.log" 2>&1
HEAD_RC=$?
grep -E "^\[head\]|^  (depth|edge):|^    epoch" "${LOG}/struct_head.log" | tail -14 || true
if [[ "${HEAD_RC}" -ne 0 ]]; then
  log "WARN structure head training failed (rc=${HEAD_RC}) - falling back to the linear banks"
  tail -n 20 "${LOG}/struct_head.log"
fi

# The sigma sweep and the M6 pool's base arm use the LINEAR bank, because it is the
# known-good deployable choice (E2 = incep 0.7363, the best structure result we have).
# The trained head gets its OWN arms instead of being pre-judged on embedding metrics --
# those have already misled us once (dev_corr 0.352 bought ~5% of the GT gain, while
# dev_corr 1.0 bought +0.128), so the head's verdict comes from generation, not from
# val dev_corr.  The tier below is a prediction, not a decision.
if [[ -s "${HEAD_REP}" ]]; then
  log "  head tiers (predicted, to be checked against generation):"
  "${PYTHON}" - "${HEAD_REP}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for m, g in d.get("gate", {}).items():
    h = d["head"][m]
    print(f"    {m:<6} {g['tier'].upper():<7} val dev_top1 x{g['val_gain_top1']:.2f}  "
          f"val dev_corr x{g['val_gain_corr']:.2f}  test dev_corr "
          f"{h['test']['dev_corr']:.4f} vs linear {h['linear_test']['dev_corr']:.4f}  "
          f"[{h['variant']}]")
PY
fi
HEAD_DEP="${CONDS}/head_depth1024_${STAG}_test.npy"
HEAD_EDGE="${CONDS}/head_edge1024_${STAG}_test.npy"
HAVE_HEAD=0
if [[ -f "${HEAD_DEP}" && -f "${HEAD_EDGE}" ]]; then HAVE_HEAD=1; fi
log "  trained head banks present: ${HAVE_HEAD} (1 = the head arms will be generated)"
for m in depth edge; do
  f="${CONDS}/varestridge_${m}1024_${STAG}_test.npy"
  [[ -f "${f}" ]] || { echo "[FATAL] no linear structure bank for ${m}" >&2; exit 1; }
done

log "===== [2.5] what each available structure bank carries ====="
"${PYTHON}" - "${CC}" "${CONDS}" "${STAG}" "${HAVE_HEAD}" <<'PY'
import sys
from pathlib import Path
import numpy as np
cc, conds, stag, have_head = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
def l2n(x): return (x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)).astype(np.float32)
def esplit(x):
    x=l2n(x); m=x.mean(0,keepdims=True); d=x-m
    return float((m**2).sum()), float((d**2).sum(1).mean())
def dm(pred,tgt):
    """top1 and cosine on BOTH full rows and the mean-removed deviations.  The two rows
    differ a lot (row cosine 0.72 vs deviation cosine 0.36 on the same bank), so printing
    only one of them is how a diagnostic table ends up contradicting the analysis."""
    p,t=l2n(pred),l2n(tgt); S=p@t.T
    top1=float((S.argmax(1)==np.arange(len(t))).mean())
    rowc=float((p*t).sum(1).mean())
    pd,td=l2n(p-p.mean(0,keepdims=True)),l2n(t-t.mean(0,keepdims=True))
    return top1,rowc,float((pd*td).sum(1).mean())
print(f"  {'bank':<36}{'%E(mean)':>10}{'%E(dev)':>9}{'top1':>9}{'xchance':>9}"
      f"{'rowcos':>9}{'dev_corr':>10}")
print("  "+"-"*93)
for lab in ("depth","edge"):
    G=l2n(np.load(cc/f"clip_{lab}1024_test.npy").astype(np.float32))
    cands=[("GT (A5, the target)", None),
           ("varest linear (E2 recipe)", f"varestridge_{lab}1024_{stag}_test.npy")]
    if have_head=="1": cands.append(("TRAINED HEAD (S1)", f"head_{lab}1024_{stag}_test.npy"))
    for name,fn in cands:
        if fn is None: x=G
        else:
            p=conds/fn
            if not p.is_file(): print(f"  {name:<36} MISSING"); continue
            x=np.load(p).astype(np.float32)
        em,ed=esplit(x); t1,rc,dc=dm(x,G)
        print(f"  {name:<36}{em:>10.3f}{ed:>9.3f}{t1:>9.4f}{t1*200:>9.1f}{rc:>9.4f}{dc:>10.4f}")
PY

cond_path() {
  case "$1" in
    SEM)   echo "${SEM_COND}" ;;
    DEPL)  echo "${CONDS}/varestridge_depth1024_${STAG}_test.npy" ;;
    EDGEL) echo "${CONDS}/varestridge_edge1024_${STAG}_test.npy" ;;
    DEPH)  echo "${CONDS}/head_depth1024_${STAG}_test.npy" ;;
    EDGEH) echo "${CONDS}/head_edge1024_${STAG}_test.npy" ;;
    *)     echo "" ;;
  esac
}

# ---------------------------------------------------------------------------
log "===== [3] generation helpers ====="
gen_one() {
  # $1 arm  $2 cond-keys  $3 scale-json  $4 blur  $5 strength  $6 seed  $7 outdir
  local ARM="$1" CKEYS="$2" SJSON="$3" BLUR="$4" STRENGTH="$5" SEEDV="$6" GEN="$7"
  local CLIST="" p
  for k in ${CKEYS//,/ }; do
    p="$(cond_path "${k}")"
    if [[ -z "${p}" || ! -f "${p}" ]]; then log "WARN ${ARM}: no condition ${k}"; return 1; fi
    CLIST="${CLIST:+${CLIST},}${p}"
  done
  [[ -f "${GEN}/generated/199.png" ]] && { log "--- ${ARM}: present - skip"; return 0; }
  if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/generate_layered_decode.py" \
      --cond-npys "${CLIST}" --ip-scale-json "${SPECS}/${SJSON}.json" \
      --prompts-json "${EMPTY_PROMPTS}" --output-dir "${GEN}" --tag "${ARM}_${STAG}" \
      --pipeline turbo --use-cn 0 --use-init 1 --lowlevel-rgb-dir "${INIT_RGB}" \
      --cn-scale 0.28 --strength "${STRENGTH}" --init-blur-sigma "${BLUR}" \
      --gen-steps 15 --gen-guidance 0.0 --gen-size 512 --seed "${SEEDV}" --device cuda:0 \
      --layer-report "${OUT_ROOT}/arms/${ARM}/layer_report.json" \
      > "${LOG}/${ARM}_gen.log" 2>&1; then
    log "WARN ${ARM}: generation FAILED"; tail -n 12 "${LOG}/${ARM}_gen.log"; return 1
  fi
  return 0
}

eval_one() {
  # $1 arm  $2 gen-dir
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
  return 0
}

run_arm() {
  # $1 arm  $2 cond-keys  $3 scale-json  $4 blur  $5 strength  $6 seed
  local ARM="$1" GEN="${OUT_ROOT}/arms/$1/gen/${STAG}"
  mkdir -p "${GEN}"
  gen_one "$1" "$2" "$3" "$4" "$5" "$6" "${GEN}" || return 1
  eval_one "${ARM}" "${GEN}"
}

DONE=()

# Reference arms are REUSED, not regenerated -- these are the same operator at the same
# anchor, already on disk with evaluations:
#   E1_sem_s30     SEM only, cc1, sigma 3.0        -> the semantic-only floor
#   E2_varest_s30  SEM + varest linear, cc3, 3.0   -> the operating point (incep 0.7363)
#   E2_seed43/44/45                                -> seeds for the M6 pool
# Only genuinely new arms get GPU time below.

# --- [3a] anchor sigma sweep UP (the untested direction) --------------------
hr; log "sigma sweep upward: 4.0 / 5.0 / 6.0 / 8.0  (sigma<3 only ever hurt SSIM)"
log "  structure = variance-restored linear (the known-good deployable bank)"
for s in 4.0 5.0 6.0 8.0; do
  A="F_s${s/./}"
  hr; log "arm ${A}: SIGMA UP, blur ${s}"
  run_arm "${A}" "SEM,DEPL,EDGEL" cc3 "${s}" 0.92 "${SEED}" && DONE+=("${A}")
done

# --- [3b] the trained structural head, measured by generation ---------------
# One crossed variable against E2_varest_s30 (identical layout, structure branch, anchor).
# If G3 does not beat E2, the head has not recovered what the linear map misses, and the
# structure branch can be closed for good rather than tuned again.
if [[ "${HAVE_HEAD}" == "1" ]]; then
  hr; log "trained head arms (crossed against E2_varest_s30, same layout, same anchor)"
  for s in 3.0 5.0; do
    A="G3_head_s${s/./}"
    hr; log "arm ${A}: SEM + TRAINED HEAD structure, blur ${s}"
    run_arm "${A}" "SEM,DEPH,EDGEH" cc3 "${s}" 0.92 "${SEED}" && DONE+=("${A}")
  done
else
  hr; log "no head banks produced - skipping the head arms"
fi

# --- [3c] one extra seed to fill the M6 pool, and evaluations for the seeds -----
hr; log "M6 pool: seed 46 (43/44/45 already on disk), plus evaluations for all seeds"
GEN="${OUT_ROOT}/arms/E2_seed46/gen/${STAG}"; mkdir -p "${GEN}"
gen_one "E2_seed46" "SEM,DEPL,EDGEL" cc3 3.0 0.92 46 "${GEN}" || log "WARN E2_seed46 failed"
for s in 43 44 45 46; do
  A="E2_seed${s}"; GD="${OUT_ROOT}/arms/${A}/gen/${STAG}"
  [[ -f "${GD}/generated/199.png" ]] && { eval_one "${A}" "${GD}" && DONE+=("${A}"); }
done

# --- [4] M6 over the heterogeneous pool -------------------------------------
log "===== [4] M6: heterogeneous multi-candidate selection, real (alpha,beta) sweep ====="
M6_POOL=""
for A in E2_varest_s30 E2_seed43 E2_seed44 E2_seed45 E2_seed46 \
         F_s40 F_s50 F_s60 F_s80 G3_head_s30 G3_head_s50; do
  GD="${OUT_ROOT}/arms/${A}/gen/${STAG}"
  [[ -f "${GD}/generated/199.png" ]] && M6_POOL="${M6_POOL:+${M6_POOL},}${GD}"
done
log "  pool: $(echo "${M6_POOL}" | tr ',' '\n' | wc -l) candidate sets"
log "  criteria: alpha * cos(CLIP(cand), sem_cond)  -  beta * L1(lowpass(cand), anchor)"
log "  no GT, no class names; the anchor is our own predicted low-level image"

M6SC="${OUT_ROOT}/m6_scores_nwv5.npz"
SWEEP="1.0:0.0 1.0:0.25 1.0:0.5 1.0:1.0 1.0:2.0 0.5:1.0 0.0:1.0 0.0:2.0"
FIRST=1
for ab in ${SWEEP}; do
  A="${ab%%:*}"; B="${ab##*:}"
  NAME="M6v5_a${A/./}b${B/./}"
  ODIR="${OUT_ROOT}/arms/${NAME}/gen/${STAG}"
  mkdir -p "${ODIR}"
  hr; log "M6 ${NAME}: alpha=${A} beta=${B}"
  # only the first call pays for encoding; the rest read the cached score matrices
  if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/nw6_m6_select.py" \
      --cand-dirs "${M6_POOL}" --sem-cond "${SEM_COND}" --anchor-dir "${INIT_RGB}" \
      --scores-npz "${M6SC}" --standardize --alpha "${A}" --beta "${B}" \
      --out-dir "${ODIR}" --report "${OUT_ROOT}/arms/${NAME}/m6_report.json" \
      --device cuda:0 > "${LOG}/${NAME}_sel.log" 2>&1; then
    log "WARN ${NAME}: selection FAILED"; tail -n 12 "${LOG}/${NAME}_sel.log"; continue
  fi
  grep -E "^\[m6\] (loaded|across|alpha|standardized|WARN)" "${LOG}/${NAME}_sel.log" | head -4
  eval_one "${NAME}" "${ODIR}" && DONE+=("${NAME}")
  FIRST=0
done

# ---------------------------------------------------------------------------
log "===== [5] 2-way (Pearson + cosine) re-scoring ====="
if [[ "${#DONE[@]}" -gt 0 ]]; then
  "${PYTHON}" "${NB_ROOT}/scripts/nda/nw4_official_twoway.py" \
    --gen-root "${OUT_ROOT}/arms" --arms "$(IFS=,; echo "${DONE[*]}")" \
    --subjects "${SUBJ}" --images-root "${IMAGES_ROOT}" \
    --out "${OUT_ROOT}/official_twoway_nwv5.json" --device cuda:0 \
    > "${LOG}/official_twoway_nwv5.log" 2>&1 || log "WARN twoway failed"
  tail -n 14 "${LOG}/official_twoway_nwv5.log" 2>/dev/null
else
  log "WARN no arm finished - skipping twoway"
fi

# ---------------------------------------------------------------------------
log "===== [6] the table, and the verdict ====="
"${PYTHON}" - "${OUT_ROOT}" "${STAG}" "${HEAD_REP}" <<'PY'
import json, sys
from pathlib import Path
root, stag, head_rep = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
KEYS = ["pixcorr", "ssim", "inception", "clip", "alex2", "alex5", "swav", "fid"]
LBL = {
    "E1_sem_s30":   "E1  SEM only, no structure, cc1, sigma 3.0   (the floor)",
    "E2_varest_s30":"E2  + varest linear, cc3, sigma 3.0         (operating point)",
    "F_s40":        "F   + varest linear, sigma 4.0",
    "F_s50":        "F   + varest linear, sigma 5.0",
    "F_s60":        "F   + varest linear, sigma 6.0",
    "F_s80":        "F   + varest linear, sigma 8.0",
    "G3_head_s30":  "G3  + TRAINED HEAD structure, sigma 3.0",
    "G3_head_s50":  "G3  + TRAINED HEAD structure, sigma 5.0",
    "M6v5_a10b00":  "M6v5 selection, 11-cand pool: alpha 1.0 beta 0.0",
    "M6v5_a10b025": "M6v5 selection, 11-cand pool: alpha 1.0 beta 0.25",
    "M6v5_a10b05":  "M6v5 selection, 11-cand pool: alpha 1.0 beta 0.5",
    "M6v5_a10b10":  "M6v5 selection, 11-cand pool: alpha 1.0 beta 1.0",
    "M6v5_a10b20":  "M6v5 selection, 11-cand pool: alpha 1.0 beta 2.0",
    "M6v5_a05b10":  "M6v5 selection, 11-cand pool: alpha 0.5 beta 1.0",
    "M6v5_a00b10":  "M6v5 selection, 11-cand pool: alpha 0.0 beta 1.0",
    "M6v5_a00b20":  "M6v5 selection, 11-cand pool: alpha 0.0 beta 2.0",
    "M6_a10b00":    "M6 selection, OLD 4-cand pool: alpha 1.0 beta 0.0",
    "M6_a10b05":    "M6 selection, OLD 4-cand pool: alpha 1.0 beta 0.5",
    "M6_a10b10":    "M6 selection, OLD 4-cand pool: alpha 1.0 beta 1.0",
    "M6_a10b20":    "M6 selection, OLD 4-cand pool: alpha 1.0 beta 2.0",
    "M6_a00b10":    "M6 selection, OLD 4-cand pool: alpha 0.0 beta 1.0",
    "M6_sweep":     "M6 sweep record (OLD pool)",
    "A5_band_cc3":  "A5  GT structure, sigma 3.0          (the target)",
    "A9_orc_cc3":   "A9  all GT                           (mechanism ceiling)",
    "A1_pure_cc1":  "A1  SEM only, no anchor",
    "E2_seed43":    "seed 43 (M6 candidate)",
    "E2_seed44":    "seed 44 (M6 candidate)",
    "E2_seed45":    "seed 45 (M6 candidate)",
    "E2_seed46":    "seed 46 (M6 candidate)",
}
rows = {}
for f in sorted((root/"arms").glob(f"*/eval/{stag}.json")):
    rows[f.parent.parent.name] = json.load(open(f))
ref = Path("/project/peilab/why/NeuroBridge/outputs/nw4_10s/arms/a_hi/eval/sub-08.json")
if ref.is_file(): rows["a_hi(shipped)"] = json.load(open(ref))

print()
print("="*128)
print(f"NW-v5 {stag}: cc3 + band anchor + no ControlNet + best structure + M6 selection")
print("="*128)
hdr = f"{'arm':<50}" + "".join(f"{k:>9}" for k in KEYS)
print(hdr); print("-"*len(hdr))
order = [k for k in LBL if k in rows] + [k for k in sorted(rows) if k not in LBL]
for a in order:
    r = rows[a]
    print(f"{LBL.get(a, a[:48]):<50}" + "".join(
        f"{r[k]:>9.4f}" if k != "fid" else f"{r[k]:>9.2f}" for k in KEYS))

print()
print("="*128)
print("published bars (10-subject means unless noted)")
print("="*128)
BARS = {
    "CogCapPro":     {"pixcorr": 0.163, "ssim": 0.398, "inception": 0.779, "clip": 0.830},
    "CogCapPro-s08": {"pixcorr": 0.166, "ssim": 0.409, "inception": 0.831, "clip": 0.903,
                      "alex2": 0.818, "alex5": 0.913},
    "D2-FOSA":       {"pixcorr": 0.193, "ssim": 0.350},
    "MB2C":          {"pixcorr": 0.188, "ssim": 0.333},
    "ENIGMA":        {"pixcorr": 0.167, "ssim": 0.426, "inception": 0.765, "clip": 0.803,
                      "alex2": 0.830, "alex5": 0.891},
    "ATM":           {"pixcorr": 0.160, "ssim": 0.345, "inception": 0.734, "clip": 0.786,
                      "alex2": 0.776, "alex5": 0.866},
}
for name, bar in BARS.items():
    print(f"  {name:<15}" + "  ".join(f"{k}={bar[k]}" for k in KEYS if k in bar))

print()
print("="*128)
print("BEST FINISHED ARM PER AXIS")
print("="*128)
for k in KEYS:
    cand = [(v[k], a) for a, v in rows.items() if k in v]
    if not cand: continue
    val, a = (min(cand) if k in ("swav", "fid") else max(cand))
    print(f"  {k:<10} {val:>9.4f}   {a}")

print()
print("="*128)
print("VERDICT")
print("="*128)
if head_rep.is_file():
    hd = json.load(open(head_rep))
    print("  S1 structural head, regularisation grid (ranked by val dev_corr, "
          "held-out CONCEPTS):")
    for m, grid in hd.get("grid", {}).items():
        linv = hd["linear"][m]["val"]
        print(f"    -- {m}: linear ridge val dev_corr {linv['dev_corr']:.4f}  "
              f"val dev_top1 {linv['dev_top1']*200:.1f}x   <-- the bar to beat")
        for r in grid:
            mark = " ***" if (r["config"] == hd["head"].get(m, {}).get("config")
                              and r["variant"] == hd["head"].get(m, {}).get("variant")) else ""
            print(f"       {r['config']:<16} {r['variant']:<8} "
                  f"val_corr {r['val_dev_corr']:.4f} ({r['val_dev_corr']/linv['dev_corr']:.2f}x)"
                  f"  val_top1 {r['val_dev_top1']*200:>5.1f}x  ep {r['epoch']:>3}  "
                  f"{r['params']/1e3:>5.0f}k{mark}")
    print("    (the grid matters when everything fails: if a rank bottleneck plus a small")
    print("     trunk is the best config, the EEG's structure information is low-rank and")
    print("     a large network only memorises trials -- that is a finding, not a bug)")
    print()
    print("  tiers (prediction only; the generation numbers below decide):")
    for m, g in hd.get("gate", {}).items():
        h = hd["head"][m]
        print(f"    {m:<6} {g['tier'].upper():<7} [{h['config']}/{h['variant']}] "
              f"val_corr x{g['val_gain_corr']:.2f}  val_top1 x{g['val_gain_top1']:.2f}  "
              f"test dev_corr {h['test']['dev_corr']:.4f} vs linear "
              f"{h['linear_test']['dev_corr']:.4f}")
    print()
    print("  Did the head help GENERATION?  (one crossed variable vs E2_varest_s30)")
    gl, gh = rows.get("E2_varest_s30"), rows.get("G3_head_s30")
    if gl and gh:
        d = {k: gh[k] - gl[k] for k in KEYS if k in gl and k in gh}
        print("    G3_head_s30 - E2_varest_s30:  "
              + "  ".join(f"{k} {d[k]:+.4f}" for k in
                          ("pixcorr", "ssim", "inception", "clip", "alex2", "alex5")))
        helped = (d.get("inception", 0) > 0.002) or (d.get("clip", 0) > 0.002)
        if helped:
            print("    => the head DID buy generation metrics the linear map could not.")
            print("       That is the first evidence of structure information above the")
            print("       linear ceiling; scale the head next (subject count, capacity).")
        else:
            print("    => the head did NOT buy generation metrics.  Its dev_top1 gain was")
            print("       real in embedding space and worthless in pixels, which is the")
            print("       same signature as the shrunk-ridge result.  Record this and close")
            print("       the structure branch: the semantic branch (A9 - A5 = +0.141 incep)")
            print("       is where the remaining headroom is.")
    else:
        print("    (E2_varest_s30 and/or G3_head_s30 missing - no comparison)")
    if not hd.get("any_passed"):
        print("    note: no config reached the 'rank' tier, i.e. the head neither clearly")
        print("          beat linear on dev_corr nor clearly on ranking.  The grid table")
        print("          above says whether that is overfitting (large configs win) or a")
        print("          low-rank ceiling (the bottleneck configs win and still fall short).")

def g(a, k): return rows[a][k] if a in rows and k in rows[a] else None
a5 = g("A5_band_cc3", "inception")
base = g("E1_sem_s30", "inception")
if a5 and base:
    print()
    print(f"  distance to the target: A5 (GT structure) incep {a5:.4f}, "
          f"semantic-only floor {base:.4f}, gap {a5-base:+.4f}")
    for a in ("E2_varest_s30", "G3_head_s30", "G3_head_s50",
              "F_s40", "F_s50", "F_s60", "F_s80"):
        v = g(a, "inception")
        if v is not None:
            print(f"    {a:<18} incep {v:.4f}   recovers "
                  f"{(v-base)/(a5-base):>6.1%} of the GT-structure gain")
    bestm6 = None
    for a in sorted(rows):
        if not a.startswith("M6v5_"): continue
        v = g(a, "inception")
        if v is not None and (bestm6 is None or v > bestm6[0]): bestm6 = (v, a)
    if bestm6:
        print(f"    best M6 arm        incep {bestm6[0]:.4f}   ({bestm6[1]})")
    # the seven-axis check against the only strictly comparable single-subject bar
    print()
    print("  seven-axis check vs CogCapPro sub-08 (the only like-for-like published bar):")
    best = None
    for a in ("M6v5_a10b00", "M6v5_a10b05", "M6v5_a10b10", "M6v5_a10b20",
              "G3_head_s30", "E2_varest_s30"):
        if a not in rows: continue
        r = rows[a]
        wins = sum(1 for k, bar in (("pixcorr", 0.166), ("ssim", 0.409), ("inception", 0.831),
                                    ("clip", 0.903), ("alex2", 0.818), ("alex5", 0.913))
                   if (r[k] > bar))
        swav_ok = r["swav"] < 0.489
        tot = wins + (1 if swav_ok else 0)
        print(f"    {a:<20} {tot}/7 axes better than CogCapPro-s08  "
              f"(pixcorr {r['pixcorr']:.4f} ssim {r['ssim']:.4f} incep {r['inception']:.4f} "
              f"clip {r['clip']:.4f} swav {r['swav']:.4f})")
PY

log "===== done ====="
