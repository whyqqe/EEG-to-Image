#!/usr/bin/env bash
# =============================================================================
# NW-v4 sub-08: the configuration the evidence points at, run end to end.
#
# WHAT THIS JOB IS
# ----------------
# Not another probe.  It is the reportable sub-08 experiment for the architecture
# change that nw5/nw5b/nw5c converged on, with the SOTA bars printed beside it.
#
# THE OPERATOR  (from nw5: reading CogCapPro's public code, then measuring it)
# --------------------------------------------------------------------------
#   cc3 asymmetric IP-Adapter layout, empty prompt, SDXL-Turbo, 15 steps, CFG 0.0,
#   and NO ControlNet.  Structure is injected in FEATURE space, not as pixels.
#   The only pixel path kept is the low-frequency band anchor (L3): our predicted
#   low-level image, Gaussian-blurred to sigma, used as the img2img init.
#
# Measured on sub-08 with these same scorers (nw5_s08):
#
#   arm                                   pixcorr    ssim  incep    clip
#   a_hi  (what we have been shipping)     0.1684  0.3573  0.7106  0.8115
#   A6    +GT structure, still CN+init     0.1621  0.3541  0.8034  0.8889
#   A2    +GT structure, no pixel machine  0.1781  0.2810  0.8394  0.9094
#   A5    +GT structure, band anchor s=3   0.1980  0.3756  0.8400  0.9123
#   A9    all GT                           0.2122  0.2932  0.9811  0.9921
#
# A5 is Pareto-dominant over a_hi on every axis (+,+ on the V1 criterion the NW brief
# was waiting on), and over CogCapPro's published numbers on PixCorr/Incep/CLIP.  Two
# things in it are load-bearing and cheap:
#   * dropping ControlNet while keeping a LOW-FREQUENCY anchor is +0.095 SSIM over A2
#     at ZERO semantic cost (A2 -> A5: ssim 0.2810->0.3756, incep 0.8394->0.8400),
#   * A5 pins structure to one level at 0.5 (cc3).  The mass-matched uniform layout
#     (A3b) scores incep 0.7280 vs 0.8394, so the asymmetry is mechanical, not a
#     strength artefact.
#
# THE ONE THING A5 CANNOT DO IS DEPLOY: it uses GT depth/edge.  Our own structure path
# contributes +0.0000 (A8 incep 0.7077 vs A1 0.7116), so this job has to close that.
#
# WHY OUR STRUCTURE PATH FAILED, AND THE FIX  (nw5b + nw5c)
# --------------------------------------------------------
# nw5b established that the structure branch needs TRIAL-ALIGNED information, not a
# prior.  Replacing the GT rows with a constant centroid (C1), with the true bank's rows
# misaligned (C3), or with our honest linear EEG->structure map (C2) all landed at or
# BELOW the semantic-only floor A1 -- shuffled GT rows scored -0.0287 incep.  An
# unaligned structure condition is not neutral, it is harmful.
#
# nw5c found why C2 failed anyway.  Mean pairwise cosine across the 200 trials, where
# near 1.0 means the bank is effectively one vector:
#
#     GT depth / edge            0.566 / 0.530     <- what the branch has to carry
#     a_hi semantic              0.373 (target 0.391)  <- healthy
#     ridge structure            0.955 / 0.935     <- collapsed
#     UCK -> CLIP structure      0.874 / 0.938     <- collapsed
#
# Our semantic head is correctly scaled.  The structure banks vary six times less than
# the quantity they predict: least squares minimises squared error by predicting the
# mean, so a shared vector absorbed the energy and the trial-specific component -- the
# entire value of the branch -- was crushed.  Rescaling only the deviation from the bank
# mean, with identical weights and identical EEG, lifts top-1 retrieval against the GT
# bank from 3x/11x chance to 26x/35x chance.  That condition has never been generated
# from.
#
# THE ARMS  (all: empty prompt, turbo, 15 steps, CFG 0, no ControlNet, band anchor)
# -------------------------------------------------------------------------------
#  E1_sem_s30      SEM only, sigma 3.0                     -- deployable floor
#  E2_varest_s30   SEM + variance-restored ridge structure -- the A5 point, deployable
#  E3_uck_s30      SEM + variance-restored UCK structure   -- no new head at all
#  E4_varest_s20   sigma 2.0                               -- buy SSIM with the surplus
#  E5_varest_s15   sigma 1.5                                 we have in
#  E6_varest_s05   sigma 0.5, strength 0.95                  Incep/CLIP
#  M6_*            multi-candidate selection over 4 seeds of E2 at several (a,b)
#
# E3 is the arm that decides whether anything has to be trained: it is the pipeline we
# already ship, with only the shrinkage removed.  E1 pins the floor so every other arm is
# readable as a delta.  E4-E6 map the anchor tradeoff, which is the only lever that moves
# PixCorr/SSIM without a new mechanism.  M6 is the only lever that can move both axes at
# once: N seeds of one configuration, then per-trial selection by
#
#     score = alpha * cos(CLIP(candidate), sem_cond) - beta * L1(lowpass(candidate), anchor)
#
# using no GT and no class names.  Because the score matrices are cached, a sweep of
# (alpha, beta) traces a whole Pareto front from one set of generations.
#
# READ THE BARS: CogCapPro 10-subject means are pixcorr 0.163 ssim 0.398 incep 0.779
# clip 0.830; D2-FOSA pixcorr 0.193 ssim 0.350; ENIGMA incep 0.765 ssim 0.426.
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
log "===== [1] variance restoration of every structure bank (nw5c) ====="
log "  the scale is fixed from the GT TRAIN bank's variability only: no test rows, no labels"
"${PYTHON}" - "${CC}" "${CONDS}" "${STAG}" "${SUBJ}" <<'PY' || exit 1
import sys, json
from pathlib import Path
import numpy as np
cc, conds, stag, subj = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]

def l2n(x):
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

def rowcos(x, cap=2000):
    if len(x) > cap:
        x = x[np.linspace(0, len(x) - 1, cap).astype(int)]
    y = l2n(x); n = len(y); S = y @ y.T
    return float((S.sum() - np.trace(S)) / (n * (n - 1)))

def solve_a(P, m, tgt):
    lo, hi = 0.0, 1e7
    for _ in range(90):
        a = 0.5 * (lo + hi)
        if rowcos(m + (P - m) * a) > tgt:
            lo = a
        else:
            hi = a
    return 0.5 * (lo + hi)

rep = {"basis": "GT TRAIN banks only (leak-free)", "banks": {}}
tgt_rc = {lab: rowcos(np.load(cc / f"clip_{lab}1024_train.npy").astype(np.float32))
          for lab in ("depth", "edge")}
rep["gt_train_rowcos"] = tgt_rc

for lab in ("depth", "edge"):
    P = np.load(conds / f"ridge_{lab}1024_{stag}_test.npy").astype(np.float32)
    m = P.mean(0, keepdims=True)
    a = solve_a(P, m, tgt_rc[lab])
    for tag, scale in (("varest", a), ("demean", 1e7)):
        arr = l2n(m + (P - m) * scale)
        p = conds / f"{tag}ridge_{lab}1024_{stag}_test.npy"
        np.save(p, arr)
        rep["banks"][f"{tag}_ridge_{lab}"] = {"path": str(p),
                                              "scale": float(scale) if scale < 1e6 else "de-mean",
                                              "rowcos_in": rowcos(P), "rowcos_out": rowcos(arr)}
    U = np.load(conds / f"eeg_{lab}1024_{stag}_test.npy").astype(np.float32)
    mu = U.mean(0, keepdims=True)
    au = solve_a(U, mu, tgt_rc[lab])
    arr = l2n(mu + (U - mu) * au)
    p = conds / f"varest_uck_{lab}1024_{stag}_test.npy"
    np.save(p, arr)
    rep["banks"][f"varest_uck_{lab}"] = {"path": str(p), "scale": float(au),
                                         "rowcos_in": rowcos(U), "rowcos_out": rowcos(arr)}

img_tr = np.load(cc / "clip_img1024_train.npy").astype(np.float32)
sem = np.load(Path("/project/peilab/why/NeuroBridge/outputs/nw4_10s/arms/a_hi/conds")
              / stag / "cal_test.npy").astype(np.float32)
ms = sem.mean(0, keepdims=True)
a_sem = solve_a(sem, ms, rowcos(img_tr))
arr = l2n(ms + (sem - ms) * a_sem)
p = conds / f"varest_sem_{stag}_test.npy"
np.save(p, arr)
rep["banks"]["varest_sem"] = {"path": str(p), "scale": float(a_sem),
                              "rowcos_in": rowcos(sem), "rowcos_out": rowcos(arr),
                              "gt_train_rowcos": rowcos(img_tr)}
(conds / "varest_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
for k, v in rep["banks"].items():
    sc = v["scale"]
    scs = f"x{sc:.2f}" if isinstance(sc, float) else str(sc)
    print(f"  {k:<22} rowcos {v['rowcos_in']:.4f} -> {v['rowcos_out']:.4f}  ({scs})")
PY

# ---------------------------------------------------------------------------
log "===== [2] what each bank carries after restoration ====="
"${PYTHON}" - "${CC}" "${CONDS}" "${STAG}" "${SUBJ}" <<'PY'
import sys
from pathlib import Path
import numpy as np
cc, conds, stag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
def l2n(x): return (x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)).astype(np.float32)
def rc(x):
    y=l2n(x); n=len(y); S=y@y.T
    return float((S.sum()-np.trace(S))/(n*(n-1)))
def diag(p,t): return float((l2n(p)*l2n(t)).sum(1).mean())
def top1(p,t): return float(((l2n(p)@l2n(t).T).argmax(1)==np.arange(len(t))).mean())
gt = {l: l2n(np.load(cc/f"clip_{l}1024_test.npy").astype(np.float32)) for l in ("depth","edge")}
rows = [("ridge shrunk (C2)", f"ridge_depth1024_{stag}_test.npy", "depth"),
        ("ridge varest (E2)",  f"varestridge_depth1024_{stag}_test.npy", "depth"),
        ("UCK shrunk (A8)",    f"eeg_depth1024_{stag}_test.npy", "depth"),
        ("UCK varest (E3)",    f"varest_uck_depth1024_{stag}_test.npy", "depth"),
        ("ridge shrunk (C2)",  f"ridge_edge1024_{stag}_test.npy", "edge"),
        ("ridge varest (E2)",  f"varestridge_edge1024_{stag}_test.npy", "edge"),
        ("UCK shrunk (A8)",    f"eeg_edge1024_{stag}_test.npy", "edge"),
        ("UCK varest (E3)",    f"varest_uck_edge1024_{stag}_test.npy", "edge")]
print(f"  {'bank':<22}{'rowcos':>9}{'diag':>9}{'top1':>8}{'xchance':>9}")
print("  "+"-"*57)
for name, fn, lab in rows:
    p = conds/fn
    if not p.is_file():
        print(f"  {name:<22}  MISSING {fn}"); continue
    x = np.load(p).astype(np.float32); t = gt[lab]
    print(f"  {name:<22}{rc(x):>9.4f}{diag(x,t):>9.4f}{top1(x,t):>8.4f}{top1(x,t)*200:>9.1f}")
PY

cond_path() {
  case "$1" in
    SEM)   echo "${SEM_COND}" ;;
    DEPV)  echo "${CONDS}/varestridge_depth1024_${STAG}_test.npy" ;;
    EDGEV) echo "${CONDS}/varestridge_edge1024_${STAG}_test.npy" ;;
    DEPU)  echo "${CONDS}/varest_uck_depth1024_${STAG}_test.npy" ;;
    EDGEU) echo "${CONDS}/varest_uck_edge1024_${STAG}_test.npy" ;;
    *)     echo "" ;;
  esac
}

# ---------------------------------------------------------------------------
log "===== [3] generate the main arms ====="
# name | cond keys | scale json | pipeline | steps | guidance | use_cn | use_init |
# blur sigma | strength | cn_scale | prompts
ARMS=(
  "E1_sem_s30|SEM|cc1|turbo|15|0.0|0|1|3.0|0.92|0.28|empty"
  "E2_varest_s30|SEM,DEPV,EDGEV|cc3|turbo|15|0.0|0|1|3.0|0.92|0.28|empty"
  "E3_uck_s30|SEM,DEPU,EDGEU|cc3|turbo|15|0.0|0|1|3.0|0.92|0.28|empty"
  "E4_varest_s20|SEM,DEPV,EDGEV|cc3|turbo|15|0.0|0|1|2.0|0.92|0.28|empty"
  "E5_varest_s15|SEM,DEPV,EDGEV|cc3|turbo|15|0.0|0|1|1.5|0.92|0.28|empty"
  "E6_varest_s05|SEM,DEPV,EDGEV|cc3|turbo|15|0.0|0|1|0.5|0.95|0.28|empty"
)

gen_one() {
  # $1 arm  $2 cond-keys  $3 scale-json  $4 blur  $5 strength  $6 seed  $7 outdir
  local ARM="$1" CKEYS="$2" SJSON="$3" BLUR="$4" STRENGTH="$5" SEEDV="$6" GEN="$7"
  local CLIST="" N=0 p
  for k in ${CKEYS//,/ }; do
    p="$(cond_path "${k}")"
    if [[ -z "${p}" || ! -f "${p}" ]]; then
      log "WARN ${ARM}: condition ${k} unavailable (${p})"; return 1
    fi
    CLIST="${CLIST:+${CLIST},}${p}"; N=$((N+1))
  done
  [[ -f "${GEN}/generated/199.png" ]] && { log "--- ${ARM}: generation present - skip"; return 0; }
  if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/generate_layered_decode.py" \
      --cond-npys "${CLIST}" --ip-scale-json "${SPECS}/${SJSON}.json" \
      --prompts-json "${EMPTY_PROMPTS}" --output-dir "${GEN}" --tag "${ARM}_${STAG}" \
      --pipeline turbo --use-cn 0 --use-init 1 \
      --lowlevel-rgb-dir "${INIT_RGB}" \
      --cn-scale 0.28 --strength "${STRENGTH}" --init-blur-sigma "${BLUR}" \
      --gen-steps 15 --gen-guidance 0.0 \
      --gen-size 512 --seed "${SEEDV}" --device cuda:0 \
      --layer-report "${OUT_ROOT}/arms/${ARM}/layer_report.json" \
      > "${LOG}/${ARM}_gen.log" 2>&1; then
    log "WARN ${ARM}: generation FAILED"; tail -n 15 "${LOG}/${ARM}_gen.log"; return 1
  fi
  grep -E "^\[layout|^\[init-only|^\[pure|^\[INFO\] pipeline" "${LOG}/${ARM}_gen.log" | head -3
  return 0
}

eval_one() {
  # $1 arm  $2 gen-dir
  local ARM="$1" GEN="$2"
  local EV="${OUT_ROOT}/arms/${ARM}/eval/${STAG}.json"
  mkdir -p "${OUT_ROOT}/arms/${ARM}/eval"
  if [[ -f "${EV}" ]]; then log "--- ${ARM}: eval present - skip"; return 0; fi
  if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
      --gen-dir "${GEN}/generated" --output-json "${EV}" --tag "${ARM}_${STAG}" \
      --images-root "${IMAGES_ROOT}" --device cuda:0 \
      > "${LOG}/${ARM}_eval.log" 2>&1; then
    log "WARN ${ARM}: eval FAILED"; tail -n 15 "${LOG}/${ARM}_eval.log"; return 1
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

DONE=()
for spec_line in "${ARMS[@]}"; do
  IFS='|' read -r ARM CKEYS SJSON PIPE STEPS GUID UCN UINIT BLUR STRENGTH CNSC PROMPTS EXTRA <<< "${spec_line}"
  if [[ -n "${EXTRA:-}" || -z "${ARM}" || -z "${PROMPTS}" ]]; then
    echo "[FATAL] arm definition is not 12 fields: '${spec_line}'" >&2; exit 1
  fi
  hr; log "arm ${ARM}: keys=${CKEYS} scales=${SJSON} turbo 15step g=0 cn=0 init=1 blur=${BLUR} str=${STRENGTH}"
  GEN="${OUT_ROOT}/arms/${ARM}/gen/${STAG}"
  mkdir -p "${GEN}"
  gen_one "${ARM}" "${CKEYS}" "${SJSON}" "${BLUR}" "${STRENGTH}" "${SEED}" "${GEN}" || continue
  eval_one "${ARM}" "${GEN}" && DONE+=("${ARM}")
done

# ---------------------------------------------------------------------------
log "===== [4] M6: extra candidate seeds for the A5-point configuration ====="
M6_BASE_CKEYS="SEM,DEPV,EDGEV"
M6_BASE_SJSON="cc3"
M6_DIRS="${OUT_ROOT}/arms/E2_varest_s30/gen/${STAG}"
for s in 43 44 45; do
  A="E2_seed${s}"
  GEN="${OUT_ROOT}/arms/${A}/gen/${STAG}"
  mkdir -p "${GEN}"
  hr; log "M6 candidate ${A} (seed ${s})"
  if gen_one "${A}" "${M6_BASE_CKEYS}" "${M6_BASE_SJSON}" 3.0 0.92 "${s}" "${GEN}"; then
    M6_DIRS="${M6_DIRS},${GEN}"
  else
    log "WARN ${A}: candidate failed - continuing with the seeds that worked"
  fi
done

log "===== [5] M6 selection sweep (no GT, no class names) ====="
M6SC="${OUT_ROOT}/m6_scores.npz"
mkdir -p "${OUT_ROOT}/arms/M6_sweep"
# selection is GPU-free after the first call caches the score matrices, so a sweep of
# the tradeoff weight costs nothing
SWEEP=("1.0:0.0" "1.0:0.5" "1.0:1.0" "1.0:2.0" "0.0:1.0")
for ab in "${SWEEP[@]}"; do
  A="${ab%%:*}"; B="${ab##*:}"
  NAME="M6_a${A/./}b${B/./}"
  ODIR="${OUT_ROOT}/arms/${NAME}/gen/${STAG}"
  mkdir -p "${ODIR}"
  log "M6 ${NAME}: alpha=${A} beta=${B}"
  if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/nw6_m6_select.py" \
      --cand-dirs "${M6_DIRS}" --sem-cond "${SEM_COND}" --anchor-dir "${INIT_RGB}" \
      --scores-npz "${M6SC}" --standardize --alpha "${A}" --beta "${B}" \
      --out-dir "${ODIR}" --report "${OUT_ROOT}/arms/${NAME}/m6_report.json" \
      --device cuda:0 > "${LOG}/${NAME}_sel.log" 2>&1; then
    log "WARN ${NAME}: selection FAILED"; tail -n 15 "${LOG}/${NAME}_sel.log"; continue
  fi
  grep -E "^\[m6\] (across|alpha|standardized|WARN)" "${LOG}/${NAME}_sel.log" | head -4
  eval_one "${NAME}" "${ODIR}" && DONE+=("${NAME}")
done

[[ "${#DONE[@]}" -eq 0 ]] && { log "[FATAL] no arm finished"; exit 1; }
log "finished arms: ${DONE[*]}"

# ---------------------------------------------------------------------------
log "===== [6] 2-way (Pearson + cosine) re-scoring ====="
"${PYTHON}" "${NB_ROOT}/scripts/nda/nw4_official_twoway.py" \
  --gen-root "${OUT_ROOT}/arms" --arms "$(IFS=,; echo "${DONE[*]}")" \
  --subjects "${SUBJ}" --images-root "${IMAGES_ROOT}" \
  --out "${OUT_ROOT}/official_twoway_nwv4.json" --device cuda:0 \
  > "${LOG}/official_twoway_nwv4.log" 2>&1 || { log "WARN twoway failed"; tail -n 15 "${LOG}/official_twoway_nwv4.log"; }
tail -n 14 "${LOG}/official_twoway_nwv4.log" 2>/dev/null

# ---------------------------------------------------------------------------
log "===== [7] the table, against every published bar ====="
"${PYTHON}" - "${OUT_ROOT}" "${STAG}" <<'PY'
import json, sys
from pathlib import Path
root, stag = Path(sys.argv[1]), sys.argv[2]
KEYS = ["pixcorr", "ssim", "inception", "clip", "alex2", "alex5", "swav", "fid"]
LBL = {
    "E1_sem_s30":    "E1  SEM only, anchor s=3.0        (floor)",
    "E2_varest_s30": "E2  + varest ridge, s=3.0        (the A5 point)",
    "E3_uck_s30":    "E3  + varest UCK->CLIP, s=3.0     (no new head)",
    "E4_varest_s20": "E4  + varest ridge, s=2.0",
    "E5_varest_s15": "E5  + varest ridge, s=1.5",
    "E6_varest_s05": "E6  + varest ridge, s=0.5 str.95",
}
rows = {}
for f in sorted((root/"arms").glob(f"*/eval/{stag}.json")):
    rows[f.parent.parent.name] = json.load(open(f))
ref = Path("/project/peilab/why/NeuroBridge/outputs/nw4_10s/arms/a_hi/eval/sub-08.json")
if ref.is_file(): rows["a_hi(shipped)"] = json.load(open(ref))
a5 = root/"arms"/"A5_band_cc3"/"eval"/f"{stag}.json"
if a5.is_file(): rows["A5_band_cc3(GT structure)"] = json.load(open(a5))

print()
print("="*126)
print(f"NW-v4 {stag}: the CogCapPro operator + our frequency-band anchor + variance-restored "
      f"structure, deployable")
print("="*126)
hdr = f"{'arm':<46}" + "".join(f"{k:>9}" for k in KEYS)
print(hdr); print("-"*len(hdr))
order = [k for k in LBL if k in rows] + [k for k in rows if k not in LBL]
for a in order:
    r = rows[a]
    lbl = LBL.get(a, a[:44])
    print(f"{lbl:<46}" + "".join(f"{r[k]:>9.4f}" if k != "fid" else f"{r[k]:>9.2f}" for k in KEYS))

print()
print("="*126)
print("published bars (all 10-subject means unless noted)")
print("="*126)
BARS = {
    "CogCapPro":   {"pixcorr": 0.163, "ssim": 0.398, "inception": 0.779, "clip": 0.830},
    "ATM":         {"pixcorr": 0.160, "ssim": 0.345, "alex2": 0.776, "alex5": 0.866,
                    "inception": 0.734, "clip": 0.786},
    "D2-FOSA":     {"pixcorr": 0.193, "ssim": 0.350},
    "MB2C":        {"pixcorr": 0.188, "ssim": 0.333},
    "ENIGMA":      {"pixcorr": 0.167, "ssim": 0.426, "alex2": 0.830, "alex5": 0.891,
                    "inception": 0.765, "clip": 0.803},
    "CogCapPro-s8": {"pixcorr": 0.166, "ssim": 0.409, "alex2": 0.818, "alex5": 0.913,
                     "inception": 0.831, "clip": 0.903},
}
for name, bar in BARS.items():
    print(f"  {name:<14}" + "  ".join(f"{k}={bar[k]}" for k in KEYS if k in bar))

# best per axis among finished arms
print()
print("="*126)
print("BEST FINISHED ARM PER AXIS")
print("="*126)
for k in KEYS:
    best = max(((v[k], a) for a, v in rows.items() if k in v),
               default=None)
    if not best: continue
    val, a = best
    print(f"  {k:<10} {val:>9.4f}   {a}")
PY

log "===== done ====="
