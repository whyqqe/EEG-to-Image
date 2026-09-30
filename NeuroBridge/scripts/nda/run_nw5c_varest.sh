#!/usr/bin/env bash
# =============================================================================
# NW5c sub-08: our structural condition is SHRUNK.  Restore its variance.
#
# THE DIAGNOSIS THAT MAKES THIS JOB NECESSARY
# -------------------------------------------
# nw5b settled one question decisively and raised a sharper one.
#
# Settled: the structure branch is TRIAL INFORMATION, not a prior.  Feeding the branch
# a constant centroid (C1), the true bank's rows misaligned (C3), or our honest linear
# EEG->structure map (C2) all landed at or BELOW the semantic-only floor A1:
#
#     A1  SEM only              incep 0.7116   clip 0.8245
#     A2  SEM + GT depth/edge   incep 0.8394   clip 0.9094   (+0.1278 / +0.0849)
#     C3  SEM + GT shuffled     incep 0.6829   clip 0.8271   ( -0.0287 / +0.0026)
#     C1  SEM + centroid        incep 0.6990   clip 0.8184   ( -0.0126 / -0.0062)
#     C2  SEM + ridge           incep 0.7082   clip 0.8307   ( -0.0035 / +0.0062)
#     A8  SEM + our EEG depth   incep 0.7077   clip 0.8244   ( -0.0039 / -0.0001)
#
# Shuffling GT rows RECOVERS NOTHING, so per-trial alignment is load-bearing.  A
# structure condition that is not aligned to this trial is not merely useless: it is
# worse than asking for no structure at all.
#
# Raised: then why did C2's leak-free ridge fail too, when its retrieval accuracy is
# real?  Look at how much trial-to-trial VARIATION each bank actually carries:
#
#     bank                       rowcos   (mean pairwise cosine across the 200 trials;
#                                         near 1.0 == all 200 rows are one vector)
#     GT depth / edge            0.566 / 0.530
#     GT image (semantic)        0.391
#     a_hi  (our shipped SEM)    0.373      <-- healthy, matches its target
#     ridge depth / edge         0.955 / 0.935
#     our UCK->CLIP depth/edge   0.874 / 0.938
#     C1 centroid                1.000
#
# Our semantic head is FINE (0.373 vs the 0.391 its target bank has).  Structure is not:
# the ridge output varies 6x less across trials than the thing it is predicting.  This is
# textbook regression-to-the-mean -- squared error is minimised by predicting the mean,
# so nearly all of the bank's energy went into one shared vector and the trial-specific
# component that A2 proves is the whole value of the branch got crushed.
#
# And restoring it is not a cosmetic fix.  Same ridge weights, same test EEG, only the
# deviation from the bank mean is rescaled:
#
#     variant                        top1 (chance 0.005)
#     ridge as emitted     depth  0.0150 ( 3x)   edge  0.0550 (11x)
#     a = 1.5 (GT-matched) depth  0.1300 (26x)   edge  0.1750 (35x)
#     fully de-meaned      depth  0.1500 (30x)   edge  0.2200 (44x)
#
# A 26x/35x retrievable structure condition has never been fed to the generator.  A8 and
# C2 were handed 3x/11x.  This job feeds the restored versions.
#
# THE ARMS  (cc3 layout, SDXL-Turbo 15 steps CFG 0, no CN, no init, empty prompt --
#            identical to A1/A2/C1/C2/C3, only the banks differ)
# -----------------------------------------------------------------------------------
#  D1_varest   SEM + variance-restored ridge depth/edge.  Deviation rescaled so the bank
#              carries the SAME trial-to-trial variability as the GT bank it predicts.
#              The scale is fixed from the GT *train* bank only -- no test statistics, no
#              labels, nothing leaky.  This is C2 with its shrinkage removed.
#  D2_demean   SEM + fully de-meaned ridge depth/edge.  The a -> inf limit: maximum trial
#              contrast, but each row is now far from the CLIP manifold (diag 0.24 vs the
#              0.75 a constant scores).  Upper bound on how much contrast IP-Adapter can
#              use before the condition stops looking like a structure at all.
#  D3_varest_uck  SEM + variance-restored UCK->CLIP depth/edge -- the SHIPPED path
#              (A8's banks, rowcos 0.874/0.938), restored the same way.  This is the
#              deployable question: if A8's failure was only shrinkage, this arm fixes
#              the production pipeline without training anything new.
#  D4_varest_sem  fully variance-restored SEM + variance-restored ridge depth/edge.
#              a_hi is already correctly scaled, but a semantic bank's trial contrast is
#              the one thing we have never deliberately maximised.  Separates "fix the
#              structure branch" from "fix the structure branch AND sharpen semantics".
#
# READ IT AGAINST:  A1 (floor, 0.7116)  C2 (shrunken ridge, 0.7082)
#                   A5 (GT structure + band anchor, 0.8400)  <-- the target to approach
#                   A9 (all-GT, 0.9811)                      <-- the mechanism's ceiling
#
#   D1 ~= C2        -> shrinkage was not the problem; a trained structure head is required
#   D1 >  C2 > A1   -> shrinkage WAS the problem and a variance-regularised structure head
#                      is the deliverable; D1 is then the reportable architecture
#   D1 ~= A5        -> a linear EEG->structure map plus variance restoration recovers the
#                      entire GT-structure gain: no trained structure head needed at all
#   D3 ~= D1        -> the UCK path is already good enough once rescaled: ship D3
#   D2 << D1        -> there is an optimum in between; contrast alone is not the lever
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

for f in "${SEM_COND}" "${EMPTY_PROMPTS}" "${SPECS}/cc3.json" \
         "${CONDS}/ridge_depth1024_sub-${STAG#sub-}_test.npy" \
         "${CONDS}/eeg_depth1024_${STAG}_test.npy"; do
  [[ -f "${f}" ]] || { echo "[FATAL] missing ${f}" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
log "===== [0] fail fast on CUDA ====="
"${PYTHON}" - <<'PY' || exit 1
import torch
if not torch.cuda.is_available():
    raise SystemExit("[FATAL] CUDA unavailable - refusing to silently fall back to CPU")
print(f"  torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
PY

# ---------------------------------------------------------------------------
log "===== [1] variance restoration ====="
log "  scale is fixed from the GT TRAIN bank's variability only (no test rows, no labels)"
"${PYTHON}" - "${CC}" "${CONDS}" "${STAG}" "${SEED}" <<'PY' || exit 1
import sys, json
from pathlib import Path
import numpy as np
cc, conds, stag, seed = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
sid = stag.replace("sub-", "")
rng = np.random.default_rng(seed)

def l2n(x):
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

def rowcos(x, cap=2000):
    """mean pairwise cosine across rows.  near 1.0 == the bank is one vector."""
    if len(x) > cap:
        x = x[rng.choice(len(x), cap, replace=False)]
    y = l2n(x); n = len(y); S = y @ y.T
    return float((S.sum() - np.trace(S)) / (n * (n - 1)))

def solve_a(P, m, tgt):
    """rowcos(m + a*(P-m)) decreases monotonically in a; find a hitting tgt."""
    lo, hi = 0.0, 1e7
    for _ in range(90):
        a = 0.5 * (lo + hi)
        if rowcos(m + (P - m) * a) > tgt:
            lo = a
        else:
            hi = a
    return 0.5 * (lo + hi)

rep = {"targets_from": "GT TRAIN banks (leak-free)", "banks": {}}

# what each structural condition is trying to predict, measured on TRAIN only
tgt_rc = {}
for lab in ("depth", "edge"):
    tr = np.load(cc / f"clip_{lab}1024_train.npy").astype(np.float32)
    tgt_rc[lab] = rowcos(tr)
    rep[f"gt_train_rowcos_{lab}"] = tgt_rc[lab]

for lab in ("depth", "edge"):
    # --- the leak-free ridge bank (C2's input) --------------------------------
    P = np.load(conds / f"ridge_{lab}1024_sub-{sid}_test.npy").astype(np.float32)
    m = P.mean(0, keepdims=True)
    a = solve_a(P, m, tgt_rc[lab])
    for tag, scale in (("varest", a), ("demean", 1e7)):
        arr = l2n(m + (P - m) * scale)
        p = conds / f"{tag}ridge_{lab}1024_{stag}_test.npy"
        np.save(p, arr)
        rep["banks"][f"{tag}_ridge_{lab}"] = {
            "path": str(p), "scale": float(scale) if scale < 1e6 else "de-mean(inf)",
            "rowcos_in": rowcos(P), "rowcos_out": rowcos(arr)}

    # --- the SHIPPED UCK -> CLIP bank (A8's input) ---------------------------
    U = np.load(conds / f"eeg_{lab}1024_sub-{stag}_test.npy").astype(np.float32)
    mu = U.mean(0, keepdims=True)
    au = solve_a(U, mu, tgt_rc[lab])
    arr = l2n(mu + (U - mu) * au)
    p = conds / f"varest_uck_{lab}1024_{stag}_test.npy"
    np.save(p, arr)
    rep["banks"][f"varest_uck_{lab}"] = {
        "path": str(p), "scale": float(au),
        "rowcos_in": rowcos(U), "rowcos_out": rowcos(arr)}

# --- the semantic bank, restored the same way (D4) --------------------------
img_tr = np.load(cc / "clip_img1024_train.npy").astype(np.float32)
sem = np.load(Path("/project/peilab/why/NeuroBridge/outputs/nw4_10s/arms/a_hi/conds")
              / stag / "cal_test.npy").astype(np.float32)
ms = sem.mean(0, keepdims=True)
asem = solve_a(sem, ms, rowcos(img_tr))
arr = l2n(ms + (sem - ms) * asem)
p = conds / f"varest_sem_{stag}_test.npy"
np.save(p, arr)
rep["banks"]["varest_sem"] = {"path": str(p), "scale": float(asem),
                              "rowcos_in": rowcos(sem), "rowcos_out": rowcos(arr),
                              "gt_train_rowcos_img": rowcos(img_tr)}

(conds / "varest_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
print(json.dumps(rep, indent=2))
print("[ok] variance-restored banks written")
PY

# ---------------------------------------------------------------------------
log "===== [2] what each bank carries after restoration ====="
"${PYTHON}" - "${CC}" "${CONDS}" "${STAG}" <<'PY'
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

print(f"  chance top1 = {1/200:.4f}\n")
print(f"  {'bank':<34}{'rowcos':>9}{'diag':>9}{'top1':>8}{'xchance':>9}")
print("  "+"-"*69)
show = [("ridge (C2 as emitted)", f"ridge_depth1024_sub-08_test.npy", "depth"),
        ("  + variance restored",  f"varestridge_depth1024_{stag}_test.npy", "depth"),
        ("  + de-meaned",          f"demeanridge_depth1024_{stag}_test.npy", "depth"),
        ("UCK->CLIP (A8 shipped)", f"eeg_depth1024_{stag}_test.npy", "depth"),
        ("  + variance restored",  f"varest_uck_depth1024_{stag}_test.npy", "depth"),
        ("---", "", ""),
        ("ridge (C2 as emitted)", f"ridge_edge1024_sub-08_test.npy", "edge"),
        ("  + variance restored",  f"varestridge_edge1024_{stag}_test.npy", "edge"),
        ("  + de-meaned",          f"demeanridge_edge1024_{stag}_test.npy", "edge"),
        ("UCK->CLIP (A8 shipped)", f"eeg_edge1024_{stag}_test.npy", "edge"),
        ("  + variance restored",  f"varest_uck_edge1024_{stag}_test.npy", "edge"),
        ("---", "", ""),
        ("a_hi SEM (shipped)",     "SEM", "img"),
        ("  + variance restored",  f"varest_sem_{stag}_test.npy", "img")]
for name, fn, lab in show:
    if name == "---":
        print("  "+"-"*69); continue
    p = (Path("/project/peilab/why/NeuroBridge/outputs/nw4_10s/arms/a_hi/conds")/stag/"cal_test.npy"
         if fn == "SEM" else conds/fn)
    if not p.is_file():
        print(f"  {name:<34}  MISSING"); continue
    x = np.load(p).astype(np.float32); t = gt[lab]
    print(f"  {name:<34}{rc(x):>9.4f}{diag(x,t):>9.4f}{top1(x,t):>8.4f}{top1(x,t)*200:>9.1f}")
PY

# ---------------------------------------------------------------------------
cond_path() {
  case "$1" in
    SEM)   echo "${SEM_COND}" ;;
    SVR)   echo "${CONDS}/varest_sem_${STAG}_test.npy" ;;
    DEPV)  echo "${CONDS}/varestridge_depth1024_${STAG}_test.npy" ;;
    EDGEV) echo "${CONDS}/varestridge_edge1024_${STAG}_test.npy" ;;
    DEPD)  echo "${CONDS}/demeanridge_depth1024_${STAG}_test.npy" ;;
    EDGED) echo "${CONDS}/demeanridge_edge1024_${STAG}_test.npy" ;;
    DEPU)  echo "${CONDS}/varest_uck_depth1024_${STAG}_test.npy" ;;
    EDGEU) echo "${CONDS}/varest_uck_edge1024_${STAG}_test.npy" ;;
    *)     echo "" ;;
  esac
}

# name | cond keys | scale json | branch spec | pipeline | steps | guidance |
# use_cn | use_init | blur | strength | cn_scale | prompts
ARMS=(
  "D1_varest|SEM,DEPV,EDGEV|cc3||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
  "D2_demean|SEM,DEPD,EDGED|cc3||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
  "D3_varest_uck|SEM,DEPU,EDGEU|cc3||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
  "D4_varest_sem|SVR,DEPV,EDGEV|cc3||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
)

log "===== [3] generate + score ${#ARMS[@]} arms on ${STAG} ====="
DONE=()
for spec_line in "${ARMS[@]}"; do
  IFS='|' read -r ARM CKEYS SJSON BSPEC PIPE STEPS GUID UCN UINIT BLUR STRENGTH CNSC PROMPTS EXTRA <<< "${spec_line}"
  if [[ -n "${EXTRA:-}" || -z "${ARM}" || -z "${PIPE}" || -z "${PROMPTS}" ]]; then
    echo "[FATAL] arm definition is not 13 fields: '${spec_line}'" >&2; exit 1
  fi
  hr; log "arm ${ARM}: keys=${CKEYS} scales=${SJSON} ${PIPE} ${STEPS}step g=${GUID}"

  GEN="${OUT_ROOT}/arms/${ARM}/gen/${STAG}"
  EV="${OUT_ROOT}/arms/${ARM}/eval/${STAG}.json"
  mkdir -p "${GEN}" "${OUT_ROOT}/arms/${ARM}/eval"

  N=0; CLIST=""
  for k in ${CKEYS//,/ }; do
    p="$(cond_path "${k}")"
    if [[ -z "${p}" || ! -f "${p}" ]]; then
      log "WARN ${ARM}: condition ${k} unavailable (${p})"; N=-1; break
    fi
    CLIST="${CLIST:+${CLIST},}${p}"; N=$((N+1))
  done
  [[ "${N}" -eq -1 ]] && { log "WARN ${ARM}: skipped"; continue; }

  if [[ -f "${GEN}/generated/199.png" ]]; then
    log "--- ${ARM}: generation already present - skip"
  else
    if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/generate_layered_decode.py" \
        --cond-npys "${CLIST}" --ip-scale-json "${SPECS}/${SJSON}.json" \
        --prompts-json "${EMPTY_PROMPTS}" --output-dir "${GEN}" --tag "${ARM}_${STAG}" \
        --pipeline "${PIPE}" --use-cn "${UCN}" --use-init "${UINIT}" \
        --cn-scale "${CNSC}" --strength "${STRENGTH}" --init-blur-sigma "${BLUR}" \
        --gen-steps "${STEPS}" --gen-guidance "${GUID}" \
        --gen-size 512 --seed "${SEED}" --device cuda:0 \
        --layer-report "${OUT_ROOT}/arms/${ARM}/layer_report.json" \
        > "${LOG}/${ARM}_gen.log" 2>&1; then
      log "WARN ${ARM}: generation FAILED"; tail -n 20 "${LOG}/${ARM}_gen.log"; continue
    fi
    grep -E "^\[layout|^\[pure|^\[INFO\] pipeline" "${LOG}/${ARM}_gen.log" | head -4
  fi

  if [[ -f "${EV}" ]]; then
    log "--- ${ARM}: eval present - skip"
  elif ! "${PYTHON}" "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
      --gen-dir "${GEN}/generated" --output-json "${EV}" --tag "${ARM}_${STAG}" \
      --images-root "${IMAGES_ROOT}" --device cuda:0 \
      > "${LOG}/${ARM}_eval.log" 2>&1; then
    log "WARN ${ARM}: eval FAILED"; tail -n 20 "${LOG}/${ARM}_eval.log"; continue
  fi
  DONE+=("${ARM}")
  "${PYTHON}" - "${EV}" "${ARM}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  [{sys.argv[2]}] pixcorr {d['pixcorr']:.4f}  ssim {d['ssim']:.4f}  "
      f"incep {d['inception']:.4f}  clip {d['clip']:.4f}  alex2 {d['alex2']:.4f}  "
      f"alex5 {d['alex5']:.4f}  swav {d['swav']:.4f}  fid {d['fid']:.2f}")
PY
done

[[ "${#DONE[@]}" -eq 0 ]] && { log "no arm finished"; exit 1; }

# ---------------------------------------------------------------------------
log "===== [4] 2-way re-scoring ====="
"${PYTHON}" "${NB_ROOT}/scripts/nda/nw4_official_twoway.py" \
  --gen-root "${OUT_ROOT}/arms" --arms "$(IFS=,; echo "${DONE[*]}")" \
  --subjects "${SUBJ}" --images-root "${IMAGES_ROOT}" \
  --out "${OUT_ROOT}/official_twoway_varest.json" --device cuda:0 \
  > "${LOG}/official_twoway_varest.log" 2>&1 || { log "WARN twoway failed"; tail -20 "${LOG}/official_twoway_varest.log"; }
tail -n 16 "${LOG}/official_twoway_varest.log" 2>/dev/null

# ---------------------------------------------------------------------------
log "===== [5] summary ====="
"${PYTHON}" - "${OUT_ROOT}" "${STAG}" <<'PY'
import json, sys
from pathlib import Path
root, stag = Path(sys.argv[1]), sys.argv[2]
KEYS = ["pixcorr", "ssim", "inception", "clip", "alex2", "alex5", "swav", "fid"]
LBL = {
    "A1_pure_cc1":  "A1  SEM only                     (floor)",
    "C1_const":     "C1  + centroid                   (prior, no trial info)",
    "C3_shuf":      "C3  + GT rows shuffled           (misaligned)",
    "C2_ridge":     "C2  + ridge, SHRUNKEN            (rowcos 0.955)",
    "A8_eegdep":    "A8  + UCK->CLIP, SHRUNKEN        (rowcos 0.874)",
    "D1_varest":    "D1  + ridge, VARIANCE RESTORED   (rowcos 0.566)",
    "D2_demean":    "D2  + ridge, fully de-meaned",
    "D3_varest_uck":"D3  + UCK->CLIP, RESTORED        (deployable)",
    "D4_varest_sem":"D4  + RESTORED SEM and structure",
    "A5_band_cc3":  "A5  + GT structure + band anchor (target)",
    "A2_pure_cc3":  "A2  + GT structure               ",
    "A9_orc_cc3":   "A9  all GT                      (ceiling)",
}
ORDER = list(LBL)
rows = {}
for a in ORDER:
    f = root/"arms"/a/"eval"/f"{stag}.json"
    if f.is_file(): rows[a] = json.load(open(f))

print()
print("="*120)
print(f"NW5c {stag}: the structure branch was SHRUNK.  Does restoring its variance recover "
      f"the GT-structure gain?")
print("="*120)
hdr = f"{'arm':<44}" + "".join(f"{k:>9}" for k in KEYS)
print(hdr); print("-"*len(hdr))
for a in ORDER:
    if a not in rows: continue
    r = rows[a]
    print(f"{LBL[a]:<44}" + "".join(
        f"{r[k]:>9.4f}" if k != "fid" else f"{r[k]:>9.2f}" for k in KEYS))
print()
print("  published: ATM s08  pix .160 ssim .345 incep .734 clip .786")
print("             CogCapPro 10subj  pix .163 ssim .398 incep .779 clip .830")
print("             D2-FOSA  pix .193 ssim .350")
print()
print("="*120)
print("VERDICT")
print("="*120)
def g(a,k): return rows[a][k] if a in rows else None
need = [a for a in ("A1_pure_cc1","C2_ridge","D1_varest") if a not in rows]
if need:
    print(f"  cannot conclude - missing {need}"); raise SystemExit(0)
for k in ("inception","clip"):
    a1, c2, d1, a5, a9 = (g(a,k) for a in ("A1_pure_cc1","C2_ridge","D1_varest","A5_band_cc3","A9_orc_cc3"))
    d2, d3, d4 = g("D2_demean",k), g("D3_varest_uck",k), g("D4_varest_sem",k)
    a8 = g("A8_eegdep",k)
    print(f"\n  --- {k} ---")
    print(f"    A1  semantic floor                {a1:>8.4f}")
    print(f"    C2  ridge, shrunken               {c2:>8.4f}   ({c2-a1:+.4f} vs floor)")
    if d1 is not None:
        print(f"    D1  ridge, variance restored      {d1:>8.4f}   ({d1-a1:+.4f} vs floor, "
              f"{d1-c2:+.4f} vs C2)")
    if d3 is not None:
        print(f"    D3  UCK->CLIP, restored           {d3:>8.4f}   ({d3-a1:+.4f} vs floor, "
              f"{d3-a8:+.4f} vs A8 as shipped)  [DEPLOYABLE]")
    if d4 is not None:
        print(f"    D4  restored SEM + structure      {d4:>8.4f}   ({d4-d1:+.4f} vs D1)")
    if d2 is not None:
        print(f"    D2  fully de-meaned               {d2:>8.4f}   ({d2-d1:+.4f} vs D1)")
    if a5 is not None:
        print(f"    A5  GT structure (target)         {a5:>8.4f}")
    if a9 is not None:
        print(f"    A9  all-GT (ceiling)              {a9:>8.4f}")
    if d1 is None:
        continue
    gt_gain = (a5 - a1) if a5 else (a9 - a1)
    if d1 - c2 > 0.02:
        frac = (d1-a1)/gt_gain if gt_gain > 0 else float("nan")
        print(f"    => SHRINKAGE WAS THE BUG.  Restoring the ridge's variance adds "
              f"{d1-c2:+.4f} over the same conditions un-restored, and recovers "
              f"{frac:.0%} of the GT-structure gain.")
        if d3 is not None and d3 - a8 > 0.02:
            print(f"       D3 says the same for the SHIPPED pipeline: {d3-a8:+.4f} over "
                  "A8 without retraining anything.  If D3 holds, the production fix is a "
                  "one-line rescale, and a trained structure head is optional rather "
                  "than required.")
        elif d1 > 0.5*gt_gain + a1:
            print("       Remaining gap to A5/A9 is what a TRAINED structure head must "
                  "close; its target objective is contrast restoration, not just MSE.")
    else:
        print(f"    => shrinkage was not the lever (D1-C2 = {d1-c2:+.4f}).  A trained "
              "structure head is required, and the linear ridge ceiling is the thing to "
              "beat before investing in one.")
PY

log "===== done ====="
