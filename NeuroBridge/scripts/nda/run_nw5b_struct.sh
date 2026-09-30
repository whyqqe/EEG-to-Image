#!/usr/bin/env bash
# =============================================================================
# NW5b sub-08: is the structure branch carrying TRIAL information, or only a PRIOR?
#
# WHERE WE ARE
# ------------
# nw5_s08 established, on this subject and with these scorers:
#
#   A1  SEM only              incep 0.7490  clip 0.8245  ssim 0.2760
#   A2  SEM + GT depth/edge   incep 0.8616  clip 0.9091  ssim 0.3430
#   A3  GT depth/edge only    incep 0.7484  clip 0.8480
#   A8  SEM + our EEG depth   incep 0.7490  clip 0.8245   <-- +0.000
#
# So the structure branch is worth +0.113 incep / +0.085 clip when it is fed the
# GROUND-TRUTH depth/edge CLIP rows, and exactly +0.000 when fed ours.  A8's
# failure is not a tuning gap; it is total.
#
# TWO DIAGNOSES, BOTH MEASURED (nw5_struct_probe.py, job on this node)
# -------------------------------------------------------------------
# (1) OUR DEPLOYABLE STRUCTURE PATH IS WORSE THAN A CONSTANT.
#     Against the GT depth/edge test banks, with the mean train embedding as a
#     reference point that costs nothing to compute:
#
#        target   constant   ours(cal)   ours(raw)   ridge(leak-free)   chance
#        depth     0.7480     0.3907      0.4757        0.7823            0.0050
#        edge      0.7218    -0.0320      0.1271        0.7605            0.0050
#
#     Both shipped banks sit BELOW the constant, and the edge bank is *negative*.
#     Injecting them cannot beat injecting nothing, which is exactly what A8 showed.
#
# (2) THE CONCENTRATION CALIBRATION WAS DESTROYING THE CONDITION.
#     `nw5_make_struct_conds.py` ran gem_calib.py on the structure banks, i.e. it
#     quantile-matched each row to the GT bank's row-norm distribution.  For a
#     structural condition the row direction IS the signal, and that step moved
#     depth 0.4757 -> 0.3907 and edge 0.1271 -> -0.0320.  Calibration is a semantic
#     tool (where the bank is diffuse and rows are redundant); on a concentrated
#     structural bank it only adds noise.  Both banks are kept below so the claim is
#     falsifiable.
#
# AND ONE THING THE PROBE CANNOT SETTLE
# -------------------------------------
# The depth bank's constant baseline is 0.7480: 97% of the ridge's 0.7823 is
# available without looking at any EEG at all.  The same holds for edge (0.7218 of
# 0.7605).  If IP-Adapter cares mostly about the *prior* a condition represents
# rather than its per-trial variation -- which is plausible, because A2's gain came
# from adding a whole branch, not from a per-trial correction -- then the whole
# EEG->structure prediction problem may be unnecessary: a fixed learned bank would
# score the same.  Top-1 retrieval, the axis a constant cannot fake, is only 3x
# (depth) and 11x (edge) chance.  No linear probe can tell us which of these
# regimes we are in.  Generation can, in three arms.
#
# THE ARMS  (all: cc3 layout, SDXL-Turbo 15 steps CFG 0, no CN, no init, empty prompt,
#            identical to A1/A2/A3 except for what the two structural branches receive)
# -----------------------------------------------------------------------------------
#  C1_const  SEM + train-bank MEAN depth/edge      -- if this reproduces A2, the
#            structure branch is a PRIOR and the deployable model needs no EEG->
#            structure head at all.  That is the strongest possible outcome: it
#            removes the one component the probe showed we cannot yet build.
#  C2_ridge  SEM + leak-free ridge depth/edge (raw, uncalibrated, l2n)  -- the honest
#            EEG-only structure, top1 3x/11x chance.  Pairs with C1: C2-C1 is what
#            per-trial information is actually worth.
#  C3_shuf   SEM + GT depth/edge rows SHUFFLED across trials (fixed seed) -- same
#            content distribution, wrong row alignment, i.e. a *matched* structure
#            prior drawn from the real bank.  This is the load-bearing control:
#            if C3 == A2, per-trial alignment is irrelevant; if C3 < A2, trial
#            information in the structural condition is real and must be predicted.
#            C1 vs C3 separates "any prior" from "the true prior's distribution".
#
# Read the three together against A1 / A2 / A8, which are already scored:
#   C3 == A2        -> structure is a prior; ship C1, drop the structure head
#   C3 <  A2, C2 > A8 -> per-trial structure matters and ridge already beats ours
#   C1 == C3 == A2  -> the bank is so concentrated it is one vector; ship the constant
#   C2 == C1        -> even a 11x-chance ridge adds nothing; the head is not the lever
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

for f in "${SEM_COND}" "${EMPTY_PROMPTS}" "${SPECS}/cc1.json" "${SPECS}/cc3.json"; do
  [[ -f "${f}" ]] || { echo "[FATAL] missing ${f}" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
log "===== [0] fail fast on CUDA ====="
if [[ "${PREP_ONLY:-0}" == "1" ]]; then
  log "PREP_ONLY=1 -> skipping the CUDA gate (this node builds condition banks only)"
else
  "${PYTHON}" - <<'PY' || exit 1
import torch
if not torch.cuda.is_available():
    raise SystemExit("[FATAL] CUDA unavailable - refusing to silently fall back to CPU")
print(f"  torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
PY
fi

# ---------------------------------------------------------------------------
log "===== [1] build the three structural condition banks ====="
"${PYTHON}" - "${CC}" "${CONDS}" "${STAG}" "${SUBJ}" "${SEED}" <<'PY' || exit 1
import sys, json
from pathlib import Path
import numpy as np
cc, conds, stag, subj, seed = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
conds.mkdir(parents=True, exist_ok=True)
rep = {}

def l2n(x):
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

for lab in ("depth", "edge"):
    tr = np.load(cc / f"clip_{lab}1024_train.npy").astype(np.float32)
    te = np.load(cc / f"clip_{lab}1024_test.npy").astype(np.float32)
    n = len(te)

    # --- C1: the train-bank centroid, i.e. the prior with no trial information ---
    const = l2n(tr.mean(0, keepdims=True)).repeat(n, axis=0)
    # --- C3: real bank rows, wrong trials (matched distribution, broken alignment) ---
    rng = np.random.default_rng(seed)
    shuf = l2n(te[rng.permutation(n)])
    for tag, arr in (("const", const), ("shuf", shuf)):
        p = conds / f"{tag}_{lab}1024_{stag}_test.npy"
        np.save(p, arr)
        rep[f"{tag}_{lab}"] = str(p)

    # how far is the centroid from the mean row? (if ~1.0 the bank is one vector)
    rep[f"{lab}_centroid_vs_meanrow_cos"] = float(
        (const[0] @ l2n(te.mean(0, keepdims=True))[0]))
    rep[f"{lab}_const_row_spread"] = float(np.linalg.norm(const - const[0], axis=1).max())
    rep[f"{lab}_shuf_rowmin_l2n"] = float(np.linalg.norm(shuf, axis=1).min())

(conds / "struct_banks_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
print(json.dumps(rep, indent=2))
print("[ok] const + shuffled banks written")
PY

# ---------------------------------------------------------------------------
log "===== [1.5] emit the leak-free ridge structure banks (C2) ====="
"${PYTHON}" "${NB_ROOT}/scripts/nda/nw5_struct_probe.py" \
  --subject "${SUBJ}" --emit "${CONDS}" \
  --out "${OUT_ROOT}/struct_probe_sub-${STAG#sub-}.json" \
  > "${LOG}/probe_emit.log" 2>&1 || { log "WARN probe failed"; tail -n 20 "${LOG}/probe_emit.log"; }
grep -E "^\[emit\]|^   ridge" "${LOG}/probe_emit.log" || tail -n 10 "${LOG}/probe_emit.log"

for lab in depth edge; do
  f="${CONDS}/ridge_${lab}1024_sub-${STAG#sub-}_test.npy"
  [[ -f "${f}" ]] || { log "WARN missing ${f} - C2 will be skipped"; }
done

cond_path() {
  case "$1" in
    SEM)    echo "${SEM_COND}" ;;
    DEPG)   echo "${CC}/clip_depth1024_test.npy" ;;
    EDGEG)  echo "${CC}/clip_edge1024_test.npy" ;;
    DEPC)   echo "${CONDS}/const_depth1024_${STAG}_test.npy" ;;
    EDGEC)  echo "${CONDS}/const_edge1024_${STAG}_test.npy" ;;
    DEPS)   echo "${CONDS}/shuf_depth1024_${STAG}_test.npy" ;;
    EDGES)  echo "${CONDS}/shuf_edge1024_${STAG}_test.npy" ;;
    DEPR)   echo "${CONDS}/ridge_depth1024_sub-${STAG#sub-}_test.npy" ;;
    EDGER)  echo "${CONDS}/ridge_edge1024_sub-${STAG#sub-}_test.npy" ;;
    *)      echo "" ;;
  esac
}

# ---------------------------------------------------------------------------
log "===== [2] what each bank actually carries (the reading of the arms) ====="
"${PYTHON}" - "${CC}" "$(cond_path DEPC)" "$(cond_path EDGEC)" "$(cond_path DEPS)" \
    "$(cond_path EDGES)" "$(cond_path DEPR)" "$(cond_path EDGER)" <<'PY'
import sys
from pathlib import Path
import numpy as np
cc = Path(sys.argv[1])
def l2n(x): return (x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)).astype(np.float32)
def diag(pred, tgt): return float((l2n(pred)*l2n(tgt)).sum(1).mean())
def top1(pred, tgt): return float(((l2n(pred)@l2n(tgt).T).argmax(1) == np.arange(len(tgt))).mean())

banks = list(zip(("DEPC const", "EDGEC const", "DEPS shuf", "EDGES shuf",
                  "DEPR ridge", "EDGER ridge"), sys.argv[2:8]))
gt = {lab: l2n(np.load(cc/f"clip_{lab}1024_test.npy").astype(np.float32))
      for lab in ("depth", "edge")}
# what a constant scores on each bank: the bar every structural condition must clear
const_bar = {lab: diag(l2n(np.load(cc/f"clip_{lab}1024_train.npy").astype(np.float32)
                           .mean(0, keepdims=True)).repeat(len(gt[lab]), 0), gt[lab])
             for lab in ("depth", "edge")}
print(f"  chance top1 = {1/len(gt['depth']):.4f}    "
      f"constant-bank diag: depth {const_bar['depth']:.4f}  edge {const_bar['edge']:.4f}")
print()
print(f"  {'bank':<16}{'-> depth':>18}{'-> edge':>18}")
print(f"  {'':<16}{'diag':>9}{'top1':>9}{'diag':>9}{'top1':>9}")
print("  " + "-"*52)
for name, p in banks:
    if not Path(p).is_file():
        print(f"  {name:<16}  MISSING {p}"); continue
    x = np.load(p).astype(np.float32)
    tgt = gt["depth"] if "depth" in name.lower() or name.startswith("DEP") else gt["edge"]
    lab = "depth" if name.startswith("DEP") else "edge"
    print(f"  {name:<16}{diag(x,gt['depth']):>18.4f}{top1(x,gt['depth']):>9.4f}"
          f"{diag(x,gt['edge']):>9.4f}{top1(x,gt['edge']):>9.4f}")
PY

# ---------------------------------------------------------------------------
# name | cond keys | scale json | branch spec | pipeline | steps | guidance |
# use_cn | use_init | blur | strength | cn_scale | prompts
ARMS=(
  "C1_const|SEM,DEPC,EDGEC|cc3||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
  "C2_ridge|SEM,DEPR,EDGER|cc3||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
  "C3_shuf|SEM,DEPS,EDGES|cc3||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
)

log "===== [3] generate + score ${#ARMS[@]} arms on ${STAG} ====="
if [[ "${PREP_ONLY:-0}" == "1" ]]; then
  log "PREP_ONLY=1 -> stopping before generation.  Only the constant bank (C1) and, if"
  log "  the emitted ridge banks are absent, none of C2 depend on a GPU for their"
  log "  construction, but the shuf arm needs GT test banks only.  Submit this script"
  log "  to a GPU node for the actual run."
  exit 0
fi
DONE=()
for spec_line in "${ARMS[@]}"; do
  IFS='|' read -r ARM CKEYS SJSON BSPEC PIPE STEPS GUID UCN UINIT BLUR STRENGTH CNSC PROMPTS EXTRA <<< "${spec_line}"
  if [[ -n "${EXTRA:-}" || -z "${ARM}" || -z "${PIPE}" || -z "${PROMPTS}" ]]; then
    echo "[FATAL] arm definition is not 13 fields: '${spec_line}'" >&2; exit 1
  fi
  if [[ -n "${SJSON}" && -n "${BSPEC}" ]]; then
    echo "[FATAL] ${ARM}: both a scale json and a branch spec" >&2; exit 1
  fi
  hr; log "arm ${ARM}: keys=${CKEYS} scales=${SJSON:-spec:${BSPEC}} ${PIPE} ${STEPS}step g=${GUID}"

  CDIR="${OUT_ROOT}/arms/${ARM}/conds/${STAG}"
  GEN="${OUT_ROOT}/arms/${ARM}/gen/${STAG}"
  EV="${OUT_ROOT}/arms/${ARM}/eval/${STAG}.json"
  mkdir -p "${CDIR}" "${OUT_ROOT}/arms/${ARM}/eval"

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

if [[ "${#DONE[@]}" -eq 0 ]]; then
  log "no arm finished - nothing to score"; exit 1
fi

# ---------------------------------------------------------------------------
log "===== [4] 2-way (Pearson + cosine) re-scoring of the new arms ====="
"${PYTHON}" "${NB_ROOT}/scripts/nda/nw4_official_twoway.py" \
  --gen-root "${OUT_ROOT}/arms" --arms "$(IFS=,; echo "${DONE[*]}")" \
  --subjects "${SUBJ}" --images-root "${IMAGES_ROOT}" \
  --out "${OUT_ROOT}/official_twoway_new.json" --device cuda:0 \
  > "${LOG}/official_twoway_new.log" 2>&1 || { log "WARN twoway failed"; tail -n 20 "${LOG}/official_twoway_new.log"; }
tail -n 16 "${LOG}/official_twoway_new.log" 2>/dev/null

# ---------------------------------------------------------------------------
log "===== [5] summary: is the structure branch trial information or a prior? ====="
"${PYTHON}" - "${OUT_ROOT}" "${STAG}" "$(IFS=,; echo "${DONE[*]}")" <<'PY'
import json, sys
from pathlib import Path
root, stag = Path(sys.argv[1]), sys.argv[2]
new = sys.argv[3].split(",") if len(sys.argv) > 3 else []

KEYS = ["pixcorr", "ssim", "inception", "clip", "alex2", "alex5", "swav", "fid"]
ORDER = ["A1_pure_cc1", "A2_pure_cc3", "A3_pure_all3", "A8_eegdep", "A9_orc_cc3"] + new
LABEL = {
    "A1_pure_cc1":  "A1  SEM only                (reference floor)",
    "A2_pure_cc3":  "A2  SEM + GT depth/edge     (structure ceiling)",
    "A3_pure_all3": "A3  GT depth/edge only      (no semantic)",
    "A8_eegdep":    "A8  SEM + our EEG depth     (the failure)",
    "A9_orc_cc3":   "A9  GT img+depth+edge       (absolute ceiling)",
    "C1_const":     "C1  SEM + bank CENTROID     (prior, no trial info)",
    "C2_ridge":     "C2  SEM + RIDGE depth/edge  (leak-free EEG)",
    "C3_shuf":      "C3  SEM + GT rows SHUFFLED  (matched, misaligned)",
}

rows = {}
for arm in ORDER:
    f = root / "arms" / arm / "eval" / f"{stag}.json"
    if f.is_file():
        rows[arm] = json.load(open(f))
ref = Path("/project/peilab/why/NeuroBridge/outputs/nw4_10s/arms/a_hi/eval/sub-08.json")
if ref.is_file():
    rows["a_hi(shipped)"] = json.load(open(ref))

print()
print("=" * 122)
print(f"NW5b {stag}: does the structural condition need to know WHICH TRIAL it is?")
print("=" * 122)
hdr = f"{'arm':<40}" + "".join(f"{k:>9}" for k in KEYS)
print(hdr); print("-" * len(hdr))
for arm in ORDER + (["a_hi(shipped)"] if "a_hi(shipped)" in rows else []):
    if arm not in rows:
        continue
    r = rows[arm]
    lbl = LABEL.get(arm, arm[:38])
    print(f"{lbl:<40}" + "".join(
        f"{r[k]:>9.4f}" if k != "fid" else f"{r[k]:>9.2f}" for k in KEYS))

print()
print("published bars (10-subject means):")
for name, bar in (("ATM sub-08", "pixcorr .160 ssim .345 incep .734 clip .786"),
                  ("CogCapPro 10subj", "pixcorr .163 ssim .398 incep .779 clip .830"),
                  ("D2-FOSA", "pixcorr .193 ssim .350")):
    print(f"  {name:<18} {bar}")

# ---- the verdict, stated as the three comparisons the arms were built for ----
print()
print("=" * 122)
print("VERDICT")
print("=" * 122)
def g(a, k):
    return rows[a][k] if a in rows else None
missing = [a for a in ("A1_pure_cc1", "A2_pure_cc3", "C1_const", "C2_ridge", "C3_shuf")
           if a not in rows]
if missing:
    print(f"  cannot conclude - missing {missing} (rerun this script; finished arms are skipped)")
    raise SystemExit(0)

for k in ("inception", "clip"):
    a1, a2, c1, c2, c3 = (g(a, k) for a in ("A1_pure_cc1", "A2_pure_cc3",
                                            "C1_const", "C2_ridge", "C3_shuf"))
    a3, a8 = g("A3_pure_all3", k), g("A8_eegdep", k)
    print(f"\n  --- {k} ---")
    print(f"    A1 semantic only      {a1:.4f}")
    print(f"    A2 + GT structure     {a2:.4f}   (structure is worth {a2-a1:+.4f})")
    print(f"    C3 + GT shuffled      {c3:.4f}   (recovered by a MISALIGNED GT bank: "
          f"{c3-a1:+.4f} of {a2-a1:+.4f})")
    print(f"    C1 + bank centroid    {c1:.4f}   (recovered by a CONSTANT: "
          f"{c1-a1:+.4f})")
    print(f"    C2 + ridge (leak-free){c2:.4f}   (recovered by our best honest path: "
          f"{c2-a1:+.4f})")
    print(f"    A8 + our shipped path {a8:.4f}   (recovered by the current pipeline: "
          f"{a8-a1:+.4f})")

    a1_, a2_, c1_, c2_, c3_ = g("A1_pure_cc1", k), g("A2_pure_cc3", k), c1, c2, c3
    gain = a2_ - a1_
    if abs(gain) < 1e-4:
        print("    => A2 failed to reproduce; the structure branch itself is unstable here")
        continue
    r_shuf = (c3_ - a1_) / gain
    r_const = (c1_ - a1_) / gain
    r_ridge = (c2_ - a1_) / gain
    print(f"    recovered fraction of the GT-structure gain: "
          f"shuffled {r_shuf:+.0%}   centroid {r_const:+.0%}   ridge {r_ridge:+.0%}")
    if r_shuf > 0.5 and r_const > 0.5:
        print("    => STRUCTURE IS A PRIOR.  Misaligned GT rows and a single constant "
              "vector both recover most of the gain, so the branch is asking for the "
              "right *statistics*, not for this trial's layout.  Ship a fixed learned "
              "bank (C1) and spend no capacity on an EEG->structure head.")
    elif r_shuf < 0.25:
        print("    => STRUCTURE IS TRIAL INFORMATION.  Shuffling the rows destroys the "
              "gain, so per-trial alignment is load-bearing and the condition must be "
              "PREDICTED.  Compare C2 (ridge) against A8 (shipped): the remaining gap "
              "is exactly what a trained structure head must close.")
    else:
        print(f"    => MIXED.  Shuffled rows recover {r_shuf:.0%} of the gain: part "
              "prior, part trial information.  A constant cannot do the job, but the "
              "per-trial component is smaller than the branch's total effect.")
    if r_ridge < 0.1:
        print(f"       NOTE: the leak-free ridge recovers only {r_ridge:+.0%}. Linear "
              "EEG->structure is not the lever; a trained head would have to beat it "
              "by a wide margin to matter.")

print()
print("(C1 vs C3 is the finer question: the centroid is one vector, the shuffled bank "
      "is the true bank's distribution with the labels removed.  C1 ~= C3 means the "
      "bank is so concentrated that one vector IS the distribution.)")
PY

log "===== done ====="
