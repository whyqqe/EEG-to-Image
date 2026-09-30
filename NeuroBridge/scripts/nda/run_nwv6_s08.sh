#!/usr/bin/env bash
# =============================================================================
# NW-v6 sub-08: the CogCapPro-style ENCODER-SIDE upgrade, training to evaluation.
#
# WHY THE ENCODER SIDE
# --------------------
# Our generator is not the bottleneck.  A5 -- our operator, our semantic bank, plus
# GROUND-TRUTH depth/edge -- gets incep 0.8400 / clip 0.9123, at or above CogCapPro's
# published sub-08 (0.831 / 0.903).  Fully deployable we get 0.7440 / 0.8525.  The whole
# interval is the structure branch:
#
#   E1  no structure                       0.7302 / 0.8122   floor
#   E2  + variance-restored linear ridge    0.7363 / 0.8215   5.5% of the GT gain
#   G3  + per-modality trained head         0.7483 / 0.8314   16.5%
#   A5  + GT depth/edge                     0.8400 / 0.9123   the target
#
# CogCapPro's own upgrade over CogCap was also encoder-side: its generator is the same
# SDXL-Turbo + multi-branch IP-Adapter + empty prompt + CFG 0 stack, and its headline
# +25.9% Top-1 / +10.6% Top-5 are retrieval numbers.  We already have its two
# generator-side pieces (cc3 asymmetric injection, text dropped at inference) and none of
# its four encoder-side ones.  This job adds them.
#
# THE FOUR UPGRADES
# -----------------
#  (1) STH-Align: ONE shared trunk + THREE projection heads, JOINTLY trained, instead of
#      three independent regressors.  Also 3x cheaper -- one run now serves all three
#      modalities.  nwv6_align_train.py.
#  (2) SCM-Loss: multi-positive InfoNCE, because the task is one-to-many.  The bank is
#      1654 concepts x 10 images and the script VERIFIES the block ordering at runtime
#      (same-concept cos 0.75 vs different-concept 0.40).  Hard-positive InfoNCE labels
#      two images of the same concept as mutual errors.  w is gridded, with w=0 as the
#      hard-positive control under identical code.
#  (3) FUSION: feed shared_r (+) specific_s.  Probing already showed cat beats every
#      single representation at layout (top-5 0.38 vs 0.35/0.355), while specific_s ALONE
#      is the WORST (0.225) -- so the appealing "geometry lives in the private part"
#      story is false, but the fusion still adds information.
#  (4) UNCERTAINTY WEIGHTING: learned per-modality log-variance (Kendall), clamped to
#      +-4, so joint training does not need a hand-tuned loss balance.  Plain mean kept
#      as a control in the grid.
#
# WHAT IS ALREADY KNOWN, SO IT IS NOT RETRIED
# -------------------------------------------
#   * capacity is NOT the limit -- bigger was worse (3.1M MLP 0.3036 < 461k low-rank
#     0.3424 in val dev_corr), so only the objective and the input features change here;
#   * sigma >= 4 saturates (sigma 3->8 moves ssim by 0.0003 against 0.0019 seed noise),
#     so the sweep is not repeated -- sigma 3.0 and the measured-best 4.0 are used;
#   * "use the subject-specific part for geometry" is REFUTED (verdict in
#     outputs/ocf/ss_parts_probe_sub08.json); only the fusion form is used.
#
# THE ARMS
# --------
#   W1  JIMG + JDEP + JEDGE        all three branches from the joint stack
#   W2  SEM  + JDEP + JEDGE        isolates the structure gain (crosses E2 / G3 / M6)
#   W3  same as W1 at sigma 4.0    the measured-best anchor
#   W4  IMGB + JDEP + JEDGE        GT image + our structure  } the 2x2 decomposition:
#   W5  JIMG + GT depth/edge       our image + GT structure  } which side owns the gap?
#   W6  IMGB + JDEP + JEDGE s4.0
# W4 and W5 are the cells we never had.  W5 vs A5 (SEM + GT structure, 0.8400) measures
# the new image branch alone; W4 vs A5 measures how much of the structure gap closed.
#   then M6 over a 13-candidate heterogeneous pool.
# =============================================================================

set -uo pipefail

NB_ROOT="/project/peilab/why/NeuroBridge"
cd "${NB_ROOT}"

PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
STAG="${STAG:-sub-08}"
SUBJ="${SUBJ:-8}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-${NB_ROOT}/outputs/nw6_s08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
CC="${NB_ROOT}/outputs/gem/cond_cache"
SPECS="${NB_ROOT}/outputs/nw5_s08/specs"
CONDS="${OUT_ROOT}/conds"
NW5_CONDS="${NB_ROOT}/outputs/nw5_s08/conds"
INIT_RGB="${NB_ROOT}/outputs/sdedit_ll_full10/${STAG}/vae_head/pred_lowlevel_rgb_512"
CHAB="${NB_ROOT}/outputs/chab/sub-${STAG#sub-}/z_warm0/${STAG}"

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export HOME="${XDG_CACHE_HOME}"

LOG="${OUT_ROOT}/logs"
EMPTY_PROMPTS="${NB_ROOT}/outputs/nw5_s08/conds/prompts_empty.json"
SEM_COND="${NB_ROOT}/outputs/nw4_10s/arms/a_hi/conds/${STAG}/cal_test.npy"
ALIGN_REP="${CONDS}/joint_${STAG}_report.json"
mkdir -p "${LOG}" "${CONDS}" "${OUT_ROOT}/arms"

log() { echo "[$(date +%H:%M:%S)] $*"; }
hr()  { echo "------------------------------------------------------------------------"; }

for f in "${SEM_COND}" "${EMPTY_PROMPTS}" "${SPECS}/cc1.json" "${SPECS}/cc3.json" \
         "${CC}/clip_img1024_train.npy" "${CC}/clip_depth1024_train.npy" \
         "${CC}/clip_edge1024_train.npy" "${CHAB}/specific_s_train.npy" \
         "${CHAB}/shared_r_train.npy" "${NW5_CONDS}/head_depth1024_${STAG}_test.npy" \
         "${NW5_CONDS}/head_edge1024_${STAG}_test.npy"; do
  [[ -f "${f}" ]] || { echo "[FATAL] missing ${f}" >&2; exit 1; }
done
[[ -d "${INIT_RGB}" ]] || { echo "[FATAL] missing anchor dir ${INIT_RGB}" >&2; exit 1; }

log "===== [0] fail fast on CUDA ====="
"${PYTHON}" - <<'PY' || exit 1
import torch
if not torch.cuda.is_available():
    raise SystemExit("[FATAL] CUDA unavailable - refusing to silently fall back to CPU")
print(f"  torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
PY

log "===== [0.5] the two features that will be fused must be the same encoder ====="
"${PYTHON}" - "${NB_ROOT}" "${SUBJ}" <<'PY' || exit 1
import sys
from pathlib import Path
import numpy as np
root, sid = Path(sys.argv[1]), int(sys.argv[2])
def l2n(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)
a = l2n(np.load(root/f"outputs/ocf/intra_z/sub-{sid:02d}/shared_r_train.npy").astype(np.float32))
b = l2n(np.load(root/f"outputs/chab/sub-{sid:02d}/z_warm0/sub-{sid:02d}/shared_r_train.npy").astype(np.float32))
c = float((a*b).sum(1).mean())
print(f"  shared_r: intra_z vs z_warm0 cos {c:.6f}")
if c < 0.999:
    raise SystemExit("[FATAL] different encoders - fusing their parts would mix spaces")
print(f"  OK: specific_s from z_warm0 is safe to concatenate")
PY

log "===== [1] STH-Align: shared trunk + 3 heads, joint, SCM-Loss, uncertainty weighted ====="
"${PYTHON}" "${NB_ROOT}/scripts/nda/nwv6_align_train.py" \
  --subject "${SUBJ}" --stag "${STAG}" --out "${ALIGN_REP}" --device cuda:0 \
  --epochs 150 --patience 25 --batch 384 --out-dir "${CONDS}" \
  --features cat,shared --modalities img,depth,edge \
  > "${LOG}/align_train.log" 2>&1
ALIGN_RC=$?
grep -E "^\[concepts\]|^\[align\] (img|depth|edge): BEST|^        test dev_corr|^        emitted|^\[align\] wrote" \
  "${LOG}/align_train.log" || true
if [[ "${ALIGN_RC}" -ne 0 ]]; then
  log "FATAL alignment training failed (rc=${ALIGN_RC})"; tail -n 25 "${LOG}/align_train.log"; exit 1
fi
for m in img depth edge; do
  [[ -f "${CONDS}/joint_${m}1024_${STAG}_test.npy" ]] \
    || { echo "[FATAL] no joint bank for ${m}" >&2; exit 1; }
done

log "===== [1.5] what each condition bank carries (GT / ridge / head / joint) ====="
"${PYTHON}" - "${CC}" "${CONDS}" "${NW5_CONDS}" "${STAG}" <<'PY'
import sys
from pathlib import Path
import numpy as np
cc, conds, nw5, stag = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
def l2n(x): return (x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)).astype(np.float32)
def esplit(x):
    x=l2n(x); m=x.mean(0,keepdims=True); d=x-m
    return float((m**2).sum()), float((d**2).sum(1).mean())
def dm(pred,tgt):
    p,t=l2n(pred),l2n(tgt); S=p@t.T
    t1=float((S.argmax(1)==np.arange(len(t))).mean())
    rc=float((p*t).sum(1).mean())
    pd,td=l2n(p-p.mean(0,keepdims=True)),l2n(t-t.mean(0,keepdims=True))
    return t1,rc,float((pd*td).sum(1).mean())
print(f"  {'modality':<8}{'bank':<34}{'%E(mean)':>10}{'%E(dev)':>9}{'top1':>9}{'xchance':>9}"
      f"{'rowcos':>9}{'dev_corr':>10}")
print("  "+"-"*98)
banks = {
    "img":   [("GT (A9 target)", None),
              ("a_hi SEM (shipped)", None),
              ("JOINT img (new)", f"{conds}/joint_img1024_{stag}_test.npy")],
    "depth": [("GT (A5 target)", None),
              ("varest ridge (E2)", f"{nw5}/varestridge_depth1024_{stag}_test.npy"),
              ("head (G3)", f"{nw5}/head_depth1024_{stag}_test.npy"),
              ("JOINT depth (new)", f"{conds}/joint_depth1024_{stag}_test.npy")],
    "edge":  [("GT (A5 target)", None),
              ("varest ridge (E2)", f"{nw5}/varestridge_edge1024_{stag}_test.npy"),
              ("head (G3)", f"{nw5}/head_edge1024_{stag}_test.npy"),
              ("JOINT edge (new)", f"{conds}/joint_edge1024_{stag}_test.npy")],
}
sem = Path("/project/peilab/why/NeuroBridge/outputs/nw4_10s/arms/a_hi/conds")/stag/"cal_test.npy"
for lab, rows in banks.items():
    G = l2n(np.load(cc/f"clip_{lab}1024_test.npy").astype(np.float32))
    for name, fp in rows:
        if name.startswith("a_hi"):
            x = np.load(sem).astype(np.float32)
        elif fp is None:
            x = G
        else:
            p = Path(fp)
            if not p.is_file(): print(f"  {lab:<8}{name:<34} MISSING"); continue
            x = np.load(p).astype(np.float32)
        em,ed = esplit(x); t1,rc,dc = dm(x,G)
        print(f"  {lab:<8}{name:<34}{em:>10.3f}{ed:>9.3f}{t1:>9.4f}{t1*200:>9.1f}{rc:>9.4f}{dc:>10.4f}")
PY

cond_path() {
  case "$1" in
    SEM)   echo "${SEM_COND}" ;;
    JIMG)  echo "${CONDS}/joint_img1024_${STAG}_test.npy" ;;
    JDEP)  echo "${CONDS}/joint_depth1024_${STAG}_test.npy" ;;
    JEDGE) echo "${CONDS}/joint_edge1024_${STAG}_test.npy" ;;
    IMGB)  echo "${CC}/clip_img1024_test.npy" ;;
    DEPG)  echo "${CC}/clip_depth1024_test.npy" ;;
    EDGEG) echo "${CC}/clip_edge1024_test.npy" ;;
    DEPH)  echo "${NW5_CONDS}/head_depth1024_${STAG}_test.npy" ;;
    EDGEH) echo "${NW5_CONDS}/head_edge1024_${STAG}_test.npy" ;;
    DEPL)  echo "${NW5_CONDS}/varestridge_depth1024_${STAG}_test.npy" ;;
    EDGEL) echo "${NW5_CONDS}/varestridge_edge1024_${STAG}_test.npy" ;;
    *)     echo "" ;;
  esac
}

# ---------------------------------------------------------------------------
log "===== [2] generation helpers ====="
gen_one() {
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
  local ARM="$1" GEN="${OUT_ROOT}/arms/$1/gen/${STAG}"
  mkdir -p "${GEN}"
  gen_one "$1" "$2" "$3" "$4" "$5" "$6" "${GEN}" || return 1
  eval_one "${ARM}" "${GEN}"
}

DONE=()

hr; log "the four upgrade arms, plus the 2x2 decomposition cells"
while IFS='|' read -r A K S B; do
  [[ -z "${A}" ]] && continue
  hr; log "arm ${A}: keys=${K} layout=${S} anchor-sigma=${B}"
  run_arm "${A}" "${K}" "${S}" "${B}" 0.92 "${SEED}" && DONE+=("${A}")
done <<'SPEC'
W1_joint_all_s30|JIMG,JDEP,JEDGE|cc3|3.0
W2_joint_struct_s30|SEM,JDEP,JEDGE|cc3|3.0
W3_joint_all_s40|JIMG,JDEP,JEDGE|cc3|4.0
W4_gtimg_jstruct_s30|IMGB,JDEP,JEDGE|cc3|3.0
W5_jimg_gtstruct_s30|JIMG,DEPG,EDGEG|cc3|3.0
W6_gtimg_jstruct_s40|IMGB,JDEP,JEDGE|cc3|4.0
SPEC

log "===== [3] M6 over a 13-candidate heterogeneous pool ====="
M6_POOL=""
for A in W1_joint_all_s30 W2_joint_struct_s30 W3_joint_all_s40 W4_gtimg_jstruct_s30 \
         W5_jimg_gtstruct_s30 W6_gtimg_jstruct_s40 \
         E2_varest_s30 E2_seed43 E2_seed44 E2_seed45 E2_seed46 F_s40 G3_head_s30; do
  if [[ "${A}" == E2_* || "${A}" == F_* || "${A}" == G3_* ]]; then
    GD="${NB_ROOT}/outputs/nw5_s08/arms/${A}/gen/${STAG}"
  else
    GD="${OUT_ROOT}/arms/${A}/gen/${STAG}"
  fi
  [[ -f "${GD}/generated/199.png" ]] && M6_POOL="${M6_POOL:+${M6_POOL},}${GD}"
done
log "  pool: $(echo "${M6_POOL}" | tr ',' '\n' | wc -l) candidate sets"
log "  criteria: alpha*cos(CLIP(cand), sem_cond) - beta*L1(lowpass(cand), anchor)"
log "  nothing here sees GT or a class name; the anchor is our own low-level prediction"

M6SC="${OUT_ROOT}/m6_scores_nwv6.npz"
for ab in 1.0:0.0 1.0:0.25 1.0:0.5 1.0:1.0 1.0:2.0 0.5:1.0 0.0:1.0 0.0:2.0; do
  A="${ab%%:*}"; B="${ab##*:}"; NAME="M6v6_a${A/./}b${B/./}"
  ODIR="${OUT_ROOT}/arms/${NAME}/gen/${STAG}"; mkdir -p "${ODIR}"
  hr; log "M6 ${NAME}: alpha=${A} beta=${B}"
  if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/nw6_m6_select.py" \
      --cand-dirs "${M6_POOL}" --sem-cond "${SEM_COND}" --anchor-dir "${INIT_RGB}" \
      --scores-npz "${M6SC}" --standardize --alpha "${A}" --beta "${B}" \
      --out-dir "${ODIR}" --report "${OUT_ROOT}/arms/${NAME}/m6_report.json" \
      --device cuda:0 > "${LOG}/${NAME}_sel.log" 2>&1; then
    log "WARN ${NAME}: selection FAILED"; tail -n 12 "${LOG}/${NAME}_sel.log"; continue
  fi
  grep -E "^\[m6\] (loaded|across|standardized|alpha)" "${LOG}/${NAME}_sel.log" | head -3
  eval_one "${NAME}" "${ODIR}" && DONE+=("${NAME}")
done

if [[ "${#DONE[@]}" -gt 0 ]]; then
  log "===== [4] 2-way (Pearson + cosine) re-scoring ====="
  "${PYTHON}" "${NB_ROOT}/scripts/nda/nw4_official_twoway.py" \
    --gen-root "${OUT_ROOT}/arms" --arms "$(IFS=,; echo "${DONE[*]}")" \
    --subjects "${SUBJ}" --images-root "${IMAGES_ROOT}" \
    --out "${OUT_ROOT}/official_twoway_nwv6.json" --device cuda:0 \
    > "${LOG}/official_twoway_nwv6.log" 2>&1 || log "WARN twoway failed"
  tail -n 12 "${LOG}/official_twoway_nwv6.log" 2>/dev/null
fi

log "===== [5] the table, the 2x2 decomposition, and the verdict ====="
"${PYTHON}" - "${OUT_ROOT}" "${NB_ROOT}" "${STAG}" "${ALIGN_REP}" <<'PY'
import json, sys
from pathlib import Path
root, nb, stag, align_rep = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
KEYS = ["pixcorr", "ssim", "inception", "clip", "alex2", "alex5", "swav", "fid"]
LBL = {
    "E1_sem_s30":  "E1   no structure, cc1, sigma 3.0      (the floor)",
    "E2_varest_s30":"E2   + varest linear structure, cc3           [prev operating point]",
    "G3_head_s30": "G3   + per-modality head structure            [prev best structure]",
    "M6v5_a10b10": "M6v5 + selection, 11-cand pool                 [prev best deployable]",
    "A5_band_cc3": "A5   + GT depth/edge  (ORACLE structure)       [the target]",
    "A9_orc_cc3":  "A9   + GT image/depth/edge (ORACLE all)        [mechanism ceiling]",
    "W1_joint_all_s30":    "W1   JOINT img+depth+edge, sigma 3.0",
    "W2_joint_struct_s30": "W2   SEM + JOINT depth+edge, sigma 3.0   <- crosses E2/G3/M6",
    "W3_joint_all_s40":    "W3   JOINT img+depth+edge, sigma 4.0",
    "W4_gtimg_jstruct_s30":"W4   GT img + JOINT structure         <- 2x2 cell",
    "W5_jimg_gtstruct_s30":"W5   JOINT img + GT structure         <- 2x2 cell",
    "W6_gtimg_jstruct_s40":"W6   GT img + JOINT structure, sigma 4.0",
    "M6v6_a10b00": "M6v6 selection: a=1.0 b=0.0", "M6v6_a10b025": "M6v6 selection: a=1.0 b=0.25",
    "M6v6_a10b05": "M6v6 selection: a=1.0 b=0.5", "M6v6_a10b10": "M6v6 selection: a=1.0 b=1.0",
    "M6v6_a10b20": "M6v6 selection: a=1.0 b=2.0", "M6v6_a05b10": "M6v6 selection: a=0.5 b=1.0",
    "M6v6_a00b10": "M6v6 selection: a=0.0 b=1.0", "M6v6_a00b20": "M6v6 selection: a=0.0 b=2.0",
}
rows = {}
for f in (root/"arms").glob(f"*/eval/{stag}.json"):
    rows[f.parent.parent.name] = json.load(open(f))
for f in (nb/"outputs/nw5_s08/arms").glob(f"*/eval/{stag}.json"):
    rows.setdefault(f.parent.parent.name, json.load(open(f)))

print()
print("="*132)
print(f"NW-v6 {stag}: STH-Align joint stack + SCM-Loss + shared/specific fusion + uncertainty weighting")
print("="*132)
hdr = f"{'arm':<54}" + "".join(f"{k:>9}" for k in KEYS)
print(hdr); print("-"*len(hdr))
order = [k for k in LBL if k in rows] + [k for k in sorted(rows) if k not in LBL]
for a in order:
    r = rows[a]
    print(f"{LBL.get(a, a[:52]):<54}" + "".join(
        f"{r[k]:>9.4f}" if k != "fid" else f"{r[k]:>9.2f}" for k in KEYS))

def g(a, k): return rows[a][k] if a in rows and k in rows[a] else None

print()
print("="*132)
print("THE 2x2 DECOMPOSITION -- which branch owns the remaining gap?")
print("="*132)
cells = [("SEM  + GT structure    (A5, oracle structure)", "A5_band_cc3"),
         ("JOINT img + GT structure (W5)", "W5_jimg_gtstruct_s30"),
         ("GT img + JOINT structure (W4)", "W4_gtimg_jstruct_s30"),
         ("GT img + GT structure    (A9)", "A9_orc_cc3")]
for lbl, a in cells:
    v = g(a, "inception")
    print(f"  {lbl:<48} incep {v:.4f}" if v is not None else f"  {lbl:<48} (missing)")
a5, w5, w4, a9 = (g("A5_band_cc3", "inception"), g("W5_jimg_gtstruct_s30", "inception"),
                  g("W4_gtimg_jstruct_s30", "inception"), g("A9_orc_cc3", "inception"))
if None not in (a5, w5, w4, a9):
    print()
    print(f"  image branch, holding structure at GT:   W5 - A5 = {w5-a5:+.4f}")
    print(f"  structure branch, holding image at GT:   W4 - A5 = {w4-a5:+.4f}   "
          f"of a possible {a9-a5:+.4f}")
    closed = (w4 - a5) / (a9 - a5) if abs(a9 - a5) > 1e-9 else 0.0
    print(f"  => the new structure branch closes {closed:.1%} of the oracle-structure gap "
          f"(GT image held fixed on both sides)")
    if closed < 0.15:
        print("     The encoder-side upgrade did NOT move the structure branch materially.")
        print("     Report it and stop tuning it: the remaining headroom is the image branch")
        print(f"     ({w5-a5:+.4f} available there) and the eval protocol, not this axis.")
    else:
        print("     The encoder-side upgrade DID move the structure branch.  Scale it")
        print("     (subjects, seeds) before drawing the paper's conclusion.")

print()
print("="*132)
print("DEPLOYABLE vs THE TARGET  (recovery of the oracle-structure gain)")
print("="*132)
base = g("E1_sem_s30", "inception")
if a5 and base:
    print(f"  floor {base:.4f} -> oracle structure {a5:.4f}   gap {a5-base:+.4f}")
    for a in ("E2_varest_s30", "G3_head_s30", "W2_joint_struct_s30", "M6v5_a10b10",
              "W1_joint_all_s30", "W3_joint_all_s40", "W4_gtimg_jstruct_s30",
              "M6v6_a10b10", "M6v6_a10b00"):
        v = g(a, "inception")
        if v is not None:
            print(f"    {a:<22} incep {v:.4f}   recovers {(v-base)/(a5-base):>6.1%}")

print()
print("="*132)
print("BEST FINISHED ARM PER AXIS, and the like-for-like published bar")
print("="*132)
for k in KEYS:
    cand = [(v[k], a) for a, v in rows.items() if k in v]
    if not cand: continue
    val, a = (min(cand) if k in ("swav", "fid") else max(cand))
    print(f"  {k:<10} {val:>9.4f}   {a}")
print()
BAR = {"pixcorr": 0.166, "ssim": 0.409, "inception": 0.831, "clip": 0.903,
       "alex2": 0.818, "alex5": 0.913}
print("  vs CogCapPro sub-08 (0.166/0.409/0.831/0.903/0.818/0.913), 6 scored axes:")
for a in ("A5_band_cc3", "M6v5_a10b10", "W2_joint_struct_s30", "M6v6_a10b10",
          "M6v6_a10b00", "M6v6_a00b10"):
    r = rows.get(a)
    if not r: continue
    wins = [(k, r[k] > b) for k, b in BAR.items()]
    print(f"    {a:<22} {sum(w for _, w in wins)}/6   "
          + " ".join(f"{k[:4]}={'W' if w else 'l'}" for k, w in wins))

if align_rep.is_file():
    hd = json.load(open(align_rep))
    print()
    print("="*132)
    print("STH-Align grid (ranked by val dev_corr, held-out concepts) and the tiers")
    print("="*132)
    for m, grid in hd.get("grid", {}).items():
        lk = max((k for k in hd["linear"] if k.endswith("|" + m)),
                 key=lambda k: hd["linear"][k]["val"]["dev_corr"], default=None)
        if lk is None: continue
        lv = hd["linear"][lk]["val"]
        print(f"  -- {m}: linear control [{lk.split('|')[0]}, {hd['linear'][lk]['dim']}-d] "
              f"val_corr {lv['dev_corr']:.4f}  val_top1 {lv['dev_top1']*200:.1f}x")
        bestc = hd["head"].get(m, {}).get("config"); bestv = hd["head"].get(m, {}).get("variant")
        for r in grid:
            mk = " ***" if (r["config"] == bestc and r["variant"] == bestv) else ""
            ls = r.get("log_sigma")
            print(f"     {r['config']:<22} {r['variant']:<8} [{r['feat']:<6}] "
                  f"val_corr {r['val_dev_corr']:.4f} ({r['val_dev_corr']/lv['dev_corr']:.2f}x) "
                  f" val_top1 {r['val_dev_top1']*200:>5.1f}x  ep {r['epoch']:>3}  "
                  f"{r['params']/1e3:>5.0f}k"
                  + (f"  ls={ls:+.2f}" if ls is not None else "") + mk)
    print()
    for m, gate in hd.get("gate", {}).items():
        h = hd["head"][m]
        print(f"    {m:<6} {gate['tier'].upper():<7} val_corr x{gate['val_gain_corr']:.2f}  "
              f"val_top1 x{gate['val_gain_top1']:.2f}   [{h['config']}/{h['variant']}, "
              f"{h['feat']}]  test dev_corr {h['test']['dev_corr']:.4f} vs linear "
              f"{h['linear_test']['dev_corr']:.4f}")
PY

log "===== done ====="
