#!/bin/bash
# =============================================================================
# NW4 10-SUBJECT PIPELINE  (STAGE-driven; one arm per invocation)
# =============================================================================
# Answers the question the whole line of work has been building to: does the NW4
# semantic path, with everything we learned, beat published THINGS-EEG SOTA on all 10
# subjects at once?
#
# WHAT IS KEPT FROM EARLIER WORK, AND WHY
# ---------------------------------------
# The spatial pathway is deliberately NOT ours.  `sdedit_ll_full10` produced the init
# image and `uck` the depth map for all 10 subjects, and those are the exact inputs that
# generated the 0.7280 row we are trying to beat.  Swapping in the NW4 S1 spatial decoder
# would add an unvalidated variable to a run whose purpose is to test the SEMANTIC path.
# So: same init, same depth, same generator, same 28 steps / guidance 5.0 / strength 0.82
# / cn 0.40 / seed 42 as `x_hyb_1br`, which reproduced the published recipe on our chain
# at 0.6952.  Everything that differs is the conditioner.
#
# THE CONDITION PIPELINE, AND THE THREE FIXES IT CARRIES
# -----------------------------------------------------
# Measured on sub-08, each fix verified independently before being used here:
#
#  1. SELECTOR.  Checkpoints are chosen by `img_rsa` (corr between condition-condition
#     and target-target similarity trees) rather than `img_vs_cl`.  Cheap and free:
#     the same head selected this way scores RSA 0.2414 against 0.2085, and its images
#     measure 0.6757 against the shipped 0.6620.
#  2. CONCENTRATION.  `gem_calib` quantile-matches each row's concentration onto the
#     TRAIN bank's own distribution.  The raw head sits at rowcos 0.9380 where the real
#     bank is 0.6275 -- every row points the same way, so trials cannot differ and the
#     adapter renders one averaged appearance.  This one omitted step is worth +0.039
#     inception on its own (0.6228 -> 0.6620) and +0.053 clip.
#  3. ON-MANIFOLD.  `nw4_make_conds.py` can snap each row to its nearest TRAIN-bank row.
#     The oracle test motivates this: real CLIP embeddings fed straight to the generator
#     reach 0.7946, against our 0.6757, and the only difference is that the oracle's rows
#     ARE real embeddings.  Snapping also dominates the direct head on both measurable
#     axes at once (RSA 0.2973 vs 0.2414, nn_true 0.6634 vs 0.6389).  This is what A2 was
#     reaching for and missed: A2 mixed top-K neighbour DIRECTIONS when the defect was
#     the bank's CONCENTRATION, so it could not help however it was tuned.
#
# WHAT IS *NOT* USED, WITH THE MEASUREMENT THAT KILLED IT
# -------------------------------------------------------
#  * SPA at large weight.  `batch_rsa` raises RSA as designed (w_spa 0 -> 20 gives test
#    RSA 0.181 -> 0.315) but that does NOT convert: spa0 and spa5 differ by +0.047 RSA
#    and by -0.0013 inception (0.6757 vs 0.6744).  0.20 is the useful end and it is the
#    anchor+selector that got us there, so only w_spa=0 (control) and w_spa=20 (to
#    complete the curve on 10 subjects) are trained.
#  * Dense fusion (A3's fused control).  Fusing costs trial structure even after
#    calibration: fused RSA 0.1857 against the single branch's 0.2085.
#  * Multi-branch.  Single branch won by +0.0366 inception over fused in the 11-arm grid.
#  * S2's ridge projection.  It lowers RSA (0.1584 -> 0.1362 on the shipped head), so the
#    condition is taken straight from the S1 head.
#
# ARMS (4, run as parallel jobs; each loops all 10 subjects)
#   a_cal      calibrated,              strength 0.82 / cn 0.40   <- baseline candidate
#   a_raw      NOT calibrated,          strength 0.82 / cn 0.40   <- isolates the calib gain
#   a_ret1cal  snap-to-real + calib,    strength 0.82 / cn 0.40   <- on-manifold bet
#   a_hi       calibrated,              strength 0.92 / cn 0.28   <- SOTA-seeking
#
# WHY THESE FOUR
# --------------
# 1. SPA IS DROPPED, ON MEASUREMENT.  `batch_rsa` does exactly what it was designed to
#    (test RSA 0.181 -> 0.315 as w_spa goes 0 -> 20) but that RSA does not become images:
#        spa0_cal   RSA 0.2414  ->  incep 0.6757  clip 0.8030
#        spa5_cal   RSA 0.2884  ->  incep 0.6744  clip 0.8081
#        spa20_cal  RSA 0.3037  ->  incep 0.6672  clip 0.8043
#    +0.06 RSA, -0.009 inception.  So the 0.967 correlation that made RSA look like the
#    gate was driven by the range spanned by degenerate conditions (nw3's collapsed bank
#    at RSA 0.048) and the oracle (RSA 1.0), not by any local gradient worth following.
#    Training the spa20 heads would therefore cost ~35 min of GPU per subject to buy
#    nothing, so only the spa0 head is trained.
#
# 2. SINGLE-SUBJECT GAPS ARE NOISE.  Inception on 200 images has SE ~ 0.032 (2 sigma ~
#    0.065).  The 0.02 gap that drove a lot of sub-08 tuning -- hybrid 0.6952 against our
#    0.6757 -- is inside that, and the replicate spread agrees: the 11-arm grid gave
#    0.6228 / 0.6352 / 0.6593 for the SAME condition under different injection, and the
#    hybrid family spans 0.6324 to 0.7280.  So the arms kept here are the ones whose
#    effect was large enough to survive that floor (calibration +0.039, selector +0.014,
#    single-vs-fused +0.037), and the point of ten subjects is to cut the SE of the mean
#    to ~0.010 so the comparison against published numbers means something.
#
# 3. `a_hi` GOES WHERE THE SOTA BARS ARE MISSED.  On sub-08 we already clear ATM on clip
#    (0.803 vs 0.786), ssim (0.371 vs 0.345) and pixcorr (0.171 vs 0.160), and miss it on
#    inception (0.676 vs 0.734) and FID (189 vs D2-FOSA's 146).  Those are two ends of one
#    axis: strength 0.82 with cn 0.40 leans on the VAE-decoded init, which buys fidelity
#    (pixcorr/ssim) and costs realism (FID/inception).  `a_hi` moves to 0.92 / 0.28 to buy
#    realism back at some fidelity cost.  If it lands near ATM's inception and FID while
#    keeping pixcorr above 0.160 and ssim above 0.345, we hold all five at once, which is
#    the only form of "SOTA" worth claiming; if it cannot, the run says so with n=10
#    instead of with one subject's noise.
#
# 4. `a_ret1cal` IS KEPT DESPITE LOOKING WEAKER ON PAPER.  From the spa0 head it measures
#    RSA 0.2414 (cal) vs 0.2584 (ret1_cal) but vs_true 0.5711 vs 0.4612, so the axes we
#    can see do not favour it.  It stays because the axis we cannot see is the one the
#    oracle points at: the oracle is REAL bank rows and scores 0.7946, the best result
#    this chain has produced.  Snapping is the cheapest way to make our rows real, so if
#    on-manifold placement is worth anything, this is where it shows.
#
# PUBLISHED BARS THIS RUN IS SCORED AGAINST (10-subject protocol)
#   ATM         pixcorr 0.160  ssim 0.345  alex2 0.776  alex5 0.866  incep 0.734  clip 0.786
#   CogCap      pixcorr 0.150  ssim 0.347  alex2 0.754  incep 0.669  clip 0.715  (10subj mean)
#   CogCapPro   pixcorr 0.163  ssim 0.398                      incep 0.779  clip 0.830
#   D2-FOSA     pixcorr 0.193  ssim 0.350  FID 146.33
#   MB2C        pixcorr 0.188  ssim 0.333  FID 163.94
#   our prior best (HCMA 10subj): pooled FID 129.47, pixcorr 0.0886, ssim 0.2262
#
# Usage:
#   STAGE=train run_nw4_10subj.sh                  # S1 heads for all subjects
#   STAGE=arm ARM=a_spa0_cal run_nw4_10subj.sh     # cond -> gen -> eval -> pooled
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."                     # -> NeuroBridge/
NB_ROOT="$(pwd)"

PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export HOME="${XDG_CACHE_HOME}"

STAGE="${STAGE:-arm}"
SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9,10}"
# name:w_spa -- w_spa=0 is the control, 20 is the top of the swept range
HEADS="${HEADS:-spa0:0.0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-30}"

OUT_ROOT="${OUT_ROOT:-${NB_ROOT}/outputs/nw4_10s}"
PROMPTS="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
REF_BANK="${NB_ROOT}/outputs/gem/cond_cache/clip_img1024_train.npy"
TEST_BANK="${NB_ROOT}/outputs/gem/cond_cache/clip_img1024_test.npy"
# subject-independent counts only (all subjects are 1654x10 rows in the same order)
META_DIR="${NB_ROOT}/outputs/nda_ss/sub-08/clip_text"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/shared"

log(){ echo "[$(date -Iseconds)] $*"; }

# ---- one-time preflight: every shared asset and flag must exist -------------
require(){ [[ -e "$1" ]] || { log "FATAL missing $1"; exit 1; }; }
require "${PROMPTS}"; require "${REF_BANK}"; require "${TEST_BANK}"
require "${META_DIR}/train/meta.json"
require "${NB_ROOT}/outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512/199.png"
require "${NB_ROOT}/outputs/uck/sub-08/full/spatial/pred_depth_rgb_512/199.png"
for f in nw4_s1_train.py nw4_make_conds.py gem_calib.py generate_layered_decode.py \
         eval_official_seven_dir.py; do
  require "${NB_ROOT}/scripts/nda/${f}"
done

arm_spec(){  # arm -> "<head> <variant>"
  case "$1" in
    a_cal)      echo "spa0  cal" ;;       # calibrated          <- measured baseline
    a_raw)      echo "spa0  direct" ;;    # NOT calibrated      <- isolates calib gain
    a_ret1cal)  echo "spa0  ret1_cal" ;;  # snap-to-real + cal  <- on-manifold bet
    a_hi)       echo "spa0  cal" ;;       # calibrated + more generation freedom
    *) echo "" ;;
  esac
}

# Per-arm generator overrides "<strength> <cn_scale>".  Defaults match the recipe that
# reproduced the published 0.7280 row (strength 0.82, cn 0.40).  `a_hi` moves along the
# fidelity-vs-realism axis, which is the one lever that trades directly between the axes
# we already lead on and the ones we do not -- see the header for the reasoning.
arm_gspec(){
  case "$1" in
    a_hi) echo "0.92 0.28" ;;
    *)    echo "0.82 0.40" ;;
  esac
}

# =============================================================================
# STAGE=train -- S1 heads for every subject (fast, ~3-4 min per subject per head)
# =============================================================================
if [[ "${STAGE}" == "train" ]]; then
  IFS=',' read -ra SID_ARR <<< "${SUBJECTS}"
  for SID in "${SID_ARR[@]}"; do
    STAG=$(printf "sub-%02d" "${SID}")
    for HS in ${HEADS}; do
      HNAME="${HS%%:*}"; WSPA="${HS##*:}"
      D="${OUT_ROOT}/${STAG}/s1_${HNAME}"
      if [[ -f "${D}/conds/z_img_test.npy" ]]; then log "S1 ${STAG}/${HNAME} present - skip"; continue; fi
      log "===== S1 ${STAG} head=${HNAME} w_spa=${WSPA} (select=img_rsa, w_anchor=400) ====="
      if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/nw4_s1_train.py" \
          --out "${D}" --test-subject "${SID}" --device cuda:0 \
          --epochs "${EPOCHS}" \
          --w-inst-img 1.0 --w-inst-vith 1.0 --w-inst-attr 1.0 \
          --w-anchor 400.0 --w-spa "${WSPA}" \
          --w-depth 1.0 --w-vae 1.0 --w-recon 0.2 \
          --select-metric img_rsa \
          --clip-text-dir "${META_DIR}" \
          > "${OUT_ROOT}/logs/s1_${STAG}_${HNAME}.log" 2>&1; then
        log "WARN S1 ${STAG}/${HNAME} failed (see logs/s1_${STAG}_${HNAME}.log)"
        tail -n 12 "${OUT_ROOT}/logs/s1_${STAG}_${HNAME}.log" || true
      else
        # report the head's own RSA so a degenerate subject is visible before generation
        "${PYTHON}" - "${D}/conds/z_img_test.npy" "${STAG}/${HNAME}" <<'PY' 2>/dev/null || true
import sys, numpy as np
sys.path.insert(0, "scripts/nda")
from nw4_s2_project import l2n, row2mean
B = l2n(np.load("outputs/gem/cond_cache/clip_img1024_test.npy").astype(np.float32))
z = l2n(np.load(sys.argv[1]).astype(np.float32))
cz, ct = z @ z.T, B @ B.T
iu = np.triu_indices(len(cz), 1)
print(f"[head] {sys.argv[2]}: RSA={np.corrcoef(cz[iu],ct[iu])[0,1]:.4f} "
      f"rowcos={row2mean(z):.4f} vs_true={float((z*B).sum(1).mean()):.4f}")
PY
      fi
    done
  done
  log "STAGE=train done"
  exit 0
fi

# =============================================================================
# STAGE=arm -- conditions -> generation -> seven-metric eval, for all subjects
# =============================================================================
ARM="${ARM:-}"
read -r AHEAD AVAR < <(arm_spec "${ARM}")
read -r GSTR GCN < <(arm_gspec "${ARM}")
GSTR="${GSTR:-0.82}"; GCN="${GCN:-0.40}"
if [[ -z "${AHEAD}" ]]; then
  log "FATAL unknown ARM='${ARM}'. Known: a_cal a_raw a_ret1cal a_hi"
  exit 1
fi
log "===== ARM=${ARM} head=${AHEAD} variant=${AVAR} strength=${GSTR} cn=${GCN} subjects=${SUBJECTS} ====="
ADIR="${OUT_ROOT}/arms/${ARM}"
mkdir -p "${ADIR}"/{conds,gen,eval}

IFS=',' read -ra SID_ARR <<< "${SUBJECTS}"
NSKIP=0; NFAIL=0
for SID in "${SID_ARR[@]}"; do
  STAG=$(printf "sub-%02d" "${SID}")
  HEAD="${OUT_ROOT}/${STAG}/s1_${AHEAD}/conds/z_img_test.npy"
  if [[ ! -f "${HEAD}" ]]; then log "WARN ${STAG}: no S1 head (${HEAD}) - skip"; NFAIL=$((NFAIL+1)); continue; fi

  INIT="${NB_ROOT}/outputs/sdedit_ll_full10/${STAG}/vae_head/pred_lowlevel_rgb_512"
  DEPTH="${NB_ROOT}/outputs/uck/${STAG}/full/spatial/pred_depth_rgb_512"
  if [[ ! -f "${INIT}/199.png" || ! -f "${DEPTH}/199.png" ]]; then
    log "WARN ${STAG}: missing proven init/depth - skip"; NFAIL=$((NFAIL+1)); continue
  fi

  COND_JSON="${ADIR}/conds/${STAG}/${STAG}_conds.json"
  COND="${ADIR}/conds/${STAG}/${AVAR}_test.npy"

  # ---- 1. conditions ------------------------------------------------------
  # NB the output directory is PER SUBJECT.  Writing all ten subjects into one
  # directory would make the files indistinguishable (`cal_test.npy` from whichever
  # subject ran last), and any fallback that picks "the newest *_test.npy" would then
  # silently generate one subject's images from another subject's condition.
  if [[ ! -f "${COND}" ]]; then
    log "--- ${STAG}: build condition (${AVAR})"
    if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/nw4_make_conds.py" \
        --head "${HEAD}" --out-dir "${ADIR}/conds/${STAG}" \
        --variants "${AVAR}" --json "${COND_JSON}" --tag "${STAG}" \
        >> "${OUT_ROOT}/logs/${ARM}_${STAG}_conds.log" 2>&1; then
      log "WARN ${STAG}: condition build failed"; NFAIL=$((NFAIL+1)); continue
    fi
    if [[ ! -f "${COND}" ]]; then
      log "WARN ${STAG}: condition builder produced no ${COND}"; NFAIL=$((NFAIL+1)); continue
    fi
  fi

  RSA=$( "${PYTHON}" - "${COND}" <<'PY' 2>/dev/null || echo "nan"
import sys, numpy as np
sys.path.insert(0,"scripts/nda")
from nw4_s2_project import l2n
B=l2n(np.load("outputs/gem/cond_cache/clip_img1024_test.npy").astype(np.float32))
z=l2n(np.load(sys.argv[1]).astype(np.float32))
cz,ct=z@z.T,B@B.T
iu=np.triu_indices(len(cz),1)
print(f"{np.corrcoef(cz[iu],ct[iu])[0,1]:.4f}")
PY
)
  log "--- ${STAG}: condition RSA=${RSA}"

  # ---- 2. generation -----------------------------------------------------
  GEN="${ADIR}/gen/${STAG}"
  if [[ -f "${GEN}/generated/199.png" ]]; then
    log "--- ${STAG}: gen present - skip"
  else
    log "--- ${STAG}: generate"
    if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/generate_layered_decode.py" \
        --cond-npys "${COND}" --branch-spec "all:0.9" \
        --prompts-json "${PROMPTS}" --output-dir "${GEN}" --tag "${ARM}_${STAG}" \
        --pipeline base --depth-rgb-dir "${DEPTH}" --lowlevel-rgb-dir "${INIT}" \
        --cn-scale "${GCN}" --strength "${GSTR}" --gen-steps 28 --gen-guidance 5.0 \
        --gen-size 512 --seed "${SEED}" --device cuda:0 \
        > "${OUT_ROOT}/logs/${ARM}_${STAG}_gen.log" 2>&1; then
      log "WARN ${STAG}: generation failed"; NFAIL=$((NFAIL+1)); continue
    fi
  fi

  # ---- 3. seven-metric eval ----------------------------------------------
  EV="${ADIR}/eval/${STAG}.json"
  if [[ -f "${EV}" ]]; then
    log "--- ${STAG}: eval present - skip"
  else
    log "--- ${STAG}: eval"
    if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
        --gen-dir "${GEN}/generated" --output-json "${EV}" --tag "${ARM}_${STAG}" \
        --images-root "${IMAGES_ROOT}" --device cuda:0 \
        > "${OUT_ROOT}/logs/${ARM}_${STAG}_eval.log" 2>&1; then
      log "WARN ${STAG}: eval failed"; NFAIL=$((NFAIL+1)); continue
    fi
  fi
  NSKIP=$((NSKIP+1))
done
log "arm ${ARM}: ${NSKIP} subjects complete, ${NFAIL} skipped/failed"

# ---- 4. pooled report over the subjects that finished ----------------------
"${PYTHON}" - "${ADIR}/eval" "${ARM}" "${OUT_ROOT}/arms/${ARM}_summary" <<'PY'
import json, sys
from pathlib import Path
import numpy as np

evd, arm, outstem = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
files = sorted(evd.glob("sub-*.json"))
if not files:
    print(f"[pooled] {arm}: no per-subject evals found -> nothing to pool"); raise SystemExit(0)

MET = ["inception", "clip", "pixcorr", "ssim", "fid", "alex2", "alex5"]
# published 10-subject / primary-table bars; every one of these is on THINGS-EEG
SOTA = {
    "ATM":       {"pixcorr": 0.160, "ssim": 0.345, "alex2": 0.776, "alex5": 0.866, "inception": 0.734, "clip": 0.786},
    "CogCap":    {"pixcorr": 0.150, "ssim": 0.347, "alex2": 0.754, "alex5": 0.623, "inception": 0.669, "clip": 0.715},
    "CogCapPro": {"pixcorr": 0.163, "ssim": 0.398, "inception": 0.779, "clip": 0.830},
    "D2-FOSA":   {"pixcorr": 0.193, "ssim": 0.350, "fid": 146.33},
    "MB2C":      {"pixcorr": 0.188, "ssim": 0.333, "fid": 163.94},
    "our HCMA prior": {"pixcorr": 0.0886, "ssim": 0.2262, "fid": 129.47},
}

rows = {}
for f in files:
    d = json.loads(f.read_text())
    rows[f.stem] = {m: d.get(m) for m in MET}
    rows[f.stem]["fid"] = d.get("fid")

print()
print("=" * 96)
print(f"NW4 10-SUBJECT POOLED REPORT -- arm {arm}  (n={len(rows)} subjects)")
print("=" * 96)
print("per subject:")
print(f"  {'subject':<10}" + "".join(f"{m:>10}" for m in MET))
for s in sorted(rows):
    print(f"  {s:<10}" + "".join(
        (f"{rows[s][m]:>10.4f}" if isinstance(rows[s][m], (int, float)) else f"{'--':>10}")
        for m in MET))

mean = {}
for m in MET:
    v = [rows[s][m] for s in rows if isinstance(rows[s].get(m), (int, float))]
    mean[m] = float(np.mean(v)) if v else float("nan")
sd = {}
for m in MET:
    v = [rows[s][m] for s in rows if isinstance(rows[s].get(m), (int, float))]
    sd[m] = float(np.std(v)) if v else float("nan")

print()
print("mean over subjects (the protocol CogCap/CogCapPro report):")
print(f"  {'metric':<12}{'mean':>10}{'std':>10}")
for m in MET:
    print(f"  {m:<12}{mean[m]:>10.4f}{sd[m]:>10.4f}")

print()
print("vs published THINGS-EEG SOTA  (+ = ours better; FID inverted)")
hdr = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "fid"]
print(f"  {'method':<18}" + "".join(f"{h:>11}" for h in hdr))
print(f"  {'NW4 (' + arm + ')':<18}" + "".join(
    (f"{mean[h]:>11.4f}" if isinstance(mean.get(h), float) and mean[h] == mean[h] else f"{'--':>11}")
    for h in hdr))
for name, bar in SOTA.items():
    print(f"  {name:<18}" + "".join(
        (f"{bar[h]:>11.4f}" if h in bar else f"{'--':>11}") for h in hdr))

print()
beats = []
for name, bar in SOTA.items():
    wins, losses = [], []
    for h, b in bar.items():
        a = mean.get(h)
        if not isinstance(a, float) or a != a:
            continue
        if h == "fid":
            (wins if a < b else losses).append(f"{h} {a:.3f} vs {b:.3f}")
        else:
            (wins if a > b else losses).append(f"{h} {a:.4f} vs {b:.4f}")
    if wins and not losses:
        beats.append(f"{name}: DOMINATES ({', '.join(wins)})")
    elif wins:
        print(f"  vs {name:<14} wins: {', '.join(wins)}")
        print(f"  {'':<17} loses: {', '.join(losses)}")
    else:
        print(f"  vs {name:<14} loses on every shared metric")
for b in beats:
    print(f"  [PARETO] {b}")

# the honest headline: which axes are SOTA and which are not, against the union of bars
print()
print("headline, against the union of published bars:")
for m in ("pixcorr", "ssim", "inception", "clip", "fid"):
    vals = [bar[m] for bar in SOTA.values() if m in bar]
    if not vals or not isinstance(mean.get(m), float) or mean[m] != mean[m]:
        continue
    best_pub = min(vals) if m == "fid" else max(vals)
    ok = mean[m] < best_pub if m == "fid" else mean[m] > best_pub
    print(f"  {m:<10} ours {mean[m]:>8.4f}  best published {best_pub:>8.4f}  "
          f"{'SOTA' if ok else 'below'}")

rep = {"stage": "nw4_10subj_pooled", "arm": arm, "n": len(rows),
       "per_subject": rows, "mean": mean, "std": sd, "sota_bars": SOTA}
outstem.parent.mkdir(parents=True, exist_ok=True)
outstem.with_suffix(".json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
print(f"\n[pooled] wrote {outstem.with_suffix('.json')}")
PY

cp -f "${ADIR}/eval"/*.json "${OUT_ROOT}/shared/" 2>/dev/null || true
log "arm ${ARM} done"
