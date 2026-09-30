#!/bin/bash
#SBATCH --job-name=nw4_10s_all
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=11:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_10s_all_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_10s_all_%j.err
#
# ONE JOB, WHOLE OVERNIGHT RUN.  Consumes a single GPU and does everything in order:
# S1 heads for all ten subjects, then the four arms back to back, each ending in its own
# pooled SOTA comparison.
#
# BUDGET (measured, not guessed):
#   S1 training      ~2 min per subject   ->  ~20 min for ten subjects
#   gen + eval       ~7 min per subject   ->  ~70 min per arm
#   4 arms                                -> ~280 min
#   total                                 -> ~5 h, inside the 11 h limit
# `run_nw4_10subj.sh` is stage-driven and skips anything already on disk, so if this job
# dies the same submission resumes instead of redoing work.
#
# ARM ORDER IS DELIBERATE -- by size of the effect being tested, largest first, so a
# truncated run still answers the question that matters most:
#
#   1. a_cal      calibrated, strength 0.82 / cn 0.40
#        The baseline candidate.  Calibration is the single largest verified effect on
#        this problem (+0.0392 inception, +0.0391 clip on sub-08) and the step that was
#        simply missing from the earlier pipeline; the raw head sits at rowcos 0.9380
#        against the real bank's 0.6275, i.e. every row points the same way and trials
#        cannot differ.  Expected to clear ATM on clip, ssim and pixcorr and to trail on
#        inception and FID.
#   2. a_hi       calibrated, strength 0.92 / cn 0.28
#        The SOTA-seeking arm.  Inception (0.676 vs ATM 0.734) and FID (189 vs D2-FOSA
#        146) are the two bars we miss, and they sit at the opposite end of one axis from
#        pixcorr/ssim where we lead: strength 0.82 / cn 0.40 leans on the VAE-decoded init
#        for fidelity.  This arm buys realism back.  If it reaches ATM's inception and FID
#        while holding pixcorr > 0.160 and ssim > 0.345, all five bars fall at once.
#   3. a_raw      NOT calibrated, strength 0.82 / cn 0.40
#        Ablation, same head as a_cal, to price the calibration step at n=10 rather than
#        at n=1 where its standard error is ~0.032 and cannot settle anything.
#   4. a_ret1cal  snap-to-real + calibration, strength 0.82 / cn 0.40
#        The on-manifold bet.  The oracle test -- real CLIP embeddings fed straight to the
#        generator -- scores 0.7946 on this chain against our 0.6757, and the only thing
#        separating them is that the oracle's rows ARE real embeddings.  Snapping each row
#        to its nearest TRAIN-bank row is the cheapest way to make ours real; it needs no
#        test targets and no labels.  This is what A2 was reaching for and missed (A2
#        mixed neighbour DIRECTIONS when the defect was bank CONCENTRATION).
#
# WHAT IS NOT RUN HERE, AND WHY (so the omission is not mistaken for an oversight)
#   * SPA (batch_rsa).  It works exactly as designed -- test RSA 0.181 -> 0.315 as w_spa
#     goes 0 -> 20 -- but it does not convert into image quality:
#         spa0_cal   RSA 0.2414 -> incep 0.6757     spa20_cal  RSA 0.3037 -> incep 0.6672
#     +0.06 RSA for -0.009 inception.  The +0.967 RSA/inception correlation that made RSA
#     look like the gate was driven by the space between degenerate banks (nw3 at 0.048)
#     and the oracle (1.0), i.e. it is a range artefact with no usable local gradient.
#   * Multi-branch and dense fusion.  Single branch beat fused by +0.0366 inception, and
#     fusing costs trial structure even after calibration (fused RSA 0.1857 vs single
#     0.2085).
#   * S2's ridge projection.  It lowers RSA (0.1584 -> 0.1362), so the condition is taken
#     straight from the S1 head.
#
# PUBLISHED BARS SCORED IN THE REPORT (THINGS-EEG)
#   ATM       pixcorr 0.160  ssim 0.345  alex2 0.776  alex5 0.866  incep 0.734  clip 0.786
#   CogCap    pixcorr 0.150  ssim 0.347  alex2 0.754  alex5 0.623  incep 0.669  clip 0.715
#   CogCapPro pixcorr 0.163  ssim 0.398                      incep 0.779  clip 0.830
#   D2-FOSA   pixcorr 0.193  ssim 0.350  FID 146.33
#   MB2C      pixcorr 0.188  ssim 0.333  FID 163.94
#   our prior best (HCMA 10subj, pooled FID): 129.47, pixcorr 0.0886, ssim 0.2262
#
# A NOTE ON HOW TO READ THE RESULT.  Every bar except FID is a HIGHER-is-better score, and
# on n=10 the standard error of the mean drops to ~0.010 (against ~0.032 per subject), so
# differences of ~0.02 become meaningful where on sub-08 alone they were not.  The
# replicate spread that forced this: the 11-arm grid returned 0.6228 / 0.6352 / 0.6593 for
# the SAME condition under different injection, and the hybrid family spans 0.6324-0.7280.

set -uo pipefail
cd /project/peilab/why/NeuroBridge

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9,10}"
export HEADS="${HEADS:-spa0:0.0}"
export OUT_ROOT="${OUT_ROOT:-/project/peilab/why/NeuroBridge/outputs/nw4_10s}"
ARMS="${ARMS:-a_cal a_hi a_raw a_ret1cal}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/shared"

echo "=============================================================="
echo " NW4 10-subject single-job run   job=${SLURM_JOB_ID:-local}  node=$(hostname)"
echo " subjects=${SUBJECTS}"
echo " arms=${ARMS}"
echo " start=$(date -Iseconds)"
echo "=============================================================="

# ------------------------------------------------------------------ fail fast
# A GPU node whose driver predates the torch build silently runs on CPU at 10-20x the
# cost, which on a 5 h job means waking up to nothing.  Refuse to start instead.
"${PYTHON}" - <<'PY' || { echo "[FATAL] no usable CUDA device on $(hostname); refusing to run for hours on CPU"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print(f"[FATAL] torch {torch.__version__} sees no CUDA device "
          f"(built for {torch.version.cuda})"); sys.exit(1)
print(f"[gpu] {torch.cuda.get_device_name(0)}  torch={torch.__version__} cuda={torch.version.cuda}")
PY

# ------------------------------------------------------------------ stage 1
echo
echo "########## STAGE 1/2: S1 heads for all subjects ##########"
echo "stage=  train started=$(date -Iseconds)"
STAGE=train bash scripts/nda/run_nw4_10subj.sh
echo "stage=train finished=$(date -Iseconds)"

# How many subjects actually produced a head?  A missing head costs one subject in every
# arm, so surface it here rather than as four identical silent skips later.
NHEAD=$(ls -d "${OUT_ROOT}"/sub-*/s1_*/conds/z_img_test.npy 2>/dev/null | wc -l)
echo "[check] S1 heads on disk: ${NHEAD} (expected $(echo "${SUBJECTS}" | tr ',' '\n' | wc -l) x $(echo ${HEADS} | wc -w))"

# ------------------------------------------------------------------ stage 2
echo
echo "########## STAGE 2/2: generation + eval, ${ARMS} ##########"
for A in ${ARMS}; do
  echo
  echo "=============================================================="
  echo " ARM ${A}   started=$(date -Iseconds)"
  echo "=============================================================="
  STAGE=arm ARM="${A}" bash scripts/nda/run_nw4_10subj.sh
  echo "ARM ${A} finished=$(date -Iseconds)"
  # per-arm pooled report is written by the orchestrator; echo the verdict immediately so
  # the tail of this log is readable in the morning without digging
  if [[ -f "${OUT_ROOT}/arms/${A}_summary.json" ]]; then
    "${PYTHON}" - "${OUT_ROOT}/arms/${A}_summary.json" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
m = d["mean"]
print(f"  [pooled n={d['n']}] " + "  ".join(
    f"{k}={m[k]:.4f}" for k in ("inception", "clip", "pixcorr", "ssim", "fid")
    if isinstance(m.get(k), float) and m[k] == m[k]))
PY
  else
    echo "  [pooled] no summary produced for this arm"
  fi
done

# ------------------------------------------------------------------ final table
echo
echo "########## ALL ARMS: pooled over 10 subjects ##########"
"${PYTHON}" - "${OUT_ROOT}/arms" <<'PY'
import json, sys
from pathlib import Path
import numpy as np

arms_dir = Path(sys.argv[1])
summary = {}
for f in sorted(arms_dir.glob("*_summary.json")):
    summary[f.stem.replace("_summary", "")] = json.loads(f.read_text())

if not summary:
    print("no arm summaries found"); raise SystemExit(0)

# published THINGS-EEG numbers, for the same table
SOTA = {
    "ATM":          {"pixcorr": 0.160, "ssim": 0.345, "alex2": 0.776, "alex5": 0.866, "inception": 0.734, "clip": 0.786},
    "CogCap":       {"pixcorr": 0.150, "ssim": 0.347, "alex2": 0.754, "alex5": 0.623, "inception": 0.669, "clip": 0.715},
    "CogCapPro":    {"pixcorr": 0.163, "ssim": 0.398, "inception": 0.779, "clip": 0.830},
    "D2-FOSA":      {"pixcorr": 0.193, "ssim": 0.350, "fid": 146.33},
    "MB2C":         {"pixcorr": 0.188, "ssim": 0.333, "fid": 163.94},
    "our HCMA prev":{"pixcorr": 0.0886, "ssim": 0.2262, "fid": 129.47},
}
cols = ["inception", "clip", "pixcorr", "ssim", "alex2", "alex5", "fid"]

print()
print(f"{'run':<20}" + "".join(f"{c:>10}" for c in cols))
print("-" * (20 + 10 * len(cols)))
print(f"{'NW4 10subj (Ours)':<20}" + "".join(
    (f"{summary['a_cal']['mean'][c]:>10.4f}" if c in summary.get('a_cal', {}).get('mean', {})
     and isinstance(summary['a_cal']['mean'].get(c), float) else f"{'--':>10}")
    for c in cols))
print("-" * (20 + 10 * len(cols)))
for name, bar in SOTA.items():
    print(f"{name:<20}" + "".join(
        (f"{bar[c]:>10.4f}" if c in bar else f"{'--':>10}") for c in cols))

print()
print("per-arm means (n = subjects completed):")
print(f"  {'arm':<12}{'n':>4}" + "".join(f"{c:>10}" for c in cols))
for a in sorted(summary):
    m, n = summary[a]["mean"], summary[a]["n"]
    print(f"  {a:<12}{n:>4}" + "".join(
        (f"{m[c]:>10.4f}" if isinstance(m.get(c), float) and m[c] == m[c] else f"{'--':>10}")
        for c in cols))

# ---- ablated effects, the thing that makes n=10 worth the night -------------------
print()
print("ablated effects at n=10 (largest-tested first):")
if "a_cal" in summary and "a_raw" in summary:
    d = {c: summary["a_cal"]["mean"].get(c, float("nan")) - summary["a_raw"]["mean"].get(c, float("nan"))
         for c in ("inception", "clip", "pixcorr", "ssim", "fid")}
    print(f"  calibration  (a_cal - a_raw)      " + "  ".join(f"{k}={v:+.4f}" for k, v in d.items()))
    print(f"     -> the step that was missing from the earlier pipeline; on sub-08 alone it")
    print(f"        measured +0.0392 inception, and n=1 could not tell whether that was real")
if "a_hi" in summary and "a_cal" in summary:
    d = {c: summary["a_hi"]["mean"].get(c, float("nan")) - summary["a_cal"]["mean"].get(c, float("nan"))
         for c in ("inception", "clip", "pixcorr", "ssim", "fid")}
    print(f"  realism knob (a_hi - a_cal)       " + "  ".join(f"{k}={v:+.4f}" for k, v in d.items()))
    print(f"     -> does moving strength 0.82->0.92 / cn 0.40->0.28 trade fidelity for realism")
    print(f"        as intended, and does it clear ATM's inception 0.734 / D2-FOSA's FID 146")
if "a_ret1cal" in summary and "a_cal" in summary:
    d = {c: summary["a_ret1cal"]["mean"].get(c, float("nan")) - summary["a_cal"]["mean"].get(c, float("nan"))
         for c in ("inception", "clip", "pixcorr", "ssim", "fid")}
    print(f"  on-manifold  (a_ret1cal - a_cal)  " + "  ".join(f"{k}={v:+.4f}" for k, v in d.items()))
    print(f"     -> worth checking against the oracle: real embeddings scored 0.7946 on this")
    print(f"        chain vs our 0.6757, and snapping is the only cheap way to close that")

# ---- SOTA verdict, per arm, against the union of bars -----------------------------
print()
print("SOTA verdict (against the union of published bars on each axis):")
best_pub = {}
for c in cols:
    vals = [b[c] for b in SOTA.values() if c in b]
    if vals:
        best_pub[c] = min(vals) if c == "fid" else max(vals)
for a in sorted(summary):
    m = summary[a]["mean"]
    verdict = {}
    for c, bp in best_pub.items():
        if not isinstance(m.get(c), float) or m[c] != m[c]:
            continue
        verdict[c] = (m[c] < bp) if c == "fid" else (m[c] > bp)
    ok = [c for c, v in verdict.items() if v]
    no = [c for c, v in verdict.items() if not v]
    print(f"  {a:<12} SOTA on {ok if ok else 'none'}")
    if no:
        print(f"  {'':<12} below on {no}")
PY

echo
echo "all done = $(date -Iseconds)"
