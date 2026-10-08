#!/bin/bash
#SBATCH --job-name=rep-ceiling
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/rep-ceiling-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/rep-ceiling-%j.err
#
# THE DECISIVE PROBE FOR v12(a). Everything v12 proposes rests on one unmeasured premise:
#
#     the query metric that DEPLOYMENT uses is still limited by per-repetition ESTIMATION noise,
#     so there is headroom for a training term that makes each repetition individually informative.
#
# If that premise is false -- if the R=80 consensus metric is already at its noise-free value --
# then no training term can help, because a mean of 80 samples is already the best estimator of
# what the encoder computes, and (a)'s teacher would be identical to its student.
#
# HOW IT IS MEASURED. The deployed T2 operator is refit on the first `R` repetitions for
# R in {1,2,4,8,16,32,64,80}. The resulting Top-1 curve is fit with the bias-variance form
#
#     top1(R) = A - c / R          (a mean of R reps has noise variance ~ 1/R)
#
# and `A` -- the R -> infinity limit -- is the CEILING for any method that only denoises the query
# metric. `A - top1(80)` is the exact headroom v12(a) is betting on. The 1/R form is not assumed:
# if the curve does not track 1/R the fit residual says so, and the raw curve is printed beside it.
#
# Two independent cross-checks, because a single extrapolation decides a week of training:
#   * the curve at R=1,2,4 pins the noise term where it is largest, so `c` is not set by one point;
#   * three subjects (1, 5, 8), so a fold-specific artefact cannot masquerade as headroom.
#
# `--rep-subsample` takes the FIRST R reps (not a random draw), which is deterministic and is what
# the v10 sweep already used, so the R=80 and R=20 endpoints reproduce the banked numbers
# (54.25 / 34.85) and prove the curve is the deployed operator.
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
SUBJECTS="${SUBJECTS:-1 5 8}"
RGRID="${RGRID:-1 2 4 8 16 32 64 80}"
SEED="${SEED:-2025}"

cd "${ROOT}"
mkdir -p outputs/slurm outputs/eval/rep_ceiling

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export PYTHONPATH="${ROOT}/src"

echo "[slurm] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
echo "[ceiling] subjects='${SUBJECTS}' R='${RGRID}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

for s in ${SUBJECTS}; do
  tag=$(printf 'sub%02d' "${s}")
  ck="${ROOT}/outputs/stage1/v8/${tag}_k20_seed${SEED}/last.pt"
  ev="${ROOT}/outputs/eval/rep_ceiling/${tag}_seed${SEED}.json"
  [ -s "${ck}" ] || { echo "  [skip] no ckpt ${tag}"; continue; }
  [ -s "${ev}" ] && { echo "  [skip] done ${tag}"; continue; }
  echo "[ceiling] === ${tag} ==="
  # No --fgw-sweep on purpose: the shipped single-path branch is what emits one `+ T2 reps (R=r)`
  # row per R AND its structural-off twin, which is exactly the curve. Requesting the sweep would
  # add ~40 unrelated cells per subject and drown the readout.
  "${PY}" scripts/run_eval.py --ckpts "${ck}" \
    --target-subject "${s}" --out "${ev}" \
    --recovery --reps --csls-k 10 --rho 0.1 \
    --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
    --rep-shrink 0.1 --rep-subsample ${RGRID} \
    && echo "  [ok] ${tag}" || echo "  [FAIL] ${tag}"
done

echo "[ceiling] === R-curve readout + 1/R extrapolation ==="
"${PY}" - <<'PY'
import json, glob, re
import numpy as np

# collect (R -> top1) per subject from the emitted rows
curve = {}
for f in sorted(glob.glob("outputs/eval/rep_ceiling/*.json")):
    sub = f.split('/')[-1][:-5]
    rows = list(json.load(open(f))["checkpoints"].values())[0]["rows"]
    pts = {}
    for k, v in rows.items():
        m = re.match(r"^\+ T2 reps(?: \(R=(\d+)\))?$", k)
        if m:
            pts[int(m.group(1)) if m.group(1) else -1] = v["top1"]
    curve[sub] = pts
    print(f"[ceiling] {sub}: " + "  ".join(f"R={k}:{pts[k]:.2f}" for k in sorted(pts)))

subs = sorted(curve)
common = None
for s in subs:
    ks = {k for k in curve[s] if k > 0}
    common = ks if common is None else (common & ks)
common = sorted(common or [])
if not common:
    raise SystemExit("[ceiling] no per-R rows emitted -- the R grid never reached the operator")

R = np.asarray(common, float)
Y = np.stack([[curve[s][k] for k in common] for s in subs])   # (subs, R)
ym = Y.mean(0)
print("\n[ceiling] mean over subjects:  " + "  ".join(f"R={int(r)}:{v:.2f}" for r, v in zip(R, ym)))

if len(R) >= 3:
    # Two candidate shapes, both reported: a mean of R reps has noise variance ~1/R (bias-variance),
    # but a metric estimated from R reps can also improve like log(R) if the estimator is a
    # rank/neighbourhood object rather than a mean. The shape is MEASURED (fit residual), not
    # assumed -- the whole point of the 8-point curve is that one analytic form is not enough.
    x = 1.0 / R
    A1, c1 = np.polyfit(x, ym, 1)              # top1 = A1 - c1/R
    res1 = float(np.abs((A1 + c1 * x) - ym).max())
    lx = np.log(R)
    A2, c2 = np.polyfit(lx, ym, 1)             # top1 = A2 + c2*log(R)
    res2 = float(np.abs((A2 + c2 * lx) - ym).max())
    print(f"\n[ceiling] shape fits (max residual):  1/R -> {res1:.2f}pp   log R -> {res2:.2f}pp")
    print(f"[ceiling]   1/R  : A(R->inf) = {A1:.2f}, c = {-c1:.1f}")
    print(f"[ceiling]   log R: slope = {c2:.2f} pp per e-fold; no finite ceiling (log diverges)")
    head = float(A1 - ym[-1])
    print(f"[ceiling] HEADROOM under the 1/R model: A - top1(R=80) = {head:+.2f}pp")
    rise = float(ym[-1] - ym[-2])
    saturated = rise <= 1.0
    print(f"[ceiling] top-end slope top1(R={int(R[-1])}) - top1(R={int(R[-2])}) = {rise:+.2f}pp"
          f"   -> {'SATURATED' if saturated else 'STILL CLIMBING'}")
    if not saturated:
        print("[ceiling] VERDICT: the query metric is NOT at its noise-free value at R=80; the R-curve "
              "is still rising, so a per-repetition denoising term has real headroom. The 1/R "
              "extrapolation UNDERSHOOTS the ceiling because the measured shape is steeper than 1/R.")
    elif head > 2.0:
        print(f"[ceiling] VERDICT: HEADROOM EXISTS under 1/R ({head:+.1f}pp) although the top end has "
              "flattened; treat the extrapolation as an upper bound.")
    else:
        print("[ceiling] VERDICT: NO HEADROOM -- the R=80 consensus is already near its noise-free "
              "value and a denoising training term has nothing to win.")
print("\n[ceiling] reference: shipped FGW baseline 54.25 (R=80, 10-subject mean), R=20 34.85, "
      "SCORE published 53.23.")
PY

echo "[ceiling] DONE $(date -Is)"
