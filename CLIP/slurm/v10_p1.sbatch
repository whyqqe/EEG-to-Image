#!/bin/bash
#SBATCH --job-name=v10-p1
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v10-p1-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v10-p1-%j.err
#
# v10 STAGE 3 -- P1, THE SPECTRAL RANK PRIOR (docs/eeg2image_v10_m1_theory.md §10).
#
# THE TWO FALSIFICATIONS THAT DEFINE THE PROBLEM. They bracket the entire space of "use the
# source data", and this job only tests the one shape of source information that survives both.
#
#   M1 (job 645455) -- LEAKAGE. Source metric used with its INDICES participating: +15.4pp
#     Top-1, 30/30 folds positive. The permutation control (job 645593) destroyed the concept
#     correspondence while leaving the geometry intact and the gain flipped to -14.5pp, with
#     corr(de_src, di) collapsing 0.782 -> 0.001. `de_src[i,j]` carries concept i's identity in
#     its own subscript and `di` carries the gallery slot in its own, so the FGW optimum became
#     the IDENTITY plan -- and under this protocol the identity plan IS the answer key.
#
#   C1 (job 645606) -- NO HEADROOM. A monotone elementwise map phi fitted on the sources:
#     +0.00pp Top-1, +0.0019 correlation (0.593 -> 0.594). Provably bounded -- a monotone map
#     cannot reorder anything, so the most it can buy is corr - Spearman, and the measured
#     post-calibration correlation sits exactly on that ceiling. A synthetic bound under the
#     most favourable possible conditions (a shared monotone deformation) is +0.0088. The
#     0.593 -> 0.782 gap is therefore ORDER, not SHAPE.
#
# WHAT SURVIVES. Something that is NEITHER an index correspondence (leaks) NOR a monotone
# rescaling (no headroom). The EIGENVALUE SPECTRUM is exactly that:
#
#   * relabelling concepts conjugates the metric by a permutation, which leaves the spectrum
#     unchanged -- so any scalar read off the spectrum carries NO correspondence and cannot leak;
#   * it is not monotone -- so the C1 ceiling does not apply.
#
# The prior is the intrinsic dimension: how many directions of `de` are signal. `de` is replaced
# by its top-k eigen-approximation, with k taken from the SOURCES.
#
# WHY THIS IS NOT THE STAGE-2 SHRINKAGE THAT ALREADY FAILED. Shrinkage, `(1-l)S + lI`, INFLATES
# the noise subspace toward `l`; it reduced the structural gain monotonically at every R
# (3.08 -> 1.13pp at R=80, job 645436). Truncation DELETES the noise subspace instead. With the
# target metric's spectrum measuring ~23-30 effective dimensions out of 200 (measured on the
# banked source templates: src eff-rank 28.6-30.1, top-16 = 64.8% of variance), that is the
# difference between inflating and removing ~170 noise directions.
#
# PRE-REGISTERED CRITERION, and the two failure modes mean different things.
#
#   * Top-1(rank=src) <= Top-1(rank=full), paired per fold -> the rank prior carries nothing.
#     Direction dies. Note the tail already predicts a DECLINE: rank 200 -> 30 discards 65% of
#     the variance, so if the structural term has been using the tail this will be negative.
#   * monotone DECLINE in k -> the term wants MORE rank, i.e. the "noise" the manifold argument
#     calls noise is carrying signal the term uses. That is a strong and reportable negative.
#   * INTERIOR PEAK at/near the source estimate -> the prior is calibrated. This is the pass.
#
# `spec=0` is emitted as the paired twin of every truncated cell and is ASSERTED bit-identical to
# the shipped operator (smoke §14), so the comparison is against a reproduction.
#
# NO-LEAKAGE IS ASSERTED, NOT ARGUED: smoke §14 shuffles the source concept axis and asserts the
# P1 output is BIT-IDENTICAL. That is the exact control that killed M1.
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-2025 2026 2027}"
# 0 = full rank = the shipped operator. The explicit values bracket the measured effective rank
# (28.6-30.1) so the peak's LOCATION is measured, not assumed.
SPEC="${SPEC:-0,8,16,32,48}"

cd "${ROOT}"
mkdir -p outputs/slurm outputs/eval/v10_p1 outputs/scores/v10_p1

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"      # node-local, NOT NFS (AGENTS.md 2.4)
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export PYTHONPATH="${ROOT}/src"

echo "[slurm] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
echo "[p1] subjects='${SUBJECTS}' seeds='${SEEDS}' spec='${SPEC}'"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

for s in ${SUBJECTS}; do
  tag=$(printf 'sub%02d' "${s}")
  if [ "${s}" -eq 8 ]; then REF="--ref-top1 22.00 --ref-top5 50.00"; else REF="--ref-top1 unknown"; fi
  for seed in ${SEEDS}; do
    ck="${ROOT}/outputs/stage1/v8/${tag}_k20_seed${seed}/last.pt"
    sm="${ROOT}/outputs/src_metric/v8/${tag}_seed${seed}.npz"
    ev="${ROOT}/outputs/eval/v10_p1/${tag}_seed${seed}.json"
    [ -s "${ck}" ] || { echo "  [skip] no ckpt ${tag}/seed${seed}"; continue; }
    [ -s "${sm}" ] || { echo "  [skip] no source template ${tag}/seed${seed}"; continue; }
    [ -s "${ev}" ] && { echo "  [skip] done ${tag}/seed${seed}"; continue; }
    echo "[p1] === ${tag}/seed${seed} ==="
    # `--fgw-src-mix 0` and no `--fgw-src-calib`: P1 is its own operator on `de`, and leaving the
    # M1/C1 paths off guarantees no other source channel can contaminate this readout.
    "${PY}" scripts/run_eval.py --ckpts "${ck}" \
      --target-subject "${s}" --out "${ev}" \
      --recovery --reps --fgw-sweep --csls-k 10 --rho 0.1 \
      --recovery-operator fgw --recovery-tau 0.01 --recovery-alpha 0.75 \
      --fgw-src-metric "${sm}" --fgw-src-mix 0 \
      --fgw-spec-rank "${SPEC}" --fgw-spec-from-src \
      --rep-subsample 80 20 \
      --save-scores "${ROOT}/outputs/scores/v10_p1/${tag}_seed${seed}.npz" ${REF} \
      && echo "  [ok] ${tag}/seed${seed}" || echo "  [FAIL] ${tag}/seed${seed}"
  done
done

echo "[p1] === pre-registered readout ==="
"${PY}" - <<'PY'
import json, glob, numpy as np
rows = {}
for f in sorted(glob.glob("outputs/eval/v10_p1/*.json")):
    rows[f.split('/')[-1][:-5]] = list(json.load(open(f))["checkpoints"].values())[0]["rows"]

def col(tag):
    return np.array([r[tag]["top1"] for r in rows.values() if tag in r])

def tstat(d):
    return d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 and d.std(ddof=1) else float("nan")

for R in (80, 20):
    base = col(f"+ T2 R={R},a=0.75,t=0.03,spec=0")
    if base.size == 0:
        continue
    print(f"\n[R={R}] rank curve, alpha=0.75 tau=0.03  (baseline spec=0 = {base.mean():.2f})")
    print(f"{'rank':>6} {'Top-1':>7} {'sd':>6} {'d vs full':>10} {'t':>7} {'pos':>7}")
    specs = sorted({k.split("spec=")[1] for r in rows.values() for k in r
                    if f"R={R}," in k and ",spec=" in k and "structural-off" not in k})
    for sp in specs:
        c = col(f"+ T2 R={R},a=0.75,t=0.03,spec={sp}")
        if c.size != base.size:
            print(f"{sp:>6}  (n={c.size} != baseline n={base.size}; pairing refused)")
            continue
        d = c - base
        print(f"{sp:>6} {c.mean():>7.2f} {c.std(ddof=1):>6.2f} {d.mean():>+10.2f} "
              f"{tstat(d):>7.2f} {int((d>0).sum()):>3}/{len(d)}")
    src_rank = [r[f"+ T2 R={R},a=0.75,t=0.03,spec=src"]["diag"].get("spec_rank_from_src")
                for r in rows.values() if f"+ T2 R={R},a=0.75,t=0.03,spec=src" in r]
    src_rank = [x for x in src_rank if x is not None]
    if src_rank:
        print(f"  source-estimated rank: mean={np.mean(src_rank):.1f}  "
              f"range={min(src_rank)}..{max(src_rank)}")
    # structural term's own worth at each rank (on - off), which isolates the term from the ladder
    print(f"  {'rank':>6} {'on':>7} {'off':>7} {'term':>7}")
    for sp in specs:
        on = col(f"+ T2 R={R},a=0.75,t=0.03,spec={sp}")
        off = col(f"+ T2 R={R},a=0.75,t=0.03,spec={sp} (structural-off)")
        if on.size and on.size == off.size:
            print(f"  {sp:>6} {on.mean():>7.2f} {off.mean():>7.2f} {(on-off).mean():>+7.2f}")

print("\n[P1 verdict]")
b80 = col("+ T2 R=80,a=0.75,t=0.03,spec=0")
if b80.size:
    curve = {}
    for sp in sorted({k.split("spec=")[1] for r in rows.values() for k in r
                      if "R=80," in k and ",spec=" in k and "structural-off" not in k}):
        c = col(f"+ T2 R=80,a=0.75,t=0.03,spec={sp}")
        if c.size == b80.size:
            curve[sp] = float((c - b80).mean())
    if curve:
        best = max(curve, key=lambda k: curve[k])
        edge = best in ("0",)
        print(f"  best rank={best} ({curve[best]:+.2f}pp vs full rank)")
        print(f"  -> {'FALSIFIED: peak at full rank, the prior carries nothing' if edge else 'INTERIOR PEAK at rank ' + best + ': the source rank prior is calibrated'}")
        print(f"  full curve: " + "  ".join(f"{k}:{v:+.2f}" for k, v in curve.items()))
PY

echo "[p1] DONE $(date -Is)"
