#!/bin/bash
#SBATCH --job-name=v10-c1
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v10-c1-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v10-c1-%j.err
#
# v10 STAGE 3 -- C1, SOURCE-CALIBRATED METRIC TRANSFER. docs/eeg2image_v10_m1_theory.md §9.
#
# WHAT M1 GOT WRONG, IN ONE LINE. M1's +15.4pp was an INDEX correspondence, not geometry. The
# permutation control (job 645593) is unambiguous: shuffling the source template's concept axis
# left the geometry intact and turned +15.4pp into -14.5pp, with corr(de_src, di) collapsing
# 0.782 -> 0.001. `de_src[i,j]` carries the identity of concept i in its own subscript and `di`
# carries the gallery slot in its own, so the FGW optimum became the IDENTITY plan -- and under
# this protocol the identity plan IS the answer key. It was reading the labels off the index.
#
# WHAT SURVIVES, AND IS THE WHOLE BASIS OF C1. One measurement is real and is not an artifact:
#
#       corr(de_src,    di) = 0.782        corr(de_target, di) = 0.593
#
# The SOURCE subjects' metric is a materially better proxy for the gallery metric than the
# target's own -- unsurprising, since the encoder was trained on source subjects to align
# exactly this EEG metric to the image metric. C1 spends that information the only way it can
# be spent without leakage: learn the map ON THE SOURCES (where the source<->gallery
# correspondence is known and using it is ordinary supervised transfer), then apply the learned
# map to the TARGET'S OWN metric.
#
#       phi       = argmin_phi  sum_{s in sources} sum_{i,j} ( phi(de_s[i,j]) - di[i,j] )^2
#       de_used   = phi( de_target )
#
# `phi` is applied ELEMENTWISE. It never sees an index, its input is the target's metric in the
# target's own frame, and its output stays in that frame. There is no cross-frame correspondence
# anywhere in the APPLICATION, so M1's failure mode is absent BY CONSTRUCTION and not by an
# added penalty. The correspondence appears only inside the FITTING, on source data.
#
# PRE-REGISTERED CRITERION -- it can fail on the first fold, and the two failure modes mean
# different things:
#
#   * corr(phi(de_target), di) <= corr(de_target, di)   -> phi does NOT transfer across subjects.
#     The direction dies here, for the cost of one GPU job. This is the honest kill switch.
#   * corr rises but Top-1 does not                      -> the source/target metric gap is
#     SCALING (a monotone map captures it), not ORDERING. `phi` is monotone, so it cannot repair
#     a mis-ordered `de_target`; this outcome says the residual gap lives in the order, and the
#     fix has to be a re-estimation (C2/C3), not a recalibration.
#
# Every fold also emits a `,calib` row next to its own `m=0` twin from the SAME embedding, so the
# comparison is paired on the identical repetition subset and the identical checkpoint.
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-2025 2026 2027}"

cd "${ROOT}"
mkdir -p outputs/slurm outputs/eval/v10_c1 outputs/scores/v10_c1

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
echo "[c1] subjects='${SUBJECTS}' seeds='${SEEDS}'"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

for s in ${SUBJECTS}; do
  tag=$(printf 'sub%02d' "${s}")
  if [ "${s}" -eq 8 ]; then REF="--ref-top1 22.00 --ref-top5 50.00"; else REF="--ref-top1 unknown"; fi
  for seed in ${SEEDS}; do
    ck="${ROOT}/outputs/stage1/v8/${tag}_k20_seed${seed}/last.pt"
    sm="${ROOT}/outputs/src_metric/v8/${tag}_seed${seed}.npz"
    ev="${ROOT}/outputs/eval/v10_c1/${tag}_seed${seed}.json"
    [ -s "${ck}" ] || { echo "  [skip] no ckpt ${tag}/seed${seed}"; continue; }
    [ -s "${sm}" ] || { echo "  [skip] no source template ${tag}/seed${seed}"; continue; }
    [ -s "${ev}" ] && { echo "  [skip] done ${tag}/seed${seed}"; continue; }
    echo "[c1] === ${tag}/seed${seed} ==="
    # `--fgw-src-mix 0` on purpose: C1 is a different operator on `de`, not a setting of the M1
    # blend. Keeping the M1 path off means no `m>0` cell can contaminate this readout.
    "${PY}" scripts/run_eval.py --ckpts "${ck}" \
      --target-subject "${s}" --out "${ev}" \
      --recovery --reps --fgw-sweep --csls-k 10 --rho 0.1 \
      --recovery-operator fgw --recovery-tau 0.01 --recovery-alpha 0.75 \
      --fgw-src-metric "${sm}" --fgw-src-mix 0 --fgw-src-calib \
      --rep-subsample 80 20 \
      --save-scores "${ROOT}/outputs/scores/v10_c1/${tag}_seed${seed}.npz" ${REF} \
      && echo "  [ok] ${tag}/seed${seed}" || echo "  [FAIL] ${tag}/seed${seed}"
  done
done

echo "[c1] === pre-registered readout ==="
"${PY}" - <<'PY'
import json, glob, numpy as np
rows = {}
for f in sorted(glob.glob("outputs/eval/v10_c1/*.json")):
    rows[f.split('/')[-1][:-5]] = list(json.load(open(f))["checkpoints"].values())[0]["rows"]

def col(tag, key="top1"):
    return np.array([r[tag][key] for r in rows.values() if tag in r])

print(f"\n{'cell':46s} {'Top-1':>7} {'sd':>6}  n")
for R in (80, 20):
    for tag, lab in ((f"+ T2 R={R},a=0.75,t=0.03,m=0",            f"m=0 baseline R={R}"),
                     (f"+ T2 R={R},a=0.75,t=0.03,calib",           f"C1 calibrated  R={R}"),
                     (f"+ T2 R={R},a=0.75,t=0.03,calib (structural-off)",
                      f"C1 structural-off  R={R}")):
        c = col(tag)
        if c.size:
            print(f"{lab:46s} {c.mean():>7.2f} {c.std(ddof=1):>6.2f}  {c.size}")

print("\n[transfer criterion] corr(phi(de_target), di) vs corr(de_target, di)")
for R in (80, 20):
    cl = [r[f"+ T2 R={R},a=0.75,t=0.03,calib"]["diag"]
           for r in rows.values() if f"+ T2 R={R},a=0.75,t=0.03,calib" in r]
    if cl:
        a = np.mean([d.get("corr_de_target_di_precalib", np.nan) for d in cl])
        b = np.mean([d.get("corr_de_target_di_calib", np.nan) for d in cl])
        g = np.array([d.get("c1_transfer_gain_corr", np.nan) for d in cl])
        g = g[np.isfinite(g)]
        v = "PASS (phi transfers)" if np.isfinite(a) and b > a else "FAIL (phi does not transfer)"
        print(f"  R={R:>3}: pre={a:+.3f} -> post={b:+.3f}  gain={g.mean():+.4f} "
              f"(t={g.mean()/(g.std(ddof=1)/np.sqrt(len(g))):.2f}, n={len(g)})  -> {v}")

print("\n[paired Top-1 effect of C1]")
for R in (80, 20):
    b = col(f"+ T2 R={R},a=0.75,t=0.03,m=0")
    c = col(f"+ T2 R={R},a=0.75,t=0.03,calib")
    if b.size and c.size == b.size:
        d = c - b
        print(f"  R={R:>3}: {b.mean():.2f} -> {c.mean():.2f}  d={d.mean():+.2f}pp  "
              f"t={d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):.2f}  pos={int((d>0).sum())}/{len(d)}")
PY

echo "[c1] DONE $(date -Is)"
