#!/bin/bash
#SBATCH --job-name=v11-fuse
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v11-fuse-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v11-fuse-%j.err
#
# v11 L2/L3/L4 -- STRUCTURE ENSEMBLE + RELIABILITY-WEIGHTED FUSION. ONE SEED, TEN SUBJECTS,
# PAIRED. This is the first ARCHITECTURE arm built on a probe that PASSED, and the whole reason
# it is being submitted is `scripts/probe_fusion.py` (job 645697):
#
#     pooling independent structure estimates raises their agreement with a held-out view
#     monotonically in the number of views (cross-modal 0.209 -> 0.247 over K=1..8, 10/10 folds,
#     t=+42) WHILE the correspondence-destroying control stays pinned at ~0.000 for every K.
#
# The control is the load-bearing line: a plain average of unordered structure gains nothing, so
# the measured gain is the SHARED structure and not an artefact of pooling. That is the discipline
# that killed M1 (+15.4pp real / -14.5pp shuffled) and every arm here is subject to it.
#
# WHY FUSION IS NOT JUST ANOTHER REDUCTION. P1, A1/A2 and A4 all REDUCED the structural object --
# dropping eigenvalues, or binarising to a graph -- and all three lost, because job 645630 measured
# the structural term to be worth MORE with a richer object (+3.85pp whitened vs +2.50pp
# unwhitened). Fusion is the opposite move: it deletes no direction and averages the ESTIMATION
# noise that the recovery rung was measured to be limited by (its gain is flat in input quality,
# corr ~ -0.19, the signature of an estimator-noise ceiling).
#
# WHAT IS POOLED AND WHY IT CARRIES NO INDEX LEAKAGE. The target's own R repetitions are split into
# B contiguous blocks; each block is an independent estimate of the SAME 200 concepts, so their
# metrics share one concept order and averaging is index-consistent BY CONSTRUCTION. This is not
# the M1 template: M1's index came from a DIFFERENT subject aligned to the gallery and leaked; here
# the index is the same subject's same trials, which T2 already relies on. `smoke_test.py` §16 pins
# the deployed operator: fuse=0/1 are a strict no-op, the weights DOWN-WEIGHT a noisy block
# (0.175 vs 0.075 in the synthetic two-noise-level test), relabelling concepts relabels the scores
# identically and leaves the weights untouched, and a pure-noise cloud self-reports `is_flat` and
# disables the fusion.
#
# PRE-REGISTERED PASS/FAIL (read off at the bottom, not told after seeing the number):
#   * fuse=B passes iff it beats its OWN fuse=0 twin, paired per subject, on a majority of the ten
#     subjects, at the deployed repetition level (R=80). The `fuse=0` cell is the shipped
#     single-mean metric, so "the fusion is the only change" is a two-row comparison.
#   * the STRUCTURAL-OFF family is reported separately: `fuse=B (structural-off)` must stay ~flat.
#     A gain that appears with the structural term OFF means the fusion is leaking through the
#     cross-modal cost and not through the structure, and that falsifies the mechanism.
#   * R=20 is the gate-limited level: the probe's K-curve says more views buy more, so the fusion
#     should help MORE (relatively) where the single estimate is noisiest.
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-2025}"
FUSE="${FUSE:-2,4,8,16}"

cd "${ROOT}"
mkdir -p outputs/slurm outputs/eval/v11_fuse

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
echo "[v11] subjects='${SUBJECTS}' seeds='${SEEDS}' fuse='${FUSE}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

for s in ${SUBJECTS}; do
  tag=$(printf 'sub%02d' "${s}")
  if [ "${s}" -eq 8 ]; then REF="--ref-top1 22.00 --ref-top5 50.00"; else REF="--ref-top1 unknown"; fi
  for seed in ${SEEDS}; do
    ck="${ROOT}/outputs/stage1/v8/${tag}_k20_seed${seed}/last.pt"
    ev="${ROOT}/outputs/eval/v11_fuse/${tag}_seed${seed}.json"
    [ -s "${ck}" ] || { echo "  [skip] no ckpt ${tag}/seed${seed}"; continue; }
    [ -s "${ev}" ] && { echo "  [skip] done ${tag}/seed${seed}"; continue; }
    echo "[v11] === ${tag}/seed${seed} ==="
    # The M1/C1/P1 source channels stay OFF (`no --fgw-src-metric`): the readout must be able to
    # attribute any movement to the FUSION alone. `--recovery-operator fgw --recovery-alpha 0.75`
    # is the deployed structural operator; the `fuse=B` rows build their own structural-off twin
    # so the term's contribution is isolated inside each fused cell.
    "${PY}" scripts/run_eval.py --ckpts "${ck}" \
      --target-subject "${s}" --out "${ev}" \
      --recovery --reps --csls-k 10 --rho 0.1 \
      --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
      --rep-shrink 0.1 --fgw-struct-fuse "${FUSE}" ${REF} \
      && echo "  [ok] ${tag}/seed${seed}" || echo "  [FAIL] ${tag}/seed${seed}"
  done
done

echo "[v11] === pre-registered readout (one seed, PAIRED per subject) ==="
"${PY}" - <<'PY'
import json, glob, numpy as np

rows = {}
for f in sorted(glob.glob("outputs/eval/v11_fuse/*.json")):
    rows[f.split('/')[-1][:-5]] = list(json.load(open(f))["checkpoints"].values())[0]["rows"]
subs = sorted(rows)
print(f"[v11] {len(subs)} subjects: {subs}")
if not subs:
    raise SystemExit("[v11] no results")

def col(tag):
    v, used = [], []
    for s in subs:
        if tag in rows[s]:
            v.append(rows[s][tag]["top1"]); used.append(s)
    return np.array(v, float), used

def paired(name, base_tag):
    a, sa = col(name); b, sb = col(base_tag)
    if a.size == 0 or b.size == 0 or sa != sb:
        return None
    d = a - b
    t = (d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std(ddof=1) > 0 else float("nan")
    return dict(mean=a.mean(), sd=a.std(ddof=1) if a.size > 1 else 0.0,
                base=b.mean(), delta=d.mean(), t=t, pos=int((d > 0).sum()), n=len(d))

def show(label, r):
    if r is None:
        print(f"  {label:<46s} (missing)"); return
    print(f"  {label:<46s} {r['mean']:6.2f} +- {r['sd']:<5.2f}  vs {r['base']:6.2f}  "
          f"delta {r['delta']:+6.2f}pp  t={r['t']:5.2f}  {r['pos']:>2}/{r['n']} pos")

verdicts = {}
for R in (80, 20):
    BASE = f"+ T2 R={R},a=0.75,t=0.03,fuse=0"
    b, _ = col(BASE)
    print(f"\n[R={R}]  shipped single-mean metric (fuse=0) = {b.mean():.2f}" if b.size
          else f"\n[R={R}]  (no fuse=0 twin)")
    for B in (2, 4, 8, 16):
        t_on = f"+ T2 R={R},a=0.75,t=0.03,fuse={B}"
        if col(t_on)[0].size == 0:
            continue
        show(f"fuse={B} vs fuse=0", paired(t_on, BASE))
        show(f"fuse={B} (structural-off) vs fuse=0 (must stay flat)",
             paired(t_on + " (structural-off)", BASE + " (structural-off)"))
    # the structural term's own worth INSIDE the fused family
    print(f"  {'fused family: on - structural-off':<46s} {'on':>7} {'off':>7} {'term':>7}")
    for B in (2, 4, 8, 16):
        t_on = f"+ T2 R={R},a=0.75,t=0.03,fuse={B}"
        on, _ = col(t_on); off, _ = col(t_on + " (structural-off)")
        if on.size and on.size == off.size:
            print(f"  fuse={str(B):<40s} {on.mean():>7.2f} {off.mean():>7.2f} {(on-off).mean():>+7.2f}")
    verdicts[R] = {}

print("\n[v11] PRE-REGISTERED VERDICT")
R0 = 80
BASE = f"+ T2 R={R0},a=0.75,t=0.03,fuse=0"
curve = {}
for B in (2, 4, 8, 16):
    r = paired(f"+ T2 R={R0},a=0.75,t=0.03,fuse={B}", BASE)
    if r is not None:
        curve[B] = r
if not curve:
    print("  (no fused cells emitted -- the flag never reached the operator)")
else:
    best = max(curve, key=lambda k: curve[k]["delta"])
    r = curve[best]
    pass_shape = r["delta"] > 0 and r["pos"] >= (r["n"] + 1) // 2 and r["t"] > 2.0
    print("  fuse curve vs fuse=0 (delta pp): "
          + "  ".join(f"{B}:{curve[B]['delta']:+.2f}" for B in sorted(curve)))
    if pass_shape:
        print(f"  -> PASS: fuse={best} beats the shipped single-mean metric by {r['delta']:+.2f}pp "
              f"({r['pos']}/{r['n']} subjects, t={r['t']:.2f}). Structure pooling survives "
              f"deployment.")
    else:
        print(f"  -> NOT PASSED: best fuse={best} is {r['delta']:+.2f}pp "
              f"({r['pos']}/{r['n']} subjects, t={r['t']:.2f}). The probe's within-view gain did "
              f"NOT convert to retrieval; the bottleneck is downstream of the structure.")
    print("\n  reference points: SCORE published 53.23%; internal paired on the banked v8 encoder "
          "at 80 reps: FGW 53.80%, S3R 50.22%.")
PY

echo "[v11] DONE $(date -Is)"
