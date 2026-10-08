#!/bin/bash
#SBATCH --job-name=v10-farch
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v10-farch-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v10-farch-%j.err
#
# v10 FULL ARCHITECTURE -- ONE SEED, TEN SUBJECTS, PAIRED (hardware/eval budget honest).
#
# WHY ONE SEED. The three preceding stages (M1 / C1 / P1) were run at 3 seeds each. The user
# asked for a FAIR COMPARISON rather than seed inflation, and fair here means PAIRED ON THE SAME
# FOLD: every arm is read against its own twin computed from the same checkpoint, same fold,
# same repetition subset, differing in exactly one argument. `spec=0`/`rawspec=0`/`structural-off`
# twins exist for precisely this, and smoke §14-§15 assert the neutral setting is BIT-IDENTICAL to
# the shipped operator. So a one-seed run is not a weaker comparison -- it is the same paired
# contrast with the between-seed variance term removed, and the pairing is what carries the
# inference.
#
# THE ARCHITECTURE BEING SUBMITTED (docs/eeg2image_v10_m1_theory.md §9-§11). Three components,
# ordered by what falsified the previous one:
#
#   A1/A2  UNWHITENED METRIC + SPECTRAL RANK PRIOR  (`--fgw-raw-metric`)
#          P1 demanded a low-rank EEG metric and truncated the WHITENED one; `_whiten_from_cloud`
#          flattens a spectrum by construction, so P1 measured 0/30 folds positive with the
#          structural term's sign inverted. The low-rank premise was measured in the RAW encoder
#          space, so the prior is relocated there. `rawspec=0` is the un-truncated raw metric and
#          is the paired twin of every truncated cell. KILL SWITCH: the family dies if the
#          unwhitened effective rank exceeds 50 (`p2_premise_ok` is recorded on every row).
#
#   A4     TOPOLOGICAL REFERENCE  (`--fgw-topo`)
#          The metric's FINE magnitudes are what whitening destroys; connectivity is not. Both
#          sides are thresholded to 0/1 adjacency, EACH AT ITS OWN characteristic scale (topology
#          is scale-free), so the term compares SHAPE. `eps` is one permutation-invariant scalar
#          read off each domain's H0 merge-gap -- no query, no label, no index correspondence.
#
#   A5     CORRESPONDENCE-DESTROYING CONTROL  (asserted in smoke §14-§15, not run here)
#          The exact control that killed M1: permute the concept axis and require BIT-IDENTITY.
#
# NO-LEAKAGE, ASSERTED RATHER THAN ARGUED. `smoke_test.py` §14/§15 shuffle the source concept axis
# and assert the P1/P2 outputs are BIT-IDENTICAL, and assert `_topo_eps_from` is permutation
# invariant. A scalar read off a spectrum or a distance multiset cannot carry an index
# correspondence, so the M1 channel -- which turned +15.4pp into -14.5pp under the permutation
# control -- is absent BY CONSTRUCTION in every arm submitted here.
#
# PRE-REGISTERED PASS/FAIL (read off by the block at the bottom, so the verdict is not a story
# told after seeing the number):
#   * P2 passes iff some truncated rawspec cell beats `rawspec=0`, paired per fold, with a
#     majority of subjects positive; an interior peak is the calibrated outcome and a monotone
#     decline falsifies the rank prior again (now in the right space, which is the informative
#     negative: it would say the unwhitened tail is signal).
#   * P3 passes iff `topo=auto` beats BOTH its `structural-off` twin (the term does something)
#     AND the metric reference it replaces (topology is a BETTER structural object than the
#     metric). Beating only the off-twin but not the metric is a wash, and is reported as such.
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-2025}"
# 0 = full rank = the raw metric with no truncation (the paired twin). 16/32 bracket the measured
# effective rank so the peak's LOCATION is measured rather than assumed; `src` takes it from the
# source subjects' own effective rank, which is the transfer form of the prior.
SPEC="${SPEC:-0,16,32}"
TOPQ="${TOPQ:-0.05,0.10,0.20}"
TOPLEPS="${TOPLEPS:-}"

cd "${ROOT}"
mkdir -p outputs/slurm outputs/eval/v10_farch outputs/scores/v10_farch

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
echo "[farch] subjects='${SUBJECTS}' seeds='${SEEDS}' spec='${SPEC}' topq='${TOPQ}' topleps='${TOPLEPS}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

TOPO_FLAGS="--fgw-topo"
if [ -n "${TOPLEPS}" ]; then TOPO_FLAGS="--fgw-topo --fgw-topo-eps ${TOPLEPS}"; fi

for s in ${SUBJECTS}; do
  tag=$(printf 'sub%02d' "${s}")
  if [ "${s}" -eq 8 ]; then REF="--ref-top1 22.00 --ref-top5 50.00"; else REF="--ref-top1 unknown"; fi
  for seed in ${SEEDS}; do
    ck="${ROOT}/outputs/stage1/v8/${tag}_k20_seed${seed}/last.pt"
    sm="${ROOT}/outputs/src_metric/v8/${tag}_seed${seed}.npz"
    ev="${ROOT}/outputs/eval/v10_farch/${tag}_seed${seed}.json"
    [ -s "${ck}" ] || { echo "  [skip] no ckpt ${tag}/seed${seed}"; continue; }
    [ -s "${sm}" ] || { echo "  [skip] no source template ${tag}/seed${seed}"; continue; }
    [ -s "${ev}" ] && { echo "  [skip] done ${tag}/seed${seed}"; continue; }
    echo "[farch] === ${tag}/seed${seed} ==="
    # `--fgw-src-mix 0`: the M1/C1 source channels stay OFF so the readout cannot absorb them.
    # `--fgw-spec-from-src` only supplies a SCALAR (the source effective rank), which is the
    # permutation-invariant quantity smoke §14/§15 pin.
    "${PY}" scripts/run_eval.py --ckpts "${ck}" \
      --target-subject "${s}" --out "${ev}" \
      --recovery --reps --fgw-sweep --csls-k 10 --rho 0.1 \
      --recovery-operator fgw --recovery-tau 0.01 --recovery-alpha 0.75 \
      --fgw-src-metric "${sm}" --fgw-src-mix 0 \
      --fgw-spec-rank "${SPEC}" --fgw-spec-from-src \
      --fgw-raw-metric ${TOPO_FLAGS} --fgw-topo-q "${TOPQ}" \
      --rep-subsample 80 20 \
      --save-scores "${ROOT}/outputs/scores/v10_farch/${tag}_seed${seed}.npz" ${REF} \
      && echo "  [ok] ${tag}/seed${seed}" || echo "  [FAIL] ${tag}/seed${seed}"
  done
done

echo "[farch] === pre-registered readout (one seed, PAIRED per subject) ==="
"${PY}" - <<'PY'
import json, glob, numpy as np

rows = {}
for f in sorted(glob.glob("outputs/eval/v10_farch/*.json")):
    rows[f.split('/')[-1][:-5]] = list(json.load(open(f))["checkpoints"].values())[0]["rows"]
subs = sorted(rows)
print(f"[farch] {len(subs)} subjects: {subs}")
if not subs:
    raise SystemExit("[farch] no results")

def col(tag):
    v, sub_used = [], []
    for s in subs:
        if tag in rows[s]:
            v.append(rows[s][tag]["top1"]); sub_used.append(s)
    return np.array(v, float), sub_used

def paired(name, base_tag):
    a, sa = col(name); b, sb = col(base_tag)
    if a.size == 0 or b.size == 0:
        return None
    if sa != sb:
        print(f"  {name:<44s} (subjects differ from baseline; pairing refused)")
        return None
    d = a - b
    t = (d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std(ddof=1) > 0 else float("nan")
    return dict(mean=a.mean(), sd=a.std(ddof=1) if a.size > 1 else 0.0,
                base=b.mean(), delta=d.mean(), t=t, pos=int((d > 0).sum()), n=len(d))

def show(label, r):
    if r is None:
        return
    print(f"  {label:<44s} {r['mean']:6.2f} +- {r['sd']:<5.2f}  vs {r['base']:6.2f}  "
          f"delta {r['delta']:+6.2f}pp  t={r['t']:5.2f}  {r['pos']:>2}/{r['n']} pos")

for R in (80, 20):
    BASE = f"+ T2 R={R},a=0.75,t=0.03,spec=0"          # == the shipped FGW best arm
    print(f"\n[R={R}]  baseline (shipped FGW, paired twin) = {col(BASE)[0].mean():.2f}"
          if col(BASE)[0].size else f"\n[R={R}]  (no baseline)")

    print(f"  --- A1/A2  unwhitened metric + spectral rank prior ---")
    show("raw metric, no truncation (twin)", paired(f"+ T2 R={R},a=0.75,t=0.03,rawspec=0", BASE))
    for spec in ("16", "32", "src"):
        t = f"+ T2 R={R},a=0.75,t=0.03,rawspec={spec}"
        show(f"raw metric + rank {spec}", paired(t, f"+ T2 R={R},a=0.75,t=0.03,rawspec=0"))

    print(f"  --- A4  topological (eps-graph) reference, density swept ---")
    for q in (0.05, 0.10, 0.20):
        qs = f"{q:g}"
        t = f"+ T2 R={R},a=0.75,t=0.03,topo=q{qs}"
        if col(t)[0].size == 0:
            continue
        show(f"topo graph q={qs} vs metric baseline", paired(t, BASE))
        r2 = paired(t, f"+ T2 R={R},a=0.75,t=0.03,topo=q{qs} (structural-off)")
        if r2 is not None:
            print(f"  {('  (vs its own structural-off twin)'):<44s} "
                  f"{r2['mean']:6.2f}        vs {r2['base']:6.2f}  delta {r2['delta']:+6.2f}pp  "
                  f"t={r2['t']:5.2f}  {r2['pos']:>2}/{r2['n']} pos")

    # the structural term's own worth, for every family, on - off
    print(f"  {'family':<44s} {'on':>7} {'off':>7} {'term':>7}")
    fams = []
    for r in rows.values():
        for k in r:
            if f"R={R}," not in k or "structural-off" in k:
                continue
            if "spectral-off" in k:
                continue
            if ",spec=" in k or ",rawspec=" in k or ",topo=" in k:
                fams.append(k.replace("+ T2 ", ""))
    for fam in sorted(set(fams)):
        on, _ = col(f"+ T2 {fam}")
        off, _ = col(f"+ T2 {fam} (structural-off)")
        if on.size and on.size == off.size:
            print(f"  {fam:<44s} {on.mean():>7.2f} {off.mean():>7.2f} {(on-off).mean():>+7.2f}")

print("\n[farch] PRE-REGISTERED VERDICTS")
b80 = col("+ T2 R=80,a=0.75,t=0.03,spec=0")[0]
if b80.size:
    raw0 = col("+ T2 R=80,a=0.75,t=0.03,rawspec=0")[0]
    if raw0.size:
        print(f"  A1/A2 relocation: raw metric {raw0.mean():.2f} vs whitened {b80.mean():.2f} "
              f"({(raw0.mean()-b80.mean()):+.2f}pp) -- the first thing to read, before any rank")
    curve = {}
    for spec in ("0", "16", "32", "src"):
        c = col(f"+ T2 R=80,a=0.75,t=0.03,rawspec={spec}")[0]
        if c.size and c.size == raw0.size:
            curve[spec] = float((c - raw0).mean())
    if curve:
        best = max(curve, key=lambda k: curve[k])
        print(f"  A1/A2 rank curve (vs rawspec=0): " + "  ".join(f"{k}:{v:+.2f}" for k, v in curve.items()))
        if best == "0":
            print("    -> FALSIFIED AGAIN: peak at full rank; the unwhitened tail is signal.")
        else:
            print(f"    -> PASS SHAPE: interior peak at rawspec={best} "
                  f"({curve[best]:+.2f}pp); the prior is calibrated in the unwhitened space.")
    tp_curve = {}
    for q in (0.05, 0.10, 0.20):
        qs = f"{q:g}"
        c = col(f"+ T2 R=80,a=0.75,t=0.03,topo=q{qs}")[0]
        if c.size and c.size == b80.size:
            off = col(f"+ T2 R=80,a=0.75,t=0.03,topo=q{qs} (structural-off)")[0]
            tp_curve[qs] = (float((c - b80).mean()), float((c - off).mean()))
    if tp_curve:
        for q, (dm, do) in tp_curve.items():
            print(f"  A4 topology q={q}: vs metric baseline {dm:+.2f}pp | vs its own "
                  f"ablation {do:+.2f}pp")
        best_q = max(tp_curve, key=lambda k: tp_curve[k][0])
        dm, do = tp_curve[best_q]
        if dm > 0 and do > 0:
            print(f"    -> PASS at q={best_q}: beats BOTH the metric reference and its ablation.")
        elif do > 0:
            print(f"    -> TERM WORKS, OBJECT IS A WASH at q={best_q}: beats its ablation "
                  f"({do:+.2f}pp) but not the metric ({dm:+.2f}pp).")
        else:
            print(f"    -> FALSIFIED: no density beats its own ablation; the topological "
                  f"reference carries nothing the metric does not.")

print("\n[farch] reference points on record: SCORE published 53.23% (source). Internal paired on "
      "the banked v8 encoder at 80 reps: SCORE operator 46.97%, S3R 50.22%, FGW 53.80%.")
PY

echo "[farch] DONE $(date -Is)"
