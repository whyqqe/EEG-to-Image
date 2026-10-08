#!/bin/bash
#SBATCH --job-name=v12-e2
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v12-e2-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v12-e2-%j.err
#
# E2 -- THE DIFFERENTIATOR. 10 folds x 1 seed, NO TRAINING. Every cell is produced from the
# ALREADY-BANKED v8 checkpoint of that fold, so the seeded baseline is not re-measured (the
# encoders are identical, which makes the comparison paired BY CONSTRUCTION rather than by a
# join that could silently drop runs).
#
# WHAT IS BEING TESTED (docs/eeg2image_commet_plan.md, verdict V2). The field's strongest
# label-free test-time method is SCORE (coordinate recovery) / SATTC (similarity-matrix
# calibration), and BOTH operate on a SINGLE MEAN embedding per concept -- an order-1 object.
# COMMET's claim is that the right object is the order-2 metric estimated from the repetition
# cloud. This job emits, from ONE checkpoint:
#
#   point-based baseline : `+ SATTC(whiten + CSLS)`  -- SATTC's structural expert (mutual-NN
#                          agreement + bidirectional top-k + hubness, weighted PoE) on the mean
#                          embedding's similarity matrix, the strongest calibration rung.
#   COMMET (ours)        : `+ T2 R=80,...,fuse=16`  -- the FGW structural fusion over the R=80
#                          per-trial cloud, the E1 headline cell.
#
# V2 PASSES iff COMMET beats the point-based baseline, paired across the ten folds. Top-5 is
# reported beside it, because E1's fusion improved Top-5 as well and a "cloud > point" claim that
# degrades Top-5 would not be deployable.
#
# A NOTE ON WHY THE BASELINE IS EMITTED EVEN THOUGH IT IS "EXISTING WORK". `structural_scores`
# was implemented but never wired into a report, and its `_ranks` helper was BROKEN on any
# real-sized matrix (the backward rank came out non-integer and out-of-range, poisoning `log`
# with NaN). That is fixed and pinned in smoke_test §18. So the baseline number on our encoder did
# not previously exist and must be produced here -- run on the SAME folds as COMMET, not quoted
# from the paper.
#
# Submit: sbatch slurm/v12_e2.sbatch
# Smoke:  SUBJECTS=8 sbatch slurm/v12_e2.sbatch
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEED="${SEED:-2025}"
TAGV="${TAGV:-v8}"
OUTDIR="${OUTDIR:-${ROOT}/outputs/eval/v12_e2}"
mkdir -p "${ROOT}/outputs/slurm" "${OUTDIR}"

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
cd "${ROOT}"

echo "[e2] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
echo "[e2] subjects='${SUBJECTS}' seed=${SEED} tag=${TAGV}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

for s in ${SUBJECTS}; do
  tag=$(printf 'sub%02d' "${s}")
  ck="${ROOT}/outputs/stage1/${TAGV}/${tag}_k20_seed${SEED}/last.pt"
  ev="${OUTDIR}/${tag}_seed${SEED}.json"
  if [ ! -s "${ck}" ]; then echo "  [skip] no ckpt ${tag}/seed${SEED}"; continue; fi
  if [ -s "${ev}" ]; then echo "  [skip] done ${tag}/seed${SEED}"; continue; fi
  if [ "${s}" -eq 8 ]; then REF="--ref-top1 22.00 --ref-top5 50.00"; else REF="--ref-top1 unknown"; fi
  echo "[e2] === ${tag}/seed${SEED} ==="
  # `--fgw-struct-fuse 0,16` emits the paired fuse=0 twin and the fuse=16 COMMET cell; the
  # structural-off twin is emitted automatically for each. `--sim-calib satc` adds the point-based
  # baseline rows on top of the whiten+CSLS rung.
  "${PY}" scripts/run_eval.py --ckpts "${ck}" \
    --target-subject "${s}" --out "${ev}" \
    --recovery --reps --csls-k 10 --rho 0.1 \
    --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
    --rep-shrink 0.1 --fgw-struct-fuse 0,16 \
    --sim-calib satc --sim-calib-lam 0.2 \
    --sim-calib-base "+ whiten + CSLS + recovery,+ CSLS + recovery" ${REF} \
    && echo "  [ok] ${tag}/seed${SEED}" || echo "  [FAIL] ${tag}/seed${SEED}"
done

echo "[e2] === V2 readout: cloud (COMMET) vs point (SATTC), PAIRED on the same checkpoints ==="
GLOB="${OUTDIR}/sub*_seed*.json" "${PY}" - <<'PY'
import json, glob, os, numpy as np
cells = {}
for f in sorted(glob.glob(os.environ["GLOB"])):
    cells[os.path.basename(f)[:-5]] = list(json.load(open(f))["checkpoints"].values())[0]["rows"]
keys = sorted(cells)
print(f"[e2] {len(keys)} folds: {keys}")
if not keys:
    raise SystemExit("[e2] no cells")

def col(tag, metric):
    ks = [k for k in keys if tag in cells[k]]
    return np.array([cells[k][tag][metric] for k in ks], float), ks

def paired(a, b, metric):
    va, ka = col(a, metric); vb, kb = col(b, metric)
    if va.size == 0 or vb.size == 0 or ka != kb:
        return None
    d = va - vb
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    t = d.mean() / (sd / np.sqrt(d.size)) if sd > 0 else float("nan")
    return dict(a=va.mean(), b=vb.mean(), d=d.mean(), t=t, pos=int((d > 0).sum()), n=d.size)

# The point-based baseline is the STRONGEST SATTC row (same-recovery rung), so the cloud-vs-point
# comparison holds the recovery operator fixed -- the only difference is mean vs cloud metric.
SATTC_CANDIDATES = ["+ SATTC(whiten + CSLS + recovery)", "+ SATTC(CSLS + recovery)",
                    "+ SATTC(whiten + CSLS)"]
avail = [t for t in SATTC_CANDIDATES if col(t, "top1")[0].size]
best_satc = max(avail, key=lambda t: col(t, "top1")[0].mean()) if avail else None
COMMET = "+ T2 R=80,a=0.75,t=0.03,fuse=16"
POINTF = "+ T2 R=80,a=0.75,t=0.03,fuse=0"
print(f"\n[e2] point-based baseline (best SATTC row) = {best_satc}  "
      f"Top-1 {col(best_satc, 'top1')[0].mean():.2f}" if best_satc else "[e2] no SATTC rows")
for metric, nm in (("top1", "Top-1"), ("top5", "Top-5")):
    print(f"\n=== {nm} ===")
    if best_satc:
        for label, tgt in (("COMMET (cloud)", COMMET), ("shipped fgw point (fuse=0)", POINTF)):
            r = paired(tgt, best_satc, metric)
            if r:
                print(f"  {label:<26s} {r['a']:6.2f} vs SATTC {r['b']:6.2f}  delta {r['d']:+6.2f}pp"
                      f"  t={r['t']:5.2f}  {r['pos']}/{r['n']} pos")
    # SATTC on the pre-recovery rung, for reference
    r0 = paired("+ SATTC(whiten + CSLS)", "+ whiten + CSLS", metric)
    if r0:
        print(f"  [ref] SATTC on whiten+CSLS: {r0['d']:+.2f}pp (the paper's rung)")

enr = [cells[k].get(best_satc, {}).get("diag", {}).get("mutual_topk_enrichment")
       for k in keys] if best_satc else []
enr = [e for e in enr if e is not None]
if enr:
    print(f"\n[e2] mean SATTC mutual-top-k enrichment = {np.mean(enr):.2f} "
          f"(~1 means the structural signal is noise on this fold)")
PY

echo "[e2] DONE $(date -Is)"
