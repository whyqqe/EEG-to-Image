#!/bin/bash
#SBATCH --job-name=v10-baseline
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v10-baseline-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v10-baseline-%j.err
#
# v10 CLAIM-A BASELINE -- the INTERNAL operator comparison, run contemporaneously.
#
# THE CLAIM. On the identical v8 encoder, identical R=80 repetition cloud, identical folds, and
# identical seeds, the ONLY thing that changes between these two rows is the transport operator:
#
#     + T2 reps  with the DEFAULT operator       = SCORE's `coordinate_recovery`
#                                                  (mutual-NN landmarks + orthogonal Procrustes)
#     + T2 reps  with `--recovery-operator fgw`   = our second-order structural transport
#
# Banked numbers say 46.97 vs 53.80, i.e. **+6.83pp (t=9.42, 10/10 subjects)**, which is an
# order of magnitude more defensible than "53.80 vs SCORE's published 53.23" (+0.57pp against a
# quoted sd of 1.62, across two different pipelines). But those two rows live in directories
# written at different times by different jobs, so a reviewer is right to ask whether the code
# changed between them. This job removes that question: both rows are produced HERE from the
# SAME checkpoint, in ONE invocation each, with the current code.
#
# Why the default is SCORE's operator: `_recovery_fn_for` returns `None` for `--recovery-operator
# hard`, and `rep_cloud_scores(recovery_fn=None)` falls through to `calibration.coordinate_recovery`
# -- which IS SCORE's deployed operator. So `--recovery --reps` with no operator flag is exactly
# SCORE's operator applied to our repetition cloud. Nothing is reimplemented for the baseline.
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
cd "${ROOT}"
mkdir -p outputs/slurm outputs/eval/v10_baseline/score outputs/eval/v10_baseline/fgw \
         outputs/scores/v10_baseline/score outputs/scores/v10_baseline/fgw

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
nvidia-smi --query-gpu=name --format=csv,noheader || true
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

for s in 1 2 3 4 5 6 7 8 9 10; do
  tag=$(printf 'sub%02d' "${s}")
  if [ "${s}" -eq 8 ]; then REF="--ref-top1 22.00 --ref-top5 50.00"; else REF="--ref-top1 unknown"; fi
  for seed in 2025 2026 2027; do
    ck="${ROOT}/outputs/stage1/v8/${tag}_k20_seed${seed}/last.pt"
    [ -s "${ck}" ] || { echo "  [skip] no ckpt ${tag}/seed${seed}"; continue; }

    # --- arm 1: SCORE's operator (default 'hard' -> coordinate_recovery) ---
    ev="${ROOT}/outputs/eval/v10_baseline/score/${tag}_seed${seed}.json"
    sc="${ROOT}/outputs/scores/v10_baseline/score/${tag}_seed${seed}.npz"
    if [ -s "${ev}" ] && [ -s "${sc}" ]; then echo "  [skip] score ${tag}/seed${seed}"; else
      "${PY}" scripts/run_eval.py --ckpts "${ck}" --target-subject "${s}" --out "${ev}" \
        --recovery --reps --csls-k 10 --rho 0.1 --save-scores "${sc}" ${REF} >/dev/null 2>&1 \
        && echo "  [ok] score ${tag}/seed${seed}" || echo "  [FAIL] score ${tag}/seed${seed}"
    fi

    # --- arm 2: our structural operator, SAME checkpoint, SAME cloud ---
    ev="${ROOT}/outputs/eval/v10_baseline/fgw/${tag}_seed${seed}.json"
    sc="${ROOT}/outputs/scores/v10_baseline/fgw/${tag}_seed${seed}.npz"
    if [ -s "${ev}" ] && [ -s "${sc}" ]; then echo "  [skip] fgw ${tag}/seed${seed}"; else
      "${PY}" scripts/run_eval.py --ckpts "${ck}" --target-subject "${s}" --out "${ev}" \
        --recovery --reps --csls-k 10 --rho 0.1 \
        --recovery-operator fgw --recovery-alpha 0.75 --recovery-tau 0.03 \
        --save-scores "${sc}" ${REF} >/dev/null 2>&1 \
        && echo "  [ok] fgw ${tag}/seed${seed}" || echo "  [FAIL] fgw ${tag}/seed${seed}"
    fi
  done
done

# --- the paired comparison, per-subject, so the claim carries its own statistic ---
echo "[baseline] === internal operator comparison (paired over 30 runs) ==="
"${PY}" - <<'PY'
import glob, json
from pathlib import Path
import numpy as np

def t2(d):
    out = {}
    for f in glob.glob(f"{d}/sub*_seed*.json"):
        j = json.load(open(f))
        for c in j["checkpoints"].values():
            r = c["rows"].get("+ T2 reps")
            if r:
                out[(j["target_subject"], Path(f).name)] = float(r["top1"])
    return out

sc, fg = t2("outputs/eval/v10_baseline/score"), t2("outputs/eval/v10_baseline/fgw")
keys = sorted(set(sc) & set(fg))
if not keys:
    raise SystemExit("[baseline] no paired cells -- nothing to report")
a = np.array([sc[k] for k in keys]); b = np.array([fg[k] for k in keys]); d = b - a
subj = [k[0] for k in keys]
print(f"runs paired: {len(keys)}")
print(f"  SCORE's operator (coordinate_recovery), T2 : {a.mean():.2f} +- {a.std(ddof=1):.2f}")
print(f"  ours (FGW structural), T2                  : {b.mean():.2f} +- {b.std(ddof=1):.2f}")
print(f"  paired gain                                : {d.mean():+.2f}pp  sd={d.std(ddof=1):.2f}  "
      f"t={d.mean()/(d.std(ddof=1)/len(d)**0.5):.2f}  pos={int((d>0).sum())}/{len(d)}")
for s in sorted(set(subj)):
    m = [d[i] for i, x in enumerate(subj) if x == s]
    print(f"    sub{s:02d}: {np.mean(m):+6.2f}pp")
PY

echo "[baseline] DONE $(date -Is)"
