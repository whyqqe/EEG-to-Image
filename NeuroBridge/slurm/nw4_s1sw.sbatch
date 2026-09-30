#!/bin/bash
#SBATCH --job-name=nw4_s1sw
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_s1sw_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_s1sw_%j.err
#
# Why re-train S1 at all, when the first run looked healthy?
#
# `nw4_diag_vstrue.py` measured the exported condition against the ONLY metric that
# tracks generation quality -- mean cos(condition_i, trial i's true clip_img) minus
# the same quantity for a constant vector, which already scores 0.6201:
#
#     nw4 head_img (best retriever, top1 0.350)   vs_cl = -0.3450
#     hybrid_s08 ip_uck (generated 0.728)         vs_cl = +0.0367
#
# Our condition was far BELOW a constant, because the checkpoint was selected by
# `img_mini_csls` and the objective was dominated by three InfoNCE terms (weight 3.0)
# against a single MSE anchor (0.5).  InfoNCE is invariant to a per-row shift along a
# direction common to all rows, so it discards the large common component CLIP
# embeddings share -- which is most of what IP-Adapter reads.
#
# So sweep the two knobs that control this, measure vs_cl on the val split, and only
# then decide whether the generation arms are worth running.
#
#   w_anchor      : 0.5 (as before) .. 20.0 (anchor dominates)
#   select_metric : img_vs_cl (new) vs img_mini_csls (old, to prove the point)
#
# Each run is ~2.5 min; S1 is small.  The gate is COND_GATE["vs_cl"] = 0.0450.

set -uo pipefail
cd /project/peilab/why/NeuroBridge

PYTHON=/project/peilab/why/eeg-brainit/.venv/bin/python
export OMP_NUM_THREADS=8
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export HOME="${XDG_CACHE_HOME}"

OUT=outputs/nw4/s1sweep4
mkdir -p "${OUT}" outputs/slurm

# tag:anchor:selector
# w_anchor=0 is the CONTROL: with cosine distance it must reproduce the old inert
# behaviour (~0.257), which proves the 1/D scaling bug was the cause.  If a0_ctrl and
# a5 agree, the anchor is still not reaching the head and the fix is incomplete.
# sweep2 established the trend: raw vs_cl rises monotonically with w_anchor
#   a0_ctrl -0.3645 | a0p5 -0.2737 | a2 -0.1221 | a5 -0.0268 | a10 +0.0123
# so push the anchor until it dominates and find where this design saturates.
# sweep3 showed raw vs_cl saturating upward: a20 +0.0365, a50 +0.0408, a100 +0.0423,
# a200 +0.0489 -- already past hybrid's best condition (+0.0450).  Find where it peaks
# and, crucially, check whether rowcos starts overshooting the real bank (0.6275),
# which would mean the head is regressing toward the mean rather than predicting.
RUNS=(
  "a400_vscl:400.0:img_vs_cl"
  "a1000_vscl:1000.0:img_vs_cl"
)

for spec in "${RUNS[@]}"; do
  IFS=: read -r tag anchor sel <<<"${spec}"
  d="${OUT}/${tag}"
  if [[ -f "${d}/report.json" ]]; then
    echo "[SKIP] ${tag} (done)"
    continue
  fi
  echo "===== S1 ${tag}  w_anchor=${anchor}  select=${sel} @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/nw4_s1_train.py \
    --out "${d}" --test-subject 8 --epochs 30 \
    --w-anchor "${anchor}" --select-metric "${sel}" \
    --device cuda:0 --decode-rgb 0 \
    > "${OUT}/log_${tag}.log" 2>&1
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[WARN] ${tag} exited ${rc}; see ${OUT}/log_${tag}.log"
    tail -n 15 "${OUT}/log_${tag}.log"
    continue
  fi
  echo "[ok]   ${tag}"
  grep -o "best .*" "${OUT}/log_${tag}.log" | tail -1 || true
  tail -n 6 "${OUT}/log_${tag}.log"
done

# ------------------------------------------------------------------ the verdict
# Measure every trained head with the same metric the gate uses, on the TEST split
# (test is only READ here, never used for selection -- selection was val_b).
echo
echo "=============== condition gate (vs_cl) ==============="
"${PYTHON}" scripts/nda/nw4_gate_cond.py \
  --sweep "${OUT}" \
  --out outputs/nw4/sub-08/s1sweep4_vscl.json

echo "[nw4-s1sw] done @ $(date -Iseconds)"
