#!/bin/bash
# =====================================================================================
# v6c REPETITION-LEVER REPRODUCIBILITY -- one job, every checkpoint we have, CPU-cheap.
#
# WHY. `probe_rep_dose.py` on `v5-a1-k20` (the 35.50 anchor) produced a lever with all the
# properties a real mechanism has and none of the ones an artefact has:
#
#     fuse(T1best, T1best)          35.50   <- the null for "average two estimates"
#     fuse(T1best, T1 w/o recovery) 35.50   <- is it just 'recovery twice'? no
#     fuse(T1best, random)          14.00   <- fusion is NOT generically helpful; it punishes
#     fuse(T1best, random2)         13.00      an uninformative partner. So the +4.0pp is a
#                                              property of the PARTNER, not of the sum.
#     dose-response in R (reps/concept): 1 -> 21.30, 5 -> 32.70, 10 -> 36.40, 20 -> 38.10,
#                                        40 -> 38.50, 80 -> 39.50   (monotone, saturating)
#
# A monotone dose-response is the readout that separates "the repetition cloud is a second,
# lower-variance estimate of the same operator" from "the fusion normalisation happened to
# like these numbers". But it is ONE checkpoint on ONE fold, and this project has already
# been burned by single-arm results that did not survive seeds (`v5-ep2`'s -2.0pp, the
# `terms:[img]` attribution that held at k=10 and reversed at k=20). So before this lever is
# allowed anywhere near a SOTA claim it has to survive: 3 seeds of one config, and several
# different arms -- including the faithful-RAT arm whose encoder is damaged, which is the
# strongest available test of whether the lever depends on encoder quality at all.
#
# COST. Pure inference on frozen checkpoints: no training, no labels, no new data. On a GPU
# node the 16000-trial repetition pass is seconds (it is minutes only on the login node,
# which is why this is a job and not three more shell calls).
#
# OUTPUT. `outputs/probe/rep_lever_sweep.json` -- one row per checkpoint with the controls
# and the R-curve, so the claim can be stated as "x/y checkpoints, mean +a.b pp (SD c.d)"
# rather than as one number.
#
#     sbatch slurm/v6c_rep_lever.sbatch
# =====================================================================================
#SBATCH --job-name=v6c-rep-lever
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v6c-rep-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v6c-rep-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
TARGET_SUBJECT="${TARGET_SUBJECT:-8}"
mkdir -p "${ROOT}/outputs/slurm" "${ROOT}/outputs/probe"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/huggingface
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"   # node-local; see AGENTS.md §2.4
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

MASTER="${ROOT}/outputs/slurm/v6c-rep-${SLURM_JOB_ID}.log"
log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== v6c repetition-lever sweep job=${SLURM_JOB_ID} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# The checkpoint set is chosen to answer four separate reproducibility questions:
#   * same config, 3 seeds   -> v5-base_seed2025/2026/2027 and v5-rat-csls_seed*
#   * the 35.50 anchor       -> v5-a1-k20
#   * a different loss regime-> v5-ep2 (source-only episode)
#   * a damaged encoder      -> v6-rat (the faithful-RAT arm that regressed to 25.50), the
#     strongest test of whether the lever needs a healthy encoder or only a good statistic
CKPTS=(
  "outputs/stage1/v5-a1-k20/last.pt"
  "outputs/stage1/v5-base_seed2025/last.pt"
  "outputs/stage1/v5-base_seed2026/last.pt"
  "outputs/stage1/v5-base_seed2027/last.pt"
  "outputs/stage1/v5-rat-csls_seed2025/last.pt"
  "outputs/stage1/v5-rat-csls_seed2026/last.pt"
  "outputs/stage1/v5-rat-csls_seed2027/last.pt"
  "outputs/stage1/v5-ep2/last.pt"
  "outputs/stage1/v6-rat/last.pt"
)

SWEEP="${ROOT}/outputs/probe/rep_lever_sweep.json"
: > "${SWEEP}.parts"
FAILED=()
for ck in "${CKPTS[@]}"; do
  name="$(basename "$(dirname "${ck}")")"
  if [[ ! -s "${ROOT}/${ck}" ]]; then log "[skip] ${ck} missing"; continue; fi
  feat="${ROOT}/outputs/probe/c2_feats_${name}.npz"
  rows="${ROOT}/outputs/probe/c2_levers_${name}.json"
  dose="${ROOT}/outputs/probe/rep_dose_${name}.json"
  lf="${ROOT}/outputs/slurm/v6c-${name}-${SLURM_JOB_ID}.log"
  log "--- ckpt ${name} ---"
  rc=0
  "${PY}" scripts/probe_c2_levers.py --ckpt "${ROOT}/${ck}" \
      --target-subject "${TARGET_SUBJECT}" --gates 0.0 0.1 0.2 0.5 \
      --device cuda --rep-batch 512 --cache "${feat}" --out "${rows}" \
      >"${lf}" 2>&1 || rc=$?
  if [[ ${rc} -ne 0 ]]; then log "    c2_levers FAILED rc=${rc}"; tail -20 "${lf}" | tee -a "${MASTER}"; FAILED+=("${name}"); continue; fi
  rc=0
  "${PY}" scripts/probe_rep_dose.py --cache "${feat}" --out "${dose}" \
      >>"${lf}" 2>&1 || rc=$?
  if [[ ${rc} -ne 0 ]]; then log "    rep_dose FAILED rc=${rc}"; FAILED+=("${name}"); continue; fi
  "${PY}" - "${name}" "${rows}" "${dose}" >> "${SWEEP}.parts" <<'PY'
import json, sys
name, rows_p, dose_p = sys.argv[1], sys.argv[2], sys.argv[3]
rows, dose = json.load(open(rows_p)), json.load(open(dose_p))
rec = {
    "ckpt": name,
    "t1_raw": rows["rows"].get("T1 raw cosine", {}).get("top1"),
    "t1_csls": rows["rows"].get("T1 + CSLS", {}).get("top1"),
    "t1_best": rows["rows"].get("T1 + whiten + CSLS + recovery", {}).get("top1"),
    "t2_rep_whiten_csls": rows["rows"].get("T2 rep-whiten + CSLS (no gate)", {}).get("top1"),
    "t2_rep_whiten_recovery": rows["rows"].get("T2 rep-whiten + recovery + CSLS", {}).get("top1"),
    "fused_gain_top1": (dose["dose_curve"].get("80", {}).get("fused_top1_mean", 0.0)
                        - dose["controls"]["T1 best (no fusion)"]["top1"]),
    "fused_top5": dose["dose_curve"].get("80", {}).get("fused_top5_mean"),
    "ctrl_fuse_self": dose["controls"]["C2 fuse(T1best, T1best)"]["top1"],
    "ctrl_fuse_random": dose["controls"]["C1 fuse(T1best, random)"]["top1"],
    "curve": {r: round(v["fused_top1_mean"], 2) for r, v in dose["dose_curve"].items()},
    "verdict": dose["verdict"].split(":")[0],
}
print(json.dumps(rec))
PY
  log "    ok (fused gain $(grep -o '"fused_gain_top1": [-0-9.]*' "${SWEEP}.parts" | tail -1))"
done

log "=== summary ==="
"${PY}" - "${SWEEP}" <<'PY' | tee -a "${MASTER}"
import json, sys, numpy as np
parts = [json.loads(l) for l in open(sys.argv[1] + ".parts") if l.strip()]
json.dump(parts, open(sys.argv[1], "w"), indent=2)
print(f"\n{'checkpoint':<26} {'T1 raw':>7} {'T1 +CSLS':>9} {'T1 best':>8} "
      f"{'T2 alone':>9} {'fused':>7} {'gain':>6} {'ctrl:self':>10} {'ctrl:rand':>10}")
print("-" * 116)
for p in parts:
    best = p["t1_best"] or 0
    fused = best + p["fused_gain_top1"]
    print(f"{p['ckpt']:<26} {p['t1_raw']:>7.2f} {p['t1_csls']:>9.2f} {best:>8.2f} "
          f"{p['t2_rep_whiten_recovery']:>9.2f} {fused:>7.2f} "
          f"{p['fused_gain_top1']:>+6.2f} {p['ctrl_fuse_self']:>10.2f} "
          f"{p['ctrl_fuse_random']:>10.2f}")
print("-" * 116)
g = [p["fused_gain_top1"] for p in parts]
pos = [x for x in g if x > 0]
print(f"checkpoints with a POSITIVE fused gain: {len(pos)}/{len(g)}   "
      f"mean {np.mean(g):+.2f}pp  SD {np.std(g):.2f}pp  "
      f"(positive-only mean {np.mean(pos):+.2f}pp)" if g else "no rows")
print("\ncurve (fused Top-1 vs reps/concept):")
for p in parts:
    print(f"  {p['ckpt']:<26} " + "  ".join(f"R{k}={v}" for k, v in p["curve"].items()))
print("\nCONTROL  ctrl:self must equal T1 best (null for 'average two estimates'); "
      "ctrl:rand must be far BELOW T1 best (fusion punishes noise). "
      "If either fails, the lever is a fusion artefact.")
PY

if [[ ${#FAILED[@]} -gt 0 ]]; then log "FAILED: ${FAILED[*]}"; exit 1; fi
log "ALL OK (sweep ${SWEEP})"
