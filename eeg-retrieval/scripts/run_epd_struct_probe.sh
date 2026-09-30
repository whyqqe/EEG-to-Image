#!/usr/bin/env bash
# =============================================================================
# THE STRUCTURAL-BRANCH PROBE. One question, asked properly.
#
# The structural tower has failed three times: variance ratio 0.0068 (a constant
# field), then 0.6577 with an instance contribution of +0.0002 (a scaled shared
# field), measured on sub-08 across three different pretrained backbones. Every one
# of those verdicts came from a TRAINED model, which is the wrong instrument for the
# question, because a bad end-to-end score cannot separate:
#
#   (a) the interface never let the information through, so nothing downstream could
#       have recovered it; from
#   (b) the information was in the features and the decoder / objective discarded it.
#
# A trained checkpoint is blind to the distinction BY CONSTRUCTION: whatever the loss
# dropped is absent from the features too. So the useful comparison is not "trained
# model vs constant" but "features BEFORE training vs features AFTER", both scored by
# the SAME closed-form probe that already measured the raw-EEG ceiling.
#
# The reference this is read against, from `probe_targets_all.json` (identical split,
# identical 200-way protocol, raw 63-channel EEG as the ridge input):
#
#   vae target    val top1 5.07   test top1 6.00   rank 24.8
#                 r(pred,gt) +0.1647  vs constant +0.1030  -> margin +0.0617
#                 across-sample variance ratio 0.4411
#
# So a linear map from the raw EEG achieves 5.07% instance Top-1 (chance 0.50%) on
# this target. Anything at chance is strictly worse than a linear map on the same
# signal, which is the fact that makes this a bug hunt and not a difficulty report.
#
# The rows, and what each one rules out
# ------------------------------------
#   raw@all        the published 5.07% reference, re-run under this script. A
#                  protocol check: if this does not reproduce ~5.07, nothing below
#                  is comparable and the run should be thrown away.
#   raw@4          the SAME input with 4 fit slots instead of 10. Every feature row
#                  is fitted at 4 slots (see the cost note), so this -- not raw@all --
#                  is the baseline the feature rows are read against.
#   pre_gridmean   the structural trunk's token grid, mean-pooled, at INITIALISATION.
#                  Tests (a) directly.
#   pre_pooled     the same trunk's pooled+fused vector, at initialisation.
#   pre_grid       the full 48x768 token grid, at initialisation (the exact tensor the
#                  convolutional decoder consumes).
#   tr_gridmean    the trained trunk's grid mean. Tests (b).
#   tr_pooled      the trained trunk's pooled vector.
#   tr_grid        the trained trunk's full grid.
#   tr_sem         the TRAINED SEMANTIC tower's 1024-d embedding -- a positive control
#                  that does not depend on the structural trunk at all. That tower
#                  scores 54.50 top-1 on its own target from the same EEG, so if it
#                  also fails to decode the VAE latent, the structural target is
#                  simply not linearly reachable through a deep ViT interface and the
#                  branch's premise needs rethinking rather than its loss.
#
# How to read the outcome
# -----------------------
#   pre_* ~ raw@4                     the interface is fine, the loss threw it away
#   pre_* ~ chance, tr_* ~ chance     the interface is the problem; no loss fixes it
#   pre_* >> tr_*                     training actively destroyed the information
#   tr_sem also at chance             the target is not reachable through any of our
#                                     deep interfaces, and the question changes
#
# Cost note: the ridge is fitted through a dual-SVD whose cost scales with the number
# of FIT ROWS cubed, not with the feature width, but the thin SVD of an n x D matrix
# still costs O(n^2 * D). At 10 fit slots the fit matrix is 15040 rows, and the
# 36864-dim grid row alone would be ~8e12 flops of SVD. At 4 slots it is 6016 rows,
# which is 6.25x cheaper across the board. That is why the feature rows all use
# --fit-slots 4 and why raw@4 exists: the cost is bought back as comparability, not
# as a shortcut.
#
# Usage:
#   CKPT=outputs/sub08/epd_dual_dino3_best.pt bash scripts/run_epd_struct_probe.sh
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
cd "${ROOT}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
SUBJ="${SUBJ:-8}"
CKPT="${CKPT:-${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")/epd_dual_dino3_best.pt}"
TAG="${TAG:-epd_struct_probe}"
OUTD="${OUTD:-${ROOT}/outputs/probe/${TAG}}"
FEATS="${FEATS:-${OUTD}/feats}"
PROBES="${PROBES:-${OUTD}/probes}"

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

[[ -f "${CKPT}" ]] || die "no checkpoint at ${CKPT}"
mkdir -p "${OUTD}" "${FEATS}" "${PROBES}"

# ---------------------------------------------------------------- extraction
# Both variants come out of ONE process on purpose: they share the data load, and
# `extract_probe_features.py` refuses to write a cache whose two variants are
# bit-identical (which would mean the checkpoint never loaded, and would make every
# comparison below a report of the same run twice).
if [[ ! -f "${FEATS}/manifest.json" ]]; then
  log "[feat] extracting trunk features (trained + pretrained) from $(basename "${CKPT}")"
  "${PY}" -u "${ROOT}/scripts/epd/extract_probe_features.py" \
    --subject "${SUBJ}" \
    --ckpt "${CKPT}" \
    --out-dir "${FEATS}" \
    --tag "${TAG}" \
    --device cuda:0 || die "feature extraction failed"
else
  log "[feat] ${FEATS}/manifest.json exists, reusing (delete it to re-extract)"
fi

# ---------------------------------------------------------------- probes
# One invocation per ridge input, because the probe fits its SVD on X and reuses it
# across targets -- a different input is a different invocation, not a different flag.
probe() {           # probe <label> <outfile> [extra args...]
  local label="$1" out="$2"; shift 2
  if [[ -f "${out}" ]]; then
    log "[probe] ${label}: ${out} exists, reusing"
    return 0
  fi
  log "[probe] ${label}"
  "${PY}" -u "${ROOT}/scripts/epd/probe_targets.py" \
    --subject "${SUBJ}" \
    --channels all \
    --targets vae \
    --lams 1e1 1e2 1e3 1e4 1e5 1e6 \
    --device cuda:0 \
    --out "${out}" "$@" 2>&1 | grep -E "^\[probe\] (vae|ridge|SVD|fit)|^\[skip\]" || true
}

# The published reference, at its original fit-slot count. Reproduces 5.07 or the
# protocol has drifted and nothing else in this run can be read.
probe "raw@10 (protocol check)" "${PROBES}/raw_slots10.json"

# The baseline every feature row is compared against.
probe "raw@4 (baseline)" "${PROBES}/raw_slots4.json" --fit-slots 4

for variant in pretrained trained; do
  for feat in struct_gridmean struct_pooled struct_grid; do
    probe "${variant}_${feat}" "${PROBES}/${variant}_${feat}.json" \
      --fit-slots 4 \
      --feature-cache "${FEATS}" --feature-source "${variant}_${feat}"
  done
done

# Positive control: a trunk that provably works, on a target that may not be
# reachable through it. See the header.
probe "trained_sem_embed (control)" "${PROBES}/trained_sem_embed.json" \
  --fit-slots 4 --feature-cache "${FEATS}" --feature-source "trained_sem_embed"

# ---------------------------------------------------------------- report
log "[report] assembling"
"${PY}" - "$PROBES" "$OUTD" <<'PYEOF'
import json
import sys
from pathlib import Path

probes, outd = Path(sys.argv[1]), Path(sys.argv[2])
order = ["raw_slots10", "raw_slots4",
         "pretrained_struct_gridmean", "pretrained_struct_pooled", "pretrained_struct_grid",
         "trained_struct_gridmean", "trained_struct_pooled", "trained_struct_grid",
         "trained_sem_embed"]

rows = []
for name in order:
    p = probes / f"{name}.json"
    if not p.is_file():
        print(f"[report] MISSING {p}")
        continue
    d = json.loads(p.read_text())
    r = next((x for x in d["results"] if x["target"] == "vae"), None)
    if r is None:
        print(f"[report] no `vae` row in {p}")
        continue
    rows.append({"label": name, "ridge_input": d.get("ridge_input", "?"),
                 "dim": r["dim"], "lam": r["lam"],
                 "val_top1": r["val_top1"], "test_top1": r["test_top1"],
                 "test_mean_rank": r["test_mean_rank"],
                 "r_margin": r.get("val_r_margin"), "var_ratio": r.get("val_pred_var_ratio")})

lines = []
lines.append("")
lines.append("=" * 108)
lines.append("STRUCTURAL-BRANCH PROBE  --  can the trunk carry the VAE latent? "
             "(closed-form ridge, val-selected lambda, 200-way)")
lines.append("=" * 108)
lines.append(f"  {'row':>28} {'D':>7} {'val top1':>9} {'test top1':>10} {'rank':>7} "
             f"{'margin':>9} {'var ratio':>10}")
for r in rows:
    m = f"{r['r_margin']:+.4f}" if r["r_margin"] is not None else "n/a"
    v = f"{r['var_ratio']:.4f}" if r["var_ratio"] is not None else "n/a"
    lines.append(f"  {r['label']:>28} {r['dim']:>7} {r['val_top1']:>9.2f} "
                 f"{r['test_top1']:>10.2f} {r['test_mean_rank']:>7.1f} {m:>9} {v:>10}")
lines.append("")
lines.append("  chance: top1 0.50, rank 100.5 (200-way); SE on top1 ~2.8 points.")
lines.append("  A row at chance is WORSE than a linear map on the raw EEG (5.07), which")
lines.append("  is the fact this probe exists to localise: interface or objective.")
lines.append("")

ref = next((r for r in rows if r["label"] == "raw_slots4"), None)
if ref is None:
    ref = next((r for r in rows if r["label"] == "raw_slots10"), None)
base = ref["val_top1"] if ref else None

def verdict(r):
    if not r["r_margin"] is None and r["r_margin"] <= 0.0:
        return "at/below the constant predictor -- carries nothing usable"
    if base is not None and r["val_top1"] >= 0.8 * base:
        return "comparable to the raw-EEG ceiling"
    if base is not None and r["val_top1"] >= 0.3 * base:
        return "carries the information, weakly"
    return "near chance -- the information is not in this tensor"

for r in rows:
    if r["label"].startswith("raw"):
        continue
    lines.append(f"  {r['label']:>28}: {verdict(r)}")

pre = [r for r in rows if r["label"].startswith("pretrained_")]
tr = [r for r in rows if r["label"].startswith("trained_struct")]
if pre and tr:
    best_pre = max(x["val_top1"] for x in pre)
    best_tr = max(x["val_top1"] for x in tr)
    lines.append("")
    lines.append(f"  best pretrained {best_pre:.2f} vs best trained {best_tr:.2f} "
                 f"-> delta {best_tr - best_pre:+.2f}")

    # The verdict is decided by the MARGIN, not by top-1. A margin at or below zero
    # says the predictor is no better than the constant map, which is the definitive
    # statement of "this tensor carries nothing usable" and is exactly what the
    # per-row verdicts above already report. An earlier version of this block keyed
    # off `top1 < 0.3 * raw_baseline` instead, and since the raw baseline at 4 fit
    # slots is only 3.67, a structural row at 1.40 val top1 (margin -0.0745, i.e. at
    # chance AND worse than the constant) slipped over that bound and was reported as
    # "the interface carries it" -- directly contradicting the line printed five rows
    # above it. Top-1 near a 0.5% chance floor is mostly noise; the margin is the
    # quantity that separates "carries something" from "carries nothing".
    def carries(r):
        return r["r_margin"] is not None and r["r_margin"] > 0.0

    n_pre, n_tr = sum(carries(x) for x in pre), sum(carries(x) for x in tr)
    if n_pre and n_tr:
        lines.append("  The interface DOES carry it, before and after training: the fault "
                     "is downstream of the features (the decoder), or in the objective.")
    elif n_pre and not n_tr:
        lines.append("  Pretrained carries it and trained does NOT: training destroyed it. "
                     "The objective is the fault; the geometry is fine.")
    else:
        lines.append("  The INTERFACE does not carry it -- at initialisation either, so no "
                     "objective could have extracted it. Changing the loss cannot help; "
                     "the tokenizer/trunk has to change.")
        if best_pre <= 2.0 and best_tr <= 2.0:
            lines.append(f"  (both maxima are ~{max(best_pre, best_tr):.2f} val top1 against a "
                         f"0.50 chance floor, and every margin is negative: these tensors "
                         f"are at the constant predictor, not merely weak.)")

print("\n".join(lines))
(outd / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
(outd / "summary.json").write_text(json.dumps(
    {"probe": "structural branch reachability", "rows": rows}, indent=2, default=float))
print(f"[report] wrote {outd / 'REPORT.txt'} and summary.json")
PYEOF

log "===== DONE @ $(date -Iseconds) ====="
