#!/usr/bin/env bash
# ============================================================================
# CHAB / sub-08 -- CHANNEL ABLATION: is the 17-electrode montage costing us?
#
# THE QUESTION
#   Every result this project has produced fed the encoder 17 posterior
#   electrodes.  `grep` across the tree finds that choice DESCRIBED but never
#   COMPARED -- there is no channel ablation in any log or report -- while the
#   published THINGS-EEG2 baselines explicitly keep all 63 ("All electrodes were
#   preserved for analysis").  So the montage is the one preprocessing parameter
#   where we knowingly differ from the literature, and it is untested.
#
#   The size of the stake is what forces this to be the next run: an independent
#   faithful reimplementation of the same protocol reports a PLAIN encoder at
#   61.2% Top-1 on 200-way, whereas our best single route on this subject is
#   0.4050 (vith_cat5, job 581704).  A 15-20pp gap attributed to a single
#   unexamined parameter is either the cheapest large win available or the
#   fastest way to rule out the cheapest explanation, and both outcomes pay.
#
# WHY IT IS A CLEAN ONE-VARIABLE EXPERIMENT
#   `EEGProjectWide.forward` flattens (channels, samples) row-major, so the
#   encoder's input projection is a flat weight and a montage change is purely a
#   width change of two tensors (`shared.model.0`, `specific.<sid>.net.0`).  The
#   17 posterior electrodes turn out to be the CONTIGUOUS TAIL BLOCK 46..62 of
#   the 63-channel order, so `warm_start_with_dilation` can rebuild the wider
#   layer exactly: old per-channel blocks at their new positions, zeros
#   elsewhere.  The dilated model therefore computes EXACTLY the old model on the
#   old electrodes, and the arms below start from a model that has lost nothing.
#   "Did the extra 46 electrodes help?" is then the only question the numbers can
#   answer -- not "did re-initialising hurt?".
#
# STAGES
#   0  correctness audit (`channel_ablation_audit.py`). HARD GATE. Proves the
#      dilation is exact on real EEG and that the negative controls move, so the
#      equivalence above is falsifiable rather than vacuous.
#   1  END-TO-END EQUIVALENCE. Materialise the dilated encoder with
#      `--warm-start-only`, then run the REAL exporter on it and compare its
#      `shared_r` against the pre-dilation export.  Stage 0 only proves equality
#      inside one process; this proves it through the actual artifact path that
#      every downstream script consumes, including the new montage guard.
#   2  `c63_warm`  -- dilated 63ch, then the SAME 30+15 epoch budget the
#      baseline encoder had.  Matched budget, non-zero start.
#   3  `c63_fresh` -- 63ch from scratch, same budget.  This is the literal
#      "train the architecture on all electrodes" arm, and it separates
#      "the dataset has 63 channels of signal" from "the 17-channel solution
#      extends well".
#   4  route probe per arm with the BASELINE's exact settings (13 legacy routes,
#      80 epochs, fuse-topk 4, val_top1 selection) so the only changed input is
#      the feature space.  No --banks: the expanded bank exists but has never run,
#      and mixing it in would confound the comparison with a second change.
#   5  table + verdict against job 581704.
#
# WHAT COUNTS AS AN ANSWER (pre-registered, before seeing the numbers)
#   Baseline to beat, sub-08 / 200-way, from job 581704 (frozen arm):
#       best single route vith_cat5      0.4050 raw / 0.4950 CSLS
#       4-route fusion                   0.3800 raw / 0.4900 CSLS / 0.6650 +Sinkhorn
#   * CHANNELS EXPLAIN IT  if vith_cat5 rises into the 0.55-0.65 band. Then the
#     montage was the defect, the fix is one flag, and the fusion work finally
#     sits on a competitive base -- after which the whole bank/imagenet-scale
#     multi-level program becomes worth re-running.
#   * CHANNELS DO NOT  if it lands within +-0.02. Then the 15-20pp lives in the
#     encoder architecture or the read-out objective, the montage question is
#     CLOSED (and stays at 17, which is cheaper), and the next experiment targets
#     the read-out instead. Either way this run removes a whole branch of doubt.
#   * PARTIAL (0.44-0.55) means both matter and the next run is the read-out arm.
#   The threshold is stated here, before the numbers exist, because a result read
#   off a curve that was chosen afterwards is not a result.
#
# RESUME
#   Every stage skips when its artifact exists, so a wall-time kill costs only
#   the interrupted stage.  Delete a directory to force just that stage.
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
SUBJ="${SUBJ:-8}"
STAG="sub-08"
SD="$(printf '%02d' "${SUBJ}")"
OUT="${OUT:-${NB_ROOT}/outputs/chab/${STAG}}"
DEVICE="${DEVICE:-cuda:0}"
PRE="${PRE:-30}"          # pretrain epochs -- the baseline encoder's budget
CAL="${CAL:-15}"          # calib epochs    -- the baseline encoder's budget
EPOCHS="${EPOCHS:-80}"    # route-probe epochs, matching the baseline probe
BASE_ENC="${BASE_ENC:-${NB_ROOT}/outputs/ocf/intra_enc/sub-${SD}}"
BASE_Z="${BASE_Z:-${NB_ROOT}/outputs/ocf/intra_z}"
BASE_PROBE="${BASE_PROBE:-${NB_ROOT}/outputs/cfmsf_all/sub-${SD}/frozen/probe/route_probe.json}"

cd "${NB_ROOT}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${OUT}"/{logs,probe}

PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
[[ -x "${PYTHON}" ]] || { echo "[FATAL] missing interpreter ${PYTHON}"; exit 1; }

log() { echo; echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }

# ---------------------------------------------------------------- preconditions
log "===== CHAB sub-${SD}: channel ablation (17 -> 63 electrodes) ====="
# SUMMARY_ONLY re-prints the verdict from artifacts already on disk. It exists for
# two reasons: a finished job whose summary text needed a fix (that happened --
# the first summary looked up a bare "mlp" fusion key while every probe since the
# fusion sweep writes "mlp|by=..|k=..", so a successful run printed nan), and a
# long job where re-deriving the table should not require a GPU allocation.
if [[ "${SUMMARY_ONLY:-0}" != "1" ]]; then
  "${PYTHON}" - <<PY
import torch, sys
if not torch.cuda.is_available():
    sys.exit("[FATAL] CUDA unavailable. A CPU fallback would train at a different "
             "speed and could be killed half-way, producing a mixed run.")
print(f"[env] torch {torch.__version__} cuda {torch.cuda.get_device_name(0)}")
PY
fi

require "${BASE_ENC}/checkpoint_ss_calib_best.pth"
require "${BASE_Z}/sub-${SD}/shared_r_train.npy"
require "${BASE_Z}/sub-${SD}/shared_r_test.npy"
require "${BASE_PROBE}"
require data/things_eeg/preprocessed_eeg/info.json

# ============================================================ STAGE 0: audit
if [[ "${SUMMARY_ONLY:-0}" != "1" ]]; then
log "===== [0/5] correctness audit: is the dilation exact and falsifiable? ====="
"${PYTHON}" scripts/nda/channel_ablation_audit.py \
  --subject "${SUBJ}" --checkpoint "${BASE_ENC}/checkpoint_ss_calib_best.pth" \
  2>&1 | tee "${OUT}/logs/audit.log"
# `set -o pipefail` + the audit's non-zero exit means a failed audit stops the job
# here, which is the point: no arm below is interpretable if dilation is not exact.

# ================================================ STAGE 1: end-to-end equality
# Materialise the dilated encoder WITHOUT training and push it through the real
# exporter. If the montage plumbing is right, its shared_r must match the
# pre-dilation export, because the extra 46 electrodes carry exactly zero weight.
W0="${OUT}/enc_warm0"
log "===== [1/5] end-to-end equivalence: dilated encoder -> real exporter ====="
if [[ ! -f "${W0}/checkpoint_ss_calib_best.pth" ]]; then
  "${PYTHON}" scripts/nda/nda_ss_pretrain.py \
    --output-dir "${W0}" --train-subjects "${SUBJ}" --calib-subject "${SUBJ}" \
    --channels all --init-ss-checkpoint "${BASE_ENC}/checkpoint_ss_calib_best.pth" \
    --warm-start-only --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/enc_warm0.log"
else
  echo "[SKIP] warm0 encoder exists"
fi
require "${W0}/checkpoint_ss_calib_best.pth"

Z0="${OUT}/z_warm0"
if [[ ! -f "${Z0}/sub-${SD}/shared_r_test.npy" ]]; then
  "${PYTHON}" scripts/nda/ocf_export_intra_z.py \
    --subject "${SUBJ}" --checkpoint "${W0}/checkpoint_ss_calib_best.pth" \
    --out "${Z0}" --channels all --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/export_warm0.log"
else
  echo "[SKIP] warm0 export exists"
fi

# The equality claim, measured on the files every downstream script reads.
"${PYTHON}" - <<PY | tee "${OUT}/logs/equivalence.log"
import json, sys
from pathlib import Path
import numpy as np

z0 = Path("${Z0}/sub-${SD}")
z1 = Path("${BASE_Z}/sub-${SD}")
fails = []
print("dilated-encoder export vs the 17-channel baseline export")
for tag in ("train", "test"):
    a = np.load(z0 / f"shared_r_{tag}.npy")
    b = np.load(z1 / f"shared_r_{tag}.npy")
    ok_shape = a.shape == b.shape
    dmax = float(np.abs(a - b).max()) if ok_shape else float("nan")
    drel = dmax / max(float(np.abs(b).max()), 1e-9) if ok_shape else float("nan")
    # float32 reduction order differs between a 17-wide and a 63-wide matmul, so
    # the bound is relative, not exact-zero
    ok = ok_shape and drel < 1e-5
    print(f"  [{'ok  ' if ok else 'FAIL'}] shared_r_{tag}: shape {a.shape} vs {b.shape}  "
          f"max|diff|={dmax:.3e} (rel {drel:.2e})")
    if not ok:
        fails.append(tag)
if fails:
    sys.exit("[FATAL] the dilated encoder does NOT reproduce the 17-channel export "
             f"for {fails}. The montage plumbing is wrong somewhere below the model "
             "(exporter channel selection, checkpoint field, or the dilation itself), "
             "so the trained arms would be unattributable. Stopping.")
print("EQUIVALENCE OK: extra electrodes carry zero weight through the real export path")
PY

# ============================================================ one arm at a time
# arm_encoder <arm> <extra nda_ss_pretrain flags...>
arm_encoder() {
  local arm="$1"; shift
  local d="${OUT}/enc_${arm}"
  if [[ -f "${d}/checkpoint_ss_calib_best.pth" ]]; then
    echo "[SKIP] encoder ${arm} exists"; return 0
  fi
  log "--- encoder ${arm}: ${*}"
  "${PYTHON}" scripts/nda/nda_ss_pretrain.py \
    --output-dir "${d}" --train-subjects "${SUBJ}" --calib-subject "${SUBJ}" \
    --channels all --pretrain-epochs "${PRE}" --calib-epochs "${CAL}" \
    --device "${DEVICE}" "$@" \
    2>&1 | tee "${OUT}/logs/enc_${arm}.log"
}

arm_export() {
  local arm="$1"
  local d="${OUT}/z_${arm}"
  if [[ -f "${d}/sub-${SD}/shared_r_test.npy" ]]; then
    echo "[SKIP] export ${arm} exists"; return 0
  fi
  log "--- export ${arm}"
  "${PYTHON}" scripts/nda/ocf_export_intra_z.py \
    --subject "${SUBJ}" --checkpoint "${OUT}/enc_${arm}/checkpoint_ss_calib_best.pth" \
    --out "${d}" --channels all --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/export_${arm}.log"
}

arm_probe() {
  local arm="$1"
  local d="${OUT}/probe_${arm}"
  if [[ -f "${d}/route_probe.json" ]]; then
    echo "[SKIP] probe ${arm} exists"; return 0
  fi
  log "--- route probe ${arm} (baseline settings: 13 routes, ${EPOCHS} epochs)"
  # Deliberately identical to the baseline probe except for the feature space:
  # same routes, same epochs, same fuse-topk, same val_top1 selection, no --banks.
  "${PYTHON}" scripts/nda/cfmsf_route_probe.py \
    --out "${d}" --test-subject "${SUBJ}" --epochs "${EPOCHS}" \
    --z-root "${OUT}/z_${arm}" --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/probe_${arm}.log"
}

arm() {
  local name="$1"; shift
  log "===== arm ${name} @ $(date -Iseconds) ====="
  arm_encoder "${name}" "$@"
  arm_export "${name}"
  arm_probe "${name}"
}

# 63ch, dilated warm start, matched epoch budget
arm c63_warm --init-ss-checkpoint "${BASE_ENC}/checkpoint_ss_calib_best.pth"
# 63ch from scratch, same budget
arm c63_fresh
fi   # end of SUMMARY_ONLY gate for the training stages

# ================================================================= summary
log "===== [5/5] verdict ====="
"${PYTHON}" - <<PY | tee "${OUT}/summary.txt"
import json
from pathlib import Path

OUT = Path("${OUT}")
BASE = Path("${BASE_PROBE}")

def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None

def best_single(r):
    if not r:
        return None
    return max(((k, v["mlp"]["top1"], v["mlp"]["top1_csls"])
                for k, v in r["targets"].items()), key=lambda t: t[1])

def _fusion_map(r):
    """{estimator|by|k -> entry}, tolerating both key conventions.

    The baseline probe (job 581704) predates the fusion-sweep change and writes a
    bare "mlp"; every probe since writes "mlp|by=<stat>|k=<n>".  The first version
    of this summary looked up a bare "mlp" only, so it printed nan/0 routes for
    both new arms -- the bug that made a successful run look empty.
    """
    return dict((r or {}).get("fusion") or {})

def fusion_same_rule(r, ref_key):
    """The apples-to-apples entry: the same selection rule the baseline used."""
    m = _fusion_map(r)
    if ref_key in m:
        return ref_key, m[ref_key]
    for k, v in m.items():
        if k.split("|")[0] == ref_key.split("|")[0] and "by=val_top1" in k and "k=4" in k:
            return k, v
    return None, None

def fusion_best(r):
    m = _fusion_map(r)
    if not m:
        return None, None
    k = max(m, key=lambda k: m[k]["csls"]["sinkhorn_top1"])
    return k, m[k]

rows = [("baseline_17ch", BASE)]
for arm in ("c63_warm", "c63_fresh"):
    rows.append((arm, OUT / f"probe_{arm}" / "route_probe.json"))
# the untrained dilated encoder is a pipeline control, not a contender
rows.append(("c63_warm0_CONTROL", OUT / "z_warm0" / "_no_probe_"))

print(f"{'arm':<20}{'best route':<22}{'top1':>8}{'csls':>8}"
      f"{'same-rule csls':>16}{'same-rule +sink':>17}{'n_rt':>5}")
print("-" * 96)
base_best = None
ref_key = "mlp"
for name, path in rows:
    r = load(path)
    if r is None and "CONTROL" not in name:
        print(f"{name:<20}{'(missing)':<22}{'-':>8}{'-':>8}{'-':>16}{'-':>17}{'-':>5}")
        continue
    if r is None:
        print(f"{name:<20}{'(pipeline control -- see logs/equivalence.log)':<68}")
        continue
    k, t1, cs = best_single(r)
    rk, fu = fusion_same_rule(r, ref_key)
    print(f"{name:<20}{k:<22}{t1:>8.4f}{cs:>8.4f}"
          f"{(fu['csls']['top1'] if fu else float('nan')):>16.4f}"
          f"{(fu['csls']['sinkhorn_top1'] if fu else float('nan')):>17.4f}"
          f"{(len(fu['routes']) if fu else 0):>5}")
    if name == "baseline_17ch":
        base_best = (k, t1, cs)
        if rk:
            ref_key = rk
        print(f"{'':<20}(baseline fusion key = {rk!r}; new probes write "
              f"{sorted(_fusion_map(r))[:1]} style keys)")

print()
print("best fusion rule available per arm (the fused composite is the reported "
      "metric, so this is where a real gain would show):")
for name, path in rows:
    r = load(path)
    if not r or not (r.get("fusion")):
        continue
    k, fu = fusion_best(r)
    print(f"  {name:<20}{k:<28}csls={fu['csls']['top1']:.4f} "
          f"+sinkhorn={fu['csls']['sinkhorn_top1']:.4f}")

print()
print("PRE-REGISTERED VERDICT (thresholds fixed before these numbers existed)")
print("  judged on the BEST single route, the statistic the threshold was set on --")
print("  kept as written rather than swapped for a friendlier one after the fact.")
if base_best:
    for arm in ("c63_warm", "c63_fresh"):
        r = load(OUT / f"probe_{arm}" / "route_probe.json")
        if not r:
            continue
        k, t1, cs = best_single(r)
        d = t1 - base_best[1]
        if d >= 0.15:
            v = "CHANNELS EXPLAIN IT -- montage was the defect; re-run the full pipeline at 63ch"
        elif d >= 0.04:
            v = "PARTIAL -- channels matter but do not close the gap; next run is the read-out arm"
        elif d > -0.02:
            v = "CHANNELS DO NOT EXPLAIN IT -- close the montage question, target the encoder/read-out"
        else:
            v = "63ch is WORSE -- the 17-channel montage is a working choice after all"
        print(f"  {arm}: best-route delta vs baseline {d:+.4f}  ->  {v}")

print()
print("PAIRED PER-ROUTE READING (the statistically meaningful version)")
print("  'best route' compares two maxima over 13 noisy routes, so it inherits the")
print("  winner's-curse.  Comparing every route against itself across arms removes")
print("  that, and is what the falsification should be judged on.")
base_routes = load(BASE)
if base_routes:
    for arm in ("c63_warm", "c63_fresh"):
        r = load(OUT / f"probe_{arm}" / "route_probe.json")
        if not r:
            continue
        for metric in ("top1", "top1_csls"):
            d, nz, pos = [], 0, 0
            for k in base_routes["targets"]:
                if k not in r["targets"]:
                    continue
                x = r["targets"][k]["mlp"][metric] - base_routes["targets"][k]["mlp"][metric]
                d.append(x)
                if x != 0:
                    nz += 1
                    pos += int(x > 0)
            mean = sum(d) / max(len(d), 1)
            print(f"  {arm:<10}{metric:<11} mean delta {mean:+.4f} over {len(d)} routes, "
                  f"wins {pos}/{nz}")
    print("  interpretation: a mean near 0 with wins near n/2 means the extra 46")
    print("  electrodes do not change what any individual route can retrieve, whatever")
    print("  the fused composite happens to do.")
print()
print("for reference, the published same-protocol range this is being judged against:")
print("  plain encoder baseline 61.2% Top-1 ; +multi-level blur targets 82.8-85.3%")
PY

log "===== done @ $(date -Iseconds) ====="
du -sh "${OUT}" | sed 's/^/[disk] /'
