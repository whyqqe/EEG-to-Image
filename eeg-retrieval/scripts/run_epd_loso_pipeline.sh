#!/usr/bin/env bash
# =============================================================================
# run_epd_loso_pipeline -- from the published baseline to our own arm, sub-08
# =============================================================================
# One fold (sub-08 held out, nine sources in), several arms, each a full 50-epoch
# training followed by the deployment-side ladder. The point is that every arm is a
# SINGLE-VARIABLE change from the one above it, so a difference can be attributed.
#
# THE LADDER THIS REPRODUCES
# --------------------------
# SCORE's ablation (arXiv 2608.19134, Table 4) is the specification, and its shape is
# the reason this pipeline is ordered the way it is:
#
#     train   SAMGA objective            26.22 Top-1
#             + multi-positive           28.63        (+2.41)
#             + simulated recovery       29.33        (+0.70)
#     test    CSLS ranking               39.08        (+9.75)
#             + mean and scale           43.80        (+4.72)
#             + recovery                 50.98        (+7.18)
#             + identity regularization  53.23        (+2.25)
#
# 23.90 of the 27.01 points are on the test side, on FROZEN features. So this driver
# runs `--recovery` on every arm: the whole test-side ladder is CPU post-processing and
# costs nothing next to the training, and a run that reports only its Top-1 throws away
# the four numbers that matter most.
#
# THE ARMS
# --------
#   a0-pairwise   SAMGA's objective (five routed layers) + pairwise InfoNCE, on OUR
#                 stack. SCORE's "Original" row in method, but NOT the baseline:
#                 lr, batch, augmentation, EEG encoder and head width all differ from
#                 inter.sh (see "WHY a0 IS NOT THE BASELINE" below).
#   a1-multipos   a0 + `--multipos` (SCORE Eq. 1). The single training change whose
#                 effect SCORE measured as +2.41.
#
# Adding an arm is appending to ARMS below; nothing else keys off the tag.
#
# WHY a0 IS NOT THE BASELINE, AND WHERE THE REAL ONE IS
# ----------------------------------------------------
# The baseline of record is SAMGA's OWN code, run unmodified, by a different driver:
#
#     ./scripts/run_samga_official_baseline.sh      (slurm/samga_official_inter.sbatch)
#
# a0 exists to be a CONTROL, and its only job is to make a1 interpretable: the two arms
# differ by exactly one flag, so a0-to-a1 is attributable and a1 against the official
# baseline is not (that gap moves five things at once). It must not be quoted as "our
# baseline" or subtracted from SAMGA's published 26.22 -- the hyper-parameters and the
# encoder are not theirs. The reasoning for each difference is in the commit that split
# the two drivers; the short version is that a0 was configured before the official run
# existed, from `results/` conventions rather than from `inter.sh`.
#
# WHY THE TARGET IS ROUTED AND MULTI-LAYER, NOT `single`
# ------------------------------------------------------
# Two reasons, and the second is not obvious.
#   1. It is the baseline SAMGA and SCORE actually evaluate: "subject-aware
#      multi-granularity" is the method, not a detail. A `--target-fusion single`
#      baseline is a DIFFERENT, weaker baseline and comparing against it would inflate
#      our own arm.
#   2. `--multipos` is INERT without it. When every subject's target for a picture is
#      the same vector -- which is exactly what `single` or a plain tiled target gives
#      -- the multi-positive loss is algebraically EQUAL to the pairwise one: equal
#      logits, so the average over the positive set collapses to the single term the
#      pairwise loss already had, in both directions. `test_epd_multipos.py` measures
#      the value and gradient difference as exactly zero. So a1 differs from a0 ONLY
#      because the routed target makes each subject's copy a distinct vector. Turning
#      the flag on without the router would have produced a null result and would have
#      been reported as "multi-positive does not help".
#
# Five CLIP ViT-H-14 layers (22/24/26/28/30) stand in for SAMGA's five InternViT layers
# (20/24/28/32/36). We do not have InternViT features; this is the honest substitution
# and it is a real difference from the published setup, so the absolute number here is
# ours and not theirs. The ablation BETWEEN arms is unaffected because every arm uses
# the same five.
#
# MVNN, CHANNELS, SELECTION: NOT TUNABLE HERE
# -------------------------------------------
# All 63 channels, `--val-concepts 0`, `--select-last`, MVNN per role -- the protocol,
# transcribed in PROTOCOL_INTER.md and set by SCORE/SAMGA/'Shallow Alignment'. The
# header of run_epd_loso.sh carries the reasoning; it is not repeated.
#
# COST
# ----
# Each arm is 9 x 1654 x 10 = 148,860 rows per epoch, ~4.5x the intra-subject run.
# ARMS= is the knob for a partial run; the default runs both.
#
# USAGE
#   ./scripts/run_epd_loso_pipeline.sh --dry-run          # validate every arm's flags
#   ./scripts/run_epd_loso_pipeline.sh --smoke            # 1 short arm, end to end
#   ./scripts/run_epd_loso_pipeline.sh                    # all arms (default)
#   ARMS="a0-pairwise" ./scripts/run_epd_loso_pipeline.sh # just the control arm
#   sbatch slurm/epd_loso_pipeline.sbatch
#
# THE OTHER HALF OF THE COMPARISON IS A SEPARATE JOB
# -------------------------------------------------
# The baseline of record is the official implementation, submitted on its own:
#
#     sbatch slurm/samga_official_inter.sbatch
#
# The two jobs are independent on purpose -- they finish at different times and neither
# can wait on the other -- so the merged table is produced afterwards, on demand, by
#
#     python scripts/compare_inter_arms.py
#
# which reads THIS driver's per-arm jsons and the official run's result.csv together and
# prints them with the caveats attached (which of the official numbers is test-selected,
# and why the published 26.22 is a reference rather than a target here). Read its header
# before quoting any difference between the two.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"

# /home is at 100% capacity and a model download there fails with ENOSPC mid-run.
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"     # no downloads from a compute node
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ----------------------------------------------------------------------------- config
TARGET="${TARGET_SUBJECT:-8}"
# Every other subject, ascending. THE ORDER IS PART OF THE CONFIG: the per-subject
# embedding's row i means "the i-th subject listed here", not "sub-(i+1)".
SOURCES=(${SOURCE_SUBJECTS:-1 2 3 4 5 6 7 9 10})
SEED="${SEED:-2025}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-${BATCH_SIZE:-512}}"
ARMS="${ARMS:-a0-pairwise a1-multipos}"

OUT="${ROOT}/outputs/loso/sub$(printf '%02d' "${TARGET}")"
FEAT="${ROOT}/outputs/features/clip_h14_layers"
# SAMGA's five granularities, mapped onto CLIP ViT-H-14's depth. Same band, same width
# between layers; see the header for why this stands in for InternViT 20/24/28/32/36.
TLAYERS=(block22 block24 block26 block28 block30)

log() { printf '[pipe ] %s\n' "$*" >&2; }
die() { printf '[pipe ] FATAL: %s\n' "$*" >&2; exit 1; }

MODE="run"
case "${1:-}" in
  --dry-run) MODE="dry" ;;
  --smoke)   MODE="smoke" ;;
  "")        ;;
  *) die "unknown argument '$1' (expected --dry-run, --smoke or nothing)" ;;
esac

# ----------------------------------------------------------------------- shared flags
# The EEG representation: EEGiT's, validated in this codebase, identical in every arm.
sem=(
  --out-dir "${OUT}"
  --seed "${SEED}"
  --split-seed 2025
  # ---- inter-subject: nine sources, one holdout -------------------------------
  --source-subjects "${SOURCES[@]}"
  --target-subject "${TARGET}"
  --channels all                      # 63 -- the inter-subject finding, not 17
  --val-concepts 0                    # the published runs use all 1654 concepts
  --select-last                       # SCORE: "report the final epoch"
  # ---- EEGiT's EEG patch representation ---------------------------------------
  --tokenizer eegit
  --patch-style time-region
  --patch-size 16
  --n-patches-w 14
  --backbone timm:vit_b16_in21k_orig
  --freeze-blocks 0
  --layers 12
  --fusion-mode none
  --pool mean
  --timm-global-pool avg
  --head-kind eegit
  --img-head-kind eegit
  --head-drop 0.5
  --d-embed 1024
  # ---- SAMGA's subject-aware multi-granularity target -------------------------
  --target-features "${FEAT}"
  --target-layers "${TLAYERS[@]}"
  --target-fusion routed_sr
  # ---- MVNN: the inter-subject preprocessing this project was missing ---------
  --mvnn train                        # resolves per role inside load_loso
  --mvnn-shrinkage lw
)

# EEGiT's objective, transcribed from the released code. `--softplus` is not optional:
# the prose says tau=0.07, the code softpluses it to an effective 0.367.
obj=(
  --fixed-temp
  --softplus
  --no-eeg-l2norm
)

# The deployment-side ladder, on every arm. CPU, frozen features, ~seconds.
rec=(
  --recovery
  --rec-rho 0.1                       # SCORE's value; --rec-rho 0 also reported
  --rec-k 10                          # SCORE's ten CSLS neighbours
  --rec-max-landmarks 160             # SCORE's 12-to-160 range, top end
  # Left at 0 (gate off) for the FIRST run on purpose: an unguarded ladder measures
  # where the threshold is for our representation, and the gate's job is to act on
  # that measurement. Setting it blind would mean never seeing the failing regime.
  --rec-min-landmark-rate 0.0
)

# SAMGA's coarse-to-fine schedule. Stage 1 holds the lr and blends in MMD; stage 2
# drops to the contrastive term alone. inter.sh uses mmd_end 0.5, NOT the argparse
# default of 0.2 -- reading the default instead of the launcher is a real way to
# reproduce the wrong method. `--stage1-epochs` and `--cosine` are mutually exclusive
# (train.py refuses the pair) because SAMGA holds the lr flat inside a stage.
sched_base=(
  --optimizer adamw
  --lr 5e-4
  --backbone-lr-mult 0.1
  --warmup-epochs 5
  --ema-decay 0.999
  --ema-warmup-steps 200
  --epochs "${EPOCHS}"
  --batch-size "${BATCH}"
  --patience 0
  --aug full
  --stage1-epochs 20
  --mmd-start 0.9
  --mmd-end 0.5
)

if [[ "${MODE}" == "smoke" ]]; then
  # One arm, two epochs, a couple of hundred rows. Verifies the wiring of the new
  # pieces (multi-layer routed target, multipos grouping, the recovery block) on real
  # data, which `--dry-run` cannot: it only validates flags, and every bug found so far
  # in these paths was found at run time, not at parse time.
  ARMS="a0-pairwise"
  EPOCHS=2
  BATCH=64
  sched_base=(
    --optimizer adamw --lr 5e-4 --backbone-lr-mult 0.1
    --epochs 2 --batch-size 64 --patience 0 --aug none
    --stage1-epochs 1 --mmd-start 0.9 --mmd-end 0.5
    --limit-samples 512
  )
fi

# ---------------------------------------------------------------------------- the arms
# `arm_flags <tag>` prints the flags that make `tag` differ from the arm above it.
arm_flags() {
  case "$1" in
    a0-pairwise)
      # The control: SAMGA's target and objective on our stack, no multi-positive.
      ;;
    a1-multipos)
      # SCORE Eq. 1. Non-trivial ONLY because the target above is routed; see the
      # header. Costs one `torch.unique` per batch and reduces exactly to a0 when no
      # two rows in a batch share a stimulus.
      echo "--multipos"
      ;;
    *)
      die "unknown arm '$1'; known: a0-pairwise a1-multipos" ;;
  esac
}

arm_desc() {
  case "$1" in
    a0-pairwise) echo "CONTROL   SAMGA objective (pairwise), 5-layer routed target, our stack" ;;
    a1-multipos) echo "+multi-positive  SCORE Eq.1 (needs the routed target to be live)" ;;
  esac
}

# ------------------------------------------------------------------------- [1] dry run
if [[ "${MODE}" == "dry" ]]; then
  # `${#ARMS[@]}` is not the count here: ARMS is a plain string, not an array, and
  # under `set -u` that expansion errors out and lets the script fall through to the
  # training step -- which is how a "dry run" ended up asking the login node for a GPU.
  set -- ${ARMS}
  log "validating $# arm(s) against target sub-$(printf '%02d' "${TARGET}")"
  for arm in ${ARMS}; do
    extra=($(arm_flags "${arm}"))
    log "  ${arm}: ${extra[*]:-<no extra flags>}"
    "${PYTHON}" -u "${ROOT}/scripts/epd/train.py" \
      "${sem[@]}" "${obj[@]}" "${sched_base[@]}" "${rec[@]}" "${extra[@]}" \
      --tag "loso_sub$(printf '%02d' "${TARGET}")_${arm}" \
      --allow-cpu --validate-only \
      || die "flag validation failed for arm ${arm}"
  done
  [[ -d "${FEAT}/train" && -d "${FEAT}/test" ]] || die "missing target features under ${FEAT}"
  for L in "${TLAYERS[@]}"; do
    [[ -f "${FEAT}/train/${L}.npy" && -f "${FEAT}/test/${L}.npy" ]] \
      || die "missing ${L}.npy under ${FEAT}/{train,test}"
  done
  # The whiteners must already exist, or the first training job pays minutes of
  # single-threaded preprocessing on a GPU allocation for the same answer every time.
  for s in "${SOURCES[@]}"; do
    [[ -f "${ROOT}/outputs/cache/mvnn_W_sub$(printf '%02d' "$s")_all63_train_lw.npy" ]] \
      || die "no MVNN whitener for source sub-$(printf '%02d' "$s"); run scripts/epd/build_mvnn_cache.py"
  done
  [[ -f "${ROOT}/outputs/cache/mvnn_W_sub$(printf '%02d' "${TARGET}")_all63_test_lw.npy" ]] \
    || die "no MVNN whitener for held-out sub-$(printf '%02d' "${TARGET}"); run scripts/epd/build_mvnn_cache.py"
  log "dry run ok for: ${ARMS}"
  exit 0
fi

mkdir -p "${OUT}" "${ROOT}/outputs/slurm"

# ------------------------------------------------------------------------ [2] the arms
for arm in ${ARMS}; do
  TAG="loso_sub$(printf '%02d' "${TARGET}")_${arm}"
  RES="${OUT}/${TAG}_result.json"
  CKPT="${OUT}/${TAG}_best.pt"
  extra=($(arm_flags "${arm}"))

  log "=============================================================="
  log "arm ${arm}: $(arm_desc "${arm}")"
  log "  tag ${TAG}"
  log "=============================================================="

  # Idempotent on the RESULT JSON, not the checkpoint: train.py writes the checkpoint
  # inside the epoch loop (epoch 1 always beats the +/-inf baseline) and the json only
  # at the end of main, so a killed run leaves a checkpoint that looks finished.
  if [[ -f "${RES}" && -f "${CKPT}" ]]; then
    log "  already complete (result json + checkpoint), skipped"
  else
    [[ -f "${CKPT}" ]] && { log "  checkpoint without a result json: previous run was killed; retraining"; rm -f "${CKPT}"; }
    "${PYTHON}" -u "${ROOT}/scripts/epd/train.py" \
      "${sem[@]}" "${obj[@]}" "${sched_base[@]}" "${rec[@]}" "${extra[@]}" \
      --tag "${TAG}" || die "arm ${arm} failed"
    [[ -f "${CKPT}" ]] || die "arm ${arm}: no checkpoint at ${CKPT}"
  fi
  [[ -f "${RES}" ]] || die "arm ${arm}: no result json at ${RES}"
done

# ------------------------------------------------------------------------- [3] summary
"${PYTHON}" - "${OUT}" "${TARGET}" ${ARMS} <<'PY'
import json, sys
from pathlib import Path

out, tgt, arms = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
print(f"\n{'=' * 88}")
print(f"INTER-SUBJECT (LOSO) PIPELINE -- held out sub-{tgt:02d}   "
      f"{out}")
print(f"{'=' * 88}")

rows, refs = [], None
for arm in arms:
    p = out / f"loso_sub{tgt:02d}_{arm}_result.json"
    if not p.is_file():
        print(f"  {arm:<14s} MISSING ({p.name})")
        continue
    r = json.loads(p.read_text())
    t = r["test"]
    refs = refs or r.get("reference_sota") or {}
    rows.append((arm, t, t.get("recovery") or {}))

# The training-side headline first, because that is what the arm changed.
print(f"\n  {'arm':<14s} {'Top-1':>7s} {'Top-5':>7s}  {'r(+-)':>6s}  "
      f"{'mdiff':>6s}  {'mutual':>6s}")
for arm, t, rec in rows:
    ci = t.get("ci95") or [float("nan")] * 2
    m = (rec.get("recovery_diag") or {}).get("n_mutual_pairs")
    print(f"  {arm:<14s} {t['top1']:7.2f} {t['top5']:7.2f}  "
          f"{ci[1] - t['top1']:6.2f}  {t.get('min_detectable_diff', float('nan')):6.2f}  "
          f"{(m if m is not None else -1):6d}")

# The control question: is the difference between two arms larger than what this run
# can resolve? `min_detectable_diff` is the one-sided 95% threshold on a DIFFERENCE, so
# a gap below it is not evidence no matter which direction it points.
if len(rows) > 1 and len({r[1]["n"] for r in rows}) == 1:
    a, b = rows[0], rows[-1]
    gap = b[1]["top1"] - a[1]["top1"]
    thr = max(a[1].get("min_detectable_diff", 0.0), b[1].get("min_detectable_diff", 0.0))
    verdict = "ABOVE the resolution" if abs(gap) >= thr else "BELOW the resolution"
    print(f"\n  {b[0]} minus {a[0]}: {gap:+.2f} Top-1, threshold {thr:.2f} -> {verdict}")
    if abs(gap) < thr:
        print(f"  Read this as 'no measured difference', not as 'no effect': one fold on "
              f"one subject cannot resolve a gap this small.")

# The deployment ladder, per arm. Separate table because it is a different phase and
# does not depend on the training-side change.
if any(r[2] for r in rows):
    print(f"\n  SCORE Table 4 shape (frozen features, CPU), per arm:")
    print(f"  {'arm':<14s} {'cosine':>7s} {'CSLS':>7s} {'+mom':>7s} "
          f"{'rec rho0':>9s} {'+idreg':>7s} {'rate':>6s}")
    for arm, _t, rec in rows:
        if not rec:
            continue
        d = rec.get("recovery_diag") or {}
        abst = "  ABSTAINED" if d.get("abstained") else ""
        print(f"  {arm:<14s} {rec['cosine']:7.2f} {rec['csls']:7.2f} "
              f"{rec['moment_match_csls']:7.2f} {rec['recovery_rho0']:9.2f} "
              f"{rec['recovery']:7.2f} {d.get('landmark_rate', float('nan')):6.2f}{abst}")
    print(f"\n  'rec rho0' is SCORE's '+recovery' row; '+idreg' is their "
          f"'+identity regularization' row (rho=0.1).")
    print(f"  'rate' is the mutual-landmark rate -- the label-free proxy for whether "
          f"the pseudo-matches were trustworthy.")

print(f"\n  published, same protocol: " + ", ".join(
    f"{k} {v['top1']:.2f}" for k, v in refs.items()
    if isinstance(v, dict) and "top1" in v) if refs else "")
print(f"{'=' * 88}")
PY

log "[3] done -> ${OUT}"
