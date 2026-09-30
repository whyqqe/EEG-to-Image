#!/usr/bin/env bash
# =============================================================================
# epd_sem -- the semantic arm, run as a ladder instead of a search
#
# Context for the arm set
# -----------------------
# Two measurements, made by the previous runs, fix what this job is allowed to vary.
#
# 1. The interface is already EEGiT's. `run_epd_da2.sh`'s semantic block is
#    `--tokenizer eegit --patch-style time-region` (their H=time/W=regions layout,
#    anterior->posterior, one 2D bilinear per region), the pretrained `patch_embed`
#    Conv2d as the interface, `--head-kind eegit` with `--fusion-mode none` and
#    `--pool mean`, d_embed 1024. So "adopt the EEGiT interface" is DONE, and the
#    remaining 50.0-vs-70.4 retrieval gap is not sitting in the tokenizer.
#
# 2. The rest of the EEGiT recipe costs, on this split. `run_eegit_gate.sh` measured
#    the released loss and optimizer (Adam, one flat 5e-5, no warmup, no decay,
#    softplus temperature, no EEG L2-norm) at 39.50 against 46.50 for the schedule
#    used here, and adopting more of the recipe made retrieval monotonically worse.
#    The schedule is therefore NOT under test. What was never tested is the loss
#    TEMPERATURE on its own, with the schedule held fixed -- the gate arms changed
#    the loss and the optimizer together, so the two cannot be separated from those
#    runs. `A3` below is that one-factor test.
#
# The five axes, each a single-flag change from `A0`
# --------------------------------------------------
#   A0  control: the shipped semantic block, `--struct-*` absent entirely.
#       Two jobs. It is the reference for A1..A4, AND it is the same-seed partner of
#       the recorded `sem_only` arm: that arm's images were produced by a semantic
#       tower trained JOINTLY with the DA2 structural tower, at seed 2025, with the
#       same semantic flags. So A0@2025 vs `sem_only`@2025 is a same-seed A/B of
#       "structural tower present vs absent" -- the architecture claim, measured
#       rather than asserted. The two towers are disjoint parameter sets, so the only
#       couplings are the global gradient clip and the shared LR schedule; this arm
#       prices them.
#   A1  `--epochs 25`. The recorded curve peaks at epoch 16-21 of 100 and then falls
#       34.1 -> 22.3 by epoch 100. Selection happens on 150 held-out concepts, so the
#       last 80 epochs are 80 additional chances to pick a lucky checkpoint: the
#       reported val_top1 of a 100-epoch run is optimistically biased in a way a
#       25-epoch run's is not. This tests whether the SHORTER schedule selects a
#       BETTER checkpoint, which is a different question from whether it trains less.
#   A2  `--channels occipito_parietal`. A closed-form ridge on the raw EEG scores
#       this set HIGHER than all 63 (27.5 vs 22.0 test Top-1 on the same split), and
#       SAMGA's intra-subject protocol uses exactly this set. It is not a pure channel
#       ablation: `resolve_target_plan` pins this set to 2 EEGiT regions, so the token
#       grid changes from 14x5=70 to 14x2=28 and the EEG image from 224x80 to 224x32.
#       Stated rather than hidden -- the arm tests "SAMGA's montage through EEGiT's
#       interface", not "17 channels with everything else equal".
#   A3  `--fixed-temp --softplus`. The released code's loss temperature:
#       `softplus(log(1/0.07))` = 2.727, i.e. an effective temperature of 0.367 rather
#       than 0.07, a ~5x softer objective, with tau pinned rather than learned. Given
#       that the recorded failure mode of this tower is overfitting (train batch
#       accuracy saturates while val decays), a softer objective is the one EEGiT
#       recipe element with a mechanism that could help HERE -- and it has never been
#       tested against this schedule in isolation.
#   A4  `--target-layer _pooled`. The alignment target's SPACE: the 1024-d CLIP joint
#       space that the generation stack actually consumes (`visual.proj`), instead of
#       block26's 1280-d residual-stream CLS. `img_head` currently has to learn a
#       1280->1024 map, and the export's softmax-over-gallery happens in the learned
#       1024-d space either way; only the target changes. The ridge probe preferred
#       block26 on the raw EEG (27.5 vs 25.0), so this arm tests whether that ordering
#       survives a trained tower and reaches the generated images.
#
# Three seeds per arm
# -------------------
# `--seed` varies (2025/2026/2027) and `--split-seed` does NOT (fixed at 2025 for
# every arm), because the point of the replication is to separate init/augmentation
# noise from the arm effect while keeping every arm scored on the identical 150 val
# and 200 test concepts. The reported per-concept vectors are then averaged over the
# three seeds within an arm before pairing, which keeps the pairing valid.
#
# Why this is affordable
# ----------------------
# 5 arms x 3 seeds = 15 tasks. One task is: train (~35 min, semantic trunk only, no
# 25M structural trunk), export the IP conditions (~2 min), generate ONE arm
# (~6.5 min), score it (~1 min). The tasks are independent and submitted as a SLURM
# array, so the wall clock is one task, not fifteen. `--array` indices map to
# (arm, seed) through the same table the task itself reads, so a single-task rerun
# after a node failure cannot silently become a different arm.
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
SUBJ="${SUBJ:-8}"
ARM="${ARM:?set ARM (A0|A1|A2|A3|A4)}"
SEED="${SEED:?set SEED (2025|2026|2027)}"

# Overridable so a re-run of the same arm can carry a different tag and therefore a
# different artefact set. It has to be `TAG` and not `OUT`/`EXP`/`GEN`: those are
# derived from the tag below, and overriding them one at a time is how a run ends up
# writing its checkpoint into one directory and reading its metrics from another.
# The default is unchanged, so every existing caller resolves the same paths.
TAG="${TAG:-epd_sem_${ARM}_s${SEED}}"
OUT="${OUT:-${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")}"
TGT="${TGT:-${ROOT}/outputs/struct_targets}"
EXP="${EXP:-${OUT}/${TAG}_export}"
GEN="${GEN:-${OUT}/${TAG}_gen}"
MET="${MET:-${OUT}/${TAG}_metrics}"

FEAT="${FEAT:-${ROOT}/outputs/features/clip_h14_layers}"
# The two-dimensional structure the SAME training run appends every epoch to. This
# arm trains one subject, so it is a vector; the per-epoch trajectory is what makes
# "does A1 select a better checkpoint" answerable from the log rather than from the
# final score alone.
CKPT="${CKPT:-${OUT}/${TAG}_best.pt}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[FATAL] %s\n' "$*" >&2; exit 1; }

# =============================================================================
# [0] configuration
# =============================================================================
# Exposed as a function so the dry run and the real run build the flag list the same
# way, and so the deltas below read as the one-line changes they are.
arm_flags() {
  case "${ARM}" in
    A0) : ;;
    A1) echo "--epochs 25" ;;
    A2) echo "--channels occipito_parietal" ;;
    A3) echo "--fixed-temp --softplus" ;;
    A4) echo "--target-layer _pooled" ;;
    *)  die "unknown ARM=${ARM}; expected A0|A1|A2|A3|A4" ;;
  esac
}

sem_common=(
  --subject "${SUBJ}"
  --tag "${TAG}"
  --out-dir "${OUT}"
  --seed "${SEED}"
  --split-seed 2025              # FIXED across arms: every arm sees the same split
  # ---- EEGiT's EEG patch representation, unchanged -------------------------
  --tokenizer eegit
  --patch-style time-region
  --patch-size 16
  --n-patches-w 14
  --channels all                 # A2 overrides this; see the header
  --backbone timm:vit_b16_in21k_orig
  --freeze-blocks 0
  # ---- EEGiT's readout: ONE layer, NO fusion -------------------------------
  --layers 12
  --fusion-mode none
  --pool mean
  --timm-global-pool avg
  --head-kind eegit
  --img-head-kind eegit
  --head-drop 0.5
  --d-embed 1024
  # ---- the alignment target ------------------------------------------------
  --target-features "${FEAT}"
  --target-layer block26         # A4 overrides this
  --target-fusion single
)

sched=(
  --optimizer adamw
  --lr 5e-4
  --backbone-lr-mult 0.1
  --warmup-epochs 5
  --cosine
  --min-lr-ratio 0.01
  --ema-decay 0.999
  --ema-warmup-steps 200
  --epochs 100                   # A1 overrides this
  --batch-size 128
  --patience 0
  --aug full
  --stage1-epochs 0
)

# `--struct-*` is absent by construction, and that is the architecture claim: the
# depth-ControlNet branch is removed. The evidence is the paired comparison of
# `depth_cn070` against `sem_only` (CLIP -0.086 with a paired CI excluding zero) and
# the oracle control, which shows a GROUND-TRUTH depth map through the same route is
# no better than the EEG one -- so the map's quality was never the binding term.
# shellcheck disable=SC2207
train_cfg=("${sem_common[@]}" "${sched[@]}" $(arm_flags))

# =============================================================================
# [1] dry run
# =============================================================================
if [[ "${1:-}" == "--dry-run" ]]; then
  log "validating ARM=${ARM} SEED=${SEED} (no GPU, no dataset)"
  # `--validate-only` parses and cross-checks the flag list without touching data or
  # the GPU, which is what catches a flag that does not exist and a combination that
  # `resolve_target_plan` refuses (e.g. `--head-kind eegit` with a fused encoder).
  "${PY}" -u "${ROOT}/scripts/epd/train.py" "${train_cfg[@]}" --validate-only \
    || die "flag validation failed for ARM=${ARM}"
  for f in "${ROOT}/scripts/epd/train.py" \
           "${ROOT}/scripts/epd/export_conds.py" \
           "${NB_ROOT}/scripts/nda/generate_struct_inject_decode.py" \
           "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py"; do
    [[ -f "$f" ]] || die "pipeline references a script that does not exist: $f"
  done
  [[ -d "${FEAT}/train" && -d "${FEAT}/test" ]] || die "missing target features under ${FEAT}"
  log "dry run ok; flags: $(arm_flags)"
  exit 0
fi

mkdir -p "${OUT}" "${GEN}" "${MET}"

# =============================================================================
# [2] train
# =============================================================================
if [[ -f "${OUT}/${TAG}_result.json" && -f "${CKPT}" ]]; then
  log "[2] training already complete (result json + checkpoint present), skipped"
else
  # The result json is the completion marker, not the checkpoint: `train.py` saves
  # `{tag}_best.pt` inside the epoch loop (epoch 1 always beats the +/-inf baseline, so
  # epoch 1 always writes one) and writes the json only at the end of `main`. Guarding
  # on the checkpoint alone makes a run killed mid-training report "training skipped",
  # skip the training that never happened, and die on the missing json -- which is
  # exactly how the sibling job `epd_sem_eegit`'s first submission failed in 4 seconds.
  if [[ -f "${CKPT}" ]]; then
    log "[2] a checkpoint exists at ${CKPT} but there is no result json: a previous"
    log "    run was killed mid-training. Retraining from scratch and overwriting it."
  fi
  log "[2] training ARM=${ARM} SEED=${SEED} -> ${CKPT}"
  "${PY}" -u "${ROOT}/scripts/epd/train.py" "${train_cfg[@]}" || die "training failed"
  [[ -f "${CKPT}" ]] || die "no checkpoint at ${CKPT} after training"
fi
[[ -f "${OUT}/${TAG}_result.json" ]] || die "no result json at ${OUT}/${TAG}_result.json"

# =============================================================================
# [3] export the IP-Adapter conditions (semantic only)
# =============================================================================
# `export_conds.py` used to hard-fail on a checkpoint with no structure tower, which
# is exactly this one; it now cross-checks `cfg["struct_backbone"]` against the built
# model in both directions and exports the IP conditions alone. There is no init and
# no depth map in this arm, so the generation below is txt2img with the ControlNet at
# scale 0. The `--cond-dir` still has to point at 200 readable PNGs because the
# generator loads it unconditionally; the depth-condition directory written by the
# previous run is reused for that, and at scale 0 its contents are multiplied out.
IP_DEPLOY="${EXP}/conds/ip_deploy_test.npy"
if [[ -f "${IP_DEPLOY}" ]]; then
  log "[3] IP conditions present, skipping export"
else
  log "[3] export IP conditions -> ${EXP}"
  "${PY}" -u "${ROOT}/scripts/epd/export_conds.py" \
    --ckpt "${CKPT}" \
    --out-dir "${EXP}" \
    --tag "${TAG}" \
    --arms deploy noise \
    --depth-dev-gain 1.0 \
    --device cuda:0 || die "export failed"
fi
[[ -f "${IP_DEPLOY}" ]] || die "no IP condition at ${IP_DEPLOY} after export"
[[ -f "${EXP}/conds/ip_noise_test.npy" ]] || die "no noise IP condition after export"
# The conditions are written BEFORE the report, so a run that dies on the report
# still leaves valid .npy files behind. Checking the report is what distinguishes
# "the export finished" from "the export wrote its conditions and then broke" --
# the pipeline reads the report back for provenance, and an earlier revision of
# `export_conds.py` shadowed the output Path and raised only on this last line.
[[ -f "${EXP}/export_report.json" ]] || die "no export_report.json at ${EXP} (the export did not run to completion)"
"${PY}" - <<PYEOF || die "export_report.json is not valid JSON"
import json, pathlib
json.loads(pathlib.Path("${EXP}/export_report.json").read_text())
PYEOF

# =============================================================================
# [4] generate
# =============================================================================
# Settings are byte-identical to the `sem_only` arm of the DA2 run, which is the
# comparison this arm exists to make: same generator, same steps, same guidance, same
# seed, same ControlNet (depth, scale 0), same `control-guidance-end`. The ONLY
# difference is whose EEG produced the IP condition.
#
#   `generate_struct_inject_decode.py` takes `--output-dir X` and writes the PNGs to
#   `X/generated`, so the images live one level BELOW the output dir. The variable
#   below is the directory that actually holds `000.png..199.png` and it is the one
#   the steps below read. Hiding this in a `gdir` that is NOT derived from the
#   output dir is how a previous revision of this family of scripts reported a
#   finished arm as absent and then failed at the metric step instead.
gen_out="${GEN}/sem/generated"
gdir="${gen_out}/generated"
if [[ -f "${gdir}/199.png" ]]; then
  log "[4] 200 images present under ${gdir}, generation skipped"
else
  log "[4] generating 200 images -> ${gdir}"
  # 200 readable PNGs for a ControlNet that contributes nothing. The previous run's
  # depth conditions are the natural choice: they are already 512x512, and using them
  # rather than regenerating a placeholder keeps this arm's inputs identical to the
  # `sem_only` arm's inputs except for the IP.
  COND_DIR="${OUT}/epd_da2_depth8_export/spatial/cond_depth_test_g1"
  [[ -f "${COND_DIR}/199.png" ]] || die "no 200-PNG condition dir at ${COND_DIR}"
  "${PY}" -u "${NB_ROOT}/scripts/nda/generate_struct_inject_decode.py" \
    --mode txt2img \
    --embed-npy "${EXP}/conds/ip_deploy_test.npy" \
    --cond-dir "${COND_DIR}" \
    --output-dir "${gen_out}" \
    --tag "${TAG}" \
    --control-type depth \
    --cn-scale 0.0 \
    --ip-scale 1.0 \
    --control-guidance-start 0.0 \
    --control-guidance-end 0.5 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --gen-size 512 \
    --seed 42 \
    --skip-metrics || die "generation failed"
fi
[[ -f "${gdir}/199.png" ]] || die "expected 200 PNGs under ${gdir}"

# =============================================================================
# [5] seven metrics
# =============================================================================
outj="${MET}/${TAG}_seven.json"
if [[ -f "${outj}" ]]; then
  log "[5] metrics present, skipping"
else
  log "[5] seven metrics (Pearson two-way, the official protocol)"
  "${PY}" -u "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
    --gen-dir "${gdir}" \
    --output-json "${outj}" \
    --tag "${TAG}" \
    --images-root "${IMAGES_ROOT}" \
    --device cuda:0 \
    --batch-size 16 || die "seven-metric evaluation failed"
fi
[[ -f "${outj}" ]] || die "no metrics at ${outj}"

# =============================================================================
# [6] this task's own line
# =============================================================================
TAG="${TAG}" ARM="${ARM}" SEED="${SEED}" CKPT="${CKPT}" \
OUT="${OUT}" EXP="${EXP}" MET="${MET}" "${PY}" - <<'PY'
import json
import os
from pathlib import Path

OUT = Path(os.environ["OUT"])
TAG = os.environ["TAG"]
res = json.loads((OUT / f"{TAG}_result.json").read_text(encoding="utf-8"))
met = json.loads(Path(os.environ["MET"], f"{TAG}_seven.json").read_text(encoding="utf-8"))
test = res["test"]
print(f"== {TAG} ==")
bv = res.get("best_val", {})
print(f"  retrieval test Top-1 {test['top1']:.2f} (top5 {test['top5']:.2f}, "
      f"mean_rank {test['mean_rank']:.2f}), selected at epoch {bv.get('epoch')} "
      f"(val Top-1 {bv.get('top1', float('nan')):.2f}, "
      f"from {'EMA' if bv.get('is_ema') else 'raw'} weights)")
# The per-concept vector is what makes this arm comparable to another one at a
# resolution the unpaired 200-way score cannot reach; assert it was written rather
# than discovering its absence in the aggregate step.
pc = test.get("per_concept")
if pc is None:
    raise SystemExit("[FATAL] result json has no `test.per_concept`; the cross-arm "
                     "paired comparison cannot be computed from this run")
print(f"  per-concept vector: {len(pc['top1'])} concepts, "
      f"mean {100.0 * sum(pc['top1']) / len(pc['top1']):.2f} (must equal the above)")
# The temperature the selected checkpoint was actually trained under. Printed
# because it is the number that explained two full rounds of arm comparisons after
# the fact, and it used to require reconstructing from LayerNorm gammas: a
# learnable temperature that converged elsewhere changes what "the same loss" means,
# and `temp_learnable` alone did not distinguish those cases.
crit = res.get("criterion") or {}
sel_s = crit.get("selected_effective_scale")
fin_s = crit.get("final_effective_scale")
ini_s = crit.get("init_effective_scale")
if sel_s is None:
    print("  criterion: not recorded (pre-instrumentation checkpoint)")
else:
    drift = "" if fin_s is None or not ini_s else \
        f", final {fin_s:.2f} ({fin_s / ini_s:.3f}x of init)"
    ls = crit.get("selected_logit_scale")
    ls_txt = "n/a" if ls is None else f"{ls:.4f}"
    print(f"  criterion: softplus={crit.get('softplus')} "
          f"init scale {ini_s:.2f} -> selected {sel_s:.2f} "
          f"(logit_scale {ls_txt}, at epoch {crit.get('selected_epoch')}){drift}")
print(f"  PixCorr {met['pixcorr']:.3f}  SSIM {met['ssim']:.3f}  "
      f"AlexNet(5) {met['alex5']:.3f}  Inception {met['inception']:.3f}  "
      f"CLIP {met['clip']:.3f}  SwAV {met['swav']:.3f}  FID {met['fid']:.1f}")
# The curve, not just the score: whether a shorter schedule selects a better
# checkpoint is a statement about where the maximum sits, and it can only be read
# off the trajectory. `epochs_run` shorter than `epochs` also means early stopping
# fired, which for `--patience 0` it should not.
hist = res.get("history", [])
if hist:
    best_i = max(range(len(hist)), key=lambda i: hist[i]["sel"])
    print(f"  curve: sel peaks at epoch {hist[best_i]['epoch']}/{res.get('epochs_run')} "
          f"(sel {hist[best_i]['sel']:.2f}, val_top1 {hist[best_i]['val_top1']:.2f}), "
          f"final epoch sel {hist[-1]['sel']:.2f}")
print(f"  train {res.get('timing', {}).get('wall_seconds', float('nan')):.0f}s, "
      f"{res.get('timing', {}).get('steps', '?')} steps")
PY

log "[done] $(date -Iseconds)"
