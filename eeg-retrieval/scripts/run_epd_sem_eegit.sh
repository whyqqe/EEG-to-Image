#!/usr/bin/env bash
# =============================================================================
# epd_sem_eegit -- the semantic tower, trained on EEGiT's own objective
#
# ===========================================================================
# CORRECTION, added after the first run of this file was scored
# ===========================================================================
# An earlier revision of this header claimed that "EEGiT's objective is
# `--fixed-temp` ALONE", reading the paper's prose -- "The temperature parameter tau
# is fixed at 0.07, as in CLIP" -- and treating `--softplus` as SAMGA's convention
# that this codebase had measured as harmful. THE CLAIM WAS WRONG, and the run built
# on it was not EEGiT's objective.
#
# `run_eegit_gate.sh` transcribes the released `ClipLoss` / `PLModel.forward` line by
# line, and it says the opposite:
#
#     * `logit_scale = softplus(log(1/0.07))` = 2.727 -> the effective temperature
#       is 0.367, not 0.07. The paper's prose says tau=0.07; the code softpluses
#       it, which softens the objective ~5x.
#
# So the official CODE softpluses. The exp path (14.29) is the paper's prose, not the
# implementation, and running P3 without `--softplus` tested a temperature 5.24x too
# sharp on top of the loss's own free scale (see below). Measured consequence: the arm
# sat at chance (val_top1 0.60-0.73% against a 0.67% floor, loss 4.856 = ln(128), i.e.
# a uniform softmax) for SIXTEEN consecutive epochs before it moved at all, and
# `--fixed-temp` means the temperature could not compensate while `--no-eeg-l2norm`
# left `||z_e||` free, so the initial effective logit scale was
# `14.286 * sqrt(1024) = 457` against the official `2.727 * 32 = 87.3`.
#
# The lesson is narrower than "read the code": the two conventions differ by 5.2x and
# one of them is wrong, so the flag set has to be taken from the transcription rather
# than from the paper. `arm_flags` below now carries `--softplus` for every arm whose
# temperature is meant to be EEGiT's, which makes P3 the cell that was claimed but
# never run.
#
# The second element is the normalisation. EEGiT normalises only the image side
# (`img_z = img_z / img_z.norm(...)`) and passes the EEG embedding in raw, so the
# EEG embedding's norm stays free and acts as a per-sample logit scale on top of the
# fixed temperature. This codebase normalises both sides (`--no-eeg-l2norm`
# reproduces the released behaviour). The two elements were only ever changed
# together, inside `run_eegit_gate.sh`, and that job also changed the optimizer,
# un-flattened the lr, and dropped warmup and decay -- so its 39.50-vs-46.50 verdict
# prices a bundle, not these two flags.
#
# The arm set: a 2x2 factorial. Two corners are still unrun
# ---------------------------------------------------------
#                      | both-side L2-norm      | image-side only
#   -------------------+------------------------+--------------------------
#   learned tau        | A0 (exists, 52.5)      | P2 = `L2`    <- NEVER RUN
#   tau fixed (2.727)  | P1 = `TAU` <- NEVER RUN| P3 = `TAU_L2` = EEGiT exactly
#
# A0's checkpoint, export, 200 generations and seven metrics already exist in
# `epd_sem_*` under the same Pearson protocol this runner scores with, at the same
# seeds and the same `--split-seed 2025`, so the control is REUSED rather than
# retrained. That is not a shortcut around the comparison: all four corners differ in
# exactly these two flags and nothing else, `--split-seed` pins the val/test concept
# sets, and the per-concept vectors pair positionally. Re-training the control would
# buy seed noise we already have.
#
#      P1 `TAU`    --fixed-temp --softplus                  tau pinned, both normalised
#      P2 `L2`     --no-eeg-l2norm                          tau learned, image side only
#      P3 `TAU_L2` --fixed-temp --softplus --no-eeg-l2norm  EEGiT's objective, complete
#
# Everything else is byte-identical to A0. In particular the SCHEDULE is deliberately
# NOT EEGiT's: `run_eegit_gate.sh` measured the released optimizer (Adam, one flat
# 5e-5, no warmup, no decay) at 39.50 against 46.50 for the schedule here, and this
# job exists to isolate the objective from the schedule, not to re-litigate it.
# `--backbone-lr-mult 0.1` on `--lr 5e-4` already puts the ViT backbone at exactly
# EEGiT's stated 5e-5; the head keeps 5e-4 because it is randomly initialised.
#
# Batch size 1024 is deliberately excluded, and not only because of cost. At 16540
# training pairs that is 15 steps/epoch and 1500 steps total, while `--ema-decay
# 0.999` has a horizon of ~1000 steps -- the EMA that selection and saving both read
# would never converge. Making that arm faithful would require changing the EMA decay
# as well, so it is no longer a one-flag difference and cannot be interpreted
# alongside these three. It is a separate question with a separate design.
#
# Three seeds per arm
# -------------------
# `--seed` varies (2025/2026/2027) and `--split-seed` does NOT (fixed at 2025), so the
# arm effect is separable from init/augmentation noise and every arm is scored on the
# identical 150 val and 200 test concepts. Per-concept vectors are averaged over the
# three seeds within an arm before pairing.
#
# The condition is exported for generation, and the generation is measured
# ----------------------------------------------------------------------
# Steps [3]-[5] are the pure-semantic injection chain, unchanged from `sem_only`:
# `export_conds.py` writes the 1024-d EEG embedding as the IP-Adapter condition, the
# generator runs txt2img with the ControlNet at `--cn-scale 0.0`, and all four
# settings (28 steps, guidance 5.0, 512px, seed 42, `--control-guidance-end 0.5`) are
# identical to the arms this will be compared against. The `--cond-dir` still needs
# 200 readable PNGs because the generator loads it unconditionally; at scale 0 its
# contents are multiplied out, and the previous run's depth maps are reused for it.
#
# Step [6] also prints `cos(cond, oracle)`, the cosine between the arm's exported
# condition and the ground-truth CLIP joint embedding
# (`build_oracle_ip_cond.py`). This is the diagnostic that matters here: the A4
# result showed a tower can retrieve WORSE while generating just as well, so
# retrieval Top-1 is not a sufficient selection criterion for a generation condition.
# `cos(cond, oracle)` is. Python only, no GPU.
#
# Usage:
#   ARM=TAU_L2 SEED=2025 bash scripts/run_epd_sem_eegit.sh
#   ARM=TAU bash scripts/run_epd_sem_eegit.sh --dry-run
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
SUBJ="${SUBJ:-8}"
ARM="${ARM:?set ARM (TAU|L2|TAU_L2)}"
SEED="${SEED:?set SEED (2025|2026|2027)}"

TAG="epd_sem_eegit_${ARM}_s${SEED}"
OUT="${OUT:-${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")}"
TGT="${TGT:-${ROOT}/outputs/struct_targets}"
EXP="${EXP:-${OUT}/${TAG}_export}"
GEN="${GEN:-${OUT}/${TAG}_gen}"
MET="${MET:-${OUT}/epd_sem_eegit_metrics}"

FEAT="${FEAT:-${ROOT}/outputs/features/clip_h14_layers}"
CKPT="${CKPT:-${OUT}/${TAG}_best.pt}"

# Arm-independent: the IP-Adapter condition built from the ground-truth CLIP
# embedding of each test image. Only used for the step [6] diagnostic.
ORACLE_IP="${OUT}/epd_da2_depth8_export/conds/ip_oracle_test.npy"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[FATAL] %s\n' "$*" >&2; exit 1; }

# =============================================================================
# [0] configuration
# =============================================================================
# The three arms are one-line deltas from A0, and this is the only place they differ.
arm_flags() {
  case "${ARM}" in
    TAU)    echo "--fixed-temp --softplus" ;;                          # tau pinned at the released 2.727
    L2)     echo "--no-eeg-l2norm" ;;                                  # EEGiT's normalisation
    TAU_L2) echo "--fixed-temp --softplus --no-eeg-l2norm" ;;           # EEGiT's objective, all three items
    *)      die "unknown ARM=${ARM}; expected TAU|L2|TAU_L2" ;;
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
  --channels all
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
  --target-layer block26
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
  --epochs 100
  --batch-size 128
  --patience 0
  --aug full
  --stage1-epochs 0
)

# `--struct-*` is absent by construction: this is the pure semantic tower, and step
# [4] injects its output alone.
# shellcheck disable=SC2207
train_cfg=("${sem_common[@]}" "${sched[@]}" $(arm_flags))

# =============================================================================
# [1] dry run
# =============================================================================
if [[ "${1:-}" == "--dry-run" ]]; then
  log "validating ARM=${ARM} SEED=${SEED} (no GPU, no dataset)"
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
# The RESULT JSON is the completion marker, not the checkpoint. `train.py` saves
# `{tag}_best.pt` inside the epoch loop (the first epoch always beats the +/-inf
# baseline, so epoch 1 always writes one) and writes `{tag}_result.json` only at the
# very end of `main`, after training and the final test pass. A run killed mid-flight
# therefore leaves a 346MB checkpoint that looks finished and is not: it holds an
# untrained model with one epoch behind it.
#
# Guarding on the checkpoint alone is how job 608183 failed in four seconds -- it
# printed "checkpoint present, training skipped", skipped the training that had never
# happened, and then died one line later on the missing json. Checking BOTH is what
# makes this step idempotent for the right reason.
if [[ -f "${OUT}/${TAG}_result.json" && -f "${CKPT}" ]]; then
  log "[2] training already complete (result json + checkpoint present), skipped"
else
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
# [3] export the IP-Adapter condition (semantic only)
# =============================================================================
# No structure tower in this checkpoint, so `export_conds.py` writes the IP conditions
# alone -- it cross-checks `cfg["struct_backbone"]` against the built model in both
# directions and no longer hard-fails on a semantic-only checkpoint.
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
# The conditions are written BEFORE the report, so a run that dies on the report still
# leaves valid .npy files behind. Checking the report is what distinguishes "the export
# finished" from "the export wrote its conditions and then broke" -- an earlier
# revision of `export_conds.py` shadowed its output Path and raised only on that last
# line, and the pipeline reads the report back for provenance.
[[ -f "${EXP}/export_report.json" ]] || die "no export_report.json at ${EXP} (the export did not run to completion)"
"${PY}" - <<PYEOF || die "export_report.json is not valid JSON"
import json, pathlib
json.loads(pathlib.Path("${EXP}/export_report.json").read_text())
PYEOF

# =============================================================================
# [4] generate -- pure semantic injection, ControlNet at scale 0
# =============================================================================
# 200 readable PNGs for a ControlNet that contributes nothing. The previous run's
# depth conditions are the natural choice: already 512x512, and reusing them keeps
# this arm's generator inputs identical to every arm it will be compared against
# except for whose EEG built the IP.
gen_out="${GEN}/sem/generated"
gdir="${gen_out}/generated"
if [[ -f "${gdir}/199.png" ]]; then
  log "[4] 200 images present under ${gdir}, generation skipped"
else
  log "[4] generating 200 images -> ${gdir}"
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
OUT="${OUT}" EXP="${EXP}" MET="${MET}" ORACLE_IP="${ORACLE_IP}" \
"${PY}" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ["OUT"])
MET = Path(os.environ["MET"])
EXP = Path(os.environ["EXP"])
TAG = os.environ["TAG"]

res = json.loads((OUT / f"{TAG}_result.json").read_text())
test = res.get("test", {})
sel = res.get("best_val", {})
seven = json.loads((MET / f"{TAG}_seven.json").read_text())

# ---- condition quality ----------------------------------------------------
# How much of the ground-truth CLIP embedding the tower's condition actually
# carries. This is the number that predicts generated CLIP; retrieval Top-1 does
# not, as the A4 arm showed (worse retrieval, same generated CLIP).
cond_cos = None
oracle_p = Path(os.environ["ORACLE_IP"])
deploy_p = EXP / "conds" / "ip_deploy_test.npy"
if oracle_p.is_file() and deploy_p.is_file():
    o = np.load(oracle_p).astype(np.float32)
    c = np.load(deploy_p).astype(np.float32)
    o /= np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-8)
    c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
    cond_cos = float((o * c).sum(1).mean())
else:
    print(f"[warn] oracle condition or deploy condition missing; skipping cos(cond,oracle)")

print()
print(f"===== {TAG} =====")


def num(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


print(f"  val   top1 {num(sel.get('top1')):>8}  epoch {sel.get('epoch')!s:>4}  "
      f"ema={sel.get('is_ema')}")
print(f"  test  top1 {num(test.get('top1')):>8}  top5 {num(test.get('top5')):>8}  "
      f"mean_rank {num(test.get('mean_rank')):>8}")
if cond_cos is not None:
    print(f"  cos(cond, oracle)  {cond_cos:.4f}   <- the number that predicts generated CLIP")
print(f"  pixcorr {seven['pixcorr']:+.4f}  ssim {seven['ssim']:.4f}  "
      f"alex2 {seven['alex2']:.4f}  alex5 {seven['alex5']:.4f}")
print(f"  inception {seven['inception']:.4f}  clip {seven['clip']:.4f}  "
      f"swav {seven['swav']:.4f}  fid {seven['fid']:.2f}")
print()
PY
