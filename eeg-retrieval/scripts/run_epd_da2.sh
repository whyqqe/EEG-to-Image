#!/usr/bin/env bash
# =============================================================================
# EEG Patch, one structural arm: Depth Anything V2 as the structural trunk,
# with its depth map fed to the decoder as a ControlNet-depth condition.
#
# What changed from `run_epd_dual.sh`, and why
# -------------------------------------------
# Three things, and each one is a measurement rather than a preference.
#
# 1. THE STRUCTURAL TRUNK IS A PRETRAINED DEPTH MODEL, NOT A TRUNK PLUS A
#    FROM-SCRATCH DECODER.
#    `StructureTower` owned a ViT trunk and a randomly-initialised conv decoder
#    that had to learn both "what a spatial field looks like" and "which one this
#    EEG implies" from 1504 training concepts. `DepthTower` is Depth Anything V2
#    whole -- DINOv2-S backbone, DPT neck, depth head -- with only its INPUT
#    interface replaced by EEG patches at patch 14. That is EEGiT's move applied
#    to a model that is about *where things are*, and it brings a pretrained
#    features-to-depth path with it.
#
# 2. THE STRUCTURAL TARGET IS DEPTH, CENTRED ON THE FIT-SET MEAN FIELD.
#    An earlier reading of `probe_targets.py` recorded depth as unreachable:
#    r(pred,gt) +0.16 against the constant map's +0.53, margin -0.37 on four
#    independent arms. That comparison was against an UNCENTRED target, and
#    `pearson_rows` centres each ROW -- so the "constant" that beat us was the
#    fit-set mean depth MAP, which is shape-similar to essentially every COCO
#    depth map because they all share one layout (far at the top, near at the
#    bottom). The ridge could not have expressed that mean map anyway: it is
#    z-scored per feature, so it has mean zero and no intercept, and `lam*` was
#    selected at 1e5 -- the heaviest shrink in the grid.
#
#    `--center-spatial` subtracts that free component and makes the constant
#    predictor the zero vector, whose row Pearson r is identically 0. Sub-08, all
#    63 channels:
#
#      target     r(pred,gt)   floor    margin    test top1
#      depth        +0.1872   +0.007   +0.1800       4.50
#      depth4       +0.1978   -0.024   +0.2219       5.00
#      depth8       +0.2147   -0.008   +0.2227       4.00     <- this run
#      depth16      +0.1703   -0.010   +0.1802       5.00
#      depth32      +0.1614   +0.006   +0.1558       5.50
#      vae @ 64     +0.1646   +0.001   +0.1639       6.00     <- what shipped before
#      vae4         +0.3672   +0.005   +0.3623      11.50
#
#    8x8 is chosen because the margin is flat from 4x4 to 64x64 while the
#    dimension count rises 256x: scoring finer spends almost all of the loss on
#    coordinates the EEG cannot address. Depth at 8x8 also beats the VAE-64 target
#    the previous run shipped (+0.2227 against +0.1639), and unlike a VAE latent it
#    is the condition ControlNet-depth was trained on.
#
#    The first submitted run of this arm proved the argument above was necessary but
#    NOT SUFFICIENT, and the missing piece was in the pretrained checkpoint's output
#    convention rather than in any of the three decisions:
#
#    THE RELEASED DEPTH HEAD ENDS IN A ReLU. `DepthAnythingForDepthEstimation` is a
#    relative-depth model (`depth_estimation_type="relative"`), so its head computes
#    `ReLU(conv3(x)) * max_depth` and CANNOT emit a negative number. The supervised
#    target here is mean-centred by `--struct-center`, which makes 51% of its entries
#    negative with every pixel needing both signs (per-pixel negative share 0.40..
#    0.69). A non-negative predictor of a zero-mean target can describe only half of
#    it, and MSE settles it at the smallest non-negative field available.
#
#    Measured on the epoch-21 checkpoint that the export would have consumed:
#
#      field range                    min +0.0000  max +0.4217  <- the ReLU signature
#      variance ratio                 0.0598   (target std 0.98)
#      per-sample r(own target)      +0.0421
#      same for the mean-field pred  +0.0880
#      margin over that floor        -0.0459   <- BELOW the constant predictor
#
#    So the structural condition would have been the mean field plus a small positive
#    residue for every sample, i.e. the `null_cn070` arm, and the run's own null pair
#    would have detected it -- 3 GPU-hours later.
#
#    The fix is to drop that activation (the ReLU encodes the relative-depth TARGET
#    SPACE, not the features-to-depth path this tower inherits) and pin `max_depth`
#    to 1 so the output scale is O(1) against the target. Both live in
#    `DA2EEGEncoder.__init__`, both are asserted by `test_epd_arch.py`, and the
#    pretrained conv path is untouched.
#
#    A SECOND DEFECT WAS FOUND IN THE INSTRUMENT, not the model: the `vae_*` columns
#    in the epoch log were computed from the RAW weights, while the checkpoint that
#    gets saved and exported is the EMA weights. The two disagreed by an order of
#    magnitude (raw `vae_var 0.708` at epoch 20 against the EMA checkpoint's measured
#    0.060 at epoch 21), so the printed gate said "the head is alive" about an
#    artifact in which it was below the constant floor. The log now prints the
#    selecting curve's structural numbers and labels the raw ones `[raw ...]`.
#
# 3. THE INJECTION IS A CONTROLNET RESIDUAL, NOT AN img2img INIT.
#    The previous arm decoded predicted VAE latents and fed them to SDEdit at
#    strength 0.80. Its own report says it COST CLIP 2-way (-0.015) while buying
#    PixCorr +0.021 -- and `noise_sdedit`, which consumes no EEG, reached PixCorr
#    0.105 against the structural arm's 0.109. So the PixCorr came from "an init
#    image exists", not from what was in it. A ControlNet condition is a residual:
#    at scale 0 it is a strict no-op, so the arm cannot be worse than its own
#    control except through the optimisation. That is what makes it measurable.
#
# The same correction explains the OLD COLLAPSE rather than merely permitting a
# retry. The old depth head was trained with `latent_l1` against the uncentred map,
# and the only normalisation applied was a per-channel SCALAR (`vae_mean` is
# reshaped to (C,1,1) in `AuxTargetDataset`), which removes a global offset and
# leaves the mean FIELD intact. Under L1 the conditional median of a target whose
# mean field dominates IS the mean field, so a head that learned nothing but the
# mean was near L1-optimal -- and the run reported exactly that: variance ratio
# 0.0068 with a healthy-looking loss.
#
# The arm matrix, and what each pair prices
# -----------------------------------------
#   sem_only      txt2img, CN 0.00, IP(deploy)   the semantic tower alone
#   depth_cn035   txt2img, CN 0.35, IP(deploy)   + the structural condition
#   depth_cn070   txt2img, CN 0.70, IP(deploy)   + the structural condition, harder
#   null_txt2img  txt2img, CN 0.00, IP(noise)    no EEG in the semantic path
#   null_cn070    txt2img, CN 0.70, IP(noise)    CN on, but its map comes from
#                                                zeroed EEG
#
# `depth_cn* - sem_only` prices the structural branch with the semantic condition
# held fixed. `null_cn070 - null_txt2img` prices the structural CONDITION with the
# semantic condition held fixed, which is the pair that separates "this depth map
# carries EEG information" from "having any depth map at all helps SDXL". Without
# that second pair a positive first difference would be unattributable: the mean
# field is a plausible scene prior even when it carries nothing per-concept.
#
# Cost: 100 epochs at 70 + 70 tokens is *cheaper* than the previous arm (the DA2
# trunk is 25M parameters against DINOv3-B's 86M), plus 5 generation arms instead
# of 4. ~3 h of GPU; the sbatch asks for 6.
#
# Usage:
#   bash scripts/run_epd_da2.sh --dry-run   # validate flags, touch nothing
#   bash scripts/run_epd_da2.sh             # the real thing
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
SUBJ="${SUBJ:-8}"

TAG="${TAG:-epd_da2_depth8}"
OUT="${OUT:-${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")}"
# Structural target caches live at a shared path, NOT under the subject's output
# dir: a depth map is a property of the stimulus, not of the recording.
TGT="${TGT:-${ROOT}/outputs/struct_targets}"
EXP="${EXP:-${OUT}/${TAG}_export}"
GEN="${GEN:-${OUT}/${TAG}_gen}"
MET="${MET:-${OUT}/${TAG}_metrics}"

FEAT="${FEAT:-${ROOT}/outputs/features/clip_h14_layers}"
DEPTH_CACHE="${DEPTH_CACHE:-${OUT}/patch_dual_targets/gt_depth}"
COARSE_ROOT="${COARSE_ROOT:-${TGT}/coarse}"

# The structural target's resolution. A named variable because the whole design
# rests on it and it must not drift between the training and export steps.
STRUCT_SCALE="${STRUCT_SCALE:-8}"
# ControlNet scales to sweep. 0.0 is the semantic-only arm and the reference every
# other number is read against; the generator accepts a scale of 0 in txt2img mode
# and multiplies the condition out before it reaches the residual, so the arm is a
# strict no-op rather than a differently-seeded run.
CN_MAIN="${CN_MAIN:-0.35}"
CN_HIGH="${CN_HIGH:-0.70}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[FATAL] %s\n' "$*" >&2; exit 1; }

# =============================================================================
# [0] configuration
# =============================================================================
# The semantic tower is EEGiT's, unchanged from `run_epd_dual.sh`, and it has to be:
# the structural branch is priced by the DIFFERENCE between two arms that share it,
# so any drift here invalidates the comparison against the recorded baselines.
#
#   deploy_txt2img (semantic only, previous run)  CLIP 0.795  PixCorr 0.088
#   deploy_sdedit  (semantic + VAE init)          CLIP 0.781  PixCorr 0.109
#   ATM (NeurIPS'24), sub-08                      CLIP 0.786  PixCorr 0.160
#
# The bar is therefore PixCorr toward 0.160 WITHOUT paying CLIP: the previous
# structural branch bought +0.021 PixCorr for -0.015 CLIP while its PixCorr sat at
# the noise-init level (0.109 against `noise_sdedit`'s 0.105).
sem_common=(
  --subject "${SUBJ}"
  --tag "${TAG}"
  --out-dir "${OUT}"
  # ---- EEGiT's EEG patch representation, exactly ---------------------------
  --tokenizer eegit
  --patch-style time-region
  --patch-size 16
  --n-patches-w 14               # 14 x 16 = 224 along time; 5 x 16 = 80 regions
  --channels all                 # 5 EEGiT regions -> 14 x 5 = 70 patches
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
  # ---- the alignment target: the one necessary deviation from EEGiT --------
  # Frozen CLIP ViT-H-14 block26 (1280-d, 1024-d after the joint projection).
  # EEGiT aligns to its own trainable ViT-B/16, so their target space bends toward
  # whatever the EEG can predict; this pipeline cannot do that, because the
  # generation stack consumes 1024-d CLIP joint embeddings and a learned space is
  # not one. Stated rather than tuned away.
  --target-features "${FEAT}"
  --target-layer block26
  --target-fusion single
)

# The schedule is ours, not EEGiT's. `run_eegit_gate.sh` measured the released
# recipe (Adam, one flat 5e-5, no warmup, no decay) as the worst of four arms --
# 35.50 against 46.50 -- on a split where adopting more of it made retrieval
# monotonically worse (z=-2.38, p=0.018).
sched=(
  --optimizer adamw
  --lr 5e-4                      # the new parts: heads
  --backbone-lr-mult 0.1         # the blocks at 5e-5, which IS EEGiT's encoder LR
  --warmup-epochs 5
  --cosine
  --min-lr-ratio 0.01
  --ema-decay 0.999
  --ema-warmup-steps 200
  --epochs 100
  --batch-size 128               # fp32, no autocast
  --patience 0                   # 0 = disabled, so the schedule completes
  --aug full
  --stage1-epochs 0              # MMD off
  --seed 2025
)

# The structural tower. Every flag that `run_epd_dual.sh` set for `StructureTower`
# -- `--struct-layers`, `--struct-fusion-mode`, `--struct-base-ch`,
# `--struct-field-ch` -- is absent rather than defaulted, and `train.py` refuses
# them for `da2` instead of ignoring them: they name a trunk-plus-decoder design
# this arm does not have, and a config that kept them would claim a layer fusion
# nothing reads.
struct_cfg=(
  --struct-arch da2
  --struct-backbone da2          # the flag that builds a tower at all
  --struct-patch-size 14         # Depth Anything V2's conv; not adjustable
  --struct-tokenizer eegit       # EEGiT's region-band geometry, as the semantic side
  --struct-n-patches-w 14        # 14 x 14 = 196 along time, 5 x 14 = 70 regions
  --struct-out-hw "${STRUCT_SCALE}"
  --struct-vae-ch 1              # one depth channel, not four latent channels
  --struct-drop 0.1
  --struct-freeze-blocks 0       # 25M trunk; the capacity brake is not needed
  --struct-lr 5e-5
  --struct-head-lr 5e-4
  --struct-target depth
  --struct-scale "${STRUCT_SCALE}"
  --struct-center                # THE fix; see the header
  --depth-cache "${DEPTH_CACHE}"
  --coarse-root "${COARSE_ROOT}"
  --vae-loss mse
  --w-vae 1.0
  # The variance floor is ON, and its margin is DERIVED from the probe rather than
  # copied. Turning it off was tried first, on the reasoning that MSE already
  # charges for noise, so the instance pressure would be in the loss instead of in
  # a term that can be gamed. A 2-epoch GPU smoke refuted that within 248 steps:
  # `vae_cos +0.000` and `vae_var 0.000`, both exactly zero, which is only
  # consistent with a prediction that is spatially FLAT per sample -- the
  # conditional mean of a centred target whose per-concept signal is weak. MSE
  # does charge for noise, but "flat at the global mean" is not noise, it is the
  # optimum of the unregularised problem, and no amount of MSE pressure removes it.
  #
  # The margin is 0.15 and not the usual 0.5, because 0.5 would be a floor the
  # input cannot reach: `probe_targets.py --center-spatial` puts a closed-form
  # ridge's across-sample variance ratio on depth@8x8 at 0.2372, i.e. the linear
  # ceiling is ~24% of the target's spread. A floor above that would be met by
  # inflating the prediction's variance rather than by predicting better, which is
  # the previous run's failure (var ratio 0.6577 carrying nothing, per-sample
  # r +0.063 against the constant's +0.157). 0.15 forbids the flat solution while
  # staying under what the input demonstrably contains.
  --w-var 1.0
  --var-margin 0.15
  --struct-sel-w 0.5
)

train_cfg=("${sem_common[@]}" "${sched[@]}" "${struct_cfg[@]}")

# =============================================================================
# [1] dry run
# =============================================================================
if [[ "${1:-}" == "--dry-run" ]]; then
  log "validating the exact flag list (no GPU, no dataset): TAG=${TAG}"
  "${PY}" -u "${ROOT}/scripts/epd/train.py" "${train_cfg[@]}" --validate-only || exit 1
  for f in "${ROOT}/scripts/epd/train.py" \
           "${ROOT}/scripts/epd/depth_tower.py" \
           "${ROOT}/scripts/epd/export_conds.py" \
           "${NB_ROOT}/scripts/nda/generate_struct_inject_decode.py" \
           "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py"; do
    [[ -f "$f" ]] || die "pipeline references a script that does not exist: $f"
  done
  [[ -d "${FEAT}/train" ]] || die "no semantic target features at ${FEAT}/train"
  [[ -f "${DEPTH_CACHE}/train_depth_64.npy" ]] || die "no depth cache at ${DEPTH_CACHE}"
  [[ -f "${COARSE_ROOT}/train_depth_${STRUCT_SCALE}.npy" ]] \
    || die "no ${STRUCT_SCALE}x${STRUCT_SCALE} depth target at ${COARSE_ROOT}"
  [[ -f "${COARSE_ROOT}/test_depth_${STRUCT_SCALE}.npy" ]] \
    || die "no test ${STRUCT_SCALE}x${STRUCT_SCALE} depth target at ${COARSE_ROOT}"
  # The training and the export MUST agree on the cache; the export needs it again
  # to run the collapse gate against a co-resolution ground truth. Checked here
  # because the failure would otherwise land after the training, in a step whose
  # only symptom is a missing gate.
  if [[ "${STRUCT_SCALE}" != "64" ]]; then
    log "note: the depth collapse gate will compare against ${COARSE_ROOT}/test_depth_${STRUCT_SCALE}.npy"
  fi
  log "dry run ok"
  exit 0
fi

# =============================================================================
# [2] train
# =============================================================================
mkdir -p "${EXP}" "${GEN}" "${MET}" "${ROOT}/outputs/logs"
cd "${ROOT}"

CKPT="${OUT}/${TAG}_best.pt"
if [[ ! -f "${CKPT}" ]]; then
  log "[2] train @ $(date -Iseconds)"
  log "  semantic  : EEGiT patches (70) -> vit_b16_in21k -> block26, no fusion"
  log "  struct    : EEGiT patches (70) -> Depth Anything V2 -> depth @ ${STRUCT_SCALE}x${STRUCT_SCALE}"
  log "  target    : depth, CENTRED on the fit-set per-pixel mean"
  log "  schedule  : ${TAG}, 100 epochs, batch 128, adamw, cosine"
  "${PY}" -u "${ROOT}/scripts/epd/train.py" "${train_cfg[@]}" || die "training failed"
else
  log "[2] checkpoint present, training skipped: ${CKPT}"
fi
[[ -f "${CKPT}" ]] || die "no checkpoint at ${CKPT} after training"

# =============================================================================
# [3] export
# =============================================================================
# Two products: the IP-Adapter conditions (the semantic tower's, unchanged) and the
# ControlNet-depth conditioning PNGs (the structural tower's). The export also runs
# the across-sample collapse gate on the depth head BEFORE anything is generated,
# because a collapsed head is not a loud failure -- it writes a plausible-looking
# condition, decodes to a plausible-looking image, and completes the whole metric
# run.
COND_DIR="${EXP}/spatial/cond_depth_test_g1"
if [[ ! -f "${COND_DIR}/199.png" ]]; then
  log "[3] export @ $(date -Iseconds)"
  "${PY}" -u "${ROOT}/scripts/epd/export_conds.py" \
    --ckpt "${CKPT}" \
    --out-dir "${EXP}" \
    --tag "${TAG}" \
    --arms deploy noise \
    --depth-dev-gain 1.0 \
    --device cuda:0 || die "export failed"
else
  log "[3] conditions present, export skipped: ${COND_DIR}"
fi
[[ -f "${COND_DIR}/199.png" ]] || die "no depth conditions under ${COND_DIR}"
[[ -f "${EXP}/conds/ip_deploy_test.npy" ]] || die "no IP conditions"

# The contrast of the conditioning image is the number that says whether this is a
# condition or a flat card. Printed from the export report rather than recomputed,
# so the log and the report cannot disagree.
"${PY}" - <<PY || true
import json
from pathlib import Path
p = Path("${EXP}/export_report.json")
if p.is_file():
    r = json.loads(p.read_text(encoding="utf-8")).get("depth", {})
    if r:
        print(f"[3] depth condition: u8 mean {r.get('u8_mean'):.1f} "
              f"std {r.get('u8_std'):.1f} over display range "
              f"{[round(v, 4) for v in r.get('display_range', [])]}")
        g = r.get("collapse_gate", {})
        if g and "passed" in g:
            print(f"[3] depth collapse gate: var ratio {g['pred_var_ratio']:.4f} "
                  f"(floor {g['var_ratio_floor']:.2f}) margin "
                  f"{g['margin_over_constant']:+.4f} -> "
                  f"{'PASS' if g['passed'] else 'FAIL'}")
PY

# =============================================================================
# [4] generate
# =============================================================================
gen_arm() {
  local arm="$1" cn="$2" iparm="$3"
  # `generate_struct_inject_decode.py` takes `--output-dir X` and writes the PNGs to
  # `X/generated`, so an arm's images live one level BELOW its output dir. This is
  # the directory that actually holds `000.png..199.png`, and it is the one both the
  # idempotency check here and the step-[5] metrics must read. The `--output-dir` is
  # therefore `<arm>/generated` and NOT this variable: passing `gdir` here would make
  # the generator write to `<arm>/generated/generated/generated` while every guard
  # checks `<arm>/generated/generated`, i.e. it would regenerate on every resubmit and
  # then fail at step [5] with "expected 200 PNGs".
  #
  # The original revision had a second, separate bug in this same line: it used
  # `<arm>/generated` as BOTH the output dir and the guard, which does exist but
  # contains only the `generated/` subdirectory and a metrics.json -- so
  # `<arm>/generated/199.png` was never present, step [5] reported "no generations,
  # skipped" for all five arms, and the run finished with an empty table instead of an
  # error. Do not reintroduce a `gdir` that is not the generator's actual output.
  local gdir="${GEN}/${arm}/generated/generated"
  if [[ -f "${gdir}/199.png" ]]; then
    log "[4] ${arm}: present, skipped"
    return 0
  fi
  log "[4] ${arm}: txt2img cn=${cn} ip=${iparm}"
  # `--cond-dir` is loaded unconditionally by the generator even when the
  # ControlNet's scale is 0, so it has to point at a directory with 200 readable
  # PNGs. At scale 0 the condition is multiplied out before it reaches the
  # residual, so the load is harmless and no separate artifact is needed for a
  # path that contributes nothing.
  "${PY}" -u "${NB_ROOT}/scripts/nda/generate_struct_inject_decode.py" \
    --mode txt2img \
    --embed-npy "${EXP}/conds/ip_${iparm}_test.npy" \
    --cond-dir "${COND_DIR}" \
    --output-dir "${GEN}/${arm}/generated" \
    --tag "${arm}" \
    --control-type depth \
    --cn-scale "${cn}" \
    --ip-scale 1.0 \
    --control-guidance-start 0.0 \
    --control-guidance-end 0.5 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --gen-size 512 \
    --seed 42 \
    --skip-metrics || die "generation failed for arm ${arm}"
}

ARMS=""
if [[ "${SKIP_GEN:-0}" != "1" ]]; then
  log "[4] generate @ $(date -Iseconds)"
  # The pair the conclusion rests on goes first, so a wall-clock overrun still
  # leaves the comparison on disk.
  gen_arm "sem_only"    0.0            deploy
  gen_arm "depth_cn035" "${CN_MAIN}"   deploy
  gen_arm "depth_cn070" "${CN_HIGH}"   deploy
  # The controls. `null_cn070` is the one that makes the structural result
  # attributable: it holds the ControlNet, its scale and the whole generation
  # stack fixed and changes only whether the depth map came from EEG or from
  # zeroed EEG.
  gen_arm "null_txt2img" 0.0           noise
  gen_arm "null_cn070"   "${CN_HIGH}"  noise
  ARMS="sem_only depth_cn035 depth_cn070 null_txt2img null_cn070"
else
  log "[4] skipped by SKIP_GEN"
  ARMS="$(cd "${GEN}" 2>/dev/null && ls -d */ 2>/dev/null | tr -d '/' | tr '\n' ' ')"
fi

# =============================================================================
# [5] the seven standard metrics
# =============================================================================
# ATM / MindEye / CogCap protocol: PixCorr and SSIM on grey @425 with a gaussian
# (NOT the RGB@256 variant, which is not comparable to the published tables),
# AlexNet(2)/AlexNet(5)/Inception/CLIP as two-way identification, SwAV as mean
# correlation distance. FID is computed too but is not part of the seven.
if [[ "${SKIP_METRICS:-0}" != "1" ]]; then
  log "[5] seven metrics @ $(date -Iseconds)"
  for arm in ${ARMS}; do
    gdir="${GEN}/${arm}/generated/generated"
    # Hard failure rather than a skip. The failure mode this guards against is not a
    # missing generation (that is what `SKIP_GEN` and the generator's own `[SKIP]` are
    # for) but a path that resolves to nothing: the previous revision skipped all five
    # arms here and the run then died at the table with `no seven-metric JSONs found`,
    # which reads as "the experiment produced nothing" instead of "the paths are wrong".
    [[ -f "${gdir}/199.png" ]] || die "[5] ${arm}: expected 200 PNGs under ${gdir}"
    outj="${MET}/${arm}_seven.json"
    [[ -f "${outj}" ]] && { log "[5] ${arm}: metrics present, skipping"; continue; }
    "${PY}" -u "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
      --gen-dir "${gdir}" \
      --output-json "${outj}" \
      --tag "${arm}" \
      --images-root "${IMAGES_ROOT}" \
      --device cuda:0 \
      --batch-size 16 || die "seven-metric evaluation failed for arm ${arm}"
  done
  find "${GEN}" -type d -name '_twoway_cache' -exec rm -rf {} + 2>/dev/null || true
else
  log "[5] skipped by SKIP_METRICS"
fi

# =============================================================================
# [6] table
# =============================================================================
log "[6] summary @ $(date -Iseconds)"
TAG="${TAG}" SUBJ="${SUBJ}" OUT="${OUT}" MET="${MET}" GEN="${GEN}" EXP="${EXP}" \
STRUCT_SCALE="${STRUCT_SCALE}" "${PY}" - <<'PY'
import json, os
from pathlib import Path

met, exp = Path(os.environ["MET"]), Path(os.environ["EXP"])
rows = {p.stem.replace("_seven", ""): json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(met.glob("*_seven.json"))}
if not rows:
    raise SystemExit("[6] no seven-metric JSONs found; nothing to summarise")

# The keys are the ones `eval_official_seven_dir.py` actually writes, which are
# `alex2` / `alex5` rather than `alexnet2` / `alexnet5`. Checked against an existing
# result file rather than assumed: the previous revision of this table looked for
# `alexnet2` and would have printed an em dash for two of the seven columns while
# still looking like a complete table.
KEYS = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]
HDR = ["PixCorr↑", "SSIM↑", "AlexNet(2)↑", "AlexNet(5)↑", "Inception↑", "CLIP↑",
       "SwAV↓", "FID↓"]
ORDER = ["sem_only", "depth_cn035", "depth_cn070", "null_txt2img", "null_cn070"]


def get(r, k):
    for cand in (k, f"{k}_twoway"):
        if cand in r:
            return r[cand]
    d = r.get("metrics", r)
    if isinstance(d, dict):
        for cand in (k, f"{k}_twoway"):
            if cand in d:
                return d[cand]
    return None


print()
print("| arm | " + " | ".join(HDR) + " |")
print("|---|" + "---:|" * len(HDR))
for arm in [a for a in ORDER if a in rows] + [a for a in rows if a not in ORDER]:
    r = rows[arm]
    cells = []
    for k in KEYS:
        v = get(r, k)
        cells.append("—" if v is None else f"{float(v):.3f}")
    print(f"| {arm} | " + " | ".join(cells) + " |")

print()
print("## What the pairs say")
print()


def d(a, b, k):
    if a not in rows or b not in rows:
        return None
    va, vb = get(rows[a], k), get(rows[b], k)
    if va is None or vb is None:
        return None
    return float(va) - float(vb)


for hi, lo, label in (("depth_cn070", "sem_only",
                       "Structural condition (CN 0.70 - semantic only)"),
                      ("depth_cn035", "sem_only",
                       "Structural condition (CN 0.35 - semantic only)"),
                      ("null_cn070", "null_txt2img",
                       "Structural condition with NO EEG (null CN 0.70 - null)")):
    parts = []
    for k, nm in (("clip", "CLIP"), ("inception", "Inception"), ("pixcorr", "PixCorr"),
                  ("ssim", "SSIM"), ("alex5", "AlexNet(5)")):
        v = d(hi, lo, k)
        if v is not None:
            parts.append(f"{nm} {v:+.3f}")
    if parts:
        print(f"- {label}: " + ", ".join(parts))

# ---------------------------------------------------------------------------
# Paired significance on the two-way metrics.
#
# The point differences above are not readable on their own. For 2-way
# identification with n = 200 the score is the mean over the 200 concepts of a
# per-concept fraction, so its standard error is ~sqrt(p(1-p)/n) ~ 0.03 at p ~ 0.78
# -- every delta in the table above is of that order. The 200 * 199 comparisons
# inside one similarity matrix are NOT independent (they share the same 200 rows),
# so the effective n is 200 and not 39800; a confidence interval computed as if the
# comparisons were independent would be ~14x too narrow.
#
# The arms ARE paired: same 200 concepts, same order, same seed. So the right test is
# on the per-concept differences, which `eval_official_seven_dir.py` now writes as
# `<arm>_persample.json`. Two tests are reported because they fail differently:
# a paired bootstrap CI (sensitive to the size of the shift) and a sign test on the
# per-concept differences (sensitive only to how often the direction flips).
import numpy as _np

PAIRS = [("depth_cn070", "sem_only", "structural CN 0.70 vs semantic only"),
         ("depth_cn035", "sem_only", "structural CN 0.35 vs semantic only"),
         ("null_cn070", "null_txt2img", "structural condition, NO EEG"),
         ("depth_cn070", "null_cn070", "EEG depth vs zeroed-EEG depth at CN 0.70")]
TWKEYS = [("clip", "CLIP"), ("inception", "Inception"), ("alex5", "AlexNet(5)"),
          ("alex2", "AlexNet(2)")]


def q(arm, key):
    # The metrics script writes `<arm>_seven.json` and, alongside it,
    # `<arm>_seven_persample.json` (it derives the name with
    # `out_p.stem + "_persample"`). Both spellings are tried so a run whose files
    # were produced by the earlier `_persample`-only naming still reads.
    for cand in (f"{arm}_seven_persample.json", f"{arm}_persample.json"):
        p = met / cand
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            v = d.get("q", {}).get(key)
            return None if v is None else _np.asarray(v, dtype=_np.float64)
    return None


rng = _np.random.default_rng(0)
print()
print("## Paired tests (bootstrap 95% CI, 10k resamples over the 200 concepts)")
print()
print("| pair | metric | delta | 95% CI | sign test p | n |")
print("|---|---|---:|---|---:|---:|")
any_test = False
for hi, lo, label in PAIRS:
    for key, nm in TWKEYS:
        a, b = q(hi, key), q(lo, key)
        if a is None or b is None or a.shape != b.shape:
            continue
        any_test = True
        dv = a - b
        n = dv.size
        boots = _np.array([dv[rng.integers(0, n, n)].mean() for _ in range(10000)])
        lo_ci, hi_ci = _np.percentile(boots, [2.5, 97.5])
        # Exact-ish two-sided sign test without scipy: the number of non-zero
        # differences favouring `hi`, against Binomial(n_nonzero, 0.5).
        nz = dv[dv != 0]
        k = int((nz > 0).sum())
        m = nz.size
        if m == 0:
            p_sign = 1.0
        else:
            from math import comb
            tail = sum(comb(m, i) for i in range(0, min(k, m - k) + 1)) / (2.0 ** m)
            p_sign = min(1.0, 2.0 * tail)
        flag = "" if (lo_ci <= 0.0 <= hi_ci) else " *"
        print(f"| {label} | {nm} | {dv.mean():+.3f} | "
              f"[{lo_ci:+.3f}, {hi_ci:+.3f}]{flag} | {p_sign:.3f} | {n} |")

if not any_test:
    print("| (no per-sample files found -- nothing testable) | | | | | |")
print()
print("`*` marks a CI that excludes 0. Note that 4 metrics x 4 pairs = 16 tests are")
print("reported, so a single `*` at p ~ 0.04 is what one expects by chance.")

# The pair that has to be read together. If `depth_cn - sem_only` is positive while
# `null_cn - null` is of the same size, the gain came from the presence of a
# plausible depth map rather than from the EEG that produced it.
gain_cn = d("depth_cn070", "sem_only", "clip")
gain_null = d("null_cn070", "null_txt2img", "clip")
if gain_cn is not None and gain_null is not None:
    print()
    print(f"- The attributing comparison, on CLIP 2-way: with EEG depth the CN moves "
          f"{gain_cn:+.3f}; with zeroed-EEG depth it moves {gain_null:+.3f}. "
          f"The EEG-attributable part is the difference of those two, "
          f"{gain_cn - gain_null:+.3f}.")

r = {}
p = exp / "export_report.json"
if p.is_file():
    r = json.loads(p.read_text(encoding="utf-8"))
g = r.get("depth", {})
if g:
    print()
    print("## Provenance")
    print()
    print(f"- depth condition: u8 mean {g.get('u8_mean'):.1f} std {g.get('u8_std'):.1f}, "
          f"display range {[round(v, 4) for v in g.get('display_range', [])]}")
    cg = g.get("collapse_gate") or {}
    if "passed" in cg:
        print(f"- depth collapse gate: var ratio {cg['pred_var_ratio']:.4f} "
              f"(floor {cg['var_ratio_floor']:.2f}), margin "
              f"{cg['margin_over_constant']:+.4f} -> "
              f"{'PASS' if cg['passed'] else 'FAIL'}")
    else:
        print(f"- depth collapse gate: skipped ({cg.get('skipped', 'not run')})")
    for arm in ORDER:
        if arm in rows:
            print(f"- `{arm}`: {os.environ['GEN']}/{arm}/generated/generated")
PY

log "[done] $(date -Iseconds)"
