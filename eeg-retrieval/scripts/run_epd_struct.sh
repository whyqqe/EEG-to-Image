#!/usr/bin/env bash
# =============================================================================
# EEG Patch (epd) STRUCTURE TOWER, sub-08, TRAINING ONLY.
#
# One run, no generation, no seven metrics. The object of study is the structural
# tower itself, so the output is `_best.pt` plus `_result.json` and the training
# curve; nothing here consumes a diffusion model.
#
# ---------------------------------------------------------------------------
# The one axis that was never varied
# ---------------------------------------------------------------------------
# Three structural runs exist on this subject and all three regressed the SAME
# target: the fine 64x64 SDXL VAE latent, 16384 coordinates, normalised only by a
# per-channel SCALAR (`vae_normalisation.mean` has four entries -- it removes a
# global offset and leaves every spatial structure of the mean in place). The
# backbone was swept (DINOv3-B, DINOv2-L), the interface was swept (topography ->
# eegit) and the objective was swept (L1 + floor -> MSE). The target was held fixed
# throughout, and it is the axis with the largest measured leverage:
#
#   closed-form ridge on RAW EEG -> target, sub-08, 63ch, val 150-way (chance 0.67%)
#   from outputs/probe/layer_sweep/{centered,coarse}.json
#
#     target                              dim     val@1   test@1   margin    used by
#     vae @ 64x64, uncentred            16384      5.07     6.00   +0.0617   ALL THREE runs
#     vae @ 64x64, centred              16384      5.13     6.00   +0.1639   never
#     vae @ 16x16, centred               1024      6.07     8.50   +0.2731   never
#     vae @ 8x8,   centred                256      6.67     9.50   +0.3171   never  <- THIS RUN
#     vae @ 4x4,   centred                 64      6.53    11.50   +0.3623   unreachable
#
# `margin` is r(pred,gt) minus r(constant,gt) and is the quantity this codebase
# scores structural tensors by -- the probe's own verdict logic keys off its sign,
# not off top-1 (see run_epd_struct_probe.sh, which records that an earlier
# top-1-keyed version produced a verdict contradicting the rows printed above it).
# On that criterion all three runs were fitted to the WORST version of their target:
# +0.3171 against +0.0617 is 5.1x, split as +0.1639 for centring alone and the rest
# for the coarse end.
#
# ---------------------------------------------------------------------------
# WHY 8x8 AND NOT 4x4, and why a code change was needed to get here at all
# ---------------------------------------------------------------------------
# 4x4 is the top of the ladder by margin (+0.3623) and it is UNREACHABLE, for a
# reason that is arithmetic rather than a preference. On the pooled decoder branch
# the head is
#
#     proj: Linear(d, base_ch * base_hw**2)  ->  (B, base_ch, base_hw, base_hw)
#     then three x2 `_up_block`s  ->  (B, vae_ch, out_hw, out_hw)
#
# so `out_hw == base_hw * 8` is the architecture's own parameterisation, and the
# ONLY legal `out_hw` at the default `base_hw=8` is 64. `losses.py` raises "latent
# shape mismatch" when prediction and target disagree -- there is no resample of the
# target anywhere in the training path -- so `--struct-scale` and `--struct-out-hw`
# must agree, and with `base_hw` pinned at 8, `--struct-scale 64` was the only
# reachable value. 4x4 would need `base_hw = 0.5`.
#
# `base_hw` had no flag. `--struct-out-hw 8 --struct-scale 8` therefore constructs
# and then dies in the loss, and this was verified rather than reasoned about:
#
#     tokenizer     out_hw   result
#     topography    4        OK      dense=True  up_sizes=[]        (the geometry the probe killed)
#     eegit         8        RAISE   "out_hw 8 is not base_hw 8 upsampled x2 three times"
#     eegit         16       RAISE   same
#     eegit         64       OK      dense=False                    (the fine target only)
#
# because `StructureTower.dense = (tokenizer_kind == "topography")`, so the measured-
# good `eegit` geometry takes the pooled branch, and the pooled branch ignores the
# 14x5 token grid entirely -- it decodes the POOLED fusion vector and expands it from
# base_hw. That is why the coarse end of the ladder was unreachable with the geometry
# the probe says to ship, and it is why this run adds `--struct-base-hw` (model.py /
# train.py, additive, default 8, every existing config byte-identical) rather than
# settling for `--struct-scale 64 --struct-center`, which is reachable and worth only
# +0.1639 against +0.3171.
#
# 8x8 over 16x16 (margin +0.3171 against +0.2731) for the same reason, and over 4x4
# because 4x4 does not exist. It is also the highest `val@1` of the entire centred
# ladder (6.67), which matters because `val_vae_top1` is half of what selects the
# checkpoint. `--struct-base-hw 1` is what `out_hw 8` requires.
#
# ---------------------------------------------------------------------------
# `--struct-center`: the first time it is applied to the VAE target
# ---------------------------------------------------------------------------
# `--struct-center` subtracts the FIT-split per-pixel mean field, and it is the
# documented cause of both recorded collapses: with a mean-field-dominated target
# under L1, the conditional median IS the mean field, so a head that learned nothing
# was near-optimal and reported variance ratio 0.0068 with a healthy-looking loss.
# DA2 uses it -- for `depth`. No VAE-target run ever has. At the fine scale it is
# worth +0.0617 -> +0.1639 on its own, which is why the three previous runs are
# priced against the uncentred row.
#
# ---------------------------------------------------------------------------
# What is deliberately NOT changed
# ---------------------------------------------------------------------------
# Every other flag is `epd_opt_dino3_eegit`'s, the best structural run so far
# (vae_top1 3.67%, vae_var_ratio 0.3165 against the topography run's 0.0068):
#
#   interface  `--struct-tokenizer eegit`, 70 tokens, grid (14,5), image (3,224,80).
#              Measured: the topography geometry sits at the constant predictor on
#              all six probed tensors at initialisation AND after training
#              (margin -0.07..-0.10), while eegit reaches the raw-EEG ceiling.
#   trunk      DINOv3-B/16, kept. The probe's "the interface does not carry it"
#              verdict was measured on the TOPOGRAPHY checkpoint and has never been
#              re-run on the eegit one, so the trunk's own contribution is currently
#              unattributed -- but it is also not the flag with a 5.1x measurement
#              attached to it, and moving two unmeasured things at once buys nothing.
#   loss       `--vae-loss mse`. Every successful row of the probe is a closed-form L2
#              fit; there is no evidence anywhere in this project that L1 suffices on
#              these features.
#   schedule   the semantic tower's, verbatim: ours, not EEGiT's. `run_eegit_gate`
#              measured the released optimizer at -4 points here.
#
# The variance floor is ON (`--w-var 1.0 --var-margin 0.15`), and this is measured
# rather than cautious. DA2's header records that turning it off under centring+MSE
# was tried and refuted within 248 steps of a 2-epoch smoke: `vae_cos +0.000` and
# `vae_var 0.000`, both exactly zero, which is only consistent with a spatially FLAT
# prediction. Centring removes the mean-field attractor that made a near-constant
# output L1-optimal; it does not remove the collapse, because with a weak per-concept
# signal the conditional mean of a centred target is still the zero field. margin
# 0.15 fires only below 15% of the target's per-channel std, i.e. only on genuine
# collapse -- the ridge's own ratio at this target is 0.5142.
#
# ---------------------------------------------------------------------------
# Checkpoint selection is changed, and it is the one judgment call here
# ---------------------------------------------------------------------------
# `sel = val_top1 + struct_sel_w * val_vae_top1` on a 0-100 scale. The inherited
# weight is 0.5, and at 0.5 the structural term cannot move the decision: the
# semantic top1 spans roughly 15 points across the schedule while 0.5 * vae_top1
# spans about 3, so "best" was in practice the semantic optimum and the saved
# checkpoint was not the structural tower's best epoch -- the wrong checkpoint for a
# structural-tower run, and the one that would be exported next. At this target the
# two terms sit at ~38 and ~6.7, so w = 5 brings them into the same dynamic range.
# Unlike the target changes this is a judgment, not a measurement, and the per-epoch
# history in the log is the primary read if it turns out to have been the wrong one.
#
# ---------------------------------------------------------------------------
# Injection, stated rather than discovered later
# ---------------------------------------------------------------------------
# An 8x8x4 latent decodes to a 64x64 image, so it cannot be a 512px `img2img` init
# either. This target is injectable through a learned channel -- a zero-init residual
# on the UNet's `conv_in`, which upsamples the field to the latent grid itself -- and
# that channel does not exist yet. This run is training-only precisely because that
# decision is open, and scoring generation before it is settled would price a route
# this target cannot take.
#
# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
#   nothing to build: `outputs/struct_targets/coarse/{train,test}_vae_8.npy` exist and
#                     are CONTENT-VERIFIED below against the fine cache (a sampled
#                     8x8-block area-average, not a file-exists check). The DA2-run
#                     validation covered depth only, so the vae ladder is checked here.
#   train 100 epochs  ~1.5-2 h (two ViT-B towers, fp32, no autocast, batch 128)
# so ~2 h expected; 4 h requested so a slow node finishes rather than being killed
# with the checkpoint unwritten.
# =============================================================================
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"
mkdir -p outputs/slurm outputs/logs

export PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

SUBJ="${SUBJ:-8}"
# `dino3` by default for the reason in the header: the trunk is the unmeasured axis
# and this run does not try to settle it. mae|in21k stay reachable so the same target
# can be re-run on another trunk without editing this file.
ARCH="${ARCH:-dino3}"
case "${ARCH}" in
  dino3) STRUCT_BACKBONE="timm:dinov3_b16" ;;
  mae)   STRUCT_BACKBONE="timm:mae_b16"    ;;
  in21k) STRUCT_BACKBONE="timm:vit_b16_in21k_orig" ;;
  *) echo "[FATAL] ARCH must be dino3|mae|in21k, got '${ARCH}'" >&2; exit 2 ;;
esac

# The target's spatial scale. Named rather than a literal because it is the run's
# decisive number and it determines THREE flags that must agree:
# `--struct-scale`, `--struct-out-hw`, and `--struct-base-hw` (= out_hw / 8).
STRUCT_SCALE="${STRUCT_SCALE:-8}"
case "${STRUCT_SCALE}" in
  8|16|32|64) ;;
  # 4 is refused by the decoder, not by preference: it needs base_hw = 0.5. See the
  # reachability table in the header.
  4) echo "[FATAL] STRUCT_SCALE=4 is unreachable: out_hw 4 needs base_hw 0.5 and the" >&2
     echo "        pooled decoder's up-block chain is fixed at three x2 steps. Use 8." >&2
     exit 2 ;;
  *) echo "[FATAL] STRUCT_SCALE must be 8|16|32|64, got '${STRUCT_SCALE}'" >&2; exit 2 ;;
esac
STRUCT_BASE_HW=$(( STRUCT_SCALE / 8 ))
# `c` = centred, in the tag, so an uncentred run of the same target is never confused
# with this one on disk.
TAG="${TAG:-epd_struct_${ARCH}_eegit_vae${STRUCT_SCALE}c}"

OUT="${OUT:-${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")}"
TGT="${TGT:-${ROOT}/outputs/struct_targets}"
FEAT="${FEAT:-${ROOT}/outputs/features/clip_h14_layers}"
VAE_CACHE="${VAE_CACHE:-${TGT}/vae_cache}"
COARSE_ROOT="${COARSE_ROOT:-${TGT}/coarse}"

CKPT="${OUT}/${TAG}_best.pt"
RESULT="${OUT}/${TAG}_result.json"

SEED="${SEED:-2025}"

# /home is at 100% and a download there fails with ENOSPC partway through a run, so
# every cache is pinned to /project before torch or huggingface_hub is imported.
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

# ---------------------------------------------------------------- flags
# The semantic tower, byte-for-byte `run_epd_dual.sh`'s: this run changes the
# structural target and nothing on the semantic side, so the semantic numbers stay
# comparable with every earlier run and remain usable as a sanity signal.
sem_cfg=(
  --subject "${SUBJ}"
  --tag "${TAG}"
  --out-dir "${OUT}"
  --tokenizer eegit
  --patch-style time-region
  --patch-size 16
  --n-patches-w 14
  --channels all
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
  --target-features "${FEAT}"
  --target-layer block26
  --target-fusion single
)

sched_cfg=(
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
  --patience 0                    # 0 = disabled, so the cosine schedule completes
  --aug full
  --stage1-epochs 0               # MMD off
  --seed "${SEED}"
  --fit-diagnostic                # the fit ceiling, which separates "cannot fit"
                                  # from "fits but does not generalise" -- the exact
                                  # question the previous structural runs left open
)

struct_cfg=(
  --struct-backbone "${STRUCT_BACKBONE}"
  --struct-patch-size 16          # must equal the trunk's patch_embed kernel
  --struct-tokenizer eegit
  --struct-n-patches-w 14         # 14 x 16 = 224 along time, 5 x 16 = 80 regions
  --struct-layers 8 10 12
  --struct-fusion-mode uniform
  --struct-freeze-blocks 0
  --struct-drop 0.1
  --struct-base-ch 128
  --struct-field-ch 32
)

# The LRs belong with the target block, not above it: absolute, not multipliers of
# `--lr`, because the decoder is a randomly-initialised conv stack and cannot be
# trained at the pretrained trunk's step size.
struct_lr_cfg=(
  --struct-lr 5e-5
  --struct-head-lr 5e-4
)

target_cfg=(
  --struct-target vae
  --struct-scale "${STRUCT_SCALE}"      # which cache is LOADED
  --struct-out-hw "${STRUCT_SCALE}"     # where the head EMITS and the loss is scored.
                                        # Must equal --struct-scale: losses.py raises
                                        # "latent shape mismatch" otherwise, and there
                                        # is no resample of the target in the path.
  --struct-base-hw "${STRUCT_BASE_HW}"  # the pooled decoder's seed resolution; the
                                        # contract is out_hw == base_hw * 8, so this
                                        # is what makes a coarse out_hw legal at all
  --struct-center                       # subtract the fit-set mean field. Measured
                                        # +0.0617 -> +0.1639 at the fine scale, and
                                        # never applied to the vae target before.
  --coarse-root "${COARSE_ROOT}"
  --vae-latents "${VAE_CACHE}"          # provenance/export record; the cache actually
                                        # read at scale<64 is the coarse ladder
  --w-vae 1.0
  --vae-loss mse
  --w-var 1.0
  --var-margin 0.15
  --struct-sel-w 5.0                    # see the header: 0.5 makes selection semantic-only
)

train_cfg=("${sem_cfg[@]}" "${sched_cfg[@]}" "${struct_cfg[@]}" "${struct_lr_cfg[@]}" "${target_cfg[@]}")

# =============================================================================
if [[ "${1:-}" == "--dry-run" ]]; then
  # Flag-level validation without a GPU and without the dataset, so a bad combination
  # fails here rather than after an allocation has been granted. Note that
  # `--validate-only` returns BEFORE the model is built, so it does NOT prove the
  # head constructs; `scripts/test_epd_arch.py` covers that, and it is run against
  # this exact base_hw/out_hw pair.
  log "validating the exact flag list (no GPU, no dataset): TAG=${TAG} scale=${STRUCT_SCALE} base_hw=${STRUCT_BASE_HW}"
  "${PY}" -u "${ROOT}/scripts/epd/train.py" "${train_cfg[@]}" --validate-only || exit 1
  [[ -d "${FEAT}/train" ]] || die "no semantic target features at ${FEAT}/train"
  [[ -f "${COARSE_ROOT}/train_vae_${STRUCT_SCALE}.npy" ]] || die "no train_vae_${STRUCT_SCALE}.npy under ${COARSE_ROOT}"
  [[ -f "${COARSE_ROOT}/test_vae_${STRUCT_SCALE}.npy" ]]  || die "no test_vae_${STRUCT_SCALE}.npy under ${COARSE_ROOT}"
  log "dry run ok"
  exit 0
fi

mkdir -p "${OUT}" "${ROOT}/outputs/logs"
cd "${ROOT}"

log "=============================================================="
log "EEG Patch (epd) STRUCTURE TOWER, subject ${SUBJ}, tag ${TAG}  [TRAINING ONLY]"
log "  sem    : EEGiT patch image -> vit_b16_in21k_orig -> CLIP block26"
log "  struct : EEGiT geometry -> ${ARCH} (${STRUCT_BACKBONE}) -> vae @ ${STRUCT_SCALE}x${STRUCT_SCALE}, CENTRED"
log "           pooled decoder, base_hw ${STRUCT_BASE_HW} (out_hw ${STRUCT_SCALE} = base_hw x 8)"
log "  ckpt   : ${CKPT}"
log "  outputs: ${RESULT}"
log "=============================================================="

# -----------------------------------------------------------------------------
# [1] Structural target: existence AND content
# -----------------------------------------------------------------------------
# The coarse ladder is a deterministic area-average of the fine cache, and the DA2
# run's validation covered `depth` only. A `vae` ladder that were stale, mis-scaled
# or pooled over the wrong axis would train silently against the wrong target and
# report a plausible-looking loss, so the check here is numeric: recompute the
# pooling for a sample of rows and compare.
log "[1] verifying the coarse vae ladder against the fine cache"
"${PY}" - "${COARSE_ROOT}" "${VAE_CACHE}" "${STRUCT_SCALE}" <<'PYEOF' || die "coarse vae ladder failed verification"
import sys, numpy as np, pathlib

coarse_root, vae_cache, scale = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])

for split in ("train", "test"):
    c_path = coarse_root / f"{split}_vae_{scale}.npy"
    if not c_path.is_file():
        raise SystemExit(f"missing {c_path}: build the coarse ladder from the fine cache first")
    coarse = np.load(c_path, mmap_mode="r")
    if coarse.ndim != 4 or coarse.shape[1:] != (4, scale, scale):
        raise SystemExit(f"{c_path} has shape {coarse.shape}, expected (N,4,{scale},{scale})")

    fine = np.load(vae_cache / f"{split}_vae_latents_f16.npy", mmap_mode="r")
    if len(fine) != len(coarse):
        raise SystemExit(f"{split}: {len(fine)} fine rows against {len(coarse)} coarse rows")
    step = 64 // scale
    idx = np.linspace(0, len(fine) - 1, 6).astype(int)

    # (n,4,64,64) -> mean over each (step,step) block, for six sampled rows only.
    # A reshape of the whole cache would materialise 16540x4x64x64 floats to check
    # six of them.
    worst = 0.0
    f = np.asarray(fine[idx], dtype=np.float32)
    for j, row in enumerate(f):
        blocks = row.reshape(4, scale, step, scale, step).mean(axis=(2, 4))
        worst = max(worst, float(np.abs(blocks - np.asarray(coarse[idx[j]], dtype=np.float32)).max()))
    if worst > 1e-5:
        raise SystemExit(f"{split}: area-average(fine) disagrees with {c_path.name} by {worst:.3e} "
                         f"-- the ladder is stale, mis-scaled, or pooled over the wrong axis")
    print(f"  {split}: {coarse.shape} float{coarse.dtype.itemsize*8}, "
          f"area-avg(fine {fine.shape}) matches 6/{len(fine)} sampled rows to {worst:.1e}  ok")

print(f"  target verified: vae @ {scale}x{scale}, the fit-set mean field will be subtracted (--struct-center)")
PYEOF

# -----------------------------------------------------------------------------
# [2] Train
# -----------------------------------------------------------------------------
# The guard checks `_result.json` as well as the checkpoint. `train.py` saves a
# checkpoint as soon as the first epoch improves `sel`, so a killed run leaves a
# `_best.pt` behind that looks like a finished one; every downstream step then
# fails on the missing json. Two files, or retrain.
if [[ -f "${RESULT}" && -f "${CKPT}" ]]; then
  log "[2] training already complete (result json + checkpoint present), skipped"
else
  if [[ -f "${CKPT}" ]]; then
    log "[2] a checkpoint exists at ${CKPT} but there is no result json: a previous"
    log "    run was killed mid-training. Retraining from scratch and overwriting it."
  fi
  log "[2] training TAG=${TAG} -> ${CKPT}"
  "${PY}" -u "${ROOT}/scripts/epd/train.py" "${train_cfg[@]}" || die "training failed"
  [[ -f "${CKPT}" ]] || die "no checkpoint at ${CKPT} after training"
fi
[[ -f "${RESULT}" ]] || die "no result json at ${RESULT}"

# -----------------------------------------------------------------------------
# [3] The training result, read back from the json rather than from the log
# -----------------------------------------------------------------------------
# No generation and no seven metrics: this is not an evaluation, it is the training
# run's own record, and the per-epoch history is what says whether the structural
# head learned the target or merely the fit-set mean.
log "[3] training result"
"${PY}" - "${RESULT}" "${TAG}" "${STRUCT_SCALE}" <<'PYEOF'
import json, sys

res, tag, scale = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(res))
bv = d.get("best_val", {}) or {}
te = d.get("test", {}) or {}
st = d.get("structure_tower", {}) or {}
tg = st.get("targets", {}) or {}
pr = st.get("params", {}) or {}


def num(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


print()
print(f"===== {tag} (training only) =====")
print(f"  target        vae @ {tg.get('scale')}x{tg.get('scale')}  "
      f"{'CENTRED on the fit-set mean field' if tg.get('centred') else 'NOT CENTRED'}  "
      f"field_mean_shape {tg.get('field_mean_shape')}")
print(f"  out_shape     {st.get('out_shape')}   field_shape {st.get('field_shape')}   "
      f"grid {st.get('patch_grid')} ({st.get('n_tokens')} tokens)")
print(f"  params        blocks {pr.get('struct_blocks')}  "
      f"interface {pr.get('struct_interface')}  heads {pr.get('struct_heads')}")
print(f"  selection     {st.get('selection')}")
print()
print(f"  SEMANTIC (sanity signal, not this run's object)")
print(f"    val  top1 {num(bv.get('top1'), 2):>8}  top5 {num(bv.get('top5'), 2):>8}  "
      f"epoch {bv.get('epoch')}  ema={bv.get('is_ema')}")
print(f"    test top1 {num(te.get('top1'), 2):>8}  top5 {num(te.get('top5'), 2):>8}  "
      f"mean_rank {num(te.get('mean_rank'), 2)}")
print()
print(f"  STRUCTURAL (this run's object; vae_top1 is n_concepts-way, chance = 1/n)")
print(f"    val  vae_top1 {num(bv.get('vae_top1'), 2):>8}  vae_cos {num(bv.get('vae_cos'), 4):>8}  "
      f"var_ratio {num(bv.get('vae_var_ratio'), 4):>8}")
if te:
    print(f"    test vae_top1 {num(te.get('vae_top1'), 2):>8}  vae_cos {num(te.get('vae_cos'), 4):>8}  "
          f"var_ratio {num(te.get('vae_var_ratio'), 4):>8}")
fit = d.get("fit_diagnostic") or d.get("fit") or {}
if fit:
    print(f"    fit  (ceiling) {json.dumps(fit)[:220]}")
print()

# The reference points this run is read against, printed so the comparison does not
# have to be reconstructed from another run's json.
print(f"  REFERENCE (closed-form ridge on raw EEG -> the SAME target, sub-08, 63ch, 150-way)")
if scale == 8:
    print(f"    val_top1 6.67   test_top1 9.50   margin +0.3171   var_ratio 0.5142")
else:
    print(f"    see outputs/probe/layer_sweep/centered.json for vae{scale}")
print(f"    the three previous structural runs sat BELOW their own linear baseline:")
print(f"    vae_top1 3.67 (dino3+eegit, fine uncentred) and 2.07 (da2 depth) against")
print(f"    5.07-5.13, i.e. these deep towers lost information a closed-form ridge keeps.")
print()

# What "did better" looks like, stated before the number is read.
print(f"  READ THIS AS: did the trained tower beat the linear ceiling above? A result at")
print(f"  or under it means the head is still discarding what the EEG carries, and the")
print(f"  next lever is the decoder's spatial path (the pooled branch never consumes the")
print(f"  14x5 token grid) rather than the target.")
print()
print(f"  ckpt  {res.replace('_result.json', '_best.pt')}")
PYEOF

log "done: ${RESULT}"
