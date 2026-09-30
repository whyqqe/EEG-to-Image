#!/usr/bin/env bash
# =============================================================================
# EEG Patch dual tower -> SDXL -> the seven standard metrics, subject 08.
#
# One command, six stages. Stages are idempotent: a stage whose artefact already
# exists is skipped, so a requeue or a re-run after a failure resumes instead of
# rebuilding. Set SKIP_<STAGE>=1 to force a skip.
#
# -----------------------------------------------------------------------------
# What is being built
# -----------------------------------------------------------------------------
# Two EEG Patch towers over the SAME input interface, with different objectives:
#
#   semantic tower   EEGiT patch representation -> ViT-B/16 (ImageNet-21k)
#                    layers {8,10,12} fused uniformly -> 512-d shared space
#                    objective: InfoNCE against CLIP ViT-H-14 block26
#                    readout:   an IP-Adapter condition in CLIP's joint space
#
#   structure tower  EEGiT patch representation -> DINOv2-L (LVD-142M)
#                    layers {16,20,24} fused uniformly -> 8x8x128 field
#                    objective: L1 on SDXL VAE latents (x0.13025, standardised)
#                               + L1 + gradient-L1 on a monocular depth map
#                    readout:   a 512x512 SDEdit init image (VAE head, decoded)
#                               + a 512x512 ControlNet-depth condition (depth head)
#
# "Semantic to CLIP, structure to VAE" is the brief. Both towers are EEG Patch
# towers, i.e. both hand the EEG to a *pretrained Conv2d patch_embed* the way
# EEGiT does; that is the interface, and it is held fixed so the only thing that
# differs between the towers is what they are asked to predict.
#
# -----------------------------------------------------------------------------
# Why the structure tower has a trunk at all, and why it is mostly frozen
# -----------------------------------------------------------------------------
# The structural target is a dense 4x64x64 regression. Its ceiling is set by how
# much of the image layout is present in the EEG, not by decoder capacity, so the
# trunk is not where the risk or the win is. What the trunk *does* control is
# whether the run overfits: DINOv2-L is 304M parameters against 16540 training
# pairs (1654 concepts x 10 images), which is roughly 18k parameters per pair.
# `--struct-freeze-blocks 20` freezes blocks 1..20 and trains blocks 21..24, the
# per-layer projections, and the decoder -- about 60M live parameters, which is
# the only capacity brake available and is the one train.py's own help text names.
# The frozen blocks still run forward: they are the pretrained feature extractor,
# and their being frozen is not the same as their being unused.
#
# `--struct-layers 16 20 24` is the same shape of choice the semantic tower made
# (8/10/12 of 12 = 0.67, 0.83, 1.0 of depth), scaled to 24 blocks. It is a
# starting point, not a measured optimum: no per-layer scan exists for this target
# and until one does, matching the relative depths of the axis that *was* measured
# is the least arbitrary thing to do.
#
# -----------------------------------------------------------------------------
# Training strategy: EEGiT's, adapted to a two-tower model
# -----------------------------------------------------------------------------
# The semantic tower is set by `SEM` (see the config block). With `SEM=eegit` it is
# the released code's recipe exactly, verified against a transcription of that code
# by `scripts/test_eegit_official_interface.py`:
#   * EEG patch image and `pos_embed` bit-identical to the official construction;
#   * `global_pool='avg'` -> mean then `fc_norm`, single final block, official
#     ProjectionHead (Linear -> GELU -> Linear -> Dropout(0.5) -> + pre-GELU
#     projection -> LayerNorm);
#   * `torch.optim.Adam(wd=1e-4)` on all parameters, one flat LR of 5e-5, 100
#     epochs, no warmup and no decay;
#   * fixed temperature, softplus-ed to an effective logit scale of 2.727, and the
#     EEG embedding left unnormalised inside the loss.
#
# What is NOT taken from the official code, and why it is a bigger difference than
# any of the above: official EEGiT aligns EEG to its OWN trainable ViT-B/16 +
# ProjectionHead, so the target space bends toward whatever the EEG can predict. The
# semantic tower here aligns to frozen CLIP ViT-H/14 `block26`, because the
# generation stack consumes 1024-d CLIP joint embeddings and EEGiT's learned space
# is not one. This is the main reason to expect their reported retrieval number to be
# out of reach, and it is stated rather than tuned away.
#
# The structure tower shares the interface and the optimizer family, but keeps its
# own absolute LRs (`--struct-lr 5e-5`, `--struct-head-lr 5e-4`). Its decoder is
# ~40M randomly-initialised parameters against 16540 training pairs; EEGiT's flat
# 5e-5 was validated on a model whose only new modules were two small projection
# heads. Expressing the decoder's LR as a multiple of the semantic one would have
# produced "x10", which hides the decision.
#
# Augmentation `full` on the EEG only; the image targets are not perturbed. MMD off
# (`--stage1-epochs 0`): it is fixed, but the earlier reading that it hurt measured
# its absence, so the judgement has to be re-earned rather than assumed.
# -----------------------------------------------------------------------------
# Checkpoint selection: which "best" for a two-objective model
# -----------------------------------------------------------------------------
# `sel = val_top1 + 0.5 * vae_top1 + 5.0 * depth_pearson`, all three in the same
# 0-100 units. Selection runs on a concept-level holdout of the TRAIN concepts;
# the 200 test concepts are scored once, at the end, by the selected checkpoint.
# Both structural readouts are reported next to their weights in the result JSON
# so it stays visible which of the three terms actually made the choice.
#
# -----------------------------------------------------------------------------
# The four decode arms
# -----------------------------------------------------------------------------
#   deploy_sdedit   SDEdit from the VAE-decoded prediction (structure = the VAE head
#                   only), IP condition from the semantic tower's retrieval over the
#                   training gallery. The arm the brief asks for.
#   deploy_txt2img  the same IP condition, from pure noise, ControlNet at scale 0.
#                   The semantic tower alone.
#   noise_sdedit    the same init as deploy_sdedit, IP condition built from zeros.
#   noise_txt2img   the same as deploy_txt2img, IP condition built from zeros.
#
# The pair `deploy_sdedit` / `deploy_txt2img` is the measurement. They are one code
# path with one difference -- the structural init -- so their CLIP difference is what
# the structural branch is worth, end to end, with the decoder and the IP condition
# held fixed. The previous pipeline had no such arm: every decode consumed the same
# init, so a structural branch that helped, one that did nothing and one that hurt
# all produced the identical comparison, and the branch's value was unmeasurable
# from inside the pipeline that depended on it.
#
# `noise_txt2img` is the clean EEG-null control because it touches no EEG-derived
# input at all. The older control (`noise_sdedit`) still consumed the structural
# init, which is EEG-derived, so it shared an input with the arm it controlled for.
# With the VAE head known to be near-constant the leak was small; it would not have
# stayed small once the head started working, which is the point of the pair above.
#
# Only the two `deploy_*` arms are headline results; `noise_sdedit` exists so the
# headline is interpretable and runs last so a wall-clock overrun still leaves the
# requested result on disk.
#
# `export_conds.py` also implements a fourth arm (`raw`: the EEG embedding read
# back through pinv(img_head), no gallery). It is deliberately NOT generated here,
# because it is not well posed for this configuration: this run aligns the semantic
# tower to block26, which is 1280-d, while IP-Adapter consumes 1024-d joint
# embeddings. `pinv(img_head)` returns a vector in block26's space, and there is no
# honest map from an intermediate residual stream to `visual.proj`'s output. The
# script raises a SystemExit if asked for it, rather than emitting a wrong-width
# array that would silently be interpreted as an image embedding.
# It is available for a run whose alignment target is the final projected layer.
#
# -----------------------------------------------------------------------------
# Cost
# -----------------------------------------------------------------------------
# Estimated on an H100-class 80 GB GPU, batch 128:
#   [0] structural target caches   16540 VAE encodes + 16540 depth inference  ~25 min
#   [1] train                      100 epochs x 117 steps                   ~2.5 h
#   [2] export                     200 forwards + 200 VAE decodes            ~10 min
#   [3] generate                   3 arms x 200 images x 28 steps            ~35 min
#   [4] seven metrics              3 arms x (2-way + FID + SwAV)             ~20 min
# so ~4 h of GPU; the sbatch asks for 6 h.
#
# Why batch 128 and not 256
# -------------------------
# The training loop runs in fp32 with no autocast and no gradient checkpointing,
# so the structure trunk stores activations for all 24 of its blocks -- freezing
# blocks 1..20 does not help, because the path from the trainable patch_embed
# requires grad at every block and autograd therefore saves every one of them. At
# batch 256 that is ~55 GB of activations alone, which fits an 80 GB card and
# OOMs a 40 GB one, and the normal partition is not homogeneous enough to bet on
# which one the scheduler hands out. Halving the batch halves the memory and
# doubles the step count for the same total time, so nothing is given up -- and
# more steps at the same cost is the direction the earlier 800-step deficit
# argues for anyway.
#
# Why the smoke uses the real batch size
# --------------------------------------
# The smoke's job is to catch exactly this class of failure, so it runs the real
# `--batch-size`, the real trunks and the real losses, and shortens the run only
# with --limit-samples. A smoke at a smaller batch would exercise a memory
# profile the real run never has.
#
# Usage:
#   bash scripts/run_patch_dual.sh --dry-run   # validate flags, touch nothing
#   bash scripts/run_patch_dual.sh             # the real thing
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
SUBJ="${SUBJ:-8}"

TAG="${TAG:-patch_dual}"
OUT="${OUT:-${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")}"
# Structural target caches live at a single shared path, NOT under the subject's
# output dir. The targets are VAE encodings of the stimulus images; they do not
# depend on which subject's EEG is being decoded. Deriving this from ${OUT} -- which
# is what the previous version did -- pointed sub-10 at an empty directory and would
# have re-encoded the same 16540 images for no gain. Anything that IS subject-specific
# (trained weights, conditions, generations, metrics) stays under ${OUT}.
TGT="${TGT:-${ROOT}/outputs/struct_targets}"
EXP="${EXP:-${OUT}/${TAG}_export}"         # conditions handed to SDXL
GEN="${GEN:-${OUT}/${TAG}_gen}"            # generations, one dir per arm
MET="${MET:-${OUT}/${TAG}_metrics}"        # seven-metric JSONs, one per arm

FEAT="${FEAT:-${ROOT}/outputs/features/clip_h14_layers}"
VAE_CACHE="${VAE_CACHE:-${TGT}/vae_cache}"

SKIP_TARGETS="${SKIP_TARGETS:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_METRICS="${SKIP_METRICS:-0}"

# ---------------------------------------------------------------- caches
# /home is at 100% and a download there fails with ENOSPC mid-run, so every cache
# is pinned to /project before torch or HF is imported.
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ---------------------------------------------------------------- helpers
log()  { echo "[$(date +%H:%M:%S)] $*"; }
die()  { echo "[FATAL] $*" >&2; exit 1; }
require() { for f in "$@"; do [[ -e "$f" ]] || die "missing required artefact: $f"; done; }

# ---------------------------------------------------------------- vae cache check
# Validate the shared VAE latent cache by CONTENT. Existence is not enough, and
# neither is finiteness, because of how the file is actually produced:
#
#   `NeuroBridge/scripts/nda/build_gt_vae_latents.py` allocates the output with
#   `np.lib.format.open_memmap(..., mode="w+")`, which creates the file at its FULL
#   size (16540 x 4 x 64 x 64 x f16 = 517 MB) and then fills it in batches. Its own
#   resume check is `np.isfinite(arr).all()` on a small sample -- and ZERO IS FINITE.
#
# So a reader that arrives while a writer is mid-fill sees a correct-looking file at
# the correct path and the correct shape, passes the finite check, and trains against
# a target that is mostly zeros. That is not hypothetical: it happened on the first
# array submission, where task 1 read the cache task 0 was still writing and trained
# for two minutes against a target that was 58.65% zeros (rows 8268..16539 entirely
# zero, overall std 0.5468 against a true 0.85). The tell was in its own log, one
# debug line apart: the same cache reported per-channel stds of 0.315/0.246/0.269/0.214
# at the smoke and 0.527/0.411/0.447/0.356 at the training start, because the writer
# was still adding non-zero rows in between.
#
# The checks below are aimed at that specific failure: sample rows ACROSS the whole
# array including the last one (a truncated write is zero in the tail), reject a
# non-trivial zero fraction, and require a plausible spread. A legitimately all-zero
# patch is a real thing inside an individual latent, which is why this is a fraction
# over many sampled rows rather than a check on any single one.
vae_cache_ok() {
  "${PY}" - "$1" <<'PYEOF'
import sys
import numpy as np

path = sys.argv[1]
a = np.load(path, mmap_mode="r")
if a.ndim != 4 or a.shape[1:] != (4, 64, 64):
    sys.exit(f"shape {a.shape}, expected (N, 4, 64, 64)")
n = a.shape[0]
if n < 100:
    sys.exit(f"only {n} rows")
# 96 rows spread over the full extent, always including the final row.
idx = np.unique(np.concatenate([np.linspace(0, n - 1, 96).astype(int), [n - 1]]))
blk = np.asarray(a[idx]).astype(np.float32)
if not np.isfinite(blk).all():
    sys.exit("contains non-finite values")
zero_frac = float((blk == 0).mean())
if zero_frac > 0.05:
    sys.exit(f"zero_frac {zero_frac:.4f} over {len(idx)} sampled rows -- the file looks "
             f"partially written (a mid-fill memmap reads as zeros; isfinite() cannot "
             f"detect it)")
sd = float(blk.std())
if not (0.30 < sd < 2.00):
    sys.exit(f"std {sd:.4f} outside [0.30, 2.00] for SDXL latents (a zero-filled or "
             f"wrongly scaled cache lands here)")
print(f"ok shape {a.shape} dtype {a.dtype} std {sd:.4f} zero_frac {zero_frac:.4f}")
PYEOF
}

# =============================================================================
# The single training configuration. Every line is a decision, documented above.
#
# SEM selects the semantic tower's recipe. It exists because the semantic side was
# supposed to BE EEGiT and was not: reading the released code turned up four
# discrepancies (patch-image layout, `pos_embed` antialias, loss/temperature,
# optimizer), and `scripts/run_eegit_gate.sh` measured what each is worth on
# retrieval alone, in three arms, on the identical split:
#
#   arm (EEGiT-consistency)                    test@1   val@1   eps@peak   fit@1
#   nw8's own config          (nothing)         46.50   34.40      19       91.5
#   + official patch image    (layout)          43.50   32.40      19       89.6
#   + official loss/optimizer (fuse)            39.50   31.13      28       99.3
#   + official head, 1 layer  (full)            35.50   31.20      20       91.3
#
# A Cochran-Armitage trend test over that ordering of "how much of the released
# recipe is adopted" gives z=-2.38, p=0.018 on test: adopting more of it makes
# retrieval WORSE, monotonically. The val split (150 concepts disjoint from test)
# orders the same way at both ends, though its trend is not significant on its own.
# The `full` arm differs from official EEGiT in exactly two ways -- the alignment
# target and a frozen image encoder -- so ~35 points of their 70.4 is attributable
# to aligning against a TARGET THAT IS TRAINED, which is not a space this pipeline
# can use: IP-Adapter consumes 1024-d CLIP joint embeddings.
#
# So `eegit` and `fuse` are kept only as recorded, runnable recipes. They are NOT
# the recommended configuration, and `prev` -- the best measured one -- is the
# default here.
#   prev   nw8's semantic config verbatim: `nw` patch layout, AdamW, 5e-4 for the
#          new parts and 5e-5 for the blocks, 5-epoch warmup then cosine, 3-layer
#          uniform fusion over blocks {8,10,12}, `nw` MLP head
#   eegit  full EEGiT consistency: official layout + official head + single final
#          layer + official loss and optimizer            (gate arm `full`)
#   fuse   that strategy, our 3-layer uniform fusion + nw head  (gate arm `fuse`)
# The structural side is synced with whichever is chosen where the setting is
# genuinely shared (patch layout, optimizer family, schedule shape) and keeps its
# own LRs where it is not (a ~40M randomly-initialised decoder is not what EEGiT's
# flat 5e-5 was validated on).
# =============================================================================
SEM="${SEM:-prev}"

# One tag per semantic recipe, and it is derived HERE -- before anything expands
# `${TAG}` -- rather than next to `train_cfg` where it used to sit.
#
# Where it used to sit is the whole reason this run had to be relaunched. The block
# was after `sem_cfg` was built, and `sem_cfg` contains `--tag "${TAG}"`; bash expands
# an array element at assignment time, so `sem_cfg` captured the pre-suffix
# `patch_dual` while `EXP/GEN/MET` were re-derived afterwards to `patch_dual_prev_*`.
# Training therefore wrote `patch_dual_best.pt` and the pipeline then demanded
# `patch_dual_prev_best.pt` and died at the `require` before the export. The smoke run
# escaped it because it passes `--tag "_smoke_${TAG}"` on the command line, *after*
# `train_cfg`, where it wins -- so the bug could only ever surface at the real export,
# two hours into a job. Deriving the tag first removes the class, not the instance.
#
# Without the suffix at all, a second SEM run under the same default tag would find
# `patch_dual_best.pt` from the first and skip training silently -- the stage-skipping
# design working exactly as documented, and exactly wrongly here.
if [ "${SEM}" != "eegit" ]; then
  TAG="${TAG}_${SEM}"
  OUT="${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")"
  # TGT is deliberately NOT re-derived here: it is the shared, subject-independent
  # target root set above, and this block only re-keys the SUBJECT-specific paths.
  EXP="${OUT}/${TAG}_export"
  GEN="${OUT}/${TAG}_gen"
  MET="${OUT}/${TAG}_metrics"
fi

# Shared by every arm EXCEPT that `--patch-style` is now per-arm: it is the one
# setting the gate actually measured, and it measured it as a loss for the semantic
# tower. `eegit_official` is the released code's `use_kinematic` +
# `spatial_interpolate` exactly -- H=time, W=regions, anterior->posterior, the
# dataset's own channel order, one 2D bilinear interpolation per region.
# `--patch-style` reaches BOTH towers (it is one input interface), so choosing it
# for the semantic side also sets the structure tower's EEG image. That coupling is
# deliberate but UNMEASURED for the structure tower: there is no arm that varies the
# layout for the structural target alone, so `prev` inherits `nw` on both sides and
# `struct`-side layout is not evidence-based either way.
# `vit_b16_in21k_orig` is the tag the official code names; it resolves to the same
# weights as `vit_b16_in21k`, and naming it means the record says what it used.
# `nw` is the layout nw8 used and the gate measured as better; `eegit_official` is
# the released code's. Set per arm below, before `sem_common` reads it.
SEM_PATCH_STYLE="${SEM_PATCH_STYLE:-}"
if [ -z "${SEM_PATCH_STYLE}" ]; then
  case "${SEM}" in
    prev)        SEM_PATCH_STYLE="nw" ;;
    eegit|fuse)  SEM_PATCH_STYLE="eegit_official" ;;
  esac
fi

sem_common=(
  --subject "${SUBJ}"
  --tag "${TAG}"
  --out-dir "${OUT}"
  --tokenizer eegit
  --patch-style "${SEM_PATCH_STYLE:-eegit_official}"
  --patch-size 16
  --n-patches-w 14
  --channels all                 # 5 EEGiT regions -> 5x14 = 70 tokens
  --backbone timm:vit_b16_in21k_orig
  --target-features "${FEAT}"
  --target-layer block26
  --target-fusion single
)

case "${SEM}" in
  eegit)
    # EEGiT's encoder: read the final block, pool with `global_pool='avg'`, feed one
    # vector to the official ProjectionHead. No fusion module may sit between the
    # pretrained encoder and the head, so `--fusion-mode none` is required, not
    # stylistic. `--d-embed 1024` is the paper's 768 -> 1024 FC.
    sem_cfg=("${sem_common[@]}"
      --layers 12
      --fusion-mode none
      --head-kind eegit
      --img-head-kind eegit
      --head-drop 0.5
      --d-embed 1024
      --pool mean
    )
    sem_sched=(
      # official `PLModel.configure_optimizers`: torch.optim.Adam, ONE weight_decay
      # for the whole group, EEG encoder and heads both at lr*10 = 5e-5, flat, no
      # warmup and no decay.
      --optimizer adam
      --wd-all-params
      --lr 5e-5
      --backbone-lr-mult 1.0
      --fixed-temp                     # official's logit_scale is never optimised
      --softplus                       # scale softplus(log(1/0.07)) = 2.727
      --no-eeg-l2norm                  # official normalises only the image side
    )
    ;;
  fuse)
    # Same strategy and interface; our 3-layer uniform fusion and the nw MLP head.
    sem_cfg=("${sem_common[@]}"
      --layers 8 10 12
      --fusion-mode uniform
      --pool mean
    )
    sem_sched=(
      --optimizer adam
      --wd-all-params
      --lr 5e-5
      --backbone-lr-mult 1.0
      --fixed-temp
      --softplus
      --no-eeg-l2norm
    )
    ;;
  prev)
    # patch_dual's architecture and schedule, on the REPAIRED interface. Not exactly
    # what patch_dual ran -- the tokenizer layout and the `pos_embed` antialias are
    # shared by every arm now -- which is the point: it isolates the semantic
    # side's loss/optimizer from the interface repair, using the same architecture
    # the deployed dual tower used.
    sem_cfg=("${sem_common[@]}"
      --layers 8 10 12
      --fusion-mode uniform
    )
    sem_sched=(
      --lr 5e-4
      --backbone-lr-mult 0.1
      --warmup-epochs 5
      --cosine
      --min-lr-ratio 0.01
    )
    ;;
  *) die "SEM must be eegit|fuse|prev, got '${SEM}'" ;;
esac

train_cfg=("${sem_cfg[@]}" "${sem_sched[@]}"

  # ---- structure tower: separate EEG Patch tower, structure-preserving trunk,
  # ---- capacity braked by freezing all but the four deepest blocks
  --struct-backbone timm:dinov2_l_reg4
  --struct-patch-size 14         # must equal DINOv2's patch_embed kernel; checked
  --struct-n-patches-w 16        # 16x14 = 224 along time, as in the semantic tower
  --struct-layers 16 20 24
  --struct-fusion-mode uniform
  --struct-freeze-blocks 20
  # Absolute, not multipliers of `--lr`: the semantic base LR is now EEGiT's 5e-5,
  # and the decoder is ~40M randomly-initialised parameters that the previous dual
  # run trained at 5e-4. Writing 10x for the multiplier form would hide that; these
  # two values are the SAME absolute LRs as before, so the structural side's
  # behaviour stays comparable to patch_dual instead of being silently re-tuned.
  --struct-lr 5e-5
  --struct-head-lr 5e-4
  --struct-drop 0.1
  --struct-base-ch 128
  --struct-field-ch 32

  # ---- structural target: VAE latents ONLY, no depth head ---------------------
  # The depth head is removed rather than retuned, on a measurement rather than a
  # judgement. The stage-0 target probe (outputs/sub*/probe_targets_*.json) fitted a
  # closed-form ridge from the raw EEG to every candidate space on all 10 slots and
  # scored each against the constant predictor that emits the training-set mean map
  # -- the exact thing L1 falls back to when the target is not in the input:
  #
  #   target   subject/channels    r(pred,gt)   r(const,gt)     margin   var ratio
  #   depth    sub-08 / 63ch          +0.1598      +0.5333    -0.3735      0.2233
  #   depth    sub-08 / 17ch          +0.1617      +0.5333    -0.3716      0.3211
  #   depth    sub-10 / 63ch          +0.1577      +0.5333    -0.3756      0.2218
  #   depth    sub-10 / 17ch          +0.1695      +0.5333    -0.3638      0.1924
  #
  # A constant map correlates +0.5333 with a real depth map while the EEG's best
  # LINEAR read-out reaches +0.16. The margin is negative on four independent arms,
  # so the linear ceiling for depth is the mean: no head design recovers this, and
  # the previously shipped depth head (r = +0.6485 against the constant's +0.6540)
  # was already at that ceiling. See `--struct-sel-w` below for what replaced it.
  #
  # The VAE latent is the one structural space that clears its floor:
  #
  #   target   subject/channels    val@1   test@1     margin   var ratio
  #   vae      sub-08 / 63ch         5.07     6.00    +0.0617      0.4411
  #   vae      sub-08 / 17ch         5.73     7.00    +0.1194      0.3178
  #   vae      sub-10 / 63ch         4.40     7.00    +0.0566      0.4124
  #   vae      sub-10 / 17ch         5.27     8.00    +0.0812      0.4575
  #
  # Clearing the floor is necessary and not sufficient, and the honest reading of
  # that table is uncomfortable for the previous run: a closed-form linear map from
  # the raw EEG reaches 5.07% instance Top-1 (chance 0.50%, mean rank 24.8 of 200),
  # while the trained VAE head reached 1.00% (mean rank 85) with a variance ratio of
  # 0.0068. The trained head is FIVE TIMES WORSE than a linear baseline on the same
  # test, so the VAE branch's problem is the training, not the target's existence.
  # That is why this run keeps the head and adds the floor check below instead of
  # declaring the branch dead.
  #
  # Both keys are concept*10+slot, and the caches are SHARED across subjects: a VAE
  # latent is a property of the stimulus, not of the recording, so a per-subject copy
  # would re-encode the same 16540 images for nothing.
  --vae-latents "${VAE_CACHE}"
  --w-vae 1.0

  # ---- checkpoint selection ---------------------------------------------------
  # What "best" means now has one structural term instead of two. `struct_sel_w`
  # is deliberately unchanged at 0.5: the previous run's 0.5*val_vae_top1 was not
  # what let the VAE head collapse (the VAE head was already at chance under it),
  # so re-weighting it would tune a term that has not been shown to be mis-set,
  # and would break comparability with the runs it was set for.
  --struct-sel-w 0.5

  # ---- shared schedule
  --epochs 100
  --batch-size 128               # memory, not a tuning choice: see the cost note
  --patience 0                   # 0 = disabled, so the schedule completes
  --aug full
  --stage1-epochs 0
  --seed 2025
  --fit-diagnostic
)

# =============================================================================
if [[ "${1:-}" == "--dry-run" ]]; then
  # Flag-level validation without a GPU, without the dataset, and without the
  # structural caches: this is the check that a bad combination fails here rather
  # than after an allocation has been granted.
  log "validating the exact flag list (no GPU, no dataset)"
  "${PY}" -u "${ROOT}/scripts/nwret/train.py" "${train_cfg[@]}" --validate-only || exit 1
  for f in "${ROOT}/scripts/nwret/train.py" \
           "${ROOT}/scripts/nwret/export_conds.py" \
           "${NB_ROOT}/scripts/nda/generate_struct_inject_decode.py" \
           "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
           "${NB_ROOT}/scripts/nda/build_gt_vae_latents.py" \
           "${NB_ROOT}/scripts/nda/build_gt_depth_cache.py"; do
    [[ -f "$f" ]] || die "pipeline references a script that does not exist: $f"
  done
  [[ -d "${FEAT}/train" ]] || die "no semantic target features at ${FEAT}/train"
  log "dry run ok"
  exit 0
fi

mkdir -p "${TGT}" "${EXP}" "${GEN}" "${MET}" "${ROOT}/outputs/logs"
cd "${ROOT}"

log "=============================================================="
log "EEG Patch dual tower, subject ${SUBJ}, tag ${TAG}"
log "  semantic target : ${FEAT} block26 -> CLIP ViT-H-14 joint space"
log "  structure target: ${VAE_CACHE} (VAE latents only; the depth head is removed)"
log "  outputs         : ${EXP} -> ${GEN} -> ${MET}"
log "=============================================================="

# -----------------------------------------------------------------------------
# [0] Structural target caches
# -----------------------------------------------------------------------------
# Both builders walk `sorted(concept_dirs)` then `sorted(images_in_dir)`, which is
# the row order `load_subject()` produces: row = concept*10 + slot. That
# correspondence is what lets a flat memmap stand in for a keyed lookup, and it is
# asserted at dataset construction rather than trusted.
#
# The train halves are the expensive ones (16540 images). Whether the ordering is
# actually right is not asserted anywhere -- it cannot be, from the files alone --
# but it is *observable* in the training run: if the caches were misaligned, the
# depth head's val Pearson r would sit at 0 and vae_top1 at chance, because the
# model would be asked to predict one image's layout from another image's EEG.
if [[ "${SKIP_TARGETS}" != "1" ]]; then
  log "[0] structural target caches @ $(date -Iseconds)"
  VAE_TRAIN="${VAE_CACHE}/train_vae_latents_f16.npy"
  VAE_TEST="${VAE_CACHE}/test_vae_latents_f16.npy"
  mkdir -p "${TGT}"

  # Fast path: a cache that is already there AND content-valid. Validating here rather
  # than trusting the filename is what rejects a cache left half-written by a killed
  # job -- exactly the artefact the cancel above produced (full size, correct shape,
  # 58.65% zeros).
  if vae_cache_ok "${VAE_TRAIN}" >/dev/null 2>&1 && vae_cache_ok "${VAE_TEST}" >/dev/null 2>&1; then
    log "[0] vae latents present and content-valid, skipping (delete them to rebuild)"
  else
    # The cache is shared across subjects, so the two array tasks reach this point at
    # the same time. `flock` serialises them; without it both would encode the same
    # 16540 images and, worse, publish into the same paths concurrently.
    exec {VAE_LOCK}>"${TGT}/.vae_cache.lock"
    log "[0] waiting on the vae cache lock (held by another array task, if any)"
    flock "${VAE_LOCK}"
    # Re-check under the lock: the other task may have finished the build while this
    # one waited, in which case there is nothing left to do.
    if vae_cache_ok "${VAE_TRAIN}" >/dev/null 2>&1 && vae_cache_ok "${VAE_TEST}" >/dev/null 2>&1; then
      log "[0] the build finished while waiting for the lock; reusing it"
    else
      # Build into a staging dir and publish with `mv`. Same filesystem, so the rename
      # is atomic: a concurrent reader sees either the previous complete file or the
      # new one, never a partial one. This is the actual fix for the race -- the lock
      # alone would not protect a reader that does not take it (e.g. an interactive
      # export), whereas an atomic publish does, by construction.
      STAGE="${TGT}/.vae_cache.staging.$$"
      rm -rf "${STAGE}"
      mkdir -p "${STAGE}"
      log "[0] building vae latents into staging: ${STAGE}"
      "${PY}" "${NB_ROOT}/scripts/nda/build_gt_vae_latents.py" \
        --images-root "${IMAGES_ROOT}" \
        --output-dir "${STAGE}" \
        --splits train,test \
        --batch-size 8 \
        --device cuda:0 || die "vae latent build failed"

      # Validate BEFORE publishing. An invalid build must never reach the shared path,
      # because the next run's fast path would accept it.
      vae_cache_ok "${STAGE}/train_vae_latents_f16.npy" \
        || die "the just-built train cache failed content validation in ${STAGE}"
      vae_cache_ok "${STAGE}/test_vae_latents_f16.npy" \
        || die "the just-built test cache failed content validation in ${STAGE}"

      mkdir -p "${VAE_CACHE}"
      mv -f "${STAGE}/train_vae_latents_f16.npy" "${VAE_TRAIN}" || die "publish train failed"
      mv -f "${STAGE}/test_vae_latents_f16.npy"  "${VAE_TEST}"  || die "publish test failed"
      mv -f "${STAGE}/vae_latent_report.json"    "${VAE_CACHE}/" 2>/dev/null || true
      rm -rf "${STAGE}"
      log "[0] published vae latents atomically to ${VAE_CACHE}"
    fi
    exec {VAE_LOCK}>&-
  fi
  require "${VAE_TRAIN}" "${VAE_TEST}"

  # No depth cache is built. The head that consumed it is gone (see the measurement
  # in `train_cfg`), and `build_gt_depth_cache.py` costs ~13 min of GPU on 16540
  # images to produce a target whose linear ceiling is the mean.
  #
  # `gt_depth/test_depth_64.npy` is still read by `export_conds.py`, but only as the
  # REFERENCE the collapse gate compares against -- never as a condition fed to the
  # generator. If it is absent the gate reports itself skipped rather than failing,
  # so this branch is an optimisation, not a requirement:
  if [[ -f "${TGT}/gt_depth/test_depth_64.npy" ]]; then
    log "[0] depth caches present, kept for the collapse gate's reference only"
  fi
else
  log "[0] skipped by SKIP_TARGETS"
fi

# -----------------------------------------------------------------------------
# [1] Train
# -----------------------------------------------------------------------------
# The smoke gate runs the REAL flag list through the same code path for 3 epochs
# on 1024 samples: the two tokenizers, the two z-score statistics, both trunks,
# both losses, the three LR groups, the dual selection, the export provenance.
# Without it a flag that only breaks on the second epoch (or an OOM at the real
# batch size) surfaces an hour later.
if [[ "${SKIP_TRAIN}" != "1" ]]; then
  log "[1] unit tests @ $(date -Iseconds)"
  # Run once and keep the output: the gate needs the exit code and the reader
  # needs the tail, and running the suite twice costs a minute of GPU-node CPU
  # for nothing.
  UT_LOG="${ROOT}/outputs/logs/${TAG}_unit_tests.log"
  if ! "${PY}" -u "${ROOT}/scripts/test_dual_tower.py" >"${UT_LOG}" 2>&1; then
    tail -40 "${UT_LOG}"
    die "scripts/test_dual_tower.py failed (full log: ${UT_LOG})"
  fi
  tail -6 "${UT_LOG}"

  log "[1] smoke: the real config, 3 epochs, 1024 fit samples (real batch size)"
  "${PY}" -u "${ROOT}/scripts/nwret/train.py" "${train_cfg[@]}" \
    --tag "_smoke_${TAG}" --epochs 3 --warmup-epochs 1 \
    --limit-samples 1024 --fit-diag-concepts 20 2>&1 | tail -30
  SMOKE_RC=${PIPESTATUS[0]}
  if [[ "${SMOKE_RC}" -ne 0 ]]; then
    die "smoke run failed (rc=${SMOKE_RC}); not starting the real run"
  fi
  require "${OUT}/_smoke_${TAG}_best.pt"

  # ---- export smoke ---------------------------------------------------------
  # The export runs hours after training, in a separate process, against a
  # checkpoint it cannot inspect interactively -- exactly the place a drift
  # between the training constructor and the export constructor is discovered
  # too late to fix. So the same code path is exercised here on the smoke
  # checkpoint, on 8 test concepts, including the VAE decode and the depth
  # rendering, into a directory that is thrown away.
  log "[1] smoke: the export path on the smoke checkpoint, 8 concepts"
  # `--allow-collapse` is required here and only here. A 3-epoch smoke is a collapsed
  # model by construction, so the across-sample gate would abort it -- and the smoke
  # exists to catch a wiring error between the training and export constructors, not
  # to check convergence. The real export below is NOT given this flag: that is where
  # the gate has to hold, and it is the only place it can, because it is the last
  # point with the ground truth still on disk.
  "${PY}" -u "${ROOT}/scripts/nwret/export_conds.py" \
    --ckpt "${OUT}/_smoke_${TAG}_best.pt" \
    --out-dir "${OUT}/_smoke_${TAG}_export" \
    --tag "_smoke_${TAG}" \
    --arms deploy noise \
    --decode-rgb \
    --limit 8 \
    --allow-collapse \
    --device cuda:0
  EXPORT_RC=$?
  if [[ "${EXPORT_RC}" -ne 0 ]]; then
    die "export smoke failed (rc=${EXPORT_RC}): the checkpoint loads but its \
conditions do not build, so the real run would train for 2.5 h and then fail at \
stage 2"
  fi
  rm -rf "${OUT}/_smoke_${TAG}_export"

  # The smoke writes real artefacts under a reserved tag; remove them so they can
  # never be picked up by summarize.py or mistaken for the experiment.
  rm -f "${OUT}/_smoke_${TAG}_result.json" "${OUT}/_smoke_${TAG}_best.pt"
  log "[1] smoke passed (train + export)"

  if [[ -f "${OUT}/${TAG}_best.pt" ]]; then
    log "[1] ${TAG}_best.pt exists, skipping training (delete it to retrain)"
  else
    log "[1] training @ $(date -Iseconds)"
    "${PY}" -u "${ROOT}/scripts/nwret/train.py" "${train_cfg[@]}"
    RC=$?
    [[ "${RC}" -eq 0 ]] || die "training failed (rc=${RC})"
  fi
else
  log "[1] skipped by SKIP_TRAIN"
fi
require "${OUT}/${TAG}_best.pt" "${OUT}/${TAG}_result.json"

# -----------------------------------------------------------------------------
# [2] Export the three conditions
# -----------------------------------------------------------------------------
# deploy: soft retrieval over the 1654 TRAIN concept CLIP embeddings, performed in
#         the model's own 512-d space (where the metric is meaningful) and
#         realised as a weighted sum of the gallery's real CLIP embeddings (where
#         IP-Adapter can consume it). The retrieval cannot return the answer: the
#         gallery is disjoint from the 200 test concepts.
# noise:  EEG replaced by zeros.
# Plus, from the structure tower: pred_vae_test_scaled.npy, the decoded RGB init,
# and the depth map rendered to RGB.
if [[ "${SKIP_EXPORT}" != "1" ]]; then
  log "[2] export conditions @ $(date -Iseconds)"
  "${PY}" -u "${ROOT}/scripts/nwret/export_conds.py" \
    --ckpt "${OUT}/${TAG}_best.pt" \
    --out-dir "${EXP}" \
    --tag "${TAG}" \
    --arms deploy noise \
    --decode-rgb \
    --device cuda:0 || die "condition export failed"
else
  log "[2] skipped by SKIP_EXPORT"
fi
require "${EXP}/conds/ip_deploy_test.npy" "${EXP}/conds/ip_noise_test.npy" \
        "${EXP}/spatial/pred_lowlevel_rgb_512/000.png"

# -----------------------------------------------------------------------------
# [3] Generate
# -----------------------------------------------------------------------------
# SDXL-base + IP-Adapter (ViT-H) + ControlNet-depth + SDEdit, obtained from
# NeuroBridge's `generate_struct_inject_decode.py` rather than reimplemented: it
# already encodes the house conventions this project's other decoders were tuned
# against (the doubled IP embedding under guidance > 1, the fp16 variants, the
# offline/local_files_only resolution of the SDXL and IP-Adapter snapshots).
#
# No text prompt on any arm. The semantic condition is the IP embedding alone,
# which keeps the comparison to the noise arm exact and avoids reintroducing a
# text path that would have to be built from EEG to be legitimate.
#
# `--control-guidance-end 0.5` confines the ControlNet to the layout phase of
# denoising, when one is used at all; letting it act through the last steps is what
# makes a ControlNet overwrite the semantics the IP condition is carrying.
gen_arm() {
  local arm="$1" mode="$2" cn="$3" iparm="$4" sval="$5"
  local gdir="${GEN}/${arm}"
  if [[ -f "${gdir}/generated/199.png" ]]; then
    log "[3] ${arm}: 200 images present, skipping"
    return 0
  fi
  log "[3] ${arm}: mode=${mode} cn=${cn} ip=${iparm} strength=${sval}"
  # `--cond-dir` is loaded unconditionally by the generator in BOTH modes, even when
  # the ControlNet's scale is 0, so it has to point at a directory with 200 readable
  # PNGs. Both modes reuse the VAE-decoded init dir for that: at cn-scale 0 the map
  # is multiplied out before it reaches the residual, and no new artifact is created
  # for a path that contributes nothing. With the depth head removed there is no
  # depth map to point at anyway.
  local init_args=""
  if [[ "${mode}" == "img2img" ]]; then
    init_args="--init-dir ${EXP}/spatial/pred_lowlevel_rgb_512 --strength ${sval}"
  fi
  # shellcheck disable=SC2086
  "${PY}" -u "${NB_ROOT}/scripts/nda/generate_struct_inject_decode.py" \
    --mode "${mode}" \
    --embed-npy "${EXP}/conds/ip_${iparm}_test.npy" \
    --cond-dir "${EXP}/spatial/pred_lowlevel_rgb_512" \
    ${init_args} \
    --output-dir "${gdir}" \
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

# -----------------------------------------------------------------------------
# The four arms, and why the PAIRING is the experiment
# -----------------------------------------------------------------------------
#   deploy_sdedit   VAE-decoded init (structural tower) + IP(deploy), img2img 0.80
#                   Both towers in the path -- the arm the brief asks for.
#   deploy_txt2img  NO init, from pure noise + IP(deploy), ControlNet at scale 0
#                   The semantic tower alone.
#   noise_sdedit    the SAME init as deploy_sdedit, IP(noise)
#   noise_txt2img   the same as deploy_txt2img, IP(noise)
#
# `deploy_sdedit` and `deploy_txt2img` differ in exactly one thing: the structural
# init. That is what prices the structural branch, and no arm in this pipeline could
# answer it before, because every arm consumed the same init -- so a structural
# tower that helped, a structural tower that did nothing, and a structural tower
# that actively hurt all produced the same comparison.
#
# `noise_txt2img` is the one arm that touches no EEG-derived input at all, so it is
# what the "is any of this from the EEG" delta is read against. The previous
# pipeline's control was `noise_sdedit`, which still consumed the structural init --
# and the structural init is EEG-derived, so the control shared a pathway with the
# arm it was meant to control for. The leak was small once the head was known to be
# near-constant, but a control that shares an input with its treatment is not a
# control, and it stops being small the moment the head starts working.
ARMS=""
if [[ "${SKIP_GEN}" != "1" ]]; then
  log "[3] generate @ $(date -Iseconds)"
  # The pair the conclusion rests on goes first, so a wall-clock overrun still
  # leaves the comparison on disk.
  gen_arm "deploy_sdedit"  img2img 0.0 deploy 0.80
  gen_arm "deploy_txt2img" txt2img 0.0 deploy -
  # Controls last: they can be dropped without removing the headline pair.
  gen_arm "noise_sdedit"   img2img 0.0 noise  0.80
  gen_arm "noise_txt2img"  txt2img 0.0 noise  -
  ARMS="deploy_sdedit deploy_txt2img noise_sdedit noise_txt2img"
else
  log "[3] skipped by SKIP_GEN"
  ARMS="$(cd "${GEN}" 2>/dev/null && ls -d */ 2>/dev/null | tr -d '/' | tr '\n' ' ')"
fi

# -----------------------------------------------------------------------------
# [4] The seven standard metrics
# -----------------------------------------------------------------------------
# ATM / MindEye / CogCap protocol: PixCorr and SSIM on grey @425 with a gaussian
# (NOT the RGB@256 variant, which is not comparable to the published tables),
# AlexNet(2)/AlexNet(5)/Inception/CLIP as two-way identification, SwAV as mean
# correlation distance. FID is computed too but is not part of the seven.
if [[ "${SKIP_METRICS}" != "1" ]]; then
  log "[4] seven metrics @ $(date -Iseconds)"
  for arm in ${ARMS}; do
    gdir="${GEN}/${arm}/generated"
    [[ -f "${gdir}/199.png" ]] || { log "[4] ${arm}: no generations, skipped"; continue; }
    outj="${MET}/${arm}_seven.json"
    [[ -f "${outj}" ]] && { log "[4] ${arm}: metrics present, skipping"; continue; }
    "${PY}" -u "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
      --gen-dir "${gdir}" \
      --output-json "${outj}" \
      --tag "${arm}" \
      --images-root "${IMAGES_ROOT}" \
      --device cuda:0 \
      --batch-size 16 || die "seven-metric evaluation failed for arm ${arm}"
  done
  # The per-arm feature caches are large and fully regenerable from the images.
  find "${GEN}" -type d -name '_twoway_cache' -exec rm -rf {} + 2>/dev/null || true
else
  log "[4] skipped by SKIP_METRICS"
fi

# -----------------------------------------------------------------------------
# [5] Table
# -----------------------------------------------------------------------------
log "[5] summary @ $(date -Iseconds)"
TAG="${TAG}" SUBJ="${SUBJ}" OUT="${OUT}" MET="${MET}" GEN="${GEN}" EXP="${EXP}" PY="${PY}" "${PY}" - <<'PY'
import json, os
from pathlib import Path

met = Path(os.environ["MET"])
rows = []
for p in sorted(met.glob("*_seven.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    rows.append(r)
if not rows:
    raise SystemExit("[5] no seven-metric JSONs found; nothing to summarise")

# Order: the headline pair first and ADJACENT, then the controls. The pair is printed
# next to each other because the structural branch is priced by their difference, and
# a table that separates them makes the reader reconstruct the subtraction.
HEADLINE = ("deploy_sdedit", "deploy_txt2img")
order = {"deploy_sdedit": 0, "deploy_txt2img": 1,
         "noise_txt2img": 8, "noise_sdedit": 9}
rows.sort(key=lambda r: order.get(r["tag"], 6))

KEYS = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]

# The published references, copied from eval_standard_seven_table.py so the table
# is readable on its own. Only the sub-08 rows are directly comparable.
sota = [
    ("ATM (NeurIPS'24), sub-08",
     dict(pixcorr=0.160, ssim=0.345, alex2=0.776, alex5=0.866, inception=0.734, clip=0.786, swav=0.582)),
    ("CogCap-all (AAAI'25), sub-08",
     dict(pixcorr=0.175, ssim=0.366, clip=0.744)),
    ("CogCapPro, THINGS-EEG",
     dict(pixcorr=0.163, ssim=0.398, inception=0.779, clip=0.830, swav=0.553)),
]

def f(v):
    return "—" if v is None else f"{float(v):.3f}"

lines = [
    "# EEG Patch dual tower, sub-08 reconstruction, seven standard metrics",
    "",
    "Protocol: PixCorr↑, SSIM↑ (skimage grey@425 gaussian), AlexNet(2)↑, AlexNet(5)↑,",
    "Inception↑, CLIP↑ (all four as two-way identification), SwAV↓ (mean correlation",
    "distance). FID is reported but is not one of the seven. 200 test concepts, one",
    "generated image each, compared against the 200 ground-truth test images.",
    "",
    "| arm | PixCorr↑ | SSIM↑ | AlexNet(2)↑ | AlexNet(5)↑ | Inception↑ | CLIP↑ | SwAV↓ | FID↓ |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    # Only the deployed configuration is a result; the rest are ablations of it and
    # are left unbolded so the table does not read as a sweep.
    cell = (lambda k: ("**" + f(r[k]) + "**") if r["tag"] in HEADLINE else f(r[k]))
    label = r["tag"]
    if r["tag"] == "deploy_txt2img":
        label += " (semantic only)"
    elif r["tag"].startswith("noise_"):
        label += " (EEG-null control)"
    lines.append("| " + label + " | " + " | ".join(cell(k) for k in KEYS) + " |")
lines.append("")
lines.append("## Published references")
lines.append("")
lines.append("| method | PixCorr↑ | SSIM↑ | AlexNet(2)↑ | AlexNet(5)↑ | Inception↑ | CLIP↑ | SwAV↓ |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for name, d in sota:
    lines.append("| " + name + " | " + " | ".join(f(d.get(k)) for k in KEYS[:7]) + " |")
lines.append("")

sdedit = next((r for r in rows if r["tag"] == "deploy_sdedit"), None)
sem_only = next((r for r in rows if r["tag"] == "deploy_txt2img"), None)
noise = next((r for r in rows if r["tag"] == "noise_txt2img"), None)
noise_sd = next((r for r in rows if r["tag"] == "noise_sdedit"), None)
notes = []
# Whether the retrieval step did anything at all. `deploy` mixes real CLIP embeddings
# with weights from the EEG query; if those weights are near-uniform the mix is just
# the training-set centroid and the arm demonstrates the gallery, not the encoder.
# Only reported when the export recorded it, which every run since the two-space
# split does.
try:
    _ex = json.loads((EXP / "conds" / "export_report.json").read_text())
    _r = _ex.get("retrieval")
except Exception:
    _r = None
if _r:
    e, u = float(_r["mean_entropy_nats"]), float(_r["uniform_entropy_nats"])
    frac = e / u if u else 1.0
    notes.append(
        f"- Retrieval sharpness: mean softmax entropy {e:.3f} of {u:.3f} nats max "
        f"({frac * 100:.1f}% of uniform), mean top-1 weight "
        f"{float(_r['mean_top1_weight']):.3f}. "
        + ("The query is selecting a small subset of the gallery, so the deployed "
           "condition is EEG-specific." if frac < 0.95 else
           "The weights are essentially uniform, so the deployed condition is the "
           "training-set centroid and this arm tests the gallery rather than the "
           "encoder. Raise --gallery-temp or lower --gallery-topk before drawing "
           "any conclusion about the semantic tower."))
# ---- the structural branch, priced by the single difference between the pair
# `deploy_sdedit` and `deploy_txt2img` are the same code path, the same IP condition
# and the same seed; the init is the only difference. So this subtraction is the
# cleanest estimate of what the structural tower contributes that this pipeline can
# produce, and it is the number that decides whether the branch stays.
if sdedit and sem_only:
    for k in ("clip", "inception", "pixcorr", "ssim"):
        delta = float(sdedit[k]) - float(sem_only[k])
        notes.append(f"- Structural init (deploy_sdedit - deploy_txt2img) moves {k} "
                     f"by {delta:+.3f}.")
    dc = float(sdedit["clip"]) - float(sem_only["clip"])
    notes.append(
        "- The two deployed arms differ in exactly one thing, the structural init. "
        + ("The init helps on CLIP, so the branch is earning its place -- read the "
           "margin against the +-2.8-point standard error on a 200-way score before "
           "believing a small one."
           if dc > 0.01 else
           "The init does not help, so the structural branch is not contributing and "
           "the semantic arm is the one to keep. No arm of the previous pipeline "
           "could have said this, because every arm consumed the same init and the "
           "difference cancelled."))
if sdedit and noise:
    d = float(sdedit["clip"]) - float(noise["clip"])
    notes.append(f"- The EEG-null control (noise_txt2img, no EEG-derived input at "
                 f"all) scores CLIP 2-way {f(noise['clip'])} against "
                 f"{f(sdedit['clip'])} for deploy_sdedit (delta {d:+.3f}). "
                 + ("The gap is what the EEG contributes beyond the SDXL prior."
                    if d > 0.01 else
                    "This gap is within noise: the pictures are coming from the prior, "
                    "not from the EEG."))
if noise and noise_sd:
    d = float(noise_sd["clip"]) - float(noise["clip"])
    notes.append(f"- The previous pipeline's control (`noise_sdedit`) shares the "
                 f"structural init with `deploy_sdedit` and therefore consumes an "
                 f"EEG-derived input; it scores {d:+.3f} on CLIP against the clean "
                 f"control. That gap is the size of the leak the old control carried, "
                 f"and it is quoted so the older tables can be read against these.")
if sem_only and noise and noise_sd:
    notes.append(f"- Quote the semantic arm against the clean control: "
                 f"deploy_txt2img - noise_txt2img = "
                 f"{float(sem_only['clip']) - float(noise['clip']):+.3f} on CLIP, "
                 f"against {float(sem_only['clip']) - float(noise_sd['clip']):+.3f} "
                 f"if the old control is used instead.")
if notes:
    lines.append("## What the arms say")
    lines.append("")
    lines.extend(notes)
    lines.append("")

lines.append("## Provenance")
lines.append("")
for r in rows:
    lines.append(f"- `{r['tag']}`: n={r['n']}, generated in `{r['gen_dir']}`")
lines.append("")

out_md = met / "PATCH_DUAL_SEVEN.md"
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

summary = {
    "pipeline": "EEG Patch dual tower",
    "subject": f"sub-{int(os.environ['SUBJ']):02d}",
    "tag": os.environ.get("TAG", ""),
    "arms": {r["tag"]: {k: r[k] for k in KEYS} for r in rows},
    "table": str(out_md),
}
(met / "patch_dual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\n".join(lines))
print(f"[5] wrote {out_md}")
print(f"[5] wrote {met / 'patch_dual_summary.json'}")
PY

log "===== DONE @ $(date -Iseconds) ====="
