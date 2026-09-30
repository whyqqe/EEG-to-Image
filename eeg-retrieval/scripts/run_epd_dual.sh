#!/usr/bin/env bash
# =============================================================================
# EEG Patch (epd) dual tower -> SDXL -> the seven standard metrics.
#
# One command, six stages. Stages are idempotent: a stage whose artefact already
# exists is skipped, so a requeue or a re-run after a failure resumes instead of
# rebuilding. Set SKIP_<STAGE>=1 to force a skip.
#
# This is `run_patch_dual.sh`'s successor. What changed, and why each change is a
# measurement rather than a preference, is in the "what changed" block below.
#
# -----------------------------------------------------------------------------
# The architecture
# -----------------------------------------------------------------------------
# Two INDEPENDENT EEG Patch towers over the same raw EEG, with different trunks,
# different input geometry, and different targets:
#
#   semantic tower   EEGiT patch representation (time x anatomical regions)
#                    -> ViT-B/16 ImageNet-21k, the released EEGiT backbone
#                    -> global_pool='avg' + EEGiT ProjectionHead -> 1024-d
#                    objective: InfoNCE against CLIP ViT-H-14 block26
#                    readout:   an IP-Adapter condition in CLIP's joint space
#
#   structure tower  EEGiT's region-band patch geometry (same interface as above)
#                    -> DINOv3 ViT-B/16 (see ARCH), a structure-focused prior
#                    -> 14x5 patch tokens -> conv decoder -> 4x64x64 VAE latent
#                    objective: MSE on SDXL VAE latents (`--vae-loss mse`)
#                    readout:   a 512x512 SDEdit init image (VAE head, decoded)
#
# "Independent" is the load-bearing word and it is a consequence of the design, not
# a style choice. The two towers are initialised from *different* pretrained
# checkpoints, so from step 0 there is no parameter they share. There is no
# configuration in which they could have shared a trunk.
#
# -----------------------------------------------------------------------------
# The premise being tested
# -----------------------------------------------------------------------------
# EEGiT's ablation puts a price on two things (THINGS-EEG, intra-subject Top-1):
#
#   pretrained ViT weights      +6.8   (63.6 -> 70.4)
#   EEG patch representation   +16.4   (54.0 -> 70.4)
#
# Read together, the claim is: *a pretrained vision model's prior transfers into an
# EEG encoder when the EEG is presented in a form that model's own layers can read*,
# and the tokenization interface is worth more than the choice of backbone. The
# semantic tower here takes that claim at face value and reproduces both halves.
#
# The structural tower asks the same question of a prior that is about *where*
# rather than *what*. That is the run's actual question, and it has a control: the
# ARCH axis includes EEGiT's own backbone, so "a structure-focused prior" can be
# compared against "EEGiT's generic prior, same mechanism, same geometry, same
# losses". Without that arm the architectural change and the prior change would be
# confounded.
#
# -----------------------------------------------------------------------------
# What changed from run_patch_dual.sh, and the measurement behind each
# -----------------------------------------------------------------------------
# 1. STRUCTURAL TRUNK: DINOv2-L -> DINOv3/MAE ViT-B/16.
#    DINOv2-L was 304M parameters against 16540 training pairs, i.e. ~18k parameters
#    per pair, and the whole `--struct-freeze-blocks 20` machinery existed only to
#    brake that. ViT-B/16 is 86M. More importantly the *reason* for choosing a
#    structural backbone was never checked: the `dino_l` probe entry (val@1 8.33 vs
#    clip_pooled 14.13) was read as "DINOv2 is only half as decodable as CLIP", but
#    that probe fitted a ridge to DINOv2's GLOBAL POOLED vector -- the one readout
#    that throws away the spatial arrangement the structural tower exists to use.
#    The probe measured the wrong tensor for this decision.
#
# 2. STRUCTURAL GEOMETRY: a genuine 2D scalp topography -> BACK TO EEGiT's region
#    bands. This is a reversal of the previous run, and it is a measurement.
#
#    The previous run's argument for topography was sound on its own terms: EEGiT
#    interpolates each anatomical region along ONE axis ("linear interpolation ...
#    along the spatial dimension"), so nothing in its layout distinguishes left from
#    right -- P7 and P8 collapse toward the same coordinate -- and the visual evoked
#    response IS lateralised, so a spatial target arguably needs the left/right axis.
#    `ScalpTopographyTokenizer` was built to supply it: `n_time_bands` 2D scalp maps
#    by inverse-distance weighting from the 63 10-20 positions, stacked along the
#    image height, so every 16x16 patch is a locally coherent 2D tile of the scalp at
#    one time band.
#
#    What was never checked was whether a trunk fed that image can still be read for
#    the VAE latent. `run_epd_struct_probe.sh` checked it, by fitting a closed-form
#    ridge from each candidate tensor to the latent and scoring it against the
#    constant predictor (val-selected lambda, 200-way, sub-08):
#
#      row (all fitted to the SAME VAE latent)      val@1  test@1    rank  margin
#      raw EEG, 10 view slots (the floor to beat)     5.07    6.00    24.8  +0.0617
#      pretrained struct_grid                         1.40    0.00    90.4  -0.0745
#      pretrained struct_gridmean                     0.87    1.00   100.1  -0.0992
#      pretrained struct_pooled                       1.00    0.00   100.3  -0.0928
#      TRAINED    struct_grid                         1.20    0.00   101.5  -0.0923
#      TRAINED    struct_gridmean                     0.93    1.00   102.1  -0.0920
#      TRAINED    struct_pooled                       1.27    0.50    98.9  -0.0917
#      TRAINED    semantic tower's 1024-d embed       4.47    6.50    38.9  +0.0502
#      (chance: val@1 0.50, rank 100.5)
#
#    Three things follow, and the third is the one that decides this run.
#
#    (a) The topography trunk carries NOTHING -- not "little", nothing. Every margin
#        is negative, i.e. every one of those six tensors is at or below the constant
#        map, which is a stronger statement than a low Top-1. One of them
#        (trained_struct_grid) has a variance ratio of 0.9797, so the tensor does
#        move; it moves in directions unrelated to the stimulus.
#
#    (b) Training did not destroy anything: best pretrained 1.40 vs best trained 1.27.
#        So no loss function could have extracted the latent from these features, and
#        item 4 below cannot be blamed for what happened. This is what the probe
#        existed to separate.
#
#    (c) The VAE latent IS reachable through a deep ViT EEG encoder, at the raw-EEG
#        ceiling, using EEGiT's OWN geometry: the semantic tower's 1024-d embedding
#        scores margin +0.0502 / test 6.50 / rank 38.9 against the raw-EEG ridge's
#        +0.0403 / 6.50 / 27.9. The target is not the obstacle. The interface is, and
#        there is a working interface in this codebase to copy.
#
#    That ordering -- interface decides, backbone does not -- is EEGiT's own ablation
#    (+16.4 layout vs +6.8 pretrained weights) reproduced in a different context, and
#    it was reproduced the hard way: the three-arm run varied the BACKBONE across
#    DINOv3/MAE/in21k on the topography geometry and all three failed identically.
#
#    So the structural tower uses `--struct-tokenizer eegit`: the semantic tower's
#    region-band geometry, (3, 224, 80) -> 14x5 = 70 tokens. It keeps the DINOv3
#    trunk, because the separate trunk is what makes this a second tower rather than
#    a copy of the first -- and because the probe says the trunk was never the lever.
#
# 3. THE DEPTH HEAD IS GONE, and this one is not a preference at all. The stage-0
#    ridge probe (`outputs/sub*/probe_targets_*.json`) fitted a closed-form map from
#    the raw EEG to every candidate target and scored it against the constant
#    predictor -- which is exactly what L1 falls back to when the target is not in
#    the input:
#
#      target  subject/channels   r(pred,gt)  r(const,gt)    margin   var ratio
#      depth   sub-08 / 63ch        +0.1598     +0.5333     -0.3735     0.2233
#      depth   sub-08 / 17ch        +0.1617     +0.5333     -0.3716     0.3211
#      depth   sub-10 / 63ch        +0.1577     +0.5333     -0.3756     0.2218
#      depth   sub-10 / 17ch        +0.1695     +0.5333     -0.3638     0.1924
#
#    A constant map correlates +0.5333 with a real depth map while EEG's best LINEAR
#    read-out reaches +0.16, on four independent arms. The margin is negative
#    everywhere, so the linear ceiling for depth is the mean, no head design recovers
#    it, and the head that was shipped (r = +0.6485 against the constant's +0.6540)
#    had already reached that ceiling. Retraining it spends gradient on a target the
#    input does not contain. Generating the depth cache also cost ~13 min of GPU for
#    16540 images, which this pipeline no longer pays.
#
# 4. THE VARIANCE FLOOR IS REPLACED BY A PLAIN MSE (`--vae-loss mse --w-var 0`).
#    The previous run's floor did not work, and the probe says why it could not have.
#
#    The floor's premise was that L1's optimum on a weakly predictable target is the
#    constant mean field, so a hinge pushing prediction variance up would force the
#    head off that solution. The premise about L1 is correct -- and it is still true
#    of the shipped configuration. But the hinge has an escape that makes it useless
#    as written: it constrains only the prediction's VARIANCE, so "the mean field plus
#    zero-mean noise" satisfies it exactly while costing almost nothing in L1. That is
#    not a hypothesis; it is what the last run produced (variance ratio 0.6577,
#    healthy; per-sample r = +0.063 against the constant predictor's +0.157, i.e.
#    worse than the mean field). The floor was measuring the head's liveliness and
#    reporting it as progress.
#
#    MSE closes the same escape by construction rather than by adding a term to catch
#    it: a prediction is penalised by the squared error, so zero-mean noise is charged
#    for its own variance and a constant is charged for the target's full variance.
#    The gradient pressure toward an instance estimate is then first-order rather than
#    optional. It is also the loss every successful row of the probe above was fitted
#    with (a closed-form L2 ridge, margin +0.05 at the semantic-tower features and
#    +0.06 on raw EEG), so there is direct evidence that L2 suffices on these
#    features; the corresponding evidence for L1 does not exist anywhere in this
#    project.
#
#    Honest ranking of what is being changed here: item 2 is MEASURED (six tensors at
#    the constant predictor, and a working alternative at the raw-EEG ceiling); this
#    item is ARGUED (a proven degenerate solution in the old term, and an L2 fit that
#    demonstrably works on the features). If this run's VAE branch still fails while
#    the trunk probe above now reads positive, item 4 is the next thing to reopen --
#    not the geometry, and not the backbone.
#
# 5. THE SEMANTIC TOWER KEEPS EEGiT's ARCHITECTURE and does NOT keep its schedule.
#    This is the one deviation worth stating plainly, because it looks like
#    cherry-picking and is not. `run_eegit_gate.sh` measured four arms of increasing
#    EEGiT-consistency on the identical split:
#
#      arm (EEGiT-consistency)                test@1   val@1   eps@peak
#      our own config            (nothing)     46.50   34.40      19
#      + official patch image    (layout)      43.50   32.40      19
#      + official loss/optimizer (fuse)        39.50   31.13      28
#      + official head, 1 layer  (full)        35.50   31.20      20
#
#    A Cochran-Armitage trend test gives z = -2.38, p = 0.018 on test: adopting MORE
#    of the released recipe made retrieval WORSE, monotonically. The `full` arm
#    differs from official EEGiT in exactly two ways -- the alignment target and a
#    frozen image encoder -- so the honest reading is that a large part of their 70.4
#    comes from aligning against a target THAT IS TRAINED, which bends the target
#    space toward whatever the EEG can predict. That is not available to this
#    pipeline: IP-Adapter consumes 1024-d CLIP joint embeddings, so the target has to
#    be a frozen layer of a model whose projected space IP-Adapter reads.
#
#    So the interface, the backbone, the pooling, the head and the 768->1024
#    projection are EEGiT's exactly -- that is the +16.4 and the +6.8, i.e. the
#    innovation -- and the optimizer/schedule is ours, because that is the part that
#    was measured. Concretely: `time-region` layout, `vit_b16_in21k_orig`,
#    `global_pool=avg`, `pool=mean`, `head_kind=eegit` on both sides, `d_embed=1024`;
#    AdamW at 5e-4/5e-5 with 5-epoch warmup + cosine and EMA, which is what the best
#    local semantic run used.
#
# -----------------------------------------------------------------------------
# Checkpoint selection
# -----------------------------------------------------------------------------
# `sel = val_top1 + 0.5 * vae_top1`, both in the same 0-100 units. Selection runs on
# a concept-level holdout of the TRAIN concepts; the 200 test concepts are scored
# once, at the end, by the selected checkpoint.
#
# The depth term was dropped with the head (5.0 * depth_pearson stopped being a
# number worth 5 points when the target's linear ceiling is the mean). `struct_sel_w`
# stays at 0.5 rather than being re-tuned: it was not what let the VAE head collapse
# -- the head was already at chance under it -- so re-weighting it would tune a term
# that has not been shown to be mis-set and would break comparability with the runs
# it was set for.
#
# EMA is on (`--ema-decay 0.999`). Selection reads the EMA weights, and the EMA
# weights are what get saved, so the checkpoint that is scored is the checkpoint that
# is shipped. In `epd_localize.sh` this moved the val peak to within 4 epochs of the
# test peak for an arm whose raw val peak lagged test by 37.
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
# held fixed.
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
# -----------------------------------------------------------------------------
# Cost, per array task
# -----------------------------------------------------------------------------
# Estimated on an 80 GB GPU, batch 128:
#   [0] VAE target cache        already built and content-validated; skipped
#   [1] train, 100 epochs, 2 towers, ~117 steps/epoch                  ~1.5-2 h
#   [2] export                  200 forwards + 200 VAE decodes            ~10 min
#   [3] generate                4 arms x 200 images x 28 steps            ~40 min
#   [4] seven metrics           4 arms x (2-way + FID + SwAV)             ~25 min
# so ~3 h of GPU; the sbatch asks for 6 h.
#
# Batch 128 and not EEGiT's 1024
# -----------------------------
# The loop runs in fp32 with no autocast and no gradient checkpointing, so both
# towers store activations for all 12 of their blocks. At batch 128 that is
# comfortable on either a 40 or an 80 GB card, and the partition is not homogeneous
# enough to bet on which one the scheduler hands out. The batch size was also the one
# axis the six-arm diagnostic sweep tested directly (`b512`), and 512 was not better.
#
# Why the smoke uses the real batch size
# --------------------------------------
# The smoke's job is to catch a memory failure at the real batch size, so it runs the
# real config, the real trunks and the real losses, and shortens the run only with
# --limit-samples. A smoke at a smaller batch would exercise a memory profile the real
# run never has.
#
# Usage:
#   bash scripts/run_epd_dual.sh --dry-run   # validate flags, touch nothing
#   bash scripts/run_epd_dual.sh             # the real thing
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
SUBJ="${SUBJ:-8}"

# The structural trunk. This is the run's independent variable: every other flag is
# held fixed across the arms, so the difference between them is attributable to the
# pretrained prior alone.
#
#   dino3   self-supervised with Gram anchoring. DINOv3's stated motivation is that
#           "global metrics ... continue to improve, but dense metrics ... degrade";
#           the fix regularises the pairwise similarity between PATCH features, i.e.
#           it pins exactly the quantity "structure" means here. ViT-L reaches depth
#           RMSE 0.352 and 54.9 mIoU on frozen-backbone dense probes.
#   mae     trained by reconstructing masked pixels, so it retains the most
#           recoverable appearance: LPIPS 0.11 / recon-FID 0.16 against DINOv2-B's
#           0.255 / 0.49 in a controlled decoder comparison. The control for the
#           claim that "self-supervised spatial coherence" beats "pixel fidelity".
#   in21k   EEGiT's own backbone. NOT a structure-focused prior, and that is the
#           point: it holds the mechanism, the geometry, the decoder, the losses and
#           the schedule identical to the other two and changes only the pretrained
#           weights, so it prices "a structural prior exists" against "any prior".
ARCH="${ARCH:-dino3}"
case "${ARCH}" in
  dino3)  STRUCT_BACKBONE="timm:dinov3_b16" ;;
  mae)    STRUCT_BACKBONE="timm:mae_b16"    ;;
  in21k)  STRUCT_BACKBONE="timm:vit_b16_in21k_orig" ;;
  *) echo "[FATAL] ARCH must be dino3|mae|in21k, got '${ARCH}'" >&2; exit 2 ;;
esac

# The structural tower's INPUT GEOMETRY. This is the run's decisive flag, so it is a
# named variable rather than a literal buried in the flag list. `eegit` is the
# measured-good interface (see item 2 at the top of this file); `topography` is kept
# reachable so the probe's finding can be re-checked end-to-end without editing this
# script, but shipping anything other than `eegit` means shipping a geometry the
# probe placed at the constant predictor.
STRUCT_TOKENIZER="${STRUCT_TOKENIZER:-eegit}"
case "${STRUCT_TOKENIZER}" in
  eegit|topography|grid) ;;
  *) echo "[FATAL] STRUCT_TOKENIZER must be eegit|topography|grid, got '${STRUCT_TOKENIZER}'" >&2; exit 2 ;;
esac

TAG="${TAG:-epd_dual_${ARCH}}"
OUT="${OUT:-${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")}"
# Structural target caches live at a single shared path, NOT under the subject's
# output dir. The targets are VAE encodings of the stimulus images; they do not
# depend on which subject's EEG is being decoded. Deriving this from ${OUT} would
# point sub-10 at an empty directory and re-encode the same 16540 images.
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
SKIP_TESTS="${SKIP_TESTS:-0}"

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
# array submission, where one task read the cache another was still writing and
# trained for two minutes against a target that was 58.65% zeros. The tell was in its
# own log, one debug line apart: the same cache reported per-channel stds of
# 0.315/0.246/0.269/0.214 at the smoke and 0.527/0.411/0.447/0.356 at the training
# start, because the writer was still adding non-zero rows in between.
#
# The checks below are aimed at that specific failure: sample rows ACROSS the whole
# array including the last one (a truncated write is zero in the tail), reject a
# non-trivial zero fraction, and require a plausible spread.
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
# The training configuration. Every line is a decision.
# =============================================================================
sem_common=(
  --subject "${SUBJ}"
  --tag "${TAG}"
  --out-dir "${OUT}"
  # ---- EEGiT's EEG patch representation, exactly ---------------------------
  # `time-region` reproduces the released code's construction: H = time (250 -> 224
  # by one 2D bilinear resample), W = 5 anatomical regions anterior -> posterior in
  # the dataset's own channel order, one 2D interpolation per region. This is the
  # +16.4 half of EEGiT's ablation, so it is taken verbatim rather than improved.
  --tokenizer eegit
  --patch-style time-region
  --patch-size 16
  --n-patches-w 14               # 14 x 16 = 224 along time; 5 x 16 = 80 regions
  --channels all                 # 5 EEGiT regions -> 5 x 14 = 70 patches
  # ---- EEGiT's backbone, named by the tag the official code names ----------
  # `vit_b16_in21k_orig` resolves to the same weights as `vit_b16_in21k` (timm maps
  # the deprecated name to `vit_base_patch16_224.augreg_in21k`); the key is kept so
  # the record can say it used the official tag.
  --backbone timm:vit_b16_in21k_orig
  --freeze-blocks 0
  # ---- EEGiT's readout, and it has to be ONE layer with NO fusion ------------
  # The released code reads the FINAL block's pooled vector and feeds that single
  # tensor to its ProjectionHead. A layer-fusion module therefore cannot sit in
  # front of it: the head would be consuming a blend the official code never
  # produces, while the config still recorded `head_kind=eegit`. `train.py` refuses
  # the combination for exactly that reason, and this arm -- whose entire purpose is
  # to be the EEGiT-consistent half of the pair -- takes the official readout.
  #
  # This is a deliberate trade, not a free one. `run_eegit_gate.sh` measured four
  # arms of increasing EEGiT-consistency on this same target space, and only two of
  # those differences are one-variable:
  #
  #   layout   our fusion + our head + OUR loss/optimizer    test@1 43.50  (46.50 unscheduled)
  #   fuse     our fusion + our head + OFFICIAL loss/opt      test@1 39.50
  #   full     NO fusion + official head + OFFICIAL loss/opt   test@1 35.50
  #
  # `layout -> fuse` isolates the official loss/optimizer at about -4 points, and
  # that is the half this arm declines (the schedule below is ours). `fuse -> full`
  # is where the readout changes -- and it changes TWO things at once, dropping the
  # fusion AND swapping the head, so it does NOT establish that the official head is
  # worse under our optimizer. No arm in that gate isolates the head, so the choice
  # here rests on the design intent rather than on a measurement, and the 4-point gap
  # is the size of what that intent costs if the head turn out to be the weaker half.
  # Tracking it is the point of running the head-consistent arm at all.
  --layers 12
  --fusion-mode none
  # ---- EEGiT's head --------------------------------------------------------
  # `global_pool=avg` makes timm build an `fc_norm` LayerNorm applied after average
  # pooling, so it is NOT the same tensor as pool=mean with an empty global_pool.
  # `head_kind=eegit` is the released ProjectionHead on both sides (Linear -> GELU ->
  # Linear -> Dropout(0.5) -> + the PRE-GELU projection -> LayerNorm), and `d_embed
  # 1024` is the paper's 768 -> 1024 FC.
  --pool mean
  --timm-global-pool avg
  --head-kind eegit
  --img-head-kind eegit
  --head-drop 0.5
  --d-embed 1024
  # ---- the alignment target: the one necessary deviation from EEGiT --------
  # Frozen CLIP ViT-H-14 block26 (1280-d, 1024-d after the joint projection). EEGiT
  # aligns to its OWN trainable ViT-B/16 + ProjectionHead, so their target space
  # bends toward whatever the EEG can predict. This pipeline cannot do that: the
  # generation stack consumes 1024-d CLIP joint embeddings, and a learned space is
  # not one. This is the main reason to expect their reported retrieval number to be
  # out of reach, and it is stated rather than tuned away.
  --target-features "${FEAT}"
  --target-layer block26
  --target-fusion single
)

# ---- the schedule: ours, not EEGiT's, and the gate above says why ------------
# EEGiT uses torch.optim.Adam with one flat 5e-5, no warmup and no decay. That was
# measured here as the worst of four arms (35.50 against 46.50), on a split where
# adopting more of the released recipe made retrieval monotonically worse (z=-2.38,
# p=0.018). So the schedule is the one the best local semantic run used instead.
epd_sched=(
  --optimizer adamw
  --lr 5e-4                      # the new parts: fusion, heads
  --backbone-lr-mult 0.1         # the blocks at 5e-5, which IS EEGiT's encoder LR
  --warmup-epochs 5
  --cosine
  --min-lr-ratio 0.01
  --ema-decay 0.999
  --ema-warmup-steps 200
  --epochs 100
  --batch-size 128               # fp32, no autocast; see the cost note
  --patience 0                   # 0 = disabled, so the schedule completes
  --aug full
  --stage1-epochs 0              # MMD off
  --seed 2025
  --fit-diagnostic
)

train_cfg=("${sem_common[@]}" "${epd_sched[@]}"

  # ---- structure tower: an independent EEG Patch tower ---------------------
  --struct-backbone "${STRUCT_BACKBONE}"
  --struct-patch-size 16         # must equal the trunk's patch_embed kernel
  # EEGiT's own region-band geometry, (3, 224, 80) -> 14x5 = 70 tokens. Chosen by
  # `run_epd_struct_probe.sh`, which fitted a ridge from each candidate tensor to the
  # VAE latent: this interface's features reach the raw-EEG ceiling (margin +0.0502)
  # while the topography interface's sit at the constant predictor at initialisation
  # AND after training (six tensors, margin -0.07 to -0.10). See item 2 at the top.
  --struct-tokenizer "${STRUCT_TOKENIZER}"
  --struct-n-patches-w 14        # the time-axis extent: 14 x 16 = 224 px
  --struct-layers 8 10 12
  --struct-fusion-mode uniform
  --struct-freeze-blocks 0       # 86M trunk, not 304M: the brake is not needed
  # Absolute LRs, not multipliers of `--lr`: the decoder is a randomly-initialised
  # conv stack and cannot be trained at the pretrained trunk's step size. Writing the
  # multiplier form would produce a number whose meaning is invisible at the call
  # site.
  --struct-lr 5e-5
  --struct-head-lr 5e-4
  --struct-drop 0.1
  --struct-base-ch 128
  --struct-field-ch 32

  # ---- structural target: VAE latents ONLY, no depth head ------------------
  # See the depth table at the top of this file: the depth target's linear ceiling is
  # the constant predictor on four independent arms, so the head is removed rather
  # than retuned. Both keys are concept*10+slot, and the cache is SHARED across
  # subjects because a VAE latent is a property of the stimulus, not of the recording.
  --vae-latents "${VAE_CACHE}"
  --w-vae 1.0
  # MSE, not L1, and the floor is off. The previous `--w-var 1.0` hinge constrained
  # only the prediction's variance, so "mean field + zero-mean noise" satisfied it at
  # a variance ratio of 0.6577 while carrying nothing (per-sample r +0.063 against the
  # constant's +0.157). MSE charges for that noise by construction, so the instance
  # pressure is in the loss rather than in a term that can be gamed. See item 4.
  --vae-loss mse
  --w-var 0

  # ---- checkpoint selection -------------------------------------------------
  --struct-sel-w 0.5
)

# =============================================================================
if [[ "${1:-}" == "--dry-run" ]]; then
  # Flag-level validation without a GPU, without the dataset, and without the
  # structural caches: this is the check that a bad combination fails here rather
  # than after an allocation has been granted.
  log "validating the exact flag list (no GPU, no dataset): ARCH=${ARCH}"
  "${PY}" -u "${ROOT}/scripts/epd/train.py" "${train_cfg[@]}" --validate-only || exit 1
  for f in "${ROOT}/scripts/epd/train.py" \
           "${ROOT}/scripts/epd/export_conds.py" \
           "${NB_ROOT}/scripts/nda/generate_struct_inject_decode.py" \
           "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
           "${NB_ROOT}/scripts/nda/build_gt_vae_latents.py"; do
    [[ -f "$f" ]] || die "pipeline references a script that does not exist: $f"
  done
  [[ -d "${FEAT}/train" ]] || die "no semantic target features at ${FEAT}/train"
  log "dry run ok"
  exit 0
fi

mkdir -p "${TGT}" "${EXP}" "${GEN}" "${MET}" "${ROOT}/outputs/logs"
cd "${ROOT}"

log "=============================================================="
log "EEG Patch (epd) dual tower, subject ${SUBJ}, tag ${TAG}"
log "  sem       : EEGiT patch image -> ${BACKBONE_LABEL:-vit_b16_in21k} -> CLIP block26"
log "  struct    : EEGiT geometry (${STRUCT_TOKENIZER}) -> ${ARCH} (${STRUCT_BACKBONE}) -> VAE latents"
log "  outputs   : ${EXP} -> ${GEN} -> ${MET}"
log "=============================================================="

# -----------------------------------------------------------------------------
# [0] Structural target cache
# -----------------------------------------------------------------------------
# The train half is the expensive one (16540 images). Whether the row order is
# actually right is not asserted anywhere -- it cannot be, from the files alone --
# but it is *observable* in the training run: if the cache were misaligned, vae_top1
# would sit at chance and the variance ratio at 0, because the model would be asked
# to predict one image's layout from another image's EEG.
if [[ "${SKIP_TARGETS}" != "1" ]]; then
  log "[0] structural target cache @ $(date -Iseconds)"
  VAE_TRAIN="${VAE_CACHE}/train_vae_latents_f16.npy"
  VAE_TEST="${VAE_CACHE}/test_vae_latents_f16.npy"
  mkdir -p "${TGT}"

  if vae_cache_ok "${VAE_TRAIN}" >/dev/null 2>&1 && vae_cache_ok "${VAE_TEST}" >/dev/null 2>&1; then
    log "[0] vae latents present and content-valid, skipping (delete them to rebuild)"
  else
    # The cache is shared across subjects and across ARCH arms, so several array
    # tasks reach this point at once. `flock` serialises them; without it they would
    # all encode the same 16540 images and publish into the same paths concurrently.
    exec {VAE_LOCK}>"${TGT}/.vae_cache.lock"
    log "[0] waiting on the vae cache lock (held by another array task, if any)"
    flock "${VAE_LOCK}"
    if vae_cache_ok "${VAE_TRAIN}" >/dev/null 2>&1 && vae_cache_ok "${VAE_TEST}" >/dev/null 2>&1; then
      log "[0] the build finished while waiting for the lock; reusing it"
    else
      # Build into a staging dir and publish with `mv`. Same filesystem, so the
      # rename is atomic: a concurrent reader sees either the previous complete file
      # or the new one, never a partial one. This is the actual fix for the race --
      # the lock alone would not protect a reader that does not take it, whereas an
      # atomic publish does, by construction.
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

      # Validate BEFORE publishing. An invalid build must never reach the shared
      # path, because the next run's fast path would accept it.
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
  # in the header), and `build_gt_depth_cache.py` costs ~13 min of GPU on 16540
  # images to produce a target whose linear ceiling is the mean.
else
  log "[0] skipped by SKIP_TARGETS"
fi

# -----------------------------------------------------------------------------
# [1] Train
# -----------------------------------------------------------------------------
# The smoke gate runs the REAL flag list through the same code path for 3 epochs on
# 1024 samples: both tokenizers, the two z-score statistics, both trunks, the RoPE
# path if the trunk uses one, both losses, the three LR groups, the dual selection,
# the export provenance. Without it a flag that only breaks on the second epoch (or
# an OOM at the real batch size) surfaces an hour later.
if [[ "${SKIP_TRAIN}" != "1" ]]; then
  if [[ "${SKIP_TESTS}" != "1" ]]; then
    log "[1] unit tests @ $(date -Iseconds)"
    # Run once and keep the output: the gate needs the exit code and the reader
    # needs the tail, and running the suite twice costs a minute of GPU-node CPU
    # for nothing.
    UT_LOG="${ROOT}/outputs/logs/${TAG}_unit_tests.log"
    if ! "${PY}" -u "${ROOT}/scripts/test_epd_arch.py" >"${UT_LOG}" 2>&1; then
      tail -40 "${UT_LOG}"
      die "scripts/test_epd_arch.py failed (full log: ${UT_LOG})"
    fi
    tail -6 "${UT_LOG}"
  fi

  log "[1] smoke: the real config, 3 epochs, 1024 fit samples (real batch size)"
  "${PY}" -u "${ROOT}/scripts/epd/train.py" "${train_cfg[@]}" \
    --tag "_smoke_${TAG}" --epochs 3 --warmup-epochs 1 \
    --limit-samples 1024 --fit-diag-concepts 20 2>&1 | tail -30
  SMOKE_RC=${PIPESTATUS[0]}
  if [[ "${SMOKE_RC}" -ne 0 ]]; then
    die "smoke run failed (rc=${SMOKE_RC}); not starting the real run"
  fi
  require "${OUT}/_smoke_${TAG}_best.pt"

  # ---- export smoke ---------------------------------------------------------
  # The export runs hours after training, in a separate process, against a
  # checkpoint it cannot inspect interactively -- exactly the place a drift between
  # the training constructor and the export constructor is discovered too late to
  # fix. So the same code path is exercised here on the smoke checkpoint, on 8 test
  # concepts, including the VAE decode, into a directory that is thrown away.
  log "[1] smoke: the export path on the smoke checkpoint, 8 concepts"
  # `--allow-collapse` is required here and only here. A 3-epoch smoke is a
  # collapsed model by construction, so the across-sample gate would abort it -- and
  # the smoke exists to catch a wiring error between the training and export
  # constructors, not to check convergence. The real export below is NOT given this
  # flag: that is where the gate has to hold, and it is the only place it can,
  # because it is the last point with the ground truth still on disk.
  "${PY}" -u "${ROOT}/scripts/epd/export_conds.py" \
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
conditions do not build, so the real run would train for ~2 h and then fail at stage 2"
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
    "${PY}" -u "${ROOT}/scripts/epd/train.py" "${train_cfg[@]}"
    RC=$?
    [[ "${RC}" -eq 0 ]] || die "training failed (rc=${RC})"
  fi
else
  log "[1] skipped by SKIP_TRAIN"
fi
require "${OUT}/${TAG}_best.pt" "${OUT}/${TAG}_result.json"

# -----------------------------------------------------------------------------
# [2] Export the conditions
# -----------------------------------------------------------------------------
# deploy: soft retrieval over the 1654 TRAIN concept CLIP embeddings, performed in
#         the model's own 1024-d space (where the metric is meaningful) and
#         realised as a weighted sum of the gallery's real CLIP embeddings (where
#         IP-Adapter can consume it). The retrieval cannot return the answer: the
#         gallery is disjoint from the 200 test concepts.
# noise:  EEG replaced by zeros.
# Plus, from the structure tower: pred_vae_test_scaled.npy and the decoded RGB init.
if [[ "${SKIP_EXPORT}" != "1" ]]; then
  log "[2] export conditions @ $(date -Iseconds)"
  "${PY}" -u "${ROOT}/scripts/epd/export_conds.py" \
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
# No text prompt on any arm. The semantic condition is the IP embedding alone, which
# keeps the comparison to the noise arm exact and avoids reintroducing a text path
# that would have to be built from EEG to be legitimate.
#
# `--control-guidance-end 0.5` confines the ControlNet to the layout phase of
# denoising; letting it act through the last steps is what makes a ControlNet
# overwrite the semantics the IP condition is carrying.
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
# init. That is what prices the structural branch, and no arm could answer it
# before, because every arm consumed the same init -- so a structural tower that
# helped, one that did nothing, and one that actively hurt all produced the same
# comparison.
#
# `noise_txt2img` is the one arm that touches no EEG-derived input at all, so it is
# what the "is any of this from the EEG" delta is read against. The previous
# pipeline's control was `noise_sdedit`, which still consumed the structural init --
# and the structural init is EEG-derived, so the control shared a pathway with the
# arm it was meant to control for. That leak was small once the head was known to be
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
TAG="${TAG}" SUBJ="${SUBJ}" ARCH="${ARCH}" OUT="${OUT}" MET="${MET}" GEN="${GEN}" \
EXP="${EXP}" PY="${PY}" "${PY}" - <<'PY'
import json, os
from pathlib import Path

met = Path(os.environ["MET"])
arch = os.environ.get("ARCH", "")
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

# The published references, copied from eval_standard_seven_table.py so the table is
# readable on its own. Only the sub-08 rows are directly comparable.
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
    f"# EEG Patch dual tower ({arch}), sub-{int(os.environ['SUBJ']):02d} reconstruction, "
    f"seven standard metrics",
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
if sdedit and sem_only:
    for k in ("clip", "inception", "pixcorr", "ssim"):
        delta = float(sdedit[k]) - float(sem_only[k])
        notes.append(f"- Structural init (deploy_sdedit - deploy_txt2img) moves {k} "
                     f"by {delta:+.3f}.")
    dc = float(sdedit["clip"]) - float(sem_only["clip"])
    notes.append(
        "- The two deployed arms differ in exactly one thing, the structural init. "
        + ("The init helps on CLIP, so the branch is earning its place -- read the "
           "margin against the ~2.8-point standard error on a 200-way score before "
           "believing a small one."
           if dc > 0.01 else
           "The init does not help, so the structural branch is not contributing and "
           "the semantic arm is the one to keep."))
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
                 f"control. That gap is the size of the leak the old control carried.")
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

out_md = met / "EPD_DUAL_SEVEN.md"
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

summary = {
    "pipeline": "EEG Patch (epd) dual tower",
    "arch": arch,
    "subject": f"sub-{int(os.environ['SUBJ']):02d}",
    "tag": os.environ.get("TAG", ""),
    "arms": {r["tag"]: {k: r[k] for k in KEYS} for r in rows},
    "table": str(out_md),
}
(met / "epd_dual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\n".join(lines))
print(f"[5] wrote {out_md}")
print(f"[5] wrote {met / 'epd_dual_summary.json'}")
PY

log "===== DONE @ $(date -Iseconds) ====="
