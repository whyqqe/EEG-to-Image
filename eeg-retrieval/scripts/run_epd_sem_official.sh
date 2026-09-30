#!/usr/bin/env bash
# =============================================================================
# epd_sem_official -- the semantic tower trained on EEGiT's FOUR-item design
#
# WHAT THIS IS AND IS NOT -- read this before quoting it as a reproduction
# ----------------------------------------------------------------------
# This run is NOT a reproduction of EEGiT. It is: EEGiT's interface, readout
# architecture, loss and optimizer, on THIS project's data protocol. Three of the
# protocol's ingredients were never taken from EEGiT and are not claimed to match it:
#
#   * 150 of the 1654 training concepts are held out for checkpoint selection
#     (`concept_split(150, 2025)`), so this trains on 1504 concepts where EEGiT trains
#     on all 1654. That is strictly less supervision.
#   * The reported checkpoint is the best on that held-out split. EEGiT performs no
#     selection at all -- intra-subject it runs the full 100 epochs and tests the LAST
#     one. Selection on a held-out set can only help, so these two protocols are not
#     interchangeable and the direction of the bias is known.
#   * `--aug full` is this project's own stacked augmentation (time shift + Gaussian
#     noise + smoothing + channel dropout; `augment.py` calls it "our stacked
#     variant", and the rest of that registry mirrors SAMGA, not EEGiT). EEGiT's
#     augmentation is not among the transcribed parts of its source, so this is an
#     UNVERIFIED match rather than a known one.
#
# Add the target-space deviation below and the correct description is "EEGiT's
# objective, this project's evaluation protocol" -- which is exactly the comparison
# worth making, but it is not the same claim as "we reproduced EEGiT's number".
#
# =============================================================================
# Where the config comes from, and why not from a clone
# -----------------------------------------------------
# EEGiT's code is NOT public. The CVPR 2026 paper is listed on the CVF open-access
# site with "Code: Not disclosed", there is no repository, and `third_party/` here
# holds SAMGA only -- so `git clone` cannot be the source of truth. What the project
# DOES have is a line-by-line reading of the released `base/data_eeg.py`,
# `ClipLoss` and `PLModel.forward`, written down in `run_eegit_gate.sh` and
# `epd/tokenizer.py` with the numbers quoted from the source. That transcription is
# the authority used below, and `scripts/test_eegit_official_interface.py` is the
# machine-checkable half of it: it asserts our patch tensor and our resampled
# `pos_embed` match the official ones rather than merely having the same shape.
#
# The four items, all ON in this run
# ----------------------------------
# 1. EEG patch layout: H = time, W = regions, anterior -> posterior, electrodes in
#    the dataset's own order, ONE 2D bilinear interpolation. Ours:
#    `--patch-style time-region` (`eegit_official` is kept as a legacy alias of it,
#    see `tokenizer.py`). The transpose -- H = regions, W = time, posterior ->
#    anterior, electrodes sorted by montage x -- has the same 70 tokens and the same
#    tensor SHAPE while being a different tensor, so nothing ever raised.
# 2. `pos_embed` resampling with the antialias filter timm uses
#    (`resample_abs_pos_embed`). Our reimplementation had dropped it: max absolute
#    deviation 9.38 in `pos_embed`. Fixed inside `encoders.resample_pos_embed`, so it
#    is ON here by being the shared encoder, not by a flag.
# 3. The loss, `ClipLoss` + `PLModel.forward`:
#      * `logit_scale = softplus(log(1/0.07))` = 2.727, NOT exp(...) = 14.29. The
#        paper's prose says tau = 0.07; the CODE soft-pluses it, which softens the
#        objective 5.24x. `--softplus`.
#      * `self.logit_scale` is a Parameter but is absent from the optimizer's three
#        explicit groups, so tau is FIXED rather than learned. `--fixed-temp`.
#      * Only the image side is L2-normalised before the loss; the raw EEG embedding
#        goes in, so `||z_e||` is a free per-sample logit scale on top of tau.
#        `--no-eeg-l2norm`.
#    These three interact and must be read together: with a fixed tau, `||z_e||` is
#    the only thing controlling how peaked the softmax is, and nothing bounds it.
#    That is why the earlier P3 arm -- which had two of the three, with `exp` instead
#    of `softplus` -- started at an effective logit scale of 14.286*sqrt(1024) = 457
#    and sat at chance for 16 epochs. Here the initial scale is 2.727*32 = 87.3.
# 4. The optimizer: `torch.optim.Adam(weight_decay=1e-4)` over three groups, EEG
#    encoder at lr*10, ProjectionHeads at lr*10, image encoder at lr, launched with
#    `--lr 5e-6` -- i.e. 5e-5 for everything that trains here. Flat: no warmup, no
#    cosine, no EMA. `--optimizer adam --wd-all-params --lr 5e-5
#    --backbone-lr-mult 1.0`, and `--warmup-epochs`/`--cosine` are simply absent
#    (their defaults are 0 and off), as is `--ema-decay` (default 0 = disabled).
#
# The ONE deliberate deviation, stated rather than tuned away
# ----------------------------------------------------------
# The alignment target stays the frozen CLIP ViT-H/14 `block26` (1280-d -> 1024-d),
# not EEGiT's own trainable ViT-B/16 + ProjectionHead. EEGiT aligns EEG to a space
# that ADAPTS to whatever the EEG can predict; we align to a fixed CLIP space,
# because that is the space the generation stack consumes -- an IP-Adapter embedding
# must be a CLIP joint embedding, and a learned EEGiT space would be unusable
# downstream. This is also the single largest reason to expect EEGiT's reported 70.4
# to be unreachable here. It is a real cost of the pipeline, not a bug, and it is the
# one place where "reproduce EEGiT" and "produce a usable condition" genuinely pull
# apart.
#
# Generation is pure IP-Adapter: NO ControlNet, NO depth map
# ---------------------------------------------------------
# The semantic branch is scored on its own, so nothing structural may enter the
# generator. This run therefore uses `generate_ip_txt2img.py`, which is a plain
# `StableDiffusionXLPipeline` + IP-Adapter and never loads a ControlNet at all --
# as opposed to the `cn-scale 0.0` trick, which loads a depth ControlNet, feeds it a
# depth map from a discarded branch, and then multiplies its output by zero. Those
# two are not equivalent in what they prove: `cn-scale 0.0` leaves the arm's inputs
# contaminated by an artefact of the branch we removed, and makes the arm depend on
# that branch's files continuing to exist.
#
# Usage:
#   bash scripts/run_epd_sem_official.sh --dry-run
#   SUBJ=8 SEED=2025 bash scripts/run_epd_sem_official.sh
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
SUBJ="${SUBJ:-8}"
SEED="${SEED:-2025}"

# `--split-seed` is pinned at 2025 for EVERY arm in this project, so this run's 150
# val / 200 test concepts are the same concepts the existing arms were scored on and
# the per-concept vectors pair positionally.
SPLIT_SEED="${SPLIT_SEED:-2025}"
TAG="${TAG:-epd_sem_official_s${SEED}}"
OUT="${OUT:-${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")}"
EXP="${EXP:-${OUT}/${TAG}_export}"
GEN="${GEN:-${OUT}/${TAG}_gen}"
MET="${MET:-${OUT}/${TAG}_metrics}"
FEAT="${FEAT:-${ROOT}/outputs/features/clip_h14_layers}"
CKPT="${CKPT:-${OUT}/${TAG}_best.pt}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[FATAL] %s\n' "$*" >&2; exit 1; }

# =============================================================================
# [0] the official configuration
# =============================================================================
cfg=(
  --subject "${SUBJ}"
  --tag "${TAG}"
  --out-dir "${OUT}"
  --seed "${SEED}"
  --split-seed "${SPLIT_SEED}"

  # ---- item 1: the EEG patch image, official layout -------------------------
  --tokenizer eegit
  --patch-style time-region       # H=time (14 patches), W=regions (5) = 70 tokens
  --patch-size 16
  --n-patches-w 14
  --channels all                  # EEGiT uses 63 channels in BOTH settings
  --backbone timm:vit_b16_in21k_orig   # the tag the official code names
  --freeze-blocks 0

  # ---- the official readout: final block, global avg pool, one projection ---
  --layers 12                     # `--fusion-mode none` requires exactly one layer
  --fusion-mode none              # no fusion module may sit between ViT and head
  --pool mean
  --timm-global-pool avg          # official's global_pool="avg"
  --head-kind eegit               # official ProjectionHead, transcribed verbatim
  --img-head-kind eegit           # official uses the same head on BOTH sides
  --head-drop 0.5
  --d-embed 1024                  # the paper's 768 -> 1024 FC

  # ---- the target: the documented deviation, see the header ----------------
  --target-features "${FEAT}"
  --target-layer block26
  --target-fusion single

  # ---- item 3: the official loss -------------------------------------------
  --fixed-temp                    # logit_scale is not in the optimizer's groups
  --softplus                      # softplus(log(1/0.07)) = 2.727, not 14.29
  --no-eeg-l2norm                 # only the image side is normalised

  # ---- item 4: the official optimizer and schedule -------------------------
  --optimizer adam
  --wd-all-params                 # Adam's weight_decay is an L2 penalty in-grad
  --lr 5e-5                       # official 5e-6 * 10 for encoder and heads
  --backbone-lr-mult 1.0          # official has ONE lr for the whole encoder
  --batch-size 256                # official train_batch_size
  --epochs 100
  --patience 0                    # official runs the full 100 epochs and tests last
  --aug full
  --stage1-epochs 0               # MMD: absent from the official code
  # Deliberately absent: --warmup-epochs (default 0), --cosine (default off),
  # --ema-decay (default 0 = off), --min-lr-ratio (meaningless without --cosine).
  # The official schedule is flat, and a default that happens to agree with the
  # official value is still being relied on -- so it is named here in a comment.
)

# =============================================================================
# [1] dry run
# =============================================================================
if [[ "${1:-}" == "--dry-run" ]]; then
  log "validating the official config (no GPU, no dataset)"
  "${PY}" -u "${ROOT}/scripts/epd/train.py" "${cfg[@]}" --validate-only \
    || die "flag validation failed"
  for f in "${ROOT}/scripts/epd/train.py" \
           "${ROOT}/scripts/epd/export_conds.py" \
           "${NB_ROOT}/scripts/nda/generate_ip_txt2img.py" \
           "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py"; do
    [[ -f "$f" ]] || die "pipeline references a script that does not exist: $f"
  done
  [[ -d "${FEAT}/train" && -d "${FEAT}/test" ]] || die "missing target features under ${FEAT}"
  log "dry run ok"
  exit 0
fi

mkdir -p "${OUT}" "${GEN}" "${MET}" "${ROOT}/outputs/logs"

# =============================================================================
# [2] train
# =============================================================================
if [[ -f "${OUT}/${TAG}_result.json" && -f "${CKPT}" ]]; then
  log "[2] training already complete, skipped"
else
  # The result json is the completion marker, not the checkpoint: `train.py` writes
  # `{tag}_best.pt` from inside the epoch loop (epoch 1 always beats the -inf
  # baseline, so epoch 1 always writes one) and writes the json only at the end of
  # `main`. Guarding on the checkpoint alone turns a job killed mid-training into
  # "training skipped" followed by a fatal missing-json -- which is exactly how this
  # project's first `epd_sem_eegit` submission failed after four seconds.
  if [[ -f "${CKPT}" ]]; then
    log "[2] a checkpoint exists but there is no result json: a previous run was"
    log "    killed mid-training. Retraining from scratch and overwriting it."
  fi
  log "[2] training the official config -> ${CKPT}"
  "${PY}" -u "${ROOT}/scripts/epd/train.py" "${cfg[@]}" || die "training failed"
  [[ -f "${CKPT}" ]] || die "no checkpoint at ${CKPT} after training"
fi
[[ -f "${OUT}/${TAG}_result.json" ]] || die "no result json at ${OUT}/${TAG}_result.json"

# =============================================================================
# [3] export the IP-Adapter conditions (semantic only)
# =============================================================================
# `export_conds.py` keys its structural half off `cfg["struct_backbone"]`, which is
# absent for this checkpoint, so it exports the IP conditions and skips the rest --
# and cross-checks args against weights in both directions first, so a stale config
# cannot silently decide the architecture.
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
# still leaves valid .npy files. Checking the report is what separates "the export
# finished" from "the export wrote its conditions and then broke".
[[ -f "${EXP}/export_report.json" ]] || die "no export_report.json at ${EXP}"
"${PY}" - <<PYEOF || die "export_report.json is not valid JSON"
import json, pathlib
json.loads(pathlib.Path("${EXP}/export_report.json").read_text())
PYEOF

# =============================================================================
# [4] generate -- pure IP-Adapter, no ControlNet, no depth map
# =============================================================================
# `generate_ip_txt2img.py` imports `StableDiffusionXLPipeline` only. It takes no
# `--cond-dir` because there is no condition to take, which is what makes this arm a
# clean read of the semantic branch: the EEG's IP embedding is the ONLY input.
gdir="${GEN}/ip/generated"
if [[ -f "${gdir}/199.png" ]]; then
  log "[4] 200 images present under ${gdir}, generation skipped"
else
  log "[4] generating 200 images (IP-Adapter only) -> ${gdir}"
  # Settings held identical to the existing semantic arms -- same generator, steps,
  # guidance, size and seed -- so the only difference between this run and those is
  # whose EEG built the condition. `--seed 42` matches `generate_struct_inject_decode`
  # in the sibling runners.
  "${PY}" -u "${NB_ROOT}/scripts/nda/generate_ip_txt2img.py" \
    --embed-npy "${IP_DEPLOY}" \
    --output-dir "${GEN}/ip" \
    --tag "${TAG}" \
    --ip-scale 1.0 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --gen-size 512 \
    --seed 42 || die "generation failed"
fi
[[ -f "${gdir}/199.png" ]] || die "expected 200 PNGs under ${gdir}"

# =============================================================================
# [5] the seven official metrics
# =============================================================================
outj="${MET}/${TAG}_seven.json"
if [[ -f "${outj}" ]]; then
  log "[5] metrics present, skipping"
else
  log "[5] seven metrics (Pearson two-way, the standard protocol)"
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
# [6] this run's line
# =============================================================================
TAG="${TAG}" OUT="${OUT}" MET="${MET}" "${PY}" - <<'PY'
import json
import os
from pathlib import Path

OUT = Path(os.environ["OUT"])
TAG = os.environ["TAG"]
res = json.loads((OUT / f"{TAG}_result.json").read_text(encoding="utf-8"))
met = json.loads(Path(os.environ["MET"], f"{TAG}_seven.json").read_text(encoding="utf-8"))
test = res["test"]
bv = res.get("best_val", {})
print(f"== {TAG} ==")
print(f"  retrieval test Top-1 {test['top1']:.2f} (top5 {test['top5']:.2f}, "
      f"mean_rank {test['mean_rank']:.2f}), selected at epoch {bv.get('epoch')} "
      f"(val Top-1 {bv.get('top1', float('nan')):.2f}, "
      f"from {'EMA' if bv.get('is_ema') else 'raw'} weights)")
pc = test.get("per_concept")
if pc is None:
    raise SystemExit("[FATAL] no `test.per_concept`: cross-arm paired comparison "
                     "cannot be computed from this run")
print(f"  per-concept vector: {len(pc['top1'])} concepts, "
      f"mean {100.0 * sum(pc['top1']) / len(pc['top1']):.2f} (must equal the above)")
# The temperature this run actually used. Official fixes it, so this should read
# exactly 2.727 with no drift -- and printing it is how "the loss was EEGiT's" stops
# being an assertion about flags and becomes a measurement.
crit = res.get("criterion") or {}
if crit.get("selected_effective_scale") is None:
    print("  criterion: not recorded (pre-instrumentation checkpoint)")
else:
    print(f"  criterion: softplus={crit.get('softplus')} "
          f"init scale {crit.get('init_effective_scale'):.3f} -> "
          f"selected {crit['selected_effective_scale']:.3f} "
          f"(official expects 2.727, fixed)")
print(f"  PixCorr {met['pixcorr']:.3f}  SSIM {met['ssim']:.3f}  "
      f"AlexNet(2) {met['alex2']:.3f}  AlexNet(5) {met['alex5']:.3f}  "
      f"Inception {met['inception']:.3f}  CLIP {met['clip']:.3f}  "
      f"SwAV {met['swav']:.3f}  FID {met['fid']:.1f}")
hist = res.get("history", [])
if hist:
    bi = max(range(len(hist)), key=lambda i: hist[i]["sel"])
    print(f"  curve: sel peaks at epoch {hist[bi]['epoch']}/{res.get('epochs_run')} "
          f"(sel {hist[bi]['sel']:.2f}, val_top1 {hist[bi]['val_top1']:.2f}), "
          f"final epoch sel {hist[-1]['sel']:.2f}")
    # A chance-level plateau is the signature of a logit scale that is too sharp, and
    # it is worth seeing in the output rather than only in the log: this project has
    # now paid for that diagnosis once.
    flat = [h for h in hist[:20] if h["val_top1"] < 2.0]
    if len(flat) >= 8:
        print(f"  WARNING: {len(flat)} of the first 20 epochs are below 2% val Top-1 "
              f"(chance is 0.67%). Check the effective logit scale above.")
print(f"  train {res.get('timing', {}).get('wall_seconds', float('nan')):.0f}s")
PY

log "[done] $(date -Iseconds)"
