"""Turn a trained dual-tower checkpoint into the three conditions SDXL needs.

The generation stack this feeds takes three independent inputs per test image:

  1. an IP-Adapter embedding -- a 1024-d vector in CLIP ViT-H-14's projected joint
     space, carrying what the picture is of          <- semantic tower
  2. an img2img init RGB image, carrying the coarse colour/luminance layout
                                                    <- structure tower, VAE head

There is no third condition. The ControlNet-depth input was removed together with
the depth head: `probe_targets.py` measured depth's best linear read-out from EEG at
r = +0.16 against +0.53 for the constant-map predictor, on four independent arms
(margin -0.37 each), so the head was regressing a target the input does not carry.
The generation pipeline still accepts a depth condition and is still run with
`--control-type depth --cn-scale 0`, i.e. multiplied out before it reaches the
residual; it is pointed at the decoded init directory so the loader has readable
PNGs. That is a path which contributes nothing, kept so the generator's signature
does not have to change.

Why the IP condition is not simply the semantic tower's output
-------------------------------------------------------------
The semantic tower's embedding is `d_embed=512`-d, because 512 is the width of the
space both sides of InfoNCE are projected into. That space is *aligned* but it is
not CLIP's space: the map from CLIP to it is `img_head`, a learned linear layer, and
the map back is not unique. Feeding a 512-d vector to IP-Adapter is not possible.

So the condition is built the way a memory bank is used: the EEG query is matched
against the gallery of *training* concept embeddings **inside the model's own
512-d space**, where the geometry was trained, and the resulting weights are then
used to mix **real CLIP joint embeddings** of those same training concepts.

Two arrays, because they live in different spaces and neither can stand in for the
other:

  keys
    `(1654, 10, D_key)` features of the run's own alignment target. The semantic
    tower was aligned to a specific layer of the image tower, and that layer's width
    is what `img_head` reads -- 1280-d for CLIP ViT-H-14 block26, 1024-d for the
    final `visual.proj` output. Using the wrong width does not fail at load time
    (`img_head` is built *from* the gallery), it fails at `load_state_dict`.
  atoms
    `(1654, 10, 1024)` CLIP joint-space embeddings, i.e. the training images as CLIP
    actually represents them. This is what `ip-adapter_sdxl_vit-h.bin` was trained
    against, so it is what the condition must be made of.

Retrieval belongs where the metric is meaningful (`keys`); the output belongs where
the consumer expects it (`atoms`). The two are indexed identically -- 1654 training
concepts, same order -- so the retrieved weights select the right atoms.

Three arms, and why all three are exported
------------------------------------------
  deploy : soft retrieval over the training gallery, then a weighted sum of the
           atoms. The intended configuration.
  raw    : `pinv(img_head) @ z`, i.e. the model's embedding read back with no
           gallery involved. Isolates how much the memory step -- a non-parametric,
           training-free component -- is actually contributing. Without this arm a
           good deploy number cannot be attributed to the learned encoder rather
           than to the bank of real CLIP embeddings. Only well posed when the
           alignment target IS the condition space; the arm refuses otherwise rather
           than emitting a wrong-width array.
  noise  : the EEG input is replaced by zeros and pushed through the identical
           code path. The control for "is any of this driven by the EEG at all".
           A pipeline that scores the same on `noise` as on `deploy` has
           demonstrated that its generation quality comes from the prior.

The gallery is the 1654 *training* concepts, which are disjoint from the 200 test
concepts, so the retrieval step cannot retrieve the answer.

Note on the calibration step that follows: `gem_calib.py` in NeuroBridge
quantile-matches the concentration of a condition onto the training bank's
distribution. That correction is computed *after* this script and is not optional
for the deployed arm -- see the pipeline script.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd import config
from epd.data import load_subject
from epd.encoders import EEGiTProjectionHead
from epd.model import build_from_args

# Width of the CLIP joint-space embedding `ip-adapter_sdxl_vit-h.bin` consumes when
# it is loaded without its own image encoder. Not a tunable: it is a property of the
# checkpoint, and every condition this project has produced is this wide.
IP_ADAPTER_WIDTH = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="path to <tag>_best.pt")
    p.add_argument("--out-dir", required=True,
                   help="everything is written under <out-dir>/{conds,spatial}")
    p.add_argument("--tag", required=True)

    g = p.add_argument_group("semantic condition")
    g.add_argument("--gallery", type=str, default=None,
                   help="(n_concepts, n_img, D) features for the TRAIN concepts, in "
                        "the space `img_head` consumes -- i.e. the run's own "
                        "alignment target. Defaults to the cached features for that "
                        "target; used ONLY to build the retrieval keys, so its width "
                        "follows the checkpoint, not IP-Adapter.")
    g.add_argument("--atoms", type=str, default=None,
                   help="(n_concepts, n_img, 1024) real CLIP joint-space embeddings "
                        "for the same TRAIN concepts, one row per gallery entry. The "
                        "retrieved weights are used to mix THESE, because the "
                        "condition handed to IP-Adapter has to be in the space the "
                        "IP-Adapter was trained against. Defaults to the shipped "
                        "ViT-H-14 image_train.npy.")
    g.add_argument("--gallery-temp", type=float, default=20.0,
                   help="softmax temperature on cosine similarity, i.e. beta in "
                        "softmax(beta * s). Higher = harder assignment.")
    g.add_argument("--gallery-topk", type=int, default=0,
                   help="keep only the k nearest gallery entries (0 = all). Acts as "
                        "a second hardness knob and bounds how much of the condition "
                        "is an average of unrelated CLIP embeddings.")
    g.add_argument("--arms", type=str, nargs="+",
                   default=["deploy", "raw", "noise"],
                   choices=["deploy", "raw", "noise"])

    s = p.add_argument_group("structural conditions")
    s.add_argument("--decode-rgb", action="store_true",
                   help="decode the predicted VAE latents into 512x512 PNGs via the "
                        "SDXL VAE. Needs the VAE weights in the HF cache.")
    s.add_argument("--depth-dev-gain", type=float, default=1.0,
                   help="scale the INSTANCE-specific part of the depth prediction "
                        "before rendering it as a ControlNet conditioning image, "
                        "leaving the shared fit-set mean field at weight 1. The two "
                        "parts are genuinely different things: the mean field is the "
                        "same map for every sample, so raising it would change nothing "
                        "about which image is produced, while raising the deviation "
                        "changes exactly that. At 1.0 the image is the model's actual "
                        "prediction, which is the honest arm; above 1.0 it is a "
                        "sharpened condition, which is a legitimate control knob "
                        "because ControlNet consumes the map as a soft spatial "
                        "residual rather than as a target to reproduce.")
    s.add_argument("--device", default="cuda")
    s.add_argument("--allow-collapse", action="store_true",
                   help="ship a structural condition even if its head fails the "
                        "across-sample collapse gate. Default is to refuse. A "
                        "collapsed head is not a loud failure: it writes a "
                        "plausible-looking condition, decodes to a plausible-looking "
                        "image, and completes the whole generation and metric run, so "
                        "the check has to fire here -- the last point where the ground "
                        "truth is still on disk to compare against -- rather than be "
                        "reconstructed later from a metric table that cannot tell a "
                        "structured init from a blurry constant.")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def build_depth_condition(
    pred: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    rng,
    field: np.ndarray | None,
    gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a standardised depth prediction into (physical map, 0-1 conditioning map).

    The four operations, and what each one has to be paired with:

      1. `pred * std + mean` -- undo the training normalisation. `AuxTargetDataset`
         applies `(v - vae_mean) / vae_std` with `vae_mean` reshaped to (C,1,1), so
         the inverse has to use the same reshape. Doing it with a scalar would be
         right for channel 0 only.
      2. `+ field` -- add back the fit-set per-pixel mean field that
         `--struct-center` removed. This is a CONSTANT across samples by
         construction, so it cannot be recovered from the prediction and must come
         from the checkpoint args.
      3. `+ (gain - 1) * dev` where `dev` is the step-1 result. Note what `dev` is:
         the prediction in the target's own units BEFORE the mean field is added.
         So the gain multiplies the instance-specific part and leaves the shared
         mean field at exactly 1. Applying the gain to `full` instead would scale
         both, which is a different operation with a different meaning -- the mean
         field is the same map for every sample, so scaling it changes nothing about
         which image is produced while scaling `dev` changes exactly that.
      4. quantise with a FIXED range (the fit split's 0.5-99.5 percentiles), not a
         per-image min-max. The target cache was written per-image normalised, which
         is right for a target -- it stops the loss being owned by whichever scene
         had the largest depth extent -- but applying it to the prediction would
         divide by a quantity that is mostly noise (the reachable per-concept signal
         is r = +0.22) and produce full-contrast speckle. A shared range keeps the
         true contrast, so the decoder sees what the model actually predicted.

    Returned separately because the physical map is what the collapse gate and the
    `.npy` artifact need, and the 0-1 map is what the PNGs need; deriving one from
    the other downstream is how they drift.
    """
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    dev = pred * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
    full = dev
    if field is not None:
        f = np.asarray(field, dtype=np.float32)
        if f.shape != dev.shape[1:]:
            raise SystemExit(f"stored mean field has shape {f.shape}, expected "
                             f"{dev.shape[1:]}")
        full = dev + f.reshape(1, *f.shape)
    if float(gain) != 1.0:
        full = full + (float(gain) - 1.0) * dev
    lo, hi = float(rng[0]), float(rng[1])
    if not hi > lo:
        raise SystemExit(f"display range [{lo}, {hi}] is not increasing")
    u8 = np.clip((full - lo) / (hi - lo), 0.0, 1.0)
    return full, u8


def field_mean(cfg: dict) -> np.ndarray | None:
    """The centred target's mean field, in the target's own (C, H, W) shape.

    The checkpoint stores this as a FLAT list plus its shape, because a saved
    `args` has to be plain JSON: `train.py` writes `field.reshape(-1).tolist()`
    alongside `_struct_field_mean_shape`. THREE consumers here read it and, before
    this function existed, only two applied the shape -- and one of those did it
    with a different expression. The depth export therefore handed a (64,) vector
    to `build_depth_condition`, which refused it against a (1, 8, 8) prediction and
    aborted the pipeline with

        stored mean field has shape (64,), expected (1, 8, 8)
        [FATAL] export failed

    at the export step of a run whose 100 epochs of training had completed and
    whose checkpoint was fine. Reshaping in ONE place is what keeps the next
    consumer from repeating it; the alternative -- each caller remembering to pair
    the flat array with `_struct_field_mean_shape` -- is the arrangement that
    already failed once.
    """
    flat = cfg.get("_struct_field_mean")
    if flat is None:
        return None
    shape = cfg.get("_struct_field_mean_shape")
    if not shape:
        raise SystemExit(
            "the checkpoint stores _struct_field_mean without "
            "_struct_field_mean_shape, so the flat mean field cannot be aligned to "
            "the prediction's (C, H, W). This checkpoint predates the shape being "
            "recorded and cannot be exported; retrain it.")
    return np.asarray(flat, dtype=np.float32).reshape(tuple(shape))


def collapse_gate(pred: np.ndarray, gt: np.ndarray, name: str,
                  var_floor: float = 0.20, margin_floor: float = 0.02) -> dict:
    """Refuse to ship a structural head that predicts the conditional mean.

    Why the check that this replaces could not see the failure
    ----------------------------------------------------------
    The deployed structural tower was verified by `per_sample_range`: the dynamic
    range WITHIN each predicted map. It scored 0.723 and read as healthy. The head
    had in fact collapsed to the training-set mean map, which has plenty of
    within-sample structure, so the statistic was measuring the target's own
    contrast and reporting it as the model's. The collapse is in the ACROSS-sample
    direction and the diagnostic has to live there:

        across-sample var / total var   0.0068      (the target's own ratio: 0.63)
        mean pairwise cosine            0.9991
        r(pred_i, GT_i)                +0.6485      r(constant, GT_i): +0.6540

    Three numbers, each against an explicit reference so that a raw value cannot be
    read as healthy in isolation:
      * variance ratio, next to the target's own -- how much of the variation that
        exists is reproduced at all;
      * r(pred_i, GT_i), next to r(constant, GT_i) -- the margin over a predictor
        that ignores the EEG entirely. The constant is the right reference because
        it is exactly what L1 returns when the target is not in the input, which is
        the hypothesis under test;
      * the margin, which is the only one of the three that a high correlation
        cannot fake. A constant map correlates strongly with every real map; that is
        what "floor" means here.

    Clearing these is necessary and not sufficient: they establish that the head is
    not the mean, not that what it predicts is the image.
    """
    p = pred.reshape(pred.shape[0], -1).astype(np.float64)
    g = gt.reshape(gt.shape[0], -1).astype(np.float64)
    if p.shape != g.shape:
        raise SystemExit(f"[{name}] collapse gate cannot compare predictions "
                         f"{p.shape} against ground truth {g.shape}")

    def var_ratio(x: np.ndarray) -> float:
        return float(x.var(axis=0).mean() / (x.var() + 1e-12))

    def rows_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        num = (a * b).sum(axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
        return num / den

    pr, gr = var_ratio(p), var_ratio(g)
    r_pred = float(rows_r(p, g).mean())
    mu = g.mean(axis=0, keepdims=True)
    r_const = float(rows_r(np.broadcast_to(mu, g.shape), g).mean())
    margin = r_pred - r_const
    ok = (pr >= var_floor) and (margin >= margin_floor)
    return {"pred_var_ratio": pr, "target_var_ratio": gr,
            "var_ratio_floor": var_floor,
            "r_pred_to_gt": r_pred, "r_constant_to_gt": r_const,
            "margin_over_constant": margin, "margin_floor": margin_floor,
            "passed": bool(ok)}


def resolve_gallery(cfg: dict, override: str | None) -> Path:
    """Where the TRAIN-concept gallery lives, in the space `img_head` consumes.

    The gallery has to be the same representation the EEG was aligned to, and the
    two are not interchangeable: the alignment target can be any layer of the image
    tower (`--target-features`/`--target-layer`), and those layers do not share a
    width. `image_train.npy` is the 1024-d output of CLIP ViT-H-14's `visual.proj`,
    while block26 of the same tower is 1280-d. Loading the wrong one does not fail
    at load time -- `img_head` is built *from* the gallery's width -- it fails at
    `load_state_dict`, which is to say after training.
    """
    if override:
        p = Path(override)
        if not p.is_file():
            raise SystemExit(f"--gallery {p} does not exist")
        return p

    # Single-target runs record `target_layer` as a bare string and leave
    # `target_layers` at None (the list form is only built in the result file, not
    # in the args that travel with the checkpoint), so both have to be read.
    tdir = cfg.get("target_features")
    tlayers = cfg.get("target_layers") or (
        [cfg["target_layer"]] if cfg.get("target_layer") else None)
    if tdir and tlayers:
        if len(tlayers) > 1:
            raise SystemExit(
                f"this run blended {len(tlayers)} target layers {tlayers} "
                f"(target_fusion={cfg.get('target_fusion')}); the gallery would have "
                f"to be blended the same way. Export the blended train features "
                f"yourself and pass them with --gallery.")
        p = Path(tdir) / "train" / f"{tlayers[0]}.npy"
        if not p.is_file():
            raise SystemExit(f"checkpoint aligned to {tlayers[0]} via {tdir}, but "
                             f"{p} does not exist; pass --gallery explicitly")
        return p

    return config.IMAGE_FEATURE_DIR / "image_train.npy"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    conds = out / "conds"
    spatial = out / "spatial"
    conds.mkdir(parents=True, exist_ok=True)
    spatial.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["args"]
    print(f"[load] {args.ckpt} (epoch {ckpt.get('epoch')}, tag {cfg.get('tag')})")

    channels = None if cfg.get("channels", "all") == "all" else config.CHANNELS_OCCIPITO_PARIETAL
    ch_names = (list(channels) if channels
                else json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"])
    _, te_eeg = load_subject(cfg["subject"], channels)
    if args.limit:
        te_eeg = te_eeg[:args.limit]
    n = te_eeg.shape[0]
    print(f"[data ] test EEG {te_eeg.shape} ({n} concepts, {len(ch_names)} channels)")

    # The width of `img_head`'s input, read off the checkpoint's own weights. This
    # is the authoritative source and it is always present: `train.py` only started
    # recording `_image_dim` in args after the fact, so older checkpoints have to be
    # handled too, and the input layer's `shape[1]` is exactly the number training
    # built it at. Deriving it any other way risks the one failure that costs a full
    # training run -- `img_head` is built *from* the gallery width, so a wrong
    # gallery does not fail at load time, it fails at `load_state_dict`.
    #
    # Three head kinds exist and the input layer sits at a different key in each:
    #   `nw` + --img-projector linear   nn.Linear          -> `img_head.weight`
    #   `nw` (default)                  nn.Sequential      -> `img_head.0.weight`
    #   `eegit`                         EEGiTProjectionHead-> `img_head.projection.weight`
    # Looking up only `img_head.weight` therefore works for the 1024-d single-Linear
    # case and crashes for the official head -- which was the intended head for the
    # EEGiT-consistent run, so this would have failed after training, at export.
    # The explicit candidates are tried first (unambiguous, and they name the
    # architecture); the fallback picks the 2-D `img_head.*` tensor with the widest
    # input, which is the input projection in every configuration above, because all
    # downstream widths are `d_embed` <= `image_dim`.
    def _head_input_width(sd: dict) -> tuple[int, str]:
        for key in ("img_head.weight", "img_head.0.weight",
                    "img_head.projection.weight"):
            w = sd.get(key)
            if w is not None and w.ndim == 2:
                return int(w.shape[1]), key
        cands = {k: v for k, v in sd.items()
                 if k.startswith("img_head.") and v.ndim == 2}
        if not cands:
            raise SystemExit(
                "checkpoint has no 2-D `img_head.*` weight; cannot determine the "
                "width the semantic embedding was trained at. Keys present: "
                f"{sorted(k for k in sd if k.startswith('img_head.'))}")
        key = max(cands, key=lambda k: cands[k].shape[1])
        return int(cands[key].shape[1]), key

    model_sd = ckpt["model"]
    image_dim, head_key = _head_input_width(model_sd)
    print(f"[head ] img_head input width {image_dim} read from `{head_key}`")
    recorded = cfg.get("_image_dim")
    if recorded is not None and int(recorded) != image_dim:
        raise SystemExit(f"checkpoint args say _image_dim={int(recorded)} but "
                         f"`{head_key}` reads {image_dim}-d")

    gpath = resolve_gallery(cfg, args.gallery)
    gall = np.load(gpath)
    if int(gall.shape[-1]) != image_dim:
        raise SystemExit(
            f"the checkpoint's img_head reads {image_dim}-d (its alignment target) "
            f"but --gallery {gpath} holds {int(gall.shape[-1])}-d features. This is "
            f"the run's alignment target vs the shipped visual.proj output -- "
            f"different layers of the same image tower, different widths. Pass the "
            f"gallery the run actually trained against (target_features/target_layer "
            f"in its args) with --gallery.")
    print(f"[gall ] keys {gpath} {tuple(gall.shape)} dtype {gall.dtype}")

    # The output atoms: real CLIP joint-space embeddings for the same train
    # concepts, in the order the keys are in. Deliberately independent of the key
    # file -- see the note where the two are used.
    apath = Path(args.atoms) if args.atoms else (config.IMAGE_FEATURE_DIR / "image_train.npy")
    if not apath.is_file():
        raise SystemExit(f"--atoms {apath} does not exist")
    atoms = np.load(apath)
    if atoms.shape[0] != gall.shape[0]:
        raise SystemExit(
            f"--atoms {apath} has {atoms.shape[0]} concepts but the keys have "
            f"{gall.shape[0]}; the retrieval weights index both, so they must be the "
            f"same concepts in the same order.")
    g_atoms_raw = torch.from_numpy(atoms.astype(np.float32)).mean(dim=1)   # (G, 1024)
    if int(g_atoms_raw.shape[1]) != IP_ADAPTER_WIDTH:
        raise SystemExit(
            f"--atoms {apath} is {int(g_atoms_raw.shape[1])}-d, but "
            f"ip-adapter_sdxl_vit-h (loaded without an image encoder) consumes "
            f"{IP_ADAPTER_WIDTH}-d CLIP joint embeddings. Every other condition in "
            f"this project is {IP_ADAPTER_WIDTH}-d for that reason.")
    print(f"[gall ] atoms {apath} {tuple(atoms.shape)} dtype {atoms.dtype}")

    model = build_from_args(cfg, ch_names, image_dim).to(device).eval()
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint does not match this model: missing={list(missing)[:6]} "
                         f"unexpected={list(unexpected)[:6]}")
    # A checkpoint with no structure tower is a SUPPORTED configuration, not an
    # error: the architecture the measurements select is semantic-only (a
    # ground-truth depth map through the ControlNet costs quality at CN 0.70 and is
    # inert at CN 0.35, so the branch is removed rather than improved). It still
    # needs its IP-Adapter conditions exported -- those are the semantic condition
    # and the arm cannot be generated without them. What is skipped is the
    # structural half of this script, which is keyed off `have_struct` below.
    have_struct = bool(cfg.get("struct_backbone"))
    if model.struct is None and have_struct:
        raise SystemExit("checkpoint args claim a structure tower but the built model "
                         "has none; args and weights disagree")
    if model.struct is not None and not have_struct:
        raise SystemExit("checkpoint has a structure tower but its args do not; "
                         "refusing to guess whether to export a structural condition")
    if not have_struct:
        print("[struct] no structure tower in this checkpoint: exporting the "
              "IP-Adapter conditions only (semantic-only architecture)")
    # The z-score buffers came back with the state dict; re-checking here because a
    # model would still run with stats_set False and silently z-score by 0/1.
    if not bool(model.encoder.tokenizer.stats_set):
        raise SystemExit("checkpoint does not carry the tokenizer's z-score statistics")

    # ---------------- forward the test set once
    zs, vps = [], []
    # Keyed on the presence of a structure tower, NOT on `vae_latents`. The latter
    # names the fine VAE cache's directory, so it is empty on a depth run and this
    # collected nothing -- which would have made the depth export below receive
    # `latent = None` and quietly report no structural condition for a checkpoint
    # that has one.
    with torch.no_grad():
        for i in range(0, n, 128):
            x = torch.from_numpy(np.ascontiguousarray(te_eeg[i:i + 128, 0])).to(device)
            # Named `fwd`, NOT `out`: `out` is the output Path bound at the top of
            # `main()` and is used again at the very end to write
            # `export_report.json`. Binding the forward result to `out` shadowed the
            # Path (no NameError, no warning) and made the final write raise
            # `TypeError: unsupported operand type(s) for /: 'dict' and 'str'` --
            # after every condition had already been written, so the run looked
            # successful in the log right up to the traceback and left the pipeline
            # without the report it reads back.
            fwd = model.forward_all(x, None, training=False)
            zs.append(F.normalize(model.encode_eeg(x, None, training=False)[0].float(), dim=-1).cpu())
            if have_struct:
                # `fwd["struct"]` is only present when a tower was built, so this
                # read has to stay inside the guard rather than be guarded after.
                vps.append(fwd["struct"]["vae"].float().cpu())
    z = torch.cat(zs)
    latent = torch.cat(vps) if vps else None
    print(f"[fwd  ] semantic z {tuple(z.shape)}"
          + (f", structural field {tuple(latent.shape)}" if latent is not None else ""))

    # ---------------- semantic: two different spaces, on purpose
    #
    # The retrieval keys must live in the space the EEG was TRAINED to match, and
    # the output atoms must live in the space IP-Adapter CONSUMES. Those are not the
    # same space, and one array cannot serve both:
    #   keys  -- block26 of CLIP ViT-H-14, 1280-d, because that is the layer the
    #            semantic tower was aligned to and `img_head` was built for
    #   atoms -- the 1024-d output of `visual.proj`, which is what
    #            `ip-adapter_sdxl_vit-h.bin` (loaded with image_encoder_folder=None)
    #            was trained against, and what every other condition in this project
    #            is (`mem_decode_a50.npy`, `blend_nda_cfm_f_a40_test.npy`: 200x1024)
    # Both are indexed identically, so the retrieved weights mix the right atoms.
    # That is guaranteed rather than assumed: `extract_layers.py` verifies its
    # per-layer arrays row-by-row against the shipped features before writing them
    # (manifest.json -> verify_vs_shipped: 16540 rows, min cosine 0.998), and it
    # refuses to write a split it cannot align.
    #
    # The gallery centroid is taken AFTER projecting each of the 10 images per
    # concept into the shared space, not before. `encode_image` ends in a projection,
    # so the mean of 10 projections is a different vector from the projection of the
    # mean -- and only the former is the centroid of where those images actually
    # land, which is the quantity the EEG query is being matched against. Training
    # and evaluation both score one image slot at a time, so no averaging happens
    # there; this is the one place a concept centroid is needed at all.
    n_img_per_conc = int(gall.shape[1])
    flat = torch.from_numpy(gall.astype(np.float32)).reshape(-1, int(gall.shape[-1]))
    with torch.no_grad():
        gz_flat = F.normalize(model.encode_image(flat.to(device)).float(), dim=-1).cpu()
    g_z = F.normalize(gz_flat.reshape(int(gall.shape[0]), n_img_per_conc, -1).mean(dim=1),
                      dim=-1)
    g_atoms = F.normalize(g_atoms_raw, dim=-1)                          # (G, 1024)
    print(f"[gall ] retrieval keys {tuple(gall.shape)} -> {tuple(g_z.shape)} in the "
          f"shared space ({n_img_per_conc} image slots averaged after projection); "
          f"output atoms {tuple(g_atoms.shape)}")

    sim = z @ g_z.t()                                                  # (n, G)
    if args.gallery_topk and args.gallery_topk < sim.shape[1]:
        thr = sim.topk(args.gallery_topk, dim=1).values[:, -1:]
        sim = sim.masked_fill(sim < thr, float("-inf"))
    w = torch.softmax(args.gallery_temp * sim, dim=1)                  # (n, G)
    # Normalise AFTER mixing, not before: the atoms are unit vectors, so a softmax
    # mixture of them has norm < 1 whenever the weights are spread out. Re-normalising
    # puts the condition back on the unit sphere IP-Adapter's embeddings live on and
    # leaves the direction -- which is all the adapter uses -- untouched.
    ip_mem = F.normalize(w @ g_atoms, dim=-1)
    print(f"[cond ] retrieval: mean top-1 weight {float(w.max(dim=1).values.mean()):.3f}, "
          f"mean entropy {float((-(w.clamp_min(1e-9).log() * w).sum(1)).mean()):.3f} nats "
          f"(max {float(np.log(w.shape[1])):.3f})")

    # `raw` arm: read the shared embedding back through the head that put it there.
    # lstsq/pinv gives the minimum-norm pre-image, which is the only canonical
    # choice when the map is many-to-one.
    #
    # This one is only well posed when the training target IS the condition space.
    # `pinv(img_head)` lands in the space of whatever the EEG was aligned to, so for
    # a block26 run it returns a 1280-d vector that IP-Adapter cannot consume at all;
    # there is no honest way to bridge the two, because arriving at `visual.proj`'s
    # output from an intermediate residual stream would require the tower's own
    # `ln_post` and `proj`, which were fitted to the final layer's statistics and not
    # to block26's. So the arm refuses rather than emitting a wrong-width array.
    #
    # Built only when asked for. Computing it up front and relying on the write loop
    # to skip it means the refusal above fires on every run, including the ones that
    # never wanted this arm.
    ip_raw = None
    if "raw" in args.arms:
        head = model.img_head
        # `raw` needs a single affine map so that `pinv` is meaningful. The EEGiT
        # head is `LayerNorm(fc(gelu(P(x))) + P(x))` -- non-linear, with a LayerNorm
        # on top -- so it is not merely a different key to read, the arm is not
        # well posed for it. Named explicitly rather than lumped into the generic
        # message so the refusal says which head kind was found.
        if isinstance(head, EEGiTProjectionHead):
            raise SystemExit(
                "the raw arm needs an affine img_head; this checkpoint uses the "
                "EEGiT ProjectionHead (Linear -> GELU -> Linear -> residual -> "
                "LayerNorm), which is not affine. Use the `deploy` or `noise` arms.")
        if not hasattr(head, "weight"):
            raise SystemExit(f"the raw arm needs an affine img_head, got "
                             f"{type(head).__name__}")
        W = head.weight.detach().float().cpu()                          # (d_embed, D_key)
        if int(W.shape[1]) != int(g_atoms.shape[1]):
            raise SystemExit(
                f"the raw arm is not available for this checkpoint: img_head reads "
                f"{int(W.shape[1])}-d (the alignment target) while IP-Adapter "
                f"consumes {int(g_atoms.shape[1])}-d. `pinv(img_head)` returns a "
                f"vector in the target's space, which is not a CLIP joint embedding. "
                f"Use the `deploy` or `noise` arms.")
        b = (head.bias.detach().float().cpu() if head.bias is not None
             else torch.zeros(W.shape[0]))
        Wp = torch.linalg.pinv(W)                                       # (D_key, d_embed)
        ip_raw = F.normalize((z - b) @ Wp.t(), dim=-1)

    # `noise` arm: identical path, EEG replaced by zeros.
    ip_noise = None
    if "noise" in args.arms:
        with torch.no_grad():
            zn = []
            for i in range(0, n, 128):
                x = torch.zeros_like(torch.from_numpy(
                    np.ascontiguousarray(te_eeg[i:i + 128, 0]))).to(device)
                zn.append(F.normalize(
                    model.encode_eeg(x, None, training=False)[0].float(), dim=-1).cpu())
            zn = torch.cat(zn)
        w_n = torch.softmax(args.gallery_temp * (zn @ g_z.t()), dim=1)
        ip_noise = F.normalize(w_n @ g_atoms, dim=-1)

    written = []
    g_atoms_np = g_atoms.numpy()
    c_ref = g_atoms_np.mean(0, keepdims=True)
    for arm, ip in (("deploy", ip_mem), ("raw", ip_raw), ("noise", ip_noise)):
        if ip is None:
            continue
        f = conds / f"ip_{arm}_test.npy"
        arr = ip.numpy().astype(np.float32)
        np.save(f, arr)
        written.append(str(f))
        # The concentration of the condition relative to the training bank is the
        # quantity gem_calib corrects, so it is measured here rather than assumed.
        c_self = float(np.mean(np.sum(arr * c_ref, axis=1)))
        print(f"[cond ] {arm:7s} -> {f.name}  width {arr.shape[1]}  "
              f"mean c_self {c_self:+.4f}")

    # ---------------- structural: decode the latents
    report = {"tag": args.tag, "n": int(n),
              "arms": {a: float(np.mean(np.sum(ip.numpy() * c_ref, axis=1)))
                       for a, ip in (("deploy", ip_mem), ("raw", ip_raw),
                                     ("noise", ip_noise)) if ip is not None},
              # How peaked the retrieval actually was. Uniform weights (entropy ==
              # log G) mean the condition degenerates to the mean training embedding
              # and the EEG query contributed nothing to which atoms were mixed --
              # which is the difference between "retrieval works" and "we shipped
              # the gallery centroid".
              "retrieval": {"mean_top1_weight": float(w.max(dim=1).values.mean()),
                            "mean_entropy_nats": float(
                                (-(w.clamp_min(1e-9).log() * w).sum(1)).mean()),
                            "uniform_entropy_nats": float(np.log(w.shape[1])),
                            "gallery_temp": float(args.gallery_temp)},
              "gallery_keys": str(gpath), "gallery_size": int(g_z.shape[0]),
              "key_width": int(gall.shape[-1]), "atom_width": int(g_atoms.shape[1]),
              "atoms": str(apath),
              "gallery_topk": int(args.gallery_topk),
              "image_dim": int(image_dim),
              "conds": written}

    # The VAE-latent route. Skipped when the structural target is depth: the same
    # tensor would be written under a VAE name and scored against a ground truth that
    # does not exist, and its `--decode-rgb` output is an img2img init that the depth
    # arm does not consume at all.
    if latent is not None and str(cfg.get("struct_target", "vae")) != "depth":
        mean = np.asarray(cfg.get("_vae_mean", []), dtype=np.float32)
        std = np.asarray(cfg.get("_vae_std", []), dtype=np.float32)
        # The training normalisation is stored in the checkpoint args under the keys
        # written by train.py so the export does not have to recompute fit-split
        # statistics (which would be impossible here: the fit split is a training
        # artefact and the val holdout must not influence the test conditions).
        if mean.size != latent.shape[1] or std.size != latent.shape[1]:
            raise SystemExit("checkpoint args do not carry _vae_mean/_vae_std; the "
                             "predicted latents cannot be returned to the VAE's basis")
        lat = (latent.numpy() * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1))
        lat = lat.astype(np.float16)
        np.save(spatial / "pred_vae_test_scaled.npy", lat)
        # `.std()` on a float16 array accumulates in float16, and the sum of squares
        # over 200*4*64*64 elements overflows float16's max of 65504 -- so the
        # reported std came out as `inf` while the saved array was perfectly finite
        # (max |x| = 1.16, true std 0.2589). The array is what the decoder consumes
        # and it was never affected; this is a reporting bug, but it lands in
        # export_report.json, which is the file the run is compared on. Min/max/mean
        # are single-pass reductions with no such accumulation, so only std needs the
        # wider accumulator.
        rng = (float(lat.min()), float(lat.max()), float(lat.mean()),
               float(lat.std(dtype=np.float32)))
        report["vae"] = {"npy": str(spatial / "pred_vae_test_scaled.npy"),
                         "shape": list(lat.shape), "dtype": "float16",
                         "min": rng[0], "max": rng[1], "mean": rng[2], "std": rng[3]}
        print(f"[vae  ] predicted latents (already x scaling_factor) "
              f"shape {lat.shape} min {rng[0]:.3f} max {rng[1]:.3f} "
              f"mean {rng[2]:+.4f} std {rng[3]:.4f}")

        # ---- the check the previous run did not have -------------------------
        # Deliberately before the VAE decode: a collapsed head should fail in
        # seconds, not after 200 decodes, and the decode is the only slow thing in
        # this block.
        gtp = Path(str(cfg.get("vae_latents", ""))) / "test_vae_latents_f16.npy"
        if gtp.is_file():
            gt_raw = np.load(gtp).astype(np.float32)[:latent.shape[0]]
            if gt_raw.shape[0] != latent.shape[0]:
                raise SystemExit(f"[vae] collapse gate: ground truth has "
                                 f"{gt_raw.shape[0]} rows, predictions have "
                                 f"{latent.shape[0]}")
            # Compared in the STANDARDISED space the head was trained in, not the
            # VAE's basis. The cached ground truth is the raw latent and the two
            # lines above are the conversion back, so comparing after it would test
            # the rescaling alongside the head and could hide a collapse behind a
            # mis-stored mean/std.
            gt_std = ((gt_raw - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1))
            gate = collapse_gate(latent.numpy(), gt_std, "vae")
            report["vae"]["collapse_gate"] = gate
            print(f"[vae  ] collapse gate: across-sample var ratio "
                  f"{gate['pred_var_ratio']:.4f} (target {gate['target_var_ratio']:.4f}, "
                  f"floor {gate['var_ratio_floor']:.2f}) | r(pred,gt) "
                  f"{gate['r_pred_to_gt']:+.4f} vs constant {gate['r_constant_to_gt']:+.4f} "
                  f"-> margin {gate['margin_over_constant']:+.4f}")
            if not gate["passed"] and not args.allow_collapse:
                within = float((lat.max(axis=(1, 2, 3)) - lat.min(axis=(1, 2, 3))).mean())
                raise SystemExit(
                    f"[vae] COLLAPSED. The head reproduces {gate['pred_var_ratio']:.4f} of "
                    f"the across-sample variation (floor {gate['var_ratio_floor']:.2f}) and "
                    f"beats the constant predictor by only "
                    f"{gate['margin_over_constant']:+.4f} (floor "
                    f"{gate['margin_floor']:+.2f}).\n"
                    f"  This is the failure the previous run shipped undetected: the "
                    f"within-sample range reported here is {within:.3f}, i.e. healthy, "
                    f"because the training-set mean latent has plenty of within-sample "
                    f"structure. The head is a blurry constant.\n"
                    f"  Everything downstream consumes this condition -- including the "
                    f"EEG-null control, which is handed the SAME init -- so shipping it "
                    f"would make every arm of the comparison a measurement of the SDXL "
                    f"prior.\n"
                    f"  Pass --allow-collapse to ship it anyway, and label the run.")
        else:
            report["vae"]["collapse_gate"] = {"skipped": f"no ground truth at {gtp}"}
            print(f"[vae  ] collapse gate SKIPPED: no ground truth at {gtp}")

        if args.decode_rgb:
            sys.path.insert(0, "/project/peilab/why/NeuroBridge/scripts/nda")
            from train_eeg_vae_head import decode_latents, resolve_vae  # type: ignore
            import os
            hub = Path(os.environ.get("HF_HUB_CACHE",
                                      "/project/peilab/why/cache/eeg-brainit/hf/hub"))
            vae = resolve_vae(hub, device)
            scaling = 0.13025
            rgb_dir = spatial / "pred_lowlevel_rgb_512"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            with torch.no_grad():
                for i in range(0, n, 8):
                    blk = torch.from_numpy(lat[i:i + 8].astype(np.float32)).to(device)
                    for j, im in enumerate(decode_latents(vae, blk, scaling)):
                        im.save(rgb_dir / f"{i + j:03d}.png")
            print(f"[vae  ] decoded {n} PNGs -> {rgb_dir}")

    # ---------------- structural: the ControlNet-depth condition --------------
    # Only when the checkpoint's target is depth. The two structural targets are
    # consumed by different parts of the decoder -- a depth map is a ControlNet
    # condition, a VAE latent is an img2img initialisation -- so an export cannot
    # silently substitute one for the other.
    if str(cfg.get("struct_target", "vae")) == "depth" and latent is not None:
        mean = np.asarray(cfg.get("_vae_mean", []), dtype=np.float32)
        std = np.asarray(cfg.get("_vae_std", []), dtype=np.float32)
        rng = cfg.get("_struct_target_range")
        if mean.size != latent.shape[1] or std.size != latent.shape[1]:
            raise SystemExit("[depth] checkpoint args do not carry _vae_mean/_vae_std; "
                             "the prediction cannot be returned to the target's basis")
        if not rng or mean.size != 1:
            raise SystemExit(
                f"[depth] checkpoint args lack the pieces this export needs "
                f"(display range {rng}, {mean.size} channels). A depth condition has "
                f"to be expressed in the target's own units, and the range is a "
                f"fit-split artefact that cannot be recomputed here.")
        # The arithmetic lives in `build_depth_condition` so it can be tested without
        # a checkpoint. It is four operations with three ways to be subtly wrong --
        # the order of the denormalisation and the mean-field addition, the gain
        # being applied to the deviation rather than to the whole map, and the
        # quantisation using a per-image range -- and every one of them produces a
        # well-formed conditioning image that is simply the wrong one.
        full, u8 = build_depth_condition(
            latent.numpy(), mean, std, rng, field_mean(cfg),
            float(args.depth_dev_gain))
        gain = float(args.depth_dev_gain)
        cond = spatial / f"cond_depth_test_g{gain:g}"
        cond.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        # Written at 512x512 rather than at the supervised 8x8, for two reasons and
        # neither is about information. The project's existing depth conditions are
        # 512 (`build_gt_depth_cache.py`'s `test_rgb_512`), so anything downstream
        # that reads a condition directory sees the size it already expects. And the
        # upscale is done here with a known filter rather than being left to
        # `load_cond`, which would bicubic the 8x8 anyway -- doing it here just makes
        # the artifact on disk say what it is. No information is added: the source is
        # the same 8x8 map the loss was defined on, resampled.
        for i in range(n):
            arr = Image.fromarray((u8[i, 0] * 255.0).round().astype(np.uint8), mode="L")
            arr = arr.resize((512, 512), Image.Resampling.BILINEAR)
            Image.merge("RGB", (arr, arr, arr)).save(cond / f"{i:03d}.png")
        # The instance part, recomputed only for the report's spread statistic so it
        # is measured in the same units as the full map rather than inferred.
        dev = latent.numpy() * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
        # The display range has to be unpacked for the report. It was previously
        # unpacked INSIDE `build_depth_condition` (which still does, and validates it
        # there), and this dict kept referring to the outer names -- so the depth
        # export raised `NameError: name 'lo' is not defined` at the very end of its
        # report, after writing all 200 PNGs. Nothing caught it because the depth
        # branch had never run before this arm: the previous structural target was
        # VAE latents, which use the img2img path.
        lo, hi = float(rng[0]), float(rng[1])
        report["depth"] = {
            "npy": str(spatial / "pred_depth_test_scaled.npy"),
            "cond_dir": str(cond), "n": int(n), "dev_gain": gain,
            "cond_size": 512,
            "display_range": [lo, hi],
            "pred_range": [float(full.min()), float(full.max())],
            # The variance ratio and the contrast of the conditioning image are the
            # two numbers that say whether this is a condition or a flat card. A
            # condition whose per-sample spread is negligible is the previous
            # failure wearing a new name, and `--cn-scale 0` would reproduce it
            # exactly, so it is measured at export rather than eyeballed.
            "pred_dev_std": float(dev.std(axis=(0, 2, 3)).mean()),
            "full_std": float(full.std(axis=(0, 2, 3)).mean()),
            "u8_mean": float(u8.mean()), "u8_std": float(u8.std()),
        }
        print(f"[depth] dev gain {gain:g}: pred range "
              f"[{float(full.min()):.4f}, {float(full.max()):.4f}] shown over "
              f"[{lo:.4f}, {hi:.4f}] -> u8 mean {float(u8.mean()):.1f} "
              f"std {float(u8.std()):.1f}")
        print(f"[depth] wrote {n} conditioning PNGs -> {cond}")

        gtp = None
        scale = int(cfg.get("struct_scale", 64))
        if scale == 64:
            if str(cfg.get("depth_cache", "")):
                gtp = Path(str(cfg["depth_cache"])) / "test_depth_64.npy"
        else:
            # The coarse ladder is a separate cache, and pointing this at the 64x64
            # file would have compared an 8x8 prediction against a 64x64 target and
            # silently pooled one of them -- which is exactly the shape of mistake
            # the gate exists to catch, so the resolution match is asserted rather
            # than assumed below.
            root = cfg["_coarse_root"]
            gtp = Path(root) / f"test_depth_{scale}.npy"
        if gtp is not None and gtp.is_file():
            gt = np.load(gtp).astype(np.float32)
            if gt.ndim == 3:
                gt = gt[:, None]
            gt = gt[:n]
            fm = field_mean(cfg)
            if fm is not None:
                # `fm` is already (C, H, W) = (1, 8, 8) and `gt` is (N, 1, 8, 8):
                # the channel axis broadcasts, so no extra leading axis is needed.
                gt = gt - fm
            gtn = (gt - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
            if tuple(gtn.shape[2:]) != tuple(latent.shape[2:]):
                raise SystemExit(
                    f"[depth] ground truth is {tuple(gtn.shape[2:])} but the "
                    f"prediction is {tuple(latent.shape[2:])}; the gate compares "
                    f"variances and a cross-sample correlation, both of which change "
                    f"under resampling. Refusing rather than pooling one side.")
            gate = collapse_gate(latent.numpy(), gtn, "depth")
            report["depth"]["collapse_gate"] = gate
            print(f"[depth] collapse gate: across-sample var ratio "
                  f"{gate['pred_var_ratio']:.4f} (target {gate['target_var_ratio']:.4f}, "
                  f"floor {gate['var_ratio_floor']:.2f}) | r(pred,gt) "
                  f"{gate['r_pred_to_gt']:+.4f} vs constant "
                  f"{gate['r_constant_to_gt']:+.4f} -> margin "
                  f"{gate['margin_over_constant']:+.4f}")
            if not gate["passed"] and not args.allow_collapse:
                raise SystemExit(
                    f"[depth] COLLAPSED. The head reproduces "
                    f"{gate['pred_var_ratio']:.4f} of the across-sample variation "
                    f"(floor {gate['var_ratio_floor']:.2f}) and beats the constant "
                    f"predictor by {gate['margin_over_constant']:+.4f} (floor "
                    f"{gate['margin_floor']:+.2f}). Shipping it would make every arm "
                    f"of the comparison a measurement of the SDXL prior. Pass "
                    f"--allow-collapse to ship it anyway.")
        else:
            report["depth"]["collapse_gate"] = {
                "skipped": f"no co-resolution ground truth at {gtp} (scale {scale})"}
            print(f"[depth] collapse gate SKIPPED: {gtp}")

    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] {out / 'export_report.json'}")


if __name__ == "__main__":
    main()
