"""Train one subject: pretrained-ViT EEG encoder aligned to frozen image features.

Usage is driven by run_sub08.sh / slurm/*.sbatch. Key discipline enforced here:
  * test is scored exactly once, at the very end, on the best-val checkpoint
  * val is a concept-level holdout from the *training* concepts, so it never
    touches the 200 test concepts
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nwret import config
from nwret.augment import AUG_NAMES, build_aug
from nwret.data import AuxTargetDataset, TestDataset, TrainDataset, concept_split, load_subject
from nwret.encoders import backbone_patch_size
from nwret.losses import InfoNCE, grad_l1, latent_l1, mmd_rbf
from nwret.metrics import mean_rank, retrieval_report
from nwret.model import RetrievalModel, build_from_args


def require_cuda(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if not allow_cpu:
        raise SystemExit(
            "CUDA is not available. The login node has no GPU -- submit this via sbatch. "
            "Pass --allow-cpu only for a deliberate smoke test."
        )
    print("WARNING: running on CPU (--allow-cpu). This is a smoke test, not a result.")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # data / protocol
    p.add_argument("--subject", type=int, default=8)
    p.add_argument("--channels", type=str, default="all",
                    choices=["all", "occipito_parietal"])
    p.add_argument("--val-concepts", type=int, default=150)
    p.add_argument("--split-seed", type=int, default=2025)

    # alignment target -- WHICH layer of the image tower the EEG is asked to hit.
    # This is the axis the design doc prices as the largest single lever
    # (docs section 7, phase 1). With no override we use the cached final-layer
    # features, i.e. `visual.proj` output, which is what every earlier arm used.
    p.add_argument("--target-features", type=str, default=None,
                    help="directory of per-layer image features from extract_layers.py, "
                         "containing train/ and test/ subdirs of <key>.npy arrays")
    p.add_argument("--target-layer", type=str, default="_pooled",
                    help="which key to align to inside --target-features "
                         "(e.g. block16). '_pooled' is the final projected layer and "
                         "reproduces the shipped cached features.")
    p.add_argument("--target-layers", type=str, nargs="+", default=None,
                    help="align to SEVERAL image-tower layers at once, blended by "
                         "--target-fusion. Use this instead of --target-layer. Layers "
                         "must share a width, since they are blended before projection.")
    p.add_argument("--target-fusion", type=str, default="single",
                    choices=["single", "mean", "routed"],
                    help="how to combine several target layers. 'mean' pins equal "
                         "weights (the doc's '先均匀融合': do these layers carry "
                         "complementary information at all), 'routed' learns the blend "
                         "weights ('再上可学习路由'). Only meaningful with --target-layers.")
    p.add_argument("--fusion-mode", type=str, default="routed",
                    choices=["routed", "uniform", "none"],
                    help="how to combine several LAYERS OF THE EEG ENCODER (--layers). "
                         "This is a different axis from --target-fusion, which blends "
                         "layers of the frozen IMAGE tower. 'uniform' pins the EEG-side "
                         "weights at 1/k; 'routed' learns them (SAMGA's form); 'none' is "
                         "a single-layer pass-through with NO projection module, for "
                         "arms that must reproduce an external recipe that reads one "
                         "pooled vector straight into a head (EEGiT does).")

    # backbone
    p.add_argument("--backbone", type=str, default="timm:vit_b16_in21k",
                    help="openclip:<model> or timm:<name>; see encoders.build_encoder")
    p.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12])
    p.add_argument("--prior-center", type=int, default=None)
    p.add_argument("--prior-strength", type=float, default=1.0)
    p.add_argument("--freeze-blocks", type=int, default=0)
    p.add_argument("--freeze-all", action="store_true")
    p.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])
    p.add_argument("--no-pretrained", action="store_true")

    # tokenizer
    p.add_argument("--grid-h", type=int, default=7)
    p.add_argument("--grid-w", type=int, default=7)
    p.add_argument("--n-time-windows", type=int, default=4)

    # optimisation
    p.add_argument("--epochs", type=int, default=50,
                    help="SAMGA trains 50 epochs (20 stage-1 + 30 stage-2)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-embed", type=int, default=512)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--layer-dropout", type=float, default=0.1)
    p.add_argument("--subject-dropout", type=float, default=0.3)
    p.add_argument("--stage1-epochs", type=int, default=0,
                    help="epochs of coarse MMD alignment before contrastive-only training "
                         "(SAMGA: 20 of 50). Default 0 = MMD off. This deviates from SAMGA "
                         "deliberately: SAMGA's EEG encoder is a from-scratch projector, so "
                         "it needs a geometry warm-up before instance discrimination can "
                         "work. Ours is a pretrained ViT, which already carries a usable "
                         "geometry, and the measured effect of the warm-up here was negative "
                         "(sub-08: 20.00 -> 16.50 test Top-1). Non-zero re-enables the "
                         "SAMGA-faithful schedule, for the ablation.")
    p.add_argument("--mmd-start", type=float, default=0.9,
                    help="MMD weight at epoch 1 when stage 1 is enabled (SAMGA 0.9). The "
                         "contrastive weight in stage 1 is its complement, 1 - mmd_w, as in "
                         "SAMGA -- a plain sum would change the effective lr per epoch.")
    p.add_argument("--mmd-end", type=float, default=0.2,
                    help="MMD weight at the end of stage 1 (SAMGA 0.2)")
    p.add_argument("--stage2-lr", type=float, default=5e-5,
                    help="learning rate after stage 1 (SAMGA 5e-5). SAMGA uses a constant "
                         "lr within each stage rather than a decay, and so do we.")
    p.add_argument("--softplus", action="store_true",
                    help="apply softplus to the loss temperature as SAMGA does, giving an "
                         "effective logit scale of ~2.7 instead of ~14.3. Off by default: "
                         "the softer objective was measured worse here (sub-08: 21.00 -> "
                         "17.00 test Top-1 when combined with stage-1 MMD), consistent with "
                         "this run underfitting the alignment rather than overfitting it. "
                         "NOTE this flag is about the loss temperature, never the embedding; "
                         "softplus on the embedding was a bug and is gone.")
    p.add_argument("--no-img-l2norm", action="store_true")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--patience", type=int, default=12,
                    help="stop early after N epochs without a val Top-1 improvement "
                         "(0 disables). The unaugmented baseline peaked at epoch 8 and "
                         "then degraded for 22 more epochs, so this saves real time. The "
                         "counter is suspended during stage 1, where stagnation is expected.")
    p.add_argument("--test-every", type=int, default=0,
                   help="DIAGNOSTIC ONLY: evaluate the 200-way test set every N epochs and "
                        "record the result in `history` as `test_top1_diag`. Default 0 (off). "
                        "This deliberately exists because the run it was added for could not "
                        "answer its own central question: sub-08 peaked on the 150-concept "
                        "holdout at epoch 16 and decayed 9.93 points by epoch 100, but only "
                        "the best checkpoint was kept, so there is no way to know whether the "
                        "200-way test set decayed with it -- and the two are NOT the same "
                        "measurement (the holdout averages 4 EEG repetitions, the test set "
                        "80, so the test query is ~4.5x cleaner and is a materially easier "
                        "task). A val curve can be a valid proxy for test, or a bad one, and "
                        "nothing in the existing artefacts distinguishes those cases. "
                        "This flag never touches `sel`, never writes a checkpoint, and the "
                        "headline number stays val-selected: a test curve read for model "
                        "selection would be test-set over-selection bias, which is exactly "
                        "what the val split exists to avoid.")
    p.add_argument("--train-slots", type=int, nargs="+", default=None,
                    help="which image slots to TRAIN on, per concept (default: all "
                         "10). What changes is the TASK: with 10 images per concept "
                         "the model must also separate the ten images of one concept "
                         "from each other, so the objective carries an "
                         "instance-discrimination term that the val/test protocol -- "
                         "one image per concept, 200-way -- does not ask for. Passing "
                         "`--train-slots 0` removes that term at the cost of 90%% of "
                         "the training data (1654 rows instead of 16540), which is a "
                         "confound of its own, so it is an arm to measure rather than "
                         "an obvious fix. It is NOT about false negatives: measured on "
                         "this split, only 0.06%% of negative pairs are same-concept, "
                         "at any batch size "
                         "(`(B-1)*(S-1)/(C*S-1)` is invariant to B). Validation and "
                         "test are unaffected: selection still sweeps all 10 slots. "
                         "NOT a difference from EEGiT, despite what this comment said "
                         "for several months: the released code's `train_avg: True` "
                         "averages the EEG REPETITIONS (its `sorted_data` axis 1) and "
                         "keeps every image, which is why its "
                         "`sorted_session_list` is `np.zeros((16540, 4))` -- 16540 rows, "
                         "all ten slots. It trains on the same 16540 pairs we do, so it "
                         "is not a 10x data advantage and cannot explain the gap.")
    p.add_argument("--no-slot-sweep", action="store_true",
                    help="score validation on image slot 0 only. Off by default: the "
                         "sweep over all 10 slots cuts selection noise ~10x for free.")
    p.add_argument("--aug", type=str, default="full",
                    choices=list(AUG_NAMES),
                    help="EEG augmentation for the training split only (see nwret.augment)")

    # ---- tokenisation interface (EEGiT-style patches vs the legacy grid) -----
    p.add_argument("--tokenizer", type=str, default="grid", choices=["grid", "eegit"],
                    help="'eegit' builds EEGiT's patch representation (anatomical "
                         "regions -> 16x16 space-time patches) and feeds it through the "
                         "PRETRAINED patch_embed Conv2d. 'grid' is the legacy path, "
                         "which learned a random MLP on per-cell waveforms and never "
                         "called patch_embed at all.")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--n-patches-w", type=int, default=14,
                    help="temporal patches; 14 x 16 = 224 is EEGiT's width, giving "
                         "'14 x 5 = 70 spatiotemporal patches' with all 63 channels")
    p.add_argument("--patch-style", type=str, default="nw",
                    choices=["nw", "eegit_official"],
                    help="how the EEG patch image is laid out. 'eegit_official' "
                         "reproduces the released EEGiT code exactly: H = time "
                         "(resampled 250 -> n_patches_w*patch_size), W = regions, "
                         "regions anterior -> posterior in the dataset's own channel "
                         "order, and ONE 2D bilinear F.interpolate over the "
                         "(time, electrode) plane per region. 'nw' is the previous "
                         "interface (H = regions, W = time, posterior -> anterior, "
                         "montage-x sorted, 1D interpolation), which every result "
                         "produced before this flag existed was trained on.")
    p.add_argument("--no-zscore", action="store_true",
                    help="skip the per-channel z-score that puts EEG into the value "
                         "range the pretrained patch_embed expects")

    # ---- head ---------------------------------------------------------------
    p.add_argument("--head-kind", type=str, default="nw", choices=["nw", "eegit"],
                    help="'eegit' replaces the EEG projection head with the released "
                         "code's ProjectionHead: Linear -> GELU -> Linear -> "
                         "Dropout(0.5) -> + the PRE-GELU projection -> LayerNorm. It "
                         "is not the same block as the 'nw' 1-hidden-layer MLP; the "
                         "residual takes the projection, not the second linear.")
    p.add_argument("--img-head-kind", type=str, default="nw", choices=["nw", "eegit"],
                    help="same choice for the frozen-image-side projection. The "
                         "official code uses its ProjectionHead on BOTH sides.")
    p.add_argument("--head-drop", type=float, default=0.5,
                    help="dropout inside the 'eegit' projection heads (official 0.5).")
    p.add_argument("--timm-global-pool", type=str, default="", choices=["", "avg", "token"],
                    help="value passed as timm's `global_pool`. The official code uses "
                         "'avg', which makes timm create an `fc_norm` LayerNorm that is "
                         "applied after average pooling -- so 'avg' here is not the same "
                         "tensor as pool='mean' with an empty global_pool. Auto-set to "
                         "'avg' by --head-kind eegit.")

    # ---- post-norm -----------------------------------------------------------
    p.add_argument("--no-pool-norm", action="store_true",
                    help="do NOT apply the ViT's final LayerNorm before pooling. The "
                         "default (apply it) is the fix: the per-layer loop bypasses "
                         "timm's forward_features, so without it the alignment target "
                         "was an unnormalised residual stream. Pre-existing arms were "
                         "run with --no-pool-norm semantics.")

    # ---- optimisation (B1/B2 in the analysis) --------------------------------
    p.add_argument("--backbone-lr-mult", type=float, default=1.0,
                    help="learning-rate multiplier for the pretrained transformer "
                         "blocks. 1.0 = everything at --lr (the old behaviour). At "
                         "--lr 5e-4 with 0.1 the blocks sit at 5e-5, which is EEGiT's "
                         "encoder LR, while the new interface and heads train faster.")
    p.add_argument("--warmup-epochs", type=int, default=0,
                    help="linear LR warmup. 0 = off (the old behaviour). Warmup is "
                         "standard when fine-tuning a pretrained ViT; its absence was "
                         "a defect, not a tuning choice.")
    p.add_argument("--cosine", action="store_true",
                    help="cosine-decay the LR after warmup, down to --min-lr-ratio")
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam"],
                    help="'adam' is the released EEGiT code's optimizer "
                         "(`torch.optim.Adam`). The difference is not cosmetic: Adam's "
                         "weight_decay is an L2 penalty added to the gradient, AdamW's "
                         "is decoupled from the update. The paper's Implementation "
                         "Details says AdamW; the code says Adam, and the code wins "
                         "here because it is the artifact that produced the numbers.")
    p.add_argument("--wd-all-params", action="store_true",
                    help="apply weight decay to 1-D parameters (biases, LayerNorm "
                         "scales) as well. Off is the usual convention and what every "
                         "earlier arm did; ON is what the official code does, which "
                         "passes one weight_decay=1e-4 for the whole parameter group.")
    p.add_argument("--no-eeg-l2norm", action="store_true",
                    help="do NOT L2-normalise the EEG embedding inside InfoNCE. The "
                         "official code normalises ONLY the image side before the "
                         "loss (`img_z /= img_z.norm(...)`) and feeds the raw EEG "
                         "embedding in, so the EEG embedding's NORM is free and acts "
                         "as a learned per-sample logit scale -- a different objective "
                         "from the one this codebase trains by default. Retrieval is "
                         "always cosine, official included, so this changes training "
                         "only.")
    # Default is LEARNABLE, which is what every existing arm used. EEGiT fixes tau
    # at 0.07, so this is exposed rather than flipped: changing it would alter a
    # component that has already been validated here (F1/F3 kept softplus off and
    # exp on) and would confound the tokenisation change with a loss change.
    p.add_argument("--fixed-temp", action="store_true",
                    help="pin the InfoNCE temperature instead of learning it "
                         "(EEGiT fixes tau=0.07). Off by default: every existing "
                         "arm learned it.")

    # ---- diagnostics ---------------------------------------------------------
    p.add_argument("--no-cls-token", action="store_true",
                    help="do NOT add vit.cls_token to the cls slot's position "
                         "embedding. The default (add it) is the fix: the slot used "
                         "to hold pos_embed[0] alone, which dropped a pretrained "
                         "parameter from the input.")
    p.add_argument("--fit-diagnostic", action="store_true",
                    help="after loading the selected checkpoint, score a sample of "
                         "FIT concepts with the identical protocol as validation. "
                         "Gives the fit ceiling, which is what separates 'cannot fit' "
                         "from 'fits but does not generalise'.")
    p.add_argument("--fit-diag-concepts", type=int, default=150)

    # ---- structure tower (second tower; off unless --struct-backbone is given) --
    st = p.add_argument_group("structure tower",
        "A second EEG patch tower whose target is a spatial latent field rather than "
        "a joint-space vector. Off by default so every retrieval run stays identical.")
    st.add_argument("--struct-backbone", type=str, default="",
                    help="timm:NAME for the structure trunk. Empty = tower disabled. "
                         "Recommended: timm:dinov2_l (patch 14, 1024-d, 24 blocks).")
    st.add_argument("--struct-layers", type=int, nargs="+", default=[12, 18, 24],
                    help="structure-trunk blocks to fuse (1-based).")
    st.add_argument("--struct-patch-size", type=int, default=14,
                    help="must equal the structure backbone's patch_embed kernel; "
                         "validated against BACKBONES in resolve_target_plan.")
    st.add_argument("--struct-n-patches-w", type=int, default=16,
                    help="time-axis patch columns for the structure EEG image. Width "
                         "= n_patches_w * patch_size; 16*14 = 224 matches the "
                         "semantic tower's 14*16 = 224, so both towers see the same "
                         "1000 ms span.")
    st.add_argument("--struct-fusion-mode", type=str, default="uniform",
                    choices=("uniform", "routed"))
    st.add_argument("--struct-lr-mult", type=float, default=0.1,
                    help="LR multiplier for the structure trunk's blocks, relative to "
                         "--lr. Same convention as --backbone-lr-mult: the pretrained "
                         "trunk wants a smaller step than the randomly-initialised head.")
    st.add_argument("--struct-head-lr-mult", type=float, default=1.0,
                    help="LR multiplier for the structure in-heads (projection, "
                         "upsampling, depth/vae convs).")
    # Absolute overrides. These exist because `--lr` is the SEMANTIC tower's base LR,
    # and the two towers legitimately want different ones: the semantic side is
    # EEGiT's flat 5e-5 (one LR for a pretrained encoder plus two small heads), while
    # the structure tower's decoder is ~40M randomly-initialised parameters that
    # cannot be trained at the same step size as a pretrained trunk. Expressing the
    # structure LR as a multiple of the semantic LR then produces absurd multipliers
    # (10.0) whose meaning is not visible at the call site. With these, the call site
    # says the LR and the multiplier becomes the fallback.
    st.add_argument("--struct-lr", type=float, default=None,
                    help="absolute LR for the structure trunk's blocks; overrides "
                         "--struct-lr-mult.")
    st.add_argument("--struct-head-lr", type=float, default=None,
                    help="absolute LR for the structure in-heads; overrides "
                         "--struct-head-lr-mult.")
    st.add_argument("--struct-freeze-blocks", type=int, default=0,
                    help="freeze the first N structure-trunk blocks. The structure "
                         "tower carries ~304M parameters against 15k training pairs, "
                         "so this is the main capacity brake available.")
    st.add_argument("--struct-drop", type=float, default=0.1)
    st.add_argument("--struct-base-ch", type=int, default=128,
                    help="channels of the 8x8 field the fused feature projects to.")
    st.add_argument("--struct-field-ch", type=int, default=32)

    # ---- structure targets ---------------------------------------------------
    tg = p.add_argument_group("structure targets",
        "Auxiliary per-image targets for the structure tower. Both are keyed by "
        "concept_index*10+slot, which is the order the EEG arrays are already in.")
    tg.add_argument("--vae-latents", type=str, default=None,
                    help="directory holding train_vae_latents_f16.npy / "
                         "test_vae_latents_f16.npy, (N,4,64,64) float16, ALREADY "
                         "multiplied by the VAE's scaling_factor.")
    tg.add_argument("--depth-train", type=str, default=None)
    tg.add_argument("--depth-test", type=str, default=None)
    tg.add_argument("--w-depth", type=float, default=1.0)
    tg.add_argument("--w-vae", type=float, default=1.0)
    tg.add_argument("--lambda-grad", type=float, default=0.5,
                    help="weight of the depth gradient term (edge term). Applied to "
                         "the depth head only; see losses.grad_l1.")

    # ---- selection -----------------------------------------------------------
    se = p.add_argument_group("checkpoint selection",
        "What 'best' means once the model has two objectives. Both structural "
        "components are in the same 0-100 units as val Top-1 so the weights read as "
        "points of a 200-point scale rather than as an opaque sum.")
    se.add_argument("--struct-sel-w", type=float, default=0.5,
                    help="weight of val VAE-latent retrieval Top-1 (0-100).")
    se.add_argument("--depth-sel-w", type=float, default=5.0,
                    help="weight of val depth Pearson r (0-1), so 5.0 makes it worth "
                         "up to 5 points.")

    # runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--smoke", action="store_true", help="tiny run: 2 epochs, 2 val concepts")
    p.add_argument("--validate-only", action="store_true",
                    help="resolve and check the configuration, then exit without "
                         "touching the dataset or the GPU. Exists so a bad flag "
                         "combination fails in seconds on the login node instead of "
                         "after an allocation has been granted.")
    p.add_argument("--limit-samples", type=int, default=0,
                    help="cap the number of fit samples (debug/smoke only; 0 = no cap)")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: RetrievalModel, eeg: np.ndarray, feat: np.ndarray,
             device: torch.device, batch: int = 256, l2norm: bool = True,
             slot: int = 0) -> dict:
    """Rep-averaged retrieval over a fixed item set; diagonal is the answer.

    Both sides must pass through their respective projectors before comparison:
    the EEG embedding and the (projected) image embedding have to share a width,
    and the image side is NOT the raw cached feature.

    `slot` picks which image of each concept to use; the resulting task is
    `n_items`-way where n_items is the number of concepts in `eeg`.
    """
    model.eval()
    ds = TestDataset(eeg, feat, l2norm=l2norm, slot=slot)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    zs, fs = [], []
    for x, f, _ in dl:
        x = x.to(device, non_blocking=True)
        z, _, _ = model.encode_eeg(x, None, training=False)
        zs.append(z.float().cpu())
        fs.append(model.encode_image(f.to(device, non_blocking=True)).float().cpu())
    z = torch.cat(zs).numpy()
    f = torch.cat(fs).numpy()
    if z.shape[1] != f.shape[1]:
        raise RuntimeError(f"embedding width mismatch: eeg {z.shape[1]} vs image {f.shape[1]}")
    rep = retrieval_report(z, f)
    rep["mean_rank"] = mean_rank(z, f)
    return rep


@torch.no_grad()
def evaluate_selection(model: RetrievalModel, eeg: np.ndarray, feat: np.ndarray,
                       device: torch.device, sweep: bool = True, **kw) -> dict:
    """Validation metric for checkpoint selection.

    The validation split holds out *concepts*, and each concept still carries all
    10 of its images, so `eeg` is (n_concepts, 10, Ch, T). Scoring only slot 0 --
    which is what the original TestDataset did unconditionally -- throws away 90%
    of the holdout and makes the per-epoch val Top-1 noisy enough that it peaked
    at epoch 8 and drifted for 22 epochs afterwards.

    Sweeping all 10 slots and averaging gives ten independent readings of "can we
    retrieve the right image among n_concepts candidates". Same task difficulty as
    before (so it stays comparable to the 200-way test), ~10x less selection noise.
    """
    n_img = eeg.shape[1]
    slots = range(n_img) if sweep else [0]
    per = [evaluate(model, eeg, feat, device, slot=s, **kw) for s in slots]
    out = {
        "top1": float(np.mean([p["top1"] for p in per])),
        "top5": float(np.mean([p["top5"] for p in per])),
        "mean_rank": float(np.mean([p["mean_rank"] for p in per])),
        "n": int(per[0]["n"]),
        "n_slots": len(per),
        # Per-slot spread is the honest error bar on the selection signal.
        "top1_std": float(np.std([p["top1"] for p in per])),
    }
    return out


def _slot(rows: np.ndarray, s: int, name: str, want_tail, n_conc: int) -> torch.Tensor:
    """Index one image slot out of an (n_concepts, n_img, ...) target array.

    Exists because the two layouts are easy to confuse and the wrong one happens to
    be the *training* layout, not the validation one: `AuxTargetDataset` indexes
    rows as `concept * n_img + slot` in a flat (N, ...) array, whereas validation
    slices by slot with `rows[:, s]` and therefore needs the flat array already
    reshaped to (n_concepts, n_img, ...). Passing the flat array here does not
    silently produce a plausible number -- indexing dim 1 of (N, 4, 64, 64) drops
    a real spatial axis and the similarity matmul then fails -- but the error is a
    `mat1 and mat2 shapes cannot be multiplied`, which says nothing about the
    cause. So the shape is asserted here instead.
    """
    if rows.ndim < 2 or rows.shape[0] != n_conc:
        raise ValueError(
            f"{name} has shape {tuple(rows.shape)}; the evaluator needs "
            f"(n_concepts, n_img, ...) with n_concepts={n_conc}. If this is the "
            f"flat (n_concepts*n_img, ...) training layout, reshape it with "
            f"`.reshape(n_concepts, -1, *tail)` before passing it in.")
    got = tuple(rows.shape[2:])
    if want_tail is not None and got != tuple(want_tail):
        raise ValueError(f"{name}[2:] is {got} but the prediction is "
                         f"{tuple(want_tail)}; slot {s} is being indexed out of the "
                         f"wrong axis.")
    return torch.from_numpy(np.asarray(rows[:, s], dtype=np.float32))


@torch.no_grad()
def evaluate_selection_dual(
    model: RetrievalModel,
    eeg: np.ndarray,
    feat: np.ndarray,
    device: torch.device,
    vae_rows: np.ndarray | None = None,
    depth_rows: np.ndarray | None = None,
    vae_mean: np.ndarray | None = None,
    vae_std: np.ndarray | None = None,
    batch: int = 256,
    l2norm: bool = True,
    sweep: bool = True,
) -> dict:
    """Validation metrics for both towers, from ONE forward pass per slot.

    The structure trunk is the most expensive module in the model (304M params on
    24 blocks), so scoring the two towers in separate passes would roughly double
    validation time for exactly the same numbers.

    What the structural numbers mean
    --------------------------------
    `vae_top1` is the structural analogue of the semantic Top-1: rank the n_concepts
    ground-truth latents by similarity to the predicted one and ask whether the
    right one wins. It is computed on **mean-centred** flattened latents. Without
    centring, every latent shares a large common component (the average image
    layout, which the VAE encoder emits for any input) and the similarity is
    dominated by it -- so the metric would be measuring how well the prediction
    matches "a generic image" rather than "this image". Centring removes that shared
    direction and leaves only the instance-specific part, which is the part the EEG
    could plausibly determine.

    A constant predictor scores 1/n_concepts on `vae_top1` (centred or not), so the
    chance level is well defined and comparable to the semantic side.

    `depth_pearson` is the mean per-sample Pearson r between the predicted and true
    64x64 depth maps. It is scale-free, which matters because the depth head's
    output is squashed through a sigmoid and its calibration against the ground
    truth's absolute range is not meaningful.
    """
    n_conc, n_img = eeg.shape[0], eeg.shape[1]
    slots = list(range(n_img)) if sweep else [0]
    sem, v1, vc, dr = [], [], [], []
    for s in slots:
        zs, fs, vps, dps = [], [], [], []
        for i in range(0, n_conc, batch):
            x = torch.from_numpy(np.ascontiguousarray(eeg[i:i + batch, s])).to(device)
            f = torch.from_numpy(np.ascontiguousarray(feat[i:i + batch, s])).to(device)
            if l2norm:
                f = F.normalize(f.float(), dim=-1)
            out = model.forward_all(x, None, training=False)
            zs.append(F.normalize(out["z"].float(), dim=-1).cpu())
            fs.append(F.normalize(model.encode_image(f).float(), dim=-1).cpu())
            if out["struct"] is not None:
                vps.append(out["struct"]["vae"].float().cpu())
                dps.append(out["struct"]["depth"][:, 0].float().cpu())
        zc, fc = torch.cat(zs).numpy(), torch.cat(fs).numpy()
        r = retrieval_report(zc, fc)
        # `retrieval_report` returns only top1/top5/n -- mean_rank is a separate
        # function and has to be added here, exactly as `evaluate` does it.
        # Selection uses top1, but the result file reports mean_rank and a missing
        # key would only be discovered when someone reads the file.
        r["mean_rank"] = mean_rank(zc, fc)
        sem.append(r)

        if vps and vae_rows is not None:
            p = torch.cat(vps)
            if vae_mean is not None:
                m = torch.from_numpy(np.asarray(vae_mean, dtype=np.float32)).view(1, -1, 1, 1)
                sdv = torch.from_numpy(np.asarray(vae_std, dtype=np.float32)).view(1, -1, 1, 1)
                p = p * sdv + m
            gt = _slot(vae_rows, s, "vae_rows", p.shape[1:], n_conc)
            pf, gf = p.flatten(1), gt.flatten(1)
            pf = pf - pf.mean(0, keepdim=True)
            gf = gf - gf.mean(0, keepdim=True)
            sim = F.normalize(pf, dim=-1) @ F.normalize(gf, dim=-1).t()
            v1.append(100.0 * float((sim.argmax(1) == torch.arange(len(sim))).float().mean()))
            vc.append(float(sim.diag().mean()))

        if dps and depth_rows is not None:
            p = torch.cat(dps)
            gt = _slot(depth_rows, s, "depth_rows", None, n_conc)
            pc = p - p.mean(dim=(1, 2), keepdim=True)
            gc = gt - gt.mean(dim=(1, 2), keepdim=True)
            den = pc.flatten(1).norm(dim=1) * gc.flatten(1).norm(dim=1) + 1e-8
            dr.append(float(((pc * gc).sum(dim=(1, 2)) / den).mean()))

    return {
        "top1": float(np.mean([p["top1"] for p in sem])),
        "top5": float(np.mean([p["top5"] for p in sem])),
        "mean_rank": float(np.mean([p["mean_rank"] for p in sem])),
        "n": int(sem[0]["n"]),
        "n_slots": len(slots),
        "top1_std": float(np.std([p["top1"] for p in sem])),
        "vae_top1": float(np.mean(v1)) if v1 else None,
        "vae_top1_std": float(np.std(v1)) if v1 else None,
        "vae_cos": float(np.mean(vc)) if vc else None,
        "depth_pearson": float(np.mean(dr)) if dr else None,
        "depth_pearson_std": float(np.std(dr)) if dr else None,
    }


def load_ridge_ref(subject: int) -> dict | None:
    """Read the closed-form ridge baseline for this subject, if it has been run.

    The ridge number is the floor every deep configuration must clear. Without it
    a score like "21.5% Top-1" looks like progress when a plain linear map on the
    same features and split scores 25.0%. See docs section 0.5.
    """
    p = config.OUTPUTS / f"sub{subject:02d}" / "baseline_ridge.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    return {"test_top1": d["test"]["top1"], "test_top5": d["test"]["top5"],
            "val_top1": d["val"]["top1"], "n_channels": d.get("n_channels")}


def assign_param_groups(model: RetrievalModel) -> dict[str, list]:
    """Split trainable parameters into LR groups by name.

    Every parameter used to share one LR. That is a defect when the model mixes
    randomly-initialised modules (the tokenizer interface, the heads) with
    pretrained backbones: the two want LRs an order of magnitude apart, and a
    single value is a compromise that is wrong for both.

    Anything that matches no rule raises rather than landing silently in a default
    group -- an unnoticed parameter with the wrong LR is exactly the kind of silent
    failure HANDOFF section 6.3 warns about. Extracted from main() so the test suite
    can assert that every parameter of every configuration is covered, rather than
    discovering the gap when a new module is added and a run starts.
    """
    g: dict[str, list] = {k: [] for k in
                          ("interface", "blocks", "heads",
                           "s_interface", "s_blocks", "s_heads")}
    unknown = []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        # The structure tower is matched first: its names are prefixed `struct.`,
        # so they would otherwise fall through every semantic rule.
        if name.startswith("struct.encoder.vit.blocks."):
            g["s_blocks"].append((name, prm))
        elif name.startswith(("struct.encoder.vit.patch_embed.",
                              "struct.encoder.vit.pos_embed",
                              "struct.encoder.vit.cls_token",
                              "struct.encoder.vit.norm",
                              "struct.encoder.tokenizer.")):
            g["s_interface"].append((name, prm))
        elif name.startswith(("struct.encoder.", "struct.fusion.", "struct.proj.",
                              "struct.up.", "struct.depth_head.", "struct.vae_head.")):
            g["s_heads"].append((name, prm))
        elif name.startswith("encoder.vit.blocks."):
            g["blocks"].append((name, prm))
        elif name.startswith(("encoder.vit.patch_embed.", "encoder.vit.pos_embed",
                              "encoder.vit.cls_token", "encoder.vit.norm",
                              "encoder.tokenizer.")):
            g["interface"].append((name, prm))
        elif name.startswith(("encoder.", "fusion.", "eeg_head.", "img_head.")):
            g["heads"].append((name, prm))
        else:
            unknown.append(name)
    if unknown:
        raise SystemExit(f"parameters matching no LR group: {unknown[:5]} "
                         f"(+{max(0, len(unknown) - 5)} more); refusing to guess a LR")
    return g


def resolve_target_plan(args) -> list[str]:
    """Validate the target-layer/fusion combination and return the layer keys.

    Split out of main() so it runs without CUDA or the dataset. A mistyped flag
    combination then fails in seconds on the login node instead of after an
    allocation has been granted -- and it lets the pipeline's dry run exercise the
    real flag lists produced by run_arm.sh, rather than a copy of them that can
    drift out of sync.
    """
    keys = list(args.target_layers) if args.target_layers else [args.target_layer]
    if len(keys) > 1 and args.target_fusion == "single":
        raise SystemExit(f"--target-layers {keys} names {len(keys)} layers but "
                         f"--target-fusion single blends nothing; use mean or routed")
    if args.target_fusion != "single" and len(keys) < 2:
        raise SystemExit(f"--target-fusion {args.target_fusion} needs >=2 target layers, "
                         f"got {keys}")
    if len(keys) > 1 and args.target_layer != "_pooled":
        print(f"[data ] --target-layers given; ignoring --target-layer {args.target_layer!r}")
    if not args.target_features and len(keys) > 1:
        raise SystemExit("--target-layers requires --target-features")
    if not args.target_features and args.target_fusion != "single":
        raise SystemExit("--target-fusion requires --target-features "
                         "(the cached file holds only the final layer)")

    # ---- tokenisation interface -------------------------------------------
    # Everything here is checkable without the dataset or a GPU, so it belongs in
    # the dry run. A wrong token count is expensive to discover later: it does not
    # crash, it just trains a model with a different interface than intended.
    if args.tokenizer == "eegit":
        # The EEG image is patchified by the backbone's own pretrained conv, so
        # --patch-size is not free: it must equal that conv's kernel. Hardcoding 16
        # was correct while vit_b16_in21k was the only backbone, and silently wrong
        # the moment a patch-14 trunk (DINOv2) was added -- it would have produced
        # a 80x224 image for a 14x14 conv, i.e. patches straddling region and time
        # boundaries that the conv has never seen.
        want = backbone_patch_size(args.backbone)
        if want is None:
            raise SystemExit(f"cannot determine the patch size of {args.backbone!r}; "
                             f"register it in encoders.BACKBONES first")
        if args.patch_size != want:
            raise SystemExit(f"--backbone {args.backbone} has a {want}x{want} "
                             f"patch_embed conv, so --patch-size must be {want} "
                             f"(got {args.patch_size}); the pretrained conv is the "
                             f"interface and cannot be re-kernelled")
        n_regions = 5 if args.channels == "all" else 2
        n_tokens = n_regions * args.n_patches_w
        if args.patch_style == "eegit_official":
            h, w = args.n_patches_w * args.patch_size, n_regions * args.patch_size
        else:
            h, w = n_regions * args.patch_size, args.n_patches_w * args.patch_size
        print(f"[tok  ] eegit patches ({args.patch_style}): {n_regions} regions x "
              f"{args.n_patches_w} time patches = {n_tokens} tokens; EEG image "
              f"(3, H={h}, W={w}); pretrained patch_embed "
              f"Conv2d(3,{want},{want}) is the interface")
    else:
        if args.backbone_lr_mult != 1.0 or args.warmup_epochs or args.cosine:
            print("[tok  ] note: grid tokenizer with a non-default LR schedule; these "
                  "options were introduced for the EEGiT interface")

    # ---- EEGiT-official interface ------------------------------------------
    # The three flags below form one interface: read the pooled final layer, feed it
    # to the official projection head. Checked together because each of them is a
    # documented part of that recipe and any one of them silently missing produces a
    # run whose config claims a reproduction it did not perform.
    if args.head_kind == "eegit":
        if args.pool != "mean":
            raise SystemExit("--head-kind eegit reads the official encoder's "
                             "`global_pool='avg'` output; use --pool mean")
        if args.fusion_mode != "none":
            raise SystemExit("--head-kind eegit expects a single pooled vector; use "
                             "--fusion-mode none with one --layers entry, otherwise a "
                             "randomly-initialised fusion module sits between the "
                             "pretrained encoder and the head")
        if args.timm_global_pool != "avg":
            args.timm_global_pool = "avg"
            print("[tok  ] --head-kind eegit: setting --timm-global-pool avg, which is "
                  "what makes timm create the `fc_norm` LayerNorm the official code "
                  "pools through")
        if args.patch_style != "eegit_official":
            print("[tok  ] note: --head-kind eegit with --patch-style "
                  f"{args.patch_style}: the head is official, the EEG patch layout is "
                  "not")

    if args.fusion_mode == "none" and len(args.layers) != 1:
        raise SystemExit(f"--fusion-mode none is a pass-through for a single layer, "
                         f"got --layers {args.layers}")

    if args.no_eeg_l2norm and args.softplus:
        print("[loss ] note: --no-eeg-l2norm with --softplus is the released code's "
              "exact loss (unnormalised EEG, scale softplus(log(1/0.07))=2.73); with "
              "--softplus off the scale is exp(log(1/0.07))=14.29 as the paper states")

    # These are meaningful for any pretrained backbone, so they are checked outside
    # the tokenizer branch: they used to live inside it, which meant the moment the
    # structure tower added a second backbone with a non-eegit interface they would
    # have gone unchecked.
    if args.min_lr_ratio <= 0 or args.min_lr_ratio > 1:
        raise SystemExit(f"--min-lr-ratio must be in (0, 1], got {args.min_lr_ratio}")
    if args.train_slots is not None:
        # Checked here, against the documented constant, rather than only after the
        # data arrays are loaded: `--validate-only` and the pipeline's dry run both
        # exit before data loading, so a slot index typo would otherwise survive
        # every pre-flight check and fail inside a DataLoader worker on the GPU node.
        n_slots = config.N_IMAGES_PER_CONCEPT
        bad = [s for s in args.train_slots if not (0 <= s < n_slots)]
        if bad:
            raise SystemExit(f"--train-slots {bad} outside 0..{n_slots - 1}")
        if len(set(args.train_slots)) != len(args.train_slots):
            print(f"[data ] note: --train-slots {args.train_slots} has duplicates; "
                  f"using {sorted(set(args.train_slots))}")
    if args.backbone_lr_mult <= 0:
        raise SystemExit(f"--backbone-lr-mult must be > 0, got {args.backbone_lr_mult}")
    if args.warmup_epochs >= args.epochs:
        raise SystemExit(f"--warmup-epochs {args.warmup_epochs} >= --epochs {args.epochs}")
    if args.cosine and args.stage1_epochs > 0:
        raise SystemExit("--cosine and --stage1-epochs are mutually exclusive: the "
                         "scheduler would overwrite the stage-2 LR")

    # ---- structure tower ----------------------------------------------------
    if args.struct_backbone:
        want_s = backbone_patch_size(args.struct_backbone)
        if want_s is None:
            raise SystemExit(f"cannot determine the patch size of "
                             f"{args.struct_backbone!r}; register it in BACKBONES")
        if args.struct_patch_size != want_s:
            raise SystemExit(f"--struct-backbone {args.struct_backbone} has a "
                             f"{want_s}x{want_s} patch_embed conv, so "
                             f"--struct-patch-size must be {want_s} "
                             f"(got {args.struct_patch_size})")
        if not args.struct_layers:
            raise SystemExit("--struct-backbone given but --struct-layers is empty")
        if not (args.vae_latents or (args.depth_train and args.depth_test)):
            raise SystemExit("--struct-backbone given but no structure targets: pass "
                             "--vae-latents and/or --depth-train/--depth-test. A "
                             "structure tower with no structural loss is an untrained "
                             "304M-parameter decoder attached to the run.")
        if bool(args.depth_train) != bool(args.depth_test):
            raise SystemExit("--depth-train and --depth-test must be given together")
        if args.struct_lr_mult <= 0 or args.struct_head_lr_mult <= 0:
            raise SystemExit("--struct-lr-mult and --struct-head-lr-mult must be > 0")
        # The absolute overrides win over the multipliers, so a non-positive value
        # here is a typo that would otherwise be silently replaced by the
        # multiplier -- or, with 0, freeze the structure tower's training entirely
        # while the log claimed a LR.
        for _flag, _v in (("--struct-lr", args.struct_lr),
                          ("--struct-head-lr", args.struct_head_lr)):
            if _v is not None and _v <= 0:
                raise SystemExit(f"{_flag} must be > 0, got {_v:g}")
        # 64 = 8 << 3: the upsampling stack is fixed at three x2 steps. Checked here
        # because out_hw is baked into the export path, so getting it wrong yields a
        # latent the VAE decoder rejects -- after training, not before.
        if args.struct_base_ch % 2:
            raise SystemExit(f"--struct-base-ch {args.struct_base_ch} must be even; the "
                             f"first up-block halves it")
        n_tok_s = (5 if args.channels == "all" else 2) * args.struct_n_patches_w
        n_b = args.struct_layers[-1]
        # Report (H, W) in the same order the semantic branch above does, and honour
        # `--patch-style`: the two styles swap which axis is time, so a hardcoded
        # order printed a transposed geometry for exactly the runs that changed it.
        n_reg_s = 5 if args.channels == "all" else 2
        if args.patch_style == "eegit_official":
            h_s, w_s = args.struct_n_patches_w * args.struct_patch_size, \
                n_reg_s * args.struct_patch_size
        else:
            h_s, w_s = n_reg_s * args.struct_patch_size, \
                args.struct_n_patches_w * args.struct_patch_size
        print(f"[tok  ] structure tower: {args.struct_backbone} patch "
              f"{args.struct_patch_size}, EEG image ({args.patch_style}) "
              f"(3, H={h_s}, W={w_s}), {n_tok_s} tokens, "
              f"layers {args.struct_layers}, fusion {args.struct_fusion_mode}")
        if any(l < 1 or l > 24 for l in args.struct_layers):
            raise SystemExit(f"--struct-layers {args.struct_layers} outside 1..24; a "
                             f"block index past the trunk's depth would silently be "
                             f"absent from the fusion dict")
        print(f"[loss ] structure targets: vae={bool(args.vae_latents)} "
              f"depth={bool(args.depth_train)} at weights "
              f"w_vae={args.w_vae:g} w_depth={args.w_depth:g} "
              f"lambda_grad={args.lambda_grad:g}")
    else:
        for flag in ("vae_latents", "depth_train", "depth_test"):
            if getattr(args, flag):
                print(f"[cfg  ] --{flag.replace('_', '-')} ignored: no --struct-backbone")
    return keys


def main() -> None:
    args = parse_args()
    keys = resolve_target_plan(args)

    if args.validate_only:
        # Config-only check: no data, no GPU, no downloads. Used by the pipeline's
        # dry run so that every arm's flag combination is verified before submission.
        print(f"[ok   ] config valid: backbone={args.backbone} eeg-layers={args.layers} "
              f"fusion={args.fusion_mode} | targets={keys} "
              f"target-fusion={args.target_fusion} | "
              f"{'eegit' if args.tokenizer == 'eegit' else 'grid'}"
              f"-tokens="
              f"{((5 if args.channels == 'all' else 2) * args.n_patches_w) if args.tokenizer == 'eegit' else args.grid_h * args.grid_w * args.n_time_windows}"
              f" | pool_norm={not args.no_pool_norm} "
              f"cls_token={not args.no_cls_token} | lr={args.lr:g} "
              f"blocks x{args.backbone_lr_mult:g} warmup={args.warmup_epochs} "
              f"cosine={args.cosine} | fit_diag={args.fit_diagnostic}")
        return

    set_seed(args.seed)

    device = require_cuda(args.allow_cpu) if args.device == "cuda" else torch.device(args.device)
    l2norm = not args.no_img_l2norm
    RIDGE_REF = load_ridge_ref(args.subject)

    if args.smoke:
        args.epochs = 2
        args.val_concepts = min(args.val_concepts, 20)

    channels = None if args.channels == "all" else config.CHANNELS_OCCIPITO_PARIETAL
    ch_names = list(channels) if channels else None

    # ---------------- data
    tr_eeg, te_eeg = load_subject(args.subject, channels)
    if ch_names is None:
        ch_names = json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"]

    # Which image-tower layer the EEG encoder is asked to hit. Choosing this is the
    # design doc's phase-1 lever; see probe_layers.py for how the choice is made.
    # Several layers can be named at once and blended (--target-fusion). The
    # combination was already validated by resolve_target_plan() above, before the
    # GPU and the dataset were required.
    if args.target_features:
        tdir = Path(args.target_features)
        arrs_tr, arrs_te = [], []
        for key in keys:
            f_tr, f_te = tdir / "train" / f"{key}.npy", tdir / "test" / f"{key}.npy"
            if not (f_tr.is_file() and f_te.is_file()):
                raise SystemExit(
                    f"--target-features {tdir} has no {key}.npy under both train/ and test/. "
                    f"Found train/: {sorted(p.stem for p in (tdir / 'train').glob('*.npy'))[:6]}")
            arrs_tr.append(np.load(f_tr))
            arrs_te.append(np.load(f_te))
        dims = {int(a.shape[-1]) for a in arrs_tr} | {int(a.shape[-1]) for a in arrs_te}
        if len(dims) != 1:
            # Blending requires a common width; two layers with different widths are
            # not two views of one space and would need separate heads.
            raise SystemExit(f"target layers have different widths {sorted(dims)}; "
                             f"blending them is not meaningful")
        if len(keys) > 1:
            # (C, I, k, D): the k axis is the target index. Normalisation and the
            # batch layout downstream treat the last axis as the feature, so nothing
            # else has to change to carry k targets.
            img_tr = np.stack(arrs_tr, axis=2)
            img_te = np.stack(arrs_te, axis=2)
            print(f"[data ] alignment target = {len(keys)} blended layers {keys} "
                  f"({img_tr.shape[-1]}-d each, fusion={args.target_fusion}) from {tdir}")
        else:
            img_tr, img_te = arrs_tr[0], arrs_te[0]
            print(f"[data ] alignment target = {keys[0]} ({img_tr.shape[-1]}-d) from {tdir}")
    else:
        # No target-features: the shipped cached file, which holds only the final
        # projected layer. resolve_target_plan() already refused multi-target or
        # blended configurations here, so keys is exactly one name.
        img_tr = np.load(config.IMAGE_FEATURE_DIR / "image_train.npy")
        img_te = np.load(config.IMAGE_FEATURE_DIR / "image_test.npy")
        print(f"[data ] alignment target = cached final-layer features "
              f"({img_tr.shape[-1]}-d), i.e. visual.proj output")

    # The feature and EEG arrays must agree concept-for-concept, image-for-image.
    # If they disagree the pairing is scrambled and every number below is noise that
    # still looks like a result, so fail loudly instead.
    if img_tr.shape[:2] != tr_eeg.shape[:2]:
        raise SystemExit(
            f"alignment target does not match the EEG layout: features {img_tr.shape[:2]} "
            f"vs EEG {tr_eeg.shape[:2]}. Wrong layer file or wrong concept order.")
    if img_te.shape[:2] != te_eeg.shape[:2]:
        raise SystemExit(
            f"test layout mismatch: features {img_te.shape[:2]} vs EEG {te_eeg.shape[:2]}")

    # The width of `img_head`'s input, parked on `args` so it travels into the
    # checkpoint. The export builds the model hours later, in a separate process,
    # and would otherwise have to guess it from whichever gallery file it happened
    # to load -- a guess that is wrong for any run whose alignment target is an
    # intermediate layer rather than the final projected one (block26 is 1280-d,
    # while the shipped `image_train.npy` is the 1024-d `visual.proj` output).
    # That mismatch surfaces as a `size mismatch for img_head.weight` RuntimeError
    # after training has already been paid for.
    args._image_dim = int(img_tr.shape[-1])

    split = concept_split(args.val_concepts, args.split_seed)
    augment = build_aug(args.aug)
    # Resolved here rather than inline so the validator can check it before the
    # dataset is built: a slot index past the array's width would otherwise surface
    # as an IndexError from inside a DataLoader worker.
    train_slots = sorted(set(int(s) for s in args.train_slots)) if args.train_slots else None
    if train_slots is not None:
        n_slots = int(tr_eeg.shape[1])
        bad = [s for s in train_slots if not (0 <= s < n_slots)]
        if bad:
            raise SystemExit(f"--train-slots {bad} outside 0..{n_slots - 1}; the train "
                             f"array has {n_slots} images per concept")

    # ---- structure targets ---------------------------------------------------
    # Loaded as memmaps: the VAE cache is 542 MB and the depth cache 271 MB, and
    # neither is read more than once per sample. Both are indexed by
    # concept_index*10+slot, which is the row order `load_subject` produces.
    vae_tr = vae_te = depth_tr = depth_te = None
    vae_mean = vae_std = None
    if args.struct_backbone:
        if args.vae_latents:
            vdir = Path(args.vae_latents)
            f_tr, f_te = vdir / "train_vae_latents_f16.npy", vdir / "test_vae_latents_f16.npy"
            if not (f_tr.is_file() and f_te.is_file()):
                raise SystemExit(f"--vae-latents {vdir} needs train_vae_latents_f16.npy "
                                 f"and test_vae_latents_f16.npy")
            vae_tr = np.load(f_tr, mmap_mode="r")
            vae_te = np.load(f_te, mmap_mode="r")
        if args.depth_train:
            depth_tr = np.load(args.depth_train, mmap_mode="r")
            depth_te = np.load(args.depth_test, mmap_mode="r")

        n_img = tr_eeg.shape[1]
        expected = tr_eeg.shape[0] * n_img
        for nm, a in (("vae", vae_tr), ("depth", depth_tr)):
            if a is not None and a.shape[0] != expected:
                raise SystemExit(
                    f"{nm} cache has {a.shape[0]} rows but the EEG layout implies "
                    f"{expected} ({tr_eeg.shape[0]}x{n_img}). Everything downstream "
                    f"assumes row = concept*{n_img}+slot, so a mismatch means every "
                    f"image is paired with another image's target.")
        if vae_te is not None and vae_te.shape[0] != te_eeg.shape[0]:
            raise SystemExit(f"vae test cache has {vae_te.shape[0]} rows, expected "
                             f"{te_eeg.shape[0]} (one image per test concept)")
        if depth_te is not None and depth_te.shape[0] != te_eeg.shape[0]:
            raise SystemExit(f"depth test cache has {depth_te.shape[0]} rows, expected "
                             f"{te_eeg.shape[0]}")

        if vae_tr is not None:
            # Statistics from the FIT concepts only. Using all rows would leak the
            # val holdout's scale into training, exactly as it would for the EEG
            # z-score. Per-channel (4 numbers each) because the 4 latent channels
            # have visibly different variances and a single global scale would let
            # the dominant channel own the loss.
            s = np.zeros(vae_tr.shape[1], dtype=np.float64)
            ss = np.zeros(vae_tr.shape[1], dtype=np.float64)
            n_seen = 0
            n_ch = vae_tr.shape[1]
            for c0 in range(0, len(split.fit_concepts), 128):
                cs = split.fit_concepts[c0:c0 + 128]
                rows = (cs[:, None] * n_img + np.arange(n_img)[None, :]).ravel()
                blk = np.asarray(vae_tr[np.sort(rows)], dtype=np.float64)
                blk = blk.reshape(blk.shape[0], n_ch, -1)     # (rows, ch, 64*64)
                s += blk.sum(axis=(0, 2))
                ss += (blk ** 2).sum(axis=(0, 2))
                n_seen += blk.shape[0] * blk.shape[2]
            vae_mean = s / n_seen
            vae_std = np.sqrt(np.maximum(ss / n_seen - vae_mean ** 2, 1e-12))
            # Parked on `args` so they travel into the checkpoint's saved args. The
            # export step runs in a separate process hours later and cannot
            # recompute them: the fit split is a training artefact, and re-deriving
            # the normalisation there would be one more chance to silently disagree
            # with the basis the head was trained in.
            args._vae_mean = [float(v) for v in vae_mean]
            args._vae_std = [float(v) for v in vae_std]
            print(f"[data ] VAE latents {tuple(vae_tr.shape)} f16, normalised by "
                  f"fit-split per-channel stats: mean {np.round(vae_mean, 4).tolist()} "
                  f"std {np.round(vae_std, 4).tolist()}")
        if depth_tr is not None:
            print(f"[data ] depth {tuple(depth_tr.shape)} in [{float(np.asarray(depth_tr[0]).min()):.3f}, "
                  f"{float(np.asarray(depth_tr[0]).max()):.3f}], range assumed 0-1 from "
                  f"the depth head's sigmoid")

    ds_fit = TrainDataset(tr_eeg, img_tr, split.fit_concepts, l2norm=l2norm,
                          augment=augment, seed=args.seed, slots=train_slots)
    if args.struct_backbone:
        ds_fit = AuxTargetDataset(
            tr_eeg, img_tr, split.fit_concepts,
            aux_vae=vae_tr, aux_depth=depth_tr,
            vae_mean=vae_mean, vae_std=vae_std,
            l2norm=l2norm, augment=augment, seed=args.seed, slots=train_slots)
    if train_slots is not None:
        # Say so loudly: this changes the contrastive TASK, not just its size. With
        # ten images per concept the model is additionally asked to separate the ten
        # images of one concept from each other, an instance-discrimination term the
        # one-image-per-concept retrieval protocol never asks for. It is not about
        # false negatives -- measured here, only 0.06% of negative pairs share a
        # concept, independent of batch size.
        print(f"[data ] TRAINING ON {len(train_slots)} IMAGE SLOT(S) {train_slots} PER "
              f"CONCEPT ({len(split.fit_concepts) * len(train_slots)} pairs, was "
              f"{len(split.fit_concepts) * tr_eeg.shape[1]}); this drops the "
              f"instance-discrimination term the official code never trains")
    # Validation keeps all 10 images per concept; evaluate_selection sweeps them.
    val_eeg = tr_eeg[split.val_concepts]
    val_feat = img_tr[split.val_concepts]
    val_vae = val_depth = None
    if args.struct_backbone:
        rows = (split.val_concepts[:, None] * tr_eeg.shape[1]
                + np.arange(tr_eeg.shape[1])[None, :])
        val_vae = vae_tr[rows] if vae_tr is not None else None
        val_depth = depth_tr[rows] if depth_tr is not None else None
    has_vae = bool(val_vae is not None)
    has_depth = bool(val_depth is not None)
    if augment is not None:
        print(f"[aug  ] train-only augmentation: {augment!r}")
    else:
        print("[aug  ] augmentation DISABLED (--aug none)")

    dl_fit = DataLoader(ds_fit, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=(device.type == "cuda"))

    if args.limit_samples:
        # Debug path: truncate the fit set so a smoke test does not need a GPU
        # for an hour. Never used for a reported number.
        from torch.utils.data import Subset
        n_keep = min(args.limit_samples, len(ds_fit))
        dl_fit = DataLoader(Subset(ds_fit, list(range(n_keep))),
                            batch_size=min(args.batch_size, n_keep), shuffle=True,
                            num_workers=2, drop_last=True,
                            pin_memory=(device.type == "cuda"))
        if len(dl_fit) == 0:
            raise SystemExit(f"--limit-samples {args.limit_samples} too small for batch "
                             f"{args.batch_size}; need at least one full batch")
        print(f"[debug] --limit-samples active: {n_keep} fit samples, {len(dl_fit)} batches")

    # ---------------- model
    model = build_from_args(vars(args), ch_names, int(img_tr.shape[-1])).to(device)

    # The z-score statistics must come from the FIT split only. Computing them on
    # train+test would leak the test set's scale into training, and computing them
    # per batch would make the patch_embed's input distribution drift during the run.
    # Both towers get them: the tokenizer is parameter-free and the two towers see
    # the identical EEG, so any difference between them here would be pure noise
    # injected into the structural branch.
    if args.tokenizer == "eegit" and not args.no_zscore:
        tok = model.encoder.tokenizer
        tok.set_norm_stats(tr_eeg[split.fit_concepts])
        print(f"[zscor] per-channel z-score from {len(split.fit_concepts)} fit concepts "
              f"(mean {float(tok.eeg_mean.mean()):+.3f}, std {float(tok.eeg_std.mean()):.3f})")
    if model.struct is not None and not args.no_zscore:
        stok = model.struct.encoder.tokenizer
        stok.set_norm_stats(tr_eeg[split.fit_concepts])

    # Count the WHOLE model, not just the encoder. `model.encoder.trainable_parameter_summary()`
    # omits LayerFusion and the two heads, which is 3.41M params (~4%) on the 63ch
    # EEGiT config -- so the recorded provenance understated the real model size in
    # every earlier run. Reported here as (trainable, total) over model.parameters().
    tr_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot_p = sum(p.numel() for p in model.parameters())
    print(f"[model] backbone={args.backbone} layers={args.layers} "
          f"tokens={model.encoder.tokenizer.n_tokens} grid={model.encoder.dst_grid} "
          f"src_grid={model.encoder.src_grid}")
    print(f"[model] eeg-layer fusion={args.fusion_mode} over {args.layers}; "
          f"target fusion={args.target_fusion} over {keys if len(keys) > 1 else keys[0]}")
    if model.struct is not None:
        print(f"[model] structure tower {args.struct_backbone} "
              f"tokens={model.struct.encoder.tokenizer.n_tokens} "
              f"grid={model.struct.encoder.dst_grid} "
              f"layers={args.struct_layers} fusion={args.struct_fusion_mode} "
              f"field=({args.struct_field_ch},{model.struct.out_hw},{model.struct.out_hw})")
    print(f"[model] trainable params {tr_p/1e6:.2f}M / {tot_p/1e6:.2f}M")

    # ---- parameter groups ---------------------------------------------------
    _pg = assign_param_groups(model)
    g_interface, g_blocks, g_heads = _pg["interface"], _pg["blocks"], _pg["heads"]
    g_s_interface, g_s_blocks, g_s_heads = _pg["s_interface"], _pg["s_blocks"], _pg["s_heads"]

    lr_backbone = args.lr * args.backbone_lr_mult
    # `--struct-lr` / `--struct-head-lr` are absolute and win over the multipliers:
    # see the note where they are declared. `is not None` rather than a truthiness
    # test -- `--struct-lr 0` is a typo, not a request to fall back to the
    # multiplier, and it is rejected in the dry run instead of being misinterpreted
    # here. Default None keeps every earlier run's meaning intact.
    lr_struct = (args.struct_lr if args.struct_lr is not None
                 else args.lr * args.struct_lr_mult)
    lr_struct_head = (args.struct_head_lr if args.struct_head_lr is not None
                      else args.lr * args.struct_head_lr_mult)

    def _group(params, lr, tag_, wd):
        # Weight decay is not applied to 1-D parameters (biases and LayerNorm
        # scales), the usual convention: decaying them pulls the normalisation
        # scales towards zero, which fights the LayerNorm's job. `--wd-all-params`
        # turns the split off, which is what the official EEGiT code does -- it
        # passes one `weight_decay=1e-4` for the whole parameter list, so its
        # LayerNorm scales and biases ARE decayed.
        if args.wd_all_params:
            out = []
            if params:
                out.append({"params": [p for _, p in params], "lr": lr,
                            "weight_decay": wd, "group": tag_})
            return out
        decay = [p for n, p in params if p.ndim > 1]
        no_decay = [p for n, p in params if p.ndim <= 1]
        out = []
        if decay:
            out.append({"params": decay, "lr": lr, "weight_decay": wd, "group": tag_})
        if no_decay:
            out.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "group": tag_})
        return out

    groups = (_group(g_interface, args.lr, "interface", args.weight_decay)
              + _group(g_blocks, lr_backbone, "blocks", args.weight_decay)
              + _group(g_heads, args.lr, "heads", args.weight_decay)
              + _group(g_s_interface, args.lr, "s_interface", args.weight_decay)
              + _group(g_s_blocks, lr_struct, "s_blocks", args.weight_decay)
              + _group(g_s_heads, lr_struct_head, "s_heads", args.weight_decay))
    # Capture the CONFIGURED LR per group before any scheduler touches it. Reading
    # `param_groups[i]["lr"]` when the result JSON is written happens after the
    # cosine schedule has decayed to min_lr_ratio, so it would record 5e-6 for a run
    # that actually trained at 5e-4 -- a provenance value off by 100x.
    lr_groups_cfg = {g["group"]: g["lr"] for g in groups if "group" in g}
    optim_cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    optim = optim_cls(groups, betas=(0.9, 0.999))
    print(f"[optim] {args.optimizer} (betas 0.9/0.999, wd {args.weight_decay:g}"
          f"{' on ALL params' if args.wd_all_params else ', 1-D params excluded'})")
    n_interface = sum(p.numel() for _, p in g_interface)
    n_blocks_p = sum(p.numel() for _, p in g_blocks)
    print(f"[optim] groups: interface {n_interface/1e6:.2f}M @ lr {args.lr:g} | "
          f"blocks {n_blocks_p/1e6:.2f}M @ lr {lr_backbone:g} "
          f"(base x{args.backbone_lr_mult:g}) | heads "
          f"{sum(p.numel() for _, p in g_heads)/1e6:.2f}M @ lr {args.lr:g}")
    if g_s_blocks or g_s_heads:
        # Report which of the two mechanisms set each LR, rather than always printing
        # a multiplier: with `--struct-lr` given, "base x0.1" would be a lie.
        def _how(mult, abs_):
            return f"absolute {abs_:g}" if abs_ is not None else f"base x{mult:g}"

        print(f"[optim] structure: s_blocks "
              f"{sum(p.numel() for _, p in g_s_blocks)/1e6:.2f}M @ lr {lr_struct:g} "
              f"({_how(args.struct_lr_mult, args.struct_lr)}) | s_heads "
              f"{sum(p.numel() for _, p in g_s_heads)/1e6:.2f}M @ lr {lr_struct_head:g} "
              f"({_how(args.struct_head_lr_mult, args.struct_head_lr)}) | s_interface "
              f"{sum(p.numel() for _, p in g_s_interface)/1e6:.2f}M @ lr {args.lr:g}")

    # ---- schedule: linear warmup then cosine --------------------------------
    steps_per_epoch = max(1, len(dl_fit))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    if warmup_steps >= total_steps:
        raise SystemExit(f"--warmup-epochs {args.warmup_epochs} >= --epochs {args.epochs}")
    min_ratio = args.min_lr_ratio

    def lr_lambda(step: int) -> float:
        # Returns a multiplier on each group's configured LR, so the per-group
        # ratios set above are preserved for the whole run.
        if warmup_steps and step < warmup_steps:
            return 0.1 + 0.9 * (step / warmup_steps)
        if not args.cosine:
            return 1.0
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    print(f"[optim] schedule: {warmup_steps} warmup steps then "
          f"{'cosine' if args.cosine else 'constant'} to x{min_ratio:g} over "
          f"{total_steps} total steps ({steps_per_epoch}/epoch)")

    # SAMGA holds the lr constant inside each stage and drops it at the stage
    # boundary (1e-4 -> 5e-5). We keep that behaviour available (--cosine off) but
    # a schedule is the default path for the pretrained-interface configuration.
    lr_stage2 = min(args.stage2_lr, args.lr)
    criterion = InfoNCE(softplus=args.softplus, learnable=not args.fixed_temp,
                        l2norm_a=not args.no_eeg_l2norm).to(device)
    _scale0 = float(criterion.effective_scale().detach())
    print(f"[loss ] InfoNCE temperature via {'softplus' if args.softplus else 'exp'}"
          f"(logit_scale) -> initial effective logit scale {_scale0:.2f}, "
          f"{'FIXED' if args.fixed_temp else 'learnable'}; "
          f"EEG side {'NOT ' if args.no_eeg_l2norm else ''}L2-normalised "
          f"(image side always normalised)")
    if args.stage1_epochs > 0:
        print(f"[loss ] stage 1 = {args.stage1_epochs} epochs of "
              f"{args.mmd_start}*MMD + {1 - args.mmd_start:.1f}*contrastive (MMD annealed to "
              f"{args.mmd_end}), then contrastive only at lr {lr_stage2:g}")
    else:
        print("[loss ] MMD disabled (--stage1-epochs 0); contrastive only from epoch 1")

    out_dir = Path(args.out_dir) if args.out_dir else (config.OUTPUTS / f"sub{args.subject:02d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.backbone}_L{'-'.join(map(str, args.layers))}"

    # ---------------- loop
    best = {"sel": -1e9, "top1": -1.0, "epoch": -1}
    hist = []
    t0 = time.time()
    t_train_s = 0.0
    t_val_s = 0.0
    n_steps = 0
    epochs_no_gain = 0
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        # SAMGA's two-stage schedule: coarse alignment first, then contrastive only.
        in_stage1 = args.stage1_epochs > 0 and epoch <= args.stage1_epochs
        if in_stage1:
            if args.stage1_epochs <= 1:
                mmd_w = args.mmd_end
            else:
                prog = (epoch - 1) / (args.stage1_epochs - 1)
                mmd_w = args.mmd_start + (args.mmd_end - args.mmd_start) * max(0.0, min(1.0, prog))
        else:
            mmd_w = 0.0
        contrast_w = 1.0 - mmd_w if in_stage1 else 1.0
        if args.stage1_epochs > 0 and epoch == args.stage1_epochs + 1:
            # Only meaningful when no schedule is active. With --cosine the
            # LambdaLR would multiply this constant by the schedule factor on the
            # very next step, so the assignment would be silently reverted -- the
            # exact class of no-op that looks like it worked.
            if args.cosine:
                raise SystemExit("--stage1-epochs with --cosine are mutually exclusive: "
                                 "the scheduler overwrites the stage-2 LR on the next step")
            for g in optim.param_groups:
                g["lr"] = lr_stage2
            print(f"[lr   ] stage 2 begins: lr -> {lr_stage2:g}")

        tot, nb, tr_top1 = 0.0, 0, 0
        tot_sem = tot_vae = tot_dep = 0.0
        n_vae = n_dep = 0
        t_ep = time.time()
        for batch in dl_fit:
            x = batch[0].to(device, non_blocking=True)
            f = batch[1].to(device, non_blocking=True)
            subj = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            aux = batch[3:]
            a_vae = a_dep = None
            k = 0
            if has_vae:
                a_vae = aux[k].to(device, non_blocking=True)
                k += 1
            if has_depth:
                a_dep = aux[k].to(device, non_blocking=True)

            if model.struct is None:
                z_e, z_i, _w = model(x, f, subj, training=True)
                sd = None
            else:
                out = model.forward_all(x, subj, training=True)
                z_e = out["z"]
                z_i = model.encode_image(f)
                sd = out["struct"]

            # Complementary weighting, as in SAMGA: mmd_w*MMD + (1-mmd_w)*contrastive.
            # A plain sum would scale the total gradient magnitude with mmd_w and
            # change the effective learning rate between epochs. The structural terms
            # are added OUTSIDE that weighting on purpose: the MMD schedule is a
            # statement about the shared semantic geometry, and letting it also scale
            # the latent regression would mean the structure tower's effective LR
            # depended on a hyper-parameter that has nothing to do with it.
            loss = contrast_w * criterion(z_e, z_i)
            if mmd_w > 0:
                loss = loss + mmd_w * mmd_rbf(z_e, z_i)
            tot_sem += float(loss.detach())
            if sd is not None:
                if a_vae is not None:
                    l_vae = latent_l1(sd["vae"], a_vae)
                    loss = loss + args.w_vae * l_vae
                    tot_vae += float(l_vae.detach())
                    n_vae += 1
                if a_dep is not None:
                    l_dep = F.l1_loss(sd["depth"], a_dep) + args.lambda_grad * grad_l1(sd["depth"], a_dep)
                    loss = loss + args.w_depth * l_dep
                    tot_dep += float(l_dep.detach())
                    n_dep += 1

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            sched.step()

            with torch.no_grad():
                sim = F.normalize(z_e, dim=-1) @ F.normalize(z_i, dim=-1).t()
                tr_top1 += int((sim.argmax(dim=1) == torch.arange(x.shape[0], device=device)).sum())
            tot += float(loss.detach())
            nb += 1
        n_steps += nb
        t_after_train = time.time()
        t_train_s += t_after_train - t_ep

        t_v = time.time()
        if model.struct is None:
            val = evaluate_selection(model, val_eeg, val_feat, device, l2norm=l2norm,
                                     sweep=not args.no_slot_sweep)
        else:
            val = evaluate_selection_dual(
                model, val_eeg, val_feat, device,
                vae_rows=val_vae, depth_rows=val_depth,
                vae_mean=vae_mean, vae_std=vae_std,
                l2norm=l2norm, sweep=not args.no_slot_sweep)
        t_after_val = time.time()
        t_val_s += t_after_val - t_v

        # ---- diagnostic test trajectory ---------------------------------------
        # Read for DIAGNOSIS ONLY. It is recorded as `test_*_diag` in `history`,
        # printed with a `[diag]` prefix, and consumed by nothing: `sel` below is
        # computed before this runs and does not see it, no checkpoint is written
        # here, and the reported headline stays the val-selected one. Picking a
        # checkpoint on this curve would be test-set over-selection, which is the
        # single bias the val split exists to prevent, so keeping the two paths
        # physically separate in the code is the point rather than caution.
        #
        # Why it is worth the forward passes: whether a concept-level holdout is a
        # valid PROXY for the test set is an assumption, not a fact, and it fails
        # whenever the two differ in ways that matter. Here they differ in one
        # measurable way already -- the holdout EEG averages 4 repetitions of each
        # image and the test EEG averages 80 -- so "validation peaked at epoch 16"
        # and "the test set peaked at epoch 16" are two different claims, and only
        # the second one decides whether the remaining 84 epochs were wasted.
        test_diag = None
        if args.test_every and (epoch % args.test_every == 0 or epoch == args.epochs):
            test_diag = evaluate(model, te_eeg, img_te, device, l2norm=l2norm)
            # `evaluate` switches to eval mode and leaves it there; restore the
            # training mode or the next epoch trains with dropout disabled.
            model.train()
            print(f"[diag] ep {epoch:3d} TEST {config.TEST_WAY}-way (diagnosis only, "
                  f"NOT used for selection) top1 {test_diag['top1']:.2f} "
                  f"top5 {test_diag['top5']:.2f} rank {test_diag['mean_rank']:.1f}",
                  flush=True)

        # One scalar to compare checkpoints on. The semantic term is the 0-100
        # Top-1; the two structural terms are converted into the same 0-100-ish
        # units so their weights read as "how many points of the scale this is
        # worth" rather than as a unitless sum whose balance is invisible.
        sel = val["top1"]
        if val.get("vae_top1") is not None:
            sel += args.struct_sel_w * val["vae_top1"]
        if val.get("depth_pearson") is not None:
            sel += args.depth_sel_w * val["depth_pearson"]

        rec = {"epoch": epoch, "loss": tot / max(1, nb), "sel": sel,
               "loss_sem": tot_sem / max(1, nb),
               "loss_vae": (tot_vae / n_vae) if n_vae else None,
               "loss_depth": (tot_dep / n_dep) if n_dep else None,
               # Diagnostic-only columns. They are absent (None) unless
               # `--test-every` was given, and `sel`/`best` never read them.
               "test_top1_diag": (test_diag["top1"] if test_diag else None),
               "test_top5_diag": (test_diag["top5"] if test_diag else None),
               "test_mean_rank_diag": (test_diag["mean_rank"] if test_diag else None),
               "train_top1_inbatch": 100.0 * tr_top1 / max(1, len(ds_fit) // args.batch_size * args.batch_size),
               "val_top1": val["top1"], "val_top5": val["top5"], "val_mean_rank": val["mean_rank"],
               "val_top1_std": val["top1_std"], "val_slots": val["n_slots"],
               "val_vae_top1": val.get("vae_top1"), "val_vae_cos": val.get("vae_cos"),
               "val_depth_pearson": val.get("depth_pearson"),
               "mmd_w": mmd_w, "contrast_w": contrast_w,
               "train_s": round(t_after_train - t_ep, 2),
               "val_s": round(t_after_val - t_v, 2)}
        hist.append(rec)
        extra = ""
        if val.get("vae_top1") is not None:
            extra += f" | vae_top1 {val['vae_top1']:.1f}+-{val['vae_top1_std']:.1f}"
            if val.get("vae_cos") is not None:
                extra += f" vae_cos {val['vae_cos']:+.3f}"
        if val.get("depth_pearson") is not None:
            extra += f" | depth_r {val['depth_pearson']:.3f}"
        print(f"[ep {epoch:3d}/{args.epochs}] loss {rec['loss']:.4f} "
              f"train_top1(batch) {rec['train_top1_inbatch']:.2f} "
              f"val_top1 {val['top1']:.2f}+-{val['top1_std']:.2f} top5 {val['top5']:.2f} "
              f"rank {val['mean_rank']:.0f} mmd_w {mmd_w:.2f}{extra} "
              f"(sel {sel:.2f}) "
              f"({rec['train_s']:.0f}s train / {rec['val_s']:.0f}s val)", flush=True)

        if sel > best["sel"]:
            best = {"sel": sel, "top1": val["top1"], "top5": val["top5"],
                    "mean_rank": val["mean_rank"], "top1_std": val["top1_std"],
                    "vae_top1": val.get("vae_top1"), "vae_cos": val.get("vae_cos"),
                    "depth_pearson": val.get("depth_pearson"), "epoch": epoch}
            epochs_no_gain = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                       out_dir / f"{tag}_best.pt")
        elif in_stage1:
            # Do not let stage 1 exhaustion trigger an early stop. While MMD
            # dominates (weight 0.9 -> 0.2) the contrastive objective is scaled
            # down to almost nothing, so val Top-1 can legitimately stagnate for
            # the whole of stage 1. Counting that as "no improvement" would kill
            # the run before stage 2 -- where the actual retrieval gain happens --
            # ever begins.
            epochs_no_gain = 0
        else:
            epochs_no_gain += 1
            if args.patience and epochs_no_gain >= args.patience:
                print(f"[stop] no val improvement for {args.patience} epochs; "
                      f"stopping at epoch {epoch} (best {best['top1']:.2f} @ ep {best['epoch']})")
                stopped_early = True
                break

    wall = time.time() - t0
    print(f"\n[timo] wall {wall:.0f}s = train {t_train_s:.0f}s + val {t_val_s:.0f}s "
          f"+ {wall - t_train_s - t_val_s:.0f}s overhead; "
          f"{n_steps} steps, {1000 * t_train_s / max(1, n_steps):.0f} ms/step")

    # ---------------- single test scoring
    ckpt = torch.load(out_dir / f"{tag}_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, te_eeg, img_te, device, l2norm=l2norm)

    # ---------------- diagnostic test trajectory
    # The summary of `--test-every`. Its whole purpose is to settle whether the
    # val holdout is a faithful proxy for the test set, so it reports the two
    # peak locations side by side: if test peaks at the same epoch as val, then
    # "train for N epochs and select on val" is sound and the extra epochs are
    # simply wasted; if test keeps climbing while val decays, the holdout is
    # mis-calibrated and the selection rule has to be fixed before anything else.
    # Reported, never acted on -- see the note where the curve is collected.
    test_diag = None
    diag = [(r["epoch"], r["test_top1_diag"]) for r in hist
            if r.get("test_top1_diag") is not None]
    if diag:
        t_peak_ep, t_peak_val = max(diag, key=lambda t: t[1])
        v_peak_ep = best["epoch"]
        print(f"\n[diag] test trajectory ({len(diag)} points, every "
              f"{args.test_every} epochs) -- diagnosis only, not selection:")
        for e, v in diag:
            print(f"[diag]   ep {e:>3}  test {v:>6.2f}%"
                  + ("   <- test peak" if e == t_peak_ep else "")
                  + ("   <- val peak" if e == v_peak_ep else ""))
        lag = t_peak_ep - v_peak_ep
        verdict = ("SAME peak location: the holdout is a faithful proxy and the "
                   "post-peak epochs are wasted time, not lost accuracy"
                   if abs(lag) <= max(1, args.test_every // 2) else
                   f"peaks are {lag:+d} epochs apart: the holdout is NOT tracking the "
                   f"test set, so the selection rule needs fixing before the recipe does")
        print(f"[diag] test peaks ep {t_peak_ep} ({t_peak_val:.2f}%) vs val peaks "
              f"ep {v_peak_ep} ({best['top1']:.2f}%); val-selected ckpt scored "
              f"{test['top1']:.2f}% on test")
        print(f"[diag] {verdict}")
        test_diag = {
            "curve": [{"epoch": e, "test_top1": v} for e, v in diag],
            "test_peak_epoch": int(t_peak_ep), "test_peak_top1": float(t_peak_val),
            "val_peak_epoch": int(v_peak_ep), "val_peak_top1": float(best["top1"]),
            "peak_lag_epochs": int(lag), "verdict": verdict,
            "note": "diagnostic only; this run's reported number is val-selected",
        }

    # ---------------- fit diagnostic
    # Separates "cannot fit the training set" (an optimisation or capacity problem)
    # from "fits but does not generalise" (a regularisation problem). Without this
    # the two are indistinguishable from the val curve alone. Uses the identical
    # protocol as validation so the two numbers are directly comparable, and only
    # FIT concepts -- never the val holdout, never the test split.
    fit_diag = None
    if args.fit_diagnostic:
        rng = np.random.default_rng(args.split_seed)
        pool = split.fit_concepts
        n_take = min(args.fit_diag_concepts, len(pool))
        pick = np.sort(rng.choice(pool, size=n_take, replace=False))
        fd = evaluate_selection(model, tr_eeg[pick], img_tr[pick], device,
                                l2norm=l2norm, sweep=not args.no_slot_sweep)
        fit_diag = {"top1": fd["top1"], "top5": fd["top5"], "n_concepts": int(n_take),
                    "n_slots": int(fd["n_slots"]), "top1_std": fd["top1_std"],
                    "val_top1": best["top1"], "gap_vs_val": fd["top1"] - best["top1"],
                    "protocol": "identical to validation, but on FIT concepts"}
        fit_diag["verdict"] = (
            "the model FITS the training set (fit is far above val), so the binding "
            "constraint is GENERALISATION, not capacity or optimisation -- regularise "
            "or reduce capacity; more steps will not help"
            if fit_diag["gap_vs_val"] > 20.0 else
            "the model does NOT fit its own training set (fit is close to val), so the "
            "binding constraint is OPTIMISATION or capacity, not generalisation -- more "
            "steps, a higher LR, or fewer frozen blocks"
        )
        print(f"[fit  ] fit-concept ceiling (same protocol as val, n={n_take}): "
              f"top1 {fd['top1']:.2f} top5 {fd['top5']:.2f} "
              f"| val {best['top1']:.2f} | test {test['top1']:.2f}")
        print(f"[fit  ] gap fit-val {fit_diag['gap_vs_val']:+.1f} points -> {fit_diag['verdict']}")

    def _sel_text() -> str:
        """One sentence naming exactly what 'best checkpoint' meant in this run.

        Recorded rather than assumed: the single-tower runs selected on val Top-1
        alone, and a result file that does not say which criterion produced the
        checkpoint cannot be compared against one that used a composite.
        """
        std = best.get("top1_std")
        std_txt = f"{std:.2f} std across image slots" if std is not None else "no spread recorded"
        if model.struct is None:
            return (f"best val top1 on concept-level holdout ({std_txt}); "
                    f"test scored once")
        return (f"best selection score on concept-level holdout: "
                f"val_top1 + {args.struct_sel_w:g}*val_vae_top1 + "
                f"{args.depth_sel_w:g}*val_depth_pearson ({std_txt}); test scored once")

    result = {
        "tag": tag,
        "subject": args.subject,
        "backbone": args.backbone,
        "layers": args.layers,
        "channels": args.channels,
        "n_channels": len(ch_names),
        "epochs": args.epochs,
        "epochs_run": len(hist),
        "stopped_early": stopped_early,
        "freeze_blocks": args.freeze_blocks,
        "freeze_all": args.freeze_all,
        "aug": args.aug,
        "softplus": args.softplus,
        "target_features": args.target_features,
        "target_layer": args.target_layer,
        "target_layers": keys,
        "target_fusion": args.target_fusion,
        "target_weights": (None if model.target_weights() is None
                           else [round(float(v), 5) for v in model.target_weights()]),
        "fusion_mode": args.fusion_mode,
        "eeg_layer_weights": [round(float(v), 5) for v in model.fusion.layer_weights()],
        # --- interface / optimisation provenance (added with the NW-v8 fixes) ---
        "tokenizer": args.tokenizer,
        "patch_size": args.patch_size,
        "patch_style": args.patch_style,
        "n_patches_w": args.n_patches_w,
        "n_tokens": int(model.encoder.tokenizer.n_tokens
                        if args.tokenizer == "grid"
                        else model.encoder.tokenizer.n_regions * args.n_patches_w),
        "patch_grid": (list(model.encoder.dst_grid)),
        "eeg_image": [int(model.encoder.tokenizer.height),
                      int(model.encoder.tokenizer.width)],
        "uses_pretrained_patch_embed": args.tokenizer == "eegit",
        "head_kind": args.head_kind,
        "img_head_kind": args.img_head_kind,
        "head_drop": args.head_drop,
        "timm_global_pool": args.timm_global_pool,
        "optimizer": args.optimizer,
        "wd_all_params": args.wd_all_params,
        "eeg_l2norm_in_loss": not args.no_eeg_l2norm,
        "pool_norm": not args.no_pool_norm,
        "cls_token_prefix": not args.no_cls_token,
        "zscore": (not args.no_zscore) and args.tokenizer == "eegit",
        "n_time_windows": (args.n_time_windows if args.tokenizer == "grid" else None),
        "lr": args.lr,
        "backbone_lr_mult": args.backbone_lr_mult,
        "backbone_lr": lr_backbone,
        "warmup_epochs": args.warmup_epochs,
        "cosine": args.cosine,
        "min_lr_ratio": args.min_lr_ratio,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "temp_learnable": not args.fixed_temp,
        "temp_init": 0.07,
        "lr_groups": lr_groups_cfg,
        "lr_groups_final": {g.get("group"): g["lr"] for g in optim.param_groups},
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "fit_diagnostic": fit_diag,
        "image_dim": int(img_tr.shape[-1]),
        "structure_tower": (
            None if model.struct is None else {
                "backbone": args.struct_backbone,
                "layers": args.struct_layers,
                "fusion_mode": args.struct_fusion_mode,
                "eeg_layer_weights": [round(float(v), 5)
                                      for v in model.struct.fusion.layer_weights()],
                "patch_size": args.struct_patch_size,
                "n_patches_w": args.struct_n_patches_w,
                "n_tokens": int(model.struct.encoder.tokenizer.n_tokens),
                "patch_grid": list(model.struct.encoder.dst_grid),
                "eeg_image": [int(model.struct.encoder.tokenizer.height),
                              int(model.struct.encoder.tokenizer.width)],
                "field_shape": [args.struct_field_ch, model.struct.out_hw,
                                model.struct.out_hw],
                "out_shape": [model.struct.vae_ch, model.struct.out_hw,
                              model.struct.out_hw],
                "lr_mult": args.struct_lr_mult,
                "head_lr_mult": args.struct_head_lr_mult,
                # The LRs actually handed to the optimizer. Recorded alongside the
                # multipliers because the absolute overrides win over them, so the
                # multipliers alone do not say what was trained.
                "lr": lr_struct,
                "head_lr": lr_struct_head,
                "lr_abs": args.struct_lr,
                "head_lr_abs": args.struct_head_lr,
                "freeze_blocks": args.struct_freeze_blocks,
                "targets": {
                    "vae_latents": (str(args.vae_latents) if args.vae_latents else None),
                    "vae_shape": (list(int(s) for s in vae_tr.shape[1:])
                                  if vae_tr is not None else None),
                    "vae_normalisation": (
                        None if vae_mean is None else {
                            "source": "fit concepts only",
                            "mean": [round(float(v), 5) for v in vae_mean],
                            "std": [round(float(v), 5) for v in vae_std]}),
                    "depth_train": args.depth_train,
                    "depth_shape": (list(int(s) for s in depth_tr.shape[1:])
                                    if depth_tr is not None else None),
                    "w_vae": args.w_vae, "w_depth": args.w_depth,
                    "lambda_grad": args.lambda_grad,
                },
                "selection": {
                    "score": ("val_top1 + %.2f*val_vae_top1 + %.2f*val_depth_pearson"
                              % (args.struct_sel_w, args.depth_sel_w)),
                    "struct_sel_w": args.struct_sel_w,
                    "depth_sel_w": args.depth_sel_w,
                    "vae_top1_note": "centred cosine, n_concepts-way, chance = 1/n",
                },
                "params": {
                    "struct_blocks": sum(p.numel() for _, p in g_s_blocks),
                    "struct_interface": sum(p.numel() for _, p in g_s_interface),
                    "struct_heads": sum(p.numel() for _, p in g_s_heads),
                },
            }),
        "stage1_epochs": args.stage1_epochs,
        "stage2_lr": lr_stage2,
        "mmd_start": args.mmd_start,
        "mmd_end": args.mmd_end,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "trainable_params": tr_p,
        "best_val": best,
        "test": test,
        "test_diag": test_diag,
        "protocol": {
            "way": config.TEST_WAY,
            "rep_averaging": "train 4, test 80",
            "selection": _sel_text(),
            "val_way": int(val["n"]),
            "val_slots_swept": int(val["n_slots"]),
            "metric_impl": "SAMGA module/util.py retrieve_all (verbatim)",
        },
        "timing": {
            "wall_seconds": round(wall, 1),
            "train_seconds": round(t_train_s, 1),
            "val_seconds": round(t_val_s, 1),
            "steps": n_steps,
            "ms_per_step": round(1000 * t_train_s / max(1, n_steps), 1),
        },
        "history": hist,
    }
    # The published line this run is trying to beat, plus the local floor it must
    # clear before any number is worth interpreting (see docs section 0.5).
    #
    # NOTE the reference is the SUB-08 column, not the 10-subject average. This run
    # is single-subject, and subject 8 sits ABOVE the average for every method in
    # SAMGA's Table 2 (NICE +6.8, ATM +11.7, UBP +7.7, SAMGA +3.5), so quoting the
    # average understated the gap by ~3.5 points.
    result["reference_sota"] = {
        "SAMGA_intra_sub08": {"top1": 94.8, "top5": None},
        "SAMGA_intra_10subj_avg": {"top1": 91.3, "top5": 98.8},
        "EEGiT_intra_10subj_avg": {"top1": 70.4, "top5": 95.1,
                                   "note": "same backbone as this run; the patch "
                                           "representation is its +16.4 ablation arm"},
        "ShallowAlignment_sub08": {"top1": 86.9},
        "linear_ridge_same_split": RIDGE_REF,
    }

    (out_dir / f"{tag}_result.json").write_text(json.dumps(result, indent=2))
    print(f"\n[TEST] top1 {test['top1']:.2f}  top5 {test['top5']:.2f}  "
          f"mean_rank {test['mean_rank']:.1f}  (n={test['n']}, {config.TEST_WAY}-way)")
    ridge_msg = (f"  ridge floor {RIDGE_REF['test_top1']:.2f} -> "
                 f"{'PASS' if test['top1'] > RIDGE_REF['test_top1'] else 'FAIL'}") if RIDGE_REF else ""
    print(f"[ref ] SAMGA intra SUB-08 94.8 (10-subject avg 91.3) | "
          f"EEGiT intra avg 70.4 (same backbone){ridge_msg}")
    print(f"[done] {out_dir / f'{tag}_result.json'}")


if __name__ == "__main__":
    main()
