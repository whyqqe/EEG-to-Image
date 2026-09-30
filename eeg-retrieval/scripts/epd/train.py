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

from epd import config
from epd.augment import AUG_NAMES, build_aug
from epd.data import AuxTargetDataset, TestDataset, TrainDataset, concept_split, load_subject
from epd.encoders import backbone_n_blocks, backbone_patch_size
from epd.losses import (
    InfoNCE,
    latent_l1,
    latent_mse,
    mmd_rbf,
    stimulus_groups_from_ids,
    variance_floor,
)
from epd.metrics import mean_rank, retrieval_per_concept, retrieval_report
from epd.model import RetrievalModel, build_from_args


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
    p.add_argument("--val-concepts", type=int, default=150,
                    help="Training concepts held out for checkpoint selection. "
                         "**0 is the SOTA protocol, not a degenerate setting.** "
                         "Shallow Alignment: 'the main experiments follow the "
                         "standard protocol and train on the full training set "
                         "without a validation split'; SCORE and Shallow Alignment "
                         "both report the FINAL epoch. So 0 has to be paired with "
                         "--select-last -- with a non-zero cost, since with no "
                         "holdout there is no selection signal at all. Valid values "
                         "are 0 or >= 2 (a 1-concept holdout is a coin flip that "
                         "still looks like a metric).")
    p.add_argument("--split-seed", type=int, default=2025)

    # ---- inter-subject (LOSO) ------------------------------------------------
    # Enough of the SOTA inter-subject protocol to be a drop-in addition rather
    # than a second training script: `--source-subjects` turns the run from
    # "one subject" into "nine subjects, one held out", and nothing else in the
    # pipeline has to change. See PROTOCOL_INTER.md for the transcription of the
    # paper the defaults come from.
    p.add_argument("--source-subjects", type=int, nargs="*", default=None,
                    help="Inter-subject (LOSO) training subjects. When given, the "
                         "fit set becomes the CONCATENATION of these subjects' "
                         "training trials -- one row per (subject, concept) -- each "
                         "subject z-scored with its OWN training statistics, and "
                         "every row carries its subject id into `LayerFusion` so the "
                         "per-subject embedding is actually trained. The order "
                         "defines the subject index, so it is part of the config: "
                         "a run's `subject_residual` row 3 means 'the 4th subject "
                         "listed here', not 'sub-04'. Requires --target-subject.")
    p.add_argument("--target-subject", type=int, default=None,
                    help="The subject held out for test. Must NOT appear in "
                         "--source-subjects; that is asserted rather than trusted, "
                         "because a fold that accidentally trains on its own test "
                         "subject reports the highest number in the sweep and is the "
                         "one result nobody re-checks. Only that subject's "
                         "per-channel SCALE is read (label-free), never its pairs.")
    p.add_argument("--select-last", action="store_true",
                    help="Report the FINAL epoch instead of the best-validation one. "
                         "This is the leak-free, comparable policy the inter-subject "
                         "papers use -- SCORE: 'train each model for 50 epochs, and "
                         "report the final epoch'. It is also the only policy "
                         "available once --val-concepts 0 removes the holdout, so "
                         "the two flags are really one decision. Implies no early "
                         "stopping and no EMA-based selection; the last epoch's "
                         "weights are what gets saved and scored.")

    # ---- preprocessing: MVNN -------------------------------------------------
    # Every inter-subject paper on this benchmark applies MVNN, and this pipeline
    # never has. See epd/mvnn.py for the method and for why the fit split is a
    # protocol decision rather than an implementation detail.
    p.add_argument("--mvnn", choices=["off", "train", "test"], default="off",
                   help="Multivariate noise normalisation. 'off' leaves the cached "
                        "arrays alone (the historical behaviour, kept so the ablation "
                        "is one flag). 'train' whitens from each subject's own "
                        "labelled TRAINING residuals, which is what the literature's "
                        "'MVNN is applied to the training data' means and is the "
                        "correct choice for the nine source subjects. 'test' whitens "
                        "from the trial's own within-condition test residuals -- the "
                        "only option for the held-out subject, whose training split "
                        "the fold excludes; it is label-free in the same sense as "
                        "averaging repetitions, which the standard protocol already "
                        "requires. Under LOSO the two roles are chosen per subject "
                        "automatically.")
    p.add_argument("--mvnn-shrinkage", choices=["lw", "fixed"], default="lw",
                   help="'lw' = Ledoit-Wolf towards the identity in correlation "
                        "space, no tuning and so no validation set needed -- which "
                        "matters because --val-concepts 0 leaves none. 'fixed' pins "
                        "the intensity, so 0.0 reaches the unshrunk end for an "
                        "ablation.")
    p.add_argument("--mvnn-fixed", type=float, default=0.1,
                   help="The intensity --mvnn-shrinkage fixed uses.")
    p.add_argument("--mvnn-max-cond", type=int, default=0,
                   help="Fit the whitener from only the first N conditions. A smoke "
                        "override: the whitener is cached under a key that includes "
                        "it, so a subsampled fit can never be mistaken for a full one.")

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
                    choices=["single", "mean", "routed", "routed_sr"],
                    help="how to combine several target layers. 'mean' pins equal "
                         "weights (the doc's '先均匀融合': do these layers carry "
                         "complementary information at all), 'routed' learns a "
                         "subject-INDEPENDENT blend, 'routed_sr' learns the blend with a "
                         "per-subject residual ('再上可学习路由'). Only meaningful with "
                         "--target-layers. 'routed_sr' is SAMGA Eq. 4/7 -- subject-aware "
                         "training, subject-agnostic inference, since the residual is "
                         "dropped at test -- and it is the option that makes `--multipos` "
                         "a real objective rather than a no-op, because it is the only "
                         "one that gives different subjects different target VECTORS "
                         "for the same picture. See model.build_from_args.")
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
    p.add_argument("--recovery", action="store_true",
                    help="SCORE Eq. 3-9: after the frozen test evaluation, run the "
                         "label-free coordinate-recovery ladder on the SAME feature "
                         "matrix (CSLS -> moment match -> orthogonal recovery -> "
                         "identity regularisation) and record all four steps. This is "
                         "CPU post-processing on frozen features and is worth +23.90 of "
                         "SCORE's +27.01 points, so it is reported even when it does "
                         "not help -- see --rec-min-landmark-rate for the abstention.")
    p.add_argument("--rec-rho", type=float, default=0.1,
                   help="identity-regularisation strength for Eq. 6, dimensionless "
                        "(SCORE: 0.1). 0.0 gives unregularised Procrustes, which is the "
                        "'+ recovery' row of their Table 4; 0.1 is the '+identity "
                        "regularization' row and is worth +2.25 Top-1 over it.")
    p.add_argument("--rec-k", type=int, default=10, help="CSLS neighbours (SCORE: 10)")
    p.add_argument("--rec-max-landmarks", type=int, default=160,
                   help="cap on the landmark count (SCORE uses 12 to 160)")
    p.add_argument("--rec-min-landmark-rate", type=float, default=0.0,
                   help="abstain from recovery when the fraction of queries that became "
                        "mutual nearest-neighbour landmarks is below this. Recovery is "
                        "an active LOSS past its threshold -- a map fitted from wrong "
                        "correspondences is a real rotation in the wrong direction -- "
                        "and `test_epd_recover.py` measures the two regimes at 0.90 and "
                        "0.55. 0.0 disables the gate.")
    p.add_argument("--multipos", action="store_true",
                    help="SCORE Eq. 1: rows that share a stimulus are POSITIVES of each "
                         "other, not negatives. This is not an optional refinement in a "
                         "LOSO fold -- `expand_loso_images` tiles each image feature once "
                         "per source subject, so nine rows in a batch are the same "
                         "picture and the pairwise objective spends its gradient pushing "
                         "those nine subjects apart. Off: the pairwise baseline, which is "
                         "the correct setting for an intra-subject run and the "
                         "slightly-wrong one here. SCORE measures +2.41 Top-1 for the "
                         "swap. Degenerates exactly to the pairwise loss when no two rows "
                         "share a stimulus, so it is safe to leave on.")
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
    p.add_argument("--ema-decay", type=float, default=0.0,
                    help="exponential moving average of the model WEIGHTS, decay per "
                         "step. 0 disables (default). When enabled the averaged weights "
                         "are what gets evaluated on validation, what checkpoint "
                         "selection ranks, and what gets saved -- the raw weights are "
                         "recorded but never shipped. This is two fixes at once, both "
                         "required before any architecture comparison in this project "
                         "can be believed. (1) SELECTION: the 6-arm sweep showed the "
                         "validation peak moving by up to 28 epochs between settings "
                         "whose test scores are within 2 sigma of each other (sigma is "
                         "3.5 points on the 200-way test, 1.2 on the 1500-trial "
                         "validation, so 'best epoch' was being read off a curve whose "
                         "noise was comparable to its shape). Averaging the weights "
                         "smooths the curve that the argmax is taken over, which is "
                         "exactly the fix for picking an epoch off noise. (2) "
                         "GENERALISATION: the same run measured fit-top1 86.73 against "
                         "holdout 33.67 at an identical protocol, so ~53 points of "
                         "capacity were going into memorisation, and EMA is a standard "
                         "regulariser against precisely that. Typical values: 0.999 "
                         "over ~7k steps gives a ~1000-step window (about 8-9 epochs "
                         "here); 0.9999 will not have warmed up by the end of a "
                         "60-epoch run at this step count.")
    p.add_argument("--ema-warmup-steps", type=int, default=0,
                    help="skip the EMA update for the first N steps, so the average is "
                         "not dragged toward a randomly-initialised model. 0 = start "
                         "immediately from the init weights, which is the convention in "
                         "timm/DINO/MAE and is fine given the decay.")
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
                    help="EEG augmentation for the training split only (see epd.augment)")

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
    p.add_argument("--patch-style", type=str, default="region-time",
                    choices=["region-time", "time-region",
                             "nw", "eegit_official"],  # legacy aliases
                    help="how the EEG patch image is laid out, named for which axis "
                         "goes where. 'time-region' reproduces the released EEGiT code "
                         "exactly: H = time (resampled 250 -> n_patches_w*patch_size), "
                         "W = regions, regions anterior -> posterior in the dataset's "
                         "own channel order, and ONE 2D bilinear F.interpolate over "
                         "the (time, electrode) plane per region. 'region-time' is the "
                         "other interface (H = regions, W = time, posterior -> "
                         "anterior, montage-x sorted, 1D interpolation), which every "
                         "result produced before this flag existed was trained on. The "
                         "old names 'nw' and 'eegit_official' are accepted as aliases "
                         "because they are in every saved config and result file; the "
                         "old names said nothing about the layout, only about which "
                         "project or paper it came from.")
    p.add_argument("--no-zscore", action="store_true",
                    help="skip the per-channel z-score that puts EEG into the value "
                         "range the pretrained patch_embed expects")

    # ---- head ---------------------------------------------------------------
    p.add_argument("--head-kind", type=str, default="mlp", choices=["mlp", "eegit", "nw"],
                    help="'eegit' replaces the EEG projection head with the released "
                         "code's ProjectionHead: Linear -> GELU -> Linear -> "
                         "Dropout(0.5) -> + the PRE-GELU projection -> LayerNorm. It "
                         "is not the same block as the 'mlp' 1-hidden-layer MLP; the "
                         "residual takes the projection, not the second linear.")
    p.add_argument("--img-head-kind", type=str, default="mlp", choices=["mlp", "eegit", "nw"],
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
                         "Structure-focused priors: timm:dinov3_b16 (self-supervised "
                         "with Gram anchoring, RoPE, patch 16, 768-d, 12 blocks) or "
                         "timm:mae_b16 (masked-pixel reconstruction, patch 16, 768-d, "
                         "12 blocks) as its control.")
    st.add_argument("--struct-arch", type=str, default="vit", choices=("vit", "da2"),
                    help="which structural tower to build. 'vit' is `StructureTower`: "
                         "a pretrained trunk plus a decoder this project trains from "
                         "scratch, emitting VAE latents. 'da2' is `DepthTower`: Depth "
                         "Anything V2 whole, with only its input interface replaced by "
                         "EEG patches, emitting a depth field -- the condition "
                         "ControlNet-depth was trained on. The two are not settings of "
                         "one architecture: 'da2' has no --struct-layers, no fusion and "
                         "no --struct-field-ch, because it inherits a pretrained "
                         "features-to-depth path instead of learning one.")
    st.add_argument("--struct-out-hw", type=int, default=64,
                    help="the resolution the structural loss is scored at. For 'da2' "
                         "this is the COARSE scale the probe found reachable (see "
                         "`probe_targets.py --center-spatial`): the margin is flat from "
                         "4x4 to 64x64 while the dimension count rises 256x, so scoring "
                         "finer puts almost all of the loss on coordinates the EEG "
                         "cannot address.")
    st.add_argument("--da2-model", type=str, default="",
                    help="the Depth Anything V2 checkpoint. Defaults to the Small "
                         "release, which is the model the depth targets were built "
                         "with -- using a different depth model for the target and the "
                         "trunk would put a systematic bias between them that no EEG "
                         "side work can remove.")
    st.add_argument("--da2-allow-download", action="store_true",
                    help="permit a depth checkpoint that is not already in HF_HOME. Off "
                         "by default so a typo'd model id fails immediately instead of "
                         "reaching the network mid-run.")
    st.add_argument("--struct-layers", type=int, nargs="+", default=[8, 10, 12],
                    help="structure-trunk blocks to fuse (1-based). Bounds are "
                         "validated against the trunk's actual depth. Unused by "
                         "--struct-arch da2, which fuses inside its own pretrained neck.")
    st.add_argument("--struct-patch-size", type=int, default=16,
                    help="must equal the structure backbone's patch_embed kernel; "
                         "validated against BACKBONES in resolve_target_plan. For "
                         "--struct-arch da2 it must be 14, the depth conv's kernel.")
    st.add_argument("--struct-n-patches-w", type=int, default=14,
                    help="unused by the topography tokenizer (its geometry comes from "
                         "--struct-scalp-res / --struct-n-time-bands); kept because it "
                         "is the time-axis extent of the EEGiT-geometry path.")
    st.add_argument("--struct-tokenizer", type=str, default="topography",
                    choices=("topography", "eegit", "grid"),
                    help="the structure tower's input geometry. 'topography' builds a "
                         "genuine 2D scalp map per time band, so both spatial axes "
                         "survive and the token grid is spatially meaningful; 'eegit' "
                         "reuses the semantic tower's region-band geometry (no "
                         "left/right axis).")
    st.add_argument("--struct-scalp-res", type=int, default=64,
                    help="topography resolution (pixels per spatial axis, must be a "
                         "multiple of --struct-patch-size).")
    st.add_argument("--struct-n-time-bands", type=int, default=3,
                    help="contiguous time bands, each becoming one scalp map; the maps "
                         "stack along the image's height axis.")
    st.add_argument("--struct-band-channels", type=str, default="replicate",
                    choices=("replicate", "moments"),
                    help="'replicate' copies the band topography across the 3 input "
                         "channels (EEGiT's own convention); 'moments' spends them on "
                         "mean / std / |first difference| to carry temporal dynamics.")
    st.add_argument("--struct-fusion-mode", type=str, default="uniform",
                    choices=("uniform", "routed"))
    st.add_argument("--struct-lr-mult", type=float, default=0.1,
                    help="LR multiplier for the structure trunk's blocks, relative to "
                         "--lr. Same convention as --backbone-lr-mult: the pretrained "
                         "trunk wants a smaller step than the randomly-initialised head.")
    st.add_argument("--struct-head-lr-mult", type=float, default=1.0,
                    help="LR multiplier for the structure in-heads (grid mix, "
                         "upsampling blocks, vae conv).")
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
                    help="channels of the spatial field the token grid decodes into.")
    st.add_argument("--struct-base-hw", type=int, default=8,
                    help="seed resolution of the POOLED decoder, which is the branch "
                         "every non-topography tokenizer uses. `proj` emits "
                         "base_ch*base_hw**2 channels reshaped to "
                         "(B,base_ch,base_hw,base_hw) and three x2 `_up_block`s carry "
                         "it to `--struct-out-hw`, so the ONLY legal combination is "
                         "`out_hw == base_hw * 8`; `StructureTower` raises otherwise. "
                         "With the default 8 the sole legal `out_hw` is 64, i.e. the "
                         "fine target -- to score a COARSE target with the eegit "
                         "geometry, lower this with `out_hw` (8x8 wants base_hw 1). "
                         "The default is unchanged, so existing configs construct "
                         "byte-identically.")
    st.add_argument("--struct-field-ch", type=int, default=32)
    st.add_argument("--struct-vae-ch", type=int, default=4,
                    help="channels of the structural target: 4 for SDXL VAE latents, 1 "
                         "for depth. Read by both architectures, which is why it is a "
                         "named argument and not a `struct_cfg` entry -- `struct_cfg` "
                         "is splatted only into `StructureTower`, so a da2 config would "
                         "have silently kept 4 and failed on a shape mismatch inside "
                         "the loss rather than at construction.")

    # ---- structure targets ---------------------------------------------------
    tg = p.add_argument_group("structure targets",
        "Auxiliary per-image targets for the structure tower, keyed by "
        "concept_index*10+slot, which is the order the EEG arrays are already in.")
    tg.add_argument("--vae-latents", type=str, default=None,
                    help="directory holding train_vae_latents_f16.npy / "
                         "test_vae_latents_f16.npy, (N,4,64,64) float16, ALREADY "
                         "multiplied by the VAE's scaling_factor. Used when "
                         "--struct-target vae and --struct-scale 64.")
    tg.add_argument("--depth-cache", type=str, default=None,
                    help="directory holding train_depth_64.npy / test_depth_64.npy, "
                         "(N,64,64) float32, the monocular depth of each training "
                         "image. Used when --struct-target depth and --struct-scale 64.")
    tg.add_argument("--coarse-root", type=str, default=None,
                    help="directory holding train_{target}_{scale}.npy / "
                         "test_{target}_{scale}.npy for scale in 4/8/16/32, the "
                         "area-averaged versions of the fine caches. Defaults to "
                         "outputs/struct_targets/coarse.")
    tg.add_argument("--struct-target", type=str, default="vae",
                    choices=("vae", "depth"),
                    help="which structural target the tower regresses.\n"
                         "  vae   -- SDXL VAE latents (4ch). Strongest measured "
                         "signal at coarse scale: r(pred,gt) +0.3672 at 4x4.\n"
                         "  depth -- monocular depth. Weaker (+0.2227 at 8x8) but it "
                         "is the condition ControlNet-depth consumes, so the decoder "
                         "gets an additive spatial residual rather than an img2img "
                         "init -- see `depth_tower.py`.")
    tg.add_argument("--struct-scale", type=int, default=64,
                    choices=(4, 8, 16, 32, 64),
                    help="spatial resolution of the structural target. This is the "
                         "load-bearing knob, and it is a measurement: with the target "
                         "centred on the fit-set per-pixel mean the linear margin is "
                         "flat from 4x4 (+0.2219) to 64x64 (+0.1800) while the "
                         "dimension count rises 256x, so the coarse end is where the "
                         "signal per coordinate is. 64 selects the original fine cache.")
    tg.add_argument("--struct-center", action="store_true",
                    help="subtract the FIT-split per-pixel mean field from the target "
                         "before scaling it, and add it back at export. This is the "
                         "correction for both recorded collapses, and the reason is "
                         "specific: the only normalisation the stack applied was a "
                         "per-channel SCALAR (`vae_mean` is reshaped to (C,1,1)), which "
                         "removes a global offset and leaves every spatial structure "
                         "of the mean in place. Under L1 the conditional median of a "
                         "target whose mean field dominates IS the mean field, so a "
                         "head that learned nothing but the mean was near-optimal, and "
                         "the run reported exactly that: variance ratio 0.0068 with a "
                         "healthy-looking loss. The mean field is a constant every "
                         "decoder gets as a bias; the head should not be paid to "
                         "rediscover it, and the loss should not be dominated by it.")
    tg.add_argument("--w-vae", type=float, default=1.0)
    tg.add_argument("--vae-loss", type=str, default="l1", choices=("l1", "mse"),
                    help="the latent regression's loss. `l1` (default) is the "
                         "conditional median and is what every run through the "
                         "topography interface collapsed under: with a weakly "
                         "predictable target -- and the VAE latent is, at 5.07%% "
                         "instance Top-1 on a 0.50%% chance floor -- the conditional "
                         "median is close to the global mean field, so a near-constant "
                         "prediction is close to L1-optimal and the instance term "
                         "receives almost no gradient. `mse` is the conditional mean "
                         "and is the loss the probe's successful rows were all fitted "
                         "with (a closed-form L2 ridge); see `run_epd_optimal.sh`.")
    # On `depth`, an earlier reading of `probe_targets.py` concluded the target was
    # unreachable -- "r(pred,gt) +0.16 against the constant map's +0.53, margin -0.37
    # on all four arms". That comparison was against an UNCENTRED target, and
    # `pearson_rows` centres each ROW, so the "constant" it beat us with was the
    # fit-set mean depth MAP, which is shape-similar to every COCO depth map for a
    # reason that has nothing to do with EEG. The ridge could not have expressed it
    # anyway: its design matrix is z-scored per feature, so it has mean zero and no
    # intercept. Centring removes the free component and makes the constant
    # predictor the zero vector, whose row r is identically 0; the depth ladder then
    # reads +0.1800 (64x64) to +0.2227 (8x8) on sub-08 with all 63 channels.
    tg.add_argument("--w-var", type=float, default=1.0,
                    help="weight of the variance-floor hinge on the VAE field. 0 "
                         "disables it; the collapsed head this exists to prevent sat "
                         "at a variance ratio of 0.0068 against a closed-form ridge "
                         "baseline of 0.4411.")
    tg.add_argument("--var-margin", type=float, default=0.5,
                    help="fraction of the per-channel target std the prediction must "
                         "reach before the hinge goes quiet. <1 because matching the "
                         "target's global std exactly is stricter than 'did not "
                         "collapse'.")

    # ---- selection -----------------------------------------------------------
    se = p.add_argument_group("checkpoint selection",
        "What 'best' means once the model has two objectives. Both structural "
        "components are in the same 0-100 units as val Top-1 so the weights read as "
        "points of a 200-point scale rather than as an opaque sum.")
    se.add_argument("--struct-sel-w", type=float, default=0.5,
                    help="weight of val VAE-latent retrieval Top-1 (0-100).")

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
    return _validate(p.parse_args())


def _validate(a):
    """Cross-flag checks, run before any data or GPU is touched.

    These are the combinations that are individually legal and jointly nonsensical,
    which is the class of mistake a `--validate-only` dry run exists to catch on the
    login node rather than after an allocation.
    """
    if a.val_concepts < 0:
        raise SystemExit(f"--val-concepts must be >= 0, got {a.val_concepts}")
    if a.val_concepts == 1:
        raise SystemExit(
            "--val-concepts 1 is not a holdout: one concept cannot rank two "
            "checkpoints, so selection is a coin flip that still reports a number. "
            "Use 0 with --select-last, or >= 2.")
    want_loso = a.source_subjects is not None or a.target_subject is not None
    if want_loso and not (a.source_subjects and a.target_subject is not None):
        raise SystemExit(
            "--source-subjects and --target-subject are one setting: giving only "
            "one of them would silently fall back to the intra-subject path "
            f"(got source={a.source_subjects}, target={a.target_subject})")
    if want_loso and a.struct_backbone:
        raise SystemExit(
            "LOSO + the structure tower is not wired yet: the aux-target dataset "
            "indexes its caches by row = concept*10+slot, which is a single-subject "
            "layout, so the task targets would be read from the wrong rows. Run the "
            "inter-subject arms single-tower until AuxTargetDataset takes "
            "subject_of_row.")
    if a.target_subject is not None and not (1 <= a.target_subject <= 10):
        raise SystemExit(f"--target-subject must be 1..10, got {a.target_subject}")
    if a.mvnn_fixed < 0.0 or a.mvnn_fixed > 1.0:
        raise SystemExit(f"--mvnn-fixed must be in [0, 1], got {a.mvnn_fixed}")
    if a.mvnn_max_cond < 0:
        raise SystemExit(f"--mvnn-max-cond must be >= 0, got {a.mvnn_max_cond}")
    if a.source_subjects:
        bad = [s for s in a.source_subjects if not (1 <= s <= 10)]
        if bad:
            raise SystemExit(f"--source-subjects has out-of-range entries {bad} (1..10)")
        if a.target_subject in a.source_subjects:
            raise SystemExit(
                f"--target-subject {a.target_subject} is also in --source-subjects: "
                f"the fold would train on its own test subject")
    if a.select_last and a.smoke and a.val_concepts > 0:
        print("[warn ] --select-last with a holdout: the holdout will be evaluated "
              "but not used for selection")
    if a.select_last and a.patience > 0:
        # Early stopping and last-epoch reporting contradict each other: stopping
        # early makes the last epoch the best one by construction, which silently
        # reads as "the protocol says last epoch" while a validation signal is
        # still choosing the stopping point.
        print(f"[warn ] --select-last ignores --patience {a.patience}: early stopping "
              f"chooses the epoch, which is the selection --select-last removes")
    if a.target_fusion == "routed_sr" and not want_loso and a.source_subjects is None:
        # A per-subject residual over a single subject is a free parameter that
        # absorbs the whole global bias; there is nothing for the shared part to
        # hold. Legal, but it is not the experiment the flag names.
        print("[warn ] --target-fusion routed_sr on an intra-subject run: the subject "
              "embedding has one row, so the global/residual split is unidentified")
    return a


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _binom_ci95(pct: float, n: int) -> tuple[float, float]:
    """Wilson score interval for a Top-1 percentage, in points.

    Present because this project has been making go/no-go decisions on 2-7 point
    differences that are inside the noise. The 200-way test set gives ONE trial per
    concept over 200 concepts, so a Top-1 of 47% carries a standard error of 3.5
    points: a 6.5-point "win" over another arm is z=1.31, i.e. not a win. Wilson
    rather than the normal approximation because Top-1 near 0 or 100 is common here
    (the collapsed structural head sat at 1.2%) and the normal interval leaves the
    unit range there.

    The threshold to remember, not the formula: on the 200-way test, differences
    below ~7 points are not evidence. `min_detectable_diff` below reports the
    smallest gap this run could have resolved.
    """
    if n <= 0:
        return (0.0, 100.0)
    p = pct / 100.0
    z = 1.959963985
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100.0 * max(0.0, c - h), 100.0 * min(1.0, c + h))


@torch.no_grad()
def evaluate(model: RetrievalModel, eeg: np.ndarray, feat: np.ndarray,
             device: torch.device, batch: int = 256, l2norm: bool = True,
             slot: int = 0, return_features: bool = False) -> dict:
    """Rep-averaged retrieval over a fixed item set; diagonal is the answer.

    Both sides must pass through their respective projectors before comparison:
    the EEG embedding and the (projected) image embedding have to share a width,
    and the image side is NOT the raw cached feature.

    `slot` picks which image of each concept to use; the resulting task is
    `n_items`-way where n_items is the number of concepts in `eeg`.

    `@torch.no_grad()` was missing here while `evaluate_selection` had it, so every
    direct call -- the final test score, and the `--test-every` trajectory -- built an
    autograd graph over the whole item set. It surfaced only when a configuration left
    the EEG embedding carrying `requires_grad` (the `.numpy()` below then raises), but
    in every configuration it was wasted memory and time in an evaluation path that
    cannot use a gradient.
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
    # The per-concept decomposition, over exactly the same matrix, so a paired
    # comparison against another arm can cancel the stimulus-difficulty term that
    # dominates the aggregate's standard error. Asserted here rather than trusted:
    # if these ever disagree the decomposition is worse than useless, because it is
    # the file every cross-arm claim is computed from.
    pc = retrieval_per_concept(z, f, ks=(1, 5))
    if abs(100.0 * float(np.mean(pc["top1"])) - rep["top1"]) > 1e-9:
        raise RuntimeError(
            f"per-concept Top-1 ({100.0 * float(np.mean(pc['top1'])):.6f}) does not "
            f"reproduce the aggregate ({rep['top1']:.6f}); the ranking conventions in "
            f"metrics.rank_vector and metrics.retrieve_all have drifted apart")
    rep["per_concept"] = pc
    if return_features:
        # Handed back so the deployment-side recovery ladder can be computed on
        # EXACTLY the matrix this Top-1 came from. Re-encoding for the recovery pass
        # would risk a second code path for the same tensors, and the gap between two
        # such paths is precisely the kind of difference that would be misread as
        # recovery's effect.
        rep["_z"] = z
        rep["_f"] = f
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
    vae_std: np.ndarray | None = None,
    batch: int = 256,
    l2norm: bool = True,
    sweep: bool = True,
) -> dict:
    """Validation metrics for both towers, from ONE forward pass per slot.

    The structure trunk is the most expensive module in the model (86M params
    against the semantic tower's 86M), so scoring the two towers in separate passes
    would roughly double validation time for exactly the same numbers.

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

    `vae_var_ratio` is the diagnostic that matters more than either of those:
    prediction std / target std, averaged over channels. A field can hold a fine
    `vae_top1` while being nearly constant, because the centred similarity is
    computed after re-normalisation. The shipped collapsed head sat at 0.0068; a
    closed-form ridge fit on the same inputs reaches 0.4411. Reported per epoch so
    collapse is visible while it is happening rather than inferred afterwards.
    """
    n_conc, n_img = eeg.shape[0], eeg.shape[1]
    slots = list(range(n_img)) if sweep else [0]
    sem, v1, vc, vr = [], [], [], []
    for s in slots:
        zs, fs, vps = [], [], []
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
            gt_raw = _slot(vae_rows, s, "vae_rows", p.shape[1:], n_conc)
            if vae_std is not None:
                # Compare in the NORMALISED space, because that is the space the
                # loss is defined in (train.py standardises both sides). Measuring
                # the ratio against raw latents would report a ratio scaled by the
                # per-channel standard deviations and could not be compared to the
                # ridge baseline's 0.4411, which is computed on normalised latents.
                sdv = torch.from_numpy(np.asarray(vae_std, dtype=np.float32)).view(1, -1, 1, 1)
                p = p / sdv.clamp_min(1e-8)
                gt_raw = gt_raw / sdv.clamp_min(1e-8)
            vr.append(float((p.std(dim=(0, 2, 3)) / gt_raw.std(dim=(0, 2, 3)).clamp_min(1e-8)).mean()))
            pf, gf = p.flatten(1), gt_raw.flatten(1)
            pf = pf - pf.mean(0, keepdim=True)
            gf = gf - gf.mean(0, keepdim=True)
            sim = F.normalize(pf, dim=-1) @ F.normalize(gf, dim=-1).t()
            v1.append(100.0 * float((sim.argmax(1) == torch.arange(len(sim))).float().mean()))
            vc.append(float(sim.diag().mean()))

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
        "vae_var_ratio": float(np.mean(vr)) if vr else None,
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
        elif name.startswith("struct.encoder.model.backbone.encoder.layer."):
            # `--struct-arch da2`: the depth trunk's transformer blocks. They are a
            # pretrained DINOv2 and want the same small step as any other pretrained
            # trunk, so they must land here rather than in the `s_heads` fallback --
            # which is where a rule keyed only on `struct.encoder.vit.` would have
            # put them, quietly training a pretrained backbone at the decoder's LR.
            g["s_blocks"].append((name, prm))
        elif name.startswith(("struct.encoder.model.backbone.embeddings.",
                              "struct.encoder.model.backbone.layernorm.")):
            # The depth trunk's input interface and final norm, the analogue of the
            # `patch_embed` / `pos_embed` / `norm` group above. The `neck` and `head`
            # are deliberately NOT here: they are the DPT decoder, they are what this
            # tower contributes over a generic trunk, and they take the head LR.
            g["s_interface"].append((name, prm))
        elif name.startswith("struct."):
            # Everything under `struct.` that is not the trunk's blocks or its input
            # interface is the decoder -- `struct.fusion`, `struct.up`, `struct.vae_head`
            # and, on the pooled branch, `struct.proj`. Matching by exclusion rather
            # than by listing the decoder's module names is deliberate: the list form
            # is what broke. It named `grid_mix`/`stem` (the CONVOLUTIONAL decoder's
            # modules, which is the branch every run had used) and so raised
            # "parameters matching no LR group: struct.proj.weight" the first time a
            # config selected `--struct-tokenizer eegit`, whose decoder is an MLP.
            # A rule keyed on the module names of one branch cannot survive the other
            # branch existing, and `struct` has exactly one non-decoder child
            # (`encoder`), which the two rules above already claim.
            g["s_heads"].append((name, prm))
        elif name.startswith("encoder.vit.blocks."):
            g["blocks"].append((name, prm))
        elif name.startswith(("encoder.vit.patch_embed.", "encoder.vit.pos_embed",
                              "encoder.vit.cls_token", "encoder.vit.norm",
                              "encoder.tokenizer.")):
            g["interface"].append((name, prm))
        elif name.startswith(("encoder.", "fusion.", "eeg_head.", "img_head.",
                              "target_router.")):
            # `target_router` is the `--target-fusion routed_sr` router: a
            # randomly-initialised projection over the TARGET layers, trained from
            # scratch exactly like the heads, so it takes the head LR. It is listed
            # here rather than covered by the `else: unknown` branch on purpose --
            # that branch exists to make a new module's LR an explicit decision.
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
        if args.patch_style == "time-region":
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
        if args.patch_style != "time-region":
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
        if args.struct_arch == "da2":
            # The depth trunk's own constraints replace the `BACKBONES`-registry
            # ones wholesale rather than being merged with them: `--struct-layers`,
            # `--struct-patch-size` and `--struct-backbone` name a trunk-plus-decoder
            # design that `da2` does not have, and accepting them here would let a
            # config claim a fusion of layers 8/10/12 that nothing reads.
            if args.struct_backbone != "da2":
                raise SystemExit(
                    f"--struct-arch da2 fixes the trunk, so --struct-backbone must be "
                    f"'da2' (got {args.struct_backbone!r}). It is still required, "
                    f"because it is the flag that builds a structure tower at all.")
            if args.struct_patch_size != 14:
                raise SystemExit(
                    f"--struct-arch da2 needs --struct-patch-size 14 (Depth Anything "
                    f"V2's conv is 14x14); got {args.struct_patch_size}. The EEG image "
                    f"is patchified by that conv, so this is not adjustable.")
            if args.struct_tokenizer != "eegit":
                raise SystemExit(
                    f"--struct-arch da2 needs --struct-tokenizer eegit (EEGiT's "
                    f"region-band geometry); got {args.struct_tokenizer!r}. The "
                    f"topography geometry feeds a scalp map to a depth model, whose "
                    f"conv and positional grid were trained on natural images and on "
                    f"the region-band layout respectively.")
            if args.channels == "all":
                # 5 regions for the 63-channel montage; the EEGiT region list is what
                # both towers group by, so the number is derived, not chosen.
                n_reg_d = 5
            else:
                n_reg_d = 2
            if args.patch_style == "time-region":
                # Time on the height axis, regions on the width -- EEGiT's released
                # layout, and the one the semantic tower is using in this config.
                h_d, w_d = args.struct_n_patches_w * 14, n_reg_d * 14
            else:
                h_d, w_d = n_reg_d * 14, args.struct_n_patches_w * 14
            print(f"[tok  ] structure tower: {args.da2_model or 'Depth-Anything-V2-Small'} "
                  f"patch 14, EEG image ({args.patch_style}) "
                  f"(3, H={h_d}, W={w_d}), {n_reg_d * args.struct_n_patches_w} tokens, "
                  f"pretrained backbone + DPT neck/head, scored at "
                  f"{args.struct_out_hw}x{args.struct_out_hw}")
        else:
            want_s = backbone_patch_size(args.struct_backbone)
            if want_s is None:
                raise SystemExit(f"cannot determine the patch size of "
                                 f"{args.struct_backbone!r}; register it in BACKBONES")
            if args.struct_patch_size != want_s:
                raise SystemExit(f"--struct-backbone {args.struct_backbone} has a "
                                 f"{want_s}x{want_s} patch_embed conv, so "
                                 f"--struct-patch-size must be {want_s} "
                                 f"(got {args.struct_patch_size})")
        if args.struct_arch == "vit" and not args.struct_layers:
            raise SystemExit("--struct-backbone given but --struct-layers is empty")
        if args.struct_arch == "vit" and not (args.vae_latents or args.depth_cache):
            raise SystemExit("--struct-backbone given but no structure target: pass "
                             "--vae-latents or --depth-cache. A structure tower with "
                             "no structural loss is an untrained decoder attached to "
                             "the run.")
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
        if args.struct_base_ch % 2:
            raise SystemExit(f"--struct-base-ch {args.struct_base_ch} must be even; the "
                             f"first GroupNorm in the decoder needs a group size that "
                             f"divides it")
        if args.struct_arch == "vit":
            if args.struct_tokenizer == "topography":
                if args.struct_patch_size != 16:
                    print(f"[tok  ] note: topography with patch {args.struct_patch_size} "
                          f"(DINOv3/MAE are 16)")
                if args.struct_scalp_res % args.struct_patch_size:
                    raise SystemExit(
                        f"--struct-scalp-res {args.struct_scalp_res} is not a multiple of "
                        f"--struct-patch-size {args.struct_patch_size}; the scalp map could "
                        f"not be tiled by whole patches")
                gh = (args.struct_n_time_bands * args.struct_scalp_res) // args.struct_patch_size
                gw = args.struct_scalp_res // args.struct_patch_size
                h_s = args.struct_n_time_bands * args.struct_scalp_res
                w_s = args.struct_scalp_res
                n_tok_s = gh * gw
                print(f"[tok  ] structure tower: {args.struct_backbone} patch "
                      f"{args.struct_patch_size}, SCALP TOPOGRAPHY image "
                      f"(3, H={h_s}, W={w_s}) = {args.struct_n_time_bands} bands x "
                      f"{args.struct_scalp_res}x{args.struct_scalp_res}, "
                      f"{n_tok_s} tokens ({gh}x{gw}), channels "
                      f"{args.struct_band_channels}, layers {args.struct_layers}, "
                      f"fusion {args.struct_fusion_mode}")
            else:
                n_reg_s = 5 if args.channels == "all" else 2
                if args.patch_style == "time-region":
                    h_s, w_s = args.struct_n_patches_w * args.struct_patch_size, \
                        n_reg_s * args.struct_patch_size
                else:
                    h_s, w_s = n_reg_s * args.struct_patch_size, \
                        args.struct_n_patches_w * args.struct_patch_size
                n_tok_s = n_reg_s * args.struct_n_patches_w
                print(f"[tok  ] structure tower: {args.struct_backbone} patch "
                      f"{args.struct_patch_size}, EEG image ({args.patch_style}) "
                      f"(3, H={h_s}, W={w_s}), {n_tok_s} tokens, "
                      f"layers {args.struct_layers}, fusion {args.struct_fusion_mode}")
            # Bounds were hardcoded to 1..24 when the only registered structural
            # backbone was DINOv2-L. DINOv3/MAE ViT-B has 12 blocks, so a
            # `--struct-layers 24` copied from an older config would have passed this
            # check and then silently dropped every index past 12 from the fusion
            # dict. `da2` skips it: it has no layer fusion, so there is no index to
            # bound.
            n_blocks_s = backbone_n_blocks(args.struct_backbone)
            if n_blocks_s is None:
                raise SystemExit(f"cannot determine the block count of "
                                 f"{args.struct_backbone!r}; register it in BACKBONES")
            bad_s = [l for l in args.struct_layers if l < 1 or l > n_blocks_s]
            if bad_s:
                raise SystemExit(f"--struct-layers {bad_s} outside 1..{n_blocks_s} "
                                 f"({args.struct_backbone} has {n_blocks_s} blocks); a "
                                 f"block index past the trunk's depth would silently be "
                                 f"absent from the fusion dict")
        print(f"[loss ] structure target: {args.struct_target} @ "
              f"{args.struct_scale}x{args.struct_scale}"
              f"{' (centred on the fit-set mean field)' if args.struct_center else ''} "
              f"at weight w_vae={args.w_vae:g} under {args.vae_loss.upper()}, "
              f"variance floor w_var={args.w_var:g} (margin {args.var_margin:g})")
    else:
        for flag in ("vae_latents", "depth_cache"):
            if getattr(args, flag):
                print(f"[cfg  ] --{flag.replace('_', '-')} ignored: no --struct-backbone")
    return keys


def load_struct_target(
    args, tr_shape: tuple[int, ...], n_test: int, split
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load the structural target as (N, C, H, W), optionally centred.

    Two independent choices are resolved here and both are recorded on `args`, so
    the export step in a separate process hours later can undo exactly what this
    did:

      * WHICH cache. `--struct-target {vae,depth}` x `--struct-scale {4..64}`. At
        scale 64 the original fine caches are used (VAE latents as float16
        memmaps, depth as float32); below that the area-averaged versions under
        `--coarse-root`, written by the pooling step, which are already float32
        and small enough to hold directly.
      * WHETHER the fit-set mean FIELD is removed. This is `--struct-center`, and
        it is not the same operation as the per-channel scalar normalisation
        `AuxTargetDataset` applies downstream: that one reshapes `vae_mean` to
        (C,1,1), which removes a global offset per channel and leaves every
        spatial structure of the mean intact. See the flag's help for why that
        distinction is the difference between the head learning the deviation and
        the head learning the mean.

    The centring is done ONCE over the whole array rather than per item in the
    dataset, because the mean field is a property of the split and not of a
    sample; doing it per item would also have to be re-derived identically in the
    export process, which is the class of silent disagreement `args._vae_mean`
    already exists to avoid. The field is stored on `args` for that reason.

    Depth is returned with an explicit channel axis -- (N,1,H,W) rather than
    (N,H,W) -- so that every downstream consumer (the per-channel statistics, the
    loss, the collapse gate) keeps working with `n_ch = 1` instead of needing a
    second code path for the single-channel case.
    """
    kind, scale = args.struct_target, int(args.struct_scale)
    if scale == 64:
        if kind == "vae":
            if not args.vae_latents:
                raise SystemExit(
                    "--struct-target vae --struct-scale 64 needs --vae-latents DIR")
            vdir = Path(args.vae_latents)
            f_tr = vdir / "train_vae_latents_f16.npy"
            f_te = vdir / "test_vae_latents_f16.npy"
        else:
            if not args.depth_cache:
                raise SystemExit(
                    "--struct-target depth --struct-scale 64 needs --depth-cache DIR")
            vdir = Path(args.depth_cache)
            f_tr = vdir / "train_depth_64.npy"
            f_te = vdir / "test_depth_64.npy"
        if not (f_tr.is_file() and f_te.is_file()):
            raise SystemExit(f"{vdir} needs {f_tr.name} and {f_te.name}")
        tr = np.load(f_tr, mmap_mode="r")
        te = np.load(f_te, mmap_mode="r")
    else:
        root = Path(args.coarse_root) if args.coarse_root else (
            config.OUTPUTS / "struct_targets" / "coarse")
        # Recorded so the export can find the coarse test target for the collapse
        # gate. The gate needs a ground truth at the SAME resolution as the
        # prediction, and at scale < 64 that is a different file from the one
        # `--depth-cache` names -- so the path cannot be re-derived from the flags
        # the export sees.
        args._coarse_root = str(root)
        f_tr = root / f"train_{kind}_{scale}.npy"
        f_te = root / f"test_{kind}_{scale}.npy"
        if not (f_tr.is_file() and f_te.is_file()):
            raise SystemExit(
                f"{root} needs {f_tr.name} and {f_te.name}. Build the coarse ladder "
                f"from the fine caches first (area-average pooling; the targets are "
                f"deterministic functionals of the fine ones, so no re-encoding and "
                f"no GPU is involved).")
        tr = np.load(f_tr)
        te = np.load(f_te)

    def _with_channel(a: np.ndarray) -> np.ndarray:
        # (N, H, W) -> (N, 1, H, W). Depth is the only single-channel target and it
        # is the only case where this fires.
        return a[:, None] if a.ndim == 3 else a

    tr, te = _with_channel(tr), _with_channel(te)
    src_dtype = tr.dtype

    fit_rows = np.sort(
        (split.fit_concepts[:, None] * tr_shape[1]
         + np.arange(tr_shape[1])[None, :]).ravel())

    # ---- the DISPLAY range, for turning the prediction into a condition -------
    # A fixed, fit-split range rather than a per-image one, and the difference is
    # not cosmetic. The cached target itself was written per-image normalised
    # (`build_gt_depth_cache.py` does `(d - d.min()) / (d.max() - d.min())` before
    # saving), which is right for a target whose dynamic range IS structure: it
    # makes every sample span [0,1] so the loss cannot be dominated by whichever
    # scene happened to have the largest depth extent.
    #
    # Applying the same operation to our PREDICTION would be wrong in a way that
    # matters. The prediction's dynamic range is small by construction -- the probe
    # puts the reachable per-concept signal at r = +0.22 -- so dividing by its own
    # min-max would divide by a quantity that is mostly noise, and the conditioning
    # image would become full-contrast speckle. That is the exact failure the
    # previous structural init had, arrived at from the other direction: a
    # condition that carries a little information, rescaled until it looks like it
    # carries a lot. A shared range keeps the true contrast, so what the ControlNet
    # sees is what the model actually predicted.
    #
    # Percentiles rather than min/max so a single outlier image cannot flatten the
    # range. Estimated on a stride subsample of the fit rows: the VAE cache is
    # 15040 x 4 x 64 x 64 and materialising it just to take two percentiles would
    # cost a gigabyte for a number that 2048 rows estimate to well within a percent.
    _sub = fit_rows[:: max(1, len(fit_rows) // 2048)]
    _samp = np.asarray(tr[_sub], dtype=np.float32).reshape(_sub.size, -1)
    _lo, _hi = np.percentile(_samp, [0.5, 99.5])
    args._struct_target_range = [float(_lo), float(_hi)]
    del _samp
    if _hi - _lo < 1e-6:
        raise SystemExit(f"{kind}@{scale}: the fit-split 0.5-99.5 percentile range is "
                         f"degenerate ({_lo:g}, {_hi:g}); the display map would divide "
                         f"by zero")
    print(f"[data ] struct target display range (fit split, 0.5-99.5 pct) "
          f"[{_lo:+.4f}, {_hi:+.4f}]")

    if args.struct_center:
        # float32 because the subtraction is over ~1500 rows per pixel and the
        # float16 fine cache cannot accumulate it; the copy is 271 MB at the VAE
        # cache's worst case and 1 MB at depth 8x8.
        tr = np.ascontiguousarray(tr, dtype=np.float32)
        te = np.asarray(te, dtype=np.float32)
        s = np.zeros(tr.shape[1:], dtype=np.float64)
        seen = 0
        for c0 in range(0, len(fit_rows), 4096):
            blk = np.asarray(tr[fit_rows[c0:c0 + 4096]], dtype=np.float64)
            s += blk.sum(axis=0)
            seen += blk.shape[0]
        field = (s / seen).astype(np.float32)
        tr = tr - field
        te = te - field
        args._struct_field_mean = field.reshape(-1).tolist()
        args._struct_field_mean_shape = list(field.shape)
        # A depth target is per-image normalised to [0,1], so its mean field is
        # itself a [0,1]-scaled map and `||mean||` is directly comparable to the
        # target's own norm. Reading it: the larger the share, the more of the
        # previous loss was being spent on a component no head needed to learn.
        print(f"[data ] centred {kind}@{scale} on the fit-set per-pixel mean "
              f"(||mean field||={float(np.linalg.norm(field)):.3f}, "
              f"per-pixel std of the mean field={float(field.std()):.4f})")
    else:
        args._struct_field_mean = None
        args._struct_field_mean_shape = None

    if te.shape[0] != n_test:
        raise SystemExit(f"{kind}@{scale} test cache has {te.shape[0]} rows, expected "
                         f"{n_test} (one image per test concept)")
    print(f"[data ] struct target {kind}@{scale} {tuple(tr.shape)} {tr.dtype} "
          f"(source {src_dtype}{', centred' if args.struct_center else ''})")
    return tr, te


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
    # Intra-subject: one subject, so both roles are the same subject and there is no
    # holdout to protect -- 'train' is the protocol here, and `--mvnn test` on a
    # single-subject run would be fitting the whitener on the test split's
    # residuals, which for this path is a genuine (if mild) calibration on test
    # data. The LOSO path below selects per role instead and is where the
    # distinction actually bites.
    if args.mvnn == "test" and not args.source_subjects:
        print("[warn ] --mvnn test on an intra-subject run fits the whitener on the "
              "test split's own residuals; the leakage is label-free but this is not "
              "the protocol the inter-subject papers describe. Use --mvnn train.")
    if args.mvnn == "off":
        tr_eeg, te_eeg = load_subject(args.subject, channels)
    else:
        # Absolute, not relative: train.py is invoked as a script, so __package__ is
        # empty and a relative import here dies with "attempted relative import with no
        # known parent package". The top of the file already imports epd.data this way.
        from epd.data import load_subject_std
        tr_eeg, te_eeg = load_subject_std(
            args.subject, channels, mvnn=args.mvnn,
            mvnn_shrinkage=args.mvnn_shrinkage, mvnn_max_cond=args.mvnn_max_cond,
            verbose=True)
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

    # ---- inter-subject (LOSO) ------------------------------------------------
    # Replaces the single-subject EEG with the concatenation of the source
    # subjects, and repeats the ALREADY-CHOSEN alignment target over them so that
    # a LOSO fold trains against exactly the same image representation as the
    # intra-subject arm it will be compared to. Done after the layout checks above
    # because those checks are about the single-subject file layout, which the
    # concatenation preserves behind the subject axis.
    subject_of_row = None
    if args.source_subjects:
        # Absolute for the same reason as the load_subject_std import above: this line
        # was never reached before the first real LOSO run, so the relative form had
        # gone unnoticed.
        from epd.data import expand_loso_images, load_loso

        loso = load_loso(args.source_subjects, args.target_subject,
                         channels, cache_dir=config.OUTPUTS / "cache",
                         mvnn=args.mvnn)
        subject_of_row = loso.tr_subject_of_row
        # The stimulus id per row, needed only by `--multipos`. `loso.n_concepts` is
        # the per-subject concept count, so the un-tiled concept index of row r is
        # `r % n_concepts` -- which is what `expand_loso_images`' tiling implies and
        # what the dataset turns into a global stimulus id using the ON-DISK slot
        # count. Built here rather than inside `load_loso` because it is a property of
        # the loss, not of the data, and a fold that does not use it should not carry
        # it into a checkpoint's provenance.
        stimulus_of_row = None
        if args.multipos:
            if len(args.source_subjects) < 2:
                raise SystemExit(
                    f"--multipos needs at least 2 source subjects; with "
                    f"{len(args.source_subjects)} every row is its own stimulus and the "
                    f"loss is the pairwise baseline wearing a different name")
            stimulus_of_row = (np.arange(loso.n_subjects)[:, None] * loso.n_concepts
                               + np.arange(loso.n_concepts)[None, :]).ravel() % loso.n_concepts
            if stimulus_of_row.shape[0] != loso.tr_eeg.shape[0]:
                raise SystemExit(
                    f"stimulus_of_row covers {stimulus_of_row.shape[0]} rows but the "
                    f"fold has {loso.tr_eeg.shape[0]}; the grouping would pair rows "
                    f"from different pictures")
        tr_eeg, te_eeg = loso.tr_eeg, loso.te_eeg
        img_tr = expand_loso_images(img_tr, loso.n_subjects)
        # `img_te` is NOT expanded: the test set is one held-out subject, so there
        # is exactly one row per test concept and no repetition to make.
        if img_tr.shape[:2] != tr_eeg.shape[:2]:
            raise SystemExit(
                f"LOSO image/EEG layout disagree after expansion: features "
                f"{img_tr.shape[:2]} vs EEG {tr_eeg.shape[:2]}. The image features "
                f"must be concept-major with {loso.n_concepts} concepts so that "
                f"tiling gives subject-major blocks.")
        if te_eeg.shape[0] != img_te.shape[0]:
            raise SystemExit(
                f"LOSO test layout disagree: features {img_te.shape[0]} vs EEG "
                f"{te_eeg.shape[0]}")
        # Set on `args` so it travels into the checkpoint through `vars(args)`: the
        # export and the metrics step rebuild the model hours later in another
        # process, and a `subject_residual` table built with the default width
        # would fail to load -- or, worse, load with the wrong rows assigned.
        args.n_subjects = loso.n_subjects
        print(f"[loso ] protocol: train {loso.n_subjects} subjects, hold out "
              f"sub-{args.target_subject:02d}; val_concepts={args.val_concepts}, "
              f"select={'last' if args.select_last else 'best-val'}")


    # The width of `img_head`'s input, parked on `args` so it travels into the
    # checkpoint. The export builds the model hours later, in a separate process,
    # and would otherwise have to guess it from whichever gallery file it happened
    # to load -- a guess that is wrong for any run whose alignment target is an
    # intermediate layer rather than the final projected one (block26 is 1280-d,
    # while the shipped `image_train.npy` is the 1024-d `visual.proj` output).
    # That mismatch surfaces as a `size mismatch for img_head.weight` RuntimeError
    # after training has already been paid for.
    args._image_dim = int(img_tr.shape[-1])
    # Explicit for the intra path so the value in the checkpoint is a decision
    # rather than an absence; `--source-subjects` overwrites it above.
    args.n_subjects = int(getattr(args, "n_subjects", 1))

    split = concept_split(args.val_concepts, args.split_seed)
    # Whether a selection signal exists at all. With --val-concepts 0 there is none,
    # and the only leak-free policy left is the last epoch -- so the two flags are
    # one decision, and `select_last` is derived rather than required to be
    # consistent by hand.
    has_val = len(split.val_concepts) > 0
    select_last = bool(args.select_last) or not has_val
    if not has_val:
        print("[sel  ] --val-concepts 0: no holdout, so the reported checkpoint is the "
              "LAST epoch. This is the SOTA inter-subject protocol (SCORE: 'report the "
              "final epoch'; Shallow Alignment: 'train on the full training set "
              "without a validation split'), and it is only leak-free because nothing "
              "is selected -- see PROTOCOL_INTER.md section 3.")
    elif select_last:
        print(f"[sel  ] --select-last: the {len(split.val_concepts)}-concept holdout is "
              f"still evaluated for diagnosis but does NOT select the checkpoint; the "
              f"last epoch is scored.")
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
    # Loaded as memmaps: the VAE cache is 542 MB, and
    # neither is read more than once per sample. Both are indexed by
    # concept_index*10+slot, which is the row order `load_subject` produces.
    vae_tr = vae_te = None
    vae_mean = vae_std = None
    if args.struct_backbone:
        vae_tr, vae_te = load_struct_target(args, tr_eeg.shape, te_eeg.shape[0], split)
        n_img = tr_eeg.shape[1]
        expected = tr_eeg.shape[0] * n_img
        for nm, a in (("vae", vae_tr),):
            if a is not None and a.shape[0] != expected:
                raise SystemExit(
                    f"{nm} cache has {a.shape[0]} rows but the EEG layout implies "
                    f"{expected} ({tr_eeg.shape[0]}x{n_img}). Everything downstream "
                    f"assumes row = concept*{n_img}+slot, so a mismatch means every "
                    f"image is paired with another image's target.")
        if vae_te is not None and vae_te.shape[0] != te_eeg.shape[0]:
            raise SystemExit(f"vae test cache has {vae_te.shape[0]} rows, expected "
                             f"{te_eeg.shape[0]} (one image per test concept)")

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
                blk = blk.reshape(blk.shape[0], n_ch, -1)     # (rows, ch, H*W)
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
            print(f"[data ] struct target {args.struct_target} @ "
                  f"{tuple(vae_tr.shape)} {vae_tr.dtype}, per-channel stats from the "
                  f"fit split: mean {np.round(vae_mean, 4).tolist()} "
                  f"std {np.round(vae_std, 4).tolist()}")

    # In a LOSO fold `tr_eeg`'s FIRST AXIS is not the 1654 concepts -- it is the
    # (subject, concept) rows that `load_loso` stacked subject-major, so it has
    # S*1654 of them and row r means subject r//C, concept r%C. `TrainDataset`
    # indexes `eeg[c, j]` with `c` taken straight out of this array, so passing the
    # 1654 concept-space `split.fit_concepts` here indexes subject 0's rows over and
    # over while `subject_of_row` still has S*C entries -- which trips the dataset's
    # own length guard rather than silently mislabelling, and is why every LOSO run
    # so far died before the first step. The expanded array is the fix, and it is the
    # same expression for both dataset classes.
    #
    # Note this is NOT `np.tile`: tiling repeats the concept-space indices, which
    # only happens to coincide with the row layout when the fit set is all 1654
    # concepts. With a val holdout it would pair a row with whichever subject
    # happened to sit at that flat offset. Offsetting per subject is correct either
    # way, and reduces to the identity when val_concepts is 0.
    ds_fit_concepts = split.fit_concepts
    if args.source_subjects:
        ds_fit_concepts = (np.arange(loso.n_subjects)[:, None] * loso.n_concepts
                           + split.fit_concepts[None, :]).ravel()
        if ds_fit_concepts.size != subject_of_row.shape[0]:
            raise SystemExit(
                f"LOSO row-concepts {ds_fit_concepts.size} do not cover the "
                f"{subject_of_row.shape[0]} subject rows; the expansion and "
                f"`load_loso` disagree about the layout")

    ds_fit = TrainDataset(tr_eeg, img_tr, ds_fit_concepts, l2norm=l2norm,
                          augment=augment, seed=args.seed, slots=train_slots,
                          subject_of_row=subject_of_row,
                          stimulus_of_row=stimulus_of_row)
    if args.struct_backbone:
        ds_fit = AuxTargetDataset(
            tr_eeg, img_tr, ds_fit_concepts,
            aux_vae=vae_tr, aux_depth=None,
            vae_mean=vae_mean, vae_std=vae_std,
            l2norm=l2norm, augment=augment, seed=args.seed, slots=train_slots,
            subject_of_row=subject_of_row, stimulus_of_row=stimulus_of_row)
    if train_slots is not None:
        # Say so loudly: this changes the contrastive TASK, not just its size. With
        # ten images per concept the model is additionally asked to separate the ten
        # images of one concept from each other, an instance-discrimination term the
        # one-image-per-concept retrieval protocol never asks for. It is not about
        # false negatives -- measured here, only 0.06% of negative pairs share a
        # concept, independent of batch size.
        print(f"[data ] TRAINING ON {len(train_slots)} IMAGE SLOT(S) {train_slots} PER "
              f"CONCEPT ({ds_fit_concepts.size * len(train_slots)} pairs, was "
              f"{ds_fit_concepts.size * tr_eeg.shape[1]}); this drops the "
              f"instance-discrimination term the official code never trains")
    # Validation keeps all 10 images per concept; evaluate_selection sweeps them.
    val_eeg = tr_eeg[split.val_concepts]
    val_feat = img_tr[split.val_concepts]
    val_vae = None
    if args.struct_backbone:
        rows = (split.val_concepts[:, None] * tr_eeg.shape[1]
                + np.arange(tr_eeg.shape[1])[None, :])
        val_vae = vae_tr[rows] if vae_tr is not None else None
    has_vae = bool(val_vae is not None)
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
    if args.tokenizer in ("eegit", "topography") and not args.no_zscore:
        tok = model.encoder.tokenizer
        tok.set_norm_stats(tr_eeg[split.fit_concepts])
        print(f"[zscor] semantic per-channel z-score from {len(split.fit_concepts)} fit "
              f"concepts (mean {float(tok.eeg_mean.mean()):+.3f}, "
              f"std {float(tok.eeg_std.mean()):.3f})")
    if model.struct is not None and not args.no_zscore:
        # Fit separately even though the statistics are numerically identical to the
        # semantic tower's: each tokenizer carries its own buffer, and a checkpoint
        # must be loadable on its own without assuming some other tower fitted it.
        stok = model.struct.encoder.tokenizer
        stok.set_norm_stats(tr_eeg[split.fit_concepts])
        print(f"[zscor] structural per-channel z-score (mean "
              f"{float(stok.eeg_mean.mean()):+.3f}, "
              f"std {float(stok.eeg_std.mean()):.3f})")

    # Per-channel target std, in the SAME normalised space the L1 loss is defined in.
    # Computed once from the fit split and used by `variance_floor` as the floor the
    # prediction must reach. This is the number whose absence let the shipped head
    # collapse to a variance ratio of 0.0068 while its L1 looked healthy.
    vae_target_std = None
    if has_vae:
        _blk = np.asarray(vae_tr, dtype=np.float32)
        _rows = np.sort((split.fit_concepts[:, None] * tr_eeg.shape[1]
                         + np.arange(tr_eeg.shape[1])[None, :]).ravel())
        _fit = _blk[_rows]
        if vae_std is not None:
            _fit = _fit / np.asarray(vae_std, dtype=np.float32).reshape(1, -1, 1, 1)
        vae_target_std = _fit.std(axis=(0, 2, 3)).astype(np.float32)
        del _blk, _fit
        print(f"[loss ] VAE target std (normalised, fit split) "
              f"{np.round(vae_target_std, 4).tolist()}; variance floor is "
              f"{args.var_margin:g}x this")

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
        if getattr(args, "struct_arch", "vit") == "da2":
            # `--struct-layers` and `--struct-fusion-mode` are NOT part of this
            # architecture, and printing them anyway was the misleading part: the
            # line named a layer fusion that does not exist on a DepthTower, and a
            # `field=(32, ...)` whose 32 is `--struct-field-ch`, also unused. The
            # grid and the scoring resolution are what this tower actually has.
            print(f"[model] structure tower da2 ({args.da2_model or 'DA2-Small'}) "
                  f"tokens={model.struct.encoder.tokenizer.n_tokens} "
                  f"grid={model.struct.encoder.dst_grid} "
                  f"eeg-image={model.struct.encoder.tokenizer.height}x"
                  f"{model.struct.encoder.tokenizer.width} "
                  f"map={tuple(model.struct.field_hw)} "
                  f"scored_at={model.struct.out_hw}x{model.struct.out_hw} "
                  f"channels={model.struct.vae_ch}")
        else:
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
    if args.multipos:
        # Stated once at startup rather than per step, because it is a property of the
        # RUN, not of a batch: with fewer than two sources there is nothing to make a
        # positive out of and the flag would silently do the pairwise baseline while
        # the log claimed otherwise.
        if args.source_subjects is None:
            print("[loss ] --multipos on an intra-subject run: each stimulus appears once "
                  "per batch, so the mask is the identity and this IS the pairwise loss")
        elif len(args.source_subjects) < 2:
            print(f"[warn ] --multipos with {len(args.source_subjects)} source subject: "
                  f"no two rows can share a stimulus, so the loss degenerates to the "
                  f"pairwise baseline")
        else:
            print(f"[loss ] multi-positive alignment ON (SCORE Eq. 1): rows sharing a "
                  f"stimulus are positives across the {len(args.source_subjects)} source "
                  f"subjects; degenerate batches still reduce to the pairwise loss")
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

    # ---------------- weight EMA (see --ema-decay)
    # A shadow state dict plus one reusable shadow model. The shadow model is built
    # once and reloaded each epoch rather than deep-copied per epoch: the state dict
    # is ~0.36 GB at 89M params and a per-epoch copy would allocate and free that 60
    # times for no benefit.
    ema_state: dict | None = None
    ema_model = None
    if args.ema_decay and args.ema_decay > 0:
        ema_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        import copy as _copy
        ema_model = _copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()
        print(f"[ema  ] weight EMA on: decay {args.ema_decay}, warmup "
              f"{args.ema_warmup_steps} steps; EMA weights are selected and saved",
              flush=True)
    t0 = time.time()
    t_train_s = 0.0
    t_val_s = 0.0
    n_steps = 0
    global_step = 0
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
        tot_sem = tot_vae = tot_var = 0.0
        n_vae = n_var = 0
        _mp_rows = _mp_paired = 0
        t_ep = time.time()
        for batch in dl_fit:
            global_step += 1
            x = batch[0].to(device, non_blocking=True)
            f = batch[1].to(device, non_blocking=True)
            subj = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            if subject_of_row is not None:
                # The per-row subject, from the dataset. Hardcoding zeros here is
                # what made LOSO look like it was training subject-aware: the
                # embedding existed, was scheduled, and was saved in the checkpoint,
                # while every row said "subject 0".
                subj = batch[3].to(device, non_blocking=True)
            # Stated once per run, because "the flag is on" and "the term is doing
            # anything" are different claims and only the log can tell them apart. A
            # batch draws 256 rows from ~15k, so the nine copies of one picture mostly
            # land in different batches and the positive set is usually empty: the
            # first-epoch fraction below is the honest measure of whether SCORE Eq. 1
            # got any traction, and if it is small the fix is the sampler, not the flag.
            n_meta = (0 if subject_of_row is None else 1) + (0 if stimulus_of_row is None else 1)
            mp_groups = None
            if stimulus_of_row is not None:
                mp_groups = stimulus_groups_from_ids(
                    batch[3 + (1 if subject_of_row is not None else 0)].to(device))
                if epoch == 1:
                    # Fraction of rows that have at least one OTHER row of the same
                    # picture in this batch. 0 means the mask is the identity and the
                    # multi-positive loss is exactly the pairwise baseline, whatever
                    # the flag says.
                    sizes = torch.bincount(mp_groups)
                    paired = int((sizes[mp_groups] > 1).sum())
                    _mp_rows += x.shape[0]
                    _mp_paired += paired
            aux = batch[3 + n_meta:]
            a_vae = None
            if has_vae:
                a_vae = aux[0].to(device, non_blocking=True)

            if model.struct is None:
                z_e, z_i, _w = model(x, f, subj, training=True)
                sd = None
            else:
                out = model.forward_all(x, subj, training=True)
                z_e = out["z"]
                z_i = model.encode_image(f, subj, training=True)
                sd = out["struct"]

            # Complementary weighting, as in SAMGA: mmd_w*MMD + (1-mmd_w)*contrastive.
            # A plain sum would scale the total gradient magnitude with mmd_w and
            # change the effective learning rate between epochs. The structural terms
            # are added OUTSIDE that weighting on purpose: the MMD schedule is a
            # statement about the shared semantic geometry, and letting it also scale
            # the latent regression would mean the structure tower's effective LR
            # depended on a hyper-parameter that has nothing to do with it.
            loss = contrast_w * criterion(z_e, z_i, mp_groups)
            if mmd_w > 0:
                loss = loss + mmd_w * mmd_rbf(z_e, z_i)
            tot_sem += float(loss.detach())
            if sd is not None:
                if a_vae is not None:
                    # The latent regression's loss is selectable because the choice is
                    # the difference between the branch working and not. See
                    # `--vae-loss`: L1's conditional median on a weakly predictable
                    # spatial target is near the global mean field, so L1 is nearly
                    # indifferent between a constant and a noisy instance estimate,
                    # while MSE is not.
                    l_vae = (latent_l1(sd["vae"], a_vae) if args.vae_loss == "l1"
                             else latent_mse(sd["vae"], a_vae))
                    loss = loss + args.w_vae * l_vae
                    tot_vae += float(l_vae.detach())
                    n_vae += 1
                    if args.w_var > 0 and vae_target_std is not None:
                        # One-sided hinge; see losses.variance_floor. Added rather
                        # than blended into the regression weight, because the two
                        # terms have different units and a single weight could not
                        # express "keep the regression objective, refuse to collapse".
                        l_var = variance_floor(sd["vae"], vae_target_std, args.var_margin)
                        loss = loss + args.w_var * l_var
                        tot_var += float(l_var.detach())
                        n_var += 1

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            sched.step()

            # EMA update, after the step so the average sees post-update weights.
            # Buffers (num_batches_tracked) are copied, not averaged -- averaging an
            # integer counter is meaningless. The `.float()` on the copy keeps the
            # average in fp32 even if the model runs in a reduced precision.
            if ema_state is not None and global_step > args.ema_warmup_steps:
                with torch.no_grad():
                    msd = model.state_dict()
                    for k, v in msd.items():
                        if v.dtype.is_floating_point:
                            ema_state[k].mul_(args.ema_decay).add_(
                                v.detach().float(), alpha=1.0 - args.ema_decay)
                        else:
                            ema_state[k].copy_(v)

            with torch.no_grad():
                sim = F.normalize(z_e, dim=-1) @ F.normalize(z_i, dim=-1).t()
                tr_top1 += int((sim.argmax(dim=1) == torch.arange(x.shape[0], device=device)).sum())
            tot += float(loss.detach())
            nb += 1
        n_steps += nb
        if epoch == 1 and stimulus_of_row is not None and _mp_rows:
            _frac = 100.0 * _mp_paired / _mp_rows
            print(f"[mp   ] epoch 1: {_mp_paired}/{_mp_rows} rows ({_frac:.1f}%) shared a "
                  f"stimulus with another row in their batch, so those rows had a "
                  f"non-trivial positive set. At 0% the multi-positive loss IS the "
                  f"pairwise baseline and the gap is the sampler, not the objective.")
            if _frac < 1.0:
                print(f"[warn ] only {_frac:.2f}% of rows had a co-stimulus partner; SCORE "
                       f"Eq. 1 is effectively off. Raise --batch-size (SAMGA/SAMGE use "
                       f"1024) or group co-stimulus rows into the same batch.")
        t_after_train = time.time()
        t_train_s += t_after_train - t_ep

        t_v = time.time()
        if not has_val:
            # No holdout: there is nothing to select ON. `val` stays None rather than
            # a plausible-looking zero, so that every downstream reader (the per-epoch
            # print, `rec`, the `best` dict, the timing line) has to decide explicitly
            # what "no validation" means rather than average a fake number into a
            # result json that no one can tell was never measured.
            val = None
            val_ema = None
        else:
            if model.struct is None:
                val = evaluate_selection(model, val_eeg, val_feat, device, l2norm=l2norm,
                                         sweep=not args.no_slot_sweep)
            else:
                val = evaluate_selection_dual(
                    model, val_eeg, val_feat, device,
                    vae_rows=val_vae, vae_std=vae_std,
                    l2norm=l2norm, sweep=not args.no_slot_sweep)

            # EMA validation. Cheap next to the raw pass (val is ~1 s/epoch here) and
            # the only way to see whether averaging is helping before the run ends, so
            # both curves are recorded even though only one of them selects.
            if ema_model is not None:
                ema_model.load_state_dict({k: v.to(device) for k, v in ema_state.items()},
                                          strict=True)
                if ema_model.struct is None:
                    val_ema = evaluate_selection(ema_model, val_eeg, val_feat, device,
                                                 l2norm=l2norm, sweep=not args.no_slot_sweep)
                else:
                    val_ema = evaluate_selection_dual(
                        ema_model, val_eeg, val_feat, device,
                        vae_rows=val_vae, vae_std=vae_std,
                        l2norm=l2norm, sweep=not args.no_slot_sweep)
                model.train()
            else:
                val_ema = None
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
            # Score the weights that would actually be SHIPPED. With EMA on that is the
            # averaged model; reading the trajectory off the raw weights would describe
            # a model nobody ships, which is the same mistake as selecting on one model
            # and saving another.
            test_diag = evaluate(ema_model if ema_model is not None else model,
                                 te_eeg, img_te, device, l2norm=l2norm)
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
        #
        # Which curve this is computed from is the instrument fix: with EMA on, the
        # score comes from the AVERAGED weights, because those are the weights that
        # would be shipped. Scoring the raw weights and then saving the averaged ones
        # would be selecting on one model and shipping another.
        if val is None:
            # No holdout at all. Nothing selects, so the score is a placeholder and
            # the checkpoint decision is driven entirely by `select_last` below --
            # which is the whole content of the SOTA protocol: report the last epoch.
            sel_src = None
            sel = float("nan")
        else:
            sel_src = val_ema if val_ema is not None else val
            # The structural terms drop out on their own when there is no structural
            # tower (`vae_top1` is None), which is exactly how the semantic-only arms
            # behave. They must NOT be left at a non-zero weight when a structural head
            # exists but has collapsed: a term sitting at chance (1.2% against a 0.67%
            # floor) injects a per-epoch random draw into the selection score, noise in
            # the one place it does the most damage. Callers running a collapsed head
            # pass --struct-sel-w 0.
            sel = sel_src["top1"]
            if sel_src.get("vae_top1") is not None:
                sel += args.struct_sel_w * sel_src["vae_top1"]

        rec = {"epoch": epoch, "loss": tot / max(1, nb), "sel": sel,
               "loss_sem": tot_sem / max(1, nb),
               "loss_vae": (tot_vae / n_vae) if n_vae else None,
               "loss_var": (tot_var / n_var) if n_var else None,
               # Diagnostic-only columns. They are absent (None) unless
               # `--test-every` was given, and `sel`/`best` never read them.
               "test_top1_diag": (test_diag["top1"] if test_diag else None),
               "test_top5_diag": (test_diag["top5"] if test_diag else None),
               "test_mean_rank_diag": (test_diag["mean_rank"] if test_diag else None),
               "train_top1_inbatch": 100.0 * tr_top1 / max(1, len(ds_fit) // args.batch_size * args.batch_size),
               "val_top1": (val["top1"] if val else None),
               "val_top5": (val["top5"] if val else None),
               "val_mean_rank": (val["mean_rank"] if val else None),
               "val_top1_std": (val["top1_std"] if val else None),
               "val_slots": (val["n_slots"] if val else None),
               # Same row, second curve. `val_*` is the raw weights, `val_*_ema` the
               # averaged ones; only one of them selects (see --ema-decay).
               "val_top1_ema": (val_ema["top1"] if val_ema else None),
               "val_top5_ema": (val_ema["top5"] if val_ema else None),
               "sel_is_ema": bool(val_ema is not None),
               "selected": select_last or (val is not None and sel > best["sel"]),
               "val_vae_top1": (val.get("vae_top1") if val else None),
               "val_vae_cos": (val.get("vae_cos") if val else None),
               "val_vae_var_ratio": (val.get("vae_var_ratio") if val else None),
               "mmd_w": mmd_w, "contrast_w": contrast_w,
               "train_s": round(t_after_train - t_ep, 2),
               "val_s": round(t_after_val - t_v, 2)}
        hist.append(rec)
        extra = ""
        # Printed from `sel_src`, which is `val_ema` whenever EMA is on. That is not a
        # cosmetic choice of curve: `best` below saves the EMA `state_dict`, and the
        # export then reads that file. Reading the structural diagnostics off `val`
        # (the raw weights) therefore reports a model that is thrown away -- and the
        # two can differ by an order of magnitude on a young or oscillating head. The
        # observed case: the raw column read `vae_var 0.708` at epoch 20 while the
        # EMA weights that were actually saved measured 0.060 at epoch 21, i.e. the
        # printed gate said "the structural head is alive" about a checkpoint in
        # which it was below the constant-predictor floor. Raw values are kept, but
        # labelled, so neither curve can hide the other.
        if sel_src is not None and sel_src.get("vae_top1") is not None:
            extra += f" | vae_top1 {sel_src['vae_top1']:.1f}+-{sel_src['vae_top1_std']:.1f}"
            if sel_src.get("vae_cos") is not None:
                extra += f" vae_cos {sel_src['vae_cos']:+.3f}"
        if sel_src is not None and sel_src.get("vae_var_ratio") is not None:
            extra += f" | vae_var {sel_src['vae_var_ratio']:.3f}"
        if val_ema is not None and val is not None and val.get("vae_var_ratio") is not None:
            extra += (f" [raw vae_var {val['vae_var_ratio']:.3f} "
                      f"vae_cos {val['vae_cos']:+.3f}]")
        # The val half of the line is printed only when a val set exists. Emitting
        # `nan` there would look like a measurement that went wrong rather than like
        # a run that deliberately has no holdout.
        val_txt = (
            f"val_top1 {val['top1']:.2f}+-{val['top1_std']:.2f} top5 {val['top5']:.2f} "
            f"rank {val['mean_rank']:.0f} "
            if val else "val NONE (--val-concepts 0) ")
        print(f"[ep {epoch:3d}/{args.epochs}] loss {rec['loss']:.4f} "
              f"train_top1(batch) {rec['train_top1_inbatch']:.2f} "
              f"{val_txt}mmd_w {mmd_w:.2f}{extra} "
              + (f"| ema {val_ema['top1']:.2f} " if val_ema else "")
              + (f"(sel {sel:.2f}) " if val else "(sel last) ")
              + f"({rec['train_s']:.0f}s train / {rec['val_s']:.0f}s val)", flush=True)

        # `--select-last` takes every epoch, so the file on disk is always the most
        # recent one and needs no epoch bookkeeping. `sel > best["sel"]` is the
        # val-selected path and is unreachable when there is no val set.
        take = select_last or (val is not None and sel > best["sel"])
        if take:
            # `top1` here is the selected curve's number, so the reported best_val is
            # the one the checkpoint actually achieves. With EMA on, the saved weights
            # must be the averaged ones -- saving the raw ones would ship a model that
            # was never the one scored.
            #
            # `src` is None only when there is no holdout, and then every metric field
            # is None too: the selection is "last epoch" and there is no validation
            # number to attach to it. Writing 0.0 here instead would put a fabricated
            # measurement into the result json, which is the one artifact a later
            # comparison cannot tell from a real one.
            src = sel_src
            best = {"sel": sel,
                    "top1": (src["top1"] if src else None),
                    "top5": (src["top5"] if src else None),
                    "mean_rank": (src["mean_rank"] if src else None),
                    "top1_std": (src["top1_std"] if src else None),
                    "vae_top1": (src.get("vae_top1") if src else None),
                    "vae_cos": (src.get("vae_cos") if src else None),
                    "vae_var_ratio": (src.get("vae_var_ratio") if src else None),
                    "epoch": epoch,
                    "is_ema": bool(val_ema is not None),
                    "selected_by": "last" if select_last else "val",
                    "raw_top1_at_this_epoch": (val["top1"] if val else None),
                    # The temperature AS IT WAS at the epoch this checkpoint came
                    # from. Captured here rather than read off the criterion at the
                    # end of training, because with a learnable temperature those are
                    # different numbers, and the one that describes the saved weights
                    # is this one.
                    "logit_scale": float(criterion.logit_scale.detach()),
                    "effective_scale": float(criterion.effective_scale().detach())}
            epochs_no_gain = 0
            weights = ({k: v.to(device) for k, v in ema_state.items()}
                       if ema_state is not None else model.state_dict())
            # The criterion's own state travels with the checkpoint. It did not, and
            # that gap cost a whole round of analysis: `InfoNCE.logit_scale` is a
            # separate `nn.Module`, so it was never in `model.state_dict()` and a
            # finished checkpoint could not say what effective logit scale its
            # embeddings were trained under. For a LEARNABLE temperature that is the
            # single number that explains the run -- reading it back required
            # inferring it from LayerNorm gammas, which is exactly the kind of
            # inference a checkpoint should not force. Recorded as both the raw
            # parameter and the effective scale, because they differ by `softplus`
            # and that difference is a factor of 5.2 at the same `--init-temp`.
            #
            # NOTE this is the criterion at the CURRENT epoch, which for a learnable
            # temperature continues to move after the last time `best` was written.
            # The selected epoch's value is in the result json's `criterion` block.
            torch.save({"model": weights, "args": vars(args), "epoch": epoch,
                        "criterion": {
                            "logit_scale": float(criterion.logit_scale.detach()),
                            "effective_scale": float(
                                criterion.effective_scale().detach()),
                            "learnable": not args.fixed_temp,
                            "softplus": bool(args.softplus),
                        }},
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
    test = evaluate(model, te_eeg, img_te, device, l2norm=l2norm,
                    return_features=bool(args.recovery))

    if args.recovery:
        # SCORE's Table 4, test side, computed on the frozen features of THIS run so
        # the four steps are directly comparable to the baseline above them. The
        # ordering matters and is their ablation: CSLS ranking, then moment matching,
        # then the orientation recovery, then the identity regularisation. Each row is
        # also a usable answer, so all of them are kept rather than only the last --
        # if the gate abstains, the honest number is the moment-matched one, and
        # reporting only the final row would hide that recovery declined to act.
        from epd.recover import csls_scores, recover as _recover

        z_raw = test.pop("_z")
        f_raw = test.pop("_f")
        zt = torch.from_numpy(np.asarray(z_raw, dtype=np.float32))
        ft = torch.from_numpy(np.asarray(f_raw, dtype=np.float32))
        if zt.shape[0] != ft.shape[0]:
            raise SystemExit(
                f"recovery needs a square query/gallery set for a 200-way read, but got "
                f"{zt.shape[0]} queries and {ft.shape[0]} gallery items")
        n_way = int(zt.shape[0])
        labs = torch.arange(n_way)

        def _acc(x: torch.Tensor, use_csls: bool) -> float:
            s = csls_scores(x, ft, k=args.rec_k) if use_csls else x @ ft.t()
            return float((s.argmax(dim=1) == labs).float().mean() * 100.0)

        ladder: dict = {"n_way": n_way, "rho": float(args.rec_rho),
                        "k_csls": int(args.rec_k),
                        "max_landmarks": args.rec_max_landmarks,
                        "min_landmark_rate": float(args.rec_min_landmark_rate)}
        ladder["cosine"] = _acc(zt, use_csls=False)
        ladder["csls"] = _acc(zt, use_csls=True)
        q_mm, _d_mm = _recover(zt, ft, k=args.rec_k, rho=args.rec_rho,
                               max_landmarks=args.rec_max_landmarks, moment=True,
                               orientation=False)
        ladder["moment_match_csls"] = _acc(q_mm, use_csls=True)
        # `row_key`, NOT `tag`. This loop used to bind `tag`, the OUTER variable holding
        # the run's tag: after the loop that name was left holding "recovery", so the
        # json write produced `recovery_result.json` instead of `{tag}_result.json` and
        # the json recorded `"tag": "recovery"`.
        #
        # It only bit when `--recovery` was on, and its worst form was not the wrong
        # filename: EVERY arm in the LOSO pipeline runs with `--recovery`, so every arm
        # wrote to the same path and a1 would have silently overwritten a0's result --
        # two hours of training replaced by one file, with no error raised anywhere. The
        # ladder keys must stay "recovery_rho0"/"recovery" because the pipeline and the
        # summary read those names, so only the variable is renamed.
        for row_key, rho in (("recovery_rho0", 0.0), ("recovery", float(args.rec_rho))):
            qh, diag = _recover(zt, ft, k=args.rec_k, rho=rho,
                                max_landmarks=args.rec_max_landmarks,
                                min_landmark_rate=args.rec_min_landmark_rate)
            ladder[row_key] = _acc(qh, use_csls=True)
            ladder[f"{row_key}_diag"] = diag
        test["recovery"] = ladder
        print(f"[rec  ] {n_way}-way CPU ladder (SCORE Eq. 3-9, frozen features):")
        print(f"[rec  ]   cosine {ladder['cosine']:.2f}  ->  CSLS {ladder['csls']:.2f}"
              f"  ->  +mean+scale {ladder['moment_match_csls']:.2f}"
              f"  ->  +recovery(rho=0) {ladder['recovery_rho0']:.2f}"
              f"  ->  +identity reg {ladder['recovery']:.2f}")
        _d = ladder["recovery_diag"]
        _abstain = (" (ABSTAINED: " + str(_d["abstain_reason"]) + ")") if _d.get("abstained") else ""
        _rif = _d.get("r_minus_i_frobenius")
        _rif_s = "n/a (abstained)" if _rif is None else f"{_rif:.3f}"
        print(f"[rec  ]   {_d['n_mutual_pairs']} mutual pairs "
              f"(rate {_d['landmark_rate']:.2f}), ||R*-I||_F {_rif_s}{_abstain}")
        if ladder["recovery"] < ladder["moment_match_csls"] - 1.0 and not _d.get("abstained"):
            # Not fatal -- it is a result -- but it is the number that gets quoted, and
            # a recovery BELOW its own input is the signature of the wrong-correspondence
            # regime rather than of a weak map. Say so where it will be read.
            print(f"[warn ] recovery scored {ladder['recovery']:.2f} BELOW the "
                  f"{ladder['moment_match_csls']:.2f} it started from. Past the landmark "
                  f"threshold a map fitted from wrong pseudo-matches is an active loss; "
                  f"set --rec-min-landmark-rate above {_d['landmark_rate']:.2f} to "
                  f"abstain instead, or report the moment-matched row.")

    # Error bars on the headline, and the smallest difference this run could resolve.
    # Both are stored so that no later comparison has to re-derive them and, more to
    # the point, so that a 3-point gap cannot be read as a result. `min_detectable_diff`
    # is the one-sided 95% threshold on the DIFFERENCE against another run of the same
    # size, which is what an A/B comparison actually needs -- it is larger than twice
    # the single-run half-width because the two runs' errors add.
    test_ci = _binom_ci95(test["top1"], int(test["n"]))
    p = test["top1"] / 100.0
    test["ci95"] = [round(test_ci[0], 2), round(test_ci[1], 2)]
    test["min_detectable_diff"] = round(
        1.959963985 * np.sqrt(2 * p * (1 - p) / max(1, int(test["n"]))) * 100.0, 2)

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
        # With --val-concepts 0 there is no val curve to locate a peak on, so the
        # comparison this block exists for ("is the holdout a faithful proxy?") has
        # no left-hand side. `None` rather than a placeholder epoch: the whole point
        # of the block is the LAG, and a fabricated val peak would invent one.
        v_peak_ep = best["epoch"] if best.get("top1") is not None else None
        v_peak_txt = (f"{best['top1']:.2f}%" if best.get("top1") is not None else "n/a")
        print(f"\n[diag] test trajectory ({len(diag)} points, every "
              f"{args.test_every} epochs) -- diagnosis only, not selection:")
        for e, v in diag:
            print(f"[diag]   ep {e:>3}  test {v:>6.2f}%"
                  + ("   <- test peak" if e == t_peak_ep else "")
                  + ("   <- val peak" if e == v_peak_ep else ""))
        if v_peak_ep is None:
            verdict = ("no validation split (--val-concepts 0), so there is no val "
                       "curve to compare against; the reported checkpoint is the LAST "
                       "epoch by construction")
            lag = None
        else:
            lag = t_peak_ep - v_peak_ep
            verdict = ("SAME peak location: the holdout is a faithful proxy and the "
                       "post-peak epochs are wasted time, not lost accuracy"
                       if abs(lag) <= max(1, args.test_every // 2) else
                       f"peaks are {lag:+d} epochs apart: the holdout is NOT tracking "
                       f"the test set, so the selection rule needs fixing before the "
                       f"recipe does")
        print(f"[diag] test peaks ep {t_peak_ep} ({t_peak_val:.2f}%) vs val peaks "
              f"ep {v_peak_ep if v_peak_ep is not None else 'n/a'} ({v_peak_txt}); "
              f"shipped ckpt scored {test['top1']:.2f}% on test")
        print(f"[diag] {verdict}")
        test_diag = {
            "curve": [{"epoch": e, "test_top1": v} for e, v in diag],
            "test_peak_epoch": int(t_peak_ep), "test_peak_top1": float(t_peak_val),
            "val_peak_epoch": (int(v_peak_ep) if v_peak_ep is not None else None),
            "val_peak_top1": (float(best["top1"]) if best.get("top1") is not None else None),
            "peak_lag_epochs": (int(lag) if lag is not None else None),
            "verdict": verdict,
            "note": f"diagnostic only; this run's reported number is "
                    f"{best.get('selected_by', 'val')}-selected",
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
                    "val_top1": best.get("top1"),
                    "gap_vs_val": (fd["top1"] - best["top1"]
                                   if best.get("top1") is not None else None),
                    "protocol": "identical to validation, but on FIT concepts"}
        # The verdict needs a val number to compare against. With --val-concepts 0
        # there is none, and inventing one would turn "fit >> val" into a claim about
        # a quantity this run never measured.
        fit_diag["verdict"] = (
            "no validation split (--val-concepts 0), so the fit/val gap is undefined; "
            "compare `top1` against the test number instead"
            if fit_diag["gap_vs_val"] is None else
            "the model FITS the training set (fit is far above val), so the binding "
            "constraint is GENERALISATION, not capacity or optimisation -- regularise "
            "or reduce capacity; more steps will not help"
            if fit_diag["gap_vs_val"] > 20.0 else
            "the model does NOT fit its own training set (fit is close to val), so the "
            "binding constraint is OPTIMISATION or capacity, not generalisation -- more "
            "steps, a higher LR, or fewer frozen blocks"
        )
        _v = best.get("top1")
        _v_txt = f"{_v:.2f}" if _v is not None else "n/a"
        print(f"[fit  ] fit-concept ceiling (same protocol as val, n={n_take}): "
              f"top1 {fd['top1']:.2f} top5 {fd['top5']:.2f} "
              f"| val {_v_txt} | test {test['top1']:.2f}")
        print(f"[fit  ] gap fit-val {fit_diag['gap_vs_val']:+.1f} points -> {fit_diag['verdict']}")

    def _sel_text() -> str:
        """One sentence naming exactly what 'best checkpoint' meant in this run.

        Recorded rather than assumed: the single-tower runs selected on val Top-1
        alone, and a result file that does not say which criterion produced the
        checkpoint cannot be compared against one that used a composite.
        """
        if best.get("selected_by") == "last":
            # Name the protocol, not just the epoch: "last epoch" IS the finding,
            # because it is what makes the number leak-free and comparable.
            return ("LAST epoch (no selection; --val-concepts "
                    f"{args.val_concepts} left no holdout to select on) -- the "
                    "SOTA inter-subject policy; test scored once")
        std = best.get("top1_std")
        std_txt = f"{std:.2f} std across image slots" if std is not None else "no spread recorded"
        if model.struct is None:
            return (f"best val top1 on concept-level holdout ({std_txt}); "
                    f"test scored once")
        return (f"best selection score on concept-level holdout: "
                f"val_top1 + {args.struct_sel_w:g}*val_vae_top1 ({std_txt}); "
                f"test scored once")

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
        # What the temperature actually DID, as opposed to how it was configured.
        # `temp_learnable` alone said `True` and left the reader unable to tell a run
        # whose scale converged near its initial value from one that moved by 10x.
        "criterion": {
            "softplus": bool(args.softplus),
            "init_effective_scale": float(_scale0),
            "selected_effective_scale": best.get("effective_scale"),
            "selected_logit_scale": best.get("logit_scale"),
            "final_effective_scale": float(criterion.effective_scale().detach()),
            "selected_epoch": best.get("epoch"),
        },
        "lr_groups": lr_groups_cfg,
        "lr_groups_final": {g.get("group"): g["lr"] for g in optim.param_groups},
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "fit_diagnostic": fit_diag,
        "image_dim": int(img_tr.shape[-1]),
        "structure_tower": (
            None if model.struct is None else {
                "backbone": args.struct_backbone,
                # `arch` decides which of the two structures the fields below
                # describe. They are not variants of one thing: `da2` has no layer
                # fusion and no `field_ch`, so recording those keys regardless --
                # and reading `struct.fusion` to do it -- was an AttributeError on
                # the first `da2` run, at the very end, after the training had
                # already been paid for.
                "arch": getattr(args, "struct_arch", "vit"),
                "layers": (None if getattr(args, "struct_arch", "vit") == "da2"
                           else args.struct_layers),
                "fusion_mode": (None if getattr(args, "struct_arch", "vit") == "da2"
                                else args.struct_fusion_mode),
                "eeg_layer_weights": (
                    None if getattr(args, "struct_arch", "vit") == "da2"
                    else [round(float(v), 5)
                          for v in model.struct.fusion.layer_weights()]),
                "patch_size": args.struct_patch_size,
                "n_patches_w": args.struct_n_patches_w,
                "n_tokens": int(model.struct.encoder.tokenizer.n_tokens),
                "patch_grid": list(model.struct.encoder.dst_grid),
                "eeg_image": [int(model.struct.encoder.tokenizer.height),
                              int(model.struct.encoder.tokenizer.width)],
                # The map the tower emits is the EEG image's own geometry for `da2`
                # (the DPT head runs at input resolution) and the decoded field for
                # `vit`. Both are recorded from the module rather than reconstructed
                # from flags, because the export path writes one of them.
                "field_hw": list(getattr(model.struct, "field_hw",
                                         (model.struct.out_hw, model.struct.out_hw))),
                "pretrained_depth_model": getattr(args, "da2_model", "") or None,
                "field_shape": ([model.struct.vae_ch, model.struct.out_hw,
                                 model.struct.out_hw]
                                if getattr(args, "struct_arch", "vit") == "da2"
                                else [args.struct_field_ch, model.struct.out_hw,
                                      model.struct.out_hw]),
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
                    "depth_cache": (str(args.depth_cache) if args.depth_cache else None),
                    "kind": args.struct_target,
                    "scale": int(args.struct_scale),
                    # The mean field is the whole point of `--struct-center`: it is
                    # what the loss was previously dominated by and what the export
                    # has to add back. Recorded in full (flattened) rather than as a
                    # summary because the export process cannot recompute it -- the
                    # fit split is a training artefact.
                    "centred": bool(args.struct_center),
                    "field_mean_shape": args._struct_field_mean_shape,
                    "field_mean": args._struct_field_mean,
                    "display_range": getattr(args, "_struct_target_range", None),
                    "coarse_root": getattr(args, "_coarse_root", None),
                    "vae_shape": (list(int(s) for s in vae_tr.shape[1:])
                                  if vae_tr is not None else None),
                    "vae_normalisation": (
                        None if vae_mean is None else {
                            "source": "fit concepts only",
                            "mean": [round(float(v), 5) for v in vae_mean],
                            "std": [round(float(v), 5) for v in vae_std]}),
                    "w_vae": args.w_vae,
                    "w_var": args.w_var, "var_margin": args.var_margin,
                    "vae_target_std": (None if vae_target_std is None else
                                       [round(float(v), 5) for v in vae_target_std]),
                },
                "selection": {
                    "score": ("val_top1 + %.2f*val_vae_top1"
                              % args.struct_sel_w),
                    "struct_sel_w": args.struct_sel_w,
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
            "val_slots_swept": (int(val["n_slots"]) if val else None),
            "val_way": (int(val["n"]) if val else None),
            "source_subjects": args.source_subjects,
            "target_subject": args.target_subject,
            "n_subjects": int(getattr(args, "n_subjects", 1)),
            "per_subject_zscore": bool(args.source_subjects),
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
    # Which table to quote depends on the protocol, and quoting the wrong one is not
    # a rounding error -- the two differ by a factor of four. An intra-subject run is
    # read against SAMGA's per-subject column; a LOSO run is read against the
    # inter-subject table, where SAMGA's own inter-subject number is 26.22 and the
    # best published result is SCORE's 53.23.
    if args.source_subjects:
        result["reference_sota"] = {
            "protocol": "inter-subject (LOSO), 9 subjects -> 1 held out, 63 ch, 200-way",
            "ATM_inter": {"top1": 5.5, "top5": 20.0,
                          "note": "SATTC's Table 1; the original cross-subject baseline"},
            "SATTC_inter": {"top1": 14.8, "top5": 38.4,
                            "note": "frozen encoders + label-free test-time calibration"},
            # Two SAMGA rows, because there are two different numbers in circulation and
            # conflating them is what made "are we reproducing it" unanswerable. `26.22` is
            # SCORE's Table 2 "Original | None" -- the SAMGA encoder re-measured under
            # SCORE's protocol (50 epochs, FINAL epoch reported). SAMGA's own Table 2
            # reports 34.4/64.8, averaged over five seeds, and its released launcher keeps
            # the best TEST-set epoch (`--early_stop_patience 10`), so its published figure
            # is test-selected. Our arms use `--select-last`: 26.22 is the row to compare
            # against, and 34.4 is a bound we did not earn.
            "SAMGA_inter": {"top1": 26.22, "top5": 57.98,
                            "note": "SAMGA encoder under SCORE's protocol (final epoch); "
                                    "this is the one our --select-last runs answer to"},
            "SAMGA_published_inter": {"top1": 34.4, "top5": 64.8,
                                      "note": "SAMGA's OWN Table 2 (5 seeds, best TEST-set "
                                              "epoch); sub-08 cell is 28.7/59.5"},
            "SCORE_inter": {"top1": 53.23, "top5": 83.55,
                            "note": "= SAMGA + label-free coordinate recovery; the bar"},
            "sources": "PROTOCOL_INTER.md section 7",
        }
    else:
        result["reference_sota"] = {
            "protocol": "intra-subject, sub-08",
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
    print(f"[test] 95% CI [{test['ci95'][0]:.1f}, {test['ci95'][1]:.1f}]  |  this run "
          f"resolves differences of {test['min_detectable_diff']:.1f} points or more; "
          f"anything smaller is not evidence")
    ridge_msg = (f"  ridge floor {RIDGE_REF['test_top1']:.2f} -> "
                 f"{'PASS' if test['top1'] > RIDGE_REF['test_top1'] else 'FAIL'}") if RIDGE_REF else ""
    if args.source_subjects:
        print(f"[ref ] INTER-SUBJECT bar: SAMGA 26.22/57.98, SCORE 53.23/83.55, "
              f"ATM 5.5/20.0 (PROTOCOL_INTER.md s7). The ridge floor above is "
              f"INTRA-subject and is not the floor for this run.")
        print(f"[ref ] held out sub-{args.target_subject:02d}; trained on "
              f"{args.source_subjects}. Any comparison to an intra-subject number, "
              f"including this project's own earlier arms, is a cross-protocol "
              f"subtraction and must not be made.")
    else:
        print(f"[ref ] SAMGA intra SUB-08 94.8 (10-subject avg 91.3) | "
              f"EEGiT intra avg 70.4 (same backbone){ridge_msg}")
        print(f"[ref ] NOTE: SAMGA has NO validation split and selects "
              f"checkpoint_test_best.pth on test Top-1, so its published number is a "
              f"maximum over ~60 test evaluations, not a held-out score. See "
              f"scripts/run_epd_anchor.sh for the measurement of that gap.")
    print(f"[done] {out_dir / f'{tag}_result.json'}")


if __name__ == "__main__":
    main()
