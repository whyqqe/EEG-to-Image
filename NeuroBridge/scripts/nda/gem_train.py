#!/usr/bin/env python3
"""GEM -- Grounded Embedding Model.  sub-08 intra-subject.

THE THREE TOWERS, AND THE ONE THING THAT MAKES THEM DIFFERENT FROM EACH OTHER
----------------------------------------------------------------------------
A multi-tower design is only as good as the disagreement between its towers.  So
each tower here is defined by a target that the OTHER towers cannot express, and
every target is checked for that property in the report:

  T1 SEMANTIC / TEXT DECODER.  The target is T5-base's own encoder output for the
     composed description (concept + overall + subject + background + detail), and
     the tower is trained through a FROZEN T5 DECODER with teacher forcing.  What
     the tower outputs is therefore not an embedding that has to be interpreted:
     it is a string, produced by the same decoder that would have produced the
     caption from its own encoder.  The bottleneck is the text, and the loss is
     the log-likelihood of the actual words.
     `--w-text-decode` is that term's weight; it is the tower's primary signal and
     the embedding regression is secondary support, not the objective.

  T2 STRUCTURAL / VAE.  The target is the 4x16x16 low-frequency SDXL VAE latent
     (the 64x64 latent is low-pass filtered and downsampled; the high-frequency
     half is unmeasurable from a 28% reproducible signal).  It is a DIFFERENT
     OBJECT from any CLIP embedding: it is a spatial grid of a specific
     autoencoder's activations, and no amount of contrastive alignment produces it.
     Its job in generation is the img2img INIT, not an IP vector.

  T3 CLIP IMAGE PROJECTION.  The target is `encode_image()` of the real image --
     the 1024-d PROJECTED ViT-H-14 feature, which is exactly what
     `ip-adapter_sdxl_vit-h.bin` consumes.  It is a different object from T1's
     target twice over: a different encoder, and the image rather than the text.

THE SPACE FIX
-------------
The previous design regressed EEG onto `sem_image_*.npy`, a 1280-d penultimate
activation, and then fused that into a 1024-d IP condition.  1280 is the width of
the transformer block, not of the shared text/image space, so the "image"
condition was being fitted to a representation the image adapter never reads.
T3 is fitted to `encode_image()` (1024, verified identical to
`ln_post[:,0] @ proj` in `gem_clip_img.py`).

ATTRIBUTION, WITHOUT AN ORACLE
-----------------------------
`--attr-id-oracle 0` (the default) and it must stay 0.

Generation is anchored by an IP-ADAPTER EMBEDDING, so for EEG to genuinely drive
the image the embedding must be a function of the EEG rather than a constant of
the battery.  This is measured behaviourally, by generating from counterfactual
conditions and scoring them under the OFFICIAL same-concept protocol:

    self     the condition predicted for this row
    concept  the SAME EEG, but the condition decoded FOR THIS ROW'S CONCEPT
             (i.e. what this row's own image can tell the model) -- the ORACLE row,
             which must WIN, and by a margin that says how much headroom is left
    swap     the same EEG asked for another concept's description
    noise    the condition decoded for a row of unrelated EEG

`--attr-draws` conditions per row are generated from descriptions DRAWN FROM THE
ROW'S CONCEPT in the TRAIN description pool.  Nothing about the row's own test
image enters a draw: the draw is labelled by concept, and the concept label is the
standard retrieval supervision this project's rows already use.  The alternatives
(averaging over the whole pool, or over the test set) were measured earlier:
pooling over test concepts gives a target whose best solution is a constant, and
`tower_redundancy_probe.py` measured exactly that (`reps_used 1`, `top1_lift 0.000`).

THE COMPOSED-DESCRIPTION TRICK, WHICH IS WHAT MAKES ATTRIBUTION EXACT
---------------------------------------------------------------------
The semantic target is the CLIP-text embedding of the CONCATENATION
`"<concept>. <overall>. <subject>. <background>. <detail>"`, not the mean of the
four per-field embeddings.  That single choice buys three things at once:
  * the pool and the prompt are THE SAME STRING, so a condition is decodable by
    construction and the pool is not an average that no decoder inverts;
  * the full attribution pool is preserved -- every field and the concept word sit
    in the string, so a constant cannot satisfy it;
  * it is what a caption IS.  The per-field heads are kept for measurement (which
    field grounds) and the composed string is what generation consumes.

WHAT IS REPORTED SO THE CLAIMS CAN BE CHECKED, NOT BELIEVED
----------------------------------------------------------
  * `front.json` -- the subject fingerprint: aperiodic slope and its R^2, the
    participation ratio and rank per band, the equalisation gains.
  * the FROZEN-CONDITION check: the condition predicted from the NOISE arm, as a
    pairwise accuracy against the true CLIP bank.  0.5 is chance.  A model that
    hallucinates a plausible centroid scores 0.5 and says so.
  * the ROW-IDENTITY check: is the condition row-specific at all?  Measured as the
    accuracy with which the row's own condition retrieves its own row among all
    200 test rows, i.e. with the concept label removed.  1/200 is chance.
  * the CROSS-PREDICTION matrix: each tower's condition is also produced through
    the OTHER towers' heads (a ridge fitted on TRAIN activations only), which is
    the honest pipeline-level version of the drift test.
  * per-field grounding: which description fields the concept word actually moves.
  * `repoi`: the high-frequency half of the SDXL VAE latent, which `g2` measured at
    0.717 of the full latent energy and which this target leaves out on purpose.

LEAK-FREE
  * encoder, towers, fingerprint and VAE head see that subject's TRAIN rows only;
  * the concept gallery is the 1654 TRAIN concepts, asserted disjoint from the 200
    test concepts before a single image is generated;
  * `shared_r` is the intra-subject export (`ocf_export_intra_z.py`), so no
    multi-subject checkpoint can enter;
  * CLIP patch-token and image targets are TRAIN images only;
  * theta is solved on held-in TRAIN rows against a TRAIN statistic;
  * every generation hyper-parameter is fixed a priori on the command line.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocf_train import (                                        # noqa: E402
    GRANS, GENERIC_PROMPT, calibrate_quantile, clip_checked, l2n, l2t,
    multi_pos_nce, self_concentration,
)
from gem_front import (                                        # noqa: E402
    BAND_NAMES, NB, N_PATCH, PurifiedFront, fit_fingerprint, load_fingerprint,
    save_fingerprint,
)

T5_NAME = "google-t5/t5-base"
# `t5-base` d_model is 768.  512 is `d_kv * num_heads` (64 x 8), i.e. the WIDTH OF
# THE ATTENTION INTERNALS, not the width of `last_hidden_state`.  This module
# reads/writes `last_hidden_state` only, so the tower has to be built at 768; the
# run asserts it against `t5.config.d_model` rather than trusting this constant,
# because getting it wrong does not degrade the run, it crashes on the first step.
T5_D_MODEL = 768
S_FIELD_MAXLEN = 24          # tokens per description field


# --------------------------------------------------------------------- helpers
def downsample_latent(v: torch.Tensor, g: int = 4) -> torch.Tensor:
    """(B,4,64,64) -> (B,4,16,16) by average pooling.

    The low-frequency half is what is left: `g2` measured that the low-pass half
    carries 0.717 of the latent energy while the high-frequency half carries the
    rest, and the reproducible share of single-trial EEG variance is 0.28.  A
    target with more detail than signal teaches the read-out to fit noise, and the
    img2img initialisation does not need it -- SDEdit at strength 0.82 destroys
    most of the high-frequency content anyway.
    """
    return F.avg_pool2d(v.float(), g)


def _rowcos(a: np.ndarray) -> float:
    """Mean off-diagonal row-to-row cosine of the flattened array.

    1.0 means every row is the same row, i.e. the quantity carries NO per-row
    information.  This is the direct measurement of the failure that `condition`
    and the tower activations must never exhibit.
    """
    z = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)
    S = z @ z.T
    return float(S[~np.eye(len(z), dtype=bool)].mean())


def _cond_degeneracy(ote: dict) -> dict:
    """POST-TRAINING degeneracy measurement of everything the generator consumes.

    The init-time guard in `main` cannot catch a collapse that DEVELOPS during
    training, and that is exactly what happened on the first clean sub-08 run: the
    guard passed at initialisation (fused row-cos 0.7837) and then the trained model
    exported conditions with c_self 0.999995 and a row-to-row spread of 9.4e-5 --
    a constant.  The run still reported `trained=True`, a falling loss and zero
    gradient skips, and the semantic metrics came out at chance.  So the same
    measurement is repeated here, after training, on the arrays that were actually
    written to disk and handed to the generator.
    """
    out: dict = {}
    for key in ("fused", "clip", "s_code", "vae", "field"):
        if key not in ote:
            continue
        a = ote[key].detach().cpu().numpy().astype(np.float32)
        out[f"{key}_rowcos"] = _rowcos(a)
        out[f"{key}_std_rows"] = float(a.reshape(len(a), -1).std(0).mean())
    f = ote["fused"].detach().float()
    fn = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
    mu = f.mean(0, keepdim=True)
    out["c_self_predicted"] = float(
        (fn * (mu / mu.norm().clamp_min(1e-12))).sum(1).mean())
    # the SAME statistic computed on the concept the row actually depicts, which is
    # what row identity would need to beat
    out["degenerate"] = bool(out.get("clip_rowcos", out.get("fused_rowcos", 0.0)) > 0.90)
    out["note"] = (
        "MEASURED, not enforced.  `*_rowcos` is the mean row-to-row cosine: 1.0 means "
        "the array is a constant and carries no per-row information, so nothing "
        "downstream of it can be row-specific.  `degenerate` flags the HEADLINE "
        "condition (clip / V_img) row-cos > 0.90.  fused is retired and is not "
        "what generation consumes.")
    return out


def pair_nce(q: torch.Tensor, k: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric in-batch InfoNCE between two already-aligned embedding batches.

    Used for the vision head against `encode_image()`, so both sides live in
    CLIP-IMAGE space.  The retired `bank_nce(clip, text_concept_bank)` compared
    an image vector to a text centroid and is not this function.
    """
    q, k = l2t(q), l2t(k)
    logits = (q @ k.T) / max(float(tau), 1e-6)
    y = torch.arange(q.shape[0], device=q.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


def compose(concept: str, fields: dict) -> str:
    """The single string that is BOTH the attribution pool and the prompt."""
    bits = [concept.replace("_", " ").strip()]
    for g in GRANS:
        s = str(fields.get(g, "")).strip().rstrip(".")
        if s:
            bits.append(s)
    return ". ".join(b for b in bits if b) + "."


def ridge_fit(X: np.ndarray, Y: np.ndarray, lams=(1e-2, 1e-1, 1.0, 10.0, 100.0),
              split: float = 0.85, seed: int = 0):
    """Standardised ridge with lambda chosen on a HELD-IN split of X/Y.

    Held-in, not held-out: this is a linear map between two TRAIN arrays, and the
    caller only ever applies it to test activations of a model that never saw the
    test rows.  Using a held-out slice of TRAIN here (rather than the test split)
    is what keeps the cross-prediction rows leak-free.
    """
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    fit, val = idx[: int(split * n)], idx[int(split * n):]
    Xm, Xs = X[fit].mean(0), X[fit].std(0).clip(1e-6)
    Ym, Ys = Y[fit].mean(0), Y[fit].std(0).clip(1e-6)
    Xf = (X[fit] - Xm) / Xs
    Yf = (Y[fit] - Ym) / Ys
    G = Xf.T @ Xf
    eye = np.eye(Xf.shape[1])
    best_lam, best = lams[0], -1e18
    Xv, Yv = (X[val] - Xm) / Xs, Y[val]
    for lam in lams:
        W = np.linalg.solve(G + lam * eye, Xf.T @ Yf)
        P = Xv @ W * Ys + Ym
        r = 1.0 - ((P - Yv) ** 2).sum(-1) / ((Yv - Ym) ** 2).sum(-1)
        if float(np.mean(r)) > best:
            best, best_lam = float(np.mean(r)), lam
    W = np.linalg.solve(G + best_lam * eye, Xf.T @ Yf)
    return {"W": W, "Xm": Xm, "Xs": Xs, "Ym": Ym, "Ys": Ys,
            "lam": float(best_lam), "val_r2": float(best)}

    def _apply(Xq: np.ndarray) -> np.ndarray:                     # pragma: no cover
        return ((Xq - Xm) / Xs) @ W * Ys + Ym


def ridge_apply(fit: dict, Xq: np.ndarray) -> np.ndarray:
    return ((Xq - fit["Xm"]) / fit["Xs"]) @ fit["W"] * fit["Ys"] + fit["Ym"]


def row_identity_acc(pred: np.ndarray, targ: np.ndarray) -> float:
    """Is a condition row-specific at all?  Top-1 retrieval of a row's own target
    among all rows, diagonal masked.  Chance is 1/n."""
    P, T = l2n(pred.astype(np.float32)), l2n(targ.astype(np.float32))
    S = P @ T.T
    np.fill_diagonal(S, -np.inf)
    return float((S.argmax(1) == np.arange(len(P))).mean())


def bank_nce(pred: torch.Tensor, bank: torch.Tensor, tix: torch.Tensor,
             tau: float, k: int = 8) -> torch.Tensor:
    """InfoNCE of a batch against a whole CONCEPT BANK, with local scaling.

    NOT `ocf_train.multi_pos_nce`, which assumes the positive mask is
    `tix[:,None] == tix[None,:]`, i.e. that the target bank is the BATCH.  Here the
    bank is the 1654-concept gallery, so that expression builds an (B,B) mask and
    the `torch.where` then fails with
        "The size of tensor a (8) must match the size of tensor b (1654)".
    The positive of row i is the single bank entry whose concept IS `tix[i]`.

    Local scaling is kept for the same reason it was introduced: 1654 concepts
    with 10 near-duplicate trials each and many semantically close pairs make a
    few bank entries absorb the gradient (hubness), and subtracting the k-th
    neighbour similarity is the standard label-free correction.

    NORMALISED BY ln(n_bank), WHICH IS ITS CHANCE VALUE.  This is not cosmetic.
    A cross-entropy over 1654 classes starts at ln(1654) = 7.41, while every other
    term in this loss (cosines, a smooth-L1 on standardised latents, an MSE) is
    O(1) and bounded.  Left raw at weight 1.0 the two NCE terms contributed 14.89 +
    3.72 = 18.61 of a 29.19 total on sub-08 -- and `fcls` sat exactly AT chance
    (7.446 vs ln 1654 = 7.411), i.e. it was a constant that dominated the loss
    without carrying any usable signal.  Dividing by the chance value makes each
    term read as "1.0 = chance, 0 = perfect" and puts the weights back in a range
    where they mean something.  The RAW value is returned for reporting, because
    the ratio to chance is the interpretable quantity.
    """
    n_bank = bank.shape[0]
    S = l2t(pred) @ l2t(bank).T                       # (B, n_bank)
    if k > 0:
        with torch.no_grad():
            kk = min(k + 1, S.shape[1])
            r = S.topk(kk, dim=-1).values[:, -1:]      # (B,1) local scale
        S = (S - r) / tau
    else:
        S = S / tau
    pos = tix[:, None] == torch.arange(n_bank, device=pred.device)[None, :]
    lse_pos = torch.logsumexp(torch.where(pos, S, torch.full_like(S, -1e9)), dim=-1)
    raw = (torch.logsumexp(S, dim=-1) - lse_pos).mean()
    return raw / math.log(n_bank)


# ----------------------------------------------------------------------- model
class GEMNet(nn.Module):
    """Fingerprint front end -> one trunk -> three towers -> one fused condition.

    One trunk, three attention-pooled reads.  The towers share the EEG
    representation and differ in what they are asked to produce, which is the
    property that makes the cross-prediction matrix below meaningful: if the three
    reads differed only by a linear map, every cross-prediction would score the
    same and the matrix would be uninformative.
    """

    def __init__(self, fp: dict, d: int = 192, tf_layers: int = 3, tf_heads: int = 4,
                 drop: float = 0.15, n_ch: int = 17, n_time: int = 250,
                 sfreq: float = 250.0, use_front: bool = True,
                 vae_out: int = 16, clip_dim: int = 1024, t5_dim: int = T5_D_MODEL,
                 n_field: int = 4, field_len: int = S_FIELD_MAXLEN,
                 fuse_dim: int = 1024, n_anchor: int = 384,
                 nvol_dim: int = 1280):
        super().__init__()
        self.d = d
        self.front = PurifiedFront(fp, d, use_front=use_front, n_ch=n_ch,
                                   n_time=n_time, sfreq=sfreq, drop=drop)
        tok_probe, _ = self.front(torch.zeros(1, n_ch, n_time),
                                  torch.zeros(1, 1024))
        n_tok = tok_probe.shape[1]
        enc = nn.TransformerEncoderLayer(d, tf_heads, d * 4, dropout=drop,
                                         batch_first=True, norm_first=True,
                                         activation="gelu")
        self.trunk = nn.TransformerEncoder(enc, tf_layers)
        # SEVEN READS.  Four are the original towers/fusion; the last three are the
        # GVM additions and each exists because it has a target the others cannot
        # express:
        #   4 anchor  -- a DISCRETE per-field word set.  Not a CLIP vector, not a
        #                string produced by a decoder: a multi-label problem.
        #   5 nvol    -- the 1280-d penultimate ViT-H-14 activation, i.e. a
        #                DIFFERENT LAYER, not a reparameterisation of the 1024-d
        #                projected feature.
        #   6 arb     -- a two-scalar SELF-ASSESSMENT (semantic reliability,
        #                structural reliability).  Its target is the model's own
        #                per-level accuracy, so it is the only read whose target
        #                depends on the current weights.
        self.pool_q = nn.Parameter(torch.randn(7, d) * 0.02)   # txt/vae/clip/fuse/anc/nvol/arb
        self.pool_ln = nn.LayerNorm(d)

        # ---- T1 semantic: per-field embeddings + one composed-pool embedding
        self.n_field, self.field_len, self.t5_dim = n_field, field_len, t5_dim
        self.txt_proj = nn.Linear(d, n_field * field_len * t5_dim)
        self.txt_ln = nn.LayerNorm(t5_dim)
        self.pool_proj = nn.Linear(d, t5_dim)
        # maps the T5 pool into the CLIP-TEXT space, where the composed string's
        # target lives.  Init by the run from the ridge fitted on TRAIN activations
        # of this subject's train descriptions (see --t5-to-clip-init).
        self.t5_to_clip = nn.Linear(t5_dim, clip_dim, bias=False)

        # ---- T2 structural: low-frequency VAE latent
        self.vae_proj = nn.Linear(d, 4 * vae_out * vae_out)
        self.vae_out = vae_out

        # ---- T3 CLIP image projection (+ spatial tokens for iREPA)
        self.clip_proj = nn.Linear(d, clip_dim)
        self.n_q = 64
        self.spatial_q = nn.Parameter(torch.randn(64, d) * 0.02)
        self.spatial_proj = nn.Linear(d, 1280)

        # ---- M2 ANCHOR HEAD.  Per-field multi-label logits over the anchor
        # vocabulary.  `n_field` separate heads, not one shared head: the four
        # fields are different linguistic slots, and a single head would have to
        # resolve "orange" (an attribute) against "floor" (a background) in one
        # logit vector.  Per-field also gives the field-grounding diagnostic for
        # free, since each head's top-k is directly readable as that field's words.
        self.n_anchor = n_anchor
        self.anchor_proj = nn.Linear(d, n_field * n_anchor)

        # ---- M3 NEURAL-VISIBILITY HEAD.  Predicts the 1280-d penultimate
        # activation.  `ln_post[:,0] @ proj` maps it EXACTLY onto the 1024-d space
        # that IP-Adapter consumes (`gem_clip_img.py` verifies that identity
        # numerically), so this is a strictly richer target than the 1024-d one:
        # it is the same object plus the 256 directions the projection discards.
        # Which of the two actually serves as the T3 condition is decided by
        # MEASUREMENT on held-in rows at export time, not by assumption.
        self.nvol_proj = nn.Linear(d, nvol_dim)
        self.nvol_dim = nvol_dim

        # ---- M4 ARBITRATION HEAD.  Two scalars per row: how reliable THIS row's
        # semantic prediction is, and how reliable its structural prediction is.
        self.arb_proj = nn.Linear(d, 2)

        # ---- fusion: bottleneck over the three tower codes, then the condition
        #
        # The THREE CODES ARE NOT ALL t5_dim.  `fuse_in` has to be dimensioned for
        # what is actually concatenated: the semantic code lives in the CLIP-TEXT
        # space (clip_dim), the VAE code is a flattened 4 x vae_out x vae_out map,
        # and the CLIP code is clip_dim.  The first version used `t5_dim + ...`,
        # which is 768 and raised "mat1 and mat2 shapes cannot be multiplied
        # (8x3072 and 2816x1024)" -- 3072 is 1024 + 1024 + 1024, the true width.
        self.fuse_in = nn.Linear(clip_dim + 4 * vae_out * vae_out + clip_dim, fuse_dim)
        self.fuse = nn.Sequential(nn.LayerNorm(fuse_dim), nn.Linear(fuse_dim, fuse_dim),
                                  nn.SiLU(), nn.Linear(fuse_dim, fuse_dim))

    # -- pooling ---------------------------------------------------------------
    def _pool(self, tok: torch.Tensor, i: int) -> torch.Tensor:
        q = torch.tanh(self.pool_q[i])[None, None, :].expand(tok.shape[0], -1, -1)
        a = (q * tok).sum(-1) / math.sqrt(self.d)
        return self.pool_ln((a.softmax(-1)[:, :, None] * tok).sum(1))

    # -- forward ---------------------------------------------------------------
    def forward_all(self, x: torch.Tensor, csm: torch.Tensor) -> tuple[dict, dict]:
        tok, diag = self.front(x, csm)
        tok = self.trunk(tok)
        h_txt = self._pool(tok, 0)
        h_vae = self._pool(tok, 1)
        h_clip = self._pool(tok, 2)
        h_fuse = self._pool(tok, 3)
        h_anc = self._pool(tok, 4)
        h_nvol = self._pool(tok, 5)
        h_arb = self._pool(tok, 6)

        B = x.shape[0]
        # RESHAPE, THEN NORMALISE.  `txt_proj` emits n_field*field_len*t5_dim flat, so
        # `LayerNorm(t5_dim)` applied before the reshape sees [B, 4*24*768] and raises
        # "expected input with shape [*, 768], but got input of size [8, 73728]".
        emb = self.txt_ln(self.txt_proj(h_txt).reshape(
            B, self.n_field, self.field_len, self.t5_dim))
        pool = self.pool_proj(h_txt)                                  # (B,t5_dim)
        vae = self.vae_proj(h_vae).reshape(B, 4, self.vae_out, self.vae_out)
        clip = l2t(self.clip_proj(h_clip))
        sp = self.spatial_proj(
            self._cross(h_vae, tok))                                  # (B,64,1280)

        # M2: per-field anchor logits.  NOT reduced here -- the loss is
        # `binary_cross_entropy_with_logits`, and the prompt builder needs the full
        # per-field ranking, not a thresholded decision shared with the loss.
        anchor = self.anchor_proj(h_anc).reshape(B, self.n_field, self.n_anchor)
        # M3: the penultimate activation.  Left UN-normalised on purpose: its own
        # scale carries the residual norm, and the 1024-d projection below is what
        # gets normalised.
        nvol = self.nvol_proj(h_nvol)
        # M4: reliability in (0, 1) per level.  Index 0 = semantic, 1 = structural.
        r_est = torch.sigmoid(self.arb_proj(h_arb))

        s_code = l2t(self.t5_to_clip(pool))                           # CLIP-TEXT
        v_code = torch.tanh(vae.flatten(1))
        c_code = clip
        fused_in = self.fuse_in(torch.cat([s_code, v_code, c_code], 1))
        # NO spherical interpolation, and NO norm-balancing loss.
        #
        # The condition is the fusion output, unit-normalised.  Two mechanisms that
        # used to sit here were removed on purpose:
        #
        # 1. A solved theta rotating the condition from the tower mean toward the
        #    residual.  It duplicated `gem_calib.py`, which moves each row's
        #    `c_self = cos(x, mean)` onto the quantiles of the TRAIN concept bank's
        #    own `c_self`.  The calibration is per-row, exact and monotone; theta is
        #    one global scalar chosen on a grid that the documentation itself notes
        #    is NOT monotone in theta (hence 37 forward passes instead of a
        #    bisection).  Keeping the weaker of two mechanisms for the same quantity
        #    buys nothing and costs the grid search plus a class of indexing bug:
        #    the theta call took `va_sel` indices into a tensor holding only
        #    `tr_sel` rows and crashed with a device-side assert.
        #
        # 2. A `share` balance loss forcing the three codes' NORM FRACTIONS to each
        #    be 1/3.  Two of the three codes are unit-normalised and the third is
        #    `tanh` (norm < 1), so the target was unreachable except by saturating
        #    the tanh -- a constraint on an internal norm, not on anything measured.
        #    Whether a tower is redundant is answered by `tower_attribution` below,
        #    which measures cross-prediction on held-out rows; a hand-set regulariser
        #    would only have hidden the answer.
        fused = l2t(self.fuse(fused_in))
        out = {"field": emb, "pool": pool, "vae": vae, "clip": clip, "spatial": sp,
               "s_code": s_code, "fused": fused, "anchor": anchor, "nvol": nvol,
               "r_est": r_est}
        return out, diag

    def _cross(self, h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        q = torch.tanh(self.spatial_q)[None].expand(tok.shape[0], -1, -1)
        a = (q @ tok.transpose(1, 2)) / math.sqrt(self.d)
        return h.unsqueeze(1) + (a.softmax(-1) @ tok)


# `_solve_theta` and the spherical interpolation it calibrated have been REMOVED.
# See the comment at `fused = l2t(self.fuse(fused_in))` for why: the angle was a
# weaker duplicate of the per-row quantile calibration in `gem_calib.py`, its grid
# was non-monotone so it could not be bisected, and its call site mixed two index
# spaces (`va_sel` into a `tr_sel` tensor) and crashed on CUDA.


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subjects", type=int, nargs="+", required=True)
    ap.add_argument("--test-subject", type=int, required=True)
    ap.add_argument("--raw-cache", type=str, default=str(NB_ROOT / "outputs/tdm/cache"))
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--targets-dir", type=str, default=str(NB_ROOT / "outputs/g2/targets"))
    ap.add_argument("--clip-img-dir", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--captions-dir", type=str, default=str(NB_ROOT / "outputs/g2/captions"))
    ap.add_argument("--vae-cache", type=str,
                    default=str(NB_ROOT / "outputs/sdedit_ll_full10/shared/vae_cache"))
    ap.add_argument("--clip-patch-npy", type=str,
                    default=str(NB_ROOT / "outputs/tdm/clip_patch/train_patch_f16.npy"))
    ap.add_argument("--clip-text-dir", type=str,
                    default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=26)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--tf-layers", type=int, default=3)
    ap.add_argument("--tf-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--vae-out", type=int, default=16)
    # ---- loss weights.
    #
    # `w-text-decode` DEFAULTS TO 0.0 AND THAT IS THE GVM CHANGE, NOT A TUNING
    # KNOB.  It is the teacher-forced token-level cross-entropy through the frozen
    # T5 decoder, and the EEG-to-text literature is explicit about why it has to
    # go.  Two independent 2026 results:
    #
    #  * Brain-CLIPLM: non-invasive EEG canNOT support full lexical-syntactic
    #    reconstruction; what it preserves is ORDERED SEMANTIC ANCHORS, and quality
    #    peaks at about five keywords, WORSE at full-sentence granularity.  A
    #    token-level CE is precisely the objective that asks for full sentences.
    #  * A feasibility benchmark: teacher forcing "introduced language bias and
    #    could not effectively utilize EEG signals", and recommends supervising
    #    with TEXT EMBEDDINGS and scoring with embedding correlation instead.
    #
    # The measurement agrees.  On the last sub-08 run with this weight at 1.0 the
    # held-in TRAIN token CE fell 6.35 -> 1.51 while the VALIDATION CE ROSE
    # 6.90 -> 8.03 -- it memorised and got worse at generalising, and it was 47% of
    # the total loss while doing so.  Its scored contribution is replaced by
    # `--w-anchor` below.  The term is kept switchable only so the regression can
    # be reproduced as an ablation.
    ap.add_argument("--w-text-decode", type=float, default=0.0)
    ap.add_argument("--w-text-emb", type=float, default=0.3)
    ap.add_argument("--w-sem-align", type=float, default=1.0)
    ap.add_argument("--w-vae", type=float, default=1.0)
    ap.add_argument("--w-clip", type=float, default=1.0)
    ap.add_argument("--w-clip-mse", type=float, default=1.0,
                    help="ENIGMA-style MSE of the unit CLIP-image prediction against "
                         "encode_image().  Together with `--w-clip` (cosine) and "
                         "`--w-clip-nce` (in-batch InfoNCE vs the IMAGE embeddings, "
                         "not the text concept bank) this is the vision objective.")
    ap.add_argument("--w-clip-nce", type=float, default=1.0,
                    help="symmetric in-batch InfoNCE of predicted CLIP-image vs the "
                         "true encode_image() embeddings.  Replaces the old "
                         "bank_nce(clip, text_concept_bank), which pulled the vision "
                         "head toward CLIP-TEXT centroids.")
    ap.add_argument("--w-irepa", type=float, default=0.5)
    ap.add_argument("--w-image", type=float, default=0.5)
    # ---- M2 granularity ladder: the anchor head that REPLACES token decoding
    ap.add_argument("--w-anchor", type=float, default=1.0)
    ap.add_argument("--n-anchor", type=int, default=384)
    ap.add_argument("--anchor-min-count", type=int, default=12,
                    help="an anchor word must appear in at least this many TRAIN "
                         "descriptions.  The vocabulary is built from TRAIN text "
                         "only, so it cannot carry test information.")
    ap.add_argument("--anchor-topk", type=int, default=2,
                    help="PER-FIELD CAP on anchor words in an assembled prompt.  This "
                         "is a cap and not the budget: `--anchor-total` sets how many "
                         "words the whole prompt may contain.  Kept so no single field "
                         "can take the entire budget.  The loss is multi-label over "
                         "the whole vocabulary and is not top-k at all.")
    ap.add_argument("--anchor-total", type=int, default=5,
                    help="TOTAL anchor words per assembled prompt, ranked ACROSS the "
                         "four fields.  Default 5 follows Brain-CLIPLM's measured "
                         "optimum of ~5 keywords: only a handful of ordered semantic "
                         "anchors survive EEG decoding, so a larger budget adds words "
                         "the head predicted at chance level.")
    ap.add_argument("--anchor-thr", type=float, default=0.0,
                    help="floor on the SPECIFICITY of a word (logit minus that word's "
                         "mean logit over held-in TRAIN rows).  0.0 = 'more than the "
                         "average trial predicts this word'.  Only the prompt "
                         "selection uses it; the loss always sees all logits.")
    ap.add_argument("--nvol-proj-dim", type=int, default=1280)
    ap.add_argument("--w-nvol", type=float, default=0.0,
                    help="OFF.  The 1280-d penultimate A/B is reported, never chosen "
                         "as the generation condition: IP-Adapter consumes the 1024-d "
                         "projected encode_image() feature.")
    ap.add_argument("--w-arb", type=float, default=0.0,
                    help="OFF.  Per-row strength/ip_scale did not track achieved "
                         "reliability; generation uses the fixed operating point.")
    # ---- M4 learned condition arbitration
    ap.add_argument("--arb-sens", type=float, default=4.0,
                    help="sensitivity of the per-row img2img strength to the model's "
                         "OWN predicted reliability difference (structural - semantic).")
    ap.add_argument("--arb-lo", type=float, default=0.66)
    ap.add_argument("--arb-hi", type=float, default=0.98)
    ap.add_argument("--arb-ip-lo", type=float, default=0.70)
    ap.add_argument("--arb-ip-hi", type=float, default=1.30)
    ap.add_argument("--arb-center", type=float, default=0.82,
                    help="the img2img strength returned when the two reliabilities "
                         "are EQUAL.  Set to the a-priori fixed operating point "
                         "(0.82) so that an inert arbitration reproduces the fixed "
                         "baseline exactly and cannot win by moving the operating "
                         "point.")
    # ---- innovation 4: information-weighted loss scheduling
    ap.add_argument("--w-sched", type=int, default=1,
                    help="re-weight the three level losses by 1 / measured "
                         "cross-prediction R^2 after a warmup.  Weights follow what "
                         "the subject's data says is recoverable instead of being "
                         "hand-set.")
    ap.add_argument("--sched-warmup", type=int, default=10,
                    help="epoch at which recoverability is measured.  Late enough "
                         "that the three levels have begun to differentiate, since "
                         "an early measurement cannot tell 'unrecoverable for this "
                         "subject' from 'not yet trained'.")
    ap.add_argument("--sched-floor", type=float, default=0.02,
                    help="floor on the measured R^2, so a level whose relationship "
                         "has not formed is not divided by ~0.  When ALL levels land "
                         "on the floor the multipliers are exactly 1.0 and the "
                         "schedule abstains.")
    ap.add_argument("--sched-clamp", type=float, default=4.0,
                    help="bound on the multiplier, so a noisy early R^2 cannot "
                         "collapse a level out of the loss.")
    ap.add_argument("--sched-rows", type=int, default=4096)
    ap.add_argument("--w-fuse-cls", type=float, default=0.0,
                    help="OFF.  fused is retired as a generation condition; a "
                         "concept-level NCE on it cannot be the headline objective.")
    ap.add_argument("--w-fuse-row", type=float, default=0.0,
                    help="OFF.  This was the term that trained fused into CLIP-TEXT "
                         "space and then handed it to an IP-Adapter that reads "
                         "CLIP-IMAGE.  The headline condition is `ip_clip_test`.")
    ap.add_argument("--w-balance", type=float, default=0.0,
                    help="REMOVED mechanism; kept only so old launch scripts still "
                         "parse.  It is never multiplied into the loss.")
    ap.add_argument("--w-nce", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.07)
    # ---- attribution (TRAIN-side, no oracle)
    ap.add_argument("--attr-draws", type=int, default=3)
    ap.add_argument("--attr-swap-draws", type=int, default=1)
    ap.add_argument("--attr-noise-draws", type=int, default=1)
    ap.add_argument("--attr-k2", type=int, default=3)
    ap.add_argument("--attr-desc-pool", type=int, default=20)
    ap.add_argument("--attr-concept-oracle", type=int, default=0,
                    help="MUST stay 0. 1 would label conditions with the test image's "
                         "own description, which is the oracle this project removed.")
    # ---- front end
    ap.add_argument("--fingerprint", type=str, default="")
    ap.add_argument("--use-front", type=int, default=0,
                    help="OFF.  PurifiedFront did not beat shared_r on sub-08 "
                         "(nofront ≈ full); the shared encoder is the intra "
                         "`shared_r` plus the existing linear map.")
    ap.add_argument("--noise-arm", type=int, default=0,
                    help="replace raw EEG with standard normal of the train per-channel "
                         "scale.  Same code, same targets, no signal.  The row it "
                         "generates is the HALLUCINATION FLOOR and the noise-instability "
                         "probe for the frozen-condition claim.")
    # ---- run
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--text-limit", type=int, default=0,
                    help="SMOKE TEST ONLY: truncate the CLIP-text encoding of the "
                         "composed TRAIN strings.  0 = all (the real run).")
    # `--export-angles` has been REMOVED with the spherical interpolation it drove.
    ap.add_argument("--preflight-only", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    (out / "conds").mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    sid = args.test_subject
    sd = f"{sid:02d}"
    print(f"[gem] sub-{sd} device={dev} front={bool(args.use_front)} "
          f"noise_arm={bool(args.noise_arm)} out={out}")

    # ============================================================ [1] load inputs
    def load_raw(split: str) -> np.ndarray:
        e = Path(args.raw_cache) / f"sub{sd}_{split}_eeg.npy"
        r = Path(args.raw_cache) / f"sub{sd}_{split}_row.npy"
        if not (e.is_file() and r.is_file()):
            raise SystemExit(f"[FATAL] raw EEG cache missing: {e}")
        x, row = np.load(e), np.load(r)
        if not np.array_equal(row, np.arange(len(row))):
            x = x[np.argsort(row)]
        return x.astype(np.float32)

    Xtr_np, Xte_np = load_raw("train"), load_raw("test")
    Ztr_np = np.load(Path(args.z_root) / f"sub-{sd}" / "shared_r_train.npy").astype(np.float32)
    Zte_np = np.load(Path(args.z_root) / f"sub-{sd}" / "shared_r_test.npy").astype(np.float32)
    assert len(Xtr_np) == len(Ztr_np), "raw EEG and shared_r row counts differ"
    print(f"[gem] raw {Xtr_np.shape}/{Xte_np.shape}  shared_r {Ztr_np.shape}/{Zte_np.shape}")

    cd = Path(args.captions_dir)
    caps_tr = [json.loads(l) for l in (cd / "captions_train.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    caps_te = [json.loads(l) for l in (cd / "captions_test.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if not (len(caps_tr) == len(Xtr_np) and len(caps_te) == len(Xte_np)):
        raise SystemExit(f"[FATAL] captions rows {len(caps_tr)}/{len(caps_te)} vs EEG "
                         f"{len(Xtr_np)}/{len(Xte_np)}")

    def concept_of(p: str) -> str:
        # The concept directory is `01271_shower_cap`; the gallery phrase list uses
        # SPACES, and this is not cosmetic -- `gal_ix[con_tr[i]]` looked the raw
        # underscore form up and raised `KeyError: 'shower_cap'`.  The strip is the
        # same normalisation `build_concept_bank` already applies.
        return Path(p).parent.name.split("_", 1)[1].replace("_", " ")

    con_tr = [concept_of(c["path"]) for c in caps_tr]
    con_te = [concept_of(c["path"]) for c in caps_te]

    # leak audit BEFORE anything is built
    cid = Path(args.clip_text_dir)
    gallery = json.loads((cid / "train" / "concept_phrases.json").read_text(encoding="utf-8"))
    bank = l2n(np.load(cid / "train" / "text_concept_clip.npy").astype(np.float32))
    if len(gallery) != len(bank):
        raise SystemExit("[FATAL] concept bank and phrase list differ in length")
    inter = {str(c).strip().lower() for c in gallery} & {c.strip().lower() for c in con_te}
    if inter:
        raise SystemExit(f"[FATAL] {len(inter)} test concepts appear in the train gallery, "
                         f"e.g. {sorted(inter)[:5]}")
    print(f"[audit] train gallery {len(gallery)} concepts, test {len(set(con_te))} "
          f"concepts, intersection 0")

    Vtr = np.load(Path(args.vae_cache) / "train_vae_latents_f16.npy", mmap_mode="r")
    Vte = np.load(Path(args.vae_cache) / "test_vae_latents_f16.npy", mmap_mode="r")
    Ctr = np.load(Path(args.clip_img_dir) / "clip_img1024_train.npy", mmap_mode="r")
    Cte = np.load(Path(args.clip_img_dir) / "clip_img1024_test.npy", mmap_mode="r")
    if Ctr.shape[1] != 1024:
        raise SystemExit(f"[FATAL] expected projected 1024-d image features, got {Ctr.shape}")
    # M3: the 1280-d penultimate activation, i.e. the layer the 1024-d feature was
    # projected FROM.  Cached by the same audited pass (`gem_clip_img.py`), which
    # also verified numerically that `ln_post[:, 0] @ proj == encode_image()`.
    _nvol_p = Path(args.clip_img_dir) / "clip_img1280_train.npy"
    _nvole_p = Path(args.clip_img_dir) / "clip_img1280_test.npy"
    Ntr = np.load(_nvol_p, mmap_mode="r") if _nvol_p.is_file() else None
    Nte = np.load(_nvole_p, mmap_mode="r") if _nvole_p.is_file() else None
    if Ntr is None or Ntr.shape[1] != args.nvol_proj_dim:
        raise SystemExit(
            f"[FATAL] M3 needs {_nvol_p} with width {args.nvol_proj_dim}.  It is "
            f"produced by `gem_clip_img.py`, which caches the penultimate ViT-H-14 "
            f"activation alongside the projected feature.  Run that step first: the "
            f"visibility comparison is between TWO layers and cannot be made with "
            f"one.")
    P = Path(args.clip_patch_npy)
    Ptr = np.load(P, mmap_mode="r") if P.is_file() else None
    if Ptr is None:
        print("[WARN] no CLIP patch cache: iREPA is reported as skipped, not zeroed")

    if args.text_limit:
        # SMOKE TEST ONLY.  Subset EVERY train-side array to the SAME first
        # `text_limit` rows, at load time, BEFORE any split is drawn.
        #
        # Truncating only the caption list would be a silent corruption rather than
        # a shortcut: `tr_sel`/`va_sel` are indices into the FULL 16 540-row train
        # array, so `con_tr[i]` and `Etr[vs]` would either raise or, worse, pair an
        # EEG row with some other row's caption.  Subsetting everything keeps every
        # index space consistent, so the smoke run exercises the real code paths on
        # a real (small) dataset.
        k = args.text_limit
        n_tr_full = len(Xtr_np)
        Xtr_np, Ztr_np = Xtr_np[:k], Ztr_np[:k]
        caps_tr, con_tr = caps_tr[:k], con_tr[:k]
        Vtr, Ctr = Vtr[:k], Ctr[:k]
        if Ntr is not None:
            Ntr = Ntr[:k]
        if Ptr is not None:
            Ptr = Ptr[:k]
        print(f"[gem] SMOKE MODE: train arrays cut {n_tr_full} -> {k} rows.  Every "
              f"number from this run is a shape check, NOT a result.")

    # ======================================================== [2] frozen T5 + CLIP
    import os
    os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/hf_gem")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput

    # `cache_dir` is NOT passed: the shell has already exported HF_HOME and
    # HF_HUB_CACHE, and passing `cache_dir=HF_HOME` would point transformers at
    # `.../hf` while the snapshots live under `.../hf/hub`, i.e. at the wrong level.
    tokz = AutoTokenizer.from_pretrained(T5_NAME)
    t5 = T5ForConditionalGeneration.from_pretrained(T5_NAME).to(dev).eval()
    for p in t5.parameters():
        p.requires_grad_(False)
    print(f"[gem] frozen {T5_NAME}: d_model {t5.config.d_model}, "
          f"d_ff {t5.config.d_ff}, vocab {t5.config.vocab_size}")
    if t5.config.d_model != T5_D_MODEL:
        raise SystemExit(f"[FATAL] {T5_NAME} d_model is {t5.config.d_model} but the "
                         f"towers are dimensioned for {T5_D_MODEL}. This is a hard "
                         f"shape mismatch, not a tuning difference.")
    t5_dim = int(t5.config.d_model)

    def t5_enc(texts: list[str], maxlen: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = tokz(texts, return_tensors="pt", padding="max_length", truncation=True,
                 max_length=maxlen)
        ids = b["input_ids"].to(dev)
        am = b["attention_mask"].to(dev)
        with torch.no_grad():
            h = t5.encoder(input_ids=ids, attention_mask=am).last_hidden_state
        return h, ids, am

    # composed strings: the SAME string is the semantic target and the prompt
    def composed(caps: list[dict], concepts: list[str]) -> list[str]:
        return [compose(c_, x) for c_, x in zip(concepts, caps)]

    # the CLIP-TEXT target of the composed string, computed at runtime so the
    # pool and the prompt cannot drift apart
    import open_clip
    ct_model, _, ct_pre = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=dev)
    ct_tok = open_clip.get_tokenizer("ViT-H-14")
    ct_model.eval()

    # ---- M3's bridge between the two layers.  `encode_image(x) == ln_post(x)[:, 0]
    # @ proj`, verified numerically by `gem_clip_img.py`, so `proj` is the EXACT
    # linear map from the 1280-d penultimate activation to the 1024-d space
    # IP-Adapter reads.  Taking it from the loaded model rather than hard-coding it
    # means the NVOL head's condition is the same object as the projected cache by
    # construction.
    _proj = getattr(ct_model.visual, "proj", None)
    if _proj is None:
        raise SystemExit("[FATAL] the loaded ViT-H-14 visual has no `proj`; M3 cannot "
                         "map the penultimate activation into the IP-Adapter space.")
    PROJ_NP = _proj.detach().float().cpu().numpy()          # (1280, 1024)
    if PROJ_NP.shape != (args.nvol_proj_dim, 1024):
        raise SystemExit(f"[FATAL] visual.proj is {PROJ_NP.shape}, expected "
                         f"({args.nvol_proj_dim}, 1024)")
    print(f"[gem] M3: visual.proj {PROJ_NP.shape} loaded; the 1280-d head is mapped "
          f"through it EXACTLY as `encode_image` does")

    @torch.no_grad()
    def clip_text(texts: list[str]) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), 128):
            b = ct_tok(texts[i:i + 128]).to(dev)
            outs.append(ct_model.encode_text(b).float().cpu().numpy())
        return l2n(np.concatenate(outs))

    print("[gem] encoding the composed TRAIN description strings with CLIP text "
          "(this is the attribution pool AND the pool's target)")
    t0 = time.time()
    # Already cut at load time when `--text-limit` is set (see SMOKE MODE above), so
    # this encodes the full 16 540 strings in a real run and only the smoke subset
    # otherwise.
    STR_tr = composed(caps_tr, con_tr)
    STR_te = composed(caps_te, con_te)
    Etr = clip_text(STR_tr)
    print(f"[gem] {len(STR_tr)} strings in {time.time()-t0:.0f}s -> {Etr.shape}")

    # concept-only strings for the swap draw, and the concept's own description pool
    by_concept: dict[str, list[int]] = {}
    for i, c in enumerate(con_tr):
        by_concept.setdefault(c, []).append(i)
    # the swap/foreign draw should be far away in the bank, not merely a different
    # row: two concepts that are near-duplicates would make "swap" a no-op
    bank_l2 = l2n(bank)
    concept_centroid = {c: l2n(Etr[v].mean(0, keepdims=True))[0] for c, v in by_concept.items()}

    # ====================================================== [2b] M2 anchor vocabulary
    #
    # THE GRANULARITY LADDER'S BOTTOM RUNG.  Instead of asking a frozen decoder for
    # a sentence, the model is asked for the SET OF CONTENT WORDS the description
    # contains, per field.  This is the granularity the EEG-to-text literature
    # measures as recoverable (Brain-CLIPLM: ~5 keywords, and full sentences score
    # BELOW that), and it is a discrete, finite vocabulary, so prompt composition
    # cannot degenerate the way greedy decoding did -- there is no decoder to run
    # away into a repeated token.
    #
    # LEAK-FREE: the vocabulary and its frequency counts come from the TRAIN
    # descriptions only.  Test descriptions are mapped through the same vocabulary
    # and words outside it are dropped; nothing about the test set selects a word.
    import re as _re
    _STOP = {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "with", "is",
        "are", "as", "by", "for", "from", "that", "this", "it", "its", "his", "her",
        "their", "there", "some", "very", "while", "into", "over", "near", "next",
        "which", "has", "have", "been", "being", "was", "were", "be", "he", "she",
        "they", "you", "we", "i", "not", "but", "also", "can", "are", "one", "two",
    }

    def content_words(s: str) -> list[str]:
        return [w for w in _re.findall(r"[a-z]+", str(s).lower())
                if w not in _STOP and len(w) > 2]

    # ---- VOCABULARY SELECTION: MUTUAL INFORMATION WITH THE CONCEPT, NOT FREQUENCY.
    #
    # The first version took the `n_anchor` MOST FREQUENT surviving words, and on the
    # real captions that is close to the worst available choice.  These descriptions
    # are VLM output, so their most frequent content words are the GENERATOR'S OWN
    # BOILERPLATE -- measured on the 16 540 TRAIN rows the top of that ranking is
    # `surface` (df 0.54), `visible` (0.23), `setting` (0.14), `slightly` (0.14),
    # `possibly` (0.11), `suggesting` (0.10), `appears` (0.10) -- words that say
    # nothing about which trial this is.  Truncating by frequency therefore spends the
    # entire budget on them, and the assembled prompt came out as
    # "razor blade. features view texture hair wooden. silver orange appearance
    #  distance fluffy. ..." in the smoke run: a real concept word plus template
    # filler.
    #
    # Frequency is the wrong statistic because a word being common is not the same as
    # a word being about the image.  What the anchor head is FOR is the EEG-to-text
    # question "which visual words does this trial support", so the ranking has to be
    # a measure of how much a word tells you about WHICH trial you are looking at.
    #
    # `I(w; concept) = p(w) * KL( p(concept | w) || p(concept) )` is exactly that, and
    # it is the word-level mutual information the EEG-to-text literature scores
    # against.  It is the product of two things that must BOTH hold: the word occurs
    # often enough to be learnable (`p(w)`, which is what frequency alone measures)
    # and it is concentrated on some concepts rather than spread evenly over all of
    # them (the KL term, which frequency alone is blind to).  A boilerplate word has
    # near-zero KL and drops out no matter how frequent it is; a word used for one
    # concept only has a large KL but a small `p(w)` and does not dominate.
    #
    # LEAK-FREE: concepts and counts come from TRAIN captions only, and the concepts
    # the head is scored on at test time are disjoint from the gallery by the audit
    # above, so no test trial contributes to the vocabulary.
    _cnt: dict[str, int] = {}
    _wcon: dict[str, dict[str, int]] = {}
    for _i, _c in enumerate(caps_tr):
        _cn = con_tr[_i]
        for _g in GRANS:
            for _w in set(content_words(_c.get(_g, "") or "")):
                _cnt[_w] = _cnt.get(_w, 0) + 1
                _d = _wcon.setdefault(_w, {})
                _d[_cn] = _d.get(_cn, 0) + 1
    _N = max(len(caps_tr), 1)
    _cprior = {c: n / _N for c, n in Counter(con_tr).items()}

    def _word_mi(w: str) -> float:
        n = _cnt[w]
        if n <= 0:
            return 0.0
        return (n / _N) * sum((k / n) * math.log((k / n) / _cprior[cn])
                              for cn, k in _wcon[w].items())

    _cand = [w for w, n in _cnt.items() if n >= args.anchor_min_count]
    _mi = {w: _word_mi(w) for w in _cand}
    anchor_words = sorted(_cand, key=lambda w: (-_mi[w], -_cnt[w], w))[: args.n_anchor]
    anchor_ix = {w: i for i, w in enumerate(anchor_words)}
    n_anchor = len(anchor_words)
    if n_anchor < 32:
        raise SystemExit(f"[FATAL] only {n_anchor} anchor words survived "
                         f"--anchor-min-count {args.anchor_min_count}.  The anchor "
                         f"head would have almost nothing to predict and the prompt "
                         f"would be a constant.")
    print(f"[gem] M2 anchor vocabulary: {n_anchor} words by mutual information with the "
          f"concept (>= {args.anchor_min_count} TRAIN descriptions, "
          f"{len(_cnt)} candidates)")
    print(f"[gem]   most informative: {' '.join(anchor_words[:12])}")
    # what frequency-truncation WOULD have picked, printed so the difference between
    # the two rules stays visible in every log rather than only in this commit
    _byfreq = sorted(_cand, key=lambda w: (-_cnt[w], w))[:12]
    print(f"[gem]   (frequency-truncation would have taken: {' '.join(_byfreq)})")

    # how many of a row's own anchors are IN the vocabulary: if this is low the
    # target is nearly empty and the head is being asked to predict nothing
    # (`A_tr`/`A_te` and the coverage print follow `anchor_multihot`'s definition
    # below, since the multi-hot needs `anchor_ix` to exist first).

    def anchor_multihot(caps: list[dict]) -> np.ndarray:
        """(N, 4, n_anchor) float32 multi-hot of the content words per field."""
        A = np.zeros((len(caps), len(GRANS), n_anchor), dtype=np.float32)
        for i, c in enumerate(caps):
            for gi, g in enumerate(GRANS):
                for w in set(content_words(c.get(g, "") or "")):
                    j = anchor_ix.get(w)
                    if j is not None:
                        A[i, gi, j] = 1.0
        return A

    A_tr = anchor_multihot(caps_tr)
    A_te = anchor_multihot(caps_te)

    # ---- THE SPECIFICITY CENTRING, AND WHY THE PRIOR CORRECTION ALONE FAILED.
    #
    # `binary_cross_entropy_with_logits` over a vocabulary where a word appears in
    # only ~9% of rows has its optimum at `logit = log(p / (1 - p))`, which is
    # NEGATIVE for every word -- about -2.3 at p = 0.09 -- so a rule of the form
    # "keep the words whose logit is positive" selects NOTHING for any row.  The
    # first smoke run assembled 200 prompts from ZERO anchor words this way.
    #
    # Subtracting the prior in log-odds fixes the SIGN but not the SELECTION.  The
    # second smoke run then produced 17.0 words per row out of a 20-word budget with
    # a pairwise Jaccard of 0.9538 -- i.e. nearly the same prompt for every trial,
    # built out of words like "features", "indicating", "suggesting" that the head
    # predicts for everyone because they are frequent AND easy.  A prior adjustment
    # cannot remove that: the head is genuinely confident about them, and it is
    # confident in the same direction for every row.
    #
    # So the quantity that is actually removed here is the ROW-INDEPENDENT component
    # of the prediction: the word's mean logit over HELD-IN TRAIN rows.  What is left
    # is the specificity score -- "how much more does THIS trial predict this word
    # than the average trial does" -- which is the only version of the head's output
    # that belongs in a per-row prompt, and the one whose collapse the Jaccard
    # diagnostic can actually detect.
    #
    # TRAIN rows as the reference, never the test batch: a cross-row statistic taken
    # over the test set would be label-free but transductive, and the rest of this
    # pipeline deliberately keeps every such reference inside TRAIN.
    #
    # `anchor_mu_tr` itself is computed further down, in the export block, because it
    # needs the model's logits over the held-in rows and those only exist after the
    # forward pass.

    # how many of a row's own anchors are IN the vocabulary: if this is low the
    # target is nearly empty and the head is being asked to predict nothing
    _n_own = np.array([len({w for g in GRANS for w in set(content_words(c.get(g, "") or ""))})
                       for c in caps_tr], dtype=np.float32)
    _n_cov = A_tr.sum((1, 2))
    print(f"[gem] anchor coverage on TRAIN: {_n_cov.mean():.2f} of the "
          f"{_n_own.mean():.2f} content words per row are in-vocabulary "
          f"({100 * _n_cov.sum() / max(_n_own.sum(), 1):.1f}%); "
          f"{float((_n_cov == 0).mean()):.4f} of rows have an EMPTY target")

    # ====================================================== [3] fingerprint front end
    fpp = Path(args.fingerprint) if args.fingerprint else (out / "front.pt")
    if fpp.is_file():
        fp, front_rep = load_fingerprint(fpp)
        print(f"[gem] fingerprint loaded from {fpp}")
    else:
        print("[gem] fitting the subject fingerprint (TRAIN rows only)")
        fp, front_rep = fit_fingerprint(Xtr_np, 250.0)
        save_fingerprint(fp, front_rep, fpp)
    (out / "front.json").write_text(json.dumps(front_rep, indent=2), encoding="utf-8")

    # ================================================================ [4] datasets
    n_tr = len(Xtr_np)
    n_va = int(args.val_frac * n_tr)
    perm = np.random.default_rng(args.seed).permutation(n_tr)
    va_sel, tr_sel = perm[:n_va], perm[n_va:]
    if args.limit_train:
        tr_sel = tr_sel[: args.limit_train]
    print(f"[gem] train {len(tr_sel)} held-in val {len(va_sel)} test {len(Xte_np)}")

    # concept name -> gallery index, built ONCE.  This was a dict comprehension
    # inside the batch loop: 1654 entries rebuilt for every row of every batch.
    gal_ix = {c: i for i, c in enumerate(gallery)}

    Xtr_mu = Xtr_np[tr_sel].mean(axis=(0, 2), keepdims=True)
    Xtr_sd = np.clip(Xtr_np[tr_sel].std(axis=(0, 2), keepdims=True), 1e-6, None)
    print(f"[gem] EEG scale: mean|mu| {np.abs(Xtr_mu).mean():.4g}  sd {Xtr_sd.mean():.4g}")

    # ============================ INPUT CONSTRUCTION, IN EXACTLY ONE PLACE
    #
    # `--noise-arm` is supposed to answer "does the result come from the EEG at all".
    # It did not, and the reason is a textbook multi-site-control bug worth recording
    # because the failure was invisible in every log line and every headline metric:
    #
    #   * The model consumes TWO inputs -- the raw EEG `x` and the pretrained
    #     encoder's latent `shared_r` -- and `PurifiedFront.forward` ALWAYS emits the
    #     `shared_r` token, even when `use_front` is False (it returns that token
    #     alone).  So `shared_r` is a live EEG path in every arm.
    #   * The ablation was applied at only 2 of the 8 `forward_all` call sites, both
    #     times to `x` only.  `shared_r` was passed through unchanged everywhere.
    #   * The TEST export (`ote`) applied no ablation AT ALL, so the noise arm's
    #     generated conditions were inferred from the REAL test EEG.
    #
    # Measured consequences on the sub-08 run: the `noise` arm's exported condition
    # agreed with the `full` arm's ROW BY ROW at cosine +0.9938 (ip_clip), +0.9905
    # (ip_sem), +0.9724 (ip_fused).  A control that moves the condition by 1% has not
    # controlled for anything, and a paired arm comparison against it is measuring
    # initialisation noise while being reported as an EEG ablation.
    #
    # The fix is not to add the missing noising at the other 6 sites -- that is how
    # this happened.  It is to route EVERY call through this one function, so a
    # control can only ever be applied to both streams at once.  Test-time uses the
    # same function, so the exported condition is consistent with what was trained.
    Ztr_sd = np.clip(Ztr_np[tr_sel].std(axis=0, keepdims=True), 1e-6, None)
    print(f"[gem] shared_r scale: sd {Ztr_sd.mean():.4g}  (a SECOND live EEG path; "
          f"`--noise-arm` must silence BOTH or it is not a control)")

    def _matched_noise(a: np.ndarray, sd: np.ndarray, seed: int) -> np.ndarray:
        """Gaussian noise with the same shape and per-feature spread as `a`."""
        return (np.random.default_rng(seed).standard_normal(a.shape).astype(np.float32)
                * sd)

    def inputs(rows, *, source: str = "train", seed: int | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """(x, shared_r) tensors for `rows`, with `--noise-arm` applied to BOTH.

        `seed=None` samples fresh noise; pass a fixed seed for a reproducible export.
        The noise is resampled per optimisation step rather than fixed once, so the
        model cannot learn a constant noise pattern and appear to be reading it.
        """
        if source == "train":
            X = np.stack([Xtr_np[i] for i in rows]); Z = np.stack([Ztr_np[i] for i in rows])
            sdx, sdz = Xtr_sd, Ztr_sd
        else:
            X = np.stack([Xte_np[i] for i in rows]); Z = np.stack([Zte_np[i] for i in rows])
            sdx = np.clip(Xte_np.std(axis=(0, 2), keepdims=True), 1e-6, None)
            sdz = np.clip(np.asarray(Zte_np).std(axis=0, keepdims=True), 1e-6, None)
        if args.noise_arm:
            s = args.seed if seed is None else seed
            X = _matched_noise(X, sdx, s)
            Z = _matched_noise(Z, sdz, s + 101)
        return to_t(X).float(), to_t(Z).float()

    if args.noise_arm:
        print("[gem] NOISE ARM: raw EEG AND shared_r are both replaced by "
              "spread-matched Gaussian noise, at TRAIN, VALIDATION and EXPORT.  The "
              "two streams are noised independently (seeds s and s+101) so the pair "
              "carries no shared row-specific content.")

    # sampling frame for the draw-averaged semantic supervision: for each row, a
    # description from ANOTHER TRAIN trial of the SAME concept
    rng = np.random.default_rng(args.seed + 1)

    # ============================================================ [5] model + loss
    model = GEMNet(fp, d=args.d_model, tf_layers=args.tf_layers,
                   tf_heads=args.tf_heads, drop=args.dropout,
                   n_ch=Xtr_np.shape[1], n_time=Xtr_np.shape[2], sfreq=250.0,
                   use_front=bool(args.use_front), vae_out=args.vae_out,
                   t5_dim=t5_dim, n_anchor=n_anchor,
                   nvol_dim=args.nvol_proj_dim).to(dev)
    # the last layers of every read-out start small-random rather than exactly
    # zero: an exactly-zero head makes the first gradient a function of a zero
    # vector (the defect that froze an earlier run for 754 steps)
    for m_ in [model.txt_proj, model.pool_proj, model.vae_proj, model.clip_proj,
               model.spatial_proj, model.fuse_in, model.anchor_proj,
               model.nvol_proj, model.arb_proj] + list(model.fuse.modules()):
        if isinstance(m_, nn.Linear):
            nn.init.normal_(m_.weight, std=1e-2)
            if m_.bias is not None:
                nn.init.zeros_(m_.bias)

    params = [p for p in model.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in params)
    n_front = sum(p.numel() for p in model.front.parameters() if p.requires_grad)
    print(f"[gem] learnable {n_par/1e6:.3f}M, of which front end {n_front} "
          f"({100*n_front/max(n_par,1):.3f}%)")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(tr_sel) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=0.15, anneal_strategy="cos")

    def to_t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev)

    # ---------------------------------------------------------------- preflight
    print("[gem] PREFLIGHT: forward+backward on a small batch")
    rs = tr_sel[: min(16, len(tr_sel))]
    o, dg = model.forward_all(*inputs(rs, seed=args.seed + 9000))
    # The probe builds the GVM terms rather than the retired token CE: its job is to
    # prove a FINITE GRADIENT REACHES EVERY PARAMETER, so it has to touch the heads
    # that actually carry the objective now.  `l_ce` is included only when it is
    # switched on, because with `--w-text-decode 0` the decode is dead weight.
    l_align = (1.0 - (l2t(o["s_code"]) * to_t(Etr[rs]).float()).sum(-1)).mean()
    l_anchor = F.binary_cross_entropy_with_logits(o["anchor"], to_t(A_tr[rs]))
    l_nvol = (1.0 - (l2t(o["nvol"]) * l2t(to_t(
        np.stack([Ntr[i] for i in rs])).float())).sum(-1)).mean()
    l_arb = F.mse_loss(o["r_est"], torch.full_like(o["r_est"], 0.5))
    l_ce = torch.zeros((), device=dev)
    if args.w_text_decode > 0:
        h_, ids_, am_ = t5_enc([STR_tr[i] for i in rs], 48)
        l_ce = t5(encoder_outputs=BaseModelOutput(last_hidden_state=o["pool"][:, None, :]),
                  labels=ids_.masked_fill(am_ == 0, -100), return_dict=True).loss
    loss = (args.w_text_decode * l_ce + l_anchor + l_align + l_nvol + l_arb)
    opt.zero_grad()
    loss.backward()
    total, ok = clip_checked(params, 1.0)
    nz = sum(1 for p in params if p.grad is not None and p.grad.abs().sum() > 0)

    # ---- DEGENERACY GUARD.  A condition can be finite, trainable and worthless.
    #
    # This check exists because the first sub-08 run produced EXACTLY that: every
    # exported condition had c_self = 1.000, i.e. one constant vector repeated for
    # all 200 test rows, and the run still reported `trained=True`, a decreasing
    # loss and a clean gradient.  The cause was a 15x-too-large `nn.Linear` bias on
    # the EEG projection layers, which made every trial's tokens identical, which
    # zeroed the gradient on that path -- so it could never recover either.
    #
    # Row-to-row cosine is the direct measurement: 1.0 means every row is the same
    # row, i.e. the EEG cannot control generation no matter what the loss says.  It
    # is computed HERE, on real data before any training, so the failure costs a
    # minute instead of a 20-hour job plus the generation and evaluation after it.
    with torch.no_grad():
        fs = o["fused"].float()
        fn = fs / fs.norm(dim=1, keepdim=True).clamp_min(1e-12)
        S = fn @ fn.T
        off = S[~torch.eye(len(fn), dtype=torch.bool, device=fn.device)]
        f_rowcos = float(off.mean())
        mu = fs.mean(0, keepdim=True)
        c_self = float((fn * (mu / mu.norm().clamp_min(1e-12))).sum(1).mean())
        # the EEG branch specifically: a healthy front end is MORE discriminative
        # than its input, not less
        if o.get("field") is not None:
            fld = o["field"].float().reshape(len(fs), -1)
            fld = fld / fld.norm(dim=1, keepdim=True).clamp_min(1e-12)
            Sf = fld @ fld.T
            field_rowcos = float(Sf[~torch.eye(len(fld), dtype=torch.bool,
                                               device=fld.device)].mean())
        else:
            field_rowcos = float("nan")
    print(f"[gem] DEGENERACY GUARD: fused row-cos {f_rowcos:.4f}, c_self {c_self:.4f}, "
          f"EEG-token row-cos {field_rowcos:.4f} (raw EEG reference ~0.19)")
    # 0.90 is deliberately loose: a real condition can be strongly concentrated
    # (the TRAIN concept bank's own c_self is 0.76), so this targets the degenerate
    # regime, not an aggressive one.
    if f_rowcos > 0.90 or field_rowcos > 0.90:
        raise SystemExit(
            f"[FATAL] the EEG branch is DEGENERATE before training: fused row-cos "
            f"{f_rowcos:.4f}, EEG-token row-cos {field_rowcos:.4f}.  Every trial "
            f"produces nearly the same vector, so the condition cannot be "
            f"row-specific and generation cannot be EEG-driven -- training would "
            f"still look successful.  Check for a constant offset that is large "
            f"relative to the signal (the usual cause is an `nn.Linear` bias on a "
            f"path whose activations are small).")
    print(f"[gem] PREFLIGHT loss {float(loss):.4f} = "
          f"text_decode {float(l_ce):.3f} + sem_align {float(l_align):.3f} "
          f"[reduced 2-term probe only] | grad_norm {total:.4g} finite={ok} "
          f"params_with_grad={nz}/{len(params)}")
    # NOTE: this probe deliberately builds only `l_ce + l_align`.  It is a
    # "does a gradient reach every parameter" test, not a loss-balance test, so the
    # per-term split that catches an inflated normaliser lives in the EPOCH loop
    # below, where all ten terms actually exist.  Reporting the loop's local names
    # here raised
    #     UnboundLocalError: cannot access local variable 'l_emb'
    # because they are assigned inside that loop and not in this block.
    if not ok or nz == 0:
        raise SystemExit("[FATAL] preflight: the gradient is non-finite or no parameter "
                         "received one. Training would silently do nothing.")
    opt.zero_grad()
    if args.preflight_only:
        print("[gem] --preflight-only: exiting after a clean preflight")
        return

    # ================================================================== [6] train
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    hist: list[dict] = []

    # ---- INNOVATION 4: INFORMATION-WEIGHTED LEVEL SCHEDULING.
    #
    # The three level losses were hand-weighted (1.0 each), which is an implicit
    # claim that equal information is recoverable at each level.  That claim is
    # false and it is MEASURABLE -- and the last sub-08 run is what it costs: the
    # token CE took 47% of the total loss at weight 1.0 while its held-in
    # validation loss ROSE (6.90 -> 8.03) against a falling training loss
    # (6.35 -> 1.51).  That is memorisation, and it was bought with roughly half the
    # objective.
    #
    # So the level weights are DERIVED rather than set.  `_info_weights` fits a
    # ridge from each level's own activation to that level's TARGET on held-in
    # TRAIN rows and reads off the validated R^2; the multiplier is C / mean(C),
    # clamped.  A level the subject's data supports gets more of the objective; a
    # level it does not support gets less, because gradient on an unrecoverable
    # target can only fit noise.
    #
    # WHAT THIS CANNOT DO, STATED PLAINLY.  C is measured on the model's own
    # activations, so an EARLY measurement cannot separate "unrecoverable for this
    # subject" from "not yet trained".  Three consequences follow and all three are
    # handled by reporting rather than by hoping:
    #   * the measurement is taken at `--sched-warmup`, late enough (default 10 of
    #     26 epochs) that the levels have begun to differentiate;
    #   * it is taken ONCE and frozen.  Re-measuring every epoch would make the
    #     objective non-stationary and turn a capacity argument into a feedback loop
    #     between the weights and the thing they weight;
    #   * if every level lands on the floor the multipliers come out at exactly 1.0,
    #     i.e. the schedule abstains and the a-priori weights stand.  The R^2 values
    #     are printed and stored in `history`, so a run where the schedule abstained
    #     is visible as such instead of being reported as an improvement.
    #
    # The clamp keeps a noisy R^2 from deleting a level, and the floor keeps a level
    # whose relationship has not formed from being divided by ~0.
    sm = si = sv = 1.0
    sched_info: dict = {"applied_epoch": None, "r2": {}, "mult": {},
                        "policy": "mult = clip(C / mean(C), 1/clamp, clamp), one-shot"}

    def _info_weights() -> tuple[float, float, float, dict]:
        model.eval()
        rs = tr_sel[: args.sched_rows]
        with torch.no_grad():
            ob, _ = model.forward_all(*inputs(rs, seed=args.seed + 8000))
        got = {
            "sem": ob["s_code"].float().cpu().numpy().astype(np.float32),
            "img": ob["clip"].float().cpu().numpy().astype(np.float32),
            "vae": ob["vae"].flatten(1).float().cpu().numpy().astype(np.float32),
        }
        want = {
            "sem": Etr[rs].astype(np.float32),
            "img": np.asarray(Ctr[rs], dtype=np.float32),
            "vae": downsample_latent(to_t(np.stack([Vtr[i] for i in rs])))
                    .flatten(1).float().cpu().numpy(),
        }
        r2 = {}
        for k in got:
            # `ridge_fit` splits internally, so the R^2 it returns is on rows the
            # ridge did not see.
            r2[k] = max(float(ridge_fit(got[k], want[k])["val_r2"]), args.sched_floor)
        mean_r2 = sum(r2.values()) / 3.0
        mult = {k: float(np.clip(v / mean_r2, 1.0 / args.sched_clamp,
                                 args.sched_clamp)) for k, v in r2.items()}
        model.train()
        return mult["sem"], mult["img"], mult["vae"], {"r2": r2, "mult": mult}

    # `l_ce` is only defined inside the batch loop.  With `--w-text-decode 0` the term
    # is never multiplied in, but the logged value must still exist, so it is given a
    # zero HERE rather than being left to raise `UnboundLocalError` on the epoch that
    # happens to skip the only branch that assigned it.
    l_ce = torch.zeros((), device=dev)

    best = {"score": -1e18, "epoch": -1, "state": None}
    grad_skips = 0
    for ep in range(args.epochs):
        # ---- innovation 4: apply the information-weighted schedule ONCE
        if (args.w_sched and ep == args.sched_warmup
                and sched_info["applied_epoch"] is None):
            sm, si, sv, _inf = _info_weights()
            sched_info = {"applied_epoch": ep, **_inf}
            print(f"[gem] INFO-WEIGHTED SCHEDULE at ep {ep}: "
                  f"recoverable R2 " + " ".join(f"{k}={v:.4f}"
                                                for k, v in _inf["r2"].items())
                  + "  ->  multipliers "
                  + " ".join(f"{k}={v:.3f}" for k, v in _inf["mult"].items())
                  + "  (1.0 = a-priori weight; the bound is "
                  f"[1/{args.sched_clamp:.0f}, {args.sched_clamp:.0f}])")
        model.train()
        order = np.random.default_rng(args.seed + ep).permutation(len(tr_sel))
        agg: dict[str, float] = {}
        nb = 0
        t0 = time.time()
        for s in range(0, len(order) - args.batch_size + 1, args.batch_size):
            bidx = order[s:s + args.batch_size]
            rows = tr_sel[bidx]
            # ---- attribution: average the prediction over draws that are labelled
            # by the ROW'S CONCEPT and described by OTHER TRAIN trials of it
            draw_src = []
            for k in range(args.attr_draws):
                for i in rows:
                    pool_i = by_concept.get(con_tr[i])
                    draw_src.append(pool_i[rng.integers(0, len(pool_i))] if pool_i else i)
            # every input -- the real batch and the attribution draws -- goes through
            # the single `inputs` entry point, so `--noise-arm` cannot be applied to
            # one of them and forgotten on the other
            o_d, _ = model.forward_all(*inputs(
                draw_src, seed=args.seed + ep * 1000 + nb + 7))
            s_draws = o_d["s_code"].reshape(args.attr_draws, len(rows), -1)

            o, dg = model.forward_all(*inputs(rows, seed=args.seed + ep * 1000 + nb))

            # ---- T1 semantic: teacher-forced decode from the predicted embeddings
            #
            # SKIPPED ENTIRELY WHEN ITS WEIGHT IS 0.  This is not just a semantics
            # change: the decode is a full T5 forward AND its label preparation over
            # 4 x B sequences per step, and with the default `--w-text-decode 0` the
            # term contributes no gradient.  Running it would be a large per-step
            # cost for a number that is only printed.
            fld_ids, fld_am = [], []
            for gi, g in enumerate(GRANS):
                b = tokz([str(caps_tr[i].get(g, "") or "") for i in rows],
                         return_tensors="pt", padding="max_length", truncation=True,
                         max_length=S_FIELD_MAXLEN)
                fld_ids.append(b["input_ids"].to(dev))
                fld_am.append(b["attention_mask"].to(dev))
            ids_cat = torch.stack(fld_ids, 1).reshape(-1, S_FIELD_MAXLEN)
            am_cat = torch.stack(fld_am, 1).reshape(-1, S_FIELD_MAXLEN)
            # PAD MUST BE -100, NOT 0.  T5's loss is
            # `CrossEntropyLoss(ignore_index=-100)`, and does NOT convert the pad
            # token for us: passing `pad_token_id = 0` as the label means the model
            # is asked to PREDICT pad at every padded position.  Descriptions run
            # ~12 tokens in a 24-token window, so over half of every batch is pad,
            # and the cheapest way to reduce that loss is to emit pad everywhere --
            # `generate()` would then decode empty strings and the semantic tower
            # would look trained while producing nothing.
            ids_cat = ids_cat.masked_fill(am_cat == 0, -100)
            if args.w_text_decode > 0:
                emb_flat = o["field"].reshape(len(rows) * 4, S_FIELD_MAXLEN, -1)
                # BATCH-MAJOR on both sides.  `o["field"]` is (B, 4, L, d), so its
                # `.reshape(B*4, L, d)` orders b0f0, b0f1, ... .  Building the token
                # ids as `cat([f[None] for f in fields])` would order f0b0..f0bB,
                # f1b0.. -- a different permutation, so the decode loss would be
                # computed from each row's embedding against ANOTHER row's words.
                # It would still descend (the words are real captions) and would
                # silently cap the semantic tower at "generic caption".  Stacking
                # then reshaping keeps the two orders identical.
                dout = t5(encoder_outputs=BaseModelOutput(last_hidden_state=emb_flat),
                          attention_mask=am_cat, labels=ids_cat, return_dict=True)
                l_ce = dout.loss

            # field-embedding regression, as support rather than as the objective
            tgt_emb, tgt_am = [], []
            for gi, g in enumerate(GRANS):
                h, _, _ = t5_enc([str(caps_tr[i].get(g, "") or "") for i in rows],
                                 S_FIELD_MAXLEN)
                tgt_emb.append(h)
                tgt_am.append((fld_am[gi] > 0).float())
            T = torch.stack(tgt_emb, 1)                       # (B,4,L,t5_dim)
            M = torch.stack(tgt_am, 1)[..., None]             # (B,4,L,1) real-token mask
            # DENOMINATOR IS TOKENS x DIMS, NOT TOKENS.
            #
            # The numerator is a `.sum()` over the last axis as well, so `.sum()` is
            # over (B,4,L,t5_dim) values while `M.sum()` counts only the masked
            # (B,4,L) TOKENS -- `M` has a trailing singleton axis and does not
            # multiply out the 768 dimensions.  Dividing one by the other made this
            # term 768x its intended value: measured on the first sub-08 run, the
            # total loss was 256.7 for a cross-entropy of 6.2 and an InfoNCE of 14.9,
            # i.e. ~230 of the 256 came from this single regression and was
            # multiplied into the loss at weight 0.3.
            #
            # The consequence was not a cosmetic scale difference.  Every other
            # objective -- the T5 text-decode CE that IS the semantic tower's
            # primary signal, the CLIP-text alignment, the two InfoNCE terms -- was
            # ~1% of the gradient, so the run would have reported a three-tower model
            # while training essentially only this auxiliary regression, which the
            # comment above it explicitly describes as support rather than the
            # objective.  With the correct normaliser the term is bounded by
            # 2*(1 - cos) <= 4, because both sides are unit-variance per token
            # (`txt_ln` is a LayerNorm and T5 ends in a variance-only `T5LayerNorm`).
            n_emb = (M.sum() * T.shape[-1]).clamp_min(1.0)
            l_emb = (((o["field"] - T) ** 2 * M).sum() / n_emb) \
                + (1.0 - ((l2t(o["field"]) * l2t(T)).sum(-1) * M[..., 0]).sum()
                   / M.sum().clamp_min(1.0))

            # ---- the composed string: pool alignment in CLIP-TEXT space
            Etg = to_t(Etr[rows]).float()
            s_self = l2t(o["s_code"])
            # the pool is trained against the row's OWN composed string (exact
            # target, batch-only, not a pool statistic) -- the DRAW averaging below
            # exists only to decide which string generation should ask for.
            # BOTH terms are (B,) cosines and MUST be reduced: `.sum(-1)` alone
            # leaves a vector and the summed `loss` below then fails on a shape
            # mismatch rather than on anything the numbers would reveal.
            l_align = (1.0 - (s_self * Etg).sum(-1)).mean()
            l_sup = (1.0 - (l2t(s_draws.reshape(-1, s_draws.shape[-1]))
                            * Etg.repeat(args.attr_draws, 1)).sum(-1)).mean()

            # ---- T2 structural
            vt = downsample_latent(to_t(np.stack([Vtr[i] for i in rows])))
            vmu, vsd = vt.mean((0, 2, 3), keepdim=True), vt.std((0, 2, 3), keepdim=True).clamp_min(1e-6)
            vn = (vt - vmu) / vsd
            vp = (o["vae"] - o["vae"].mean((0, 2, 3), keepdim=True)) \
                / o["vae"].std((0, 2, 3), keepdim=True).clamp_min(1e-6)
            l_vae = F.smooth_l1_loss(vp, vn)

            # ---- T3 CLIP image projection + iREPA
            # BOTH of these are per-row cosine terms, so they are already (B,) and
            # (B,64,1)->(B,64) BEFORE the mean.  Reducing to a scalar is not
            # cosmetic: the multi-term `loss` expression below adds them to scalars,
            # and an unreduced (B,64) raised
            #   "The size of tensor a (8) must match the size of tensor b (64)".
            Ct = to_t(np.stack([Ctr[i] for i in rows])).float()
            Ct = l2t(Ct)
            l_clip = (1.0 - (o["clip"] * Ct).sum(-1)).mean()
            l_clip_mse = F.mse_loss(o["clip"], Ct)
            l_clip_nce = pair_nce(o["clip"], Ct, args.tau)
            if Ptr is not None:
                Pt = l2t(to_t(np.stack([Ptr[i] for i in rows])).float())
                # mean over BOTH the rows and the 64 spatial tokens: the 64 tokens
                # are a feature map, not 64 independent samples
                l_rep = (1.0 - (l2t(o["spatial"]) * Pt).sum(-1)).mean()
            else:
                l_rep = torch.zeros((), device=dev)

            # ---- multi-tower: the concept the row depicts, so the two concepts of
            # one image are both positives (this is what the earlier multi-positive
            # InfoNCE was for, and it is kept)
            tix = to_t(np.asarray([gal_ix[con_tr[i]] for i in rows], dtype=np.int64))
            tgt_bank = to_t(bank).float()
            # `bank_nce` returns the value ALREADY DIVIDED by ln(n_bank) = 7.411, so
            # each term reads 1.0 at chance and is commensurable with the O(1)
            # cosine terms.  Multiplying back by ln(n_bank) recovers the raw
            # cross-entropy for the log, which is the interpretable form.
            # Language NCE stays against the CLIP-TEXT concept bank.  Vision NCE
            # is `l_clip_nce` above, against the IMAGE embeddings of this batch.
            # Mixing `o["clip"]` into the text bank was a silent space error: it
            # rewarded the IP condition for looking like a caption centroid.
            l_nce = bank_nce(o["s_code"], tgt_bank, tix, args.tau)
            l_fcls = bank_nce(l2t(o["fused"]), tgt_bank, tix, args.tau)
            _nb_ln = math.log(len(bank))
            nce_raw, fcls_raw = float(l_nce) * _nb_ln, float(l_fcls) * _nb_ln

            # ---- ROW-LEVEL SUPERVISION ON THE FUSED CONDITION ITSELF.
            #
            # This was the structural hole behind the collapsed condition.  `fused`
            # is what generation consumes, and its ONLY objective was `l_fcls`, an
            # InfoNCE against the concept bank -- i.e. CONCEPT-level supervision.
            # Nothing asked it to distinguish two trials of the same concept, and
            # `fcls` sat exactly at chance on sub-08 (raw 7.446 vs ln 1654 = 7.411)
            # while the exported condition came out with c_self 0.999995, a constant.
            # A term that only rewards concept identity cannot produce a row-specific
            # vector, so the collapse was the expected outcome of the objective, not
            # an accident of optimisation.
            #
            # The fused condition is therefore aligned to the row's own composed
            # description in CLIP-text space -- the same leak-free TRAIN target the
            # semantic tower uses.  Both row-level and concept-level objectives now
            # act on the array that is actually handed to the generator.
            l_frow = (1.0 - (l2t(o["fused"]) * Etg).sum(-1)).mean()

            # ---- M2 ANCHOR LADDER.  This term is what `--w-text-decode` used to
            # be, and it is the reason the teacher-forced CE is off.
            #
            # Per-field multi-label over a FINITE vocabulary.  Two properties the
            # token CE did not have: the target is a SET, so the loss cannot reward
            # a fluent-but-wrong word order (there is no order), and it has no
            # autoregressive decoder to run away into a repeated token -- the
            # `WRWRWRWR...` collapse is not reachable from this objective because
            # every logit is scored independently against a real word's presence.
            #
            # BCE, not softmax.  A softmax over 384 anchors would make the words
            # compete, but a description genuinely contains several co-occurring
            # content words ("orange", "tabby", "floor").
            At = to_t(A_tr[rows])
            l_anchor = F.binary_cross_entropy_with_logits(o["anchor"], At)

            # ---- M3 NVOL.  Regress the PENULTIMATE activation.  Left unnormalised
            # in the loss because the residual norm is part of what `proj` expects;
            # only the cosine is used, so the scale is free and the direction is
            # what is supervised.
            _Nt = to_t(np.stack([Ntr[i] for i in rows])).float()
            l_nvol = (1.0 - (l2t(o["nvol"]) * l2t(_Nt)).sum(-1)).mean()

            # ---- M4 ARBITRATION.  The target is the model's OWN per-level accuracy
            # on this very row, detached.  That is what makes the head a
            # self-assessment rather than a guess: it is asked to predict, from the
            # EEG alone, how well it is about to do at each level.  Both targets are
            # cosines mapped to [0, 1] so they live in the sigmoid's range.
            # `.detach()` is load-bearing -- without it the cheapest way to reduce
            # this loss is to make the ACHIEVEMENT worse, not the estimate better.
            #
            # The two levels are exactly the two generation hyper-parameters that
            # are otherwise hand-set: index 0 feeds `ip_scale`, index 1 feeds
            # img2img `strength`.  The structural target is the VAE latent cosine,
            # not the CLIP-image cosine, because the structural CONDITION in
            # generation is the img2img init and that init IS the VAE latent.
            with torch.no_grad():
                r_sem_t = 0.5 * (1.0 + (s_self * Etg).sum(-1))
                # Against the RAW low latent `vt`, NOT the batch-standardised `vn`.
                # `vn` is (vt - batch mean) / (batch sd), so a cosine against it is a
                # function of the other rows in the same batch: the regression target
                # for "how well did I predict the structure of THIS trial" would shift
                # every time the batch composition changed, training `r_est` to track
                # sampling noise.  The quantity being predicted is the raw latent, so
                # that is what the reliability has to be measured against.
                r_str_t = 0.5 * (1.0 + (l2t(o["vae"].flatten(1))
                                        * l2t(vt.flatten(1))).sum(-1))
                r_tgt = torch.stack([r_sem_t, r_str_t], 1)
            l_arb = F.mse_loss(o["r_est"], r_tgt)

            # ---- NO balance loss.  The three towers' contribution is DIAGNOSED by
            # the cross-prediction matrix (`tower_attribution`), not enforced by a
            # regulariser: a hand-set objective on internal norms could only hide a
            # collapsed tower, and the point of the matrix is to reveal it.
            #
            # The LEVEL weights below are SCHEDULED, not fixed: from
            # `--sched-warmup` on they are 1/measured-recoverability (see
            # `_info_weights`).  `sm`/`si`/`sv` are 1.0 before the first
            # measurement, so the early epochs run on the a-priori weights.
            loss = (args.w_text_decode * l_ce + args.w_text_emb * l_emb
                    + args.w_anchor * l_anchor
                    + args.w_sem_align * sm * (l_align + l_sup)
                    + args.w_vae * sv * l_vae
                    + args.w_clip * si * l_clip
                    + args.w_clip_mse * si * l_clip_mse
                    + args.w_clip_nce * si * l_clip_nce
                    + args.w_clip * si * args.w_nvol * l_nvol
                    + args.w_irepa * l_rep
                    + args.w_arb * l_arb
                    + args.w_nce * l_nce + args.w_fuse_cls * l_fcls
                    + args.w_fuse_row * l_frow)
            opt.zero_grad()
            loss.backward()
            total, ok = clip_checked(params, 1.0)
            if not ok:
                grad_skips += 1
                opt.zero_grad()
                continue
            opt.step()
            sched.step()
            nb += 1
            for k, v in (("loss", loss), ("ce", l_ce), ("emb", l_emb), ("align", l_align),
                         ("sup", l_sup), ("vae", l_vae), ("clip", l_clip),
                         ("clip_mse", l_clip_mse), ("clip_nce", l_clip_nce),
                         ("rep", l_rep),
                         ("nce", l_nce), ("fcls", l_fcls), ("frow", l_frow),
                         ("anchor", l_anchor), ("nvol", l_nvol), ("arb", l_arb),
                         ("nce_raw", nce_raw), ("fcls_raw", fcls_raw)):
                agg[k] = agg.get(k, 0.0) + float(v)
        if nb == 0:
            raise SystemExit("[FATAL] no optimiser step was taken this epoch")
        agg = {k: v / nb for k, v in agg.items()}

        # ---- held-in validation on TRAIN rows only
        model.eval()
        with torch.no_grad():
            vs = va_sel[: min(1024, len(va_sel))]
            ov, _ = model.forward_all(*inputs(vs, seed=args.seed + 7000 + ep))
            Ev = to_t(Etr[vs]).float()
            v_align = float((l2t(ov["s_code"]) * Ev).sum(-1).mean())
            Cv = l2t(to_t(np.stack([Ctr[i] for i in vs])).float())
            v_clip = float((l2t(ov["clip"]) * Cv).sum(-1).mean())
            vv = downsample_latent(to_t(np.stack([Vtr[i] for i in vs])))
            v_vae = float(1.0 - F.smooth_l1_loss(
                (ov["vae"] - ov["vae"].mean((0, 2, 3), keepdim=True))
                / ov["vae"].std((0, 2, 3), keepdim=True).clamp_min(1e-6),
                (vv - vv.mean((0, 2, 3), keepdim=True))
                / vv.std((0, 2, 3), keepdim=True).clamp_min(1e-6)))
            score = 0.4 * v_clip + 0.4 * v_align + 0.2 * v_vae
            b = tokz([STR_tr[i] for i in vs], return_tensors="pt", padding="max_length",
                     truncation=True, max_length=48)
            # same -100 masking as the training loss: without it the validation
            # number would be dominated by "did you emit pad", which is not the
            # quantity being selected on
            vlab = b["input_ids"].to(dev).masked_fill(b["attention_mask"].to(dev) == 0, -100)
            decv = t5(encoder_outputs=BaseModelOutput(last_hidden_state=ov["pool"][:, None, :]),
                      labels=vlab, return_dict=True)
            v_ce = float(decv.loss)
        # AGGREGATE LOSSES CAN HIDE A BROKEN BALANCE.  The first sub-08 run had a
        # normaliser bug that made `emb` ~768x its intended size -- roughly 230 of a
        # 256.7 total -- and nothing in the log said so: the loss fell, the gradient
        # was finite, every parameter updated.  Nine tenths of the objective was one
        # auxiliary term the code itself calls "support rather than the objective".
        # So the first epoch states the split and flags any single term that owns
        # most of the total.
        if ep == 0:
            _parts = (("text_decode", args.w_text_decode * agg["ce"]),
                      ("text_emb", args.w_text_emb * agg["emb"]),
                      ("sem_align", args.w_sem_align * (agg["align"] + agg["sup"])),
                      ("vae", args.w_vae * agg["vae"]),
                      ("clip", args.w_clip * agg["clip"]),
                      ("clip_mse", args.w_clip_mse * agg["clip_mse"]),
                      ("clip_nce", args.w_clip_nce * agg["clip_nce"]),
                      ("irepa", args.w_irepa * agg["rep"]),
                      ("nce_lang", args.w_nce * agg["nce"]),
                      ("fuse_cls", args.w_fuse_cls * agg["fcls"]),
                      ("fuse_row", args.w_fuse_row * agg["frow"]))
            print("[gem] loss split "
                  + "  ".join(f"{k}={v:.3f}" for k, v in _parts))
            _bn, _bv = max(_parts, key=lambda kv: kv[1])
            if _bv > 0.75 * agg["loss"]:
                print(f"[WARN] one term dominates: {_bn}={_bv:.3f} of "
                      f"{agg['loss']:.3f}. Verify the weights and the normalisers "
                      f"before reading a three-tower result out of this run.")
        row = {"epoch": ep, "sec": time.time() - t0, "grad_skips": grad_skips,
               "val_pool_decode_ce": v_ce, "val_align_cos": v_align,
               "val_clip_cos": v_clip, "val_vae": v_vae, "score": score,
               "sched": dict(sched_info), **agg}
        hist.append(row)
        print(f"[gem] ep {ep:02d} loss {agg['loss']:.3f} "
              f"| terms ce {agg['ce']:.3f} emb {agg['emb']:.3f} "
              f"anc {agg['anchor']:.3f} nvol {agg['nvol']:.3f} arb {agg['arb']:.3f} "
              f"align {agg['align']:.3f} sup {agg['sup']:.3f} vae {agg['vae']:.3f} "
              f"clip {agg['clip']:.3f} cmse {agg['clip_mse']:.3f} "
              f"cnce {agg['clip_nce']:.3f} rep {agg['rep']:.3f} "
              f"nce {agg['nce']:.3f} fcls {agg['fcls']:.3f} frow {agg['frow']:.3f} "
              f"| raw nce {agg['nce_raw']:.3f} fcls {agg['fcls_raw']:.3f} "
              f"(chance {math.log(len(bank)):.3f}) "
              f"| val: ce {v_ce:.3f} align {v_align:.3f} "
              f"clip {v_clip:.3f} vae {v_vae:.3f} score {score:.4f} "
              f"({row['sec']:.0f}s, skips {grad_skips})")
        if score > best["score"]:
            best = {"score": score, "epoch": ep,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            torch.save({"model": model.state_dict(), "args": vars(args), "fp": fp},
                       out / "best.pth")
        torch.save({"model": model.state_dict(), "epoch": ep}, out / "last.pth")
        (out / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    print(f"[gem] selected epoch {best['epoch']} (held-in TRAIN score {best['score']:.4f})")

    # ============================================================== [7] export
    # No theta to solve: the condition is the fusion output and its concentration is
    # handled per row by `gem_calib.py` against the TRAIN concept bank.

    with torch.no_grad():
        # TEST export goes through the SAME `inputs` entry point as training.  It
        # previously applied no ablation at all, so in the noise arm the generated
        # conditions were inferred from the REAL test EEG -- the arm was not a
        # control in any sense, it was a differently-initialised model reading the
        # same signal, which is exactly what the measured row-wise agreement of
        # 0.994 with the `full` arm showed.
        te_x, te_z = inputs(np.arange(len(Xte_np)), source="test", seed=args.seed + 4242)
        ote, _ = model.forward_all(te_x, te_z)
        # CHUNKED.  The front end materialises (B, NB, C, T) per stream, so a
        # single call over 14886 train rows allocates ~1.3 GB before the trunk even
        # starts.  It fits on an 80 GB H800 but it is pure waste and would OOM on
        # anything smaller, and the export is the one place where the row count is
        # the whole training set rather than a batch.
        otr = {}
        with torch.no_grad():
            buf: dict[str, list[np.ndarray]] = {}
            for s in range(0, len(tr_sel), 2048):
                r = tr_sel[s:s + 2048]
                ob, _ = model.forward_all(*inputs(r, seed=args.seed + 4243))
                for k, v in ob.items():
                    buf.setdefault(k, []).append(v.float().cpu().numpy())
            otr = {k: np.concatenate(v, 0) for k, v in buf.items()}
    # ---- greedy prompt from the pool embedding, through the FROZEN T5 decoder
    def decode_prompt(emb: np.ndarray, gen: int = 28) -> list[str]:
        """Greedy decode through the frozen T5, from EITHER of two ranks.

        `pool` is a single vector per row, `(B, d)`, and is the T5 encoder's one
        position, so it must be given a sequence axis.  `field` is ALREADY a sequence,
        `(B, L, d)` -- `o["field"]` is `(B, 4, L, d)` and one field of it is `(B, L, d)`
        -- so it is the encoder input as-is and must NOT get another axis.

        The old code applied `pe[:, None, :]` unconditionally.  On the 3-D field
        tensor that indexing yields a 4-D `(B, 1, L, d)`, and T5 then fails inside the
        decoder with the misleading
            ValueError: too many values to unpack (expected 3)
        from `encoder_batch_size, encoder_sequence_length, _ =
        encoder_hidden_states.size()`.  The rank is decided here instead of being
        assumed, so the two call sites can pass what they actually hold.
        """
        outs = []
        for i in range(0, len(emb), 64):
            pe = to_t(emb[i:i + 64]).float()
            if pe.dim() == 2:
                pe = pe[:, None, :]                     # (B, d)   -> (B, 1, d)
            elif pe.dim() != 3:
                raise ValueError(f"decode_prompt expects a (B, d) pooled vector or a "
                                 f"(B, L, d) sequence, got shape {tuple(pe.shape)}")
            g = t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=pe),
                            max_new_tokens=gen, num_beams=1, do_sample=False)
            outs.extend(tokz.batch_decode(g, skip_special_tokens=True))
        return outs

    pool_te = ote["pool"].cpu().numpy()

    # ================================================== [7a] M2 THE ANCHOR PROMPT
    #
    # THE PROMPT IS NO LONGER DECODED, IT IS ASSEMBLED FROM DISCRETE WORDS.
    #
    # The previous design ran `generate()` through the frozen T5 over a predicted
    # embedding.  On sub-08 that produced "WRWRWRWRWRWRWRWRWR..." for every one of
    # 200 rows, so `gem_swap` and `gem_ll_self` scored IDENTICALLY and the text
    # demonstrably could not have influenced an image.  That is not a tuning failure
    # -- it is the documented instability of decoding a caption directly from an
    # aligned neural representation, and it is unreachable from the design below:
    # there is no autoregressive decoder here, so there is no degenerate continuation
    # to fall into.  Each field's words are the top-k of an INDEPENDENT set of logits,
    # scored against a real word's presence.
    #
    # Granularity is `--anchor-topk` per field, which is where the EEG-to-text
    # literature measures the information to actually be (~5 keywords), rather than
    # the full sentence the token CE was asking for.
    anchor_te = ote["anchor"].float().cpu().numpy()               # (n, 4, n_anchor)
    anchor_words_arr = np.asarray(anchor_words)

    # ---- THE SPECIFICITY REFERENCE.  See the long note above `anchor_mu_tr`'s use
    # site for why the prior correction alone was not enough; this is the quantity
    # that replaced it.  Computed here because it needs the model's logits over the
    # held-in TRAIN rows, which do not exist until the export forward pass.
    anchor_mu_tr = otr["anchor"].mean(0).astype(np.float32)      # (4, n_anchor)
    _spread = float(np.abs(otr["anchor"] - anchor_mu_tr[None]).mean())
    print(f"[gem] M2 specificity reference: per-word TRAIN mean logit, mean absolute "
          f"row-deviation {_spread:.4f} (0.0 = the head predicts the same logits for "
          f"every trial, i.e. no row-specific content to put in a prompt)")

    def field_words(row_logits: np.ndarray, gi: int, k: int, thr: float) -> list[str]:
        """Top-k words for one field, ranked by SPECIFICITY against the TRAIN mean.

        Kept because `prompts_field_<g>` still needs a per-field read-out for the
        grounding measurement; the ASSEMBLED prompt no longer uses it (see
        `row_words`, which ranks across fields under a global budget).

        `thr` is a floor on the ADJUSTED score, not on the raw logit.  A floor of 0
        means "more than the average trial predicts this word", which is the only
  1660|        reading under which an absolute threshold is meaningful here.  Two measured
        failures stand behind that:
          * on the RAW logit every word in the vocabulary sits below 0 at the BCE
            optimum (~ -2.3 at p = 0.09), so a raw floor of 0 selected NOTHING and
            0.0% of rows had a single anchor word;
          * subtracting only the PRIOR fixes the sign but not the selection -- it
            still admitted the 17.0-of-20 words that every row shares, for a pairwise
            Jaccard of 0.9538, because words that are frequent are also easy and the
            head is row-independently confident about them.
  1670|        See `anchor_mu_tr` for the full derivation.
        """
        adj = np.asarray(row_logits, dtype=np.float32) - anchor_mu_tr[gi]
        k = min(k, len(anchor_words_arr))
        ix = np.argsort(-adj)[:k]
        keep = [int(j) for j in ix if adj[j] > thr]
        return [str(anchor_words_arr[j]) for j in keep]

    def row_words(i: int) -> dict[str, list[str]]:
        """Field -> words for ONE row, under a GLOBAL budget across the four fields.

        WHY A GLOBAL BUDGET.  The EEG-to-text literature's most consistent finding is
        that only a handful of ORDERED SEMANTIC ANCHORS survive decoding, not a
        sentence: Brain-CLIPLM measures the optimum at about 5 keywords and reports
        that scoring more of them buys nothing.  The previous rule was a fixed
        `--anchor-topk` PER FIELD, i.e. `4 * anchor_topk` words per prompt -- 20 at the
        default, four times the recoverable amount -- and the smoke run confirmed it by
        filling 19.5 of 20 slots on every row.  A prompt budget larger than what the
        EEG carries does not add information; it adds words the head predicted at
        chance level, and those are exactly the words that make the generated image
        drift away from the trial.

        Ranking ACROSS fields rather than within each one is the second half of the
        fix.  A fixed 5-per-field split forces every prompt to spend budget on a
        `background` phrase even when that trial's evidence is entirely in `subject`,
        and it makes the four fields compete only for slots they cannot share.  Sorting
        all fields' candidates together lets each trial spend its budget where its own
        evidence is strongest, which is the per-trial allocation the literature's
        "ordered anchors" result implies.

        `--anchor-topk` survives as a PER-FIELD CAP, so no single field can take the
        whole budget and the prompt cannot degenerate back into one field's word salad.
        `thr` is a floor on the SPECIFICITY score, not on the raw logit -- see
        `anchor_mu_tr` for why the raw logit had to be abandoned.
        """
        cand: list[tuple[float, int, str]] = []
        for gi in range(len(GRANS)):
            adj = anchor_te[i, gi] - anchor_mu_tr[gi]
            k = min(args.anchor_topk, len(anchor_words_arr))
            for j in np.argsort(-adj)[:k]:
                if adj[j] > args.anchor_thr:
                    cand.append((float(adj[j]), gi, str(anchor_words_arr[j])))
        cand.sort(key=lambda t: (-t[0], t[1], t[2]))
        out: dict[str, list[str]] = {g: [] for g in GRANS}
        for _s, gi, w in cand[: max(int(args.anchor_total), 0)]:
            out[GRANS[gi]].append(w)
        return out

    # the concept word: nearest entry of the 1654-concept TRAIN gallery to the row's
    # predicted composed-description embedding.  Leak-free by the same argument the
    # whole run relies on -- the audit asserted the test concepts are disjoint.
    W_tc = ridge_fit(otr["pool"].astype(np.float32), Etr[tr_sel].astype(np.float32))
    r_tc = ridge_apply(W_tc, pool_te.astype(np.float32))
    concept_ix = (l2n(r_tc) @ bank.T).argmax(1)

    # M2's own degeneracy check, because a CONSTANT PROMPT is the failure this
    # section exists to remove and it must be impossible to miss: if every row
    # assembles the same words, swapping prompts changes nothing and the text tower
    # is again inert no matter how good its loss looks.
    _frag = {"self": [row_words(i) for i in range(len(anchor_te))]}
    prompts_self = [compose(str(gallery[int(ci)]),
                            {g: " ".join(_frag["self"][i][g]) for g in GRANS})
                    for i, ci in enumerate(concept_ix)]
    _uniq_p = len(set(prompts_self))
    _sets = [set(w for g in GRANS for w in _frag["self"][i][g])
             for i in range(len(anchor_te))]
    _sizes = np.array([len(s) for s in _sets], dtype=np.float32)
    # Rows with NO word at all are counted separately rather than folded into the
    # Jaccard: two empty sets have an undefined Jaccard, and treating 0/0 as 0 made
    # the first smoke run report "jaccard 0.0000", which reads as MAXIMALLY specific
    # when the truth was that there was nothing to compare.
    _nonempty = float((_sizes > 0).mean())
    # Mean pairwise Jaccard over a sample of rows: how much the assembled word sets
    # differ row to row.  1.0 means every row says the same thing.
    _rng_j = np.random.default_rng(0)
    _ja = []
    for _ in range(2000):
        a, b = _rng_j.integers(0, len(_sets), 2)
        u = _sets[a] | _sets[b]
        if u:
            _ja.append(len(_sets[a] & _sets[b]) / len(u))
    jaccard = float(np.mean(_ja)) if _ja else float("nan")
    print(f"[gem] M2 prompt assembly: {_uniq_p} unique of {len(prompts_self)} "
          f"({100 * _uniq_p / len(prompts_self):.1f}%), "
          f"{100 * _nonempty:.1f}% of rows have at least one anchor word, "
          f"mean {_sizes.mean():.2f} anchor words per row "
          f"(budget {args.anchor_total} total, cap {args.anchor_topk}/field), "
          f"mean pairwise Jaccard {jaccard:.4f} (1.0 = constant prompt)")
    if _nonempty < 0.5:
        print(f"[WARN] M2 ANCHOR STARVATION: only {100 * _nonempty:.1f}% of test rows "
              f"produced ANY anchor word above the log-odds-ratio floor "
              f"{args.anchor_thr}.  The prompt is then the concept word alone, the "
              f"text condition is close to constant, and `gem_swap` cannot be read. "
              f"Lower `--anchor-thr` or raise `--n-anchor`.")
    elif _uniq_p < 0.5 * len(prompts_self):
        print(f"[WARN] M2 PROMPT COLLAPSE: only {_uniq_p} unique assembled prompts. "
              f"The anchor head is not producing row-specific words, so the text "
              f"condition is again close to a constant.")
    # kept as a MEASURED diagnostic only: this is the path that produced the constant
    # string, so a change to it stays visible instead of silent
    prompts_legacy_t5 = decode_prompt(pool_te)
    print(f"[gem]   legacy direct-from-pool T5 decode "
          f"{len(set(prompts_legacy_t5))} unique (NOT used for generation)")
    for p_ in prompts_self[:3]:
        print(f"    pred: {p_[:110]}")
    for p_ in STR_te[:2]:
        print(f"    true: {p_[:110]}")
    print(f"[gem] T5-pool -> CLIP-text ridge val R2 {W_tc['val_r2']:.4f}")
    Etg_te = clip_text(STR_te)
    print(f"[gem] pool->CLIP-text cosine: ridge {float((l2n(r_tc) * Etg_te).sum(-1).mean()):.4f} "
          f"| model linear head "
          f"{float((l2t(ote['s_code']) * to_t(Etg_te).float()).sum(-1).mean()):.4f} "
          f"| centreline {float((l2n(Etg_te.mean(0, keepdims=True)) * Etg_te).sum(-1).mean()):.4f}")
    # anchor accuracy: the ladder's own score, on the words the model was asked for.
    # Scored on the ADJUSTED scores, because that is the ranking the prompt uses --
    # scoring the raw logits would measure a different rule than the one shipped.
    _adj = anchor_te - anchor_mu_tr[None]
    _anc_pred = _adj > args.anchor_thr
    anc_stats = {
        "per_field_precision": {g: float((_anc_pred[:, gi] & (A_te[:, gi] > 0)).sum()
                                         / max(_anc_pred[:, gi].sum(), 1))
                                for gi, g in enumerate(GRANS)},
        "per_field_recall": {g: float((_anc_pred[:, gi] & (A_te[:, gi] > 0)).sum()
                                      / max((A_te[:, gi] > 0).sum(), 1))
                             for gi, g in enumerate(GRANS)},
        "prompt_unique": _uniq_p, "prompt_n": len(prompts_self),
        "prompt_jaccard": jaccard, "rows_with_any_anchor": _nonempty,
        "anchors_per_row_mean": float(_sizes.mean()),
        "threshold": float(args.anchor_thr),
        "rule": ("a word is selected when its logit exceeds its TRAIN field prior's "
                 "log-odds by more than `threshold`, i.e. it is more likely here than "
                 "on average; this is a comparison BETWEEN rows, not an absolute one")}

    # ================================================ [7b] M3 WHICH LAYER IS T3?
    #
    # The EEG-to-image literature's central claim is that different visual components
    # have DIFFERENT neural visibility, and that aligning to a FIXED final layer is a
    # cross-modal mismatch.  That claim is testable here rather than assumed, because
    # both candidate layers are cached: the 1280-d penultimate activation and the
    # 1024-d feature projected from it.  `proj` maps one to the other EXACTLY, so the
    # comparison is "same object, with or without the 256 directions the projection
    # discards" -- a fair A/B on one variable.
    #
    # The choice is made on HELD-IN TRAIN rows, never on test, and the losing branch
    # is reported.  A margin that is inside the noise would be an honest "either
    # works"; a large one is evidence about where this subject's EEG is visible.
    proj_t = to_t(PROJ_NP).float()
    with torch.no_grad():
        ova, _ = model.forward_all(*inputs(va_sel, seed=args.seed + 4244))
    Cva = l2n(np.asarray(Ctr[va_sel], dtype=np.float32))
    _p_1024 = l2n(ova["clip"].float().cpu().numpy())
    _p_nvol = l2n((l2t(ova["nvol"]) @ proj_t).float().cpu().numpy())
    vis = {"cos_direct_1024": float((l2n(_p_1024) * Cva).sum(-1).mean()),
           "cos_penultimate_1280_through_proj":
               float((l2n(_p_nvol) * Cva).sum(-1).mean())}
    use_nvol = vis["cos_penultimate_1280_through_proj"] > vis["cos_direct_1024"]
    vis["chosen"] = "direct_1024"
    vis["reported_winner_if_ab"] = "nvol_1280" if use_nvol else "direct_1024"
    vis["margin"] = abs(vis["cos_penultimate_1280_through_proj"] - vis["cos_direct_1024"])
    vis["note"] = (
        "A/B is REPORTED, not selected.  The generation condition is always the "
        "1024-d encode_image() prediction: that is what IP-Adapter was trained on.  "
        "Choosing the 1280-d penultimate by a held-in cosine previously handed the "
        "adapter a mapped-through-proj vector it does not consume.")
    print(f"[gem] M3 A/B (reported only): direct-1024 "
          f"{vis['cos_direct_1024']:.4f} vs penultimate-1280->proj "
          f"{vis['cos_penultimate_1280_through_proj']:.4f}  "
          f"(would-have-chosen {vis['reported_winner_if_ab']}, "
          f"margin {vis['margin']:.4f}).  GENERATION uses direct_1024.")

    ip_test = ote["clip"].float().cpu().numpy()

    # ============================================ [7c] M4 LEARNED ARBITRATION
    #
    # THE OPEN PROBLEM THIS CLOSES.  Both 2026 low-level-feature papers state it in
    # the same words: the balance between semantic and structural conditioning is
    # "currently managed by a sensitive hyperparameter", and the way forward is a
    # fusion that varies "without manual intervention".  `SD_STRENGTH=0.82` and
    # `IP_SCALE=1.0` in `run_gem_intra.sh` ARE that manual intervention.
    #
    # Here each row gets its own pair, taken from the model's own predicted
    # reliability for that row (the M4 head, supervised above against its realised
    # accuracy).  Direction, stated so it can be checked against the arrow of the
    # mechanism:
    #   * structural reliable  -> KEEP the img2img init -> LOWER  strength
    #   * semantic  reliable  -> trust the IP condition  -> HIGHER ip_scale
    #
    # CENTRED ON THE ESTABLISHED OPERATING POINT, and that is the whole reason this
    # is an honest ablation rather than a re-tuning: the offsets are measured against
    # the held-in TRAIN mean, not zero, so the MEAN strength is exactly 0.82 and the
    # MEAN ip_scale exactly 1.0 -- the same point `fixalpha` generates at.  Only the
    # row-to-row SPREAD differs.  Any gap between the two arms is therefore
    # attributable to per-row adaptation and nothing else.
    r_va = ova["r_est"].float().cpu().numpy()
    r_bar = r_va.mean(0)
    r_sd = np.maximum(r_va.std(0), 1e-4)
    # a gain placing +-2 sd of the model's own reliability spread across the full
    # range: self-normalising, so it does not depend on the head's absolute scale
    gain_str = (args.arb_hi - args.arb_lo) / (4.0 * float(r_sd[1]))
    gain_sem = (args.arb_ip_hi - args.arb_ip_lo) / (4.0 * float(r_sd[0]))
    r_te = ote["r_est"].float().cpu().numpy()
    alpha_test = np.clip(args.arb_center - gain_str * (r_te[:, 1] - r_bar[1]),
                         args.arb_lo, args.arb_hi)
    ipsc_test = np.clip(1.0 + gain_sem * (r_te[:, 0] - r_bar[0]),
                        args.arb_ip_lo, args.arb_ip_hi)
    np.save(out / "alpha_test.npy", alpha_test.astype(np.float32))
    np.save(out / "ipscale_test.npy", ipsc_test.astype(np.float32))
    arb = {
        "center_strength": args.arb_center, "strength_range": [args.arb_lo, args.arb_hi],
        "ipscale_range": [args.arb_ip_lo, args.arb_ip_hi],
        "held_in_mean_r_sem": float(r_bar[0]), "held_in_mean_r_str": float(r_bar[1]),
        "held_in_sd_r_sem": float(r_sd[0]), "held_in_sd_r_str": float(r_sd[1]),
        "alpha_mean": float(alpha_test.mean()), "alpha_sd": float(alpha_test.std()),
        "alpha_min": float(alpha_test.min()), "alpha_max": float(alpha_test.max()),
        "ipscale_mean": float(ipsc_test.mean()), "ipscale_sd": float(ipsc_test.std()),
        "ipscale_min": float(ipsc_test.min()), "ipscale_max": float(ipsc_test.max()),
    }

    # ---- IS THE PER-ROW OPERATING POINT INFORMED, OR JUST NOISE?
    #
    # The obvious thing to report here would be corr(alpha, r_str) and
    # corr(ip_scale, r_sem).  Both are USELESS numbers, and the first smoke run
    # printed them as -1.0000 and +1.0000, which reads as a perfect mechanism and is
    # in fact an identity: `alpha_test` is defined as `arb_center - gain * (r_str -
    # mean)`, an affine function of `r_str`, so its Pearson correlation with `r_str`
    # is -1 by construction for ANY `r_est`, including a constant-free random one.
    # The sign is a design decision, not an empirical finding, and reporting it as an
    # r of 1.0 invites exactly the wrong conclusion.
    #
    # What M4 actually claims is that the head can tell, PER TRIAL and from the EEG
    # evidence alone, whether this trial's semantic or structural read-out is the
    # more trustworthy one.  So the check is calibration: how well does the predicted
    # reliability track the reliability the model ACHIEVED on the same row?  A head
    # that outputs a constant scores 0.0 here by definition, so any positive r is
    # real signal, and a permutation of the targets gives the null for free.
    _h, _w = Vte.shape[-2] // 4, Vte.shape[-1] // 4
    _zl = np.asarray(Vte, dtype=np.float32)
    _vlow = _zl.reshape(len(_zl), _zl.shape[1], _h, 4, _w, 4).mean((-3, -1))
    _vlow = _vlow.reshape(len(_zl), -1)
    _sem_act = 0.5 * (1.0 + (l2n(ote["s_code"].float().cpu().numpy())
                             * l2n(Etg_te)).sum(-1))
    _str_act = 0.5 * (1.0 + (l2n(ote["vae"].flatten(1).float().cpu().numpy())
                             * l2n(_vlow)).sum(-1))
    _rr = np.random.default_rng(0)
    for _nm, _pred, _act in (("sem", r_te[:, 0], _sem_act),
                             ("str", r_te[:, 1], _str_act)):
        _r = float(np.corrcoef(_pred, _act)[0, 1])
        _r0 = float(np.corrcoef(_pred, _rr.permutation(_act))[0, 1])
        arb[f"corr_r_est_{_nm}_vs_achieved"] = _r
        arb[f"corr_r_est_{_nm}_vs_permuted"] = _r0
        arb[f"achieved_{_nm}_mean"] = float(_act.mean())
        arb[f"achieved_{_nm}_sd"] = float(_act.std())
        print(f"[gem]   M4 calibration {_nm}: r_est vs ACHIEVED {_r:+.4f} "
              f"(permuted null {_r0:+.4f}; 0.0 = no better than a constant "
              f"reliability, which is what `fixalpha` assumes)")
    # stated so the direction is not mistaken for a result
    print(f"[gem]   (corr(alpha, r_est_str) is {float(np.corrcoef(alpha_test, r_te[:, 1])[0, 1]):+.4f} "
          f"BY CONSTRUCTION -- alpha is an affine function of r_est, so this is a "
          f"design choice, not evidence)")
    print(f"[gem] M4 arbitration: alpha {arb['alpha_mean']:.4f} +- {arb['alpha_sd']:.4f} "
          f"[{arb['alpha_min']:.3f}, {arb['alpha_max']:.3f}]; "
          f"ip_scale {arb['ipscale_mean']:.4f} +- {arb['ipscale_sd']:.4f} "
          f"[{arb['ipscale_min']:.3f}, {arb['ipscale_max']:.3f}]")
    if arb["alpha_sd"] < 0.005:
        print(f"[WARN] M4 ARBITRATION IS INERT: alpha sd {arb['alpha_sd']:.5f} across "
              f"rows.  The head predicts the same reliability for every trial, so this "
              f"arm reduces to the fixed operating point and cannot be claimed as a "
              f"gain over `fixalpha`.")
    elif min(arb["corr_r_est_sem_vs_achieved"],
             arb["corr_r_est_str_vs_achieved"]) < 0.05:
        print(f"[WARN] M4 ARBITRATION IS NOISE: alpha still spreads "
              f"(sd {arb['alpha_sd']:.4f}), but the predicted reliability does not "
              f"track achieved reliability (r {arb['corr_r_est_sem_vs_achieved']:+.4f} "
              f"sem / {arb['corr_r_est_str_vs_achieved']:+.4f} str).  The per-row "
              f"parameter then moves for reasons unrelated to the row, and the "
              f"`gem_arb` vs `gem_fixalpha` gap would be read as adaptation when it "
              f"is sampling noise.")

    W_ci = None
    citr = np.load(Path(args.clip_img_dir) / "clip_img1024_train.npy", mmap_mode="r")
    W_ci = ridge_fit(Etr[tr_sel].astype(np.float32),
                     np.asarray(citr[tr_sel], dtype=np.float32))
    print(f"[gem] CLIP-text -> CLIP-image ridge val R2 {W_ci['val_r2']:.4f}")

    # ==================================================== [7b] tower attribution
    # THE QUESTION: is the three-tower split real, or is one tower a relabelling of
    # another?  A 3x3 matrix of cross-prediction ridges answers it: fit on TRAIN
    # rows, score on TEST rows, for every (source activation, target) pair.  If, say,
    # sem->clip_img scores as high as clip->clip_img, the CLIP tower is not carrying
    # anything the semantic tower does not already have, and the honest conclusion is
    # to drop it rather than to report a three-tower result.
    #
    # Fitted and scored here rather than in a separate script because Etr/Etg_te and
    # the activations all exist in this process; reloading them elsewhere would mean
    # recomputing the CLIP-text pass over 16540 strings.
    src_tr = {"sem": otr["s_code"].astype(np.float32),
              "clip": otr["clip"].astype(np.float32),
              "vae": otr["vae"].reshape(len(tr_sel), -1).astype(np.float32)}
    src_te = {"sem": ote["s_code"].float().cpu().numpy().astype(np.float32),
              "clip": ote["clip"].float().cpu().numpy().astype(np.float32),
              "vae": ote["vae"].flatten(1).float().cpu().numpy().astype(np.float32)}

    def low_latent(a) -> np.ndarray:
        """(...,4,64,64) -> (...,4,16,16) by 4x4 block average, in numpy.

        Identical to `downsample_latent` (which uses `avg_pool2d`) so the target of
        this cross-prediction is the same object the VAE tower predicts: (4,16,16)
        flattened to 1024.  Written as a reshape-mean because doing 16 540
        `avg_pool2d` calls would cost more than the ridge it feeds.
        """
        x = np.asarray(a, dtype=np.float32)
        h, w = x.shape[-2] // 4, x.shape[-1] // 4
        return x.reshape(*x.shape[:-2], h, 4, w, 4).mean((-3, -1))

    tgt_tr = {"txt": Etr[tr_sel].astype(np.float32),
              "img": np.asarray(citr[tr_sel], dtype=np.float32),
              "vae": low_latent(np.asarray(Vtr[tr_sel])).reshape(len(tr_sel), -1),
              # M3's target, in the matrix so the visibility claim is checkable in
              # the same table as everything else: if the penultimate layer were
              # just a relabelling of the projected feature, `sem->nvol` and
              # `sem->img` would move together.
              "nvol": np.asarray(Ntr[tr_sel], dtype=np.float32)}
    tgt_te = {"txt": Etg_te.astype(np.float32),
              "img": np.asarray(Cte, dtype=np.float32),
              "vae": low_latent(np.asarray(Vte)).reshape(len(Xte_np), -1),
              "nvol": np.asarray(Nte, dtype=np.float32)}
    xmat: dict = {}
    for sn, X in src_tr.items():
        for tn, Y in tgt_tr.items():
            W = ridge_fit(X, Y)
            P = ridge_apply(W, src_te[sn])
            xmat[f"{sn}->{tn}"] = {
                "val_r2": W["val_r2"],
                "test_cos": float((l2n(P) * l2n(tgt_te[tn])).sum(-1).mean()),
                "row_identity": row_identity_acc(P, tgt_te[tn]),
                "chance_row_identity": 1.0 / len(P)}
    print("[gem] 3x3 tower attribution (source -> target), test cosine:")
    for tn in tgt_tr:
        cells = "  ".join(f"{sn}:{xmat[f'{sn}->{tn}']['test_cos']:.4f}" for sn in src_tr)
        print(f"    target {tn:<4} {cells}")
    # is any tower's OWN target better predicted by another tower?  That is the
    # redundancy verdict, stated as a number rather than as a claim.
    redundancy = {}
    for own_src, own_tgt in (("sem", "txt"), ("clip", "img"), ("vae", "vae")):
        base = xmat[f"{own_src}->{own_tgt}"]["test_cos"]
        beats = {sn: xmat[f"{sn}->{own_tgt}"]["test_cos"] for sn in src_tr}
        redundancy[f"{own_src}->{own_tgt}"] = {
            "own": base, "others": {k: v for k, v in beats.items() if k != own_src},
            "best_other_minus_own": float(max(v for k, v in beats.items()
                                               if k != own_src) - base)}
    for k, v in redundancy.items():
        print(f"    {k}: own {v['own']:.4f}, best other "
              f"{max(v['others'].values()):.4f} (delta {v['best_other_minus_own']:+.4f})")

    # ---- FIELD grounding gate.  Is each description field concept-specific at all?
    #
    # NOT a classifier: the train and test CONCEPT SETS ARE DISJOINT, so a probe
    # fitted to train labels cannot be scored on test labels -- the label spaces do
    # not overlap.  The measurable question is whether a test row's field embedding
    # is nearest to another TEST row of the SAME concept.  Under a random pairing the
    # expected rate is sum_c (n_c - 1) / (N - 1); a field at or below that carries
    # no concept information for this read-out.  The label-shuffled column is the
    # control: it must sit at chance.
    def nn_same_concept(E: np.ndarray, labels: list[str], csls_k: int = 10) -> dict:
        """Same-concept top-1 rate, RAW and CSLS-CORRECTED.

        The CSLS column is not decoration.  `2*cos(i,j) - r(i) - r(j)`, with `r` the
        mean of a row's top-k similarities to all other rows, removes the advantage a
        HUB row gets from being near everything rather than near its own concept --
        which is precisely the signature of a partially collapsed read-out.  The
        EEG-to-image literature reports this correction moving Top-1 from 78.1% to
        86.4% on the same predictions, so reporting raw cosine alone would both
        understate the method and hide how much of the raw score is hub structure.
        `mean_r` is the hub statistic itself.
        """
        Z = l2n(E.astype(np.float32))
        S = Z @ Z.T
        np.fill_diagonal(S, -np.inf)
        nn = S.argmax(1)
        lab = np.asarray(labels)
        rate = float(np.mean(lab[nn] == lab))
        cnt = {c: int((lab == c).sum()) for c in set(labels)}
        chance = float(sum(v * (v - 1) for v in cnt.values())
                       / (len(labels) * (len(labels) - 1)))
        k = int(min(csls_k, S.shape[1] - 1))
        r = np.sort(S, axis=1)[:, -k:].mean(1) if k > 0 else np.zeros(len(S))
        Sc = 2.0 * S - r[:, None] - r[None, :]
        rate_csls = float(np.mean(lab[np.argmax(Sc, 1)] == lab))
        return {"same_concept_nn_rate": rate, "chance": chance,
                "lift": rate - chance,
                "same_concept_nn_rate_csls": rate_csls,
                "lift_csls": rate_csls - chance, "csls_k": k,
                "hub_mean_r": float(r.mean()), "hub_sd_r": float(r.std())}

    # `otr` was built by the chunked loop above, so it is already ON THE CPU as
    # numpy.  `.float().cpu()` are torch calls: on a numpy array they do not exist,
    # and leaving them in fails with `AttributeError: 'numpy.ndarray' object has no
    # attribute 'float'` after the whole training run has finished.
    field_mean_te = ote["field"].float().mean(2).cpu().numpy()      # (n,4,768)
    field_mean_tr = otr["field"].mean(2).astype(np.float32)
    ground: dict = {}
    for gi, g in enumerate(GRANS):
        m = nn_same_concept(field_mean_te[:, gi], con_te)
        m["token_norm_cv"] = float(
            (ote["field"][:, gi].float().norm(dim=-1).std(-1)
             / ote["field"][:, gi].float().norm(dim=-1).mean(-1).clamp_min(1e-9)).mean())
        m["grounded"] = bool(m["lift"] > 0.02)
        ground[g] = m
    for g, m in ground.items():
        print(f"    field {g:<10} same-concept NN {m['same_concept_nn_rate']:.4f} "
              f"(CSLS {m['same_concept_nn_rate_csls']:.4f}) vs "
              f"chance {m['chance']:.4f} (lift {m['lift']:+.4f}) "
              f"grounded={m['grounded']}")
    # the SAME measurement for the fused condition, which is what generation consumes
    ground["_fused_condition"] = nn_same_concept(
        np.asarray(ote["fused"].float().cpu().numpy()), con_te)
    ground["_prompt_pool"] = nn_same_concept(r_tc, con_te)

    # ---- save what the offline report needs.  fp16: the analysis standardises
    # anyway, and this is 16540 x 3840 floats, i.e. 127 MB instead of 254 MB.
    np.savez(out / "acts_train.npz", **{k: v.astype(np.float16)
                                        for k, v in src_tr.items()},
             field_mean=field_mean_tr.astype(np.float16),
             pool=otr["pool"].astype(np.float16))
    np.savez(out / "acts_test.npz", **{k: v.astype(np.float16)
                                       for k, v in src_te.items()},
             field_mean=field_mean_te.astype(np.float16),
             pool=pool_te.astype(np.float16))

    def export_cond(name: str, arr: np.ndarray) -> str:
        pth = out / "conds" / f"{name}.npy"
        np.save(pth, l2n(arr.astype(np.float32)))
        return str(pth)

    conds = {}
    # HEADLINE = V_img = the 1024-d encode_image() prediction.  fused is still
    # written (so older skip/require paths do not crash) but it is a COPY of clip,
    # not the text-supervised fusion that previously scored +0.11 vs the image.
    conds["clip"] = export_cond("ip_clip_test", ip_test)
    conds["img"] = export_cond("ip_img_test", ip_test)
    conds["fused_auto"] = export_cond("ip_fused_auto_test", ip_test)
    conds["sem"] = export_cond("ip_sem_test", ridge_apply(W_ci, r_tc))
    conds["clip_loser"] = export_cond("ip_clip_loser_test",
                                      l2n((l2t(ote["nvol"]) @ proj_t)
                                          .float().cpu().numpy()))
    conds["fused"] = conds["fused_auto"]
    print("[gem] MAIN generation condition = ip_clip_test "
          "(1024-d encode_image(); fused_auto is an alias of the same array)")

    # ---- prompts
    def write_prompts(name: str, texts: list[str]) -> str:
        p = out / "prompts" / f"{name}.json"
        p.write_text(json.dumps(texts, indent=1), encoding="utf-8")
        return str(p)

    prompts = {"self": write_prompts("prompts_self", prompts_self),
               "generic": write_prompts("prompts_generic", [GENERIC_PROMPT] * len(prompts_self))}
    # per-field prompts, for the grounding measurement.  Now ASSEMBLED FROM ANCHORS
    # rather than decoded: `prompts_field_<g>` is that field's top-k words, which is
    # the same object the loss supervised, so grounding is measured on what was
    # actually optimised.
    field_prompts = {}
    for gi, g in enumerate(GRANS):
        field_prompts[g] = write_prompts(
            f"prompts_field_{g}",
            [" ".join(_frag["self"][i][g]) for i in range(len(anchor_te))])
    write_prompts("prompts_true", STR_te)

    # ---- oracle row (never fed to the model): if a condition built FROM this
    # row's own description does not beat the predicted one, generation is not
    # reading the condition and that has to be visible rather than assumed
    conds["oracle"] = export_cond("ip_oracle_test", ridge_apply(W_ci, Etg_te))
    # ---- the EXACT constant control.  `calibrate_quantile` gives a STATISTICALLY
    # matched row, but a generation pipeline can also be fed the battery's mean
    # condition outright.  If `gem_stat` scores what `gem_ll_self` scores, the EEG
    # is not being read -- this is the shortest possible version of that test and it
    # costs one generation row.
    conds["static"] = export_cond(
        "ip_static_test", np.repeat(otr["clip"].mean(0, keepdims=True), len(Xte_np), 0))
    # ---- the img2img init.  The VAE tower predicts the LOW-FREQUENCY latent
    # (4,16,16); `generate_atm_aligned_decode.py --vae-latent-npy` decodes full
    # (4,64,64) latents by dividing by `scaling_factor` first, so the 16x16 map is
    # upsampled in LATENT space and stored already scaled.  Nearest-neighbour, not
    # bilinear: bilinear on a latent is a smoothing operator with no meaning in the
    # VAE's own coordinates, whereas nearest simply holds each low-frequency
    # coefficient across its 4x4 block, which is exactly the block-constant
    # structure that average-pooling the target in `downsample_latent` assumes.
    np.save(out / "vae_low_test.npy", ote["vae"].cpu().numpy().astype(np.float32))
    lu = F.interpolate(ote["vae"].float(), scale_factor=4, mode="nearest")
    np.save(out / "pred_vae_test_scaled.npy",
            (lu * 0.13025).cpu().numpy().astype(np.float32))

    # ================================================================= [8] report
    def top1(pred: np.ndarray, targ: np.ndarray) -> float:
        return row_identity_acc(pred, targ)

    hub = {"n": int(len(Etg_te))}
    rep: dict = {
        "subject": sid, "protocol": "intra-subject, leak-free",
        "use_front": bool(args.use_front), "noise_arm": bool(args.noise_arm),
        "main_condition": "ip_clip_test",
        "main_condition_space": "clip_image_1024_encode_image",
        "trained": bool(best["epoch"] >= 0), "grad_skips": grad_skips,
        "best": {"epoch": best["epoch"], "score": best["score"]},
        "history_tail": hist[-3:],
        "n_learnable": int(n_par), "n_front_learnable": int(n_front),
        "condition_concentration": _cond_degeneracy(ote),
        "fingerprint": front_rep,
        "frozen_condition_check": {
            "pool_to_clip_text_cos": float((l2n(r_tc) * Etg_te).sum(-1).mean()),
            "clip_tower_cos": float((l2n(ip_test) * l2n(to_t(np.asarray(
                Cte, dtype=np.float32)).float().cpu().numpy())).sum(-1).mean()),
            "t3_layer_used": vis["chosen"],
            "row_identity_acc_pool": top1(r_tc, Etg_te),
            "row_identity_acc_clip": top1(ote["clip"].cpu().numpy(),
                                          np.asarray(Cte, dtype=np.float32)),
            "chance_row_identity": 1.0 / len(Etg_te),
            "note": ("row identity is top-1 retrieval of a row's own target among all "
                     "test rows with the diagonal masked: 1/n is chance.  A condition "
                     "that is a constant of the battery scores chance by construction.")},
        "attribution": {
            "draws_self": args.attr_draws, "draws_swap": args.attr_swap_draws,
            "draws_noise": args.attr_noise_draws,
            "concept_oracle_used_as_model_input": bool(args.attr_concept_oracle),
            "note": ("draws are labelled by the row's CONCEPT and described by other "
                     "TRAIN trials of that concept.  The test image's own description "
                     "is the ORACLE row and is never an input.")},
        "ridges": {"t5pool_to_cliptext_val_r2": W_tc["val_r2"],
                   "cliptext_to_clipimg_val_r2": W_ci["val_r2"]},
        "tower_attribution": xmat,
        "tower_redundancy": redundancy,
        "field_grounding": ground,
        "w_text_decode": args.w_text_decode,
        "M2_anchor_ladder": {
            "n_vocab": int(n_anchor), "min_count": args.anchor_min_count,
            "topk_per_field": args.anchor_topk, "total_per_prompt": args.anchor_total,
            "w_anchor": args.w_anchor,
            "vocab_rule": "mutual information with the concept, TRAIN only",
            "top_informative_words": anchor_words[:24],
            **anc_stats,
            "note": ("the semantic tower has TWO objectives with different jobs, and "
                     "they are not interchangeable:\n"
                     "  PRIMARY (drives generation) -- `l_align`/`l_emb`: the EEG latent "
                     "is regressed onto the CLIP-text encoding of the trial's "
                     "description, and the generation condition is built by mapping "
                     "that predicted encoding into the IP-Adapter image space "
                     "(`conds['sem']`).  NO words are generated on this path, which is "
                     "the point: the EEG-to-text literature's robust result is that "
                     "embedding-level alignment is what transfers, while lexical "
                     "decoding is where it degrades.\n"
                     "  AUXILIARY (interpretability + EEG-to-text consistency) -- "
                     "`l_anchor`: a per-field set of in-vocabulary content words, "
                     "scored with the word-level precision/recall above.  It exists so "
                     "the claim is comparable to EEG-to-text work and so the assembled "
                     "`prompts_self` is inspectable; it is NOT the source of the "
                     "generation condition.\n"
                     "`prompt_jaccard` is the check that the assembled prompt is "
                     "row-specific: 1.0 would mean the text condition is again a "
                     "constant and `gem_swap` cannot be read.")},
        "M3_neural_visibility": {**vis,
            "note": ("which layer serves as T3 is decided by a held-in A/B between the "
                     "1024-d projected feature and the 1280-d penultimate activation "
                     "passed through the model's own `visual.proj`.  The losing branch "
                     "is reported so a margin inside the noise stays visible.")},
        "M4_arbitration": {**arb,
            "note": ("per-row img2img strength and IP-Adapter scale, predicted from the "
                     "model's own reliability for that row and CENTRED on the held-in "
                     "mean, so the mean reproduces the fixed 0.82 / 1.0 operating point "
                     "exactly.  A gap versus the `fixalpha` arm is therefore per-row "
                     "adaptation and not a moved operating point.")},
        "innovation4_weight_schedule": sched_info,
        "conds": conds, "prompts": prompts, "prompts_field": field_prompts,
        "bank_dim": int(bank.shape[1]),
        "hubness": hub,
    }
    # per-field grounding is measured above (`nn_same_concept`) and lives in `ground`
    (out / "gem_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[gem] wrote {out/'gem_report.json'}")
    print(f"[gem] frozen-condition check: pool->CLIP-text cos "
          f"{rep['frozen_condition_check']['pool_to_clip_text_cos']:.4f} | row identity "
          f"{rep['frozen_condition_check']['row_identity_acc_pool']:.4f} "
          f"(chance {1/len(Etg_te):.4f})")
    print(f"[gem] conds: {sorted(conds)}")


if __name__ == "__main__":
    main()
