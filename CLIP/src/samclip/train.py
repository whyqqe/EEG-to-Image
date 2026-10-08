"""Stage A: single-stage training of the shared cross-subject EEG encoder.

    All source subjects, all 1654 train concepts, no validation holdout, last-epoch
    selection -- the SOTA protocol for this task (see PROTOCOL_INTER.md §3). The
    held-out subject's data is never seen during this stage except by the periodic
    *evaluation*, which is why `--epochs 0` is a legitimate "score the initialisation"
    run.

WHAT THIS STAGE NO LONGER IS
----------------------------
v1 had three stages and a mapping network. All three are gone, and each removal is a
measured decision rather than a simplification for its own sake:

  * **No Stage 1 subject conditioning.** Known subjects used to be addressed through an
    embedding table and modulated by FiLM/LoRA. Removed: the learned subject vector
    collapsed to a constant (cosine 0.9999 across subjects) because nothing in the
    objective rewarded it being informative. See ``models/subject_conditioning.py``.
  * **No Stage 2 episodic meta-training.** It existed only to train the hypernetwork
    that inferred ``z_s`` from a support set. With the hypernetwork gone there is
    nothing for it to train.
  * **No Stage B mapping network.** A second network mapping EEG embeddings into the
    image space was proposed and then dropped: it is a linear map fitted between two
    representations that the calibration stage already fits *from unlabelled test data*
    (SCORE's orthogonal Procrustes recovery, 26.22 -> 53.23 Top-1 with no target
    labels). Training a network to do a job that a closed-form, label-free fit does
    better and cannot overfit to the nine source subjects is strictly worse.

STAGE C (DEPLOYMENT CALIBRATION) IS NOT IN THIS FILE
----------------------------------------------------
SAW whitening, adaptive CSLS and coordinate recovery are *test-time* operations applied
to frozen embeddings -- they have no parameters to train and no loss. They live in
``samclip.calibration`` and are driven by ``scripts/run_eval.py``. Keeping them out of
the training file is deliberate: they are the largest single lever on this problem, and
a reader looking for "where does the 53.23 come from" should not find it tangled into
the optimiser loop.

THE OBJECTIVE -- TWO OF THEM, SELECTED BY ``objective``
------------------------------------------------------
``objective: v3`` is the previous eight-term objective, kept verbatim so every recorded
result stays reproducible::

    L = w_img  * L_clip   + w_xs * L_xs    + w_mmd * L_mmd(subject)
      + w_proto* L_proto  + w_reg * (L_var + L_cov) + w_dec * L_hsic
      + w_rkd  * L_rkd    + w_adv * L_adv

``objective: v4`` is the redesigned four-term objective (see
``docs/eeg2image_v4_architecture.md`` §3.7)::

    L = w_mmd * L_mmd(EEG cloud <-> image cloud)
      + w_fine * (w_img * L_clip + w_xs * L_xs)
      + w_spec * L_spec   + w_reg * L_cov

``w_fine`` and ``w_mmd`` come from the coarse-to-fine schedule and are the reference
implementation's two-term ramp (``w_mmd: 0.9 -> 0.5`` on Stage 1, contrast only on
Stage 2 at ``w_fine = 1.0``). The four REMOVED terms each have a measurement behind
them, not a preference -- see the ``objective == "v4"`` branches in ``Trainer.assemble``
and the loss-package docstring.

The v4 additions beyond topology (``models/smn.py``, ``share_head``) are:

  * ``L_mmd`` becomes CROSS-MODAL. The name "coarse-to-fine" in the v3 config referred
    to ``mmd_subject``, which matches subjects against each other -- a different
    quantity from the reference's ``MMD(z_eeg cloud, z_image cloud)``, and one that can
    be satisfied inside a subject-specific subspace the image cloud never occupies.
  * ``L_spec`` replaces VICReg's variance hinge. The hinge pushes all ``d`` axes outward
    with equal force while the task's variance lives in ~16 directions: it cannot be
    satisfied (0/512 dims reached std 1.0 on a trained v3 checkpoint) yet it held 18.8%
    of the gradient reaching the encoder.

``w_xs`` survives into v4 because it is the one alignment-preserving term with a
positive measured effect: an independent LOSO study reports +2.23pp Top-1 (95% CI
[+1.28, +3.19]) from pulling same-stimulus cross-subject representations together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config, evaluate
from .data import things_eeg, targets as target_mod
from .data.augment import build_augment
from .data.sampler import CrossSubjectBatchSampler
from .losses import (
    InfoNCE,
    SubjectAdversary,
    clip_alignment_loss,
    cross_subject_loss,
    csls_correct,
    gram_distill_loss,
    hsic_subject,
    mmd_crossmodal,
    mmd_subject,
    metric_self_distill,
    metric_subject_consistency,
    metric_anchor_loss_by_subject,
    subject_consensus_anchor_loss,
    gallery_metric,
    recovery_aware_alignment,
    repetition_collapse_loss,
    score_fusion_loss,
    soft_plan_loss,
    spectral_concentration,
    spectral_rank_target,
    spectrum_report,
    vicreg_terms,
)
from .losses.regularizers import PrototypeEMA
from .models import build_model
from .utils import (AverageMeter, count_parameters, get_logger, human_int,
                    prune_epoch_checkpoints, to_device)

LOG = get_logger("samclip.train")

#: Objectives `assemble` knows how to interpret. `v6` is v4's shared-head topology with
#: a MULTI-ROUTE target stack and score-level fusion; it reuses every v4 sub-term and adds
#: only `fuse`. See `docs/eeg2image_v6_architecture.md`.
OBJECTIVES = ("v3", "v4", "v6")


# ------------------------------------------------------------------ criteria
@dataclass
class LossWeights:
    img: float = 1.0
    cross: float = 0.5        # cross-subject same-stimulus InfoNCE (the "xs" term)
    dec: float = 0.05         # HSIC subject decorrelation -- light, see module docstring
    mmd: float = 0.1          # coarse distribution matching (kind depends on `objective`)
    reg: float = 0.5          # VICReg (var+cov on v3; cov only on v4)
    proto: float = 0.1        # EMA neural class anchors
    rkd: float = 0.0          # Gram-matrix geometry matching (off by default)
    adv: float = 0.0          # gradient-reversal adversary (off by default)
    spec: float = 0.0         # spectral concentration (v4 only; 0 keeps v3 bit-identical)
    router: float = 0.0       # negative-entropy bonus on the layer blend (see below)
                              # Off by default so an existing run is bit-identical; the
                              # v3 recipe turns it on. It exists because the learned
                              # blend was measured collapsing onto the shallowest layer
                              # ([0.74, 0.13, 0.04, 0.01, 0.07]), which spends the
                              # five-layer target down to one layer and removes the
                              # multi-granularity the target stack is there to provide.
                              # `mean` avoids that only by learning nothing; the entropy
                              # bonus keeps a learned reweighting while refusing a
                              # one-hot blend.

    @classmethod
    def from_cfg(cls, cfg: dict) -> "LossWeights":
        w = cfg.get("loss_weights", {}) or {}
        known = set(cls().__dict__)
        unknown = set(w) - known
        if unknown:
            # A typo'd weight is silently ignored by a plain `get` loop, which turns
            # "this term is part of the objective" into a run that never applied it --
            # the exact failure mode that let a dead conditioning arm report a healthy
            # loss for a whole pipeline run.
            raise ValueError(
                f"unknown loss_weights key(s) {sorted(unknown)}; known: {sorted(known)}")
        return cls(**{k: float(w.get(k, getattr(cls(), k))) for k in known})

    def validate(self, objective: str) -> None:
        """Reject weights that would be silently ignored by the selected objective.

        Two silent-failure modes are possible and both have happened on this project in
        spirit: a term in the YAML that the objective never reads (a run that reports
        occupying a mechanism it does not use), and a term whose weight is set on the
        WRONG objective's key so the intended term stays at zero. Raising here is cheap;
        discovering it after 50 epochs on a GPU node is not.
        """
        if objective == "v3":
            if self.spec > 0:
                raise ValueError(
                    "loss_weights.spec > 0 with objective='v3': L_spec is a v4 term. "
                    "Use `objective: v4` (or `arch: v4`) for the redesigned objective, "
                    "or zero `spec` -- the v3 path never reads it.")
        elif objective == "v4":
            removed = (("dec", self.dec), ("proto", self.proto),
                       ("rkd", self.rkd), ("adv", self.adv), ("reg", self.reg))
            dead = [k for k, v in removed if v > 0]
            if dead:
                raise ValueError(
                    f"objective='v4' does not read {sorted(dead)} (the terms were removed "
                    f"for measured reasons -- see docs/eeg2image_v4_architecture.md "
                    f"§3.5 and §3.7 and the `reg` comment in `Trainer.assemble`: on v4 the "
                    f"covariance penalty measures 4.6e-09 and moves by only 14% between a "
                    f"random representation and one with an axis exactly duplicated, so a "
                    f"weight on it is decoration). Set them to 0.0.")
            if self.img <= 0 or self.cross <= 0:
                raise ValueError(
                    "objective='v4' scales `img` and `cross` by the schedule's fine-phase "
                    "weight, so both must be > 0; the coarse term alone is a distribution "
                    "match and carries no concept-level gradient.")
        elif objective == "v6":
            # v6 is v4's topology with a multi-route target stack, so it inherits v4's
            # removals verbatim; the fusion weight is NOT a `loss_weights` key (it lives in
            # `fusion.weight`, parsed in `Trainer.__post_init__`) precisely so that adding
            # it cannot silently change what a recorded v3/v4 config means.
            removed = (("dec", self.dec), ("proto", self.proto),
                       ("rkd", self.rkd), ("adv", self.adv), ("reg", self.reg))
            dead = [k for k, v in removed if v > 0]
            if dead:
                raise ValueError(
                    f"objective='v6' does not read {sorted(dead)} (inherit from v4 -- see "
                    f"the `objective='v4'` message in `LossWeights.validate`). Set 0.0.")
            if self.img <= 0 or self.cross <= 0:
                raise ValueError(
                    "objective='v6' scales `img` and `cross` by the schedule's fine-phase "
                    "weight (it uses v4's phase/ratio split), so both must be > 0.")
        else:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")


@dataclass
class CoarseToFine:
    """Two-phase loss schedule: coarse distribution alignment first, fine contrast second.

    This is SAMGA's recipe, transcribed from its released code rather than inferred,
    because our own v2/v3 pipeline runs the *fine* half from step one and pins the
    coarse half at a weight five times smaller (`mmd: 0.2` constant, single 60-epoch
    stage):

        stage1_epochs          20
        stage1_mmd_start       0.9      ->  MMD dominates early
        stage2_learning_rate   5e-05

    and its log shows exactly that ramp::

        Epoch [1/50]  | stage1 mmd=0.900 contrast=0.100
        Epoch [11/50] | stage1 mmd=0.689 contrast=0.311
        Epoch [21/50] | stage2 contrastive only

    The interpolation is `t = epoch / (stage1_epochs - 1)` on a 0-based epoch, which
    reproduces SAMGA's printed values to three decimals (their epoch 11 -> mmd 0.689,
    contrast 0.311). The direction matters: the coarse term is a *distribution-matching*
    term, so it is what makes the two clouds occupy the same region, and the contrastive
    term is what then sharpens concept structure inside the aligned frame. Doing the fine
    term first spends the encoder's capacity before the frames agree.

    THE COEFFICIENTS ARE (mmd, cross), AND THEIR INTERPRETATION IS THE OBJECTIVE'S
    ----------------------------------------------------------------------------
    The schedule is deliberately OBJECTIVE-AGNOSTIC: it emits two coefficients per epoch
    and each objective spends them differently, so a schedule cannot silently mean one
    thing in the log and another in the loss.

      * ``objective: v4`` (the reference's own form) -- the two coefficients are a convex
        split: ``mmd = 0.9 -> 0.5`` and ``cross = 1 - mmd``, i.e. ``cross_start 0.1``,
        ``cross_end 0.5``. Stage 2 is contrast-only, so ``stage2_mmd 0.0`` **and**
        ``stage2_cross 1.0``.
      * ``objective: v3`` -- the coefficients are independent absolute weights, the v3
        config's `mmd 0.9 -> 0.2 / cross 0.1 -> 0.7` ramp, and Stage 2 keeps a small
        residual MMD (``stage2_mmd 0.05``) with ``cross_end``.

    `stage2_cross` exists precisely so that v4's "contrast at full weight" is expressible
    without a branch in `at()`: it defaults to `None`, meaning "reuse `cross_end`", which
    is the v3 behaviour.

    WHERE THE ENDPOINTS COME FROM. `inter.sh` (the inter-subject protocol) passes
    `--stage1_mmd_start 0.9 --stage1_mmd_end 0.5`; `train.py`'s argparse default for the
    end is 0.2. We follow `inter.sh`, because the intra-subject default is the wrong
    protocol for this task. The reference's *absolute* magnitudes still cannot be
    transplanted uncritically -- its Stage 1 has two terms while v4 has four, so the same
    coefficient is not the same fraction of the gradient -- but for v4 the split is at
    least mathematically the reference's (the two coefficients sum to 1).

    STAGE 2 FREEZES THE ENCODER. Under v3, Stage 2 continued joint training and the log
    shows the failure: a dip-then-recover curve (Top-1 21.0 -> 17.5 -> 20.5 across the
    phase), i.e. the phase boundary cost a third of the epoch budget's progress and
    returned to where it started. The reference instead freezes the encoder and trains
    only the projectors at `stage2_learning_rate`; on a representation that is already in
    the deployed metric, joint training can only push it back out. `freeze_encoder_stage2`
    defaults to False so v3 runs are unchanged, and the v4 config turns it on.
    """

    enabled: bool = False
    stage1_epochs: int = 20
    mmd_start: float = 0.9
    mmd_end: float = 0.2
    stage2_mmd: float = 0.05
    cross_start: float = 0.1
    cross_end: float = 0.7
    stage2_cross: float | None = None   # None -> reuse `cross_end` (the v3 behaviour)
    stage2_lr: float = 5e-5
    freeze_encoder_stage2: bool = False
    lr: float = 1e-3

    @classmethod
    def from_cfg(cls, cfg: dict) -> "CoarseToFine":
        s = cfg.get("schedule", {}) or {}
        known = {"coarse_to_fine", "stage1_epochs", "mmd_start", "mmd_end", "stage2_mmd",
                 "cross_start", "cross_end", "stage2_cross", "stage2_lr",
                 "freeze_encoder_stage2"}
        unknown = set(s) - known
        if unknown:
            # Same reasoning as `LossWeights.from_cfg`: a typo'd schedule key would
            # silently leave the phase running at its default, which is a schedule the
            # run did not intend and the log would not reveal.
            raise ValueError(f"unknown schedule key(s) {sorted(unknown)}; "
                             f"known: {sorted(known)}")
        epochs = int(cfg.get("epochs", 50))
        s1 = int(s.get("stage1_epochs", 20))
        enabled = bool(s.get("coarse_to_fine", False))
        # Only validated when the schedule is ON: with it off, `stage1_epochs` is never
        # read, and rejecting a short smoke run (`--epochs 1`) over a value that no code
        # path consumes would make the cheap DAG test unusable.
        if enabled and s1 >= epochs:
            raise ValueError(
                f"schedule.stage1_epochs={s1} >= epochs={epochs}: there is no Stage 2, "
                f"so `coarse_to_fine` would be a mislabelled single-stage run")
        if enabled and s1 < 1:
            raise ValueError(f"schedule.stage1_epochs must be >= 1, got {s1}")
        stage2_cross = s.get("stage2_cross")
        return cls(
            enabled=enabled,
            stage1_epochs=s1,
            mmd_start=float(s.get("mmd_start", 0.9)),
            mmd_end=float(s.get("mmd_end", 0.2)),
            stage2_mmd=float(s.get("stage2_mmd", 0.05)),
            cross_start=float(s.get("cross_start", 0.1)),
            cross_end=float(s.get("cross_end", 0.7)),
            stage2_cross=None if stage2_cross is None else float(stage2_cross),
            stage2_lr=float(s.get("stage2_lr", 5e-5)),
            freeze_encoder_stage2=bool(s.get("freeze_encoder_stage2", False)),
            lr=float(cfg.get("lr", 1e-3)),
        )

    def at(self, epoch: int) -> tuple[float, float, float, str]:
        """``(coarse_weight, fine_weight, lr, phase)`` for a 0-based epoch index."""
        if not self.enabled:
            raise RuntimeError("CoarseToFine.at() called with the schedule disabled")
        if epoch < self.stage1_epochs:
            t = epoch / max(1, self.stage1_epochs - 1)
            return (self.mmd_start + (self.mmd_end - self.mmd_start) * t,
                    self.cross_start + (self.cross_end - self.cross_start) * t,
                    self.lr, "stage1")
        return (self.stage2_mmd,
                self.cross_end if self.stage2_cross is None else self.stage2_cross,
                self.stage2_lr, "stage2")

    def describe(self) -> str:
        if not self.enabled:
            return "single-stage (fixed loss weights, cosine lr)"
        s2_x = self.cross_end if self.stage2_cross is None else self.stage2_cross
        return (f"coarse-to-fine | stage1 {self.stage1_epochs} epochs "
                f"coarse {self.mmd_start}->{self.mmd_end}, fine {self.cross_start}->"
                f"{self.cross_end} @ lr {self.lr:g} | stage2 coarse={self.stage2_mmd:g}, "
                f"fine={s2_x:g} @ lr {self.stage2_lr:g}"
                f"{' (encoder frozen)' if self.freeze_encoder_stage2 else ''}")


@dataclass
class Trainer:
    model: nn.Module
    cfg: dict
    device: torch.device
    n_subjects: int
    weights: LossWeights = field(default_factory=LossWeights)
    prototype: PrototypeEMA | None = None
    adversary: SubjectAdversary | None = None

    def __post_init__(self) -> None:
        self.crit_img = InfoNCE(init_temp=self.cfg.get("temp_img", 0.07),
                                softplus=self.cfg.get("softplus", True)).to(self.device)
        self.crit_cross = InfoNCE(init_temp=self.cfg.get("temp_cross", 0.1),
                                  softplus=self.cfg.get("softplus", True)).to(self.device)
        # Which objective the weights are interpreted by, and therefore which terms are
        # live. Resolved once, here, so `assemble` has no way to read a config that
        # changed under it mid-run, and so the checkpoint records the same value the
        # optimiser ran with.
        self.objective = str(self.cfg.get("objective", "v3"))
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, "
                             f"got {self.objective!r}")
        self.weights.validate(self.objective)
        # `r0` for the spectral term. 16 is the MEASURED dimension of the concept
        # manifold (93-96% of query, 98% of target variance in the top 16 PCs), not a
        # tuned constant -- see `probe_subspace_alignment.py`.
        self.spec_r0 = int(self.cfg.get("spec_r0", 16))
        # WHICH SPECTRAL TERM `spec > 0` GETS. Default stays `"tail"` (the historical
        # `spectral_concentration`) so no recorded run changes meaning, but `"tail"` is
        # now documented as pathological: minimising the tail fraction is maximising the
        # head fraction, whose optimum is rank-1 collapse, and it measures ~1e-07 at both
        # rank 1 and rank 16. It was also measured INERT at the shipped weight (removing
        # it moved Top-1 by -0.00pp, p = 1.00). `"rank"` selects the corrected bilateral
        # target, which has an interior optimum and cannot be satisfied by collapsing.
        self.spec_mode = str(self.cfg.get("spec_mode", "tail"))
        if self.spec_mode not in ("tail", "rank"):
            raise ValueError(
                f"spec_mode must be 'tail' or 'rank', got {self.spec_mode!r}. 'tail' is "
                f"the historical form and is pathological (see "
                f"losses.regularizers.spectral_concentration); 'rank' is the bilateral "
                f"replacement.")

        # RECOVERY-AWARE TRAINING (v5 pillar A). Parsed once here, read by `assemble`.
        # `enabled: false` (the default) leaves the objective exactly as recorded runs
        # had it, so turning this on cannot silently change a reproduced result.
        #
        # `block_per_subject` and `csls_k` are separate knobs on purpose: turning them on
        # together would make the first run unattributable, and attribution is the whole
        # reason the earlier v4 regression took a 2x2 to localise.
        ra = self.cfg.get("recovery_aware") or {}
        if not isinstance(ra, dict):
            raise ValueError(f"`recovery_aware` must be a mapping, got {type(ra).__name__}")
        self.ra_enabled = bool(ra.get("enabled", False))
        self.ra_block = bool(ra.get("block_per_subject", True))
        # `null` csls_k means "block only, no hubness correction" -- a legitimate arm. Any
        # other value must be a positive int, because k <= 0 makes `topk` meaningless and
        # would silently degrade to k = 1 via the clamps in `csls_correct`.
        raw_k = ra.get("csls_k", 10)
        if raw_k is not None and int(raw_k) < 1:
            raise ValueError(f"recovery_aware.csls_k must be >= 1 or null, got {raw_k!r}")
        terms = ra.get("terms", ["img", "cross"])
        unknown_terms = set(terms) - {"img", "cross"}
        if unknown_terms:
            raise ValueError(
                f"recovery_aware.terms may only name {'img', 'cross'} (the two fine "
                f"contrasts; `mmd` is a distribution match with no logits to correct), "
                f"got {sorted(unknown_terms)}")
        # THE EFFECTIVE VALUES ARE GATED ON `enabled`, and that gating is load-bearing.
        # An earlier version parsed `csls_k` unconditionally, so `enabled: false` still
        # applied the CSLS correction in `assemble` -- i.e. every run would have been
        # "recovery-aware" while its config, its log and its comparison arm all said
        # "raw cosine". A flag whose off position is not inert is worse than no flag,
        # because the baseline it is compared against is silently the treatment. The
        # fields below are therefore left at their inert values for the whole disabled
        # path, and `assemble` reads only these (never the raw config).
        if self.ra_enabled:
            self.ra_csls_k = None if raw_k is None else int(raw_k)
            self.ra_terms = tuple(dict.fromkeys(terms))   # de-duped, order-stable
        else:
            self.ra_csls_k = None
            self.ra_terms = ()

        # ---- SCORE's source-only episode (the second half of recovery-aware training) ---
        # `docs/eeg2image_v5_master_plan.md` §10.2, §11.1. Deploy-time coordinate recovery
        # lifted a frozen v5 checkpoint from 27.00 to 35.50 Top-1 with NO retraining, which
        # is the largest single gain measured in this project. But it was applied to an
        # encoder that was never told recovery would happen. SCORE trains the other half:
        # one source subject per step is treated as a temporary target, its EEG-image
        # pairing is hidden, THE DEPLOYMENT RECOVERY IS APPLIED, and only then are the
        # matches revealed to compute the loss. The encoder is then optimised in a frame
        # it can actually recover.
        #
        # Same gating discipline as above: the effective values are inert when disabled,
        # and `assemble` reads only these fields. An earlier version of the CSLS flag was
        # parsed unconditionally and silently turned every "raw" baseline into the
        # treatment; this is the same trap and it is closed the same way.
        self.ep_enabled = bool(ra.get("recovery_episode", False))
        self._ep_fires = 0
        self._ep_abstains = 0
        self._ep_last_diag: dict = {}
        if self.ep_enabled:
            self.ep_rho = float(ra.get("episode_rho", 0.1))
            # `episode_csls_k` defaults to the loss's CSLS k, but it is separable because
            # the whole point of v6's faithful episode is that the EPISODE's k must equal
            # DEPLOYMENT's (10), which is not necessarily the k the contrastive logits are
            # trained at. `v5-ep2` conflated the two and lost 2.0pp on deployment.
            ep_k_raw = ra.get("episode_csls_k")
            self.ep_k = (self.ra_csls_k if self.ra_csls_k else 10) \
                if ep_k_raw is None else int(ep_k_raw)
            self.ep_min_landmarks = int(ra.get("episode_min_landmarks", 8))
            # v6 PILLAR A -- "faithful RAT": run the WHOLE deployment ladder
            # (`whiten -> CSLS(k) -> mutual-NN -> Procrustes -> apply`) inside the episode,
            # not just its recovery tail. `full: false` reproduces `v5-ep2` bit-for-bit so
            # the two arms differ by exactly this one switch.
            self.ep_full = bool(ra.get("in_loop_full", False))
            self.ep_whiten = bool(ra.get("episode_whiten", True))
            self.ep_shrink = float(ra.get("episode_shrink", 0.1))
            self.ep_max_cond = float(ra.get("episode_max_cond", 1e3))
            if self.ep_rho < 0:
                raise ValueError(f"recovery_aware.episode_rho must be >= 0, "
                                 f"got {self.ep_rho}")
            if self.ep_k < 1:
                raise ValueError(f"recovery_aware.episode_csls_k must be >= 1, "
                                 f"got {self.ep_k}")
        else:
            self.ep_rho = 0.0
            self.ep_k = 10
            self.ep_min_landmarks = 8
            self.ep_full = False
            self.ep_whiten = True
            self.ep_shrink = 0.1
            self.ep_max_cond = 1e3
        self._episode_step = 0

        # ---- v7 T2': cross-trial repetition collapse --------------------------------
        # The ONLY training-time term that is not computable from the inputs every other
        # recipe trains on. `cross_subject_loss` already supplies the cross-SUBJECT half of
        # the concept invariant; this supplies the cross-TRIAL half, which the standard
        # averaging destroys (see `losses.contrastive.repetition_collapse_loss`).
        #
        # Same gating discipline as `recovery_aware`: the effective values are inert unless
        # `enabled`, so `enabled: false` is bit-identical to the G3 recipe rather than a
        # treatment whose flag reads "off".
        concept = self.cfg.get("concept") or {}
        if not isinstance(concept, dict):
            raise ValueError(f"`concept` must be a mapping, got {type(concept).__name__}")
        self.concept_enabled = bool(concept.get("enabled", False))
        if self.concept_enabled:
            self.reps_weight = float(concept.get("reps_weight", 0.0))
            # `n_rows` x `r_use` is the number of EXTRA encoder passes the term costs
            # relative to a batch of `batch_size`. It is explicit rather than implied by
            # the batch size because that is the whole compute/benefit trade, and hiding it
            # would make a 4x-slower run look like a free term.
            self.reps_n_rows = int(concept.get("reps_n_rows", 64))
            self.reps_r_use = int(concept.get("reps_r_use", 2))
            reps_k = concept.get("reps_csls_k")
            self.reps_csls_k = None if reps_k is None else int(reps_k)
            # `row` reproduces the first (failing) implementation; `stimulus` is the fix.
            # See `losses.contrastive.repetition_collapse_loss` for why row-grouping is an
            # anti-T1 term on a cross-subject batch.
            self.reps_group = str(concept.get("reps_group", "row"))
            if self.reps_group not in ("row", "stimulus"):
                raise ValueError(
                    f"concept.reps_group must be 'row' or 'stimulus', got {self.reps_group!r}")
            # `n_stim` x `subj_sel` x `r_use` points, in `n_stim` stimulus groups. Balanced
            # on purpose: taking a contiguous PREFIX of the batch instead would draw all its
            # rows from the first few stimuli (the sampler is stimulus-major), which makes
            # the contrast a handful of classes and leaves the rest of the batch unused as
            # negatives. `subj_sel` keeps the group from being the whole 9-subject tile.
            self.reps_n_stim = int(concept.get("reps_n_stim", 32))
            self.reps_subj_sel = int(concept.get("reps_subj_sel", 2))
            # The gate is excluded from this term's gradient by default when the term is on:
            # the term and the SMN's scale correction are substitutes, and the measured
            # failure was the term winning that competition and taking the gate to 0.
            self.reps_detach_gate = bool(concept.get("reps_detach_gate", True))
            #: Cached row positions for the reps term, keyed by batch shape. The sampler
            #: emits rows stimulus-major with an EQUAL group size, so which POSITIONS to
            #: take is fixed across steps even though the stimulus IDs at those positions
            #: change every step. Recomputing it per step would be a 1152-element Python
            #: scan 6450 times per run for a value that cannot change; `_reps_rows` verifies
            #: the layout on the first batch and then reuses the positions.
            self._reps_pos_cache: dict[tuple[int, int, int], torch.Tensor] = {}
            if self.reps_weight < 0:
                raise ValueError(f"concept.reps_weight must be >= 0, got {self.reps_weight}")
            if self.reps_r_use < 2:
                raise ValueError(
                    f"concept.reps_r_use must be >= 2 (a single repetition has no positive, "
                    f"so the term would be a no-op reported as a mechanism), "
                    f"got {self.reps_r_use}")
            if self.reps_n_rows < 2:
                raise ValueError(
                    f"concept.reps_n_rows must be >= 2 (the contrast needs negatives), "
                    f"got {self.reps_n_rows}")
        else:
            self.reps_weight = 0.0
            self.reps_n_rows = 0
            self.reps_r_use = 0
            self.reps_csls_k = None
            self.reps_group = "row"
            self.reps_n_stim = 0
            self.reps_subj_sel = 0
            self.reps_detach_gate = False

        # ------------------------------------------------ v8: soft-plan alignment
        # The FIRST training term aligned with the operator that is actually deployed. Four
        # earlier pillars (T2', T2'', v7b's SCORE episode, G-a's low-rank frame) all tried to
        # raise the HARD mutual-NN landmark rate, a quantity the operator does not consume --
        # which is why its +3.62 +- 1.72 contribution measured FLAT (corr with raw = -0.19,
        # with landmark rate = -0.21, 30 runs). Swapping the ESTIMATOR for a Sinkhorn soft plan
        # moved the headline +2.80pp with 10/10 folds improving (job 643819), which is what
        # makes the plan a quantity training can shape. See `losses/soft_plan.py`.
        #
        # Same gating discipline as every other block: `enabled: false` leaves the effective
        # weight at 0 and no branch is taken, so an ablated arm is bit-identical to G3 rather
        # than a treatment whose flag merely reads "off".
        sp = self.cfg.get("soft_plan") or {}
        if not isinstance(sp, dict):
            raise ValueError(f"`soft_plan` must be a mapping, got {type(sp).__name__}")
        if bool(sp.get("enabled", False)):
            self.sp_weight = float(sp.get("weight", 0.5))
            # NOT the deployment tau. Deployment wants the sharpest plan (0.01-0.05, gradients
            # be damned); at training tau that small the plan collapses to a near-permutation
            # and the Jacobian through the Sinkhorn loop vanishes, so the term would report as
            # active while contributing nothing. The measured |grad| at 0.05/0.1/0.2 is
            # 2.2e1/1.6e1/8.0e0, so 0.1 is chosen for flow.
            self.sp_tau = float(sp.get("tau", 0.1))
            self.sp_iters = int(sp.get("iters", 20))
            sp_k = sp.get("csls_k", 20)
            self.sp_csls_k = None if sp_k is None else int(sp_k)
            # TRUE is not a tuning knob: a doubly-stochastic plan's column marginal is only
            # the right prior when the block is a 1-to-1 retrieval set, and deployment always
            # scores one held-out subject. Measured on a 2x8 toy block, an UNBLOCKED plan puts
            # 54% of its mass cross-subject. Note this differs from
            # `recovery_aware.block_per_subject: false` in G3, which was measured inert: there
            # the block was a density-estimation window, here it is a structural requirement.
            self.sp_block = bool(sp.get("block_per_subject", True))
            # v9: the plan becomes a FUSED Gromov-Wasserstein coupling. 0.0 keeps the plain
            # Sinkhorn-on-CSLS plan, which is bit-identical to the v8 term.
            self.sp_alpha = float(sp.get("alpha", 0.0))
            self.sp_fgw_outer = int(sp.get("fgw_outer", 10))
            if self.sp_alpha < 0.0 or self.sp_alpha > 1.0:
                raise ValueError(f"soft_plan.alpha must be in [0, 1], got {self.sp_alpha}")
            if self.sp_weight < 0:
                raise ValueError(f"soft_plan.weight must be >= 0, got {self.sp_weight}")
            if self.sp_tau <= 0:
                raise ValueError(f"soft_plan.tau must be > 0, got {self.sp_tau}")
            if self.sp_iters < 1:
                raise ValueError(f"soft_plan.iters must be >= 1, got {self.sp_iters}")
            if self.sp_csls_k is not None and self.sp_csls_k < 1:
                raise ValueError(
                    f"soft_plan.csls_k must be >= 1 or null, got {self.sp_csls_k}")
        else:
            self.sp_weight = 0.0
            self.sp_tau = 0.1
            self.sp_iters = 20
            self.sp_csls_k = None
            self.sp_block = True
            self.sp_alpha = 0.0
            self.sp_fgw_outer = 10

        # ------------------------------------------------------- v12: metric self-distillation
        # The claim (docs/eeg2image_v12_core_claim.md) commits us to moving the lever measured in
        # Part C into TRAINING: the query metric is still estimation-limited at the deployed R
        # (Top-1 4.0 -> 54.5 from R=1 to R=80 and still climbing +3.33pp, job 645719), so a term
        # that makes a FEW repetitions carry what the whole cloud carries is the one lever with
        # headroom. The teacher is the v11 reliability-fused consensus metric (the estimator
        # `probe_fusion.py` measured to carry shared structure with a permutation control at zero,
        # job 645697); the student is a single-repetition metric.
        #
        # Same gating discipline as every other block: with `enabled: false` (or both weights 0)
        # no branch is taken and no encoder pass is spent, so an ablated arm is bit-identical to
        # the recipe without it rather than a treatment whose flag merely reads "off".
        v12 = self.cfg.get("v12") or {}
        if not isinstance(v12, dict):
            raise ValueError(f"`v12` must be a mapping, got {type(v12).__name__}")
        self.v12_enabled = bool(v12.get("enabled", False))
        if self.v12_enabled:
            self.v12_distill = float(v12.get("metric_distill", 0.0))
            self.v12_consistency = float(v12.get("metric_consistency", 0.0))
            # Repetitions the v12 terms consume per selected row. Separate from
            # `concept.reps_r_use` because the teacher needs several INDEPENDENT blocks: with the
            # concept term's default of 2 there is no teacher to distil from, and a term that
            # silently reduces to "compare a metric to itself" is the failure this project keeps
            # paying for.
            self.v12_r_use = int(v12.get("r_use", 8))
            self.v12_student_reps = int(v12.get("student_reps", 1))
            self.v12_teacher_blocks = int(v12.get("teacher_blocks", 4))
            self.v12_n_stim = int(v12.get("n_stim", 32))
            self.v12_subj_sel = int(v12.get("subj_sel", 2))
            # The gate is excluded by default for the same measured reason as the concept term:
            # the v12 terms and the SMN's scale correction are substitutes, and the concept term
            # already won that competition once (gate 0.498 -> 0.000) before the gate was detached.
            self.v12_detach_gate = bool(v12.get("detach_gate", True))
            if self.v12_distill < 0 or self.v12_consistency < 0:
                raise ValueError("v12 weights must be >= 0")
            if self.v12_r_use < 3:
                raise ValueError(
                    f"v12.r_use must be >= 3 (student 1 rep + a teacher from >= 2 disjoint reps), "
                    f"got {self.v12_r_use}")
            if self.v12_student_reps < 1 or self.v12_student_reps > self.v12_r_use - 2:
                raise ValueError(
                    f"v12.student_reps must be in [1, r_use-2] so the teacher has >= 2 disjoint "
                    f"reps, got student_reps={self.v12_student_reps}, r_use={self.v12_r_use}")
        else:
            self.v12_distill = 0.0
            self.v12_consistency = 0.0
            self.v12_r_use = 0
            self.v12_student_reps = 1
            self.v12_teacher_blocks = 4
            self.v12_n_stim = 0
            self.v12_subj_sel = 0
            self.v12_detach_gate = False

        # ------------------------------------------------ v13: order-2 BIAS ANCHOR
        # The claim (docs/eeg2image_v13_croma.md) is that the deployed object's error has exactly
        # three components -- bias, variance, rank -- and that a measurement (job 650135) leaves
        # exactly ONE with headroom:
        #   * variance is exhausted: repetition-error rho <= 0 on 10/10 folds, K_eff = R = 80, so
        #     pooling already removes every independent component. There is deliberately NO
        #     variance term here; adding one would be decoration whose loss curve cannot fall.
        #   * bias is live AND subject-specific: a 9-subject pooled EEG metric (9 independent
        #     encodings, ~720 repetitions) agrees with the gallery at 0.796 while the target's own
        #     80 repetitions reach 0.616 (+0.180, t=+22.2, 10/10 folds, shuffle control -0.0005).
        #     Pooling cannot remove a subject's own deviation; averaging over encodings can.
        # The anchor is the GALLERY metric D_g, which is exogenous (not a function of the encoder)
        # and full-rank, which is why this term does not open v12's collapse channel: a shrinkable
        # self-consistent target admitted a rank-2 solution; a fixed full-rank target cannot.
        #
        # Same gating discipline as every other block: `enabled: false` (or weight 0) takes no
        # branch and spends no encoder pass, so the twin baseline is bit-identical rather than a
        # treatment whose flag reads "off".
        b13 = self.cfg.get("bias_anchor") or {}
        if not isinstance(b13, dict):
            raise ValueError(f"`bias_anchor` must be a mapping, got {type(b13).__name__}")
        self.bias_anchor_enabled = bool(b13.get("enabled", False))
        if self.bias_anchor_enabled:
            self.ba_weight = float(b13.get("weight", 0.0))
            self.ba_r_use = int(b13.get("r_use", 8))
            self.ba_n_stim = int(b13.get("n_stim", 32))
            self.ba_subj_sel = int(b13.get("subj_sel", 3))
            # `None` -> the student is the POOLED metric (the deployed readout); an int restricts it
            # to a prefix of repetitions, which is the single-view arm of the discrimination.
            _sr = b13.get("student_reps", None)
            self.ba_student_reps = None if _sr is None else int(_sr)
            self.ba_min_concepts = int(b13.get("min_concepts", 4))
            # ---- v14 FRAME FIX. `raw` is v13's (failed) frame; `whitened` scores the metric in
            # the frame the deployed operator reads, which makes the loss invariant to per-subject
            # invertible linear maps -- the anisotropy class v13's gradient was spent on (raw-frame
            # gain +0.025..+0.041, whitened +0.001..+0.011, Top-1 -9..-13pp). Default `raw` keeps
            # every v13 number on disk interpretable.
            self.ba_frame = str(b13.get("frame", "raw"))
            if self.ba_frame not in ("raw", "whitened"):
                raise ValueError(f"bias_anchor.frame must be 'raw' or 'whitened', got {self.ba_frame}")
            self.ba_shrink = float(b13.get("whiten_shrink", 0.1))
            # Repetitions must come from the batch, and `batch["reps"]` only exists when
            # `concept.enabled` is true (see `build_loader`). Failing loudly beats a term that
            # silently never fires -- this project has paid for that exact failure mode.
            if not self.concept_enabled:
                raise ValueError(
                    "bias_anchor is enabled but `concept.enabled` is false, so the loader never "
                    "passes per-repetition EEG and the term could not fire. Set "
                    "`concept: {enabled: true, reps_weight: 0.0}`.")
            if self.ba_weight < 0:
                raise ValueError(f"bias_anchor.weight must be >= 0, got {self.ba_weight}")
            if self.ba_r_use < 1:
                raise ValueError(f"bias_anchor.r_use must be >= 1, got {self.ba_r_use}")
            if self.ba_student_reps is not None and self.ba_student_reps < 1:
                raise ValueError(
                    f"bias_anchor.student_reps must be >= 1 or null, got {self.ba_student_reps}")
        else:
            self.ba_weight = 0.0
            self.ba_r_use = 0
            self.ba_n_stim = 0
            self.ba_subj_sel = 0
            self.ba_student_reps = None
            self.ba_min_concepts = 4

        # ---- v14 C-ANCHOR: a CONSENSUS anchor (gallery + the other subjects in the batch) rather
        # than a single gallery view. Parsed OUTSIDE the bias_anchor block on purpose: it is an
        # independent term with its own switch, and nesting it would have made `cs_*` undefined
        # whenever `bias_anchor.enabled` was false -- an AttributeError at the first training step,
        # i.e. the failure mode that only shows up after the queue has started burning GPUs.
        cs = self.cfg.get("cs_anchor") or {}
        if not isinstance(cs, dict):
            raise ValueError(f"`cs_anchor` must be a mapping, got {type(cs).__name__}")
        self.cs_enabled = bool(cs.get("enabled", False))
        self.cs_weight = 0.0
        self.cs_frame = "whitened"
        self.cs_shrink = 0.1
        self.cs_include_gallery = True
        self.cs_min_concepts = 4
        if self.cs_enabled:
            self.cs_weight = float(cs.get("weight", 0.0))
            self.cs_frame = str(cs.get("frame", "whitened"))
            if self.cs_frame not in ("raw", "whitened"):
                raise ValueError(f"cs_anchor.frame must be 'raw' or 'whitened', got {self.cs_frame}")
            self.cs_shrink = float(cs.get("whiten_shrink", 0.1))
            self.cs_include_gallery = bool(cs.get("include_gallery", True))
            self.cs_min_concepts = int(cs.get("min_concepts", 4))
            if self.cs_weight < 0:
                raise ValueError(f"cs_anchor.weight must be >= 0, got {self.cs_weight}")
            if not self.concept_enabled:
                raise ValueError(
                    "cs_anchor is enabled but `concept.enabled` is false, so the loader never "
                    "passes per-repetition EEG and the term could not fire. Set "
                    "`concept: {enabled: true, reps_weight: 0.0}`.")
        # D5 / §10.3-4: freeze the IMAGE-SIDE projector. `img_pre` is 204,864 parameters
        # (18.1% of the model) and it is currently trainable, which means the "frozen
        # teacher" anchor is not frozen at all -- the target space can bend toward the
        # source subjects, and a bend toward the source is exactly what widens the
        # train/held-out subject gap. SCORE and SVTL both keep the image side fixed.
        # This is one line on purpose: it is a single-variable diagnostic for the
        # +31.22pp gap, not an architecture change.
        self.freeze_img_pre = bool(self.cfg.get("freeze_img_pre", False))
        if self.freeze_img_pre:
            if not hasattr(self.model, "img_pre"):
                raise ValueError(
                    "freeze_img_pre is set but this model has no `img_pre` projector "
                    "(arch=v4 exposes it); the flag would silently do nothing")
            for p in self.model.img_pre.parameters():
                p.requires_grad_(False)

        # Which id a prototype is keyed by. A data decision, made once here and read
        # by both `assemble` (loss) and `update_prototype` (EMA), so the two cannot
        # disagree about what a prototype means.
        self.proto_level = str(self.cfg.get("prototype_level", "concept"))
        if self.proto_level not in ("concept", "stimulus"):
            raise ValueError("prototype_level must be 'concept' or 'stimulus', got "
                             f"{self.proto_level!r}")
        # Which kernel the v3 subject-MMD term uses. Only read when `objective == 'v3'`;
        # see `mmd_subject`: the RBF form cannot see the subject shift on this
        # representation, so `linear` is the default and `rbf` exists only as an ablation.
        self.mmd_kernel = str(self.cfg.get("mmd_kernel", "linear"))
        # The schedule's two coefficients for the CURRENT epoch, written by the training
        # loop before the batch loop and read by `assemble`. Kept as attributes rather
        # than as mutated `LossWeights` fields because v4 multiplies them into the
        # objective's weights while v3 lets the loop overwrite those weights outright --
        # see the branch at the top of `assemble`. Defaults of 1.0/1.0 make a run with
        # `coarse_to_fine: false` reproduce the YAML weights exactly.
        self.coarse_weight = 1.0
        self.fine_weight = 1.0

        # v6 SCORE-LEVEL FUSION. Parsed here and only here, and only for `objective: v6`
        # -- a fusion block on a v3/v4 run would be read by nobody, which is the silent
        # failure `LossWeights.validate` exists to refuse. The weight is a config value
        # rather than a `loss_weights` key so that adding it cannot change what a
        # recorded v3/v4 config means (the same reason `spec` defaults to 0).
        fus = self.cfg.get("fusion") or {}
        if not isinstance(fus, dict):
            raise ValueError(f"`fusion` must be a mapping, got {type(fus).__name__}")
        self.fuse_weight = float(fus.get("weight", 1.0))
        #: Global per-route score normalisation before the sum (`score_fusion_loss`).
        #: On by default: CSLS-corrected scores are unbounded and raw cosines are in
        #: [-1, 1], so an unnormalised sum lets the CSLS route silently dominate.
        self.fuse_normalize = bool(fus.get("normalize", True))
        if self.objective != "v6" and (self.fuse_weight != 1.0 or not self.fuse_normalize):
            raise ValueError(
                f"`fusion` is only read by objective='v6', but objective={self.objective!r} "
                f"was given weight={self.fuse_weight}, normalize={self.fuse_normalize}. A "
                f"fusion block on this objective is inert decoration; remove it or use "
                f"objective='v6'.")

    def criterion_parameters(self) -> list[torch.nn.Parameter]:
        """The contrast temperatures, for the optimiser.

        The criteria live on the `Trainer`, not on the model, and `Trainer` is a plain
        dataclass rather than an `nn.Module` -- so `list(model.parameters())` silently
        omits them and `logit_scale` stays pinned at `init_temp` for the whole run,
        despite being documented as learnable.

        `PrototypeEMA` is included for the same reason: it is attached to the Trainer,
        so its temperature is invisible to `model.parameters()` too. Only its
        `logit_scale` is a Parameter -- the prototype bank and its counts are buffers
        and must NOT reach the optimiser.
        """
        params = list(self.crit_img.parameters()) + list(self.crit_cross.parameters())
        if self.prototype is not None:
            params += list(self.prototype.parameters())
        return params

    def criterion_state(self) -> dict:
        """The temperatures, for the checkpoint.

        They are optimised parameters that live outside the model, so a checkpoint that
        stores only `model.state_dict()` silently discards them: a resumed run would
        rebuild `InfoNCE` at `init_temp` and the contrast sharpness would not be the one
        the saved model was scored with.
        """
        state = {"img": self.crit_img.state_dict(),
                 "cross": self.crit_cross.state_dict()}
        if self.prototype is not None:
            state["proto"] = self.prototype.state_dict()
        return state

    def load_criterion_state(self, blob: dict | None) -> bool:
        """Restore temperatures from a checkpoint. Returns False for legacy/no state."""
        if not blob:
            return False
        self.crit_img.load_state_dict(blob["img"])
        self.crit_cross.load_state_dict(blob["cross"])
        # `proto` is absent from checkpoints written before the prototype contrast had a
        # temperature, so its presence is the version marker rather than an assumption.
        if self.prototype is not None and "proto" in blob:
            self.prototype.load_state_dict(blob["proto"])
        return True

    def prototype_group(self, batch: dict) -> torch.Tensor | None:
        """The id a prototype is keyed by, per row (see `PrototypeEMA`)."""
        key = "concept" if self.proto_level == "concept" else "stimulus"
        g = batch.get(key)
        if g is None:
            raise KeyError(
                f"prototype_level={self.proto_level!r} needs batch[{key!r}], which this "
                f"loader did not provide. Keys present: {sorted(batch)}. A silently "
                f"missing key would disable the prototype term while the log still "
                f"reported it, so this raises instead.")
        return g

    def _recovery_episode(self, z_e: torch.Tensor, z_i: torch.Tensor,
                          subject: torch.Tensor,
                          s_star: torch.Tensor | None = None) -> torch.Tensor:
        """Recover ONE source subject's coordinates, as deployment would for a new one.

        SCORE (arXiv 2608.19134) constructs "source-only recovery episodes" during
        training: each mini-batch treats one source subject as a temporary target, hides
        its EEG-image matches, runs the SAME recovery used at deployment, and only then
        reveals the matches to compute the loss.

        The subject is chosen by a counter rather than at random so that every source
        subject is the temporary target equally often and a run is reproducible from its
        seed. Only that subject's rows are replaced; the rest of the batch keeps its raw
        coordinates. That is deliberate -- the episode is a *simulation of deployment for
        one unseen subject*, so applying it to the whole batch would simulate a world in
        which every subject is unseen at once, which is a different (and easier) problem.
        """
        subs = torch.unique(subject)
        if int(subs.numel()) < 2:
            return z_e
        if s_star is None:
            s_star = subs[self._episode_step % int(subs.numel())]
            self._episode_step += 1
        mask = subject == s_star
        if int(mask.sum()) < 4:
            # too few rows to form landmarks; the episode abstains rather than fitting a
            # rotation from almost nothing
            return z_e

        from .losses import recovery_episode as _recover
        from .losses.episode import deploy_stack_episode as _deploy_episode

        if self.ep_full:
            z_rec, diag = _deploy_episode(
                z_e[mask], z_i, k=self.ep_k, rho=self.ep_rho,
                min_landmarks=self.ep_min_landmarks, whiten=self.ep_whiten,
                shrink=self.ep_shrink, max_cond=self.ep_max_cond,
            )
        else:
            z_rec, diag = _recover(
                z_e[mask], z_i, k=self.ep_k, rho=self.ep_rho,
                min_landmarks=self.ep_min_landmarks,
            )
        # The last episode's diagnostics are kept for logging: "the episode silently
        # abstained on every step" and "the episode ran" produce identical training
        # curves, so a run that never fires must be distinguishable from one that does.
        self._ep_last_diag = diag
        if diag.get("abstained"):
            self._ep_abstains += 1
            return z_e
        self._ep_fires += 1
        out = z_e.clone()
        out[mask] = z_rec
        return out

    def _assemble_v6(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """v6 objective: N routes, each trained in the deployed metric, then SCORE fusion.

        Structure (see `docs/eeg2image_v6_architecture.md` §2 and §3):

          1. ONE shared trunk pass, N per-route ``(z_eeg, z_img)`` pairs.
          2. Each route's ``img`` / ``cross`` terms are scored EXACTLY as on v4 -- same
             per-subject block structure and same CSLS correction -- so "route r landed at
             X" means the same thing here as a v4 run of that route alone. Averaging the
             per-route terms (rather than summing) keeps the objective's scale comparable
             to v4, which matters because the temperature is shared.
          3. The ``fuse`` term adds the routes' score matrices INSIDE each subject block
             and scores the sum with the SAME criterion. Per block, not per batch: the
             density CSLS removes is a property of the retrieval set, so fusing across a
             mixed-subject batch would train against a density deployment never sees --
             the same argument `recovery_aware_alignment` records.

        The fusion is a TRAINING term and the routes' own terms are kept alongside it
        deliberately: a fusion-only objective has no gradient that says "route gamma must
        be individually good", and a route that is individually useless but adds a little
        complementary information is exactly the case where summing helps and specialising
        does not. Keeping both means a route that collapses shows up as its own ``img``
        term rising, rather than being hidden inside a flat fused curve.
        """
        eeg = batch["eeg"]
        subject = batch.get("subject")
        stimulus = batch.get("stimulus")
        names = list(getattr(self.model, "route_names", []))
        if not names:
            raise ValueError(
                "objective='v6' requires a multi-route model (`MultiRouteSAMCLIP`); the "
                "configured model exposes no `route_names`. Build it via "
                "`build_model(cfg)` with a `routes:` list.")
        targets: dict[str, torch.Tensor] = {}
        primary = str(getattr(self.model, "primary", names[0]))
        for n in names:
            key = f"target__{n}"
            # The PRIMARY route's stack is carried by the plain `target` key (the loader
            # builds `target__*` only for the non-primary routes), so duplicating it into
            # `target__*` would put a second full copy of a ~1 GB stack in every
            # DataLoader worker. The fallback is therefore explicit and primary-only.
            if key in batch:
                targets[n] = batch[key]
            elif n == primary and "target" in batch:
                targets[n] = batch["target"]
            else:
                raise KeyError(
                    f"objective='v6' cannot find a target stack for route {n!r}: neither "
                    f"{key!r} nor (for the primary route) 'target' is in the batch. The "
                    f"loader builds `target__*` from the route caches (`build_targets`), "
                    f"so this means the route list and the target stack disagree. Got "
                    f"{sorted(batch)}.")

        out = self.model(eeg, targets, subject_ids=subject, training=True)
        views = out["routes"]

        # The source-only episode, applied PER ROUTE but to the SAME temporary target
        # subject, chosen once here. Each route owns its own `z_i`, so there is no single
        # rotation that recovers all of them -- but letting each route advance the
        # counter independently would have the three routes simulate three DIFFERENT
        # held-out subjects in one step, which is three deployment simulations rather than
        # one. Choosing `s_star` here keeps "this step simulates subject A" true for every
        # route. Applied BEFORE any term, as on v4.
        if self.ep_enabled and subject is not None:
            subs = torch.unique(subject)
            s_star = (subs[self._episode_step % int(subs.numel())]
                      if int(subs.numel()) >= 2 else None)
            if s_star is not None:
                self._episode_step += 1
            for n in names:
                views[n]["z_eeg"] = self._recovery_episode(
                    views[n]["z_eeg"], views[n]["z_img"], subject, s_star=s_star)

        w_img = self.fine_weight * self.weights.img
        w_cross = self.fine_weight * self.weights.cross
        w_mmd = self.coarse_weight * self.weights.mmd
        w_fuse = self.fine_weight * self.fuse_weight
        ra_k = self.ra_csls_k if "img" in self.ra_terms else None
        cross_k = self.ra_csls_k if "cross" in self.ra_terms else None

        parts: dict[str, torch.Tensor] = {}
        loss = None
        img_terms: list[torch.Tensor] = []
        per_route_img: dict[str, float] = {}
        for n in names:
            z_e, z_i = views[n]["z_eeg"], views[n]["z_img"]
            t = recovery_aware_alignment(z_e, z_i, subject, self.crit_img, csls_k=ra_k)
            img_terms.append(t)
            per_route_img[n] = float(t.detach())
        img_term = torch.stack(img_terms).mean()
        loss = w_img * img_term
        parts["img"] = img_term.detach()
        for n, v in per_route_img.items():
            parts[f"img_{n}"] = torch.as_tensor(v, device=img_term.device)
        parts["sc_img"] = self.crit_img.effective_scale().detach()
        parts["sc_x"] = self.crit_cross.effective_scale().detach()

        if w_cross > 0 and stimulus is not None:
            ts = [cross_subject_loss(views[n]["z_eeg"], stimulus, self.crit_cross,
                                     csls_k=cross_k) for n in names]
            l_cross = torch.stack(ts).mean()
            loss = loss + w_cross * l_cross
            parts["cross"] = l_cross.detach()

        if w_mmd > 0:
            ms = [mmd_crossmodal(views[n]["z_eeg"], views[n]["z_img"]) for n in names]
            l_mmd = torch.stack(ms).mean()
            loss = loss + w_mmd * l_mmd
            parts["mmd"] = l_mmd.detach()

        if w_fuse > 0:
            # Blocks of >= 2 rows; a 1-row block cannot form a neighbourhood and is
            # skipped rather than contributing a degenerate ln(1)=0 gradient.
            blocks: list[list[torch.Tensor]] = []
            if subject is not None:
                for s in torch.unique(subject):
                    rows = subject == s
                    if int(rows.sum()) < 2:
                        continue
                    mats = []
                    for n in names:
                        cos = views[n]["z_eeg"][rows] @ views[n]["z_img"][rows].t()
                        if ra_k is not None:
                            cos = csls_correct(cos, k=ra_k)
                        mats.append(cos)
                    blocks.append(mats)
            if not blocks:
                mats = []
                for n in names:
                    cos = views[n]["z_eeg"] @ views[n]["z_img"].t()
                    if ra_k is not None:
                        cos = csls_correct(cos, k=ra_k)
                    mats.append(cos)
                blocks = [mats]
            fuse_terms = [score_fusion_loss(m, self.crit_img,
                                            normalize=self.fuse_normalize)
                          for m in blocks]
            l_fuse = torch.stack(fuse_terms).mean()
            loss = loss + w_fuse * l_fuse
            parts["fuse"] = l_fuse.detach()
            parts["fuse_blocks"] = torch.as_tensor(float(len(blocks)),
                                                   device=l_fuse.device)

        if self.weights.router > 0:
            # Read off the route-0 forward pass the same way v4 does; `None` means uniform
            # fusion, which is legitimate and must not add a term.
            pen = self.model.router_entropy()
            if pen is not None:
                loss = loss + self.weights.router * pen
                parts["router"] = pen.detach()

        parts["total"] = loss.detach()
        return loss, parts

    def assemble(self, batch: dict) -> tuple[torch.Tensor, dict]:
        if self.objective == "v6":
            return self._assemble_v6(batch)
        eeg = batch["eeg"]
        target = batch["target"]
        stimulus = batch.get("stimulus")
        subject = batch.get("subject")

        out = self.model(eeg, target, subject_ids=subject, training=True)
        z_e, z_i = out["z_eeg"], out["z_img"]

        # SCORE's source-only episode, applied BEFORE any term is scored, so every fine
        # term sees the recovered coordinates rather than the raw ones. Disabled is a
        # no-op (the branch is not taken at all), which is what keeps the flag honest.
        if self.ep_enabled and subject is not None:
            z_e = self._recovery_episode(z_e, z_i, subject)

        parts: dict[str, torch.Tensor] = {}
        # HOW THE SCHEDULE MULTIPLIES IN depends on the objective, and the difference is
        # not cosmetic. v3's config expresses the schedule DIRECTLY as `weights.mmd` /
        # `weights.cross` (the loop assigns the coefficients there), so the weights are
        # absolute. v4 separates the two levels: `loss_weights` holds the RATIO between
        # the fine terms (`img: 1.0`, `cross: 0.7`, `mmd: 1.0`) and the schedule holds
        # the PHASE coefficient, and the two multiply. Applying the phase coefficient to
        # weights the loop had already overwritten would square the ramp.
        if self.objective == "v4":
            w_img = self.fine_weight * self.weights.img
            w_cross = self.fine_weight * self.weights.cross
            w_mmd = self.coarse_weight * self.weights.mmd
        else:
            w_img, w_cross, w_mmd = self.weights.img, self.weights.cross, self.weights.mmd

        # RECOVERY-AWARE OBJECTIVE (v5 pillar A). When enabled, the fine terms are scored
        # in the metric the score is PRODUCED in, rather than in raw cosine:
        #   * `block_per_subject` scores `img` one subject at a time. This is not
        #     cosmetic -- CSLS's neighbourhood density is a property of the retrieval set,
        #     and deployment always scores ONE subject's queries against the gallery. A
        #     mixed-subject matrix has a density no deployment sees.
        #   * `csls_k` replaces the logits with the CSLS-corrected ones, so the encoder is
        #     pushed off hub structure instead of being handed it at test time.
        # Both default off, and off is bit-identical to every recorded run (there is no
        # branch taken, not a no-op branch).
        ra_k = self.ra_csls_k if "img" in self.ra_terms else None
        if self.ra_enabled and self.ra_block:
            img_term = recovery_aware_alignment(z_e, z_i, subject, self.crit_img,
                                                csls_k=ra_k)
        else:
            img_term = clip_alignment_loss(z_e, z_i, self.crit_img, csls_k=ra_k)
        loss = w_img * img_term
        parts["img"] = img_term.detach()
        # Logged every step: `effective_scale` pinned at its floor is the signature of
        # the temperature collapse described at `contrastive.SCALE_MIN`, and it is the
        # one failure that looks exactly like a plateau in the loss.
        parts["sc_img"] = self.crit_img.effective_scale().detach()
        parts["sc_x"] = self.crit_cross.effective_scale().detach()

        if w_cross > 0 and stimulus is not None:
            cross_k = self.ra_csls_k if "cross" in self.ra_terms else None
            l_cross = cross_subject_loss(z_e, stimulus, self.crit_cross, csls_k=cross_k)
            loss = loss + w_cross * l_cross
            parts["cross"] = l_cross.detach()

        if w_mmd > 0:
            if self.objective == "v4":
                # C4: the reference's term, EEG cloud against IMAGE cloud. Only
                # meaningful on a model whose two heads land in one space, which is
                # exactly what `share_head` guarantees (`arch: v4`). `LossWeights.
                # validate` does not enforce that pairing -- L_mmd is a legitimate arm on
                # the v3 topology -- but a `objective: v4` run on `arch: v3` would be
                # matching two separately-parameterised projections, which is not the
                # same mechanism, so it is refused rather than reported as one.
                if getattr(self.model, "arch", "v3") == "v3":
                    raise ValueError(
                        "objective='v4' builds L_mmd between the EEG and image clouds, "
                        "which is only a distribution match in a SHARED space; the "
                        "configured model is `arch: v3` with two independent heads. "
                        "Set `arch: v4` or use `objective: v3`.")
                l_mmd = mmd_crossmodal(z_e, z_i)
            elif subject is not None:
                l_mmd = mmd_subject(z_e, subject, kernel=self.mmd_kernel)
            else:
                l_mmd = None
            if l_mmd is not None:
                loss = loss + w_mmd * l_mmd
                parts["mmd"] = l_mmd.detach()

        if self.weights.spec > 0:            # C3: computed on `z_eeg`, the POST-SMN, L2-normalised embedding -- the space
            # retrieval is scored in. It must NOT be computed on `z_eeg_raw`: the raw
            # embedding still contains the per-subject offset, which is a high-variance
            # direction, and a term that rewards concentrating energy in the head would
            # then be REWARDED for pouring energy into exactly the subject structure the
            # redesign exists to remove. Both spectral forms are scale-invariant, so
            # normalising first costs the term nothing.
            #
            # `spec_mode` picks the functional form, and the default is NOT the one we
            # would choose today -- see the `spec_mode` note in `__init__` and the
            # pathology section of `spectral_concentration`. `tail` (default, historical)
            # has total collapse as its global optimum and measured inert in an ablation;
            # `rank` is the corrected bilateral target.
            l_spec = (spectral_rank_target(z_e, r0=self.spec_r0)
                      if self.spec_mode == "rank"
                      else spectral_concentration(z_e, r0=self.spec_r0))
            loss = loss + self.weights.spec * l_spec
            parts["spec"] = l_spec.detach()

        if self.weights.reg > 0:
            # UN-NORMALISED, and on v4 this term is FORBIDDEN (`LossWeights.validate`
            # rejects `reg > 0` on `objective: v4`) -- see the measurements below. v3
            # keeps it verbatim so every recorded v3 run stays reproducible.
            #
            # `var` was already excluded on v4, where `L_spec` takes its place. The v3
            # measurement behind that switch: all 512 per-dim stds sat at 0.28-0.55
            # (below the 1.0 hinge target), so `var` was a large UNIFORM push (0.60) that
            # could never be satisfied, while `cov` contributed 0.057 -- 0.09x as much --
            # and the representation still lived in ~13 effective dimensions. A uniform
            # variance push cannot raise the rank; asking for concentration is the
            # attainable version of the same request.
            #
            # WHY `cov` IS DROPPED ON v4 RATHER THAN RE-SCALED. On v4 the term measured
            # 4.6e-09: the head is Linear/GELU/Linear, so `z_eeg_raw` has mean row norm
            # 0.26 against v3's O(1), and `(off**2).sum()/d` scales as ||z||^4 -- a 4x
            # smaller embedding is a 256x smaller penalty, i.e. `reg: 0.04` was
            # decoration, which is why the objective must REFUSE it instead of carrying
            # it. Re-scaling does not rescue it, because the problem is not the scale --
            # the term barely responds to the structure it is supposed to penalise. On a
            # 384x64 batch, compared against a representation with one axis EXACTLY
            # duplicated, the penalty moves by 14%:
            #
            #   form              random input   one duplicated axis
            #   raw  (as written)   6.93e-03         7.86e-03
            #   row-normalised      1.63e-06         1.87e-06
            #
            # Row-normalising does make it scale-invariant, but it also collapses every
            # covariance entry to ~1/d, so the term becomes a constant ~1e-6 that no
            # amount of redundancy can move. `L_spec` already asks for the attainable
            # version of the same request (a concentrated spectrum) and is scale-invariant
            # by construction, so `cov` adds a near-constant to the objective and nothing
            # else. A term that cannot distinguish its own target from noise is not a
            # weak regulariser, it is an absent one with a weight attached.
            reg = vicreg_terms(out["z_eeg_raw"],
                               cov_weight=float(self.cfg.get("vicreg_cov_weight", 0.04)))
            l_reg = reg["cov"] if self.objective == "v4" else (reg["var"] + reg["cov"])
            loss = loss + self.weights.reg * l_reg
            parts["cov"] = reg["cov"].detach()
            if self.objective != "v4":
                parts["var"] = reg["var"].detach()

        if self.weights.dec > 0 and subject is not None and subject.unique().numel() > 1:
            l_dec = hsic_subject(z_e, subject, self.n_subjects)
            loss = loss + self.weights.dec * l_dec
            parts["dec"] = l_dec.detach()

        if self.weights.proto > 0 and self.prototype is not None:
            group = self.prototype_group(batch)
            l_proto = self.prototype.proto_contrast(z_e, group)
            # The asymmetric half: the IMAGE embeddings are pulled to the EEG anchors.
            # See `PrototypeEMA` -- the direction is what makes this an anchor rather
            # than a second, competing contrastive term.
            l_anchor = self.prototype.image_anchor_contrast(z_i, group)
            loss = loss + self.weights.proto * (l_proto + l_anchor)
            parts["proto"] = l_proto.detach()
            parts["anchor"] = l_anchor.detach()
            # Logged for the same reason `sc_img` is: without a temperature the cosine
            # softmax is uniform and this term decouples from the encoder while its
            # loss value still looks like a normal cross-entropy.
            parts["sc_proto"] = self.prototype.effective_scale().detach()

        if self.weights.rkd > 0:
            l_rkd = gram_distill_loss(z_e, z_i)
            loss = loss + self.weights.rkd * l_rkd
            parts["rkd"] = l_rkd.detach()

        if self.weights.adv > 0 and self.adversary is not None and subject is not None:
            logits = self.adversary(z_e)
            l_adv = SubjectAdversary.loss(logits, subject)
            loss = loss + self.weights.adv * l_adv
            parts["adv"] = l_adv.detach()

        if self.weights.router > 0:
            # Read off the SAME forward pass that produced `z_i` above, so the bonus is
            # applied to the blend the loss was actually computed through. `None` means
            # the fusion has no learned weights (`mean`) or the model is not training --
            # both are legitimate, and neither should add a term.
            pen = self.model.router_entropy()
            if pen is not None:
                loss = loss + self.weights.router * pen
                parts["router"] = pen.detach()

        if self.reps_weight > 0 and batch.get("reps") is not None:
            # v7 T2'' -- CROSS-TRIAL REPETITION COLLAPSE, grouped by STIMULUS. The
            # repetitions are encoded by a SECOND call of the same shared encoder, not a
            # different one: the term's claim is that the deployed encoder maps every
            # (subject, repetition) view of one image to one point, which is only a
            # statement about the encoder that is actually deployed.
            #
            # ROW SELECTION IS A BALANCED (stimulus, subject) DESIGN, not a prefix. The
            # sampler emits rows stimulus-major (9 subjects per stimulus), so a prefix of
            # 64 rows would span only ~7 stimuli -- a 7-way contrast that wastes the other
            # 121 as negatives. Taking `n_stim` stimuli x `subj_sel` subjects each gives a
            # `n_stim`-way problem over the SAME number of encoder passes.
            reps = batch["reps"]                       # (B, R, Ch, T)
            stim = batch.get("stimulus")
            r_use = min(self.reps_r_use, reps.shape[1])
            if self.reps_group == "stimulus" and stim is not None:
                sel = self._reps_rows(stim)
                grp = stim[sel]                        # STIMULUS id per selected row
            else:
                sel = torch.arange(min(self.reps_n_rows, reps.shape[0]), device=reps.device)
                grp = None
            n_sel = int(sel.numel())
            if n_sel < 2:
                raise RuntimeError(
                    f"the reps term selected {n_sel} rows; the contrast needs >= 2 groups. "
                    f"Check `reps_n_stim`/`reps_subj_sel` against the sampler's "
                    f"`batch_stimuli`.")
            sub = reps[sel][:, :r_use]                 # (n_sel, r_use, Ch, T)
            flat = sub.reshape(n_sel * r_use, *sub.shape[2:])
            subj = None if subject is None else subject[sel].repeat_interleave(r_use)
            z_rep = self.model.encode_eeg(flat, subject_ids=subj,
                                          detach_gate=self.reps_detach_gate)
            l_reps = repetition_collapse_loss(
                z_rep.reshape(n_sel, r_use, -1), self.crit_cross,
                csls_k=self.reps_csls_k, groups=grp)
            loss = loss + self.reps_weight * l_reps
            parts["reps"] = l_reps.detach()

        if self.v12_distill > 0 or self.v12_consistency > 0:
            # v12 METRIC SELF-DISTILLATION (docs/eeg2image_v12_core_claim.md Part C). The teacher
            # is the v11 reliability-fused consensus metric over independent repetition blocks --
            # the estimator `probe_fusion.py` measured to carry shared structure with a
            # correspondence-destroying control at zero (job 645697). The student is a
            # SINGLE-repetition metric, so the term forces one trial to carry what the whole cloud
            # carries, which is the only lever the R-curve showed to have headroom (Top-1 still
            # climbing +3.33pp at R=80, job 645719). It is deliberately NOT another contrastive
            # term: InfoNCE already shapes the averaged embedding, and the R-curve says the
            # averaged embedding is not what limits retrieval.
            if batch.get("reps") is None:
                raise RuntimeError(
                    "v12 metric terms need `batch['reps']` (per-repetition EEG). Train with the "
                    "concept/reps loader enabled; otherwise the term would be a no-op reported as "
                    "a mechanism.")
            if subject is None:
                raise RuntimeError("v12 metric terms need subject ids to form per-subject blocks.")
            _reps = batch["reps"]
            _stim = batch.get("stimulus")
            if _stim is None:
                raise RuntimeError("v12 metric terms need stimulus ids to define the metric rows.")
            _r = int(min(self.v12_r_use, _reps.shape[1]))
            _sel = self._reps_rows(_stim, n_stim=self.v12_n_stim, subj_sel=self.v12_subj_sel)
            _n = int(_sel.numel())
            _sub = _reps[_sel][:, :_r]
            _flat = _sub.reshape(_n * _r, *_sub.shape[2:])
            _subj_flat = subject[_sel].repeat_interleave(_r)
            _z = self.model.encode_eeg(_flat, subject_ids=_subj_flat,
                                       detach_gate=self.v12_detach_gate)
            _z = _z.reshape(_n, _r, -1)
            _grp = _stim[_sel]
            _subj_row = subject[_sel]
            if self.v12_distill > 0:
                _ld, _dd = metric_self_distill(
                    _z, _grp, _subj_row, student_reps=self.v12_student_reps,
                    teacher_blocks=self.v12_teacher_blocks, return_diag=True)
                loss = loss + self.v12_distill * _ld
                parts["v12_distill"] = _ld.detach()
                parts.update(_dd)
            if self.v12_consistency > 0:
                _lc, _dc = metric_subject_consistency(
                    _z, _grp, _subj_row, student_reps=self.v12_student_reps, return_diag=True)
                loss = loss + self.v12_consistency * _lc
                parts["v12_consistency"] = _lc.detach()
                parts.update(_dc)

        # ---- v14: the per-subject REPETITION BLOCK, encoded ONCE for both order-2 terms ---------
        # `bias_anchor` and `cs_anchor` consume the same object (per-repetition EEG for a
        # stimulus x subject tile) and differ only in the ANCHOR, so encoding it twice would spend a
        # second encoder pass on identical input. More importantly, a shared block makes the two
        # terms' diagnostics directly comparable -- two separately-constructed tiles could differ in
        # row order and the disagreement would look like a mechanism.
        _zb = _sel = _stim_t = _subj_t = None
        if self.ba_weight > 0 or self.cs_weight > 0:
            if batch.get("reps") is None:
                raise RuntimeError("order-2 anchors need `batch['reps']`; see Trainer.__init__.")
            if subject is None:
                raise RuntimeError("order-2 anchors need subject ids to form per-subject blocks.")
            _reps = batch["reps"]
            _stim_all = batch.get("stimulus")
            if _stim_all is None:
                raise RuntimeError("order-2 anchors need stimulus ids to define the metric rows.")
            _sel = self._reps_rows(_stim_all, n_stim=max(self.ba_n_stim, 32),
                                   subj_sel=max(self.ba_subj_sel, 3))
            _n = int(_sel.numel())
            _sub = _reps[_sel]
            _r_all = int(_sub.shape[1])
            _flat = _sub.reshape(_n * _r_all, *_sub.shape[2:])
            _subj_flat = subject[_sel].repeat_interleave(_r_all)
            _zb = self.model.encode_eeg(_flat, subject_ids=_subj_flat,
                                        detach_gate=self.v12_detach_gate).reshape(_n, _r_all, -1)
            _stim_t = _stim_all[_sel]
            _subj_t = subject[_sel]

        if self.ba_weight > 0:
            # v13 ORDER-2 BIAS ANCHOR, scored in the v14 `frame`. The encoder is asked to make each
            # SUBJECT's pooled concept metric agree with the exogenous gallery metric. `z_i` is
            # reused from the fine contrast (same rows, same order), so the anchor is the metric of
            # the very images this batch is already labelled with -- a richer readout of the
            # existing supervision, not new information. `student_reps=None` scores the POOLED
            # metric, i.e. the deployed readout, because the measured gap is a pooled-level subject
            # bias.
            _r = int(min(self.ba_r_use, _zb.shape[1]))
            l_ba, ba_diag = metric_anchor_loss_by_subject(
                _zb[:, :_r], z_i[_sel], _stim_t, _subj_t,
                student_reps=self.ba_student_reps, min_concepts=self.ba_min_concepts,
                return_diag=True, frame=self.ba_frame, whiten_shrink=self.ba_shrink)
            loss = loss + self.ba_weight * l_ba
            parts["bias_anchor"] = l_ba.detach()
            parts["ba_agreement"] = torch.tensor(ba_diag["bias_anchor_agreement"],
                                                 device=l_ba.device)
            parts["ba_student_rank"] = torch.tensor(ba_diag["bias_anchor_student_rank"],
                                                    device=l_ba.device)
            parts["ba_pairs"] = torch.tensor(float(ba_diag["bias_anchor_pairs"]),
                                             device=l_ba.device)

        if self.cs_weight > 0:
            # v14 C-ANCHOR: the anchor is the CONSENSUS of the gallery and the other subjects in the
            # batch, i.e. "subjects must agree", which a single gallery view cannot express. The
            # measured target is the subject-specific gap (9 encodings 0.796 vs one subject's 0.616,
            # t=+22.2); this term is the training-side gradient of that quantity.
            l_cs, cs_diag = subject_consensus_anchor_loss(
                _zb, z_i[_sel], _stim_t, _subj_t, min_concepts=self.cs_min_concepts,
                return_diag=True, frame=self.cs_frame, whiten_shrink=self.cs_shrink,
                include_gallery=self.cs_include_gallery)
            loss = loss + self.cs_weight * l_cs
            parts["cs_anchor"] = l_cs.detach()
            parts["cs_agreement"] = torch.tensor(cs_diag["cs_agreement"], device=l_cs.device)
            parts["cs_student_rank"] = torch.tensor(cs_diag["cs_student_rank"],
                                                    device=l_cs.device)
            parts["cs_pairs"] = torch.tensor(float(cs_diag["cs_pairs"]), device=l_cs.device)

        if self.sp_weight > 0:
            # v8 SOFT-PLAN ALIGNMENT. Built in the score's own metric (`csls_k`) inside a
            # per-subject block, on the SAME z_e/z_i the fine contrast used, so the term
            # shapes the representation the deployed operator will be fitted to rather than a
            # separate head. `diag_mass` is the plan's mass on the true correspondences: it
            # starts near 1/C and must RISE, and a flat or falling value is the signal that
            # the term is not doing what its name says -- which is the diagnostic whose
            # absence let three earlier pillars run to completion before being read as
            # failures.
            l_sp, sp_diag = soft_plan_loss(
                z_e, z_i, block=(subject if self.sp_block else None),
                tau=self.sp_tau, iters=self.sp_iters, csls_k=self.sp_csls_k,
                alpha=self.sp_alpha, fgw_outer=self.sp_fgw_outer,
                return_diag=True)
            loss = loss + self.sp_weight * l_sp
            parts["soft_plan"] = l_sp.detach()
            parts["sp_diag_mass"] = torch.tensor(
                sp_diag["soft_plan_diag_mass"], device=l_sp.device)
            parts["sp_entropy"] = torch.tensor(
                sp_diag["soft_plan_entropy"], device=l_sp.device)

        parts["total"] = loss.detach()
        return loss, parts

    def _reps_rows(self, stim: torch.Tensor, n_stim: int | None = None,
                   subj_sel: int | None = None) -> torch.Tensor:
        """Row positions for the reps term: ``n_stim`` stimuli x ``subj_sel`` rows each.

        ASSERTS THE LAYOUT IT RELIES ON rather than assuming it. `CrossSubjectBatchSampler`
        emits rows stimulus-major with an equal group size (``batch_stimuli`` chunks of
        ``subjects_per_stimulus * images_per_pair``), so the first ``subj_sel`` rows of each
        block are distinct subjects viewing one image. If a future sampler change breaks
        that, the check below fires instead of the term silently grouping across stimuli --
        which is the difference between "this mechanism failed" and "this mechanism was
        never tested". The selection is cached because the positions cannot change while the
        batch shape and the layout hold.

        `n_stim`/`subj_sel` override the concept term's values: the v12 metric terms select
        their own rows (they need more repetitions per row than the concept term uses), and a
        silent fallback to the concept term's selection would make a differently-configured v12
        arm quietly run the concept term's rows instead.
        """
        n_stim = self.reps_n_stim if n_stim is None else int(n_stim)
        subj_sel = self.reps_subj_sel if subj_sel is None else int(subj_sel)
        n = int(stim.shape[0])
        key = (n, n_stim, subj_sel, int(stim.device.index or -1))
        cached = self._reps_pos_cache.get(key)
        if cached is not None and cached.device == stim.device:
            return cached
        uniq = int(torch.unique(stim).numel())
        g = n // max(uniq, 1)
        if g < 1 or g * uniq != n:
            raise RuntimeError(
                f"the reps term needs an equal-sized, contiguous stimulus layout; batch "
                f"has {n} rows and {uniq} distinct stimuli, which does not divide evenly. "
                f"Set `concept.reps_group: row` to run the (anti-T1) row-grouped arm, or fix "
                f"the sampler.")
        blk = stim.reshape(uniq, g)
        if not bool((blk == blk[:, :1]).all()):
            raise RuntimeError(
                "the batch is not stimulus-major, so `reps_group: stimulus` would group "
                "rows that share a POSITION but not an image -- the term would then train "
                "the opposite of the configured objective. See `CrossSubjectBatchSampler`.")
        n_stim = min(n_stim, uniq)
        subj_sel = min(subj_sel, g)
        ar = torch.arange
        idx = (ar(n_stim, device=stim.device).unsqueeze(1) * g
               + ar(subj_sel, device=stim.device).unsqueeze(0)).reshape(-1)
        self._reps_pos_cache[key] = idx
        return idx

    @torch.no_grad()
    def update_prototype(self, batch: dict) -> None:
        """Update the EMA bank from the CURRENT batch's EEG embeddings.

        Called after `opt.step()`, so the prototypes track the encoder as it moves
        rather than lagging a step behind it. `encode_eeg` is run in eval mode semantics
        (no dropout) via `self.model.eval()` toggling at the call site -- the model is
        put back into train mode by the loop, so a dropout draw here would not perturb
        training but would add noise to the anchors for no benefit.
        """
        if self.prototype is None:
            return
        was = self.model.training
        self.model.eval()
        z = self.model.encode_eeg(batch["eeg"])
        self.model.train(was)
        self.prototype.update(z, self.prototype_group(batch))


# --------------------------------------------------------------- data helpers
def freeze_encoder_for_stage2(model: nn.Module, trainer: "Trainer",
                              extra: list[nn.Parameter] | None = None
                              ) -> tuple[list[nn.Parameter], int]:
    """Freeze the EEG trunk and return ``(trainable_params, n_frozen_scalars)``.

    Called once, on the first Stage-2 epoch. This is the reference's recipe
    (``build_optimizer(stage2_learning_rate, include_share_encoder=False)``) and it
    replaces v3's joint Stage 2, whose measured signature was a dip-then-recover
    (Top-1 21.0 -> 17.5 -> 20.5) -- a phase boundary that cost progress and gave it back.

    What is frozen is the EEG TRUNK. The modality-private pre-projections, the shared
    head and the router stay trainable: those are the "projectors/router" the reference
    keeps optimising at ``stage2_lr``. The SMN's gate is also kept -- it is a parameter
    of the forward pass, not of the encoder, and freezing it would prevent the fine phase
    from adjusting the one quantity the redesign exists to control.

    A helper rather than inline in the loop because it is the kind of thing that is
    silent when broken: a freeze that misses a submodule still trains, and a freeze that
    is never applied still trains. Both are asserted by property in `smoke_test`.
    """
    trunk = getattr(model, "trunk", None)
    n_frozen = 0
    if trunk is not None:
        for p in trunk.parameters():
            if p.requires_grad:
                n_frozen += p.numel()
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    params += trainer.criterion_parameters()
    if extra:
        params += [p for p in extra if p.requires_grad]
    return params, n_frozen


def build_targets(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    img = cfg.get("image", {}) or {}
    fs = img.get("feature_set", "clip_h14_multilevel")
    layers = img.get("layers")
    l2 = img.get("l2norm", True)
    tr = target_mod.load_target_stack(fs, layers, "train", l2norm=l2)
    te = target_mod.load_target_stack(fs, layers, "test", l2norm=l2)
    return tr, te


def build_route_targets(cfg: dict, routes: list[dict],
                        ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-route target stacks, keyed by route NAME, for `objective: v6`.

    Route 0's stack is deliberately NOT the `image` block's stack unless the config says
    so: every route names its own `feature_set` and `layers`, and the route list is the
    single source of truth (it is what `resolve_routes` validated the caches against). A
    route whose stack silently came from the `image` block would train on a geometry its
    own name does not describe.
    """
    l2 = cfg.get("image", {}).get("l2norm", True)
    tr: dict[str, np.ndarray] = {}
    te: dict[str, np.ndarray] = {}
    for spec in routes:
        name = str(spec["name"])
        fs = str(spec["feature_set"])
        tr[name] = target_mod.load_target_stack(fs, list(spec["layers"]), "train",
                                                l2norm=l2)
        te[name] = target_mod.load_target_stack(fs, list(spec["layers"]), "test",
                                                l2norm=l2)
        LOG.info("Stage A | route %-8s feature_set=%-22s layers=%s dim=%d",
                 name, fs, list(spec["layers"]), tr[name].shape[-1])
    return tr, te


def build_loaders(cfg: dict, data: things_eeg.LosoData, targets_tr: np.ndarray,
                  extra_targets: dict[str, np.ndarray] | None = None):
    # `augment` has been a parameter of `LosoTrainDataset` since it was written and was
    # never passed one. Wired here rather than inside the dataset so that "is this run
    # augmented?" is answered by the config dump at the top of every log, not by reading
    # the dataset source. See `data/augment.py` for why the failure mode needs it.
    augment = build_augment(cfg)
    LOG.info("Stage A | augmentation %s",
             getattr(augment, "__name__", "OFF (raw epochs)"))
    # v7 T2': the un-averaged repetitions are loaded ONLY when the cross-trial term is
    # enabled. Loading them unconditionally would add ~19 GB of page-cache pressure to every
    # run -- including the recorded baselines this arm is compared against -- for a tensor
    # nothing reads. Gated on the same `concept.enabled` switch the Trainer reads, so the
    # config dump answers "did this run see repetitions?" for both halves of the wiring.
    tr_reps = None
    if bool((cfg.get("concept") or {}).get("enabled", False)):
        # Source subjects are TRAINING subjects, so the whitener role is `train` -- the same
        # choice `load_loso` makes for their averaged blocks, and it must match or the reps
        # would sit on a different scale than the rows the encoder is trained on.
        reps_mvnn = "train" if cfg.get("mvnn", "off") != "off" else "off"
        tr_reps = [
            things_eeg.load_train_reps(s, data.channels, mvnn=reps_mvnn)
            for s in data.source_subjects
        ]
        LOG.info("Stage A | T2' train repetitions shape=%s R=%d mvnn=%s",
                 tuple(tr_reps[0].shape), tr_reps[0].shape[2], reps_mvnn)
    ds = things_eeg.LosoTrainDataset(data, targets_tr, augment=augment,
                                     seed=cfg.get("seed", 2025),
                                     extra_targets=extra_targets,
                                     train_reps=tr_reps)
    sampler = CrossSubjectBatchSampler(
        n_subjects=data.n_subjects, n_concepts=data.n_concepts, n_images=data.n_images,
        batch_stimuli=cfg.get("batch_stimuli", 8),
        subjects_per_stimulus=cfg.get("subjects_per_stimulus"),
        images_per_pair=cfg.get("images_per_pair", 1),
        drop_last=True, seed=cfg.get("seed", 2025),
    )
    loader = DataLoader(ds, batch_sampler=sampler,
                        num_workers=cfg.get("num_workers", 4),
                        collate_fn=things_eeg.collate, pin_memory=True)
    return loader, sampler


def build_test_loader(cfg: dict, data: things_eeg.LosoData, targets_te: np.ndarray,
                      extra_targets: dict[str, np.ndarray] | None = None):
    ds = things_eeg.TestDataset(data.te_eeg, targets_te, extra_targets=extra_targets)
    return DataLoader(ds, batch_size=cfg.get("eval_batch", 200), shuffle=False,
                      num_workers=0, collate_fn=things_eeg.collate)


# ------------------------------------------------------------------- Stage A
def train_stage_a(cfg: dict) -> dict:
    from .utils import set_seed
    set_seed(cfg.get("seed", 2025))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    target_subject = int(cfg["target_subject"])
    source_subjects = cfg.get("source_subjects") or \
        [s for s in config.all_subjects() if s != target_subject]
    # The 17-vs-63 channel choice is a PROTOCOL decision, not a knob to tune: the
    # inter-subject literature consistently finds all 63 channels better, while
    # intra-subject prefers the 17 occipito-parietal ones (PROTOCOL_INTER.md §1).
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if cfg.get("channel_set", "all63") == "occipital17" else None)

    data = things_eeg.load_loso(source_subjects, target_subject, channels,
                                mvnn=cfg.get("mvnn", "off"))
    objective = str(cfg.get("objective", "v3"))
    routes: list[dict] = []
    extra_tr = extra_te = None
    if objective == "v6":
        from .models.multiroute import resolve_routes

        routes = resolve_routes(cfg)
        extra_tr, extra_te = build_route_targets(cfg, routes)
        # PRIMARY route: route 0, declared by `resolve_routes` to be the one the
        # in-training evaluator and the log read (see `MultiRouteSAMCLIP.primary`).
        primary = str(routes[0]["name"])
        targets_tr, targets_te = extra_tr[primary], extra_te[primary]
        # Drop the primary from the extra stacks: its target already travels under the
        # plain `target` key, and keeping it here would put a second copy of the stack in
        # every DataLoader worker (see the fallback in `_assemble_v6`).
        extra_tr = {n: v for n, v in extra_tr.items() if n != primary}
        extra_te = {n: v for n, v in extra_te.items() if n != primary}
    else:
        targets_tr, targets_te = build_targets(cfg)
    n_layers = targets_tr.shape[2]
    image_dim = targets_tr.shape[-1]

    cfg_eff = dict(cfg)
    cfg_eff["n_subjects"] = data.n_subjects
    cfg_eff["n_channels"] = data.tr_eeg[0].shape[-2]
    cfg_eff["n_timepoints"] = data.tr_eeg[0].shape[-1]
    if objective == "v6":
        from .models.multiroute import MultiRouteSAMCLIP

        model = MultiRouteSAMCLIP(cfg_eff, routes, cfg_eff).to(device)
        LOG.info("Stage A | objective=v6 routes=%s primary=%s",
                 model.route_names, model.primary)
    else:
        model = build_model(cfg_eff, n_layers, image_dim).to(device)

    weights = LossWeights.from_cfg(cfg)
    # Validate BEFORE the data load: a weight that the selected objective never reads is
    # an hour of GPU time to discover, and `Trainer` would only raise after the fold's
    # EEG and target stacks are already resident.
    weights.validate(str(cfg.get("objective", "v3")))
    proto = None
    if weights.proto > 0:
        n_classes = (data.n_concepts if str(cfg.get("prototype_level", "concept"))
                     == "concept" else data.n_concepts * data.n_images)
        # The prototype bank is keyed and sized in the ALIGNMENT space, which on v4 is
        # `d_align` (64), not `d_embed` (512). Passing the wrong width builds a bank whose
        # `proto` buffer cannot be matched against any embedding, and the failure would
        # appear as a silent broadcast error deep inside the contrast.
        proto_d = int(getattr(model, "d_align", cfg.get("d_embed", 512)))
        proto = PrototypeEMA(n_classes, proto_d,
                             init_temp=cfg.get("temp_proto", 0.07),
                             softplus=cfg.get("softplus", True))
    adv = SubjectAdversary(int(getattr(model, "d_align", cfg.get("d_embed", 512))),
                           data.n_subjects).to(device) if weights.adv > 0 else None

    trainer = Trainer(model=model, cfg=cfg, device=device, n_subjects=data.n_subjects,
                      weights=weights, prototype=proto, adversary=adv)

    if objective == "v6":
        # A shared trunk is reachable from N route heads, so `model.parameters()` yields
        # it N times. Handing AdamW duplicates applies the update N times -- an effective
        # N x LR on exactly the shared encoder, which trains happily and just looks like a
        # badly tuned LR. See `dedup_parameters`.
        from .models.multiroute import dedup_parameters

        params = dedup_parameters(model) + trainer.criterion_parameters()
    else:
        params = list(model.parameters()) + trainer.criterion_parameters()
    if adv is not None:
        params += list(adv.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 1e-3),
                            weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = int(cfg.get("epochs", 50))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    schedule = CoarseToFine.from_cfg(cfg)
    # Record the RESOLVED schedule (defaults filled in), not just what the YAML happened
    # to spell out: the checkpoint is the eval stage's source of truth, and a schedule
    # that lived only in the training process would make the run unreproducible.
    cfg_eff["schedule"] = dict(vars(schedule))
    loader, sampler = build_loaders(cfg, data, targets_tr, extra_targets=extra_tr)
    test_loader = build_test_loader(cfg, data, targets_te, extra_targets=extra_te)

    layer_w = model.target_layer_weights()
    LOG.info("Stage A | fold sub-%02d held out | %s params | %d epochs | %d batches/epoch",
             target_subject, human_int(count_parameters(model)), epochs, len(loader))
    LOG.info("Stage A | schedule %s", schedule.describe())
    LOG.info("Stage A | arch=%s objective=%s | d_align=%d d_latent=%s | smn=%s",
             getattr(model, "arch", "?"), trainer.objective,
             int(getattr(model, "d_align", -1)), getattr(model, "d_latent", None),
             "off" if getattr(model, "smn", None) is None else
             f"on (gate_scale={getattr(model.smn, 'gate_scale', None)}, "
             f"init_gate={float(model.smn.gate()):.3f}, "
             f"min_rows={getattr(model.smn, 'min_rows', None)})")
    LOG.info("Stage A | target_fusion=%s%s | loss weights %s",
             cfg.get("target_fusion", "mean"),
             "" if layer_w is None else f" (init weights {[round(w, 3) for w in layer_w]})",
             {k: v for k, v in vars(weights).items()})
    LOG.info("Stage A | prototype_level=%s%s", trainer.proto_level,
             "" if proto is None else f" | {proto.n_classes} prototype slots")

    out_dir = Path(cfg.get("out_dir", config.OUTPUTS / "stage_a"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate BEFORE training. Two reasons, and the first is not cosmetic: with
    # `epochs == 0` this makes the script a legitimate "score the initialisation" run,
    # which is how the pipeline is smoke-tested without burning a GPU-hour. The second
    # is that the untrained number is the floor every later epoch is read against, and
    # reconstructing it afterwards is impossible.
    metrics = evaluate_fold(model, test_loader, device, spec_r0=trainer.spec_r0)
    LOG.info("epoch -1 (init) | test top1 %.2f top5 %.2f meanrank %.1f | "
             "offset %.3f spec_top %.3f", metrics["top1"], metrics["top5"],
             metrics["mean_rank"], metrics["offset_ratio"], metrics["spec_top_frac"])
    best = metrics["top1"]

    # Persist the initialisation when there is no training loop to persist from.
    # Without this `--epochs 0` left no `last.pt`, so the eval job that the DAG chains
    # onto it would fail on a missing file -- i.e. the cheap way to test the pipeline
    # could not test it.
    #
    # Note what `--epochs 0` does and does not cover: it runs the real data path (all 9
    # subjects, the target stack, model construction at production width) and the full
    # 200-way eval, but returns BEFORE the loop -- so no loss is assembled and no
    # prototype is updated. Use `--epochs 1 --debug-steps N` to exercise those too.
    if epochs == 0:
        state = {"model": model.state_dict(), "cfg": cfg_eff, "epoch": -1,
                 "metrics": metrics, "crit": trainer.criterion_state()}
        torch.save(state, out_dir / "last.pt")
        LOG.info("epochs=0: saved the initialisation to %s (epoch -1)", out_dir / "last.pt")
        return {"last": metrics, "best_seen": best, "out_dir": str(out_dir)}

    max_steps = int(cfg.get("debug_steps") or 0)
    stage2_switched = False
    for epoch in range(epochs):
        # The schedule is applied BEFORE the batch loop so the whole epoch -- every
        # logged step and every gradient -- runs at one coherent set of weights. Doing
        # it mid-epoch would make `parts` in the log describe two different objectives.
        phase = ""
        if schedule.enabled:
            coarse, fine, lr_now, phase = schedule.at(epoch)
            trainer.coarse_weight, trainer.fine_weight = coarse, fine
            if trainer.objective == "v3":
                # v3's objective reads ABSOLUTE weights, so the schedule overwrites them
                # here (this is exactly what v3 always did). v4 keeps `loss_weights` as
                # the ratio between terms and multiplies the phase coefficient in
                # `assemble`, so overwriting them there would square the ramp.
                trainer.weights.mmd = coarse
                trainer.weights.cross = fine
            for g in opt.param_groups:
                g["lr"] = lr_now

        # Stage-2 freeze: on the FIRST Stage-2 epoch, freeze the encoder and rebuild the
        # optimiser over what is left at `stage2_lr` (see `freeze_encoder_for_stage2`).
        #
        # `params` is reassigned, and the same list is what `clip_grad_norm_` sees, so a
        # frozen tensor cannot leak back in through the gradient clipping.
        if (schedule.enabled and schedule.freeze_encoder_stage2
                and not stage2_switched and epoch >= schedule.stage1_epochs):
            params, n_frozen = freeze_encoder_for_stage2(
                model, trainer, list(adv.parameters()) if adv is not None else None)
            opt = torch.optim.AdamW(params, lr=schedule.stage2_lr,
                                    weight_decay=cfg.get("weight_decay", 1e-4))
            stage2_switched = True
            LOG.info("Stage 2 begins at epoch %d: froze %s trunk parameters; optimiser "
                     "rebuilt over %s trainable parameters at lr %g", epoch,
                     human_int(n_frozen),
                     human_int(sum(p.numel() for p in params)), schedule.stage2_lr)

        sampler.set_epoch(epoch)
        model.train()
        if adv is not None:
            adv.train()
        meter = AverageMeter()
        for step, batch in enumerate(loader):
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss, parts = trainer.assemble(batch)
            loss.backward()
            # A non-finite loss must be reported HERE, where the culprit is still adjacent.
            # Without this, a NaN propagates into the optimiser and the run dies tens of
            # steps later inside an unrelated diagnostic (observed: "linalg.eigh: the
            # algorithm failed to converge"), which sends you to the wrong file entirely.
            if not torch.isfinite(loss):
                bad = [k for k, v in parts.items() if not torch.isfinite(v)]
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch} step {step}: loss={float(loss)}; "
                    f"non-finite terms={bad}; all terms={ {k: float(v) for k, v in parts.items()} }; "
                    f"subject={batch.get('subject') is not None}")
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.get("grad_clip", 1.0))
            # The gradient must be checked as well as the loss, and this is the check that
            # would actually have caught the recovery bug: there the LOSS was finite at
            # every step and only the GRADIENT was NaN (SVD/eigh backward on a low-rank
            # matrix), so a loss-only guard sees nothing. `clip_grad_norm_` returns the
            # pre-clip total norm, which is NaN exactly when any gradient entry is.
            if not torch.isfinite(gn):
                raise FloatingPointError(
                    f"non-finite gradient at epoch {epoch} step {step}: loss={float(loss)} "
                    f"(finite), grad_norm={float(gn)}; terms={ {k: float(v) for k, v in parts.items()} }")
            opt.step()
            trainer.update_prototype(batch)
            meter.update(float(loss), batch["eeg"].shape[0])
            if step % cfg.get("log_every", 50) == 0:
                LOG.info("epoch %d step %d/%d loss %.4f %s", epoch, step, len(loader),
                         meter.avg, {k: round(float(v), 4) for k, v in parts.items()})
            if max_steps and step + 1 >= max_steps:
                LOG.info("debug_steps=%d reached; ending epoch %d early", max_steps, epoch)
                break
        if not schedule.enabled:
            # A cosine decay over the whole run is the single-stage policy. Under
            # coarse-to-fine the learning rate is set per phase instead (constant within
            # each), so stepping it here would overwrite the phase's rate.
            sched.step()

        metrics = evaluate_fold(model, test_loader, device, spec_r0=trainer.spec_r0)
        lw = model.target_layer_weights()
        gate = model.smn_gate()
        # The recovery episode is a SILENT branch: a run in which it abstains on every step
        # and a run in which it fires on every step produce the SAME loss curve. These
        # counters are the only thing that distinguishes them, and the pre-registered
        # reading for pillar A2 ("the recovery margin should shrink") is meaningless if the
        # episode never ran. Logged per epoch rather than per step to avoid flooding.
        ep = ""
        if getattr(trainer, "ep_enabled", False):
            keep = ("abstained", "reason", "n_mutual", "landmark_rate")
            last = {k: v for k, v in trainer._ep_last_diag.items() if k in keep}
            ep = (f" | episode fires {trainer._ep_fires} "
                  f"abstains {trainer._ep_abstains} last={last}")
        LOG.info("epoch %d%s | lr %.2e | coarse %.3f fine %.3f | loss %.4f | "
                 "test top1 %.2f top5 %.2f meanrank %.1f | offset %.3f spec_top %.3f%s%s%s",
                 epoch, f" [{phase}]" if phase else "", opt.param_groups[0]["lr"],
                 trainer.coarse_weight, trainer.fine_weight, meter.avg,
                 metrics["top1"], metrics["top5"], metrics["mean_rank"],
                 metrics["offset_ratio"], metrics["spec_top_frac"],
                 "" if gate is None else f" | smn_gate {gate:.3f}",
                 "" if lw is None else
                 f" | layer_w {[round(float(x), 3) for x in lw]}", ep)
        state = {"model": model.state_dict(), "cfg": cfg_eff, "epoch": epoch,
                 "metrics": metrics, "crit": trainer.criterion_state()}
        torch.save(state, out_dir / f"epoch{epoch:03d}.pt")
        # last-epoch policy: the final checkpoint is the reported one (no test-based
        # selection). We still keep `best.pt` for diagnostics only.
        torch.save(state, out_dir / "last.pt")
        # Cap the per-epoch files. Without this the loop wrote one ~3.8 MB checkpoint per
        # epoch forever (16 GB accumulated across the ablation arms), and -- the reason it
        # matters more than the space -- a scancelled run left a `last.pt` next to a full
        # ladder of epoch files, so a PARTIAL run could not be distinguished from a
        # finished one without inspecting it by hand. See `prune_epoch_checkpoints`.
        prune_epoch_checkpoints(out_dir, keep_last=int(cfg.get("keep_last_epochs", 2)))
        best = max(best, metrics["top1"])
    LOG.info("Stage A done. last-epoch top1 %.2f | best-seen (diagnostic only) %.2f",
             metrics["top1"], best)
    return {"last": metrics, "best_seen": best, "out_dir": str(out_dir)}


#: Backwards-compatible alias: the entry point used to be called `train_stage1` and
#: older Slurm scripts / notes still refer to it by that name. It is one function, not
#: two -- the v1 "Stage 1" and the v2 "Stage A" are the same training run with a
#: different (conditioning-free) model.
train_stage1 = train_stage_a


@torch.no_grad()
def evaluate_fold(model, loader, device: torch.device, spec_r0: int = 16) -> dict:
    """Raw-cosine retrieval + the two diagnostics the v4 redesign is judged by.

    ``offset_ratio`` is the falsifiable prediction for C1/C2: it is
    ``||per-subject mean|| / mean||row||`` of the PRE-SMN EEG embedding, the quantity the
    v3 checkpoints measured at 0.27-0.42. If the shared head does what §2.5 says it must,
    this drops even before the SMN sees it, and `center QUERY only` stops buying anything.

    ``spec_top_frac`` is the prediction for C3: the share of second-moment energy in the
    top ``spec_r0`` directions. It is read on the POST-SMN, normalised embedding, i.e.
    the one the spectral term actually shapes.
    """
    feats = evaluate.extract_features(model, loader, device)
    rep = evaluate.retrieval_report(feats["eeg"], feats["img"])
    raw = torch.as_tensor(feats["eeg_raw"], dtype=torch.float32, device=device)
    emb = torch.as_tensor(feats["eeg"], dtype=torch.float32, device=device)
    rep["offset_ratio"] = model.subject_offset_ratio(raw)
    rep["spec_top_frac"] = spectrum_report(emb, r0=int(spec_r0))["top_frac"]
    return rep
