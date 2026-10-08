"""SAMCLIP: a shared EEG encoder aligned to a structured multi-layer image target.

Two head topologies live here, selected by ``arch`` (config key, default ``v3``):

``arch: v3`` -- the v3.2 recipe, kept bit-exact so every checkpoint and probe
    already recorded stays loadable::

        eeg (B,C,T) --[trunk]--> --[eeg_head]--> z_e     (L2-normalised)
        target (B,K,D) --[router]--> --[img_head]--> z_i (L2-normalised)

``arch: v4`` -- Subject-as-Modality: one SHARED projection into a low-rank
    alignment space, with the per-subject offset removed inside the forward pass::

        eeg    --[trunk]--> --[eeg_pre]--+
                                         +--[share_head]--> --[SMN]--> z_e
        target --[router]--> --[img_pre]--+

    logits = scale * z_e z_i^T

WHY THE TWO PATHS ARE NOT TWO ARCHITECTURES
-------------------------------------------
v3's ``eeg_head`` and ``img_head`` are SEPARATE maps with independent, unconstrained
bias terms. Nothing in the objective ties them together, and the measured consequence
is a per-subject displacement of the EEG cloud (``||mean(q)||/mean||q|| = 0.27-0.42``)
pointing almost orthogonally to the image cloud's own offset
(``cos(mean_q, mean_g) = +0.10..+0.14``). On a cosine metric that constant unmatched
offset is a per-gallery-item bias -- hubness -- and centring it alone was worth
+6.5/+4.5/+2.5 Top-1 on the three trained checkpoints, more than any coordinate change
(`scripts/probe_signal_weights.py`).

v4 fixes it STRUCTURALLY rather than with a penalty: one shared ``Linear`` maps both
modalities, so a subject-specific bias in the EEG branch cannot appear in the image
branch by construction -- the two clouds are displaced by the same map or not at all.
The reference architecture needs no test-time centring for exactly this reason. The
remaining same-map offset is then removed by :class:`SubjectModalityNorm`.

``d_align`` is deliberately small (64). The subspace probe puts the task concept
manifold at ~16 dimensions (93-96% of query variance, 98% of target variance in the
top 16 PCs; `scripts/probe_subspace_alignment.py`), so a 512-d alignment space leaves
InfoNCE free to separate the training classes in ~496 non-generalising
subject-specific directions. 64 keeps the bulk below that regime while leaving the
head enough room to reweight the 16 informative directions.

Three things changed from v1 and each one is a deliberate removal, not a rewrite:

  * NO SUBJECT CONDITIONING. There is no ``z_s``, no FiLM, no LoRA, no support-set
    forward pass, and no ``CondParams``. The encoder is one shared function of the EEG
    -- for training subjects and for an unseen one alike. The evidence for removing it
    is recorded in ``models/subject_conditioning.py`` and in the plan doc §2.
  * NO FREEZING MACHINERY. ``freeze_shared`` / ``frozen_state`` / ``frozen_drift``
    existed to protect a Stage-1 geometry while Stage 2 trained a hypernetwork. There
    is no Stage 2 and no hypernetwork, so they are gone. The BatchNorm lesson they
    encoded is preserved in the note at the top of ``models/backbone.py``.
  * NO SEPARATE MAPPING STAGE. The EEG head lands directly in the image-target space
    the retrieval metrics are computed in. The retrieval-side "mapping" that v1
    delegated to a trained network is done by label-free test-time geometry
    (``samclip.calibration``), which the evidence says is both cheaper and stronger
    (SCORE: 26.22 -> 53.23 Top-1 on a frozen encoder, no target labels).

What is kept, because it is measured to matter more than the encoder:

  * ``LayerRouter`` -- the structured multi-layer visual target. Supervising EEG with
    several learned visual views rather than one global image embedding is worth
    ~5.7 Top-1 points in the published comparison (29.6 -> 35.3), which is larger
    than the difference between most EEG backbones. See the class docstring.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import EEGTrunk
from .smn import SubjectModalityNorm, subject_offset_ratio


ARCHITECTURES = ("v3", "v4")

#: Variance attenuation of GELU on a standard-normal pre-activation: ``E[gelu(z)^2]`` for
#: ``z ~ N(0, 1)``, measured at 0.425014 over 4e6 samples. A linear layer with orthogonal
#: weights followed by GELU therefore shrinks the signal to ~sqrt(0.425) = 65% of its
#: norm, so the layer AFTER it is initialised with gain ``1/sqrt(0.425)`` to put the head
#: back near-isometric. ((`nn.init.calculate_gain("relu")` = 1.414 is the usual stand-in;
#: 1.534 is the GELU value and costs nothing to use.))
GELU_PRESERVE_GAIN = 1.533905


def _mlp(d_in: int, d_hidden: int, d_out: int, n_layers: int = 1,
         drop: float = 0.1) -> nn.Sequential:
    mods: list[nn.Module] = [nn.Linear(d_in, d_hidden), nn.GELU()]
    for _ in range(max(0, n_layers - 1)):
        mods += [nn.Dropout(drop), nn.Linear(d_hidden, d_hidden), nn.GELU()]
    mods += [nn.Linear(d_hidden, d_out)]
    return nn.Sequential(*mods)


def _head(d_in: int, d_hidden: int, d_out: int, mode: str = "mlp",
          n_layers: int = 1, drop: float = 0.1) -> nn.Module:
    """Projection head, either a single linear map or the hidden-layer MLP.

    ``linear`` exists because of a measured parameter-count inversion, not as a style
    choice. At the default geometry the MLP image head is ``Linear(1280,1280) +
    Linear(1280,512) = 2.30M`` parameters -- **2.27x the entire EEG encoder**
    (trunk 869k + eeg_head 143k = 1.01M), i.e. 69% of a 3.31M model is spent mapping a
    FROZEN target. Two consequences follow, both bad:

      * the head is applied to targets that never move, so at initialisation it is a
        random 1280->1280 map that *destroys* the already-good CLIP geometry the EEG is
        supposed to align to, and the first part of training is spent learning a
        projection of a representation that was fine as it was;
      * capacity that should buy cross-subject invariance in the encoder is instead
        spent on the one component that cannot improve generalisation -- it only
        reparametrises a frozen input.

    A single ``Linear(1280, d_embed)`` is 0.66M at ``d_embed=512`` (0.33M at 256) and is
    an orthogonal-ish change of basis, so it starts near-isometric. This is the
    hypothesis that the loss curve and retrieval should decide (see plan doc §2.5); it
    is a config key so both arms can be run rather than argued about.
    """
    if mode == "linear":
        return nn.Linear(d_in, d_out)
    if mode == "mlp":
        return _mlp(d_in, d_hidden, d_out, n_layers=n_layers, drop=drop)
    raise ValueError(f"img_head/eeg_head mode must be 'linear' or 'mlp', got {mode!r}")


class LayerRouter(nn.Module):
    """Blend K image-target layers into one vector.

    ``mean``      -- fixed uniform weights. The "uniform first" arm: it answers "do
                     these layers carry complementary information at all?" BEFORE the
                     question "can a router combine them better than equally?". Per
                     nothing is learned here, so a loss to ``mean`` is evidence about
                     the target stack, not about routing capacity.
    ``routed``    -- a learned global weight vector over the K layers.
    ``routed_sr`` -- global weights + a per-subject residual, **dropped at inference**.

    Why ``routed_sr`` drops its residual at inference (this is the SAMGA mechanism,
    and it is the part worth keeping in mind): conditioning the *target* on the row's
    subject during training lets the target absorb subject-dependent granularity
    differences, while inference needs no subject identity at all because the deployed
    blend is the global prior. It is an ablation arm by default rather than the
    shipped setting for a measured reason: making the training target a function of
    the row's subject turns the alignment objective into an incentive to encode
    subject identity, which directly opposes the decorrelation terms. Measured on this
    pipeline it drove the subject-dependence term UP (0.22 -> 0.48 in the first
    epochs) instead of down. That was under v1's objective; under v2's (which no
    longer contains a strong decorrelation pressure) the trade-off is different, so it
    is registered as a live ablation rather than a rejected option.
    """

    def __init__(
        self,
        n_layers: int,
        d_in: int,
        mode: str = "mean",
        n_subjects: int = 1,
        prior_strength: float = 1.0,
        subject_dropout: float = 0.3,
        temperature: float = 1.0,
        layer_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in ("mean", "routed", "routed_sr"):
            raise ValueError(f"target_fusion must be mean/routed/routed_sr, got {mode!r}")
        if temperature <= 0:
            raise ValueError(f"router temperature must be > 0, got {temperature}")
        if not 0.0 <= layer_dropout < 1.0:
            raise ValueError(f"router layer_dropout must be in [0,1), got {layer_dropout}")
        if mode == "routed_sr" and n_layers < 2:
            raise ValueError(
                "target_fusion='routed_sr' needs >=2 target layers: a per-subject "
                "residual over a single target is constant after softmax, so it would "
                "train no parameter and claim a mechanism it does not have")
        self.mode = mode
        self.n_layers = n_layers
        self.subject_dropout = float(subject_dropout)
        self.prior_strength = float(prior_strength)
        # --- anti-collapse controls on the learned blend -------------------------
        # `temperature` divides the logits before the softmax: >1 flattens the
        # distribution toward uniform, which is the direction that keeps all K layers
        # alive. `layer_dropout` zeroes a random subset of the per-row weights and
        # renormalises, so no single layer can become the only one the loss ever sees.
        # Both are borrowed from SAMGA's released recipe (`router_temperature`,
        # `router_layer_dropout`), where the deployed weights are diffuse
        # ([0.07, 0.18, 0.49, 0.19, 0.07] over layers 20/24/28/32/36) rather than
        # collapsed onto one layer. That contrast is the reason these knobs exist.
        self.temperature = float(temperature)
        self.layer_dropout = float(layer_dropout)
        #: Weights from the most recent TRAINING forward pass, kept so the Trainer can
        #: add an entropy bonus without `forward` changing its return type. Cleared at
        #: the top of every forward so a stale tensor can never be read as the current
        #: one (evaluation runs would otherwise leave the last training weights behind).
        self._last_weights: torch.Tensor | None = None
        if mode in ("routed", "routed_sr"):
            self.w = nn.Parameter(torch.zeros(n_layers))
        else:
            self.w = None  # type: ignore[assignment]
        self.residual = (nn.Embedding(n_subjects, n_layers) if mode == "routed_sr" else None)
        if self.residual is not None:
            nn.init.zeros_(self.residual.weight)

    def layer_weights(self) -> torch.Tensor | None:
        """The deployed layer weights, for the run record."""
        if self.w is None:
            return None
        return torch.softmax(self.w.detach() / self.temperature, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        training: bool = False,
        require_subject_ids: bool = True,
    ) -> torch.Tensor:
        """``x``: ``(B, K, D)`` -> ``(B, D)``.

        `require_subject_ids` exists because "no subject ids" has two meanings that
        must not be conflated. During Stage A training it means the training loop
        forgot to wire the ids through, and the per-subject residual would silently
        never train -- a run that reports a mechanism it is not using. At evaluation
        the batch is deliberately subject-agnostic and the global blend is exactly the
        deployed behaviour. The caller knows which of the two it is.
        """
        self._last_weights = None
        if x.dim() == 2:
            return x
        if x.shape[1] == 1:
            return x[:, 0]
        if self.mode == "mean":
            return x.mean(dim=1)

        logits = self.w.unsqueeze(0).expand(x.shape[0], -1) / self.temperature
        if self.mode == "routed_sr":
            if training and subject_ids is not None:
                res = self.residual(subject_ids)                # (B, K)
                if self.subject_dropout > 0:
                    keep = (torch.rand(x.shape[0], 1, device=x.device)
                            >= self.subject_dropout).to(res.dtype)
                    res = res * keep
                logits = logits + self.prior_strength * res
            elif training and require_subject_ids:
                raise RuntimeError(
                    "target_fusion='routed_sr' is training without subject_ids, so the "
                    "subject residual would never receive a gradient and the run would "
                    "report a mechanism it is not using. Pass the batch's subject ids, "
                    "or mark the call as deliberately subject-agnostic via "
                    "require_subject_ids=False")
            # otherwise: residual dropped, so the blend is subject-agnostic
        w = torch.softmax(logits, dim=-1)                        # (B, K)
        # The entropy bonus is computed on the CLEAN softmax, not on the dropout-perturbed
        # one. Two reasons, and the first is why it matters: at initialisation every logit
        # is 0, so the clean distribution is exactly uniform, its entropy is already
        # maximal (ln K), and the bonus contributes ZERO gradient -- the router is free to
        # specialise, and the bonus only starts resisting once it tries to collapse toward
        # a corner. Computing it after `layer_dropout` would give it a non-zero gradient at
        # step 1 (the perturbation is below maximal entropy), i.e. it would fight the
        # dropout rather than the collapse. The dropout stays a data-level regulariser on
        # the blend; the learned parameter is regularised cleanly.
        w_used = w
        if training and self.layer_dropout > 0:
            keep = (torch.rand(w.shape, device=w.device)
                    >= self.layer_dropout).to(w.dtype)
            w_used = w * keep
            denom = w_used.sum(dim=-1, keepdim=True)
            # A row whose every layer was dropped has no weight left; falling back to
            # uniform is the only choice that keeps the blend a convex combination
            # (leaving it at 0 would silently return a ZERO target for that row).
            uniform = torch.full_like(w_used, 1.0 / w_used.shape[-1])
            w_used = torch.where(denom > 0, w_used / denom.clamp_min(1e-12), uniform)
        if training:
            # Stashed only while training: the Trainer reads it to add the entropy bonus,
            # and an inference pass must not be able to leave training weights behind for
            # the next training step to pick up.
            self._last_weights = w
        return (x * w_used.unsqueeze(-1)).sum(dim=1)

    def entropy_penalty(self) -> torch.Tensor | None:
        """``-H(w)`` over the last training forward's layer weights, or None.

        Returned as a MINIMISATION target (negative entropy), so the Trainer adds it
        with a positive weight. Its purpose is the one failure this router has already
        demonstrated on this project: the learned blend collapsed onto the shallowest
        layer (weights ~[0.74, 0.13, 0.04, 0.01, 0.07] at epoch 59), i.e. it spent a
        five-layer target down to essentially one layer, which removes exactly the
        multi-granularity the target stack exists to provide. `mean` avoids that only
        by refusing to learn anything at all. Maximising the entropy is the third
        option: keep *some* learned reweighting while refusing a one-hot blend.
        """
        w = self._last_weights
        if w is None:
            return None
        return (w * torch.log(w.clamp_min(1e-9))).sum(dim=-1).mean()


class SAMCLIP(nn.Module):
    """Shared EEG encoder + structured image target + CLIP-style alignment heads.

    ``arch='v3'`` reproduces the previous recipe exactly (separate heads, no SMN).
    ``arch='v4'`` is the Subject-as-Modality topology: shared head + low-rank
    alignment + in-forward subject centring. See the module docstring.
    """

    def __init__(
        self,
        n_channels: int = 63,
        n_timepoints: int = 250,
        n_target_layers: int = 1,
        image_dim: int = 1280,
        n_subjects: int = 9,
        d_model: int = 200,
        d_embed: int = 512,
        n_heads: int = 4,
        n_blocks: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        target_fusion: str = "mean",
        target_subject_dropout: float = 0.3,
        router_temperature: float = 1.0,
        router_layer_dropout: float = 0.0,
        img_head_mode: str = "mlp",
        eeg_head_mode: str = "mlp",
        agg_width: int = 16,
        agg_pool: int = 8,
        front_end: str = "linear",
        front_pool: int = 5,
        gqf: bool = False,
        arch: str = "v3",
        d_latent: int | None = None,
        d_align: int | None = None,
        share_head_hidden: int | None = None,
        smn_enabled: bool = True,
        smn_gate_scale: bool = True,
        smn_init_gate: float = 0.0,
        smn_min_rows: int = 4,
    ) -> None:
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"arch must be one of {ARCHITECTURES}, got {arch!r}")
        self.arch = arch
        self.d_embed = int(d_embed)
        #: GQF (group-quotient front-end). Stored as a plain attribute rather than a module:
        #: it has no parameters, and registering it would add a `state_dict` key that no
        #: banked checkpoint has, which would make every recorded run fail to load strictly.
        #: `None` = off; a string names the mode ("whiten" / "rowspace"). `True` is a
        #: shorthand for the deployed "whiten".
        self.gqf = None if not gqf else (gqf if isinstance(gqf, str) else "whiten")
        self.trunk = EEGTrunk(
            n_channels=n_channels, n_timepoints=n_timepoints, d_model=d_model,
            n_heads=n_heads, n_blocks=n_blocks, dim_ff=dim_ff, dropout=dropout,
            agg_width=agg_width, agg_pool=agg_pool,
            front_end=front_end, front_pool=front_pool,
        )
        self.target_router = LayerRouter(
            n_layers=n_target_layers, d_in=image_dim, mode=target_fusion,
            n_subjects=n_subjects, subject_dropout=target_subject_dropout,
            temperature=router_temperature, layer_dropout=router_layer_dropout,
        )
        if arch == "v3":
            self.eeg_head = _head(self.trunk.out_dim, self.trunk.out_dim, d_embed,
                                  mode=eeg_head_mode, n_layers=1, drop=head_dropout)
            self.img_head = _head(image_dim, image_dim, d_embed,
                                  mode=img_head_mode, n_layers=1, drop=head_dropout)
            # `smn` stays UNSET (not a disabled module) so a v3 checkpoint's state_dict
            # keeps exactly the key set it had when it was written: registering an
            # unused `gate_raw` would make every recorded v3 checkpoint fail to load
            # strictly, which would orphan the results that justify v4 in the first
            # place. `apply_smn` below is the single place that has to know.
            self.smn = None  # type: ignore[assignment]
            self.d_align = int(d_embed)
        else:
            dl = int(d_latent or d_align or 64)
            da = int(d_align or dl)
            if dl < 1 or da < 1:
                raise ValueError(f"d_latent/d_align must be >= 1, got {dl}/{da}")
            # --- modality-private pre-projections: capacity lives HERE, per modality,
            # so the shared map can stay a single linear layer (C1).
            self.eeg_pre = _head(self.trunk.out_dim, self.trunk.out_dim, dl,
                                 mode=eeg_head_mode, n_layers=1, drop=head_dropout)
            self.img_pre = _head(image_dim, image_dim, dl,
                                 mode=img_head_mode, n_layers=1, drop=head_dropout)
            # --- C1: ONE map applied to both modalities. A per-subject bias injected by
            # the EEG branch is therefore also applied to the image branch, so it lands
            # in the SAME direction in both clouds and cancels instead of displacing one
            # against the other (which is what the measured 0.27-0.42 offset does).
            #
            # THE HEAD MUST BE NONLINEAR OR IT CONSTRAINS NOTHING. This is not a
            # preference -- with a linear head the sharing is an exact
            # reparameterisation and C1 is inert. `img_pre` is linear (`img_head:
            # linear`) and `share_head` was linear, so their composition is ONE linear
            # map, and the image side has far more parameters to place it with
            # (768x64 = 49k) than the shared map being imposed has rows (64x64 = 4k).
            # Concretely: to hold `share_head` at whatever the EEG branch wants while
            # realising an ARBITRARY effective image map M, set img_pre = W^+ M, which
            # exists whenever W has full row rank. Measured: a frozen random W plus a
            # fitted `img_pre` reproduces a target PCA map to 8.3e-07 RELATIVE residual,
            # i.e. exactly, while the un-shared model has the same reachable function
            # class. So a linear `share_head` cannot be credited with removing D1 -- the
            # SMN would be doing all of it.
            #
            # Inserting a nonlinearity INTO the shared map fixes that: the image branch
            # is then forced through `GELU` at the same place the EEG branch is, and a
            # private LINEAR `img_pre` provably cannot absorb it (the same fit leaves
            # 0.75 relative residual). `share_head_hidden=0` restores the single linear
            # layer as an explicit ablation arm -- and it is an arm, not a default.
            #
            # The LAST op stays linear on purpose: `z_eeg_raw` is the D1 diagnostic and
            # the input to `L_spec`/VICReg, and a trailing LayerNorm would rescale each
            # row by its own statistics, which would blur exactly the per-subject
            # offset that diagnostic exists to measure.
            dh = (2 * da) if share_head_hidden is None else int(share_head_hidden)
            if dh > 0:
                self.share_head = nn.Sequential(
                    nn.Linear(dl, dh), nn.GELU(), nn.Linear(dh, da))
            else:
                self.share_head = nn.Linear(dl, da)
            # Orthogonal at init: a Square-ish map with kaiming-uniform weights is a
            # random contraction that mixes the informative directions with the
            # uninformative ones BEFORE training starts, and the loss then has to undo
            # that mixing. Orthogonal keeps the alignment space near-isometric at step
            # 0, with the second layer's gain compensating GELU's variance attenuation.
            if dh > 0:
                nn.init.orthogonal_(self.share_head[0].weight)
                nn.init.zeros_(self.share_head[0].bias)
                nn.init.orthogonal_(self.share_head[2].weight, gain=GELU_PRESERVE_GAIN)
                nn.init.zeros_(self.share_head[2].bias)
            else:
                nn.init.orthogonal_(self.share_head.weight)
                nn.init.zeros_(self.share_head.bias)
            self.smn = SubjectModalityNorm(
                da, enabled=smn_enabled, gate_scale=smn_gate_scale,
                init_gate=smn_init_gate, min_rows=smn_min_rows,
            )
            self.d_latent = dl
            self.d_align = da
            self.share_head_hidden = dh

    # ----------------------------------------------------------------- sharing
    @property
    def share_head_linears(self) -> list[nn.Linear]:
        """The ``nn.Linear`` layers inside `share_head`, in forward order.

        Exposed because the head is a `Sequential` on the default path and a bare
        `Linear` on the ablation path, and every reader that wants "the shared weights"
        (the smoke test, the run record, the non-absorbability check) must work on both
        without an `isinstance` chain at each call site.
        """
        if isinstance(self.share_head, nn.Linear):
            return [self.share_head]
        return [m for m in self.share_head if isinstance(m, nn.Linear)]

    @property
    def share_head_is_nonlinear(self) -> bool:
        """True when the shared map genuinely constrains the hypothesis class.

        False means `share_head` is an exact reparameterisation that a private linear
        `img_pre` can absorb, so C1 does nothing and the SMN is carrying D1 alone --
        see the construction comment above. This is a property rather than a comment
        because it is the difference between a mechanism and a decoration.
        """
        return self.arch == "v4" and len(self.share_head_linears) > 1

    # ---------------------------------------------------------------- encoding
    def embed_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` -> ``(B, d_align)`` BEFORE the SMN and before normalisation.

        This is the view the regularisers and the offset diagnostic operate on: an
        unnormalised, un-centred embedding is the only one for which "how big is the
        per-subject mean relative to a row" is a meaningful question.
        """
        return self.embed_from_h(
            self.trunk(group_quotient(eeg, self.gqf) if self.gqf else eeg))

    def embed_from_h(self, h: torch.Tensor) -> torch.Tensor:
        """``(B, trunk.out_dim)`` -> ``(B, d_align)``, the modality heads only.

        Split out for the v6 multi-route model: every route shares ONE trunk pass, so the
        route-specific part must be callable on a precomputed trunk output. Calling
        `embed_eeg` per route instead would run the trunk N times on identical input --
        pure waste, and it would also make the routes' gradients flow through separate
        trunk calls, which is not the shared-encoder premise (§2.3).
        """
        if self.arch == "v3":
            return self.eeg_head(h)
        return self.share_head(self.eeg_pre(h))

    def apply_smn(self, z: torch.Tensor,
                  subject_ids: torch.Tensor | None = None,
                  detach_gate: bool = False) -> torch.Tensor:
        """No-op for ``v3``, per-subject centring for ``v4`` (see `models/smn.py`).

        ``detach_gate`` is forwarded to the SMN so an auxiliary objective can shape the
        encoder without moving the deployment-critical scale gate; see the note there.
        """
        if self.smn is None:
            return z
        return self.smn(z, subject_ids, detach_gate=detach_gate)

    def encode_eeg(self, eeg: torch.Tensor, normalize: bool = True,
                   subject_ids: torch.Tensor | None = None,
                   detach_gate: bool = False) -> torch.Tensor:
        """``(B, C, T)`` -> ``(B, d_align)``. One shared function for every subject.

        ``subject_ids=None`` means "one subject" and is the deployment call: the batch
        IS that subject's query set. Passing ids during training groups them per
        subject; the two are the same operation, which is why the training-time
        estimate is a matched proxy for the test-time one rather than a leak.
        """
        out = self.apply_smn(self.embed_eeg(eeg), subject_ids, detach_gate=detach_gate)
        return F.normalize(out, dim=-1) if normalize else out

    def encode_target(
        self,
        target: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        training: bool = False,
        normalize: bool = True,
        require_subject_ids: bool = True,
    ) -> torch.Tensor:
        """``(B, K, D)`` -> ``(B, d_align)``. At inference the blend is subject-agnostic."""
        fused = self.target_router(target, subject_ids=subject_ids, training=training,
                                   require_subject_ids=require_subject_ids)
        out = self.img_head(fused) if self.arch == "v3" else \
            self.share_head(self.img_pre(fused))
        return F.normalize(out, dim=-1) if normalize else out

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        eeg: torch.Tensor,
        target: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        training: bool = True,
    ) -> dict[str, Any]:
        # ONE encoder pass, and three views of it are returned. `z_eeg` is what the
        # contrastive terms use; `z_eeg_raw` is the pre-SMN, pre-normalisation embedding
        # the spectral-concentration term and the offset diagnostic need. Deriving the
        # raw view from a second `embed_eeg` call would double the encoder cost for no
        # reason, and the two views must come from the SAME forward pass or the
        # regulariser and the contrast would be shaping two different representations.
        raw = self.embed_eeg(eeg)
        z_e = F.normalize(self.apply_smn(raw, subject_ids), dim=-1)
        z_i = self.encode_target(target, subject_ids, training)
        return {"z_eeg": z_e, "z_img": z_i, "z_eeg_raw": raw}

    # ------------------------------------------------------------- utilities
    def target_layer_weights(self) -> list[float] | None:
        """Deployed target-fusion weights, for logging. None for uniform fusion."""
        w = self.target_router.layer_weights()
        return None if w is None else [float(x) for x in w.tolist()]

    def router_entropy(self) -> torch.Tensor | None:
        """``-H`` of the last training blend, for the Trainer's entropy bonus."""
        return self.target_router.entropy_penalty()

    def smn_gate(self) -> float | None:
        """The scale exponent's current value, or None on the v3 path. 0 == centring only."""
        return None if self.smn is None else float(self.smn.gate())

    def subject_offset_ratio(self, z: torch.Tensor,
                             subject_ids: torch.Tensor | None = None) -> float:
        """``||per-subject mean|| / mean||row||`` of ``z`` (arch-agnostic)."""
        return subject_offset_ratio(z, subject_ids)


def group_quotient(x: torch.Tensor, mode: str = "whiten", eps: float = 1e-4) -> torch.Tensor:
    """GQF: quotient a trial by the volume-conduction group `GL(C)`, shape-preserving.

    `x` is `(B, C, T)`. Volume conduction is `x -> M x` with `M in GL(C)` -- an invertible
    linear recombination of the channels -- because mixing is the physical model of how one
    subject's montage relates to another's.

    `GL(C)` factors as `O(C) x PD(C)` (polar decomposition): a re-referencing ROTATION and
    a positive-definite STRETCH (per-channel gain plus channel-correlation reshaping). Two
    modes, and the choice is a statement about which part is nuisance:

    ``whiten`` (default, and the one that is deployed)
        `W = (x x^T + eps*I)^{-1/2} x`, shape `(B, C, T)`. This quotients the `PD(C)` factor
        EXACTLY: for any `M`, `W(Mx) = O W(x)` with `O` orthogonal, so `W` fixes the
        amplitude/stretch content that the group inflates. `PD(C)` is exactly the subgroup
        the training augmentation's `gain` direction covers, so this makes the front-end and
        the augmentation the same statement on two axes.
        TRUTH IN LABELLING: the residual `O(C)` is NOT quotiented -- `W` is canonical only
        up to a channel rotation. That residual is the re-referencing rotation, and it is
        the same object the augmentation distribution draws from, so it is covered on the
        training side rather than here. An exactly-`GL(C)`-invariant, shape-preserving
        tensor does not exist: the invariant of the action is the ROW SPACE, whose only
        canonical representative is the `(T, T)` projector `x^T (x x^T)^{-1} x`, which
        cannot be fed to a `(C, T)` trunk.

    ``rowspace``
        `Vh` of the SVD -- the right-singular subspace. This is the row space's own basis and
        is the closest thing to the exact invariant that fits the shape. It is NOT the
        default because singular values of EEG are near-degenerate, so within a degenerate
        cluster the basis is only defined up to `O(r)` and a per-row sign fix cannot pin it
        (measured: `max|Vh(Mx) - Vh(x)| = 0.41` at `||M-I||_F = 3.15`). It is kept as an
        option for the ablation that shows exactly this, rather than being silently wrong.
    """
    if mode == "rowspace":
        vh = torch.linalg.svd(x, full_matrices=False).Vh          # (B, C, T)
        idx = vh.abs().argmax(dim=-1, keepdim=True)
        sign = torch.sign(torch.gather(vh, -1, idx))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return vh * sign
    if mode != "whiten":
        raise ValueError(f"gqf mode must be 'whiten' or 'rowspace', got {mode!r}")
    c = x @ x.transpose(-1, -2)                                   # (B, C, C) row covariance
    scale = c.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)   # (B, 1)
    c = c + eps * scale.unsqueeze(-1) * torch.eye(
        c.shape[-1], device=x.device, dtype=x.dtype)
    # SOLVER. Both branches compute a valid `C^{-1/2}`, and that is the only thing the
    # quotient needs: for ANY `W` with `W W^T = I`, `W(Mx) = A W(x)` with `A` orthogonal
    # (proof: `A = (M C M^T)^{-1/2} M C^{1/2}` has `A A^T = I`), so the `GL(C) -> O(C)`
    # reduction and the `O(C)`-invariant `W^T W` are the same for every solver.
    #
    # `chol` is the DEFAULT because `eigh` is ~13x slower in training: its BACKWARD is
    # O(C^6) (each matrix needs a C^2 x C^2 Sylvester solve, 3969^2 here) and cuSOLVER's
    # batched symmetric eigendecomposition is a synchronisation point. `eigh` is kept for
    # the record because it is the symmetric representative, but it is not worth 13x.
    try:
        L = torch.linalg.cholesky(c)
    except Exception:
        # A Cholesky failure here means the regularisation was too small for a rank-
        # deficient trial; fall back to the (slower but always-defined) symmetric form.
        w, q = torch.linalg.eigh(c)
        w = w.clamp_min(eps * scale.abs().clamp_min(1e-12))
        return (((q * w.rsqrt().unsqueeze(-2)) @ q.transpose(-1, -2)) @ x)
    return torch.linalg.solve_triangular(L, x, upper=False)


def build_model(cfg: dict, n_target_layers: int, image_dim: int) -> SAMCLIP:
    """Single source of truth for the constructor call (train and eval must agree).

    Reads ``arch`` and the ``smn`` block with defaults that reproduce v3, so a config or
    checkpoint written before v4 existed builds the same model it did then.
    """
    smn_cfg = cfg.get("smn", {}) or {}
    return SAMCLIP(
        n_channels=cfg.get("n_channels", 63),
        n_timepoints=cfg.get("n_timepoints", 250),
        n_target_layers=n_target_layers,
        image_dim=image_dim,
        n_subjects=cfg.get("n_subjects", 9),
        d_model=cfg.get("d_model", 200),
        d_embed=cfg.get("d_embed", 512),
        n_heads=cfg.get("n_heads", 4),
        n_blocks=cfg.get("n_blocks", 2),
        dim_ff=cfg.get("dim_ff", 512),
        dropout=cfg.get("dropout", 0.1),
        head_dropout=cfg.get("head_dropout", 0.1),
        target_fusion=cfg.get("target_fusion", "mean"),
        target_subject_dropout=cfg.get("target_subject_dropout", 0.3),
        router_temperature=cfg.get("router_temperature", 1.0),
        router_layer_dropout=cfg.get("router_layer_dropout", 0.0),
        img_head_mode=cfg.get("img_head", "mlp"),
        eeg_head_mode=cfg.get("eeg_head", "mlp"),
        agg_width=cfg.get("agg_width", 16),
        agg_pool=cfg.get("agg_pool", 8),
        front_end=cfg.get("front_end", "linear"),
        front_pool=cfg.get("front_pool", 5),
        arch=cfg.get("arch", "v3"),
        gqf=cfg.get("gqf", False),
        d_latent=cfg.get("d_latent"),
        d_align=cfg.get("d_align"),
        share_head_hidden=cfg.get("share_head_hidden"),
        smn_enabled=smn_cfg.get("enabled", True),
        smn_gate_scale=smn_cfg.get("gate_scale", True),
        smn_init_gate=smn_cfg.get("init_gate", 0.0),
        smn_min_rows=smn_cfg.get("min_rows", 4),
    )
