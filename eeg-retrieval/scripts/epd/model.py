"""The retrieval model: pretrained ViT as EEG encoder -> layer fusion -> shared space."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import EEGiTProjectionHead, LayerFusion, build_encoder


def _mlp(d_in: int, d_hidden: int, d_out: int, n_layers: int, drop: float) -> nn.Sequential:
    mods: list[nn.Module] = [nn.Linear(d_in, d_hidden), nn.GELU()]
    for _ in range(max(0, n_layers - 1)):
        mods += [nn.Dropout(drop), nn.Linear(d_hidden, d_hidden), nn.GELU()]
    mods += [nn.Linear(d_hidden, d_out)]
    return nn.Sequential(*mods)


def _up_block(d_in: int, d_out: int) -> nn.Sequential:
    """x2 nearest-neighbour upsample then a conv. No transposed conv: at 8x8 -> 64x64
    a strided deconv on this little data is a reliable source of checkerboard
    artefacts, and the extra parameters buy nothing the following conv does not."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(d_in, d_out, 3, padding=1),
        nn.GroupNorm(min(8, d_out), d_out),
        nn.GELU(),
    )


class StructureTower(nn.Module):
    """The second tower: EEG scalp topography -> a spatial latent field, not a joint vector.

    Why this is a separate tower and not a second head on the semantic tower
    -----------------------------------------------------------------------
    The two targets are different *kinds* of object. The semantic target is a
    1024-d vector in CLIP's projected joint space: direction carries the signal,
    magnitude is free, and the loss is a competition between instances. The
    structural target is a 4x64x64 tensor whose every coordinate has a fixed
    meaning in the VAE's decoder basis; there is no notion of "similar images"
    being close, only of being right.

    Those two objectives want opposite things from the trunk. A shared trunk would
    have to hold a basis that is simultaneously rotation-free (cosine needs a
    direction, not a metric) and metrically calibrated (MSE/L1 needs the metric).

    This tower is now a *fully independent* trunk, not a second head: it owns its
    own backbone weights, its own input geometry, and its own decoder. That is the
    stronger form of the argument above rather than a new claim -- two trunks
    initialised from *different* pretrained checkpoints (the semantic one from
    EEGiT's `in21k`, this one from DINOv3 or MAE) cannot share a forward pass at
    all, because from step 0 there is no parameter they have in common.

    Two priors, chosen for what each one is about
    --------------------------------------------
    The design question EEGiT answers is "can a pretrained vision model's prior be
    transferred into an EEG encoder by presenting EEG as image-like patches?" Their
    ablation prices it at +16.4 intra-subject Top-1, the single largest term. This
    project's answer for the *structural* branch is to ask the same question of a
    prior that is about *where things are* instead of *what things are*:

      * `dinov3_b16` -- self-supervised with Gram anchoring, introduced precisely
        because dense metrics decay while global ones improve; it pins patch-to-patch
        similarity, i.e. exactly "structure". DINOv3 ViT-L reaches depth RMSE 0.352
        and 54.9 mIoU on frozen-backbone dense probes.
      * `mae_b16` -- trained by reconstructing masked pixels, so it retains the most
        recoverable *appearance* (LPIPS 0.11 / recon-FID 0.16 against DINOv2-B's
        0.255 / 0.49). The control arm.

    The structural target is *not* either of those backbones' feature space, and
    that is a deliberate split rather than an oversight. The probe that decided it
    (`probe_targets.py`, sub-08 and sub-10, 150-concept val split) measured linear
    decodability of each candidate target from EEG:

        clip_pooled   14.13 val / 26.50 test      <- the semantic target
        vae            5.07 val /  6.00 test      <- clears its floor, margin +0.06
        depth          2.80 val /  2.50 test      <- margin -0.37, BELOW the constant

    Chance is 0.667%. Depth is the space DINOv3 is *best* at, and it is the one
    space EEG cannot decode: a constant map correlates +0.53 with a real depth map
    while EEG's best linear read-out reaches +0.16, on four independent arms. So the
    structural target stays the VAE latent, and the pretrained backbone is used for
    what EEGiT's ablation says it is worth -- an *initialisation prior*, not a target.

    Head geometry, and why it changed
    ---------------------------------
    The token grid this tower receives is a genuine 2D scalp map (see
    `ScalpTopographyTokenizer`), so the decoder starts from it directly instead of
    from a learned 8x8 field. The previous justification for the learned field was
    that "the token grid is (n_regions, n_time_patches) ... with no relation to image
    rows and columns, and warping it into 64x64 would bake in an arbitrary
    correspondence" -- true for the region-band geometry, and no longer true here.

    What the decoder does with the grid:
      1. reshape tokens to (B, D, Hp, Wp) and mix them with a 3x3 conv;
      2. add the *pooled* feature, broadcast to every location. This is the division
         of labour the two readouts make possible: the pooled vector carries "what",
         the token grid carries "where", and the grid alone would have to infer the
         global content from local tiles;
      3. upsample progressively to 64x64 and emit 4 channels.

    `nn.Upsample(nearest)` + conv rather than a transposed conv, for the reason the
    earlier head already recorded: at these scales a strided deconv is a reliable
    source of checkerboard artefacts and buys nothing the following conv does not.
    """

    def __init__(
        self,
        backbone: str,
        channel_names: list[str],
        layers: list[int],
        n_subjects: int,
        n_timepoints: int = 250,
        pretrained: bool = True,
        freeze_blocks: int = 0,
        freeze_all: bool = False,
        pool: str = "cls",
        drop: float = 0.1,
        fusion_mode: str = "uniform",
        prior_center: int | None = None,
        prior_strength: float = 1.0,
        layer_dropout: float = 0.1,
        subject_dropout: float = 0.3,
        projector: str = "linear",
        patch_size: int = 16,
        n_patches_w: int = 14,
        zscore: bool = True,
        pool_norm: bool = True,
        cls_token_prefix: bool = True,
        base_ch: int = 128,
        base_hw: int = 8,
        field_ch: int = 32,
        out_hw: int = 64,
        vae_ch: int = 4,
        style: str = "region-time",
        tokenizer_kind: str = "topography",
        scalp_res: int = 64,
        n_time_bands: int = 3,
        band_channels: str = "replicate",
    ) -> None:
        super().__init__()
        self.out_hw = int(out_hw)
        self.vae_ch = int(vae_ch)
        self.tokenizer_kind = tokenizer_kind
        # `dense` is what makes the scalp grid reachable by the decoder. The grid
        # tokenizer pool-writes random MLP features and has no spatial meaning, so it
        # keeps the old pooled path; only the patch-embed geometries produce a token
        # grid whose axes are spatial.
        self.dense = tokenizer_kind == "topography"

        self.encoder = build_encoder(
            backbone,
            channel_names=channel_names,
            n_timepoints=n_timepoints,
            pretrained=pretrained,
            freeze_blocks=freeze_blocks,
            freeze_all=freeze_all,
            pool=pool,
            drop=drop,
            tokenizer_kind=tokenizer_kind,
            pool_norm=pool_norm,
            cls_token_prefix=cls_token_prefix,
            patch_size=patch_size,
            n_patches_w=n_patches_w,
            zscore=zscore,
            style=style,
            scalp_res=scalp_res,
            n_time_bands=n_time_bands,
            band_channels=band_channels,
            layers=layers,
        )
        self.fusion = LayerFusion(
            layers=layers,
            d_in=self.encoder.feat_dim,
            n_subjects=n_subjects,
            d_out=self.encoder.feat_dim,
            prior_center=prior_center,
            prior_strength=prior_strength,
            layer_dropout=layer_dropout,
            subject_dropout=subject_dropout,
            projector=projector,
            fusion_mode=fusion_mode,
        )
        d = self.encoder.feat_dim

        # The token grid is a property of the encoder, not of the decoder branch, so it
        # is set here rather than inside `if self.dense`. `StructureTower.grid_hw` is
        # read by the export path's shape checks and by the tests, and a branch-local
        # attribute means those raise AttributeError on whichever geometry was not the
        # one in use when the attribute was introduced.
        self.grid_hw = tuple(self.encoder.dst_grid)
        self.n_tokens = int(self.grid_hw[0] * self.grid_hw[1])

        if self.dense:
            # Token grid -> spatial field. `grid` is the backbone's patch grid over
            # the topography image (e.g. (12, 4) for a 192x64 image at patch 16).
            gh, gw = self.grid_hw
            self.grid_mix = nn.Sequential(
                nn.Conv2d(d, base_ch, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, base_ch), base_ch),
                nn.GELU(),
            )
            # The two readouts are concatenated along channels: local grid features
            # and the global pooled vector broadcast over every location.
            in_ch = base_ch + d
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, base_ch), base_ch),
                nn.GELU(),
            )
            # Upsample schedule computed from the token grid, ending exactly at
            # out_hw. Doubling until the longer axis reaches out_hw and then a final
            # resize to the square target: the scaling from scalp space to image
            # latent space is anisotropic anyway, so there is nothing to preserve by
            # holding the aspect ratio, and a fixed schedule keeps the parameter
            # count and the memory independent of the geometry a config chose.
            sizes: list[tuple[int, int]] = []
            h, w = gh, gw
            while max(h, w) < self.out_hw and len(sizes) < 4:
                h, w = h * 2, w * 2
                sizes.append((h, w))
            self.up_sizes = sizes
            self.up = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),
                    nn.GroupNorm(min(8, base_ch), base_ch),
                    nn.GELU(),
                )
                for _ in sizes
            ])
            # Final conv to the latent, at the target resolution.
            self.vae_head = nn.Sequential(
                nn.Conv2d(base_ch, field_ch, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, field_ch), field_ch),
                nn.GELU(),
                nn.Conv2d(field_ch, vae_ch, 3, padding=1),
            )
        else:
            if out_hw != base_hw * (2 ** 3):
                raise ValueError(f"out_hw {out_hw} is not base_hw {base_hw} upsampled x2 "
                                 f"three times; adjust base_hw or add/remove an _up_block")
            self.proj = nn.Linear(d, base_ch * base_hw * base_hw)
            self.base_ch, self.base_hw = int(base_ch), int(base_hw)
            self.up = nn.Sequential(
                _up_block(base_ch, base_ch // 2),
                _up_block(base_ch // 2, field_ch),
                _up_block(field_ch, field_ch),
            )
            self.vae_head = nn.Conv2d(field_ch, vae_ch, 3, padding=1)
            self.up_sizes = []

        # NOTE: the depth head is gone, and its removal is a measurement rather than
        # a simplification. Depth's best linear read-out from EEG (r = +0.16) sits
        # BELOW the constant-map predictor (r = +0.53) on all four probe arms
        # (sub-08/10 x 63/17 channels, margin -0.37 each), so no head design could
        # recover it, and the head that was shipped had already reached that ceiling
        # (r = +0.6485 against the constant's +0.6540). Training it was spending
        # gradient on a target the input does not contain.
        # The head is zero-initialised on BOTH branches so that training starts at the
        # mean field, not at noise. This is not cosmetic: the targets are z-scored, so
        # a zero output IS the constant predictor, and the loss at step 1 is therefore
        # exactly the target's own variance -- the quantity the collapse gate and the
        # probe's `var ratio` column both compare against. Starting from noise instead
        # would make the first epochs a fit to the mean that the numbers cannot be read
        # against, and, under L1, would start the head closer to the collapse basin
        # than to the mean field.
        #
        # `vae_head` is a `Sequential` on the dense branch and a bare `Conv2d` on the
        # pooled one, and `[-1]` on a Conv2d is a `TypeError` at construction time, not
        # at run time -- that asymmetry was latent until a config actually selected
        # `--struct-tokenizer eegit`.
        last = self.vae_head[-1] if isinstance(self.vae_head, nn.Sequential) else self.vae_head
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.drop = nn.Dropout(drop)

    def forward(
        self, eeg: torch.Tensor, subject_ids: torch.Tensor | None = None, training: bool = True
    ) -> dict[str, torch.Tensor]:
        feats = self.encoder(eeg, self.fusion.layers, dense=self.dense)
        if self.dense:
            # Pool the dense tokens for the fusion module. `LayerFusion` is written
            # for per-layer (B, D) features and its job -- deciding how much each
            # depth contributes -- is a per-sample scalar decision, so summarising the
            # grid here is correct rather than a shortcut.
            pooled = {k: v.mean(dim=1) for k, v in feats.items()}
            fused, w = self.fusion(pooled, subject_ids=subject_ids, training=training)
            gh, gw = self.grid_hw
            stacked = torch.stack([feats[l] for l in self.fusion.layers], dim=1)
            # (B, K, N, D) -> same (K)-weighted blend the fusion applied, but kept
            # spatially resolved. Reusing `w` rather than re-deriving weights keeps
            # one blend decision for both readouts.
            mixed = (stacked * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)      # (B, N, D)
            b, n, d = mixed.shape
            if n != gh * gw:
                raise RuntimeError(f"structure token grid mismatch: encoder gave {n} "
                                   f"tokens, dst_grid {self.grid_hw} implies {gh * gw}")
            grid = mixed.transpose(1, 2).reshape(b, d, gh, gw)
            h = self.grid_mix(grid)
            ctx = fused.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, gh, gw)
            h = self.stem(torch.cat([h, ctx], dim=1))
            for size, blk in zip(self.up_sizes, self.up):
                h = F.interpolate(h, size=size, mode="nearest")
                h = blk(h)
            h = self.drop(h)
            if h.shape[-1] != self.out_hw or h.shape[-2] != self.out_hw:
                h = F.interpolate(h, size=(self.out_hw, self.out_hw),
                                  mode="bilinear", align_corners=False)
            return {
                "fused": fused,
                "layer_w": w,
                "field": h,
                "grid": mixed,
                "vae": self.vae_head(h),                             # (B, 4, 64, 64)
            }

        fused, w = self.fusion(feats, subject_ids=subject_ids, training=training)
        h = self.proj(fused)
        field = self.up(h.reshape(h.shape[0], self.base_ch, self.base_hw, self.base_hw))
        field = self.drop(field)
        return {
            "fused": fused,
            "layer_w": w,
            "field": field,
            "vae": self.vae_head(field),                             # (B, 4, 64, 64)
        }


class RetrievalModel(nn.Module):
    """EEG branch (pretrained ViT) aligned to frozen image features.

    The image branch is frozen and only projected: the cached features already
    live in a well-formed space (CLIP joint space), so there is nothing to learn
    on that side. All capacity goes into the EEG encoder and the fusion.

    A second, structurally independent tower can be attached (`struct_backbone`);
    see `StructureTower`. It is optional so that every existing retrieval run --
    and its result files -- stay reproducible from the same code.
    """

    def __init__(
        self,
        backbone: str,
        channel_names: list[str],
        layers: list[int],
        n_subjects: int,
        d_embed: int = 512,
        grid_h: int = 7,
        grid_w: int = 7,
        n_time_windows: int = 4,
        n_timepoints: int = 250,
        image_dim: int = 1024,
        pretrained: bool = True,
        freeze_blocks: int = 0,
        freeze_all: bool = False,
        pool: str = "cls",
        prior_center: int | None = None,
        prior_strength: float = 1.0,
        layer_dropout: float = 0.1,
        subject_dropout: float = 0.3,
        projector: str = "linear",
        img_projector: str = "linear",
        drop: float = 0.1,
        fusion_mode: str = "routed",
        target_fusion: str = "single",
        n_targets: int = 1,
        tokenizer_kind: str = "grid",
        pool_norm: bool = True,
        cls_token_prefix: bool = True,
        patch_size: int = 16,
        n_patches_w: int = 14,
        zscore: bool = True,
        style: str = "region-time",
        head_kind: str = "mlp",
        img_head_kind: str = "mlp",
        head_drop: float = 0.5,
        timm_global_pool: str = "",
        struct_backbone: str = "",
        struct_arch: str = "vit",
        struct_layers: list[int] | None = None,
        struct_patch_size: int = 16,
        struct_n_patches_w: int = 14,
        struct_fusion_mode: str = "uniform",
        struct_freeze_blocks: int = 0,
        struct_drop: float = 0.1,
        struct_tokenizer: str = "topography",
        struct_cfg: dict | None = None,
        struct_out_hw: int = 64,
        vae_ch: int = 4,
        da2_model: str = "",
        da2_local_files_only: bool = True,
    ) -> None:
        super().__init__()
        if target_fusion not in ("single", "mean", "routed", "routed_sr"):
            raise ValueError(f"target_fusion must be single|mean|routed|routed_sr, "
                             f"got {target_fusion!r}")
        if target_fusion != "single" and n_targets < 2:
            raise ValueError(f"target_fusion={target_fusion} needs >=2 targets, got {n_targets}")
        # `mlp` is the local head: one hidden layer, GELU, no residual. `eegit` is the
        # released code's ProjectionHead. The old name for the first one was `nw`, a
        # project codename that described no property of the head; canonicalise so
        # saved configs keep loading.
        head_kind = {"nw": "mlp"}.get(head_kind, head_kind)
        img_head_kind = {"nw": "mlp"}.get(img_head_kind, img_head_kind)
        if head_kind not in ("mlp", "eegit"):
            raise ValueError(f"head_kind must be 'mlp' or 'eegit' (legacy: 'nw'), "
                             f"got {head_kind!r}")
        if img_head_kind not in ("mlp", "eegit"):
            raise ValueError(f"img_head_kind must be 'mlp' or 'eegit' (legacy: 'nw'), "
                             f"got {img_head_kind!r}")
        self.head_kind = head_kind
        self.img_head_kind = img_head_kind
        self.style = style
        self.target_fusion = target_fusion
        self.n_targets = n_targets
        self.encoder = build_encoder(
            backbone,
            channel_names=channel_names,
            grid_h=grid_h,
            grid_w=grid_w,
            n_time_windows=n_time_windows,
            n_timepoints=n_timepoints,
            pretrained=pretrained,
            freeze_blocks=freeze_blocks,
            freeze_all=freeze_all,
            pool=pool,
            drop=drop,
            tokenizer_kind=tokenizer_kind,
            pool_norm=pool_norm,
            cls_token_prefix=cls_token_prefix,
            patch_size=patch_size,
            n_patches_w=n_patches_w,
            zscore=zscore,
            style=style,
            timm_global_pool=timm_global_pool,
            layers=layers,
        )
        self.fusion = LayerFusion(
            layers=layers,
            d_in=self.encoder.feat_dim,
            n_subjects=n_subjects,
            d_out=self.encoder.feat_dim,
            prior_center=prior_center,
            prior_strength=prior_strength,
            layer_dropout=layer_dropout,
            subject_dropout=subject_dropout,
            projector=projector,
            fusion_mode=fusion_mode,
        )
        if head_kind == "eegit":
            self.eeg_head = EEGiTProjectionHead(self.encoder.feat_dim, d_embed, head_drop)
        else:
            self.eeg_head = _mlp(self.encoder.feat_dim, self.encoder.feat_dim, d_embed, 1, drop)

        # Multi-target blending over image-tower layers. Done here, inside
        # encode_image, rather than in the training loop, so that training and
        # evaluation provably share one code path -- selection and test then cannot
        # disagree about which representation the EEG was aligned to.
        if target_fusion == "single":
            self.target_w: nn.Parameter | None = None
            self.target_router: LayerFusion | None = None
        elif target_fusion == "mean":
            self.target_w = None          # fixed uniform: the doc's "先均匀融合"
            self.target_router = None
        elif target_fusion == "routed":
            self.target_w = nn.Parameter(torch.zeros(n_targets))
            self.target_router = None
        else:
            # `routed_sr`: SAMGA's global-residual granularity routing, applied to
            # the TARGET axis instead of the EEG layer axis.
            #
            # SAMGA Eq. 4 puts the subject on the visual-target weights --
            # `alpha_n = softmax((q + r_n * b_s) / tau)` -- and Eq. 7 drops the
            # residual at inference, which is what "subject-aware training,
            # subject-agnostic inference" means. `LayerFusion` already implements
            # exactly that factorisation, with the subject residual, the subject
            # dropout that forces the global-only path to be optimised, and the
            # layer dropout that keeps the routing distribution valid. The target
            # axis is the same object with `k = n_targets`, so it is reused rather
            # than reimplemented: a second copy is a second place for the dropout
            # semantics to drift.
            #
            # `target_w` stays None so that a config cannot end up with two
            # competing target-weight parameters, one of which is never read.
            if n_targets < 2:
                raise ValueError(
                    "target_fusion='routed_sr' needs >=2 target layers: a per-subject "
                    "residual over a single target is a constant after softmax, so it "
                    "would train no parameter and claim a mechanism it does not have")
            self.target_w = None
            self.target_router = LayerFusion(
                layers=list(range(n_targets)),
                d_in=image_dim,
                n_subjects=n_subjects,
                d_out=image_dim,
                prior_center=prior_center,
                prior_strength=prior_strength,
                layer_dropout=layer_dropout,
                subject_dropout=subject_dropout,
                projector=projector,
                fusion_mode="routed",
            )

        if img_head_kind == "eegit":
            self.img_head = EEGiTProjectionHead(image_dim, d_embed, head_drop)
        elif img_projector == "linear":
            self.img_head = nn.Linear(image_dim, d_embed)
        elif img_projector == "identity":
            assert image_dim == d_embed, "identity projector requires matching dims"
            self.img_head = nn.Identity()
        else:
            self.img_head = _mlp(image_dim, image_dim, d_embed, 2, drop)

        # ---- optional second tower ------------------------------------------
        # Left None rather than built-and-frozen: a run that does not ask for the
        # structure tower must not pay for it in memory, in `state_dict()`, or in
        # the parameter-group assignment below.
        #
        # Two structural architectures, and the choice is a claim about where the
        # knowledge of "structure" should come from:
        #
        #   `vit` (default) -- `StructureTower`: a pretrained trunk plus a decoder
        #     this project trains from scratch. It chooses its own output space
        #     (VAE latents), so the decoder has to learn both "what a spatial field
        #     looks like" and "which one this EEG implies" from a 1504-concept
        #     training set.
        #   `da2` -- `DepthTower`: Depth Anything V2 whole, with only its input
        #     interface replaced by EEG patches. It brings a pretrained
        #     features-to-depth path with it, and its output is the condition
        #     ControlNet-depth was trained on.
        #
        # They are not two settings of one thing; `da2` has no `layers`, no fusion
        # and no `field_ch`, because those were `StructureTower`'s answers to a
        # question it no longer has to ask. Kept as separate classes rather than
        # branches inside one, so that neither architecture's assumptions can leak
        # into the other's defaults.
        self.struct: nn.Module | None = None
        if struct_backbone:
            if struct_arch == "da2":
                from .depth_tower import DA2_SMALL, DA2EEGEncoder, DepthTower

                self.struct = DepthTower(
                    DA2EEGEncoder(
                        channel_names=channel_names,
                        model_id=da2_model or DA2_SMALL,
                        patch_size=struct_patch_size,
                        n_patches_w=struct_n_patches_w,
                        n_timepoints=n_timepoints,
                        style=style,
                        zscore=zscore,
                        drop=struct_drop,
                        freeze_blocks=struct_freeze_blocks,
                        local_files_only=da2_local_files_only,
                    ),
                    out_hw=struct_out_hw,
                    vae_ch=vae_ch,
                    drop=struct_drop,
                )
            elif struct_arch == "vit":
                self.struct = StructureTower(
                    backbone=struct_backbone,
                    channel_names=channel_names,
                    layers=list(struct_layers or []),
                    n_subjects=n_subjects,
                    n_timepoints=n_timepoints,
                    pretrained=pretrained,
                    freeze_blocks=struct_freeze_blocks,
                    freeze_all=freeze_all,
                    pool=pool,
                    drop=struct_drop,
                    fusion_mode=struct_fusion_mode,
                    prior_center=prior_center,
                    prior_strength=prior_strength,
                    layer_dropout=layer_dropout,
                    subject_dropout=subject_dropout,
                    projector=projector,
                    patch_size=struct_patch_size,
                    n_patches_w=struct_n_patches_w,
                    zscore=zscore,
                    style=style,
                    pool_norm=pool_norm,
                    cls_token_prefix=cls_token_prefix,
                    tokenizer_kind=struct_tokenizer,
                    **(struct_cfg or {}),
                )
            else:
                raise KeyError(f"struct_arch must be 'vit' or 'da2', got {struct_arch!r}")

    def target_weights(self) -> torch.Tensor | None:
        """Resulting blend weights over the target layers, for the run record.

        Worth recording rather than assuming: if a routed run drives all its mass
        onto one layer, that is the honest statement that multi-layer fusion bought
        nothing and the winning single layer was already sufficient.
        """
        if self.target_fusion == "single" or self.n_targets == 1:
            return None
        if self.target_router is not None:
            # The GLOBAL weights, not the subject-conditioned ones: inference uses
            # the global prior, so recording the per-subject mixture would describe
            # a model that is never deployed.
            return self.target_router.layer_weights()
        if self.target_w is None:
            return torch.full((self.n_targets,), 1.0 / self.n_targets)
        return torch.softmax(self.target_w.detach(), dim=0)

    # ------------------------------------------------------------------ fwd
    def encode_eeg(
        self, eeg: torch.Tensor, subject_ids: torch.Tensor | None = None, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.encoder(eeg, self.fusion.layers)
        fused, w = self.fusion(feats, subject_ids=subject_ids, training=training)
        z = self.eeg_head(fused)
        # No softplus here. It belongs on the loss temperature (see losses.InfoNCE);
        # applying it to the embedding forces every coordinate positive and, after
        # L2 normalisation, collapses the geometry into the positive orthant.
        return z, fused, w

    def encode_image(
        self,
        feat: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        training: bool = False,
    ) -> torch.Tensor:
        """Frozen image features -> the space the EEG is compared in.

        `feat` is (B, D) for a single target, or (B, k, D) when several image-tower
        layers are blended. Blending here (not in the caller) keeps the evaluation
        path identical to training.

        `subject_ids` matters only for `target_fusion='routed_sr'`, where the blend
        is subject-conditioned during training and reverts to the global prior at
        inference (SAMGA Eq. 4 / Eq. 7). The default is `training=False` so that the
        many inference-only callers -- `evaluate`, the export, the diagnostics --
        get the deployed behaviour without having to know the mode exists.

        The guard below exists because getting this wrong is silent: with
        `subject_ids=None` the router still runs, still returns a blend, and still
        trains, just without the residual branch that is the entire reason the mode
        was selected.
        """
        if (
            self.target_router is not None
            and training
            and subject_ids is None
        ):
            raise RuntimeError(
                "target_fusion='routed_sr' is being trained without subject_ids, so "
                "the per-subject target residual would never receive a gradient and "
                "the run would report a mechanism it is not using. Pass the batch's "
                "subject ids, or select target_fusion='routed'.")
        if self.target_fusion != "single" and feat.dim() == 3:
            if self.target_router is not None:
                fused, _w = self.target_router(
                    {i: feat[:, i, :] for i in range(feat.shape[1])},
                    subject_ids=subject_ids,
                    training=training,
                )
                return self.img_head(fused)
            if self.target_w is None:
                feat = feat.mean(dim=1)
            else:
                w = torch.softmax(self.target_w, dim=0)
                feat = (feat * w.view(1, -1, 1)).sum(dim=1)
        return self.img_head(feat)

    def forward(self, eeg, image_feat, subject_ids=None, training=True):
        z_e, fused, w = self.encode_eeg(eeg, subject_ids, training)
        z_i = self.encode_image(image_feat, subject_ids, training)
        return z_e, z_i, w

    def forward_all(
        self, eeg, subject_ids=None, training: bool = True, with_struct: bool = True
    ) -> dict:
        """One EEG pass producing both towers' outputs.

        Exists because validation needs *both* signals per epoch but the structure
        trunk is by far the most expensive module in the model: two independent
        forward calls would double validation time for the same numbers. Training
        and validation both go through here so there is exactly one code path that
        can be wrong.
        """
        z, fused, w = self.encode_eeg(eeg, subject_ids, training)
        struct = None
        if with_struct and self.struct is not None:
            struct = self.struct(eeg, subject_ids=subject_ids, training=training)
        return {"z": z, "fused": fused, "layer_w": w, "struct": struct}


def build_from_args(cfg: dict, channel_names: list[str], image_dim: int) -> RetrievalModel:
    """Rebuild exactly the model a run's saved args describe.

    A single source of truth for the constructor call. Training and the condition
    export both need it, and a divergence between the two is not a loud failure: it
    surfaces as a `load_state_dict` key mismatch (at best) or as a head with a
    different width silently reloading (at worst). The export step runs hours after
    training, so it is exactly the place an unnoticed drift would be discovered too
    late to fix.
    """
    def g(key, default):
        return cfg.get(key, default)

    return RetrievalModel(
        backbone=cfg["backbone"],
        channel_names=channel_names,
        layers=list(cfg["layers"]),
        n_subjects=g("n_subjects", 1),
        d_embed=g("d_embed", 512),
        grid_h=g("grid_h", 7),
        grid_w=g("grid_w", 7),
        n_time_windows=g("n_time_windows", 4),
        n_timepoints=g("n_timepoints", 250),
        image_dim=image_dim,
        pretrained=not g("no_pretrained", False),
        freeze_blocks=g("freeze_blocks", 0),
        freeze_all=g("freeze_all", False),
        pool=g("pool", "cls"),
        prior_center=g("prior_center", None),
        prior_strength=g("prior_strength", 1.0),
        layer_dropout=g("layer_dropout", 0.1),
        subject_dropout=g("subject_dropout", 0.3),
        drop=g("drop", 0.1),
        fusion_mode=g("fusion_mode", "routed"),
        target_fusion=g("target_fusion", "single"),
        n_targets=(len(cfg["target_layers"]) if cfg.get("target_layers") else 1),
        tokenizer_kind=g("tokenizer", "grid"),
        pool_norm=not g("no_pool_norm", False),
        cls_token_prefix=not g("no_cls_token", False),
        patch_size=g("patch_size", 16),
        n_patches_w=g("n_patches_w", 14),
        zscore=not g("no_zscore", False),
        struct_backbone=g("struct_backbone", ""),
        struct_arch=g("struct_arch", "vit"),
        struct_layers=g("struct_layers", None),
        struct_patch_size=g("struct_patch_size", 16),
        struct_n_patches_w=g("struct_n_patches_w", 14),
        struct_fusion_mode=g("struct_fusion_mode", "uniform"),
        struct_freeze_blocks=g("struct_freeze_blocks", 0),
        struct_drop=g("struct_drop", 0.1),
        struct_tokenizer=g("struct_tokenizer", "topography"),
        # `struct_out_hw` / `vae_ch` are named arguments rather than `struct_cfg`
        # entries because `DepthTower` reads them too, and `struct_cfg` is splatted
        # only into `StructureTower` -- a config that selected `--struct-arch da2`
        # would otherwise silently keep the 64/4 defaults.
        struct_out_hw=g("struct_out_hw", 64),
        vae_ch=g("struct_vae_ch", 4),
        da2_model=g("da2_model", ""),
        da2_local_files_only=not g("da2_allow_download", False),
        # The topography geometry travels in `struct_cfg` rather than as named
        # `RetrievalModel` arguments, because it is meaningful only for the
        # `topography` tokenizer and every other path must not grow a parameter it
        # does not read.
        # `base_hw` is the seed resolution the pooled decoder expands from: its
        # `proj` emits `base_ch * base_hw**2` channels, reshaped to
        # `(B, base_ch, base_hw, base_hw)`, and the three `_up_block`s then carry it
        # to `out_hw`. So `out_hw == base_hw * 8` is the architecture's own
        # parameterisation, not an accident, and the seed shrinks with the target.
        # It is exposed because `out_hw` alone cannot express a COARSE target: with
        # the default `base_hw=8` the only legal `out_hw` is 64, which is exactly the
        # fine cache every structural run so far has regressed -- the coarsest end of
        # the ladder the probe prices at 5x. The default is unchanged, so every
        # existing config constructs byte-identically.
        struct_cfg={"base_ch": g("struct_base_ch", 128),
                    "base_hw": g("struct_base_hw", 8),
                    "field_ch": g("struct_field_ch", 32),
                    "scalp_res": g("struct_scalp_res", 64),
                    "n_time_bands": g("struct_n_time_bands", 3),
                    "band_channels": g("struct_band_channels", "replicate"),
                    "vae_ch": g("struct_vae_ch", 4),
                    "out_hw": g("struct_out_hw", 64)},
        style=g("patch_style", "region-time"),
        head_kind=g("head_kind", "mlp"),
        img_head_kind=g("img_head_kind", "mlp"),
        head_drop=g("head_drop", 0.5),
        timm_global_pool=g("timm_global_pool", ""),
    )
