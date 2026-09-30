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
    """The second tower: EEG patches -> a spatial latent field, not a joint vector.

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
    The design document reached the same conclusion from the parameter side: the
    winning semantic backbone is 768-d and the structural one is 1024-d, so there
    is no width at which the two features could be compared, let alone shared.

    The interface, however, *is* shared and that is the point: this tower runs its
    own `EEGPatchTokenizer` instance with `patch_size` set to its own backbone's
    conv kernel (14 for DINOv2), so the EEG is presented as an EEG image tileable
    by that backbone. The tokenizer is parameter-free, so "sharing the interface"
    costs nothing and the two towers are trained on exactly the same EEG input
    representation.

    DINOv2 for the trunk, for a property the semantic trunk does not have: its
    self-supervised objective is explicitly variance- and patch-level, so its
    features stay informative about *where* things are rather than only about
    what. A CLIP trunk's last layers are trained to discard exactly the pose and
    layout nuisance that this tower is being asked to predict.

    Head geometry: the fused feature is projected to a small 8x8 field and then
    upsampled x8 to the VAE's 64x64 grid. Starting dense (8x8 = 64 cells) rather
    than from the backbone's token grid is deliberate -- the token grid is
    (n_regions, n_time_patches), i.e. a brain-region x time layout with no
    relation to image rows and columns, and warping it into 64x64 would bake in an
    arbitrary correspondence. The tokens influence the field through attention,
    which is global anyway, and the field is free to learn the image layout from
    the loss.
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
        patch_size: int = 14,
        n_patches_w: int = 16,
        zscore: bool = True,
        pool_norm: bool = True,
        cls_token_prefix: bool = True,
        base_ch: int = 128,
        base_hw: int = 8,
        field_ch: int = 32,
        out_hw: int = 64,
        vae_ch: int = 4,
        style: str = "nw",
    ) -> None:
        super().__init__()
        if out_hw != base_hw * (2 ** 3):
            raise ValueError(f"out_hw {out_hw} is not base_hw {base_hw} upsampled x2 three "
                             f"times; adjust base_hw or add/remove an _up_block")
        self.out_hw = int(out_hw)
        self.vae_ch = int(vae_ch)

        self.encoder = build_encoder(
            backbone,
            channel_names=channel_names,
            n_timepoints=n_timepoints,
            pretrained=pretrained,
            freeze_blocks=freeze_blocks,
            freeze_all=freeze_all,
            pool=pool,
            drop=drop,
            tokenizer_kind="eegit",
            pool_norm=pool_norm,
            cls_token_prefix=cls_token_prefix,
            patch_size=patch_size,
            n_patches_w=n_patches_w,
            zscore=zscore,
            style=style,
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
        self.proj = nn.Linear(self.encoder.feat_dim, base_ch * base_hw * base_hw)
        self.base_ch, self.base_hw = int(base_ch), int(base_hw)
        self.up = nn.Sequential(
            _up_block(base_ch, base_ch // 2),
            _up_block(base_ch // 2, field_ch),
            _up_block(field_ch, field_ch),
        )
        self.depth_head = nn.Conv2d(field_ch, 1, 3, padding=1)
        self.vae_head = nn.Conv2d(field_ch, vae_ch, 3, padding=1)
        # Zero-init the latent head so step 0 predicts the conditional mean, i.e.
        # the exactly-optimal constant predictor for L1. It removes the first few
        # hundred steps that would otherwise be spent un-learning a random field,
        # and it is what makes the val curve readable from epoch 1. Not applied to
        # the depth head, whose output is squashed and would then sit at 0.5 for
        # every pixel -- a plateau its gradient has no reason to leave.
        nn.init.zeros_(self.vae_head.weight)
        nn.init.zeros_(self.vae_head.bias)
        self.drop = nn.Dropout(drop)

    def forward(
        self, eeg: torch.Tensor, subject_ids: torch.Tensor | None = None, training: bool = True
    ) -> dict[str, torch.Tensor]:
        feats = self.encoder(eeg, self.fusion.layers)
        fused, w = self.fusion(feats, subject_ids=subject_ids, training=training)
        h = self.proj(fused)
        field = self.up(h.reshape(h.shape[0], self.base_ch, self.base_hw, self.base_hw))
        field = self.drop(field)
        return {
            "fused": fused,
            "layer_w": w,
            "field": field,
            "depth": torch.sigmoid(self.depth_head(field)),          # (B, 1, 64, 64)
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
        style: str = "nw",
        head_kind: str = "nw",
        img_head_kind: str = "nw",
        head_drop: float = 0.5,
        timm_global_pool: str = "",
        struct_backbone: str = "",
        struct_layers: list[int] | None = None,
        struct_patch_size: int = 14,
        struct_n_patches_w: int = 16,
        struct_fusion_mode: str = "uniform",
        struct_freeze_blocks: int = 0,
        struct_drop: float = 0.1,
        struct_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        if target_fusion not in ("single", "mean", "routed"):
            raise ValueError(f"target_fusion must be single|mean|routed, got {target_fusion!r}")
        if target_fusion != "single" and n_targets < 2:
            raise ValueError(f"target_fusion={target_fusion} needs >=2 targets, got {n_targets}")
        if head_kind not in ("nw", "eegit"):
            raise ValueError(f"head_kind must be 'nw' or 'eegit', got {head_kind!r}")
        if img_head_kind not in ("nw", "eegit"):
            raise ValueError(f"img_head_kind must be 'nw' or 'eegit', got {img_head_kind!r}")
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
        elif target_fusion == "mean":
            self.target_w = None          # fixed uniform: the doc's "先均匀融合"
        else:
            self.target_w = nn.Parameter(torch.zeros(n_targets))

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
        self.struct: StructureTower | None = None
        if struct_backbone:
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
                **(struct_cfg or {}),
            )

    def target_weights(self) -> torch.Tensor | None:
        """Resulting blend weights over the target layers, for the run record.

        Worth recording rather than assuming: if a routed run drives all its mass
        onto one layer, that is the honest statement that multi-layer fusion bought
        nothing and the winning single layer was already sufficient.
        """
        if self.target_fusion == "single" or self.n_targets == 1:
            return None
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

    def encode_image(self, feat: torch.Tensor) -> torch.Tensor:
        """Frozen image features -> the space the EEG is compared in.

        `feat` is (B, D) for a single target, or (B, k, D) when several image-tower
        layers are blended. Blending here (not in the caller) keeps the evaluation
        path identical to training.
        """
        if self.target_fusion != "single" and feat.dim() == 3:
            if self.target_w is None:
                feat = feat.mean(dim=1)
            else:
                w = torch.softmax(self.target_w, dim=0)
                feat = (feat * w.view(1, -1, 1)).sum(dim=1)
        return self.img_head(feat)

    def forward(self, eeg, image_feat, subject_ids=None, training=True):
        z_e, fused, w = self.encode_eeg(eeg, subject_ids, training)
        z_i = self.encode_image(image_feat)
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
        n_subjects=1,                       # intra-subject
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
        struct_layers=g("struct_layers", None),
        struct_patch_size=g("struct_patch_size", 14),
        struct_n_patches_w=g("struct_n_patches_w", 16),
        struct_fusion_mode=g("struct_fusion_mode", "uniform"),
        struct_freeze_blocks=g("struct_freeze_blocks", 0),
        struct_drop=g("struct_drop", 0.1),
        struct_cfg={"base_ch": g("struct_base_ch", 128),
                    "field_ch": g("struct_field_ch", 32)},
        style=g("patch_style", "nw"),
        head_kind=g("head_kind", "nw"),
        img_head_kind=g("img_head_kind", "nw"),
        head_drop=g("head_drop", 0.5),
        timm_global_pool=g("timm_global_pool", ""),
    )
