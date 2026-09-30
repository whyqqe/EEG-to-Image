"""Pretrained ViT backbones repurposed as EEG encoders, with multi-layer fusion.

Design
------
The core move: *do not* train an EEG encoder from scratch and *do not* bolt a
frozen generic EEG foundation model in front. Instead take a pretrained vision /
language model and swap its input interface:

    patch_embed (14x14x3 conv, image pixels)  ->  EEGTokenizer projection

Everything downstream -- attention, MLP blocks, LayerNorm, positional structure
-- keeps its pretrained weights. The pretrained blocks carry priors about how
tokens on a 2D grid relate to one another; the tokenizer's job is to present EEG
in a form where those priors are applicable.

Two backbones are supported, for two different jobs:

  semantic  : CLIP visual tower. Bidirectional (the CLIP *text* tower is causal,
              which would cut half the attention directions for EEG's
              bidirectionally coupled time axis), patch-native, and its
              visual.proj maps into the joint image-text space that the cached
              image features already live in.
  structure : DINOv2-L. Trained with iBOT masked-image modelling, which forces it
              to retain recoverable detail (dense tasks +3%), and KoLeo, which
              spreads features for retrieval (+8% on instance retrieval).
              CLIP is explicitly trained to be *invariant* to layout/texture, so
              it discards exactly what the structure branch needs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import EEGPatchTokenizer, EEGTokenizer, ScalpTopographyTokenizer


# ---------------------------------------------------------------- backbones
# (timm name, embedding dim, native patch grid for a square input)
BACKBONES: dict[str, dict] = {
    "clip_vit_l14": dict(
        timm_name="vit_large_patch14_clip_224.openai", d_model=1024, src_grid=(16, 16),
        patch=14, n_blocks=24,
    ),
    "dinov2_l": dict(
        timm_name="vit_large_patch14_dinov2.lvd142m", d_model=1024, src_grid=(37, 37),
        patch=14, n_blocks=24,
    ),
    "dinov2_l_reg4": dict(
        timm_name="vit_large_patch14_reg4_dinov2.lvd142m", d_model=1024, src_grid=(37, 37),
        patch=14, n_blocks=24,
    ),
    "vit_b16_in21k": dict(
        timm_name="vit_base_patch16_224.augreg_in21k", d_model=768, src_grid=(14, 14),
        patch=16, n_blocks=12,
    ),
    # ---- structural-tower backbones (EEGiT's interface, a different prior) -----
    # The plan's second tower asks a specific question: can the *structural* branch
    # inherit a prior that is about *where things are*, the way the semantic branch
    # inherits EEGiT's `vit_base_patch16_224_in21k` prior? The two entries below are
    # the candidates, and they are not equivalent -- they are two different answers
    # to "what does a structure-oriented prior look like".
    #
    # `dinov3_b16` -- self-supervised, and specifically built to keep *dense*
    # features. DINOv3's contribution over DINOv2 is Gram anchoring, which the
    # authors introduce because "global metrics ... continue to improve, but dense
    # metrics ... degrade": the fix is to regularise the pairwise similarity between
    # patch features, i.e. to pin exactly the quantity that "structure" means here.
    # It also uses RoPE rather than a learned `pos_embed`, which matters for a
    # reason specific to this project: `pos_embed` has to be *resampled* onto a
    # non-square EEG grid (14x5 in EEGiT's own geometry), and RoPE is defined for
    # any grid, so the whole class of "the resampled positional grid is a different
    # interface than pretraining" defects does not arise. `src_grid` is therefore
    # unused for this entry (kept for the registry's shape and for reporting).
    # NOTE: `d_model=768` and 5 prefix tokens (1 cls + 4 registers).
    "dinov3_b16": dict(
        timm_name="vit_base_patch16_dinov3.lvd1689m", d_model=768, src_grid=(16, 16),
        patch=16, n_blocks=12, rope=True,
    ),
    # `mae_b16` -- the control arm, and the *literal* answer to "a structure-focused
    # prior": MAE is trained by reconstructing masked *pixels*, so its features are
    # the ones that provably retain the most recoverable appearance. A controlled
    # decoder comparison puts MAE-B at LPIPS 0.11 / recon-FID 0.16 against DINOv2-B's
    # 0.255 / 0.49, i.e. MAE reconstructs markedly better.
    #
    # It is the control rather than the default for a reason worth stating: DINOv3's
    # sharpness is *spatial coherence* (depth, segmentation), while MAE's is *pixel
    # appearance* (texture, colour). This project's own generation evidence says the
    # seven-metric gains on PixCorr/SSIM came from low-frequency layout -- the
    # `i2i_ll_s028` arm reached SSIM 0.365 / PixCorr 0.294 from a Gaussian-blur
    # initialisation, outperforming ATM -- so layout is the axis that pays. But that
    # is an inference, not a measurement of these two backbones as EEG trunks, which
    # is why both are registered and the run matrix trains both.
    "mae_b16": dict(
        timm_name="vit_base_patch16_224.mae", d_model=768, src_grid=(14, 14),
        patch=16, n_blocks=12,
    ),
    # The exact timm tag the released EEGiT code names
    # (`EEGVitEncoder(model_name="vit_base_patch16_224_in21k")`). Verified against
    # timm 1.0.28: this tag is a DEPRECATED ALIAS for `vit_base_patch16_224.augreg_in21k`,
    # i.e. the entry above -- `timm.create_model` prints
    #   "Mapping deprecated model name vit_base_patch16_224_in21k to current
    #    vit_base_patch16_224.augreg_in21k"
    # and loads the identical weights. So the "different pretrained weights" that
    # looked like a candidate explanation for the retrieval gap is NOT one; both
    # keys resolve to the same checkpoint. The key is kept so a config can name the
    # official tag verbatim and a result file can say it did.
    "vit_b16_in21k_orig": dict(
        timm_name="vit_base_patch16_224_in21k", d_model=768, src_grid=(14, 14),
        patch=16, n_blocks=12,
    ),
}

# The EEG "image" handed to `patch_embed` must be tileable by the backbone's own
# conv, so the tokenizer's `patch_size` is not a free parameter: it has to equal
# the pretrained conv's kernel. Requesting anything else produces patches the conv
# was never trained on, which silently voids the whole premise of the interface.
def backbone_patch_size(kind: str) -> int | None:
    """Kernel size of a registered backbone's `patch_embed` conv, or None."""
    name = kind.split(":", 1)[1] if ":" in kind else kind
    spec = BACKBONES.get(name)
    return int(spec["patch"]) if spec else None


def backbone_n_blocks(kind: str) -> int | None:
    """Transformer depth of a registered backbone, or None.

    Exists because this was hardcoded to 24 at the one call site that validates
    `--struct-layers`, which was correct when DINOv2-L was the only registered
    structural trunk and silently wrong the moment a ViT-B (12 blocks) was added:
    an index of 24 would pass validation and then simply be missing from the fusion
    dict, so the tower would train on fewer depths than the config claimed.
    """
    name = kind.split(":", 1)[1] if ":" in kind else kind
    spec = BACKBONES.get(name)
    return int(spec["n_blocks"]) if spec and spec.get("n_blocks") else None


def resample_pos_embed(grid_pe: torch.Tensor, src_grid: tuple[int, int],
                       dst_grid: tuple[int, int]) -> torch.Tensor:
    """Resample a patch positional embedding from one grid to another.

    grid_pe: (1, src_h*src_w, D) -> (1, dst_h*dst_w, D).

    Delegates to timm's `resample_abs_pos_embed` rather than reimplementing it.
    That is not laziness, it is the only way this can be correct: when a model is
    built with an `img_size` that differs from its pretrained cfg, timm resizes the
    checkpoint's `pos_embed` through exactly this function
    (`vision_transformer.checkpoint_filter_fn`), so anything else here produces a
    positional embedding that differs from the one the official arm actually uses.

    The first version of this function did reimplement it, as
    `F.interpolate(mode="bicubic", align_corners=False)` -- which is the same call
    *minus* `antialias=True`, the default timm passes. On a 14x14 -> 14x5 resize
    that is not a rounding-level difference: the un-antialiased version showed a
    max absolute deviation of 9.4 from the official `pos_embed`, which then
    propagated through 12 blocks into a 2.6 deviation in the pooled feature the
    EEG side is aligned against. Every EEGiT-patch arm built before this fix
    (including the 46.5% run) used the non-antialiased grid.

    `num_prefix_tokens=0` because the caller has already split the prefix rows off;
    this function only ever handles the grid rows.
    """
    if src_grid == dst_grid:
        return grid_pe
    _, n, d = grid_pe.shape
    sh, sw = src_grid
    if n != sh * sw:
        raise ValueError(f"pos_embed has {n} entries, expected {sh * sw} for grid {src_grid}")
    from timm.layers import resample_abs_pos_embed

    out = resample_abs_pos_embed(
        grid_pe.float(),
        new_size=list(dst_grid),
        old_size=list(src_grid),
        num_prefix_tokens=0,
    )
    return out.reshape(1, dst_grid[0] * dst_grid[1], d).to(grid_pe.dtype)


class OpenCLIPViTEEGEncoder(nn.Module):
    """open_clip vision tower with its patch interface replaced by EEG tokens.

    Why this class exists alongside `ViTEEGEncoder`
    ------------------------------------------------
    The cached image features (`data/image_feature/ViT-H-14/*.npy`, 1024-d) were
    produced by open_clip's ViT-H-14 as `cls @ visual.proj`. Because
    `visual.proj` is a (1280, 1024) map into the *joint* image-text space, using
    the same tower as the EEG encoder means the EEG side is asked to land in
    exactly the space the targets already occupy -- no cross-space translation,
    no separate target backbone.

    Attention here is bidirectional (only the CLIP *text* tower is causal, see
    docs section 4.2), and the input is patch-native, so replacing `conv1` +
    `positional_embedding` with EEG tokens keeps the pretrained blocks meaningful.
    """

    def __init__(
        self,
        channel_names: list[str],
        grid_h: int = 7,
        grid_w: int = 7,
        n_time_windows: int = 4,
        n_timepoints: int = 250,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79K",
        freeze_blocks: int = 0,
        freeze_all: bool = False,
        pool: str = "cls",
        drop: float = 0.1,
        proj_to_joint: bool = True,
        layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        import open_clip

        self.backbone_name = f"openclip:{model_name}"
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        visual = clip_model.visual
        self.d_model = int(visual.class_embedding.shape[0])
        self.n_blocks = len(visual.transformer.resblocks)
        self.pool = pool
        self.proj_to_joint = proj_to_joint and visual.proj is not None

        # keep only the vision tower; the text tower is a frozen target, not a model
        self.visual = visual
        del clip_model

        # ---- input interface swap ------------------------------------------
        self.tokenizer = EEGTokenizer(
            channel_names=channel_names, d_model=self.d_model, grid_h=grid_h, grid_w=grid_w,
            n_time_windows=n_time_windows, n_timepoints=n_timepoints, dropout=drop,
        )
        dst_grid = (grid_h, grid_w * n_time_windows)

        # ---- positional embedding ------------------------------------------
        pe = self.visual.positional_embedding.detach().unsqueeze(0)   # (1, 1+src, D)
        src_grid = tuple(int(s // p) for s, p in zip(self.visual.image_size, self.visual.patch_size))
        self.src_grid, self.dst_grid = src_grid, dst_grid
        cls_pe, grid_pe = pe[:, :1], pe[:, 1:]
        self.register_buffer("prefix_pos", cls_pe)
        self.register_buffer("grid_pos", resample_pos_embed(grid_pe, src_grid, dst_grid))

        if freeze_all:
            for p in self.visual.parameters():
                p.requires_grad_(False)
        for i, blk in enumerate(self.visual.transformer.resblocks):
            if i < freeze_blocks:
                for p in blk.parameters():
                    p.requires_grad_(False)
        self.deepest = (max(layers) if layers else self.n_blocks)
        if layers and self.deepest < self.n_blocks:
            for i, blk in enumerate(self.visual.transformer.resblocks):
                if i + 1 > self.deepest:
                    for p in blk.parameters():
                        p.requires_grad_(False)

        self.drop = nn.Dropout(drop)

    def trainable_parameter_summary(self) -> tuple[int, int]:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return tr, tot

    @property
    def feat_dim(self) -> int:
        """Width of the pooled per-layer feature this encoder emits."""
        return self.visual.proj.shape[1] if self.proj_to_joint else self.d_model

    def forward(self, eeg: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
        bad = [l for l in layers if not (1 <= l <= self.n_blocks)]
        if bad:
            raise ValueError(f"layers {bad} out of range 1..{self.n_blocks}")

        tok = self.tokenizer(eeg)                                   # (B, N, D)
        b = tok.shape[0]
        cls = self.visual.class_embedding.to(tok.dtype).view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat([cls + self.prefix_pos.to(tok.dtype),
                       tok + self.grid_pos.to(tok.dtype)], dim=1)
        x = self.drop(x)
        x = self.visual.ln_pre(x)

        # open_clip blocks are sequence-first
        x = x.permute(1, 0, 2)                                       # (L, B, D)
        want = set(layers)
        deepest = max(want)
        out: dict[int, torch.Tensor] = {}
        for i, blk in enumerate(self.visual.transformer.resblocks, start=1):
            x = blk(x)
            if i in want:
                out[i] = self._pool(x.permute(1, 0, 2))              # back to (B, L, D)
            if i >= deepest:
                # Nothing past the deepest requested layer can reach the loss: the
                # only consumer of `x` is the next block, and the fusion reads
                # `out`. Continuing would be pure wasted compute, and -- worse --
                # it would leave those blocks looking trainable when their gradient
                # is provably None. See `__init__` for the matching freeze.
                break
        return out

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        x = self.visual.ln_post(x)
        if self.pool == "cls":
            h = x[:, 0]
        elif self.pool == "mean":
            h = x[:, 1:].mean(dim=1)
        else:
            raise ValueError(f"unknown pool {self.pool!r}")
        if self.proj_to_joint:
            h = h @ self.visual.proj
        return h


class ViTEEGEncoder(nn.Module):
    """A pretrained timm ViT whose input interface has been replaced by EEG tokens.

    Returns per-layer pooled features so a downstream fusion module can combine
    several depths. Aligning to intermediate layers is the single largest
    reported lever in this literature, so the layer set is an explicit argument
    rather than a hard-coded "last layer".
    """

    def __init__(
        self,
        backbone: str,
        channel_names: list[str],
        grid_h: int = 7,
        grid_w: int = 7,
        n_time_windows: int = 4,
        n_timepoints: int = 250,
        pretrained: bool = True,
        freeze_blocks: int = 0,
        freeze_all: bool = False,
        pool: str = "cls",
        drop: float = 0.1,
        proj_to_joint: bool = False,
        tokenizer_kind: str = "grid",
        pool_norm: bool = True,
        cls_token_prefix: bool = True,
        patch_size: int = 16,
        n_patches_w: int = 14,
        zscore: bool = True,
        layers: list[int] | None = None,
        style: str = "region-time",
        timm_global_pool: str = "",
        scalp_res: int = 64,
        n_time_bands: int = 3,
        band_channels: str = "replicate",
    ) -> None:
        super().__init__()
        import timm   # imported lazily so this module stays importable without weights

        if backbone not in BACKBONES:
            raise KeyError(f"unknown backbone {backbone!r}; have {sorted(BACKBONES)}")
        if tokenizer_kind not in ("grid", "eegit", "topography"):
            raise KeyError(f"tokenizer_kind must be 'grid', 'eegit' or 'topography', "
                           f"got {tokenizer_kind!r}")
        style = {"nw": "region-time", "eegit_official": "time-region"}.get(style, style)
        if style not in ("region-time", "time-region"):
            raise KeyError(f"style must be 'region-time' or 'time-region' "
                           f"(legacy: 'nw', 'eegit_official'), got {style!r}")
        if timm_global_pool not in ("", "avg", "token"):
            raise KeyError(f"timm_global_pool must be '', 'avg' or 'token', got "
                           f"{timm_global_pool!r}")
        spec = BACKBONES[backbone]
        self.backbone_name = backbone
        self.d_model = spec["d_model"]
        self.pool = pool
        self.proj_to_joint = proj_to_joint
        self.tokenizer_kind = tokenizer_kind
        self.style = style

        self.vit = timm.create_model(
            spec["timm_name"], pretrained=pretrained, num_classes=0,
            global_pool=timm_global_pool,
        )
        self.n_blocks = len(self.vit.blocks)
        self.n_prefix = int(getattr(self.vit, "num_prefix_tokens", 1))

        # ---- input interface swap ------------------------------------------
        if tokenizer_kind in ("eegit", "topography"):
            # EEGiT's representation: the EEG "image" is patchified by the
            # PRETRAINED Conv2d. That conv is the +16.4 half of their ablation, so
            # it must stay trainable -- replacing it with a random projection is
            # exactly the defect this path exists to remove.
            #
            # `topography` uses the same mechanism with a different EEG geometry
            # (a real 2D scalp map instead of anatomical region bands); see
            # `ScalpTopographyTokenizer`. It is the structural tower's interface.
            if tokenizer_kind == "eegit":
                self.tokenizer = EEGPatchTokenizer(
                    channel_names=channel_names,
                    patch_size=patch_size,
                    n_patches_w=n_patches_w,
                    n_timepoints=n_timepoints,
                    zscore=zscore,
                    dropout=drop,
                    style=style,
                )
            else:
                self.tokenizer = ScalpTopographyTokenizer(
                    channel_names=channel_names,
                    patch_size=patch_size,
                    scalp_res=scalp_res,
                    n_time_bands=n_time_bands,
                    n_timepoints=n_timepoints,
                    band_channels=band_channels,
                    zscore=zscore,
                    dropout=drop,
                )
            self.patch_embed = self.vit.patch_embed
            for p in self.patch_embed.parameters():
                p.requires_grad_(True)
            # timm builds PatchEmbed with the training resolution baked in and
            # asserts that every input matches. The EEG image is deliberately not
            # the backbone's native size, so the strict check has to go. The relaxed
            # branch only requires divisibility by the patch size, which holds for
            # every geometry this code builds: both tokenizers set height and width
            # to an exact multiple of `patch_size` by construction. The conv itself
            # is size-agnostic.
            #
            # `strict_img_size` defaults to True on some checkpoints (MAE,
            # in21k) and False on others (DINOv3, which is built for
            # `dynamic_img_size`). Only the True case needs clearing; requiring
            # True -- as this code did before -- made every DINOv3 structural arm
            # impossible to construct, which is how this was found.
            if getattr(self.patch_embed, "strict_img_size", False):
                self.patch_embed.strict_img_size = False
            dst_grid = self.tokenizer.grid
        else:
            self.tokenizer = EEGTokenizer(
                channel_names=channel_names,
                d_model=self.d_model,
                grid_h=grid_h,
                grid_w=grid_w,
                n_time_windows=n_time_windows,
                n_timepoints=n_timepoints,
                dropout=drop,
            )
            dst_grid = (grid_h, grid_w * n_time_windows)

        # ---- positional embedding ------------------------------------------
        # Two families, and the difference is not cosmetic:
        #
        #   * learned `pos_embed` (in21k, MAE, CLIP, DINOv2) -- a table over the
        #     pretrained image grid, which has to be RESAMPLED onto the EEG grid.
        #     The resample is the interface, and getting its interpolation wrong is
        #     a real defect this file has already been bitten by (see
        #     `resample_pos_embed`).
        #   * RoPE (DINOv3) -- `pos_embed` is None and there is nothing to resample,
        #     because rotation is applied to Q/K inside attention from the grid
        #     SHAPE. Any grid works, which is why a non-square EEG geometry needs no
        #     positional surgery at all.
        #
        # The RoPE path is not free: `RotaryEmbeddingDinoV3` is consumed by the
        # blocks as an ARGUMENT (`blk(x, rope=rot_pos_embed)`), not from inside the
        # block. A block called as `blk(x)` -- which is what this file did before
        # this branch existed -- therefore runs with NO positional information at
        # all, silently, and still returns a tensor of the right shape. The
        # `rope_embed` buffer and the `rope_carrier` flag below exist so `forward`
        # can supply it.
        self.rope_carrier = bool(spec.get("rope", False)) or (self.vit.pos_embed is None)
        # Deliberately NOT `self.rope = self.vit.rope`: assigning it would register
        # the same module a second time and emit every `rope.*` key twice in
        # `state_dict()`, which then breaks `load_state_dict` strictness and doubles
        # the reported parameter count. `forward` reaches it through `self.vit`.
        if self.rope_carrier and getattr(self.vit, "rope", None) is None:
            raise RuntimeError(
                f"{backbone}: pos_embed is absent but the model carries no `rope` "
                f"module either, so this encoder would have no positional "
                f"information whatsoever. Refusing to build it.")

        pe = None if self.vit.pos_embed is None else self.vit.pos_embed.detach()
        src_grid = spec["src_grid"]
        if pe is None:
            prefix_pe = torch.zeros(1, self.n_prefix, self.d_model)
            grid_pe = torch.zeros(1, dst_grid[0] * dst_grid[1], self.d_model)
            src_grid = dst_grid
        else:
            if src_grid is None:
                n_grid_src = pe.shape[1] - self.n_prefix
                side = int(round(math.sqrt(n_grid_src)))
                if side * side != n_grid_src:
                    raise ValueError(f"cannot infer source grid from pos_embed {pe.shape[1]}")
                src_grid = (side, side)
            n_grid_src = src_grid[0] * src_grid[1]

            if pe.shape[1] == self.n_prefix + n_grid_src:
                prefix_pe, grid_pe = pe[:, : self.n_prefix], pe[:, self.n_prefix:]
            elif pe.shape[1] == n_grid_src:
                prefix_pe = torch.zeros(1, self.n_prefix, self.d_model)
                grid_pe = pe
            else:
                raise ValueError(
                    f"pos_embed length {pe.shape[1]} matches neither "
                    f"{self.n_prefix + n_grid_src} (prefix+grid) nor {n_grid_src} (grid)"
                )

            grid_pe = resample_pos_embed(grid_pe, src_grid, dst_grid)

        # timm assembles the input as `cat([cls_token, patches]) + pos_embed`, i.e.
        # the cls slot carries cls_token + pos_embed[0]. The loop below previously
        # put pos_embed[0] in that slot ALONE, which drops `vit.cls_token` -- a
        # pretrained parameter -- from the input entirely, and makes the cls slot
        # identical for every sample at initialisation. Same family as the missing
        # final LayerNorm: the interface did not reproduce what the pretrained
        # weights were trained against.
        ct = getattr(self.vit, "cls_token", None)
        if cls_token_prefix and ct is not None and self.n_prefix >= 1:
            add = torch.zeros_like(prefix_pe)
            add[:, 0] = ct.detach().reshape(-1)[: self.d_model]
            prefix_pe = prefix_pe + add

        # Same argument, for the register variants. timm builds
        # `cat([cls, *registers, patches])`; a backbone with registers therefore
        # has num_prefix_tokens > 1, and leaving those rows at zero would feed four
        # constant vectors where the checkpoint expects four learned ones -- an
        # interface that is wrong for exactly the rows it is quietest about. It
        # also shows up as `reg_token` being the one parameter that never receives
        # gradient, which is how this was found.
        rt = getattr(self.vit, "reg_token", None)
        if rt is not None and self.n_prefix > 1:
            n_reg = self.n_prefix - 1
            r = rt.detach().reshape(-1, self.d_model)
            if r.shape != (n_reg, self.d_model):
                raise ValueError(f"reg_token is {tuple(r.shape)} but the prefix needs "
                                 f"({n_reg}, {self.d_model})")
            add = torch.zeros_like(prefix_pe)
            add[:, 1:1 + n_reg] = r
            prefix_pe = prefix_pe + add

        self.register_buffer("prefix_pos", prefix_pe)
        self.register_buffer("grid_pos", grid_pe)
        self.src_grid = src_grid
        self.dst_grid = dst_grid

        # Post-norm on the pooled feature. timm's forward_features ends with
        # `x = self.norm(x)`, but the per-layer loop below calls the blocks
        # directly, so without this the alignment target was an UNNORMALISED
        # residual stream. That is wrong for the last block (norm is calibrated
        # for it, and it is the depth every winning arm used) and inconsistent
        # across depths, which matters because LayerFusion compares and blends
        # several depths against each other.
        self.pool_norm = pool_norm

        # ---- freezing -------------------------------------------------------
        # `pos_embed` and `cls_token` are consumed through `.detach()` above: their
        # values are baked into the `prefix_pos` / `grid_pos` buffers at
        # construction time, and nothing in the forward pass reads those parameters
        # again. They were nevertheless left with requires_grad=True, so every run
        # counted them as trainable and both `trainable_params` and the per-group
        # LR report overstated the live set -- by 0.6M on ViT-B/16 and 2.4M on
        # DINOv2-L. AdamW skips parameters with no gradient, so this changes no
        # number in any existing result; it changes what those results *say* about
        # the model, which is what they are read for.
        for attr in ("pos_embed", "cls_token", "reg_token"):
            p = getattr(self.vit, attr, None)
            if p is not None and p.requires_grad:
                p.requires_grad_(False)
        if freeze_all:
            for p in self.vit.parameters():
                p.requires_grad_(False)
        for i, blk in enumerate(self.vit.blocks):
            if i < freeze_blocks:
                for p in blk.parameters():
                    p.requires_grad_(False)
        # Blocks past the deepest layer anyone asked for cannot receive gradient --
        # `forward` stops there because nothing downstream reads them. Leaving them
        # requires_grad=True would put ~250M inert parameters on DINOv2-L into the
        # optimizer and into every reported trainable count. Recorded and warned
        # rather than silently applied, because "I asked for layers 6/12/18 and got
        # the parameter count of a 18-block model" is a surprising thing to discover
        # from a result file.
        self.deepest = (max(layers) if layers else self.n_blocks)
        if layers and self.deepest < self.n_blocks:
            for i, blk in enumerate(self.vit.blocks):
                if i + 1 > self.deepest:
                    for p in blk.parameters():
                        p.requires_grad_(False)
            print(f"[enc  ] {backbone}: deepest requested layer is {self.deepest} of "
                  f"{self.n_blocks}; blocks {self.deepest + 1}..{self.n_blocks} are "
                  f"never executed and are frozen (not counted as trainable)")

        self.drop = nn.Dropout(drop)

    # ------------------------------------------------------------------ utils
    def trainable_parameter_summary(self) -> tuple[int, int]:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return tr, tot

    @property
    def feat_dim(self) -> int:
        """Width of the pooled per-layer feature this encoder emits."""
        return self.d_model

    # ------------------------------------------------------------------ fwd
    def forward(self, eeg: torch.Tensor, layers: list[int],
                dense: bool = False) -> dict[int, torch.Tensor]:
        """eeg: (B, C, T). Returns {layer_index: feature} for layer_index in `layers`.

        Layer indices are 1-based block indices, so layer N means "after block N".

        `dense=False` (default): each value is the POOLED `(B, D)` feature. That is
        what a contrastive alignment target needs -- a global vector per sample.

        `dense=True`: each value is the full token sequence `(B, L, D)`, prefix tokens
        stripped. Needed by any head that wants the spatial arrangement of the tokens
        rather than a summary of them, which is the entire point of the structural
        tower's scalp-topography interface: there, the token grid *is* a 2D scalp
        map, and pooling it away would discard the layout the interface was built to
        preserve. Returning both from one pass keeps training and validation on a
        single code path.
        """
        bad = [l for l in layers if not (1 <= l <= self.n_blocks)]
        if bad:
            raise ValueError(f"layers {bad} out of range 1..{self.n_blocks}")

        if self.tokenizer_kind in ("eegit", "topography"):
            # (B, 3, H, W) -> pretrained Conv2d -> (B, N, D). timm's PatchEmbed
            # applies its own LayerNorm when configured, matching how the conv is
            # used on images during pretraining.
            img = self.tokenizer(eeg)
            tok = self.patch_embed(img)
            # timm's `PatchEmbed` returns channels-LAST `(B, H, W, C)` whenever
            # `output_fmt == "NHWC"`, which is what DINOv3 uses; the older NCHW
            # `(B, C, H, W)` is still the default for several checkpoints. Both are
            # in use, and guessing wrong is silent -- flattening NHWC as if it were
            # NCHW yields `(B, W*C, H)`, a tensor whose token count is the image
            # WIDTH and whose feature width is garbage. Dispatch on the module's own
            # declared format rather than on `ndim`, which is identical for both.
            if tok.ndim == 4:
                if getattr(self.patch_embed, "output_fmt", "NCHW") == "NHWC":
                    tok = tok.flatten(1, 2)                        # (B, H*W, C)
                else:
                    tok = tok.flatten(2).transpose(1, 2)           # (B, C, H, W)
            if tok.shape[1] != self.dst_grid[0] * self.dst_grid[1]:
                raise RuntimeError(
                    f"patch_embed produced {tok.shape[1]} tokens but the tokenizer's "
                    f"grid {self.dst_grid} implies {self.dst_grid[0] * self.dst_grid[1]}. "
                    f"The two interfaces disagree; this would otherwise surface as a "
                    f"mismatch against grid_pos much later.")
            if getattr(self.patch_embed, "norm", None) is not None:
                tok = self.patch_embed.norm(tok)
        else:
            tok = self.tokenizer(eeg)                              # (B, N, D)
        b = tok.shape[0]

        # ---- RoPE ----------------------------------------------------------
        # See `__init__`: on a RoPE backbone the positional information is an
        # ARGUMENT to each block, not a buffer added to the input. Computing it
        # here from the destination grid is what `dynamic_img_size` path in timm's
        # `_pos_embed` does, minus the `pos_embed` resample that does not exist on
        # this backbone. `get_embed(shape=...)` is the only supported way to ask
        # for it: the table is a property of the grid shape.
        rope = None
        if self.rope_carrier:
            rope = self.vit.rope.get_embed(shape=tuple(self.dst_grid)).to(tok.dtype)

        if self.n_prefix > 0:
            prefix = self.prefix_pos.expand(b, -1, -1).to(tok.dtype)
            x = torch.cat([prefix, tok + self.grid_pos.to(tok.dtype)], dim=1)
        else:
            x = tok + self.grid_pos.to(tok.dtype)

        x = self.drop(x)
        if getattr(self.vit, "patch_drop", None) is not None:
            x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        want = set(layers)
        deepest = max(want)
        out: dict[int, torch.Tensor] = {}
        for i, blk in enumerate(self.vit.blocks, start=1):
            x = blk(x, rope=rope) if rope is not None else blk(x)
            if i in want:
                if dense:
                    # Strip the cls/register prefix: they are global summary tokens,
                    # not part of the spatial grid the caller is going to reshape.
                    h = x[:, self.n_prefix:]
                    if self.pool_norm:
                        # Same argument as `_pool`: the per-layer loop bypasses
                        # timm's `forward_features`, so the final LayerNorm has to be
                        # applied here or the dense features are an unnormalised
                        # residual stream -- calibrated for nothing, and inconsistent
                        # across the depths LayerFusion blends.
                        norm = getattr(self.vit, "norm", None)
                        if norm is not None:
                            h = norm(h)
                    out[i] = h
                else:
                    out[i] = self._pool(x)
            if i >= deepest:
                # See OpenCLIPViTEEGEncoder.forward: blocks past the deepest
                # requested layer have no path to the loss, so computing them wastes
                # the most expensive part of the step and misreports them as
                # trainable. `__init__` freezes the same range.
                break
        return out

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_norm:
            # See the note in __init__: the per-layer loop bypasses timm's
            # forward_features, so the final LayerNorm has to be applied here.
            norm = getattr(self.vit, "norm", None)
            if norm is not None:
                x = norm(x)
        if self.pool == "cls":
            h = x[:, 0]
        elif self.pool == "mean":
            h = x[:, self.n_prefix:].mean(dim=1)
        else:
            raise ValueError(f"unknown pool {self.pool!r}")
        # timm's `forward_head` ends with `fc_norm` when `global_pool == "avg"` --
        # a LayerNorm created specifically for average pooling (`use_fc_norm` is
        # inferred from the pooling mode). The released EEGiT code builds its
        # encoder with `global_pool="avg"`, so its pooled feature is
        # `fc_norm(mean(norm(x)[:, prefix:]))`; without this line the official arm
        # would be reading one LayerNorm short of what it is a reproduction of.
        # `timm_global_pool=""` (every other arm) leaves `fc_norm` an Identity, so
        # this is a no-op there.
        fc = getattr(self.vit, "fc_norm", None)
        if fc is not None and not isinstance(fc, nn.Identity):
            h = fc(h)
        return h


class EEGiTProjectionHead(nn.Module):
    """The released EEGiT code's `ProjectionHead`, transcribed verbatim.

    ```python
    projected = self.projection(x)
    x = self.gelu(projected)
    x = self.fc(x)
    x = self.dropout(x)
    x += projected            # NOTE: the residual is the PRE-GELU projection
    return self.layer_norm(x)
    ```

    Three details that are easy to get wrong and all change the numbers, which is
    why this is a separate module rather than a flag on `_mlp`:

      * the residual branch is `projection(x)`, not `fc(gelu(projection(x)))` --
        so the block is `LayerNorm(fc(gelu(P(x))) + P(x))` and the non-linearity
        sits *inside* the residual;
      * `LayerNorm` is applied last, after the residual add, and is the only
        normalisation in the head;
      * dropout 0.5, on the residual branch only.

    The official code uses it for BOTH sides (`ProjectionHead(768, 768, 0.5)` for
    the EEG encoder and for the image encoder), which is why it is instantiated
    twice per run with different widths here.
    """

    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        h = self.gelu(projected)
        h = self.fc(h)
        h = self.dropout(h)
        h = h + projected
        return self.layer_norm(h)


def build_encoder(kind: str, channel_names: list[str], **kw) -> nn.Module:
    """Factory so the model code does not branch on backbone family.

    Two families:
      * open_clip vision tower (`openclip:*`) -- used for the semantic side, because
        its `proj` lands in the same joint space as the cached image features.
      * timm ViT (`timm:*`) -- DINOv2-L for the structure side, plus a ViT-B/16
        ImageNet-21K reference point (the exact EEGiT backbone).
    """
    common = dict(
        channel_names=channel_names,
        grid_h=kw.pop("grid_h", 7),
        grid_w=kw.pop("grid_w", 7),
        n_time_windows=kw.pop("n_time_windows", 4),
        n_timepoints=kw.pop("n_timepoints", 250),
        freeze_blocks=kw.pop("freeze_blocks", 0),
        freeze_all=kw.pop("freeze_all", False),
        pool=kw.pop("pool", "cls"),
        drop=kw.pop("drop", 0.1),
        tokenizer_kind=kw.pop("tokenizer_kind", "grid"),
        pool_norm=kw.pop("pool_norm", True),
        cls_token_prefix=kw.pop("cls_token_prefix", True),
        patch_size=kw.pop("patch_size", 16),
        n_patches_w=kw.pop("n_patches_w", 14),
        zscore=kw.pop("zscore", True),
        layers=kw.pop("layers", None),
        style=kw.pop("style", "region-time"),
        timm_global_pool=kw.pop("timm_global_pool", ""),
    )
    if kind.startswith("openclip:"):
        return OpenCLIPViTEEGEncoder(model_name=kind.split(":", 1)[1], **common)
    if kind.startswith("timm:"):
        name = kind.split(":", 1)[1]
        if name not in BACKBONES:
            raise KeyError(f"unknown timm backbone {name!r}; have {sorted(BACKBONES)}")
        return ViTEEGEncoder(
            backbone=name,
            pretrained=kw.pop("pretrained", True),
            proj_to_joint=kw.pop("proj_to_joint", False),
            **common,
        )
    raise KeyError(f"unrecognised encoder kind {kind!r}")


class LayerFusion(nn.Module):
    """Subject-aware multi-granularity fusion over intermediate layers.

    Follows the SAMGA formulation, with one deliberate difference: the global
    layer prior is initialised **centred on a layer index chosen by a scan**
    rather than at uniform, because a uniform prior spends most of its mass on
    depths known to be poor.

    weights = softmax( (global_prior + per_subject_residual * gate) / tau )
    final   = sum_k weights_k * proj_k(feat_k)

    Two modes, because the doc's phase 2 asks two separate questions:
      * `uniform` -- weights pinned at 1/k. "Do these layers carry complementary
        information at all?" Per-layer projections are still learned, so this is a
        real fusion and not just an average of the raw features.
      * `routed`  -- learnable weights plus the subject residual. "Can the model
        combine them better than equally?" Only worth asking if `uniform` wins.

    At evaluation the per-subject residual is bypassed (subject-agnostic
    inference), matching `--router_eval_mode global` in the SAMGA launcher.
    """

    def __init__(
        self,
        layers: list[int],
        d_in: int,
        n_subjects: int,
        d_out: int | None = None,
        prior_center: int | None = None,
        prior_strength: float = 1.0,
        tau: float = 1.0,
        layer_dropout: float = 0.1,
        subject_dropout: float = 0.3,
        projector: str = "linear",
        fusion_mode: str = "routed",
    ) -> None:
        super().__init__()
        if fusion_mode not in ("routed", "uniform", "none"):
            raise ValueError(f"fusion_mode must be 'routed', 'uniform' or 'none', "
                             f"got {fusion_mode!r}")
        self.layers = list(layers)
        self.k = len(self.layers)
        d_out = d_out or d_in
        self.tau = tau
        self.layer_dropout = layer_dropout
        self.subject_dropout = subject_dropout
        self.fusion_mode = fusion_mode
        if fusion_mode == "none":
            # Single-layer pass-through. Exists for arms that must reproduce an
            # external recipe exactly: the released EEGiT code reads one pooled
            # vector from the final block and puts it straight into a projection
            # head, so any extra module here -- even a single Linear -- is a
            # randomly-initialised transform that recipe does not have, and it
            # would be trained from scratch alongside pretrained weights.
            if self.k != 1:
                raise ValueError(f"fusion_mode 'none' needs exactly one layer, got "
                                 f"{self.k}: {self.layers}")
            self.proj = None
            self.register_buffer("_uniform_w", torch.ones(1), persistent=False)
            return

        # Per-layer projection into a common space. Linear by default: SAMGA's
        # ablation reports MLPs are worse here, which makes sense -- all inputs
        # already live in the same residual stream, so this only re-coordinates.
        if projector == "linear":
            self.proj = nn.ModuleList([nn.Linear(d_in, d_out, bias=False) for _ in self.layers])
        else:
            self.proj = nn.ModuleList([
                nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
                for _ in self.layers
            ])

        # Global prior over layers, and the per-subject residual. Both exist only in
        # "routed" mode: uniform fusion holds the weights at 1/k so that the fusion
        # question ("do these layers carry complementary information at all?") is
        # answered before the routing question ("can the model pick between them?").
        # Creating them in uniform mode would leave parameters that receive no
        # gradient, and a parameter that is reported as trainable but never moves is
        # a silent lie in the config record.
        if fusion_mode == "routed":
            if prior_center is None:
                prior_center = self.layers[self.k // 2]
            centre_idx = (self.layers.index(prior_center)
                          if prior_center in self.layers else self.k // 2)
            prior = torch.zeros(self.k)
            prior[centre_idx] = prior_strength
            self.global_prior = nn.Parameter(prior)
            self.subject_residual = nn.Embedding(n_subjects, self.k)
            nn.init.zeros_(self.subject_residual.weight)
        else:
            self.register_buffer("_uniform_w", torch.full((self.k,), 1.0 / self.k),
                                 persistent=False)

    def layer_weights(self) -> torch.Tensor:
        """Current global layer weights, for the run record.

        In routed mode this is the softmax of the learned prior; in uniform mode it
        is the fixed 1/k. Reported either way so a run that claims to use fusion
        shows what it actually fused.
        """
        if self.fusion_mode == "uniform":
            return self._uniform_w.detach()
        if self.fusion_mode == "none":
            return self._uniform_w.detach()
        return torch.softmax(self.global_prior.detach() / self.tau, dim=-1)

    def forward(
        self,
        feats: dict[int, torch.Tensor],
        subject_ids: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (fused (B, D), layer_weights (B, K))."""
        if self.fusion_mode == "none":
            h = feats[self.layers[0]]
            w = self._uniform_w.to(h.dtype).unsqueeze(0).expand(h.shape[0], -1)
            return h, w

        stacked = torch.stack([self.proj[i](feats[l]) for i, l in enumerate(self.layers)], dim=1)

        if self.fusion_mode == "uniform":
            # Fixed equal weights. This is the doc's "先均匀融合" step: it tests
            # whether several layers carry complementary information, independently
            # of whether a router can exploit it. If uniform fusion loses to the best
            # single layer, extra layers add nothing and routing has nothing to win.
            w = self._uniform_w.to(stacked.dtype).unsqueeze(0).expand(stacked.shape[0], -1)
            return (stacked * w.unsqueeze(-1)).sum(dim=1), w

        logits = self.global_prior.unsqueeze(0).expand(stacked.shape[0], -1).clone()
        if subject_ids is not None and training:
            res = self.subject_residual(subject_ids)
            if self.subject_dropout > 0:
                keep = (torch.rand_like(res) > self.subject_dropout).float()
                res = res * keep / (1.0 - self.subject_dropout)
            logits = logits + res

        if training and self.layer_dropout > 0:
            keep = (torch.rand_like(logits) > self.layer_dropout).float()
            # never zero out an entire row of weights
            dead = keep.sum(dim=-1, keepdim=True) == 0
            keep = torch.where(dead, torch.ones_like(keep), keep)
            logits = logits.masked_fill(keep == 0, float("-inf"))

        w = torch.softmax(logits / self.tau, dim=-1)
        return (stacked * w.unsqueeze(-1)).sum(dim=1), w
