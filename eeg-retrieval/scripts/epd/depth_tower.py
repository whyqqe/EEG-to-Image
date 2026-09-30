"""The structural tower as Depth Anything V2 with an EEG input interface.

What this file is, and why it is not another `StructureTower`
-----------------------------------------------------------
`StructureTower` owns a ViT trunk and a *randomly initialised* decoder that it
trains to emit a latent field. This module is the other design: a **pretrained
model whose output already is the structural control signal**, with only its
INPUT interface replaced by EEG patches -- which is the move EEGiT makes on the
semantic side, applied to a model that is about *where things are* rather than
*what things are*.

The model is Depth Anything V2 (Small). Its output is a monocular depth map, which
is also the canonical conditioning signal for ControlNet-depth. So the tower's
prediction is consumed by the decoder as a *spatial control* -- an additive,
per-location residual on the denoiser -- rather than as an img2img initialisation.
That distinction is the point of the redesign and it is measured, not assumed:

  * The previous structural branch emitted SDXL VAE latents that were decoded and
    fed to SDEdit at `strength 0.80`. Its own report shows it *cost* CLIP 2-way
    (-0.015) while buying PixCorr +0.021 over the semantic-only arm, and a
    `noise_sdedit` control that consumes NO EEG reached PixCorr 0.105 against the
    structural arm's 0.109. So the arm's PixCorr came from "an init image exists",
    not from what was in it.
  * An `img2img` init at high strength is an assertion that the whole field is
    known up to isotropic noise. EEG's spatial information is anisotropic and
    weak, so the mechanism imposes a prior the evidence does not support.
  * A ControlNet condition is a residual: scale 0 makes it a strict no-op, and
    the arm therefore cannot be worse than its own control except through the
    optimisation, which is what makes it measurable.

Why depth, given that a probe once said depth was not decodable
-------------------------------------------------------------
Because that probe measured the wrong thing, and the correction is in
`probe_targets.py --center-spatial`. The recorded verdict was

    depth @ 64x64   r(pred,gt) +0.1598   r(constant,gt) +0.5333   margin -0.3735

and `r(constant,gt)` there is `pearson_rows(mu_t, gt)`, which centres each ROW.
It is therefore the correlation between the SHAPE of the fit-set mean depth map
and the shape of each individual map -- and depth maps of COCO scenes nearly all
share one layout (far at the top, near at the bottom), so that shape is highly
correlated with every member of the set for reasons that have nothing to do with
EEG. VAE latents have no such shared layout, which is why their floor was only
+0.10..+0.17, and the comparison between the two spaces was never like for like.

The ridge also could not have expressed that mean map even in principle: the
design matrix is z-scored per feature, so it has mean zero and no intercept.
Against a target whose mean field carries most of its shape, "predict the mean"
is not available to it, and `lam*` was selected at 1e5 -- the heaviest shrink in
the grid -- which is what a fit that finds nothing at every lambda looks like.

Centring the target on the fit-set per-pixel mean removes exactly that free
component, and the constant predictor becomes the zero vector whose row Pearson r
is identically 0. Sub-08, all 63 channels, 150-concept val split:

    target     r(pred,gt)   floor   margin    test top1   chance 0.50%
    depth         +0.1872   0.007   +0.1800      4.50
    depth4        +0.1978  -0.024   +0.2219      5.00
    depth8        +0.2147  -0.008   +0.2227      4.00
    depth16       +0.1703  -0.010   +0.1802      5.00
    depth32       +0.1614  +0.006   +0.1558      5.50
    vae   @ 64    +0.1646  +0.001   +0.1639      6.00
    vae4          +0.3672  +0.005   +0.3623     11.50

So the EEG does carry the per-concept depth deviation, at every scale, and the
coarse end is where it is strongest -- which is also the end a ControlNet can use,
since a depth condition is consumed as a smooth low-frequency field.

The same correction explains the *old* collapse rather than merely permitting a
retry. The old depth head was trained with `latent_l1` against the UNCENTRED map,
and the only normalisation applied to it was a per-channel scalar (`vae_mean` is
reshaped to (C, 1, 1) in `AuxTargetDataset`), which removes a global offset and
leaves the mean FIELD intact. Under L1 the conditional median of a target whose
mean field dominates is the mean field itself, so a head that learned nothing but
the mean was close to L1-optimal, and the run reported exactly that: variance
ratio 0.0068 with a healthy-looking loss. The fix is not a cleverer decoder; it is
to stop asking the head for a component that is a constant, and to score it on the
part that is not.

What is deliberately NOT done here
----------------------------------
The EEG geometry is EEGiT's, unchanged: region bands along the image height, time
along the width, the patch grid taken from the checkpoint's own conv. It is a
fabricated 2D layout with no retinotopic meaning, and that is not a defect to fix
here -- EEGiT's ablation prices the pretrained conv + patch interface at +16.4
Top-1 with exactly this layout, so the prior transfers without the axes being
image-like. Two candidate "more meaningful" layouts were tried and rejected on
measurement rather than taste: a genuine scalp topography carried no linearly
decodable VAE content from EEG (margins negative, worse than a constant, in
`run_epd_struct_probe.sh`), and a DINOv3 trunk aimed at the same targets did not
beat using the semantic tower's own EEG patches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import EEGPatchTokenizer

# The checkpoint the depth targets were built with, so that "EEG -> depth" is
# scored against the same depth function that produced the ground truth. Using a
# different depth model for the target and the trunk would put a systematic bias
# between them that no amount of EEG side work can remove: the head would be asked
# to reproduce one model's conventions through another model's basis.
DA2_SMALL = "depth-anything/Depth-Anything-V2-Small-hf"


class DA2EEGEncoder(nn.Module):
    """Depth Anything V2 with EEG patches in place of RGB pixels.

    The only interface that changes is the input. Everything downstream --
    `backbone.encoder.layer`, the DPT `neck`, and `head` -- is the released
    checkpoint's, so the tower inherits a pretrained, multi-scale,
    fusion-learned path from features to a depth field. That path is the reason
    this design exists: `StructureTower` had to learn an equivalent mapping from
    scratch from a 1504-concept training set, and it did not have the capacity or
    the signal to do it.

    Two properties of the HF implementation make this work without surgery, and
    both were checked against the source rather than assumed:

      * `DepthAnythingForDepthEstimation.forward` derives the patch grid from the
        INPUT (`patch_height = height // patch_size`), not from
        `config.image_size`. So a 112 x 196 EEG image produces a 112 x 196 depth
        map, and no fixed-resolution assumption is violated.
      * `Dinov2Embeddings.interpolate_pos_encoding` is called from
        `embeddings.forward` and bicubically resamples the 37x37 pretrained
        positional grid onto whatever grid arrives. There is no silent
        truncation: the check is `num_patches == num_positions and height ==
        width` before an early return, and for any EEG geometry the first clause
        already fails, so the interpolation always runs. (This is the same class
        of defect as the DINOv3 RoPE omission found earlier in this project, where
        a block called as `blk(x)` ran with no positional information at all and
        still returned a correctly-shaped tensor. It was verified here for that
        reason.)
    """

    def __init__(
        self,
        channel_names: list[str],
        model_id: str = DA2_SMALL,
        patch_size: int = 14,
        n_patches_w: int = 14,
        n_timepoints: int = 250,
        style: str = "region-time",
        zscore: bool = True,
        drop: float = 0.0,
        freeze_blocks: int = 0,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        from transformers import DepthAnythingForDepthEstimation

        # `patch_size` is not a free parameter: the EEG image is patchified by the
        # checkpoint's own conv, so it has to be 14 or the conv sees a scale it was
        # never trained on. Asserted rather than documented because a mismatch here
        # produces a working forward pass with wrong numbers.
        if int(patch_size) != 14:
            raise ValueError(
                f"Depth Anything V2's patch conv is 14x14; got patch_size={patch_size}. "
                f"The EEG image must be patchified by the pretrained conv, so this is "
                f"not adjustable.")
        # Both layouts are accepted, and the choice is left to `--patch-style` so that
        # it matches whatever the semantic tower is using. That matters for what the
        # experiment can claim: with the same layout the two towers differ in their
        # PRIOR and their TARGET and nothing else, whereas a hardcoded orientation
        # here would make "the structural tower is the semantic tower with a
        # structure-focused prior" false in a way no result file would record.
        #
        # What is NOT allowed is a mismatch: `time-region` puts time on the height
        # axis (EEGiT's released layout, grid (n_time_patches, n_regions)) and
        # `region-time` the other way (grid (n_regions, n_time_patches)). Both are
        # well formed, and the depth map they are scored against is pooled from a
        # square target either way, so there is nothing to choose between them except
        # consistency -- which is why this is validated rather than defaulted.
        if style not in ("region-time", "time-region"):
            raise ValueError(f"style must be 'region-time' or 'time-region', got {style!r}")

        self.tokenizer = EEGPatchTokenizer(
            channel_names=channel_names,
            patch_size=int(patch_size),
            n_patches_w=int(n_patches_w),
            n_timepoints=int(n_timepoints),
            zscore=zscore,
            dropout=drop,
            style=style,
        )
        self.dst_grid = tuple(self.tokenizer.grid)
        self.model_id = model_id

        self.model = DepthAnythingForDepthEstimation.from_pretrained(
            model_id, local_files_only=local_files_only,
        )
        # ---- the output convention, which has to change -----------------------
        # `DepthAnythingDepthEstimationHead.forward` ends with
        # `activation2(x) * max_depth`, and for a relative-depth checkpoint
        # (`depth_estimation_type="relative"`, which is this checkpoint's setting)
        # `activation2` is a ReLU. So the released model cannot emit a negative
        # number, and the measured output was `min +0.0000 max +0.4217`.
        #
        # That is incompatible with the target by construction. `--struct-center`
        # subtracts the fit-set per-pixel mean, and the resulting field is 51%
        # negative with EVERY pixel needing both signs (per-pixel negative share
        # 0.40..0.69). A non-negative predictor of a zero-mean target spends its
        # entire range on the positive half, and MSE then drives it to the smallest
        # non-negative field it can -- which is what the run produced: variance
        # ratio 0.0598, per-sample r(own target) +0.0421 against the sample-
        # independent mean field's +0.0880, i.e. a margin of -0.0459, BELOW the
        # constant predictor the floor is defined against.
        #
        # The ReLU is a property of the RELATIVE-DEPTH TARGET SPACE, not of the
        # features-to-depth path this tower exists to inherit: it exists so that
        # "disparity" is expressed as a positive quantity. The path is
        # conv1 -> bilinear upsample -> conv2 -> ReLU -> conv3, and only the last
        # activation encodes that convention. Replacing it with the identity keeps
        # every pretrained weight and lets the same head emit a signed field, which
        # is also exactly what the export's arithmetic already assumes: it adds the
        # prediction to the mean field and clamps to [0, 1], so a signed deviation
        # is the input it is written for.
        #
        # Asserted rather than assumed, because a checkpoint swap that changed the
        # head class or the estimation type would otherwise silently reinstate the
        # constraint and reproduce the same collapse with no symptom in the loss.
        head = self.model.head
        if not isinstance(head.activation2, nn.ReLU):
            raise RuntimeError(
                f"expected the relative-depth head's ReLU at `head.activation2`, got "
                f"{type(head.activation2).__name__}. This tower's target is "
                f"mean-centred and therefore signed, so the output activation must be "
                f"replaced by the identity; re-check which checkpoint "
                f"{model_id!r} resolves to before removing this guard.")
        head.activation2 = nn.Identity()
        # The released head multiplies by `max_depth` AFTER the activation. With the
        # ReLU in place that scale maps the head's O(1) pre-activation onto the
        # checkpoint's relative-depth range; with the identity it would multiply an
        # already-unit-scale signed value by the same factor and start the run at 20x
        # the target's scale (target std 1.0 in the normalised space the loss uses).
        # Pinned to 1 so the head's initial output is O(1). Only the output scale
        # changes; no weight is touched.
        head.max_depth = 1.0
        self.signed_output = True
        # `backbone_config` is a `Dinov2Config` instance, not a dict, so it is
        # attribute access -- `["hidden_size"]` raises TypeError. It is also a nested
        # config object rather than the backbone module's own, which is why this reads
        # through `config` instead of `self.model.backbone.config`.
        self.d_model = int(self.model.config.backbone_config.hidden_size)
        self.n_blocks = len(self.model.backbone.encoder.layer)

        if freeze_blocks:
            k = min(int(freeze_blocks), self.n_blocks)
            for blk in self.model.backbone.encoder.layer[:k]:
                for p in blk.parameters():
                    p.requires_grad_(False)
            # Also freeze the input interface's own parameters, because they are
            # part of the representation being held fixed. The patch conv is NOT
            # frozen: it is the half of the interface the EEG actually flows
            # through, and EEGiT's ablation puts it at +16.4 Top-1.
            for p in self.model.backbone.embeddings.parameters():
                p.requires_grad_(False)

    def forward(self, eeg: torch.Tensor, dense: bool = False) -> torch.Tensor:
        """eeg: (B, C, T) -> (B, 1, H, W) depth, at the EEG image's own resolution.

        The field is SIGNED. See `__init__`: the checkpoint's relative-depth ReLU is
        replaced by the identity, because the supervised target is mean-centred and
        a non-negative predictor of a zero-mean target cannot represent half of it.

        `dense` is accepted and ignored so this encoder is drop-in against
        `ViTEEGEncoder`'s call signature, where it selects unpooled tokens. The
        depth map IS the dense readout here, so there is nothing to select.
        """
        img = self.tokenizer(eeg)                     # (B, 3, H, W)
        out = self.model(pixel_values=img)
        return out.predicted_depth.unsqueeze(1)       # (B, 1, H, W)


class DepthTower(nn.Module):
    """The structural tower: EEG -> a coarse depth field, for ControlNet-depth.

    The contract is `StructureTower`'s -- `forward` returns a dict carrying `vae`
    (the supervised field), so the training loop, the selection metric, the
    collapse gate and the export path all keep working without a second code path
    -- but the values in it are a depth field rather than VAE latents, and
    `out_hw` is the coarse scale the supervision uses rather than 64.

    Why the prediction is pooled before it is scored
    -----------------------------------------------
    The head emits a map at the EEG image's resolution (112 x 196 for the eight
    ThINGs-EEG regions at patch 14). That is 21952 dimensions per sample, and the
    probe's coarse ladder is unambiguous about which part of it is reachable: the
    margin is flat from 4x4 to 64x64 (+0.22 to +0.16) while the *dimension count*
    goes up by 256x, so scoring at full resolution would put almost all of the loss
    on coordinates the EEG cannot address and would let the head fit noise in the
    fine bins at no cost to the coarse ones. Pooling first makes the loss measure
    the component the probe actually found.

    `adaptive_avg_pool2d` on both sides rather than a fixed integer factor,
    because the head's grid (8 rows) and the target cache's grid (64) do not share
    a divisor with the supervision grid; adaptive pooling is defined for any pair
    of sizes and applies the same operator to prediction and target, so the two
    sides are compared in the same basis.
    """

    def __init__(
        self,
        encoder: DA2EEGEncoder,
        out_hw: int = 8,
        vae_ch: int = 1,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.out_hw = int(out_hw)
        self.vae_ch = int(vae_ch)
        if self.vae_ch != 1:
            raise ValueError(
                f"the depth tower emits one channel; got vae_ch={self.vae_ch}. If a "
                f"multi-channel structural target is wanted, it is a different target "
                f"and needs its own tower -- see the VAE ladder in `--struct-target`.")
        # Read by the export path's shape checks and by the tests, exactly as on
        # `StructureTower`. Kept as attributes of the tower rather than reachable
        # only through `encoder` because that indirection is what previously made
        # `grid_hw` raise AttributeError on one of two geometries.
        self.grid_hw = tuple(self.encoder.dst_grid)
        self.n_tokens = int(self.grid_hw[0] * self.grid_hw[1])
        # The map's resolution, exposed for the same reason `grid_hw` is: the export
        # path writes it as a conditioning image and the run record quotes it, and
        # reaching through `encoder.tokenizer` in both places is how those two drift.
        self.field_hw = (int(self.encoder.tokenizer.height),
                         int(self.encoder.tokenizer.width))
        self.layers: list[int] = []
        self.drop = nn.Dropout(drop)

    def forward(
        self, eeg: torch.Tensor, subject_ids: torch.Tensor | None = None,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        full = self.encoder(eeg)                                  # (B, 1, H, W)
        pooled = F.adaptive_avg_pool2d(full, (self.out_hw, self.out_hw))
        pooled = self.drop(pooled) if training else pooled
        # `field` is the full-resolution map and is what the export writes as the
        # ControlNet conditioning image; `vae` is the pooled field the loss is
        # defined on. They are the same function at two resolutions rather than two
        # heads, so a collapse in `vae` cannot be masked by a healthy-looking
        # `field`: it is the same tensor, averaged.
        return {
            "fused": F.adaptive_avg_pool2d(full, 1).flatten(1),
            "layer_w": None,
            "field": full,
            "grid": full,
            "vae": pooled,
        }
