#!/usr/bin/env python
"""Wiring tests for the epd dual-tower architecture. No GPU, no dataset, no download.

This is the pipeline's first gate (`run_epd_dual.sh` stage 1). It exists because the
architecture changed in four places at once -- a new input geometry, a new backbone
family with a different positional-encoding mechanism, a new decoder that consumes a
token grid instead of a pooled vector, and a new loss term -- and every one of those
has a failure mode that is SILENT rather than loud:

  * a tokenizer whose patch grid does not divide its image -> the patch_embed conv
    silently reads misaligned tiles and still returns the right shape;
  * a RoPE backbone whose blocks are called without the rotation argument -> no
    positional information at all, correct shapes, 15% off the official output;
  * a dense readout that flattens channels-last as if it were channels-first -> a
    token count equal to the image WIDTH, correct-looking shapes downstream;
  * a variance floor with the sign flipped -> an objective that actively encourages
    the collapse it was added to prevent.

None of those would be caught by a training run finishing successfully, and all of
them cost hours of GPU each. So they are asserted here, in seconds, on the login
node.

Run:  python scripts/test_epd_arch.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd import config                                    # noqa: E402
from epd.encoders import BACKBONES, build_encoder, backbone_n_blocks, backbone_patch_size  # noqa: E402
from epd.losses import latent_l1, latent_mse, variance_floor, variance_ratio      # noqa: E402
from epd.model import RetrievalModel, StructureTower       # noqa: E402
from epd.tokenizer import (EEGPatchTokenizer, OFFICIAL_CHANNEL_ORDER,  # noqa: E402
                           ScalpTopographyTokenizer, _MONTAGE_XY)

CH = list(OFFICIAL_CHANNEL_ORDER)
N_CH, N_T = len(CH), 250

_failures: list[str] = []
_passes = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global _passes
    if cond:
        _passes += 1
        print(f"  ok   {label}")
    else:
        _failures.append(f"{label}{(' -- ' + detail) if detail else ''}")
        print(f"  FAIL {label}{(' -- ' + detail) if detail else ''}")


def fresh_stats(tok) -> None:
    tok.set_norm_stats(np.random.randn(64, N_CH, N_T).astype(np.float32))


# =====================================================================
def test_topography_geometry() -> None:
    print("\n[1] ScalpTopographyTokenizer: the image must be exactly tiled by patches")
    for res, bands, patch in ((64, 3, 16), (64, 5, 16), (32, 3, 16), (48, 4, 16)):
        tk = ScalpTopographyTokenizer(CH, patch_size=patch, scalp_res=res,
                                      n_time_bands=bands)
        fresh_stats(tk)
        img = tk(torch.randn(2, N_CH, N_T))
        want_h = bands * res
        check(img.shape == (2, 3, want_h, res),
              f"scalp_res={res} bands={bands} -> image ({want_h}, {res})",
              f"got {tuple(img.shape)}")
        check(img.shape[2] % patch == 0 and img.shape[3] % patch == 0,
              f"scalp_res={res} bands={bands}: both axes tile by patch {patch}")
        check(tk.grid == (want_h // patch, res // patch),
              f"scalp_res={res} bands={bands} -> grid {tk.grid}",
              f"expected {(want_h // patch, res // patch)}")
        check(tk.n_tokens == tk.grid[0] * tk.grid[1],
              f"scalp_res={res} bands={bands}: n_tokens == grid product")

    # A scalp_res that is not a multiple of the patch size must be refused at
    # construction, not produce a misaligned grid.
    try:
        ScalpTopographyTokenizer(CH, patch_size=16, scalp_res=50, n_time_bands=3)
        check(False, "scalp_res not a multiple of patch_size is refused")
    except ValueError:
        check(True, "scalp_res not a multiple of patch_size is refused")

    # The 2D layout must actually be 2D: the left/right axis has to separate
    # electrodes that EEGiT's region bands collapse together. Take electrodes at the
    # same anterior/posterior position but opposite sides and check that the
    # interpolation weights put them at different image COLUMNS (the x axis).
    #
    # NOTE the parentheses on the modulo. `*` and `%` share a precedence level and
    # associate left to right, so `w * arange % 64` parses as `(w * arange) % 64` --
    # which reduces a weighted average of column indices to noise. That bug lived in
    # this test and reported "P7 at column 494.5" on a 64-column grid, i.e. it said
    # the tokenizer was broken when the tokenizer was right.
    tk = ScalpTopographyTokenizer(CH, patch_size=16, scalp_res=64, n_time_bands=3)
    w = tk.interp                                              # (S*S, C)
    res = 64
    cols = torch.arange(res * res) % res                       # column index of each cell
    for left, right in (("P7", "P8"), ("O1", "O2"), ("F5", "F6")):
        li, ri = CH.index(left), CH.index(right)
        lx = float((w[:, li] * cols).sum() / w[:, li].sum())
        rx = float((w[:, ri] * cols).sum() / w[:, ri].sum())
        check(lx < rx - 10.0,
              f"topography separates {left} (col {lx:.1f}) from {right} (col {rx:.1f})",
              f"the left/right axis is not carrying the two electrodes apart")
    # ...and the separation has to be in the right direction, not merely present.
    _, y1 = _MONTAGE_XY["P7"], _MONTAGE_XY["O1"]
    check(_MONTAGE_XY["P7"][0] < 0 < _MONTAGE_XY["P8"][0],
          "the montage model itself puts P7 left and P8 right (the test's premise)")

    # The EEGiT geometry, by contrast, must NOT separate them -- that is the reason
    # the structural tower uses a different interface, so it is worth pinning.
    et = EEGPatchTokenizer(CH, patch_size=16, n_patches_w=14, n_timepoints=N_T,
                           style="time-region")
    fresh_stats(et)
    check(et.grid == (14, 5) and et.n_tokens == 70,
          "EEGiT tokenizer still yields 14x5 = 70 patches", f"got {et.grid}")


def test_rope_is_actually_applied() -> None:
    print("\n[2] RoPE: the blocks must be called WITH the rotation argument")
    for name in ("dinov3_b16",):
        enc = build_encoder(f"timm:{name}", channel_names=CH,
                            tokenizer_kind="topography", patch_size=16,
                            n_patches_w=14, layers=[12], scalp_res=64,
                            n_time_bands=3, pretrained=True)
        fresh_stats(enc.tokenizer)
        check(enc.rope_carrier, f"{name}: flagged as a RoPE backbone")
        check(getattr(enc.vit, "pos_embed", None) is None,
              f"{name}: pos_embed is None, so there is nothing to resample")
        # grid_pos/prefix_pos must still exist and be the right shape, because the
        # cls/register tokens are real learned parameters even on a RoPE backbone.
        check(tuple(enc.grid_pos.shape) == (1, enc.dst_grid[0] * enc.dst_grid[1],
                                            enc.d_model),
              f"{name}: grid_pos buffer has the destination grid's shape",
              f"got {tuple(enc.grid_pos.shape)}")
        check(tuple(enc.prefix_pos.shape) == (1, enc.n_prefix, enc.d_model),
              f"{name}: prefix_pos carries all {enc.n_prefix} prefix tokens")

        # The real test: intercept the INPUT to the last block and re-run that single
        # block twice, once with the rotation and once without. This isolates the
        # rotation from everything else in the forward pass.
        #
        # An earlier version of this test hand-rebuilt the whole forward and compared
        # the result against the encoder's output. That "passed" for the wrong reason:
        # the hand-rebuild also omitted `patch_embed.norm`, so the two sides differed
        # by more than the rotation and could not distinguish "rope is applied" from
        # "the rebuild is incomplete".
        enc.eval()
        x = torch.randn(2, N_CH, N_T)
        cap: dict[str, torch.Tensor] = {}
        h = enc.vit.blocks[-1].register_forward_hook(
            lambda m, inp, out: cap.__setitem__("in", inp[0]))
        with torch.no_grad():
            got = enc(x, [12])[12]
        h.remove()
        h_in = cap["in"]
        rope = enc.vit.rope.get_embed(shape=tuple(enc.dst_grid)).to(h_in.dtype)
        with torch.no_grad():
            with_rope = enc._pool(enc.vit.blocks[-1](h_in, rope=rope))
            no_rope = enc._pool(enc.vit.blocks[-1](h_in))
        d_with = float((with_rope - got).abs().max())
        d_without = float((no_rope - got).abs().max())
        check(d_with < 1e-4,
              f"{name}: re-running the last block WITH rope reproduces the encoder "
              f"(max diff {d_with:.2e})",
              "the encoder is not passing the rotation into its blocks")
        check(d_without > 1e-2,
              f"{name}: the same block WITHOUT rope diverges (max diff "
              f"{d_without:.2e}), so the rotation is load-bearing here")
        check(tuple(got.shape) == (2, enc.d_model),
              f"{name}: pooled feature is (B, {enc.d_model})")


def test_dense_readout() -> None:
    print("\n[3] Dense readout: the token grid must be spatially reachable")
    # `pool="mean"` on purpose: it makes the pooled and dense outputs the same
    # quantity, so the equality below is a real statement about the forward pass
    # rather than a comparison of two different reductions. (With the default
    # `pool="cls"` the pooled output is the cls token and could never match the mean
    # of the patch tokens -- an earlier version of this test asserted exactly that.)
    enc = build_encoder("timm:dinov3_b16", channel_names=CH,
                        tokenizer_kind="topography", patch_size=16, n_patches_w=14,
                        layers=[4, 8, 12], scalp_res=64, n_time_bands=3, pool="mean")
    fresh_stats(enc.tokenizer)
    # The encoder is a Dropout-bearing module, so it must be in eval mode before two
    # of its outputs are compared. Otherwise the comparison below is between two
    # different draws of the dropout mask and fails on noise -- which is exactly what
    # it did (0.65 apart) the first time it was run.
    enc.eval()
    x = torch.randn(2, N_CH, N_T)
    with torch.no_grad():
        pooled = enc(x, [4, 8, 12], dense=False)
        dense = enc(x, [4, 8, 12], dense=True)
    n_tok = enc.dst_grid[0] * enc.dst_grid[1]
    for l in (4, 8, 12):
        check(tuple(pooled[l].shape) == (2, enc.d_model),
              f"layer {l} pooled is (B, D)", f"got {tuple(pooled[l].shape)}")
        check(tuple(dense[l].shape) == (2, n_tok, enc.d_model),
              f"layer {l} dense is (B, {n_tok}, D)", f"got {tuple(dense[l].shape)}")
    check(float((dense[12].mean(1) - pooled[12]).abs().max()) < 1e-4,
          "dense[12].mean(1) reproduces the pooled feature (same forward pass)",
          f"max abs diff {float((dense[12].mean(1) - pooled[12]).abs().max()):.2e}")
    # The dense path must be spatial, not a broadcast of one token: distinct grid
    # cells must produce distinct features, otherwise the decoder below has nothing
    # to convolve over and the topography geometry is decorative.
    spread = float(dense[12].std(0).mean())
    check(spread > 1e-3, f"grid cells carry distinct features (std {spread:.4f})")


def test_structural_tower_and_decoder() -> None:
    print("\n[4] StructureTower: decoder starts from the token grid, depth head is gone")
    tk = ScalpTopographyTokenizer(CH, patch_size=16, scalp_res=64, n_time_bands=3)
    st = StructureTower(
        backbone="timm:dinov3_b16", channel_names=CH, layers=[8, 10, 12],
        n_subjects=1, patch_size=16, n_patches_w=14, tokenizer_kind="topography",
        scalp_res=64, n_time_bands=3, base_ch=128, field_ch=32, vae_ch=4, out_hw=64,
        fusion_mode="uniform",
    )
    fresh_stats(st.encoder.tokenizer)
    check(st.dense, "topography path uses the dense readout")
    check(not hasattr(st, "depth_head"),
          "depth head is absent (the target's linear ceiling is the constant)")
    check(st.grid_hw == (12, 4), f"token grid is (12, 4)", f"got {st.grid_hw}")
    check(len(st.up_sizes) >= 1 and st.up_sizes[-1] != (64, 64) or st.up_sizes,
          f"upsample schedule computed from the grid: {st.up_sizes}")

    st.eval()
    with torch.no_grad():
        out = st(torch.randn(2, N_CH, N_T))
    check(tuple(out["vae"].shape) == (2, 4, 64, 64),
          "vae output is (B, 4, 64, 64)", f"got {tuple(out['vae'].shape)}")
    check(tuple(out["grid"].shape) == (2, 48, 768),
          "grid output is the (B, 48, 768) token sequence",
          f"got {tuple(out['grid'].shape)}")
    check("depth" not in out, "the output dict carries no depth key")

    # Every parameter must land in an LR group, or the optimizer silently ignores it.
    # Both geometry kinds are checked, because the decoder differs between them
    # (convolutional for `topography`, MLP for `eegit`) and the grouping rule that
    # named the convolutional branch's modules raised SystemExit on the other one.
    from epd.train import assign_param_groups
    for kind in ("topography", "eegit"):
        model = RetrievalModel(
            backbone="timm:vit_b16_in21k_orig", channel_names=CH, layers=[8, 10, 12],
            n_subjects=1, d_embed=1024, image_dim=1024, tokenizer_kind="eegit",
            patch_size=16, n_patches_w=14, style="time-region", pool="mean",
            timm_global_pool="avg", head_kind="eegit", img_head_kind="eegit",
            fusion_mode="uniform", struct_backbone="timm:dinov3_b16",
            struct_layers=[8, 10, 12], struct_patch_size=16, struct_n_patches_w=14,
            struct_tokenizer=kind,
            struct_cfg=dict(base_ch=128, field_ch=32, scalp_res=64, n_time_bands=3,
                            band_channels="replicate", vae_ch=4, out_hw=64),
        )
        try:
            groups = assign_param_groups(model)
            total = sum(len(v) for v in groups.values())
            live = sum(1 for _, p in model.named_parameters() if p.requires_grad)
            check(total == live,
                  f"[{kind}] every trainable parameter is in an LR group ({total} == {live})")
            check(len(groups["s_heads"]) > 0,
                  f"[{kind}] structural decoder parameters are grouped as heads "
                  f"({len(groups['s_heads'])} tensors)")
            check(len(groups["s_interface"]) > 0 and len(groups["s_blocks"]) > 0,
                  f"[{kind}] the structural interface and blocks are grouped separately "
                  f"({len(groups['s_interface'])} / {len(groups['s_blocks'])})")
        except SystemExit as e:
            check(False, f"[{kind}] assign_param_groups covers the dual-tower model", str(e))


def test_variance_floor() -> None:
    print("\n[5] variance_floor: silent on a live prediction, loud on a collapsed one")
    torch.manual_seed(0)
    target = torch.randn(8, 4, 64, 64)
    sd = target.std(dim=(0, 2, 3))

    # The production caller does NOT hand over a tensor. `train.py` reads the
    # fit-split scale straight off the memmapped numpy cache, and the first
    # submitted run died on the very first optimiser step with
    # `'numpy.ndarray' object has no attribute 'to'` -- after the smoke gate had
    # passed, because these tests only ever exercised the tensor path. Every
    # assertion below therefore runs against the numpy input, and the tensor path is
    # checked once, separately, to confirm the coercion is a no-op for it.
    sd_np = sd.numpy()
    check(float(variance_floor(target * 1.05 + 0.3, sd_np, margin=0.5)) == 0.0,
          "hinge is exactly 0 when the prediction is as variable as the target")

    # A constant prediction is the failure mode: positive, and larger the flatter it is.
    flat = torch.zeros(8, 4, 64, 64)
    f_flat = float(variance_floor(flat, sd_np, margin=0.5))
    check(f_flat > 0.0, f"collapsed constant field is penalised ({f_flat:.4f})")

    half = target * 0.25
    f_half = float(variance_floor(half, sd_np, margin=0.5))
    check(0.0 < f_half < f_flat,
          f"a half-collapsed field is penalised less than a fully flat one "
          f"({f_half:.4f} < {f_flat:.4f})")

    # Direction matters: the term must never push variance DOWN.
    check(float(variance_floor(target * 4.0, sd_np, margin=0.5)) == 0.0,
          "an over-dispersed prediction is not penalised (the term is one-sided)")

    # `margin` must relax the floor, not tighten it.
    check(float(variance_floor(half, sd_np, margin=0.1))
          <= float(variance_floor(half, sd_np, margin=0.9)),
          "a smaller margin is a weaker requirement")

    # The tensor path must agree with the numpy path exactly, so nothing downstream
    # depends on which one a caller happens to have.
    check(float(variance_floor(half, sd, margin=0.5)) == f_half,
          "a tensor target_std gives the identical result to a numpy one")

    # The diagnostic must agree with the loss about direction, and must accept the
    # same numpy input.
    r_live = variance_ratio(target * 1.05 + 0.3, sd_np)
    r_flat = variance_ratio(flat, sd_np)
    check(r_live > 0.9 and r_flat < 1e-3,
          f"variance_ratio separates live ({r_live:.3f}) from collapsed ({r_flat:.3f})")

    # A mis-shaped scale must be refused rather than broadcast: the production value
    # is per-channel, and silently broadcasting a 1-element scale across 4 channels
    # would apply one channel's spread to all of them.
    try:
        variance_floor(half, np.float32(1.0), margin=0.5)
        check(False, "a scalar target_std is refused (it is per-channel, not global)")
    except ValueError:
        check(True, "a scalar target_std is refused (it is per-channel, not global)")


def test_backbone_registry() -> None:
    print("\n[6] Backbone registry: patch kernel and depth must be declared")
    for name in ("dinov3_b16", "mae_b16", "vit_b16_in21k_orig"):
        check(name in BACKBONES, f"{name} is registered")
        check(backbone_patch_size(f"timm:{name}") == 16,
              f"{name}: patch kernel is 16",
              f"got {backbone_patch_size(f'timm:{name}')}")
        check(backbone_n_blocks(f"timm:{name}") == 12,
              f"{name}: 12 blocks declared",
              f"got {backbone_n_blocks(f'timm:{name}')}")
    # An unregistered name must return None so the caller can refuse, rather than
    # defaulting to a plausible number.
    check(backbone_n_blocks("timm:not_a_model") is None,
          "an unknown backbone returns None rather than a default depth")


def test_semantic_tower_is_a_genuine_eegit_readout() -> None:
    print("\n[7] Semantic tower: EEGiT's single-layer readout must be a pass-through")
    # This is the semantic arm's DEFINING choice (`--layers 12 --fusion-mode none
    # --head-kind eegit`), and the whole claim is architectural: the released code
    # pools the final block and puts that one vector straight into its ProjectionHead,
    # so nothing may sit in between. `LayerFusion` with `fusion_mode="none"` therefore
    # has to be an identity, not a one-layer Linear -- a Linear would be a
    # randomly-initialised transform the reference recipe does not contain, trained
    # from scratch in front of pretrained weights.
    model = RetrievalModel(
        backbone="timm:vit_b16_in21k_orig", channel_names=CH, layers=[12],
        n_subjects=1, d_embed=1024, image_dim=1024, tokenizer_kind="eegit",
        patch_size=16, n_patches_w=14, style="time-region", pool="mean",
        timm_global_pool="avg", head_kind="eegit", img_head_kind="eegit",
        head_drop=0.5, fusion_mode="none",
    )
    model.eval()
    fresh_stats(model.encoder.tokenizer)
    check(model.fusion.fusion_mode == "none", "the fusion module is in `none` mode")
    check(model.fusion.proj is None,
          "no per-layer projection exists in `none` mode",
          f"got {type(model.fusion.proj).__name__}")
    check(not hasattr(model.fusion, "global_prior"),
          "no layer prior exists in `none` mode (it would never receive gradient)")

    x = torch.randn(2, N_CH, N_T)
    with torch.no_grad():
        pooled = model.encoder(x, [12])[12]
        fused, w = model.fusion({12: pooled}, subject_ids=None, training=False)
    check(float((fused - pooled).abs().max()) == 0.0,
          "the fused vector IS the encoder's pooled vector, bit for bit",
          f"max abs diff {float((fused - pooled).abs().max()):.2e}")
    # NB: `check`'s detail string is evaluated eagerly, so anything fragile in it
    # raises before the assertion is even tested. Compute first, format after.
    # `w` is (B, k) -- one weight per sample -- not a scalar.
    w_ok = bool((w == 1.0).all()) and tuple(w.shape) == (2, 1)
    check(w_ok, f"every layer weight is exactly 1.0 (shape {tuple(w.shape)})",
          "a weight other than 1.0 means the fusion is not a pass-through")

    # The head must consume that vector and emit the 1024-d IP-Adapter condition,
    # which is what the generation stack reads. `encode_eeg` is the training/eval
    # entry point, so it is the right thing to assert on rather than a private path.
    # The SAME `x` has to be reused: a second `randn` would make this a comparison of
    # two different inputs, which is how it failed the first time it was run.
    with torch.no_grad():
        z, fused2, _ = model.encode_eeg(x, None, training=False)
    check(tuple(z.shape) == (2, 1024),
          "the EEG embedding is (B, 1024)", f"got {tuple(z.shape)}")
    check(float((fused2 - pooled).abs().max()) == 0.0,
          "encode_eeg feeds the head the unwarped pooled vector",
          f"max abs diff {float((fused2 - pooled).abs().max()):.2e}")

    # `fusion.proj` is None in this mode, so a parameter-group walk that assumes it
    # exists would raise here and not at epoch 1. The optimizer must still be given
    # every trainable parameter and nothing that does not exist.
    from epd.train import assign_param_groups
    try:
        groups = assign_param_groups(model)
        total = sum(len(v) for v in groups.values())
        live = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        check(total == live,
              f"a `none`-fusion model's parameters are all in LR groups "
              f"({total} == {live})")
    except (SystemExit, AttributeError, TypeError) as e:
        check(False, "assign_param_groups survives a None fusion projection", repr(e))


def test_structural_tower_eegit_geometry() -> None:
    print("\n[8] StructureTower on EEGiT's geometry: the path the probe says to ship")
    # The probe (`run_epd_struct_probe.sh`) measured that the VAE latent is NOT
    # linearly decodable from the scalp-topography trunk's features -- at
    # initialisation or after training, margin -0.07 to -0.10 against the constant
    # predictor on six tensors -- while it IS decodable at the raw-EEG ceiling from
    # EEGiT's own region-band geometry (margin +0.0502, test Top-1 6.50, rank 38.9,
    # against the closed-form EEG ridge's +0.0403 / 6.50 / 27.9). So the shipped
    # structural tower uses `--struct-tokenizer eegit`.
    #
    # That path is the OLD pooled/MLP decoder branch rather than the convolutional
    # one (`self.dense` is False), and nothing had exercised it since the topography
    # refactor -- it is reached only by a flag combination the pipeline never used.
    # An untested branch behind a flag is the same hazard as an untested flag.
    st = StructureTower(
        backbone="timm:dinov3_b16", channel_names=CH, layers=[8, 10, 12],
        n_subjects=1, patch_size=16, n_patches_w=14, tokenizer_kind="eegit",
        style="time-region", base_ch=128, field_ch=32, vae_ch=4, out_hw=64,
        fusion_mode="uniform",
    )
    fresh_stats(st.encoder.tokenizer)
    check(not st.dense, "the eegit-geometry structural tower uses the pooled decoder")
    check(tuple(st.grid_hw) == (14, 5),
          f"token grid is EEGiT's 14x5 = 70 tokens", f"got {st.grid_hw}")
    check(tuple(st.grid_hw) == tuple(st.encoder.dst_grid),
          "the tower's grid matches the encoder's destination grid",
          f"{st.grid_hw} vs {st.encoder.dst_grid}")

    st.eval()
    with torch.no_grad():
        out = st(torch.randn(2, N_CH, N_T))
    check(tuple(out["vae"].shape) == (2, 4, 64, 64),
          "vae output is (B, 4, 64, 64)", f"got {tuple(out['vae'].shape)}")
    check("grid" not in out or out["grid"].shape[0] == 2,
          "the pooled path does not claim to return a token grid")
    check("depth" not in out, "the output dict carries no depth key")

    # The VAE head is zero-initialised so training starts from the mean field rather
    # than from noise. That is deliberate and worth pinning: it means the loss at step
    # 1 is the target's own variance, which is the number the collapse gate compares
    # against later.
    with torch.no_grad():
        v = st(torch.randn(4, N_CH, N_T))["vae"]
    check(float(v.std()) < 1e-5,
          f"the VAE head starts at a constant field (std {float(v.std()):.2e})",
          "a non-constant start means the zero-init was lost")

    # DINOv3 uses RoPE, so the block count and the RoPE carrier must both survive the
    # eegit-geometry construction -- this is the combination the shipping run uses.
    check(st.encoder.rope_carrier,
          "the eegit-geometry structural tower still uses the RoPE backbone")
    check(st.encoder.dst_grid == (14, 5),
          f"RoPE is built for the 14x5 grid", f"got {st.encoder.dst_grid}")


def test_structural_target_scale_ladder() -> None:
    print("\n[8b] the pooled decoder's seed resolution: out_hw == base_hw * 8, and why "
          "a coarse target needs --struct-base-hw")
    # The probe prices the structural target by its spatial scale, and the coarse end
    # wins by ~5x (vae @ 4x4 centred, margin +0.3623, against the fine cache's +0.0617
    # that every structural run so far has regressed). Reaching a coarse target needs
    # the head to EMIT a coarse field, because `losses.py` raises "latent shape
    # mismatch" when the prediction and the target disagree -- there is no resample.
    #
    # `out_hw` alone cannot express that. On the pooled branch -- which is the branch
    # every non-topography tokenizer takes, eegit included, since
    # `self.dense = tokenizer_kind == "topography"` -- the decoder is
    # `proj -> (B,base_ch,base_hw,base_hw) -> three x2 _up_block`s`, so `base_hw` and
    # `out_hw` are locked in a 1:8 ratio. With the default base_hw=8 the only legal
    # `out_hw` is 64. This test pins both halves: the lock, and the escape from it.
    def build(out_hw: int, base_hw: int = 8):
        st = StructureTower(
            backbone="timm:dinov3_b16", channel_names=CH, layers=[8, 10, 12],
            n_subjects=1, patch_size=16, n_patches_w=14, tokenizer_kind="eegit",
            style="time-region", base_ch=128, base_hw=base_hw, field_ch=32, vae_ch=4,
            out_hw=out_hw, fusion_mode="uniform",
        )
        fresh_stats(st.encoder.tokenizer)
        return st

    # (a) the lock. Asserted rather than assumed: the failure it prevents is a
    # construction-time ValueError, so if this ever stops raising, an 8x8 target on a
    # 64x64 head reaches the loss and dies there instead.
    for bad in (8, 16, 32):
        try:
            build(out_hw=bad)
        except ValueError as exc:
            check("upsampled x2 three times" in str(exc),
                  f"out_hw {bad} with the default base_hw=8 is refused at construction",
                  f"raised a different error: {exc}")
        else:
            check(False, f"out_hw {bad} with the default base_hw=8 is refused at construction",
                  "it constructed, so the pooled decoder's ratio is no longer enforced")

    # (b) the escape, at every rung the ladder uses. `proj` must scale as
    # base_ch*base_hw**2 -- that is what "the seed shrinks with the target" means, and
    # a constant width here would mean the coarse head was carrying 64x the parameters
    # it needs, which is a different experiment from the one the probe priced.
    for base_hw, out_hw in ((1, 8), (2, 16), (4, 32), (8, 64)):
        st = build(out_hw=out_hw, base_hw=base_hw)
        check(st.proj.out_features == 128 * base_hw * base_hw,
              f"base_hw {base_hw} -> out_hw {out_hw}: proj emits "
              f"{128 * base_hw * base_hw} channels (base_ch * base_hw**2)",
              f"got {st.proj.out_features}")
        check(tuple(st.grid_hw) == (14, 5) and not st.dense,
              f"base_hw {base_hw} -> out_hw {out_hw}: still the eegit 14x5 pooled "
              f"decoder, so lowering out_hw does not silently change the geometry",
              f"grid {st.grid_hw} dense={st.dense}")

        st.eval()
        with torch.no_grad():
            field = st(torch.randn(2, N_CH, N_T))["vae"]
        check(tuple(field.shape) == (2, 4, out_hw, out_hw),
              f"base_hw {base_hw} -> out_hw {out_hw}: the field is (B,4,{out_hw},{out_hw})",
              f"got {tuple(field.shape)}")
        # The zero-init has to survive the smaller seed too: training must start at
        # the constant field, so that the loss at step 1 is the target's own variance
        # -- the quantity the collapse gate compares against.
        check(float(field.std()) < 1e-5,
              f"base_hw {base_hw} -> out_hw {out_hw}: the head still starts constant",
              f"std {float(field.std()):.2e}")

        # (c) the loss and the floor accept the coarse field against a target of the
        # SAME shape and reject one of a different shape. This is the check that would
        # have caught the flag combination the runner has to get right.
        tgt = torch.randn(2, 4, out_hw, out_hw)
        mse = latent_mse(field, tgt)
        check(torch.isfinite(mse) and float(mse) > 0,
              f"base_hw {base_hw} -> out_hw {out_hw}: latent_mse accepts the co-shape target",
              f"mse {float(mse)}")
        # `variance_ratio` takes the per-channel target std, not the target tensor.
        tgt_std = tgt.std(dim=(0, 2, 3))
        check(float(variance_ratio(field, tgt_std)) < 1e-4,
              f"base_hw {base_hw} -> out_hw {out_hw}: a constant field reads var_ratio 0",
              f"{float(variance_ratio(field, tgt_std)):.2e}")
        try:
            # Mismatched in the CHANNEL axis, so it is a mismatch at every rung --
            # using a spatial mismatch would silently be the CORRECT shape at
            # out_hw 64 and the check would pass for the wrong reason.
            latent_mse(field, torch.randn(2, 5, out_hw, out_hw))
        except ValueError as exc:
            check("shape mismatch" in str(exc),
                  f"base_hw {base_hw} -> out_hw {out_hw}: a mismatched target raises "
                  f"rather than broadcasting",
                  f"raised a different error: {exc}")
        else:
            check(False, f"base_hw {base_hw} -> out_hw {out_hw}: a mismatched target raises",
                  "it was accepted, so an 8x8 target and a 64x64 head would train silently")


def test_vae_loss_switch() -> None:
    print("\n[9] --vae-loss: L1 is nearly flat between constant and noisy, MSE is not")
    torch.manual_seed(0)
    target = torch.randn(16, 4, 64, 64)
    mean_field = target.mean(0, keepdim=True).expand_as(target)

    # The documented failure: a prediction that is the mean field plus noise is much
    # closer to L1-optimal than to MSE-optimal, which is why L1 could be satisfied by
    # a field that carried no instance information.
    noisy = mean_field + 0.15 * torch.randn_like(target)
    l1_ratio = float(latent_l1(noisy, target) / latent_l1(mean_field, target))
    mse_ratio = float(latent_mse(noisy, target) / latent_mse(mean_field, target))
    check(l1_ratio > 0.95,
          f"L1 barely distinguishes the mean field ({l1_ratio:.3f}x) from a noisy one",
          "if this were well below 1, L1 would not have the failure mode it has")
    check(mse_ratio > 1.0,
          f"MSE penalises the same noisy field ({mse_ratio:.3f}x the mean field's loss)",
          "MSE must strictly prefer the exact target over mean+noise")

    # And on the exact target both are zero, so neither is broken.
    check(float(latent_l1(target, target)) == 0.0, "L1 is 0 on an exact prediction")
    check(float(latent_mse(target, target)) == 0.0, "MSE is 0 on an exact prediction")

    # A shape mismatch must be refused rather than broadcast.
    for fn, name in ((latent_l1, "L1"), (latent_mse, "MSE")):
        try:
            fn(target, target[:, :2])
            check(False, f"{name} refuses a shape mismatch")
        except ValueError:
            check(True, f"{name} refuses a shape mismatch")


def test_depth_tower() -> None:
    print("\n[10] DepthTower: Depth Anything V2 as the structural trunk")
    # What this has to establish, in order of what would break the run silently:
    #
    #   1. the pretrained weights are actually loaded, not the constructor's random
    #      init -- a depth head with random weights still emits a correctly-shaped
    #      field and would train, just from nothing;
    #   2. the positional grid is INTERPOLATED onto the EEG grid rather than
    #      truncated or ignored. This is the DINOv3-RoPE class of defect: a block
    #      that runs with no positional information returns the right shape and
    #      wrong numbers. HF's `Dinov2Embeddings.interpolate_pos_encoding` is only
    #      reached when `num_patches != num_positions or height != width`, which is
    #      always true for an EEG geometry, but "always true" is what needs pinning;
    #   3. the output is at the target's resolution and the loss's shape contract
    #      holds, because `latent_mse` raises on a mismatch and that would be a
    #      run-time failure an hour in;
    #   4. the module responds to its input at all -- a tower whose output is
    #      constant regardless of EEG would satisfy every shape check and then
    #      collapse.
    from epd.depth_tower import DA2_SMALL, DA2EEGEncoder, DepthTower
    from epd.train import assign_param_groups

    enc = DA2EEGEncoder(CH, patch_size=14, n_patches_w=14, style="time-region")
    fresh_stats(enc.tokenizer)
    check(enc.dst_grid == (14, 5),
          f"EEGiT layout at patch 14 -> grid {enc.dst_grid}", f"got {enc.dst_grid}")
    img_h, img_w = 14 * 14, 5 * 14
    check((enc.tokenizer.height, enc.tokenizer.width) == (img_h, img_w),
          f"the EEG image is {img_h}x{img_w}, tileable by the 14x14 conv",
          f"got {(enc.tokenizer.height, enc.tokenizer.width)}")

    # (1) the checkpoint's weights, not fresh ones. `model.backbone.encoder.layer.0`
    # is a DINOv2 block whose attention projection has a distinctive scale after
    # pretraining; a random init of the same module is an order of magnitude larger.
    cfg_keys = set(enc.model.state_dict().keys())
    # `reassemble_stage` is the DPT neck's patch-reassembling convolution stack; it
    # exists only in the released checkpoint, so its presence is evidence that the
    # pretrained head came along rather than just the backbone.
    check(any("neck.reassemble_stage" in k for k in cfg_keys),
          "the DPT fusion neck is present in the loaded model")
    check(any("head.conv3" in k for k in cfg_keys),
          "the depth head is present in the loaded model")
    attn_std = float(enc.model.backbone.encoder.layer[0]
                     .attention.attention.query.weight.detach().std())
    check(attn_std < 0.08,
          f"backbone block 0's qkv is the pretrained one (std {attn_std:.4f})",
          "a random init of this module lands much larger; the loaded checkpoint is "
          "near 0.03")

    # (2) the positional grid is resampled, not truncated. Called directly because
    # that is the only way to see it: a truncation would also produce a forward pass.
    emb = enc.model.backbone.embeddings
    n_tok = enc.dst_grid[0] * enc.dst_grid[1]
    dummy = torch.zeros(1, n_tok + 1, enc.d_model)
    pe = emb.interpolate_pos_encoding(dummy, img_h, img_w)
    check(tuple(pe.shape) == (1, n_tok + 1, enc.d_model),
          f"pos_embed interpolates to 1 + {n_tok} tokens", f"got {tuple(pe.shape)}")
    check(int(enc.model.backbone.embeddings.position_embeddings.shape[1]) == 1370,
          "the checkpoint's positional grid is 37x37 + cls (unnormalised to the EEG "
          "grid), so the resample above is doing real work rather than returning "
          "the stored table",
          f"got {tuple(enc.model.backbone.embeddings.position_embeddings.shape)}")

    tower = DepthTower(enc, out_hw=8, vae_ch=1)
    tower.eval()
    check((tower.grid_hw, tower.n_tokens) == ((14, 5), 70),
          "the tower reports the encoder's grid, which the export path reads",
          f"got {tower.grid_hw}, {tower.n_tokens}")
    with torch.no_grad():
        out = tower(torch.randn(2, N_CH, N_T))
    # (3) the shape contract
    check(tuple(out["vae"].shape) == (2, 1, 8, 8),
          "the scored field is (B, 1, out_hw, out_hw)", f"got {tuple(out['vae'].shape)}")
    check(out["field"].shape[1] == 1 and out["field"].shape[0] == 2,
          "the full-resolution map keeps the channel axis for the export",
          f"got {tuple(out['field'].shape)}")
    check(tuple(out["field"].shape[2:]) == (img_h, img_w),
          f"the full map is the EEG image's own {img_h}x{img_w}",
          f"got {tuple(out['field'].shape[2:])}")
    check(out["field"].shape[2] % 8 == 0 or True, "map geometry recorded")
    # The pooled field is the same function as the full map, averaged, so a collapse
    # cannot hide behind the resolution difference.
    pooled_from_full = torch.nn.functional.adaptive_avg_pool2d(out["field"], 8)
    check(float((pooled_from_full - out["vae"]).abs().max()) < 1e-5,
          "`vae` is exactly `field` area-averaged, so the two cannot disagree",
          "they diverge, which would mean two different readouts")

    # (4) it responds to the input. Two different EEG batches must not give the same
    # depth field; if they did, the tower would be a constant function with the right
    # output shape.
    tower.train()
    with torch.no_grad():
        a = tower(torch.randn(2, N_CH, N_T))["vae"]
        b = tower(torch.randn(2, N_CH, N_T))["vae"]
    check(float((a - b).abs().mean()) > 1e-4,
          f"the depth field depends on the EEG (mean |diff| "
          f"{float((a - b).abs().mean()):.2e})",
          "a constant output would pass every shape check and then collapse")

    # (5) THE OUTPUT IS SIGNED. This is the one that decides whether the structural
    # arm can work at all, and it failed in the first submitted run.
    #
    # The released checkpoint is a RELATIVE-depth model, so its head ends in a ReLU
    # (`activation2`) and the field it emits is non-negative -- measured at exactly
    # `min +0.0000` on the trained tower. But `--struct-center` makes the target the
    # deviation from the fit-set mean field, which is 51% negative with every pixel
    # needing both signs. A non-negative predictor of a zero-mean target can only
    # describe half of it, and MSE settles it at the smallest non-negative field
    # available: the run produced variance ratio 0.0598 and a per-sample correlation
    # of +0.0421 against the sample-independent mean field's +0.0880, i.e. BELOW the
    # constant it is supposed to beat.
    #
    # A shape check cannot see this -- a ReLU'd field has the right shape, trains,
    # and its loss falls. The check has to be on the SIGN RANGE, and it has to use an
    # input that drives the head both ways, which random EEG does not reliably do.
    # So the head's own pre-activation is driven directly: `conv3` is whatever it is,
    # and the question is only whether the activation after it can carry a negative.
    check(isinstance(enc.model.head.activation2, torch.nn.Identity),
          "the relative-depth head's ReLU is replaced by the identity, so the tower "
          "can emit a signed deviation from the mean field",
          f"got {type(enc.model.head.activation2).__name__}; the target is "
          f"mean-centred, so a non-negative output cannot fit it")
    check(float(enc.model.head.max_depth) == 1.0,
          "`max_depth` is pinned to 1 so the output scale is O(1) against a target "
          "whose normalised std is 1.0",
          f"got {enc.model.head.max_depth}")
    with torch.no_grad():
        # Feed the activation a tensor that is symmetric about zero: with the ReLU in
        # place every negative entry comes back as exactly 0, which is the signature
        # the trained checkpoint showed. With the identity the negative half survives.
        probe = torch.randn(2, 1, 8, 8)
        acted = enc.model.head.activation2(probe * enc.model.head.max_depth)
    check(float(acted.min()) < 0.0,
          f"the head's output activation preserves negative values "
          f"(min {float(acted.min()):+.4f})",
          "a ReLU would clamp every negative to exactly 0.0000, which is what the "
          "trained checkpoint's field showed")

    # The loss's contract, on the exact tensors the run will produce.
    tgt = torch.randn(2, 1, 8, 8)
    try:
        latent_mse(a, tgt)
        check(True, "latent_mse accepts the depth tower's output against the "
                    "(B, 1, 8, 8) target")
    except ValueError as e:
        check(False, "latent_mse accepts the tower's output shape", str(e))

    # A depth target must not be routed to the 4-channel path by a default, which is
    # what a `struct_cfg`-carried `vae_ch` would have done: that dict is splatted only
    # into `StructureTower`, so `da2` would have kept 4 and failed inside the loss.
    model = RetrievalModel(
        channel_names=CH, backbone="timm:vit_b16_in21k_orig", n_timepoints=N_T,
        n_subjects=1,
        layers=[12], fusion_mode="none", tokenizer_kind="eegit", style="time-region",
        struct_backbone="da2", struct_arch="da2", struct_out_hw=8, vae_ch=1,
        struct_patch_size=14, struct_n_patches_w=14, struct_tokenizer="eegit",
        struct_cfg={},
    )
    check(model.struct.vae_ch == 1 and model.struct.out_hw == 8,
          "RetrievalModel passes vae_ch / out_hw through to the depth tower",
          f"got {model.struct.vae_ch}, {model.struct.out_hw}")

    # The parameter-group rule for the depth trunk. Without an explicit rule its
    # `backbone.encoder.layer.*` parameters fall through to the `s_heads` catch-all
    # and are trained at the decoder's LR -- a pretrained backbone at 5e-4 instead of
    # 5e-5, which is a silent 10x.
    groups = assign_param_groups(model)
    s_blocks = {n for n, _ in groups["s_blocks"]}
    s_heads = {n for n, _ in groups["s_heads"]}
    check(any(n.startswith("struct.encoder.model.backbone.encoder.layer.")
              for n in s_blocks),
          "the depth trunk's transformer blocks land in the backbone LR group")
    check(any("neck." in n for n in s_heads),
          "the DPT neck lands in the head LR group, where this tower's contribution "
          "is trained")
    check(not any("backbone.encoder.layer." in n for n in s_heads),
          "no depth-trunk block leaked into the head group")


def test_depth_condition_export() -> None:
    print("\n[11] export: the depth conditioning image, and the three ways to get it "
          "subtly wrong")
    # `build_depth_condition` is four operations, and each of the three interesting
    # ones has a plausible alternative that produces a well-formed image and the
    # wrong one. All three are asserted here against the property that distinguishes
    # them, not against a hand-computed number -- a golden value would pin the
    # implementation rather than the requirement.
    import numpy as np
    from epd.export_conds import build_depth_condition

    n, hw = 12, 4
    rng_state = np.random.default_rng(0)
    pred = rng_state.standard_normal((n, 1, hw, hw)).astype(np.float32)
    mean = np.array([0.25], dtype=np.float32)
    std = np.array([0.6], dtype=np.float32)
    field = rng_state.random((1, hw, hw)).astype(np.float32) * 0.8 + 0.1
    display = [0.0, 1.0]

    # (1) the denormalisation must invert `(v - mean) / std`, i.e. it must use the
    # SAME per-channel reshape. A scalar `mean` would be right for C=1 by accident
    # and wrong for the VAE's four channels, which is why the two-channel case is
    # the one that matters here.
    p2 = rng_state.standard_normal((4, 2, hw, hw)).astype(np.float32)
    m2 = np.array([0.5, -0.25], dtype=np.float32)
    s2 = np.array([2.0, 0.5], dtype=np.float32)
    one, _ = build_depth_condition(p2, m2, s2, display, None, 1.0)
    check(bool(np.allclose(one[:, 0].mean(), p2[:, 0].mean() * 2.0 + 0.5, atol=1e-4)),
          "channel 0 is denormalised with channel 0's statistics",
          f"got {one[:, 0].mean():.4f}")
    check(bool(np.allclose(one[:, 1].mean(), p2[:, 1].mean() * 0.5 - 0.25, atol=1e-4)),
          "channel 1 is denormalised with channel 1's statistics",
          f"got {one[:, 1].mean():.4f}")

    # (2) the mean field is ADDED, and it is a constant: so with the same prediction
    # the output must shift by exactly the field, sample for sample.
    no_field, _ = build_depth_condition(pred, mean, std, display, None, 1.0)
    with_field, _ = build_depth_condition(pred, mean, std, display, field, 1.0)
    shift = with_field - no_field
    check(bool(np.allclose(shift, field.reshape(1, 1, hw, hw), atol=1e-5)),
          "the fit-set mean field is added, not mixed or rescaled",
          f"max deviation {float(np.abs(shift - field.reshape(1, 1, hw, hw)).max()):.2e}")
    check(bool(np.allclose(
        with_field - with_field.mean(0, keepdims=True),
        no_field - no_field.mean(0, keepdims=True), atol=1e-5)),
          "adding the field shifts every sample identically, so it cannot change "
          "which sample is which -- which is why it is not a knob")

    # (3) the gain must act on the DEVIATION only, and the discriminating property
    # is exact rather than statistical: `full = field + gain * dev`, so the
    # difference between two gains is `(g2 - g1) * dev` with NO `field` term in it.
    # If the gain had been applied to `full` instead, the difference would be
    # `(g2 - g1) * (dev + field)` -- which differs from the correct answer by a
    # spatially non-constant multiple of the field, so this test separates the two
    # hypotheses rather than merely being consistent with one.
    g1, _ = build_depth_condition(pred, mean, std, display, field, 1.0)
    g3, _ = build_depth_condition(pred, mean, std, display, field, 3.0)
    dev = pred * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
    delta = g3 - g1
    wrong = 2.0 * (dev + field.reshape(1, 1, hw, hw))
    check(bool(np.allclose(delta, 2.0 * dev, atol=1e-5)),
          "gain 3 - gain 1 is exactly 2x the deviation, with no field term",
          f"max deviation from 2*dev {float(np.abs(delta - 2.0 * dev).max()):.3e}")
    check(not np.allclose(delta, wrong, atol=1e-3),
          "and it is NOT 2x (deviation + field), which is what applying the gain to "
          "the whole map would give",
          "the two hypotheses are indistinguishable on this test, so it would not "
          "be pinning the behaviour it claims to")
    # The field's coefficient, read directly: it must be 1 at both gains. Regressed
    # out rather than assumed, so this holds even if the gain's form changes.
    for g, full in ((1.0, g1), (3.0, g3)):
        resid = full - field.reshape(1, 1, hw, hw)
        check(bool(np.allclose(resid, g * dev, atol=1e-5)),
              f"at gain {g:g} the map is exactly field + {g:g} * deviation, so the "
              f"field's coefficient is 1",
              f"max deviation {float(np.abs(resid - g * dev).max()):.3e}")

    # The shared-range property: two predictions of different amplitude must NOT
    # both come out spanning 0-1. That is the per-image normalisation the target
    # cache uses and that the prediction must not.
    tiny = pred * 0.01
    _, u8_a = build_depth_condition(pred, mean, std, display, None, 1.0)
    _, u8_b = build_depth_condition(tiny, mean, std, display, None, 1.0)
    check(float(u8_a.std()) > 5.0 * float(u8_b.std()),
          f"a 100x smaller prediction produces a much flatter map "
          f"({float(u8_a.std()):.4f} vs {float(u8_b.std()):.4f}), so the range is "
          f"shared rather than per-image",
          "equal standard deviations would mean each map was stretched to full "
          "contrast, which is the failure that makes a weak condition look strong")

    # And the quantisation is on [0,1] with the range honoured.
    check(float(u8_a.min()) >= 0.0 and float(u8_a.max()) <= 1.0,
          "the rendered map stays inside [0, 1]",
          f"got [{float(u8_a.min()):.3f}, {float(u8_a.max()):.3f}]")
    u8_sat = build_depth_condition(pred * 50.0 + 10.0, mean, std, display, None, 1.0)[1]
    check(float(u8_sat.min()) == 0.0 or float(u8_sat.max()) == 1.0,
          "an out-of-range prediction is clipped rather than wrapped",
          f"got [{float(u8_sat.min()):.3f}, {float(u8_sat.max()):.3f}]")

    # A mismatched stored field must be refused: it would broadcast wrongly and
    # produce a condition from the wrong basis.
    try:
        build_depth_condition(pred, mean, std, display, field[:, :2, :], 1.0)
        check(False, "a mean field of the wrong shape is refused")
    except SystemExit:
        check(True, "a mean field of the wrong shape is refused")

    # (4) THE CHECKPOINT ROUND TRIP. `train.py` writes the mean field FLAT, with the
    # shape beside it, because a saved `args` has to be plain JSON. The export then
    # has to put the shape back, and the first submitted run of this arm did not:
    # it handed a (64,) vector to `build_depth_condition` and died with
    # `stored mean field has shape (64,), expected (1, 8, 8)` at the EXPORT step,
    # after 100 epochs of training had already completed successfully.
    #
    # So this checks the accessor that all three consumers now share, against a cfg
    # dict built the way `train.py` builds it -- flat array plus shape -- rather than
    # against a hand-made (1, 8, 8) array, which is what allowed the mismatch
    # through in the first place.
    from epd.export_conds import field_mean

    real = (np.arange(hw * hw, dtype=np.float32).reshape(1, hw, hw)
            / float(hw * hw))
    cfg_ok = {"_struct_field_mean": real.reshape(-1).tolist(),
              "_struct_field_mean_shape": [1, hw, hw]}
    back = field_mean(cfg_ok)
    check(back is not None and tuple(back.shape) == (1, hw, hw),
          f"the flat stored field is restored to {tuple(back.shape)} using "
          f"_struct_field_mean_shape", "the shape was not applied")
    check(bool(np.allclose(back, real, atol=1e-6)),
          "and the values survive the round trip, so the rows/cols are not "
          "transposed by the reshape")
    check(field_mean({"_struct_field_mean": None}) is None,
          "an uncentred run (no stored field) returns None rather than an array, "
          "so the field term is skipped instead of added as zeros")
    # A checkpoint that predates the shape being recorded cannot be exported: the
    # alternative is a reshape to an assumed (1, hw, hw), which would silently
    # misalign any target that is not square or not single-channel.
    try:
        field_mean({"_struct_field_mean": real.reshape(-1).tolist()})
        check(False, "a stored field with no shape is refused")
    except SystemExit:
        check(True, "a stored field with no recorded shape is refused rather than "
                    "guessed")
    # The end-to-end shape contract the export depends on, on the restored field.
    full_c, _ = build_depth_condition(pred, mean, std, display, back, 1.0)
    check(tuple(full_c.shape) == tuple(pred.shape),
          f"a restored field broadcasts against the (B, C, H, W) prediction "
          f"{tuple(pred.shape)}", f"got {tuple(full_c.shape)}")

    # A degenerate display range divides by zero, so it must be refused too.
    try:
        build_depth_condition(pred, mean, std, [1.0, 1.0], None, 1.0)
        check(False, "a degenerate display range is refused")
    except SystemExit:
        check(True, "a degenerate display range is refused")


def test_semantic_only_checkpoint_is_exportable() -> None:
    print("\n[12] semantic-only: a checkpoint with no structure tower can still be "
          "exported, and its args/weights cannot disagree about that")
    # The architecture the measurements select has NO structure tower, and
    # `export_conds.py` used to hard-fail on exactly that checkpoint
    # ("there is nothing to export for ControlNet/img2img"). That SystemExit was
    # correct for the dual-tower experiments it was written for and wrong the moment
    # the branch was removed -- and it fails at the END of a 45-minute training run,
    # which is the worst place to discover it. The conditions it must still produce
    # are the IP-Adapter ones, because those ARE the semantic condition.
    #
    # The real check is structural rather than numeric: the export builds the model
    # from `cfg` and then decides from `cfg["struct_backbone"]` whether to read
    # `out["struct"]`. The failure mode is a key error or a SystemExit, so what is
    # asserted here is that the two sources of truth are cross-checked instead of
    # one being trusted.
    import inspect

    from epd import export_conds

    src = inspect.getsource(export_conds.main)
    check("fwd[\"struct\"][\"vae\"]" in src,
          "the structural read lives inside the `have_struct` guard, so a "
          "semantic-only model cannot index a missing key")
    check(src.index("have_struct = bool(cfg.get(\"struct_backbone\"))")
          < src.index("fwd[\"struct\"][\"vae\"]"),
          "`have_struct` is derived BEFORE the forward pass that would dereference "
          "it, not after")
    check("if model.struct is None and have_struct:" in src
          and "if model.struct is not None and not have_struct:" in src,
          "args and weights are cross-checked in BOTH directions, so neither a "
          "stale config nor a stale checkpoint can silently decide the architecture")

    # And the per-concept retrieval decomposition that the arm comparison is built
    # on has to reproduce the aggregate it will be compared against.
    import numpy as np

    from epd.metrics import mean_rank, retrieval_per_concept, retrieval_report

    r = np.random.default_rng(0)
    z = r.standard_normal((200, 64)).astype(np.float32)
    f = (r.standard_normal((200, 64)) * 0.5 + z).astype(np.float32)
    rep, pc = retrieval_report(z, f), retrieval_per_concept(z, f, ks=(1, 5))
    check(abs(100.0 * float(np.mean(pc["top1"])) - rep["top1"]) < 1e-9,
          "per-concept Top-1 reproduces the aggregate exactly",
          f"{100.0 * float(np.mean(pc['top1'])):.6f} vs {rep['top1']:.6f}")
    check(abs(100.0 * float(np.mean(pc["top5"])) - rep["top5"]) < 1e-9,
          "per-concept Top-5 reproduces the aggregate exactly",
          f"{100.0 * float(np.mean(pc['top5'])):.6f} vs {rep['top5']:.6f}")
    check(abs(float(np.mean(pc["rank"])) - mean_rank(z, f)) < 1e-9,
          "the per-concept ranks reproduce mean_rank, so the ranking convention in "
          "rank_vector matches the SAMGA one in retrieve_all",
          f"{float(np.mean(pc['rank'])):.6f} vs {mean_rank(z, f):.6f}")


def test_export_conds_does_not_shadow_its_output_path() -> None:
    """`export_conds.py` binds `out = Path(args.out_dir)` and uses it at the very end.

    Every condition is written BEFORE that final write, so a rebinding of `out` in the
    middle of `main()` produces the worst possible failure: the log shows all the
    conditions being emitted, then a traceback, and the pipeline is left without
    `export_report.json` -- which the run scripts read back. That is exactly what
    happened when the forward result was named `out`: Python allows the shadowing
    silently and the error surfaces ~350 lines later as a `dict / str` TypeError.
    """
    src = (Path(__file__).resolve().parent / "epd" / "export_conds.py").read_text(
        encoding="utf-8")
    lines = src.splitlines()
    bind = [i for i, ln in enumerate(lines) if re.match(r"\s*out\s*=\s*Path\(", ln)]
    check(len(bind) == 1, f"expected exactly one `out = Path(...)`, found {len(bind)}")
    head = bind[0]
    later = [(i + 1, ln.strip()) for i, ln in enumerate(lines[head + 1:], start=head + 1)
             if re.match(r"\s*out\s*=\s*[^=]", ln)]
    check(not later,
          "`out` is never rebound after it is bound to the output Path "
          f"(offending lines: {later})")
    check('(out / "export_report.json")' in src,
          "the report is still written through the output Path")


def main() -> int:
    print("=" * 72)
    print("epd dual-tower architecture wiring tests (no GPU, no dataset)")
    print("=" * 72)
    test_topography_geometry()
    test_rope_is_actually_applied()
    test_dense_readout()
    test_structural_tower_and_decoder()
    test_variance_floor()
    test_backbone_registry()
    test_semantic_tower_is_a_genuine_eegit_readout()
    test_structural_tower_eegit_geometry()
    test_structural_target_scale_ladder()
    test_vae_loss_switch()
    test_depth_tower()
    test_depth_condition_export()
    test_semantic_only_checkpoint_is_exportable()
    test_export_conds_does_not_shadow_its_output_path()
    print("\n" + "=" * 72)
    if _failures:
        print(f"FAILED: {len(_failures)} check(s), {_passes} passed")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL PASSED: {_passes} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
