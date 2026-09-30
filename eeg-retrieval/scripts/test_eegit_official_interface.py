"""Does `--patch-style eegit_official` actually reproduce the released EEGiT code?

The claim under test is narrow and checkable: for the same z-scored EEG input, the
EEG image this repo builds must be bit-comparable to the image
`third_party/EEGiT/base/data_eeg.py` builds, and the pooled feature + head must be
the same function the official `EEGVitEncoder` computes.

Checking it by eye is not enough. The two implementations differ in ways that
produce the same SHAPE and therefore the same "it ran" signal:

  * axis order         (official: H=time, W=regions; our older style: the transpose)
  * region order       (official: frontal -> occipital; ours: the reverse)
  * electrode order    (official: dataset order; ours: sorted by montage x)
  * interpolation      (official: one 2D bilinear over (time, electrode);
                        ours: 1D along electrodes, then a separate time resample)
  * pooling            (official: `global_pool='avg'` -> timm creates an `fc_norm`
                        LayerNorm after the mean; ours: mean of the normed tokens)

A 70-token image of the right shape comes out of all of these. So the test is
numeric equality against a transcription of the official functions, not a shape
assertion.

Runs on CPU with `pretrained=False` where possible, because the failure this is
guarding against is a silent interface difference, not a missing checkpoint.

Usage:  python scripts/test_eegit_official_interface.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# The reference model needs pretrained ViT-B/16 weights. `/home` on this cluster is
# full, so an unset HF cache makes this test die with ENOSPC before it checks
# anything -- which looks like a code failure. Point at the project cache the
# pipeline scripts use unless the caller already chose one.
import os  # noqa: E402

os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

from nwret.encoders import BACKBONES, EEGiTProjectionHead, build_encoder  # noqa: E402
from nwret.tokenizer import (  # noqa: E402
    EEGIT_REGIONS_OFFICIAL,
    OFFICIAL_CHANNEL_ORDER,
    EEGPatchTokenizer,
)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILED.append(name)


# ---------------------------------------------------------------------------
# A transcription of the released code, copied from `base/data_eeg.py` and
# `base/eeg_backbone.py` with only the names changed. Deliberately kept as close
# to the original as possible (including the loop-carried `result`), because a
# "cleaned up" version would test my understanding of the official code rather
# than the official code.
# ---------------------------------------------------------------------------
def official_use_kinematic(data: torch.Tensor) -> torch.Tensor:
    """`EEGDataset.use_kinematic` on already-z-scored EEG. In: (B,C,T) Out: (B,C,T,C')"""
    data = data.permute(0, 2, 1)
    data = data.unsqueeze(-1).repeat(1, 1, 1, 3)          # (B, T, C, 3)
    return official_spatial_interpolate(data)


def official_spatial_interpolate(input_tensor: torch.Tensor) -> torch.Tensor:
    """`EEGDataset.spatial_interpolate`. In: (B,T,C,3) Out: (B,T,P*R,3)"""
    device = input_tensor.device
    channel_index = {c: i for i, c in enumerate(OFFICIAL_CHANNEL_ORDER)}
    segmentation = [
        [channel_index[c] for c in members]
        for _, members in EEGIT_REGIONS_OFFICIAL
    ]
    interpolated_blocks = []
    result = None
    for block_indices in segmentation:
        block = input_tensor[:, :, block_indices, :]
        block_reshaped = block.permute(0, 3, 1, 2)
        interpolated = F.interpolate(
            block_reshaped,
            size=(224, 16),
            mode="bilinear",
            align_corners=False,
        )
        interpolated = interpolated.permute(0, 2, 3, 1)
        interpolated_blocks.append(interpolated)
        result = torch.cat(interpolated_blocks, dim=2)
    return result.to(device)


def main() -> int:
    torch.manual_seed(0)
    n_ch = len(OFFICIAL_CHANNEL_ORDER)
    b, t = 4, 250

    print("[1] EEG patch image: ours vs a transcription of the released code")
    eeg = torch.randn(b, n_ch, t)
    tok = EEGPatchTokenizer(
        channel_names=list(OFFICIAL_CHANNEL_ORDER), patch_size=16, n_patches_w=14,
        n_timepoints=t, zscore=False, dropout=0.0, style="eegit_official",
    )
    tok.eval()
    ours = tok(eeg)                                        # (B, 3, H, W)

    ref = official_use_kinematic(eeg).permute(0, 3, 1, 2)   # (B, 3, T', P*R)
    check("shape", tuple(ours.shape) == tuple(ref.shape),
          f"ours {tuple(ours.shape)} vs official {tuple(ref.shape)}")
    check("layout is (time, regions), not (regions, time)",
          tuple(ours.shape) == (b, 3, 224, 80),
          f"got {tuple(ours.shape)}; official img_size=(224, 5*16)=(224,80)")
    check("values match the official construction",
          torch.allclose(ours, ref, atol=1e-6),
          f"max abs diff {float((ours - ref).abs().max()):.3e}")

    print("[2] the layouts are genuinely different, not two names for one thing")
    tok_nw = EEGPatchTokenizer(
        channel_names=list(OFFICIAL_CHANNEL_ORDER), patch_size=16, n_patches_w=14,
        n_timepoints=t, zscore=False, dropout=0.0, style="nw",
    )
    tok_nw.eval()
    old = tok_nw(eeg)
    check("nw token count identical (70), so only the arrangement differs",
          tok_nw.n_tokens == tok.n_tokens == 70, f"{tok_nw.n_tokens} vs {tok.n_tokens}")
    check("nw grid is the transpose",
          tuple(tok_nw.grid) == (5, 14) and tuple(tok.grid) == (14, 5),
          f"nw {tok_nw.grid} vs official {tok.grid}")
    # Compare in a common layout before asserting the values differ: the two styles
    # emit transposed tensors, so a raw `allclose` raises on the shape rather than
    # returning the answer the check is about.
    old_common = old.permute(0, 1, 3, 2)                    # (B,3,224,80) == official layout
    check("the two styles produce the same shape once transposed",
          tuple(old_common.shape) == tuple(ours.shape),
          f"{tuple(old_common.shape)} vs {tuple(ours.shape)}")
    check("the two images differ (a flag that changed nothing would be a lie)",
          not torch.allclose(old_common, ours, atol=1e-4),
          "nw and eegit_official produced the same image")
    # Region IDENTITY, not just "the numbers differ". Both styles have five regions
    # and both would produce a well-formed image if one of them ordered them
    # wrongly, so check the names.
    nw_names = [n for n, _, _ in tok_nw.region_specs]
    off_names = [n for n, _, _ in tok.region_specs]
    check("region order: official anterior->posterior, nw the reverse",
          off_names == ["frontal", "central", "temporal", "parietal", "occipital"]
          and nw_names == off_names[::-1],
          f"nw {nw_names} official {off_names}")
    # And electrode order within a region: official keeps the dataset's channel
    # order, nw re-sorts by montage x. On the central region those disagree.
    def member_names(t):
        idx_of = {c: i for i, c in enumerate(t.channel_names)}
        rev = {v: k for k, v in idx_of.items()}
        return {n: [rev[i] for i in idxs] for n, idxs, _ in t.region_specs}

    nw_m, off_m = member_names(tok_nw), member_names(tok)
    check("electrode order within `central` differs between the two styles",
          nw_m["central"] != off_m["central"],
          f"both {nw_m['central']}")
    check("official `central` keeps the dataset's channel order",
          off_m["central"] == ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
                               "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                               "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
          f"{off_m['central']}")

    print("[3] z-score is applied to the raw signal, as official does")
    # `alldataset_mean_std` runs on (N, C, T) with mean/std over (trials, time) --
    # i.e. one scalar per channel, computed BEFORE the patch image is built.
    tok2 = EEGPatchTokenizer(
        channel_names=list(OFFICIAL_CHANNEL_ORDER), patch_size=16, n_patches_w=14,
        n_timepoints=t, zscore=True, dropout=0.0, style="eegit_official",
    )
    tok2.eval()
    x = torch.randn(16, n_ch, t)
    tok2.set_norm_stats(x.numpy().astype(np.float64))
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    z = (x - mean) / std
    check("z-scored input reproduces official mean/std",
          torch.allclose(tok2(x), official_use_kinematic(z).permute(0, 3, 1, 2),
                         atol=1e-5), "z-score statistics differ")

    print("[4] pooled feature + head reproduce `EEGVitEncoder`")
    import timm
    enc = build_encoder(
        "timm:vit_b16_in21k_orig",
        channel_names=list(OFFICIAL_CHANNEL_ORDER),
        tokenizer_kind="eegit", patch_size=16, n_patches_w=14, n_timepoints=t,
        zscore=False, layers=[12], pool="mean", pool_norm=True,
        style="eegit_official", timm_global_pool="avg", drop=0.0,
    )
    enc.eval()
    check("encoder's patch_embed kernel matches the tokenizer's patch size",
          tuple(enc.patch_embed.proj.weight.shape[-2:]) == (16, 16),
          f"{tuple(enc.patch_embed.proj.weight.shape[-2:])}")
    with torch.no_grad():
        feats = enc(eeg, [12])
    check("12 layers -> one feature", list(feats) == [12], f"{list(feats)}")

    # The reference is built EXACTLY as `EEGVitEncoder.__init__` builds it:
    #   timm.create_model(model_name, pretrained=True, num_classes=0,
    #                     global_pool="avg", img_size=(224, patch_size * 5))
    # The `img_size` argument is load-bearing and easy to drop: it is what makes
    # timm interpolate the 14x14 pretrained pos_embed down to the 14x5 grid at
    # construction. Our encoder does that interpolation itself
    # (`resample_pos_embed`), so if the two disagree this comparison is the only
    # thing that would notice.
    ref_vit = timm.create_model(
        BACKBONES["vit_b16_in21k_orig"]["timm_name"], pretrained=True,
        num_classes=0, global_pool="avg", img_size=(224, 16 * 5),
    )
    ref_vit.eval()
    check("official pos_embed is the 14x5 grid, i.e. img_size did its job",
          tuple(ref_vit.pos_embed.shape[1:]) == (1 + 70, 768),
          f"{tuple(ref_vit.pos_embed.shape)}")
    # Compare against the tensor timm actually FEEDS the blocks, not against the raw
    # `pos_embed`. timm's `_pos_embed` builds `cat([cls_token, patches]) + pos_embed`,
    # so the cls slot is `cls_token + pos_embed[:, 0]`; our `prefix_pos` folds the
    # cls_token in at construction time. Comparing to `pos_embed` directly therefore
    # reports a difference of exactly |cls_token| (9.38 here) even when the grid rows
    # -- the only rows that get resampled -- agree bit for bit.
    ref_input_pos = torch.cat(
        [ref_vit.pos_embed[:, :1] + ref_vit.cls_token.reshape(1, -1),
         ref_vit.pos_embed[:, 1:]], dim=1).detach()
    our_pos = torch.cat([enc.prefix_pos, enc.grid_pos], dim=1).detach()
    check("our pos_embed matches the official one for the same source weights",
          torch.allclose(ref_input_pos, our_pos, atol=1e-5),
          f"max abs diff {float((ref_input_pos - our_pos).abs().max()):.3e}")
    check("the resampled grid rows agree EXACTLY (antialias is the whole point)",
          float((ref_vit.pos_embed[:, 1:] - enc.grid_pos).abs().max()) == 0.0,
          f"grid max abs diff "
          f"{float((ref_vit.pos_embed[:, 1:] - enc.grid_pos).abs().max()):.3e}; "
          f"a nonzero value here means the resize is not timm's")
    with torch.no_grad():
        y = ref_vit(ours)                                  # (B, 768)
    check("pooled feature == official forward_features+forward_head(fc_norm|avg)",
          torch.allclose(feats[12], y, atol=1e-5),
          f"max abs diff {float((feats[12] - y).abs().max()):.3e}")
    # `global_pool='avg'` does NOT equal a bare mean over the patch tokens: timm
    # inserts `fc_norm` (a LayerNorm) after the mean. Check that it is actually
    # applied, by its defining property -- the output is per-sample standardised
    # across the feature axis. A bare mean of post-norm tokens would not be.
    check("the pooling really is mean-then-LayerNorm (output is standardised)",
          bool((y.mean(dim=-1).abs() < 1e-4).all()
               and ((y.std(dim=-1) - 1.0).abs() < 1e-2).all()),
          f"mean|.|={float(y.mean(dim=-1).abs().max()):.3e} "
          f"std={float(y.std(dim=-1).min()):.4f}..{float(y.std(dim=-1).max()):.4f}")
    check("with global_pool='avg' timm makes `norm` an Identity and `fc_norm` a "
          "LayerNorm -- so `pool_norm` must NOT be relied on for the final norm",
          isinstance(ref_vit.norm, torch.nn.Identity)
          and isinstance(ref_vit.fc_norm, torch.nn.LayerNorm),
          f"norm={type(ref_vit.norm).__name__} fc_norm={type(ref_vit.fc_norm).__name__}")

    print("[5] the official projection head")
    head = EEGiTProjectionHead(768, 1024, dropout=0.0).eval()
    with torch.no_grad():
        h = head(y)
    check("output width is 1024, not 768",
          h.shape == (b, 1024), f"{tuple(h.shape)}")
    # residual is the PRE-GELU projection: reconstruct explicitly
    with torch.no_grad():
        p = head.projection(y)
        h2 = head.layer_norm(head.fc(head.dropout(head.gelu(p))) + p)
    check("residual branch is projection(x), not fc(gelu(projection(x)))",
          torch.allclose(h, h2, atol=1e-6), "head does not match the official formula")
    with torch.no_grad():
        p = head.projection(y)
        alt = head.layer_norm(head.fc(head.gelu(p)) + head.fc(head.gelu(p)))
    check("the two residual forms are not accidentally equal (the test has teeth)",
          not torch.allclose(h, alt, atol=1e-4), "")

    print("[6] both tower styles agree on token count, which is what patch_embed needs")
    check("70 tokens at patch 16 -> a 5x14 grid; the conv sees 80x224 in one case "
          "and 224x80 in the other, and both tile exactly",
          ours.shape[-1] % 16 == 0 and ours.shape[-2] % 16 == 0, f"{tuple(ours.shape)}")

    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} check(s): {FAILED}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
