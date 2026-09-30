#!/usr/bin/env python3
"""Unit-ish checks that do not require GPU."""

from __future__ import annotations

import torch

from eeg_brainit.models import BITCrossFusion, EEGTokenProjector, MDTFCAEEncoder, VirtualFMRIBranch


def test_shapes() -> None:
    enc = MDTFCAEEncoder(in_channels=63, hidden_dim=64, d_eeg=128, grid_size=4)
    proj = EEGTokenProjector(d_eeg=128, brain_dim=256, num_tokens=4, mode="multiscale")
    virt = VirtualFMRIBranch(
        in_channels=64,
        volume_depth=8,
        num_clusters=16,
        brain_dim=256,
        prefer_official_decoder=False,
    )
    bit = BITCrossFusion(
        brain_dim=256,
        num_brain_tokens=16,
        num_query_tokens=32,
        num_blocks=1,
        num_heads=4,
        clip_dim=64,
        vgg_dim=32,
        inject_eeg_as="kv",
    )
    spec = torch.randn(2, 63, 33, 64)
    enc_out = enc(spec)
    virt_out = virt(enc_out["feat_map"])
    eeg_kv = proj(enc_out["z_eeg"], enc_out["eeg_tokens"])
    bit_out = bit(virt_out["brain_tokens"], eeg_kv)
    assert enc_out["z_eeg"].shape == (2, 128)
    assert virt_out["brain_tokens"].shape == (2, 16, 256)
    assert eeg_kv.shape[0] == 2 and eeg_kv.shape[-1] == 256
    assert bit_out["clip_tokens"].shape == (2, 32, 64)
    assert bit_out["kv_len"] == 16 + eeg_kv.shape[1]
    print("[OK] test_shapes")


if __name__ == "__main__":
    test_shapes()
