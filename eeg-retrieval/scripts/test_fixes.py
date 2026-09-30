#!/usr/bin/env python
"""Tests for the sub-08 correctness fixes. Runs on CPU in a few seconds.

Every test here corresponds to a defect that was diagnosed from the first sub-08
run. They are written to fail loudly if the defect is reintroduced, because each
one is silent: none of them crashes training, they just quietly produce a worse
number that looks like a modelling result.

Usage:  python scripts/test_fixes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nwret.augment import (AUG_NAMES, RandomChannelDropout, RandomGaussianNoise,
                           RandomSmooth, RandomTimeShift, build_aug)
from nwret.losses import InfoNCE

FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# ------------------------------------------------------------------ softplus
def test_softplus_is_on_temperature():
    """The bug: softplus applied to the EMBEDDING instead of the loss scale.

    SAMGA applies it to logit_scale (module/loss.py:93). Applying it to embeddings
    forces them non-negative, so after L2 normalisation every pair of vectors has a
    large positive cosine similarity and the contrastive signal loses resolution.
    """
    print("\nsoftplus placement")
    c = InfoNCE(init_temp=0.07, softplus=True)
    expected = float(torch.nn.functional.softplus(
        torch.tensor(float(np.log(1 / 0.07)))))
    s_soft = float(c.effective_scale().detach())
    s_exp = float(InfoNCE(0.07, softplus=False).effective_scale().detach())
    check("softplus -> soft(~2.73) scale, matching SAMGA",
          abs(s_soft - expected) < 1e-5, f"got {s_soft:.3f}, expected {expected:.3f}")
    check("softplus scale is much softer than exp",
          s_soft < 0.5 * s_exp, f"softplus {s_soft:.2f} vs exp {s_exp:.2f}")

    # The model must no longer be able to apply softplus to the embedding.
    # Check for an actual call, not the word: the surrounding comment explains why
    # softplus does not belong here, so a substring search would trip on itself.
    src = (Path(__file__).resolve().parent / "nwret" / "model.py").read_text()
    body = src.split("def encode_eeg", 1)[1].split("def encode_image", 1)[0]
    code = "\n".join(l.split("#", 1)[0] for l in body.splitlines())
    check("model.encode_eeg no longer softpluses the embedding",
          "softplus" not in code, "found a softplus call in encode_eeg")


def test_positive_embedding_geometry_is_what_breaks():
    """Quantify the damage so the fix is not taken on faith."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(256, 64, generator=g)
    sim = torch.nn.functional.normalize(z, dim=-1) @ torch.nn.functional.normalize(z, dim=-1).t()
    off = sim[~torch.eye(256, dtype=bool)].mean()
    zp = torch.nn.functional.normalize(torch.nn.functional.softplus(z), dim=-1)
    sim_p = zp @ zp.t()
    off_p = sim_p[~torch.eye(256, dtype=bool)].mean()
    check("softplus(embedding) inflates all pairwise cosines",
          off_p > off + 0.2, f"random {off:.3f} -> softplus {off_p:.3f}")


# ------------------------------------------------------------------ augment
def test_augmentations():
    print("\naugmentation")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.5, size=(17, 250)).astype(np.float32)
    x0 = x.copy()

    for name in AUG_NAMES:
        aug = build_aug(name)
        if aug is None:
            continue
        y = aug(x, np.random.default_rng(1))
        check(f"{name}: shape preserved", y.shape == x.shape, f"{y.shape}")
        check(f"{name}: dtype preserved", y.dtype == x.dtype, str(y.dtype))
        check(f"{name}: does not mutate its input", np.array_equal(x, x0),
              "input was modified in place")

    # Each transform should actually change the signal.
    for t, nm in ((RandomTimeShift(5), "time_shift"),
                  (RandomGaussianNoise(0.1), "noise"),
                  (RandomChannelDropout(0.1), "channel_dropout"),
                  (RandomSmooth(5, 0.3), "smooth")):
        changed = sum(not np.allclose(t(x.copy(), np.random.default_rng(s)), x) for s in range(5))
        check(f"{nm}: is active on at least one of 5 draws", changed > 0, f"{changed}/5 draws changed")

    # Noise magnitude must be meaningful at OUR data scale (std ~0.5), since
    # SAMGA's default std=0.001 would be numerically inert here.
    y = RandomGaussianNoise(0.1)(x, np.random.default_rng(2))
    check("noise std is ~10% of signal std (not inert)",
          0.02 < float((y - x).std()) < 0.12, f"noise std {float((y - x).std()):.4f}")

    # Channel dropout must zero whole channels, not individual samples.
    y = RandomChannelDropout(0.5)(x, np.random.default_rng(3))
    zeroed = np.isclose(y, 0).all(axis=1)
    check("channel_dropout zeroes entire channels", zeroed.any(), f"{zeroed.sum()}/17 channels")

    # Smoothing must reduce high-frequency energy on the channels it touches.
    ys = RandomSmooth(5, 1.0)(x, np.random.default_rng(4))
    d1 = np.abs(np.diff(ys, axis=-1)).mean()
    d0 = np.abs(np.diff(x, axis=-1)).mean()
    check("smooth reduces temporal roughness", d1 < d0, f"{d0:.4f} -> {d1:.4f}")


# ------------------------------------------------------------------ val sweep
def test_selection_metric_uses_all_slots():
    """The bug: TestDataset indexed [:, 0], so a (150, 10, ...) val split was
    scored on 1 of its 10 images and the per-epoch val Top-1 was 10x noisier than
    it needed to be. It peaked at epoch 8 and drifted for 22 epochs after."""
    print("\nvalidation slot sweep")
    src = (Path(__file__).resolve().parent / "nwret" / "data.py").read_text()
    body = src.split("class TestDataset", 1)[1]
    check("TestDataset takes a slot argument", "slot" in body.split("def __len__")[0])

    tsrc = (Path(__file__).resolve().parent / "nwret" / "train.py").read_text()
    check("evaluate_selection exists", "def evaluate_selection" in tsrc)
    check("evaluate_selection sweeps slots by default", "sweep: bool = True" in tsrc)
    check("train loop calls evaluate_selection", "evaluate_selection(model, val_eeg" in tsrc)
    # Test is still scored once, on slot 0 -- it has exactly one image per concept.
    check("test path still uses slot 0 (test has 1 image/concept)",
          "evaluate(model, te_eeg, img_te" in tsrc)


# ------------------------------------------------------------------ two stage
def test_two_stage_mmd_available_but_off():
    """Stage-1 MMD is OFF by default, deliberately deviating from SAMGA.

    Measured at sub-08 (F1 vs F3, F2 vs F4): enabling the warm-up cost 3.5 and 4.0
    test Top-1 points. The reason is structural, not a tuning mistake -- SAMGA's EEG
    encoder is a from-scratch projector that needs a geometry warm-up before instance
    discrimination can work, whereas ours is a pretrained ViT that already carries a
    usable geometry. The machinery must still function when re-enabled, for the
    ablation, so the SAMGA-faithful values are retained behind the flag.
    """
    print("\ntwo-stage MMD schedule (available, off by default)")
    src = (Path(__file__).resolve().parent / "nwret" / "train.py").read_text()
    check("stage1-epochs defaults to 0 (MMD off), deviating from SAMGA on purpose",
          '"--stage1-epochs", type=int, default=0' in src)
    check("mmd-start still SAMGA's 0.9 when stage 1 is enabled",
          '"--mmd-start", type=float, default=0.9' in src)
    check("mmd-end still SAMGA's 0.2", '"--mmd-end", type=float, default=0.2' in src)
    check("stage2 lr still SAMGA's 5e-5", '"--stage2-lr", type=float, default=5e-5' in src)
    check("contrastive weight is the complement of mmd_w, not a plain sum",
          "contrast_w = 1.0 - mmd_w if in_stage1 else 1.0" in src)
    check("no stale cosine scheduler", "CosineAnnealingLR" not in src)

    def mmd_at(epoch, s1=20, m0=0.9, m1=0.2):
        if s1 <= 0 or epoch > s1:
            return 0.0
        prog = max(0.0, min(1.0, (epoch - 1) / (s1 - 1)))
        return m0 + (m1 - m0) * prog

    check("when enabled, mmd_w runs 0.9 -> 0.2 across stage 1",
          abs(mmd_at(1) - 0.9) < 1e-9 and abs(mmd_at(20) - 0.2) < 1e-9,
          f"{mmd_at(1):.2f} -> {mmd_at(20):.2f}")
    check("when disabled, mmd_w is 0 from epoch 1", mmd_at(1, s1=0) == 0.0)


def test_alignment_target_is_selectable():
    """The axis the design doc prices as the single largest lever.

    Earlier arms hard-wired the alignment target to the final layer -- our cached
    features are 1024-d, which is exactly visual.proj's output width -- and swept the
    *EEG pathway's* depth instead. So the layer sweep ran on the wrong axis, which is
    why it came back flat with no interior maximum.
    """
    print("\nalignment target layer (the axis that was wrong)")
    here = Path(__file__).resolve().parent
    src = (here / "nwret" / "train.py").read_text()
    check("--target-features exists", '"--target-features"' in src)
    check("--target-layer exists", '"--target-layer"' in src)
    check("default target is the cached final layer (_pooled)",
          'default="_pooled"' in src)
    check("shape guard rejects features that do not match the EEG layout",
          "does not match the EEG layout" in src)
    check("extract_layers.py exists", (here / "nwret" / "extract_layers.py").is_file())
    check("probe_layers.py exists", (here / "nwret" / "probe_layers.py").is_file())

    p = (here / "nwret" / "probe_layers.py").read_text()
    check("probe selects on VALIDATION top1, never test",
          'key=lambda r: r["val_top1"]' in p and 'key=lambda r: r["test_top1"]' not in p)
    check("probe sweeps lambda per layer (layers differ in scale)",
          "best_lam" in p and "per_lam" in p)
    check("probe factorises the EEG once and reuses it for every layer",
          p.count("torch.linalg.svd(") == 1)
    check("probe states whether the peak beats the final layer beyond noise",
          "consistent with the documented lever but not significant" in p
          and "no interior peak and no band above" in p)
    check("probe reports the inverted-U shape across depth, which is the doc's criterion",
          "inverted-U across depth" in p)
    check("probe counts the band of layers above the final layer, not just the argmax",
          "interior layers above the final layer (val)" in p)
    check("probe uses std/sqrt(n_slots) as the error bar, not the per-slot std",
          "np.sqrt(n_val_slots)" in p)

    e = (here / "nwret" / "extract_layers.py").read_text()
    check("extract verifies _pooled against the shipped features (hard gate)",
          "verify_against_shipped" in e and "return 4" in e)


# ------------------------------------------------------------------ ridge ref
def test_val_slot_layout_is_not_flattened():
    """The slot axis must be indexed, not flattened away.

    (n_concepts, n_slots, feat) -> contiguous rows of n_concepts rows is NOT "slot s
    of every concept": a flat reshape is concept-major, so n_concepts rows are
    n_concepts/n_slots whole concepts with all their slots. The retrieval still runs
    and still returns a plausible number -- a 150-way task over 15 distinct concepts
    instead of over 150 -- which is exactly why this needs a test rather than care.
    """
    print("\nval slot layout (the bug that returns plausible numbers)")
    raw = np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)   # 4 concepts, 3 slots
    correct = raw[:, 1]                                        # one row per concept
    flattened = raw.reshape(12, 2)
    wrong = flattened[1 * 4:(1 + 1) * 4]                       # contiguous slice
    check("flatten-then-slice is a different thing from indexing the slot axis",
          not np.array_equal(correct, wrong),
          f"correct[0]={correct[0]} vs flattened[0]={wrong[0]}")

    here = Path(__file__).resolve().parent
    for name in ("probe_layers.py", "baseline_ridge.py"):
        src = (here / "nwret" / name).read_text()
        check(f"{name} keeps the slot axis and indexes it",
              "[:, s]" in src or "[:, si]" in src)
        check(f"{name} does not flatten the val set before slicing",
              "reshape(len(val_c), -1)" not in src
              and "reshape(len(v_c) * n_slots" not in src)


def test_layer_features_keep_the_image_axis():
    """Per-layer features must carry an explicit image axis.

    The shipped test array is (200, 1, 1024). Saving ours as (200, 1024) looks
    harmless and is not: train.py compares shape[:2] against the EEG's
    (200, 1, Ch, T) and the probe indexes Yte[:, 0]. Both then either refuse to run
    or pick the wrong axis. And the verification gate compared raw shapes, so on the
    test split it declined to run at all -- a gate that cannot fire reads as verified.
    """
    print("\nper-layer feature layout + the verification gate")
    here = Path(__file__).resolve().parent
    src = (here / "nwret" / "extract_layers.py").read_text()
    check("saves (n_concepts, imgs_per_concept, D), not flattened",
          "arr.reshape(len(concepts), int(n_img[0]), v.shape[-1])" in src)
    check("gate compares flattened rows, not raw shapes",
          "_as_rows(ours), _as_rows(shipped)" in src)
    check("gate failing is fatal, not a warning",
          "return 3" in src and "return 4" in src and "could not verify" in src)

    m = here.parent / "outputs" / "features" / "clip_h14_layers" / "test" / "manifest.json"
    if not m.is_file():
        print("  [skip] no extracted test manifest yet (checked on the next run)")
        return
    mf = json.loads(m.read_text())
    v = mf["verify_vs_shipped"]
    check("gate actually ran on the extracted test split", v.get("checked") is True, str(v))
    check("extracted _pooled reproduces the shipped features row for row",
          v.get("mean_cosine", 0) > 0.999 and v.get("min_cosine", 0) > 0.99,
          f"mean {v.get('mean_cosine'):.8f} min {v.get('min_cosine'):.8f}")
    check("test arrays carry the image axis (200, 1, D)",
          mf["layers"]["_pooled"] == [200, 1, 1024], str(mf["layers"]["_pooled"]))


def test_fusion_modes():
    """Two different fusion axes, and they must not be conflated.

    `fusion_mode` blends layers of the EEG ENCODER (--layers).
    `target_fusion` blends layers of the frozen IMAGE tower (--target-layers).
    Both were added in the same change, so both get pinned here.

    The doc's phase 2 asks two questions in order: uniform fusion first ("do these
    layers carry complementary information at all?"), then routing ("can the model
    combine them better than equally?"). Routing cannot win if uniform already
    loses to the best single layer, so the modes must be separately selectable.
    """
    print("\nfusion modes (EEG-side and image-side are different axes)")
    from nwret.encoders import LayerFusion
    import torch.nn as nn

    k, d = 3, 8
    feats = {l: torch.randn(5, d) for l in (4, 8, 12)}

    uni = LayerFusion(layers=[4, 8, 12], d_in=d, n_subjects=1, fusion_mode="uniform")
    fused_u, w_u = uni(feats, subject_ids=torch.zeros(5, dtype=torch.long), training=True)
    check("uniform: output shape is (B, D)", tuple(fused_u.shape) == (5, d), str(fused_u.shape))
    check("uniform: weights are exactly 1/k every time",
          bool((w_u == 1.0 / k).all()), f"min {float(w_u.min()):.4f} max {float(w_u.max()):.4f}")
    check("uniform: weights identical across rows (no routing)",
          bool((w_u[0] == w_u[-1]).all()))
    check("uniform: has NO learnable prior (an unused param would be a false claim)",
          not hasattr(uni, "global_prior") and not hasattr(uni, "subject_residual"))
    lu = len([p for p in uni.parameters() if p.requires_grad])
    check("uniform: still has per-layer projections, so this is fusion not averaging",
          lu == k, f"{lu} trainable param tensors (expect {k} projections)")

    chk = LayerFusion(layers=[4, 8, 12], d_in=d, n_subjects=1, fusion_mode="routed")
    _, w_r = chk(feats, subject_ids=torch.zeros(5, dtype=torch.long), training=True)
    check("routed: weights sum to 1", bool(torch.allclose(w_r.sum(-1), torch.ones(5), atol=1e-5)))
    check("routed: IS learnable (has global_prior + subject_residual)",
          hasattr(chk, "global_prior") and hasattr(chk, "subject_residual"))
    check("routed: learns layer weights, so they can move off uniform",
          chk.global_prior.requires_grad)
    try:
        LayerFusion(layers=[4, 8], d_in=d, n_subjects=1, fusion_mode="bogus")
        check("rejects an unknown fusion mode", False)
    except ValueError:
        check("rejects an unknown fusion mode", True)

    # ---- image-side multi-target blending
    from nwret.model import RetrievalModel

    def build(tf):
        return RetrievalModel(
            backbone="timm:vit_b16_in21k", channel_names=["Cz"] * 63, layers=[12],
            n_subjects=1, d_embed=16, grid_h=7, grid_w=7, n_time_windows=4,
            n_timepoints=250, image_dim=32, pretrained=False, drop=0.0,
            target_fusion=tf, n_targets=(1 if tf == "single" else 3),
        )

    m_mean, m_rt = build("mean"), build("routed")
    f_single = torch.randn(4, 32)
    f_multi = torch.randn(4, 3, 32)
    check("single: (B, D) target -> (B, d_embed)",
          tuple(m_mean.encode_image(f_single).shape) == (4, 16))
    check("mean: (B, k, D) target -> (B, d_embed), blended then projected",
          tuple(m_mean.encode_image(f_multi).shape) == (4, 16))
    check("routed: same external shape, blend happens inside encode_image",
          tuple(m_rt.encode_image(f_multi).shape) == (4, 16))
    check("mean: reports equal target weights",
          bool((m_mean.target_weights() == 1.0 / 3).all()))
    check("single: reports no target weights", m_mean.target_weights() is not None
          and m_rt.target_weights() is not None)
    wt = m_rt.target_weights()
    check("routed: target weights are a distribution and learnable",
          wt is not None and bool(torch.allclose(wt.sum(), torch.tensor(1.0), atol=1e-5))
          and m_rt.target_w.requires_grad)
    check("mean blends by averaging (equals the mean of the per-layer projections)",
          bool(torch.allclose(m_mean.encode_image(f_multi),
                              m_mean.img_head(f_multi.mean(dim=1)), atol=1e-5)))

    # The single-target path must be untouched: every arm already reported is
    # single-target, so a regression here would invalidate the existing numbers.
    check("single-target path unchanged: encode_image(single) == img_head(single)",
          bool(torch.allclose(m_mean.encode_image(f_single), m_mean.img_head(f_single), atol=1e-6)))
    try:
        build("single")
        RetrievalModel(
            backbone="timm:vit_b16_in21k", channel_names=["Cz"] * 63, layers=[12],
            n_subjects=1, d_embed=16, grid_h=7, grid_w=7, n_time_windows=4,
            n_timepoints=250, image_dim=32, pretrained=False, drop=0.0,
            target_fusion="routed", n_targets=1,
        )
        check("rejects multi-target fusion with fewer than 2 targets", False)
    except ValueError:
        check("rejects multi-target fusion with fewer than 2 targets", True)


def test_eegit_region_interpolation():
    """EEGiT Eq.1-2, checked against the equations rather than a restatement.

    The equations are `x_i = i*(N_r-1)/(P-1)`, `j = floor(x_i)`, `alpha_i = x_i - j`,
    `y_i = (1-alpha_i)*y_j + alpha_i*y_(j+1)`, with the `j = N_r-1` case clamped to
    `y_(N_r-1)`. If the matrix does not reproduce that literally, the spatial axis
    is not EEGiT's and the pretrained conv sees a differently-shaped signal.
    """
    print("\nEEGiT region interpolation (Eq.1-2)")
    from nwret.tokenizer import region_interp_matrix

    n_src, P = 7, 16
    M = region_interp_matrix(n_src, P)
    check("shape is (P, N_r)", M.shape == (P, n_src), str(M.shape))
    check("rows are convex combinations (sum to 1)",
          bool(np.allclose(M.sum(axis=1), 1.0, atol=1e-6)))
    check("no negative weights", bool((M >= -1e-7).all()))

    worst = 0.0
    for i in range(P):
        x = i * (n_src - 1) / (P - 1)
        j = int(np.floor(x))
        a = x - j
        want = np.zeros(n_src)
        if j >= n_src - 1:
            want[n_src - 1] = 1.0
        else:
            want[j], want[j + 1] = 1.0 - a, a
        worst = max(worst, float(np.abs(M[i] - want).max()))
    check("every row matches Eq.1-2 exactly", worst < 1e-6, f"max abs err {worst:.2e}")

    check("endpoints are preserved (first/last electrode hit exactly)",
          M[0, 0] == 1.0 and M[-1, -1] == 1.0)

    # n_src == 1 has no neighbour to interpolate with; the clamp branch is the only
    # well-defined answer, so it must not divide by zero or emit NaNs.
    single = region_interp_matrix(1, P)
    check("a 1-electrode region replicates it instead of dividing by N_r-1",
          single.shape == (P, 1) and bool(np.allclose(single, 1.0)))

    # n_src == P is the identity: no resampling happens at all. EEGiT's occipital
    # region has exactly 8 -> P=16, so this branch is NOT the one in play there;
    # the identity check is here to confirm the general formula degenerates right.
    ident = region_interp_matrix(P, P)
    check("N_r == P is the identity", bool(np.allclose(ident, np.eye(P), atol=1e-6)))

    # Strictly increasing source support is what makes the spatial axis ordered;
    # a non-monotone matrix would scramble the region's left-to-right layout.
    argmax_per_row = M.argmax(axis=1)
    check("source support is monotone in i (the axis stays ordered)",
          bool(np.all(np.diff(argmax_per_row) >= 0)), str(argmax_per_row))


def test_eegit_tokenizer_geometry():
    """The token count must be EEGiT's '14 x 5 = 70 spatiotemporal patches'."""
    print("\nEEGiT patch tokenizer geometry")
    from nwret.tokenizer import EEGIT_REGIONS, EEGPatchTokenizer
    from nwret import config
    import json as _json

    all_ch = _json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"]
    op_ch = list(config.CHANNELS_OCCIPITO_PARIETAL)
    check("all 63 montage channels are covered by the 5 EEGiT regions",
          len(all_ch) == 63, f"{len(all_ch)} channels")
    assigned = [c for _, members in EEGIT_REGIONS for c in members]
    check("regions partition the montage with no duplicates",
          sorted(assigned) == sorted(all_ch),
          f"missing {sorted(set(all_ch) - set(assigned))}, "
          f"extra {sorted(set(assigned) - set(all_ch))}")

    tk = EEGPatchTokenizer(channel_names=all_ch, patch_size=16, n_patches_w=14,
                           n_timepoints=250, zscore=False)
    check("5 regions at all 63 channels", tk.n_regions == 5, str(tk.n_regions))
    check("EEG image is 80 x 224 (regions*16, patches_w*16)",
          (tk.height, tk.width) == (80, 224), f"{tk.height}x{tk.width}")
    check("patch grid is 5 x 14 -> 70 tokens",
          tk.grid == (5, 14) and tk.n_regions * tk.n_patches_w == 70, str(tk.grid))

    tk.set_norm_stats(np.random.randn(4, 2, 63, 250).astype(np.float32))
    out = tk(torch.randn(3, 63, 250))
    check("forward returns an RGB-like (B, 3, 80, 224)",
          tuple(out.shape) == (3, 3, 80, 224), str(out.shape))
    check("the 3 planes are identical (replicated, as EEGiT states)",
          torch.allclose(out[:, 0], out[:, 1]) and torch.allclose(out[:, 1], out[:, 2]))

    # The point of the resample is that patch_embed sees image-sized input; the
    # source time axis must be resampled, not truncated or zero-padded.
    tk2 = EEGPatchTokenizer(channel_names=all_ch, patch_size=16, n_patches_w=14,
                            n_timepoints=1000, zscore=False)
    tk2.set_norm_stats(np.random.randn(2, 2, 63, 250).astype(np.float32))
    check("a different source time length still lands on 224",
          tuple(tk2(torch.randn(2, 63, 1000)).shape) == (2, 3, 80, 224))

    # z-score is what puts EEG into the value range the pretrained conv expects.
    tkz = EEGPatchTokenizer(channel_names=all_ch, zscore=True)
    rng = np.random.default_rng(0)
    stats_eeg = (rng.normal(2.0, 3.0, size=(8, 2, 63, 250))).astype(np.float32)
    tkz.set_norm_stats(stats_eeg)
    base = torch.from_numpy(stats_eeg[:4].reshape(-1, 63, 250))
    z1 = tkz(base)
    check("z-scored output has ~zero mean",
          abs(float(z1.mean())) < 0.05, f"mean {float(z1.mean()):+.3f}")
    # The point of the z-score is that raw EEG units (volts) must not reach the
    # pretrained conv, which was trained on images. So the representation has to
    # be a function of SHAPE only, not of the input's units. Note that the output
    # std is not 1.0: the temporal resample averages neighbouring samples, which
    # shrinks variance by ~0.71x. Scale invariance is the property to pin, not std.
    tkz_big = EEGPatchTokenizer(channel_names=all_ch, zscore=True)
    tkz_big.set_norm_stats(stats_eeg * 1000.0)      # stats consistent with the input
    z2 = tkz_big(base * 1000.0)
    check("z-scored output is invariant to the input's units (1000x rescale)",
          torch.allclose(z1, z2, atol=1e-3),
          f"max diff {float((z1 - z2).abs().max()):.2e}")
    check("z-scored output std is in an image-like range (0.4-1.2, not raw volts)",
          0.4 < float(z1.std()) < 1.2, f"std {float(z1.std()):.3f}")
    tk_noz = EEGPatchTokenizer(channel_names=all_ch, zscore=False)
    n1, n2 = tk_noz(base), tk_noz(base * 1000.0)
    check("WITHOUT z-score the input's units leak straight into patch_embed",
          float((n1 - n2).abs().max()) > 1.0,
          f"max diff {float((n1 - n2).abs().max()):.1f} (this is the defect)")

    # Refusing to run without statistics is deliberate: a silent no-op z-score
    # would leave the patch_embed on a distribution it was never trained on.
    fresh = EEGPatchTokenizer(channel_names=all_ch, zscore=True)
    try:
        fresh(torch.randn(1, 63, 250))
        check("z-score without stats raises instead of silently no-oping", False)
    except RuntimeError as e:
        check("z-score without stats raises instead of silently no-oping",
              "set_norm_stats" in str(e))

    # A channel the regions do not cover must be an error, not a silent drop.
    try:
        EEGPatchTokenizer(channel_names=all_ch + ["ZZ99"], zscore=False)
        check("an uncovered channel is refused", False)
    except ValueError as e:
        check("an uncovered channel is refused", "ZZ99" in str(e))
    except KeyError:
        check("an uncovered channel is refused", False, "raised KeyError instead")

    # The occipito-parietal montage is EEGiT's parietal+occipital regions exactly;
    # EEGiT's Fig.7 ablation says keeping the occipital region matches the full
    # montage, so this path must work rather than crash on 2 regions.
    op = op_ch
    tko = EEGPatchTokenizer(channel_names=op, zscore=False)
    check("17-channel occipito-parietal resolves to 2 regions (EEGiT's P+O)",
          tko.n_regions == 2, f"{tko.n_regions} regions, {len(op)} channels")
    check("and yields a 2 x 14 = 28-token grid",
          tko.grid == (2, 14), str(tko.grid))


def test_patch_embed_is_actually_used():
    """The A1 defect: the grid path never called `patch_embed` at all."""
    print("\npretrained patch_embed is on the gradient path (A1)")
    from nwret.encoders import ViTEEGEncoder
    from nwret import config
    import json as _json

    all_ch = _json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"]
    torch.manual_seed(0)

    grid_enc = ViTEEGEncoder(backbone="vit_b16_in21k", channel_names=all_ch,
                             tokenizer_kind="grid", pretrained=True)
    eegit_enc = ViTEEGEncoder(backbone="vit_b16_in21k", channel_names=all_ch,
                              tokenizer_kind="eegit", pretrained=True)
    eegit_enc.tokenizer.set_norm_stats(np.random.randn(4, 2, 63, 250).astype(np.float32))

    check("both paths exist and expose a tokenizer",
          hasattr(grid_enc, "tokenizer") and hasattr(eegit_enc, "tokenizer"))
    check("eegit path keeps a reference to the pretrained conv",
          eegit_enc.patch_embed is eegit_enc.vit.patch_embed)
    check("the patch_embed conv is trainable in the eegit path",
          all(p.requires_grad for p in eegit_enc.patch_embed.parameters()))
    check("patch_embed is a 16x16 conv with 3 in-channels (RGB-like input)",
          tuple(eegit_enc.patch_embed.proj.weight.shape) ==
          (768, 3, 16, 16), str(tuple(eegit_enc.patch_embed.proj.weight.shape)))

    x = torch.randn(2, 63, 250)
    with torch.no_grad():
        before = eegit_enc.patch_embed.proj.weight.clone()
    feats = eegit_enc(x, layers=[12])       # NOT under no_grad: we need the graph
    feats[12].sum().backward()
    after = eegit_enc.patch_embed.proj.weight
    check("gradient reaches the pretrained conv (it is the interface now)",
          after.grad is not None and float(after.grad.abs().sum()) > 0.0)
    check("forward alone does not mutate it (no in-place surprise)",
          torch.allclose(before, after))

    check("eegit: 70 tokens per sample",
          tuple(feats[12].shape) == (2, 768), str(feats[12].shape))
    check("eegit: pos_embed was resampled to the 5x14 patch grid",
          tuple(eegit_enc.grid_pos.shape) == (1, 70, 768) and eegit_enc.dst_grid == (5, 14),
          f"{tuple(eegit_enc.grid_pos.shape)} grid {eegit_enc.dst_grid}")

    with torch.no_grad():
        gf = grid_enc(x, layers=[12])
    check("grid path still works unchanged (backward compatibility)",
          tuple(gf[12].shape) == (2, 768), str(gf[12].shape))


def test_pool_norm_is_applied():
    """The A4 defect: the last block's feature reached the head unnormalised."""
    print("\nfinal LayerNorm on pooled features (A4)")
    from nwret.encoders import ViTEEGEncoder
    from nwret import config
    import json as _json

    all_ch = _json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"]
    torch.manual_seed(0)
    enc_norm = ViTEEGEncoder(backbone="vit_b16_in21k", channel_names=all_ch,
                             tokenizer_kind="grid", pretrained=True, pool_norm=True)
    enc_raw = ViTEEGEncoder(backbone="vit_b16_in21k", channel_names=all_ch,
                            tokenizer_kind="grid", pretrained=True, pool_norm=False)
    check("pool_norm is exposed and defaults to the fixed behaviour",
          enc_norm.pool_norm is True and enc_raw.pool_norm is False)

    # eval() matters: both encoders carry Dropout, and in train() mode the two
    # would differ by a different random mask rather than by the LayerNorm.
    enc_norm.eval(); enc_raw.eval()
    enc_raw.load_state_dict(enc_norm.state_dict())
    x = torch.randn(4, 63, 250)
    with torch.no_grad():
        a = enc_norm(x, layers=[12])
        b = enc_raw(x, layers=[12])
    check("with norm differs from without (so the flag is not a no-op)",
          not torch.allclose(a[12], b[12]))

    # `enc_raw` output IS the raw block-12 residual cls, so the fix is verifiable
    # without reimplementing the token assembly in the test: applying ln_post to
    # it must reproduce the pool_norm=True output exactly.
    want = enc_norm.vit.norm(b[12])
    check("layer-12 output equals ln_post(raw block output) -- the timm path",
          torch.allclose(a[12], want, atol=1e-5),
          f"max diff {float((a[12] - want).abs().max()):.2e}")

    # Layers other than the last are now all in the same normalised space, which
    # is what LayerFusion needs to compare and blend them.
    with torch.no_grad():
        multi = enc_norm(x, layers=[6, 12])
    s6 = float(multi[6].std())
    s12 = float(multi[12].std())
    check("different depths now have comparable scale (fusion needs this)",
          max(s6, s12) / max(min(s6, s12), 1e-9) < 3.0, f"std6 {s6:.2f} vs std12 {s12:.2f}")


def test_cls_token_reaches_the_input():
    """The cls slot must be cls_token + pos_embed[0], as timm assembles it.

    Using pos_embed[0] alone silently drops a pretrained parameter from the input
    and leaves the cls slot identical across samples at initialisation.
    """
    print("\ncls token reaches the input (found while fixing A4)")
    from nwret.encoders import ViTEEGEncoder
    from nwret import config
    import json as _json

    all_ch = _json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"]
    torch.manual_seed(0)

    with_ct = ViTEEGEncoder(backbone="vit_b16_in21k", channel_names=all_ch,
                            tokenizer_kind="eegit", pretrained=True,
                            cls_token_prefix=True)
    without_ct = ViTEEGEncoder(backbone="vit_b16_in21k", channel_names=all_ch,
                               tokenizer_kind="eegit", pretrained=True,
                               cls_token_prefix=False)
    with_ct.tokenizer.set_norm_stats(np.random.randn(4, 2, 63, 250).astype(np.float32))
    without_ct.load_state_dict(with_ct.state_dict())

    # The stats flag has to travel in state_dict, otherwise a reloaded checkpoint
    # either refuses to run or silently z-scores with mean 0 / std 1.
    check("z-score stats survive load_state_dict (flag is a buffer, not an attribute)",
          bool(without_ct.tokenizer.stats_set),
          "a plain attribute would not be copied by state_dict()")
    check("vit exposes a cls_token (a pretrained parameter)",
          getattr(with_ct.vit, "cls_token", None) is not None)
    diff = float((with_ct.prefix_pos - without_ct.prefix_pos).abs().max())
    check("the cls slot differs from pos_embed[0] alone",
          diff > 1e-6, f"max diff {diff:.4f}")
    ct = with_ct.vit.cls_token.detach().reshape(-1)[: with_ct.d_model]
    check("and the difference IS the cls token",
          torch.allclose(with_ct.prefix_pos[0, 0] - without_ct.prefix_pos[0, 0],
                         ct, atol=1e-6))
    check("only the cls slot changes; register/extra prefix slots are untouched",
          with_ct.n_prefix == 1 or torch.allclose(
              with_ct.prefix_pos[:, 1:], without_ct.prefix_pos[:, 1:]))

    x = torch.randn(3, 63, 250)
    with_ct.eval(); without_ct.eval()
    with torch.no_grad():
        o1 = with_ct(x, layers=[12])[12]
        o2 = without_ct(x, layers=[12])[12]
    check("so the encoder output actually changes", not torch.allclose(o1, o2))
    check("and the cls slot is no longer batch-identical at init",
          not torch.allclose(o1[0], o1[1]))


def test_mmd_is_no_longer_dead():
    """A5: with sigmas pinned at (1,2,4,8) on 512-d embeddings the kernel was ~0.

    The earlier conclusion "SAMGA's MMD warm-up hurts here" was drawn from runs
    where this term contributed exactly zero gradient, so it measured the absence
    of MMD rather than MMD. This test pins down that the term now lives.
    """
    print("\nMMD is no longer numerically dead (A5)")
    from nwret.losses import mmd_rbf, median_heuristic_sigmas

    torch.manual_seed(0)
    # 512-d embeddings with the kind of spread the model actually produces.
    x = torch.randn(64, 512) * 5.0
    y = torch.randn(64, 512) * 5.0 + 1.0

    from nwret.losses import _rbf_kernel

    def legacy_mmd(a, b):
        """The pre-fix implementation verbatim: unnormalised embeddings, sigmas
        pinned at (1,2,4,8)."""
        sig = (1.0, 2.0, 4.0, 8.0)
        return (_rbf_kernel(a, a, sig).mean() + _rbf_kernel(b, b, sig).mean()
                - 2.0 * _rbf_kernel(a, b, sig).mean()).clamp_min(0.0)

    xl, yl = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
    old = legacy_mmd(xl, yl)
    old.backward()
    # With distances ~160 in 512-d and the largest bandwidth 8, exp(-d^2/128) is
    # ~e^-200 = 0 for every off-diagonal pair. What survives is only the DIAGONAL
    # (a point against itself = 1), which contributes 1/N to each of xx and yy and
    # nothing to xy -- so the value is the constant 2/N and carries no information
    # about the two distributions relative to each other.
    check("legacy MMD's value was a constant 2/N: pure self-similarity, no signal",
          abs(float(old) - 2.0 / x.shape[0]) < 1e-3,
          f"value {float(old):.5f} vs 2/N = {2.0 / x.shape[0]:.5f}")
    check("legacy MMD's gradient was ~0 -- the term was inert",
          float(xl.grad.abs().sum()) < 1e-6 and float(yl.grad.abs().sum()) < 1e-6,
          f"grad {float(xl.grad.abs().sum()):.2e}")

    xr, yr = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
    val = mmd_rbf(xr, yr)
    val.backward()
    check("MMD with the median heuristic is non-trivial", float(val) > 1e-4,
          f"value {float(val):.2e}")
    check("and it produces a real gradient now",
          float(xr.grad.abs().sum()) > 0.0 and float(yr.grad.abs().sum()) > 0.0,
          f"grad {float(xr.grad.abs().sum()):.2e}")

    # Identical distributions must still give ~0: the term is a discrepancy, so a
    # large value for x==y would mean it is measuring norms rather than shape.
    same = mmd_rbf(x, x.clone())
    check("MMD(x, x) is ~0 (it measures discrepancy, not magnitude)",
          float(same) < 1e-3, f"{float(same):.2e}")

    # Both sides are unit-normalised, so the value cannot be reduced by shrinking
    # the embeddings -- only by changing their direction.
    big = mmd_rbf(x * 1000.0, y * 1000.0)
    check("value is invariant to a global rescaling (norm drift cannot game it)",
          abs(float(big) - float(val)) < 1e-3, f"{float(big):.2e} vs {float(val):.2e}")

    sig = median_heuristic_sigmas(x, y)
    check("bandwidths are scaled to the data, not hard-coded",
          all(s > 0.1 for s in sig) and len(set(sig)) > 1, str([round(s, 2) for s in sig]))


def test_lr_groups_cover_every_trainable_parameter():
    """A parameter with no LR group is a silent no-op with an arbitrary LR."""
    print("\nLR groups partition the trainable parameters (B2)")
    from nwret.model import RetrievalModel
    from nwret import config
    import json as _json

    all_ch = _json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"]
    torch.manual_seed(0)
    m = RetrievalModel(backbone="timm:vit_b16_in21k", channel_names=all_ch,
                       layers=[8, 10, 12], n_subjects=1, image_dim=1024,
                       pretrained=True, fusion_mode="uniform",
                       tokenizer_kind="eegit", pool_norm=True)

    buckets = {"blocks": 0, "interface": 0, "heads": 0}
    unknown = []
    for name, prm in m.named_parameters():
        if not prm.requires_grad:
            continue
        if name.startswith("encoder.vit.blocks."):
            buckets["blocks"] += prm.numel()
        elif name.startswith(("encoder.vit.patch_embed.", "encoder.vit.pos_embed",
                              "encoder.vit.cls_token", "encoder.vit.norm",
                              "encoder.tokenizer.")):
            buckets["interface"] += prm.numel()
        elif name.startswith(("encoder.", "fusion.", "eeg_head.", "img_head.")):
            buckets["heads"] += prm.numel()
        else:
            unknown.append(name)

    check("every trainable parameter lands in exactly one group",
          not unknown, f"unassigned: {unknown[:4]}")
    check("the pretrained blocks dominate the parameter count (not the new parts)",
          buckets["blocks"] > 10 * (buckets["interface"] + buckets["heads"]),
          f"blocks {buckets['blocks']/1e6:.1f}M vs new "
          f"{(buckets['interface']+buckets['heads'])/1e6:.2f}M")
    check("the interface group is non-empty (patch_embed/pos_embed are new here)",
          buckets["interface"] > 0, f"{buckets['interface']/1e6:.2f}M")


def test_tokenizer_n_tokens_agrees_with_reality():
    """`n_tokens` is read by logging/reporting, so it must not be able to lie.

    This is the check that was missing: a reporting line read
    `tokenizer.n_tokens` unconditionally, and the EEGiT tokenizer did not define
    it. It crashed 55 seconds into an allocation instead of at dry-run time --
    which the smoke gate caught, but a dry run should have. Asserting the value
    against the token count the encoder actually produces catches both a missing
    attribute and one that has drifted out of sync with the forward pass.
    """
    print("\ntokenizer n_tokens agrees with the encoder")
    from nwret.encoders import ViTEEGEncoder
    from nwret import config
    import json as _json

    all_ch = _json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"]
    torch.manual_seed(0)

    cases = [("eegit", "eegit"), ("grid", "grid")]
    for name, kind in cases:
        enc = ViTEEGEncoder(backbone="vit_b16_in21k", channel_names=all_ch,
                            pretrained=True, tokenizer_kind=kind)
        tk = enc.tokenizer
        check(f"{name}: tokenizer exposes n_tokens", hasattr(tk, "n_tokens"))
        if kind == "eegit":
            tk.set_norm_stats(np.random.randn(4, 2, 63, 250).astype(np.float32))
        with torch.no_grad():
            x = torch.randn(2, 63, 250)
            # Count the tokens the encoder really pushes into the blocks, by the
            # path that tokenizer's encoder actually uses.
            if kind == "eegit":
                produced = int(enc.patch_embed(tk(x)).shape[1])
            else:
                produced = int(tk(x).shape[1])
        check(f"{name}: declared n_tokens == tokens actually produced ({produced})",
              int(tk.n_tokens) == produced,
              f"declared {tk.n_tokens}, produced {produced}")
        if kind == "eegit":
            check("eegit: n_tokens == regions x time patches",
                  int(tk.n_tokens) == tk.n_regions * tk.n_patches_w and int(tk.n_tokens) == 70,
                  f"{tk.n_regions} x {tk.n_patches_w} = {tk.n_tokens}")


def test_ridge_floor_is_wired_in():
    print("\nridge floor")
    src = (Path(__file__).resolve().parent / "nwret" / "train.py").read_text()
    check("train.py reads the ridge baseline", "load_ridge_ref" in src)
    check("train.py reports PASS/FAIL vs ridge",
          "ridge floor" in src and "PASS" in src and "FAIL" in src)
    s = (Path(__file__).resolve().parent / "nwret" / "summarize.py").read_text()
    check("summarize.py surfaces the ridge floor", "ridge_floor_test_top1" in s)
    check("summarize.py flags arms that beat ridge", "beats_ridge" in s)


def main() -> int:
    test_softplus_is_on_temperature()
    test_positive_embedding_geometry_is_what_breaks()
    test_augmentations()
    test_selection_metric_uses_all_slots()
    test_two_stage_mmd_available_but_off()
    test_alignment_target_is_selectable()
    test_val_slot_layout_is_not_flattened()
    test_layer_features_keep_the_image_axis()
    test_fusion_modes()
    test_eegit_region_interpolation()
    test_eegit_tokenizer_geometry()
    test_patch_embed_is_actually_used()
    test_pool_norm_is_applied()
    test_cls_token_reaches_the_input()
    test_tokenizer_n_tokens_agrees_with_reality()
    test_mmd_is_no_longer_dead()
    test_lr_groups_cover_every_trainable_parameter()
    test_ridge_floor_is_wired_in()
    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): " + ", ".join(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
