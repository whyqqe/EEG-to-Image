"""Structural checks for the dual-tower configuration.

These are the failures that do not raise. A model whose second tower is wired to
the wrong EEG geometry, or whose new parameters fall into no learning-rate group,
or whose structural head receives no gradient, trains happily and produces a
decreasing loss and a plausible-looking result file. Each check below exists
because it is cheap here and expensive after a GPU allocation.

Run:  python scripts/test_dual_tower.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nwret import config
from nwret.encoders import backbone_patch_size
from nwret.losses import grad_l1, latent_l1
from nwret.model import build_from_args
from nwret.tokenizer import EEGPatchTokenizer

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


CFG = dict(
    backbone="timm:vit_b16_in21k", layers=[8, 10, 12], d_embed=512,
    grid_h=7, grid_w=7, n_time_windows=4, n_timepoints=250,
    freeze_blocks=0, freeze_all=False, pool="cls", prior_center=None,
    prior_strength=1.0, layer_dropout=0.1, subject_dropout=0.3, drop=0.1,
    fusion_mode="uniform", target_fusion="single", target_layers=["_pooled"],
    tokenizer="eegit", pool_norm=True, cls_token_prefix=True,
    patch_size=16, n_patches_w=14, zscore=True, no_pretrained=True,
    struct_backbone="timm:dinov2_l_reg4", struct_layers=[22, 23, 24],
    struct_patch_size=14, struct_n_patches_w=16, struct_fusion_mode="uniform",
    struct_freeze_blocks=0, struct_drop=0.1, struct_base_ch=32, struct_field_ch=8,
)


def main() -> None:
    torch.manual_seed(0)
    info = __import__("json").loads((config.EEG_DIR / "info.json").read_text())
    chans = list(info["ch_names"])
    n_img = 2
    eeg = torch.randn(n_img * 3, len(chans), config.N_TIMEPOINTS)

    print("\n[1] geometry: each tower's EEG image must tile its own patch_embed conv")
    for kind, ps, npw in (("timm:vit_b16_in21k", 16, 14), ("timm:dinov2_l_reg4", 14, 16)):
        check(backbone_patch_size(kind) == ps, f"{kind} declares patch {ps}")
        tok = EEGPatchTokenizer(chans, patch_size=ps, n_patches_w=npw,
                                n_timepoints=250, zscore=True)
        tok.set_norm_stats(np.random.randn(8, len(chans), 250).astype(np.float64))
        img = tok(torch.randn(2, len(chans), 250))
        check(img.shape == (2, 3, 5 * ps, npw * ps),
              f"{kind} EEG image {tuple(img.shape)} == (2,3,{5 * ps},{npw * ps})")
        check(img.shape[2] % ps == 0 and img.shape[3] % ps == 0,
              f"{kind} image divides evenly by the conv kernel")
        check(tok.n_tokens == 5 * npw, f"{kind} emits {5 * npw} tokens")

    print("\n[2] model: shapes and the shared z-score interface")
    cfg = dict(CFG)
    model = build_from_args(cfg, chans, 1024)
    check(model.struct is not None, "structure tower was built")
    tok_s = model.struct.encoder.tokenizer
    tok_m = model.encoder.tokenizer
    check(tok_s is not tok_m, "the two towers hold separate tokenizer instances")
    tok_m.set_norm_stats(np.random.randn(8, len(chans), 250).astype(np.float64))
    tok_s.set_norm_stats(np.random.randn(8, len(chans), 250).astype(np.float64))

    out = model.forward_all(eeg, None, training=True)
    check(out["z"].shape == (n_img * 3, 512), f"semantic z {tuple(out['z'].shape)}")
    sd = out["struct"]
    check(sd["vae"].shape == (n_img * 3, 4, 64, 64), f"vae head {tuple(sd['vae'].shape)}")
    check(sd["depth"].shape == (n_img * 3, 1, 64, 64), f"depth head {tuple(sd['depth'].shape)}")
    check(bool(((sd["depth"] >= 0) & (sd["depth"] <= 1)).all()), "depth head is in [0,1]")
    check(sd["field"].shape == (n_img * 3, 8, 64, 64), f"shared field {tuple(sd['field'].shape)}")

    print("\n[3] the zero-initialised latent head starts at the conditional mean")
    check(float(sd["vae"].abs().max()) == 0.0,
          "vae head outputs exactly zero at step 0 (zero-init)")
    check(float(sd["depth"].std()) > 0.0, "depth head is NOT zero-init (would plateau)")

    print("\n[4] every trainable parameter gets a gradient and an LR group")
    from nwret.train import assign_param_groups
    groups = assign_param_groups(model)
    check(all(not v or True for v in groups.values()), "assign_param_groups raised nothing")
    n_grouped = sum(len(v) for v in groups.values())
    n_train = sum(1 for _ in model.named_parameters() if _[1].requires_grad)
    check(n_grouped == n_train, f"all {n_train} trainable tensors are grouped")
    check(len(groups["s_blocks"]) > 0, f"s_blocks populated ({len(groups['s_blocks'])})")
    check(len(groups["s_heads"]) > 0, f"s_heads populated ({len(groups['s_heads'])})")

    vae_t = torch.randn(n_img * 3, 4, 64, 64)
    dep_t = torch.rand(n_img * 3, 1, 64, 64)
    # A loss that exercises BOTH towers end to end. Using only the structural terms
    # would leave `img_head` legitimately gradient-free and the liveness check below
    # would then pass for the wrong reason.
    from nwret.losses import InfoNCE
    z_i = model.encode_image(torch.randn(n_img * 3, 1024))
    loss = (InfoNCE()(out["z"], z_i)
            + latent_l1(sd["vae"], vae_t)
            + 0.5 * grad_l1(sd["depth"], dep_t))
    loss.backward()
    dead = [(n, p.numel()) for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None]
    check(not dead, f"no trainable tensor is gradient-dead (found {len(dead)}: "
                    f"{[d[0] for d in dead[:6]]})")
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    check("encoder.vit.pos_embed" in frozen and "encoder.vit.cls_token" in frozen
          and "struct.encoder.vit.pos_embed" in frozen,
          "pos_embed / cls_token are marked frozen on both towers (they are detached "
          "into buffers at construction and never read again)")
    # The register variants put num_prefix_tokens-1 learned rows into the prefix.
    # Leaving them zero-filled is invisible in the shapes and shows up only as a
    # trainable parameter that never receives gradient, so it is checked directly.
    pp = model.struct.encoder.prefix_pos
    check(tuple(pp.shape[:2]) == (1, model.struct.encoder.n_prefix),
          f"prefix_pos carries exactly n_prefix rows ({tuple(pp.shape[:2])})")
    if model.struct.encoder.n_prefix > 1:
        filled = int((pp[0, 1:].abs().sum(-1) > 0).sum())
        check(filled == model.struct.encoder.n_prefix - 1,
              f"all {model.struct.encoder.n_prefix - 1} register rows are filled "
              f"from the checkpoint (not zero) -- {filled} filled")
        check("struct.encoder.vit.reg_token" in frozen,
              "reg_token is detached into prefix_pos and marked frozen")
    grads = [(n, float(p.grad.abs().sum())) for n, p in model.named_parameters()
             if p.requires_grad and n.startswith(("struct.vae_head", "struct.depth_head"))]
    check(all(g > 0 for _, g in grads),
          f"both structural heads receive non-zero gradient ({len(grads)} tensors)")
    # The projection feeding the field must be live on the FIRST step: with a
    # zero-initialised conv at the end, the gradient reaching it exists only
    # through that conv's weights, and a mistake there would look like a model
    # that simply never learns.
    proj_g = model.struct.proj.weight.grad
    check(proj_g is not None and float(proj_g.abs().sum()) > 0,
          "struct.proj receives gradient through the zero-init head")

    print("\n[5] the structure tower is optional and costs nothing when off")
    m2 = build_from_args({**CFG, "struct_backbone": ""}, chans, 1024)
    check(m2.struct is None, "no structure tower when --struct-backbone is empty")
    check(not any(k.startswith("struct.") for k in m2.state_dict()),
          "no struct.* keys in state_dict")
    g2 = assign_param_groups(m2)
    check(sum(len(v) for v in g2.values())
          == sum(1 for _ in m2.named_parameters() if _[1].requires_grad),
          "retrieval-only config still groups everything")

    print("\n[5b] a shallower --struct-layers must freeze, not fake-train, the rest")
    # The trap this guards: the fusion only reads the requested depths, so blocks
    # past the deepest one have no path to the loss. Left trainable they inflate the
    # reported parameter count by ~250M and sit in the optimizer as dead weight.
    m3 = build_from_args({**CFG, "struct_layers": [2, 3, 4]}, chans, 1024)
    frozen_deep = [n for n, p in m3.named_parameters()
                   if n.startswith("struct.encoder.vit.blocks.")
                   and not p.requires_grad]
    check(len(frozen_deep) > 0,
          f"blocks past the deepest requested layer are frozen ({len(frozen_deep)} tensors)")
    idx = {int(n.split(".blocks.", 1)[1].split(".", 1)[0]) for n in frozen_deep}
    # Block indices in the module names are 0-based, so a deepest layer of 4
    # (1-based) leaves blocks 4..23 frozen.
    check(idx == set(range(4, 24)),
          f"the frozen set spans exactly blocks 5..24 (got {min(idx)+1}..{max(idx)+1})")
    check(m3.struct.encoder.deepest == 4, "deepest requested layer recorded on the encoder")
    tr3 = sum(p.numel() for p in m3.parameters() if p.requires_grad)
    tr_full = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check(tr3 < tr_full - 100e6,
          f"shallow config is materially smaller: {tr3/1e6:.1f}M vs {tr_full/1e6:.1f}M")

    print("\n[6] the validation path: every key selection reads must exist")
    # This check exists because of a failure it would have caught. `retrieval_report`
    # returns only top1/top5/n; `mean_rank` is a separate function the single-tower
    # `evaluate` adds by hand. `evaluate_selection_dual` called `retrieval_report`
    # directly and then read `p["mean_rank"]`, so it raised KeyError the first time
    # it ran -- which was inside the smoke gate on a GPU allocation, three stages
    # into the pipeline, rather than here.
    from nwret.train import evaluate_selection, evaluate_selection_dual
    n_conc = 6
    # (n_concepts, n_img, C, T) and (n_concepts, n_img, D), the layout the real
    # validation split has. NOTE that the *training* targets are flat
    # (n_concepts*n_img, ...) because AuxTargetDataset indexes concept*n_img+slot --
    # the validation layout is the reshaped one, and passing the flat array here is
    # the mistake this section's shape guard exists to catch.
    val_eeg = torch.randn(n_conc, n_img, len(chans), config.N_TIMEPOINTS).numpy()
    val_feat = torch.randn(n_conc, n_img, 1024).numpy()
    vae_rows = np.random.randn(n_conc, n_img, 4, 64, 64).astype(np.float32)
    dep_rows = np.random.rand(n_conc, n_img, 64, 64).astype(np.float32)
    dev = torch.device("cpu")
    dual = evaluate_selection_dual(
        model, val_eeg, val_feat, dev,
        vae_rows=vae_rows, depth_rows=dep_rows,
        vae_mean=np.zeros(4, np.float32), vae_std=np.ones(4, np.float32),
        sweep=True, l2norm=True,
    )
    want = ("top1", "top5", "mean_rank", "n", "n_slots", "top1_std",
            "vae_top1", "vae_cos", "depth_pearson")
    missing = [k for k in want if k not in dual]
    check(not missing, f"evaluate_selection_dual returns every key selection reads "
                       f"(missing {missing})")
    check(dual["n_slots"] == n_img, f"the dual evaluator sweeps all {n_img} slots")
    check(dual["vae_top1"] is not None and dual["depth_pearson"] is not None,
          "both structural readouts are produced from the same pass")
    check(0.0 <= dual["top1"] <= 100.0 and 0.0 <= dual["vae_top1"] <= 100.0,
          "the two Top-1 readouts are on the same 0-100 scale")
    # The single-tower path must keep working unchanged: it is what every earlier
    # result was selected with, so a regression here invalidates the comparison to
    # them rather than failing loudly.
    single = evaluate_selection(model, val_eeg, val_feat, dev, sweep=True)
    missing1 = [k for k in ("top1", "top5", "mean_rank", "n", "n_slots", "top1_std")
                if k not in single]
    check(not missing1, f"evaluate_selection still returns its key set "
                        f"(missing {missing1})")

    # The shape guard: the flat training layout must raise, not compute. A silent
    # pass here would produce a structural readout built from an array indexed on
    # the wrong axis -- a number that looks like a number.
    raised = ""
    try:
        evaluate_selection_dual(
            model, val_eeg, val_feat, dev,
            vae_rows=vae_rows.reshape(n_conc * n_img, 4, 64, 64),
            depth_rows=dep_rows, vae_mean=np.zeros(4, np.float32),
            vae_std=np.ones(4, np.float32), sweep=True, l2norm=True)
    except ValueError as e:
        raised = str(e)
    check("n_concepts" in raised,
          "the flat training layout raises a shape error naming n_concepts rather "
          f"than a matmul failure (got: {raised[:60]!r})")

    print("\n[7] parameter counts")
    for nm, m in (("dual", model), ("retrieval-only", m2)):
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {nm:15s} trainable {tr/1e6:8.2f}M")

    print("\n[8] --train-slots: narrowing the training set must not move the rows")
    # The structural target caches are addressed as `concept * n_slots_total + slot`.
    # If `--train-slots` renumbered rows by the *narrowed* width instead, the
    # structure tower would be trained on one image's EEG against another image's
    # latent: the loss would still fall, and the decoded pictures would still look
    # plausible, so nothing downstream would notice. This is the check for that.
    from nwret.data import AuxTargetDataset
    n_conc_s, n_slot_s, ch_s, t_s = 4, 10, 3, 8
    rng_s = np.random.default_rng(0)
    eeg_s = rng_s.standard_normal((n_conc_s, n_slot_s, ch_s, t_s)).astype(np.float32)
    feat_s = rng_s.standard_normal((n_conc_s, n_slot_s, 5)).astype(np.float32)
    # Encode the identity of each (concept, slot) into its target: row r -> value r.
    vae_s = np.broadcast_to(
        np.arange(n_conc_s * n_slot_s, dtype=np.float32).reshape(-1, 1, 1, 1),
        (n_conc_s * n_slot_s, 1, 4, 4)).copy()
    conc_s = np.arange(n_conc_s)

    def rows_for(ds):
        """(concept, the value the target says it is) for every item."""
        return [(int(ds[i][2]), int(np.asarray(ds[i][3]).flatten()[0])) for i in range(len(ds))]

    full = AuxTargetDataset(eeg_s, feat_s, conc_s, aux_vae=vae_s,
                            vae_mean=np.zeros(4, np.float32),
                            vae_std=np.ones(4, np.float32))
    one = AuxTargetDataset(eeg_s, feat_s, conc_s, aux_vae=vae_s,
                           vae_mean=np.zeros(4, np.float32),
                           vae_std=np.ones(4, np.float32), slots=[0])
    two = AuxTargetDataset(eeg_s, feat_s, conc_s, aux_vae=vae_s,
                           vae_mean=np.zeros(4, np.float32),
                           vae_std=np.ones(4, np.float32), slots=[3, 7])

    check(len(full) == n_conc_s * n_slot_s and full.n_img == n_slot_s,
          f"the default trains on every slot and reports the on-disk width "
          f"(len={len(full)} n_img={full.n_img})")
    check(len(one) == n_conc_s and one.n_img == 1,
          f"--train-slots 0 is one item per concept (len={len(one)} n_img={one.n_img})")
    check(len(two) == 2 * n_conc_s and two.slots == [3, 7],
          f"--train-slots 3 7 is two items per concept, in sorted order "
          f"(len={len(two)} slots={two.slots})")

    # The decisive one: for a given (concept, slot), the target the subsetted dataset
    # hands out must be the SAME value the full dataset handed out, and it must equal
    # concept * 10 + slot. Anything else is a silent mispairing.
    #
    # Matched on the EEG array with `array_equal`, not on `float(x.sum())`: the two
    # are summed by torch and by numpy respectively, whose orders differ, so float
    # sums are not a safe identity here -- an earlier version of this test reported
    # six spurious mismatches for exactly that reason.
    def value_for(ds, want_c, want_slot):
        for i in range(len(ds)):
            x, _f, c = ds[i][:3]
            if int(c) != want_c:
                continue
            if np.array_equal(x.numpy(), ds.eeg[want_c, want_slot]):
                return int(np.asarray(ds[i][3]).flatten()[0])
        return None

    mism = []
    for c in range(n_conc_s):
        for slot in (0, 3, 7):
            got = value_for(one if slot == 0 else two, c, slot)
            want = c * n_slot_s + slot
            if got != want:
                mism.append((c, slot, got, want))
    check(not mism,
          f"a subsetted dataset reads the SAME on-disk target row the full one does "
          f"(mismatches {mism[:4]}, expected value == concept*10 + slot)")

    # The same identity, checked against the un-subsetted dataset: rows (c, slot) and
    # (c, slot') must carry different targets, otherwise the previous check could pass
    # on a dataset that ignores `slots` entirely and always reads slot 0.
    full_vals = {i: int(np.asarray(full[i][3]).flatten()[0]) for i in range(len(full))}
    distinct = len(set(full_vals.values())) == len(full_vals)
    check(distinct and full_vals[0 * n_slot_s + 7] == 7 and full_vals[2 * n_slot_s + 3] == 23,
          f"the fixture actually distinguishes rows: {len(set(full_vals.values()))} "
          f"distinct of {len(full_vals)}; full[c=0,slot=7]={full_vals[7]} "
          f"full[c=2,slot=3]={full_vals[23]}")

    # A narrowed cache (only the slots being trained) is NOT the on-disk layout, so
    # it must be rejected rather than silently accepted with shifted rows.
    narrow = np.zeros((n_conc_s, 4, 4), dtype=np.float32)
    raised = ""
    try:
        AuxTargetDataset(eeg_s, feat_s, conc_s, aux_vae=narrow,
                         vae_mean=np.zeros(4, np.float32),
                         vae_std=np.ones(4, np.float32), slots=[0])
    except ValueError as e:
        raised = str(e)
    check("rows" in raised,
          f"a target cache sized by the narrowed width is rejected, not silently used "
          f"(raised: {raised[:60]!r})")

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print(f"  - {f}")
        raise SystemExit(1)
    print("all dual-tower structural checks passed")


if __name__ == "__main__":
    main()
