#!/usr/bin/env python
"""v6 regression tests: the faithful deploy-stack episode (pillar A).

Run on the LOGIN NODE (CPU, seconds):
    python scripts/smoke_v6.py

These are the assertions that make "the episode textually matches deployment" a check
rather than a comment.  Each one failed at some point in this project's history, so each
one is written as a property of the code rather than a restatement of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import calibration  # noqa: E402
from samclip.losses.episode import deploy_stack_episode, whiten_map  # noqa: E402
from samclip.losses.contrastive import csls_correct  # noqa: E402
from samclip.losses.recovery import orthogonal_procrustes  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'' if not detail else '  ' + detail}")
    if not ok:
        FAILED.append(name)


def low_rank(n: int, d: int, rank: int, seed: int = 0) -> torch.Tensor:
    """`(n, d)` with genuine rank `rank` -- the regime that broke the SVD backward.

    `L < d` with rank-16 embeddings is the DEFAULT case here, not an edge case: the
    concept manifold is measured at 16 dimensions and an episode's landmark set is
    smaller than the embedding width.  A test on full-rank Gaussian input would pass
    while the real code produced NaN gradients (that is exactly what happened).
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, rank, generator=g)
    b = torch.randn(rank, d, generator=g)
    return a @ b + 0.01 * torch.randn(n, d, generator=g)


def t_whiten_matches_numpy() -> None:
    print("[1] whiten_map == calibration.saw_whiten (the deployment operator)")
    x = low_rank(42, 64, 16, seed=1).numpy()
    q = calibration.saw_whiten(x, shrink=0.1)[0]
    mu, w_map, diag = whiten_map(torch.as_tensor(x, dtype=torch.float32), shrink=0.1)
    got = ((torch.as_tensor(x, dtype=torch.float32) - mu) @ w_map).numpy()
    rel = float(np.abs(got - q).max() / max(np.abs(q).max(), 1e-9))
    check("max relative deviation < 1e-5", rel < 1e-5, f"rel={rel:.3e}")
    check("rank-deficient flag reported", bool(diag["rank_deficient"]),
          f"n={diag['n_samples']} d={diag['d_embed']}")


def t_csls_matches_numpy() -> None:
    print("[2] torch csls_correct == calibration.csls_scores")
    g = torch.Generator().manual_seed(2)
    a = torch.nn.functional.normalize(torch.randn(30, 64, generator=g), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(50, 64, generator=g), dim=-1)
    got = csls_correct(a @ b.t(), k=10).numpy()
    want = calibration.csls_scores(a.numpy(), b.numpy(), k=10)
    d = float(np.abs(got - want).max())
    check("max abs deviation < 1e-5", d < 1e-5, f"max|Δ|={d:.3e}")


def t_gradient_finite() -> None:
    print("[3] episode gradient finite for L<d, rank-16, rho in {0.05,0.1,0.3,1.0}")
    for rho in (0.05, 0.1, 0.3, 1.0):
        x = torch.nn.functional.normalize(
            low_rank(42, 64, 16, seed=3), dim=-1).requires_grad_(True)
        gal = torch.nn.functional.normalize(low_rank(64, 64, 16, seed=4), dim=-1)
        z, diag = deploy_stack_episode(x, gal, k=10, rho=rho, min_landmarks=4)
        loss = (z ** 2).sum()
        loss.backward()
        gfin = bool(torch.isfinite(x.grad).all())
        check(f"rho={rho:<4} loss finite={bool(torch.isfinite(loss))} grad finite={gfin} "
              f"abstained={diag['abstained']}", gfin)


def t_recovers_known_rotation() -> None:
    print("[4] episode recovers a known rotation (the mechanism, not just no-NaN)")
    g = torch.Generator().manual_seed(5)
    base = torch.nn.functional.normalize(torch.randn(200, 64, generator=g), dim=-1)
    # A block-diagonal rotation in the first 16 dims, then a gallery = rotated EEG.  The
    # episode sees no pairing; it must find the rotation from mutual-NN landmarks alone.
    theta = torch.linspace(0.0, 1.5, 8)
    r_true = torch.eye(64)
    for i, th in enumerate(theta):
        c, s = torch.cos(th), torch.sin(th)
        i0, i1 = 2 * i, 2 * i + 1
        r_true[i0, i0], r_true[i1, i1] = c, c
        r_true[i0, i1], r_true[i1, i0] = -s, s
    gallery = base @ r_true
    before = (base @ gallery.t()).diagonal().mean().item()
    z, diag = deploy_stack_episode(base, gallery, k=10, rho=0.05, min_landmarks=8,
                                   whiten=False)
    after = torch.nn.functional.normalize(z, dim=-1) @ gallery.t()
    after = after.diagonal().mean().item()
    check("diagonal similarity rises after recovery", after > before + 0.05,
          f"{before:.4f} -> {after:.4f}  (mutual={diag.get('n_mutual')})")


def t_abstains() -> None:
    print("[5] episode abstains instead of fitting noise")
    g = torch.Generator().manual_seed(6)
    x = torch.nn.functional.normalize(torch.randn(6, 64, generator=g), dim=-1)
    y = torch.nn.functional.normalize(torch.randn(6, 64, generator=g), dim=-1)
    z, diag = deploy_stack_episode(x, y, k=10, min_landmarks=8)
    check("abstained with too few landmarks", bool(diag["abstained"]))
    check("input returned unchanged", torch.equal(z, x))


def t_procrustes_orthogonal() -> None:
    print("[6] orthogonal_procrustes is orthogonal, detached and scale-robust")
    # Rows are normalised because that is what every real caller passes (the episode
    # normalises before choosing landmarks). `|x| ~ 0.01/sqrt(d)` dominates the scale of
    # `x^T y`, so a test on unnormalised rows measures the harness, not the code.
    x = torch.nn.functional.normalize(low_rank(42, 64, 16, seed=7), dim=-1)
    y = torch.nn.functional.normalize(low_rank(42, 64, 16, seed=8), dim=-1)
    r, diag = orthogonal_procrustes(x.requires_grad_(True), y, rho=0.1)
    err = float((r.t() @ r - torch.eye(64)).abs().max())
    check("orthogonality err < 1e-5", err < 1e-5, f"err={err:.3e}")
    check("R detached", not r.requires_grad)
    check("det > 0 (a rotation, not a reflection)", diag["det"] > 0, f"det={diag['det']:.4f}")
    # The scale-robustness regression: the SAME directions at 100x the magnitude must stay
    # FINITE. This was NaN before the solve moved to float64, and the failing
    # configuration (rank-deficient `x^T y`, small rho) is the default one. Note the
    # rotation itself is NOT expected to be identical at 100x -- `rho I` does not scale,
    # so the fit/regularisation balance genuinely changes. Asserting scale-invariance here
    # would be asserting a property the objective does not have.
    r_big, diag_big = orthogonal_procrustes(100.0 * x, 100.0 * y, rho=0.1)
    check("finite and orthogonal at 100x scale",
          bool(torch.isfinite(r_big).all()) and diag_big["orthogonality_err"] < 1e-5,
          f"orth_err={diag_big['orthogonality_err']:.3e}")


def t_rho_zero_rejected() -> None:
    print("[7] rho = 0 is rejected (partial isometry is not a rotation)")
    try:
        orthogonal_procrustes(low_rank(10, 8, 4, 9), low_rank(10, 8, 4, 10), rho=0.0)
        check("raises on rho=0", False)
    except ValueError:
        check("raises on rho=0", True)


def _block_rotation(d: int, n_block: int, angle: float) -> torch.Tensor:
    """Rotation by `angle` in each of the first `n_block` coordinate pairs."""
    r = torch.eye(d)
    for i in range(n_block):
        c, s = float(torch.cos(torch.tensor(angle))), float(torch.sin(torch.tensor(angle)))
        i0, i1 = 2 * i, 2 * i + 1
        r[i0, i0], r[i1, i1] = c, c
        r[i0, i1], r[i1, i0] = -s, s
    return r


def t_refine_with_reps() -> None:
    print("[8] C2 rep-refinement recovers a subject shift (T2 mechanism)")
    g = torch.Generator().manual_seed(11)
    C, d, R = 200, 64, 12
    gal = torch.nn.functional.normalize(torch.randn(C, d, generator=g), dim=-1)
    # A shift strong enough to actually degrade the raw ranking (measured: 78.0), so the
    # test measures the refinement's mechanism rather than saturating at 100 twice.
    q_shift = _block_rotation(d, 32, 1.2)         # the subject's unknown misalignment
    q_bar = torch.nn.functional.normalize(gal @ q_shift, dim=-1)
    reps = q_bar[:, None, :] + 0.10 * torch.randn(C, R, d, generator=g)
    raw = calibration.report_with_scores(
        calibration.csls_scores(q_bar.numpy(), gal.numpy(), k=10))["top1"]
    scores, diag = calibration.refine_with_reps(reps.numpy(), gal.numpy(), k=10)
    ref = calibration.report_with_scores(scores)["top1"]
    check("refinement strictly improves the shifted query",
          ref > raw + 8.0, f"raw {raw:.1f} -> refined {ref:.1f}  "
          f"(validated {diag.get('n_validated')}/{diag.get('n_mutual')} mutual, "
          f"agree={diag.get('mean_rep_agreement')})")
    check("landmarks were validated by repetition agreement",
          int(diag.get("n_validated") or 0) >= 8)


def t_refine_is_label_free() -> None:
    print("[9] C2 refinement does not read the pairing (label-free check)")
    g = torch.Generator().manual_seed(12)
    C, d, R = 200, 64, 8
    gal = torch.nn.functional.normalize(torch.randn(C, d, generator=g), dim=-1)
    phi = _block_rotation(d, 8, 0.5)
    q = torch.nn.functional.normalize(gal @ phi, dim=-1)
    reps = q[:, None, :] + 0.1 * torch.randn(C, R, d, generator=g)
    # PERMUTE the concepts' rows: a label-free operator is invariant to a consistent
    # permutation of the query/repetition rows (it only ever sees two clouds).
    perm = torch.randperm(C, generator=g)
    s1, _ = calibration.refine_with_reps(reps.numpy(), gal.numpy(), k=10)
    s2, _ = calibration.refine_with_reps(reps[perm].numpy(), gal.numpy(), k=10)
    # Under a consistent row permutation the score matrix must be the SAME up to rows.
    # The correct pairing for row k of `s2` is gallery index `perm[k]`, so Top-1 is read
    # against the permuted target rather than against the diagonal.
    same = float(np.abs(s2 - s1[perm.numpy()]).max())
    pred2 = s2.argmax(axis=1)
    top1_2 = float((pred2 == perm.numpy()).mean() * 100.0)
    top1_1 = calibration.report_with_scores(s1)["top1"]
    check("score matrix is equivariant to the row permutation", same < 1e-9,
          f"max|Δ|={same:.2e}")
    check("Top-1 invariant to a consistent row permutation", abs(top1_1 - top1_2) < 1e-9,
          f"{top1_1:.2f} vs {top1_2:.2f}")


def _tiny_route(name: str, dim: int, n_layers: int) -> dict:
    return {"name": name, "feature_set": "synthetic", "layers": list(range(n_layers)),
            "n_layers": n_layers, "image_dim": dim}


def _tiny_v6_cfg() -> dict:
    return {
        "objective": "v6",
        "arch": "v4",
        "n_channels": 63, "n_timepoints": 250, "n_subjects": 4,
        "n_target_layers": 2, "image_dim": 8,
        "d_model": 32, "d_embed": 32, "n_heads": 2, "n_blocks": 1, "dim_ff": 32,
        "d_align": 16, "d_latent": 16,
        "dropout": 0.0, "head_dropout": 0.0,
        "target_fusion": "mean", "smn": {"enabled": True},
        "loss_weights": {"img": 1.0, "cross": 0.5, "mmd": 0.1, "dec": 0.0,
                         "proto": 0.0, "reg": 0.0, "rkd": 0.0, "adv": 0.0,
                         "spec": 0.0, "router": 0.0},
        "recovery_aware": {"enabled": True, "block_per_subject": True, "csls_k": 10,
                           "terms": ["img", "cross"]},
        "fusion": {"weight": 1.0, "normalize": True},
        "seed": 2025,
    }


def _tiny_v6_model(cfg: dict):
    from samclip.models.multiroute import MultiRouteSAMCLIP

    routes = [_tiny_route("alpha", 8, 2), _tiny_route("gamma", 6, 1),
              _tiny_route("beta", 4, 1)]
    return MultiRouteSAMCLIP(cfg, routes, cfg), routes


def _tiny_v6_batch(cfg: dict, routes: list[dict], n_subjects: int = 3,
                   per_subject: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n = n_subjects * per_subject
    eeg = torch.randn(n, cfg["n_channels"], cfg["n_timepoints"], generator=g)
    subject = torch.arange(n_subjects).repeat_interleave(per_subject)
    stimulus = torch.arange(n) // n_subjects          # a clean 1-to-1 pairing
    primary = routes[0]["name"]
    batch = {
        "eeg": eeg, "subject": subject.long(), "stimulus": stimulus.long(),
        # `target` carries a GLOBAL stimulus id per row (as the real loader does), so
        # build it from `stimulus` and make each route's stack a deterministic function
        # of it -- that is what gives the fused score matrix a learnable signal.
        "target": _fake_stack(stimulus, routes[0]["n_layers"], routes[0]["image_dim"]),
        "concept": stimulus,
    }
    for spec in routes[1:]:
        batch[f"target__{spec['name']}"] = _fake_stack(
            stimulus, spec["n_layers"], spec["image_dim"])
    return batch, primary


def _fake_stack(stimulus: torch.Tensor, n_layers: int, dim: int) -> torch.Tensor:
    """`(N, n_layers, dim)` from a stimulus id, so identical stimuli share a target."""
    g = torch.Generator().manual_seed(1234)
    codebook = torch.randn(int(stimulus.max()) + 1, n_layers, dim, generator=g)
    return codebook[stimulus]


def _rotation_on(d: int, lo: int, hi: int, angle: float) -> torch.Tensor:
    """Identity except for a rotation by `angle` in coordinate pairs `[lo, hi)`."""
    r = torch.eye(d)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    for i in range(lo, hi, 2):
        r[i, i], r[i + 1, i + 1] = c, c
        r[i, i + 1], r[i + 1, i] = -s, s
    return r


def t_score_fusion_mechanism() -> None:
    print("[10] score fusion: complementary routes beat the best route alone")
    from samclip.losses.contrastive import InfoNCE, score_fusion_loss

    g = torch.Generator().manual_seed(21)
    C, d, half = 200, 32, 8
    gal = torch.nn.functional.normalize(torch.randn(C, d, generator=g), dim=-1)
    # FOUR routes, each carrying a DISJOINT 8-dim slice of the coordinates and zeroes
    # elsewhere. A route alone retrieves in its 8-dim subspace (measured here at ~55-65%
    # Top-1, i.e. clearly weak), while the sum sees all 32 dimensions. This is the
    # mechanism late fusion exists for, so a test that only checked "the fused loss is
    # finite" could not distinguish fusion from averaging. The slice size matters and was
    # measured, not guessed: 16 dims per route already scores 99.5% alone, which would
    # make the assertion vacuous.
    routes = []
    for lo in range(0, d, half):
        e = torch.zeros_like(gal)
        e[:, lo:lo + half] = gal[:, lo:lo + half]
        routes.append(torch.nn.functional.normalize(e, dim=-1))
    gal_n = torch.nn.functional.normalize(gal, dim=-1)
    crit = InfoNCE(init_temp=0.07)
    per_route = [calibration.report_with_scores(e.numpy() @ gal_n.numpy().T)["top1"]
                 for e in routes]
    fused_mat = calibration.fuse_scores([(e @ gal_n.t()).numpy() for e in routes],
                                        normalize=True)
    fused_top1 = calibration.report_with_scores(fused_mat)["top1"]
    check("each route alone is weak", max(per_route) < 75.0,
          f"route top-1 {[round(x, 1) for x in per_route]}")
    check("the fused score matrix beats the best route", fused_top1 > max(per_route) + 15.0,
          f"best route {max(per_route):.1f} -> fused {fused_top1:.1f}")
    l = score_fusion_loss([e @ gal_n.t() for e in routes], crit)
    check("score_fusion_loss is finite and positive",
          bool(torch.isfinite(l)) and float(l) > 0, f"loss={float(l):.4f}")
    # Normalisation is load-bearing: scale one route's scores by 50 and the unnormalised
    # sum is essentially that route alone, so the fusion gain disappears. The normalised
    # sum is unchanged. Both halves of that are asserted -- "close to route 0" alone would
    # also pass if fusion never helped in the first place.
    big = [50.0 * (routes[0] @ gal_n.t())] + [e @ gal_n.t() for e in routes[1:]]
    off_top1 = calibration.report_with_scores(
        calibration.fuse_scores([x.numpy() for x in big], normalize=False))["top1"]
    on_top1 = calibration.report_with_scores(
        calibration.fuse_scores([x.numpy() for x in big], normalize=True))["top1"]
    check("unnormalised fusion is dragged back to the loud route",
          abs(off_top1 - per_route[0]) < 10.0 and off_top1 < fused_top1 - 15.0,
          f"{off_top1:.1f} vs route-0 {per_route[0]:.1f}, fused {fused_top1:.1f}")
    check("normalisation removes the scale and restores the fusion", on_top1 == fused_top1,
          f"{on_top1:.1f} == {fused_top1:.1f}")


def t_multiroute_shared_trunk() -> None:
    print("[11] multi-route model: ONE trunk, per-route heads, no duplicated parameters")
    from samclip.models.multiroute import dedup_parameters

    cfg = _tiny_v6_cfg()
    model, routes = _tiny_v6_model(cfg)
    batch, primary = _tiny_v6_batch(cfg, routes)
    out = model(batch["eeg"], {r["name"]: batch.get(f"target__{r['name']}",
                                                    batch["target"]) for r in routes},
                subject_ids=batch["subject"], training=True)
    check("forward emits one view per route",
          set(out["routes"]) == {r["name"] for r in routes}, f"{sorted(out['routes'])}")
    check("legacy keys point at the primary route",
          out["z_eeg"] is out["routes"][primary]["z_eeg"])
    check("`model.smn` is exposed for the run record (a wrapper with no `smn` would make "
          "the training log print `smn=off` for a model whose routes all have one)",
          model.smn is not None and set(model.smn_gates()) == {r["name"] for r in routes},
          f"gates={ {k: round(v, 4) for k, v in model.smn_gates().items()} }")
    # What sharing LOOKS like on disk: `state_dict` is a recursive traversal and does NOT
    # de-duplicate, so the one trunk appears under every route's prefix. That is the
    # signature the sharing happened.
    prefixes = {k.split(".trunk.")[0] for k in model.state_dict() if ".trunk." in k}
    check("state_dict exposes the shared trunk under every route", len(prefixes) == len(routes),
          f"{sorted(prefixes)}")
    uniq = dedup_parameters(model)
    n_trunk = sum(p.numel() for p in model.trunk.parameters())
    check("dedup_parameters agrees with nn.Module.parameters() (which de-duplicates "
          "itself; the helper makes it explicit and stable across traversal styles)",
          [id(p) for p in uniq] == [id(p) for p in model.parameters() if p.requires_grad],
          f"{len(uniq)} tensors")
    check("dedup keeps every trunk scalar exactly once",
          sum(p.numel() for p in uniq
              if any(p is q for q in model.trunk.parameters())) == n_trunk,
          f"{n_trunk} trunk scalars")
    # The shared-trunk premise: a gradient on ONE route must move the trunk. If the routes
    # had their own trunks this would still pass, so it is paired with the state_dict count
    # above.
    loss = out["routes"][primary]["z_eeg"].pow(2).sum()
    loss.backward()
    gnorm = sum(float(p.grad.abs().sum()) for p in model.trunk.parameters()
                if p.grad is not None)
    check("gradient reaches the shared trunk from the route loss", gnorm > 0,
          f"|grad|={gnorm:.3e}")


def t_assemble_v6_end_to_end() -> None:
    print("[12] Trainer.assemble on objective=v6: every term present, gradient finite")
    from samclip.train import LossWeights, Trainer

    cfg = _tiny_v6_cfg()
    model, routes = _tiny_v6_model(cfg)
    batch, primary = _tiny_v6_batch(cfg, routes)
    trainer = Trainer(model=model, cfg=cfg, device=torch.device("cpu"), n_subjects=3,
                      weights=LossWeights.from_cfg(cfg))
    loss, parts = trainer.assemble(batch)
    for key in ("img", "fuse", "cross", "mmd", "sc_img", "sc_x", "total"):
        check(f"part {key!r} present", key in parts, f"{float(parts.get(key, float('nan'))):.4f}")
    for r in routes:
        key = f"img_{r['name']}"
        check(f"per-route term {key}", key in parts,
              f"{float(parts.get(key, float('nan'))):.4f}")
    check("loss is finite and positive", bool(torch.isfinite(loss)) and float(loss) > 0,
          f"loss={float(loss):.4f}")
    loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all()))]
    check("every trainable parameter gets a finite gradient", not bad, f"{bad[:4]}")
    check("the fusion actually ran", float(parts["fuse_blocks"]) >= 1.0,
          f"blocks={float(parts['fuse_blocks']):.0f}")


def t_assemble_v6_is_v4_when_one_route() -> None:
    print("[13] a ONE-route v6 run is v4's objective plus the (trivial) fusion term")
    from samclip.train import Trainer, LossWeights, clip_alignment_loss

    cfg = _tiny_v6_cfg()
    routes = [_tiny_route("alpha", 8, 2)]
    from samclip.models.multiroute import MultiRouteSAMCLIP

    model = MultiRouteSAMCLIP(cfg, routes, cfg)
    batch, _ = _tiny_v6_batch(cfg, routes)
    trainer = Trainer(model=model, cfg=cfg, device=torch.device("cpu"), n_subjects=3,
                      weights=LossWeights.from_cfg(cfg))
    loss, parts = trainer.assemble(batch)
    # With one route the fused matrix IS the (normalised) route matrix, so the fused term
    # must equal the same contrast on that matrix -- i.e. the fusion adds nothing
    # surprising. Compared against a recomputation rather than against `img` because
    # `score_fusion_loss` divides by the matrix std and `recovery_aware_alignment` scores
    # per block; the equality asserted is only that the term is that contrast.
    views = model(batch["eeg"], {"alpha": batch["target"]},
                  subject_ids=batch["subject"], training=True)["routes"]["alpha"]
    s = views["z_eeg"] @ views["z_img"].t()
    from samclip.losses.contrastive import csls_correct, score_fusion_loss

    blocks = []
    for subj in torch.unique(batch["subject"]):
        rows = batch["subject"] == subj
        if int(rows.sum()) < 2:
            continue
        blocks.append([csls_correct(views["z_eeg"][rows] @ views["z_img"][rows].t(),
                                    k=trainer.ra_csls_k)])
    want = torch.stack([score_fusion_loss(b, trainer.crit_img) for b in blocks]).mean()
    check("the fused term equals the single-route contrast in the deployed metric",
          abs(float(want) - float(parts["fuse"])) < 1e-6,
          f"{float(want):.6f} vs {float(parts['fuse']):.6f}")
    check("only one route reported", len([k for k in parts if k.startswith("img_")]) == 1)
    _ = (s, clip_alignment_loss, LossWeights)   # imported for the reader, unused


def t_v6_rejects_inert_config() -> None:
    print("[14] v6 refuses a removal-term weight and a fusion block on a v3 run")
    from samclip.train import LossWeights, Trainer

    w = LossWeights.from_cfg({"loss_weights": {"reg": 0.5}})
    try:
        w.validate("v6")
        check("reg > 0 on v6 is rejected", False)
    except ValueError:
        check("reg > 0 on v6 is rejected", True)
    cfg = {"objective": "v3", "fusion": {"weight": 0.3}}
    try:
        Trainer(model=torch.nn.Linear(2, 2), cfg=cfg, device=torch.device("cpu"),
                n_subjects=2)
        check("a fusion block on objective=v3 is rejected", False)
    except ValueError:
        check("a fusion block on objective=v3 is rejected", True)


def t_loader_carries_route_targets() -> None:
    print("[15] the datasets expose target__<route> keys and the plain primary target")
    from samclip.data import things_eeg

    te = np.zeros((6, 1, 4, 8), dtype=np.float32)
    tg = np.zeros((6, 1, 2, 5), dtype=np.float32)
    ds = things_eeg.TestDataset(te, tg,
                                extra_targets={"beta": np.zeros((6, 1, 1, 3),
                                                                dtype=np.float32)})
    item = ds[0]
    check("primary stays under `target`", item["target"].shape == (2, 5),
          f"{tuple(item['target'].shape)}")
    check("non-primary route under target__beta",
          item["target__beta"].shape == (1, 3), f"{tuple(item['target__beta'].shape)}")
    batched = things_eeg.collate([ds[0], ds[1]])
    check("collate stacks the route key to (B, K, D)",
          tuple(batched["target__beta"].shape) == (2, 1, 3),
          f"{tuple(batched['target__beta'].shape)}")


def main() -> None:
    torch.manual_seed(0)
    for fn in (t_whiten_matches_numpy, t_csls_matches_numpy, t_gradient_finite,
               t_recovers_known_rotation, t_abstains, t_procrustes_orthogonal,
               t_rho_zero_rejected, t_refine_with_reps, t_refine_is_label_free,
               t_score_fusion_mechanism, t_multiroute_shared_trunk,
               t_assemble_v6_end_to_end, t_assemble_v6_is_v4_when_one_route,
               t_v6_rejects_inert_config, t_loader_carries_route_targets):
        fn()
    print()
    if FAILED:
        print(f"smoke_v6: {len(FAILED)} FAILED -> {FAILED}")
        raise SystemExit(1)
    print("smoke_v6: all checks passed")


if __name__ == "__main__":
    main()
