#!/usr/bin/env python3
"""CF-MSF Stage A: train concept-factorized multi-route EEG heads (sub-08).

Routes (each is an MLP z → 1024, L2-normalised at use-site):
  img    — gallery NCE against train-concept mean CLIP-image (G_img)
  text   — gallery NCE against train-concept CLIP-text (G_text)
  depth  — gallery NCE against train-concept mean CLIP-depth
  edge   — gallery NCE against train-concept mean CLIP-edge

WHY concept gallery, not instance bank
--------------------------------------
THINGS-EEG2's 200-way test is 200 images from 200 DISTINCT concepts, so
ranking images ≡ identifying concepts. Concept-mean targets denoise the
10 training images/concept; instance cosine is kept only as a weak auxiliary.

LEAK-FREE
---------
Gradients on `fit` concepts; checkpoint on `val_b` concepts (leakfree/split.json).
Test concepts never enter training or selection. Galleries are train-1654 only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocf_train import build_concept_bank, l2n, l2t  # noqa: E402
import leakfree as LF  # noqa: E402


class Head(nn.Module):
    """MLP z(in_dim) -> target(out_dim).

    in_dim and out_dim are SEPARATE: a route must EMIT a vector in the same space
    as the gallery it is scored against, and those two dimensions need not match
    (e.g. a concatenated multi-level target is 4096-d while the EEG feature is
    1024-d).  out_dim defaults to in_dim so every pre-existing route is unchanged.
    """

    def __init__(self, dim: int = 1024, hidden: int = 1024, depth: int = 2,
                 drop: float = 0.1, out_dim: int | None = None):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        layers: list[nn.Module] = []
        d = dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def gallery_nce(q: torch.Tensor, gallery: torch.Tensor, cid: torch.Tensor, tau: float) -> torch.Tensor:
    return F.cross_entropy((l2t(q) @ l2t(gallery).T) / tau, cid)


def cos_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return (1.0 - (l2t(pred) * l2t(tgt)).sum(-1)).mean()


def build_concept_mean(feat: np.ndarray, cid: np.ndarray, n_cls: int) -> np.ndarray:
    g = np.zeros((n_cls, feat.shape[1]), dtype=np.float64)
    cnt = np.zeros(n_cls, dtype=np.int64)
    for i in range(len(cid)):
        c = int(cid[i])
        g[c] += feat[i]
        cnt[c] += 1
    ok = cnt > 0
    g[ok] /= cnt[ok, None]
    g[~ok] = np.random.randn(int((~ok).sum()), feat.shape[1]) * 1e-3  # unused slots
    return l2n(g.astype(np.float32))


def two_way_stat(sim: torch.Tensor, labels: torch.Tensor, neg_idx: torch.Tensor) -> float:
    """Two-way identification: the correct column vs `neg_idx.shape[1]` distractors.

    `sim` is (n_val, n_gallery) already L2-normalised.  The correct column is masked
    out before gathering distractors, so a row can never be scored against itself.

    The SAME `neg_idx` is used for every epoch -- it is built once, from a seeded
    generator.  That is deliberate: with a shared distractor set the statistic becomes
    a PAIRED comparison across epochs, so the difference between two epochs' scores
    reflects the model rather than which distractors happened to be drawn.  Redrawing
    per epoch would add distractor noise on top of the quantity being measured, which
    is the exact failure mode that made `val_top1` useless as a selector: the whole
    gallery is 1654 columns, so top-1 has 1/1654 granularity and is essentially
    constant across epochs (`cfmsf_select_audit.py` measured the epoch it picks as
    near-random, and `cfmsf_sel_probe.py` measured the post-convergence curve as
    1.04x a matched pure-noise null).
    """
    n = sim.shape[0]
    rows = torch.arange(n, device=sim.device)
    masked = sim.clone()
    masked[rows, labels] = -1e4
    correct = sim[rows, labels][:, None]
    negs = masked.gather(1, neg_idx)
    wins = (correct > negs).float() + 0.5 * (correct == negs).float()
    return float(wins.mean().cpu())


def csls_t(sim: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Cross-domain similarity local scaling on a torch similarity matrix."""
    k = max(1, min(k, sim.shape[0] - 1, sim.shape[1] - 1))
    q = sim.topk(k, dim=1).values.mean(1, keepdim=True)
    b = sim.topk(k, dim=0).values.mean(0, keepdim=True)
    return 2.0 * sim - q - b


# Every statistic the trainer records, so that selection is a choice among MEASURED
# quantities rather than a hard-coded one.  The names here are the CLI-facing ones; the
# history rows use `SELECTOR_KEY` because `two_way` is stored as `val_two_way` for
# symmetry with the other val_b statistics.  Keeping the two in one place is what stops
# a rename in one spot from silently selecting on a missing key.
SELECTORS = ("val_top1", "mini_csls", "two_way", "mini_top1")
SELECTOR_KEY = {"val_top1": "val_top1", "mini_csls": "mini_csls",
                "two_way": "val_two_way", "mini_top1": "mini_top1"}


def train_route(name: str, ztr: np.ndarray, tgt_inst: np.ndarray, gallery: np.ndarray,
                cid: np.ndarray, fit_i: np.ndarray, val_i: np.ndarray,
                args, out: Path, dev: torch.device,
                out_dim: int = 1024) -> dict:
    """`out_dim` = target dimensionality (the route's head must emit a vector that
    lives in the SAME space as the gallery it is scored against).  Default 1024
    keeps every existing route bit-identical; the route-quality probe uses it to
    score CONCATENATED multi-level targets (e.g. image|blur|lowres) as one route."""
    torch.manual_seed(args.seed)
    head = Head(dim=ztr.shape[1], depth=args.depth, drop=args.drop, out_dim=out_dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    z_t = torch.from_numpy(ztr).to(dev)
    t_t = torch.from_numpy(tgt_inst).to(dev)
    g_t = torch.from_numpy(gallery).to(dev)
    cid_t = torch.from_numpy(cid.astype(np.int64)).to(dev)
    fit_t = torch.from_numpy(fit_i.astype(np.int64)).to(dev)
    val_t = torch.from_numpy(val_i.astype(np.int64)).to(dev)

    # every selector's key must be present from the start: the selection branch below
    # reads `best[select_by]`, so a missing key fails only when that selector is chosen
    best = {"val_top1": -1.0, "epoch": -1, "val_inst_cos": -2.0,
            "val_top1_shuffled": -1.0, "val_two_way": -1.0,
            "mini_top1": -1.0, "mini_csls": -1.0}
    hist = []
    ckpt = out / f"head_{name}.pth"

    # ---- selection statistics -------------------------------------------------
    # Selection is opt-in.  Every existing caller (`cfmsf_train.main`,
    # `cfmsf_route_probe`, `run_cfmsf_*.sh`) omits --select-by, so the checkpoint it
    # gets stays bit-identical to the ones behind the already reported numbers.
    #
    # Why this is a menu and not a single "better" rule.  `cfmsf_sel_probe.py` measured,
    # over 8 routes x 80 epochs on the same frozen encoder:
    #   * the post-convergence curve is 1.04x a matched pure-noise null -- there is no
    #     real "best epoch" to find, so any val rule is selecting on noise;
    #   * `val_top1` is the second-WORST of seven rules tested: its gallery is 1654
    #     concepts wide, so one trial moves it by 1/1654 and it is nearly constant;
    #   * on the CSLS+Sinkhorn composite the best rules were `two_way` (+0.0269) and
    #     `mini_csls` (+0.0244), both exact sign p=0.063 over 8 routes;
    #   * but `two_way` SATURATES: 86% of epochs land within 1% of its maximum, so its
    #     argmax is close to a coin flip among ~69 epochs -- the same failure mode that
    #     made `margin` anti-correlated and picked epoch 0 in job 581657.  `mini_csls`
    #     has the same measured gain with only 17% of epochs that close to its max.
    #
    # No non-legacy rule is the DEFAULT, because a +0.02 effect at p=0.063 is not yet
    # established and should be confirmed on a subject-level run before it is baked
    # into the reported pipeline.  `selection.saturation` below makes the saturation
    # failure mode a recorded number rather than something a reader has to notice.
    select_by = getattr(args, "select_by", "val_top1")
    if select_by not in SELECTORS:
        raise SystemExit(f"[FATAL] --select-by {select_by!r} not in {SELECTORS}")
    # one shared distractor draw for the whole route, so `two_way` is paired across
    # epochs -- see `two_way_stat`'s docstring for why this must not be redrawn
    _g = torch.Generator().manual_seed(int(getattr(args, "seed", 42)) + 7919)
    neg_idx = torch.randint(0, gallery.shape[0], (len(val_i), 64),
                            generator=_g).to(dev)
    _val_rows = torch.arange(len(val_i), device=dev)
    # `mini_*` scores against a gallery restricted to the concepts that actually occur
    # in val_b.  Same question as `val_top1`, but with one column per val concept
    # instead of 1654, so a single correct trial moves the statistic by 1/n_val_concepts
    # rather than 1/1654 -- that is the resolution `val_top1` lacks.
    _cons = np.unique(cid[val_i])
    _mini_col = torch.from_numpy(np.searchsorted(_cons, cid[val_i]).astype(np.int64)).to(dev)
    _mb = torch.from_numpy(l2n(gallery[_cons])).to(dev)
    if len(_cons) < 8:
        print(f"[{name}] WARNING: only {len(_cons)} val concepts; mini_* is very coarse")

    for ep in range(args.epochs):
        head.train()
        perm = fit_t[torch.randperm(len(fit_t), device=dev)]
        tot = 0.0
        n = 0
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            opt.zero_grad(set_to_none=True)
            q = head(z_t[idx])
            loss = gallery_nce(q, g_t, cid_t[idx], args.tau)
            if args.inst_weight > 0:
                loss = loss + args.inst_weight * cos_loss(q, t_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += 1
        sch.step()

        head.eval()
        with torch.no_grad():
            qv = head(z_t[val_t])
            qvn = l2t(qv)
            sim_v = qvn @ l2t(g_t).T
            top1 = float((sim_v.argmax(1).eq(cid_t[val_t])).float().mean().cpu())
            # shuffled control: does the head just latch onto gallery geometry?
            shuf = cid_t[val_t][torch.randperm(len(val_t), device=dev)]
            top1_shuf = float((sim_v.argmax(1).eq(shuf)).float().mean().cpu())
            icos = float((qvn * l2t(t_t[val_t])).sum(-1).mean().cpu())
            tw = two_way_stat(sim_v, cid_t[val_t], neg_idx)
            sim_mini = qvn @ _mb.T
            mini_t1 = float(sim_mini.argmax(1).eq(_mini_col).float().mean().cpu())
            mini_cs = float(csls_t(sim_mini).argmax(1).eq(_mini_col).float().mean().cpu())
        row = {"epoch": ep, "train_loss": tot / max(n, 1),
               "val_top1": top1, "val_top1_shuffled": top1_shuf,
               "val_inst_cos": icos, "val_two_way": tw,
               "mini_top1": mini_t1, "mini_csls": mini_cs}
        hist.append(row)
        # `val_top1` keeps its legacy tie-break on `inst_cos` so the legacy checkpoint
        # is bit-identical; the others tie-break on the legacy statistic itself.
        if select_by == "val_top1":
            cand, prev = (row["val_top1"], row["val_inst_cos"]), \
                         (best["val_top1"], best["val_inst_cos"])
        else:
            k = SELECTOR_KEY[select_by]
            cand, prev = (row[k], row["val_top1"]), (best[k], best["val_top1"])
        if cand > prev:
            best = {"val_top1": top1, "val_top1_shuffled": top1_shuf,
                    "val_inst_cos": icos, "val_two_way": tw,
                    "mini_top1": mini_t1, "mini_csls": mini_cs, "epoch": ep}
            torch.save({"state_dict": head.state_dict(), "epoch": ep,
                        "val_top1": top1, "val_two_way": tw,
                        "mini_top1": mini_t1, "mini_csls": mini_cs,
                        "route": name}, ckpt)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"[{name} ep{ep:03d}] loss={row['train_loss']:.3f} "
                  f"val_top1={top1:.4f} shuf={top1_shuf:.4f} icos={icos:.4f} "
                  f"tw={tw:.4f} mini={mini_t1:.4f} minic={mini_cs:.4f}")

    head.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["state_dict"])
    head.eval()
    # Saturation audit.  A selection statistic whose epochs all pile up near its own
    # maximum cannot rank epochs: its argmax is then decided by rank-level noise.  This
    # is the failure mode that made `margin` disagree with every other statistic and
    # pick epoch 0 in job 581657, and it is invisible in a single number, so it is
    # recorded and warned about here instead of being left for a reader to spot.
    sat = {}
    for s in SELECTORS:
        c = np.array([h[SELECTOR_KEY[s]] for h in hist], dtype=np.float64)
        span = float(c.max() - c.min())
        near = float((c >= c.max() - 0.01 * abs(c.max())).mean()) if c.max() else 1.0
        sat[s] = {"argmax_epoch": int(c.argmax()), "range": span,
                  "frac_epochs_within_1pct_of_max": near}
    if sat[select_by]["frac_epochs_within_1pct_of_max"] > 0.5:
        print(f"[{name}] WARNING: --select-by {select_by} is SATURATED "
              f"({sat[select_by]['frac_epochs_within_1pct_of_max']:.0%} of epochs are "
              f"within 1% of its maximum); the chosen epoch is close to arbitrary. "
              f"Consider a statistic with more resolution, e.g. mini_csls.")
    return {"best": best, "history_tail": hist[-5:], "ckpt": str(ckpt),
            "selection": {"by": select_by, "saturation": sat}}, head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--routes", type=str, default="img,text,depth,edge")
    ap.add_argument("--select-by", type=str, default="val_top1", choices=list(SELECTORS),
                    help=("which val_b statistic picks the head checkpoint. 'val_top1' is "
                          "the legacy rule and the default, so existing commands "
                          "reproduce their exact checkpoints. The selection probe "
                          "measured a 1654-way gallery top-1 as the second-worst of seven "
                          "rules, and found 'mini_csls' the best combination of gain "
                          "(+0.0244 on the CSLS+Sinkhorn composite, exact sign p=0.063) "
                          "and resolution (17%% of epochs near its maximum, vs 86%% for "
                          "'two_way' and 58%% for 'val_top1'). None of the alternatives is "
                          "the default because a p=0.063 effect should be confirmed on a "
                          "subject-level run first. `selection.saturation` in the report "
                          "records each statistic's resolution."))
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--gallery-cache", type=str, default=str(NB_ROOT / "outputs/uck/shared"))
    ap.add_argument("--clip-text-dir", type=str,
                    default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    ap.add_argument("--captions-jsonl", type=str,
                    default=str(NB_ROOT / "outputs/g2/captions/captions_train.jsonl"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--inst-weight", type=float, default=0.2,
                    help="auxiliary instance cosine weight (0 = concept-only)")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "conds").mkdir(parents=True, exist_ok=True)
    (out / "galleries").mkdir(parents=True, exist_ok=True)
    # clip_text_dir is per-subject
    sid = f"{args.test_subject:02d}"
    args.clip_text_dir = str(NB_ROOT / f"outputs/nda_ss/sub-{sid}/clip_text")

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ztr = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32))
    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", len(ztr))
    val_i = LF.rows_for(split, "val_b", len(ztr))
    if set(fit_i.tolist()) & set(val_i.tolist()):
        raise SystemExit("[FATAL] fit/val overlap")

    g_text, cid, phrases = build_concept_bank(Path(args.clip_text_dir), Path(args.captions_jsonl))
    if len(cid) != len(ztr):
        raise SystemExit(f"[FATAL] cid {len(cid)} vs z {len(ztr)}")
    n_cls = len(phrases)
    print(f"[cfmsf] sub-{sid} z={ztr.shape} gallery={n_cls} fit={len(fit_i)} val_b={len(val_i)}")

    # galleries
    gcache = Path(args.gallery_cache)
    g_img_p = gcache / "g_img_concept.npy"
    if g_img_p.is_file():
        g_img = l2n(np.load(g_img_p).astype(np.float32))
        if g_img.shape != (n_cls, 1024):
            raise SystemExit(f"[FATAL] g_img shape {g_img.shape}")
    else:
        raise SystemExit(f"[FATAL] missing {g_img_p}")
    g_text = l2n(g_text.astype(np.float32))

    cond = Path(args.cond_cache)
    img_tr = l2n(np.load(cond / "clip_img1024_train.npy").astype(np.float32))
    img_te = l2n(np.load(cond / "clip_img1024_test.npy").astype(np.float32))
    dep_tr = l2n(np.load(cond / "clip_depth1024_train.npy").astype(np.float32))
    dep_te = l2n(np.load(cond / "clip_depth1024_test.npy").astype(np.float32))
    edg_tr = l2n(np.load(cond / "clip_edge1024_train.npy").astype(np.float32))
    edg_te = l2n(np.load(cond / "clip_edge1024_test.npy").astype(np.float32))

    g_depth = build_concept_mean(dep_tr, cid, n_cls)
    g_edge = build_concept_mean(edg_tr, cid, n_cls)
    np.save(out / "galleries" / "g_img.npy", g_img)
    np.save(out / "galleries" / "g_text.npy", g_text)
    np.save(out / "galleries" / "g_depth.npy", g_depth)
    np.save(out / "galleries" / "g_edge.npy", g_edge)
    (out / "galleries" / "phrases.json").write_text(json.dumps(phrases), encoding="utf-8")

    # text instance auxiliary: per-row text embeds (same length as ztr)
    text_flat_tr = l2n(np.load(Path(args.clip_text_dir) / "train" / "text_flat_clip.npy").astype(np.float32))
    text_te = l2n(np.load(Path(args.clip_text_dir) / "test" / "text_concept_clip.npy").astype(np.float32))
    if len(text_flat_tr) != len(ztr) or len(text_te) != len(zte):
        raise SystemExit(f"[FATAL] text rows {len(text_flat_tr)}/{len(text_te)}")

    route_cfg = {
        # (instance_train, instance_test_bank, gallery)
        "img":   (img_tr, img_te, g_img),
        "text":  (text_flat_tr, text_te, g_text),
        "depth": (dep_tr, dep_te, g_depth),
        "edge":  (edg_tr, edg_te, g_edge),
    }

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    report: dict = {"subject": f"sub-{sid}", "params": {k: getattr(args, k)
                    for k in ("epochs", "lr", "tau", "inst_weight", "depth", "drop", "seed",
                              "batch_size", "weight_decay")},
                    "routes": {}, "n_cls": n_cls, "fit": len(fit_i), "val_b": len(val_i)}

    for name in routes:
        if name not in route_cfg:
            raise SystemExit(f"[FATAL] unknown route {name}")
        inst_tr, inst_te, gal = route_cfg[name]
        info, head = train_route(name, ztr, inst_tr, gal, cid, fit_i, val_i, args, out, dev)
        with torch.no_grad():
            qte = head(torch.from_numpy(zte).to(dev)).cpu().numpy().astype(np.float32)
            qtr = head(torch.from_numpy(ztr).to(dev)).cpu().numpy().astype(np.float32)
        np.save(out / "conds" / f"q_{name}_test.npy", l2n(qte))
        np.save(out / "conds" / f"q_{name}_train.npy", l2n(qtr))

        # post-hoc: 200-way against the matching TEST bank (honest per-route metric)
        bank = inst_te
        sim = l2n(qte) @ l2n(bank).T
        top1 = float((sim.argmax(1) == np.arange(200)).mean())
        top5 = float(np.mean([i in np.argsort(-sim[i])[:5] for i in range(200)]))
        info["test200_raw"] = {"top1": top1, "top5": top5, "bank": f"{name}_test"}
        # neural-address check on VAL: paired vs shuffled concept labels
        info["neural_address_val"] = {
            "paired_top1": info["best"]["val_top1"],
            "shuffled_top1": info["best"]["val_top1_shuffled"],
            "gain": info["best"]["val_top1"] - info["best"]["val_top1_shuffled"],
            "chance": 1.0 / n_cls,
            "NOTE": "val_b concepts only; tests whether the head recovers concept identity",
        }
        report["routes"][name] = info
        print(f"[{name}] best_ep={info['best']['epoch']} val_top1={info['best']['val_top1']:.4f} "
              f"test200_raw top1={top1:.4f} top5={top5:.4f}")

    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[cfmsf-train] wrote {out}")


if __name__ == "__main__":
    main()
