#!/usr/bin/env python3
"""UCK-NAT K-hypothesis: address + residual heads → K IP conditions.

Pipeline (sub-08 feasibility):
  Stage A   neural address on frozen μ  (optionally light top-K NCE on a projector)
  Stage A′  residual head τ(r) predicting instance offset Δ = G_img_i − G_img[cid]
  Stage B   assemble  c_k = l2( IP_uck + λ · (G[addr_k] + τ(r) − IP_uck) )
            so λ=0 reproduces UCK bit-wise; λ=1 is pure address+residual

Diversity is guaranteed by disjoint addr_k (concept anchors), not by a diversity loss.
Does NOT retrain UCK IP / F. Leak-free: fit for gradients, val_b for selection.
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

from ocf_train import l2n, l2t, build_concept_bank  # noqa: E402
import leakfree as LF  # noqa: E402


def concept_means(z: np.ndarray, cid: np.ndarray, n_cls: int) -> np.ndarray:
    mu = np.zeros((n_cls, z.shape[1]), dtype=np.float32)
    cnt = np.zeros(n_cls, dtype=np.int64)
    for i, c in enumerate(cid):
        mu[c] += z[i]
        cnt[c] += 1
    alive = cnt > 0
    mu[alive] /= cnt[alive, None]
    return l2n(mu), cnt


def residual(z: np.ndarray, mu: np.ndarray, tau: float = 0.07
             ) -> tuple[np.ndarray, np.ndarray]:
    """Hard top-1 residual, then explicit Gram–Schmidt off the prototype.

    Soft residuals leave concept mass in r (probe ≫ chance). Hard projection
    + one GS step is the minimal fix that keeps r ≈ orthogonal to μ[top1].
    """
    sim = l2n(z) @ l2n(mu).T
    top = sim.argmax(1)
    proj = mu[top]
    r = z - proj
    ph = proj / np.clip(np.linalg.norm(proj, axis=-1, keepdims=True), 1e-8, None)
    r = r - (r * ph).sum(-1, keepdims=True) * ph
    # soft α kept only for logging / optional soft path
    alpha = np.exp(sim / tau)
    alpha = alpha / np.clip(alpha.sum(1, keepdims=True), 1e-8, None)
    return r.astype(np.float32), alpha.astype(np.float32)


def recall_at_k(sim: np.ndarray, cid: np.ndarray, ks: list[int]) -> dict:
    order = np.argsort(-sim, axis=1)
    out = {}
    for k in ks:
        hit = np.array([int(cid[i]) in set(order[i, :k].tolist()) for i in range(len(cid))])
        out[f"recall@{k}"] = float(hit.mean())
    out["top1"] = float((order[:, 0] == cid).mean())
    return out


class ResHead(nn.Module):
    def __init__(self, dim: int = 1024, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class AddrProj(nn.Module):
    """Optional light projector for top-K NCE addressing."""
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def topk_nce(q: torch.Tensor, bank: torch.Tensor, cid: torch.Tensor,
             k_neg: int = 64, tau: float = 0.07) -> torch.Tensor:
    """NCE that only keeps the hardest negatives (top-k_neg among non-positives)."""
    sim = (l2t(q) @ l2t(bank).T) / tau  # B x C
    B, C = sim.shape
    pos = sim[torch.arange(B, device=sim.device), cid]
    # mask positives
    mask = torch.ones_like(sim, dtype=torch.bool)
    mask[torch.arange(B, device=sim.device), cid] = False
    neg = sim.masked_fill(~mask, -1e4)
    topv, _ = neg.topk(min(k_neg, C - 1), dim=-1)
    logits = torch.cat([pos[:, None], topv], dim=1)
    return F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=sim.device))


def ridge_r2(x: np.ndarray, y: np.ndarray, rows_fit: np.ndarray, rows_val: np.ndarray,
             lam: float = 1e-2) -> float:
    """Per-dim ridge R² of y ~ x, averaged. Diagnostic only."""
    Xf, yf = x[rows_fit], y[rows_fit]
    Xv, yv = x[rows_val], y[rows_val]
    d = Xf.shape[1]
    # shared ridge: solve (X'X + λI) W = X'Y  for multi-output
    XtX = Xf.T @ Xf + lam * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(XtX, Xf.T @ yf)
    pred = Xv @ W
    ss_res = ((yv - pred) ** 2).sum(0)
    ss_tot = ((yv - yv.mean(0)) ** 2).sum(0) + 1e-8
    return float(np.mean(1.0 - ss_res / ss_tot))


def assemble(ip_uck: np.ndarray, g_img: np.ndarray, addr_idx: np.ndarray,
             delta: np.ndarray, lam: float) -> np.ndarray:
    """c = l2( IP + λ (G[addr] + Δ − IP) ). λ=0 → IP."""
    g = g_img[addr_idx]
    target = g + delta
    c = ip_uck + lam * (target - ip_uck)
    return l2n(c.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--captions-dir", type=str, default=str(NB_ROOT / "outputs/g2/captions"))
    ap.add_argument("--clip-text-dir", type=str,
                    default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    ap.add_argument("--clip-img-dir", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--gallery-cache", type=str, default=str(NB_ROOT / "outputs/uck/shared"))
    ap.add_argument("--uck-ip", type=str,
                    default=str(NB_ROOT / "outputs/uck/sub-08/full/conds/ip_mem_test.npy"))
    ap.add_argument("--uck-ip-train", type=str, default="",
                    help="optional train IP; if missing, use G_img[cid] as UCK proxy on train")
    ap.add_argument("--mu-npy", type=str, default="",
                    help="optional precomputed μ; else build from train z")
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--lambdas", type=str, default="0,0.3,0.5,1.0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--k-neg", type=int, default=64)
    ap.add_argument("--train-addr", type=int, default=1)
    ap.add_argument("--w-addr", type=float, default=0.5)
    ap.add_argument("--w-res", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--diag-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    (out / "conds").mkdir(parents=True, exist_ok=True)
    (out / "proto").mkdir(exist_ok=True)
    sid = f"{args.test_subject:02d}"
    sub = f"sub-{sid}"
    K = int(args.K)
    lams = [float(x) for x in args.lambdas.split(",") if x.strip()]

    ztr = l2n(np.load(Path(args.z_root) / sub / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / sub / "shared_r_test.npy").astype(np.float32))
    g_text, cid_np, phrases = build_concept_bank(
        Path(args.clip_text_dir), Path(args.captions_dir) / "captions_train.jsonl")
    if len(cid_np) != len(ztr):
        raise SystemExit(f"[FATAL] cid {len(cid_np)} vs z {len(ztr)}")
    n_cls = len(phrases)

    clip_tr = l2n(np.load(Path(args.clip_img_dir) / "clip_img1024_train.npy").astype(np.float32))
    clip_te = l2n(np.load(Path(args.clip_img_dir) / "clip_img1024_test.npy").astype(np.float32))
    g_img = l2n(np.load(Path(args.gallery_cache) / "g_img_concept.npy").astype(np.float32))
    if g_img.shape != (n_cls, 1024):
        raise SystemExit(f"[FATAL] G_img {g_img.shape}")

    ip_te = l2n(np.load(args.uck_ip).astype(np.float32))
    if len(ip_te) != len(zte):
        raise SystemExit(f"[FATAL] UCK IP {len(ip_te)} vs zte {len(zte)}")

    # train-side UCK proxy: prefer real train IP if provided, else concept mean
    if args.uck_ip_train and Path(args.uck_ip_train).is_file():
        ip_tr = l2n(np.load(args.uck_ip_train).astype(np.float32))
        if len(ip_tr) != len(ztr):
            raise SystemExit(f"[FATAL] train IP {len(ip_tr)} vs ztr {len(ztr)}")
    else:
        ip_tr = g_img[cid_np]

    if args.mu_npy and Path(args.mu_npy).is_file():
        mu = l2n(np.load(args.mu_npy).astype(np.float32))
        cnt = np.ones(n_cls, dtype=np.int64)
    else:
        mu, cnt = concept_means(ztr, cid_np, n_cls)
    if int((cnt == 0).sum()) > 0 and not args.mu_npy:
        print(f"[WARN] empty prototypes: {(cnt == 0).sum()}")
    np.save(out / "proto" / "mu_all.npy", mu)

    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", len(ztr))
    val_i = LF.rows_for(split, "val_b", len(ztr))
    if set(fit_i.tolist()) & set(val_i.tolist()):
        raise SystemExit("[FATAL] fit/val overlap")

    # ---------- diagnostics (zero-cost kill switches) ----------
    sim_tr = l2n(ztr) @ mu.T
    sim_val = l2n(ztr[val_i]) @ mu.T
    rec_all = recall_at_k(sim_tr, cid_np, [1, 2, 4, 8, 16])
    rec_val = recall_at_k(sim_val, cid_np[val_i], [1, 2, 4, 8, 16])
    r_tr, alpha_tr = residual(ztr, mu, args.tau)
    r_te, _ = residual(zte, mu, args.tau)
    delta_tr = clip_tr - g_img[cid_np]
    r2 = ridge_r2(r_tr, delta_tr, fit_i, val_i)
    # concept probe on residual (must ≈ chance)
    probe = (l2n(r_tr[val_i]) @ mu.T).argmax(1)
    probe_top1 = float((probe == cid_np[val_i]).mean())
    chance = 1.0 / n_cls

    diag = {
        "subject": sub,
        "recall_train": rec_all,
        "recall_val_b": rec_val,
        "residual_ridge_r2_val": r2,
        "residual_concept_probe_top1_val": probe_top1,
        "chance_1654": chance,
        "kill_switches": {
            "recall@8_val_ok": bool(rec_val.get("recall@8", 0) >= 0.30),
            "residual_r2_ok": bool(r2 > 0.01),
            "residual_no_concept_leak": bool(probe_top1 < 5 * chance),
        },
    }
    # if residual is useless or leaky, still run K-hyp but with Δ≡0 (pure anchors)
    use_residual = bool(diag["kill_switches"]["residual_r2_ok"]
                        and diag["kill_switches"]["residual_no_concept_leak"])
    diag["use_residual"] = use_residual
    if not use_residual:
        print("[WARN] residual kill-switch tripped → export Δ=0 (anchor-only K-hyp)")
    (out / "diag.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(json.dumps(diag, indent=2))
    if args.diag_only:
        print("[uck_nat] diag-only done")
        return

    # ---------- train residual (+ optional address projector) ----------
    res_head = ResHead().to(dev)
    addr_proj = AddrProj().to(dev) if args.train_addr else None
    params = list(res_head.parameters())
    if addr_proj is not None:
        params += list(addr_proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    mu_t = torch.from_numpy(mu).to(dev)
    g_t = torch.from_numpy(g_img).to(dev)
    z_t = torch.from_numpy(ztr).to(dev)
    r_t = torch.from_numpy(r_tr).to(dev)
    d_t = torch.from_numpy(delta_tr.astype(np.float32)).to(dev)
    cid_t = torch.from_numpy(cid_np.astype(np.int64)).to(dev)
    fit_t = torch.from_numpy(fit_i.astype(np.int64)).to(dev)
    val_t = torch.from_numpy(val_i.astype(np.int64)).to(dev)

    best = {"score": -1e9, "epoch": -1}
    hist = []
    for ep in range(args.epochs):
        res_head.train()
        if addr_proj is not None:
            addr_proj.train()
        perm = fit_t[torch.randperm(len(fit_t), device=dev)]
        tr_loss = tr_res = tr_addr = 0.0
        n_steps = 0
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = z_t.new_zeros(())
            # residual: only on samples whose true cid is in address top-K
            with torch.no_grad():
                q_for_addr = addr_proj(z_t[idx]) if addr_proj is not None else z_t[idx]
                sim = l2t(q_for_addr) @ l2t(mu_t).T
                topk = sim.topk(K, dim=-1).indices
                in_top = (topk == cid_t[idx][:, None]).any(dim=1)
            pred_d = res_head(r_t[idx])
            if in_top.any():
                l_res = F.mse_loss(pred_d[in_top], d_t[idx][in_top])
            else:
                l_res = pred_d.new_zeros(())
            loss = loss + args.w_res * l_res
            l_addr = pred_d.new_zeros(())
            if addr_proj is not None:
                l_addr = topk_nce(addr_proj(z_t[idx]), mu_t, cid_t[idx],
                                  k_neg=args.k_neg, tau=args.tau)
                loss = loss + args.w_addr * l_addr
            loss.backward()
            opt.step()
            tr_loss += float(loss.detach())
            tr_res += float(l_res.detach())
            tr_addr += float(l_addr.detach())
            n_steps += 1

        # val: residual MSE on in-top-K + address recall@K
        res_head.eval()
        if addr_proj is not None:
            addr_proj.eval()
        with torch.no_grad():
            qv = addr_proj(z_t[val_t]) if addr_proj is not None else z_t[val_t]
            sim_v = (l2t(qv) @ l2t(mu_t).T).cpu().numpy()
            rec_v = recall_at_k(sim_v, cid_np[val_i], [1, 4, 8])
            pred_v = res_head(r_t[val_t]).cpu().numpy()
            mse_v = float(np.mean((pred_v - delta_tr[val_i]) ** 2))
            # in-top-K subset MSE
            topk_np = np.argsort(-sim_v, axis=1)[:, :K]
            mask = np.array([cid_np[val_i][i] in set(topk_np[i].tolist()) for i in range(len(val_i))])
            mse_in = float(np.mean((pred_v[mask] - delta_tr[val_i][mask]) ** 2)) if mask.any() else mse_v
        score = rec_v.get("recall@8", 0) - 0.1 * mse_in
        row = {
            "epoch": ep, "tr_loss": tr_loss / max(n_steps, 1),
            "tr_res": tr_res / max(n_steps, 1), "tr_addr": tr_addr / max(n_steps, 1),
            "val_mse": mse_v, "val_mse_in_topk": mse_in, "val_recall": rec_v, "score": score,
        }
        hist.append(row)
        print(f"[ep{ep:02d}] loss={row['tr_loss']:.4f} res={row['tr_res']:.4f} "
              f"addr={row['tr_addr']:.4f} val_mse_in={mse_in:.4f} "
              f"R@8={rec_v.get('recall@8', 0):.3f}")
        if score > best["score"]:
            best = {"score": score, "epoch": ep, **rec_v, "mse_in": mse_in}
            ckpt = {
                "res_head": res_head.state_dict(),
                "addr_proj": (addr_proj.state_dict() if addr_proj is not None else None),
                "epoch": ep, "args": vars(args),
            }
            torch.save(ckpt, out / "best.pth")

    # ---------- export conditions ----------
    ckpt = torch.load(out / "best.pth", map_location=dev, weights_only=False)
    res_head.load_state_dict(ckpt["res_head"])
    if addr_proj is not None and ckpt.get("addr_proj") is not None:
        addr_proj.load_state_dict(ckpt["addr_proj"])
    res_head.eval()
    if addr_proj is not None:
        addr_proj.eval()

    with torch.no_grad():
        q_te = addr_proj(torch.from_numpy(zte).to(dev)) if addr_proj is not None \
            else torch.from_numpy(zte).to(dev)
        sim_te = (l2t(q_te) @ l2t(mu_t).T).cpu().numpy()
        addr_te = np.argsort(-sim_te, axis=1)[:, :K].astype(np.int64)
        if use_residual:
            delta_te = res_head(torch.from_numpy(r_te).to(dev)).cpu().numpy().astype(np.float32)
        else:
            delta_te = np.zeros_like(r_te)
        # also export train-side for audit
        q_tr = addr_proj(torch.from_numpy(ztr).to(dev)) if addr_proj is not None \
            else torch.from_numpy(ztr).to(dev)
        sim_tr2 = (l2t(q_tr) @ l2t(mu_t).T).cpu().numpy()
        addr_tr = np.argsort(-sim_tr2, axis=1)[:, :K].astype(np.int64)
        if use_residual:
            delta_tr_hat = res_head(torch.from_numpy(r_tr).to(dev)).cpu().numpy().astype(np.float32)
        else:
            delta_tr_hat = np.zeros_like(r_tr)

    np.save(out / "conds" / "addr_topk_test.npy", addr_te)
    np.save(out / "conds" / "addr_topk_train.npy", addr_tr)
    np.save(out / "conds" / "delta_test.npy", delta_te)
    np.save(out / "conds" / "delta_train.npy", delta_tr_hat)
    np.save(out / "conds" / "r_test.npy", r_te)
    np.save(out / "conds" / "ip_uck_test.npy", ip_te)

    # pairwise diversity of raw anchors
    div = {}
    for lam in lams:
        stack = []
        for k in range(K):
            c = assemble(ip_te, g_img, addr_te[:, k], delta_te, lam)
            np.save(out / "conds" / f"ip_lam{lam:g}_k{k}_test.npy", c)
            stack.append(c)
        stack_a = np.stack(stack, axis=1)  # N x K x D
        np.save(out / "conds" / f"ip_lam{lam:g}_K{K}_test.npy", stack_a)
        # mean pairwise cosine across K for each row
        pcs = []
        for i in range(len(stack_a)):
            X = l2n(stack_a[i])
            S = X @ X.T
            iu = np.triu_indices(K, 1)
            pcs.append(float(S[iu].mean()))
        div[f"lam{lam:g}_mean_pairwise_cos"] = float(np.mean(pcs))

    # λ=0 must equal UCK
    c0 = assemble(ip_te, g_img, addr_te[:, 0], delta_te, 0.0)
    fuse_ok = bool(np.allclose(c0, ip_te, atol=1e-5, rtol=1e-5))
    max_abs = float(np.max(np.abs(c0 - ip_te)))

    # random-K control: shuffle addr across rows (keeps marginal, breaks pairing)
    rng = np.random.default_rng(args.seed)
    addr_rand = addr_te.copy()
    for k in range(K):
        addr_rand[:, k] = rng.permutation(addr_te[:, k])
    for lam in (0.5, 1.0):
        for k in range(min(K, 4)):
            c = assemble(ip_te, g_img, addr_rand[:, k], delta_te, lam)
            np.save(out / "conds" / f"ip_rand_lam{lam:g}_k{k}_test.npy", c)

    # address quality on TEST is not reported as a selection metric (no labels used
    # during training selection); we still dump predicted names for audit only.
    caps_te = [json.loads(l) for l in (Path(args.captions_dir) / "captions_test.jsonl")
               .read_text(encoding="utf-8").splitlines() if l.strip()]

    def concept_of(p: str) -> str:
        return Path(p).parent.name.split("_", 1)[1].replace("_", " ")

    con_te = [concept_of(c["path"]) for c in caps_te]
    pred_names = [[phrases[int(addr_te[i, k])] for k in range(K)] for i in range(len(addr_te))]
    # NOTE: test labels used ONLY for post-hoc audit report, never for training/selection
    audit_hit = {
        f"test_recall@{k}": float(np.mean([
            con_te[i] in {pred_names[i][j] for j in range(k)} for i in range(len(con_te))
        ])) for k in (1, 4, 8)
    }

    report = {
        "pipeline": "uck_nat_khyp",
        "subject": sub,
        "K": K,
        "lambdas": lams,
        "use_residual": use_residual,
        "best": best,
        "diag": diag,
        "diversity": div,
        "fuse_lambda0_equals_uck": fuse_ok,
        "fuse_lambda0_max_abs": max_abs,
        "test_audit_recall_POSTHOC_only": audit_hit,
        "history_tail": hist[-5:],
        "conds_dir": str(out / "conds"),
        "note": (
            "λ=0 must match UCK IP bit-wise. Diversity from disjoint addr_k. "
            "If residual kill-switch trips, Δ≡0 (anchor-only). "
            "Test recall in audit is POST-HOC only; checkpoint selected on val_b."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    (out / "conds" / "pred_names_topk.json").write_text(
        json.dumps({"pred_topk": pred_names, "true_POSTHOC": con_te}, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "fuse_lambda0_equals_uck", "fuse_lambda0_max_abs", "diversity",
        "test_audit_recall_POSTHOC_only", "best")}, indent=2))
    print(f"[uck_nat] wrote {out}")


if __name__ == "__main__":
    main()
