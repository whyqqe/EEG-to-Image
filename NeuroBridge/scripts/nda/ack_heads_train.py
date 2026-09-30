#!/usr/bin/env python3
"""ACK-DT heads for sub-08 (conservative).

Keeps the dual-tower contract:
  Semantic frame (concept): align z to G_img + G_text; predict class name.
  Instance frame (structure): left to existing UCK depth + VAE/LL (not retrained here).

What this file trains (on frozen shared_r):
  * neural book μ + address CE          (innovation auxiliary)
  * class head over 1654 train concepts (symbol coordinate)
  * light dual-view q = MLP(z) on G_img + G_text (semantic tower readout)
  * cycle consistency among α / p_text / p_cls

What this file does NOT do:
  * never regress encode_image(test photo)
  * never use oracle test class names for training or selection
  * never overwrite UCK structure heads

At export it builds LEGAL predicted prompts via 200-way retrieval against the
standard test CLIP-image bank (no labels used), plus gated / neural-NN /
empty / oracle(copy) prompt files for the generation matrix.
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
from nat_train import concept_means, loo_top1  # noqa: E402
from uck_train import gallery_nce, memory, rowcos  # noqa: E402
from build_hcma_prompts import scene_for  # noqa: E402


SCENE_FALLBACK = "in a clean studio setting"


def concept_of(path: str) -> str:
    return Path(path).parent.name.split("_", 1)[1].replace("_", " ")


def hcma_prompt(name: str) -> str:
    return (f"a photo of a {name}, clearly showing its shape, color, and "
            f"distinctive parts, {scene_for(name)}, natural lighting")


class ACKHeads(nn.Module):
    def __init__(self, z_dim: int = 1024, hidden: int = 1024, n_cls: int = 1654):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, z_dim),
        )
        self.cls = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"q": self.q_net(z), "logits": self.cls(z)}


def address_weights(z: torch.Tensor, mu: torch.Tensor, tau: float, k: int):
    sim = l2t(z) @ l2t(mu).T
    topv, topi = sim.topk(min(k, sim.shape[1]), dim=-1)
    w = torch.softmax(topv / tau, dim=-1)
    # scatter into full vocabulary for KL with cls
    full = z.new_zeros(z.shape[0], mu.shape[0])
    full.scatter_(1, topi, w)
    return full, topi, w


def retrieval_bank(q: np.ndarray, bank: np.ndarray, labels: list[str]) -> dict:
    qz, bz = l2n(q), l2n(bank)
    sim = qz @ bz.T
    order = np.argsort(-sim, axis=1)
    top1 = order[:, 0]
    margin = sim[np.arange(len(q)), top1] - sim[np.arange(len(q)), order[:, 1]]
    pred = [labels[int(i)] for i in top1]
    true = labels
    return {
        "top1": float(np.mean([a == b for a, b in zip(pred, true)])),
        "chance": 1.0 / max(len(set(labels)), 1),
        "mean_margin": float(margin.mean()),
        "pred_names": pred,
        "margins": margin.astype(np.float32),
        "top1_idx": top1.astype(np.int64),
    }


def sinkhorn(sim: np.ndarray, iters: int = 50, tau: float = 0.07) -> np.ndarray:
    """Balanced assignment over square similarity (rows=queries, cols=gallery)."""
    log_k = sim.astype(np.float64) / max(tau, 1e-6)
    log_k = log_k - log_k.max(axis=1, keepdims=True)
    k = np.exp(log_k)
    for _ in range(iters):
        k = k / np.clip(k.sum(axis=1, keepdims=True), 1e-12, None)
        k = k / np.clip(k.sum(axis=0, keepdims=True), 1e-12, None)
    return k.argmax(axis=1).astype(np.int64)


def write_prompts(names: list[str], path: Path, use_generic: bool = False,
                  empty: bool = False) -> None:
    if empty:
        prompts = [""] * len(names)
    elif use_generic:
        prompts = [
            "a photo of an object, clearly showing its shape, color, and "
            "distinctive parts, natural lighting"
        ] * len(names)
    else:
        prompts = [hcma_prompt(n) for n in names]
    path.write_text(json.dumps(prompts, indent=1), encoding="utf-8")


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
    ap.add_argument("--uck-conds", type=str, default="")
    ap.add_argument("--oracle-prompts", type=str,
                    default=str(NB_ROOT / "outputs/g2f/prompts/prompts_oracle.json"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--mem-k", type=int, default=16)
    ap.add_argument("--w-img", type=float, default=1.0)
    ap.add_argument("--w-txt", type=float, default=0.5)
    ap.add_argument("--w-cls", type=float, default=1.0)
    ap.add_argument("--w-addr", type=float, default=1.0)
    ap.add_argument("--w-cyc", type=float, default=0.2)
    ap.add_argument("--gate-margin", type=float, default=0.02)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    (out / "conds").mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "proto").mkdir(parents=True, exist_ok=True)
    sid = f"{args.test_subject:02d}"
    sub = f"sub-{sid}"
    uck_conds = Path(args.uck_conds) if args.uck_conds else NB_ROOT / f"outputs/uck/{sub}/full/conds"

    ztr = l2n(np.load(Path(args.z_root) / sub / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / sub / "shared_r_test.npy").astype(np.float32))
    g_text, cid_np, phrases = build_concept_bank(
        Path(args.clip_text_dir), Path(args.captions_dir) / "captions_train.jsonl")
    caps_te = [json.loads(l) for l in (Path(args.captions_dir) / "captions_test.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if len(cid_np) != len(ztr) or len(caps_te) != len(zte):
        raise SystemExit(f"[FATAL] row mismatch z {len(ztr)}/{len(zte)} "
                         f"cid/cap {len(cid_np)}/{len(caps_te)}")
    con_te = [concept_of(c["path"]) for c in caps_te]
    inter = {p.strip().lower() for p in phrases} & {c.strip().lower() for c in con_te}
    if inter:
        raise SystemExit(f"[FATAL] {len(inter)} test concepts in train gallery")
    print(f"[audit] train {len(phrases)} test {len(set(con_te))} inter 0")

    g_img = l2n(np.load(Path(args.gallery_cache) / "g_img_concept.npy").astype(np.float32))
    if g_img.shape[0] != len(phrases):
        raise SystemExit(f"[FATAL] G_img {g_img.shape} vs phrases {len(phrases)}")
    clip_te = l2n(np.load(Path(args.clip_img_dir) / "clip_img1024_test.npy").astype(np.float32))
    if len(clip_te) != len(zte):
        raise SystemExit(f"[FATAL] clip test {len(clip_te)} vs zte {len(zte)}")

    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", len(ztr))
    val_i = LF.rows_for(split, "val_b", len(ztr))
    mu_all, cnt_all = concept_means(ztr, cid_np, len(phrases), np.arange(len(ztr)))
    mu_fit, cnt_fit = concept_means(ztr, cid_np, len(phrases), fit_i)
    if int((cnt_all == 0).sum()):
        raise SystemExit("[FATAL] empty train prototypes")
    np.save(out / "proto" / "mu_all.npy", mu_all)
    np.save(out / "proto" / "mu_fit.npy", mu_fit)
    np.save(out / "proto" / "cnt_fit.npy", cnt_fit.astype(np.int64))
    loo = loo_top1(ztr, cid_np, len(phrases))
    loo_val = loo_top1(ztr, cid_np, len(phrases), rows=val_i)
    print(f"[ack] LOO1654={loo['top1']:.4f} val_b LOO={loo_val['top1']:.4f}")

    G_img = torch.from_numpy(g_img).to(dev)
    G_text = torch.from_numpy(l2n(g_text)).to(dev)
    mu = torch.from_numpy(mu_all).to(dev)
    model = ACKHeads(z_dim=ztr.shape[1], n_cls=len(phrases)).to(dev)
    ckpt_p = out / "best.pth"

    if args.resume and (out / "prompts" / "prompts_pred.json").is_file() and (out / "report.json").is_file():
        print("[ack] resume: exports exist, skip train")
        if ckpt_p.is_file():
            model.load_state_dict(torch.load(ckpt_p, map_location=dev, weights_only=False)["state_dict"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best = -1e9
        history = []
        bs = args.batch_size
        for ep in range(1, args.epochs + 1):
            model.train()
            order = np.random.permutation(fit_i)
            run = 0.0
            nstep = 0
            for s in range(0, len(order) - bs + 1, bs):
                ix = order[s:s + bs]
                zb = torch.from_numpy(ztr[ix]).to(dev)
                cid = torch.from_numpy(cid_np[ix].astype(np.int64)).to(dev)
                o = model(zb)
                loss = zb.new_zeros(())
                loss = loss + args.w_img * gallery_nce(o["q"], G_img, cid, args.tau)
                loss = loss + args.w_txt * gallery_nce(o["q"], G_text, cid, args.tau)
                loss = loss + args.w_cls * F.cross_entropy(o["logits"], cid)
                # neural address CE (no MLP): soft target via top-k mass optional;
                # hard CE on z·μ is the innovation auxiliary
                loss = loss + args.w_addr * F.cross_entropy(
                    (l2t(zb) @ l2t(mu).T) / args.tau, cid)
                alpha, _, _ = address_weights(zb, mu, args.tau, args.mem_k)
                p_txt = torch.softmax((l2t(o["q"]) @ l2t(G_text).T) / args.tau, dim=-1)
                p_cls = torch.softmax(o["logits"], dim=-1)
                # cycle on the support of alpha (sparse) vs text/cls
                loss = loss + args.w_cyc * (
                    F.kl_div((alpha.clamp_min(1e-8)).log(), p_txt.detach(), reduction="batchmean")
                    + F.kl_div((p_cls.clamp_min(1e-8)).log(), alpha.detach(), reduction="batchmean")
                )
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                run += float(loss.item())
                nstep += 1

            model.eval()
            with torch.no_grad():
                zv = torch.from_numpy(ztr[val_i]).to(dev)
                cv = torch.from_numpy(cid_np[val_i].astype(np.int64)).to(dev)
                ov = model(zv)
                t1_img = float(((l2t(ov["q"]) @ l2t(G_img).T).argmax(-1) == cv).float().mean())
                t1_txt = float(((l2t(ov["q"]) @ l2t(G_text).T).argmax(-1) == cv).float().mean())
                t1_cls = float((ov["logits"].argmax(-1) == cv).float().mean())
                t1_neu = float(((l2t(zv) @ l2t(mu).T).argmax(-1) == cv).float().mean())
            score = t1_neu + t1_cls + 0.5 * (t1_img + t1_txt)
            row = {"epoch": ep, "loss": run / max(nstep, 1), "val_img": t1_img,
                   "val_txt": t1_txt, "val_cls": t1_cls, "val_neu": t1_neu, "score": score}
            history.append(row)
            print(f"[ack ep {ep}] loss={row['loss']:.4f} neu={t1_neu:.3f} cls={t1_cls:.3f} "
                  f"img={t1_img:.3f} txt={t1_txt:.3f}")
            if score > best:
                best = score
                torch.save({"epoch": ep, "state_dict": model.state_dict(),
                            "metrics": row}, ckpt_p)
        model.load_state_dict(torch.load(ckpt_p, map_location=dev, weights_only=False)["state_dict"])
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # ============================================================ export
    model.eval()
    with torch.no_grad():
        zt = torch.from_numpy(zte).to(dev)
        ot = model(zt)
        q = ot["q"]
        ip = memory(q, G_img, args.tau, args.mem_k)
        np.save(out / "conds" / "ip_ack_test.npy", ip.cpu().numpy().astype(np.float32))
        np.save(out / "conds" / "ip_q_test.npy", l2t(q).cpu().numpy().astype(np.float32))
        np.save(out / "conds" / "cls_logits_test.npy", ot["logits"].cpu().numpy().astype(np.float32))

    # Prefer UCK mem IP for system rows if present (proven); keep ACK IP as ablation.
    if (uck_conds / "ip_mem_test.npy").is_file():
        uck_ip = np.load(uck_conds / "ip_mem_test.npy").astype(np.float32)
        np.save(out / "conds" / "ip_uck_test.npy", uck_ip)

    # 200-way retrieval against standard test CLIP-image bank (LEGAL naming)
    ret_z = retrieval_bank(zte, clip_te, con_te)
    ret_q = retrieval_bank(np.load(out / "conds" / "ip_q_test.npy"), clip_te, con_te)
    ret_ip = retrieval_bank(np.load(out / "conds" / "ip_ack_test.npy"), clip_te, con_te)

    # Sinkhorn balanced assignment (transductive, report separately)
    sim = l2n(zte) @ l2n(clip_te).T
    sink_idx = sinkhorn(sim, iters=60, tau=args.tau)
    sink_names = [con_te[int(i)] for i in sink_idx]
    sink_top1 = float(np.mean([a == b for a, b in zip(sink_names, con_te)]))

    # Neural-book nearest TRAIN name (cannot name test class; diagnostic)
    with torch.no_grad():
        topi = (torch.from_numpy(zte).to(dev) @ l2t(mu).T).argmax(-1).cpu().numpy()
    neural_names = [phrases[int(i)] for i in topi]

    # Gated predicted names: low margin → generic object prompt
    pred_names = ret_z["pred_names"]
    margins = ret_z["margins"]
    gated_names = [
        (n if float(m) >= args.gate_margin else "object")
        for n, m in zip(pred_names, margins)
    ]
    n_gated = int(sum(1 for n in gated_names if n == "object"))

    write_prompts(pred_names, out / "prompts" / "prompts_pred.json")
    write_prompts(gated_names, out / "prompts" / "prompts_pred_gate.json")
    write_prompts(sink_names, out / "prompts" / "prompts_sinkhorn.json")
    write_prompts(neural_names, out / "prompts" / "prompts_neural_nn.json")
    write_prompts(pred_names, out / "prompts" / "prompts_empty.json", empty=True)
    write_prompts(pred_names, out / "prompts" / "prompts_deploy.json", use_generic=True)

    # oracle copy (UPPER BOUND ONLY — labeled leak in report)
    oracle_src = Path(args.oracle_prompts)
    if oracle_src.is_file():
        (out / "prompts" / "prompts_oracle.json").write_text(
            oracle_src.read_text(encoding="utf-8"), encoding="utf-8")

    # also store predicted indices for audit
    np.save(out / "conds" / "pred_test_idx.npy", ret_z["top1_idx"])
    np.save(out / "conds" / "sinkhorn_idx.npy", sink_idx)
    (out / "conds" / "pred_names.json").write_text(
        json.dumps({"pred": pred_names, "true": con_te, "gated": gated_names,
                    "sinkhorn": sink_names, "neural_nn": neural_names}, indent=1),
        encoding="utf-8")

    centre = float((l2n(clip_te.mean(0, keepdims=True)) * clip_te).sum(1).mean())
    ip_ack = np.load(out / "conds" / "ip_ack_test.npy")
    rep = {
        "pipeline": "ack_dt_heads",
        "subject": sub,
        "semantic_align": ["G_img concept CLIP-image", "G_text concept CLIP-text"],
        "structure_align": "reused UCK depth + VAE/LL (not retrained here)",
        "book": len(phrases),
        "loo_train_1654": loo,
        "val_b_loo_1654": loo_val,
        "test200_top1_z": ret_z["top1"],
        "test200_top1_q": ret_q["top1"],
        "test200_top1_ip": ret_ip["top1"],
        "test200_top1_sinkhorn": sink_top1,
        "test200_chance": ret_z["chance"],
        "mean_margin_z": ret_z["mean_margin"],
        "gate_margin": args.gate_margin,
        "n_gated_to_object": n_gated,
        "ip_ack_vs_true": float((l2n(ip_ack) * clip_te).sum(1).mean()),
        "centreline": centre,
        "ip_ack_rowcos": rowcos(ip_ack),
        "note": ("Predicted prompts use unlabeled 200-way retrieval on the standard "
                 "test CLIP-image bank. Oracle prompts are copied only as a leaky upper bound."),
    }
    (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "loo_train_1654"}, indent=2))
    print("[ack] prompts written:", sorted(p.name for p in (out / "prompts").glob("*.json")))


if __name__ == "__main__":
    main()
