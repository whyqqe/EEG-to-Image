#!/usr/bin/env python3
"""Hybrid IP export for Address-then-Translate + UCK gallery refine.

No training. Builds test-side IP conditions from frozen z / μ / G_img / UCK q:

  nat     : IP = softmax(z·μ)_topk @ G_img                 (pure translate)
  snap    : IP = mem(nat, G_img)                           (translate then snap)
  short   : neural top-m shortlist; mem(UCK q, G_img[m])   (address + CLIP refine)
  gate    : neural α as soft prior, reweighted by UCK q·G  (α ⊙ softmax(q·G))
  uck     : copy of UCK mem                                (control)
  blend   : l2( β·nat + (1-β)·uck_mem )                    (sanity mix)

Hard constraints:
  * train-concept book / gallery only (1654), disjoint from test 200
  * never encode_image(test photo)
  * never fuse toward CLIP-text
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocf_train import build_concept_bank, l2n, l2t  # noqa: E402
from uck_train import memory, rowcos  # noqa: E402
from nat_train import concept_means, loo_top1, translate  # noqa: E402


def vs_bank(ip: np.ndarray, bank: np.ndarray) -> dict:
    a, b = l2n(ip), l2n(bank)
    centre = float((l2n(b.mean(0, keepdims=True)) * b).sum(1).mean())
    true = float((a * b).sum(1).mean())
    return {"vs_true": true, "centreline": centre, "vs_cl": true - centre, "rowcos": rowcos(ip)}


def gated_memory(z: torch.Tensor, mu: torch.Tensor, q: torch.Tensor,
                 g_img: torch.Tensor, tau_n: float, tau_c: float, k: int):
    """Neural top-k prior × CLIP affinity on the same shortlist."""
    sim_n = l2t(z) @ l2t(mu).T
    kk = min(k, sim_n.shape[1])
    topv, topi = sim_n.topk(kk, dim=-1)
    prior = torch.softmax(topv / tau_n, dim=-1)
    # CLIP affinity of UCK q against shortlisted concept images
    g_sel = l2t(g_img)[topi]  # B,k,d
    qn = l2t(q).unsqueeze(1)  # B,1,d
    sim_c = (qn * g_sel).sum(-1)  # B,k
    # combine in log space
    w = torch.softmax(torch.log(prior.clamp_min(1e-8)) + sim_c / tau_c, dim=-1)
    ip = l2t((w.unsqueeze(-1) * g_sel).sum(1))
    return ip, topi, w


def shortlist_memory(z: torch.Tensor, mu: torch.Tensor, q: torch.Tensor,
                     g_img: torch.Tensor, tau: float, m: int):
    """Neural shortlist of m concepts; soft-retrieve with UCK q inside it."""
    sim_n = l2t(z) @ l2t(mu).T
    topi = sim_n.topk(min(m, sim_n.shape[1]), dim=-1).indices
    # per-row memory against the shortlist bank
    outs = []
    for i in range(z.shape[0]):
        bank = g_img[topi[i]]
        outs.append(memory(q[i:i + 1], bank, tau, k=min(16, bank.shape[0])))
    return torch.cat(outs, 0), topi


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
    ap.add_argument("--nat-proto", type=str, default="")
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--mem-k", type=int, default=16)
    ap.add_argument("--short-m", type=int, default=64)
    ap.add_argument("--blend-beta", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "conds").mkdir(parents=True, exist_ok=True)
    sid = f"{args.test_subject:02d}"
    sub = f"sub-{sid}"
    uck_conds = Path(args.uck_conds) if args.uck_conds else NB_ROOT / f"outputs/uck/{sub}/full/conds"
    nat_proto = Path(args.nat_proto) if args.nat_proto else NB_ROOT / f"outputs/nat/{sub}/full/proto"

    zte = l2n(np.load(Path(args.z_root) / sub / "shared_r_test.npy").astype(np.float32))
    ztr = l2n(np.load(Path(args.z_root) / sub / "shared_r_train.npy").astype(np.float32))
    g_text, cid_np, phrases = build_concept_bank(
        Path(args.clip_text_dir), Path(args.captions_dir) / "captions_train.jsonl")
    g_img = l2n(np.load(Path(args.gallery_cache) / "g_img_concept.npy").astype(np.float32))
    if g_img.shape[0] != len(phrases):
        raise SystemExit(f"[FATAL] G_img {g_img.shape} vs phrases {len(phrases)}")

    caps_te = [json.loads(l) for l in (Path(args.captions_dir) / "captions_test.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if len(caps_te) != len(zte):
        raise SystemExit(f"[FATAL] test rows {len(zte)} vs captions {len(caps_te)}")

    if (nat_proto / "mu_all.npy").is_file():
        mu_np = l2n(np.load(nat_proto / "mu_all.npy").astype(np.float32))
    else:
        mu_np, _ = concept_means(ztr, cid_np, len(phrases), np.arange(len(ztr)))
    loo = loo_top1(ztr, cid_np, len(phrases))

    q_uck = np.load(uck_conds / "ip_q_test.npy").astype(np.float32)
    mem_uck = np.load(uck_conds / "ip_mem_test.npy").astype(np.float32)
    if len(q_uck) != len(zte) or len(mem_uck) != len(zte):
        raise SystemExit("[FATAL] UCK cond rows mismatch test z")

    clip_te = l2n(np.load(Path(args.clip_img_dir) / "clip_img1024_test.npy").astype(np.float32))
    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    zt = torch.from_numpy(zte).to(dev)
    mu = torch.from_numpy(mu_np).to(dev)
    G = torch.from_numpy(g_img).to(dev)
    qt = torch.from_numpy(l2n(q_uck)).to(dev)

    with torch.no_grad():
        ip_nat, _, topi_n, _ = translate(zt, mu, G, args.tau, args.mem_k)
        ip_snap = memory(ip_nat, G, args.tau, args.mem_k)
        ip_short, topi_s = shortlist_memory(zt, mu, qt, G, args.tau, args.short_m)
        ip_gate, topi_g, _ = gated_memory(zt, mu, qt, G, args.tau, args.tau, args.mem_k)

    ip_nat_np = ip_nat.cpu().numpy().astype(np.float32)
    ip_snap_np = ip_snap.cpu().numpy().astype(np.float32)
    ip_short_np = ip_short.cpu().numpy().astype(np.float32)
    ip_gate_np = ip_gate.cpu().numpy().astype(np.float32)
    ip_blend_np = l2n(args.blend_beta * ip_nat_np + (1.0 - args.blend_beta) * mem_uck).astype(np.float32)

    exports = {
        "ip_nat_test.npy": ip_nat_np,
        "ip_snap_test.npy": ip_snap_np,
        "ip_short_test.npy": ip_short_np,
        "ip_gate_test.npy": ip_gate_np,
        "ip_blend_test.npy": ip_blend_np,
        "ip_uck_test.npy": mem_uck.astype(np.float32),
    }
    for name, arr in exports.items():
        np.save(out / "conds" / name, arr)

    # hard top-1 neural address → that concept's G_img (diagnostic)
    hard = g_img[topi_n[:, 0].cpu().numpy()]
    np.save(out / "conds" / "ip_hard_test.npy", hard.astype(np.float32))
    exports["ip_hard_test.npy"] = hard.astype(np.float32)

    rep = {
        "pipeline": "hybrid_export",
        "subject": sub,
        "book": len(phrases),
        "loo_train_1654": loo,
        "short_m": args.short_m,
        "mem_k": args.mem_k,
        "blend_beta": args.blend_beta,
        "rows": {k: vs_bank(v, clip_te) for k, v in exports.items()},
        "note": "Hybrid IPs: neural address + CLIP dictionary refine; no test encode_image",
    }
    (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
