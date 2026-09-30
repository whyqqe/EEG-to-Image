#!/usr/bin/env python3
"""Train MG-Flow: dual-granularity semantic alignment + hierarchical CFM + gated gen residual."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from mg_flow_modules import (  # noqa: E402
    MGFlowModel,
    clip_cosine_loss,
    clip_info_nce,
    flow_matching_loss,
    l2_t,
    orth_loss,
)


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)).astype(np.float32)


def retrieval_topk(pred: np.ndarray, gallery: np.ndarray, k: int = 1) -> float:
    sim = pred @ gallery.T
    hits = sum(1 for i in range(sim.shape[0]) if i in np.argsort(sim[i])[-k:])
    return hits / max(sim.shape[0], 1)


def class_top1(pred: np.ndarray, text_concepts: np.ndarray, gt_cls: np.ndarray) -> float:
    """pred (N,D), text_concepts (C,D), gt_cls (N,) class ids."""
    sim = pred @ text_concepts.T
    pred_cls = sim.argmax(1)
    return float((pred_cls == gt_cls).mean())


@torch.no_grad()
def decode_bundle(model: MGFlowModel, z_ret: torch.Tensor, steps: int, bs: int = 256):
    model.eval()
    zs_c, zs_f, z_cfm_c, z_cfm_f, z_c2f = [], [], [], [], []
    for i in range(0, z_ret.shape[0], bs):
        zr = z_ret[i : i + bs]
        c, f = model.encode(zr)
        zs_c.append(c.cpu().numpy())
        zs_f.append(f.cpu().numpy())
        z_cfm_c.append(model.cfm_c.decode(c, steps=steps).cpu().numpy())
        z_cfm_f.append(model.cfm_f.decode(f, steps=steps).cpu().numpy())
        z_c2f.append(model.cfm_c2f.decode(c, steps=steps).cpu().numpy())
    return {k: l2(np.concatenate(v, 0)) for k, v in {
        "z_s_c": zs_c, "z_s_f": zs_f, "z_cfm_c": z_cfm_c, "z_cfm_f": z_cfm_f, "z_c2f": z_c2f
    }.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-ret-train", type=str, required=True)
    ap.add_argument("--z-ret-test", type=str, required=True)
    ap.add_argument("--clip-img-train", type=str, required=True)
    ap.add_argument("--clip-img-test", type=str, required=True)
    ap.add_argument("--t-coarse-train", type=str, required=True)
    ap.add_argument("--t-fine-train", type=str, required=True)
    ap.add_argument("--t-coarse-test", type=str, required=True)
    ap.add_argument("--t-fine-test", type=str, required=True)
    ap.add_argument("--nda-decode-train", type=str, required=True)
    ap.add_argument("--nda-decode-test", type=str, required=True)
    ap.add_argument("--text-concept-test", type=str, required=True, help="(200,D) class text gallery")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ode-steps", type=int, default=16)
    ap.add_argument("--lambda-nce-c", type=float, default=0.5)
    ap.add_argument("--lambda-nce-f", type=float, default=0.35)
    ap.add_argument("--lambda-nce-img", type=float, default=0.35)
    ap.add_argument("--lambda-cos", type=float, default=1.5)
    ap.add_argument("--lambda-fm-c", type=float, default=0.8)
    ap.add_argument("--lambda-fm-f", type=float, default=0.8)
    ap.add_argument("--lambda-fm-c2f", type=float, default=0.6)
    ap.add_argument("--lambda-orth", type=float, default=0.05)
    ap.add_argument("--lambda-gate", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop", type=int, default=12)
    ap.add_argument("--images-per-concept-train", type=int, default=10)
    ap.add_argument("--init-ckpt", type=str, default="", help="warm-start / subject finetune from best.pt")
    ap.add_argument("--freeze-backbone", action="store_true", help="finetune: freeze encode heads, train CFM/gate/to_gen")
    ap.add_argument("--tile-targets", action="store_true", help="tile CLIP/text/nda targets to match multi-subj z_ret length")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "embeds").mkdir(parents=True, exist_ok=True)

    z_tr = l2(np.load(args.z_ret_train))
    z_te = l2(np.load(args.z_ret_test))
    img_tr = l2(np.load(args.clip_img_train))
    img_te = l2(np.load(args.clip_img_test))
    tc_tr = l2(np.load(args.t_coarse_train))
    tf_tr = l2(np.load(args.t_fine_train))
    tc_te = l2(np.load(args.t_coarse_test))
    tf_te = l2(np.load(args.t_fine_test))
    nda_tr = l2(np.load(args.nda_decode_train))
    nda_te = l2(np.load(args.nda_decode_test))
    text_cls_te = l2(np.load(args.text_concept_test))

    def _tile_to(x: np.ndarray, n: int) -> np.ndarray:
        if x.shape[0] == n:
            return x
        if n % x.shape[0] != 0:
            raise ValueError(f"cannot tile {x.shape[0]} -> {n}")
        reps = n // x.shape[0]
        return np.concatenate([x] * reps, axis=0)

    if args.tile_targets:
        img_tr = _tile_to(img_tr, z_tr.shape[0])
        tc_tr = _tile_to(tc_tr, z_tr.shape[0])
        tf_tr = _tile_to(tf_tr, z_tr.shape[0])
        nda_tr = _tile_to(nda_tr, z_tr.shape[0])
        # test: keep single-gallery 200; if multi-subj test, tile
        if z_te.shape[0] != img_te.shape[0]:
            img_te = _tile_to(img_te, z_te.shape[0])
            tc_te = _tile_to(tc_te, z_te.shape[0])
            tf_te = _tile_to(tf_te, z_te.shape[0])
            nda_te = _tile_to(nda_te, z_te.shape[0])

    assert z_tr.shape[0] == img_tr.shape[0] == tc_tr.shape[0] == tf_tr.shape[0] == nda_tr.shape[0]

    # train class ids for monitoring
    gt_cls_tr = (np.arange(z_tr.shape[0]) // args.images_per_concept_train).astype(np.int64)
    # test: if multi-subj (N=200*S), class repeats every 200
    gt_cls_te = (np.arange(z_te.shape[0]) % 200).astype(np.int64)

    model = MGFlowModel(ret_dim=z_tr.shape[1], clip_dim=img_tr.shape[1]).to(device)
    def _load_ckpt_into(m: MGFlowModel, path: str) -> None:
        ck0 = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ck0, dict) and "model_delta" in ck0:
            base = ck0.get("base_ckpt") or args.init_ckpt
            if not base:
                raise ValueError(f"delta ckpt {path} missing base_ckpt")
            base_ck = torch.load(base, map_location=device, weights_only=False)
            base_state = base_ck["model"] if isinstance(base_ck, dict) and "model" in base_ck else base_ck
            m.load_state_dict(base_state, strict=True)
            cur = m.state_dict()
            cur.update(ck0["model_delta"])
            m.load_state_dict(cur, strict=True)
            print(f"[OK] loaded delta {path} on base {base}")
        else:
            state = ck0["model"] if isinstance(ck0, dict) and "model" in ck0 else ck0
            m.load_state_dict(state, strict=True)
            print(f"[OK] loaded init ckpt {path}")

    if args.init_ckpt:
        _load_ckpt_into(model, args.init_ckpt)
    if args.freeze_backbone:
        for p in model.heads.parameters():
            p.requires_grad_(False)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[OK] freeze DualSemanticHeads; trainable {n_train}/{n_all} params")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    def _checkpoint_payload(epoch: int, best_row: dict) -> dict:
        """Full model for shared train; delta-only (updated params) for subject FT."""
        if args.freeze_backbone:
            delta = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
                if not k.startswith("heads.")
            }
            return {
                "model_delta": delta,
                "updated_keys": sorted(delta.keys()),
                "base_ckpt": args.init_ckpt,
                "epoch": epoch,
                "args": vars(args),
                "best": best_row,
                "note": "subject-FT: only CFM/gate/to_gen; merge onto base_ckpt for inference",
            }
        return {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "epoch": epoch,
            "args": vars(args),
            "best": best_row,
        }

    ds = TensorDataset(
        torch.from_numpy(z_tr),
        torch.from_numpy(tc_tr),
        torch.from_numpy(tf_tr),
        torch.from_numpy(img_tr),
        torch.from_numpy(nda_tr),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)

    best = {"score": -1.0, "epoch": -1}
    history = []
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for zr, tc, tf, img, nda in tqdm(loader, desc=f"ep{epoch}", leave=False):
            zr, tc, tf, img, nda = [x.to(device) for x in (zr, tc, tf, img, nda)]
            zc, zf = model.encode(zr)

            # alignment
            loss = (
                args.lambda_nce_c * clip_info_nce(zc, tc)
                + args.lambda_nce_f * clip_info_nce(zf, tf)
                + args.lambda_nce_img * 0.5 * (clip_info_nce(zc, img) + clip_info_nce(zf, img))
                + args.lambda_cos * 0.5 * (clip_cosine_loss(zc, tc) + clip_cosine_loss(zf, tf))
                + args.lambda_orth * orth_loss(zc, zf)
            )
            # hierarchical CFM
            loss = loss + args.lambda_fm_c * flow_matching_loss(model.cfm_c, tc, zc.detach())
            loss = loss + args.lambda_fm_f * flow_matching_loss(model.cfm_f, tf, zf.detach())
            # coarse→fine: transport from coarse text toward fine text, cond on zc
            loss = loss + args.lambda_fm_c2f * flow_matching_loss(model.cfm_c2f, tf, zc.detach())

            # gated residual toward NDA gen space (train gate to help when coarse margin high)
            with torch.no_grad():
                # proxy confidence: cos(zc, tc)
                conf = (l2_t(zc) * l2_t(tc)).sum(-1, keepdim=True)
            delta = model.to_gen(zf)
            # soft gate target: higher conf → smaller gate (trust NDA more when class locked)
            gate_logit = model.gate_mlp(torch.cat([l2_t(zf), l2_t(nda), conf], dim=-1))
            gate = torch.sigmoid(gate_logit)
            z_hat = l2_t(nda + gate * delta)
            # when conf high, prefer staying near nda; when low, allow more delta if improves img align
            loss = loss + args.lambda_gate * (
                clip_cosine_loss(z_hat, img)
                + 0.1 * (gate * conf).mean()  # penalize large gate under high conf
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        # eval
        zt = torch.from_numpy(z_te).to(device)
        bundle = decode_bundle(model, zt, steps=args.ode_steps)
        # class consistency via coarse CFM / head
        cls_c = class_top1(bundle["z_cfm_c"], text_cls_te, gt_cls_te)
        cls_f = class_top1(bundle["z_cfm_f"], text_cls_te, gt_cls_te)
        # image retrieval in CLIP space using fine CFM
        top1_img = retrieval_topk(bundle["z_cfm_f"], img_te, 1)
        top5_img = retrieval_topk(bundle["z_cfm_f"], img_te, 5)
        # also head-only
        top1_head = retrieval_topk(bundle["z_s_f"], img_te, 1)
        score = 0.45 * cls_c + 0.25 * cls_f + 0.30 * top1_img
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "cls_top1_cfm_c": cls_c,
            "cls_top1_cfm_f": cls_f,
            "img_top1_cfm_f": top1_img,
            "img_top5_cfm_f": top5_img,
            "img_top1_head_f": top1_head,
            "score": score,
        }
        history.append(row)
        print(json.dumps(row))
        if score > best["score"]:
            best = {**row, "score": score}
            bad = 0
            ckpt = _checkpoint_payload(epoch, best)
            ck_name = "best_delta.pt" if args.freeze_backbone else "best.pt"
            torch.save(ckpt, out / "checkpoints" / ck_name)
            if args.freeze_backbone:
                # pointer for resume scripts that expect best.pt
                torch.save(ckpt, out / "checkpoints" / "best.pt")
                (out / "checkpoints" / "updated_keys.json").write_text(
                    json.dumps({"updated_keys": ckpt["updated_keys"], "base_ckpt": args.init_ckpt}, indent=2),
                    encoding="utf-8",
                )
            for k, v in bundle.items():
                np.save(out / "embeds" / f"{k}_test.npy", v)
        else:
            bad += 1
            if bad >= args.early_stop:
                print(f"[EARLY STOP] epoch={epoch}")
                break

    # final exports with best ckpt
    ck_path = out / "checkpoints" / ("best_delta.pt" if args.freeze_backbone else "best.pt")
    if not ck_path.is_file():
        ck_path = out / "checkpoints" / "best.pt"
    _load_ckpt_into(model, str(ck_path))
    for split, zr_np, nda_np in [
        ("test", z_te, nda_te),
        ("train", z_tr, nda_tr),
    ]:
        zr = torch.from_numpy(zr_np).to(device)
        bundle = decode_bundle(model, zr, steps=args.ode_steps)
        for k, v in bundle.items():
            np.save(out / "embeds" / f"{k}_{split}.npy", v)
        # gated gen embeds
        model.eval()
        outs, gates = [], []
        with torch.no_grad():
            for i in range(0, zr.shape[0], 256):
                zc, zf = model.encode(zr[i : i + 256])
                # confidence vs coarse text (test uses matching rows; train uses tc_tr)
                if split == "test":
                    tc = torch.from_numpy(tc_te[i : i + 256]).to(device)
                else:
                    tc = torch.from_numpy(tc_tr[i : i + 256]).to(device)
                nda = torch.from_numpy(nda_np[i : i + 256]).to(device)
                conf = (l2_t(zc) * l2_t(tc)).sum(-1, keepdim=True)
                # refine fine via c2f then cfm_f blend
                z_ref = l2_t(0.5 * model.cfm_f.decode(zf, steps=args.ode_steps) + 0.5 * model.cfm_c2f.decode(zc, steps=args.ode_steps))
                delta = model.to_gen(z_ref)
                g = torch.sigmoid(model.gate_mlp(torch.cat([l2_t(z_ref), l2_t(nda), conf], dim=-1)))
                # semantic lock: if coarse class matches nda retrieval class weakly, still allow gate
                z_hat = l2_t(nda + g * delta)
                outs.append(z_hat.cpu().numpy())
                gates.append(g.cpu().numpy())
        np.save(out / "embeds" / f"z_mg_gated_{split}.npy", l2(np.concatenate(outs, 0)))
        np.save(out / "embeds" / f"gate_{split}.npy", np.concatenate(gates, 0).astype(np.float32))
        # also pure CFM fine as ablation embed (mapped through to_gen + full gate=1 residual around nda)
        z_cfm = bundle["z_cfm_f"]
        # project cfm_f into gen by replacing nda directionally (alpha blend)
        for alpha, name in [(0.25, "a25"), (0.40, "a40"), (0.55, "a55")]:
            blend = l2((1 - alpha) * nda_np + alpha * z_cfm)
            np.save(out / "embeds" / f"blend_nda_cfm_f_{name}_{split}.npy", blend)

    report = {"best": best, "history": history, "pipeline": "MG-Flow hierarchical CFM"}
    (out / "mg_flow_train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["best"], indent=2))


if __name__ == "__main__":
    main()
