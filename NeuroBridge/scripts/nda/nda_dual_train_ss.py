#!/usr/bin/env python3
"""NDA dual-stream training on SharedSpecific (MindCross/MindBridge) backbone.

Semantic: CLIP-Image (RN50/SSP) + CLIP-Text
Perception: NVOL-HCF + DINOv2
Decode: ViT-H
During dual train: freeze shared wide backbone; keep subject adapter/embedder + heads trainable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nmb"))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from decode_aligner_modules import ClipInfoNCE, DifferentiableSoftMemory, ProbeDecoder, l2norm  # noqa: E402
from nda_dual_train import DualDataset, build_hcf, dla_weights, DEFAULT_CHANNELS  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402
from module.util import retrieve_all  # noqa: E402
from ss_modules import SharedSpecificEncoder, diff_loss  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    ap.add_argument("--ss-checkpoint", type=str, required=True)
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--clip-layers-dir", type=str, required=True)
    ap.add_argument("--nvol-json", type=str, default="")
    ap.add_argument("--layers", type=str, default="")
    ap.add_argument("--dino-train-npy", type=str, required=True)
    ap.add_argument("--dino-test-npy", type=str, required=True)
    ap.add_argument("--clip-train-npy", type=str, required=True)
    ap.add_argument("--clip-test-npy", type=str, required=True)
    ap.add_argument("--text-train-npy", type=str, required=True)
    ap.add_argument("--text-test-npy", type=str, required=True)
    ap.add_argument("--probe-supervision", type=str, default="")
    ap.add_argument("--num-epochs", type=int, default=40)
    ap.add_argument("--phase1-epochs", type=int, default=10)
    ap.add_argument("--phase2-epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-4)
    ap.add_argument("--lambda-diff", type=float, default=0.05)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda-rn50", type=float, default=0.8)
    ap.add_argument("--lambda-txt", type=float, default=0.2)
    ap.add_argument("--lambda-vith", type=float, default=0.35)
    ap.add_argument("--lambda-hcf", type=float, default=1.0)
    ap.add_argument("--lambda-dino", type=float, default=0.5)
    ap.add_argument("--lambda-probe", type=float, default=0.5)
    ap.add_argument("--lambda-mem", type=float, default=0.25)
    ap.add_argument("--soft-k", type=int, default=5)
    ap.add_argument("--soft-tau", type=float, default=0.07)
    args = ap.parse_args()

    root = Path(args.nb_root)
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.layers:
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]
    elif args.nvol_json and Path(args.nvol_json).is_file():
        layer_ids = json.loads(Path(args.nvol_json).read_text())["top_k_layers"]
    else:
        layer_ids = [8, 10, 14]
    print(f"[INFO] HCF layers={layer_ids}")

    hcf_train = out / "hcf_train.npy"
    hcf_test = out / "hcf_test.npy"
    layers_dir = Path(args.clip_layers_dir)
    if not hcf_train.is_file():
        build_hcf(layers_dir, "train", layer_ids, hcf_train)
    if not hcf_test.is_file():
        build_hcf(layers_dir, "test", layer_ids, hcf_test)

    eeg_dir = str(root / "data/things_eeg/preprocessed_eeg")
    rn50_dir = str(root / "data/things_eeg/image_feature/RN50")
    vith_dir = str(root / "data/things_eeg/image_feature/ViT-H-14")

    train_ds = DualDataset(
        str(hcf_train), args.dino_train_npy, vith_dir, args.text_train_npy,
        [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
        rn50_dir, "", False, [], True, False, None, True, False, False, False,
    )
    test_ds = DualDataset(
        str(hcf_test), args.dino_test_npy, vith_dir, args.text_test_npy,
        [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
        rn50_dir, "", False, [], True, False, None, False, False, False, False,
    )

    ckpt_path = Path(args.ss_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    img_dim = int(ckpt.get("img_dim", train_ds.image_features.shape[-1]))
    feature_dim = int(ckpt.get("feature_dim", 512))
    eeg_len = int(ckpt.get("eeg_sample_points", train_ds.num_sample_points))
    channels_num = int(ckpt.get("channels_num", train_ds.channels_num))
    subjects = [int(s) for s in ckpt.get("subjects", [args.subject])]
    if args.subject not in subjects:
        subjects.append(args.subject)

    model = SharedSpecificEncoder(
        subject_ids=subjects,
        feature_dim=img_dim,
        eeg_sample_points=eeg_len,
        channels_num=channels_num,
        n_extra_blocks=int(ckpt.get("n_extra_blocks", 1)),
        use_adapter=True,
    ).to(device)
    eeg_projector = ProjectorLinear(img_dim, feature_dim).to(device)
    img_projector = ProjectorLinear(img_dim, feature_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
    if ckpt.get("img_projector_state_dict"):
        img_projector.load_state_dict(ckpt["img_projector_state_dict"])
    img_projector.eval()
    for p in img_projector.parameters():
        p.requires_grad = False

    # MindBridge calibrate mode: freeze shared, train subject path
    model.train_only_subject(args.subject)
    for p in eeg_projector.parameters():
        p.requires_grad = True

    hcf_dim = int(np.load(hcf_train).shape[1])
    dino_dim = int(np.load(args.dino_train_npy).shape[1])
    text_dim = int(np.load(args.text_train_npy).shape[1])
    vith_dim = 1024

    perc_head = nn.Sequential(nn.Linear(img_dim, 1024), nn.GELU(), nn.Linear(1024, hcf_dim)).to(device)
    dino_head = nn.Linear(img_dim, dino_dim).to(device)
    text_adapter = nn.Sequential(nn.Linear(text_dim, 512), nn.GELU(), nn.Linear(512, 512)).to(device)
    vith_head = nn.Sequential(nn.Linear(img_dim, 1024), nn.GELU(), nn.Linear(1024, vith_dim)).to(device)
    fuse_gate = nn.Sequential(nn.Linear(vith_dim * 2 + 512, 256), nn.GELU(), nn.Linear(256, 3)).to(device)
    probe = ProbeDecoder(dim=vith_dim).to(device)
    nce = ClipInfoNCE(0.07).to(device)

    clip_train = torch.tensor(np.load(args.clip_train_npy), device=device, dtype=torch.float32)
    with torch.no_grad():
        keys = []
        for batch in DataLoader(train_ds, batch_size=512, shuffle=False):
            eeg = batch[0].to(device)
            sid = batch[6].to(device)
            keys.append(eeg_projector(model(eeg, sid)))
        gallery_keys = torch.cat(keys, dim=0)
    soft_mem = DifferentiableSoftMemory(clip_train, gallery_keys, args.soft_k, args.soft_tau).to(device)

    probe_sup = None
    if args.probe_supervision and Path(args.probe_supervision).is_file():
        data = np.load(args.probe_supervision)
        probe_sup = {
            "eeg": torch.tensor(data["eeg_embed"], device=device),
            "anchor": torch.tensor(data["anchor_embed"], device=device),
            "clip_gen": torch.tensor(data["clip_gen"], device=device),
        }

    head_params = (
        list(perc_head.parameters())
        + list(dino_head.parameters())
        + list(vith_head.parameters())
        + list(text_adapter.parameters())
        + list(fuse_gate.parameters())
        + list(probe.parameters())
    )
    adapter_params = [p for p in model.parameters() if p.requires_grad] + list(eeg_projector.parameters())
    opt = optim.AdamW(
        [
            {"params": head_params, "lr": args.lr},
            {"params": adapter_params, "lr": args.adapter_lr},
        ],
        weight_decay=1e-4,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)
    best_score, best_epoch, history = -1e9, 0, []

    for epoch in range(1, args.num_epochs + 1):
        wp, ws, wd = dla_weights(epoch, args.phase1_epochs, args.phase2_epochs)
        for m in (perc_head, dino_head, vith_head, text_adapter, fuse_gate, probe, eeg_projector):
            m.train()
        model.train()
        model.freeze_shared()

        ep_loss = 0.0
        for batch in tqdm(train_loader, desc=f"ep{epoch}"):
            eeg, rn50, vith, hcf, dino, txt, sid, *_ = batch
            eeg, rn50, vith = eeg.to(device), rn50.to(device), vith.to(device)
            hcf, dino, txt, sid = hcf.to(device), dino.to(device), txt.to(device), sid.to(device)

            raw, s, r = model(eeg, sid, return_parts=True)
            z_sem = eeg_projector(raw)
            z_p = perc_head(raw)
            z_dino = dino_head(raw)
            z_v = vith_head(raw)
            z_txt = text_adapter(txt)

            loss_hcf = (1 - (l2norm(z_p) * l2norm(hcf)).sum(-1)).mean() + nce(z_p, hcf)
            loss_dino = (1 - (l2norm(z_dino) * l2norm(dino)).sum(-1)).mean()
            rn50_512 = img_projector(rn50)
            loss_img = (1 - (l2norm(z_sem) * l2norm(rn50_512)).sum(-1)).mean() + nce(z_sem, rn50_512)
            loss_txt = (1 - (l2norm(z_sem) * l2norm(z_txt)).sum(-1)).mean() + nce(z_sem, z_txt)
            loss_sem = args.lambda_rn50 * loss_img + args.lambda_txt * loss_txt
            loss_vith = (1 - (l2norm(z_v) * l2norm(vith)).sum(-1)).mean() + 0.5 * nce(z_v, vith)

            anchor = soft_mem(z_sem)
            z_p_v = z_p[:, :vith_dim] if z_p.shape[-1] >= vith_dim else F.pad(z_p, (0, vith_dim - z_p.shape[-1]))
            g = torch.softmax(
                fuse_gate(torch.cat([l2norm(z_p_v), l2norm(z_v), l2norm(z_sem)], dim=-1)), dim=-1
            )
            loss_mem = (1 - (l2norm(anchor) * l2norm(vith)).sum(-1)).mean()
            probe_pred = l2norm(probe(l2norm(z_v), anchor))
            loss_probe = (1 - (probe_pred * l2norm(vith)).sum(-1)).mean()
            if probe_sup is not None:
                ps = l2norm(probe(probe_sup["eeg"], probe_sup["anchor"]))
                loss_probe = 0.5 * loss_probe + 0.5 * (1 - (ps * probe_sup["clip_gen"]).sum(-1)).mean()

            loss = (
                wp * (args.lambda_hcf * loss_hcf + args.lambda_dino * loss_dino)
                + ws * (loss_sem + args.lambda_vith * loss_vith)
                + wd * (args.lambda_probe * loss_probe + args.lambda_mem * loss_mem)
                + 0.05 * args.lambda_rn50 * loss_img
                + args.lambda_diff * diff_loss(s, r)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params + adapter_params, 1.0)
            opt.step()
            ep_loss += float(loss.item())

        # Eval
        for m in (perc_head, vith_head, text_adapter, fuse_gate, probe, eeg_projector, model):
            m.eval()
        with torch.no_grad():
            sem_list, vith_list, gt_v, gt_rn, fuse_list, txt_cos_list = [], [], [], [], [], []
            for batch in test_loader:
                eeg, rn50, vith, hcf, dino, txt, sid, *_ = batch
                eeg, txt, sid = eeg.to(device), txt.to(device), sid.to(device)
                raw = model(eeg, sid)
                z_sem = eeg_projector(raw)
                z_v = vith_head(raw)
                z_p = perc_head(raw)
                z_txt = text_adapter(txt)
                z_p_v = z_p[:, :vith_dim] if z_p.shape[-1] >= vith_dim else F.pad(z_p, (0, vith_dim - z_p.shape[-1]))
                anchor = soft_mem(z_sem)
                g = torch.softmax(
                    fuse_gate(torch.cat([l2norm(z_p_v), l2norm(z_v), l2norm(z_sem)], dim=-1)), dim=-1
                )
                z_f = l2norm(g[:, 0:1] * l2norm(z_p_v) + g[:, 1:2] * l2norm(z_v) + g[:, 2:3] * l2norm(anchor))
                sem_list.append(l2norm(z_sem).cpu().numpy())
                vith_list.append(l2norm(z_v).cpu().numpy())
                fuse_list.append(z_f.cpu().numpy())
                gt_v.append(vith.numpy())
                gt_rn.append(img_projector(rn50.to(device)).cpu().numpy())
                txt_cos_list.append((l2norm(z_sem) * l2norm(z_txt)).sum(-1).cpu().numpy())

            sem = np.concatenate(sem_list)
            vv = np.concatenate(vith_list)
            ff = np.concatenate(fuse_list)
            gtv = np.concatenate(gt_v)
            gtr = np.concatenate(gt_rn)
            gtr_n = gtr / np.linalg.norm(gtr, axis=1, keepdims=True).clip(1e-8)
            gtv_n = gtv / np.linalg.norm(gtv, axis=1, keepdims=True).clip(1e-8)
            top5, top1, total = retrieve_all(sem, gtr_n, True)
            rn50_top1 = top1 / total * 100
            vith_cos = float(np.mean(np.sum(vv * gtv_n, axis=1)))
            fuse_cos = float(np.mean(np.sum(ff * gtv_n, axis=1)))
            txt_cos = float(np.concatenate(txt_cos_list).mean())
            score = rn50_top1 + 50 * fuse_cos + 20 * txt_cos

        row = {
            "epoch": epoch,
            "loss": ep_loss / len(train_loader),
            "rn50_top1": rn50_top1,
            "vith_cos": vith_cos,
            "fuse_cos": fuse_cos,
            "txt_cos": txt_cos,
            "score": score,
            "w_perc": wp,
            "w_sem": ws,
            "w_decode": wd,
        }
        history.append(row)
        print(
            f"epoch {epoch}: loss={row['loss']:.4f} rn50_top1={rn50_top1:.1f}% "
            f"txt_cos={txt_cos:.4f} vith_cos={vith_cos:.4f} fuse_cos={fuse_cos:.4f}"
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "eeg_projector_state_dict": eeg_projector.state_dict(),
                    "img_projector_state_dict": img_projector.state_dict(),
                    "perc_head": perc_head.state_dict(),
                    "dino_head": dino_head.state_dict(),
                    "vith_head": vith_head.state_dict(),
                    "text_adapter": text_adapter.state_dict(),
                    "fuse_gate": fuse_gate.state_dict(),
                    "probe": probe.state_dict(),
                    "layer_ids": layer_ids,
                    "subjects": subjects,
                    "img_dim": img_dim,
                    "feature_dim": feature_dim,
                    "eeg_sample_points": eeg_len,
                    "channels_num": channels_num,
                    "n_extra_blocks": int(ckpt.get("n_extra_blocks", 1)),
                    "design": "NDA-SS MindCross/MindBridge + CLIP-Image/Text",
                },
                out / "checkpoint_nda_ss_best.pth",
            )

    @torch.no_grad()
    def export(loader, tag: str):
        for m in (perc_head, vith_head, fuse_gate, eeg_projector, model):
            m.eval()
        sems, viths, fuses, projs = [], [], [], []
        for batch in loader:
            eeg = batch[0].to(device)
            sid = batch[6].to(device)
            raw = model(eeg, sid)
            z_sem = eeg_projector(raw)
            z_v = vith_head(raw)
            z_p = perc_head(raw)
            z_p_v = z_p[:, :vith_dim] if z_p.shape[-1] >= vith_dim else F.pad(z_p, (0, vith_dim - z_p.shape[-1]))
            anchor = soft_mem(z_sem)
            g = torch.softmax(
                fuse_gate(torch.cat([l2norm(z_p_v), l2norm(z_v), l2norm(z_sem)], dim=-1)), dim=-1
            )
            z_f = l2norm(g[:, 0:1] * l2norm(z_p_v) + g[:, 1:2] * l2norm(z_v) + g[:, 2:3] * l2norm(anchor))
            sems.append(l2norm(z_sem).cpu().numpy())
            viths.append(l2norm(z_v).cpu().numpy())
            fuses.append(z_f.cpu().numpy())
            projs.append(z_sem.float().cpu().numpy())
        np.save(out / f"z_sem_rn50_{tag}.npy", np.concatenate(sems).astype(np.float32))
        np.save(out / f"z_decode_vith_{tag}.npy", np.concatenate(viths).astype(np.float32))
        np.save(out / f"z_fuse_{tag}.npy", np.concatenate(fuses).astype(np.float32))
        np.save(out / f"z_eeg_proj_{tag}.npy", np.concatenate(projs).astype(np.float32))

    export(DataLoader(train_ds, batch_size=512, shuffle=False), "train")
    export(test_loader, "test")

    report = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "layer_ids": layer_ids,
        "history": history,
        "semantic_primary": "CLIP-Image(RN50/SSP-512) + CLIP-Text",
        "backbone": "SharedSpecificEncoder (MindCross+MindBridge)",
        "lambda_img": args.lambda_rn50,
        "lambda_txt": args.lambda_txt,
        "checkpoint": str(out / "checkpoint_nda_ss_best.pth"),
    }
    (out / "nda_train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "train_history.csv", index=False)
    print(json.dumps({k: report[k] for k in ("best_epoch", "best_score", "backbone", "lambda_img", "lambda_txt")}, indent=2))


if __name__ == "__main__":
    main()
