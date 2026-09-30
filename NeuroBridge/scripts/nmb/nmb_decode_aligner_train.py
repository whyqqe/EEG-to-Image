#!/usr/bin/env python3
"""DecodeAligner training: CLIP + DINOv2 + Probe + soft-memory decodable alignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nb_atm"))

from decode_aligner_modules import (  # noqa: E402
    ClipInfoNCE,
    DifferentiableSoftMemory,
    ProbeDecoder,
    l2norm,
)
from module.dataset import EEGPreImageDataset  # noqa: E402
from module.eeg_encoder.model import EEGProject  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402
from module.util import retrieve_all  # noqa: E402
from nb_encoder_factory import load_train_checkpoint  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


class AlignerDataset(EEGPreImageDataset):
    def __init__(
        self,
        vith_feature_dir: str,
        dino_npy: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.train:
            vith_path = Path(vith_feature_dir) / "image_train.npy"
        else:
            vith_path = Path(vith_feature_dir) / "image_test.npy"
        self.vith_features = np.load(vith_path)
        self.dino_all = np.load(dino_npy).astype(np.float32)
        self.images_per_object = self.num_images_per_object

    def __getitem__(self, index):
        eeg, img_rn50, text, sid, obj_idx, img_idx, rep = super().__getitem__(index)
        vith = torch.tensor(self.vith_features[obj_idx, img_idx], dtype=torch.float32)
        flat = int(obj_idx) * self.images_per_object + int(img_idx)
        dino = torch.tensor(self.dino_all[flat], dtype=torch.float32)
        return eeg, img_rn50, vith, dino, text, sid, obj_idx, img_idx, rep


def load_probe_supervision(path: Path, device: torch.device) -> dict[str, torch.Tensor] | None:
    if not path.is_file():
        return None
    data = np.load(path)
    return {
        "eeg": torch.tensor(data["eeg_embed"], device=device),
        "anchor": torch.tensor(data["anchor_embed"], device=device),
        "clip_gen": torch.tensor(data["clip_gen"], device=device),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--eeg-data-dir", type=str, default="data/things_eeg/preprocessed_eeg")
    ap.add_argument("--rn50-feature-dir", type=str, default="data/things_eeg/image_feature/RN50")
    ap.add_argument("--vith-feature-dir", type=str, default="data/things_eeg/image_feature/ViT-H-14")
    ap.add_argument("--clip-train-npy", type=str, default="")
    ap.add_argument("--clip-test-npy", type=str, default="")
    ap.add_argument("--dino-train-npy", type=str, required=True)
    ap.add_argument("--dino-test-npy", type=str, required=True)
    ap.add_argument("--probe-supervision", type=str, default="")
    ap.add_argument("--output-dir", type=str, default="outputs/nb_decode_aligner/sub-08")
    ap.add_argument("--feature-dim", type=int, default=512)
    ap.add_argument("--vith-dim", type=int, default=1024)
    ap.add_argument("--num-epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--lambda-clip", type=float, default=1.0)
    ap.add_argument("--lambda-dino", type=float, default=0.35)
    ap.add_argument("--lambda-probe", type=float, default=0.5)
    ap.add_argument("--lambda-mem", type=float, default=0.25)
    ap.add_argument("--soft-k", type=int, default=5)
    ap.add_argument("--soft-tau", type=float, default=0.07)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder-type", type=str, default="eegproject", choices=["eegproject", "atm"])
    ap.add_argument("--warm-start", type=str, default="", help="optional s2_dual or prior aligner ckpt")
    args = ap.parse_args()

    root = Path(args.nb_root)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    brainit = Path("/project/peilab/why/eeg-brainit")
    clip_train_path = Path(args.clip_train_npy or brainit / "outputs/atm_bridge/clip_img_train_1024.npy")
    clip_test_path = Path(args.clip_test_npy or brainit / "outputs/atm_bridge/clip_img_test_1024.npy")
    clip_train = np.load(clip_train_path).astype(np.float32)
    clip_test_gt = np.load(clip_test_path).astype(np.float32)

    eeg_dir = str(root / args.eeg_data_dir) if not Path(args.eeg_data_dir).is_absolute() else args.eeg_data_dir
    rn50_dir = str(root / args.rn50_feature_dir) if not Path(args.rn50_feature_dir).is_absolute() else args.rn50_feature_dir
    vith_dir = str(root / args.vith_feature_dir) if not Path(args.vith_feature_dir).is_absolute() else args.vith_feature_dir
    dino_train = str(Path(args.dino_train_npy) if Path(args.dino_train_npy).is_absolute() else root / args.dino_train_npy)
    dino_test = str(Path(args.dino_test_npy) if Path(args.dino_test_npy).is_absolute() else root / args.dino_test_npy)

    train_ds = AlignerDataset(
        vith_dir, dino_train, [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
        rn50_dir, "", False, [], True, False, None, True, False, False, False,
    )
    test_ds = AlignerDataset(
        vith_dir, dino_test, [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
        rn50_dir, "", False, [], True, False, None, False, False, False, False,
    )

    latent_dim = int(train_ds.image_features.shape[-1])
    channels_num = int(train_ds.channels_num)
    eeg_len = int(train_ds.num_sample_points)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path

    model, eeg_projector, enc_meta = load_train_checkpoint(
        ckpt_path,
        args.encoder_type,
        latent_dim,
        eeg_len,
        channels_num,
        args.feature_dim,
        device,
    )
    encoder_type = enc_meta["encoder_type"]
    if encoder_type == "atm" and enc_meta["proj_out_dim"] == latent_dim:
        eeg_projector = ProjectorLinear(latent_dim, args.feature_dim).to(device)

    img_projector = ProjectorLinear(latent_dim, args.feature_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    img_proj_sd = ckpt.get("img_projector_state_dict")
    if img_proj_sd and len(img_proj_sd) > 0:
        img_projector.load_state_dict(img_proj_sd)

    head_in = latent_dim if encoder_type == "atm" else latent_dim
    eeg_head_vith = nn.Linear(head_in, args.vith_dim).to(device)
    dino_head = nn.Linear(head_in, args.vith_dim).to(device)

    def forward_raw(eeg_batch, sid_batch):
        if encoder_type == "atm":
            return model(eeg_batch, sid_batch)
        return model(eeg_batch)

    # Gallery keys: projected EEG train
    with torch.no_grad():
        gallery_keys_list = []
        loader_g = DataLoader(train_ds, batch_size=512, shuffle=False)
        for batch in loader_g:
            eeg = batch[0].to(device)
            sid = batch[5].to(device)
            gallery_keys_list.append(eeg_projector(forward_raw(eeg, sid)).cpu())
        gallery_keys = torch.cat(gallery_keys_list, dim=0)

    soft_mem = DifferentiableSoftMemory(
        torch.tensor(clip_train, device=device),
        gallery_keys.to(device),
        soft_k=args.soft_k,
        tau=args.soft_tau,
    ).to(device)

    probe = ProbeDecoder(dim=args.vith_dim).to(device)
    clip_nce = ClipInfoNCE(0.07).to(device)

    if args.warm_start:
        ws = Path(args.warm_start)
        if not ws.is_absolute():
            ws = root / ws
        if ws.is_file():
            wsd = torch.load(ws, map_location=device, weights_only=False)
            if "eeg_head_vith1024_state_dict" in wsd:
                eeg_head_vith.load_state_dict(wsd["eeg_head_vith1024_state_dict"])
            if "probe_state_dict" in wsd:
                probe.load_state_dict(wsd["probe_state_dict"])
            print(f"[INFO] warm-start from {ws}")

    probe_sup = load_probe_supervision(
        Path(args.probe_supervision) if args.probe_supervision else out_dir / "probe" / "probe_supervision.npz",
        device,
    )

    params = (
        list(model.parameters())
        + list(eeg_projector.parameters())
        + list(eeg_head_vith.parameters())
        + list(dino_head.parameters())
        + list(probe.parameters())
    )
    optimizer = optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)

    best_cos = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        eeg_projector.train()
        eeg_head_vith.train()
        dino_head.train()
        probe.train()

        # Refresh gallery keys each epoch
        with torch.no_grad():
            keys = []
            for batch in loader_g:
                eeg = batch[0].to(device)
                sid = batch[5].to(device)
                keys.append(eeg_projector(forward_raw(eeg, sid)))
            soft_mem.gallery_keys.copy_(l2norm(torch.cat(keys, dim=0)))

        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.num_epochs}"):
            eeg, _rn50, vith_gt, dino_gt, _text, sid, *_ = batch
            eeg = eeg.to(device)
            sid = sid.to(device)
            vith_gt = vith_gt.to(device)
            dino_gt = dino_gt.to(device)

            raw = forward_raw(eeg, sid)
            e_vith = eeg_head_vith(raw)
            e_proj = eeg_projector(raw)
            e_vith_n = l2norm(e_vith)
            vith_n = l2norm(vith_gt)
            dino_pred = l2norm(dino_head(raw))
            dino_n = l2norm(dino_gt)

            anchor = soft_mem(e_proj)
            loss_clip_cos = (1.0 - (e_vith_n * vith_n).sum(dim=-1)).mean()
            loss_clip_nce = clip_nce(e_vith, vith_gt)
            loss_dino = (1.0 - (dino_pred * dino_n).sum(dim=-1)).mean()
            loss_mem = (1.0 - (l2norm(anchor) * vith_n).sum(dim=-1)).mean()

            probe_pred = l2norm(probe(e_vith_n, anchor))
            loss_probe = (1.0 - (probe_pred * vith_n).sum(dim=-1)).mean()
            if probe_sup is not None:
                ps = l2norm(probe(probe_sup["eeg"], probe_sup["anchor"]))
                loss_probe_off = (1.0 - (ps * probe_sup["clip_gen"]).sum(dim=-1)).mean()
                loss_probe = 0.5 * loss_probe + 0.5 * loss_probe_off

            loss = (
                args.lambda_clip * (loss_clip_cos + loss_clip_nce)
                + args.lambda_dino * loss_dino
                + args.lambda_probe * loss_probe
                + args.lambda_mem * loss_mem
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())

        # Eval
        model.eval()
        eeg_head_vith.eval()
        direct_list, vith_gt_list, probe_list = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                eeg, _rn50, vith_gt, _dino, _text, sid, *_ = batch
                eeg = eeg.to(device)
                sid = sid.to(device)
                vith_gt = vith_gt.to(device)
                raw = forward_raw(eeg, sid)
                e_vith = l2norm(eeg_head_vith(raw))
                e_proj = eeg_projector(raw)
                anchor = soft_mem(e_proj)
                probe_out = l2norm(probe(e_vith, anchor))
                direct_list.append(e_vith.cpu().numpy())
                vith_gt_list.append(vith_gt.cpu().numpy())
                probe_list.append(probe_out.cpu().numpy())

        direct_all = np.concatenate(direct_list)
        vith_all = np.concatenate(vith_gt_list)
        probe_all = np.concatenate(probe_list)
        d_norm = direct_all / np.linalg.norm(direct_all, axis=1, keepdims=True).clip(1e-8)
        g_norm = vith_all / np.linalg.norm(vith_all, axis=1, keepdims=True).clip(1e-8)
        direct_cos = float(np.mean(np.sum(d_norm * g_norm, axis=1)))
        probe_cos = float(np.mean(np.sum(probe_all * g_norm, axis=1)))
        top5, top1, total = retrieve_all(d_norm, g_norm, True)
        vith_top1 = top1 / total * 100

        row = {
            "epoch": epoch,
            "loss": epoch_loss / len(train_loader),
            "direct_cos": direct_cos,
            "probe_cos": probe_cos,
            "vith_top1": vith_top1,
        }
        history.append(row)
        print(
            f"epoch {epoch}: loss={row['loss']:.4f} direct_cos={direct_cos:.4f} "
            f"probe_cos={probe_cos:.4f} vith_top1={vith_top1:.1f}%"
        )

        if direct_cos > best_cos:
            best_cos = direct_cos
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "eeg_projector_state_dict": eeg_projector.state_dict(),
                    "eeg_head_vith_state_dict": eeg_head_vith.state_dict(),
                    "dino_head_state_dict": dino_head.state_dict(),
                    "probe_state_dict": probe.state_dict(),
                    "soft_k": args.soft_k,
                    "soft_tau": args.soft_tau,
                },
                out_dir / "checkpoint_decode_aligner_best.pth",
            )

    # Export embeds for downstream decode
    @torch.no_grad()
    def export_split(loader, name: str):
        vith_out, proj_out = [], []
        for batch in loader:
            eeg = batch[0].to(device)
            sid = batch[5].to(device)
            vith_out.append(eeg_head_vith(forward_raw(eeg, sid)).float().cpu().numpy())
            proj_out.append(eeg_projector(forward_raw(eeg, sid)).float().cpu().numpy())
        v = np.concatenate(vith_out, axis=0)
        p = np.concatenate(proj_out, axis=0)
        v = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-8)
        np.save(out_dir / f"decode_vith1024_{name}_clip_1024.npy", v.astype(np.float32))
        np.save(out_dir / f"z_eeg_proj_{name}.npy", p.astype(np.float32))

    model.eval()
    export_split(DataLoader(train_ds, batch_size=512, shuffle=False), "train")
    export_split(test_loader, "test")

    report = {
        "best_epoch": best_epoch,
        "best_direct_cos": best_cos,
        "final_probe_cos": probe_cos,
        "final_vith_top1": vith_top1,
        "lambdas": {
            "clip": args.lambda_clip,
            "dino": args.lambda_dino,
            "probe": args.lambda_probe,
            "mem": args.lambda_mem,
        },
        "checkpoint": str(out_dir / "checkpoint_decode_aligner_best.pth"),
        "history": history,
    }
    (out_dir / "decode_aligner_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    print(json.dumps(report, indent=2))
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    main()
