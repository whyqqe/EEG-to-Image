#!/usr/bin/env python3
"""Train SDXL Condition Calibrator (SCC): frozen NB + decode-aware anchor-residual head."""

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
sys.path.insert(0, str(NB_ROOT / "scripts" / "nmb"))

from scc_modules import (  # noqa: E402
    ClipInfoNCE,
    HardNegativeNCE,
    SDXLConditionCalibrator,
    l2norm,
)
from module.dataset import EEGPreImageDataset  # noqa: E402
from module.eeg_encoder.model import EEGProject  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402
from module.util import retrieve_all  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


class SCCDataset(EEGPreImageDataset):
    def __init__(self, vith_feature_dir: str, dino_npy: str, hard_neg_npy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.train:
            vith_path = Path(vith_feature_dir) / "image_train.npy"
        else:
            vith_path = Path(vith_feature_dir) / "image_test.npy"
        self.vith_features = np.load(vith_path)
        self.dino_all = np.load(dino_npy).astype(np.float32)
        self.hard_neg = None
        if hard_neg_npy and Path(hard_neg_npy).is_file():
            self.hard_neg = np.load(hard_neg_npy)
        self.images_per_object = self.num_images_per_object

    def __getitem__(self, index):
        eeg, img_rn50, text, sid, obj_idx, img_idx, rep = super().__getitem__(index)
        vith = torch.tensor(self.vith_features[obj_idx, img_idx], dtype=torch.float32)
        flat = int(obj_idx) * self.images_per_object + int(img_idx)
        dino = torch.tensor(self.dino_all[flat], dtype=torch.float32)
        if self.hard_neg is not None:
            hn = torch.tensor(self.hard_neg[flat], dtype=torch.float32)
        else:
            hn = torch.zeros(8, 1024, dtype=torch.float32)
        return eeg, img_rn50, vith, dino, hn, text, sid, obj_idx, img_idx, rep


def load_probe_supervision(path: Path, device: torch.device) -> dict[str, torch.Tensor] | None:
    if not path.is_file():
        return None
    data = np.load(path)
    return {
        "eeg": torch.tensor(data["eeg_embed"], device=device),
        "anchor": torch.tensor(data["anchor_embed"], device=device),
        "clip_gen": torch.tensor(data["clip_gen"], device=device),
        "indices": data["indices"],
    }


def build_hard_negatives(
    gallery_keys: np.ndarray,
    gallery_clip: np.ndarray,
    out_path: Path,
    k_hard: int = 8,
) -> None:
    """Per train sample: top-k gallery CLIP embeds excluding self-match."""
    gk = gallery_keys / np.linalg.norm(gallery_keys, axis=1, keepdims=True).clip(1e-8)
    gc = gallery_clip / np.linalg.norm(gallery_clip, axis=1, keepdims=True).clip(1e-8)
    sim = gk @ gk.T
    n = sim.shape[0]
    hard = np.zeros((n, k_hard, gc.shape[1]), dtype=np.float32)
    for i in range(n):
        order = np.argsort(-sim[i])
        picked = [j for j in order if j != i][:k_hard]
        hard[i] = gc[picked]
    np.save(out_path, hard)
    print(f"[OK] hard negatives {hard.shape} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--output-dir", type=str, default="outputs/nb_scc/sub-08")
    ap.add_argument("--eeg-data-dir", type=str, default="data/things_eeg/preprocessed_eeg")
    ap.add_argument("--rn50-feature-dir", type=str, default="data/things_eeg/image_feature/RN50")
    ap.add_argument("--vith-feature-dir", type=str, default="data/things_eeg/image_feature/ViT-H-14")
    ap.add_argument("--clip-train-npy", type=str, default="")
    ap.add_argument("--clip-test-npy", type=str, default="")
    ap.add_argument("--dino-train-npy", type=str, required=True)
    ap.add_argument("--dino-test-npy", type=str, required=True)
    ap.add_argument("--probe-supervision", type=str, default="")
    ap.add_argument("--hard-neg-npy", type=str, default="")
    ap.add_argument("--num-epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--lambda-clip", type=float, default=1.0)
    ap.add_argument("--lambda-dino", type=float, default=0.35)
    ap.add_argument("--lambda-probe", type=float, default=0.5)
    ap.add_argument("--lambda-mem", type=float, default=0.25)
    ap.add_argument("--lambda-res", type=float, default=0.1)
    ap.add_argument("--soft-k", type=int, default=5)
    ap.add_argument("--soft-tau", type=float, default=0.07)
    ap.add_argument("--feature-dim", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warm-probe", type=str, default="", help="optional decode_aligner probe ckpt")
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

    hard_neg_path = args.hard_neg_npy or str(out_dir / "hard_neg_train.npy")
    if not Path(hard_neg_path).is_file():
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = root / ckpt_path
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        tmp_ds = EEGPreImageDataset(
            [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
            rn50_dir, "", False, [], True, False, None, True, False, False, False,
        )
        latent_dim = int(tmp_ds.image_features.shape[-1])
        channels_num = int(tmp_ds.channels_num)
        eeg_len = int(tmp_ds.num_sample_points)
        nb = EEGProject(feature_dim=latent_dim, eeg_sample_points=eeg_len, channels_num=channels_num)
        proj = ProjectorLinear(latent_dim, args.feature_dim)
        nb.load_state_dict(ckpt["model_state_dict"])
        proj.load_state_dict(ckpt["eeg_projector_state_dict"])
        nb.eval()
        proj.eval()
        keys_list = []
        loader_tmp = DataLoader(tmp_ds, batch_size=512, shuffle=False)
        with torch.no_grad():
            for batch in loader_tmp:
                keys_list.append(proj(nb(batch[0])).cpu().numpy())
        keys_np = np.concatenate(keys_list, axis=0)
        build_hard_negatives(keys_np, clip_train, Path(hard_neg_path))

    train_ds = SCCDataset(
        vith_dir, dino_train, hard_neg_path, [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
        rn50_dir, "", False, [], True, False, None, True, False, False, False,
    )
    test_ds = SCCDataset(
        vith_dir, dino_test, "", [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
        rn50_dir, "", False, [], True, False, None, False, False, False, False,
    )

    latent_dim = int(train_ds.image_features.shape[-1])
    channels_num = int(train_ds.channels_num)
    eeg_len = int(train_ds.num_sample_points)

    nb_model = EEGProject(feature_dim=latent_dim, eeg_sample_points=eeg_len, channels_num=channels_num).to(device)
    eeg_projector = ProjectorLinear(latent_dim, args.feature_dim).to(device)
    img_projector = ProjectorLinear(latent_dim, args.feature_dim).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    nb_model.load_state_dict(ckpt["model_state_dict"])
    eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
    img_projector.load_state_dict(ckpt["img_projector_state_dict"])

    for p in nb_model.parameters():
        p.requires_grad = False
    for p in eeg_projector.parameters():
        p.requires_grad = False
    for p in img_projector.parameters():
        p.requires_grad = False
    nb_model.eval()
    eeg_projector.eval()

    with torch.no_grad():
        gallery_keys_list = []
        loader_g = DataLoader(train_ds, batch_size=512, shuffle=False)
        for batch in loader_g:
            eeg = batch[0].to(device)
            gallery_keys_list.append(eeg_projector(nb_model(eeg)).cpu())
        gallery_keys = torch.cat(gallery_keys_list, dim=0)

    scc = SDXLConditionCalibrator(
        torch.tensor(clip_train, device=device),
        gallery_keys.to(device),
        nb_dim=args.feature_dim,
        soft_k=args.soft_k,
        soft_tau=args.soft_tau,
    ).to(device)

    if args.warm_probe:
        ws = Path(args.warm_probe)
        if not ws.is_absolute():
            ws = root / ws
        if ws.is_file():
            wsd = torch.load(ws, map_location=device, weights_only=False)
            if "probe_state_dict" in wsd:
                scc.probe.load_state_dict(wsd["probe_state_dict"])
                print(f"[INFO] warm probe from {ws}")

    probe_sup = load_probe_supervision(
        Path(args.probe_supervision) if args.probe_supervision else out_dir / "probe" / "probe_supervision.npz",
        device,
    )

    clip_nce = ClipInfoNCE(0.07).to(device)
    hard_nce = HardNegativeNCE(0.07).to(device)

    params = [p for p in scc.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)

    best_score = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.num_epochs + 1):
        scc.train()
        with torch.no_grad():
            keys = []
            for batch in loader_g:
                eeg = batch[0].to(device)
                keys.append(eeg_projector(nb_model(eeg)))
            scc.refresh_gallery_keys(torch.cat(keys, dim=0))

        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.num_epochs}"):
            eeg, _rn50, vith_gt, dino_gt, hard_neg, *_ = batch
            eeg = eeg.to(device)
            vith_gt = vith_gt.to(device)
            dino_gt = dino_gt.to(device)

            with torch.no_grad():
                z_nb = nb_model(eeg)
                z_proj = eeg_projector(z_nb)

            out = scc(eeg, z_proj)
            e_cond = out["e_cond"]
            e_mem = out["e_mem"]
            e_anchor = out["e_anchor"]
            vith_n = l2norm(vith_gt)
            dino_n = l2norm(dino_gt)

            loss_clip_cos = (1.0 - (e_cond * vith_n).sum(dim=-1)).mean()
            loss_clip_nce = clip_nce(e_cond, vith_gt)
            if hard_neg is not None:
                hn = hard_neg.to(device)
                loss_hard = hard_nce(e_cond, vith_gt, hn)
            else:
                loss_hard = torch.tensor(0.0, device=device)
            loss_dino = (1.0 - (out["dino_pred"] * dino_n).sum(dim=-1)).mean()
            loss_mem = (1.0 - (e_mem * vith_n).sum(dim=-1)).mean()
            loss_res = (out["delta"].pow(2).sum(dim=-1)).mean()

            probe_pred = out["probe_pred"]
            loss_probe = (1.0 - (probe_pred * vith_n).sum(dim=-1)).mean()
            if probe_sup is not None:
                ps = l2norm(scc.probe(probe_sup["eeg"], probe_sup["anchor"]))
                loss_probe_off = (1.0 - (ps * probe_sup["clip_gen"]).sum(dim=-1)).mean()
                loss_probe = 0.5 * loss_probe + 0.5 * loss_probe_off

            loss = (
                args.lambda_clip * (loss_clip_cos + loss_clip_nce + 0.5 * loss_hard)
                + args.lambda_dino * loss_dino
                + args.lambda_probe * loss_probe
                + args.lambda_mem * loss_mem
                + args.lambda_res * loss_res
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())

        scheduler.step()

        scc.eval()
        cond_list, probe_list, vith_gt_list, sigma_list = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                eeg, _rn50, vith_gt, _dino, *_ = batch
                eeg = eeg.to(device)
                vith_gt = vith_gt.to(device)
                z_nb = nb_model(eeg)
                z_proj = eeg_projector(z_nb)
                out = scc(eeg, z_proj)
                cond_list.append(out["e_cond"].cpu().numpy())
                probe_list.append(out["probe_pred"].cpu().numpy())
                vith_gt_list.append(vith_gt.cpu().numpy())
                sigma_list.append(out["sigma"].cpu().numpy())

        cond_all = np.concatenate(cond_list)
        probe_all = np.concatenate(probe_list)
        vith_all = np.concatenate(vith_gt_list)
        sigma_mean = float(np.mean(np.concatenate(sigma_list)))
        c_norm = cond_all / np.linalg.norm(cond_all, axis=1, keepdims=True).clip(1e-8)
        g_norm = vith_all / np.linalg.norm(vith_all, axis=1, keepdims=True).clip(1e-8)
        cond_cos = float(np.mean(np.sum(c_norm * g_norm, axis=1)))
        probe_cos = float(np.mean(np.sum(probe_all * g_norm, axis=1)))
        top5, top1, total = retrieve_all(c_norm, g_norm, True)
        vith_top1 = top1 / total * 100
        decode_score = cond_cos + 0.5 * probe_cos

        row = {
            "epoch": epoch,
            "loss": epoch_loss / len(train_loader),
            "cond_cos": cond_cos,
            "probe_cos": probe_cos,
            "decode_score": decode_score,
            "vith_top1": vith_top1,
            "sigma_mean": sigma_mean,
        }
        history.append(row)
        print(
            f"epoch {epoch}: loss={row['loss']:.4f} cond_cos={cond_cos:.4f} "
            f"probe_cos={probe_cos:.4f} score={decode_score:.4f} top1={vith_top1:.1f}%"
        )

        if decode_score > best_score:
            best_score = decode_score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "scc_state_dict": scc.state_dict(),
                    "soft_k": args.soft_k,
                    "soft_tau": args.soft_tau,
                    "decode_score": decode_score,
                    "cond_cos": cond_cos,
                    "probe_cos": probe_cos,
                },
                out_dir / "checkpoint_scc_best.pth",
            )

    @torch.no_grad()
    def export_split(loader, name: str):
        cond_out, proj_out, sigma_out = [], [], []
        for batch in loader:
            eeg = batch[0].to(device)
            z_nb = nb_model(eeg)
            z_proj = eeg_projector(z_nb)
            out = scc(eeg, z_proj)
            cond_out.append(out["e_cond"].float().cpu().numpy())
            proj_out.append(z_proj.float().cpu().numpy())
            sigma_out.append(out["sigma"].float().cpu().numpy())
        c = np.concatenate(cond_out, axis=0)
        p = np.concatenate(proj_out, axis=0)
        s = np.concatenate(sigma_out, axis=0)
        np.save(out_dir / f"scc_cond_{name}_clip_1024.npy", c.astype(np.float32))
        np.save(out_dir / f"z_eeg_proj_{name}.npy", p.astype(np.float32))
        np.save(out_dir / f"scc_sigma_{name}.npy", s.astype(np.float32))

    scc.eval()
    export_split(DataLoader(train_ds, batch_size=512, shuffle=False), "train")
    export_split(test_loader, "test")

    report = {
        "pipeline": "SDXL-Condition-Calibrator",
        "best_epoch": best_epoch,
        "best_decode_score": best_score,
        "final_cond_cos": cond_cos,
        "final_probe_cos": probe_cos,
        "final_vith_top1": vith_top1,
        "frozen_nb_checkpoint": str(ckpt_path),
        "lambdas": {
            "clip": args.lambda_clip,
            "dino": args.lambda_dino,
            "probe": args.lambda_probe,
            "mem": args.lambda_mem,
            "res": args.lambda_res,
        },
        "checkpoint": str(out_dir / "checkpoint_scc_best.pth"),
        "history": history,
    }
    (out_dir / "scc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    print(json.dumps(report, indent=2))
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    main()
