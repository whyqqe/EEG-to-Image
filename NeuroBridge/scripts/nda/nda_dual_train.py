#!/usr/bin/env python3
"""NDA-v2: Curriculum dual-stream training on frozen NeuroBridge backbone.

Semantic primary = CLIP-Image (RN50/SSP-512) + CLIP-Text (concept descriptions).
Decode / generation bridge = ViT-H-1024 head.
Perception = NVOL-HCF + DINOv2.
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nmb"))

from decode_aligner_modules import ClipInfoNCE, DifferentiableSoftMemory, ProbeDecoder, l2norm  # noqa: E402
from module.dataset import EEGPreImageDataset  # noqa: E402
from module.eeg_encoder.model import EEGProject  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402
from module.util import retrieve_all  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


class DualDataset(EEGPreImageDataset):
    def __init__(
        self,
        hcf_npy: str,
        dino_npy: str,
        vith_dir: str,
        text_flat_npy: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hcf_all = np.load(hcf_npy).astype(np.float32)
        self.dino_all = np.load(dino_npy).astype(np.float32)
        self.text_all = np.load(text_flat_npy).astype(np.float32)
        vith_path = Path(vith_dir) / ("image_train.npy" if self.train else "image_test.npy")
        self.vith_features = np.load(vith_path)
        self.images_per_object = self.num_images_per_object

    def __getitem__(self, index):
        eeg, img_rn50, _text_unused, sid, obj_idx, img_idx, rep = super().__getitem__(index)
        flat = int(obj_idx) * self.images_per_object + int(img_idx)
        hcf = torch.tensor(self.hcf_all[flat], dtype=torch.float32)
        dino = torch.tensor(self.dino_all[flat], dtype=torch.float32)
        txt = torch.tensor(self.text_all[flat], dtype=torch.float32)
        vith = torch.tensor(self.vith_features[obj_idx, img_idx], dtype=torch.float32)
        return eeg, img_rn50, vith, hcf, dino, txt, sid, obj_idx, img_idx, rep


def dla_weights(epoch: int, e1: int, e2: int) -> tuple[float, float, float]:
    """Return (w_perc, w_sem, w_decode) curriculum weights in [0,1] scaled later."""
    if epoch <= e1:
        return 1.0, 0.15, 0.0
    if epoch <= e2:
        t = (epoch - e1) / max(e2 - e1, 1)
        return 0.7 * (1 - t) + 0.35, 0.15 + 0.85 * t, 0.2 * t
    return 0.35, 1.0, 1.0


def build_hcf(layers_dir: Path, split: str, layer_ids: list[int], out_path: Path) -> None:
    arrs = []
    for li in layer_ids:
        p = layers_dir / split / f"layer_{li:02d}.npy"
        if not p.is_file():
            raise FileNotFoundError(p)
        a = np.load(p).astype(np.float32)
        a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-8)
        arrs.append(a)
    # equal-weight concat then will be projected; also save mean for simple target
    mean = np.mean(np.stack(arrs, axis=0), axis=0)
    mean = mean / np.linalg.norm(mean, axis=1, keepdims=True).clip(1e-8)
    np.save(out_path, mean.astype(np.float32))
    print(f"[OK] HCF {out_path} {mean.shape} layers={layer_ids}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--clip-layers-dir", type=str, required=True)
    ap.add_argument("--nvol-json", type=str, default="")
    ap.add_argument("--layers", type=str, default="")
    ap.add_argument("--dino-train-npy", type=str, required=True)
    ap.add_argument("--dino-test-npy", type=str, required=True)
    ap.add_argument("--clip-train-npy", type=str, required=True)
    ap.add_argument("--clip-test-npy", type=str, required=True)
    ap.add_argument("--text-train-npy", type=str, required=True, help="flat CLIP-Text features")
    ap.add_argument("--text-test-npy", type=str, required=True)
    ap.add_argument("--probe-supervision", type=str, default="")
    ap.add_argument("--num-epochs", type=int, default=40)
    ap.add_argument("--phase1-epochs", type=int, default=10)
    ap.add_argument("--phase2-epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda-rn50", type=float, default=0.8, help="CLIP-Image semantic (primary)")
    ap.add_argument(
        "--val-split-json",
        default="",
        help="leakfree.py split; when given, best checkpoint is chosen on held-in train "
        "concepts instead of the TEST set (audit finding P1/A3)",
    )
    ap.add_argument("--lambda-txt", type=float, default=0.2, help="CLIP-Text semantic regularizer")
    ap.add_argument("--lambda-vith", type=float, default=0.35, help="ViT-H decode bridge")
    ap.add_argument("--lambda-hcf", type=float, default=1.0)
    ap.add_argument("--lambda-dino", type=float, default=0.5)
    ap.add_argument("--lambda-probe", type=float, default=0.5)
    ap.add_argument("--lambda-mem", type=float, default=0.25)
    ap.add_argument("--soft-k", type=int, default=5)
    ap.add_argument("--soft-tau", type=float, default=0.07)
    ap.add_argument("--freeze-backbone", action="store_true", default=True)
    args = ap.parse_args()

    root = Path(args.nb_root)
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve NVOL layers
    if args.layers:
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]
    elif args.nvol_json and Path(args.nvol_json).is_file():
        layer_ids = json.loads(Path(args.nvol_json).read_text())["top_k_layers"]
    else:
        layer_ids = [12, 14, 16]
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

    latent_dim = int(train_ds.image_features.shape[-1])  # 1024 for EEGProject raw / actually image feat dim of RN50 path
    # EEGProject feature_dim follows image feature dim in NB training (=512 for RN50 encoder out? check)
    # In NeuroBridge RN50: image features are 1024? Let's check - ProjectorLinear(latent_dim, 512)
    # train_ds.image_features.shape[-1] is RN50 feature dim
    img_dim = int(train_ds.image_features.shape[-1])
    channels_num = int(train_ds.channels_num)
    eeg_len = int(train_ds.num_sample_points)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Infer encoder out dim from state dict
    # EEGProject typically outputs same as image feature dim used in training
    model = EEGProject(feature_dim=img_dim, eeg_sample_points=eeg_len, channels_num=channels_num).to(device)
    eeg_projector = ProjectorLinear(img_dim, 512).to(device)
    img_projector = ProjectorLinear(img_dim, 512).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
    if ckpt.get("img_projector_state_dict"):
        img_projector.load_state_dict(ckpt["img_projector_state_dict"])
    img_projector.eval()
    for p in img_projector.parameters():
        p.requires_grad = False

    hcf_dim = int(np.load(hcf_train).shape[1])
    dino_dim = int(np.load(args.dino_train_npy).shape[1])
    text_dim = int(np.load(args.text_train_npy).shape[1])
    vith_dim = 1024

    perc_head = nn.Sequential(
        nn.Linear(img_dim, 1024),
        nn.GELU(),
        nn.Linear(1024, hcf_dim),
    ).to(device)
    dino_head = nn.Linear(img_dim, dino_dim).to(device)
    # Map CLIP-Text (1024) -> SSP-512 to match z_sem
    text_adapter = nn.Sequential(
        nn.Linear(text_dim, 512),
        nn.GELU(),
        nn.Linear(512, 512),
    ).to(device)
    # Decode bridge to ViT-H
    vith_head = nn.Sequential(
        nn.Linear(img_dim, 1024),
        nn.GELU(),
        nn.Linear(1024, vith_dim),
    ).to(device)
    fuse_gate = nn.Sequential(nn.Linear(vith_dim * 2 + 512, 256), nn.GELU(), nn.Linear(256, 3)).to(device)
    probe = ProbeDecoder(dim=vith_dim).to(device)
    nce = ClipInfoNCE(0.07).to(device)

    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in eeg_projector.parameters():
            p.requires_grad = False
        model.eval()
        eeg_projector.eval()

    clip_train = torch.tensor(np.load(args.clip_train_npy), device=device, dtype=torch.float32)

    # gallery keys from frozen proj
    with torch.no_grad():
        keys = []
        for batch in DataLoader(train_ds, batch_size=512, shuffle=False):
            eeg = batch[0].to(device)
            keys.append(eeg_projector(model(eeg)))
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

    params = list(perc_head.parameters()) + list(dino_head.parameters()) + list(vith_head.parameters())
    params += list(text_adapter.parameters()) + list(fuse_gate.parameters()) + list(probe.parameters())
    if not args.freeze_backbone:
        params += list(model.parameters()) + list(eeg_projector.parameters())
    opt = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)

    # ---- LEAK-FREE selection: hold out train concepts for checkpoint choice ----
    # Audit finding P1/A3: `score` (and hence best_epoch) used to be computed on the
    # 200 TEST concepts. We now score on held-in valB concepts and keep the test
    # metrics for reporting only. The soft-memory bank also excludes valB rows so
    # validation retrieval is not inflated by the val samples' own keys.
    sel_loader = None
    if args.val_split_json:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import leakfree as LF

        sp = LF.load(args.val_split_json)
        n_rows = len(train_ds)
        fi = LF.rows_for(sp, "fit", n_rows)
        vi = LF.rows_for(sp, "val_b", n_rows)
        assert len(set(fi.tolist()) & set(vi.tolist())) == 0
        sel_loader = DataLoader(Subset(train_ds, vi.tolist()), batch_size=256, shuffle=False)
        # rebuild the memory bank without valB (keeps selection unbiased)
        with torch.no_grad():
            keys = []
            for batch in DataLoader(Subset(train_ds, fi.tolist()), batch_size=512, shuffle=False):
                keys.append(eeg_projector(model(batch[0].to(device))))
            gallery_keys = torch.cat(keys, dim=0)
        soft_mem = DifferentiableSoftMemory(clip_train, gallery_keys, args.soft_k, args.soft_tau).to(device)
        print(f"[leakfree] selection on {len(vi)} held-in val rows; memory bank {len(fi)} rows (valB removed)")

    best_score, best_epoch, history = -1e9, 0, []

    for epoch in range(1, args.num_epochs + 1):
        wp, ws, wd = dla_weights(epoch, args.phase1_epochs, args.phase2_epochs)
        perc_head.train()
        dino_head.train()
        vith_head.train()
        text_adapter.train()
        fuse_gate.train()
        probe.train()

        ep_loss = 0.0
        for batch in tqdm(train_loader, desc=f"ep{epoch}"):
            eeg, rn50, vith, hcf, dino, txt, sid, *_ = batch
            eeg = eeg.to(device)
            rn50 = rn50.to(device)
            vith = vith.to(device)
            hcf = hcf.to(device)
            dino = dino.to(device)
            txt = txt.to(device)

            with torch.no_grad() if args.freeze_backbone else torch.enable_grad():
                raw = model(eeg)
                z_sem = eeg_projector(raw)  # 512-d CLIP-Image SSP space

            z_p = perc_head(raw)
            z_dino = dino_head(raw)
            z_v = vith_head(raw)
            z_txt = text_adapter(txt)

            loss_hcf = (1 - (l2norm(z_p) * l2norm(hcf)).sum(-1)).mean() + nce(z_p, hcf)
            loss_dino = (1 - (l2norm(z_dino) * l2norm(dino)).sum(-1)).mean()
            # Semantic: CLIP-Image (primary) + CLIP-Text (regularizer)
            rn50_512 = img_projector(rn50)
            loss_img = (1 - (l2norm(z_sem) * l2norm(rn50_512)).sum(-1)).mean() + nce(z_sem, rn50_512)
            loss_txt = (1 - (l2norm(z_sem) * l2norm(z_txt)).sum(-1)).mean() + nce(z_sem, z_txt)
            loss_sem = args.lambda_rn50 * loss_img + args.lambda_txt * loss_txt
            # Decode bridge: ViT-H image space
            loss_vith = (1 - (l2norm(z_v) * l2norm(vith)).sum(-1)).mean() + 0.5 * nce(z_v, vith)

            anchor = soft_mem(z_sem)
            if z_p.shape[-1] != vith_dim:
                z_p_v = z_p[:, :vith_dim] if z_p.shape[-1] >= vith_dim else F.pad(z_p, (0, vith_dim - z_p.shape[-1]))
            else:
                z_p_v = z_p
            g_logits = fuse_gate(torch.cat([l2norm(z_p_v), l2norm(z_v), l2norm(z_sem)], dim=-1))
            g = torch.softmax(g_logits, dim=-1)
            z_f = l2norm(g[:, 0:1] * l2norm(z_p_v) + g[:, 1:2] * l2norm(z_v) + g[:, 2:3] * l2norm(anchor))

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
            )
            # Keep a small CLIP-Image anchor so retrieval doesn't collapse early
            loss = loss + 0.05 * args.lambda_rn50 * loss_img

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += float(loss.item())

        # Eval
        perc_head.eval()
        vith_head.eval()
        text_adapter.eval()
        fuse_gate.eval()
        probe.eval()

        def run_eval(loader):
            """returns the metric dict for one loader (val or test)"""
            with torch.no_grad():
                sem_list, vith_list, gt_v, gt_rn, fuse_list, txt_cos_list = [], [], [], [], [], []
                for batch in loader:
                    eeg, rn50, vith, hcf, dino, txt, sid, *_ = batch
                    eeg = eeg.to(device)
                    txt = txt.to(device)
                    raw = model(eeg)
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
                    txt_cos_list.append(
                        (l2norm(z_sem) * l2norm(z_txt)).sum(-1).cpu().numpy()
                    )

                sem = np.concatenate(sem_list)
                vv = np.concatenate(vith_list)
                ff = np.concatenate(fuse_list)
                gtv = np.concatenate(gt_v)
                gtr = np.concatenate(gt_rn)
                gtr_n = gtr / np.linalg.norm(gtr, axis=1, keepdims=True).clip(1e-8)
                gtv_n = gtv / np.linalg.norm(gtv, axis=1, keepdims=True).clip(1e-8)
                top5, top1, total = retrieve_all(sem, gtr_n, True)
                return {
                    "rn50_top1": top1 / total * 100,
                    "vith_cos": float(np.mean(np.sum(vv * gtv_n, axis=1))),
                    "fuse_cos": float(np.mean(np.sum(ff * gtv_n, axis=1))),
                    "txt_cos": float(np.concatenate(txt_cos_list).mean()),
                }

        if sel_loader is not None:
            m = run_eval(sel_loader)
            mt = run_eval(test_loader)
            rn50_top1, vith_cos = m["rn50_top1"], m["vith_cos"]
            fuse_cos, txt_cos = m["fuse_cos"], m["txt_cos"]
            test_rn50_top1, test_fuse_cos = mt["rn50_top1"], mt["fuse_cos"]
            score = rn50_top1 + 50 * fuse_cos + 20 * txt_cos
            selected_on = "val"
        else:
            m = run_eval(test_loader)
            rn50_top1, vith_cos = m["rn50_top1"], m["vith_cos"]
            fuse_cos, txt_cos = m["fuse_cos"], m["txt_cos"]
            test_rn50_top1, test_fuse_cos = rn50_top1, fuse_cos
            score = rn50_top1 + 50 * fuse_cos + 20 * txt_cos
            selected_on = "test(contaminated)"

        row = {
            "epoch": epoch,
            "loss": ep_loss / len(train_loader),
            "w_perc": wp,
            "w_sem": ws,
            "w_decode": wd,
            "rn50_top1": rn50_top1,
            "vith_cos": vith_cos,
            "fuse_cos": fuse_cos,
            "txt_cos": txt_cos,
            "score": score,
        }
        history.append(row)
        print(
            f"epoch {epoch}: loss={row['loss']:.4f} rn50_top1={rn50_top1:.1f}% "
            f"txt_cos={txt_cos:.4f} vith_cos={vith_cos:.4f} fuse_cos={fuse_cos:.4f} "
            f"wp={wp:.2f} ws={ws:.2f} wd={wd:.2f}"
        )

        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(
                {
                    "epoch": epoch,
                    "perc_head": perc_head.state_dict(),
                    "dino_head": dino_head.state_dict(),
                    "vith_head": vith_head.state_dict(),
                    "text_adapter": text_adapter.state_dict(),
                    "fuse_gate": fuse_gate.state_dict(),
                    "probe": probe.state_dict(),
                    "layer_ids": layer_ids,
                    "lambdas": {
                        "img": args.lambda_rn50,
                        "txt": args.lambda_txt,
                        "vith": args.lambda_vith,
                    },
                    "design": {
                        "semantic_primary": "CLIP-Image(RN50/SSP-512) + CLIP-Text",
                        "semantic_secondary_decode": "ViT-H-1024",
                        "perception": "HCF(NVOL)+DINOv2",
                    },
                },
                out / "checkpoint_nda_best.pth",
            )

    # Export
    @torch.no_grad()
    def export(loader, tag: str):
        perc_head.eval()
        vith_head.eval()
        fuse_gate.eval()
        sems, viths, fuses, projs = [], [], [], []
        for batch in loader:
            eeg = batch[0].to(device)
            raw = model(eeg)
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
        "lambda_img": args.lambda_rn50,
        "lambda_txt": args.lambda_txt,
        "why_semantic": (
            "Semantic stream aligns EEG to CLIP-Image (RN50/SSP, retrieval-optimal) "
            "and CLIP-Text (concept description templates), with ViT-H as decode bridge only."
        ),
        "checkpoint": str(out / "checkpoint_nda_best.pth"),
    }
    (out / "nda_train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "train_history.csv", index=False)
    print(json.dumps({k: report[k] for k in ("best_epoch", "best_score", "layer_ids", "semantic_primary", "lambda_img", "lambda_txt")}, indent=2))


if __name__ == "__main__":
    main()
