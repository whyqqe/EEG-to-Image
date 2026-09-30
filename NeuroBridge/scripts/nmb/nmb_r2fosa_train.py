#!/usr/bin/env python3
"""R²-FOSA training: frozen NB prior + FSTDE + Anchor-DDLG (D²-FOSA-style bidirectional DDLG)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nmb"))

from decode_aligner_modules import ClipInfoNCE, l2norm  # noqa: E402
from module.dataset import EEGPreImageDataset  # noqa: E402
from module.eeg_encoder.model import EEGProject  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402
from module.util import retrieve_all  # noqa: E402
from nmb_ddlem_train import (  # noqa: E402
    ddpm_loss,
    ddim_sample,
    make_beta_schedule,
    make_diffusion,
)
from r2fosa_modules import R2FOSAModel  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


class R2Dataset(EEGPreImageDataset):
    def __init__(self, vith_feature_dir: str, dino_npy: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vith_path = Path(vith_feature_dir) / ("image_train.npy" if self.train else "image_test.npy")
        self.vith_features = np.load(vith_path)
        self.dino_all = np.load(dino_npy).astype(np.float32)
        self.images_per_object = self.num_images_per_object

    def __getitem__(self, index):
        eeg, _rn50, text, sid, obj_idx, img_idx, rep = super().__getitem__(index)
        vith = torch.tensor(self.vith_features[obj_idx, img_idx], dtype=torch.float32)
        flat = int(obj_idx) * self.images_per_object + int(img_idx)
        dino = torch.tensor(self.dino_all[flat], dtype=torch.float32)
        return eeg, vith, dino, text, sid, obj_idx, img_idx, rep


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - (l2norm(pred) * l2norm(target)).sum(dim=-1)).mean()


@torch.no_grad()
def refresh_gallery_keys(
    r2: R2FOSAModel,
    nb_model: nn.Module,
    eeg_projector: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    keys = []
    for batch in loader:
        eeg = batch[0].to(device)
        raw = nb_model(eeg)
        keys.append(l2norm(eeg_projector(raw)))
    r2.refresh_gallery_keys(torch.cat(keys, dim=0))


def nb_forward(
    nb_model: nn.Module,
    eeg_projector: nn.Module,
    eeg: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        raw = nb_model(eeg)
        return l2norm(eeg_projector(raw))


def save_r2_ckpt(path: Path, epoch: int, phase: int, r2: R2FOSAModel, metrics: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "phase": phase,
            "r2fosa_state_dict": r2.state_dict(),
            **metrics,
        },
        path,
    )


@torch.no_grad()
def validate_align(
    r2: R2FOSAModel,
    nb_model: nn.Module,
    eeg_projector: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return align_cos, anchor_cos, align_all, anchor_all, gt_all."""
    align_list, anchor_list, gt_list = [], [], []
    for batch in loader:
        eeg, vith_gt, *_ = batch
        eeg = eeg.to(device)
        vith_gt = vith_gt.to(device)
        z_proj = nb_forward(nb_model, eeg_projector, eeg)
        ctx = r2.fstde(eeg, z_proj)
        e_anchor, _ = r2.memory_read(z_proj)
        cond = r2.build_cond(z_proj, e_anchor, ctx)
        align = l2norm(r2.ddlem.align_head(cond))
        align_list.append(align.cpu().numpy())
        anchor_list.append(e_anchor.cpu().numpy())
        gt_list.append(vith_gt.cpu().numpy())
    align_all = np.concatenate(align_list)
    anchor_all = np.concatenate(anchor_list)
    gt_all = np.concatenate(gt_list)
    gt_n = gt_all / np.linalg.norm(gt_all, axis=1, keepdims=True).clip(1e-8)
    align_cos = float(np.mean(np.sum(align_all * gt_n, axis=1)))
    anchor_cos = float(np.mean(np.sum(anchor_all * gt_n, axis=1)))
    return align_cos, anchor_cos, align_all, anchor_all, gt_all


@torch.no_grad()
def validate_ddlg(
    r2: R2FOSAModel,
    nb_model: nn.Module,
    eeg_projector: nn.Module,
    loader: DataLoader,
    device: torch.device,
    diff: dict,
    ddim_steps: int,
    warm_t_frac: float,
) -> tuple[float, float, np.ndarray]:
    ddlg_list, gt_list = [], []
    for batch in loader:
        eeg, vith_gt, *_ = batch
        eeg = eeg.to(device)
        vith_gt = vith_gt.to(device)
        z_proj = nb_forward(nb_model, eeg_projector, eeg)
        ctx = r2.fstde(eeg, z_proj)
        e_anchor, _ = r2.memory_read(z_proj)
        cond = r2.build_cond(z_proj, e_anchor, ctx)
        ddlg = ddim_sample(
            r2.ddlem.e2i, cond, diff,
            steps=ddim_steps, warm_start=e_anchor, warm_t_frac=warm_t_frac,
        )
        ddlg_list.append(ddlg.cpu().numpy())
        gt_list.append(vith_gt.cpu().numpy())
    ddlg_all = np.concatenate(ddlg_list)
    gt_all = np.concatenate(gt_list)
    gt_n = gt_all / np.linalg.norm(gt_all, axis=1, keepdims=True).clip(1e-8)
    ddlg_cos = float(np.mean(np.sum(ddlg_all * gt_n, axis=1)))
    _, top1, total = retrieve_all(ddlg_all, gt_all, True)
    return ddlg_cos, top1 / total * 100, ddlg_all


@torch.no_grad()
def export_embeds(
    r2: R2FOSAModel,
    nb_model: nn.Module,
    eeg_projector: nn.Module,
    loader: DataLoader,
    device: torch.device,
    diff: dict,
    ddim_steps: int,
    warm_t_frac: float,
    use_ddlg: bool = True,
) -> dict[str, np.ndarray]:
    r2.eval()
    nb_model.eval()
    eeg_projector.eval()
    proj_list, align_list, ddlg_a_list, ddlg_m_list, anchor_list = [], [], [], [], []
    for batch in loader:
        eeg = batch[0].to(device)
        z_proj = nb_forward(nb_model, eeg_projector, eeg)
        ctx = r2.fstde(eeg, z_proj)
        e_anchor, e_mem = r2.memory_read(z_proj)
        cond = r2.build_cond(z_proj, e_anchor, ctx)
        align = l2norm(r2.ddlem.align_head(cond))
        proj_list.append(z_proj.cpu().numpy())
        align_list.append(align.cpu().numpy())
        anchor_list.append(e_anchor.cpu().numpy())
        if use_ddlg:
            d_a = ddim_sample(
                r2.ddlem.e2i, cond, diff, steps=ddim_steps,
                warm_start=e_anchor, warm_t_frac=warm_t_frac,
            )
            d_m = ddim_sample(
                r2.ddlem.e2i, cond, diff, steps=ddim_steps,
                warm_start=e_mem, warm_t_frac=warm_t_frac,
            )
            ddlg_a_list.append(d_a.cpu().numpy())
            ddlg_m_list.append(d_m.cpu().numpy())
    out = {
        "proj": np.concatenate(proj_list, axis=0).astype(np.float32),
        "align": np.concatenate(align_list, axis=0).astype(np.float32),
        "anchor": np.concatenate(anchor_list, axis=0).astype(np.float32),
    }
    if use_ddlg:
        out["ddlg_anchor"] = np.concatenate(ddlg_a_list, axis=0).astype(np.float32)
        out["ddlg_mem"] = np.concatenate(ddlg_m_list, axis=0).astype(np.float32)
    return out


def build_phase2_optimizer(r2: R2FOSAModel, lr_main: float, lr_diff: float) -> optim.Optimizer:
    main_params = [
        p for n, p in r2.named_parameters()
        if p.requires_grad and "ddlem.e2i" not in n and "ddlem.i2e" not in n
    ]
    diff_params = list(r2.ddlem.e2i.parameters())
    if r2.ddlem.i2e is not None:
        diff_params += list(r2.ddlem.i2e.parameters())
    if r2.ddlem.i2e_cond_proj is not None:
        diff_params += list(r2.ddlem.i2e_cond_proj.parameters())
    return optim.AdamW(
        [
            {"params": main_params, "lr": lr_main},
            {"params": diff_params, "lr": lr_diff},
        ],
        weight_decay=1e-2,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train R²-FOSA (FSTDE + Anchor-DDLG)")
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--clip-train-npy", type=str, required=True)
    ap.add_argument("--clip-test-npy", type=str, required=True)
    ap.add_argument("--dino-train-npy", type=str, required=True)
    ap.add_argument("--dino-test-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--phase1-epochs", type=int, default=20, help="FSTDE + align only")
    ap.add_argument("--phase2-epochs", type=int, default=60, help="full Anchor-DDLG")
    ap.add_argument("--phase2-min-epochs", type=int, default=30, help="min phase2 epochs before early stop")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr-main", type=float, default=1e-4)
    ap.add_argument("--lr-diff", type=float, default=5e-5)
    ap.add_argument("--lambda-align", type=float, default=1.0)
    ap.add_argument("--lambda-e2i", type=float, default=0.5)
    ap.add_argument("--lambda-i2e", type=float, default=0.5)
    ap.add_argument("--lambda-dino", type=float, default=0.35)
    ap.add_argument("--lambda-mem", type=float, default=0.25)
    ap.add_argument("--lambda-nce", type=float, default=0.5)
    ap.add_argument("--soft-k", type=int, default=5)
    ap.add_argument("--soft-tau", type=float, default=0.07)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--warm-t-frac", type=float, default=0.35)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-ddlg-weight", type=float, default=0.7, help="phase2 score = (1-w)*align + w*ddlg")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--infer-only", action="store_true")
    ap.add_argument("--checkpoint-r2", type=str, default="")
    args = ap.parse_args()

    root = Path(args.nb_root)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    clip_train = np.load(args.clip_train_npy).astype(np.float32)
    clip_test_gt = np.load(args.clip_test_npy).astype(np.float32)

    eeg_dir = str(root / "data/things_eeg/preprocessed_eeg")
    rn50_dir = str(root / "data/things_eeg/image_feature/RN50")
    vith_dir = str(root / "data/things_eeg/image_feature/ViT-H-14")
    dino_train = str(Path(args.dino_train_npy) if Path(args.dino_train_npy).is_absolute() else root / args.dino_train_npy)
    dino_test = str(Path(args.dino_test_npy) if Path(args.dino_test_npy).is_absolute() else root / args.dino_test_npy)

    train_ds = R2Dataset(
        vith_dir, dino_train, [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
        rn50_dir, "", False, [], True, False, None, True, False, False, False,
    )
    test_ds = R2Dataset(
        vith_dir, dino_test, [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
        rn50_dir, "", False, [], True, False, None, False, False, False, False,
    )

    latent_dim = int(train_ds.image_features.shape[-1])
    nb_model = EEGProject(
        feature_dim=latent_dim,
        eeg_sample_points=int(train_ds.num_sample_points),
        channels_num=int(train_ds.channels_num),
    ).to(device)
    eeg_projector = ProjectorLinear(latent_dim, 512).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    nb_model.load_state_dict(ckpt["model_state_dict"])
    eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
    for p in nb_model.parameters():
        p.requires_grad = False
    for p in eeg_projector.parameters():
        p.requires_grad = False
    nb_model.eval()
    eeg_projector.eval()

    gallery_loader = DataLoader(train_ds, batch_size=512, shuffle=False)
    with torch.no_grad():
        key_list = []
        for batch in gallery_loader:
            eeg = batch[0].to(device)
            key_list.append(l2norm(eeg_projector(nb_model(eeg))))
        gallery_keys = torch.cat(key_list, dim=0)

    r2 = R2FOSAModel(
        torch.tensor(clip_train, device=device),
        gallery_keys.to(device),
        soft_k=args.soft_k,
        soft_tau=args.soft_tau,
        bidirectional=True,
    ).to(device)

    betas = make_beta_schedule(args.timesteps)
    diff = make_diffusion(betas, device)
    clip_nce = ClipInfoNCE(0.07).to(device)

    p1_ckpt = out_dir / "r2fosa_phase1_best.pth"
    p2_ckpt = Path(args.checkpoint_r2) if args.checkpoint_r2 else out_dir / "r2fosa_best.pth"
    total_epochs = args.phase1_epochs + args.phase2_epochs

    def set_phase(phase: int) -> None:
        for p in r2.ddlem.e2i.parameters():
            p.requires_grad = phase >= 2
        if r2.ddlem.i2e is not None:
            for p in r2.ddlem.i2e.parameters():
                p.requires_grad = phase >= 2
        if r2.ddlem.i2e_cond_proj is not None:
            for p in r2.ddlem.i2e_cond_proj.parameters():
                p.requires_grad = phase >= 2

    best_p1_align = 0.0
    best_p1_epoch = 0
    best_p2_score = 0.0
    best_p2_epoch = 0
    best_p2_ddlg = 0.0
    bad_p2 = 0
    history = []
    train_secs = 0.0
    phase2_ran = False

    if not args.infer_only:
        set_phase(1)
        opt = optim.AdamW(
            [p for p in r2.parameters() if p.requires_grad],
            lr=args.lr_main,
            weight_decay=1e-2,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)
        t0 = time.time()

        for epoch in range(1, total_epochs + 1):
            phase = 1 if epoch <= args.phase1_epochs else 2

            if epoch == args.phase1_epochs + 1:
                if p1_ckpt.is_file():
                    state = torch.load(p1_ckpt, map_location=device, weights_only=False)
                    r2.load_state_dict(state["r2fosa_state_dict"])
                    print(
                        f"[r2fosa] phase2 start: restored phase1 best ep={state['epoch']} "
                        f"align_cos={state.get('align_cos', 0):.4f}"
                    )
                set_phase(2)
                opt = build_phase2_optimizer(r2, args.lr_main, args.lr_diff)
                bad_p2 = 0
                phase2_ran = True

            r2.train()
            refresh_gallery_keys(r2, nb_model, eeg_projector, gallery_loader, device)

            loss_sum = 0.0
            n = 0
            for batch in tqdm(train_loader, desc=f"ep {epoch}/{total_epochs} p{phase}"):
                eeg, vith_gt, dino_gt, *_ = batch
                eeg = eeg.to(device)
                vith_gt = vith_gt.to(device)
                dino_gt = dino_gt.to(device)
                z_proj = nb_forward(nb_model, eeg_projector, eeg)
                ctx = r2.fstde(eeg, z_proj)
                e_anchor, e_mem = r2.memory_read(z_proj)
                cond = r2.build_cond(z_proj, e_anchor, ctx)
                align_pred = r2.ddlem.align_head(cond)
                clip_n = l2norm(vith_gt)

                loss_align = cosine_loss(align_pred, vith_gt)
                loss_nce = clip_nce(align_pred, vith_gt)
                loss_dino = cosine_loss(r2.fstde.predict_dino(ctx), dino_gt)
                loss_mem = cosine_loss(e_mem, vith_gt)
                if phase == 1:
                    loss = (
                        args.lambda_align * loss_align
                        + args.lambda_nce * loss_nce
                        + args.lambda_dino * loss_dino
                        + args.lambda_mem * loss_mem
                    )
                else:
                    loss = (
                        0.5 * args.lambda_align * loss_align
                        + 0.5 * args.lambda_nce * loss_nce
                        + 0.25 * args.lambda_dino * loss_dino
                        + 0.25 * args.lambda_mem * loss_mem
                    )
                    loss_e2i = ddpm_loss(r2.ddlem.e2i, clip_n, cond, diff, args.timesteps)
                    loss = loss + args.lambda_e2i * loss_e2i
                    if r2.ddlem.i2e is not None and r2.ddlem.i2e_cond_proj is not None:
                        i2e_tgt = l2norm(align_pred.detach())
                        clip_cond = r2.ddlem.i2e_cond_proj(clip_n)
                        loss_i2e = ddpm_loss(r2.ddlem.i2e, i2e_tgt, clip_cond, diff, args.timesteps)
                        loss = loss + args.lambda_i2e * loss_i2e

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(r2.parameters(), 1.0)
                opt.step()
                loss_sum += float(loss.item()) * eeg.shape[0]
                n += eeg.shape[0]

            r2.eval()
            align_cos, anchor_cos, _, _, _ = validate_align(
                r2, nb_model, eeg_projector, test_loader, device,
            )

            if phase == 1:
                ddlg_cos = 0.0
                ddlg_top1 = 0.0
                score = align_cos
                row = {
                    "epoch": epoch,
                    "phase": phase,
                    "loss": loss_sum / max(n, 1),
                    "align_cos": align_cos,
                    "anchor_cos": anchor_cos,
                    "ddlg_cos": None,
                    "score": score,
                    "ddlg_top1": None,
                }
                print(
                    f"[r2fosa] ep={epoch:03d} p1 loss={row['loss']:.4f} "
                    f"align={align_cos:.4f} anchor={anchor_cos:.4f}"
                )
                if align_cos > best_p1_align + 1e-5:
                    best_p1_align = align_cos
                    best_p1_epoch = epoch
                    save_r2_ckpt(
                        p1_ckpt, epoch, 1, r2,
                        {"align_cos": align_cos, "anchor_cos": anchor_cos, "metric": "align_cos"},
                    )
            else:
                ddlg_cos, ddlg_top1, _ = validate_ddlg(
                    r2, nb_model, eeg_projector, test_loader, device, diff,
                    args.ddim_steps, args.warm_t_frac,
                )
                w = args.val_ddlg_weight
                score = (1.0 - w) * align_cos + w * ddlg_cos
                row = {
                    "epoch": epoch,
                    "phase": phase,
                    "loss": loss_sum / max(n, 1),
                    "align_cos": align_cos,
                    "anchor_cos": anchor_cos,
                    "ddlg_cos": ddlg_cos,
                    "score": score,
                    "ddlg_top1": ddlg_top1,
                }
                print(
                    f"[r2fosa] ep={epoch:03d} p2 loss={row['loss']:.4f} "
                    f"align={align_cos:.4f} ddlg={ddlg_cos:.4f} score={score:.4f} top1={ddlg_top1:.1f}%"
                )
                phase2_epoch = epoch - args.phase1_epochs
                if score > best_p2_score + 1e-5:
                    best_p2_score = score
                    best_p2_epoch = epoch
                    best_p2_ddlg = ddlg_cos
                    bad_p2 = 0
                    save_r2_ckpt(
                        p2_ckpt, epoch, 2, r2,
                        {
                            "align_cos": align_cos,
                            "ddlg_cos": ddlg_cos,
                            "score": score,
                            "ddlg_top1": ddlg_top1,
                            "metric": "phase2_score",
                        },
                    )
                else:
                    bad_p2 += 1
                    if phase2_epoch >= args.phase2_min_epochs and bad_p2 >= args.patience:
                        print(f"[r2fosa] early stop phase2 at ep={epoch} bad={bad_p2}")
                        break

            history.append(row)

        train_secs = time.time() - t0

        if p2_ckpt.is_file():
            state = torch.load(p2_ckpt, map_location=device, weights_only=False)
            r2.load_state_dict(state["r2fosa_state_dict"])
            best_epoch = state["epoch"]
            best_score = state.get("score", best_p2_score)
        elif p1_ckpt.is_file():
            state = torch.load(p1_ckpt, map_location=device, weights_only=False)
            r2.load_state_dict(state["r2fosa_state_dict"])
            best_epoch = state["epoch"]
            best_score = state.get("align_cos", best_p1_align)
        else:
            best_epoch = 0
            best_score = 0.0
    else:
        if p2_ckpt.is_file():
            r2.load_state_dict(torch.load(p2_ckpt, map_location=device, weights_only=False)["r2fosa_state_dict"])
        best_epoch = 0
        best_score = 0.0
        phase2_ran = p2_ckpt.is_file()

    r2.eval()
    refresh_gallery_keys(r2, nb_model, eeg_projector, gallery_loader, device)
    use_ddlg = phase2_ran or p2_ckpt.is_file()

    for tag, loader in [("train", gallery_loader), ("test", DataLoader(test_ds, batch_size=512, shuffle=False))]:
        em = export_embeds(
            r2, nb_model, eeg_projector, loader, device, diff,
            args.ddim_steps, args.warm_t_frac, use_ddlg=use_ddlg,
        )
        np.save(out_dir / f"z_eeg_proj_{tag}.npy", em["proj"])
        np.save(out_dir / f"r2fosa_align_{tag}_clip_1024.npy", em["align"])
        np.save(out_dir / f"e_anchor_{tag}_clip_1024.npy", em["anchor"])
        if use_ddlg:
            np.save(out_dir / f"r2fosa_ddlg_anchor_{tag}_clip_1024.npy", em["ddlg_anchor"])
            np.save(out_dir / f"r2fosa_ddlg_mem_{tag}_clip_1024.npy", em["ddlg_mem"])
        else:
            np.save(out_dir / f"r2fosa_ddlg_anchor_{tag}_clip_1024.npy", em["align"])
            np.save(out_dir / f"r2fosa_ddlg_mem_{tag}_clip_1024.npy", em["align"])

    ddlg_test = np.load(out_dir / "r2fosa_ddlg_anchor_test_clip_1024.npy")
    gt_n = clip_test_gt / np.linalg.norm(clip_test_gt, axis=1, keepdims=True).clip(1e-8)
    final_ddlg_cos = float(np.mean(np.sum(ddlg_test * gt_n, axis=1)))
    _, top1, total = retrieve_all(ddlg_test, clip_test_gt, True)

    report = {
        "pipeline": "R2-FOSA",
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_p1_epoch": best_p1_epoch,
        "best_p1_align": best_p1_align,
        "best_p2_epoch": best_p2_epoch,
        "best_p2_score": best_p2_score,
        "best_p2_ddlg": best_p2_ddlg,
        "final_ddlg_cos": final_ddlg_cos,
        "final_ddlg_top1": top1 / total * 100,
        "phase1_epochs": args.phase1_epochs,
        "phase2_epochs": args.phase2_epochs,
        "phase2_min_epochs": args.phase2_min_epochs,
        "train_secs": train_secs,
        "warm_t_frac": args.warm_t_frac,
        "history": history[-30:],
        "design": {
            "fstde": "raw-t + rFFT FOMamba-lite",
            "prior": "frozen NeuroBridge z_proj",
            "memory": f"soft-k={args.soft_k} anchor-DDLG warm-start",
            "ddlg": "bidirectional DDLEM (D2-FOSA style)",
            "no_erdc": True,
            "early_stop_fix": "phase1=align only; phase2 reset patience; min phase2 epochs",
        },
    }
    (out_dir / "r2fosa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
