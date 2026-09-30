#!/usr/bin/env python3
"""Phase 2: MCAD-style dual-teacher fine-tune from RN50 NB checkpoint (sub-08)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))

from module.dataset import EEGPreImageDataset  # noqa: E402
from module.eeg_encoder.model import EEGProject  # noqa: E402
from module.loss import ContrastiveLoss  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402
from module.util import retrieve_all  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


class DualImageEEGDataset(EEGPreImageDataset):
  """EEG + RN50 image features + ViT-H image features (same object/image index)."""

  def __init__(self, vith_image_feature_dir: str, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.train:
      vith_path = os.path.join(vith_image_feature_dir, "image_train.npy")
    else:
      vith_path = os.path.join(vith_image_feature_dir, "image_test.npy")
    self.vith_image_features = np.load(vith_path)

  def __getitem__(self, index):
    eeg, image_feature, text_feature, subject_id, object_idx, image_idx, repetition_idx = super().__getitem__(index)
    vith_feature = torch.tensor(
      self.vith_image_features[object_idx, image_idx], dtype=torch.float32
    )
    return eeg, image_feature, vith_feature, text_feature, subject_id, object_idx, image_idx, repetition_idx


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
  ap.add_argument("--checkpoint", type=str, required=True)
  ap.add_argument("--subject", type=int, default=8)
  ap.add_argument("--eeg-data-dir", type=str, default="data/things_eeg/preprocessed_eeg")
  ap.add_argument("--rn50-feature-dir", type=str, default="data/things_eeg/image_feature/RN50")
  ap.add_argument("--vith-feature-dir", type=str, default="data/things_eeg/image_feature/ViT-H-14")
  ap.add_argument("--output-dir", type=str, default="outputs/nb_full_phases/sub-08/dual_teacher")
  ap.add_argument("--feature-dim", type=int, default=512)
  ap.add_argument("--lambda-vith", type=float, default=0.5)
  ap.add_argument("--lambda-direct", type=float, default=0.3)
  ap.add_argument("--num-epochs", type=int, default=25)
  ap.add_argument("--batch-size", type=int, default=1024)
  ap.add_argument("--learning-rate", type=float, default=5e-5)
  ap.add_argument("--device", type=str, default="cuda:0")
  ap.add_argument("--seed", type=int, default=2025)
  args = ap.parse_args()

  root = Path(args.nb_root)
  out_dir = Path(args.output_dir)
  if not out_dir.is_absolute():
    out_dir = root / out_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  device = torch.device(args.device if torch.cuda.is_available() else "cpu")

  eeg_dir = str(root / args.eeg_data_dir) if not Path(args.eeg_data_dir).is_absolute() else args.eeg_data_dir
  rn50_dir = str(root / args.rn50_feature_dir) if not Path(args.rn50_feature_dir).is_absolute() else args.rn50_feature_dir
  vith_dir = str(root / args.vith_feature_dir) if not Path(args.vith_feature_dir).is_absolute() else args.vith_feature_dir

  train_ds = DualImageEEGDataset(
    vith_dir, [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
    rn50_dir, "", False, [], True, False, None, True, False, False, False,
  )
  test_ds = DualImageEEGDataset(
    vith_dir, [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
    rn50_dir, "", False, [], True, False, None, False, False, False, False,
  )

  latent_dim = int(train_ds.image_features.shape[-1])
  vith_dim = int(train_ds.vith_image_features.shape[-1])
  channels_num = int(train_ds.channels_num)
  eeg_len = int(train_ds.num_sample_points)

  model = EEGProject(feature_dim=latent_dim, eeg_sample_points=eeg_len, channels_num=channels_num).to(device)
  eeg_projector = ProjectorLinear(latent_dim, args.feature_dim).to(device)
  img_projector = ProjectorLinear(latent_dim, args.feature_dim).to(device)
  eeg_projector_vith = ProjectorLinear(latent_dim, args.feature_dim).to(device)
  img_projector_vith = ProjectorLinear(vith_dim, args.feature_dim).to(device)
  eeg_head_vith1024 = nn.Linear(latent_dim, vith_dim).to(device)

  ckpt_path = Path(args.checkpoint)
  if not ckpt_path.is_absolute():
    ckpt_path = root / ckpt_path
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  model.load_state_dict(ckpt["model_state_dict"])
  eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
  img_projector.load_state_dict(ckpt["img_projector_state_dict"])
  # warm-start ViT-H heads from RN50 projectors
  eeg_projector_vith.load_state_dict(ckpt["eeg_projector_state_dict"])
  if img_projector_vith.linear.weight.shape == img_projector.linear.weight.shape:
    img_projector_vith.load_state_dict(img_projector.state_dict())
  else:
    img_projector_vith.linear.weight.data.zero_()
    if img_projector_vith.linear.bias is not None:
      img_projector_vith.linear.bias.data.zero_()

  text_projector = ProjectorLinear(latent_dim, args.feature_dim).to(device)
  criterion = ContrastiveLoss(0.07, 1.0, 1.0, False, True, False, False, True).to(device)

  params = (
    list(model.parameters())
    + list(eeg_projector.parameters())
    + list(img_projector.parameters())
    + list(eeg_projector_vith.parameters())
    + list(img_projector_vith.parameters())
    + list(eeg_head_vith1024.parameters())
    + list(text_projector.parameters())
  )
  optimizer = optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-4)

  train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)

  best_rn50_top1 = 0.0
  best_vith_top1 = 0.0
  best_epoch = 0

  for epoch in range(1, args.num_epochs + 1):
    model.train()
    for p in (
      eeg_projector, img_projector, eeg_projector_vith, img_projector_vith,
      eeg_head_vith1024, text_projector,
    ):
      p.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.num_epochs}"):
      eeg = batch[0].to(device)
      img_rn50 = batch[1].to(device)
      img_vith = batch[2].to(device)
      text = batch[3].to(device)

      raw = model(eeg)
      eeg_rn50 = eeg_projector(raw)
      eeg_vith = eeg_projector_vith(raw)
      eeg_direct = eeg_head_vith1024(raw)

      img_rn50_p = img_projector(img_rn50)
      img_vith_p = img_projector_vith(img_vith)
      text_p = text_projector(torch.zeros_like(img_rn50))

      loss_rn50 = criterion(eeg_rn50, img_rn50_p, text_p)
      loss_vith = criterion(eeg_vith, img_vith_p, text_p)
      loss_direct = 1.0 - torch.nn.functional.cosine_similarity(
        torch.nn.functional.normalize(eeg_direct, dim=-1),
        torch.nn.functional.normalize(img_vith, dim=-1),
      ).mean()

      loss = loss_rn50 + args.lambda_vith * loss_vith + args.lambda_direct * loss_direct
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += float(loss.item())

    model.eval()
    eeg_rn50_list, img_rn50_list = [], []
    eeg_vith_list, img_vith_list = [], []
    direct_list, vith_gt_list = [], []
    with torch.no_grad():
      for batch in test_loader:
        eeg = batch[0].to(device)
        img_rn50 = batch[1].to(device)
        img_vith = batch[2].to(device)
        raw = model(eeg)
        eeg_rn50_list.append(eeg_projector(raw).cpu().numpy())
        img_rn50_list.append(img_projector(img_rn50).cpu().numpy())
        eeg_vith_list.append(eeg_projector_vith(raw).cpu().numpy())
        img_vith_list.append(img_projector_vith(img_vith).cpu().numpy())
        direct_list.append(eeg_head_vith1024(raw).cpu().numpy())
        vith_gt_list.append(img_vith.cpu().numpy())

    eeg_rn50_all = np.concatenate(eeg_rn50_list)
    img_rn50_all = np.concatenate(img_rn50_list)
    eeg_vith_all = np.concatenate(eeg_vith_list)
    img_vith_all = np.concatenate(img_vith_list)
    direct_all = np.concatenate(direct_list)
    vith_gt_all = np.concatenate(vith_gt_list)

    top5_rn50, top1_rn50, total = retrieve_all(eeg_rn50_all, img_rn50_all, True)
    top5_vith, top1_vith, _ = retrieve_all(eeg_vith_all, img_vith_all, True)
    rn50_acc = top1_rn50 / total * 100
    vith_acc = top1_vith / total * 100
    d_norm = direct_all / np.linalg.norm(direct_all, axis=1, keepdims=True).clip(1e-8)
    g_norm = vith_gt_all / np.linalg.norm(vith_gt_all, axis=1, keepdims=True).clip(1e-8)
    direct_cos = float(np.mean(np.sum(d_norm * g_norm, axis=1)))

    print(
      f"epoch {epoch}: loss={total_loss/len(train_loader):.4f} "
      f"rn50_top1={rn50_acc:.1f}% vith_top1={vith_acc:.1f}% direct_cos={direct_cos:.4f}"
    )

    if rn50_acc >= best_rn50_top1 - 1.0 and vith_acc > best_vith_top1:
      best_rn50_top1 = max(best_rn50_top1, rn50_acc)
      best_vith_top1 = vith_acc
      best_epoch = epoch
      torch.save(
        {
          "epoch": epoch,
          "model_state_dict": model.state_dict(),
          "eeg_projector_state_dict": eeg_projector.state_dict(),
          "img_projector_state_dict": img_projector.state_dict(),
          "eeg_projector_vith_state_dict": eeg_projector_vith.state_dict(),
          "img_projector_vith_state_dict": img_projector_vith.state_dict(),
          "eeg_head_vith1024_state_dict": eeg_head_vith1024.state_dict(),
        },
        out_dir / "checkpoint_dual_best.pth",
      )

  # export test embeds
  @torch.no_grad()
  def encode_split(loader):
    direct_out = []
    for batch in loader:
      eeg = batch[0].to(device)
      raw = model(eeg)
      direct_out.append(eeg_head_vith1024(raw).float().cpu().numpy())
    return np.concatenate(direct_out, axis=0)

  model.eval()
  eeg_head_vith1024.eval()
  direct_test = encode_split(test_loader)
  direct_test = direct_test / np.linalg.norm(direct_test, axis=1, keepdims=True).clip(1e-8)
  np.save(out_dir / "dual_vith1024_test_clip_1024.npy", direct_test.astype(np.float32))

  report = {
    "phase": 2,
    "best_epoch": best_epoch,
    "best_rn50_top1": best_rn50_top1,
    "best_vith_top1": best_vith_top1,
    "direct_test_cos_to_gt_vith": float(np.mean(np.sum(direct_test * g_norm, axis=1))),
    "checkpoint": str(out_dir / "checkpoint_dual_best.pth"),
  }
  (out_dir / "dual_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
  pd.DataFrame(
    {
      "rn50 top1": f"{best_rn50_top1:.2f}",
      "vith top1": f"{best_vith_top1:.2f}",
      "best epoch": best_epoch,
    },
    index=[0],
  ).to_csv(out_dir / "result.csv", index=False)
  print(json.dumps(report, indent=2))
  print(f"[OK] {out_dir}")


if __name__ == "__main__":
  main()
