"""Minimal staged trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg_brainit.models.pipeline import EEGBrainITPipeline
from eeg_brainit.training.losses import EEGBrainITLoss
from eeg_brainit.training.teacher_bank import TeacherEmbeddingBank
from eeg_brainit.utils.config import ensure_dirs
from eeg_brainit.utils.metrics import pixel_correlation, ssim_simple


STAGE_LR = {
    0: 1e-3,
    1: 1e-4,
    2: 5e-5,
    3: 1e-5,
    4: 5e-6,
}


class Trainer:
    def __init__(
        self,
        model: EEGBrainITPipeline,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: dict[str, Any],
        device: torch.device,
        stage: int = 1,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.stage = stage
        self.model.apply_stage(stage)

        lr = float(cfg.get("train", {}).get("learning_rate", STAGE_LR[stage]))
        wd = float(cfg.get("train", {}).get("weight_decay", 0.01))
        head_mult = float(cfg.get("train", {}).get("head_lr_mult", 1.0))

        loss_cfg = cfg.get("loss", {})
        self.criterion = EEGBrainITLoss(
            lambda_clip=float(loss_cfg.get("lambda_clip", 1.0)),
            lambda_vgg=float(loss_cfg.get("lambda_vgg", 0.5)),
            lambda_pixel=float(loss_cfg.get("lambda_pixel", 0.0)),
            lambda_reg=float(loss_cfg.get("lambda_reg", 0.0)),
            lambda_nce=float(loss_cfg.get("lambda_nce", 0.0)),
            lambda_siglip=float(loss_cfg.get("lambda_siglip", 0.0)),
            lambda_vf=float(loss_cfg.get("lambda_vf", 0.0)),
            temperature=float(loss_cfg.get("temperature", 0.07)),
            use_proxy_fallback=bool(loss_cfg.get("use_proxy_fallback", True)),
            learnable_logit_scale=bool(loss_cfg.get("learnable_logit_scale", True)),
            init_logit_scale=float(loss_cfg.get("init_logit_scale", 2.6592)),
        ).to(device)

        head_params = []
        other_params = []
        head_ids: set[int] = set()
        if getattr(self.model, "clip_align_head", None) is not None:
            head_ids |= {id(p) for p in self.model.clip_align_head.parameters()}
        if getattr(self.model, "direct_clip_head", None) is not None:
            head_ids |= {id(p) for p in self.model.direct_clip_head.parameters()}
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in head_ids:
                head_params.append(p)
            else:
                other_params.append(p)
        # Trainable loss params (logit scale).
        for p in self.criterion.parameters():
            if p.requires_grad:
                head_params.append(p)

        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": lr * head_mult})
        if not param_groups:
            raise RuntimeError("No trainable parameters")
        self.optim = torch.optim.AdamW(param_groups, weight_decay=wd)
        self.scaler = GradScaler(enabled=device.type == "cuda")

        self.output_dir = Path(cfg.get("output_dir", "outputs/default"))
        ensure_dirs(self.output_dir, self.output_dir / "checkpoints")
        self.grad_clip = float(cfg.get("train", {}).get("grad_clip", 1.0))
        self.accum = int(cfg.get("train", {}).get("grad_accum_steps", 1))
        self.max_steps = int(cfg.get("train", {}).get("max_steps", 0))
        self.global_step = 0

        bank_cfg = cfg.get("teacher_bank", {})
        self.bank_size = int(bank_cfg.get("size", 0))
        self.bank_pool = int(bank_cfg.get("pool", max(self.bank_size * 4, self.bank_size)))
        self.hard_negatives = bool(bank_cfg.get("hard_negatives", False))
        self.teacher_bank: TeacherEmbeddingBank | None = None
        if self.bank_size > 0:
            data_cfg = cfg.get("data", {})
            teacher_raw = data_cfg.get("teacher_dir")
            if not teacher_raw:
                raise ValueError("teacher_bank.size>0 requires data.teacher_dir")
            root = Path(cfg.get("project_root", "."))
            teacher_dir = Path(teacher_raw)
            if not teacher_dir.is_absolute():
                teacher_dir = root / teacher_dir
            self.teacher_bank = TeacherEmbeddingBank(teacher_dir, device=device).to(device)
            print(
                f"[INFO] Teacher bank size={self.bank_size} pool={self.bank_pool} "
                f"hard_negatives={self.hard_negatives}"
            )

    @torch.no_grad()
    def _mine_bank(self, pred: torch.Tensor, exclude: torch.Tensor | None) -> torch.Tensor:
        assert self.teacher_bank is not None
        if not self.hard_negatives:
            return self.teacher_bank.sample(self.bank_size, exclude=exclude)
        pool = self.teacher_bank.sample(self.bank_pool, exclude=exclude)
        # Per-example hardest negatives against current predictions.
        sim = pred @ pool.t()  # (B, pool)
        k = min(self.bank_size, pool.shape[0])
        idx = sim.topk(k, dim=1).indices  # (B, k)
        hard = pool[idx]  # (B, k, D) via advanced indexing on first dim... 
        # pool[idx] with idx (B,k) gives (B,k,D) correctly in PyTorch
        return hard

    def _step(self, batch: dict[str, Any]) -> dict[str, float]:
        spec = batch["spectrogram"].to(self.device, non_blocking=True)
        batch_dev: dict[str, Any] = {
            "image": batch["image"].to(self.device, non_blocking=True),
            "id": batch.get("id"),
        }
        if "clip_emb" in batch:
            batch_dev["clip_emb"] = batch["clip_emb"].to(self.device, non_blocking=True)
        excl = batch.get("teacher_idx")
        if excl is not None:
            excl = excl.to(torch.long)

        with autocast(enabled=self.device.type == "cuda"):
            outputs = self.model(spec)
            if self.teacher_bank is not None and self.bank_size > 0 and "clip_emb" in outputs:
                pred_det = F.normalize(outputs["clip_emb"].detach().float(), dim=-1)
                batch_dev["teacher_bank"] = self._mine_bank(pred_det, excl)
            losses = self.criterion(outputs, batch_dev)
            loss = losses["total"] / self.accum
        self.scaler.scale(loss).backward()
        return {k: float(v.detach().item()) for k, v in losses.items()}

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        modules = [
            (self.model.encoder, any(p.requires_grad for p in self.model.encoder.parameters())),
            (self.model.virtual_fmri, any(p.requires_grad for p in self.model.virtual_fmri.parameters())),
            (self.model.bit, any(p.requires_grad for p in self.model.bit.parameters())),
            (self.model.projector, any(p.requires_grad for p in self.model.projector.parameters())),
        ]
        if getattr(self.model, "clip_align_head", None) is not None:
            head = self.model.clip_align_head
            modules.append((head, any(p.requires_grad for p in head.parameters())))
        if getattr(self.model, "direct_clip_head", None) is not None:
            dhead = self.model.direct_clip_head
            modules.append((dhead, any(p.requires_grad for p in dhead.parameters())))
        for module, flag in modules:
            if not flag:
                module.eval()

        meters: dict[str, float] = {}
        n_steps = 0
        self.optim.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"stage{self.stage} epoch{epoch}")
        for step, batch in enumerate(pbar, start=1):
            stats = self._step(batch)
            self.global_step += 1
            n_steps += 1
            for k, v in stats.items():
                meters[k] = meters.get(k, 0.0) + v
            if step % self.accum == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in list(self.model.parameters()) + list(self.criterion.parameters()) if p.requires_grad],
                        self.grad_clip,
                    )
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)
            pbar.set_postfix(
                loss=stats.get("total", 0.0),
                gap=stats.get("cos_gap", 0.0),
                step=self.global_step,
            )
            if self.max_steps > 0 and self.global_step >= self.max_steps:
                break
        n = max(n_steps, 1)
        return {k: v / n for k, v in meters.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}
        self.model.eval()
        totals: dict[str, float] = {}
        n = 0
        max_val = int(self.cfg.get("train", {}).get("max_val_batches", 0))
        paired_all = []
        shuffled_all = []
        for batch in self.val_loader:
            spec = batch["spectrogram"].to(self.device)
            img = batch["image"].to(self.device)
            out = self.model(spec)
            vf = out.get("virtual_fmri")
            if vf is not None and vf.shape[1] >= 2:
                a = vf[:, 0:1]
                b = vf[:, 1:2]
                totals["vf_ssim"] = totals.get("vf_ssim", 0.0) + ssim_simple(a, b)
                totals["vf_pixcorr"] = totals.get("vf_pixcorr", 0.0) + pixel_correlation(a, b)
            if "clip_tokens" in out:
                totals["clip_norm"] = totals.get("clip_norm", 0.0) + out["clip_tokens"].norm(dim=-1).mean().item()
            totals["img_mean"] = totals.get("img_mean", 0.0) + img.mean().item()
            if "clip_emb" in out and "clip_emb" in batch:
                pred = F.normalize(out["clip_emb"].float(), dim=-1)
                tgt = F.normalize(batch["clip_emb"].to(self.device).float(), dim=-1)
                paired = (pred * tgt).sum(dim=-1)
                shift = torch.roll(tgt, shifts=1, dims=0)
                shuffled = (pred * shift).sum(dim=-1)
                paired_all.append(paired.mean().item())
                shuffled_all.append(shuffled.mean().item())
                # Cheap in-batch retrieval proxy (Top-1).
                sim = pred @ tgt.t()
                pred_idx = sim.argmax(dim=1)
                labels = torch.arange(pred.shape[0], device=pred.device)
                totals["batch_top1"] = totals.get("batch_top1", 0.0) + (pred_idx == labels).float().mean().item()
            n += 1
            if max_val > 0 and n >= max_val:
                break
        out_stats = {k: v / max(n, 1) for k, v in totals.items()}
        if paired_all:
            p = float(sum(paired_all) / len(paired_all))
            s = float(sum(shuffled_all) / len(shuffled_all))
            out_stats["cos_paired"] = p
            out_stats["cos_shuffled"] = s
            out_stats["cos_gap"] = p - s
        return out_stats

    def save(self, epoch: int, tag: str = "last") -> Path:
        path = self.output_dir / "checkpoints" / f"stage{self.stage}_{tag}_e{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "stage": self.stage,
                "model": self.model.state_dict(),
                "criterion": self.criterion.state_dict(),
                "optim": self.optim.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )
        return path

    def fit(self) -> None:
        epochs = int(self.cfg.get("train", {}).get("epochs", 10))
        best = float("inf")
        best_gap = -1e9
        patience = int(self.cfg.get("train", {}).get("early_stopping_patience", 10))
        select_by_gap = bool(self.cfg.get("train", {}).get("select_by_cos_gap", True))
        select_by_top1 = bool(self.cfg.get("train", {}).get("select_by_batch_top1", False))
        bad = 0
        best_top1 = -1.0
        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(epoch)
            val_stats = self.validate()
            print(f"[epoch {epoch}] train={train_stats} val={val_stats}", flush=True)
            self.save(epoch, tag="last")
            improved = False
            if select_by_top1 and "batch_top1" in val_stats:
                top1 = float(val_stats["batch_top1"])
                if top1 > best_top1:
                    best_top1 = top1
                    best = float(train_stats.get("total", best))
                    best_gap = float(val_stats.get("cos_gap", best_gap))
                    improved = True
                    self.save(epoch, tag="best")
                    print(f"[INFO] New best val batch_top1={best_top1:.4f}", flush=True)
            elif select_by_gap and "cos_gap" in val_stats:
                gap = float(val_stats["cos_gap"])
                if gap > best_gap:
                    best_gap = gap
                    best = float(train_stats.get("total", best))
                    improved = True
                    self.save(epoch, tag="best")
                    print(f"[INFO] New best val cos_gap={best_gap:.4f}", flush=True)
            else:
                score = train_stats.get("total", best)
                if score < best:
                    best = score
                    improved = True
                    self.save(epoch, tag="best")
            if improved:
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"[INFO] Early stopping at epoch {epoch}", flush=True)
                    break
            if self.max_steps > 0 and self.global_step >= self.max_steps:
                print(f"[INFO] Reached max_steps={self.max_steps}", flush=True)
                break
        print(
            f"[INFO] Training finished stage={self.stage} best_loss={best:.6f} "
            f"best_cos_gap={best_gap if best_gap > -1e8 else float('nan'):.6f} "
            f"best_batch_top1={best_top1 if best_top1 >= 0 else float('nan'):.6f}",
            flush=True,
        )
