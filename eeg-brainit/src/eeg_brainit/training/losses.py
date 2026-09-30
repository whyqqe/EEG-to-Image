"""Training losses for staged EEG-Brain-IT fine-tuning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGBrainITLoss(nn.Module):
    def __init__(
        self,
        lambda_clip: float = 1.0,
        lambda_vgg: float = 0.5,
        lambda_pixel: float = 0.0,
        lambda_reg: float = 0.0,
        lambda_nce: float = 0.0,
        lambda_siglip: float = 0.0,
        lambda_vf: float = 0.0,
        lambda_distill: float = 0.0,
        temperature: float = 0.07,
        use_proxy_fallback: bool = True,
        learnable_logit_scale: bool = True,
        init_logit_scale: float = 2.6592,  # log(1/0.07) ≈ CLIP default
    ) -> None:
        super().__init__()
        self.lambda_clip = lambda_clip
        self.lambda_vgg = lambda_vgg
        self.lambda_pixel = lambda_pixel
        self.lambda_reg = lambda_reg
        self.lambda_nce = lambda_nce
        self.lambda_siglip = lambda_siglip
        self.lambda_vf = lambda_vf
        self.lambda_distill = lambda_distill
        self.temperature = temperature
        self.use_proxy_fallback = use_proxy_fallback
        if learnable_logit_scale:
            self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))
        else:
            self.register_buffer("logit_scale", torch.tensor(float(init_logit_scale)), persistent=False)

    def _scale(self) -> torch.Tensor:
        # Clamp to keep training stable (CLIP-style).
        return self.logit_scale.exp().clamp(1.0, 100.0)

    def _info_nce(
        self,
        pred: torch.Tensor,
        pos: torch.Tensor,
        bank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = max(self.temperature, 1e-6)
        logits = pred @ pos.t() / t
        if bank is not None and bank.numel() > 0:
            if bank.dim() == 2:
                neg = pred @ bank.t() / t
            else:
                # (B, K, D)
                neg = torch.einsum("bd,bkd->bk", pred, bank) / t
            logits = torch.cat([logits, neg], dim=1)
        labels = torch.arange(pred.shape[0], device=pred.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(pos @ pred.t() / t, labels)
        return 0.5 * (loss_i2t + loss_t2i)

    def _siglip(
        self,
        pred: torch.Tensor,
        pos: torch.Tensor,
        bank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sigmoid pairwise loss: immune to huge-softmax dilution."""
        scale = self._scale()
        # Positive pairs
        pos_logit = (pred * pos).sum(dim=-1) * scale
        loss = -F.logsigmoid(pos_logit).mean()

        # In-batch negatives (off-diagonal)
        if pred.shape[0] > 1:
            logits_ib = (pred @ pos.t()) * scale
            eye = torch.eye(pred.shape[0], device=pred.device, dtype=torch.bool)
            neg_ib = logits_ib.masked_select(~eye)
            loss = loss - F.logsigmoid(-neg_ib).mean()

        # Bank / hard negatives
        if bank is not None and bank.numel() > 0:
            if bank.dim() == 2:
                neg_logit = (pred @ bank.t()) * scale
            else:
                neg_logit = torch.einsum("bd,bkd->bk", pred, bank) * scale
            loss = loss - F.logsigmoid(-neg_logit).mean()
        return loss

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
        targets: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        ref = outputs.get("clip_emb", outputs.get("clip_tokens"))
        total = ref.new_zeros(())
        targets = targets or {}

        if "clip_emb" in outputs and "clip_emb" in batch:
            pred = F.normalize(outputs["clip_emb"].float(), dim=-1)
            tgt = F.normalize(batch["clip_emb"].float(), dim=-1)
            bank = batch.get("teacher_bank")

            if self.lambda_clip > 0:
                cosine = (pred * tgt).sum(dim=-1).mean()
                clip_loss = 1.0 - cosine
                losses["clip"] = clip_loss
                total = total + self.lambda_clip * clip_loss

            if self.lambda_nce > 0:
                nce = self._info_nce(pred, tgt, bank=bank)
                losses["nce"] = nce
                total = total + self.lambda_nce * nce

            if self.lambda_siglip > 0:
                sig = self._siglip(pred, tgt, bank=bank)
                losses["siglip"] = sig
                total = total + self.lambda_siglip * sig
                losses["logit_scale"] = self._scale().detach()

            with torch.no_grad():
                paired = (pred * tgt).sum(dim=-1).mean()
                if pred.shape[0] > 1:
                    shift = torch.roll(tgt, shifts=1, dims=0)
                    shuffled = (pred * shift).sum(dim=-1).mean()
                else:
                    shuffled = paired * 0
                losses["cos_paired"] = paired
                losses["cos_gap"] = paired - shuffled

        # Distill student CLIP head toward frozen ATM teacher (keeps EEG→fMRI-bridge semantics).
        if self.lambda_distill > 0 and "atm_emb" in outputs:
            student = outputs.get("clip_emb", outputs.get("bridge_clip"))
            if student is not None:
                teacher = F.normalize(outputs["atm_emb"].float(), dim=-1)
                student_n = F.normalize(student.float(), dim=-1)
                distill = 1.0 - (student_n * teacher).sum(dim=-1).mean()
                losses["distill"] = distill
                losses["distill_cos"] = (student_n * teacher).sum(dim=-1).mean().detach()
                total = total + self.lambda_distill * distill

        if targets is not None and "clip_tokens" in targets:
            tok_loss = F.mse_loss(outputs["clip_tokens"], targets["clip_tokens"])
            losses["clip_tokens"] = tok_loss
            total = total + self.lambda_clip * tok_loss

        if targets is not None and "vgg_features" in targets:
            vgg_loss = F.mse_loss(outputs["vgg_features"], targets["vgg_features"])
            losses["vgg"] = vgg_loss
            total = total + self.lambda_vgg * vgg_loss

        if self.lambda_vf > 0 and "virtual_fmri" in outputs:
            vf = outputs["virtual_fmri"]
            if vf.shape[1] >= 2:
                a = F.normalize(vf[:, 0].flatten(1).float(), dim=-1)
                b = F.normalize(vf[:, 1].flatten(1).float(), dim=-1)
                vf_loss = 1.0 - (a * b).sum(dim=-1).mean()
                losses["vf_consist"] = vf_loss
                total = total + self.lambda_vf * vf_loss

        semantic = any(k in losses for k in ("nce", "siglip", "clip", "clip_tokens", "vgg"))
        if self.use_proxy_fallback and not semantic and "clip_tokens" in outputs:
            clip = outputs["clip_tokens"]
            std = clip.reshape(-1, clip.shape[-1]).std(dim=0).mean()
            var_loss = F.relu(1.0 - std)
            losses["variance"] = var_loss
            total = total + var_loss

            img = batch["image"]
            img_proxy = torch.stack(
                [img.mean(dim=(2, 3)), img.std(dim=(2, 3))], dim=1
            ).flatten(1)
            pred_proxy = outputs["clip_tokens"].mean(dim=1)[:, : img_proxy.shape[1]]
            proxy_loss = F.mse_loss(pred_proxy, img_proxy)
            losses["image_proxy"] = proxy_loss
            total = total + 0.1 * proxy_loss

        if self.lambda_reg > 0 and "eeg_kv_tokens" in outputs:
            reg = outputs["eeg_kv_tokens"].pow(2).mean()
            losses["reg"] = reg
            total = total + self.lambda_reg * reg

        losses["total"] = total
        return losses
