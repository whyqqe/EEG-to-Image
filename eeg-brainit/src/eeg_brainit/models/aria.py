"""ARIA: Anchor-Relative Inter-brain Alignment for EEG→CLIP.

Core idea
---------
Cross-subject failure is largely an *absolute coordinate* problem.
ARIA encodes EEG into a working space, aligns *pairwise geometry* to CLIP
(RSA / second-order), and retrieves in *anchor-relative* coordinates that
are invariant to orthogonal/scaling drifts across subjects.

Components
----------
1) Shared ATM-style backbone + projector (no per-subject W_s in the main path)
2) Relative map r(z) = normalize(cos(z, Anchors))
3) Losses: L_rsa + L_rel [+ weak centered absolute] + identity GRL
4) Inference: relative retrieval; optional orthogonal Procrustes N-shot
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_brainit.models.subject_align import AtmSharedBackbone, MLPProjector


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def relative_from_anchors(z: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """z: (B, D), anchors: (K, D) → r: (B, K) L2-normalized cosine profile."""
    z_n = l2_normalize(z)
    a_n = l2_normalize(anchors)
    return l2_normalize(z_n @ a_n.T)


def pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    x = l2_normalize(x)
    return x @ x.T


def rsa_loss(eeg_emb: torch.Tensor, clip_emb: torch.Tensor) -> torch.Tensor:
    """Soft second-order alignment: MSE between centered pairwise cosine matrices."""
    s_e = pairwise_cosine(eeg_emb)
    s_c = pairwise_cosine(clip_emb.float())
    # ignore diagonal
    b = s_e.size(0)
    if b < 3:
        return s_e.new_zeros(())
    eye = torch.eye(b, device=s_e.device, dtype=torch.bool)
    se = s_e.masked_select(~eye)
    sc = s_c.masked_select(~eye)
    se = se - se.mean()
    sc = sc - sc.mean()
    # 1 - Pearson correlation
    num = (se * sc).sum()
    den = se.norm() * sc.norm() + 1e-8
    return 1.0 - num / den


def info_nce(q: torch.Tensor, k: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    logits = l2_normalize(q) @ l2_normalize(k).T / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class ARIAEncoder(nn.Module):
    """Shared EEG encoder → CLIP working space (absolute) + relative profiles."""

    def __init__(
        self,
        n_subjects: int = 10,
        n_channels: int = 63,
        seq_len: int = 250,
        clip_dim: int = 1024,
        nz: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.backbone = AtmSharedBackbone(
            n_channels=n_channels, seq_len=seq_len, nz=nz, clip_dim=clip_dim, dropout=dropout
        )
        self.projector = MLPProjector(nz, clip_dim=clip_dim, dropout=dropout)
        self.probe_id = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, clip_dim // 2),
            nn.GELU(),
            nn.Linear(clip_dim // 2, n_subjects),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.clip_dim = clip_dim
        self.nz = nz
        self.n_subjects = n_subjects
        # anchors registered later (K, D)
        self.register_buffer("anchors", torch.zeros(1, clip_dim), persistent=True)
        self.register_buffer("anchors_ready", torch.zeros((), dtype=torch.bool))

    def set_anchors(self, anchors: torch.Tensor) -> None:
        anchors = l2_normalize(anchors.float().detach())
        self.anchors = anchors
        self.anchors_ready.fill_(True)

    def encode_absolute(self, x: torch.Tensor, normalize: bool = True) -> dict[str, torch.Tensor]:
        z = self.backbone(x.float())
        raw = self.projector(z)
        emb = l2_normalize(raw) if normalize else raw
        return {"latent": z, "clip_raw": raw, "clip_emb": l2_normalize(raw), "clip_out": emb}

    def encode_relative(self, abs_emb: torch.Tensor) -> torch.Tensor:
        if not bool(self.anchors_ready):
            raise RuntimeError("ARIA anchors not set")
        return relative_from_anchors(abs_emb, self.anchors)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encode_absolute(x, normalize=True)
        if bool(self.anchors_ready):
            out["rel_emb"] = self.encode_relative(out["clip_emb"])
        out["logit_scale"] = self.logit_scale.exp()
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def subject_center(emb: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
    """Remove per-subject mean within the batch (weak absolute alignment helper)."""
    out = emb.clone()
    for s in subject_id.unique():
        m = subject_id == s
        out[m] = emb[m] - emb[m].mean(0, keepdim=True)
    return out


def aria_loss(
    model: ARIAEncoder,
    out: dict[str, torch.Tensor],
    clip_img: torch.Tensor,
    subject_id: torch.Tensor,
    *,
    temp: float = 0.07,
    lambda_rsa: float = 1.0,
    lambda_rel: float = 0.5,
    lambda_abs: float = 0.1,
    lambda_nce: float = 0.5,
    lambda_id: float = 0.1,
    grl_lambda: float = 1.0,
    atm_emb: torch.Tensor | None = None,
    lambda_atm: float = 0.0,
    abs_subject_center: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-term ARIA loss.

    Abs-only qualification (recommended first):
      lambda_rsa=lambda_rel=lambda_id=0, lambda_abs=1, lambda_nce≈0.5, lambda_atm≈0.3
      uses Align-style MSE + InfoNCE on absolute embeddings (not relative).
    """
    clip_img = clip_img.float()
    eeg_abs = out["clip_emb"]
    clip_n = l2_normalize(clip_img)

    stats: dict[str, float] = {}
    loss = eeg_abs.new_zeros(())

    if lambda_rsa > 0:
        lr = rsa_loss(eeg_abs, clip_n)
        loss = loss + lambda_rsa * lr
        stats["rsa"] = float(lr)

    if lambda_rel > 0 and "rel_emb" in out:
        rel_clip = relative_from_anchors(clip_n, model.anchors)
        lrel = info_nce(out["rel_emb"], rel_clip, temp=temp)
        loss = loss + lambda_rel * lrel
        stats["rel"] = float(lrel)

    if lambda_abs > 0:
        # Align-style absolute head (primary for qualification)
        lmse = F.mse_loss(out["clip_raw"], clip_img)
        q = subject_center(eeg_abs, subject_id) if abs_subject_center else eeg_abs
        k = subject_center(clip_n, subject_id) if abs_subject_center else clip_n
        lnce = info_nce(q, k, temp=temp)
        labs = lmse + lambda_nce * lnce
        loss = loss + lambda_abs * labs
        stats["abs_mse"] = float(lmse.detach())
        stats["abs_nce"] = float(lnce.detach())
        stats["abs"] = float(labs.detach())

    if lambda_atm > 0 and atm_emb is not None:
        latm = F.mse_loss(eeg_abs, l2_normalize(atm_emb.float()))
        loss = loss + lambda_atm * latm
        stats["atm"] = float(latm.detach())

    if lambda_id > 0:
        logits = model.probe_id(grad_reverse(eeg_abs, grl_lambda))
        lid = F.cross_entropy(logits, subject_id.long())
        loss = loss + lambda_id * lid
        stats["id"] = float(lid.detach())

    stats["loss"] = float(loss.detach())
    return loss, stats

@torch.no_grad()
def fit_orthogonal_procrustes(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Return Q (D,D) minimizing ||src @ Q - tgt||_F with Q orthogonal."""
    src = src.astype(np.float64)
    tgt = tgt.astype(np.float64)
    # center
    src_c = src - src.mean(0, keepdims=True)
    tgt_c = tgt - tgt.mean(0, keepdims=True)
    m = src_c.T @ tgt_c
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    q = u @ vt
    # enforce det(Q)=+1 (rotation)
    if np.linalg.det(q) < 0:
        u = u.copy()
        u[:, -1] *= -1
        q = u @ vt
    return q.astype(np.float32)


@torch.no_grad()
def apply_procrustes(x: np.ndarray, q: np.ndarray, src_mean: np.ndarray, tgt_mean: np.ndarray) -> np.ndarray:
    return ((x - src_mean) @ q) + tgt_mean


def build_class_anchors(clip_train: np.ndarray, n_classes: int = 1654, reps: int = 10) -> np.ndarray:
    """Average 10 reps → class prototypes (n_classes, D)."""
    x = clip_train.reshape(n_classes, reps, -1).mean(1)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return x.astype(np.float32)


def subsample_anchors(class_anchors: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    if k <= 0 or k >= class_anchors.shape[0]:
        return class_anchors
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(class_anchors.shape[0], size=k, replace=False))
    return class_anchors[idx]
