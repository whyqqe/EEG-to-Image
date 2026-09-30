"""Load NeuroBridge EEG encoder (EEGProject or ATMS) + projector from train checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from module.eeg_encoder.atm.atm import ATMS
from module.eeg_encoder.model import EEGProject
from module.projector import ProjectorDirect, ProjectorLinear, ProjectorMLP


def build_encoder(
    encoder_type: str,
    image_feature_dim: int,
    eeg_sample_points: int,
    channels_num: int,
    projector_kind: str,
    proj_out_dim: int,
    num_subjects: int = 10,
) -> tuple[nn.Module, nn.Module, int]:
    """Return (encoder, eeg_projector, encoder_out_dim)."""
    if encoder_type == "atm":
        model = ATMS(
            channels_num=channels_num,
            feature_dim=image_feature_dim,
            eeg_sample_points=eeg_sample_points,
            num_subjects=num_subjects,
        )
        out_dim = image_feature_dim
    else:
        model = EEGProject(
            feature_dim=image_feature_dim,
            eeg_sample_points=eeg_sample_points,
            channels_num=channels_num,
        )
        out_dim = image_feature_dim

    if projector_kind == "direct":
        eeg_projector = ProjectorDirect()
        proj_dim = out_dim
    elif projector_kind == "linear":
        eeg_projector = ProjectorLinear(out_dim, proj_out_dim)
        proj_dim = proj_out_dim
    else:
        eeg_projector = ProjectorMLP(out_dim, proj_out_dim)
        proj_dim = proj_out_dim
    return model, eeg_projector, out_dim


def load_train_checkpoint(
    ckpt_path: Path,
    encoder_type: str,
    image_feature_dim: int,
    eeg_sample_points: int,
    channels_num: int,
    proj_out_dim: int = 512,
    device: torch.device | None = None,
    num_subjects: int = 10,
) -> tuple[nn.Module, nn.Module, dict]:
    device = device or torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt.get("encoder_type", "eegproject")
    enc_type = encoder_type or meta

    # Infer projector from state dict shapes when possible
    eeg_proj_sd = ckpt.get("eeg_projector_state_dict", {})
    if eeg_proj_sd and "linear.weight" in eeg_proj_sd:
        proj_kind = "linear"
        proj_out_dim = int(eeg_proj_sd["linear.weight"].shape[0])
    elif eeg_proj_sd and "mlp.0.weight" in eeg_proj_sd:
        proj_kind = "mlp"
        proj_out_dim = int(eeg_proj_sd["mlp.2.weight"].shape[0])
    else:
        proj_kind = "direct"
        proj_out_dim = image_feature_dim

    model, eeg_projector, _ = build_encoder(
        enc_type,
        image_feature_dim,
        eeg_sample_points,
        channels_num,
        proj_kind,
        proj_out_dim,
        num_subjects=num_subjects,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if eeg_proj_sd:
        eeg_projector.load_state_dict(eeg_proj_sd, strict=True)

    model.to(device).eval()
    eeg_projector.to(device).eval()
    return model, eeg_projector, {"encoder_type": enc_type, "proj_out_dim": proj_out_dim}


def encode_eeg(
    model: nn.Module,
    eeg_projector: nn.Module,
    eeg: torch.Tensor,
    encoder_type: str,
    subject_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if encoder_type == "atm":
        if subject_ids is None:
            subject_ids = torch.zeros(eeg.shape[0], dtype=torch.long, device=eeg.device)
        raw = model(eeg, subject_ids)
    else:
        raw = model(eeg)
    return eeg_projector(raw)
