#!/usr/bin/env python3
"""SAMGA-R: SAMGA's retrieval-aligned EEG encoder, extended with a generation head.

THE DESIGN, AND WHY IT IS SHAPED THIS WAY
-----------------------------------------
SAMGA is an *inter-subject retrieval* method: its EEG branch is
`TSConv -> eeg_projector -> share_encoder`, ending in a 512-d embedding trained by
bidirectional InfoNCE (+ an MMD warm-up stage) against a router-mixed InternViT teacher.
Applied with a label-free geometric recovery step it reaches 53.23% Top-1 LOSO, against
11.84% for ATM's cross-subject encoder. That is the strongest inter-subject EEG
representation anyone has published on THINGS-EEG2.

It has never been used to *generate* anything, for one structural reason: its 512-d space
is learned, not CLIP's, and a pretrained image decoder only accepts CLIP's. So this module
adds the missing edge and nothing else:

    EEG (63,250)
      -> TSConv                       x        (1024-d)   <- shared, retrieval-trained
           |-- eeg_projector(1024->512) -> share_encoder(512->512) -> z (512-d)  [retrieval,
           |                                                                     untouched]
           `-- gen_head  (1024->1024)   -> c        (1024-d)  [generation, NEW]
                                              -> IP-Adapter -> SDXL-Turbo -> image

Three properties of this wiring are deliberate:

1. **The head reads the 1024-d encoder output, not the 512-d retrieval embedding.**
   `eeg_projector` and `share_encoder` are both single `nn.Linear` layers with no
   activation between them (`third_party/SAMGA/module/projector.py`: `ProjectorLinear`
   is `nn.Linear`; `ShareEncoder` is `nn.Sequential(nn.Linear)`). Their composition is
   therefore itself one linear map, so z = W x with W a learned 1024x512 matrix. The
   1024-d `x` is a strictly richer, linearly-equivalent view of the same representation,
   and reading it there avoids paying a rank-512 bottleneck for a 1024-d target.

2. **The retrieval path is not modified.** The head is a sibling branch. Retrieval
   behaviour is bit-identical to SAMGA's, so the retrieval numbers and the generation
   numbers are attributable to the same frozen encoder rather than to a re-tuned one.

3. **Nothing about the target subject is used.** The head is trained only on the source
   subjects' trials. At deployment the held-out subject's EEG goes through the frozen
   encoder and the frozen head. This is the LOSO contract, and it is the whole point:
   eeg-brainit's existing reconstruction numbers are *per-subject* (each subject has its
   own ATM encoder and its own diffusion prior), whereas this is one encoder and one head
   trained on nine subjects and deployed on the tenth.

RELATION TO RECONSTRUCTION SOTA
-------------------------------
The conditioning side follows the ENIGMA/ATM recipe exactly: predict the OpenCLIP
ViT-H-14 1024-d projected image embedding and hand it to SDXL-Turbo through IP-Adapter.
`extract_clip_h14.py` documents why that array, and why 1024 rather than 1280. What is new
here is only where the prediction comes from: a retrieval-aligned inter-subject encoder
instead of a per-subject one trained from scratch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
SAMGA_ROOT = REPO / "third_party" / "SAMGA"

EGG_FEATURE_DIM = 1024   # TSConv output; also OpenCLIP ViT-H-14 projection_dim
RETRIEVAL_DIM = 512      # SAMGA --feature_dim; also the InfoNCE space
CHANNELS = 63
SAMPLE_POINTS = 250


def _samga_imports():
    """Import SAMGA's own building blocks, so we cannot drift from the official model."""
    if str(SAMGA_ROOT) not in sys.path:
        sys.path.insert(0, str(SAMGA_ROOT))
    from module.eeg_encoder.model import TSConv          # noqa: E402
    from module.projector import ProjectorLinear, ShareEncoder  # noqa: E402
    return TSConv, ProjectorLinear, ShareEncoder


class GenHead(nn.Module):
    """x (1024) -> OpenCLIP ViT-H-14 image_embeds (1024), L2-normalized.

    A residual MLP rather than a bare `Linear(1024, 1024)`. The residual form starts as
    (near-)identity, which matters because the encoder output is already a retrieval-
    aligned representation -- a plain linear map would be free to rotate it before any
    gradient signal arrives, while the residual path lets the head begin as a no-op and
    only learn the correction. Dropout is on the hidden activations only; no LayerNorm on
    the output because the final L2 normalization already fixes the scale.
    """

    def __init__(self, dim_in: int = EGG_FEATURE_DIM, dim_out: int = EGG_FEATURE_DIM,
                 hidden: int = 2048, dropout: float = 0.1, residual: bool = True,
                 standardize: bool = True):
        super().__init__()
        self.residual = residual and dim_in == dim_out
        # Per-dimension input standardisation, with the statistics carried as buffers so
        # the head is self-contained: whatever it was trained on is what it will apply at
        # deployment, with no separate artifact to forget to ship. This is not cosmetic --
        # EEG-derived features typically span several orders of magnitude across
        # dimensions, and an un-normalised MLP spends its capacity re-learning that scale.
        self.standardize = standardize
        if standardize:
            self.register_buffer("in_mean", torch.zeros(dim_in))
            self.register_buffer("in_std", torch.ones(dim_in))
        self.fc1 = nn.Linear(dim_in, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim_out)
        if self.residual:
            # Small init on the second layer so the branch starts close to zero and the
            # head begins as an identity map on the (already aligned) encoder output.
            nn.init.normal_(self.fc2.weight, std=1e-3)
            nn.init.zeros_(self.fc2.bias)
            self.skip = nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out)
        else:
            self.skip = None

    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Freeze the input standardisation from a source-only pass over the features."""
        if not self.standardize:
            return
        with torch.no_grad():
            self.in_mean.copy_(mean.detach().float().reshape(-1))
            self.in_std.copy_(std.detach().float().reshape(-1).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.standardize:
            x = (x - self.in_mean) / self.in_std
        h = self.fc2(self.drop(self.act(self.fc1(x))))
        out = h + self.skip(x) if self.residual else h
        return F.normalize(out, dim=-1)


class SamgaReconModel(nn.Module):
    """Frozen SAMGA EEG encoder + trainable generation head.

    `forward` returns all three views so a single pass can serve retrieval evaluation
    (`retrieval`), the generation path (`clip_cond`), and diagnostics (`hidden`).
    """

    def __init__(self, ckpt: Path | str, gen_head: nn.Module | None = None,
                 device: torch.device | str = "cpu", head_hidden: int = 2048,
                 head_dropout: float = 0.1, strict: bool = True,
                 verbose: bool = True):
        super().__init__()
        TSConv, ProjectorLinear, ShareEncoder = _samga_imports()

        self.model = TSConv(feature_dim=EGG_FEATURE_DIM,
                            eeg_sample_points=SAMPLE_POINTS, channels_num=CHANNELS)
        self.eeg_projector = ProjectorLinear(EGG_FEATURE_DIM, RETRIEVAL_DIM)
        self.share_encoder = ShareEncoder(RETRIEVAL_DIM, RETRIEVAL_DIM)

        ckpt = Path(ckpt)
        if not ckpt.is_file():
            raise SystemExit(f"[FATAL] SAMGA checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)

        # Load only the three keys the EEG branch needs. The image-side keys
        # (img_pre_projector / img_projectors / layer_router) are irrelevant here and are
        # reported rather than silently ignored, so a checkpoint from a run configured
        # with a different --feature_dim or encoder fails loudly instead of half-loading.
        want = {
            "model": (self.model, state.get("model_state_dict")),
            "eeg_projector": (self.eeg_projector, state.get("eeg_projector_state_dict")),
            "share_encoder": (self.share_encoder, state.get("share_enc_state_dict")),
        }
        for name, (module, sd) in want.items():
            if sd is None:
                raise SystemExit(
                    f"[FATAL] checkpoint {ckpt} has no '{name}' state dict. Keys present: "
                    f"{sorted(k for k in state if k.endswith('_state_dict'))}"
                )
            try:
                module.load_state_dict(sd, strict=True)
            except RuntimeError as exc:
                raise SystemExit(
                    f"[FATAL] '{name}' failed to load strictly from {ckpt}: {exc}\n"
                    f"        This normally means the checkpoint used a different "
                    f"--feature_dim or --eeg_encoder_type than this module assumes "
                    f"({RETRIEVAL_DIM} / TSConv / {EGG_FEATURE_DIM})."
                ) from exc

        self.gen_head = gen_head if gen_head is not None else GenHead(
            hidden=head_hidden, dropout=head_dropout)
        self._load_report = {
            "ckpt": str(ckpt),
            "ckpt_epoch": state.get("epoch"),
            "ignored_keys": sorted(k for k in state
                                   if k.endswith("_state_dict")
                                   and k not in ("model_state_dict",
                                                 "eeg_projector_state_dict",
                                                 "share_enc_state_dict")),
        }
        self.to(device)
        if strict:
            self.freeze_encoder()
        if verbose:
            print(f"[INFO] SamgaReconModel loaded {ckpt}")
            print(f"[INFO]   epoch={self._load_report['ckpt_epoch']} "
                  f"ignored={self._load_report['ignored_keys']}")

    # ---------------------------------------------------------------- encoder control
    def freeze_encoder(self) -> None:
        for p in list(self.model.parameters()) + list(self.eeg_projector.parameters()) \
                + list(self.share_encoder.parameters()):
            p.requires_grad_(False)
        self.model.eval()

    def unfreeze_encoder(self) -> None:
        for p in list(self.model.parameters()) + list(self.eeg_projector.parameters()) \
                + list(self.share_encoder.parameters()):
            p.requires_grad_(True)

    # ------------------------------------------------------------------------ forward
    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        """(B,63,250) -> (B,1024). Half of the forward pass, and the reusable part."""
        return self.model(eeg.float())

    def retrieval_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """(B,1024) -> (B,512). SAMGA's own embedding, unmodified."""
        return self.share_encoder(self.eeg_projector(x))

    def forward(self, eeg: torch.Tensor, need_retrieval: bool = True) -> dict:
        x = self.encode(eeg)
        out = {"hidden": x, "clip_cond": self.gen_head(x)}
        if need_retrieval:
            out["retrieval"] = self.retrieval_embedding(x)
        return out
