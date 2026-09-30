"""CogCapPro backbone, vendored, plus POLARIS's sensor-space subject operator.

Provenance of the vendored parts (all from
`third_party/CognitionCapturerPro/src/cogcappro/models/`):

    PatchEmbedding              brain_backbone.py:14-77
    ChannelConv                 brain_backbone.py:79-84
    EEGAttention                brain_backbone.py:88-103
    PositionalEncoding          brain_backbone.py:129-150
    ResidualAdd                 brain_backbone.py:181-186
    Proj_eeg / ProjMod          brain_backbone.py:115-126, 349-361
    Enc_eeg / FlattenHead       brain_backbone.py:105-108, 245-251
    Cogcap                      brain_backbone.py:156-177
    EEGProjectLayer_..._list    brain_backbone.py:189-208
    CogcapFusion                fusion_backbone.py:45-121

Two changes are made to `Cogcap`, both of them deliberate:

1. **The subject axis is made real.** Upstream is

       self.subject_wise_linear = nn.ModuleList(
           [nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
       ...
       x = self.subject_wise_linear[0](x)          # brain_backbone.py:167,175

   with the author's own comment `# how to deal with this`. It is instantiated with
   `num_subjects=1` (`brain_backbone.py:200`), so index `[0]` is the only element and the
   layer degenerates into an unconditioned `Linear(250, 250)` shared by every subject --
   i.e. CogCapPro has no subject-specific parameterisation at all. Here a subject id is
   threaded through and `-1` (unseen / target subject) selects an explicit identity
   fallback, which also gives "no subject alignment" as a clean ablation row.

2. **SEA is added** (`--sea`): a Cayley-parameterised orthogonal operator on the channel
   axis, i.e. on the raw sensor space rather than on the 1024-d latent. Rationale is in
   the design doc §1.3/§1.5: the subject operator is shared across modality branches, so
   correcting it once at the encoder input is cheaper (C^2 not 4*d^2), needs fewer
   anchors, and keeps every non-equivariant module downstream (fusion, align, IP-Adapter)
   inside its training distribution.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange


# ============================================================ vendored primitives
class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe[: x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1).to(x.device)
        return x + pe


class ChannelConv(nn.Module):
    def __init__(self, channel=None, dropout=0.1):
        super().__init__()
        self.channelconv = nn.Sequential(
            nn.Conv2d(40, 40, (channel, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.channelconv(x)


class PatchEmbedding(nn.Module):
    """brain_backbone.py:14-53, with the channel-attention branch dropped.

    `use_channel_attn` upstream needs 17 hand-written functional regions
    (`brain_backbone.py:29-34`) that do not correspond to anything this project has, and
    the flag is False in every config shipped. Keeping only the `else` path keeps the
    vendored surface honest: no dead code pretending to be a feature.
    """

    def __init__(self, emb_size=40, channel_num=63, dropout=0.1):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (channel_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                     # b, 1, C, T
        x = self.tsconv(x)                     # b, 40, 1, T'
        return self.projection(x)              # b, T', 40


class EEGAttention(nn.Module):
    """brain_backbone.py:88-103. Transformer over *time*, width = channel count."""

    def __init__(self, channel, d_model, nhead):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)             # T, B, C
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src)
        return out.permute(1, 2, 0)            # B, C, T


class FlattenHead(nn.Sequential):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, channel_num=63, dropout=0.1):
        super().__init__(
            PatchEmbedding(emb_size, channel_num=channel_num, dropout=dropout),
            FlattenHead(),
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_out=0.1):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_out),
            )),
            nn.LayerNorm(proj_dim),
        )


class ProjMod(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.1):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


# ============================================================ POLARIS
class SEA(nn.Module):
    """Sensor-space Equivariance Adapter (design doc §3.1).

        x  ->  R_s x + t_s ,      R_s = Cayley(A_s) in SO(C)

    `subject_ids == -1` is the unseen/target subject and maps to the identity, so the
    "no subject correction" configuration is an explicit row rather than an accident.

    Cayley parameterisation is chosen over a free C x C matrix on purpose: the deployment
    recovery solves for an *orthogonal* transform, and that solve is unbiased only to the
    extent the true subject operator is itself orthogonal (design doc condition C3). Training
    inside O(C) removes the stretch component from the hypothesis class instead of hoping
    the recovery can absorb it.
    """

    def __init__(self, n_channels: int = 63, n_subjects: int = 10):
        super().__init__()
        self.C = n_channels
        self.n_subjects = n_subjects
        self.A = nn.Parameter(torch.zeros(n_subjects, n_channels, n_channels))
        self.t = nn.Parameter(torch.zeros(n_subjects, n_channels))

    def _skew(self) -> torch.Tensor:
        A = torch.triu(self.A, diagonal=1)
        return A - A.transpose(-1, -2)

    def rotations(self) -> torch.Tensor:
        A = self._skew()
        I = torch.eye(self.C, device=A.device, dtype=A.dtype)
        return (I - A) @ torch.linalg.inv(I + A)        # [S, C, C], exactly orthogonal

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor | None) -> torch.Tensor:
        if subject_ids is None:
            # Unseen/unnamed subject: SEA is the identity by definition of its contract, not
            # only at initialisation. Every deployment path relies on this branch.
            return x
        B, C, T = x.shape
        sid = subject_ids.clamp(min=0)
        R = self.rotations()[sid]                        # [B, C, C]
        t = self.t[sid]                                  # [B, C]
        out = torch.einsum("bcd,bdt->bct", R, x) + t[..., None]
        unknown = (subject_ids < 0).view(B, 1, 1)
        return torch.where(unknown, x, out)


class SubjectWiseLinear(nn.Module):
    """Per-subject `Linear(T, T)` over the time axis, with an identity fallback.

    Faithful to CogCapPro's intent (`brain_backbone.py:167`) but actually conditioned on
    the subject. Note this mixes *time*, not the latent space the paper's naming suggests
    (`W_s` is described as acting on a simulated latent vector); the shape is what decides
    what it does, so the docstring says time.
    """

    def __init__(self, sequence_length: int = 250, n_subjects: int = 10):
        super().__init__()
        self.lin = nn.ModuleList(
            [nn.Linear(sequence_length, sequence_length) for _ in range(n_subjects)]
        )
        self._zero = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor | None) -> torch.Tensor:
        if subject_ids is None:
            return x
        out = torch.zeros_like(x)
        for s, lin in enumerate(self.lin):
            m = subject_ids == s
            if bool(m.any()):
                out[m] = lin(x[m])
        unknown = (subject_ids < 0) | (subject_ids >= len(self.lin))
        if bool(unknown.any()):
            out[unknown] = x[unknown]                     # identity fallback
        return out


class Cogcap(nn.Module):
    """brain_backbone.py:156-177, with the subject axis wired up and SEA in front."""

    def __init__(self, num_channels=63, sequence_length=250, num_subjects=10,
                 dropout=0.1, subject_wise: str = "time", sea: SEA | None = None):
        super().__init__()
        self.sea = sea
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)
        self.subject_wise = subject_wise
        if subject_wise == "time":
            self.subject_wise_linear = SubjectWiseLinear(sequence_length, num_subjects)
        elif subject_wise in ("none", "identity"):
            self.subject_wise_linear = None
        else:
            raise ValueError(f"unknown subject_wise: {subject_wise}")
        self.enc_eeg = Enc_eeg(channel_num=num_channels, dropout=dropout)
        self.proj_eeg = Proj_eeg(drop_out=dropout)

    def forward(self, x, subject_ids=None):
        if self.sea is not None:
            x = self.sea(x, subject_ids)
        x = self.attention_model(x)
        if self.subject_wise_linear is not None:
            x = self.subject_wise_linear(x, subject_ids)
        return self.proj_eeg(self.enc_eeg(x))


class MultiModalCogcap(nn.Module):
    """brain_backbone.py:189-208. One *independent* Cogcap per modality.

    Upstream feeds the same `x.clone()` to every branch and, when `use_channel_attn` is
    off, the four branches differ only by initialisation -- `PatchEmbedding`
    (`brain_backbone.py:38-47`) takes no modality argument. That is the upstream behaviour
    and it is preserved: the branches are independent parameterisations of the same input,
    not four different views of it. SEA is shared across branches on purpose (design doc
    H1: the subject operator comes from the head/electrodes, not from the branch).
    """

    def __init__(self, z_dim, c_num, timesteps, modality_num=4, drop_proj=0.3,
                 subject_wise: str = "time", n_subjects: int = 10, sea: SEA | None = None,
                 modulation: str = "none"):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.modality_num = modality_num
        self.sea = sea
        # `modulation` is a placeholder for the per-branch modulation coefficients
        # (design doc §2.1). `none` = upstream behaviour: branches share nothing but the
        # input, which is exactly the configuration whose H1 test the recovery arm runs.
        self.modulation = modulation
        self.models = nn.ModuleList([
            Cogcap(num_channels=c_num, sequence_length=timesteps[1], num_subjects=n_subjects,
                   dropout=drop_proj, subject_wise=subject_wise, sea=sea)
            for _ in range(modality_num)
        ])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x, subject_ids=None):
        return [m(x, subject_ids) for m in self.models]


class CogcapFusion(nn.Module):
    """fusion_backbone.py:45-121, verbatim except for the modality count."""

    def __init__(self, modal_dims, hidden_dim=255, num_heads=1, dropout=0.1):
        super().__init__()
        self.modal_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ) for in_dim in modal_dims
        ])
        self.modal_pos_encoder = PositionalEncoding(hidden_dim)
        self.cross_attention = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.cross_attention, num_layers=2)
        self.fusion_proj = nn.Sequential(
            ResidualAdd(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
        )

    def forward(self, *modal_features):
        projected = [proj(f) for proj, f in zip(self.modal_projs, modal_features)]
        modal_stack = torch.stack(projected, dim=1)                       # B, M, H
        modal_stack = self.modal_pos_encoder(modal_stack.permute(1, 0, 2)).permute(1, 0, 2)
        attn = self.transformer_encoder(modal_stack)
        return self.fusion_proj(attn.mean(dim=1))


class CogCapPro(nn.Module):
    """The full forward, mirroring `training/module.py:196-242`.

    Divergences from upstream, all forced by running LOSO rather than intra-subject:

    * `subject_ids` is threaded through (upstream has no subject input at all).
    * the random non-EEG modality masking is a training-time augmentation and lives in the
      training step, not here, so that evaluation is deterministic.
    * `fusion_eeg` no longer detaches its inputs (upstream `module.py:213` does). Detaching
      means the fusion branch cannot shape the encoders, which makes the fusion arm useless
      as a cross-subject device (design doc §2.6). Kept behind `--fusion-detach` so the
      upstream behaviour is still runnable as a row.
    """

    def __init__(self, modalities, c_num=63, timesteps=(0, 250), drop_proj=0.3,
                 subject_wise="time", n_subjects=10, use_sea=True, fusion=True,
                 fusion_hidden=255, fusion_detach=False):
        super().__init__()
        self.modalities = list(modalities)
        self.n_modalities = len(self.modalities)
        self.fusion_detach = fusion_detach
        self.sea = SEA(c_num, n_subjects) if use_sea else None
        self.brain = MultiModalCogcap(
            z_dim=1024, c_num=c_num, timesteps=timesteps, modality_num=self.n_modalities,
            drop_proj=drop_proj, subject_wise=subject_wise, n_subjects=n_subjects,
            sea=self.sea,
        )
        self.modproj = nn.ModuleList([ProjMod() for _ in range(self.n_modalities)])
        self.use_fusion = fusion
        if fusion:
            self.fusion_eeg = CogcapFusion([1024] * self.n_modalities, hidden_dim=fusion_hidden)
            self.fusion_mod = CogcapFusion([1024] * self.n_modalities, hidden_dim=fusion_hidden)

    def forward(self, eeg, modality_features, subject_ids=None, mask_modalities=None):
        """Returns {'z': {modality: [B,1024]}, 'logit_scale': scalar}."""
        mods = self.modalities
        eeg_z = self.brain(eeg, subject_ids)                       # list of [B, 1024]

        mod_z = {}
        for name, proj, feat in zip(mods, self.modproj, modality_features):
            mod_z[name] = proj(feat)

        out = {name: z for name, z in zip(mods, eeg_z)}
        if self.use_fusion:
            fin = [e.detach() for e in eeg_z] if self.fusion_detach else list(eeg_z)
            out["fusion"] = self.fusion_eeg(*fin)
            mod_in = [mod_z[m] for m in mods]
            if mask_modalities is not None:
                mod_in = [
                    torch.zeros_like(z) if i in mask_modalities else z
                    for i, z in enumerate(mod_in)
                ]
            mod_z["fusion"] = self.fusion_mod(*mod_in)

        return {
            "z": out,
            "mod_z": mod_z,
            "logit_scale": self.brain.softplus(self.brain.logit_scale),
        }

    def trainable_parameter_groups(self, backbone_lr_mult: float = 1.0):
        sea_ids = {id(p) for p in (self.sea.parameters() if self.sea is not None else [])}
        sea = [p for p in self.parameters() if id(p) in sea_ids and p.requires_grad]
        rest = [p for p in self.parameters() if id(p) not in sea_ids and p.requires_grad]
        return [{"params": rest, "lr_mult": backbone_lr_mult}, {"params": sea, "lr_mult": 1.0}]
