"""ATM DiffusionPriorUNet (from ncclab-sustech/EEG_Image_decode Generation/diffusion_prior.py)."""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.schedulers import DDPMScheduler
from tqdm import tqdm


class DiffusionPriorUNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        cond_dim: int = 1024,
        hidden_dim: list[int] | None = None,
        time_embed_dim: int = 512,
        act_fn=nn.SiLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [1024, 512, 256, 128, 64]
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.time_proj = Timesteps(time_embed_dim, True, 0)
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )
        self.num_layers = len(hidden_dim)
        self.encode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(time_embed_dim, hidden_dim[i]) for i in range(self.num_layers - 1)]
        )
        self.encode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers - 1)]
        )
        self.encode_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                    nn.LayerNorm(hidden_dim[i + 1]),
                    act_fn(),
                    nn.Dropout(dropout),
                )
                for i in range(self.num_layers - 1)
            ]
        )
        self.decode_time_embedding = nn.ModuleList(
            [
                TimestepEmbedding(time_embed_dim, hidden_dim[i])
                for i in range(self.num_layers - 1, 0, -1)
            ]
        )
        self.decode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers - 1, 0, -1)]
        )
        self.decode_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i - 1]),
                    nn.LayerNorm(hidden_dim[i - 1]),
                    act_fn(),
                    nn.Dropout(dropout),
                )
                for i in range(self.num_layers - 1, 0, -1)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        t = self.time_proj(t)
        x = self.input_layer(x)
        hidden_activations = []
        for i in range(self.num_layers - 1):
            hidden_activations.append(x)
            t_emb = self.encode_time_embedding[i](t)
            c_emb = self.encode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.encode_layers[i](x)
        for i in range(self.num_layers - 1):
            t_emb = self.decode_time_embedding[i](t)
            c_emb = self.decode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.decode_layers[i](x)
            x = x + hidden_activations[-1 - i]
        return self.output_layer(x)


class AtmDiffusionPriorPipe:
    """DDPM sampling wrapper matching ATM Generation/diffusion_prior.Pipe."""

    def __init__(self, prior: DiffusionPriorUNet, device: torch.device) -> None:
        self.prior = prior.to(device)
        self.scheduler = DDPMScheduler()
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        c_embeds: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self.prior.eval()
        c_embeds = c_embeds.to(self.device)
        n = c_embeds.shape[0]
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        h_t = torch.randn(
            n,
            self.prior.embed_dim,
            generator=generator,
            device=self.device,
            dtype=c_embeds.dtype,
        )
        for t in tqdm(timesteps, desc="prior-sample"):
            t_batch = torch.ones(n, dtype=torch.float, device=self.device) * t
            if guidance_scale == 0:
                noise_pred = self.prior(h_t, t_batch, None)
            else:
                noise_pred_cond = self.prior(h_t, t_batch, c_embeds)
                noise_pred_uncond = self.prior(h_t, t_batch, None)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            h_t = self.scheduler.step(
                noise_pred, int(t.item()) if torch.is_tensor(t) else int(t), h_t, generator=generator
            ).prev_sample
        return h_t


def load_atm_prior(ckpt_path: str, device: torch.device, dropout: float = 0.1) -> AtmDiffusionPriorPipe:
    prior = DiffusionPriorUNet(cond_dim=1024, dropout=dropout)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    prior.load_state_dict(state, strict=True)
    return AtmDiffusionPriorPipe(prior, device)
