"""Stage 3 training engine -- corrected to match Stage-2 data/checkpoint contracts."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loso import paths
from loso.data import eeg as eeg_mod
from loso.data import things
from loso.data.targets import TargetStore
from loso.models.condition import (
    ConditionConfig, EEGImageProj, attach_lora, build_sdxl_components,
    encode_prompts, inject_ip_tokens, resolve_ip_adapter_file, sdxl_time_ids,
)
from loso.models.eeg_encoder import EEGEncoder, EncoderConfig
from loso.models.heads import AlignmentHeads, HeadConfig


@dataclass
class DiffusionConfig:
    test_subject: str = "sub-08"
    align_ckpt: str = ""
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"
    num_workers: int = 8
    log_every: int = 50
    save_every: int = 2
    seed: int = 0
    max_steps_per_epoch: int = 0
    prompt: str = ""
    snr_gamma: float = 5.0
    freeze_encoder: bool = True
    condition: ConditionConfig = field(default_factory=ConditionConfig)


def _amp(dtype_name: str, device: torch.device):
    if device.type != "cuda" or dtype_name == "none":
        return torch.float32, False
    return (torch.bfloat16 if dtype_name == "bf16" else torch.float16), True


def min_snr_weights(timesteps, alphas_cumprod, gamma: float):
    a = alphas_cumprod.to(timesteps.device)[timesteps].clamp_min(1e-6)
    snr = a / (1.0 - a).clamp_min(1e-6)
    return snr.clamp_max(gamma) / snr.clamp_min(1e-6)


def load_align_checkpoint(ckpt: Path, device: str):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "cfg_enc" in payload:
        enc_cfg = EncoderConfig(**payload["cfg_enc"])
        head_cfg = HeadConfig(**payload["cfg_head"])
    elif "cfg" in payload and isinstance(payload["cfg"], dict):
        c = payload["cfg"]
        enc_cfg = EncoderConfig(**{k: v for k, v in c.get("enc", {}).items()
                                   if k in EncoderConfig.__dataclass_fields__})
        head_cfg = HeadConfig(**{k: v for k, v in c.get("head", {}).items()
                                 if k in HeadConfig.__dataclass_fields__})
    else:
        raise KeyError(f"{ckpt} has no cfg_enc/cfg_head (or nested cfg)")
    model = EEGEncoder(enc_cfg)
    heads = AlignmentHeads(head_cfg)
    model.load_state_dict(payload["encoder"], strict=True)
    if "heads" in payload:
        heads.load_state_dict(payload["heads"], strict=True)
    model.to(device).eval()
    heads.to(device).eval()
    return model, heads, payload


def build_condition(cfg: DiffusionConfig, device: str) -> EEGImageProj:
    cond = EEGImageProj(cfg.condition).to(device)
    ip_path = resolve_ip_adapter_file()
    state = torch.load(ip_path, map_location="cpu", weights_only=True)
    n = cond.load_ip_adapter_proj(state)
    print(f"[ip-adapter] loaded {n} tensors from {ip_path.name}", flush=True)
    for p in list(cond.image_proj.parameters()) + list(cond.norm.parameters()):
        p.requires_grad_(False)
    return cond


def train_step(encoder, cond, components, batch, cfg, empty_prompt, empty_pooled,
               dtype, amp_on):
    device = batch["x"].device
    unet = components["unet"]
    scheduler = components["scheduler"]
    b = batch["x"].shape[0]

    with torch.no_grad():
        z_inv = encoder(batch["x"], batch["subject_id"], adapter_mode="subject")["z_inv"]
        latents = batch["vae_latent"].to(device=device, dtype=dtype)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (b,),
            device=device, dtype=torch.long,
        )
        noisy = scheduler.add_noise(latents.float(), noise.float(), timesteps).to(dtype)

    with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_on):
        _, ip_tokens = cond(z_inv)
        prompt = empty_prompt.expand(b, -1, -1)
        pooled = empty_pooled.expand(b, -1)
        encoder_hs = inject_ip_tokens(prompt, ip_tokens)
        add_time = sdxl_time_ids(b, cfg.condition.resolution, device, dtype)
        pred = unet(
            noisy, timesteps, encoder_hs,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time},
        ).sample
        per = F.mse_loss(pred.float(), noise.float(), reduction="none").mean(dim=(1, 2, 3))
        if cfg.snr_gamma > 0:
            w = min_snr_weights(timesteps, scheduler.alphas_cumprod, cfg.snr_gamma)
            loss = (per * w).mean()
        else:
            loss = per.mean()
    return {"loss": loss, "mse": per.detach().mean()}


def run(cfg: DiffusionConfig) -> Path:
    paths.ensure_dirs()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype, amp_on = _amp(cfg.amp_dtype, device)

    if not cfg.align_ckpt:
        raise ValueError("--align-ckpt is required")
    encoder, _heads, _meta = load_align_checkpoint(Path(cfg.align_ckpt), str(device))
    if cfg.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
    cfg.condition.d_inv = int(encoder.cfg.d_inv)

    print("[sdxl] loading frozen backbone...", flush=True)
    components = build_sdxl_components(str(device), dtype)
    if cfg.condition.use_lora:
        components["unet"] = attach_lora(
            components["unet"], cfg.condition.lora_rank, cfg.condition.lora_alpha,
        )
        components["unet"].train()
        print(f"[lora] r={cfg.condition.lora_rank} a={cfg.condition.lora_alpha}", flush=True)
    else:
        components["unet"].eval()

    cond = build_condition(cfg, str(device))
    cond.train()

    empty_prompt, empty_pooled = encode_prompts(
        components, [cfg.prompt], str(device), dtype,
    )

    train_subjects, test_subject = things.loso_split(cfg.test_subject)
    n_subjects = len(train_subjects)
    print(f"[diff] test={test_subject} train={train_subjects}", flush=True)

    per_subject, global_stats = eeg_mod.build_normalizers(
        train_subjects, "train_subjects",
        cache_path=paths.DATA_ROOT / f"norm_train_subjects_{n_subjects}.json",
    )
    norm = {s: global_stats for s in train_subjects}
    norm_keys = {s: s for s in train_subjects}
    ds = eeg_mod.TrainEEGDataset(
        train_subjects, norm, norm_keys,
        augment_cfg=eeg_mod.AugmentConfig(), seed=cfg.seed,
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, collate_fn=eeg_mod.collate,
        pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )
    store = TargetStore("train", names=("vae_latent",))

    params = [p for p in cond.parameters() if p.requires_grad]
    if cfg.condition.use_lora:
        params += [p for p in components["unet"].parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_ep = len(loader) if not cfg.max_steps_per_epoch else cfg.max_steps_per_epoch
    total_steps = cfg.epochs * max(1, steps_per_ep)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    out_dir = paths.DIFFUSION_DIR / cfg.test_subject
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1, default=str))

    global_step = 0
    best_loss = float("inf")
    best_path = out_dir / "best.pt"
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        ds.set_epoch(epoch)
        running, n_steps = 0.0, 0
        for step, batch in enumerate(loader, 1):
            if cfg.max_steps_per_epoch and step > cfg.max_steps_per_epoch:
                break
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            batch["vae_latent"] = store.gather(batch["target_slot"], device=device)["vae_latent"]

            for pg in opt.param_groups:
                pg["lr"] = cfg.lr * lr_at(global_step)
            opt.zero_grad(set_to_none=True)
            terms = train_step(
                encoder, cond, components, batch, cfg,
                empty_prompt, empty_pooled, dtype, amp_on,
            )
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            running += float(terms["loss"])
            n_steps += 1
            global_step += 1
            if global_step % cfg.log_every == 0:
                print(f"[ep {epoch}/{cfg.epochs} step {global_step}] "
                      f"loss={float(terms['loss']):.4f} mse={float(terms['mse']):.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)

        mean_loss = running / max(1, n_steps)
        print(f"[ep {epoch}] mean_loss={mean_loss:.4f} "
              f"elapsed={(time.time()-t0)/60:.1f} min", flush=True)

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            ckpt = {
                "epoch": epoch, "mean_loss": mean_loss,
                "condition": cond.state_dict(),
                "condition_cfg": asdict(cfg.condition),
                "align_ckpt": cfg.align_ckpt,
                "cfg": asdict(cfg),
            }
            if cfg.condition.use_lora:
                ckpt["lora"] = {
                    k: v.detach().cpu()
                    for k, v in components["unet"].state_dict().items()
                    if "lora_" in k
                }
            torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pt")
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(ckpt, best_path)
                print(f"[ckpt] best {best_loss:.4f} -> {best_path}", flush=True)

    print(f"[done] Stage-3 -> {out_dir}", flush=True)
    return best_path
