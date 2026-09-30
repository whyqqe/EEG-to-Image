# 实验时间线 W7–W14

## 目标演进

- **初始**：PixCorr ≥ 0.15，同时保持 CLIP 优势（相对官方 ATM ~0.37）。
- **中期发现**：SDXL-base 路线 Pix 天花板 ~0.133；换官方 Turbo 栈后 Pix 跳至 0.15+。
- **收尾**：merge bank + fuse + 10 被试 + 2WC + 机制对照。

---

## 阶段表

| 阶段 | Slurm / Job | 状态 | 要点 |
|------|-------------|------|------|
| W7 | closed loop | ✅ | IP-Adapter + loop 基线 |
| W8 | erdc_w8_ras | ✅ | RAS bit lowstr, Pix~0.117 |
| W10 | erdc_w10, 527209 | ✅ | fuse λ=0.35, **CLIP=0.449**, Pix=0.133 |
| W11 | 527445 | ❌ 取消 | SDXL-only 扫参 ROI 低 |
| S4 微调 | 527446 | ✅ | Top-1 仍 0.365，Pix 几乎无增益 |
| W12 | 527477→527507 | ✅ | 官方 Turbo 栈；**atm brain Pix=0.173** |
| W13 | 527543 | ✅ | merge 3 bank；**fuse λ=0.15: Pix=0.159, CLIP=0.418** |
| W14 | 527571 | ✅ | 2WC + 机制 + 10 被试 + freeze |

---

## W12 关键修复

- **Bug**：`load_train_latents()` 需从 dict 取键 `image_latent`，否则 prior_latent 路线 TypeError。
- **资产**：`scripts/download_atm_official_assets.py` → Turbo + `train_image_latent_512.pt`
- **路径**：`checkpoints/_hf_atm_ds/train_image_latent_512.pt` (16540,4,64,64)

---

## W12 最终结果（sub-08）

| 方法 | Pix | CLIP |
|------|-----|------|
| 官方 ATM | 0.159 | 0.371 |
| w12_fuse_bit λ=0.25 | 0.150 | 0.417 |
| w12_official_atm_brain | **0.173** | 0.379 |
| w12_official_prior_latent_brain | 0.165 | 0.383 |

Freeze：`results/w12_official_main_freeze.txt`

---

## W13 merge fuse

合并：`w12_official_bit_img` + `w12_official_atm_img` + `w12_official_prior_latent` → **k=20**

| λ | Pix | CLIP |
|---|-----|------|
| 0.15 | **0.159** | **0.418** |
| 0.25 | 0.153 | 0.421 |
| brain only | **0.162** | 0.406 |

Freeze：`results/w13_merge_fuse_freeze.txt`

---

## W14 收尾（Job 527571）

### 已完成

- ✅ fuse 机制对照 shuffle/misalign/zero/retrieve
- ✅ sub-08 全主线的 2WC
- ✅ 10 被试 bit_clip embed 导出
- ✅ 10 被试 Turbo bit fuse + atm brain
- ✅ `w14_paper_finalize_freeze.txt`

### W14 重要发现

1. **sub-08 主方法仍成立**：merged fuse Pix≈官方，CLIP +0.048，2WC 84.4%
2. **10 被试泛化不足**：fuse Pix_mean=**0.117**（0/10 超官方 Pix）；因 **全被试共用 sub-08 S3 ckpt**，非 LOSO
3. **机制对照弱**：shuffle ≈ retrieve，结构证据叙事需谨慎
4. **2WC**：W13 fuse 84.4% > 官方 78.5%

---

## 环境

```bash
source /project/peilab/why/eeg-brainit/scripts/activate.sh
export XFORMERS_DISABLED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
# Slurm 排除 dgx-10, dgx-11
```

---

## 不再投入的方向

- W11 式 SDXL-base 大规模 Pix 扫参
- S4 继续微调编码器冲 Pix（ROI 低）
- 无 LOSO 的 10 被试表作 main claim（仅可作 preliminary）
