# ERDC 架构与原理

## 1. 问题设定

**输入**：被试观看 THINGS 图像时的 EEG（测试集 200-way）。  
**输出**：与真实图像在 PixCorr / CLIP / 2WC 上对齐的重建图。  
**基线**：ATM (NeurIPS'24) — 高层 CLIP embed + 低层 neighbor img2img + SDXL-Turbo + IP-Adapter。

## 2. ERDC 核心思想

ERDC = **Evidence-Routed Diffusion Control**（证据路由的扩散控制）

在**不访问 GT** 的前提下，从多个 diffusion 候选中选出与 EEG 最一致、且受检索结构支持的结果。

### 2.1 三阶段

```
EEG ──► 编码器 ──► CLIP embed ──► 官方混合解码器 ──► K 个候选图
                      │                                    │
                      │         ┌──────────────────────────┘
                      ▼         ▼
              检索 train gallery          脑一致分数 brain_cos(eeg, gen_i)
              得 neighbor 低层图/latent          +
                                         λ · struct_cos(gen_i, neighbor_CLIP)
                      │                              │
                      └──────────► fuse 选优 ──► 最终输出
```

1. **检索结构（Retrieval Structure）**  
   用 CLIP 空间在 train gallery 找 top-M 邻居；低层 img2img / latent 注入保留 Pix 相关结构。

2. **脑一致闭环（Brain-consistency loop）**  
   对每个候选算 `brain_cos = cos(eeg_embed, CLIP(gen_i))`，选 brain 最大者。

3. **融合证据重选（Fused evidence reselect）**  
   `score_i = brain_i + λ · struct_i`  
   其中 `struct_i = cos(CLIP(gen_i), CLIP(neighbor_k))`。  
   **零重生成**：仅在已有 candidate bank 上重算分数并复制 PNG。

### 2.2 关键脚本

| 脚本 | 作用 |
|------|------|
| `erdc_official_atm_pipeline.py` | W12：官方 Turbo + IP-Adapter + 低层 img2img，生成 bank |
| `erdc_fuse_reselect.py` | fuse / shuffle / misalign / zero 机制对照 |
| `erdc_merge_banks.py` | 合并 bit / atm / prior_latent 银行 → mega-bank |
| `erdc_ras_closed_loop.py` | W8–W10：SDXL-base RAS 路线（Pix 较弱） |
| `erdc_twoway_metrics.py` | 2WC 指标 |
| `eval_atm_pipeline.py` | 导出 bit_clip embed、检索 Top-1 |

## 3. 编码器路线

| Embed | 来源 | 检索 Top-1 (sub-08) | 角色 |
|-------|------|----------------------|------|
| raw ATM | ATM backbone | ~34.5% | 官方对齐，Turbo 上 Pix 最高 |
| prior_atm | diffusion prior | — | 消融 |
| **bit_clip** | distill S3 head | **36.5%** | **论文主方法编码器** |

S3 训练配置：`configs/atm_distill_s3_sub08.yaml`  
ckpt 路径（主项目）：`outputs/atm_distill_s3_sub08/checkpoints/atm_stage3_best.pt`

## 4. 解码器：为何 W12 改变一切

| 栈 | Pix 天花板 (sub-08) | 说明 |
|----|---------------------|------|
| SDXL-base 30步 (W10) | ~0.133 | guidance=5，无官方低层栈 |
| **SDXL-Turbo 4步 + 官方低层 (W12)** | **0.15–0.17** | 与 ATM 论文一致 |

**结论**：Pix 短板主要在解码器/步数，不在编码器 alone。

## 5. 选优的 Pix–CLIP 权衡

同一 bank 内（bit Turbo, k=8）：

| 策略 | Pix | CLIP |
|------|-----|------|
| random | **0.165** | 0.365 |
| brain | 0.144 | 0.403 |
| fuse λ=0.25 | 0.150 | 0.417 |
| oracle | **0.164** | **0.456** |

- brain 优化语义（CLIP），牺牲 Pix（倾向高 strength）。
- fuse 在两者之间；merge 3 银行后 brain 可达 Pix **0.162**、CLIP **0.406**。

## 6. 机制对照（W14）

merged bank, λ=0.15：

| struct_mode | Pix | CLIP | 解读 |
|-------------|-----|------|------|
| retrieve | 0.159 | **0.418** | 正对照 |
| shuffle | **0.162** | 0.417 | Pix 略升，CLIP 几乎不变 → **结构项贡献有限，需论文如实写** |
| misalign | 0.158 | 0.414 | 低于 retrieve |
| zero (=brain) | 0.162 | 0.406 | 无结构项 |

## 7. 推荐论文叙事

**主方法行**：官方 Turbo 混合解码器 + **bit_clip** + **merge bank ERDC fuse**  
→ Pix ≈ 官方，CLIP +0.05，2WC 84.4%。

**上限行**：同栈 + raw ATM + brain → Pix 0.173。

**消融**：W10 SDXL fuse → CLIP 强、Pix 弱；证明 ERDC 与解码器解耦（decoder-agnostic）。
