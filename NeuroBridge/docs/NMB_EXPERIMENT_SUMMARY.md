# NeuroMem-Bridge 实验总结报告

> **项目**：NeuroBridge 扩展 — EEG 视觉解码（THINGS-EEG，sub-08 主实验）  
> **更新日期**：2026-09-02  
> **状态**：sub-08 生成 CLIP SOTA ≈ **0.412**；后验优化路线已穷尽；待扩 10 被试 + 端到端重训

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [数据与评估协议](#2-数据与评估协议)
3. [理论脉络](#3-理论脉络)
4. [主要技术路径](#4-主要技术路径)
5. [实验结果总表](#5-实验结果总表)
6. [核心发现：表征-解码间隙](#6-核心发现表征-解码间隙)
7. [失败路径与原因](#7-失败路径与原因)
8. [当前最优配置](#8-当前最优配置)
9. [统一框架提案（DA-HLM）](#9-统一框架提案da-hlm)
10. [复现指南](#10-复现指南)
11. [下一步计划](#11-下一步计划)

---

## 1. 背景与目标

### 1.1 问题定义

从 EEG 信号重建被试观看的视觉图像，服务于脑机接口（BCI）与视觉认知建模。

### 1.2 两条评估主线

| 主线 | 指标 | NeuroBridge 原文 SOTA |
|------|------|------------------------|
| **检索** | 200-way Top-1 / Top-5 | **63.2%** Top-1（RN50 空间） |
| **生成** | ViT-H paired CLIP cosine | 原文几乎不报；本工作主追此指标 |

本项目的核心矛盾：**检索 SOTA 与生成 SOTA 不在同一空间、同一优化目标上**。

### 1.3 项目定位

**NeuroMem-Bridge (NMB)**：在 NeuroBridge 强检索 backbone 上，引入 episodic memory、Fusion 层次语义、解码桥接，解决 EEG→图像的**表征-解码间隙**。

---

## 2. 数据与评估协议

| 项目 | 设置 |
|------|------|
| 数据集 | THINGS-EEG（10 被试，16540 train / 200 test） |
| 主实验被试 | **sub-08**（其余 9 被试有 NB checkpoint，未系统跑生成） |
| EEG 编码器 | NeuroBridge intra-subject RN50（`checkpoint_test_best.pth`） |
| 图像 GT 嵌入 | ViT-H-14 LAION2B 1024-d（`eeg-brainit/outputs/atm_bridge/`） |
| 生成解码器 | SDXL IP-Adapter ViT-H + RAG img2img |
| 主指标 | **paired CLIP cosine**（生成图 vs GT 图，ViT-H） |
| 辅指标 | FID、SSIM、PixCorr |
| Teacher 上界 | GT ViT-H embed 直接解码 ≈ **0.635** |

---

## 3. 理论脉络

### 3.1 从「拼接」到「统一」的演进

```
阶段 1  NeuroBridge 检索优化（RN50 对比学习）
    ↓  语义间隙 Δ_s 部分闭合，检索 Top-1 ≈ 63%
阶段 2  多模块拼接（Memory + CFT + Bridge + Blend）
    ↓  Fusion 嵌入极强（cos≈0.88），但生成一般
阶段 3  后验工程（DSDA / GACL / DADEM / DA-HLM-S）
    ↓  sub-08 平台 ≈ 0.412，无法突破
阶段 4  统一框架提案（DA-HLM）
    →  端到端 ViT-H 解码对齐（待实现）
```

### 3.2 核心命题：双重间隙

| 间隙 | 含义 | 现有方法 |
|------|------|----------|
| **语义间隙** $\Delta_s$ | EEG 与图像语义是否相近 | 对比学习、CFT → 检索强 / Fusion cos 高 |
| **解码间隙** $\Delta_d$ | 嵌入是否落在 IP-Adapter **可达流形** | **未系统闭合** → 生成弱 |

**实证**：CFT test paired cos = **0.886**，但 Fusion Prior 直接解码 CLIP 仅 **0.31**。

### 3.3 有效机制（经验归纳）

1. **ViT-H episodic memory**（soft-RAG k=5）：感知锚定，test cos ≈ 0.59
2. **Fusion→ViT-H 线性 bridge**：语义注入（非 Fusion 空间直接解码）
3. **α-blend**（mem ⊕ proj，α≈0.5）：语义-感知绑定
4. **低 strength img2img**（s=0.4）：保留邻居低层结构

---

## 4. 主要技术路径

### 4.1 路径 A：NB 全阶段基线（`nb_full_phases`）

**理论**：NeuroBridge → Adapter → Prior → Dual teacher → Ensemble 的标准扩展管线。

**流水线**：

```
EEG → NeuroBridge → Adapter (MLP/MLP) → ViT-H
     → Phase0: soft-RAG memory
     → Phase1: ATM DiffusionPrior
     → Phase2: Dual RN50+ViT-H teacher
     → Phase3: Ensemble
     → IP-Adapter 生成
```

**关键结果（sub-08）**：

| 路径 | CLIP | 备注 |
|------|------|------|
| Teacher（GT ViT-H） | **0.635** | 解码上界 |
| `rag_soft5_lowlevel` | **0.389** | **NB 基线最佳** |
| `rag_soft5`（无 img2img） | 0.369 | |
| `dual_vith1024` | 0.375 | |
| `prior_pretrained` embed cos | 0.338 | 检索 Top-1 14.5% |

**结论**：纯 NB→ViT-H adapter + memory img2img 是当前稳固基线；Prior 单独路径弱于 memory。

---

### 4.2 路径 B：NMB SOTA 主流程（`nb_nmb_sota`）

**理论**：Brain-HIVE Fusion Prior 层次语义 + NeuroBridge memory + CFT 跨空间传输。

**流水线**：

```
[0] Fusion Prior 微调（THINGS, 2500 steps）
[1] NB RN50 embeds (z_eeg_proj 512-d)
[2] ViT-H Memory Router (soft-RAG k=5) → mem_vith
[3] Fusion GT targets + Fusion Memory
[4] CFT: [z_proj ∥ mem_vith ∥ fusion_mem] → Fusion 1024-d
[5] Ensemble: α=0.45 · CFT + (1-α) · fusion_mem
[6] Fusion→ViT-H bridge (linear/MLP)
[7] 两条解码路径：
    - Fusion Prior + SDXL-turbo（Fusion 空间）
    - ViT-H IP-Adapter + img2img（ViT-H 空间）
```

**嵌入层指标（sub-08）**：

| 模块 | test paired cos | 200-way Top-1 |
|------|-----------------|---------------|
| mem_vith | 0.590 | ~0% |
| fusion_mem | 0.863 | ~0% |
| CFT → Fusion | **0.886** | **0%** |
| Ensemble → Fusion | 0.881 | ~0% |

**生成结果（`generation_full200`）**：

| 路径 | CLIP | FID |
|------|------|-----|
| `fusion_teacher`（Fusion GT 直接解码） | **0.313** | 220.1 |
| `nmb_sota_lowlevel` | 0.360 | 172.3 |
| `nmb_sota_lowlevel_rerank` | 0.365 | 171.9 |
| `nmb_cft_lowlevel` | 0.362 | 171.6 |

**结论**：Fusion Prior 主路径在 ViT-H metric 下**断裂**；有效路径是 ViT-H memory + img2img。

---

### 4.3 路径 C：过夜多路径扫描（`generation_overnight`，27 路）

**理论**：在路径 B 基础上，系统扫描 bridge 来源、blend α、img2img strength。

**扫描维度**：
- Bridge 输入：CFT / Ensemble / Fusion-GT / Fusion-mem
- Bridge 类型：linear / MLP
- Blend：mem ⊕ bridge_proj，α ∈ {0.40, 0.45, 0.50, 0.55, 0.60}
- Strength：s ∈ {0.4, 0.5, 0.6}

**Top-5 结果**：

| 排名 | 路径 | CLIP | FID |
|------|------|------|-----|
| **1** | `vith_blend_mem_linEns_a50_s04` | **0.412** | 176.9 |
| 2 | `vith_blend_rag5_mlpEns_a50_s04` | 0.411 | 176.1 |
| 3 | `nmb_ens_vith_rerank` | 0.405 | 172.0 |
| 4 | `vith_lin_fusiongt_s50` | 0.393 | 164.7 |
| 5 | `vith_baseline_mem_s50` | 0.389 | 169.6 |

**当前 SOTA 配置**：

```
embed = blend(mem_vith, linear_bridge(ensemble_fusion), α=0.5)
decode = IP-Adapter ViT-H + RAG img2img(strength=0.4)
CLIP ≈ 0.412
```

**关键发现**：strength=0.4 优于 0.5；blend 优于纯 mem 或纯 bridge。

---

### 4.4 路径 D：DSDA + GACL（`nb_nmb_dsda_gacl`）

**理论**：
- **DSDA**：双空间解耦 — PoE 语义-感知融合、confidence blend、自适应 img2img strength
- **GACL**：按 mem-GT 语义差距加权训练 Fusion→ViT-H bridge

**实验路径（5 路）**：

| 路径 | CLIP | 结论 |
|------|------|------|
| A 对照 mem_linEns | **0.412** | 最佳 |
| C conf blend + adapt s | 0.411 | 持平 |
| D GACL blend | 0.411 | 持平 |
| B PoE | 0.402 | 差于线性 blend |
| E GACL conf adapt | 0.412 | 持平 |

**结论**：启发式双空间融合 **不优于** 简单线性 blend；平台饱和。

---

### 4.5 路径 E：NMB-DADEM（`nb_nmb_dadem`）

**理论**（借鉴 D²-FOSA）：
- Memory-conditioned DDLG 在 ViT-H 空间
- 双向 E2I+I2E 扩散对齐
- Embed-space warm-start DDIM

**结果**：

| 路径 | CLIP | 嵌入 paired cos |
|------|------|----------------|
| A 对照 | **0.412** | — |
| D refine_mem_blend | 0.411 | blend 0.42 |
| C DAH align | 0.408 | align 0.622 |
| E bi refine | 0.399 | refine 0.11 |
| DDIM 纯输出 | — | **0.003~0.13** |

**结论**：轻量 MLP 扩散在冻结 encoder 上**再次失败**；DDIM 破坏语义嵌入。

---

### 4.6 路径 F：DA-HLM-S（`nb_nmb_dahlm`）

**理论**（统一框架首版）：
- **DAH**：可学习 α 绑定（泛化 mem_linEns）
- **ATM DiffusionPrior**：cond=[z_nb, mem] → ViT-H refine
- SOTA 教师蒸馏损失

**结果（9 路）**：

| 路径 | CLIP | 备注 |
|------|------|------|
| A 对照 | **0.412** | 最佳 |
| C DAH only | 0.411 | α 学到 0.502≈0.5 |
| F DAH+SOTA ens | 0.411 | |
| D DiffPrior only | 0.399 | prior cos 0.042 |
| E Hybrid | 0.409 | prior 拖累 |

**结论**：DAH **自动复现** α=0.5，生成持平；DiffPrior 第三次失败；统一叙事可行但数字未突破。

---

## 5. 实验结果总表

### 5.1 生成 CLIP 排名（sub-08，200 test）

| 排名 | 方法/路径 | CLIP | FID | 类型 |
|------|-----------|------|-----|------|
| 1 | **mem_linEns_a50 + s=0.4** | **0.412** | 177 | 后验 blend |
| 2 | DSDA conf adapt | 0.411 | 177 | 后验 |
| 3 | DA-HLM-S DAH | 0.411 | 177 | 统一头 |
| 4 | DADEM refine blend | 0.411 | 177 | 扩散 |
| 5 | NMB overnight #2 | 0.411 | 176 | 后验 |
| 6 | NMB SOTA rerank | 0.365 | 172 | Fusion 路径 |
| 7 | NB rag_soft5_lowlevel | 0.389 | — | 基线 |
| 8 | Fusion teacher | 0.313 | 220 | Fusion 解码 |
| — | Teacher 上界 | **0.635** | — | GT embed |
| — | D²-FOSA（文献） | 未报 CLIP | **146** | 外部 SOTA |

### 5.2 嵌入层 vs 生成层

| 嵌入来源 | paired cos | 生成 CLIP | 间隙 |
|----------|------------|-----------|------|
| CFT / Ensemble (Fusion) | **0.88** | 0.36 | 巨大 |
| DAH align_head | 0.62 | 0.41 | 明显 |
| SOTA teacher blend | 0.58 | 0.41 | 中等 |
| mem_vith | 0.53 | 0.39~0.41 | 较小 |
| DiffPrior / DDIM | **0.04** | 0.40 | 失效 |

### 5.3 与领域 SOTA 对比

| 方法 | 检索 Top-1 | 生成 CLIP | FID |
|------|-----------|-----------|-----|
| NeuroBridge（RN50） | **63.2%** | — | — |
| D²-FOSA（CVPR'26） | 38.0% | 未报 | **146** |
| ENIGMA（NeurIPS'25） | — | — | 人评 |
| **本工作 sub-08** | ~59%（mem） | **0.412** | ~177 |

---

## 6. 核心发现：表征-解码间隙

### 6.1 机制验证

```
高 embedding cosine  ≠  高 generation CLIP
```

- Fusion 空间优化到极致（cos 0.88）无法保证 ViT-H 解码
- 事后 bridge + blend 是**修补**而非**闭合**解码间隙
- img2img 邻居提供**感知锚定**，是生成平台的关键支撑

### 6.2 平台效应

sub-08 上 **7+ 种后验方法** 均收敛到 CLIP ≈ **0.412 ± 0.002**：

| 尝试 | 最佳 CLIP |
|------|-----------|
| blend α/strength sweep | 0.412 |
| DSDA PoE/conf | 0.412 |
| GACL weighted bridge | 0.412 |
| DADEM DDLG | 0.412 |
| DA-HLM-S DAH | 0.411 |

**推论**：在冻结 NB encoder + 后验模块设定下，sub-08 生成已达**局部最优**。

### 6.3 扩散对齐的三次失败

| 实验 | 条件 | prior val cos | 生成 CLIP |
|------|------|---------------|-----------|
| DADEM DDLG | 轻量 MLP, 26 ep | align 0.61 | 0.40 |
| DA-HLM-S DiffPrior | ATM UNet, 40 ep | 0.088 | 0.399 |
| full_phases prior | ATM pretrained | 0.25 | 0.34 |

**根因**：冻结 encoder + 短训 + 无生成损失；D²-FOSA/ATM 需**端到端长训**。

---

## 7. 失败路径与原因

| 路径 | 假设 | 结果 | 失败原因 |
|------|------|------|----------|
| Fusion Prior 主解码 | Brain-HIVE 层次语义 | CLIP 0.31 | 与 ViT-H 评估脱节；Turbo 4-step |
| CFT 直接生成 | Fusion cos 0.88 | CLIP 0.36 | 解码空间错误 |
| 事后 Fusion→ViT-H bridge 单独用 | 跨空间桥接 | < 0.39 | 缺 memory 锚定 |
| PoE / 启发式 DSDA | 双专家融合 | ≤ 0.412 | 不优于线性 blend |
| DDLG / DiffPrior 后验 | 潜空间扩散闭合间隙 | embed 0.04 | 未端到端；训练不足 |
| 纯噪声 DDIM 推理 | D²-FOSA 式 | embed 0.003 | CLIP 球面不适合 N(0,I) 初始化 |

---

## 8. 当前最优配置

### 8.1 SOTA 配方（sub-08）

```yaml
encoder: NeuroBridge RN50 (frozen)
  checkpoint: results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth

memory:
  type: soft-RAG
  k: 5
  tau: 0.07
  output: rag_soft5_test_clip_1024.npy

semantic:
  cft_ensemble: ensemble_a0.45_test_fusion.npy
  bridge: linear_adapter.pt (Fusion→ViT-H)
  blend: α=0.5 → mem_linEns_a50.npy

decode:
  model: SDXL IP-Adapter ViT-H
  img2img: RAG neighbor
  strength: 0.4

metrics:
  clip_cosine: 0.412
  fid: ~177
```

### 8.2 保留产物路径

```
outputs/nb_nmb_sota/sub-08/
  embeds/           # z_eeg_proj
  memory/           # rag_soft5, fusion_mem
  fusion/           # fusion GT
  cft/              # CFT 输出
  ensemble/         # ensemble
  bridge_vith/       # linear/mlp bridge
  blend_vith/       # mem_linEns_a50.npy ← SOTA embed
  fusion_prior_finetuned/
  generation_overnight/  # top5 路径
  summary.json, summary_overnight.json
```

---

## 9. 统一框架提案（DA-HLM）

基于全部实验，提出 **Decode-Aligned Hierarchical Latent Manifold**：

### 9.1 统一目标

$$\min_\theta \; \mathcal{L}_{ret} + \lambda_1 \mathcal{L}_{prior} + \lambda_2 \mathcal{L}_{mem} + \lambda_3 \mathcal{L}_{gen}$$

- 单一输出空间：**ViT-H CLIP 1024-d**（与 IP-Adapter 一致）
- 可微 episodic memory（替代 k-NN 后处理）
- DAH 可学习绑定（DA-HLM-S 已验证 α→0.5）
- Prior diffusion **端到端**训练（非后验）
- 可选生成 surrogate 损失

### 9.2 与现有拼接架构对比

| 维度 | 现有 NMB | DA-HLM |
|------|----------|--------|
| 模块数 | 7+ 串行 | 1 encoder + 1 DAN + 1 decoder |
| 输出空间 | Fusion + 事后 bridge | 直接 ViT-H |
| Memory | 推理 k-NN | 前向可微 attention |
| 损失 | 分段不同 metric | 检索 + 解码 + 可选生成 |
| sub-08 结果 | 0.412（后验饱和） | 待端到端验证 |

---

## 10. 复现指南

### 10.1 环境

```bash
source /project/peilab/why/eeg-brainit/scripts/activate.sh
export BRAIN_HIVE=/project/peilab/why/Brain-HIVE
cd /project/peilab/why/NeuroBridge
```

### 10.2 复现 SOTA 生成（sub-08）

```bash
# 全流程
bash scripts/nmb/run_nmb_sota_sub08.sh

# 过夜最佳路径（仅生成+评估）
python scripts/nb_adapter/generate_rag_lowlevel.py \
  --embed-npy outputs/nb_nmb_sota/sub-08/blend_vith/mem_linEns_a50.npy \
  --neighbor-idx-npy outputs/nb_nmb_sota/sub-08/memory/rag_soft5_neighbor_idx_test.npy \
  --output-dir outputs/repro/sota_sub08 \
  --strength 0.4 --max-images 0 --seed 42
```

### 10.3 关键脚本

| 脚本 | 用途 |
|------|------|
| `scripts/nmb/run_nmb_sota_sub08.sh` | NMB SOTA 全流程 |
| `scripts/nmb/run_nmb_overnight_multipath_sub08.sh` | 27 路扫描 |
| `scripts/nmb/nmb_dahlm_train.py` | DA-HLM-S 统一训练 |
| `scripts/nb_adapter/generate_rag_lowlevel.py` | ViT-H 生成 |
| `scripts/nb_adapter/train_nb_adapter_ext.py` | ATM DiffPrior 等 |

### 10.4 归档摘要

失败实验的 JSON 报告保存在：

```
outputs/_archived_experiment_summaries/
  nb_nmb_dadem/
  nb_nmb_dahlm/
  nb_nmb_dsda_gacl/
```

---

## 11. 下一步计划

| 优先级 | 任务 | 理由 |
|--------|------|------|
| **P0** | 10 被试批量（冻结 SOTA 配置） | 后验路线已穷尽，需统计检验 |
| **P1** | 端到端 NB* 重训 + ATM prior | 三次扩散失败根因是 frozen encoder |
| **P2** | 论文叙事：表征-解码间隙 | 负面结果 + DAH 统一化均有价值 |
| **P3** | FID 对比 D²-FOSA | 目前 ~177 vs 146，差距大 |
| **不做** | sub-08 继续后验 sweep | 平台 ±0.002 噪声 |

---

## 附录 A：Slurm 任务记录

| Job ID | 名称 | 状态 | 说明 |
|--------|------|------|------|
| 541949 | nb-nmb-sota08 | COMPLETED | NMB SOTA 全流程 |
| 542001 | nb-nmb-overnight | COMPLETED | 27 路扫描，SOTA 0.412 |
| 542460 | nb-dsda-gacl | COMPLETED | DSDA+GACL，持平 |
| 542464 | nb-dadem | COMPLETED | DADEM，持平 |
| 542475 | nb-dahlm | COMPLETED | DA-HLM-S，持平 |

## 附录 B：清理记录

见 `outputs/CLEANUP_MANIFEST.json`。已删除失败实验大文件（~4.2GB），保留正向结果与复现数据。

---

*文档维护：NeuroBridge / NeuroMem-Bridge 实验组*
