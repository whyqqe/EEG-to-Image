# 顶会攻关方案：从「表征-解码间隙」到「可解码对齐」——DA-HLM v2 端到端统一框架

> **项目**：EEG→图像重建（THINGS-EEG，主实验 sub-08，扩展 10 被试）
> **撰写日期**：2026-09-02（与《报告.md》同步）
> **基线**：NeuroMem-Bridge SOTA = CLIP **0.412**（sub-08，后验饱和）/ FID ≈ 177；Teacher 上界 = **0.635**
> **方案一句话**：放弃「冻结编码器 + 后验修补」，改为**多被试统一编码器 + 频率-语义双路 + 可解码对齐 + 检索锚定残差扩散 + 多条件生成 + 测试时证据路由**的端到端联合训练，把解码器可达流形作为表征学习的**硬约束**而非事后修补目标。

---

## 目录

1. [目标与可行性判断](#1-目标与可行性判断)
2. [瓶颈根因诊断：为什么 0.412 是平台](#2-瓶颈根因诊断为什么-0412-是平台)
3. [领域前沿调研（2024–2026）与启示](#3-领域前沿调研20242026与启示)
4. [总体架构：DA-HLM v2](#4-总体架构da-hlm-v2)
5. [模块详细设计](#5-模块详细设计)
6. [训练策略（四阶段 + 工程细节 + 算力预算）](#6-训练策略四阶段--工程细节--算力预算)
7. [实验设计与里程碑](#7-实验设计与里程碑)
8. [消融与对比计划](#8-消融与对比计划)
9. [风险分析与回退计划](#9-风险分析与回退计划)
10. [论文叙事与投稿策略](#10-论文叙事与投稿策略)
11. [实施路线图（16 周）](#11-实施路线图16-周)
12. [附录：关键文献清单](#12-附录关键文献清单)

---

## 1. 目标与可行性判断

### 1.1 目标

| 指标 | 当前（sub-08） | 中期目标 | 最终目标 | Teacher 上界 |
|------|----------------|----------|----------|--------------|
| 生成 CLIP（paired, ViT-H） | 0.412 | **0.46–0.48** | **0.50–0.54** | 0.635 |
| FID | ~177 | ≤ 165 | ≤ 155（对比 D²-FOSA 146） | — |
| 嵌入 paired cos（ViT-H） | 0.59（mem） | **≥ 0.70** | ≥ 0.75 | 1.0 |
| 200-way 检索 Top-1 | ~59% | ≥ 65% | ≥ 70% | 100% |
| 10 被试平均 CLIP | ~0.41（sub-08 单被试） | — | **≥ 0.47，10/10 优于单被试基线** | — |
| 跨被试泛化（train 9 / test 1） | 未跑 | — | **检索 Top-1 ≥ 40%** | — |

**可行性依据**：Teacher 上界 0.635 说明解码栈本身不是瓶颈（GT 嵌入直接解码即可达 0.635）；瓶颈在于「EEG 表征能否落到解码器可达流形」。当前 0.412→0.635 之间存在 0.22 的间隙，其中约一半可归因于表征质量（mem cos 0.59 vs teacher 1.0），另一半归因于解码条件（锚点/文本/结构）与测试时路由。本方案从两端同时闭合，**捕获该间隙的 40–60% 是现实目标**。

### 1.2 三个必须放弃的旧设定

1. **放弃 sub-08 后验 sweep**（平台 ±0.002 噪声，已有 7+ 方法验证）；
2. **放弃冻结 NeuroBridge 编码器**（三次扩散对齐失败的共同根因）；
3. **放弃「高嵌入余弦 = 高生成」的隐含假设**（实证：Fusion cos 0.886 → 生成 0.31）。

---

## 2. 瓶颈根因诊断：为什么 0.412 是平台

《报告.md》已给出完整证据链，这里归纳为四条结构性根因——**四条根因均不是「调参不足」，而是「目标函数与架构错配」，因此必须改架构与训练目标**：

### 2.1 检索锚定天花板（生成层）

```
解码 = IP-Adapter(embedding) + img2img(邻居, strength=0.4)
```

- strength=0.4 意味着 **60% 的低层结构来自检索邻居**；生成图与 GT 的相似度上限 ≈ 邻居质量 × 语义注入质量。
- mem_vith paired cos = 0.59 → 生成 CLIP = 0.41；邻居质量饱和 ⇒ 生成平台。
- **推论**：在「检索邻居即结构锚」设定下，提升的唯一途径是 (a) 提高邻居检索质量，(b) **让生成可以学到地离开锚点**（见 §5.5 AnchorDiff），(c) 引入独立于检索的文本/结构条件（见 §5.6）。

### 2.2 表征-解码间隙未被闭合（表征层）

- 现有训练目标：InfoNCE / 余弦对齐到 **CLIP 空间**（检索导向）。
- 解码成功取决于：嵌入是否落在 **IP-Adapter 可达流形**、与锚点互补性、是否保留可解码的感知结构。
- 事后 bridge/blend/DAH 都是在**冻结空间内的修补**——空间本身从未为解码而塑造。
- **推论**：必须让解码器（或解码器的可微代理）参与表征训练（§5.4 DecodeAligner）。

### 2.3 冻结编码器 + 无生成反馈（训练层）

三次扩散对齐失败的共同条件：

| 实验 | 冻结 encoder | 训练量 | 生成损失 | 结果 |
|------|:---:|:---:|:---:|------|
| DADEM DDLG | ✅ | 26 ep | ❌ | align 0.61 → gen 0.40 |
| DA-HLM-S DiffPrior | ✅ | 40 ep | ❌ | prior cos 0.088 |
| full_phases ATM prior | ✅ | 短 | ❌ | prior cos 0.25 |

- D²-FOSA / ATM / ENIGMA 等 SOTA 均为**端到端长训 + 含生成导向目标**。
- DDIM 纯噪声推理失败根因：CLIP 嵌入分布在**高维球面上的狭窄流形**，与 N(0,I) 初始化严重不匹配——扩散必须先知道「从哪儿出发」。

### 2.4 评估/条件空间错配

- Fusion 空间（cos 0.886）与 ViT-H/IP-Adapter 空间不一致 ⇒ 解码断裂（0.31–0.36）。
- 单条件（仅图像嵌入）生成缺少类别级语义先验与结构先验 ⇒ 文本条件 + 结构条件可提供**互补信息**（§5.6）。

---

## 3. 领域前沿调研（2024–2026）与启示

### 3.1 关键工作速览

| 工作 | 会议/年份 | 核心贡献 | 对本方案的启示 |
|------|-----------|----------|----------------|
| [D²-FOSA](https://www.openaccess.thecvf.com/content/CVPR2026/papers/Yu_D2-FOSA_Dual-Diffusion_Guided_EEG-to-Image_Reconstruction_with_Frequency-Oriented_Semantic_Alignment_CVPR_2026_paper.pdf) | CVPR 2026 | **双扩散 + 频率定向语义对齐**；THINGS-EEG 检索 38%、FID **146** | 频率特征路径有效；**语义扩散 + 结构扩散双路径**优于单路径（[解读](https://papernotes.org/CVPR2026/medical_imaging/d2-fosa_dual-diffusion_guided_eeg-to-image_reconstruction_with_frequency-oriente/)） |
| [ENIGMA](https://neurips.cc/virtual/2025/loc/san-diego/132694) | NeurIPS 2025 | **统一轻量多被试** EEG→image；15 分钟训练、<1% 参数 | 多被试统一模型可行且高效，不必逐被试重训 |
| [NEED](https://neurips.cc/virtual/2025/loc/san-diego/poster/118579) | NeurIPS 2025 | **跨被试、跨任务**（视频+图像）泛化 | 跨被试表征是可推广的；共享空间 + 轻量适配 |
| [MindBridge](https://xplorestaging.ieee.org/document/10655620) | NeurIPS 2024 | 跨被试解码框架（统一空间 + 被试特定投影） | Subject Adapter 设计范本 |
| [MindEye2](https://hackernoon.com/lite/mindeye2-shared-subject-models-enable-fmri-to-image-with-1-hour-of-data)（fMRI） | NeurIPS 2024 | 共享被试模型 + 被试特定 MLP，新被试 1 小时数据适配 | **共享 backbone + 轻量适配器**的数据效率红利 |
| [CognitionCapturer](https://ojs.aaai.org/index.php/AAAI/article/view/33587) | AAAI 2025 | 多模态信息（CLIP 图像+文本+…）联合解码 | 多模态条件显著增强解码 |
| [MindAlign](https://www.semanticscholar.org/paper/MindAlign%3A-Bridging-EEG%2C-Vision%2C-and-Language-for-Chen-Liu/8a0805a6df188217d22421feb074c2fc56ac8ed1) | arXiv 2025 | EEG+视觉+语言**零样本**解码 | 语言桥接提供类别级语义，利于零样本/跨类泛化 |
| [EEG-CLIP](https://www.sciencedirect.com/science/article/abs/pii/S0893608025010470) | Neural Networks 2025 | Transformer 框架 EEG 引导图像生成 | 生成导向的 EEG 编码可端到端训练 |
| [Autoregressive Visual Decoding from EEG](https://iclr.cc/virtual/2026/poster/10009323) | ICLR 2026 | 自回归式视觉解码新范式 | 差异化创新候选（风险高，作并行轨道） |
| [SCORE](https://scirate.com/arxiv/2608.19134) | arXiv 2025 | 无标签跨被试坐标恢复（检索） | 跨被试对齐的坐标级方案 |
| [SEBrain-CLIP](https://ieeexplore.ieee.org/document/11437695) | IEEE TNNLS 2025 | Masking + guided attention 鲁棒解码 | 掩码自监督提升 EEG 编码鲁棒性 |
| LaBraM / [EEGPT](https://axi.lims.ac.uk/paper/2410.19779) / [ALFEE](https://ar5iv.labs.arxiv.org/html/2505.06291) | 2023–2025 | EEG 基础模型（对比+MAE / 自回归） | 编码器可用基础模型初始化，或做自监督预训练 |
| [EEG-SVRec](https://github.com/hezy18/EEG-SVRec) | NeurIPS 2024 | 视频重建 + 时间动态对齐 + 跨被试 | 时间维度对齐思路可迁移到多 trial 建模 |
| [DynaMind](https://ar5iv.labs.arxiv.org/html/2509.01177) | arXiv 2025 | 动态视觉场景 + 时间-语义双对齐 | 「语义对齐 + 时间/结构对齐」双目标范式 |
| [BrainDecoder](https://ar5iv.labs.arxiv.org/html/2409.05279) | arXiv 2024 | 风格化视觉解码 | 风格/结构解耦条件化 |
| **ERDC（本组已有）** | 内部 | 证据路由 + 多候选融合重选（+0.05 CLIP） | 直接作为测试时模块集成（§5.7） |

### 3.2 从文献提炼的五条设计原则

1. **端到端 > 后验**：所有 SOTA 生成方法都是端到端联合训练（D²-FOSA、ENIGMA、ATM、EEG-CLIP）。
2. **频率特征是 EEG 的强先验**：D²-FOSA 的频率定向对齐、EEG-SVRec 的时间动态、DynaMind 的时间-语义双对齐，均说明「时域 + 频域」双路优于纯时域。
3. **跨被试统一表征是趋势且省算力**：ENIGMA（<1% 参数）、MindEye2（1 小时适配）、NEED（跨任务）证明共享表征 + 轻量适配是成熟范式。
4. **多条件解码 > 单条件**：图像嵌入（实例级）+ 文本（类别级）+ 结构（布局级）三者互补（CognitionCapturer、MindAlign、Brain-imager）。
5. **测试时路由可白赚分数**：ERDC 已证明多候选 + 脑一致重选 +0.05 CLIP 且无信息泄露。

---

## 4. 总体架构：DA-HLM v2

### 4.1 架构总览

```
                        ┌─────────────────────────────────────────────┐
                        │            DA-HLM v2（端到端联合训练）         │
                        │                                             │
 EEG(128ch×T) ──► [M1 Freq2Sem 频率-语义双路前端]                       │
                    时域分支 ─┐                                        │
                    频域分支 ─┴─ attention 融合 ──► [M2 UniEnc 多被试统一编码器]
                                                     │ 共享 backbone      │
                                                     │ + subject adapter  │
                                                     ▼                   │
                                        [M3 DecodeAligner 可解码对齐]     │
                                          ├─ InfoNCE → CLIP ViT-H (语义) │
                                          ├─ 余弦   → DINOv2 (感知结构)   │
                                          ├─ L2     → SD-VAE 粗潜码(可选)│
                                          └─ Probe-Decoder 代理生成损失    │
                                                     │                   │
                                        [M4 AnchorDiff 检索锚定残差扩散]    │
                                          EEG + 锚点 ──► 残差扩散 → 目标嵌入│
                                                     │                   │
                                        [M5 MultiCond 多条件生成栈]        │
                                          IP-Adapter(图像嵌入)            │
                                          + 文本提示(EEG→caption)         │
                                          + 结构条件(锚点 Canny/Depth)    │
                                          + DAH 可学习 α 绑定             │
                                                     │                   │
                                        [M6 测试时证据路由 ERDC]           │
                                          多候选生成 + 脑一致重选 ──► 输出图像│
                        └─────────────────────────────────────────────┘
```

### 4.2 与旧 NMB / DA-HLM-S 的差异

| 维度 | 旧 NMB（后验拼接） | DA-HLM-S（统一头） | **DA-HLM v2（本方案）** |
|------|--------------------|--------------------|--------------------------|
| 编码器 | 冻结 NB RN50 | 冻结 | **端到端可训练（LoRA 或全量）** |
| 数据 | sub-08 单被试 | sub-08 | **10 被试联合（≈10× 数据）** |
| 前端 | 纯时域 | 纯时域 | **频率-语义双路（Freq2Sem）** |
| 对齐目标 | CLIP（事后 bridge） | CLIP | **CLIP + DINOv2 + VAE 粗潜码 + 代理生成** |
| 扩散 | 后验、短训、无生成损失 | 后验 | **锚定残差扩散 + 端到端 + 生成代理损失** |
| 解码条件 | 仅图像嵌入 + 邻居 img2img | 同左 | **图像 + 文本 + 结构 三条件 + 学习 α** |
| 测试时 | 单次生成 | 单次生成 | **多候选证据路由（ERDC 集成）** |

### 4.3 统一优化目标

$$\min_\theta\; \underbrace{\mathcal{L}_{ret}}_{\text{检索}} + \lambda_1 \underbrace{\mathcal{L}_{percep}}_{\text{感知结构对齐}} + \lambda_2 \underbrace{\mathcal{L}_{probe}}_{\text{代理生成}} + \lambda_3 \underbrace{\mathcal{L}_{prior}}_{\text{锚定扩散}} + \lambda_4 \underbrace{\mathcal{L}_{cap}}_{\text{文本条件(可选)}} + \lambda_5 \underbrace{\mathcal{L}_{mem}}_{\text{可微记忆}}$$

- **单一输出空间**：ViT-H CLIP 1024-d（IP-Adapter 原生条件空间），DINOv2/VAE 作为辅助目标而非主输出。
- **可微 episodic memory**：gallery attention 前向可微，替代 k-NN 后处理，允许梯度流回编码器。
- **解码器（代理）参与训练**：Probe-Decoder 把「解码成功与否」变成可微信号——这是与所有后验方法的本质区别。

---

## 5. 模块详细设计

### 5.1 数据与目标准备（Stage 0，可并行）

| 目标 | 来源 | 用途 |
|------|------|------|
| CLIP ViT-H-14 LAION2B 1024-d 图像嵌入 | 已有（`eeg-brainit/outputs/atm_bridge/`） | InfoNCE 主目标、IP-Adapter 条件 |
| DINOv2 [CLS] 嵌入 | 新计算（官方权重，一次前向） | 感知结构对齐目标 |
| SD-XL VAE 潜码（64×64×4）粗化版本（低通/平均池化） | 新计算 | 结构头可选目标 |
| 图像 caption（BLIP-2 或邻居 gallery 已有 caption） | 新计算 | 文本条件、EEG→caption 蒸馏目标 |
| 多 trial 聚合 | 每图 4 trial：可选平均、集合注意力、trial-dropout 增强 | 输入增强与鲁棒性 |
| 频率特征（STFT / 滤波器组功率） | 新计算（每段 1–50Hz） | Freq2Sem 频域分支 |

**注意**：所有目标一次性离线算好（一次 CLIP/DINOv2/VAE/caption 前向），训练期零额外成本。

### 5.2 M1：Freq2Sem 频率-语义双路前端

- **时域分支**：raw 段（带通 1–50Hz，z-score），Conv1D 栈（或复用 NB 前端）。
- **频域分支**：STFT → 对数功率谱 → 频率滤波组（δ/θ/α/β/γ 与宽带），轻量 Conv/MLP → 频域 token。
- **融合**：跨分支 cross-attention（频域 token 引导时域 token），输出生理先验增强的 token 序列。
- **动机**：D²-FOSA 的频率定向对齐是当前 FID SOTA（146）的关键；EEG 的判别信息集中在频带与相位结构中。
- **成本**：< 1M 参数，训练不增负。

### 5.3 M2：UniEnc 多被试统一编码器

- **共享 backbone**：初始化自 NeuroBridge RN50 权重（保留检索先验）或 EEG 基础模型（LaBraM/EEGPT，若可获取）；继续训练（LoRA 起步，S4 放开）。
- **被试适配器**：每个被试一个轻量 FiLM 层（或 MindEye2 式单层 MLP 投影），参数 ~0.1–1M/被试。
- **联合训练**：10 被试共享 backbone，batch 内混被试；同一图像的 trial 视为正样本对。
- **预期收益**：数据量 ×10 ⇒ 表征统计显著更强；跨被试泛化（train 9 / test 1）成为论文卖点。
- **可微记忆**：gallery embedding 库作为 attention keys；EEG 查询经 attention 读出（软检索，k=5 软注意力替代硬 k-NN），梯度可回传。

### 5.4 M3：DecodeAligner 可解码对齐（核心创新）

三个子目标 + 一个代理：

1. **语义对齐（检索）**：InfoNCE(EEG, CLIP ViT-H 图像嵌入)，**硬负样本挖掘**——把检索邻居（语义易混淆项）与同类别其他图像作为负样本，提升判别性。
2. **感知结构对齐**：余弦对齐 EEG → DINOv2 [CLS]（感知/结构表征，MindEye 等已证明与 CLIP 互补）。
3. **解码原生对齐（可选结构头）**：EEG → SD-VAE 粗潜码（低通后的 8×8×4 编码），L2 损失——迫使表征携带**解码器原生空间**的信息，闭合「解码间隙」by construction。
4. **Probe-Decoder 代理生成损失（关键）**：
   - 训练一个轻量代理（MLP/小型 Transformer）：输入 [EEG 表征, 锚点嵌入, 文本嵌入]，输出「模拟解码后图像的 CLIP 嵌入」；
   - 用真实 SDXL IP-Adapter + img2img 在**少量 GT 嵌入上蒸馏**代理（离线几百个样本即可）；
   - 训练期用代理计算 $\mathcal{L}_{probe}=1-\cos(\hat{y}_{clip},\; y_{gt}^{clip})$——**把「解码是否成功」变成每步可微损失**，无需每步跑 SDXL。

> **为什么这是可发表的创新点**：文献中「表征-解码间隙」被反复观察到（本组实证 + 社区共识），但多数方法用后验修补；DecodeAligner 是**首个把解码器可达性作为训练期硬约束**的方案（对比 D²-FOSA 的端到端但无显式间隙定义）。

### 5.5 M4：AnchorDiff 检索锚定残差扩散

- **问题**：DDIM 从 N(0,I) 出发失败（CLIP 球面流形不兼容）；img2img 从邻居出发有效但被锚点锁死。
- **方案**：学习「锚点 → 目标」的**残差扩散**：
  - 训练期：对 (锚点嵌入 $e_a$, GT 嵌入 $e_g$) 构造残差 $r = e_g - e_a$，扩散模型以条件 $[\text{EEG表征}, e_a]$ 生成 $r$；
  - 推理期：$e_{out} = e_a + \text{denoise}(e_a, \text{EEG})$，输出仍在 CLIP 球面附近（流形兼容），且**可系统离开锚点**。
- **与 D²-FOSA 双扩散的差异**：D²-FOSA 是「语义扩散 + 频率结构扩散」双通道；AnchorDiff 是「检索先验 + 残差修正」单通道，更贴合已有 memory+img2img 基线的成功经验。
- **实现**：条件 DDPM/DDIM（UNet 或 Transformer 均可，嵌入空间 1024-d，轻量 ~50M）；这是 DA-HLM-S DiffPrior 的**修正版**：有锚点起点 + 端到端梯度。

### 5.6 M5：MultiCond 多条件生成栈

- **条件 1（实例语义）**：IP-Adapter(CLIP ViT-H 嵌入)。
- **条件 2（类别语义）**：文本提示。来源二选一或并用：
  - (a) 检索邻居的 caption（gallery 已有）；
  - (b) 训练轻量 EEG→caption 模块（蒸馏到 BLIP-2 文本嵌入，MindAlign 式），支持零样本跨类。
- **条件 3（结构）**：锚点图的 Canny/Depth（ControlNet 或低层 img2img），提供布局先验。
- **绑定**：DAH 可学习 α 泛化为多条件加权（已验证 α 可学且收敛，DA-HLM-S 学到 0.502≈0.5，说明机制稳定）。
- **预期收益**：文本条件补足「类别正确性」（CLIP 指标本质是语义相似度，文本类别先验直接加分）；结构条件补足「布局保真」（FID 下降）。

### 5.7 M6：测试时证据路由（集成 ERDC，已有代码）

- 多候选生成（多锚点、多 strength、多 seed、合并 bank）→ 脑一致评分（EEG 嵌入 vs 候选 CLIP）→ 结构一致评分 → 融合重选。
- ERDC 已证明 +0.048 CLIP / 2WC 84.4%（sub-08），且选优阶段不接触 GT，公平。
- **集成后管线**：`DA-HLM v2 模型 → 多候选 → ERDC 重选`，两段增益可叠加。

---

## 6. 训练策略（四阶段 + 工程细节 + 算力预算）

### 6.1 阶段划分（课程式，每阶段有保底产物）

| 阶段 | 内容 | 冻结/训练 | 主要损失 | 保底产物 |
|------|------|-----------|----------|----------|
| **S1** | 目标离线计算 + 基线复现 | — | — | CLIP/DINOv2/VAE/caption 目标、gallery、冻结基线 ≥ 0.412 |
| **S2** | 可解码对齐（UniEnc + Freq2Sem + DecodeAligner） | 解码栈冻结；encoder/adapter 训练 | $\mathcal{L}_{ret}+\mathcal{L}_{percep}+\mathcal{L}_{probe}$ | 嵌入 cos ≥ 0.70、检索 Top-1 ≥ 65%（sub-08） |
| **S3** | AnchorDiff prior + EEG→caption | 表征冻结（S2 产物）；prior/caption 训练 | $\mathcal{L}_{prior}+\mathcal{L}_{cap}$ | 锚点残差扩散可生成且优于纯锚点 |
| **S4** | 端到端联合微调（关键阶段） | 解冻 encoder（或 LoRA）+ prior + 条件栈 | 全部损失 + 周期性真实解码代理 | **生成 CLIP ≥ 0.50、FID ≤ 165（sub-08）** |

### 6.2 S4 端到端的工程要点（前三次扩散失败的修复清单）

1. **解冻**：encoder 以 LoRA 起步（lr 1e-4），稳定后可选全量（lr 1e-5）。
2. **生成反馈闭环**：每 50–100 步，用小批量（4–8 张）× SDXL-turbo（4–8 步）真实解码，计算 **CLIP/DreamSim 感知损失** vs GT——把代理损失与真实解码对齐（代理保效率，真实保准确）。
3. **课程退火**：S4 初期以「检索锚点图」为 warm-start 解码目标，逐步退火到「纯生成图」，避免早期梯度噪声。
4. **教师嵌入 EMA**：CLIP/VAE 目标是离线固定的；对 gallery 记忆库用 EMA 更新，避免记忆漂移。
5. **多 trial 集合建模**：同图多 trial 用集合注意力（EEG-SVRec 思路）或 trial-dropout 增强，训练期随机丢 trial 提升鲁棒性。
6. **稳定技巧**：混合精度、梯度裁剪、EMA 权重、学习率 warmup+cosine、按被试采样平衡（保证 sub-08 不因多被试稀释）。

### 6.3 算力预算估算（供排期）

| 项 | 量级 | 说明 |
|----|------|------|
| 目标离线计算 | 1–2 GPU·天 | CLIP/DINOv2/VAE/caption 各一次前向 |
| S2 对齐训练 | 4–8 GPU·天（10 被试联合） | 66k trial/被试 × 10；单卡 epoch ~1h |
| S3 prior 训练 | 2–3 GPU·天 | 轻量 50M 模型 |
| S4 端到端 | 8–16 GPU·天 | 含周期性真实解码（每 50 步 8 张 turbo） |
| 全量消融 | 10–20 GPU·天 | 8–12 组消融 × 单被试快速版 |

总计约 **25–50 GPU·天**，在现有 Slurm 集群上 3–4 周内可完成一轮完整闭环（与路线图 §11 一致）。

---

## 7. 实验设计与里程碑

### 7.1 主线实验

| 编号 | 实验 | 配置 | 判定标准 |
|------|------|------|----------|
| E1 | S2 可解码对齐（单被试 sub-08） | 对齐训练 + 现有解码栈 | 生成 CLIP ≥ 0.45（vs 0.412） |
| E2 | S2 多被试联合 | 10 被试统一 + adapter | sub-08 ≥ E1；均值 ≥ 0.44 |
| E3 | S3 AnchorDiff | prior + 锚点残差 | 生成 ≥ E2，且「离开锚点」消融为正 |
| E4 | S4 端到端（核心） | 全部模块 + 真实解码反馈 | **CLIP ≥ 0.50，FID ≤ 165** |
| E5 | +ERDC 测试时路由 | 多候选 + 重选 | CLIP +0.03~0.05 叠加 |
| E6 | 跨被试泛化 | train 9 / test 1 + adapter 微调 | 检索 Top-1 ≥ 40%；新被试少量数据适配 |

### 7.2 评估协议（对齐领域规范）

- 检索：200-way Top-1/Top-5（对齐 NeuroBridge 63.2%、D²-FOSA 38%）。
- 生成：paired CLIP cosine（ViT-H）、PixCorr、FID、SSIM、2-way 人评（对齐 ATM/ENIGMA 协议）。
- 统计：Bootstrap 95% CI（ERDC 已建立该协议）；10 被试配对检验。

### 7.3 里程碑（可检查点）

| 里程碑 | 时间 | 通过条件 | 失败对策 |
|--------|------|----------|----------|
| M0 | 第 2 周末 | 冻结基线复现 ≥ 0.412 + 目标库就绪 | 排查数据/代码 |
| M1 | 第 5 周末 | S2 嵌入 cos ≥ 0.70、检索 ≥ 65%、生成 ≥ 0.45 | 调对齐权重/硬负样本 |
| M2 | 第 8 周末 | S4 sub-08 CLIP ≥ 0.50、FID ≤ 165 | 见 §9 回退 |
| M3 | 第 12 周末 | 10 被试 + 跨被试全套结果 | 缩减为单被试+精选扩展 |
| M4 | 第 16 周末 | 论文初稿 + 复现包 | — |

---

## 8. 消融与对比计划

### 8.1 消融设计（每个组件一个开关）

| 组件 | 消融方式 | 预期影响 |
|------|----------|----------|
| Freq2Sem 频域分支 | 仅时域 | 检索/生成下降（参照 D²-FOSA） |
| 多被试联合 | 单被试 sub-08 | 表征质量下降 |
| Subject Adapter | 去掉（直接共享） | 跨被试泛化下降 |
| DINOv2 感知对齐 | 去掉 | FID 变差 |
| Probe-Decoder 代理损失 | 去掉 | 生成与嵌入 cos 脱钩（间隙重现） |
| AnchorDiff | 退化为纯锚点（旧基线） | 回到 0.412 平台 |
| 文本条件 | 去掉 | 类别级语义下降 |
| ERDC 路由 | 单次生成 | -0.03~0.05 |

### 8.2 外部对比

| 方法 | 对比协议 | 目标 |
|------|----------|------|
| NeuroBridge（本组基线） | 检索 Top-1 | ≥ 63.2% |
| ATM 官方（THINGS-EEG2 协议） | 生成 CLIP/PixCorr | CLIP 0.370→≥0.50 |
| D²-FOSA | 检索 38% / FID 146 | 检索 ≥ 38%，FID ≤ 155 |
| ENIGMA | 多被试生成 | 均值 CLIP 超越 |
| ERDC（本组已有） | 2WC/CLIP | 叠加增益为正 |

---

## 9. 风险分析与回退计划

| 风险 | 概率 | 影响 | 对策 |
|------|:---:|------|------|
| S4 端到端不稳定（梯度爆炸/表征漂移） | 中 | 高 | LoRA 限定解冻；S2 冻结检查点保底；课程退火 |
| VAE 粗潜码对齐与 CLIP 冲突 | 低-中 | 中 | λ₂ 退火；仅保留 DINOv2+CLIP 双目标 |
| Probe 代理与真实解码偏差 | 中 | 中 | 周期性真实解码校准（§6.2-2）；代理每 500 步重蒸馏 |
| 多被试联合稀释 sub-08 | 低-中 | 中 | 按被试采样平衡；sub-08 权重 ×2 |
| 跨被试泛化不达 40% | 中 | 低-中 | 降级为「10 被试被试内统一模型」叙事（ENIGMA 路线） |
| 文本条件引入噪声 | 低 | 低 | EEG→caption 置信度门控；退化为邻居 caption |
| 算力不足 | 低 | 中 | 4 阶段逐一保底（每阶段产物都可投稿） |
| **极端回退**：端到端全部失败 | — | — | 保底投稿角度：「后验路线系统的负面分析 + 表征-解码间隙实证」（已有 0.412 平台 + 7 方法全失败证据，DADEM/DAH 等已有结果） |

**成功率的判断依据**：本方案每个模块都有 1 个以上已发表工作背书（Freq2Sem→D²-FOSA；多被试→ENIGMA/NEED/MindEye2；代理生成→深度学习蒸馏通用范式；锚定扩散→本组 img2img 成功经验 + D²-FOSA 双扩散；文本条件→CognitionCapturer/MindAlign；路由→ERDC 已证）。**没有无依据的赌注**；主要风险集中在 S4 端到端稳定性，已配置三重保底。

---

## 10. 论文叙事与投稿策略

### 10.1 核心故事线

```
观察：表征-解码间隙（嵌入 cos 0.88 → 生成 0.31；7+ 后验方法平台 0.412）
  ↓ 定义与量化
问题：现有训练目标只对齐 CLIP 空间，不解码器可达流形
  ↓ 机制性修复（非修补）
方案：DA-HLM v2 = 可解码对齐（解码原生/代理目标进表征训练）
     + 检索锚定残差扩散（流形兼容的生成）
     + 多被试统一（10× 数据）+ 多条件解码 + 测试时路由
  ↓
证据：检索 + 生成 + 跨被试 + 系统消融（每个组件贡献为正）
```

### 10.2 三个卖点（对齐顶会口味）

1. **新问题形式化**：把「表征-解码间隙」定义为可测量的双指标 gap（嵌入 cos vs 生成 CLIP），给出归因与闭合方法——方法论贡献，独立于具体数字。
2. **可解码对齐训练范式**：解码器（代理）参与表征训练，可迁移到任何「脑信号→生成模型」任务（EEG/fMRI→图/文/视频），受众广。
3. **跨被试统一 + 数据效率**：10 被试联合 + 轻量适配 + 跨被试泛化，切中 BCI 实用化痛点（ENIGMA/NEED 同赛道）。

### 10.3 投稿策略

- 主投：**NeurIPS / CVPR / ICLR**（生成 + 表征交叉，与 D²-FOSA/ENIGMA 同赛道）。
- 备选：AAAI（CognitionCapturer 同赛道）、TMI/TIP（期刊长文）。
- 差异化定位：相对 D²-FOSA（双扩散）强调「间隙闭合 + 代理训练」；相对 ENIGMA（轻量）强调「解码质量 + 间隙理论」。

---

## 11. 实施路线图（16 周）

| 周 | 任务 | 产出 |
|----|------|------|
| 1–2 | S1：目标离线计算（DINOv2/VAE/caption）、gallery 构建、冻结基线复现 | 目标库 + 基线 ≥ 0.412 |
| 3–4 | M1 Freq2Sem + M2 UniEnc（单被试版）实现；S2 训练框架搭建 | 代码 + 首次 S2 训练 |
| 5 | **M0 检查点**：S2 sub-08 嵌入 cos ≥ 0.70 / 检索 ≥ 65% / 生成 ≥ 0.45 | 里程碑判定 |
| 6–7 | 10 被试联合训练 + subject adapter + 可微记忆 | 多被试 S2 |
| 8 | M3 AnchorDiff prior 实现与训练 | prior 收敛 |
| 9–10 | **S4 端到端**：LoRA 解冻 + 代理损失 + 周期性真实解码 | 端到端检查点 |
| 11 | **M1 检查点**：sub-08 CLIP ≥ 0.50 / FID ≤ 165；如失败启动 §9 回退 | 里程碑判定 |
| 12 | E5 ERDC 集成 + E6 跨被试实验 | 完整主线结果 |
| 13–14 | 系统消融（§8.1）+ 统计检验（Bootstrap/配对） | 消融表 |
| 15–16 | 论文写作 + 复现包整理（脚本、checkpoint、README） | 初稿 + 复现包 |

---

## 12. 附录：关键文献清单

### 方法对标
- **D²-FOSA**（CVPR 2026）：Dual-Diffusion + Frequency-Oriented Semantic Alignment —— [论文](https://www.openaccess.thecvf.com/content/CVPR2026/papers/Yu_D2-FOSA_Dual-Diffusion_Guided_EEG-to-Image_Reconstruction_with_Frequency-Oriented_Semantic_Alignment_CVPR_2026_paper.pdf) · [解读](https://papernotes.org/CVPR2026/medical_imaging/d2-fosa_dual-diffusion_guided_eeg-to-image_reconstruction_with_frequency-oriente/)
- **ENIGMA**（NeurIPS 2025）：统一轻量多被试 EEG→Image —— [官方页](https://neurips.cc/virtual/2025/loc/san-diego/132694)
- **NEED**（NeurIPS 2025）：跨被试、跨任务视频/图像重建 —— [官方页](https://neurips.cc/virtual/2025/loc/san-diego/poster/118579)
- **MindBridge**（NeurIPS 2024）：跨被试解码框架 —— [IEEE 收录页](https://xplorestaging.ieee.org/document/10655620)
- **CognitionCapturer**（AAAI 2025）：多模态信息 EEG 视觉解码 —— [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33587)
- **EEG-CLIP**（Neural Networks 2025）：Transformer EEG 引导图像生成 —— [期刊页](https://www.sciencedirect.com/science/article/abs/pii/S0893608025010470)
- **Autoregressive Visual Decoding from EEG**（ICLR 2026）—— [ICLR 页](https://iclr.cc/virtual/2026/poster/10009323)
- **EEG-SVRec**（NeurIPS 2024）：视频重建 + 时间对齐 + 跨被试 —— [GitHub](https://github.com/hezy18/EEG-SVRec)
- **DynaMind**（arXiv 2025）：动态场景时间-语义双对齐 —— [arXiv](https://ar5iv.labs.arxiv.org/html/2509.01177)
- **MindAlign**（arXiv 2025）：EEG+视觉+语言零样本解码 —— [Semantic Scholar](https://www.semanticscholar.org/paper/MindAlign%3A-Bridging-EEG%2C-Vision%2C-and-Language-for-Chen-Liu/8a0805a6df188217d22421feb074c2fc56ac8ed1)
- **SCORE**（arXiv 2025）：无标签跨被试坐标恢复 —— [arXiv](https://scirate.com/arxiv/2608.19134)
- **SEBrain-CLIP**（IEEE TNNLS 2025）：掩码 + 引导注意力 —— [IEEE](https://ieeexplore.ieee.org/document/11437695)
- **UniBrain**：跨被试统一解码 —— [arXiv](https://ar5iv.labs.arxiv.org/html/2308.07428)
- **BrainDecoder**（arXiv 2024）：风格化视觉解码 —— [arXiv](https://ar5iv.labs.arxiv.org/html/2409.05279)

### 跨模态/解码器侧参考
- **MindEye2**（fMRI, NeurIPS 2024）：共享被试 + 轻量适配 —— [解读](https://hackernoon.com/lite/mindeye2-shared-subject-models-enable-fmri-to-image-with-1-hour-of-data)
- **Brain-Diffuser**（fMRI）：生成潜扩散范式 —— [GitHub](https://github.com/yohann-benchetrit/brain-diffuser)
- **Brain-imager**：多模态重建+描述 —— [Springer](https://link.springer.com/article/10.1186/s40708-025-00282-x)
- **EEG2IMAGE**（Miyapuram 组）：EEG 图像重建 —— [OpenReview](https://openreview.net/forum?id=9OsFsJZgfk)
- **Saliency-Guided EEG 重建**（arXiv 2025）—— [arXiv](https://ar5iv.labs.arxiv.org/html/2510.26391)

### EEG 基础模型
- **EEGPT**：自回归预训练 —— [论文页](https://axi.lims.ac.uk/paper/2410.19779)
- **ALFEE**：自适应大模型调研（含 LaBraM 对比）—— [arXiv](https://ar5iv.labs.arxiv.org/html/2505.06291)

### 本组已有资产（复用）
- 《报告.md》：NeuroMem-Bridge 全量实验记录（0.412 平台、7+ 失败路径、DA-HLM 提案）
- 《ERDC.md》：证据路由扩散控制（CLIP +0.048，测试时模块直接集成）
- NeuroBridge 仓库脚本：`run_nmb_sota_sub08.sh`、`nmb_dahlm_train.py`、`generate_rag_lowlevel.py`、`train_nb_adapter_ext.py`

---

*文档维护：EEG 视觉解码组 · 攻关方案 v1.0（与《报告.md》同步更新）*
