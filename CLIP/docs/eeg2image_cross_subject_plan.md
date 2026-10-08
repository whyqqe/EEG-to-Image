# SAM-CLIP：把被试当作「模态」的跨被试 EEG-to-Image 对齐方案

> ⚠️ **本文档是 v1 历史设计，其中的核心机制（被试条件化 \(z_s\) / FiLM / LoRA / hypernetwork、
> 以及三级训练流程）已在 v2 中被实测否证并移除。** 当前有效方案见
> [`docs/eeg2image_v2_plan.md`](./eeg2image_v2_plan.md)；否证的证据与替代设计见该文档 §2，
> 代码层面的墓碑见 `src/samclip/models/subject_conditioning.py`。
>
> 本文档**不删除**是有意的：§4.4 的被试描述子分离度测量、§7 的校准论证、以及各处的协议约定
> 仍是 v2 的依据，且 v2 的多处结论是**以本文档的实测数据为前提**的。读的时候请把 §4.4 与
> 「三阶段」当作**已被推翻的假设**，而不是待实现的设计。

> 工作名 **SAM-CLIP**（**S**ubject-**A**s-**M**odality **CLIP** alignment）。
> 目标：在 THINGS-EEG2 上解决 **跨被试（inter-subject / LOSO）泛化**，并支持**用少量校准数据适配新被试**。
> 文档版本：2026-10-03。作者：待定。状态：设计稿（待评审后进入实现）。

---

## 0. TL;DR

- **核心视角**：把「被试」视为一种**观测模态（modality）**——同一刺激经过不同被试的解剖/阻抗/参考/注意力差异，形成不同的「观测算子」。因此共享 EEG 编码器的前向过程应当是**被 subject-conditioned 的**：\( f_\theta(x_s; z_s) \)。
- **三条技术支柱**（对应你的原设想）：
  1. **被试特征 \(z_s\)**（§4.4.1，2026-10-03 修订）：**统计量锚 + 低秩学习修正**——
     \(z_s = W_{\text{stat}}[\overline{x}_c(\mathcal{S}_s);\sigma_c(\mathcal{S}_s)] + H_\phi(\mathcal{S}_s)\)，
     且**训练与部署使用同一个 set-encoder 估计器**（`z_source: support`）。原"训练走可学习
     embedding 表"的设计已被实测否证：自由被试向量无带宽上限，主干会把它当"被试代码"用。
  2. **调制共享编码器**：\(z_s\) 经由 **FiLM + 低秩 adapter (LoRA-style)** 注入共享 trunk——即「共享全秩主干 + 被试专属低秩修正」，对应 SuLoRA / Stacked-LoRA / FiLM-EEG 的已验证做法。
  3. **两级对比对齐**：**跨被试对比**（同一图像、不同被试的 EEG 为正样本）去除个体差异；**EEG–图像对比**把共享表征映射进冻结的 **CLIP 图像空间**保留图像语义。
- **部署侧必须补一刀**：按 `PROTOCOL_INTER.md` 的结论，inter-subject 榜单顶端（SCORE 53.23）主要收益来自**免标签的测试时几何校准**（subject-adaptive whitening + CSLS / coordinate recovery），而不是编码器本身。本方案把这一层作为**独立、可开关**的模块（§7）纳入。
- **训练策略**：三阶段——(0) 缓存与预处理 → (1) 多被试 subject-conditioned 预训练 → (2) **episodic 元训练**让 \(H_\phi\) 学会「从 K 个试次推断 \(z_s\)」→ (3) 新被试少样本校准（只更新被试专属参数）。
- **评测**：严格 LOSO、63 导（inter）/ 17 导（intra）、200-way Top-1/Top-5、final-epoch 选点（防泄漏），并报告 \(N\in\{5,10,20,50\}\) 校准曲线。

---

## 1. 问题定义与定位

### 1.1 任务

给定被试 \(s\) 观看图像 \(I\) 时记录的 EEG 片段 \(x_s \in \mathbb{R}^{C\times T}\)（THINGS-EEG2：\(C=63\), \(T=250\)），学习编码器映射到与图像语义对齐的嵌入空间，用于：

| 读出 | 输出 | 主要依赖 |
|---|---|---|
| **检索 (retrieval)** | 200 候选中给出排名 | 语义（猫 vs 狗） |
| **重建 (reconstruction)** | 像素 | 语义 + 结构（姿态、布局、轮廓） |

本方案以**检索**为主目标（对齐到 CLIP 空间天然支持检索），并把重建作为下游可选分支（复用 §4.6 的 CLIP 空间输出接 diffusion prior）。

### 1.2 关键约束（决定方案形态）

1. **零样本**：1654 训练概念与 200 测试概念**完全不重叠**，模型必须学「怎么解码」而非「记住什么」。
2. **训练/测试重复数不对称**：训练每图 4 次、测试每图 80 次；测试平均后 SNR 约为训练的 \(\sqrt{80/4}\approx 4.5\) 倍，测试集只有 200 行。
3. **被试间分布漂移**是主矛盾：同一图像在两个被试上的 EEG 表征差异，往往大于同一被试上两张不同图像的差异。
4. **导联数随设定切换**：intra-subject 用 17 导（occipito-parietal）更好，但 **inter-subject 用全部 63 导** 更稳（前部导联提供「锚定」信息，主要提升 Top-5）。这一点在 `PROTOCOL_INTER.md` 有跨论文的一致证据。
5. **选点不能用测试集**：必须 final-epoch（或概念级留出），否则口径不可比。

### 1.3 你的原始想法 → 形式化映射

| 你的表述 | 形式化 | 本文对应 |
|---|---|---|
| 把不同被试视为不同观测方式的「模态」 | 被试 = 条件 \(z_s\)，前向为 \(f_\theta(x;z_s)\) | §4.4 |
| 用每位被试少量 EEG 学习被试特征 | \(z_s = H_\phi(\mathcal{S}_s)\)，\(\mathcal{S}_s\) 为 support 集 | §4.4.2, §6 |
| 用该特征调节共享 EEG 编码器 | FiLM + LoRA 注入每层 | §4.4.3 |
| 对比学习对齐不同被试看同一图像的 EEG | 跨被试 InfoNCE | §4.5 |
| 联合 EEG–图像对齐损失映射到 CLIP 空间 | EEG–image InfoNCE | §4.6 |
| 保留图像信息 + 减小个体差异 | 对抗/去相关 + 不变性正则 | §4.5.3, §4.7 |
| 少量校准适配新被试 | 只训练被试专属参数 | §6 |

---

## 2. 相关前沿工作总结

> 下表为 2024–2026 与本题直接相关的工作。**加粗**为最值得借鉴的机制。

### 2.1 EEG-to-Image（检索 / 对齐）

| 方法 | 年份/出处 | 核心机制 | inter-subject 表现 | 对本方案的价值 |
|---|---|---|---|---|
| ATM | NeurIPS 2024 (arXiv 2403.07721) | channel-wise Transformer + temporal-spatial conv + **subject token/linear** | 5.5 / 20.0 | 主干与 subject-token 起点 |
| NICE | 2024 (eeyhsong/NICE-EEG) | 时空 CNN + CLIP 对齐 | 弱 | 预处理/评测脚手架 |
| NeuroCLIP | arXiv 2511.09250 | **prompt tuning** + 改良对比损失 | 17.0 / 40.3 | prompt 形式注入被试条件 |
| NeuroBridge | arXiv 2511.06836 | CPA + **双向语义对齐** | 19.0 / 45.9 | 双向 InfoNCE |
| SAMGA | ESWA 2026 (arXiv 2604.17782) | **subject-aware 多粒度视觉目标** + coarse-to-fine | 34.4 / 64.8（best-epoch） | 目标构造 + routed target |
| SCORE | 2026 (arXiv 2608.19134) | SAMGA + **免标签坐标恢复** | **53.23 / 83.55** | 测试时几何校准 |
| SATTC | CVPR 2026 (arXiv 2603.20738) | **SAW whitening + adaptive CSLS + PoE** | 14.8 / 38.4 | 免标签校准头 |
| SVTL | arXiv 2609.36971 | 结构化多视图目标 + **MMD** + 免训练 refinement | 35.3 / 65.6 → 48.1 / 77.1 | MMD + refinement |
| SUP-MCRL | arXiv 2606.16615 | **UEE 多尺度 atrous + 原型 EMA 池** | 24.0 / 52.9 | 抗坍缩原型 |
| ENIGMA | arXiv 2602.10361 | **subject-unified backbone + multi-subject latent alignment** | 15 min 校准新被试，<1% 参数 | 新被试快速适配范式 |
| NVOL | arXiv 2609.02582 | 逐层选择「神经可见最优层」 | 78.1 → 86.4 (CSLS) | 中间层对齐目标 |

### 2.2 被试建模（「被试即模态」的直接依据）

| 方法 | 机制 | 启示 |
|---|---|---|
| **SuLoRA** (arXiv 2510.08059) | 权重 = **共享全秩 + 被试专属低秩**；drop-in 替换 Linear/Conv | 「共享 + 被试低秩修正」的模板 |
| **Stacked LoRA** (arXiv 2607.03094) | Global adapter + Subject-Specific adapter 在同一 forward 内解耦 | 双路 adapter 训练策略 |
| **FiLM-EEG** (arXiv 2509.23247) | 用 subject embedding 做 **per-feature 仿射** \(h\leftarrow\gamma_s\odot h+\beta_s\) | 最轻量的被试调制 |
| **HyperEEGNet** (OpenReview 04RGjODVj3) | **超网络**从静息态生成分类器权重 | hypernetwork 范式 |
| 被试专属编码器 (arXiv 2606.16462) | 每被试一个 encoder 头 + 共享分类器 | 共享空间 + 个体投影 |
| CL-SSTER (arXiv 2402.14213) | **同一刺激、不同被试的 EEG 互为正样本**的 InfoNCE | 跨被试对比的可行性证据 |
| MindTuner / **MindAdapter** (arXiv 2605.24679) | 冻结对齐主干 + **轻量残差 adapter** 少样本校准 | 少样本校准防过拟合 |

**结论性判断（写进设计假设）**：
- 单纯「subject token 拼接」（ATM 做法）在 LOSO 下不够，需要**结构性解耦**（shared vs subject-specific 参数分离）——SuLoRA/Stacked-LoRA/被试专属编码器三篇独立指向同一结论。
- 跨被试对比（CL-SSTER）可作为**额外的中间监督**，与「对齐到 CLIP」形成互补：前者管「被试不变」，后者管「图像语义」。
- 顶端收益有一大块在**部署时几何**，方案必须显式包含且可消融。

---

## 3. 数据盘点（集群内实测，均已就绪）

### 3.1 EEG（THINGS-EEG2，10 被试）

```
/project/peilab/why/NeuroBridge/data/things_eeg/preprocessed_eeg/
├── info.json                       # ch_names(63), times(-0.2..1.0 @250Hz)
├── sub-01/train.npy                # (1654, 10, 4, 63, 250)  condition×image×rep×ch×time
├── sub-01/test.npy                 # (200, 1, 80, 63, 250)   → 平均后 (200,63,250)
└── sub-02..sub-10/                 # 同构
```

- 训练：1654 概念 × 10 图 × 4 重复；测试：200 概念 × 1 图 × 80 重复。
- 预处理口径（与 NICE/ATM/UBP/NeuroBridge/SAMGA/SCORE 一致）：0.1–100 Hz 带通 → 0–1000 ms epoch → **200 ms 刺激前基线校正** → 250 Hz → **MVNN**（仅训练集拟合）→ 重复平均。

**已缓存的派生 EEG**（可直接复用，省去重建）：
```
/project/peilab/why/eeg-retrieval/outputs/cache/
  eeg_{train,test}_sub{01..10}_17ch.npy            # (1654,10,17,250) / (200,1,17,250)
  eeg_{train,test}_sub{01..10}_all63.npy           # 63 导版本
  eeg_{train,test}_sub{01..10}_all63_std.npy       # 逐被试 z-score
  eeg_{train,test}_sub{01..10}_all63_mvnntrain_std.npy  # MVNN + std
  mvnn_W_sub{01..10}_all63_{train,test}_lw.npy     # 白化矩阵（可复现）
```
> 注意 MVNN 必须拟合在**未平均**的试次上（`PROTOCOL_INTER.md` §4 有严谨论证：白化矩阵线性、可与平均交换；必须在相关空间做 shrinkage）。

### 3.2 图像与图像特征（对齐目标）

| 资源 | 路径 | 形状 |
|---|---|---|
| 原图（训练/测试） | `/project/peilab/why/data/images_set/{training_images,test_images}/` | 16540 / 200 jpg |
| 元数据 | `/project/peilab/why/data/images_set/image_metadata.npy` | 文件名↔概念映射（行对齐基准） |
| 测试图像张量 | `.../test_images_tensor/all_images.pt` | 150 MB |
| **CLIP ViT-H-14 多层** | `/project/peilab/why/NeuroBridge/data/things_eeg/image_feature/clip_h14_multilevel/image_{train,test}_layer{20,24,28,32,36}.npy` | `(1654,10,1280)` / `(200,1,1280)` |
| **InternViT-6B 多层** | `.../image_feature/internvit_multilevel_20_24_28_32_36/` | `(1654,10,3200)` / `(200,1,3200)` |
| CLIP ViT-H-14 终层 | `.../image_feature/ViT-H-14/image_{train,test}.npy` | `(1654,10,·)` / `(200,1,·)` |
| IP-Adapter / RN50 / CogCap(depth,edge) | `.../image_feature/{clip_h14_ip_adapter,RN50,cogcap_depth,cogcap_edge}/` | 供重建分支 |
| 1024 维 CLIP 图像嵌入 | `/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_{train,test}_1024.npy` | `(16540,1024)` / `(200,1024)` |

**建议主对齐目标**：`internvit_multilevel` 或 `clip_h14_multilevel` 的**多层融合**（SAMGA/SVTL 均证明融合 > 单层；`PROTOCOL_INTER.md` §8 记录的 intra 探针也是 block24 附近最优）。

### 3.3 文本（可选辅助目标）

```
/project/peilab/why/data/captions/
  captions_shared_{train,test}.json      # 1654 / 200 条，长描述（SDXL prompt 风格）
  captions_detailed_{train,test}.json    # 更细粒度
  captions_sub08_{train,test}.json       # 单被试子集
```
可用于打造 **text-CLIP 辅助目标**（UCK 的经验：图像 CLIP 更细、文本 CLIP 更粗，二者互补）。

### 3.4 可复用的既有代码

- `/project/peilab/why/eeg-retrieval/scripts/epd/`：`data.py`、`mvnn.py`、`metrics.py`、`model.py`、`recover.py`、`losses.py` —— **LOSO 加载、MVNN、白化/CSLS、评测**已实现，直接复用，不要重写。
- `/project/peilab/why/Brain-HIVE/`：EEG↔视觉对比训练（HF Trainer）+ 融合先验 + SDXL 重建的完整参考。
- `/project/peilab/why/eeg-brainit/`：ATMS 编码器权重与 1024 维 EEG/图像嵌入缓存。

---

## 4. 方法设计：SAM-CLIP

### 4.1 形式化

- 被试集合 \(\mathcal{S}=\{1,\dots,S\}\)，每个被试数据 \(\mathcal{D}_s=\{(x_s^{(i)}, I^{(i)})\}_{i=1}^{n_s}\)。
- **共享主干** \(\;f_\theta(\cdot)\)：被试无关的时空表征。
- **被试条件** \(\;z_s\in\mathbb{R}^{d_z}\)：描述该被试的「观测算子」。
- **调制函数** \(\;\Phi(z_s)\)：把 \(z_s\) 变成注入主干的参数（FiLM 的 \(\gamma,\beta\) 与 LoRA 的 \(A,B\)）。
- **条件编码器**：\(\;e = f_\theta(x_s;\Phi(z_s))\)。
- **投影头** \(\;g_\psi(e)\to \mathbb{R}^{d_{\text{CLIP}}}\) 对齐到 CLIP 图像空间。
- **新被试推断**：\(\;z_s = H_\phi(\mathcal{S}_s)\)，\(\mathcal{S}_s\) 是少量 support（未标注或弱标注）。

> **「被试即模态」的正式含义**：不同的 \(z_s\) 定义了不同的观测算子 \(\Phi(z_s)\)，共享的 \(f_\theta\) 是模态无关的表征提取器。这与「把每个被试当成一个 domain/adapter」等价，但在表征上把「个体差异」显式地限制在低维、低秩的子空间里。

### 4.2 总体架构

```
                         ┌─────────────────────── 被试专属（少量参数） ───────────────────────┐
   support S_s ──► H_φ ──►  z_s ∈ R^{d_z} ──► [ FiLM heads: γ_l, β_l ]  +  [ LoRA: A_l, B_l ]
        (K trials)                          └──────────────┬─────────────────────────┬──────┘
                                                          │ (per-block affine)       │ (low-rank on QKV/proj)
   EEG x_s (63×250) ─► [ Patch/Temporal Conv ] ─► [ Channel-wise Transformer × L ] ─► [ Temporal-Spatial Conv ]
                                                          ▲                          │
                                                          └──────── θ (shared) ──────┘
                                                                                     │
                                                                     e ──► g_ψ ──► ê  (d=1024, L2-norm)
                                                                                     │
                        ┌────────────────────────────────────────────────────────────┼──────────────────┐
                        │                                │                           │                  │
                   L_img: InfoNCE                  L_cross: InfoNCE            L_adv / L_mmd      L_reg
                 (ê ↔ CLIP 图像目标)          (同图不同被试互为正样本)         (去被试信息/分布对齐)  (抗坍缩)
```

### 4.3 组件 A：共享 EEG 主干 \(f_\theta\)

**推荐主干（主方案）**：ATM/iTransformer 风格，与现有基建一致、参数小、在 THINGS-EEG2 上有公开可复现结果。

1. **前置时空卷积**（ShallowNet 风格）：`Conv2d(1→8,(1,64)) → BN → ELU → Conv2d(8→16,(63,1)) → BN → ELU → AvgPool(1,4)`，把 \(63\times250\) 降采样并提局部波形特征。
2. **Channel-wise Transformer**（iTransformer）：把每个通道时序作为 token，\(C=63\) 个 token，正弦位置编码，\(L\in\{1,2\}\) 层、\(d=200\)、4 头、channel-wise attention。
3. **Temporal-Spatial Conv**：`Conv2d(16,16,(1,16)) → BN → ELU → Conv2d(16,16,(1,16)) → BN → ELU → AvgPool(1,4)`（ATM 原设，参数少、抗过拟合）。
4. **MLP Projector**：残差块 ×M → Linear → LayerNorm，输出 \(e\in\mathbb{R}^{1024}\)。

**备选主干（消融）**：
- **CBraMod / LaBraM**：冻结的 EEG 基础模型 + 轻量 adapter。`AdaBrain-Bench` 显示 LaBraM/CBraMod 跨被试 macro 最好；`Channel Adaptation` 显示 CBraMod 的 ACPE 对导联变化更鲁棒。
- **EEGNet/ShallowNet**：作为最弱 baseline 下界。

> 主干选择本身**不是本方案的主要创新**，应作为消融项；创新点在 §4.4–§4.6。

### 4.4 组件 B：被试作为模态（subject-as-modality conditioning）

这是方法的核心。三种机制叠加，参数量从重到轻解耦：

#### 4.4.1 被试特征 \(z_s\) 的来源：**统计量锚 + 低秩学习修正**（2026-10-03 修订）

> **本节在此次修订中做了实质性改动，原因是一次可复现的失败。**
> 原设计用「训练被试走 `nn.Embedding` 表 / 新被试走 set-encoder」两条通路。实测发现：
> 两条通路的尺度差 ~30–200×（表 `|z|≈0.07`，support encoder `|z|≈2–4.5`），而共享的
> FiLM/hyper 头无法同时适配。试图用 `z_norm: unit`（归一化到 `|z|=√d_z=8`）调和时，
> **训练完全死亡**：`img` 在 12k 步内钉在 \(\ln 72 = 4.2767\)（对比损失的"零信息"下界），
> test top-1 恒为初始化的 0.50，而 HSIC 不减反增（0.483 → 0.517），即表征在变得更被试特异。
> 2×2 分离实验（两个 seed，`target_fusion ∈ {routed, routed_sr}` × `z_norm ∈ {none, unit}`）
> 结论唯一：**`z_norm: unit` 单独就足以杀死训练，`target_fusion` 无关**。
>
> 诊断是**结构性**的：一个自由学习的被试向量没有带宽上限，主干可以把它当作"被试代码"来用，
> 而不是去学被试无关的表征。放大尺度等于把这张空白支票的额度调大。

**文献共识**（三条独立证据都指向"统计量为主"）：

| 证据 | 内容 |
|---|---|
| **SATTC** (CVPR 2026) | **被试自适应白化（SAW）是跨被试增益的主要驱动**；且**全局白化反而比不自化更差**（6.5/23.0 vs 9.2/30.5）——只有"按被试"的统计量才有用 |
| **Euclidean Alignment** 冻结编码器实验 | 仅靠按被试的协方差重中心化，就在线性探测上 +14.82pp；而事后**线性抹除被试身份无效**（被试信息与任务信号纠缠） |
| **Latent Alignment** (BEETL 冠军) | 用被试自身试次的 per-feature 均值/方差做标准化，**被试专属参数为零**；可写成 deep set |
| **SuLoRA** / 被试专属编码器 | 有学习参数时也是**低带宽**的：每被试低秩修正 `r=1..16` 即可恢复全模型 |

**修订后的形式化**：

\[
z_s \;=\; W_{\text{stat}}\big[\;\overline{x}_c(\mathcal{S}_s)\;;\;\sigma_c(\mathcal{S}_s)\;\big] \;+\; H_\phi(\mathcal{S}_s)
\]

- 第一项是**统计量锚**：support 集每通道的均值与标准差（\(2C=126\) 维）经一个线性映射。
  它在第 0 步就提供了"这个被试是谁"的可迁移部分，无需学习；也把被试向量**限制在统计量的张成空间**里。
- 第二项是**低秩学习修正**：保留原有 permutation-invariant attention pooling 的 set-encoder，
  用于补充二阶统计量遗漏的任务相关变化。`support_anchor: none` 可还原旧行为做消融。

**单一估计器**：`z_source: support` 现在**真正生效**（此前它在构造时被校验、之后**从未被读取**——
`train.py` 走 ID 表、部署走 support encoder，正是上面那个 mismatch 的来源）。训练与推理现在用
**同一个估计器**，`z_norm` 因此不再需要（它存在的唯一理由就是调和两个估计器），默认改为 `none`。

**训练期即用 support 通路**（对应 SCORE 的 recovery-aware source training）：
每个 Stage-1 step 为每个源被试抽 K 个**无标签**试次作为 support，且**排除该 batch 自身占用的
`(concept, image)` 槽位**——部署时 target 的 support 来自其 train split、query 来自 test split，
两者不相交，训练期必须一致。SCORE 的核心发现是：**源被试之间对齐并不保证未见被试落在同一坐标系**，
所以未见被试的通路必须在训练期被走一遍。

\(H_\phi\) 的结构（**permutation-invariant attention pooling**，对 support 集合可交换、与 \(K\) 无关）：

```
z_s = Linear( MeanPool( MHA( Q=learned_query, K=V=MLP(f_θ(x_i)) ) ) )  +  W_stat[moments(S)]
```

#### 4.4.2 FiLM（逐特征仿射）

在第 \(l\) 个 Transformer/Conv block 前，对 LayerNorm 后的激活做仿射：
\[
h \leftarrow \gamma_l(z_s)\odot \mathrm{LN}(h) + \beta_l(z_s),\qquad
[\gamma_l;\beta_l]=\mathrm{MLP}_l(z_s)
\]
- 初始化 \(\gamma=1,\beta=0\)（恒等），保证训练初期不破坏主干。
- 成本：每层 \(2d_z\cdot 2d_h\)，可忽略。文献支持：FiLM-EEG（arXiv 2509.23247）在小样本 EEG 上优于 scalar projection。

#### 4.4.3 低秩被试 adapter（LoRA-style，冻结主干 + 修正）

对主干中的 Linear 权重 \(W_0\in\mathbb{R}^{d_{out}\times d_{in}}\)（QKV、FFN、projector）：
\[
W = W_0 + \Delta W_s,\qquad \Delta W_s = B(z_s)\,A(z_s),\quad A\in\mathbb{R}^{r\times d_{in}},\ B\in\mathbb{R}^{d_{out}\times r},\ r\in\{4,8\}
\]
- \(B\) 零初始化 → \(\Delta W_s=0\) 起步。
- **两种变体**（消融）：
  - **(a) 静态 LoRA**：每个被试一套 \(A_s,B_s\)（SuLoRA 做法，需已知被试身份）。
  - **(b) 超网络生成 LoRA**：\(A(z_s),B(z_s)\) 由 \(z_s\) 生成——**这才是「被试即模态」的完整形态**，支持未见被试。
- 关键设计：**共享全秩 \(W_0\) 永远激活，被试低秩修正只在需要时叠加**（SuLoRA 的核心结论：这样最抗漂移）。

#### 4.4.4 参数量与解耦

| 部分 | 归属 | 是否在校准时更新 |
|---|---|---|
| 主干 \(f_\theta\) | 共享 | ❌ 冻结 |
| 投影头 \(g_\psi\) | 共享 | ❌ 冻结 |
| 训练被试 embedding | 被试专属 | —（仅训练） |
| hypernetwork \(H_\phi\) | 共享（学「如何推断被试」） | ⚠️ 可选微调 |
| FiLM heads / 超网络 LoRA 生成器 | 共享 | ❌（生成器的输出是被试专属） |
| 新被试的 \(z_s\)（及可选 FiLM 直接参数） | 被试专属 | ✅ 仅此项 |

> 校准阶段**只训练被试专属的少数参数**（\(d_z\) 个 + 可选少量 FiLM）——这是「少量数据适配而不破坏全局几何」的关键（MindAdapter/SuLoRA 的一致结论）。

### 4.5 组件 C：跨被试对比对齐（去个体差异）

#### 4.5.1 采样：跨被试正样本

构造 batch 时保证**同一图像条件 \(i\) 至少被两个被试 \(a,b\) 观测**。以被试 \(a\) 的 \(x_a^{(i)}\) 为 anchor：
- **正样本**：被试 \(b\) 对同一图像的 \(x_b^{(i)}\)（跨被试正样本）；被试 \(a\) 的其它重复 \(x_a^{(i,r)}\)（时间不变性正样本）。
- **负样本**：其它图像 \(i'\ne i\) 的任意被试样本。

#### 4.5.2 损失：跨被试 InfoNCE
\[
\mathcal{L}_{\text{cross}} = -\frac{1}{|\mathcal{P}|}\sum_{(p,p^+)\in\mathcal{P}} \log \frac{\exp(\mathrm{sim}(\hat e_p,\hat e_{p^+})/\tau_c)}{\sum_{n\in\mathcal{N}(p)}\exp(\mathrm{sim}(\hat e_p,\hat e_n)/\tau_c)}
\]
其中 \(\hat e\) 为 L2 归一化嵌入。可扩展为 **multi-positive InfoNCE**（同图多被试/多重复全部计入正样本）。

文献支持：CL-SSTER（arXiv 2402.14213）用「同刺激跨被试为正、异刺激为负」的 InfoNCE 学到共享时空表征；这是本方案最直接的先例。

#### 4.5.3 去被试信息：对抗 / 去相关 / 分布对齐（三选一或组合）

目标是「保留图像信息、抹掉被试信息」。有三种机制，建议**主用 (2)，(1)(3) 作消融**：

1. **对抗（GRL）**：接一个 subject classifier，梯度反转，\(\mathcal{L}_{\text{adv}}\) 最大化被试分类熵。
   - ⚠️ 风险：被试信息与图像信息部分纠缠，强行抹除可能损失信号。**必须有 ablation**。
2. **去相关（推荐主用）**：\(\mathcal{L}_{\text{dec}}\) 用 **HSIC / 线性 CKA** 惩罚嵌入与被试 one-hot 的统计相关；比对抗稳定，不引入 min-max 训练不稳定。
3. **分布对齐**：\(\mathcal{L}_{\text{mmd}}\) 在源被试嵌入分布间做 **MMD / CORAL**（SVTL 用 MMD 降源被试分布差）。

### 4.6 组件 D：EEG–图像 CLIP 对齐（保留图像信息）

#### 4.6.1 目标构造（多粒度融合）

对齐目标不用单层：把冻结视觉编码器的多层特征**融合**（SAMGA / SVTL / Shallow Alignment 一致证明融合 > 单层）：
\[
t_I = \mathrm{Fuse}\big(\{u^{(\ell)}\}_{\ell\in\mathcal{L}}\big),\qquad \mathcal{L}\subseteq\{20,24,28,32,36\}\ (\text{CLIP-H14 或 InternViT})
\]
融合方式三选：`mean` / `single(best)` / **`routed`（subject-aware，SAMGA 的 \(b_s\)）**。建议主用 `routed`，其「训练时被试感知、推理时被试无关」的设计正是本方案的全景。

#### 4.6.2 损失：对称 InfoNCE（CLIP loss）
\[
\mathcal{L}_{\text{img}} = \tfrac12\big[\mathrm{CE}(\hat e\, t_I^\top/\tau_i,\ \mathrm{diag}) + \mathrm{CE}(\hat e\, t_I^\top{}^\top/\tau_i,\ \mathrm{diag})\big]
\]
- 可选 **text-CLIP 辅助**：以 caption（§3.3）的 CLIP 文本嵌入为额外目标，\(\mathcal{L}_{\text{txt}}\)（UCK 经验：图文互补，共用同一投影查询）。
- 温度 \(\tau_i\) 用可学习 `logit_scale`（初始 `ln(1/0.07)=2.6592`）。
  **必须设下界** \(s_{\min}=1.0\)（CLIP 自身的 `logit_scale.clamp(0, log 100)` 范围）。
  \(d\mathcal{L}/ds\) 是**非对角 logit 的均值**——一个与温度无关的 O(1) 量；Adam 按梯度幅度
  归一化，所以一旦它持续为负，就会在几十步内把尺度推到 0。而在 \(s=0\) 时所有 logit 为 0：
  对比损失**恰等于 \(\ln N\)**（不表示任何东西就能达到的最优值），且编码器梯度 ∝ \(s\) 而消失——
  编码器完全停止学习，而 `var`/`dec`/`mmd` 照常训练，所以它表现为"平台期"而不是"故障"。
  本 fold 实测：`img` 在 12k 步内钉在 \(\ln 72\)、test top-1 停在初始化值。
  温度现已**写入 checkpoint**（`crit` 键）——它是优化参数却住在 `Trainer` 上，
  只存 `model.state_dict()` 会静默丢弃它。

### 4.7 抗坍缩与总损失

小 batch + 强对齐 + 去被试惩罚极易**表征坍缩**。加入：
- **VICReg 式方差/协方差正则** \(\mathcal{L}_{\text{reg}}\)：hinge 惩罚每个维度的 std 下界 + 惩罚维度间协方差。
- **原型 EMA 池**（SUP-MCRL 的 PPA）：维护每概念的 EMA 伪特征，作为稳定的额外正样本，防坍缩。

**总损失**：
\[
\boxed{\ \mathcal{L}=\lambda_1\mathcal{L}_{\text{img}} + \lambda_2\mathcal{L}_{\text{cross}} + \lambda_3\mathcal{L}_{\text{dec}} + \lambda_4\mathcal{L}_{\text{mmd}} + \lambda_5\mathcal{L}_{\text{reg}} + \lambda_6\mathcal{L}_{\text{txt}}\ }
\]
初始建议 \(\lambda=(1.0,\ 0.5,\ 0.2,\ 0.1,\ 0.5,\ 0.2)\)，再按 §9 消融调参。\(\mathcal{L}_{\text{adv}}\) 默认关闭（\(\lambda_{\text{adv}}=0\)），作为消融臂。

---

## 5. 训练策略

### Stage 0 — 预处理与缓存（一次性，可复现）

1. 用 `epd/mvnn.py` 在**未平均**训练试次上拟合 MVNN，应用到平均后的 train/test。
2. **逐被试** z-score（用该被试自己的训练统计量；**不要**9 个被试拼一起用全局统计量）。
3. 可选 **Euclidean Alignment**（EA）：对每个被试试次做参考协方差白化。注意 SCORE 在强编码器上实测 EA 只 +1.23 Top-1 且不显著——**当作可选正则，不作核心卖点**。
4. 加载已有图像特征缓存（§3.2），或按需生成多层融合目标。
5. 生成 manifest（行↔概念↔文件名对齐证明），复用 `PROTOCOL_INTER.md` §5 的对齐校验。

### Stage 1 — 多被试 subject-conditioned 预训练

- **数据**：9 个源被试，**全部 1654 概念**、每图 10 图、4 重复（**不设验证留出**，`--val-concepts 0`）。
- **被试条件**：`z_source: support`（默认）。每个 step 为每个源被试抽 K=10 个无标签试次作为
  support（排除该 batch 自身的槽位），经 §4.4.1 的 `统计量锚 + set-encoder` 得到 \(z_s\)，
  再以 `z=z_all[subject]` 注入。**不再使用 `nn.Embedding` 表**（保留为 `z_source: ids` 消融臂）。
  - 这一步同时完成了三件事：① 让训练与部署用同一个 \(z_s\) 估计器；② 让主干学到"从 K 个试次
    推断被试"而非"查表"（Stage 2 的元训练因此变成可选精调而非必需）；③ 实现了 SCORE 的
    source-only recovery episode 思想。
- **目标**：\(\mathcal{L}\)（§4.7），其中 \(\lambda_{\text{adv}}=0\)。
- **采样**：iterable dataset，每 batch 保证 ≥2 被试共享若干图像条件。
- **优化**：AdamW，lr \(1\text{e-}3\)（主干） / \(1\text{e-}2\)（被试条件与 hypernet），wd \(1\text{e-}4\)，cosine，warmup 5%，epochs 50（对齐 SCORE/SAMGA 发布代码口径），grad clip 1.0。
- **选点**：**final epoch**（防测试集泄漏，见 `PROTOCOL_INTER.md` §3）。
- **产物**：共享主干 \(f_\theta\)、投影头 \(g_\psi\)、\(H_\phi\)、FiLM/LoRA 生成器。

> ⚠️ **`clip_alignment_loss` 必须保持对角形式**（不要在 `z_target` 上套 multi-positive mask）。
> 曾怀疑"batch 内 g 个被试共享同一图像 target"会把其余 g−1 个相同向量当作负类，因而需要 mask。
> 该怀疑已被**证明为错**：当组内各列（target 向量）完全相同时，\(A_{ij}=f(i)\) 与 \(j\) 无关，故
> \(\sum_{j\in G}\text{mean}_{i\in G}A_{ij}=\sum_{j\in G}A_{jj}\)，两个方向的 grouped loss 与
> diagonal loss **逐位相等**（实测差 1.19e−7）。`smoke_test.py` 9f-b 固化了这条恒等式，
> 布局一旦改变就会报警，而不是让这条"结论"悄悄失效。
> 但该布局确实有一个**无法用 mask 消除**的后果：g 个等价正类使两个方向的下界都是 \(\ln g\)，
> 即图像对比在本 fold 上**在 \(\ln 9\) 处饱和**。这正是温度必须**设下界**的原因（见下）。

### Stage 2 — Episodic 元训练（让 \(H_\phi\) 学会「从 K 个试次推断被试」）

这是让「用少量数据学被试特征」真正可行的关键，**不可省略**。

- **Episode 构造**：从源被试中抽一个作为「模拟新被试」\(s^*\)，其余作为「参考被试」。从 \(s^*\) 采 **support 集 \(\mathcal{S}_{s^*}\)（K 个试次，K∈{5,10,20}）** 与 **query 集 \(\mathcal{Q}_{s^*}\)**。
- **内循环**：\(z_{s^*}=H_\phi(\mathcal{S}_{s^*})\)（可选：对 \(z_{s^*}\) 做几步梯度）。
- **外循环**：在 query 上算 \(\mathcal{L}_{\text{img}}+\mathcal{L}_{\text{cross}}\)，只更新 \(H_\phi\)（与可选 FiLM 生成器），主干冻结（或极小 lr）。
- **算法选择**：从简到繁——(1) **episodic + Reptile**（最稳）；(2) **MAML/FO-MAML**（若显存允许二阶）；(3) **ProtoNet 式**（\(z_{s^*}\) 直接由 support 原型初始化）。
- **验证**：跨被试 episode 的 query Top-1 作为选点信号（此时可用源被试的留出概念，**绝不碰 200 测试概念**）。

### Stage 3 — 新被试少样本校准（部署）

见 §6。

---

## 6. 新被试少样本校准

**目标**：给定新被试 \(s^\star\) 的 \(N\) 个试次（\(N\in\{5,10,20,50,100\}\)），使其检索性能尽量接近其 full-calibration 上界。

**协议（两阶段，借 SATTC 的两相部署）**：

| 阶段 | 做什么 | 用不用标签 |
|---|---|---|
| **Phase 1 — 校准** | 用 \(\mathcal{S}_{s^\star}\) 推断 \(z_{s^\star}=H_\phi(\mathcal{S}_{s^\star})\)；可选对 \(z_{s^\star}\) 做 K 步无监督/弱监督优化；估计几何统计量（§7） | 默认**免标签**；可给出「N 个配对」的有监督变体作对照 |
| **Phase 2 — 推理** | 冻结全部参数，逐试次对冻结的相似度结构做矩阵向量查询 | 无 |

**只更新被试专属参数**：\(z_{s^\star}\)（\(d_z\) 维）+ 可选 FiLM 直接参数。**冻结**主干、投影、hypernet。这符合 MindAdapter（冻结对齐主干 + 轻量残差 adapter）与 SuLoRA 的结论：**避免过拟合、保留全局几何**。

**度量**：报告 \(N\) 的校准曲线，并给出 **N=50 恢复到 full 的百分比**（SATTC 的对照：N=50 已恢复 94.8% Top-5）。

---

## 7. 部署时几何校准（独立模块，强烈建议纳入）

> **为什么必须有**：`PROTOCOL_INTER.md` §7 的核心判断——SCORE 在 SAMGA 编码器之上、两者都冻结、且**无目标标签**的情况下，把 Top-1 从 26.22 拉到 53.23。顶端的收益主要来自**几何**，不是编码器。忽视这一层就是在优化小项。

三个即插即用、**只作用于相似度矩阵**的组件（均为免标签）：

1. **标准化推理**：L2 归一化嵌入 + 余弦相似度 + **candidate whitening**。SATTC 证明仅此一步就把冻结 ATM 特征从 5.5/20.0 提到 9.2/30.5。
2. **Subject-Adaptive Whitening (SAW)**：用目标被试**未标注** EEG 嵌入估计均值/协方差，白化其坐标。
3. **Adaptive CSLS**：基于行列局部密度做跨域相似度局部缩放，缓解 hubness。
4. **（可选）Coordinate Recovery**：估计一个正交映射，把被试识别出的坐标框架对齐到图像空间（SCORE 的核心；可视为 Procrustes/正交修正）。

实现可直接复用 `epd/recover.py` 与 `epd/metrics.py`。**本模块作为独立开关**，报告「有/无」两组数，确保创新点与几何收益可分离归因。

---

## 8. 评测协议

严格对齐 `PROTOCOL_INTER.md`，保证与公开榜单**可直接相减**。

| 项 | intra-subject | **inter-subject (LOSO)** |
|---|---|---|
| 训练 | 单个被试 | **9 个源被试** |
| 测试 | 同一被试 | **1 个留出被试** |
| 导联 | 17（OP） | **63（全部）** |
| 概念 | 全 1654 训练概念，无验证留出 | 同 |
| 检索 | 200-way，对角线为正 | 同 |
| 选点 | final epoch | final epoch |

- **指标**：Top-1 / Top-5 / mean rank（随机 mean rank = 100.5）。
- **重复平均**：train 4 次、test 80 次均平均。
- **随机性**：≥3 seeds，报告 mean ± std（对照 SAMGA 5 seeds、SCORE 3 seeds 的方差）。
- **必须报告**：单折 sub-08（唯一有公开可比 cell：SAMGA Table 2 = 28.7/59.5）。
- **禁止**：用测试集选 checkpoint（SAMGA 发布代码有 `--early_stop_patience` 即测试集选点，我们**不复制**）。

**对照基线（THINGS-EEG2，inter，63ch，200-way）**：

| 方法 | Top-1 | Top-5 | 备注 |
|---|---:|---:|---|
| ATM (NeurIPS 2024) | 5.5 | 20.0 | 原始跨被试基线 |
| ATM + 标准化推理 | 9.2 | 30.5 | 仅几何 |
| SATTC (CVPR 2026) | 14.8 | 38.4 | + 免标签校准头 |
| NeuroBridge | 19.0 | 45.9 | |
| SUP-MCRL | 24.0 | 52.9 | |
| SAMGA encoder（SCORE 重测，final-epoch） | 26.22 ± 1.08 | 57.98 ± 0.88 | 与我们对齐的口径 |
| SAMGA (ESWA 2026, best-epoch) | 34.4 | 64.8 | 注：测试集选点 |
| SVTL | 35.3 | 65.6 | + 免训练 refinement → 48.1 / 77.1 |
| **SCORE (2026)** | **53.23 ± 1.62** | **83.55 ± 1.13** | **要打败的目标** |

---

## 9. 实验矩阵与消融

### 9.1 主实验

1. **LOSO 全 10 折 × 3 seeds**：intra（17ch）+ inter（63ch）。
2. **few-shot 校准曲线**：\(N\in\{5,10,20,50,100\}\)，报告 Top-1/Top-5 与恢复百分比。
3. **单折 sub-08×seed2025**：与 SAMGA 公开 cell 直接对照。

### 9.2 消融（每一项回答一个「是否有效」）

| # | 消融 | 问题 |
|---|---|---|
| A1 | 无被试条件（纯共享主干） | 被试建模是否必要 |
| A2 | subject-token 拼接（ATM 式） vs FiLM vs LoRA vs FiLM+LoRA | **哪种调制最好** |
| A3 | 超网络生成 LoRA vs 静态 per-subject LoRA | 支持未见被试的价值 |
| A4 | \(H_\phi\)：有无 Stage 2 元训练 | 少样本推断是否真的学会 |
| A5 | \(\lambda_2=0\)（无跨被试对比） | 跨被试对齐的贡献 |
| A6 | 去被试：无 / HSIC / MMD / GRL | **去个体差异的最优机制** |
| A7 | \(\lambda_5=0\)（无抗坍缩） | 是否坍缩 |
| A8 | 目标：单层 vs 多层 mean vs routed | 目标构造的贡献 |
| A9 | 视觉编码器：CLIP-H14 vs InternViT | 目标空间选择 |
| A10 | 主干：ATM vs CBraMod vs LaBraM | 主干选择 |
| A11 | 几何校准 on/off（§7） | 创新点与几何收益的分离 |
| A12 | 有无 text-CLIP 辅助目标 | 文本监督价值 |

### 9.3 诊断（不只是刷点）

- **表征几何**：CKA / RSA（`EEG-FM-Bench` 与 generalization 框架的做法），检验跨被试表征是否真的对齐。
- **被试可分性**：线性 probe 从共享嵌入预测被试身份（越低越说明去个体化成功）。
- **Hubness**：特征/相似度分布、per-class 检索方差（SATTC 关注）。
- **信息含量**：共享嵌入对图像概念的线性 probe（确保「去被试」没有把图像信息一起抹掉）。

---

## 10. 实现路线图

### 10.1 目录（在 `CLIP/` 下，遵守 `AGENTS.md` §1）

```
CLIP/
├── configs/
│   ├── default.yaml
│   ├── loso_s1.yaml              # Stage 1
│   ├── meta_s2.yaml              # Stage 2
│   └── calib_s3.yaml             # Stage 3
├── src/samclip/
│   ├── data/
│   │   ├── things_eeg.py         # 复用 epd/data.py 的 LOSO 加载
│   │   ├── mvnn.py               # 复用 epd/mvnn.py
│   │   └── cross_subject_sampler.py   # 保证同图跨被试 batch
│   ├── models/
│   │   ├── backbone.py           # ATM/iTransformer + ShallowNet 前端
│   │   ├── subject_conditioning.py   # FiLM + LoRA + hypernetwork H_φ
│   │   └── samclip.py            # 组装 + 投影头
│   ├── losses/
│   │   ├── contrastive.py        # img / cross / text InfoNCE
│   │   ├── invariance.py         # HSIC / MMD / GRL
│   │   └── regularizers.py       # VICReg / 原型 EMA
│   ├── calibrate.py              # Stage 3 + 几何校准
│   └── eval.py                   # LOSO / few-shot / Top-k / mean rank
├── scripts/
│   ├── build_cache.py
│   ├── run_stage1.py
│   ├── run_stage2_meta.py
│   ├── run_calib.py
│   └── run_loso.py
└── slurm/
    ├── stage1.sbatch
    ├── meta.sbatch
    └── loso.sbatch
```

### 10.2 里程碑（每个都可独立交付、可中断恢复）

| M | 内容 | 交付物 | 依赖 |
|---|---|---|---|
| M0 | 数据缓存 + manifest + 对齐校验 | `data/manifest.jsonl`，形状/行对齐断言通过 | `epd/` |
| M1 | 主干 + 投影 + **无被试条件** baseline（inter, sub-08 单折） | 复现 ATM 量级 baseline | M0 |
| M2 | 加 **FiLM 被试条件** + \(\mathcal{L}_{\text{img}}\) | 第一版 SAM-CLIP | M1 |
| M3 | 加 **跨被试 InfoNCE** + 抗坍缩 | A5/A7 消融 | M2 |
| M4 | 加 **超网络 \(H_\phi\)** + **Stage 2 元训练** | few-shot 曲线 | M3 |
| M5 | 加 **几何校准**（§7） | A11；对齐 SCORE 口径 | M4 |
| M6 | 全 10 折 × 3 seeds + 全部消融 + 诊断 | 论文级结果表 | M5 |

### 10.3 Slurm 与工程约定

- 一律 `--partition=normal --account=peilab`，排除 `dgx-09,dgx-11,dgx-17,dgx-30`。
- **每个 stage 必须 `[SKIP]` 已存在的输出**（可重入），日志落 `outputs/slurm/`。
- 启动时打印完整 config（subject fold、seed、dims、λ、目标层、data hash）。
- 缓存变量（§2.4）在 import torch/HF 前导出；`TMPDIR=/tmp`。
- 显存以 H800 80GB 为准，主干很小，瓶颈在 batch 内跨被试采样与图像特征加载。
- **`/project` 仅剩 ~59 GB**：EP 结果与 checkpoint 要限额保留（只留 final + best-of-3 seeds），中间嵌入能删则删。

---

## 11. 风险与权衡

| 风险 | 说明 | 缓解 |
|---|---|---|
| **去被试信息伤到图像信息** | 对抗/去相关可能抹掉有用信号 | 主用 HSIC（比 GRL 稳）；用 §9.3 的图像 probe 监控；GRL 仅作消融 |
| **hypernet 少样本不可辨识** | K=5 试次可能不足以确定 \(z_s\) | Stage 2 元训练 + 低维 \(z_s\)(64) + 原型初始化；报告随 N 的曲线 |
| **表征坍缩** | 强对齐 + 小 batch | VICReg + 原型 EMA + 温度调参（A7 必测） |
| **泄漏** | 测试概念/测试集选点 | final-epoch；概念级留出只用于元训练选点；`assert` 概念交集为空 |
| **几何收益掩盖创新贡献** | 读者会问「是不是只靠 CSLS」 | §7 严格 on/off 分离报告；标准化推理基线单独列出 |
| **口径不可比** | best-epoch vs final-epoch 差 8 分 | 与 SCORE 的 final-epoch 行对齐；SAMGA 行标注 best-epoch |
| **存储** | `/project` 仅 ~59 GB | 缓存复用、不重复下载模型（§4 shared cache） |
| **MVNN 误用** | 必须拟合在未平均试次、相关空间 shrinkage | 复用 `epd/mvnn.py` + 其单元测试 |

---

## 12. 与现有基建的关系（避免重复造轮子）

| 已有资产 | 复用于 |
|---|---|
| `eeg-retrieval/scripts/epd/{data,mvnn,metrics,recover,losses}.py` | LOSO 加载、MVNN、白化/CSLS/评测、对比损失 |
| `eeg-retrieval/PROTOCOL_INTER.md` | 协议、选点、目标层开放的实证记录 |
| `Brain-HIVE/` | EEG↔视觉对比训练脚手架 + 融合先验 + SDXL 重建 |
| `eeg-brainit/outputs/atm_bridge/*_1024.npy` | 现成的 ATMS EEG/CLIP 嵌入（快速 sanity check 用） |
| 共享缓存 `cache/{open_clip,huggingface,torch}` | CLIP/SigLIP/DINOv2 backbone，**勿重复下载** |

**本方案的净新增**只在：`subject_conditioning.py`（FiLM+LoRA+hypernet）、`cross_subject_sampler.py`、`invariance.py`、`regularizers.py`、以及 §7 的校准开关。

---

## 13.5 修订记录 — 2026-10-03：被试特征提取的重新设计

> 触发：原设计在 sub-08 单折上**训练完全死亡**（`img` 钉在 \(\ln 72 = 4.2767\)，top-1 恒为
> 初始化值）。下面是根因、据此做的架构改动，以及每一项对应的验证。

### 13.5.1 根因（一）：**自由被试向量没有带宽上限**

| 假设 | 实验 | 结论 |
|---|---|---|
| `target_fusion: routed_sr` 有问题 | 2×2（2 seed × {routed, routed_sr}） | **否**。两个 arm 在 `z_norm: none` 下都正常（top-1 5.00% / 3.50%） |
| `z_norm: unit`（→ `√d_z`）有问题 | 同上，{none, unit} | **是**。`unit` 单独就让两个 seed 全部死亡（top-1 0.50%，meanrank 100.5 = 随机），与 fusion 无关 |
| 温度塌陷 | 12k 步轨迹 | **是**。`img` 从第 1 步的 4.2767 起 12k 步不动；`sc_img` 从 2.73 涨到 3.7 = 温度在往**下**走 |

**机制**：自由 embedding 的 `|z|` 从 0.07 被放大到 `√64=8`（114×）。放大本身不是问题，
问题是**被动放大也会把调制的有效学习率放大**（见 §13.5.3 的机理）。结果不是"训练变慢"，
而是主干被诱导去**超专化被试身份**（`dec` 从 0.483 升到 0.517，方向反了），跨被试图像对齐被
阻断。温度下界（§4.6.2）是**第二重**保险：即使对齐被阻断，`s → 0` 也会让对比梯度消失，
从而把"故障"伪装成"平台期"。

### 13.5.2 架构改动（已验证）

| # | 改动 | 依据 | 验证 |
|---|---|---|---|
| 1 | `z_s = W_stat[mean_c(S) ; std_c(S)] + H_φ(S)`，**统计量锚 + 低秩残差** | SATTC（SAW 是主驱动、全局白化更差）、Latent Alignment（零被试参数）、SuLoRA（r=1..16） | smoke 9c-b：锚点存在性、moments 的置换不变性与 K 无关性、锚确实改变函数 |
| 2 | `z_source: support` **真正生效**（此前被校验后从未读取） | SCORE：源被试互相对齐**不保证**未见被试落在同一坐标系 | smoke 2：dropout 现在覆盖 support/z 通路 |
| 3 | Stage 1 改为 support 通路，每 step 抽 K 个**无标签**试次，且**排除该 batch 自身槽位** | SCORE 的 source-only episode；部署时 support/query 不相交 | smoke 8b：形状、**零泄漏**、support encoder 在第 1 步开始收到梯度 |
| 4 | `z_norm` 默认改 `none` | 单一估计器后已无尺度可调和 | smoke 9c：旧 config 缺键时仍还原旧行为 |
| 5 | `cond_dropout` 覆盖**所有**条件通路 | 原实现只在 ID 路径生效 → `z_source: support` 下无条件通路永不出现 | smoke 2b：support_x 与 z 两条通路 |
| 6 | 温度写入 checkpoint（`crit` 键） | 温度是优化参数却住在 `Trainer` 上，只存 `model.state_dict()` 会静默丢弃 | smoke 9e |

### 13.5.3 根因（二）：**条件化的输入尺度**，而非 support 通路本身

3 臂探针（job 637386）显示：`z_source: support` 的**两个 anchor 臂都死了**（`img` 钉在
4.2767，top-1 0.50 = 随机），而 `ids` 对照存活。且死臂的 `dec` 从 0.486 **升到** 0.516——
与 `z_norm: unit` 死臂**同一个签名**。所以真正的病根还没找到，必须先区分两个机制：

| 假设 | 预测 |
|---|---|
| **H1（尺度）** | 条件化输入太大。`z_from_support` 返回原始输出 `|z|≈4.5`（ID 表是 0.16）。FiLM 末层零初始化，其输入 `GELU(Linear(z))` 的幅度——以及调制项的梯度——正比于 `|z|`。缩小 `z` 应能复活 |
| **H2（噪声）** | support 每步重抽 → `z` 每步抖动 → 调制变成噪声。冻结 support 应能复活 |

`scripts/diag_zscale.py`（job 637396）四臂 × 400 步，**只**改变 `z` 的产生方式：

| arm | `z` 来源 | \|z\| | `img` | `dec` | top-1 |
|---|---|---|---|---|---|
| A | ID 表 | **0.164** | 4.274 → 3.657 | 0.486→0.22 | **3.50** ✅ |
| B1 | support 每步重抽 | 12.28 | 4.275 → 4.266 | 0.486→**0.51** | 0.50 ❌ |
| B2 | support **固定** | 12.87 | 4.275 → 4.282 | 0.486→0.46 | 0.50 ❌ |
| B3 | support 每步重抽 **×0.03** | **0.139** | 4.275 → 3.619 | 0.486→**0.22** | **1.50** ✅ |

**结论：H1 成立，H2 被否证。** 固定 support 集完全无效（B2 仍死）；把 `|z|` 降到与 ID 表同
量级即复活（B3），且 `dec` 的**方向翻转**（0.51 → 0.22）——这才是最有信息量的信号。
B1 中 `|z|` 还从 4.5 **发散到 12.3**，所以固定乘数也不是稳定解。

**机理**：`film_head` 计算 `W₂·GELU(W₁z+b₁)+b₂`，且 `W₂=b₂=0`（零初始化 → 初始恒等）。
Adam 的步长对梯度幅度近似不变，但**同样的 `W₂` 步长产生的 `dgamma` 变化 ∝ |z|**——
所以 `|z|` 直接决定调制的**有效学习率**。`|z|≈12` 时调制失控，主干无法收敛到它上面。

**因此 `z_norm` 原本就是病因而不是解药**：它归一到 `√d_z`（来自 CLIP 的 logit-scale 约定），
`√64 = 8` **恰好落在坏区间内**。修订为：

```
z_norm: mark   →   z ← F.normalize(z) · (EMBED_INIT_STD · √d_z),  EMBED_INIT_STD = 0.02
```

即**对齐到 embedding 表自身的初始化尺度**（`0.02·√64 = 0.16`），由测量定标而非推导。
`z_norm: unit` 现在是**硬报错**（附实测证据），因为保留一个已知会杀死训练的选项比让它响亮失败更糟。
`normalize_z` 的 docstring 与 `configs/default.yaml` 都记录了这四臂表。

### 13.5.4 修复与验证（job 637403，1 epoch × 3 臂，其余完全相同）

```
z_norm: mark   →   z ← F.normalize(z) · (EMBED_INIT_STD · √d_z),  EMBED_INIT_STD = 0.02
```

即**对齐到 embedding 表自身的初始化尺度**（`0.02·√64 = 0.16`），由测量定标而非推导。
`z_norm: unit` 现在是**硬报错**（附实测证据），因为保留一个已知会杀死训练的选项比让它响亮失败更糟。

| arm | `z_source` | `anchor` | top-1 | top-5 | meanrank | 结论 |
|---|---|---|---|---|---|---|
| A | `ids` | stats | 2.00 | 15.00 | 33.8 | 对照 |
| B | `support` | **none** | **8.00** | **23.00** | **28.4** | ✅ 新默认 |
| C | `support` | stats | 5.50 | 17.00 | 32.8 | 锚点无增益 |

**两个 support 臂都复活，且都超过了 ID 表对照**（`img` 4.275→~3.4，`dec` 0.486→~0.20）。
这说明 support 通路**不只是可行，而且更好**——因为训练用的正是部署时要走的那条路。
在 1 epoch 尺度上，B vs C 的 2.5 分差（200 个测试项里约 7 项）不显著，且两臂的 `img`/`dec`
轨迹几乎重合，所以：

**决定：`support_anchor` 默认改为 `none`，`stats` 保留为消融臂。**
理由不是"统计量不重要"，而是**它已经被用在最该用的地方**——评测路径用
`calibration.saw_whiten` 在冻结嵌入上做被试自适应白化，这正是文献所归功的那种形式。
在条件化模块里再加一条统计量通道，目前**没有可测收益**；而它原本想防的发散问题，现在由
`|z|` 的有界归一化解决——那是一个**已验证**的修复，胜过**假设**的修复。
`stats` 的剩余主张（长期训练中对 `z_s` **方向**的带宽限制）需要长跑才能检验，故留在消融矩阵里。

### 13.5.6 新发现：学习的 set-encoder **坍塌为与被试无关的常数**

DAG 端到端验证（job 637413→637416，2 epoch / 40 episode）**无崩溃**，Stage 1 正常
（top-1 6.00，`dec` 下降）。但 Stage 2 自身的诊断报警，`scripts/diag_zcollapse.py` 确认：

| 测量 | 值 | 含义 |
|---|---|---|
| 同一被试、不同 support 集的 `cos(z_i, z_j)` | 0.9999 | — |
| **不同被试**的 `cos(z_i, z_j)` | 0.9999 | — |
| 分离度（same − different） | **−0.00000** | `z_s` 完全不含被试信息 |
| `\|z\|`（归一化前） | 36.48 | 尺度巨大，但**方向恒定** |

**这是结构性缺陷，不是 bug。** 目标函数里**没有任何一项**要求 `z_s` 携带被试信息，而
`dec`(HSIC) 与 `mmd` **主动奖励**表征变得被试无关——一个常数的 `z_s` 会让 `dec` 平凡地
最优。所以"坍塌为常数"是子问题的最优解，而不是训练出了错。

这解释了评测表里为什么 `conditioned` 仍优于 `shared` 却毫无被试特异性：**常数的 FiLM
调制仍然是一个可学习的常数重参数化**，它能改善分数，但并不做被试适配。

> 这也回答了本节的原始问题：**被试特征不能只是"接一个 set-encoder 然后等它学"**。
> 必须让目标函数**要求** `z_s` 有信息量。

**候选修法**（按文献支持度）：

1. **统计量锚改为"条件化本体"而非残差**（Latent Alignment 式，被试专属参数为零）。
   统计量**不可能**坍塌——它由输入直接算出。最稳，且与 §13.5.4 的测量不冲突：
   该测量否定的是"残差加上去"，不是"统计量作为本体"。
2. **匹配/错配对比目标**（SCORE 的 recovery episode 直接变成损失）：要求"用被试自己的
   `z_s` 去解码其试次"严格优于"用另一个被试的 `z_s`"。这正是 Stage 2 诊断在测的量
   （`gap`），把它从**诊断**变成**损失**。
3. **`z_s` 加被试辨识辅助头**（从 `z_s` 预测被试 id）。最便宜，且与"事后抹除被试身份
   无效"的负结果不矛盾——那条说的是共享**表征**里的被试信息，这里要求的是 `z_s` 里的。

### 13.5.5 判据复盘（DAG 验证链）

| 判据 | 结果 |
|---|---|
| `img` 是否离开 4.2767 | ✅ 3.390（B）/ 3.419（C） |
| `dec` 是否下降 | ✅ 0.486 → 0.196 / 0.197 |
| top-1 是否远离 0.50 | ✅ 8.00 / 5.50 |
| support 是否优于 ID 表 | ✅ 8.00 > 2.00 |

**下一步**：跑完整流水线（cache → stage1 → stage2 → eval），并把这四个机制
（`z_source ∈ {support, ids}`、`anchor ∈ {none, stats}`、`z_norm ∈ {mark, none}`、
`target_fusion ∈ {routed, routed_sr}`）纳入消融矩阵。注意本轮全部是 **1 epoch / 单 seed**
的机制验证，**不是**性能结论——正式结果必须跑满 `epochs` 并报 ≥3 seed。

---

## 13. 参考文献

**EEG-to-Image 对齐 / 检索**
1. Li et al. *Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion (ATM)*. NeurIPS 2024. arXiv:2403.07721
2. Song et al. *NICE-EEG*. 2024. github.com/eeyhsong/NICE-EEG
3. *NeuroCLIP: Brain-Inspired Prompt Tuning for EEG-to-Image Multimodal Contrastive Learning*. arXiv:2511.09250
4. *NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding*. arXiv:2511.06836
5. *SAMGA: Subject-Aware Multi-Granularity Alignment*. arXiv:2604.17782
6. *SCORE: Subject Coordinate Recovery for Label-Free Cross-Subject EEG-to-Image Retrieval*. arXiv:2608.19134
7. *SATTC: Structure-Aware Label-Free Test-Time Calibration*. CVPR 2026. arXiv:2603.20738
8. *SVTL: Structured Visual Target Learning for Cross-Subject EEG-to-Image Retrieval*. arXiv:2609.36971
9. *SUP-MCRL: Subject-aware Unified Pseudo-feature Coded MCRL*. arXiv:2606.16615
10. *ENIGMA: EEG-to-Image in 15 Minutes Using Less Than 1% of the Parameters*. arXiv:2602.10361
11. *NVOL: EEG-based Visual Retrieval and Reconstruction*. arXiv:2609.02582
12. *EEG-EditBench*. arXiv:2607.27857

**被试建模 / 少样本适配**
13. *SuLoRA: Subject-Specific Low-Rank Adapter*. arXiv:2510.08059
14. *Stacked LoRA for Subject-Adaptive EEG Foundation Models*. arXiv:2607.03094
15. *FiLM-based subject conditioning for EEG*. arXiv:2509.23247
16. *HyperEEGNet: Adaptive Weight Generation from Resting-State EEG using HyperNetworks*. OpenReview 04RGjODVj3
17. *Learning aligned EEG representations with subject-specific encoders*. arXiv:2606.16462 / Sci. Rep. 2026
18. *CL-SSTER: Contrastive Learning of Shared Spatiotemporal EEG Representations Across Individuals*. arXiv:2402.14213
19. *MindAdapter: Few-Shot Parameter-Efficient Residual Calibration*. arXiv:2605.24679
20. *MindTuner* (fMRI, LoRA subject fingerprints) — 见 MindAdapter 引用

**EEG 基础模型 / 评测**
21. Jiang et al. *LaBraM: Large Brain Model*. arXiv:2405.18765
22. Wang et al. *EEGPT*; Wang et al. *CBraMod*; Zhou et al. *CSBrain*
23. *AdaBrain-Bench: Benchmarking Brain Foundation Models for BCI*. arXiv:2507.09882
24. *EEG-FM-Bench*. arXiv:2508.17742
25. *Channel Adaptation for EEG Foundation Models*. arXiv:2604.23091

**数据集**
26. Gifford et al. *THINGS-EEG2*. OSF osf.io/3jk45 (CC-BY 4.0)；预处理 anp5v，图像 y63gw
27. Hebart et al. *THINGS database*. PLOS ONE 2019
28. *Brain-HIVE: Learning Brain Representation with Hierarchical Visual Embeddings*. (同账户参考实现)

---

## 附：一页速览（给评审）

- **做什么**：把被试当模态 → subject-conditioned 共享编码器 → 跨被试对比 + EEG–图像 CLIP 对齐 → 少样本校准。
- **创新点**：①「被试即模态」的超网络调制（FiLM+LoRA 统一形态）；②跨被试与跨图像**两级对比**解耦；③元训练 + 只调被试专属参数的校准范式；④几何校准作为可分离模块（诚实归因）。
- **主结果目标**：inter-subject LOSO，63ch，200-way，final-epoch —— 目标**超过 26.22（SAMGA encoder, final-epoch）**并挑战 **53.23（SCORE）**；few-shot N=50 恢复 ≥90% full。
- **最快验证路径**：M0→M1→M2 三周内可看出被试条件是否带来增益；M4 的 few-shot 曲线是方案成立与否的判据。
