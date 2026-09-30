# xENIGMA：在重建 SOTA 主干上叠加 inter 增强的架构设计

> **x**-**ENIGMA** = cross-subject ENIGMA
> 设计立场：**不自己造主干**。以 2026 年最新的「多被试 EEG→图像重建」SOTA（ENIGMA）为骨架，
> 只在其 inter 空洞处叠加跨被试增强组件。所有结论标注来源；未标来源者为**待验证假设**并附证伪条件。

本文取代 `DESIGN_CANVAS_v2_inter_reconstruction.md` 的**主干选择**部分（v2 用 SAMGA 检索编码器当主干，
本文改用重建 SOTA）——但 v2 的 §4.1 缺陷诊断、§4.4 诚实性协议、§5 评测协议**全部继续有效**，见 §10 的增量 diff。

---

## 0. 一句话

**ENIGMA 已经提供了跨被试重建所需的一切——除了「目标被试的坐标系」。**
`subject_wise_linear` 是逐被试硬门控，LOSO 下目标被试的键**根本不存在**（§2.1，代码级确认）。
而 SCORE 证明了「目标坐标系」可以用**无标签闭式解**直接恢复，且在同一个主干上值 +27.01 Top-1（§3.2）。

⇒ **xENIGMA = ENIGMA 主干（零改动）+ 目标被试坐标恢复（闭式、无标签）+ 三项训练侧对齐修正。**

---

## 1. 主干选择：为什么是 ENIGMA（「最新 SOTA」的核实过程）

### 1.1 2026-09 时点上的 EEG→图像重建方法时间线

| 论文 | 时间 | 官方代码 | 评价协议 | THINGS-EEG2 关键数字 | 可否当主干 |
|---|---|---|---|---|---|
| ATM (NeurIPS 2024) | 2024 | ✅ `dongyangli-del/EEG_Image_decode` | **被试内** | PixCorr 0.136 / SSIM 0.392 | ❌ 太旧 |
| Perceptogram | 2024 | ✅ | 单被试 | PixCorr 0.247 / SSIM 0.431 | ⚠️ 无跨被试 |
| **ENIGMA** | **2026-02** `arXiv:2602.10361` | ✅ `Alljoined/ENIGMA` | **多被试（30 被试联合）** | **PixCorr 0.1668 / SSIM 0.4264 / Alex(2) 82.99 / CLIP 80.33 / 人评 86.04%** | ✅ **本文选它** |
| Hierarchical Visual Embeddings | 2026-02 `arXiv:2602.07495` | ❌ 未见 | 被试内 + 被试平均 | PixCorr **0.195**（更高）/ SSIM 0.336（更低） | ❌ 无码、无跨被试 |
| CogCapPro | 2026-03 `arXiv:2603.12722` | ✅ `XiaoZhangYES/CognitionCapturerPro` | **被试内**（sub-08） | SSIM 0.409 / CLIP-H 2WC 0.903 | ⚠️ 见 §1.3 |
| SGDM | 2026-04 `arXiv:2604.22649` | ❌ **无代码** | 被试内 | 自称 SSIM/SwAV/CLIP 最优 | ❌ 无码 |
| NEED (NeurIPS 2025) | 2025-09 | ❌ 无代码 | 跨被试**视频** | SSIM 0.352（静态迁移） | ❌ 无码、非静态图像 |

**核实方法**：GitHub API 列出 `Alljoined/ENIGMA` 全树（`source/{models,training,dataset,utils}.py` + `train.py` +
`recon_inference.py` + `evaluate_recons.py` + 3 个 config）、读取论文 HTML 全文（含附录 A.1–A.8）。
CogCapPro 的代码可用性经官方 repo + 第三方复现 repo 双向确认；SGDM 的搜索结果只命中 BMVC 2023 的**同名无关**工作。

### 1.2 为什么是 ENIGMA，而不是更新的 CogCapPro

这是本节最需要说清楚的一个判断，因为 **CogCapPro（2026-03）比 ENIGMA（2026-02）更新**。

| 判据 | ENIGMA | CogCapPro |
|---|---|---|
| **是否跨被试重建** | ✅ **是**。论文主表就是 multi-subject vs single-subject | ❌ **否**。训练/评测都在**单个被试**内（sub-08 起家）；跨被试不在其协议里 |
| 是否多被试统一模型 | ✅ 30 被试共享 99% 参数 | ❌ 逐被试训练 |
| 与**我们现有生成栈**是否同构 | ✅ **逐项完全一致**（§1.3） | ⚠️ 同族但有 4 条模态分支 |
| 条件模态 | 图像 CLIP 1 条 | 图像+文本+深度+边缘 4 条 → 需 4 套 dataset/feature 管线 |
| 参数量 | 2.38M（多被试全 10 人） | 更大，且 ×被试数 |
| 它的「最新」体现在哪 | — | 多模态融合 + uncertainty-weighted masking + asymmetric alignment |

**结论**：CogCapPro 的「新」主要落在**多模态条件**，而不是**跨被试**。我们的课题是 inter 重建；
把主干换成 CogCapPro 会把工程量花在 4 条模态分支上，而**不解决目标被试坐标系缺失的问题**。

⇒ **主干 = ENIGMA；CogCapPro 的多模态条件作为 L4 可选增强（P2），不是主干。**

> ⚠️ 一个必须标注的数字陷阱：CogCapPro 报告的 `CLIP-H/14 2WC = 0.903` 与 ENIGMA 的 `CLIP = 80.33%`
> **不是同一个量**（被试内 vs 多被试、不同的干扰池与采样次数）。不能直接比，也不能拿它论证「CogCapPro 更强」。

### 1.3 ENIGMA 与 `eeg-brainit` 生成栈的逐项同构性（已核到权重名）

| 组件 | ENIGMA（`source/models.py: SDXL_Reconstructor`） | 我们 `eeg-brainit` | 一致？ |
|---|---|---|---|
| 扩散模型 | `AutoPipelineForText2Image` ← `stabilityai/sdxl-turbo`, `variant="fp16"` | SDXL-Turbo | ✅ |
| 推理步数 | `num_inference_steps=4` | 4 | ✅ |
| guidance | `guidance_scale=0.0` | 0.0 | ✅ |
| IP-Adapter | `h94/IP-Adapter` → `sdxl_models/ip-adapter_sdxl_vit-h.safetensors` | `h94/IP-Adapter` → `sdxl_models/ip-adapter_sdxl_vit-h.bin` | ✅ **同权重，仅序列化格式不同** |
| IP 尺度 | `set_ip_adapter_scale(1)`，`c_i=[embed]` | `set_ip_adapter_scale(1.0)` | ✅ |
| 图像编码器 | `CLIPVisionModelWithProjection("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")`, `size=224,crop=224` | 同权重 | ✅ |
| 条件维度 | `image_embeds` **1024-d** | `(N,1024)` npy | ✅ |
| VAE 精度 | `pipe.vae.to(torch.float32)`（防 fp16 溢出） | — | ✅ 我们已同处理 |

**这条同构性带来一个巨大的工程量优势**：我们已有的
`extract_clip_h14.py`、`erdc_full_metrics.py`、`erdc_twoway_metrics.py`、`clip_img_{train,test}_1024.npy`
**全部可直接复用，不需要任何重跑**。

### 1.4 ENIGMA 的诚实局限（写进论文 limitation，不要藏）

1. **ENIGMA 自己从未做过 LOSO**（§2.1/§2.2）。它的「跨被试」= 联合训练后**在训练过的被试上评测**，
   或**用 15 分钟带标签数据微调**新被试。**带标签微调 ≠ 我们要求的无标签留一。**
2. **谈不上绝对 SOTA**：Perceptogram 的 PixCorr 0.247 高于 ENIGMA 0.1668（单被试）；
   Hierarchical Visual Embeddings 的平均 PixCorr 0.195 也更高（SSIM 更低）。
   ENIGMA 的强项是**多被试配置下的全面领先**，这正是我们的场景。
3. **两篇论文对同一配置的数字不一致**：ENIGMA 论文 Table 1 报 Alljoined 多被试 Alex(2)=68.33%，
   而 Alljoined-1.6M 论文 Table 1 报 63.62%。复现时以**我们自己的跑分**为准。
4. **人评 86.04% 无法复现**（需要 545 名在线被试），我们只能报代理指标。

---

## 2. 代码级审计：ENIGMA 的 inter 空洞究竟在哪（设计起点）

以下每条都是**读代码得出的事实**，不是推测。

### 2.1 硬事实：LOSO 下 ENIGMA 根本跑不起来

`source/models.py: ENIGMA.__init__`：

```python
self.subject_wise_linear = nn.ModuleDict(
    {subject: nn.Linear(sequence_length, sequence_length) for subject in subjects}
)
```

`forward` 里：

```python
for subject in unique_subjects:
    subj_mask = torch.from_numpy(subjects == subject)
    x_eeg[subj_mask] = self.subject_wise_linear[subject](x[subj_mask])   # ← KeyError
```

* `subjects` 来自 `train.py` 的 `subj_ids` / `recon_inference.py` 的同名字段。
* 在 LOSO 下，目标被试不在 `subj_ids` 里 ⇒ **该 key 不存在 ⇒ `KeyError`，前向直接崩**。
* 这不是「性能下降」，是**结构性缺失**。

### 2.2 证明 ENIGMA 从未做过 LOSO

* `train.py`：`subjects = [f"sub-{s:02d}" for s in subj_ids]` → `ENIGMA(subjects=subjects)`。
* `recon_inference.py`：**用同一批 `subj_ids`** 建模型，然后 `load_state_dict(..., strict=False)`。
* ⇒ 评测的每个被试都在训练集里。所谓「多被试」= **联合训练 + 联合测试于同一批被试**。

> 这解释了一个表面矛盾：论文声称多被试配置 SOTA，代码里却没有任何留出被试的路径。

### 2.3 两处 paper/code 不一致（可白捡的收益，零成本）

**不一致 1：训练集没有做重复平均。**

* 论文 A.1 原文：「For training and inference, we average together these multiple trials for each image presentation
  to further boost the SNR of the data.」
* 代码 `source/dataset.py:80`：`if self.df_partition == "stim_test": self.average_eeg()`
  ⇒ **只有 test 被平均，train 没有。**

**实测确认**（`data/preprocessed_data/things_eeg2/sub-08/`）：

```
train preprocessed_eeg_data: (66160, 63, 250)
metadata stim_train rows: 66160, unique image_path: 16540  ⇒ 每图 4 次重复，未平均
test  16000 (=200×80) → average_eeg() → 200
```

**不一致 2：`W_s` 的维度不是论文说的那个。**

* 论文 §3.3：`W_s ∈ R^{N_z × N_z}`（在**隐空间** 184×184 上对齐）。
* 代码：`nn.Linear(sequence_length, sequence_length)` = `Linear(250, 250)`，作用于**原始输入的最后一维（时间轴）**，
  即一个**逐通道的时间混合矩阵**，不是隐空间对齐。
* 量级：250² = 62,500 参数/被试 ×10 = 625k，占论文宣称的多被试总量 2.38M 的 **26%**。

> 这给了我们一个**论文级卖点**：把 `W_s` 从「时间轴混合」改成「隐空间对齐 / 可闭式恢复的对齐」，
> 既更符合原论文声称的设计，又让它在 LOSO 下**有定义**。

### 2.4 ENIGMA 自己带着和 SAMGA-R 完全相同的 multi-positive 缺陷

`source/training.py: ClipLoss.forward`：

```python
labels = torch.arange(num_logits, device=device, dtype=torch.long)
total_loss = (F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)) / 2
```

`arange` ⇒ 第 $i$ 行只认第 $i$ 行匹配。而 `EEGDataset` 的 train split **按 trial 排布且不平均**，所以：

| 配置 | batch | 每图行数 | 每 batch 唯一图像数 | **同图互斥负样本占比** |
|---|---:|---:|---:|---:|
| 单被试 | 512 | 4（4 次重复） | 128 | **≈ 75%** |
| 多被试（9 源） | 512 | 4×9 = 36 | ≈ 14 | **≈ 97%** |

**即：多被试训练时，一个 batch 里 97% 的行是与自己共享同一张目标图像的其他被试行/重复行，
却被 InfoNCE 强制当作负样本推开。** 这与跨被试对齐的目标**直接冲突**。

> 这和我们在 `scripts/recon/train_gen_head.py:95-98` 里发现的缺陷 A 是**同一个病**，现在是
> **SOTA 官方代码里也有**。SCORE 的受控测量给出该项的增益：**+2.41 Top-1**（26.22→28.63）。

### 2.5 ENIGMA 与 SAMGA 共有的结构缺陷：空间坍缩

`source/models.py: Spatio_Temporal_CNN`，`conv2_kernel=(num_channels, 1)`：

```text
[B,1,63,250] → Conv2d(1,40,(1,5))            → [B,40,63,246]
             → AvgPool2d((1,17),(1,5))       → [B,40,63,46]
             → Conv2d(40,40,(63,1))          → [B,40, 1,46]   ← 63 个电极被核高 63 压成 1
             → Conv2d(40, 4, (1,1)) + Rearrange → [B, 184]
```

`hidden_dim=184 = 46 × 4`，**没有空间轴、没有电极拓扑、没有位置编码**——与 SAMGA 的 `TSConv` 同病。

⇒ 两篇论文（v2 §2.4 的四篇通道拓扑证据）指向同一个未被占据的杠杆：**通道角色分解**（前部=被试不变锚定、
后部=被试特异细节）。**注意这不只是 ENIGMA 的缺陷，也是它和 SAMGA 共同的天花板**，标记为 P3。

---

## 3. 设计逻辑：ENIGMA 给了什么、缺什么

### 3.1 缺什么，恰好只缺一样

| 组件 | LOSO 下是否可用 | 说明 |
|---|---|---|
| `tsencoder`（时空卷积） | ✅ 可迁移 | 共享参数，源被试训练 |
| `mlp_proj`（184→1024） | ✅ 可迁移 | 共享参数 |
| 输出空间 = CLIP ViT-H/14 1024-d | ✅ | 与 gallery 同空间 |
| **`subject_wise_linear[target]`** | ❌ **不存在（KeyError）** | **唯一的缺口** |

**唯一缺的是「目标被试的坐标系」。** 这是一个几何问题，不是容量问题。

### 3.2 为什么这个缺口可以闭式补上（SCORE 的实测证据）

SCORE（`arXiv:2608.19134`，2026-08）在**同一个 SAMGA 编码器、双方编码器全冻结、目标零标签**下测得：

| 部署方法 | Top-1 | Top-5 |
|---|---:|---:|
| None（裸用） | 26.22 ± 1.08 | 57.98 ± 0.88 |
| +CSLS | 35.78 ± 1.62 | 67.85 ± 1.21 |
| **+正交坐标恢复（闭式）** | **48.75 ± 1.25** | 80.93 ± 0.50 |
| +恢复感知训练 + 完整 SCORE | **53.23 ± 1.62** | **83.55 ± 1.13** |

**诊断依据**：源与被试的 EEG RSM 相关性 = 0.687 ± 0.035 ⇒ 概念**结构共享**；
用留出概念 3 折交叉验证把 target EEG 映到 source EEG：直接匹配 16.89 / Ridge 20.01 / **正交 28.22**。
**正交（约束更强）反而比 Ridge 高 8.21** ⇒ 该变换本质是**坐标旋转**，Ridge 的多余自由度在过拟合。

**泛化性证据（对我们最关键）**：冻结映射迁移到**不相交的 100 概念 gallery**：39.53 → **64.16**。
⇒ 恢复出的是**可复用的被试-图像关系**，不是对特定概念的拟合。

### 3.3 ENIGMA 相比 SAMGA 的一个天然优势：恢复是「原生维度」的

v2 设计里最纠结的问题（§4.3）是「在 SAMGA 512-d 隐空间拟合，还是在 CLIP 1024-d 拟合」——
因为正交映射要求两侧同维，而 SAMGA 的输出空间 ≠ CLIP 空间，需要另算图像教师。

**ENIGMA 没有这个问题**：它的主干输出 `c_eeg ∈ R^1024` **就是** CLIP ViT-H/14 空间，
而 gallery 的 `clip_img_{train,test}_1024.npy` 也在同一空间。
⇒ **$R \in O(1024)$ 的拟合是原生维度的，不需要任何中间投影，也不需要方案 A/B 的取舍。**

这直接消掉了 v2 里一整节的设计复杂度。

---

## 4. 架构总览

```text
┌── 阶段 0：主干复用（对 ENIGMA 官方实现零改动）─────────────────────────┐
│  EEG [B,63,250]                                                        │
│      │                                                                 │
│      ├──(源被试) subject_wise_linear[s]  ← 保留官方实现，逐被试时间混合   │
│      └──(目标被试) ⛔ 该键不存在 ⇒ xENIGMA 在此处接入 ↓                  │
│                              │                                         │
│                     tsencoder (共享, 可迁移)                            │
│                              │                                         │
│                     mlp_proj (共享 184→1024, 可迁移)                    │
│                              │                                         │
│                     c_eeg ∈ R^1024  （≈ CLIP ViT-H/14 空间）            │
└──────────────────────────────┼─────────────────────────────────────────┘
                               │
┌── 阶段 1：部署侧坐标恢复（全无标签、全闭式、无参数训练）────────────────┐
│  ① SAW         W_s = (Σ_s + λI)^{-1/2}，用目标被试无标签试次估           │
│  ② 矩匹配      逐维均值/方差对齐 EEG 与图像特征（正交映射管不了平移）      │
│  ③ CSLS landmark  互最近邻 + top1/top2 margin 加权（抑制 hubness）       │
│  ④ 正交恢复     R* = U Vᵀ，M = X̃ᵀ W Ỹ + λI = U Σ Vᵀ （加权 Procrustes） │
│                 λ = ρ‖X̃ᵀ W Ỹ‖₂, ρ = 0.1 （identity 正则可省不可缺）    │
│  ⑤ 在恢复后空间重算 CSLS                                                │
└──────────────────────────────┼─────────────────────────────────────────┘
                               │ 恢复后的 (200,1024) 条件
┌── 阶段 2：生成（零改动，复用现有栈）───────────────────────────────────┐
│  SDXL-Turbo(4步, gs=0) + IP-Adapter(ip-adapter_sdxl_vit-h, 1024-d)     │
│  邻域库 retrieve_neighbors 换 CSLS（修 v2 缺陷 B）                      │
└──────────────────────────────┼─────────────────────────────────────────┘
                               │
┌── 阶段 3：选择 ────────────────────────────────────────────────────────┐
│  CSLS 分数 + 多样性惩罚（防 hub 塌缩）                                   │
└────────────────────────────────────────────────────────────────────────┘
```

**与 v2 的关键差异**：主干从「SAMGA 检索编码器 + 我们自建的 GenHead」换成「ENIGMA 官方重建主干」。
好处有三：① 主干本身是被验证过的**重建** SOTA（不是检索模型改的）；② 生成侧零胶水代码；
③ 恢复的拟合空间是原生 1024-d，消掉 v2 的方案 A/B 取舍。

---

## 5. 分层组件设计

### L0 — 主干复用（P0）

直接使用官方 `ENIGMA` + `SDXL_Reconstructor`。**唯一必要的改动**：给 `subject_wise_linear` 加一个
**目标被试的回退路径**（默认 identity 或共享层），否则 LOSO 前向会 `KeyError`。

```python
# 建议的最小侵入式改法（不破坏官方 checkpoint 的加载兼容性）
def forward(self, x, subjects):
    x_eeg = torch.zeros_like(x)
    subjects = np.array(subjects)
    for subject in np.unique(subjects):
        m = torch.from_numpy(subjects == subject)
        if subject in self.subject_wise_linear:
            x_eeg[m] = self.subject_wise_linear[subject](x[m])
        else:
            # 目标被试（LOSO 留出）→ 走可恢复路径
            x_eeg[m] = self.fallback(x[m])   # fallback 初值 = Identity
    return self.mlp_proj(self.tsencoder(x_eeg))
```

`fallback` 的初值取 **Identity**，使「无恢复」时等于「不做被试对齐」，作为消融的 None 行。

### L1 — 训练侧 inter 增强（源被试上）

| 编号 | 组件 | 依据 | 代价 |
|---|---|---|---|
| **E1** | **训练集重复平均**：把 4 次重复平均成 1 行（复刻论文 A.1 的声称行为） | §2.3 实测 66160 = 16540×4；论文自称做了但代码没做 | ~5 行 |
| **E2** | **Multi-positive 对齐**：正样本集 $\mathcal{P}(i)=\{j: y_j=y_i\}$，权重 $1/|\mathcal{P}(i)|$，双向 | §2.4 确认缺陷；SCORE **+2.41** | ~15 行 |
| **E3** | **恢复感知训练（recovery-aware episodes）**：每 batch 抽一个源被试当 pseudo-target，隐藏其匹配，跑一遍完整恢复流程，**梯度不穿映射**，再算损失 | SCORE **+0.70**，且放大恢复收益 | 中 |
| **E4** | **共享对齐层替代逐被试时间混合**：把 `Linear(250,250)` 换成隐空间上的共享对齐 + SAW 归一化，使缺层时仍有定义 | §2.3 不一致 2；§1.4 | 中 |

> **E1 与 E2 是互补而非重复的**：E1 消除「同一刺激的重复试次」这一种近重复；E2 处理「同一刺激的**不同被试**」
> 这一种。多被试训练下细粒度地看，E2 覆盖的是 E1 覆盖不到的部分（跨被试同刺激），两者都做才完整。

### L2 — 部署侧坐标恢复（目标被试，全无标签、全闭式）

严格按 SCORE 五步，**无参数训练**：

1. **SAW**：$W_s=(\Sigma_s+\lambda I)^{-1/2}$，$\tilde z = W_s(z-\mu_s)/\|W_s(z-\mu_s)\|_2$。
   * 依据：SATTC 9.2/30.5 → 13.7/36.4；SCORE 在 SAMGA 编码器上受控测量 26.22 → **30.98 (+4.76)**。
   * 校准成本：**N=50 个无标签试次即达 N=200 上界的 94.8%**——对 EEG 很友好。
2. **矩匹配**：逐维对齐 EEG 与图像特征的均值/方差（正交映射**管不了平移**，这步不可省）。
3. **CSLS landmark 选择**：$\text{CSLS}(x,y)=2\cos(x,y)-r_G(x)-r_Q(y)$；只保留**互最近邻**；
   用 top-1 与 top-2 的**间距**加权（EEG 信噪比低 ⇒ 不确定的对权重小）。
4. **加权正交恢复**：
   $$R^\star=\arg\min_{R^\top R=I}\ \|W^{1/2}(\tilde X R-\tilde Y)\|_F^2+\lambda\|R-I\|_F^2$$
   闭式解 $R^\star=UV^\top$，$M=\tilde X^\top W\tilde Y+\lambda I=U\Sigma V^\top$。
   **identity 正则不可省**：部署 batch 只有几百 query，landmark 数 $m\ll d=1024$，无约束方向欠定；$\lambda$ 让无证据方向留在 identity。
5. 恢复后空间重算 CSLS。

**SCORE 的孤立贡献（累积消融，我们据此排优先级）**：
`+CSLS` **+9.75** → `+mean/scale 匹配` +4.72 → `+正交恢复` **+7.18** → `+identity 正则` +2.25。

> ⚠️ **只借鉴 SAW，不要借鉴 SATTC 的完整算子**：SATTC 在弱编码器（ATM）上 9.2→14.8，
> 但在**强编码器上反而变差**（SAMGA：26.12 < 26.22）。SCORE 明确指出「完整的 SATTC 算子对冻结的强表征贡献极小」。
> ENIGMA 主干也是强表征，同理。

### L3 — 生成与选择

| 编号 | 组件 | 依据 |
|---|---|---|
| **E9** | **邻域库换 CSLS**：`retrieve_neighbors` 现在是裸 `np.argsort(-sim)`（`erdc_ras_closed_loop.py:64-71`） | v2 缺陷 B；给出 FID 反常的机制（hub 塌缩 ⇒ 多样性降 ⇒ FID 降但语义差） |
| **E10** | **条件不确定度驱动候选数** | 低信噪比样本多采样，高置信样本少采样 |

### L4 — 可选增强（非主干，按优先级插入）

| 编号 | 组件 | 依据 | 优先级 |
|---|---|---|---|
| **E11** | **多模态条件**（CogCapPro 式：图像+文本+深度+边缘 多 IP-Adapter） | CogCapPro 是**最新的**重建 SOTA（2026-03）且代码可用 | P2 |
| **E12** | **多视角 foveated 目标**（SIMON）：把 CLIP 目标从「单张中心裁剪」改为多视角 foveated，携带位置/尺度 | SIMON inter 19.6；对**像素保真**可能比对检索更有价值 | P2 |
| **E13** | **通道角色分解**：前部=被试不变锚定，后部=被试特异细节，双分支后融合 | v2 §2.4 四篇论文收敛；**且 ENIGMA 与 SAMGA 同病（§2.5）** | P3（需重训） |

---

## 6. 维度与归一化细节（ENIGMA 特有的两个坑）

### 6.1 恢复在哪个空间做——原生 1024-d

* 主干输出 `c_eeg ∈ R^1024`；gallery `clip_img_test_1024.npy ∈ R^{200×1024}`；`clip_img_train_1024.npy ∈ R^{16540×1024}`。
* **同维同源** ⇒ $R\in O(1024)$ 可直接拟合（§3.3）。**不需要图像教师，不需要中间投影。**

### 6.2 条件要不要 L2 归一化——ENIGMA 与 ATM 相反，这是**有意设计**，必须保持

`source/utils.py: get_eegfeatures` **不做任何归一化**，直接返回 `backbone.float()`。
这与论文 A.2 的自述一致：

> 「We chose **not** to normalize $f_{CLIP}(\text{image})$ in the MSE component to ensure that the learned $c_{EEG}$
> respects the **geometry of the CLIP embedding space**, and note that doing so negates the need for the secondary
> diffusion prior training stage.」（对应消融 Fig.5 `[14]`）

⇒ ENIGMA 的条件是**带模长**的 CLIP 向量，模长由 MSE 学到。

**处理策略**（重要，否则会悄悄破坏主干）：
* **landmark 匹配 / CSLS 打分**：用 **L2 归一化**后的方向（语义由方向承载）。
* **最终送入 IP-Adapter 的条件**：用**带模长**的向量。
* **正交映射 $R$ 对两者都安全**：$R^\top R=I$ ⇒ $\|Rz\|=\|z\|$，模长自动保持。
  ⇒ 做法是：对归一化向量拟合 $R$，然后把这个 $R$ **原样作用回未归一化的 `c_eeg`**。模长不变，方向已校正。

---

## 7. 评测协议

### 7.1 ⚠️ 三个必须先修的口径问题（否则数字会误导，甚至得出错误结论）

这是本次审计中**与架构同等重要**的发现，全部来自我们自己的代码。

**陷阱 1：我们的 SSIM 不是论文级 SSIM，不能和 ENIGMA 的 0.4264 比。**

`eeg-brainit/src/eeg_brainit/utils/metrics.py: ssim_simple` 的 docstring 自述：

> 「Lightweight luminance/contrast SSIM **proxy for monitoring (not paper-grade)**」

它只在全局（`dim=(-2,-1)`）算一个均值/方差，**没有滑窗、没有局部结构项**——是标量 MAP-SSIM 的退化形式。
ENIGMA / ATM / MindEye 用的是 Wang et al. 2004 的标准 SSIM（灰度、滑窗、逐块平均）。
⇒ **我们已有的 LOSO SSIM 数字与文献不可比，必须换成 `skimage.metrics.structural_similarity`（grayscale）重算。**

**陷阱 2：`alexnet2` 这个名字在我们的两个脚本里指两个不同的量。**

| 脚本 | 输出 | 实际含义 | 对应 ENIGMA 表里的列 |
|---|---|---|---|
| `erdc_full_metrics.py` | `alexnet2`, `alexnet5`, `inception` | **Pearson 相关**（`_pearson_flat`） | ❌ **无对应** |
| `erdc_twoway_metrics.py` | `alex2`, `alex5`, `inception`, `clip` | **200-way 2WC 百分比** | ✅ `Alex(2)`, `Alex(5)`, `Incep`, `CLIP` |

⇒ 把 `erdc_full_metrics.py` 的 `alexnet2=0.61` 放到 ENIGMA 的 `Alex(2)=82.99%` 旁边是**假比较**。
**报告时必须显式区分**，建议在 JSON 里改名为 `alexnet2_corr` / `alex2_2wc`。

**陷阱 3：ENIGMA 报的 SwAV 距离我们没有。**

ENIGMA Table 1 的 `Eff ↓` / `SwAV ↓` 是 EfficientNet-B13 / SwAV-ResNet50 的**平均相关距离**。
我们有 `effnet_b1`（相关，方向一致可近似对应），但**完全没有 SwAV 分支**。
⇒ 若要逐列对齐 ENIGMA 表，需补一个 SwAV-ResNet50 特征提取器（约 30 行 + 一个权重下载）。

**顺带确认的好消息**：`erdc_twoway_metrics.py: twoway()` 用完整 `n=200` 池、每行遍历所有 $j\neq i$
⇒ 就是 **200-way / 199 干扰项**，与 ENIGMA 附录 A.3 的口径（「199 other test-set stimuli as distractors」）**一致**，
可以直接比。这一点比很多论文的「2-way / 1 干扰项」更严格，是我们的优势，**必须在报告中标注**。

### 7.2 四行基准 + oracle 归一化

仓库**没有任何跨被试重建基线**，所以现有数字无法自答「好不好」。必须补：

| 行 | 作用 | 现状 |
|---|---|---|
| 随机条件 | 下界，扣掉生成器自带能力 | 待建 |
| **oracle（真 CLIP 条件）** | 上界，分离「条件误差」与「生成器饱和」 | ✅ 已有 0.456 / 0.164 |
| **被试内上界** | 用**同一套指标代码**跑被试内，量化 inter 代价 | ⚠️ 有，但受陷阱 1 影响需重算 |
| 本方法（LOSO） | 待评 | ✅ 已有 0.3435 / 0.1257（head）、0.2502 / 0.0478（identity） |

（锚点为 CLIP cosine / PixCorr）

$$\text{score}_{\text{norm}}=\frac{\text{score}-\text{chance}}{\text{oracle}-\text{chance}}$$

**分层报告是强制的**：条件层（CLIP cosine、检索 Top-1/5）与像素层（PixCorr、SSIM、FID、2WC）分开报，
否则条件层的改进会被生成器饱和掩盖（生成器永远把任何条件渲成一张「像图」的图）。

### 7.3 诚实性协议：映射拟合不能用评测对（**本文档最重要的方法论约束**）

重建的指标是「生成图 vs 被试真实看过的图」。如果用**同一批 200 概念**既拟合 $R^\star$ 又评测，
$R^\star$ 可能拟合到那 200 个特定配对上，**抬高 PixCorr/CLIP 却无真实泛化**。

**协议（优先级从高到低）：**

1. **首选：用 16540 张训练集图像当 gallery 拟合 $R^\star$**（完全无标签、与评测概念不相交），
   再作用到 200 个测试概念上生成评测。这比 SCORE 的设定**更严格**，也更贴合我们的诚实性要求。
2. **必做：概念集切成不相交的 A/B（各 100）**：用 A 拟合，**冻结**，作用到 B，在 B 上报全部指标。
   直接复刻 SCORE Table 6 的设定（该设定下增益**更大**：+24.63）。
3. **辅助：全 200 拟合只作为参考行，必须标注**，不得作为主结果。
4. 拟合只用无标签量：**无被试标签、无配对真值**。

### 7.4 统计功效

* sub-08 单折 CLIP 的 CI 约 ±0.014（bootstrap），而待检测增益可能只有 0.02–0.03 ⇒ **≥5 折**。
* checkpoint 一律用 `last.pth`（final epoch）。ENIGMA 官方 `recon_inference.py` 还 assert 了
  「必须存在 `last.pth`，若用 `best.pth` 需有验证集」——与 SCORE 用 final epoch 的做法一致，我们跟随。
* 采样数：ENIGMA 官方默认 `--repetitions 10`（每样本 10 张，取指标均值）。我们沿用 10 以可比。

---

## 8. 优先级与首批里程碑

| 优先级 | 项 | 预期 | 代价 | 依据 |
|---|---|---|---|---|
| **P0** | L0：`subject_wise_linear` 加 LOSO 回退（否则跑不起来） | 解除硬阻断 | ~10 行 | §2.1 代码确认 |
| **P0** | 陷阱 1：换论文级 SSIM | 使数字可比 | 低 | 我们自己的 docstring |
| **P0** | 陷阱 2：重命名/区分 `alexnet*_corr` vs `alex*_2wc` | 防止假比较 | 极低 | 两个脚本对照 |
| **P0** | E1：训练集重复平均 | 复刻论文声称行为 | ~5 行 | §2.3 实测 66160=16540×4 |
| **P0** | E2：multi-positive 对齐 | **+2.4 量级** | ~15 行 | §2.4 缺陷确认；SCORE +2.41 |
| **P0** | E9：邻域库 CSLS | 可能同时改善 FID 与语义 | 低 | 裸 `argsort` 确认；CSLS +9.75 |
| **P0** | §7.2 四行基准 | 建立可比性 | 低 | 当前无基线，最阻塞 |
| **P1** | **E5+E6+E7+E8：完整坐标恢复** | **最大单点杠杆** | 中 | +4.76(SAW) / +9.75(CSLS) / +7.18(正交) |
| **P1** | §7.3 诚实性协议（训练画廊 + A/B） | 使上项可信 | 低 | SCORE Table 6 |
| **P2** | E3：恢复感知 episode | 放大恢复收益 | 中 | +0.70 |
| **P2** | E11：多模态条件（CogCapPro 式） | 高保真 | 中-高 | CogCapPro 2026-03 |
| **P2** | E12：foveated 多视角目标 | 提升 PixCorr/SSIM | 中 | SIMON |
| **P3** | E4：共享对齐层替代时间混合 | 使缺层时有定义 | 中 | §2.3 不一致 2 |
| **P3** | E13：通道角色分解 | 同时服务 CLIP 与 PixCorr | 高（重训） | 4 篇收敛；ENIGMA 同病 |
| **P3** | 多折扩展（≥5） | 统计功效 | 中 | QOS 限 8 并发，需滴灌 |

### 建议的第一个里程碑

**P0 全部 + P1 的完整坐标恢复。**

理由：P0 里有**四项是纠错而非改进**（§2.1 硬阻断、陷阱 1、陷阱 2、§2.4 缺陷），代价近乎零；
P1 是全文档证据最强的杠杆（+4.76 / +9.75 / +7.18），且**全部是闭式、无需重训**，
可以直接叠加在**已有的 ENIGMA retrieval checkpoint 与 609259 产物**上先验证核心假设。

跑完这一步就能回答最关键的问题：**「坐标恢复对重建到底有没有效」**，
再决定是否投入 P2/P3（尤其 E13 需要重训编码器）。

---

## 9. 风险与证伪条件

| 风险 | 症状 | 证伪 / 备选 |
|---|---|---|
| **正交恢复在重建上不迁移**（最大风险） | 条件层 CLIP 提升但像素层（PixCorr/SSIM）不动 | 这本身是有价值的负结论：说明检索的几何增益不传递到像素。备选：转 E12（目标空间）+ E13（空间分支） |
| **映射过拟合到 200 概念** | A/B 切分后增益消失 | 已被 §7.3 强制暴露；若消失则说明恢复不可用 |
| **E2 在 ENIGMA 主干上无效** | 训练 loss 正常但 LOSO Top-1 不动 | ENIGMA 主干强表征可能已隐式吸收了跨被试结构（SATTC 的教训：强表征上算子增益衰减） |
| **恢复的 landmark 在低信噪比下不可靠** | CSLS 互最近邻对很少 | 降 gallery 粒度（用 category 级 landmark）；或用训练画廊 16540 张提高覆盖率 |
| **hubness 假设错误** | 换 CSLS 后多样性不动 | 那 FID 反常需另找机制（量化类内/类间多样性） |
| **E13 需重训且收益未验证** | 成本高 | 先用 1 折、短 epoch 验证方向 |
| **多折统计功效不足** | CI 宽于增益 | ≥5 折；先 3 折看方向 |
| **ENIGMA 官方训练脚本无验证集** | 无法 early stop | 官方设计如此（用 final epoch）；我们用 `last.pth` 并遵守其 assert |

---

## 10. 与 CANVAS v2 的关系（增量 diff）

| 维度 | CANVAS v2 | xENIGMA（本文） |
|---|---|---|
| 主干 | SAMGA **检索**编码器（冻结）+ 自建 GenHead | **ENIGMA 官方重建主干** |
| 主干是否重建 SOTA | ❌ 检索 SOTA 改的 | ✅ 是（2026-02，多被试重建 SOTA） |
| 生成侧工程量 | 需要自建 + 胶水（`train_gen_head.py` 等） | **零胶水**（栈逐项同构，§1.3） |
| 恢复的拟合空间 | ⚠️ 需在 SAMGA 512-d 与 CLIP 1024-d 之间取舍（方案 A/B） | ✅ **原生 1024-d，无取舍**（§3.3） |
| LOSO 可行性 | 可跑（SAMGA 前向不依赖被试） | ⚠️ 需先加回退（§2.1 硬阻断）→ P0 |
| 缺陷 A（multi-positive） | 我们自己的 `train_gen_head.py` | **SOTA 官方代码里也有**（+ 我们自己也修） |
| 缺陷 B（CSLS 邻域） | 同 | 同（保留） |
| 诚实性协议 | ✅ §4.4 | ✅ §7.3（更强：训练画廊优先） |
| 评测协议 | §5（但未发现三个口径陷阱） | §7.1 **新增**：SSIM proxy / Alex 同名异实 / 缺 SwAV |
| SAW / CSLS / 正交恢复 | ✅ | ✅（证据与数值不变） |

**v2 中被本文推翻的**：主干选择（SAMGA → ENIGMA），以及 v2 §4.3「方案 A/B」的取舍问题（被 ENIGMA 的原生维度消掉）。
**v2 中被本文保留的**：§4.1 两个缺陷诊断、§4.4 诚实性协议、§5 评测协议、§7 明确不借鉴清单、v1 的 FiLM 被推翻的结论。

**v2 中仍然有效、且本文新增证据的**：§2.4 的通道拓扑结论——现在有**双重**动机（既是未被占据的杠杆，
又是 ENIGMA 与 SAMGA **共同**的结构缺陷，§2.5）。

---

## 附：引用来源

**外部**
- **ENIGMA**：`arXiv:2602.10361`（Kneeland, Jiang, Bruzadin Nunes, Scotti, Delorme, Xu / Alljoined）—
  Table 1/2，附录 A.1（预处理）/A.2（损失）/A.3（指标定义）/A.7（通道消融）/A.8（人评）
- **ENIGMA 代码**：`github.com/Alljoined/ENIGMA`（commit `460af4b`，2025-05-24）；
  读过的文件：`source/models.py`、`source/training.py`、`source/dataset.py`、`source/utils.py`、
  `train.py`、`recon_inference.py`、`configs/things_eeg2.json`
- **CogCapPro**：`arXiv:2603.12722`（2026-03-13）；代码 `github.com/XiaoZhangYES/CognitionCapturerPro`；
  第三方复现 `github.com/Chikit-WONG/DL_Project`
- **SCORE**：`arXiv:2608.19134`（Cui, Kan, Li, Wang, Wu, HUST）— Table 1/2/3/4/5/6，§Cross-Subject Coordinate Diagnosis
- **SATTC**：CVPR 2026 pp.16887-16896 / `arXiv:2603.20738`（Huang, Zhu, Yunnan U）— Table 1/2，§3.2 SAW
- **SIMON**：`arXiv:2605.00401`（Wei, Lin, Tsai）— inter 19.6/49.9，通道拓扑
- **Hierarchical Visual Embeddings**：`arXiv:2602.07495` — 被试平均 PixCorr 0.186~0.195 / SSIM 0.336
- **SGDM**：`arXiv:2604.22649`（2026-04）— 无官方代码（搜索结果只命中 BMVC 2023 同名无关工作）
- **NEED**：NeurIPS 2025 / `arXiv` — 无官方代码
- **Alljoined-1.6M**：`arXiv:2508.18571` — Table 1（ENIGMA/ATM-S/Perceptogram 复现对照）

**本仓库实测**
- ENIGMA LOSO 硬阻断：`ENIGMA/source/models.py: ENIGMA.__init__ / forward`
- ENIGMA 未做过 LOSO：`ENIGMA/train.py` + `ENIGMA/recon_inference.py`（同一批 `subj_ids`）
- 训练集未平均：实测 `ENIGMA/data/preprocessed_data/things_eeg2/sub-08/`
  `preprocessed_eeg_training_flat.npy = (66160, 63, 250)`，`unique image_path = 16540` ⇒ 4 reps
- multi-positive 缺陷：`ENIGMA/source/training.py: ClipLoss.forward`（`torch.arange`）
- `W_s` 维度不一致：论文 §3.3 说 `N_z×N_z`；`source/models.py` 是 `Linear(250,250)`（时间轴）
- 条件不归一化：`ENIGMA/source/utils.py: get_eegfeatures`（无 `F.normalize`）+ 论文 A.2
- 空间坍缩：`ENIGMA/source/models.py: Spatio_Temporal_CNN`（`conv2_kernel=(63,1)`）
- **陷阱 1**：`eeg-brainit/src/eeg_brainit/utils/metrics.py: ssim_simple`（自述 "not paper-grade"）
- **陷阱 2**：`eeg-brainit/scripts/erdc_full_metrics.py: _pearson_flat` vs `erdc_twoway_metrics.py: twoway`
- **陷阱 3**：无 SwAV 分支（对照 ENIGMA Table 1 的 `SwAV ↓` 列）
- 缺陷 B：`eeg-brainit/scripts/erdc_ras_closed_loop.py:64-71`；调用点 `erdc_official_atm_pipeline.py:329`
- 缺陷 A（我们自己）：`eeg-retrieval/scripts/recon/train_gen_head.py:95-98`
- 现有锚点：oracle 0.456/0.164、head 0.3435/0.1257、identity 0.2502/0.0478（CLIP cosine / PixCorr）
- 生成栈同构性：`eeg-brainit/scripts/erdc_official_atm_pipeline.py:99-114`、`erdc_full_metrics.py:85-87`
- 2WC 口径一致：`eeg-brainit/scripts/erdc_twoway_metrics.py:42-60`（200-way / 199 干扰项）
