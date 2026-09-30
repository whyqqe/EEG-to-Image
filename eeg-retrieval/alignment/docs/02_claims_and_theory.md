# 主张与理论（定稿 v3 · 裁决式脑科学论文）

> v1 bounds-first / v2 时间分离 → **均被 2026 文献覆盖**（见第 0 部分）。
> v3 改为**裁决式（adjudication）**：不提出新现象，而是用决定性对照回答已有结论是否成立。
> 依据：文献**自己点名**缺少容量匹配对照。冻结日期 2026-09-17。

---

## 第 0 部分：已死主张（禁止再宣称）

| 想宣称的 | 已被谁做 | 证据 |
|---|---|---|
| 视觉早 / 语义晚时间分离 | [eLife 108915](https://elifesciences.org/reviewed-preprints/108915)、[arXiv 2506.19497](https://arxiv.org/pdf/2506.19497) | 视觉成分峰 90 ms、语言成分峰 365 ms，EEG，10 被试，16k 图 |
| EEG 维度时程（升-峰-降） | [Chen et al. CCN/ bioRxiv](https://2025.ccneuro.org/abstract_pdf/Chen_2025_Neural_Dimensionality_Temporal_Dynamics_Visual_Representations.pdf) | **同一数据集 THINGS-EEG2**；峰 150–350 ms |
| "模型解释不了的神经维度" | Chen et al. | 其头条结论 |
| 子空间可识别性框架 | [Han et al., PMLR v337](https://proceedings.mlr.press/v337/han26a.html) | 基不变子空间识别 |
| 目标脑侧恢复剖面 | [arXiv 2605.20127](https://arxiv.org/html/2605.20127) | fMRI，NSD，8 被试 |
| 对齐分数的混淆批判 | [Nat Comms 2026](https://doi.org/10.1038/s41467-026-72253-7)、[2605.14025](https://arxiv.org/html/2605.14025v1)、[2505.12196](https://arxiv.org/html/2505.12196v1)、[2601.22722](https://arxiv.org/html/2601.22722) | 该赛道已拥挤 |
| 编码器总预测力分离 | [BrainSAIL](https://doi.org/10.48550/arxiv.2410.05266) | fMRI 上 CLIP/DINO/SigLIP 相当 |

---

## 第 1 部分：开着的门（我们唯一的立足点）

关于 eLife 时间分离工作，检索综合明确指认其缺口：

> "...this does not establish a uniquely semantic late response: language and vision embeddings differ in dimensionality, information content, and training, and the study used GPT-4V-generated descriptions **rather than capacity-matched controls**... A convincing test requires matched-dimensional, complexity-controlled vision and language representations, cross-validated variance partitioning, temporal generalization, and artifact-resistant preprocessing."

关于 Chen et al. 的维度结论，其原文自述：

> "the EEG data **approaches full rank** during this peak period **despite the high signal spread across channels and low signal-to-noise ratio**"

**低信噪比下的维度估计天然虚高**，该陈述未经噪声校正检验。

⇒ **两个被点名、未被满足的对照，恰好是我们的工具能做的。**

---

## 第 2 部分：论文主轴

> **人脑视觉响应看起来有多丰富，其中有多少是真的？**
>
> 两个裁决：
> **(i)** 晚期"语义"响应是真的语义，还是模型容量？
> **(ii)** "接近满秩"是真的高维，还是噪声？
>
> 两裁决共用同一把钥匙：**留出可复现子空间 + 容量/维度匹配对照**。

**方法定位**：留出筛选、NV、\(N/S\)、虚假地板全部降级为**工具**，写在方法中，不作贡献宣称（同时规避 Nat Comms 2026 已占的混淆批判赛道）。

---

## 第 3 部分：主张集

### 主张 1（头条）：晚期语义响应中容量伪影的占比 [NOVEL — 文献点名缺失]

> 将视觉与语言特征**投影到同维数**，并对**未训练对照模型**做残差化（依 [2505.12196](https://arxiv.org/html/2505.12196v1) 的方法），在留出可复现子空间内重测 365 ms 语言成分。报告真实语义贡献 vs 容量贡献的分解。

- **脑科学含义**：直接回答物体识别理论的核心问题——人脑晚期是否真进入语义阶段
- **新颖性依据**：eLife 原文自承缺此对照；无人补做
- **两种结局都可发表**：存活 → 语义阶段确证；消失 → 纠正广泛引用的结论
- **执行要点**：
  - 维数匹配：两模态 PCA/随机投影到同一 \(k\)，\(k\) 扫描
  - 容量对照：未训练同架构模型残差化
  - 交叉验证方差分解 + 时间泛化
  - 统计：跨被试聚类置换检验（Maris–Oostenveld）

### 主张 2：EEG"接近满秩"是噪声所致 [NOVEL — 直接检验 Chen et al.]

> 用噪声校正的参与比重测维度时程：
> $$\hat S_{\text{sig}}=S_{\text{mean}}-\tfrac{1}{T}S_{\text{resid}},\qquad r_{\text{rel}}=\mathrm{PR}(\hat S_{\text{sig}}),\qquad \mathrm{PR}(S)=\frac{(\mathrm{tr}S)^2}{\mathrm{tr}S^2}$$
> \(S_{\text{resid}}\) 需投影至 PSD 锥。若 \(r_{\text{rel}}\) 显著低于满秩，则"低信噪比下满秩"为噪声假象。

- **脑科学含义**：人脑视觉编码的**真实**维度——神经编码维数核心问题
- **新颖性依据**：Chen et al. 用分半 PCA 投影相关，**未做噪声校正参与比**；同数据集直接检验
- **风险**：估计器必须极稳，否则反被质疑

### 主张 3：可复现子空间的时间剖面 ≠ 原始 EEG 的时间剖面 [NOVEL]

> 在跨试次可复现子空间内重测时间剖面，与原始 EEG 剖面比较。若形状不同，则既有时间分辨研究均受噪声结构污染。

- **脑科学含义**：时间剖面的形状可能不是脑的，而是噪声的
- **新颖性依据**：所有时间分辨研究（含 eLife）均在原始 EEG 上做

### 主张 4：可复现子空间的跨被试共享 vs 个体私有及其时间剖面 [NOVEL，中等]

> 每被试可复现子空间分解为跨人共享 / 个体私有，比较时间剖面。

- **脑科学含义**：通用视觉代码有多窄、个体差异在哪个时间窗最大
- **新颖性依据**：跨**模态**共享/独有已做（Cichy MEG-EEG）；跨**被试**可复现子空间分解 + 时间剖面未做
- **风险**：CorrCA/M-CCA 方法成熟，创新仅在具体分解

---

## 第 4 部分：不宣称清单

| ❌ 不宣称 | 原因 |
|---|---|
| 时间分离现象本身 | eLife 108915 |
| EEG 维度时程现象 | Chen et al.（同数据集） |
| 模型未覆盖的神经维度存在 | Chen et al. |
| 子空间可识别性 | Han et al. |
| 虚假地板/NV/\(N/S\) 作为方法贡献 | Nat Comms 2026 赛道已占；降级为工具 |

---

## 第 5 部分：风险与目标期刊

### 风险
1. **Chen et al. 用同一数据集（THINGS-EEG2）** → 主张 2/3 是"在其地盘裁决"，审稿人可能判增量。**缓解**：主张 1 作头条（不同赛道），2/3 作支撑。
2. **主张 1 说"他人分离是伪影"需极强统计** → 最大执行风险。
3. **能力要求提高**：需要极稳估计器 + 严密对照设计，难度远高于跑检索。

### 目标
| 条件 | 目标 |
|---|---|
| 主张 1 裁决成功（任一方向）+ 主张 2 成立 | **eLife / PLOS Biology / Nature Communications** |
| 仅主张 2/3 成立 | **Imaging Neuroscience / NeuroImage** |
| 主张全不显著 | 方法论文（TMLR / NeurIPS D&B） |

---

## 第 6 部分：实验映射

| 主张 | 依赖实验 | 状态 |
|---|---|---|
| 主张 1 | **A5（新）：容量匹配 + 未训练残差化的时间裁决** | TODO |
| 主张 2 | **A3b**：噪声校正参与比 | TODO（低风险，优先补） |
| 主张 3 | **A6（新）**：可复现子空间时间剖面 | TODO |
| 主张 4 | **A7（新）**：跨被试共享/私有分解 | TODO |
| 工具 | A0 合成（估计器有效性）、A1 控制电池、A2 \(N/S\) | 门控运行中 |

**执行顺序**：A3b（低成本、修主张 2）→ 门控 A0/A1/A2 出数 → 依 \(r_{\text{rel}}\) 实曲线定 A5/A6/A7 窗口划分 → GPU 抽取（已挂 `afterok`）→ A5–A7。
