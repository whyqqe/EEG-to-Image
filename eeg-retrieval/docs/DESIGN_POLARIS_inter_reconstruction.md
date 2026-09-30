# POLARIS：基于 CogCapPro 的跨被试 EEG→图像重建新架构

> **POLARIS** = **P**er-subject **O**rthogonal **L**andmark-**A**nchored **R**otation
> **I**dentification in **S**ensor space
>
> 主干：CogCapPro（`arXiv:2603.12722`，2026-03，官方代码 commit `cf3fc5b`，已 clone 至
> `third_party/CognitionCapturerPro/`）。
>
> 与 `DESIGN_xCogCap_inter_reconstruction.md` 的关系：那一版是**审计 + 修补清单**（哪里坏了、补什么）；
> 本文是**架构本身**——从理论推出架构，每个设计选择都有推导或可证伪的预测。
> 审计发现（三个硬阻断、`top_k` 截断、两个指标口径陷阱、诚实性协议）全部保留，作为架构的**前置约束**而非主体。
>
> 所有未标注来源的判断均标为**假设/待验证**并附证伪条件。

---

## 0. 结论先行

理论分析给出三个结论，它们共同决定了架构形态：

1. **被试差异在一阶近似下是「一个群作用」**——具体是作用于共享概念码的
   *拉伸 + 旋转 + 平移*（§1.2）。CogCapPro 自己的叙事把 inter 问题称作 "representational shift"，
   本文把这个 shift 形式化。

2. **修正必须尽可能早**（§1.5）。下游（融合 / 扩散先验 / align / IP-Adapter）**全都不是等变的**，
   所以修正每晚一层就多一层不可逆的失真。最早的可用位置是**传感空间**。
   由此得到一个反直觉的推论：**不该在 1024 维隐空间修 4 个旋转，而该在 63 维传感空间修 1 个旋转**
   （假设 H1，§1.3）。参数量从 $4\times1024^2\approx4.2\text{M}$ 降到 $63^2\approx4\text{K}$。

3. **闭式正交恢复的有效性有三个充要条件**（§1.4），它们**可以变成训练目标**：
   (C1) 秩预算 $r_{\text{eff}}\le m_{\text{anchor}}$；(C2) 锚点张成覆盖；(C3) 被试映射的**谱平坦**。
   ⇒ 这三条直接给出训练侧的三个新损失（§3.1），使架构**「按构造可辨识」**，
   而不是"训练完再祈祷它能被恢复"。

**与上一版方案的核心区别**：上一版把恢复当**部署期后处理**；本架构把恢复的**可辨识性**当成
**训练期目标**，部署期只是取闭式解。这是"加一个 patch"与"设计一个架构"的分界。

---

## 1. 理论分析

### 1.1 生成模型与记号

| 记号 | 含义 | 取值 |
|---|---|---|
| $x_s(c)$ | 被试 $s$ 看概念 $c$ 时的传感记录 | $\mathbb{R}^{C\times T}$，$C{=}63$, $T{=}250$ |
| $P_s$ | 被试观测算子（电极位置 / 颅骨电导 / 参考） | $\mathbb{R}^{CT\times D}$ |
| $\varphi(c)$ | **被试无关**的概念码 | $\mathbb{R}^{D}$ |
| $b_s,\ \Sigma_s$ | 被试偏移、噪声协方差 | — |
| $E_m$ | 第 $m$ 个 EEG 分支（CogCapPro 有 4 个独立分支） | $\mathbb{R}^{C\times T}\to\mathbb{R}^{1024}$ |
| $z^{(m)}_s(c)$ | 分支输出 | $\mathbb{R}^{1024}$ |
| $\mathcal{D}$ | 冻结下游：融合 → [扩散先验] → align → IP-Adapter | — |

**生成模型**

$$x_s(c) \;=\; P_s\,\varphi(c) \;+\; b_s \;+\; \varepsilon_s,\qquad \varepsilon_s\sim(0,\Sigma_s)$$

**假设 A1（共享概念码）**：$\varphi$ 与被试无关，被试差异全部进 $P_s,b_s,\Sigma_s$。
*依据*：SCORE 在 THINGS-EEG2 同折上测得源-目标 **EEG 的 RSM 相关性 $0.687\pm0.035$**
⇒ 概念的**几何结构是共享的**，若不共享则跨被试检索只能到随机水平。
*证伪条件*：若 RSM 相关性接近 0，则本架构的前提不成立，退回到被试特异训练。

### 1.2 仿射被试模型（ASM）：把 encoder 也纳入

对 $E_m$ 在数据流形附近线性化。记 $J_m=\partial E_m/\partial x$ 在被试均值处取值：

$$z^{(m)}_s(c)\;\approx\;\underbrace{J_mP_s}_{\textstyle A^{(m)}_s}\big(\varphi(c)-\bar\varphi\big)\;+\;t^{(m)}_s
\qquad(\text{ASM})$$

**这一步的意义**：把"被试差异"从"encoder 内部未知的东西"变成"输出空间里一个显式的线性算子 $A^{(m)}_s$"。
$A^{(m)}_s$ 就是可修正对象。三项分别：

| 成分 | 数学 | 处理 |
|---|---|---|
| $t^{(m)}_s$ 平移 | 加性偏移 | 居中 / 矩匹配 |
| $A^{(m)}_s$ 拉伸+旋转 | 一般线性 | **本文主体**（§1.4 给出何时可简化为纯旋转） |
| $\varepsilon_s$ 噪声 | 协方差 $\Sigma_s$ | SAW |

### 1.3 假设 H1：共享被试算子 ⇒ 传感空间单个旋转

从 ASM 立刻读出 CogCapPro 特有的结构：

$$\boxed{\;A^{(m)}_s=J_m\,P_s\;}\qquad\text{——}\ P_s\ \textbf{被 4 个分支共享}$$

**推导**：$P_s$ 来自**头/电极的物理**（lead field、电导率、电极位移），
4 个分支读的是**同一条 EEG 记录**，所以物理算子 $P_s$ 只有一个。
各分支只通过自己**已知且与被试无关**的雅可比 $J_m$（训练好后冻结）产生差异。

**H1 的三个强度层级**（应分开检验，不要一次全上）：

| 形式 | 断言 | 可检验性 |
|---|---|---|
| **H1-weak** | 4 个分支的误差由**一个共同因子**生成（低维而非 4 个独立自由度） | 对 $\{R_m\}$ 做 PCA，看主成分解释比 |
| **H1-mid** | 存在**单个传感空间线性算子** $Q_s\in\mathbb{R}^{C\times C}$ 使 $E_m(Q_sx)$ 同时校正 4 个分支 | 直接优化 $Q_s$ 看 4 分支残差是否同时下降 |
| **H1-strong** | $Q_s$ 可取为**正交**（$Q_s\in O(C)$，1953 个自由参数） | 与一般线性 $Q_s$ 对比（同 SCORE 的 ridge vs orthogonal 对照） |

**若 H1 成立，代价对比**：

| 方案 | 参数 | 需要拟合的坐标系 |
|---|---:|---:|
| 逐分支隐空间恢复（上一版方案） | $4\times1024^2\approx4.2\text{M}$ | 4 个 |
| **H1-mid 传感空间单算子** | $63^2=3969$ | **1 个** |
| **H1-strong 传感空间正交** | $63\cdot62/2=1953$ | **1 个** |

⇒ 参数量降低约 $10^3$ 倍，锚点需求同比例下降，且**跨模态一致性自动成立**（不再需要额外约束）。

**为什么这是"最早修正"原则的自然结果**（见 §1.5）：$Q_s$ 作用于 $x$，位于
$E$ 的**最上游**，一次修正穿透全部 4 个分支与所有下游模块。

**诚实标注**：$Q_s$ 在传感空间的作用能否**精确**模拟 $P_s$ 在 $D$ 维概念子空间上的作用，
取决于 $P_s$ 的值域是否落在 $Q_s$ 可达的方向上——一般不能精确。
所以 H1-strong 是**近似**，H1-weak 才是安全形式。**H1 的检验是本文最重要的一个消融（§8 AB-1）。**

### 1.4 可辨识性：正交恢复何时有效

部署期要解的问题。给定 $m$ 个锚点对 $\{(X_i,Y_i)\}$（目标侧 vs 源/共识侧，$X,Y\in\mathbb{R}^{m\times1024}$）、
权重 $W=\mathrm{diag}(w)$：

$$R^\star=\arg\min_{R^\top R=I}\ \big\|W^{1/2}(X R-Y)\big\|_F^2+\lambda\|R-I\|_F^2
\;\;\Longrightarrow\;\;
M=X^\top WY+\lambda I=U\Sigma V^\top,\quad R^\star=UV^\top$$

**三个条件**（推导见下）：

#### (C1) 秩预算：$r_{\text{eff}}\le m$

$M$ 的秩 $\le\min(m,1024)$。锚点张成的子空间 $\mathrm{span}(X)$ 之外的**全部方向无信息**；
identity 正则给出的解是在那些方向取 $R=I$。这是一个**良定义的最小范数补全**，不是启发式：
它是"在无证据方向不引入改变"的唯一合理选择。

⇒ **但要注意一个被文献忽略的后果**：对**检索**而言，无信号方向上的误差无害（不改变排序）；
对**重建**而言，生成器可能对无信号方向敏感，把 $R=I$ 留在那里**可能有害**。
这是一个新的、未被报告的风险，必须实测（§8 AB-7）。

#### (C2) 锚点张成：$\mathrm{span}(X)$ 必须覆盖概念活跃子空间

若活跃子空间有 $r$ 维而锚点只张成其中 $r'<r$ 维，则剩下 $r-r'$ 维未校正。
⇒ 锚点选择应当是**覆盖性设计**，而不是"取互最近邻"（§3.2 的杠杆分数选择）。

#### (C3) 谱平坦：真实被试映射的拉伸部分 $\propto I$

写真实映射的 SVD：$A_s=U_s\Sigma_s V_s^\top$。"旋转分量"是 $U_sV_s^\top$，"拉伸分量"是 $\Sigma_s$。
正交假设 $R\approx U_sV_s^\top$ **无偏 $\iff$ $\Sigma_s\propto I$**。
若 $\Sigma_s$ 各向异性，则任何正交 $R$ 都**系统性偏离**真值。

*这里澄清一个易混点*：**概念码 $\varphi$ 本身各向异性无害**（它属于共享子空间，
被试旋转是作用在它**外围**的坐标系上，各向异性在该旋转下不变，不需要被"修掉"）。
真正有害的是**被试算子自身的拉伸** $D_s:=\Sigma_s$。

*为什么 $\Sigma_s$ 有理由接近平坦*：CogCapPro 的输入是逐被试白化过的
（`Preprocessed_data_250Hz_whiten`），它把**传感空间**的逐通道方差拉到相近。
但那是**对角**平坦，在**隐空间**不成立 ⇒ 残余拉伸仍在，需要 (C3) 作为训练目标。

#### (C4) 锚点噪声的衰减偏差

若锚点中有比例 $q$ 的随机错配，估计量向 0 衰减，近似
$$\mathbb{E}[\widehat{U_sV_s^\top}]\approx(1-2q)\,U_sV_s^\top$$
⇒ **系统性欠校正**。这与 identity 正则同向，二者叠加会显著削弱恢复。
⇒ 必须**显式估计 $q$ 并做校准**（§3.2 的稳健 Procrustes）：
残差 $\{r_i=\|X_iR^\top-Y_i\|\}$ 是"内点 + 均匀离群点"的混合分布，$q$ 可**无标签**拟合。

#### (C5) SCORE 数值对上述条件的支持

| 对照 | 结果 | 支持的条件 |
|---|---|---|
| ridge（一般线性）20.01 vs **orthogonal** 28.22（留出概念 3 折） | 约束更强反而更好 +8.21 | (C3) 成立（纯旋转），ridge 的自由度纯属方差 |
| 裸用 26.22 → +CSLS 35.78 → +正交恢复 **48.75** → +恢复感知训练 53.23 | 正交恢复单步 **+12.97** | 恢复有效且可叠加 |
| identity 正则单列 +2.25 | — | (C1) 补全确有贡献 |
| 不相交 gallery 迁移 +24.63 | — | **无标签拟合且泛化**（对应 §7 诚实性协议） |

### 1.5 设计律 DR1：修正要尽可能早

**命题**：设被试变换 $g_s$ 作用在表示上，$E(x_t)=g_s\cdot E(x_s)$（同概念）。
- 在 $E$ 的**输出**处修正：应用 $g_s^{-1}$，**精确**，前提是 $\mathcal{D}$ 作用在修正后的输入上。
- 在 $\mathcal{D}$ 的**输出**处修正：需要 $\mathcal{D}$ 对 $g_s$ 等变（$\mathcal{D}(g_s z)=h_s\mathcal{D}(z)$）。
  $\mathcal{D}$ 由 Transformer + LayerNorm + 多模态融合 + 多 IP-Adapter 组成，**一般不等变**。

⇒ **推论**：修正位置越早，看到失真的非等变模块越少，误差越小。
最早的位置是**传感空间** $x$。这与 §1.3 的 H1 相互印证：
**H1 给"在传感空间修正"提供了参数化，DR1 给"应该在传感空间修正"提供了理由。**

*落地*：把恢复放在 `Cogcap` 分支的**输入侧**。若日后发现传感空间不可行，
一级退路是**分支输出侧**（上一版方案），但那时已经过 4 个非线性分支。

### 1.6 统一视角：SAW / hubness / 谱平坦是同一现象的三个投影

三者此前在文献里是三个独立 trick，在本框架下是 ASM 的三个项：

| 现象 | ASM 中对应 | 数学对象 |
|---|---|---|
| SAW / 白化 | $\Sigma_s$ | 噪声协方差的**二阶**差异 |
| hubness | $\Sigma_s$ 各向异性导致的相似度集中 | 同上，**体现为检索侧的病理** |
| (C3) 谱平坦 | $A_s$ 的拉伸 $D_s$ | 被试算子的**尺度**差异 |

⇒ **推论**：CSLS 不是"补丁"，而是 $\Sigma_s$ 未校正时的必要补偿。
若 (C3) 被训练目标改善，CSLS 的必要性应下降。这是一个可测量的预测（§8 AB-4）。

*这条统一视角是把三个 trick 收编成一个架构的理论收益，而不是三个各自独立的 heuristic。*

### 1.7 从理论到架构：映射表

| 理论对象 | 架构组件 | 章节 |
|---|---|---|
| $\Sigma_s$ 二阶差异 | 传感空间 SAW（**SW**） | §3.2 |
| $A_s$ 拉伸 (C3) | **L_spec** 谱平坦正则 + **L_aug** 旋转增强一致性 | §3.1 |
| $A_s$ 旋转 | **RP** 稳健加权正交恢复 | §3.2 |
| 秩/张成 (C1)(C2) | **AS** 杠杆分数锚点选择 + identity 正则 | §3.2 |
| 锚点噪声 (C4) | **RP** 衰减校准 | §3.2 |
| H1 共享算子 | **SEA** 传感空间被试算子（正交参数化） | §3.1 |
| 修正位置 (DR1) | 恢复挂载在分支输入侧 | §2 |
| $P_s$ 物理性 | **CRD** 通道角色分解 | §3.1 |
| 多分支一致性 | **BC** 分支一致性准则（无标签） | §3.2 |
| 跨被试正样本 | **L_mp**（修 `top_k` 截断） | §3.1 |

---

## 2. 架构总览

```text
╔═════════════════════════ 训练期（源被试，9 人）═════════════════════════╗
║                                                                        ║
║  x [B,63,250]                                                          ║
║   │                                                                    ║
║   ├─ CRD  通道角色分解：C_anchor(被试不变锚定) ⊎ C_detail(被试特异细节)  ║
║   │                                                                     ║
║   ├─ L_aug 旋转增强一致性：x → R x (R~O(63)) ⇒ 表示应不变到概念保持     ║
║   │                          （把残余变换压成近似正交，喂给 RP）        ║
║   │                                                                     ║
║   ├─ EEGAttention (官方)                                                ║
║   ├─ SEA  传感空间被试算子：正交参数化(Cayley) + 源被试各自的 t_s       ║
║   │        目标被试缺失 ⇒ Identity fallback（= 显式消融基线）           ║
║   └─ ×4 独立 Cogcap 分支 ⇒ z_m ∈ R^1024                                ║
║         │                                                              ║
║         ├─ L_mp   跨被试多正样本（取消 top_k 对正样本的门控，见 §3.1 勘误）║
║         ├─ L_spec 共识映射的谱平坦（在概念活跃子空间内评估）            ║
║         └─ CogcapFusion（官方）→ [MultiModalDiffusionPrior] → align      ║
╚════════════════════════════════════════════════════════════════════════╝
                                   │
╔══════════════ 部署期（目标被试，全无标签、闭式、仅 1953 参数）═══════════╗
║  ① SW   传感空间 SAW：W=(Σ+λI)^{-1/2}，目标被试无标签试次估计           ║
║  ② AS   锚点选择：CSLS → MNN → 4 分支投票 → **杠杆分数覆盖性筛选**      ║
║  ③ RP   稳健加权正交恢复：M = X^T W Y + λI = UΣV^T，R=UV^T              ║
║          + 离群点混合拟合估 q + 衰减校准                                ║
║  ④ BC   分支一致性准则：无标签挑选/否决候选 R（H1 的验证器）            ║
║  ⑤ 冻结 R，作用于测试集                                                ║
╚════════════════════════════════════════════════════════════════════════╝
                                   │
╔══════════════════ 冻结下游（零改动，仅稳健化增强）═══════════════════════╗
║  CogcapFusion → [扩散先验] → SimpleAlignNet → 3× IP-Adapter            ║
║  SDXL-Turbo, guidance_scale=0.0, 官方 per-block scale                   ║
║  + align 期被试变换增强（使 align 对残余恢复误差稳健）                  ║
╚════════════════════════════════════════════════════════════════════════╝
                                   │
╔══════════════════ 生成与选择 ═══════════════════════════════════════════╗
║  邻域库 retrieve_neighbors 换 CSLS（修裸 argsort 的 hub 塌缩）           ║
║  条件不确定度驱动候选数                                                 ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## 3. 模块细节

### 3.1 训练侧

#### SEA — Sensor-space Equivariance Adapter（H1 的参数化）

```python
# 感官空间被试算子：x_corrected = R_s @ x + t_s
# R_s 用 Cayley 参数化保证正交（训练时永远落在 O(63) 流形上）
class SEA(nn.Module):
    def __init__(self, C=63, n_subjects=10):
        self.A = nn.ParameterList([   # 反对称矩阵，C(C-1)/2 = 1953 参数
            nn.Parameter(torch.zeros(C, C)) for _ in range(n_subjects)])
        self.t = nn.ParameterList([nn.Parameter(torch.zeros(1, C, 1)) for _ in range(n_subjects)])
    def R(self, s):
        A = self.A[s].triu(1); A = A - A.T               # 强制反对称
        I = torch.eye(A.shape[0], device=A.device)
        return (I - A) @ torch.linalg.inv(I + A)         # Cayley ⇒ 严格正交
    def forward(self, x, subject_id):
        if subject_id is None:      # 目标被试：Identity fallback
            return x
        return self.R(subject_id) @ x + self.t[subject_id]
```

**为什么正交参数化而非一般线性**：与 (C3) 直接对应——把假设类**限制**在旋转上，
消掉拉伸，使部署期的闭式正交恢复**无偏**。这与 SCORE 的 ridge(20.01) < orthogonal(28.22) 是同一逻辑，
但 SCORE 只在**部署期**施加约束，这里把它前移到**训练期参数化**。

**Identity fallback 是刻意的**：让"不做被试对齐"成为显式消融基线（§8 的 None 行），
并且回避"学一个被试编码器去逼近闭式几何解"——SCORE 用闭式正交在同一编码器上拿到 +27.01，
学习式被试码很难逼近它。SEA 的作用**不是**估计目标被试的变换，
而是**塑造源被试侧的表示，使被试变换的群结构变得良态**（谱平坦、近似正交）。

#### CRD — Channel Role Decomposition

依据 SIMON（`arXiv:2605.00401`）：后部通道承载**被试特异**响应、广覆盖通道承载**被试不变**信息。

* 用 $C_{\text{anchor}}$（广覆盖/前部）估计被试变换与锚点 → **降低校准路径上的噪声源**。
* $C_{\text{detail}}$（后部）只在校正**之后**注入 → 保留被试特异细节而不污染校准。
* 与 §3.2 的锚点选择耦合：锚点只在 $C_{\text{anchor}}$ 上算 CSLS。

*依据强度*：SIMON 的通道拓扑结论 + SCORE 的 RSM 共享性；**未在本仓库验证**，标为待验证。

#### 损失

**L_mp — 跨被试多正样本**（前置约束：先修掉官方实现的截断）

官方 `ClipLoss_Modified_DDP` 的软标签是 `topk(目标相似度, k=10) ⊙ (img_index 相同)` 的**交集**，
即正样本还必须落进 top-10 才被计入。

> **勘误**（实现在 `scripts/cogcap/losses.py`，由 `scripts/test_cogcap.py` 实测；
> 本节初稿的算术与修复方式**都是错的**）
>
> 初稿写"每图 36 行、25 个被推开"。**错**：`train_avg: True`（`configs/cogcappro.yaml:35`）
> 会在 `load_data`（`data/eeg.py:280-288`）里先把 4 次重复平均掉，损失根本看不到重复维，
> 所以每刺激行数是 **S=9**，不是 4×9=36。S=9 且 `top_k=10` 时，若目标特征可区分刺激，
> 8 个兄弟行恰好是相似度最大的 8 行、全部落在 top-10 内 ⇒ **一个都不掉**。
>
> 实测（S=9、`top_k=10`、`scripts/test_cogcap.py`）：
>
> | 目标特征的 tie 结构 | 每行正样本 | upstream 保留 | **被丢弃** |
> |---|---:|---:|---:|
> | 可区分（per-image） | 9 | 9.00 | **0** |
> | 跨刺激 tie（重复 caption / per-concept） | 9 | 4.11 | **4.89（54%）** |
>
> 所以：结论（会掉正样本，且掉的恰是 logit 最大、梯度贡献最大的那些行）**成立**，
> 但**触发条件是目标特征的 tie 结构**，不是重复次数。CogCapPro 的 `text` 分支目标是
> per-image 的 BLIP2 caption，而 BLIP2 短 caption 在同类图上高度重复 ⇒ 那正是最容易踩中的分支。

修正：**取消对正样本的相似度门控**：
$$\mathcal{P}=\{(i,j):\text{img\_index}_i=\text{img\_index}_j\},\qquad \mathcal{M}=\mathcal{P}$$

> ⚠️ 修正**不是**"交集改并集"——初稿的 $\mathcal{M}=\mathcal{P}\cup\tilde{\mathcal{N}}$
> （$\tilde{\mathcal{N}}=\text{topk}\setminus\mathcal{P}$）同样是错的，而且更糟：
> softmax 分母本就含全部行，`top_k` 从来不是在挑负样本，它只是在**给正样本设卡**；
> 而 $\text{mask\_sim}\setminus\mathcal{P}$ 里的行是**目标相似但属于不同刺激**的行，
> 把它们计入 label 等于教模型混淆两张图。
> 实测该写法在良性情形下也把 loss 从 4.008 改成 4.040
> （`test_repair_equals_upstream_when_nothing_is_truncated` 因此失败）——
> 在不需要修复的地方引入了错误。

**L_spec — 共识映射的谱平坦**（(C3) 的训练化）

对每个源被试 $s$，用 batch 内同概念跨被试配对（标签已有）算共识映射
$M_s=Z_s^\top W Z_{\setminus s}$，罚其奇异谱偏离平坦：

$$\mathcal{L}_{\text{spec}}=\sum_s\Bigg\|\frac{\mathrm{diag}(\Sigma_s)}{\|\Sigma_s\|_F}-\frac{I}{\sqrt{d}}\Bigg\|_F^2,
\qquad M_s=U_s\Sigma_sV_s^\top$$

**只在概念活跃子空间内评估**（取 $M_s$ 的前 $r_{\text{eff}}$ 个奇异方向），
否则会把与被试无关的噪声方向也拉平、浪费容量。

*实现注意*：`torch.linalg.svd` 在奇异值重根处梯度不稳。稳妥实现用
$\|M_s^\top M_s/\|M_s^\top M_s\|_F-I/d\|_F^2$（绕开 SVD，梯度良态、代价更低）。

*与 SCORE 的关系*：SCORE 的"recovery-aware training"（+4.48）是这条的**经验版本**；
L_spec 是把它写成**显式矩条件**。二者互为消融（§8 AB-5）。

**L_aug — 旋转增强一致性**（L_spec 的采样版本）

对 EEG 施加随机传感空间旋转 $R\sim O(63)$，要求**概念结构保持**：

$$\mathcal{L}_{\text{aug}}=\sum_m D_{\text{RSM}}\Big(S\big(E_m(Rx)\big),\,S\big(E_m(x)\big)\Big)$$

用 RSM 距离而非逐元素 L2，因为**要保的是概念几何（"哪些图相似"），不是表示本身**——
逐元素不变会退化为不变性表示，丢掉概念信息。

*为什么这合法*：被试变换是**冗余**，概念是**信号**（假设 A1）。故对**被试变换**求不变
**不损失概念信息**。而之所以还需要部署期恢复，是因为增强只能采样到**旋转**这一个子群，
真实 $P_s$ 更丰富（增益、噪声协方差、非旋转分量）⇒ 增强把残余变换**压成近似正交**，
恰好使闭式正交恢复适用。**这是 SCORE "recovery-aware training" 的小增益（+4.48）的一个解释。**

#### 前置约束（P0 纠错，非本架构的创新）

| 编号 | 内容 | 依据 |
|---|---|---|
| CE1 | 被试轴：`subject_wise_linear[0]` 硬编码 + `num_subjects=1` ⇒ 空操作。改为 SEA 接管 | `brain_backbone.py:166-176, 200` |
| CE2 | 放开多被试：`--subjects` → `nargs="+"`，`paths.py:321` | `runtime/paths.py:321` |
| CE3 | 修 `inter-subject` 崩：`ModelCheckpoint(save_last=True)` 无 monitor 却 `test(ckpt_path="best")` | `cli/train.py:85,117` |
| CE4 | 修 `top_k` 截断（= L_mp） | `utils.py:296-310` |

### 3.2 部署侧（闭式、无标签、无梯度）

#### SW — 传感空间 SAW

$$\mu_s=\mathbb{E}[x_t],\quad \Sigma_s=\mathrm{Cov}(x_t),\quad W_s=(\Sigma_s+\lambda I)^{-1/2},
\quad x\!\leftarrow\! W_s(x-\mu_s)$$

*放在传感空间而非隐空间*（DR1）：一次修正穿透全部 4 分支与所有下游。
*依据*：SATTC 的 SAW 在同编码器上 9.2/30.5 → **13.7/36.4**；
SCORE 的受控测量 26.22 → **30.98**；校准成本 **N=50 无标签试次即达 N=200 上界的 94.8%**。

⚠️ SATTC 的**完整算子组合反而变差**（26.12 < 26.22，强编码器上）⇒ **只取 SAW**，
其余 SATTC 组件（Adaptive CSLS 等）作为消融而非默认开启。

#### AS — 锚点选择（(C2) 的落地）

四步，前两步提纯、后两步保证**覆盖性**：

1. **CSLS 打分** $\text{CSLS}(x,y)=2\cos(x,y)-r_G(x)-r_Q(y)$（反 hubness）。
2. **MNN + 多分支投票**：只有当一个对在 $\ge K$ 个分支上同时为互最近邻才收为候选。
   CogCapPro 给 **4 个独立分数场**（+fusion 5 个），这是相对单分支主干的结构性红利。
   SCORE 只有 1 个分数场，无法投票。
3. **残差混合拟合**：对候选拟合两成分混合（内点 / 均匀离群），估离群比例 $q$（(C4) 需要）。
4. **杠杆分数覆盖性筛选**：这是相对"取互最近邻"的**改进**。
   (C2) 要求 $\mathrm{span}(X)$ 覆盖概念活跃子空间，而不是"锚点越多越好"。
   取活跃子空间的正交基 $U_r$（来自源侧概念协方差），算杠杆分数
   $\ell_i=\|U_r^\top \tilde X_i\|_2^2$，用 **D-最优 / 贪心最大行列式**选子集，
   使 $\sum_i \ell_i$ 在 $U_r$ 各方向上均衡覆盖。

*为什么第 4 步重要*：MNN 倾向于选出**同类且易分辨**的锚点（近邻密集区），
它们张成的子空间**有偏**——恰好漏掉难分辨的方向，而那正是需要校正的方向。

#### RP — 稳健加权正交恢复（(C1)(C3)(C4) 的落地）

$$M=X^\top WY+\lambda I=U\Sigma V^\top,\qquad R^\star=UV^\top$$

三个不可省的细节：

* **identity 正则 $\lambda=\rho\|X^\top WY\|_2$**，$\rho=0.1$（SCORE 单列 +2.25）。
  作用：(C1) 在无证据方向取 $R=I$ 的良定义最小范数补全。
* **稳健权重 $W$**：用第 3 步估的 $q$，对残差做 winsorize / trimmed，
  抑制离群锚点对 $M$ 的污染。
* **衰减校准**：按 (C4) 用 $\hat R\leftarrow\hat R$ 的逆缩放修正（角域按 $\arccos$ 线性化处理），
  补偿 $\mathbb{E}[\hat R]\approx(1-2q)R$ 的系统性欠校正。

#### BC — 分支一致性准则（H1 的验证器 + 无标签模型选择）

计算逐分支的 $R_m$（H1-mid 下由单一 $Q_s$ 诱导），定义**缺陷度**

$$\Delta=\max_{m\ne m'}\big\|R_mR_{m'}^\top-I\big\|_F$$

两个用途：
1. **失败检测**：$\Delta$ 超阈值 ⇒ 恢复不可信，回退到 identity 或共享 $R$。
2. **无标签模型选择**（**升级自上一版方案的"自检"**）：不同锚点子集 / 不同 CSLS 参数 /
   不同 $\lambda$ 会给出不同候选 $R$，**选 $\Delta$ 最小的那个**。全程不需要任何标签。

*依据*：H1 预测各分支误差应由共同因子生成 ⇒ 校正后 $R_m$ 应相互一致。
一致性差本身即 H1 不成立的证据（§8 AB-1）。

### 3.3 下游稳健化

#### align 期被试变换增强

align 阶段（`SimpleAlignPipe.train`）在**冻结 EEG 编码器**之上训练。
训练时对 EEG 施加随机 $R\sim O(63)$（+ 小扰动），要求 align 输出不变：

$$\mathcal{L}_{\text{align-aug}}=\big\|\mathrm{Align}(E(Rx))-\mathrm{Align}(E(x))\big\|_2^2$$

作用：使 align **对残余恢复误差稳健**，把最终误差从"恢复精度"降级为"恢复精度的平方"。
代价极低（align 阶段本就独立），且**直接对齐 CogCapPro 自己的 "asymmetric alignment" 叙事**。

#### 已核实的一处**非缺陷**（负面结果，避免误改）

`SimpleAlignMLP.forward` 末尾 `F.normalize`，而 align 的监督目标 $h$（IP-Adapter 条件）
**未归一化**。表面看是尺度错配（MSE 项存在不可消除下界、reg 项退化为常数）。
**已核实 `diffusers.models.embeddings.ImageProjection` 含 `nn.LayerNorm(cross_attention_dim)`**
（对本环境 diffusers 源码实测输出确认）⇒ IP-Adapter 内部先做 per-token LayerNorm
⇒ **输入幅值被归一化掉，只有方向有效** ⇒ 该错配**不是缺陷，不要改**。

---

## 4. 训练流程与损失汇总

沿用 CogCapPro 的三阶段（`staged_training`），只做**叠加**：

| 阶段 | 官方内容 | POLARIS 叠加 |
|---|---|---|
| **stage 1**（单模态） | 4 分支 + 4 模态对齐，`L_mp` | `+ L_mp` 修正、`+ L_spec`、`+ L_aug`、SEA/CRD 生效 |
| **stage 2**（联合） | 加 fusion 分支 | 同上，权重不变 |
| **stage 3**（仅融合） | 只训 fusion | 同上 |
| **align** | `SimpleAlignPipe`，`SDEmbeddingLoss` | `+ L_align-aug` |

$$\mathcal{L}=\underbrace{\sum_{k\in\{\text{img,txt,depth,edge,fus}\}}\tfrac12\big(\ell^{\text{eeg}}_k+\ell^{\text{mod}}_k\big)}_{\text{官方 ClipLoss(已修)}}\;+\;\alpha\mathcal{L}_{\text{spec}}\;+\;\beta\mathcal{L}_{\text{aug}}$$

**超参起点**：$\alpha=\beta=0.1$；$\lambda_{\text{spec}}$ 的评估子空间取 $r_{\text{eff}}$ 由源侧概念协方差的
能量累积到 90% 定；$K$（投票门限）从 2 起；$\rho=0.1$。

**训练期不引入任何目标被试信息**（除 CE2 的 LOSO 切分）。SEA 对目标被试恒为 Identity。

---

## 5. 部署算法（严格伪代码）

```python
# ===== 输入 =====
# M_S   : 9 源被试训练好的 CogCapPro + SEA + fusion + align（全冻结）
# X_t_tr: 目标被试【无标签】训练集 EEG（16540 图 × 4 重复，train_avg）
# X_t_te: 目标被试【无标签】测试集 EEG（200 图 × 80 重复，test_avg）
# G_tr  : 16540 张训练图的 IP-Adapter 条件锚（= generator._image_to_embedding）
#         ★ 与 200 张测试刺激【完全不相交】
# G_te  : 200 张测试图的同空间锚（只用于评测，不参与拟合）

# ===== 步骤 1：传感空间 SAW（DR1：最早修正）=====
mu, Sigma = estimate(X_t_tr)                       # 目标被试无标签
W = inv_sqrt(Sigma + lam * I)
X_sw = W @ (X_t_tr - mu)                           # [16540, 63, 250]

# ===== 步骤 2：算 4 分支表示 + 逐模态矩匹配 =====
for m in (image, text, depth, edge):
    Z[m]   = E_m(X_sw)                             # [16540, 1024]
    Z[m]   = moment_match(Z[m], Gtr_hat[m])         # 对齐 gallery 侧逐维 mean/var
                                                    # （正交映射管不了平移，必须显式做）

# ===== 步骤 3：锚点选择（AS）=====
for m:
    S[m]  = csls(Z[m], Gtr_hat[m])                  # 2cos - r_G - r_Q
    MNN[m]= mutual_nearest_neighbor(S[m], C_anchor) # 只用通道角色 anchor 子集
vote = sum(MNN[m])                                  # 0..4
cand = {(i, argmax_j S[m][i,j]) for i if vote[i] >= K}
q, inlier_mask = fit_two_component_mixture(residuals(cand))   # 估离群比例 (C4)
U_r  = top_r_left_singular(concept_covariance(M_S, Gtr))       # 概念活跃子空间
sel  = d_optimal_greedy(cand, U_r, leverage_scores)            # (C2) 覆盖性设计

# ===== 步骤 4：稳健加权正交恢复（RP）=====
for m:
    X, Y = Z[m][sel.src], Gtr_hat[m][sel.dst]
    w    = robust_weight(residuals_before, q)                    # winsorize
    Cs   = covariance(X) if H1_mid else None
    Mmat = X.T @ diag(w) @ Y + rho * norm(X.T @ diag(w) @ Y) * I # identity 正则 (C1)
    U, _, Vt = svd(Mmat)
    R[m] = debias_attenuation(U @ Vt, q)                         # (C4) 衰减校准
    if H1_mid:                                                   # 单一传感算子
        Q = argmin_Q || R[m] - J_m Q ||   ->  R[m] = J_m @ Q
# H1 检验：4 个 R[m] 是否由共同因子生成（PCA 解释比）

# ===== 步骤 5：分支一致性（BC）作无标签模型选择 =====
if max_{m!=m'} ||R[m] @ R[m'].T - I||_F > tau:
    R = select_candidate_by_min_defect(all_candidates)           # 换/否决候选
    if still_bad: R = shared_R(all_landmarks)                    # 或退回 identity

# ===== 步骤 6：应用到测试集（R 冻结，绝不重拟合）=====
Zte = {}
for m:
    Zte[m] = R[m] @ moment_match(W @ (X_t_te - mu), Gtr_hat[m])
    Zte[m] = normalize(Zte[m])

# ===== 步骤 7：冻结下游 → 生成 =====
cond   = align_net(fusion_and_branches(Zte))          # 官方冻结模块
images = ip_adapter_generator(cond.image, cond.depth, cond.edge)
```

**唯一性约束**：步骤 1–5 只用**无标签、无配对真值**的量；
$R$ 在**训练 gallery**上拟合并冻结，**绝不在测试集上重新拟合**。

---

## 6. 差异表

### 相对 CogCapPro 官方

| 维度 | CogCapPro | POLARIS |
|---|---|---|
| 被试轴 | `subject_wise_linear[0]` 硬编码 ⇒ **空操作** | **SEA**：传感空间正交算子（1953 参数/被试） |
| 被试变换的作用位置 | 无 | **传感空间**（分支输入侧，DR1） |
| inter 支持 | `exp_setting` 已写但 `ckpt_path="best"` **崩溃** | 修（CE3） |
| 多被试 CLI | `subjects=[args.subjects]` 单元素 | 修（CE2） |
| 跨被试正样本 | **已有**，但 `top_k=10` 会门控正样本（tie 结构下最多丢 54%，见 §3.1 勘误） | **L_mp**（取消门控） |
| 被试映射的谱 | 无约束 | **L_spec** 谱平坦（(C3)） |
| 对被试变换的鲁棒性 | 无 | **L_aug** + align 增强 |
| 校准 | 逐被试**离线**预处理白化 | **SW** 在线无标签 + **RP** 闭式正交 |
| 多模态的利用 | 4 分支参与训练与对齐；**生成只用 3 个** IP-Adapter（image/depth/edge，text 不进 IP-Adapter） | **额外**用 4 个分数场做锚点投票 |
| 通道角色 | 无 | **CRD** |

### 相对上一版方案（xCogCap）

| 维度 | xCogCap（审计版） | POLARIS（本文） |
|---|---|---|
| 恢复位置 | 分支**输出**侧（1024 维） | **传感空间**（63 维，DR1 + H1） |
| 待估变换数 | 4 个独立 $R_m$ | **1 个** $Q_s$ |
| 参数量 | $\approx 4.2$M | **1953** |
| 恢复的角色 | 部署期后处理 | 部署期取闭式解；**可辨识性作为训练目标** |
| 训练侧新增 | 仅纠错（CE1–CE4） | **+ L_spec / L_aug / SEA / CRD**（4 个新机制） |
| 锚点选择 | MNN + margin 加权 | MNN + 投票 + **杠杆分数覆盖性设计** |
| 一致性检查 | 事后自检 | **无标签模型选择** |
| 理论 | 分层差距模型（现象学） | **可辨识性的充要条件 (C1)–(C4)** |

---

## 7. 评测协议

### 7.1 三个必须先修的口径陷阱（来自本仓库）

| 陷阱 | 问题 | 后果 |
|---|---|---|
| **SSIM 非论文级** | `eeg-brainit/src/eeg_brainit/utils/metrics.py: ssim_simple` 自述 *"lightweight … not paper-grade"*，无滑窗、无局部结构项 | 现有 LOSO SSIM **不可与文献比**；须换 `skimage.metrics.structural_similarity` 重算 |
| **`alexnet2` 同名异实** | `erdc_full_metrics.py` 的 `alexnet2` 是 **Pearson 相关**；`erdc_twoway_metrics.py` 的 `alex2` 是 **200-way 2WC%** | 同表并列是**假比较**；改名 `alexnet2_corr` / `alex2_2wc` |
| **缺 SwAV** | ENIGMA 报 `SwAV ↓`，本仓库无该分支 | 无法逐列对齐文献表 |

**好消息**：`erdc_twoway_metrics.py: twoway()` 用完整 `n=200` 池、每行遍历所有 $j\ne i$
⇒ 是 **200-way / 199 干扰项**，与 ENIGMA 附录 A.3 口径一致，**可直接比**，
且比常见的 2-way/1 干扰项严格。**必须标注。**

### 7.2 诚实性协议

重建指标（PixCorr/SSIM）是「生成图 vs 被试真实看过的图」，**同一批 200 概念**若既用于拟合 $R$ 又用于评测，
$R$ 可能拟合到那 200 个特定配对 ⇒ **虚高**。

**主协议**：用目标被试**训练集** EEG（无标签）↔ **16540 张训练图 gallery** 拟合 $R$，
与 200 张测试刺激**完全不相交**，冻结后评测。

⇒ **诚实的偏差，正是统计上必需的**：(C1) 要求 $r_{\text{eff}}\le m_{\text{anchor}}$。
只拟合 200 个锚点时 $m\le 200$，在 1024 维隐空间里**欠定**；用训练 gallery 时 $m$ 可达数千，
才可能张成概念活跃子空间。**所以训练 gallery 拟合不是"更保守的折中"，而是唯一的可行配置。**
（这一条也解释了 SCORE Table 6 的"不相交 gallery 迁移"为何增益**更大**：+24.63。）

**辅助报告**：
1. **A/B 概念切分**（各 100）：A 拟合、在 B 上报。
2. 全 200 拟合**只作参考行且必须标注**，不得作主结果。
3. 报告 $r_{\text{eff}}$、$m_{\text{anchor}}$、$q$ 三个量——它们是 (C1)(C2)(C4) 的直接指标，
   而文献普遍不报。

### 7.3 四行基准 + oracle 归一化

本仓库**没有任何跨被试重建基线** ⇒ 现有数字无法自答"好不好"。

| 行 | 作用 | 现状 |
|---|---|---|
| 随机条件 | 下界 | 待建 |
| **oracle（真 CLIP 条件）** | 上界，分离"条件误差"与"生成器饱和" | ✅ 已有 0.456 / 0.164 |
| **被试内上界** | 同指标代码跑被试内，量化 inter 代价 | ⚠️ 有但受陷阱 1 影响需重算 |
| **POLARIS（LOSO）** | 待评 | 待跑 |

$$\text{score}_{\text{norm}}=\frac{\text{score}-\text{chance}}{\text{oracle}-\text{chance}}$$

**分层报告是强制的**：条件层（CLIP cosine、检索 Top-1/5）与像素层（PixCorr、SSIM、FID、2WC）分开。
否则条件层改进会被生成器饱和掩盖。

### 7.4 统计功效

sub-08 单折 CLIP 的 bootstrap CI 约 ±0.014，而待检测增益可能只有 0.02–0.03 ⇒ **≥5 折**。
checkpoint 用 `last.pth` / final epoch；采样步数固定（官方 `generator.py` 默认 5）。

---

## 8. 消融（每条都是可证伪的预测）

| 编号 | 消融 | 预测 | 若否证意味着 |
|---|---|---|---|
| **AB-0** | 基线：官方 + 修好 inter 设置，无任何恢复 | 显著低于被试内 | — |
| **AB-1** | **H1 三级检验**：独立 $R_m$ vs 单传感 $Q_s$（H1-mid）vs 正交 $Q_s$（strong） | 未知，**本文最重要的实验** | H1 不成立 ⇒ 退回隐空间逐分支恢复（上一版方案） |
| **AB-2** | **恢复位置**：传感空间 vs 分支输出 vs align 输出 | 传感空间最好（DR1） | 下游对旋转不敏感 ⇒ DR1 判据需修正 |
| **AB-3** | **锚点选择**：MNN 原版 vs +投票 vs +杠杆分数 | 逐级改善（(C2)） | (C2) 不构成瓶颈 ⇒ 张成已足够 |
| **AB-4** | **CSLS 的必要性**：开/关，且在 L_spec 训练前后各测一次 | 训练后 CSLS 增益下降（§1.6） | 三者的统一视角有误 |
| **AB-5** | **L_spec vs L_aug vs 两者** | 都能改善 (C3)；两者互补 | 若无效 ⇒ (C3) 不是瓶颈，或实现有误 |
| **AB-6** | **L_mp 修复**（取消 `top_k` 门控） | inter 训练改善；若目标是 per-image 可区分的，则**无差异**（见 §3.1 勘误的实测表） | 有差异 ⇒ tie 结构确实踩中；无差异 ⇒ 该分支不受此缺陷影响 |
| **AB-7** | **无信号方向的生成器敏感性**（(C1) 的新风险） | 期望无害但**可能有害** | 若有害 ⇒ 需在无信号方向强制投影到源侧均值 |
| **AB-8** | **衰减校准**开关（(C4)） | 开更准 | $q$ 过小 ⇒ 衰减可忽略 |
| **AB-9** | **诚实性**：训练 gallery vs 测试 gallery 拟合 | 测试 gallery 虚高 | 若相近 ⇒ 恢复捕获的是可复用关系（**强正面证据**） |
| **AB-10** | **CRD** 开关 | 开更稳 | 通道拓扑结论不适用本数据集 |

---

## 9. 优先级与里程碑

| 优先级 | 项 | 预期 | 代价 |
|---|---|---|---|
| **P0** | CE3 修 `inter-subject` 崩（3 行） | 解除硬阻断 | 极低 |
| **P0** | CE2 放开多被试 CLI（5 行） | 解除硬阻断 | 极低 |
| **P0** | CE1 被试轴 + Identity fallback（15 行） | 解除硬阻断 + None 基线 | 低 |
| **P0** | CE4 修 `top_k` 截断（= L_mp，10 行） | **只在 inter 下发作的核心缺陷** | 低 |
| **P0** | 陷阱 1/2 修指标 + 邻域库换 CSLS | 使数字可比、防 hub 塌缩 | 低 |
| **P0** | §7.3 四行基准 | 建立可比性（当前最阻塞） | 低 |
| **P1** | **SEA + SW + AS + RP + BC**（H1-mid 全链） | **最大单点杠杆** | 中 |
| **P1** | §7.2 诚实性协议（训练 gallery + A/B） | 使上项可信 | 低 |
| **P1** | **AB-1 / AB-2 / AB-3**（三个形态决策） | 决定最终架构 | 低-中 |
| **P2** | L_spec / L_aug / align 增强 | (C3) 的训练化 | 中 |
| **P2** | AB-7 生成器敏感性 | 排除 (C1) 的新风险 | 低 |
| **P3** | CRD、CE6 放开 detach、≥5 折扩展 | 稳健性 / 统计功效 | 中-高 |

### 建议的第一个里程碑

**P0 全部（6 项）+ P1 的 H1-mid 全链 + AB-1/2/3。**

理由：
* P0 里 **5 项是纠错而非改进**（3 个硬阻断 + `top_k` 截断 + 2 个指标口径），代价近乎零。
* 其中 **CE4 只在 inter 设置下发作**——不修它，后续所有 inter 实验都建立在一个被削弱的基线上。
* P1 是全文档证据最强的杠杆，且**全为闭式、无需重训**，可直接叠加在已有的 CogCapPro 产物上。
* AB-1/2/3 决定架构最终形态（是"传感空间单算子"还是"隐空间四算子"，
  是"杠杆锚点"还是"MNN 锚点"），**必须先定型再投算力**。

跑完这一步就能回答最关键的问题：**「一个 63 维传感空间的旋转，能否同时校正 4 个 1024 维分支」
（AB-1）**，并确定恢复位置（AB-2）与锚点策略（AB-3）。

---

## 10. 风险与证伪条件

| 风险 | 症状 | 应对 / 证伪 |
|---|---|---|
| **H1 不成立** | AB-1 显示单 $Q_s$ 只能校正部分分支 | 本文核心假设失败 ⇒ 退回逐分支隐空间恢复（上一版方案）。**这是有价值的负结论** |
| **传感空间不可达** | $Q_s$ 拟合后各分支残差下降很少 | $P_s$ 的值域超出 $O(63)$ 可达方向 ⇒ 用非正交 $Q_s$，或退回隐空间 |
| **(C3) 改善无效** | AB-5 无变化 | 说明 $A_s$ 的拉伸不是主要误差源 ⇒ 重新审视 SCORE 的 ridge/orthogonal 差距来源 |
| **(C2) 不是瓶颈** | AB-3 无改善 | 概念活跃子空间维度低，MNN 已足够 ⇒ 简化流程 |
| **(C1) 的无信号方向有害** | AB-7 显示像素层指标退化 | 强制把无信号方向投影到源侧均值再生成 |
| **锚点纯度太差** | $q$ 高、$\Delta$ 大、BC 频繁报警 | 提高投票门限 $K$、缩小锚点集换纯度；或退回共享 $R$ |
| **衰减校准过度** | 校准后反而变差 | $q$ 估计有偏 ⇒ 用交叉验证选校准强度 |
| **多模态冗余不成立** | AB-3 的投票项无改善 | 各分支误差高度相关（共享同一 EEG 噪声源）⇒ 多模态只提供 1 个有效分数场 |
| **L_aug 破坏概念信息** | 检索 Top-1 下降 | 旋转增强把信号也当冗余 ⇒ 减小 $\beta$ 或改为**保留概念 RSM** 的更强约束 |
| **L_spec 与任务冲突** | 检索指标下降 | 谱平坦与"任务需要判别性方向"存在张力（§1.4 C3 注）⇒ 只在前 $r_{\text{eff}}$ 内施加 |
| **i.i.d. 假设破裂** | `trial_subject` 相等假设失效致索引错位 | 加断言：`set(len(d['eeg']) for d in loaded_data)` 应为单元素 |
| **统计功效不足** | CI 宽于增益 | ≥5 折；先 3 折看方向 |

---

## 11. 来源

### 外部
- **CogCapPro**：`arXiv:2603.12722`（2026-03-13，Zhang, He, Ke, Ji, Wu, Wang, Gao / Xidian）——
  自述两个病根：**fidelity loss** 与 **representational shift**（本文形式化的对象）；
  三大组件：uncertainty-weighted masking、fusion encoder、asymmetric alignment。
  THINGS-EEG 上 Top-1/Top-5 较 CognitionCapturer 提升 **25.9% / 10.6%**
- **CogCapPro 代码**：`github.com/XiaoZhangYES/CognitionCapturerPro`，commit `cf3fc5b`，
  已 clone 至 `third_party/CognitionCapturerPro/`
- **SCORE**：`arXiv:2608.19134`（Cui, Kan, Li, Wang, Wu, HUST）——RSM $0.687\pm0.035$（支持 A1）；
  正交 28.22 vs ridge 20.01（支持 (C3)）；累积消融 26.22→35.78→48.75→53.23；
  不相交 gallery 迁移 +24.63（支持 §7.2）
- **SATTC**：CVPR 2026 / `arXiv:2603.20738`——SAW 9.2/30.5 → 13.7/36.4；
  N=50 无标签试次达 N=200 的 94.8%；**强编码器上完整算子反而变差 26.12 < 26.22** ⇒ 只取 SAW
- **SIMON**：`arXiv:2605.00401`——inter 19.6/49.9；通道拓扑（后部被试特异、广覆盖被试不变）⇒ CRD
- **ENIGMA**：`arXiv:2602.10361`——多被试重建 SOTA（前一份方案主干）；2WC 口径见其附录 A.3
- **NeurIPS 2025 / Alljoined-1.6M**：`arXiv:2508.18571`——跨方法复现对照

### 本仓库代码级证据
- 被试轴空操作：`third_party/CognitionCapturerPro/src/cogcappro/models/brain_backbone.py:166-176`（`[0]` + 官方注释 `# how to deal with this`），`:200`（`num_subjects=1`）
- `inter-subject` 崩溃：`.../cli/train.py:85`（`ModelCheckpoint(save_last=True)` 无 monitor）vs `:117`（`ckpt_path="best"`）
- 多被试 CLI 受限：`.../runtime/paths.py:321`（`[args.subjects]`）+ `cli/train.py:18`（`type=str`）
- LOSO 切分已实现：`.../data/eeg.py:53-65`（`all_subjects = [f'sub-{i:02}' for i in range(1,11)]`）
- 跨被试全局刺激身份：`.../data/eeg.py:157-164`（`img_path_to_idx` 跨所有被试）
- `top_k` 截断：`.../utils.py:296-310`（`topk(k=10)` ⊙ `mask_class`）；调用 `training/module.py:234`
- 逐样本被试标签已存在未被用：`.../data/eeg.py:431`（`sample['subject']`）
- 空间坍缩：`.../models/brain_backbone.py:40-48`（`Conv2d(40,40,(channel_num,1))`），`hidden_dim=1440=36×40`
- align 目标 = IP-Adapter 条件：`.../generate_image/generator.py:270`（`_image_to_embedding`）+ `align/data.py:129-145`
- align 输出 L2 归一化：`align/diffusion_pipe.py`（`SimpleAlignMLP.forward`）
- **已核实非缺陷**：本环境 `diffusers/models/embeddings.py: ImageProjection` 含 `nn.LayerNorm`
  ⇒ IP-Adapter 先做 per-token LayerNorm，输入幅值被消掉，只有方向有效 ⇒ 尺度错配不构成缺陷
- IP-Adapter 3× 同权重 + per-block scale：`.../generate_image/generator.py:88-101, 143-155`
- 融合 `.detach()`：`.../training/module.py:213`
- 不确定度三档仅训练期生效：`.../data/eeg.py:377-408`（测试恒为 `'medium'`）⇒ 对 inter 部署零贡献
- 陷阱 1（SSIM proxy）：`eeg-brainit/src/eeg_brainit/utils/metrics.py: ssim_simple`
- 陷阱 2（Alex 同名异实）：`eeg-brainit/scripts/erdc_full_metrics.py: _pearson_flat` vs `erdc_twoway_metrics.py: twoway`
- 缺陷 B（邻域库裸 argsort）：`eeg-brainit/scripts/erdc_ras_closed_loop.py:64-71`
- 2WC 口径（200-way / 199 干扰项）：`erdc_twoway_metrics.py:42-60`
- 现有锚点：oracle 0.456/0.164、head 0.3435/0.1257、identity 0.2502/0.0478（CLIP cosine / PixCorr）
- 生成栈同源：`eeg-brainit/scripts/erdc_official_atm_pipeline.py:99-114`
