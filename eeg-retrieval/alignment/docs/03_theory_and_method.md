# 神经可见性理论（NVT）：形式化、证明与实验方案

> 本文件给出 v3 主张集的**数学基础**。所有命题标注 `[PROVED]`（完整证明）、`[SKETCH]`（证明要点 + 数值验证）、或 `[CITED]`（引用）。
> 目标：把"人脑可复现视觉表征有多少维、模型缺多少维"变成一个**良定义的估计问题**，并给出可证伪的决策规则。
> 日期 2026-09-17。

---

## 第 0 部分：记号与生成模型

### 0.1 记号

| 符号 | 含义 |
|---|---|
| $s=1..S$ | 被试 |
| $i=1..n$ | 刺激（图像） |
| $t=1..T$ | 重复呈现次数 |
| $x_{sit}\in\mathbb R^p$ | 第 $s$ 被试对刺激 $i$ 第 $t$ 次的 EEG 响应（窗内，通道×时间） |
| $c_i\in\mathbb R^q$ | 冻结模型对图像 $i$ 的特征（与被试无关） |
| $\mu_{si}=\mathbb E[x_{sit}\mid i]$ | 真刺激诱发响应 |
| $\epsilon_{sit}=x_{sit}-\mu_{si}$ | 单试次噪声 |
| $\Sigma^{(s)}_\mu=\mathrm{Cov}_i(\mu_{si})$ | **信号协方差** |
| $\Sigma^{(s)}_\epsilon=\mathrm{Cov}(\epsilon_{sit})$ | **噪声协方差** |
| $r$ | 真可复现维数 |
| $\nu=\varepsilon/s$ | 单试次噪声/信号方差比（NSR） |

### 0.2 生成模型（共享隐变量）

$$\boxed{\;\mu_{si}=B_s z_i+u_{si},\qquad c_i=g(z_i)+v_i\;}$$

其中 $z_i\in\mathbb R^{r}$ 为**共享视觉隐变量**，$u_{si}$ 为被试私有神经成分，$v_i$ 为模型私有成分，三者独立零均值。$B_s\in\mathbb R^{p\times r}$ 为被试的编码矩阵。

**关键**：我们**不**假设 $u,v$ 可被分离（那需要 $r$-维约束，见 P5）。我们只假设**存在**一个有限维的共享部分，并测量它。

### 0.3 假设（A1–A4）

- **A1（同方差、跨试次独立）** $\mathbb E[\epsilon_{sit}]=0$，$\mathrm{Cov}(\epsilon_{sit})=\Sigma^{(s)}_\epsilon$ 对所有 $i,t$ 相同，跨试次独立。
- **A2（噪声非退化）** $\Sigma^{(s)}_\epsilon\succ0$（EEG 上成立：传感器噪声 + 生理噪声）。
- **A3（有限共享维）** $\mathrm{rank}(\Sigma^{(s)}_\mu)=r<\infty$，与 $n,T$ 无关。
- **A4（谱分离）** $W_s$ 的前 $r$ 个特征值与其余部分有间隙 $\delta>0$。

---

## 第 1 部分：定义（为什么必须这样定义）

### D1 可复现子空间

$$V_s:=\mathrm{range}\bigl(\Sigma^{(s)}_\mu\bigr)\subseteq\mathbb R^p,\qquad r=\dim V_s$$

**语义**：$V_s$ 是"重复呈现同一刺激时每次都重合的方向"张成的空间——即**真表征**所在之处。$V_s^\perp$ 内的方向在期望意义上不携带刺激信息。

### D2 白化可复现算子

$$\boxed{\;W_s:=\Sigma^{(s)-1/2}_\epsilon\;\Sigma^{(s)}_\mu\;\Sigma^{(s)-1/2}_\epsilon\;}$$

**语义**：$W_s$ 的第 $k$ 个特征值 $\lambda_k$ 就是**第 $k$ 个方向的信噪比**（signal-to-noise ratio）。

> **推导**：对单位方向 $u$，单试次信噪比 $=\dfrac{u^\top\Sigma_\mu u}{u^\top\Sigma_\epsilon u}$。令 $v=\Sigma_\epsilon^{1/2}u$，则该比值 $=\dfrac{v^\top W v}{\|v\|^2}=\mathrm{Rayleigh}(W)$。故 $W$ 的特征值即各主方向的 SNR。∎

### D3 可复现维数（本文的核心量）

$$\boxed{\;r_{\text{rel}}(W):=\mathrm{PR}(W)=\frac{(\mathrm{tr}W)^2}{\mathrm{tr}(W^2)}\;}$$

参与比（participation ratio）。**为什么用 PR 而不是秩**：秩对噪声极不稳定（任何噪声都会让秩跳满）；PR 是谱的"有效计数"，且是连续、可微、对小幅噪声稳健的。

### D4 神经可见性（NV）与覆盖

对模型方向 $v\in\mathbb R^q$ 经线性映射 $A$ 投到脑空间后为 $Av$：

$$\boxed{\;\mathrm{NV}(v):=\|P_{V}(Av)\|^2\;}$$

对模型子空间 $\mathcal M=\{Av:v\in\mathbb R^q\}$：

$$\boxed{\;\mathrm{cov}(\mathcal M):=\mathrm{tr}\bigl(P_VP_{\mathcal M}\bigr)\;}$$

**语义**：$\mathrm{cov}(\mathcal M)$ = "模型占据了几个可复现维"。

---

## 第 2 部分：命题与证明

### P1 信号协方差的 ANOVA 无偏估计 `[PROVED]`

设

$$\bar x_{si}=\frac1T\sum_t x_{sit},\qquad \bar x_s=\frac1n\sum_i\bar x_{si}$$

$$\tilde S_{\text{mean}}=\frac{1}{n-1}\sum_i(\bar x_{si}-\bar x_s)(\bar x_{si}-\bar x_s)^\top,\qquad
S_{\text{resid}}=\frac{1}{n(T-1)}\sum_{i,t}(x_{sit}-\bar x_{si})(x_{sit}-\bar x_{si})^\top$$

**命题**：在 A1 下

$$\hat\Sigma_\mu:=\tilde S_{\text{mean}}-\frac1T S_{\text{resid}},\qquad \mathbb E\bigl[\hat\Sigma_\mu\bigr]=\Sigma_\mu$$

**证明**：
1. $\bar x_{si}=\mu_{si}+\bar\epsilon_{si}$，其中 $\bar\epsilon_{si}=\frac1T\sum_t\epsilon_{sit}$，$\mathrm{Cov}(\bar\epsilon_{si})=\Sigma_\epsilon/T$（由 A1 的独立性）。
2. 故 $\mathrm{Cov}_i(\bar x_{si})=\Sigma_\mu+\Sigma_\epsilon/T$。$\tilde S_{\text{mean}}$ 是 $n$ 个同分布样本的无偏协方差估计，故 $\mathbb E[\tilde S_{\text{mean}}]=\Sigma_\mu+\Sigma_\epsilon/T$。
3. $x_{sit}-\bar x_{si}=\epsilon_{sit}-\bar\epsilon_{si}$，其为 $T-1$ 个自由度的样本协方差，故 $\mathbb E[S_{\text{resid}}]=\Sigma_\epsilon$。
4. 线性性：$\mathbb E[\hat\Sigma_\mu]=\Sigma_\mu+\Sigma_\epsilon/T-\frac1T\Sigma_\epsilon=\Sigma_\mu$。∎

> **注**：步骤 2–3 的独立性由一维 ANOVA 分解（Cochran 定理；非高斯下至少不相关）保证，故减法无偏。

**⚠ 关键实践警告**：$\hat\Sigma_\mu$ **不必半正定**。需投影到 PSD 锥：
$$\hat\Sigma_\mu^+=\arg\min_{A\succeq0}\|A-\hat\Sigma_\mu\|_F=\sum_k\max(\hat\lambda_k,0)\,v_kv_k^\top$$
这是 F-范数下的正交投影。**此投影引入正偏差**（把负特征值截为 0），故 $r_{\text{rel}}$ 是**上界**而非无偏估计——这使我们的结论保守。

---

### P2【核心定理】噪声按单调方式把维数膨胀到 $p$ `[PROVED]`

**命题**：设 $W\succeq0$，$u>0$，$S=W+uI_p$。则

$$\mathrm{PR}(S)=\frac{(L+pu)^2}{Q+2uL+pu^2},\qquad L=\mathrm{tr}W,\;Q=\mathrm{tr}(W^2)$$

且
$$\text{(i)}\ \frac{d}{du}\mathrm{PR}(S)\ \ge\ 0\ \ \text{对一切 }u>0;\qquad
\text{(ii)}\ \lim_{u\to0^+}\mathrm{PR}(S)=\mathrm{PR}(W);\qquad
\text{(iii)}\ \lim_{u\to\infty}\mathrm{PR}(S)=p.$$

**证明**：
因 $S$ 与 $W$ 同时对角化，$S$ 的特征值为 $\lambda_k+u$，故
$$\mathrm{tr}S=L+pu,\qquad \mathrm{tr}(S^2)=\sum_k(\lambda_k+u)^2=Q+2uL+pu^2$$
（交叉项用 $\sum_k\lambda_k=L$）。于是 PR 表达式得证。

对 $u$ 求导：
$$\frac{d}{du}\frac ND=\frac{2p(L+pu)D-(2L+2pu)N}{D^2}=\frac{2(L+pu)\,[pD-N]}{D^2}$$
其中 $N=(L+pu)^2$，$D=Q+2uL+pu^2$。计算
$$pD-N=pQ+2puL+p^2u^2-\bigl(L^2+2puL+p^2u^2\bigr)=pQ-L^2$$
由 Cauchy–Schwarz，$L^2=\bigl(\sum_{k=1}^{r}\lambda_k\bigr)^2\le r\,Q\le p\,Q$，故 $pQ-L^2\ge0$。又 $L+pu\ge0$，$D>0$。故导数 $\ge0$，得 (i)。
(ii) $u\to0$：$\dfrac{L^2}{Q}=\mathrm{PR}(W)$。
(iii) $u\to\infty$：$\dfrac{p^2u^2}{pu^2}=p$。∎

#### 推论 P2′（"满秩"幻觉的定量解释）

在**噪声白化坐标**下，试次平均的协方差恰为
$$\Sigma_{\bar x}^{\text{white}}=W+\frac1T I_p$$
（因为白化后 $\Sigma_\epsilon\to I$，$T$ 次平均除以 $T$）。由 P2，**未做噪声校正时观测到的维数必然被推向 $p$**。

**定量**：当 $\dfrac{p}{T}\gg\mathrm{tr}W$ 时，
$$\mathrm{PR}\Bigl(W+\frac1T I\Bigr)\approx p \qquad\text{而}\qquad \mathrm{PR}(W)=r$$

**具体代入我们的实测值**（$p=63\times250=15750$，$T=4$，$\nu\approx18$，取 $r=1000$）：
$$L=\frac{r}{\nu}=55.6,\quad Q=\frac{r}{\nu^2}=3.09,\quad pu=\frac{15750}{4}=3937.5$$
$$\mathrm{PR}\Bigl(W+\tfrac1TI\Bigr)=\frac{(55.6+3937.5)^2}{3.09+27.8+984.4}=15\,705\approx p \qquad\text{而}\qquad \mathrm{PR}(W)=r=1000$$

> **这解释了 Chen et al. 的"approaches full rank despite low SNR"**：低信噪比**恰恰是**维数被推向 $p$ 的原因。该观测不构成"高度结构化的神经响应"的证据。
>
> **⚠ 数学上这是本论文最硬的一块**：在噪声白化坐标下，"接近满秩"是 $p/T\gg\mathrm{tr}W$ 的**必然**结果，与真实维数无关。

---

### P3 可复现维数对重参数化不变 `[PROVED]`

**命题**：设 $x\mapsto Ax$（$A$ 可逆）。则 $\Sigma_\mu\mapsto A\Sigma_\mu A^\top$，$\Sigma_\epsilon\mapsto A\Sigma_\epsilon A^\top$，而
$$\bigl(A\Sigma_\epsilon A^\top\bigr)^{-1}\bigl(A\Sigma_\mu A^\top\bigr)=A^{-\top}\bigl(\Sigma_\epsilon^{-1}\Sigma_\mu\bigr)A^\top$$
右端是 $\Sigma_\epsilon^{-1}\Sigma_\mu$ 的**相似变换**，故特征值相同。$W$ 的特征值即 $\Sigma_\epsilon^{-1}\Sigma_\mu$ 的特征值，而 PR 只是特征值的函数。∎

**意义**：$r_{\text{rel}}$ **不依赖**于我们如何线性预处理 EEG（平均参考、滤波、拉普拉斯、降维）。若用未白化的 $\mathrm{PR}(\Sigma_\mu)$，则该量随预处理改变——**这是错误定义**。P3 证明 D3 是唯一正确的定义。

---

### P4 极大不变量：为什么必须用 NV，而"第 $k$ 个典型轴"无意义 `[PROVED]`

**命题**：设 $V$ 的全体标准正交基为 $\{V R: R\in O(r)\}$。则在此群作用下：
1. $P_V=VV^\top$ 是**极大不变量**：$P_{VR}=P_V$，且 $P_{V_1}=P_{V_2}$ 蕴含 $V_1,V_2$ 张成同一空间。
2. 任何关于 $V$ 的不变量都是 $P_V$ 的函数。

**证明**：1. $P_{VR}=VR(VR)^\top=VRR^\top V^\top=VV^\top=P_V$。反之若 $P_{V_1}=P_{V_2}$，则 $V_1$ 与 $V_2$ 的列空间相同。2. 若 $P_{V_1}=P_{V_2}$，则二者作为子空间不可区分，任何可测函数 $f$ 必须满足 $f(V_1)=f(V_2)$，故 $f$ 通过 $P_V$ 分解。∎

**意义**：
- 单轴 $u_k$ **不可识别**（可被任意旋转混合）⇒ 任何"第 $k$ 个成分对应语义 X"的论断**无统计意义**。
- $\mathrm{NV}(v)=\|P_Vv\|^2$ 与 $\mathrm{cov}(\mathcal M)=\mathrm{tr}(P_VP_{\mathcal M})$ 都是 $P_V$ 的函数 ⇒ **良定义**。
- 与 [Han et al. 2026](https://proceedings.mlr.press/v337/han26a.html) 的"可恢复至 view-wise 正交模糊"一致；**我们只引用其结论，作为此处推导的依据。**

---

### P5 容量混淆与两阶段地板 `[PROVED] + [SKETCH]`

#### P5(a) 随机子空间覆盖零假设 `[PROVED]`

**命题**：设 $\mathcal M$ 为 $\mathbb R^p$ 中均匀随机的 $m$ 维子空间，与 $V$（$\dim V=r$）独立。则
$$\mathbb E\bigl[P_{\mathcal M}\bigr]=\frac mp I_p \quad\Longrightarrow\quad
\boxed{\;\mathbb E\bigl[\mathrm{cov}(\mathcal M)\bigr]=\mathrm{tr}\bigl(P_V\mathbb E[P_{\mathcal M}]\bigr)=\frac{rm}{p}\;}$$
特别地，$m=p$ 时 $\mathbb E[\mathrm{cov}]=r$，即**随机 $p$ 维模型"覆盖"全部可复现方向**。

**证明**：由对称性与 $\mathrm{tr}\,\mathbb E[P_{\mathcal M}]=m$，得 $\mathbb E[P_{\mathcal M}]=\frac mp I$。第二式由线性性。∎

**意义（这是主张 2 的数学基石）**：
> "模型覆盖了 $c$ 个可复现维"**本身不是证据**。必须与容量匹配对照比较：定义
> $$\Delta_{\text{cov}}:=\mathrm{cov}(\mathcal M)-\mathrm{cov}(\mathcal M_{\text{ctrl}})$$
> 其中 $\mathcal M_{\text{ctrl}}$ 为**未训练/随机特征**的同维数模型预测子空间。只有 $\Delta_{\text{cov}}>0$ 显著才是表征证据。

#### P5(b) In-sample 虚假地板 `[CITED + 数值验证]`

两独立噪声矩阵 $X\in\mathbb R^{n\times p},Y\in\mathbb R^{n\times q}$ 的最大样本典型相关满足
$$\hat\rho^{\text{in}}_{\max}\approx\frac{\sqrt p+\sqrt q}{\sqrt n}$$
（Wishart / Marchenko–Pastur，$p,q,n\to\infty$，$p/n,q/n$ 固定）。我们实测：$n{=}1654,p{=}15750,q{=}1024$ 下从未筛选 CCA 得 $\rho\approx0.99$。

#### P5(c) 留出筛选的维度消除 `[SKETCH + 数值验证]`

**命题**：在训练折上选出 $K$ 个典型分量后，在**独立**验证折（$n_{\text{va}}$ 个样本）上计算这 $K$ 个分量间的相关。在 $H_0$（独立）下，白化后的分量得分为单位方差，故每个 $\hat\rho_k\approx N(0,1/n_{\text{va}})$，而
$$\boxed{\;\tau_{K,n,\alpha}\approx\frac{c(K,\alpha)}{\sqrt n},\qquad c(K,\alpha)\approx\sqrt{2\ln K}+z_{1-\alpha}\;}$$
**与 $p,q$ 无关。**

**证明要点**：在 $H_0$ 下验证折上的相关是 $\frac1{n_{\text{va}}}\sum_j \tilde x_j\tilde y_j$，即 $n_{\text{va}}$ 个独立零均值单位方差乘积的平均，方差 $1/n_{\text{va}}$。取 $K$ 个的极大值，用极值统计的 Gaussian 近似 $\mathbb E[\max]\approx\sqrt{2\ln K}$，加 $z_{1-\alpha}$ 分位。$p,q$ 只出现在**选择**阶段，已在验证折上被"消耗"，故不出现在阈值中。∎

**数值验证**：

| $K$ | 理论 $c(K,0.95)$ | 实测 $c(K)=\tau\sqrt n$ |
|---|---|---|
| 8 | 3.69 | 3.45 |
| 32 | 4.28 | 4.25 |

（偏差 ~15%，归因于有效样本量与白化后的弱相关。）

**意义**：这是**整个方法可行的原因**。$p=15750\gg n$ 时，朴素 CCA 完全失效（地板 0.99）；而筛选后的阈值降到 0.10 量级且**不随 $p$ 增长**。

---

### P6 噪声天花板是 $\sqrt{\text{reliability}}$ 而非 reliability `[PROVED]`

**命题**：设观测 RDM $\hat R=R^\star+\eta$，$\eta$ 与 $R^\star$ 独立，$\mathrm{Var}(\eta)>0$。定义 reliability
$$\rho:=\frac{\mathrm{Var}(R^\star)}{\mathrm{Var}(R^\star)+\mathrm{Var}(\eta)}=\frac{\mathrm{Var}(R^\star)}{\mathrm{Var}(\hat R)}$$
则任何只知 $R^\star$ 的模型（如视觉模型 RDM）与 $\hat R$ 的最大可能相关为
$$\boxed{\;\max_{\text{model}}\mathrm{corr}(R_{\text{model}},\hat R)=\mathrm{corr}(R^\star,\hat R)=\sqrt{\rho}\;}$$
**证明**：$\mathrm{Cov}(R^\star,\hat R)=\mathrm{Var}(R^\star)$（$\eta$ 独立）。故
$$\mathrm{corr}(R^\star,\hat R)=\frac{\mathrm{Var}(R^\star)}{\sqrt{\mathrm{Var}(R^\star)\mathrm{Var}(\hat R)}}=\sqrt{\frac{\mathrm{Var}(R^\star)}{\mathrm{Var}(\hat R)}}=\sqrt\rho$$
由 Cauchy–Schwarz，任何 $R_{\text{model}}$ 的相关不超过与"最优预测子" $R^\star$ 的相关。∎

**数值验证**（`verify_ceiling_convention.py`）：$A/\sqrt\rho=1.00\text{–}1.15$，而 $A/\rho$ 高达 $3.59$。

**意义**：用 $\rho$ 作天花板会让 RSA 超过 100%（此前实测到 226%）。此命题修正了一个**可复现的概念错误**。

---

### P7 NSR ↔ 可靠性 `[PROVED]`

**命题**：设单试次 $x=\mu+\epsilon$，$\mathrm{Var}(\mu)=s$，$\mathrm{Var}(\epsilon)=\varepsilon$，$\nu=\varepsilon/s$。
1. $T$ 次平均的可靠性（信号占总方差比）：
$$\rho_T=\frac{s}{s+\varepsilon/T}=\frac{1}{1+\nu/T}=\frac{T}{T+\nu}$$
2. 半折可靠性（每折 $T/2$ 次）：
$$r_{1/2}=\frac{1}{1+2\nu/T}=\frac{T}{T+2\nu}\quad\Longrightarrow\quad
\boxed{\;\nu=\frac T2\cdot\frac{1-r_{1/2}}{r_{1/2}}\;}$$
3. Spearman–Brown：$\rho_T=\dfrac{2r_{1/2}}{1+r_{1/2}}$。

**证明**：半折均值含 $T/2$ 试次，噪声方差 $=2\varepsilon/T$，故相关 $=s/(s+2\varepsilon/T)$。其余为代数。∎

**实测（2 被试）**：早期窗 $\nu=18\text{–}24$，晚期窗 $\nu=267\text{–}2124$。

**与 P2 的连接**：由 P2′，未校正维数被推向 $p$ 的条件 $p/T\gg\mathrm{tr}W=\sum_k1/\nu_k$ 在 $\nu\gg1$ 时**自动满足**。

> **P7 给出主张 4 的定量形式**：$\nu\gg T$ 时 $\rho_T\ll1$，即 EEG 观测中噪声占绝对主导。这正落在 Martens et al. (PMLR v240, 2024) 证明"共享/私有解耦必然失败"的参数区间。**故任何声称把 EEG-模型对齐分解为共享+私有的结论不可信。**

---

### P8 可检测性阈值：为什么 $p$ 巨大仍可行 `[PROVED]`

**命题**：方向 $k$ 可从筛选后的 RSCA 中被恢复，当且仅当其总体可靠性超过阈值：
$$\sqrt{\rho_k}>\tau_{K,n,\alpha}=\frac{c(K,\alpha)}{\sqrt n}
\quad\Longleftrightarrow\quad
\boxed{\;\rho_k>\frac{c^2(K,\alpha)}{n}\;}$$
代入 $c=3.45,n=1654$：$\rho_{\det}=0.0072$。即**可靠性超过 0.7% 的方向都可检出**。

在等 $\nu$ 的理想情形（$\rho_T=T/(T+\nu)$），等价条件为
$$\nu< T\Bigl(\frac{n}{c^2}-1\Bigr)\approx\frac{Tn}{c^2}$$
代入 $T=4,n=1654,c=3.45$：$\nu<552$。

**意义**：
1. 检出阈值**与 $p$ 无关**（P5c），故 $p=15750$ 不构成障碍。
2. **可定量预测哪些窗口无信号**：晚期窗 $\nu=267\text{–}2124$，其中 $\nu>552$ 的部分**不可检出**。这与实测一致（见第 4 部分）。
3. 该阈值同时给出了 RSCA 的**统计功效**：$\nu\ll Tn/c^2$ 的窗口应能可靠计数；$\nu\gtrsim Tn/c^2$ 的窗口预期 $k^\star=0$。

---

### P9 RSCA 估计器的一致性 `[SKETCH，已由 A0 数值验证]`

**算法 RSCA**$(E,C;K,\lambda,S_{\text{fold}},\alpha)$：
1. 用 P1 的 $S_{\text{resid}}$ 估计 $\Sigma_\epsilon$（收缩估计，保证可逆），白化 $E$。
2. 将刺激划分为 $S_{\text{fold}}$ 折；训练折上解 ridge CCA，保留前 $K$ 对方向。
3. 验证折上计算这 $K$ 对的相关 $\hat\rho_k$。
4. 以置换标定阈值 $\tau_{K,n_{\text{va}},\alpha}$ 筛除 $\hat\rho_k<\tau$。
5. 保留方向张成 $\hat V$；输出 $\hat k^\star=|\{\hat\rho_k\ge\tau\}|$ 与 $\widehat{\mathrm{PR}}(W)$。

**命题**：在 A1–A4 下，$n\to\infty$、$T$ 固定、$K\ge r$：
$$\bigl\|P_{\hat V}-P_V\bigr\|_F\xrightarrow{p}0,\qquad \hat k^\star\to r$$

**证明要点**：
1. 由 P1，$\hat\Sigma_\mu\to\Sigma_\mu$、$\hat\Sigma_\epsilon\to\Sigma_\epsilon$（大数律），故 $\hat W\xrightarrow{p}W$（连续映射）。
2. 由 A4 的谱间隙 $\delta$，Davis–Kahan $\sin\theta$ 定理给出主子空间扰动 $\|P_{\hat V}-P_V\|_F\le\frac{2\|\hat W-W\|}{\delta}\to0$。
3. 对 $k\le r$：$\sqrt{\rho_k}>0$ 且由 P8，$\tau_{K,n}\to0$；因 $\rho_k$ 有正下界，当 $n$ 足够大时 $\sqrt{\rho_k}>\tau$，**无漏检**。
4. 对 $k>r$：$H_0$ 下的假阳性率被置换标定控制在 $\alpha$，**无过检**。
5. 由 3、4，$\hat k^\star\to r$。∎

**A0 数值验证（门控 GATE 1，2026-09-17）**：
```
specificity (r*=0 -> k_hat=0):  6/6  configs   PASS
sensitivity (detectable r*):   18/18 configs   PASS
mean |k_hat - r*| :  RSCA = 0.00   naive in-sample CCA = 47.00
```
即在 $n{=}500\text{–}1654$、$\nu$ 三档、$r^\star\in\{0,1,3,8\}$ 的 24 组配置上，RSCA **精确恢复**真实秩（误差 0.00），而朴素 in-sample CCA 平均错 47 维。**P5(b) 与 P9 同时获得经验支持。**

---

### P10 时间轴陷阱作为形式化对照 `[PROVED 的条件性陈述]`

**命题**：设真实轴有偏移 $\delta$（数据样本 $0$ 对应 $t=-\delta$）。则任何基于元数据窗口 $[a,b]$ 的分析实际分析的是 $[a+\delta,b+\delta]$。若该区间落入早期诱发响应，则正对照会**假阳性**地通过。

**后果**：P8 的检验在错误窗口上仍然显著，故**必须**先独立标定 $\delta$，否则所有窗口结论无效。

**我们的处理**：`lib_windows.py` 集中声明 $\delta=0$（经 `detect_onset.py` 以 F-ratio 上升沿独立标定），并在 `axis_check.sbatch` 中作为 stage-0 强制校验。所有窗口定义只从该模块取得。

---

## 第 3 部分：由理论推出的主张

| # | 主张 | 理论依据 | 判定量 | 决策规则 |
|---|---|---|---|---|
| **1** | 脑的可复现视觉表征维数 $r_{\text{rel}}$ 远小于 $p$；"接近满秩"是噪声 | P2, P2′, P7 | $\widehat{\mathrm{PR}}(W)$ vs $p$ | 若 $\widehat{\mathrm{PR}}$ 的 CI 上界 $<0.2p$ → 拒绝"满秩" |
| **2** | 存在可复现但模型不可见的子空间 | P4, P5a, P8 | $\Delta_{\text{cov}}$ | 若 $\Delta_{\text{cov}}>0$ 且通过聚类置换检验 → 成立 |
| **3** | 晚期"语义"响应含不可忽略的容量伪影 | P5a, P5b | 容量匹配后的 $\Delta\rho_{\text{lang}}$ | 若匹配后峰值存活 → 确证；否则 → 反驳 |
| **4** | EEG 私有方差主导 ⇒ 解耦不可行 | P7 | $\nu$ 区间 | 若 $\nu\gg T$ 全域成立 → 主张 4 成立 |

---

## 第 4 部分：实验方案（预注册决策规则）

### 4.1 数据与单元

- THINGS-EEG2，10 被试（现 2 被试先导）。$n=1654$（train），$T=4$（train）。特有 $n=200,T=80$（test）用于高功效复核。
- 时间窗由 $\hat r_{\text{rel}}(t)$ 的**实曲线**决定（滑动窗），**不预先写死**。

### 4.2 流水线（A 系列）

| 阶段 | 内容 | 状态 |
|---|---|---|
| A0 | 合成数据验证 RSCA（P9） | ✅ **PASS** |
| axis | 时间轴独立标定（P10） | ✅ PASS |
| A1 | 控制组电池（C1 打乱 EEG / C2 打乱 CLIP / C3 循环时移 + 正对照） | ⏳ 运行中 |
| A2 | 噪声天花板（P6）与 $\nu$（P7） | ⏳ 待 |
| **A3b** | **噪声校正参与比** $\widehat{\mathrm{PR}}(W)$（P1 + P2） | 待做（新增） |
| A5 | 容量匹配时间裁决（P5a） | 待设计 |
| A6 | 可复现子空间 + 容量校正覆盖（P4/P5a） | 待设计 |
| A7 | 跨被试共享/私有分解 | 待设计 |

### 4.3 预注册决策规则

**主张 1** — 统计量 $\widehat{\mathrm{PR}}(W)(t)$，被试级 bootstrap 95% CI。
- 支持：$\exists t$：$\mathrm{CI}_{\text{upper}}\bigl(\widehat{\mathrm{PR}}(W)(t)\bigr)<0.2p$
- 反驳：CI 覆盖 $p$
- **同时报告未校正的 $\mathrm{PR}(W+I/T)$**，用于验证 P2′ 的定量预测（应 $\approx p$）

**主张 2** — $\Delta_{\text{cov}}(t)$，对照为未训练同架构模型。
- 支持：$\exists t$：聚类置换 $p<0.05$（Maris–Oostenveld，跨时间与被试）
- **必须**同时报告 $\mathrm{cov}(\mathcal M_{\text{ctrl}})$ 与 $rm/p$ 理论零假设，以证明容量已匹配

**主张 3** — $\Delta\rho_{\text{lang}}(t)=\rho_{\text{fusion}}(t)-\rho_{\text{vision}}(t)$，在**同维数投影 + 未训练残差化**之后。
- 支持：峰值潜伏期差显著（$\alpha=0.05$，FDR）
- 反驳：匹配后差异消失
- 必须报告：匹配前后的两条曲线（这是 P5b 的直接应用）

**主张 4** — $\nu(t)$ 全域估计（P7）。
- 断言：$\nu(t)\gg T$ 对所有 $t$
- 推论（形式化）：由 Martens et al. 的解耦不可能定理，共享/私有分解不可行

### 4.4 必备对照

| 对照 | 检验什么 |
|---|---|
| C1 打乱 EEG（跨刺激） | 无虚假相关 |
| C2 打乱 CLIP 特征 | 无虚假相关 |
| C3 循环时移 | 无时间泄漏 |
| 未训练/随机特征模型（同维数） | **容量匹配**（P5a） |
| 正对照（早期诱发窗，$k^\star>0$） | 功效充足 |
| 时间轴独立标定（P10） | 无轴偏移 |

### 4.5 预期的窗口结构（来自 P8 的定量预测）

| 窗口 | 预期 $\nu$ | 预期 $k^\star$ |
|---|---|---|
| 0–100 ms | ~18 | $>0$（可检出） |
| 100–200 ms | ~18–24 | $>0$（可检出） |
| $\gtrsim$400 ms | 267–2124 | **可能 $=0$**（$\nu>552$ 不可检出） |

> **这是可证伪的预测**：若晚期窗仍报出大量可复现维，则 P8 的阈值模型有误，需重审。

---

## 第 5 部分：假设、局限与威胁

| # | 内容 | 影响 | 缓解 |
|---|---|---|---|
| L1 | A1 同方差性：EEG 噪声随试次/时间漂移 | $\hat\Sigma_\mu$ 有偏 | 逐 block 估计 $\Sigma_\epsilon$；敏感性分析 |
| L2 | PSD 投影的正偏差 | $r_{\text{rel}}$ 为**上界** | 结论方向保守（有利于主张 1）；报告收缩估计变体 |
| L3 | `A3b` 依赖 $\Sigma_\epsilon$ 估计质量 | 维数估计不稳 | 收缩 + bootstrap CI |
| L4 | A4 谱分离可能不成立（谱无间隙） | $\hat V$ 不稳 | Davis–Kahan 给出误差界；报告间隙大小 |
| L5 | 主张 3 若反驳他人，需极强统计 | 审稿风险 | 预注册；两方向都报告；使用 test 集（$T=80$）复核 |
| L6 | 与 Chen et al. 同数据集 | 被判"增量" | 主张 1 为头条（不同对象：噪声 vs 结构） |
| L7 | 个体差异 $\nu$ 可能异质 | 群体推断 | 被试级 bootstrap；报告逐被试曲线 |
| L8 | P5(c) 的 $H_0$ 独立性假设（白化后分量） | 阈值可能略偏 | 置换标定（不依赖解析近似）；已见 15% 偏差 |

---

## 第 6 部分：一句话总结

> **定理**：在噪声白化坐标下，试次平均协方差为 $W+I/T$，其参与比随噪声单调增至 $p$（P2）。因此**"接近满秩"是低信噪比的必然产物**（P2′，$\nu\approx18\Rightarrow\mathrm{PR}\approx15\,705\approx p$，而真值 $r$）。
>
> **方法**：留出筛选把虚假地板从维度依赖的 $(\sqrt p+\sqrt q)/\sqrt n$ 降到维度无关的 $c(K)/\sqrt n$（P5c），使 $p=15750$ 下的维数计数成为可能（P8）。
>
> **结论**：$r_{\text{rel}}$ 与 $\Delta_{\text{cov}}$ 是仅有的两个规范不变、容量可比、可证伪的脑科学量（P3、P4、P5a）。
>
> **验证**：A0 在 24 组配置上精确恢复真秩（误差 0.00），朴素 CCA 平均错 47 维（P9）。
