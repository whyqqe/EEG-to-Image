# xCogCap：把 CogCapPro 改造成 inter 重建模型的完整方案

> **xCogCap** = cross-subject CogCapPro
> 主干：CogCapPro（`arXiv:2603.12722`，2026-03，官方代码 `XiaoZhangYES/CognitionCapturerPro`，commit `cf3fc5b`，已克隆到
> `third_party/CognitionCapturerPro/`）。
> 设计立场：**不替换主干**。先精确审计「它已经有什么 / 缺什么」，只在真正的缺口上叠加 inter 组件。
> 所有判断标注来源；未标来源者为**待验证假设**并附证伪条件。

---

## 0. 结论先行

**CogCapPro 的 inter 脚手架比 ENIGMA 好得多，但它有一个反直觉的性质：
它已经内置了「跨被试多正样本」这套 SCORE 认为最值钱的训练机制，
却在真正开启 inter 设置的那一刻，被一个 `top_k=10` 静默破坏掉。**

三个核心判断：

1. **不用重造多正样本**。`img_index` 是跨被试全局刺激身份（`img_path_to_idx` 跨所有被试构建），
   class-mask 软标签在 9 源被试同训时**自动**把同图不同被试的试次变成正样本。这是 ENIGMA 和 SAMGA 都没有的。
2. **但它同时有一个只在 inter 下才发作的缺陷**：`top_k=10` 会把「同图的 35 个其它被试试次」**截断到 10 个**，
   剩下 25 个被当作**负样本推开**。单被试时（同图只有 3 个其它试次）不发作。
   ⇒ **开启 inter 训练会静默损伤效果**（§2.4，代码级推导）。
3. **真正的 inter 缺口是「目标被试坐标系 + 跨模态一致性」，而后者是 CogCapPro 独有的新问题**（§3）。
   4 个模态分支各自学到不同的映射 ⇒ 目标被试有 **4 个不同的旋转误差**，而不是 1 个。

---

## 1. 代码级审计：CogCapPro 已经有什么（不要重造）

### 1.1 完整前向链路（逐段核实）

```text
EEG [B,63,250]
  │
  ├─ EEGAttention            transformer over TIME，d_model=63(=通道数)，PE 在时间轴
  │                          （brain_backbone.py:93-103）
  ├─ subject_wise_linear[0]  ⚠️ Linear(250,250)，硬编码 index 0，注释 "how to deal with this"
  │                          （brain_backbone.py:172-175）
  │
  └─ ×4 独立分支（modality_num=4）—— 每个模态一套独立参数的 Cogcap
        Enc_eeg = PatchEmbedding: Conv2d(1,40,(1,25)) → AvgPool(1,51),(1,5)
                                  → Conv2d(40,40,(63,1)) ← 63 电极压成 1（空间坍缩）
                                  → flatten 1440
        Proj_eeg: 1440 → 1024 (Linear+ResidualAdd+LayerNorm)
     ↓
  z_m ∈ R^1024  for m ∈ {image, text, depth, edge}      ← ★ 记作「空间 (1)」
     │
     ├─ CogcapFusion ×2：fusion_eeg(4 个 z_m .detach()) ；fusion_mod(图像侧特征, 随机遮蔽 0~3 个模态)
     │    每模态 proj → stack 成 token → TransformerEncoder(2 层) → mean → MLP → 1024
     │
     ├─ 损失 ClipLoss_Modified_DDP(top_k=10, cos_batch=512)
     │    软标签 = normalize( top-k(文本相似度) ⊙ (img_index 相同) )     （utils.py:262-320）
     │
     └─ [可选] MultiModalDiffusionPrior：4 个独立 UNet
            ↓
          SimpleAlignNet：concat(4 模态) → MLP → 4 个 head → **L2 normalize**
            ↓
          {image, depth, edge} ∈ S^1023                              ← ★ 记作「空间 (3)」
            ↓
          IPAdapterGenerator：SDXL-Turbo，**3 个 IP-Adapter 载同一权重**
            ip-adapter_sdxl_vit-h.safetensors × 3（generator.py:146）
            每模态独立 per-block scale（generator.py:88-101）：
              image: down.block_2=[1,1]      up.block_0=[1,1,1]   ← 全开
              depth: down.block_2=[0,0.5]    up.block_0=[0,0,0]
              edge : down.block_2=[0,0.5]    up.block_0=[0,0,0]
            条件 = stack([zeros, embed]) 逐模态（CFG 式），guidance_scale=0.0
```

### 1.2 已有资产清单（inter 相关的）

| 资产 | 位置 | inter 价值 |
|---|---|---|
| **`exp_setting='inter-subject'` 模式** | `data/eeg.py:53-65` | ✅ **LOSO 切分已实现**：test=目标被试，train/val=`all 10 − 目标` |
| **多被试数据集** | `data/eeg.py:108,154-167` | ✅ `subjects` 是 list，逐被试 `.pt`，`trial_all_subjects = trial_subject × n_subj` |
| **跨被试全局刺激身份** | `data/eeg.py:157-164` | ✅ **`img_path_to_idx` 跨所有被试构建** ⇒ 同图不同被试 → 同一 `img_index` |
| **跨被试多正样本** | `utils.py:296-310` | ✅ class-mask 软标签在 9 源同训时自动生效（**但见 §2.4 缺陷**） |
| **逐样本被试标签** | `data/eeg.py:431` | ✅ `sample['subject'] = subject` 已在 batch 里，只是没被模型用 |
| **训练/测试试次平均** | `cogcappro.yaml:35-36` `train_avg/test_avg: True` | ✅ 与 ENIGMA 相反，它**确实做了**（ENIGMA 只在 test 做） |
| **不确定度感知三档特征** | `data/eeg.py:377-408` | ⚠️ **仅训练期**（`if self.mode == 'train'`），测试恒为 `'medium'` ⇒ **对 inter 部署无贡献** |
| **分阶段训练 + 冻结** | `module.py:on_train_epoch_start` | ✅ stage1 单模态 / stage2 联合 / stage3 仅融合 |
| **模态掩蔽增强** | `module.py:_random_mask_non_eeg_modals` | ✅ 随机置零 0~3 个非 EEG 模态 |
| **align 阶段目标 = IP-Adapter 条件本身** | `generate_image/generator.py:270` + `align/data.py:129-145` | ✅ **`h` 由真图经 `prepare_ip_adapter_image_embeds` 得到** ⇒ align 输出空间**就是** IP-Adapter 条件空间 |

> **1.2 最后一条是这份设计的关键杠杆。** `prepare_embedding()` 把**真实图像的 image/depth/edge 渲染图**
> 送进 IP-Adapter 的 CLIP ViT-H 图像编码器，得到 align 阶段的监督目标 `h`。
> ⇒ 这意味着 align 阶段的输出空间与我们要对齐的 gallery **同一个空间、同一套权重**，
> 而且 gallery 对任意图片都可直接计算。坐标恢复因此有了一个**天然的锚**。

### 1.3 与 `eeg-brainit` 生成栈的关系

`IPAdapterGenerator` 用 `ip-adapter_sdxl_vit-h.safetensors`，权重与我们的
`ip-adapter_sdxl_vit-h.bin` **同源**（同一 `h94/IP-Adapter` 仓库的同一模型，仅序列化格式不同），
SDXL-Turbo、`guidance_scale=0.0` 也一致。
⇒ **目标空间是同一个 CLIP ViT-H/14 1024-d 空间**，我们已有的
`clip_img_{train,test}_1024.npy`（`open_clip ViT-H-14 laion2b_s32b_b79k`）**可直接当 gallery 复用**。

⚠️ 两处口径差异必须记住：
- CogCapPro 的 align 输出是 **L2 归一化**的（`SimpleAlignMLP.forward` 末尾 `F.normalize`），
  而 `h` 是 `prepare_ip_adapter_image_embeds` 的**未归一化**输出。二者**存在尺度不一致**（§4.3 要处理）。
- CogCapPro 默认 `vision_backbone=RN50`（`z_dim=1024`），论文的更强配置是 `ViT-H-14`（`z_dim=1024`）。
  **两者 z_dim 都是 1024**，所以切换到 ViT-H-14 不改变任何维度。建议直接用 ViT-H-14。
- 生成步数：`generator.py` 默认 `num_inference_steps=5`，其 `main()` 里用 15；我们统一到 4~5 并固定。

---

## 2. 代码级审计：真正缺什么、什么坏了

### 2.1 【硬阻断 1】被试轴是个空操作

```python
# brain_backbone.py:166-176
self.subject_wise_linear = nn.ModuleList(
    [nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
...
x = self.subject_wise_linear[0](x) # how to deal with this     ← 硬编码 [0]
```

* 实例化时 `EEGProjectLayer_multimodal_cogcap_list` 传 **`num_subjects=1`**（`brain_backbone.py:200`）
  ⇒ `ModuleList` 只有一个元素，`[0]` 恒被取用 ⇒ **这个层变成了一个所有被试共享的、无条件的 `Linear(250,250)`**。
* 官方注释 `# how to deal with this` 是作者留下的自认未解问题。
* 注意：它是 `Linear(250,250)`，作用于**时间轴**（与 ENIGMA 的 `subject_wise_linear` 同病），
  **不是**一个被试身份嵌入。

⇒ **结论：CogCapPro 没有任何被试特异参数化。** 它与被试相关的一切都只来自
``Preprocessed_data_250Hz_whiten`` 的逐被试预处理。

### 2.2 【硬阻断 2】CLI 只接受一个被试

```python
# runtime/paths.py:321
config["data"]["subjects"] = [args.subjects]        # args.subjects 是 str，默认 "sub-08"
```

`--subjects` 是 `type=str`（`cli/train.py:18`），被包成单元素 list。
⇒ **多被试训练无法从命令行开启**，`data/eeg.py` 里的多被试逻辑虽有实现但走不到。
要走 inter 必须先把这个改掉（§6 的 CE2）。

### 2.3 【硬阻断 3】`inter-subject` 分支在 test 时会崩

```python
# cli/train.py:85-121
checkpoint_callback = ModelCheckpoint(save_last=True)     # ← 没有 monitor ⇒ 不会保存 "best"
...
if config["exp_setting"] == "inter-subject":
    test_results = trainer.test(ckpt_path="best", ...)     # ← "best" 不存在
```

`ModelCheckpoint(save_last=True)` 未指定 `monitor` ⇒ `best_model_path` 为空
⇒ `ckpt_path="best"` 会抛 `MisconfigurationException`。
⇒ **`exp_setting=inter-subject` 这条路径从未被真正跑通过**（这也解释了为什么 README 里 grep 不到任何
inter-subject 字样，`--exp_setting` 默认值是 `intra-subject`）。

### 2.4 【核心缺陷】`top_k=10` 在 inter 训练下静默截断跨被试正样本

`ClipLoss_Modified_DDP.forward` 的软标签构造（`utils.py:296-310`）：

```python
values, indices = torch.topk(sim.fill_diagonal_(0), k=min(self.top_k, N-1), dim=1)  # k=10
mask_sim.scatter_(1, indices, 1).fill_diagonal_(1)
mask_class = self._get_class_mask(gathered_img_index)     # img_index 相同
sim_mask = (mask_sim * mask_class).float()                # ← 交集
labels = sim_mask / row_sums
```

`sim` 是**文本特征**的余弦相似度。同一张图 ⇒ 同一概念 ⇒ 同一文本 ⇒ `cos = 1`（并列最大值）。

**逐配置推导**（`train_batch_size = 1024`，THINGS-EEG2 训练集每图 4 次重复）：

| 配置 | 每图试次行数 | 同图「其它」行数 | top-10 保留 | **被误当负样本推开的同图行** |
|---|---:|---:|---:|---:|
| **单被试**（原论文） | 4 | 3 | 3（全保留） | **0** |
| **inter（9 源同训）** | 4 × 9 = 36 | 35 | **10** | **25** |

⇒ **单被试时完全不发作；一开 inter 就发作，且每次有 25/36 的同图正样本被当作负样本压低。**

更糟的是这些被误判的行 logit 是**最大值**（文本 cos=1），却被赋予 label 0
⇒ 它们在 `F.cross_entropy` 里贡献**最大**的惩罚梯度。这是一个**主动把跨被试对齐破坏掉**的机制。

**修复**（§6 的 CE3）：交集改并集，或直接从 `img_index` 构造正样本集合，让 `top_k` 只用于挑非正样本硬负例：

```python
pos = mask_class.bool()                                  # 直接来自 img_index，不做 top-k
neg_topk = mask_sim.bool() & ~pos
mask = (pos | neg_topk).float()                          # 并集
```

> 这条是本方案里**最高性价比、且只在 inter 设置下才显现**的修复。
> 它也解释了 SCORE 为什么把 multi-positive 单列为一项增益（+2.41）；CogCapPro 有这项机制，
> 但需要先修掉这个截断才能真正拿到收益。

### 2.5 【结构缺陷】空间坍缩

`PatchEmbedding`（`brain_backbone.py:40-48`）：`Conv2d(40, 40, (channel_num, 1))`
把 63 个电极用核高 63 **一次压成 1** ⇒ 无电极拓扑、无位置编码。
`hidden_dim = 1440 = 36 × 40`（`Proj_eeg(embedding_dim=1440)`），**没有空间轴**。

与 ENIGMA / SAMGA 完全同病。对**像素保真**（PixCorr/SSIM）是结构性上限。
标记为 P3（需重训），依据是 v2 文档 §2.4 四篇论文收敛的通道拓扑结论。

### 2.6 其它次要问题（记录备查，不必先修）

| 问题 | 位置 | 影响 |
|---|---|---|
| `fusion_eeg` 对所有模态 `.detach()` | `module.py:213` | 融合无法反传到编码器；inter 下若想让融合成为跨被试装置，需要放开 |
| `leave_one_subjects_config = config` 别名后原地改 `subjects` | `data/eeg.py:60-61` | 别名隐患：后续 `load_model(config,...)` 拿到的是被改过的 config |
| `trial_subject = loaded_data[0]['eeg'].shape[0]` | `data/eeg.py:166` | 假设各被试试次数完全相等；若有被试丢试次会导致索引错位，需加断言 |
| `n_cls = 1654 if train else 200` | `data/eeg.py:150` | 1654 与 THINGS 的 1854 概念数不符，疑为笔误（不影响本方案） |
| val 用**源被试**的 test split | `data/eeg.py:62` | 无目标泄漏，但早停依据是源被试检索 ⇒ 与目标被试性能可能不相关 |
| 不确定度三档在测试期全部退化为 `'medium'` | `data/eeg.py:389-390` | 该机制对 inter 部署**零贡献**（不要指望它） |

---

## 3. 理论分析：inter 差距的三个来源

记共享编码器为 $E_\theta$（4 个分支 $E_m$ + 融合 $F$ + align $A$），源被试集合 $\mathcal{S}$，目标被试 $t$。

### 3.1 (G1) 输入分布漂移

头皮/阻抗/参考电极差异 ⇒ $x_t$ 与 $x_s$ 的边缘分布不同。
CogCapPro 依赖 `Preprocessed_data_250Hz_whiten` 的逐被试白化，它消除**逐通道二阶统计**，
但**不消除被试特异的空间模式**（电极→源的映射差异）。

⇒ 处理方式：**SAW（subject-adaptive whitening）**，用目标被试**无标签**试次估全协方差白化
（严格泛化了「居中 + 尺度」）。依据：SATTC 9.2/30.5 → **13.7/36.4**；SCORE 在同一编码器上受控测量 26.22 → **30.98**。
校准成本极低：**N=50 个无标签试次即达 N=200 上界的 94.8%**。

### 3.2 (G2) 坐标系旋转 —— 主项

SCORE 的诊断（THINGS-EEG2，同一 LOSO 折）：

* 源与目标的 **EEG RSM 相关性 = 0.687 ± 0.035** ⇒ 概念**结构是共享的**。
* 用留出概念 3 折交叉验证做 target EEG → source EEG 映射：

| 对齐方式 | Top-1 | Top-5 |
|---|---:|---:|
| 直接匹配 | 16.89 | 41.95 |
| Ridge（任意线性） | 20.01 | 47.15 |
| **正交（仅旋转+反射）** | **28.22** | **59.16** |

**约束更强（正交）反而比 Ridge 高 8.21 Top-1** ⇒ 该变换**本质是一次坐标旋转**，Ridge 的多余自由度在过拟合。

**部署时闭式恢复的实测增益（同一编码器、双方冻结、目标零标签）**：

| 步骤 | Top-1 | 增量 |
|---|---:|---:|
| 裸用 | 26.22 | — |
| + CSLS 重标定 | 35.78 | **+9.56** |
| + 正交坐标恢复（闭式） | **48.75** | **+12.97** |
| + 恢复感知训练 + identity 正则（完整 SCORE） | **53.23** | +4.48 |

⇒ **这是全文最大单点杠杆，且是闭式解、不需要训练。**

### 3.3 (G3) 跨模态坐标系不一致 —— **CogCapPro 独有的新问题**

**这是本方案相对 ENIGMA 方案的核心理论增量。**

CogCapPro 有 **4 个独立参数的 EEG 分支**（`modality_num=4`，各自一套 `Cogcap`）。
每个分支学到的映射 $E_m$ 不同 ⇒ 目标被试在**每个分支上各有自己的旋转误差** $R_m \in O(1024)$，
**不是单一旋转 $R$**。

#### 3.3.1 一致性到底需不需要担心？—— 一个需要澄清的伪问题

直觉上会担心：融合网络 $F$ 是**非线性**的（每模态 proj → stack 成 token → Transformer → mean → MLP），
它对「各模态被独立旋转」**不具等变性**，所以独立 $R_m$ 会破坏融合。

**但这个担心用错了判据。** 正确的判据不是等变性，而是**输入分布匹配**：

> $F$ 的输出正确 ⟺ 它的**输入元组分布**与训练时一致。

训练时 $F$ 的输入是源被试的 $z_m$（近似 CLIP 帧）。测试时若我们把目标被试的 $z_m$ 用 $R_m^{-1}$ 校正回 CLIP 帧，
则 $F$ 的输入重新落在训练分布内 ⇒ **$F$ 的输出自动正确，完全不需要 $F$ 本身等变。**

⇒ **独立 $R_m$ 在理论上是成立的。** 这个澄清把「4 模态一致性」从一个看似难解的约束降级为一个工程细节。

#### 3.3.2 由此得到「在哪个空间做恢复」的判据

我们把所有冻结的下游模块的**共同输入空间**找出来：

| 下游冻结模块 | 它的输入 |
|---|---|
| `CogcapFusion.fusion_eeg` | 4 个 $z_m$（空间 1） |
| `MultiModalDiffusionPrior`（可选） | `condition_embeds = eeg_z`（空间 1） |
| `SimpleAlignNet` | `concat` 4 个 $z_m$ + fusion（空间 1） |

**三者全部消费空间 (1)。** 而空间 (3)（align 输出）只是空间 (1) 经一个冻结非线性模块的结果。

⇒ **判据：把恢复放在空间 (1)（EEG 编码器输出，逐模态）。**
一次校正即让**所有**下游冻结模块的输入回到分布内；放在空间 (3) 则只能修 align 之后，
融合与（可选的）扩散先验仍然看到被旋转的输入。

*推论*：$R_m$ 在空间 (1) 的**锚（gallery）**是 `ProjMod_m(CLIP_m(image))` —— 即训练损失里
`mod_z[m]` 的那一侧。对任意图片可计算，且与训练目标定义一致。

*推论*：由于空间 (1) 未做 L2 归一化（只有 LayerNorm），匹配/打分需先归一化；
但**正交映射与归一化可交换**（$\|Rz\|=\|z\|$ ⇒ $R(z/\|z\|) = Rz/\|Rz\|$），
所以「用归一化后的向量拟合 $R$，再把 $R$ 原样作用到未归一化的 $z$」是**严格等价且安全**的。

#### 3.3.3 独立 $R_m$ vs 共享 $R$：一个有意义的取舍

| 方案 | 优点 | 缺点 |
|---|---|---|
| **独立 $R_m$**（推荐主方案） | 每个分支的误差被各自校正；理论上是正确对象（§3.3.2） | 每模态各自需要足够 landmark；$m \ll d=1024$ 的欠定问题在每模态上更严重 |
| **共享 $R$（$R_m \equiv R$）** | landmark 池扩大 4 倍 ⇒ Procrustes 估计方差下降；identity 正则更有效 | 无法表达分支间的真实差异 |

⇒ **这是一个可测量、可证伪的取舍，应作为核心消融**（§8 AB-2）。
两条路都在维度上合法（4 个模态的 $z_m$ 与目标**全部是 1024-d**）。

### 3.4 多模态带来的一个新优势：landmark 冗余

恢复的质量取决于 landmark 的**数量**与**纯度**。SCORE 只有 1 个分数场（EEG↔图像），
而 CogCapPro 给我们 **4 个（+fusion 5 个）独立分数场** $s_m(i,j)$。

这带来三个可用的机制：

1. **投票提纯**：只有当一个对应 $(i,j)$ 在 $\ge k$ 个模态上同时是互最近邻（MNN）时才收为 landmark
   ⇒ 大幅降低误配率（EEG 信噪比低，单模态 MNN 误配率天然高）。
2. **landmark 池化**：为估计共享 $R$ 而把 4 个模态的 landmark 合并 ⇒ 样本量 ×4。
3. **一致性自检**：$R_m$ 之间应当高度一致（若各模态的误差真来自「同一个被试坐标系」）。
   若 $\{R_m\}$ 彼此差异巨大，说明恢复失败或某分支被噪声主导 ⇒ **这是一个无需标签的失败检测器**。

> Procrustes 估计误差随 landmark 数与误配率变化，故上述三点都是**可量化的预测**，
> 而非定性期望。见 §8 AB-3。

### 3.5 文本分支是一个「被试不变的锚」

文本特征只依赖概念名，**跨被试完全相同**（`h_text = f(concept)`，与被试无关）。
⇒ 文本分支的**监督目标不含被试噪声**，它是 4 个分支里最「被试无关」的一个。

两个用法：

1. **概念级 landmark**：文本分支给出的是概念级对应。THINGS-EEG2 test 是 200 概念 × 1 图，
   与实例级**恰好重合** ⇒ 测试集上文本分支与图像分支同等判别力。
   （但在 16540 图 / 1854 概念的训练 gallery 上，文本分支有概念歧义 ⇒ **仅用于测试集或一致性检查**。）
2. **无标签验证 $R$**：文本特征跨被试共享 ⇒ 可用「校正后的目标被试文本分支 RSM
   与概念层级的相似结构」是否恢复来判断 $R$ 是否可信。**不需要任何标签或配对真值。**

### 3.6 一个有利条件：试次平均

目标被试每个测试图有 **80 次重复**，`test_avg: True` 会平均。
⇒ 目标被试的**单样本信噪比**远高于单次试次，
这使 landmark 选择（恢复流程的第一步）比在原始单试次上可靠得多。
**这是我们相对 SCORE 设定（若其未做同等平均）的一个结构性优势**，应在报告中说明。

---

## 4. 理论总结：一个「四层差距模型」

| 层 | 差距 | 量级依据 | 对应组件 | 代价 |
|---|---|---|---|---|
| **L-A** | 输入分布漂移（一阶+二阶） | SAW +4.76 | SAW（逐模态） | 低 |
| **L-B** | 坐标系旋转（主项） | +9.56(CSLS) / +12.97(正交) | 矩匹配 + CSLS landmark + 加权正交恢复 + identity 正则 | 中（闭式） |
| **L-C** | 跨模态坐标系不一致 | 理论上被 §3.3.1 消解；残差需实测 | 逐模态 $R_m$（主）/ 共享 $R$（消融） | 低 |
| **L-D** | 空间坍缩（像素保真上限） | 无直接文献，4 篇收敛 | 通道角色分解 | **高（重训）** |

---

## 5. 架构总览

```text
┌── 阶段 0：主干训练（源被试，9 人；对官方代码的最小侵入改动）──────────────┐
│  EEG [B,63,250]                                                        │
│    │                                                                   │
│    ├─ EEGAttention (官方)                                               │
│    ├─ subject_wise_linear[subject_id]   ← ★CE1 修好被试轴（源被试真用上）│
│    │                                      目标被试走 Identity fallback   │
│    └─ ×4 独立 Cogcap 分支 → z_m ∈ R^1024                                │
│              │                                                          │
│              ├─ 损失：★CE3 修好 top_k 截断的跨被试多正样本              │
│              ├─ CogcapFusion（官方，可选放开 detach）                     │
│              └─ ★CE5 恢复感知 episode（把一个源被试当 pseudo-target）    │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌── 阶段 1：部署侧坐标恢复（目标被试；全无标签、全闭式、无参数训练）────────┐
│  ① 逐模态 SAW          W_m = (Σ_m + λI)^{-1/2}，目标被试无标签试次估      │
│  ② 逐维矩匹配          对齐 z_m 与 ProjMod_m(CLIP_m) 的均值/方差          │
│  ③ 多模态 landmark     CSLS 打分 → MNN → ★跨模态投票（k-of-4）           │
│                        → top1/top2 margin 加权                          │
│  ④ 逐模态正交恢复      R_m* = U Vᵀ,                                    │
│                        M = X̃ᵀ W Ỹ + λI = U Σ Vᵀ  (加权 Procrustes)     │
│                        λ = ρ‖X̃ᵀ W Ỹ‖₂, ρ = 0.1（identity 正则，不可省）│
│  ⑤ ★跨模态一致性自检   {R_m} 的相互距离 → 无标签失败检测                 │
│  ⑥ 在恢复后空间重算 CSLS                                                │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ 校正后的 z_m
┌── 阶段 2：冻结下游（零改动）─────────────────────────────────────────────┐
│  [可选] MultiModalDiffusionPrior → SimpleAlignNet → 3× IP-Adapter        │
│  SDXL-Turbo, guidance_scale=0.0, 每模态 per-block scale 保持官方设置      │
└──────────────────────────────────┬──────────────────────────────────────┘
┌── 阶段 3：生成与选择 ───────────────────────────────────────────────────┐
│  邻域库 retrieve_neighbors 换 CSLS（修 v2 缺陷 B，防 hub 塌缩）           │
│  条件不确定度驱动候选数                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**与 ENIGMA 方案（xENIGMA）的关键差异**：ENIGMA 的缺口是「**1 个**坐标系」（单分支 ⇒ 1 个 $R$）；
CogCapPro 的缺口是「**4 个**坐标系 + 它们之间的一致性」，但换来 **4 倍 landmark 与投票提纯能力**。
**净效果是未知的，必须实测**（§8 AB-1 直接对比两者）。

---

## 6. 组件设计

### 训练侧（源被试）

| 编号 | 组件 | 具体做法 | 优先级 |
|---|---|---|---|
| **CE1** | **修好被试轴** | ① `EEGProjectLayer_multimodal_cogcap_list` 传 `num_subjects=len(train_subjects)+1`；② `Cogcap.forward(x, subject_ids)` 按 id 取 `subject_wise_linear[i]`，源被试各用自己的，**目标被试（不存在）走 Identity fallback**；③ 从 `batch['subject']` 传入（字段已存在） | **P0** |
| **CE2** | **放开多被试训练** | `--subjects` 改为 `nargs="+"`；`paths.py:321` 改为 `config["data"]["subjects"] = list(args.subjects)`；修 `data/eeg.py:60-61` 的 config 别名 | **P0** |
| **CE3** | **修 `top_k` 截断** | `sim_mask = (mask_sim * mask_class)` → 直接由 `img_index` 构造正样本（并集），`top_k` 只用于选非正样本硬负例。**见 §2.4** | **P0** |
| **CE4** | **修 `inter-subject` 崩溃** | `ModelCheckpoint(save_last=True, monitor="val_top1_acc_fusion", mode="max", save_top_k=1)`，使 `ckpt_path="best"` 可用 | **P0** |
| **CE5** | **恢复感知 episode** | 每 mini-batch 抽 1 个源被试当 pseudo-target：隐藏其匹配 → 跑完整恢复流程 → 再算损失；**梯度不穿映射** | P2 |
| **CE6** | 放开融合的 `detach` | `fusion_eeg(*[e.detach() for e in eeg_z])` → 去掉 detach（可选；需实测是否损害 stage 划分） | P3 |
| **CE7** | 共享 alignment 层 | 把 `subject_wise_linear` 从「时间轴混合 `Linear(250,250)`」改为「隐空间对齐」，使其在缺层时有定义（与 xENIGMA 的 E4 同源） | P3 |

> **CE1 的 Identity fallback 是刻意的**：它让「不做被试对齐」成为显式的消融基线（None 行），
> 而且回避了「学一个被试编码器去逼近一个可闭式求解的几何问题」——
> SCORE 用闭式正交映射在同一编码器上拿到 +27.01，学习式被试码很难逼近它。

### 部署侧（目标被试；闭式、无标签）

| 编号 | 组件 | 公式 / 做法 | 优先级 |
|---|---|---|---|
| **CD1** | 逐模态 SAW | $W_m=(\Sigma_m+\lambda I)^{-1/2}$，$\tilde z = W_m(z-\mu_m)/\|W_m(z-\mu_m)\|_2$，用目标被试**无标签**试次估 $\Sigma_m,\mu_m$ | **P1** |
| **CD2** | 逐维矩匹配 | 对齐 $\tilde z$ 与 `ProjMod_m(CLIP_m)` 的逐维均值/方差（正交映射**管不了平移**） | **P1** |
| **CD3** | **多模态 landmark** | CSLS $\text{CSLS}(x,y)=2\cos(x,y)-r_G(x)-r_Q(y)$ → MNN → **k-of-4 跨模态投票** → margin 加权 | **P1** |
| **CD4** | 逐模态正交恢复 | $R_m^\star=\arg\min_{R^\top R=I}\|W^{1/2}(\tilde X R-\tilde Y)\|_F^2+\lambda\|R-I\|_F^2$，闭式 $R^\star=UV^\top$ | **P1** |
| **CD5** | 跨模态一致性自检 | 计算 $\{R_m\}$ 两两 $\|R_m R_{m'}^\top-I\|_F$；超阈值则回退到共享 $R$ 或退回 identity | **P2** |
| **CD6** | 恢复后重算 CSLS | 在 $R_m$ 校正并重新归一化后的空间重算检索 | **P1** |

**identity 正则为什么不可省**：部署 batch 只有几百个 query，landmark 数 $m \ll d=1024$，
无约束方向上是**欠定**的；$\lambda\|R-I\|_F^2$ 让无证据的方向留在 identity。
SCORE 的累积消融显示该项单独值 **+2.25**。

### 生成侧

| 编号 | 组件 | 依据 | 优先级 |
|---|---|---|---|
| **CG1** | 邻域库换 CSLS | 现有 `retrieve_neighbors` 是裸 `np.argsort(-sim)`（`eeg-brainit/scripts/erdc_ras_closed_loop.py:64-71`） | **P0** |
| **CG2** | 条件不确定度驱动候选数 | 低信噪比样本多采样 | P2 |
| **CG3** | 通道角色分解 | 前部=被试不变锚定、后部=被试特异细节（§2.5 空间坍缩） | P3（需重训） |

---

## 7. 部署算法（严格伪代码）

```python
# ---------- 输入 ----------
# A_S : 9 个源被试训练好的 CogCapPro（4 分支 + fusion + align）
# X_t_train : 目标被试【无标签】训练集 EEG（16540 图 × 4 重复），test_avg 后 16540 条
# X_t_test  : 目标被试【无标签】测试集 EEG（200 图 × 80 重复），test_avg 后 200 条
# G_train   : 16540 张训练图的 {ProjMod_m(CLIP_m(img))}  ← 与 200 测试刺激【不相交】
# G_test    : 200 张测试图的 {ProjMod_m(CLIP_m(img))}

# ---------- 步骤 1：逐模态 SAW（用目标被试无标签数据）----------
for m in (image, text, depth, edge):
    mu_m, Sigma_m = estimate(X_t_train, branch=m)
    W_m = inverse_sqrt(Sigma_m + lam * I)                    # lam 用缩尾估计
    z_hat_t_train[m] = normalize(W_m @ (z_t_train[m] - mu_m))

# ---------- 步骤 2：逐维矩匹配（对齐到 gallery 侧）----------
for m:
    z_hat[m] = moment_match(z_hat[m], G_train[m])            # 逐维 mean/var

# ---------- 步骤 3：landmark 选择（多模态 + 投票）----------
S = {}                                                       # S[m] : (N_t, N_g) 相似度
for m:
    S[m] = csls(z_hat_t_train[m], G_train[m])                # 2cos - r_G - r_Q
    MNN[m] = mutual_nearest_neighbor(S[m])
vote = sum(MNN[m] for m in modalities)                       # 0..4
landmarks = {i : argmax_j S[m][i,j]  for i where vote[i] >= K}   # K=2 起步，消融
weights = margin_weight(S, landmarks)                        # top1 与 top2 的间距

# ---------- 步骤 4：逐模态加权正交恢复 ----------
for m:
    X = z_hat_t_train[m][landmarks.keys()]                   # (n_lm, 1024)
    Y = G_train[m][landmarks.values()]
    M = X.T @ diag(weights) @ Y + lam_id * I                 # lam_id = rho * ||X.T W Y||_2
    U, _, Vt = svd(M)
    R[m] = U @ Vt                                            # 闭式 Procrustes 解

# ---------- 步骤 5：跨模态一致性自检（无标签）----------
if pairwise_defect({R[m]}) > tau:
    warn("recovery unreliable")                              # 或回退到共享 R / identity

# ---------- 步骤 6：应用到测试集，冻结 R ----------
for m:
    z_test[m] = normalize(R[m] @ apply_saw_and_match(X_t_test[m], m))

# ---------- 步骤 7：冻结下游 → 生成 ----------
cond = align_net(fusion_and_branches(z_test))                # 官方冻结模块
images = ip_adapter_generator(cond["image"], cond["depth"], cond["edge"])
```

**关键约束**：步骤 1–5 只用**无标签、无配对真值**的量；
$R_m$ 一旦在**训练 gallery**上拟合并冻结，**绝不在测试集上重新拟合**（§8 诚实性协议）。

---

## 8. 评测协议

### 8.1 三个必须先修的口径陷阱（来自我们自己的代码）

| 陷阱 | 问题 | 后果 |
|---|---|---|
| **SSIM 非论文级** | `eeg-brainit/src/eeg_brainit/utils/metrics.py: ssim_simple` 自述 *"lightweight … proxy for monitoring (not paper-grade)"*，只在全局算一个均值/方差，**无滑窗、无局部结构项** | 现有 LOSO SSIM **不可与文献比**，须换 `skimage.metrics.structural_similarity`（grayscale）重算 |
| **`alexnet2` 同名异实** | `erdc_full_metrics.py` 的 `alexnet2` 是 **Pearson 相关**；`erdc_twoway_metrics.py` 的 `alex2` 是 **200-way 2WC%** | 放同一张表是**假比较**；建议改名 `alexnet2_corr` / `alex2_2wc` |
| **缺 SwAV** | ENIGMA 报 `SwAV ↓`（SwAV-ResNet50 相关距离），我们**没有该分支** | 无法逐列对齐文献表 |

**好消息**：`erdc_twoway_metrics.py: twoway()` 用完整 `n=200` 池、每行遍历所有 $j\ne i$
⇒ 是 **200-way / 199 干扰项**，与 ENIGMA 附录 A.3 口径一致，**可直接比**，且比常见的「2-way / 1 干扰项」严格。
**必须在报告中标注。**

### 8.2 诚实性协议（本方案最重要的方法论约束）

重建指标是「生成图 vs 被试真实看过的图」。若用**同一批 200 概念**既拟合 $R_m$ 又评测，
$R_m$ 可能拟合到那 200 个特定配对上 ⇒ **抬高 PixCorr/CLIP 却无真实泛化**。

**主协议（比 SCORE 自身更严格）**：

1. **用目标被试的训练集 EEG（无标签）↔ 16540 张训练图 gallery 拟合 $R_m$。**
   训练图与 200 张测试刺激**完全不相交**（THINGS-EEG2：16740 = 16540 + 200）。
2. **冻结 $R_m$**，作用到测试集 EEG，生成并在 200 测试图上报告全部指标。
3. 这直接复刻 SCORE Table 6 的「不相交 gallery 迁移」设定（该设定下增益**更大**：+24.63 Top-1）。

**辅助报告**：

4. **A/B 概念切分**（各 100）：用 A 拟合、冻结、在 B 上报。
5. 全 200 拟合**只作参考行且必须标注**，不得作主结果。
6. 所有拟合**只用无标签量**：无被试标签、无配对真值。

### 8.3 四行基准 + oracle 归一化

仓库**没有任何跨被试重建基线** ⇒ 现有数字无法自答「好不好」。

| 行 | 作用 | 现状 |
|---|---|---|
| 随机条件 | 下界，扣掉生成器自带能力 | 待建 |
| **oracle（真 CLIP 条件）** | 上界，分离「条件误差」与「生成器饱和」 | ✅ 已有 0.456 / 0.164 |
| **被试内上界** | 同一套指标代码跑被试内，量化 inter 代价 | ⚠️ 有但受陷阱 1 影响需重算 |
| **xCogCap（LOSO）** | 待评 | 待跑 |

（锚点 = CLIP cosine / PixCorr）

$$\text{score}_{\text{norm}}=\frac{\text{score}-\text{chance}}{\text{oracle}-\text{chance}}$$

**分层报告是强制的**：条件层（CLIP cosine、检索 Top-1/5）与像素层（PixCorr、SSIM、FID、2WC）分开报。
否则条件层的改进会被生成器饱和掩盖（生成器会把任何条件渲成一张「像图」的图）。

### 8.4 统计功效

* sub-08 单折 CLIP 的 bootstrap CI 约 ±0.014，而待检测增益可能只有 0.02–0.03 ⇒ **≥5 折**。
* checkpoint 用 `last.pth` / final epoch（与 SCORE 一致）。
* 采样数固定（CogCapPro `generator.py` 默认 5 步；沿用并固定随机种子集）。

---

## 9. 消融矩阵（每条都是可证伪的预测）

| 编号 | 消融 | 预测 | 若否证 |
|---|---|---|---|
| **AB-0** | 基线：官方 CogCapPro + 修好的 inter 设置，无任何恢复 | 显著低于被试内 | — |
| **AB-1** | **xCogCap vs xENIGMA（单分支）** | 未知 —— 4 倍 landmark 与 4 个 $R_m$ 的净效果 | 若 xCogCap 更差 ⇒ 说明「单坐标系」假设下 CogCapPro 的多分支是负担，应改用共享 $R$ 或退到 ENIGMA |
| **AB-2** | **逐模态 $R_m$ vs 共享 $R$** | 共享 $R$ 在 landmark 少时更好（方差 ↓），逐模态在 landmark 多时更好 | 这是 §3.3.3 的核心取舍 |
| **AB-3** | **landmark：单模态 vs k-of-4 投票** | 投票显著提高纯度 ⇒ 恢复改善 | 若无改善 ⇒ §3.4 的冗余假设错误 |
| **AB-4** | **恢复空间：空间 (1) vs 空间 (3)** | 空间 (1) 更好（§3.3.2 的分布匹配判据） | 若空间 (3) 更好 ⇒ 说明下游冻结模块对输入旋转不敏感，判据需修正 |
| **AB-5** | **逐模态 SAW vs 全局 SAW vs 现有白化** | 逐模态更好（各分支分布不同） | — |
| **AB-6** | **CE3 修复（top_k 并集）** | inter 训练下明显改善 | 若无 ⇒ §2.4 的截断推导有误，需重查 batch 组成 |
| **AB-7** | **CE5 恢复感知 episode** | 放大恢复收益 | SCORE 该单项仅 +0.70，预期小 |
| **AB-8** | **诚实性：训练 gallery vs 测试 gallery 拟合** | 测试 gallery 数字虚高 | 若两者**相近** ⇒ 恢复确实捕获了可复用关系，是强正面证据 |

---

## 10. 优先级与里程碑

| 优先级 | 项 | 预期 | 代价 | 依据 |
|---|---|---|---|---|
| **P0** | CE4 修 `inter-subject` 崩溃 | 解除硬阻断 | ~3 行 | §2.3 |
| **P0** | CE2 放开多被试 CLI | 解除硬阻断 | ~5 行 | §2.2 |
| **P0** | CE1 修被试轴 + Identity fallback | 解除硬阻断 + 得到 None 基线 | ~15 行 | §2.1 |
| **P0** | CE3 修 `top_k` 截断 | **只在 inter 下发作的核心缺陷** | ~10 行 | §2.4 |
| **P0** | 陷阱 1/2 修指标（SSIM、Alex 改名） | 使数字可比 | 低 | §8.1 |
| **P0** | CG1 邻域库 CSLS | 可能同时改善 FID 与语义 | 低 | 裸 `argsort` 确认 |
| **P0** | §8.3 四行基准 | 建立可比性 | 低 | 当前无基线，最阻塞 |
| **P1** | **CD1–CD4、CD6：完整坐标恢复** | **最大单点杠杆** | 中 | +4.76 / +9.56 / +12.97 |
| **P1** | §8.2 诚实性协议（训练 gallery + A/B） | 使上项可信 | 低 | SCORE Table 6 |
| **P1** | AB-2 / AB-3 / AB-4（三个设计取舍） | 决定最终形态 | 低-中 | §3.3.3、§3.4 |
| **P2** | CD5 一致性自检、CE5 恢复感知 episode | 稳健性 / 小幅增益 | 中 | +0.70 |
| **P3** | CE6 放开 detach、CE7 共享对齐层、CG3 通道角色分解 | 未知 / 需重训 | 高 | §2.5、§2.6 |
| **P3** | 多折扩展（≥5） | 统计功效 | 中 | QOS 限 8 并发，需滴灌 |

### 建议的第一个里程碑

**P0 全部 + P1 的完整恢复（CD1–CD4、CD6）+ AB-2/AB-3/AB-4。**

理由：P0 里有 **5 项是纠错而非改进**（三个硬阻断 + `top_k` 截断 + 两个指标口径），代价近乎零，
且其中 **CE3 是一个只在 inter 设置下才发作的缺陷**——不修它，后续所有 inter 实验都在一个被削弱的基线上比较。
P1 是全文档证据最强的杠杆，且**全为闭式、无需重训**，可直接叠加在已有的 CogCapPro 训练产物上。

跑完这一步就能回答最关键的问题：**「多模态的 4 倍 landmark 与投票提纯，能否补偿 4 个独立坐标系的复杂度」**
（AB-1），并确定最终形态（AB-2/3/4）。

---

## 11. 风险与证伪条件

| 风险 | 症状 | 证伪 / 备选 |
|---|---|---|
| **恢复在 4 分支上不迁移** | 条件层 CLIP 提升但像素层不动 | 这是有价值的负结论；转 CG3（空间分支）或 AB-4 |
| **`top_k` 截断假设错误** | 修 CE3 后 inter 训练无变化 | 重查 batch 实际组成（打印每 batch 的 unique `img_index` 数与同图行数）；若确实无截断则 §2.4 推导需修正 |
| **align 输出与 gallery 不同帧** | 只用旋转拟合的 $R$ 效果差，但加矩匹配后变好 | 正是 §1.3 的尺度不一致警告；矩匹配（CD2）是必需的，不是可选的 |
| **多模态 landmark 冗余不成立** | AB-3 无改善 | 说明各模态误差高度相关（共享同一 EEG 噪声源），此时多模态只提供 1 个有效分数场 |
| **`{R_m}` 差异过大** | CD5 频繁报警 | 改用共享 $R$（AB-2 的另一侧） |
| **映射过拟合到 200 概念** | AB-8 显示训练 gallery 增益远小于测试 gallery | 已被 §8.2 强制暴露；若消失则说明恢复不可用 |
| **hubness 假设错误** | 换 CSLS 后多样性不动 | FID 反常需另找机制（量化类内/类间多样性） |
| **i.i.d. 假设被打破** | `trial_subject` 相等假设失效导致索引错位 | 加断言：`set(len(d['eeg']) for d in loaded_data)` 应为单元素 |
| **多折统计功效不足** | CI 宽于增益 | ≥5 折；先 3 折看方向 |

---

## 12. 与既有方案的关系

| 维度 | xENIGMA（前一版方案） | **xCogCap（本文）** |
|---|---|---|
| 主干 | ENIGMA（2026-02，多被试重建 SOTA） | **CogCapPro（2026-03，更新的重建 SOTA）** |
| 主干是否多模态 | ❌ 单条件（图像 CLIP） | ✅ **4 条件**（图像+文本+深度+边缘）+ 融合 |
| LOSO 脚手架 | ❌ 完全没有（`KeyError` 硬阻断） | ⚠️ **已实现但从未跑通**（`ckpt_path="best"` 崩溃） |
| 跨被试多正样本 | ❌ 缺失（`torch.arange`，§2.4 of xENIGMA） | ⚠️ **已有，但被 `top_k=10` 静默截断** |
| 需要恢复的坐标系数量 | **1 个** | **4 个**（+ 一致性） |
| landmark 冗余 | 1 个分数场 | **4~5 个分数场** ⇒ 可投票提纯 |
| 被试轴 | 逐被试 `Linear`，目标缺失 ⇒ 崩 | 逐被试 `Linear`，但**恒取 `[0]`（空操作）** |
| 恢复的锚 | 原生 CLIP 1024-d | 原生 CLIP 1024-d（且 align 目标**就是** IP-Adapter 条件） |
| 生成栈 | 与我们同构（零胶水） | 同族（3× IP-Adapter + per-block scale），需适配 |
| 训练试次平均 | ❌ 训练集未平均（论文声称有） | ✅ `train_avg: True` |

**共同保留的部分**：v2 文档的两个缺陷诊断（缺陷 B 邻域库 CSLS）、诚实性协议、评测协议、
「明确不借鉴清单」（SATTC 结构 PoE、HyFI 双曲、EA、v1 的 FiLM+被试编码器），
以及三篇外部方法（SCORE / SATTC-SAW / SIMON）的可借鉴结论。

---

## 附：引用来源

**外部**
- **CogCapPro**：`arXiv:2603.12722`（2026-03-13，Zhang, He, Ke, Ji, Wu, Wang, Gao / Xidian）
  — 多模态协同训练、uncertainty-weighted similarity、fusion encoder、asymmetric alignment；
  THINGS-EEG Top-1/Top-5 较 CognitionCapturer 提升 25.9% / 10.6%
- **CogCapPro 代码**：`github.com/XiaoZhangYES/CognitionCapturerPro`（commit `cf3fc5b`）；
  已 clone 至 `third_party/CognitionCapturerPro/`；读过的文件：
  `configs/cogcappro.yaml`、`src/cogcappro/models/{brain_backbone,fusion_backbone}.py`、
  `src/cogcappro/training/module.py`、`src/cogcappro/utils.py`、`src/cogcappro/data/eeg.py`、
  `src/cogcappro/align/{main,model,data}.py`、`src/cogcappro/align/diffusion_pipe.py`、
  `src/cogcappro/generate_image/generator.py`、`src/cogcappro/runtime/paths.py`、`src/cogcappro/cli/train.py`
- **第三方复现**：`github.com/Chikit-WONG/DL_Project`（Task 2 采用 CogCapPro；
  用 `ClipLoss_Modified_DDP`、`SimpleAlignPipe`，5 步 SDXL-Turbo、`guidance_scale=0.0`；
  THINGS-EEG2 SSIM 0.409 / CLIP-H 2WC 0.903，**被试内**）
- **SCORE**：`arXiv:2608.19134`（Cui, Kan, Li, Wang, Wu, HUST）— RSM 0.687±0.035；
  正交 28.22 vs Ridge 20.01；累积消融 26.22→53.23；Table 5/6（扩大 gallery / 不相交 gallery 迁移）
- **SATTC**：CVPR 2026 pp.16887-16896 / `arXiv:2603.20738` — SAW 9.2/30.5 → 13.7/36.4；
  **强编码器上完整算子反而变差（26.12 < 26.22）** ⇒ 只取 SAW
- **SIMON**：`arXiv:2605.00401` — inter 19.6/49.9；通道拓扑（后部被试特异、广覆盖被试不变）
- **NeurIPS 2025 / Alljoined-1.6M**：`arXiv:2508.18571` — 跨方法复现对照（ENIGMA/ATM-S/Perceptogram）
- **ENIGMA**：`arXiv:2602.10361` — 前一份方案（xENIGMA）的主干

**本仓库代码级证据**
- 被试轴空操作：`third_party/CognitionCapturerPro/src/cogcappro/models/brain_backbone.py:166-176`（`[0]` + 官方注释 `# how to deal with this`），`:200`（`num_subjects=1`）
- CLI 单被试：`:321`（`[args.subjects]`）+ `src/cogcappro/cli/train.py:18`
- `inter-subject` 崩溃：`src/cogcappro/cli/train.py:85`（`ModelCheckpoint(save_last=True)` 无 monitor）vs `:117`（`ckpt_path="best"`）
- LOSO 切分：`src/cogcappro/data/eeg.py:53-65`（`all_subjects = [f'sub-{i:02}' for i in range(1,11)]`）
- 跨被试全局刺激身份：`data/eeg.py:157-164`（`img_path_to_idx` 跨所有被试）
- `top_k` 截断：`utils.py:296-310`（`topk(k=10)` ⊙ `mask_class`）；调用处 `training/module.py:234`（`top_k=10`）
- 融合 detach：`training/module.py:213`
- 空间坍缩：`models/brain_backbone.py:40-48`（`Conv2d(40,40,(channel_num,1))`）
- align 目标 = IP-Adapter 条件：`generate_image/generator.py:270`（`_image_to_embedding`）+ `align/data.py:129-145`
- align 输出 L2 归一化：`align/diffusion_pipe.py`（`SimpleAlignMLP.forward` 末尾 `F.normalize`）
- IP-Adapter 3× 同权重 + per-block scale：`generate_image/generator.py:88-101, 143-155`
- 不确定度仅训练期生效：`data/eeg.py:377-408`
- 陷阱 1（SSIM proxy）：`eeg-brainit/src/eeg_brainit/utils/metrics.py: ssim_simple`
- 陷阱 2（Alex 同名异实）：`eeg-brainit/scripts/erdc_full_metrics.py: _pearson_flat` vs `erdc_twoway_metrics.py: twoway`
- 缺陷 B（邻域库裸 argsort）：`eeg-brainit/scripts/erdc_ras_closed_loop.py:64-71`；调用点 `erdc_official_atm_pipeline.py:329`
- 2WC 口径（200-way/199 干扰项，与 ENIGMA 一致）：`erdc_twoway_metrics.py:42-60`
- 现有锚点：oracle 0.456/0.164、head 0.3435/0.1257、identity 0.2502/0.0478（CLIP cosine / PixCorr）
- 生成栈同源性：`eeg-brainit/scripts/erdc_official_atm_pipeline.py:99-114`
