# EEG 视觉检索项目 —— 交接文档（给下一个 Agent）

> 写于 2026-09-16。
> 本项目的唯一目标是：**在 THINGS-EEG2 零样本图像检索上冲击 SOTA**。
> 请先通读 **第 2 节（硬性约定）** 和 **第 6 节（实验规范）**——这两节能避免你造成不可逆的损失和无效算力消耗。

---

## 1. 目标与定位

### 1.1 目标

把 EEG 信号解码成视觉表征，在 **THINGS-EEG2 官方 200-way 零样本检索**协议下做到 SOTA。

这是一个**独立的新项目**，不是已有项目的延续。它可以是干净的重新实现，也可以借鉴/复用工作区里的既有产物（见第 5 节）。**最终要能产出一篇顶会级论文。**

### 1.2 为什么"现在做检索"是有机会的

过去这个工作区把主要精力放在**生成/重建**线上。但从公开文献看，检索线的技术路线在 2026 年发生了**一次范式转移**，而且转移的落点是"**监督目标怎么构造**"，而不是"编码器怎么设计"：

- UBP（CVPR 2025）在 THINGS-EEG2 被试内 Top-1 约 **50.9%**
- Shallow Alignment 把对齐目标从**最终层**换成**中间层**，一步到 **82.6%**
- SAMGA 在此基础上引入**被试感知的多粒度路由**，到 **91.3%**

也就是说，**同一件事上还有 ~40 个百分点的空间已经被证明存在**，且机制是清楚的、可复现的。这正是值得下注的地方。

### 1.3 起点参考：同协议横向表（这张表是你的基准尺）

下表摘自 **SAMGA 论文 Table 2**（THINGS-EEG 被试内，200-way Top-1，10 被试平均）。**这一列数字的价值在于它们是同一张表、同一协议下的可比值**：

| 方法 | 平均 Top-1 | 平均 Top-5 | sub-08 Top-1 |
|---|---|---|---|
| NICE (ICLR'24) | 16.1 | 43.6 | 22.9 |
| ATM (NeurIPS'24) | 27.1 | 58.1 | 38.8 |
| UBP (CVPR'25) | 50.9 | 79.7 | 58.6 |
| NeuroBridge (AAAI'26) | 63.2 | 89.9 | 71.2 |
| Shallow Alignment | 82.6 | 98.3 | 86.9 |
| **SAMGA** | **91.3** | **98.8** | **94.8** |

被试间（leave-one-subject-out）:

| 方法 | 平均 Top-1 | 平均 Top-5 |
|---|---|---|
| NICE | 6.2 | 21.4 |
| ATM | 11.9 | 33.8 |
| UBP | 12.4 | 33.4 |
| NeuroBridge | 19.0 | 45.9 |
| Shallow Alignment | 22.4 | 50.8 |
| **SAMGA** | **34.4** | **64.8** |

**EEGiT (CVPR 2026)** 不在上表内，它自己的报告是：被试内 **70.4 / 95.1**，被试间 **24.0 / 55.6**（10 被试平均）。

⚠️ **重要警告**：这些数字**不能跨论文直接相减**，原因见第 3.3 节。上表内部可比，因为出自同一篇论文的同一张表。

---

## 2. 硬性约定 —— 必须遵守

这一节是**红线**，违反会造成不可逆损失或影响他人。

### 2.1 绝对不要碰的目录

| 路径 | 说明 |
|---|---|
| `/home/*` | **不要写入任何东西。** home 配额极小，历史上出现过 `No space left on device` 导致作业崩溃。所有缓存必须显式重定向（见 2.4） |
| `/project/peilab/why/` 下**除 `eeg-retrieval/` 以外**的所有子目录 | **只读**。包括 `NeuroBridge/`、`eeg-brainit/`、`cache/` 等 |
| 其他用户的家目录、`/cm/shared/apps` 等系统目录 | 只读 |

**允许写入的只有**：
- `/project/peilab/why/eeg-retrieval/`（本项目根目录，你的主战场）
- `/project/peilab/why/cache/`（共享缓存，**只用于 HF/torch 模型缓存**，不要往里塞数据集）
- Slurm 分配给你的计算节点上的 `/tmp`（**是临时的，作业结束即消失，不要存结果**）

**如果你确实需要读别的项目的东西**：读是可以的（见第 5 节的可复用资产），但**不要修改**。需要复制的话，复制到本项目目录内再改。

### 2.2 Python 解释器（唯一正确的那个）

```bash
/project/peilab/why/eeg-brainit/.venv/bin/python
```

- Python 3.11.5，torch 2.13.0+cu130，numpy 1.26.4
- 已装：`transformers 4.46.3`、`timm 1.0.28`、`open_clip_torch 3.3.0`、`diffusers 0.31.0`、`scikit-learn 1.8.0`、`scipy 1.17.1`、`einops`、`pandas`、`matplotlib`、`h5py`、`torchmetrics`、`torch_fidelity`
- **优先复用这个 venv。** 如果你确实需要额外的包，优先 `pip install --target` 到本项目内，或者在本项目下建 `.venv`（用 `--system-site-packages` 继承），**不要动 `eeg-brainit/.venv`**。

### 2.3 数据放在哪里

**所有下载的数据、中间产物、模型输出都放在 `/project/peilab/why/eeg-retrieval/` 下面。**

推荐的目录结构（已建好骨架）：

```
eeg-retrieval/
├── HANDOFF.md      # 本文档
├── data/           # 数据集（软链接或副本，见下）
├── docs/           # 设计与结论文档
├── outputs/        # 实验产物、日志、结果 JSON
├── scripts/        # 实验脚本
└── slurm/          # sbatch 提交脚本
```

**已有数据不要重复下载**，用软链接指过去（见第 4 节）。

### 2.4 缓存必须重定向（否则可能写爆 home）

在任何脚本/sbatch 的最前面加上：

```bash
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
```

⚠️ 注意 `XDG_CACHE_HOME` 这一项：有些既有脚本会进一步 `export HOME=$XDG_CACHE_HOME`，那是一个**兼容性技巧**（让不认这些变量的库也写到磁盘上）。**你新写的代码不要依赖这个技巧**，而是显式设好上面的变量。

### 2.5 计算资源：登录节点没有 GPU

**在登录节点直接跑训练一定会失败**（`torch.cuda.is_available()` 为 `False`）。所有需要 GPU 的工作必须通过 Slurm 提交。

```bash
# 提交模板（本项目 slurm/ 下自己建）
#!/bin/bash
#SBATCH --job-name=xxxx
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/%j.err
```

要点：
- `--partition=normal`、`--account=peilab` 是必需的
- `--exclude=dgx-09,dgx-11,dgx-17,dgx-30`（这几个节点历史上不稳定）
- **在脚本里加一个 CUDA 可用性硬门**，不可用就立即非零退出，避免静默地在 CPU 上跑几小时：

```bash
"${PYTHON}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo '[FATAL] no CUDA'; exit 1; }
```

- 常用命令：`sinfo`（看分区）、`squeue -u $USER`（看自己的队列）、`scancel <jobid>`
- 分区：`normal`（23 节点）、`preempt`（21 节点，**会被抢占，长作业别放这**）、`cpu`

### 2.6 不要在登录节点做重活

`numpy` 扫描几千个 `.npy`、跑数据预处理这类操作请在 Slurm 上做，或者至少限制成单文件、小范围。登录节点是共享资源。

---

## 3. 数据集与评测协议

### 3.1 THINGS-EEG2

| 项 | 值 |
|---|---|
| 被试数 | 10 |
| 训练集 | 1654 概念 × 10 图 × 4 重复 → **重复平均后 16540 对** |
| 测试集 | 200 概念 × 1 图 × 80 重复 → **重复平均后 200 行** |
| 范式 | RSVP，100 ms 呈现 + 100 ms 空白 |
| 电极 | 63 通道（10-20 系统） |
| 采样率 | 1000 Hz 原始 → 常用 250 Hz |
| 滤波 | 0.1–100 Hz 带通 |
| 时间窗 | 刺激后 **0–1000 ms** |
| 基线校正 | 刺激前 **200 ms** |
| 归一化 | **MVNN**（多变量噪声归一化） |
| 检索协议 | **官方 200-way**，测试集全部同时充当 query 和 gallery |
| chance | **0.5%**（Top-1，200 选 1） |

**最关键的一点**：测试集的 80 次重复是**先平均再评测**的。所以测试集是 200 行，不是 16000 行。这是所有主流论文的共识口径（NICE / UBP / Shallow Alignment / SAMGA 都如此）。**如果你算出 16000 行，说明你的口径错了。**

### 3.2 三个必须分清的口径差异

很多论文的数字不能直接比，因为：

1. **电极数**：SAMGA 被试内**只用 17 个枕区+顶区电极**，被试间才用全部 63 通道。EEGiT 两个设置都用 63 通道。UBP 用 63。
   → **用 63 通道是更保守、更安全的选择**。如果你也做 17 通道的消融，一定要在论文里说清楚。

2. **重复平均次数**：主流是"训练 4 次平均、测试 80 次平均"。有些工作（few-repetition 方向）会改成更少的平均次数，此时绝对数字会低很多，**不能跟 80 次平均的数字比**。

3. **是否 transductive**：`CSLS`、`Sinkhorn`、hubness 校正这些技术**用到了"200 个候选互斥"这个测试集结构**，属于 transductive。而**纯 EEG→图像单向检索是 inductive**。
   → **汇报时必须区分**。inductive 是主指标，transductive 只能放在附录或明确标注。

### 3.3 关于"对齐目标维度"的坑

不同视觉骨干的输出维度不同：CLIP RN50 = 1024，ViT-H-14 = 1024，InternViT-6B = 3200+（需投影）。**目标空间的维度和语义轴必须全程一致**。历史上这里出过一次错：编码器用 1280-d 目标训练，但下游检索用 1024-d ViT-H-14 gallery，两个轴不一致，导致结果无法解释。

**建议**：从一开始就固定一个骨干 + 一个层，把这个决定写进文档，之后所有实验都用它。改骨干 = 换实验，不是一个超参。

---

## 4. 已有数据（**不要重复下载**）

### 4.1 预处理好的 EEG（最省事的入口）

```
/project/peilab/why/NeuroBridge/data/things_eeg/preprocessed_eeg/
├── info.json          # 权威元信息：ch_names(63)、times(300)、sfreq=250、normalization=mvnn
├── sub-01/ ... sub-10/
│   ├── train.npy      # (16540, 63, 300) fp32，约 4.2 GB
│   └── test.npy       # (200, 63, 300)，约 1.0 GB
```

这是**已经做过 MVNN、降采样到 250 Hz、截取 0–1000 ms** 的版本，可以直接用于检索实验。**强烈建议从这里起步**，不要自己从原始数据重做预处理。

**✅ 软链接已经建好**，你可以直接用：

```
eeg-retrieval/data/
├── preprocessed_eeg -> .../NeuroBridge/data/things_eeg/preprocessed_eeg
├── image_feature    -> .../NeuroBridge/data/things_eeg/image_feature
├── images_set       -> /project/peilab/why/data/images_set
└── hf_cache         -> /project/peilab/why/cache/eeg-brainit/hf
```

自检（应输出 `channels 63 times 300 sfreq 250.0 norm mvnn`）：

```bash
python3 -c "import json;d=json.load(open('data/preprocessed_eeg/info.json'));\
print('channels',len(d['ch_names']),'times',len(d['times']),'sfreq',d['sfreq'],'norm',d['normalization'])"
```

⚠️ 这些是**软链接**，指向其他项目的目录——**只读**。不要通过它们写回任何东西。

### 4.2 图像特征

```
/project/peilab/why/NeuroBridge/data/things_eeg/image_feature/
├── RN50/image_train.npy, image_test.npy
├── RN50/{GaussianBlur,GaussianNoise,LowResolution,Mosaic}/...
└── ViT-H-14/...
```

注意目录里那些 `GaussianBlur` / `LowResolution` / `Mosaic` / `GaussianNoise` 是**多层级扰动目标**（UBP 那条思路的产物），在文献里这个方向值 +21 个点。

### 4.3 原始图像

```
/project/peilab/why/data/images_set/
├── training_images/    # 1654 个概念目录 × 10 张 jpg（概念按字母序嵌套编号）
├── test_images/        # 200 个概念目录
├── image_metadata.npy
└── DESCRIPTION.md      # 读它！里面写了 EEG event 号 ↔ 图像文件的对应规则
```

**if you need to extract new features**，用这批图自己跑骨干提取，写到你自己的 `data/` 下。

### 4.4 模型缓存（已下载，会被复用）

`/project/peilab/why/cache/eeg-brainit/hf/hub/` 下已有：
`models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K`、`models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K`、`models--timm--vit_large_patch14_*`、`models--stabilityai--sdxl-*`、`models--h94--IP-Adapter` 等。

**先检查这里有没有你要的模型，有就不要重新下载。**

注意：**InternViT-6B（SAMGA 用的骨干）本地没有**，如果需要得自己下（约数十 GB，请下到本项目 `data/` 下并注意磁盘）。

---

## 5. 可复用的既有资产（**只读，不要改**）

工作区里有一个前序项目 `/project/peilab/why/NeuroBridge/`（它是 AAAI 2026 论文 NeuroBridge 的官方实现）。它的产物对你有用：

| 资产 | 路径 | 用途 |
|---|---|---|
| 预处理 EEG | `NeuroBridge/data/things_eeg/preprocessed_eeg/` | 直接当输入 |
| 图像特征 | `NeuroBridge/data/things_eeg/image_feature/` | 直接当监督目标 |
| **Leak-free 划分** | `NeuroBridge/outputs/leakfree/split.json` | 见下 |
| 编码器导出特征 | `NeuroBridge/outputs/ocf/intra_z/sub-XX/shared_r_{train,test}.npy` | `(16540,1024)/(200,1024)`，可当强起点 |
| 评测脚本参考 | `NeuroBridge/scripts/nda/eval_standard7.py` | 生成侧七指标，检索用不到但可参考写法 |
| 前序交接文档 | `NeuroBridge/docs/HANDOFF_NEXT_AGENT.md` | **值得一读**，里面有大量踩坑记录 |

**Leak-free 划分**（`split.json`，`seed=20260910`，concept-disjoint）：
- `fit_rows` 14890 行（1489 概念）→ 训练
- `val_a_rows` 830 行（83 概念）→ 选择
- `val_b_rows` 820 行（82 概念）→ 二次确认
- 测试 200 概念**从不参与任何选择**

**直接复用这个划分**，这样你的结果和前序项目可比，也省得自己造轮子。

---

## 6. 实验规范 —— 这套纪律比任何技巧都重要

前序项目用真实算力换来的教训，**不要为了快而绕过**：

### 6.1 数据纪律

1. **Leak-free 划分**：`fit`（梯度）/ `val_a` / `val_b`（选择）/ **test 只在最后评测一次**。
2. **绝不按 test 指标选 checkpoint**。曾经因为"按 test 选"污染过实验产物。**在代码里写硬断言**：
   ```python
   assert selected_on == "val", "checkpoint 只能按 val 选"
   ```
3. **预登记判据**：在实验脚本**头部**用注释写明"什么算成功、什么算失败"的**具体阈值**，写在你看到数字**之前**。这能防止事后为了自圆其说而改写标准。

### 6.2 统计纪律

4. **每个断言都要有负控制**。真实教训：一次"负控制"因为切片到了空张量而**通过了**，实际什么都没测。必须验证负控制确实产生了变化。
5. **配对统计，而不是比较最大值**。"我这条路线最好"是 13 个噪声里取最大值的比较，天然带 winner's curse。要比就**逐项配对 + Wilcoxon 符号秩检验**。10 被试的配对检验下，Wilcoxon exact 的 p 值最小也只能到 0.00195——**报告 p 值时要如实说明这个下限**。
6. **单被试的结果不能当结论**。sub-08 上的 +0.02 很可能只是噪声。要么跨 10 被试验证，要么明确标注"单被试、未验证"。

### 6.3 工程纪律

7. **静默失败是头号风险**。EEG 数据是稠密数值矩阵，**喂错电极顺序、错维度、错时间窗都不会报错**，只会输出看起来合理的垃圾。任何跨 montage / 跨维度 / 跨骨干的操作都要**显式守卫 + 断言**。
8. **作业必须可续跑**：每个阶段在有产物时 `[SKIP]`。被墙钟杀掉只损失当前阶段，不要从零重跑。
9. **日志要落到文件**，不要只看 stdout。Slurm 作业被抢占后日志是你唯一的线索。
10. **每个提交的作业都要打印它的关键配置**（ckpt 路径、电极数、维度、seed、数据 hash），否则事后无法确认你跑的是不是你以为的东西。

---

## 7. 已证伪清单 —— 不要重做

以下结论来自前序项目的真实实验（多为 10 被试、带统计检验），**请勿重试**：

| 假设 | 结论 | 证据 |
|---|---|---|
| 端到端联合微调编码器更好 | ❌ **明确更差** | 10 被试中 `joint` **0/10 胜**，`frozen` 10/10 胜，Wilcoxon exact p=0.00195，mean Δ = −0.0375 |
| 电极数（17 vs 63）是关键变量 | ❌ **无显著效果** | 逐路配对 +0.0019 (p=0.587) / −0.0042 (p=0.619)。23 点差距不在电极数上 |
| 用 `val_top1` 选 checkpoint 可靠 | ❌ Spearman 仅 +0.379，动态范围极小 | 等于接近随机选 epoch |
| 用 `margin` 当选择器 | ❌ **与泛化反相关** | 最大化它会选到 epoch 0（几乎没训练的头） |
| 用 2-way identification 做选择 | ❌ 严重饱和 | 均值 0.9185，86.4% 的 epoch 落在最大值 1% 内，argmax 不可靠 |
| 线性模型（Ridge）够用 | ❌ 差 **3.76×** | EEG→视觉空间是强非线性的 |
| 评测协议不一致是差距的原因 | ❌ **已逐项核验一致** | MVNN ✓ 250Hz ✓ 16540 ✓ 200 行 ✓ 200-way ✓ chance=0.5% ✓ |
| **更多训练步数会更好** | ❌ **本轮证伪** | NW-v8：val 在 **epoch 19** 达峰（34.40），其后 80 个 epoch 单调下滑到 21.47。预算给到 100 epoch 时有效部分只有前 19 个 |

**另外**：venv 里是 **scipy 1.17.1**，`scipy.stats.wilcoxon` 的签名已变化——旧的 `mode=` 参数已被移除，现在用 `method=`（可选 `'auto'/'exact'/'asymptotic'`）：

```python
wilcoxon(x, y, zero_method='wilcox', correction=False,
         alternative='two-sided', method='auto')
```

若你要拿"exact p 值下限 = 0.00195"这个事实做论断（n=10 配对），用 `method='exact'` 显式指定。

---

## 8. 三条当前最可能的攻击路线

按我判断的优先级排列，**并按 NW-v8 的结果更新了状态**。

> **NW-v8 之后的现状（先读这一段）**
> sub-08 test Top-1 = **46.50**（ridge 下限 25.00，SAMGA sub-08 = 94.8，达成 49%）。
> 训练集 fit Top-1 = **91.47**，val 34.40 → **间隙 57.1 点**，
> 且 val 在 epoch 19 达峰后**长期下滑**。
> **所以当前瓶颈是泛化，不是对齐、不是容量、不是步数。**
> 这三条路线（尤其 P0/P2）依然有效，但**任何一条都必须先做归因对照**，
> 否则会重复「一次开五个开关、+8 点无法归因」的错误（见设计文档 §0.9）。

### P0：把粒度机制做扎实（**最高性价比**）

**做什么**：不要用最终层特征当监督目标，要从**视觉编码器的多个中间层**构造目标，并让"怎么加权"成为可学习的。

**为什么**：这是文献里**最大的单一效应**。UBP 用最终层约 50.9% → Shallow Alignment 换中间层 82.6%，**一步 +31.7 点**。SAMGA 在这之上再做路由到 91.3%。

**状态（本轮更新）**：
- ✅ **步骤 1（单层扫描）已完成**。32 层 + `_pooled` 的闭式 ridge 探针已跑通，
  曲线是**清晰的倒 U**：`block26` 峰值 17.33 val / 27.50 test，
  vs 最终层 `_pooled` 的 15.40 / 25.00（+2.50 test，SE ≈2.8 点，**在噪声量级内**）。
  这条曲线**不是** UBP 说的 +31.7 点那么大的效应——本项目的测量是 +2.5 点。
- ⚠️ **步骤 2（多层融合）已实现但未测出增益**：
  均匀融合 `eeg_fuse_uniform` 38.50 略高于单层 `align_block26` 34.00（+4.50），
  但 `target_fuse_routed` 用 `_pooled` 目标只拿到 37.00。
  **这些臂都在坏接口下跑的**，不能作为结论。
- ⏸️ **步骤 3（路由）未做**。

**具体起点**（顺序已按新证据重排）：
1. ~~先做单层扫描~~ **已完成**，见上。
2. **先补归因对照**：`--tokenizer grid` vs `eegit`，其余全同。
   这一步便宜（一个臂），但它是把「+8 点来自哪儿」从相关变成因果的唯一办法。
3. **再做正则化消融**（新 P0）：坐实 57 点间隙。`--freeze-blocks`、
   dropout、更强的 EEG 增强。同时把 epoch 从 100 降到 ~30 并恢复 early stop。
4. 然后才是多层融合 / 路由，且**只加一个开关**。

**注意**：
- 用**线性投影**把各层投到公共空间——SAMGA 消融显示 **MLP 反而更差**。
- 路由权重的初始化点很关键：SAMGA 用 `[-2,-1,0,1,2]` 的 logits，**峰值放在中间层**（它的第 28 层）。不要从均匀开始。
- **如果你要主张新颖性，P0 本身不够**——这条路线 Shallow Alignment 和 SAMGA 都已经占了。你需要往下走到 P1 或 P2 才有新东西。但 **P0 必须做**，因为它是你的基线，不做你的数字就没有说服力。

### P1：让"训练时用被试信息、推理时不用"这个机制更进一步

**做什么**：SAMGA 的做法是把被试信息用来**校准视觉目标**（学生成监督信号的权重）。它自己在 limitation 里承认了两点没做：
- 粒度只建模到**被试级**，不能刻画注意/状态导致的**试次级**变化
- 候选层是**预先给定**的，没有去**发现**哪一层值得用

**为什么有机会**：这两个是它公开承认的缺口，而且是**方法层面的**，不是调参层面的。

**具体起点**：
- 试次级路由：把路由的输入从"被试 ID"换成"当前 EEG 试次的某个统计量"（比如该试次的信噪比 / 置信度），让粒度随试次变化而自适应。**风险**：试次级信号很弱，容易过拟合，必须用 leak-free 划分严格验证。
- 可发现层：让候选集合本身可学（比如用可微的层选择 / 稀疏门控），而不是人工指定 20/24/28/32/36。
- 另有一个未验证的方向：**被试间（LOSO）**。SAMGA 的收益在被试间远小于被试内（34.4 vs 91.3），且它说自己"跨被试增益更大"——这里可能有空间。

### P2：把目标构造推广到"非语义"的监督

**做什么**：现在所有方法都在对齐**语义 CLIP 特征**。但前序项目和我自己的分析都指向同一个结论：**EEG 里有相当一部分信息是低层视觉属性**（纹理、结构、空间布局），而 CLIP 这类语义不变的表示**恰好把这些压掉了**。

**具体起点**：
- 用**多层级 + 多性质**的目标族：语义 CLIP + 深层特征 + 低层特征（边缘/深度/纹理），作为**互补的监督信号**而不是只取语义。
- 前序项目独立发现了同一点：它的"多层级扰动目标"路线银行里，`vith_cat5` 单路 0.4050 vs 普通 `img_clip` 0.3050。**方向被文献独立验证，但执行差很多**（文献 82.8% vs 它 38%）。**这中间的执行差距本身就是一个可以攻的问题。**
- UBP 的"不确定性感知模糊"也是这条路线的变体：主动把图像侧的高频细节糊掉，减少与脑信号的失配。

---

## 9. 建议的第一步

不要一上来就复现 SAMGA。建议顺序：

1. **把地基跑通**（1–2 天）：用第 4 节的预处理 EEG + ViT-H-14 特征，在 sub-08 上实现一个**干净的、可复现的零样本 200-way 检索基线**。目标不是高分，是**确认口径正确**：
   - 测试集形状必须是 `(200, D)`
   - chance 必须是 0.5%
   - 训练集必须是 `(16540, D)`
   - 用 `split.json` 的 `fit` 训练、`val_a` 选、test 只评一次
2. **做单层扫描**（P0 第 1 步）：确认你自己手上的"层深 vs 性能"曲线确实是倒 U 形，且中间层明显好于最终层。**如果你复现不出这个现象，先停下来找原因**——那说明你的实现和文献有系统性差异，在这个基础上做什么都是错的。
3. **再往上叠**：多层融合 → 可学习路由 → （若要走 P1）试次级/可发现层。
4. **每加一个组件，都要有负控制和配对统计**。

---

## 10. 参考论文清单

### 检索主线（必读）

| 论文 | 出处 | 核心贡献 | 代码 |
|---|---|---|---|
| **Shallow Alignment**<br>*Deep Models, Shallow Alignment* | arXiv 2601.21948 | **最重要的一篇**。发现"粒度失配"：深度视觉模型的最终层为语义不变性压掉了局部纹理，而 EEG 保留了低层结构。改用**中间层**对齐，+22%~58%。THINGS-EEG 被试内 82.6% | [yangdu-neuroai/shallow-alignment](https://github.com/yangdu-neuroai/shallow-alignment) 有 |
| **SAMGA**<br>*Subject-Aware Multi-Granularity Alignment* | arXiv 2604.17782 | 在 Shallow Alignment 之上加：**全局粒度先验 + 被试残差路由 + 被试/层 dropout + 由粗到细对齐**。关键是"被试感知训练 + 被试无关推理"。被试内 91.3 / 被试间 34.4 | [LinJiang8/SAMGA](https://github.com/LinJiang8/SAMGA) 有 |
| **EEGiT**<br>*Teaching ViTs to Understand the EEG signal* | CVPR 2026 | 路线不同：把 EEG **重排成图像块**（按脑区插值 + 时间重采样 → 16×16 patch，复制 3 份当 RGB），让 ImageNet-21K 预训练 ViT-B/16 直接当 EEG 编码器。被试内 70.4 / 被试间 24.0 | ❌ **代码未公开** |
| **UBP**<br>*Uncertainty-Aware Blur Prior* | CVPR 2025, arXiv 2503.04207 | 主动**模糊图像侧高频细节**以匹配脑信号的粒度。被试内 50.9 / 被试间 12.4。是"改进目标侧"这条线的开端 | — |
| **NICE**<br>*Decoding Natural Images from EEG* | ICLR 2024 | 奠定 THINGS-EEG 零样本对比对齐范式的基石工作。被试内 16.1 | — |
| **NVOL** | arXiv 2609.02582 | 78.1% raw / 86.4% +CSLS。**已占"多层级目标 + CSLS + 检索生成耦合"**，架构层面不再是空白 | — |

### 参考其他模态 / 其他任务（找灵感）

| 论文 | 说明 |
|---|---|
| **ENIGMA** (arXiv 2602.10361) | 生成/重建线 SOTA，多被试。PixCorr 0.1668 / SSIM 0.4264 |
| **CogCapPro** | 生成线。它的**编码器侧**贡献（不确定性掩码、融合编码器、非对称对齐、语义标签对比损失）值得借鉴思路 |
| **NeuroBridge** (AAAI 2026, arXiv 2511.06836) | 前序项目。认知先验增强 + 双向语义对齐 |
| **SCORE** | LOSO 设定的 SOTA 参考值 53.23%（跨被试很难） |
| **SATTC** (CVPR 2026) | 跨被试 hubness 校准 |
| **Beyond Trial Averaging** (arXiv 2608.19128) | 研究**少重复次数**下检索为何退化。如果你关心实际可用性，这是新方向 |

### 评测框架（论文写作时要对齐）

近两年出现了新的细粒度评测框架，**只报一个 Top-1 可能不够**：
- **BASIC**（Brain-Aligned Structural, Inferential, and Contextual similarity）
- **EEG-EditBench**
- **OmniEEG-Bench**

---

## 11. 一句话总结

**主战场是"监督目标该怎么构造"，不是"编码器该怎么设计"。** 文献已经证明了最大的一块收益（最终层 → 中间层，+31.7 点）就在这条轴上，而且公开承认的两个缺口（试次级粒度、层集合可发现）可以继续攻。

**你的第一步不是复现别人的最高分，而是先确认自己的口径正确、能复现出"中间层优于最终层"这个基本现象。** 复现不出这个现象，说明基础实现有问题，之后所有工作都是在一个错的底座上做优化。

**最危险的事**：直接冲一个复杂架构，但没有干净的 leak-free 划分、没有负控制、没有配对统计——那样得到的"提升"无法区分于噪声，写进论文会被审稿人一击即碎。
