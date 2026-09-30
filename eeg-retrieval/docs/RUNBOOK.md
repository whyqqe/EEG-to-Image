# NW-Retrieval: 实验运行手册

> 本文件说明**怎么跑**和**怎么读**。设计与理论依据见
> [`docs/DESIGN_semantic_structure_branches.md`](DESIGN_semantic_structure_branches.md)。
> 项目通用约定（解释器、缓存、数据位置）见 [`HANDOFF.md`](HANDOFF.md)。

---

## 1. 一条流水线，一个 job

整个 sub-08 研究（22 个实验臂）在**单个 job 内顺序执行**。原因有两个：

1. 集群 QOS 限制 `MaxSubmitJobs=10`，多臂并行会撞上限
2. 多臂研究不需要 22 份独立配额

```bash
cd /project/peilab/why/eeg-retrieval
sbatch slurm/nwret_sub08.sbatch
```

### 为什么要设计成「可续跑」

`slurm/nwret_sub08.sbatch` 带 `--requeue`。流水线对**每个臂**做幂等检查：

- 结果 JSON 存在**且含完整的 `test` 块** → 跳过
- 否则重跑该臂

因此被抢占或触墙只损失**正在跑的那一个臂**，不是整个流水线。这个守卫做过四种情形
的测试（完整 / 缺 test 块 / JSON 损坏 / 文件不存在），行为均正确。

### 时间守卫

`PIPELINE_HOURS`（默认 22）是预算。每个臂有一个时间下限，剩余时间不足时
**不再启动新臂**，但仍会执行汇总。所以核心结果有保证，尾部优雅降级而不是直接失败。

`--requeue` + 时间守卫 + 幂等跳过三者配合，才使「一个 job 跑完整研究」可靠。

---

## 2. 实验臂与优先级

臂的**顺序即优先级**（信息价值降序）。这不是随意排的：

| Tier | 臂 | 内容 | 为什么在这个位置 |
|---|---|---|---|
| **T1** | `B02`–`B12` | ViT-B/16 层扫描（6 个单层） | **最大已知杠杆**（文献 +31.7），且用最便宜的骨干（12 块 / 86M）才扫得起 |
| **T2** | `A1`–`A3` | tokenization 变体 | EEGiT 消融把「EEG→patch 接口」定价为 **+16.4**，大于预训练权重项（+6.8） |
| **T3** | `C06`–`C24` | DINOv2-L 结构侧（24 块 / 1024-d） | iBOT 掩码目标强制保留结构（稠密 +3%）、KoLeo 为检索优化（+8%） |
| **T4** | `D08`–`D32` | CLIP ViT-H-14 语义侧（32 块） | `visual.proj` 正好落到缓存特征所在空间，无需跨空间翻译 |
| **T5** | `E1`–`E2` | 多层融合：均匀先验 vs 可学习路由 | 把「多层本身」和「自适应组合」两个增量分开 |

`B12` 同时充当 tokenization 的基线臂（原 `A0` 与它完全等价，故省略）。
完整臂列表见 `scripts/run_arm.sh` 的 `ARMS` 数组（也是 array 索引映射）。

`A4`（随机权重负对照）留在 dispatcher 里但不在默认流水线中——它回答的是
「预训练权重值多少」，可在核心结果出来后单独补跑。

---

## 3. 评测协议（必须与 SOTA 口径一致）

### 3.1 口径（复刻 SAMGA）

| 项 | 值 |
|---|---|
| 检索 | **200-way** 对角检索 |
| 重复平均 | **train 4 次、test 80 次** |
| 指标实现 | `third_party/SAMGA/module/util.py::retrieve_all`，**逐字节复制** |
| 参照线 | SAMGA intra 17ch = **Top-1 91.3 / Top-5 98.8** |

指标实现**已实测与 SAMGA 参考实现输出完全一致**（同一输入下 `(top5, top1, n)` 三元组相等）。
流水线启动时会自动跑这个一致性检查，不一致就直接退出。

> ⚠️ **不要「改进」这个指标实现**。不同的并列名次处理或归一化约定会让与已发表
> SOTA 的比较失去意义。

### 3.2 与 SAMGA 的一处**故意**差异：checkpoint 选择

SAMGA 用**测试集 Top-1** 选 checkpoint。**这是泄漏**，我们不这么做。

我们把 **1654 个训练概念**里留出 **150 个**当验证集（概念级划分，故验证概念的
任何图像都不参与训练），只用它选 checkpoint；测试集**只评一次**，在最后。

后果：我们的数字**不可**与 SAMGA 逐位对比（选择集不同），但**更可信**。
排行榜里同时给出 `val` 和 `test` 两列，便于看出选择是否过拟合。

---

## 4. 代码结构

```
scripts/nwret/
  config.py      路径、缓存重定向、数据几何、通道子集、协议常量
  tokenizer.py   EEG → token：montage 插值 + 时间窗切片 + 投影
  encoders.py    两类编码器 + LayerFusion
  model.py       RetrievalModel（EEG 分支 + 图像投影）
  data.py        数据加载、概念级划分、两个 Dataset
  losses.py      InfoNCE、MMD
  metrics.py     检索指标（与 SAMGA 一致）+ mean rank
  train.py       单臂训练与评测（CUDA 硬门、test 只评一次）
  summarize.py   汇总排行榜
scripts/run_arm.sh      臂分发器（ARM=<name>）
scripts/run_pipeline.sh 单 job 完整流水线（幂等 + 时间守卫）
slurm/nwret_sub08.sbatch
```

### 两类编码器（对应两条支路）

| 工厂前缀 | 类 | 用途 |
|---|---|---|
| `openclip:ViT-H-14` | `OpenCLIPViTEEGEncoder` | **语义侧**。open_clip 视觉塔，`proj`(1280→1024) 落到目标空间 |
| `timm:dinov2_l` | `ViTEEGEncoder` | **结构侧**。自监督，保留结构 |
| `timm:vit_b16_in21k` | `ViTEEGEncoder` | 参考点（EEGiT 的确切骨干） |

两者都做同样一件事：**丢弃原 `patch_embed`（14×14×3 卷积核），换成 EEG tokenizer**，
并把 `pos_embed` 从源网格**双三次重采样**到目标网格 `(7, 28)`。

已处理的坑（都实测过）：

- **`pos_embed` 的两种约定**：非 register 的 DINOv2 / CLIP 是 `(1, 1+src_h*src_w, D)`
  （含 cls），而 **register 版 DINOv2 是 `(1, src_h*src_w, D)`**（cls 与 register 都没有
  位置嵌入，故 `prefix_pos` 补零）。用源网格而非目标网格去比对长度，否则误报。
- **open_clip 的 block 是 sequence-first**（`(L,B,D)`），timm 是 batch-first。混用会静默算错。
- **`visual.proj` 已在编码器内应用**，故 `feat_dim` 是 1024 而非 1280；`LayerFusion`
  必须按 `feat_dim` 建层，否则维度不匹配。

### 每层输出都过了 `ln_post`

`OpenCLIPViTEEGEncoder._pool` 对中间层也应用 `ln_post` 与 `proj`。这是**有意**的：
让所有层的输出处在同一规整空间，否则前几层的未归一化残差会主导融合权重。

---

## 5. 输出与读法

```
outputs/sub08/
  <arm>_result.json      单臂完整记录（逐轮历史、协议、耗时）
  leaderboard.md         排行榜（含 SOTA 百分比）
  leaderboard.json       机器可读版
outputs/logs/<arm>.log   单臂训练日志
```

**读结果的两个要点**：

1. **Top-1 远离饱和时，`mean_rank` 信息量更大**。200-way 随机猜的均值排名是 100.5；
   靠 Top-1 只能看出「相对随机好多少」，均值排名能看出连续改善。
2. **不要只看测试集**。若 `val_top1` 涨而 `test_top1` 不涨，说明在选择集上过拟合
   （或概念级划分的方差太大），这个信号比绝对数字更有价值。

---

## 6. 已实测的坑（不要再踩）

| 坑 | 症状 | 状态 |
|---|---|---|
| `evaluate` 忘了过 `encode_image` | `Incompatible dimension 512 vs 1024` | 已修，并加了显式宽度断言 |
| `import json` 放在函数内遮蔽模块级导入 | `UnboundLocalError: json` | 已修 |
| `pos_embed` 长度按目标网格比对 | register DINOv2 误报维度错 | 已修，改按源网格 |
| `token_grid` 用 tuple 赋值给 float tensor | `TypeError: can't assign a tuple` | 已修，用 `repeat_interleave` |
| `DINOv2-L` 305M / `ViT-H-14` 633M 显存 | 需降 batch | 臂里已设 batch 256 / 128 |
