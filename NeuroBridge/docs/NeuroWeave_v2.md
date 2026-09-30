# NeuroWeave v2 — 修正后的主线（给实验与下一 Agent）

> 对应代码：`scripts/nda/neuroweave_*.py`，sub-08 作业 `nweave_s08`。
> 原方案见 `NeuroWeave.md`。本文只写**保留主线后的修正**与**预登记判据**。

## 保留的主线

1. **Hierarchical EEG representation**（§5.2）—— 但必须有**层级监督**，不能只是多尺度编码器。
2. **Retrieval-augmented**（§5.6）—— 拆成双重角色：生成端外观先验 + 后验端概念画廊融合（CF-MSF 已验证）。
3. **Neural-consistent generation**（§5.7–5.9）—— 分层条件注入 + **判别式** cycle，而非原始余弦。
4. **不确定性 / 多样本后验**（§5.8）—— 保留，但不领跑。

## 本轮在测的选择（不要先验拍板）

| 轴 | 臂 | 问题 |
|---|---|---|
| H 层级目标 | `frozen` / `lora` / `direct` / `multi_head` | 层级监督能否抬高底座？ |
| T 时间/因果 | `lora`(drop) / `anytime_train` / `causal_stage` + anytime 评测曲线 | 标题里要不要留 Causal？ |
| C cycle | `cycle_raw` vs `cycle_disc`（在已有生成图上） | 用哪个 cycle 写论文？ |

**明确不做**：端到端 `joint` 全参微调（已 0/10 败，p=0.002）。

## 预登记判据

- **H**：`best(lora,direct,multi_head)` 的 fuse CSLS ≥ `frozen` + 0.02，否则回退。
- **T**：若 anytime 曲线 `full−early < 0.05` → 标题去掉 Causal；若 `causal_stage` 赢且 gap≥0.05 → 保留为 progressive decoding。
- **C**：若 `cycle_disc` 与语义指标同向、与 SSIM 反向，而 `cycle_raw` 相反 → 采用 disc。

## 输出

`outputs/neuroweave/sub-08/`

- `s1_report.json` / `anytime_report.json` / `cycle/cycle_report.json` / `summary.json`
- 每臂 `enc/sub-08/shared_r_*.npy` + `probe/route_probe.json`
