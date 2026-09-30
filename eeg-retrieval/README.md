# eeg-retrieval

THINGS-EEG2 零样本图像检索，目标 SOTA。

## 先读这个

**[HANDOFF.md](HANDOFF.md)** —— 项目交接文档。包含：
- 实验目标与同协议 SOTA 基准表
- **硬性约定**（解释器路径、数据位置、禁止写入的目录、缓存重定向、Slurm 提交方式）
- 数据集与评测协议（含三个必须分清的口径差异）
- 可复用的既有数据与资产
- 实验规范（leak-free 划分、负控制、配对统计）
- 已证伪清单（不要重做）
- 三条建议的攻击路线

## 最小环境速查

```bash
# 解释器（唯一正确的那个）
PYTHON=/project/peilab/why/eeg-brainit/.venv/bin/python

# 缓存重定向（必须，否则可能写爆 home）
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
```

登录节点**没有 GPU**，训练必须走 `sbatch`（`--partition=normal --account=peilab`）。

## 目录约定

```
data/     数据集（用软链接指向已有数据，不要重复下载）
docs/     设计与结论文档
outputs/  实验产物、日志、结果 JSON
scripts/  实验脚本
slurm/    sbatch 提交脚本
```

**只在本目录内写入。** 不要写 `/home`，不要改工作区里其他项目的任何文件。
