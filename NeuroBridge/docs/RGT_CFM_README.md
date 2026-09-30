# RGT-CFM：检索→生成运输（跨被试）

**主张**：检索最优空间 \(z_{ret}\) 与生成流形 \(z_{gen}\) 不相容；用**被试条件 CFM** 做可学习运输，并用邻域保持损失保留检索身份。

```
多被试 EEG → SharedSpecific → z_ret (512 SSP)
                              ↓  RGT-CFM (+ subject FiLM)
                           z_gen (ViT-H 1024)
                              ↓  mem⊕CFM → SDXL/IP-Adapter
```

| 组件 | 作用 |
|------|------|
| `rgt_build_bank.py` | 9 被试 z_ret 大数据银行 |
| `rgt_cfm_train.py` | CFM + NCE/cos + neighborhood preserve；线性映射作鸿沟诊断 |
| `run_rgt_cfm_sub08.sh` | 编码→训练→memory→生成→指标 |

输出：`outputs/rgt_cfm/sub-08/`  
对照：`rgt_linear_s40` vs `rgt_cfm_*`（看鸿沟是否被桥接）
