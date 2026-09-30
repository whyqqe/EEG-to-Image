# NDA Shared+Specific（MindCross / MindBridge）

基于公开代码经验的跨被试改造：

| 来源 | 借鉴点 | 本实现 |
|------|--------|--------|
| [MindCross](https://github.com/XuanhaoLiu/MindCross) | `shared_embedder` + 每被试 `embedder`，`ResFuse(s,r)`，`diff_loss(s*r→0)`，新被试校准只更新 subject 路径 | `ss_modules.SharedSpecificEncoder` + `nda_ss_pretrain.py` Phase B |
| [MindBridge](https://github.com/littlepure2333/MindBridge) | `ModuleDict` 被试 Adapter；reset-tuning | `SubjectAdapter` 零初始化残差；校准冻结 shared |
| ShaSpec / 先前讨论 | 略放大共享骨干 | `EEGProjectWide` 多一块 residual |

流水线：`run_nda_ss_sub08.sh`  
输出：`outputs/nda_ss/sub-08/`  
默认训练被试：`1,2,4,5,6,7,8,9,10`（跳过易负迁移的 sub-03）
