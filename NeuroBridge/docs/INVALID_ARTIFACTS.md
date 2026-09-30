# INVALID 产物禁入清单

> 原因：这些运行使用了**含 GT 类名**的提示词文件，属于语义信息泄露。
> 实测：含 GT 类名条件可达 Incep 0.912 / CLIP 0.945；干净条件仅 0.69–0.74 / 0.79–0.86。
> **单一泄露变量价值 ≈ +0.17 Incep，超过历史上任何"架构创新"。**

## 泄露提示词文件（禁用于任何主表）

| 文件 | 自身类名命中 |
|---|---|
| `prompts_full_hcma_test.json` | 200/200 |
| `prompts_dual_test.json` | 200/200 |
| `prompts_fine_test.json` | 200/200 |
| `prompts_coarse_test.json` | 200/200 |
| `prompts_det_test.json` | 200/200 |
| `prompts_subj_test.json` | 200/200 |
| `prompts_subj_det_test.json` | 200/200 |
| `prompts_oracle.json` | 200/200 |
| `prompts_true.json` | 200/200 |
| `prompts_true_test.json` | 200/200 |

## 受污染的 run 脚本（生成的产物一律作废）

共 **41** 个：

- `scripts/nda/run_ab_struct_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_ack_reprompt_sub08.sh`  ← `prompts_oracle`
- `scripts/nda/run_ack_sub08.sh`  ← `prompts_oracle`
- `scripts/nda/run_alex2_sota_official_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_atm_aligned_decode_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_atm_struct_sweep_v1_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_clean_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_dual_ctrl_struct_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_g2_final.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_g2_multi.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_g2_pipeline.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_g2f.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_gcc_p0p1_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_gem_intra.sh`  ← `prompts_true`
- `scripts/nda/run_hcma_10subj.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_hcma_lite_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_hcma_loso.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_hcma_loso_fid129.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_hcma_s_full10.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_hcma_s_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_inter_ll_full10.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_intra_hcma_s_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_lg_gate_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_loso_inter_hcma_s_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_mg_flow_a40_all.sh`  ← `prompts_dual_test`
- `scripts/nda/run_mg_flow_sub08.sh`  ← `prompts_dual_test`
- `scripts/nda/run_oracle_chase_sub08.sh`  ← `prompts_oracle`
- `scripts/nda/run_oracle_chase_v2_sub08.sh`  ← `prompts_oracle`
- `scripts/nda/run_overnight_struct_sweep_sub08.sh`  ← `prompts_dual_test`
- `scripts/nda/run_overnight_struct_v2_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_percept_flow_sub08.sh`  ← `prompts_dual_test`
- `scripts/nda/run_rcfm_ll_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_rgt_v3_dualcond_sub08.sh`  ← `prompts_oracle`
- `scripts/nda/run_rgt_v4_txtfix_sub08.sh`  ← `prompts_oracle`
- `scripts/nda/run_sdedit_ll_full10.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_shb_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_struct_inject_v1_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_struct_lora_v1_sub08.sh`  ← `prompts_full_hcma_test`
- `scripts/nda/run_tcda_salfuse_v2.sh`  ← `prompts_dual_test`
- `scripts/nda/run_tcda_sub08.sh`  ← `prompts_dual_test`
- `scripts/nda/run_vae_head_fix_sub08.sh`  ← `prompts_full_hcma_test`

## 受污染的七指标文件

`outputs/**/*_seven.json` 共 **49** 个。
报告任何基于这些文件的 Incep/CLIP/Alex 数值前，必须先用 `prompts_deploy.json` 重跑。

## 合法（generic 无类名）跑法

| 脚本 | 提示词 |
|---|---|
| `run_balanced_decoder_sub08.sh` | `prompts_cpa.json` |
| `run_cn_ip_decode_sub08.sh` | `prompts_pred.json` |
| `run_coca_phase_ab_sub08.sh` | `prompts_pred.json` |
| `run_layered_sub08.sh` | `prompts_deploy.json` |
| `run_lowlevel_decoder_sub08.sh` | `prompts_cpa.json` |
| `run_mac_r_sub08.sh` | `prompts_pred.json` |
| `run_overnight_ablation_sub08.sh` | `prompts_cpa.json` |
| `run_scr_sub08.sh` | `prompts_pred.json` |
| `run_top1_structure_sub08.sh` | `prompts_cpa.json` |