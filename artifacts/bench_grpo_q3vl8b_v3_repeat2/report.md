## Benchmark 对比报告
模型: ~/data/models/Qwen3-VL-8B-Instruct | 样本数: 162 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 10044.9 |
| LLM 时间(ms) | 10041.4 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 3.5 |
| 迭代次数 | 3.1 |
| Token 用量 | 7064.5 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 73% |
| pass^2 | 69% |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 2 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 97% |
| 严格成功率 | 73% |
| 工具调用准确率 | 83% |
| 参数准确率 | 83% |
| 指代解析准确率 | 86% |
| 阻塞终止语义准确率 | 0% |
| 假完成率 | 4% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.2 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 9698.7 |
| 工具调用 / 成功 | 2.9 |
| 耗时(ms) / 成功 | 13790.5 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 2.5 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 3 |
| 声称完成但错 | 39 |
| 异常终止 | 2 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 19 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 6 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 56 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 2 | 16788 | 100% | 100% | 100% |
| analyze_cv5_ref1_desc | 2 | 16566.5 | 100% | 100% | 100% |
| analyze_cv5_ref3_trend | 2 | 16707 | 100% | 100% | 100% |
| analyze_ro5_ref2_axes | 2 | 16798.5 | 100% | 50% | 100% |
| chain_ai5_dl2_tr2 | 2 | 15805.5 | 100% | 100% | 100% |
| chain_ai5_read_then_summary | 2 | 21178 | 100% | 0% | 0% |
| chain_cl5_dl3_tr3 | 2 | 15885 | 100% | 100% | 100% |
| chain_cv5_cache_dl_tr_cache | 2 | 19876 | 100% | 100% | 100% |
| chain_lg10_dl_tr_last | 2 | 14990 | 100% | 50% | 100% |
| chain_ro5_dl_three | 2 | 15848.5 | 100% | 100% | 100% |
| constraint_cache_only | 2 | 5046 | 100% | 100% | 100% |
| constraint_download_only | 2 | 4941.5 | 100% | 100% | 100% |
| constraint_search_no_file | 2 | 6232 | 100% | 0% | 100% |
| constraint_search_only | 2 | 6425.5 | 100% | 100% | 100% |
| constraint_translate_only | 2 | 6445.5 | 100% | 100% | 100% |
| constraint_two_downloads_only | 2 | 8664.5 | 100% | 100% | 100% |
| figure_ai5_ref4 | 2 | 13301 | 100% | 100% | 100% |
| figure_cv5_ref1 | 2 | 12496 | 100% | 100% | 100% |
| figure_cv5_ref3 | 2 | 12503.5 | 100% | 100% | 100% |
| figure_ro5_ref2 | 2 | 12502.5 | 100% | 100% | 100% |
| infeasible_index_out_of_range | 2 | 5145.5 | 100% | 0% | 0% |
| infeasible_no_session | 2 | 8347.5 | 100% | 0% | 0% |
| infeasible_unknown_id | 2 | 4757.5 | 100% | 0% | 0% |
| infeasible_unsupported_action | 2 | 18976.5 | 0% | 0% | 0% |
| infeasible_zero_index | 2 | 4826 | 100% | 0% | 0% |
| multi_ai3_tr1 | 2 | 9069.5 | 100% | 100% | 100% |
| multi_ai5_dl1_tr | 2 | 12365 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 2 | 12512 | 100% | 100% | 100% |
| multi_cr5_cache1 | 2 | 9563.5 | 100% | 100% | 100% |
| multi_cv3_dl1 | 2 | 9116 | 100% | 100% | 100% |
| multi_cv5_dl1_cache_tr | 2 | 15847 | 100% | 100% | 100% |
| multi_lg10_dl3 | 2 | 9524 | 100% | 100% | 100% |
| multi_ro3_dl_two | 2 | 13035.5 | 100% | 100% | 100% |
| opt_force_dl | 2 | 5162 | 100% | 100% | 100% |
| opt_force_tr | 2 | 5217 | 100% | 100% | 100% |
| opt_keep_dual | 2 | 5507.5 | 100% | 100% | 100% |
| opt_service | 2 | 5210.5 | 100% | 50% | 100% |
| opt_threads | 2 | 5357.5 | 100% | 100% | 100% |
| qa_cr5_abstract | 2 | 12584 | 100% | 50% | 100% |
| read_ai5_default | 2 | 12581 | 100% | 100% | 100% |
| read_cl5_result | 2 | 13116.5 | 100% | 100% | 100% |
| read_cv5_method | 2 | 13133.5 | 100% | 100% | 100% |
| read_lg5_conclusion | 2 | 13594 | 100% | 100% | 100% |
| ref_ctrl_id_cache | 2 | 7467.5 | 100% | 100% | 100% |
| ref_ctrl_id_download | 2 | 4724 | 100% | 0% | 100% |
| ref_ctrl_null_cache | 2 | 5496 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 2 | 5913.5 | 100% | 100% | 100% |
| ref_ctrl_word_handover | 2 | 14901 | 50% | 0% | 50% |
| ref_ctrl_word_marionette | 2 | 6561 | 100% | 50% | 100% |
| ref_stress_image_restoration | 2 | 6163 | 100% | 100% | 100% |
| ref_stress_learning_state | 2 | 6665 | 100% | 0% | 100% |
| ref_stress_state_across | 2 | 9824.5 | 100% | 0% | 0% |
| ref_stress_uncertainty_aware | 2 | 4924 | 100% | 50% | 100% |
| search_AI_1d_3 | 2 | 9487.5 | 100% | 100% | 100% |
| search_AI_30d_25 | 2 | 6553 | 100% | 100% | 100% |
| search_CL_30d_10 | 2 | 6565.5 | 100% | 100% | 100% |
| search_CL_7d_5 | 2 | 6179.5 | 100% | 100% | 100% |
| search_CR_14d_10 | 2 | 5821 | 100% | 100% | 100% |
| search_CR_7d_5 | 2 | 6831 | 100% | 100% | 100% |
| search_CV_14d_3 | 2 | 5766.5 | 100% | 100% | 100% |
| search_CV_1d_1 | 2 | 6884 | 100% | 100% | 100% |
| search_LG_14d_1 | 2 | 6305 | 100% | 100% | 100% |
| search_LG_3d_10 | 2 | 6711 | 100% | 100% | 100% |
| search_RO_30d_25 | 2 | 6724 | 100% | 100% | 100% |
| search_RO_3d_8 | 2 | 6204 | 100% | 100% | 100% |
| search_kw_agentic_rl | 2 | 6723.5 | 100% | 0% | 100% |
| search_kw_llm | 2 | 25626 | 0% | 0% | 0% |
| search_kw_rag | 2 | 6605 | 100% | 0% | 100% |
| state_cache_before_dl | 2 | 8131.5 | 100% | 100% | 100% |
| state_cache_two | 2 | 10576 | 100% | 100% | 100% |
| state_dl_then_cache | 2 | 8570 | 100% | 100% | 100% |
| state_ref_cache_active | 2 | 5300 | 100% | 100% | 100% |
| state_ref_last_active | 2 | 5618.5 | 100% | 100% | 100% |
| state_ref_last_of_10 | 2 | 4785.5 | 100% | 100% | 100% |
| state_ref_ordinal_cn | 2 | 4881.5 | 100% | 100% | 100% |
| state_translate_active | 2 | 5494 | 100% | 100% | 100% |
| summary_ai5_tldr60 | 2 | 16587.5 | 100% | 0% | 0% |
| summary_cl5_bullet250 | 2 | 16850 | 100% | 0% | 0% |
| summary_cv5_structured120 | 2 | 16994.5 | 100% | 0% | 0% |
| summary_lg5_tldr120 | 2 | 17065.5 | 100% | 0% | 0% |
| summary_ro5_structured250 | 2 | 17609 | 100% | 0% | 0% |