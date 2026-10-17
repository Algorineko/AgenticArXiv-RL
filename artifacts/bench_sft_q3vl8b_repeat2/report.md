## Benchmark 对比报告
模型: /home/tione/notebook/home/arianliu/data/models/Qwen3-VL-8B-Instruct | 样本数: 162 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 13636.6 |
| LLM 时间(ms) | 13633 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 3.7 |
| 迭代次数 | 3.1 |
| Token 用量 | 7087.7 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 61% |
| pass^2 | 53% |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 2 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 95% |
| 严格成功率 | 61% |
| 工具调用准确率 | 69% |
| 参数准确率 | 72% |
| 指代解析准确率 | 66% |
| 阻塞终止语义准确率 | 40% |
| 假完成率 | 10% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.3 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 11598.0 |
| 工具调用 / 成功 | 3.4 |
| 耗时(ms) / 成功 | 22314.5 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 3 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 8 |
| 声称完成但错 | 55 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 25 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 13 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 43 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 2 | 27726 | 50% | 50% | 50% |
| analyze_cv5_ref1_desc | 2 | 27505.5 | 50% | 50% | 50% |
| analyze_cv5_ref3_trend | 2 | 21129 | 100% | 100% | 100% |
| analyze_ro5_ref2_axes | 2 | 21972 | 100% | 50% | 100% |
| chain_ai5_dl2_tr2 | 2 | 20478 | 100% | 100% | 100% |
| chain_ai5_read_then_summary | 2 | 25424.5 | 100% | 100% | 100% |
| chain_cl5_dl3_tr3 | 2 | 19767 | 100% | 100% | 100% |
| chain_cv5_cache_dl_tr_cache | 2 | 34108.5 | 100% | 50% | 100% |
| chain_lg10_dl_tr_last | 2 | 22429.5 | 100% | 0% | 0% |
| chain_ro5_dl_three | 2 | 25930.5 | 50% | 0% | 50% |
| constraint_cache_only | 2 | 5518 | 100% | 100% | 100% |
| constraint_download_only | 2 | 10493.5 | 100% | 50% | 50% |
| constraint_search_no_file | 2 | 8622.5 | 100% | 0% | 100% |
| constraint_search_only | 2 | 8282 | 100% | 100% | 100% |
| constraint_translate_only | 2 | 9501 | 100% | 100% | 100% |
| constraint_two_downloads_only | 2 | 14399.5 | 100% | 50% | 50% |
| figure_ai5_ref4 | 2 | 6616.5 | 100% | 0% | 0% |
| figure_cv5_ref1 | 2 | 13209.5 | 100% | 100% | 100% |
| figure_cv5_ref3 | 2 | 11751 | 100% | 50% | 50% |
| figure_ro5_ref2 | 2 | 16241.5 | 100% | 100% | 100% |
| infeasible_index_out_of_range | 2 | 11365.5 | 100% | 0% | 50% |
| infeasible_no_session | 2 | 14258 | 100% | 0% | 0% |
| infeasible_unknown_id | 2 | 9594.5 | 100% | 0% | 0% |
| infeasible_unsupported_action | 2 | 3825.5 | 100% | 100% | 100% |
| infeasible_zero_index | 2 | 24010 | 50% | 0% | 0% |
| multi_ai3_tr1 | 2 | 24726 | 50% | 0% | 0% |
| multi_ai5_dl1_tr | 2 | 15522.5 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 2 | 16557 | 100% | 100% | 100% |
| multi_cr5_cache1 | 2 | 8799.5 | 100% | 0% | 50% |
| multi_cv3_dl1 | 2 | 25089 | 50% | 50% | 50% |
| multi_cv5_dl1_cache_tr | 2 | 24021.5 | 100% | 100% | 100% |
| multi_lg10_dl3 | 2 | 8584.5 | 100% | 50% | 50% |
| multi_ro3_dl_two | 2 | 26183 | 100% | 0% | 50% |
| opt_force_dl | 2 | 5912 | 100% | 100% | 100% |
| opt_force_tr | 2 | 6572 | 100% | 100% | 100% |
| opt_keep_dual | 2 | 7595.5 | 100% | 100% | 100% |
| opt_service | 2 | 8540 | 100% | 100% | 100% |
| opt_threads | 2 | 6305 | 100% | 100% | 100% |
| qa_cr5_abstract | 2 | 16041.5 | 100% | 0% | 100% |
| read_ai5_default | 2 | 18080 | 100% | 100% | 100% |
| read_cl5_result | 2 | 14871.5 | 100% | 100% | 100% |
| read_cv5_method | 2 | 19512 | 100% | 100% | 100% |
| read_lg5_conclusion | 2 | 18206 | 50% | 0% | 0% |
| ref_ctrl_id_cache | 2 | 6399.5 | 100% | 100% | 100% |
| ref_ctrl_id_download | 2 | 23734 | 100% | 0% | 0% |
| ref_ctrl_null_cache | 2 | 6348.5 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 2 | 7185.5 | 100% | 100% | 100% |
| ref_ctrl_word_handover | 2 | 5436 | 100% | 0% | 0% |
| ref_ctrl_word_marionette | 2 | 10184 | 100% | 0% | 0% |
| ref_stress_image_restoration | 2 | 18177 | 50% | 0% | 0% |
| ref_stress_learning_state | 2 | 11602 | 100% | 0% | 0% |
| ref_stress_state_across | 2 | 23333.5 | 100% | 0% | 0% |
| ref_stress_uncertainty_aware | 2 | 11515 | 100% | 0% | 0% |
| search_AI_1d_3 | 2 | 19121 | 100% | 100% | 100% |
| search_AI_30d_25 | 2 | 7493 | 100% | 100% | 100% |
| search_CL_30d_10 | 2 | 6911 | 100% | 100% | 100% |
| search_CL_7d_5 | 2 | 6332.5 | 100% | 100% | 100% |
| search_CR_14d_10 | 2 | 6442 | 100% | 50% | 100% |
| search_CR_7d_5 | 2 | 7821 | 100% | 100% | 100% |
| search_CV_14d_3 | 2 | 6299 | 100% | 100% | 100% |
| search_CV_1d_1 | 2 | 23815 | 100% | 0% | 0% |
| search_LG_14d_1 | 2 | 13001.5 | 100% | 50% | 50% |
| search_LG_3d_10 | 2 | 7704.5 | 100% | 100% | 100% |
| search_RO_30d_25 | 2 | 7849 | 100% | 100% | 100% |
| search_RO_3d_8 | 2 | 7601 | 100% | 100% | 100% |
| search_kw_agentic_rl | 2 | 7228.5 | 100% | 100% | 100% |
| search_kw_llm | 2 | 8182 | 100% | 100% | 100% |
| search_kw_rag | 2 | 7664.5 | 100% | 100% | 100% |
| state_cache_before_dl | 2 | 9734.5 | 100% | 100% | 100% |
| state_cache_two | 2 | 8459 | 100% | 100% | 100% |
| state_dl_then_cache | 2 | 20801 | 100% | 0% | 0% |
| state_ref_cache_active | 2 | 6366.5 | 100% | 100% | 100% |
| state_ref_last_active | 2 | 8189.5 | 100% | 100% | 100% |
| state_ref_last_of_10 | 2 | 8790 | 100% | 50% | 50% |
| state_ref_ordinal_cn | 2 | 12111 | 100% | 0% | 0% |
| state_translate_active | 2 | 8555 | 100% | 100% | 100% |
| summary_ai5_tldr60 | 2 | 6219.5 | 100% | 0% | 0% |
| summary_cl5_bullet250 | 2 | 11481 | 100% | 50% | 50% |
| summary_cv5_structured120 | 2 | 15973 | 100% | 100% | 100% |
| summary_lg5_tldr120 | 2 | 16746 | 100% | 0% | 100% |
| summary_ro5_structured250 | 2 | 18483.5 | 100% | 100% | 100% |