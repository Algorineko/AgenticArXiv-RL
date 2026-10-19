## Benchmark 对比报告
模型: ~/data/models/Qwen3.8-27B | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 148613.5 |
| LLM 时间(ms) | 148605.9 |
| 工具时间(ms) | 0.4 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 7.2 |
| 迭代次数 | 4.4 |
| Token 用量 | 10501 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 21% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 67% |
| 严格成功率 | 21% |
| 工具调用准确率 | 26% |
| 参数准确率 | 52% |
| 指代解析准确率 | 56% |
| 阻塞终止语义准确率 | 40% |
| 假完成率 | 5% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.3 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 50034.1 |
| 工具调用 / 成功 | 16.2 |
| 耗时(ms) / 成功 | 708099.8 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 5 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 27 |
| 声称完成但错 | 37 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 64 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 17 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 311991 | 0% | 0% | 0% |
| analyze_cv5_ref1_desc | 1 | 174468 | 100% | 0% | 100% |
| analyze_cv5_ref3_trend | 1 | 279861 | 100% | 0% | 0% |
| analyze_ro5_ref2_axes | 1 | 384833 | 100% | 0% | 0% |
| chain_ai5_dl2_tr2 | 1 | 140263 | 0% | 0% | 0% |
| chain_ai5_read_then_summary | 1 | 267541 | 0% | 0% | 0% |
| chain_cl5_dl3_tr3 | 1 | 311064 | 0% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 1 | 261025 | 100% | 0% | 0% |
| chain_lg10_dl_tr_last | 1 | 288351 | 0% | 0% | 0% |
| chain_ro5_dl_three | 1 | 131431 | 100% | 100% | 100% |
| constraint_cache_only | 1 | 104416 | 100% | 0% | 0% |
| constraint_download_only | 1 | 40992 | 100% | 100% | 100% |
| constraint_search_no_file | 1 | 154612 | 0% | 0% | 0% |
| constraint_search_only | 1 | 103319 | 100% | 100% | 100% |
| constraint_translate_only | 1 | 50979 | 100% | 100% | 100% |
| constraint_two_downloads_only | 1 | 176775 | 100% | 0% | 0% |
| figure_ai5_ref4 | 1 | 111810 | 100% | 0% | 100% |
| figure_cv5_ref1 | 1 | 103149 | 100% | 100% | 100% |
| figure_cv5_ref3 | 1 | 135356 | 0% | 0% | 0% |
| figure_ro5_ref2 | 1 | 135545 | 100% | 0% | 0% |
| infeasible_index_out_of_range | 1 | 106927 | 100% | 0% | 0% |
| infeasible_no_session | 1 | 20826 | 100% | 100% | 100% |
| infeasible_unknown_id | 1 | 154925 | 100% | 0% | 0% |
| infeasible_unsupported_action | 1 | 149448 | 100% | 0% | 0% |
| infeasible_zero_index | 1 | 62503 | 100% | 0% | 0% |
| multi_ai3_tr1 | 1 | 265328 | 0% | 0% | 0% |
| multi_ai5_dl1_tr | 1 | 163249 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 1 | 155985 | 0% | 0% | 0% |
| multi_cr5_cache1 | 1 | 295201 | 0% | 0% | 0% |
| multi_cv3_dl1 | 1 | 209745 | 100% | 0% | 0% |
| multi_cv5_dl1_cache_tr | 1 | 208044 | 100% | 100% | 100% |
| multi_lg10_dl3 | 1 | 134340 | 100% | 0% | 100% |
| multi_ro3_dl_two | 1 | 86268 | 100% | 100% | 100% |
| opt_force_dl | 1 | 97880 | 0% | 0% | 0% |
| opt_force_tr | 1 | 112795 | 100% | 0% | 0% |
| opt_keep_dual | 1 | 170029 | 100% | 0% | 0% |
| opt_service | 1 | 382243 | 0% | 0% | 0% |
| opt_threads | 1 | 369358 | 0% | 0% | 0% |
| qa_cr5_abstract | 1 | 181459 | 0% | 0% | 0% |
| read_ai5_default | 1 | 171361 | 100% | 0% | 100% |
| read_cl5_result | 1 | 330751 | 0% | 0% | 0% |
| read_cv5_method | 1 | 194658 | 100% | 0% | 0% |
| read_lg5_conclusion | 1 | 140465 | 0% | 0% | 0% |
| ref_ctrl_id_cache | 1 | 64598 | 100% | 0% | 0% |
| ref_ctrl_id_download | 1 | 87734 | 100% | 0% | 0% |
| ref_ctrl_null_cache | 1 | 122372 | 0% | 0% | 0% |
| ref_ctrl_null_translate | 1 | 194002 | 0% | 0% | 0% |
| ref_ctrl_word_handover | 1 | 150406 | 100% | 0% | 0% |
| ref_ctrl_word_marionette | 1 | 86381 | 100% | 0% | 0% |
| ref_stress_image_restoration | 1 | 95667 | 0% | 0% | 0% |
| ref_stress_learning_state | 1 | 89489 | 100% | 0% | 0% |
| ref_stress_state_across | 1 | 57146 | 100% | 0% | 0% |
| ref_stress_uncertainty_aware | 1 | 45533 | 100% | 100% | 100% |
| search_AI_1d_3 | 1 | 153758 | 100% | 0% | 0% |
| search_AI_30d_25 | 1 | 94747 | 100% | 0% | 0% |
| search_CL_30d_10 | 1 | 107434 | 100% | 0% | 0% |
| search_CL_7d_5 | 1 | 73350 | 100% | 0% | 0% |
| search_CR_14d_10 | 1 | 149595 | 0% | 0% | 0% |
| search_CR_7d_5 | 1 | 30026 | 100% | 100% | 100% |
| search_CV_14d_3 | 1 | 217216 | 0% | 0% | 0% |
| search_CV_1d_1 | 1 | 29139 | 100% | 100% | 100% |
| search_LG_14d_1 | 1 | 231112 | 100% | 0% | 0% |
| search_LG_3d_10 | 1 | 37454 | 100% | 100% | 100% |
| search_RO_30d_25 | 1 | 33283 | 100% | 100% | 100% |
| search_RO_3d_8 | 1 | 91794 | 100% | 0% | 0% |
| search_kw_agentic_rl | 1 | 54744 | 100% | 0% | 0% |
| search_kw_llm | 1 | 52930 | 100% | 0% | 0% |
| search_kw_rag | 1 | 55519 | 100% | 0% | 0% |
| state_cache_before_dl | 1 | 100577 | 100% | 0% | 0% |
| state_cache_two | 1 | 93351 | 0% | 0% | 0% |
| state_dl_then_cache | 1 | 240822 | 0% | 0% | 0% |
| state_ref_cache_active | 1 | 30050 | 100% | 100% | 100% |
| state_ref_last_active | 1 | 72365 | 100% | 0% | 0% |
| state_ref_last_of_10 | 1 | 87756 | 0% | 0% | 0% |
| state_ref_ordinal_cn | 1 | 115275 | 100% | 0% | 0% |
| state_translate_active | 1 | 26415 | 100% | 100% | 100% |
| summary_ai5_tldr60 | 1 | 233977 | 0% | 0% | 0% |
| summary_cl5_bullet250 | 1 | 323708 | 100% | 0% | 0% |
| summary_cv5_structured120 | 1 | 210254 | 0% | 0% | 0% |
| summary_lg5_tldr120 | 1 | 106393 | 0% | 0% | 0% |
| summary_ro5_structured250 | 1 | 159756 | 100% | 100% | 100% |