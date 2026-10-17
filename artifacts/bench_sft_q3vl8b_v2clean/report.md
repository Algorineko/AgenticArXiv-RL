## Benchmark 对比报告
模型: /home/tione/notebook/home/arianliu/data/models/Qwen3-VL-8B-Instruct | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 16189 |
| LLM 时间(ms) | 16184.9 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 4.1 |
| 迭代次数 | 3.7 |
| Token 用量 | 8077.9 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 60% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 80% |
| 严格成功率 | 60% |
| 工具调用准确率 | 65% |
| 参数准确率 | 74% |
| 指代解析准确率 | 72% |
| 阻塞终止语义准确率 | 20% |
| 假完成率 | 7% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.7 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 13353.3 |
| 工具调用 / 成功 | 4.4 |
| 耗时(ms) / 成功 | 26761.4 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 6 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 16 |
| 声称完成但错 | 16 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 32 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 49 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 27545 | 0% | 0% | 0% |
| analyze_cv5_ref1_desc | 1 | 25473 | 100% | 0% | 0% |
| analyze_cv5_ref3_trend | 1 | 30712 | 0% | 0% | 0% |
| analyze_ro5_ref2_axes | 1 | 17146 | 100% | 0% | 0% |
| chain_ai5_dl2_tr2 | 1 | 22573 | 100% | 100% | 100% |
| chain_ai5_read_then_summary | 1 | 22142 | 100% | 100% | 100% |
| chain_cl5_dl3_tr3 | 1 | 14210 | 100% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 1 | 33545 | 100% | 100% | 100% |
| chain_lg10_dl_tr_last | 1 | 31262 | 0% | 0% | 0% |
| chain_ro5_dl_three | 1 | 20832 | 100% | 100% | 100% |
| constraint_cache_only | 1 | 9557 | 100% | 100% | 100% |
| constraint_download_only | 1 | 6266 | 100% | 100% | 100% |
| constraint_search_no_file | 1 | 10041 | 100% | 0% | 100% |
| constraint_search_only | 1 | 29826 | 0% | 0% | 0% |
| constraint_translate_only | 1 | 37916 | 0% | 0% | 0% |
| constraint_two_downloads_only | 1 | 13738 | 100% | 100% | 100% |
| figure_ai5_ref4 | 1 | 15178 | 100% | 0% | 100% |
| figure_cv5_ref1 | 1 | 13517 | 100% | 100% | 100% |
| figure_cv5_ref3 | 1 | 19013 | 100% | 100% | 100% |
| figure_ro5_ref2 | 1 | 14631 | 100% | 100% | 100% |
| infeasible_index_out_of_range | 1 | 15207 | 100% | 0% | 0% |
| infeasible_no_session | 1 | 24361 | 0% | 0% | 0% |
| infeasible_unknown_id | 1 | 36760 | 0% | 0% | 0% |
| infeasible_unsupported_action | 1 | 4096 | 100% | 100% | 100% |
| infeasible_zero_index | 1 | 21506 | 0% | 0% | 0% |
| multi_ai3_tr1 | 1 | 19483 | 100% | 0% | 0% |
| multi_ai5_dl1_tr | 1 | 16331 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 1 | 14700 | 100% | 0% | 100% |
| multi_cr5_cache1 | 1 | 13622 | 100% | 100% | 100% |
| multi_cv3_dl1 | 1 | 14124 | 100% | 100% | 100% |
| multi_cv5_dl1_cache_tr | 1 | 26155 | 0% | 0% | 0% |
| multi_lg10_dl3 | 1 | 25657 | 100% | 0% | 0% |
| multi_ro3_dl_two | 1 | 15258 | 100% | 100% | 100% |
| opt_force_dl | 1 | 5342 | 100% | 100% | 100% |
| opt_force_tr | 1 | 6200 | 100% | 100% | 100% |
| opt_keep_dual | 1 | 10870 | 100% | 100% | 100% |
| opt_service | 1 | 6209 | 100% | 100% | 100% |
| opt_threads | 1 | 6452 | 100% | 100% | 100% |
| qa_cr5_abstract | 1 | 16007 | 100% | 100% | 100% |
| read_ai5_default | 1 | 16111 | 100% | 0% | 100% |
| read_cl5_result | 1 | 14566 | 100% | 100% | 100% |
| read_cv5_method | 1 | 20993 | 100% | 0% | 0% |
| read_lg5_conclusion | 1 | 16152 | 100% | 100% | 100% |
| ref_ctrl_id_cache | 1 | 8139 | 100% | 100% | 100% |
| ref_ctrl_id_download | 1 | 6780 | 100% | 100% | 100% |
| ref_ctrl_null_cache | 1 | 7947 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 1 | 10283 | 100% | 100% | 100% |
| ref_ctrl_word_handover | 1 | 8173 | 100% | 0% | 0% |
| ref_ctrl_word_marionette | 1 | 7749 | 100% | 0% | 0% |
| ref_stress_image_restoration | 1 | 21009 | 0% | 0% | 0% |
| ref_stress_learning_state | 1 | 25704 | 0% | 0% | 0% |
| ref_stress_state_across | 1 | 24118 | 100% | 0% | 0% |
| ref_stress_uncertainty_aware | 1 | 9441 | 100% | 0% | 0% |
| search_AI_1d_3 | 1 | 36083 | 100% | 0% | 0% |
| search_AI_30d_25 | 1 | 7418 | 100% | 100% | 100% |
| search_CL_30d_10 | 1 | 8787 | 100% | 100% | 100% |
| search_CL_7d_5 | 1 | 7404 | 100% | 100% | 100% |
| search_CR_14d_10 | 1 | 6204 | 100% | 100% | 100% |
| search_CR_7d_5 | 1 | 7510 | 100% | 100% | 100% |
| search_CV_14d_3 | 1 | 8255 | 100% | 100% | 100% |
| search_CV_1d_1 | 1 | 36467 | 0% | 0% | 0% |
| search_LG_14d_1 | 1 | 29067 | 0% | 0% | 0% |
| search_LG_3d_10 | 1 | 7806 | 100% | 100% | 100% |
| search_RO_30d_25 | 1 | 7611 | 100% | 100% | 100% |
| search_RO_3d_8 | 1 | 6123 | 100% | 100% | 100% |
| search_kw_agentic_rl | 1 | 6567 | 100% | 100% | 100% |
| search_kw_llm | 1 | 10150 | 100% | 100% | 100% |
| search_kw_rag | 1 | 40149 | 0% | 0% | 0% |
| state_cache_before_dl | 1 | 9437 | 100% | 100% | 100% |
| state_cache_two | 1 | 8811 | 100% | 100% | 100% |
| state_dl_then_cache | 1 | 9526 | 100% | 100% | 100% |
| state_ref_cache_active | 1 | 5872 | 100% | 100% | 100% |
| state_ref_last_active | 1 | 10169 | 100% | 100% | 100% |
| state_ref_last_of_10 | 1 | 5496 | 100% | 100% | 100% |
| state_ref_ordinal_cn | 1 | 5753 | 100% | 100% | 100% |
| state_translate_active | 1 | 9027 | 100% | 100% | 100% |
| summary_ai5_tldr60 | 1 | 16346 | 100% | 100% | 100% |
| summary_cl5_bullet250 | 1 | 17249 | 100% | 100% | 100% |
| summary_cv5_structured120 | 1 | 26725 | 0% | 0% | 0% |
| summary_lg5_tldr120 | 1 | 16378 | 100% | 100% | 100% |
| summary_ro5_structured250 | 1 | 34292 | 0% | 0% | 0% |