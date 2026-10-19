## Benchmark 对比报告
模型: ~/data/models/Qwen3-VL-8B-Instruct | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 11328.3 |
| LLM 时间(ms) | 11324.3 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 3.9 |
| 迭代次数 | 3.1 |
| Token 用量 | 6971.7 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 73% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 98% |
| 严格成功率 | 73% |
| 工具调用准确率 | 83% |
| 参数准确率 | 83% |
| 指代解析准确率 | 86% |
| 阻塞终止语义准确率 | 0% |
| 假完成率 | 6% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.2 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 9571.3 |
| 工具调用 / 成功 | 2.9 |
| 耗时(ms) / 成功 | 15552.4 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 2.5 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 1 |
| 声称完成但错 | 20 |
| 异常终止 | 1 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 22 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 59 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 19634 | 100% | 100% | 100% |
| analyze_cv5_ref1_desc | 1 | 19452 | 100% | 100% | 100% |
| analyze_cv5_ref3_trend | 1 | 17950 | 100% | 100% | 100% |
| analyze_ro5_ref2_axes | 1 | 19787 | 100% | 100% | 100% |
| chain_ai5_dl2_tr2 | 1 | 18832 | 100% | 100% | 100% |
| chain_ai5_read_then_summary | 1 | 20063 | 100% | 100% | 100% |
| chain_cl5_dl3_tr3 | 1 | 14508 | 100% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 1 | 23211 | 100% | 100% | 100% |
| chain_lg10_dl_tr_last | 1 | 16182 | 100% | 100% | 100% |
| chain_ro5_dl_three | 1 | 18613 | 100% | 100% | 100% |
| constraint_cache_only | 1 | 5863 | 100% | 100% | 100% |
| constraint_download_only | 1 | 5846 | 100% | 100% | 100% |
| constraint_search_no_file | 1 | 7360 | 100% | 0% | 100% |
| constraint_search_only | 1 | 7853 | 100% | 100% | 100% |
| constraint_translate_only | 1 | 5999 | 100% | 100% | 100% |
| constraint_two_downloads_only | 1 | 12367 | 100% | 0% | 0% |
| figure_ai5_ref4 | 1 | 15826 | 100% | 100% | 100% |
| figure_cv5_ref1 | 1 | 14655 | 100% | 100% | 100% |
| figure_cv5_ref3 | 1 | 14179 | 100% | 100% | 100% |
| figure_ro5_ref2 | 1 | 14752 | 100% | 100% | 100% |
| infeasible_index_out_of_range | 1 | 5975 | 100% | 0% | 0% |
| infeasible_no_session | 1 | 10631 | 100% | 0% | 0% |
| infeasible_unknown_id | 1 | 5551 | 100% | 0% | 0% |
| infeasible_unsupported_action | 1 | 21169 | 0% | 0% | 0% |
| infeasible_zero_index | 1 | 5573 | 100% | 0% | 0% |
| multi_ai3_tr1 | 1 | 10115 | 100% | 100% | 100% |
| multi_ai5_dl1_tr | 1 | 13953 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 1 | 13993 | 100% | 100% | 100% |
| multi_cr5_cache1 | 1 | 11057 | 100% | 100% | 100% |
| multi_cv3_dl1 | 1 | 10278 | 100% | 100% | 100% |
| multi_cv5_dl1_cache_tr | 1 | 17542 | 100% | 100% | 100% |
| multi_lg10_dl3 | 1 | 10539 | 100% | 100% | 100% |
| multi_ro3_dl_two | 1 | 14423 | 100% | 100% | 100% |
| opt_force_dl | 1 | 5695 | 100% | 100% | 100% |
| opt_force_tr | 1 | 5373 | 100% | 100% | 100% |
| opt_keep_dual | 1 | 6082 | 100% | 100% | 100% |
| opt_service | 1 | 5810 | 100% | 100% | 100% |
| opt_threads | 1 | 5923 | 100% | 100% | 100% |
| qa_cr5_abstract | 1 | 14162 | 100% | 0% | 100% |
| read_ai5_default | 1 | 13978 | 100% | 100% | 100% |
| read_cl5_result | 1 | 14533 | 100% | 100% | 100% |
| read_cv5_method | 1 | 14644 | 100% | 100% | 100% |
| read_lg5_conclusion | 1 | 14876 | 100% | 100% | 100% |
| ref_ctrl_id_cache | 1 | 5882 | 100% | 100% | 100% |
| ref_ctrl_id_download | 1 | 5175 | 100% | 0% | 100% |
| ref_ctrl_null_cache | 1 | 6115 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 1 | 6624 | 100% | 100% | 100% |
| ref_ctrl_word_handover | 1 | 6698 | 100% | 0% | 100% |
| ref_ctrl_word_marionette | 1 | 8793 | 100% | 100% | 100% |
| ref_stress_image_restoration | 1 | 6880 | 100% | 100% | 100% |
| ref_stress_learning_state | 1 | 8408 | 100% | 0% | 100% |
| ref_stress_state_across | 1 | 11022 | 100% | 0% | 0% |
| ref_stress_uncertainty_aware | 1 | 5401 | 100% | 0% | 100% |
| search_AI_1d_3 | 1 | 19735 | 100% | 100% | 100% |
| search_AI_30d_25 | 1 | 7589 | 100% | 100% | 100% |
| search_CL_30d_10 | 1 | 7336 | 100% | 100% | 100% |
| search_CL_7d_5 | 1 | 7070 | 100% | 100% | 100% |
| search_CR_14d_10 | 1 | 6455 | 100% | 100% | 100% |
| search_CR_7d_5 | 1 | 7897 | 100% | 100% | 100% |
| search_CV_14d_3 | 1 | 6978 | 100% | 100% | 100% |
| search_CV_1d_1 | 1 | 7375 | 100% | 100% | 100% |
| search_LG_14d_1 | 1 | 7207 | 100% | 100% | 100% |
| search_LG_3d_10 | 1 | 7881 | 100% | 100% | 100% |
| search_RO_30d_25 | 1 | 7370 | 100% | 100% | 100% |
| search_RO_3d_8 | 1 | 6910 | 100% | 100% | 100% |
| search_kw_agentic_rl | 1 | 7339 | 100% | 0% | 100% |
| search_kw_llm | 1 | 30560 | 0% | 0% | 0% |
| search_kw_rag | 1 | 7421 | 100% | 0% | 100% |
| state_cache_before_dl | 1 | 9126 | 100% | 100% | 100% |
| state_cache_two | 1 | 10770 | 100% | 100% | 100% |
| state_dl_then_cache | 1 | 9469 | 100% | 100% | 100% |
| state_ref_cache_active | 1 | 5906 | 100% | 100% | 100% |
| state_ref_last_active | 1 | 5885 | 100% | 100% | 100% |
| state_ref_last_of_10 | 1 | 5248 | 100% | 100% | 100% |
| state_ref_ordinal_cn | 1 | 6060 | 100% | 100% | 100% |
| state_translate_active | 1 | 6625 | 100% | 100% | 100% |
| summary_ai5_tldr60 | 1 | 18427 | 100% | 0% | 0% |
| summary_cl5_bullet250 | 1 | 19635 | 100% | 0% | 0% |
| summary_cv5_structured120 | 1 | 19687 | 100% | 0% | 0% |
| summary_lg5_tldr120 | 1 | 20304 | 100% | 0% | 0% |
| summary_ro5_structured250 | 1 | 19520 | 100% | 0% | 0% |