## Benchmark 对比报告
模型: ~/data/models/Qwen3.8-27B | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 160932.9 |
| LLM 时间(ms) | 160927.8 |
| 工具时间(ms) | 0.1 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 5 |
| 迭代次数 | 3.9 |
| Token 用量 | 9218.4 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 14% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 64% |
| 严格成功率 | 14% |
| 工具调用准确率 | 20% |
| 参数准确率 | 34% |
| 指代解析准确率 | 28% |
| 阻塞终止语义准确率 | 60% |
| 假完成率 | 25% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.5 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 67880.8 |
| 工具调用 / 成功 | 21.5 |
| 耗时(ms) / 成功 | 1185051.3 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 5 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 29 |
| 声称完成但错 | 41 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 70 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 11 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 204832 | 0% | 0% | 0% |
| analyze_cv5_ref1_desc | 1 | 29425 | 100% | 0% | 0% |
| analyze_cv5_ref3_trend | 1 | 237031 | 100% | 0% | 0% |
| analyze_ro5_ref2_axes | 1 | 154032 | 100% | 0% | 0% |
| chain_ai5_dl2_tr2 | 1 | 91968 | 100% | 0% | 0% |
| chain_ai5_read_then_summary | 1 | 331122 | 0% | 0% | 0% |
| chain_cl5_dl3_tr3 | 1 | 420109 | 0% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 1 | 132057 | 100% | 0% | 0% |
| chain_lg10_dl_tr_last | 1 | 285359 | 0% | 0% | 0% |
| chain_ro5_dl_three | 1 | 157043 | 0% | 0% | 0% |
| constraint_cache_only | 1 | 25221 | 100% | 0% | 0% |
| constraint_download_only | 1 | 134081 | 100% | 0% | 0% |
| constraint_search_no_file | 1 | 256424 | 0% | 0% | 0% |
| constraint_search_only | 1 | 69947 | 100% | 0% | 0% |
| constraint_translate_only | 1 | 65762 | 100% | 0% | 100% |
| constraint_two_downloads_only | 1 | 159974 | 0% | 0% | 0% |
| figure_ai5_ref4 | 1 | 117857 | 100% | 0% | 0% |
| figure_cv5_ref1 | 1 | 143799 | 100% | 0% | 0% |
| figure_cv5_ref3 | 1 | 85789 | 100% | 0% | 0% |
| figure_ro5_ref2 | 1 | 264548 | 0% | 0% | 0% |
| infeasible_index_out_of_range | 1 | 75120 | 100% | 100% | 100% |
| infeasible_no_session | 1 | 33924 | 100% | 100% | 100% |
| infeasible_unknown_id | 1 | 243742 | 0% | 0% | 0% |
| infeasible_unsupported_action | 1 | 45093 | 100% | 100% | 100% |
| infeasible_zero_index | 1 | 58601 | 100% | 0% | 0% |
| multi_ai3_tr1 | 1 | 174856 | 100% | 0% | 0% |
| multi_ai5_dl1_tr | 1 | 42188 | 100% | 0% | 0% |
| multi_cl5_dl2_cache | 1 | 187699 | 0% | 0% | 0% |
| multi_cr5_cache1 | 1 | 128502 | 100% | 0% | 0% |
| multi_cv3_dl1 | 1 | 274768 | 100% | 0% | 0% |
| multi_cv5_dl1_cache_tr | 1 | 228744 | 100% | 0% | 0% |
| multi_lg10_dl3 | 1 | 195655 | 100% | 100% | 100% |
| multi_ro3_dl_two | 1 | 68520 | 100% | 0% | 0% |
| opt_force_dl | 1 | 178184 | 0% | 0% | 0% |
| opt_force_tr | 1 | 164998 | 0% | 0% | 0% |
| opt_keep_dual | 1 | 45913 | 100% | 100% | 100% |
| opt_service | 1 | 103545 | 100% | 0% | 100% |
| opt_threads | 1 | 155612 | 100% | 0% | 0% |
| qa_cr5_abstract | 1 | 417117 | 0% | 0% | 0% |
| read_ai5_default | 1 | 376703 | 0% | 0% | 0% |
| read_cl5_result | 1 | 133188 | 100% | 0% | 0% |
| read_cv5_method | 1 | 260677 | 100% | 0% | 0% |
| read_lg5_conclusion | 1 | 255775 | 100% | 0% | 100% |
| ref_ctrl_id_cache | 1 | 127861 | 100% | 0% | 0% |
| ref_ctrl_id_download | 1 | 151851 | 0% | 0% | 0% |
| ref_ctrl_null_cache | 1 | 64031 | 100% | 0% | 0% |
| ref_ctrl_null_translate | 1 | 138660 | 0% | 0% | 0% |
| ref_ctrl_word_handover | 1 | 142757 | 100% | 0% | 0% |
| ref_ctrl_word_marionette | 1 | 169283 | 0% | 0% | 0% |
| ref_stress_image_restoration | 1 | 145526 | 0% | 0% | 0% |
| ref_stress_learning_state | 1 | 220923 | 0% | 0% | 0% |
| ref_stress_state_across | 1 | 165811 | 0% | 0% | 0% |
| ref_stress_uncertainty_aware | 1 | 62998 | 100% | 100% | 100% |
| search_AI_1d_3 | 1 | 214488 | 100% | 0% | 0% |
| search_AI_30d_25 | 1 | 112568 | 100% | 0% | 0% |
| search_CL_30d_10 | 1 | 72810 | 100% | 100% | 100% |
| search_CL_7d_5 | 1 | 315252 | 100% | 0% | 0% |
| search_CR_14d_10 | 1 | 118019 | 100% | 0% | 0% |
| search_CR_7d_5 | 1 | 204886 | 0% | 0% | 0% |
| search_CV_14d_3 | 1 | 146274 | 100% | 100% | 100% |
| search_CV_1d_1 | 1 | 82692 | 100% | 0% | 100% |
| search_LG_14d_1 | 1 | 314727 | 100% | 0% | 0% |
| search_LG_3d_10 | 1 | 298881 | 0% | 0% | 0% |
| search_RO_30d_25 | 1 | 174520 | 100% | 0% | 0% |
| search_RO_3d_8 | 1 | 146523 | 100% | 100% | 100% |
| search_kw_agentic_rl | 1 | 200468 | 0% | 0% | 0% |
| search_kw_llm | 1 | 209968 | 0% | 0% | 0% |
| search_kw_rag | 1 | 162985 | 100% | 0% | 0% |
| state_cache_before_dl | 1 | 176364 | 0% | 0% | 0% |
| state_cache_two | 1 | 176207 | 0% | 0% | 0% |
| state_dl_then_cache | 1 | 131304 | 100% | 100% | 100% |
| state_ref_cache_active | 1 | 143914 | 100% | 0% | 0% |
| state_ref_last_active | 1 | 58707 | 100% | 100% | 100% |
| state_ref_last_of_10 | 1 | 78066 | 100% | 0% | 0% |
| state_ref_ordinal_cn | 1 | 158729 | 0% | 0% | 0% |
| state_translate_active | 1 | 152453 | 0% | 0% | 0% |
| summary_ai5_tldr60 | 1 | 47303 | 100% | 0% | 0% |
| summary_cl5_bullet250 | 1 | 187254 | 100% | 0% | 100% |
| summary_cv5_structured120 | 1 | 86100 | 100% | 0% | 0% |
| summary_lg5_tldr120 | 1 | 194631 | 0% | 0% | 0% |
| summary_ro5_structured250 | 1 | 64266 | 100% | 0% | 0% |