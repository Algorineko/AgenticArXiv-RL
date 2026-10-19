## Benchmark 对比报告
模型: ~/data/models/Qwen3.8-27B | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 186548.1 |
| LLM 时间(ms) | 186542.2 |
| 工具时间(ms) | 0.1 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 5.8 |
| 迭代次数 | 4.5 |
| Token 用量 | 10636.7 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 15% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 51% |
| 严格成功率 | 15% |
| 工具调用准确率 | 20% |
| 参数准确率 | 43% |
| 指代解析准确率 | 39% |
| 阻塞终止语义准确率 | 20% |
| 假完成率 | 12% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.5 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 71798 |
| 工具调用 / 成功 | 23.8 |
| 耗时(ms) / 成功 | 1259199.6 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 6 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 40 |
| 声称完成但错 | 29 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 69 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 12 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 321650 | 100% | 0% | 0% |
| analyze_cv5_ref1_desc | 1 | 160744 | 100% | 0% | 0% |
| analyze_cv5_ref3_trend | 1 | 290643 | 0% | 0% | 0% |
| analyze_ro5_ref2_axes | 1 | 556882 | 0% | 0% | 0% |
| chain_ai5_dl2_tr2 | 1 | 367746 | 0% | 0% | 0% |
| chain_ai5_read_then_summary | 1 | 335148 | 100% | 0% | 0% |
| chain_cl5_dl3_tr3 | 1 | 341474 | 0% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 1 | 71099 | 100% | 0% | 0% |
| chain_lg10_dl_tr_last | 1 | 226312 | 100% | 0% | 100% |
| chain_ro5_dl_three | 1 | 358624 | 0% | 0% | 0% |
| constraint_cache_only | 1 | 158823 | 0% | 0% | 0% |
| constraint_download_only | 1 | 45275 | 100% | 100% | 100% |
| constraint_search_no_file | 1 | 226050 | 0% | 0% | 0% |
| constraint_search_only | 1 | 379817 | 0% | 0% | 0% |
| constraint_translate_only | 1 | 94515 | 100% | 0% | 0% |
| constraint_two_downloads_only | 1 | 145176 | 0% | 0% | 0% |
| figure_ai5_ref4 | 1 | 119518 | 100% | 100% | 100% |
| figure_cv5_ref1 | 1 | 190729 | 0% | 0% | 0% |
| figure_cv5_ref3 | 1 | 108231 | 100% | 0% | 0% |
| figure_ro5_ref2 | 1 | 249478 | 0% | 0% | 0% |
| infeasible_index_out_of_range | 1 | 34042 | 100% | 0% | 100% |
| infeasible_no_session | 1 | 132710 | 0% | 0% | 0% |
| infeasible_unknown_id | 1 | 148934 | 100% | 0% | 0% |
| infeasible_unsupported_action | 1 | 34787 | 100% | 100% | 100% |
| infeasible_zero_index | 1 | 149658 | 0% | 0% | 0% |
| multi_ai3_tr1 | 1 | 60127 | 100% | 0% | 0% |
| multi_ai5_dl1_tr | 1 | 236997 | 0% | 0% | 0% |
| multi_cl5_dl2_cache | 1 | 186503 | 100% | 0% | 0% |
| multi_cr5_cache1 | 1 | 347288 | 0% | 0% | 0% |
| multi_cv3_dl1 | 1 | 217745 | 100% | 0% | 0% |
| multi_cv5_dl1_cache_tr | 1 | 284718 | 0% | 0% | 0% |
| multi_lg10_dl3 | 1 | 259566 | 0% | 0% | 0% |
| multi_ro3_dl_two | 1 | 228544 | 0% | 0% | 0% |
| opt_force_dl | 1 | 168865 | 0% | 0% | 0% |
| opt_force_tr | 1 | 176032 | 0% | 0% | 0% |
| opt_keep_dual | 1 | 116298 | 100% | 0% | 0% |
| opt_service | 1 | 266096 | 0% | 0% | 0% |
| opt_threads | 1 | 134580 | 0% | 0% | 0% |
| qa_cr5_abstract | 1 | 292024 | 0% | 0% | 0% |
| read_ai5_default | 1 | 218640 | 0% | 0% | 0% |
| read_cl5_result | 1 | 131301 | 100% | 0% | 0% |
| read_cv5_method | 1 | 265082 | 100% | 0% | 0% |
| read_lg5_conclusion | 1 | 197301 | 0% | 0% | 0% |
| ref_ctrl_id_cache | 1 | 59608 | 100% | 100% | 100% |
| ref_ctrl_id_download | 1 | 48529 | 100% | 100% | 100% |
| ref_ctrl_null_cache | 1 | 58346 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 1 | 130959 | 100% | 0% | 0% |
| ref_ctrl_word_handover | 1 | 446463 | 0% | 0% | 0% |
| ref_ctrl_word_marionette | 1 | 245439 | 100% | 0% | 0% |
| ref_stress_image_restoration | 1 | 80278 | 100% | 0% | 100% |
| ref_stress_learning_state | 1 | 133208 | 100% | 0% | 0% |
| ref_stress_state_across | 1 | 196799 | 0% | 0% | 0% |
| ref_stress_uncertainty_aware | 1 | 225100 | 100% | 0% | 0% |
| search_AI_1d_3 | 1 | 86374 | 100% | 100% | 100% |
| search_AI_30d_25 | 1 | 31545 | 100% | 100% | 100% |
| search_CL_30d_10 | 1 | 73779 | 100% | 100% | 100% |
| search_CL_7d_5 | 1 | 109780 | 100% | 100% | 100% |
| search_CR_14d_10 | 1 | 284331 | 0% | 0% | 0% |
| search_CR_7d_5 | 1 | 260428 | 0% | 0% | 0% |
| search_CV_14d_3 | 1 | 82372 | 100% | 0% | 100% |
| search_CV_1d_1 | 1 | 71408 | 100% | 100% | 100% |
| search_LG_14d_1 | 1 | 248769 | 0% | 0% | 0% |
| search_LG_3d_10 | 1 | 155845 | 100% | 0% | 0% |
| search_RO_30d_25 | 1 | 301536 | 0% | 0% | 0% |
| search_RO_3d_8 | 1 | 79891 | 100% | 100% | 100% |
| search_kw_agentic_rl | 1 | 224514 | 0% | 0% | 0% |
| search_kw_llm | 1 | 187444 | 0% | 0% | 0% |
| search_kw_rag | 1 | 147515 | 100% | 0% | 0% |
| state_cache_before_dl | 1 | 201625 | 0% | 0% | 0% |
| state_cache_two | 1 | 86094 | 100% | 0% | 0% |
| state_dl_then_cache | 1 | 228439 | 100% | 0% | 0% |
| state_ref_cache_active | 1 | 139845 | 0% | 0% | 0% |
| state_ref_last_active | 1 | 57086 | 100% | 0% | 0% |
| state_ref_last_of_10 | 1 | 53319 | 100% | 0% | 0% |
| state_ref_ordinal_cn | 1 | 139969 | 0% | 0% | 0% |
| state_translate_active | 1 | 122994 | 100% | 0% | 0% |
| summary_ai5_tldr60 | 1 | 221940 | 0% | 0% | 0% |
| summary_cl5_bullet250 | 1 | 266019 | 0% | 0% | 0% |
| summary_cv5_structured120 | 1 | 148439 | 100% | 0% | 0% |
| summary_lg5_tldr120 | 1 | 214522 | 0% | 0% | 0% |
| summary_ro5_structured250 | 1 | 224072 | 0% | 0% | 0% |