## Benchmark 对比报告
模型: ~/data/models/Qwen3.8-27B | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 179554.7 |
| LLM 时间(ms) | 179545.5 |
| 工具时间(ms) | 0.1 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 9.1 |
| 迭代次数 | 4.3 |
| Token 用量 | 10125.8 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 22% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 56% |
| 严格成功率 | 22% |
| 工具调用准确率 | 27% |
| 参数准确率 | 41% |
| 指代解析准确率 | 36% |
| 阻塞终止语义准确率 | 20% |
| 假完成率 | 12% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.5 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 45566.2 |
| 工具调用 / 成功 | 14.8 |
| 耗时(ms) / 成功 | 807996.3 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 6 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 36 |
| 声称完成但错 | 27 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 63 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 18 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 186113 | 100% | 0% | 0% |
| analyze_cv5_ref1_desc | 1 | 251676 | 0% | 0% | 0% |
| analyze_cv5_ref3_trend | 1 | 358826 | 100% | 0% | 100% |
| analyze_ro5_ref2_axes | 1 | 28831 | 100% | 0% | 0% |
| chain_ai5_dl2_tr2 | 1 | 236839 | 0% | 0% | 0% |
| chain_ai5_read_then_summary | 1 | 406004 | 0% | 0% | 0% |
| chain_cl5_dl3_tr3 | 1 | 131460 | 100% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 1 | 439386 | 0% | 0% | 0% |
| chain_lg10_dl_tr_last | 1 | 388275 | 100% | 0% | 0% |
| chain_ro5_dl_three | 1 | 151045 | 100% | 0% | 0% |
| constraint_cache_only | 1 | 197049 | 0% | 0% | 0% |
| constraint_download_only | 1 | 176496 | 0% | 0% | 0% |
| constraint_search_no_file | 1 | 50164 | 100% | 100% | 100% |
| constraint_search_only | 1 | 204846 | 0% | 0% | 0% |
| constraint_translate_only | 1 | 62609 | 100% | 0% | 100% |
| constraint_two_downloads_only | 1 | 182211 | 0% | 0% | 0% |
| figure_ai5_ref4 | 1 | 164306 | 0% | 0% | 0% |
| figure_cv5_ref1 | 1 | 323710 | 100% | 100% | 100% |
| figure_cv5_ref3 | 1 | 121876 | 100% | 0% | 0% |
| figure_ro5_ref2 | 1 | 409216 | 0% | 0% | 0% |
| infeasible_index_out_of_range | 1 | 313897 | 0% | 0% | 0% |
| infeasible_no_session | 1 | 167946 | 0% | 0% | 0% |
| infeasible_unknown_id | 1 | 234219 | 100% | 0% | 0% |
| infeasible_unsupported_action | 1 | 44003 | 100% | 100% | 100% |
| infeasible_zero_index | 1 | 195026 | 0% | 0% | 0% |
| multi_ai3_tr1 | 1 | 177873 | 100% | 0% | 0% |
| multi_ai5_dl1_tr | 1 | 192827 | 0% | 0% | 0% |
| multi_cl5_dl2_cache | 1 | 342211 | 0% | 0% | 0% |
| multi_cr5_cache1 | 1 | 289517 | 0% | 0% | 0% |
| multi_cv3_dl1 | 1 | 335080 | 100% | 0% | 0% |
| multi_cv5_dl1_cache_tr | 1 | 242823 | 0% | 0% | 0% |
| multi_lg10_dl3 | 1 | 193918 | 0% | 0% | 0% |
| multi_ro3_dl_two | 1 | 468528 | 0% | 0% | 0% |
| opt_force_dl | 1 | 156593 | 0% | 0% | 0% |
| opt_force_tr | 1 | 60112 | 100% | 0% | 0% |
| opt_keep_dual | 1 | 140157 | 100% | 0% | 0% |
| opt_service | 1 | 117230 | 100% | 0% | 0% |
| opt_threads | 1 | 134593 | 0% | 0% | 0% |
| qa_cr5_abstract | 1 | 105357 | 100% | 0% | 0% |
| read_ai5_default | 1 | 200395 | 0% | 0% | 0% |
| read_cl5_result | 1 | 94267 | 100% | 0% | 0% |
| read_cv5_method | 1 | 208677 | 100% | 100% | 100% |
| read_lg5_conclusion | 1 | 347296 | 0% | 0% | 0% |
| ref_ctrl_id_cache | 1 | 52338 | 100% | 100% | 100% |
| ref_ctrl_id_download | 1 | 48018 | 100% | 100% | 100% |
| ref_ctrl_null_cache | 1 | 62066 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 1 | 180479 | 0% | 0% | 0% |
| ref_ctrl_word_handover | 1 | 260681 | 100% | 0% | 0% |
| ref_ctrl_word_marionette | 1 | 142023 | 100% | 0% | 0% |
| ref_stress_image_restoration | 1 | 80463 | 100% | 0% | 100% |
| ref_stress_learning_state | 1 | 96552 | 100% | 0% | 100% |
| ref_stress_state_across | 1 | 193859 | 0% | 0% | 0% |
| ref_stress_uncertainty_aware | 1 | 205243 | 0% | 0% | 0% |
| search_AI_1d_3 | 1 | 79712 | 100% | 0% | 0% |
| search_AI_30d_25 | 1 | 34485 | 100% | 100% | 100% |
| search_CL_30d_10 | 1 | 198209 | 0% | 0% | 0% |
| search_CL_7d_5 | 1 | 89517 | 100% | 100% | 100% |
| search_CR_14d_10 | 1 | 58758 | 100% | 100% | 100% |
| search_CR_7d_5 | 1 | 84447 | 100% | 0% | 0% |
| search_CV_14d_3 | 1 | 204261 | 100% | 0% | 0% |
| search_CV_1d_1 | 1 | 72676 | 100% | 100% | 100% |
| search_LG_14d_1 | 1 | 67995 | 100% | 100% | 100% |
| search_LG_3d_10 | 1 | 206948 | 0% | 0% | 0% |
| search_RO_30d_25 | 1 | 42738 | 100% | 100% | 100% |
| search_RO_3d_8 | 1 | 72351 | 100% | 100% | 100% |
| search_kw_agentic_rl | 1 | 230674 | 0% | 0% | 0% |
| search_kw_llm | 1 | 186100 | 0% | 0% | 0% |
| search_kw_rag | 1 | 286778 | 0% | 0% | 0% |
| state_cache_before_dl | 1 | 167382 | 0% | 0% | 0% |
| state_cache_two | 1 | 43263 | 100% | 100% | 100% |
| state_dl_then_cache | 1 | 283282 | 0% | 0% | 0% |
| state_ref_cache_active | 1 | 52464 | 100% | 100% | 100% |
| state_ref_last_active | 1 | 43405 | 100% | 100% | 100% |
| state_ref_last_of_10 | 1 | 66552 | 100% | 0% | 0% |
| state_ref_ordinal_cn | 1 | 65779 | 100% | 0% | 0% |
| state_translate_active | 1 | 74309 | 100% | 100% | 100% |
| summary_ai5_tldr60 | 1 | 70621 | 100% | 0% | 0% |
| summary_cl5_bullet250 | 1 | 389457 | 100% | 0% | 0% |
| summary_cv5_structured120 | 1 | 284627 | 0% | 0% | 0% |
| summary_lg5_tldr120 | 1 | 317207 | 0% | 0% | 0% |
| summary_ro5_structured250 | 1 | 214731 | 0% | 0% | 0% |