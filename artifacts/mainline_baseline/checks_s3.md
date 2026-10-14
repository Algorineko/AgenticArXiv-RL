# S3 冻结源码后的中央安全回归

日期：2026-10-14，Asia/Shanghai。变更：`simplify-sft-grpo-react-mainline`。

## 范围、输入与安全边界

主代理通知最后usage闭环完成、相关源码冻结后，执行一次中央回归：沿 `checks_s2.md` 的31个现有核心模块，追加7个既有纯转换/消息helper用例，不再扩测或重复执行。

追加范围只有 `test_generate_sft_data.py::SplitLeakageGuardTest` 的4项切分选择检查、`::DataLineageTest` 的1项原history到消息转换检查，以及 `test_augment_sft_data.py::PromptBoundaryTest` 的2项字符串边界检查。没有选择会调用专家执行的 `ExpertFailFastTest`，没有执行生成器主函数、生产扩增/混合、build/backfill、完整任务回放或数据导出。

只读检查相关diff确认：原CORRECT/BARE_FINISH等固定响应、原101/102采样ID和原固定模板 `[1, 2]` 继续用于已有单元测试；消息前缀的mock编码仅做接口适配，DataLineage仅从原history计算新字段和新增断言。默认策略输入从退出的原路由测试迁入，没有增加替代模型响应。**这些existing mocks只用于接口断言，不是新增行为基线，也不是实际模型、真实tokenizer或环境执行证据。** 新增 `test_mainline_behavior_baseline.py` 未执行。

使用S0/S2同一安全wrapper `/tmp/aa-mainline-check-o5cVGB/run_checks.sh`：当前 `.venv/bin/python`，已有本机uv缓存pytest9.0.2等依赖，禁止联网安装；`DOTENV_DISABLED=1`、HF离线、memory store、socket禁止连接、bytecode/cache禁写、临时产物根和原conftest隔离。未读取凭据或真实 `.env`，未操作真实缓存；安全wrapper不提供网络或模型模拟响应。

缺torch的原单项 `TestTransformersClientResponse::test_adapts_local_generation_to_openai_response` 继续显式deselect。训练依赖与训练模块导入没有改变，不重复S2已确认的training collection。

## 实际命令

```sh
/bin/sh /tmp/aa-mainline-check-o5cVGB/run_checks.sh -m pytest -q -ra --tb=short \
  -p no:cacheprovider --basetemp=/tmp/aa-mainline-check-o5cVGB/pytest-s3 \
  --deselect=AgenticArxiv/tests/test_llm_request_params.py::TestTransformersClientResponse::test_adapts_local_generation_to_openai_response \
  AgenticArxiv/tests/test_splits.py \
  AgenticArxiv/tests/test_task_spec.py \
  AgenticArxiv/tests/test_reference_resolution.py \
  AgenticArxiv/tests/test_tool_sequence_strict.py \
  AgenticArxiv/tests/test_cost_metrics.py \
  AgenticArxiv/tests/test_reliability.py \
  AgenticArxiv/tests/test_false_finish.py \
  AgenticArxiv/tests/test_constraint_tasks.py \
  AgenticArxiv/tests/test_trajectory.py \
  AgenticArxiv/tests/test_reward_discrimination.py \
  AgenticArxiv/tests/test_multigranular_reward.py \
  AgenticArxiv/tests/test_observability.py \
  AgenticArxiv/tests/test_analyze_curriculum.py \
  AgenticArxiv/tests/test_cli_entrypoints.py \
  AgenticArxiv/tests/test_tool_bootstrap.py \
  AgenticArxiv/tests/test_agent_session_leak.py \
  AgenticArxiv/tests/test_multiturn_env.py \
  AgenticArxiv/tests/test_rollout_sandbox.py \
  AgenticArxiv/tests/test_terminal_action.py \
  AgenticArxiv/tests/test_benchmark_offline.py \
  AgenticArxiv/tests/test_rollout.py \
  AgenticArxiv/tests/test_api_rollout_model.py \
  AgenticArxiv/tests/test_keyword_search.py \
  AgenticArxiv/tests/test_llm_client_retry.py \
  AgenticArxiv/tests/test_llm_request_params.py \
  AgenticArxiv/tests/test_pdf_downloader.py \
  AgenticArxiv/tests/test_grpo_reward.py \
  AgenticArxiv/tests/test_rollout_audit.py \
  AgenticArxiv/tests/test_grpo_signal_split.py \
  AgenticArxiv/tests/test_badcases.py \
  AgenticArxiv/tests/test_mainline_assets.py \
  AgenticArxiv/tests/test_generate_sft_data.py::SplitLeakageGuardTest \
  AgenticArxiv/tests/test_generate_sft_data.py::DataLineageTest \
  AgenticArxiv/tests/test_augment_sft_data.py::PromptBoundaryTest
```

## 首次实际结果

```text
462 passed, 4 skipped, 1 deselected, 1 warning, 470 subtests passed in 1.12s
PYTEST_EXIT=0
```

首次全部可运行项通过，没有新增业务失败，不需要修复或定向重跑。462项是本轮结果，不能与S0/S2重复累计为更大覆盖。

4个原有skip：`test_analyze_curriculum.py` 第147、162、125行的3个可选tensorboard用例，以及 `test_rollout_audit.py:235` 的缺默认离线快照用例。1个deselect是上述缺torch的真实本地响应适配项，未记为通过。1个warning是 `cache_status_tool.py:40` 使用Pydantic `dict()` 的既有弃用警告，与S2相同。

原CLI用例包含从仓库根及包目录解析benchmark/rescore等帮助，本轮实际通过。main、rollout、badcase的直接帮助已在 `checks_s2.md` 记录；本轮没有额外重复这些独立命令或执行API来扩展结论。

## 本次结论与未验证范围

本轮在当前依赖与现有输入范围内确认：资产定义、评分/旧记录投影、解析/终止、工具注册、会话/隔离、普通/API薄适配、原GRPO token/mask回归和共同消息/转换helper没有未解决失败。它不证明所有未执行模块均可运行，不构成真实tokenizer前缀、模型加载或训练集成证据。

保留S0/S2已有依赖限制：Canary模块缺torch，training_guards经train_grpo导入缺datasets；TRL、transformers、tokenizers等训练栈不在当前 `.venv`。未伪造这些包或版本，没有重复失败collection来获取新结果。

真实tokenizer/template、真实checkpoint加载、默认完整快照与86任务执行、SFT数据生成/扩增/混合产出、实际训练、真实外部API/翻译/VLM和模型性能均未验证。它们不是本次待补实施任务，也不能被现有mock、历史报告或本次绿灯替代为已通过。

## 证据索引

- 原始输出：`/tmp/aa-mainline-check-o5cVGB/s3.log`，SHA256 `faf0eff00b27bca4e83fa3d3c91af47690d6d958b9a6b5f6e32f11655bdd80b1`，仅检查日志，不是轨迹、快照或数据集。
- 执行wrapper与网络禁用代码：临时目录内 `run_checks.sh`、`sitecustomize.py`，没有写入业务源码或作为行为fixture。
- 上游范围与已有依赖结果：`check_inventory.md`、`checks_s2.md`。

执行结束时的关键源码指纹（用于定位本轮实际检查对象，不代表历史实验指纹）：

| 文件 | SHA256 |
| --- | --- |
| `AgenticArxiv/agents/react_contract.py` | `51ae668ddeaed6b518e1939c2e18f62f8afb69ee9bd1dd168505c24ffd78782a` |
| `AgenticArxiv/agents/agent_engine.py` | `36b9e95f9e126565ec7f49516339adf054313c4dc14add5d92dd50f0a98b6e19` |
| `AgenticArxiv/rl/grpo_reward.py` | `0ac4a4e080927019770bf191b869a03a53044a230b0c1aaca0974fab3a47aeb8` |
| `scripts/generate_sft_data.py` | `17529886f75d64e366d4cb27098038eac8ff2cc998a4abb0c740b0e88500ad0d` |
| `AgenticArxiv/tests/test_generate_sft_data.py` | `40268a9c039d1942b1ba1feca6d417258a301c8a1c2be490cf17744c2a3ba2a7` |
| `AgenticArxiv/tests/test_augment_sft_data.py` | `854886f75b9c13ce9d273ce87804c441af87daf041ea619c29228dc66b7cc4a4` |

全部finite command已经结束；本记录没有再启动训练、数据产出或后台任务。
