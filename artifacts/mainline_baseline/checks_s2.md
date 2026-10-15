# S2 中央已有回归检查

日期：2026-10-14，Asia/Shanghai。变更：`simplify-sft-grpo-react-mainline`。

## 范围与安全边界

在所有 S2 消费者迁移和审查修复后，复用 S0 的30个安全测试模块，另加已存在的 `test_mainline_assets.py`，只执行一次中央回归。没有新增论文、快照、响应、token、轨迹或其他测试输入，没有执行生产生成器、build/backfill、模型训练、真实 API 或完整86任务回放。

执行环境与 [`check_inventory.md`](check_inventory.md) 相同：当前 `.venv/bin/python`，临时 `PYTHONPATH` 引用已有本机 pytest 9.0.2 等 uv 缓存，`DOTENV_DISABLED=1`、HF离线、memory store、禁止写bytecode与pytest cache，PDF/译文/图像/缓存根隔离到临时目录，临时 `sitecustomize.py` 阻断 socket 连接而不提供响应。原有 conftest 再次隔离各测试的产物根。

明确不执行新增 `test_mainline_behavior_baseline.py`；缺torch的 `TestTransformersClientResponse::test_adapts_local_generation_to_openai_response` 单项显式 deselect，不把它记录为通过。训练栈模块仅collection，不执行测试体。S0列明的所有资产缺口仍为本次未验证，而非待补实施任务。

## 中央回归实际命令

```sh
/bin/sh /tmp/aa-mainline-check-o5cVGB/run_checks.sh -m pytest -q -ra --tb=short \
  -p no:cacheprovider --basetemp=/tmp/aa-mainline-check-o5cVGB/pytest-s2 \
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
  AgenticArxiv/tests/test_mainline_assets.py
```

首次实际输出：`1 failed, 455 passed, 4 skipped, 1 deselected, 1 warning, 462 subtests passed in 1.61s`，pytest退出1。唯一失败是业务/接口回归，不归入资产豁免，已立即报告给主代理与运行时owner：

`test_tool_bootstrap.py::AgentsUseTheSharedBootstrapTest::test_no_agent_keeps_its_own_shared_try_block` 的 `agent='agents/agent_engine.py'` 子测试在第107行仍要求源码含 `register_all_tools`，新Agent通过 `require_all_tools("ReAct")` 使用共同工具bootstrap。该原断言需迁移到当前共享bootstrap入口，而不是删除工具注册检查或制造新输入。

主代理仅将原源码断言适配为实际 `require_all_tools("ReAct")` 调用，原有缺工具失败保护保留，没有扩写输入。随后只定向重跑该模块，不重复全组：

```sh
/bin/sh /tmp/aa-mainline-check-o5cVGB/run_checks.sh -m pytest -q -ra --tb=short \
  -p no:cacheprovider --basetemp=/tmp/aa-mainline-check-o5cVGB/pytest-s2-bootstrap \
  AgenticArxiv/tests/test_tool_bootstrap.py
```

实际结果：`7 passed, 2 subtests passed in 0.17s`，pytest退出0。首次失败记录保留；该唯一新增断言失败已关闭，不能把定向重跑的7项再次累加为新的覆盖范围。

4个原有跳过：3个 `test_analyze_curriculum.py` 的可选tensorboard项，1个 `test_rollout_audit.py` 的缺默认快照项。1个warning为既有 cache_status_tool 使用 Pydantic `dict()` 的弃用警告，不是失败。S0 中 GRPO 两个原方法及四个subtest的缺TRL失败在迁移后的原接口回归中未再出现；不能据此宣称真实TRL训练集成已验证。

## CLI 帮助实际结果

以下命令均沿相同安全wrapper执行，仅获取帮助，没有运行任务或初始化真实模型/API。

| 命令 | 结果 |
| --- | --- |
| `run_checks.sh -m AgenticArxiv.main --help` | 退出0，argparse usage |
| `run_checks.sh -m AgenticArxiv.rl.rollout --help` | 退出0，Fire NAME帮助 |
| `run_checks.sh -m AgenticArxiv.benchmark.run_benchmark --help` | 退出0，argparse usage |
| `run_checks.sh eval/badcase_replay.py --help` | 退出0，包含replay/run/capture帮助 |

S0 的 `main --help` 会直接执行任务，因此未执行；S2 的 main 已有真正的参数解析，本次安全帮助检查实际通过。CLI帮助和原CLI回归仅验证入口解析，不作为真实运行证据。

## 训练栈 collection

```sh
/bin/sh /tmp/aa-mainline-check-o5cVGB/run_checks.sh -m pytest --collect-only -q \
  -p no:cacheprovider --continue-on-collection-errors \
  AgenticArxiv/tests/test_canary.py AgenticArxiv/tests/test_training_guards.py
```

实际输出 `no tests collected, 2 errors in 0.17s`，退出1：Canary仍缺torch；training_guards导入train_grpo时仍缺datasets。两项与S0同因，未误写为重构新增失败，也未伪造依赖或模型资产取得通过。

## 当前结论

已执行消费者、解析/评分、会话/隔离、原token/mask接口、历史坏例及10工具/86任务/切分定义回归，结论仅覆盖上面实际检查的现有输入。首轮唯一新增的共享bootstrap源码断言失败已通过接口适配及定向回归关闭；当前范围内没有未解决的新增业务失败。未重复全组，不宣称末态全套重新执行。

真实tokenizer/template、实际checkpoint加载、生产数据生成、完整86任务运行、训练、真实翻译/API及模型性能仍未验证，不作为本次遗留实施任务，也不据已有mock宣称已验证。

## 原始输出

- 中央回归：`/tmp/aa-mainline-check-o5cVGB/s2.log`，SHA256 `8dc5fa9d597d871106788ba00bcd211b8d3f15b30fc49cb28305c57b8bc5865c`。
- bootstrap关闭回归：`/tmp/aa-mainline-check-o5cVGB/s2-bootstrap.log`，SHA256 `d6c5776abe56e1f8d8e7a5a50b5d8eebe03a627dc65d6fe686ef9ec5cf3cf8e3`。
- 训练栈collection：`/tmp/aa-mainline-check-o5cVGB/s2-training-collection.log`，SHA256 `edd5172fb64131a596bc7d10f12f328fd5a09529ed33bfe9ef3849672c293b4a`。
- 四个帮助日志：临时目录内 `s2-AgenticArxiv.main.help.log`、`s2-AgenticArxiv.rl.rollout.help.log`、`s2-AgenticArxiv.benchmark.run_benchmark.help.log`、`s2-badcase.help.log`。

以上仅为检查日志，不是新增研究数据或行为基线。完整范围与首轮失败已原样保存，不以再次全量执行掩盖起始结果。
