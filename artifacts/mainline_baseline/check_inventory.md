# S0 当前可运行检查清单与起始结果

日期：2026-10-14，Asia/Shanghai。变更：`simplify-sft-grpo-react-mainline`。

## 本次边界

本记录按修订后的规划完成任务 1.3：只盘点并执行当前材料支持的检查，不要求补齐外部资产。`input_inventory.json` 保留原盘点的路径与哈希，但其“缺资产阻塞工程验收”的旧结论已被本次修订取消；快照、真实 tokenizer、模型、数据正文和完整历史训练记录缺失仅限制验证结论，不是本次待补任务。

没有改业务代码、测试输入或任务勾选。没有运行训练、真实 API、生成器、完整 86 任务回放，没有新增响应、token、轨迹或行为 fixture。既有单元测试仅复用其原有输入、mock 和临时输出；不把它们作为真实模型或工具执行证据。

规划来源：`openspec/changes/simplify-sft-grpo-react-mainline/design.md`（读取时 SHA256 `a523dae3aad3c04a5ad0bee1fd6b6e3af782a3ae4270806641a8ac19031d1564`）及 `tasks.md`（读取时 SHA256 `4644a1d74493393262a2a4dbd57539f42a139b86d022793f1200fa38a4b08553`）。所有相对路径以当前仓库根 `/Users/arkalin/Documents/MyDocuments/agent_rl/AgenticArXiv-RL` 为基准。

## 依赖与安全执行环境

起始命令 `python3 -m pytest --version`、`.venv/bin/python -m pytest --version` 和应用 bundled Python 的 `-m pytest --version` 均失败，原因都是没有安装 pytest。没有联网安装或修改 `.venv`。发现已有本机 uv 缓存后，仅通过临时 `PYTHONPATH` 使用 pytest 9.0.2、pluggy 1.6.0、iniconfig 2.3.0、packaging 26.0、Pygments 2.20.0；缓存 pytest 的 `--version` 实际通过。

本地 `.venv` 可用依赖：arxiv 4.0.1、requests 2.34.2、python-dotenv 1.2.4、loguru 0.7.3、pydantic 2.14.1、fire 0.7.1、PyMuPDF 1.28.2、Pillow 12.3.0、fastapi 0.143.0、sqlalchemy 2.1.4、pymysql 1.2.3。torch、transformers、tokenizers、trl、datasets、accelerate、peft、pytest、pdf2zh 不在该 `.venv`；这是当前检查环境，不代表历史训练依赖版本。

所有下列检查使用 `/tmp/aa-mainline-check-o5cVGB/run_checks.sh`，以 `.venv/bin/python` 执行，设置：

```sh
PYTHONDONTWRITEBYTECODE=1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
DOTENV_DISABLED=1
STORE_BACKEND=memory
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

`PYTHONPATH` 包含仓库 `AgenticArxiv`、以下已有缓存路径及临时安全目录：

```text
/Users/arkalin/.cache/uv/archive-v0/ZSMKGLRjSWMVC3xIE2hyC
/Users/arkalin/.cache/uv/archive-v0/kpZnhoZGMERwdY5HDJkqV
/Users/arkalin/.cache/uv/archive-v0/jBkcDA81GB-wu1hkX8MI6
/Users/arkalin/.cache/uv/archive-v0/ffkSdlLABDHFfeY5aB3mj
/Users/arkalin/.cache/uv/archive-v0/LfiR8QOc75h0_xnNcLneQ
```

临时 `sitecustomize.py` 将 `socket.socket.connect`、`connect_ex`、`create_connection` 替换为明确抛错的禁止连接函数；它只阻断网络，不提供模拟响应。所有 PDF/译文/图像/缓存环境变量指向 `/tmp/aa-mainline-check-o5cVGB`，pytest 的既有 `conftest.py` 进一步将测试产物根绑定 `tempfile.mkdtemp`，每组传入独立 `--basetemp`，并用 `-p no:cacheprovider` 禁止仓库 pytest cache。子进程帮助检查继承这些环境设置。不读取真实 `.env`，不操作真实产物根。

## 当前测试盘点

- `git ls-files AgenticArxiv/tests/test*.py` 有 62 个已跟踪测试模块；AST 计数为 824 个 `test_` 函数定义（含因依赖缺失不能收集的定义），不是实际执行数。
- 现有 `test_llm_pdf_one_step.py` 和 `test_mock_env.py` 无可收集的 `test_` 函数，是手动脚本。
- 未提交的 `test_mainline_assets.py` 与 `test_mainline_behavior_baseline.py` 均未纳入本次执行或 collection，避免把新建立的行为材料纳入修订后的基线。资产代理单独执行资产核对，不属于下面的总计。
- 实际 collection 命令：

```sh
/bin/sh /tmp/aa-mainline-check-o5cVGB/run_checks.sh -m pytest AgenticArxiv/tests \
  --collect-only --continue-on-collection-errors -q -p no:cacheprovider \
  --ignore=AgenticArxiv/tests/test_mainline_behavior_baseline.py \
  --ignore=AgenticArxiv/tests/test_mainline_assets.py
```

结果：706 项收集，4 个模块收集错误。`test_canary.py`、`test_opd.py`、`test_opd_multiturn.py` 缺 torch；`test_training_guards.py` 导入 `rl.train_grpo` 时缺 datasets。没有执行测试体。首次 collection 输出经 `tee` 保存，外层管道退出 0，不应将其误写为 pytest 成功。

## 实际执行的现有分组

每组命令前缀均为 `/bin/sh /tmp/aa-mainline-check-o5cVGB/run_checks.sh -m pytest -q -p no:cacheprovider --basetemp=<独立临时目录>`，随后是下面列明的原测试文件。

### 核心评分、任务与切分组

```text
AgenticArxiv/tests/test_splits.py
AgenticArxiv/tests/test_task_spec.py
AgenticArxiv/tests/test_reference_resolution.py
AgenticArxiv/tests/test_tool_sequence_strict.py
AgenticArxiv/tests/test_cost_metrics.py
AgenticArxiv/tests/test_reliability.py
AgenticArxiv/tests/test_false_finish.py
AgenticArxiv/tests/test_constraint_tasks.py
AgenticArxiv/tests/test_trajectory.py
AgenticArxiv/tests/test_reward_discrimination.py
AgenticArxiv/tests/test_multigranular_reward.py
AgenticArxiv/tests/test_observability.py
AgenticArxiv/tests/test_analyze_curriculum.py
```

`--basetemp=/tmp/aa-mainline-check-o5cVGB/pytest-core`。原输出：`255 passed, 3 skipped, 416 subtests passed in 0.45s`。3 个跳过来自 `test_analyze_curriculum.py` 的可选 tensorboard 条件。pytest 已结束；其后 shell 尝试写 zsh 只读变量 `status`，外层 shell 退出 1，是记录退出码的编排错误，不是用例失败，没有重复执行来伪造绿灯。

### 入口、会话、离线环境与客户端组

```text
AgenticArxiv/tests/test_cli_entrypoints.py
AgenticArxiv/tests/test_tool_bootstrap.py
AgenticArxiv/tests/test_agent_session_leak.py
AgenticArxiv/tests/test_multiturn_env.py
AgenticArxiv/tests/test_rollout_sandbox.py
AgenticArxiv/tests/test_terminal_action.py
AgenticArxiv/tests/test_benchmark_offline.py
AgenticArxiv/tests/test_rollout.py
AgenticArxiv/tests/test_api_rollout_model.py
AgenticArxiv/tests/test_keyword_search.py
AgenticArxiv/tests/test_llm_client_retry.py
AgenticArxiv/tests/test_llm_request_params.py
AgenticArxiv/tests/test_pdf_downloader.py
```

`--basetemp=/tmp/aa-mainline-check-o5cVGB/pytest-entry`。原输出：`1 failed, 115 passed, 33 subtests passed in 0.92s`，pytest 退出 1。唯一失败：`test_llm_request_params.py::TestTransformersClientResponse::test_adapts_local_generation_to_openai_response` 在 `import torch` 时失败，没有进入真实本地生成。

### GRPO 接口、审计、信号选择与既有坏例组

```text
AgenticArxiv/tests/test_grpo_reward.py
AgenticArxiv/tests/test_rollout_audit.py
AgenticArxiv/tests/test_grpo_signal_split.py
AgenticArxiv/tests/test_badcases.py
```

该组额外带 `-ra`，`--basetemp=/tmp/aa-mainline-check-o5cVGB/pytest-signal`。原输出：`6 failed, 79 passed, 1 skipped, 12 subtests passed in 0.29s`，pytest 退出 1。失败均为既有代码调用 `require_rollout_func_support()` 获取 TRL 版本时没有包元数据：

- `test_grpo_reward.py::TestMultiTurnTrajectory::test_custom_rollout_masks_environment_tokens`。
- `test_grpo_reward.py::MultiTurnRolloutCardinalityTest::test_generation_batch_is_not_inflated`。
- `test_grpo_reward.py::MultiTurnRolloutCardinalityTest::test_output_length_matches_input_length` 的 4 个原有 subtest（pytest 总结按 4 个 subfail 计入 6 failed）。

跳过：`test_rollout_audit.py` 的快照依赖用例，原因 `offline snapshot not available`。没有补造快照或伪造 TRL 版本让检查继续。

合计三组：449 passed，4 skipped，7 failed（含上述 4 个 subfail），461 subtests passed。已执行分组没有发现业务断言失败；依赖失败已经作为起始状态保留，不表示全套通过。

## 实际 CLI 与导入检查

| 命令或既有检查 | 起始结果 | 证据边界 |
| --- | --- | --- |
| `test_cli_entrypoints.py` 的两个原有测试 | 通过 | `run_benchmark`、`run_baselines`、`rescore_traces` 从仓库根和包目录的 6 次 `--help` 均可运行；没有执行 benchmark |
| `run_checks.sh -m AgenticArxiv.rl.train_sft --help` | 退出 1，缺 trl | 帮助在导入阶段失败，不启动训练 |
| `run_checks.sh -m AgenticArxiv.rl.train_grpo --help` | 退出 1，缺 datasets | 帮助在导入阶段失败，不启动训练 |
| `run_checks.sh -c 'import AgenticArxiv.main; import rl.rollout; import benchmark.runner; import rl.grpo_reward; print("main/rollout/benchmark.runner/grpo_reward imports passed")'` | 退出 0 | 仅导入；不调用 main 或 Agent，不生成数据 |

不能直接以 `python -m AgenticArxiv.main --help` 作为安全帮助检查：该文件没有参数解析，`__main__` 会直接执行在线任务。本次仅检查导入。

## 后续可运行范围与排除规则

上述通过的用例构成当前依赖下已实际验证的可运行起始范围。后续回归复用同样分组与输入；按保留接口需要适配原断言，不扩写输入。依赖失败的方法和上述 4 个收集失败模块单独报告；无需取得外部资产或安装训练栈来完成本次工程验收。其他 706 项中未执行的用例没有自动获得“通过”结论。

明确不执行：

- `test_agent.py::test_prompt_generation` 会直接调用真实 arXiv 工具，未标 skip。本次只 collection。
- `test_modules.py::test_arxiv`、`test_modules.py::test_llm`、`test_tool_registry.py::test_execute_tool` 是有 skip 标记的网络/凭据手动检查；不通过手动脚本绕过 skip。`test_llm_pdf_one_step.py::run_one_step` 会访问 LLM API 并执行 PDF 工具，整个脚本不运行。
- `test_mock_env.py::main` 依赖缺失的真实默认快照，不运行，也不运行其说明里的联网 `build_snapshot`。
- 专家生成/参数化生成、T5 backfill、新快照构建与数据扩增/混合的产出流程不运行。相关模块可以源码核对或 collection；本次没有执行 `test_generate_sft_data.py`、`test_t5_parametric_sft.py`、`test_t5_snapshot_backfill.py`、`test_build_snapshot.py` 的测试体来补行为数据。
- 所有真实 `Trainer.train()`、训练 CLI 主函数、模型 `from_pretrained` 和 `generate` 集成都不运行。源码检索未发现本次所选用例直接调用真实 `Trainer.train()`；`test_canary.py` 中的 `model.train()` 只是切换训练模式，不能误称已训练，但该模块仍因 torch 不可用未执行。
- `test_mainline_behavior_baseline.py` 明确排除，未用其新增 token 或轨迹证明修订后的契约。

## 本次未验证的资产依赖检查

| 检查 | 原因 | 本次处理 |
| --- | --- | --- |
| 真实 tokenizer/template 的 token 前缀、长度与追加边界 | 当前仓库无资产，当前 `.venv` 无 transformers/tokenizers | 不补资产；只保留实现及既有替身回归证据，不宣称真实集成通过 |
| 默认完整快照离线执行、86 任务完整执行 | `data/mock_arxiv_snapshot.json` 缺失，两个局部测试快照不等于完整输入 | 不重建、不下载；仅工具/任务/切分定义核对和已有回归 |
| 实际 SFT JSONL 正文与训练 manifest | 正文及完整训练 manifest 缺失，12 个历史 SFT manifest 存在 | 保留原资产索引；不生成新正文、不重建历史记录 |
| 真实 SFT 到 GRPO checkpoint 交接 | 本地权重缺失且训练栈未安装 | 原有元数据/错误分支用例可运行范围及源码核对；不训练取得模型 |
| 真实外部 LLM、翻译与 VLM 后端 | 无授权真实运行，pdf2zh/模型依赖不齐 | 不联网、不启动真实翻译，不把 mock 当真实能力证据 |

这些项目不是本次未完成实施任务，不要求补资产；结果始终标为未验证而不是通过。未来新增的导入/接口/业务断言失败必须与这里已有的缺依赖失败区分，不能用资产豁免掩盖重构回归。

## 原始输出索引

以下临时日志是此次有限检查的原始输出，不是轨迹或数据集；临时目录不属于需保留的研究资产，关键结果已完整摘要在本文。

| 临时文件 | SHA256 |
| --- | --- |
| `/tmp/aa-mainline-check-o5cVGB/collection.log` | `cdd846194da7567b2449bce2d55def1990f8a270b199185efccf7b02ddfd96b4` |
| `/tmp/aa-mainline-check-o5cVGB/core.log` | `9a70a73f05d00ed6f8d3fc22960f6951baab601d91ae15661f9e73f20efdf968` |
| `/tmp/aa-mainline-check-o5cVGB/entry.log` | `cbaf39a9adaffbad218242bc8a75a2b52dbed8fb5ddce1b184c3e505b9a11837` |
| `/tmp/aa-mainline-check-o5cVGB/signal.log` | `8e3b01d6a1cb3f7a5127314088ee8406a59beb59016317084fb4c32db3192c7b` |

本记录已经提供实际命令、当前结果与未验证范围，满足修订后的 1.3。其他代理执行的核心、splits 和资产检查没有混入上述计数，避免重复累计。
