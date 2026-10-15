# 本地实验管理命令与配置

本页说明基础安装即可使用的 `agentic-arxiv experiment` 接口。管理命令不导入训练器、不读取 `.env`，不管理远程 SSH，不构建或自动拉取镜像，不自动生成研究输入或串联训练阶段。执行下面涉及 Docker 的操作前，仍须单独确认目标服务器与 GPU 授权；本次仅完成本地工程检查。

## 命令

所有叶命令均接受 `--profile PATH`（必填）、`--workspace PATH` 和 `--json`。相对配置文件路径按显式工作区解释，profile 内资源路径按 profile 的 `workspace` 解释。帮助无需提供实际文件，不连接容器服务。

```console
agentic-arxiv experiment --help
agentic-arxiv experiment check recipes/M1.json --profile profiles/server.json --json
agentic-arxiv experiment register M1 --model inputs/sft/final --profile profiles/server.json --json
agentic-arxiv experiment start recipes/M3.json --profile profiles/server.json --gpus GPU-EXPLICIT-UUID --seed 42 --name m3-main --json
agentic-arxiv experiment list --profile profiles/server.json --json
agentic-arxiv experiment status RUN_ID --profile profiles/server.json --json
agentic-arxiv experiment logs RUN_ID --profile profiles/server.json --rank 0 --tail 100 --json
agentic-arxiv experiment stop RUN_ID --profile profiles/server.json --json
agentic-arxiv experiment stop RUN_ID --profile profiles/server.json --force --json
agentic-arxiv experiment resume RUN_ID --checkpoint /explicit/attempt/output/checkpoint-50 --profile profiles/server.json --json
agentic-arxiv experiment evaluate /explicit/base --kind benchmark --evaluation-config benchmark.json --gpus GPU-EXPLICIT-UUID --profile profiles/server.json --json
agentic-arxiv experiment evaluate /explicit/final --kind stage --evaluation-config stage.json --gpus GPU-EXPLICIT-UUID --profile profiles/server.json --json
```

示例中的路径、运行标识和 GPU 都是待人工填写的参数，不是可直接启动的正式配方。`check` 不建立运行；`start` 只执行一个所选作业；M0 只评测。阶段验证重试不会重新训练。普通停止仅请求安全边界保存，长时间等待不会升级为强杀；只有 `--force` 才请求强制中断，不能承诺保存最新进度。

`evaluate` 的引用还可写成 `registration:ID`、`run:RUN_ID:ATTEMPT_ID:final`，或用单引号包裹完整 JSON 引用对象，以包含精确 `verification_ref`。模型与 checkpoint 的相对路径使用 profile 工作区；`--metadata` 和 `--evaluation-config` 等配置文件路径使用命令显式工作区。

## 三层严格配置

配置仅使用 JSON，`schema_version` 必须是整数 `1`。未知字段、重复 JSON 键、错误类型、非有限数和冲突设置均拒绝，不接受任意脚本或命令字符串。现有独立训练配置格式保持不变。

服务器 profile 必填字段为 `schema_version`、`workspace`、`run_root`、`input_roots`、`cache_root`、`image`、`uid`、`gid`、`gpus`、`shm_size`，另可设置 `min_free_bytes`。UID/GID 必须是正整数；镜像必须是已核验的 `repository@sha256:<64位摘要>`；GPU 允许范围必须显式且不重复。profile 不能写入学习率、种子或其他算法参数。

recipe 必填字段为 `schema_version`、`group`、`stage`、`training_config`、`inputs`、`training_overrides`、`checkpoint`、`evaluation`、`budget`，可补充 `name`、`task_pool`、`filter_policy`。组别对应阶段固定为 M0/evaluate、M1/sft、M2/dpo、M3/grpo、M4/grpo、C1/grpo。`training_config: null` 采用包内既有主线默认配置，不创造另一套算法默认值。M0 的训练配置和 checkpoint 都必须为 `null`，且训练覆盖为空。

训练覆盖使用 `{"defaults": {}, "trainer_parameters": {}}`。最终优先级为既有训练配置、recipe 训练覆盖、显式本次覆盖。训练字段的完整集合和类型继续由原 `stage_config` 校验。管理保存策略通过 `checkpoint.save_steps` 和 `checkpoint.save_total_limit` 指定，均为正整数，不改变裸训练命令默认值；重复位置提供冲突策略会拒绝。

显式种子用 `--seed`。其他允许的现有训练字段可逐项覆盖，例如：

```console
agentic-arxiv experiment check recipes/M1.json --profile profiles/server.json --set defaults.lr=0.0001 --set trainer_parameters.logging_steps=5 --json
```

`--set` 的右侧是 JSON，不是字符串猜测。输入模型、数据、输出目录、恢复来源等不能通过训练覆盖绕过管理边界；`--seed` 与重复的 `--set defaults.seed=...` 也会拒绝。最终配置记录逐字段来源、本次覆盖和指纹，不修改 recipe 或旧产物。

`inputs` 支持 `model`、`base_model`、`data`、`data_manifest`、`snapshot`、`split`、`source_sft`、`source_dpo`、`preference_cost`。引用是明确路径字符串，或以下严格对象之一：

```json
{"kind": "path", "path": "/explicit/model"}
```

```json
{"kind": "registration", "registration_id": "EXPLICIT_ID"}
```

```json
{"kind": "run", "run_id": "RUN_ID", "attempt_id": "ATTEMPT_ID", "artifact": "final"}
```

不接受 `latest`。模型最终产物与训练恢复点是不同资格，登记不会升级旧 checkpoint。模板中的 `null` 只表示待人工补齐，不能替代真实路径或预算。

若已有 final 的阶段验证被中断或失败，可显式重验，并在任一引用对象加上 `verification_ref: {"run_id": "EVALUATION_RUN", "attempt_id": "EVALUATION_ATTEMPT"}` 精确引用通过的独立阶段验证证据；管理层核对其模型身份，不自动扫描“最近一次通过”。登记时也可在 `--metadata` 指定的 JSON 内提供这一引用。旧模型目录仍然只读。

`budget` 显式保存 `max_steps`、可选 `epochs`、`comparison_group`（`M4` 或 `null`）、`accounting: "gpu_hours"`、`strict_match: false` 和 `costs`。训练预算二选一：按步数时 `max_steps` 为正整数、`epochs` 为 `null` 或省略，并与最终 `training.defaults.max_steps` 一致；按轮数时 `max_steps: null`、`epochs` 为正整数，且最终训练配置保持 `max_steps: -1`、`epochs` 与预算一致。不能同时设置两种预算，不能靠显式参数使训练配置偏离预算。两者均为 `null` 仅供模板填写，正式启动预检仍拒绝。旧的按步数 recipe 无需补充 `epochs` 字段。

按轮数训练不在控制端猜测步数。训练器按实际数据、批量与原有规则核定 `state.max_steps`，在首次优化之前绑定并保存核定记录；完整 checkpoint 保存这一实际总步数。恢复沿用原 `max_steps: -1` 与同一 `epochs`，并重新核对核定总步数一致，不将剩余轮数另发为一份新预算。

成本必须分别列出 `preference_generation_gpu_hours`、`dpo_gpu_hours`、`grpo_gpu_hours`，未知用 `null`，不能用零冒充未知。C1 另须在 `comparison_reference` 精确引用 M4 配方/成本报告路径或明确运行与尝试，不要求先训练 M4。C1 不会因增加 epoch 或缺失旧成本自动标为严格等预算。

`evaluation` 可用 `{"config": "benchmark.json", "kind": "benchmark"}` 引用固定评测配置，也可内联受支持评测参数，不能两者混用。模型和输出位置由管理层决定，不接受评测配置覆盖。`task_pool` 和 `filter_policy` 使用 `{"path": null, "fingerprint": null}` 模板，正式 GRPO 配方必须填写可核对的同源输入。

GRPO 的 `inputs.split` 明确引用切分文件，同时在训练配置或 `training_overrides.defaults.split` 指定同一文件的 `FILE:train` 或 `FILE:rl_train`。管理层不猜切分名称，也不允许使用留出测试切分。两处文件身份必须一致；`split` 覆盖只用于明确这一训练切分，不豁免输入审计。

## JSON 结果和退出码

`--json` 输出单个版本化对象，顶层字段固定为 `schema_version`、`operation`、`ok`、`exit_code`、`data`、`error`、`completion_scope`。最后一项始终为 `management_operation_only`：返回零只证明这次管理操作成功，后台启动成功不能证明训练预算完成、阶段验证通过或真实恢复可用。训练和验证结论必须读取独立状态维度与证据。

退出码：`0` 管理操作成功；`2` 配置、输入或操作校验失败；`3` 资源或活动尝试冲突；`4` 所需文件、服务或依赖不可访问；`5` 启动或状态无法确认。无法确认时保留历史证据和持久资源声明，不自动重试、换卡或将未知标成已停止。语法错误仍由统一参数解析器返回 `2` 并显示帮助诊断，不承诺在参数未能解析时输出 JSON。

工程测试只能证明参数、配置、控制边界和既有入口回归；镜像构建、GPU 容器执行、真实三阶段单卡/四卡恢复和模型质量评测仍需要独立证据。
