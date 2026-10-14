# Design

## Context

动机及能力边界见 `proposal.md`。本设计适用于现有仓库，不引入新的 Agent 框架。探索时的源码基线为 `774846e205e79d9054a67b144325fa436fa87b8b`；实施开始时还须记录实际起点及用户已有改动。

当前可核实的差异如下：

| 位置 | 现状及设计影响 |
| --- | --- |
| `rl/train_sft.py`、`rl/train_grpo.py` | 已能直接 SFT 到 GRPO，但默认输出/输入未对齐，GRPO 仍默认读 DPO 产物、使用 8 条冒烟任务和 4 轮预算。 |
| `agents/agent_engine.py`、`agents/base_agent.py` | 解析器有猜工具和文本降级，解析失败与 FINISH 共用 `None`；基类同时处理路由、状态、翻译队列和服务日志。 |
| `rl/grpo_reward.py` | 有独立严格解析、多轮采样、原始 token 保留和环境 mask；也保留单轮合成轨迹与缺失轨迹降级。 |
| `rl/multiturn_env.py`、`rl/env.py` | 既有每轨迹状态与回放能力，但与普通 Agent 的引用、fallback、翻译和缓存处理不一致。 |
| `scripts/generate_sft_data.py` | 专家执行再次实现工具状态处理，按每一步重建 user prompt，不能只改训练入口而忽略数据生成。 |
| `benchmark/semantic_oracle.py` | 使用多轮环境执行标准步骤推导预期论文对象，是必须迁移但不能暴露给策略的消费者。 |
| `rl/canary.py`、`rl/stage_verifier.py` | 仍存在单轮生成后合成轨迹的质量检查路径，不能代表真实多轮能力。 |
| `models/store.py`、`rl/sandbox.py` | 存在进程级 store 切换和恢复共享产物根的能力；测试已用临时路径避免误删真实缓存，正式环境仍需明确所有权。 |

目前主规格为空。已核实的仓库材料包括 10 个注册工具、86 条扩展任务、切分、既有奖励、测试、数据 manifest 和 `artifacts/` 下的历史报告与轨迹。默认快照、实际 SFT JSONL、发布 checkpoint、完整训练 manifest 和真实 tokenizer 资产不齐。本次仅依赖当前仓库完成工程重构，不补齐这些研究输入，不运行训练或生成新数据；材料缺失只限制可得出的验证结论，不成为额外实施任务。工作区已有但未提交的代码和产物属于现有工作，不能因调整计划而删除、覆盖或当作本次新执行的证据。

## Goals / Non-Goals

**Goals:**

- 使工具执行和状态转移不依赖训练器、路由器或 Web 服务；训练适配器只能消费共同契约，不能各自维护另一套世界状态。
- 保留当前训练器和模型加载体系，只抽取有重复且有明确契约的逻辑，避免重建插件系统或通用工作流引擎。
- 让源码中的配置传递、运行时语义、上下文格式与已有数据来源分别可追踪，能解释每一阶段的差异和证据限制。
- 保留研究资产和后续运行能力，且让正式入口在缺少模型、快照、工具或关键元数据时尽早失败；本次不要求实际具备这些输入。

**Non-Goals:**

- 不切换为原始 GRPO、其他损失或新的奖励方案，不以新算法性能目标作为本次精简的完成条件。
- 不减少工具或任务，不改变业务参数、任务标准答案、既有留出集成员或原始研究产物。
- 不新增原生 tool-calling、Web 替代服务、多 Agent 编排、模型部署平台或另一套训练框架。
- 不保证旧 checkpoint 在新上下文中的行为不变，也不把跨协议分数变化解释为算法收益。
- 本次不运行 SFT/GRPO 训练，包括少量步数的训练冒烟测试；不执行依赖新 checkpoint 的信号审计或 Base/SFT/GRPO 正式对照。这些能力保留供后续单独运行，但训练效果、收敛性和模型性能不属于本次验收结论。
- 不补齐仓库外历史配方证据、快照、模型、数据或 tokenizer，不要求完整 86 条任务执行或真实 tokenizer 集成验证，不承诺历史实验等价。
- 不新增合成论文、快照、模型响应、轨迹或测试/训练数据集，不为缺失输入生成行为 fixture，也不执行 SFT 数据重生成。可以复用既有测试输入和 mock、只读分析已有记录以及计算清单/哈希，但不扩充造数或将 mock 当成真实运行证据。

## Decisions

### 1. Preserve Repository Configuration Semantics

选择“保全当前源码主线路径的显式配置和参数传递”，替代“先还原仓库外历史有效配方”的前置关口。这不是切换为原始 GRPO，也不是把当前安装库的默认值追认为历史事实。记录实施起点的源文件、默认值、分支及传参条件，已退出变体的配置不再作为活动选择，但不能因删除开关而改变保留路径的算法参数。

基线分为以下证据类型：

| 类型 | 本次处理方式 |
| --- | --- |
| 当前源码显式字段及默认值 | 从仓库原文提取并保全，区分算法/优化、资源、普通运行参数 |
| 未设置或条件传递字段 | 记录是否省略参数及委托训练库的边界，不猜测其最终有效值 |
| 仓库已有实验记录 | 保留来源和记录值，仅作历史注记，不自动覆盖当前主线配置 |
| 未在仓库提供的历史有效值和版本 | 标记未知，不要求外部补证，不据此宣称历史等价 |

例如当前 `loss_type`、重要性采样级别和奖励缩放选项未指定时，源码只在非 `None` 时向训练配置传入这些字段。去掉变体选择后仍须保留相应的省略语义，不能猜填某版本默认值，不能将 `None` 错报为训练器已解析出的值。保留仓库已有依赖声明和必要的兼容检查，不在本次重构中顺带升级或降级训练框架。

`data/splits/v6_grpo_train.json` 的 `formal_run` 是现有历史证据：它指向 `outputs/sft/final` 和从 `v3_81.json:train` 派生的 33 条 `rl_train`，并记录部分超参数，但不构成完整有效配方。缺失的发布 manifest、启动记录或精确依赖版本不再是 S0 的阻塞项，也不从历史摘要重建这些文件。

主线配置可以落为小型 `configs/training/mainline.json`，仅承载当前可确认字段及明确的未设置语义，不建立配方框架。后续实际运行时，再由配置对象记录请求值、解析值、可获得的派生值和真实依赖版本；这项记录能力的代码需要实现，但本次不为取得这些值启动训练或准备资产。显式字段不得被兼容层静默丢弃。模型默认路径、正式命令的任务选择与完整任务轮数属于另行列明的入口/运行时修正，不伪装成所有默认行为均未变化。

### 2. Separate Three Kinds of Change

采用以下阶段边界，而不是一次性删除、纠错、换 prompt 后只跑一个最终分数：

| 阶段 | 允许变化 | 不变项及关口 |
| --- | --- | --- |
| S0 仓库基线 | 索引当前源码、配置、测试和已有记录，计算资产清单与哈希 | 不增加行为输入，不补历史材料；明确可运行检查及范围外验证 |
| S1 工程清理 | 删除独立旁支、抽取公共逻辑、收敛安装和配置入口 | 保全主线显式算法配置/传参；在已有输入可检查的范围内比较行为，CLI 默认修正单独说明 |
| S2 语义统一 | 统一错误、引用、fallback、翻译、观测及多轮验证代码 | 奖励公式不变；逐项记录预期差异，以源码、既有测试和记录核对，不声称全量行为等价 |
| S3 上下文统一 | 改造共同消息、数据生成、SFT 标签和 GRPO token/mask 代码 | 新协议版本不覆盖历史；不运行生成器、不新增 fixture、不要求真实 tokenizer 或模型执行 |

S1 不要求三种历史路径先变成完全相同。先依据当前实现和已有检查抽取公共部分，再在 S2 消除明确的不一致。S2 中因同步翻译而修正的提示词语义也需记录文本哈希，不将其藏在“格式未变”中。S3 完成后只保留一种活动上下文实现；历史复现依赖原源码版本和产物，不永久维护双套运行时，本次也不执行历史复现。

本次工程验收采用以下证据边界：

- 必需检查是源码/导入/配置检查、当前可运行的相关测试、现有轨迹与报告读取及评分兼容性、10 工具/86 任务定义/切分的完整对比。新增断言可以引用现有输入，不新建合成输入或行为基线 fixture；清单、哈希及差分报告是对现有内容的计算记录，不是新的样本数据。
- 既有测试中的 mock、固定响应和 token 序列可继续用于原有回归并适配接口，但不扩充为新模拟数据集或端到端成功轨迹。已有 `artifacts/*/traces.jsonl` 只支持其实际记录内容的解析、评分和兼容性检查，不代表当前代码重新执行了工具，也不从这些记录补造缺失快照。
- 实施开始时按当前材料确定可运行检查，记录依赖与输入限制。缺少研究资产或真实训练栈的检查列为“本次未验证”；保留相应代码能力和运行时保护，不把这些检查列成后续必做项。不得将重构引入的导入、接口或可运行测试失败改列为资产缺失；既有失败与新增回归分别报告。
- 不执行完整 86 任务回放、SFT 数据生成、训练、真实 tokenizer 或模型集成。全部范围内检查通过后可完成工程重构，但不宣称全量任务执行、历史配方等价、真实 token 前缀一致、梯度行为或模型能力已获验证。

### 3. Delete by Dependency Boundary

以下是最终删除清单；共享调用者迁移之前，不能靠移除失败测试强行删除。

| 单元 | 最终删除范围 | 先处理的引用或保留项 |
| --- | --- | --- |
| 其他策略训练 | `rl/train_dpo.py`、`rl/train_opd.py`、`rl/opd_multiturn.py`、`rl/train_ppo.py`、`scripts/generate_dpo_data.py` | 精度、数据 IO、模型加载、日志和 SFT/GRPO guard 保留；删除 StageVerifier 的 DPO 分支 |
| 自进化 | `rl/self_evolve.py` 及专属测试 | `eval/badcase_replay.py`、badcases、rollout audit 和信号筛选保留 |
| 独立 VLM 后训练 | `rl/train_vlm_figure_qa.py`、`scripts/build_figure_qa_dataset.py`、`scripts/eval_figure_qa_vlm.py` 及专属测试 | `tools/figure_analysis_tool.py`、现有 VLM/抽取式推理、快照构建与 T5 backfill 测试保留 |
| GRPO 变体 | `train_grpo.py` 的 DAPO preset、GSPO/Dr.GRPO 选择和动态采样分支，之后删除 `rl/dynamic_sampling.py` | 先核对保留主线的源码配置与传参；保留方差诊断、显式早停配置、`build_grpo_signal_split.py` 和 `audit_grpo_rollouts.py` |
| Jev | `routing/`、`scripts/jev_route_smoke.py`、`scripts/jev_project_integration_smoke.py`、专属测试与 fixture、`jev_config.example.env` | 普通策略路径断言迁入核心测试；活动 `JEV_ROUTER.md` 退出，历史结果只读保留 |
| 其他 Agent | `mcp_protocol/`、`skill_cli/` | benchmark 单一 ReAct 构造与 CLI 选项迁移，移除路由/架构比较专属报表字段 |
| Web/服务/数据库 | `AgenticArxivWeb/`、`api/`、`services/`、`models/db.py`、`models/orm.py`、`models/store_mysql.py`、`bin/start.sh`、`bin/shutdown.sh` | 先剥离 MySQL/SSE 日志，翻译状态与调用移入共同环境；普通 `main.py` 和外部 LLM API 客户端保留 |
| 重复运行时 | `agents/context_manager.py` 及导出；迁移后删除 `agents/side_effects.py`，移除冗余 `BaseAgent` 抽象 | 不再有多 Agent 子类后，由精简的具体 ReAct 运行器承担普通循环 |
| 旧奖励降级 | 单轮 `synthesize_trajectory`、旧消息合成以及缺失轨迹 fallback 的活动调用 | Canary、阶段验证和 reward callback 先消费真实轨迹；旧报告读取不需要执行旧工具链 |

`models/store_memory.py` 保留会话和资产能力，删除异步任务 CRUD；`models/schemas.py` 保留 `Paper`、`PdfAsset`、`TranslateAsset`，清理异步任务和 API 日志专属 schema。`utils/io_utils.py`、`rl/precision.py`、`rl/observability.py`、`benchmark/task_io.py` 等共享模块，即使注释提到已删除算法，也不因此删除。

测试采用同样边界。删除独立 DPO/OPD、自进化、VLM 后训练、Jev 和动态采样专属测试；`test_training_guards.py` 只删除 DAPO/GSPO/PPO 等组并更新入口枚举，保留 SFT、GRPO、精度、QLoRA、分布式、manifest、数据长度、注册与验证器测试。`test_cli_entrypoints.py`、`test_api_rollout_model.py` 不是退出功能的专属测试，继续保留。

### 4. Share a Small Protocol and Episode Environment

共同协议放在现有 `agents/` 范围内，环境继续利用 `rl/env.py` 的后端和 `rl/multiturn_env.py` 的轨迹能力，不引入插件注册层。最终依赖方向如下：

```text
Expert data       GRPO token sampler       CLI / Benchmark
      \                  |                       /
       +----------- ReAct contract ------------+
                parse / messages / trace
                          |
                  Episode environment
                   reset / setup / step
                          |
                  replay / live / record
                          |
                      10 tools
```

解析层只输出调用、结束或格式错误，不自行执行工具。调用要求明确 `name` 与对象 `args`；裸 FINISH 及空参数 JSON FINISH 精确匹配。格式错误直接结束并记录 `PARSE_ERROR`；模型生成的框架失败标记不转成 FINISH。语法正确但未知的工具或非法业务参数在执行前被拒绝，返回一致错误观测，不调用后端；允许模型在剩余预算内响应工具错误。没有隐式参数修复、路由纠正或额外政策重试。

环境统一 `reset/setup/step`，将工具注册清单作为唯一能力来源，保留注册顺序。调用参数与框架参数分开：会话 ID、内部解析对象只在执行时注入，工具公开业务签名不变，模型动作和评分不包含内部字段。初始化和语义 oracle 共用执行契约，但 oracle 的 gold steps、expected IDs 和评分信息留在评测侧。

每个 episode 持有自己的 MemoryStore、引用状态和临时产物根。保留 `models.store` 门面以减少工具改动，但将 episode 的 store 绑定为上下文局部对象，而非使用 `use_memory_store()` 全局切换；不新增服务或后台线程。产物根同样由 episode 上下文提供，禁止临时改写进程全局 settings 来模拟隔离。直接工具测试可显式绑定本地上下文。

回放资产和预取内容只读共享；既有离线运行若需要临时占位文件，只能写入 episode 自有目录，不将其保存为新快照或验收数据，本次不为验证扩充此类输入。reset/close 仅清理该 episode 拥有的目录，保留 conftest 的测试隔离。对 search fallback 不更新候选，所有引用工具按同一规则更新最近操作对象；下载、翻译和缓存状态只保留一份可查询真值，不再一边用集合、一边用共享资产 store 各自判断。

真实翻译同步调用现有翻译工具；回放模拟相同业务状态，但返回 `offline` 或等价来源字段，不声称创建真实 PDF。真实工具失败和快照未命中统一成可审计错误；离线模式绝不自动联网。

### 5. Keep Structured Results and a Stable Scoring Projection

每一步保留原始模型文本、解析动作、结构化工具结果、模型可见观测、错误与时间/token 信息。共同观测渲染器按稳定字段顺序输出，只截断大段正文；论文 ID、引用映射、状态、失败原因不得因粗暴 `str(result)[:1000]` 而丢失。可见预算及截断事实属于版本记录。

继续使用现有 `rl/trajectory.py`、`rl/reward.py` 与 `benchmark/metrics.py`，先由共同轨迹投影为现有评分输入，不顺带改奖励权重、分量或失败上限。相同已记录轨迹的重评分必须保持一致；因 S2 实际轨迹变化导致的分数差异单独列出。FINISH 只表示结束，blocked 任务的最终 Thought 与原因判定、虚假完成保护均保留。

奖励回调只读已执行轨迹并返回标量，不重新调用工具，也不再次计算组内相对优势。缺少完整轨迹属于接线错误，应立即失败。历史字段兼容只放在读取/转换层，保留来源版本；不将缺失字段推断成成功或新的协议事实。

### 6. Use One Message Protocol With Distinct Training Adapters

S3 采用初始 user 任务消息、assistant 文本动作、user 环境 Observation 交替的消息序列，避免要求外部 API 支持 assistant prefill 或原生工具角色。环境消息明确标识为数据；工具内容不能改变协议规则。任务与工具描述只在初始消息建立，不再每轮把全部历史重新塞入一个 user prompt。

```text
user       task + tools + visible setup
assistant  Thought + Action
user       Observation from environment
assistant  Thought + Action
user       Observation from environment
assistant  Thought + FINISH
```

- SFT 数据仍是逐决策点的 prompt/completion：prompt 为此前完整消息，completion 仅为当前 assistant 响应。改造生成器和 `train_sft.py` 的转换/标签代码，保留参数/语言扩增、来源任务、切分、去重与样本哈希能力。仅在未来另行运行生成器时写入新版本样本和 manifest；本次不生成、重生成或导出数据集，不启动训练循环。
- GRPO 保留独立 token 采样循环和原始采样 ID。实际运行时由指定 tokenizer/chat template 生成可追加且不改写前缀的环境消息片段，环境与运行时插入边界 mask 为 0，策略采样 token 为 1；采样得到的结束 token 不重复插入。不能 decode/re-encode 已采样响应充当训练输出。代码保留前缀一致性检查，不支持的模板应明确失败；本次不下载 tokenizer、不以新增模拟 token 替代真实集成验证。
- 模型伪造 Observation 时只保留原始采样 token 的合法前缀，不能执行伪造的后续动作。标记跨 token、预算截断、空输出和多轮结束的保护通过源码与已有可运行用例核对，未有现成输入覆盖的边界如实列为未验证，不为补覆盖造数。
- 保持当前支持版本下的 prompt 分组与采样数量契约，不在 rollout 中再次按 G 扩增。完整轨迹一对一随采样结果传给奖励函数。
- 普通本地与 API 推理共用消息构造，外部 API 只承诺逻辑消息一致。训练与推理用同一支持的 chat template，不保留只在训练侧偷偷替换模板的分叉。本次检查调用边界与已有消息用例，真实本地 tokenizer 和外部 API 集成不作为验收条件。

元数据写入代码记录 `runtime_version`、`context_version`、实际可得的模板哈希和训练协议。历史文件没有的字段保持未知，不补造或改写原件；今后生成的新数据使用新版本标识。新协议训练拒绝不兼容的已有 SFT 数据/模型元数据；评测旧 checkpoint 时必须明确其训练协议与评测协议的关系，未知时不能默认为兼容。最终不保留长期的旧格式执行开关。

`rollout_func` 与私有采样 hook 的版本风险通过现有兼容层、运行时接口检查和仓库可运行测试控制，不通过移除 `trl_compat` 或降级到单轮绕过。没有真实依赖栈或 tokenizer 的动态证据时记录限制，不宣称固定版本集成已经验证；本次不以补齐外部文档或环境作为重构前置条件。

### 7. Migrate Every Consumer and Keep One Evaluation Standard

共同消费者清单为普通 ReAct、`rl/rollout.py`、`benchmark/runner.py`、SFT 专家生成、GRPO rollout、Canary、StageVerifier、语义 oracle 和坏例回放。迁移时逐项核对 setup、状态和轨迹的调用契约，运行已有可用检查，不能只替换两个主入口；没有运行条件的入口仍要完成代码迁移并标明动态验证缺口。

Canary 入口使用小规模但完整的 episode；正式 benchmark 入口使用声明切分及重复次数。保留当前 SFT/GRPO 验证阈值和覆盖说明，不能为了适应更严格的多轮检查悄悄放松阈值。本次复用现有测试和历史记录检查评分、阈值及完整轨迹接线，不新增成功/失败轨迹，不要求真实模型通过阶段验证。Canary 仍只是健康检查，历史记录的读取也不能代替当前模型在 86 条任务和留出集上的能力报告。

正式训练命令显式绑定任务集与已有训练切分，保留 8 条冒烟任务作为显式选项，但本次不运行训练。历史记录继续指向原 33 条 `rl_train`；86 条任务集的 `v7_86.json` 为 train 55、dev 8、iid_test 19、ood_test 4，不自动替代历史训练子集。后续另行重测训练信号时，只在声明 train 中产生新选择产物，不覆盖原文件；本次仅保留并检查信号诊断接口，不要求新模型审计结果。

SFT 产物通过主线配置及 manifest 交给 GRPO，不使用“扫描最新目录”或默认 DPO 路径；保留全参与适配器加载能力。Makefile、依赖、CI、README 及活动文档只展示支持路径。删除服务配置示例，不改写用户真实 `.env`、凭据或未受支持的历史输出；移除 Web 依赖时保留 `pdf2zh`、Pillow、PyMuPDF 和按需日志后端。

## Risks / Trade-offs

- [历史有效配方和依赖行为未知] -> 只保全当前显式配置、默认值和省略/传参语义，不猜填历史值；未设置字段仍受未来实际依赖版本影响，不宣称历史等价或完整算法复现。
- [共享逻辑抽取混入语义修正] -> S1 依据当前源码、现有测试与记录做有限范围对比；S2 逐项记录预期差异和检查方式，不能把静态检查写成全量动态证明。
- [新上下文影响旧 checkpoint] -> 实现版本和兼容检查但不生成新数据；旧模型表现和新协议训练均未验证，后续实验不列为本次必做任务。
- [采样 token 或 mask 被破坏] -> 复用已有 token/mask 用例并审查原始 token、EOS、G 分组和奖励映射保护，实际运行校验前缀；缺少真实 tokenizer/训练栈导致的集成风险如实保留，不以新增合成 token 补证。
- [文件或会话串扰] -> episode 显式拥有临时根和上下文局部 store，复用既有隔离测试及只读输入，不清扫共享产物根，不新增模拟快照。
- [删除专属测试时误删核心保护] -> 按断言和调用者分类迁移混合测试；保留工具、引用、数据泄漏、虚假完成、资源及版本 guard。
- [同步真实翻译阻塞更久] -> 沿用现有超时/失败处理并清楚展示耗时，研究主线不再承诺 Web 异步响应；离线训练不调用真实翻译。
- [范围扩张成重写] -> 保留现有模块和训练框架，以最小共享契约消除重复，不引入插件、远程服务或多模型编排。
- [只凭现有材料无法覆盖全部新边界] -> 规格保留功能要求，验收报告逐项标明源码审查、已有测试、已有记录或未验证；完整 86 任务执行、真实模型/工具与 tokenizer 集成不作为本次完成条件，也不声称已通过。

## Migration Plan

1. **S0**：记录当前源码配置与传参、工具/任务/切分定义及已有记录的来源/哈希；列出当前可运行检查和资产依赖的范围外检查。不捕获新的脚本化轨迹、不生成 fixture、不补齐历史材料。
2. **S1**：先删除独立训练与接入旁支，再移除服务/数据库依赖和活动入口；保留尚未迁移的最小本地状态适配。用源码及已有输入核对保留路径，单独记录修正后的 CLI 默认路径。
3. **S2**：接入共同解析和 episode 环境，修正错误/终止、引用、fallback、同步翻译和观测；迁移全部消费者，再删除本地副作用体系与合成轨迹降级。执行现有可运行回归和历史记录兼容性检查，保留预期差异清单，不要求 86 条任务回放。
4. **S3**：实现共同消息协议、SFT 转换与标签、GRPO token/mask 和版本记录；用现有输入检查可验证路径并审查其余契约。保留后续生成数据、训练及模型评测命令，但本次不执行这些流程、不新增模型响应/token/轨迹输入、不要求真实 tokenizer 集成。
5. **收尾**：保存工程检查报告并更新活动文档；核对工具/任务/切分定义、删除边界和消费者迁移，确认范围内检查无未解决的新增回归。资产依赖项记录为本次未验证，不标记通过，也不保留为实施阻塞任务；工程重构可独立完成，不宣称历史复现、完整任务执行或训练/模型效果。

回退以阶段差异和源码/资产标识为边界：保留用户改动，通过对应阶段的反向补丁恢复代码，再运行仓库内可用检查；不覆盖旧模型、切分或快照，也不使用清空工作区的方式回退。S2/S3 的失败不要求重新引入已经独立验收的 Web 或其他训练旁支。本规划不授权提交、推送或发布。
