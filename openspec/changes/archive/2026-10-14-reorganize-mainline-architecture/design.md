# Design

## Context

动机与范围见 [proposal.md](proposal.md)。这是跨模块、安装接口和历史证据保全的变更，需要设计文档；不是目录重命名。

当前已观察到的边界问题：

- `agents/agent_engine.py` 使用 `rl.multiturn_env` 与 `rl.run_manifest`；`rl` 同时包含通用执行、数据准备、训练和诊断。
- `rl/multiturn_env.py::setup` 自行查询 `benchmark.tasks`，解析依赖任务并执行标准步骤；运行时与任务目录耦合。
- `rl/reward.py` 使用 `benchmark.metrics`，而 benchmark 的基线与坏例又导入奖励模块。
- `rl/train_grpo.py` 从 `rl/train_sft.py` 导入四个 QLoRA 私有辅助函数；训练入口同时承担公共库职责。
- 数据脚本彼此导入，并与多个入口一样注入 `sys.path`。当前没有包构建配置；CI 安装完整 requirements 并设置 `PYTHONPATH`，不能证明基础安装独立可用。
- `scripts/freeze_mainline_assets.py` 被现有资产测试直接加载，不是可以退出的一次性脚本。另两个迁移审计脚本包含固定源码路径与旧阶段信息。
- 两个未跟踪的行为基线文件未纳入 CI，也不是当前验收依据。前端目录只剩缓存与空目录；这些都是实施时需要重新确认的工作区事实，不能据规划时状态盲删。

已有三个主规格仍是语义约束。本次仅为 `training-mainline` 和 `research-validation` 增加安装及验收契约，`react-runtime` 不产生协议增量。当前 `react-runtime-v2`、`react-messages-v3` 和提示词模板内容不能因源码搬迁改变；源码位置及源码文件哈希变化不等于协议变化。

## Goals / Non-Goals

**Goals:**

- 形成可安装、可从任意目录调用的单一命名空间，消除运行位置与源码布局的隐式依赖。
- 让运行时、任务准备、评分、训练和评测各有明确所有者，并以静态依赖和实际轻量导入检查守住边界。
- 减少安装负担，同时保留全部工具、既有数据流程和训练保护。
- 将结构与命令变化同研究语义、协议和历史证据区分验证。

**Non-Goals:**

- 不保留旧命令、旧导入路径或双套实现；不增加插件系统、通用工作流引擎、依赖注入容器或新的 Agent 架构。
- 不将所有模型调用强行抽象成一个采样循环，不改写 GRPO token，也不优化或修订既有奖励、初始化、工具及模型行为。
- 不升级训练依赖，不改变已有依赖版本保护和平台/GPU要求；不发布包。
- 不重写历史报告、旧迁移记录或归档 OpenSpec，不生成训练数据或执行真实 tokenizer、模型、训练及完整任务集。

## Decisions

### 1. 单一安装包和薄命令入口

采用 `src` 布局、`pyproject.toml`、常规 setuptools 构建和 `agentic-arxiv` console script。`python -m agentic_arxiv` 仅调用同一 CLI，供模块启动及分布式 launcher 使用，不形成第二套参数或实现。使用现有标准库 argparse 统一命令解析，迁移后无活动使用方的 `fire` 不再成为依赖。

```text
pyproject.toml
src/agentic_arxiv/
    __init__.py
    __main__.py
    cli/
    runtime/
    tools/
    state/
    llm/
    tasks/
    scoring/
    training/
    evaluation/
    data_pipeline/
    contracts.py
    provenance.py
    config.py
    resources/mainline.json
tests/
scripts/maintenance/
configs/accelerate/
data/
artifacts/
docs/
openspec/
```

包内统一使用 `agentic_arxiv.*` 或包内相对导入，禁止路径注入和旧命名空间别名。根包及子包初始化不加载训练器、模型、用户 `.env`、日志文件或研究资产。CLI 在选定实际操作后才加载执行实现和操作配置；帮助只读取参数声明与必要的包内默认配置。

选择该方案而非保留根目录可执行脚本或兼容 shim，是因为用户明确优先架构清晰而非旧接口兼容。参数名统一为 CLI 的连字符形式并明确映射到现有配置键；布尔开关、默认值、显式覆盖与未设置字段不能在 argparse 迁移中失真。

### 2. 职责所有权与依赖方向

| 目标位置 | 主要来源 | 边界 |
| --- | --- | --- |
| `contracts.py` | `react_contract.py` 的类型、纯结果投影；`trajectory.py` 的记录类型 | 纯数据、枚举与转换，不读取模板、环境、任务目录或模型 |
| `runtime/` | Agent engine、提示词与消息协议、env、multiturn_env、sandbox | 执行已准备的初始化与策略动作；保留 live/record/replay、隔离和终止语义 |
| `tools/` | 现有工具及 PDF 下载/翻译等工具辅助代码 | 工具业务与注册声明；不引用任务、评分或训练 |
| `state/` | schemas、内存 store、上下文绑定、artifact_paths | 会话与文件归属；通用安全路径处理不能反向依赖 PDF 业务实现 |
| `llm/` | `utils/llm_client.py` | API 与本地后端，按能力延迟加载；不通过训练入口获得模型加载能力 |
| `tasks/` | TaskSpec、任务目录、splits、task_io、semantic_oracle、初始化依赖解析 | 任务与参考答案准备；纯计划编译与显式环境执行分开 |
| `scoring/` | metrics、reward、纯轨迹健康与异常判定 | 只读判定；不导入模型、执行工具、加载文件或写报告 |
| `training/` | SFT/GRPO、grpo_reward、precision、trl_compat、训练回调 | 训练器适配、共享模型配置、训练保护和日志；两个阶段不互相导入入口 |
| `evaluation/` | benchmark runner/report/baselines/badcases、rollout、现有诊断脚本的业务逻辑 | 组合运行与评分，读写评测报告；不依赖训练器或训练回调 |
| `data_pipeline/` | 生成、增强、混合、signal split、build_snapshot、backfill | 可导入的数据准备函数，不导入 CLI 或训练阶段入口 |
| `provenance.py` | run_manifest、轨迹文件读写等来源记录能力 | 接收显式数据与路径；不反向查询训练器、任务目录或当前运行时模板 |
| `config.py`、`cli/` | settings、各入口参数和组装逻辑 | 配置解析与用户入口；配置本身不构造执行对象 |

原 `utils/` 不机械复制成新的杂项目录：PDF 处理归 tools，JSONL/轨迹持久化归 provenance，会话路径归 state，日志初始化由明确调用的配置辅助函数负责。简单辅助逻辑保留为函数，不为每个概念新增框架。

内部依赖允许列表如下；标准库与对应安装层的外部依赖另行检查：

| 使用方 | 允许使用的项目模块 |
| --- | --- |
| `contracts` | 无 |
| `config`、`provenance` | `contracts`；包资源仅由 config 显式读取 |
| `state` | `contracts`、`config`、`provenance` |
| `tools`、`llm` | `contracts`、`config`、`provenance`、`state` |
| `runtime` | `contracts`、`config`、`provenance`、`state`、`tools`、`llm` |
| `tasks` | 纯目录/切分/计划使用基础契约与来源工具；仅 reference 准备模块可以显式使用 `runtime` |
| `scoring` | `contracts` 和只处理已提供数据的来源辅助函数；不得调用文件或执行层 |
| `evaluation`、`data_pipeline` | 上述基础模块、`runtime`、`tasks`、`scoring`，不依赖彼此的 CLI 或 `training` |
| `training` | 上述共享模块及 `evaluation` 的模型无关评测服务，不导入数据生产脚本或 CLI |
| `cli` | 按所选子命令组装所有能力，不被业务模块反向导入 |

训练可以调用公共评测服务，评测不能导入训练：这允许 Canary 复用完整 episode 评测，而不把训练回调放进评测包。纯评分与任务声明保持独立，运行时不能为了初始化查找任务目录。依赖检查覆盖普通导入、函数内导入、包重导出及已有动态导入清单；不能通过动态 import 绕过约束。

未选择只新增 `runtime/` 并留下其余交叉引用，因为那会保留训练入口私有函数和 benchmark/RL 的双向职责关系；也不选择大而全的 `common/`，以免隐藏依赖方向。

### 3. 任务准备与执行分离，但保持初始化行为

在 `tasks` 编译任务的依赖链和 setup 为明确的初始化动作序列，保留现有查找范围、错误条件、执行顺序和重复动作语义。纯编译不执行工具，输出仅含执行需要的动作及参数，不携带任务标准答案或评分字段。

上层工作流分别持有策略任务文本、初始化计划、参考答案和审计身份。`runtime` 执行初始化后只从实际会话生成可见状态，再开始策略循环；初始化不进入策略 history、奖励动作或学习标签。任务 ID 可以作为执行外部的审计关联，但不写入模型消息。

`tasks/reference` 显式创建独立 replay 环境，使用同一初始化计划解释 gold steps，得到既有语义目标；它不是评分函数，也不能在历史只读重评分中被隐式调用。普通任务执行与参考答案准备不共享可变 episode。

不改变 setup 的内容、工具调用行为、状态过滤或不可用输入的处理。既有输入未覆盖的情况标为未验证，而不是为结构迁移创建新的任务或响应。

### 4. 共同评分与训练适配器各归其位

将奖励与指标合入 `scoring` 的不同模块，保留原公式、权重课程、失败上限、终止判定与缺失字段处理；不趁机合并或简化计算路径。训练与评测传入相同轨迹和参考数据，评分只返回数据。

SFT/GRPO 的共享 QLoRA 配置、运行时检查、可训练参数检查移入 `training/model_setup.py`，两种阶段只导入该公共模块。当前主线配置、传参省略、训练保护、精度规则、模型来源及全参/适配器契约继续按原有调用路径验证，不把代码抽取变成模型加载策略升级。

GRPO 的采样与 token 追加适配器留在 `training`；普通 ReAct 与 GRPO 共享消息协议、解析和环境转移，不要求共用文本生成循环。保留原始 sampled IDs、prefix 校验、模型/环境 mask、分组大小及奖励归一化责任；SFT 仍仅监督当前目标 assistant。

Canary 的固定任务执行和阶段验证的通用 episode/指标计算移入 `evaluation` 服务，训练 callback、阈值策略和停止逻辑留在 `training`。公共评测服务接收已提供的 client/任务/环境工厂，不导入训练回调。现有轨迹异常纯计算移入 scoring，训练审计 writer 留在 training，独立历史诊断留在 evaluation，共享 provenance 而不相互导入入口。

### 5. 命令覆盖完整，但不保留旧调用方式

以下为目标命令族，具体参数按现有操作逐项映射；实际执行必须经过原有输入与依赖保护。

| 现有操作来源 | 新命令 |
| --- | --- |
| `main.py` | `agentic-arxiv run` |
| `rl/rollout.py` | `agentic-arxiv evaluate rollout` |
| `benchmark/run_benchmark.py` | `agentic-arxiv evaluate run` |
| `benchmark/run_baselines.py`、`rescore_traces.py` | `agentic-arxiv evaluate baselines`、`evaluate rescore` |
| `eval/badcase_replay.py` | `agentic-arxiv evaluate badcases replay`、`capture`、`run` |
| `rl/train_sft.py`、`train_grpo.py` | `agentic-arxiv train sft`、`train grpo` |
| `rl/build_snapshot.py`、`backfill_figure_analysis.py` | `agentic-arxiv data snapshot`、`data backfill-figures` |
| 两类 SFT 生成与增强脚本、mix | `agentic-arxiv data sft generate`、`generate-parametric`、`augment`、`augment-parametric`、`mix` |
| `scripts/build_grpo_signal_split.py` | `agentic-arxiv data grpo select-signal` |
| rollout 审计、课程分析 | `agentic-arxiv audit rollouts`、`audit curriculum` |
| 长期资产冻结/核对 | `agentic-arxiv audit assets` |

rollout 与普通推理的默认执行模式当前不同，不因统一入口统一成同一个默认；坏例 replay 与实际 run 仍明确分开。原有基线策略诊断功能保留，但本次不执行会产生合成研究轨迹的数据生产或基线生成命令。资产核对默认为只读；原冻结能力仍需显式选项且拒绝覆盖。

历史重评分只读取轨迹中已冻结的逐步 `expected_paper_ids`、显式 `--references` JSON 或任务已经明确声明的参考。新 benchmark 轨迹保存实际用于评分的参考，不向模型 history 注入；旧轨迹不回填。论文工具需要的参考缺失或冲突时，在写报告前明确拒绝重算参数、指代和严格成功指标，不退回 `ref` 字面比较，也不从策略观测猜答案。`rescore --snapshot` 仅保留显式路径校验，不再强制要求没有读取用途的默认快照。

CLI 的参数声明独立于重依赖模块。帮助、参数解析和安装指引不要求模型或训练依赖；真正使用本地后端/训练时，先按所需能力检查依赖，再进行原有输入校验和执行。`.env` 只在显式启动操作时从工作区读取，既有环境变量优先级不改变；不在 import 或帮助过程中读取凭据或创建日志文件。

普通推理交付继续打印实际观测及最终说明，保存完整结果并在 episode 清理前导出自有产物；不丢失已修复的日期序列化和持久文件交付。CLI 只组织交付，文件复制与轨迹序列化可由来源/产物辅助函数承担。

分布式调用使用同一模块 CLI，例如 `accelerate launch ... --module agentic_arxiv train grpo ...`；原 DDP/FSDP 配置内容不变。本次只检查入口接线与帮助，不启动分布式训练。

### 6. 安装层级、包资源和工作区

| 安装层 | 声明范围 |
| --- | --- |
| 基础 | arxiv、requests、python-dotenv、loguru、pydantic、PyMuPDF、Pillow，保留既有版本约束 |
| `local` | torch、transformers、accelerate、peft；用于现有本地推理、默认 LoRA 适配器产物和按需模型工具后端 |
| `train` | 包含 local 的同版本约束，再加 trl、datasets、bitsandbytes 及训练代码直接使用的依赖 |
| `translate` | 现有 pdf2zh 能力 |
| `logging` | 现有 tensorboard 声明；wandb 仍可按既有显式方式使用，不自动启用 |
| `dev` | 现有测试、覆盖率、lint 与包构建/元数据检查工具，不隐含 train |

extras 的重复版本约束在元数据检查中保持一致；不额外引入另一套依赖文件或新依赖管理器。补齐现有代码直接使用却依赖传递安装的包时，记录原因，不借此调整已有下限。保留 TRL 多轮 rollout、QLoRA 等原版本保护。基础包不承诺执行本地 VLM 或真实翻译，但必须注册全部业务 schema，并在使用相应能力时明确失败。

当前 `configs/training/mainline.json` 作为活动默认配置，按原字节迁入包资源 `resources/mainline.json`，通过资源 API 读取，不保留两个可分别编辑的默认副本；加速配置继续放在 `configs/accelerate/`。它们不是历史训练 manifest，迁移清单需记录配置原路径及相同内容哈希。显式 `--config` 仍覆盖默认配置，CLI 显式项继续覆盖配置，库委托的五个 GRPO 字段仍省略传参。

引入全局 `--workspace`，默认当前工作目录；相对输入、配置及输出路径统一相对 workspace，绝对路径不改写。模型默认逻辑路径、snapshot 和输出子路径保留对应旧操作的默认值；原来固定在源码目录下的工具缓存改由工作区解析，并在迁移表明确记录这一允许的路径变化。不得写入 site-packages，也不得自动找到另一个仓库或下载缺失资产。

包仅携带代码、任务定义和必要的默认配置，不包含 `data/`、模型、轨迹、报告、历史基线或输出缓存。训练 manifest 记录实际使用的资源、配置与源码位置；历史 manifest 不被回填。

### 7. 历史保全与维护工具

实施前在新的 `artifacts/architecture_migration/` 证据目录记录当前源码、配置、任务资产、研究文件及已有工作状态。只从现有文件计算清单与哈希，复用上次清单并核对覆盖范围；不重冻覆盖 `artifacts/mainline_baseline/`。模型仅作文件保全核对，不加载权重；大文件按流式哈希处理。

实施结束后，用户明确要求删除该临时证据目录。仅维护审计仍依赖的独立来源映射按原字节移至 `scripts/maintenance/source_map.json`，相关默认路径和活动文档同步更新；一次性报告、日志与取证脚本不继续保留。此前研究资产、旧迁移证据及两项非活动归档不在此次清理范围。

源码映射单独记录旧路径到新路径或拆分后多个路径的关系。历史文件里的旧路径和哈希保持原义；维护审计通过显式映射定位当前实现，同时保留旧源码身份。源码指纹改变正常，但模板内容、协议版本和现有默认配置内容不应因此改变。审计无法对应字段时应报告问题，不能将结构变化自动降级为“未知”以绕过原有可检查项。

长期资产核对实现进入 `evaluation/assets.py` 并由 `audit assets` 和现有资产测试共用，保留完整 schema、顺序、任务和切分核对。历史记录与配置审计进入 `scripts/maintenance/`，仅依赖已安装包和显式 workspace；修改定位与导入，不覆写旧阶段报告。

非活动文件归档清单仅包含：

| 原路径 | 归档路径 |
| --- | --- |
| `AgenticArxiv/tests/test_mainline_behavior_baseline.py` | `docs/archive/mainline-behavior-baseline/test_mainline_behavior_baseline.py.txt` |
| `artifacts/mainline_baseline/behavior.json` | `docs/archive/mainline-behavior-baseline/behavior.json` |

文件内容保持字节一致，原路径、归档路径、哈希和“不作为当前验收证据”写入新归档说明。使用 `.py.txt` 避免进入测试发现或被误认为活动代码；不修复其旧导入、不运行它、不扩充其中的输入。其他数据、模型、快照、manifest、报告和已归档 OpenSpec 原位保留。

本地清理仅针对重新确认后的 `AgenticArxivWeb/node_modules`、`AgenticArxivWeb/.vite` 与空 `AgenticArxivWeb/src/`、根目录 `bin/` 等残留；删除缓存不跟随符号链接，删目录前确认无额外内容。有新增文件则保留并报告，不扩大清理范围。新包所在的根目录 `src/` 不在清理范围。缓存是本地维护结果，不冒充 Git 中的代码减少。

新迁移说明放在 `docs/architecture_migration.md`。活动 README 各语言版本、Overview、贡献说明及运行指南更新新命令；旧报告、`docs/mainline_migration.md` 和归档变更不做全库替换。已有测试 fixture 随测试目录迁移时字节不变并记录来源；它们与研究资产及上述非活动基线分别核对。

### 8. 以现有材料证明边界与语义，而不是以目录证明

实施时先记录可运行回归范围及既有失败，再迁移同一批测试；接口适配和断言允许调整，不添加新的论文、快照、响应、token 序列或行为数据。新增架构和安装检查只读取实际源码、构建元数据、命令帮助及已有材料。

| 验证面 | 方法与已有依据 |
| --- | --- |
| 工具、任务、切分 | 迁移 `test_mainline_assets.py`、`test_task_spec.py`、`test_splits.py`，对现有规范化资产作全内容和哈希比较 |
| 初始化与状态隔离 | 原 `test_multiturn_env.py`、`test_reference_resolution.py`、`test_rollout_sandbox.py`、session leak 用例改接口，核对同一初始化序列和可见状态 |
| 协议与训练责任 | 原 `test_grpo_reward.py`、`test_training_guards.py`、终止及轨迹测试，保持提示词哈希、标签、原始 token、prefix/mask 与分组规则 |
| 评分与历史读取 | 原奖励、metrics、badcases、benchmark 测试和历史 traces 只读审计；不重新执行模型，不更新输入 |
| 训练配置 | 适配现有源码配置审计，逐项核对默认、优先级、未设置传参及保护逻辑，不以新路径为由删掉检查 |
| 依赖方向 | AST 解析实际导入及已声明动态导入；检查各模块允许列表、无旧命名空间、无路径注入，辅以导入隔离探针 |
| 基础安装 | 无训练栈的隔离环境检查全部子命令帮助、工具注册、现有历史读取及轻量回归；不能仅靠 monkeypatch 阻止 import 宣称依赖安装隔离通过 |
| 包与路径 | 检查编辑式安装和构建后的正常安装，仓库外工作目录、显式 workspace、默认资源、缺失输入错误；源码不能偶然补全 wheel 遗漏 |
| 本地/训练安装 | 核对 extras 元数据及版本约束；支持的平台上做对应依赖导入和已有无模型回归。CI 保留 Python 3.10/3.11/3.12，训练组用适合现有依赖的 Linux 环境；不要求 CUDA 执行 |
| 资产与交付 | 历史资产文件哈希、归档映射、临时产物隔离，以及现有日期序列化和 CLI 产物持久化回归 |

安装及构建会产生普通软件构建产物，不是研究数据生成。依赖安装未能验证的环境必须如实标注；真实模型、tokenizer、梯度、分布式训练和完整任务结果仍是范围外，不得将包安装或 mock 结果称为这些集成已验证。原本可运行检查因重构失败必须修复，不使用范围外声明掩盖回归。

## Risks / Trade-offs

- [大量导入与测试定位变更] -> 按共享契约到上层入口迁移，逐阶段检查，并在结束时删除旧命名空间；不长期维持两套实现。
- [初始化抽取改变任务行为或泄漏参考答案] -> 保留实际顺序和失败条件，计划与评分参考分离，仅复用现有输入核对；未覆盖情形不猜测等价。
- [仅改 CLI 却改了默认执行模式或未设置传参] -> 逐入口建立参数映射，重点覆盖 rollout 的离线默认、布尔开关和五个库委托字段。
- [评分抽取或训练公共模块抽取导致数值变化] -> 不优化公式、权重和加载策略，使用原有轨迹与配置审计定位差异。
- [安装隔离被开发环境掩盖] -> 基础、local、train 元数据分别检查，基础环境禁止存在训练依赖；正常 wheel 安装不能读取源码补缺。
- [包内默认配置与用户配置漂移] -> 默认资源只有一个来源，搬迁时内容哈希一致，显式覆盖和来源记录可追溯。
- [源码路径移动破坏历史审计] -> 使用独立来源映射，旧证据不改写，当前源码哈希不冒充历史哈希。
- [清理损失未跟踪工作或运行产物] -> 实施前重新盘点，两个基线逐字节归档；其余只清理白名单缓存和已确认空目录。
- [重组被误当成模型等价验证] -> 报告区分结构、已有回归和未验证集成，不生成模型样本或要求训练。

## Migration Plan

1. 固定实施起点与新迁移证据位置，记录可运行检查、资产、源码和两个非活动文件；旧材料只读。
2. 建立包骨架、配置资源、共享契约与来源/状态边界，随后迁移 tools/llm/runtime。临时迁移状态不作为完成结果。
3. 抽出任务初始化与参考答案准备、共享评分和公共评测服务，再迁移训练及数据流水线；按原有检查逐项核对。
4. 接通统一 CLI、安装分组、根 tests 和 CI，完成所有旧入口的操作映射；不留下兼容转发入口。
5. 更新活动文档与维护审计，按白名单归档和清理，执行包安装、依赖隔离、既有回归和资产保全检查。
6. 最终报告新路径/命令、已执行检查、未验证边界和任何既有失败。没有训练和模型结果也不妨碍工程验收，但实际工程回归不得豁免。

回退以分阶段、仅涉及本变更的源码与配置补丁为单位；不使用破坏性重置、不覆盖用户并行改动。研究资产从始至终不迁移、不重写，因此不依赖“恢复研究数据”完成回退；两个非活动文件可根据归档索引和哈希恢复原位置。规划及后续实施均不包含自动提交、推送或包发布。
