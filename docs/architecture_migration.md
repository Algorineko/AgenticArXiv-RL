# 主线架构与安装接口迁移

本次变更将活动源码重组为 `src/agentic_arxiv/`，使用 [pyproject.toml](../pyproject.toml) 安装，并以 `agentic-arxiv` 提供唯一命令族。`python -m agentic_arxiv` 调用同一 CLI，供模块启动和分布式 launcher 使用。旧 `AgenticArxiv.*` 导入、目录内模块调用和直接脚本入口退出，不提供转发 shim。

本文是当前迁移说明，不是模型实验报告。此前的 [mainline_migration.md](mainline_migration.md)、历史报告、manifest 和归档 OpenSpec 保留原文；其中的旧路径、命令和哈希描述当时来源，不能作为当前调用方式。

## 安装分层

从仓库安装时：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
agentic-arxiv --help
python -m agentic_arxiv --help

# 只按实际能力选择，不要求全部安装
python -m pip install '.[local]'
python -m pip install '.[train]'
python -m pip install '.[translate]'
python -m pip install '.[logging]'
python -m pip install '.[dev]'
```

开发修改可使用 `python -m pip install -e '.[dev]'`；编辑式安装检查不能代替正常 wheel 安装检查。安装仅产生软件环境和构建产物，不下载研究模型、数据或快照。

| 安装层 | 能力与边界 |
| --- | --- |
| 基础 | `arxiv`、`requests`、`python-dotenv`、`loguru`、`pydantic`、PyMuPDF、Pillow；API 推理、工具声明和历史轨迹评分，不强制本地模型或训练栈 |
| `local` | `torch`、`transformers`、`accelerate`、PEFT；现有本地推理、默认 LoRA 适配器产物及按需模型工具，不包含训练器 |
| `train` | 包含 `local` 的同版本约束，以及 TRL、datasets、bitsandbytes 等训练直接依赖 |
| `translate` | 真实 PDF 翻译的 `pdf2zh` |
| `logging` | TensorBoard；wandb 保留显式按需安装，不自动启用 |
| `dev` | 测试、覆盖率、lint、包构建与元数据检查，不隐含 `train` |

依赖版本约束以包元数据为准，迁移不升级或降级训练栈，也不放松 TRL 多轮 rollout、QLoRA、GPU 与精度保护。全部 10 个工具始终注册；基础安装不承诺执行真实翻译或本地 VLM。只有请求相应能力时才检查可选依赖，并给出明确失败与安装指引，不能伪造成功或减少工具声明。

帮助只读取参数声明及必要的包内默认配置，不加载模型、训练器或研究资产，不读取用户 `.env`，不写日志。API 使用时显式设置 `LLM_BASE_URL`、`LLM_API_KEY`、`MODEL`；实际启动操作才从工作区读取 `.env`，已有环境变量仍优先，不保存凭据。

## 命令迁移表

下表覆盖此次设计中的全部活动操作。旧列仅用于定位来源，不表示仍可运行。详细参数使用新命令的 `--help`；参数名统一为连字符形式，不能继续传入 Fire 的下划线旗标。

| 原操作来源 | 当前受支持入口 |
| --- | --- |
| `AgenticArxiv/main.py` | `agentic-arxiv run` |
| `AgenticArxiv/rl/rollout.py` | `agentic-arxiv evaluate rollout` |
| `AgenticArxiv/benchmark/run_benchmark.py` | `agentic-arxiv evaluate run` |
| `AgenticArxiv/benchmark/run_baselines.py` | `agentic-arxiv evaluate baselines` |
| `AgenticArxiv/benchmark/rescore_traces.py` | `agentic-arxiv evaluate rescore` |
| `eval/badcase_replay.py` 的默认/replay、capture、run | `agentic-arxiv evaluate badcases replay`、`capture`、`run` |
| `AgenticArxiv/rl/train_sft.py`、`train_grpo.py` | `agentic-arxiv train sft`、`train grpo` |
| `AgenticArxiv/rl/build_snapshot.py`、`backfill_figure_analysis.py` | `agentic-arxiv data snapshot`、`data backfill-figures` |
| `scripts/generate_sft_data.py`、`generate_parametric_sft_data.py` | `agentic-arxiv data sft generate`、`generate-parametric` |
| `scripts/augment_sft_data.py`、`augment_parametric_sft_data.py` | `agentic-arxiv data sft augment`、`augment-parametric` |
| `scripts/build_sft_train_mix.py` | `agentic-arxiv data sft mix` |
| `scripts/build_grpo_signal_split.py` | `agentic-arxiv data grpo select-signal` |
| `scripts/audit_grpo_rollouts.py`、`analyze_curriculum.py` | `agentic-arxiv audit rollouts`、`audit curriculum` |
| `scripts/freeze_mainline_assets.py` | `agentic-arxiv audit assets` |

普通 `run` 保留 API/live 默认；`evaluate rollout` 保留离线工具默认，显式关闭离线才使用 live。模型后端与工具模式是两个独立选择：API 模型也可使用快照，本地模型也可使用真实工具。坏例 `replay` 和历史 `rescore` 只读已记录轨迹，不执行模型、初始化或工具；坏例 `run` 才是实际完整 episode。资产核对默认只读，原冻结能力需显式选择，并拒绝覆盖原基线。

所有未来生成、录制、翻译、训练和模型评测示例都不是本次验收步骤。不会为了命令帮助或安装验证运行数据生产、基线轨迹生成、API 请求、真实 tokenizer/模型或训练。

## 工作区、默认配置与路径优先级

全局 `--workspace PATH` 默认当前工作目录。安装后的命令可从非源码目录使用；相对输入、显式配置、模型、快照和输出路径均相对于工作区，绝对路径不改写。例如，在不同目录使用同一绝对工作区及同一相对输入，会解析到同一文件：

```bash
agentic-arxiv --workspace /path/to/AgenticArXiv-RL train grpo --help
```

工作区不是安装位置。包不包含 `data/`、模型、轨迹、实验报告、历史基线或输出缓存；不写入 site-packages，也不自动寻找其他仓库或联网补齐缺失资产。缺少输入时错误应包含实际解析路径。快照缺失不能回退 live，指定本地模型缺失不能替换成 API 或其他模型。

原活动 `configs/training/mainline.json` 按原字节迁为 `src/agentic_arxiv/resources/mainline.json`，通过资源 API 加载，只有一个可编辑默认来源。默认配置不依赖工作区文件是否存在。优先级为：包内默认配置 < 显式 `--config PATH` < CLI 中显式提供的字段。未提供 CLI 字段不得覆盖配置；GRPO 五个库委托字段仍在值为 `None` 时省略向 TRL 传参，不猜填其有效值。

`configs/accelerate/ddp_2gpu.yaml` 和 `fsdp_2gpu.yaml` 内容保持不变，仍由 launcher 显式选择。旧操作的模型、快照和输出逻辑子路径保留；原依赖源码层级定位的工具缓存改从工作区定位，episode 自有产物仍位于独立临时空间。这是允许的路径归属变化，不是工具业务或训练语义变化。

未被 episode 路径绑定覆盖时，缓存默认位置的明确变化如下；已有缓存不会被搬迁或自动复制，显式环境变量路径仍优先：

| 原源码目录默认 | 当前工作区默认 | 显式环境变量 |
| --- | --- | --- |
| `AgenticArxiv/output/pdf_raw` | `<workspace>/output/pdf_raw` | `PDF_RAW_PATH` |
| `AgenticArxiv/output/pdf_cache.json` | `<workspace>/output/pdf_cache.json` | `PDF_CACHE_PATH` |
| `AgenticArxiv/output/pdf_figures` | `<workspace>/output/pdf_figures` | `PDF_FIGURES_PATH` |
| `AgenticArxiv/output/pdf_translated` | `<workspace>/output/pdf_translated` | `PDF_TRANSLATED_PATH` |
| `AgenticArxiv/output/pdf_translated_log` | `<workspace>/output/pdf_translated_log` | `PDF_TRANSLATED_LOG_PATH` |
| `AgenticArxiv/output/translate_cache.json` | `<workspace>/output/translate_cache.json` | `TRANSLATE_CACHE_PATH` |

这些变量的相对路径也相对工作区，绝对路径不变。显式开启文件日志才写 `<workspace>/output/log.txt`；仅导入或帮助不创建该文件。SFT 原始基座的 Hub 标识是模型来源标识，不把 `Qwen/Qwen2.5-1.5B-Instruct` 当作工作区文件路径；GRPO 仍只消费明确的本地模型/适配器产物，迁移验收不触发 Hub 下载。

## 模块所有权与数据流

| 模块 | 所有权 |
| --- | --- |
| `contracts` | 动作、轨迹类型与纯投影，不读取配置、模板、任务或模型 |
| `config`、`provenance` | 显式配置与来源/轨迹文件读写；不反向查询训练器或任务目录 |
| `state` | 会话 store、上下文绑定和产物路径安全 |
| `tools`、`llm` | 工具声明/业务及按能力加载的 API/本地模型后端 |
| `runtime` | 消息协议、解析、环境和执行循环，只消费已经准备的初始化动作 |
| `tasks` | 任务声明、切分、初始化计划编译及显式参考答案准备 |
| `scoring` | 既有轨迹与已提供参考数据的只读评分，不做工具或文件 I/O |
| `evaluation` | 组合运行与评分、历史诊断、报告和资产核对，不导入训练回调 |
| `training` | SFT/GRPO、共同模型配置、原始 token 适配、回调与停止策略 |
| `data_pipeline` | 可导入的数据生成/增强/混合/快照/信号选择，不反向依赖 CLI |
| `cli` | 参数解析、按子命令组装及交付，不被业务模块导入 |

任务准备和运行的边界如下：

```text
task definition + dependency lookup
    -> tasks.initialization: pure ordered initialization plan
    -> runtime.reset/setup: execute provided actions in an isolated episode
    -> actual visible state + task text -> policy messages -> actual trajectory

same plan + gold steps -> tasks.reference: explicit independent replay episode
    -> reference data

actual trajectory + provided reference data -> scoring -> scores
    -> evaluation report or training adapter
```

计划只包含动作名和参数，保留依赖查找范围、顺序、重复动作和错误条件，不携带标准答案或评分字段。运行时不读取任务目录、不解析任务 ID，不把初始化计入策略 history、奖励动作或 SFT 标签。模型只看到实际过滤后的必要初始状态；任务 ID 仅可用于执行外部审计。

`tasks.reference` 显式用同一初始化计划创建独立 replay episode，得到参考语义目标。参考准备不是评分函数，也不是模型上下文，不与普通执行共享可变状态。历史只读评分不隐式调用它；若历史缺少所需字段，应按原规则保留未知，不补执行。

训练可使用公共模型无关 episode 评测服务，评测不能导入训练 callback。Canary/阶段验证的阈值、停止策略与 writer 仍在训练侧。SFT/GRPO 从 `training/model_setup.py` 共享 QLoRA 和模型检查，不互相导入阶段入口的私有函数。GRPO 原始 token 采样适配仍属于训练，不强行合并为普通文本生成循环。

## 不变项与允许变化

本次仅改变安装、命名空间、模块所有权、命令名和明确的工作区路径规则。以下语义不得因源码搬迁改变：

- 10 工具的名称、业务 schema、能力与注册顺序；86 扩展任务、8 冒烟任务及既有所有版本切分的内容。
- 任务依赖/setup 的动作、顺序、错误条件和重复动作语义；引用解析、状态隔离及共享输入只读边界。
- `react-runtime-v2` / `react-messages-v3`、实际提示词模板内容与哈希、消息角色、解析和终止语义。
- SFT 只监督当前目标 assistant；GRPO 保留 sampled IDs、严格 token prefix、模型 mask=1/环境及插入边界 mask=0、分组数量与归一化责任。
- 奖励公式、计算顺序、课程权重、失败上限和缺失字段处理；训练默认值、显式覆盖、未设置传参及版本/GPU/长度/日志保护。
- live/record/replay 区别、普通运行与 rollout 各自默认模式、结果字段和 episode 清理前的持久交付。

此前 S2 的语义修正、S3 的消息迁移和已有模型路径/任务选择修正不属于本次新增变化。源码哈希改变不代表协议升级，也不能据此把历史模型标成兼容。

## 来源映射与历史保全

一次性迁移证据目录已按用户要求清理，没有覆盖 `artifacts/mainline_baseline/` 或此前研究报告。维护审计仍需使用的 [独立源码映射](../scripts/maintenance/source_map.json) 保留在 `scripts/maintenance/`，记录原路径、当前一个或多个路径，以及原/当前身份；拆分模块可一对多，退出入口注明退出，不能伪造旧路径仍存在或旧哈希等于新哈希。

历史 manifest 的路径、命令字符串、来源哈希和协议字段保持原义。维护审计只通过独立映射定位当前实现，不回填历史字段，不以搬迁为由把原本可核对的参数自动降级为未知。默认配置原/新路径的 SHA256 均为 `3479911b31a7740321283a30c70f929f1fd8835be9b9974573d2c377a894b34c`；fixture 迁移另记录来源与字节一致性，不与研究资产或非活动基线混在一起。

配置与历史记录的只读审计入口位于 `scripts/maintenance/`，使用已安装包、显式工作区和保留的源码映射，可按需重新执行。临时结果已随证据目录清理；新的输出不替换旧配置指纹或原实验报告。

仅下列两个非活动文件按原字节归档，原/新路径和 SHA256 由归档说明记录；不运行、修复旧导入或重新激活为当前验收证据：

| 原路径 | 归档位置 |
| --- | --- |
| `AgenticArxiv/tests/test_mainline_behavior_baseline.py` | `docs/archive/mainline-behavior-baseline/test_mainline_behavior_baseline.py.txt` |
| `artifacts/mainline_baseline/behavior.json` | `docs/archive/mainline-behavior-baseline/behavior.json` |

其他历史数据、模型、快照、manifest、轨迹、报告和归档 OpenSpec 原路径原字节保留。只清理重新确认后的旧前端缓存和空目录白名单；遇到新内容或外部符号链接时保留并报告，不扩大删除范围。

## 验证与未验证边界

验收分别记录源码依赖检查、编辑式安装、正常 wheel 安装、实际基础隔离环境、全部帮助、已有回归输入、只读历史记录和资产哈希。完整开发环境通过或 import mock 不能代替基础安装隔离；全部任务定义保全不等于全部任务执行通过。

本次不验证真实 tokenizer 前缀/EOS 集成、模型加载与行为、梯度/on-policy、训练收敛、CUDA/多卡分布式训练、真实翻译/VLM 或完整 86 任务性能。缺少这些材料不要求补造研究输入，但原本可运行的检查若因迁移失败必须修复。本文不是执行证据；临时工程报告已清理，不把设计说明或历史结果标为新的执行验证。

当前运行说明见 [Python 主线](python_mainline.md)、[运行时契约](react_runtime.md)、[训练与数据指南](rl_building.md)、[评测指南](evaluation.md) 和 [坏例说明](../eval/readme.md)。
