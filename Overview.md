# AgenticArXiv-RL 工程概览

默认活动主线是 **SFT → GRPO → 文本 ReAct 评测**，显式可选 **SFT → DPO → GRPO**。不选择 DPO 不触发中间阶段，GRPO 默认仍为 `outputs/sft_qlora/final`。普通 CLI、数据生成、训练采样和评测使用同一套 10 工具及任务资产，不再选择其他 Agent 架构或启动应用服务。

## 入口与模块

| 入口 | 职责 |
| --- | --- |
| `agentic-arxiv run` | 普通文本 ReAct 推理，保留外部 LLM API |
| `agentic-arxiv evaluate rollout` | 执行任务并记录实际轨迹与奖励 |
| `agentic-arxiv evaluate run` | 指定任务集、切分及重复次数的评测 |
| `agentic-arxiv data sft generate` | 按声明的 train 切分执行专家步骤，未来生成 SFT 数据 |
| `agentic-arxiv data sft generate-parametric` | train 父任务的参数化派生与来源审计 |
| `agentic-arxiv train sft` | SFT，默认 QLoRA，也支持全参 |
| `agentic-arxiv data dpo generate` | 显式本地 SFT + expanded 版本化 train + 已有快照的实际轨迹偏好 |
| `agentic-arxiv train dpo` | 可选 DPO，支持 `--inspect-only`、默认多轮阶段验证 |
| `agentic-arxiv train grpo` | 默认直接加载 SFT；显式 `--model` 可加载当前协议 DPO |
| `agentic_arxiv.scoring` | 五分量评分、失败上限与纯诊断 |
| `agentic_arxiv.state` | 本地会话、论文及 PDF/译文资产状态 |

包位于 `src/agentic_arxiv/`，工具注册来自包内 `tools/bootstrap.py` 和 `tools/tool_registry.py`。业务能力包括检索、下载、同步翻译、缓存查询、正文读取、总结、抽图、图表分析和译文读取；可选依赖缺失不能让工具注册静默缩水。旧导入命名空间、直接脚本入口和路径注入退出，不保留转发层。

`tasks` 纯编译依赖/setup 计划，`runtime` 只执行计划并产生实际可见状态；独立 `tasks.reference` 显式 replay 准备参考答案，参考不进入模型上下文。`scoring` 只消费已有轨迹和已提供参考数据，不执行工具/文件 I/O；`evaluation` 组织评测而不导入训练 callback，`training` 保留停止策略、writer 与原始 token/mask 适配。

## 安装与检查

以下安装命令以仓库目录为安装来源，安装后可从任意目录使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
# 按需安装；dev 不隐含训练依赖
python -m pip install '.[local]'
python -m pip install '.[train]'
python -m pip install '.[translate,logging]'
python -m pip install '.[dev]'

agentic-arxiv evaluate rollout --help
agentic-arxiv evaluate run --help
agentic-arxiv run --help
python -m pytest tests/
```

依赖由 [pyproject.toml](pyproject.toml) 分层声明：基础 API/历史评分、local 本地推理与 DPO 数据生成、train 含 local 的 SFT/DPO/GRPO、translate、logging 和 dev。基础不强制训练栈，local 不强制训练器，dev 不隐含 train；测试需 dev。版本约束和训练保护保留，可运行范围与环境限制以工程报告为准，本文示例未执行，不能当作已执行的完整测试结果。

全局 `--workspace PATH` 默认当前工作目录，相对输入/输出/显式配置相对工作区，绝对路径不变。包内唯一 `resources/mainline.json` 为默认训练配置，显式 `--config` 覆盖默认，显式 CLI 字段再覆盖配置。帮助不读取 `.env`、写日志或加载模型；缺资产不自动查找其他仓库或下载。

使用 API 后端时，显式配置 `LLM_BASE_URL`、`LLM_API_KEY` 与 `MODEL`；不要提交真实 `.env` 或凭据。使用本地训练模型不要求 API key。真实 PDF 翻译使用可选 `pdf2zh`，PDF/图像处理所需 PyMuPDF 与 Pillow 保留在核心依赖中。

## 任务、切分与研究边界

- 8 条冒烟任务用于显式小范围检查，不代表全部任务执行。
- 86 条扩展 TaskSpec 的身份、文本、标准步骤和初始化定义保留。
- `data/splits/v7_86.json` 为 train 55、dev 8、iid_test 19、ood_test 4；完整任务集不等于训练集。
- `data/splits/v6_grpo_train.json:rl_train` 是历史 33 条信号子集，来源为旧切分，不能报告为 v7 全量训练。
- 历史数据、模型、报告和轨迹保持原样；历史字段缺失保持未知，不补写新协议事实。

正式命令必须明确使用哪份切分。默认 QLoRA 的 SFT 输出为 `outputs/sft_qlora/final`，GRPO 直接加载该目录；可选 DPO 生成仅允许显式 expanded `FILE:train`，不接受留出集、裸 train 或历史信号子集。DPO 输出默认 `outputs/dpo/final`，只有显式指定才交给 GRPO；全参使用显式输出与模型目录。配置新增 DPO 节，原 SFT/GRPO 默认不变，旧配置缺该节时仅在内存补默认。配置、模型交接、资源策略及未执行示例见 [主线指南](docs/rl_building.md)。

当前唯一消息协议为 `react-runtime-v2` / `react-messages-v3`：初始任务后 assistant 与环境 user Observation 交替，SFT 只监督当前 completion，DPO 在真实完整上下文完全相同下只比较当前未截断 assistant，不补 FINISH，GRPO 保留原采样 ID、严格前缀与 policy/env 掩码边界。旧/未知训练产物不会自动升级成兼容；历史读取与跨协议评测分别标注来源。

DPO 以完整轨迹奖励排序而非证明单步因果优势，每任务最多一对，报告请求任务/组合两种分母。零对保留轨迹/失败报告并非零退出，低产出只报警。策略与冻结参考来自同一完整 SFT，两份独立模型增加显存；设计默认不是历史有效配方。`--inspect-only` 不加载策略权重，但需要实际本地 tokenizer；长度/模板不兼容失败，不截断。默认完整多轮 Canary 验证含严格成功、解析、虚假完成、终止、覆盖和协议关系，冒烟通过不代表优于 SFT；失败保留模型/报告并非零退出，`--no-verify` 记 skipped 而非 PASS，不自动启动 GRPO。本次未生成数据、训练或模型评测，真实 tokenizer、参考优化后冻结、GPU/DDP 和效果未验证。

## 退出范围

OPD、PPO、GRPO 变体选择、自进化编排、独立 VLM 后训练、Jev、MCP/Skill CLI、Web/HTTP、MySQL 和 SSE 活动入口仍退出。本次只恢复统一命令下的可选 DPO，不恢复旧脚本/导入或旧数据兼容。图表工具的既有推理后端、快照构建、T5 backfill 和外部 LLM API 保留。旧算法对照和架构设计只作历史记录，不是当前运行方式。

本次工程重构不生成数据、不运行训练、不下载研究资产，不执行完整 86 任务、真实 tokenizer 集成或模型性能对照。帮助/源码核对、现有测试、历史记录读取和资产定义核对各自有独立证据，不能互相替代；安装接口、来源保全和未验证边界见 [架构迁移说明](docs/architecture_migration.md)。[旧迁移记录](docs/mainline_migration.md) 保留原文，历史来源字段不回填。

## 文档

- [三语 README](README.md)：安装、主线命令与历史发布记录。
- [Python 主线](docs/python_mainline.md)：文本 ReAct、工具与本地状态。
- [评测指南](docs/evaluation.md)：切分、评分、报告与历史读取。
- [主线指南](docs/rl_building.md)：配置、SFT 到 GRPO 与未来执行边界。
- [坏例说明](eval/readme.md)：既有轨迹重评分与模型执行的区别。
- [运行时契约](docs/react_runtime.md)：初始化、模式、解析、同步翻译及轨迹边界。
- [架构迁移说明](docs/architecture_migration.md)：全部命令映射、安装分层、工作区与历史来源规则。
