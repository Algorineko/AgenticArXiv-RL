# AgenticArXiv-RL 工程概览

当前活动主线是 **SFT → GRPO → 文本 ReAct 评测**。普通 CLI、专家数据生成、训练采样和评测使用同一套 10 工具及任务资产，不再选择其他 Agent 架构或启动应用服务。

## 入口与模块

| 入口 | 职责 |
| --- | --- |
| `AgenticArxiv/main.py` | 普通文本 ReAct 推理，保留外部 LLM API |
| `AgenticArxiv/rl/rollout.py` | 执行任务并记录实际轨迹与奖励 |
| `AgenticArxiv/benchmark/run_benchmark.py` | 指定任务集、切分及重复次数的评测 |
| `scripts/generate_sft_data.py` | 按声明的 train 切分执行专家步骤，未来生成 SFT 数据 |
| `scripts/generate_parametric_sft_data.py` | train 父任务的参数化派生与来源审计 |
| `AgenticArxiv/rl/train_sft.py` | SFT，默认 QLoRA，也支持全参 |
| `AgenticArxiv/rl/train_grpo.py` | 直接加载 SFT 产物的多轮 GRPO |
| `AgenticArxiv/rl/reward.py` | 五分量评分、失败上限与诊断 |
| `AgenticArxiv/models/store_memory.py` | 本地会话、论文及 PDF/译文资产状态 |

工具注册来自 `tools/bootstrap.py` 和 `tools/tool_registry.py`。业务能力包括检索、下载、同步翻译、缓存查询、正文读取、总结、抽图、图表分析和译文读取；可选依赖缺失不能让工具注册静默缩水。

## 安装与检查

以下命令从仓库根目录执行：

```bash
make install
source .venv/bin/activate
# 按需安装真实 PDF 翻译和训练曲线后端
make install-extra

python -m AgenticArxiv.rl.rollout --help
python -m AgenticArxiv.benchmark.run_benchmark --help
python -m AgenticArxiv.main --help
python -m pytest AgenticArxiv/tests/
```

测试命令另外需要 pytest，核心运行依赖不包含它；可运行范围与当前环境限制以迁移记录为准，不能把这条示例当作已执行的完整测试结果。

使用 API 后端时，显式配置 `LLM_BASE_URL`、`LLM_API_KEY` 与 `MODEL`；不要提交真实 `.env` 或凭据。使用本地训练模型不要求 API key。真实 PDF 翻译使用可选 `pdf2zh`，PDF/图像处理所需 PyMuPDF 与 Pillow 保留在核心依赖中。

## 任务、切分与研究边界

- 8 条冒烟任务用于显式小范围检查，不代表全部任务执行。
- 86 条扩展 TaskSpec 的身份、文本、标准步骤和初始化定义保留。
- `data/splits/v7_86.json` 为 train 55、dev 8、iid_test 19、ood_test 4；完整任务集不等于训练集。
- `data/splits/v6_grpo_train.json:rl_train` 是历史 33 条信号子集，来源为旧切分，不能报告为 v7 全量训练。
- 历史数据、模型、报告和轨迹保持原样；历史字段缺失保持未知，不补写新协议事实。

正式命令必须明确使用哪份切分。默认 QLoRA 的 SFT 输出为 `outputs/sft_qlora/final`，GRPO 直接加载该目录；全参使用显式输出与模型目录。配置、模型交接、资源策略及未来命令见 [主线指南](docs/rl_building.md)。

当前唯一消息协议为 `react-runtime-v2` / `react-messages-v3`：初始任务后 assistant 与环境 user Observation 交替，SFT 只监督当前 completion，GRPO 保留原采样 ID、严格前缀与 policy/env 掩码边界。旧/未知训练产物不会自动升级成兼容；历史读取与跨协议评测分别标注来源。

## 退出范围

DPO、OPD、PPO、GRPO 变体选择、自进化编排、独立 VLM 后训练、Jev、MCP/Skill CLI、Web/HTTP、MySQL 和 SSE 的活动入口已退出。图表工具的既有推理后端、快照构建、T5 backfill 和外部 LLM API 保留。旧算法对照和架构设计只作历史记录，不是当前运行方式。

本次工程重构不生成数据、不运行训练、不下载研究资产，不执行完整 86 任务、真实 tokenizer 集成或模型性能对照。帮助/源码核对、现有测试、历史记录读取和资产定义核对各自有独立证据，不能互相替代；实际结果与未验证路径见 [迁移记录](docs/mainline_migration.md)。

## 文档

- [三语 README](README.md)：安装、主线命令与历史发布记录。
- [后端说明](AgenticArxiv/readme.md)：文本 ReAct、工具与本地状态。
- [benchmark 说明](AgenticArxiv/benchmark/readme.md)：切分、评分、报告与历史读取。
- [主线指南](docs/rl_building.md)：配置、SFT 到 GRPO 与未来执行边界。
- [坏例说明](eval/readme.md)：既有轨迹重评分与模型执行的区别。
- [运行时契约](docs/react_runtime.md)：初始化、模式、解析、同步翻译及轨迹边界。
