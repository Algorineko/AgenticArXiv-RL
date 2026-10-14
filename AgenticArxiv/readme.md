# AgenticArxiv Python 主线

此目录是文本 ReAct 运行、SFT/GRPO 训练与研究评测的 Python 包。安装和命令默认从仓库根目录执行，不需要应用服务或数据库。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r AgenticArxiv/requirements.txt
# 按需：真实 PDF 翻译与训练曲线
pip install -r AgenticArxiv/requirements-extra.txt

python -m AgenticArxiv.rl.rollout --help
python -m AgenticArxiv.benchmark.run_benchmark --help
python -m AgenticArxiv.main --help
```

普通 API 推理显式配置 `LLM_BASE_URL`、`LLM_API_KEY` 和 `MODEL`。本地模型由指定路径加载，不应缺模型后静默改用 API。核心依赖保留 PyMuPDF/Pillow；真实翻译需要可选 `pdf2zh`，图表/摘要模型后端按需使用本地模型。

## 模块边界

| 模块 | 作用 |
| --- | --- |
| `agents/agent_engine.py` | 普通 ReAct 运行器，模型只生成当前 Thought/Action |
| `agents/prompt_templates.py` | 任务、工具描述及模型可见初始状态 |
| `tools/bootstrap.py`、`tools/tool_registry.py` | 10 工具完整注册及业务 schema |
| `tools/*.py` | 真实工具能力，内部会话信息在执行边界注入 |
| `models/store.py`、`models/store_memory.py` | 本地会话、候选论文、最近操作与资产状态 |
| `rl/env.py`、`rl/multiturn_env.py` | 快照后端与 episode 执行 |
| `rl/trajectory.py`、`rl/reward.py` | 实际轨迹、终止类型与既有评分 |
| `rl/train_sft.py`、`rl/train_grpo.py` | 仅 SFT/GRPO 训练与直接模型交接 |
| `benchmark/` | 任务定义、切分、语义判定、指标和报告 |

## 工具与状态

10 工具的业务 schema 和注册顺序保持不变。`session_id` 是执行边界的框架参数，不是策略要生成或学习的动作字段。查询、下载、翻译和读取操作围绕当前候选论文及最近操作对象进行引用解析；完整任务允许 6 轮，FINISH 占一轮。

离线回放使用已有快照，不自动联网补请求，也不启动真实翻译。真实翻译同步返回完成或失败，后续缓存与译文读取消费相同状态。`FINISH` 只表示模型请求结束，是否成功由实际轨迹与任务语义决定；格式错误和执行错误分别记录，不合成成功结束。

## 训练、评测与资产

正式训练绑定显式任务集及版本化训练切分，完整 86 条任务不隐式用于训练。默认 QLoRA SFT 产物为 `outputs/sft_qlora/final`，GRPO 直接读取其模型及 manifest；全参和适配器使用明确的加载契约，缺少输入或必要元数据必须报错。

命令、配置默认值、消息协议、mask 与未来数据流程统一见 [主线指南](../docs/rl_building.md)。评测参数及报告说明见 [benchmark 文档](benchmark/readme.md)。

初始化、严格解析、执行模式、同步翻译、轨迹投影和当前 `react-messages-v3` 交替消息契约见 [运行时说明](../docs/react_runtime.md)。S2 语义修正与 S3 消息迁移分别记录，不能把代码契约或历史读取说成真实 tokenizer/模型验证。

## 已退出入口

旧 Web/HTTP、MySQL、SSE、MCP/Skill CLI、Jev 和额外训练算法不是可选兼容服务，已从活动代码退出。这里不再提供服务部署、异步翻译队列或三架构扩展指南。历史源码与研究报告可按原版本查看；历史产物不因旁支清理而删除。

本次仅完成工程检查，不生成训练数据、不运行训练或真实模型评测，不宣称完整任务执行、真实 tokenizer 集成或旧模型在新协议中的行为已验证。检查结果见 [迁移记录](../docs/mainline_migration.md)。
