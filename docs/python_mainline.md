# Python 主线与模块边界

活动 Python 包位于 `src/agentic_arxiv/`，提供文本 ReAct、默认 SFT → GRPO、显式可选 SFT → DPO → GRPO、数据准备与研究评测。旧 `AgenticArxiv.*` 命名空间及直接脚本入口退出，不需要应用服务或数据库。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
# 按需选择本地推理、训练、真实翻译或训练曲线
python -m pip install '.[local]'
python -m pip install '.[train]'
python -m pip install '.[translate,logging]'

agentic-arxiv evaluate rollout --help
agentic-arxiv evaluate run --help
agentic-arxiv run --help
```

普通 API 推理显式配置 `LLM_BASE_URL`、`LLM_API_KEY` 和 `MODEL`。本地模型由指定路径加载，不应缺模型后静默改用 API。基础安装保留 PyMuPDF/Pillow；真实翻译需要可选 `pdf2zh`，图表/摘要模型后端按需使用本地模型。具体依赖与版本见 [pyproject.toml](../pyproject.toml)。

安装后可从任意目录调用，无需 `PYTHONPATH` 或源码目录注入。全局 `--workspace PATH` 默认当前工作目录，相对输入和输出相对工作区、绝对路径不变。包内默认训练配置与研究资产分离；不从 site-packages 找数据、不下载缺失模型或快照。帮助不读取用户 `.env`、写日志或加载训练栈，实际操作才读取工作区配置。

## 模块边界

| 模块 | 作用 |
| --- | --- |
| `runtime/agent_engine.py` | 普通 ReAct 运行器，模型只生成当前 Thought/Action |
| `runtime/prompt_templates.py`、`runtime/react_contract.py` | 提示词、解析与共同消息协议 |
| `tools/bootstrap.py`、`tools/tool_registry.py` | 10 工具完整注册及业务 schema |
| `tools/*.py` | 真实工具能力，内部会话信息在执行边界注入 |
| `state/store.py`、`state/store_memory.py`、`state/artifact_paths.py` | 本地会话、论文状态与产物归属 |
| `runtime/env.py`、`runtime/multiturn_env.py` | 快照后端与已准备 episode 执行，不读取任务目录 |
| `contracts.py`、`provenance.py` | 纯动作/轨迹契约、投影及显式来源记录/文件读写 |
| `llm/` | API 与本地模型后端，按能力延迟加载，不依赖训练入口 |
| `tasks/` | 声明、切分、纯初始化计划编译及独立参考准备 |
| `scoring/` | 轨迹与已提供参考数据的只读奖励/指标，无工具或文件 I/O |
| `training/` | SFT/DPO/GRPO、共同模型配置、token/mask 适配与训练保护 |
| `evaluation/` | 公共 episode 评测、历史诊断、报告与资产检查，不导入训练 callback |
| `data_pipeline/`、`cli/` | 可复用数据准备；独立薄参数解析与组件组装 |

表内模块路径均相对 `src/agentic_arxiv/`；依赖方向与全部命令映射见 [架构迁移说明](architecture_migration.md)。

## 工具与状态

10 工具的业务 schema 和注册顺序保持不变。`session_id` 是执行边界的框架参数，不是策略要生成或学习的动作字段。查询、下载、翻译和读取操作围绕当前候选论文及最近操作对象进行引用解析；完整任务允许 6 轮，FINISH 占一轮。

离线回放使用已有快照，不自动联网补请求，也不启动真实翻译。真实翻译同步返回完成或失败，后续缓存与译文读取消费相同状态。`FINISH` 只表示模型请求结束，是否成功由实际轨迹与任务语义决定；格式错误和执行错误分别记录，不合成成功结束。

## 训练、评测与资产

正式训练绑定显式任务集及版本化训练切分，完整 86 条任务不隐式用于训练。默认 QLoRA SFT 产物为 `outputs/sft_qlora/final`，GRPO 直接读取其模型及 manifest；全参和适配器使用明确的加载契约，缺少输入或必要元数据必须报错。

可选 DPO 使用 `data dpo generate`（local 层）和 `train dpo`（train 层），仅本地当前协议 SFT、已有快照、显式 expanded `FILE:train`；严格比较实际完整 messages，只训练真实未截断当前响应，不补 FINISH。`--inspect-only` 不加载策略权重但需真实 tokenizer，超长不截断。独立冻结参考包含全部 SFT 权重，双模型增加资源代价。默认验证失败保留模型/报告、非零退出，`--no-verify` 为 skipped 而非 PASS；只有显式 `train grpo --model outputs/dpo/final` 才交接，无自动阶段或最新目录选择。默认值及所有实验示例未经本次真实执行，不能当作历史有效配方或模型提升证据。

命令、配置默认值、消息协议、mask 与未来数据流程统一见 [主线指南](rl_building.md)。评测参数及报告说明见 [评测指南](evaluation.md)。

任务侧编译初始化计划，运行时只执行计划；参考答案在独立 replay episode 显式准备，不进入模型上下文，也不由历史评分隐式执行。初始化、严格解析、执行模式、同步翻译、轨迹投影和当前 `react-messages-v3` 交替消息契约见 [运行时说明](react_runtime.md)。S2 语义修正与 S3 消息迁移分别保留历史记录，本次只重组架构，不升级协议，不能把代码契约或历史读取说成真实 tokenizer/模型验证。

## 已退出入口

旧 Web/HTTP、MySQL、SSE、MCP/Skill CLI、Jev、OPD/PPO、GRPO 变体、自进化和独立 VLM 后训练仍退出；本次只恢复统一命令下的可选 DPO，不恢复旧脚本/命名空间。这里不再提供服务部署、异步翻译队列或三架构扩展指南。历史源码与研究报告可按原版本查看；历史产物不因旁支清理而删除。

本次仅完成工程检查，不生成训练数据、不运行训练或真实模型评测，不宣称完整任务执行、真实 tokenizer 集成或旧模型在新协议中的行为已验证。旧报告、数据和 manifest 不改写；旧源码身份通过独立来源映射定位，不能回填历史哈希。检查边界见 [架构迁移说明](architecture_migration.md)，以前的 [迁移记录](mainline_migration.md) 保留原文。
