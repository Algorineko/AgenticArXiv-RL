# 文本 ReAct 运行时契约

本文说明共同解析、episode 环境、轨迹与消费者的代码边界，不提供新响应、轨迹或 fixture。当前安装、来源映射和验证边界见 [架构迁移说明](architecture_migration.md)，训练配置见 [主线指南](rl_building.md)。此前的 [迁移记录](mainline_migration.md) 保留历史原文，不代表当前入口。

## 阶段与版本

当前唯一活动协议是 `agentic_arxiv.runtime.react_contract` 的 `RUNTIME_VERSION=react-runtime-v2` / `CONTEXT_VERSION=react-messages-v3`。当前 prompt 模板 SHA256 为 `80d5bd11f77feb3764041b05aea2e8e6dc1abcbb1e07ce0131fe0693ed4717cb`，没有旧单 user 执行开关。本次源码和安装重组不改变这些版本、模板内容或训练消息。

S2 阶段记录保留为历史边界：当时上下文为 `react-single-user-v1`，同步翻译模板 SHA256 为 `a0a17ff6b9d30abd438710831de2f1bddfd6406d52b56ab68758319173a45441`。这些是对应实际 prompt 文本的计算记录，不是 tokenizer 哈希，也不是仍受支持的旧执行实现。

运行时版本、上下文版本与训练算法是不同维度：解析/状态修正不能归因成算法收益，消息迁移也不承诺旧 checkpoint 行为不变。历史记录没有的版本或模板字段保持未知，不回写原件。

## 动作与环境观测

`parse_react_response` 返回 `CALL`、`FINISH` 或 `PARSE_ERROR`，解析不执行工具。

- 工具动作必须包含非空字符串 `name` 和 JSON 对象 `args`；坏 JSON、缺少 Action、非对象参数不会被修复或猜工具。
- 完整 FINISH 忽略首尾空白与大小写；JSON 包装 FINISH 只允许 `args` 缺省或为空对象。FINISH 前缀文本不能代替完整标记。
- 模型生成的 FORCE_STOP、ERROR、PARSE_ERROR 是格式错误，不转换为模型成功结束。
- 模型续写的 Observation 及其后动作不能执行。`policy_text_before_observation` 保留当前响应的原字符前缀，完整 `raw_response` 另外保留用于审计。
- 语法正确的未知工具或非法业务参数由 `AgenticArxivMultiTurnEnv.step` 在执行前拒绝，返回一致错误观测。普通循环允许模型在剩余预算中响应错误，不隐式重试或改参数。
- 会话及内部解析字段只在执行边界注入。策略 args 携带 `session_id` 或下划线前缀内部字段时明确拒绝为执行错误，不能靠隐式丢弃/覆盖后继续调用；这些框架字段也不作为业务评分或 SFT 目标参数。模型原始输出和实际业务动作分开保存。

来源：`runtime/react_contract.py::parse_react_response`、`contracts.py::business_action`、`runtime/multiturn_env.py::step`、`runtime/agent_engine.py::run`，均位于 `src/agentic_arxiv/`。

## 初始化、引用与隔离

每个 episode 持有独立 `MemoryStore` 与 `TemporaryDirectory`。`state/store.py::bind_store` 和 `state/artifact_paths.py` 的上下文绑定把工具指向本 episode，不切换进程全局 store 或修改全局 settings。

`tasks.initialization.compile_initialization(task_def, task_lookup=...)` 在任务侧纯编译依赖和 setup，保留查找范围、动作顺序、重复动作及错误条件，不执行工具。计划只含动作名和参数，不含标准答案。上层将计划显式传给 `reset(initialization_plan=...)` / `setup(plan)`；环境先清理自己的状态与目录，再按计划执行初始化，不查询任务目录或任务 ID。

初始化不进入策略轨迹、奖励工具序列或训练标签。`visible_state` 只描述实际候选论文、最近操作对象和缓存事实，不从 gold steps、expected IDs、task_id 或评分字段构造模型输入。`ReActAgent.run(..., initialization_plan=...)` 同样只消费计划，任务审计身份由上层保留在模型消息之外。

参考答案由 `tasks.reference` 显式创建独立 replay episode，使用同一初始化计划解释 gold steps。参考准备与普通执行不共享可变状态；参考答案不是模型上下文，也不是评分函数的一部分。`scoring` 只消费调用方已经提供的轨迹和参考数据，历史 rescore/坏例 replay 不执行参考准备、初始化或工具。

序号、论文 ID、标题和空引用共用本地解析状态；搜索候选与最近操作对象按实际操作更新。带 `offline_fallback` 的关键词或 recent-search 降级子集不当作命中结果，不覆盖当前候选列表。未知引用返回错误，不临时指向另一篇论文完成任务。recent-search 仅有任意论文池、没有相同领域或通配记录的动态分支无既有输入覆盖，本次只作源码确认。

快照与预取资产只读共享；`reset/close` 只处理 episode 自有空间，不清扫用户产物、其他轨迹或快照构建输出。

保留工具已有的显式搜索输出路径能力：新外部路径是调用者明确选择的输出，不纳入 episode 清理；已存在的外部文件拒绝覆盖。默认搜索输出仍在本 episode 目录中。这一文件所有权边界不能误写成“清理时恢复/删除所有共享目录”。

来源：`runtime/multiturn_env.py::reset/setup/visible_state/step/close`、`tasks/initialization.py`、`tasks/reference.py`、`state/store.py`、`state/artifact_paths.py`。并行/引用的动态结论只覆盖现有可运行输入，不能据代码契约宣称全部动态边界已执行。

## 执行模式与同步翻译

共同环境支持显式 `replay`、`live`、`record`。模式含义与模型后端无关：API 模型也可以使用 replay 工具，本地模型也可以显式执行 live 工具。

| 模式 | 行为 |
| --- | --- |
| replay | 只读已有快照，不联网，不调用真实翻译；缺快照明确失败 |
| live | 真实工具执行，按能力使用网络或本地模型/翻译依赖 |
| record | 明确的真实执行及快照录制，不伪装成离线检查 |

真实 `translate_arxiv_pdf` 同步返回完成或失败，不再只返回待处理任务、依赖应用队列或要求等待推送。后续缓存与译文读取使用相同资产状态。回放下载/翻译只更新 episode 内模拟状态，带 `offline`/`offline_simulation` 来源，不声称产生真实 PDF；失败不能写伪成功。

`prompt_templates.py` 的同步提示词不再指导模型翻译后直接 FINISH：任务要求读译文或查缓存时，应根据工具真实结果继续。该提示词修正属于 S2 语义变化，不是“prompt 没变”的工程等价抽取。

来源：`runtime/multiturn_env.py::__init__/step/_simulate_asset`、`runtime/env.py`、`tools/pdf_translate_tool.py`、`runtime/prompt_templates.py`。

## 轨迹、观测与评分

`make_history_step` 同时保存原始模型输出、合法策略前缀、实际业务动作、完整结构化 `tool_result`、模型可见 Observation、错误、耗时/token 信息及结束原因。

`render_observation` 按稳定字段顺序渲染。默认总正文预算 4000 字符，按字段顺序分配，只局部裁剪明确的正文类字段；保存正文总长/可见长及逐路径裁剪长度，不是每个字段另给 4000。论文 ID、ref 映射、状态、来源和错误不占正文预算，也不因整体字符串切片丢失。不透明字符串没有可靠字段边界时不强行截断。

`project_history/project_result` 让新轨迹使用未裁剪的 `scoring_observation`，历史轨迹没有该字段时沿用原 Observation。奖励公式、五分量权重和失败上限不因此重设计；奖励/指标只消费实际轨迹，不重执行工具、不缺轨迹合成 FINISH、不在回调再做第二次组归一化。

搜索结果的 count/ref/papers 包装只用于模型展示；评分投影不添加引用包装，保持原完整结果结构，空列表仍是空列表。明确 error/FAILED 结果保留失败前缀供旧失败上限判定。该区分避免仅因展示包装出现 `papers` 字段而改变旧结果质量评分；没有既有输入覆盖的空列表等动态边界仍只作源码核对。

共同渲染可能改变可见文本和对旧裁剪行为的评分输入，这些是需要单独记录的 S2 预期差异。历史记录的只读重评分不能证明当前模型或工具重新执行了该轨迹。

## 结束与预算

| 实际原因 | 含义 |
| --- | --- |
| FINISH | 模型请求结束，不自动代表任务成功 |
| PARSE_ERROR | 当前响应不符合动作格式，结束交互且保留错误 |
| RUNTIME_ERROR | 模型请求/运行异常；既有评分投影可使用 ERROR |
| MODEL_END | token 采样路径中模型结束但未给出 FINISH，不表示完成任务 |
| MAX_TURNS | 达到轮数预算；框架记录 FORCE_STOP，不生成模型 FINISH |
| TOKEN_LIMIT | completion 预算耗尽；与 FINISH 分开记录 |

普通完整任务默认允许 6 轮，FINISH 本身占一轮；框架结束记录不是额外模型采样。GRPO 的 `max_completion_length` 是总 completion 序列预算，包含环境插入片段，不包含初始 prompt，因此满足轮数预算并不保证 token 预算足够。

受阻任务仍按最终 Thought 的具体原因及既有任务语义评分，提前或虚假 FINISH 不因格式正确而成功。`run` CLI 的结束码也不能代替正式评测的严格成功指标。

## 唯一消息协议与训练适配

`build_react_messages` 初始 user 只建立一次任务、工具及实际可见 setup；`append_react_turn` 追加原始合法 assistant 前缀和明确标为环境数据的 user Observation。任务边界沿用 TASK_START/TASK_END，后续反馈不伪装成新任务或原生工具角色。消息从 user 开始交替，prompt 以 user 结束，包含当前目标的监督样本以 assistant 结束。

普通本地与 API 推理、专家生成及 GRPO 使用同一逻辑消息构造。旧记录缺 `policy_text` 或 `visible_observation` 时只允许读取/评分，不从 Thought/Action 重建当前协议响应继续运行，也不每轮 flatten 历史为一个 user prompt。

SFT 转换把此前完整消息作为 prompt，仅最后一条当前 assistant 作为 completion，强制 completion-only loss。历史 assistant、初始化和环境 Observation 不计本样本监督；长度与 manifest 保护保留。

GRPO 保留原始采样 token ID，不 decode/re-encode 响应。`_append_environment_tokens` 对聊天模板的完整候选前缀逐 token 比较，只有完全兼容才追加 suffix；环境与运行时消息边界 mask 为 0，原始 policy token 为 1。已采样 EOS 属于原前缀，不重复插入；不兼容模板明确失败。总 completion 预算包含环境片段，首次没有合法 token 失败，后续空结束记录 MODEL_END。G 的采样数量由训练器/适配接口保证，rollout 不再扩增一次。

以上是转换、接线与保护代码的契约，不是实际 tokenizer、EOS 前缀或梯度验证。已有固定 token 输入只支持对应回归，不能替代本次未执行的真实 tokenizer/模型/训练集成。

来源：`runtime/react_contract.py::build_react_messages/append_react_turn/validate_react_messages`、`runtime/agent_engine.py::build_messages`、`data_pipeline/generate_sft_data.py::build_sft_samples_from_history`、`training/train_sft.py::_to_prompt_completion`、`training/grpo_reward.py::_chat_template_ids/_append_environment_tokens/make_multiturn_rollout_func`。

## 元数据与兼容读取

`protocol_metadata` 的目标声明供新数据和训练使用：runtime/context/training_protocol、prompt_template_hash 与实际可得的 chat_template_hash。prompt 哈希来自实际 `REACT_PROMPT_TEMPLATE` UTF-8 文本；chat 哈希来自 tokenizer 实际选定模板，没有 tokenizer 时保留 None。legacy `template_hash` 只读取，不推断其含义或替代新字段。

执行轨迹与报告只把实际运行版本当执行事实。模型的 `training_protocol_metadata` 必须从真实模型 manifest 或明确声明读取，不能自动 stamp 当前 context；API 模型没有声明时仍未知。`protocol_relation` 分别列出 different_fields 和 unknown_fields：存在明确差异为 cross_protocol，无差异但缺字段为 unknown，齐全相符才为 same_protocol。未知或跨协议不是同口径复现，模板哈希也不能证明真实 token 前缀已验证。

新协议训练拒绝旧/未知/不匹配数据或需延续训练的模型产物。SFT 的 `--skip-data-manifest-check` 不跳过协议检查；无 tokenizer 生成的数据可保留未知 chat 哈希，已知哈希必须匹配实际模板。GRPO 的 SFT 起点必须有必要 manifest 及兼容实际 chat 哈希；原始初始化基座的先前训练协议未知时，只记录来源，不宣称其已按当前 ReAct 协议训练。

新数据/模型/报告写入独立协议声明；历史原件、数值与缺字段状态不改。新信号选择继承实际 probe 协议并独立保存来源模型关系，不把旧探测记录 stamp 成当前版本。

来源：`provenance.py::read_protocol_metadata/require_training_protocol/evaluation_protocol_relation`、SFT/GRPO 入口、`training/rollout_audit.py`、`evaluation/report.py` 与 `data_pipeline/build_grpo_signal_split.py`。

## CLI 与消费者

以下帮助命令不执行模型或工具：

```bash
agentic-arxiv run --help
agentic-arxiv evaluate rollout --help
agentic-arxiv evaluate run --help
agentic-arxiv evaluate badcases run --help
```

普通 `run` 默认 API/live；`--offline --snapshot PATH` 显式 replay，`--llm-backend transformers --model PATH` 指定本地模型，默认 `--max-iterations 6`。`evaluate rollout` 默认离线，使用 `--offline` / `--no-offline`，轮数同样为 `--max-iterations 6`；缺指定模型不回退 API。参数统一为连字符形式，不再接受 Fire 的下划线参数。

普通 `run` 会展示各步工具结果及最终 Thought，并在关闭 episode 前将完整 `result.json` 和本次生成文件保存到工作区的 `output/runs/run-*`；`--output-dir PATH` 可指定结果根目录，每次仍使用独立子目录，不覆盖已有结果。JSON 的 `artifact_exports` 将轨迹中原临时文件路径映射到持久化副本；显式写到 episode 之外的工具输出保持原位置，外部共享输入和符号链接不复制。缓存资产时间在结构化轨迹中保存为 ISO 字符串，原可见观测和评分文本不因此变化。交付由 CLI 与 `provenance` 组织，不由评分器写文件。

`evaluate run` 默认 API 模型且未传 offline 时执行 live 工具；正式评测显式使用任务集、版本化切分、已有快照和声明后端。`evaluate badcases replay` 只读既有记录并评分；`evaluate badcases run --model PATH --offline --snapshot PATH --max-iterations 6` 才执行当前模型完整 episode，历史原件不改。后续执行这些运行入口需要相应真实输入，本次没有执行。

安装后无需切换源码子目录或设置 `PYTHONPATH`。全局 `--workspace PATH` 默认当前目录，所有相对输入与输出相对工作区，绝对路径保持不变；工具缓存从工作区解析，不从 site-packages 猜测研究资产。基础安装可查看全部帮助及全部 10 工具声明；实际本地推理、翻译等能力按 [安装分层](architecture_migration.md) 检查依赖，帮助不读取 `.env`、写日志或加载模型。

必须共同接线的消费者包括普通 ReAct、rollout、benchmark、专家/参数化生成、语义 oracle、GRPO token rollout、Canary、StageVerifier 与坏例实际执行。Canary/StageVerifier 用完整 episode；SFT 解析率阈值 0.3 和 GRPO mean reward 阈值 −0.2 保留，解析率、完成率和覆盖任务分别报告，不把健康检查当 86 任务能力证据。

本次证据仅来自源码、现有测试及历史输入读取；实际结果以迁移记录为准。缺失的真实 tokenizer、快照/模型集成、数据生成、训练和完整任务评测均注明本次未验证，不要求补齐，不标为通过，也不列为本次工程遗留待办。
