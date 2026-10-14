# SFT → GRPO 文本 ReAct 主线迁移记录

## 范围与阶段

变更：`simplify-sft-grpo-react-mainline`。本次只依赖当前仓库实施工程重构，不执行数据生成、训练、模型性能对照、完整 86 任务回放或真实 tokenizer 集成，不补齐外部研究资产，不新增合成论文、响应、token 序列、快照、轨迹或行为 fixture。

| 阶段 | 改动边界 | 证据边界 |
| --- | --- | --- |
| S0 | 当前源码、资产、检查和历史记录索引 | 清单/哈希、现有测试、只读历史记录 |
| S1 | 退出算法、路由与服务，主线配置/模型交接 | 保全显式算法参数与省略语义；CLI 路径/任务/预算修正单列 |
| S2 | 共同解析、每 episode 环境、同步翻译和轨迹 | 运行时预期差异逐项记录，奖励公式不变 |
| S3 | 共同多消息上下文、SFT 标签、GRPO token/mask 和协议版本 | 代码与已有输入核对；不产生新数据或声明真实集成通过 |

外部资产依赖项是本次未验证范围，不是遗留必做项。当前可运行检查出现的重构回归必须修复，不能归为缺资产。

## S0 仓库起点

实施起点源码：`ad676bfb6dd51d0763b37e7fa081e663fc3af060`。设计探索版本 `774846e205e79d9054a67b144325fa436fa87b8b` 不是本次实施起点。

起始未提交文件：`AgenticArxiv/tests/test_mainline_assets.py`、`AgenticArxiv/tests/test_mainline_behavior_baseline.py`、`scripts/freeze_mainline_assets.py` 及 `artifacts/mainline_baseline/` 下的资产/行为/旧配方/输入记录。保留这些已有工作，不覆盖或删除。旧 `recipe_evidence.json` 和 `input_inventory.json` 中基于修订前规划的阻塞结论不再是本次关口；原记录保持原样。旧 `behavior.json` 与 `test_mainline_behavior_baseline.py` 包含合成输入，本次不运行、不扩充、不作为验收证据。

### 配置事实

详细字段、默认值、传参条件、源码哈希和行号见 [`source_configuration.json`](../artifacts/mainline_baseline/source_configuration.json)。它记录 14 份源码、69 个 main/CLI 参数和 312 条定位引用。

- SFT 当前源码：`learning_rate=1e-4`、`batch_size=1`、`gradient_accumulation_steps=8`、`epochs=3`。
- GRPO 当前源码：`learning_rate=1e-5`、`beta=0.04`、`reward_curriculum_steps=30`、`num_generations=4`、`batch_size=4`、`gradient_accumulation_steps=1`、`temperature=1.0`、`max_completion_length=256`。
- `loss_type`、`importance_sampling_level`、`scale_rewards`、`epsilon_high`、`mask_truncated_completions` 原为 `None` 时不传入 `GRPOConfig`。这是库委托边界，不是已解析有效值。
- 零方差 guard 原默认开启，`patience=5`、`grace_steps=20`；Canary 间隔 50，默认 `min_canary_reward=-1` 禁用该阈值停止。
- 原 SFT 输出、GRPO 默认模型、任务选择和 4 轮预算不对齐，后续修正不归为算法配置等价。

`data/splits/v6_grpo_train.json:formal_run` 是已有历史摘要，不能替代完整训练 manifest。历史有效损失、依赖版本和其他未提供字段保持未知；本次不宣称历史配方等价。

### 工具、任务与切分

[`assets.json`](../artifacts/mainline_baseline/assets.json) 保存 10 工具的业务 schema/注册顺序、86 完整 TaskSpec、8 冒烟任务及 9 份切分的规范化和原文件哈希。规范化总 SHA256：`751b45441cb88cc59b92d2a26e2e6f4633a76b30e34339653169bc163a41cd12`。框架 `session_id` 不属于业务 schema，依据设计和原执行参数复制注入契约单独剔除。

实际起始检查：`.venv/bin/python scripts/freeze_mainline_assets.py` 通过；`test_mainline_assets.py` 8/8、`test_splits.py` 68/68 通过。`v6` 的 33 条 `rl_train` 来自 `v3:train`；`v7` 的 train/dev/iid_test/ood_test 为 55/8/19/4，无交叉污染。任务定义与切分 `git diff` 为空。这只证明定义和成员保全，不是 86 任务运行证据。

### 已有记录的只读核对

[`historical_records.json`](../artifacts/mainline_baseline/historical_records.json) 的命令为 `.venv/bin/python scripts/audit_historical_records.py --json`。索引 299 个已有文件、53 个 traces 文件、1865 条记录、5075 个步骤；直接读取原动作，不补造模型原文、快照或新轨迹。重复评分、输入不变及报告投影检查通过，历史文件哈希未变。

1865 条记录的终止、工具路径、工具准确、解析和工具失败指标与历史 details 一致。当前只读评分与历史指标存在 78 条参数分、488 条引用分、55 条虚假完成差异，审计记录明确列出；缺少 semantic oracle 的历史 expected IDs，没有重建它们，当前中性引用分不是正确性证据。历史版本/权重未知，不宣称历史奖励复现。

benchmark 的 `history` 格式与训练 `Trajectory.steps` 不同，训练 loader 对这些原文件不适用；直接 JSON 和 benchmark 报告投影读取通过。旧文件缺少协议、原始模型响应、奖励分解、token 或耗时字段时保持未知，不能补填当前版本。普通 Agent、GRPO 和专家路径既有搜索观测、翻译状态和解析失败差异仅从源码引用记录，未捕获新的脚本化执行。

### 可运行检查与未验证项

检查清单及起始实际命令由 [`check_inventory.md`](../artifacts/mainline_baseline/check_inventory.md) 记录。仓库 `.venv` 没有训练栈，宿主和 bundled Python 没有 pytest；使用已存在本地 uv 缓存的 pytest 9.0.2 临时 `PYTHONPATH`，不联网安装、不改环境依赖。测试产物根使用临时目录，禁用 dotenv、外部网络和 Hugging Face 在线请求。

起始可运行安全分组共 449 项通过；缺 torch 的单用例、缺 TRL 元数据的既有检查和训练栈 collection/帮助失败分别记录为起始依赖限制，而不是未来回归。完整快照、真实 tokenizer、发布 checkpoint、完整训练 manifest 和实际 SFT JSONL 缺口见原输入清单，仅限制动态验证结论，不要求补齐。本次没有调用真实外部 API、下载 PDF、真实翻译或执行训练。

## 阶段结果

### S1 工程清理

独立训练模块、Jev/MCP/Skill CLI、Web/API/数据库/SSE 与服务启停脚本退出活动入口。共享 PDF/翻译/图像推理、日志、外部 LLM API、快照构建、信号诊断及历史数据/模型/报告保留。通用父任务筛选的两个既有断言和普通默认全工具策略断言从退出功能的混合测试迁入核心测试，未扩充输入。前端既有 `node_modules`/`.vite` 缓存保全，原忽略规则迁至根目录。

小型 [`mainline.json`](../configs/training/mainline.json) 接入两训练入口，未知显式训练配置字段不再被兼容层静默过滤。五项 `None` 条件传参保全；显式优化参数、方差诊断、早停 guard 和 G 采样基数保持原语义。独立审查发现的全参 GRPO 新增 dtype 已撤回：全参 GRPO 保持起点 dtype 省略，SFT 原有 dtype 逻辑保持。未获取训练库解析值，不声明历史算法等价。

以下入口修正与算法保全分开：GRPO 默认模型从 DPO 目录改为实际 SFT 默认产物 `outputs/sft_qlora/final`；正式训练要求显式 expanded 任务池及 train/rl_train 切分，冒烟为显式 `--smoke`，不能用它绕过留出集保护；完整任务默认预算改为 6 轮含 FINISH。全参/适配器产物按元数据、类型和必要文件加载，缺失明确失败，不自动换模型。真实 checkpoint 加载未执行。

`rl/run_manifest.py` 与训练入口已接入请求/实际可得解析及派生配置、当前依赖版本、源码基线与运行源码、资源覆盖、数据/模型/切分/快照指纹及停止原因记录。保护器退出和异常保留诊断而不保存虚假成功 final；非有限 loss 明确记录 `invalid_training_loss`。未知元数据继续未知，凭据字段排除。本次未启动训练或生成训练 manifest，只核对记录能力与既有输入。

有限检查与差异记录：

- 配置只读 AST 对照 161 项通过，五项省略条件和 GRPO 全参 dtype 另作源码核对；配置报告见 `artifacts/mainline_baseline/configuration_s1.json`。
- 既有切分及资产 76 项、路由退出核心 93 项、服务退出核心 57 项通过；这些集合存在重叠，不累计为独立用例总数。模型/训练栈缺失项没有计入通过。
- `historical_records_s1.json` 与 [`comparison_s1.json`](../artifacts/mainline_baseline/comparison_s1.json) 核对 299 个已有文件哈希不变。53 份原轨迹的 1865 条记录、5075 步、3210 个原动作 JSON、评分分解/权重哈希、终止、报告投影与 S0 一致。原历史指标差异仍保留，未重执行环境状态，不能声称未提供状态的动态等价。
- 普通入口/rollout/benchmark 的导入、完整 10 工具注册、可运行帮助和 `make -n` 检查通过，未载入已退出服务模块。训练帮助仍因缺 trl/datasets 无法运行；已完成可用的纯配置解析和源码检查，不伪报帮助通过。三语已发布模型表和历史段落原样保留；旧算法设计加历史标识，不作为活动运行指南。

检查边界补充：一次保留图表/T5 单元回归运行了原测试的临时 PDF/快照副本及固定 mock 写回，随后由原测试清理；没有持久新 fixture 或生产快照，也没有真实模型/联网。它超出 S0 预定的纯源码/既有读取分组，本次不将这些临时输出纳入行为基线或工程对比证据，后续不扩大此类构建检查。

日志安全差异：既有 `utils/logger.py` 导入时以 `mode="w"` 打开 `output/log.txt`，起始检查可能已截断该 ignored 本地日志，无法确认或恢复内容。现改为追加，避免继续覆盖。不能据历史资产哈希未变推断这个运行日志未受影响。真实 `.env` 未修改。

### S2 共同执行语义

共同契约为 `agents/react_contract.py`，环境为 `rl/multiturn_env.py`。运行时版本 `react-runtime-v2`；本阶段仍使用 `react-single-user-v1` 的单 user 布局，不把 S3 消息变化混入本阶段。同步翻译提示词模板 SHA256 为 `a0a17ff6b9d30abd438710831de2f1bddfd6406d52b56ab68758319173a45441`，来源是实际 `REACT_PROMPT_TEMPLATE` 文本，不代表真实 tokenizer 或整个动态输入已验证。

| 预期差异 | 当前代码与检查依据 |
| --- | --- |
| 解析 `None`、猜工具和文本修复退出 | 共同解析器区分 CALL/FINISH/PARSE_ERROR；精确 FINISH，框架标记不算模型完成。复用既有 parser/terminal 输入；未有输入覆盖的坏 args/前缀边界以源码审查为据 |
| 观测只能由环境提供 | 共同解析在行首 Observation 边界截断当前动作，原模型文本独立保留；既有伪观测输入回归，不新增模型响应 |
| 全局 store/settings 切换退出 | ContextVar 绑定 episode store 和产物根，reset/close 仅清 owned 目录；直接工具旧测试接口绑定当前上下文，不切换进程全局后端 |
| 初始化及引用状态一致 | 所有入口使用 reset/setup/step；真实候选、last_active、缓存事实生成可见初态，gold/task ID/评分字段留在环境或评测侧。GRPO 在真实 setup 后才编码单 user 初始 prompt，不再凭声明虚称准备成功 |
| 离线 fallback 不污染状态 | replay 缺输入不联网，recent 自动合成与 auto 模式退出，关键词 fallback 明确失败且不替换候选；实际离线状态标记 simulation/snapshot 来源 |
| 翻译同步完成或失败 | live 使用真实同步工具，失败不写伪 READY；离线不启动翻译进程，既有读取/缓存测试核对统一真值，不宣称真实 PDF 已生成 |
| 长观测展示与评分分开 | 确定性可见渲染仅裁总计 4000 字符的明确正文，保留标识/状态/错误与裁剪元数据；完整结果独立保存、未裁原结构投影给旧评分，显示 papers/ref 包装不改变空搜索质量 |
| 终止与成功分开 | FINISH、PARSE_ERROR、RUNTIME_ERROR、MODEL_END、MAX_TURNS、TOKEN_LIMIT 分别审计；框架预算记录映射旧 FORCE_STOP，不计额外模型轮数/token/loss，不伪造 FINISH。普通与 GRPO 预算评分已对齐 |
| 框架参数仅执行时注入 | 模型描述/标签不含 session_id；策略提交 session_id 或私有字段明确拒绝，不静默修复，原工具 schema 不变 |

普通 Agent、rollout、benchmark、专家/参数化生成、semantic oracle、GRPO sampler/reward、Canary、StageVerifier、坏例运行及历史读取均已接共同边界。普通 ReAct 是具体运行器；BaseAgent、SideEffectManager、ContextManager、单轮合成奖励降级和异步 CRUD/schema 删除。旧用户行为 fixture 原样保全但不活动运行。Canary/Verifier 使用完整 episode，SFT parse 阈值 0.3、GRPO reward 阈值 -0.2 未放松；报告区分完整严格完成率和解析率，默认少量冒烟覆盖不是全任务能力证据。

本地客户端现在保存实际原 decode 文本，按可得 stop/EOS/长度事实记录结束原因；未知 EOS 条件不猜。普通入口保留原文并只解析合法边界前动作，provider 的 length 不执行截断动作。当前模型执行字段不被当作旧 checkpoint 的训练协议，模型训练事实仍须读其原 metadata，缺失为未知。

S2 有限检查：

- 中央 [`checks_s2.md`](../artifacts/mainline_baseline/checks_s2.md) 的 31 模块首次 455 passed、4 skipped、1 个既有 torch 用例未选；唯一新业务 subfail 为旧 bootstrap 源码断言，改为实际 require_all_tools 调用后定向 7 项和 2 subtests 通过。当前范围无未解决新增失败。最后初始化和本地终止修正由对应原输入定向回归/源码核对关闭，没有重复扩跑全组。
- GRPO 7 模块原输入 103 passed、1 skipped、63 subtests；环境隔离/引用定向回归通过；普通三入口 83 项通过。集合重叠，不合计为新独立测试总数。真实 tokenizer、Canary/训练 guard 的训练栈动态路径未执行。
- [`assets_s2_check.md`](../artifacts/mainline_baseline/assets_s2_check.md)：规范化总哈希不变，资产 8/8、切分 68/68，10/86/8 和所有原成员保全，定义文件 diff 为空。
- [`comparison_s2.json`](../artifacts/mainline_baseline/comparison_s2.json)：299 个历史文件、53 traces、1865 行和评分/权重哈希与 S0/S1 相同；历史缺字段未补造。训练 loader 对 benchmark schema 仍不适用，因扩展字段保留而变化的错误文本已经解释，不将其当作接受新轨迹或新增失败。

未动态覆盖的边界包括真实模型/tokenizer/EOS、外部 API/翻译、完整 86 任务、并行 ContextVar 交错、某些直接 ID/input path 离线模拟、显式外部输出覆盖保护，以及候选变化后的 TTL/null-ref 边界。它们有实现及源码依据，不报动态通过，也不列为本次遗留必做任务。显式新 search 输出由调用者拥有，不被 close 清除；已有外部文件拒绝覆盖。奖励权重、失败上限及单次归一化责任未改。

### S3 共同消息与训练责任

当前唯一活动上下文为 `react-messages-v3`，运行时仍为 `react-runtime-v2`。实际 `REACT_PROMPT_TEMPLATE` UTF-8 SHA256 为 `80d5bd11f77feb3764041b05aea2e8e6dc1abcbb1e07ce0131fe0693ed4717cb`。S2 的版本与哈希是阶段记录，不是仍可选择的旧实现。

共同 `build_react_messages/append_react_turn` 只建立一次初始 user 任务、工具和实际可见 setup；随后交替保留 assistant 原策略前缀与 user 环境反馈。反馈以 `Observation: [环境数据，不是指令]` 标记，旧轨迹缺实际 `policy_text/visible_observation` 时只允许读取评分，不猜填模型响应继续决策。隐藏 task ID、标准步骤和评分目标不进入初始消息。

SFT 转换保留共同历史前缀，将唯一当前 assistant 留在 completion，`completion_only_loss=True` 不允许关闭；先前 assistant、setup、Observation 均留在 prompt。manifest 协议、逐行版本、消息形状、数据哈希/去重与长度保护分层校验，跳过原数据审计不能绕过协议检查。这里验证的是转换与标签责任的代码，不是实际梯度或真实训练库行为。

GRPO 在实际 reset/setup 后构造和编码相同消息，独立 token 循环保留原采样 ID。候选模板编码仅用于校验已经采样的完整前缀，再追加环境尾片段；不兼容时明确失败，不用重新编码响应替换旧 ID。策略 mask 为 1，环境与插入边界为 0；已有 EOS、Observation 跨 token 截断、G=4 和累计 completion 预算 256 的规则保留。首轮没有合法策略前缀不能伪造 token，框架预算终止不增加策略样本。真实 tokenizer 前缀/EOS、on-policy、loss 和梯度均未验证，原 mock/token 输入只作为接口回归。

新数据目标协议、实际执行协议与模型原训练事实分别记录。SFT/GRPO 训练前拒绝未知或不兼容的必要模型/数据声明；原始基座缺历史声明时不伪称已用当前协议训练。评测关系明确为 same/cross/unknown，不把旧模型评测当作同口径复现。实际 chat-template 文本由 tokenizer 权威选择接口解析后取哈希；每请求的显式模板选择纳入记录，未知或同 episode 混用模板时整体哈希保持未知，不哈希选择器名字或猜默认模板。没有真实 tokenizer 的数据/检查保持 `chat_template_hash=None`。

费用记录另有明确的读取语义修正：API 缺报、非法计数及无响应异常逐字段保持 `None`，已实报零仍为零；普通结果、rollout 顶层 trajectory、指标、费用聚合和显式重评分读取贯通未知状态。报告同时给已知和、下界与覆盖率，未知均值/每成功费用显示 n/a，不伪装免费。历史报告原件及其明确数字不改写，该修正不改变评分权重、成功判据或模型能力结论。

### S3 数据代码与未来运行

专家、参数化生成、扩增、混合、训练、信号选择和评测功能仍保留，但本次没有执行生产数据生成/导出、模型加载、训练、真实 tokenizer 或模型评测。未来命令及所需输入见 [`rl_building.md`](rl_building.md)，消息契约见 [`react_runtime.md`](react_runtime.md)。

- 新数据标识使用 `sft_react_messages_v3_*`，SFT 缺省数据入口为 `data/sft/sft_react_messages_v3_train_mix.jsonl`。这是 S3 协议身份修正，不是算法配方变化；本次未生成该文件。
- 生成只使用声明的 train；父任务、source_task_id/source_split、切分 SHA、完整消息指纹、去重与长度保护保留。语言扩增只改首任务 span 与当前目标 Thought，既有中间策略和反馈不变。
- 文档的混合管线使用同一 `v3_81.json:train` 来源和 SHA，拒绝把不同版本来源静默混合。原 v3 train 的 51 个成员是 v7 train 55 个成员的子集，但不能据此把 v3 参数化输出称作完整 v7 输出；v7 原始生成单独列出。
- 信号选择仅接受 `FILE:train`，验证成员唯一且不与 dev/iid_test/ood_test 交叉；在 probe/模型访问前拒绝已有输出，最终独占创建新选择文件。原 v6 的 33 条历史 rl_train 仍可供 GRPO 显式使用，但不能充作新的信号选择候选 train。没有生成新 probe/选择记录或覆盖原切分。

### S3 实际检查与集成限制

| 证据 | 实际结果与范围 |
| --- | --- |
| [`checks_s3.md`](../artifacts/mainline_baseline/checks_s3.md) | 最后源码冻结后单次中央 31 核心模块及 7 个既有纯 helper：462 passed、470 subtests passed、4 skipped、1 缺 torch 项 deselected；退出 0，无新增失败 |
| [`configuration_s3.json`](../artifacts/mainline_baseline/configuration_s3.json)、[`checks_s3_configuration.md`](../artifacts/mainline_baseline/checks_s3_configuration.md) | 161 项 AST/JSON 配置对照通过；共同协议、唯一 completion、实际模板来源、五项 None 省略、G 和预算的源码核对完成 |
| [`checks_s3_metadata.md`](../artifacts/mainline_baseline/checks_s3_metadata.md) | 原序列化/费用/选择输入及 63 份原元数据、51 份原报告只读检查通过；未知字段不补造，原明确值与文件哈希不变；保护分支与真实集成分列 |
| [`historical_records_s3.json`](../artifacts/mainline_baseline/historical_records_s3.json)、[`comparison_s3.json`](../artifacts/mainline_baseline/comparison_s3.json) | 单次读取 299 文件、53 traces、1865 行、5075 步和 3210 原 JSON 动作；与 S0/S1/S2 文件、解析、评分分解/权重哈希、终止及投影一致，错误列表为空 |
| [`assets_final_check.md`](../artifacts/mainline_baseline/assets_final_check.md) | 最终规范化总哈希不变；10/86/8、9 份切分及历史 tracked 资产无变化；未覆盖 S0 基线 |
| 最终源码及规格检查 | 使用 `git ls-files --cached --others --exclude-standard` 的现存 Python 文件列表，经 `ast.parse` 核对 135 文件语法；对 134 活动文件的 `ast.Import/ImportFrom` 核对已删除模块引用为 0；旧用户行为测试只读语法但不活动执行。`git diff --check` 与 `openspec validate simplify-sft-grpo-react-mainline --strict` 均退出 0 |

上述用例集合与阶段定向检查有重叠，不累计成更大的独立测试总数。中央检查后的最后审查发现重评分报告未传入原 summary 的可得协议声明，现已直接沿用原声明，而非 stamp 当前协议；只定向重跑 CLI/费用 14 项及 6 subtests，通过，不重复全组或生产重评分。额外的原输入标签/长度、GRPO、消息和 verifier 辅助检查只证明对应代码；缺训练依赖的模块没有通过动态 collection。4 个 skip 和 1 个 deselect 的原因见中央报告，没有归入通过项。

S3 新增历史用量统计中，1865 条原 history 都缺 token_usage，因此当前只读投影为未知、覆盖率 0；已知和 0 仅为空集汇总，不是已测量零消耗。原 summary 中明确的 avg_tokens 原样核对保全。S0–S2 没有保存这项用量聚合，不能伪称对旧聚合值进行了动态重测。原 78/488/55 条参数/引用/虚假完成差异仍按 S0 边界保留；loader 对 benchmark schema 仍拒绝，S3 与 S2 错误文本一致，没有新增不兼容。

## 全部消费者的最终调用链

| 消费者 | 当前共同边界及本次证据 |
| --- | --- |
| 普通 main、本地/外部 API、rollout、benchmark | `ReActAgent.run → env.reset/setup → build_react_messages → llm_client → parse_react_response → env.step → make_history_step`；原接口/隔离/CLI 回归，真实 API 与模型未执行 |
| 专家、参数化 SFT | `generate_deterministic_trajectories → env.reset/step`，转换调用 `build_react_messages/append_react_turn`；原转换和来源 helper 加源码核对，生产生成未执行 |
| semantic oracle | 使用同一 `AgenticArxivMultiTurnEnv.reset/step` 计算目标；它不调用模型，不需要决策消息；接线源码已核对，本次未运行 oracle 或补快照 |
| GRPO sampler/reward | 实际环境初始化后共同消息/token 追加、共同解析/轨迹，再由 `RewardCalculator` 读取实际结果；缺轨迹明确失败，原 token/mask 测试，不是实际 TRL 集成 |
| Canary、SFT/GRPO StageVerifier、坏例显式 run | 完整 `ReActAgent` episode 及共同评分；原阈值保全，覆盖与协议关系明确；可用原检查和源提取 helper，真实训练栈/模型验证未执行 |
| 历史坏例/报告/轨迹读取 | 只读原记录及共同评分投影，不执行工具、不补响应/协议，不覆盖原分数；S0–S3 历史审计与元数据读取证据 |

## 规格契约证据索引

下表逐项对应三个 delta spec 的 Requirement；“源码”不代表该场景已动态执行。全部源码路径以仓库根为基准，阶段结果与命令由上文证据索引提供。

| 规格契约 | 实现与证据 | 本次未验证或边界 |
| --- | --- | --- |
| 训练：受支持的训练与运行入口 | S1 删除清单、CLI/文档、最终活动 import AST 检查 | 训练帮助受训练栈缺失限制，不伪报可运行 |
| 训练：仓库内配置基线与传参保全 | source_configuration、configuration_s1/s3 的 161 项及 mainline.json | 库委托有效值与历史配方未知；路径/切分/预算修正单列 |
| 训练：运行配置可追溯 | run_manifest 与训练写入调用、元数据检查 | 未启动训练或创建训练产物，真实权重/Hub revision 未取得 |
| 训练：SFT 到 GRPO 直接模型交接 | train_grpo 加载类型/基座/文件/协议 guard，原输入和源码核对 | 真实全参/PEFT checkpoint 加载未执行 |
| 训练：依赖与工具能力分离 | requirements/Makefile/CI、普通入口导入与 10 工具注册 | 真实翻译/PDF/VLM/API 和可选后端未执行 |
| 运行时：明确动作与解析结果 | react_contract.parse_react_response、原 parser/terminal 输入及历史动作 | 没有现成响应覆盖的 JSON/前缀分支仅源码审查 |
| 运行时：环境独占观测来源 | policy_text_before_observation、原伪观测输入、实际结果字段 | 真实模型 stop 与 EOS 集成未验证 |
| 运行时：一致初始化与转移 | multiturn_env.reset/setup/step、原引用/约束/状态检查 | TTL/null-ref 等未有现成输入的动态边界不报通过 |
| 运行时：显式离线/在线模式 | env replay/live/record、缺快照 guard、同步翻译调用 | 真实联网/翻译、完整默认快照与录制未执行 |
| 运行时：每轨迹隔离 | store/artifact_paths ContextVar、owned 目录、原 session/sandbox 检查 | 并发交错和部分显式外部输出路径只作源码核对；日志例外已披露 |
| 运行时：稳定观测与审计轨迹 | render_observation/make_history_step/Trajectory、原正文/错误/轨迹输入 | 不透明字符串不盲裁；新结构不是新模型执行证据 |
| 运行时：结束原因与成功分离 | terminal 类型/评分投影、false-finish/blocked/预算原检查 | 6 轮定义和原 mock 不能证明真实最长任务完成 |
| 运行时：版本化上下文与训练责任 | 共同 builder、SFT sole completion、GRPO 原 ID/mask/前缀 guard、S3 版本校验 | 真实 tokenizer/EOS/on-policy/梯度与新数据训练未验证 |
| 验收：工具与任务完整 | S0/final 规范化、原资产与 splits 测试 | 86 条定义保全不是 86 任务执行 |
| 验收：切分与信号隔离 | 原切分/选择 helper、源 SHA、train 与独占输出 guard | 未生成 SFT/信号子集；坏交叉/重复分支仅源码核对 |
| 验收：共同评分与实际轨迹 | reward/project_result、缺轨迹失败、原 GRPO/评分与历史哈希 | 历史只有记录，没有缺失 oracle 状态，不补造历史真值 |
| 验收：多轮验证与覆盖声明 | Canary/Verifier/benchmark/坏例完整 episode 接线、原 guard/报告 | 实际全模型/全任务验证未执行，阈值未放松 |
| 验收：分阶段可归因 | 本文 S0/S1/S2/S3、三份 comparison 和阶段 checks | 有限输入对比不扩展为历史配方或全动态等价 |
| 验收：仅仓库输入且不造数 | 输入索引、原 test diff 检查、旧合成材料排除、新增文件边界审核 | 保留原 mock 属性；一次临时 T5/图表单元输出边界已单列 |
| 验收：无需训练或完整运行 | 本次未执行流程清单、未来命令与依赖限制 | 外部资产验证不是本次剩余实施任务，也不是已通过 |
| 验收：研究产物与证据诚实保全 | 299 历史哈希、原报告明确数字、未知元数据、起始改动保全 | 不能宣称 ignored log 未受起始覆盖写入影响；未隐瞒或删除原证据 |

## 工程验收结论

S1 主线清理、S2 共同执行语义、S3 消息/标签/token-mask/版本迁移及范围内可运行检查齐备，当前没有未解决的重构新增测试失败。任务完成只覆盖工程实现和上述已执行检查，不包含历史有效配方等价、真实 tokenizer/模型加载/训练库集成、完整 86 任务运行、SFT/GRPO 训练、收敛性或模型性能。

本次没有生成/重生成生产数据、补齐外部资产、运行训练冒烟、提交或推送 Git，也没有归档 OpenSpec 变更。研究输入与报告保全结论以哈希及读取范围为准；既有 logger 可能截断 ignored 日志的限制仍单独保留，不以工程完成掩盖。后续数据生成、训练及评测是另行授权的研究执行，不是本变更遗留必做项。

## 归档后的审查修复（2026-10-14）

变更随后已按用户选择同步主规格并归档；上述阶段记录及源码指纹保留为当时证据，不回写为后续修复的检查结果。

- 共同轨迹边界将工具结构化结果中的日期时间递归转换为 ISO 字符串，修复缓存查询结果导致 benchmark/rollout JSON 序列化失败的问题；不修改工具原对象、模型可见观测或评分文本。
- 普通 CLI 在清理 episode 前展示工具结果与最终 Thought，向独立 `output/runs/run-*` 目录保存完整结果和本次产物；可用 `--output-dir` 指定根目录。输出保留原轨迹与导出路径映射，不覆盖已有结果，不复制共享输入链接。
- 复用原环境缓存结果、原模型响应和原文本扩展断言。`test_multiturn_env`、`test_agent_session_leak`、`test_trajectory`、`test_rollout`、`test_benchmark_offline`、`test_terminal_action`、`test_cli_entrypoints` 共 75 项及 8 subtests 通过；仍有既有 Pydantic `dict()` 弃用警告。沿用离线隔离 wrapper，未执行真实模型/API、训练或生产数据生成；临时单元序列化输出不作为新行为基线。
