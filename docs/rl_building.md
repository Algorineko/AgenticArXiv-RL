# SFT → GRPO 与文本 ReAct 主线指南

活动训练主线仅为 **SFT → GRPO**；普通推理、专家数据生成、训练采样与评测使用文本 ReAct。本文列出当前配置契约与后续另行执行的命令，不是要求为本次工程验收准备资产或运行实验。

本次不生成/重生成数据，不新增论文、快照、模型响应、token 序列、轨迹或行为 fixture，不运行任何训练（包括少量步数冒烟），不下载真实 tokenizer，不执行完整 86 任务或模型性能对照。当前安装与来源保全规则见 [架构迁移说明](architecture_migration.md)，此前的 [mainline_migration.md](mainline_migration.md) 保留历史原文。

## 安装与运行边界

以仓库为安装来源创建环境；安装后命令可从其他目录使用，不需源码路径注入：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
# 训练包含本地推理能力；仅本地推理可选择 .[local]
python -m pip install '.[train]'
# 按需安装真实 PDF 翻译与训练曲线依赖
python -m pip install '.[translate,logging]'

agentic-arxiv evaluate rollout --help
agentic-arxiv evaluate run --help
agentic-arxiv train sft --help
agentic-arxiv train grpo --help
agentic-arxiv run --help
```

基础安装包含 API 客户端、日志、PyMuPDF 和 Pillow，不强制本地模型/训练栈；`local` 安装本地推理但不包含训练器，`train` 包含 local 加训练直接依赖。真实翻译选择 `translate` 的 pdf2zh，曲线选择 `logging` 的 TensorBoard 或单独安装 wandb，测试开发选 `dev`（不隐含 train）。全部 10 工具始终注册，只在实际使用能力时检查可选依赖，不启动应用服务或数据库。API 操作显式配置 `LLM_BASE_URL`、`LLM_API_KEY`、`MODEL`，不提交真实 `.env` 或凭据；帮助不读 `.env`、写日志或加载模型。

版本约束见 [pyproject.toml](../pyproject.toml)：local 为 torch≥2.0.0、transformers≥4.45.0、accelerate≥0.25.0、PEFT≥0.17.1，其中 PEFT 也用于加载默认 LoRA 适配器产物；train 使用相同约束并加 TRL≥0.28.0、datasets≥2.14.0、bitsandbytes≥0.45.5。直接使用 `packaging.version` 的 GRPO 版本保护显式声明 packaging，原来仅靠传递安装；不借分层升级、降级或替换训练实现。

离线回放只读已有快照，快照缺失明确失败，不自动转入 live/record。真实翻译是同步工具调用，返回完成或失败；离线翻译记录模拟来源，不声称新建了真实 PDF。工具初始化/依赖任务不计策略步骤，模型只看到过滤后的必要初始状态，不看到标准 steps、任务 ID、预期论文 ID 或评分信息。

普通 `run` 默认 API/live，显式 `--offline --snapshot PATH` 使用快照工具；`evaluate rollout` 默认离线，只有显式 `--no-offline` 才用 live。两者 `--max-iterations` 默认 6，指定本地模型缺失不会改走 API。参数全部使用连字符，布尔配置用明确的正/反开关而不是旧 Fire 参数。解析错误与 FINISH、执行错误与预算截断分别记录，具体契约及历史 S2 差异见 [运行时说明](react_runtime.md)。

全局 `--workspace PATH` 默认当前目录，相对模型、数据、快照、输出和显式配置路径统一相对工作区，绝对路径不改写。安装位置不承担研究资产根目录角色，包不携带/下载这些资产；错误报告实际缺失路径，不回退其他仓库或替代模型。

## 小型主线配置与默认值

原 `configs/training/mainline.json` 按原字节迁入唯一包资源 `resources/mainline.json`，由 `agentic_arxiv.config` 通过资源 API 读取。省略 `--config` 即使用包内默认，显式 `--config PATH` 使用工作区路径；CLI 中显式提供的字段再覆盖配置，未提供字段不能覆盖配置。配置来源及实际覆盖记录在 manifest。它不是新配方框架，也不还原仓库外历史有效值。

| 字段 | SFT | GRPO |
| --- | --- | --- |
| 模型/交接 | `Qwen/Qwen2.5-1.5B-Instruct` | `outputs/sft_qlora/final` |
| 输出根 | `outputs/sft_qlora` | `outputs/grpo` |
| epochs / max_steps | 3 / −1 | 1 / −1 |
| micro batch / grad_accum | 1 / 8 | 4 / 1 |
| lr | `1e-4` | `1e-5` |
| 资源策略 | QLoRA 默认开启 | QLoRA 默认开启 |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 | 16 / 32 / 0.05 |
| optimizer / checkpointing | `paged_adamw_8bit` / true | `paged_adamw_8bit` / true |
| 长度 | `max_length=4096` | `max_completion_length=256`（总 completion 序列预算，含环境插入片段，不含初始 prompt） |
| 采样 | 不适用 | `num_generations=4`、temperature 1.0 |
| beta / reward_curriculum_steps | 不适用 | 0.04 / 30 |
| seed / report_to | 42 / none | 42 / none |
| 完整任务轮数 | 由样本长度保护 | `max_turns=6`，FINISH 占一轮 |

以下 GRPO 字段默认为 `None`，仅显式设置时传入训练配置：`loss_type`、`epsilon_high`、`mask_truncated_completions`、`importance_sampling_level`、`scale_rewards`。省略表示继续委托实际训练库，不表示已知其有效值，不猜填“原始 GRPO”损失或某版本默认值。已经退出的变体选择不再受支持；显式参数不能被兼容层静默过滤。

方差诊断、`allow_zero_variance=False`、Canary 每 50 步、min reward −1.0、patience 3、阶段验证和其他源码保护保留。SFT 阶段可解析率阈值默认 0.3，GRPO mean reward 阈值 −0.2；完整 episode 的解析率、完成率和实际奖励应分别报告，不能用首步正确替代完成任务。

此前模型默认路径、正式任务选择和 6 轮预算修正保留原历史归因，不属于本次安装重组新增行为。当前表的默认值、条件传参和保护不变；历史 `v6_grpo_train.json` 中的记录不是完整有效配方，不用于覆盖本表。

## 显式任务切分

| 输入 | 成员 | 用途 |
| --- | ---: | --- |
| `agentic_arxiv.tasks.tasks` | 8 | 显式冒烟模式 |
| `agentic_arxiv.tasks.tasks_expanded` | 86 | 完整任务定义，不等于训练集 |
| `data/splits/v7_86.json:train` | 55 | 当前扩展集的正式训练范围 |
| `data/splits/v7_86.json:dev` | 8 | 开发，不生成训练数据 |
| `data/splits/v7_86.json:iid_test` | 19 | 同模板留出 |
| `data/splits/v7_86.json:ood_test` | 4 | 未见形态留出 |
| `data/splits/v6_grpo_train.json:rl_train` | 33 | 历史信号子集，来源为旧训练切分 |

正式 GRPO 必须写明 `--task-set expanded --split FILE:train` 或有效训练信号子集 `FILE:rl_train`。不写任务选择不能隐式训练 86 条；`--smoke` 显式使用 8 条冒烟任务。不要用裸 `--split train`、`iid_test` 把历史默认文件误当当前切分。

未来新信号子集只能在声明的 train 中选择并写新文件，不能从 dev/iid/ood 筛任务，不能覆盖原切分或历史 rates。更换模型/协议后旧 rates 不代表当前采样信息，但本次不要求重新测量。

## SFT 直接交给 GRPO

下面命令需已有新协议数据及 manifest、模型、真实 tokenizer、快照、训练依赖和合适资源。本次没有执行。

```bash
# 默认 QLoRA：生成适配器及其基座模型元数据
agentic-arxiv train sft \
  --data data/sft/sft_react_messages_v3_train_mix.jsonl \
  --output-dir outputs/sft_qlora

# 直接加载上述 SFT 产物；无中间训练阶段
agentic-arxiv train grpo \
  --model outputs/sft_qlora/final \
  --task-set expanded --split data/splits/v7_86.json:train \
  --snapshot data/mock_arxiv_snapshot.json --max-turns 6

# 显式全参：输入与输出目录不混用
agentic-arxiv train sft \
  --data data/sft/sft_react_messages_v3_train_mix.jsonl \
  --no-qlora --output-dir outputs/sft_full
agentic-arxiv train grpo \
  --model outputs/sft_full/final --no-qlora \
  --task-set expanded --split data/splits/v7_86.json:train \
  --snapshot data/mock_arxiv_snapshot.json --max-turns 6

# 多卡仍是同一条 GRPO 主线；资源配置不是新算法
accelerate launch --config_file configs/accelerate/ddp_2gpu.yaml \
  --module agentic_arxiv train grpo \
  --model outputs/sft_full/final --no-qlora \
  --task-set expanded --split data/splits/v7_86.json:train \
  --snapshot data/mock_arxiv_snapshot.json --max-turns 6
```

全参目录应包含完整模型及必要元数据；适配器目录按其基座信息加载基座和适配器。缺少指定模型、权重或必要 manifest 时必须在采样前失败，不扫描“最新目录”、退回基座、其他本地模型或远程 API。真实 checkpoint 加载和多卡运行本次未验证。

未来数据/manifest/长度预检可用 `agentic-arxiv train sft --inspect-only`，但仍需实际 tokenizer，不能作为本次必须补资产的检查。日志可显式加 `--report-to tensorboard` 或 `wandb`，后端依赖与 API 凭据按需准备；资源覆盖必须记入运行指纹，不伪装成原配方。SFT/GRPO QLoRA 已增加单机 DDP 路径与独立启动器，四卡配置为 `configs/accelerate/ddp_4gpu.yaml`；全参仍显式使用 `--no-qlora`。QLoRA 分片与 GRPO FSDP/vLLM 不在支持范围内；原有非量化 SFT 分片入口未在本次验证。launcher 的 `--config_file` 属于 accelerate，应用参数仍使用连字符。详见 [多卡训练指南](training_distributed.md)；本次只做工程回归，不启动真实多卡训练。

## 消息、监督与采样责任

当前唯一活动协议为 `react-runtime-v2` / `react-messages-v3`，由 `runtime/react_contract.py::build_react_messages/append_react_turn` 构造初始 user 任务/工具/实际可见 setup，随后 assistant 动作与 user 环境 Observation 交替。任务与工具只建立一次，环境反馈明确标为数据，不能改变协议规则。旧单 user 历史拼接实现和旧执行开关已退出；缺实际 `policy_text/visible_observation` 不猜填旧响应继续运行。

```text
user       task + tools + visible setup
assistant  current Thought / Action
user       environment Observation (data)
assistant  next Thought / Action or FINISH
```

- SFT 每个决策点保存此前完整消息前缀与当前唯一 assistant completion；`training/train_sft.py::_to_prompt_completion` 将 `messages[:-1]` 与 `messages[-1:]` 分开，并强制 `completion_only_loss=True`。历史 assistant、setup 和 Observation 不计本样本 loss，不能只截取第一条 user 消息丢掉历史。
- GRPO 保留独立 token 采样循环及原始采样 ID。模型 token 的 mask 为 1，环境反馈与运行时插入边界的 mask 为 0；不能 decode/re-encode 已采样响应来替代训练输出。
- 后续追加消息必须保持已采样 token 前缀，实际模板不支持时明确失败，不静默重写。已采样结束 token 不重复插入，模型伪造 Observation 不驱动后续工具动作。
- 训练器已完成 prompt 分组/采样数量契约，rollout 不再次按 G 扩增；每条实际轨迹与奖励一对一，回调不再做第二次组归一化。
- FINISH、解析失败、执行错误与轮数/token 截断分开；耗尽预算不合成成功。语法正确但未知工具/非法业务参数返回执行前错误，不调用后端。

`grpo_reward.py::_append_environment_tokens` 按完整原始 token 前缀逐项比较，只追加模板 suffix；EOS 已属于原前缀时不再插一次。`max_completion_length` 包含 policy 与环境片段，环境追加也受剩余预算限制；首轮没有合法采样 token 明确失败，后续模型结束但没有 FINISH 记录 MODEL_END，不当作任务完成。

模板哈希只计算实际文本：`prompt_template_hash` 是当前 `REACT_PROMPT_TEMPLATE` 的 UTF-8 SHA256，`chat_template_hash` 是 tokenizer 实际选定模板的哈希，没有 tokenizer 时为 None。当前 prompt 哈希为 `80d5bd11f77feb3764041b05aea2e8e6dc1abcbb1e07ce0131fe0693ed4717cb`。真实 tokenizer 前缀/EOS 集成、梯度和 on-policy 行为本次未验证；已有测试输入只证明对应代码回归，不替代真实集成。

新 SFT 数据及 manifest 必须声明 runtime/context/training_protocol/prompt hash；无 tokenizer 的数据允许 chat hash 未知，已有明确 chat hash 则必须与实际模板匹配。继续训练的既有产物、适配器和 GRPO 起点还要求真实模型 manifest 及兼容的 chat hash，旧/未知/不匹配输入严格失败，`--skip-data-manifest-check` 不跳过协议检查。显式初始化原始基座可以缺少先前训练协议，只记录其实际来源为未知/未分类，不宣称历史兼容。

执行轨迹的当前 context 不等于来源模型的训练事实。新轨迹/报告分别记录实际执行 `protocol_metadata`、仅从真实来源读取的 `training_protocol_metadata` 和 `protocol_relation`；已知不同字段为 cross_protocol，字段不足为 unknown，齐全相符才是 same_protocol。历史文件不改写，legacy `template_hash` 不猜成 prompt/chat 哈希；跨协议或 unknown 评测不是同口径复现。

## 未来数据生成、扩增与混合

以下流程只在用户另行选择实际实验时执行。输出使用新协议独立文件名，保留全部历史 JSONL、manifest 和模型；不要覆盖历史默认路径。输入需要已有快照及所需正文/译文/图表回放内容，缺失明确失败，不联网补齐，也不从摘要重建。

数据参数不使用训练的 `--config`：以各操作原默认值为基线，显式 CLI 字段覆盖；所有路径再由工作区统一解析。训练的 JSON 优先级不扩展成另一套数据配置机制。以下默认值与 [轻量参数声明](../src/agentic_arxiv/cli/data_arguments.py) 及业务实现对应；默认研究路径不是推荐覆盖目标，实际新实验应显式选新输出。

| 命令（省略 `agentic-arxiv data`） | 既有默认与输入边界 |
| --- | --- |
| `sft generate` | `task-set=basic`、`mode=replay`、不启用 LLM；默认快照 `data/mock_arxiv_snapshot.json`、输出 `data/sft/sft_react_messages_v3_seed.jsonl`；扩展生成必须显式版本化 train |
| `sft generate-parametric` | 切分 `data/splits/v3_81.json`、默认同一快照；默认 `sft_react_messages_v3_parametric_seed.jsonl`，`--include-t5` 时默认 `sft_react_messages_v3_t5_parametric_seed.jsonl`（均在 `data/sft/`）；父任务必须属于 train |
| `sft augment` | `--input`、`--output` 必填，manifest 默认种子旁文件，默认审计切分 `data/splits/v2_62.json`；示例显式覆盖为 v3 |
| `sft augment-parametric` | 输入 `data/sft/sft_react_messages_v3_parametric_seed.jsonl`、输出 `data/sft/sft_react_messages_v3_parametric_linguistic.jsonl`、切分 v3_81、`wrapper-set=base`；targeted 与 base 措辞集合不重叠 |
| `sft mix` | 原始 linguistic + 参数化 linguistic，输出 `data/sft/sft_react_messages_v3_train_mix.jsonl`，`seed=42`；同版本 train 主干与协议须匹配 |
| `grpo select-signal` | probe、source-split、model、output 必填；`num-generations=4`、`seed=43`、`min-informative-groups=2`；仅 `FILE:train` 输入，输出拒绝覆盖 |
| `snapshot` | 默认同一快照路径，原研究方向/关键词列表不变；max-results 50、days 30、content-workers 8、prefetch-budget 900 秒、content-max-ref 0（全部）、translate-max-ref 1；默认固定参考论文、不允许部分失败、不跳过预取 |
| `backfill-figures` | snapshot 必填，默认所有适用论文、`force=False`；`--paper-id` 可重复，`--force` 才覆盖已有图表分析条目 |

样本标识、来源任务/父任务、决策点、manifest kind/protocol/hash/行数、切分路径及去重规则不因包迁移改变。语言变体数不等于独立任务数；缺失历史字段不能从当前默认值推断补齐。

### 1. 显式 train 专家数据

```bash
agentic-arxiv data sft generate \
  --task-set expanded --split data/splits/v3_81.json:train \
  --mode replay --snapshot data/mock_arxiv_snapshot.json \
  --output data/sft/sft_react_messages_v3_seed.jsonl
```

扩展生成拒绝未声明切分、裸 train 或 dev/iid/ood。初始化不写成监督目标，专家执行失败不写伪成功。每行保留来源任务、切分、决策点与样本哈希，新 metadata 记录实际可得的协议/模板信息。

完整混合示例的两支统一声明 `v3_81.json:train`（51 条），与固定参数化父级清单一致，不声称包含 v7 的全部 55 条 train。原始专家单支也可使用当前 v7，另写独立输出；不要把下面的 v7 种子与本例 v3 参数化数据混合：

```bash
agentic-arxiv data sft generate \
  --task-set expanded --split data/splits/v7_86.json:train \
  --mode replay --snapshot data/mock_arxiv_snapshot.json \
  --output data/sft/sft_react_messages_v3_v7_seed.jsonl
```

`--use-llm` 只选择专家策略后端，工具模式仍由 `--mode` 控制；replay 缺快照失败，live/record 是显式真实执行，本次均未运行。

### 2. train 父任务参数化派生

```bash
agentic-arxiv data sft generate-parametric \
  --split-file data/splits/v3_81.json \
  --snapshot data/mock_arxiv_snapshot.json --include-t5 \
  --output data/sft/sft_react_messages_v3_parametric_seed.jsonl
```

参数化清单沿用其父任务约束，T5 选项要求 `v3_81.json:train`，不能把不同版本传进去绕过 parent∈train 守卫。原 v3 train 的 51 条成员均属于 v7 train，此命令的来源仍标为 v3，而不是声称生成了 v7 全部 55 条。父级 ID、派生 ID、参数、标准步骤、文件 SHA256 与切分来源保留。`--only-parents ID,ID` 可显式限制既有 train 父任务，不启用已退出的自进化编排。

### 3. 保语义语言扩增与审计

```bash
agentic-arxiv data sft augment \
  --input data/sft/sft_react_messages_v3_seed.jsonl \
  --split-file data/splits/v3_81.json \
  --output data/sft/sft_react_messages_v3_linguistic.jsonl
agentic-arxiv data sft augment-parametric \
  --input data/sft/sft_react_messages_v3_parametric_seed.jsonl \
  --split-file data/splits/v3_81.json \
  --output data/sft/sft_react_messages_v3_parametric_linguistic.jsonl
```

扩增只改声明允许的任务/Thought 语言，不改动作参数、先前历史或环境结果；核对来源、父任务、留出隔离、行数、唯一消息指纹与 manifest SHA256。语言变体行数不是独立任务数。新协议不把旧两消息数据静默标成兼容输入。

### 4. 混合与冻结

```bash
agentic-arxiv data sft mix \
  --original data/sft/sft_react_messages_v3_linguistic.jsonl \
  --parametric data/sft/sft_react_messages_v3_parametric_linguistic.jsonl \
  --seed 42 --output data/sft/sft_react_messages_v3_train_mix.jsonl
```

混合核对两来源 kind/manifest/文件哈希/实际行数、声明的同一版本化 train 主干、协议与已知 chat hash 冲突、跨来源重复消息和任务 ID 冲突，保留来源密度并确定性 shuffle。训练前再核对数据 manifest、协议、样本长度和 completion 标签；不得用跳过校验的运行当作正式可追溯实验。所有生成/扩增/混合入口发现输出或其 manifest 已存在时拒绝覆盖。

这些命令仅与保留源码的参数/保护对应。本次没有输出上述 JSONL、manifest、翻译结果或训练产物，未来输入要求不是本次未完成事项。

### 5. 快照与图表补录

快照构建是显式网络/真实工具录制，不是历史只读评分。图表补录会读取 PDF 并按既有本地 VLM 后端产生分析，因此两者本次只检查帮助及现有回归，不运行。未来另行授权时使用新快照路径，不能覆盖现有研究快照：

```bash
agentic-arxiv data snapshot --help
agentic-arxiv data backfill-figures --help

# 未来录制：真实翻译需 translate；明确的新文件，不是本次验收命令
agentic-arxiv data snapshot \
  --snapshot data/snapshots/NEW_EXPERIMENT.json \
  --max-results 50 --days 30 --translate-max-ref 1

# 未来补录：对上述独立快照操作；本地图表模型需 local 及已有权重
agentic-arxiv data backfill-figures \
  --snapshot data/snapshots/NEW_EXPERIMENT.json
```

`NEW_EXPERIMENT` 是未来用户选择的新文件名。replay 使用已有内容，不偷偷 record；正文/译文/图表缺失不能联网补齐或从摘要重建。补录未指定 `--force` 时保留已有条目，显式覆盖也只应用于所选新实验快照。

## 未来评测与信号审计

```bash
# 新协议模型对照：每个模型分别运行同一声明切分和参数
agentic-arxiv evaluate run \
  --backend transformers --model outputs/grpo/final \
  --task-set expanded --split data/splits/v7_86.json:iid_test \
  --offline --snapshot data/mock_arxiv_snapshot.json \
  --repeat 3 --seed 45 --save-traces \
  --output artifacts/eval_react_messages_iid

# 只读已有采样记录，审计原始输出/方差/异常
agentic-arxiv audit rollouts outputs/grpo/rollout_traces.jsonl \
  --summary-out outputs/audits/rollout_audit.json

# 仅从已有 train 探测汇总创建新的信号选择文件，不覆盖 v6/v7
agentic-arxiv data grpo select-signal \
  --probe outputs/grpo_probe/reward_probe_summary.json \
  --source-split data/splits/v7_86.json:train \
  --model outputs/sft_qlora/final \
  --output data/splits/react_messages_signal.json
```

dev、iid_test、ood_test 分别用同一显式文件及协议评测；旧 checkpoint 与新评测协议必须记录兼容关系，未知不能当兼容。历史报告读取、坏例只重评分和当前模型实际执行是不同证据。参数、报告及重评分入口见 [评测指南](evaluation.md) 和 [坏例说明](../eval/readme.md)。

## 运行记录与退出范围

每次未来实际训练由 `provenance.py` 记录源码基线、请求配置、实际可得的有效/派生值、版本、输入指纹、模型/数据/切分来源、资源覆盖、协议、实际步数与停止原因。无法取得的值标未知，不把 CLI null 写成已解析的有效算法值，不保存凭据或敏感环境变量。本次实现写入能力，没有为验收生成训练 manifest。

共同 QLoRA、参数统计和版本/精度检查归 `training/model_setup.py` 及相邻公共模块；SFT/GRPO 不互相导入命令入口私有函数。Canary/阶段验证的公共 episode 执行归 `evaluation`，阈值、停止策略和审计 writer 留在 `training`。评分归 `scoring`，只消费实际轨迹与已准备参考，不重新执行工具。任务初始化由 `tasks` 纯编译、`runtime` 执行，参考在独立 replay episode 显式准备，不进入模型消息或学习标签。

配置只读审计位于 `scripts/maintenance/audit_mainline_configuration.py`，历史记录审计位于 `scripts/maintenance/audit_historical_records.py`；它们使用已安装包、显式工作区与 [独立源码映射](../scripts/maintenance/source_map.json)，不覆写历史 source_configuration 或旧报告。新结果可写入独立 `outputs/audits/`；临时迁移证据目录已清理，旧路径和旧指纹继续描述原来源，不能回填成当前值。

活动入口已删除 DPO/OPD/PPO、GRPO 变体选择、动态重采样、自进化编排、独立 VLM 后训练、Jev、MCP/Skill CLI、Web/HTTP、MySQL/SSE。共享 IO、奖励、精度、PDF/图像、外部 LLM API、图表推理、快照构建和 T5 backfill 保留。历史算法结果见 [历史范式对照](rl_paradigm_comparison.md)，不再据旧旗标提供活动运行方式。

可运行帮助、解析、源码/现有测试/历史记录与资产核对按各自实际证据验收。缺少模型、数据、默认快照或真实 tokenizer 的动态路径明确为本次未验证，不补输入、不标通过，也不改列为工程遗留必做项。

