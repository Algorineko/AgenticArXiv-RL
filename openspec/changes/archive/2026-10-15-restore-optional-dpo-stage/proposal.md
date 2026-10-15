# Proposal

## Why

DPO 在主线精简时连同数据生成、训练和阶段验证入口一起退出，目前不能在现有 SFT 与 GRPO 之间开展可追溯的偏好训练。需要按当前包架构和 ReAct 协议恢复完整但可选的 DPO 阶段，并修正旧配对逻辑将动作相同误当作完整上下文相同、为缺失步骤补造 FINISH 的问题，而不是回滚整个精简变更。

## What Changes

- 保留默认 `SFT -> GRPO`，新增显式 `SFT -> DPO -> GRPO` 路径；GRPO 默认输入仍为 `outputs/sft_qlora/final`，不自动选择最新 DPO 目录或自动启动中间阶段。
- 在统一命令族恢复 `agentic-arxiv data dpo generate` 和 `agentic-arxiv train dpo`，按现有职责分层实现数据生成、训练、阶段验证和模型交接，不恢复旧脚本及旧导入命名空间。
- 从当前协议的本地 SFT 模型和显式训练切分采样实际多轮轨迹，复用任务初始化、隔离环境和共同评分；只构造完整消息前缀严格一致的单步决策偏好。chosen/rejected 均来自实际模型当前响应，不拼接不同历史、不把框架结束标记转为 FINISH。
- 保存偏好对、来源轨迹、数据 manifest 和诊断报告，记录任务覆盖、偏好对产出率、各类丢弃原因与评分配置。零有效样本明确失败，低产出只报警，不自动降低匹配条件、奖励差阈值或伪造负例。
- DPO 复用现有配置、精度、QLoRA、单进程/DDP、日志和产物记录机制；策略与冻结参考策略均从同一 SFT 产物初始化，完整保留 SFT 适配器权重。支持明确的全参资源路径，不增加损失变体或独立调度系统。
- DPO 默认执行完整多轮阶段验证，保留奖励、严格成功率、解析率、虚假完成与终止原因，以及验证覆盖和协议关系；训练完成、验证通过和效果优于 SFT 分开表述。
- 扩展 GRPO 产物检查以接受当前协议的 DPO 全参/适配器产物；保留原 SFT 路径、现有参数与奖励语义，补充帮助、安装隔离检查、回归检查和活动文档。
- 工程实施与验收不生成研究数据、不执行训练或模型评测，不下载模型/快照，不改写历史研究资产；未覆盖的真实 tokenizer、参考策略梯度、GPU/DDP 和模型效果验证明确披露。

## Capabilities

### New Capabilities

无。DPO 作为现有训练、运行时和研究验证能力的扩展，不引入重复的平行规格。

### Modified Capabilities

- `training-mainline`: 将 DPO 从退出入口中单独恢复为可选阶段，定义统一命令、配置兼容、SFT/DPO/GRPO 产物交接和冻结参考策略契约。
- `react-runtime`: 将共同消息协议与训练责任边界覆盖到 DPO，定义完整上下文严格一致的真实单步偏好及无截断的目标边界。
- `research-validation`: 扩展训练切分隔离、多轮阶段验证和工程验收边界，增加偏好来源追溯、产出率与丢弃原因诊断要求。

## Impact

- 数据与运行边界：`src/agentic_arxiv/data_pipeline/`、`runtime/react_contract.py`、`tasks/`、`scoring/`、`provenance.py`；生成器不依赖训练器或评测编排层。
- 训练与模型：`src/agentic_arxiv/training/`、`llm/`，以及 `resources/mainline.json`；只抽取本次必要的产物校验与加载共享逻辑，不让训练入口互相导入。
- 用户接口与验证：`src/agentic_arxiv/cli/`、`scripts/check_installation.py`、相关 `tests/`、三语 README、`Overview.md` 与现有训练/架构/运行时文档。
- 依赖继续使用现有 local/train 分层与版本约束；对实际训练库接口做显式能力检查，不借恢复 DPO 升降级整个训练栈。
- 现有规格明确排除了 DPO，需由本变更的增量规格有意修订；其他已退出算法、服务与运行方式保持退出。既有 SFT/GRPO 配置继续可用，历史 DPO 字符串数据和缺少当前协议元数据的旧 checkpoint 不自动升级或视为兼容。

## Non-goals

- 不将 DPO 改为必经阶段，不增加自动流水线或自动选择训练起点。
- 不恢复 OPD、PPO、自进化、独立 VLM 后训练、算法变体、Web、Jev 或 MCP/Skill CLI。
- 不实现整轨迹 DPO、共享前缀分支重采样、专家/外部 API 混合偏好或人工构造负例。
- 不调整 SFT/GRPO 配方、任务集、切分和奖励公式，不承诺历史实验等价或 DPO 必然改善模型能力。
