# Spec Delta

## MODIFIED Requirements

### Requirement: 版本化上下文与训练责任边界

统一协议 SHALL 使用相同的初始任务消息、assistant 动作消息和环境 Observation 消息序列，应用于 SFT、DPO、GRPO 与推理。SFT MUST 只监督当前目标 assistant 响应；DPO MUST 在共同 prompt 下仅对 chosen/rejected 的当前单条 assistant 响应计算偏好目标，不将历史、setup 或环境反馈计入目标；GRPO MUST 保留原始采样 token，模型 token 与环境 token 的损失掩码必须区分，且后续消息追加不能改写已有采样前缀。实际运行使用的模板无法保持前缀时 MUST 明确失败，不通过重新编码 GRPO 模型响应绕过。DPO 离线消息的训练编码 SHALL 校验实际模板和共同前缀，不冒充 on-policy 原始 token 重放。协议迁移 MUST 使用新协议版本；新增 DPO 数据 SHALL 使用独立数据标识并声明当前上下文协议，不静默混用不兼容训练数据或模型元数据。本次 SHALL 改造相应转换、标签、掩码及元数据代码，而不要求实际生成数据、训练或补齐 tokenizer；验证依据和未覆盖部分按工程验收规格记录。

#### Scenario: 同一历史到达下一决策点
- **WHEN** 数据生成、DPO 配对、GRPO 和本地推理在相同历史、协议和 tokenizer 下准备下一步输入
- **THEN** 得到相同的消息内容和上下文 token 前缀；外部 API 至少接收相同逻辑消息序列

#### Scenario: 多轮训练标签和掩码
- **WHEN** 训练样本包含先前动作与环境反馈
- **THEN** SFT 仅当前目标响应有监督标签，DPO 两侧均仅当前 assistant 参与偏好目标，GRPO 的环境反馈及运行时插入边界不参与策略 loss，GRPO 模型原始采样 token 不被重新编码替换

#### Scenario: 新旧协议混用
- **WHEN** 训练输入声明的协议与当前训练协议不兼容
- **THEN** 系统在训练前报告不兼容；跨协议评测必须显式标注，不能作为同口径复现结果

#### Scenario: 代码迁移但尚未生成新数据
- **WHEN** 本次仅完成协议适配及数据转换代码改造，未执行数据生成或真实 tokenizer 集成
- **THEN** 历史数据和 manifest 保持原样，现有材料检查与未验证集成分别记录，不为验收合成样本、token 或轨迹

## ADDED Requirements

### Requirement: 单步偏好要求实际完整上下文一致

系统 SHALL 仅在同一任务、初始化、快照、来源模型、协议和采样设置下比较真实轨迹，且配对双方当前实际请求的完整消息序列 MUST 完全一致，包括角色、顺序、任务/工具说明、可见 setup、历史 Thought/Action 及环境 Observation 的原文本。系统 MUST 不通过忽略 Thought、只比较动作、裁剪差异、归一化消息文本、近似匹配或补齐一侧历史达成匹配。每个轨迹组合 SHALL 只考虑其共同前缀下首个有效动作分歧；奖励差用于排序，不宣称已经证明单步因果优势。

#### Scenario: 动作历史相同但 Thought 不同
- **WHEN** 两条轨迹之前调用了相同工具，但模型在当前决策前实际看到的历史 Thought 文本不同
- **THEN** 当前决策不能配对，记录上下文不一致，不删除 Thought 后继续匹配

#### Scenario: 环境反馈或初始化不同
- **WHEN** 工具动作相同，但可见 Observation、setup 或来源环境身份不同
- **THEN** 该组合不产生偏好，记录对应来源或上下文差异，不借用另一轨迹的观测

#### Scenario: 当前动作相同而响应文本不同
- **WHEN** 在同一完整请求下，两侧当前动作相同但 Thought 或响应原文本不同
- **THEN** 本轮不作为动作偏好，且两条响应不会被视为后续轮次的共同历史；后续请求不一致时明确丢弃

#### Scenario: 共同完整前缀后动作不同
- **WHEN** 两侧当前请求完全相同，当前真实动作不同，双方响应和轨迹评分可用且奖励差严格超过配置阈值
- **THEN** 仅将该轮两条真实 assistant 响应构成候选，以高分轨迹为 chosen，保留两侧完整来源与当前轮次

### Requirement: 偏好目标必须是真实当前模型决策

chosen/rejected SHALL 各为实际模型生成的一条当前响应，使用共同运行时的策略文本边界，并能够解析为 CALL 或 FINISH。系统 MUST 不把模型续写的 Observation、框架终止标记、空响应或缺失步骤改造为训练目标；被 token 截断的当前响应 MUST 不作为完整决策配对。真实提前 FINISH SHALL 按实际轨迹评分参与比较，不能仅凭 FINISH 判断成功。缺失步骤、预算耗尽或错误 MUST 保持真实终止原因，不补造 FINISH。

#### Scenario: 短轨迹没有对应决策
- **WHEN** 一条轨迹在候选轮次之前耗尽预算或停止，缺少该轮模型响应
- **THEN** 丢弃该组合并记录缺失原因，不创建“任务完成”文本或 FINISH 作为替代

#### Scenario: 真实提前 FINISH
- **WHEN** 在完全相同的 prompt 下，一侧模型真实输出 FINISH，另一侧输出工具动作
- **THEN** 系统按实际轨迹奖励及差值条件决定是否形成偏好，不固定把 FINISH 视为正例或负例

#### Scenario: 模型续写虚假环境内容
- **WHEN** 原始模型响应在当前 Action 后续写 Observation 或后续动作
- **THEN** 仅使用共同运行时保留的当前策略文本，伪造观测与后续动作既不进入环境历史也不进入偏好目标

### Requirement: 偏好编码不得以截断改变上下文

训练前系统 MUST 使用实际 tokenizer 与模板审计共同前缀、两侧单条 assistant 目标和序列长度。prompt 中所有历史/环境内容 MUST 排除于当前目标的概率统计；超出配置预算或模板不能保持共同上下文时 MUST 明确失败，不通过左截断、右截断或删除任一侧内容继续训练。两侧在编码后不再包含有效目标时 MUST 拒绝。

#### Scenario: 长上下文超预算
- **WHEN** 原始配对满足完整上下文一致，但任一侧编码后的完整序列超过训练预算
- **THEN** 在优化前报告样本与长度并失败，不截去历史或动作以保留该对

#### Scenario: 模板改变前缀或目标边界
- **WHEN** 实际模板处理使两侧 prompt 前缀不同，或把历史 assistant/Observation 混入当前目标
- **THEN** 审计失败，不将不同条件下的概率或环境文本用于偏好优化
