# 容器化实验管理：本地工程验证记录

验证日期：2026-10-15。变更：`containerized-experiment-management`。

## 结论与授权边界

本地实现已完成，包括用户追加要求的受管 epoch 预算恢复；已执行的工程检查通过，但本地容器验证尚未完成。OpenSpec 共完成 **40/47** 项；`10.2` 实际训练镜像构建与容器执行，以及 `11.1–11.6` 服务器和真实训练门禁保持未完成，不归档变更。

本轮没有连接或操作服务器，没有启动真实研究模型训练、六组正式实验、正式 benchmark 或数据生成。唯一实际 Docker 操作为本机信息/镜像元数据读取与 Compose 配置解析，没有启动任务或预检容器。

## 实现范围

- 基础安装可使用 `experiment` 的检查、登记、启动、列表、状态、日志、停止、恢复和独立评测入口；管理进程不导入训练栈、不读取 `.env`。
- 严格三层 JSON 配置和六组单作业配方随 wheel 分发；空模板缺真实输入、预算和镜像时拒绝正式运行。
- 文件事务、独立 run/attempt、持久 GPU 声明、最小挂载、精确容器身份、未知结果保守处理和显式强制停止。
- 三阶段事务 checkpoint、提交后轮转、来源与内容核验、原始冻结参考、GRPO 生成缓存/采样位置/保护统计及安全停止接线。
- 独立 benchmark 与阶段重验、可取消边界、精确重验证据引用、分尝试日志和成本摘要；未知指标不填零，诊断结果不晋升模型资格。

按步数与按 epoch 的预算均已实现。epoch 模式保留原 `max_steps=-1` 和 epoch 数，由真实 Trainer 接口在第一次优化前核定完整总步数，写入 `resolved-budget.json`，恢复时再次核对原核定预算。管理器不从原始行数猜算、不改调度计划，也不另发剩余预算。

## 审查问题修复复验

### 对抗性复审修复记录（第七轮）

修复受管 SFT/DPO 的 TensorBoard 事件遗漏：管理读取同时扫描当前尝试的 `logs` 与 `output`，只收录文件，按完整路径去重并排序。保留 GRPO 的 `output/logs` 和历史布局，不迁移或改写事件，不改变训练、保存、恢复及日志生成逻辑；显式标量读取使用同一事件清单。

- 修复前，TensorBoard 专项中 5 项复现失败、3 项通过；修复后全部通过。覆盖两类日志根目录及嵌套目录、重复路径只读一次、不同目录同名文件保留、事件同名目录排除、缺少可选依赖、缺失尝试不创建目录和原文件内容不变。
- 基础环境的评测/观测、运行管理、CLI 与架构组合回归：**141 passed，101 个子测试通过**，包含上述 8 项专项；未重复运行全量训练回归。
- 严重级静态检查、差异空白检查和 OpenSpec 严格验证通过。日志：`/tmp/agentic-review7-before.log`、`/tmp/agentic-review7-focused.log`。

标量读取使用 EventAccumulator 接口替身确认接线；本地验证环境没有安装可选 TensorBoard，未新增依赖或宣称实际事件反序列化已验证。本轮未操作服务器、启动容器或真实训练，未提交或推送；OpenSpec 仍为 **40/47**，真实镜像与训练恢复门禁保持未完成。

### 对抗性复审修复记录（第六轮）

本轮修复最新复审确认的两项预检问题：

- DPO 启用阶段验证时要求明确的 `inputs.snapshot`，在偏好包/tokenizer 审计前用同一离线回放后端加载快照。缺失、损坏或无法加载时返回机器可读失败，不再等训练完成才暴露问题；关闭验证时不加载快照，也不要求该输入。不更改非受管 DPO 入口。
- M4 同时填写 `model` 和 `source_dpo` 时，独立完成两项输入审计后要求其解析到同一产物路径；冲突在容器预检、GPU 声明及运行创建前拒绝。单独使用任一字段、或不同引用形式指向同一产物仍受支持，预检、新启动和恢复选择同一策略与基座。

验证结果：

- 修复前，新增的 8 项缺陷回归均复现失败；修复后通过。DPO 检查复用已有快照 fixture，确认不执行工具、不保存快照、不创建输出、原文件哈希不变，并覆盖验证关闭和失败 JSON。M4 检查覆盖 full/adapter、两种模型字段、同路径双引用、冲突拒绝、QLoRA 开关及恢复重渲染。
- 回放环境依赖仅加入容器预检入口的精确白名单；新增反向架构检查，继续禁止宿主 service/artifacts/config 导入该后端，预检模块导入本身仍只加载标准库。
- 相关模块及架构回归：**184 passed，101 个子测试通过**。
- 当前源码基础环境全量离线回归：**1086 passed，42 skipped，54 deselected，869 个子测试通过**，6 条既有弃用警告。
- 当前源码固定训练依赖环境全量离线回归：**1328 passed，11 skipped，1 deselected，869 个子测试通过**，6 条既有弃用警告。两套环境用例重叠，不累加；跳过及排除项不算通过。
- 严重级静态检查、差异空白检查和 OpenSpec 严格验证通过。日志：`/tmp/agentic-review6-before.log`、`/tmp/agentic-review6-focused.log`、`/tmp/agentic-review6-base.log`、`/tmp/agentic-review6-training.log`。

本轮仅执行本地工程检查，没有操作服务器、启动容器或真实训练，未重建分发包或镜像，未提交或推送。元数据/控制接线和已有快照加载不证明真实模型能够完成训练或正确恢复；OpenSpec 仍为 **40/47**，实际镜像与真实训练门禁保持未完成。

### 对抗性复审修复记录（第五轮）

本轮修复两位独立复审者确认的三项问题：

- SFT/DPO 受管恢复在当前 Trainer 的 `train` 作用域内保留跳批加载器的真实 epoch。适配固定 Transformers 4.57.6 / Accelerate 1.11.0，使用原训练循环代码对象和隔离的 `skip_first_batches` 依赖，不改采样器、种子、跳过数量、梯度累积或预算。正常及异常退出均还原实例方法，不改模块全局、非受管路径或 GRPO。
- 预检、新启动和恢复共用同一模型参数渲染：仅已核验的 adapter 来源提供执行 `base_model`，全参 SFT/DPO 来源保留基座溯源和只读挂载但将执行值设为 `null`。移除启动时的重复赋值，防止预检重新覆盖；保留 adapter 冲突基座拒绝及 M1 输入核验。
- GRPO 受管成功出口统一使用已记录的训练终态。零方差保护恰好在最后预算步进入 `converged` 时，仍保留 `protection_stopped`，不导出成功 `final`、不进入正常验证，也不覆盖为预算完成。非受管历史行为保持不变。

验证结果：

- 当前源码基础环境全量离线回归：**1066 passed，42 skipped，54 deselected，869 个子测试通过**，6 条既有弃用警告。
- 当前源码固定训练依赖环境全量离线回归：**1308 passed，11 skipped，1 deselected，869 个子测试通过**，6 条既有弃用警告。两套环境用例重叠，不累加；跳过及排除项不算通过。
- 新增 65 项加载器回归，执行生产 SFT/DPO 配置表达式、真实 Trainer 采样/预算接口和 Accelerate 加载器，覆盖单分片及四分片各 rank、steps/epochs 预算、首/后续 epoch 中途、整 epoch 边界和恢复后继续跨 epoch 的控制索引一致性。另检查实例绑定、正常/异常还原、版本/接口拒绝及基础环境依赖隔离。
- 新增 26 项模型绑定回归，覆盖 M2/M3/M4/C1、full/adapter 来源、本次 QLoRA 开关、预检与实际 CLI 配置、恢复重渲染、基座冲突和 M1 输入身份。新增 10 项 GRPO 完成边界回归，使用真实默认流程回调及保护器，执行生产完成片段检查保护优先级、正常完成与非受管行为。
- 已同步既有 SFT/DPO 控制片段测试的上下文替身；最终全量检查无失败。严重级静态检查、差异空白检查和 OpenSpec 严格验证通过。日志：`/tmp/agentic-adversarial-rereview-fixes-base.log`、`/tmp/agentic-adversarial-rereview-fixes-training.log`。

新增索引检查的四分片仅为本机 CPU 接口模拟，不是四进程 GPU 训练；配置和控制测试不证明真实模型、量化优化器或端到端 RNG/权重等价。本轮未操作服务器、启动容器或真实训练，未重建分发包或镜像，未提交或推送；OpenSpec 仍为 **40/47**，实际镜像和真实训练门禁保持未完成。

### 对抗性审查修复记录（第四轮）

本轮修复两位审查者确认的五项问题：

- M2 明确解析 DPO 包必需的来源 manifest、切分、轨迹和报告文件，核对路径与声明哈希后冻结身份；预检和训练使用同一最小只读挂载清单，模型加载前重新核对，不递归历史来源或挂载整个输入根目录。
- 受管 GRPO 在创建策略及新 LoRA 前设置共同初始化种子，随后汇总各 rank 的参考指纹；保留 TRL 的 rank-specific 生成种子，以及 DDP 包装和恢复后的参考检查。
- SFT、DPO、GRPO 的受管 PEFT 标准加载和根策略补载均禁止自动升精度；优化器恢复前核对参数对象、dtype、冻结状态、活动 adapter 及优化器引用，避免 BF16 静默变为 FP32。
- GRPO 重建完整参考前，仅解除旧参考在 Accelerator 中的身份匹配注册并回收，不清空策略、优化器或其他登记。
- DPO 加载阶段验证模型前，一并释放受管 checkpoint 句柄和停止回调，避免其强引用保留 Trainer、策略、参考和优化器。

验证结果：

- 当前源码基础环境全量离线回归：**1040 passed，40 skipped，54 deselected，869 个子测试通过**，6 条既有弃用警告。
- 当前源码固定训练依赖环境全量离线回归：**1207 passed，11 skipped，1 deselected，869 个子测试通过**，6 条既有弃用警告。两套环境的用例重叠，不累加；跳过及排除项不计入通过。
- 新增真实 PEFT/Trainer 的 CPU 微型保存加载检查覆盖 BF16/FP32、safetensors/bin、标准 SFT/DPO 优化器和调度器恢复，以及 GRPO 双 adapter 嵌套加载。初始化和引用生命周期检查使用 CPU 微型层、真实 Accelerator 与弱引用；没有模型下载、前向训练或量化模型执行。
- DPO 输入检查使用文件合同和命令替身，证明依赖清单、白名单、哈希与只读挂载接线，不作为真实偏好包/tokenizer 容器审计证据。
- 严重级静态检查、差异空白检查及 OpenSpec 严格验证通过。日志：`/tmp/agentic-adversarial-fixes-base.log`、`/tmp/agentic-adversarial-fixes-training.log`。

本轮未操作服务器、启动容器或真实训练，未重建分发包或镜像。已消除上述确定性代码缺陷，但没有验证实际 GPU 显存、CUDA/NCCL 或研究模型完成性；OpenSpec 仍为 **40/47**，镜像与真实训练门禁保持未完成。

### 第三轮审查修复记录

明确的容器启动拒绝现已独立存证；只有核验同一容器仍为 `created` 且未运行，才结束失败尝试并释放 GPU 声明。暂时无法核验时保留声明，由后续状态查询确认；超时、身份不匹配或启动后查询失败不作为释放依据。受管 benchmark 则要求显式离线模式和明确已有快照，在创建运行记录前检查快照文件及回放 JSON 结构，不更改容器禁网策略或非受管入口。

- 当前源码基础安装全量离线回归：**1019 passed，38 skipped，54 deselected，869 个子测试通过**，6 条既有弃用警告。
- 运行管理、评测、CLI、容器配置及架构组合回归：**148 passed，101 个子测试通过**。与全量结果重叠，不累加。
- 严重级静态检查、差异空白检查及 OpenSpec 严格验证通过。日志：`/tmp/agentic-experiment-review3-base.log`、`/tmp/agentic-experiment-review3-focused.log`。
- 本轮仅使用本地控制替身及已有快照 fixture，不操作服务器、启动容器或真实训练；没有重建分发包或镜像。容器与 GPU 门禁仍未完成。

### 第二轮审查修复记录

第二轮审查的两项问题已修复：受管 GRPO 双 adapter 恢复在标准加载边界显式补载根目录 `default` 策略，并核对冻结 `ref` 未改变；启动时间改为启动后读取并在状态查询时校正，Docker 零时间不参与成本计算。目标 Python 3.10 的纳秒时间解析兼容性已通过本机 Python 3.10.19 实测。

- 第二轮当前源码基础安装全量离线回归：**993 passed，38 skipped，54 deselected，866 个子测试通过**，6 条既有弃用警告。
- 固定训练依赖环境中 GRPO、checkpoint、受管训练、writer、停止控制、运行管理及成本组合回归：**232 passed**。
- 双 adapter 专项使用真实 PEFT 与 Transformers 加载接口及微型线性层，覆盖 safetensors/bin、策略参数值、参数对象和优化器引用、冻结参考以及不完整权重拒绝；没有训练或生成，不等同于真实量化模型恢复验证。
- 严重级静态检查、差异空白检查、OpenSpec 严格验证通过；日志为 `/tmp/agentic-experiment-review2-base.log` 与 `/tmp/agentic-experiment-review2-training.log`。未启动容器、操作服务器或执行真实训练，未重建分发包或镜像。

### 第一轮审查修复记录

本次修复四项审查问题：运行挂载及训练入口只复核明确输入与基座，不追读历史来源或对照运行路径；受管 SFT 阶段验证绑定明确快照，并在预检拒绝缺失/损坏快照；最终产物身份等容器退出后再冻结；旧产物校验的文本退出错误在适配边界转换为 JSON 错误及退出码 2。

- 当前源码基础安装全量离线回归：**969 passed，38 skipped，54 deselected，866 个子测试通过**，6 条既有弃用警告。
- 固定训练依赖环境中的运行管理、受管训练、阶段 writer、预检、产物、CLI、评测和架构接线回归：**174 passed，98 个子测试通过**。
- SFT 快照、writer、预检、配置资源和原 CLI 专项回归：**62 passed，145 个子测试通过**。该项与上一项有重叠，不累加为独立用例数。
- 严重级别静态检查、差异空白检查与 OpenSpec 严格验证通过。没有操作服务器、启动容器或执行真实训练。快照成功路径使用接口替身，不能视为真实容器内快照验证。
- 新证据日志：`/tmp/agentic-experiment-review-fixes-base.log`、`/tmp/agentic-experiment-review-fixes-training.log`。

下面的完整 wheel 和构建上下文身份保留为修复前历史证据，并不代表已包含本次修复；本轮未重新构建镜像或分发包。

## 修复前已执行检查

| 检查 | 实际结果 | 证据范围 |
|---|---|---|
| 基础依赖源码安装全量离线回归 | 928 passed，38 skipped，50 deselected；862 个子测试通过 | 排除 train/local/network，未加载真实模型 |
| 正常 wheel 训练安装全量离线回归 | 1059 passed，4 skipped，1 deselected；865 个子测试通过 | 从 site-packages 导入，无源码搜索路径注入 |
| 基础与训练正常 wheel 安装探针 | 均通过 | 仓库外命令帮助、资源、缺失输入保护、依赖层隔离、10 个工具声明 |
| wheel 与当前源码核对 | 121 个包内文件逐字节一致 | 包含 Compose、profile 和六份 recipe，无研究资产或凭据 |
| 两个正常安装环境的依赖一致性检查 | 均通过 | 未改动项目原 `.venv` |
| 四进程 CPU/Gloo 控制与既有微型张量回归 | 已包含在工程测试中通过 | 不是 CUDA/NCCL 或真实研究模型恢复证据 |
| 三阶段 steps/epochs 配置往返 | 通过 | 实际 CLI 解析、受管身份、原 Trainer 预算计算与恢复接线 |
| 跨模块失败路径 | 通过 | GPU 争用、创建/启动/强杀响应丢失、服务不可达、损坏断点、旧尝试来源复用、停止竞态 |
| 完整目标依赖闭包 | 94 包元数据解析通过 | 精确版本、发行物哈希与实际来源，不等于目标环境安装通过 |
| 构建上下文生成 | 通过 | 白名单复制、真实代码哈希、dirty 标记，未携带模型/数据/凭据/缓存/输出 |
| 本机 Compose 配置解析 | 通过 | 仅 `config --quiet`，没有构建或启动容器 |
| 语法与未定义名称检查 | 0 个错误 | E9/F63/F7/F82，包含新增构建工具 |
| OpenSpec 严格验证及差异空白检查 | 通过 | 不替代运行验证 |

工程测试使用控制元数据、已有研究输入的只读审计、既有替身和微型张量。没有新造模型、论文、token 序列、偏好包或轨迹充当真实恢复证明。正常安装探针只读读取既有 24 条历史轨迹并核对内容未改；`data/`、`artifacts/`、`configs/` 和包内原 `mainline.json` 没有本次改动。

基础回归有 6 条依赖弃用警告；训练 wheel 回归有 20 条警告，涉及既有 PyMuPDF/Pydantic 与 PyTorch JIT 弃用提示，未作为通过项隐藏或改动无关代码。跳过及排除项不计为已通过。

## 环境与可复现身份

本机为 macOS ARM64、Python 3.12.10。训练接口工程环境采用 PyTorch 2.10.0、Transformers 4.57.6、TRL 0.29.1、PEFT 0.17.1、Accelerate 1.11.0；本机可安装的 bitsandbytes 为 0.50.2。该环境能检查正常训练 extra 安装及固定 Trainer/TRL 接口，**不是目标 Linux CUDA 环境的等价替身**。

目标镜像独立固定 Python 3.10.19、Linux amd64、PyTorch 2.10.0+cu126 和 bitsandbytes 0.45.5 等基线。基础 Python 镜像摘要已向官方 registry 核验；训练镜像本身尚未构建，因此正式训练镜像摘要仍为 `null`，启动不会用基础镜像摘要或标签冒充训练镜像身份。

- 本次 wheel：`/tmp/agentic-experiment-dist/agentic_arxiv-0.1.0-py3-none-any.whl`。
- wheel SHA256：`d8475990de9d949e33792efe36bf29abcaa806e189486f73d91afee23641e9fd`。
- 目标依赖锁 SHA256：`bb3d615a1ceb3e7fb659fbaf5a77c289dcbf155065a735b13e4fba2b0280d7aa`。
- 构建输入树 SHA256：`26a9abb43f4e954ba800164d9de9fcc337b323149ea4787942ccd48277754d95`。
- 构建上下文：`/tmp/agentic-experiment-build-context-final`，其中 `build-identity.json` 明确记录 `dirty: true`，不将本次未提交实现伪称为干净提交。

临时证据文件：

```text
/tmp/agentic-experiment-base-tests-final.log
/tmp/agentic-experiment-train-wheel-tests-final.log
/tmp/agentic-experiment-wheel-probe-final.json
/tmp/agentic-experiment-train-wheel-probe-final.json
/tmp/agentic-experiment-wheel-build.log
/tmp/agentic-experiment-build-context-final.json
```

这些临时文件用于复查，本页保存结论及验证边界；临时目录不保证长期保留。

## 未验证矩阵

| 层级或组合 | 状态 | 后续门禁 |
|---|---|---|
| 目标 amd64/CUDA 训练镜像构建、依赖实际安装、容器内无 GPU 入口与只读 tokenizer 审计 | 未执行 | `10.2`；当前本机为 ARM64，未启动大型跨架构构建 |
| 服务器 Docker 权限、NVIDIA 运行时、磁盘与空闲设备 | 未执行 | 独立授权后执行 `11.1`，历史状态不能代替 |
| SFT 单卡 steps/epochs 连续与停止恢复对照 | 未验证 | `11.2`，比较步数、学习率、数据消费、随机状态与权重 |
| DPO 单卡连续与停止恢复对照 | 未验证 | `11.3`，另核原始冻结 SFT 与优化器排除 |
| GRPO 单卡连续与停止恢复对照 | 未验证 | `11.4`，另核缓存消费、原始参考、课程与保护统计 |
| 三阶段四卡 CUDA/NCCL 恢复及失败协调 | 未验证 | `11.5`，实际 GPU UUID/rank 枚举、rank 状态及部分保存失败须实测 |
| 正式模型登记、收益和统一留出集评测 | 未执行 | `11.6` 及用户逐项明确操作；不自动启动六组实验 |

真实对照须从开始采用相同总预算，预先记录浮点容差，离散控制与消费位置要求一致。仅权重可加载、容器退出码为 0、存在 `final`、测试替身通过或管理启动成功，都不能替代真实训练恢复和模型质量证据。已有虚拟环境与原入口保留，回退不删除模型、旧容器、checkpoint 或失败证据。

## 复查入口

- [命令与配置](experiment-cli.md)
- [输入登记与六组配方](experiment_inputs.md)
- [运行事务与身份保护](experiment-runtime.md)
- [容器准备与权限前提](container-deployment.md)
- [只读容器预检](experiment-preflight.md)
- [checkpoint 格式与 epoch 核定](managed-checkpoint-format.md)
- [SFT/DPO 恢复](sft-dpo-managed-resume.md)
- [GRPO 恢复](grpo-managed-resume.md)
- [安全停止协议](training-control.md)
