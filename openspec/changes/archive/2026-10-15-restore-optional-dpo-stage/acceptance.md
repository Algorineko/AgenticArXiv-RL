# 工程验收记录

## 范围与结论

日期：2026-10-15（Asia/Shanghai）。本记录对应 `restore-optional-dpo-stage` 的三份增量规格，区分源码审查、既有输入回归、历史记录读取和真实研究执行。默认仍为 SFT → GRPO，可选 DPO 只能显式启动。

本轮不生成研究数据、不加载模型权重、不执行训练或模型评测，不下载模型或快照，不补造响应、token、轨迹或负例。新增检查限于现有输入、文件元数据、计数、参数及源码结构。正常构建和安装 Python 包不属于研究资产生成。

## 执行环境与命令

- 源码基础安装：`/tmp/agentic-arxiv-packaging/editable/bin/python`，实际编辑式安装指向当前 `src/agentic_arxiv`，不使用 `PYTHONPATH`。
- 源码无模型回归：`/tmp/agentic-arxiv-ddp-20261015/bin/python`，复用已有 torch、Transformers、PEFT、TRL 0.29.1 和 pytest。该环境缺 `bitsandbytes`，不是完整 train 安装。
- 正常 wheel：`uv build --wheel --out-dir /tmp/restore-dpo-wheel-dist`，在独立临时环境安装，不以编辑式安装替代分发包检查。
- 所有 pytest 仍沿用现有离线保护及临时产物目录，禁止真实网络连接。主动排除 `tests/test_training_ddp_integration.py`，因为该文件启动 CPU/Gloo 进程并执行 `backward`，不属于本轮不训练边界。

中央回归命令：

```bash
/tmp/agentic-arxiv-ddp-20261015/bin/python -m pytest -q -rs \
  --ignore=tests/test_training_ddp_integration.py -m 'not network'
/tmp/agentic-arxiv-ddp-20261015/bin/python -m flake8 \
  src/agentic_arxiv scripts/maintenance scripts/check_installation.py \
  --count --select=E9,F63,F7,F82 --show-source --statistics
/tmp/agentic-arxiv-ddp-20261015/bin/python \
  scripts/maintenance/audit_mainline_configuration.py
git diff --check
```

初轮实现的中央结果：**806 passed，735 subtests passed，7 skipped，1 deselected**。错误级 flake8 为 0，配置对照为 `static_checks_passed` 且 `failures=[]`，`git diff --check` 通过。测试日志保存在 `/tmp/restore-dpo-final-pytest.log`，配置对照在 `/tmp/restore-dpo-final-config-audit.json`。这些临时日志不是研究记录。审查后修复的检查范围见文末补记，不混用两轮结果。

安装检查实际执行 `scripts/check_installation.py --layer base|local --history artifacts/base_pilot_v1/traces.jsonl`；正常 wheel 检查额外传 `--require-wheel`：

- 源码 base：通过，Python 3.12，实际根包位于当前源码目录。
- 正常 wheel base：通过，Python 3.13，根包位于独立 `/tmp/restore-dpo-wheel-base` 的 site-packages。
- 正常 wheel local：通过，Python 3.12，在已有 `/tmp/agentic-arxiv-local` 仅重装当前 wheel，根包位于 site-packages；脚本确认 local 声明依赖齐全且无 TRL/datasets/bitsandbytes，不用训练依赖掩盖隔离问题。
- 上述三次均从仓库外目录核对 **32 个 console/module 帮助**、工作区路径、缺能力/缺输入保护、10 工具声明及 24 条既有历史记录只读读取。帮助不加载重依赖、模型或研究数据，不写工作区。
- 初轮 wheel 内 97 个源码/资源文件逐字节与当时的 `src/agentic_arxiv` 一致，无多余旧模块或研究资产。wheel SHA256：`cbf2211f655351c15beda7bdec77e5e18bcf57573551b9565731c98d35f31ef2`。
- 源码 train 安装脚本实际尝试后在依赖关口失败：`train 安装缺少声明依赖: ['bitsandbytes']`。未绕过检查或伪造安装，也未为此安装 GPU 训练依赖；正常 train wheel 的完整安装检查未执行。已安装 TRL 的字段/签名和无模型训练保护测试通过，不能代替完整 train 安装、QLoRA 或 GPU 训练。

已知跳过项：3 个可选 TensorBoard 分析、3 个手动网络/API 检查、1 个缺已有快照的 rollout audit；另有 1 个 network 标记用例未选择。原有 PyMuPDF/Pydantic 弃用警告不代表本次训练行为验证失败。

## 增量规格逐项证据

| 规格要求 | 检查依据 | 证据层级与边界 |
| --- | --- | --- |
| 受支持的训练与运行入口 | CLI 根/阶段/数据帮助、`test_cli_entrypoints.py`、活动文档 | 参数与导入检查；只恢复 DPO，其他退出入口不恢复 |
| 直接 SFT → GRPO 与显式 DPO 交接 | `llm/artifacts.py`、`training/model_setup.py`、GRPO 来源链、`test_model_artifacts.py`、`test_training_guards.py` | 既有错误输入、文件元数据与源码；未加载 checkpoint |
| 主线依赖与工具能力分离 | `test_architecture.py`、`test_install_metadata.py`、安装脚本 | 源码依赖边界与安装层隔离；10 工具声明完整，不实际执行工具后端 |
| 可安装包与统一命令 | console/module 帮助、wheel 文件清单、安装脚本 | 在仓库外目录检查；不执行命令对应的研究操作 |
| 显式工作区与资产路径 | CLI 路径断言、安装脚本缺输入保护、新生成器拒绝检查 | 不搜索其他目录、不联网补齐，缺输入不产生研究产物 |
| DPO 配置兼容 | `test_config_resources.py`、旧配置临时副本、CLI 显式覆盖、配置静态审计 | 缺整节只在内存补默认，显式三组严格校验；旧文件不写回 |
| 冻结的完整 SFT 参考策略 | 两个独立 loader、storage/freeze/eval/optimizer 保护、训练入口结构检查 | 源码及已安装接口；未验证真实优化后参考权重不变 |
| DPO 训练审计与来源完整 | `load_dpo_package`、真实 tokenizer 审计接线、`--inspect-only`、manifest/writer 结构 | 包格式错误与来源保护、源码；未运行成功包或真实 tokenizer 集成 |
| 版本化上下文与目标边界 | 当前 runtime/消息协议回归、SFT/GRPO 原保护、DPO 三列目标与 TRL 预处理后逐 ID 重核 | 既有输入及源码；不是 on-policy token 重放证据 |
| 实际完整上下文一致 | 实际请求深拷贝、逐消息比较、历史连续绑定、首个动作分歧 | 配对源码审查；正例动态分支未用新合成轨迹补证 |
| 真实当前模型决策 | 原始响应与 history/policy_text 绑定、已知停止条件、CALL/FINISH 与框架标记排除 | 既有解析/终止回归及源码；不补 FINISH，未知停止拒绝 |
| 编码不得截断改变上下文 | 两侧完整模板前缀/有效目标/长度审计、TRL `max_length=None`、预处理后逐 token 比较 | 已安装库源码与接口检查；真实 tokenizer 未执行 |
| 训练切分隔离 | 复用现有 expanded `:train` 选择、既有切分定义、生成拒绝用例、包成员检查 | 留出集、裸 train、缺输入在采样前拒绝；SFT 选择范围不变 |
| 多轮验证与覆盖声明 | `StageVerifier.verify_dpo`、共同 episode/Canary、既有 false_finish 输入 | 109 项相关既有回归中覆盖评分/阈值/错误状态；没有运行模型 Canary |
| 无需训练的工程验收 | 本记录、无模型测试、资产及安装检查 | 不要求补资产或训练，不将未验证项写成通过 |
| 偏好及实际轨迹可追溯 | source/manifest/任务/切分/初始化/快照/采样/评分/hash/trace 绑定、独占发布 | 元数据拒绝/零数据诊断 IO 与源码；成功采样包未真实执行 |
| 产出率与丢弃原因 | `test_dpo_pair_contract.py` 纯计数、请求组合枚举、去重/稳定排序/唯一分类源码 | 分母含失败请求，零有效对不发布训练包，低产出不改阈值；正向偏好效果未验证 |

训练接口依据交叉核对了本机 TRL 0.29.1 与官方 [DPOTrainer 源码](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/dpo_trainer.py)、[DPOConfig 源码](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/dpo_config.py)。显式参考模型、默认 completion mask 和禁截断参数的源码核对，不等于训练集成成功。

## 保全与本次修复

- `sft`、`grpo`、原 `source_baseline` 三子树与 HEAD 内容相同；新增 DPO 训练及生成默认单独记录，整体资源哈希已更新，不能声称原文件字节未变。
- 当前资源 SHA256：`f3c958144123a522254657cf036bf004291af77dccf4a72d0bfecfa09183d883`。
- 资产规范哈希：`751b45441cb88cc59b92d2a26e2e6f4633a76b30e34339653169bc163a41cd12`，对应 10 工具、86 TaskSpec、8 冒烟任务和 9 份切分；这不是运行了 86 个任务。
- 本轮前后 300 个受跟踪任务/工具/研究文件和 279 个现存研究文件的 SHA256 清单分别比对通过，历史训练结果、归档行为基线不改写；历史 JSONL 只读 24 条用于安装读取检查，没有重标为新协议训练证据。
- 三语 README 历史结果表格和迁移说明历史正文保留；16 条新 DPO/交接示例仅通过参数解析，不执行生成、审计或训练。
- 首轮回归暴露的 Canary Mock 属性兼容与资源旧哈希问题已修复；后续审查修复了 manifest 命名、初始化计划类型、语义评分参考、基础设施失败负例污染、未知停止条件、来源及组合计数绑定。错误级 lint 暴露的资源释放闭包引用已修复，不以禁用规则绕过。
- 本地产物采样路径显式关闭二次编码的特殊 token 添加，对齐 Transformers 模板编码责任边界；原非 artifact / `from_loaded` 路径保持原行为。这里只完成库源码与 AST 检查，没有生成真实或合成 token 作为证据。
- `openspec validate restore-optional-dpo-stage --strict --json` 通过，增量规格没有校验问题；不归档或自动同步主规格。

## 明确未验证

- 真实 tokenizer、实际模板 token 边界及超长数据的端到端集成。
- 本地 SFT 实际采样、真实正向偏好包及完整包到训练器的端到端执行。
- 真实 policy/reference 加载、优化后参考冻结/梯度、adapter 合并与导出后的实际推理。
- CUDA/QLoRA 显存容量、GPU/DDP 训练、真实资源释放及跨 rank 训练失败通信。
- 完整 86 任务或留出集执行、DPO 默认阶段模型验证、收敛性、历史等价性与相对 SFT 的收益。

这些均为本轮规格明确排除的资产/研究执行验证，不是必须补做训练才能完成本变更的遗留任务。工程检查通过、训练完成、阶段验证通过及模型提升是四种不同结论。

## 审查后修复补记（2026-10-15）

- 修复 DPO 默认阶段验证的编码模式丢失：从 `StageVerifier` 经 `CanaryEvaluator` 到 `TransformersLLMClient.from_loaded(..., artifact=True)` 显式透传严格模式；实际采样客户端因此不重复添加模板特殊 token。其他阶段与默认 `from_loaded()` 继续原行为。新增回归在执行 episode 前截住调用，检查三阶段实际构造的客户端模式，不制造响应、token 或轨迹。
- 修复安装脚本同时缺模型和数据时的错误预期：DPO 实际首先拒绝模型，因此 train 分支匹配 `missing-model`。缺输入命令定义抽为纯函数，以同一组实际参数调用 DPO 输入校验进行回归；base/local 仍提示安装 train。这个检查没有伪造完整 train 安装。
- 相关检查：`/tmp/agentic-arxiv-ddp-20261015/bin/python -m pytest -q tests/test_model_artifacts.py tests/test_training_guards.py tests/test_canary.py tests/test_llm_request_params.py tests/test_install_metadata.py tests/test_architecture.py`，结果 **121 passed，20 subtests passed**。
- 无本地/训练依赖的 base 环境：`/tmp/agentic-arxiv-packaging/editable/bin/python -m pytest -q tests/test_model_artifacts.py tests/test_install_metadata.py`，结果 **17 passed**。
- 本次涉及文件的错误级 flake8 为 0，`git diff --check` 通过。没有重复全量回归或重新构建/安装 wheel，上述初轮 wheel 指纹不代表包含这次修复。
- 未加载真实模型或 tokenizer，未执行数据生成、训练或模型评测；完整 train 安装仍缺 `bitsandbytes`，未将直接输入校验回归声称为完整安装验证通过。
