# 容器侧只读输入预检

`python -m agentic_arxiv.experiments.preflight` 是管理端明确委托的内部检查入口，不是训练命令。模块导入只使用标准库；读取请求后才懒加载所选阶段的只读审计函数。它不构造模型、训练器、优化器或正式运行目录，不下载输入，也不读取 `.env` 或启用文件日志。

标准输入为 JSON 对象：

```json
{
  "stage": "sft",
  "training": {"defaults": {}, "trainer_parameters": {}},
  "inputs": {},
  "workspace": "/tmp"
}
```

上面的空配置仅说明字段结构，不能通过检查。`training` 必须包含现有主线解析器要求的完整阶段配置，`inputs` 使用管理层解析出的明确实际路径。容器工作区可为只读检查容器的临时目录；全部真实模型、数据、manifest、快照与切分文件使用明确绝对路径的最小只读挂载，不挂整个源码或用户目录。GRPO 必须指定与已解析输入文件一致的 `FILE:train` 或 `FILE:rl_train`，不猜测切分，不接受冒烟或留出集。

- SFT：复用 `_verify_data_manifest`、协议与现有产物模板检查、`_load_sft_rows`、`_check_lengths`；不跳过 manifest 审计，不为检查构建磁盘数据集。启用阶段验证时必须在配方 `inputs.snapshot` 声明已有快照，预检先确认该文件可由离线回放后端加载；缺失或损坏时在加载 tokenizer、分配训练尝试前失败，不依赖容器工作目录下的默认快照。
- DPO：启用阶段验证时必须显式提供已审计的 `inputs.snapshot`，先用训练完成后的同一离线回放后端加载文件，缺失、损坏或无法加载时在偏好包/tokenizer 审计前失败，不接受仅在训练配置中提供的隐式快照。关闭验证时不加载快照，也不要求该输入。随后复用现有 `inspect_only=True` 分支，保留参数、严格偏好包、原始 SFT 来源、真实模板前缀/目标 token 边界和长度审计；不构造策略，也不创建训练输出。
- GRPO：复用现有正式任务切分选择、快照标准链语义检查、工具注册、隐藏任务标识及 datasets 内存转换检查、源模型/实际 tokenizer 模板协议和标准动作生成长度保护；不运行策略采样或工具驱动的模型 episode。

Tokenizer 一律 `local_files_only=True`。管理层还须通过无 GPU、无网络、只读根文件系统的容器边界执行本入口；仅 Python 检查本身不能替代容器权限隔离。管理端的资源、空间、镜像与归属检查仍独立执行。

成功时退出码为 0，标准输出是单个 `schema_version=1`、`status="passed"`、阶段与检查摘要 JSON；失败为退出码 2、`status="failed"` 与具体错误。原有审计日志转到标准错误，不污染机器输出。成功结果明确声明 `model_weights_loaded=false`、`training_started=false`，不将预检冒充真实恢复验证。

工程测试仅检查参数拒绝、既有审计接线、现有留出切分拒绝、输出格式与无写入边界，使用控制元数据和既有仓库路径，不创建研究样本、真实 tokenizer 或模型。当前未运行真实检查容器或服务器任务，缺少本地真实模型、偏好包或快照时，不生成替代输入证明通过。
