# 运行记录、资源保护和停止

管理命令由 profile 指定的普通用户执行，UID 与当前用户一致，GID 为其所属组；不自动改所有权。运行根目录与外部研究输入、显式缓存分离。输入路径先规范化，拒绝符号链接、上级跳转、越界以及整个允许根目录的宽挂载。已完成的受管 `final` 可作为历史只读输入，不要求复制到外部目录。

## 记录结构

```text
experiment-root/
  .management.lock
  gpu-claims.json
  registrations/<artifact-id>/reference.json
  runs/<run-id>/
    run.json
    resolved-config.json
    inputs.json
    attempts/<attempt-id>/
      attempt.json
      image-metadata.json
      training-config.json
      docker-command.json
      events.jsonl
      status.json
      control/managed-context.json
      control/stop-request.json
      control/stop-status.json
      control/training-result.json
      logs/
      output/
    evaluations/<evaluation-id>/evaluation.json
```

配置与输入身份在首次创建时保存；后续尝试不得修改旧输出、轨迹和 checkpoint。恢复必须显式指定本运行任一历史尝试的已提交 checkpoint，验证文件内容、原预算、相同镜像与 GPU UUID/rank 顺序后建立新尝试。恢复加载失败时，仍可显式选择之前的完整恢复点，不自动选择“最新”或其他断点。

预算明确选择优化步或 epochs：步数预算与原 `max_steps` 一致；epochs 预算保持原 `max_steps=-1` 和 `epochs` 不变。管理器不估算 epochs 对应步数，而由训练器在第一次优化之前核定完整总步数，按尝试持久化 `control/resolved-budget.json` 并绑定 checkpoint。恢复承接这份已核定总步数，不重设 epochs 或发放新预算。核定之前展示预算未解析，不伪造训练进度。

JSON 写入采用同文件系统临时文件、文件持久化、原子替换及目录持久化；短期文件锁保护创建和更新。GPU 声明持久化，启动命令退出后仍有效。并发争用只允许一个请求；发现其他计算进程直接拒绝，不排队、换卡或杀其他任务。该保护仅覆盖同一管理根，不是跨用户调度或硬件独占保证，外部用户仍可能在复查后启动任务。

## 启动与未知结果

预检使用无 GPU、离线、只读的短生命周期容器，通过标准输入传结构化配置；不会建立训练运行或启动优化。预检容器自动清理不等于任务容器自动删除。正式任务保留容器及证据，使用普通用户、明确 GPU UUID、只读根文件系统、最小只读输入挂载及当前尝试的 `output/control/logs` 可写挂载。不会挂载整个工作区源码、主目录、Docker socket，或发布端口。

创建容器之前先保存唯一尝试、最终 `compose.json`、参数数组和创建意图，随后保存确切容器 ID，再启动。客户端响应丢失时仅核验同一个容器，不再次创建或启动。核验同时要求名称、标签、run/attempt、UID/GID、镜像和设备映射一致；名称相似不算归属。GPU UUID 顺序显式写入 `CUDA_VISIBLE_DEVICES`，DDP 启动器沿用同一 UUID 顺序，不假定 Docker 设备声明顺序就是 CUDA rank 顺序。容器暂时不存在也不足以证明超时请求未发生，结果未知时保留声明，等待明确查询。

若 `docker start` 返回明确拒绝，并核验同一容器仍为 `created` 且未运行，则记录启动失败（执行状态 `create_failed`）并释放本次 GPU 声明，保留原容器和错误记录。明确拒绝但暂时无法核验时先保留声明，后续 `status` 核验未启动后结束尝试；真正的超时或响应丢失不能仅凭 `created` 就释放声明。不会自动重试或删除容器。

`status` 是证据快照，分别展示执行、训练、阶段验证；正式评测另有独立结果。退出码 0、`final` 目录存在均不能证明训练预算完成或验证通过。Docker 不可达时执行状态为未知，保留历史事实及 GPU 声明；查明核验过的原容器已退出，或确认上述明确启动失败后才释放声明。没有训练结果证据时不会补造成功。

## 停止与日志

默认停止只写带当前尝试身份的幂等请求，不调用 `docker stop`，不设超时自动升级。长采样、工具调用或通信尚未返回时仍显示停止处理中，不能承诺立即释放 GPU。只有完整 checkpoint 重新校验通过、训练确认安全停止且容器已退出，才展示安全停止。失败、保护终止和预算完成不被并发用户请求掩盖。

显式 `--force` 核对目标身份后只对其确切容器 ID 发强制中断，并保留强制请求事件。它不伪造新 checkpoint，也不删除旧容器或恢复点；只承诺可以检查此前完整提交的断点，不保证最新进度已保存。评测停止不产生训练恢复点，评测失败或中断也不改变原训练的预算事实。

日志读取无训练副作用；主进程读取确切容器日志，rank 日志限当前尝试和已分配 rank。已有 TensorBoard 路径、实际分段时间、分配 GPU-hours、失败重算均可检查，缺少指标标为未知，不能用零填补或自动宣称 C1/M4 等预算。

## 本地验证边界

`tests/test_experiment_runtime.py` 使用控制元数据、临时目录和注入的命令执行替身，覆盖并发事务、路径最小权限、持久争用、容器身份、创建/启动响应丢失、未知声明保留、三维状态、停止及三阶段配置经真实 CLI 的往返检查。这些不是模型训练或真实 GPU 恢复证据。镜像构建、服务器权限、GPU 运行时、真实单卡/四卡训练恢复与模型质量评测，均需另行授权并单独验证。
