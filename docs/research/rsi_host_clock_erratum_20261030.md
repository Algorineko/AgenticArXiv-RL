# AppWorld 宿主计时修正（2026-10-30）

发现：AppWorld 的模拟时间上下文同时冻结宿主 `datetime.now` 和 `time.monotonic`。旧 API 客户端因此记录 0 秒请求耗时和任务内模拟日期，而非真实请求时间。简单的 0.05 秒 sleep 在上下文内分别测得普通 monotonic 增量 0、系统原生 monotonic 约 0.05 秒。

修正：`rsi_api_pilot.py` 的 API 限速与耗时改用 `time.clock_gettime(CLOCK_MONOTONIC)`，UTC 使用 `CLOCK_REALTIME` 显式转换。本实现针对当前 Linux 研究服务器。环境自身时间不改动，以保留任务语义。

独立验证：`scripts/verify_rsi_host_clock.py` 在环境外和已暴露 train 场景内部均等待 0.08 秒；修正后的计时都约 0.08 秒且 UTC 与系统时钟相符，环境原 monotonic 仍冻结。无 API 调用、无 GPU、未读取最终任务。

历史影响：既有 AppWorld 运行的 elapsed_s 和逐请求/episode UTC 不可用于 wall-clock 效率或请求间隔分析；原先配置 15 秒限速不能从这些日志确证为实际间隔。请求次数、提供方 token 用量、预算累计、工具观察和独立终局判定不依赖这些时间字段，保留。当前正在运行的 24 步 horizon-v1 进程加载的是修正前的冻结源码，完成后仍须明确这一局限；不覆盖历史原始日志、不回填猜测时间。未来运行使用新代码和新源码摘要。
