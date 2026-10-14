# 一次性维护审计

这些脚本只读取显式工作区中的已有材料及已安装的 `agentic_arxiv` 包，不是受支持的业务命令入口。长期工具、任务和切分检查使用 `agentic-arxiv audit assets`。

先安装项目，再从任意目录使用脚本的绝对路径并明确工作区：

```bash
python /path/to/project/scripts/maintenance/audit_mainline_configuration.py --workspace /path/to/project
python /path/to/project/scripts/maintenance/audit_historical_records.py --workspace /path/to/project --json
```

默认独立来源映射为工作区内的 `scripts/maintenance/source_map.json`，也可用 `--source-map` 指定。映射保留实施起点旧源码指纹并列出拆分后的当前路径；审计分别报告旧身份和当前内容哈希，不要求新源码冒充旧哈希。缺少映射、当前文件或配置字段的定位时检查失败，不因搬迁自动豁免。

配置检查默认读取安装包内唯一的 `resources/mainline.json`。`--configuration` 和 `--baseline` 的相对路径均相对工作区，历史配置基线仍在原位置。检查只解析 AST，不导入训练器、不构造真实训练配置、不声称训练库有效值或历史训练等价。

历史检查只读取原报告和 traces，禁止联网和工具执行。当前只读评分与原历史值分别列出；缺少的模型响应、协议字段、token 用量及奖励保持未知，不重建参考答案。非活动行为基线和一次性工程审计输出均不作为历史行为输入。

两项审计默认只向 stdout 输出。若保存新报告，应另选独立输出目录（例如 `outputs/audits/`），不得覆盖 `artifacts/mainline_baseline/`、历史报告、manifest 或轨迹。临时迁移证据目录已按用户要求清理，审计不依赖它。真实模型、tokenizer、数据生成、训练及完整任务评测不属于这些检查。
