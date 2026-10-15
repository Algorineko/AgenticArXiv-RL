# 非活动行为基线归档

这两个文件在实施起点为未跟踪文件，未纳入活动 CI。本次只按原字节移动，不修复旧导入、不运行、不增加输入，**不作为当前验收证据**。

| 原路径 | 归档路径 | SHA256 |
| --- | --- | --- |
| `AgenticArxiv/tests/test_mainline_behavior_baseline.py` | `docs/archive/mainline-behavior-baseline/test_mainline_behavior_baseline.py.txt` | `a8b57aab695153b4303eae43035294a080d7b473881a62971860b94bb551362a` |
| `artifacts/mainline_baseline/behavior.json` | `docs/archive/mainline-behavior-baseline/behavior.json` | `74977433596dcd7174a507dc782f7a249741da806e46d350ca1e70766b66365b` |

原位置和原哈希还保存在 [独立源码映射](../../../scripts/maintenance/source_map.json)。测试使用 `.py.txt` 后缀，根测试发现仅覆盖 `tests/`；安装包仅包含 `src/agentic_arxiv/`，不会包含此归档。其他旧资产、报告与旧迁移证据不适用这个移动例外，仍保持原路径和内容。
