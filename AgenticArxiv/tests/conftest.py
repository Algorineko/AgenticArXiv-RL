"""测试进程的产物目录隔离。

`RolloutSandbox` 的 reset 契约是「把显式给出的产物根恢复到构造时的基线」——
它会删掉根目录下任何不在基线里的文件。`AgenticArxivMultiTurnEnv` 默认拿
`settings.pdf_raw_path` / `pdf_translated_path` / `figures_path` 作为这些根，
而测试会实例化这个环境。

两者相遇就是一个真实的破坏性 bug：只要测试运行时 `output/pdf_raw` 里有文件
（比如刚跑过 `rl.build_snapshot`，或者 build_snapshot 正**同时**在预取），
一次 sandbox reset 就会把这些文件删掉——它们不在测试构造环境时的基线里。
一次全量快照构建要花几十分钟下载几百个 PDF，被 pytest 顺手清空是不能接受的。

所以测试进程一律把产物根指向临时目录。这里在 conftest 里设置环境变量，
早于任何测试模块导入 `config`（`settings` 在 config 导入时按环境变量求值），
因此不需要改动任何既有测试。
"""

import os
import tempfile

_ARTIFACT_ENV_VARS = (
    "PDF_RAW_PATH",
    "PDF_TRANSLATED_PATH",
    "PDF_TRANSLATED_LOG_PATH",
    "PDF_CACHE_PATH",
    "TRANSLATE_CACHE_PATH",
    "PDF_FIGURES_PATH",
)

_scratch = tempfile.mkdtemp(prefix="aa-rl-tests-")
for _name in _ARTIFACT_ENV_VARS:
    os.environ[_name] = os.path.join(_scratch, _name.lower())
