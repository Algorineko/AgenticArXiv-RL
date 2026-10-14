"""测试进程的产物目录隔离。

`RolloutSandbox` 的 reset 契约是「把显式给出的产物根恢复到构造时的基线」——
它会删掉根目录下任何不在基线里的文件。`AgenticArxivMultiTurnEnv` 默认拿
`settings.pdf_raw_path` / `pdf_translated_path` / `figures_path` 作为这些根，
而测试会实例化这个环境。

两者相遇就是一个真实的破坏性 bug：只要测试运行时 `output/pdf_raw` 里有文件
（比如刚跑过 `data snapshot`，或者 build_snapshot 正**同时**在预取），
一次 sandbox reset 就会把这些文件删掉——它们不在测试构造环境时的基线里。
一次全量快照构建要花几十分钟下载几百个 PDF，被 pytest 顺手清空是不能接受的。

所以测试进程一律把产物根指向临时目录。这里在 conftest 里设置环境变量，
早于任何测试模块首次使用 `settings`，因此延迟配置与旧调用都读取隔离根。
"""

import os
import tempfile

import pytest

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

# 工程回归只复用现有输入；即使误入可选后端也不能下载模型或联网补材料。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"

# 包导入不读取环境；测试在首次业务对象构造前显式绑定临时产物根。
from agentic_arxiv.config import initialize_operation

initialize_operation(load_env=False)


def pytest_collection_modifyitems(items):
    for item in items:
        if "network" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="原手动网络冒烟不属于离线工程验收"))


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """保留已有 mock 请求测试，拒绝遗漏 mock 的真实连接。"""
    import socket

    def reject_connection(*args, **kwargs):
        raise AssertionError("离线工程回归禁止真实网络连接")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    monkeypatch.setattr(socket.socket, "connect", reject_connection)
