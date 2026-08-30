"""工具注册的测试。

核心是一条：**缺一个第三方依赖不能让四个工具全都消失**。

原来三个 Agent 各自复制了同一段代码，四个 import 共用一个 try。
`tools.arxiv_tool` 依赖第三方包 `arxiv` 且排在第一个，环境里没有它时，
连纯本地零依赖的 `cache_status_tool` 也一起注册不上。而 registry 为空时
模型不会报错 —— 它会编工具名，benchmark 照样跑完并写出成功率。
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.arxiv_tool  # noqa: F401,E402
import tools.cache_status_tool  # noqa: F401,E402
import tools.pdf_download_tool  # noqa: F401,E402
import tools.pdf_translate_tool  # noqa: F401,E402
from tools.bootstrap import (  # noqa: E402
    TOOL_MODULES,
    missing_tools,
    register_all_tools,
    registered_tool_count,
    require_all_tools,
)
from tools.tool_registry import registry  # noqa: E402


class RegisterAllToolsTest(unittest.TestCase):
    def test_all_tools_register_in_a_healthy_environment(self):
        self.assertEqual(register_all_tools(), {})
        self.assertEqual(missing_tools(), [])
        expected_count = sum(len(names) for names in TOOL_MODULES.values())
        self.assertEqual(registered_tool_count(), expected_count)
        self.assertIn("search_arxiv_papers", {
            tool["name"] for tool in registry.list_tools()
        })

    def test_one_broken_module_does_not_take_down_the_rest(self):
        """这是整个模块存在的理由。

        缺 `arxiv` 时只应该少 get_recently_submitted_cs_papers 一个，
        另外三个纯本地工具必须照常可用。
        """
        real_import = __import__

        def fake_import(name, *a, **kw):
            if name == "tools.arxiv_tool":
                raise ModuleNotFoundError("No module named 'arxiv'")
            return real_import(name, *a, **kw)

        with mock.patch("importlib.import_module") as im:
            def side_effect(module):
                if module == "tools.arxiv_tool":
                    raise ModuleNotFoundError("No module named 'arxiv'")
                return real_import(module)
            im.side_effect = side_effect
            failures = register_all_tools()

        self.assertEqual(sorted(failures), ["tools.arxiv_tool"])
        self.assertIn("arxiv", failures["tools.arxiv_tool"])
        # 其余三个模块仍然被尝试导入了，没有因为第一个失败而跳过
        self.assertEqual(len(failures), 1)


class RequireAllToolsTest(unittest.TestCase):
    def test_passes_when_everything_is_registered(self):
        require_all_tools("单元测试")     # 不抛异常即可

    def test_raises_with_actionable_message_when_a_tool_is_missing(self):
        with mock.patch("tools.bootstrap.missing_tools",
                        return_value=["get_recently_submitted_cs_papers"]), \
             mock.patch("tools.bootstrap.register_all_tools",
                        return_value={"tools.arxiv_tool": "ModuleNotFoundError: No module named 'arxiv'"}):
            with self.assertRaises(SystemExit) as ctx:
                require_all_tools("Benchmark")
        message = str(ctx.exception)
        self.assertIn("Benchmark", message)
        self.assertIn("get_recently_submitted_cs_papers", message)
        self.assertIn("pip install arxiv", message)
        # 必须点明后果，否则读的人会以为「少个工具而已，能跑就行」
        self.assertIn("编造工具名", message)

    def test_benchmark_entrypoint_checks_before_running(self):
        """跑 96 条之后才发现工具没装上，代价太大。"""
        source = (Path(__file__).resolve().parents[1]
                  / "benchmark" / "run_benchmark.py").read_text(encoding="utf-8")
        self.assertIn("require_all_tools", source)
        self.assertLess(source.index("require_all_tools(\"Benchmark\")"),
                        source.index("BenchmarkRunner("),
                        "校验必须在真正开跑之前")


class AgentsUseTheSharedBootstrapTest(unittest.TestCase):
    """三个 Agent 曾各自复制同一段注册代码，改一处漏两处。"""

    AGENTS = ("agents/agent_engine.py", "mcp_protocol/mcp_agent.py",
              "skill_cli/skill_agent.py")

    def test_no_agent_keeps_its_own_shared_try_block(self):
        root = Path(__file__).resolve().parents[1]
        for rel in self.AGENTS:
            source = (root / rel).read_text(encoding="utf-8")
            with self.subTest(agent=rel):
                self.assertIn("register_all_tools", source)
                self.assertNotIn("import tools.arxiv_tool", source,
                                 f"{rel} 仍在自己导入工具模块")

    def test_bootstrap_import_comes_after_sys_path_setup(self):
        """tools 包要等 PROJECT_ROOT 进 sys.path 之后才可导入。"""
        root = Path(__file__).resolve().parents[1]
        for rel in self.AGENTS:
            lines = (root / rel).read_text(encoding="utf-8").split("\n")
            path_line = next((i for i, l in enumerate(lines)
                              if "sys.path.insert" in l), None)
            boot_line = next((i for i, l in enumerate(lines)
                              if "from tools.bootstrap import" in l), None)
            with self.subTest(agent=rel):
                self.assertIsNotNone(boot_line, f"{rel} 没有引入 bootstrap")
                if path_line is not None:
                    self.assertGreater(boot_line, path_line,
                                       f"{rel} 的 bootstrap import 早于 sys.path 设置")


if __name__ == "__main__":
    unittest.main()
