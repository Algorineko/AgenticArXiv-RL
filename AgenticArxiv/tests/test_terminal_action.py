"""JSON 外壳包裹的终止动作不应被当成工具调用。

模型有时不写裸 `Action: FINISH`，而是套上和其他动作一样的 JSON 外壳
`{"name": "FINISH", "args": {}}` —— prompt 里其余动作都是 JSON，这么写很自然。
修复前它会被当成一次工具调用：白跑一轮迭代（工具执行失败，错误再喂回模型），
并在 tool_call_sequence 里多出一个 "FINISH"，把本来完全正确的轨迹判成
accurate=False。

实测（Qwen3.5-9B，43 任务 × 3 范式 × 2 次 = 258 次运行）：命中 5 次
（regex 3 / mcp 2），5 次全部被误判为 accurate=False，其中至少 2 次实际已
正确完成任务，例如：

    get_recently_submitted_cs_papers,download_arxiv_pdf,get_paper_cache_status,
    translate_arxiv_pdf,FINISH        <- 完整正确序列，仅多出末尾的 FINISH

分布并不均匀：在需要条件分支的任务上命中率可达 50%（8/16），足以把该任务的
成功率整体压到 0。skill_cli 不受影响 —— 它的解析器要求子命令必须是 4 个
CLI 名之一，"FINISH" 匹配不上就走终止分支；实测 0 次命中，与此一致。
"""

import json
import unittest

from agents.base_agent import TERMINAL_ACTIONS, is_terminal_action
from agents.agent_engine import ReActAgent
from benchmark.metrics import _check_tool_sequence, _extract_tool_sequence


def _step(action, observation=""):
    return {"thought": "t", "action": action, "observation": observation}


class IsTerminalActionTest(unittest.TestCase):
    def test_accepts_every_terminal_marker(self):
        for name in TERMINAL_ACTIONS:
            self.assertTrue(is_terminal_action(name), name)

    def test_is_case_and_whitespace_insensitive(self):
        for text in ("finish", " FINISH ", "Finish\n"):
            self.assertTrue(is_terminal_action(text), repr(text))

    def test_rejects_tool_names_and_non_strings(self):
        for value in ("download_arxiv_pdf", "", None, 123, {"name": "FINISH"}):
            self.assertFalse(is_terminal_action(value), repr(value))


class ReActParseTerminalTest(unittest.TestCase):
    """_parse_react_text 不使用 self，可用哑实例直接调用。"""

    def _parse(self, response):
        return ReActAgent._parse_react_text(None, response)

    def test_bare_finish_terminates(self):
        _, action = self._parse("Thought: 完成了\nAction: FINISH")
        self.assertIsNone(action)

    def test_json_wrapped_finish_terminates(self):
        _, action = self._parse(
            'Thought: 完成了\nAction: {"name": "FINISH", "args": {}}'
        )
        self.assertIsNone(action)

    def test_json_wrapped_force_stop_terminates(self):
        _, action = self._parse(
            'Thought: 停\nAction: {"name": "FORCE_STOP", "args": {}}'
        )
        self.assertIsNone(action)

    def test_real_tool_call_still_parses(self):
        _, action = self._parse(
            'Thought: 下载\nAction: {"name": "download_arxiv_pdf", "args": {"ref": 1}}'
        )
        self.assertEqual(action, {"name": "download_arxiv_pdf", "args": {"ref": 1}})


class MetricsTerminalActionTest(unittest.TestCase):
    """兜底层：历史结果文件和第三方 Agent 的轨迹仍可能带 JSON 终止动作。"""

    FINISH_JSON = json.dumps({"name": "FINISH", "args": {}})

    def test_json_finish_is_not_counted_as_a_tool(self):
        history = [
            _step(json.dumps({"name": "download_arxiv_pdf", "args": {"ref": 1}})),
            _step(self.FINISH_JSON),
        ]
        self.assertEqual(_extract_tool_sequence(history), ["download_arxiv_pdf"])

    def test_bare_terminal_actions_still_excluded(self):
        history = [
            _step(json.dumps({"name": "download_arxiv_pdf", "args": {"ref": 1}})),
            _step("FINISH"),
        ]
        self.assertEqual(_extract_tool_sequence(history), ["download_arxiv_pdf"])

    def test_correct_trajectory_is_no_longer_judged_inaccurate(self):
        # 实测样本 ref_stress_image_restoration/regex：正确完成却被判 False
        history = [
            _step(json.dumps({"name": "download_arxiv_pdf", "args": {"ref": 1}})),
            _step(self.FINISH_JSON),
        ]
        self.assertTrue(
            _check_tool_sequence(
                _extract_tool_sequence(history), ["download_arxiv_pdf"]
            )
        )


if __name__ == "__main__":
    unittest.main()
