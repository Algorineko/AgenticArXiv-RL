"""训练/留出集切分。

随机按任务切会泄漏：同一模板的不同参数实例分别落在训练与测试两侧，
测出来的「泛化」其实是记忆。所以切分必须在模板层面进行。
"""

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.splits import (
    DEFAULT_SPLIT_PATH,
    difficulty_band,
    load_split,
    make_split,
    rl_train_ids,
    summarize,
    template_key,
)
from benchmark.tasks_expanded import EXPANDED_TASKS

PINNED_PATH = Path(__file__).resolve().parents[2] / "data" / "splits" / "v1.json"


def _task(tid, category="search", tools=1, template=None):
    t = {"id": tid, "category": category,
         "expected_tools": ["get_recently_submitted_cs_papers"] * tools}
    if template:
        t["template"] = template
    return t


def _family(prefix, n, category="search", tools=1, template=None):
    return [_task(f"{prefix}_{i}", category, tools, template) for i in range(n)]


class TemplateKeyTest(unittest.TestCase):
    def test_same_category_and_chain_length_share_a_key(self):
        self.assertEqual(template_key(_task("a")), template_key(_task("b")))

    def test_chain_length_separates_keys(self):
        self.assertNotEqual(template_key(_task("a", tools=1)), template_key(_task("b", tools=2)))

    def test_explicit_template_field_overrides_category(self):
        # 同一 category 下区分能力不同的子族（如指代形态的对照组与压力组）
        a = _task("a", category="ref_form", template="ref_ctrl")
        b = _task("b", category="ref_form", template="ref_stress")
        self.assertNotEqual(template_key(a), template_key(b))

    def test_missing_fields_do_not_crash(self):
        self.assertEqual(template_key({"id": "x"}), ("?", 0))


class DifficultyBandTest(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(difficulty_band(0.0), "floor")
        self.assertEqual(difficulty_band(0.2), "middle")     # 边界归中间带
        self.assertEqual(difficulty_band(0.8), "middle")
        self.assertEqual(difficulty_band(1.0), "ceiling")


class MakeSplitTest(unittest.TestCase):
    def setUp(self):
        self.tasks = (_family("s", 8) + _family("m", 4, "composite", 2)
                      + _family("t", 4, "state", 1))

    def _split(self, **kw):
        return make_split(self.tasks, **kw)

    def test_every_task_lands_in_exactly_one_split(self):
        split = self._split()
        allocated = [tid for ids in split.values() for tid in ids]
        self.assertEqual(sorted(allocated), sorted(t["id"] for t in self.tasks))
        self.assertEqual(len(allocated), len(set(allocated)))

    def test_ood_keys_are_held_out_whole(self):
        split = self._split(ood_keys=[("composite", 2)])
        self.assertEqual(split["ood_test"], sorted(t["id"] for t in self.tasks
                                                   if t["category"] == "composite"))
        self.assertFalse(any(tid.startswith("m_") for tid in split["train"]))

    def test_no_template_appears_in_both_train_and_ood(self):
        split = self._split(ood_keys=[("state", 1)])
        by_id = {t["id"]: t for t in self.tasks}
        train_keys = {template_key(by_id[t]) for t in split["train"]}
        ood_keys = {template_key(by_id[t]) for t in split["ood_test"]}
        self.assertFalse(train_keys & ood_keys)

    def test_iid_templates_are_all_seen_in_training(self):
        # iid 的含义就是「同模板、未见过的实例」，模板必须在训练侧出现过
        split = self._split()
        by_id = {t["id"]: t for t in self.tasks}
        train_keys = {template_key(by_id[t]) for t in split["train"]}
        for tid in split["iid_test"]:
            self.assertIn(template_key(by_id[tid]), train_keys, tid)

    def test_a_template_is_never_emptied_into_iid(self):
        # 掏空一个模板就把它变成了 ood，两种泛化会被混淆
        split = make_split(_family("s", 2), iid_ratio=0.9)
        self.assertEqual(len(split["train"]), 1)
        self.assertEqual(len(split["iid_test"]), 1)

    def test_single_task_template_stays_in_training(self):
        split = make_split(_family("only", 1))
        self.assertEqual(split["train"], ["only_0"])
        self.assertEqual(split["iid_test"], [])

    def test_is_deterministic_for_a_given_seed(self):
        self.assertEqual(self._split(seed=7), self._split(seed=7))

    def test_seed_changes_the_selection(self):
        splits = {tuple(self._split(seed=s)["iid_test"]) for s in range(6)}
        self.assertGreater(len(splits), 1)

    def test_stratifies_by_difficulty_when_rates_are_given(self):
        # 8 条里 4 条 ceiling 4 条 middle，留出 50% 时两档都该被抽到
        tasks = _family("s", 8)
        rates = {f"s_{i}": (1.0 if i < 4 else 0.5) for i in range(8)}
        split = make_split(tasks, iid_ratio=0.5, rates=rates, seed=1)
        held = {difficulty_band(rates[t]) for t in split["iid_test"]}
        self.assertEqual(held, {"ceiling", "middle"})

    def test_rejects_unknown_ood_key(self):
        with self.assertRaises(ValueError):
            self._split(ood_keys=[("nonexistent", 9)])

    def test_rejects_out_of_range_ratio(self):
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                self._split(iid_ratio=bad)


class RlTrainIdsTest(unittest.TestCase):
    """成功率贴近 0 或 1 的任务对 GRPO 不产生梯度，不该进 RL 训练集。"""

    def test_keeps_only_middle_band_tasks_from_train(self):
        tasks = _family("s", 4)
        rates = {"s_0": 0.0, "s_1": 0.5, "s_2": 0.6, "s_3": 1.0}
        split = make_split(tasks, iid_ratio=0.0, rates=rates, seed=0)
        self.assertEqual(rl_train_ids(split, rates), ["s_1", "s_2"])

    def test_never_reaches_into_the_test_splits(self):
        tasks = _family("s", 4) + _family("m", 2, "composite", 2)
        rates = {t["id"]: 0.5 for t in tasks}
        split = make_split(tasks, ood_keys=[("composite", 2)], rates=rates, seed=0)
        self.assertFalse(set(rl_train_ids(split, rates)) & set(split["ood_test"]))
        self.assertFalse(set(rl_train_ids(split, rates)) & set(split["iid_test"]))

    def test_tasks_without_a_measured_rate_are_excluded(self):
        tasks = _family("s", 2)
        split = make_split(tasks, iid_ratio=0.0)
        self.assertEqual(rl_train_ids(split, {"s_0": 0.5}), ["s_0"])


class SummarizeTest(unittest.TestCase):
    def test_counts_tasks_templates_and_bands(self):
        tasks = _family("s", 4) + _family("m", 2, "composite", 2)
        rates = {t["id"]: 0.5 for t in tasks}
        split = make_split(tasks, ood_keys=[("composite", 2)], rates=rates, seed=0)
        out = summarize(split, tasks, rates)
        self.assertEqual(out["ood_test"]["count"], 2)
        self.assertEqual(out["ood_test"]["templates"], 1)
        self.assertEqual(out["train"]["bands"]["middle"], out["train"]["count"])

    def test_marks_tasks_without_rates_as_unknown(self):
        tasks = _family("s", 2)
        out = summarize(make_split(tasks, iid_ratio=0.0), tasks)
        self.assertEqual(out["train"]["bands"], {"unknown": 2})


class PinnedV1SplitTest(unittest.TestCase):
    """data/splits/v1.json 必须跟得上任务集。

    这个 pin 存在的意义是让训练前后两次运行引用同一份切分。可它是手工
    落盘的静态文件，任务集一扩它就悄悄漏掉新任务——切分依然「有效」，
    只是少了几条，没有任何东西会报错。同类漂移在本仓库已经发生过两次
    （README 任务条数 7→8、58→59），所以这里直接钉死。
    """

    @classmethod
    def setUpClass(cls):
        cls.pinned = json.loads(PINNED_PATH.read_text(encoding="utf-8"))
        cls.split = cls.pinned["split"]
        cls.ids = {t["id"] for t in EXPANDED_TASKS}

    def test_covers_every_task_exactly_once(self):
        assigned = [tid for ids in self.split.values() for tid in ids]
        self.assertEqual(len(assigned), len(set(assigned)), "有任务被切到多份里")
        self.assertEqual(set(assigned), self.ids)

    def test_is_reproducible_from_the_recorded_seed_and_keys(self):
        """记下 seed / ood_keys / rates 就是为了能重算出同一份切分。"""
        rebuilt = make_split(
            EXPANDED_TASKS,
            ood_keys=[tuple(k) for k in self.pinned["ood_keys"]],
            seed=self.pinned["seed"],
            rates=self.pinned["rates"],
        )
        self.assertEqual(rebuilt, self.split)

    def test_no_template_straddles_train_and_the_held_out_sets(self):
        by_id = {t["id"]: t for t in EXPANDED_TASKS}
        keys = {name: {template_key(by_id[tid]) for tid in ids}
                for name, ids in self.split.items()}
        self.assertEqual(keys["train"] & keys["ood_test"], set())

    def test_recorded_rates_refer_to_real_tasks(self):
        self.assertEqual(set(self.pinned["rates"]) - self.ids, set())


class LoadSplitTest(unittest.TestCase):
    """按名字取切分：benchmark 与训练脚本共用一份读取逻辑。

    两处各写一遍 json.load 就是两处会各自漂的地方，而「训练用了哪一份切分」
    这件事一旦对不上，训练前后的数字就不可比 —— 那正是切分存在的理由。
    """

    def test_bare_name_uses_the_default_pinned_file(self):
        self.assertEqual(load_split("iid_test"),
                         json.loads(DEFAULT_SPLIT_PATH.read_text())["split"]["iid_test"])

    def test_explicit_path_overrides_the_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v9.json"
            path.write_text(json.dumps({"split": {"train": ["only_one"]}}))
            self.assertEqual(load_split(f"{path}:train"), ["only_one"])

    def test_rl_train_is_computed_not_stored(self):
        """rl_train 不在文件里，是 train 与 rates 算出来的。"""
        payload = json.loads(DEFAULT_SPLIT_PATH.read_text())
        self.assertNotIn("rl_train", payload["split"])
        self.assertEqual(load_split("rl_train"),
                         rl_train_ids(payload["split"], payload["rates"]))

    def test_rl_train_is_a_subset_of_train_in_the_middle_band(self):
        payload = json.loads(DEFAULT_SPLIT_PATH.read_text())
        rates = payload["rates"]
        ids = load_split("rl_train")
        self.assertTrue(set(ids) <= set(payload["split"]["train"]))
        for tid in ids:
            self.assertEqual(difficulty_band(rates[tid]), "middle", tid)

    def test_unknown_name_lists_what_is_available(self):
        with self.assertRaises(ValueError) as ctx:
            load_split("nope")
        message = str(ctx.exception)
        self.assertIn("rl_train", message)
        self.assertIn("iid_test", message)

    def test_missing_file_is_reported_as_such(self):
        with self.assertRaises(FileNotFoundError):
            load_split("/nonexistent/path.json:train")

    def test_rl_train_without_rates_is_an_error_not_an_empty_set(self):
        """没有 rates 就没有中间带。静默返回空集会让训练集悄悄变空。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "norates.json"
            path.write_text(json.dumps({"split": {"train": ["a", "b"]}}))
            with self.assertRaises(ValueError):
                load_split(f"{path}:rl_train")

    def test_rl_train_that_comes_out_empty_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "allceiling.json"
            path.write_text(json.dumps(
                {"split": {"train": ["a", "b"]}, "rates": {"a": 1.0, "b": 0.0}}))
            with self.assertRaises(ValueError):
                load_split(f"{path}:rl_train")


if __name__ == "__main__":
    unittest.main()
