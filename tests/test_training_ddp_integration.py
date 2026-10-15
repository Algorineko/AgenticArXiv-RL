"""四进程 CPU/Gloo 工程回归：不加载模型、真实 tokenizer 或研究数据。

仅对微型张量反传，调用真实 Trainer/TRL 的归一化和采样接线。
不能替代 CUDA/NCCL/4-bit 多卡硬件验收。
"""

import json
from pathlib import Path
import time

import pytest


class _CharacterTokenizer:
    chat_template = "synthetic-role-boundaries"
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = "".join(f"<{m['role']}>{m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(ch) + 10 for ch in text] if tokenize else text

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(token) - 10) for token in ids if int(token) >= 10)

    def batch_decode(self, values, skip_special_tokens=True):
        return [self.decode(ids) for ids in values]


class _Environment:
    def reset(self, **kwargs):
        pass

    def get_recently_submitted_cs_papers(self, **kwargs):
        return [{"id": "synthetic", "title": "unit-test paper"}]

    def close(self):
        pass


def _gloo_worker(rank, world_size, directory):
    import datetime
    import os
    import sys
    from collections import defaultdict
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from accelerate import PartialState
    from accelerate.data_loader import prepare_data_loader
    from transformers import Trainer
    from trl import GRPOConfig, GRPOTrainer
    from trl.trainer.utils import RepeatSampler

    from agentic_arxiv.training.canary import CanaryCallback, CanaryResult
    from agentic_arxiv.training.distributed import TrainingContext
    from agentic_arxiv.training.grpo_reward import make_grpo_reward_fn, make_multiturn_rollout_func
    from agentic_arxiv.training.observability import RewardComponentTracker
    from agentic_arxiv.training.rollout_audit import RolloutAuditWriter

    root = Path(directory)
    os.environ.update({"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size),
                       "LOCAL_WORLD_SIZE": str(world_size), "MASTER_ADDR": "127.0.0.1",
                       "MASTER_PORT": "29500", "OMP_NUM_THREADS": "1",
                       "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo"})
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=60))
    try:
        PartialState(cpu=True)
        context = TrainingContext.from_environment()

        def gather_tensor(value):
            output = [torch.empty_like(value) for _ in range(world_size)]
            dist.all_gather(output, value)
            return torch.stack(output) if value.dim() == 0 else torch.cat(output)

        accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=world_size,
                                      process_index=rank, is_main_process=rank == 0,
                                      distributed_type="MULTI_CPU", gather=gather_tensor)

        # SFT 数据分片只交给 Accelerate，不额外重复或套第二个 DistributedSampler。
        loader = prepare_data_loader(DataLoader(list(range(16)), batch_size=1),
                                     num_processes=world_size, process_index=rank)
        partitions = context.gather([int(batch.item()) for batch in loader])
        assert sorted(item for part in partitions for item in part) == list(range(16))

        # 每个 rank 的两个微批次有不同监督长度，prompt/padding labels=-100。
        class TokenLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[0.2, -0.3], [0.4, 0.1], [-0.2, 0.5]], dtype=torch.float64))

            def forward(self, input_ids, labels, num_items_in_batch):
                valid = labels.ne(-100)
                loss = torch.nn.functional.cross_entropy((input_ids @ self.weight)[valid], labels[valid], reduction="sum")
                return {"loss": loss / num_items_in_batch}

        def batch_for(process, microbatch):
            count = process + microbatch + 1
            labels = torch.tensor([-100] + [i % 2 for i in range(count)] + [-100] * (6 - count))
            features = torch.arange(len(labels) * 3, dtype=torch.float64).reshape(-1, 3) / 20 + process / 10
            return {"input_ids": features, "labels": labels}

        local_batches = [batch_for(rank, index) for index in range(2)]
        trainer = Trainer.__new__(Trainer)
        trainer.accelerator = accelerator
        trainer.args = SimpleNamespace(average_tokens_across_devices=True, world_size=world_size, n_gpu=0, past_index=-1)
        trainer.model_accepts_loss_kwargs = True
        trainer.compute_loss_func = None
        trainer.label_smoother = None
        denominator = trainer._get_num_items_in_batch(local_batches, torch.device("cpu"))
        assert int(denominator) == sum(process + microbatch + 1 for process in range(world_size) for microbatch in range(2))
        module = DistributedDataParallel(TokenLoss())
        for index, batch in enumerate(local_batches):
            with module.no_sync() if index == 0 else nullcontext():
                trainer.compute_loss(module, batch, num_items_in_batch=denominator).backward()
        reference = TokenLoss()
        losses = [reference(**batch_for(process, index), num_items_in_batch=denominator)["loss"]
                  for process in range(world_size) for index in range(2)]
        sum(losses).backward()
        torch.testing.assert_close(module.module.weight.grad, reference.weight.grad, rtol=1e-9, atol=1e-10)

        # GRPO 的原生 RepeatSampler + Accelerate 应把 G=4 的一组分到四个 rank。
        repeated = DataLoader(["t", "other"], batch_size=1,
                              sampler=RepeatSampler(["t", "other"], mini_repeat_count=4, shuffle=False))
        repeated = prepare_data_loader(repeated, num_processes=world_size, process_index=rank)
        assert context.gather(next(iter(repeated))[0]) == ["t"] * world_size
        config = GRPOConfig(output_dir=str(root / f"rank-{rank}"), use_cpu=True, bf16=False,
                            fp16=False, per_device_train_batch_size=1, gradient_accumulation_steps=1,
                            num_generations=4, report_to=[])
        assert config.generation_batch_size == 4 and config.per_device_train_batch_size == 1

        task = {"id": "t", "task": "unit test", "expected_tools": []}
        tracker = RewardComponentTracker()
        audit = RolloutAuditWriter(root / "rollouts.jsonl", num_generations=4,
                                   allowed_tools=["get_recently_submitted_cs_papers"]) if rank == 0 else None

        class Reward:
            def compute_reward_breakdown(self, task, result, training_step):
                return SimpleNamespace(total=float(rank)), None

        reward_fn = make_grpo_reward_fn({"t": task}, reward_calc=Reward(), tracker=tracker,
                                        auditor=audit, distributed=context, num_generations=4)
        native = GRPOTrainer.__new__(GRPOTrainer)
        native.accelerator = accelerator
        native.args = config
        native.state = SimpleNamespace(global_step=1, num_input_tokens_seen=0)
        native.model = SimpleNamespace(training=True, is_gradient_checkpointing=False)
        native.processing_class = _CharacterTokenizer()
        native.pad_token_id = 0
        native.eos_token_id = 2
        native.use_vllm = False
        native.environments = None
        native.tools = None
        native.num_generations = 4
        native.num_generations_eval = 4
        native.num_iterations = 1
        native.max_completion_length = 1024
        native.mask_truncated_completions = False
        native.beta = 0.0
        native.multi_objective_aggregation = config.multi_objective_aggregation
        native.scale_rewards = config.scale_rewards
        native.reward_funcs = [reward_fn]
        native.reward_func_names = [reward_fn.__name__]
        native.reward_processing_classes = [None]
        native.reward_weights = torch.ones(1)
        native._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        native._logs = {"prompt": [], "completion": [], "rewards": defaultdict(list), "advantages": []}
        tracker.bind(native)
        native.rollout_func = make_multiturn_rollout_func(_Environment, max_turns=2, tasks_by_id={"t": task})
        calls = 0

        def generate_single_turn(prompt_ids, images, fields):
            nonlocal calls
            calls += 1
            search = 'Thought: search\nAction: {"name":"get_recently_submitted_cs_papers","args":{"aspect":"AI","days":7,"max_results":5}}'
            finished = "Thought: done\nAction: FINISH"
            text = search if rank % 2 == 0 and calls == 1 else finished
            ids = [ord(ch) + 10 for ch in text] + ([] if text == search else [2])
            return [ids for _ in prompt_ids], None, {}

        native._generate_single_turn = generate_single_turn
        inputs = [{"prompt": [{"role": "user", "content": "unit test", "_task_id": "t"}], "task_id": "t"}]
        output = native._generate_and_score_completions(inputs)
        expected_advantage = (rank - 1.5) / (torch.arange(4, dtype=torch.float32).std() + 1e-4)
        torch.testing.assert_close(output["advantages"], expected_advantage.reshape(1))
        assert calls == (2 if rank % 2 == 0 else 1)
        assert bool((output["tool_mask"] == 0).any()) == (rank % 2 == 0)
        assert len(output["advantages"]) == 1  # 不能再次按 G 展开。
        assert native._metrics["train"]["frac_reward_zero_std"][-1] == 0.0
        stats = tracker.task_summary()["tasks"]["t"]
        assert stats["sample_count"] == 4 and stats["group_count"] == 1
        assert stats["informative_group_fraction"] == 1.0

        # Canary 只执行一次，但其停止决定必须到达全部 rank。
        class Evaluation:
            def evaluate(self, step):
                (root / f"canary-rank-{rank}.txt").write_text(str(step))
                return CanaryResult(step=step, mean_reward=-0.5,
                                    task_completion_rate=0.0, per_task_rewards={"t": -0.5})

        canary = CanaryCallback(Evaluation(), steps=1, min_reward=0.0, patience=1, distributed=context)
        control = SimpleNamespace(should_training_stop=False)
        canary.on_log(None, native.state, control)
        assert control.should_training_stop and canary.tripped
        context.run_on_main(lambda: (root / "main-only.txt").write_text(str(rank)))

        def fail_verification():
            raise ValueError("intentional verification failure")

        with pytest.raises((ValueError, RuntimeError), match="intentional verification failure"):
            context.run_on_main(fail_verification)
        (root / f"result-{rank}.json").write_text(json.dumps({"passed": True, "rank": rank,
                                                             "advantage": float(output["advantages"][0])}))
    finally:
        dist.destroy_process_group()


@pytest.mark.train
def test_four_process_gloo_sft_and_grpo_contracts(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("accelerate")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("需要 CPU/Gloo 分布式后端")
    processes = torch.multiprocessing.spawn(_gloo_worker, args=(4, str(tmp_path)), nprocs=4, join=False)
    deadline = time.monotonic() + 120
    try:
        while not processes.join(timeout=5):
            assert time.monotonic() < deadline, "DDP 工程回归超时，可能存在跨 rank 死锁"
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    assert [json.loads((tmp_path / f"result-{rank}.json").read_text())["passed"] for rank in range(4)] == [True] * 4
    records = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").read_text().splitlines()]
    assert len(records) == 4 and {record["group_index"] for record in records} == {0}
    assert [record["generation_index"] for record in records] == [0, 1, 2, 3]
    assert [record["reward"] for record in records] == [0.0, 1.0, 2.0, 3.0]
    assert [path.name for path in tmp_path.glob("canary-rank-*.txt")] == ["canary-rank-0.txt"]
    assert (tmp_path / "main-only.txt").read_text() == "0"
