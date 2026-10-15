"""单机 DDP 启动器；与单卡后台任务分离，不自动调整训练超参数。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def build_launch(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-processes", type=int, default=4, help="进程数，每个进程使用一张卡")
    parser.add_argument("--gpus", help="显式选择 GPU，例如 0,1,2,3；省略时保留 CUDA_VISIBLE_DEVICES")
    parser.add_argument("--log-dir", type=Path, help="torchrun 各 rank 的 stdout/stderr 日志目录")
    parser.add_argument("--dry-run", action="store_true", help="只显示命令，不初始化 GPU 或启动训练")
    parser.add_argument("stage", choices=("sft", "grpo"))
    parser.add_argument("training_args", nargs=argparse.REMAINDER, help="原样传给 train sft/grpo 的参数")
    args = parser.parse_args(argv)
    if args.num_processes < 2:
        parser.error("DDP --num-processes 必须至少为 2；单卡请直接使用 agentic-arxiv train")
    if any(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        parser.error("不能在已有分布式 worker 内嵌套启动 torchrun")
    environment = os.environ.copy()
    visible = args.gpus if args.gpus is not None else environment.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [item.strip() for item in visible.split(",")]
        if not all(devices) or len(set(devices)) != len(devices) or "-1" in devices:
            parser.error("GPU 列表必须非空、不重复且不能包含 -1")
        if len(devices) < args.num_processes:
            parser.error("可见 GPU 数少于 --num-processes；不会自动占用未指定的卡")
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nnodes=1", f"--nproc-per-node={args.num_processes}"]
    if args.log_dir is not None:
        command += ["--log-dir", str(args.log_dir.expanduser().resolve()), "--tee=3"]
    forwarded = args.training_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    command += ["--module", "agentic_arxiv", "train", args.stage, *forwarded]
    return args, command, environment


def main(argv=None):
    args, command, environment = build_launch(argv)
    if args.dry_run:
        print(json.dumps({"command": command,
                          "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES")},
                         ensure_ascii=False, indent=2))
        return 0
    return subprocess.call(command, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
