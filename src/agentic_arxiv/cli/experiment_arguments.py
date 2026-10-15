"""实验管理参数声明；帮助不访问文件、容器或训练依赖。"""

from __future__ import annotations

import argparse


def register_commands(parser, add_command):
    commands = parser.add_subparsers(dest="experiment_command", required=True)
    descriptions = {
        "check": "只读检查配方、输入及运行环境，不建立训练运行",
        "register": "只读登记已有最终模型，不赋予旧 checkpoint 恢复资格",
        "start": "后台启动一个明确作业，不构建、拉取镜像或执行前置阶段",
        "list": "读取已有运行列表", "status": "核对运行状态与历史证据",
        "logs": "读取当前尝试主进程或指定 rank 日志",
        "stop": "请求安全停止；只有显式 --force 才强制中断",
        "resume": "从明确完整 checkpoint 建立新尝试，继续原总预算",
        "evaluate": "手动独立评测或只重试阶段验证，不重新训练",
    }
    for name, description in descriptions.items():
        leaf = add_command(commands, name, help=description, description=description)
        leaf.set_defaults(_operation=f"experiment.{name}")
        leaf.add_argument("--profile", required=True, help="服务器 profile JSON，相对显式工作区")
        leaf.add_argument("--json", action="store_true", help="输出版本化 JSON 结果")
        if name in ("check", "start"):
            leaf.add_argument("recipe", help="一个实验配方 JSON")
            leaf.add_argument("--gpus", help="显式 GPU 索引或 UUID，逗号分隔且保留 rank 顺序")
            leaf.add_argument("--seed", type=int, default=None, help="显式覆盖训练种子并记录差异")
            leaf.add_argument("--name", default=None, help="独立运行名称")
            leaf.add_argument("--set", dest="set_values", action="append", default=[],
                              metavar="GROUP.KEY=JSON", help="覆盖 defaults/trainer_parameters 的现有训练字段")
        elif name == "register":
            leaf.add_argument("group", choices=("M1", "M2", "M3", "M4", "C1"))
            leaf.add_argument("--model", required=True, help="已有本地最终模型目录")
            leaf.add_argument("--artifact-id", help="登记标识，默认分配独立标识")
            leaf.add_argument("--metadata", help="可选来源与成本 JSON 文件")
        elif name in ("status", "logs", "stop", "resume"):
            leaf.add_argument("run_id", help="明确受管运行标识")
            if name == "logs":
                leaf.add_argument("--rank", type=int, default=None)
                leaf.add_argument("--tail", type=int, default=200)
            if name == "stop":
                leaf.add_argument("--force", action="store_true", help="核验身份后强制终止，不保证保存新进度")
            if name == "resume":
                leaf.add_argument("--checkpoint", required=True, help="明确完整 checkpoint 路径，禁止 latest")
                leaf.add_argument("--gpus", help="必须与原运行设备及 rank 顺序相同")
        elif name == "evaluate":
            leaf.add_argument("reference", help="本地模型路径、registration:ID、run:RUN:ATTEMPT:final 或 JSON 引用对象")
            leaf.add_argument("--kind", choices=("benchmark", "stage"), default="benchmark")
            leaf.add_argument("--evaluation-config", required=True, help="固定评测配置 JSON")
            leaf.add_argument("--gpus", required=True, help="显式 GPU 索引或 UUID，逗号分隔")
            leaf.add_argument("--diagnostic", action="store_true", help="诊断评测不晋升模型输入资格")
