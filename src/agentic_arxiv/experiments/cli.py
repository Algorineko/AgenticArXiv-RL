"""独立管理命令结果边界，控制端不初始化训练环境。"""

from __future__ import annotations

import json
from pathlib import Path

from .config import _reference, load_profile, load_recipe, normalize_profile, parse_overrides, resolve
from .paths import canonical


EXIT_OK = 0
EXIT_INVALID = 2
EXIT_CONFLICT = 3
EXIT_UNAVAILABLE = 4
EXIT_UNCERTAIN = 5


def _result(operation, *, data=None, error=None, code=EXIT_OK):
    return {"schema_version": 1, "operation": operation, "ok": code == EXIT_OK,
            "exit_code": code, "data": data, "error": error,
            "completion_scope": "management_operation_only"}


def execute(args):
    operation = args._operation.removeprefix("experiment.")
    try:
        from .service import ExperimentService

        def path(value):
            item = Path(value).expanduser()
            return (item if item.is_absolute() else args.workspace / item).resolve()

        profile = normalize_profile(load_profile(path(args.profile)), args.workspace)
        # 即便只读命令也规范化路径，但不要求配方、不探测训练依赖。
        profile_root = Path(profile["workspace"])

        def resource(value):
            value = Path(value).expanduser()
            return str(canonical(value if value.is_absolute() else profile_root / value, exists=False))
        resolved = None
        if operation in ("check", "start"):
            recipe = load_recipe(path(args.recipe))
            overrides = parse_overrides(args.set_values, seed=args.seed, gpus=args.gpus, name=args.name)
            resolved = resolve(profile, recipe, overrides, workspace=args.workspace)
        for key in ("metadata", "evaluation_config"):
            if getattr(args, key, None):
                setattr(args, key, str(path(getattr(args, key))))
        for key in ("model", "checkpoint"):
            if getattr(args, key, None):
                setattr(args, key, resource(getattr(args, key)))
        if operation == "evaluate":
            raw = args.reference
            if raw.startswith("registration:"):
                args.reference = {"kind": "registration", "registration_id": raw.removeprefix("registration:")}
            elif raw.startswith("run:"):
                fields = raw.split(":")
                if len(fields) != 4:
                    raise ValueError("运行引用必须为 run:RUN_ID:ATTEMPT_ID:final")
                args.reference = dict(zip(("kind", "run_id", "attempt_id", "artifact"), fields))
            elif raw.startswith("{"):
                args.reference = json.loads(raw)
            _reference(args.reference, "evaluation.reference")
            if isinstance(args.reference, str):
                args.reference = resource(args.reference)
            elif args.reference["kind"] == "path":
                args.reference["path"] = resource(args.reference["path"])
        data = ExperimentService(profile, workspace=profile_root).handle(operation, args, resolved=resolved)
        # 服务可保留已知历史证据并报告当前执行状态未知。
        code = int(data.pop("management_exit_code", EXIT_OK)) if isinstance(data, dict) else EXIT_OK
        result = _result(operation, data=data, code=code)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        code = getattr(exc, "exit_code", None)
        if code not in (EXIT_INVALID, EXIT_CONFLICT, EXIT_UNAVAILABLE, EXIT_UNCERTAIN):
            code = EXIT_UNAVAILABLE if isinstance(exc, (OSError, ImportError)) else EXIT_INVALID
        result = _result(operation, code=code, error={"type": type(exc).__name__, "message": str(exc)})
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    elif result["ok"]:
        print(json.dumps(result["data"], ensure_ascii=False, indent=2, allow_nan=False))
        if operation in ("start", "resume", "evaluate"):
            print("管理操作已完成；后台启动不代表训练或验证最终成功。")
    else:
        print(f"管理操作失败（退出码 {result['exit_code']}）：{result['error']['message']}" if result["error"] else
              f"管理操作结果尚未确认（退出码 {result['exit_code']}）；保留历史证据与资源声明。")
    return result["exit_code"]
