"""规范路径和最小权限挂载；容器内保留输入的绝对路径。"""
from pathlib import Path


def overlaps(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    return left == right or left in right.parents or right in left.parents


def canonical(path, *, exists=True):
    value = Path(path).expanduser()
    if not value.is_absolute():
        raise ValueError(f"必须使用绝对路径: {path}")
    if ".." in value.parts:
        raise ValueError(f"路径不能含上级跳转: {path}")
    result = value.resolve(strict=exists)
    # 别名不应掩盖实际输入身份或挂载目的地。
    if value != result:
        raise ValueError(f"拒绝符号链接或路径别名: {path} -> {result}")
    if any(token in str(result) for token in (",", "\n", "\r")):
        raise ValueError("路径包含不支持的挂载分隔符")
    return result


def input_path(path, allowed_roots, *, managed_root=None):
    result = canonical(path)
    roots = [canonical(root) for root in allowed_roots]
    managed = False
    if managed_root is not None:
        root = canonical(managed_root, exists=False)
        try:
            relative = result.relative_to(root)
            parts = relative.parts
            managed = len(parts) >= 6 and parts[0] == "runs" and parts[2] == "attempts" and parts[4] == "output" and parts[5] == "final"
        except ValueError:
            pass
    if not managed and not any(result != root and root in result.parents for root in roots):
        raise ValueError(f"输入必须位于允许根目录之内，不能挂载整个根目录: {result}")
    if result in {Path.home(), Path("/"), Path("/data"), Path("/etc"), Path("/var"), Path("/tmp")}:
        raise ValueError("输入挂载过宽")
    if ".ssh" in result.parts or "docker.sock" in result.parts:
        raise ValueError("受保护路径不能挂载")
    return result


def validate_layout(run_root, inputs, *, cache_root=None, attempt_dir=None):
    root = canonical(run_root, exists=False)
    protected = [canonical(path) for path in inputs]
    for path in protected:
        managed_history = root in path.parents and len(path.relative_to(root).parts) >= 6 and path.relative_to(root).parts[0] == "runs" and path.relative_to(root).parts[2] == "attempts" and path.relative_to(root).parts[4] == "output" and path.relative_to(root).parts[5] == "final"
        if overlaps(root, path) and not managed_history:
            raise ValueError(f"输入与管理输出存在祖先/后代重叠: {path}")
    if cache_root:
        cache = canonical(cache_root, exists=False)
        if any(overlaps(cache, path) for path in protected) or overlaps(cache, root):
            raise ValueError("缓存与输入或管理记录重叠")
        if cache in {Path.home(), Path("/"), Path("/data"), Path("/tmp")}:
            raise ValueError("可写缓存目录过宽")
    if attempt_dir:
        attempt = canonical(attempt_dir)
        if not root in attempt.parents:
            raise ValueError("尝试目录越界")
        if any(overlaps(attempt / child, source) for child in ("output", "control", "logs") for source in protected):
            raise ValueError("当前尝试可写目录与只读历史输入重叠")
    return root


def minimal_inputs(paths):
    result = []
    for path in sorted(set(Path(path) for path in paths), key=lambda item: len(item.parts)):
        if not any(parent == path or parent in path.parents for parent in result):
            result.append(path)
    return result


def mounts(inputs, attempt_dir, cache_root=None, extra_readonly=()):
    attempt = Path(attempt_dir)
    result = [{"source": str(path), "target": str(path), "readonly": True}
              for path in minimal_inputs([*inputs, *extra_readonly])]
    for child in ("output", "control", "logs"):
        path = attempt / child
        result.append({"source": str(path), "target": str(path), "readonly": False})
    if cache_root:
        result.append({"source": str(cache_root), "target": str(cache_root), "readonly": False})
    return result
