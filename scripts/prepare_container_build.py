"""将白名单文件复制到全新构建目录，记录实际代码身份；不调用 Docker。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT_FILES = (
    "pyproject.toml", "README.md", "LICENSE", ".dockerignore",
    "containers/Dockerfile", "containers/base-image.json", "containers/lock-resolution.json",
    "containers/training-baseline.in", "containers/requirements-linux-py310-cu126.lock",
)


def source_files(root):
    result = [root / name for name in ROOT_FILES]
    result += [path for path in (root / "src/agentic_arxiv").rglob("*")
               if path.is_file() and path.suffix in {".py", ".json"}
               and "__pycache__" not in path.parts]
    for path in result:
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"构建输入不能使用符号链接: {path}")
    return sorted(result)


def prepare(root, destination):
    root = Path(root).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.exists():
        raise ValueError("构建目录必须不存在，不能覆盖旧证据")
    selected = source_files(root)
    content = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in selected}
    identity = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True)
    manifest = {"schema_version": 1, "git_commit": commit, "dirty": bool(status.strip()),
                "tree_sha256": identity, "files": content,
                "dependency_lock_sha256": content["containers/requirements-linux-py310-cu126.lock"],
                "recovery_protocol": 1}
    destination.mkdir(parents=True)
    for path in selected:
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != content[str(path.relative_to(root))]:
            raise ValueError("复制期间输入发生变化，丢弃本次构建目录后重新准备")
    (destination / "build-identity.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="全新构建上下文目录")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.destination), ensure_ascii=False, indent=2))
