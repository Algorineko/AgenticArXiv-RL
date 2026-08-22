"""GitHub/Gitee repository search and safe source archive download tools.

Downloads are ZIP archives only.  They are never executed or automatically
extracted, which keeps both interactive use and RL rollouts predictable.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote, urlparse

import requests

from config import settings
from tools.tool_registry import registry

_GITHUB_API = "https://api.github.com"
_GITEE_API = "https://gitee.com/api/v5"
_SAFE_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
_SEARCH_MEMORY: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
_MEMORY_LOCK = threading.RLock()


def remember_repository_results(
    session_id: str, platform: str, repositories: List[Dict[str, Any]]
) -> None:
    """Remember search results so a later download can use a 1-based ref."""
    with _MEMORY_LOCK:
        _SEARCH_MEMORY.setdefault(session_id or "default", {})[platform] = list(repositories)


def clear_repository_memory(session_id: Optional[str] = None) -> None:
    """Test/helper API; clear one session or all repository search memory."""
    with _MEMORY_LOCK:
        if session_id is None:
            _SEARCH_MEMORY.clear()
        else:
            _SEARCH_MEMORY.pop(session_id, None)


def get_repository_results(session_id: str, platform: str) -> List[Dict[str, Any]]:
    with _MEMORY_LOCK:
        return list(_SEARCH_MEMORY.get(session_id or "default", {}).get(platform, []))


def _checked_limit(max_results: int) -> int:
    value = int(max_results)
    if not 1 <= value <= 20:
        raise ValueError("max_results must be between 1 and 20")
    return value


def _raise_for_api(response: requests.Response, platform: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = ""
        try:
            payload = response.json()
            detail = payload.get("message") or payload.get("error") or ""
        except Exception:
            detail = response.text[:200]
        hint = "；可配置访问令牌以提高限额或访问私有资源" if response.status_code in (401, 403, 429) else ""
        raise RuntimeError(f"{platform} API 请求失败 ({response.status_code}): {detail}{hint}") from exc


def _github_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "AgenticArxiv-RL",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _gitee_params(params: Dict[str, Any]) -> Dict[str, Any]:
    token = os.getenv("GITEE_TOKEN", "").strip()
    if token:
        params["access_token"] = token
    return params


def _normalise_github(item: Dict[str, Any]) -> Dict[str, Any]:
    owner = item.get("owner") or {}
    return {
        "platform": "github",
        "id": item.get("id"),
        "full_name": item.get("full_name"),
        "name": item.get("name"),
        "owner": owner.get("login"),
        "description": item.get("description") or "",
        "html_url": item.get("html_url"),
        "clone_url": item.get("clone_url"),
        "default_branch": item.get("default_branch"),
        "language": item.get("language"),
        "stars": item.get("stargazers_count", 0),
        "forks": item.get("forks_count", 0),
        "updated_at": item.get("updated_at"),
        "archived": bool(item.get("archived", False)),
    }


def _normalise_gitee(item: Dict[str, Any]) -> Dict[str, Any]:
    owner = item.get("owner") or {}
    full_name = item.get("full_name") or item.get("path_with_namespace")
    return {
        "platform": "gitee",
        "id": item.get("id"),
        "full_name": full_name,
        "name": item.get("name") or item.get("path"),
        "owner": owner.get("login") or owner.get("username") or (full_name or "/").split("/")[0],
        "description": item.get("description") or "",
        "html_url": item.get("html_url") or item.get("url"),
        "clone_url": item.get("clone_url") or item.get("http_url_to_repo"),
        "default_branch": item.get("default_branch"),
        "language": item.get("language"),
        "stars": item.get("stargazers_count", item.get("stars_count", 0)),
        "forks": item.get("forks_count", 0),
        "updated_at": item.get("updated_at") or item.get("last_push_at"),
        "archived": bool(item.get("archived", False)),
    }


def search_github_repositories(
    query: str,
    language: Optional[str] = None,
    sort: str = "stars",
    order: str = "desc",
    max_results: int = 5,
    session_id: str = "default",
) -> List[Dict[str, Any]]:
    query = str(query).strip()
    if not query:
        raise ValueError("query cannot be empty")
    if sort not in ("stars", "forks", "updated") or order not in ("asc", "desc"):
        raise ValueError("invalid sort or order")
    if language:
        query = f"{query} language:{language.strip()}"
    response = requests.get(
        f"{_GITHUB_API}/search/repositories",
        params={"q": query, "sort": sort, "order": order, "per_page": _checked_limit(max_results), "page": 1},
        headers=_github_headers(),
        timeout=(5, 20),
    )
    _raise_for_api(response, "GitHub")
    results = [_normalise_github(item) for item in response.json().get("items", [])]
    remember_repository_results(session_id, "github", results)
    return results


def search_gitee_repositories(
    query: str,
    language: Optional[str] = None,
    sort: str = "stars",
    order: str = "desc",
    max_results: int = 5,
    session_id: str = "default",
) -> List[Dict[str, Any]]:
    query = str(query).strip()
    if not query:
        raise ValueError("query cannot be empty")
    if sort not in ("stars", "forks", "updated") or order not in ("asc", "desc"):
        raise ValueError("invalid sort or order")
    sort_map = {"stars": "stars_count", "forks": "forks_count", "updated": "last_push_at"}
    params: Dict[str, Any] = {
        "q": query,
        "sort": sort_map.get(sort, sort),
        "order": order,
        "per_page": _checked_limit(max_results),
        "page": 1,
    }
    if language:
        params["language"] = language.strip()
    response = requests.get(
        f"{_GITEE_API}/search/repositories",
        params=_gitee_params(params),
        headers={"Accept": "application/json", "User-Agent": "AgenticArxiv-RL"},
        timeout=(5, 20),
    )
    _raise_for_api(response, "Gitee")
    payload = response.json()
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    results = [_normalise_gitee(item) for item in items]
    remember_repository_results(session_id, "gitee", results)
    return results


def _split_full_name(value: str, platform: str) -> tuple[str, str]:
    value = value.strip()
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        expected = "github.com" if platform == "github" else "gitee.com"
        if parsed.hostname not in (expected, f"www.{expected}"):
            raise ValueError(f"repository URL must belong to {expected}")
        value = parsed.path.strip("/")
    if value.endswith(".git"):
        value = value[:-4]
    parts = value.split("/")
    if (
        len(parts) != 2
        or any(part in (".", "..") for part in parts)
        or not all(_SAFE_PART.fullmatch(part) for part in parts)
    ):
        raise ValueError("repository must be owner/name, a repository URL, or a search-result index")
    return parts[0], parts[1]


def _resolve_repository(
    platform: str, session_id: str, repository: Union[int, str, None]
) -> Dict[str, Any]:
    results = get_repository_results(session_id, platform)
    if repository is None:
        if not results:
            raise ValueError(f"no previous {platform} search results; search first or provide owner/name")
        return results[0]
    if isinstance(repository, int) and not isinstance(repository, bool):
        if repository < 1 or repository > len(results):
            raise ValueError(f"repository index out of range: 1..{len(results)}")
        return results[repository - 1]
    owner, name = _split_full_name(str(repository), platform)
    return {"platform": platform, "full_name": f"{owner}/{name}", "owner": owner, "name": name}


def _default_branch(platform: str, owner: str, name: str) -> str:
    if platform == "github":
        response = requests.get(
            f"{_GITHUB_API}/repos/{quote(owner)}/{quote(name)}",
            headers=_github_headers(), timeout=(5, 20),
        )
    else:
        response = requests.get(
            f"{_GITEE_API}/repos/{quote(owner)}/{quote(name)}",
            params=_gitee_params({}),
            headers={"Accept": "application/json", "User-Agent": "AgenticArxiv-RL"}, timeout=(5, 20),
        )
    _raise_for_api(response, platform.title())
    branch = response.json().get("default_branch")
    if not branch:
        raise RuntimeError(f"{platform} repository metadata has no default_branch")
    return str(branch)


def _archive_url(platform: str, owner: str, name: str, ref: str) -> str:
    if platform == "github":
        return f"{_GITHUB_API}/repos/{quote(owner)}/{quote(name)}/zipball/{quote(ref, safe='')}"
    return f"{_GITEE_API}/repos/{quote(owner)}/{quote(name)}/zipball"


def _download_repository(
    platform: str,
    repository: Union[int, str, None] = None,
    ref: Optional[str] = None,
    force: bool = False,
    session_id: str = "default",
) -> Dict[str, Any]:
    item = _resolve_repository(platform, session_id, repository)
    owner, name = _split_full_name(item["full_name"], platform)
    branch = (ref or item.get("default_branch") or "").strip()
    if not branch:
        branch = _default_branch(platform, owner, name)
    if len(branch) > 200 or any(ch in branch for ch in ("\0", "\r", "\n")):
        raise ValueError("invalid ref")

    output_dir = Path(settings.repository_download_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_ref = re.sub(r"[^A-Za-z0-9_.-]+", "_", branch).strip("._") or "default"
    target = (output_dir / f"{platform}__{owner}__{name}__{safe_ref}.zip").resolve()
    if output_dir not in target.parents:
        raise ValueError("unsafe output path")
    if target.exists() and target.stat().st_size > 0 and not force:
        return _archive_result(platform, owner, name, branch, target, True)

    headers = _github_headers() if platform == "github" else {"Accept": "application/zip", "User-Agent": "AgenticArxiv-RL"}
    response = requests.get(
        _archive_url(platform, owner, name, branch),
        headers=headers,
        params=_gitee_params({"ref": branch}) if platform == "gitee" else None,
        stream=True, allow_redirects=True, timeout=(5, 60),
    )
    _raise_for_api(response, platform.title())
    max_bytes = max(1, int(settings.repository_max_download_mb)) * 1024 * 1024
    declared = int(response.headers.get("Content-Length") or 0)
    if declared > max_bytes:
        response.close()
        raise ValueError(f"archive exceeds {settings.repository_max_download_mb} MB limit")
    partial = target.with_suffix(target.suffix + ".part")
    total = 0
    try:
        with open(partial, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"archive exceeds {settings.repository_max_download_mb} MB limit")
                file_obj.write(chunk)
        os.replace(partial, target)
    finally:
        response.close()
        if partial.exists():
            partial.unlink()
    return _archive_result(platform, owner, name, branch, target, False)


def _archive_result(platform: str, owner: str, name: str, ref: str, path: Path, existed: bool) -> Dict[str, Any]:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "platform": platform,
        "repository": f"{owner}/{name}",
        "ref": ref,
        "local_path": str(path),
        "status": "READY",
        "existed": existed,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "extracted": False,
    }


def download_github_repository(repository=None, ref=None, force=False, session_id="default"):
    return _download_repository("github", repository, ref, force, session_id)


def download_gitee_repository(repository=None, ref=None, force=False, session_id="default"):
    return _download_repository("gitee", repository, ref, force, session_id)


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "仓库关键词或主题", "minLength": 1},
        "language": {"type": "string", "description": "可选编程语言筛选"},
        "sort": {"type": "string", "description": "排序字段", "enum": ["stars", "forks", "updated"], "default": "stars"},
        "order": {"type": "string", "description": "排序方向", "enum": ["desc", "asc"], "default": "desc"},
        "max_results": {"type": "integer", "description": "最大结果数", "minimum": 1, "maximum": 20, "default": 5},
        "session_id": {"type": "string", "description": "内部会话标识", "default": "default", "x-internal": True},
    },
    "required": ["query"],
}

DOWNLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "repository": {"description": "搜索结果序号（从1开始）、owner/name 或仓库URL", "anyOf": [{"type": "integer", "minimum": 1}, {"type": "string"}]},
        "ref": {"type": "string", "description": "可选分支、Tag或提交；默认使用仓库默认分支"},
        "force": {"type": "boolean", "description": "是否覆盖已下载归档", "default": False},
        "session_id": {"type": "string", "description": "内部会话标识", "default": "default", "x-internal": True},
    },
    "required": [],
}

registry.register_tool("search_github_repositories", "在 GitHub 搜索代码仓库；结果可供下载工具按序号引用", SEARCH_SCHEMA, search_github_repositories)
registry.register_tool("search_gitee_repositories", "在 Gitee 搜索代码仓库；结果可供下载工具按序号引用", SEARCH_SCHEMA, search_gitee_repositories)
registry.register_tool("download_github_repository", "下载 GitHub 仓库源码 ZIP；不执行、不自动解压", DOWNLOAD_SCHEMA, download_github_repository)
registry.register_tool("download_gitee_repository", "下载 Gitee 仓库源码 ZIP；不执行、不自动解压", DOWNLOAD_SCHEMA, download_gitee_repository)
