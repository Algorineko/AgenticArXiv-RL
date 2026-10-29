"""Small, new real-metadata replay screen; no historical snapshot dependency.

This measures tool/state control, not PDF comprehension or translation quality.
Different splits use disjoint paper IDs. Development shares primitive families
with train; final_test reserves new compositions, not new primitive tools.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from benchmark.task_spec import Step, TaskSpec

ASPECTS = ("AI", "LG", "CL", "CV")
ROLES = {"train": "AI", "search_dev": "LG", "promotion_val": "CL", "final_test": "CV"}
PRIMITIVES = ("search", "download", "translate", "cache", "search_download", "download_translate")
COMPOSITIONS = ("search_download_translate", "search_download_cache", "double_download",
                "download_cache_translate", "active_cache_translate", "translate_cache")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_atom(raw):
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(raw)
    result = []
    for entry in root.findall("a:entry", ns):
        uri = entry.findtext("a:id", "", ns)
        if "/abs/" not in uri:
            raise ValueError("arXiv feed contains an error/non-paper entry")
        cats = [x.attrib["term"] for x in entry.findall("a:category", ns)]
        primary = entry.find("x:primary_category", ns)
        links = entry.findall("a:link", ns)
        pdf = next((x.attrib["href"] for x in links if x.attrib.get("title") == "pdf"),
                   uri.replace("/abs/", "/pdf/"))
        result.append({"id": uri.rsplit("/abs/", 1)[1],
                       "title": " ".join(entry.findtext("a:title", "", ns).split()),
                       "summary": " ".join(entry.findtext("a:summary", "", ns).split()),
                       "authors": [x.findtext("a:name", "", ns) for x in entry.findall("a:author", ns)],
                       "published": entry.findtext("a:published", "", ns),
                       "updated": entry.findtext("a:updated", "", ns), "pdf_url": pdf,
                       "primary_category": primary.attrib["term"] if primary is not None else cats[0],
                       "categories": cats})
    return result


def base_id(value):
    return value.rsplit("v", 1)[0] if value.rsplit("v", 1)[-1].isdigit() else value


def disjoint_pools(pools, minimum=6):
    seen = set()
    result = {}
    for aspect in ASPECTS:
        papers = []
        for paper in pools[aspect]:
            key = base_id(paper["id"])
            if key not in seen:
                papers.append(paper)
                seen.add(key)
        if len(papers) < minimum:
            raise ValueError(f"Need at least {minimum} paper-disjoint records in {aspect}, got {len(papers)}")
        result[aspect] = papers
    return result


def _task(role, aspect, family, index):
    limit, ref = index + 3, index + 1
    search = Step("get_recently_submitted_cs_papers", {"aspect": aspect, "days": 7, "max_results": limit})
    download = Step("download_arxiv_pdf", {"ref": ref})
    translate = Step("translate_arxiv_pdf", {"ref": ref})
    cache = Step("get_paper_cache_status", {"ref": ref})
    query = f"检索最近7天cs.{aspect}的论文，最多{limit}篇"
    setup = (search,)
    if family == "search":
        text, steps, setup = query + "。", (search,), ()
    elif family == "download":
        text, steps = f"下载当前列表中第{ref}篇论文的PDF。", (download,)
    elif family == "translate":
        text, steps, setup = f"翻译当前列表中第{ref}篇论文。", (translate,), (search, download)
    elif family == "cache":
        text, steps, setup = f"查看当前列表中第{ref}篇论文的缓存状态。", (cache,), (search, download)
    elif family == "search_download":
        text, steps, setup = query + f"，然后下载第{ref}篇的PDF。", (search, download), ()
    elif family == "download_translate":
        text, steps = f"先下载当前列表中第{ref}篇论文的PDF，再翻译这篇论文。", (download, translate)
    elif family == "search_download_translate":
        text, steps, setup = query + f"，下载第{ref}篇的PDF，然后翻译它。", (search, download, translate), ()
    elif family == "search_download_cache":
        text, steps, setup = query + f"，下载第{ref}篇的PDF，再查看这篇的缓存状态。", (search, download, cache), ()
    elif family == "double_download":
        other = ref + 1
        text, steps = f"按顺序下载当前列表中第{ref}篇和第{other}篇论文的PDF。", (download, Step("download_arxiv_pdf", {"ref": other}))
    elif family == "download_cache_translate":
        text, steps = f"下载当前列表中第{ref}篇论文，查看其缓存状态，最后翻译它。", (download, cache, translate)
    elif family == "active_cache_translate":
        text = "先查看最近操作的那篇论文的缓存状态，再翻译它；两次都使用ref=null。"
        steps, setup = (Step("get_paper_cache_status", {"ref": None}), Step("translate_arxiv_pdf", {"ref": None})), (search, download)
    elif family == "translate_cache":
        text, steps, setup = f"翻译当前列表中第{ref}篇论文，然后查看它的缓存状态。", (translate, cache), (search, download)
    else:
        raise ValueError(family)
    task = TaskSpec(id=f"rescue_v1_{role}_{family}_{index}", task=text, steps=steps,
                    setup=setup, category="rescue_" + family, template=family,
                    requires_offline=True, max_iterations=6).to_task()
    task.update(rescue_split=role, rescue_family=family, catalog_aspect=aspect)
    return task


def build_catalog(pools):
    pools = disjoint_pools(pools)
    snapshot = {"get_recently_submitted_cs_papers": {}}
    for aspect, papers in pools.items():
        args = {"aspect": aspect, "days": 7, "max_results": len(papers)}
        snapshot["get_recently_submitted_cs_papers"][json.dumps(args, sort_keys=True, ensure_ascii=False)] = {
            "args": args, "result": papers}
    tasks, splits = [], {}
    for role, aspect in ROLES.items():
        families = COMPOSITIONS if role == "final_test" else PRIMITIVES
        n = 4 if role in ("search_dev", "final_test") else 2
        current = [_task(role, aspect, family, index) for family in families for index in range(n)]
        tasks.extend(current)
        splits[role] = [x["id"] for x in current]
    return snapshot, tasks, {"version": "rescue_metadata_v1", "split": splits,
                            "paper_ids_by_split": {role: [p["id"] for p in pools[aspect]]
                                                   for role, aspect in ROLES.items()}}


def verify_catalog(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("kind") != "real_metadata_replay_screen_v1":
        raise ValueError("Unexpected catalog kind")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != expected:
            raise ValueError(f"Catalog hash/path mismatch: {name}")
    tasks = json.loads((directory / "tasks.json").read_text())
    splits = json.loads((directory / "splits.json").read_text())
    by_id = {t["id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Duplicate task IDs")
    seen_tasks, seen_papers = set(), set()
    for role in ROLES:
        ids = splits["split"][role]
        papers = [base_id(x) for x in splits["paper_ids_by_split"][role]]
        if not ids or len(set(ids)) != len(ids) or seen_tasks.intersection(ids):
            raise ValueError(f"Overlapping/empty task split: {role}")
        if not papers or len(set(papers)) != len(papers) or seen_papers.intersection(papers):
            raise ValueError(f"Overlapping/empty paper split: {role}")
        for task_id in ids:
            if by_id[task_id]["rescue_split"] != role:
                raise ValueError("Task split label mismatch")
        seen_tasks.update(ids)
        seen_papers.update(papers)
    if seen_tasks != set(by_id):
        raise ValueError("Unassigned tasks")
    return manifest, tasks, splits
