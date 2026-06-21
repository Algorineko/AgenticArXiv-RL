---
name: arxiv-tools
description: |
  arXiv 论文搜索、下载、翻译、缓存状态查询工具集。
  提供 4 个 CLI 命令，用于管理计算机科学领域学术论文。
  触发关键词：论文搜索、下载PDF、翻译论文、缓存状态
---

# arXiv 论文管理 CLI 工具

## CLI 工具位置

```
skill_cli/tool_cli.py
```

## 重要：session_id 由系统自动管理，命令中不需要指定 session_id 参数。

## 使用方法

### 1. 搜索论文 (search_papers)

搜索最近提交的计算机科学领域论文。

```bash
python skill_cli/tool_cli.py search_papers --max_results=20 --aspect=AI --days=7
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--max_results` | int | 50 | 最大返回结果数 (1-100) |
| `--aspect` | string | "*" | CS 子领域代码，可选: AI, CL, CV, LG, RO, SE 等, * 表示全部 |
| `--days` | int | 7 | 查询最近多少天的论文 (1-30) |

### 2. 下载 PDF (download_pdf)

下载指定论文的 PDF 文件到本地。

```bash
python skill_cli/tool_cli.py download_pdf --ref=1
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ref` | int/string/None | None | 论文引用：1-based 序号、arxiv ID 或标题子串；None 表示最近操作的论文 |
| `--force` | bool | False | 是否强制重新下载 |

### 3. 翻译 PDF (translate_pdf)

翻译论文 PDF 为中文。

```bash
python skill_cli/tool_cli.py translate_pdf --ref=1 --service=bing --threads=4
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ref` | int/string/None | None | 论文引用 |
| `--force` | bool | False | 是否强制重新翻译 |
| `--service` | string | "bing" | 翻译服务：bing, deepl, google |
| `--threads` | int | 4 | 翻译线程数 (1-32) |
| `--keep_dual` | bool | False | 是否保留双语 PDF |

### 4. 缓存状态 (cache_status)

查询论文的本地缓存状态（是否已下载、是否已翻译）。

```bash
python skill_cli/tool_cli.py cache_status --ref=1
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ref` | int/string/None | None | 论文引用 |
| `--paper_id` | string | None | 直接指定 paper_id |

## 输出格式

所有命令输出 JSON 到 stdout。

**搜索结果示例**:
```json
[{"id": "2401.12345", "title": "论文标题", "authors": ["作者"], "pdf_url": "https://..."}]
```

**下载/翻译结果示例**:
```json
{"paper_id": "2401.12345", "status": "READY", "output_pdf_path": "/path/to/file.pdf"}
```

**缓存状态示例**:
```json
{"paper_id": "2401.12345", "pdf_ready": true, "translated_ready": false}
```

## 完整示例

### 示例1：搜索 AI 领域最近论文

```bash
python skill_cli/tool_cli.py search_papers --max_results=10 --aspect=AI --days=3
```

### 示例2：下载搜索结果中第2篇论文

```bash
python skill_cli/tool_cli.py download_pdf --ref=2
```

### 示例3：翻译最近操作的论文

```bash
python skill_cli/tool_cli.py translate_pdf --service=bing --threads=4
```

### 示例4：检查论文缓存

```bash
python skill_cli/tool_cli.py cache_status --paper_id=2401.12345
```

## 注意事项

- **不要在命令中指定 --session_id，系统会自动注入正确的 session_id**
- 必须先执行 search_papers 获取论文列表，才能使用 ref 序号下载/翻译
- translate_pdf 是同步执行，可能耗时较长；在 Agent 模式下由 Agent 负责异步调度
- ref=None 时自动使用最近操作的论文（需先执行搜索/下载）
- 所有输出为 JSON 格式，方便程序化解析
