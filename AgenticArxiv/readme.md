# AgenticArxiv 后端 — Agent 核心模块文档

本文档深入讲解后端架构，重点是**三种 Agent 实现方案**及其核心组件。

## 快速导航

- 🏗️ **Agent 三层架构**：BaseAgent (通用) → 三种实现 → 工具执行
- 🔧 **工具系统**：tool_registry（进程内）+ 工具定义
- 📊 **数据持久化**：ORM + Store + 日志记录
- 🚀 **实时推送**：SSE + Event Bus + 异步翻译

---

## Agent 系统设计

### 核心概念：ReAct 循环

所有三种 Agent 都基于 **ReAct** (Reasoning + Acting) 框架：

```
┌──────────────────┐
│ Thought: 分析任务 │
└────────┬─────────┘
         │
┌────────▼──────────┐
│ Action: 调用工具  │  ← 三种方案差异在这里
└────────┬──────────┘
         │
┌────────▼──────────┐
│ Observation: 结果 │
└────────┬──────────┘
         │
      重复，直到
      Action = FINISH
```

### 三层架构

```python
# 第一层：通用执行循环（BaseAgent）
class BaseAgent(ABC):
    def run(task, agent_model, session_id):
        for iteration in range(max_iterations):
            # 1. LLM 调用
            response = self.llm_client.chat_completions(...)
            
            # 2. 子类实现的解析
            thought, action_dict = self.parse_response(response)
            
            # 3. 统一的副作用处理（session、翻译异步、日志）
            observation = self._execute_with_side_effects(action_dict)
            
            # 4. 记录步骤 + SSE 推送
            self._log_step(...)

# 第二层：三种实现（各自的 parse_response + invoke_tool）
class ReActAgent(BaseAgent):
    def parse_response(raw_response) -> (thought, action_dict):
        # 正则提取 Thought/Action/Observation
        return thought, {"name": tool_name, "args": {...}}
    
    def invoke_tool(tool_name, args) -> result:
        return registry.execute_tool(tool_name, args)

class MCPAgent(BaseAgent):
    def invoke_tool(tool_name, args) -> result:
        return await self._session.call_tool(tool_name, args)

class SkillAgent(BaseAgent):
    def parse_response(raw_response) -> (thought, action_dict):
        # 提取 Command: bash\n...\n
        return thought, {"name": tool_name, "args": {...}}
    
    def invoke_tool(tool_name, args) -> result:
        return subprocess.run(["python", "tool_cli.py", ...])

# 第三层：工具执行（registry）
registry.execute_tool(tool_name, args) -> result
```

---

## A. ReActAgent — 正则 + 同步执行

**文件**：`agents/agent_engine.py`

### 工作流程

#### 1. 工具发现
```python
def discover_tools(self) -> List[Dict]:
    return registry.list_tools()
    # 返回格式: [
    #   {
    #     "name": "get_recently_submitted_cs_papers",
    #     "description": "...",
    #     "parameters": {"type": "object", "properties": {...}}
    #   },
    #   ...
    # ]
```

#### 2. Prompt 构建
```python
def build_messages(self, task, tools_description, history_text):
    prompt = get_react_prompt(
        task=task,                    # "获取最近7天的论文..."
        tools_description=tools_description,  # 工具列表格式化
        history=history_text,         # 之前的 Thought/Action/Observation
    )
    return [{"role": "user", "content": prompt}], {}
```

**Prompt 模板** (`agents/prompt_templates.py`)：
```
你是一个AI研究助手，可以获取最新的arXiv计算机科学论文。你有以下工具可以使用：

{tools_description}

当前任务：{task}

请按照ReAct框架的格式思考和行动:
Thought: 分析当前情况和下一步需要做什么
Action: {"name":"工具名称","args":{...}}
Observation: 工具执行的结果

当你认为任务已经完成时，使用以下格式结束：
Thought: 任务已完成
Action: FINISH

{history}
```

#### 3. LLM 响应解析
```python
def parse_response(self, raw_response: Dict) -> Tuple[str, Optional[Dict]]:
    content = raw_response["choices"][0]["message"]["content"]
    return self._parse_react_text(content)

def _parse_react_text(self, response: str) -> Tuple[str, Optional[Dict]]:
    # 正则提取 Thought
    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", response, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else "..."
    
    # 正则提取 Action
    action_match = re.search(r"Action:\s*(.*?)(?=\nObservation:|$)", response, re.DOTALL)
    action_text = action_match.group(1).strip()
    
    # 判断是否结束
    if action_text.upper() == "FINISH":
        return thought, None  # ← None 表示结束
    
    # JSON 解析 Action
    try:
        json_match = re.search(r"({.*})", action_text, re.DOTALL)
        if json_match:
            action_json = json.loads(json_match.group(1))
            if "name" in action_json and "args" in action_json:
                return thought, {
                    "name": action_json["name"],
                    "args": action_json["args"],
                }
    except json.JSONDecodeError:
        log.error(f"JSON解析失败")
    
    return thought, None
```

#### 4. 工具执行
```python
def invoke_tool(self, tool_name: str, args: Dict) -> Any:
    return registry.execute_tool(tool_name, args)
    # ← 直接同步调用，进程内
```

### 优缺点

| 优点 | 缺点 |
|---|---|
| 最快（无通信延迟） | 对 JSON 格式要求严格 |
| 最简单（纯正则） | LLM 易生成格式错误 |
| 最稳定（同步） | 工具无隔离 |

---

## B. MCPAgent — MCP 协议跨进程

**文件**：`mcp_protocol/mcp_agent.py` + `mcp_protocol/server.py`

### 什么是 MCP？

Model Context Protocol 是 Anthropic 开发的标准，允许 LLM 通过 JSON-RPC 发现和调用远程工具。

### 架构

```
FastAPI + MCPAgent          MCP 服务器 (子进程)
┌──────────────────┐        ┌──────────────────┐
│  LLM 循环        │        │  Tool Registry   │
│  (主线程)        │        │  + Executors     │
│                  │        │  (worker 线程)   │
└────────┬─────────┘        └─────────┬────────┘
         │ asyncio            │
         │ stdio JSON-RPC     │
         └────────────────────┘
```

### 工作流程

#### 1. 启动 MCP 会话

```python
class MCPAgent(BaseAgent):
    agent_type = "mcp"
    
    def run(self, task, agent_model, session_id):
        # 覆写 run，在 MCP 会话中执行
        return self._run_with_mcp(task, agent_model, session_id)
    
    async def _async_run(self):
        # 启动 MCP 服务器作为子进程
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_protocol.server"],  # ← 子进程模块
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                
                # 通过 MCP 获取工具列表
                tools_result = await session.list_tools()
                self._mcp_tools = [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema or {},
                    }
                    for t in tools_result.tools
                ]
                
                # 在线程池中运行同步的 BaseAgent.run()
                result = await self._loop.run_in_executor(
                    None,
                    lambda: super(MCPAgent, self).run(task, agent_model, session_id),
                )
                
                return result
```

**关键设计**：
- 保持 event loop 处理 JSON-RPC（async）
- 在线程池运行同步的 BaseAgent（executor）
- 两者通过 `asyncio.run_coroutine_threadsafe()` 通信

#### 2. 工具调用

```python
def invoke_tool(self, tool_name: str, args: Dict) -> Any:
    # 从工作线程调度 MCP call_tool 到 event loop
    async def _call():
        result = await self._session.call_tool(tool_name, arguments=args)
        texts = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
        return "\n".join(texts) if texts else ""
    
    # 从线程池调度到 event loop
    future = asyncio.run_coroutine_threadsafe(_call(), self._loop)
    return future.result(timeout=120)
```

#### 3. MCP 服务器 (子进程)

```python
# mcp_protocol/server.py
from mcp.server import Server
from tools.tool_registry import registry

server = Server("agentic-arxiv")

@server.list_tools()
async def list_tools():
    # 返回 registry 中的所有工具
    return [
        Tool(
            name=tool["name"],
            description=tool["description"],
            inputSchema=tool["parameters"],
        )
        for tool in registry.list_tools()
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    # 执行工具
    result = registry.execute_tool(name, arguments)
    return [TextContent(text=json.dumps(result))]
```

**流程**：
1. FastAPI 启动 MCPAgent
2. MCPAgent 启动子进程运行 `mcp_protocol/server.py`
3. 主进程通过 stdio 与子进程通信
4. 工具执行在子进程的 registry 中

### 优缺点

| 优点 | 缺点 |
|---|---|
| 符合 MCP 标准 | 通信延迟（stderr/stdout） |
| 工具隔离（不同进程） | 配置复杂 |
| 易扩展（支持远程服务器） | 需要 mcp 库依赖 |

---

## C. SkillAgent — CLI 命令驱动

**文件**：`skill_cli/skill_agent.py` + `skill_cli/tool_cli.py` + `skill_cli/SKILL.md`

### 核心思想

与其向 LLM 发送工具列表 JSON，不如发送**易读的文档**，让 LLM 理解后**生成命令行**。

### 工作流程

#### 1. 加载 SKILL 文档

```python
class SkillAgent(BaseAgent):
    agent_type = "skill_cli"
    
    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client)
        self._skill_doc = self._load_skill_doc()
    
    @staticmethod
    def _load_skill_doc() -> str:
        with open("skill_cli/SKILL.md", "r", encoding="utf-8") as f:
            content = f.read()
        # 去掉 YAML frontmatter
        match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
        if match:
            return content[match.end():]
        return content
```

**SKILL.md 内容示例**：
```markdown
# 可用命令

## search_papers — 检索论文
python tool_cli.py search_papers --session_id=<id> --days=<days> --aspect=<cs.ML|cs.AI|...> --max_results=<n>

## download_pdf — 下载论文 PDF
python tool_cli.py download_pdf --session_id=<id> --ref=<序号>

...
```

#### 2. Prompt 构建

```python
def format_tools_for_prompt(self, tools: List[Dict]) -> str:
    # 覆写方法，返回 SKILL.md 而非工具 JSON
    return self._skill_doc

def build_messages(self, task, tools_description, history_text):
    # tools_description 现在是 SKILL.md 内容
    prompt = get_skill_prompt(
        task=task,
        skill_document=tools_description,  # ← SKILL.md
        history=history_text,
    )
    return [{"role": "user", "content": prompt}], {}
```

**Skill Prompt 模板**：
```
你是一个CLI专家，能够理解和执行shell命令。

根据以下可用命令文档，为用户的请求生成合适的命令：

{skill_document}

用户请求：{task}

请按照以下格式输出：
Thought: 分析请求，决定使用哪个命令
Command:
```bash
<实际命令>
```

Observation: 命令执行的结果

{history}
```

#### 3. 响应解析

```python
def parse_response(self, raw_response: Dict) -> Tuple[str, Optional[Dict]]:
    content = raw_response["choices"][0]["message"]["content"]
    return self._parse_skill_text(content)

def _parse_skill_text(self, response: str) -> Tuple[str, Optional[Dict]]:
    # 提取 Thought
    thought_match = re.search(r"Thought:\s*(.*?)(?=\nCommand:|$)", response, re.DOTALL)
    thought = thought_match.group(1).strip()
    
    # 提取 Command
    cmd_match = re.search(r"Command:\s*(.*?)(?=\nObservation:|$)", response, re.DOTALL)
    cmd_text = cmd_match.group(1).strip()
    
    if cmd_text.upper() == "FINISH":
        return thought, None
    
    # 提取 ```bash ... ``` 代码块
    bash_match = re.search(r"```(?:bash)?\s*\n?(.*?)\n?```", cmd_text, re.DOTALL)
    raw_cmd = bash_match.group(1).strip() if bash_match else cmd_text.strip()
    
    # 从命令解析出子命令和参数
    tool_name, args = self._parse_cli_command(raw_cmd)
    if not tool_name:
        return thought, None
    
    # 映射到 registry 工具名
    registry_name = CLI_TO_REGISTRY.get(tool_name, tool_name)
    
    return thought, {"name": registry_name, "args": args}

@staticmethod
def _parse_cli_command(raw_cmd: str) -> Tuple[Optional[str], Dict]:
    """从 CLI 命令字符串中解析子命令和参数"""
    parts = shlex.split(raw_cmd)
    
    # 查找子命令 (search_papers, download_pdf, translate_pdf, ...)
    sub_cmd = None
    for p in parts:
        if p in CLI_TO_REGISTRY:
            sub_cmd = p
            break
    
    if not sub_cmd:
        return None, {}
    
    # 解析 --key=value 参数
    args = {}
    for p in parts:
        m = re.match(r"--(\w+)=(.+)", p)
        if m:
            key, val = m.group(1), m.group(2)
            # 类型推断
            if val.lower() in ("true", "false"):
                args[key] = val.lower() == "true"
            elif val.lower() == "none":
                args[key] = None
            else:
                try:
                    args[key] = int(val)
                except ValueError:
                    args[key] = val
    
    return sub_cmd, args
```

#### 4. 工具执行（子进程）

```python
def invoke_tool(self, tool_name: str, args: Dict) -> Any:
    cli_name = REGISTRY_TO_CLI.get(tool_name)
    if not cli_name:
        # 非 CLI 工具，回退到 registry
        from tools.tool_registry import registry
        return registry.execute_tool(tool_name, args)
    
    # 从修正后的 args 重建命令
    cmd_parts = [sys.executable, "skill_cli/tool_cli.py", cli_name]
    for k, v in args.items():
        if v is not None and not k.startswith("_"):
            cmd_parts.append(f"--{k}={v}")
    
    log.info(f"执行 CLI: {' '.join(cmd_parts)}")
    
    try:
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT,
        )
    except subprocess.TimeoutExpired:
        return "命令执行超时 (120s)"
    except Exception as e:
        return f"命令执行异常: {e}"
    
    if result.returncode != 0:
        return f"命令失败 (exit {result.returncode}): {result.stderr[:500]}"
    
    stdout = result.stdout.strip()
    
    # 尝试解析为 JSON
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return stdout[:1000]
```

**CLI 子进程** (`skill_cli/tool_cli.py`)：
```python
import argparse
from tools.tool_registry import registry

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    # search_papers 子命令
    sp = subparsers.add_parser("search_papers")
    sp.add_argument("--session_id", required=True)
    sp.add_argument("--days", type=int, default=7)
    sp.add_argument("--aspect", default="cs.AI")
    sp.add_argument("--max_results", type=int, default=5)
    
    args = parser.parse_args()
    
    # 调用 registry
    result = registry.execute_tool(
        "get_recently_submitted_cs_papers",
        {
            "session_id": args.session_id,
            "days": args.days,
            "aspect": args.aspect,
            "max_results": args.max_results,
        }
    )
    
    # 输出 JSON，供 subprocess 调用者解析
    print(json.dumps(result))

if __name__ == "__main__":
    main()
```

### 优缺点

| 优点 | 缺点 |
|---|---|
| 文档式，易理解 | 子进程开销 |
| 命令可读性高 | 安全风险（bash 注入） |
| 兼容现有 CLI | 启动多次进程 |

---

## 通用部分：BaseAgent

**文件**：`agents/base_agent.py`

### 执行循环

```python
def run(self, task: str, agent_model: str = None, session_id: str = "default") -> Dict:
    log.info(f"开始执行任务: {task}")
    self.session_id = session_id
    msg_id = uuid.uuid4().hex
    
    # 记录用户消息
    try:
        log_service.create_chat_log(session_id, msg_id, "user", task, agent_type=self.agent_type)
    except Exception:
        pass
    
    tools = self.discover_tools()  # 子类实现
    tools_description = self.format_tools_for_prompt(tools)
    
    # 注入会话上下文，帮助 LLM 避免重复搜索
    enriched_task = self._enrich_task_with_context(task, session_id)
    
    history: List[Dict[str, str]] = []
    
    for iteration in range(self.max_iterations):  # 最多 5 次迭代
        history_text = self.format_history(history)
        
        # 构建 LLM 请求（子类实现）
        messages, extra = self.build_messages(enriched_task, tools_description, history_text)
        
        try:
            # LLM 调用
            t0 = time.time()
            response = self.llm_client.chat_completions(
                model=agent_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
                stream=False,
                extra=extra or None,
            )
            llm_ms = int((time.time() - t0) * 1000)
            
            # 解析响应（子类实现）
            thought, action_dict = self.parse_response(response)
            log.info(f"Thought: {thought}")
            
            # 判断是否结束
            if action_dict is None:
                log.info("任务完成")
                observation = "任务完成"
                history.append({"thought": thought, "action": "FINISH", "observation": observation})
                self._log_step(msg_id, iteration, thought, "FINISH", "{}", observation, llm_ms, 0, session_id)
                break
            
            # 执行工具（带副作用）
            t1 = time.time()
            observation = self._execute_with_side_effects(action_dict)
            tool_ms = int((time.time() - t1) * 1000)
            
            # 记录历史
            history.append({
                "thought": thought,
                "action": json.dumps(action_dict, ensure_ascii=False),
                "observation": observation,
            })
            
            # 记录日志 + SSE 推送
            self._log_step(msg_id, iteration, thought, action_dict.get("name", ""), 
                          json.dumps(action_dict.get("args", {})), observation[:4000], 
                          llm_ms, tool_ms, session_id)
            
            # 达到迭代限制
            if iteration == self.max_iterations - 1:
                log.warning("达到最大迭代次数，强制结束")
                break
        
        except Exception as e:
            error_msg = f"LLM调用失败: {str(e)}"
            log.error(error_msg)
            history.append({"thought": "LLM调用失败", "action": "ERROR", "observation": error_msg})
            break
    
    # 提取最终结果
    final_observation = history[-1]["observation"] if history else "无执行结果"
    reply = ""
    for step in reversed(history):
        if step.get("action") not in ("FINISH", "FORCE_STOP", "ERROR"):
            reply = step.get("observation", "")
            break
    reply = reply or final_observation
    
    # 记录助手回复
    try:
        log_service.create_chat_log(session_id, msg_id + "_reply", "assistant", reply, agent_type=self.agent_type)
    except Exception:
        pass
    
    return {
        "task": task,
        "msg_id": msg_id,
        "history": history,
        "final_observation": final_observation,
    }
```

### 会话上下文注入

```python
def _enrich_task_with_context(self, task: str, session_id: str) -> str:
    """向任务注入当前会话的论文列表，帮助 LLM 避免重复搜索"""
    try:
        papers = store.get_last_papers(session_id)
        if papers:
            titles = [f"  {i+1}. {p.title}" for i, p in enumerate(papers[:10])]
            ctx = f"\n\n[会话上下文] 当前会话已有 {len(papers)} 篇论文:\n" + "\n".join(titles)
            ctx += "\n可直接用 ref 序号引用，无需重新搜索。"
            return task + ctx
    except Exception:
        pass
    return task
```

### 副作用处理

```python
def _execute_with_side_effects(self, action_dict: Dict[str, Any]) -> str:
    """统一处理 session_id 覆盖、翻译异步、paper 状态写入"""
    try:
        tool_name = action_dict["name"]
        args = action_dict.get("args", {}) or {}
        
        # 强制覆盖 session_id（防止 LLM 传错）
        try:
            tool = registry.get_tool(tool_name)
            props = (tool or {}).get("parameters", {}).get("properties", {})
            if "session_id" in props:
                args["session_id"] = self.session_id
        except Exception:
            pass
        
        # 验证工具存在
        available_tools = [t["name"] for t in registry.list_tools()]
        if tool_name not in available_tools:
            return f"错误: 工具 '{tool_name}' 不存在"
        
        # 翻译工具异步处理
        if tool_name == "translate_arxiv_pdf":
            t = translate_runner.enqueue(
                session_id=self.session_id,
                ref=args.get("ref"),
                force=bool(args.get("force", False)),
                service=args.get("service") or settings.pdf2zh_service,
                threads=int(args.get("threads") or settings.pdf2zh_threads),
                keep_dual=bool(args.get("keep_dual", False)),
                paper_id=args.get("paper_id"),
                pdf_url=args.get("pdf_url"),
                input_pdf_path=args.get("input_pdf_path"),
            )
            return f"已创建翻译任务 task_id={t.task_id}，状态={t.status}"
        
        # 调用工具（子类实现）
        result = self.invoke_tool(tool_name, args)
        
        # 论文 ID 写入 last_active
        try:
            if isinstance(result, dict):
                pid = result.get("paper_id")
                if isinstance(pid, str) and pid.strip():
                    store.set_last_active_paper_id(self.session_id, pid.strip())
        except Exception:
            pass
        
        # arXiv 搜索结果存入 session
        if tool_name == "get_recently_submitted_cs_papers":
            if isinstance(result, list) and result:
                papers_obj = [Paper(**p) for p in result]
                store.set_last_papers(self.session_id, papers_obj)
                return f"成功获取 {len(result)} 篇论文"
        
        # 通用格式化
        if isinstance(result, list):
            return f"成功获取 {len(result)} 条记录"
        elif isinstance(result, str):
            return result[:1000] if len(result) > 1000 else result
        else:
            return str(result)[:1000]
    
    except Exception as e:
        log.error(f"工具执行失败: {str(e)}", exc_info=True)
        return f"工具执行失败: {str(e)}"
```

### 日志 + SSE

```python
def _log_step(self, msg_id, step_index, thought, action_name, action_args, observation, llm_ms, tool_ms, session_id):
    try:
        # 数据库记录
        log_service.save_agent_step(
            msg_id=msg_id,
            step_index=step_index,
            thought=thought,
            action_name=action_name,
            action_args=action_args,
            observation=observation,
            llm_latency_ms=llm_ms,
            tool_latency_ms=tool_ms,
        )
        
        # SSE 实时推送
        event_bus.publish(session_id, {
            "type": "agent_step",
            "step": {
                "thought": thought,
                "action_name": action_name,
                "observation": observation[:500],
                "step_index": step_index,
                "llm_latency_ms": llm_ms,
                "tool_latency_ms": tool_ms,
            },
        })
    except Exception as e:
        log.warning(f"Failed to log step: {e}")
```

---

## 工具系统

**文件**：`tools/tool_registry.py` + `tools/arxiv_tool.py` + ...

### 工具注册

```python
# tools/tool_registry.py
class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self._executors = {}
    
    def register_tool(self, name: str, description: str, parameters: Dict, handler: Callable):
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        self._executors[name] = handler
    
    def list_tools(self) -> List[Dict]:
        return list(self._tools.values())
    
    def execute_tool(self, name: str, args: Dict) -> Any:
        handler = self._executors.get(name)
        if not handler:
            raise ValueError(f"Tool {name} not found")
        return handler(**args)
    
    def get_tool(self, name: str) -> Dict:
        return self._tools.get(name)

registry = ToolRegistry()
```

### 工具定义示例

```python
# tools/arxiv_tool.py
def get_recently_submitted_cs_papers(session_id: str, aspect: str, days: int, max_results: int) -> List[Dict]:
    """
    检索最近 N 天内计算机科学领域的论文
    
    Args:
        session_id: 会话 ID
        aspect: 研究方向 (cs.AI, cs.ML, ...)
        days: 天数范围
        max_results: 最多结果数
    
    Returns:
        论文列表 (JSON 序列化)
    """
    # 实现逻辑
    papers = arxiv.query(f"cat:{aspect} AND submittedDate:[{start_date} TO {end_date}]")
    return papers[:max_results]

# 注册工具
register_tool(
    name="get_recently_submitted_cs_papers",
    description="检索最近 N 天内计算机科学领域的论文",
    parameters={
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "会话 ID"},
            "aspect": {"type": "string", "description": "研究方向", "enum": ["cs.AI", "cs.ML", ...]},
            "days": {"type": "integer", "description": "天数范围", "default": 7},
            "max_results": {"type": "integer", "description": "最多结果数", "default": 5},
        },
        "required": ["session_id", "aspect"],
    },
    handler=get_recently_submitted_cs_papers,
)
```

---

## 数据持久化

### ORM 模型

```python
# models/orm.py

class ChatLogRow(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(128), nullable=False, index=True)
    msg_id = Column(String(64), unique=True, nullable=False, index=True)
    role = Column(String(16), nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=True)
    model = Column(String(128), nullable=True)
    agent_type = Column(String(32), nullable=True)  # regex | mcp | skill_cli
    created_at = Column(DateTime, default=datetime.now)

class AgentStepRow(Base):
    __tablename__ = "agent_steps"
    id = Column(Integer, primary_key=True)
    msg_id = Column(String(64), nullable=False, index=True)
    step_index = Column(Integer, nullable=False)
    thought = Column(Text, nullable=True)
    action_name = Column(String(128), nullable=True)
    action_args = Column(Text, nullable=True)  # JSON string
    observation = Column(Text, nullable=True)
    llm_latency_ms = Column(Integer, nullable=True)
    tool_latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
```

### 日志服务

```python
# services/log_service.py
class LogService:
    def create_chat_log(self, session_id, msg_id, role, content, model=None, agent_type=None):
        log_entry = ChatLogRow(
            session_id=session_id,
            msg_id=msg_id,
            role=role,
            content=content,
            model=model,
            agent_type=agent_type,
        )
        session = get_sync_session()
        session.add(log_entry)
        session.commit()
    
    def save_agent_step(self, msg_id, step_index, thought, action_name, action_args, observation, llm_latency_ms, tool_latency_ms):
        step_entry = AgentStepRow(
            msg_id=msg_id,
            step_index=step_index,
            thought=thought,
            action_name=action_name,
            action_args=action_args,
            observation=observation,
            llm_latency_ms=llm_latency_ms,
            tool_latency_ms=tool_latency_ms,
        )
        session = get_sync_session()
        session.add(step_entry)
        session.commit()

log_service = LogService()
```

---

## 开发指南

### 添加新 Agent 方案

```python
# agents/my_custom_agent.py
class MyCustomAgent(BaseAgent):
    agent_type = "my_custom"
    
    def discover_tools(self) -> List[Dict]:
        # 实现工具发现逻辑
        pass
    
    def build_messages(self, task, tools_description, history_text) -> Tuple[List[Dict], Dict]:
        # 实现 Prompt 构建逻辑
        pass
    
    def parse_response(self, raw_response) -> Tuple[str, Optional[Dict]]:
        # 实现 LLM 响应解析逻辑
        pass
    
    def invoke_tool(self, tool_name, args) -> Any:
        # 实现工具调用逻辑
        pass
```

然后在 `api/endpoints.py` 注册：
```python
AGENT_CLASSES = {
    "regex": ReActAgent,
    "mcp": MCPAgent,
    "skill_cli": SkillAgent,
    "my_custom": MyCustomAgent,  # ← 新增
}
```

---

## 总结

| 架构 | 优点 | 缺点 | 使用场景 |
|---|---|---|---|
| **ReActAgent** | 快、简、稳 | JSON 格式依赖强 | ✅ 推荐默认 |
| **MCPAgent** | 标准、隔离、可扩展 | 延迟、复杂 | 团队开发、工具服务化 |
| **SkillAgent** | 易理解、命令可读 | 子进程开销、安全风险 | 学习研究、CLI 工作流 |

