# AgenticArxiv/agents/prompt_templates.py

from typing import Any, Dict, Mapping, Sequence

REACT_PROMPT_TEMPLATE = """你是一个AI研究助手,可以获取最新的arXiv计算机科学论文。你有以下工具可以使用：

{tools_description}

当前任务：{task}
请按照ReAct框架的格式思考和行动:
Thought: 分析当前情况和下一步需要做什么
Action: {{"name":"工具名称","args":{{参数对象}}}}
Observation: 工具执行的结果
当你认为任务已经完成时，使用以下格式结束：
Thought: 任务已完成
Action: FINISH
如果任务因为缺少必要上下文、参数非法或工具能力不支持而无法执行，不要编造
参数或调用注定失败的工具；请在 Thought 中明确说明无法执行的原因，再用
Action: FINISH 结束。此时不要声称任务已经完成。
注意：只能使用上面列出的工具。每次只能执行一个动作。
强约束：
- Action 后必须是“严格 JSON”(双引号、true/false/null, 小写)
- 禁止 Python 风格 True/False/None
- 禁止尾随逗号、注释、以及任何额外文本
关于“指代上一条操作的论文”：
- 当用户没有给出明确的论文序号/ID/标题，但你判断用户是在指代“最近一次操作过的那篇论文”（例如上下文中的指代性表达），
  则在调用需要论文引用的工具时，把 args 里的 ref 设置为 null(JSON null), 由工具自动定位最近操作的论文。
- 如果用户给出了明确序号/ID/标题，则正常传 ref。
关于异步任务（翻译）：
- translate_arxiv_pdf 是异步任务，调用后会立即返回 task_id 和 PENDING 状态。
- 翻译进度由前端通过 SSE 实时推送，你不需要也不应该轮询检查翻译状态。
- 调用 translate_arxiv_pdf 之后，直接 FINISH 即可，不要再调用 get_paper_cache_status 去查看翻译是否完成。
正确示例：
Action: {{"name":"translate_arxiv_pdf","args":{{"ref":2,"session_id":"demo1","force":false,"service":"bing","threads":4,"keep_dual":false}}}}
Action: {{"name":"download_arxiv_pdf","args":{{"ref":null,"session_id":"demo1","force":false}}}}
现在开始执行任务：
{history}
"""

def get_react_prompt(task: str, tools_description: str, history: str = "") -> str:
    """生成ReAct提示词"""
    return REACT_PROMPT_TEMPLATE.format(
        task=task,
        tools_description=tools_description,
        history=history
    )


def build_visible_setup_context(task: Mapping[str, Any]) -> str:
    """把任务开始前的 ``setup`` 转成模型可见的会话状态。

    ``setup`` 是 benchmark/RL 框架为了还原既有会话而执行的动作，不是
    当前策略应该生成的专家步骤。模型若完全看不到这些状态，同一句
    “把刚才那篇论文翻译一下”在空会话和已有活跃论文的会话中会拥有
    相反的最优动作，形成不可学习的观测混叠。

    这里只描述 setup 执行后必然成立的状态，不包含 task_id、``steps``、
    expected_tools 或奖励，因此不会把标准答案泄漏给策略。
    """
    setup: Sequence[Mapping[str, Any]] = task.get("setup") or []
    lines = ["[任务开始前的会话状态]"]
    if not setup:
        lines.append(
            "- 当前会话没有既有论文列表，也没有最近操作的论文；"
            "ref=null 暂时无法解析。"
        )
        return "\n".join(lines)

    has_paper_list = False
    last_active_ref = None
    for action in setup:
        name = str(action.get("name") or "")
        args: Dict[str, Any] = dict(action.get("args") or {})
        if name == "get_recently_submitted_cs_papers":
            count = int(args.get("max_results", 50))
            aspect = str(args.get("aspect", "*"))
            days = int(args.get("days", 7))
            has_paper_list = True
            lines.append(
                f"- 此前已检索最近 {days} 天的 cs.{aspect} 论文，"
                f"当前候选列表最多有 {count} 篇，可直接按 ref 序号引用。"
            )
        elif name == "search_arxiv_papers":
            count = int(args.get("max_results", 10))
            query = str(args.get("query") or "").strip()
            has_paper_list = True
            lines.append(
                f"- 此前已按关键词 {query!r} 检索论文，"
                f"当前候选列表最多有 {count} 篇，可直接按 ref 序号引用。"
            )
        elif name == "download_arxiv_pdf":
            last_active_ref = args.get("ref")
            lines.append(
                f"- 此前已下载 ref={last_active_ref!r} 的论文；"
                "它现在是最近操作的论文，ref=null 会指向它。"
            )
        elif name == "translate_arxiv_pdf":
            last_active_ref = args.get("paper_id", args.get("ref"))
            lines.append(
                f"- 此前已翻译 ref={last_active_ref!r} 的论文；"
                "它现在是最近操作的论文，ref=null 会指向它。"
            )
        elif name == "get_paper_cache_status":
            last_active_ref = args.get("paper_id", args.get("ref"))
            lines.append(
                f"- 此前已查看 ref={last_active_ref!r} 的缓存状态；"
                "它现在是最近操作的论文，ref=null 会指向它。"
            )

    if not has_paper_list:
        lines.append("- 当前会话没有由先前检索建立的论文候选列表。")
    lines.append("以上是已存在的会话状态，不属于当前任务动作，请勿重复执行。")
    return "\n".join(lines)

def format_tool_description(tools) -> str:
    """格式化工具描述 - 更详细的版本"""
    if not tools:
        return "当前没有可用工具"
    
    descriptions = []
    for tool in tools:
        name = tool.get('name', '未知工具')
        desc = tool.get('description', '无描述')
        
        params_info = ""
        if 'parameters' in tool and 'properties' in tool['parameters']:
            params = []
            for param_name, param_spec in tool['parameters']['properties'].items():
                param_type = param_spec.get('type', 'unknown')
                param_desc = param_spec.get('description', '')
                default_val = param_spec.get('default', '无默认值')
                
                if 'enum' in param_spec:
                    enum_vals = param_spec['enum']
                    if len(enum_vals) > 5:
                        param_info = f"{param_name} ({param_type}): {param_desc}, 可选值: {enum_vals[:3]}...等{len(enum_vals)}个值"
                    else:
                        param_info = f"{param_name} ({param_type}): {param_desc}, 可选值: {enum_vals}"
                else:
                    param_info = f"{param_name} ({param_type}): {param_desc}, 默认: {default_val}"
                
                params.append(f"    {param_info}")
            
            if params:
                params_info = "\n" + "\n".join(params)
        
        tool_desc = f"- {name}: {desc}"
        if params_info:
            tool_desc += "\n  参数:"
            tool_desc += params_info
        
        descriptions.append(tool_desc)
    
    return "\n\n".join(descriptions)
