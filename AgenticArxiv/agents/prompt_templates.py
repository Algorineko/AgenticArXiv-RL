# AgenticArxiv/agents/prompt_templates.py

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