import json
import re
from typing import Dict, Any, Callable, Tuple, Optional
from loguru import logger


class ReActAgent:
    """基于ReAct模式的智能Agent"""

    def __init__(self, llm_client, tools: Dict[str, Callable]):
        """
        Args:
            llm_client: LLM客户端实例
            tools: 工具字典，键为工具名，值为工具函数
        """
        self.llm = llm_client
        self.tools = tools
        self.max_steps = 5
        self.conversation_history = []

    def _create_system_prompt(self) -> str:
        """创建系统提示词"""
        tools_desc = []
        for name, func in self.tools.items():
            doc = func.__doc__ or "无描述"
            tools_desc.append(f"- {name}: {doc}")

        return f"""你是一个天气查询助手，可以使用以下工具：

                {chr(10).join(tools_desc)}

                请严格按照以下格式思考：

                思考: <你的推理过程>
                行动: <工具名称>[<参数>]
                观察: <工具返回的结果>
                回答: <给用户的最终回答>

                规则:
                1. 每次只能使用一个工具
                2. 工具参数必须是字符串格式
                3. 如果工具执行成功，根据结果回答用户
                4. 如果工具执行失败，向用户说明情况

                示例:
                用户: 北京天气怎么样？
                思考: 用户想了解北京的天气，我需要使用get_weather工具。
                行动: get_weather[北京]
                观察: {{"city": "北京", "temperature": 22, "weather": "晴天", "humidity": 45, "status": "success"}}
                思考: 根据工具返回的结果，北京的天气是晴天，温度22度，湿度45%。
                回答: 北京目前是晴天，温度22摄氏度，湿度45%。

                现在开始：
                """

    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """解析行动部分"""
        patterns = [
            r'行动:\s*(\w+)\[([^\]]+)\]',
            r'行动：\s*(\w+)\[([^\]]+)\]',
            r'行动:\s*(\w+)\(([^)]+)\)',
            r'行动:\s*使用(\w+)工具，参数为([^。]+)',
            r'Action:\s*(\w+)\[([^\]]+)\]',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                tool_name = match.group(1).strip()
                param = match.group(2).strip()
                # 清理参数
                param = param.strip('"\'')
                return tool_name, param

        return None, None

    def _extract_answer(self, text: str) -> Optional[str]:
        """提取回答部分"""
        # 多种模式匹配回答
        patterns = [
            r'回答:\s*(.+)',
            r'回答：\s*(.+)',
            r'Answer:\s*(.+)',
            r'最终回答:\s*(.+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # 清理回答中的思考内容 - 修复这里！
                # 使用split方法分割字符串，而不是re.split
                if '思考:' in answer:
                    answer = answer.split('思考:')[0].strip()
                if '行动:' in answer:
                    answer = answer.split('行动:')[0].strip()
                if '观察:' in answer:
                    answer = answer.split('观察:')[0].strip()
                return answer

        # 尝试匹配"所以"开头的回答
        so_pattern = r'所以[，,]?\s*(.+)'
        so_match = re.search(so_pattern, text, re.DOTALL)
        if so_match:
            answer = so_match.group(1).strip()
            # 清理
            if '思考:' in answer:
                answer = answer.split('思考:')[0].strip()
            if '行动:' in answer:
                answer = answer.split('行动:')[0].strip()
            return answer

        # 如果没有找到回答格式，尝试提取最后一句
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith(('思考:', '行动:', '观察:', '思考：', '行动：', '观察：')):
                # 检查是否是完整句子
                if len(line) > 10 and not line.startswith(('如果', '那么', '但是', '然而')):
                    return line

        return None

    def _extract_thought(self, text: str) -> str:
        """提取思考部分"""
        thought_patterns = [
            r'思考:\s*(.+)',
            r'思考：\s*(.+)',
            r'Thought:\s*(.+)',
        ]

        for pattern in thought_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                thought = match.group(1).strip()
                if '行动:' in thought:
                    thought = thought.split('行动:')[0].strip()
                if '观察:' in thought:
                    thought = thought.split('观察:')[0].strip()
                if '回答:' in thought:
                    thought = thought.split('回答:')[0].strip()
                return thought

        return ""

    def _execute_tool(self, tool_name: str, params: str) -> str:
        """执行工具调用"""
        logger.info(f"执行工具: {tool_name}, 参数: {params}")

        if tool_name in self.tools:
            try:
                result = self.tools[tool_name](params)
                return json.dumps(result, ensure_ascii=False)
            except Exception as e:
                error_msg = f"工具执行失败: {str(e)}"
                logger.error(error_msg)
                return json.dumps({"error": error_msg, "status": "error"}, ensure_ascii=False)
        else:
            error_msg = f"工具'{tool_name}'不存在"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "status": "error"}, ensure_ascii=False)

    def reset_history(self):
        """重置对话历史"""
        self.conversation_history = []
        logger.info("对话历史已重置")

    def run(self, user_input: str) -> Dict[str, Any]:
        """运行Agent处理用户输入

        Returns:
            包含响应和元数据的字典
        """
        logger.info(f"处理用户输入: {user_input}")

        # 创建完整的提示词
        system_prompt = self._create_system_prompt()

        # 构建消息历史
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        response_text = ""
        step_details = []
        last_observation = None

        for step in range(self.max_steps):
            logger.info(f"第 {step + 1}/{self.max_steps} 步")

            # 获取LLM响应
            llm_response = self.llm.generate(messages)
            response_text += llm_response + "\n"

            # 解析响应
            thought = self._extract_thought(llm_response)
            tool_name, params = self._parse_action(llm_response)
            answer = self._extract_answer(llm_response)

            step_detail = {
                "step": step + 1,
                "thought": thought,
                "llm_response": llm_response[:200] + "..." if len(llm_response) > 200 else llm_response
            }

            if tool_name:
                # 执行工具
                observation = self._execute_tool(tool_name, params)
                last_observation = observation

                step_detail["action"] = {
                    "tool": tool_name,
                    "params": params,
                    "observation": observation[:200] + "..." if len(observation) > 200 else observation
                }

                # 构建下一轮的消息
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content": f"观察: {observation}"})

            elif answer:
                # 找到最终答案
                logger.info(f"找到最终答案: {answer}")

                # 更新对话历史
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": answer})

                step_detail["answer"] = answer
                step_details.append(step_detail)

                return {
                    "query": user_input,
                    "response": answer,
                    "full_process": response_text.strip(),
                    "steps": step_details,
                    "total_steps": step + 1,
                    "status": "success"
                }
            else:
                # 没有行动也没有答案，尝试继续
                step_detail["action"] = "无行动，继续思考"
                step_details.append(step_detail)

                # 添加观察提示
                if last_observation:
                    continue_prompt = f"请继续思考。观察结果: {last_observation}"
                else:
                    continue_prompt = "请继续思考，并给出回答。"

                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content": continue_prompt})

        # 达到最大步数
        logger.warning(f"达到最大步数 {self.max_steps}，未找到答案")

        # 尝试从最后一次响应中提取任何可能的回答
        last_answer = self._extract_answer(llm_response)
        if last_answer:
            return {
                "query": user_input,
                "response": last_answer,
                "full_process": response_text.strip(),
                "steps": step_details,
                "total_steps": self.max_steps,
                "status": "partial_success"
            }

        return {
            "query": user_input,
            "response": "抱歉，我思考了太久还是没能找到答案。请尝试更清晰的提问，比如'北京天气怎么样？'",
            "full_process": response_text.strip(),
            "steps": step_details,
            "total_steps": self.max_steps,
            "status": "timeout",
            "error": "达到最大思考步数"
        }
    