from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
from loguru import logger
import sys
import os
import urllib.parse
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.core.llm_client import LLMClient
from src.core.react_agent import ReActAgent
from src.tools.weather_tool import WeatherTool

# 初始化FastAPI应用
app = FastAPI(
    title="Weather Agent API",
    description="基于ReAct模式的天气查询智能Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
agent = None
weather_tool = None


# 数据模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="用户查询语句")
    reset_history: bool = Field(False, description="是否重置对话历史")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "北京天气怎么样？",
                "reset_history": False
            }
        }


class QueryResponse(BaseModel):
    """查询响应模型"""
    query: str
    response: str
    status: str
    total_steps: Optional[int] = None
    available_cities: Optional[List[str]] = None
    error: Optional[str] = None
    full_process: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "query": "北京天气怎么样？",
                "response": "北京目前是晴天，温度22摄氏度，湿度45%。",
                "status": "success",
                "total_steps": 2,
                "available_cities": ["北京", "上海", "广州", "深圳", "杭州", "成都"]
            }
        }


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    service: str
    llm_connected: bool
    available_tools: List[str]
    timestamp: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "weather-agent",
                "llm_connected": True,
                "available_tools": ["get_weather"],
                "timestamp": "2024-01-15T10:00:00"
            }
        }


@app.on_event("startup")
async def startup_event():
    """启动应用时初始化Agent"""
    global agent, weather_tool

    logger.info("正在启动Weather Agent服务...")

    try:
        # 初始化LLM客户端
        llm_client = LLMClient()

        # 初始化天气工具
        weather_tool = WeatherTool()

        # 初始化Agent
        agent = ReActAgent(
            llm_client=llm_client,
            tools={"get_weather": weather_tool.get_weather}
        )

        logger.info("Weather Agent初始化成功")

        # 测试LLM连接
        test_response = llm_client.generate([{"role": "user", "content": "你好"}])
        if "Error" not in test_response:
            logger.info(f"LLM连接测试成功: {test_response[:50]}...")
        else:
            logger.warning(f"LLM连接测试可能有问题: {test_response}")

    except Exception as e:
        logger.error(f"Agent初始化失败: {e}")
        raise


@app.get("/", tags=["首页"])
async def root():
    """API首页"""
    return {
        "message": "欢迎使用Weather Agent API",
        "docs": "/docs",
        "endpoints": {
            "健康检查": "/health",
            "查询天气": "/query",
            "支持的城市": "/cities",
            "直接对话": "/chat",
            "演示页面": "/demo"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["系统状态"])
async def health_check():
    """健康检查端点"""
    llm_connected = False
    if agent:
        try:
            # 简单测试LLM连接
            test_response = agent.llm.generate([{"role": "user", "content": "test"}])
            llm_connected = "Error" not in test_response
        except Exception as e:
            logger.error(f"LLM连接测试失败: {e}")
            llm_connected = False

    return {
        "status": "healthy",
        "service": "weather-agent",
        "llm_connected": llm_connected,
        "available_tools": list(agent.tools.keys()) if agent else [],
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.get("/cities", tags=["工具"])
async def get_available_cities():
    """获取支持查询的城市列表"""
    if not weather_tool:
        raise HTTPException(status_code=503, detail="服务未就绪")

    cities = weather_tool.get_all_cities()
    return {
        "available_cities": cities,
        "count": len(cities),
        "description": "支持查询天气的城市列表"
    }


@app.post("/query", response_model=QueryResponse, tags=["核心功能"])
async def query_agent(request: QueryRequest = Body(...)):
    """向Agent发送查询请求"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent未初始化")

    try:
        # 如果需要重置历史
        if request.reset_history:
            agent.reset_history()
            logger.info("已重置对话历史")

        # 处理查询
        result = agent.run(request.query)

        # 获取支持的城市列表
        if weather_tool:
            cities = weather_tool.get_all_cities()
        else:
            cities = []

        response_data = {
            "query": result["query"],
            "response": result["response"],
            "status": result["status"],
            "total_steps": result.get("total_steps"),
            "available_cities": cities,
            "error": result.get("error"),
            "full_process": result.get("full_process") if request.query.startswith("debug:") else None
        }

        logger.info(f"查询处理完成: {request.query} -> 状态: {result['status']}")
        return response_data

    except Exception as e:
        logger.error(f"处理查询时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理查询时出错: {str(e)}")


@app.get("/chat", tags=["快速测试"])
async def chat(
        message: str = Query(..., description="用户消息"),
        reset: bool = Query(False, description="是否重置历史"),
        show_process: bool = Query(False, description="是否显示完整过程")
):
    """快速聊天接口（GET请求）"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent未初始化")

    try:
        # 解码URL编码的消息
        decoded_message = urllib.parse.unquote(message)

        if reset:
            agent.reset_history()

        # 处理查询
        result = agent.run(decoded_message)

        response_data = {
            "user_message": decoded_message,
            "assistant_response": result["response"],
            "status": result["status"],
            "steps": result.get("total_steps", 0)
        }

        if show_process:
            response_data["full_process"] = result.get("full_process")

        return response_data

    except Exception as e:
        logger.error(f"处理聊天消息时出错: {e}")
        return {
            "user_message": message,
            "assistant_response": f"处理消息时出错: {str(e)}",
            "status": "error"
        }


@app.get("/demo", response_class=HTMLResponse, tags=["演示"])
async def demo():
    """演示页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Weather Agent Demo</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
            .container { background: white; border-radius: 10px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; margin-bottom: 30px; }
            .chat-container { height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; background-color: #fafafa; }
            .message { margin: 10px 0; padding: 10px 15px; border-radius: 10px; max-width: 80%; }
            .user { background-color: #e3f2fd; margin-left: auto; }
            .assistant { background-color: #f0f0f0; }
            .input-container { display: flex; gap: 10px; margin-top: 20px; }
            input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
            button { padding: 12px 24px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background-color: #45a049; }
            .reset-btn { background-color: #f44336; }
            .reset-btn:hover { background-color: #d32f2f; }
            .cities { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
            .city-btn { padding: 8px 16px; background-color: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .city-btn:hover { background-color: #0b7dda; }
            .status { color: #666; font-size: 14px; margin-top: 10px; text-align: center; }
            .timestamp { color: #999; font-size: 12px; float: right; }
            .thinking { color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌤️ Weather Agent Demo</h1>

            <div id="chat" class="chat-container"></div>

            <div class="input-container">
                <input type="text" id="message" placeholder="输入消息，如：北京天气怎么样？" autocomplete="off">
                <button onclick="sendMessage()">发送</button>
                <button onclick="resetChat()" class="reset-btn">重置对话</button>
            </div>

            <div class="cities">
                <div>快速查询：</div>
                <button class="city-btn" onclick="quickQuery('北京天气怎么样？')">北京</button>
                <button class="city-btn" onclick="quickQuery('上海天气如何？')">上海</button>
                <button class="city-btn" onclick="quickQuery('广州温度多少？')">广州</button>
                <button class="city-btn" onclick="quickQuery('深圳湿度多少？')">深圳</button>
                <button class="city-btn" onclick="quickQuery('杭州天气怎么样？')">杭州</button>
                <button class="city-btn" onclick="quickQuery('成都是什么天气？')">成都</button>
            </div>

            <div class="status">
                <div>支持的城市: 北京、上海、广州、深圳、杭州、成都</div>
                <div id="connection-status">连接状态: 正在检查...</div>
            </div>
        </div>

        <script>
            // 检查连接状态
            async function checkConnection() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    document.getElementById('connection-status').innerHTML = 
                        `连接状态: ✅ 正常 (LLM: ${data.llm_connected ? '已连接' : '未连接'})`;
                } catch (error) {
                    document.getElementById('connection-status').innerHTML = 
                        '连接状态: ❌ 无法连接到服务器';
                }
            }

            // 页面加载时检查连接
            window.onload = function() {
                checkConnection();
                addMessage('assistant', '你好！我是天气助手，可以帮你查询天气信息。试试点击上面的城市按钮或直接输入问题。');
            };

            // 每30秒检查一次连接
            setInterval(checkConnection, 30000);

            // 添加消息到聊天窗口
            function addMessage(sender, content, isThinking = false) {
                const chat = document.getElementById('chat');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;

                const timestamp = new Date().toLocaleTimeString();
                let contentHtml = content;

                if (isThinking) {
                    contentHtml = `<span class="thinking">🤔 ${content}</span>`;
                }

                messageDiv.innerHTML = `
                    <strong>${sender === 'user' ? '👤 你' : '🤖 助手'}:</strong>
                    ${contentHtml}
                    <span class="timestamp">${timestamp}</span>
                `;

                chat.appendChild(messageDiv);
                chat.scrollTop = chat.scrollHeight;
            }

            // 发送消息
            async function sendMessage() {
                const input = document.getElementById('message');
                const message = input.value.trim();

                if (!message) return;

                // 清空输入框
                input.value = '';

                // 添加用户消息
                addMessage('user', message);

                try {
                    // 显示思考中
                    addMessage('assistant', '正在思考...', true);

                    // 发送请求到API（使用GET /chat接口）
                    const encodedMessage = encodeURIComponent(message);
                    const response = await fetch(`/chat?message=${encodedMessage}`);

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }

                    const data = await response.json();

                    // 移除思考中的消息
                    const chat = document.getElementById('chat');
                    const lastMessage = chat.lastChild;
                    if (lastMessage && lastMessage.textContent.includes('正在思考')) {
                        chat.removeChild(lastMessage);
                    }

                    // 添加助手回复
                    addMessage('assistant', data.assistant_response);

                } catch (error) {
                    // 移除思考中的消息
                    const chat = document.getElementById('chat');
                    const lastMessage = chat.lastChild;
                    if (lastMessage && lastMessage.textContent.includes('正在思考')) {
                        chat.removeChild(lastMessage);
                    }

                    addMessage('assistant', `❌ 请求失败: ${error.message}`);
                }
            }

            // 快速查询
            function quickQuery(cityQuery) {
                document.getElementById('message').value = cityQuery;
                sendMessage();
            }

            // 重置对话
            async function resetChat() {
                const chat = document.getElementById('chat');
                chat.innerHTML = '';

                try {
                    await fetch('/chat?message=reset&reset=true');
                    addMessage('assistant', '对话历史已重置。有什么可以帮助你的吗？');
                } catch (error) {
                    addMessage('assistant', '重置失败: ' + error.message);
                }
            }

            // 按回车发送消息
            document.getElementById('message').addEventListener('keypress', function(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
    