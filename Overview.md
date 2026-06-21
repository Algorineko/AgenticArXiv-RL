# AgenticArxiv

## 项目概览

ReAct Agent + arXiv 论文管理系统。支持三种 Agent 架构（ReAct、MCP、Skill/CLI），通过简单配置切换。核心特色：

- 🤖 **可扩展 Agent 框架**：三层架构（BaseAgent 通用循环 → 三种实现 → 工具执行）
- 🔄 **实时 SSE 推送**：Agent 思考步骤、翻译进度实时更新
- 💾 **自动化数据初始化**：FastAPI 启动时自动创建 7 张表，无需手动迁移
- 🎨 **现代化前端**：Vue 3 + TypeScript，5 页面布局（KeepAlive 状态保留）

## 技术栈

- **后端**：FastAPI + uvicorn, SQLAlchemy 2.0 (pymysql), MySQL 8
- **前端**：Vue 3 + TypeScript + Pinia + Vite
- **Agent**：三种架构并存
  - `regex`：正则解析 + 进程内调用（推荐默认，最快）
  - `mcp`：MCP JSON-RPC 跨进程（标准、隔离）
  - `skill_cli`：Skill 文档 + CLI 子进程（易理解）

## 关键架构设计

### Agent 三层架构

```
BaseAgent (通用执行循环)
  ├─ ReActAgent (正则解析)
  ├─ MCPAgent (MCP 跨进程)
  └─ SkillAgent (CLI 命令)
       └─ ToolRegistry (进程内函数)
```

### 工作流程

1. **工具发现**：子类 `discover_tools()` 返回 `[{name, description, parameters}]`
2. **Prompt 构建**：`build_messages()` 注入历史和工具列表
3. **LLM 调用**：通过 `llm_client.chat_completions()`
4. **响应解析**：`parse_response()` 提取 `thought` 和 `action_dict`
5. **副作用处理**：`_execute_with_side_effects()` 统一处理 session、翻译异步、论文状态
6. **工具执行**：`invoke_tool()` 调用底层工具
7. **日志 + SSE**：`_log_step()` 记录数据库 + 实时推送到前端

### 数据库初始化（关键）

**完全自动化**，无需手动迁移脚本：

```python
# api/app.py 启动时执行（lifespan context manager）
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Initializing database tables...")
    init_db()  # ← 自动创建 7 张表
    store.validate_local_paths()
    yield

# models/db.py
def init_db():
    import models.orm  # 导入 ORM 注册到 Base.metadata
    Base.metadata.create_all(bind=engine)  # SQLAlchemy 自动创建
```

**特性**：
- ✅ 幂等性（可安全重复调用）
- ✅ 无需 Alembic/FlywayDB
- ✅ 自动创建索引
- 表修改需手动 ALTER（对学生项目无问题）

## 目录结构

```
AgenticArxiv/                    # Python 后端
  agents/                        # Agent 核心
    ├─ base_agent.py            # 通用执行循环 + SSE + 日志（最重要）
    ├─ agent_engine.py          # ReActAgent 实现
    ├─ prompt_templates.py      # Prompt 模板
    └─ context_manager.py       # 上下文管理
  mcp_protocol/                 # MCP 方案
    ├─ server.py                # MCP 服务器
    └─ mcp_agent.py             # MCPAgent 实现
  skill_cli/                    # Skill/CLI 方案
    ├─ SKILL.md                 # 工具文档（LLM 读这个）
    ├─ skill_agent.py           # SkillAgent 实现
    └─ tool_cli.py              # CLI 入口
  tools/                        # 工具实现
    ├─ tool_registry.py         # 工具注册表
    ├─ arxiv_tool.py            # arXiv 搜索
    ├─ pdf_download_tool.py     # PDF 下载
    ├─ pdf_translate_tool.py    # PDF 翻译
    └─ cache_status_tool.py     # 缓存查询
  models/                       # 数据层
    ├─ db.py                    # SQLAlchemy 引擎 + init_db()（关键）
    ├─ orm.py                   # ORM 定义（7 张表）
    ├─ store.py                 # 业务逻辑
    └─ schemas.py               # Pydantic 模型
  services/                     # 服务层
    ├─ log_service.py           # 日志记录
    ├─ runtime.py               # 翻译 Runner + Event Bus
    └─ event_bus.py             # SSE 事件管理
  api/                          # FastAPI 应用
    ├─ app.py                   # 应用入口 + lifespan（自动初始化数据库）
    └─ endpoints.py             # 路由定义
  config.py                     # 环境变量配置
  requirements.txt

AgenticArxivWeb/                 # Vue 3 前端
  src/
    ├─ components/              # 5 大页面组件
    │   ├─ ChatPanel.vue        # 对话 + Agent 思考链
    │   ├─ PapersPanel.vue      # 论文列表
    │   ├─ AssetsPanel.vue      # PDF 缓存 + 翻译进度
    │   ├─ LogsPanel.vue        # 执行日志
    │   ├─ SettingsPanel.vue    # 设置 (Agent 模式、主题、会话)
    │   └─ Sidebar.vue          # 导航
    ├─ stores/
    │   └─ appStore.ts          # Pinia 状态管理
    ├─ api/
    │   ├─ client.ts            # axios
    │   ├─ sse.ts               # SSE 管理
    │   └─ types.ts             # 类型定义
    ├─ App.vue                  # 主容器 (Sidebar + KeepAlive)
    └─ main.ts                  # 入口
```

## 开发指南

### 快速开发命令

```bash
# 全自动启动（一键）
make                # 启动前后端

# 分别启动（调试）
cd AgenticArxiv && uvicorn api.app:app --reload --port 8000
cd AgenticArxivWeb && npm run dev -- --port 5173
```

### 三种 Agent 模式的差异

| 模式 | 解析方式 | 执行方式 | 优点 | 缺点 | 最佳用途 |
|---|---|---|---|---|---|
| **regex** | 正则 | 进程内 | 快、简、稳 | JSON 格式敏感 | 推荐默认 |
| **mcp** | 正则 | JSON-RPC | 标准、隔离 | 延迟、复杂 | 团队开发 |
| **skill_cli** | 正则 | subprocess | 易理解 | 子进程开销 | 学习研究 |

**如何切换**？编辑 `.env`：
```env
AGENT_TYPE=regex  # 改为 mcp 或 skill_cli
```

### 常见修改

#### 增加新工具

1. `tools/my_tool.py`：实现工具函数
2. 调用 `register_tool()`
3. Agent 自动发现

#### 增加新 Agent 方案

1. 继承 `BaseAgent`
2. 实现 5 个抽象方法
3. 在 `api/endpoints.py` 注册

#### 修改 Prompt

编辑 `agents/prompt_templates.py` 中的 `REACT_PROMPT_TEMPLATE` 或 `skill_cli/SKILL.md`

#### 修改数据库表

编辑 `models/orm.py`，启动时自动创建

### 调试技巧

```bash
# 1. SQL 日志
# models/db.py
engine = create_engine(..., echo=True)

# 2. 查看 Agent 详细日志
# utils/logger.py
log.debug(f"详细信息: {content}")

# 3. FastAPI Swagger
http://localhost:8000/docs

# 4. 前端浏览器控制台
// SSE 连接状态、API 响应

# 5. 查看数据库
mysql -u arxiv -parxiv123 agentic_arxiv
SHOW TABLES;
SELECT * FROM chat_logs;
SELECT * FROM agent_steps;
```

## 部署清单

### 新 Linux 环境部署

```bash
# 1. MySQL 初始化（一次性）
mysql -u root -p << EOF
CREATE DATABASE IF NOT EXISTS agentic_arxiv DEFAULT CHARACTER SET utf8mb4;
CREATE USER IF NOT EXISTS 'arxiv'@'localhost' IDENTIFIED BY 'arxiv123';
GRANT ALL PRIVILEGES ON agentic_arxiv.* TO 'arxiv'@'localhost';
FLUSH PRIVILEGES;
EOF

# 2. 环境配置
cat > AgenticArxiv/.env << EOF
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-xxxxxxxx
MODEL=gpt-4-turbo
MYSQL_URI=mysql+pymysql://arxiv:arxiv123@localhost:3306/agentic_arxiv?charset=utf8mb4
AGENT_TYPE=regex
EOF

# 3. 依赖安装
make install

# 4. 启动（自动创建数据库表）
make

# ✅ 数据库表自动创建，无需额外步骤
```

### 生产部署（systemd）

```bash
sudo tee /etc/systemd/system/agentic-arxiv-api.service > /dev/null << 'EOF'
[Unit]
Description=Agentic Arxiv API
After=network.target mysql.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/agentic-arxiv
Environment="PATH=/opt/agentic-arxiv/.venv/bin"
ExecStart=/opt/agentic-arxiv/.venv/bin/python -m uvicorn AgenticArxiv.api.app:app \
  --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable agentic-arxiv-api
sudo systemctl start agentic-arxiv-api
```

## 常见坑

❌ **错误**：启动时表未创建 → 删除数据库，重新启动（`init_db()` 会自动创建）

❌ **错误**：LLM 生成 JSON 格式错误（True/False/None）→ ReAct prompt 中有严格约束

❌ **错误**：翻译进度卡住 → 检查后端日志，查看 SSE 连接是否正常

❌ **错误**：Agent 重复搜索论文 → 会话上下文注入工作正常，LLM 有时会忽略

## 文档位置

- 📄 主 README：`README.md`（整体架构、快速开始、部署）
- 📄 后端文档：`AgenticArxiv/readme.md`（三种 Agent 深入讲解）
- 📄 前端文档：`AgenticArxivWeb/readme.md`（Vue 3 页面详解）
- 📄 项目指南：此文件（Overview.md）

## 关键代码位置

| 功能 | 文件 | 行号 |
|---|---|---|
| Agent 执行循环 | `agents/base_agent.py` | 66-173 |
| 副作用处理 | `agents/base_agent.py` | 192-277 |
| 数据库初始化 | `api/app.py` | 19-25 |
| 数据库初始化 | `models/db.py` | 27-30 |
| ReAct 解析 | `agents/agent_engine.py` | 58-121 |
| MCP 工作流 | `mcp_protocol/mcp_agent.py` | 96-148 |
| Skill 解析 | `skill_cli/skill_agent.py` | 157-197 |
| 工具注册 | `tools/tool_registry.py` | - |
| 日志记录 | `services/log_service.py` | - |
| SSE 推送 | `services/event_bus.py` | - |
