# AgenticArxivWeb — Vue 3 + TypeScript 前端

现代化前端 SPA，与后端 FastAPI 通过 HTTP + SSE 实时通信。

## 快速开始

```bash
cd AgenticArxivWeb

# 开发模式（热重载）
npm run dev -- --host 0.0.0.0 --port 5173

# 生产构建
npm run build

# 预览构建结果
npm run preview
```

访问 http://localhost:5173

---

## 环境配置

### .env.development
```env
VITE_API_BASE=http://localhost:8000
```

### .env.production
```env
VITE_API_BASE=https://api.example.com
```

---

## 页面结构

### 5 大组件（KeepAlive 动态切换）

```
┌──────────────────────────────────────┐
│          Sidebar (60px 宽)           │
│  A: ChatPanel                        │
│  B: PapersPanel                      │
│  C: AssetsPanel                      │
│  D: LogsPanel                        │
│  E: SettingsPanel                    │
└──────────────────────────────────────┘
```

**布局**：
- 左侧 60px 固定宽度 Sidebar，包含 5 个导航按钮
- 主区域 KeepAlive 包裹的动态组件，保持各页面状态
- 主题切换：data-theme="light" | "dark"

### A. ChatPanel — 对话界面

**功能**：
- 发送消息给 Agent
- 实时显示 Agent 思考链（Thought/Action/Observation）
- SSE 事件流处理

**关键组件**：
```typescript
// src/components/ChatPanel.vue

// 1. 消息输入和发送
async function sendMessage() {
    const response = await client.post('/chat', {
        session_id: currentSessionId,
        message: inputText,
    });
    
    messagesRef.push({
        role: 'user',
        content: inputText,
        msg_id: response.msg_id,
    });
    
    // 2. 订阅 SSE（实时 Agent 步骤）
    subscribeToSSE(currentSessionId, (event) => {
        if (event.type === 'agent_step') {
            stepsRef.push(event.step);  // 实时显示思考步骤
            thinkingProgressRef.push({
                step_index: event.step.step_index,
                thought: event.step.thought,
                observation: event.step.observation,
            });
        }
        if (event.type === 'translate_progress') {
            translationProgressRef[event.task_id] = event.progress;  // 翻译进度
        }
    });
    
    // 3. 轮询获取 Agent 最终结果（可选，通常 SSE 已推送）
    const { history } = await client.get(`/logs/messages/${response.msg_id}/steps`);
    
    messagesRef.push({
        role: 'assistant',
        content: response.final_observation,
        msg_id: response.msg_id,
        steps: history,
    });
}

// 界面渲染
// - 消息列表（role 区分）
// - Thinking 展示框（展示当前 Agent 步骤）
// - 翻译进度条（实时更新）
```

**SSE 事件类型**：
```typescript
type SSEEvent = 
    | { type: 'agent_step', step: AgentStep }
    | { type: 'translate_progress', task_id: string, progress: number }
    | { type: 'task_created', task_id: string, paper_id: string };
```

### B. PapersPanel — 论文列表

**功能**：
- 显示当前 session 检索到的论文
- 支持批量下载/翻译
- 一键跳转到缓存管理

**数据源**：
```typescript
// GET /sessions/{sid}/papers
interface Paper {
    paper_id: string;
    title: string;
    authors: string[];
    summary: string;
    published: string;
    updated: string;
    pdf_url: string;
    primary_category: string;
    categories: string[];
    comment?: string;
    links: Link[];
    position: number;  // 搜索结果序号
}
```

**交互**：
```typescript
// 下载单篇
async function downloadPaper(paperId: string) {
    const response = await client.post('/pdf/download', {
        session_id: currentSessionId,
        paper_id: paperId,
    });
    notifySuccess(`下载成功: ${response.local_path}`);
    // 触发刷新缓存列表
}

// 翻译单篇
async function translatePaper(paperId: string) {
    const response = await client.post('/pdf/translate/async', {
        session_id: currentSessionId,
        paper_id: paperId,
        force: false,
        service: 'bing',
        threads: 4,
    });
    // SSE 推送翻译进度
}

// 批量操作
async function downloadAll() {
    for (const paper of papers) {
        await downloadPaper(paper.paper_id);
    }
}
```

### C. AssetsPanel — 缓存管理

**功能**：
- 列出已下载的 PDF（原文）
- 列出已翻译的 PDF
- 实时翻译任务进度
- 一键删除

**数据源**：
```typescript
// GET /pdf/assets
interface PDFAsset {
    paper_id: string;
    pdf_url: string;
    local_path: string;
    status: 'NOT_DOWNLOADED' | 'DOWNLOADING' | 'READY' | 'ERROR';
    size_bytes: number;
    sha256: string;
    downloaded_at: string;
    error?: string;
}

// GET /translate/assets
interface TranslateAsset {
    paper_id: string;
    input_pdf_path: string;
    output_mono_path: string;
    output_dual_path?: string;
    status: 'NOT_TRANSLATED' | 'TRANSLATING' | 'READY' | 'ERROR';
    service: string;
    threads: number;
    translated_at?: string;
    error?: string;
}

// GET /translate/tasks
interface TranslateTask {
    task_id: string;
    session_id: string;
    paper_id: string;
    status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
    progress: 0.0 ~ 100.0;
    input_pdf_path: string;
    output_pdf_path: string;
    error?: string;
    meta: object;
}
```

**操作**：
```typescript
// 查看 PDF（浏览器）
function viewPDF(paperId: string, type: 'raw' | 'translated') {
    const url = type === 'raw' 
        ? `/pdf/view/raw/${paperId}`
        : `/pdf/view/translated/${paperId}`;
    window.open(url, '_blank');
}

// 删除 PDF
async function deletePDF(paperId: string, type: 'raw' | 'translated') {
    if (type === 'raw') {
        await client.delete(`/pdf/assets/${paperId}`);
    } else {
        await client.delete(`/translate/assets/${paperId}`);
    }
    await refreshAssets();
}

// 下载文件
function downloadFile(filePath: string) {
    // 通过后端获取文件流或直接链接
    window.open(`/download/${filePath}`, '_blank');
}
```

**翻译进度**：
- SSE 实时更新 `translate_progress` 事件
- 进度条显示百分比和转速 (pdf2zh 原生)
- 完成后自动刷新列表

### D. LogsPanel — 日志查看

**功能**：
- 会话列表（消息计数、最后更新时间）
- 某会话的消息时间线
- 某消息的 Agent 思考链

**数据源**：
```typescript
// GET /logs/sessions
interface Session {
    session_id: string;
    message_count: number;
    last_message_at: string;
}

// GET /logs/sessions/{sid}/messages
interface Message {
    msg_id: string;
    role: 'user' | 'assistant';
    content: string;
    model: string;
    agent_type: string;
    created_at: string;
    step_count?: number;
}

// GET /logs/messages/{msg_id}/steps
interface AgentStep {
    step_index: number;
    thought: string;
    action_name: string;
    action_args: object;
    observation: string;
    llm_latency_ms: number;
    tool_latency_ms: number;
    created_at: string;
}
```

**交互**：
```typescript
// 1. 选择会话
async function selectSession(sessionId: string) {
    currentSessionId = sessionId;
    const messages = await client.get(`/logs/sessions/${sessionId}/messages`);
    messagesRef = messages;
}

// 2. 点击消息查看详情
async function viewMessageSteps(msgId: string) {
    const { steps } = await client.get(`/logs/messages/${msgId}/steps`);
    
    // 展示 Thought/Action/Observation 链路
    steps.forEach((step) => {
        console.log(`
            Step ${step.step_index}:
            Thought: ${step.thought}
            Action: ${step.action_name} ${JSON.stringify(step.action_args)}
            Observation: ${step.observation}
            Latency: LLM=${step.llm_latency_ms}ms Tool=${step.tool_latency_ms}ms
        `);
    });
}

// 3. 删除会话
async function deleteSession(sessionId: string) {
    await client.delete(`/sessions/${sessionId}`);
    await refreshSessions();
}
```

### E. SettingsPanel — 设置面板

**功能**：
- Session 管理（创建、切换、删除）
- Agent 模式选择
- SSE 连接状态
- 主题切换

**交互**：
```typescript
// Session 管理
const currentSessionId = ref('default');

function createNewSession(name: string) {
    currentSessionId.value = name;
    localStorage.setItem('current_session_id', name);
    notifySuccess(`新建会话: ${name}`);
}

// Agent 模式选择
const agentType = ref('regex');  // 'mcp' | 'skill_cli'

function changeAgentType(type: string) {
    agentType.value = type;
    localStorage.setItem('agent_type', type);
    notifyInfo(`已切换为 ${type} Agent，刷新页面生效`);
}

// SSE 连接状态
const sseConnected = ref(false);

function toggleSSE() {
    if (sseConnected.value) {
        sseClient.disconnect();
        sseConnected.value = false;
    } else {
        sseClient.connect(currentSessionId.value);
        sseConnected.value = true;
    }
}

// 主题切换
const theme = ref('light');  // 'dark'

function toggleTheme() {
    theme.value = theme.value === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', theme.value);
    localStorage.setItem('theme', theme.value);
}
```

---

## 状态管理 (Pinia)

### appStore

```typescript
// src/stores/appStore.ts

export const useAppStore = defineStore('app', {
    state: () => ({
        // Session
        currentSessionId: 'default',
        sessions: [],
        
        // Chat
        messages: [],
        thinking: null,
        
        // Papers
        papers: [],
        selectedPaperId: null,
        
        // Assets
        pdfAssets: [],
        translateAssets: [],
        translateTasks: [],
        
        // Settings
        agentType: 'regex',
        theme: 'light',
        sseConnected: false,
    }),
    
    actions: {
        // Session
        setCurrentSession(sessionId: string) {
            this.currentSessionId = sessionId;
        },
        
        // Chat
        pushMessage(role: string, content: string, msgId: string) {
            this.messages.push({
                role,
                content,
                msgId,
                timestamp: new Date(),
            });
        },
        
        setThinking(step: AgentStep) {
            this.thinking = step;
        },
        
        // 其他...
    },
    
    getters: {
        unreadMessageCount: (state) => {
            return state.messages.filter(m => !m.read).length;
        },
        
        translatingCount: (state) => {
            return state.translateTasks.filter(t => t.status === 'RUNNING').length;
        },
    },
});
```

---

## API 客户端

### 初始化

```typescript
// src/api/client.ts

import axios from 'axios';

const apiBaseURL = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export const httpClient = axios.create({
    baseURL: apiBaseURL,
    timeout: 30000,
});

// 错误处理
httpClient.interceptors.response.use(
    response => response.data,
    error => {
        const message = error.response?.data?.detail || error.message;
        console.error(`API Error: ${message}`);
        throw new Error(message);
    }
);
```

### SSE 管理

```typescript
// src/api/sse.ts

export class SSEClient {
    private eventSource: EventSource | null = null;
    private sessionId: string = '';
    private callbacks: Map<string, Function[]> = new Map();
    
    connect(sessionId: string) {
        this.sessionId = sessionId;
        const url = `${import.meta.env.VITE_API_BASE}/events?session_id=${sessionId}`;
        
        this.eventSource = new EventSource(url);
        
        this.eventSource.addEventListener('agent_step', (event) => {
            const step = JSON.parse(event.data);
            this.emit('agent_step', step);
        });
        
        this.eventSource.addEventListener('translate_progress', (event) => {
            const progress = JSON.parse(event.data);
            this.emit('translate_progress', progress);
        });
        
        this.eventSource.addEventListener('task_created', (event) => {
            const task = JSON.parse(event.data);
            this.emit('task_created', task);
        });
    }
    
    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }
    
    on(eventType: string, callback: Function) {
        if (!this.callbacks.has(eventType)) {
            this.callbacks.set(eventType, []);
        }
        this.callbacks.get(eventType)?.push(callback);
    }
    
    private emit(eventType: string, data: any) {
        this.callbacks.get(eventType)?.forEach(cb => cb(data));
    }
}

export const sseClient = new SSEClient();
```

---

## 样式 & 主题

### 主题系统

```typescript
// src/styles/theme.css

:root[data-theme="light"] {
    --bg-primary: #ffffff;
    --bg-secondary: #f5f5f5;
    --text-primary: #000000;
    --text-secondary: #666666;
    --border-color: #dddddd;
    --accent-color: #0066cc;
}

:root[data-theme="dark"] {
    --bg-primary: #1a1a1a;
    --bg-secondary: #2a2a2a;
    --text-primary: #ffffff;
    --text-secondary: #cccccc;
    --border-color: #444444;
    --accent-color: #6699ff;
}

body {
    background-color: var(--bg-primary);
    color: var(--text-primary);
}
```

### Sidebar 样式

```css
.sidebar {
    width: 60px;
    height: 100vh;
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-top: 20px;
}

.sidebar-item {
    width: 50px;
    height: 50px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 10px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.sidebar-item.active {
    background-color: var(--accent-color);
    color: white;
}
```

---

## 文件结构

```
AgenticArxivWeb/
├── src/
│   ├── components/
│   │   ├── ChatPanel.vue        # 对话
│   │   ├── PapersPanel.vue      # 论文列表
│   │   ├── AssetsPanel.vue      # 缓存管理
│   │   ├── LogsPanel.vue        # 日志
│   │   ├── SettingsPanel.vue    # 设置
│   │   └── Sidebar.vue          # 导航
│   ├── stores/
│   │   └── appStore.ts          # Pinia 状态
│   ├── api/
│   │   ├── client.ts            # axios
│   │   ├── sse.ts               # SSE 管理
│   │   └── types.ts             # 类型定义
│   ├── styles/
│   │   ├── main.css             # 全局样式
│   │   └── theme.css            # 主题
│   ├── App.vue                  # 主容器
│   └── main.ts                  # 入口
├── vite.config.ts               # Vite 配置
├── tsconfig.json                # TS 配置
├── package.json
└── .env.development             # 开发环境变量
```

---

## 开发指南

### 本地运行

```bash
# 安装依赖
npm install

# 开发模式（热重载）
npm run dev

# 查看 Storybook（可选）
npm run storybook
```

### 构建生产版本

```bash
npm run build

# 预览构建结果
npm run preview
```

### 代码规范

```bash
# Lint
npm run lint

# Format
npm run format
```

---

## 常见问题

### Q: 消息为什么没有实时更新？

检查 SSE 连接：
```typescript
sseClient.connect(sessionId);
// 应该看到浏览器控制台的 EventSource 连接建立
```

### Q: 翻译进度卡住了怎么办？

1. 检查后端日志是否有错误
2. 刷新前端页面，重新连接 SSE
3. 在 AssetsPanel 查看任务状态

### Q: 如何调整 UI 布局？

修改 `App.vue` 中的 flex 布局：
```typescript
<div class="container">
    <Sidebar />
    <div class="main">
        <KeepAlive>
            <component :is="currentComponent" />
        </KeepAlive>
    </div>
</div>

<style>
.container {
    display: flex;
    height: 100vh;
}
.main {
    flex: 1;
    overflow: auto;
}
</style>
```

---

## 扩展开发

### 添加新页面

1. 创建 `src/components/MyPanel.vue`
2. 在 `Sidebar.vue` 添加导航按钮
3. 在 `App.vue` 注册组件

### 集成第三方库

推荐库：
- UI 组件：Vue DevUI、Element Plus、Ant Design Vue
- 图表：ECharts、Recharts
- 代码编辑：Monaco Editor

---

## 部署

### Docker 部署

```dockerfile
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
```

### Nginx 配置

```nginx
server {
    listen 80;
    server_name _;

    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://backend:8000;
    }
}
```

---

## 许可证

MIT
