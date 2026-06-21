<template>
  <section class="logs-page">
    <div class="sessions-col">
      <div class="col-header">
        <div class="col-title">会话列表</div>
        <button class="btn" @click="fetchSessions">刷新</button>
      </div>
      <div class="session-list">
        <div
          v-for="s in sessions"
          :key="s.session_id"
          class="session-item"
          :class="{ active: selectedSession === s.session_id }"
          @click="selectSession(s.session_id)"
        >
          <div class="sid">{{ s.session_id }}</div>
          <div class="meta">{{ s.message_count }} 条 · {{ fmtDate(s.last_active_at) }}</div>
        </div>
        <div v-if="sessions.length === 0" class="empty">暂无会话日志</div>
      </div>
    </div>

    <div class="messages-col">
      <div class="col-header">
        <div class="col-title">
          {{ selectedSession ? `会话: ${selectedSession}` : "选择一个会话查看" }}
        </div>
      </div>
      <div class="message-list">
        <div v-for="m in messages" :key="m.msg_id" class="msg-card" :class="m.role">
          <div class="msg-header">
            <span class="role-badge" :class="m.role">{{ m.role === "user" ? "用户" : "Agent" }}</span>
            <span v-if="m.role === 'assistant' && m.agent_type" class="agent-type-tag">{{ agentLabel(m.agent_type) }}</span>
            <span class="time">{{ fmtDate(m.created_at) }}</span>
            <span v-if="m.model" class="model">{{ m.model }}</span>
            <button
              v-if="m.role === 'assistant'"
              class="btn ghost expand-btn"
              @click="toggleSteps(baseMsgId(m.msg_id))"
            >
              {{ expandedMsgIds.has(baseMsgId(m.msg_id)) ? "收起步骤" : "展开步骤" }}
            </button>
          </div>

          <div v-if="m.role === 'assistant' && expandedMsgIds.has(baseMsgId(m.msg_id))" class="steps-area">
            <div v-if="stepsLoading.has(baseMsgId(m.msg_id))" class="loading">加载中...</div>
            <div v-else-if="(stepsMap[baseMsgId(m.msg_id)] || []).length === 0" class="empty">无步骤记录</div>
            <div v-else class="steps">
              <div v-for="step in stepsMap[baseMsgId(m.msg_id)]" :key="step.step_index" class="step">
                <div class="step-header">
                  <span class="step-idx">Step {{ step.step_index + 1 }}</span>
                  <span v-if="step.action_name" class="action-name">{{ step.action_name }}</span>
                  <span class="latency" v-if="step.llm_latency_ms">LLM: {{ step.llm_latency_ms }}ms</span>
                  <span class="latency" v-if="step.tool_latency_ms">Tool: {{ step.tool_latency_ms }}ms</span>
                </div>
                <div v-if="step.thought" class="step-section">
                  <div class="step-label">Thought</div>
                  <pre class="step-text">{{ step.thought }}</pre>
                </div>
                <div v-if="step.action_args && step.action_args !== '{}'" class="step-section">
                  <div class="step-label">Args</div>
                  <pre class="step-text">{{ step.action_args }}</pre>
                </div>
                <div v-if="step.observation" class="step-section">
                  <div class="step-label">Observation</div>
                  <pre class="step-text">{{ step.observation }}</pre>
                </div>
              </div>
            </div>
          </div>
          <pre class="msg-content">{{ m.content }}</pre>
        </div>
        <div v-if="selectedSession && messages.length === 0" class="empty">该会话暂无消息</div>
        <div v-if="!selectedSession" class="empty">请在左侧选择一个会话</div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from "vue";
import { api } from "@/api/client";
import type { LogSessionSummary, ChatLogItem, AgentStepItem } from "@/api/types";

const sessions = ref<LogSessionSummary[]>([]);
const selectedSession = ref("");
const messages = ref<ChatLogItem[]>([]);
const stepsMap = reactive<Record<string, AgentStepItem[]>>({});
const expandedMsgIds = reactive(new Set<string>());
const stepsLoading = reactive(new Set<string>());

const agentLabelMap: Record<string, string> = {
  regex: "ReAct",
  mcp: "MCP",
  skill_cli: "Skill/CLI",
};
function agentLabel(t: string): string {
  return agentLabelMap[t] || t;
}

function fmtDate(d: string | null) {
  if (!d) return "-";
  return new Date(d).toLocaleString();
}

function baseMsgId(msgId: string): string {
  return msgId.endsWith("_reply") ? msgId.slice(0, -6) : msgId;
}

async function fetchSessions() {
  try {
    const res = await api.get<{ sessions: LogSessionSummary[] }>("/logs/sessions");
    sessions.value = res.data?.sessions || [];
  } catch { /* ignore */ }
}

async function selectSession(sid: string) {
  selectedSession.value = sid;
  try {
    const res = await api.get<{ messages: ChatLogItem[] }>(`/logs/sessions/${encodeURIComponent(sid)}/messages`);
    messages.value = res.data?.messages || [];
  } catch {
    messages.value = [];
  }
}

async function toggleSteps(msgId: string) {
  if (expandedMsgIds.has(msgId)) {
    expandedMsgIds.delete(msgId);
    return;
  }
  expandedMsgIds.add(msgId);
  if (stepsMap[msgId]) return; // already loaded

  stepsLoading.add(msgId);
  try {
    const res = await api.get<{ steps: AgentStepItem[] }>(`/logs/messages/${encodeURIComponent(msgId)}/steps`);
    stepsMap[msgId] = res.data?.steps || [];
  } catch {
    stepsMap[msgId] = [];
  } finally {
    stepsLoading.delete(msgId);
  }
}

onMounted(fetchSessions);
</script>

<style scoped>
.logs-page {
  display: grid;
  grid-template-columns: 240px 1fr;
  gap: 14px;
  height: calc(100vh - 36px);
}

.sessions-col, .messages-col {
  display: flex;
  flex-direction: column;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: rgba(255,255,255,0.02);
  overflow: hidden;
}

.col-header {
  padding: 12px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-shrink: 0;
}

.col-title { font-weight: 700; }

.session-list, .message-list {
  flex: 1;
  overflow: auto;
  padding: 8px;
}

.session-item {
  padding: 10px;
  border-radius: 10px;
  cursor: pointer;
  margin-bottom: 4px;
}
.session-item:hover { background: rgba(255,255,255,0.04); }
.session-item.active { background: rgba(89,153,255,0.12); }
.sid { font-weight: 600; font-size: 13px; }
.meta { color: var(--muted); font-size: 11px; margin-top: 2px; }

.msg-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px;
  margin-bottom: 10px;
  background: var(--bg2);
}
.msg-card.user { border-left: 4px solid #4b9; }
.msg-card.assistant { border-left: 4px solid #59f; }

.msg-header {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 6px;
}

.role-badge {
  font-size: 11px;
  padding: 1px 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
}
.role-badge.user { border-color: rgba(0,255,153,.35); }
.role-badge.assistant { border-color: rgba(89,153,255,.35); }

.time { color: var(--muted); font-size: 11px; }
.model { color: var(--muted); font-size: 11px; }

.agent-type-tag {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 4px;
  background: rgba(89,153,255,0.12);
  color: var(--accent, #59f);
  font-weight: 600;
}

.expand-btn { font-size: 11px; height: 24px; padding: 0 8px; }

.msg-content {
  white-space: pre-wrap;
  margin: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 13px;
  max-height: 200px;
  overflow: auto;
}

.steps-area {
  margin-top: 10px;
  border-top: 1px solid var(--border);
  padding-top: 10px;
}

.steps { display: flex; flex-direction: column; gap: 8px; }

.step {
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px;
  padding: 8px;
  background: rgba(255,255,255,0.015);
}

.step-header {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 6px;
}

.step-idx { font-weight: 700; font-size: 12px; }
.action-name {
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 6px;
  background: rgba(89,153,255,0.12);
  color: #59f;
}
.latency { color: var(--muted); font-size: 11px; }

.step-section { margin-top: 6px; }
.step-label { color: var(--muted); font-size: 11px; margin-bottom: 2px; }
.step-text {
  margin: 0;
  white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
  max-height: 150px;
  overflow: auto;
  padding: 6px;
  border-radius: 8px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.05);
}

.empty { color: var(--muted); padding: 16px; text-align: center; }
.loading { color: var(--muted); padding: 8px; text-align: center; }

@media (max-width: 768px) {
  .logs-page { grid-template-columns: 1fr; height: auto; }
}
</style>
