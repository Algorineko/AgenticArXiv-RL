<template>
  <section class="chat">
    <header class="chat-header">
      <div class="header-left">
        <span class="title">对话</span>
        <span
          v-if="!editingSession"
          class="session-id clickable"
          title="点击编辑 Session ID"
          @click="startEditSession"
        >{{ store.sessionId }}</span>
        <input
          v-else
          ref="sessionInput"
          class="session-id-input"
          v-model="sessionDraft"
          @keydown.enter="applySession"
          @blur="applySession"
        />
      </div>
      <div class="header-actions">
        <button class="btn ghost" @click="store.clearChat()">清空</button>
        <button class="btn primary" @click="store.newSession()">新会话</button>
      </div>
      <p v-if="store.lastError" class="error">{{ store.lastError }}</p>
    </header>

    <div class="messages" ref="msgBox">
      <div v-if="store.messages.length === 0 && !store.isThinking" class="empty">
        示例: <code>获取最近7天内AI(cs.AI)方向论文，最多5篇</code>
      </div>

      <div v-for="(m, idx) in store.messages" :key="idx" class="msg" :class="m.role">
        <!-- assistant: expandable thinking history -->
        <div v-if="m.role === 'assistant' && m.history && m.history.length > 0" class="thinking-section">
          <button class="btn ghost think-toggle" @click="toggleHistory(idx)">
            {{ expandedHistoryIds.has(idx) ? '收起思考' : `思考过程 (${m.history.length}步)` }}
          </button>
          <div v-if="expandedHistoryIds.has(idx)" class="think-steps">
            <div v-for="(h, si) in m.history" :key="si" class="think-step">
              <div class="think-step-head">
                <span class="step-idx">Step {{ si + 1 }}</span>
                <span v-if="h.action && h.action !== 'FINISH'" class="action-tag">{{ extractToolName(h.action) }}</span>
              </div>
              <div v-if="h.thought" class="think-item">
                <span class="think-label">Thought</span>
                <pre class="think-text">{{ h.thought }}</pre>
              </div>
              <div v-if="h.observation" class="think-item">
                <span class="think-label">Observation</span>
                <pre class="think-text">{{ truncate(h.observation, 500) }}</pre>
              </div>
            </div>
          </div>
        </div>

        <div class="meta">
          <span class="role">{{ m.role === "user" ? "你" : "Agent" }}</span>
          <span v-if="m.role === 'assistant' && m.agentType" class="agent-type-tag">{{ agentLabel(m.agentType) }}</span>
          <span class="time">{{ fmt(m.ts) }}</span>
        </div>
        <pre class="content">{{ m.content }}</pre>

        <!-- inline translation progress -->
        <div v-if="m.taskId && getTask(m.taskId)" class="task-progress">
          <div class="task-progress-header">
            <span class="task-status-pill" :class="taskStatusClass(getTask(m.taskId)!.status)">
              {{ getTask(m.taskId)!.status }}
            </span>
            <span class="task-pct">{{ taskPct(getTask(m.taskId)!.progress) }}%</span>
            <span class="task-detail" v-if="store.taskDetailLine[m.taskId]">{{ store.taskDetailLine[m.taskId] }}</span>
          </div>
          <div class="pbar" role="progressbar"
               :aria-valuenow="taskPct(getTask(m.taskId)!.progress)"
               aria-valuemin="0" aria-valuemax="100">
            <div class="pbar-fill" :style="{ width: taskPct(getTask(m.taskId)!.progress) + '%' }"></div>
          </div>
        </div>
      </div>

      <!-- streaming thinking card (while agent is processing) -->
      <div v-if="store.isThinking" class="msg assistant thinking-card">
        <div class="meta">
          <span class="role">Agent</span>
          <span class="thinking-indicator">思考中...</span>
        </div>
        <div v-if="store.thinkingSteps.length > 0">
          <!-- latest step always visible -->
          <div class="think-latest">
            <span class="step-idx">Step {{ store.thinkingSteps.length }}</span>
            <span v-if="latestStep?.action_name" class="action-tag">{{ latestStep.action_name }}</span>
          </div>
          <pre class="think-text latest-thought">{{ latestStep?.thought || '...' }}</pre>

          <!-- expand all steps -->
          <button v-if="store.thinkingSteps.length > 1" class="btn ghost think-toggle"
                  @click="showAllThinking = !showAllThinking">
            {{ showAllThinking ? '收起' : `展开全部 (${store.thinkingSteps.length}步)` }}
          </button>
          <div v-if="showAllThinking" class="think-steps">
            <div v-for="(s, si) in store.thinkingSteps.slice(0, -1)" :key="si" class="think-step">
              <div class="think-step-head">
                <span class="step-idx">Step {{ si + 1 }}</span>
                <span v-if="s.action_name" class="action-tag">{{ s.action_name }}</span>
              </div>
              <pre v-if="s.thought" class="think-text">{{ s.thought }}</pre>
            </div>
          </div>
        </div>
        <div v-else class="think-text">等待 Agent 响应...</div>
      </div>
      <!-- Translate progress during thinking -->
      <div v-if="thinkingTask" class="task-progress">
        <div class="task-progress-header">
          <span class="task-status-pill" :class="taskStatusClass(thinkingTask.status)">
            {{ thinkingTask.status }}
          </span>
          <span class="task-pct">{{ taskPct(thinkingTask.progress) }}%</span>
          <span class="task-detail" v-if="thinkingTask.task_id && store.taskDetailLine[thinkingTask.task_id]">{{ store.taskDetailLine[thinkingTask.task_id] }}</span>
        </div>
        <div class="pbar" role="progressbar"
             :aria-valuenow="taskPct(thinkingTask.progress)"
             aria-valuemin="0" aria-valuemax="100">
          <div class="pbar-fill" :style="{ width: taskPct(thinkingTask.progress) + '%' }"></div>
        </div>
      </div>
    </div>

    <footer class="composer">
      <textarea
        class="textarea"
        v-model="draft"
        :disabled="store.loading"
        placeholder="输入一句话，让 agent 自己决定调用工具..."
        @keydown.enter.exact.prevent="send"
      />

      <div class="actions">
        <button class="btn" type="button" @click="quick('获取最近7天内AI(cs.AI)方向论文，最多5篇')">
          快捷: 拉取AI 5篇
        </button>

        <button class="btn primary" type="button" @click="send" :disabled="store.loading || !draft.trim()">
          {{ store.loading ? "发送中..." : "发送" }}
        </button>
      </div>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { nextTick, onMounted, ref, watch, computed, reactive } from "vue";
import { useAppStore } from "@/stores/appStore";
import type { TranslateTask } from "@/api/types";

const store = useAppStore();
const draft = ref("");
const msgBox = ref<HTMLDivElement | null>(null);
const showAllThinking = ref(false);
const expandedHistoryIds = reactive(new Set<number>());
const editingSession = ref(false);
const sessionDraft = ref("");
const sessionInput = ref<HTMLInputElement | null>(null);

function startEditSession() {
  sessionDraft.value = store.sessionId;
  editingSession.value = true;
  nextTick(() => sessionInput.value?.focus());
}

function applySession() {
  const val = (sessionDraft.value || "").trim();
  editingSession.value = false;
  if (!val || val === store.sessionId) return;
  store.setSessionId(val);
  store.refreshSnapshot();
}

function fmt(ts: number) {
  if (!ts) return "-";
  return new Date(ts).toLocaleString();
}

function toggleHistory(idx: number) {
  if (expandedHistoryIds.has(idx)) expandedHistoryIds.delete(idx);
  else expandedHistoryIds.add(idx);
}

function extractToolName(action: string): string {
  try {
    const parsed = JSON.parse(action);
    return parsed.name || action;
  } catch {
    return action;
  }
}

const agentLabelMap: Record<string, string> = {
  regex: "ReAct",
  mcp: "MCP",
  skill_cli: "Skill/CLI",
};
function agentLabel(t: string): string {
  return agentLabelMap[t] || t;
}

function truncate(s: string, max: number): string {
  if (!s || s.length <= max) return s;
  return s.slice(0, max) + "...";
}

function getTask(taskId: string): TranslateTask | undefined {
  return store.tasks.find((t) => t.task_id === taskId);
}

function taskPct(p: number | undefined): number {
  const v = Number(p || 0);
  if (!Number.isFinite(v)) return 0;
  return Math.round(Math.max(0, Math.min(1, v)) * 100);
}

function taskStatusClass(s: string): string {
  const x = (s || "").toUpperCase();
  if (x.includes("SUCC")) return "ok";
  if (x.includes("FAIL")) return "bad";
  if (x.includes("RUN")) return "warn";
  return "neutral";
}

const latestStep = computed(() => {
  const steps = store.thinkingSteps;
  return steps.length > 0 ? steps[steps.length - 1] : null;
});

const thinkingTask = computed(() => {
  const tid = store.thinkingTaskId;
  if (!tid) return null;
  return store.tasks.find((t) => t.task_id === tid) || null;
});

async function send() {
  const text = draft.value;
  draft.value = "";
  showAllThinking.value = false;
  await store.sendChat(text);
  await nextTick();
  msgBox.value?.scrollTo({ top: msgBox.value.scrollHeight, behavior: "smooth" });
}

function quick(text: string) {
  draft.value = text;
}

onMounted(() => {
  store.ensureSse();
  store.refreshSnapshot();
});

watch(
  () => [store.messages.length, store.thinkingSteps.length],
  async () => {
    await nextTick();
    msgBox.value?.scrollTo({ top: msgBox.value.scrollHeight });
  }
);
</script>

<style scoped>
.chat {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 36px);
  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--card-bg);
  overflow: hidden;
}

.chat-header {
  padding: 12px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  flex-shrink: 0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.title { font-weight: 700; }
.session-id {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
  color: var(--muted);
  padding: 2px 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg2);
}
.session-id.clickable {
  cursor: pointer;
  transition: border-color 0.15s;
}
.session-id.clickable:hover {
  border-color: var(--accent);
  color: var(--fg);
}
.session-id-input {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
  color: var(--fg);
  padding: 2px 8px;
  border: 1px solid var(--accent);
  border-radius: 6px;
  background: var(--bg2);
  outline: none;
  width: 140px;
}

.header-actions { margin-left: auto; display: flex; gap: 6px; }

.messages { flex: 1; overflow: auto; padding: 10px; }
.empty { color: var(--muted); padding: 12px; }

.msg {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px;
  margin-bottom: 10px;
  background: var(--bg2);
}
.msg.user { border-left: 4px solid #4b9; }
.msg.assistant { border-left: 4px solid var(--accent); }

.meta {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 6px;
}
.meta .time {
  margin-left: auto;
}

.agent-type-tag {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 4px;
  background: rgba(89,153,255,0.12);
  color: var(--accent);
  font-weight: 600;
}

.content {
  white-space: pre-wrap;
  margin: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 13px;
}

/* Thinking section in assistant messages */
.thinking-section {
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

.think-toggle {
  font-size: 11px;
  height: 24px;
  padding: 0 8px;
  color: var(--accent);
}

.think-steps {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 6px;
}

.think-step {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 6px 8px;
  background: var(--card-bg);
}

.think-step-head {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
}

.step-idx { font-weight: 700; font-size: 11px; }
.action-tag {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 4px;
  background: rgba(89,153,255,0.12);
  color: var(--accent);
}

.think-item { margin-top: 4px; }
.think-label { color: var(--muted); font-size: 10px; display: block; margin-bottom: 2px; }
.think-text {
  margin: 0;
  white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
  max-height: 120px;
  overflow: auto;
  padding: 4px 6px;
  border-radius: 6px;
  background: var(--card-bg);
  border: 1px solid var(--border);
}

/* Streaming thinking card */
.thinking-card {
  border-left-color: rgba(255,200,0,0.6);
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.8; }
}

.thinking-indicator {
  color: rgba(255,200,0,0.8);
  font-size: 12px;
}

.think-latest {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
}

.latest-thought {
  border: none;
  background: transparent;
  padding: 0;
  max-height: 80px;
  color: var(--fg);
}

/* Inline translation progress */
.task-progress {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--border);
}

.task-progress-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.task-status-pill {
  display: inline-flex;
  align-items: center;
  height: 18px;
  padding: 0 6px;
  border-radius: 999px;
  border: 1px solid var(--border);
  font-size: 11px;
}
.task-status-pill.ok { border-color: rgba(0,255,153,.35); }
.task-status-pill.bad { border-color: rgba(255,107,107,.35); }
.task-status-pill.warn { border-color: rgba(255,200,0,.35); }

.task-pct {
  font-size: 11px;
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.task-detail {
  font-size: 10px;
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  margin-left: auto;
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pbar {
  height: 6px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--bg2);
  overflow: hidden;
}

.pbar-fill {
  height: 100%;
  border-radius: 999px;
  background: rgba(89,153,255,0.55);
  transition: width 180ms ease;
}

/* Composer */
.composer {
  border-top: 1px solid var(--border);
  padding: 10px 12px;
  flex-shrink: 0;
}

.textarea {
  width: 100%;
  min-height: 80px;
  padding: 10px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--bg2);
  color: var(--fg);
  resize: vertical;
}

.actions {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 10px;
}

.error {
  color: #ff6b6b;
  font-size: 12px;
  margin: 0;
  width: 100%;
}
</style>
