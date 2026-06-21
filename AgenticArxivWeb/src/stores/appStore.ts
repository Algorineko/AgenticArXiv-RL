// src/stores/appStore.ts
import { defineStore } from "pinia";
import { api, getApiBase } from "@/api/client";
import { openEventsSse } from "@/api/sse";
import type {
  AgentType,
  ChatResponse,
  Paper,
  PdfAsset,
  TranslateAsset,
  Role,
  ReactStep,
  TranslateTask,
  SseEnvelope,
  DeleteAssetResponse,
} from "@/api/types";

export interface ChatMessage {
  role: Role;
  content: string;
  ts: number;
  taskId?: string;
  history?: ReactStep[];
  agentType?: AgentType;
}

function nowTs() {
  return Date.now();
}

let _es: EventSource | null = null;
let _reconnectTimer: number | null = null;
let _reconnectBackoffMs = 1000;

function clearReconnectTimer() {
  if (_reconnectTimer != null) {
    window.clearTimeout(_reconnectTimer);
    _reconnectTimer = null;
  }
}

export const useAppStore = defineStore("app", {
  state: () => ({
    apiBase: getApiBase(),
    sessionId: (localStorage.getItem("session_id") || "demo1") as string,

    loading: false,
    lastError: "" as string,

    messages: [] as ChatMessage[],

    papers: [] as Paper[],
    pdfAssets: [] as PdfAsset[],
    translateAssets: [] as TranslateAsset[],

    tasks: [] as TranslateTask[],
    lastHistory: [] as ReactStep[],

    preferAgent: true,
    agentType: (localStorage.getItem("agent_type") || "regex") as AgentType,

    thinkingSteps: [] as any[],
    isThinking: false,

    // Track task_ids created via SSE during the current sendChat() call
    _newTaskIds: new Set<string>() as Set<string>,
    // TaskId created during thinking phase (for showing progress in thinking card)
    thinkingTaskId: "" as string,
    // Latest detail line per task (e.g. " 83% 20/24 [01:16<00:17]")
    taskDetailLine: {} as Record<string, string>,

    sseStatus: "idle" as "idle" | "connecting" | "connected" | "error",
    sseLastEvent: "" as string,
    sseLastEventTs: 0 as number,
  }),

  getters: {
    pdfMap(state) {
      const m = new Map<string, PdfAsset>();
      (state.pdfAssets || []).forEach((a) => m.set(a.paper_id, a));
      return m;
    },
    translateMap(state) {
      const m = new Map<string, TranslateAsset>();
      (state.translateAssets || []).forEach((a) => m.set(a.paper_id, a));
      return m;
    },
  },

  actions: {
    // -------- URL helpers --------
    _joinApi(path: string) {
      const b = (this.apiBase || "").replace(/\/+$/, "");
      const p = path.startsWith("/") ? path : `/${path}`;
      return `${b}${p}`;
    },

    getRawPdfViewUrl(paperId: string) {
      const pid = encodeURIComponent((paperId || "").trim());
      const sid = encodeURIComponent(this.sessionId || "default");
      return this._joinApi(`/pdf/view/raw/${pid}?session_id=${sid}`);
    },

    getTranslatedPdfViewUrl(paperId: string, variant: "mono" | "dual" = "mono") {
      const pid = encodeURIComponent((paperId || "").trim());
      const sid = encodeURIComponent(this.sessionId || "default");
      const v = encodeURIComponent(variant);
      return this._joinApi(`/pdf/view/translated/${pid}?variant=${v}&session_id=${sid}`);
    },

    _openNewTab(url: string) {
      if (typeof window === "undefined") return;
      window.open(url, "_blank", "noopener,noreferrer");
    },

    openRawPdf(paperId: string) {
      this._openNewTab(this.getRawPdfViewUrl(paperId));
    },

    openTranslatedPdf(paperId: string, variant: "mono" | "dual" = "mono") {
      this._openNewTab(this.getTranslatedPdfViewUrl(paperId, variant));
    },

    // -------- original logic --------
    setSessionId(id: string) {
      const next = (id || "default").trim() || "default";
      if (next === this.sessionId) return;

      this.sessionId = next;
      localStorage.setItem("session_id", this.sessionId);

      this.tasks = [];
      this.closeSse();
      this.ensureSse();
    },

    newSession() {
      const id = "session-" + crypto.randomUUID().slice(0, 8);
      this.clearChat();
      this.papers = [];
      this.pdfAssets = [];
      this.translateAssets = [];
      this.tasks = [];
      this.setSessionId(id);
      this.refreshSnapshot();
    },

    pushUser(content: string) {
      this.messages.push({ role: "user", content, ts: nowTs() });
    },
    pushAssistant(content: string, opts?: { taskId?: string; history?: ReactStep[]; agentType?: AgentType }) {
      this.messages.push({ role: "assistant", content, ts: nowTs(), ...opts });
    },

    ensureSse() {
      if (typeof window === "undefined") return;
      const sid = this.sessionId || "default";
      if (_es) return;

      clearReconnectTimer();
      this.sseStatus = "connecting";

      _es = openEventsSse(this.apiBase, sid, {
        onOpen: () => {
          this.sseStatus = "connected";
          this.sseLastEvent = "open";
          this.sseLastEventTs = nowTs();
          _reconnectBackoffMs = 1000;
        },
        onError: () => {
          this.sseStatus = "error";
          this.sseLastEvent = "error";
          this.sseLastEventTs = nowTs();

          const readyState = _es?.readyState;
          if (readyState === 2) {
            this.closeSse();
            this._scheduleReconnect();
          }
        },
        onEvent: (evt) => this._handleSseEvent(evt),
      });
    },

    closeSse() {
      clearReconnectTimer();
      if (_es) {
        try {
          _es.close();
        } catch { }
      }
      _es = null;
      this.sseStatus = "idle";
    },

    _scheduleReconnect() {
      if (typeof window === "undefined") return;
      if (_reconnectTimer != null) return;

      const delay = Math.min(_reconnectBackoffMs, 15000);
      _reconnectBackoffMs = Math.min(_reconnectBackoffMs * 2, 15000);

      _reconnectTimer = window.setTimeout(() => {
        _reconnectTimer = null;
        this.closeSse();
        this.ensureSse();
      }, delay);
    },

    _upsertTask(t: TranslateTask) {
      const id = t?.task_id;
      if (!id) return;

      const idx = this.tasks.findIndex((x) => x.task_id === id);
      if (idx >= 0) this.tasks[idx] = { ...this.tasks[idx], ...t };
      else this.tasks.unshift(t);

      if (this.tasks.length > 50) this.tasks.length = 50;
    },

    _removePdfAssetLocal(paperId: string) {
      this.pdfAssets = (this.pdfAssets || []).filter((x) => x.paper_id !== paperId);
    },
    _removeTranslateAssetLocal(paperId: string) {
      this.translateAssets = (this.translateAssets || []).filter((x) => x.paper_id !== paperId);
    },

    async _handleSseEvent(evt: SseEnvelope) {
      if (!evt || typeof evt !== "object") return;

      this.sseLastEvent = evt.type || "message";
      this.sseLastEventTs = nowTs();

      if (evt.type === "connected") {
        this.sseStatus = "connected";
        return;
      }

      if (evt.type === "agent_step" && evt.step) {
        this.thinkingSteps.push(evt.step);
        return;
      }

      if (evt.type === "asset_deleted" && evt.paper_id) {
        const kind = String(evt.kind || "").toLowerCase();
        if (kind === "pdf") this._removePdfAssetLocal(evt.paper_id);
        if (kind === "translate") this._removeTranslateAssetLocal(evt.paper_id);
        return;
      }

      if (evt.kind === "translate" && evt.task) {
        this._upsertTask(evt.task);
        // Store detail line for debugging (e.g. " 83% 20/24 [01:16<00:17]")
        const line = evt.detail?.line;
        if (line && evt.task.task_id) {
          this.taskDetailLine[evt.task.task_id] = String(line).trim();
        }
      }

      // Track new tasks created during sendChat (for Issue 1 & 2)
      if (evt.type === "task_created" && evt.kind === "translate" && evt.task?.task_id) {
        if (this.loading) {
          this._newTaskIds.add(evt.task.task_id);
          this.thinkingTaskId = evt.task.task_id;
        }
      }

      if ((evt.type === "task_created" || evt.type === "task_started") && evt.kind === "translate") {
        await Promise.allSettled([this.fetchTranslateAssets(), this.fetchPdfAssets()]);
      }

      if (evt.type === "task_succeeded" && evt.kind === "translate") {
        await Promise.allSettled([this.fetchTranslateAssets(), this.fetchPdfAssets()]);
      }

      if (evt.type === "task_failed" && evt.kind === "translate") {
        const msg = evt.task?.error || evt.message || "翻译任务失败";
        this.lastError = msg;
      }
    },

    async refreshSnapshot() {
      try {
        this.ensureSse();

        const [papersRes, pdfRes, trRes] = await Promise.all([
          api.get(`/sessions/${encodeURIComponent(this.sessionId)}/papers`),
          api.get(`/pdf/assets`),
          api.get(`/translate/assets`),
        ]);

        this.papers = papersRes.data?.papers || [];
        this.pdfAssets = pdfRes.data?.assets || [];
        this.translateAssets = trRes.data?.assets || [];
      } catch (e: any) {
        this.lastError = e?.response?.data?.detail || e?.message || String(e);
      }
    },

    async sendChat(message: string) {
      const text = (message || "").trim();
      if (!text) return;

      this.ensureSse();

      this.loading = true;
      this.isThinking = true;
      this.thinkingSteps = [];
      this._newTaskIds = new Set();
      this.thinkingTaskId = "";
      this.lastError = "";
      this.pushUser(text);

      try {
        const res = await api.post<ChatResponse>("/chat", {
          session_id: this.sessionId,
          message: text,
          agent_type: this.agentType,
        });

        const data = res.data;
        const history = data.history || [];
        // Only assign taskId if a new task was created during THIS chat interaction
        const taskId = Array.isArray(data.tasks)
          ? data.tasks.find((t) => this._newTaskIds.has(t.task_id))?.task_id
          : undefined;
        const agentType = (data.agent_type || this.agentType) as AgentType;
        this.pushAssistant(data.reply || "(空回复)", { history, taskId, agentType });
        this.lastHistory = history;

        this.papers = data.papers || [];
        this.pdfAssets = data.pdf_assets || [];
        this.translateAssets = data.translate_assets || [];

        if (Array.isArray(data.tasks)) {
          for (const t of data.tasks) this._upsertTask(t);
        }
      } catch (e: any) {
        const msg = e?.response?.data?.detail || e?.message || String(e);
        this.lastError = msg;
        this.pushAssistant(`请求失败：${msg}`);
      } finally {
        this.loading = false;
        this.isThinking = false;
        this.thinkingSteps = [];
        this._newTaskIds = new Set();
        this.thinkingTaskId = "";
      }
    },

    async fetchPdfAssets() {
      try {
        const resp = await api.get<{ assets: PdfAsset[] }>("/pdf/assets");
        this.pdfAssets = resp.data?.assets || [];
      } catch { }
    },

    async fetchTranslateAssets() {
      try {
        const resp = await api.get<{ assets: TranslateAsset[] }>("/translate/assets");
        this.translateAssets = resp.data?.assets || [];
      } catch { }
    },

    async deletePdfAsset(paperId: string) {
      const pid = (paperId || "").trim();
      if (!pid) return;

      this.lastError = "";
      try {
        const resp = await api.delete<DeleteAssetResponse>(`/pdf/assets/${encodeURIComponent(pid)}`, {
          params: { session_id: this.sessionId },
        });
        this._removePdfAssetLocal(pid);
        await this.fetchPdfAssets();
        return resp.data;
      } catch (e: any) {
        const msg = e?.response?.data?.detail || e?.message || String(e);
        this.lastError = msg;
        throw new Error(msg);
      }
    },

    async deleteTranslateAsset(paperId: string) {
      const pid = (paperId || "").trim();
      if (!pid) return;

      this.lastError = "";
      try {
        const resp = await api.delete<DeleteAssetResponse>(
          `/translate/assets/${encodeURIComponent(pid)}`,
          { params: { session_id: this.sessionId } }
        );
        this._removeTranslateAssetLocal(pid);
        await this.fetchTranslateAssets();
        return resp.data;
      } catch (e: any) {
        const msg = e?.response?.data?.detail || e?.message || String(e);
        this.lastError = msg;
        throw new Error(msg);
      }
    },

    async downloadPaper(refIndex1based: number) {
      if (this.preferAgent) {
        await this.sendChat(`下载第${refIndex1based}篇论文PDF`);
        return;
      }

      this.loading = true;
      this.lastError = "";
      try {
        const res = await api.post("/pdf/download", {
          session_id: this.sessionId,
          ref: refIndex1based,
          force: false,
        });
        this.pushAssistant(`下载结果：${JSON.stringify(res.data)}`);
        await this.refreshSnapshot();
      } catch (e: any) {
        const msg = e?.response?.data?.detail || e?.message || String(e);
        this.lastError = msg;
        this.pushAssistant(`下载失败：${msg}`);
      } finally {
        this.loading = false;
      }
    },

    async translatePaper(refIndex1based: number) {
      const tip = `准备翻译第${refIndex1based}篇论文（若未下载将自动先下载再翻译）`;

      if (this.preferAgent) {
        await this.sendChat(`${tip}。请开始翻译并输出中文PDF。`);
        return;
      }

      this.ensureSse();

      this.loading = true;
      this.lastError = "";
      try {
        this.pushAssistant(tip);
        const resp = await api.post<{ task: TranslateTask }>("/pdf/translate/async", {
          session_id: this.sessionId,
          ref: refIndex1based,
          force: false,
          service: "bing",
          threads: 4,
          keep_dual: false,
        });

        const task = resp.data?.task;
        if (task?.task_id) {
          this._upsertTask(task);
          this.pushAssistant(`已创建翻译任务：task_id=${task.task_id}（等待 SSE 推送完成）`);
        } else {
          this.pushAssistant(`已发起翻译任务（等待 SSE 推送完成）`);
        }
      } catch (e: any) {
        const msg = e?.response?.data?.detail || e?.message || String(e);
        this.lastError = msg;
        this.pushAssistant(`翻译失败：${msg}`);
      } finally {
        this.loading = false;
      }
    },

    clearChat() {
      this.messages = [];
      this.lastHistory = [];
      this.lastError = "";
    },
  },
});
