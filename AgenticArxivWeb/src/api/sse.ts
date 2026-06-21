// src/api/sse.ts
import type { SseEnvelope } from "@/api/types";

function joinUrl(base: string, path: string) {
  const b = (base || "").replace(/\/+$/, "");
  const p = (path || "").startsWith("/") ? path : `/${path}`;
  return `${b}${p}`;
}

export function buildEventsUrl(apiBase: string, sessionId: string) {
  const url = joinUrl(apiBase, "/events");
  const sid = encodeURIComponent(sessionId || "default");
  return `${url}?session_id=${sid}`;
}

export type SseOnEvent = (evt: SseEnvelope) => void;

export interface OpenSseOptions {
  onOpen?: () => void;
  onError?: (e: Event) => void;
  onEvent?: SseOnEvent;
  withCredentials?: boolean; // 如你后端依赖 cookie，可开启
}

/**
 * 打开后端 /events SSE 连接
 * 后端会发：event: <type>\ndata: <json>\n\n
 */
export function openEventsSse(apiBase: string, sessionId: string, opts: OpenSseOptions) {
  const url = buildEventsUrl(apiBase, sessionId);

  const es = new EventSource(url, { withCredentials: !!opts.withCredentials });

  const handle = (e: MessageEvent<string>) => {
    if (!e?.data) return;
    try {
      const obj = JSON.parse(e.data) as SseEnvelope;
      opts.onEvent?.(obj);
    } catch {
      // 忽略无法解析的 data
    }
  };

  // 后端 _sse_pack 会把 obj.type 作为 event 名
  const eventNames = [
    "connected",
    "task_created",
    "task_started",
    "task_progress",
    "task_succeeded",
    "task_failed",
    "asset_deleted",
    "agent_step",
    "event_bus_error",
    "error",
    "message",
  ];

  for (const name of eventNames) {
    es.addEventListener(name, handle as unknown as EventListener);
  }

  es.onopen = () => opts.onOpen?.();
  es.onerror = (e) => opts.onError?.(e);

  return es;
}
