// src/api/sseClient.ts
export type SseEvent = {
  event: string;   // event: xxx（可能为空）
  data: string;    // data: ...（可能是 JSON 字符串，也可能是纯文本）
  id?: string;
};

type OnEvent = (evt: SseEvent) => void;

function parseSseBlock(block: string): SseEvent | null {
  // SSE 一个事件块用 \n\n 分隔
  // 可能形式：
  // event: delta
  // data: {"text":"hi"}
  // data: {"text":"!"}
  // id: 123
  const lines = block.split(/\r?\n/);

  let event = "";
  let id: string | undefined = undefined;
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line.trim()) continue;
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("id:")) id = line.slice(3).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    // retry: 这里一般用不到
  }

  // 允许后端只发 data，不发 event
  const data = dataLines.join("\n");
  if (!event && !data) return null;
  return { event, data, id };
}

/**
 * 用 fetch(POST) 连接 SSE，并逐条回调 onEvent
 * 返回一个 abort() 用于中断连接
 */
export function postSse(
  url: string,
  body: unknown,
  onEvent: OnEvent,
  opts?: { signal?: AbortSignal; headers?: Record<string, string>; credentials?: RequestCredentials }
) {
  const controller = new AbortController();

  // 外部 signal 触发时也中断
  if (opts?.signal) {
    opts.signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  (async () => {
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        ...(opts?.headers || {}),
      },
      body: JSON.stringify(body),
      signal: controller.signal,
      credentials: opts?.credentials ?? "omit",
    });

    if (!resp.ok) {
      const txt = await resp.text().catch(() => "");
      throw new Error(`SSE HTTP ${resp.status}: ${txt || resp.statusText}`);
    }

    const reader = resp.body?.getReader();
    if (!reader) throw new Error("ReadableStream not supported or empty body.");

    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // 按 SSE 事件块分割
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);

        const evt = parseSseBlock(block);
        if (evt) onEvent(evt);
      }
    }
  })().catch((err) => {
    // 把错误包装成一个 error 事件回调给上层
    onEvent({ event: "error", data: String(err?.message || err) });
  });

  return {
    abort: () => controller.abort(),
  };
}
