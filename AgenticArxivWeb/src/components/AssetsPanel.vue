<template>
  <section class="assets">
    <header class="card-header">
      <div class="head">
        <div>
          <div class="title">下载/翻译缓存</div>
          <p class="muted">
            PDF: {{ pdfList.length }} 条 · 翻译: {{ trList.length }} 条 · 任务: {{ store.tasks.length }} 条
          </p>
        </div>

        <div class="head-actions">
          <button class="btn" @click="store.refreshSnapshot()" :disabled="store.loading">刷新</button>
        </div>
      </div>

      <p v-if="store.lastError" class="errline">⚠ {{ store.lastError }}</p>
    </header>

    <div class="lists">
      <!-- 任务区 -->
      <div class="block">
        <div class="block-title">最近翻译任务（SSE）</div>
        <div class="list-scroll">
          <div v-if="!taskList.length" class="empty">暂无任务（发起“翻译第N篇”后这里会实时更新）</div>

          <div v-for="t in taskList" :key="t.task_id" class="item">
            <div class="item-top">
              <div class="item-title-block">
                <div class="item-title" v-if="paperTitle(t.paper_id)">{{ paperTitle(t.paper_id) }}</div>
                <div class="mono id">{{ t.paper_id }}</div>
              </div>
              <span class="pill" :class="statusClass(t.status)">{{ t.status }}</span>
            </div>

            <div class="sub mono">task_id: {{ t.task_id }}</div>

            <!-- progress: 进度条 + 百分比（仍保留百分比数字在其后显示） -->
            <div class="sub progress-line" v-if="typeof t.progress === 'number'">
              <span class="p-label">progress:</span>

              <div
                class="pbar"
                role="progressbar"
                :aria-valuenow="pctOf(t.progress)"
                aria-valuemin="0"
                aria-valuemax="100"
                :title="pctOf(t.progress) + '%'"
              >
                <div class="pbar-fill" :style="{ width: pctOf(t.progress) + '%' }"></div>
              </div>

              <span class="p-pct">{{ pctOf(t.progress) }}%</span>
            </div>

            <div class="sub mono" v-if="t.output_pdf_path">out: {{ t.output_pdf_path }}</div>
            <div class="sub err" v-if="t.error">error: {{ t.error }}</div>
          </div>
        </div>
      </div>

      <!-- 已下载 -->
      <div class="block">
        <div class="block-title">已下载（READY）</div>
        <div class="list-scroll">
          <div v-if="!pdfList.length" class="empty">暂无已下载 PDF</div>

          <div v-for="a in pdfList" :key="a.paper_id" class="item">
            <div class="item-top">
              <div class="item-title-block">
                <div class="item-title" v-if="paperTitle(a.paper_id)">{{ paperTitle(a.paper_id) }}</div>
                <div class="mono id">{{ a.paper_id }}</div>
              </div>

              <div class="right-actions">
                <span class="pill ok">READY</span>

                <!-- 查看 raw -->
                <button
                  class="btn icon small"
                  title="查看（新标签页打开 raw PDF）"
                  aria-label="查看 raw PDF"
                  :disabled="store.loading"
                  @click="store.openRawPdf(a.paper_id)"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                      stroke-linecap="round" stroke-linejoin="round">
                    <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"></path>
                    <circle cx="12" cy="12" r="3"></circle>
                  </svg>
                </button>

                <!-- 删除 raw -->
                <button
                  class="btn icon small danger"
                  title="删除已下载 PDF（raw）"
                  aria-label="删除已下载 PDF"
                  :disabled="store.loading"
                  @click="onDeletePdf(a.paper_id)"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                      stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 6h18"></path>
                    <path d="M8 6V4h8v2"></path>
                    <path d="M6 6l1 16h10l1-16"></path>
                    <path d="M10 11v6"></path>
                    <path d="M14 11v6"></path>
                  </svg>
                </button>
              </div>
            </div>

            <div class="sub mono">path: {{ a.local_path }}</div>
            <div class="sub">
              size: {{ formatBytes(a.size_bytes) }} · updated: {{ a.updated_at }}
            </div>
          </div>
        </div>
      </div>

      <!-- 已翻译 -->
      <div class="block">
        <div class="block-title">已翻译（READY）</div>
        <div class="list-scroll">
          <div v-if="!trList.length" class="empty">暂无已翻译 PDF</div>

          <div v-for="t in trList" :key="t.paper_id" class="item">
            <div class="item-top">
              <div class="item-title-block">
                <div class="item-title" v-if="paperTitle(t.paper_id)">{{ paperTitle(t.paper_id) }}</div>
                <div class="mono id">{{ t.paper_id }}</div>
              </div>

              <div class="right-actions">
                <span class="pill ok">READY</span>

                <!-- 查看 mono -->
                <button
                  class="btn icon small"
                  title="查看（新标签页打开 mono PDF）"
                  aria-label="查看 mono PDF"
                  :disabled="store.loading"
                  @click="store.openTranslatedPdf(t.paper_id, 'mono')"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                      stroke-linecap="round" stroke-linejoin="round">
                    <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"></path>
                    <circle cx="12" cy="12" r="3"></circle>
                  </svg>
                </button>

                <!-- 可选：若有 dual -->
                <button
                  v-if="t.output_dual_path"
                  class="btn icon small"
                  title="查看（新标签页打开 dual PDF）"
                  aria-label="查看 dual PDF"
                  :disabled="store.loading"
                  @click="store.openTranslatedPdf(t.paper_id, 'dual')"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                      stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 5h7a2 2 0 0 1 2 2v14H5a2 2 0 0 1-2-2V5z"></path>
                    <path d="M14 7a2 2 0 0 1 2-2h5v14a2 2 0 0 1-2 2h-5V7z"></path>
                  </svg>
                </button>

                <!-- 删除翻译 -->
                <button
                  class="btn icon small danger"
                  title="删除已翻译 PDF（mono/dual/log）"
                  aria-label="删除已翻译 PDF"
                  :disabled="store.loading"
                  @click="onDeleteTranslate(t.paper_id)"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                      stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 6h18"></path>
                    <path d="M8 6V4h8v2"></path>
                    <path d="M6 6l1 16h10l1-16"></path>
                    <path d="M10 11v6"></path>
                    <path d="M14 11v6"></path>
                  </svg>
                </button>
              </div>
            </div>

            <div class="sub mono">out: {{ t.output_mono_path }}</div>
            <div class="sub">
              threads: {{ t.threads }} · updated: {{ t.updated_at }}
            </div>
            <div class="sub" v-if="t.output_dual_path">
              dual: <span class="mono">{{ t.output_dual_path }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useAppStore } from "@/stores/appStore";

const store = useAppStore();

const taskList = computed(() => (store.tasks || []).slice(0, 10));

const pdfList = computed(() => {
  const arr = Array.from(store.pdfMap.values()).filter((x) => x.status === "READY");
  return arr.sort((a, b) => (String(b.updated_at || "")).localeCompare(String(a.updated_at || "")));
});

const trList = computed(() => {
  const arr = Array.from(store.translateMap.values()).filter((x) => x.status === "READY");
  return arr.sort((a, b) => (String(b.updated_at || "")).localeCompare(String(a.updated_at || "")));
});

function pctOf(p: number) {
  const v = Number(p);
  if (!Number.isFinite(v)) return 0;
  const clamped = Math.max(0, Math.min(1, v));
  return Math.round(clamped * 100);
}

function formatBytes(n: number | null | undefined) {
  const v0 = Number(n || 0);
  if (!Number.isFinite(v0) || v0 <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = v0;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i === 0 ? 0 : 2)} ${units[i]}`;
}

function paperTitle(paperId: string): string | undefined {
  const p = store.papers.find((x) => x.id === paperId);
  return p?.title;
}

function statusClass(s: string) {
  const x = (s || "").toUpperCase();
  if (x.includes("SUCC")) return "ok";
  if (x.includes("FAIL")) return "bad";
  if (x.includes("RUN")) return "warn";
  return "neutral";
}

async function onDeletePdf(paperId: string) {
  const ok = window.confirm(
    `确认删除已下载 PDF？\n\npaper_id=${paperId}\n\n(只删除 raw PDF 与 pdf_cache.json 记录，不删除翻译结果)`
  );
  if (!ok) return;

  try {
    await store.deletePdfAsset(paperId);
  } catch {
    // store.lastError 已写入
  }
}

async function onDeleteTranslate(paperId: string) {
  const ok = window.confirm(
    `确认删除已翻译 PDF？\n\npaper_id=${paperId}\n\n(会删除 mono/dual/log 与 translate_cache.json 记录，不删除 raw PDF)`
  );
  if (!ok) return;

  try {
    await store.deleteTranslateAsset(paperId);
  } catch {
    // store.lastError 已写入
  }
}
</script>

<style scoped>
.assets {
  height: calc(100vh - 36px);
  display: flex;
  flex-direction: column;
  min-height: 0;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: rgba(255,255,255,0.02);
  overflow: hidden;
}

.card-header {
  padding: 12px 12px 10px;
  margin-bottom: 10px;
  border-bottom: 1px solid var(--border);
}

.head{
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap: 12px;
}

.head-actions{
  display:flex;
  gap: 8px;
  align-items:center;
}

.title { font-weight: 700; margin-bottom: 6px; }
.muted { color: var(--muted); margin: 0; font-size: 12px; }

.errline{
  margin: 8px 0 0;
  color: #ff6b6b;
  font-size: 12px;
}

.lists {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-rows: 1fr 1fr 1fr;
  gap: 12px;
}

.block {
  border: 1px solid var(--border);
  border-radius: 14px;
  background: rgba(255,255,255,0.02);
  padding: 10px;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.block-title {
  font-weight: 650;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 8px;
}

.list-scroll {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding-right: 2px;
}

.item {
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px;
  padding: 10px;
  background: rgba(255,255,255,0.015);
  margin-bottom: 8px;
}

.item-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 4px;
}

.right-actions{
  display:flex;
  align-items:center;
  gap: 8px;
}

.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
.id { font-weight: 600; font-size: 11px; }

.item-title-block { min-width: 0; flex: 1; }
.item-title {
  font-weight: 700;
  font-size: 13px;
  margin-bottom: 2px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.sub { color: var(--muted); font-size: 12px; line-height: 1.35; margin-top: 2px; }
.sub.err { color: #ff6b6b; }

.pill {
  display: inline-flex;
  align-items: center;
  height: 20px;
  padding: 0 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
  font-size: 12px;
  user-select: none;
}

.pill.ok { border-color: rgba(0,255,153,.35); }
.pill.bad { border-color: rgba(255,107,107,.35); }
.pill.warn { border-color: rgba(255,200,0,.35); }
.pill.neutral { opacity: .75; }

.empty { color: var(--muted); font-size: 12px; padding: 6px 2px; }

/* Progress bar */
.progress-line{
  display: flex;
  align-items: center;
  gap: 8px;
}

.p-label{
  color: var(--muted);
  white-space: nowrap;
}

.pbar{
  position: relative;
  height: 8px;
  width: 160px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  overflow: hidden;
  flex: 0 0 auto;
}

.pbar-fill{
  height: 100%;
  width: 0%;
  border-radius: 999px;
  background: rgba(89,153,255,0.55);
  transition: width 180ms ease;
}

.p-pct{
  min-width: 44px;
  text-align: right;
  white-space: nowrap;
  color: var(--muted);
}
</style>