<template>
  <section class="papers">
    <header class="card-header">
      <div class="title">论文信息区（session 临时记忆）</div>

      <div class="row">
        <input class="input" v-model="q" placeholder="搜索标题/作者/arxiv id…" />
        <button class="btn" @click="store.refreshSnapshot()" :disabled="store.loading">刷新</button>
      </div>

      <p class="muted">
        共 {{ filtered.length }} 篇（显示编号 1..N，便于“下载第2篇/翻译第3篇”）
      </p>
    </header>

    <!-- 独立滚动区域：只滚表格，不滚整个右侧页面 -->
    <div class="table-scroll" v-if="filtered.length">
      <table class="table">
        <thead>
          <tr>
            <th class="col-idx">#</th>
            <th>标题</th>
            <th class="col-auth">作者</th>
            <th class="col-time">时间</th>
            <th class="col-flag">下载</th>
            <th class="col-flag">翻译</th>
            <th class="col-op">操作</th>
          </tr>
        </thead>

        <tbody>
          <tr v-for="p in filtered" :key="p.id">
            <td class="mono col-idx">{{ indexOf(p.id) }}</td>

            <td>
              <div class="p-title">{{ p.title }}</div>
              <div class="sub mono">
                {{ p.id }}
                <span v-if="p.primary_category"> · {{ p.primary_category }}</span>
              </div>
              <div class="sub" v-if="p.comment">注释：{{ p.comment }}</div>

              <div class="sub" v-if="p.pdf_url">
                PDF：
                <a :href="normalizePdf(p.pdf_url)" target="_blank" rel="noreferrer">
                  {{ normalizePdf(p.pdf_url) }}
                </a>
              </div>
            </td>

            <td class="col-auth">
              <div class="sub">
                {{ (p.authors || []).slice(0, 6).join(", ") }}
                <span v-if="(p.authors || []).length > 6">…</span>
              </div>
            </td>

            <td class="sub col-time">
              <div>pub: {{ p.published || "-" }}</div>
              <div>upd: {{ p.updated || "-" }}</div>
            </td>

            <!-- ✅/❌：READY=✅，其他一律❌。用 title 保留具体状态信息 -->
            <td class="col-flag">
              <span
                class="flag"
                :class="pdfReady(p.id) ? 'ok' : 'bad'"
                :title="'PDF: ' + pdfStatusText(p.id)"
              >
                {{ pdfReady(p.id) ? "✅" : "❌" }}
              </span>
            </td>

            <td class="col-flag">
              <span
                class="flag"
                :class="trReady(p.id) ? 'ok' : 'bad'"
                :title="'Translate: ' + trStatusText(p.id)"
              >
                {{ trReady(p.id) ? "✅" : "❌" }}
              </span>
            </td>

            <td class="col-op">
              <div class="ops">
                <!-- 下载 -->
                <button
                  class="btn icon"
                  :title="pdfReady(p.id) ? '下载 PDF（覆盖/更新缓存）' : '下载 PDF'"
                  aria-label="下载 PDF"
                  @click="store.downloadPaper(indexOf(p.id))"
                  :disabled="store.loading"
                >
                  <!-- download icon -->
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                      stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 3v10"></path>
                    <path d="M8 9l4 4 4-4"></path>
                    <path d="M4 20h16"></path>
                    <path d="M6 20v-4"></path>
                    <path d="M18 20v-4"></path>
                  </svg>
                </button>

                <!-- 翻译 -->
                <button
                  class="btn primary icon"
                  :title="trReady(p.id) ? '重新翻译（更新翻译缓存）' : '翻译 PDF'"
                  aria-label="翻译 PDF"
                  @click="store.translatePaper(indexOf(p.id))"
                  :disabled="store.loading"
                >
                  <!-- translate icon -->
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                      stroke-linecap="round" stroke-linejoin="round">
                    <path d="M16 4l4 4-4 4"></path>
                    <path d="M20 8H4"></path>
                    <path d="M8 20l-4-4 4-4"></path>
                    <path d="M4 16h16"></path>
                  </svg>
                </button>
              </div>

            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="empty" v-else>
      暂无论文
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { useAppStore } from "@/stores/appStore";

const store = useAppStore();
const q = ref("");

const filtered = computed(() => {
  const kw = q.value.trim().toLowerCase();
  if (!kw) return store.papers;
  return store.papers.filter((p) => {
    const hay = [
      p.id,
      p.title,
      (p.authors || []).join(" "),
      p.primary_category || "",
      (p.categories || []).join(" "),
      p.comment || "",
    ]
      .join(" ")
      .toLowerCase();
    return hay.includes(kw);
  });
});

// 论文在 session 里的顺序就是“第N篇”
function indexOf(paperId: string): number {
  const i = store.papers.findIndex((p) => p.id === paperId);
  return i >= 0 ? i + 1 : -1;
}

function normalizePdf(url: string) {
  return url.endsWith(".pdf") ? url : `${url}.pdf`;
}

function pdfStatusText(paperId: string) {
  const a = store.pdfMap.get(paperId);
  return a ? a.status : "NOT_DOWNLOADED";
}
function trStatusText(paperId: string) {
  const a = store.translateMap.get(paperId);
  return a ? a.status : "NOT_TRANSLATED";
}

function pdfReady(paperId: string) {
  return pdfStatusText(paperId) === "READY";
}
function trReady(paperId: string) {
  return trStatusText(paperId) === "READY";
}

function shouldWarnTranslate(paperId: string) {
  const a = store.pdfMap.get(paperId);
  return !a || a.status !== "READY";
}
</script>

<style scoped>
.papers {
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

.title { font-weight: 700; margin-bottom: 6px; }
.row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.input {
  flex:1;
  min-width: 240px;
  height: 32px;
  padding: 0 10px;
  border:1px solid var(--border);
  border-radius: 8px;
  background: var(--bg2);
  color: var(--fg);
}
.muted { color: var(--muted); margin: 8px 0 0; }

.table-scroll {
  /* 关键：独立滚动条只出现在这里 */
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding-right: 2px;
}

.table {
  width: 100%;
  border-collapse: collapse;
}

thead th {
  position: sticky;   /* 表头固定，滚动时更好用 */
  top: 0;
  z-index: 1;
  background: rgba(10, 10, 12, 0.92);
  backdrop-filter: blur(8px);
}

th, td {
  border-bottom: 1px solid var(--border);
  padding: 10px;
  vertical-align: top;
}

th { text-align:left; color: var(--muted); font-weight: 600; }

.p-title { font-weight: 700; margin-bottom: 4px; }
.sub { color: var(--muted); font-size: 12px; line-height: 1.4; margin-top: 2px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }

/* ✅/❌ 小圆角胶囊，带 title 可看真实状态 */
.flag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 36px;
  height: 24px;
  padding: 0 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
  user-select: none;
}
.flag.ok { border-color: rgba(0,255,153,.35); }
.flag.bad { border-color: rgba(255,107,107,.35); }

.ops { display:flex; gap:8px; flex-wrap:wrap; }
.col-op { width: 96px; }
.hint { margin-top: 6px; color: #f6c177; font-size: 12px; }
.empty { padding: 12px; color: var(--muted); }

/* 列宽更稳定（桌面端） */
.col-idx { width: 56px; }
.col-auth { width: 220px; }
.col-time { width: 150px; }
.col-flag { width: 88px; }
.col-op { width: 160px; }

/* 小屏优化：隐藏作者/时间列，减少横向滚动 */
@media (max-width: 768px) {
  .col-auth, .col-time { display: none; }
  td.col-auth, td.col-time { display: none; }
}
</style>
