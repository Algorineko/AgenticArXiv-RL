<template>
  <Teleport to="body">
    <div v-if="open" class="modal-backdrop" @mousedown.self="close">
      <div
        class="modal"
        role="dialog"
        aria-modal="true"
        :aria-label="title || 'dialog'"
      >
        <header class="modal-header">
          <div class="modal-title">
            <slot name="title">
              {{ title }}
            </slot>
          </div>
        </header>

        <div class="modal-body">
          <slot />
        </div>

        <footer v-if="$slots.footer" class="modal-footer">
          <slot name="footer" />
        </footer>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { onBeforeUnmount, watch } from "vue";

const props = defineProps<{
  open: boolean;
  title?: string;
}>();

const emit = defineEmits<{
  (e: "close"): void;
}>();

function close() {
  emit("close");
}

function onKeydown(e: KeyboardEvent) {
  if (!props.open) return;
  if (e.key === "Escape") close();
}

watch(
  () => props.open,
  (v) => {
    // 打开时禁止 body 滚动，关闭恢复
    if (typeof document === "undefined") return;
    if (v) document.body.style.overflow = "hidden";
    else document.body.style.overflow = "";
  }
);

if (typeof window !== "undefined") {
  window.addEventListener("keydown", onKeydown);
}

onBeforeUnmount(() => {
  if (typeof window !== "undefined") {
    window.removeEventListener("keydown", onKeydown);
  }
  if (typeof document !== "undefined") {
    document.body.style.overflow = "";
  }
});
</script>

<style scoped>
.modal-backdrop{
  position: fixed;
  inset: 0;
  z-index: 999;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 18px;
  background: rgba(0,0,0,0.55);
  backdrop-filter: blur(6px);
}

.modal{
  width: min(980px, calc(100vw - 36px));
  max-height: min(84vh, 860px);
  display: flex;
  flex-direction: column;

  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--modal-bg);
  box-shadow: 0 18px 60px rgba(0,0,0,0.55);
  overflow: hidden;

  /* 关键：Teleport 到 body 后，显式设定颜色，避免默认黑字 */
  color: var(--fg);
}

.modal-header{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;

  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.02);

  color: var(--fg);
}

.modal-title{
  font-weight: 750;
  color: var(--fg);
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.modal-body{
  padding: 12px;
  overflow: auto;
  min-height: 0;
  color: var(--fg);
}

.modal-footer{
  border-top: 1px solid var(--border);
  padding: 10px 12px;
  background: rgba(255,255,255,0.02);
  color: var(--fg);
}
</style>