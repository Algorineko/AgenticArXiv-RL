<template>
  <div class="app">
    <Sidebar :currentPage="currentPage" @update:currentPage="currentPage = $event" />

    <main class="main-content">
      <KeepAlive>
        <component :is="pageComponent" />
      </KeepAlive>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import Sidebar from "@/components/Sidebar.vue";
import ChatPanel from "@/components/ChatPanel.vue";
import PapersPanel from "@/components/PapersPanel.vue";
import AssetsPanel from "@/components/AssetsPanel.vue";
import LogsPanel from "@/components/LogsPanel.vue";
import SettingsPanel from "@/components/SettingsPanel.vue";

const currentPage = ref("chat");

const pageMap: Record<string, any> = {
  chat: ChatPanel,
  papers: PapersPanel,
  assets: AssetsPanel,
  logs: LogsPanel,
  settings: SettingsPanel,
};

const pageComponent = computed(() => pageMap[currentPage.value] || ChatPanel);
</script>

<style scoped>
.app {
  min-height: 100vh;
  background: var(--bg);
  color: var(--fg);
}

.main-content {
  margin-left: 60px;
  padding: 14px 18px;
  min-height: 100vh;
}

@media (max-width: 768px) {
  .main-content {
    margin-left: 0;
    margin-bottom: 52px;
    padding: 10px;
  }
}
</style>
