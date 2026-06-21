# AgenticArxiv/services/runtime.py
from __future__ import annotations

from services.event_bus import EventBus
from services.translate_runner import TranslateRunner

# 全局单例（MVP：单进程）
event_bus = EventBus()
translate_runner = TranslateRunner(event_bus=event_bus)
