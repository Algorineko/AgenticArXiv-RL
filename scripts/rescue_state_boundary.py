"""Frozen label-only state/history separation; no expected actions are visible."""
def boundary_history(history):
    initial, marker, generated = history.partition("\n\nThought:")
    if history.startswith("Thought:"):
        initial, generated = "", history
    elif marker:
        generated = "Thought:" + generated
    return ("[请求开始前的状态记录]\n" + (initial.strip() or "无") +
            "\n[/请求开始前的状态记录]\n\n[本轮请求的动作记录]\n" +
            (generated.strip() or "无") + "\n[/本轮请求的动作记录]")
