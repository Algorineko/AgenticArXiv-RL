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


def current_actions_only(history):
    """Historical SFT-style diagnostic: real setup executes but is not rendered."""
    if history.startswith('Thought:'):
        return history
    _, marker, generated = history.partition('\n\nThought:')
    return 'Thought:' + generated if marker else ''


def state_facts_only(history):
    """Remove only the legacy imperative, preserve all prior-state facts."""
    return history.replace('以上是已存在的会话状态，不属于当前任务动作，请勿重复执行。', '')
