"""DLQ pause-pattern filter (module state).

Moved from Orchestrator.paused_patterns during Task 13. Telegram /dlq
commands mutate these via pause() / unpause(); queue.pick reads via is_paused.
"""
_patterns: set[str] = set()


def pause(pattern: str) -> None:
    _patterns.add(pattern)


def unpause(pattern: str) -> None:
    _patterns.discard(pattern)


def all_paused() -> set[str]:
    return set(_patterns)


def is_paused(task: dict) -> bool:
    cat = task.get("error_category")
    if not cat:
        return False
    return f"category:{cat}" in _patterns
