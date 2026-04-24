"""Policy and apply helpers for post-hook tasks (grading, artifact summary).

`determine_posthooks` decides which post-hooks to spawn *immediately* after
a source task completes. Summary spawning happens later — after grade
passes — and is driven by `_apply_posthook_verdict` in `apply.py`.
"""
from __future__ import annotations

# Agent types that never need post-hooks:
# - mechanical: not LLM output, nothing to grade/summarise
# - shopping_pipeline, shopping_pipeline_v2: pipeline shells that
#   orchestrate deterministic handlers + internal LLM calls; grading
#   the outer task can't meaningfully improve any specific internal
#   call, and a FAIL verdict just re-runs the whole pipeline with the
#   same bad inputs (group_label_filter_gate loop observed 2026-04-24).
# - grader, artifact_summarizer: the post-hook runners themselves
_NO_POSTHOOKS_AGENT_TYPES: frozenset[str] = frozenset({
    "mechanical",
    "shopping_pipeline",
    "shopping_pipeline_v2",
    "grader",
    "artifact_summarizer",
})


def determine_posthooks(
    task: dict, task_ctx: dict, result: dict,
) -> list[str]:
    """Return the list of post-hook kinds to spawn immediately.

    Summary is NOT in the immediately-spawned list — it's deferred until
    the grade task passes (see apply._apply_posthook_verdict).
    """
    agent_type = task.get("agent_type", "")
    if agent_type in _NO_POSTHOOKS_AGENT_TYPES:
        return []
    if task_ctx.get("requires_grading") is False:
        return []
    return ["grade"]
