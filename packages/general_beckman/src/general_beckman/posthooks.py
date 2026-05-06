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
# - reviewer: reviewer agents ARE the quality judge for their step.
#   Grading their verdict is double-judgment — burns retries on style
#   nits about how reviewers should categorize findings (mission 57
#   task 4391 1.13 research_quality_review DLQ'd 5x with "high-severity
#   findings often stem from uncited statistics rather than logical
#   fallacies" — the reviewer's verdict was structurally clean, the
#   grader was second-guessing the rubric). When you need extra QA on
#   a reviewer step's output, chain another reviewer-typed step rather
#   than running grader as judge-of-judge.
_NO_POSTHOOKS_AGENT_TYPES: frozenset[str] = frozenset({
    "mechanical",
    "shopping_pipeline_v2",
    "grader",
    "artifact_summarizer",
    "reviewer",
    # CodeReviewerAgent runs as a post-hook over a build step's output.
    # Spawning a grader on it would be judge-of-judge — same reasoning
    # as for "reviewer" above. Its verdict IS the gate.
    "code_reviewer",
})


def determine_posthooks(
    task: dict, task_ctx: dict, result: dict,
) -> list[str]:
    """Return the list of post-hook kinds to spawn immediately.

    Default policy returns ``["grade"]`` for non-excluded agent types.
    Steps may declare additional kinds via ``post_hooks`` in ctx (e.g.
    ``["verify_artifacts"]`` for build steps that should grounds-check
    declared ``produces`` paths against the workspace). Summary is NOT
    in the immediately-spawned list — deferred until grade passes (see
    ``apply._apply_posthook_verdict``).
    """
    agent_type = task.get("agent_type", "")
    if agent_type in _NO_POSTHOOKS_AGENT_TYPES:
        return []

    kinds: list[str] = []
    if task_ctx.get("requires_grading") is not False:
        kinds.append("grade")

    # Extra kinds declared on the step. Filter to known kinds so a typo
    # doesn't spawn a task with no agent handler.
    extra = task_ctx.get("post_hooks") or []
    if isinstance(extra, list):
        for k in extra:
            if not isinstance(k, str) or not k.strip():
                continue
            if k in _KNOWN_EXTRA_KINDS and k not in kinds:
                kinds.append(k)
    return kinds


# Kinds that may appear in a step's ``post_hooks`` list. Each must have a
# matching branch in apply._posthook_agent_and_payload + apply._apply_posthook_verdict.
_KNOWN_EXTRA_KINDS: frozenset[str] = frozenset({
    "verify_artifacts",
    "code_review",
    "grounding",
})
