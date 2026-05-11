"""Policy and apply helpers for post-hook tasks (grading, artifact summary).

`determine_posthooks` decides which post-hooks to spawn *immediately* after
a source task completes. Summary spawning happens later — after grade
passes — and is driven by `_apply_posthook_verdict` in `apply.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class PostHookSpec:
    """Descriptor for one post-hook kind.

    Fields
    ------
    kind:
        Canonical kind name used in step ``post_hooks`` lists and throughout
        the apply pipeline (e.g. ``"verify_artifacts"``).
    verb:
        The mr_roboto action verb dispatched for mechanical hooks, OR the
        agent_type for LLM hooks (e.g. ``"verify_artifacts"``,
        ``"check_grounding"``, ``"code_reviewer"``).  The apply layer uses
        this for routing; T1 preserves the existing mapping exactly.
    default_severity:
        ``"blocker"`` (failure DLQs source) or ``"warning"`` (future use for
        non-fatal checks).  All 3 migrated kinds are blockers.
    auto_wire_triggers:
        Glob patterns (fnmatch-style) matched against every entry in a step's
        ``produces`` list.  If any pattern matches any produce path, the kind
        is prepended to ``post_hooks`` automatically by the expander.
        Empty list = no auto-wiring (must be declared explicitly on the step).
    description:
        Human-readable one-liner for tooling / docs.
    """
    kind: str
    verb: str
    default_severity: str = "blocker"
    auto_wire_triggers: list[str] = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Central registry
#
# T2/T3 agents add new kinds here — one row per kind, no other file changes
# needed for the registry + expander auto-wire side.  The apply layer still
# needs a matching branch in `_posthook_agent_and_payload` and
# `_apply_posthook_verdict`, but the *discovery* path is fully data-driven.
# ---------------------------------------------------------------------------
POST_HOOK_REGISTRY: dict[str, PostHookSpec] = {
    "verify_artifacts": PostHookSpec(
        kind="verify_artifacts",
        verb="verify_artifacts",
        default_severity="blocker",
        # No glob triggers: verify_artifacts is wired explicitly on steps
        # that want file-existence + parse checks.  Grounding (cheaper)
        # is auto-wired on all produces; verify is an opt-in second gate.
        auto_wire_triggers=[],
        description=(
            "Mechanical check: each declared produces path exists, is non-empty, "
            "and parses for known extensions."
        ),
    ),
    "code_review": PostHookSpec(
        kind="code_review",
        verb="code_reviewer",
        default_severity="blocker",
        # No glob triggers: code review is expensive and must be opted in
        # explicitly per step.
        auto_wire_triggers=[],
        description=(
            "LLM code reviewer judges emitted code; PASS/FAIL drives the same "
            "retry-with-feedback path as verify_artifacts."
        ),
    ),
    "grounding": PostHookSpec(
        kind="grounding",
        verb="check_grounding",
        default_severity="blocker",
        # Auto-wire on ANY step that declares produces — grounding is the
        # cheap L2 floor that fires before verify_artifacts to catch agents
        # that narrated completion without ever calling write_file.
        auto_wire_triggers=["*"],
        description=(
            "Mechanical grounding check: at least one successful write_file "
            "call per produces slot.  Fires before verify_artifacts."
        ),
    ),
    # T2A: test_run ----------------------------------------------------------------
    # Auto-wire when a step produces test files.  The mr_roboto dispatcher
    # picks the right runner (run_pytest / run_jest / run_vitest) from the
    # target file extensions + optional stack_hint in the payload.
    # Idempotent: if the kind is already in post_hooks it is never added twice
    # (enforced by _auto_wire_posthooks in expander.py).
    "test_run": PostHookSpec(
        kind="test_run",
        verb="run_tests",  # logical key; actual runner picked at dispatch
        default_severity="blocker",
        auto_wire_triggers=[
            "tests/*",
            "test_*.py",
            "*.test.ts",
            "*.test.tsx",
            "*.spec.ts",
            "*.spec.tsx",
        ],
        description=(
            "Run pytest/jest/vitest on produced test files; fail on red. "
            "Runner picked by file extension; slow suite (>120s) warns but "
            "does not block."
        ),
    ),
}

# ---------------------------------------------------------------------------
# Back-compat aliases
#
# Anything that imports POST_HOOK_KINDS or _KNOWN_EXTRA_KINDS continues to
# work without changes.
# ---------------------------------------------------------------------------

#: Frozenset of all registered kind names.  Derived from registry so it
#: stays in sync automatically as T2/T3 register new kinds.
POST_HOOK_KINDS: frozenset[str] = frozenset(POST_HOOK_REGISTRY.keys())

#: Legacy alias used by apply.py and any external caller.
_KNOWN_EXTRA_KINDS: frozenset[str] = POST_HOOK_KINDS


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

    Extra kinds are validated against ``POST_HOOK_REGISTRY`` (derived from
    the registry, not hardcoded) so new T2/T3 kinds are accepted as soon as
    they register a row.
    """
    agent_type = task.get("agent_type", "")
    if agent_type in _NO_POSTHOOKS_AGENT_TYPES:
        return []

    kinds: list[str] = []
    if task_ctx.get("requires_grading") is not False:
        kinds.append("grade")

    # Extra kinds declared on the step. Filter to registry so a typo
    # doesn't spawn a task with no agent handler.
    extra = task_ctx.get("post_hooks") or []
    if isinstance(extra, list):
        for k in extra:
            if not isinstance(k, str) or not k.strip():
                continue
            if k in POST_HOOK_REGISTRY and k not in kinds:
                kinds.append(k)
    return kinds
