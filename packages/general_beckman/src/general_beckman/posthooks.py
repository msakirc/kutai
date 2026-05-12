"""Policy and apply helpers for post-hook tasks (grading, artifact summary).

`determine_posthooks` decides which post-hooks to spawn *immediately* after
a source task completes. Summary spawning happens later — after grade
passes — and is driven by `_apply_posthook_verdict` in `apply.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, runtime_checkable

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
@dataclass
class MissionDialContext:
    """Founder-controlled dials passed to callable auto_wire_triggers.

    All fields are optional with conservative defaults so that existing
    callers (and T1C's future wiring) see identical behaviour until the
    real dial values are supplied.

    Fields
    ------
    qa_dial:
        Quality-assurance intensity.  ``"off"`` suppresses optional QA
        hooks; ``"standard"`` (default) applies the baseline set;
        ``"strict"`` adds additional verification passes.
    accessibility_dial:
        Accessibility-check level.  ``"off"`` (default) skips a11y hooks;
        ``"warn"`` fires them as warnings; ``"strict"`` promotes to blocker.
    multi_file_expansion:
        When ``True`` the expander is allowed to replace a single step with
        N per-file sub-steps (Z3 T2 multi-file feature expansion).
        ``False`` by default — safe for all existing missions.
    integration_replay:
        Integration-test replay mode.  ``"off"`` (default) skips replay;
        ``"smoke"`` runs the fastest subset; ``"full"`` runs the complete
        suite.
    """
    qa_dial: str = "standard"
    accessibility_dial: str = "off"
    multi_file_expansion: bool = False
    integration_replay: str = "off"


# Conservative default — no dials set; behaviour identical to pre-T1A.
_DEFAULT_DIAL_CONTEXT: MissionDialContext = MissionDialContext()


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
    cost_band:
        Relative cost of running this post-hook.  Used by T1C's review-
        density resolver to shed expensive hooks under tight dial settings
        without touching the registry entries themselves.

        - ``"cheap"``: mechanical / near-zero LLM cost (grounding, imports,
          pattern_lint, verify_artifacts).
        - ``"moderate"``: one LLM call or a non-trivial subprocess
          (code_review, test_run, design_system_check, openapi_sync,
          typescript_sync).
        - ``"heavy"``: spins an ephemeral DB, container, or long subprocess
          (migration_apply).

        Default is ``"cheap"`` — the safe conservative value for any new
        kind that hasn't been classified yet.
    auto_wire_triggers:
        Controls automatic wiring into steps whose ``produces`` list has a
        matching path.  Two forms are accepted:

        **Static form** — ``list[str]``:
            Glob patterns (fnmatch-style) matched against every entry in a
            step's ``produces`` list.  If any pattern matches any produce
            path, the kind is prepended to ``post_hooks`` automatically by
            the expander.  Empty list = no auto-wiring (must be declared
            explicitly on the step).

        **Callable form** — ``Callable[[MissionDialContext], list[str]]``:
            A function that receives the mission's :class:`MissionDialContext`
            and returns a glob list (same semantics as the static form).
            Use this when the trigger set should vary with dial settings —
            e.g. suppress a hook entirely when ``qa_dial == "off"``, or add
            extra globs when ``qa_dial == "strict"``.

            T1C will plug in real dial values; until then the expander passes
            :data:`_DEFAULT_DIAL_CONTEXT` so behaviour is identical to the
            static form with the same glob list.

            Example stub (returns the same globs regardless of dials)::

                auto_wire_triggers=lambda _ctx: ["*.py", "*.ts"]

    description:
        Human-readable one-liner for tooling / docs.
    """
    kind: str
    verb: str
    default_severity: str = "blocker"
    cost_band: Literal["cheap", "moderate", "heavy"] = "cheap"
    auto_wire_triggers: "list[str] | Callable[[MissionDialContext], list[str]]" = field(
        default_factory=list
    )
    description: str = ""

    def resolve_triggers(
        self,
        dial_ctx: "MissionDialContext | None" = None,
    ) -> list[str]:
        """Resolve ``auto_wire_triggers`` to a concrete glob list.

        Handles both the static ``list[str]`` form and the callable form.
        When *dial_ctx* is ``None``, :data:`_DEFAULT_DIAL_CONTEXT` is used
        so existing callers that don't pass dials yet see no behaviour change.
        """
        ctx = dial_ctx if dial_ctx is not None else _DEFAULT_DIAL_CONTEXT
        triggers = self.auto_wire_triggers
        if callable(triggers):
            return triggers(ctx)
        return list(triggers)


# ---------------------------------------------------------------------------
# Central registry
#
# T2/T3 agents add new kinds here — one row per kind, no other file changes
# needed for the registry + expander auto-wire side.  The apply layer still
# needs a matching branch in `_posthook_agent_and_payload` and
# `_apply_posthook_verdict`, but the *discovery* path is fully data-driven.
#
# SEVERITY_RAMP_TODO — v1 ramp policy (Z2 v2 plan):
#   The following kinds currently ship at default_severity="warning" and
#   should be promoted to "blocker" once semgrep is bundled into the CI
#   image and verified non-flaky across the full mission corpus:
#     - pattern_lint          (T2C, verb=run_semgrep, forbidden.yml)
#     - design_system_check   (T3C, verb=run_semgrep, design_system.yml)
#   Tracking lives here (not in an issue tracker) on purpose — the rule
#   row is the single source of truth, so the promotion is a one-line flip
#   in this file. Bump occurrences count in this comment when promoting.
# ---------------------------------------------------------------------------
POST_HOOK_REGISTRY: dict[str, PostHookSpec] = {
    "verify_artifacts": PostHookSpec(
        kind="verify_artifacts",
        verb="verify_artifacts",
        default_severity="blocker",
        cost_band="cheap",
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
        cost_band="moderate",
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
        cost_band="cheap",
        # Auto-wire on ANY step that declares produces — grounding is the
        # cheap L2 floor that fires before verify_artifacts to catch agents
        # that narrated completion without ever calling write_file.
        auto_wire_triggers=["*"],
        description=(
            "Mechanical grounding check: at least one successful write_file "
            "call per produces slot.  Fires before verify_artifacts."
        ),
    ),
    # Z2 T2B — static import checker.
    "imports_check": PostHookSpec(
        kind="imports_check",
        verb="check_imports",
        default_severity="blocker",
        cost_band="cheap",
        auto_wire_triggers=["*.py", "*.ts", "*.tsx"],
        description=(
            "Verify imports resolve against project manifest "
            "(pyproject.toml/requirements*.txt for Python; package.json for TS); "
            "fail on missing."
        ),
    ),
    # Z2 T2A — test_run.
    "test_run": PostHookSpec(
        kind="test_run",
        verb="run_tests",
        default_severity="blocker",
        cost_band="moderate",
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
    # Z2 T2C — pattern_lint via semgrep.
    "pattern_lint": PostHookSpec(
        kind="pattern_lint",
        verb="run_semgrep",
        default_severity="warning",
        cost_band="cheap",
        auto_wire_triggers=["*.py", "*.ts", "*.tsx", "*.js", "*.jsx"],
        description=(
            "Run semgrep with forbidden-patterns rule pack; warn on hits. "
            "Soft-skipped when semgrep is not installed."
        ),
    ),
    # Z2 T3B — openapi_sync.
    "openapi_sync": PostHookSpec(
        kind="openapi_sync",
        verb="regen_and_diff",
        default_severity="blocker",
        cost_band="moderate",
        auto_wire_triggers=[
            "**/routes/*.py",
            "**/routers/*.py",
            "**/api/*.py",
            "openapi.json",
            "openapi.yaml",
        ],
        description=(
            "Regenerate OpenAPI spec from routes; fail on drift vs committed openapi.json."
        ),
    ),
    # Z2 T3B — typescript_sync.
    "typescript_sync": PostHookSpec(
        kind="typescript_sync",
        verb="regen_and_diff",
        default_severity="blocker",
        cost_band="moderate",
        auto_wire_triggers=[
            "openapi.json",
            "openapi.yaml",
            "types/api.ts",
            "types/api/*.ts",
        ],
        description=(
            "Regenerate frontend API types via openapi-typescript; fail on drift."
        ),
    ),
    # Z2 T3C — design_system_check via shared semgrep engine.
    "design_system_check": PostHookSpec(
        kind="design_system_check",
        verb="run_semgrep",
        default_severity="warning",
        cost_band="moderate",
        auto_wire_triggers=["*.tsx", "*.jsx"],
        description=(
            "Run semgrep with design-system rule pack on JSX/TSX; warn on hits. "
            "Soft-skipped when semgrep is not installed. Rule pack is "
            "project-overridable via design_system_rule_pack in step context."
        ),
    ),
    # Z2 T3A — migration_apply.
    "migration_apply": PostHookSpec(
        kind="migration_apply",
        verb="apply_migration",
        default_severity="blocker",
        cost_band="heavy",
        auto_wire_triggers=[
            "migrations/*.py",
            "migrations/*.sql",
            "alembic/versions/*.py",
            "*.sql",
        ],
        description=(
            "Apply migration to ephemeral DB; fail on apply error. "
            "SQLite stack: direct apply via sqlite3. "
            "Postgres: testcontainers (opt-in via enable_testcontainers). "
            "Unknown stack: alembic offline mode (syntax check only)."
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
