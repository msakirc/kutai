"""Policy and apply helpers for post-hook tasks (grading, artifact summary).

`determine_posthooks` decides which post-hooks to spawn *immediately* after
a source task completes. Summary spawning happens later — after grade
passes — and is driven by `_apply_posthook_verdict` in `apply.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# MissionDialContext — Z3 T1A/T1C: founder-dial shape used by the resolver
# and passed to post-hook expanders at expand-time (T2+ wire-up).
# ---------------------------------------------------------------------------

@dataclass
class MissionDialContext:
    """Resolved founder dials for one mission.

    Passed to post-hook expanders so they can adjust gate severity,
    add/skip kinds, or promote warnings → blockers based on founder
    settings.  Populated by ``src.workflows.review_density.get_dials``.

    Fields
    ------
    qa_dial:
        Overall QA intensity. ``quick`` skips expensive reviewers;
        ``standard`` runs defaults; ``strict`` promotes warnings to
        blockers and adds extra reviewer rounds.
    accessibility_dial:
        ``on`` → auto-wire a11y semgrep pack on JSX/TSX; ``off`` → skip.
    multi_file_expansion:
        ``True`` → expander breaks template steps into per-file sub-tasks
        (T2 feature); ``False`` → single-call emission (default).
    integration_replay:
        ``quick|standard|strict`` — how thoroughly the integration
        post-hook re-runs end-to-end tests after multi-file expansion.
    """
    qa_dial: str = "standard"
    accessibility_dial: str = "off"
    multi_file_expansion: bool = False
    integration_replay: str = "standard"


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
# Conservative default — no dials set; behaviour identical to pre-T1A.
# Vocab/defaults defined on MissionDialContext above; aligns with
# src/workflows/review_density.py validation:
#   qa_dial ∈ {quick, standard, strict}
#   accessibility_dial ∈ {on, off}
#   multi_file_expansion ∈ {True, False}
#   integration_replay ∈ {quick, standard, strict}
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
    # IntegrationReviewerAgent is a cross-file consistency reviewer.
    # Its verdict IS the gate — same judge-of-judge reasoning as
    # code_reviewer. Chain another reviewer-typed step for extra QA.
    "integration_reviewer",
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
# SEVERITY_RAMP — Promoted 2026-05-12.
#   pattern_lint + design_system_check now ship at default_severity="blocker".
#   semgrep pinned in requirements.txt (install via `pip install -r requirements.txt`).
#   Soft-skip path preserved when semgrep binary is absent.
#   Rule packs (forbidden.yml, design_system.yml) may need false-positive tuning
#   on real mission output — edit the rule packs to reduce noise before enabling
#   in high-throughput missions.
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
        default_severity="blocker",
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
        default_severity="blocker",
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
    # Z1 T5A (P6) — declared on step 1.11a (compliance_overlay_generation).
    "compliance_template_present": PostHookSpec(
        kind="compliance_template_present",
        verb="compliance_template_present",
        default_severity="blocker",
        auto_wire_triggers=[],
        description=(
            "Z1 T5A — assert each compliance_overlay generated_template_path "
            "exists on disk under compliance_templates/."
        ),
    ),
    # Z1 T5A (P6) — phase-boundary blocker check on step 6.6 (project_plan_review).
    "compliance_blocker_check": PostHookSpec(
        kind="compliance_blocker_check",
        verb="compliance_blocker_check",
        default_severity="blocker",
        auto_wire_triggers=[],
        description=(
            "Z1 T5A — phase-boundary check that required compliance docs are "
            "rendered before phase ≤6 closes."
        ),
    ),
    # Z1 T6A (A7) — declared on step 0.1 (product_charter_with_brand).
    "find_similar_missions": PostHookSpec(
        kind="find_similar_missions",
        verb="find_similar_missions",
        # Handler returns needs_review (not failed) when matches surface.
        # Warning severity keeps source uninterrupted; the needs_review row
        # surfaces to the founder via the standard surface path.
        default_severity="warning",
        auto_wire_triggers=[],
        description=(
            "Z1 T6A — cross-mission idea dedup. Embeds idea_summary and "
            "queries mission_ideas ChromaDB collection. needs_review on hit."
        ),
    ),
    # Z1 T6A (A7) — declared on step 0.1 alongside find_similar_missions.
    "index_idea_fingerprint": PostHookSpec(
        kind="index_idea_fingerprint",
        verb="index_idea_fingerprint",
        default_severity="warning",
        auto_wire_triggers=[],
        description=(
            "Z1 T6A — embed idea_brief into mission_ideas ChromaDB collection "
            "after charter lock. Feeds future find_similar_missions calls."
        ),
    ),
    # Z1 T6A (P9) — declared on step 0.5 (clarification_questions).
    "surface_prior_mission_hints": PostHookSpec(
        kind="surface_prior_mission_hints",
        verb="surface_prior_mission_hints",
        # Handler always completes (advisory hints only — no fail/needs_review).
        default_severity="warning",
        auto_wire_triggers=[],
        description=(
            "Z1 T6A — advisory hints from prior missions' ADR + compliance "
            "decisions. Always completes."
        ),
    ),
    # Z1 T6B (P5) — declared on step 1.0 (prior_art_search).
    "prior_art_min_coverage": PostHookSpec(
        kind="prior_art_min_coverage",
        verb="prior_art_min_coverage",
        default_severity="blocker",
        auto_wire_triggers=[],
        description=(
            "Z1 T6B — assert prior_art_report has attempted_solutions, "
            "key_lessons, resolvable URLs."
        ),
    ),
    # Z1 T2 (P4) — declared on every phase-3 step that emits a
    # commitment-shaped artifact (functional_requirements, etc.).
    # Note: each of these steps also has a sibling `.verify` mechanical
    # step. The post-hook is the cheaper earlier gate; the sibling step
    # is the belt-and-suspenders standalone.
    "verify_falsification_present": PostHookSpec(
        kind="verify_falsification_present",
        verb="verify_falsification_present",
        default_severity="blocker",
        auto_wire_triggers=[],
        description=(
            "Z1 T2 P4 — assert each commitment-shaped item carries "
            "risk_if_wrong / validation_method / falsification_signal."
        ),
    ),
    # Z1 T5C (B4) — standalone critic-gate post-hook. Inline critic-gate
    # already fires inside git_commit + notify_user; this slot lets a
    # workflow step declare `post_hooks: ["critic_gate"]` to bolt the gate
    # onto any other step (e.g. irreversible vendor calls, public repo
    # creation). action_name + target_payload are propagated from the
    # source step's context.
    "critic_gate": PostHookSpec(
        kind="critic_gate",
        verb="critic_gate",
        default_severity="blocker",
        auto_wire_triggers=[],
        description=(
            "Z1 T5C — standalone critic-gate. Veto fails the post-hook "
            "and the source step retries."
        ),
    ),
    # Z3 T2C — integration_review.
    "integration_review": PostHookSpec(
        kind="integration_review",
        verb="integration_reviewer",
        default_severity="blocker",
        cost_band="moderate",
        # No auto-wire triggers: injected by the expander as a sibling step
        # on multi-file expansions, not by file-pattern auto-wiring.
        auto_wire_triggers=[],
        description=(
            "Cross-file consistency review after multi-file feature expansion. "
            "AST signature mechanical pre-check (extract_signatures) feeds "
            "context into LLM integration_reviewer."
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
