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
    # SP3: grade / code_review post-hook children both run as raw_dispatch
    # "reviewer" tasks; spawning a grade on a reviewer is judge-of-judge.
    "reviewer",
    # SP3: the summary post-hook child runs as a raw_dispatch "summarizer"
    # task. This is the recursion guard — a summarizer child must NEVER get a
    # grade post-hook (it would spawn an endless grade→summary→grade chain).
    "summarizer",
    # IntegrationReviewerAgent is a cross-file consistency reviewer.
    # Its verdict IS the gate — same judge-of-judge reasoning as
    # code_reviewer. Chain another reviewer-typed step for extra QA.
    "integration_reviewer",
    # AdrDriftJudgeAgent is the LLM gray-zone consumer for ADRs whose
    # falsification_signal cannot be checked mechanically. Its verdict IS
    # the gate — judge-of-judge would loop forever.
    "adr_drift_judge",
    # SP3b: self_reflect + constrained_emit are post-hook child tasks
    # themselves. They must NEVER spawn grade/reflect/emit hooks of their own
    # — that would create an infinite reflect→grade→reflect or
    # emit→grade→emit chain.
    "self_reflect",
    "constrained_emit",
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

def _dial_get(ctx, key: str, default):
    """Read a dial value from a MissionDialContext OR plain dict OR None.

    Z3 T3 callable auto_wire_triggers receive either:
    - MissionDialContext (when expander calls via resolve_triggers)
    - dict (legacy/test path that mimics step context)
    - None (no dials wired yet)

    Explicit None values in either source map back to *default* — None means
    "no preference", not "force null".
    """
    if ctx is None:
        return default
    if isinstance(ctx, dict):
        val = ctx.get(key, default)
    else:
        val = getattr(ctx, key, default)
    return default if val is None else val


def _shape_check_spec(name: str, desc: str) -> "PostHookSpec":
    """A blocker PostHookSpec for a parameterized mechanical `checks` verb.

    Kind == verb (the mr_roboto action name). No auto-wire triggers — checks
    are declared explicitly on the producer via the `checks` field, never
    glob-matched. apply._CHECK_KINDS picks these up by the ``verify_*`` name
    convention; the producer-supplied payload drives the actual check.
    """
    return PostHookSpec(
        kind=name, verb=name, default_severity="blocker",
        auto_wire_triggers=[], description=desc,
    )


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
        # Descriptive metadata only — NOT read for routing. SP3: the code_review
        # LLM hook is spawned as a raw_dispatch "reviewer" child by
        # apply._enqueue_posthook_llm_child; this field is retained for the
        # historical registry contract (test_z2_t1_posthook_registry).
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
    # Parameterized mechanical CHECK verbs — converted from standalone `.verify`
    # workflow steps that dead-ended (a failed standalone verify only blocked
    # its dependents and waited for a human; the producer stayed `completed`
    # and was never re-run). Declared on the producer via the SEPARATE `checks`
    # field (NOT `post_hooks`, which stays a pure list[str]) so each carries its
    # own payload (which file + bounds). They route through the same rail as
    # grounding / verify_artifacts: a failed check re-pends the producer with
    # feedback via _apply_simple_blocker_verdict. apply._CHECK_KINDS is derived
    # from these rows by the ``verify_*`` name convention; the apply layer
    # shares ONE payload builder (reads checks[].payload verbatim) + ONE verdict
    # branch. Convert a verb = register a row here + move the standalone step's
    # payload into the producer's `checks`. No handler code.
    #
    # NOTE: verify_falsification_present is intentionally NOT in this group —
    # it predates `checks`, routes through the Z1 blocker path, and is already
    # wired as a `post_hooks` entry on steps 3.1/3.2/3.3/3.7. _derive_check_kinds
    # excludes it explicitly.
    "verify_interview_script_shape": _shape_check_spec(
        "verify_interview_script_shape",
        "Z1 A4 — well-formed interview script (front matter, Logistics, 5-7 Q "
        "blocks with Question/Probes/Looking-for).",
    ),
    "verify_reverse_pitch_shape": _shape_check_spec(
        "verify_reverse_pitch_shape", "Reverse-pitch doc shape."),
    "verify_charter_shape": _shape_check_spec(
        "verify_charter_shape", "Product-charter doc shape (solutions, brand keywords)."),
    "verify_non_goals_shape": _shape_check_spec(
        "verify_non_goals_shape", "Non-goals doc shape (item count bounds)."),
    "verify_competitive_positioning_shape": _shape_check_spec(
        "verify_competitive_positioning_shape", "Competitive-positioning doc shape."),
    "verify_taste_emphasis_shape": _shape_check_spec(
        "verify_taste_emphasis_shape", "taste_emphasis.json shape."),
    "verify_design_tokens_shape": _shape_check_spec(
        "verify_design_tokens_shape", "design_tokens.json shape."),
    "verify_surfaces_shape": _shape_check_spec(
        "verify_surfaces_shape", "surfaces.json shape."),
    "verify_user_flow_shape": _shape_check_spec(
        "verify_user_flow_shape", "user_flow.md shape."),
    "verify_screen_inventory_shape": _shape_check_spec(
        "verify_screen_inventory_shape", "screen_inventory.md shape (+ optional follow_up)."),
    "verify_shared_shell_shape": _shape_check_spec(
        "verify_shared_shell_shape", "shared_shell.md shape."),
    "verify_screen_plan_shape": _shape_check_spec(
        "verify_screen_plan_shape", "Per-screen plan shape."),
    "verify_html_prototype_shape": _shape_check_spec(
        "verify_html_prototype_shape", "HTML prototype shape."),
    "verify_premortem_shape": _shape_check_spec(
        "verify_premortem_shape", "Premortem doc shape."),
    "verify_screen_consistency": _shape_check_spec(
        "verify_screen_consistency",
        "Cross-screen inherits_shell consistency across a chunk's screen plans "
        "(reads the .screens/ dir). Multi-producer: attached to the LAST "
        "screen-plan producer; reads all sibling plans from the shared dir."),
    "verify_adr_shape": _shape_check_spec(
        "verify_adr_shape", "ADR decision JSON shape (+ schema version)."),
    "verify_cost_curve_present": _shape_check_spec(
        "verify_cost_curve_present", "ADR carries a cost-curve section."),
    "verify_adr_register": _shape_check_spec(
        "verify_adr_register", "ADR register.md is consistent with the .adr dir."),
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
    # Z3 T3A — security_review composite (semgrep + bandit + npm audit).
    "security_review": PostHookSpec(
        kind="security_review",
        verb="security_review",
        default_severity="blocker",
        cost_band="moderate",
        # Callable: empty list when qa_dial=off; else source-file globs.
        # Accepts MissionDialContext OR dict OR None (defaults to standard).
        auto_wire_triggers=lambda ctx: (
            []
            if _dial_get(ctx, "qa_dial", "standard") == "off"
            else ["*.py", "*.ts", "*.tsx", "*.js", "*.jsx", "requirements.txt", "package.json"]
        ),
        description=(
            "Composite security gate: semgrep security.yml + bandit (Python) + "
            "npm audit (JS). Aggregated severity-blocker verdict."
        ),
    ),
    # Z3 T3B — accessibility_review via @axe-core/cli against tunneled preview.
    "accessibility_review": PostHookSpec(
        kind="accessibility_review",
        verb="run_axe",
        default_severity="blocker",
        cost_band="heavy",
        # Callable: when accessibility_dial=on → JSX/TSX globs; else empty.
        # Accepts MissionDialContext OR dict OR None.
        auto_wire_triggers=lambda ctx: (
            ["*.tsx", "*.jsx"]
            if _dial_get(ctx, "accessibility_dial", "off") == "on"
            else []
        ),
        description=(
            "axe-core a11y scan against tunneled preview URL. "
            "critical/serious impact → blocker; moderate → warning; minor → info."
        ),
    ),
    # Z3 T3C — contract_review via schemathesis.
    "contract_review": PostHookSpec(
        kind="contract_review",
        verb="run_schemathesis",
        default_severity="blocker",
        cost_band="moderate",
        auto_wire_triggers=lambda ctx: (
            []
            if _dial_get(ctx, "qa_dial", "standard") == "off"
            else ["**/routes/*.py", "**/routers/*.py", "**/api/*.py"]
        ),
        description=(
            "schemathesis contract fuzz against running app via openapi spec. "
            "Any 5xx OR schema mismatch → blocker."
        ),
    ),
    # Z3 T3C — performance_review (lighthouse OR k6 by payload mode).
    "performance_review": PostHookSpec(
        kind="performance_review",
        verb="performance_review",
        default_severity="blocker",
        cost_band="heavy",
        # Opt-in only via explicit post_hooks: ["performance_review"].
        auto_wire_triggers=[],
        description=(
            "lighthouse (web) or k6 (api) perf gate. Mode + thresholds in step "
            "context. Threshold breach → blocker."
        ),
    ),
    # Z3 R3 — adr_drift_judge (LLM gray-zone consumer for ADRs without
    # mechanical falsification_signals). Spawned by apply.py after a passing
    # mechanical adr_drift_check that flagged judgment_only_adr_ids — never
    # auto-wired directly.
    "adr_drift_judge": PostHookSpec(
        kind="adr_drift_judge",
        verb="adr_drift_judge",
        default_severity="blocker",
        cost_band="moderate",
        auto_wire_triggers=[],
        description=(
            "LLM verdict on ADRs whose falsification_signal is a free-form "
            "sentence / null / unknown shape. Spawned only by apply.py after "
            "the mechanical check returns judgment_only_adr_ids."
        ),
    ),
    # Z3 T5 — integration_replay (rerun suite against prior commits in shuffle; bisect on red).
    "integration_replay": PostHookSpec(
        kind="integration_replay",
        verb="integration_replay",
        default_severity="blocker",
        cost_band="heavy",
        # Expander-injected only — sibling of integration_review on parent feature step.
        auto_wire_triggers=[],
        description=(
            "Re-run test suite against current commit + N prior feature commits "
            "in random shuffle; bisect on fail emits mission_lessons row."
        ),
    ),
    # Z5 T4b — mobile_smoke (Maestro mobile-QA flow gate).
    "mobile_smoke": PostHookSpec(
        kind="mobile_smoke",
        verb="maestro",
        default_severity="blocker",
        # Heavy: drives a long-running Maestro flow against a launched app
        # (sign in → onboard → core action → sign out).
        cost_band="heavy",
        # No glob auto-wire: Maestro needs an already-running app + a flow
        # YAML, so mobile_smoke is opted in explicitly per step (the mobile
        # recipes ship a smoke-flow .yaml and declare the hook).
        auto_wire_triggers=[],
        description=(
            "Run a recipe-driven Maestro mobile-QA flow (sign in → onboard → "
            "core action → sign out) against the running app; gate the step "
            "on the Maestro exit code / failed-flow count. Soft-skipped when "
            "the Maestro CLI is not installed."
        ),
    ),
    # Z3 T4B — adr_drift_check (mechanical violation gate against ADR falsification_signal).
    "adr_drift_check": PostHookSpec(
        kind="adr_drift_check",
        verb="check_adr_drift",
        default_severity="blocker",
        cost_band="cheap",
        # When qa_dial != "off", scan all code files for ADR violations.
        # Verb soft-skips when mission has no .adr/ directory.
        auto_wire_triggers=lambda ctx: (
            []
            if _dial_get(ctx, "qa_dial", "standard") == "off"
            else ["**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx"]
        ),
        description=(
            "Mechanical drift gate: parse ADR falsification_signal (v2 structured) "
            "and scan produced files for forbidden imports/patterns/coverage gaps."
        ),
    ),
    # ── Z7 T1.0: humanish-layers posthook reservations ──────────────────────
    # Four new kinds, each with an isolated stub handler in
    # posthook_handlers/<kind>.py so parallel feature agents (A0, A5, A6,
    # B5/B9) can each edit exactly one file without merge conflicts.
    # All are opt-in (no auto_wire_triggers) — workflows declare them
    # explicitly on steps that produce outbound copy / briefing artifacts.
    "briefing_compose": PostHookSpec(
        kind="briefing_compose",
        verb="briefing_compose",
        default_severity="warning",   # advisory; doesn't block the source step
        cost_band="moderate",
        auto_wire_triggers=[],
        description=(
            "Z7 T1.0 (A0): compose a mission briefing row in mission_briefings "
            "for the founder after a major milestone completes. "
            "Stub handler: posthook_handlers/briefing_compose.py."
        ),
    ),
    "brand_voice_lint": PostHookSpec(
        kind="brand_voice_lint",
        verb="brand_voice_lint",
        default_severity="blocker",
        cost_band="moderate",
        auto_wire_triggers=[],
        description=(
            "Z7 T1.0 (A5): lint produced copy artifacts against brand-voice rules "
            "from docs/templates/brand_voices/. Fails on prohibited terms or "
            "out-of-range Flesch-Kincaid/sentence-length. "
            "Stub handler: posthook_handlers/brand_voice_lint.py."
        ),
    ),
    "copy_compliance_review": PostHookSpec(
        kind="copy_compliance_review",
        verb="copy_compliance_review",
        default_severity="blocker",
        cost_band="moderate",
        auto_wire_triggers=[],
        description=(
            "Z7 T1.0 (A6): review produced copy for channel-specific compliance "
            "(disclaimers, banned words, max-length) using "
            "docs/templates/channel_rules/. "
            "Stub handler: posthook_handlers/copy_compliance_review.py."
        ),
    ),
    "audit_completeness_check": PostHookSpec(
        kind="audit_completeness_check",
        verb="audit_completeness_check",
        default_severity="warning",
        cost_band="cheap",
        auto_wire_triggers=[],
        description=(
            "Z7 T1.0 (B5/B9): assert every external send (action_confirmation "
            "with reversibility != 'full') produced an external_comms_log row; "
            "surface gaps to founder as an advisory warning. "
            "Handler: posthook_handlers/audit_completeness_check.py."
        ),
    ),
    # ── Z7 T3B: demo pipeline posthooks (A3 + A3.r1) ────────────────────────
    "demo_artifact_check": PostHookSpec(
        kind="demo_artifact_check",
        verb="demo_artifact_check",
        default_severity="blocker",
        cost_band="cheap",
        auto_wire_triggers=[],
        description=(
            "Z7 T3B (A3): verify demo pipeline output — all cut files exist "
            "(cuts/30s.mp4, cuts/60s.mp4, cuts/3min.mp4), duration within ±10% "
            "of target, and .vtt captions file is present. "
            "Handler: posthook_handlers/demo_artifact_check.py."
        ),
    ),
    "demo_accessibility_check": PostHookSpec(
        kind="demo_accessibility_check",
        verb="demo_accessibility_check",
        default_severity="blocker",
        cost_band="cheap",
        auto_wire_triggers=[],
        description=(
            "Z7 T3B (A3.r1): validate demo accessibility manifest completeness — "
            "alt_texts non-empty, audio_description_track present, "
            "keyboard_nav_variant non-empty. "
            "Handler: posthook_handlers/demo_accessibility_check.py."
        ),
    ),
    # ── Z7 T3C: press kit posthook (A4 + A4.r1) ─────────────────────────────
    "press_kit_freshness": PostHookSpec(
        kind="press_kit_freshness",
        verb="press_kit_freshness",
        default_severity="warning",
        cost_band="cheap",
        auto_wire_triggers=[],
        description=(
            "Z7 T3C (A4): monthly freshness check — if any press kit for the product "
            "is >90 days old AND the spec has changed, surfaces a founder_action "
            "'regenerate press kit?'. "
            "Handler: posthook_handlers/press_kit_freshness.py."
        ),
    ),
    # ── Z7 T3A: launch readiness gate (A2 + A2.r1) ───────────────────────────
    # Pre-T-0 blocker: 7 checks must pass before publish_synchronized fires.
    # Severity: blocker — T-0 must not fire without all hard checks green.
    # Absent subsystems degrade to non-blocking warnings (not crash).
    # Handler: posthook_handlers/launch_readiness_gate.py
    "launch_readiness_gate": PostHookSpec(
        kind="launch_readiness_gate",
        verb="launch_readiness_gate",
        default_severity="blocker",
        cost_band="moderate",
        auto_wire_triggers=[],
        description=(
            "Z7 T3A (A2.r1): pre-T-0 readiness gate — 7 checks: "
            "(a) site load, (b) payment E2E, (c) support FAQ indexed, "
            "(d) A6 copy compliance, (e) A4 press kit published, "
            "(f) B3 status page, (g) B6 crisis playbook tier1+tier2. "
            "Absent subsystem → warning (not crash). "
            "Any blocker → freeze T-0, surface founder_action 'override or fix?'. "
            "Handler: posthook_handlers/launch_readiness_gate.py."
        ),
    ),
    # ── Z7 T3D: incident comms founder-review gate (B3) ──────────────────────
    # Fires after incident/draft_update to gate publication on founder approval.
    # Severity: blocker — status update must not be published without review.
    # TODO: Add max 4hr SLA timer (B3 spec): if founder unavailable >4h,
    #   B6 crisis playbook takes over. Implement SLA timer when B6 lands in T3E.
    "incident_update_review": PostHookSpec(
        kind="incident_update_review",
        verb="incident_update_review",
        default_severity="blocker",
        cost_band="cheap",
        auto_wire_triggers=[],
        description=(
            "Z7 T3D (B3): founder-review gate before publishing a customer-facing "
            "status update. Emits a founder_action with the draft text; blocks "
            "incident/publish_status until approved. "
            "TODO: 4hr SLA timer → B6 crisis playbook escalation (T3E). "
            "Handler: posthook_handlers/incident_update_review.py."
        ),
    ),
    # ── Z7 T4 A8: documentation_gap_detect ───────────────────────────────────
    # Fires on every support escalation. Semantic-searches the question against
    # the per-language support_docs Chroma collection. No match → docs_gap_log
    # row. Weekly faq_regen surfaces clusters; A0 briefing appends digest.
    # Severity: warning (advisory — doesn't block the escalation path).
    "documentation_gap_detect": PostHookSpec(
        kind="documentation_gap_detect",
        verb="documentation_gap_detect",
        default_severity="warning",
        cost_band="cheap",
        auto_wire_triggers=[],
        description=(
            "Z7 T4 A8: fire on support escalation; semantic-search question against "
            "per-language support_docs_<lang> Chroma collection; write docs_gap_log "
            "row if no match found. Weekly faq_regen surfaces gap clusters. "
            "Handler: posthook_handlers/documentation_gap_detect.py."
        ),
    ),
    # ── Z7 T5 B2: changelog_freshness ────────────────────────────────────────
    # Fires monthly (or on-demand). Checks whether each completed
    # goal:public_release mission has a corresponding changelog_entries row.
    # If missing, surfaces a founder_action to publish one.
    "changelog_freshness": PostHookSpec(
        kind="changelog_freshness",
        verb="changelog_freshness",
        default_severity="warning",
        cost_band="cheap",
        auto_wire_triggers=[],
        description=(
            "Z7 T5 B2: monthly changelog freshness check. "
            "If any goal:public_release mission has no changelog_entries row, "
            "surfaces a founder_action 'publish changelog entry?'. "
            "Handler: posthook_handlers/changelog_freshness.py."
        ),
    ),
    # ── Z7 T6 A7: outreach_deliverability_check ──────────────────────────────
    # Fires after an outreach/send batch completes. Checks bounce rate + complaint
    # rate for the product+list. If thresholds exceeded, pauses campaign and
    # surfaces a founder_action. Severity: warning (advisory; doesn't block
    # the send batch itself — retrospective check).
    "outreach_deliverability_check": PostHookSpec(
        kind="outreach_deliverability_check",
        verb="outreach_deliverability_check",
        default_severity="warning",
        cost_band="cheap",
        auto_wire_triggers=[],
        description=(
            "Z7 T6 A7: post-send deliverability health check. "
            "Bounce rate >5% or complaint rate >0.1% → campaign paused + "
            "founder_action surfaced. Read-only scan; advisory verdict. "
            "Handler: mr_roboto.outreach_deliverability_check."
        ),
    ),
    # Z4 T3A — visual_review via vision-model diff against tunneled preview URL.
    # STATIC trigger list (not dial-gated): founder decision — always auto-wire
    # on frontend produces; the verb soft-skips when no preview URL is available.
    "visual_review": PostHookSpec(
        kind="visual_review",
        verb="visual_review",
        default_severity="blocker",
        cost_band="heavy",
        auto_wire_triggers=["*.tsx", "*.jsx", "*.vue", "*.svelte"],
        description=(
            "vision-model diff against tunneled preview URL. "
            "Color/layout/font-size/missing-component blockers fail; "
            "shadow + micro-spacing info does not block."
        ),
    ),
    # Yalayut Phase 4 — internal-hint auto-capture. Replaces the old
    # src/memory/skills.py auto-capture. Auto-wired on every gradeable task
    # via the "*" trigger; the executor itself no-ops on <2-iteration or
    # failed tasks. Warning severity — a capture miss must never DLQ the
    # source task.
    "capture_hint": PostHookSpec(
        kind="capture_hint",
        verb="capture_hint",
        default_severity="warning",
        cost_band="cheap",
        auto_wire_triggers=["*"],
        description=(
            "Yalayut P4 — capture a successful 2+-iteration task as an "
            "internal_hint artifact in yalayut_index. Mechanical; advisory."
        ),
    ),
    # Z3 T4C — domain_layer_check via layer-filtered semgrep.
    # Fires on any step that produces a domain-layer Python file.
    # Delegates to run_semgrep_layer_filtered (mr_roboto), which filters
    # target_files to only domain-layer paths before running semgrep with
    # forbidden_in_domain.yml. Soft-skips on semgrep crash (same pattern
    # as pattern_lint). Real findings cascade the source (blocker).
    "domain_layer_check": PostHookSpec(
        kind="domain_layer_check",
        verb="run_semgrep_layer_filtered",
        default_severity="blocker",
        cost_band="cheap",
        # Match any file produced under a domain/ directory.
        # fnmatch.fnmatchcase treats * as matching slashes (Python fnmatch
        # implementation), so **/domain/*.py matches src/domain/user.py and
        # app/domain/models/user.py; src/domain/**/*.py covers nested files.
        auto_wire_triggers=["**/domain/*.py", "src/domain/**/*.py"],
        description=(
            "Run semgrep with forbidden_in_domain.yml rule pack via the layer-filtered "
            "wrapper. Only files that resolve to the 'domain' architectural layer are "
            "scanned; infra/adapter/test files are skipped. Blocker: findings with "
            "ERROR severity (e.g. requests/sqlalchemy imports in domain code) cause "
            "a retry-with-feedback. Soft-dropped when semgrep is absent or crashes."
        ),
    ),
    # Z2 cross-mission lessons — READ side. The expander prepends this
    # kind to the first phase-0 task's post_hooks (workflow_engine
    # expander.py); the coulson consumer renders "Watch out for" from
    # lessons_top_n. Without a registry entry, determine_posthooks() was
    # dropping it and the loop ran open. (Sweep handoff 2026-05-18, Z2 P1.)
    "inject_lessons": PostHookSpec(
        kind="inject_lessons",
        verb="inject_lessons",
        default_severity="warning",
        cost_band="cheap",
        auto_wire_triggers=[],
        description=(
            "Z2 cross-mission learning — pull top mission_lessons rows and "
            "inject them into the next mission's first task context."
        ),
    ),
    # SP3b: constrained_emit — re-emit draft as schema-conforming JSON.
    # Result-rewriting post-hook (replaces tasks.result). Runs before grade.
    # Auto-wire is schema-driven (Task 6), not produces-glob — empty triggers.
    "constrained_emit": PostHookSpec(
        kind="constrained_emit",
        verb="constrained_emit",   # routed by _enqueue_posthook_llm_child (Task 5)
        default_severity="blocker",
        cost_band="moderate",
        # Auto-wired by determine_posthooks on steps that declare a constrainable
        # artifact_schema (Task 6). No glob trigger (schema lives in ctx, not produces).
        auto_wire_triggers=[],
        description=(
            "SP3b: re-emit the draft as schema-conforming JSON (response_format). "
            "Result-rewriting post-hook (replaces tasks.result). Runs before grade."
        ),
    ),
    # SP3b: self_reflect — role-specific self-review of the draft.
    # Result-rewriting when verdict=fix (applies corrected_result).
    # Runs after emit, before grade.
    "self_reflect": PostHookSpec(
        kind="self_reflect",
        verb="self_reflect",
        default_severity="warning",
        cost_band="moderate",
        auto_wire_triggers=[],
        description=(
            "SP3b: role-specific self-review of the draft. Result-rewriting when "
            "verdict=fix (applies corrected_result). Runs after emit, before grade."
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


def _emit_is_constrainable(artifact_schema: dict, task_ctx: dict) -> bool:
    """True when *artifact_schema* can be turned into a json_schema
    response_format (i.e. has at least one object/array artifact).

    Reuses ``src.core.reflection_posthook.schema_response_format`` — the same
    constrainable check the emit child itself applies (None ⇒ markdown/string ⇒
    skip the emit, let the validator cover shape). Best-effort: if the helper
    can't be imported (lean test env), fall back to a light object/array probe
    so the ordering still wires emit ahead of grade.
    """
    try:
        from src.core.reflection_posthook import schema_response_format
        step_id = task_ctx.get("workflow_step_id") or "artifact"
        return schema_response_format(artifact_schema, step_id) is not None
    except Exception:  # noqa: BLE001 — never let a constrainability probe crash
        return any(
            isinstance(r, dict) and r.get("type") in ("object", "array")
            for r in artifact_schema.values()
        )


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

    SP3b Task 6 ordering — the two RESULT-REWRITING post-hooks run FIRST so
    their rewrites land on the source before the terminal grade gate reads it:
      1. ``constrained_emit`` — when the step declares a *constrainable*
         artifact_schema (object/array; markdown/string is unconstrainable and
         covered by the validator instead).
      2. ``self_reflect`` — when the step opted into self-reflection.
      3. ``grade`` — the existing terminal gate, always AFTER the rewriters.
      4. explicit extras (verify_artifacts/...), unchanged tail position.
    The apply layer sequences 1→2→3 as a cursor walk (``_posthook_queue``); the
    rewriters do not gate, only grade + extras do.
    """
    agent_type = task.get("agent_type", "")
    no_posthooks = agent_type in _NO_POSTHOOKS_AGENT_TYPES

    kinds: list[str] = []

    # 1) constrained_emit — rewrite the draft as schema-conforming JSON.
    # Auto-wired only when the step carries a CONSTRAINABLE artifact_schema
    # (object/array). schema_response_format returns None for markdown/string
    # schemas — those are not re-emittable, so the emit kind is skipped and the
    # downstream validator covers shape. Recursion-guarded agent types
    # (constrained_emit/self_reflect children) never re-wire emit on themselves.
    if not no_posthooks:
        artifact_schema = task_ctx.get("artifact_schema")
        if isinstance(artifact_schema, dict) and artifact_schema:
            if _emit_is_constrainable(artifact_schema, task_ctx):
                kinds.append("constrained_emit")

    # 2) self_reflect — role-specific self-review of the draft. Opt-in via the
    # task row or its context. Warning severity: rewrites only on verdict=fix,
    # never fails the source.
    if not no_posthooks and (
        task.get("enable_self_reflection")
        or task_ctx.get("enable_self_reflection")
    ):
        kinds.append("self_reflect")

    # 3) Default `grade` hook — suppressed for the no-posthooks agent types
    # (mechanical / reviewers etc.: a grader on a mechanical result is
    # pointless, a grader on a reviewer is judge-of-judge). Always AFTER the
    # rewriters so it grades the rewritten result.
    if not no_posthooks and task_ctx.get("requires_grading") is not False:
        kinds.append("grade")

    # Extra kinds EXPLICITLY declared on the step. These are honoured even
    # for the no-posthooks agent types: a mechanical step that declares
    # `incident_update_review` (Z7 B3 founder-review gate) must still spawn
    # it — skipping it silently bypasses a trust-critical gate. Only the
    # implicit `grade` hook is suppressed for those agent types, never an
    # explicitly-requested hook. Filter to registry so a typo doesn't spawn
    # a task with no handler.
    extra = task_ctx.get("post_hooks") or []
    if isinstance(extra, list):
        for k in extra:
            if not isinstance(k, str) or not k.strip():
                continue
            if k in POST_HOOK_REGISTRY and k not in kinds:
                kinds.append(k)

    # Separate pot — `checks`: parameterized mechanical verifiers that carry
    # their own payload (converted standalone `.verify` steps). Each entry is
    # ``{"kind": ..., "payload": {...}}``; the kind enters the pending list
    # exactly like an explicit post_hook (payload is read later by the apply
    # layer from ``checks``). Kept separate from `post_hooks` so that field
    # stays a pure list[str] — no str|dict union.
    checks = task_ctx.get("checks") or []
    if isinstance(checks, list):
        for entry in checks:
            if not isinstance(entry, dict):
                continue
            k = entry.get("kind")
            if isinstance(k, str) and k in POST_HOOK_REGISTRY and k not in kinds:
                kinds.append(k)
    return kinds
