"""Expand workflow definitions into concrete tasks for the orchestrator.

Handles v2 features: Phase -1 conditional inclusion, recurring step types,
template expansion with context_strategy, and agent name mapping.
"""

from __future__ import annotations

import fnmatch
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from general_beckman.posthooks import MissionDialContext

from src.infra.logging_config import get_logger

logger = get_logger("workflows.engine.expander")

DIFFICULTY_MAP: dict[str, int] = {
    "easy": 3,
    "medium": 6,
    "hard": 8,
}

# Maps workflow agent names to system agent types.
# Most map 1:1; only special case is router -> executor.
AGENT_MAP: dict[str, str] = {
    "router": "executor",
}


def map_agent_type(agent_name: str) -> str:
    """Map a workflow agent name to the system agent type.

    Uses :data:`AGENT_MAP` for known overrides; unmapped names pass through
    unchanged.
    """
    return AGENT_MAP.get(agent_name, agent_name)


def _is_valid_produces_entry(entry) -> bool:
    """A produces slot is either a non-empty string (literal/glob) or a
    non-empty list of non-empty strings (any_of alternatives)."""
    if isinstance(entry, str):
        return bool(entry.strip())
    if isinstance(entry, list):
        return bool(entry) and all(
            isinstance(x, str) and x.strip() for x in entry
        )
    return False


def filter_steps_for_context(
    steps: list[dict],
    has_existing_codebase: bool = False,
) -> list[dict]:
    """Filter steps based on project context.

    If *has_existing_codebase* is ``False``, steps belonging to
    ``phase_-1`` are excluded (they only apply when onboarding an
    existing project).  Otherwise all steps are returned.
    """
    if has_existing_codebase:
        return list(steps)
    return [s for s in steps if s.get("phase") != "phase_-1"]


def _phase_to_priority(phase: str) -> int:
    """Derive a numeric priority from a phase ID.

    Phase -1 and 0 receive priority 10 (highest).  Phase 15 receives
    priority 1 (lowest).  Intermediate phases are linearly interpolated.
    """
    # Extract the numeric part from "phase_N" or "phase_-1"
    try:
        phase_num = int(phase.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 5  # default for unparseable phases

    if phase_num <= 0:
        return 10
    if phase_num >= 15:
        return 1
    # Linear interpolation: phase 1 -> 9, phase 14 -> 2
    return max(1, 10 - phase_num)


def _auto_wire_posthooks(
    context: dict,
    dial_ctx: Optional["MissionDialContext"] = None,
) -> None:
    """Prepend post-hook kinds whose registry triggers match the step's produces.

    Iterates ``POST_HOOK_REGISTRY`` in insertion order.  For each spec,
    resolves its ``auto_wire_triggers`` (static list *or* callable) via
    :meth:`PostHookSpec.resolve_triggers`, then checks whether any returned
    glob (fnmatch-style) matches any entry in ``context["produces"]``.
    Produces entries that are lists (any_of alternatives) have each
    alternative tested individually — if any alternative matches, the
    trigger fires.

    Prepends matched kinds so cheapest checks (grounding) fire before
    more-expensive ones (verify_artifacts, code_review).  Idempotent: a kind
    already present is never duplicated.

    Called only when ``context["produces"]`` is non-empty.

    Parameters
    ----------
    context:
        Mutable step context dict.  ``post_hooks`` is updated in-place.
    dial_ctx:
        Optional :class:`~general_beckman.posthooks.MissionDialContext`
        forwarded to callable ``auto_wire_triggers`` so the trigger set can
        vary with founder dial settings (T1C).  When ``None``, the default
        conservative context is used — behaviour is identical to the static
        form until real dials are wired.
    """
    from general_beckman.posthooks import (  # lazy import avoids circular
        POST_HOOK_REGISTRY,
        MissionDialContext as _MDC,
    )

    _dial = dial_ctx if dial_ctx is not None else _MDC()

    produces: list = context.get("produces") or []
    existing: list[str] = list(context.get("post_hooks") or [])

    # Flatten produces to candidate path strings for glob matching.
    # any_of entries (list of strings) contribute all alternatives.
    candidate_paths: list[str] = []
    for entry in produces:
        if isinstance(entry, str):
            candidate_paths.append(entry)
        elif isinstance(entry, list):
            candidate_paths.extend(s for s in entry if isinstance(s, str))

    to_prepend: list[str] = []
    for spec in POST_HOOK_REGISTRY.values():
        resolved_triggers = spec.resolve_triggers(_dial)
        if not resolved_triggers:
            continue
        if spec.kind in existing or spec.kind in to_prepend:
            continue
        for trigger in resolved_triggers:
            if any(fnmatch.fnmatchcase(p, trigger) for p in candidate_paths):
                to_prepend.append(spec.kind)
                break

    if to_prepend:
        context["post_hooks"] = to_prepend + existing


def _apply_hint_from_targets(
    context: dict,
    workspace_path: Optional[str] = None,
) -> None:
    """Strip ``write_file`` from tools_hint when a produce target already exists.

    If the step's ``tools_hint`` contains ``"write_file"`` AND any path in
    ``context["produces"]`` already exists under *workspace_path*, remove
    ``"write_file"`` from the hint so the agent is nudged toward patch/edit
    tools instead of blindly overwriting.

    Decision — any_of alternatives (list of strings): if *any* of the
    alternatives exists, ``write_file`` is stripped.  Rationale: an any_of
    slot represents "one of these files will satisfy the requirement"; if one
    already exists the agent should edit, not overwrite.

    Founder override: ``context["force_write"] = True`` skips the strip.

    Idempotent: calling twice yields the same result.
    Does nothing when ``tools_hint`` is absent or doesn't include
    ``"write_file"``.
    """
    tools_hint = context.get("tools_hint")
    if not tools_hint or "write_file" not in tools_hint:
        return
    if context.get("force_write"):
        return

    produces: list = context.get("produces") or []
    if not produces:
        return

    # Resolve workspace path: if not supplied, no filesystem check is possible
    # so we skip (test callers pass tmp_path explicitly; production callers
    # should pass the mission workspace root).
    if workspace_path is None:
        return

    def _exists(rel: str) -> bool:
        return os.path.exists(os.path.join(workspace_path, rel))

    should_strip = False
    for entry in produces:
        if isinstance(entry, str):
            if _exists(entry):
                should_strip = True
                break
        elif isinstance(entry, list):
            if any(isinstance(alt, str) and _exists(alt) for alt in entry):
                should_strip = True
                break

    if should_strip:
        context["tools_hint"] = [t for t in tools_hint if t != "write_file"]


def expand_steps_to_tasks(
    steps: list[dict],
    mission_id: str,
    initial_context: Optional[dict] = None,
) -> list[dict]:
    """Convert workflow step dicts into task dicts for DB insertion.

    Parameters
    ----------
    steps:
        List of step dicts from a :class:`WorkflowDefinition`.
    mission_id:
        The mission ID to associate each task with.
    initial_context:
        Optional dict of initial context (e.g. user idea) to propagate
        into each task's ``workflow_context``.

    Returns
    -------
    list[dict]
        Task dicts ready for DB insertion.  ``depends_on_steps`` contains
        step ID strings; the runner resolves these to actual DB task IDs.
    """
    tasks: list[dict] = []

    # Mission-level template params applied uniformly to every step's
    # `instruction`, `produces`, `payload`, and `done_when` fields.
    # Without this pass the literal `{mission_id}` placeholder survives
    # into the agent prompt + grounding guard's produces list, the agent
    # invents a path (drops `.charter/`), and the verify post-hook reads
    # the unsubstituted template path — both fail (mission 69 step 0.0z,
    # 2026-05-14).
    _mission_params = {"mission_id": str(mission_id) if mission_id is not None else ""}

    for raw_step in steps:
        step = _substitute_payload(raw_step, _mission_params)
        step_id = step["id"]
        phase = step.get("phase", "phase_0")

        context: dict = {
            "workflow_step_id": step_id,
            "step_name": step.get("name", ""),
            "workflow_phase": phase,
            "input_artifacts": step.get("input_artifacts", []),
            "output_artifacts": step.get("output_artifacts", []),
            "may_need_clarification": step.get("may_need_clarification", False),
            "is_workflow_step": True,
        }

        # Optional fields — only include if present on the step
        if "condition" in step:
            context["condition"] = step["condition"]
        if "type" in step:
            context["step_type"] = step["type"]
        if "trigger" in step:
            context["trigger"] = step["trigger"]
        if "done_when" in step:
            context["done_when"] = step["done_when"]
        if initial_context is not None:
            context["workflow_context"] = initial_context

        # v3 fields — difficulty, tools_hint, artifact_schema, skip_when
        difficulty = step.get("difficulty")
        if difficulty and difficulty in DIFFICULTY_MAP:
            context["difficulty"] = DIFFICULTY_MAP[difficulty]
            if difficulty == "hard":
                context["needs_thinking"] = True
                context["prefer_quality"] = True

        tools_hint = step.get("tools_hint")
        if tools_hint and isinstance(tools_hint, list):
            context["tools_hint"] = tools_hint

        api_hints = step.get("api_hints")
        if api_hints and isinstance(api_hints, list):
            context["api_hints"] = api_hints

        artifact_schema = step.get("artifact_schema")
        if artifact_schema and isinstance(artifact_schema, dict):
            context["artifact_schema"] = artifact_schema

        # Files this step is supposed to write under the mission workspace.
        # Consumed by the verify_artifacts post-hook (mechanical) which checks
        # each path exists, is non-empty, and (for known extensions) parses.
        # Entries may be:
        #   - string: literal path or glob (``*``/``?``/``[]``)
        #   - list of strings: ``any_of`` slot — at least one must match
        produces = step.get("produces")
        if produces and isinstance(produces, list):
            context["produces"] = [
                p for p in produces if _is_valid_produces_entry(p)
            ]

        # Post-hooks declared on the step. Combined with the default ["grade"]
        # (or [] when policy excludes it) by determine_posthooks.
        post_hooks = step.get("post_hooks")
        if post_hooks and isinstance(post_hooks, list):
            context["post_hooks"] = [k for k in post_hooks if isinstance(k, str) and k.strip()]

        # Auto-wire post-hooks from registry (T1B).
        # For each registered kind whose auto_wire_triggers are non-empty,
        # prepend that kind when any trigger glob matches any produces entry.
        # The existing grounding block is *migrated into this loop* rather
        # than kept as a separate branch — "grounding" has auto_wire_triggers=["*"]
        # which matches every produces entry, preserving identical behavior.
        # Idempotent: a kind already present in post_hooks is never added again.
        if context.get("produces"):
            _auto_wire_posthooks(context)

        # Hint-from-targets pass (T1C): if tools_hint includes "write_file"
        # and any declared produce path already exists in the workspace,
        # strip "write_file" so the agent is nudged toward patch/edit tools.
        # Z2 Item-3 followup — thread workspace_path so the pass actually
        # activates in production (was no-op while workspace_path=None).
        _ws_path: Optional[str] = None
        try:
            if mission_id:
                from src.tools.workspace import WORKSPACE_DIR
                import os.path as _osp
                _candidate = _osp.join(WORKSPACE_DIR, f"mission_{mission_id}")
                if _osp.isdir(_candidate):
                    _ws_path = _candidate
        except Exception:
            _ws_path = None
        _apply_hint_from_targets(context, workspace_path=_ws_path)

        skip_when = step.get("skip_when")
        if skip_when:
            if isinstance(skip_when, list):
                context["skip_when"] = skip_when
            elif isinstance(skip_when, str):
                # String expression form: evaluated at dispatch time against
                # loaded artifacts. Shopping workflows use this shape
                # (e.g. "gate_result.gate.kind != 'chosen'") — previously
                # silently dropped here because the branch above only
                # accepted lists, and the step then ran regardless of the
                # gate outcome, producing empty output that downstream
                # grading rejected as quality failures.
                context["skip_when_expr"] = skip_when

        if step.get("triggers_clarification"):
            context["triggers_clarification"] = True

        # Z6 T1A: real-world bridge flags. These all need to make it onto
        # task.context so beckman admission (T1C) and add_task's column
        # hoist see them. ``needs_real_tools`` and ``reversibility`` are
        # additionally hoisted to indexed columns by add_task.
        if step.get("needs_real_tools"):
            context["needs_real_tools"] = True
        if step.get("reversibility") in ("full", "partial", "irreversible"):
            context["reversibility"] = step["reversibility"]
        if "real_tool_kind" in step:
            context["real_tool_kind"] = step["real_tool_kind"]
        if "cost_estimate_usd" in step:
            context["cost_estimate_usd"] = step["cost_estimate_usd"]

        # Propagate any step-level `context` dict from the workflow JSON.
        # Without this, fields like `per_site_n`, `max_groups`, or
        # `requires_grading` declared on a step are silently dropped.
        # Merge last so step JSON overrides computed defaults.
        step_ctx = step.get("context")
        if isinstance(step_ctx, dict):
            for k, v in step_ctx.items():
                context[k] = v

        # Mechanical-executor steps (mr_roboto): propagate executor tag + payload
        # into context so the orchestrator can route them without an LLM call.
        agent_name = step.get("agent", "executor")
        if step.get("executor") == "mechanical" or agent_name == "mechanical":
            # Fallback: when the step's context used the legacy shape
            # `{"executor": "<action>", ...}` (e.g. clarify_variant's
            # `{"executor": "clarify", "kind": "variant_choice", ...}`),
            # translate it into the canonical _mechanical_context shape
            # BEFORE we overwrite context["executor"] below. Otherwise
            # mr_roboto.run receives no `action` and fails with
            # `unknown mechanical action: None`.
            if (
                "payload" not in step
                and "payload" not in context
                and isinstance(step_ctx, dict)
            ):
                _legacy_action = step_ctx.get("executor")
                if _legacy_action and _legacy_action != "mechanical":
                    _skip = {"executor", "payload"}
                    extras = {k: v for k, v in step_ctx.items() if k not in _skip}
                    context["payload"] = {"action": _legacy_action, **extras}
            context["executor"] = "mechanical"
            if "payload" in step:
                context["payload"] = step["payload"]

        # Phase D — orchestrator dispatches by task.runner.
        # Mechanical steps run mr_roboto (no LLM); everything else is a
        # ReAct-loop agent. Workflow JSON does not currently emit
        # single-call OVERHEAD steps, so 'direct' is unused here.
        _runner = "mechanical" if context.get("executor") == "mechanical" else "react"

        task = {
            "title": f"[{step_id}] {step['name']}",
            "description": step.get("instruction", ""),
            "agent_type": map_agent_type(agent_name),
            "runner": _runner,
            "mission_id": mission_id,
            "depends_on_steps": list(step.get("depends_on", [])),
            "context": context,
            "priority": _phase_to_priority(phase),
            "tier": "auto",
        }

        tasks.append(task)

    # Z2 T4C — auto-wire inject_lessons on the first phase_0 step so
    # cross-mission lessons land in the mission context before any LLM runs.
    tasks = _inject_lessons_at_mission_start(tasks, initial_context)

    return tasks


# ---------------------------------------------------------------------------
# Z3 T2C — multi-file expansion helpers
# ---------------------------------------------------------------------------


def _maybe_expand_multifile(
    step: dict,
    mission_dials,  # MissionDialContext | None
    artifacts: Optional[dict] = None,
) -> "list[dict] | None":
    """Attempt multi-file template expansion for *step*.

    Returns ``None`` when:
    - ``mission_dials`` is ``None``
    - ``mission_dials.multi_file_expansion`` is ``False``
    - No rule exists for the (template_id, stack) combo

    Returns a list of step dicts (N sub-task steps + 1 integration_review
    sibling) when expansion succeeds.

    The integration_review sibling inherits the parent's ``phase`` and
    ``depends_on`` all N sub-task step IDs.

    This function is pure / sync — DB and async calls belong in the
    async wrapper ``expand_steps_to_tasks_with_dials``.
    """
    if mission_dials is None:
        return None
    if not getattr(mission_dials, "multi_file_expansion", False):
        return None

    # Lazy imports to avoid circulars
    from src.workflows.engine.multifile import expand_template, SubTaskSpec

    # Determine template_id and stack
    step_ctx = step.get("context") or {}
    if isinstance(step_ctx, str):
        import json as _json
        try:
            step_ctx = _json.loads(step_ctx)
        except Exception:
            step_ctx = {}

    template_id: str = str(
        step_ctx.get("template_id")
        or step.get("template_id")
        or getattr(mission_dials, "template_id", None)
        or ""
    )
    stack: str = str(
        step_ctx.get("stack_slug")
        or step.get("stack_slug")
        or getattr(mission_dials, "stack_slug", None)
        or (artifacts or {}).get("tech_stack_detected")
        or ""
    ).lower()

    if not template_id or not stack:
        return None

    sub_specs = expand_template(
        template_id=template_id,
        stack=stack,
        parent_step=step,
        artifacts=artifacts,
    )
    if sub_specs is None:
        return None  # No rule — LLM fallback

    parent_id = step.get("id", "")
    parent_phase = step.get("phase", "phase_0")
    parent_depends = list(step.get("depends_on", []))

    sub_steps: list[dict] = []
    sub_step_ids: list[str] = []

    # T1B SubTaskSpec fields: step_id, template_id, target_file, produces,
    # inherited_post_hooks, inherited_from.  Derive other step keys here.
    feature_name = str(step_ctx.get("feature_name") or step_ctx.get("feature") or "")
    feature_slug = feature_name.lower().replace(" ", "_").replace("-", "_")
    role_inference = {
        "model": "coder",
        "schema": "coder",
        "service": "coder",
        "repository": "coder",
        "error_mapper": "coder",
        "fixtures": "test_generator",
        "tests": "test_generator",
        "component": "coder",
        "hook": "coder",
        "story": "coder",
        "test": "test_generator",
    }

    for spec in sub_specs:
        # spec.step_id is already "<parent>.<role>".  Override parent prefix
        # in case parent_id changed since spec built.
        role = spec.step_id.rsplit(".", 1)[-1] if "." in spec.step_id else spec.step_id
        child_id = f"{parent_id}.{role}" if parent_id else role
        sub_step_ids.append(child_id)
        produces_resolved = [
            p.replace("{{feature}}", feature_slug) if feature_slug else p
            for p in spec.produces
        ]
        child_step = {
            "id": child_id,
            "name": f"{role}: {feature_name or parent_id}".strip(),
            "instruction": f"Emit the {role} file for feature {feature_name or '<unnamed>'} ({template_id} template, {stack} stack).",
            "agent": role_inference.get(role, "coder"),
            "phase": parent_phase,
            "depends_on": list(parent_depends),
            "produces": produces_resolved,
            "post_hooks": list(spec.inherited_post_hooks),
            "context": {"feature_name": feature_name, "role": role, "template_id": template_id, "stack": stack},
            "_multifile_parent": parent_id,
        }
        sub_steps.append(child_step)

    # Integration-review sibling: fires after ALL sub-tasks complete.
    # It holds the integration_review post-hook so beckman/apply.py can
    # dispatch the LLM reviewer with signature context injected.
    ir_step_id = f"{parent_id}.integration_review" if parent_id else "integration_review"
    # Collect all produces from sub-tasks for the review context.
    # Use the resolved (feature-substituted) produces from sub_steps so the
    # integration_review payload matches what sub-tasks will actually emit.
    all_produces: list[str] = []
    for s in sub_steps:
        all_produces.extend(s.get("produces", []))

    integration_review_step = {
        "id": ir_step_id,
        "name": f"Integration review: {step.get('name', parent_id)}",
        "instruction": (
            "Review cross-file consistency of the expanded sub-tasks. "
            "Check import contracts, interface alignment, and integration seams."
        ),
        "agent": "integration_reviewer",
        "phase": parent_phase,
        "depends_on": list(sub_step_ids),
        "produces": [],
        "post_hooks": ["integration_review"],
        # Carries the full produces list for the pre-check verb.
        "context": {
            "sub_task_ids": sub_step_ids,
            "parent_step_id": parent_id,
            "all_sub_task_produces": all_produces,
        },
        "_multifile_parent": parent_id,
    }

    out_steps = sub_steps + [integration_review_step]

    # Z3 T5 — integration_replay sibling. Mode driven by review-density dial.
    # When dial == "off" the sibling is skipped entirely; otherwise the mode
    # propagates to the mechanical verb via context.
    replay_mode = getattr(mission_dials, "integration_replay", "standard")
    if replay_mode and replay_mode != "off":
        replay_step_id = (
            f"{parent_id}.integration_replay" if parent_id else "integration_replay"
        )
        # shuffle_seed defaults to parent step id hash for determinism;
        # apply.py overrides with mission_id when wiring the payload.
        out_steps.append({
            "id": replay_step_id,
            "name": f"Integration replay: {step.get('name', parent_id)}",
            "instruction": (
                "Re-run integration suite against current commit (and prior "
                "commits in strict mode). Bisect on fail emits a mission_lessons "
                "row pointing at the breaking commit pair."
            ),
            "agent": "mechanical",
            "executor": "mechanical",
            "phase": parent_phase,
            # Replay depends on review passing first.
            "depends_on": [ir_step_id],
            "depends_on_steps": [ir_step_id],
            "produces": [],
            "post_hooks": ["integration_replay"],
            "context": {
                "parent_step_id": parent_id,
                "mode": replay_mode,
                "integration_replay_mode": replay_mode,
                "integration_suite_glob": "tests/integration/**",
                "payload": {
                    "action": "integration_replay",
                    "mode": replay_mode,
                    "suite_glob": "tests/integration/**",
                },
            },
            "_multifile_parent": parent_id,
        })

    return out_steps


def _inject_lessons_at_mission_start(
    tasks: list[dict],
    initial_context: dict | None,
) -> list[dict]:
    """Idempotently prepend ``inject_lessons`` to the first phase_0 task's
    ``post_hooks`` list (Z2 T4C).

    The verb fires at mission start so the ``lessons_top_n`` bucket is
    populated before any LLM step runs.  When ``initial_context`` carries a
    ``tech_stack_detected`` field that field is forwarded as ``stack``; when
    not available the verb defaults to an empty stack and returns
    ``lessons_count=0`` gracefully.

    Idempotent: skips if the hook is already present.
    """
    _ctx = initial_context or {}
    # Build a stable stack string if available in the initial context.
    _stack = str(_ctx.get("tech_stack_detected") or _ctx.get("tech_stack") or "")

    for task in tasks:
        task_ctx = task.get("context") or {}
        phase = task_ctx.get("workflow_phase", "")
        # Target: first phase_0 task (non-mechanical so it has a natural
        # post-hook window).
        if phase != "phase_0":
            continue
        if task_ctx.get("executor") == "mechanical":
            continue

        existing_hooks: list[str] = list(task_ctx.get("post_hooks") or [])
        if "inject_lessons" in existing_hooks:
            break  # already wired — stop here

        task_ctx["post_hooks"] = ["inject_lessons"] + existing_hooks
        # Store the stack so posthook machinery can forward it to the verb.
        if _stack:
            task_ctx.setdefault("inject_lessons_stack", _stack)
        task["context"] = task_ctx
        logger.debug(
            "inject_lessons_at_mission_start: wired on step=%s stack=%r",
            task_ctx.get("workflow_step_id"),
            _stack,
        )
        break  # wire once — first eligible phase_0 step only

    return tasks


def _substitute_payload(payload: dict, params: dict) -> dict:
    """Recursively str.format every string value in ``payload`` against
    ``params``. Lists/dicts walked; non-string leaves passed through. A
    KeyError/IndexError on an unmatched placeholder leaves the value as-is
    so a typo doesn't crash expansion (validator catches downstream)."""
    def _sub_str(s: str) -> str:
        try:
            return s.format(**(params or {}))
        except (KeyError, IndexError):
            return s

    def _walk(v):
        if isinstance(v, str):
            return _sub_str(v)
        if isinstance(v, list):
            return [_walk(item) for item in v]
        if isinstance(v, dict):
            return {k: _walk(item) for k, item in v.items()}
        return v

    return _walk(payload)


def expand_template(
    template: dict,
    params: dict,
    prefix: str = "",
) -> list[dict]:
    """Expand a template into concrete step dicts.

    Parameters
    ----------
    template:
        A template dict containing ``steps``, ``context_artifacts``,
        and optionally ``context_strategy``.
    params:
        Parameter values to substitute into step instructions.
        Placeholders like ``{feature_name}`` are replaced.
    prefix:
        Prefix for generated step IDs.  If non-empty, IDs become
        ``"{prefix}.{template_step_id}"``.

    Returns
    -------
    list[dict]
        Concrete step dicts with substituted instructions and propagated
        context strategy.
    """
    context_artifacts = template.get("context_artifacts", [])
    context_strategy = template.get("context_strategy")
    expanded: list[dict] = []

    for tpl_step in template.get("steps", []):
        tpl_step_id = tpl_step["template_step_id"]

        # Build the step ID. Defensive separator handling: strip a
        # trailing dot from prefix before joining so callers passing
        # ``"8.<fid>."`` (the pattern in hooks._trigger_template_
        # expansion) and callers passing ``"8.<fid>"`` produce the
        # same shape. Mission 46 phase 8 saw every task titled
        # ``[8.<fid>..feat.X]`` (double dot) because the caller's
        # trailing dot collided with this f-string's separator.
        if prefix:
            step_id = f"{prefix.rstrip('.')}.{tpl_step_id}"
        else:
            step_id = tpl_step_id

        # Parameter substitution in instruction
        instruction = tpl_step.get("instruction", "")
        for param_name, param_value in params.items():
            instruction = instruction.replace(f"{{{param_name}}}", str(param_value))

        # Prefix artifact names with feature_id to avoid collisions
        # across features.  e.g. "backend_service_files" becomes
        # "auth__backend_service_files" for feature_id="auth".
        feature_id = params.get("feature_id", "")
        art_prefix = f"{feature_id}__" if feature_id else ""

        output_arts = [
            f"{art_prefix}{a}" for a in tpl_step.get("output_artifacts", [])
        ]
        # Input artifacts: prefix template-local refs, keep global refs as-is
        template_output_names = set()
        for ts in template.get("steps", []):
            template_output_names.update(ts.get("output_artifacts", []))

        raw_inputs = tpl_step.get("input_artifacts", context_artifacts)
        input_arts = [
            f"{art_prefix}{a}" if a in template_output_names else a
            for a in raw_inputs
        ]

        step: dict = {
            "id": step_id,
            "name": tpl_step.get("name", ""),
            "agent": tpl_step.get("agent", "executor"),
            "instruction": instruction,
            "output_artifacts": output_arts,
            "input_artifacts": input_arts,
        }

        # Propagate condition if present
        if "condition" in tpl_step:
            step["condition"] = tpl_step["condition"]

        # Propagate context_strategy from template
        if context_strategy is not None:
            step["context_strategy"] = dict(context_strategy)

        # Propagate v3 fields from template steps if present
        if "difficulty" in tpl_step:
            step["difficulty"] = tpl_step["difficulty"]
        if "tools_hint" in tpl_step:
            step["tools_hint"] = list(tpl_step["tools_hint"])
        if "artifact_schema" in tpl_step:
            raw_schema = tpl_step["artifact_schema"]
            if art_prefix:
                step["artifact_schema"] = {
                    f"{art_prefix}{k}": v for k, v in raw_schema.items()
                }
            else:
                step["artifact_schema"] = dict(raw_schema)

        # Per-feature path interpolation: a template step may declare
        # produces with placeholders like "{feature_id}/x.py". Substitute
        # the params dict (feature_id, feature_name, etc.) so each feature
        # instance gets its own concrete path list. Preserves the any_of
        # nested-list shape — substitute inside each alternative.
        if "produces" in tpl_step and isinstance(tpl_step["produces"], list):
            substituted = []
            for p in tpl_step["produces"]:
                if isinstance(p, str):
                    try:
                        substituted.append(p.format(**(params or {})))
                    except (KeyError, IndexError):
                        substituted.append(p)
                elif isinstance(p, list):
                    sub_alts: list[str] = []
                    for alt in p:
                        if not isinstance(alt, str):
                            continue
                        try:
                            sub_alts.append(alt.format(**(params or {})))
                        except (KeyError, IndexError):
                            sub_alts.append(alt)
                    if sub_alts:
                        substituted.append(sub_alts)
            step["produces"] = substituted

        if "post_hooks" in tpl_step and isinstance(tpl_step["post_hooks"], list):
            step["post_hooks"] = list(tpl_step["post_hooks"])

        # Mechanical-step payload: template instances need the payload
        # propagated AND parameter-substituted (so e.g. an artifact name
        # baked into the payload as "{feature_id}__staging_deployment_result"
        # resolves to the per-feature concrete name). Without this,
        # per-feature mechanical steps would either lose their action or
        # share an unsubstituted artifact name across all features.
        if "payload" in tpl_step and isinstance(tpl_step["payload"], dict):
            step["payload"] = _substitute_payload(tpl_step["payload"], params)

        # `done_when` and `done_when_or` may reference artifact-prefixed
        # outputs in templates — keep them in line with the prefix scheme.
        # We don't rewrite the strings here (no canonical parser); just
        # propagate as-is so the downstream gate sees them.
        if "done_when" in tpl_step:
            step["done_when"] = tpl_step["done_when"]

        # needs_real_tools marker (drift-guard): propagate for templates that
        # still have a NEEDS-REAL-TOOLS dependency we haven't unblocked yet.
        # Z6 T1A: hoist to indexed column via add_task (also keeps the flag
        # in step + task.context for legacy readers).
        if tpl_step.get("needs_real_tools"):
            step["needs_real_tools"] = True

        # Z6 T1A: reversibility tag (full|partial|irreversible) propagates
        # from template steps. Lives on the task row (indexed) and on
        # task.context for the in-memory consumers.
        _rev = tpl_step.get("reversibility")
        if _rev in ("full", "partial", "irreversible"):
            step["reversibility"] = _rev

        # Z6 T1A: real_tool_kind + cost_estimate_usd hint for the admission
        # gate (T1C). They live in task.context only; admission resolves
        # adapter availability and cost ack from these.
        if "real_tool_kind" in tpl_step:
            step["real_tool_kind"] = tpl_step["real_tool_kind"]
        if "cost_estimate_usd" in tpl_step:
            step["cost_estimate_usd"] = tpl_step["cost_estimate_usd"]

        expanded.append(step)

    return expanded


# ---------------------------------------------------------------------------
# Z3 T2C — async variant that resolves mission dials before expansion
# ---------------------------------------------------------------------------


async def expand_steps_with_multifile(
    steps: list[dict],
    mission_id: str,
    initial_context: Optional[dict] = None,
) -> list[dict]:
    """Like ``expand_steps_to_tasks`` but also applies multi-file expansion.

    Resolves mission dials via ``review_density.get_dials()`` then processes
    each step. Steps for which ``_maybe_expand_multifile`` returns a non-None
    result have their original step replaced by the returned sub-step list
    (which includes the integration_review sibling). The resulting task list
    is then passed through the standard ``expand_steps_to_tasks`` pipeline.

    Callers that do NOT need multi-file expansion (i.e. the dial is False or
    the mission has no rule) get exactly the same output as calling
    ``expand_steps_to_tasks`` directly, so this function is a safe drop-in.

    Parameters
    ----------
    steps, mission_id, initial_context:
        Same as ``expand_steps_to_tasks``.
    """
    # Lazy import to avoid circulars (review_density imports db lazily)
    try:
        from src.workflows.review_density import get_dials, to_mission_dial_context
        raw_dials = await get_dials(mission_id)
        mission_dials = to_mission_dial_context(mission_id, raw_dials)
    except Exception:
        mission_dials = None

    # Extract artifacts from initial_context for template resolution
    artifacts: dict = {}
    if initial_context:
        artifacts = {
            k: initial_context[k]
            for k in ("tech_stack_detected", "tech_stack", "feature_name")
            if k in initial_context
        }

    # Expand multi-file steps BEFORE convert to tasks
    expanded_steps: list[dict] = []
    for step in steps:
        sub = _maybe_expand_multifile(step, mission_dials, artifacts)
        if sub is not None:
            expanded_steps.extend(sub)
            logger.debug(
                "expand_steps_with_multifile: expanded step=%s into %d sub-steps",
                step.get("id"),
                len(sub),
            )
        else:
            expanded_steps.append(step)

    # Run through standard task expansion (post-hook auto-wire, etc.)
    return expand_steps_to_tasks(expanded_steps, mission_id, initial_context)


def filter_skipped_steps(
    steps: list[dict],
    active_skip_conditions: set[str],
) -> tuple[list[dict], list[dict]]:
    """Split steps into (active, skipped) based on skip_when conditions.

    A step is skipped if ANY of its skip_when conditions are in active_skip_conditions.
    """
    active = []
    skipped = []
    for step in steps:
        skip_when = step.get("skip_when", [])
        if skip_when and active_skip_conditions.intersection(skip_when):
            skipped.append(step)
        else:
            active.append(step)
    return active, skipped
