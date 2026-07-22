"""Runtime — multi-call orchestration for LLM tasks.

Architecture spec: docs/superpowers/specs/2026-05-04-runtime-extraction-design.md
Plan: docs/superpowers/plans/2026-05-04-runtime-extraction.md

Public API:
    await execute(profile, task, progress_callback=None) -> dict

Profile is duck-typed (BaseAgent or anything matching the same attribute
surface): name, allowed_tools, max_iterations, execution_pattern,
get_system_prompt(task), enable_self_reflection, min_confidence,
can_create_subtasks, plus the delegate methods that BaseAgent exposes
(_build_full_system_prompt, _build_context, _count_tokens, etc.). The
delegate methods route through the rest of src/runtime/.

Phase A.10 status — execute() is the runtime entry. It owns the setup phase
(DB prompt override, tools_hint override, auto-strip rules, workflow step-
config refresh, _suppress_clarification flag) plus dispatch to react.run and
the restore-on-finally contract for allowed_tools mutations. (The single_shot
path and the inline constrained_emit post-pass were deleted in SP5.)

Phase A.13 will move _build_model_requirements out to fatih_hoca.
Phase A.11 cutover converts profile delegate calls (`profile._X()`) inside
this package to direct runtime imports, leaving BaseAgent as a ~80-LOC
profile shim with execute() pointing here.
"""
from __future__ import annotations

import json
from typing import Callable

from yazbunu import get_logger

from .react import run as _react_run

logger = get_logger("runtime")


_FILE_TOOLS = frozenset({"read_file", "file_tree", "project_info"})
_WEB_TOOLS = frozenset({"web_search", "smart_search", "extract_url"})
# Tools that write content to disk. For steps whose output is expected
# in the `result` field (artifact_schema = object/array/string/markdown),
# exposing these invites small models to call write_file with the output
# stuffed into a JSON-stringified "content" arg — the resulting escape-
# hell fails the parser (observed 2026-04-23 task 2865 DLQ'd after 5
# such attempts). Workflow engine already persists the `result` to
# workspace files itself, so write tools are redundant for structured-
# output steps.
_WRITE_TOOLS = frozenset({"write_file", "apply_diff", "edit_file", "patch_file"})

# Artifact-schema types. Structured types (object/array/json) put the CLEAN
# artifact directly in the final_answer ``result`` — write tools are then
# safely redundant (the engine materializes result→disk). Free-form types
# (markdown/string) are different: a narration-prone agent (e.g. analyst)
# wraps the artifact in "## Analysis …" prose when forced down the
# final_answer path, poisoning the materialized file — but writes a CLEAN
# file via write_file. So write tools are KEPT for free-form schemas.
# (task #524995, [0.0c] interview_script_generation.)
#
# ``json`` is a structured type too (the only live one is [0.0a.draft]
# intake_todo_draft, whose instruction explicitly says "do NOT call write_file
# (the tool is intentionally unavailable for this step); your returned JSON IS
# the artifact"). It was omitted here, so _apply_auto_strip left write_file
# available — contradicting the step's own contract and the intake #73 design
# (engine is the sole writer of schema'd produces paths). Including it keeps
# this predicate, materialize_produces.write_stripped, and the step instruction
# all in agreement.
_STRUCTURED_SCHEMA_TYPES = frozenset({"object", "array", "json"})


def _schema_is_structured_only(schema: dict) -> bool:
    """True when EVERY artifact in *schema* is a structured type
    (object/array/json) — the case where the ``result`` IS the clean artifact
    and stripping write tools is safe. False when ANY artifact is free-form
    (markdown/string), so write tools stay. Typeless entries default to
    ``object`` (preserves prior behaviour for object schemas that omit an
    explicit ``type``)."""
    types = [
        (v.get("type") or "object").lower()
        for v in schema.values()
        if isinstance(v, dict)
    ]
    if not types:
        return True
    return all(t in _STRUCTURED_SCHEMA_TYPES for t in types)


def _produces_has_markdown(produces) -> bool:
    """A ``.md`` produces path is the AUTHORITATIVE free-form artifact signal.

    The schema ``type`` is a VALIDATION concern (which frontmatter fields / shape
    to check); the ``produces`` extension is the AUTHORING concern (how the agent
    emits the artifact). A step may carry an OBJECT/ARRAY schema to validate a
    markdown doc's structured frontmatter (e.g. surfaces / mermaid_per_surface)
    yet still author a ``.md`` file. Keying the write-tool decision off the schema
    ``type`` alone stripped write_file on 4 analyst steps (5.0c user_flow, 5.0d
    screen_inventory/shared_shell, 4.14 register, 6.5z premortem), forcing the
    narration-prone analyst down the final_answer path — its "## Analysis …"
    wrapper then clobbered the materialized file. A ``.md`` produces keeps write
    tools regardless of schema type, so the agent writes a CLEAN file to disk."""
    return isinstance(produces, (list, tuple)) and any(
        isinstance(p, str) and p.endswith(".md") for p in produces
    )


def _write_tools_redundant(schema: dict, produces=None) -> bool:
    """True when the final_answer ``result`` IS the clean artifact and write
    tools are safely stripped: a structured-only schema AND no free-form (``.md``)
    produces path. A markdown produces always keeps write tools (the agent authors
    the file), even under an object/array schema. Sole predicate shared by
    ``_apply_auto_strip`` and ``materialize_produces.write_stripped`` so the two
    never drift (the materializer's candidate order depends on it)."""
    if _produces_has_markdown(produces):
        return False
    return _schema_is_structured_only(schema)


async def execute(profile, task: dict, progress_callback: Callable | None = None) -> dict:
    """Drive one task to completion. Routes by profile.execution_pattern.

    progress_callback: async fn(task_id, iteration, max_iter, summary)
    """
    profile.progress_callback = progress_callback

    await _load_db_prompt_override(profile)

    _task_ctx = _parse_task_ctx(task)

    _apply_hint_from_targets_runtime(_task_ctx)
    _apply_tools_hint(profile, _task_ctx)
    _apply_auto_strip(profile, _task_ctx)

    # Suppress clarification if task explicitly disallows it
    profile._suppress_clarification = _task_ctx.get("may_need_clarification") is False

    if _task_ctx.get("is_workflow_step"):
        await _refresh_workflow_step_config(task, _task_ctx)

    # INVARIANT (final word, after the live-config refresh): a .md produces MUST
    # have write_file, whatever tools_hint / auto_strip did — else the agent
    # can't author its declared file and narrates it into final_answer (clobber).
    _ensure_write_tools_for_markdown_produces(profile, _task_ctx)

    # ── Z6 T7C — detect needs_real_tools and short-circuit-or-inject ──
    # If the task is flagged ``needs_real_tools`` we re-check admission
    # before paying for an LLM call:
    #   • Missing adapter/creds → emit a founder_action via z6_admission
    #     and return ``status='blocked_on_founder_action'`` without ever
    #     entering the react loop. Closes G9 (LLM agents fabricating
    #     vendor responses).
    #   • Prereqs satisfied → inject a system-prompt warning block so
    #     the LLM uses ``vendor_call`` instead of hallucinating.
    _bail = await _maybe_detect_and_bail(task, _task_ctx)
    if _bail is not None:
        return _bail

    try:
        # SP5 (2026-06-10): single_shot path deleted — no live profile sets
        # execution_pattern="single_shot" (shopping_clarifier dropped it at the
        # v3 switch). All tasks run the react loop.
        _result = await _react_run(profile, task, progress_callback=progress_callback)

        # SP3b Task 7: constrained_emit is now a Beckman post-hook child task
        # (wired in posthooks.determine_posthooks). The inline pass was removed
        # here; behaviour is preserved via the post-hook chain.
        return _result
    finally:
        # Restore original allowed_tools ONLY if a setup phase actually
        # overrode it. We gate on a dedicated sentinel (_tools_overridden)
        # rather than hasattr/`is not None`: for a data Profile,
        # _original_allowed_tools is a declared dataclass field (always
        # present, defaults None), so hasattr is always True — restoring
        # unconditionally would wipe allowed_tools to None and silently
        # corrupt the singleton. And `is not None` is wrong because
        # _apply_auto_strip legitimately snapshots None and must restore None.
        # getattr(...,False) keeps this correct for class-backed BaseAgents
        # which never declare the field.
        if getattr(profile, '_tools_overridden', False):
            profile.allowed_tools = profile._original_allowed_tools
            profile._original_allowed_tools = None
            profile._tools_overridden = False


# ────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ────────────────────────────────────────────────────────────────────────────


async def _maybe_detect_and_bail(task: dict, task_ctx: dict) -> dict | None:
    """Z6 T7C — gate the LLM call when a task touches the real world.

    Returns ``None`` to proceed with the normal execution path. Returns a
    completed result dict (``status='blocked_on_founder_action'``) when
    the gate decides to short-circuit instead of calling the LLM. The
    task ctx is mutated in place to (a) inject the warning prompt block
    on the pass-through path and (b) flip the task status on the bail
    path.
    """
    # Hoisted column wins; fall back to ctx for older tasks where the
    # expander has not yet rewritten task rows.
    needs = task.get("needs_real_tools")
    if needs is None:
        needs = task_ctx.get("needs_real_tools")
    try:
        needs_bool = bool(int(needs)) if needs is not None else False
    except (TypeError, ValueError):
        needs_bool = bool(needs)
    if not needs_bool:
        return None

    mission_id = task.get("mission_id")
    if mission_id is None:
        return None

    # Re-use the existing admission logic so emit + de-dup behaviour
    # stays consistent with beckman's pre-dispatch gate.
    try:
        from general_beckman.z6_admission import check_z6_admission
        result = await check_z6_admission(task, int(mission_id))
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"[Task #{task.get('id','?')}] z6 detect-and-bail: admission "
            f"check raised {e!r} — proceeding without injection"
        )
        return None

    if not result.admit:
        # Short-circuit. Tell beckman to park the task.
        try:
            from general_beckman import update_task
            await update_task(
                int(task["id"]),
                status="blocked_on_founder_action",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(
                f"[Task #{task.get('id','?')}] update_task on bail "
                f"skipped: {e}"
            )
        logger.info(
            f"[Task #{task.get('id','?')}] Z6 T7C: short-circuit "
            f"({result.reason}); founder_actions emitted="
            f"{result.founder_actions_emitted}"
        )
        return {
            "status": "blocked_on_founder_action",
            "result": (
                f"Task short-circuited — needs real-world side effects "
                f"({result.reason}). Founder action emitted: "
                f"{result.founder_actions_emitted}"
            ),
            "reason": result.reason,
            "founder_actions_emitted": result.founder_actions_emitted,
            "used_model": None,
            "iterations": 0,
            "cost": 0.0,
        }

    # Prereqs satisfied — inject the warning block into the task's
    # description so the LLM sees it inline (the prompt builder reads
    # task["description"]). Cheap, leaves the profile system prompt
    # untouched.
    from .system_prompt_blocks import (
        REAL_WORLD_BLOCK_MARKER,
        real_world_side_effects_warning,
    )
    reversibility = (
        task.get("reversibility") or task_ctx.get("reversibility")
    )
    desc = task.get("description") or ""
    if REAL_WORLD_BLOCK_MARKER not in desc:
        block = real_world_side_effects_warning(reversibility)
        task["description"] = f"{block}\n\n{desc}"
        logger.info(
            f"[Task #{task.get('id','?')}] Z6 T7C: real-world warning "
            f"block injected (reversibility={reversibility})"
        )
    return None


async def _load_db_prompt_override(profile) -> None:
    """Load active prompt override from the injected prompt store (no src dep)."""
    profile._prompt_version_override = None
    try:
        from finch.store import get_active
        db_prompt = await get_active(profile.name)
        if db_prompt:
            profile._prompt_version_override = db_prompt
    except Exception:
        pass


def _apply_hint_from_targets_runtime(task_ctx: dict) -> None:
    """Strip ``write_file`` from tools_hint at dispatch time when a produce target exists.

    Calls the shared implementation in ``src.workflows.engine.expander`` with
    the mission workspace resolved from the live filesystem.  By dispatch time
    the workspace directory exists (earlier steps have already written their
    files), so the existence check is meaningful — unlike at expansion time
    when no files have been created yet.

    Safe to call for every task: returns immediately when mission_id is absent,
    when the workspace directory does not exist yet, or when no tools_hint /
    no ``write_file`` is present.
    """
    import os

    try:
        mission_id = task_ctx.get("mission_id")
        if not mission_id:
            return
        from src.tools.workspace import WORKSPACE_DIR
        ws = os.path.join(WORKSPACE_DIR, f"mission_{mission_id}")
        if not os.path.isdir(ws):
            return
        from src.workflows.engine.expander import _apply_hint_from_targets
        _apply_hint_from_targets(task_ctx, workspace_path=ws)
    except Exception:
        pass  # Never break dispatch due to a hint-strip failure


def _parse_task_ctx(task: dict) -> dict:
    _task_ctx = task.get("context")
    if isinstance(_task_ctx, str):
        try:
            _task_ctx = json.loads(_task_ctx)
        except (json.JSONDecodeError, TypeError):
            _task_ctx = {}
    if not isinstance(_task_ctx, dict):
        _task_ctx = {}
    return _task_ctx


def _apply_tools_hint(profile, task_ctx: dict) -> None:
    """Override allowed_tools from workflow tools_hint."""
    tools_hint = task_ctx.get("tools_hint")
    if tools_hint is not None and isinstance(tools_hint, list):
        profile._original_allowed_tools = profile.allowed_tools
        profile._tools_overridden = True
        profile.allowed_tools = tools_hint


def _apply_auto_strip(profile, task_ctx: dict) -> None:
    """Strip file/web/write tools when the workflow step says so.

    Prevents wasting iterations re-reading data that's already in the prompt
    (file/web) or writing files the workflow engine will persist anyway
    (write — for steps with structured-output schema).
    """
    _strip_set: set[str] = set()
    if task_ctx.get("_strip_file_tools"):
        _strip_set |= _FILE_TOOLS
    if task_ctx.get("_strip_web_tools"):
        _strip_set |= _WEB_TOOLS
    # Auto-strip write tools when the step has a STRUCTURED-output schema
    # (object/array): the final_answer result is the clean artifact and the
    # engine materializes it. Free-form schemas (markdown/string) keep write
    # tools — narration-prone agents emit clean files via write_file but wrap
    # the final_answer result in prose that poisons the materialized file
    # (task #524995). Explicit opt-out via "_allow_write_tools" for the rare
    # step that legitimately needs both schema'd result AND file side-effects.
    _schema = task_ctx.get("artifact_schema")
    if (_schema and isinstance(_schema, dict)
            and not task_ctx.get("_allow_write_tools")
            and _write_tools_redundant(_schema, task_ctx.get("produces"))):
        _strip_set |= _WRITE_TOOLS

    if not _strip_set:
        return

    if profile.allowed_tools is not None:
        if not getattr(profile, '_tools_overridden', False):
            profile._original_allowed_tools = profile.allowed_tools
            profile._tools_overridden = True
        profile.allowed_tools = [
            t for t in profile.allowed_tools if t not in _strip_set
        ]
    else:
        from src.tools import list_tool_names
        # Legitimately snapshots None (all-tools) — must be restored to None.
        if not getattr(profile, '_tools_overridden', False):
            profile._original_allowed_tools = profile.allowed_tools
            profile._tools_overridden = True
        profile.allowed_tools = [
            t for t in list_tool_names() if t not in _strip_set
        ]


def _ensure_write_tools_for_markdown_produces(profile, task_ctx: dict) -> None:
    """INVARIANT: a step declaring a ``.md`` produces path MUST have write_file.

    A ``produces`` path is a hard requirement to AUTHOR that file, but
    ``_apply_tools_hint`` OVERRIDES ``allowed_tools`` with the step's tools_hint
    verbatim — a ``tools_hint: []`` leaves the agent with NO tools. Unable to
    write, the agent dumps the document into its ``final_answer``, where a
    narration-prone analyst wraps it in a "## Analysis …" report that clobbers
    the materialized artifact (m90 4.14/5.0c/5.0d all shipped ``tools_hint: []``;
    the analyst made ZERO tool calls and the narration became the file). Every
    WORKING markdown step (0.0c/0.1/1.4a/6.5z) already lists ``write_file``;
    this makes that non-negotiable so no step can declare a file it has no tool
    to write. ``_apply_auto_strip`` only REMOVES tools, so it cannot restore
    write_file — this reconciliation is the final word, and MUST run after
    ``_refresh_workflow_step_config`` so it sees the live produces list.

    Scoped to ``.md`` (free-form authored) produces only: a ``.json`` produces is
    structured — the final_answer JSON IS the artifact, no write tool needed.
    ``allowed_tools is None`` means "all tools" (write_file already available).

    EXCEPTION — a NON-EMPTY structured-only (object/array) schema on a ``.md``
    produces is a structured-RETURN step, NOT an authoring step: the agent returns
    JSON and a mechanical post-step materializes the ``.md`` from it (4.14 ADR
    ``register.md`` is rebuilt from the returned ADR JSON; its instruction says
    "do NOT write any files yourself"). Such steps must stay write-stripped — the
    engine, not the agent, writes the file. Only markdown/string/no-schema ``.md``
    steps (the agent authors the doc) get write_file restored."""
    if not _produces_has_markdown(task_ctx.get("produces")):
        return
    _sch = task_ctx.get("artifact_schema")
    if isinstance(_sch, dict) and _sch and _schema_is_structured_only(_sch):
        return  # structured-return-to-.md (mechanical materialize); do not author
    tools = profile.allowed_tools
    if tools is None or "write_file" in tools:
        return
    if not getattr(profile, "_tools_overridden", False):
        profile._original_allowed_tools = profile.allowed_tools
        profile._tools_overridden = True
    profile.allowed_tools = list(tools) + ["write_file"]


async def _refresh_workflow_step_config(task: dict, task_ctx: dict) -> None:
    """Refresh task description + ctx fields from live workflow JSON.

    tasks.description and context.* are frozen at expander time. Edits to
    workflow JSON don't propagate to existing rows, so retries kept running
    stale config (observed task 2890: step 2.8 instruction reshaped from
    11-field use cases → 6-field stories, but task row still carried old
    text and grader kept citing "use cases" in DLQ reasons).

    Refreshed scope:
      - description (was: instruction)
      - done_when, input_artifacts, output_artifacts, artifact_schema,
        tools_hint, difficulty, estimated_output_tokens,
        may_need_clarification, triggers_clarification (in task_ctx)
      - any keys defined in the step's "context" sub-dict
    NOT refreshed (would change task identity mid-flight or are already
    evaluated at expansion time): agent_type, skip_when, depends_on.
    """
    try:
        _step_id = task_ctx.get("workflow_step_id")
        _mid = task.get("mission_id")
        if not (_step_id and _mid):
            return
        from dabidabi import get_db
        _db = await get_db()
        _cur = await _db.execute(
            "SELECT context FROM missions WHERE id = ?", (_mid,),
        )
        _row = await _cur.fetchone()
        await _cur.close()
        _mctx: dict = {}
        if _row and _row[0]:
            try:
                _mctx = json.loads(_row[0])
                if isinstance(_mctx, str):
                    _mctx = json.loads(_mctx)
            except (json.JSONDecodeError, TypeError):
                _mctx = {}
        _wf_name = (
            _mctx.get("workflow_name") if isinstance(_mctx, dict) else None
        ) or "i2p_v3"
        from src.workflows.engine.loader import load_workflow
        _wf = load_workflow(_wf_name)
        _step = _wf.get_step(_step_id)
        if not _step:
            return

        _changed_fields: list[str] = []

        _live_instr = _step.get("instruction")
        if (_live_instr
                and isinstance(_live_instr, str)
                and _live_instr != task.get("description")):
            task["description"] = _live_instr
            _changed_fields.append("description")

        # Top-level step fields that the engine plumbs through expander.py
        # into task context — refresh the same set so retries see live config.
        _CTX_FIELDS = (
            "done_when",
            "input_artifacts",
            "output_artifacts",
            "artifact_schema",
            "tools_hint",
            "difficulty",
            # `checks` (parameterized mechanical verifiers) freeze at expansion
            # like the rest; a workflow edit that ADDS a check (e.g. the
            # requirement-conservation gate) must reach already-expanded rows,
            # else the gate stays inert on in-flight missions. determine_posthooks
            # reads task_ctx["checks"] at completion, so syncing it here is enough.
            "checks",
        )
        for _f in _CTX_FIELDS:
            _live_val = _step.get(_f)
            if _live_val is None:
                continue
            if task_ctx.get(_f) != _live_val:
                task_ctx[_f] = _live_val
                _changed_fields.append(_f)

        # The step may declare a free-form "context" dict (estimated_output_tokens,
        # may_need_clarification, triggers_clarification, custom keys). Merge it
        # so additions / edits flow into live tasks.
        _step_inner_ctx = _step.get("context") or {}
        if isinstance(_step_inner_ctx, dict):
            for _k, _v in _step_inner_ctx.items():
                if task_ctx.get(_k) != _v:
                    task_ctx[_k] = _v
                    _changed_fields.append(f"context.{_k}")

        if _changed_fields:
            task["context"] = json.dumps(task_ctx)
            logger.info(
                f"[Task #{task.get('id','?')}] step-refresh: "
                f"{', '.join(_changed_fields)} re-synced from "
                f"live JSON (step={_step_id}, wf={_wf_name})"
            )
            # Persist to DB so the post-execute hook (which re-fetches the
            # task via workflow_engine.advance) validates against the LIVE
            # schema, not the stale snapshot stored at admission time.
            # Without this write, mission 57 task 4450 (6.1) kept DLQ'ing on
            # an empty-placeholder check for ``dependencies`` because
            # advance saw the old legacy schema even though base.py had
            # refreshed in-memory.
            try:
                from general_beckman import update_task as _update_task
                _persist: dict = {"context": task["context"]}
                # The grader and self_reflect children read tasks.description
                # straight from the DB. If we only persist context, an edited
                # workflow instruction reaches the worker (in-memory refresh)
                # but NOT the grader — which then judges the artifact against
                # the stale spec and DLQs a correct result (task #259351).
                if "description" in _changed_fields:
                    _persist["description"] = task["description"]
                await _update_task(task["id"], **_persist)
            except Exception as _persist_exc:
                logger.warning(
                    f"[Task #{task.get('id','?')}] step-refresh "
                    f"persist failed: {_persist_exc!r}"
                )
    except Exception as _e:
        logger.warning(
            f"[Task #{task.get('id','?')}] step config refresh failed: {_e}"
        )
