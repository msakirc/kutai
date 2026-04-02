"""Pre/post execution hooks for workflow steps in the orchestrator.

Handles artifact injection, output storage, conditional evaluation,
template expansion triggers, and CodingPipeline delegation detection.
"""
from __future__ import annotations

import json
from typing import Optional

from src.infra.logging_config import get_logger
from .artifacts import ArtifactStore, format_artifacts_for_prompt, get_phase_summaries
from .conditions import evaluate_condition, resolve_group
from .policies import ReviewTracker
from .quality_gates import evaluate_gate, format_gate_result

logger = get_logger("workflows.engine.hooks")


def validate_artifact_schema(output_value: str, schema: dict) -> tuple[bool, str]:
    """Validate an artifact output against its schema definition.

    Returns (is_valid, error_message). error_message is empty if valid.
    """
    if not schema:
        return True, ""

    for artifact_name, rules in schema.items():
        schema_type = rules.get("type", "string")

        if schema_type == "object":
            required = rules.get("required_fields", [])
            # Try JSON first
            try:
                data = json.loads(output_value) if isinstance(output_value, str) else output_value
                if isinstance(data, dict):
                    missing = [f for f in required if f not in data]
                    if missing:
                        return False, f"Missing required fields in '{artifact_name}': {missing}"
                    continue  # this artifact passed
            except (json.JSONDecodeError, TypeError):
                pass
            # Fallback: accept text/markdown if required fields appear as keywords
            # Small LLMs often produce structured text, not JSON
            if required:
                text_lower = str(output_value).lower().replace("_", " ").replace("-", " ")
                missing = [f for f in required if f.lower().replace("_", " ") not in text_lower]
                if missing:
                    return False, f"'{artifact_name}' missing content about: {missing}"

        elif schema_type == "array":
            # Try JSON first
            try:
                data = json.loads(output_value) if isinstance(output_value, str) else output_value
                if isinstance(data, list):
                    min_items = rules.get("min_items", 0)
                    if len(data) < min_items:
                        return False, f"'{artifact_name}' has {len(data)} items, need >= {min_items}"
                    item_fields = rules.get("item_fields", [])
                    if item_fields and data:
                        for i, item in enumerate(data):
                            if isinstance(item, dict):
                                missing = [f for f in item_fields if f not in item]
                                if missing:
                                    return False, f"Item {i} in '{artifact_name}' missing fields: {missing}"
                    continue  # passed
            except (json.JSONDecodeError, TypeError):
                pass
            # Fallback: accept text if it has numbered/bulleted items
            min_items = rules.get("min_items", 0)
            if min_items > 0:
                import re as _re
                items = _re.findall(r'(?:^|\n)\s*(?:\d+[\.\)]|\-|\*)\s+\S', str(output_value))
                if len(items) < min_items:
                    return False, f"'{artifact_name}' has ~{len(items)} list items, need >= {min_items}"

        elif schema_type == "string":
            min_length = rules.get("min_length", 1)
            if not output_value or len(str(output_value).strip()) < min_length:
                return False, f"'{artifact_name}' is too short (min {min_length} chars)"

        elif schema_type == "markdown":
            required_sections = rules.get("required_sections", [])
            text = str(output_value)
            missing = [s for s in required_sections if s.lower() not in text.lower()]
            if missing:
                return False, f"'{artifact_name}' missing sections: {missing}"

    return True, ""

# ── LLM-based artifact summarization ──────────────────────────────────────


async def _llm_summarize(text: str, artifact_name: str) -> Optional[str]:
    """Summarize a large artifact using the LLM (OVERHEAD call).

    Uses whatever model is loaded — no swaps. Falls back to structural
    extraction if LLM is unavailable.
    """
    try:
        from ...core.llm_dispatcher import get_dispatcher, CallCategory
        from ...core.router import ModelRequirements

        reqs = ModelRequirements(
            task="summarizer",
            difficulty=2,
            prefer_speed=True,
            prefer_local=True,
            estimated_input_tokens=min(len(text) // 4, 4000),
            estimated_output_tokens=500,
        )

        # Truncate input to 4k tokens max to fit any model
        max_input = 16000  # ~4k tokens
        truncated_text = text[:max_input]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise summarizer. Produce a summary that "
                    "preserves ALL key facts, decisions, and data points. "
                    "Target: under 400 words. No filler."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Summarize this '{artifact_name}' artifact. Keep every "
                    f"important fact, number, name, and decision:\n\n"
                    f"{truncated_text}"
                ),
            },
        ]

        response = await get_dispatcher().request(
            CallCategory.OVERHEAD, reqs, messages, tools=None
        )
        summary = response.get("content", "").strip()
        if summary and len(summary) > 50:
            return summary

    except Exception as e:
        logger.debug(f"[Workflow Hook] LLM summarization failed: {e}")

    # Fallback: structural extraction if LLM unavailable
    return _structural_summary(text)


def _structural_summary(text: str, target: int = 1500) -> str:
    """Fallback summary without LLM — keep headings + first lines."""
    if len(text) <= target:
        return text
    lines = text.split("\n")
    out: list[str] = []
    total = 0
    for line in lines:
        s = line.strip()
        if s.startswith("#") or (out and not out[-1].startswith("#") and not s):
            out.append(s)
            total += len(s) + 1
        elif s and (not out or out[-1].startswith("#")):
            out.append(s[:200])
            total += min(len(s), 200) + 1
        if total >= target:
            break
    return "\n".join(out) if out else text[:target]


# ── Module-level singleton ─────────────────────────────────────────────────

_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Return the module-level ArtifactStore singleton (lazy init)."""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore(use_db=True)
    return _artifact_store


# ── Helper functions ───────────────────────────────────────────────────────


def is_workflow_step(context: dict) -> bool:
    """Check whether the task context marks this as a workflow step."""
    return bool(context.get("is_workflow_step"))


def extract_output_artifact_names(context: dict) -> list[str]:
    """Get output_artifacts list from context, defaulting to empty."""
    return context.get("output_artifacts", [])


def _parse_context(task: dict) -> dict:
    """Parse task context, handling both dict and JSON string forms."""
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    if not isinstance(ctx, dict):
        ctx = {}
    return ctx


def enrich_task_description(task: dict, artifact_contents: dict) -> str:
    """Build an enriched description with artifact context and done_when.

    Parameters
    ----------
    task:
        Task dict with "description" and optional "context".
    artifact_contents:
        Mapping of artifact name -> content (already fetched).

    Returns
    -------
    str
        The enriched description string.
    """
    instruction = task.get("description", "")
    ctx = _parse_context(task)

    context_strategy = ctx.get("context_strategy")
    done_when = ctx.get("done_when")

    parts: list[str] = [instruction]

    # Append formatted artifacts if any are available
    # Budget must fit the WORST-CASE model (8k context local LLM).
    # Reserve: ~1500 tokens system prompt, ~500 tokens instruction,
    # ~1500 tokens for generation = 3500 reserved, ~4500 for artifacts.
    # Even "hard" steps may fall back to local if cloud is unavailable.
    difficulty = ctx.get("difficulty", 6)
    if difficulty <= 3:      # easy — minimal context needed
        max_artifact_chars = 4000   # ~1000 tokens
    elif difficulty <= 6:    # medium — standard local model
        max_artifact_chars = 12000  # ~3000 tokens
    else:                    # hard — may get cloud but must fit local too
        max_artifact_chars = 18000  # ~4500 tokens (fits 8k with headroom)

    if artifact_contents:
        filtered = {k: v for k, v in artifact_contents.items() if v is not None}
        if filtered:
            formatted = format_artifacts_for_prompt(
                filtered, context_strategy=context_strategy,
                max_total=max_artifact_chars,
            )
            if formatted:
                parts.append(f"\n\n## Context Artifacts\n\n{formatted}")

    # Append human clarification answers if available
    user_clarification = ctx.get("user_clarification")
    if user_clarification:
        parts.append(
            f"\n\n## Human Clarification Answers\n"
            f"The human has answered your questions. Use these answers to complete the task:\n\n"
            f"{user_clarification}"
        )

    # Append schema validation error from previous retry
    schema_error = ctx.get("_schema_error")
    if schema_error:
        retry_count = ctx.get("_schema_retry_count", 0)
        parts.append(
            f"\n\n## IMPORTANT: Previous Output Was Invalid (retry {retry_count}/2)\n"
            f"Your previous output failed validation: **{schema_error}**\n"
            f"Fix your output to match the required format."
        )

    # Append done_when section if present
    if done_when:
        parts.append(f"\n\n## Done When\n{done_when}")

    return "".join(parts)


# ── Pre/Post hooks ─────────────────────────────────────────────────────────


async def pre_execute_workflow_step(task: dict) -> dict:
    """Pre-hook: inject artifact context into workflow step descriptions.

    If the task is not a workflow step, returns it unchanged.
    Otherwise fetches input artifacts from the store and enriches
    the task description.
    """
    ctx = _parse_context(task)
    if not is_workflow_step(ctx):
        return task

    mission_id = ctx.get("mission_id") or task.get("mission_id")
    input_artifact_names: list[str] = ctx.get("input_artifacts", [])

    # Fetch artifacts from store
    store = get_artifact_store()
    artifact_contents: dict[str, Optional[str]] = {}
    if mission_id is not None and input_artifact_names:
        artifact_contents = await store.collect(mission_id, input_artifact_names)

    # Inject phase summaries from earlier phases
    workflow_phase = ctx.get("workflow_phase")
    if mission_id is not None and workflow_phase:
        phase_summaries = await get_phase_summaries(store, mission_id, workflow_phase)
        if phase_summaries:
            artifact_contents.update(phase_summaries)
            # Ensure phase summaries are included at reference tier
            context_strategy = ctx.get("context_strategy")
            if isinstance(context_strategy, dict):
                ref_list = context_strategy.setdefault("reference", [])
                for sname in phase_summaries:
                    if sname not in ref_list:
                        ref_list.append(sname)
                # Re-serialize updated strategy into context so enrich picks it up
                if isinstance(task.get("context"), str):
                    ctx["context_strategy"] = context_strategy
                    task["context"] = json.dumps(ctx)
                else:
                    task["context"]["context_strategy"] = context_strategy

    # Enrich description
    task["description"] = enrich_task_description(task, artifact_contents)

    logger.info(
        f"[Workflow Hook] Pre-execute: injected {len(input_artifact_names)} "
        f"artifact(s) into task description"
    )

    return task


_review_tracker = ReviewTracker()


async def post_execute_workflow_step(task: dict, result: dict) -> None:
    """Post-hook: store output artifacts, evaluate conditional groups,
    trigger template expansion, and track review cycles.

    If the task is not a workflow step, returns immediately.
    """
    ctx = _parse_context(task)
    if not is_workflow_step(ctx):
        return

    mission_id = ctx.get("mission_id") or task.get("mission_id")
    output_names = extract_output_artifact_names(ctx)
    step_id = ctx.get("workflow_step_id", "")

    if not mission_id or not output_names:
        return

    store = get_artifact_store()
    output_value = result.get("result", "")

    for name in output_names:
        await store.store(mission_id, name, output_value)
        logger.info(
            f"[Workflow Hook] Post-execute: stored artifact '{name}' "
            f"for mission {mission_id} ({len(output_value)} chars)"
        )

    # ── Auto-summarize large artifacts ──
    # If an artifact exceeds the context budget, use the LLM to produce a
    # compact summary ({name}_summary) that downstream steps consume.
    # Uses OVERHEAD category — cheap call, no model swaps.
    _SUMMARY_THRESHOLD = 3000  # chars — above this, create a summary
    if output_value and len(output_value) > _SUMMARY_THRESHOLD:
        for name in output_names:
            summary = await _llm_summarize(output_value, name)
            if summary:
                summary_name = f"{name}_summary"
                await store.store(mission_id, summary_name, summary)
                logger.info(
                    f"[Workflow Hook] LLM-summarized '{name}' -> '{summary_name}' "
                    f"({len(output_value)} -> {len(summary)} chars)"
                )

    # ── Write artifacts to disk in mission directory ──
    if output_value and mission_id:
        try:
            from ...tools.workspace import WORKSPACE_DIR
            import os
            artifact_dir = os.path.join(WORKSPACE_DIR, f"mission_{mission_id}")
            os.makedirs(artifact_dir, exist_ok=True)
            for name in output_names:
                file_path = os.path.join(artifact_dir, f"{name}.md")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(output_value)
                logger.debug(f"[Workflow Hook] Wrote artifact to {file_path}")
        except Exception as e:
            logger.debug(f"[Workflow Hook] Could not write artifact to disk: {e}")

    # ── Validate artifact schema ──
    artifact_schema = ctx.get("artifact_schema")
    if artifact_schema and output_value:
        is_valid, error_msg = validate_artifact_schema(output_value, artifact_schema)
        if not is_valid:
            retry_count = ctx.get("_schema_retry_count", 0)
            if retry_count < 2:
                logger.warning(
                    f"[Workflow Hook] Artifact schema validation failed for step "
                    f"'{step_id}': {error_msg}. Retry {retry_count + 1}/2"
                )
                # Update retry count in context for next attempt
                # The orchestrator will retry the task
                try:
                    from ...infra.db import update_task
                    new_ctx = dict(ctx)
                    new_ctx["_schema_retry_count"] = retry_count + 1
                    new_ctx["_schema_error"] = error_msg
                    # On second retry, escalate difficulty so router picks
                    # a more capable model (often cloud)
                    if retry_count >= 1:
                        current_diff = new_ctx.get("difficulty", 6)
                        new_ctx["difficulty"] = min(current_diff + 2, 10)
                        new_ctx["prefer_quality"] = True
                        new_ctx["needs_thinking"] = True
                    await update_task(
                        task.get("id"),
                        status="pending",
                        context=new_ctx,
                        error=f"Schema validation: {error_msg}",
                    )
                except Exception as e:
                    logger.debug(f"[Workflow Hook] Could not retry task: {e}")
            else:
                logger.warning(
                    f"[Workflow Hook] Artifact schema validation failed after 2 retries "
                    f"for step '{step_id}': {error_msg}. Accepting best attempt."
                )

    # ── Force needs_clarification for human-gate steps ──
    # Steps with triggers_clarification=true bypass LLM's clarify action.
    # Only fires ONCE — if clarification_history already has answers,
    # the human already responded and the step should complete normally.
    if (ctx.get("triggers_clarification")
            and output_value
            and not ctx.get("clarification_history")):
        result["status"] = "needs_clarification"
        result["clarification"] = output_value
        logger.info(
            f"[Workflow Hook] Step '{step_id}' triggers_clarification — "
            f"overriding result status to needs_clarification"
        )

    # ── Store clarification_answers artifact when human-gate step completes ──
    # Second run (after user answered): clarification_history exists, step completes.
    # Store user_clarification as the clarification_answers artifact so downstream
    # steps (e.g. idea_brief_compilation) can consume it.
    if (ctx.get("triggers_clarification")
            and ctx.get("clarification_history")):
        user_clarification = ctx.get("user_clarification", "")
        if user_clarification and mission_id:
            await store.store(mission_id, "clarification_answers", user_clarification)
            # Also write to disk
            try:
                from ...tools.workspace import WORKSPACE_DIR
                import os
                artifact_dir = os.path.join(WORKSPACE_DIR, f"mission_{mission_id}")
                os.makedirs(artifact_dir, exist_ok=True)
                file_path = os.path.join(artifact_dir, "clarification_answers.md")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(user_clarification)
            except Exception:
                pass
            logger.info(
                f"[Workflow Hook] Step '{step_id}' completed with clarification — "
                f"stored 'clarification_answers' artifact ({len(user_clarification)} chars)"
            )

    # ── Check conditional group triggers ──
    await _check_conditional_triggers(mission_id, output_names, store)

    # ── Check template expansion trigger ──
    if "implementation_backlog" in output_names:
        await _trigger_template_expansion(mission_id, output_value)

    # ── Track review status ──
    status = result.get("status", "completed")
    if status in ("needs_review", "failed"):
        action = _review_tracker.record_failure(step_id)
        if action == "escalate":
            logger.warning(
                f"[Workflow Hook] Step '{step_id}' exceeded max review "
                f"cycles — escalating to needs_clarification"
            )

    # ── Check phase completion for checkpoint/resume support ──
    workflow_phase = ctx.get("workflow_phase")
    if mission_id and workflow_phase:
        await _check_phase_completion(mission_id, workflow_phase)


async def _check_phase_completion(mission_id: int, phase_id: str) -> bool:
    """Detect when all tasks in a workflow phase are done and checkpoint it.

    Returns True if the phase is complete, False otherwise.
    """
    try:
        from ...infra.db import get_tasks_for_mission, get_workflow_checkpoint, upsert_workflow_checkpoint
    except ImportError as exc:
        logger.debug(f"[Workflow Hook] Phase completion check skipped (import): {exc}")
        return False

    try:
        tasks = await get_tasks_for_mission(mission_id)
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Could not fetch tasks for mission {mission_id}: {exc}")
        return False

    terminal_states = {"completed", "skipped", "cancelled"}
    phase_tasks = []
    for t in tasks:
        ctx = _parse_context(t)
        if ctx.get("workflow_phase") == phase_id:
            phase_tasks.append(t)

    if not phase_tasks:
        return False

    all_done = all(t.get("status") in terminal_states for t in phase_tasks)
    if not all_done:
        return False

    # Phase complete — update checkpoint
    try:
        checkpoint = await get_workflow_checkpoint(mission_id)
        completed = checkpoint["completed_phases"] if checkpoint else []
        workflow_name = checkpoint["workflow_name"] if checkpoint else ""

        if phase_id not in completed:
            completed.append(phase_id)

        await upsert_workflow_checkpoint(
            mission_id=mission_id,
            workflow_name=workflow_name,
            current_phase=phase_id,
            completed_phases=completed,
        )
        logger.info(
            f"[Workflow Hook] Phase '{phase_id}' complete for mission {mission_id} "
            f"({len(phase_tasks)} tasks). Checkpoint updated."
        )
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Could not update checkpoint: {exc}")

    # Generate a summary artifact for the completed phase
    await _generate_phase_summary(mission_id, phase_id, phase_tasks)

    # ── Evaluate quality gate ──
    await _evaluate_phase_gate(mission_id, phase_id)

    return True


async def _evaluate_phase_gate(mission_id: int, phase_id: str) -> None:
    """Evaluate the quality gate for a completed phase and store the result."""
    store = get_artifact_store()
    try:
        phase_num = phase_id.replace("phase_", "")
        passed, details = await evaluate_gate(mission_id, phase_id, store)

        # Store gate result as artifact
        result_text = format_gate_result(phase_id, passed, details)
        await store.store(mission_id, f"phase_{phase_num}_gate_result", result_text)

        if details:  # Only log if there was actually a gate
            if passed:
                logger.info(
                    f"[Workflow Hook] Quality gate for '{phase_id}' PASSED "
                    f"(mission {mission_id})"
                )
            else:
                logger.warning(
                    f"[Workflow Hook] Quality gate for '{phase_id}' FAILED "
                    f"(mission {mission_id}): {result_text}"
                )
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Quality gate evaluation failed: {exc}")


async def _generate_phase_summary(
    mission_id: int, phase_id: str, phase_tasks: list[dict]
) -> None:
    """Build a structured summary from a completed phase's output artifacts.

    The summary is stored as ``phase_{N}_summary`` in the artifact store so
    that subsequent phases can receive it as context.
    """
    from .status import PHASE_NAMES

    store = get_artifact_store()

    # Collect output artifact names from all phase tasks
    output_names: list[str] = []
    for t in phase_tasks:
        ctx = _parse_context(t)
        output_names.extend(ctx.get("output_artifacts", []))

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in output_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    # Fetch artifact contents
    artifact_contents = await store.collect(mission_id, unique_names)

    # Build summary text
    phase_name = PHASE_NAMES.get(phase_id, phase_id)
    # Extract phase number for the artifact key
    try:
        phase_num = phase_id.split("_", 1)[1]
    except IndexError:
        phase_num = phase_id

    names_with_content = [
        n for n in unique_names if artifact_contents.get(n)
    ]
    artifact_count = len(names_with_content)

    lines: list[str] = [
        f"## Phase {phase_num}: {phase_name} — Summary",
        f"**Key outputs:** {', '.join(names_with_content) if names_with_content else 'none'}",
        f"**Artifacts produced:** {artifact_count}",
        "",
    ]

    for name in names_with_content:
        content = artifact_contents[name] or ""
        excerpt = content[:200]
        if len(content) > 200:
            excerpt += "..."
        lines.append(f"### {name}\n{excerpt}")
        lines.append("")

    summary_text = "\n".join(lines).rstrip()

    summary_artifact_name = f"phase_{phase_num}_summary"
    await store.store(mission_id, summary_artifact_name, summary_text)
    logger.info(
        f"[Workflow Hook] Generated summary for '{phase_id}' "
        f"({artifact_count} artifacts) -> '{summary_artifact_name}'"
    )


async def _check_conditional_triggers(
    mission_id: int, output_names: list[str], store: ArtifactStore
) -> None:
    """Evaluate conditional groups when their trigger artifact is produced."""
    try:
        from .loader import load_workflow

        # Try loading the workflow used by this mission
        workflow_name = "i2p_v3"  # fallback
        try:
            from ...infra.db import get_mission
            mission = await get_mission(mission_id)
            if mission:
                m_ctx = mission.get("context", "{}")
                if isinstance(m_ctx, str):
                    m_ctx = json.loads(m_ctx)
                workflow_name = m_ctx.get("workflow_name", "i2p_v3")
        except Exception:
            pass
        wf = load_workflow(workflow_name)
    except Exception:
        logger.debug("[Workflow Hook] Could not load workflow for conditional eval")
        return

    for group in wf.conditional_groups:
        condition_artifact = group.get("condition_artifact", "")
        if condition_artifact not in output_names:
            continue

        artifact_value = await store.retrieve(mission_id, condition_artifact)
        if artifact_value is None:
            continue

        condition_check = group.get("condition_check", "")
        result_bool = evaluate_condition(condition_check, artifact_value)
        included, excluded = resolve_group(group, artifact_value)

        logger.info(
            f"[Workflow Hook] Conditional group '{group.get('group_id')}': "
            f"condition={result_bool}, include={len(included)}, "
            f"exclude={len(excluded)} steps"
        )

        # Update task statuses in DB for excluded steps
        if excluded:
            try:
                from ...infra.db import update_task_by_context_field, propagate_skips

                for step in excluded:
                    await update_task_by_context_field(
                        mission_id=mission_id,
                        field="workflow_step_id",
                        value=step,
                        status="skipped",
                    )
                # Cascade skips to downstream dependents
                skipped_count = await propagate_skips(mission_id)
                if skipped_count:
                    logger.info(
                        f"[Workflow Hook] Cascaded skip to {skipped_count} dependent tasks"
                    )
            except (ImportError, Exception) as e:
                logger.debug(
                    f"[Workflow Hook] Could not skip excluded steps: {e}"
                )


async def _trigger_template_expansion(mission_id: int, backlog_text: str) -> None:
    """Expand feature_implementation_template for each feature in backlog.

    Respects ``depends_on_features`` from the backlog: the first task of a
    dependent feature won't start until the last task of its prerequisite
    feature completes.  After all features are expanded, inserts a
    cross-feature integration test step.
    """
    import json as _json

    try:
        features = _json.loads(backlog_text)
        if not isinstance(features, list):
            logger.debug("[Workflow Hook] implementation_backlog is not a list")
            return
    except (ValueError, TypeError):
        logger.debug("[Workflow Hook] Could not parse implementation_backlog as JSON")
        return

    try:
        from .loader import load_workflow
        from .expander import expand_template, expand_steps_to_tasks
        from ...infra.db import add_task as insert_task, update_task

        # Try the workflow used by this mission, fall back to i2p_v3 then i2p_v2
        workflow_name = "i2p_v3"
        try:
            from ...infra.db import get_mission
            mission = await get_mission(mission_id)
            if mission:
                m_ctx = mission.get("context", "{}")
                if isinstance(m_ctx, str):
                    m_ctx = _json.loads(m_ctx)
                workflow_name = m_ctx.get("workflow_name", "i2p_v3")
        except Exception:
            pass

        wf = load_workflow(workflow_name)
        template = wf.get_template("feature_implementation_template")
        if not template:
            logger.warning("[Workflow Hook] feature_implementation_template not found")
            return

        # Track feature_id → (first_task_id, last_task_id) for cross-feature deps
        feature_task_range: dict[str, tuple[int, int]] = {}

        for feature in features:
            if not isinstance(feature, dict):
                continue
            fid = feature.get("id", feature.get("feature_id", "unknown"))
            fname = feature.get("name", feature.get("feature_name", "Unnamed"))

            expanded = expand_template(
                template,
                params={"feature_id": fid, "feature_name": fname},
                prefix=f"8.{fid}.",
            )

            tasks = expand_steps_to_tasks(
                expanded, mission_id=mission_id, initial_context={}
            )

            # Batch insert with rollback on failure
            inserted_ids = []
            try:
                for t in tasks:
                    t.pop("depends_on_steps", None)
                    task_id = await insert_task(**t)
                    inserted_ids.append(task_id)
            except Exception as insert_err:
                # Rollback: cancel partially inserted tasks
                for tid in inserted_ids:
                    try:
                        await update_task(tid, status="cancelled")
                    except Exception:
                        pass
                logger.error(
                    f"[Workflow Hook] Partial expansion rollback for '{fid}': {insert_err}"
                )
                continue  # Skip this feature, try next one

            if inserted_ids:
                feature_task_range[fid] = (inserted_ids[0], inserted_ids[-1])

            logger.info(
                f"[Workflow Hook] Expanded template for feature '{fid}' "
                f"({len(expanded)} steps \u2192 {len(tasks)} tasks)"
            )

        # ── Wire cross-feature dependencies ──
        # If feature B depends_on_features: ["A"], then B's first task
        # should wait until A's last task completes.
        for feature in features:
            if not isinstance(feature, dict):
                continue
            fid = feature.get("id", feature.get("feature_id", "unknown"))
            dep_features = feature.get("depends_on_features", [])
            if not dep_features or fid not in feature_task_range:
                continue

            first_task_id = feature_task_range[fid][0]
            prerequisite_task_ids = []
            for dep_fid in dep_features:
                if dep_fid in feature_task_range:
                    prerequisite_task_ids.append(feature_task_range[dep_fid][1])

            if prerequisite_task_ids:
                try:
                    dep_json = _json.dumps(prerequisite_task_ids)
                    await update_task(first_task_id, depends_on=dep_json)
                    logger.info(
                        f"[Workflow Hook] Feature '{fid}' first task #{first_task_id} "
                        f"depends on tasks {prerequisite_task_ids} (cross-feature)"
                    )
                except Exception as dep_err:
                    logger.debug(
                        f"[Workflow Hook] Could not set cross-feature deps: {dep_err}"
                    )

        # ── Insert cross-feature integration test step ──
        # Runs after ALL features are done — tests interactions between features
        if len(feature_task_range) >= 2:
            all_last_tasks = [last for _, last in feature_task_range.values()]
            feature_names = []
            for f in features:
                if isinstance(f, dict):
                    feature_names.append(
                        f.get("name", f.get("feature_name", f.get("id", "?")))
                    )
            try:
                integration_task_id = await insert_task(
                    title="[8.integration] Cross-feature integration tests",
                    description=(
                        f"Test interactions between all implemented features: "
                        f"{', '.join(feature_names)}. "
                        f"Verify: shared data flows correctly between features, "
                        f"auth/permissions work across feature boundaries, "
                        f"navigation between features works, "
                        f"no conflicts in shared resources (DB, API routes, state). "
                        f"Run the full test suite and report results."
                    ),
                    mission_id=mission_id,
                    agent_type="test_generator",
                    tier="auto",
                    priority=7,
                    depends_on=all_last_tasks,
                    context={
                        "workflow_step_id": "8.integration",
                        "workflow_phase": "phase_8",
                        "is_workflow_step": True,
                        "difficulty": 6,
                        "tools_hint": ["shell", "read_file", "write_file", "coverage",
                                       "query_codebase", "codebase_map"],
                        "output_artifacts": ["integration_test_results"],
                    },
                )
                logger.info(
                    f"[Workflow Hook] Created cross-feature integration test "
                    f"task #{integration_task_id} (depends on {len(all_last_tasks)} features)"
                )
            except Exception as integ_err:
                logger.debug(
                    f"[Workflow Hook] Could not create integration test task: {integ_err}"
                )

    except (ImportError, Exception) as e:
        logger.debug(f"[Workflow Hook] Template expansion failed: {e}")
