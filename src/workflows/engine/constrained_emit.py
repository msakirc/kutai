"""Post-execution structured-output guarantee for workflow steps.

Logged failure data showed models dropping the same required JSON field
across 5+ retries even when ``_schema_error`` injection named the missing
field (mission 46 step 7.4 stuck on ``connection_verified`` 25 times).
Post-hoc retry hints are whack-a-mole — the structural fix is to constrain
decoding so the omission can't occur.

Phase A.12 (2026-05-04): moved from BaseAgent._maybe_constrained_emit. Now
called directly by the runtime entry as a workflow_engine concern. Workflow
step structural concern, not multi-call orchestration concern.

Behaviour:

* No-op unless this is a workflow step with a constrainable
  ``artifact_schema`` (object/array — markdown is unconstrainable and
  handled by the validator + writer schema-aware prompt).
* No-op unless the upstream result is a normal completion. We do not
  rewrite ``needs_subtasks``, ``needs_clarification``, ``needs_review``,
  or already-failed results.
* Skips when the model picked for the fix-up call doesn't support
  json_schema — caller's degradation logic handles this.
* On any error, returns the original ``result`` unchanged. Schema
  validation hook still flags missing fields and triggers normal retry —
  fix-up is a best-effort win, never a regression.

Cost: one extra OVERHEAD call per artifact step. Acceptable when the
alternative is 5 worker retries × main_work cost on a hot loaded model.
"""
from __future__ import annotations

import json

from src.infra.logging_config import get_logger

logger = get_logger("workflows.engine.constrained_emit")


async def maybe_apply(task: dict, result: dict) -> dict:
    """Apply constrained-emit pass to a freshly completed task result.

    Returns a NEW result dict (or the original unchanged) — never raises.
    """
    if not isinstance(result, dict):
        return result
    if result.get("status") not in (None, "completed"):
        # Failures, clarifies, subtasks pass through unchanged.
        return result
    draft = result.get("result")
    if not isinstance(draft, str) or not draft.strip():
        return result

    ctx = task.get("context") or {}
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
            if isinstance(ctx, str):
                ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError, ValueError):
            return result
    if not isinstance(ctx, dict):
        return result
    if not ctx.get("is_workflow_step"):
        return result

    artifact_schema = ctx.get("artifact_schema")
    if not isinstance(artifact_schema, dict):
        return result

    from src.workflows.engine.json_schema_translator import build_response_format
    step_id = ctx.get("workflow_step_id") or "artifact"
    # JSON Schema 'name' must be alphanumeric+underscore; sanitize.
    safe_name = "step_" + "".join(
        c if c.isalnum() else "_" for c in str(step_id)
    )
    response_format = build_response_format(
        artifact_schema, name=safe_name,
    )
    if response_format is None:
        # Unconstrainable (markdown / string). Skip — validator and
        # writer-schema-aware prompt cover that path.
        return result

    # Build a tight prompt that re-emits the artifact in conforming JSON.
    # The model receives the raw JSON Schema so it can see exactly what
    # fields are required, plus the draft to anchor on.
    schema_text = json.dumps(
        response_format["json_schema"]["schema"],
        ensure_ascii=False,
        indent=2,
    )
    # Skip the emit when the draft already parses as JSON with all
    # required artifact keys present. Re-emitting in that case tends to
    # COMPRESS rather than reshape — the model sees a long rich draft,
    # gets a tight token budget, and trims content from tail fields to
    # fit (mission 57 task 4441 5.4b: draft 30751 chars with full
    # empty_states/error_states arrays became a 12826-char emit with
    # empty placeholder lists). The schema validator runs next and
    # catches genuine shape gaps; the emit pass is only valuable when
    # the draft is non-JSON or missing top-level keys.
    try:
        _parsed = json.loads(draft)
        if isinstance(_parsed, dict):
            _need = [
                n for n, r in artifact_schema.items()
                if isinstance(r, dict) and r.get("type") in ("object", "array")
            ]
            if _need and all(k in _parsed for k in _need):
                logger.info(
                    f"[Task #{task.get('id','?')}] constrained_emit skipped "
                    f"— draft parses with all required keys present "
                    f"(step={step_id}, draft={len(draft)} chars)"
                )
                return result
    except (json.JSONDecodeError, TypeError, ValueError):
        pass  # Draft isn't JSON — emit pass will reshape.

    # Cap draft to keep input token cost in line. Bumped from 12000 to
    # 30000 so big multi-artifact drafts (form_specs +
    # empty_error_state_specs) don't lose tail content before the emit
    # even sees it. Local OVERHEAD calls have no per-token cost; cap is
    # purely a context-window guardrail.
    draft_for_prompt = draft[:30000]
    system = (
        "You are a structured-output emitter. Re-emit the artifact "
        "below as JSON conforming exactly to the provided schema. "
        "Do not add commentary. Do not wrap in envelopes. Output "
        "ONLY the JSON value.\n\n"
        "Rules:\n"
        "- Every required field must be present with a real value.\n"
        "- Do not invent fields not in the schema.\n"
        "- Preserve the draft's information; restructure into the "
        "schema, do not summarize away content."
    )
    user = (
        f"Schema:\n```json\n{schema_text}\n```\n\n"
        f"Draft to fix:\n```\n{draft_for_prompt}\n```\n\n"
        f"Emit the final JSON now."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        from src.core.llm_dispatcher import get_dispatcher, CallCategory
        resp = await get_dispatcher().request(
            CallCategory.OVERHEAD,
            task="structured_emit",
            difficulty=3,
            messages=messages,
            estimated_input_tokens=max(1000, len(user) // 4),
            # Output budget: previously min(12000, len/3) which gave
            # 4000 tokens for a 12000-char prompt — way too tight for
            # multi-artifact schemas (5.4b: form_specs +
            # empty_error_state_specs). Now floors at len/3 with a
            # higher ceiling so emits can preserve a large draft
            # instead of compressing.
            estimated_output_tokens=min(
                16000,
                max(2000, len(draft_for_prompt) // 3),
            ),
            prefer_speed=True,
            response_format=response_format,
            task_obj=task,
        )
    except Exception as exc:
        logger.warning(
            f"[Task #{task.get('id','?')}] constrained_emit dispatch "
            f"failed: {exc!r} — keeping draft"
        )
        return result

    emitted = resp.get("content", "") if isinstance(resp, dict) else ""
    if isinstance(emitted, list):
        emitted = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in emitted
        )
    if not isinstance(emitted, str) or not emitted.strip():
        logger.warning(
            f"[Task #{task.get('id','?')}] constrained_emit returned "
            f"empty — keeping draft"
        )
        return result

    # Cheap shape check: must parse as JSON. The schema-validation hook
    # will do the deeper required-field check.
    try:
        json.loads(emitted)
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            f"[Task #{task.get('id','?')}] constrained_emit produced "
            f"non-JSON output (model={resp.get('model','?')}) — keeping draft"
        )
        return result

    logger.info(
        f"[Task #{task.get('id','?')}] constrained_emit applied "
        f"(model={resp.get('model','?')}, "
        f"draft={len(draft)} -> emit={len(emitted)} chars, "
        f"step={step_id})"
    )
    # Replace result while preserving metadata.
    new_result = dict(result)
    new_result["result"] = emitted
    new_result["constrained_emit_applied"] = True
    return new_result
