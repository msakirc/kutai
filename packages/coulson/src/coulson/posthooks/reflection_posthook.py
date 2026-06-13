"""Post-hook prompt layer for the reflection + constrained-emit LLM children.

SP3b Task 5: ``self_reflect`` and ``constrained_emit`` are post-hook kinds that
spawn a raw_dispatch LLM child (on the husam worker) and REWRITE the source
task's result via the verdict path. This module is the prompt layer for those
two children — it lives next to ``src/core/grading.py`` and ``src/core/
code_review.py`` (the other post-hook prompt builders).

It owns:

* ``REFLECTION_BLOCKS`` / ``STACK_BLOCKS`` / ``LAYER_BLOCKS`` /
  ``build_reflection_prompt`` — the per-agent / stack / layer self-check
  checklists. Moved here from ``coulson/reflection.py`` (which now re-exports
  them for back-compat until Task 7 removes coulson's inline self_reflect
  caller).
* ``build_reflect_messages`` — the reviewer system+user message build for the
  reflection child (lifted from the inline ``self_reflect`` in coulson).
* ``build_emit_messages`` / ``schema_response_format`` / ``should_skip_emit`` —
  the constrained-emit prompt + response_format + skip-when-conforms predicate
  (lifted from ``src/workflows/engine/constrained_emit.py``).

These are pure builders — no DB, no LLM dispatch. The apply layer
(``_enqueue_posthook_llm_child``) builds a raw_dispatch child spec from them;
the continuation handlers (``posthook_continuations.py``) parse the child
response into a rewrite verdict.
"""
from __future__ import annotations

import json

# Block CONTENT (the TEXT) lives in the Foundry leaf as pure data; this module
# keeps the pick-by-key-and-glue COMPOSITION logic (build_reflection_prompt /
# build_reflect_messages). Re-imported here so the historical
# coulson.posthooks.reflection_posthook.{STACK_BLOCKS,...} names — and the
# coulson.reflection / src.core.reflection_posthook re-exports of them —
# resolve to the same objects. (Phase 3 Task 12 Batch H — HYBRID.)
from finch import (  # noqa: F401  re-exported for back-compat
    STACK_BLOCKS,
    LAYER_BLOCKS,
    REFLECTION_BLOCKS,
    REFLECT_SYSTEM_BASE,
    _GENERIC_REFLECTION_BLOCK,
)


def build_reflection_prompt(
    agent_name: str,
    iteration: int,
    stack: str | None = None,
    layer: str | None = None,
) -> str:
    """Return a role-specific self-check checklist for *agent_name*.

    Falls back to a generic prompt for agents without a dedicated checklist
    so that all currently enabled agents (researcher, writer, shopping_advisor,
    deal_analyst, product_researcher) continue to work unchanged.

    Parameters
    ----------
    stack:
        Optional stack identifier (e.g. ``"fastapi"``) or ``+``-joined
        multi-stack string (e.g. ``"fastapi+nextjs"``).  When set and the
        stack key exists in :data:`STACK_BLOCKS`, the relevant fragment is
        appended after the role block.  Multi-stack: each token is looked up
        independently; matched blocks are deduplicated and concatenated.
    layer:
        Optional layer tag from :func:`src.tools.inspect_layer.inspect_layer`
        (one of ``"domain"``, ``"adapter"``, ``"infra"``, ``"ui"``,
        ``"test"``, ``"unknown"``).  When set and :data:`LAYER_BLOCKS` has a
        non-empty entry for it, the layer block is appended after stack
        blocks (Z3 T4C).
    """
    block = REFLECTION_BLOCKS.get(agent_name, _GENERIC_REFLECTION_BLOCK)
    parts = [f"[iteration {iteration}] {block}"]

    if stack:
        seen: set[str] = set()
        for token in stack.split("+"):
            token = token.strip().lower()
            if token and token not in seen and token in STACK_BLOCKS:
                parts.append(STACK_BLOCKS[token])
                seen.add(token)

    if layer:
        layer_block = LAYER_BLOCKS.get(layer.strip().lower(), "")
        if layer_block:
            parts.append(layer_block)

    return "\n\n".join(parts)


# ────────────────────────────────────────────────────────────────────────────
# Reflection child — reviewer message build (lifted from coulson self_reflect).
# REFLECT_SYSTEM_BASE is the reviewer system base; its TEXT now lives in the
# Foundry leaf (finch.reflection_blocks) and is imported at the top.
# ────────────────────────────────────────────────────────────────────────────


def build_reflect_messages(
    task: dict, result: str, checklist: str | None = None,
) -> list[dict]:
    """Build the reviewer system+user messages for the reflection child.

    Mirrors the inline message build from ``coulson.reflection.self_reflect``
    (~314-333). ``checklist`` is the per-agent self-check block from
    :func:`build_reflection_prompt`, appended to the reviewer system message.
    """
    system_content = (
        f"{REFLECT_SYSTEM_BASE}\n\n{checklist}" if checklist else REFLECT_SYSTEM_BASE
    )
    # NO TRUNCATION of any reviewer input, ever — same rule as the grader
    # (src/core/grading.py). A self-reflection that judges the draft against a
    # truncated spec (or a draft cut at 3000 chars) flags the cut-off tail as
    # "missing" and "fixes" against a partial contract. Feed both whole.
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": (
            f"Task: {task.get('title', '')}\n"
            f"Description: {task.get('description') or ''}\n\n"
            f"Response to review:\n{result or ''}"
        )},
    ]


# ────────────────────────────────────────────────────────────────────────────
# Constrained-emit child — response_format + message build + skip predicate.
# ────────────────────────────────────────────────────────────────────────────

def schema_response_format(artifact_schema: dict, step_id: str = "artifact"):
    """Build a json_schema response_format for *artifact_schema*.

    Returns the response_format dict, or ``None`` when the schema is
    unconstrainable (markdown / string) — caller skips the emit in that case.
    Lifted from constrained_emit.py ~70-82.
    """
    if not isinstance(artifact_schema, dict):
        return None
    from src.workflows.engine.json_schema_translator import build_response_format
    # JSON Schema 'name' must be alphanumeric+underscore; sanitize.
    safe_name = "step_" + "".join(
        c if c.isalnum() else "_" for c in str(step_id)
    )
    return build_response_format(artifact_schema, name=safe_name)


def should_skip_emit(draft: str, artifact_schema: dict) -> bool:
    """Skip the emit only when the draft already PASSES the full artifact-schema
    validation — i.e. exactly when the deterministic schema gate (#1) would pass.

    The old predicate skipped on top-level artifact-NAME presence alone. That
    let an internally-incomplete draft through: a ``monetization_strategy``
    object missing 3 of its 6 nested ``required_fields`` (#289737) or a single
    object where an 8-15 item array is required (#289735) both carry the right
    top-level key, so the emit was skipped — then the draft flowed to the grade
    / schema gate and DLQ'd on a blind retry (the grader's bare COMPLETE:NO
    gives capable producers no actionable reason to fix the shape).

    Gating on the SAME validator the schema gate uses keeps the two decisions in
    lock-step: the emit fires precisely when the gate would reject, so the
    constrained re-emit (which forces the missing nested fields / correct array
    shape on a json_schema-capable model) lands BEFORE the gate. A draft that
    already validates is left untouched — no needless re-emit, so the old
    tail-compression worry never triggers (it only re-emits failing drafts).
    """
    if not isinstance(artifact_schema, dict) or not artifact_schema:
        return True  # nothing constrainable — emit would be a no-op
    from src.workflows.engine.hooks import validate_artifact_schema
    try:
        ok, _ = validate_artifact_schema(draft, artifact_schema)
    except Exception:  # noqa: BLE001 — a validator error must not block the emit
        return False
    return bool(ok)


def build_emit_messages(draft: str, response_format: dict) -> list[dict]:
    """Build the structured-emitter system+user messages for the emit child.

    Mirrors the inline prompt build from constrained_emit.py ~124-143.
    ``response_format`` is the dict returned by :func:`schema_response_format`.
    """
    schema_text = json.dumps(
        response_format["json_schema"]["schema"],
        ensure_ascii=False,
        indent=2,
    )
    # NO TRUNCATION: the draft IS the artifact being re-serialized — lopping its
    # tail drops required fields, and the prompt explicitly orders "do not
    # summarize away content". Feed the whole draft; an oversized one is a
    # model-context concern handled by the caller's size-derived estimate.
    draft_for_prompt = draft or ""
    # Prompt TEXT lives in the Foundry rubric (rubrics/constrained_emit.yaml);
    # this builder owns only the dynamic schema_text / draft fields. The schema
    # response_format object + should_skip_emit stay in coulson (not prompt
    # content). (Phase 3 Task 12 Batch H.)
    from finch import build_messages
    return build_messages(
        "constrained_emit",
        {"schema_text": schema_text, "draft": draft_for_prompt},
    )
