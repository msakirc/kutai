"""Phase 3 Task 12 Batch H — reflection + constrained-emit Foundry migration.

Locks:
  * reflection block CONTENT now lives in the Foundry leaf
    (prompt_foundry.reflection_blocks); coulson keeps composition. The
    back-compat names (coulson.reflection / coulson.posthooks.reflection_posthook)
    must resolve to the SAME leaf objects.
  * build_reflection_prompt / build_reflect_messages composition is char-exact
    vs the frozen pre-migration strings.
  * build_emit_messages text comes from rubrics/constrained_emit.yaml via
    build_messages and is char-exact vs the original.
"""
import prompt_foundry as pf
from coulson.posthooks.reflection_posthook import (
    REFLECTION_BLOCKS,
    STACK_BLOCKS,
    LAYER_BLOCKS,
    REFLECT_SYSTEM_BASE,
    build_reflection_prompt,
    build_reflect_messages,
    build_emit_messages,
)
import coulson.reflection as cref


# ── back-compat identity ────────────────────────────────────────────────────

def test_posthook_names_are_leaf_objects():
    assert REFLECTION_BLOCKS is pf.REFLECTION_BLOCKS
    assert STACK_BLOCKS is pf.STACK_BLOCKS
    assert LAYER_BLOCKS is pf.LAYER_BLOCKS
    assert REFLECT_SYSTEM_BASE is pf.REFLECT_SYSTEM_BASE


def test_coulson_reflection_reexports_are_leaf_objects():
    assert cref.REFLECTION_BLOCKS is pf.REFLECTION_BLOCKS
    assert cref.STACK_BLOCKS is pf.STACK_BLOCKS
    assert cref.LAYER_BLOCKS is pf.LAYER_BLOCKS


# ── composition char-exact ──────────────────────────────────────────────────

def test_build_reflection_prompt_char_exact_agent_stack_layer():
    result = build_reflection_prompt("coder", 2, stack="fastapi+nextjs", layer="domain")
    expected = "\n\n".join([
        f"[iteration 2] {REFLECTION_BLOCKS['coder']}",
        STACK_BLOCKS["fastapi"],
        STACK_BLOCKS["nextjs"],
        LAYER_BLOCKS["domain"],
    ])
    assert result == expected


def test_build_reflection_prompt_generic_fallback_char_exact():
    result = build_reflection_prompt("some_unknown_agent", 1)
    assert result == f"[iteration 1] {pf._GENERIC_REFLECTION_BLOCK}"


def test_build_reflect_messages_char_exact():
    task = {"title": "T1", "description": "D1"}
    msgs = build_reflect_messages(task, "result text", checklist="CHECK")
    assert msgs[0] == {
        "role": "system",
        "content": f"{REFLECT_SYSTEM_BASE}\n\nCHECK",
    }
    assert msgs[1] == {
        "role": "user",
        "content": "Task: T1\nDescription: D1\n\nResponse to review:\nresult text",
    }


def test_build_reflect_messages_no_checklist_char_exact():
    msgs = build_reflect_messages({}, None, checklist=None)
    assert msgs[0]["content"] == REFLECT_SYSTEM_BASE
    assert msgs[1]["content"] == "Task: \nDescription: \n\nResponse to review:\n"


# ── constrained-emit char-exact ─────────────────────────────────────────────

def test_build_emit_messages_char_exact():
    import json
    rf = {
        "type": "json_schema",
        "json_schema": {
            "name": "step_x",
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
        },
    }
    draft = 'some draft with "quotes" and {braces}'
    msgs = build_emit_messages(draft, rf)
    schema_text = json.dumps(rf["json_schema"]["schema"], ensure_ascii=False, indent=2)
    expected_system = (
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
    expected_user = (
        f"Schema:\n```json\n{schema_text}\n```\n\n"
        f"Draft to fix:\n```\n{draft}\n```\n\n"
        f"Emit the final JSON now."
    )
    assert msgs[0] == {"role": "system", "content": expected_system}
    assert msgs[1] == {"role": "user", "content": expected_user}


def test_build_emit_messages_empty_draft():
    """None / empty draft → empty draft block (no crash, no truncation)."""
    rf = {
        "type": "json_schema",
        "json_schema": {"name": "step_x", "schema": {"type": "object"}},
    }
    msgs = build_emit_messages(None, rf)
    assert "Draft to fix:\n```\n\n```" in msgs[1]["content"]
