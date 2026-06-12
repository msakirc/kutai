# tests/i2p/test_i2p_swap_step.py
"""Plan 3 Task 10 — i2p_v3 step 5.35 (swap_placeholder_images) + 5.35.verify.

Deviation from the plan's inline JSON (documented in the plan-execution
report): the canonical artifact schema is
``src.agents.prompt_writer.PROMPT_WRITER_ARTIFACT_SCHEMA`` — a single source
of truth keyed by ``diffusion_prompts`` (the prompt_writer enqueue in
swap_placeholder_images.py imports the same constant). The plan's inline
schema used a top-level ``prompts`` key, which would have drifted from the
constant the executor actually arms ``constrained_emit`` with. Step 5.35
therefore mirrors the constant, and this test asserts against it.
"""
import json

from src.agents.prompt_writer import PROMPT_WRITER_ARTIFACT_SCHEMA


def _steps():
    with open("src/workflows/i2p/i2p_v3.json", encoding="utf-8") as fh:
        return json.load(fh)["steps"]


def test_swap_step_exists():
    assert any(s.get("id") == "5.35" for s in _steps())


def test_verify_step_exists():
    assert any(s.get("id") == "5.35.verify" for s in _steps())


def test_swap_step_shape_uses_mechanical_executor():
    s = next(x for x in _steps() if x["id"] == "5.35")
    assert s["agent"] == "mechanical"
    # v2 fix: executor is "mechanical" (constant), verb in payload.action.
    assert s["executor"] == "mechanical"
    assert s["payload"]["action"] == "swap_placeholder_images"
    assert "5.30c" in s["depends_on"]
    # Soft done_when accepts skipping.
    dw = s["done_when"].lower()
    assert "ok" in dw and "skipped" in dw
    assert s.get("reversibility") == "full"


def test_swap_step_artifact_schema_matches_canonical_constant():
    """v2 fix: prompt_writer needs schema constraint. The artifact_schema on
    this step is what constrained_emit.maybe_apply enforces post-call. It MUST
    equal the canonical PROMPT_WRITER_ARTIFACT_SCHEMA constant (single source
    of truth) so the i2p step and the mr_roboto enqueue spec agree."""
    s = next(x for x in _steps() if x["id"] == "5.35")
    schema = s.get("artifact_schema")
    assert isinstance(schema, dict)
    # Canonical key — NOT the plan's drifted top-level "prompts".
    assert "diffusion_prompts" in schema
    assert schema == PROMPT_WRITER_ARTIFACT_SCHEMA


def test_verify_step_shape():
    s = next(x for x in _steps() if x["id"] == "5.35.verify")
    assert s["agent"] == "mechanical"
    assert s["executor"] == "mechanical"
    assert s["payload"]["action"] == "verify_swap_placeholder_images_shape"
    assert "5.35" in s["depends_on"]


def test_emit_preview_url_depends_on_verify():
    """5.40 must depend on 5.35.verify so the URL only surfaces with a
    verified swap."""
    s = next(x for x in _steps() if x["id"] == "5.40")
    assert "5.35.verify" in s["depends_on"]


def test_phase_is_phase_5():
    for sid in ("5.35", "5.35.verify"):
        s = next(x for x in _steps() if x["id"] == sid)
        assert s["phase"] == "phase_5"
