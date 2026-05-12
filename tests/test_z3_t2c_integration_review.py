"""Z3 T2C — extract_signatures + integration_review kind + multifile expander.

Tests:
1. extract_signatures on a Python file returns expected signature shape
2. Cross-file mismatch detection works for simple arity case
3. Soft-skip on non-Python without tree-sitter
4. integration_review kind registered + correct spec
5. Expander expands multi_file step into N sub-steps + 1 integration_review sibling WHEN dial=True
6. Expander pass-through (no expansion) WHEN dial=False
7. Expander pass-through (no expansion) WHEN no rule for (template_id, stack)
8. apply.py injects signatures into step context (mock the LLM call)
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. extract_signatures — basic Python file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_signatures_python_basic(tmp_path):
    source = textwrap.dedent("""\
        def add(x, y):
            return x + y

        async def fetch(url: str, timeout: int = 5) -> bytes:
            ...

        class Processor:
            def process(self, data):
                ...
    """)
    f = tmp_path / "module.py"
    f.write_text(source)

    from mr_roboto.extract_signatures import extract_signatures

    result = await extract_signatures(["module.py"], workspace_path=str(tmp_path))

    assert result["skipped"] == []
    sigs = result["signatures"]["module.py"]
    names = {s["name"] for s in sigs}
    # Should include top-level functions and the class
    assert "add" in names
    assert "fetch" in names
    assert "Processor" in names

    # Check shape of a specific sig
    add_sig = next(s for s in sigs if s["name"] == "add")
    assert add_sig["kind"] == "function"
    assert add_sig["params"] == ["x", "y"]
    assert add_sig["line"] == 1

    fetch_sig = next(s for s in sigs if s["name"] == "fetch")
    assert "url" in fetch_sig["params"]
    assert fetch_sig["returns"] == "bytes"

    # Class method should appear
    process_sig = next(s for s in sigs if s["name"] == "process")
    assert process_sig["kind"] == "method"


@pytest.mark.asyncio
async def test_extract_signatures_missing_file(tmp_path):
    from mr_roboto.extract_signatures import extract_signatures

    result = await extract_signatures(["nonexistent.py"], workspace_path=str(tmp_path))
    assert "nonexistent.py" in result["skipped"]
    assert result["signatures"] == {}


# ---------------------------------------------------------------------------
# 2. Cross-file mismatch detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_signatures_mismatch_detection(tmp_path):
    """File A defines foo(x, y, z), file B calls foo(a, b) → mismatch."""
    file_a = textwrap.dedent("""\
        def foo(x, y, z):
            return x + y + z
    """)
    file_b = textwrap.dedent("""\
        from module_a import foo

        def bar():
            return foo(1, 2)
    """)
    (tmp_path / "module_a.py").write_text(file_a)
    (tmp_path / "module_b.py").write_text(file_b)

    from mr_roboto.extract_signatures import extract_signatures

    result = await extract_signatures(
        ["module_a.py", "module_b.py"],
        workspace_path=str(tmp_path),
    )

    # There should be a mismatch: foo called with 2 args, defined with 3
    mismatches = result["mismatches"]
    assert len(mismatches) >= 1
    m = mismatches[0]
    assert m["kind"] == "arity"
    assert "foo" in m["why"]
    assert m["caller"] == "module_b.py"
    assert m["callee"] == "module_a.py"


@pytest.mark.asyncio
async def test_extract_signatures_no_mismatch_same_arity(tmp_path):
    """No mismatch when arity matches."""
    (tmp_path / "lib.py").write_text("def greet(name):\n    return f'Hi {name}'\n")
    (tmp_path / "main.py").write_text("from lib import greet\ngreet('Alice')\n")

    from mr_roboto.extract_signatures import extract_signatures

    result = await extract_signatures(
        ["lib.py", "main.py"],
        workspace_path=str(tmp_path),
    )
    assert result["mismatches"] == []


# ---------------------------------------------------------------------------
# 3. Soft-skip for TS/JS without tree-sitter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_signatures_ts_skip_without_treesitter(tmp_path):
    """TS files soft-skip when tree-sitter is not installed."""
    ts_file = tmp_path / "component.tsx"
    ts_file.write_text("export const foo = (x: number) => x + 1;\n")

    # Ensure tree_sitter is not importable for this test
    with patch.dict(sys.modules, {"tree_sitter": None}):
        from mr_roboto.extract_signatures import extract_signatures
        result = await extract_signatures(
            ["component.tsx"],
            workspace_path=str(tmp_path),
        )

    assert "component.tsx" in result["skipped"]
    assert "component.tsx" not in result["signatures"]


# ---------------------------------------------------------------------------
# 4. integration_review kind registered + correct spec
# ---------------------------------------------------------------------------


def test_integration_review_in_registry():
    from general_beckman.posthooks import POST_HOOK_REGISTRY, POST_HOOK_KINDS

    assert "integration_review" in POST_HOOK_REGISTRY
    assert "integration_review" in POST_HOOK_KINDS

    spec = POST_HOOK_REGISTRY["integration_review"]
    assert spec.kind == "integration_review"
    assert spec.verb == "integration_reviewer"
    assert spec.default_severity == "blocker"
    # No auto-wire triggers — injected by expander
    assert spec.auto_wire_triggers == []


def test_integration_reviewer_in_no_posthooks():
    from general_beckman.posthooks import _NO_POSTHOOKS_AGENT_TYPES

    assert "integration_reviewer" in _NO_POSTHOOKS_AGENT_TYPES


# ---------------------------------------------------------------------------
# 5. Expander: dial=True → N sub-steps + 1 integration_review sibling
# ---------------------------------------------------------------------------


def test_expander_multifile_expansion_dial_true():
    from src.workflows.engine.expander import _maybe_expand_multifile
    from src.workflows.review_density import MissionDialContext

    dial = MissionDialContext(
        mission_id="test-1",
        multi_file_expansion=True,
        template_id="crud_feature",
        stack_slug="fastapi",
    )

    step = {
        "id": "3.4",
        "name": "Implement user CRUD",
        "phase": "phase_3",
        "depends_on": ["3.3"],
        "context": {"feature_name": "user"},
    }

    result = _maybe_expand_multifile(step, dial, {})
    assert result is not None
    assert len(result) > 1  # N sub-tasks + integration_review

    # Last step should be the integration_review sibling
    ir_step = result[-1]
    assert ir_step["agent"] == "integration_reviewer"
    assert "integration_review" in ir_step.get("post_hooks", [])
    assert ir_step["id"].endswith("integration_review")

    # Sub-tasks should depend on parent's depends_on
    for sub in result[:-1]:
        assert "3.3" in sub["depends_on"]

    # IR step should depend on all sub-task IDs
    sub_ids = [s["id"] for s in result[:-1]]
    for sid in sub_ids:
        assert sid in ir_step["depends_on"]


def test_expander_multifile_all_produces_collected():
    """Integration-review sibling should collect produces from all sub-tasks."""
    from src.workflows.engine.expander import _maybe_expand_multifile
    from src.workflows.review_density import MissionDialContext

    dial = MissionDialContext(
        mission_id="test-2",
        multi_file_expansion=True,
        template_id="crud_feature",
        stack_slug="fastapi",
    )
    step = {"id": "5.1", "name": "Product CRUD", "context": {"feature_name": "product"}}
    result = _maybe_expand_multifile(step, dial, {})
    assert result is not None

    ir_step = result[-1]
    all_produces = ir_step["context"]["all_sub_task_produces"]
    # Should have at least one produces path from each sub-task
    assert len(all_produces) > 0
    # Each sub-step's produces should appear in all_produces
    for sub in result[:-1]:
        for p in sub.get("produces", []):
            assert p in all_produces


# ---------------------------------------------------------------------------
# 6. Expander pass-through when dial=False
# ---------------------------------------------------------------------------


def test_expander_multifile_dial_false():
    from src.workflows.engine.expander import _maybe_expand_multifile
    from src.workflows.review_density import MissionDialContext

    dial = MissionDialContext(
        mission_id="test-3",
        multi_file_expansion=False,  # disabled
        template_id="crud_feature",
        stack_slug="fastapi",
    )
    step = {"id": "2.1", "name": "Some step", "context": {"feature_name": "order"}}
    result = _maybe_expand_multifile(step, dial, {})
    assert result is None  # pass-through


def test_expander_multifile_no_dial():
    from src.workflows.engine.expander import _maybe_expand_multifile

    step = {"id": "2.1", "name": "Some step"}
    result = _maybe_expand_multifile(step, None, {})
    assert result is None


# ---------------------------------------------------------------------------
# 7. Expander pass-through when no rule for (template_id, stack)
# ---------------------------------------------------------------------------


def test_expander_multifile_no_rule():
    from src.workflows.engine.expander import _maybe_expand_multifile
    from src.workflows.review_density import MissionDialContext

    dial = MissionDialContext(
        mission_id="test-4",
        multi_file_expansion=True,
        template_id="nonexistent_template",
        stack_slug="unknown_stack",
    )
    step = {"id": "2.1", "name": "Some step", "context": {}}
    result = _maybe_expand_multifile(step, dial, {})
    assert result is None  # no rule → pass-through


# ---------------------------------------------------------------------------
# 8. apply.py injects signatures into integration_review posthook context
# ---------------------------------------------------------------------------


def test_apply_posthook_integration_review_injects_signatures():
    """_posthook_agent_and_payload returns integration_reviewer with signature keys."""
    from general_beckman.apply import _posthook_agent_and_payload
    from general_beckman.result_router import RequestPostHook

    source_ctx = {
        "all_sub_task_produces": ["app/models/user.py", "app/routers/user.py"],
        "sub_task_ids": ["3.4.model", "3.4.router"],
        "workspace_path": "",
    }
    a = RequestPostHook(
        source_task_id=42,
        kind="integration_review",
        source_ctx=source_ctx,
    )
    source = {"id": 42, "mission_id": "m1"}

    agent_type, payload = _posthook_agent_and_payload(a, source, source_ctx)

    assert agent_type == "integration_reviewer"
    assert payload["posthook_kind"] == "integration_review"
    assert payload["source_task_id"] == 42
    assert "signatures" in payload
    assert "mismatches" in payload
    assert "all_sub_task_produces" in payload
    # sub_task_ids forwarded
    assert payload["sub_task_ids"] == ["3.4.model", "3.4.router"]


# ---------------------------------------------------------------------------
# 9. review_density: MissionDialContext defaults
# ---------------------------------------------------------------------------


def test_mission_dial_context_defaults():
    from src.workflows.review_density import MissionDialContext

    ctx = MissionDialContext()
    assert ctx.multi_file_expansion is False
    assert ctx.template_id is None
    assert ctx.stack_slug is None


def test_to_mission_dial_context_conversion():
    from src.workflows.review_density import to_mission_dial_context

    dials = {
        "multi_file_expansion": True,
        "template_id": "crud_feature",
        "stack_slug": "fastapi",
        "some_extra": 99,
    }
    ctx = to_mission_dial_context("m42", dials)
    assert ctx.mission_id == "m42"
    assert ctx.multi_file_expansion is True
    assert ctx.template_id == "crud_feature"
    assert ctx.stack_slug == "fastapi"
    assert ctx.extra == {"some_extra": 99}


# ---------------------------------------------------------------------------
# 10. multifile.expand_template — slug substitution
# ---------------------------------------------------------------------------


def test_expand_template_fastapi_slug():
    from src.workflows.engine.multifile import expand_template

    step = {"context": {"feature_name": "Order Item"}}
    specs = expand_template("crud_feature", "fastapi", parent_step=step)
    assert specs is not None
    produces_flat = [p for s in specs for p in s.produces]
    # Should have slug 'order_item' in all paths
    for p in produces_flat:
        assert "order_item" in p


def test_expand_template_returns_none_for_unknown():
    from src.workflows.engine.multifile import expand_template

    result = expand_template("no_such_template", "no_such_stack")
    assert result is None
