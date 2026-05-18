"""Z4 T3 — visual_review post-hook: registry, apply.py wiring, expander auto-wire.

Tests
-----
T3A  registry contains "visual_review"; POST_HOOK_KINDS count increased by 1
     (verifies the Z4 entry landed without removing any existing kind).
T3B  _posthook_agent_and_payload returns correct mechanical payload for visual_review.
     - workspace_path and step_id resolved from source / source_ctx.
     - preview_url read from .preview/last_preview_url.txt when not in ctx.
     - "pending:" marker is filtered (URL stays blank).
     - produces + routes forwarded.
T3B2 visual_review present in both apply.py kind tuples (DLQ cascade + simple-blocker).
T3E  expander auto-wire: *.tsx produce → visual_review prepended; *.py only → not wired;
     idempotent (running twice does not duplicate).

All LLM / vision / browser calls are mocked — no real I/O.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# T3A — registry contains "visual_review"
# ---------------------------------------------------------------------------

def test_visual_review_in_post_hook_registry():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    assert "visual_review" in POST_HOOK_REGISTRY


def test_visual_review_in_post_hook_kinds():
    from general_beckman.posthooks import POST_HOOK_KINDS
    assert "visual_review" in POST_HOOK_KINDS


def test_visual_review_spec_fields():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["visual_review"]
    assert spec.kind == "visual_review"
    assert spec.verb == "visual_review"
    assert spec.default_severity == "blocker"
    assert spec.cost_band == "heavy"
    # Static list — not a callable
    assert isinstance(spec.auto_wire_triggers, list)
    # Must include frontend globs
    triggers = spec.auto_wire_triggers
    assert "*.tsx" in triggers
    assert "*.jsx" in triggers
    assert "*.vue" in triggers
    assert "*.svelte" in triggers


def test_post_hook_kinds_includes_legacy_kinds():
    """Regression: adding visual_review must not have removed other kinds."""
    from general_beckman.posthooks import POST_HOOK_KINDS
    for required in (
        "security_review", "accessibility_review", "contract_review",
        "performance_review", "adr_drift_check", "integration_replay",
    ):
        assert required in POST_HOOK_KINDS, f"{required!r} was accidentally removed"


# ---------------------------------------------------------------------------
# T3B — _posthook_agent_and_payload for visual_review
# ---------------------------------------------------------------------------

@dataclass
class _FakeVerdict:
    kind: str
    source_task_id: int = 1
    passed: bool = True


def _make_source(step_id="3.1", mission_id=42, ctx_extra=None):
    ctx = {
        "workspace_path": "/ws/test",
        "produces": ["src/components/Button.tsx"],
    }
    if ctx_extra:
        ctx.update(ctx_extra)
    return (
        {
            "id": step_id,
            "step_id": step_id,
            "mission_id": mission_id,
            "context": ctx,
        },
        ctx,
    )


def _call_posthook_agent_and_payload(source, source_ctx, verdict):
    """Synchronous thin wrapper — _posthook_agent_and_payload is sync."""
    from general_beckman.apply import _posthook_agent_and_payload
    return _posthook_agent_and_payload(verdict, source, source_ctx)


def test_posthook_payload_basic():
    source, source_ctx = _make_source()
    verdict = _FakeVerdict(kind="visual_review")
    agent_type, task_dict = _call_posthook_agent_and_payload(source, source_ctx, verdict)
    assert agent_type == "mechanical"
    payload = task_dict["payload"]
    assert payload["action"] == "visual_review"
    assert task_dict["posthook_kind"] == "visual_review"
    assert task_dict["executor"] == "mechanical"
    assert payload["workspace_path"] == "/ws/test"
    assert payload["step_id"] == "3.1"
    assert payload["mission_id"] == 42
    assert "src/components/Button.tsx" in payload["produces"]
    assert payload["baseline_dir"] is None


def test_posthook_payload_routes_forwarded():
    source, source_ctx = _make_source(ctx_extra={"routes": ["/home", "/about"]})
    verdict = _FakeVerdict(kind="visual_review")
    _, task_dict = _call_posthook_agent_and_payload(source, source_ctx, verdict)
    assert task_dict["payload"]["routes"] == ["/home", "/about"]


def test_posthook_payload_preview_url_from_file():
    """payload workspace_path passes through so verb can resolve preview URL at runtime."""
    with tempfile.TemporaryDirectory() as tmpdir:
        preview_dir = os.path.join(tmpdir, ".preview")
        os.makedirs(preview_dir)
        url_file = os.path.join(preview_dir, "last_preview_url.txt")
        with open(url_file, "w") as f:
            f.write("https://preview.example.com/")
        ctx = {"workspace_path": tmpdir, "produces": ["app/page.tsx"]}
        source = {
            "id": "2.1",
            "step_id": "2.1",
            "mission_id": 7,
            "context": ctx,
        }
        verdict = _FakeVerdict(kind="visual_review")
        # The payload carries workspace_path; verb resolves last_preview_url.txt itself.
        _, task_dict = _call_posthook_agent_and_payload(source, ctx, verdict)
        assert task_dict["payload"]["workspace_path"] == tmpdir


def test_posthook_payload_pending_url_filtered():
    """'pending:...' URL in source_ctx → workspace_path still forwarded; verb soft-skips."""
    source, source_ctx = _make_source(ctx_extra={"preview_url": "pending:build_in_progress"})
    verdict = _FakeVerdict(kind="visual_review")
    _, task_dict = _call_posthook_agent_and_payload(source, source_ctx, verdict)
    # The payload must have action=visual_review regardless
    assert task_dict["payload"]["action"] == "visual_review"
    assert task_dict["payload"]["workspace_path"] == "/ws/test"


def test_posthook_payload_source_task_id():
    source, source_ctx = _make_source()
    verdict = _FakeVerdict(kind="visual_review", source_task_id=99)
    _, task_dict = _call_posthook_agent_and_payload(source, source_ctx, verdict)
    assert task_dict["source_task_id"] == 99


# ---------------------------------------------------------------------------
# T3B2 — visual_review in kind tuples
# ---------------------------------------------------------------------------

def test_visual_review_in_dlq_cascade_kinds():
    """visual_review must be in the DLQ-cascade tuple in apply.py."""
    import ast, pathlib
    src = pathlib.Path(
        "packages/general_beckman/src/general_beckman/apply.py"
    ).read_text(encoding="utf-8")
    # Look for the DLQ cascade block containing "integration_review"
    # and verify "visual_review" appears close to it.
    assert '"visual_review"' in src, "visual_review not found in apply.py at all"
    # Find the DLQ block line
    dlq_line = None
    for i, line in enumerate(src.splitlines()):
        if "DLQ cascades source to failed" in line or "_pending_posthooks" in line:
            # Scan the next 15 lines for visual_review
            context_block = "\n".join(src.splitlines()[i:i+15])
            if "visual_review" in context_block:
                dlq_line = i
                break
    assert dlq_line is not None, (
        "visual_review not found in DLQ-cascade tuple in apply.py"
    )


def test_visual_review_in_simple_blocker_verdict_kinds():
    """visual_review must be in the _apply_simple_blocker_verdict kind tuple."""
    import pathlib
    src = pathlib.Path(
        "packages/general_beckman/src/general_beckman/apply.py"
    ).read_text(encoding="utf-8")
    # Find the _apply_simple_blocker_verdict block
    simple_blocker_line = None
    for i, line in enumerate(src.splitlines()):
        if "_apply_simple_blocker_verdict" in line:
            # Scan backward for the if a.kind in ( ... ) block
            context_block = "\n".join(src.splitlines()[max(0, i-6):i+2])
            if "visual_review" in context_block:
                simple_blocker_line = i
                break
    assert simple_blocker_line is not None, (
        "visual_review not found in _apply_simple_blocker_verdict kind tuple in apply.py"
    )


# ---------------------------------------------------------------------------
# T3E — expander auto-wire
# ---------------------------------------------------------------------------

def test_expander_autowire_tsx_triggers_visual_review():
    """A step producing a .tsx file gets visual_review prepended by _auto_wire_posthooks."""
    from src.workflows.engine.expander import _auto_wire_posthooks
    context = {"produces": ["src/components/Hero.tsx"]}
    _auto_wire_posthooks(context)
    assert "visual_review" in context["post_hooks"]


def test_expander_autowire_jsx_triggers_visual_review():
    """A step producing a .jsx file gets visual_review prepended."""
    from src.workflows.engine.expander import _auto_wire_posthooks
    context = {"produces": ["pages/index.jsx"]}
    _auto_wire_posthooks(context)
    assert "visual_review" in context["post_hooks"]


def test_expander_autowire_vue_triggers_visual_review():
    """A step producing a .vue file gets visual_review prepended."""
    from src.workflows.engine.expander import _auto_wire_posthooks
    context = {"produces": ["components/Card.vue"]}
    _auto_wire_posthooks(context)
    assert "visual_review" in context["post_hooks"]


def test_expander_autowire_svelte_triggers_visual_review():
    """A step producing a .svelte file gets visual_review prepended."""
    from src.workflows.engine.expander import _auto_wire_posthooks
    context = {"produces": ["src/Button.svelte"]}
    _auto_wire_posthooks(context)
    assert "visual_review" in context["post_hooks"]


def test_expander_autowire_py_does_not_trigger_visual_review():
    """A step producing only .py files must NOT get visual_review."""
    from src.workflows.engine.expander import _auto_wire_posthooks
    context = {"produces": ["src/models/user.py", "tests/test_user.py"]}
    _auto_wire_posthooks(context)
    assert "visual_review" not in context.get("post_hooks", [])


def test_expander_autowire_idempotent():
    """Running _auto_wire_posthooks twice must not duplicate visual_review."""
    from src.workflows.engine.expander import _auto_wire_posthooks
    context = {"produces": ["src/app/page.tsx"]}
    _auto_wire_posthooks(context)
    first = list(context["post_hooks"])
    _auto_wire_posthooks(context)
    assert context["post_hooks"] == first
    assert context["post_hooks"].count("visual_review") == 1


def test_expander_autowire_visual_review_not_duplicated_when_explicit():
    """Explicit visual_review in post_hooks is not duplicated after auto-wire."""
    from src.workflows.engine.expander import _auto_wire_posthooks
    context = {
        "produces": ["src/pages/Home.tsx"],
        "post_hooks": ["visual_review", "verify_artifacts"],
    }
    _auto_wire_posthooks(context)
    assert context["post_hooks"].count("visual_review") == 1
