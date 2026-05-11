"""Z2 T1B — Expander auto-wire per registry triggers.

Verifies:
- Step with produces matching a fake registered kind gets it prepended.
- Running expander twice yields the same list (idempotent).
- Non-matching produce → no prepend.
- Existing grounding behavior preserved (grounding auto-wires on any produces).
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from src.workflows.engine.expander import _auto_wire_posthooks, expand_steps_to_tasks
from general_beckman.posthooks import POST_HOOK_REGISTRY, PostHookSpec


# ---------------------------------------------------------------------------
# _auto_wire_posthooks unit tests
# ---------------------------------------------------------------------------

def test_grounding_autowired_on_any_produce():
    context = {"produces": ["src/foo.py"]}
    _auto_wire_posthooks(context)
    assert "grounding" in context["post_hooks"]
    assert context["post_hooks"][0] == "grounding"


def test_grounding_not_duplicated_when_already_listed():
    context = {
        "produces": ["src/foo.py"],
        "post_hooks": ["grounding"],
    }
    _auto_wire_posthooks(context)
    assert context["post_hooks"].count("grounding") == 1


def test_no_autowire_without_produces():
    """_auto_wire_posthooks is only called when produces is non-empty,
    but an explicit empty produces list should not add anything."""
    context = {"produces": []}
    _auto_wire_posthooks(context)
    # No post_hooks key should be added (or it should be empty)
    assert context.get("post_hooks") in (None, [])


def test_fake_kind_triggers_on_py_glob(monkeypatch):
    """A fake registered kind with *.py trigger fires on foo.py produce."""
    fake_spec = PostHookSpec(
        kind="test_imports_check",
        verb="imports_check",
        auto_wire_triggers=["*.py"],
        description="fake T1B test kind",
    )
    patched = {**POST_HOOK_REGISTRY, "test_imports_check": fake_spec}
    with patch("src.workflows.engine.expander.POST_HOOK_REGISTRY", patched, create=True):
        # Also need to patch where _auto_wire_posthooks imports it from
        import src.workflows.engine.expander as _exp_mod
        old_import = None
        # Patch the lazy import inside _auto_wire_posthooks via sys.modules workaround:
        # The function does `from general_beckman.posthooks import POST_HOOK_REGISTRY`
        # so we patch the module attribute directly.
        import general_beckman.posthooks as _ph_mod
        original = _ph_mod.POST_HOOK_REGISTRY
        _ph_mod.POST_HOOK_REGISTRY = patched
        try:
            context = {"produces": ["models/auth.py"]}
            _auto_wire_posthooks(context)
        finally:
            _ph_mod.POST_HOOK_REGISTRY = original

    assert "test_imports_check" in context["post_hooks"]
    assert "grounding" in context["post_hooks"]


def test_non_matching_produce_does_not_add_fake_kind(monkeypatch):
    """A *.py trigger should NOT fire for a .json produce."""
    import general_beckman.posthooks as _ph_mod
    fake_spec = PostHookSpec(
        kind="py_only_kind",
        verb="py_check",
        auto_wire_triggers=["*.py"],
        description="only python files",
    )
    patched = {**POST_HOOK_REGISTRY, "py_only_kind": fake_spec}
    original = _ph_mod.POST_HOOK_REGISTRY
    _ph_mod.POST_HOOK_REGISTRY = patched
    try:
        context = {"produces": ["output/report.json"]}
        _auto_wire_posthooks(context)
    finally:
        _ph_mod.POST_HOOK_REGISTRY = original

    assert "py_only_kind" not in context.get("post_hooks", [])
    # grounding (wildcard "*") should still be there
    assert "grounding" in context.get("post_hooks", [])


def test_idempotent_double_call():
    """Calling _auto_wire_posthooks twice must not duplicate kinds."""
    context = {"produces": ["src/app.py"]}
    _auto_wire_posthooks(context)
    first = list(context["post_hooks"])
    _auto_wire_posthooks(context)
    assert context["post_hooks"] == first


def test_any_of_produce_triggers_grounding():
    """any_of produce entries (list of strings) contribute all alternatives for matching."""
    context = {"produces": [["src/foo.py", "src/bar.py"]]}
    _auto_wire_posthooks(context)
    assert "grounding" in context["post_hooks"]


# ---------------------------------------------------------------------------
# Integration via expand_steps_to_tasks
# ---------------------------------------------------------------------------

def test_expand_step_with_produces_gets_grounding():
    step = {
        "id": "1.1",
        "phase": "phase_1",
        "name": "write_module",
        "agent": "coder",
        "instruction": "Write module.",
        "depends_on": [],
        "input_artifacts": [],
        "output_artifacts": [],
        "produces": ["src/module.py"],
    }
    tasks = expand_steps_to_tasks([step], mission_id="m1")
    post_hooks = tasks[0]["context"]["post_hooks"]
    assert "grounding" in post_hooks


def test_expand_step_without_produces_no_grounding():
    step = {
        "id": "1.2",
        "phase": "phase_1",
        "name": "no_produce",
        "agent": "planner",
        "instruction": "Plan.",
        "depends_on": [],
        "input_artifacts": [],
        "output_artifacts": [],
    }
    tasks = expand_steps_to_tasks([step], mission_id="m1")
    post_hooks = tasks[0]["context"].get("post_hooks")
    assert not post_hooks or "grounding" not in post_hooks


def test_expand_step_explicit_grounding_not_duplicated():
    step = {
        "id": "1.3",
        "phase": "phase_1",
        "name": "write_module",
        "agent": "coder",
        "instruction": "Write.",
        "depends_on": [],
        "input_artifacts": [],
        "output_artifacts": [],
        "produces": ["src/app.py"],
        "post_hooks": ["grounding", "verify_artifacts"],
    }
    tasks = expand_steps_to_tasks([step], mission_id="m1")
    post_hooks = tasks[0]["context"]["post_hooks"]
    assert post_hooks.count("grounding") == 1
    assert "verify_artifacts" in post_hooks
