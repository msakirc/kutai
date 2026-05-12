"""Z3 T1A — cost_band + dial-aware auto_wire scaffold tests.

Covers:
- cost_band field present on all 10 registry kinds with correct values
- MissionDialContext defaults are conservative
- Static auto_wire_triggers behaviour unchanged (regression)
- Callable auto_wire_triggers resolved correctly
- resolve_triggers() returns empty list for no-trigger specs
- _auto_wire_posthooks handles callable form correctly
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    return POST_HOOK_REGISTRY


@pytest.fixture
def default_dial():
    from general_beckman.posthooks import MissionDialContext
    return MissionDialContext()


# ---------------------------------------------------------------------------
# 1. cost_band values on all 10 kinds
# ---------------------------------------------------------------------------

_EXPECTED_COST_BANDS = {
    "verify_artifacts": "cheap",
    "grounding": "cheap",
    "imports_check": "cheap",
    "pattern_lint": "cheap",
    "code_review": "moderate",
    "test_run": "moderate",
    "design_system_check": "moderate",
    "openapi_sync": "moderate",
    "typescript_sync": "moderate",
    "migration_apply": "heavy",
}


def test_cost_band_field_exists_on_all_kinds(registry):
    """Every registry entry must have a cost_band attribute."""
    for kind, spec in registry.items():
        assert hasattr(spec, "cost_band"), f"{kind} missing cost_band"


def test_cost_band_correct_value_for_all_kinds(registry):
    """Each kind has the expected cost_band value per the plan."""
    for kind, expected in _EXPECTED_COST_BANDS.items():
        assert kind in registry, f"Registry missing kind: {kind}"
        actual = registry[kind].cost_band
        assert actual == expected, (
            f"{kind}: expected cost_band={expected!r}, got {actual!r}"
        )


def test_cost_band_is_valid_literal(registry):
    """cost_band must be one of the three allowed values."""
    valid = {"cheap", "moderate", "heavy"}
    for kind, spec in registry.items():
        assert spec.cost_band in valid, (
            f"{kind}.cost_band={spec.cost_band!r} not in {valid}"
        )


def test_registry_has_exactly_10_kinds(registry):
    """Sanity: no kinds were accidentally added or removed."""
    assert len(registry) == 10, f"Expected 10 kinds, got {len(registry)}"


# ---------------------------------------------------------------------------
# 2. MissionDialContext defaults are conservative
# ---------------------------------------------------------------------------

def test_mission_dial_context_default_qa_standard(default_dial):
    assert default_dial.qa_dial == "standard"


def test_mission_dial_context_default_accessibility_off(default_dial):
    assert default_dial.accessibility_dial == "off"


def test_mission_dial_context_default_multi_file_expansion_false(default_dial):
    assert default_dial.multi_file_expansion is False


def test_mission_dial_context_default_integration_replay_off(default_dial):
    assert default_dial.integration_replay == "off"


def test_mission_dial_context_fields_present():
    """MissionDialContext can be constructed with explicit values."""
    from general_beckman.posthooks import MissionDialContext
    ctx = MissionDialContext(
        qa_dial="strict",
        accessibility_dial="warn",
        multi_file_expansion=True,
        integration_replay="smoke",
    )
    assert ctx.qa_dial == "strict"
    assert ctx.accessibility_dial == "warn"
    assert ctx.multi_file_expansion is True
    assert ctx.integration_replay == "smoke"


# ---------------------------------------------------------------------------
# 3. Static auto_wire_triggers — regression (behaviour unchanged)
# ---------------------------------------------------------------------------

def test_static_triggers_resolve_to_same_list(registry):
    """resolve_triggers() on a static-list spec returns the original list."""
    from general_beckman.posthooks import MissionDialContext
    ctx = MissionDialContext()
    spec = registry["grounding"]
    assert spec.resolve_triggers(ctx) == ["*"]


def test_static_triggers_resolve_without_dial_ctx(registry):
    """resolve_triggers(None) uses the default dial context."""
    spec = registry["imports_check"]
    result = spec.resolve_triggers(None)
    assert "*.py" in result
    assert "*.ts" in result
    assert "*.tsx" in result


def test_no_trigger_spec_returns_empty(registry):
    """verify_artifacts has no triggers — should return empty list."""
    spec = registry["verify_artifacts"]
    assert spec.resolve_triggers() == []


def test_code_review_no_triggers(registry):
    """code_review is opt-in — no auto_wire_triggers."""
    spec = registry["code_review"]
    assert spec.resolve_triggers() == []


# ---------------------------------------------------------------------------
# 4. Callable auto_wire_triggers resolved correctly
# ---------------------------------------------------------------------------

def test_callable_triggers_invoked_with_dial_ctx():
    """A callable trigger receives the MissionDialContext and its result is used."""
    from general_beckman.posthooks import MissionDialContext, PostHookSpec

    received: list = []

    def my_trigger(ctx: MissionDialContext) -> list[str]:
        received.append(ctx)
        return ["*.py", "*.ts"]

    spec = PostHookSpec(
        kind="test_callable",
        verb="test_verb",
        auto_wire_triggers=my_trigger,
    )

    dial = MissionDialContext(qa_dial="strict")
    result = spec.resolve_triggers(dial)

    assert result == ["*.py", "*.ts"]
    assert len(received) == 1
    assert received[0].qa_dial == "strict"


def test_callable_triggers_default_dial_when_none():
    """resolve_triggers(None) passes the default MissionDialContext to a callable."""
    from general_beckman.posthooks import MissionDialContext, PostHookSpec

    received: list = []

    def capture_ctx(ctx: MissionDialContext) -> list[str]:
        received.append(ctx)
        return ["*.json"]

    spec = PostHookSpec(
        kind="test_callable_none",
        verb="test_verb",
        auto_wire_triggers=capture_ctx,
    )

    result = spec.resolve_triggers(None)
    assert result == ["*.json"]
    assert received[0].qa_dial == "standard"
    assert received[0].multi_file_expansion is False


def test_callable_can_return_empty_based_on_dial():
    """A callable can suppress triggers by returning [] when dial is 'off'."""
    from general_beckman.posthooks import MissionDialContext, PostHookSpec

    def qa_aware(ctx: MissionDialContext) -> list[str]:
        if ctx.qa_dial == "off":
            return []
        return ["*.py"]

    spec = PostHookSpec(
        kind="test_qa_aware",
        verb="test_verb",
        auto_wire_triggers=qa_aware,
    )

    off_ctx = MissionDialContext(qa_dial="off")
    on_ctx = MissionDialContext(qa_dial="standard")

    assert spec.resolve_triggers(off_ctx) == []
    assert spec.resolve_triggers(on_ctx) == ["*.py"]


def test_callable_strict_dial_adds_extra_globs():
    """A callable can add extra globs when dial is 'strict'."""
    from general_beckman.posthooks import MissionDialContext, PostHookSpec

    def strict_aware(ctx: MissionDialContext) -> list[str]:
        globs = ["*.ts"]
        if ctx.qa_dial == "strict":
            globs = globs + ["*.js", "*.jsx"]
        return globs

    spec = PostHookSpec(
        kind="test_strict",
        verb="test_verb",
        auto_wire_triggers=strict_aware,
    )

    standard_ctx = MissionDialContext(qa_dial="standard")
    strict_ctx = MissionDialContext(qa_dial="strict")

    assert spec.resolve_triggers(standard_ctx) == ["*.ts"]
    assert spec.resolve_triggers(strict_ctx) == ["*.ts", "*.js", "*.jsx"]


# ---------------------------------------------------------------------------
# 5. _auto_wire_posthooks with callable triggers
# ---------------------------------------------------------------------------

def test_auto_wire_callable_trigger_fires(monkeypatch):
    """_auto_wire_posthooks resolves callable triggers and wires matching kinds."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY, PostHookSpec, MissionDialContext
    from src.workflows.engine.expander import _auto_wire_posthooks

    # Patch a single extra spec with a callable trigger into a copy of the registry
    patched_registry = dict(POST_HOOK_REGISTRY)
    call_count = [0]

    def callable_trigger(ctx: MissionDialContext) -> list[str]:
        call_count[0] += 1
        return ["*.custom"]

    patched_registry["custom_hook"] = PostHookSpec(
        kind="custom_hook",
        verb="custom_verb",
        cost_band="cheap",
        auto_wire_triggers=callable_trigger,
    )

    import general_beckman.posthooks as ph_module
    original = ph_module.POST_HOOK_REGISTRY
    monkeypatch.setattr(ph_module, "POST_HOOK_REGISTRY", patched_registry)

    try:
        context = {"produces": ["foo.custom"], "post_hooks": []}
        _auto_wire_posthooks(context)

        assert "custom_hook" in context["post_hooks"], (
            "callable trigger should have wired custom_hook"
        )
        assert call_count[0] >= 1
    finally:
        monkeypatch.setattr(ph_module, "POST_HOOK_REGISTRY", original)


def test_auto_wire_callable_suppressed_trigger_does_not_fire(monkeypatch):
    """When callable returns [], the hook is NOT wired."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY, PostHookSpec, MissionDialContext
    from src.workflows.engine.expander import _auto_wire_posthooks

    patched_registry = dict(POST_HOOK_REGISTRY)

    def suppressed_trigger(ctx: MissionDialContext) -> list[str]:
        return []  # always suppress

    patched_registry["suppressed_hook"] = PostHookSpec(
        kind="suppressed_hook",
        verb="suppressed_verb",
        cost_band="cheap",
        auto_wire_triggers=suppressed_trigger,
    )

    import general_beckman.posthooks as ph_module
    monkeypatch.setattr(ph_module, "POST_HOOK_REGISTRY", patched_registry)

    try:
        context = {"produces": ["anything.py"], "post_hooks": []}
        _auto_wire_posthooks(context)
        assert "suppressed_hook" not in context.get("post_hooks", [])
    finally:
        monkeypatch.setattr(ph_module, "POST_HOOK_REGISTRY", POST_HOOK_REGISTRY)


def test_auto_wire_passes_dial_ctx_to_callable(monkeypatch):
    """_auto_wire_posthooks forwards the dial_ctx argument to callable triggers."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY, PostHookSpec, MissionDialContext
    from src.workflows.engine.expander import _auto_wire_posthooks

    received_dials: list = []

    def capture_dial(ctx: MissionDialContext) -> list[str]:
        received_dials.append(ctx.qa_dial)
        return ["*.py"]

    patched_registry = dict(POST_HOOK_REGISTRY)
    patched_registry["dial_test_hook"] = PostHookSpec(
        kind="dial_test_hook",
        verb="dial_test_verb",
        cost_band="cheap",
        auto_wire_triggers=capture_dial,
    )

    import general_beckman.posthooks as ph_module
    monkeypatch.setattr(ph_module, "POST_HOOK_REGISTRY", patched_registry)

    try:
        strict_dial = MissionDialContext(qa_dial="strict")
        context = {"produces": ["app.py"], "post_hooks": []}
        _auto_wire_posthooks(context, dial_ctx=strict_dial)

        assert "strict" in received_dials, (
            "Expected dial_ctx with qa_dial='strict' to be forwarded"
        )
    finally:
        monkeypatch.setattr(ph_module, "POST_HOOK_REGISTRY", POST_HOOK_REGISTRY)


# ---------------------------------------------------------------------------
# 6. Static auto_wire regression — existing missions unaffected
# ---------------------------------------------------------------------------

def test_grounding_auto_wires_on_any_produces():
    """grounding (auto_wire_triggers=['*']) fires for any produce path."""
    from src.workflows.engine.expander import _auto_wire_posthooks

    context = {"produces": ["src/app/foo.py"], "post_hooks": []}
    _auto_wire_posthooks(context)
    assert "grounding" in context["post_hooks"]


def test_imports_check_auto_wires_on_py_files():
    from src.workflows.engine.expander import _auto_wire_posthooks

    context = {"produces": ["src/models/user.py"], "post_hooks": []}
    _auto_wire_posthooks(context)
    assert "imports_check" in context["post_hooks"]


def test_imports_check_does_not_wire_on_json():
    from src.workflows.engine.expander import _auto_wire_posthooks

    context = {"produces": ["config.json"], "post_hooks": []}
    _auto_wire_posthooks(context)
    assert "imports_check" not in context["post_hooks"]


def test_design_system_check_auto_wires_on_tsx():
    from src.workflows.engine.expander import _auto_wire_posthooks

    context = {"produces": ["components/Button.tsx"], "post_hooks": []}
    _auto_wire_posthooks(context)
    assert "design_system_check" in context["post_hooks"]


def test_migration_apply_auto_wires_on_sql():
    from src.workflows.engine.expander import _auto_wire_posthooks

    context = {"produces": ["migrations/001_create_users.sql"], "post_hooks": []}
    _auto_wire_posthooks(context)
    assert "migration_apply" in context["post_hooks"]


def test_existing_hooks_not_duplicated():
    """Idempotency: if grounding already in post_hooks, it isn't added again."""
    from src.workflows.engine.expander import _auto_wire_posthooks

    context = {"produces": ["app.py"], "post_hooks": ["grounding"]}
    _auto_wire_posthooks(context)
    assert context["post_hooks"].count("grounding") == 1


def test_no_behavior_change_no_produces():
    """Steps with empty/absent produces are unchanged."""
    from src.workflows.engine.expander import _auto_wire_posthooks

    context: dict = {"produces": [], "post_hooks": []}
    # _auto_wire_posthooks is only called when produces is non-empty in the
    # expander, but calling it directly with empty produces should be safe.
    _auto_wire_posthooks(context)
    assert context["post_hooks"] == []
