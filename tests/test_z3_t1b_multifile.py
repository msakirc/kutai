"""Z3 T1B — tests for src/workflows/engine/multifile.py scaffold."""

from __future__ import annotations

import dataclasses

import pytest

from src.workflows.engine.multifile import (
    MULTI_FILE_RULES,
    FILE_ROLE_TO_PATH,
    SubTaskSpec,
    expand_template,
)


# ---------------------------------------------------------------------------
# MULTI_FILE_RULES shape
# ---------------------------------------------------------------------------

def test_rules_contains_backend_service_fastapi():
    key = ("backend_service", "fastapi+nextjs")
    assert key in MULTI_FILE_RULES
    roles = MULTI_FILE_RULES[key]
    assert roles == [
        "model", "schema", "service", "repository",
        "error_mapper", "fixtures", "tests",
    ]


def test_rules_contains_frontend_component_fastapi():
    key = ("frontend_component", "fastapi+nextjs")
    assert key in MULTI_FILE_RULES
    roles = MULTI_FILE_RULES[key]
    assert roles == ["component", "hook", "story", "test"]


# ---------------------------------------------------------------------------
# expand_template — correct count + ordering
# ---------------------------------------------------------------------------

def test_expand_backend_service_count_and_order():
    parent = {"step_id": "3.auth_service", "produces": [], "post_hooks": []}
    specs = expand_template("backend_service", "fastapi+nextjs", parent, {})
    assert specs is not None
    assert len(specs) == 7
    roles = [s.step_id.split(".")[-1] for s in specs]
    assert roles == [
        "model", "schema", "service", "repository",
        "error_mapper", "fixtures", "tests",
    ]


def test_expand_frontend_component_count_and_order():
    parent = {"step_id": "5.user_card", "produces": [], "post_hooks": []}
    specs = expand_template("frontend_component", "fastapi+nextjs", parent, {})
    assert specs is not None
    assert len(specs) == 4
    roles = [s.step_id.split(".")[-1] for s in specs]
    assert roles == ["component", "hook", "story", "test"]


# ---------------------------------------------------------------------------
# expand_template — None for unknown combo
# ---------------------------------------------------------------------------

def test_expand_returns_none_for_unknown_template():
    parent = {"step_id": "x.step", "produces": [], "post_hooks": []}
    result = expand_template("unknown_template", "fastapi+nextjs", parent, {})
    assert result is None


def test_expand_returns_none_for_unknown_stack():
    parent = {"step_id": "x.step", "produces": [], "post_hooks": []}
    result = expand_template("backend_service", "django+react", parent, {})
    assert result is None


def test_expand_returns_none_for_both_unknown():
    parent = {"step_id": "x.step", "produces": [], "post_hooks": []}
    result = expand_template("nope", "nope", parent, {})
    assert result is None


# ---------------------------------------------------------------------------
# SubTaskSpec — dataclasses.asdict round-trip
# ---------------------------------------------------------------------------

def test_subtaskspec_asdict_round_trip():
    spec = SubTaskSpec(
        step_id="3.auth_service.model",
        template_id="backend_service",
        target_file="src/models/auth_service.py",
        produces=["src/models/auth_service.py"],
        inherited_post_hooks=["migration_apply"],
        inherited_from="3.auth_service",
    )
    d = dataclasses.asdict(spec)
    assert d["step_id"] == "3.auth_service.model"
    assert d["template_id"] == "backend_service"
    assert d["target_file"] == "src/models/auth_service.py"
    assert d["produces"] == ["src/models/auth_service.py"]
    assert d["inherited_post_hooks"] == ["migration_apply"]
    assert d["inherited_from"] == "3.auth_service"


# ---------------------------------------------------------------------------
# step_id format is parent.role
# ---------------------------------------------------------------------------

def test_step_id_format():
    parent = {"step_id": "4.payment_service", "produces": [], "post_hooks": []}
    specs = expand_template("backend_service", "fastapi+nextjs", parent, {})
    assert specs is not None
    for spec in specs:
        role = spec.step_id.split(".")[-1]
        assert spec.step_id == f"4.payment_service.{role}"
        assert spec.inherited_from == "4.payment_service"


def test_step_id_with_nested_parent_id():
    """Parent IDs that themselves contain dots must still produce parent.role."""
    parent = {"step_id": "phase3.feature.step2", "produces": [], "post_hooks": []}
    specs = expand_template("frontend_component", "fastapi+nextjs", parent, {})
    assert specs is not None
    for spec in specs:
        assert spec.step_id.startswith("phase3.feature.step2.")


# ---------------------------------------------------------------------------
# inherited_post_hooks — copy, not reference
# ---------------------------------------------------------------------------

def test_inherited_post_hooks_are_independent_copy():
    parent_hooks = ["migration_apply", "type_sync"]
    parent = {
        "step_id": "2.user_service",
        "produces": [],
        "post_hooks": parent_hooks,
    }
    specs = expand_template("backend_service", "fastapi+nextjs", parent, {})
    assert specs is not None

    # Mutating the parent dict's post_hooks after the fact must NOT affect specs.
    parent["post_hooks"].append("should_not_propagate")

    for spec in specs:
        assert "should_not_propagate" not in spec.inherited_post_hooks
        assert spec.inherited_post_hooks == ["migration_apply", "type_sync"]


def test_inherited_post_hooks_mutating_one_spec_does_not_affect_others():
    parent = {
        "step_id": "2.user_service",
        "produces": [],
        "post_hooks": ["hook_a"],
    }
    specs = expand_template("backend_service", "fastapi+nextjs", parent, {})
    assert specs is not None

    # Mutating one spec's hooks must not contaminate siblings.
    specs[0].inherited_post_hooks.append("extra")
    for spec in specs[1:]:
        assert "extra" not in spec.inherited_post_hooks


def test_inherited_post_hooks_empty_when_parent_has_none():
    parent = {"step_id": "1.foo", "produces": []}
    specs = expand_template("frontend_component", "fastapi+nextjs", parent, {})
    assert specs is not None
    for spec in specs:
        assert spec.inherited_post_hooks == []


# ---------------------------------------------------------------------------
# FILE_ROLE_TO_PATH sanity
# ---------------------------------------------------------------------------

def test_file_role_to_path_backend_service_has_all_roles():
    key = ("backend_service", "fastapi+nextjs")
    assert key in FILE_ROLE_TO_PATH
    role_paths = FILE_ROLE_TO_PATH[key]
    for role in MULTI_FILE_RULES[key]:
        assert role in role_paths, f"Missing path template for role '{role}'"


def test_file_role_to_path_frontend_component_has_all_roles():
    key = ("frontend_component", "fastapi+nextjs")
    assert key in FILE_ROLE_TO_PATH
    role_paths = FILE_ROLE_TO_PATH[key]
    for role in MULTI_FILE_RULES[key]:
        assert role in role_paths, f"Missing path template for role '{role}'"


def test_expand_uses_file_role_to_path_for_target_file():
    parent = {"step_id": "3.svc", "produces": [], "post_hooks": []}
    specs = expand_template("backend_service", "fastapi+nextjs", parent, {})
    assert specs is not None
    role_paths = FILE_ROLE_TO_PATH[("backend_service", "fastapi+nextjs")]
    for spec in specs:
        role = spec.step_id.split(".")[-1]
        assert spec.target_file == role_paths[role]
