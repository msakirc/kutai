"""Coverage for hooks.resolve_dynamic_constraints (min_items_from)."""
import json
import pytest
from unittest.mock import AsyncMock, patch

from src.workflows.engine.hooks import resolve_dynamic_constraints


@pytest.mark.asyncio
async def test_no_min_items_from_is_noop():
    schema = {
        "implementation_backlog": {
            "type": "array",
            "min_items": 5,
            "items": {"type": "object"},
        }
    }
    out = await resolve_dynamic_constraints(schema, mission_id=1)
    assert out["implementation_backlog"]["min_items"] == 5
    assert "min_items_from" not in out["implementation_backlog"]


@pytest.mark.asyncio
async def test_min_items_from_resolves_against_upstream_list():
    schema = {
        "implementation_backlog": {
            "type": "array",
            "min_items": 1,
            "min_items_from": {"artifact": "mvp_scope", "path": "mvp_feature_list", "floor": 5},
            "items": {"type": "object"},
        }
    }
    upstream_payload = json.dumps({
        "mvp_feature_list": [f"feat_{i}" for i in range(16)],
        "excluded_features_rationale": {},
    })
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=upstream_payload)
    with patch("src.workflows.engine.hooks.get_artifact_store", return_value=fake_store):
        out = await resolve_dynamic_constraints(schema, mission_id=42)
    rule = out["implementation_backlog"]
    # Floor is 5 but upstream has 16 → 16 wins.
    assert rule["min_items"] == 16
    assert "min_items_from" not in rule
    fake_store.retrieve.assert_awaited_once_with(42, "mvp_scope")


@pytest.mark.asyncio
async def test_floor_takes_over_when_upstream_missing():
    schema = {
        "x": {
            "type": "array",
            "min_items_from": {"artifact": "ghost", "path": "items", "floor": 7},
            "items": {"type": "string"},
        }
    }
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)
    with patch("src.workflows.engine.hooks.get_artifact_store", return_value=fake_store):
        out = await resolve_dynamic_constraints(schema, mission_id=1)
    assert out["x"]["min_items"] == 7


@pytest.mark.asyncio
async def test_floor_takes_over_when_upstream_unparseable():
    schema = {
        "x": {
            "type": "array",
            "min_items_from": {"artifact": "garbled", "floor": 3},
            "items": {"type": "string"},
        }
    }
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value="not valid json {")
    with patch("src.workflows.engine.hooks.get_artifact_store", return_value=fake_store):
        out = await resolve_dynamic_constraints(schema, mission_id=1)
    assert out["x"]["min_items"] == 3


@pytest.mark.asyncio
async def test_path_drilldown_for_nested_lists():
    schema = {
        "x": {
            "type": "array",
            "min_items_from": {"artifact": "outer", "path": "a.b.c", "floor": 1},
            "items": {"type": "string"},
        }
    }
    upstream = json.dumps({"a": {"b": {"c": [1, 2, 3, 4]}}})
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=upstream)
    with patch("src.workflows.engine.hooks.get_artifact_store", return_value=fake_store):
        out = await resolve_dynamic_constraints(schema, mission_id=1)
    assert out["x"]["min_items"] == 4


@pytest.mark.asyncio
async def test_keeps_explicit_min_items_when_higher_than_upstream():
    schema = {
        "x": {
            "type": "array",
            "min_items": 20,
            "min_items_from": {"artifact": "src", "path": "list", "floor": 1},
            "items": {"type": "string"},
        }
    }
    upstream = json.dumps({"list": [1, 2, 3]})
    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=upstream)
    with patch("src.workflows.engine.hooks.get_artifact_store", return_value=fake_store):
        out = await resolve_dynamic_constraints(schema, mission_id=1)
    # Author wrote min_items=20 explicitly → take the stricter of (3, 20).
    assert out["x"]["min_items"] == 20


@pytest.mark.asyncio
async def test_no_mission_id_falls_back_to_floor():
    schema = {
        "x": {
            "type": "array",
            "min_items_from": {"artifact": "anything", "floor": 9},
            "items": {"type": "string"},
        }
    }
    out = await resolve_dynamic_constraints(schema, mission_id=None)
    assert out["x"]["min_items"] == 9


@pytest.mark.asyncio
async def test_does_not_mutate_input_schema():
    schema = {
        "x": {
            "type": "array",
            "min_items_from": {"artifact": "src", "floor": 5},
            "items": {"type": "string"},
        }
    }
    out = await resolve_dynamic_constraints(schema, mission_id=None)
    assert "min_items_from" in schema["x"]  # original untouched
    assert "min_items_from" not in out["x"]
