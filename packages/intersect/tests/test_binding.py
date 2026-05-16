"""Unit tests for intersect.binding — static bind + cache."""
import json

import pytest

from intersect import binding


@pytest.fixture
def task_ctx():
    return {
        "task": {
            "title": "workout-tracker",
            "parent_mission": {"payload": {"project_name": "workout-tracker",
                                           "use_celery": True}},
        },
    }


def test_resolve_dotted_path_hit(task_ctx):
    val = binding._resolve_path(
        task_ctx, "task.parent_mission.payload.project_name")
    assert val == "workout-tracker"


def test_resolve_dotted_path_miss(task_ctx):
    assert binding._resolve_path(task_ctx, "task.nonexistent.field") is None


def test_static_bind_all_fields_filled(fake_artifact, task_ctx):
    art = fake_artifact(
        kind="shell_recipe", mechanizable=True,
        inputs_schema={
            "project_name": {
                "type": "string",
                "bind_from": ["task.parent_mission.payload.project_name",
                              "task.title"],
            },
            "use_celery": {
                "type": "bool",
                "bind_from": ["task.parent_mission.payload.use_celery"],
                "default": False,
            },
        },
    )
    args, complete = binding.static_bind(art, task_ctx)
    assert complete is True
    assert args == {"project_name": "workout-tracker", "use_celery": True}


def test_static_bind_uses_default_when_path_misses(fake_artifact):
    art = fake_artifact(
        kind="shell_recipe",
        inputs_schema={
            "use_celery": {"type": "bool", "bind_from": ["task.missing"],
                            "default": False},
        },
    )
    args, complete = binding.static_bind(art, {"task": {}})
    assert complete is True
    assert args == {"use_celery": False}


def test_static_bind_incomplete_when_required_field_unbound(fake_artifact):
    art = fake_artifact(
        kind="shell_recipe",
        inputs_schema={
            "project_name": {"type": "string", "bind_from": ["task.missing"]},
        },
    )
    args, complete = binding.static_bind(art, {"task": {}})
    assert complete is False
    assert args.get("project_name") is None


def test_static_bind_non_parametric_returns_empty(fake_artifact):
    art = fake_artifact(kind="prompt_skill", inputs_schema={})
    args, complete = binding.static_bind(art, {"task": {}})
    assert args == {}
    assert complete is True


@pytest.mark.asyncio
async def test_bind_cache_roundtrip(intersect_db, fake_artifact):
    art = fake_artifact(artifact_id=77, kind="shell_recipe",
                        inputs_schema={"project_name": {"type": "string"}})
    ctx = {"task": {"title": "x"}}
    miss = await binding.lookup_bind_cache(art, ctx)
    assert miss is None
    await binding.write_bind_cache(art, ctx, {"project_name": "x"})
    hit = await binding.lookup_bind_cache(art, ctx)
    assert hit == {"project_name": "x"}
