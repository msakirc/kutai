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


# ---------------------------------------------------------------------------
# P2-4 regression: Cause 1 — seed manifests use task.payload.* convention
# ---------------------------------------------------------------------------

def test_static_bind_seed_convention_task_payload(fake_artifact):
    """bind_from: [task.payload.project_name, task.title] must resolve when
    task_ctx exposes task.payload (the real seed convention used by
    cc-pypackage, cc-django, cc-data-science).

    This FAILS before the Cause-1 fix because _build_task_ctx exposes the
    payload at task.parent_mission.payload.* only — task.payload is absent.
    """
    art = fake_artifact(
        kind="shell_recipe", mechanizable=True,
        inputs_schema={
            "project_name": {
                "type": "string",
                "bind_from": ["task.payload.project_name", "task.title"],
            },
            "author_name": {
                "type": "string",
                "bind_from": ["task.payload.author_name"],
                "default": "unknown",
            },
        },
    )
    # task_ctx as _build_task_ctx will produce after the fix:
    # task.payload is the mission/task payload dict.
    task_ctx = {
        "task": {
            "id": 1,
            "title": "wt",
            "description": "",
            "mission_id": 10,
            "payload": {"project_name": "wt", "author_name": "alice"},
            "parent_mission": {"payload": {"project_name": "wt", "author_name": "alice"}},
            "context": {},
        }
    }
    args, complete = binding.static_bind(art, task_ctx)
    assert complete is True, f"binding incomplete; bound={args}"
    assert args["project_name"] == "wt"
    assert args["author_name"] == "alice"


@pytest.mark.asyncio
async def test_flash_produces_prebind_envelope_with_seed_convention(
    intersect_db, fake_artifact, monkeypatch,
):
    """End-to-end: flash must produce render='prebind' + bound_args when an
    inject-class parametric artifact uses the real seed bind_from convention
    (task.payload.*) and the task context carries payload: {project_name: 'wt'}.

    This FAILS if _build_task_ctx does not expose task.payload — binding
    degrades to incomplete → render='prose' (Cause-1 regression guard).

    The fixture is deliberately NOT mechanizable: a mechanizable T0 recipe
    with a complete bind classifies as *preempt* and routes to the mechanical
    lane (empty skills envelope) — that path is covered by
    ``test_flash_preempts_bound_seed_recipe`` below. Here we exercise the
    inject → prebind render branch.
    """
    from intersect.flash import flash as do_flash

    art = fake_artifact(
        artifact_id=42, kind="prompt_skill", mechanizable=False,
        vet_tier=0, score=1.0, name="cc-pypackage",
        inputs_schema={
            "project_name": {
                "type": "string",
                "bind_from": ["task.payload.project_name", "task.title"],
            },
        },
    )

    async def _query(_task):
        return [art]

    import yalayut
    monkeypatch.setattr(yalayut, "query", _query, raising=False)

    task = {
        "id": 99,
        "title": "[3.2] Scaffold the Python package",
        "description": "Create the package",
        "agent_type": "coder",
        "mission_id": 57,
        "context": json.dumps({
            "is_workflow_step": True,
            "recipe_lookup": True,
            "payload": {"project_name": "wt"},
        }),
    }

    out = await do_flash(task)
    skills = out.get("skills", [])
    assert skills, "expected at least one skills entry"
    entry = skills[0]
    assert entry["render"] == "prebind", (
        f"expected render='prebind' but got render={entry['render']!r}; "
        "task.payload.* bind_from not resolving — Cause 1 unfixed"
    )
    payload = entry.get("payload", {})
    bound = payload.get("bound_args") or {}
    assert bound.get("project_name") == "wt", (
        f"bound_args missing project_name='wt'; got {bound}"
    )


@pytest.mark.asyncio
async def test_flash_preempts_bound_seed_recipe(
    intersect_db, fake_artifact, monkeypatch,
):
    """End-to-end: a mechanizable T0 shell_recipe whose seed bind_from
    (task.payload.*) resolves completely classifies as *preempt* and routes
    the task to the mechanical lane with bound args — no skills envelope.

    This is the Phase-3 reality (``PHASE2_PREEMPT_ENABLED = True``, commit
    1f0c3094). It also proves the seed convention bound correctly, since
    preempt requires a complete static bind.
    """
    from intersect.flash import flash as do_flash

    art = fake_artifact(
        artifact_id=42, kind="shell_recipe", mechanizable=True,
        vet_tier=0, score=1.0, name="cc-pypackage",
        inputs_schema={
            "project_name": {
                "type": "string",
                "bind_from": ["task.payload.project_name", "task.title"],
            },
        },
    )

    async def _query(_task):
        return [art]

    import yalayut
    monkeypatch.setattr(yalayut, "query", _query, raising=False)

    task = {
        "id": 99,
        "title": "[3.2] Scaffold the Python package",
        "description": "Create the package",
        "agent_type": "coder",
        "mission_id": 57,
        "context": json.dumps({
            "is_workflow_step": True,
            "recipe_lookup": True,
            "payload": {"project_name": "wt"},
        }),
    }

    out = await do_flash(task)
    # Preempt owns the whole task: routed to the mechanical lane, no envelope.
    assert out.get("skills") == []
    assert out.get("runner") == "mechanical"
    payload = out.get("payload") or {}
    assert payload.get("action") == "yalayut_recipe"
    assert payload.get("recipe_id") == 42
    assert (payload.get("args") or {}).get("project_name") == "wt", (
        f"preempt args missing project_name='wt'; got {payload.get('args')}"
    )
