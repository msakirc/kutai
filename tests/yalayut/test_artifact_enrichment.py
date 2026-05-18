"""Task 0 — Artifact enrichment: source/owner/env_status/intent_keywords/inputs_schema."""
import pytest

from yalayut.contracts import Manifest, TaskContext, Artifact
from yalayut.index import store
from yalayut._query_engine import query_db

pytestmark = pytest.mark.asyncio


def _manifest_with_keywords():
    return Manifest(
        name="cc-pypackage",
        name_original="cc-pypackage",
        version="1.0.0",
        artifact_type="skill",
        kind="shell_recipe",
        source="github:audreyfeldroy/cookiecutter-pypackage@/",
        owner="audreyfeldroy",
        mechanizable=True,
        intent_keywords=["python", "package", "scaffold", "pyproject"],
        inputs_schema={"project_name": {"type": "string", "bind_from": ["task.title"]}},
    )


async def test_artifact_carries_source(yalayut_db):
    m = _manifest_with_keywords()
    await store(yalayut_db, m, "body", 0, {}, [1.0] + [0.0] * 767)
    results = await query_db(
        yalayut_db, TaskContext(title="python package"), [1.0] + [0.0] * 767
    )
    assert results, "expected at least one artifact"
    art = results[0]
    assert isinstance(art, Artifact)
    assert art.source == "github:audreyfeldroy/cookiecutter-pypackage@/"


async def test_artifact_carries_owner(yalayut_db):
    m = _manifest_with_keywords()
    await store(yalayut_db, m, "body", 0, {}, [1.0] + [0.0] * 767)
    results = await query_db(
        yalayut_db, TaskContext(title="python package"), [1.0] + [0.0] * 767
    )
    art = results[0]
    assert art.owner == "audreyfeldroy"


async def test_artifact_carries_env_status(yalayut_db):
    m = _manifest_with_keywords()
    await store(yalayut_db, m, "body", 0, {}, [1.0] + [0.0] * 767)
    results = await query_db(
        yalayut_db, TaskContext(title="python package"), [1.0] + [0.0] * 767
    )
    art = results[0]
    assert art.env_status == "ready"


async def test_artifact_carries_intent_keywords_from_manifest(tmp_path, yalayut_db):
    """intent_keywords loaded from manifest_path on disk."""
    import yaml
    m = _manifest_with_keywords()
    # Write a real manifest yaml to disk so _to_artifact can parse it
    yaml_path = tmp_path / "cc-pypackage.yaml"
    yaml_path.write_text(
        yaml.dump({
            "name": m.name,
            "name_original": m.name_original,
            "version": m.version,
            "artifact_type": m.artifact_type,
            "kind": m.kind,
            "source": m.source,
            "owner": m.owner,
            "mechanizable": m.mechanizable,
            "intent_keywords": ["python", "package", "scaffold", "pyproject"],
            "inputs_schema": {"project_name": {"type": "string"}},
        })
    )
    await store(yalayut_db, m, "body", 0, {}, [1.0] + [0.0] * 767,
                manifest_path=str(yaml_path))
    results = await query_db(
        yalayut_db, TaskContext(title="python package"), [1.0] + [0.0] * 767
    )
    art = results[0]
    assert isinstance(art.intent_keywords, list)
    assert "python" in art.intent_keywords


async def test_artifact_carries_inputs_schema_from_manifest(tmp_path, yalayut_db):
    """inputs_schema loaded from manifest_path on disk."""
    import yaml
    m = _manifest_with_keywords()
    yaml_path = tmp_path / "cc-pypackage2.yaml"
    yaml_path.write_text(
        yaml.dump({
            "name": m.name,
            "name_original": m.name_original,
            "version": m.version,
            "artifact_type": m.artifact_type,
            "kind": m.kind,
            "source": m.source,
            "owner": m.owner,
            "mechanizable": m.mechanizable,
            "intent_keywords": ["python"],
            "inputs_schema": {"project_name": {"type": "string", "bind_from": ["task.title"]}},
        })
    )
    await store(yalayut_db, m, "body", 0, {}, [1.0] + [0.0] * 767,
                manifest_path=str(yaml_path))
    results = await query_db(
        yalayut_db, TaskContext(title="python package"), [1.0] + [0.0] * 767
    )
    art = results[0]
    assert isinstance(art.inputs_schema, dict)
    assert "project_name" in art.inputs_schema


async def test_artifact_defaults_empty_when_no_manifest(yalayut_db):
    """No manifest_path → intent_keywords=[], inputs_schema={}."""
    m = _manifest_with_keywords()
    await store(yalayut_db, m, "body", 0, {}, [1.0] + [0.0] * 767,
                manifest_path=None)
    results = await query_db(
        yalayut_db, TaskContext(title="python package"), [1.0] + [0.0] * 767
    )
    art = results[0]
    assert art.intent_keywords == []
    assert art.inputs_schema == {}
