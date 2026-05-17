"""cookiecutter_template adapter — cookiecutter.json -> manifest."""
import json
from pathlib import Path

import pytest

from yalayut.discovery.sources.cookiecutter_template import (
    cookiecutter_json_to_manifest,
    CookiecutterAdapter,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cookiecutter.json"


def test_manifest_basic_shape():
    cc_json = json.loads(FIXTURE.read_text())
    m = cookiecutter_json_to_manifest(
        cc_json, owner="cookiecutter", repo="cookiecutter-django"
    )
    assert m["artifact_type"] == "skill"
    assert m["kind"] == "shell_recipe"
    assert m["mechanizable"] is True
    assert m["name"] == "cc-django"          # cookiecutter- prefix stripped
    assert m["name_original"] == "cookiecutter-django"
    assert m["owner"] == "cookiecutter"
    assert m["source"] == "github:cookiecutter/cookiecutter-django"


def test_invocation_step_built():
    cc_json = json.loads(FIXTURE.read_text())
    m = cookiecutter_json_to_manifest(cc_json, owner="foo", repo="cookiecutter-bar")
    steps = m["invocation"]["steps"]
    assert len(steps) == 1
    assert steps[0]["cmd"] == "uvx cookiecutter --no-input gh:foo/cookiecutter-bar"


def test_inputs_schema_lifted():
    cc_json = json.loads(FIXTURE.read_text())
    m = cookiecutter_json_to_manifest(cc_json, owner="foo", repo="cookiecutter-bar")
    schema = m["inputs_schema"]
    # Private keys (underscore-prefixed) are dropped.
    assert "_copy_without_render" not in schema
    # Plain string var.
    assert schema["project_name"]["type"] == "string"
    assert schema["project_name"]["default"] == "My Awesome Project"
    # Boolean-ish y/n var inferred as bool.
    assert schema["use_celery"]["type"] == "bool"
    assert schema["use_celery"]["default"] is False
    # List var -> choice.
    assert schema["open_source_license"]["type"] == "choice"
    assert schema["open_source_license"]["choices"][0] == "MIT"
    assert schema["open_source_license"]["default"] == "MIT"
    # Jinja-templated var is skipped (derived, not an input).
    assert "project_slug" not in schema


def test_name_no_double_prefix():
    # repo already 'cookiecutter-' prefixed -> single 'cc-' canonical.
    m = cookiecutter_json_to_manifest({}, owner="x", repo="cookiecutter-data-science")
    assert m["name"] == "cc-data-science"
    # repo without the prefix still gets a cc- canonical.
    m2 = cookiecutter_json_to_manifest({}, owner="x", repo="flask-template")
    assert m2["name"] == "cc-flask-template"


@pytest.mark.asyncio
async def test_adapter_discover_parses_fixture(monkeypatch):
    adapter = CookiecutterAdapter()

    async def fake_fetch_json(owner, repo):
        return json.loads(FIXTURE.read_text())

    monkeypatch.setattr(adapter, "_fetch_cookiecutter_json", fake_fetch_json)
    refs = await adapter.discover(
        {"source_id": "github:cookiecutter/cookiecutter-django",
         "owner": "cookiecutter", "repo": "cookiecutter-django"}
    )
    assert len(refs) == 1
    assert refs[0]["manifest"]["name"] == "cc-django"
